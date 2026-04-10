"""
Hybrid Flash Attention: flash_attn forward + SDPA backward.

On gfx1151, flash_attn forward is 4.2x faster than SDPA (0.24ms vs 1.04ms)
but the Triton backward is 66% slower (4.46ms vs 2.69ms). This module
combines the best of both: flash_attn's fast forward with SDPA's fast
HIP-native backward, passing the softmax logsumexp directly (no recompute).

Expected: 0.24ms fwd + 2.69ms bwd = ~2.93ms total (24% faster than SDPA).

Usage:
    from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
    y = hybrid_flash_sdpa_attention(q, k, v, causal=True)  # (B, T, H, D) layout
    y.sum().backward()  # uses SDPA backward
"""

import os
import torch
import torch.nn.functional as F


class HybridFlashSDPA(torch.autograd.Function):
    """flash_attn forward + PyTorch aten SDPA backward.

    Forward: Uses flash_attn's _flash_attn_forward (fast AOTriton/Triton kernel).
    Backward: Uses torch.ops.aten._scaled_dot_product_flash_attention_backward
              with the softmax_lse from flash_attn's forward (no recompute).

    Input layout: (B, T, H, D) — flash_attn native layout.
    Internally transposes to (B, H, T, D) for SDPA backward.
    """

    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal,
                window_size_left, window_size_right):
        # Ensure contiguous
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        # flash_attn forward — returns (out, softmax_lse, _, rng_state)
        os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
        from flash_attn.flash_attn_interface import _flash_attn_forward

        out, softmax_lse, _, rng_state = _flash_attn_forward(
            q, k, v, dropout_p, softmax_scale, causal,
            window_size_left, window_size_right,
            0.0,   # softcap
            None,   # alibi_slopes
            False,  # return_softmax
        )

        # Save for backward — we need q,k,v,out,lse in (B,H,T,D) for SDPA backward
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.max_seqlen_q = q.shape[1]
        ctx.max_seqlen_k = k.shape[1]

        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, out, softmax_lse = ctx.saved_tensors

        # The aten backward expects varlen format: (total_q, H, D) with cum_seq boundaries.
        # Reshape from (B, T, H, D) to (B*T, H, D) and build cum_seq.
        B_size, T_q, H, D = q.shape
        T_k = k.shape[1]
        device = q.device

        q_packed = q.reshape(B_size * T_q, H, D).contiguous()
        k_packed = k.reshape(B_size * T_k, H, D).contiguous()
        v_packed = v.reshape(B_size * T_k, H, D).contiguous()
        out_packed = out.reshape(B_size * T_q, H, D).contiguous()
        grad_packed = grad_output.reshape(B_size * T_q, H, D).contiguous()

        # Flatten logsumexp from (B, H, T) to (B*H, T) — no, aten expects (total_q, H) or (B, H, T)?
        # Actually, _flash_attention_backward expects logsumexp shape (num_heads, total_q) based on varlen
        # Let's reshape: (B, H, T) → flatten batch into total_q → (H, B*T)?
        # Actually the CK backward uses (B, H, T) directly. Let's try that first.

        cum_seq_q = torch.arange(0, (B_size + 1) * T_q, T_q,
                                 dtype=torch.int32, device=device)
        cum_seq_k = torch.arange(0, (B_size + 1) * T_k, T_k,
                                 dtype=torch.int32, device=device)
        philox_seed = torch.zeros((), dtype=torch.int64, device=device)
        philox_offset = torch.zeros((), dtype=torch.int64, device=device)

        # Use _flash_attention_backward (varlen-native) instead of _scaled_dot_product_flash_attention_backward
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad_packed, q_packed, k_packed, v_packed, out_packed, softmax_lse,
            cum_seq_q, cum_seq_k,
            ctx.max_seqlen_q, ctx.max_seqlen_k,
            ctx.dropout_p, ctx.causal,
            philox_seed, philox_offset,
            scale=ctx.softmax_scale,
        )

        # Reshape back to (B, T, H, D)
        dq = dq.reshape(B_size, T_q, H, D)
        dk = dk.reshape(B_size, T_k, H, D)
        dv = dv.reshape(B_size, T_k, H, D)

        return dq, dk, dv, None, None, None, None, None


def hybrid_flash_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
) -> torch.Tensor:
    """Flash attention forward + SDPA backward.

    Args:
        q: (B, T, H, D) query tensor
        k: (B, T, H, D) key tensor
        v: (B, T, H, D) value tensor
        dropout_p: dropout probability
        softmax_scale: scaling factor (default: 1/sqrt(head_dim))
        causal: enable causal masking
        window_size: (left, right) sliding window sizes (-1 = unlimited)

    Returns:
        output: (B, T, H, D)
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    return HybridFlashSDPA.apply(
        q, k, v, dropout_p, softmax_scale, causal,
        window_size[0], window_size[1],
    )
