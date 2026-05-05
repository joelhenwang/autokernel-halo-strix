"""Conv blocks: ShortConvBlock, GQABlock, MoDAGQABlock, HyPEShortConvBlock.

ShortConvBlock:   GatedConv mixer + inlined momentum + SwiGLU FFN (from argus_prime).
GQABlock:         GQA Attention + QK-Norm + inlined momentum + SwiGLU/TTT FFN (from argus_prime).
MoDAGQABlock:     GQA with MoDA depth-attention + XSA + momentum + SwiGLU (de-duplicated from tyr_halo + baldr_halo).
HyPEShortConvBlock: Momentum-free ShortConv with RoPE on conv gate signal (ODIN-HALO).
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._components import RMSNorm, SwiGLU, GatedConv, _HAS_CAUSAL_CONV1D

if _HAS_CAUSAL_CONV1D:
    from causal_conv1d import causal_conv1d_fn

from models.components.attention import Attention, CodaAttention

try:
    from models.argus import TTTSwiGLU
    from models.argus_prime import MultiStepTTTSwiGLU
except ImportError:
    TTTSwiGLU = None
    MultiStepTTTSwiGLU = None


class ShortConvBlock(nn.Module):
    """GatedConv mixer + inlined momentum + SwiGLU FFN. Compile-friendly."""

    def __init__(self, d_model: int, d_conv: int, ffn_inner: int,
                 conv_kernel: int = 3, momentum_beta: float = 0.5):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_conv, conv_kernel)
        self.out_proj = nn.Linear(d_conv, d_model, bias=False)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta / (1 - momentum_beta)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x, velocity):
        normed = self.pre_norm(x)
        conv_out = self.conv(normed)
        mixer_out = self.out_proj(conv_out)

        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + mixer_out
        velocity = velocity.clamp(-8.0, 8.0)

        x = x + velocity
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight

        x = x + self.ffn(normed)
        return x, velocity


class GQABlock(nn.Module):
    """GQA Attention + QK-Norm + inlined momentum + SwiGLU/TTT FFN."""

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4,
                 momentum_beta: float = 0.5,
                 ttt_mode: str = "none",
                 ttt_chunk: int = 512, ttt_lr_init: float = 0.01):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, qk_norm=True)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta / (1 - momentum_beta)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ttt_mode = ttt_mode

        if ttt_mode == "single" and TTTSwiGLU is not None:
            self.ffn = TTTSwiGLU(d_model, ffn_inner, ttt_chunk, ttt_lr_init)
        elif ttt_mode == "multi" and MultiStepTTTSwiGLU is not None:
            self.ffn = MultiStepTTTSwiGLU(d_model, ffn_inner, ttt_chunk, ttt_lr_init)
        else:
            self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x, velocity, freqs_cis, ttt_target=None):
        attn_out = self.attn(self.pre_norm(x), freqs_cis)

        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + attn_out
        velocity = velocity.clamp(-8.0, 8.0)

        x = x + velocity
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight

        if self.ttt_mode != "none" and ttt_target is not None:
            x = x + self.ffn(normed, ttt_target=ttt_target)
        else:
            x = x + self.ffn(normed)

        return x, velocity


class MoDAGQABlock(nn.Module):
    """GQA with MoDA depth-attention + XSA + momentum + SwiGLU FFN.

    MoDA: each attention head attends to both sequence KVs and depth KVs
    (representations from same token position in prior loop iterations).
    """

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4,
                 momentum_beta: float = 0.5, use_xsa: bool = True):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_kv_heads
        self.pre_norm = RMSNorm(d_model)
        if use_xsa:
            self.attn = CodaAttention(d_model, n_heads, n_kv_heads,
                                      qk_norm=True, exclusive=True)
        else:
            self.attn = Attention(d_model, n_heads, n_kv_heads, qk_norm=True)
        self.use_xsa = use_xsa
        self.depth_kv_proj = nn.Linear(d_model, n_kv_heads * self.head_dim * 2, bias=False)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta / (1 - momentum_beta)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, velocity: torch.Tensor,
                freqs_cis: torch.Tensor,
                depth_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.pre_norm(x)
        if self.use_xsa:
            attn_out = self.attn(normed, freqs_cis, depth_kvs=depth_kvs)
        else:
            attn_out = self.attn(normed, freqs_cis)

        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + attn_out
        velocity = velocity.clamp(-8.0, 8.0)

        x = x + velocity
        x = x + self.ffn(self.ffn_norm(x))
        return x, velocity

    def compute_depth_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        kv = self.depth_kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        return k, v


class HyPEShortConvBlock(nn.Module):
    """Momentum-free ShortConv with RoPE applied to the gate signal.

    Designed for HyPE (ODIN-HALO): RoPE provides local positional awareness
    in the sequential conv blocks, while the GQA block uses NoPE for length
    generalization.

    Architecture: RMSNorm → RoPE(gate) * up → causal_conv1d → out_proj → SwiGLU
    No momentum — direct residuals (MoDA + loop pos + skip carry cross-iter info).
    """

    def __init__(self, d_model: int, d_conv: int, ffn_inner: int,
                 conv_kernel: int = 3, head_dim: int = 64):
        super().__init__()
        self.d_conv = d_conv
        self.rope_head_dim = head_dim
        self.n_rope_heads = d_conv // head_dim
        self.pre_norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, 3 * d_conv, bias=False)
        self.out_proj = nn.Linear(d_conv, d_model, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)
        self.conv_kernel = conv_kernel

        if _HAS_CAUSAL_CONV1D:
            self.conv_weight = nn.Parameter(torch.randn(d_conv, conv_kernel))
            self.conv_bias = nn.Parameter(torch.zeros(d_conv))
        else:
            self.conv = nn.Conv1d(
                d_conv, d_conv, kernel_size=conv_kernel,
                padding=conv_kernel - 1, groups=d_conv, bias=True,
            )

    def _manual_causal_conv1d(self, y):
        """Compile-friendly causal conv1d via F.conv1d (no DaoAILab C++ ext).

        y: [B, T, d_conv] → returns [B, T, d_conv]
        Uses grouped conv so each channel has its own kernel (depthwise),
        matching causal_conv1d_fn's behavior.
        """
        import torch.nn.functional as F
        B, T, C = y.shape
        y_t = y.transpose(1, 2)  # [B, C, T]
        if _HAS_CAUSAL_CONV1D:
            # Causal conv1d uses weight shape [d_conv, kernel_size]; reshape for F.conv1d
            w = self.conv_weight.unsqueeze(1)  # [C, 1, k]
            b = self.conv_bias
            k = self.conv_kernel
        else:
            # nn.Conv1d path: weight shape [C, 1, k]
            w = self.conv.weight
            b = self.conv.bias
            k = self.conv_kernel
        # Left-padded (causal) grouped conv
        z = F.conv1d(y_t, w, bias=b, padding=k - 1, groups=C)
        return z[:, :, :T].transpose(1, 2)

    def _rope_on_gate(self, b: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = b.shape
        b_r = b.float().reshape(B, T, self.n_rope_heads, self.rope_head_dim)
        b_pairs = b_r.reshape(B, T, self.n_rope_heads, self.rope_head_dim // 2, 2)
        b_c = torch.view_as_complex(b_pairs)
        freqs = freqs_cis[:T][None, :, None, :self.rope_head_dim // 2]
        b_rotated = torch.view_as_real(b_c * freqs).flatten(2)
        return b_rotated.to(b.dtype)

    def _rope_gate_mul_native(self, b, h_tilde, freqs_cis, T):
        """Compile-friendly RoPE + gate multiply using pure real arithmetic.

        Matches HIP fused_rope_gate_mul kernel math bit-for-bit:
          - rotation computed in fp32 (upcast from fp16)
          - final multiply in fp32, cast back to input dtype
        """
        B = b.shape[0] // T
        d_conv = self.d_conv
        rope_pairs = self.rope_head_dim // 2

        # Force contiguous slices so Inductor can lower the subsequent ops.
        cos = freqs_cis.real[:T, :rope_pairs].contiguous().float()  # fp32 [T, pairs]
        sin = freqs_cis.imag[:T, :rope_pairs].contiguous().float()

        # Reshape b from [B*T, d_conv] → [B, T, H, 2*pairs]
        b_r = b.view(B, T, self.n_rope_heads, 2 * rope_pairs).float()
        h_r = h_tilde.view(B, T, self.n_rope_heads, 2 * rope_pairs).float()
        # Split into even/odd pairs
        b_even = b_r[..., 0::2]  # [B, T, H, pairs]
        b_odd  = b_r[..., 1::2]
        h_even = h_r[..., 0::2]
        h_odd  = h_r[..., 1::2]
        # Broadcast cos/sin: [T, pairs] → [1, T, 1, pairs]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        # RoPE rotation (fp32)
        b_rot_even = b_even * cos - b_odd * sin
        b_rot_odd  = b_even * sin + b_odd * cos
        # Gate multiply before interleave (saves a pass)
        y_even = b_rot_even * h_even
        y_odd  = b_rot_odd  * h_odd
        # Interleave back to [B, T, H, 2*pairs]
        y = torch.stack([y_even, y_odd], dim=-1).flatten(-2)
        return y.view(B * T, d_conv).to(b.dtype)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        normed = self.pre_norm(x)
        b, c, h_tilde = self.proj(normed).chunk(3, dim=-1)

        # Compile-friendly path (pure PyTorch ops, Inductor-fusable) when flag set.
        if getattr(self, "_compile_friendly", False):
            y = self._rope_gate_mul_native(
                b.reshape(B * T, self.d_conv),
                h_tilde.reshape(B * T, self.d_conv),
                freqs_cis, T,
            ).view(B, T, self.d_conv)
        else:
            # Fused RoPE + gate multiply (saves intermediate b_rope tensor).
            # IMPORTANT: .real / .imag on a complex tensor returns a NON-contiguous view
            # (complex memory interleaves real+imag). The HIP kernel reads them as
            # contiguous row-major float arrays, so we MUST call .contiguous() here.
            # Without this, the kernel silently reads interleaved real/imag as cos,
            # producing garbled RoPE rotation.
            from kernels.hip.fused_rope_gate_mul import kernel_fn as fused_rope_mul
            freqs_cos = freqs_cis.real[:T, :self.rope_head_dim // 2].contiguous().float()
            freqs_sin = freqs_cis.imag[:T, :self.rope_head_dim // 2].contiguous().float()
            y = fused_rope_mul(b.reshape(B*T, self.d_conv).half(),
                               h_tilde.reshape(B*T, self.d_conv).half(),
                               freqs_cos, freqs_sin,
                               T, self.d_conv, self.rope_head_dim // 2).float().view(B, T, self.d_conv)

        if getattr(self, "_compile_friendly", False):
            # Pure PyTorch conv path — no DaoAILab extension boundary
            z = self._manual_causal_conv1d(y)
        elif _HAS_CAUSAL_CONV1D:
            z = causal_conv1d_fn(
                y.transpose(1, 2), self.conv_weight, self.conv_bias
            ).transpose(1, 2)
        else:
            z = self.conv(y.transpose(1, 2))[:, :, :T].transpose(1, 2)
        conv_out = self.out_proj(c * z)
        ffn_out = self.ffn(self.ffn_norm(x + conv_out))
        return x + conv_out + ffn_out