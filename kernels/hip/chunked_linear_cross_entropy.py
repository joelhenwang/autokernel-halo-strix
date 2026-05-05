"""
Chunked Linear + Cross Entropy Loss (v2, uses Phase 1 HIP CE kernel).

Avoids materializing the full (B*T, vocab) logits tensor by processing rows in
chunks. For each chunk:
  1. matmul chunk_logits = chunk_h @ W.T  (rocBLAS via F.linear, fp16)
  2. optional logit_softcap folded into CE kernel
  3. CE forward+backward computed via Phase 1 HIP kernel (in fused mode, with
     scale pre-baked as 1/N_valid so grad_logits_chunk is final).
  4. grad_h_chunk = grad_logits_chunk @ W   (rocBLAS, fp16)
  5. grad_W += grad_logits_chunk.T @ chunk_h   (accumulate in fp32)

Savings vs naive linear+CE:
  - Never materializes [N, V] logits or grad_logits (saves 2 * N * V * 2 bytes).
    At OdinHalo prod (N=1024, V=32768): 134 MB saved from peak.
  - Eliminates 3 reads + 2 writes of 67 MB logits tensor per step.
  - logit_softcap applied in-kernel (no separate pass over logits).

Features (all pass-through to Phase 1 kernel):
  - logit_softcap (0 = off)
  - ignore_index (default -100)
  - label_smoothing (0 = off)
  - z_loss_weight (0 = off): emits per-row LSE and adds z_weight * lse.pow(2).mean()

Tied/untied weights: transparent — caller passes `weight` tensor. If tied, the
embedding's .grad accumulates both the embed-lookup grad and our grad_W via
standard PyTorch autograd accumulation.

Usage:
    from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss
    loss_fn = ChunkedLinearCrossEntropyLoss(
        chunk_size=256, softcap=30.0, z_loss_weight=1e-4,
    )
    loss = loss_fn(h_low, lm_head_weight, targets)
    loss.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd


def _get_ce_kernel():
    """Import Phase 1 CE kernel lazily to avoid circular import and compile cost."""
    import kernel as _k
    return _k


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Per-chunk linear + CE with Phase 1 HIP kernel."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        hidden,          # [N, D] fp16
        weight,          # [V, D] fp16
        targets,         # [N] int64
        chunk_size,
        softcap,
        ignore_index,
        label_smoothing,
        z_loss_weight,
    ):
        k = _get_ce_kernel()
        N, D = hidden.shape
        V = weight.shape[0]
        device = hidden.device

        # Count valid rows globally for consistent per-row scale in fused forward.
        valid_global = (targets != ignore_index)
        n_valid_global = int(valid_global.sum().item())
        n_valid_global = max(n_valid_global, 1)
        inv_n_valid = 1.0 / n_valid_global

        hidden_contig = hidden.contiguous()
        weight_contig = weight.contiguous()
        targets_contig = targets.contiguous()

        # Outputs accumulated across chunks
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        z_sum_sq = torch.zeros((), device=device, dtype=torch.float32) if z_loss_weight > 0 else None
        grad_hidden = torch.empty_like(hidden_contig)
        grad_weight = torch.zeros_like(weight_contig, dtype=torch.float32)

        requires_grad_hidden = hidden.requires_grad
        requires_grad_weight = weight.requires_grad

        # Ensure fp16 for CE kernel
        assert hidden_contig.dtype == torch.float16, f"hidden must be fp16, got {hidden_contig.dtype}"
        assert weight_contig.dtype == torch.float16, f"weight must be fp16, got {weight_contig.dtype}"

        # Process chunks
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            h_chunk = hidden_contig[start:end]           # [chunk, D] fp16
            t_chunk = targets_contig[start:end]           # [chunk]

            # --- 1. Linear: logits_chunk = h_chunk @ W.T ---
            logits_chunk = F.linear(h_chunk, weight_contig)  # [chunk, V] fp16

            # --- 2. CE forward (fused mode, pre-scaled grad) ---
            mod = k._get_fwd_module()
            losses_chunk, lse_chunk, _, _, grad_logits_chunk = mod.cross_entropy_fwd_hip(
                logits_chunk, t_chunk,
                float(softcap), int(ignore_index), float(label_smoothing),
                1, 0,               # write_grad=1 (fused), save_max_sum=0
                float(inv_n_valid), # pre-scale grad by 1/N_valid (global)
            )

            # --- 3. Accumulate loss (weighted by chunk) ---
            # losses_chunk is per-row (for both valid and ignored rows: 0 for ignored)
            valid_chunk = (t_chunk != ignore_index).to(torch.float32)
            loss_sum = loss_sum + (losses_chunk * valid_chunk).sum()

            if z_loss_weight > 0:
                # z_loss contribution: sum of lse^2 (divided by N_valid at end)
                z_sum_sq = z_sum_sq + (lse_chunk * valid_chunk).pow(2).sum()

            # --- 4. grad_hidden chunk: grad_h = grad_logits @ W ---
            if requires_grad_hidden:
                grad_hidden[start:end] = grad_logits_chunk @ weight_contig  # fp16 GEMM

            # --- 5. grad_weight accumulation: grad_W += grad_logits.T @ h_chunk ---
            if requires_grad_weight:
                # Accumulate in fp32 for numerical stability on tied embedding
                grad_weight.addmm_(
                    grad_logits_chunk.t().to(torch.float32),
                    h_chunk.to(torch.float32),
                    beta=1.0, alpha=1.0,
                )

        # Final loss = mean over valid
        loss = loss_sum / n_valid_global

        # z_loss
        if z_loss_weight > 0:
            z_loss = z_sum_sq / n_valid_global
            loss = loss + z_loss_weight * z_loss

        # Save for backward — grad_hidden and grad_weight are already computed.
        # The actual backward is just returning pre-computed grads, scaled by grad_output.
        ctx.save_for_backward(grad_hidden, grad_weight.to(weight.dtype))
        ctx.requires_grad_hidden = requires_grad_hidden
        ctx.requires_grad_weight = requires_grad_weight

        return loss

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_hidden_saved, grad_weight_saved = ctx.saved_tensors

        grad_hidden = None
        grad_weight = None

        if ctx.requires_grad_hidden:
            # grad_output is scalar for loss.backward(); fast path if ~1.0
            if grad_output.dim() == 0 and grad_output.item() == 1.0:
                grad_hidden = grad_hidden_saved
            else:
                grad_hidden = grad_hidden_saved * grad_output.to(grad_hidden_saved.dtype)

        if ctx.requires_grad_weight:
            if grad_output.dim() == 0 and grad_output.item() == 1.0:
                grad_weight = grad_weight_saved
            else:
                grad_weight = grad_weight_saved * grad_output.to(grad_weight_saved.dtype)

        # Outputs: hidden, weight, targets, chunk_size, softcap, ignore_index,
        #          label_smoothing, z_loss_weight
        return (
            grad_hidden, grad_weight, None, None, None, None, None, None,
        )


class ChunkedLinearCrossEntropyLoss(nn.Module):
    """Drop-in replacement for `nn.Linear + nn.CrossEntropyLoss` that avoids
    materializing the full logits tensor.

    Args:
        chunk_size: rows per chunk. Lower = less peak memory, but more GEMM launches.
            Default 256 balances well at V=32768.
        softcap: logit softcap (0 = off). Applied inside CE kernel, no extra pass.
        ignore_index: target value to mask (default -100, matches PyTorch).
        label_smoothing: alpha (0 = off).
        z_loss_weight: if > 0, emit per-row logsumexp and add z*lse^2.mean() to loss.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        softcap: float = 0.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        z_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.softcap = float(softcap)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)
        self.z_loss_weight = float(z_loss_weight)

    def forward(
        self,
        hidden: torch.Tensor,     # (N, D) or (B, T, D)
        weight: torch.Tensor,     # (V, D)
        targets: torch.Tensor,    # (N,) or (B, T)
    ) -> torch.Tensor:
        # Flatten to 2D
        if hidden.dim() == 3:
            B, T, D = hidden.shape
            hidden = hidden.reshape(B * T, D)
        if targets.dim() == 2:
            targets = targets.reshape(-1)

        # Coerce to fp16 (our kernel requires it)
        if hidden.dtype != torch.float16:
            hidden = hidden.to(torch.float16)
        if weight.dtype != torch.float16:
            weight = weight.to(torch.float16)

        return _ChunkedLinearCrossEntropy.apply(
            hidden, weight, targets,
            self.chunk_size, self.softcap, self.ignore_index,
            self.label_smoothing, self.z_loss_weight,
        )
