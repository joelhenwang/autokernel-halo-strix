"""
Chunked Linear + Cross Entropy Loss.

Avoids materializing the full (B*T, vocab) logits tensor by computing
the LM head matmul and cross-entropy loss in chunks. On a 250M model
with vocab=50257, batch=16, seq=256:
  - Standard: logits = (4096, 50257) = 393 MB fp16
  - Chunked (chunk=1024): max logits = (1024, 50257) = 98 MB fp16
  - Saves ~300 MB per step + one DRAM round-trip

Approach D optimization: grad_logits (softmax - one_hot) is computed
during forward and saved in fp16 per chunk. Backward does NOT recompute
logits — just reads saved grad_logits and does 2 GEMMs instead of 3.

Usage:
    from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss

    loss_fn = ChunkedLinearCrossEntropyLoss(chunk_size=1024)
    # hidden: (B*T, D) — last hidden state BEFORE lm_head
    # weight: (V, D) — lm_head weight (tied with embedding)
    # targets: (B*T,) — target token IDs
    loss = loss_fn(hidden, weight, targets)
    loss.backward()  # gradients flow to hidden and weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Custom autograd for chunked linear + cross-entropy (Approach D).

    Forward: for each chunk, compute logits via rocBLAS, softmax, loss,
    AND grad_logits (softmax - one_hot). Save grad_logits in fp16 per chunk.

    Backward: load saved grad_logits per chunk, do 2 GEMMs (grad_hidden,
    grad_weight). No logits recomputation needed — 33% faster backward.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, hidden, weight, targets, chunk_size, ignore_index):
        N, D = hidden.shape
        V = weight.shape[0]

        total_loss = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)
        n_valid = 0
        saved_grad_logits = []  # per-chunk grad_logits in fp16

        scale_factor = 1.0  # will be set after we know n_valid

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_h = hidden[start:end]           # (chunk, D)
            chunk_t = targets[start:end]           # (chunk,)

            # GEMM 1: matmul via rocBLAS
            chunk_logits = F.linear(chunk_h.float(), weight.float())  # (chunk, V)

            # Softmax (computed once, used for both loss and grad_logits)
            chunk_probs = F.softmax(chunk_logits, dim=-1)  # (chunk, V) fp32

            # Loss: NLL from pre-computed softmax
            valid_mask = chunk_t != ignore_index
            if valid_mask.any():
                valid_probs = chunk_probs[valid_mask]
                valid_targets = chunk_t[valid_mask]
                # NLL = -log(prob[target])
                target_probs = valid_probs[torch.arange(valid_probs.shape[0], device=valid_probs.device), valid_targets]
                chunk_loss = -torch.log(target_probs.clamp(min=1e-8)).sum()
                total_loss += chunk_loss
                n_valid += valid_mask.sum().item()

            # Approach D: compute grad_logits = softmax - one_hot NOW
            # This is the CE gradient w.r.t. logits — always the same formula
            grad_logits_chunk = chunk_probs  # reuse the tensor (will modify in-place)
            if valid_mask.any():
                grad_logits_chunk[valid_mask, chunk_t[valid_mask]] -= 1.0
            # Zero out invalid positions
            if not valid_mask.all():
                grad_logits_chunk[~valid_mask] = 0.0

            # Save in fp16 to minimize memory (98 MB per chunk at V=50257)
            saved_grad_logits.append(grad_logits_chunk.half())

        if n_valid > 0:
            total_loss = total_loss / n_valid

        # Save for backward
        ctx.save_for_backward(hidden, weight)
        ctx.saved_grad_logits = saved_grad_logits  # list of (chunk, V) fp16
        ctx.chunk_size = chunk_size
        ctx.n_valid = n_valid

        return total_loss

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        hidden, weight = ctx.saved_tensors
        saved_grad_logits = ctx.saved_grad_logits
        chunk_size = ctx.chunk_size
        n_valid = ctx.n_valid

        N, D = hidden.shape

        grad_hidden = torch.zeros_like(hidden, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)

        scale = grad_output / max(n_valid, 1)

        for chunk_idx, start in enumerate(range(0, N, chunk_size)):
            end = min(start + chunk_size, N)
            chunk_h = hidden[start:end].float()  # (chunk, D)

            # Approach D: grad_logits already computed during forward!
            grad_logits = saved_grad_logits[chunk_idx].float() * scale  # (chunk, V)

            # Only 2 GEMMs (no logits recomputation):
            # GEMM 2: grad_hidden = grad_logits @ weight
            grad_hidden[start:end] = grad_logits @ weight.float()
            # GEMM 3: grad_weight += grad_logits.T @ hidden
            grad_weight += grad_logits.t() @ chunk_h

        # Free saved grad_logits
        ctx.saved_grad_logits = None

        return (
            grad_hidden.to(hidden.dtype),
            grad_weight.to(weight.dtype),
            None, None, None,  # targets, chunk_size, ignore_index
        )


class ChunkedLinearCrossEntropyLoss(nn.Module):
    """Drop-in replacement for nn.Linear + nn.CrossEntropyLoss.

    Instead of:
        logits = lm_head(hidden)              # (N, V) — huge tensor!
        loss = F.cross_entropy(logits, targets)

    Use:
        loss_fn = ChunkedLinearCrossEntropyLoss()
        loss = loss_fn(hidden, lm_head.weight, targets)

    Saves ~300 MB on a 250M model with vocab=50257, batch=16, seq=256.
    """

    def __init__(self, chunk_size: int = 1024, ignore_index: int = -100):
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden: (B*T, D) or (B, T, D) — last hidden state BEFORE lm_head
            weight: (V, D) — lm_head weight
            targets: (B*T,) or (B, T) — target token IDs

        Returns:
            Scalar loss
        """
        # Flatten to 2D
        if hidden.dim() == 3:
            B, T, D = hidden.shape
            hidden = hidden.reshape(B * T, D)
        if targets.dim() == 2:
            targets = targets.reshape(-1)

        hidden = hidden.contiguous()
        targets = targets.contiguous()

        return _ChunkedLinearCrossEntropy.apply(
            hidden, weight, targets, self.chunk_size, self.ignore_index,
        )
