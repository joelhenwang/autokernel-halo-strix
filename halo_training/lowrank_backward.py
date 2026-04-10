"""INSTANT-inspired low-rank backward projection for weight gradients.

For weight gradient computation dW = X^T @ dY, projects both X and dY
to a lower-rank subspace before computing the GEMM:
    P = random_projection(d_model, rank)
    X_proj = X @ P
    dY_proj = dY @ P
    dW_approx = X_proj^T @ dY_proj

This reduces the backward GEMM from O(N*d*d) to O(N*d*r + N*d*r + r*r)
where r << d. The random projection matrix is fixed (not learned).

WARNING: This is an approximate technique. Gradient fidelity depends on
rank selection. Verify convergence before using in production.

Usage:
    from halo_training.lowrank_backward import apply_lowrank_backward
    apply_lowrank_backward(model, rank=256, target_modules=["feed_forward"])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankLinear(torch.autograd.Function):
    """Linear layer with low-rank projected weight gradient."""

    @staticmethod
    def forward(ctx, input, weight, bias, projection):
        ctx.save_for_backward(input, weight, bias, projection)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, projection = ctx.saved_tensors

        # grad_input: exact (no approximation)
        grad_input = grad_output @ weight

        # grad_weight: low-rank approximation
        # Instead of dW = dY^T @ X (d_out, d_in),
        # project: dY_proj = dY @ P, X_proj = X @ P
        # dW_approx comes from projected outer product
        input_2d = input.reshape(-1, input.shape[-1])  # (N, d_in)
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])  # (N, d_out)

        # Project input to rank-r
        input_proj = input_2d @ projection  # (N, r)
        # Low-rank weight gradient: dW = dY^T @ X ≈ dY^T @ (X_proj @ P^T)
        # But for speed: compute (dY^T @ X_proj) @ P^T
        grad_weight_proj = grad_2d.t() @ input_proj  # (d_out, r)
        grad_weight = grad_weight_proj @ projection.t()  # (d_out, d_in)

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

        return grad_input, grad_weight, grad_bias, None


class LowRankLinearWrapper(nn.Module):
    """Wraps an existing nn.Linear with low-rank backward projection."""

    def __init__(self, linear: nn.Linear, rank: int = 256):
        super().__init__()
        self.linear = linear
        d_in = linear.in_features
        # Fixed random orthogonal projection
        proj = torch.randn(d_in, rank)
        proj, _ = torch.linalg.qr(proj)
        self.register_buffer("projection", proj)

    def forward(self, x):
        return LowRankLinear.apply(
            x, self.linear.weight, self.linear.bias, self.projection
        )

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


def apply_lowrank_backward(
    model: nn.Module,
    rank: int = 256,
    target_modules: list = None,
) -> nn.Module:
    """Replace targeted Linear layers with LowRankLinearWrapper.

    Args:
        model: The model to modify in-place
        rank: Projection rank (lower = faster but less accurate gradients)
        target_modules: List of module name prefixes to target.
            If None, targets all Linear layers with d_in > rank * 2.

    Returns:
        Modified model (same object, modified in-place)
    """
    replacements = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features <= rank * 2:
            continue
        if target_modules is not None:
            if not any(t in name for t in target_modules):
                continue
        replacements[name] = module

    for name, module in replacements.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        wrapper = LowRankLinearWrapper(module, rank=rank)
        setattr(parent, parts[-1], wrapper)

    if replacements:
        print(f"Applied low-rank backward (rank={rank}) to {len(replacements)} Linear layers")
    return model
