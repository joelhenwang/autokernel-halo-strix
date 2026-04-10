"""
AutoKernel — Register HIP kernels as torch.library custom ops.

This allows torch.compile (Inductor) to trace through a model graph that uses
our HIP kernels as opaque nodes, fusing all PyTorch-native operations AROUND them.

Backward passes use pure PyTorch ops so autograd works for training.
Forward uses HIP kernels (fast), backward uses PyTorch (correct).

Usage:
    import kernels.hip._torch_ops  # triggers registration
    # Then call via torch.ops.autokernel.<op_name>(...)
"""

import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# RMSNorm
# forward: y = x * weight / sqrt(mean(x^2) + eps)
# ---------------------------------------------------------------------------

@torch.library.custom_op("autokernel::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from kernels.hip.rmsnorm import kernel_fn
    return kernel_fn(x, weight)


@rmsnorm_op.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape)


def _rmsnorm_setup(ctx, inputs, output):
    x, weight = inputs
    ctx.save_for_backward(x, weight)
    ctx.eps = 1e-6


def _rmsnorm_backward(ctx, grad_output):
    x, weight = ctx.saved_tensors

    # Use fused HIP backward kernel for fp16 inputs
    if x.dtype == torch.float16 and x.is_cuda:
        from kernels.hip.rmsnorm_backward import kernel_fn as rmsnorm_bwd_fn
        grad_x, grad_weight = rmsnorm_bwd_fn(x, weight, grad_output)
        return grad_x, grad_weight.to(weight.dtype)

    # FP32 fallback: pure PyTorch for numerical stability
    eps = ctx.eps
    x_f = x.float()
    w_f = weight.float()
    g_f = grad_output.float()

    rms_sq = x_f.pow(2).mean(-1, keepdim=True) + eps
    rms_inv = rms_sq.rsqrt()
    normed = x_f * rms_inv
    D = x_f.shape[-1]

    grad_weight = (g_f * normed).sum(dim=tuple(range(g_f.ndim - 1)))
    grad_normed = g_f * w_f
    grad_x = grad_normed * rms_inv - normed * (grad_normed * normed).sum(-1, keepdim=True) / D

    return grad_x.to(x.dtype), grad_weight.to(weight.dtype)


rmsnorm_op.register_autograd(_rmsnorm_backward, setup_context=_rmsnorm_setup)


# ---------------------------------------------------------------------------
# Rotary Embedding (fp32 intermediate — matches LLaMA's .float() promotion)
# forward: y = x * cos + rotate_half(x) * sin
# ---------------------------------------------------------------------------

@torch.library.custom_op("autokernel::rotary_emb_fp32", mutates_args=())
def rotary_emb_fp32_op(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    from kernels.hip.rotary_embedding import kernel_fn_fp32
    return kernel_fn_fp32(x, cos, sin)


@rotary_emb_fp32_op.register_fake
def _(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return x.new_empty(x.shape)


def _rotary_setup(ctx, inputs, output):
    x, cos, sin = inputs
    ctx.save_for_backward(cos, sin)
    ctx.x_dtype = x.dtype


def _rotary_backward(ctx, grad_output):
    cos, sin = ctx.saved_tensors

    # Use fused HIP backward kernel for fp16 4D inputs
    if (grad_output.dtype == torch.float16 and grad_output.is_cuda
            and grad_output.dim() == 4):
        from kernels.hip.rotary_embedding_backward import kernel_fn as rotary_bwd_fn
        grad_x = rotary_bwd_fn(grad_output, cos, sin)
        return grad_x, None, None

    # FP32 fallback
    g = grad_output.float()
    c = cos.float()
    s = sin.float()

    def rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)

    grad_x = g * c + rotate_half(g) * (-s)

    return grad_x.to(ctx.x_dtype), None, None


rotary_emb_fp32_op.register_autograd(_rotary_backward, setup_context=_rotary_setup)


# ---------------------------------------------------------------------------
# SiLU Gate Multiply (SwiGLU activation)
# forward: y = silu(gate) * up = gate * sigmoid(gate) * up
# ---------------------------------------------------------------------------

@torch.library.custom_op("autokernel::silu_gate_mul", mutates_args=())
def silu_gate_mul_op(
    gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    from kernels.hip.silu_gate_mul import kernel_fn
    return kernel_fn(gate, up)


@silu_gate_mul_op.register_fake
def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return gate.new_empty(gate.shape)


def _silu_gate_mul_setup(ctx, inputs, output):
    gate, up = inputs
    ctx.save_for_backward(gate, up)


def _silu_gate_mul_backward(ctx, grad_output):
    gate, up = ctx.saved_tensors

    # Use fused HIP backward kernel for fp16 inputs
    if gate.dtype == torch.float16 and gate.is_cuda:
        from kernels.hip.silu_gate_mul_backward import kernel_fn as silu_bwd_fn
        return silu_bwd_fn(gate, up, grad_output)

    # FP32 fallback
    g = grad_output.float()
    gate_f = gate.float()
    up_f = up.float()

    sig = torch.sigmoid(gate_f)
    silu_gate = gate_f * sig
    d_silu = sig * (1.0 + gate_f * (1.0 - sig))
    grad_gate = g * up_f * d_silu
    grad_up = g * silu_gate

    return grad_gate.to(gate.dtype), grad_up.to(up.dtype)


silu_gate_mul_op.register_autograd(_silu_gate_mul_backward, setup_context=_silu_gate_mul_setup)


# ---------------------------------------------------------------------------
# Fused Residual Add + RMSNorm (dual output)
# forward: hidden = x + residual; normed = rmsnorm(hidden, weight)
# returns: (hidden, normed)
# ---------------------------------------------------------------------------

@torch.library.custom_op("autokernel::fused_res_rmsnorm", mutates_args=())
def fused_res_rmsnorm_op(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    from kernels.hip.fused_residual_add_rmsnorm import kernel_fn_dual
    return kernel_fn_dual(x, residual, weight)


@fused_res_rmsnorm_op.register_fake
def _(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty(x.shape), x.new_empty(x.shape)


def _fused_res_rmsnorm_setup(ctx, inputs, output):
    x, residual, weight = inputs
    hidden, normed = output
    ctx.save_for_backward(hidden, weight)
    ctx.eps = 1e-6


def _fused_res_rmsnorm_backward(ctx, grad_hidden, grad_normed):
    hidden, weight = ctx.saved_tensors

    # Use fused HIP backward kernel for fp16 inputs
    if hidden.dtype == torch.float16 and hidden.is_cuda:
        from kernels.hip.fused_residual_rmsnorm_backward import kernel_fn as fused_bwd_fn
        grad_x, grad_residual, grad_weight = fused_bwd_fn(
            hidden, weight, grad_hidden, grad_normed
        )
        return grad_x, grad_residual, grad_weight.to(weight.dtype)

    # FP32 fallback
    eps = ctx.eps
    h_f = hidden.float()
    w_f = weight.float()
    gn_f = grad_normed.float()

    rms_sq = h_f.pow(2).mean(-1, keepdim=True) + eps
    rms_inv = rms_sq.rsqrt()
    normed_h = h_f * rms_inv
    D = h_f.shape[-1]

    grad_weight = (gn_f * normed_h).sum(dim=tuple(range(gn_f.ndim - 1)))
    grad_normed_scaled = gn_f * w_f
    grad_h_from_norm = grad_normed_scaled * rms_inv - normed_h * (grad_normed_scaled * normed_h).sum(-1, keepdim=True) / D

    total_grad_h = grad_hidden.float() + grad_h_from_norm
    grad_x = total_grad_h.to(hidden.dtype)

    return grad_x, grad_x, grad_weight.to(weight.dtype)


fused_res_rmsnorm_op.register_autograd(
    _fused_res_rmsnorm_backward, setup_context=_fused_res_rmsnorm_setup
)


# ---------------------------------------------------------------------------
# Selective Scan (diagonal SSM)
# forward: state[t] = dA[t] * state[t-1] + dBx[t]; y[t] = C[t]*state[t] + D*x[t]
# ---------------------------------------------------------------------------

@torch.library.custom_op("autokernel::selective_scan", mutates_args=())
def selective_scan_op(
    dA: torch.Tensor, dBx: torch.Tensor,
    C: torch.Tensor, D: torch.Tensor, x: torch.Tensor,
) -> torch.Tensor:
    from kernels.hip.selective_scan import kernel_fn
    return kernel_fn(dA, dBx, C, D, x)


@selective_scan_op.register_fake
def _(
    dA: torch.Tensor, dBx: torch.Tensor,
    C: torch.Tensor, D: torch.Tensor, x: torch.Tensor,
) -> torch.Tensor:
    return dA.new_empty(dA.shape)


def _selective_scan_setup(ctx, inputs, output):
    dA, dBx, C, D, x = inputs
    # Save what we need for backward: recompute states via forward scan
    ctx.save_for_backward(dA, dBx, C, D, x, output)


def _selective_scan_backward(ctx, grad_y):
    dA, dBx, C, D, x, y = ctx.saved_tensors

    # Use parallel HIP backward kernel for fp32 GPU inputs
    if dA.is_cuda and dA.dtype == torch.float32:
        from kernels.hip.selective_scan_backward import kernel_fn as scan_bwd_fn
        gy = grad_y.float()
        grad_dA, grad_dBx, grad_C, grad_D, grad_x = scan_bwd_fn(
            dA, dBx, C, D, x, gy
        )
        return (
            grad_dA.to(dA.dtype),
            grad_dBx.to(dBx.dtype),
            grad_C.to(C.dtype),
            grad_D.to(D.dtype),
            grad_x.to(x.dtype),
        )

    # Fallback: sequential Python loops
    gy = grad_y.float()
    dA_f = dA.float()
    dBx_f = dBx.float()
    C_f = C.float()
    D_f = D.float()
    x_f = x.float()

    batch, seqlen, d_inner = dA.shape

    states = torch.zeros(batch, seqlen + 1, d_inner, dtype=torch.float32, device=dA.device)
    for t in range(seqlen):
        states[:, t + 1] = dA_f[:, t] * states[:, t] + dBx_f[:, t]

    grad_dA = torch.zeros_like(dA_f)
    grad_dBx = torch.zeros_like(dBx_f)
    grad_C = torch.zeros_like(C_f)
    grad_D = torch.zeros(d_inner, dtype=torch.float32, device=dA.device)
    grad_state = torch.zeros(batch, d_inner, dtype=torch.float32, device=dA.device)

    for t in range(seqlen - 1, -1, -1):
        grad_C[:, t] = gy[:, t] * states[:, t + 1]
        grad_state = grad_state + gy[:, t] * C_f[:, t]
        grad_D = grad_D + (gy[:, t] * x_f[:, t]).sum(0)
        grad_dA[:, t] = grad_state * states[:, t]
        grad_dBx[:, t] = grad_state
        grad_state = grad_state * dA_f[:, t]

    grad_x = gy * D_f

    return (
        grad_dA.to(dA.dtype),
        grad_dBx.to(dBx.dtype),
        grad_C.to(C.dtype),
        grad_D.to(D.dtype),
        grad_x.to(x.dtype),
    )


selective_scan_op.register_autograd(
    _selective_scan_backward, setup_context=_selective_scan_setup
)


# ---------------------------------------------------------------------------
# Fused PLE Gate (Per-Layer Embeddings Path A)
# forward: GELU(h @ W_down^T) @ W_up^T, then RMSNorm
# ---------------------------------------------------------------------------

@torch.library.custom_op("autokernel::fused_ple_gate", mutates_args=())
def fused_ple_gate_op(
    h: torch.Tensor, w_down: torch.Tensor,
    w_up: torch.Tensor, norm_weight: torch.Tensor,
) -> torch.Tensor:
    from kernels.hip.fused_ple_gate import kernel_fn
    return kernel_fn(h, w_down, w_up, norm_weight)


@fused_ple_gate_op.register_fake
def _(
    h: torch.Tensor, w_down: torch.Tensor,
    w_up: torch.Tensor, norm_weight: torch.Tensor,
) -> torch.Tensor:
    return h.new_empty(h.shape)


def _fused_ple_gate_setup(ctx, inputs, output):
    h, w_down, w_up, norm_weight = inputs
    ctx.save_for_backward(h, w_down, w_up, norm_weight, output)
    ctx.eps = 1e-6


def _fused_ple_gate_backward(ctx, grad_output):
    h, w_down, w_up, norm_weight, out = ctx.saved_tensors
    eps = ctx.eps

    # Recompute forward in fp32 for stable gradients
    h_f = h.float()
    wd_f = w_down.float()
    wu_f = w_up.float()
    nw_f = norm_weight.float()
    g_f = grad_output.float()

    # Forward recompute
    bottleneck_pre = h_f @ wd_f.t()                          # (M, P)
    bottleneck = torch.nn.functional.gelu(bottleneck_pre)     # (M, P)
    raw = bottleneck @ wu_f.t()                               # (M, D)

    # RMSNorm forward
    rms_sq = raw.pow(2).mean(-1, keepdim=True) + eps
    rms_inv = rms_sq.rsqrt()
    normed = raw * rms_inv

    # RMSNorm backward
    D = raw.shape[-1]
    grad_normed = g_f * nw_f
    grad_raw = grad_normed * rms_inv - normed * (grad_normed * normed).sum(-1, keepdim=True) / D
    grad_norm_weight = (g_f * normed).sum(dim=tuple(range(g_f.ndim - 1)))

    # Linear backward: raw = bottleneck @ W_up^T
    grad_bottleneck = grad_raw @ wu_f                         # (M, P)
    grad_w_up = grad_raw.t() @ bottleneck                     # (D, P)

    # GELU backward: direct derivative formula (avoids nested autograd context)
    # GELU(x) = x * Φ(x) where Φ is the standard normal CDF
    # GELU'(x) = Φ(x) + x * φ(x) where φ is the standard normal PDF
    import math
    _SQRT2 = math.sqrt(2.0)
    _INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(bottleneck_pre / _SQRT2))
    pdf = _INV_SQRT2PI * torch.exp(-0.5 * bottleneck_pre.pow(2))
    grad_bp = grad_bottleneck * (cdf + bottleneck_pre * pdf)  # (M, P)

    # Linear backward: bottleneck_pre = h @ W_down^T
    grad_h = grad_bp @ wd_f                                   # (M, D)
    grad_w_down = grad_bp.t() @ h_f                           # (P, D)

    return (
        grad_h.to(h.dtype),
        grad_w_down.to(w_down.dtype),
        grad_w_up.to(w_up.dtype),
        grad_norm_weight.to(norm_weight.dtype),
    )


fused_ple_gate_op.register_autograd(
    _fused_ple_gate_backward, setup_context=_fused_ple_gate_setup
)
