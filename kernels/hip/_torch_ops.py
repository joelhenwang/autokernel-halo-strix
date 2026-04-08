"""
AutoKernel — Register HIP kernels as torch.library custom ops.

This allows torch.compile (Inductor) to trace through a model graph that uses
our HIP kernels as opaque nodes, fusing all PyTorch-native operations AROUND them.

Usage:
    import kernels.hip._torch_ops  # triggers registration
    # Then call via torch.ops.autokernel.<op_name>(...)
"""

import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

@torch.library.custom_op("autokernel::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from kernels.hip.rmsnorm import kernel_fn
    return kernel_fn(x, weight)


@rmsnorm_op.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape)


# ---------------------------------------------------------------------------
# Rotary Embedding (fp32 intermediate — matches LLaMA's .float() promotion)
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


# ---------------------------------------------------------------------------
# SiLU Gate Multiply (SwiGLU activation)
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


# ---------------------------------------------------------------------------
# Fused Residual Add + RMSNorm (dual output)
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
