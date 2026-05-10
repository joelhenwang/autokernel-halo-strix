"""Triton fused SwiGLU elementwise kernel (Phase D.B).

Replaces the ``silu(gate) * up`` elementwise step of a SwiGLU FFN with a
single Triton dispatch. Equivalent in FLOPs to HIP's ``silu_gate_mul`` but
routed through ``torch.autograd.Function`` so gradient flow is preserved
(unlike the raw pybind path, see docs/perf/autokernel-deep-analysis.md).

Scope: this kernel does NOT fuse the surrounding w_gate_up / w_down GEMMs.
Those remain as ``nn.Linear`` calls dispatched through rocBLAS. Fusing all
three would require a proper epilogue-fusion GEMM kernel; see the roadmap
in knowledge/kernels/triton_author_guide.md.

Ship gate (per Phase D.A):
  - parity: fp16 rel_err < 5e-3 on both fwd and bwd
  - isolated fwd+bwd speedup: >= 1.05x vs the autograd-correct HIP kernel
  - e2e: >= 5% total-step wall improvement over Sprint 1.5 C3 baseline
  - loss parity at step 200: |Δloss| <= 0.05

Usage (as a direct replacement for silu(gate) * up):

    from kernels.triton.fused_swiglu import fused_swiglu
    y = fused_swiglu(gate, up)   # autograd-safe

or as an nn.Module replacement:

    from kernels.triton.fused_swiglu import TritonFusedSwiGLUModule
    ffn = TritonFusedSwiGLUModule(dim=768, hidden=2048)
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TRITON_AVAILABLE = False
    triton = None  # type: ignore
    tl = None      # type: ignore


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_swiglu_fwd_kernel(
        OUT_ptr, GATE_ptr, UP_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        """For each element: out = silu(gate) * up = (gate * sigmoid(gate)) * up."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        gate = tl.load(GATE_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(UP_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-gate))
        silu_gate = gate * sig
        out = silu_gate * up

        tl.store(OUT_ptr + offs, out.to(tl.float16), mask=mask)

    @triton.jit
    def _fused_swiglu_bwd_kernel(
        GRAD_GATE_ptr, GRAD_UP_ptr,
        GATE_ptr, UP_ptr, GRAD_OUT_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        """Compute grad_gate and grad_up given saved gate, up and upstream grad_out.

        d(silu(g)*u) / dg = u * sigmoid(g) * (1 + g * (1 - sigmoid(g)))
        d(silu(g)*u) / du = g * sigmoid(g)
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        gate = tl.load(GATE_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(UP_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(GRAD_OUT_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-gate))
        silu_gate = gate * sig
        d_silu = sig * (1.0 + gate * (1.0 - sig))

        grad_gate = grad_out * up * d_silu
        grad_up = grad_out * silu_gate

        tl.store(GRAD_GATE_ptr + offs, grad_gate.to(tl.float16), mask=mask)
        tl.store(GRAD_UP_ptr + offs, grad_up.to(tl.float16), mask=mask)


class _FusedSwiGLUTritonFn(torch.autograd.Function):
    """torch.autograd.Function wrapping the Triton elementwise SwiGLU."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available; eager path should be used")
        assert gate.is_contiguous() and up.is_contiguous(), \
            "gate and up must be contiguous; caller should call .contiguous()"
        assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"
        assert gate.dtype == torch.float16 and up.dtype == torch.float16, \
            "fp16 required (use eager fallback for fp32)"

        out = torch.empty_like(gate)
        N = gate.numel()
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _fused_swiglu_fwd_kernel[grid](out, gate, up, N, BLOCK=BLOCK)

        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available for backward")
        gate, up = ctx.saved_tensors
        assert grad_out.is_contiguous(), "grad_out must be contiguous"

        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        N = gate.numel()
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _fused_swiglu_bwd_kernel[grid](
            grad_gate, grad_up, gate, up, grad_out, N, BLOCK=BLOCK,
        )
        return grad_gate, grad_up


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Public entry point. Computes ``silu(gate) * up``.

    Autograd-safe: backward propagates gradients to both gate and up via
    the Triton backward kernel.

    Falls back to eager ``F.silu(gate) * up`` when Triton is unavailable or
    inputs are not fp16/CUDA (e.g. fp32 eval).
    """
    if (not _TRITON_AVAILABLE) or (gate.dtype != torch.float16) or (not gate.is_cuda):
        return F.silu(gate) * up
    return _FusedSwiGLUTritonFn.apply(gate.contiguous(), up.contiguous())


class TritonFusedSwiGLUModule(nn.Module):
    """Drop-in SwiGLU FFN with Triton elementwise kernel.

    Matches the layout used by OdinFlat/OdinHalo's SwiGLU FFN:
        self.w_gate_up: Linear(dim -> 2*hidden)
        self.w_down:    Linear(hidden -> dim)

    Forward:
        gate, up = w_gate_up(x).chunk(2, -1)
        activated = silu(gate) * up   # ← Triton fused here
        return w_down(activated)
    """

    def __init__(self, dim: int, hidden: int, bias: bool = False):
        super().__init__()
        self.w_gate_up = nn.Linear(dim, 2 * hidden, bias=bias)
        self.w_down = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        activated = fused_swiglu(gate.contiguous(), up.contiguous())
        return self.w_down(activated)

    @classmethod
    def from_existing(cls, original: nn.Module) -> "TritonFusedSwiGLUModule":
        """Build a new module from an existing SwiGLU-shaped module.

        Expects ``original`` to have ``.w_gate_up`` and ``.w_down`` Linears.
        """
        assert hasattr(original, "w_gate_up") and hasattr(original, "w_down"), \
            "original must have w_gate_up and w_down attrs"
        dim = original.w_gate_up.in_features
        hidden = original.w_gate_up.out_features // 2
        new = cls(dim, hidden, bias=original.w_gate_up.bias is not None)
        new.w_gate_up.load_state_dict(original.w_gate_up.state_dict())
        new.w_down.load_state_dict(original.w_down.state_dict())
        return new


__all__ = [
    "fused_swiglu",
    "TritonFusedSwiGLUModule",
    "_FusedSwiGLUTritonFn",
]
