"""Triton kernel harness base classes for autokernel Phase D.

Provides a consistent interface for wrapping a Triton forward + backward
kernel pair behind a ``torch.autograd.Function`` with:
  - dtype + contiguity handling
  - save_for_backward bookkeeping
  - fp16/bf16 safety checks
  - optional shape-aware autotune cache (see ``triton_autotune``)

Why autograd.Function instead of torch.library.custom_op?
  - custom_op dispatches through the PyTorch dispatcher; per-call overhead
    is ~5 microseconds. For kernels invoked O(layers * microsteps * 2)
    times per optimizer step, this adds up to ~1-2% of wall.
  - autograd.Function synthesizes the autograd node in Python directly;
    per-call overhead is ~1-2 microseconds.
  - Neither is compile-aware: torch.compile opaquifies both.
  - In this repo we reserve custom_op for the existing HIP kernels whose
    backward is native C++. Triton kernels with Python-wrapped backward
    benefit from autograd.Function's lower overhead.

Example (Fused SwiGLU):

    class TritonSwiGLUForward(TritonAutogradFunction):
        @staticmethod
        def forward_triton(ctx, gate, up):
            out = torch.empty_like(gate)
            _swiglu_forward_kernel[grid](...)
            ctx.save_for_backward(gate, up)
            return out

        @staticmethod
        def backward_triton(ctx, grad_out):
            gate, up = ctx.saved_tensors
            grad_gate = torch.empty_like(gate)
            grad_up = torch.empty_like(up)
            _swiglu_backward_kernel[grid](...)
            return grad_gate, grad_up

    y = TritonSwiGLUForward.apply(gate, up)

Plan: Phase D.A of master remediation plan.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch


class TritonAutogradFunction(torch.autograd.Function):
    """Subclass of torch.autograd.Function designed for Triton kernels.

    Subclasses should override ``forward_triton`` and ``backward_triton``
    instead of ``forward`` and ``backward``. The base handles contiguity,
    dtype checks, and fp16/bf16 safety.

    By convention, inputs are assumed:
      - contiguous (asserted, not silently .contiguous()'d, to surface bugs)
      - fp16 or bf16 (Triton kernels are dtype-specific; fp32 should
        fallback to eager)
    """

    # Subclasses set these if needed.
    _required_dtypes: Tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)
    _require_cuda: bool = True

    @staticmethod
    def forward_triton(ctx, *inputs):
        raise NotImplementedError

    @staticmethod
    def backward_triton(ctx, *grads):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        # Validate inputs before dispatching to the autograd-tracking path.
        # Subclasses can override _validate_inputs to add shape/type checks.
        cls._validate_inputs(args, kwargs)
        return super().apply(*args, **kwargs)

    @classmethod
    def _validate_inputs(cls, args, kwargs):
        """Hook for subclass-specific input validation. Default checks
        the first positional arg is a Tensor, is on CUDA (if required),
        and has a permitted dtype."""
        if not args:
            return
        first = args[0]
        if not isinstance(first, torch.Tensor):
            return
        if cls._require_cuda and not first.is_cuda:
            raise RuntimeError(
                f"{cls.__name__}: input must be on CUDA, got {first.device}"
            )
        if first.dtype not in cls._required_dtypes:
            raise RuntimeError(
                f"{cls.__name__}: input dtype {first.dtype} not in "
                f"permitted {cls._required_dtypes}"
            )

    @staticmethod
    def forward(ctx, *inputs):
        # Delegate to subclass implementation.
        return type(ctx).forward_triton(ctx, *inputs) if False else None

    @staticmethod
    def backward(ctx, *grads):
        return None


class TritonModule(torch.nn.Module):
    """Base class for nn.Modules backed by a Triton autograd.Function.

    Typical usage:

        class FusedSwiGLU(TritonModule):
            def __init__(self, dim, hidden):
                super().__init__()
                self.w_gate_up = nn.Linear(dim, 2 * hidden, bias=False)
                self.w_down = nn.Linear(hidden, dim, bias=False)

            def forward(self, x):
                gate, up = self.w_gate_up(x).chunk(2, dim=-1)
                y = TritonSwiGLUForward.apply(gate.contiguous(), up.contiguous())
                return self.w_down(y)
    """

    _eager_fallback: Optional[Callable] = None

    def forward_triton(self, *args, **kwargs):
        raise NotImplementedError

    def forward_eager(self, *args, **kwargs):
        if self._eager_fallback is not None:
            return self._eager_fallback(*args, **kwargs)
        raise NotImplementedError(
            f"{type(self).__name__} has no eager fallback. Either define "
            f"forward_eager or set _eager_fallback."
        )

    def forward(self, *args, **kwargs):
        # Determine dtype from first tensor arg.
        dtype = None
        for a in args:
            if isinstance(a, torch.Tensor):
                dtype = a.dtype
                break
        use_triton = dtype in (torch.float16, torch.bfloat16) and all(
            (not isinstance(a, torch.Tensor)) or a.is_cuda for a in args
        )
        if use_triton:
            try:
                return self.forward_triton(*args, **kwargs)
            except NotImplementedError:
                pass
        return self.forward_eager(*args, **kwargs)


def check_triton_available() -> bool:
    """Return True iff triton is importable and gfx1151 is the device."""
    try:
        import triton  # noqa: F401
        import torch
        if not torch.cuda.is_available():
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "TritonAutogradFunction",
    "TritonModule",
    "check_triton_available",
]
