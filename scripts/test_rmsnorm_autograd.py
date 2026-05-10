"""Gradient test: does autokernel's _RMSNormReplacement have working backward?

The _RMSNormReplacement class in autokernel/_patterns.py directly calls
the HIP kernel_fn without routing through the autograd-registered custom
op in kernels/hip/_torch_ops.py. This test checks whether gradients flow
correctly.

Compared against reference RMSNorm with proper autograd.

Run:
    python scripts/test_rmsnorm_autograd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("Need CUDA"); return 1

    from models._components import RMSNorm
    from autokernel._patterns import _RMSNormReplacement
    from kernels.hip.rmsnorm import kernel_fn

    torch.manual_seed(0)
    B, T, D = 2, 64, 768

    # Identical initial state for both paths
    ref_mod = RMSNorm(D).cuda()
    hip_mod = _RMSNormReplacement(ref_mod, kernel_fn).cuda()

    # IMPORTANT: force fp16 input directly so the HIP kernel's fp16 path
    # is exercised (not the fp32 PyTorch fallback).
    x_ref = torch.randn(B, T, D, device="cuda", dtype=torch.float16,
                        requires_grad=True)
    x_hip = x_ref.detach().clone().requires_grad_(True)

    # NO autocast — inputs already fp16.
    y_ref = ref_mod(x_ref)
    y_hip = hip_mod(x_hip)

    print(f"Forward dtype: y_ref={y_ref.dtype}  y_hip={y_hip.dtype}")
    print(f"Forward norm:  ref={y_ref.float().norm().item():.4f}  hip={y_hip.float().norm().item():.4f}")
    print(f"Forward max diff: {(y_ref.float() - y_hip.float()).abs().max().item():.6f}")

    # Backward pass with a fixed gradient
    grad_upstream = torch.randn_like(y_ref)
    y_ref.backward(grad_upstream)
    y_hip.backward(grad_upstream)

    def _describe(name, g):
        if g is None:
            return f"  {name}: None"
        return (f"  {name}: shape={tuple(g.shape)} dtype={g.dtype} "
                f"norm={g.float().norm().item():.6f} "
                f"has_nan={torch.isnan(g).any().item()} "
                f"has_inf={torch.isinf(g).any().item()}")

    print("\nGradients:")
    print(_describe("x_ref.grad", x_ref.grad))
    print(_describe("x_hip.grad", x_hip.grad))
    print(_describe("ref_mod.weight.grad", ref_mod.weight.grad))
    print(_describe("hip_mod.weight.grad", hip_mod.weight.grad))

    if x_hip.grad is None or ref_mod.weight.grad is None or hip_mod.weight.grad is None:
        print("\n**VERDICT: HIP path does NOT produce gradients (autograd broken)**")
        return 0

    x_diff = (x_ref.grad - x_hip.grad).abs()
    w_diff = (ref_mod.weight.grad - hip_mod.weight.grad).abs()
    print(f"\nGradient agreement:")
    print(f"  x.grad:      max_abs_diff={x_diff.max().item():.6f}  "
          f"mean={x_diff.mean().item():.6f}  ratio={x_hip.grad.float().norm().item() / x_ref.grad.float().norm().item():.6f}")
    print(f"  weight.grad: max_abs_diff={w_diff.max().item():.6f}  "
          f"mean={w_diff.mean().item():.6f}  ratio={hip_mod.weight.grad.float().norm().item() / ref_mod.weight.grad.float().norm().item():.6f}")

    # If ratio is ~1, gradients are correct (fp16 noise only). If anything
    # else, gradient semantics are wrong.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
