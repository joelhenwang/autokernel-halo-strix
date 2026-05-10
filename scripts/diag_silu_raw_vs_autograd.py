"""Diagnostic: does the raw silu_gate_mul kernel call preserve grad_fn?

Test what actually happens to gradients when _FusedSwiGLUReplacement
calls self.kernel_fn(gate, up) directly (the "original" code path).
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

def main():
    from kernels.hip.silu_gate_mul import kernel_fn

    gate = torch.randn(32, 2816, device="cuda", dtype=torch.float16, requires_grad=True)
    up = torch.randn(32, 2816, device="cuda", dtype=torch.float16, requires_grad=True)

    print("Input requires_grad: gate={}, up={}".format(gate.requires_grad, up.requires_grad))

    # Raw kernel call — the way _FusedSwiGLUReplacement actually does it
    out_raw = kernel_fn(gate.contiguous(), up.contiguous())
    print(f"\nRaw kernel output: shape={out_raw.shape} dtype={out_raw.dtype}")
    print(f"  requires_grad:   {out_raw.requires_grad}")
    print(f"  grad_fn:         {out_raw.grad_fn}")

    try:
        out_raw.sum().backward()
        print(f"  backward OK: gate.grad is None={gate.grad is None}  up.grad is None={up.grad is None}")
    except Exception as e:
        print(f"  backward FAIL: {type(e).__name__}: {e}")

    # Reset
    gate.grad = None
    up.grad = None

    # Autograd-registered op
    import kernels.hip._torch_ops  # noqa: F401
    out_ag = torch.ops.autokernel.silu_gate_mul(gate.contiguous(), up.contiguous())
    print(f"\nAutograd op output: shape={out_ag.shape} dtype={out_ag.dtype}")
    print(f"  requires_grad:   {out_ag.requires_grad}")
    print(f"  grad_fn:         {out_ag.grad_fn}")
    out_ag.sum().backward()
    print(f"  backward OK: gate.grad finite={torch.isfinite(gate.grad).all().item()}  up.grad finite={torch.isfinite(up.grad).all().item()}")

    print("\n=== VERDICT ===")
    print(f"  Raw kernel_fn produces tensor with grad_fn: {out_raw.grad_fn is not None}")
    print(f"  Autograd op produces tensor with grad_fn:   {out_ag.grad_fn is not None}")


if __name__ == "__main__":
    main()
