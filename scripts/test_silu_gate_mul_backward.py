"""Track 3: silu_gate_mul backward numeric harness.

Phase II P2 showed silu_gate_mul HIP kernel forward trains at loss parity
despite the raw-kernel autograd issue (w_gate_up gradient was likely
zero). Phase III attempted to enable the autograd-registered op but
training exploded.

Question this harness answers:
  A. Does the HIP silu_gate_mul_backward kernel produce numerically
     correct gradients vs. PyTorch reference?
  B. If the HIP kernel is wrong: where does it diverge?
  C. Does the autograd-registered forward (via torch.library.custom_op)
     produce correct gradients through autograd, regardless of whether
     HIP backward or PyTorch fallback backward runs?

Run:
    python scripts/test_silu_gate_mul_backward.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def reference_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: silu(gate) * up."""
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)


def reference_backward(gate, up, grad_output):
    """Hand-computed backward for verification."""
    g = grad_output.float()
    g_f = gate.float()
    u_f = up.float()

    sig = torch.sigmoid(g_f)
    silu_gate = g_f * sig
    d_silu = sig * (1.0 + g_f * (1.0 - sig))
    grad_gate = g * u_f * d_silu
    grad_up = g * silu_gate

    return grad_gate.to(gate.dtype), grad_up.to(up.dtype)


def _stats(name, hip, ref):
    diff = (hip.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-8)
    print(f"  {name}:  max_abs_err={diff.max().item():.6e}  "
          f"mean_abs_err={diff.mean().item():.6e}  "
          f"max_rel_err={rel.max().item():.6e}  "
          f"norm_ratio(hip/ref)={hip.float().norm().item() / ref.float().norm().item():.6f}")


def test_hip_backward_matches_reference():
    """Direct test: HIP backward kernel vs reference math."""
    from kernels.hip.silu_gate_mul_backward import kernel_fn as hip_bwd

    torch.manual_seed(0)
    D = 2816  # typical SwiGLU inner dim
    B_T = 4096

    gate = torch.randn(B_T, D, device="cuda", dtype=torch.float16) * 2.0
    up = torch.randn(B_T, D, device="cuda", dtype=torch.float16) * 0.5
    grad_out = torch.randn(B_T, D, device="cuda", dtype=torch.float16) * 0.1

    ref_grad_gate, ref_grad_up = reference_backward(gate, up, grad_out)

    hip_grad_gate, hip_grad_up = hip_bwd(gate, up, grad_out)

    print("\n[A] HIP silu_gate_mul_backward vs reference:")
    _stats("grad_gate", hip_grad_gate, ref_grad_gate)
    _stats("grad_up  ", hip_grad_up, ref_grad_up)


def test_autograd_forward_backward_chain():
    """Chain: forward through autograd op, loss.backward() — compare to
    PyTorch-only chain on the SAME input tensors."""
    import kernels.hip._torch_ops  # triggers registration

    torch.manual_seed(1)
    D = 2816
    B_T = 1024

    gate_src = torch.randn(B_T, D, device="cuda", dtype=torch.float16) * 2.0
    up_src = torch.randn(B_T, D, device="cuda", dtype=torch.float16) * 0.5

    # Path 1: PyTorch reference
    gate_ref = gate_src.detach().clone().requires_grad_(True)
    up_ref = up_src.detach().clone().requires_grad_(True)
    y_ref = F.silu(gate_ref.float()) * up_ref.float()
    loss_ref = y_ref.pow(2).mean()
    loss_ref.backward()

    # Path 2: autograd op (HIP forward + HIP backward)
    gate_hip = gate_src.detach().clone().requires_grad_(True)
    up_hip = up_src.detach().clone().requires_grad_(True)
    y_hip = torch.ops.autokernel.silu_gate_mul(gate_hip, up_hip)
    loss_hip = y_hip.float().pow(2).mean()
    loss_hip.backward()

    print("\n[B] autograd op (HIP forward + HIP backward) vs PyTorch reference chain:")
    print(f"  loss_ref={loss_ref.item():.6f}  loss_hip={loss_hip.item():.6f}")
    _stats("gate.grad", gate_hip.grad, gate_ref.grad)
    _stats("up.grad  ", up_hip.grad, up_ref.grad)


def test_autograd_with_pytorch_bwd_fallback():
    """Same as above but with AUTOKERNEL_NO_BWD_HIP=1 so PyTorch
    backward fallback runs. Confirms the autograd registration wiring
    itself is correct (not an issue with HIP backward specifically)."""
    # Set env BEFORE importing _torch_ops to take effect
    os.environ["AUTOKERNEL_NO_BWD_HIP"] = "1"
    # Re-import: the flag is cached at module load
    import importlib
    import kernels.hip._torch_ops as tops
    importlib.reload(tops)

    torch.manual_seed(1)
    D = 2816
    B_T = 1024

    gate_src = torch.randn(B_T, D, device="cuda", dtype=torch.float16) * 2.0
    up_src = torch.randn(B_T, D, device="cuda", dtype=torch.float16) * 0.5

    gate_ref = gate_src.detach().clone().requires_grad_(True)
    up_ref = up_src.detach().clone().requires_grad_(True)
    y_ref = F.silu(gate_ref.float()) * up_ref.float()
    loss_ref = y_ref.pow(2).mean()
    loss_ref.backward()

    gate_hip = gate_src.detach().clone().requires_grad_(True)
    up_hip = up_src.detach().clone().requires_grad_(True)
    y_hip = torch.ops.autokernel.silu_gate_mul(gate_hip, up_hip)
    loss_hip = y_hip.float().pow(2).mean()
    loss_hip.backward()

    print("\n[C] autograd op with AUTOKERNEL_NO_BWD_HIP=1 (PyTorch fallback bwd):")
    print(f"  loss_ref={loss_ref.item():.6f}  loss_hip={loss_hip.item():.6f}")
    _stats("gate.grad", gate_hip.grad, gate_ref.grad)
    _stats("up.grad  ", up_hip.grad, up_ref.grad)

    # Restore env
    os.environ["AUTOKERNEL_NO_BWD_HIP"] = "0"


def main() -> int:
    if not torch.cuda.is_available():
        print("Need CUDA"); return 1
    print("=" * 70)
    print("Track 3: silu_gate_mul backward numeric harness")
    print("=" * 70)
    test_hip_backward_matches_reference()
    test_autograd_forward_backward_chain()
    test_autograd_with_pytorch_bwd_fallback()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
