"""v3 T-3.2 (2nd half) parity test: autokernel::causal_conv1d custom_op.

Verifies:
  1. Forward parity vs direct DaoAILab causal_conv1d_fn call.
  2. Gradient flow (y.grad_fn not None, x.grad / weight.grad / bias.grad populate).
  3. Backward numerical parity vs pure-PyTorch depthwise conv1d reference.

Background: DaoAILab's causal_conv1d_fn was being called directly inside
HyPEShortConvBlock, which is @torch.compiler.disable'd from surrounding
context and caused a graph break. Wrapping as torch.library.custom_op
eliminates the break and gives compiled-autograd a clean boundary.

Shapes: production-similar (B=2, D=1536, T=128) + regression (B=4, D=256, T=64).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trigger registration
import kernels.hip._torch_ops  # noqa: F401


def _native_causal_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch depthwise causal conv1d reference.

    x      : (B, D, T)
    weight : (D, K) per-channel kernel
    bias   : (D,)
    """
    D, K = weight.shape
    x_padded = F.pad(x, (K - 1, 0))
    return F.conv1d(x_padded, weight.unsqueeze(1), bias, groups=D)


def test_forward_parity():
    """Custom op forward matches native conv1d reference."""
    device = "cuda"
    torch.manual_seed(42)
    B, D, T, K = 4, 256, 64, 4

    x = torch.randn(B, D, T, device=device, dtype=torch.float16)
    weight = torch.randn(D, K, device=device, dtype=torch.float16)
    bias = torch.randn(D, device=device, dtype=torch.float16)

    y_hip = torch.ops.autokernel.causal_conv1d(x.clone(), weight, bias)
    y_native = _native_causal_conv1d(x, weight, bias)

    rel_err = (y_hip.float() - y_native.float()).abs().max().item()
    max_val = max(y_hip.float().abs().max().item(), y_native.float().abs().max().item())
    rel = rel_err / max(max_val, 1e-6)
    print(f"  forward parity: rel_err={rel:.3e}, max_val={max_val:.3f}")
    assert rel < 5e-3, f"Forward mismatch: rel={rel}"
    print("  [PASS] forward parity HIP shim vs native reference")


def test_backward_gradflow():
    """Backward produces non-None gradients on x, weight, bias."""
    device = "cuda"
    torch.manual_seed(42)
    B, D, T, K = 2, 128, 32, 4

    x = torch.randn(B, D, T, device=device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(D, K, device=device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=torch.float16, requires_grad=True)

    y = torch.ops.autokernel.causal_conv1d(x, weight, bias)
    # grad_fn must be populated (not None) — custom_op + register_autograd restores this.
    assert y.grad_fn is not None, "Output has no grad_fn — silent-freeze!"
    print(f"  y.grad_fn = {type(y.grad_fn).__name__}")

    y.float().sum().backward()

    assert x.grad is not None, "x.grad is None — gradient severed!"
    assert weight.grad is not None, "weight.grad is None — gradient severed!"
    assert bias.grad is not None, "bias.grad is None — gradient severed!"
    print(f"  x.grad norm      = {x.grad.norm().item():.4f}")
    print(f"  weight.grad norm = {weight.grad.norm().item():.4f}")
    print(f"  bias.grad norm   = {bias.grad.norm().item():.4f}")
    assert x.grad.norm().item() > 0, "x.grad is all zeros"
    assert weight.grad.norm().item() > 0, "weight.grad is all zeros"
    print("  [PASS] gradient flow restored (no silent-freeze)")


def test_backward_parity():
    """Custom-op backward matches native-ref backward within fp16 tolerance."""
    device = "cuda"
    torch.manual_seed(42)
    B, D, T, K = 2, 128, 32, 4

    # Reference path
    x_ref = torch.randn(B, D, T, device=device, dtype=torch.float16, requires_grad=True)
    w_ref = torch.randn(D, K, device=device, dtype=torch.float16, requires_grad=True)
    b_ref = torch.randn(D, device=device, dtype=torch.float16, requires_grad=True)

    y_ref = _native_causal_conv1d(x_ref, w_ref, b_ref)
    y_ref.float().sum().backward()

    # Custom-op path (clone inputs as leaf)
    x_hip = x_ref.detach().clone().requires_grad_(True)
    w_hip = w_ref.detach().clone().requires_grad_(True)
    b_hip = b_ref.detach().clone().requires_grad_(True)
    y_hip = torch.ops.autokernel.causal_conv1d(x_hip, w_hip, b_hip)
    y_hip.float().sum().backward()

    def _rel(a, b):
        num = (a.float() - b.float()).abs().max().item()
        den = max(a.float().abs().max().item(), b.float().abs().max().item(), 1e-6)
        return num / den

    x_rel = _rel(x_hip.grad, x_ref.grad)
    w_rel = _rel(w_hip.grad, w_ref.grad)
    b_rel = _rel(b_hip.grad, b_ref.grad)
    print(f"  grad_x      rel_err = {x_rel:.3e}")
    print(f"  grad_weight rel_err = {w_rel:.3e}")
    print(f"  grad_bias   rel_err = {b_rel:.3e}")
    assert x_rel < 5e-3, f"x grad mismatch: {x_rel}"
    assert w_rel < 5e-3, f"weight grad mismatch: {w_rel}"
    assert b_rel < 5e-3, f"bias grad mismatch: {b_rel}"
    print("  [PASS] backward parity within fp16 tolerance")


def test_production_shape_forward():
    """Sanity check at production-similar shapes."""
    device = "cuda"
    torch.manual_seed(0)
    B, D, T, K = 2, 1536, 128, 4

    x = torch.randn(B, D, T, device=device, dtype=torch.float16)
    weight = torch.randn(D, K, device=device, dtype=torch.float16)
    bias = torch.randn(D, device=device, dtype=torch.float16)

    y_hip = torch.ops.autokernel.causal_conv1d(x.clone(), weight, bias)
    y_native = _native_causal_conv1d(x, weight, bias)

    rel = (y_hip.float() - y_native.float()).abs().max().item() / max(
        y_native.float().abs().max().item(), 1e-6
    )
    print(f"  prod-shape forward rel_err = {rel:.3e}")
    assert rel < 5e-3
    print("  [PASS] production-shape forward parity")


if __name__ == "__main__":
    print("Test 1: forward parity (small shape)")
    test_forward_parity()
    print("\nTest 2: backward gradient flow")
    test_backward_gradflow()
    print("\nTest 3: backward numerical parity")
    test_backward_parity()
    print("\nTest 4: production-shape forward parity")
    test_production_shape_forward()
    print("\n[PASS] T-3.2 (2nd half) causal_conv1d shim custom_op with register_autograd")
