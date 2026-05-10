"""v3 T-3.2 parity test: autokernel::fused_rope_gate_mul custom_op.

Verifies:
  1. Forward matches the original HIP kernel_fn.
  2. Backward produces gradients (not None / grad_fn=None).
  3. Backward gradients match a pure-PyTorch reference implementation
     within fp16 tolerance.

Before this fix, kernel_fn was @torch.compiler.disable'd and had no
register_autograd, so upstream b/h_tilde gradients were silently severed
(silent-freeze bug). T-3.2 wraps it as torch.library.custom_op with
register_autograd, restoring gradient flow.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trigger registration
import kernels.hip._torch_ops  # noqa: F401


def _native_fused_rope_gate_mul(b, h_tilde, freqs_cos, freqs_sin, T, D, R_half):
    """Pure PyTorch reference (matches the HIP kernel semantics)."""
    M = b.shape[0]
    B = M // T
    b_view = b.view(B, T, D // 2, 2).float()
    h_view = h_tilde.view(B, T, D // 2, 2).float()
    b_even = b_view[..., 0]
    b_odd = b_view[..., 1]
    h_even = h_view[..., 0]
    h_odd = h_view[..., 1]
    pair_idx = torch.arange(D // 2, device=b.device) % R_half
    cos_bcast = freqs_cos[:, pair_idx].unsqueeze(0)
    sin_bcast = freqs_sin[:, pair_idx].unsqueeze(0)
    rot_a = b_even * cos_bcast - b_odd * sin_bcast
    rot_b = b_even * sin_bcast + b_odd * cos_bcast
    y_even = rot_a * h_even
    y_odd = rot_b * h_odd
    y = torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    return y.view(M, D).to(b.dtype)


def test_forward_parity():
    """Forward output of custom_op matches native reference."""
    device = "cuda"
    torch.manual_seed(42)
    B, T, D = 4, 64, 256
    R_half = 32

    b = torch.randn(B * T, D, device=device, dtype=torch.float16)
    h_tilde = torch.randn(B * T, D, device=device, dtype=torch.float16)
    freqs_cos = torch.randn(T, R_half, device=device, dtype=torch.float32)
    freqs_sin = torch.randn(T, R_half, device=device, dtype=torch.float32)

    y_hip = torch.ops.autokernel.fused_rope_gate_mul(
        b.clone(), h_tilde.clone(), freqs_cos, freqs_sin, T, D, R_half
    )
    y_native = _native_fused_rope_gate_mul(b, h_tilde, freqs_cos, freqs_sin, T, D, R_half)

    rel_err = (y_hip.float() - y_native.float()).abs().max().item()
    max_val = max(y_hip.float().abs().max().item(), y_native.float().abs().max().item())
    rel = rel_err / max(max_val, 1e-6)
    print(f"  forward parity: rel_err={rel:.3e}, max_val={max_val:.3f}")
    assert rel < 5e-3, f"Forward mismatch: rel={rel}"
    print("  [PASS] forward parity HIP vs native reference")


def test_backward_gradflow():
    """Backward produces non-None gradients on b and h_tilde."""
    device = "cuda"
    torch.manual_seed(42)
    B, T, D = 2, 32, 128
    R_half = 16

    b = torch.randn(B * T, D, device=device, dtype=torch.float16, requires_grad=True)
    h_tilde = torch.randn(B * T, D, device=device, dtype=torch.float16, requires_grad=True)
    freqs_cos = torch.randn(T, R_half, device=device, dtype=torch.float32)
    freqs_sin = torch.randn(T, R_half, device=device, dtype=torch.float32)

    y = torch.ops.autokernel.fused_rope_gate_mul(
        b, h_tilde, freqs_cos, freqs_sin, T, D, R_half
    )
    # Check grad_fn populated (not None)
    assert y.grad_fn is not None, "Output has no grad_fn — silent-freeze!"
    print(f"  y.grad_fn = {type(y.grad_fn).__name__}")

    loss = y.float().sum()
    loss.backward()

    assert b.grad is not None, "b.grad is None — gradient severed!"
    assert h_tilde.grad is not None, "h_tilde.grad is None — gradient severed!"
    print(f"  b.grad norm = {b.grad.norm().item():.4f}")
    print(f"  h_tilde.grad norm = {h_tilde.grad.norm().item():.4f}")
    assert b.grad.norm().item() > 0, "b.grad is all zeros"
    assert h_tilde.grad.norm().item() > 0, "h_tilde.grad is all zeros"
    print("  [PASS] gradient flow restored (no silent-freeze)")


def test_backward_parity():
    """Custom-op backward matches pure PyTorch autograd reference."""
    device = "cuda"
    torch.manual_seed(42)
    B, T, D = 2, 32, 128
    R_half = 16

    # Reference: PyTorch autograd on the native function
    b_ref = torch.randn(B * T, D, device=device, dtype=torch.float16, requires_grad=True)
    h_ref = torch.randn(B * T, D, device=device, dtype=torch.float16, requires_grad=True)
    freqs_cos = torch.randn(T, R_half, device=device, dtype=torch.float32)
    freqs_sin = torch.randn(T, R_half, device=device, dtype=torch.float32)

    y_ref = _native_fused_rope_gate_mul(b_ref, h_ref, freqs_cos, freqs_sin, T, D, R_half)
    y_ref.float().sum().backward()

    b_hip = b_ref.detach().clone().requires_grad_(True)
    h_hip = h_ref.detach().clone().requires_grad_(True)
    y_hip = torch.ops.autokernel.fused_rope_gate_mul(
        b_hip, h_hip, freqs_cos, freqs_sin, T, D, R_half
    )
    y_hip.float().sum().backward()

    b_rel = (b_hip.grad.float() - b_ref.grad.float()).abs().max().item()
    h_rel = (h_hip.grad.float() - h_ref.grad.float()).abs().max().item()
    b_max = max(b_hip.grad.float().abs().max().item(),
                b_ref.grad.float().abs().max().item())
    h_max = max(h_hip.grad.float().abs().max().item(),
                h_ref.grad.float().abs().max().item())
    print(f"  grad_b  rel_err={b_rel / max(b_max, 1e-6):.3e}")
    print(f"  grad_h  rel_err={h_rel / max(h_max, 1e-6):.3e}")
    assert b_rel / max(b_max, 1e-6) < 5e-3
    assert h_rel / max(h_max, 1e-6) < 5e-3
    print("  [PASS] backward parity within fp16 tolerance")


if __name__ == "__main__":
    print("Test 1: forward parity")
    test_forward_parity()
    print("\nTest 2: backward gradient flow")
    test_backward_gradflow()
    print("\nTest 3: backward numerical parity")
    test_backward_parity()
    print("\n[PASS] T-3.2 fused_rope_gate_mul custom_op with register_autograd")
