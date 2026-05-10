"""v3 T-3.3 Tier 2 parity tests.

Verifies that 4 training-path custom ops produce numerically-correct
forward + backward outputs under:
  (a) vanilla fp16 inputs
  (b) torch.autocast("cuda", fp16)

This matters because v3 hypothesis H11 attributes Phase C/G divergence
(in --optimize-kernels stacks) to custom-op autocast-boundary dtype
mismatches. If Tier 2 passes under autocast, H11 is weakened; if it
fails, we have a reproduction.

Ops tested:
  1. autokernel::silu_gate_mul         (SwiGLU body)
  2. autokernel::rmsnorm                (RMSNorm)
  3. autokernel::fused_res_rmsnorm      (residual + RMSNorm, dual output)
  4. autokernel::causal_conv1d          (HyPE conv block — T-3.2 shim)

Tolerance: 5e-3 rel_err (fp16).

Shapes: production-similar where relevant.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trigger registration
import kernels.hip._torch_ops  # noqa: F401


DEVICE = "cuda"
RTOL = 5e-3  # fp16 tolerance


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.float() - b.float()).abs().max().item()
    den = max(a.float().abs().max().item(), b.float().abs().max().item(), 1e-6)
    return num / den


# --------------------------------------------------------------------------- #
# silu_gate_mul
# --------------------------------------------------------------------------- #

def _silu_gate_mul_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def _test_silu_gate_mul(tag: str, under_autocast: bool):
    torch.manual_seed(42)
    shape = (16 * 512, 2048)
    gate = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    up = torch.randn(shape, device=DEVICE, dtype=torch.float16)

    # Reference
    gate_r = gate.detach().clone().requires_grad_(True)
    up_r = up.detach().clone().requires_grad_(True)
    if under_autocast:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            y_ref = _silu_gate_mul_ref(gate_r, up_r)
    else:
        y_ref = _silu_gate_mul_ref(gate_r, up_r)
    y_ref.float().sum().backward()

    # Custom-op
    gate_h = gate.detach().clone().requires_grad_(True)
    up_h = up.detach().clone().requires_grad_(True)
    if under_autocast:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            y_hip = torch.ops.autokernel.silu_gate_mul(gate_h, up_h)
    else:
        y_hip = torch.ops.autokernel.silu_gate_mul(gate_h, up_h)
    y_hip.float().sum().backward()

    f_rel = _rel(y_hip, y_ref)
    g_rel = _rel(gate_h.grad, gate_r.grad)
    u_rel = _rel(up_h.grad, up_r.grad)
    print(f"  [{tag}] silu_gate_mul: fwd={f_rel:.2e}, grad_gate={g_rel:.2e}, grad_up={u_rel:.2e}")
    assert f_rel < RTOL, f"[{tag}] silu_gate_mul forward mismatch"
    assert g_rel < RTOL, f"[{tag}] silu_gate_mul grad_gate mismatch"
    assert u_rel < RTOL, f"[{tag}] silu_gate_mul grad_up mismatch"


# --------------------------------------------------------------------------- #
# rmsnorm
# --------------------------------------------------------------------------- #

def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x.float() * rms * weight.float()).to(x.dtype)


def _test_rmsnorm(tag: str, under_autocast: bool):
    torch.manual_seed(42)
    B_T, D = 16 * 512, 768
    x = torch.randn(B_T, D, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(D, device=DEVICE, dtype=torch.float16)

    # Reference
    x_r = x.detach().clone().requires_grad_(True)
    w_r = weight.detach().clone().requires_grad_(True)
    if under_autocast:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            y_ref = _rmsnorm_ref(x_r, w_r)
    else:
        y_ref = _rmsnorm_ref(x_r, w_r)
    y_ref.float().sum().backward()

    # Custom-op
    x_h = x.detach().clone().requires_grad_(True)
    w_h = weight.detach().clone().requires_grad_(True)
    if under_autocast:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            y_hip = torch.ops.autokernel.rmsnorm(x_h, w_h)
    else:
        y_hip = torch.ops.autokernel.rmsnorm(x_h, w_h)
    y_hip.float().sum().backward()

    f_rel = _rel(y_hip, y_ref)
    x_rel = _rel(x_h.grad, x_r.grad)
    w_rel = _rel(w_h.grad, w_r.grad)
    print(f"  [{tag}] rmsnorm: fwd={f_rel:.2e}, grad_x={x_rel:.2e}, grad_w={w_rel:.2e}")
    # RMSNorm grad_w can accumulate large magnitude differences over big reductions;
    # use a slightly looser tolerance for weight grad.
    assert f_rel < RTOL, f"[{tag}] rmsnorm forward mismatch"
    assert x_rel < RTOL, f"[{tag}] rmsnorm grad_x mismatch"
    assert w_rel < 1e-2, f"[{tag}] rmsnorm grad_w mismatch (>1%)"


# --------------------------------------------------------------------------- #
# fused_res_rmsnorm
# --------------------------------------------------------------------------- #

def _fused_res_rmsnorm_ref(x, residual, weight, eps: float = 1e-6):
    hidden = x + residual
    rms = hidden.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    normed = (hidden.float() * rms * weight.float()).to(x.dtype)
    return hidden, normed


def _test_fused_res_rmsnorm(tag: str, under_autocast: bool):
    torch.manual_seed(42)
    B_T, D = 16 * 512, 768
    x = torch.randn(B_T, D, device=DEVICE, dtype=torch.float16)
    residual = torch.randn(B_T, D, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(D, device=DEVICE, dtype=torch.float16)

    # Reference
    x_r = x.detach().clone().requires_grad_(True)
    r_r = residual.detach().clone().requires_grad_(True)
    w_r = weight.detach().clone().requires_grad_(True)
    if under_autocast:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            hidden_r, normed_r = _fused_res_rmsnorm_ref(x_r, r_r, w_r)
    else:
        hidden_r, normed_r = _fused_res_rmsnorm_ref(x_r, r_r, w_r)
    # downstream consumes both outputs, sum over both
    (hidden_r.float().sum() + normed_r.float().sum()).backward()

    # Custom-op
    x_h = x.detach().clone().requires_grad_(True)
    r_h = residual.detach().clone().requires_grad_(True)
    w_h = weight.detach().clone().requires_grad_(True)
    if under_autocast:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            hidden_h, normed_h = torch.ops.autokernel.fused_res_rmsnorm(x_h, r_h, w_h)
    else:
        hidden_h, normed_h = torch.ops.autokernel.fused_res_rmsnorm(x_h, r_h, w_h)
    (hidden_h.float().sum() + normed_h.float().sum()).backward()

    h_rel = _rel(hidden_h, hidden_r)
    n_rel = _rel(normed_h, normed_r)
    gx_rel = _rel(x_h.grad, x_r.grad)
    # Note: the HIP backward returns the same grad for x and residual (they add);
    # the native reference autograd also produces identical grads since grad_hidden
    # flows to both. We just check x.grad is in parity.
    gr_rel = _rel(r_h.grad, r_r.grad)
    gw_rel = _rel(w_h.grad, w_r.grad)
    print(f"  [{tag}] fused_res_rmsnorm: hidden={h_rel:.2e}, normed={n_rel:.2e}, "
          f"grad_x={gx_rel:.2e}, grad_r={gr_rel:.2e}, grad_w={gw_rel:.2e}")
    assert h_rel < RTOL, f"[{tag}] fused_res_rmsnorm hidden mismatch"
    assert n_rel < RTOL, f"[{tag}] fused_res_rmsnorm normed mismatch"
    assert gx_rel < RTOL, f"[{tag}] fused_res_rmsnorm grad_x mismatch"
    assert gr_rel < RTOL, f"[{tag}] fused_res_rmsnorm grad_r mismatch"
    assert gw_rel < 1e-2, f"[{tag}] fused_res_rmsnorm grad_w mismatch"


# --------------------------------------------------------------------------- #
# causal_conv1d (shim — requires A.1 landed)
# --------------------------------------------------------------------------- #

def _causal_conv1d_ref(x, weight, bias):
    D, K = weight.shape
    x_padded = F.pad(x, (K - 1, 0))
    return F.conv1d(x_padded, weight.unsqueeze(1), bias, groups=D)


def _test_causal_conv1d(tag: str, under_autocast: bool):
    torch.manual_seed(42)
    B, D, T, K = 4, 256, 64, 4  # smaller shape to keep test fast
    x = torch.randn(B, D, T, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(D, K, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(D, device=DEVICE, dtype=torch.float16)

    x_r = x.detach().clone().requires_grad_(True)
    w_r = weight.detach().clone().requires_grad_(True)
    b_r = bias.detach().clone().requires_grad_(True)
    if under_autocast:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            y_ref = _causal_conv1d_ref(x_r, w_r, b_r)
    else:
        y_ref = _causal_conv1d_ref(x_r, w_r, b_r)
    y_ref.float().sum().backward()

    x_h = x.detach().clone().requires_grad_(True)
    w_h = weight.detach().clone().requires_grad_(True)
    b_h = bias.detach().clone().requires_grad_(True)
    if under_autocast:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            y_hip = torch.ops.autokernel.causal_conv1d(x_h, w_h, b_h)
    else:
        y_hip = torch.ops.autokernel.causal_conv1d(x_h, w_h, b_h)
    y_hip.float().sum().backward()

    f_rel = _rel(y_hip, y_ref)
    gx_rel = _rel(x_h.grad, x_r.grad)
    gw_rel = _rel(w_h.grad, w_r.grad)
    gb_rel = _rel(b_h.grad, b_r.grad)
    print(f"  [{tag}] causal_conv1d: fwd={f_rel:.2e}, grad_x={gx_rel:.2e}, "
          f"grad_w={gw_rel:.2e}, grad_b={gb_rel:.2e}")
    assert f_rel < RTOL, f"[{tag}] causal_conv1d forward mismatch"
    assert gx_rel < RTOL, f"[{tag}] causal_conv1d grad_x mismatch"
    assert gw_rel < RTOL, f"[{tag}] causal_conv1d grad_w mismatch"
    assert gb_rel < RTOL, f"[{tag}] causal_conv1d grad_b mismatch"


# --------------------------------------------------------------------------- #
# Main: run 4 ops × 2 autocast flavors = 8 test cells
# --------------------------------------------------------------------------- #

def run_all():
    print("=" * 70)
    print("Tier 2 parity — 4 ops × {vanilla, autocast} = 8 cells")
    print("=" * 70)

    print("\n--- Vanilla fp16 ---")
    _test_silu_gate_mul("vanilla", under_autocast=False)
    _test_rmsnorm("vanilla", under_autocast=False)
    _test_fused_res_rmsnorm("vanilla", under_autocast=False)
    _test_causal_conv1d("vanilla", under_autocast=False)

    print("\n--- Under torch.autocast(cuda, fp16) ---")
    _test_silu_gate_mul("autocast", under_autocast=True)
    _test_rmsnorm("autocast", under_autocast=True)
    _test_fused_res_rmsnorm("autocast", under_autocast=True)
    _test_causal_conv1d("autocast", under_autocast=True)

    print("\n[PASS] T-3.3 Tier 2 parity: 8/8 cells")


if __name__ == "__main__":
    run_all()
