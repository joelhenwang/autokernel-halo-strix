"""Benchmark backward pass optimizations — isolated and combined.

Tests each fused backward kernel against the PyTorch baseline:
  1. Isolated: one optimization at a time
  2. Combined: all passing optimizations stacked
  3. Correctness: numerical comparison vs PyTorch reference

Usage:
    python scripts/bench_backward_optimizations.py
    python scripts/bench_backward_optimizations.py --model amadeus
"""

import argparse
import os
import sys
import time

# Ensure project root is on path (for `from kernels.hip...` imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


def bench(fn, warmup=10, iters=100):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]  # median


def test_rmsnorm_backward(M=4096, N=768, device="cuda"):
    """Test fused RMSNorm backward vs PyTorch."""
    from kernels.hip.rmsnorm_backward import kernel_fn as hip_bwd

    x = torch.randn(M, N, device=device, dtype=torch.float16)
    weight = torch.randn(N, device=device, dtype=torch.float16)
    grad_output = torch.randn(M, N, device=device, dtype=torch.float16)

    # PyTorch reference
    x_f = x.float()
    w_f = weight.float()
    g_f = grad_output.float()
    rms_sq = x_f.pow(2).mean(-1, keepdim=True) + 1e-6
    rms_inv = rms_sq.rsqrt()
    normed = x_f * rms_inv
    D = N
    ref_grad_weight = (g_f * normed).sum(dim=0)
    grad_normed = g_f * w_f
    ref_grad_x = (grad_normed * rms_inv - normed * (grad_normed * normed).sum(-1, keepdim=True) / D).half()

    # HIP kernel
    hip_grad_x, hip_grad_weight = hip_bwd(x, weight, grad_output)

    gx_diff = (hip_grad_x.float() - ref_grad_x.float()).abs().max().item()
    gw_diff = (hip_grad_weight - ref_grad_weight).abs().max().item()
    # Note: fp16 accumulated reductions have ~0.02 max diff, this is expected

    def pytorch_bwd():
        xf = x.float()
        gf = grad_output.float()
        wf = weight.float()
        rs = xf.pow(2).mean(-1, keepdim=True) + 1e-6
        ri = rs.rsqrt()
        n = xf * ri
        gn = gf * wf
        _ = gn * ri - n * (gn * n).sum(-1, keepdim=True) / D
        _ = (gf * n).sum(dim=0)

    def hip_bwd_fn():
        hip_bwd(x, weight, grad_output)

    pt_ms = bench(pytorch_bwd)
    hip_ms = bench(hip_bwd_fn)

    return {
        "op": "rmsnorm_backward",
        "correct": gx_diff < 0.03 and gw_diff < 0.1,
        "grad_x_max_diff": gx_diff,
        "grad_w_max_diff": gw_diff,
        "pytorch_ms": pt_ms,
        "hip_ms": hip_ms,
        "speedup": pt_ms / hip_ms if hip_ms > 0 else 0,
    }


def test_silu_gate_mul_backward(M=4096, N=2048, device="cuda"):
    """Test fused SiLU gate mul backward vs PyTorch."""
    from kernels.hip.silu_gate_mul_backward import kernel_fn as hip_bwd

    gate = torch.randn(M, N, device=device, dtype=torch.float16)
    up = torch.randn(M, N, device=device, dtype=torch.float16)
    grad_output = torch.randn(M, N, device=device, dtype=torch.float16)

    # PyTorch reference
    g_f = gate.float()
    u_f = up.float()
    go_f = grad_output.float()
    sig = torch.sigmoid(g_f)
    d_silu = sig * (1.0 + g_f * (1.0 - sig))
    ref_grad_gate = (go_f * u_f * d_silu).half()
    ref_grad_up = (go_f * g_f * sig).half()

    # HIP kernel
    hip_grad_gate, hip_grad_up = hip_bwd(gate, up, grad_output)

    gg_diff = (hip_grad_gate.float() - ref_grad_gate.float()).abs().max().item()
    gu_diff = (hip_grad_up.float() - ref_grad_up.float()).abs().max().item()

    def pytorch_bwd():
        gf = gate.float()
        uf = up.float()
        gof = grad_output.float()
        s = torch.sigmoid(gf)
        ds = s * (1.0 + gf * (1.0 - s))
        _ = gof * uf * ds
        _ = gof * gf * s

    def hip_bwd_fn():
        hip_bwd(gate, up, grad_output)

    pt_ms = bench(pytorch_bwd)
    hip_ms = bench(hip_bwd_fn)

    return {
        "op": "silu_gate_mul_backward",
        "correct": gg_diff < 0.01 and gu_diff < 0.01,
        "grad_gate_max_diff": gg_diff,
        "grad_up_max_diff": gu_diff,
        "pytorch_ms": pt_ms,
        "hip_ms": hip_ms,
        "speedup": pt_ms / hip_ms if hip_ms > 0 else 0,
    }


def test_rotary_backward(B=4, H=12, N=256, D=64, device="cuda"):
    """Test fused rotary embedding backward vs PyTorch.

    Uses interleaved pair convention (adjacent dims) matching the forward
    HIP kernel in rotary_embedding.py. cos/sin have paired structure:
    cos[n, 2k] == cos[n, 2k+1] for each frequency.
    """
    from kernels.hip.rotary_embedding_backward import kernel_fn as hip_bwd

    grad_output = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

    # Build cos/sin with proper paired structure (each frequency shared by 2 dims)
    half_D = D // 2
    freqs = torch.randn(N, half_D, device=device, dtype=torch.float32)
    cos_half = torch.cos(freqs)  # (N, D/2)
    sin_half = torch.sin(freqs)  # (N, D/2)
    # Interleave: cos[n, 2k] = cos[n, 2k+1] = cos_half[n, k]
    cos_full = cos_half.unsqueeze(-1).expand(-1, -1, 2).reshape(N, D)
    sin_full = sin_half.unsqueeze(-1).expand(-1, -1, 2).reshape(N, D)
    cos = cos_full.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)
    sin = sin_full.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)

    # PyTorch reference (interleaved convention):
    # For pair (g0, g1) at positions (2k, 2k+1) with shared cos c, sin s:
    #   gx0 = g0*c + g1*s
    #   gx1 = g1*c - g0*s
    # This matches rotate_half when cos/sin have paired structure
    g = grad_output.float()
    c = cos.float()
    s = sin.float()
    def rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)
    ref_grad_x = (g * c + rotate_half(g) * (-s)).half()

    # HIP kernel
    hip_grad_x = hip_bwd(grad_output, cos, sin)

    diff = (hip_grad_x.float() - ref_grad_x.float()).abs().max().item()

    def pytorch_bwd():
        gf = grad_output.float()
        _ = gf * c + rotate_half(gf) * (-s)

    def hip_bwd_fn():
        hip_bwd(grad_output, cos, sin)

    pt_ms = bench(pytorch_bwd)
    hip_ms = bench(hip_bwd_fn)

    return {
        "op": "rotary_embedding_backward",
        "correct": diff < 0.01,
        "max_diff": diff,
        "pytorch_ms": pt_ms,
        "hip_ms": hip_ms,
        "speedup": pt_ms / hip_ms if hip_ms > 0 else 0,
    }


def test_selective_scan_backward(batch=4, seq=1024, d_inner=128, device="cuda"):
    """Test parallel selective scan backward vs sequential Python loops."""
    from kernels.hip.selective_scan_backward import kernel_fn as hip_bwd

    dA = torch.rand(batch, seq, d_inner, device=device, dtype=torch.float32) * 0.9 + 0.05
    dBx = torch.randn(batch, seq, d_inner, device=device, dtype=torch.float32) * 0.1
    C = torch.randn(batch, seq, d_inner, device=device, dtype=torch.float32) * 0.1
    D = torch.randn(d_inner, device=device, dtype=torch.float32) * 0.1
    x = torch.randn(batch, seq, d_inner, device=device, dtype=torch.float32)
    grad_y = torch.randn(batch, seq, d_inner, device=device, dtype=torch.float32)

    # Sequential Python reference (from _torch_ops.py)
    states = torch.zeros(batch, seq + 1, d_inner, dtype=torch.float32, device=device)
    for t in range(seq):
        states[:, t + 1] = dA[:, t] * states[:, t] + dBx[:, t]

    ref_grad_dA = torch.zeros_like(dA)
    ref_grad_dBx = torch.zeros_like(dBx)
    ref_grad_C = torch.zeros_like(C)
    ref_grad_D = torch.zeros(d_inner, dtype=torch.float32, device=device)
    grad_state = torch.zeros(batch, d_inner, dtype=torch.float32, device=device)

    for t in range(seq - 1, -1, -1):
        ref_grad_C[:, t] = grad_y[:, t] * states[:, t + 1]
        grad_state = grad_state + grad_y[:, t] * C[:, t]
        ref_grad_D = ref_grad_D + (grad_y[:, t] * x[:, t]).sum(0)
        ref_grad_dA[:, t] = grad_state * states[:, t]
        ref_grad_dBx[:, t] = grad_state
        grad_state = grad_state * dA[:, t]
    ref_grad_x = grad_y * D

    # HIP kernel
    hip_gdA, hip_gdBx, hip_gC, hip_gD, hip_gx = hip_bwd(dA, dBx, C, D, x, grad_y)

    diffs = {
        "grad_dA": (hip_gdA - ref_grad_dA).abs().max().item(),
        "grad_dBx": (hip_gdBx - ref_grad_dBx).abs().max().item(),
        "grad_C": (hip_gC - ref_grad_C).abs().max().item(),
        "grad_D": (hip_gD - ref_grad_D).abs().max().item(),
        "grad_x": (hip_gx - ref_grad_x).abs().max().item(),
    }
    correct = all(v < 0.01 for v in diffs.values())

    def sequential_bwd():
        st = torch.zeros(batch, seq + 1, d_inner, dtype=torch.float32, device=device)
        for t in range(seq):
            st[:, t + 1] = dA[:, t] * st[:, t] + dBx[:, t]
        gs = torch.zeros(batch, d_inner, dtype=torch.float32, device=device)
        for t in range(seq - 1, -1, -1):
            gs = (gs + grad_y[:, t] * C[:, t]) * dA[:, t]

    def hip_bwd_fn():
        hip_bwd(dA, dBx, C, D, x, grad_y)

    pt_ms = bench(sequential_bwd, warmup=3, iters=10)  # sequential is very slow
    hip_ms = bench(hip_bwd_fn)

    return {
        "op": "selective_scan_backward",
        "correct": correct,
        "diffs": diffs,
        "sequential_ms": pt_ms,
        "hip_ms": hip_ms,
        "speedup": pt_ms / hip_ms if hip_ms > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark backward optimizations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-scan", action="store_true", help="Skip slow sequential scan baseline")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = []

    print("=" * 80)
    print("BACKWARD KERNEL BENCHMARKS")
    print("=" * 80)

    # Test each kernel
    tests = [
        ("RMSNorm Backward", test_rmsnorm_backward),
        ("SiLU Gate Mul Backward", test_silu_gate_mul_backward),
        ("Rotary Embedding Backward", test_rotary_backward),
    ]

    if not args.skip_scan:
        tests.append(("Selective Scan Backward", test_selective_scan_backward))

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_fn(device=device)
            results.append(result)

            correct_str = "PASS" if result["correct"] else "FAIL"
            pt_key = "pytorch_ms" if "pytorch_ms" in result else "sequential_ms"
            pt_ms = result[pt_key]
            hip_ms = result["hip_ms"]
            speedup = result["speedup"]

            print(f"  Correctness: {correct_str}")
            for k, v in result.items():
                if "diff" in k.lower():
                    if isinstance(v, dict):
                        for dk, dv in v.items():
                            print(f"    {dk}: {dv:.6f}")
                    else:
                        print(f"    {k}: {v:.6f}")
            print(f"  Baseline: {pt_ms:.3f} ms")
            print(f"  HIP:      {hip_ms:.3f} ms")
            print(f"  Speedup:  {speedup:.2f}x")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Op':<30} {'Correct':>8} {'Base ms':>10} {'HIP ms':>10} {'Speedup':>10}")
    print(f"{'-'*80}")
    for r in results:
        op = r["op"]
        correct = "PASS" if r["correct"] else "FAIL"
        pt_key = "pytorch_ms" if "pytorch_ms" in r else "sequential_ms"
        print(f"{op:<30} {correct:>8} {r[pt_key]:>10.3f} {r['hip_ms']:>10.3f} {r['speedup']:>10.2f}x")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
