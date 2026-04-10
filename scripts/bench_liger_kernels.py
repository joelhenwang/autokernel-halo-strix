#!/usr/bin/env python3
"""Benchmark Liger-Kernel vs our autokernel HIP kernels.

Priority 1: FusedLinearCrossEntropyLoss (the big win — eliminates full logits tensor)
Priority 2: RMSNorm, SwiGLU, fused_add_rmsnorm, RoPE head-to-head

Usage:
    python scripts/bench_liger_kernels.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 10
ITERS = 50


def bench(fn, *args, warmup=WARMUP, iters=ITERS, label=""):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    return median


def bench_fused_linear_cross_entropy():
    """Compare Liger FusedLinearCE vs standard Linear + CE."""
    print("=" * 70)
    print("  FusedLinearCrossEntropyLoss (the big win)")
    print("=" * 70)

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    D = 1024
    V = 50257
    BT = 4096  # batch=16 * seq=256

    hidden = torch.randn(BT, D, device=DEVICE, dtype=torch.float16, requires_grad=True)
    lm_weight = torch.randn(V, D, device=DEVICE, dtype=torch.float16)
    targets = torch.randint(0, V, (BT,), device=DEVICE)

    # Standard approach
    def standard_fwd():
        h = hidden.detach().requires_grad_(True)
        logits = F.linear(h, lm_weight)  # (BT, V) = 4096*50257*2 = 393 MB!
        loss = F.cross_entropy(logits, targets)
        return loss

    def standard_fwd_bwd():
        h = hidden.detach().requires_grad_(True)
        logits = F.linear(h, lm_weight)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        return loss

    # Liger fused approach
    liger_ce = LigerFusedLinearCrossEntropyLoss()

    def liger_fwd():
        h = hidden.detach().requires_grad_(True)
        loss = liger_ce(h, lm_weight, targets)
        return loss

    def liger_fwd_bwd():
        h = hidden.detach().requires_grad_(True)
        loss = liger_ce(h, lm_weight, targets)
        loss.backward()
        return loss

    # Correctness check (forward only — backward crashes on gfx1151)
    torch.manual_seed(42)
    h1 = hidden.detach()
    logits = F.linear(h1, lm_weight)
    std_loss = F.cross_entropy(logits, targets)

    torch.manual_seed(42)
    h2 = hidden.detach()
    try:
        lig_loss = liger_ce(h2, lm_weight, targets)
        loss_diff = abs(std_loss.item() - lig_loss.item())
        print(f"  Correctness: loss_diff={loss_diff:.6f} (forward only)")
    except Exception as e:
        print(f"  Liger FusedLinearCE forward FAILED: {e}")
        print("  Skipping FusedLinearCE benchmark (not compatible with gfx1151)")
        return

    # Memory comparison
    torch.cuda.reset_peak_memory_stats()
    standard_fwd()
    torch.cuda.synchronize()
    std_mem = torch.cuda.max_memory_allocated() / 1e9

    torch.cuda.reset_peak_memory_stats()
    liger_fwd()
    torch.cuda.synchronize()
    lig_mem = torch.cuda.max_memory_allocated() / 1e9

    # Speed (forward only — backward has illegal memory access on gfx1151)
    std_fwd_ms = bench(standard_fwd, label="std_fwd")
    lig_fwd_ms = bench(liger_fwd, label="lig_fwd")

    # Try backward cautiously
    try:
        std_bwd_ms = bench(standard_fwd_bwd, label="std_fwd+bwd")
    except Exception:
        std_bwd_ms = float('nan')
    try:
        lig_bwd_ms = bench(liger_fwd_bwd, label="lig_fwd+bwd")
    except Exception:
        lig_bwd_ms = float('nan')
        print("  NOTE: Liger FusedLinearCE backward FAILED on gfx1151 (hipErrorIllegalAddress)")

    print(f"\n  Shape: hidden=({BT},{D}), vocab={V}")
    print(f"  {'Method':<30} {'Fwd (ms)':>10} {'Fwd+Bwd (ms)':>14} {'Peak Mem':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*14} {'-'*10}")
    print(f"  {'Standard (Linear + CE)':<30} {std_fwd_ms:>10.2f} {std_bwd_ms:>14.2f} {std_mem:>9.2f}GB")
    print(f"  {'Liger FusedLinearCE':<30} {lig_fwd_ms:>10.2f} {lig_bwd_ms:>14.2f} {lig_mem:>9.2f}GB")
    fwd_speedup = std_fwd_ms / lig_fwd_ms if lig_fwd_ms > 0 else 0
    print(f"  {'Speedup':<30} {fwd_speedup:>10.2f}x {'N/A':>14} {std_mem/lig_mem:>9.2f}x")
    print()


def bench_rmsnorm():
    """Compare Liger RMSNorm vs our HIP RMSNorm."""
    print("=" * 70)
    print("  RMSNorm: Liger (Triton) vs AutoKernel (HIP)")
    print("=" * 70)

    from liger_kernel.ops.rms_norm import LigerRMSNormFunction

    D = 1024
    M = 4096  # B*T

    x = torch.randn(M, D, device=DEVICE, dtype=torch.float16)
    w = torch.ones(D, device=DEVICE, dtype=torch.float16)

    # PyTorch baseline
    def pytorch_rmsnorm():
        rms = (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        return (x.float() * rms * w.float()).half()

    # Our HIP kernel
    try:
        from kernels.hip.rmsnorm import kernel_fn as hip_rmsnorm
        def hip_fn():
            return hip_rmsnorm(x, w)
        has_hip = True
    except Exception as e:
        print(f"  HIP RMSNorm not available: {e}")
        has_hip = False

    # Liger Triton
    def liger_fn():
        return LigerRMSNormFunction.apply(x, w, 1e-6)

    pt_ms = bench(pytorch_rmsnorm)
    lig_ms = bench(liger_fn)
    hip_ms = bench(hip_fn) if has_hip else float('inf')

    print(f"  Shape: ({M}, {D})")
    print(f"  {'PyTorch':<25} {pt_ms:>8.3f} ms")
    print(f"  {'Liger (Triton)':<25} {lig_ms:>8.3f} ms ({pt_ms/lig_ms:.1f}x)")
    if has_hip:
        print(f"  {'AutoKernel (HIP)':<25} {hip_ms:>8.3f} ms ({pt_ms/hip_ms:.1f}x)")
        print(f"  Winner: {'HIP' if hip_ms < lig_ms else 'Liger'} ({max(hip_ms,lig_ms)/min(hip_ms,lig_ms):.1f}x faster)")
    print()


def bench_swiglu():
    """Compare Liger SwiGLU vs our HIP silu_gate_mul."""
    print("=" * 70)
    print("  SwiGLU: Liger (Triton) vs AutoKernel (HIP)")
    print("=" * 70)

    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    D = 1024
    FFN = 2560
    M = 4096

    gate = torch.randn(M, FFN, device=DEVICE, dtype=torch.float16)
    up = torch.randn(M, FFN, device=DEVICE, dtype=torch.float16)

    # PyTorch
    def pytorch_fn():
        return F.silu(gate) * up

    # Our HIP
    try:
        from kernels.hip.silu_gate_mul import kernel_fn as hip_swiglu
        def hip_fn():
            return hip_swiglu(gate, up)
        has_hip = True
    except Exception as e:
        print(f"  HIP SwiGLU not available: {e}")
        has_hip = False

    # Liger
    def liger_fn():
        return LigerSiLUMulFunction.apply(gate, up)

    pt_ms = bench(pytorch_fn)
    lig_ms = bench(liger_fn)
    hip_ms = bench(hip_fn) if has_hip else float('inf')

    print(f"  Shape: ({M}, {FFN})")
    print(f"  {'PyTorch':<25} {pt_ms:>8.3f} ms")
    print(f"  {'Liger (Triton)':<25} {lig_ms:>8.3f} ms ({pt_ms/lig_ms:.1f}x)")
    if has_hip:
        print(f"  {'AutoKernel (HIP)':<25} {hip_ms:>8.3f} ms ({pt_ms/hip_ms:.1f}x)")
        print(f"  Winner: {'HIP' if hip_ms < lig_ms else 'Liger'} ({max(hip_ms,lig_ms)/min(hip_ms,lig_ms):.1f}x faster)")
    print()


def bench_cross_entropy():
    """Compare Liger CE vs our HIP cross_entropy."""
    print("=" * 70)
    print("  CrossEntropy: Liger (Triton) vs AutoKernel (HIP)")
    print("=" * 70)

    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

    V = 50257
    M = 4096

    logits = torch.randn(M, V, device=DEVICE, dtype=torch.float16)
    targets = torch.randint(0, V, (M,), device=DEVICE)

    # PyTorch
    def pytorch_fn():
        return F.cross_entropy(logits, targets)

    # Our HIP
    try:
        from kernels.hip.cross_entropy import kernel_fn as hip_ce
        def hip_fn():
            return hip_ce(logits, targets)
        has_hip = True
    except Exception as e:
        print(f"  HIP CE not available: {e}")
        has_hip = False

    # Liger
    def liger_fn():
        return LigerCrossEntropyFunction.apply(logits, targets, 0, 1.0, 0.0, -100)

    pt_ms = bench(pytorch_fn)
    lig_ms = bench(liger_fn)
    hip_ms = bench(hip_fn) if has_hip else float('inf')

    print(f"  Shape: ({M}, {V})")
    print(f"  {'PyTorch':<25} {pt_ms:>8.3f} ms")
    print(f"  {'Liger (Triton)':<25} {lig_ms:>8.3f} ms ({pt_ms/lig_ms:.1f}x)")
    if has_hip:
        print(f"  {'AutoKernel (HIP)':<25} {hip_ms:>8.3f} ms ({pt_ms/hip_ms:.1f}x)")
        print(f"  Winner: {'HIP' if hip_ms < lig_ms else 'Liger'}")
    print()


if __name__ == "__main__":
    import subprocess, sys

    benchmarks = ["fused_linear_cross_entropy", "rmsnorm", "swiglu", "cross_entropy"]

    # If called with a specific benchmark name, run just that one
    if len(sys.argv) > 1 and sys.argv[1] in benchmarks:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")
        print()
        fn = globals()[f"bench_{sys.argv[1]}"]
        fn()
        sys.exit(0)

    # Otherwise, run each benchmark in a separate process to isolate GPU crashes
    print(f"Running each benchmark in separate process (GPU crash isolation)")
    print()

    for name in benchmarks:
        print(f"--- Running: {name} ---")
        result = subprocess.run(
            [sys.executable, __file__, name],
            capture_output=False,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
        print()
