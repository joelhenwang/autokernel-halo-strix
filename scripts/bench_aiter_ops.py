"""
Benchmark aiter HIP ops vs autokernel kernels on gfx1151.

Discovers available aiter ops via torch.ops.aiter.* and benchmarks each
against our autokernel equivalent and PyTorch baseline.

Usage:
    python scripts/bench_aiter_ops.py
    python scripts/bench_aiter_ops.py --test rmsnorm
    python scripts/bench_aiter_ops.py --test rope
    python scripts/bench_aiter_ops.py --test all
"""

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def timer(fn, warmup=5, iters=50, label=""):
    """Benchmark a function: warmup, then time `iters` iterations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / iters * 1000
    print(f"  {label}: {elapsed:.2f}ms")
    return elapsed


def check_close(a, b, name, atol=0.01, rtol=0.01):
    """Check two tensors are close, report max diff."""
    diff = (a.float() - b.float()).abs()
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_diff={diff.max().item():.6f}")
    return ok


# ============================================================================
# Discover available aiter ops
# ============================================================================

def discover_aiter_ops():
    """Try to import aiter and list available ops."""
    print("\n" + "=" * 60)
    print("DISCOVERING AITER OPS")
    print("=" * 60)

    try:
        import aiter
        print(f"  aiter version: {getattr(aiter, '__version__', 'unknown')}")
    except ImportError:
        print("  [SKIP] aiter not installed")
        return []

    # Check for aiter core module
    try:
        from aiter.jit import module_aiter_core
        print("  [OK] module_aiter_core loaded")
    except Exception as e:
        print(f"  [WARN] module_aiter_core: {e}")

    # List torch.ops.aiter.* if available
    ops = []
    try:
        aiter_ops = dir(torch.ops.aiter)
        ops = [op for op in aiter_ops if not op.startswith("_")]
        print(f"  Available ops ({len(ops)}): {', '.join(ops[:20])}")
        if len(ops) > 20:
            print(f"    ... and {len(ops) - 20} more")
    except Exception as e:
        print(f"  [WARN] Cannot enumerate torch.ops.aiter: {e}")

    return ops


# ============================================================================
# Test: RMSNorm — aiter vs autokernel vs PyTorch
# ============================================================================

def test_rmsnorm():
    print("\n" + "=" * 60)
    print("TEST: RMSNorm — aiter vs autokernel vs PyTorch")
    print("=" * 60)

    B, T, D = 8, 512, 1024
    x = torch.randn(B, T, D, dtype=torch.float16, device="cuda")
    weight = torch.ones(D, dtype=torch.float16, device="cuda")
    eps = 1e-6

    # PyTorch baseline
    def pytorch_rmsnorm():
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * norm * weight

    ref = pytorch_rmsnorm()
    t_pytorch = timer(pytorch_rmsnorm, label="PyTorch RMSNorm")

    # autokernel RMSNorm
    try:
        from kernels.hip.rmsnorm import kernel_fn as autokernel_rmsnorm
        ak_out = autokernel_rmsnorm(x, weight, eps)
        check_close(ak_out, ref, "autokernel vs PyTorch")
        t_ak = timer(lambda: autokernel_rmsnorm(x, weight, eps), label="autokernel RMSNorm")
    except Exception as e:
        print(f"  [SKIP] autokernel RMSNorm: {e}")
        t_ak = None

    # aiter RMSNorm
    try:
        from aiter.ops.norm import rms_norm as aiter_rmsnorm
        aiter_out = aiter_rmsnorm(x, weight, eps)
        check_close(aiter_out, ref, "aiter vs PyTorch")
        t_aiter = timer(lambda: aiter_rmsnorm(x, weight, eps), label="aiter RMSNorm")
    except Exception as e:
        print(f"  [SKIP] aiter RMSNorm: {e}")
        t_aiter = None

    # Summary
    print(f"\n  Summary: PyTorch={t_pytorch:.2f}ms", end="")
    if t_ak:
        print(f", autokernel={t_ak:.2f}ms ({t_pytorch/t_ak:.1f}x)", end="")
    if t_aiter:
        print(f", aiter={t_aiter:.2f}ms ({t_pytorch/t_aiter:.1f}x)", end="")
    print()


# ============================================================================
# Test: RoPE — aiter vs autokernel vs PyTorch
# ============================================================================

def test_rope():
    print("\n" + "=" * 60)
    print("TEST: RoPE (Rotary Embedding) — aiter vs autokernel vs PyTorch")
    print("=" * 60)

    B, T, H, D = 8, 512, 8, 128
    q = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")

    # Build cos/sin tables
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    t = torch.arange(T, device="cuda", dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_table = torch.cos(angles).half()
    sin_table = torch.sin(angles).half()

    # PyTorch baseline (complex)
    def pytorch_rope():
        q_ = torch.view_as_complex(q.float().reshape(B, T, H, -1, 2))
        freqs_cis = torch.polar(torch.ones_like(angles), angles)[None, :, None, :]
        return torch.view_as_real(q_ * freqs_cis).flatten(3).half()

    ref = pytorch_rope()
    t_pytorch = timer(pytorch_rope, label="PyTorch RoPE")

    # autokernel RoPE
    try:
        from kernels.hip.rotary_embedding import kernel_fn as autokernel_rope
        ak_out = autokernel_rope(q.reshape(B*T, H*D), cos_table, sin_table)
        check_close(ak_out.view(B, T, H, D), ref, "autokernel vs PyTorch", atol=0.05)
        t_ak = timer(
            lambda: autokernel_rope(q.reshape(B*T, H*D), cos_table, sin_table),
            label="autokernel RoPE"
        )
    except Exception as e:
        print(f"  [SKIP] autokernel RoPE: {e}")
        t_ak = None

    # aiter RoPE
    try:
        from aiter.ops.pos_encoding import rope as aiter_rope
        aiter_out = aiter_rope(q, cos_table.unsqueeze(0), sin_table.unsqueeze(0))
        check_close(aiter_out, ref, "aiter vs PyTorch", atol=0.05)
        t_aiter = timer(
            lambda: aiter_rope(q, cos_table.unsqueeze(0), sin_table.unsqueeze(0)),
            label="aiter RoPE"
        )
    except Exception as e:
        print(f"  [SKIP] aiter RoPE: {e}")
        t_aiter = None

    print(f"\n  Summary: PyTorch={t_pytorch:.2f}ms", end="")
    if t_ak:
        print(f", autokernel={t_ak:.2f}ms ({t_pytorch/t_ak:.1f}x)", end="")
    if t_aiter:
        print(f", aiter={t_aiter:.2f}ms ({t_pytorch/t_aiter:.1f}x)", end="")
    print()


# ============================================================================
# Test: Fused bias+activation — aiter vs autokernel vs PyTorch
# ============================================================================

def test_fused_bias_activation():
    print("\n" + "=" * 60)
    print("TEST: Fused bias+SiLU — aiter vs autokernel vs PyTorch")
    print("=" * 60)

    B, T, D = 8, 512, 2560
    x = torch.randn(B, T, D, dtype=torch.float16, device="cuda")
    bias = torch.randn(D, dtype=torch.float16, device="cuda")

    # PyTorch baseline
    def pytorch_bias_silu():
        return F.silu(x + bias)

    ref = pytorch_bias_silu()
    t_pytorch = timer(pytorch_bias_silu, label="PyTorch bias+SiLU")

    # autokernel fused_bias_silu
    try:
        from kernels.hip.fused_bias_silu import kernel_fn as autokernel_bias_silu
        ak_out = autokernel_bias_silu(x, bias)
        check_close(ak_out, ref, "autokernel vs PyTorch")
        t_ak = timer(lambda: autokernel_bias_silu(x, bias), label="autokernel fused_bias_silu")
    except Exception as e:
        print(f"  [SKIP] autokernel fused_bias_silu: {e}")
        t_ak = None

    # aiter fused activation (if available)
    try:
        from aiter.ops.activation import silu_and_mul as aiter_silu_mul
        # aiter's silu_and_mul may have different API — test cautiously
        aiter_input = torch.cat([x + bias, x + bias], dim=-1)  # gate + up format
        aiter_out = aiter_silu_mul(aiter_input)
        t_aiter = timer(
            lambda: aiter_silu_mul(aiter_input),
            label="aiter silu_and_mul"
        )
    except Exception as e:
        print(f"  [SKIP] aiter fused activation: {e}")
        t_aiter = None

    print(f"\n  Summary: PyTorch={t_pytorch:.2f}ms", end="")
    if t_ak:
        print(f", autokernel={t_ak:.2f}ms ({t_pytorch/t_ak:.1f}x)", end="")
    if t_aiter:
        print(f", aiter={t_aiter:.2f}ms", end="")
    print()


# ============================================================================
# Test: hipBLASLt env vars — Stream-K + epilogue fusion
# ============================================================================

def test_hipblaslt_envvars():
    print("\n" + "=" * 60)
    print("TEST: hipBLASLt env vars (Stream-K, epilogue fusion)")
    print("=" * 60)

    import os

    B, T, D, FFN = 8, 512, 1024, 2560
    x = torch.randn(B * T, D, dtype=torch.float16, device="cuda")
    w = torch.randn(FFN, D, dtype=torch.float16, device="cuda")

    # Baseline GEMM
    def gemm():
        return torch.mm(x, w.t())

    t_base = timer(gemm, label="Baseline GEMM (BT×D) × (FFN×D)^T")

    # Test Stream-K
    os.environ["TENSILE_SOLUTION_SELECTION_METHOD"] = "2"
    t_streamk = timer(gemm, warmup=10, label="Stream-K GEMM")
    del os.environ["TENSILE_SOLUTION_SELECTION_METHOD"]

    # Test hipBLASLt
    os.environ["ROCBLAS_USE_HIPBLASLT"] = "1"
    t_hipblaslt = timer(gemm, warmup=10, label="hipBLASLt GEMM")
    del os.environ["ROCBLAS_USE_HIPBLASLT"]

    # Test both
    os.environ["TENSILE_SOLUTION_SELECTION_METHOD"] = "2"
    os.environ["ROCBLAS_USE_HIPBLASLT"] = "1"
    t_both = timer(gemm, warmup=10, label="Stream-K + hipBLASLt GEMM")
    del os.environ["TENSILE_SOLUTION_SELECTION_METHOD"]
    del os.environ["ROCBLAS_USE_HIPBLASLT"]

    print(f"\n  Summary:")
    print(f"    Baseline:           {t_base:.2f}ms")
    print(f"    Stream-K:           {t_streamk:.2f}ms ({t_base/t_streamk:.2f}x)")
    print(f"    hipBLASLt:          {t_hipblaslt:.2f}ms ({t_base/t_hipblaslt:.2f}x)")
    print(f"    Stream-K+hipBLASLt: {t_both:.2f}ms ({t_base/t_both:.2f}x)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark aiter ops vs autokernel")
    parser.add_argument("--test", default="all",
                        choices=["discover", "rmsnorm", "rope", "activation",
                                 "hipblaslt", "all"],
                        help="Which test to run")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Run on Strix Halo machine.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    tests = {
        "discover": discover_aiter_ops,
        "rmsnorm": test_rmsnorm,
        "rope": test_rope,
        "activation": test_fused_bias_activation,
        "hipblaslt": test_hipblaslt_envvars,
    }

    if args.test == "all":
        discover_aiter_ops()
        for name, fn in tests.items():
            if name != "discover":
                try:
                    fn()
                except Exception as e:
                    print(f"\n  [ERROR] {name}: {e}")
    else:
        tests[args.test]()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
