#!/usr/bin/env python3
"""Benchmark 3 fused Engram HIP kernel variants vs PyTorch baseline.

Variant A: Hash + Gather + Gate (fused_engram_hash_gate.py)
Variant B: Gate + Value + Conv (fused_engram_gate_conv.py)
Variant C: Full fusion (fused_engram_full.py)

Usage:
    python scripts/bench_engram_kernels.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 5
ITERS = 20

# Default Engram config
D_MODEL = 1024
D_ENGRAM = 512
N_HASH_HEADS = 8
NGRAM_SIZES = [2, 3]
TABLE_SIZE = 65536
CONV_K = 3
B = 16
T = 256


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def bench_pytorch_baseline():
    """Benchmark the full EngramLayer in pure PyTorch."""
    print("=" * 70)
    print("  PyTorch Baseline: Full EngramLayer")
    print("=" * 70)

    from models.engram import EngramLayer

    engram = EngramLayer(
        d_model=D_MODEL, d_engram=D_ENGRAM, n_hash_heads=N_HASH_HEADS,
        ngram_sizes=NGRAM_SIZES, table_size=TABLE_SIZE, conv_kernel=CONV_K,
    ).to(DEVICE).half()

    hidden = torch.randn(B, T, D_MODEL, device=DEVICE, dtype=torch.float16)
    input_ids = torch.randint(0, 50257, (B, T), device=DEVICE)

    def fwd():
        return engram(hidden, input_ids)

    out = fwd()
    print(f"  Output shape: {out.shape}, norm: {out.norm().item():.4f}")

    ms = bench(fwd)
    print(f"  Forward: {ms:.3f} ms")

    # Backward
    def fwd_bwd():
        h = hidden.detach().requires_grad_(True)
        out = engram(h, input_ids)
        out.sum().backward()

    bwd_ms = bench(fwd_bwd)
    print(f"  Fwd+Bwd: {bwd_ms:.3f} ms")
    print()
    return ms, bwd_ms


def bench_variant_b():
    """Benchmark Variant B: Gate + Value + Conv."""
    print("=" * 70)
    print("  Variant B: Gate + Value + Conv (fused_engram_gate_conv.py)")
    print("=" * 70)

    try:
        from kernels.hip.fused_engram_gate_conv import kernel_fn, reference_fn
    except Exception as e:
        print(f"  FAILED to import: {e}")
        return None, None

    M = B * T
    query = torch.randn(M, D_MODEL, device=DEVICE, dtype=torch.float16)
    key = torch.randn(M, D_MODEL, device=DEVICE, dtype=torch.float16)
    value = torch.randn(M, D_MODEL, device=DEVICE, dtype=torch.float16)
    conv_w = torch.randn(D_MODEL, CONV_K, device=DEVICE, dtype=torch.float16) * 0.02
    conv_b = torch.zeros(D_MODEL, device=DEVICE, dtype=torch.float16)

    # Correctness vs reference
    ref_out = reference_fn(query, key, value, conv_w, conv_b, T)

    try:
        hip_out = kernel_fn(query, key, value, conv_w, conv_b, T)
        max_diff = (ref_out - hip_out).abs().max().item()
        print(f"  Correctness: max_diff={max_diff:.6f} {'PASS' if max_diff < 0.1 else 'FAIL'}")
    except Exception as e:
        print(f"  HIP kernel FAILED: {e}")
        return None, None

    def fwd_hip():
        return kernel_fn(query, key, value, conv_w, conv_b, T)

    def fwd_ref():
        return reference_fn(query, key, value, conv_w, conv_b, T)

    hip_ms = bench(fwd_hip)
    ref_ms = bench(fwd_ref)
    print(f"  PyTorch reference: {ref_ms:.3f} ms")
    print(f"  HIP kernel:       {hip_ms:.3f} ms ({ref_ms/hip_ms:.1f}x)")
    print()
    return hip_ms, ref_ms


def bench_variant_a():
    """Benchmark Variant A: Hash + Gather + Gate."""
    print("=" * 70)
    print("  Variant A: Hash + Gather + Gate (fused_engram_hash_gate.py)")
    print("=" * 70)

    try:
        from kernels.hip.fused_engram_hash_gate import kernel_fn, reference_fn
    except Exception as e:
        print(f"  FAILED to import: {e}")
        return None, None

    M = B * T
    n_ngrams = len(NGRAM_SIZES)
    max_ngram = max(NGRAM_SIZES)
    n_total = n_ngrams * N_HASH_HEADS
    d_head = D_ENGRAM // n_total

    hidden = torch.randn(M, D_MODEL, device=DEVICE, dtype=torch.float16)
    input_ids = torch.randint(0, 50257, (M,), device=DEVICE, dtype=torch.long)
    prev_ids = torch.randint(0, 50257, (M, max_ngram - 1), device=DEVICE, dtype=torch.long)
    emb_weight = torch.randn(n_total * TABLE_SIZE, d_head, device=DEVICE, dtype=torch.float16) * 0.02
    key_proj_w = torch.randn(D_MODEL, D_ENGRAM, device=DEVICE, dtype=torch.float16) * 0.02
    norm_weight = torch.ones(D_MODEL, device=DEVICE, dtype=torch.float16)

    hash_mults = torch.randint(1, 2**16, (n_ngrams, N_HASH_HEADS, max_ngram),
                               device=DEVICE, dtype=torch.long)

    try:
        gate, embs = kernel_fn(
            hidden, input_ids, prev_ids, emb_weight, key_proj_w, norm_weight,
            hash_mults, TABLE_SIZE, N_HASH_HEADS, n_ngrams, max_ngram,
        )
        print(f"  Output: gate {gate.shape}, embs {embs.shape}")
        print(f"  Gate range: [{gate.min().item():.4f}, {gate.max().item():.4f}]")
    except Exception as e:
        print(f"  HIP kernel FAILED: {e}")
        return None, None

    # Correctness
    ref_gate, ref_embs = reference_fn(
        hidden, input_ids, prev_ids, emb_weight, key_proj_w, norm_weight,
        hash_mults, TABLE_SIZE, N_HASH_HEADS, n_ngrams, max_ngram,
    )
    gate_diff = (gate.float() - ref_gate.float()).abs().max().item()
    embs_diff = (embs.float() - ref_embs.float()).abs().max().item()
    print(f"  Correctness: gate_diff={gate_diff:.6f}, embs_diff={embs_diff:.6f}")

    def fwd_hip():
        return kernel_fn(
            hidden, input_ids, prev_ids, emb_weight, key_proj_w, norm_weight,
            hash_mults, TABLE_SIZE, N_HASH_HEADS, n_ngrams, max_ngram,
        )

    def fwd_ref():
        return reference_fn(
            hidden, input_ids, prev_ids, emb_weight, key_proj_w, norm_weight,
            hash_mults, TABLE_SIZE, N_HASH_HEADS, n_ngrams, max_ngram,
        )

    hip_ms = bench(fwd_hip)
    ref_ms = bench(fwd_ref)
    print(f"  PyTorch reference: {ref_ms:.3f} ms")
    print(f"  HIP kernel:       {hip_ms:.3f} ms ({ref_ms/hip_ms:.1f}x)")
    print()
    return hip_ms, ref_ms


def bench_variant_c():
    """Benchmark Variant C: Full Fusion."""
    print("=" * 70)
    print("  Variant C: Full Fusion (fused_engram_full.py)")
    print("=" * 70)

    try:
        from kernels.hip.fused_engram_full import kernel_fn, reference_fn
    except Exception as e:
        print(f"  FAILED to import: {e}")
        return None, None

    M = B * T
    n_ngrams = len(NGRAM_SIZES)
    max_ngram = max(NGRAM_SIZES)
    n_total = n_ngrams * N_HASH_HEADS
    d_head = D_ENGRAM // n_total

    hidden = torch.randn(M, D_MODEL, device=DEVICE, dtype=torch.float16)
    input_ids = torch.randint(0, 50257, (M,), device=DEVICE, dtype=torch.long)
    prev_ids = torch.randint(0, 50257, (M, max_ngram - 1), device=DEVICE, dtype=torch.long)
    emb_weight = torch.randn(n_total * TABLE_SIZE, d_head, device=DEVICE, dtype=torch.float16) * 0.02
    key_proj_w = torch.randn(D_MODEL, D_ENGRAM, device=DEVICE, dtype=torch.float16) * 0.02
    val_proj_w = torch.randn(D_MODEL, D_ENGRAM, device=DEVICE, dtype=torch.float16) * 0.02
    norm_weight = torch.ones(D_MODEL, device=DEVICE, dtype=torch.float16)
    conv_w = torch.randn(D_MODEL, CONV_K, device=DEVICE, dtype=torch.float16) * 0.02
    conv_b = torch.zeros(D_MODEL, device=DEVICE, dtype=torch.float16)
    hash_mults = torch.randint(1, 2**16, (n_ngrams, N_HASH_HEADS, max_ngram),
                               device=DEVICE, dtype=torch.long)

    try:
        out = kernel_fn(
            hidden, input_ids, prev_ids, emb_weight, key_proj_w, val_proj_w,
            norm_weight, conv_w, conv_b, hash_mults,
            TABLE_SIZE, N_HASH_HEADS, n_ngrams, max_ngram, T,
        )
        print(f"  Output shape: {out.shape}, norm: {out.norm().item():.4f}")
    except Exception as e:
        print(f"  HIP kernel FAILED: {e}")
        return None, None

    # Correctness
    ref_out = reference_fn(
        hidden, input_ids, prev_ids, emb_weight, key_proj_w, val_proj_w,
        norm_weight, conv_w, conv_b, hash_mults,
        TABLE_SIZE, N_HASH_HEADS, n_ngrams, max_ngram, T,
    )
    max_diff = (out.float() - ref_out.float()).abs().max().item()
    print(f"  Correctness: max_diff={max_diff:.6f} {'PASS' if max_diff < 0.5 else 'FAIL'}")

    def fwd_hip():
        return kernel_fn(
            hidden, input_ids, prev_ids, emb_weight, key_proj_w, val_proj_w,
            norm_weight, conv_w, conv_b, hash_mults,
            TABLE_SIZE, N_HASH_HEADS, n_ngrams, max_ngram, T,
        )

    def fwd_ref():
        return reference_fn(
            hidden, input_ids, prev_ids, emb_weight, key_proj_w, val_proj_w,
            norm_weight, conv_w, conv_b, hash_mults,
            TABLE_SIZE, N_HASH_HEADS, n_ngrams, max_ngram, T,
        )

    hip_ms = bench(fwd_hip)
    ref_ms = bench(fwd_ref)
    print(f"  PyTorch reference: {ref_ms:.3f} ms")
    print(f"  HIP kernel:       {hip_ms:.3f} ms ({ref_ms/hip_ms:.1f}x)")
    print()
    return hip_ms, ref_ms


if __name__ == "__main__":
    import subprocess

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: B={B}, T={T}, D={D_MODEL}, d_engram={D_ENGRAM}")
    print(f"        n_heads={N_HASH_HEADS}, ngrams={NGRAM_SIZES}, table={TABLE_SIZE}")
    print()

    # Run each in separate process to isolate GPU crashes
    benchmarks = ["baseline", "variant_b", "variant_a", "variant_c"]

    if len(sys.argv) > 1 and sys.argv[1] in benchmarks:
        fn = globals()[f"bench_{'pytorch_' if sys.argv[1] == 'baseline' else ''}{sys.argv[1]}"]
        fn()
        sys.exit(0)

    print("Running each benchmark in separate process (GPU crash isolation)\n")
    for name in benchmarks:
        print(f"--- {name} ---")
        result = subprocess.run(
            [sys.executable, __file__, name],
            capture_output=False,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})\n")
        print()
