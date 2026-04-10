#!/usr/bin/env python3
"""Benchmark ChunkedLinearCrossEntropyLoss vs standard Linear + CE.

Measures: correctness (loss + gradients), speed, memory.

Usage:
    python scripts/bench_chunked_ce.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda"
WARMUP = 5
ITERS = 20


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


def main():
    from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss

    D = 1024
    V = 50257

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: D={D}, V={V}")
    print()

    for BT_label, BT in [("B=16,T=256", 4096), ("B=16,T=1024", 16384)]:
        print("=" * 70)
        print(f"  {BT_label} (BT={BT})")
        print("=" * 70)

        weight = torch.randn(V, D, device=DEVICE, dtype=torch.float16)
        targets = torch.randint(0, V, (BT,), device=DEVICE)

        # ---- Correctness ----
        h_std = torch.randn(BT, D, device=DEVICE, dtype=torch.float16, requires_grad=True)
        logits = F.linear(h_std.float(), weight.float())
        std_loss = F.cross_entropy(logits, targets)
        std_loss.backward()
        std_grad = h_std.grad.clone()

        for chunk_size in [512, 1024, 2048]:
            chunked_ce = ChunkedLinearCrossEntropyLoss(chunk_size=chunk_size)

            h_chk = h_std.detach().clone().requires_grad_(True)
            chk_loss = chunked_ce(h_chk, weight, targets)
            chk_loss.backward()
            chk_grad = h_chk.grad.clone()

            loss_diff = abs(std_loss.item() - chk_loss.item())
            grad_diff = (std_grad.float() - chk_grad.float()).abs().max().item()
            grad_cos = F.cosine_similarity(
                std_grad.float().reshape(1, -1),
                chk_grad.float().reshape(1, -1),
            ).item()
            print(f"  chunk={chunk_size}: loss_diff={loss_diff:.6f}, "
                  f"grad_max_diff={grad_diff:.6f}, grad_cos={grad_cos:.6f}")

        print()

        # ---- Speed (chunk_size=1024) ----
        chunked_ce = ChunkedLinearCrossEntropyLoss(chunk_size=1024)

        def standard_fwd():
            h = torch.randn(BT, D, device=DEVICE, dtype=torch.float16)
            logits = F.linear(h.float(), weight.float())
            return F.cross_entropy(logits, targets)

        def chunked_fwd():
            h = torch.randn(BT, D, device=DEVICE, dtype=torch.float16)
            return chunked_ce(h, weight, targets)

        def standard_fwd_bwd():
            h = torch.randn(BT, D, device=DEVICE, dtype=torch.float16, requires_grad=True)
            logits = F.linear(h.float(), weight.float())
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            return loss

        def chunked_fwd_bwd():
            h = torch.randn(BT, D, device=DEVICE, dtype=torch.float16, requires_grad=True)
            loss = chunked_ce(h, weight, targets)
            loss.backward()
            return loss

        std_fwd = bench(standard_fwd)
        chk_fwd = bench(chunked_fwd)
        std_bwd = bench(standard_fwd_bwd)
        chk_bwd = bench(chunked_fwd_bwd)

        # ---- Memory ----
        torch.cuda.reset_peak_memory_stats()
        standard_fwd_bwd()
        torch.cuda.synchronize()
        std_mem = torch.cuda.max_memory_allocated() / 1e9

        torch.cuda.reset_peak_memory_stats()
        chunked_fwd_bwd()
        torch.cuda.synchronize()
        chk_mem = torch.cuda.max_memory_allocated() / 1e9

        logits_size_gb = BT * V * 2 / 1e9  # fp16

        print(f"  Logits tensor size: {logits_size_gb:.3f} GB (eliminated by chunking)")
        print()
        print(f"  {'Method':<30} {'Fwd (ms)':>10} {'Fwd+Bwd (ms)':>14} {'Peak Mem':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*14} {'-'*10}")
        print(f"  {'Standard (Linear + CE)':<30} {std_fwd:>10.2f} {std_bwd:>14.2f} {std_mem:>9.2f}GB")
        print(f"  {'Chunked (chunk=1024)':<30} {chk_fwd:>10.2f} {chk_bwd:>14.2f} {chk_mem:>9.2f}GB")
        fwd_ratio = std_fwd / chk_fwd if chk_fwd > 0 else 0
        bwd_ratio = std_bwd / chk_bwd if chk_bwd > 0 else 0
        mem_saved = std_mem - chk_mem
        print(f"  {'Ratio':<30} {fwd_ratio:>10.2f}x {bwd_ratio:>13.2f}x  {mem_saved:>+8.2f}GB")
        print()


if __name__ == "__main__":
    main()
