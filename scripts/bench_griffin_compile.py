"""
Benchmark Tempest124M throughput: before vs after compile-optimized Griffin block.

Tests:
1. eager (fp32)
2. autokernel only (fp16 kernels)
3. autokernel + compile
4. Training mode (fp16 AMP + optimizer)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn as nn
import torch.cuda.amp as amp


def bench_model(label, model, B=8, T=256, vocab=32000, n_steps=30, warmup=5, training=False):
    """Benchmark model throughput."""
    x = torch.randint(0, vocab, (B, T), device="cuda")

    if training:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = amp.GradScaler()

    # Warmup
    for _ in range(warmup):
        if training:
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(dtype=torch.float16):
                logits = model(x)
                loss = logits.view(-1, logits.size(-1)).float().log_softmax(-1)[:, 0].mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            logits.sum().backward()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        if training:
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(dtype=torch.float16):
                logits = model(x)
                loss = logits.view(-1, logits.size(-1)).float().log_softmax(-1)[:, 0].mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            logits.sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tok_s = B * T * n_steps / elapsed
    ms_step = elapsed / n_steps * 1000
    print(f"  {label:30s}: {tok_s:>8,.0f} tok/s  ({ms_step:>6.1f} ms/step)")
    return tok_s


def main():
    from models.tempest_124m import Tempest124M
    import autokernel

    print("=" * 70)
    print("BENCHMARK: Compile-Optimized Griffin Block (Tempest124M)")
    print("=" * 70)

    B, T = 8, 256
    configs = [
        ("eager (fp32)", False, False, False),
        ("autokernel (fp16)", True, False, False),
        ("AK + compile", True, True, False),
        ("training (AMP + optimizer)", True, True, True),
    ]

    results = {}
    for label, use_ak, use_compile, training in configs:
        torch.cuda.empty_cache()
        model = Tempest124M()

        if use_ak:
            model = autokernel.optimize(model, compile=use_compile, training=training)
            if not training:
                model = model.half().cuda()
        else:
            model = model.cuda()

        # Print applied patterns (once)
        report = autokernel.report(model) if use_ak else {}
        if report.get("patterns"):
            patterns = list(report["patterns"].keys())
            print(f"  Patterns: {', '.join(patterns)}")

        tok_s = bench_model(label, model, B=B, T=T, training=training)
        results[label] = tok_s

        del model
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    baseline = results.get("eager (fp32)", 1)
    for label, tok_s in results.items():
        speedup = tok_s / baseline
        print(f"  {label:30s}: {tok_s:>8,.0f} tok/s  ({speedup:.2f}x vs eager)")

    # Previous baselines for comparison
    print("\n  Previous baselines (before this change):")
    print(f"    Tempest124M AK+compile:      20,184 tok/s")
    print(f"    LlamaModel124M AK+compile:   49,302 tok/s")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
