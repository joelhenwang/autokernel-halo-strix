"""Before/after performance comparison for backward pass optimizations.

Runs training steps on LlamaModel and AMADEUS with backward HIP kernels
disabled (BEFORE) then enabled (AFTER). Compares tok/s, backward ms,
total step ms, and loss progression.

Usage:
    python scripts/bench_before_after.py
    python scripts/bench_before_after.py --steps 100 --warmup 10
"""

import argparse
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


def load_model(model_path, class_name):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls()


def run_training_bench(model, label, batch_size=16, seq_len=256, steps=50, warmup=10,
                       optimize=True, compile_model=False):
    """Run training steps, return timing stats."""
    device = torch.device("cuda")
    model = model.to(device).train()

    if optimize:
        try:
            import autokernel
            model = autokernel.optimize(model, training=True)
            print(f"  autokernel optimizations applied")
        except Exception as e:
            print(f"  autokernel: {e}")

    if compile_model:
        try:
            model = torch.compile(model, mode="default")
            print(f"  torch.compile applied")
        except Exception as e:
            print(f"  compile failed: {e}")

    vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is None:
        # Try to get from embedding layer
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                vocab_size = m.weight.shape[0]
                break
    if vocab_size is None:
        vocab_size = 50257
    n_params = sum(p.numel() for p in model.parameters())
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, fused=True)

    # Warmup
    for _ in range(warmup):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(input_ids)
            if isinstance(output, torch.Tensor) and output.dim() == 0:
                loss = output
            else:
                logits = output if isinstance(output, torch.Tensor) else output.logits
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    # Timed runs
    step_times = []
    fwd_times = []
    bwd_times = []
    losses = []
    tokens_per_step = batch_size * seq_len

    for step in range(steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        # Forward
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(input_ids)
            if isinstance(output, torch.Tensor) and output.dim() == 0:
                loss = output
            else:
                logits = output if isinstance(output, torch.Tensor) else output.logits
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )

        torch.cuda.synchronize()
        t_fwd = time.perf_counter()

        # Backward
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t_bwd = time.perf_counter()

        # Optimizer step
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        step_times.append((t_end - t_start) * 1000)
        fwd_times.append((t_fwd - t_start) * 1000)
        bwd_times.append((t_bwd - t_fwd) * 1000)
        losses.append(loss.item())

    # Stats (skip first 5 steps for stability)
    skip = min(5, steps // 4)
    step_ms = sorted(step_times[skip:])
    fwd_ms = sorted(fwd_times[skip:])
    bwd_ms = sorted(bwd_times[skip:])

    median_step = step_ms[len(step_ms) // 2]
    median_fwd = fwd_ms[len(fwd_ms) // 2]
    median_bwd = bwd_ms[len(bwd_ms) // 2]
    tok_s = tokens_per_step / (median_step / 1000)

    # MFU
    peak_tflops = 59.4e12
    flops_per_tok = 6 * n_params
    achieved_tflops = tok_s * flops_per_tok
    mfu = achieved_tflops / peak_tflops * 100

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    return {
        "label": label,
        "params_m": n_params / 1e6,
        "median_step_ms": round(median_step, 2),
        "median_fwd_ms": round(median_fwd, 2),
        "median_bwd_ms": round(median_bwd, 2),
        "tok_s": round(tok_s, 0),
        "mfu": round(mfu, 1),
        "peak_mem_gb": round(peak_mem, 2),
        "first_loss": round(losses[0], 3),
        "last_loss": round(losses[-1], 3),
        "bwd_pct": round(median_bwd / median_step * 100, 1),
    }


def print_comparison(before, after):
    """Print side-by-side comparison."""
    bwd_speedup = before["median_bwd_ms"] / after["median_bwd_ms"] if after["median_bwd_ms"] > 0 else 0
    step_speedup = before["median_step_ms"] / after["median_step_ms"] if after["median_step_ms"] > 0 else 0
    tok_speedup = after["tok_s"] / before["tok_s"] if before["tok_s"] > 0 else 0

    print(f"\n{'Metric':<25} {'BEFORE':>12} {'AFTER':>12} {'Change':>12}")
    print(f"{'-'*61}")
    print(f"{'Step (ms)':<25} {before['median_step_ms']:>12.2f} {after['median_step_ms']:>12.2f} {step_speedup:>11.2f}x")
    print(f"{'Forward (ms)':<25} {before['median_fwd_ms']:>12.2f} {after['median_fwd_ms']:>12.2f}")
    print(f"{'Backward (ms)':<25} {before['median_bwd_ms']:>12.2f} {after['median_bwd_ms']:>12.2f} {bwd_speedup:>11.2f}x")
    print(f"{'Backward % of step':<25} {before['bwd_pct']:>11.1f}% {after['bwd_pct']:>11.1f}%")
    print(f"{'Throughput (tok/s)':<25} {before['tok_s']:>12.0f} {after['tok_s']:>12.0f} {tok_speedup:>11.2f}x")
    print(f"{'MFU (%)':<25} {before['mfu']:>11.1f}% {after['mfu']:>11.1f}%")
    print(f"{'Peak Memory (GB)':<25} {before['peak_mem_gb']:>12.2f} {after['peak_mem_gb']:>12.2f}")
    print(f"{'First Loss':<25} {before['first_loss']:>12.3f} {after['first_loss']:>12.3f}")
    print(f"{'Last Loss':<25} {before['last_loss']:>12.3f} {after['last_loss']:>12.3f}")


def main():
    parser = argparse.ArgumentParser(description="Before/after backward optimization benchmark")
    parser.add_argument("--steps", type=int, default=50, help="Training steps per run")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile")
    parser.add_argument("--optimize", action="store_true", help="Apply autokernel.optimize()")
    parser.add_argument("--model", choices=["llama", "amadeus", "both"], default="both")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Steps: {args.steps}, Warmup: {args.warmup}, Batch: {args.batch_size}, Seq: {args.seq_len}")
    print()

    models_to_test = []
    if args.model in ("llama", "both"):
        models_to_test.append(("models/llama_7b.py", "LlamaModel"))
    if args.model in ("amadeus", "both"):
        models_to_test.append(("models/amadeus.py", "Amadeus"))

    all_results = []

    for model_path, class_name in models_to_test:
        print(f"{'='*70}")
        print(f"MODEL: {class_name} ({model_path})")
        print(f"{'='*70}")

        # --- BEFORE: backward HIP disabled ---
        print(f"\n--- BEFORE (PyTorch backward) ---")
        os.environ["AUTOKERNEL_NO_BWD_HIP"] = "1"
        # Force reload of _torch_ops to pick up env change
        import kernels.hip._torch_ops as _to
        _to._USE_BWD_HIP = False
        print(f"  Backward HIP kernels: DISABLED")

        model_before = load_model(model_path, class_name)
        torch.cuda.reset_peak_memory_stats()
        before = run_training_bench(
            model_before, f"{class_name} BEFORE",
            batch_size=args.batch_size, seq_len=args.seq_len,
            steps=args.steps, warmup=args.warmup,
            optimize=True, compile_model=args.compile,
        )
        del model_before
        torch.cuda.empty_cache()
        print(f"  tok/s: {before['tok_s']:.0f}, bwd: {before['median_bwd_ms']:.2f}ms, step: {before['median_step_ms']:.2f}ms")

        # --- AFTER: backward HIP enabled ---
        print(f"\n--- AFTER (HIP backward) ---")
        os.environ.pop("AUTOKERNEL_NO_BWD_HIP", None)
        _to._USE_BWD_HIP = True
        print(f"  Backward HIP kernels: ENABLED")

        model_after = load_model(model_path, class_name)
        torch.cuda.reset_peak_memory_stats()
        after = run_training_bench(
            model_after, f"{class_name} AFTER",
            batch_size=args.batch_size, seq_len=args.seq_len,
            steps=args.steps, warmup=args.warmup,
            optimize=True, compile_model=args.compile,
        )
        del model_after
        torch.cuda.empty_cache()
        print(f"  tok/s: {after['tok_s']:.0f}, bwd: {after['median_bwd_ms']:.2f}ms, step: {after['median_step_ms']:.2f}ms")

        print_comparison(before, after)
        all_results.append((class_name, before, after))

    # Final summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Before tok/s':>14} {'After tok/s':>14} {'Speedup':>10} {'Bwd Speedup':>12}")
        print(f"{'-'*70}")
        for name, before, after in all_results:
            tok_sp = after["tok_s"] / before["tok_s"] if before["tok_s"] > 0 else 0
            bwd_sp = before["median_bwd_ms"] / after["median_bwd_ms"] if after["median_bwd_ms"] > 0 else 0
            print(f"{name:<20} {before['tok_s']:>14.0f} {after['tok_s']:>14.0f} {tok_sp:>9.2f}x {bwd_sp:>11.2f}x")


if __name__ == "__main__":
    main()
