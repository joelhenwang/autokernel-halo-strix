"""Benchmark all hypothesis architectures: instantiation, forward, training step.

Reports: param count, tok/s (eager), tok/s (optimized), MFU, peak memory.

Usage:
    python scripts/bench_all_hypotheses.py
    python scripts/bench_all_hypotheses.py --steps 30 --warmup 5
    python scripts/bench_all_hypotheses.py --model spectral_hydra --class-name SpectralHydra
"""

import argparse
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


ALL_MODELS = [
    ("models/llama_7b.py", "LlamaModel", "LlamaModel 124.7M"),
    ("models/amadeus.py", "Amadeus", "AMADEUS 243.8M"),
    ("models/tempest.py", "Tempest", "TEMPEST 244.5M"),
    ("models/prometheus.py", "Prometheus", "PROMETHEUS 216M"),
    ("models/virtuoso.py", "Virtuoso", "VIRTUOSO 252M"),
    ("models/maestro_prima.py", "MaestroPrima", "MAESTRO-PRIMA 241.7M"),
    ("models/spectral_hydra.py", "SpectralHydra", "SPECTRAL-HYDRA 245M"),
    ("models/resonant_loop.py", "ResonantLoop", "RESONANT-LOOP 58.8M"),
    ("models/dual_cortex.py", "DualCortex", "DUAL-CORTEX 231M"),
    ("models/obsidian.py", "Obsidian", "OBSIDIAN 224M"),
]


def load_model(model_path, class_name):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls()


def bench_model(model_path, class_name, label, batch_size=8, seq_len=256,
                steps=30, warmup=5, optimize=False, compile_model=False):
    """Benchmark a single model. Returns dict of metrics."""
    device = torch.device("cuda")

    try:
        model = load_model(model_path, class_name)
    except Exception as e:
        return {"label": label, "error": str(e)}

    model = model.to(device).train()
    n_params = sum(p.numel() for p in model.parameters())

    if optimize:
        try:
            import autokernel
            model = autokernel.optimize(model, training=True)
        except Exception:
            pass

    if compile_model:
        try:
            model = torch.compile(model, mode="default")
        except Exception:
            pass

    # Detect vocab size
    vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is None:
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                vocab_size = m.weight.shape[0]
                break
    if vocab_size is None:
        vocab_size = 50257

    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, fused=True)
    tokens_per_step = batch_size * seq_len

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
    losses = []

    for step in range(steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

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
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)
        losses.append(loss.item())

    # Stats
    skip = min(3, steps // 4)
    times = sorted(step_times[skip:])
    median_ms = times[len(times) // 2]
    tok_s = tokens_per_step / (median_ms / 1000)

    peak_tflops = 59.4e12
    flops_per_tok = 6 * n_params
    mfu = tok_s * flops_per_tok / peak_tflops * 100

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    return {
        "label": label,
        "params_m": round(n_params / 1e6, 1),
        "median_step_ms": round(median_ms, 2),
        "tok_s": round(tok_s, 0),
        "mfu": round(mfu, 1),
        "peak_mem_gb": round(peak_mem, 2),
        "first_loss": round(losses[0], 3),
        "last_loss": round(losses[-1], 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark all hypothesis architectures")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--optimize", action="store_true", help="Apply autokernel.optimize()")
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile")
    parser.add_argument("--model", type=str, default=None, help="Specific model file to test")
    parser.add_argument("--class-name", type=str, default=None, help="Class name for --model")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Steps: {args.steps}, Warmup: {args.warmup}, Batch: {args.batch_size}, Seq: {args.seq_len}")
    print(f"Optimize: {args.optimize}, Compile: {args.compile}")
    print()

    if args.model and args.class_name:
        models = [(args.model, args.class_name, f"{args.class_name}")]
    else:
        models = ALL_MODELS

    results = []
    for model_path, class_name, label in models:
        print(f"{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        result = bench_model(
            model_path, class_name, label,
            batch_size=args.batch_size, seq_len=args.seq_len,
            steps=args.steps, warmup=args.warmup,
            optimize=args.optimize, compile_model=args.compile,
        )

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Params: {result['params_m']}M")
            print(f"  tok/s: {result['tok_s']:.0f}")
            print(f"  MFU: {result['mfu']:.1f}%")
            print(f"  Step: {result['median_step_ms']:.2f}ms")
            print(f"  Peak mem: {result['peak_mem_gb']:.2f} GB")
            print(f"  Loss: {result['first_loss']:.3f} → {result['last_loss']:.3f}")

        results.append(result)
        # cleanup
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print()

    # Summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY {'(optimized)' if args.optimize else '(eager)'} {'+ compile' if args.compile else ''}")
    print(f"{'='*90}")
    print(f"{'Model':<30} {'Params':>8} {'tok/s':>8} {'MFU':>6} {'Step(ms)':>10} {'Mem(GB)':>8} {'Loss':>8}")
    print(f"{'-'*90}")
    for r in results:
        if "error" in r:
            print(f"{r['label']:<30} {'ERROR':>8}: {r['error'][:40]}")
        else:
            print(f"{r['label']:<30} {r['params_m']:>7.1f}M {r['tok_s']:>8.0f} {r['mfu']:>5.1f}% "
                  f"{r['median_step_ms']:>10.2f} {r['peak_mem_gb']:>8.2f} {r['last_loss']:>8.3f}")


if __name__ == "__main__":
    main()
