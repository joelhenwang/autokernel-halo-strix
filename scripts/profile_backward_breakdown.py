"""Profile backward pass per-op breakdown.

Runs N training steps and captures per-op backward timing using torch.profiler.
Groups operations by custom op name (rmsnorm, rotary, silu_gate_mul, etc.).

Usage:
    python scripts/profile_backward_breakdown.py --model models/llama_7b.py --class-name LlamaModel
    python scripts/profile_backward_breakdown.py --model models/amadeus.py --class-name AMADEUS
"""

import argparse
import importlib.util
import json
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


def load_model(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls()


def run_profiling(model, batch_size=4, seq_len=256, n_steps=20, warmup=5, optimize=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()

    if optimize:
        try:
            import autokernel
            model = autokernel.optimize(model, training=True)
            print("autokernel optimizations applied")
        except Exception as e:
            print(f"autokernel not available: {e}")

    vocab_size = getattr(model, "vocab_size", 50257)
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

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

    # Profile backward only
    backward_times = {}
    total_backward_ms = 0.0

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=False,
        record_shapes=True,
    ) as prof:
        for step in range(n_steps):
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

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            bwd_ms = (time.perf_counter() - t0) * 1000.0
            total_backward_ms += bwd_ms

            scaler.step(optimizer)
            scaler.update()

    # Parse profiler events
    events = prof.key_averages()
    op_summary = {}
    for evt in events:
        name = evt.key
        if "backward" in name.lower() or "Backward" in name:
            cuda_time_ms = evt.cuda_time_total / 1000.0  # us to ms
            count = evt.count
            if name not in op_summary:
                op_summary[name] = {"calls": 0, "total_ms": 0.0}
            op_summary[name]["calls"] += count
            op_summary[name]["total_ms"] += cuda_time_ms

    avg_backward_ms = total_backward_ms / n_steps

    results = {
        "n_steps": n_steps,
        "avg_backward_ms": round(avg_backward_ms, 2),
        "ops": {},
    }

    print(f"\n{'='*70}")
    print(f"Backward Pass Breakdown ({n_steps} steps, avg {avg_backward_ms:.2f} ms/step)")
    print(f"{'='*70}")
    print(f"{'Op Name':<50} {'Calls':>6} {'Total ms':>10} {'%':>6}")
    print(f"{'-'*70}")

    sorted_ops = sorted(op_summary.items(), key=lambda x: -x[1]["total_ms"])
    for name, data in sorted_ops[:20]:
        calls = data["calls"]
        total_ms = data["total_ms"]
        pct = (total_ms / (avg_backward_ms * n_steps)) * 100 if avg_backward_ms > 0 else 0
        mean_ms = total_ms / calls if calls > 0 else 0
        print(f"{name[:50]:<50} {calls:>6} {total_ms:>10.2f} {pct:>5.1f}%")
        results["ops"][name] = {
            "calls": calls,
            "total_ms": round(total_ms, 3),
            "mean_ms": round(mean_ms, 4),
            "pct_of_backward": round(pct, 2),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile backward pass per-op breakdown")
    parser.add_argument("--model", required=True, help="Path to model .py file")
    parser.add_argument("--class-name", required=True, help="Model class name")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    args = parser.parse_args()

    print(f"Loading model: {args.model}::{args.class_name}")
    model = load_model(args.model, args.class_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")

    results = run_profiling(
        model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_steps=args.steps,
        warmup=args.warmup,
        optimize=not args.no_optimize,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
