"""
Profile a full training step to identify bottlenecks.

Measures time spent in: data loading, forward, loss, backward, optimizer step,
gradient clipping, torch.compile overhead, and logging.

Usage:
    python scripts/profile_training_step.py --model models/amadeus.py --class-name Amadeus
    python scripts/profile_training_step.py --model models/tempest.py --class-name Tempest --compile
    python scripts/profile_training_step.py --model models/prometheus.py --class-name Prometheus --compile --optimize-kernels
    python scripts/profile_training_step.py --steps 50 --compile --optimize-kernels
"""

import argparse
import importlib.util
import os
import sys
import time

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_model_class(model_path, class_name):
    """Load a model class from a file path."""
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def profile_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"PyTorch: {torch.__version__}")

    # Load model
    ModelClass = load_model_class(args.model, args.class_name)
    model = ModelClass().to(device)  # fp32 master weights, autocast handles fp16
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.class_name} ({n_params/1e6:.1f}M params)")

    # Apply optimizations
    if args.optimize_kernels:
        try:
            import autokernel
            model = autokernel.optimize(model, training=True)
            print("Applied autokernel.optimize()")
        except Exception as e:
            print(f"autokernel.optimize() failed: {e}")

    if args.compile:
        model = torch.compile(model, mode="default")
        print("Applied torch.compile()")

    # Setup optimizer (no GradScaler — external kernels like mamba-ssm may produce fp16 grads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, fused=True)

    # Setup data
    from halo_training.data import BabyLMDataset, build_dataloader
    dataset = BabyLMDataset(block_size=args.block_size)
    loader = build_dataloader(
        dataset, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader_iter = iter(loader)

    # Timing accumulators
    timings = {
        "data_load": [],
        "forward": [],
        "loss": [],
        "backward": [],
        "grad_clip": [],
        "optimizer_step": [],
        "scaler_update": [],
        "total_step": [],
    }

    # Track compile overhead
    compile_times = []

    print(f"\nProfiling {args.steps} steps (batch={args.batch_size}, seq={args.block_size})...")
    print("-" * 70)

    for step in range(args.steps):
        step_start = time.time()
        torch.cuda.synchronize()

        # Data loading
        t0 = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        # BabyLMDataset returns (input_ids, targets) tuple
        if isinstance(batch, (list, tuple)):
            input_ids, targets = batch[0].to(device), batch[1].to(device)
        else:
            input_ids = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
        torch.cuda.synchronize()
        timings["data_load"].append(time.time() - t0)

        # Forward pass
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids)
        torch.cuda.synchronize()
        timings["forward"].append(time.time() - t0)

        # Loss computation
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            if logits.dim() == 0:
                # Model returned scalar loss (e.g., adaptive head)
                loss = logits
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
        torch.cuda.synchronize()
        timings["loss"].append(time.time() - t0)

        # Backward pass
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        timings["backward"].append(time.time() - t0)

        # Gradient clipping
        t0 = time.time()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.cuda.synchronize()
        timings["grad_clip"].append(time.time() - t0)

        # Optimizer step
        t0 = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        timings["optimizer_step"].append(time.time() - t0)

        # Zero grad
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        timings["scaler_update"].append(time.time() - t0)

        total = time.time() - step_start
        timings["total_step"].append(total)

        tokens = input_ids.numel()
        tok_s = tokens / total

        if step % 10 == 0 or step < 5:
            print(f"  step {step:4d}: loss={loss.item():.3f}, "
                  f"tok/s={tok_s:.0f}, total={total*1000:.1f}ms, "
                  f"grad_norm={grad_norm:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)

    # Skip first few steps (compile warmup)
    skip = min(5, args.steps // 2)
    print(f"(Excluding first {skip} steps as warmup)\n")

    total_avg = 0
    for phase, times in timings.items():
        if phase == "total_step":
            continue
        t_list = times[skip:]
        if not t_list:
            continue
        avg_ms = sum(t_list) / len(t_list) * 1000
        total_avg += avg_ms
        pct = avg_ms / (sum(timings["total_step"][skip:]) / len(timings["total_step"][skip:]) * 1000) * 100
        print(f"  {phase:20s}: {avg_ms:8.2f}ms ({pct:5.1f}%)")

    step_avg = sum(timings["total_step"][skip:]) / len(timings["total_step"][skip:]) * 1000
    tokens_per_step = args.batch_size * args.block_size
    tok_s_avg = tokens_per_step / (step_avg / 1000)

    print(f"  {'TOTAL':20s}: {step_avg:8.2f}ms")
    print(f"\n  Avg tok/s: {tok_s_avg:.0f}")
    print(f"  Tokens/step: {tokens_per_step:,}")

    # torch.compile recompilation stats
    if args.compile:
        try:
            counters = torch._dynamo.utils.counters
            print(f"\n  torch.compile stats:")
            for k, v in counters.items():
                if v:
                    print(f"    {k}: {dict(v)}")
        except Exception:
            pass

    # Top bottlenecks
    print(f"\n  TOP 3 BOTTLENECKS:")
    phase_avgs = {}
    for phase, times in timings.items():
        if phase == "total_step":
            continue
        t_list = times[skip:]
        if t_list:
            phase_avgs[phase] = sum(t_list) / len(t_list) * 1000

    for i, (phase, avg) in enumerate(sorted(phase_avgs.items(), key=lambda x: -x[1])[:3]):
        pct = avg / step_avg * 100
        print(f"    {i+1}. {phase}: {avg:.2f}ms ({pct:.1f}%)")

    # Use torch.profiler for detailed GPU trace if requested
    if args.trace:
        print(f"\nGenerating Chrome trace ({args.trace_steps} steps)...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for step in range(args.trace_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)

                if isinstance(batch, (list, tuple)):
                    input_ids, targets = batch[0].to(device), batch[1].to(device)
                else:
                    input_ids = batch[:, :-1].to(device)
                    targets = batch[:, 1:].to(device)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids)
                    if logits.dim() == 0:
                        loss = logits
                    else:
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        trace_path = f"profile_trace_{args.class_name.lower()}.json"
        prof.export_chrome_trace(trace_path)
        print(f"Chrome trace saved to {trace_path}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    parser = argparse.ArgumentParser(description="Profile training step breakdown")
    parser.add_argument("--model", default="models/amadeus.py", help="Model file path")
    parser.add_argument("--class-name", default="Amadeus", help="Model class name")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps to profile")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--block-size", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--optimize-kernels", action="store_true", help="Use autokernel")
    parser.add_argument("--trace", action="store_true", help="Generate Chrome trace")
    parser.add_argument("--trace-steps", type=int, default=10, help="Steps for Chrome trace")
    args = parser.parse_args()
    profile_training(args)


if __name__ == "__main__":
    main()
