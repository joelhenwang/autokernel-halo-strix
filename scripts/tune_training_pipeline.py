"""
Test training pipeline optimizations: DataLoader configs, batch sizes,
compile modes, and unified memory scheduling.

Usage:
    python scripts/tune_training_pipeline.py --model models/amadeus.py --class-name Amadeus
    python scripts/tune_training_pipeline.py --test dataloader
    python scripts/tune_training_pipeline.py --test batch-size
    python scripts/tune_training_pipeline.py --test compile-mode
    python scripts/tune_training_pipeline.py --test all
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


def measure_throughput(model, loader, optimizer, device, steps=30, warmup=5):
    """Run training steps and return avg tok/s (excluding warmup)."""
    model.train()
    loader_iter = iter(loader)
    tok_per_step = []

    for step in range(steps):
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

        torch.cuda.synchronize()
        t0 = time.time()

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

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        tokens = input_ids.numel()

        if step >= warmup:
            tok_per_step.append(tokens / elapsed)

    return sum(tok_per_step) / len(tok_per_step) if tok_per_step else 0


# ============================================================================
# Test: DataLoader configurations
# ============================================================================

def test_dataloader(args):
    print("\n" + "=" * 60)
    print("TEST: DataLoader configurations")
    print("=" * 60)

    from halo_training.data import BabyLMDataset, build_dataloader

    device = torch.device("cuda")
    ModelClass = load_model_class(args.model, args.class_name)
    dataset = BabyLMDataset(block_size=args.block_size)

    configs = [
        {"num_workers": 0, "pin_memory": False, "prefetch_factor": None},
        {"num_workers": 2, "pin_memory": False, "prefetch_factor": 2},
        {"num_workers": 4, "pin_memory": False, "prefetch_factor": 2},
        {"num_workers": 4, "pin_memory": True, "prefetch_factor": 2},
        {"num_workers": 4, "pin_memory": True, "prefetch_factor": 4},
        {"num_workers": 8, "pin_memory": True, "prefetch_factor": 2},
    ]

    results = []
    for cfg in configs:
        model = ModelClass().to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, fused=True)

        # Build DataLoader directly to test pin_memory/prefetch_factor
        loader_kwargs = {
            "batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": cfg["num_workers"],
            "pin_memory": cfg["pin_memory"],
            "drop_last": True,
        }
        if cfg["num_workers"] > 0 and cfg["prefetch_factor"] is not None:
            loader_kwargs["prefetch_factor"] = cfg["prefetch_factor"]

        loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

        tok_s = measure_throughput(model, loader, optimizer, device, steps=30)
        desc = f"workers={cfg['num_workers']}, pin={cfg['pin_memory']}, prefetch={cfg['prefetch_factor']}"
        print(f"  {desc:50s} → {tok_s:.0f} tok/s")
        results.append((desc, tok_s))

        del model, optimizer, scaler
        torch.cuda.empty_cache()

    best = max(results, key=lambda x: x[1])
    print(f"\n  Best: {best[0]} → {best[1]:.0f} tok/s")


# ============================================================================
# Test: Batch size tuning
# ============================================================================

def test_batch_size(args):
    print("\n" + "=" * 60)
    print("TEST: Batch size tuning")
    print("=" * 60)

    from halo_training.data import BabyLMDataset, build_dataloader

    device = torch.device("cuda")
    ModelClass = load_model_class(args.model, args.class_name)
    dataset = BabyLMDataset(block_size=args.block_size)

    batch_sizes = [4, 8, 16, 32, 64]
    results = []

    for bs in batch_sizes:
        try:
            model = ModelClass().to(device)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, fused=True)
            scaler = torch.amp.GradScaler("cuda")

            loader = build_dataloader(
                dataset, batch_size=bs, num_workers=4,
            )

            tok_s = measure_throughput(model, loader, optimizer, device, steps=30)
            tokens_per_step = bs * args.block_size
            print(f"  batch_size={bs:3d} (tokens/step={tokens_per_step:6d}) → {tok_s:.0f} tok/s")
            results.append((bs, tok_s))

            del model, optimizer, scaler
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  batch_size={bs:3d} → OOM")
                torch.cuda.empty_cache()
            else:
                raise

    if results:
        best = max(results, key=lambda x: x[1])
        print(f"\n  Best: batch_size={best[0]} → {best[1]:.0f} tok/s")


# ============================================================================
# Test: torch.compile mode comparison
# ============================================================================

def test_compile_mode(args):
    print("\n" + "=" * 60)
    print("TEST: torch.compile mode comparison")
    print("=" * 60)

    from halo_training.data import BabyLMDataset, build_dataloader

    device = torch.device("cuda")
    ModelClass = load_model_class(args.model, args.class_name)
    dataset = BabyLMDataset(block_size=args.block_size)

    modes = [
        ("eager (no compile)", None),
        ("compile default", "default"),
        ("compile reduce-overhead", "reduce-overhead"),
        ("compile max-autotune", "max-autotune"),
    ]

    results = []
    for desc, mode in modes:
        model = ModelClass().to(device)
        model.train()

        if mode is not None:
            print(f"  Compiling with mode={mode}...")
            t0 = time.time()
            model = torch.compile(model, mode=mode)
            compile_time = time.time() - t0
            print(f"    Compile setup: {compile_time:.1f}s")
        else:
            compile_time = 0

        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, fused=True)
        loader = build_dataloader(
            dataset, batch_size=args.batch_size, num_workers=4,
        )

        # Extra warmup for compiled models (first step triggers compilation)
        tok_s = measure_throughput(model, loader, optimizer, device,
                                   steps=40, warmup=10)
        print(f"  {desc:35s} → {tok_s:.0f} tok/s (compile={compile_time:.1f}s)")
        results.append((desc, tok_s))

        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # Check dynamo recompilation stats
    try:
        counters = torch._dynamo.utils.counters
        if any(counters.values()):
            print(f"\n  torch.compile recompilation stats:")
            for k, v in counters.items():
                if v:
                    print(f"    {k}: {dict(v)}")
    except Exception:
        pass

    # Check inductor tuning options
    print(f"\n  Inductor config tips:")
    print(f"    torch._dynamo.config.cache_size_limit = {torch._dynamo.config.cache_size_limit}")
    print(f"    Increase if recompilations are happening")

    if results:
        best = max(results, key=lambda x: x[1])
        print(f"\n  Best: {best[0]} → {best[1]:.0f} tok/s")


# ============================================================================
# Test: Inductor GEMM tuning
# ============================================================================

def test_inductor_tuning(args):
    print("\n" + "=" * 60)
    print("TEST: Inductor GEMM tuning (coordinate_descent_tuning)")
    print("=" * 60)

    from halo_training.data import BabyLMDataset, build_dataloader

    device = torch.device("cuda")
    ModelClass = load_model_class(args.model, args.class_name)
    dataset = BabyLMDataset(block_size=args.block_size)

    configs = [
        ("default", {}),
        ("coordinate_descent", {"torch._inductor.config.coordinate_descent_tuning": True}),
    ]

    results = []
    for desc, inductor_cfg in configs:
        # Apply inductor config
        for key, val in inductor_cfg.items():
            parts = key.rsplit(".", 1)
            mod = eval(parts[0])
            setattr(mod, parts[1], val)

        model = ModelClass().to(device)
        model.train()
        model = torch.compile(model, mode="default")

        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, fused=True)
        loader = build_dataloader(
            dataset, batch_size=args.batch_size, num_workers=4,
        )

        tok_s = measure_throughput(model, loader, optimizer, device,
                                   steps=40, warmup=10)
        print(f"  {desc:35s} → {tok_s:.0f} tok/s")
        results.append((desc, tok_s))

        del model, optimizer, scaler
        torch.cuda.empty_cache()

    if results:
        best = max(results, key=lambda x: x[1])
        print(f"\n  Best: {best[0]} → {best[1]:.0f} tok/s")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tune training pipeline")
    parser.add_argument("--model", default="models/amadeus.py", help="Model file path")
    parser.add_argument("--class-name", default="Amadeus", help="Model class name")
    parser.add_argument("--batch-size", type=int, default=8, help="Default batch size")
    parser.add_argument("--block-size", type=int, default=512, help="Sequence length")
    parser.add_argument("--test", default="all",
                        choices=["dataloader", "batch-size", "compile-mode",
                                 "inductor", "all"],
                        help="Which test to run")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Run on Strix Halo machine.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    tests = {
        "dataloader": test_dataloader,
        "batch-size": test_batch_size,
        "compile-mode": test_compile_mode,
        "inductor": test_inductor_tuning,
    }

    if args.test == "all":
        for name, fn in tests.items():
            try:
                fn(args)
            except Exception as e:
                print(f"\n  [ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        tests[args.test](args)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
