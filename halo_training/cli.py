"""Command-line interface for Halo Training Stack.

Usage:
    python -m halo_training --model models/llama_7b.py --class-name LlamaModel --dataset babylm
    python -m halo_training --model models/llama_7b.py --class-name LlamaModel --smoke
    python -m halo_training --model models/llama_7b.py --class-name LlamaModel --optimize-kernels
"""

import argparse
import importlib.util
import sys
import os


def load_model_from_file(model_path: str, class_name: str, **kwargs):
    """Dynamically load a model class from a Python file (same pattern as verify.py)."""
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model from: {model_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, class_name):
        available = [n for n in dir(mod) if not n.startswith("_")]
        raise AttributeError(
            f"Class '{class_name}' not found in {model_path}. Available: {available}"
        )

    cls = getattr(mod, class_name)
    return cls(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Halo Training Stack")
    parser.add_argument("--model", required=True, help="Path to model .py file")
    parser.add_argument("--class-name", required=True, help="Model class name")
    parser.add_argument("--dataset", default="datasets/babylm-strict-small", help="Dataset path or 'babylm'")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test instead of training")

    # Training params
    parser.add_argument("--time-budget", type=float, default=45.0, help="Training time budget in minutes")
    parser.add_argument("--batch-size", type=int, default=16, help="Microbatch size")
    parser.add_argument("--block-size", type=int, default=1024, help="Sequence length")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=8e-4, help="Base learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile")
    parser.add_argument("--optimize-kernels", action="store_true", help="Apply autokernel.optimize()")
    parser.add_argument("--mode", default="auto", choices=["auto", "A", "B"], help="Training mode")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint save directory")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")

    args = parser.parse_args()

    # Load model
    sys.path.insert(0, ".")
    model = load_model_from_file(args.model, args.class_name)

    if args.smoke:
        from halo_training.smoke import run_smoke_test
        result = run_smoke_test(
            model,
            dataset=args.dataset,
            steps=200,
            batch_size=args.batch_size,
            block_size=min(args.block_size, 512),  # smaller for smoke
        )
        sys.exit(0 if result["passed"] else 1)

    from halo_training.trainer import train
    stats = train(
        model,
        dataset=args.dataset,
        epochs=args.epochs,
        time_budget_minutes=args.time_budget,
        batch_size=args.batch_size,
        block_size=args.block_size,
        accum_steps=args.accum_steps,
        base_lr=args.lr,
        compile=args.compile,
        optimize_kernels=args.optimize_kernels,
        mode=args.mode,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
    )

    print(f"\nFinal stats: {stats}")


if __name__ == "__main__":
    main()
