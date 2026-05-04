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
    # Register in sys.modules so torch.compile/dynamo can resolve the module
    import sys
    sys.modules["user_model"] = mod
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
    parser.add_argument("--time-budget", type=float, default=None, help="Wall-clock limit in minutes (optional safety net, prefer --epochs)")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Microbatch size")
    parser.add_argument("--block-size", type=int, default=1024, help="Sequence length")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=8e-4, help="Base learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile")
    parser.add_argument("--no-compile-cache", action="store_true", help="Disable Inductor compile cache (force fresh compilation)")
    parser.add_argument("--optimize-kernels", action="store_true", help="Apply autokernel.optimize()")
    parser.add_argument("--muon", action="store_true", help="Use Muon optimizer (2x token-efficiency)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 instead of float16 (no GradScaler needed)")
    parser.add_argument("--mtp", action="store_true", help="Use Multi-Token Prediction auxiliary loss (for models with MTP heads)")
    parser.add_argument("--ema", action="store_true", help="Use EMA of weights (decay=0.999, TRM paper: +7.5%% generalization)")
    parser.add_argument("--mode", default="auto", choices=["auto", "A", "B"], help="Training mode")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint save directory")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Save checkpoint every N steps (default: log_interval * 10)")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--resume-from", default=None, help="Checkpoint path for continued pre-training (loads weights only, fresh optimizer)")

    # SFT arguments
    parser.add_argument("--phase", default="pretrain", choices=["pretrain", "eos-warmup", "sft", "dpo"],
                        help="Training phase: pretrain, eos-warmup, sft, or dpo")
    parser.add_argument("--sft-dataset", default=None,
                        help="SFT dataset: 'alpaca', 'openhermes', or path to local JSONL/parquet")
    parser.add_argument("--sft-format", default="alpaca", choices=["alpaca", "sharegpt", "chatml"],
                        help="Dataset format adapter")
    parser.add_argument("--eos-weight", type=float, default=1.0,
                        help="EOS token loss weight multiplier (default 1.0, use 5.0 for Phase 0)")
    parser.add_argument("--warmup-steps", type=int, default=300, help="LR warmup steps (300 recommended with MTP to avoid gradient spikes)")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "wsd"],
                        help="LR schedule: cosine (default) or wsd (warmup-stable-decay)")
    parser.add_argument("--z-loss", type=float, default=0.0,
                        help="Z-loss weight on logit magnitudes (e.g., 1e-4)")
    parser.add_argument("--z-loss-fraction", type=float, default=0.4,
                        help="Fraction of training to apply z-loss (default 0.4 = first 40%%)")
    parser.add_argument("--context-schedule", default=None,
                        help="Progressive context: '0.2:256,0.6:512,1.0:1024'")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Custom HuggingFace tokenizer .json path")
    parser.add_argument("--wd-start", type=float, default=0.1,
                        help="Weight decay at start (for WSD annealing)")
    parser.add_argument("--wd-end", type=float, default=0.01,
                        help="Weight decay at end of decay phase")
    parser.add_argument("--min-lr-ratio", type=float, default=0.0,
                        help="Minimum LR ratio for WSD/cosine decay floor (e.g., 0.1 = 10%%%% of peak)")
    parser.add_argument("--polar-ns", action="store_true",
                        help="Use Polar-Express NS coefficients in Muon optimizer")
    parser.add_argument("--model-kwarg", action="append", default=[],
                        help="Model constructor kwargs as key=value (repeatable)")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.",
                        help="System prompt for ChatML formatting")
    parser.add_argument("--no-packing", action="store_true",
                        help="Disable conversation packing (pad each conversation individually)")
    parser.add_argument("--tool-use", action="store_true",
                        help="Enable tool-use tokens (<tool_call>, </tool_call>) — uses domain-sft tokenizer with vocab 50262")

    # DPO arguments
    parser.add_argument("--dpo-dataset", default=None,
                        help="DPO preference dataset (JSONL with chosen/rejected pairs)")
    parser.add_argument("--dpo-beta", type=float, default=0.1,
                        help="DPO beta (KL penalty strength, default 0.1)")
    parser.add_argument("--dpo-batch-size", type=int, default=4,
                        help="DPO batch size (smaller than SFT due to 2x model memory)")

    args = parser.parse_args()

    # Disable compile cache if requested (must happen before trainer import)
    if getattr(args, 'no_compile_cache', False):
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
        os.environ.pop("TORCHINDUCTOR_AUTOTUNE_CACHE", None)

    # Load model
    sys.path.insert(0, ".")
    model_kwargs = {}
    for kv in args.model_kwarg:
        key, val = kv.split("=", 1)
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        model_kwargs[key] = val
    model = load_model_from_file(args.model, args.class_name, **model_kwargs)

    # --- Phase-specific setup ---
    loss_fn = None
    dataset_obj = None
    resize_vocab = None

    if args.phase == "eos-warmup":
        from halo_training.sft_loss import build_sft_loss_fn
        loss_fn = build_sft_loss_fn(eos_weight=args.eos_weight)

    elif args.phase == "sft":
        from halo_training.chat_template import build_tokenizer
        from halo_training.sft_data import SFTDataset
        from halo_training.sft_loss import build_sft_loss_fn

        tok_phase = "domain-sft" if args.tool_use else "sft"
        tokenizer = build_tokenizer(phase=tok_phase)
        resize_vocab = tokenizer.vocab_size

        dataset_obj = SFTDataset(
            data_path=args.sft_dataset or args.dataset,
            tokenizer=tokenizer,
            format=args.sft_format,
            block_size=args.block_size,
            system_prompt=args.system_prompt,
            pack=not args.no_packing,
        )
        loss_fn = build_sft_loss_fn(eos_weight=args.eos_weight)

    elif args.phase == "dpo":
        from halo_training.chat_template import build_tokenizer, resize_embeddings
        from halo_training.dpo import DPODataset, train_dpo

        tok_phase = "domain-sft" if args.tool_use else "sft"
        tokenizer = build_tokenizer(phase=tok_phase)
        resize_vocab = tokenizer.vocab_size

        dpo_data = args.dpo_dataset or args.dataset
        dpo_dataset = DPODataset(
            data_path=dpo_data,
            tokenizer=tokenizer,
            block_size=args.block_size,
        )

        # Resize embeddings before loading checkpoint
        model = resize_embeddings(model, resize_vocab)

        # Apply autokernel if requested
        if args.optimize_kernels:
            try:
                import autokernel
                model = autokernel.optimize(model, training=True)
                print("autokernel optimizations applied")
            except ImportError:
                print("autokernel not available, skipping")

        # Load checkpoint
        if args.resume_from:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            import torch
            ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                try:
                    model.load_state_dict(ckpt["model_state_dict"])
                except RuntimeError:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                print(f"Loaded checkpoint: {args.resume_from} (step {ckpt.get('step', '?')})")
            del ckpt

        stats = train_dpo(
            model,
            dataset=dpo_dataset,
            epochs=args.epochs,
            batch_size=args.dpo_batch_size,
            lr=args.lr,
            beta=args.dpo_beta,
            checkpoint_dir=args.checkpoint_dir or "checkpoints/dpo",
            log_interval=args.log_interval,
            max_steps=args.max_steps,
            time_budget_minutes=args.time_budget,
            compile=args.compile,
        )
        print(f"\nDPO stats: {stats}")
        sys.exit(0)

    if args.mtp and loss_fn is None:
        from halo_training.mtp_loss import build_mtp_loss_fn
        loss_fn = build_mtp_loss_fn()

    if args.smoke:
        from halo_training.smoke import run_smoke_test
        result = run_smoke_test(
            model,
            dataset=args.dataset,
            steps=200,
            batch_size=args.batch_size,
            block_size=min(args.block_size, 512),  # smaller for smoke
            compile=args.compile,
            optimize_kernels=args.optimize_kernels,
            use_muon=args.muon,
        )
        sys.exit(0 if result["passed"] else 1)

    # Mixture dataset support: --dataset mixture:path/to/config.json
    if dataset_obj is None and args.dataset.startswith("mixture:"):
        from halo_training.mixture_data import MixtureDataset
        dataset_obj = MixtureDataset(args.dataset[len("mixture:"):], block_size=args.block_size)

    context_schedule = None
    if args.context_schedule:
        context_schedule = []
        for part in args.context_schedule.split(","):
            frac, bs = part.split(":")
            context_schedule.append((float(frac), int(bs)))

    from halo_training.trainer import train
    stats = train(
        model,
        dataset=dataset_obj if dataset_obj else args.dataset,
        epochs=args.epochs,
        time_budget_minutes=args.time_budget,
        batch_size=args.batch_size,
        block_size=args.block_size,
        accum_steps=args.accum_steps,
        base_lr=args.lr,
        loss_fn=loss_fn,
        compile=args.compile,
        optimize_kernels=args.optimize_kernels,
        mode=args.mode,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        use_muon=args.muon,
        use_bf16=args.bf16,
        use_ema=args.ema,
        resume_from=args.resume_from,
        resize_vocab=resize_vocab,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        scheduler_type=args.scheduler,
        z_loss_weight=args.z_loss,
        z_loss_fraction=args.z_loss_fraction,
        context_schedule=context_schedule,
        tokenizer_path=args.tokenizer_path,
        wd_start=args.wd_start,
        wd_end=args.wd_end,
        min_lr_ratio=args.min_lr_ratio,
        polar_ns=args.polar_ns,
    )

    print(f"\nFinal stats: {stats}")


if __name__ == "__main__":
    main()
