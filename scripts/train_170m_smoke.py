"""Train all 170M hypothesis models on smoke-test-dataset for 2 epochs.

Shuffled each epoch, 10% validation split. Reports loss, tok/s, steps/sec.

Usage:
    python scripts/train_170m_smoke.py
    python scripts/train_170m_smoke.py --model Tempest
"""

import argparse
import importlib.util
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from halo_training.data import BabyLMDataset


# ~170M configurations for each model
CONFIGS = {
    "Tempest": {
        "path": "models/tempest.py", "class": "Tempest",
        "kwargs": dict(vocab_size=50257, d_model=896, n_layers=14,
                       d_conv=512, d_griffin=384, ffn_inner=2304),
    },
    "SpectralHydra": {
        "path": "models/spectral_hydra.py", "class": "SpectralHydra",
        "kwargs": dict(vocab_size=50257, d_model=896, n_layers=14,
                       n_heads=14, d_head=64, ffn_inner=2304),
    },
    "ResonantLoop": {
        "path": "models/resonant_loop.py", "class": "ResonantLoop",
        "kwargs": dict(vocab_size=50257, d_model=896, ffn_inner=1792,
                       max_iterations=16),
    },
    "MaestroPrima": {
        "path": "models/maestro_prima.py", "class": "MaestroPrima",
        "kwargs": dict(vocab_size=50257, d_model=896, n_layers=12,
                       d_conv=512, d_mamba=384, dstate=64, n_ssm_heads=6,
                       ffn_inner=2304, d_film=64, film_start=6),
    },
    "Prometheus": {
        "path": "models/prometheus.py", "class": "Prometheus",
        "kwargs": dict(vocab_size=50257, d_model=896, n_layers=14,
                       d_conv=512, d_griffin=384, ffn_inner=2304,
                       n_attn_heads=8, n_kv_heads=2, head_dim=112,
                       attn_layers=(3, 10)),
    },
    "DualCortex": {
        "path": "models/dual_cortex.py", "class": "DualCortex",
        "kwargs": dict(vocab_size=50257, d_embed=896, d_fast=256,
                       d_slow=896, n_fast_layers=6, n_slow_layers=8,
                       ffn_fast=512, ffn_slow=2304),
    },
    "Obsidian": {
        "path": "models/obsidian.py", "class": "Obsidian",
        "kwargs": dict(vocab_size=50257, d_model=896, d_reflex=256,
                       n_reflex_layers=6, d_reflex_ffn=512,
                       n_genius_layers=8, d_genius_rec=896, ffn_genius=2304),
    },
    "Amadeus": {
        "path": "models/amadeus.py", "class": "Amadeus",
        "kwargs": dict(vocab_size=50257, d_model=896, n_layers=12,
                       d_conv=512, d_mamba=384, dstate=64, n_ssm_heads=6,
                       ffn_inner=2304, d_film=64, film_start=6),
    },
    "Virtuoso": {
        "path": "models/virtuoso.py", "class": "Virtuoso",
        "kwargs": dict(vocab_size=50257, d_model=896, n_layers=14,
                       d_conv=512, d_griffin=384, ffn_inner=2304),
    },
}


def load_model(path, cls_name, kwargs):
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)(**kwargs)


def evaluate(model, val_loader, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, targets = batch
            input_ids, targets = input_ids.to(device), targets.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(input_ids)
                logits = output if isinstance(output, torch.Tensor) else output.logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def train_model(name, cfg, dataset_path, n_epochs=2, batch_size=16, block_size=256,
                log_interval=20, optimize=True):
    """Train a single model, return results dict."""
    device = torch.device("cuda")
    print(f"\n{'='*70}")
    print(f"  {name} ({cfg['path']})")
    print(f"{'='*70}")

    # Load model
    try:
        model = load_model(cfg["path"], cfg["class"], cfg["kwargs"])
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")

    model = model.to(device).train()

    # Apply autokernel
    if optimize:
        try:
            import autokernel
            model = autokernel.optimize(model, training=True)
            print(f"  autokernel applied")
        except Exception as e:
            print(f"  autokernel failed: {e}")

    # Load dataset
    full_dataset = BabyLMDataset(root=dataset_path, block_size=block_size)

    # 90/10 train/val split
    n_val = max(1, int(len(full_dataset) * 0.1))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Data: {n_train} train / {n_val} val chunks (block_size={block_size})")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.1, fused=True)
    total_steps = len(train_loader) * n_epochs
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    results = {
        "name": name, "params_m": n_params / 1e6,
        "train_losses": [], "val_losses": [],
        "tok_s_history": [], "steps_s_history": [],
    }
    global_step = 0
    tokens_since_log = 0
    t_log = time.time()
    t_start = time.time()
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids, targets = batch
            input_ids, targets = input_ids.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(input_ids)
                logits = output if isinstance(output, torch.Tensor) else output.logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1
            tokens_since_log += input_ids.numel()

            if global_step % log_interval == 0:
                torch.cuda.synchronize()
                t_now = time.time()
                dt = t_now - t_log
                avg_loss = running_loss / n_batches
                tok_s = tokens_since_log / dt
                steps_s = log_interval / dt
                lr = scheduler.get_last_lr()[0]

                print(f"  [epoch {epoch+1} step {global_step:>4d}] "
                      f"loss={avg_loss:.4f}  lr={lr:.2e}  "
                      f"tok/s={tok_s:,.0f}  steps/s={steps_s:.1f}")

                results["train_losses"].append(avg_loss)
                results["tok_s_history"].append(tok_s)
                results["steps_s_history"].append(steps_s)

                running_loss = 0.0
                n_batches = 0
                tokens_since_log = 0
                t_log = t_now

        # End of epoch: validate
        torch.cuda.synchronize()
        val_loss = evaluate(model, val_loader, device)
        results["val_losses"].append(val_loss)
        best_val_loss = min(best_val_loss, val_loss)
        elapsed = time.time() - t_start
        print(f"  --- Epoch {epoch+1} done | val_loss={val_loss:.4f} | "
              f"best_val={best_val_loss:.4f} | elapsed={elapsed:.0f}s ---")

    # Final summary
    total_elapsed = time.time() - t_start
    avg_tok_s = sum(results["tok_s_history"]) / len(results["tok_s_history"]) if results["tok_s_history"] else 0
    avg_steps_s = sum(results["steps_s_history"]) / len(results["steps_s_history"]) if results["steps_s_history"] else 0
    final_train_loss = results["train_losses"][-1] if results["train_losses"] else float("inf")

    results["final_train_loss"] = final_train_loss
    results["best_val_loss"] = best_val_loss
    results["avg_tok_s"] = avg_tok_s
    results["avg_steps_s"] = avg_steps_s
    results["total_steps"] = global_step
    results["elapsed_s"] = total_elapsed

    print(f"  FINAL: train_loss={final_train_loss:.4f} val_loss={best_val_loss:.4f} "
          f"tok/s={avg_tok_s:,.0f} steps/s={avg_steps_s:.1f} ({total_elapsed:.0f}s)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Specific model name to test")
    parser.add_argument("--dataset", type=str, default="datasets/smoke-test-dataset")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Block: {args.block_size}")
    print()

    if args.model:
        configs = {args.model: CONFIGS[args.model]}
    else:
        configs = CONFIGS

    all_results = []
    for name, cfg in configs.items():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        import gc; gc.collect()

        result = train_model(
            name, cfg, args.dataset,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            block_size=args.block_size,
            log_interval=args.log_interval,
            optimize=not args.no_optimize,
        )
        if result:
            all_results.append(result)

        # Cleanup between models
        torch.cuda.empty_cache()
        import gc; gc.collect()

    # Summary table
    print(f"\n{'='*100}")
    print(f"TRAINING SUMMARY (2 epochs on {args.dataset})")
    print(f"{'='*100}")
    print(f"{'Model':20s} {'Params':>8s} {'Train Loss':>11s} {'Val Loss':>10s} "
          f"{'tok/s':>8s} {'steps/s':>8s} {'Time':>6s}")
    print(f"{'-'*100}")
    for r in sorted(all_results, key=lambda x: x["best_val_loss"]):
        print(f"{r['name']:20s} {r['params_m']:7.1f}M "
              f"{r['final_train_loss']:>11.4f} {r['best_val_loss']:>10.4f} "
              f"{r['avg_tok_s']:>8,.0f} {r['avg_steps_s']:>8.1f} "
              f"{r['elapsed_s']:>5.0f}s")


if __name__ == "__main__":
    main()
