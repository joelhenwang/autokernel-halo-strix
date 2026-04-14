"""Standalone NVIDIA GPU training script — no ROCm/autokernel dependencies.

Self-contained training with Muon optimizer, pre-tokenized .bin datasets,
checkpoint save/load, and torch.compile support.

Dependencies: torch, tiktoken, numpy

Usage:
    # Fresh training
    python train_nvidia.py \
        --model-file argus_prime_standalone.py --class-name ArgusPrime \
        --dataset my_data.bin --checkpoint-dir checkpoints --epochs 2

    # Continued pre-training from checkpoint
    python train_nvidia.py \
        --model-file argus_prime_standalone.py --class-name ArgusPrime \
        --resume-from checkpoints/step_7200.pt \
        --dataset my_data.bin --checkpoint-dir checkpoints
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ===========================================================================
# Muon Optimizer (inlined from halo_training/muon.py)
# ===========================================================================

def zeropower_via_newtonschulz5(G, steps=5):
    """Compute UV^T from SVD via 5 Newton-Schulz iterations."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer with internal AdamW fallback for non-2D params."""

    def __init__(self, muon_params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.01, adamw_params=None,
                 adamw_lr=8e-4, adamw_betas=(0.9, 0.95), adamw_wd=0.0):
        if isinstance(muon_params, dict):
            muon_params = [muon_params]
        muon_groups = []
        for group in muon_params:
            if isinstance(group, dict):
                g = dict(group)
                g.setdefault("lr", lr)
                g.setdefault("momentum", momentum)
                g.setdefault("nesterov", nesterov)
                g.setdefault("ns_steps", ns_steps)
                g.setdefault("weight_decay", weight_decay)
                g["_optimizer_type"] = "muon"
                muon_groups.append(g)
            else:
                muon_groups.append({
                    "params": list(group) if hasattr(group, "__iter__") and not isinstance(group, torch.Tensor) else [group],
                    "lr": lr, "momentum": momentum, "nesterov": nesterov,
                    "ns_steps": ns_steps, "weight_decay": weight_decay,
                    "_optimizer_type": "muon",
                })

        adamw_groups = []
        if adamw_params is not None:
            if isinstance(adamw_params, dict):
                adamw_params = [adamw_params]
            for group in adamw_params:
                if isinstance(group, dict):
                    g = dict(group)
                    g.setdefault("lr", adamw_lr)
                    g.setdefault("betas", adamw_betas)
                    g.setdefault("weight_decay", g.pop("weight_decay", adamw_wd))
                    g["_optimizer_type"] = "adamw"
                    adamw_groups.append(g)
                else:
                    adamw_groups.append({
                        "params": list(group) if hasattr(group, "__iter__") and not isinstance(group, torch.Tensor) else [group],
                        "lr": adamw_lr, "betas": adamw_betas,
                        "weight_decay": adamw_wd, "_optimizer_type": "adamw",
                    })

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(muon_groups + adamw_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("_optimizer_type", "muon") == "muon":
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group):
        lr, mu = group["lr"], group["momentum"]
        nesterov, ns_steps, wd = group["nesterov"], group["ns_steps"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buf = state["momentum_buffer"]
            buf.mul_(mu).add_(grad)
            g = grad.add(buf, alpha=mu) if nesterov else buf.clone()
            g = zeropower_via_newtonschulz5(g, steps=ns_steps)
            scale = max(g.shape[0], g.shape[1]) ** 0.5 * 0.2
            g.mul_(scale)
            if wd > 0:
                p.data.mul_(1 - lr * wd)
            p.data.add_(g, alpha=-lr)

    def _adamw_step(self, group):
        lr = group["lr"]
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        wd, eps = group.get("weight_decay", 0.0), group.get("eps", 1e-8)
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            bc1 = 1 - beta1 ** state["step"]
            bc2 = 1 - beta2 ** state["step"]
            if wd > 0:
                p.data.mul_(1 - lr * wd)
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
            p.data.addcdiv_(exp_avg, denom, value=-lr / bc1)


_ADAMW_FORCE_PATTERNS = (
    "ssm", "mamba", "conv", "scan", "A_log", "dt_", "D_param",
    "target", "film", "embedding", "embed", "output.weight",
    "log_gamma", "log_eta", "log_beta", "omega", "gamma_param",
    "decay", "conductor", "engram", "meta_token",
)


def split_params_for_muon(model):
    embedding_params = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters():
                embedding_params.add(id(p))
    muon_params, adamw_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in embedding_params:
            adamw_params.append((name, param))
        elif any(pat in name for pat in _ADAMW_FORCE_PATTERNS):
            adamw_params.append((name, param))
        elif param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append((name, param))
    return muon_params, adamw_params


# ===========================================================================
# Data Loading (.bin pre-tokenized)
# ===========================================================================

class PreTokenizedDataset(Dataset):
    """Load pre-tokenized .bin file (uint16 numpy array)."""

    def __init__(self, bin_path: str, block_size: int = 256):
        self.block_size = block_size
        raw = np.fromfile(bin_path, dtype=np.uint16)
        n_tokens = len(raw)
        n_chunks = n_tokens // (block_size + 1)
        tokens = raw[: n_chunks * (block_size + 1)].astype(np.int64)
        self.tokens = torch.from_numpy(tokens).view(n_chunks, block_size + 1)
        print(f"Dataset: {n_tokens:,} tokens -> {n_chunks:,} chunks of {block_size}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        chunk = self.tokens[idx]
        return chunk[:-1], chunk[1:]


# ===========================================================================
# LR Scheduler
# ===========================================================================

def build_scheduler(optimizer, total_steps, warmup_steps=100, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===========================================================================
# Throughput Tracker
# ===========================================================================

class ThroughputTracker:
    def __init__(self, model_params: int, peak_flops: float = 312e12):
        self.model_params = model_params
        self.peak_flops = peak_flops
        self.total_tokens = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def update(self, tokens: int):
        self.total_tokens += tokens

    def get_stats(self):
        elapsed = time.time() - self.start_time if self.start_time else 1.0
        tok_s = self.total_tokens / elapsed if elapsed > 0 else 0
        achieved = self.total_tokens * 6 * self.model_params / elapsed if elapsed > 0 else 0
        mfu = achieved / self.peak_flops if self.peak_flops > 0 else 0
        return {"tok_s": tok_s, "mfu": mfu}


def compute_bpb(ce_loss: float) -> float:
    return (ce_loss / math.log(2)) / 3.6


# ===========================================================================
# Model Loading
# ===========================================================================

def load_model_from_file(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls()


# ===========================================================================
# Optimizer Builder
# ===========================================================================

def build_optimizer(model, base_lr=0.0012, muon_lr=0.005, weight_decay=0.1):
    muon_params, adamw_named = split_params_for_muon(model)
    adamw_groups = []
    for name, param in adamw_named:
        if "norm" in name or (name.endswith(".bias") and "decay" not in name):
            adamw_groups.append({"params": [param], "lr": base_lr, "weight_decay": 0})
        else:
            adamw_groups.append({"params": [param], "lr": base_lr, "weight_decay": weight_decay})
    n_muon = len(muon_params)
    n_adamw = sum(len(g["params"]) for g in adamw_groups)
    print(f"Muon optimizer: {n_muon} params via Muon (lr={muon_lr}), {n_adamw} params via AdamW (lr={base_lr})")
    return Muon(
        muon_params=[{"params": muon_params}], lr=muon_lr, weight_decay=0.01,
        adamw_params=adamw_groups, adamw_lr=base_lr,
    )


# ===========================================================================
# Checkpoint
# ===========================================================================

def save_checkpoint(model, optimizer, step, checkpoint_dir, total_tokens=0):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_tokens": total_tokens,
    }, path)
    print(f"Checkpoint saved: {path}")


# ===========================================================================
# Training Loop
# ===========================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model_from_file(args.model_file, args.class_name)
    model = model.to(device)

    # Resume from checkpoint (Approach B: weights only, fresh optimizer)
    if args.resume_from:
        print(f"Loading checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
            print("  Warning: loaded with strict=False")
        prev_step = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0
        print(f"  Resumed from step {prev_step}, fresh optimizer (Approach B)")
        del ckpt, state_dict

    model.train()

    # Compile
    if args.compile:
        print("Compiling model with torch.compile (reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")

    # Data
    dataset = PreTokenizedDataset(args.dataset, block_size=args.block_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # Optimizer + scheduler
    optimizer = build_optimizer(model, base_lr=args.lr, muon_lr=args.muon_lr)
    total_steps = len(dataloader) * args.epochs // args.accum_steps
    scheduler = build_scheduler(optimizer, total_steps)

    # Loss
    ce_loss_fn = nn.CrossEntropyLoss()

    # Metrics
    n_params = sum(p.numel() for p in model.parameters())
    throughput = ThroughputTracker(n_params, peak_flops=args.peak_tflops * 1e12)
    throughput.start()

    # bf16 AMP (no GradScaler needed for bf16)
    amp_dtype = torch.bfloat16

    eff_batch = args.batch_size * args.accum_steps
    print(f"Training: {n_params / 1e6:.1f}M params, batch={args.batch_size}x{args.accum_steps}={eff_batch}, "
          f"block={args.block_size}, lr={args.lr}")
    print(f"Steps/epoch: {len(dataloader)}, optimizer steps: {total_steps}, epochs: {args.epochs}")

    global_step = 0
    total_tokens = 0
    running_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()
    deadline = start_time + args.time_budget * 60 if args.time_budget > 0 else None

    ckpt_every = args.checkpoint_interval or (args.log_interval * 10)

    try:
        for epoch in range(args.epochs):
            for batch_idx, (input_ids, targets) in enumerate(dataloader):
                if deadline and time.time() > deadline:
                    print(f"Time budget reached at step {global_step}")
                    break

                input_ids = input_ids.to(device)
                targets = targets.to(device)

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    output = model(input_ids, targets=targets)
                    if isinstance(output, torch.Tensor) and output.dim() == 0:
                        loss = output / args.accum_steps
                    else:
                        logits = output
                        loss = ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1)) / args.accum_steps

                loss.backward()

                tokens_in_batch = input_ids.numel()
                total_tokens += tokens_in_batch
                throughput.update(tokens_in_batch)
                running_loss += loss.item() * args.accum_steps

                if (batch_idx + 1) % args.accum_steps == 0:
                    try:
                        torch.compiler.cudagraph_mark_step_begin()
                    except (AttributeError, RuntimeError):
                        pass

                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        print(f"[step {global_step}] Non-finite grad norm, skipping")

                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if global_step % args.log_interval == 0:
                        tp = throughput.get_stats()
                        avg_loss = running_loss / args.log_interval
                        bpb = compute_bpb(avg_loss)
                        lr = scheduler.get_last_lr()[0]
                        mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                        print(
                            f"[step {global_step:>6d}] "
                            f"loss={avg_loss:.4f} bpb={bpb:.3f} lr={lr:.2e} "
                            f"grad={grad_norm:.2f} tok/s={tp['tok_s']:,.0f} "
                            f"mfu={tp['mfu']:.1%} mem={mem_gb:.1f}GB"
                        )
                        running_loss = 0.0
                        best_loss = min(best_loss, avg_loss)

                    if args.checkpoint_dir and global_step % ckpt_every == 0:
                        save_checkpoint(model, optimizer, global_step, args.checkpoint_dir, total_tokens)
            else:
                continue
            break

    except KeyboardInterrupt:
        print(f"\nInterrupted at step {global_step}")

    # Final checkpoint
    if args.checkpoint_dir and global_step > 0:
        save_checkpoint(model, optimizer, global_step, args.checkpoint_dir, total_tokens)

    elapsed = time.time() - start_time
    print(f"\nDone: {global_step} steps, {total_tokens:,} tokens in {elapsed:.0f}s "
          f"({total_tokens / elapsed:,.0f} tok/s), best loss={best_loss:.4f}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="NVIDIA GPU Training (standalone)")
    parser.add_argument("--model-file", required=True, help="Path to model .py file")
    parser.add_argument("--class-name", default="ArgusPrime", help="Model class name")
    parser.add_argument("--dataset", required=True, help="Pre-tokenized .bin file")
    parser.add_argument("--resume-from", default=None, help="Checkpoint .pt for continued pre-training")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for saving checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Steps between checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0012, help="Base LR for AdamW params")
    parser.add_argument("--muon-lr", type=float, default=0.005, help="LR for Muon params")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true", default=True, help="Use torch.compile (default: on)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--time-budget", type=float, default=0, help="Time budget in minutes (0=unlimited)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--peak-tflops", type=float, default=312, help="Peak TFLOPS for MFU (A100=312, H100=989, 4090=165)")
    args = parser.parse_args()

    if args.no_compile:
        args.compile = False

    train(args)


if __name__ == "__main__":
    main()
