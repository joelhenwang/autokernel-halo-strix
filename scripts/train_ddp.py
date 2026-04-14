"""Distributed Data Parallel training across 2 Strix Halo machines via TB4.

Each machine holds a full model copy, processes different data batches, and
synchronizes gradients over Thunderbolt 4 using RCCL (ROCm's NCCL).

Usage (run on BOTH machines simultaneously):
  # Machine 0 (master):
  source ~/ddp_env.sh
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    scripts/train_ddp.py --model models/argus_prime.py --class-name ArgusPrime \
    --dataset datasets/common_crawl_sample.bin --epochs 2 ...

  # Machine 1 (worker):
  source ~/ddp_env.sh
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    scripts/train_ddp.py --model models/argus_prime.py --class-name ArgusPrime \
    --dataset datasets/common_crawl_sample.bin --epochs 2 ...

Paths are resolved relative to each machine's cwd — so model, dataset, and
checkpoint paths can differ as long as the CLI args point to the right place
on each machine.
"""

import argparse
import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# ===========================================================================
# Muon Optimizer (inlined — identical to halo_training/muon.py)
# ===========================================================================

def zeropower_via_newtonschulz5(G, steps=5):
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
    def __init__(self, muon_params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.01, adamw_params=None,
                 adamw_lr=8e-4, adamw_betas=(0.9, 0.95), adamw_wd=0.0):
        if isinstance(muon_params, dict):
            muon_params = [muon_params]
        muon_groups = []
        for group in muon_params:
            if isinstance(group, dict):
                g = dict(group)
                g.setdefault("lr", lr); g.setdefault("momentum", momentum)
                g.setdefault("nesterov", nesterov); g.setdefault("ns_steps", ns_steps)
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
                    g.setdefault("lr", adamw_lr); g.setdefault("betas", adamw_betas)
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
            g.mul_(max(g.shape[0], g.shape[1]) ** 0.5 * 0.2)
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
            bc1, bc2 = 1 - beta1 ** state["step"], 1 - beta2 ** state["step"]
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


def build_muon_optimizer(model, base_lr=0.0012, muon_lr=0.005, weight_decay=0.1):
    muon_params, adamw_named = split_params_for_muon(model)
    adamw_groups = []
    for name, param in adamw_named:
        if "norm" in name or (name.endswith(".bias") and "decay" not in name):
            adamw_groups.append({"params": [param], "lr": base_lr, "weight_decay": 0})
        else:
            adamw_groups.append({"params": [param], "lr": base_lr, "weight_decay": weight_decay})
    n_muon = len(muon_params)
    n_adamw = sum(len(g["params"]) for g in adamw_groups)
    print(f"[rank {dist.get_rank()}] Muon: {n_muon} params via Muon (lr={muon_lr}), {n_adamw} via AdamW (lr={base_lr})")
    return Muon(
        muon_params=[{"params": muon_params}], lr=muon_lr, weight_decay=0.01,
        adamw_params=adamw_groups, adamw_lr=base_lr,
    )


# ===========================================================================
# Data Loading
# ===========================================================================

class PreTokenizedDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int = 256):
        self.block_size = block_size
        raw = np.fromfile(bin_path, dtype=np.uint16)
        n_tokens = len(raw)
        n_chunks = n_tokens // (block_size + 1)
        tokens = raw[: n_chunks * (block_size + 1)].astype(np.int64)
        self.tokens = torch.from_numpy(tokens).view(n_chunks, block_size + 1)
        print(f"[rank {dist.get_rank()}] Dataset: {n_tokens:,} tokens -> {n_chunks:,} chunks of {block_size}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        chunk = self.tokens[idx]
        return chunk[:-1], chunk[1:]


# ===========================================================================
# Helpers
# ===========================================================================

def build_scheduler(optimizer, total_steps, warmup_steps=100, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_model_from_file(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)()


def save_checkpoint(model, optimizer, step, checkpoint_dir, total_tokens=0):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    # Unwrap DDP -> compiled -> raw model
    raw = model.module if hasattr(model, "module") else model
    raw = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
    torch.save({
        "step": step,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_tokens": total_tokens,
    }, path)
    print(f"[rank 0] Checkpoint saved: {path}")


def compute_bpb(ce_loss: float) -> float:
    return (ce_loss / math.log(2)) / 3.6


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="DDP Training (2x Strix Halo)")
    parser.add_argument("--model", required=True, help="Path to model .py file")
    parser.add_argument("--class-name", required=True, help="Model class name")
    parser.add_argument("--dataset", required=True, help="Pre-tokenized .bin file")
    parser.add_argument("--resume-from", default=None, help="Checkpoint for CPT (weights only)")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0012)
    parser.add_argument("--muon-lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--optimize-kernels", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--time-budget", type=float, default=0, help="Minutes, 0=unlimited")
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"],
                        help="DDP backend: nccl (RCCL, fast) or gloo (CPU, fallback)")
    args = parser.parse_args()

    # --- DDP init ---
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(0)  # 1 GPU per machine
    device = torch.device("cuda", 0)

    print(f"[rank {rank}] Initialized DDP: world_size={world_size}, device={device}")

    # --- Model ---
    sys.path.insert(0, ".")
    model = load_model_from_file(args.model, args.class_name)
    model = model.to(device)

    # Resume (Approach B: weights only)
    if args.resume_from:
        if rank == 0:
            print(f"Loading checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print("  Warning: loaded with strict=False")
        prev_step = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0
        if rank == 0:
            print(f"  Resumed from step {prev_step}, fresh optimizer (Approach B)")
        del ckpt, state_dict

    model.train()

    # autokernel (optional, ROCm only)
    if args.optimize_kernels:
        try:
            import autokernel
            model = autokernel.optimize(model, training=True)
            if rank == 0:
                print("autokernel optimizations applied")
        except Exception as e:
            if rank == 0:
                print(f"autokernel skipped: {e}")

    # Wrap in DDP FIRST (needs to see raw parameters before compile)
    model = DDP(model, device_ids=[0])

    # fp16 gradient compression — halves sync payload (672 MB -> 336 MB)
    # Only available with NCCL backend
    if args.backend == "nccl":
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
        if rank == 0:
            print("Registered fp16 gradient compression hook")

    # torch.compile AFTER DDP wrapping
    if args.compile:
        compile_mode = "default" if args.optimize_kernels else "reduce-overhead"
        if rank == 0:
            print(f"Compiling model ({compile_mode})...")
        model = torch.compile(model, mode=compile_mode)

    # --- Data ---
    dataset = PreTokenizedDataset(args.dataset, block_size=args.block_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=False, drop_last=True,
    )

    # --- Optimizer + Scheduler ---
    # Build optimizer on the unwrapped model (DDP .module)
    raw_model = model.module._orig_mod if hasattr(model.module, "_orig_mod") else model.module
    optimizer = build_muon_optimizer(raw_model, base_lr=args.lr, muon_lr=args.muon_lr)

    # total_steps accounts for world_size: each machine sees len(dataloader) batches
    total_steps = len(dataloader) * args.epochs // args.accum_steps
    scheduler = build_scheduler(optimizer, total_steps)

    # --- Loss + AMP ---
    ce_loss_fn = nn.CrossEntropyLoss()
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda")

    # --- Metrics (rank 0 only) ---
    n_params = sum(p.numel() for p in raw_model.parameters())
    log_file = None
    if rank == 0 and args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        log_file = os.path.join(args.checkpoint_dir, "train_log.jsonl")

    eff_batch = args.batch_size * args.accum_steps * world_size
    if rank == 0:
        print(f"Training: {n_params / 1e6:.1f}M params, "
              f"batch={args.batch_size}x{args.accum_steps}x{world_size}={eff_batch}, "
              f"block={args.block_size}, lr={args.lr}")
        print(f"Steps/epoch: {len(dataloader)}, optimizer steps: {total_steps}, epochs: {args.epochs}")

    # --- Training loop ---
    global_step = 0
    total_tokens = 0
    running_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()
    deadline = start_time + args.time_budget * 60 if args.time_budget > 0 else None
    ckpt_every = args.checkpoint_interval or (args.log_interval * 10)

    try:
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)  # shuffle differently each epoch
            for batch_idx, (input_ids, targets) in enumerate(dataloader):
                if deadline and time.time() > deadline:
                    if rank == 0:
                        print(f"Time budget reached at step {global_step}")
                    break

                input_ids = input_ids.to(device)
                targets = targets.to(device)

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    output = model(input_ids, targets=targets)
                    if isinstance(output, torch.Tensor) and output.dim() == 0:
                        loss = output / args.accum_steps
                    else:
                        loss = ce_loss_fn(
                            output.view(-1, output.size(-1)), targets.view(-1)
                        ) / args.accum_steps

                scaler.scale(loss).backward()

                tokens_in_batch = input_ids.numel()
                total_tokens += tokens_in_batch
                running_loss += loss.item() * args.accum_steps

                if (batch_idx + 1) % args.accum_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if torch.isfinite(grad_norm):
                        scaler.step(optimizer)
                    else:
                        if rank == 0:
                            print(f"[step {global_step}] Non-finite grad norm, skipping")

                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    # Logging (rank 0 only)
                    if rank == 0 and global_step % args.log_interval == 0:
                        elapsed = time.time() - start_time
                        # total_tokens is per-machine; multiply by world_size for global
                        global_tokens = total_tokens * world_size
                        tok_s = global_tokens / elapsed if elapsed > 0 else 0
                        achieved = global_tokens * 6 * n_params / elapsed if elapsed > 0 else 0
                        mfu = achieved / 118.8e12  # 2x Strix Halo = 2 * 59.4 TFLOPS
                        avg_loss = running_loss / args.log_interval
                        bpb = compute_bpb(avg_loss)
                        lr = scheduler.get_last_lr()[0]
                        mem_gb = torch.cuda.max_memory_allocated() / 1e9

                        line = (
                            f"[step {global_step:>6d}] "
                            f"loss={avg_loss:.4f} bpb={bpb:.3f} lr={lr:.2e} "
                            f"grad={grad_norm:.2f} tok/s={tok_s:,.0f} "
                            f"mfu={mfu:.1%} mem={mem_gb:.1f}GB"
                        )
                        print(line)

                        if log_file:
                            import json
                            with open(log_file, "a") as f:
                                f.write(json.dumps({
                                    "step": global_step, "loss": avg_loss,
                                    "bpb": bpb, "lr": lr, "tok_s": tok_s,
                                    "mfu": mfu, "mem_gb": mem_gb,
                                    "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                                }) + "\n")

                        running_loss = 0.0
                        best_loss = min(best_loss, avg_loss)

                    # Checkpoint (rank 0 only)
                    if rank == 0 and args.checkpoint_dir and global_step % ckpt_every == 0:
                        save_checkpoint(model, optimizer, global_step,
                                        args.checkpoint_dir, total_tokens * world_size)
            else:
                continue
            break

    except KeyboardInterrupt:
        if rank == 0:
            print(f"\nInterrupted at step {global_step}")

    # Final
    if rank == 0:
        elapsed = time.time() - start_time
        global_tokens = total_tokens * world_size
        print(f"\nDone: {global_step} steps, {global_tokens:,} tokens in {elapsed:.0f}s "
              f"({global_tokens / elapsed:,.0f} tok/s), best loss={best_loss:.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
