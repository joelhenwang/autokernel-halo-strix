"""Distributed Data Parallel training across 2 Strix Halo machines via TB4.

Each machine holds a full model copy, processes different data batches, and
synchronizes gradients over Thunderbolt 4. Uses manual async allreduce with
fp16 compression to overlap communication with compute.

Usage (run on BOTH machines simultaneously):
  # Machine 0 (master):
  source scripts/ddp_env.sh
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    scripts/train_ddp.py --model models/argus_prime.py --class-name ArgusPrime \
    --dataset datasets/common_crawl_sample.bin --epochs 2 --accum-steps 8 ...

  # Machine 1 (worker):
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    scripts/train_ddp.py --model models/argus_prime.py --class-name ArgusPrime \
    --dataset datasets/common_crawl_sample.bin --epochs 2 --accum-steps 8 ...
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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# ===========================================================================
# Muon Optimizer (inlined from halo_training/muon.py)
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
        if dist.is_initialized():
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
# Stability Guard: auto-recovery from NaN/loss spikes
# ===========================================================================

class StabilityGuard:
    """Detects training instability and auto-recovers from last checkpoint.

    Three detection mechanisms:
    1. NaN loss — immediate rollback
    2. Loss spike — loss > spike_factor * EMA triggers rollback
    3. Parameter NaN — periodic weight scan catches silent corruption

    On trigger: reload last checkpoint, reduce LR by decay_factor, continue.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_rollbacks: int = 5,
        spike_factor: float = 2.0,
        lr_decay_on_rollback: float = 0.5,
        ema_alpha: float = 0.99,
        param_check_interval: int = 500,
        rank: int = 0,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_rollbacks = max_rollbacks
        self.spike_factor = spike_factor
        self.lr_decay_on_rollback = lr_decay_on_rollback
        self.ema_alpha = ema_alpha
        self.param_check_interval = param_check_interval
        self.rank = rank

        self.loss_ema = None
        self.rollback_count = 0
        self.last_good_step = 0
        self._steps_seen = 0
        self._spike_warmup = 2000

    def _find_latest_checkpoint(self, before_step: int = None):
        """Find the most recent valid checkpoint."""
        ckpt_dir = self.checkpoint_dir
        if not os.path.exists(ckpt_dir):
            return None
        checkpoints = []
        for f in os.listdir(ckpt_dir):
            if f.startswith("step_") and f.endswith(".pt"):
                try:
                    step = int(f.replace("step_", "").replace(".pt", ""))
                except ValueError:
                    step = 0
                if before_step is None or step < before_step:
                    checkpoints.append((step, os.path.join(ckpt_dir, f)))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0]  # (step, path)

    def check_loss(self, loss_val: float, step: int) -> bool:
        """Returns True if loss is healthy, False if rollback needed."""
        self._steps_seen += 1

        if math.isnan(loss_val) or math.isinf(loss_val):
            if self.rank == 0:
                print(f"  [StabilityGuard] NaN/Inf loss at step {step}")
            return False

        if self.loss_ema is None:
            self.loss_ema = loss_val
            return True

        self.loss_ema = self.ema_alpha * self.loss_ema + (1 - self.ema_alpha) * loss_val

        # Skip spike detection during warmup (loss is volatile early)
        if self._steps_seen < self._spike_warmup:
            self.last_good_step = step
            return True

        if loss_val > self.spike_factor * self.loss_ema:
            if self.rank == 0:
                print(f"  [StabilityGuard] Loss spike at step {step}: "
                      f"{loss_val:.4f} > {self.spike_factor}x EMA {self.loss_ema:.4f}")
            return False

        self.last_good_step = step
        return True

    def check_params(self, model: nn.Module, step: int) -> bool:
        """Periodic parameter NaN check. Returns True if healthy."""
        if step % self.param_check_interval != 0:
            return True
        for name, p in model.named_parameters():
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                if self.rank == 0:
                    print(f"  [StabilityGuard] NaN/Inf in param '{name}' at step {step}")
                return False
        return True

    def rollback(self, model, optimizer, device):
        """Reload last good checkpoint and reduce LR. Returns (step, success)."""
        if self.rollback_count >= self.max_rollbacks:
            if self.rank == 0:
                print(f"  [StabilityGuard] Max rollbacks ({self.max_rollbacks}) reached, aborting")
            return -1, False

        latest = self._find_latest_checkpoint(before_step=None)
        if latest is None:
            if self.rank == 0:
                print("  [StabilityGuard] No checkpoint found for rollback")
            return -1, False

        ckpt_step, ckpt_path = latest
        if self.rank == 0:
            print(f"  [StabilityGuard] Rolling back to step {ckpt_step} from {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        raw = model.module if hasattr(model, "module") else model
        raw = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
        raw.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                if self.rank == 0:
                    print("  [StabilityGuard] Could not restore optimizer, using fresh state")

        # Reduce LR
        decay = self.lr_decay_on_rollback
        for pg in optimizer.param_groups:
            old_lr = pg["lr"]
            pg["lr"] = old_lr * decay
            if self.rank == 0 and pg is optimizer.param_groups[0]:
                print(f"  [StabilityGuard] LR reduced: {old_lr:.2e} -> {pg['lr']:.2e}")

        self.rollback_count += 1
        self.loss_ema = None  # reset EMA after rollback

        if self.rank == 0:
            print(f"  [StabilityGuard] Rollback #{self.rollback_count} complete, "
                  f"resuming from step {ckpt_step}")

        del ckpt
        return ckpt_step, True


# ===========================================================================
# Manual Async Allreduce with fp16 Compression
# ===========================================================================

def compress_grads_fp16(model):
    """Cast all gradients to fp16 in-place to halve allreduce payload."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = p.grad.data.half()


def decompress_grads_fp32(model):
    """Cast gradients back to fp32 for optimizer step."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def allreduce_grads_async(model):
    """Launch async allreduce on all gradients. Returns handles."""
    handles = []
    for p in model.parameters():
        if p.grad is not None:
            handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            handles.append(handle)
    return handles


def allreduce_grads_sync(model):
    """Synchronous allreduce on all gradients."""
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)


def average_grads(model, world_size):
    """Divide all gradients by world_size (after allreduce SUM)."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data /= world_size


_consecutive_skips = 0

def _complete_step(model, optimizer, scaler, scheduler, world_size, max_grad_norm, rank):
    """Unscale, clip, step optimizer, update scaler. Returns grad_norm."""
    global _consecutive_skips
    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if torch.isfinite(grad_norm):
        scaler.step(optimizer)
        _consecutive_skips = 0
    else:
        _consecutive_skips += 1
        if rank == 0:
            print(f"  Non-finite grad norm, skipping step (consecutive: {_consecutive_skips})")
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    return grad_norm


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
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--optimize-kernels", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--time-budget", type=float, default=0, help="Minutes, 0=unlimited")
    parser.add_argument("--backend", default="gloo", choices=["nccl", "gloo"])
    parser.add_argument("--no-async", action="store_true", help="Disable async allreduce overlap")
    parser.add_argument("--no-fp16-compress", action="store_true", help="Disable fp16 grad compression")
    args = parser.parse_args()

    use_async = not args.no_async
    use_fp16 = not args.no_fp16_compress

    # --- DDP init ---
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    if rank == 0:
        print(f"DDP: world_size={world_size}, backend={args.backend}, "
              f"async={use_async}, fp16_compress={use_fp16}")

    # --- Model ---
    sys.path.insert(0, ".")
    model = load_model_from_file(args.model, args.class_name)
    model = model.to(device)

    # autokernel BEFORE checkpoint load (checkpoint has fused QKV keys)
    if args.optimize_kernels:
        try:
            import autokernel
            model = autokernel.optimize(model, training=True)
            if rank == 0:
                print("autokernel optimizations applied")
        except Exception as e:
            if rank == 0:
                print(f"autokernel skipped: {e}")

    # Resume (Approach B: weights only, fresh optimizer)
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
        if rank == 0:
            print(f"  Resumed from step {ckpt.get('step', '?')}, fresh optimizer")
        del ckpt, state_dict

    model.train()

    # DDP wrapper — find_unused_parameters needed for TTT conditional paths
    model = DDP(model, device_ids=[0], find_unused_parameters=True,
                gradient_as_bucket_view=True)

    # torch.compile AFTER DDP
    if args.compile:
        raw_for_compile = model.module if hasattr(model, "module") else model
        if hasattr(raw_for_compile, "compile_zones"):
            if rank == 0:
                print("Compiling model (per-zone for looped model)...")
            raw_for_compile.compile_zones()
        else:
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
    raw_model = model
    # Unwrap compile -> DDP -> raw
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    if hasattr(raw_model, "module"):
        raw_model = raw_model.module

    optimizer = build_muon_optimizer(raw_model, base_lr=args.lr, muon_lr=args.muon_lr)
    total_steps = len(dataloader) * args.epochs // args.accum_steps
    scheduler = build_scheduler(optimizer, total_steps)

    # --- Loss + AMP ---
    ce_loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    # --- Metrics ---
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

    # --- Stability Guard ---
    guard = StabilityGuard(
        checkpoint_dir=args.checkpoint_dir or "checkpoints/ddp",
        max_rollbacks=5,
        spike_factor=2.0,
        lr_decay_on_rollback=0.5,
        param_check_interval=500,
        rank=rank,
    )
    # Seed the guard with the resume checkpoint as fallback
    if args.resume_from and args.checkpoint_dir:
        fallback_dir = args.checkpoint_dir
        fallback_path = os.path.join(fallback_dir, "step_0_fallback.pt")
        if not os.path.exists(fallback_path):
            import shutil
            os.makedirs(fallback_dir, exist_ok=True)
            shutil.copy2(args.resume_from, fallback_path)
            if rank == 0:
                print(f"StabilityGuard: seeded fallback checkpoint from {args.resume_from}")

    # --- Training loop ---
    global_step = 0
    total_tokens = 0
    running_loss = 0.0
    step_loss = 0.0  # accumulates across microsteps within one optimizer step
    best_loss = float("inf")
    start_time = time.time()
    deadline = start_time + args.time_budget * 60 if args.time_budget > 0 else None
    ckpt_every = args.checkpoint_interval or (args.log_interval * 10)
    last_grad_norm = torch.tensor(0.0)

    # Async state: pending allreduce handles from previous optimizer step
    pending_handles = None

    _rollback_restart = False
    try:
        for epoch in range(args.epochs):
            if _rollback_restart:
                _rollback_restart = False
            sampler.set_epoch(epoch)
            for batch_idx, (input_ids, targets) in enumerate(dataloader):
                if deadline and time.time() > deadline:
                    if rank == 0:
                        print(f"Time budget reached at step {global_step}")
                    break

                input_ids = input_ids.to(device)
                targets = targets.to(device)
                is_last_microstep = (batch_idx + 1) % args.accum_steps == 0

                # All microsteps use no_sync — we do manual allreduce
                with model.no_sync():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        output = model(input_ids, targets=targets)
                        if isinstance(output, torch.Tensor) and output.dim() == 0:
                            loss = output / args.accum_steps
                        else:
                            loss = ce_loss_fn(
                                output.view(-1, output.size(-1)), targets.view(-1)
                            ) / args.accum_steps

                    scaler.scale(loss).backward()

                total_tokens += input_ids.numel()
                loss_val = loss.item()
                if not math.isnan(loss_val):
                    step_loss += loss_val
                else:
                    # Transient NaN in microstep — GradScaler will handle via inf grads
                    if rank == 0:
                        print(f"  [StabilityGuard] NaN microstep loss at batch {batch_idx}, continuing")

                if not is_last_microstep:
                    continue

                # === End of optimizer step boundary ===

                # 1. Complete PREVIOUS step's pending allreduce (if any)
                if pending_handles is not None:
                    for h in pending_handles:
                        h.wait()
                    average_grads(model, world_size)
                    if use_fp16:
                        decompress_grads_fp32(model)
                    last_grad_norm = _complete_step(
                        model, optimizer, scaler, scheduler,
                        world_size, args.max_grad_norm, rank,
                    )
                    global_step += 1

                    # Logging
                    if rank == 0 and global_step % args.log_interval == 0:
                        elapsed = time.time() - start_time
                        global_tokens = total_tokens * world_size
                        tok_s = global_tokens / elapsed if elapsed > 0 else 0
                        achieved = global_tokens * 6 * n_params / elapsed if elapsed > 0 else 0
                        mfu = achieved / 118.8e12
                        avg_loss = running_loss / args.log_interval
                        bpb = compute_bpb(avg_loss)
                        lr = scheduler.get_last_lr()[0]
                        mem_gb = torch.cuda.max_memory_allocated() / 1e9
                        print(
                            f"[step {global_step:>6d}] "
                            f"loss={avg_loss:.4f} bpb={bpb:.3f} lr={lr:.2e} "
                            f"grad={last_grad_norm:.2f} tok/s={tok_s:,.0f} "
                            f"mfu={mfu:.1%} mem={mem_gb:.1f}GB"
                        )
                        if log_file:
                            with open(log_file, "a") as f:
                                f.write(json.dumps({
                                    "step": global_step, "loss": avg_loss,
                                    "bpb": bpb, "lr": lr, "tok_s": tok_s,
                                    "mfu": mfu, "mem_gb": mem_gb,
                                    "grad_norm": last_grad_norm.item() if hasattr(last_grad_norm, "item") else last_grad_norm,
                                }) + "\n")
                        running_loss = 0.0
                        best_loss = min(best_loss, avg_loss)

                    # Checkpoint
                    if rank == 0 and args.checkpoint_dir and global_step % ckpt_every == 0:
                        save_checkpoint(model, optimizer, global_step,
                                        args.checkpoint_dir, total_tokens * world_size)

                # 2. Record loss for THIS step + stability checks
                running_loss += step_loss
                grad_is_nan = hasattr(last_grad_norm, 'item') and not torch.isfinite(last_grad_norm)
                too_many_skips = _consecutive_skips >= 5
                guard_active = guard._steps_seen >= 10
                if guard_active and (
                    not guard.check_loss(step_loss, global_step) or
                    too_many_skips or
                    not guard.check_params(model, global_step)
                ):
                    if too_many_skips and rank == 0:
                        print(f"  [StabilityGuard] {_consecutive_skips} consecutive grad skips, rolling back")
                    optimizer.zero_grad(set_to_none=True)
                    if pending_handles is not None:
                        for h in pending_handles:
                            h.wait()
                        pending_handles = None
                    # Save emergency checkpoint before rollback
                    if rank == 0 and args.checkpoint_dir:
                        save_checkpoint(model, optimizer, global_step,
                                        args.checkpoint_dir, total_tokens * world_size)
                    rollback_step, ok = guard.rollback(model, optimizer, device)
                    if not ok:
                        raise RuntimeError("StabilityGuard: unrecoverable instability")
                    global_step = rollback_step
                    step_loss = 0.0
                    running_loss = 0.0
                    _rollback_restart = True
                    dist.barrier()
                    break
                step_loss = 0.0

                # 3. Launch allreduce for THIS step's gradients
                if use_fp16:
                    compress_grads_fp16(model)

                if use_async:
                    pending_handles = allreduce_grads_async(model)
                    # Don't wait — next microsteps run while allreduce transfers
                else:
                    allreduce_grads_sync(model)
                    average_grads(model, world_size)
                    if use_fp16:
                        decompress_grads_fp32(model)
                    last_grad_norm = _complete_step(
                        model, optimizer, scaler, scheduler,
                        world_size, args.max_grad_norm, rank,
                    )
                    global_step += 1
                    pending_handles = None

            else:
                continue
            if _rollback_restart:
                continue
            break

        # Flush final pending allreduce
        if pending_handles is not None:
            for h in pending_handles:
                h.wait()
            average_grads(model, world_size)
            if use_fp16:
                decompress_grads_fp32(model)
            _complete_step(model, optimizer, scaler, scheduler,
                           world_size, args.max_grad_norm, rank)
            global_step += 1

    except KeyboardInterrupt:
        if rank == 0:
            print(f"\nInterrupted at step {global_step}")

    # Final checkpoint
    if rank == 0 and args.checkpoint_dir:
        save_checkpoint(model, optimizer, global_step,
                        args.checkpoint_dir, total_tokens * world_size)

    if rank == 0:
        elapsed = time.time() - start_time
        global_tokens = total_tokens * world_size
        print(f"\nDone: {global_step} steps, {global_tokens:,} tokens in {elapsed:.0f}s "
              f"({global_tokens / elapsed:,.0f} tok/s), best loss={best_loss:.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
