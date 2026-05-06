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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# ===========================================================================
# Muon Optimizer — imported from halo_training/muon.py (single source of truth)
# ===========================================================================

from halo_training.muon import Muon, split_params_for_muon


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
        ns_dtype=torch.float16,
    )


# ===========================================================================
# Data Loading — imported from halo_training/data.py (single source of truth)
# ===========================================================================

from halo_training.data import PreTokenizedDataset


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
    # Strip per-layer compile wrappers. compile_zones() replaces each ModuleList
    # entry with torch.compile(layer), inserting ._orig_mod. into state dict keys.
    # The outer raw._orig_mod unwrap above only handles a single top-level compile.
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in raw.state_dict().items()}
    torch.save({
        "step": step,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "total_tokens": total_tokens,
    }, path)
    print(f"[rank 0] Checkpoint saved: {path}")
    return path


def spawn_auto_eval(checkpoint_path: str, model_path: str, class_name: str) -> None:
    """Fire scripts/eval_checkpoint.py as a detached subprocess post-save.

    Sprint 2 --auto-eval hook. Runs independently of the trainer so a failing
    evaluator never crashes training. STDOUT/STDERR are captured to
    ``<checkpoint>.eval.log`` for post-hoc inspection. A warning is emitted
    conspicuously to rank0 stdout so eval-job failures are visible in train
    logs without requiring log-file spelunking.
    """
    import subprocess

    log_path = checkpoint_path + ".eval.log"
    cmd = [
        sys.executable, "scripts/eval_checkpoint.py",
        "--checkpoint", checkpoint_path,
        "--model", model_path,
        "--class-name", class_name,
    ]
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            # start_new_session detaches so it won't be killed when the trainer
            # terminates via SIGINT (unless trainer gets SIGKILL, which is fine
            # — the eval was started speculatively).
            subprocess.Popen(
                cmd,
                stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"[rank 0] >>> auto-eval spawned for {checkpoint_path} (log: {log_path})")
    except Exception as exc:  # noqa: BLE001
        # Conspicuous warning per Sprint 2 spec: log + warn, do NOT block training.
        print(f"[rank 0] WARNING: auto-eval spawn FAILED for {checkpoint_path}")
        print(f"[rank 0] WARNING: {type(exc).__name__}: {exc}")
        print(f"[rank 0] WARNING: training continuing; investigate offline")


def compute_bpb(ce_loss: float) -> float:
    return (ce_loss / math.log(2)) / 3.6


_DOC_IDS_SUPPORT_CACHE: dict = {}


def _model_accepts_doc_ids(model) -> bool:
    """Return True iff ``model.forward`` accepts a ``doc_ids`` kwarg.

    Sprint 1: only OdinFlat currently accepts it; other halo variants ignore.
    Cached per-module-type so introspection cost is amortized over training.
    """
    import inspect

    core = model.module if hasattr(model, "module") else model
    core = core._orig_mod if hasattr(core, "_orig_mod") else core
    key = type(core)
    cached = _DOC_IDS_SUPPORT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(core.forward)
        supported = "doc_ids" in sig.parameters
    except (TypeError, ValueError):
        supported = False
    _DOC_IDS_SUPPORT_CACHE[key] = supported
    return supported


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
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--optimize-kernels", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--time-budget", type=float, default=0, help="Minutes, 0=unlimited")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N optimizer steps, 0=unlimited")
    parser.add_argument("--backend", default="gloo", choices=["nccl", "gloo"])
    parser.add_argument("--no-async", action="store_true", help="Disable async allreduce overlap")
    parser.add_argument("--no-fp16-compress", action="store_true", help="Disable fp16 grad compression")
    parser.add_argument("--no-muon", action="store_true", help="Use AdamW instead of Muon")
    parser.add_argument("--warmup-steps", type=int, default=300, help="LR warmup steps")
    parser.add_argument("--num-workers", type=int, default=12, help="DataLoader worker processes")
    parser.add_argument("--auto-eval", action="store_true",
                        help="Spawn scripts/eval_checkpoint.py as detached subprocess "
                             "after each checkpoint save (Sprint 2 scorecard).")

    # Sprint 1: IMU-1 foundation-wins flags. Defaults OFF during validation;
    # Phase 6 flips them to ON after Run 2 gate passes.
    parser.add_argument("--intra-doc-mask", dest="intra_doc_mask", action="store_true",
                        help="Sprint 1: mask attention to intra-document boundaries "
                             "(requires model with use_intra_doc_mask attr, e.g. OdinFlat)")
    parser.add_argument("--no-intra-doc-mask", dest="intra_doc_mask", action="store_false",
                        help="Disable intra-document attention masking.")
    parser.set_defaults(intra_doc_mask=False)

    parser.add_argument("--imu1-groups", dest="imu1_groups", action="store_true",
                        help="Sprint 1: IMU-1 param grouping (2D -> lr_2d WD 0.1, "
                             "1D/embed/lm_head -> lr_1d WD 0). Requires --no-muon "
                             "and default optimizer path.")
    parser.add_argument("--no-imu1-groups", dest="imu1_groups", action="store_false")
    parser.set_defaults(imu1_groups=False)

    parser.add_argument("--lr-2d", type=float, default=None,
                        help="Sprint 1: LR for 2D matrix group (default 0.0235 for NorMuon).")
    parser.add_argument("--lr-1d", type=float, default=None,
                        help="Sprint 1: LR for 1D/embed/lm_head group (default 0.007).")
    parser.add_argument("--normuon", dest="normuon", action="store_true",
                        help="Sprint 1 Phase 2: use NorMuon optimizer for 2D group "
                             "(requires halo_training/normuon.py).")
    parser.add_argument("--no-normuon", dest="normuon", action="store_false")
    parser.set_defaults(normuon=False)
    parser.add_argument("--value-residuals", dest="value_residuals", action="store_true",
                        help="Sprint 1 Phase 2: value residual from GQA layer 6 to 13.")
    parser.add_argument("--no-value-residuals", dest="value_residuals", action="store_false")
    parser.set_defaults(value_residuals=False)
    parser.add_argument("--head-gating", dest="head_gating", action="store_true",
                        help="Sprint 1 Phase 2: per-head sigmoid gate on attention output.")
    parser.add_argument("--no-head-gating", dest="head_gating", action="store_false")
    parser.set_defaults(head_gating=False)
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

    # Sprint 1: honor --intra-doc-mask / --value-residuals / --head-gating by
    # toggling model flags if supported. OdinFlat exposes all three; other
    # models simply ignore.
    sprint1_flag_map = {
        "use_intra_doc_mask": bool(args.intra_doc_mask),
        "use_value_residuals": bool(args.value_residuals),
        "use_head_gating": bool(args.head_gating),
    }
    any_supported = False
    for attr, value in sprint1_flag_map.items():
        if hasattr(model, attr):
            setattr(model, attr, value)
            any_supported = True
    if rank == 0 and any_supported:
        flags_state = {k: ("ON" if v else "off") for k, v in sprint1_flag_map.items()
                       if hasattr(model, k)}
        print(f"[Sprint 1] feature flags on {type(model).__name__}: {flags_state}")
    elif rank == 0 and any(sprint1_flag_map.values()):
        print(f"[Sprint 1] WARNING: feature flags requested but "
              f"{type(model).__name__} does not expose any; all ignored.")

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

    model = DDP(model, device_ids=[0], find_unused_parameters=False,
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
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
    )

    # --- Optimizer + Scheduler ---
    raw_model = model
    # Unwrap compile -> DDP -> raw
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    if hasattr(raw_model, "module"):
        raw_model = raw_model.module

    if args.no_muon:
        if args.imu1_groups:
            # Sprint 1: IMU-1 param grouping — 2D matrices vs 1D/embed/head
            from halo_training.optimizer import build_imu1_optimizer
            lr_2d = args.lr_2d if args.lr_2d is not None else args.lr
            lr_1d = args.lr_1d if args.lr_1d is not None else args.lr * 0.3
            optimizer = build_imu1_optimizer(
                raw_model,
                lr_2d=lr_2d,
                lr_1d=lr_1d,
                weight_decay_2d=0.1,
                betas=(0.9, 0.95),
                use_normuon=args.normuon,
            )
            # build_imu1_optimizer prints on rank 0 already; no duplicate log
        else:
            optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr,
                                           betas=(0.9, 0.95), weight_decay=0.1, fused=True)
            if rank == 0:
                print(f"Using AdamW (lr={args.lr}, wd=0.1)")
    else:
        optimizer = build_muon_optimizer(raw_model, base_lr=args.lr, muon_lr=args.muon_lr)
    total_steps = len(dataloader) * args.epochs // args.accum_steps
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps=args.warmup_steps)

    # --- Loss + AMP ---
    ce_loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, backoff_factor=0.25)

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
    last_log_time = start_time
    last_log_tokens = 0
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
            for batch_idx, batch in enumerate(dataloader):
                # Dataloader yields (input_ids, targets, doc_ids) post-Sprint-1.
                # Accept legacy 2-tuples as well, synthesizing zero doc_ids.
                if len(batch) == 3:
                    input_ids, targets, doc_ids = batch
                else:
                    input_ids, targets = batch
                    doc_ids = None
                if deadline and time.time() > deadline:
                    if rank == 0:
                        print(f"Time budget reached at step {global_step}")
                    break
                if args.max_steps > 0 and global_step >= args.max_steps:
                    if rank == 0:
                        print(f"Max steps reached: {global_step}")
                    break

                input_ids = input_ids.to(device)
                targets = targets.to(device)
                if doc_ids is not None:
                    doc_ids = doc_ids.to(device)
                is_last_microstep = (batch_idx + 1) % args.accum_steps == 0

                # All microsteps use no_sync — we do manual allreduce
                with model.no_sync():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        # Pass doc_ids only if the model accepts it. Backward-
                        # compatible for models unaware of Sprint 1.
                        if doc_ids is not None and _model_accepts_doc_ids(model):
                            output = model(input_ids, targets=targets, doc_ids=doc_ids)
                        else:
                            output = model(input_ids, targets=targets)
                        if isinstance(output, torch.Tensor) and output.dim() == 0:
                            loss = output / args.accum_steps
                        elif isinstance(output, dict):
                            logits = output["logits"]
                            loss = ce_loss_fn(
                                logits.view(-1, logits.size(-1)), targets.view(-1)
                            ) / args.accum_steps
                            if "mtp1" in output:
                                mtp1 = output["mtp1"]
                                mtp_targets = targets[:, 2:].reshape(-1)
                                loss = loss + 0.3 * ce_loss_fn(
                                    mtp1.reshape(-1, logits.size(-1)), mtp_targets
                                ) / args.accum_steps
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

                    # Logging (instantaneous tok/s, not cumulative average)
                    if rank == 0 and global_step % args.log_interval == 0:
                        now = time.time()
                        interval_elapsed = now - last_log_time
                        interval_tokens = (total_tokens - last_log_tokens) * world_size
                        tok_s = interval_tokens / interval_elapsed if interval_elapsed > 0 else 0
                        achieved = interval_tokens * 6 * n_params / interval_elapsed if interval_elapsed > 0 else 0
                        mfu = achieved / 118.8e12
                        last_log_time = now
                        last_log_tokens = total_tokens
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
                        saved_path = save_checkpoint(model, optimizer, global_step,
                                                     args.checkpoint_dir, total_tokens * world_size)
                        if args.auto_eval:
                            spawn_auto_eval(saved_path, args.model, args.class_name)

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
                        saved_path = save_checkpoint(model, optimizer, global_step,
                                                     args.checkpoint_dir, total_tokens * world_size)
                        if args.auto_eval:
                            spawn_auto_eval(saved_path, args.model, args.class_name)
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
        saved_path = save_checkpoint(model, optimizer, global_step,
                                     args.checkpoint_dir, total_tokens * world_size)
        if args.auto_eval:
            spawn_auto_eval(saved_path, args.model, args.class_name)

    if rank == 0:
        elapsed = time.time() - start_time
        global_tokens = total_tokens * world_size
        print(f"\nDone: {global_step} steps, {global_tokens:,} tokens in {elapsed:.0f}s "
              f"({global_tokens / elapsed:,.0f} tok/s), best loss={best_loss:.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
