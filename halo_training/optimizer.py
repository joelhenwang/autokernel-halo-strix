"""Optimizer factory with DeepSpeed CPUAdam and COOKBOOK.md param groups."""

import math
from typing import List, Optional, Type

import torch
import torch.nn as nn


def _patch_deepspeed():
    """Fix DeepSpeed 0.17.5 bug: cxx_args() returns None on ROCm."""
    try:
        import deepspeed.ops.op_builder.builder as builder
        _orig = builder.OpBuilder.strip_empty_entries

        def _patched(self, args):
            return [x for x in args if x is not None and len(x) > 0]

        builder.OpBuilder.strip_empty_entries = _patched
    except ImportError:
        pass


def _get_cpu_adam():
    """Try to import DeepSpeed CPUAdam with ROCm fixes."""
    _patch_deepspeed()
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        return DeepSpeedCPUAdam
    except (ImportError, RuntimeError):
        return None


def build_optimizer(
    model: nn.Module,
    base_lr: float = 8e-4,
    weight_decay: float = 0.1,
    optimizer_cls: Optional[Type] = None,
    use_cpu_adam: bool = False,
    use_muon: bool = False,
    muon_lr: float = 0.005,
    polar_ns: bool = False,
) -> torch.optim.Optimizer:
    """Build optimizer with COOKBOOK.md param group rules.

    Param group assignment by name pattern:
    - engram tables: 5x LR, 0 WD
    - decay_bias: 0.1x LR, 0 WD
    - omega/gamma params: 0.125x LR, 0 WD
    - meta_token: 1x LR, 0.01 WD
    - norms/biases: 1x LR, 0 WD
    - everything else: 1x LR, default WD

    Args:
        use_cpu_adam: Force DeepSpeed CPUAdam. Requires params on CPU.
                     Useful for Mode B (7B+) where optimizer states dominate memory.
                     For Mode A (<2B), fused AdamW on GPU is simpler and fast enough.
        use_muon: Use Muon optimizer for 2D weight matrices, AdamW for rest.
                  ~2x token-efficiency, ~50% less optimizer memory.
        muon_lr: Learning rate for Muon params (default 0.02, different scale from AdamW).
    """
    if use_muon:
        return _build_muon_optimizer(model, base_lr, muon_lr, weight_decay, polar_ns=polar_ns)

    groups = _build_param_groups(model, base_lr, weight_decay)

    if optimizer_cls is not None:
        return optimizer_cls(groups, lr=base_lr)

    if use_cpu_adam:
        CPUAdam = _get_cpu_adam()
        if CPUAdam is not None:
            try:
                opt = CPUAdam(groups, lr=base_lr, betas=(0.9, 0.95))
                print("Using DeepSpeed CPUAdam (AVX-512 + OpenMP)")
                return opt
            except Exception as e:
                print(f"DeepSpeed CPUAdam failed ({e}), falling back to AdamW")

    print("Using torch.optim.AdamW(fused=True)")
    return torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95), fused=True)


def _build_muon_optimizer(
    model: nn.Module,
    adamw_lr: float,
    muon_lr: float,
    weight_decay: float,
    polar_ns: bool = False,
) -> torch.optim.Optimizer:
    """Build Muon optimizer: 2D weights get Muon, rest gets AdamW."""
    from halo_training.muon import Muon, split_params_for_muon

    muon_params, adamw_named = split_params_for_muon(model)

    # Build AdamW groups with COOKBOOK.md rules
    adamw_groups = []
    for name, param in adamw_named:
        if "engram" in name and "table" in name:
            adamw_groups.append({"params": [param], "lr": adamw_lr * 5.0, "weight_decay": 0})
        elif "decay_bias" in name:
            adamw_groups.append({"params": [param], "lr": adamw_lr * 0.1, "weight_decay": 0})
        elif "omega_param" in name or "gamma_param" in name:
            adamw_groups.append({"params": [param], "lr": adamw_lr * 0.125, "weight_decay": 0})
        elif "meta_token" in name:
            adamw_groups.append({"params": [param], "lr": adamw_lr, "weight_decay": 0.01})
        elif "norm" in name or (name.endswith(".bias") and "decay" not in name):
            adamw_groups.append({"params": [param], "lr": adamw_lr, "weight_decay": 0})
        else:
            adamw_groups.append({"params": [param], "lr": adamw_lr, "weight_decay": weight_decay})

    n_muon = len(muon_params)
    n_adamw = sum(len(g["params"]) for g in adamw_groups)
    print(f"Using Muon optimizer: {n_muon} params via Muon (lr={muon_lr}), "
          f"{n_adamw} params via AdamW (lr={adamw_lr})")

    return Muon(
        muon_params=[{"params": muon_params}],
        lr=muon_lr,
        weight_decay=0.01,
        adamw_params=adamw_groups,
        adamw_lr=adamw_lr,
        polar_ns=polar_ns,
    )


def _build_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
) -> List[dict]:
    """Adapted from COOKBOOK.md lines 313-331."""
    groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "engram" in name and "table" in name:
            groups.append({
                "params": [param], "lr": base_lr * 5.0,
                "weight_decay": 0, "name": name,
            })
        elif "decay_bias" in name:
            groups.append({
                "params": [param], "lr": base_lr * 0.1,
                "weight_decay": 0, "name": name,
            })
        elif "omega_param" in name or "gamma_param" in name:
            groups.append({
                "params": [param], "lr": base_lr * 0.125,
                "weight_decay": 0, "name": name,
            })
        elif "meta_token" in name:
            groups.append({
                "params": [param], "lr": base_lr,
                "weight_decay": 0.01, "name": name,
            })
        elif "norm" in name or (name.endswith(".bias") and "decay" not in name):
            groups.append({
                "params": [param], "lr": base_lr,
                "weight_decay": 0, "name": name,
            })
        else:
            groups.append({
                "params": [param], "lr": base_lr,
                "weight_decay": weight_decay, "name": name,
            })
    return groups


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.1,
    warm_restarts: bool = False,
    restart_period: Optional[int] = None,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Cosine schedule with linear warmup. Base LR 8e-4 -> 8e-5.

    Args:
        warm_restarts: Use CosineAnnealingWarmRestarts instead (for ternary/quantized models).
        restart_period: Steps between restarts (default: total_steps // 4).
    """
    if warm_restarts:
        T_0 = restart_period or max(1, total_steps // 4)
        min_lr = optimizer.defaults["lr"] * min_lr_ratio
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=min_lr,
        )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 300,
    stable_fraction: float = 0.8,
    min_lr_ratio: float = 0.0,
    wd_start: float = 0.1,
    wd_end: float = 0.01,
) -> "WSDScheduler":
    """Warmup-Stable-Decay schedule with weight decay annealing.

    Warmup: linear 0→1 over warmup_steps.
    Stable: constant 1.0 until total_steps * stable_fraction.
    Decay: linear 1.0→min_lr_ratio over remaining steps.
    Weight decay: anneals wd_start→wd_end during decay phase.
    """
    return WSDScheduler(optimizer, total_steps, warmup_steps,
                        stable_fraction, min_lr_ratio, wd_start, wd_end)


class WSDScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Warmup-Stable-Decay LR schedule with weight decay annealing."""

    def __init__(self, optimizer, total_steps, warmup_steps=300,
                 stable_fraction=0.8, min_lr_ratio=0.0,
                 wd_start=0.1, wd_end=0.01):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_start = int(total_steps * stable_fraction)
        self.min_lr_ratio = min_lr_ratio
        self.wd_start = wd_start
        self.wd_end = wd_end

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if step < self.decay_start:
                return 1.0
            decay_progress = (step - self.decay_start) / max(1, total_steps - self.decay_start)
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * decay_progress)

        super().__init__(optimizer, lr_lambda)

    def step(self, epoch=None):
        super().step(epoch)
        current_step = self.last_epoch
        if current_step >= self.decay_start:
            decay_progress = (current_step - self.decay_start) / max(
                1, self.total_steps - self.decay_start)
            decay_progress = min(1.0, decay_progress)
            wd = self.wd_start + (self.wd_end - self.wd_start) * decay_progress
            for group in self.optimizer.param_groups:
                if group.get("weight_decay", 0) > 0:
                    group["weight_decay"] = wd
