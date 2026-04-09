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
    """
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
