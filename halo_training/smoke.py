"""Standardized smoke test framework matching the llm_engineer playbook.

Pass criteria:
1. Loss decreases over first 100 steps
2. No NaN/Inf for 200 steps
3. Grad norms < 10, no spikes
4. Memory peak < threshold (default 6 GB for 250M)
5. Throughput > 10K tok/s
6. For recurrent models: state-norm ratio < 1.05
"""

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from halo_training.data import BabyLMDataset, build_dataloader
from halo_training.optimizer import build_optimizer, build_scheduler
from halo_training.model_utils import count_parameters
from halo_training.callbacks import StateNormMonitor


def run_smoke_test(
    model: nn.Module,
    dataset: str = "datasets/smoke-test-dataset",
    steps: int = 200,
    batch_size: int = 16,
    block_size: int = 512,
    accum_steps: int = 1,
    base_lr: float = 8e-4,
    max_grad_norm: float = 1.0,
    max_memory_gb: float = 6.0,
    min_tok_s: float = 10_000,
    max_state_ratio: float = 1.05,
    compile: bool = False,
) -> Dict[str, Any]:
    """Run standardized smoke test and return pass/fail results.

    Returns dict with:
    - passed: bool (all criteria met)
    - criteria: dict of {name: {passed, value, threshold}}
    - stats: training statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    n_params = count_parameters(model)

    # Setup data
    ds = BabyLMDataset(root=dataset, block_size=block_size)

    # Clamp vocab if needed
    model_vocab = _get_vocab_size(model)
    if model_vocab and ds.vocab_size > model_vocab:
        ds.tokens = ds.tokens.clamp(max=model_vocab - 1)

    dataloader = build_dataloader(ds, batch_size=batch_size, num_workers=0, shuffle=True)

    # Setup optimizer
    optimizer = build_optimizer(model, base_lr=base_lr)
    scheduler = build_scheduler(optimizer, total_steps=steps)

    # Setup state-norm monitor for recurrent architectures
    state_monitor = StateNormMonitor(model, warn_ratio=max_state_ratio)

    # Setup compile
    if compile:
        model = torch.compile(model, mode="reduce-overhead")

    # Loss function
    ce_loss = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Run training
    losses = []
    grad_norms = []
    nan_inf_count = 0
    global_step = 0
    total_tokens = 0
    start_time = time.time()

    print(f"Smoke test: {n_params/1e6:.1f}M params, {steps} steps, "
          f"batch={batch_size}, block={block_size}")

    data_iter = iter(dataloader)

    for micro_step in range(steps * accum_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids, targets = batch
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(input_ids)
            if isinstance(output, torch.Tensor):
                logits = output
            elif hasattr(output, "logits"):
                logits = output.logits
            else:
                logits = output
            loss = ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1)) / accum_steps

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            nan_inf_count += 1
            if nan_inf_count > 5:
                print(f"FAIL: {nan_inf_count} NaN/Inf losses, aborting")
                break
            continue

        scaler.scale(loss).backward()
        total_tokens += input_ids.numel()

        if (micro_step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norms.append(grad_norm.item() if torch.isfinite(grad_norm) else float("inf"))

            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            losses.append(loss.item() * accum_steps)
            global_step += 1

            # State norm monitoring
            state_monitor(model, global_step)

            if global_step % 50 == 0:
                elapsed = time.time() - start_time
                tok_s = total_tokens / elapsed
                print(f"  [step {global_step:>4d}] loss={losses[-1]:.4f} "
                      f"grad={grad_norms[-1]:.2f} tok/s={tok_s:,.0f}")

            if global_step >= steps:
                break

    elapsed = time.time() - start_time
    tok_s = total_tokens / elapsed if elapsed > 0 else 0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0

    # Evaluate criteria
    criteria = {}

    # 1. Loss decreases
    if len(losses) >= 100:
        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        loss_decreased = last_10 < first_10
        criteria["loss_decreases"] = {
            "passed": loss_decreased,
            "value": f"{first_10:.2f} -> {last_10:.2f}",
            "threshold": "last_10 < first_10",
        }
    else:
        criteria["loss_decreases"] = {
            "passed": len(losses) > 0 and losses[-1] < losses[0],
            "value": f"{losses[0]:.2f} -> {losses[-1]:.2f}" if losses else "N/A",
            "threshold": "decreasing",
        }

    # 2. No NaN/Inf
    criteria["no_nan_inf"] = {
        "passed": nan_inf_count == 0,
        "value": nan_inf_count,
        "threshold": 0,
    }

    # 3. Grad norms < 10
    max_grad = max(grad_norms) if grad_norms else 0
    criteria["grad_norms"] = {
        "passed": max_grad < 10.0,
        "value": f"max={max_grad:.2f}",
        "threshold": "< 10.0",
    }

    # 4. Memory
    criteria["memory"] = {
        "passed": peak_mem_gb < max_memory_gb,
        "value": f"{peak_mem_gb:.1f} GB",
        "threshold": f"< {max_memory_gb} GB",
    }

    # 5. Throughput
    criteria["throughput"] = {
        "passed": tok_s >= min_tok_s,
        "value": f"{tok_s:,.0f} tok/s",
        "threshold": f">= {min_tok_s:,.0f} tok/s",
    }

    # 6. State norm (recurrent stability)
    state_report = state_monitor.report()
    if state_report.get("tracked_modules"):
        criteria["state_norm"] = {
            "passed": state_report["stable"],
            "value": f"max_ratio={state_report['max_ratio']:.4f}",
            "threshold": f"< {max_state_ratio}",
        }

    state_monitor.remove_hooks()

    # Overall
    all_passed = all(c["passed"] for c in criteria.values())

    # Print report
    print(f"\n{'=' * 60}")
    print(f"SMOKE TEST {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'=' * 60}")
    for name, c in criteria.items():
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {name}: {c['value']} (threshold: {c['threshold']})")
    print(f"{'=' * 60}")
    print(f"Steps: {global_step}, Tokens: {total_tokens:,}, "
          f"Time: {elapsed:.0f}s, tok/s: {tok_s:,.0f}")

    return {
        "passed": all_passed,
        "criteria": criteria,
        "stats": {
            "steps": global_step,
            "tokens": total_tokens,
            "elapsed_s": elapsed,
            "tok_s": tok_s,
            "peak_memory_gb": peak_mem_gb,
            "final_loss": losses[-1] if losses else None,
            "max_grad_norm": max_grad,
        },
    }


def _get_vocab_size(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return module.num_embeddings
    return None
