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
from typing import Optional

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


def load_model_from_file(model_path: str, class_name: str, **kwargs):
    """Load model class from a .py file and instantiate it.

    Keyword args are forwarded to the model constructor. Any kwarg the
    model ctor doesn't accept is silently dropped, so Sprint 1.5 flags
    (use_mup, mup_base_width) can be passed without breaking older model
    classes whose ctor predates them.
    """
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    if kwargs:
        # Filter to only those kwargs this ctor accepts.
        import inspect
        sig = inspect.signature(cls.__init__)
        accepted = {name for name, p in sig.parameters.items()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return cls(**filtered)
    return cls()


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


# ===========================================================================
# Stability + NaN forensics — imported from halo_training/stability.py
# ===========================================================================
from halo_training.stability import save_nan_forensics


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

from halo_training.stability import StabilityGuard


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
    # v3 T-0.6 DDP trace: count this call; bytes estimated from param grads.
    _ddp_trace_record_allreduce(
        model, count_mode="async",
    )
    return handles


def allreduce_grads_sync(model):
    """Synchronous allreduce on all gradients."""
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
    _ddp_trace_record_allreduce(
        model, count_mode="sync",
    )


# v3 T-0.6 DDP allreduce trace state (rank-0 emission; see _ddp_trace_emit
# called from optimizer-step boundary).
_DDP_TRACE_STATE = {
    "step": 0,
    "allreduce_count": 0,
    "allreduce_bytes": 0,
    "first_allreduce_wall_ms": -1.0,
    "last_allreduce_wall_ms": -1.0,
    "backward_start_wall_ms": -1.0,
    "backward_end_wall_ms": -1.0,
    "accum_steps": 8,
    "bucket_cap_mb": 25,
    "gradient_as_bucket_view": False,
    "static_graph": False,
    "enabled": False,      # set True by trainer main() when DDP is in use
    "path": None,          # JSONL path
    "assert_no_sync": False,
}


def _ddp_trace_record_allreduce(model, count_mode: str):
    """Called from the two allreduce helpers to count + time each call."""
    import time as _t
    state = _DDP_TRACE_STATE
    if not state["enabled"]:
        return
    wall_ms = _t.perf_counter() * 1000.0
    state["allreduce_count"] += 1
    if state["first_allreduce_wall_ms"] < 0:
        state["first_allreduce_wall_ms"] = wall_ms
    state["last_allreduce_wall_ms"] = wall_ms
    # Bytes estimate: sum of grad.numel() * 2 (fp16 compressed if use_fp16)
    # or *4 (fp32). We approximate as fp16 since that's our default compress path.
    if state["allreduce_bytes"] == 0:
        nbytes = 0
        for p in model.parameters():
            if p.grad is not None:
                nbytes += p.grad.numel() * 2  # fp16-compressed grad
        state["allreduce_bytes"] = nbytes


def _ddp_trace_emit(world_size: int, rank: int):
    """v3 T-0.6 emit one JSONL record per opt step (rank 0 only)."""
    import json as _json
    state = _DDP_TRACE_STATE
    if not state["enabled"] or rank != 0 or state["path"] is None:
        return
    # Only emit if at least one allreduce happened this step (otherwise it's
    # an empty cycle after max-steps flush).
    if state["allreduce_count"] == 0:
        return
    total_ms = (state["last_allreduce_wall_ms"]
                - state["first_allreduce_wall_ms"])
    bwd_start = state["backward_start_wall_ms"]
    first_ar_offset = (state["first_allreduce_wall_ms"] - bwd_start
                       if bwd_start > 0 else 0.0)
    bwd_window = (state["backward_end_wall_ms"] - bwd_start
                  if bwd_start > 0 and state["backward_end_wall_ms"] > 0 else 0.0)
    overlap_est = 0.0
    if bwd_window > 0 and total_ms > 0:
        # Overlap = fraction of allreduce that happens during backward window
        ar_end_offset = state["last_allreduce_wall_ms"] - bwd_start
        overlapping = max(0.0, min(bwd_window, ar_end_offset)
                          - max(0.0, first_ar_offset))
        overlap_est = overlapping / total_ms if total_ms > 0 else 0.0
    record = {
        "step": int(state["step"]),
        "accum_steps": int(state["accum_steps"]),
        "no_sync_expected_microsteps": int(state["accum_steps"] - 1),
        "allreduce_count": int(state["allreduce_count"]),
        "allreduce_total_ms": float(total_ms),
        "first_allreduce_start_ms_after_backward_start": float(first_ar_offset),
        "last_allreduce_end_ms": float(state["last_allreduce_wall_ms"]),
        "overlap_ratio_estimate": float(overlap_est),
        "bucket_cap_mb": int(state["bucket_cap_mb"]),
        "gradient_as_bucket_view": bool(state["gradient_as_bucket_view"]),
        "static_graph": bool(state["static_graph"]),
        "allreduce_bytes": int(state["allreduce_bytes"]),
        "world_size": int(world_size),
    }
    try:
        with open(state["path"], "a", buffering=1) as fh:
            fh.write(_json.dumps(record) + "\n")
    except Exception:
        pass  # telemetry should never break training

    # v3 T-0.6 tripwire: if --assert-no-sync and allreduce_count > 1, abort.
    # manual-allreduce path: expect exactly 1 allreduce_count per opt step.
    if state["assert_no_sync"] and state["allreduce_count"] > 1:
        raise RuntimeError(
            f"[--assert-no-sync] step {state['step']}: "
            f"allreduce_count={state['allreduce_count']} > 1. "
            f"Accumulation no_sync() regressed."
        )


def _ddp_trace_reset_step(next_step: int):
    """Clear per-step counters AFTER emission, before the next opt step."""
    state = _DDP_TRACE_STATE
    state["step"] = next_step
    state["allreduce_count"] = 0
    state["allreduce_bytes"] = 0
    state["first_allreduce_wall_ms"] = -1.0
    state["last_allreduce_wall_ms"] = -1.0
    state["backward_start_wall_ms"] = -1.0
    state["backward_end_wall_ms"] = -1.0


def average_grads(model, world_size):
    """Divide all gradients by world_size (after allreduce SUM)."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data /= world_size


_consecutive_skips = 0

def _autokernel_autograd_preflight(model, device, args) -> tuple[bool, str]:
    """Phase E.3 preflight (2026-05-11): dispatch one forward+backward on a
    small dummy batch, then assert every parameter with requires_grad=True
    received a finite, non-None gradient. Catches autograd severance bugs
    in autokernel Replacements before committing compute budget to a run.

    Returns (ok, message). ok=False aborts the launch. message on failure
    enumerates the first offending parameters.
    """
    import torch

    # Sample a tiny batch from the dataloader's distribution. We use a
    # single-step dummy to avoid entangling with the outer training loop
    # or the compile warmup cache.
    try:
        block_size = int(args.block_size)
        batch_size = min(2, int(args.batch_size))
        # Use a lightweight vocab range; load_model_from_file guarantees
        # tok_embeddings exists; fall back to 100 if not inferrable.
        vocab = 100
        if hasattr(model, "module"):
            inner = model.module
        else:
            inner = model
        for name, mod in inner.named_modules():
            if hasattr(mod, "num_embeddings"):
                vocab = int(mod.num_embeddings)
                break
        x = torch.randint(0, vocab, (batch_size, block_size),
                          device=device, dtype=torch.long)
        t = torch.randint(0, vocab, (batch_size, block_size),
                          device=device, dtype=torch.long)
    except Exception as exc:  # noqa: BLE001
        return False, f"preflight dummy-input construction failed: {exc}"

    # Zero grads before the test so we can distinguish "got grad now" vs
    # "carried over from earlier".
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    model.train()
    try:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x, targets=t) if _model_accepts_kwargs(model, "targets") else model(x)
            if isinstance(out, torch.Tensor) and out.dim() == 0:
                loss = out
            elif isinstance(out, dict) and "logits" in out:
                logits = out["logits"]
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    t.view(-1),
                )
            elif isinstance(out, torch.Tensor):
                loss = torch.nn.functional.cross_entropy(
                    out.view(-1, out.size(-1)).float(),
                    t.view(-1),
                )
            else:
                return False, f"preflight unknown output type: {type(out)}"
        loss.backward()
    except Exception as exc:  # noqa: BLE001
        return False, f"preflight forward/backward raised: {exc}"

    # Inspect grads. A parameter is considered OK if its grad is finite and
    # not all zeros. Parameters with all-zero grads are allowed ONLY if
    # they are documented as such (e.g. v_res_scale on the first layer with
    # no v_prev input).
    ALLOWED_ZERO_PATTERNS = {
        "v_res_scale",  # first-layer, no v_prev — documented in Track 3.A
        "head_gate",    # only active when caller passes head_gate_active=True
                        # (not guaranteed in every forward; OdinHalo's
                        # shared_layer head_gate is unused per B4 diag)
    }
    offenders = []
    none_offenders = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            if any(allowed in name for allowed in ALLOWED_ZERO_PATTERNS):
                continue
            none_offenders.append(name)
            continue
        # p.grad is not None; check finite-ness
        gnorm = float(p.grad.detach().float().abs().sum().item())
        if not torch.isfinite(p.grad).all():
            offenders.append(f"{name} (non-finite grad)")
        elif gnorm == 0.0 and not any(allowed in name for allowed in ALLOWED_ZERO_PATTERNS):
            offenders.append(f"{name} (grad all zero)")

    if none_offenders:
        sample = ", ".join(none_offenders[:5])
        more = f" (+ {len(none_offenders) - 5} more)" if len(none_offenders) > 5 else ""
        return False, (
            f"{len(none_offenders)} parameters received grad=None: {sample}{more}"
        )
    if offenders:
        sample = "; ".join(offenders[:5])
        more = f" (+ {len(offenders) - 5} more)" if len(offenders) > 5 else ""
        return False, f"{len(offenders)} parameter grads malformed: {sample}{more}"

    return True, ""


def _model_accepts_kwargs(model, kwarg: str) -> bool:
    """Minimal introspection helper; skipped for DDP-wrapped models."""
    inner = model.module if hasattr(model, "module") else model
    try:
        import inspect
        sig = inspect.signature(inner.forward)
        return kwarg in sig.parameters
    except Exception:  # noqa: BLE001
        return False


def _complete_step(model, optimizer, scaler, scheduler, world_size, max_grad_norm, rank, diag_writer=None):
    """Unscale, clip, step optimizer, update scaler. Returns grad_norm.

    If ``diag_writer`` is not None, it is invoked with ``model`` after clip
    and before ``optimizer.zero_grad`` — grads are populated and (in finite
    cases) post-clip. Used by the --diag-frozen-params diagnostic.
    """
    global _consecutive_skips
    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if diag_writer is not None:
        diag_writer(model)
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
    parser.add_argument("--resume-preserve-optimizer", action="store_true",
                        help="v3 T-5 C.1: also restore optimizer + scaler state from "
                             "--resume-from. Default resume is weights-only; this enables "
                             "the preserved-state warm-start variant (tests v3 H14).")

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
    # Phase II throughput investigation (2026-05-09) — allow selective
    # enable/disable of individual autokernel patterns. Accepts comma-
    # separated pattern names (e.g. 'rmsnorm,fused_silu_gate_mul').
    parser.add_argument("--autokernel-include", type=str, default="",
                        help="Comma-separated autokernel pattern names to "
                             "include (empty = all registered). Only has "
                             "effect with --optimize-kernels.")
    parser.add_argument("--autokernel-exclude", type=str, default="",
                        help="Comma-separated autokernel pattern names to "
                             "exclude. Only has effect with --optimize-kernels.")
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

    # Sprint 1.1 Phase B — NorMuon throughput knobs
    parser.add_argument("--ns-dtype", choices=["fp32", "fp16"], default="fp16",
                        help="Sprint 1.1: Newton-Schulz inner matmul dtype. "
                             "fp16 (DEFAULT, shipped 2026-05-07) routes NS "
                             "through rocBLAS HHS_BH_ kernels — +17.5 pct "
                             "tok/s measured on Run 2b, with loss "
                             "indistinguishable from fp32 at 1 epoch. "
                             "Use fp32 to restore Phase 2 behavior.")
    parser.add_argument("--neuron-norm-min-dim", type=int, default=0,
                        help="Sprint 1.1: skip NorMuon neuron-wise norm on "
                             "2D params where min(rows, cols) < this threshold. "
                             "0=always apply (default). Try 512 to skip "
                             "embed-adjacent small projections.")
    parser.add_argument("--cautious-wd", dest="cautious_wd", action="store_true",
                        help="Sprint 1.1: enable cautious weight decay (IMU-1 default).")
    parser.add_argument("--no-cautious-wd", dest="cautious_wd", action="store_false",
                        help="Sprint 1.1: disable cautious WD (use standard decoupled WD).")
    parser.set_defaults(cautious_wd=True)

    # Sprint 1.5 Phase A: SPECTRA + μP plumbing (default OFF; opt-in flags).
    # See docs/superpowers/specs/2026-05-06-sprint1.5-spectra-mup-design.md.
    parser.add_argument("--spectra-post", dest="spectra_post", action="store_true",
                        help="Sprint 1.5: enable SPECTRA post-clipping on NorMuon updates. "
                             "Default OFF. Requires --normuon. Clip threshold via "
                             "--spectra-clip-norm (default 1.0).")
    parser.add_argument("--no-spectra-post", dest="spectra_post", action="store_false")
    parser.set_defaults(spectra_post=False)
    parser.add_argument("--spectra-clip-norm", type=float, default=1.0,
                        help="Spectral-norm ceiling per step for SPECTRA post-clip. "
                             "Default 1.0.")
    parser.add_argument("--spectra-ns-iter", type=int, default=5,
                        help="Newton-Schulz iterations for SPECTRA spectral-norm "
                             "estimate (API forward-compat; currently unused by "
                             "the power-iteration implementation).")
    # Phase D pre-decl: wired but no effect in Phase A
    parser.add_argument("--spectra-pre", dest="spectra_pre", action="store_true",
                        help="[Phase D] SPECTRA pre-clipping on gradients. "
                             "Not yet active in Phase A.")
    parser.add_argument("--no-spectra-pre", dest="spectra_pre", action="store_false")
    parser.set_defaults(spectra_pre=False)

    parser.add_argument("--mup", dest="mup", action="store_true",
                        help="Sprint 1.5: enable μP 3-way LR scaling (embedding / "
                             "hidden / readout). Default OFF. With --mup, "
                             "--lr-2d sets the embedding LR; hidden/readout "
                             "derive from d_base via the μP transfer rules.")
    parser.add_argument("--no-mup", dest="mup", action="store_false")
    parser.set_defaults(mup=False)
    parser.add_argument("--mup-base-width", type=int, default=256,
                        help="μP base width d_base (default 256; matches the "
                             "30M probe's d_model).")
    # Phase E pre-decl: wired but no effect in Phase A
    parser.add_argument("--mup-attn", dest="mup_attn", action="store_true",
                        help="[Phase E] μP 1/d_head attention scale. Not yet "
                             "active in Phase A.")
    parser.add_argument("--no-mup-attn", dest="mup_attn", action="store_false")
    parser.set_defaults(mup_attn=False)

    # Phase 3 (LEAP layer-exit aux loss) — arXiv 2605.01058. Opt-in flag.
    # Applies a sigmoid(10 * (tau - cos(h_layer_i, sg(h_final)))) aux loss
    # at the specified intermediate layers; pushes them toward h_final so
    # inference-time early exit becomes viable. See
    # halo_training/leap_layer_exit.py for details.
    parser.add_argument("--leap-layers", type=str, default="",
                        help="Comma-separated layer indices for LEAP aux loss "
                             "(e.g. '11,12'). Empty (default) disables LEAP.")
    parser.add_argument("--leap-weights", type=str, default="",
                        help="Comma-separated weights for each --leap-layers "
                             "entry (e.g. '0.3,0.5'). If empty, defaults to "
                             "uniform 1.0 per layer.")
    parser.add_argument("--leap-tau", type=float, default=0.98,
                        help="LEAP training-time cosine threshold. "
                             "Default 0.98 (paper validated).")
    parser.add_argument("--leap-layers-attr", type=str, default="layers",
                        help="Attribute on the model holding the ModuleList "
                             "of layers LEAP should hook. 'layers' for "
                             "OdinFlat, 'shared_layers' for OdinHalo.")

    # fp16 stability hardening (2026-05-07, post-dolma-10B NaN incident)
    # See knowledge/training/fp16_stability_gfx1151.md for details.
    parser.add_argument("--z-loss", type=float, default=0.0,
                        help="Auxiliary z-loss weight on logsumexp(logits)^2. "
                             "Recommended 1e-4 for looped/long-horizon runs. "
                             "Only applied when model forward returns a dict "
                             "with 'logits' (raw-logits path). 0 = off.")
    parser.add_argument("--z-loss-fraction", type=float, default=0.4,
                        help="Fraction of total steps across which z-loss "
                             "remains active (linearly ramps to 0 after). "
                             "Default 0.4 = first 40%% of training.")
    parser.add_argument("--attn-softcap", type=float, default=0.0,
                        help="Pre-softmax attention-score tanh softcap. "
                             "0 = off (uses F.scaled_dot_product_attention). "
                             ">0 = manual attention path with "
                             "scores = softcap * tanh(scores / softcap). "
                             "Recommended 50.0 for long-context looped runs.")
    parser.add_argument("--activation-monitor", dest="activation_monitor",
                        action="store_true",
                        help="Enable per-layer max-abs activation tracker. "
                             "Samples every --activation-monitor-interval steps "
                             "(default 100). Emits $CKPT_DIR/activation_stats.jsonl. "
                             "Negligible throughput cost when sampled.")
    parser.add_argument("--activation-monitor-interval", type=int, default=100,
                        help="Sampling interval (optimizer steps) for "
                             "--activation-monitor. Default 100.")

    # Track 1.1 (2026-05-10): step profiler. Writes torch.profiler Chrome
    # trace + flat op table over the specified opt-step range. Rank-0 only.
    # See docs/superpowers/plans/2026-05-10-odinflat-throughput-investigation-plan.md
    parser.add_argument("--profile-steps", type=str, default="",
                        help="Profile optimizer-step range 'start:end' "
                             "(e.g. '30:40' profiles 10 steps). Writes "
                             "profile.json (Chrome trace) and "
                             "profile-summary.txt under $CKPT_DIR. "
                             "Rank-0 only; empty=no profiling.")

    # Track 3.A (2026-05-10): per-parameter grad-norm recorder for the
    # frozen-blast-radius diagnostic. On each optimizer step, writes a
    # JSONL line {step, params: [{name, grad_norm, is_none, is_zero}]}.
    # Rank-0 only, before zero_grad, after clip. Intended for short
    # (~50-step) diagnostic runs. See same plan doc.
    parser.add_argument("--diag-frozen-params", type=str, default="",
                        help="Path to JSONL file for per-param grad norms "
                             "(Track 3.A diagnostic). Empty=disabled.")

    # Phase B.5 (2026-05-11): opt-in fused z-loss. When enabled AND
    # --z-loss > 0, routes the final logits through kernel.ce_full with
    # z_loss_weight which bakes z-loss into the HIP forward AND its
    # gradient contribution back to logits. Eliminates the separate
    # aten::logsumexp pass (16.7% of step per Track 1.3 profile).
    # Off by default pending Phase C validation.
    parser.add_argument("--use-fused-zloss", dest="use_fused_zloss",
                        action="store_true",
                        help="Route z-loss through kernel.ce_full (fused fwd+bwd).")
    parser.add_argument("--no-use-fused-zloss", dest="use_fused_zloss",
                        action="store_false")
    parser.set_defaults(use_fused_zloss=False)

    # ======================================================================
    # v3 40k throughput campaign: granular --ak-* flags
    # (docs/research/autokernel-40k-v3-execution-plan.md section 4)
    #
    # These replace the monolithic --optimize-kernels. All default OFF.
    # Each flag is independently measurable, auditable, and reversible.
    # Hidden-kernel flags must pass gradient-flow + dtype audit before
    # 2000-step gate (see CONSTRAINTS.md).
    # ======================================================================
    # Core granular kernel/runtime flags (v3 section 8.1)
    ak_group = parser.add_argument_group("AK granular kernel/runtime")
    ak_group.add_argument("--ak-loss-ce", action="store_true",
                          help="Route logits through kernel.ce_full (safe CE path).")
    ak_group.add_argument("--ak-loss-zloss", action="store_true",
                          help="Bake z-loss into fused CE (Phase B.5). Alias of --use-fused-zloss.")
    ak_group.add_argument("--ak-swiglu-fwd", action="store_true",
                          help="Use HIP silu_gate_mul forward (autograd-safe).")
    ak_group.add_argument("--ak-swiglu-bwd", action="store_true",
                          help="Use HIP silu_gate_mul backward (vs PyTorch fallback).")
    ak_group.add_argument("--ak-rmsnorm", action="store_true",
                          help="Use HIP rmsnorm custom_op.")
    ak_group.add_argument("--ak-res-rmsnorm", action="store_true",
                          help="Use HIP fused_res_rmsnorm custom_op.")
    ak_group.add_argument("--ak-rope", action="store_true",
                          help="Use HIP rotary_emb_fp32 custom_op.")
    ak_group.add_argument("--ak-rope-gate", action="store_true",
                          help="Use HIP fused_rope_gate_mul (requires T-3.2 fix).")
    ak_group.add_argument("--ak-causal-conv", action="store_true",
                          help="Use DaoAILab causal_conv1d_fn (requires T-3.2 shim).")
    ak_group.add_argument("--ak-qkv", action="store_true",
                          help="Use fused QKV custom_op.")
    ak_group.add_argument("--ak-ple-gate", action="store_true",
                          help="Use HIP fused_ple_gate custom_op.")
    # --ak-normuon: alias --normuon (already exists)
    ak_group.add_argument("--ak-compiled-autograd", action="store_true",
                          help="Enable torch._dynamo.config.compiled_autograd = True (T-4).")
    ak_group.add_argument("--ak-triton-visible", action="store_true",
                          help="Route Triton kernels through torch.library.triton_op "
                               "(vs custom_op); improves Inductor compile visibility.")
    ak_group.add_argument("--ak-sync-cleanup", action="store_true",
                          help="Branchless SPECTRA + deferred .item() aggregation. "
                               "Alias of --ak-spectra-branchless for now; expand scope in T-1.2.")
    ak_group.add_argument("--ak-ddp-tune", action="store_true",
                          help="gradient_as_bucket_view=True + tuned bucket_cap_mb.")

    # v3 add-on flags (v3 section 8.2)
    ak_group.add_argument("--ak-spectra-branchless", action="store_true",
                          help="Use branchless SPECTRA (no sigma1.item()). T-1.1.")
    ak_group.add_argument("--ak-autocast-tier",
                          choices=["none", "tier1", "all"], default="none",
                          help="Autocast rule application: none (default), tier1 "
                               "(rope_gate_mul + causal_conv1d only), or all "
                               "(register_autocast on every training-path custom op).")
    ak_group.add_argument("--ak-dtype-trace", action="store_true",
                          help="Emit dtype/autocast trace JSONL for custom ops (T-0.7).")
    ak_group.add_argument("--ak-fix-rope-gate-op", action="store_true",
                          help="Use fixed rope_gate_mul via torch.library.custom_op "
                               "(vs @torch.compiler.disable wrapper). T-3.2.")
    ak_group.add_argument("--ak-causal-conv-shim", action="store_true",
                          help="Use shimmed causal_conv1d via torch.library.custom_op. T-3.2.")
    ak_group.add_argument("--ak-normuon-telemetry", action="store_true",
                          help="Emit per-param NorMuon update telemetry JSONL (T-0.2).")
    ak_group.add_argument("--ak-normuon-impl-opt", action="store_true",
                          help="Enable NorMuon implementation optimizations: "
                               "preallocated buffers, grouped NS, branchless SPECTRA. T-2.3.")
    ak_group.add_argument("--ak-trust-cap", type=float, default=0.0,
                          help="Post-NorMuon trust cap threshold on ||update*lr|| / ||w||. "
                               "0.0 (default) = disabled. Typical diagnostic: 0.02. T-5.3.")
    ak_group.add_argument("--ak-trust-cap-scope",
                          choices=["none", "w_gate_up", "spiking", "all_2d"],
                          default="none",
                          help="Which 2D params get trust-cap applied (T-5.3).")
    ak_group.add_argument("--ak-w-gate-up-scale", type=float, default=1.0,
                          help="Initial w_gate_up post-NorMuon update scale (T-5.2). "
                               "Default 1.0 = no staging. Try 0.25 for warm-ramp.")
    ak_group.add_argument("--ak-w-gate-up-ramp-steps", type=int, default=0,
                          help="Steps over which --ak-w-gate-up-scale ramps to 1.0 (T-5.2). "
                               "0 = no ramp, scale stays at initial value.")
    ak_group.add_argument("--assert-no-sync", action="store_true",
                          help="Abort training if DDP allreduce_count > 1 per opt step "
                               "(catches accumulation no_sync() regression). T-0.6.")

    args = parser.parse_args()

    # v3 alias: --ak-loss-zloss implies --use-fused-zloss (single implementation).
    if getattr(args, "ak_loss_zloss", False):
        args.use_fused_zloss = True

    # v3-legacy: these env vars are no longer read by conv_blocks.py (the
    # autograd-safe path is now unconditional). Kept for backward compat with
    # older launch scripts that pass --ak-fix-rope-gate-op / --ak-causal-conv-shim.
    if getattr(args, "ak_fix_rope_gate_op", False):
        os.environ["AUTOKERNEL_FIX_ROPE_GATE"] = "1"
    if getattr(args, "ak_spectra_branchless", False) or getattr(args, "ak_sync_cleanup", False):
        os.environ["AUTOKERNEL_SPECTRA_BRANCHLESS"] = "1"
    if getattr(args, "ak_causal_conv_shim", False):
        os.environ["AUTOKERNEL_CAUSAL_CONV_SHIM"] = "1"

    # v3 T-4 compiled autograd activation. Must be set BEFORE any
    # torch.compile call. Risk: may regress DDP allreduce overlap
    # (PyTorch DDP notes warn whole-backward compilation prevents hooks
    # from firing until compiled backward finishes). T-0.6 DDP trace will
    # capture this. Continue-gate: >=2.5-3% net DDP tok/s + no overlap
    # regression + no recompile churn.
    if getattr(args, "ak_compiled_autograd", False):
        import torch._dynamo as _dynamo
        try:
            _dynamo.config.compiled_autograd = True
            print("[compiled-autograd] enabled: torch._dynamo.config.compiled_autograd = True")
        except AttributeError:
            # Older torch versions don't have this field; try the runtime enabler.
            try:
                import torch._dynamo.compiled_autograd as _ca
                _ca.enable(torch.compile(fullgraph=False))
                print("[compiled-autograd] enabled via runtime context (older torch)")
            except Exception as _e:
                print(f"[compiled-autograd] enable failed: {_e}")

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

        # v3 T-0.5 flag manifest: print every active --ak-* flag.
        # Every flag records itself in the run manifest (Rule 3).
        _ak_active = {k: v for k, v in vars(args).items()
                      if k.startswith("ak_") and v not in (False, 0, 0.0, "none", "")}
        if _ak_active:
            print(f"[ak-manifest] active: {_ak_active}")
        else:
            print("[ak-manifest] no --ak-* flags active (baseline mode)")

        # v3 T-0.6 DDP trace: always-on per-opt-step JSONL.
        # Populate shared trace state; activate if checkpoint_dir available.
        if args.checkpoint_dir:
            _DDP_TRACE_STATE["enabled"] = True
            _DDP_TRACE_STATE["accum_steps"] = args.accum_steps
            _DDP_TRACE_STATE["path"] = os.path.join(
                args.checkpoint_dir, f"ddp_trace_rank{rank}.jsonl"
            )
            _DDP_TRACE_STATE["assert_no_sync"] = getattr(args, "assert_no_sync", False)
            print(f"[ddp-trace] enabled: writing to {_DDP_TRACE_STATE['path']}")
            if _DDP_TRACE_STATE["assert_no_sync"]:
                print(f"[ddp-trace] --assert-no-sync active: "
                      f"will abort if allreduce_count > 1 per opt step")

    # --- Model ---
    sys.path.insert(0, ".")
    # Sprint 1.5 Phase A: pass μP construction-time kwargs when --mup is set.
    # load_model_from_file filters kwargs per ctor signature so older models
    # whose ctor predates these flags are unaffected.
    model_ctor_kwargs = {}
    if getattr(args, "mup", False):
        model_ctor_kwargs["use_mup"] = True
        model_ctor_kwargs["mup_base_width"] = getattr(args, "mup_base_width", 256)
    model = load_model_from_file(args.model, args.class_name, **model_ctor_kwargs)
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

    # fp16-stability P6: attention score softcap. Sets the attribute on
    # every Attention instance in the model graph (works for OdinHalo,
    # OdinFlat, and any halo variant using models.components.attention).
    if args.attn_softcap > 0:
        n_patched = 0
        from models.components.attention import Attention as _AttnBase
        for mod in model.modules():
            if isinstance(mod, _AttnBase):
                mod.attn_score_softcap = float(args.attn_softcap)
                n_patched += 1
        if rank == 0:
            print(f"[fp16-stability] attn_score_softcap={args.attn_softcap} "
                  f"applied to {n_patched} Attention modules")

    # autokernel BEFORE checkpoint load (checkpoint has fused QKV keys)
    if args.optimize_kernels:
        try:
            import autokernel
            # Phase II: optionally filter autokernel patterns via include/exclude lists.
            inc = [s.strip() for s in args.autokernel_include.split(",") if s.strip()]
            exc = [s.strip() for s in args.autokernel_exclude.split(",") if s.strip()]
            ak_kwargs = {"training": True}
            if inc:
                ak_kwargs["include"] = inc
            if exc:
                ak_kwargs["exclude"] = exc
            model = autokernel.optimize(model, **ak_kwargs)
            if rank == 0:
                tag_parts = []
                if inc:
                    tag_parts.append(f"include={inc}")
                if exc:
                    tag_parts.append(f"exclude={exc}")
                tag = f" ({', '.join(tag_parts)})" if tag_parts else ""
                print(f"autokernel optimizations applied{tag}")
                # Emit the applied-patterns report so bisect logs capture ground truth
                try:
                    report = autokernel.report(model)
                    for pn, info in report["patterns"].items():
                        print(f"  [autokernel] {pn}: "
                              f"{info['modules_replaced']} modules")
                except Exception:
                    pass
        except Exception as e:
            if rank == 0:
                print(f"autokernel skipped: {e}")

        # Phase E.3 (2026-05-11): autograd-safety preflight.
        # After pattern replacement, dispatch a dummy forward+backward and
        # verify every parameter with requires_grad=True received a finite
        # gradient. Catches raw-pybind replacements that sever gradient
        # flow (see docs/perf/autokernel-deep-analysis.md for history).
        if rank == 0:
            try:
                _preflight_ok, _preflight_msg = _autokernel_autograd_preflight(
                    model, device, args
                )
                if not _preflight_ok:
                    raise RuntimeError(
                        f"autokernel preflight FAILED: {_preflight_msg}\n"
                        f"This means --optimize-kernels installed a "
                        f"Replacement that severs gradient flow. Do NOT "
                        f"train with this configuration; loss will descend "
                        f"but model quality degrades. See "
                        f"docs/perf/autokernel-deep-analysis.md. "
                        f"Workaround: pass --no-optimize-kernels, OR run "
                        f"scripts/audit_autokernel_replacements.py and "
                        f"file a fix. Abort."
                    )
                print(f"  [autokernel] preflight OK: all parameters "
                      f"received gradients after dummy forward+backward")
            except RuntimeError:
                raise
            except Exception as _pfexc:  # noqa: BLE001
                # Preflight infra itself broke; don't block launch on that,
                # but emit a visible warning.
                print(f"  [autokernel] preflight skipped due to harness "
                      f"error: {_pfexc}")

    # Resume (Approach B: weights only, fresh optimizer)
    _resume_ckpt_cache = None  # v3 T-5 C.1: optimizer/scaler state to restore post-construction
    if args.resume_from:

        # fp16-stability P5: tighten max_grad_norm default on resumed runs.
        # Accumulated weight magnitude in a resumed model is higher than at
        # fresh init, making large grads more likely to overflow fp16. Only
        # overrides if user kept argparse's default (1.0); explicit user value
        # is preserved.
        if args.max_grad_norm == 1.0:
            args.max_grad_norm = 0.8
            if rank == 0:
                print(f"[fp16-stability] --resume-from set; "
                      f"tightening --max-grad-norm 1.0 -> 0.8 "
                      f"(pass --max-grad-norm 1.0 explicitly to disable)")
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
        # v3 T-5 C.1: optionally preserve optimizer + scaler state. Default
        # remains weights-only (fresh optimizer) for back-compat; the
        # preserve-state variant is gated by --resume-preserve-optimizer.
        if getattr(args, "resume_preserve_optimizer", False):
            _resume_ckpt_cache = ckpt  # keep for use after optimizer is constructed
            if rank == 0:
                print(f"  Resumed from step {ckpt.get('step', '?')}, optimizer+scaler state PRESERVED")
        else:
            if rank == 0:
                print(f"  Resumed from step {ckpt.get('step', '?')}, fresh optimizer")
            del ckpt
        del state_dict

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

    # Phase 3 LEAP: construct aux-loss handler AFTER compile_zones so hooks
    # attach to the OptimizedModule wrappers (compile_zones replaces
    # `model.layers[i]` in-place, which severs hooks registered beforehand).
    leap = None
    if args.leap_layers.strip():
        from halo_training.leap_layer_exit import LeapAuxLoss
        layer_idx_list = [int(x) for x in args.leap_layers.split(",") if x.strip()]
        if args.leap_weights.strip():
            weight_list = [float(x) for x in args.leap_weights.split(",") if x.strip()]
        else:
            weight_list = None
        leap_target = model.module if hasattr(model, "module") else model
        if hasattr(leap_target, "_orig_mod"):
            leap_target = leap_target._orig_mod
        try:
            leap = LeapAuxLoss(
                leap_target, layer_indices=layer_idx_list,
                weights=weight_list, tau=args.leap_tau,
                layers_attr=args.leap_layers_attr,
            )
            if rank == 0:
                print(f"[LEAP] aux loss ON: layers={layer_idx_list} "
                      f"weights={weight_list or 'uniform 1.0'} tau={args.leap_tau}")
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                print(f"[LEAP] init failed ({exc}); running without LEAP")
            leap = None

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
            # Sprint 1.1 Phase B: resolve ns_dtype string -> torch.dtype
            ns_dtype = torch.float16 if args.ns_dtype == "fp16" else torch.float32
            optimizer = build_imu1_optimizer(
                raw_model,
                lr_2d=lr_2d,
                lr_1d=lr_1d,
                weight_decay_2d=0.1,
                betas=(0.9, 0.95),
                use_normuon=args.normuon,
                ns_dtype=ns_dtype,
                neuron_norm_min_dim=args.neuron_norm_min_dim,
                cautious_wd=args.cautious_wd,
                # Sprint 1.5 Phase A: opt-in SPECTRA + μP
                use_mup=getattr(args, "mup", False),
                mup_base_width=getattr(args, "mup_base_width", 256),
                spectra_post=getattr(args, "spectra_post", False),
                spectra_clip_norm=getattr(args, "spectra_clip_norm", 1.0),
                spectra_ns_iter=getattr(args, "spectra_ns_iter", 5),
                # v3 40k campaign (T-0.2 telemetry + T-5.2/5.3 recovery knobs
                # + T-2.3 NorMuon implementation optimizations)
                telemetry_enabled=getattr(args, "ak_normuon_telemetry", False),
                telemetry_path=(
                    os.path.join(args.checkpoint_dir or ".", f"normuon_telem_rank{rank}.jsonl")
                    if getattr(args, "ak_normuon_telemetry", False) else None
                ),
                trust_cap=getattr(args, "ak_trust_cap", 0.0),
                trust_cap_scope=getattr(args, "ak_trust_cap_scope", "none"),
                w_gate_up_scale=getattr(args, "ak_w_gate_up_scale", 1.0),
                w_gate_up_ramp_steps=getattr(args, "ak_w_gate_up_ramp_steps", 0),
                spectra_branchless=(
                    getattr(args, "ak_spectra_branchless", False)
                    or getattr(args, "ak_sync_cleanup", False)
                    or getattr(args, "ak_normuon_impl_opt", False)
                ),
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
    # fp16-stability P4: growth_interval 2000 -> 500. Faster overflow response
    # and slower scale runaway on long-horizon runs. GradScaler doubles scale
    # after this many consecutive non-overflow steps; 500 means ~5min (at
    # 100ms/step) between scale doublings, keeping scale bounded enough that
    # a single outlier batch cannot overflow catastrophically.
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, backoff_factor=0.25,
                                  growth_interval=500)

    # v3 T-5 C.1: restore optimizer + scaler state when --resume-preserve-optimizer
    # was set. The checkpoint was cached during the resume_from block above;
    # we apply it here now that optimizer + scaler exist.
    if getattr(args, "resume_preserve_optimizer", False) and _resume_ckpt_cache is not None:
        _rc = _resume_ckpt_cache
        try:
            if "optimizer_state_dict" in _rc:
                optimizer.load_state_dict(_rc["optimizer_state_dict"])
                if rank == 0:
                    print("  [resume] optimizer state restored")
            else:
                if rank == 0:
                    print("  [resume] WARN: no optimizer_state_dict in checkpoint; "
                          "using fresh optimizer")
        except Exception as _exc:
            if rank == 0:
                print(f"  [resume] WARN: optimizer restore failed ({_exc}); fresh state")
        try:
            if "scaler_state_dict" in _rc:
                scaler.load_state_dict(_rc["scaler_state_dict"])
                if rank == 0:
                    print("  [resume] scaler state restored")
            elif "scaler" in _rc:  # v3 replay bundle layout fallback
                scaler.load_state_dict(_rc["scaler"])
                if rank == 0:
                    print("  [resume] scaler state restored (replay-bundle layout)")
        except Exception as _exc:
            if rank == 0:
                print(f"  [resume] WARN: scaler restore failed ({_exc}); fresh state")
        del _resume_ckpt_cache, _rc


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

    # fp16-stability D1: opt-in activation monitor (rank 0 only to avoid
    # file contention; per-layer stats are model-identical across ranks).
    monitor = None
    if args.activation_monitor and rank == 0:
        from halo_training.activation_monitor import ActivationMonitor
        monitor_out = os.path.join(args.checkpoint_dir or "checkpoints/ddp",
                                    "activation_stats.jsonl")
        monitor = ActivationMonitor(
            model,
            output_path=monitor_out,
            sample_every=args.activation_monitor_interval,
        )
        monitor.attach()
        # Arm for the first opt step (global_step increments to 1 before
        # monitor.step() is called, so forwards of step 1 need set_step(1)).
        monitor.set_step(1)
        print(f"[fp16-stability] activation monitor ON: "
              f"interval={args.activation_monitor_interval}, "
              f"out={monitor_out}, "
              f"tracked={len(monitor._tracked)} layers")
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
    # v3 T-1.2: tensor-side step loss accumulator (used only when
    # --ak-sync-cleanup is on; else None and per-microstep .item() path runs).
    _step_loss_t = None
    best_loss = float("inf")
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = 0
    deadline = start_time + args.time_budget * 60 if args.time_budget > 0 else None
    ckpt_every = args.checkpoint_interval or (args.log_interval * 10)
    last_grad_norm = torch.tensor(0.0)

    # Async state: pending allreduce handles from previous optimizer step
    pending_handles = None

    # fp16-stability guards initialization
    _z_loss_warned = False  # one-shot warning when z-loss skipped
    _leap_warned = False    # one-shot warning when LEAP disabled mid-run
    # Ring buffer of recent grad norms — surfaced in NaN forensics dump (R1).
    from collections import deque
    recent_grad_norms: deque = deque(maxlen=50)
    # Track rank0 scaler-scale-warning debounce (R5).
    _last_scale_warn_step = -10_000

    # Bug-fix (2026-05-07): track microbatch backwards in the current accum
    # cycle. Used by the end-of-training flush path: if we broke out mid-cycle
    # (e.g. --max-steps hit before the cycle completed), we must NOT call
    # _complete_step on the pending_handles — the scaler is in post-update()
    # READY state and unscale_ would assert. Was papered over with
    # try/except AssertionError; now handled cleanly with counter-gated flush.
    backwards_in_cycle = 0

    _rollback_restart = False

    # Track 1.1: step profiler state. Rank-0 only.
    _prof = None
    _prof_start_step = -1
    _prof_end_step = -1
    if args.profile_steps and rank == 0:
        try:
            _prof_start_step, _prof_end_step = [int(x) for x in args.profile_steps.split(":")]
        except ValueError as exc:
            raise ValueError(
                f"--profile-steps must be 'start:end' with ints, got "
                f"{args.profile_steps!r}"
            ) from exc
        if _prof_end_step <= _prof_start_step:
            raise ValueError(
                f"--profile-steps end ({_prof_end_step}) must be > start "
                f"({_prof_start_step})"
            )
        print(f"  [profiler] Armed for opt steps "
              f"[{_prof_start_step}, {_prof_end_step})")

    # Track 3.A: --diag-frozen-params JSONL writer. Rank-0 only.
    _diag_fh = None
    diag_writer = None
    if args.diag_frozen_params and rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.diag_frozen_params)) or ".",
                    exist_ok=True)
        _diag_fh = open(args.diag_frozen_params, "w")
        print(f"  [diag-frozen-params] Writing per-param grads to "
              f"{args.diag_frozen_params}")

        def diag_writer(m):
            """Closure over `global_step` + `_diag_fh`. Invoked post-clip,
            pre-zero_grad by _complete_step."""
            rec = {"step": global_step, "params": []}
            for name, p in m.named_parameters():
                if p.grad is None:
                    rec["params"].append({
                        "name": name, "grad_norm": None,
                        "is_none": True, "is_zero": False,
                    })
                else:
                    g = p.grad.detach()
                    gn = float(g.norm().item())
                    rec["params"].append({
                        "name": name, "grad_norm": gn,
                        "is_none": False, "is_zero": (gn == 0.0),
                    })
            _diag_fh.write(json.dumps(rec) + "\n")
            _diag_fh.flush()

    try:
        for epoch in range(args.epochs):
            if _rollback_restart:
                _rollback_restart = False
            sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(dataloader):
                # Track 1.1: manage profiler lifecycle at opt-step boundaries.
                # Only fires between microsteps when global_step has advanced.
                if rank == 0 and _prof_start_step >= 0:
                    if _prof is None and global_step == _prof_start_step:
                        from torch.profiler import profile, ProfilerActivity
                        _prof = profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=True,
                            with_stack=True,
                        )
                        _prof.__enter__()
                        print(f"  [profiler] Started capture at opt step "
                              f"{global_step}")
                    elif _prof is not None and global_step >= _prof_end_step:
                        _prof.__exit__(None, None, None)
                        out_dir = args.checkpoint_dir or "."
                        os.makedirs(out_dir, exist_ok=True)
                        try:
                            _prof.export_chrome_trace(os.path.join(out_dir, "profile.json"))
                        except Exception as exc:
                            print(f"  [profiler] export_chrome_trace failed: {exc}")
                        summary_path = os.path.join(out_dir, "profile-summary.txt")
                        with open(summary_path, "w") as _fh:
                            _fh.write(_prof.key_averages().table(
                                sort_by="cuda_time_total", row_limit=40))
                            _fh.write("\n\n=== sort by cpu_time_total ===\n\n")
                            _fh.write(_prof.key_averages().table(
                                sort_by="cpu_time_total", row_limit=40))
                            _fh.write("\n\n=== sort by self_cuda_time_total ===\n\n")
                            _fh.write(_prof.key_averages().table(
                                sort_by="self_cuda_time_total", row_limit=40))
                        # v3 T-0.8 sync counter: extract aten::item count + hipMemcpyWithStream
                        # CPU wall from profile and emit JSONL per v3 §9.4 schema.
                        try:
                            import json as _json
                            _prof_window = _prof_end_step - _prof_start_step
                            _item_count = 0
                            _memcpy_cpu_s = 0.0
                            for _ev in _prof.key_averages():
                                if _ev.key == "aten::item":
                                    _item_count = _ev.count
                                elif _ev.key == "hipMemcpyWithStream":
                                    _memcpy_cpu_s = _ev.cpu_time_total / 1_000_000.0
                            _sync_record = {
                                "profile_window_steps": _prof_window,
                                "aten_item_count": int(_item_count),
                                "hipMemcpyWithStream_cpu_wall_s": float(_memcpy_cpu_s),
                                "known_hot_syncs": {
                                    "spectra_sigma1_item": "unknown",
                                    "loss_item": "unknown",
                                    "valid_global_sum_item": "unknown",
                                    "jsonl_logging": "unknown",
                                },
                            }
                            _sync_path = os.path.join(out_dir, "sync-counter.jsonl")
                            with open(_sync_path, "a", buffering=1) as _sf:
                                _sf.write(_json.dumps(_sync_record) + "\n")
                            print(f"  [profiler] sync counter: aten::item={_item_count} "
                                  f"calls over {_prof_window} opt steps "
                                  f"(~{_item_count / max(_prof_window, 1):.0f}/step)")
                        except Exception as _e:
                            print(f"  [profiler] sync counter extraction failed: {_e}")
                        print(f"  [profiler] Wrote {summary_path} and "
                              f"{out_dir}/profile.json at opt step {global_step}")
                        _prof = None
                        _prof_start_step = -1  # disable re-entry

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
                            # z-loss needs logits; scalar-output path can't apply it.
                            # Warn once so users see the flag was silently skipped.
                            if args.z_loss > 0 and not _z_loss_warned:
                                if rank == 0:
                                    print(f"  [fp16-stability] --z-loss {args.z_loss} "
                                          f"requested but model returns scalar loss; "
                                          f"z-loss requires logits path. Skipping.")
                                _z_loss_warned = True
                        elif isinstance(output, dict):
                            logits = output["logits"]
                            # Phase B.5 (2026-05-11): opt-in fused z-loss via
                            # kernel.ce_full. Bakes z-loss into the HIP kernel
                            # forward AND adds the correct gradient contribution
                            # back to logits in backward. Eliminates the
                            # separate aten::logsumexp pass (16.7% of step).
                            z_active = (args.z_loss > 0
                                        and global_step < total_steps * args.z_loss_fraction)
                            if (args.use_fused_zloss and z_active
                                    and logits.dtype == torch.float16
                                    and logits.is_cuda):
                                import kernel as _ce_k
                                logits_flat = logits.view(-1, logits.size(-1))
                                loss = _ce_k.ce_full(
                                    logits_flat, targets.view(-1),
                                    softcap=0.0, ignore_index=-100,
                                    label_smoothing=0.0,
                                    mode="tiny",
                                    z_loss_weight=float(args.z_loss),
                                ) / args.accum_steps
                                if "mtp1" in output:
                                    mtp1 = output["mtp1"]
                                    mtp_targets = targets[:, 2:].reshape(-1)
                                    loss = loss + 0.3 * ce_loss_fn(
                                        mtp1.reshape(-1, logits.size(-1)), mtp_targets
                                    ) / args.accum_steps
                            else:
                                loss = ce_loss_fn(
                                    logits.view(-1, logits.size(-1)), targets.view(-1)
                                ) / args.accum_steps
                                if "mtp1" in output:
                                    mtp1 = output["mtp1"]
                                    mtp_targets = targets[:, 2:].reshape(-1)
                                    loss = loss + 0.3 * ce_loss_fn(
                                        mtp1.reshape(-1, logits.size(-1)), mtp_targets
                                    ) / args.accum_steps
                                # fp16-stability P1: z-loss auxiliary regularization.
                                # Penalizes drift of log-partition-function magnitudes.
                                # Active during first z_loss_fraction of training only.
                                # Track 2.b (2026-05-10): compute logsumexp in fp16
                                # (safe with --attn-softcap 50 → lse ≤ ~60.4, well
                                # within fp16 range) then promote the small [B*T]
                                # vector to fp32 for the pow(2).mean() reduction.
                                # Avoids a 1 GB fp32 copy of the [B, T, V] logits
                                # tensor — previously the #3 op by wall time (11%).
                                if z_active:
                                    z_loss_val = args.z_loss * logits.logsumexp(dim=-1).float().pow(2).mean()
                                    loss = loss + z_loss_val / args.accum_steps
                        else:
                            # Phase B.5 (2026-05-11): same fused-zloss path for
                            # the raw-tensor logits output convention.
                            z_active = (args.z_loss > 0
                                        and global_step < total_steps * args.z_loss_fraction)
                            if (args.use_fused_zloss and z_active
                                    and output.dtype == torch.float16
                                    and output.is_cuda):
                                import kernel as _ce_k
                                loss = _ce_k.ce_full(
                                    output.view(-1, output.size(-1)),
                                    targets.view(-1),
                                    softcap=0.0, ignore_index=-100,
                                    label_smoothing=0.0,
                                    mode="tiny",
                                    z_loss_weight=float(args.z_loss),
                                ) / args.accum_steps
                            else:
                                loss = ce_loss_fn(
                                    output.view(-1, output.size(-1)), targets.view(-1)
                                ) / args.accum_steps
                                # fp16-stability P1: z-loss on raw-logits tensor path.
                                # Track 2.b (2026-05-10): see note above.
                                if z_active:
                                    z_loss_val = args.z_loss * output.logsumexp(dim=-1).float().pow(2).mean()
                                    loss = loss + z_loss_val / args.accum_steps

                        # Phase 3 LEAP: add layer-exit aux loss if enabled.
                        # `leap` is None unless --leap-layers was passed.
                        # Uses auto-captured h_final from final_norm hook.
                        if leap is not None:
                            try:
                                leap_loss = leap.compute_aux_loss()
                                loss = loss + leap_loss / args.accum_steps
                            except RuntimeError as exc:
                                # e.g. torch.compile bypassed hooks → disable.
                                if rank == 0 and not _leap_warned:
                                    print(f"  [LEAP] disabled: {exc}")
                                leap.close()
                                leap = None
                                _leap_warned = True

                    scaler.scale(loss).backward()
                    # Bug-fix (2026-05-07): track microbatch backwards in
                    # the current accum cycle for end-of-training flush logic.
                    backwards_in_cycle += 1

                total_tokens += input_ids.numel()
                # v3 T-1.2 hot-path sync removal: when --ak-sync-cleanup is on,
                # accumulate loss tensor-side and defer .item() to step boundary.
                # At accum=8 this reduces per-opt-step .item() calls from 8 to 1.
                # NaN detection moves to end-of-step (GradScaler handles grad NaN
                # independently via inf-grad skip).
                if getattr(args, "ak_sync_cleanup", False):
                    # Lazy init of tensor accumulator
                    if "_step_loss_t" not in locals() or _step_loss_t is None:
                        _step_loss_t = torch.zeros((), device=device, dtype=torch.float32)
                    _step_loss_t = _step_loss_t + loss.detach().float()
                    loss_val = 0.0  # deferred; won't be used for per-microstep branch
                else:
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
                        diag_writer=diag_writer,
                    )
                    global_step += 1
                    # v3 T-0.6 DDP trace: emit JSONL for this step + reset state.
                    _ddp_trace_emit(world_size, rank)
                    _ddp_trace_reset_step(global_step)
                    # v3 T-1.2: materialize deferred loss accumulator (1 sync/step
                    # vs previous 8/step at accum=8). Replaces per-microstep .item().
                    if getattr(args, "ak_sync_cleanup", False):
                        try:
                            if "_step_loss_t" in locals() and _step_loss_t is not None:
                                step_loss = float(_step_loss_t.item())
                                _step_loss_t = None  # reset for next opt step
                        except Exception:
                            step_loss = 0.0
                    # Bug-fix: completed optimizer step consumed this cycle's
                    # accumulated backwards; reset counter for next cycle.
                    backwards_in_cycle = 0

                    # fp16-stability R1: ring buffer of recent grad norms for
                    # post-NaN forensics dump.
                    if hasattr(last_grad_norm, "item"):
                        recent_grad_norms.append(
                            (global_step, float(last_grad_norm.item()))
                        )

                    # fp16-stability D1: sample activation stats (no-op if off)
                    if monitor is not None:
                        monitor.step(global_step)
                        # Arm hooks for next opt step's microstep forwards.
                        # Disarmed steps fully no-op → ~10-15% throughput
                        # recovery on looped models (was `.item()`-per-forward).
                        monitor.set_step(global_step + 1)

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
                        # fp16-stability R5: surface GradScaler scale.
                        try:
                            scale_val = float(scaler.get_scale())
                        except Exception:
                            scale_val = float("nan")
                        print(
                            f"[step {global_step:>6d}] "
                            f"loss={avg_loss:.4f} bpb={bpb:.3f} lr={lr:.2e} "
                            f"grad={last_grad_norm:.2f} tok/s={tok_s:,.0f} "
                            f"mfu={mfu:.1%} mem={mem_gb:.1f}GB scale={scale_val:.1e}"
                        )
                        # Threshold warning (debounced, once per 1000 steps).
                        if scale_val > 16384.0 and (global_step - _last_scale_warn_step) >= 1000:
                            print(f"  [fp16-stability] WARN: scaler.scale={scale_val:.1e} "
                                  f"is high; fp16 grad headroom is low. Consider: "
                                  f"--resume-from + --max-grad-norm 0.5 + --z-loss 1e-4")
                            _last_scale_warn_step = global_step
                        if log_file:
                            with open(log_file, "a") as f:
                                f.write(json.dumps({
                                    "step": global_step, "loss": avg_loss,
                                    "bpb": bpb, "lr": lr, "tok_s": tok_s,
                                    "mfu": mfu, "mem_gb": mem_gb,
                                    "grad_norm": last_grad_norm.item() if hasattr(last_grad_norm, "item") else last_grad_norm,
                                    "scaler_scale": scale_val,
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
                    not guard.check_params(model, global_step) or
                    not guard.check_scaler(scaler, global_step)
                ):
                    # fp16-stability R1: dump forensics BEFORE rollback so
                    # we capture the exact state that tripped the guard.
                    # Rank 0 only; fail-quiet.
                    if rank == 0 and args.checkpoint_dir:
                        if too_many_skips:
                            trigger = "grad_skips"
                        elif math.isnan(step_loss) or math.isinf(step_loss):
                            trigger = "nan_loss"
                        elif guard.loss_ema is not None and step_loss > guard.spike_factor * guard.loss_ema:
                            trigger = "loss_spike"
                        else:
                            # At this point the trigger was either param_nan
                            # (check_params scan) or scale_collapse (check_scaler).
                            # Disambiguate cheaply:
                            try:
                                scale_now = float(scaler.get_scale())
                            except Exception:
                                scale_now = None
                            if scale_now is not None and scale_now < guard.scale_floor:
                                trigger = "scale_collapse"
                            else:
                                trigger = "param_nan"
                        save_nan_forensics(
                            dump_dir=args.checkpoint_dir,
                            step=global_step,
                            trigger=trigger,
                            loss_val=step_loss,
                            batch_idx=batch_idx,
                            input_ids=input_ids,
                            targets=targets,
                            doc_ids=doc_ids,
                            scaler=scaler,
                            model=model,
                            recent_grad_norms=recent_grad_norms,
                            monitor=monitor,
                            global_step=global_step,
                            optimizer=optimizer,
                            args=args,
                            consecutive_grad_skips=_consecutive_skips,
                        )
                    if too_many_skips and rank == 0:
                        print(f"  [StabilityGuard] {_consecutive_skips} consecutive grad skips, rolling back")
                    optimizer.zero_grad(set_to_none=True)
                    if pending_handles is not None:
                        for h in pending_handles:
                            h.wait()
                        pending_handles = None
                    # Bug-fix: rollback discards the in-flight accum cycle.
                    backwards_in_cycle = 0
                    # Save emergency checkpoint before rollback
                    if rank == 0 and args.checkpoint_dir:
                        saved_path = save_checkpoint(model, optimizer, global_step,
                                                     args.checkpoint_dir, total_tokens * world_size)
                        if args.auto_eval:
                            spawn_auto_eval(saved_path, args.model, args.class_name)
                    rollback_step, ok = guard.rollback(model, optimizer, device, scaler=scaler)
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
                        diag_writer=diag_writer,
                    )
                    global_step += 1
                    # v3 T-0.6 DDP trace emit + reset on sync path
                    _ddp_trace_emit(world_size, rank)
                    _ddp_trace_reset_step(global_step)
                    # v3 T-1.2 deferred loss materialization (sync path)
                    if getattr(args, "ak_sync_cleanup", False):
                        try:
                            if "_step_loss_t" in locals() and _step_loss_t is not None:
                                step_loss = float(_step_loss_t.item())
                                _step_loss_t = None
                        except Exception:
                            step_loss = 0.0
                    pending_handles = None
                    # Bug-fix: sync path also consumed the accum cycle.
                    backwards_in_cycle = 0

            else:
                continue
            if _rollback_restart:
                continue
            break

        # Flush final pending allreduce.
        # Bug-fix (2026-05-07): only call _complete_step if at least one
        # microbatch backward happened in the current accum cycle. Without
        # this guard, --max-steps termination AFTER a completed step could
        # reach here with pending_handles referencing already-completed
        # grads and backwards_in_cycle == 0, tripping an assertion inside
        # scaler.unscale_ (state machine expects a scaled backward since
        # last update). Wait for the handles to land regardless so the
        # collective ops don't leave the backend in a bad state.
        if pending_handles is not None:
            for h in pending_handles:
                h.wait()
            if backwards_in_cycle > 0:
                average_grads(model, world_size)
                if use_fp16:
                    decompress_grads_fp32(model)
                _complete_step(model, optimizer, scaler, scheduler,
                               world_size, args.max_grad_norm, rank,
                               diag_writer=diag_writer)
                global_step += 1
                backwards_in_cycle = 0
            else:
                if rank == 0:
                    print(f"[rank 0] Flush skipped _complete_step "
                          f"(backwards_in_cycle=0 after early termination); "
                          f"handles waited cleanly")

    except KeyboardInterrupt:
        if rank == 0:
            print(f"\nInterrupted at step {global_step}")

    # Track 1.1: close profile if training ended mid-capture (flushes trace).
    if rank == 0 and _prof is not None:
        try:
            _prof.__exit__(None, None, None)
            out_dir = args.checkpoint_dir or "."
            os.makedirs(out_dir, exist_ok=True)
            try:
                _prof.export_chrome_trace(os.path.join(out_dir, "profile.json"))
            except Exception as exc:
                print(f"  [profiler] export_chrome_trace failed at shutdown: {exc}")
            with open(os.path.join(out_dir, "profile-summary.txt"), "w") as _fh:
                _fh.write(_prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=40))
                _fh.write("\n\n=== sort by cpu_time_total ===\n\n")
                _fh.write(_prof.key_averages().table(
                    sort_by="cpu_time_total", row_limit=40))
            print(f"  [profiler] Flushed profile on shutdown ({out_dir})")
        except Exception as exc:
            print(f"  [profiler] shutdown error: {exc}")

    # Track 3.A: close diag JSONL.
    if _diag_fh is not None:
        try:
            _diag_fh.close()
            print(f"  [diag-frozen-params] Closed "
                  f"{args.diag_frozen_params}")
        except Exception:
            pass

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
