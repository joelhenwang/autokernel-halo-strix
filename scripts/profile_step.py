"""Single-node profiler for one training step — Sprint 1.1 Phase A.

Mirrors the step path in ``scripts/train_ddp.py`` (forward + backward +
GradScaler unscale/clip/step) but runs on a single GPU with no DDP,
no async allreduce, no stability guard, no logging. Its only job is to
produce a ``torch.profiler`` op-level table sorted by ``cuda_time_total``
so we can attribute step time to specific ops (NorMuon's Newton-Schulz,
neuron-wise normalize, cautious-WD mask, etc.).

Accepts the same Sprint 1 CLI flags as ``train_ddp.py``:
  --model, --class-name, --dataset
  --imu1-groups, --normuon, --lr-2d, --lr-1d, --no-muon
  --intra-doc-mask, --value-residuals, --head-gating
  --compile, --block-size, --batch-size

Usage example (Phase A commands from the plan):
  python scripts/profile_step.py \\
      --model models/odin_flat.py --class-name OdinFlat \\
      --dataset datasets/wikitext-103-odin32k.bin \\
      --warmup 50 --measure 100 \\
      --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --no-muon \\
      --intra-doc-mask --value-residuals --head-gating --compile

Output is a human-readable profiler table on stdout (>> redirect to a
file). Sprint 1.1 Phase A commits three such tables to ``docs/perf/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from halo_training.data import PreTokenizedDataset


def load_model_from_file(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)()


def _model_accepts_doc_ids(model) -> bool:
    import inspect
    core = model._orig_mod if hasattr(model, "_orig_mod") else model
    try:
        return "doc_ids" in inspect.signature(core.forward).parameters
    except (TypeError, ValueError):
        return False


def _forward_loss(model, input_ids, targets, doc_ids, ce_loss_fn):
    """Mirror train_ddp.py's forward dispatch (tensor / dict / logits)."""
    if doc_ids is not None and _model_accepts_doc_ids(model):
        output = model(input_ids, targets=targets, doc_ids=doc_ids)
    else:
        output = model(input_ids, targets=targets)
    if isinstance(output, torch.Tensor) and output.dim() == 0:
        return output
    if isinstance(output, dict):
        logits = output["logits"]
        return ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
    return ce_loss_fn(output.view(-1, output.size(-1)), targets.view(-1))


def main():
    p = argparse.ArgumentParser(description="Single-node step profiler (Sprint 1.1 Phase A)")
    # model + data
    p.add_argument("--model", required=True)
    p.add_argument("--class-name", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    # profiling knobs
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--measure", type=int, default=100)
    p.add_argument("--row-limit", type=int, default=40,
                   help="Rows in printed profiler table.")
    p.add_argument("--sort-by", default="cuda_time_total",
                   help="torch.profiler key_averages().table sort_by key")
    # optimizer flags (mirror train_ddp.py)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--muon-lr", type=float, default=0.005)
    p.add_argument("--no-muon", action="store_true")
    p.add_argument("--imu1-groups", dest="imu1_groups", action="store_true")
    p.add_argument("--no-imu1-groups", dest="imu1_groups", action="store_false")
    p.set_defaults(imu1_groups=False)
    p.add_argument("--lr-2d", type=float, default=None)
    p.add_argument("--lr-1d", type=float, default=None)
    p.add_argument("--normuon", dest="normuon", action="store_true")
    p.add_argument("--no-normuon", dest="normuon", action="store_false")
    p.set_defaults(normuon=False)
    # Sprint 1 model flags
    p.add_argument("--intra-doc-mask", dest="intra_doc_mask", action="store_true")
    p.add_argument("--no-intra-doc-mask", dest="intra_doc_mask", action="store_false")
    p.set_defaults(intra_doc_mask=False)
    p.add_argument("--value-residuals", dest="value_residuals", action="store_true")
    p.add_argument("--no-value-residuals", dest="value_residuals", action="store_false")
    p.set_defaults(value_residuals=False)
    p.add_argument("--head-gating", dest="head_gating", action="store_true")
    p.add_argument("--no-head-gating", dest="head_gating", action="store_false")
    p.set_defaults(head_gating=False)
    # compile
    p.add_argument("--compile", action="store_true")
    # misc
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)

    # --- Model ---
    print(f"Loading model: {args.model}::{args.class_name}")
    sys.path.insert(0, ".")
    model = load_model_from_file(args.model, args.class_name).to(device)

    # Sprint 1 feature flags (ignored silently if model doesn't expose attr)
    sprint1_flag_map = {
        "use_intra_doc_mask": bool(args.intra_doc_mask),
        "use_value_residuals": bool(args.value_residuals),
        "use_head_gating": bool(args.head_gating),
    }
    for attr, value in sprint1_flag_map.items():
        if hasattr(model, attr):
            setattr(model, attr, value)
    active_flags = {k: ("ON" if v else "off") for k, v in sprint1_flag_map.items()
                    if hasattr(model, k)}
    if active_flags:
        print(f"[Sprint 1 flags on {type(model).__name__}] {active_flags}")

    model.train()

    # torch.compile (must be before optimizer construction so any fused
    # ops materialize under the compiled path)
    if args.compile:
        if hasattr(model, "compile_zones"):
            print("Compiling model (per-zone for looped model)...")
            model.compile_zones()
        else:
            print("Compiling model (default mode, no cudagraphs)...")
            model = torch.compile(model, mode="default")

    # --- Optimizer (mirror train_ddp.py's logic) ---
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if args.no_muon:
        if args.imu1_groups:
            from halo_training.optimizer import build_imu1_optimizer
            lr_2d = args.lr_2d if args.lr_2d is not None else args.lr
            lr_1d = args.lr_1d if args.lr_1d is not None else args.lr * 0.3
            optimizer = build_imu1_optimizer(
                raw_model,
                lr_2d=lr_2d, lr_1d=lr_1d,
                weight_decay_2d=0.1, betas=(0.9, 0.95),
                use_normuon=args.normuon,
            )
        else:
            optimizer = torch.optim.AdamW(
                raw_model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                weight_decay=0.1, fused=True,
            )
            print(f"Using AdamW (lr={args.lr}, wd=0.1, fused=True)")
    else:
        from halo_training.muon import Muon, split_params_for_muon
        muon_params, adamw_named = split_params_for_muon(raw_model)
        adamw_groups = []
        for name, param in adamw_named:
            if "norm" in name or (name.endswith(".bias") and "decay" not in name):
                adamw_groups.append({"params": [param], "lr": args.lr, "weight_decay": 0})
            else:
                adamw_groups.append({"params": [param], "lr": args.lr, "weight_decay": 0.1})
        optimizer = Muon(
            muon_params=[{"params": muon_params}], lr=args.muon_lr, weight_decay=0.01,
            adamw_params=adamw_groups, adamw_lr=args.lr, ns_dtype=torch.float16,
        )
        print(f"Using Muon (muon_lr={args.muon_lr}, adamw_lr={args.lr})")

    # --- Data ---
    dataset = PreTokenizedDataset(args.dataset, block_size=args.block_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
    )
    batches = iter(dataloader)

    # --- Loss + AMP ---
    ce_loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, backoff_factor=0.25)

    # --- Step closure ---
    def step_once():
        try:
            batch = next(batches)
        except StopIteration:
            return False
        if len(batch) == 3:
            input_ids, targets, doc_ids = batch
        else:
            input_ids, targets = batch
            doc_ids = None
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        if doc_ids is not None:
            doc_ids = doc_ids.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = _forward_loss(model, input_ids, targets, doc_ids, ce_loss_fn)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        return True

    n_params = sum(p.numel() for p in raw_model.parameters())
    eff_toks = args.batch_size * args.block_size
    print(f"\nProfile config: {n_params/1e6:.1f}M params, "
          f"batch={args.batch_size}, block={args.block_size}, "
          f"tokens/step={eff_toks}")
    print(f"Warmup steps: {args.warmup}   Measured steps: {args.measure}")

    # --- Warmup ---
    print(f"\nWarming up ({args.warmup} steps)...")
    t0 = time.time()
    for i in range(args.warmup):
        if not step_once():
            batches = iter(dataloader)
            step_once()
    torch.cuda.synchronize()
    warmup_elapsed = time.time() - t0
    print(f"Warmup done in {warmup_elapsed:.1f}s "
          f"({warmup_elapsed/args.warmup*1000:.1f} ms/step avg)")

    # --- Profiled measurement ---
    print(f"\nProfiling {args.measure} steps...")
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    t0 = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for i in range(args.measure):
            if not step_once():
                batches = iter(dataloader)
                step_once()
    torch.cuda.synchronize()
    measured_elapsed = time.time() - t0
    per_step_ms = measured_elapsed / args.measure * 1000
    tok_s = (args.measure * eff_toks) / measured_elapsed
    print(f"Measured: {measured_elapsed:.1f}s  "
          f"{per_step_ms:.1f} ms/step  {tok_s:,.0f} tok/s (single-node)")

    print(f"\n=== TOP OPS BY {args.sort_by.upper()} "
          f"(avg across {args.measure} steps) ===")
    print(prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit))

    # Also emit a summary block at the end so grep-friendly
    print("\n=== SUMMARY ===")
    print(f"config: model={args.class_name} batch={args.batch_size} "
          f"block={args.block_size}")
    print(f"  optimizer: no_muon={args.no_muon} imu1_groups={args.imu1_groups} "
          f"normuon={args.normuon} lr_2d={args.lr_2d} lr_1d={args.lr_1d}")
    print(f"  sprint1: intra_doc_mask={args.intra_doc_mask} "
          f"value_residuals={args.value_residuals} head_gating={args.head_gating}")
    print(f"  compile: {args.compile}")
    print(f"steps: warmup={args.warmup} measured={args.measure}")
    print(f"timing: {per_step_ms:.2f} ms/step  {tok_s:,.0f} tok/s")


if __name__ == "__main__":
    main()
