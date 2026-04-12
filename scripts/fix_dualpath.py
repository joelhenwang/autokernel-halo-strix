"""Diagnostic + fix for DualCortex and Obsidian dual-path failures.

Tests:
1. Eager (no autokernel) at d_fast=256 — isolate if architecture or autokernel
2. Autokernel at d_fast=512 — test if larger fast path fixes the issue
3. Eager at d_fast=512 — baseline for the fixed architecture

Usage:
    python scripts/fix_dualpath.py
"""

import os
import sys
import time
import math
import importlib.util
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from halo_training.data import BabyLMDataset


def load(path, cls, kwargs):
    spec = importlib.util.spec_from_file_location("m", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls)(**kwargs)


def evaluate(model, val_loader, device):
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for inp, tgt in val_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(inp)
                logits = out if isinstance(out, torch.Tensor) else out.logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
            total_loss += loss.item() * inp.numel()
            total_tok += inp.numel()
    model.train()
    return total_loss / total_tok


def train_run(label, model, train_loader, val_loader, device, epochs=2, log_every=400):
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.1, fused=True)
    total_steps = len(train_loader) * epochs
    warmup = min(100, total_steps // 10)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    scaler = torch.amp.GradScaler("cuda")

    step = 0
    t0 = time.time()
    running, n_batch = 0.0, 0

    for epoch in range(epochs):
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(inp)
                logits = out if isinstance(out, torch.Tensor) else out.logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n_batch += 1
            step += 1

            if step % log_every == 0:
                avg = running / n_batch
                tok_s = step * 16 * 256 / (time.time() - t0)
                print(f"  [{label:30s}] step {step:>5d}  loss={avg:.4f}  tok/s={tok_s:,.0f}")
                running, n_batch = 0.0, 0

        val = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"  [{label:30s}] epoch {epoch+1}  val={val:.4f}  ({elapsed:.0f}s)")

    final_val = evaluate(model, val_loader, device)
    total_time = time.time() - t0
    tok_s = step * 16 * 256 / total_time
    print(f"  [{label:30s}] DONE  val={final_val:.4f}  tok/s={tok_s:,.0f}  {total_time:.0f}s  {n_params/1e6:.1f}M")
    return final_val, tok_s, n_params


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")

    ds = BabyLMDataset(root="datasets/babylm-strict-small", block_size=256)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"Data: {n_train} train / {n_val} val")
    print()

    results = []

    # =========================================================================
    # TEST 1: DualCortex d_fast=256 EAGER (no autokernel)
    # =========================================================================
    print("=" * 70)
    print("TEST 1: DualCortex d_fast=256 EAGER")
    print("=" * 70)
    kw = dict(vocab_size=50257, d_embed=896, d_fast=256, d_slow=896,
              n_fast_layers=6, n_slow_layers=8, ffn_fast=512, ffn_slow=2304)
    m = load("models/dual_cortex.py", "DualCortex", kw)
    torch.cuda.reset_peak_memory_stats()
    val, tok, n_p = train_run("DC d=256 eager", m, train_loader, val_loader, device)
    results.append(("DualCortex d=256 eager", n_p, val, tok))
    del m; torch.cuda.empty_cache(); gc.collect()
    print()

    # =========================================================================
    # TEST 2: DualCortex d_fast=512 EAGER
    # =========================================================================
    print("=" * 70)
    print("TEST 2: DualCortex d_fast=512 EAGER")
    print("=" * 70)
    kw512 = dict(vocab_size=50257, d_embed=896, d_fast=512, d_slow=896,
                 n_fast_layers=6, n_slow_layers=6, ffn_fast=1024, ffn_slow=2304)
    m = load("models/dual_cortex.py", "DualCortex", kw512)
    torch.cuda.reset_peak_memory_stats()
    val, tok, n_p = train_run("DC d=512 eager", m, train_loader, val_loader, device)
    results.append(("DualCortex d=512 eager", n_p, val, tok))
    del m; torch.cuda.empty_cache(); gc.collect()
    print()

    # =========================================================================
    # TEST 3: DualCortex d_fast=512 AUTOKERNEL
    # =========================================================================
    print("=" * 70)
    print("TEST 3: DualCortex d_fast=512 AUTOKERNEL")
    print("=" * 70)
    m = load("models/dual_cortex.py", "DualCortex", kw512)
    m = m.to(device).train()
    import autokernel
    m = autokernel.optimize(m, training=True)
    torch.cuda.reset_peak_memory_stats()
    val, tok, n_p = train_run("DC d=512 autokernel", m, train_loader, val_loader, device)
    results.append(("DualCortex d=512 autokernel", n_p, val, tok))
    del m; torch.cuda.empty_cache(); gc.collect()
    print()

    # =========================================================================
    # TEST 4: Obsidian d_reflex=256 EAGER
    # =========================================================================
    print("=" * 70)
    print("TEST 4: Obsidian d_reflex=256 EAGER")
    print("=" * 70)
    okw = dict(vocab_size=50257, d_model=896, d_reflex=256,
               n_reflex_layers=6, d_reflex_ffn=512,
               n_genius_layers=8, d_genius_rec=896, ffn_genius=2304)
    m = load("models/obsidian.py", "Obsidian", okw)
    torch.cuda.reset_peak_memory_stats()
    val, tok, n_p = train_run("Obs d=256 eager", m, train_loader, val_loader, device)
    results.append(("Obsidian d=256 eager", n_p, val, tok))
    del m; torch.cuda.empty_cache(); gc.collect()
    print()

    # =========================================================================
    # TEST 5: Obsidian d_reflex=512 EAGER
    # =========================================================================
    print("=" * 70)
    print("TEST 5: Obsidian d_reflex=512 EAGER")
    print("=" * 70)
    okw512 = dict(vocab_size=50257, d_model=896, d_reflex=512,
                  n_reflex_layers=6, d_reflex_ffn=1024,
                  n_genius_layers=6, d_genius_rec=896, ffn_genius=2304)
    m = load("models/obsidian.py", "Obsidian", okw512)
    torch.cuda.reset_peak_memory_stats()
    val, tok, n_p = train_run("Obs d=512 eager", m, train_loader, val_loader, device)
    results.append(("Obsidian d=512 eager", n_p, val, tok))
    del m; torch.cuda.empty_cache(); gc.collect()
    print()

    # =========================================================================
    # TEST 6: Obsidian d_reflex=512 AUTOKERNEL
    # =========================================================================
    print("=" * 70)
    print("TEST 6: Obsidian d_reflex=512 AUTOKERNEL")
    print("=" * 70)
    m = load("models/obsidian.py", "Obsidian", okw512)
    m = m.to(device).train()
    m = autokernel.optimize(m, training=True)
    torch.cuda.reset_peak_memory_stats()
    val, tok, n_p = train_run("Obs d=512 autokernel", m, train_loader, val_loader, device)
    results.append(("Obsidian d=512 autokernel", n_p, val, tok))
    del m; torch.cuda.empty_cache(); gc.collect()
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 90)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 90)
    print(f"{'Config':35s} {'Params':>8s} {'Val Loss':>10s} {'tok/s':>10s} {'Verdict':>15s}")
    print("-" * 90)
    for name, n_p, val, tok in results:
        if val > 5.0:
            verdict = "FAILED"
        elif val > 3.5:
            verdict = "WEAK"
        else:
            verdict = "OK"
        print(f"{name:35s} {n_p/1e6:7.1f}M {val:>10.4f} {tok:>10,.0f} {verdict:>15s}")

    print()
    print("DIAGNOSIS:")
    # Check if d=256 eager works
    dc_eager_256 = [r for r in results if "DC d=256 eager" in r[0]]
    obs_eager_256 = [r for r in results if "Obs d=256 eager" in r[0]]
    if dc_eager_256 and dc_eager_256[0][2] < 4.0:
        print("  DualCortex d=256 works in eager -> autokernel is the problem")
    elif dc_eager_256:
        print("  DualCortex d=256 fails even in eager -> architecture problem")

    if obs_eager_256 and obs_eager_256[0][2] < 4.0:
        print("  Obsidian d=256 works in eager -> autokernel is the problem")
    elif obs_eager_256:
        print("  Obsidian d=256 fails even in eager -> architecture problem")


if __name__ == "__main__":
    main()
