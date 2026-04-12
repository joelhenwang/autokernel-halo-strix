"""Head-to-head: AMADEUS vs MAESTRO-PRIMA (conductor vs no conductor).

Same data split, same hyperparameters, same seed. Logs val loss every 200 steps.
"""

import math
import os
import sys
import time
import importlib.util

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


def train_run(name, model, train_loader, val_loader, device, epochs=2, log_every=200):
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device).train()

    import autokernel
    model = autokernel.optimize(model, training=True)

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
    checkpoints = []  # (step, train_loss, val_loss)
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
                val = evaluate(model, val_loader, device)
                tok_s = step * 16 * 256 / (time.time() - t0)
                checkpoints.append((step, avg, val))
                print(f"  [{name:15s}] step {step:>5d}  train={avg:.4f}  val={val:.4f}  tok/s={tok_s:,.0f}")
                running, n_batch = 0.0, 0

        val = evaluate(model, val_loader, device)
        print(f"  [{name:15s}] epoch {epoch+1} done  val={val:.4f}")

    elapsed = time.time() - t0
    final_val = evaluate(model, val_loader, device)
    print(f"  [{name:15s}] DONE  val={final_val:.4f}  {elapsed:.0f}s  {n_params/1e6:.1f}M params")
    return checkpoints, final_val


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Same dataset, same split for both
    ds = BabyLMDataset(root="datasets/babylm-strict-small", block_size=256)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"Data: {n_train} train / {n_val} val chunks")
    print()

    kw = dict(vocab_size=50257, d_model=896, n_layers=12, d_conv=512, d_mamba=384,
              dstate=64, n_ssm_heads=6, ffn_inner=2304, d_film=64, film_start=6)

    # Run 1: AMADEUS
    print("=== AMADEUS (no conductor) ===")
    m1 = load("models/amadeus.py", "Amadeus", kw)
    torch.cuda.reset_peak_memory_stats()
    a_ckpts, a_val = train_run("Amadeus", m1, train_loader, val_loader, device)
    del m1
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print()

    # Run 2: MAESTRO-PRIMA
    print("=== MAESTRO-PRIMA (with conductor) ===")
    m2 = load("models/maestro_prima.py", "MaestroPrima", kw)
    torch.cuda.reset_peak_memory_stats()
    mp_ckpts, mp_val = train_run("MaestroPrima", m2, train_loader, val_loader, device)
    del m2
    torch.cuda.empty_cache()
    gc.collect()
    print()

    # Head-to-head table
    print("=" * 85)
    print("HEAD-TO-HEAD: Amadeus vs MaestroPrima (conductor effect)")
    print("=" * 85)
    header = f"{'Step':>6s}  {'A train':>9s} {'A val':>8s}  {'MP train':>9s} {'MP val':>8s}  {'Val diff':>9s}"
    print(header)
    print("-" * 85)

    for (s1, t1, v1), (s2, t2, v2) in zip(a_ckpts, mp_ckpts):
        diff = v2 - v1
        marker = " <-- MP better" if diff < -0.01 else (" <-- A better" if diff > 0.01 else "")
        print(f"{s1:>6d}  {t1:>9.4f} {v1:>8.4f}  {t2:>9.4f} {v2:>8.4f}  {diff:>+9.4f}{marker}")

    print()
    print(f"Final val loss:  Amadeus = {a_val:.4f}  |  MaestroPrima = {mp_val:.4f}  |  diff = {mp_val - a_val:+.4f}")
    if abs(mp_val - a_val) < 0.01:
        print("Verdict: NO SIGNIFICANT DIFFERENCE — conductor has negligible effect at this scale.")
    elif mp_val < a_val:
        print(f"Verdict: CONDUCTOR HELPS — MaestroPrima is {(a_val - mp_val):.4f} better.")
    else:
        print(f"Verdict: CONDUCTOR HURTS — Amadeus is {(mp_val - a_val):.4f} better.")


if __name__ == "__main__":
    main()
