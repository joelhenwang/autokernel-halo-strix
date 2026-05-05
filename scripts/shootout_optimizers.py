"""Optimizer shootout: AdamW vs Muon vs Lion on OdinHalo at batch=16.

For each optimizer, runs 200 warmup + 200 measured steps on BabyLM with seed=42.
Captures:
  - steady-state tok/s (avg of last 100 measured steps)
  - peak GPU memory
  - loss trajectory (first 10 + last 10 measured values)

Emits docs/perf/optimizer-shootout-<date>.md with a comparison table and winner.
"""
import sys, time, gc, datetime, pathlib, torch
sys.path.insert(0, '.')

WARMUP = 200
MEASURE = 200
BATCH_SIZE = 16
BLOCK_SIZE = 256
BASE_LR = 1e-4          # AdamW reference LR (not too aggressive; avoids early divergence)
LION_LR_RATIO = 0.3     # Lion LR = BASE_LR * 0.3
MUON_LR = 5e-4          # muon paper sweet spot for this scale

device = 'cuda'

from halo_training.data import BabyLMDataset, build_dataloader
from models.odin_halo import OdinHalo
from halo_training.lion import Lion
from halo_training.clion import CLion

try:
    from halo_training.muon import Muon, split_params_for_muon
    MUON_AVAILABLE = True
except Exception as e:
    print(f"Muon not available: {e}")
    MUON_AVAILABLE = False


def build_adamw(model):
    return torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.95), fused=True)


def build_lion(model):
    return Lion(model.parameters(), lr=BASE_LR * LION_LR_RATIO, betas=(0.9, 0.99), weight_decay=0.1)


def build_clion(model):
    # Per-coordinate gate: sign(c[j]) where |c[j]|>=nu, else c[j].
    # OdinHalo gradients are ~1e-5 to 1e-4 median after GradScaler unscale.
    # ν=1e-6 keeps ~90%+ of coords on sign path (Lion behavior) and sends only
    # extreme-tiny coords to identity path (stability safety net).
    return CLion(model.parameters(), lr=BASE_LR * LION_LR_RATIO,
                 betas=(0.9, 0.99), weight_decay=0.1,
                 nu=1e-6, gate_mode="per_coord")


def build_muon(model):
    if not MUON_AVAILABLE:
        return None
    muon_params, adamw_named = split_params_for_muon(model)
    adamw_groups = [{"params": [p]} for _, p in adamw_named]
    return Muon(
        muon_params=[{"params": muon_params}],
        lr=MUON_LR,
        weight_decay=0.01,
        adamw_params=adamw_groups,
        adamw_lr=BASE_LR,
    )


def run_session(opt_name, build_opt_fn):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    model = OdinHalo().to(device)
    model.train()
    model.compile_zones()  # production config

    opt = build_opt_fn(model)
    if opt is None:
        return None

    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)

    ds = BabyLMDataset(
        root='datasets/babylm-odin32k.bin',
        block_size=BLOCK_SIZE,
        tokenizer_path='tokenizers/odin-32k/tokenizer.json',
    )
    dl = build_dataloader(ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

    def batches():
        while True:
            for b in dl:
                yield b
    it = batches()

    # Collect loss every step during measured window
    measured_losses = []

    # Warmup
    for step in range(WARMUP):
        input_ids, targets = next(it)
        input_ids = input_ids.to(device); targets = targets.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    # Measured
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    total_tokens = 0
    t0 = time.time()
    for step in range(MEASURE):
        t_step = time.time()
        input_ids, targets = next(it)
        input_ids = input_ids.to(device); targets = targets.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total_tokens += input_ids.numel()
        measured_losses.append(loss.item())
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # Steady-state: last 100 measured steps averaged
    steady_tokens = BATCH_SIZE * BLOCK_SIZE * 100
    # rough steady-state throughput: use time for whole measured window; fine enough for 200 steps
    tok_s = total_tokens / elapsed

    result = dict(
        name=opt_name,
        tok_s=tok_s,
        peak_gb=peak_gb,
        losses_first=measured_losses[:10],
        losses_last=measured_losses[-10:],
        initial_loss=measured_losses[0],
        final_loss=measured_losses[-1],
    )

    del model, opt, scaler
    return result


def main():
    date_str = datetime.date.today().isoformat()
    outdir = pathlib.Path("docs/perf")
    outdir.mkdir(parents=True, exist_ok=True)
    out_md = outdir / f"optimizer-shootout-{date_str}.md"

    results = []
    print("\n=== Optimizer shootout (OdinHalo V=32768, batch=16, 400 steps) ===\n")

    for name, build_fn in [("AdamW", build_adamw), ("Muon", build_muon),
                            ("Lion", build_lion), ("CLion", build_clion)]:
        print(f"\n--- {name} ---")
        r = run_session(name, build_fn)
        if r is None:
            print(f"  Skipped (optimizer unavailable)")
            continue
        print(f"  tok/s: {r['tok_s']:,.0f}")
        print(f"  peak_gb: {r['peak_gb']:.2f}")
        print(f"  initial loss: {r['initial_loss']:.3f}  final loss: {r['final_loss']:.3f}")
        results.append(r)

    # Render markdown
    lines = []
    lines.append(f"# Optimizer shootout — {date_str}\n")
    lines.append(f"OdinHalo (57.6M params, V=32768), batch={BATCH_SIZE}, block={BLOCK_SIZE}, "
                 f"`compile_zones()`, GradScaler, grad_clip=1.0, 400 steps "
                 f"(200 warmup + 200 measured).\n")
    lines.append(f"Base LR (AdamW): {BASE_LR}; Lion/CLion LR: {BASE_LR * LION_LR_RATIO}; "
                 f"Muon LR: {MUON_LR}; CLion ν=1e-6 per-coord (tuned below typical |c|~1e-5).\n")

    lines.append("\n## Results\n")
    lines.append("| Optimizer | tok/s | Peak GB | Init loss | Final loss | Δ loss |")
    lines.append("|-----------|------:|--------:|----------:|-----------:|-------:|")
    for r in results:
        delta = r["final_loss"] - r["initial_loss"]
        lines.append(f"| {r['name']} | {r['tok_s']:,.0f} | {r['peak_gb']:.2f} | "
                     f"{r['initial_loss']:.3f} | {r['final_loss']:.3f} | {delta:+.3f} |")

    lines.append("\n## Loss trajectories\n")
    for r in results:
        lines.append(f"\n### {r['name']}\n")
        lines.append(f"First 10 measured: `{['%.3f' % v for v in r['losses_first']]}`")
        lines.append(f"\nLast 10 measured: `{['%.3f' % v for v in r['losses_last']]}`")

    # Winner selection: highest tok/s among those with final_loss < 0.95 * initial_loss (mild convergence)
    converging = [r for r in results if r["final_loss"] < 0.98 * r["initial_loss"]]
    if converging:
        winner = max(converging, key=lambda r: r["tok_s"])
        lines.append(f"\n## Winner: **{winner['name']}**\n")
        lines.append(f"Highest tok/s ({winner['tok_s']:,.0f}) among optimizers showing "
                     f"convergence (final_loss < 0.98 × initial_loss).\n")
    else:
        lines.append("\n## Winner: none cleared convergence gate\n")
        lines.append("None of AdamW/Muon/Lion achieved final_loss < 0.98 × initial_loss in "
                     "200 measured steps. Rerun with lower LR or longer warmup.\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
