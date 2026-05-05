"""WI-B3: max-autotune compile mode test.

Compares compile modes: default, reduce-overhead (Track A, known bad), max-autotune.
max-autotune adds triton autotune passes which can find better kernel configs
at the cost of 3-10x warmup time.

Usage:
  python scripts/wi_b3_max_autotune.py --warmup 30 --measured 100 --repeat 3
"""
import argparse, time, statistics, json, pathlib, datetime, sys
sys.path.insert(0, '.')
import torch


def measure_config(label, mode, warmup, measured, batch=16, block=256):
    torch.manual_seed(42)
    from models.odin_halo import OdinHalo
    from halo_training.data import BabyLMDataset, build_dataloader
    try:
        from kernels.hip._torch_ops import disable_hip_backward
        disable_hip_backward()
    except ImportError:
        pass

    model = OdinHalo().to('cuda')
    model.train()

    if mode == "none":
        pass
    elif mode == "default_compile_zones":
        model.compile_zones()  # uses default mode per-layer
    else:
        # Apply custom compile mode per-layer
        for i in range(len(model.shared_layers)):
            model.shared_layers[i] = torch.compile(
                model.shared_layers[i], mode=mode)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1024.0)

    ds = BabyLMDataset(
        root='datasets/babylm-odin32k.bin',
        block_size=block,
        tokenizer_path='tokenizers/odin-32k/tokenizer.json',
    )
    dl = build_dataloader(ds, batch_size=batch, num_workers=0, shuffle=True)
    it = iter(dl)

    def gb_next():
        nonlocal it
        try: return next(it)
        except StopIteration:
            it = iter(dl); return next(it)

    def step():
        input_ids, targets = gb_next()
        input_ids = input_ids.to('cuda', non_blocking=True)
        targets = targets.to('cuda', non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    print(f"  [{label}] warmup {warmup} steps...", end=" ", flush=True)
    t_w = time.perf_counter()
    for _ in range(warmup):
        last_loss = step()
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t_w
    print(f"done in {warmup_s:.1f}s (loss={last_loss:.3f})")

    t0 = time.perf_counter()
    n_tok = 0
    for _ in range(measured):
        last_loss = step()
        n_tok += batch * block
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    tok_s = n_tok / (t1 - t0)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "label": label, "mode": mode, "warmup_s": warmup_s,
        "tok_s": tok_s, "peak_gb": peak_gb, "final_loss": last_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=str,
                        default="docs/perf/phase3-wi-b3-max-autotune.md")
    args = parser.parse_args()

    # max-autotune: enables autotune_pointwise, coordinate descent tuning, etc.
    # max-autotune-no-cudagraphs: same but without cudagraph capture (safer on HIP)
    configs = [
        ("A: compile_zones (default mode, baseline)", "default_compile_zones"),
        ("B: per-layer max-autotune", "max-autotune"),
        ("C: per-layer max-autotune-no-cudagraphs", "max-autotune-no-cudagraphs"),
    ]

    results = {label: [] for label, _ in configs}
    for label, mode in configs:
        for r in range(args.repeat):
            print(f"Run {r+1}/{args.repeat} for {label}")
            try:
                res = measure_config(label, mode, args.warmup, args.measured)
                results[label].append(res)
            except Exception as e:
                print(f"  Error in {label}: {e}")
                results[label].append({"label": label, "mode": mode,
                                       "error": str(e), "tok_s": 0.0,
                                       "warmup_s": 0.0, "peak_gb": 0.0,
                                       "final_loss": 0.0})

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Strategy':<55}  {'median tok/s':>12}  {'Peak GB':>8}  {'warmup':>8}")
    print("-" * 80)
    baseline_med = None
    for label, _ in configs:
        runs = [r for r in results[label] if r.get("tok_s", 0) > 0]
        if not runs:
            print(f"{label:<55}  ERROR"); continue
        med_tok = statistics.median(r["tok_s"] for r in runs)
        med_gb = statistics.median(r["peak_gb"] for r in runs)
        med_w = statistics.median(r["warmup_s"] for r in runs)
        if baseline_med is None: baseline_med = med_tok
        pct = 100.0 * (med_tok - baseline_med) / baseline_med
        print(f"{label:<55}  {med_tok:>12,.0f}  {med_gb:>8.2f}  {med_w:>8.1f}  ({pct:+.2f}%)")
    print("=" * 80)

    # Write
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    lines = [
        f"# WI-B3: max-autotune compile mode — {date_str}\n",
        f"**Config:** OdinHalo batch=16 block=256 warmup={args.warmup} measured={args.measured} repeat={args.repeat}\n",
        "## Summary\n",
        "| Strategy | Median tok/s | Stdev | Peak GB | Warmup (s) | Final loss | vs A |",
        "|----------|-------------:|------:|--------:|-----------:|-----------:|-----:|",
    ]
    baseline_med = None
    for label, _ in configs:
        runs = [r for r in results[label] if r.get("tok_s", 0) > 0]
        if not runs:
            lines.append(f"| {label} | ERROR | - | - | - | - | - |"); continue
        toks = [r["tok_s"] for r in runs]
        med_tok = statistics.median(toks)
        sd_tok = statistics.stdev(toks) if len(toks) > 1 else 0
        med_gb = statistics.median(r["peak_gb"] for r in runs)
        med_w = statistics.median(r["warmup_s"] for r in runs)
        med_loss = statistics.median(r["final_loss"] for r in runs)
        if baseline_med is None: baseline_med = med_tok
        pct = 100.0 * (med_tok - baseline_med) / baseline_med
        lines.append(f"| {label} | {med_tok:,.0f} | {sd_tok:.0f} | {med_gb:.2f} | "
                     f"{med_w:.1f} | {med_loss:.4f} | {pct:+.2f}% |")

    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
