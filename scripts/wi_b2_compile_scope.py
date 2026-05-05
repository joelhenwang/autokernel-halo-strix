"""WI-B2: Compile-per-Parcae-iteration throughput test.

Compares three compile strategies:
  A: baseline compile_zones() - per-layer compile (6 graphs, each called 3 times)
  B: compile _run_shared_block - one graph spanning all 6 layers (1 function,
     called 3 times with same signature)
  C: compile the whole _forward_unrolled - maximum compile scope

Strategy B captures cross-layer residual adds that are currently eager
(Phase 2 identified these as 29% of aten::add_ at shape (16,256,768)).

Usage:
  python scripts/wi_b2_compile_scope.py --warmup 20 --measured 100 --repeat 3
"""
import argparse, time, statistics, json, pathlib, datetime, sys
sys.path.insert(0, '.')
import torch


def measure_config(label, compile_strategy, warmup, measured, batch=16, block=256):
    torch.manual_seed(42)
    from models.odin_halo import OdinHalo
    from halo_training.data import BabyLMDataset, build_dataloader

    model = OdinHalo().to('cuda')
    model.train()

    # Apply compile strategy
    if compile_strategy == "A":
        # Current: compile each shared_layers[i]
        model.compile_zones()
    elif compile_strategy == "B":
        # Compile _run_shared_block (spans all 6 layers per call)
        try:
            from kernels.hip._torch_ops import disable_hip_backward
            disable_hip_backward()
        except ImportError:
            pass
        # Bind the method via torch.compile
        original = model._run_shared_block
        compiled = torch.compile(original, mode="default")
        import types
        model._run_shared_block = compiled
    elif compile_strategy == "C":
        # Compile the entire _forward_unrolled
        try:
            from kernels.hip._torch_ops import disable_hip_backward
            disable_hip_backward()
        except ImportError:
            pass
        model._forward_unrolled = torch.compile(
            model._forward_unrolled, mode="default")
    elif compile_strategy == "none":
        pass
    else:
        raise ValueError(f"Unknown strategy: {compile_strategy}")

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
        "label": label,
        "strategy": compile_strategy,
        "warmup_s": warmup_s,
        "tok_s": tok_s,
        "peak_gb": peak_gb,
        "final_loss": last_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--out", type=str, default="docs/perf/phase3-wi-b2-compile-scope.md")
    args = parser.parse_args()

    configs = [
        ("A: compile_zones (per-layer)", "A"),
        ("B: compile _run_shared_block (per-iter-body)", "B"),
        ("C: compile _forward_unrolled (whole-model)", "C"),
    ]

    results = {label: [] for label, _ in configs}
    for label, strat in configs:
        for r in range(args.repeat):
            print(f"Run {r+1}/{args.repeat} for {label}")
            try:
                res = measure_config(label, strat, args.warmup, args.measured,
                                     batch=args.batch, block=args.block)
                results[label].append(res)
            except Exception as e:
                print(f"  Error in {label}: {e}")
                results[label].append({"label": label, "strategy": strat,
                                       "error": str(e), "tok_s": 0.0,
                                       "warmup_s": 0.0, "peak_gb": 0.0,
                                       "final_loss": 0.0})

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Strategy':<50}  {'median tok/s':>12}  {'Peak GB':>8}  {'warmup':>8}")
    print("-" * 80)
    baseline_med = None
    for label, _ in configs:
        runs = results[label]
        oks = [r for r in runs if r.get("tok_s", 0) > 0]
        if not oks:
            print(f"{label:<50}  {'ERROR':>12}")
            continue
        med_tok = statistics.median(r["tok_s"] for r in oks)
        med_gb = statistics.median(r["peak_gb"] for r in oks)
        med_w = statistics.median(r["warmup_s"] for r in oks)
        if baseline_med is None:
            baseline_med = med_tok
        pct = 100.0 * (med_tok - baseline_med) / baseline_med
        print(f"{label:<50}  {med_tok:>12,.0f}  {med_gb:>8.2f}  {med_w:>8.1f}  ({pct:+.2f}%)")
    print("=" * 80)

    # Write report
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    lines = [
        f"# WI-B2: Compile scope sweep — {date_str}\n",
        f"**Config:** OdinHalo batch={args.batch} block={args.block} warmup={args.warmup} measured={args.measured} repeat={args.repeat}\n",
        "## Summary\n",
        "| Strategy | Median tok/s | Stdev | Peak GB | Warmup (s) | Final loss | vs A |",
        "|----------|-------------:|------:|--------:|-----------:|-----------:|-----:|",
    ]
    baseline_med = None
    for label, _ in configs:
        runs = results[label]
        oks = [r for r in runs if r.get("tok_s", 0) > 0]
        if not oks:
            lines.append(f"| {label} | ERROR | - | - | - | - | - |")
            continue
        toks = [r["tok_s"] for r in oks]
        med_tok = statistics.median(toks)
        sd_tok = statistics.stdev(toks) if len(toks) > 1 else 0
        med_gb = statistics.median(r["peak_gb"] for r in oks)
        med_w = statistics.median(r["warmup_s"] for r in oks)
        med_loss = statistics.median(r["final_loss"] for r in oks)
        if baseline_med is None:
            baseline_med = med_tok
        pct = 100.0 * (med_tok - baseline_med) / baseline_med
        lines.append(f"| {label} | {med_tok:,.0f} | {sd_tok:.0f} | {med_gb:.2f} | {med_w:.1f} | {med_loss:.4f} | {pct:+.2f}% |")

    lines.append("\n## Runs\n")
    for label, _ in configs:
        runs = results[label]
        lines.append(f"### {label}\n")
        for i, r in enumerate(runs, 1):
            if r.get("tok_s", 0) > 0:
                lines.append(f"- run {i}: {r['tok_s']:,.0f} tok/s, {r['peak_gb']:.2f} GB, "
                             f"warmup {r['warmup_s']:.1f}s, loss {r['final_loss']:.4f}")
            else:
                lines.append(f"- run {i}: ERROR ({r.get('error', 'unknown')})")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")
    out.with_suffix(".json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
