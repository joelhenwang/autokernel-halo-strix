"""WI-B1: Shape sweep for OdinHalo throughput.

Measures tok/s at combinations of (block_size, batch_size) holding total tokens
per step ~constant. rocBLAS kernel performance is shape-sensitive; Inductor's
fused kernels are also tile-size-sensitive. Looking for a shape combination
that gives >=+5% tok/s at equivalent throughput budget.

Matrix (constant ~4096 tokens/step):
  block=128 batch=32 = 4096 tokens
  block=192 batch=24 = 4608 tokens (next natural fit)
  block=256 batch=16 = 4096 tokens (current baseline)
  block=384 batch=12 = 4608 tokens (next natural fit)
  block=512 batch=8  = 4096 tokens

Plus a memory-perm matrix (higher token budget):
  block=256 batch=32 = 8192 tokens (if memory allows)
  block=128 batch=64 = 8192 tokens

Memory ceiling: 5.5 GB peak (leaves 3 GB headroom on 8 GB budget).

Usage:
  python scripts/wi_b1_shape_sweep.py --warmup 20 --measured 100
"""
import os, sys, time, argparse, json, datetime, pathlib
sys.path.insert(0, '.')


def measure_shape(batch_size, block_size, warmup, measured, compile_mode="default"):
    import torch
    torch.manual_seed(42)

    result = {
        "batch_size": batch_size,
        "block_size": block_size,
        "tokens_per_step": batch_size * block_size,
        "compile_mode": compile_mode,
        "status": "UNKNOWN",
        "warmup_s": 0.0,
        "tok_s": 0.0,
        "peak_gb": 0.0,
        "final_loss": None,
        "error": None,
    }

    try:
        from models.odin_halo import OdinHalo
        from halo_training.data import BabyLMDataset, build_dataloader
        model = OdinHalo().to('cuda')
        model.train()
        model.compile_zones()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1024.0)

        ds = BabyLMDataset(
            root='datasets/babylm-odin32k.bin',
            block_size=block_size,
            tokenizer_path='tokenizers/odin-32k/tokenizer.json',
        )
        dl = build_dataloader(ds, batch_size=batch_size, num_workers=0, shuffle=True)
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

        print(f"  [b={batch_size},T={block_size}] warmup {warmup} steps...", end=" ", flush=True)
        t_w = time.perf_counter()
        for _ in range(warmup):
            last_loss = step()
        torch.cuda.synchronize()
        result["warmup_s"] = time.perf_counter() - t_w
        print(f"done in {result['warmup_s']:.1f}s (loss={last_loss:.3f})")

        print(f"  [b={batch_size},T={block_size}] measuring {measured} steps...",
              end=" ", flush=True)
        t0 = time.perf_counter()
        n_tok = 0
        for _ in range(measured):
            last_loss = step()
            n_tok += batch_size * block_size
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        result["tok_s"] = n_tok / (t1 - t0)
        result["peak_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        result["final_loss"] = last_loss
        result["status"] = "OK"
        print(f"{result['tok_s']:,.0f} tok/s, {result['peak_gb']:.2f} GB")
    except Exception as e:
        import traceback
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"ERROR: {e}")
    finally:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument("--out", type=str, default="docs/perf/phase3-wi-b1-shape-sweep.md")
    args = parser.parse_args()

    # (batch, block) grid -- constant-token-budget then extended
    shapes = [
        # Constant ~4096 tokens/step (baseline equivalent)
        (32, 128),
        (24, 192),
        (16, 256),   # baseline
        (12, 384),
        (8, 512),
        # Scaled budget (8192 tokens/step) -- higher memory
        (64, 128),
        (32, 256),
        (16, 512),
        # Sub-baseline budget (2048 tokens/step) -- diagnostic
        (8, 256),
        (16, 128),
    ]

    all_results = []
    for batch, block in shapes:
        # Memory estimate: roughly 5.3 GB * (tokens / 4096) * (d_model^2 scaling) 
        # Actually just try it and catch OOM
        r = measure_shape(batch, block, args.warmup, args.measured)
        all_results.append(r)

    # Write markdown
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()

    # Reference baseline for %-delta comparison
    baseline = next((r for r in all_results
                    if r["batch_size"] == 16 and r["block_size"] == 256
                    and r["status"] == "OK"), None)
    baseline_tok_s = baseline["tok_s"] if baseline else 14472.0

    lines = []
    lines.append(f"# WI-B1: Shape sweep for OdinHalo — {date_str}\n")
    lines.append(f"**Baseline:** batch=16, block=256 @ {baseline_tok_s:,.0f} tok/s\n")
    lines.append(f"**Config:** compile_zones default, fp16 autocast, fused AdamW, "
                 f"warmup={args.warmup}, measured={args.measured}\n")

    # Group by budget
    budgets = {}
    for r in all_results:
        b = r["tokens_per_step"]
        budgets.setdefault(b, []).append(r)

    for budget in sorted(budgets.keys()):
        lines.append(f"\n## Token budget: {budget} tokens/step\n")
        lines.append("| Batch | Block | Status | Warmup (s) | tok/s | Peak GB | vs 16×256 |")
        lines.append("|------:|------:|--------|-----------:|------:|--------:|----------:|")
        for r in sorted(budgets[budget], key=lambda x: -x["tok_s"]):
            pct = 100.0 * (r["tok_s"] - baseline_tok_s) / baseline_tok_s
            tok_s_str = f"{r['tok_s']:,.0f}" if r["tok_s"] > 0 else "-"
            gb_str = f"{r['peak_gb']:.2f}" if r["peak_gb"] > 0 else "-"
            status = r["status"]
            lines.append(f"| {r['batch_size']} | {r['block_size']} | {status} | "
                         f"{r['warmup_s']:.1f} | {tok_s_str} | {gb_str} | "
                         f"{pct:+.2f}% |")

    # Conclusions
    ok_results = [r for r in all_results if r["status"] == "OK"]
    ok_results.sort(key=lambda r: -r["tok_s"])

    lines.append("\n## Top 5 throughput shapes\n")
    lines.append("| Rank | Batch | Block | Tokens/step | tok/s | Peak GB | vs baseline |")
    lines.append("|-----:|------:|------:|------------:|------:|--------:|------------:|")
    for i, r in enumerate(ok_results[:5], 1):
        pct = 100.0 * (r["tok_s"] - baseline_tok_s) / baseline_tok_s
        lines.append(f"| {i} | {r['batch_size']} | {r['block_size']} | "
                     f"{r['tokens_per_step']} | {r['tok_s']:,.0f} | "
                     f"{r['peak_gb']:.2f} | {pct:+.2f}% |")

    if ok_results:
        best = ok_results[0]
        best_pct = 100.0 * (best["tok_s"] - baseline_tok_s) / baseline_tok_s
        lines.append(f"\n## Verdict\n")
        lines.append(f"Best shape: **batch={best['batch_size']}, block={best['block_size']}** "
                     f"@ **{best['tok_s']:,.0f} tok/s** ({best_pct:+.2f}% vs baseline), "
                     f"peak {best['peak_gb']:.2f} GB.")
        if best_pct >= 5.0:
            lines.append(f"\n**PASSES STRETCH GATE** (≥+5%).")
        elif best_pct >= 0:
            lines.append(f"\nShipping-gate eligible ({best_pct:+.2f}% ≥ -2%) but does not clear stretch (+5%).")
        else:
            lines.append(f"\nDoes not improve over baseline; shape sweep is a wash.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes)")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(all_results, indent=2, default=str),
                         encoding="utf-8")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
