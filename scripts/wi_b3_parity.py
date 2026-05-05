"""WI-B3 follow-up: loss-parity + longer-horizon throughput for max-autotune.

The initial WI-B3 run showed a +4.54% tok/s win but also that end-of-warmup
loss was 9.2 (baseline) vs 8.05 (max-autotune) — 1+ loss-unit divergence that
could indicate a numerical correctness issue.

This script runs a longer parity comparison:
- 200 training steps, logging loss every 10 steps
- Same seed (42), same data, same warmup
- Compares compile_zones (default) vs per-layer max-autotune
- Computes per-step loss delta; if max |delta| > 0.5 at any step, flags divergence
- Also measures steady-state tok/s at steps 100-199 (after autotune is done tuning)

Usage:
  python scripts/wi_b3_parity.py --steps 200
"""
import argparse, time, statistics, json, pathlib, datetime, sys
sys.path.insert(0, '.')
import torch


def run_training(mode, n_steps, batch=16, block=256):
    """Run deterministic training and return per-step loss list."""
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

    if mode == "default":
        model.compile_zones()
    elif mode == "max-autotune":
        for i in range(len(model.shared_layers)):
            model.shared_layers[i] = torch.compile(
                model.shared_layers[i], mode="max-autotune")
    else:
        raise ValueError(f"unknown mode: {mode}")

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
        ids, tgt = gb_next()
        ids = ids.to('cuda', non_blocking=True)
        tgt = tgt.to('cuda', non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), tgt.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    losses = []
    step_times = []
    print(f"  [{mode}] running {n_steps} steps...")
    t0 = time.perf_counter()
    t_last = t0
    for s in range(n_steps):
        loss = step()
        losses.append(loss)
        t_now = time.perf_counter()
        step_times.append(t_now - t_last)
        t_last = t_now
        if s < 5 or (s + 1) % 25 == 0:
            print(f"  [{mode}] step {s+1}/{n_steps} loss={loss:.4f} "
                  f"(step_time={step_times[-1]*1000:.1f}ms)")
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "mode": mode,
        "losses": losses,
        "step_times": step_times,
        "total_time": t1 - t0,
        "peak_gb": peak_gb,
        "tok_s_total": n_steps * batch * block / (t1 - t0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", type=str,
                        default="docs/perf/phase3-wi-b3-parity.md")
    args = parser.parse_args()

    print(f"Running baseline (compile_zones default) for {args.steps} steps...")
    baseline = run_training("default", args.steps)
    print(f"Running max-autotune (per-layer) for {args.steps} steps...")
    autotune = run_training("max-autotune", args.steps)

    # Compute loss deltas
    deltas = [a - b for a, b in zip(autotune["losses"], baseline["losses"])]
    max_abs_delta = max(abs(d) for d in deltas)
    mean_abs_delta = sum(abs(d) for d in deltas) / len(deltas)
    max_delta_idx = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
    
    # Steady-state tok/s: skip first 50 steps (autotune warm-up), measure 50-199
    skip = 50
    base_time = sum(baseline["step_times"][skip:])
    auto_time = sum(autotune["step_times"][skip:])
    steady_tokens = (args.steps - skip) * 16 * 256
    base_steady_toks = steady_tokens / base_time if base_time > 0 else 0
    auto_steady_toks = steady_tokens / auto_time if auto_time > 0 else 0

    # Write report
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    lines = [
        f"# WI-B3 parity: max-autotune loss divergence investigation — {date_str}\n",
        f"**Config:** OdinHalo batch=16 block=256, {args.steps} training steps, same seed\n",
        "## Summary\n",
        f"- **Baseline mode:** compile_zones (default)",
        f"- **Candidate:** per-layer max-autotune",
        f"- **Max |loss delta|:** {max_abs_delta:.4f} at step {max_delta_idx+1}",
        f"- **Mean |loss delta|:** {mean_abs_delta:.4f}",
        f"- **Final loss baseline:** {baseline['losses'][-1]:.4f}",
        f"- **Final loss autotune:** {autotune['losses'][-1]:.4f}",
        f"- **Final |delta|:** {abs(autotune['losses'][-1] - baseline['losses'][-1]):.4f}",
        f"\n## Throughput (steady-state, skip first 50 steps)\n",
        f"| Metric | Baseline | max-autotune | Δ |",
        f"|--------|---------:|-------------:|--:|",
        f"| Steady tok/s | {base_steady_toks:,.0f} | {auto_steady_toks:,.0f} | {100.0 * (auto_steady_toks - base_steady_toks) / base_steady_toks:+.2f}% |",
        f"| Peak GB | {baseline['peak_gb']:.2f} | {autotune['peak_gb']:.2f} | - |",
        "",
        "## Loss trajectory (every 10 steps)\n",
        "| Step | Baseline | max-autotune | Δ |",
        "|-----:|---------:|-------------:|--:|",
    ]
    for i in range(0, args.steps, 10):
        d = autotune["losses"][i] - baseline["losses"][i]
        lines.append(f"| {i+1} | {baseline['losses'][i]:.4f} | "
                     f"{autotune['losses'][i]:.4f} | {d:+.4f} |")
    lines.append(f"| {args.steps} | {baseline['losses'][-1]:.4f} | "
                 f"{autotune['losses'][-1]:.4f} | "
                 f"{autotune['losses'][-1] - baseline['losses'][-1]:+.4f} |")

    # Verdict
    lines.append("\n## Verdict\n")
    if max_abs_delta < 0.1:
        lines.append(f"**PARITY PASS**: max |delta| = {max_abs_delta:.4f} < 0.1. "
                     "Numerical divergence is within fp16 rounding noise.")
    elif max_abs_delta < 0.5:
        lines.append(f"**MARGINAL**: max |delta| = {max_abs_delta:.4f}. "
                     "Likely fp16-accumulation-order variance but worth a longer "
                     "training run to confirm convergence.")
    else:
        lines.append(f"**PARITY FAIL**: max |delta| = {max_abs_delta:.4f} ≥ 0.5. "
                     "Indicates real numerical difference. Should NOT ship without "
                     "understanding the source.")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"\nMax |loss delta|: {max_abs_delta:.4f} at step {max_delta_idx+1}")
    print(f"Mean |loss delta|: {mean_abs_delta:.4f}")
    print(f"Steady-state tok/s: baseline={base_steady_toks:,.0f}, "
          f"autotune={auto_steady_toks:,.0f} ({100.0 * (auto_steady_toks - base_steady_toks) / base_steady_toks:+.2f}%)")

    # JSON dump
    out.with_suffix(".json").write_text(json.dumps({
        "baseline": baseline,
        "autotune": autotune,
        "deltas": deltas,
        "max_abs_delta": max_abs_delta,
        "mean_abs_delta": mean_abs_delta,
        "base_steady_toks": base_steady_toks,
        "auto_steady_toks": auto_steady_toks,
    }, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
