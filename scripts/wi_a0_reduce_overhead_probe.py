"""WI-A0: Investigation of reduce-overhead compile mode on looped OdinHalo.

Four experiments to resolve the STATUS row 113 ambiguity and classify the
actual failure mode of reduce-overhead on Parcae-looped models.

Experiments:
  E1: compile_zones() with mode="reduce-overhead" (no clones). Measure what
      happens: OOM, NaN, wrong-answer, or success.
  E2: Same as E1, but with torch.compiler.cudagraph_mark_step_begin() called
      before each step (trainer already does this).
  E3: Same as E1, with naive .clone() injected in _forward_unrolled at 
      layer boundaries (Option X trial balloon - 4 clone sites).
  E4: E1 + chunked_ce enabled. Tests the STATUS row 114/116 claim.

For each experiment, measure:
  - Warmup time (first compile + first valid step)
  - 100-step steady-state tok/s
  - Peak memory
  - Did loss stay finite for 100 steps?
  - Final loss (for trajectory comparison)

Outputs markdown to docs/perf/phase3-wi-a0-findings.md.

Usage:
  python scripts/wi_a0_reduce_overhead_probe.py
"""
import os, sys, time, argparse, traceback, json, datetime, pathlib
sys.path.insert(0, '.')


# Set reduce-overhead mode at import time so torch.compile picks it up
def run_experiment(label, compile_mode, batch_size, block_size,
                   warmup_steps, measured_steps, enable_chunked_ce=False,
                   insert_clones=False):
    """Run one experiment configuration, return dict of results."""
    import torch
    torch.manual_seed(42)

    result = {
        "label": label,
        "compile_mode": compile_mode,
        "batch_size": batch_size,
        "block_size": block_size,
        "chunked_ce": enable_chunked_ce,
        "insert_clones": insert_clones,
        "status": "UNKNOWN",
        "warmup_s": 0.0,
        "steady_tok_s": 0.0,
        "peak_gb": 0.0,
        "loss_trajectory": [],
        "error": None,
    }

    try:
        from models.odin_halo import OdinHalo
        from halo_training.data import BabyLMDataset, build_dataloader

        if enable_chunked_ce:
            model = OdinHalo(use_chunked_ce=True).to('cuda')
        else:
            model = OdinHalo().to('cuda')
        model.train()

        if insert_clones:
            # Monkey-patch _forward_unrolled to insert clones at layer boundaries.
            # This is a trial balloon for Option X.
            orig_run_block = model._run_shared_block
            def cloned_run_block(h, freqs_cis, depth_kv_buffer):
                h, kvs = orig_run_block(h, freqs_cis, depth_kv_buffer)
                return h.clone(), kvs
            model._run_shared_block = cloned_run_block

        # Apply compilation - directly, bypassing trainer's auto-fallback
        if compile_mode == "compile_zones_default":
            # baseline: standard compile_zones (mode=default, which is what
            # compile_zones() uses internally)
            model.compile_zones()
        elif compile_mode == "compile_zones_reduce_overhead":
            # Override: apply reduce-overhead to each layer
            for i in range(len(model.shared_layers)):
                model.shared_layers[i] = torch.compile(
                    model.shared_layers[i], mode="reduce-overhead")
        elif compile_mode == "whole_model_reduce_overhead":
            # Compile the whole model with reduce-overhead
            model = torch.compile(model, mode="reduce-overhead")
        elif compile_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown compile_mode: {compile_mode}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1024.0)

        # Chunked CE setup
        chunked_ce_fn = None
        if enable_chunked_ce:
            from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss
            chunked_ce_fn = ChunkedLinearCrossEntropyLoss(chunk_size=512)

        ds = BabyLMDataset(
            root='datasets/babylm-odin32k.bin',
            block_size=block_size,
            tokenizer_path='tokenizers/odin-32k/tokenizer.json',
        )
        dl = build_dataloader(ds, batch_size=batch_size, num_workers=0, shuffle=True)
        it = iter(dl)

        def get_batch():
            nonlocal it
            try: return next(it)
            except StopIteration:
                it = iter(dl)
                return next(it)

        def step():
            input_ids, targets = get_batch()
            input_ids = input_ids.to('cuda', non_blocking=True)
            targets = targets.to('cuda', non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except (AttributeError, RuntimeError):
                pass
            with torch.amp.autocast('cuda', dtype=torch.float16):
                if enable_chunked_ce:
                    # Model returns h_low (rank hidden state), we compute loss
                    # via chunked CE against tied embed weight.
                    h_low = model(input_ids)
                    logits_weight = model.lm_head.embed_table.weight
                    loss = chunked_ce_fn(h_low, logits_weight, targets.view(-1))
                else:
                    logits = model(input_ids)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            return loss.item()

        # Warmup - count first 2 steps as warmup-for-compile, rest as normal
        print(f"  [{label}] Starting warmup ({warmup_steps} steps)...")
        warmup_start = time.perf_counter()
        last_loss = None
        for w in range(warmup_steps):
            try:
                last_loss = step()
                if w == 0 or w == warmup_steps - 1:
                    print(f"  [{label}] warmup step {w+1}/{warmup_steps} loss={last_loss:.4f}")
            except Exception as e:
                result["status"] = f"CRASHED_WARMUP_STEP_{w}"
                result["error"] = f"{type(e).__name__}: {e}"
                result["warmup_s"] = time.perf_counter() - warmup_start
                return result
            if not (last_loss == last_loss):  # NaN check
                result["status"] = f"NAN_WARMUP_STEP_{w}"
                result["warmup_s"] = time.perf_counter() - warmup_start
                return result
        torch.cuda.synchronize()
        result["warmup_s"] = time.perf_counter() - warmup_start

        # Measured
        print(f"  [{label}] Warmup done in {result['warmup_s']:.1f}s. "
              f"Measuring {measured_steps} steps...")
        losses = []
        t0 = time.perf_counter()
        n_tok = 0
        for m in range(measured_steps):
            try:
                loss = step()
                losses.append(loss)
                if not (loss == loss):
                    result["status"] = f"NAN_MEASURED_STEP_{m}"
                    result["loss_trajectory"] = losses
                    return result
                n_tok += batch_size * block_size
            except Exception as e:
                result["status"] = f"CRASHED_MEASURED_STEP_{m}"
                result["error"] = f"{type(e).__name__}: {e}"
                return result
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        result["steady_tok_s"] = n_tok / (t1 - t0)
        result["peak_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        result["loss_trajectory"] = losses
        result["final_loss"] = losses[-1]
        result["status"] = "OK"

        print(f"  [{label}] DONE: {result['steady_tok_s']:,.0f} tok/s, "
              f"{result['peak_gb']:.2f} GB, final_loss={losses[-1]:.4f}")

    except Exception as e:
        result["status"] = "CRASHED_SETUP"
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"  [{label}] SETUP ERROR: {e}")

    finally:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--measured", type=int, default=50)
    parser.add_argument("--out", type=str, default="docs/perf/phase3-wi-a0-findings.md")
    parser.add_argument("--experiments", type=str, default="all",
                        help="comma-separated: e0,e1,e1c,e2,e3,e4  or 'all'")
    args = parser.parse_args()

    experiments = args.experiments.split(",") if args.experiments != "all" else \
        ["e0", "e1", "e1c", "e2", "e3", "e4"]

    all_results = []

    if "e0" in experiments:
        # E0: baseline (compile_zones default) - this is our known 14,682 tok/s point
        all_results.append(run_experiment(
            "E0: baseline compile_zones default",
            compile_mode="compile_zones_default",
            batch_size=args.batch_size, block_size=args.block_size,
            warmup_steps=args.warmup, measured_steps=args.measured,
        ))

    if "e1" in experiments:
        # E1: compile_zones with mode="reduce-overhead" - the core question
        all_results.append(run_experiment(
            "E1: compile_zones reduce-overhead, no clones",
            compile_mode="compile_zones_reduce_overhead",
            batch_size=args.batch_size, block_size=args.block_size,
            warmup_steps=args.warmup, measured_steps=args.measured,
        ))

    if "e1c" in experiments:
        # E1c: compile_zones reduce-overhead + naive clone at layer boundary
        all_results.append(run_experiment(
            "E1c: compile_zones reduce-overhead + layer-boundary clone",
            compile_mode="compile_zones_reduce_overhead",
            batch_size=args.batch_size, block_size=args.block_size,
            warmup_steps=args.warmup, measured_steps=args.measured,
            insert_clones=True,
        ))

    if "e2" in experiments:
        # E2: whole-model reduce-overhead (not per-layer)
        all_results.append(run_experiment(
            "E2: whole-model reduce-overhead",
            compile_mode="whole_model_reduce_overhead",
            batch_size=args.batch_size, block_size=args.block_size,
            warmup_steps=args.warmup, measured_steps=args.measured,
        ))

    if "e3" in experiments:
        # E3: no compile - sanity baseline
        all_results.append(run_experiment(
            "E3: no compile (eager)",
            compile_mode="none",
            batch_size=args.batch_size, block_size=args.block_size,
            warmup_steps=args.warmup, measured_steps=args.measured,
        ))

    if "e4" in experiments:
        # E4: compile_zones + reduce-overhead + chunked_ce
        all_results.append(run_experiment(
            "E4: compile_zones reduce-overhead + chunked_ce",
            compile_mode="compile_zones_reduce_overhead",
            batch_size=args.batch_size, block_size=args.block_size,
            warmup_steps=args.warmup, measured_steps=args.measured,
            enable_chunked_ce=True,
        ))

    # Write markdown report
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    date_str = datetime.date.today().isoformat()
    lines.append(f"# Phase 3 WI-A0: reduce-overhead probe findings — {date_str}\n")
    lines.append(f"**Config:** OdinHalo batch={args.batch_size} block={args.block_size} "
                 f"warmup={args.warmup} measured={args.measured}\n")
    lines.append("## Summary table\n")
    lines.append("| Experiment | Status | Warmup (s) | tok/s | Peak GB | Final loss |")
    lines.append("|-----------|--------|-----------:|------:|--------:|-----------:|")
    for r in all_results:
        status = r["status"]
        warm = r["warmup_s"]
        toks = r["steady_tok_s"]
        gb = r["peak_gb"]
        fl = r.get("final_loss", "n/a")
        fl_str = f"{fl:.4f}" if isinstance(fl, (int, float)) else str(fl)
        toks_str = f"{toks:,.0f}" if toks > 0 else "-"
        gb_str = f"{gb:.2f}" if gb > 0 else "-"
        lines.append(f"| {r['label']} | {status} | {warm:.1f} | {toks_str} | {gb_str} | {fl_str} |")

    lines.append("\n## Per-experiment detail\n")
    for r in all_results:
        lines.append(f"### {r['label']}\n")
        lines.append(f"- **Status:** `{r['status']}`")
        lines.append(f"- **compile_mode:** `{r['compile_mode']}`")
        lines.append(f"- **insert_clones:** `{r['insert_clones']}`")
        lines.append(f"- **chunked_ce:** `{r['chunked_ce']}`")
        lines.append(f"- **Warmup:** {r['warmup_s']:.2f} s")
        lines.append(f"- **tok/s (steady):** {r['steady_tok_s']:,.1f}")
        lines.append(f"- **Peak memory:** {r['peak_gb']:.3f} GB")
        if r.get("loss_trajectory"):
            traj = r["loss_trajectory"]
            lines.append(f"- **Loss trajectory:** first={traj[0]:.4f}, "
                         f"last={traj[-1]:.4f}, min={min(traj):.4f}, max={max(traj):.4f}, "
                         f"len={len(traj)}")
        if r.get("error"):
            lines.append(f"\n**Error:**\n```\n{r['error'][:2000]}\n```\n")
        lines.append("")

    # Conclusions template
    lines.append("\n## Conclusions\n")
    lines.append("(Filled in after reviewing experiments)\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes)")

    # Also dump JSON for programmatic access
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(all_results, indent=2, default=str),
                         encoding="utf-8")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
