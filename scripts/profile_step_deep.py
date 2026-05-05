"""Deep profile of one OdinHalo training step, categorized by op type.

Output: docs/perf/odinhalo-profile-<date>/profiler.md with two sections:
  1. Raw torch.profiler key_averages table
  2. Categorized breakdown (matmul / norm / elementwise / copy / optimizer / attention / other)

Usage:
  python scripts/profile_step_deep.py
  python scripts/profile_step_deep.py --rocprof-subrun   # no torch.profiler, for rocprof wrapping
"""
import sys, os, argparse, datetime, pathlib, torch
sys.path.insert(0, '.')

from models.odin_halo import OdinHalo
from halo_training.data import BabyLMDataset, build_dataloader

WARMUP = 25
MEASURED = 5


def categorize_op(name: str) -> str:
    """Bucket an op name into a category via substring heuristics."""
    n = name.lower()
    # Order matters: check more specific patterns first
    if any(k in n for k in ["flash", "sdpa", "scaled_dot_product_attention"]):
        return "attention"
    if any(k in n for k in ["softmax", "log_softmax"]):
        return "attention"
    if any(k in n for k in ["mm", "bmm", "linear", "matmul", "addmm", "cijk_", "tensile",
                             "gemm", "gemv", "hgemm"]):
        return "matmul"
    if any(k in n for k in ["rms_norm", "layer_norm", "rmsnorm", "layernorm", "native_norm"]):
        return "norm"
    if any(k in n for k in ["adamw", "_foreach_", "optim"]):
        return "optimizer"
    if "copy" in n or "to_copy" in n or "memcpy" in n:
        return "copy"
    if any(k in n for k in ["mul", "add", "sub", "div", "neg", "sign",
                             "sigmoid", "silu", "gelu", "tanh", "exp", "log",
                             "sqrt", "rsqrt", "pow", "reciprocal", "clamp",
                             "embedding", "gather", "index"]):
        return "elementwise"
    if any(k in n for k in ["cross_entropy", "nll_loss", "loss"]):
        return "loss"
    if any(k in n for k in ["conv", "causal_conv"]):
        return "conv"
    return "other"


def run_training_steps(model, optimizer, scaler, dl, batches, n_steps, profile_ctx=None):
    """Run n_steps of training, optionally under a profiler context."""
    device = 'cuda'
    for step in range(n_steps):
        input_ids, targets = next(batches)
        input_ids = input_ids.to(device)
        targets = targets.to(device)
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
        if profile_ctx is not None:
            profile_ctx.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocprof-subrun", action="store_true",
                        help="Skip torch.profiler (used when this script is wrapped by rocprof)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--compile", action="store_true",
                        help="Apply compile_zones() to match production config")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: docs/perf/odinhalo-profile-YYYY-MM-DD)")
    args = parser.parse_args()

    device = 'cuda'
    torch.manual_seed(42)

    # Build model (production config)
    model = OdinHalo().to(device)
    model.train()

    if args.compile:
        print("Applying compile_zones() (production config)...")
        model.compile_zones()

    # Use fused AdamW (matches production config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)

    ds = BabyLMDataset(
        root='datasets/babylm-odin32k.bin',
        block_size=args.block_size,
        tokenizer_path='tokenizers/odin-32k/tokenizer.json',
    )
    dl = build_dataloader(ds, batch_size=args.batch_size, num_workers=0, shuffle=True)

    def batches():
        while True:
            for b in dl:
                yield b

    it = batches()

    # Warmup
    print(f"Warmup: {WARMUP} steps at batch={args.batch_size}, block={args.block_size}...")
    run_training_steps(model, optimizer, scaler, dl, it, WARMUP)
    torch.cuda.synchronize()

    if args.rocprof_subrun:
        # Just run measured steps without torch.profiler
        print(f"Measured (rocprof subrun): {MEASURED} steps...")
        run_training_steps(model, optimizer, scaler, dl, it, MEASURED)
        torch.cuda.synchronize()
        print("Done (rocprof subrun).")
        return

    # Profile measured steps
    from torch.profiler import profile, ProfilerActivity, schedule
    print(f"Profiling: schedule wait=1 warmup=1 active=3 repeat=1 (5 total steps)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    ) as prof:
        run_training_steps(model, optimizer, scaler, dl, it, MEASURED, profile_ctx=prof)
        torch.cuda.synchronize()

    # Prepare output directory
    date_str = datetime.date.today().isoformat()
    if args.output_dir:
        outdir = pathlib.Path(args.output_dir)
    else:
        suffix = "-compile" if args.compile else "-eager"
        outdir = pathlib.Path(f"docs/perf/odinhalo-profile-{date_str}{suffix}")
    outdir.mkdir(parents=True, exist_ok=True)
    out_md = outdir / "profiler.md"

    # Gather per-op stats
    events = prof.key_averages()
    total_cuda_ns = 0
    op_rows = []
    for e in events:
        cuda_ns = int(e.self_device_time_total) if hasattr(e, 'self_device_time_total') else int(e.self_cuda_time_total)
        if cuda_ns <= 0:
            continue
        total_cuda_ns += cuda_ns
        op_rows.append({
            "name": e.key,
            "self_cuda_us": cuda_ns / 1000.0,
            "self_cpu_us": e.self_cpu_time_total / 1000.0,
            "count": e.count,
            "category": categorize_op(e.key),
        })
    op_rows.sort(key=lambda r: r["self_cuda_us"], reverse=True)

    # Category totals
    cat_totals = {}
    for row in op_rows:
        cat = row["category"]
        cat_totals[cat] = cat_totals.get(cat, 0) + row["self_cuda_us"]
    total_us = sum(cat_totals.values())

    # Write markdown
    lines = []
    mode = "compile_zones" if args.compile else "eager"
    lines.append(f"# OdinHalo deep profile — {date_str} ({mode})\n")
    lines.append(f"**Config:** OdinHalo (57.6M params) | batch={args.batch_size} "
                 f"| block={args.block_size} | fused AdamW | fp16 autocast | mode={mode} "
                 f"| warmup={WARMUP} | measured={MEASURED}\n")
    lines.append(f"**Total GPU time measured:** {total_us:.1f} μs across {sum(r['count'] for r in op_rows)} op calls "
                 f"over 3 active profiler steps.\n")

    lines.append("\n## Categorized breakdown\n")
    lines.append("| Category | Self CUDA μs | % of wall | Top ops |")
    lines.append("|----------|-------------:|----------:|---------|")
    for cat in sorted(cat_totals.keys(), key=lambda c: -cat_totals[c]):
        pct = 100.0 * cat_totals[cat] / total_us if total_us > 0 else 0
        top_names = [r["name"] for r in op_rows if r["category"] == cat][:3]
        top_str = ", ".join(f"`{n}`" for n in top_names)
        lines.append(f"| {cat} | {cat_totals[cat]:,.0f} | {pct:.1f}% | {top_str} |")

    lines.append("\n## Top 40 ops by self CUDA time\n")
    lines.append("| # | Name | Category | Self CUDA μs | % | Calls | μs/call |")
    lines.append("|--:|------|----------|-------------:|--:|------:|--------:|")
    for i, r in enumerate(op_rows[:40], 1):
        pct = 100.0 * r["self_cuda_us"] / total_us if total_us > 0 else 0
        per_call = r["self_cuda_us"] / max(1, r["count"])
        name = r["name"][:80]
        lines.append(f"| {i} | `{name}` | {r['category']} | {r['self_cuda_us']:,.1f} | "
                     f"{pct:.2f}% | {r['count']} | {per_call:.2f} |")

    lines.append("\n## Sanity check\n")
    lines.append(f"Category totals sum: {total_us:,.0f} μs")
    lines.append(f"Raw event total: {total_cuda_ns / 1000.0:,.0f} μs")
    diff_pct = abs(total_us - total_cuda_ns / 1000.0) / max(1, total_cuda_ns / 1000.0) * 100
    lines.append(f"Delta: {diff_pct:.2f}% (must be < 5%)\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Top 5 categories:")
    for cat in sorted(cat_totals.keys(), key=lambda c: -cat_totals[c])[:5]:
        pct = 100.0 * cat_totals[cat] / total_us if total_us > 0 else 0
        print(f"  {cat:15s} {cat_totals[cat]:>12,.0f} μs  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
