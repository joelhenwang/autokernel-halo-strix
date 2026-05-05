"""WI2: Shape-annotated profile to classify aten::add_ and aten::copy_ call sites.

Runs OdinHalo under torch.profiler with record_shapes=True and with_stack=True,
then groups every aten::add_ and aten::copy_ invocation by:
  - input shape tuple
  - user-code stack frame (topmost non-torch frame)

Produces a table showing WHERE each call originates and what tensor sizes it touches.
Used to decide whether any of the 4.9% (add_) or 4.4% (copy_) wall-time cost is
attackable by code changes vs unavoidable autograd/autocast bookkeeping.

Usage:
  python scripts/profile_shape_calls.py --out docs/perf/wi2-shape-calls.md
"""
import sys, os, argparse, datetime, pathlib, collections, traceback
sys.path.insert(0, '.')

import torch

WARMUP = 10
MEASURED = 3


def pick_user_frame(stack_list):
    """Pick the topmost frame that is NOT inside torch/python internals."""
    for frame in stack_list:
        if not frame:
            continue
        f = str(frame)
        if any(bad in f for bad in [
            "torch/", "torch\\", "site-packages/torch",
            "/torch/autograd/", "/torch/amp/", "/torch/_inductor/",
            "/torch/_dynamo/", "/torch/nn/modules/module.py",
            "<built-in", "<frozen",
        ]):
            continue
        return f
    return "<unknown>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="docs/perf/wi2-shape-calls.md")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    args = parser.parse_args()

    from models.odin_halo import OdinHalo
    from halo_training.data import BabyLMDataset, build_dataloader

    device = 'cuda'
    torch.manual_seed(42)

    model = OdinHalo().to(device)
    model.train()
    if args.compile:
        print("Applying compile_zones()...")
        model.compile_zones()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)

    ds = BabyLMDataset(
        root='datasets/babylm-odin32k.bin',
        block_size=args.block_size,
        tokenizer_path='tokenizers/odin-32k/tokenizer.json',
    )
    dl = build_dataloader(ds, batch_size=args.batch_size, num_workers=0, shuffle=True)
    it = iter(dl)

    def step():
        input_ids, targets = next(it)
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

    print(f"Warmup {WARMUP} steps...")
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity, schedule
    print(f"Profiling with record_shapes=True, with_stack=True...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=1, active=MEASURED, repeat=1),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(1 + MEASURED):
            step()
            prof.step()
        torch.cuda.synchronize()

    # Walk through per-event list (NOT key_averages, which merges) to get per-invocation
    # shapes and stacks.
    events = prof.events()

    # Build buckets for each target op
    target_ops = {"aten::add_", "aten::copy_", "aten::add", "aten::mul",
                  "aten::_foreach_add", "aten::_fused_adam", "aten::zero_"}
    # group: op_name -> { (shape_key, frame) -> [(device_time_us, cpu_time_us), ...] }
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    total_by_op = collections.Counter()
    totals = collections.Counter()

    for e in events:
        op = e.name
        if op not in target_ops:
            continue
        try:
            cuda_us = int(e.self_device_time_total if hasattr(e, "self_device_time_total")
                          else e.self_cuda_time_total) / 1000.0
        except Exception:
            cuda_us = 0.0
        cpu_us = e.self_cpu_time_total / 1000.0
        # input shapes
        shapes = getattr(e, "input_shapes", None) or []
        shape_key = tuple(tuple(s) if isinstance(s, (list, tuple)) else s for s in shapes)
        # stack
        stack = getattr(e, "stack", []) or []
        frame = pick_user_frame(stack)
        buckets[op][(shape_key, frame)].append((cuda_us, cpu_us))
        total_by_op[op] += cuda_us
        totals["all"] += cuda_us

    # Write markdown
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    date_str = datetime.date.today().isoformat()
    lines.append(f"# WI2: aten::add_ / aten::copy_ shape + call-site classification — {date_str}\n")
    lines.append(f"**Config:** OdinHalo batch={args.batch_size} block={args.block_size} "
                 f"compile_zones={'yes' if args.compile else 'no'} | {MEASURED} active profiler steps\n")
    lines.append(f"**Total CUDA time across measured ops:** {total_by_op.total():.1f} μs (across {MEASURED} steps)\n")

    for op, op_buckets in sorted(buckets.items(), key=lambda kv: -total_by_op[kv[0]]):
        op_total = total_by_op[op]
        n_invocations = sum(len(v) for v in op_buckets.values())
        lines.append(f"\n## `{op}` — {op_total:.1f} μs total, {n_invocations} invocations, {len(op_buckets)} unique (shape,frame) groups\n")
        lines.append("| # | shape | frame | # calls | total μs | μs/call |")
        lines.append("|--:|-------|-------|--------:|---------:|--------:|")
        rows = []
        for (shape_key, frame), entries in op_buckets.items():
            n = len(entries)
            t = sum(e[0] for e in entries)
            rows.append((t, n, shape_key, frame))
        rows.sort(reverse=True)
        for i, (t, n, shape_key, frame) in enumerate(rows[:25], 1):
            shape_str = str(shape_key)[:80]
            frame_str = str(frame)[:100].replace("|", "\\|")
            per = t / max(1, n)
            pct = 100.0 * t / op_total if op_total > 0 else 0
            lines.append(f"| {i} | `{shape_str}` | `{frame_str}` | {n} | {t:.1f} ({pct:.1f}%) | {per:.2f} |")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    print("Op totals (μs):")
    for op, t in total_by_op.most_common():
        print(f"  {op:30s} {t:>10.1f}")


if __name__ == "__main__":
    main()
