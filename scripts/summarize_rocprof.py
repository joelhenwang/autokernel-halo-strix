"""Summarize rocprof stats CSV into a markdown report.

rocprof --stats produces a CSV with columns: Name, Calls, TotalDurationNs, AverageNs, Percentage.

This script:
  1. Sorts by total duration descending
  2. Groups kernels by prefix heuristic (rocBLAS vs our HIP kernels vs PyTorch dispatched)
  3. Emits top-20 and grouped-total tables
"""
import argparse, csv, pathlib
from collections import defaultdict


def classify_kernel(name: str) -> str:
    """Group kernel by naming prefix."""
    # Common rocBLAS / Tensile kernels
    if name.startswith("Cijk_") or "Tensile" in name:
        return "rocBLAS matmul"
    # Our HIP kernels
    if "cross_entropy" in name:
        return "HIP CE (ours)"
    if "fused_rope_gate_mul" in name:
        return "HIP RoPE+gate (ours)"
    if "fused_" in name:
        return "HIP fused (ours/aiter)"
    # PyTorch generic
    if "void at::native::" in name:
        if "elementwise" in name:
            return "PyTorch elementwise"
        if "reduce" in name.lower():
            return "PyTorch reduce"
        if "norm" in name.lower():
            return "PyTorch norm"
        if "softmax" in name.lower():
            return "PyTorch softmax"
        return "PyTorch other"
    if "multi_tensor" in name:
        return "PyTorch optimizer (foreach)"
    if "copy" in name.lower() or "memcpy" in name.lower():
        return "copy/memcpy"
    if "causal_conv" in name:
        return "HIP causal_conv1d"
    if "flash" in name.lower() or "sdpa" in name.lower():
        return "HIP flash/SDPA"
    # Triton-generated (Inductor)
    if "triton_" in name or "_triton_" in name:
        return "Inductor triton"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="rocprof stats CSV")
    parser.add_argument("--output", required=True, help="output Markdown file")
    args = parser.parse_args()

    rows = []
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                name = row.get("Name", "") or row.get("KernelName", "")
                calls = int(row.get("Calls", 0))
                total_ns = int(row.get("TotalDurationNs", 0))
                rows.append({
                    "name": name,
                    "calls": calls,
                    "total_ns": total_ns,
                    "avg_ns": total_ns // max(1, calls),
                    "category": classify_kernel(name),
                })
            except (ValueError, KeyError) as e:
                continue

    if not rows:
        print(f"No rows parsed from {args.input}")
        return

    rows.sort(key=lambda r: -r["total_ns"])
    total_ns = sum(r["total_ns"] for r in rows)
    total_us = total_ns / 1000.0
    total_ms = total_ns / 1_000_000.0

    # Group totals
    cat_totals = defaultdict(lambda: {"total_ns": 0, "calls": 0, "count": 0})
    for r in rows:
        cat_totals[r["category"]]["total_ns"] += r["total_ns"]
        cat_totals[r["category"]]["calls"] += r["calls"]
        cat_totals[r["category"]]["count"] += 1

    # Output markdown
    out = []
    out.append(f"# OdinHalo rocprof kernel-level summary\n")
    out.append(f"Total GPU time: {total_ms:,.2f} ms across {len(rows)} unique kernels, "
               f"{sum(r['calls'] for r in rows):,} total launches.\n")

    out.append("## Kernels grouped by source\n")
    out.append("| Category | Total μs | % | Unique kernels | Total launches |")
    out.append("|----------|---------:|---:|---------------:|---------------:|")
    for cat, d in sorted(cat_totals.items(), key=lambda kv: -kv[1]["total_ns"]):
        pct = 100.0 * d["total_ns"] / total_ns if total_ns > 0 else 0
        out.append(f"| {cat} | {d['total_ns']/1000:,.0f} | {pct:.1f}% | {d['count']} | {d['calls']:,} |")

    out.append("\n## Top 20 kernels by total time\n")
    out.append("| # | Kernel | Category | Total μs | % | Launches | μs/launch |")
    out.append("|--:|--------|----------|---------:|--:|---------:|----------:|")
    for i, r in enumerate(rows[:20], 1):
        pct = 100.0 * r["total_ns"] / total_ns if total_ns > 0 else 0
        name = r["name"][:100]
        out.append(f"| {i} | `{name}` | {r['category']} | {r['total_ns']/1000:,.1f} | "
                   f"{pct:.2f}% | {r['calls']:,} | {r['avg_ns']/1000:.2f} |")

    pathlib.Path(args.output).write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
