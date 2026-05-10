"""Track 3.A analyzer: summarize per-param grad norms across V0/V1/V3 diagnostic runs.

Input: checkpoints/diag-frozen-{V0,V1,V3}/diag.jsonl (one JSONL line per opt step,
each with {step, params: [{name, grad_norm, is_none, is_zero}]}).

Output: docs/perf/autokernel-frozen-blast-radius.md with:
  - per-param frozen/live classification
  - blast radius heatmap (which layers/modules have frozen w_gate_up, etc.)
  - summary tables per config

Usage:
  python scripts/analyze_diag_frozen_params.py V0 [V1] [V3]
  python scripts/analyze_diag_frozen_params.py \\
      --v0 path/to/V0.jsonl --v1 path/to/V1.jsonl --v3 path/to/V3.jsonl

Plan: docs/superpowers/plans/2026-05-10-odinflat-throughput-investigation-plan.md §5.A
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def classify_params(records):
    """Classify each parameter across steps.

    Returns a dict: {name: status} where status is one of:
      - "always_none" (grad was None at every step; autograd never traced)
      - "always_zero" (grad was 0.0 at every step; upstream broken)
      - "occasionally_finite" (some steps had finite grads, others didn't)
      - "always_finite" (grad was finite nonzero at every step)
    Also returns per-param max_grad, min_nonzero_grad, mean_grad.
    """
    per_param = defaultdict(list)
    for rec in records:
        for p in rec["params"]:
            per_param[p["name"]].append(p)

    classification = {}
    for name, entries in per_param.items():
        none_count = sum(1 for e in entries if e["is_none"])
        zero_count = sum(1 for e in entries if not e["is_none"] and e["is_zero"])
        finite_count = sum(1 for e in entries if not e["is_none"] and not e["is_zero"])
        total = len(entries)
        finite_norms = [e["grad_norm"] for e in entries
                        if not e["is_none"] and not e["is_zero"]]

        if none_count == total:
            status = "always_none"
        elif zero_count == total:
            status = "always_zero"
        elif (none_count + zero_count) == total:
            # never had a finite grad across the entire run
            status = "always_zero_or_none"
        elif finite_count == total:
            status = "always_finite"
        else:
            status = "occasionally_finite"

        max_grad = max(finite_norms) if finite_norms else None
        min_nz_grad = min(finite_norms) if finite_norms else None
        mean_grad = (sum(finite_norms) / len(finite_norms)) if finite_norms else None

        classification[name] = {
            "status": status,
            "none_frac": none_count / total if total else 0.0,
            "zero_frac": zero_count / total if total else 0.0,
            "finite_frac": finite_count / total if total else 0.0,
            "max_grad": max_grad,
            "min_nz_grad": min_nz_grad,
            "mean_grad": mean_grad,
            "n_steps": total,
        }
    return classification


def group_by_module(classification):
    """Bucket per-param classifications by top-level module prefix.

    e.g. "layers.5.attn.wqkv.weight" → group "layers.*.attn.wqkv.weight"
    """
    grouped = defaultdict(lambda: {"count": 0, "statuses": defaultdict(int)})
    for name, info in classification.items():
        # Collapse layer indices: "layers.5.X" → "layers.*.X"
        parts = name.split(".")
        stripped = []
        for p in parts:
            if p.isdigit():
                stripped.append("*")
            else:
                stripped.append(p)
        key = ".".join(stripped)
        grouped[key]["count"] += 1
        grouped[key]["statuses"][info["status"]] += 1
    return grouped


def summarize_run(label, jsonl_path):
    records = load_jsonl(jsonl_path)
    classification = classify_params(records)
    grouped = group_by_module(classification)

    status_totals = defaultdict(int)
    for info in classification.values():
        status_totals[info["status"]] += 1

    print(f"\n=== {label}: {jsonl_path} ===")
    print(f"Steps recorded: {len(records)}")
    print(f"Total unique params: {len(classification)}")
    print(f"\nStatus tally:")
    for status, count in sorted(status_totals.items(), key=lambda kv: -kv[1]):
        print(f"  {status:>25s}: {count:>4d}")

    print(f"\nGrouped by module ({len(grouped)} groups):")
    for group, info in sorted(grouped.items()):
        statuses = ", ".join(f"{s}:{c}" for s, c in sorted(info["statuses"].items()))
        print(f"  {group:<60s} n={info['count']:>3d} | {statuses}")

    return classification, grouped, status_totals


def emit_markdown(results, out_path):
    """Render the blast-radius markdown doc."""
    lines = []
    lines.append("# Autokernel Frozen-Params Blast Radius (Track 3.A, 2026-05-10)")
    lines.append("")
    lines.append(
        "Per-parameter `.grad.norm()` recorded on every optimizer step across three "
        "configurations on OdinFlat. Identifies which parameters are frozen "
        "(grad=None or grad=0) under each `--optimize-kernels` variant.\n"
    )
    lines.append("Sources:")
    for label, (_, _, _) in results.items():
        lines.append(f"  - `docs/perf/odinflat-profile-2026-05-10/diag-{label}.jsonl`")
    lines.append("")
    lines.append("## Configurations\n")
    lines.append("| Label | Flags | Expected |")
    lines.append("|---|---|---|")
    lines.append("| V0 | (none) | All params get finite nonzero grads (baseline) |")
    lines.append("| V1 | `--optimize-kernels` | `w_gate_up` should be frozen (silu HIP missing autograd) |")
    lines.append("| V3 | `--optimize-kernels --autokernel-exclude fused_silu_gate_mul` | All params should get grads if Phase III rmsnorm fix is complete |")
    lines.append("")

    for label, (classification, grouped, status_totals) in results.items():
        lines.append(f"## {label}: status tally\n")
        lines.append("| Status | Count |")
        lines.append("|---|---:|")
        total = sum(status_totals.values())
        for status in ("always_finite", "occasionally_finite", "always_zero",
                        "always_none", "always_zero_or_none"):
            count = status_totals.get(status, 0)
            pct = (count / total * 100) if total else 0
            lines.append(f"| {status} | {count} ({pct:.1f}%) |")
        lines.append("")

        lines.append(f"### {label}: grouped by module\n")
        lines.append("| Module pattern | Param count | Status distribution |")
        lines.append("|---|---:|---|")
        for group, info in sorted(grouped.items()):
            statuses = "; ".join(f"{s}:{c}" for s, c in sorted(info["statuses"].items()))
            lines.append(f"| `{group}` | {info['count']} | {statuses} |")
        lines.append("")

    lines.append("## Blast-radius comparison\n")
    lines.append("Summary of which parameter groups differ in status between V0 and V1.\n")
    labels = list(results.keys())
    if "V0" in results and "V1" in results:
        v0_cls = results["V0"][0]
        v1_cls = results["V1"][0]
        diffs = []
        for name in sorted(set(v0_cls.keys()) & set(v1_cls.keys())):
            s0 = v0_cls[name]["status"]
            s1 = v1_cls[name]["status"]
            if s0 != s1:
                diffs.append((name, s0, s1))
        if diffs:
            lines.append("| Parameter | V0 status | V1 status |")
            lines.append("|---|---|---|")
            for name, s0, s1 in diffs:
                lines.append(f"| `{name}` | {s0} | {s1} |")
        else:
            lines.append("**All parameters have the same status in V0 and V1**. "
                        "(Implies V1 fired no HIP replacement, or replacements preserve "
                        "gradient flow.)")
    lines.append("")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0", default="docs/perf/odinflat-profile-2026-05-10/diag-V0.jsonl")
    parser.add_argument("--v1", default="docs/perf/odinflat-profile-2026-05-10/diag-V1.jsonl")
    parser.add_argument("--v3", default="docs/perf/odinflat-profile-2026-05-10/diag-V3.jsonl")
    parser.add_argument("--output", default="docs/perf/autokernel-frozen-blast-radius.md")
    args = parser.parse_args()

    results = {}
    for label, path in [("V0", args.v0), ("V1", args.v1), ("V3", args.v3)]:
        if Path(path).exists():
            results[label] = summarize_run(label, path)
        else:
            print(f"\n[{label}] SKIP: {path} not found")

    if results:
        emit_markdown(results, args.output)
    else:
        print("No input files found; nothing to analyze.")


if __name__ == "__main__":
    main()
