"""Phase A.3 analysis: aggregate 14-model × 3-config diagnostic outputs
into a single blast-radius truth matrix.

Reads all diag.jsonl files under checkpoints/audit-phase-a3/, compares
V0 baseline vs V1/V3 configs per model, flags frozen params, produces
docs/perf/autokernel-audit-2026-05-11.md.

Usage:
  python scripts/analyze_audit_phase_a3.py
  python scripts/analyze_audit_phase_a3.py --audit-dir checkpoints/audit-phase-a3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-use the classifier from Track 3.A's analyzer.
from scripts.analyze_diag_frozen_params import classify_params, load_jsonl


def summarize(path: Path) -> dict:
    """Return {status: count} over all params in the file."""
    try:
        records = load_jsonl(path)
    except Exception as exc:
        return {"error": str(exc)}
    classification = classify_params(records)
    counts = defaultdict(int)
    frozen_params = []
    for name, info in classification.items():
        counts[info["status"]] += 1
        if info["status"] in ("always_none", "always_zero"):
            frozen_params.append((name, info["status"]))
    return {
        "n_params": len(classification),
        "counts": dict(counts),
        "frozen_params": frozen_params,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="checkpoints/audit-phase-a3")
    ap.add_argument("--output", default="docs/perf/autokernel-audit-2026-05-11.md")
    args = ap.parse_args()

    audit_root = REPO_ROOT / args.audit_dir
    if not audit_root.exists():
        print(f"Audit directory not found: {audit_root}")
        print("Did Phase A.3 batch complete? Expected diag.jsonl files under:")
        print(f"  {audit_root}/<label>-<config>/diag.jsonl")
        return 1

    # Find all diag.jsonl files.
    diag_files = sorted(audit_root.rglob("diag.jsonl"))
    print(f"Found {len(diag_files)} diag.jsonl files under {audit_root}")

    # Group by (label, config).
    results = {}   # (label, config) -> summary dict
    for path in diag_files:
        # Directory name encodes e.g. "odin_flat-V0".
        dir_name = path.parent.name
        if "-" not in dir_name:
            print(f"  skipping unexpected dir: {dir_name}")
            continue
        label, config = dir_name.rsplit("-", 1)
        print(f"  analyzing {label}/{config}")
        results[(label, config)] = summarize(path)

    # Extract all labels and configs.
    labels = sorted({k[0] for k in results.keys()})
    configs = sorted({k[1] for k in results.keys()})

    # Compute frozen-param deltas V1 vs V0, V3 vs V0.
    deltas = {}  # (label, cmp) -> {newly_frozen, total_frozen}
    for label in labels:
        v0 = results.get((label, "V0"), {})
        for cmp in ("V1", "V3"):
            other = results.get((label, cmp), {})
            if "error" in v0 or "error" in other or not v0 or not other:
                continue
            v0_frozen = set(p for p, _ in v0.get("frozen_params", []))
            other_frozen = set(p for p, _ in other.get("frozen_params", []))
            newly = other_frozen - v0_frozen
            deltas[(label, cmp)] = {
                "newly_frozen_count": len(newly),
                "newly_frozen_params": sorted(newly)[:10],
                "total_frozen_count": len(other_frozen),
            }

    # Emit markdown.
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Autokernel Audit Phase A (2026-05-11)")
    lines.append("")
    lines.append("Full 14-model × 3-config diagnostic truth matrix from "
                 "`scripts/audit_phase_a3_batch.sh`. Each probe ran 50 opt steps "
                 "with `--diag-frozen-params`, recording every parameter's "
                 "`.grad.norm()` on every step.\n")
    lines.append("Configurations:\n")
    lines.append("- **V0**: baseline (no `--optimize-kernels`)")
    lines.append("- **V1**: `--optimize-kernels` (all patterns active)")
    lines.append("- **V3**: `--optimize-kernels --autokernel-exclude fused_silu_gate_mul`\n")
    lines.append("## Summary: newly-frozen params vs V0 baseline\n")
    lines.append("| Model | V1 newly frozen | V3 newly frozen | V1 total frozen | V3 total frozen | Verdict |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for label in labels:
        d_v1 = deltas.get((label, "V1"), {})
        d_v3 = deltas.get((label, "V3"), {})
        v0_r = results.get((label, "V0"), {})
        v0_n = v0_r.get("n_params", "?") if "error" not in v0_r else "err"

        def _fmt(d, key):
            if not d:
                return "—"
            return str(d.get(key, 0))

        verdict_parts = []
        if d_v1.get("newly_frozen_count", 0) > 0:
            verdict_parts.append(f"**V1 freezes {d_v1['newly_frozen_count']} params**")
        if d_v3.get("newly_frozen_count", 0) > 0:
            verdict_parts.append(f"**V3 freezes {d_v3['newly_frozen_count']} params**")
        if not verdict_parts:
            if d_v1 or d_v3:
                verdict_parts.append("clean")
            else:
                verdict_parts.append("(missing data)")
        verdict = "; ".join(verdict_parts)

        lines.append(f"| `{label}` ({v0_n}p) | "
                     f"{_fmt(d_v1, 'newly_frozen_count')} | "
                     f"{_fmt(d_v3, 'newly_frozen_count')} | "
                     f"{_fmt(d_v1, 'total_frozen_count')} | "
                     f"{_fmt(d_v3, 'total_frozen_count')} | "
                     f"{verdict} |")
    lines.append("")

    # Per-model detail.
    lines.append("## Per-model detail\n")
    for label in labels:
        lines.append(f"### `{label}`\n")
        for cfg in configs:
            r = results.get((label, cfg), None)
            if r is None:
                lines.append(f"- **{cfg}**: missing")
                continue
            if "error" in r:
                lines.append(f"- **{cfg}**: load error: `{r['error']}`")
                continue
            c = r.get("counts", {})
            finite = c.get("always_finite", 0)
            n_none = c.get("always_none", 0)
            n_zero = c.get("always_zero", 0)
            n_occ  = c.get("occasionally_finite", 0)
            lines.append(f"- **{cfg}**: "
                         f"{finite} always_finite, {n_none} always_none, "
                         f"{n_zero} always_zero, {n_occ} occasionally_finite")
            fp = r.get("frozen_params", [])
            if fp and cfg != "V0":
                sample = ", ".join(f"`{n}`" for n, _ in fp[:5])
                more = f" (+ {len(fp) - 5} more)" if len(fp) > 5 else ""
                lines.append(f"  - frozen examples: {sample}{more}")
        lines.append("")

    lines.append("## Phase B fix map\n")
    lines.append("Maps each observed freeze pattern to the Replacement class that caused it:\n")
    lines.append("| Observed freeze | Root cause (from static audit) | Phase B task |")
    lines.append("|---|---|---|")
    lines.append("| `ffn.w_gate_up.weight` always_none + `ffn_norm.weight` always_zero | `_FusedSwiGLUReplacement` raw kernel_fn | B.1 |")
    lines.append("| `norm.weight` always_zero | `_LayerNormReplacement` raw kernel_fn | B.3 |")
    lines.append("| upstream of attention output always_zero | `_FusedQKVAttentionReplacement` raw rotary_fn | B.4 |")
    lines.append("| `ffn_norm.weight` always_zero when block replaced | `_FusedResidualRMSNormBlockReplacement` kernel_fn_dual | B.4b |")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    json_out = out_path.with_suffix(".json")
    json_out.write_text(json.dumps({
        "results": {f"{k[0]}-{k[1]}": v for k, v in results.items()},
        "deltas": {f"{k[0]}-{k[1]}": v for k, v in deltas.items()},
    }, indent=2, default=str), encoding="utf-8")

    print(f"\nWrote {out_path}")
    print(f"Wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
