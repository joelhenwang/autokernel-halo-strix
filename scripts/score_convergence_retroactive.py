#!/usr/bin/env python
"""Phase A': retroactively score existing checkpoints with convergence_stats.

Runs the convergence_stats evaluator (only) against a set of checkpoints and
produces a summary JSON + markdown table. Skips the other heavy evaluators
to keep wall time minimal.

Usage::

    python scripts/score_convergence_retroactive.py \\
        --checkpoint checkpoints/sprint3-iter2b-lr2_5e3/step_400.pt \\
        --model models/odin_halo.py --class-name OdinHalo

Or with a manifest file (one line per checkpoint spec)::

    python scripts/score_convergence_retroactive.py --manifest manifest.txt

Manifest format: tab-separated ``<ckpt_path>\\t<model_path>\\t<class_name>``.

Outputs:
  docs/perf/phase_a_prime_convergence_summary.json    -- full results
  docs/perf/phase_a_prime_convergence_summary.md      -- markdown table

Output contains both the raw per-checkpoint metrics and a gate-decision
summary for OdinHalo looped checkpoints:
  - mean iter_(mr-2) -> iter_(mr-1) frac_high_cos
  - recommended Phase B greenlight: True / False / Borderline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from halo_training.eval import convergence_stats  # noqa: E402
from halo_training.eval.common import (  # noqa: E402
    default_validation_splits,
    load_checkpoint,
    load_model,
)


# Gate thresholds (Phase A' plan):
#   >= 0.95 median frac_high -> GREENLIGHT Phase B
#   >= 0.85 and < 0.95       -> PARTIAL (aux loss only, no adaptive iter)
#   <  0.85                  -> KILL Phase B
GATE_GREENLIGHT = 0.95
GATE_PARTIAL = 0.85


def score_one(checkpoint_path: str, model_path: str, class_name: str,
              device: str = "cuda") -> Dict[str, Any]:
    """Load a checkpoint and run convergence_stats on it. Returns a result dict."""
    t0 = time.time()
    model = load_model(model_path, class_name).to(device).eval()
    model, step = load_checkpoint(model, checkpoint_path, device=device)
    model.half()
    splits = default_validation_splits()

    conv_out = convergence_stats.run(
        model, tokenizer=None, validation_splits=splits,
        num_batches=3, seq_len=512, batch_size=2)

    elapsed = time.time() - t0
    return {
        "checkpoint": checkpoint_path,
        "model_file": model_path,
        "class_name": class_name,
        "step": step,
        "elapsed_s": round(elapsed, 1),
        "convergence_stats": conv_out,
    }


def decide_gate(conv_out: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Apply Phase A' gate thresholds to a single result's convergence stats.

    Returns (status, details) where status in {"GREENLIGHT","PARTIAL","KILL",
    "N/A"}. Non-looped / missing -> "N/A".
    """
    if not conv_out or not conv_out.get("is_looped"):
        return "N/A", {"reason": "not a looped model"}
    fracs = conv_out.get("iter_k_cos_to_final_frac_high") or []
    if not fracs:
        return "N/A", {"reason": "no iter frac_high data"}
    # Gate on the last-before-final iter's fraction (mr-2 index since list
    # has length mr-1). Example: mr=3 -> fracs = [frac(iter0->final),
    # frac(iter1->final)]. We take the last element (iter1 vs iter2 final).
    gate_metric = fracs[-1]
    if gate_metric is None:
        return "N/A", {"reason": "null gate metric"}
    if gate_metric >= GATE_GREENLIGHT:
        status = "GREENLIGHT"
    elif gate_metric >= GATE_PARTIAL:
        status = "PARTIAL"
    else:
        status = "KILL"
    return status, {
        "gate_metric_frac_high_cos": gate_metric,
        "thresholds": {"greenlight": GATE_GREENLIGHT, "partial": GATE_PARTIAL},
        "iter_cos_to_final": conv_out.get("iter_k_cos_to_final"),
        "iter_transition_cos": conv_out.get("iter_transition_cos"),
    }


def render_markdown(results: List[Dict[str, Any]]) -> str:
    """Render the results as a markdown summary + table."""
    lines = [
        "# Phase A' convergence diagnostics",
        "",
        "Retroactive scoring of existing checkpoints against "
        "`halo_training/eval/convergence_stats`.",
        "",
        "Gate thresholds (on the last iter-pair's fraction of tokens with "
        "cos > 0.95):",
        "",
        f"  - **GREENLIGHT** Phase B: frac_high_cos >= {GATE_GREENLIGHT}",
        f"  - **PARTIAL** (aux loss only, no adaptive iter): "
        f"{GATE_PARTIAL} <= frac_high_cos < {GATE_GREENLIGHT}",
        f"  - **KILL** Phase B: frac_high_cos < {GATE_PARTIAL}",
        "",
        "## Results",
        "",
        "| Checkpoint | Step | Model | Looped | iter_k_cos_to_final | "
        "frac_high (tau=0.95) | Gate |",
        "|---|---:|---|:---:|---|---|:---:|",
    ]
    for r in results:
        ckpt = Path(r["checkpoint"]).parent.name + "/" + Path(r["checkpoint"]).name
        step = r.get("step") or "-"
        cls = r["class_name"]
        conv = r.get("convergence_stats") or {}
        looped = "yes" if conv.get("is_looped") else "no"
        cos_list = conv.get("iter_k_cos_to_final") or []
        frac_list = conv.get("iter_k_cos_to_final_frac_high") or []
        cos_str = ", ".join(f"{v:.3f}" if v is not None else "-" for v in cos_list) or "-"
        frac_str = ", ".join(f"{v:.3f}" if v is not None else "-" for v in frac_list) or "-"
        status, _ = decide_gate(conv)
        lines.append(
            f"| `{ckpt}` | {step} | {cls} | {looped} | "
            f"{cos_str} | {frac_str} | **{status}** |"
        )
    lines.append("")
    lines.append("## Gate decision (aggregate)")
    lines.append("")
    looped_results = [r for r in results
                      if (r.get("convergence_stats") or {}).get("is_looped")]
    if looped_results:
        statuses = [decide_gate(r["convergence_stats"])[0] for r in looped_results]
        greenlight = statuses.count("GREENLIGHT")
        partial = statuses.count("PARTIAL")
        kill = statuses.count("KILL")
        lines.append(f"- {greenlight} GREENLIGHT, {partial} PARTIAL, {kill} KILL "
                     f"(out of {len(looped_results)} looped checkpoints).")
        if greenlight >= len(looped_results) / 2:
            agg = "**GREENLIGHT** (majority above 0.95 threshold)"
        elif greenlight + partial >= len(looped_results) / 2:
            agg = "**PARTIAL** (majority above 0.85 but below 0.95)"
        else:
            agg = "**KILL** (majority below 0.85)"
        lines.append(f"- Aggregate recommendation: {agg}")
    else:
        lines.append("- No looped checkpoints evaluated.")
    lines.append("")
    lines.append("## Per-layer convergence (flat or last-iter view)")
    lines.append("")
    for r in results:
        conv = r.get("convergence_stats") or {}
        ckpt = Path(r["checkpoint"]).parent.name + "/" + Path(r["checkpoint"]).name
        lines.append(f"### {ckpt}")
        lines.append("")
        per_layer = conv.get("per_layer_cos_to_final") or []
        frac_pl = conv.get("per_layer_cos_to_final_frac_high") or []
        inter = conv.get("inter_layer_transition_cos") or []
        eff = conv.get("effective_rank_final")
        if per_layer:
            lines.append("per_layer_cos_to_final: " +
                         ", ".join(f"{v:.3f}" if v is not None else "-"
                                   for v in per_layer))
        if frac_pl:
            lines.append("per_layer_frac_high_cos: " +
                         ", ".join(f"{v:.3f}" if v is not None else "-"
                                   for v in frac_pl))
        if inter:
            lines.append("inter_layer_transition_cos: " +
                         ", ".join(f"{v:.3f}" if v is not None else "-"
                                   for v in inter))
        if eff is not None:
            lines.append(f"effective_rank_final: {eff:.1f}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="Checkpoint path (repeatable; use with --model/--class-name)")
    ap.add_argument("--model", action="append", default=[],
                    help="Model .py (repeatable; paired with --checkpoint)")
    ap.add_argument("--class-name", action="append", default=[],
                    help="Class name (repeatable; paired with --checkpoint)")
    ap.add_argument("--manifest", default=None,
                    help="Tab-separated manifest: ckpt<TAB>model<TAB>class")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json",
                    default="docs/perf/phase_a_prime_convergence_summary.json")
    ap.add_argument("--output-md",
                    default="docs/perf/phase_a_prime_convergence_summary.md")
    args = ap.parse_args(argv)

    specs: List[Tuple[str, str, str]] = []
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    raise ValueError(f"bad manifest line: {line!r}")
                specs.append((parts[0], parts[1], parts[2]))
    else:
        if not (len(args.checkpoint) == len(args.model) == len(args.class_name)):
            raise SystemExit("--checkpoint / --model / --class-name counts must match")
        for ckpt, mdl, cls in zip(args.checkpoint, args.model, args.class_name):
            specs.append((ckpt, mdl, cls))
    if not specs:
        raise SystemExit("no checkpoints specified")

    print(f"Scoring {len(specs)} checkpoint(s) on device={args.device} ...")
    results: List[Dict[str, Any]] = []
    for i, (ckpt, mdl, cls) in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {ckpt} ({cls})")
        try:
            r = score_one(ckpt, mdl, cls, device=args.device)
        except Exception as exc:  # noqa: BLE001 — tolerant scoring
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            r = {"checkpoint": ckpt, "model_file": mdl, "class_name": cls,
                 "error": f"{type(exc).__name__}: {exc}"}
        status, details = decide_gate(r.get("convergence_stats") or {})
        r["gate_status"] = status
        r["gate_details"] = details
        results.append(r)
        # Free GPU memory between checkpoints
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"checkpoints": results}, f, indent=2)
    md = render_markdown(results)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)
    print()
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
