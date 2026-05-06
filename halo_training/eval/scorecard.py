"""Scorecard assembly, JSON writing, JSONL index appending.

Separated from `scripts/eval_checkpoint.py` so the same assembly logic is
importable for tests without running the CLI. One-line rule of thumb:
no I/O in this module except `write_scorecard_json` and `append_jsonl_index`.

Schema version 1.0 — if fields change in a backwards-incompatible way, bump
`SCHEMA_VERSION` and handle the transition in callers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from halo_training.eval.common import (
    checkpoint_basename,
    current_timestamp_utc,
    summarise_model,
)

SCHEMA_VERSION = "1.0"


def assemble_scorecard(
    *,
    checkpoint_path: str,
    model_file: str,
    class_name: str,
    model,
    step: Optional[int],
    eval_machine: str,
    eval_duration_s: float,
    eval_config: Dict[str, Any],
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a scorecard dict conforming to schema v1.0.

    `results` is a mapping ``{evaluator_name: evaluator_output_or_None}``.
    Unknown evaluator names are preserved verbatim in the output, so adding
    new evaluators does not require changes here.
    """
    scorecard = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": checkpoint_path,
        "checkpoint_name": checkpoint_basename(checkpoint_path),
        "step": step,
        "model": {
            "file": model_file,
            "class": class_name,
            **summarise_model(model),
        },
        "eval_machine": eval_machine,
        "eval_timestamp": current_timestamp_utc(),
        "eval_duration_s": round(float(eval_duration_s), 2),
        "eval_config": eval_config,
    }
    # Evaluator results are placed at the top level under their own keys so
    # consumers can do `scorecard["per_domain_bpb"]` without nesting.
    for key, value in results.items():
        scorecard[key] = value
    return scorecard


def write_scorecard_json(
    scorecard: Dict[str, Any],
    output_dir: str,
) -> str:
    """Write the full scorecard to ``<output_dir>/<checkpoint_name>.json``.

    Returns the absolute path to the file written.
    """
    os.makedirs(output_dir, exist_ok=True)
    name = scorecard["checkpoint_name"]
    path = Path(output_dir) / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, sort_keys=False)
    return str(path.resolve())


def append_jsonl_index(
    scorecard: Dict[str, Any],
    jsonl_path: str,
) -> None:
    """Append a one-line summary of the scorecard to the global JSONL index.

    Missing evaluator sections are tolerated — missing numbers become ``null``.
    The index is a rolling, append-only file; downstream tools grep/jq it.
    """
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    summary = {
        "ts": scorecard.get("eval_timestamp"),
        "ckpt": scorecard.get("checkpoint_name"),
        "step": scorecard.get("step"),
        "machine": scorecard.get("eval_machine"),
        "avg_bpb": _compute_avg_bpb(scorecard.get("per_domain_bpb")),
        "distinct_2": _safe_get(scorecard, "sampling", "distinct_2"),
        "self_ppl": _safe_get(scorecard, "sampling", "self_ppl"),
        "tok_s_512": _safe_get(scorecard, "inference_profile", "tok_s_seq512_bs1"),
        "peak_mem_gb_512": _safe_get(scorecard, "inference_profile", "peak_mem_gb_seq512"),
        "sample_pack_hash": _safe_get(scorecard, "sample_pack", "output_hash"),
        "full": f"docs/perf/eval-scorecards/{scorecard.get('checkpoint_name')}.json",
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")


def _compute_avg_bpb(per_domain: Optional[Dict[str, Optional[float]]]) -> Optional[float]:
    """Arithmetic mean over non-null BPB values. Skips underscore-prefixed
    diagnostic keys (``_raw_ce``, ``_block_size``, etc.). Returns None if no
    valid entries."""
    if not per_domain:
        return None
    values = [
        v for k, v in per_domain.items()
        if not k.startswith("_") and isinstance(v, (int, float))
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _safe_get(scorecard: Dict[str, Any], section: str, key: str) -> Any:
    """Return ``scorecard[section][key]`` or ``None`` if any lookup step fails."""
    sec = scorecard.get(section)
    if not isinstance(sec, dict):
        return None
    return sec.get(key)
