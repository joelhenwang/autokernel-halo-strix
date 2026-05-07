"""Sprint 2 per-checkpoint evaluation scorecard CLI.

Runs all registered evaluators against a single checkpoint and emits:

    docs/perf/eval-scorecards/<name>.json    (full scorecard for the checkpoint)
    docs/perf/eval-scorecard.jsonl           (one-line summary appended to index)

Evaluators (populated progressively over Sprint 2 Phases 2-4):

    per_domain_bpb       Phase 2  — per-domain validation BPB
    sampling             Phase 3  — sampling quality at winning config
    inference_profile    Phase 3  — tok/s + peak memory at fixed seq lengths
    sample_pack          Phase 4  — 20-prompt deterministic regression
    activation_stats     Phase 4  — per-layer kurtosis / RMS / attention entropy

In Phase 1 all evaluators are stubbed; the CLI runs end-to-end and produces
valid JSON with empty metric sections. Each later phase fills in one module
at a time without changing this dispatcher.

Usage
-----
Basic (single checkpoint):

    python scripts/eval_checkpoint.py \\
        --checkpoint checkpoints/odin-flat-wikitext-ddp/step_1869.pt \\
        --model models/odin_flat.py --class-name OdinFlat

Selective evaluators (opt-out):

    python scripts/eval_checkpoint.py --checkpoint ... \\
        --skip-activation-stats --skip-sample-pack

Machine label (defaults to $EVAL_MACHINE or hostname):

    EVAL_MACHINE=b python scripts/eval_checkpoint.py --checkpoint ...
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import torch

# Ensure the repo root is on sys.path so `halo_training.eval.*` imports resolve
# regardless of whether the script is run via `python scripts/...` or `python -m`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo_training.eval.common import (
    default_validation_splits,
    load_checkpoint,
    load_model,
    load_tokenizer,
    resolve_eval_machine,
)
from halo_training.eval.scorecard import (
    append_jsonl_index,
    assemble_scorecard,
    write_scorecard_json,
)

# Evaluator registry: the order here is the order metrics appear in the scorecard.
# Each entry is a (json_key, module_name) pair. The module must expose a
# `run(model, tokenizer, validation_splits, args=None)` function. Missing
# modules are tolerated in Phase 1 (stub behaviour) — a warning is printed.
EVALUATORS = [
    ("per_domain_bpb", "halo_training.eval.per_domain_bpb"),
    ("sampling", "halo_training.eval.sampling"),
    ("inference_profile", "halo_training.eval.inference_profile"),
    ("sample_pack", "halo_training.eval.sample_pack"),
    ("activation_stats", "halo_training.eval.activation_stats"),
    # Phase A' (2026-05-07): convergence / exit-readiness diagnostics for
    # LEAP-style Phase B gating. Safe to skip via --skip-convergence-stats.
    ("convergence_stats", "halo_training.eval.convergence_stats"),
]

WALL_TIME_BUDGET_S = 15 * 60  # 15 min hard cap per spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sprint 2 per-checkpoint evaluation scorecard",
    )
    parser.add_argument("--checkpoint", required=True, help=".pt checkpoint path")
    parser.add_argument("--model", required=True, help="Model .py file (e.g. models/odin_flat.py)")
    parser.add_argument("--class-name", required=True, help="Model class inside --model")
    parser.add_argument(
        "--tokenizer-path",
        default="tokenizers/odin-32k/tokenizer.json",
        help="HF tokenizer.json path (default: odin-32k)",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/perf/eval-scorecards",
        help="Directory for per-checkpoint JSON files",
    )
    parser.add_argument(
        "--jsonl-index",
        default="docs/perf/eval-scorecard.jsonl",
        help="Global one-line-per-run JSONL index",
    )
    parser.add_argument(
        "--eval-machine",
        default=None,
        help="Override for EVAL_MACHINE env var (default: env or hostname)",
    )
    parser.add_argument(
        "--prior-checkpoint",
        default=None,
        help="Prior checkpoint for sample-pack diff (default: auto-detect)",
    )
    parser.add_argument(
        "--prompts-file",
        default="evals/sample_pack_v1.txt",
        help="Frozen sample-pack prompts file",
    )
    parser.add_argument(
        "--max-tokens-per-domain",
        type=int,
        default=50_000,
        help="Per-domain BPB evaluation token budget",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--wall-time-budget-s",
        type=int,
        default=WALL_TIME_BUDGET_S,
        help="Fail loud if total wall time exceeds this (default: 900s)",
    )

    # Skip flags: one per evaluator (all default False => run all)
    for key, _mod in EVALUATORS:
        flag = f"--skip-{key.replace('_', '-')}"
        parser.add_argument(flag, action="store_true", help=f"Skip the {key!r} evaluator")

    return parser


def _should_skip(args: argparse.Namespace, key: str) -> bool:
    attr = "skip_" + key
    return bool(getattr(args, attr, False))


def _run_single_evaluator(key: str, module_name: str, *, model, tokenizer, splits, args) -> Any:
    """Import and run a single evaluator; return its dict, or None on any failure."""
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"  [{key}] evaluator module not found ({module_name}); returning null "
              "(expected during Phase 1 scaffolding)")
        return None

    run_fn = getattr(mod, "run", None)
    if run_fn is None:
        print(f"  [{key}] module {module_name} has no run() function; returning null")
        return None

    t0 = time.time()
    try:
        result = run_fn(model, tokenizer, splits, args=args)
    except Exception as exc:  # noqa: BLE001 — we want broad isolation per evaluator
        print(f"  [{key}] FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        return {"error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.time() - t0
    print(f"  [{key}] done in {elapsed:.1f}s")
    return result


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    t_start = time.time()

    # Resolve machine label early so it appears in all log lines
    eval_machine = resolve_eval_machine(args.eval_machine)
    print(f"eval_checkpoint.py  (Sprint 2 scorecard)")
    print(f"  checkpoint:   {args.checkpoint}")
    print(f"  model:        {args.model} :: {args.class_name}")
    print(f"  machine:      {eval_machine}")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  jsonl_index:  {args.jsonl_index}")
    print()

    # Load model + checkpoint (fail loud here — scorecard can't run without model)
    print("Loading model and checkpoint ...")
    model = load_model(args.model, args.class_name).to(args.device).eval()
    model, step = load_checkpoint(model, args.checkpoint, device=args.device)
    model.half()  # fp16 for eval to match training-time behaviour
    print(f"  loaded; step={step}")

    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = load_tokenizer(args.tokenizer_path)

    validation_splits = default_validation_splits()
    available = [k for k, v in validation_splits.items() if v is not None]
    missing = [k for k, v in validation_splits.items() if v is None]
    print(f"Validation splits:   available={available}  missing={missing}")
    print()

    # Run each evaluator
    eval_config: Dict[str, Any] = {
        "max_tokens_per_domain": args.max_tokens_per_domain,
        "prompts_file": args.prompts_file,
        "prior_checkpoint": args.prior_checkpoint,
    }

    print("Running evaluators:")
    results: Dict[str, Any] = {}
    for key, module_name in EVALUATORS:
        if _should_skip(args, key):
            print(f"  [{key}] SKIPPED (--skip-{key.replace('_','-')})")
            results[key] = None
            continue
        results[key] = _run_single_evaluator(
            key, module_name,
            model=model, tokenizer=tokenizer, splits=validation_splits, args=args,
        )
    print()

    elapsed = time.time() - t_start

    # Wall-time budget enforcement: warn but don't fail the write
    # (we want the scorecard persisted even if an evaluator ran long).
    if elapsed > args.wall_time_budget_s:
        print(
            f"WARNING: wall time {elapsed:.0f}s exceeds budget "
            f"{args.wall_time_budget_s}s — investigate slow evaluator."
        )

    # Assemble scorecard
    scorecard = assemble_scorecard(
        checkpoint_path=args.checkpoint,
        model_file=args.model,
        class_name=args.class_name,
        model=model,
        step=step,
        eval_machine=eval_machine,
        eval_duration_s=elapsed,
        eval_config=eval_config,
        results=results,
    )

    # Write outputs
    json_path = write_scorecard_json(scorecard, args.output_dir)
    append_jsonl_index(scorecard, args.jsonl_index)

    print(f"Scorecard written:")
    print(f"  JSON:  {json_path}")
    print(f"  JSONL: {args.jsonl_index}")
    print(f"  Total wall time: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
