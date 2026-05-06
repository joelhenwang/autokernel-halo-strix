"""Sprint 2 evaluation subpackage.

Per-checkpoint scorecard evaluators. Each module in this package exposes a
`run(model, tokenizer, validation_splits, args=None)` function returning a
dict of metrics (or `None` if the evaluator cannot run).

Invoked via `scripts/eval_checkpoint.py`.

Modules (populated progressively during Sprint 2 Phases 2-4):

    common              shared checkpoint/model/tokenizer loading
    per_domain_bpb      Phase 2 — per-domain validation BPB
    sampling            Phase 3 — sampling quality at winning config
    inference_profile   Phase 3 — tok/s + peak memory at fixed seq lengths
    sample_pack         Phase 4 — 20-prompt deterministic regression
    activation_stats    Phase 4 — per-layer kurtosis / RMS / attention entropy

Int4 BPB was dropped from the original design (2026-05-06 revision). See the
scope note in `docs/superpowers/specs/2026-05-06-sprint2-eval-scorecard-design.md`.
"""

from halo_training.eval.common import (
    load_model,
    load_checkpoint,
    load_tokenizer,
    default_validation_splits,
    checkpoint_basename,
)

__all__ = [
    "load_model",
    "load_checkpoint",
    "load_tokenizer",
    "default_validation_splits",
    "checkpoint_basename",
]
