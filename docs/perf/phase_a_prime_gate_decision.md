# Phase A' gate decision report

**Date:** 2026-05-07 (autonomous session)
**Measured:** retroactive scoring of 7 checkpoints via `halo_training/eval/convergence_stats`
**Full data:** `docs/perf/phase_a_prime_convergence_summary.{json,md}`

## Decision

**KILL Phase B adaptive-iter for OdinHalo.**
**GREENLIGHT Phase B layer-skip for OdinFlat (but scoped smaller than original plan).**
**Defer final decision to user on return — see caveats below.**

## Reasoning

### OdinHalo iter convergence (3 iter Parcae loop): KILL

The gate metric is `fraction of tokens where cos(iter_{last-1}, iter_last) > 0.95`.
Measured across 5 OdinHalo checkpoints from this session:

| Step | iter_1 vs iter_2 cos | frac tokens with cos > 0.95 |
|---:|---:|---:|
| 200 | 0.613 | 0.001 |
| 400 (iter 2b) | 0.582 | 0.002 |
| 400 (S1.3) | 0.639 | 0.002 |
| 600 (S1.3) | 0.522 | 0.002 |
| 700 (S1.3) | 0.412 | 0.001 |

Not a single checkpoint crosses even the PARTIAL threshold (0.85). Worse,
the values actively DECREASE with more training (0.64 → 0.41 across steps
400 → 700 on S1.3) — the Parcae loop is using its iterations for
increasingly independent work, not to refine a shared representation.

Conclusion: OdinHalo's 3 iterations are doing substantively different work.
Skipping iter 3 adaptively would lose information for ~99.8% of tokens.
LEAP-style adaptive iter count is **not viable** at current training scale.

### OdinFlat per-layer convergence: PARTIAL signal for layer-skip

For flat models we measure per-layer `cos(h_layer_i, h_final)`. S1.5's
OdinFlat checkpoint at step 400 shows the classic LEAP layer-skip pattern:

```
Layer:    0    1    2    3    4    5    6   7   8   9   10  11  12  13
cos:    .494 .530 .552 .568 .581 .590 .830 .87 .89 .91 .93 .94 .94 1.0
frac>0.95:                                 .003 .01 .02 .02 .04 .14 .31 1.0
```

- Layers 0-5 (before first GQA at position 6): linear climb from 0.49 to 0.59
- Layer 6 (first GQA): JUMP to 0.83 — the attention layer does the big work
- Layers 7-12: gradually converge from 0.87 to 0.94
- Layer 13 (last, GQA): 1.0

Gate signal for Phase B layer-skip:
- **Layer 12 → final frac_high_cos = 0.31** — 31% of tokens could exit at
  layer 12, saving the last layer's compute (~7% of total forward cost).
- **Layer 11 → final = 0.14** — 14% of tokens at layer 11, saving 2 layers.
- **Layer 10 → final = 0.04** — too low for practical use.

Not a slam-dunk GREENLIGHT (we wanted majority > 0.95), but a real signal
for the last 1-2 layers. LEAP aux loss during training would likely push
these fractions higher — this is the dynamic Walmart's MiniLM-L12 paper
reports (2605.01058).

### Caveats

1. **Training scale is still limited.** Latest OdinHalo checkpoint is only
   700 steps into what should be a 52K-step epoch. Patterns may change
   substantially for a fully-trained model.
2. **Effective rank is low** (2-3 across all checkpoints) — the final
   representation lives on a very low-dim subspace. This is unusual for
   LMs and suggests training is concentrating on a small number of modes.
   Could mean the evaluator is noisier than typical; worth recomputing
   post-Sprint 3B.
3. **No OdinFlat-Sprint-3A data yet.** When the full dolma-10B run
   completes we should re-score and see if the layer-skip signal
   strengthens past 0.5 frac_high for layer 12.
4. **LEAP aux loss has never been applied here.** The paper's numbers
   (1.61x inference speedup on MiniLM-L12) come from models TRAINED with
   the aux loss. Our measurements show the pre-aux-loss baseline only.

## Recommendations

### Ship (user can review and accept on return)
- **Drop Phase B OdinHalo adaptive-iter** from roadmap. Add a note in
  the plan doc. Recover the 2-3 days of dev time.
- **Keep Phase B OdinFlat layer-skip as a low-priority optional sprint**,
  scoped to "train last-layer exit head with LEAP aux loss". 3-4 days dev,
  measurable 5-10% inference speedup upside.

### Defer (require user input)
- **When to re-run Phase A' on Sprint 3 end-of-epoch checkpoint.** If
  Sprint 3B produces a fully-trained OdinHalo, re-score and confirm the
  KILL verdict holds. Similarly for OdinFlat at Sprint 3A end — if
  layer 12 → final frac_high crosses 0.5, PARTIAL becomes GREENLIGHT.
- **Whether to block Sprint 1.5 on Phase B decision** (current plan) or
  run in parallel. I defaulted to sequential per the approved roadmap.

## Measurement methodology

Evaluator: `halo_training/eval/convergence_stats.py` (new in this session)
- 3 batches × 2 seqs × 512 tokens = 3072 tokens per checkpoint
- Hooks on layers / shared_layers at forward-hook
- cos computed per-token in fp32
- Effective rank via `torch.linalg.svdvals` on the final representation
- Wall time per checkpoint: ~30s (half that if no SVD)

Reproducibility: rerun `python scripts/score_convergence_retroactive.py
--manifest scripts/phase_a_prime_manifest.txt` on machine A.

## References

- LEAP (Walmart, arXiv 2605.01058): MiniLM-L12 + layer-exit aux loss,
  1.61x inference speedup at 2.2% STS-B quality cost. τ_train=0.98,
  θ_infer=0.95.
- Our earlier plan: `docs/superpowers/plans/2026-05-07-stage1-sprint3-execution-plan.md`
  (Phase A' discovery motivation, gate thresholds).
