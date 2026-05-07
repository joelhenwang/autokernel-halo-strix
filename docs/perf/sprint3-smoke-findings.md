# Sprint 3 Smoke: OdinHalo dolma-10B — Findings

**Date:** 2026-05-07
**Model:** OdinHalo (57.6M unique / ~156M effective, looped, d_model=768, 6L×3iter)
**Dataset:** datasets/dolma-10b-odin32k.bin
**Config:** DDP 2× Strix Halo, block=512, batch=16, accum=8, lr=8e-4, lr_2d=5e-3, lr_1d=8e-4, --compile (max-autotune-no-cudagraphs)
**Active guards:** --z-loss 1e-4 --z-loss-fraction 0.4 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 100 --imu1-groups --normuon --intra-doc-mask --value-residuals --head-gating
**Duration:** 1000 optimizer steps, 131M tokens, 93 min wall

## Outcome

**Training FAILED with NaN activations in shared layers 4 + 5 between step 900 and 1000.** GradScaler scale collapsed from 2e+03 → 2.8e-17 → 0.0e+00. Zero rollbacks triggered despite the failure (this is the bug the smoke uncovered).

### BPB progression (from auto-eval scorecards)

| step | wiki_bpb | gpt_sm | stem   | dolma  | avg    |
|-----:|---------:|-------:|-------:|-------:|-------:|
| 250  | 2.9236   | 2.9202 | 2.5424 | 2.5466 | **2.73** |
| 500  | 3.1058   | 3.0560 | 2.7685 | 2.8106 | **2.94** |
| 750  | 3.9171   | 3.8333 | 3.4678 | 3.5176 | **3.68** |
| 1000 | 4.1432   | 4.2224 | 3.8824 | 4.2861 | **4.13** |

Training loss peaked (minimum) at step 650 with `loss=3.25`, then climbed back: 3.51 → 3.53 → 3.92 → 4.33 → drifted during step 900–950 overflow phase.

### Activation growth trajectory (from activation_stats.jsonl)

Layer `shared_layers.5` maxabs over time:

| step | layer-5 maxabs | fp16 headroom | scaler.scale |
|-----:|---------------:|--------------:|-------------:|
| 100  | 14.7 (lm_head) | 4469.4        | 1.0e+03 |
| 300  | 38.1           | 1718.8        | 1.0e+03 |
| 500  | 289.7          | 226.1         | 2.0e+03 |
| 700  | 3247.8         | 20.2          | 2.0e+03 |
| 800  | 5061.9         | 12.9          | 2.0e+03 |
| 900  | **9117.5**     | **7.2**       | 5.1e+02 (backoff) |
| 950  | —              | —             | **2.8e-17** (collapse) |
| 1000 | 6 NaN layers   | —             | **0.0e+00** (dead) |

**Growth rate: ~2× per 100 steps** in shared_layers.5. The last healthy sample (step 900, maxabs 9117) was already at fp16 headroom 7.2 — below the safe-threshold recommendation of ≥10 from `knowledge/training/fp16_stability_gfx1151.md`.

## Root cause analysis

1. **LR too high for OdinHalo on dolma.** Sprint 1 validated lr_2d=5e-3 on OdinFlat/wikitext-103 (1 epoch, 936 steps). OdinHalo is looped (×3 Parcae iterations), so effective update magnitude per forward-pass is larger, AND dolma is a more diverse distribution than wikitext. Loss minimum at step 650 followed by monotonic growth is a classic "past-the-saddle" signature of LR too high.

2. **Activation growth is intra-layer, not iter-compounding.** The `iter_scales.clamp(-4, 4)` prevented drift across Parcae iterations but did not limit the within-layer activation magnitudes. shared_layers.5 grew 620× over 500 steps — not a clamped quantity.

3. **`--attn-softcap 50.0` helped but wasn't enough.** Attention scores were bounded ±50 (pre-softmax) so post-softmax attention outputs stayed in healthy range. The overflow was in FFN/SwiGLU outputs or residual paths, not in the attention.

4. **`--z-loss 1e-4` did not prevent logit-magnitude drift.** z-loss was active (first 40% of 1000 steps = steps 1-400); after step 400 it tapered off. Overflow occurred post-step-400 when z-loss was no longer applied. Either the z-loss-fraction was too short or the weight was too small for OdinHalo.

## StabilityGuard gap discovered

**The smoke's most important finding.** The Guard had 3 detection mechanisms, none fired:

1. **check_loss(step_loss)** passed because `step_loss` is filtered for NaN at the microstep level (line 939-944 of train_ddp.py). When *every* microstep produces NaN, step_loss stays 0 (or low), and `check_loss(0.0)` returns True.

2. **check_params** scan happens every 500 steps. Step 1000 would have caught it, but the job ended (max-steps) before the check fired.

3. **too_many_skips** (`_consecutive_skips >= 5`) didn't fire because many accumulation cycles had *some* finite-grad microsteps interspersed with NaN ones, resetting the skip counter.

Meanwhile `scaler.get_scale()` silently decayed 2e+03 → 2.8e-17 → 0 via GradScaler's normal overflow-backoff loop. At `scale=0`, training is effectively dead: every backward scales to zero, every unscaled gradient is NaN.

## Fix shipped (2026-05-07)

Added a **4th detection mechanism**: `StabilityGuard.check_scaler(scaler, step)`. Returns False (triggers rollback) if `scaler.get_scale() < scale_floor` (default 1.0; well below the legitimate GradScaler operating range of `[backoff_factor^N × init_scale, ..., growth_factor^M × init_scale]`).

Wired into the existing `check_loss / check_params / too_many_skips` condition block. Forensics dump classification extended with `"scale_collapse"` trigger.

Test: `test_scaler_collapse_triggers_rollback` in `scripts/test_fp16_stability.py`. 16/16 tests pass.

## Recommendations for Sprint 3 full run

1. **Lower lr_2d to 2e-3 or 3e-3** for OdinHalo on dolma-10B. Sprint 1's 5e-3 was tuned on OdinFlat/wikitext, a different model + dataset.

2. **Extend --z-loss-fraction to 1.0** (always-on) for multi-epoch runs. 0.4 ramps off too early when the model is still drifting.

3. **Keep --attn-softcap 50.0** (helpful, not sufficient alone).

4. **Monitor activation_stats.jsonl** during first ~1000 steps. If any layer's fp16_headroom drops below 20, the LR is too high — stop and lower.

5. **Checkpoint more frequently** (every 100-200 steps for first epoch) so rollback from a scale-collapse event has a recent healthy state to restore.

6. **Consider adding a scaler-underflow-warn threshold at scale < 4** (4× headroom above the hard floor of 1) to give an early-warning log before the hard rollback.

## Artifacts

- `checkpoints/sprint3-smoke-dolma/rank0.log` — full training log
- `checkpoints/sprint3-smoke-dolma/activation_stats.jsonl` (100 samples, 10 layers × 10 sample steps)
- `docs/perf/sprint3-smoke-dolma-activation-stats.jsonl` — local copy for reference
- `docs/perf/eval-scorecards/sprint3-smoke-dolma-step-{250,500,750,1000}.json` — per-step BPB scorecards
- `scripts/analyze_activation_stats.py` — helper for the table above
- `scripts/run_sprint3_smoke.sh` — reproducible runner
- NO `nan_dump_step_*.pt` was written — the guard bug prevented rollback, and forensics only fire on guard trip. With the scale-collapse trigger added, a repeat would emit a dump.

## Status

- **fp16 stability stack validation: PARTIAL PASS** — prevention captured the forensics (activation monitor worked as designed), but response was incomplete (scale-collapse now fixed).
- **Sprint 3 readiness: BLOCKED on LR tuning** — retry with lr_2d=2e-3 before committing to the 50h run.
