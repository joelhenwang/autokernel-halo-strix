---
title: "v3 Experiment Roadmap"
domain: plans
type: plan
status: roadmap
sprint: v3.0-v3.2
tags: [v3, roadmap, distillation, parcae, moe, training-dynamics, stability]
related:
  - ../specs/2026-05-07-self-distillation-odinflat-odinhalo.md
  - ../specs/2026-05-07-frankenmoe-loop-design.md
  - ../../../knowledge/architectures/v3_speculative_directions_2026.md
  - ../../../knowledge/architectures/a2_reversible_parcae_audit_2026.md
---

# v3 Experiment Roadmap (2026-05-07)

## Status

**Operational plan** for executing the v3 speculative catalogue. Each
phase has explicit triggers, budgets, decision gates, and kill dates.
Ideas without triggers fired by their kill date are deleted, not
parked.

Three phases:

- **v3.0** — low-risk wins (2–3 weeks). Fires immediately post-FrankenMoE-Loop v2 L9, or in parallel with it.
- **v3.1** — capability unlocks (3–4 weeks). Gated by v3.0 infra.
- **v3.2** — research explorations (4–6 weeks). Gated by v3.1 validation and specific triggers.

Parallel tracks (E5, E6) can run independently of the phase ordering.

## Phase v3.0 — low-risk wins

### Purpose

Ship cheap regularizers and a direct response to the Sprint 3 smoke
failure before any capacity-changing work. Each experiment is
self-contained, <2 weeks effort, with clear kill criteria.

### v3.0 experiments

| ID | Experiment | Effort | Trigger | Decision gate | Kill date |
|---|---|---|---|---|---|
| E4 | Activation-growth damping regularizer | 3 days infra + 1 smoke run | **ACTIVE** (Sprint 3 smoke failure) | Sprint 3 dolma-10B survives ≥2000 steps with no StabilityGuard rollback | 2026-06-15 |
| B1 | Hidden-state diffusion (corrected σ) | 3 days infra + 1 epoch wikitext | v3.0 slot available | ≥1% BPB improvement at step 1869 wikitext, or measurable train/val gap reduction | 2026-07-01 |
| B4 | Entropy conservation loss | 2 days infra + 1 epoch wikitext | runs concurrent with B1 | Entropy trajectory decreases monotonically AND BPB parity ≤ +1% | 2026-07-01 |

### v3.0 design deltas from original catalogue

Each of these is a **corrected** version of the v3 entry:

#### E4 Activation-growth damping

Novel to this roadmap; not in original catalogue. Born from the
2026-05-07 Sprint 3 smoke analysis (`shared_layers.5` exponential
growth 38 → 9117 over 500 steps, leading to GradScaler collapse).

**Mechanism**:

```python
# In halo_training/trainer.py, once per N steps (N=50 default):
layer_maxabs = activation_monitor.latest_maxabs_per_layer()  # dict name -> fp16 maxabs
target = 200.0  # derived from warmup percentile in first 100 steps
growth_excess = max(0, layer_maxabs - target)
reg = lambda_reg * (growth_excess ** 2).mean()
loss = loss + reg
```

- `lambda_reg` = 1e-4 default, tunable.
- `target` computed adaptively: 95th percentile of layer maxabs over
  steps 50–150, capped at 500.
- Regularizer runs on the monitored layers only (already-available
  hooks) — no additional memory.
- Expected overhead: <1% throughput (one extra reduction per monitored
  layer per step).

**Why this over clamping**: clamping is discontinuous and loses
gradient signal; an L2 penalty is smooth. It also teaches the model to
self-regulate rather than depending on forward-time hacks.

#### B1 corrected

Original v3 entry proposed σ=[0.3, 0.1, 0.0] noise schedule with last-10%
anneal. My audit flagged σ=0.3 as too aggressive for fp16 with
z-loss active.

**Corrected defaults**:

- `σ_iter0 = 0.08`, `σ_iter1 = 0.04`, `σ_iter2 = 0.0`.
- Linear anneal to all-zero across steps 70%–90% of training (wider
  window than the original 90%–100%; anneals earlier to reduce late
  inference-time shift).
- Seeded RNG per step so reproduction works across checkpoints.
- CLI flag: `--parcae-noise-schedule 0.08,0.04,0.0 --parcae-noise-anneal 0.7,0.9`.
- Fallback: `--parcae-noise-schedule 0,0,0` is the exact
  non-regularized baseline (bit-identical loss on a fixed seed).

#### B4 corrected

Original v3 entry proposed `λ=0.05`, strict monotonic penalty.

**Corrected**:

- `λ = 0.02` (lower bar to start).
- Hinge loss with tolerance: only penalize increases beyond
  `+20% of iter-0 variance`. Allows small iteration-to-iteration
  variance fluctuation without penalty.
- Variance computed in fp32 (trivial cost; avoids fp16 noise in the
  regularizer itself).
- CLI flag: `--entropy-conservation 0.02 --entropy-hinge-tolerance 0.2`.

### v3.0 compose matrix

B1, B4, E4 can all run in a single training run (orthogonal
mechanisms):

|  | E4 | B1 | B4 |
|---|:---:|:---:|:---:|
| E4 | — | ✓ compose | ✓ compose |
| B1 | ✓ compose | — | ✓ compose (B1 adds noise, B4 constrains variance growth; if B1's noise pushes variance up, B4 hinge tolerance absorbs it) |
| B4 | ✓ compose | ✓ compose | — |

**Recommended v3.0 launch**: single DDP wikitext-103 run with
all three active (`E4 + B1 + B4`). Dissect contributions via
ablation if BPB improves; if not, three 1-hour microbenches on
babylm tell us which component fires.

### v3.0 total budget

- Infra implementation: ~1.5 weeks (E4: 3 days, B1: 3 days, B4: 2
  days, testing: 2 days).
- Training: ~3 hours DDP wikitext for the combined run, plus ~3
  hours for ablations if needed.
- **Total wall**: ~2 weeks.

## Phase v3.1 — capability unlocks

### Purpose

After v3.0 ships, tackle the experiments that actually change what
the model can do. E1 distillation is highest expected value; E2 and
E3 are curriculum plays that compose.

### v3.1 experiments

| ID | Experiment | Effort | Trigger | Decision gate | Kill date |
|---|---|---|---|---|---|
| E1 | OdinFlat→OdinHalo self-distillation | 2 weeks (full spec: `specs/2026-05-07-self-distillation-odinflat-odinhalo.md`) | v3.0 ships; OdinFlat wikitext checkpoint present (already true) | ≥50% BPB gap closure on wikitext (1.888 → ≤ 1.835) AND no held-out domain regresses >3% | 2026-08-01 |
| E2 | Parcae iteration-warmup curriculum | 1 week | E4 deploys but `shared_layers.5` maxabs still grows | maxabs growth bounded below 500 over 10K steps AND BPB parity ≤ +1% | 2026-08-01 |
| E3 | Progressive context-length schedule | 1 week | orthogonal; compute-slot dependent | ≥2% BPB improvement at fixed token budget | 2026-08-15 |

### v3.1 design deltas

#### E1 (spec-ready)

Fully specified in `docs/superpowers/specs/2026-05-07-self-distillation-odinflat-odinhalo.md`.
Three variants (V1 pure KL, V2 KL+CE mix recommended, V3 top-k). No
model changes; new `halo_training/distill.py` + `distill_loss.py`
modules and CLI flags. Control checkpoint already exists; teacher
checkpoint already exists.

#### E2 Parcae iteration-warmup curriculum

**Mechanism**: train first N% of steps at `mean_recurrence=2`, then
resume with `mean_recurrence=3` plus a freshly initialized
`iter_scales[2] = 0` (bias-zero at start), letting the third iteration
"come online" as the model has already stabilized.

**Motivation**: Sprint 3 smoke analysis suggests the last layer of
the last iteration is where activation growth compounds. A warmup
that keeps `mean_recurrence=2` for the first ~20% of training gives
the preceding iterations a chance to settle before the third
iteration starts accumulating.

**Implementation**:

```python
# halo_training/cli.py new flag:
--iter-warmup-steps 1000   # how long to run at mean_recurrence - 1

# halo_training/trainer.py:
if step < iter_warmup_steps:
    model.set_mean_recurrence(model.original_mean_recurrence - 1)
else:
    model.set_mean_recurrence(model.original_mean_recurrence)
```

`set_mean_recurrence` already works on the OdinHalo forward path (it's
a simple attribute); need to verify `compile_zones` doesn't recompile
when it changes (or at minimum, only once per transition).

**Kill**: if warmup causes >2% BPB regression at end of training
vs non-warmup, abandon.

#### E3 Progressive context-length schedule

**Mechanism**: start training at `block_size=256`, transition to
`block_size=512` or `block_size=1024` at a fixed step (e.g., step
1000). Literature-proven at scale (e.g., LLaMA 2, Qwen curriculum
work).

**Implementation**: requires that the dataloader can switch mid-epoch.
Current streaming binary format supports variable-length windowing;
CLI flag `--block-size-schedule 256,512 --block-size-schedule-at 1000`.

**Kill**: if context-switch causes loss spike >0.5 at the transition,
abandon or widen the transition window.

### v3.1 compose matrix

|  | E1 | E2 | E3 |
|---|:---:|:---:|:---:|
| E1 | — | ✗ conflict (E2 changes forward behavior mid-training, breaks teacher/student compute parity) | ✓ compose (both change the data distribution progressively) |
| E2 | ✗ conflict | — | ~ partial (both change training schedule; E2's iter warmup should complete before E3's block increase) |
| E3 | ✓ compose | ~ partial | — |

**Recommended v3.1 sequencing**: E1 first (standalone), then E2 or E3
in a second run. Never E1+E2 in one run.

## Phase v3.2 — research explorations

### Purpose

Higher-risk, higher-novelty ideas, gated by explicit triggers. Each
requires a specific observation or downstream motivation; otherwise
they stay parked.

### v3.2 experiments

| ID | Experiment | Effort | Trigger | Kill date |
|---|---|---|---|---|
| A5 | Heterogeneous-capacity MoE (skip LZ77) | 3 weeks | FrankenMoE-Flat v1 L9 validated; uniform-MoE baseline exists | 2026-09-15 |
| A3 | Shared Latent Workspace | 2–3 weeks | Scorecard long-range probe regresses, OR research slack | 2026-09-01 |
| A2 | Reversible Parcae | 4 weeks (see `knowledge/architectures/a2_reversible_parcae_audit_2026.md`) | Memory becomes binding constraint on mean_recurrence or batch | 2026-09-01 |
| B2 | Temporal-contrastive iteration | 2 weeks | Iteration-divergence analysis shows iter-1 ≈ iter-2 | 2026-09-01 |

### v3.2 design deltas

All four entries are **as documented in the cut-down v3 catalogue**
with corrections applied there (A5 loses its LZ77 routing claim and
becomes strictly heterogeneous-capacity; A2 is gated on the audit's
triggers; B2 adds the DDP all-gather requirement for negatives).

### v3.2 compose matrix

These are large single-experiment commitments; they do not naturally
compose. Recommended: one at a time, each with a full DDP run.

## Parallel tracks (run anytime)

### E5 Router / expert variants beyond R2

**Description**: family of comparative ablations in the context of
FrankenMoE-Loop v2 L11. Not one experiment; a series of four
1-week ablations:

- E5a: Sinkhorn routing vs R2 sticky.
- E5b: Expert-choice routing (load-balanced by construction).
- E5c: Hash routing (free throughput, quality floor check).
- E5d: Per-token capacity routing (dynamic compute).

**Trigger**: FrankenMoE-Loop v2 L11 ablation slot. Already on the
v2 roadmap; this entry just names the sub-experiments.

**Kill**: each E5* sub-experiment has its own pass criterion (match
or beat R2 at ≤+5% compute cost). Failure archives that variant.

### E6 Training data quality pass

**Description**: dedupe + filter + rebalance the training corpus.
Often higher-value than architecture at 80M scale. Produces a new
`.bin` file; no model changes.

**Tasks**:

1. **MinHash dedup** on dolma-10B at chunk-level. Target: remove 2–5%
   near-duplicates.
2. **Quality filter** based on perplexity via the existing OdinFlat
   checkpoint (perplexity > p99 → drop).
3. **Rebalance** domain mixture if domain mix shifted after dedup.

**Trigger**: at any compute slot; completes in ~1 CPU-day per pass.

**Gate**: resulting `.bin` produces ≥2% BPB improvement at matched
training tokens vs the current dolma-10B. Otherwise archive the
filtered binary and keep the current dataset.

**Kill date**: 2026-10-01.

## Execution Gantt (approximate)

```
Weeks after 2026-05-07:

W01-02  [v3.0] E4 + B1 + B4 (combined run)
W03-04  [v3.1] E1 distillation infra + 3 runs
W05     [v3.1] E2 iteration warmup (single run)
W06     [v3.1] E3 context curriculum (single run)
W07-09  [v3.2] A5 het-MoE (if triggered) OR A3 workspace
W10-12  [v3.2] second v3.2 pick (gate-dependent)

parallel:
W01-06  [E5] FrankenMoE-Loop v2 L11 ablations as they come up
W02-03  [E6] data quality pass (1-2 days compute, rest analysis)
```

## Cross-cutting gates

Before any phase v3 launch (0, 1, or 2):

- Sprint 3 full dolma-10B baseline must be green (or fail-and-understood).
- FrankenMoE-Loop v2 L9 must be landed for phase v3.2 Group A experiments.
- `--auto-eval` scorecard infrastructure must be green (it is).
- StabilityGuard + activation monitor must be active for all runs >500 steps.

## Risk-pooled budget

Total DDP compute estimate over 12 weeks (before any dolma-10B full
runs, which are separate):

| Phase | Training wall-clock | Engineering |
|---|---|---|
| v3.0 | ~10 hours | ~2 weeks |
| v3.1 | ~20 hours (3 distillation runs + E2 + E3) | ~4 weeks |
| v3.2 | ~15 hours per experiment × 2 = 30 hours | ~6–8 weeks |
| E5 parallel | ~8 hours | ~2 weeks (absorbed in v2 L11) |
| E6 | ~2 hours (+ CPU analysis) | ~3 days |
| **Total** | **~70 hours DDP** | **~12–14 weeks single-engineer** |

Compared to Sprint 3 dolma-10B full run at ~50 hours, this is a
modest compute budget; the bottleneck is engineering time.

## Decision cadence

- **Weekly check-in**: per-experiment status. Mark complete / kill /
  continue.
- **Post-phase review**: after v3.0, v3.1, v3.2 each, re-assess:
  which triggers are still live? Which kill dates should be
  revised? Does the sprint ordering still make sense?
- **Quarterly re-rank**: update `v3_speculative_directions_2026.md`
  and this plan with new literature, internal results, and shifted
  priorities. Delete entries past kill date with no trigger fired.

## Related docs

- `knowledge/architectures/v3_speculative_directions_2026.md` —
  speculative catalogue this plan operates against (cut-down
  version).
- `knowledge/architectures/a2_reversible_parcae_audit_2026.md` —
  A2 deep audit with detailed preconditions.
- `docs/superpowers/specs/2026-05-07-self-distillation-odinflat-odinhalo.md` —
  E1 full spec.
- `docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md` —
  FrankenMoE-Loop v2 (v3.2 A5 depends on post-v2 stability).
- `STATUS.md` — current training status including Sprint 3 smoke
  findings that motivated E4.
