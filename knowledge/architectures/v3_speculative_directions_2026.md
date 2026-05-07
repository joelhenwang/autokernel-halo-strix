---
title: "v3 Speculative Directions — cut-down catalogue"
domain: architectures
type: research-menu
status: speculative
tags: [v3, speculative, research-directions, distillation, parcae, moe, training-dynamics, data, stability]
related:
  - a2_reversible_parcae_audit_2026.md
  - looped_moe_design_2026.md
  - ../training/zaya1_8b_findings_2026.md
  - cookbook.md
  - parcae_stable_looped_models.md
  - small_lm_arch_interventions_2026.md
  - ../../docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md
  - ../../docs/superpowers/specs/2026-05-07-self-distillation-odinflat-odinhalo.md
  - ../../docs/superpowers/plans/2026-05-07-v3-experiment-roadmap.md
---

# v3 Speculative Directions — cut-down catalogue (2026-05-07)

## Status

**Speculative research menu. Cut-down from 18 ideas to 14.** Each
entry has a trigger condition and a **kill date**: if the trigger
hasn't fired by the kill date, the entry is deleted, not archived.
This prevents the catalogue from accumulating aspirational cruft.

The original 1503-line version is preserved at
`archive/v3_speculative_directions_2026_original.md` for historical
reference. Deleted entries have a one-line obituary in **Appendix A**
so their reasoning isn't lost.

Operational execution is tracked separately in
[`docs/superpowers/plans/2026-05-07-v3-experiment-roadmap.md`](../../docs/superpowers/plans/2026-05-07-v3-experiment-roadmap.md).
This doc is the speculative menu; the roadmap is the commitment.

**When to touch this doc**:
- After FrankenMoE-Loop v2 L9 lands and we want to pick a v3 experiment.
- When a new paper triggers a "we should revisit that idea" moment.
- Quarterly, to delete expired entries and re-assess kill dates.

**When NOT to touch this doc**:
- During Sprint-N execution (commit to the sprint spec, not to speculation).
- To add a new idea without a trigger condition and a kill date.

## Reading guide

Each idea follows a uniform compact template:

```
### IDX. Name — one-line pitch

Trigger       One sentence
Kill date     YYYY-MM-DD
Value         /5    (quality / capability gain if it works)
Risk          /5    (higher = riskier, failure likelier)
Effort        /5    (higher = longer; 1 = days, 5 = months)

Mechanism     Short paragraph
Why novel     One sentence vs our KB
Hardware fit  One sentence on gfx1151 constraints
Standalone    One sentence on evaluation protocol
Composes      Pointer to §6 matrix
```

**Split scoring replaces the old ★★★★★ system.** A single composite
star conflated effort with expected value; the new scoring keeps
them separate.

- **Value**: 1 = small regularization, 5 = enables new capability.
- **Risk**: 1 = literature-proven, 5 = unprecedented and unclear it works at our scale.
- **Effort**: 1 = days, 3 = weeks, 5 = months.

Rule of thumb: **start experiments with (Value − Risk − Effort) ≥ −3**.

## Category layout

| Category | Theme | Ideas |
|---|---|---|
| **A** | Structural novelty (what the network *does*) | A2, A3, A5 |
| **B** | Training dynamics (how it *learns*) | B1, B2, B4 |
| **D** | Meta paradigm (training / data / inference) | D2, D3 |
| **E** | Actionable v3 entries (added 2026-05-07) | E1–E6 |

Categories A/B/D keep their historical IDs from the original
catalogue (so cross-references stay stable). Category E is new.

## Category A — Structural novelty

### A2. Reversible Parcae — invertible coupling layers for O(1) activation memory

- **Trigger**: memory becomes binding constraint on `mean_recurrence` or batch size; no cheaper lever available (gradient checkpointing, block_size reduction).
- **Kill date**: 2026-09-01.
- **Value**: 3/5 (real but smaller than originally pitched; see audit).
- **Risk**: 4/5 (fp16 non-associativity is a real engineering tax).
- **Effort**: 4/5 (4 weeks; see audit §7).

**Mechanism**. Coupling-layer bijection on the residual stream.
`u_next = u + F(v); v_next = v + G(u_next)`. Backward reconstructs
inputs from outputs, avoiding stored activations. Under honest fp16
accounting on gfx1151, option (c) fp32 inverse is the best
mitigation: ~40% memory saving, 0.77× throughput.

**Why novel**. Reversibility has never been combined with Parcae
weight-sharing iterations in the public literature.

**Hardware fit**. No MFMA so fp32 inverse is not disastrous. Wave32
+ no bf16 means algebraic invertibility breaks in fp16 without a
mitigation; audit §2 works this out.

**Standalone**. Prototype in fp32 first (week 1: `gradcheck` passes),
then fp16+fp32-inverse port (week 2), then OdinHalo integration
(week 3). Kill gates at weeks 2–4.

**Composes**. See §6. Notably ~ with B1 (noise injection breaks
invertibility unless noise is replayed on inverse — the audit caught
the original catalogue's erroneous `✓` claim). See
`knowledge/architectures/a2_reversible_parcae_audit_2026.md` for
full details.

---

### A3. Shared Latent Workspace — Global-Workspace-Theory primitive

- **Trigger**: scorecard long-range probe regresses, OR explicit research-novelty slack available.
- **Kill date**: 2026-09-01.
- **Value**: 3/5 at 80M (Perceiver-style latents historically don't help below ~1B).
- **Risk**: 3/5 (slot collapse + workspace domination are documented failure modes).
- **Effort**: 3/5 (~2–3 weeks).

**Mechanism**. Small learned tensor `W ∈ ℝ^{K×D}` (K=16–32 "concept
slots") shared across tokens, updated across Parcae iterations.
Tokens read from workspace via cross-attention each iteration;
workspace writes back via pooled token attention. Augments, doesn't
replace, self-attention.

**Why novel**. Iterative, sequence-scoped, augmentative workspace is
a unique combination (Perceiver is per-example, non-iterative,
replaces attention).

**Hardware fit**. Workspace is tiny; two extra cross-attentions per
iteration. Fp16-safe. Expected <5% throughput overhead.

**Standalone**. 40M proxy with K=16, 2K steps on babylm vs baseline.
Pass criteria: BPB parity, slot usage non-uniform (max/min < 10×),
long-range probe beats baseline +2%.

**Composes**. See §6. Clean with A2 (orthogonal tensors), B2 (both
produce iteration structure).

---

### A5. Heterogeneous-Capacity MoE (LZ77 routing deleted)

- **Trigger**: FrankenMoE-Flat v1 L9 validated (uniform-MoE baseline exists); specific hypothesis that tokens have systematically different compute needs.
- **Kill date**: 2026-09-15.
- **Value**: 3/5.
- **Risk**: 3/5 (heterogeneous dispatch adds complexity).
- **Effort**: 4/5 (~3 weeks).

**Mechanism**. Experts come in multiple sizes per MoE layer (tiny
ffn_inner=384, medium=1024, large=2048). Asymmetric PID balancing:
`p_e_target = size_e / sum(size_k)`. Router input is content +
router-confidence signal (NOT LZ77 — that was deleted as a weak
proxy; see Appendix A).

**Why novel**. Uniform-capacity is the default; heterogeneous at
training time is published only sparsely. The routing-signal change
(router confidence rather than external compressibility) is the
corrected version of the original A5 proposal.

**Hardware fit**. ScatterMoE dispatch handles variable-size output
via per-size-class dispatch calls. Fp16-safe.

**Standalone**. 4-expert uniform MoE baseline vs 4-expert het-MoE
with sizes [0.5×, 1×, 1×, 2×] at matched total compute. Pass if
BPB parity AND utilization roughly matches capacity.

**Composes**. See §6. Clean with D2 (both adaptive-compute
signals).

## Category B — Training dynamics

### B1. Hidden-state diffusion — iteration-as-denoising

- **Trigger**: v3.0 slot available; overfit symptoms observed.
- **Kill date**: 2026-07-01.
- **Value**: 2/5 (small regularizer).
- **Risk**: 2/5 (corrected schedule is conservative).
- **Effort**: 1/5 (~3 days).

**Mechanism**. Noise injection at start of each Parcae iteration
with decreasing schedule, annealed to zero in late training. The
original entry proposed σ=[0.3, 0.1, 0.0]; **corrected to σ=[0.08,
0.04, 0.0]** to respect fp16 dynamic range with z-loss active.
Anneal 70%→90% of training (wider window than the original 90%→100%).

**Why novel**. Diffusion-as-training-regularizer on iterative LMs
is unpublished; iteration schedule IS the noise schedule.

**Hardware fit**. One elementwise add per iteration. Fp16-safe. <1%
throughput.

**Standalone**. 40M proxy baseline vs B1 for 2K steps. Pass: BPB
≤ baseline, train-val gap narrower, no inference-time degradation
post-anneal.

**Composes**. See §6. Note: original matrix had A2×B1 as `✓`, which
was wrong — noise breaks algebraic invertibility unless replayed on
inverse, and replay defeats memory savings. Corrected to `~`.

---

### B2. Temporal-contrastive iteration learning

- **Trigger**: iteration-divergence analysis shows iter-1 ≈ iter-2 (underutilized iterations).
- **Kill date**: 2026-09-01.
- **Value**: 3/5.
- **Risk**: 3/5 (fragile — close positive-pair collapse is the failure mode it tries to prevent).
- **Effort**: 3/5 (~2 weeks).

**Mechanism**. InfoNCE across iteration axis: pull `iter_i` close
to `iter_{i+1}` for same token (positive); push away from same
position, different token, different example (negative). Small
projection head, discarded at inference.

**Why novel**. Iteration-axis contrastive has no LM analogue.

**Hardware fit**. Fp16-safe. InfoNCE at batch=16 per rank is
negatives-starved — **recommended to use DDP all-gather for negatives
from day one** (this is a correction vs the original entry's
"within-batch negatives" framing, which would be too weak at our
DDP config).

**Standalone**. 40M proxy, 2K steps, λ ∈ {0.1, 0.3}. Pass: BPB
within 1% of baseline AND per-iteration divergence measurably
increases.

**Composes**. See §6.

---

### B4. Entropy-conservation loss

- **Trigger**: v3.0 slot; scorecard shows `residual_norm_by_iter` not decreasing across iterations.
- **Kill date**: 2026-07-01.
- **Value**: 2/5.
- **Risk**: 3/5 (assumes "entropy should collapse across iterations" — unvalidated).
- **Effort**: 1/5 (~2 days).

**Mechanism**. Hinge penalty on per-iteration variance increase
beyond a tolerance. **Corrected** from the original entry: `λ=0.02`
not `0.05`, hinge tolerance `+20%` (not strict monotonic), fp32
variance computation.

**Why novel**. Entropy regularization on iteration axis in looped
LMs.

**Hardware fit**. Trivial; variance reduction per iteration.

**Standalone**. OdinHalo wikitext run with/without B4. Pass if
entropy trajectory decreases AND BPB ≤ baseline AND downstream probes ≥ baseline.

**Composes**. See §6. Clean with B1 and E4.

## Category D — Meta paradigm

### D2. Adaptive data-mixture curriculum

- **Trigger**: multi-domain training (dolma mix) shows uneven per-domain progress.
- **Kill date**: 2026-09-15.
- **Value**: 3/5.
- **Risk**: 2/5.
- **Effort**: 3/5 (~2 weeks).

**Mechanism**. Per-domain loss-slope tracking every K steps; upweight
stagnant domains, downweight fast-descending ones. Reactive version
of DoReMi.

**Why novel**. CLIMB (in KB) is static; real-time adaptive is
undocumented in our stack.

**Hardware fit**. Dataloader-only; no model change.

**Standalone**. Fixed CLIMB mix vs D2 on dolma-mix. Pass if total
BPB ≤ baseline +1% AND per-domain variance reduced.

**Composes**. See §6. Clean with A5 (both respond to per-domain
signal) and with E6 (post-filter rebalance).

---

### D3. Gradient-norm-gated example dropping

- **Trigger**: late-training memorization symptoms.
- **Kill date**: 2026-10-01.
- **Value**: 2/5.
- **Risk**: 3/5 (risks dropping genuinely hard examples).
- **Effort**: 3/5 (~1–2 weeks).

**Mechanism**. Per-example gradient norm tracked; examples beyond
2σ of moving median are downweighted (not dropped). Early training
preserves all; late training filters.

**Why novel**. Per-example dynamic weighting based on grad-norm is
not in our KB.

**Hardware fit**. Requires per-example gradient via microbatching
or hooks. Modest overhead.

**Standalone**. Baseline vs D3 at 2σ threshold. Pass if BPB ≤
baseline AND memorization probe accuracy reduced.

**Composes**. See §6. Clean with D2.

## Category E — Added 2026-05-07

These entries were added to replace ideas deleted from the original
catalogue. Each is grounded in an observation from STATUS.md or a
practical stack lever, not pure speculation.

### E1. OdinFlat → OdinHalo self-distillation

- **Trigger**: **ACTIVE** — teacher checkpoint (`odin-flat-wikitext-ddp/step_1869.pt`, BPB 1.79) and student baseline (`odin-halo-wikitext-ddp/step_1869.pt`, BPB 1.88) both exist with a 5.1% BPB gap that distillation has high a-priori chance of closing.
- **Kill date**: 2026-08-01.
- **Value**: 5/5 (closes a measurable quality gap; highest a-priori ROI of anything in this doc).
- **Risk**: 2/5 (teacher and student are same tokenizer, same data, same block size — maximally de-risked).
- **Effort**: 3/5 (~2 weeks, fully specified).

**Mechanism**. Teacher (OdinFlat 122M) runs frozen in fp16
inference-mode; student (OdinHalo 157M effective) trains with
KL+CE mix (V2, α=0.5, T=2.0). Chunked KL loss avoids materializing
`[N, V]` logits twice.

**Why novel**. Not novel as a technique (standard KD) but novel to
our stack. We already have a matched teacher-student pair, which
most KD research doesn't.

**Hardware fit**. Teacher fp16 inference + student fp16 training
fits in ~8 GB per rank. Expected ~1.65× step time; ~24K tok/s
aggregate on DDP.

**Standalone**. Three variants (V1 pure KL, V2 KL+CE, V3 top-k KL)
compared on wikitext-103 at matched 1-epoch budget. Pass gate: ≥1
variant achieves BPB ≤ 1.835 (50% gap closure).

**Full spec**:
[`docs/superpowers/specs/2026-05-07-self-distillation-odinflat-odinhalo.md`](../../docs/superpowers/specs/2026-05-07-self-distillation-odinflat-odinhalo.md).

**Composes**. See §6. Does NOT compose with E2 (iteration warmup
changes student compute profile, breaks teacher/student parity).

---

### E2. Parcae iteration-warmup curriculum

- **Trigger**: E4 deploys but `shared_layers.5` maxabs still grows; OR scorecard shows iter-2 contributes little to final loss.
- **Kill date**: 2026-08-01.
- **Value**: 3/5.
- **Risk**: 3/5 (curriculum switch could cause loss spike at transition).
- **Effort**: 2/5 (~1 week).

**Mechanism**. First N% of training uses `mean_recurrence - 1`; then
the final iteration is introduced with `iter_scales[last] = 0` bias.
The existing iterations have stabilized before the additional
iteration accumulates activation load.

**Why novel**. Curriculum on Parcae iteration count is not in our
KB. Sprint 3 smoke failure analysis is the specific motivation.

**Hardware fit**. `set_mean_recurrence()` already supported by
OdinHalo; may trigger one recompile at transition.

**Standalone**. 1-epoch wikitext with warmup (20% at recurrence=2)
vs without. Pass if BPB parity AND maxabs growth bounded below 500.

**Composes**. See §6. Conflicts with E1 (changes student forward).

---

### E3. Progressive context-length schedule

- **Trigger**: orthogonal; run in any v3.1 compute slot.
- **Kill date**: 2026-08-15.
- **Value**: 3/5 (literature-proven at scale).
- **Risk**: 2/5 (well-precedented; block-size switch mid-training is standard).
- **Effort**: 2/5 (~1 week).

**Mechanism**. Start at `block_size=256`, transition to 512 or 1024
at a fixed step. Dataloader supports variable windowing via existing
streaming binary format.

**Why novel**. Not novel at scale; novel to our stack. Direct
literature precedent (LLaMA 2, Qwen).

**Hardware fit**. Block change alters memory + compile cache;
expect one recompile at transition.

**Standalone**. Fixed-256 vs 256→512 at step 1000. Pass if ≥2% BPB
improvement at matched tokens.

**Composes**. See §6. Clean with E1 distillation.

---

### E4. Activation-growth damping regularizer

- **Trigger**: **ACTIVE** — Sprint 3 smoke failure on 2026-05-07 showed `shared_layers.5` maxabs growth 38 → 9117 over 500 steps, leading to GradScaler collapse.
- **Kill date**: 2026-06-15.
- **Value**: 4/5 (directly addresses a production blocker).
- **Risk**: 2/5 (mild L2 penalty; smooth, not discontinuous).
- **Effort**: 1/5 (~3 days infra).

**Mechanism**. L2 penalty on per-layer `max(0, maxabs − target)^2`
where `target` is the 95th percentile of maxabs over the warmup
phase. Adds a smooth correctional signal via the existing activation
monitor. Preferable to blunt clamping because it preserves gradient
signal.

**Why novel**. Turning activation-monitor telemetry into a
regularization signal is not documented. Complements, rather than
replaces, the existing `iter_scales.clamp(-4, 4)` approach.

**Hardware fit**. One extra reduction per monitored layer per
regularizer step (typically every 50 steps). <1% throughput
overhead.

**Standalone**. Sprint 3 full dolma-10B run with E4 active. Pass
if training survives ≥2000 steps with no StabilityGuard rollback
AND BPB trajectory matches a "hypothetical baseline" extrapolation.

**Composes**. See §6. Clean with everything — pure training-time
penalty with no interaction with forward path.

---

### E5. Router / expert variants beyond R2 sticky

- **Trigger**: FrankenMoE-Loop v2 L11 ablation slot.
- **Kill date**: 2026-09-01.
- **Value**: 3/5 (explores the R2 choice's robustness).
- **Risk**: 2/5 (each sub-experiment is small).
- **Effort**: 4/5 (four sub-experiments, ~1 week each).

**Mechanism**. A family of four ablations against R2 sticky:

| Sub-ID | Variant | Hypothesis |
|---|---|---|
| E5a | Sinkhorn routing | Better load balance at small scale than PID |
| E5b | Expert-choice routing | Automatic balance by construction |
| E5c | Hash routing | Zero-overhead routing floor |
| E5d | Per-token capacity routing | Dynamic compute via halt |

**Why novel**. R2 is a specific design choice for looped MoE; E5
validates that choice or identifies a better alternative.

**Hardware fit**. Each variant is small-diff from current ScatterMoE
dispatch. Fp16-safe.

**Standalone**. Each E5* runs a 1-epoch wikitext scorecard vs R2
baseline. Pass if variant matches or beats R2 at ≤+5% compute cost.

**Composes**. Each sub-experiment is individual; they don't compose
with each other (mutually exclusive router choices).

---

### E6. Training data quality pass

- **Trigger**: orthogonal; runs whenever compute slot available.
- **Kill date**: 2026-10-01.
- **Value**: 4/5 (at 80M, data quality often outperforms architecture).
- **Risk**: 2/5 (CPU-only; resulting dataset is opt-in).
- **Effort**: 2/5 (~1–2 CPU-days + 1 day analysis).

**Mechanism**. Three-pass:

1. MinHash near-duplicate detection at chunk level on dolma-10B.
2. Perplexity filter using OdinFlat checkpoint (drop chunks with
   ppl > p99).
3. Rebalance domain mix if dedup shifted it.

Produces a new `.bin` file; no model change.

**Why novel**. Not novel conceptually; novel to our stack (current
dolma-10B was not dedup'd or quality-filtered against one of our
own models).

**Hardware fit**. CPU-only; no GPU implications.

**Standalone**. Dedup+filtered vs original dolma-10B for 1-epoch
OdinHalo run. Pass if BPB improves ≥2% at matched tokens.

**Composes**. Clean with D2 (data-mix adaptation runs on top).

## 6. Compatibility matrix (conjectured)

Legend:

- ✓ composes cleanly (benefits plausibly add)
- · neutral (independent mechanisms)
- ~ partial conflict (needs care; may dilute gains)
- ✗ conflicts (incompatible by design)

|     | A2 | A3 | A5 | B1 | B2 | B4 | D2 | D3 | E1 | E2 | E3 | E4 | E5 | E6 |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **A2** Reversible Parcae     | — | ✓ | ✓ | ~¹ | ✓ | ✓ | · | · | · | ✗² | · | · | · | · |
| **A3** Shared Workspace      | ✓ | — | · | ✓ | ✓ | ✓ | · | · | · | · | ✓ | · | · | · |
| **A5** Het-MoE               | ✓ | · | — | ✓ | · | ✓ | ✓ | · | · | · | · | · | ~³ | ✓ |
| **B1** Hidden-state diffusion| ~¹ | ✓ | ✓ | — | ✓ | ✓ | · | · | ~⁴ | · | · | ✓ | · | · |
| **B2** Temporal-contrastive  | ✓ | ✓ | · | ✓ | — | ✓ | · | · | · | · | · | · | · | · |
| **B4** Entropy conservation  | ✓ | ✓ | ✓ | ✓ | ✓ | — | · | · | · | · | · | ✓ | · | · |
| **D2** Adaptive data mix     | · | · | ✓ | · | · | · | — | ✓ | · | · | ✓ | · | · | ✓ |
| **D3** Grad-norm gating      | · | · | · | · | · | · | ✓ | — | · | · | · | · | · | ✓ |
| **E1** Self-distillation     | · | · | · | ~⁴ | · | · | · | · | — | ✗⁵ | ✓ | ✓ | · | ✓ |
| **E2** Iter warmup           | ✗² | · | · | · | · | · | · | · | ✗⁵ | — | ~⁶ | ✓ | · | · |
| **E3** Context curriculum    | · | ✓ | · | · | · | · | ✓ | · | ✓ | ~⁶ | — | ✓ | · | ✓ |
| **E4** Activation damping    | · | · | · | ✓ | · | ✓ | · | · | ✓ | ✓ | ✓ | — | ✓ | · |
| **E5** Router variants       | · | · | ~³ | · | · | · | · | · | · | · | · | ✓ | — | · |
| **E6** Data quality pass     | · | · | ✓ | · | · | · | ✓ | ✓ | ✓ | · | ✓ | · | · | — |

### Footnotes

1. **A2 × B1 `~`** — Noise injection at forward breaks the algebraic
   invertibility guarantee of the coupling layer unless the same noise
   is replayed on inverse. Replaying requires storing the noise
   tensor per layer, which defeats the memory-saving motivation.
   **Correction from original catalogue**, which wrongly listed this
   as `✓`. To combine: store noise seeds (not tensors) and regenerate
   deterministically on inverse.
2. **A2 × E2 `✗`** — Iteration warmup changes `mean_recurrence`
   mid-training, which invalidates the coupling block's forward
   graph structure that A2 depends on.
3. **A5 × E5 `~`** — Heterogeneous-capacity dispatch and
   non-default routers both modify the ScatterMoE dispatch path; a
   combined experiment requires a shared implementation. Not
   prohibited, just more engineering.
4. **B1 × E1 `~`** — Noise injection during distillation would
   fight the teacher signal (noise makes student less able to match
   teacher logits). If combined: enable B1 only after distillation
   signal has been absorbed (late-training phase only).
5. **E1 × E2 `✗`** — Iteration-warmup curriculum breaks teacher-
   student forward parity. Student's iteration count changes mid-run,
   student's logits diverge from teacher's expected distribution
   temporarily. Documented in the E1 spec.
6. **E2 × E3 `~`** — Both curricula; can compose if E2's iteration-
   warmup finishes before E3's context expansion starts. Ordering
   matters.

### Strongest compositions

- **E4 + B1 + B4** — all three are training-dynamic regularizers
  orthogonal to each other; can run in a single v3.0 run without
  interference. **Recommended v3.0 launch**.
- **E1 + E3** — distillation plus context curriculum; teacher
  generalizes well across context sizes because it was trained on
  the same data.
- **A5 + D2 + E6** — all three respond to per-domain / per-token
  signals; natural stack for a future MoE + data experiment.

## 7. "If we could only do one" ranking

Re-ranked with split scoring. Composite shown as `V − R − E` (higher
= more attractive to start). Experiments with trigger already active
are **bolded**.

| Rank | ID | Name | V | R | E | Score |
|-----:|----|------|--:|--:|--:|------:|
| 1 | **E4** | Activation-growth damping | 4 | 2 | 1 | **+1** |
| 2 | **E1** | OdinFlat → OdinHalo distillation | 5 | 2 | 3 | **0** |
| 3 | E6 | Training data quality pass | 4 | 2 | 2 | 0 |
| 4 | B1 | Hidden-state diffusion (corrected) | 2 | 2 | 1 | −1 |
| 5 | E3 | Progressive context schedule | 3 | 2 | 2 | −1 |
| 6 | B4 | Entropy conservation (corrected) | 2 | 3 | 1 | −2 |
| 7 | D2 | Adaptive data mix | 3 | 2 | 3 | −2 |
| 8 | E2 | Iteration warmup curriculum | 3 | 3 | 2 | −2 |
| 9 | A3 | Shared workspace | 3 | 3 | 3 | −3 |
| 10 | A5 | Het-MoE (LZ77 deleted) | 3 | 3 | 4 | −4 |
| 11 | B2 | Temporal-contrastive | 3 | 3 | 3 | −3 |
| 12 | E5 | Router variants | 3 | 2 | 4 | −3 |
| 13 | D3 | Grad-norm gating | 2 | 3 | 3 | −4 |
| 14 | A2 | Reversible Parcae | 3 | 4 | 4 | −5 |

**Recommended starting order (matches the roadmap plan)**:

1. E4 immediately — trigger active, highest composite score.
2. E1 next (parallel or sequential) — highest value, cleanest spec.
3. E6 in any compute slot — orthogonal, easy wins likely.
4. B1 + B4 as a v3.0 composition.
5. E3 once v3.1 slot opens.
6. Everything else gated by explicit trigger.

## 8. Roadmap summary

For operational sequencing, decision gates, and compute budgets see
[`docs/superpowers/plans/2026-05-07-v3-experiment-roadmap.md`](../../docs/superpowers/plans/2026-05-07-v3-experiment-roadmap.md).

One-page summary:

- **Phase v3.0** (2–3 weeks): E4 + B1 + B4 as a combined run. Low risk.
- **Phase v3.1** (3–4 weeks): E1 distillation; E2 + E3 as single-issue runs.
- **Phase v3.2** (4–6 weeks): A5 het-MoE → A3 workspace → A2 reversible (gated by audit). B2 if iteration-divergence evidence appears.
- **Parallel tracks**: E5 (inside FrankenMoE-Loop v2 L11) and E6 (data quality) run at any slot.

## 9. Cross-cutting open questions

1. **Scale-appropriateness**. Many retained ideas are motivated by
   100M+ results. At 80M our evaluation protocols must be long enough
   (≥5K steps on real data, not 2K on babylm) to discriminate signal
   from noise.
2. **Fp16 dynamic-range interaction**. Any new op must coexist with
   z-loss + attn-softcap + iter_scales clamp + activation monitor +
   StabilityGuard scaler-floor (the current stack). New entries
   should call out their fp16 risk explicitly.
3. **Compile interaction**. `max-autotune-no-cudagraphs` is the
   production mode for looped models. New ops should be Inductor-
   fusable or live behind `@torch.compiler.disable` with minimal
   graph breakage.
4. **Reproducibility**. Several v3 ideas are novel enough that no
   external replication exists. First-mover means we document results
   carefully.
5. **Prioritization cadence**. Quarterly re-rank. Entries past
   kill_date without fired triggers get deleted.

## Appendix A — deleted entries (2026-05-07)

Obituaries for the 10 entries removed in this rewrite. Preserved for
the reasoning, so we don't rediscover the same dead ends.

- **A1 Complex MoE**. Phase gradient via magnitude reduction is a
  known-weak signal in complex nets; our top-1 routing invariant
  makes it doubly speculative. Revisit only if top-k≥2 becomes
  viable and we observe "expert compromise" as a specific failure mode.
- **A4 Path Superposition**. Ensemble-by-another-name; at 80M with
  P=2, reliably worse than a matched-FLOPs single-path model. High
  compute cost, no concrete motivation at our scale.
- **B3 Forward-Forward**. Memory-saving aux for a problem we don't
  have. Hinton's FF has not demonstrated competitive LM training in
  replicated work; gradient-free aux is orthogonal to our current
  stack's needs.
- **B5 Momentum Teacher (DINO-style)**. 1.8× training time is
  brutal for throughput-bound training; "teacher every N steps"
  makes teacher stale. If we want distillation, E1 is the direct
  path (teacher trained offline).
- **C1 Mycelial Graph**. Replaces established routing with a
  graph-walk mechanism; requires rebuilding ScatterMoE dispatch +
  PID balancing for node-visit-balancing. Too much scaffolding
  disturbance for research novelty at our scale.
- **C2 Cross-Example Attention**. Ahead-of-time at our batch=16 per
  rank scale; insufficient diversity for meaningful cross-attention,
  and DDP all-gather variant is impractical on TB4.
- **C3 Self-CoT Recycling**. Iter-2 output lives in a different space
  than token embeddings (RMSNorm'd vs raw); would need a projection.
  Inference-time benefit unclear without a test-time-compute
  architecture commitment.
- **C4 Hypernet-Generated Weights**. "Anti-Parcae" framing undoes
  the regularization that makes weight-sharing work. `bmm` per-batch
  is throughput-hostile.
- **C5 Lyapunov-Stable Iteration (DEQ / Hopfield)**. Pure theoretical
  aesthetics. DEQ requires implicit differentiation infrastructure
  we don't have.
- **D1 Static Prefill Cache**. Inference-time optimization for a
  deployment scenario we haven't committed to; not a research
  direction.

Additionally, the LZ77-based routing signal was deleted from **A5**:
compressibility is token-sequence information, not token-difficulty.
A token inside a repeated phrase can be *hard* because the model
must recognize the pattern. The heterogeneous-capacity part of A5
survives; the routing signal is replaced by router-confidence.

## 10. Related docs

- `knowledge/architectures/archive/v3_speculative_directions_2026_original.md` — 1503-line full original.
- `knowledge/architectures/a2_reversible_parcae_audit_2026.md` — deep audit referenced by A2 entry.
- `docs/superpowers/specs/2026-05-07-self-distillation-odinflat-odinhalo.md` — E1 full spec.
- `docs/superpowers/plans/2026-05-07-v3-experiment-roadmap.md` — operational plan for v3.0–v3.2.
- `docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md` — FrankenMoE-Loop v2 (A5/E5 depend on v2 validation).
- `knowledge/architectures/looped_moe_design_2026.md` — generalizable findings on looped MoE.
- `knowledge/architectures/cookbook.md` — primitive library.
- `knowledge/architectures/parcae_stable_looped_models.md` — Parcae reference.
- `knowledge/training/zaya1_8b_findings_2026.md` — ZAYA1 applied findings.
- `knowledge/training/fp16_stability_gfx1151.md` — fp16 stability stack.
- `STATUS.md` — current training status; source of the E4 motivation (Sprint 3 smoke failure) and E1 baseline numbers.
