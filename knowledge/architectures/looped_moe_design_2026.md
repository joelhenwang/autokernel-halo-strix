---
title: "Looped MoE Design — architectural choices and rationale"
domain: architectures
type: reference
status: design-only
tags: [moe, parcae, looped-transformer, routing, expert-sharing, iteration-schedule, frankenmoe, zaya1, odin-halo]
related:
  - parcae_stable_looped_models.md
  - looped_model_design_lessons.md
  - ../training/zaya1_8b_findings_2026.md
  - ../training/imu1_recipe_2026.md
  - ../../docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md
---

# Looped MoE Design — What MoE Means in a Parcae Loop

## Status

**Design-only.** No looped-MoE model trained yet. This doc captures the design space, the decisions we've made in the FrankenMoE-Loop v2 spec, and the rationale behind each choice, for future reference and for other halo-family models that may want to adopt MoE.

Companion spec: `docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md`.
Implementation trigger: FrankenMoE-Flat v1 L9 completion.

## Why "MoE × Parcae" is non-trivial

Parcae's core premise is **weight sharing across iterations** — the same layer weights are applied multiple times (3× in OdinHalo) to the same residual stream, with each iteration given progressively more context. MoE's core premise is **weight specialization per token** — different experts handle different token subsets.

Naively composing them raises questions the literature doesn't answer:

1. **Does a token visit the same expert each iteration, or does routing happen independently per iteration?**
2. **Are experts shared across iterations (like dense layer weights), or unique?**
3. **How does load balancing interact with 3× routing events per token per forward?**
4. **Does the router's EDA (Exponential Depth Averaging) cross iteration boundaries?**
5. **Does MoDA's cross-iteration depth-KV buffer need to know about expert identities?**

A cursory google / arXiv search finds no paper that combines Parcae-style weight-sharing looping with MoE. Perceiver IO and similar iteration-like architectures use cross-attention, not weight-sharing loops; Mixture-of-Depths is orthogonal (per-token depth selection in a non-looped backbone). Looping + MoE is genuinely open territory.

This doc picks a path through it. The choices below are **committed defaults** for FrankenMoE-Loop v2, with ablations specified to validate them.

## The three core questions and our answers

### Q1: When does routing happen?

Five options, evaluated for our constraints (fp16, HIP compile, 2× TB4 DDP, 75M active / 175M total budget):

| Scheme | Routing events / forward | Comment |
|--------|:------------------------:|---------|
| **R1** fresh per (layer, iter) | 18 (6 layers × 3 iters) | Load balancing over 18 units; most expressive; risky at small scale |
| **R2** sticky — route at iter 0, replay at iter 1-2 | 6 | **Chosen default** |
| R3 sticky + re-route on high entropy | variable | Data-dependent dispatch kills compile |
| R4 stage-biased (experts partitioned by iter) | 6 | Imposes prior; reduces effective expert pool per iter |
| R5 previous-iter expert as soft bias to current router | 12-18 | Adds state; complex debugging |

**R2 chosen because:**

- **Load balancing simplicity.** PID balancer from ZAYA1 operates on `p_{l,e}` = fraction of tokens routed to expert e at layer l. Under R2 this is the standard MoE setup, one balancing unit per layer. Under R1 we'd need to decide whether to balance per (layer, iter) — 18 units, each seeing 1/3 the tokens — or aggregate over iters, which breaks the isolation PID assumes.
- **Gradient stability.** Under R1, iteration 0's router might pick expert A and iteration 2's router might pick expert B for the same token (same layer, same iter). Gradients then flow through experts A **and** B's weights back to the shared layer weights below. Two different experts contributing to the same residual-stream update through shared preceding layers produces conflicting signals in that layer's gradient.
- **Parcae-philosophical.** "The same network, applied iteratively with more context" is exactly what sticky routing preserves for MoE. Each expert becomes a sub-function applied 3× per token.
- **Implicit compute-efficient top-k.** 1 expert × 3 iterations is strictly more expressive than top-3 MoE (three non-linear applications vs one weighted sum) at the same per-expert-call cost.
- **Generalises ZAYA1's router replay.** ZAYA1 caches expert indices from vLLM (rollout) to trainer to avoid gradient-corruption from engine-vs-trainer routing disagreements. R2 applies the same idea across iterations instead of across rollout/train — a natural extension.
- **Validated via L11 ablation.** The FrankenMoE-Loop v2 rollout explicitly includes an R1 + N3 (stickiness bonus) + N4 (iter-conditioned router input) run at full epoch. If R1 beats R2, we promote it in v3. If R1 matches R2, we have empirical validation of the simpler choice. If R1 loses, R2 is vindicated.

### Q2: Are experts shared across iterations?

| Scheme | Storage | Effective capacity | Verdict |
|--------|--------|--------------------|---------|
| **E1** shared across iters | small | small-unique, 3× effective compute | **Chosen default** |
| E2 unique per iter | 3× bigger | 3× unique | Defeats Parcae |
| E3 shared backbone + per-iter LoRA | small + ε | small-unique + tiny per-iter delta | **Documented fallback** |

**E1 chosen because:**

- Preserves the weight-sharing premise of Parcae literally. `self.experts[e]` is the same tensor at iter 0, 1, 2 — exactly as `self.shared_layers[l]` already is.
- Keeps parameter count at the 75M/175M target.
- Gradient signal per expert: 3× per token (strong, correlated). Mitigated by scaling NorMuon LRs by 1/√mean_recurrence on MoE groups at L3 onward.

**E3 kept as documented fallback** if L7 scorecard shows per-expert gradient-norm diverging across iterations (C6-style residual-norm drift by expert). LoRA adapters at rank ~16 add ~`E · mean_recurrence · (2 · rank · D) = 4 · 3 · 2 · 16 · 768 = 300K params per MoE layer`, trivial cost.

### Q3: Does routing benefit from knowing which iteration it's at?

- Under R2: **No**, routing only happens at iter 0, so iteration awareness is moot.
- Under R1 ablation: **Yes**. N4 = add `iteration_embed[iter_idx]` to the router's input after the down-projection. Cost: `mean_recurrence · R = 3 · 256 = 768 params` per MoE layer.

N4 is an R1-conditional feature. Not included in the shipping R2 path.

## Supporting architectural choices

### N1 — per-expert per-iteration output scaling γ_{e, i}

The most novel component of FrankenMoE-Loop. Each MoE layer adds a learned `Tensor[E, mean_recurrence, D]` initialised at ones and clamped ±4:

```
out_{e, i, token} = γ_{e, i} ⊙ Expert_e(x_{token})
```

**Why:** under R2 + E1, the same expert processes each of its tokens three times, but the residual stream differs materially across iterations (iter-0 input is token embeddings + bit of context; iter-2 input is near-converged representation). Different iterations benefit from different expert contribution strengths, but locking weight sharing (E1) denies this. γ_{e, i} is a cheap (`E · mean_recurrence · D ≈ 9 K per MoE layer`) trainable degree of freedom that lets the model attenuate or amplify each expert at each iteration without full weight specialisation.

**Stability:** the ±4 clamp mirrors `iter_scales.clamp(±4)` already used in OdinHalo — prevents fp16 overflow in long training if γ drifts. Parameter is never modified; only the forward-time value is bounded.

**Inductor compile:** trivial gather + elementwise multiply. Fuses cleanly.

### Per-iteration residual scaling (α_i, β_i)

Port from ZAYA1 §II-A-3, extended to per-iteration for Parcae:

```
h_next = α_{iter} ⊙ h + Layer(x) + β_{iter}
```

Per-site params: `2 · mean_recurrence · D ≈ 4.6 K`. Across 12 residual sites ≈ 55 K. Replaces / augments ZAYA1's per-block scaling which doesn't account for depth of the looped residual stream (3× passes through 12 sites = 36 residual additions per forward).

**Why per-iteration and not global:** residual norm grows differently at each iteration under Parcae. Iter 0 starts from token embeddings, iter 2 starts from near-converged representation. Pre-existing `iter_scales[iter_idx]` handles this at a coarse per-iteration scale; the new (α_i, β_i) adds per-site per-iteration granularity.

### Router: ZAYA1 MLP + EDA + PID balancing

Chosen unmodified from ZAYA1 Report §II-A-2. EDA scope is **within-iteration only** — under R2 sticky routing, the router fires only at iter 0, so "iter-crossing" EDA is a non-question.

PID balancing: separate AdamW on `b_{l,e}` with gradient `∇b_{l,e} = p_{l,e} − 1/E`. Gradient signal is the distance from uniform routing; AdamW's adaptive LR handles the varying scale across training phases.

**Router softmax in FP32** — added to the FP32 promotion list alongside the existing CE / QK-norm / RMSNorm / residual-addition promotions.

### 2.5-iteration scheduling (Sched-A + M3)

Not a routing question but an execution-schedule question; orthogonal to Q1-Q3.

**Sched-A:** iter 0 and iter 1 run all 6 layers; iter 2 runs only the two NoPE-GQA blocks (positions 2, 5). Total 14 layer-forwards vs baseline 18 = 2.33 equivalent iterations.

**M3:** at iter 2, the two GQA blocks' dense SwiGLU FFN uses half the inner width via `.narrow()` on `W_gate / W_up / W_down`. No separate weights, single narrow-view.

**Intuition:** iterations 0 and 1 do the heavy lifting (feature extraction, MoE refinement). Iteration 2 is "one more pass of global attention to integrate what the first two iterations produced." Iteration 2 doesn't need conv layers or MoE routing; it needs sequence-integration and that's what GQA provides.

**Annealing curriculum (critical):**
- Warmup (0-20%): full 3×6, no schedule — routing and γ converge under standard setup
- Anneal (20-40%): sigmoid gates λ_{l, i=2} initialised near 1, trained to drift toward 0 for scheduled-skip layers; M3 scale `m ∈ [0.5, 1.0]` trained to drift toward 0.5
- Production (40-100%): hard elision of skipped layers + `.narrow()` for half-width

**Hard-switch gate:** only transition layers with `σ(λ) < 0.05`; extended anneal for layers with higher residual gate. Prevents discontinuity spike at curriculum boundary.

**Sched-B as ablation (L8.8):** iter 2 runs layers [3, 4, 5] (back-half) instead of [2, 5]. Total 15 layer-forwards = 2.50 equivalent iterations. Tests "keep back-half refinement" vs Sched-A's "keep attention integration" inductive bias.

## What this means for MoDA depth attention

MoDA is unchanged. The depth-KV buffer is a list-of-dicts keyed by `layer_idx`, aggregated across prior iterations. Under Sched-A, at iter 2 only layers 2 and 5 run, and only layer 2 (a GQA block with MoDA) contributes new depth-KV entries. Downstream MoDA reads use `[buf[idx] for buf in depth_kv_buffer if idx in buf]` — already robust to missing entries per existing `_run_shared_block` code. No interaction bug possible.

MoE layers (positions 1, 4) are not MoDA producers (they're HyPEShortConvBlocks). MoE and MoDA operate on different block types; the composition is clean.

## What this means for fp16 stability

Looped MoE adds four new failure modes on top of OdinHalo's existing ones:

1. **Router softmax overflow** — mitigated by FP32 promotion
2. **Expert output magnitude drift across iterations** — mitigated by γ_{e, i} ± 4 clamp
3. **Cross-iteration residual norm drift per-expert-selection-pattern** — mitigated by per-iter (α_i, β_i) and the existing iter_scales clamp
4. **Anneal-phase gate non-convergence** — mitigated by hard-switch threshold + extended-anneal budget

New scorecard metrics (`halo_training/eval/moe_stats.py`):
- `gamma_ei_clipped_count` per MoE layer (rate of γ hitting ±4)
- `residual_norm_by_iter` (forensic for cross-iter drift)
- `skip_gate_lambda_by_layer` (anneal visibility)

## Generalisation to other halo models

This design is generic over the OdinHalo substrate. Other looped halo models (TyrHalo, BaldrHalo, GriffinHalo, JormungandrHalo, FenrirHalo, ChimeraHalo) can adopt the same R2 + E1 + γ_{e,i} + (α_i, β_i) + scheduled-skip stack with minor changes to:

- Which layer positions become MoE (depends on the halo variant's block composition)
- Schedule choice (Sched-A vs Sched-B depends on which layers are GQA-like integrators)
- Expert count (scales with hidden dim and target total-params)

The shared components go in `models/components/moe_ffn.py` (ZAYA1 router, PID balancer, ExpertPool, γ_{e,i} + dispatch) and `models/components/residual_scaling.py` (per-iter αβ) so any halo variant can pull them in.

## Creative additions — free throughput and quality wins

Five zero-risk (or nearly so) additions to the core design, each independent and opt-in. Elaborated in spec §3.11; the philosophy: **once you commit to R2 sticky routing + E1 shared experts + a fixed Parcae loop, certain invariants become exploitable.** Each addition exploits one.

### Throughput

**T7 — R2 permutation caching.** Under R2 the ScatterMoE dispatch permutation is identical across iterations. Cache at iter 0, reuse at iter 1-2. Zero-quality-impact single-digit throughput win. The underlying observation generalises: **any computation whose input depends only on `expert_assignments` (which is iteration-invariant under R2) can be cached across iterations.**

**T11 — Iter-2 KV reuse via the MoDA depth-KV buffer.** Under Sched-A, iter 2 runs only the NoPE-GQA blocks. MoDA already holds their iter-1 K, V. At iter 2, skip the K, V projections and reuse iter-1 KV; project only Q from iter-2's refined `h`. Savings are modest (~2 %) but **free** — no new buffers, no new params, one conditional in the GQA forward. Conceptually: **under Sched-A, iter 2 is pure attention-integration, and attention-integration benefits from mature keys more than fresh ones.** Refined query on mature keys is plausibly *better*, not just cheaper.

### Quality (training-only, zero inference cost)

**Q1 — Expert stochastic depth across iterations.** Probabilistically zero an expert's output at a random (expert, iter) slot during training. Free regularization; complements γ_{e,i} by preventing it from learning to compensate for deadweight iterations. Stochastic Depth at the expert × iteration granularity — a genuinely new locus.

**Q10 — Routing temperature annealing.** Standard MoE warmup trick (`τ: 2.0 → 1.0` over 10 % of training). Looped models stand to benefit more than non-looped because bad iter-0 routing decisions propagate through all 3 iterations under R2 — softer early routing buys exploration before commitment.

**Q3 — Iteration-varying RoPE base.** Progressive positional broadening: 10K → 30K → 100K at iters 0/1/2. Zero params (three precomputed `freqs_cis` tables indexed by iter). The thesis: Parcae's "each iteration adds context" is a semantic claim; matching it with a positional curriculum makes the inductive bias explicit.

### Pattern: invariants become wins

A design pattern emerges. The core commitments (R2 sticky + E1 shared) create invariants (permutation stable across iterations; each expert sees each token 3×; MoDA already caches iter-1 attention state). The creative additions each exploit one:

- T7 exploits permutation-invariance for compute.
- T11 exploits MoDA-already-caches-iter-1-KV for compute + potential quality upside.
- Q1 exploits same-expert-same-token-3× for regularization.
- Q10 exploits routing-happens-once for better early dynamics.
- Q3 exploits iterations-are-context-curriculum for positional bias.

**If we depart from R2+E1 (e.g., L11 ablation switches to R1), most of these stop working** — T7's permutation changes per iter, T11's KV-reuse breaks under independent routing, Q10 matters less if routing fires 3× per forward. That's a datapoint in favour of R2+E1 as the primary path: it's the design that *pays dividends beyond its own direct benefits*.

## Key unresolved questions (flagged for L11 and future)

- **Does R1 + N3 + N4 match or beat R2 at our scale?** Decided by L11 full-epoch ablation.
- **Does γ_{e, i} actually diverge meaningfully across iterations, or does it stay near 1 throughout training?** If stays near 1, N1 is overengineered; if diverges, it's doing real work. Scorecard reveals.
- **Does Sched-A beat Sched-B at 75M active, or vice versa?** Decided by L8.8.
- **Does the Parcae loop × MoE composition buy anything at 58M shared-layer unique?** Or is R2 + E1 + N1 basically "fancy dense model at MoE's FLOP cost"? L7 baseline delta vs OdinHalo answers this.
- **At what scale does this start paying off?** Hard to say without a scaling-law sweep. v3 candidate.
- **Which creative additions transfer to the R1 ablation path?** T7 and T11 presuppose R2; Q1, Q10, Q3 may transfer. If L11 promotes R1, re-evaluate the creative-additions set for that branch.
- **Does T11's "refined query, mature keys" hypothesis hold empirically?** L8.75 is the direct test; if reused-KV quality > fresh-KV quality, it's a publishable finding beyond just the compute savings.

## Related

- `docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md` — the implementation spec
- `knowledge/training/zaya1_8b_findings_2026.md` — parent synthesis (ZAYA1 findings that motivated this design)
- `knowledge/architectures/parcae_stable_looped_models.md` — Parcae stability reference
- `knowledge/architectures/looped_model_design_lessons.md` — 13 lessons from earlier halo looped models
- `knowledge/training/imu1_recipe_2026.md` — NorMuon + CWD recipe inherited unchanged
- `docs/research/zaya1_8b_technical_report/zaya1_8b_technical_report.md` — source report
