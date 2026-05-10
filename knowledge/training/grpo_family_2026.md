---
title: "GRPO Family (2025-2026): F-GRPO, Scaf-GRPO, GRPO-SG, f-GRPO, Apriel-Reasoner"
domain: training
type: reference
status: active
tags: [rlvr, grpo, reasoning, post-training, small-lm, alignment, f-grpo, scaf-grpo]
related:
  - instruct_alignment_techniques_2025_2026.md
  - alignment_implementation_details.md
  - ../../docs/research/broad-research-synthesis-2026-05-06.md
---

# GRPO Family (2025-2026) — RLVR Variants for Reasoning Training

## Why this matters

Post-DeepSeek-R1 (Jan 2025), Group Relative Policy Optimization (GRPO) became
the standard for Reinforcement Learning with Verifiable Rewards (RLVR) in LLM
post-training. A wave of variants released Oct 2025 → April 2026 each fix
specific failure modes. This doc consolidates them for when we build our
post-training pipeline.

Scale of the action: ALL of the below are evaluated at 7B+. Transfer to our
58-122M scale needs validation — but GRPO's failure modes (zero advantages,
learning cliffs, rare-trajectory neglect) intensify at small scale, so the
fixes likely matter more for us.

## Vanilla GRPO (DeepSeek-R1, Jan 2025) — baseline

For each prompt:
1. Sample G rollouts from current policy
2. Compute rewards via verifier
3. Advantage = `(r_i - mean(r)) / std(r)`
4. Update with clipped policy gradient (PPO-style) using these advantages

**Problem areas (from newer papers):**
- **Small groups miss rare-correct trajectories** (F-GRPO)
- **Zero advantage collapse** when all rollouts fail on hard problems (Scaf-GRPO)
- **Sharp updates hurt generalization** (GRPO-SG)
- **Unified framing with preference alignment** underexplored (f-GRPO)

## F-GRPO (Focal-loss-inspired) — arXiv:2602.06717 (Feb 2026)

### Problem

Large group sizes are computationally infeasible. Small groups bias learning
toward already-likely trajectories. Formal result: probability that updates
miss rare-correct modes is non-monotonic in group size. Unsampled-correct mass
can shrink even as total correct mass grows.

### Fix

Difficulty-aware advantage scaling: down-weight updates on high-success
prompts. Directly inspired by Focal loss (Lin et al. 2017, for class imbalance).

```
scale(prompt) = (1 - success_rate)^gamma   # Focal-like
advantage_i  *= scale(prompt)
```

### Drop-in compatibility

Works with **GRPO, DAPO, CISPO** — any group-relative RLVR algorithm.

### Results (Qwen2.5-7B)

| Base | pass@256 baseline | + F-GRPO |
|------|------------------:|---------:|
| GRPO | 64.1 | **70.3** |
| DAPO | 69.3 | **72.5** |
| CISPO | 73.2 | **76.8** |

**No extra compute or group-size increase.** Also preserves/improves pass@1.

## Scaf-GRPO (Scaffolded) — arXiv:2510.19807 (Oct 2025)

### Problem: the "learning cliff"

When problems are beyond the model's current capability, all rollouts fail,
reward is always 0, advantage collapses to 0, no gradient signal. Progress
stalls on the hardest problems indefinitely.

### Fix: tiered in-prompt hints when plateau detected

1. Monitor per-prompt success rate over recent steps
2. If success rate is 0 for N consecutive evaluations → inject a hint
3. Hints graduate from **abstract concepts → concrete steps**
4. Once the model solves with hints, back off to unscaffolded

### Results (Qwen2.5-Math-7B, AIME24)

**+44.3% relative pass@1** vs vanilla GRPO.

### For small-LM reasoning

**Essential.** Small models hit the learning cliff faster and more often.
Scaffolding is not optional — it's the difference between "learns" and
"stalls permanently on anything >easy".

## GRPO-SG (Sharpness-Guided) — arXiv:2511.00066 (Oct 2025)

### Problem

RLVR has limited control over generalization. High sharpness (large gradient
norms) → poor out-of-distribution behavior.

### Fix

Robustness-based generalization bound: generalization loss ≤ empirical loss
+ sharpness surrogate (gradient norm). Minimize both.

Implementation: **token-weighted GRPO** — downweight tokens that likely cause
overly large gradients. Smoother gradient-norm trajectories.

### Results

Consistent improvements over GRPO across math reasoning, logic puzzles,
tool-augmented QA. Cleaner gradient-norm trajectories.

### For Odin

Useful as a **generalization upgrade** once we have vanilla RLVR working. Not
first-priority because it requires having a GRPO baseline to improve.

## f-GRPO (divergence-based) — arXiv:2602.05946 (Feb 2026)

### Theoretical contribution

**Unified framework:** preference alignment objectives act as divergence
estimators between aligned and unaligned response distributions. The same
framing extends to RLVR.

### Concrete algorithms

1. **f-GRPO** — on-policy, generalizes GRPO to arbitrary f-divergences
2. **f-HAL** — hybrid on+off policy loss for alignment tasks

### Why this matters

f-GRPO unifies RLVR (verifiable rewards) + preference alignment (pairwise
preferences) under one objective. When building Odin's post-training pipeline,
choosing f-GRPO means **one implementation covers both reasoning RL and
alignment**.

### Results

Validated on RLVR (math reasoning) and preference alignment (safety). Superior
performance and flexibility vs current methods.

## Apriel-Reasoner — arXiv:2604.02007 (April 2026)

Not a GRPO variant per se — it's a **full multi-domain RLVR recipe** at 15B
scale. Most-recent large-scale open-weight reasoning training reference.

### Novel contributions

1. **Adaptive domain sampling:** Domains have heterogeneous rollout lengths
   (math: long, function-calling: short). Adaptive sampling preserves target
   ratios despite this.
2. **Difficulty-aware length penalty:** Longer reasoning for hard problems,
   shorter for easy. **No additional training overhead.**

### Domains covered

Math, code generation, instruction following, logical puzzles, function calling.
All public datasets.

### Deployment-critical findings

- Trained with **16K token output budget**, generalizes to 32K at inference
- Produces **30-50% shorter reasoning traces** than baseline
- Matches strong open-weight models at **lower token cost**
- Pareto frontier of accuracy vs token budget improved

### For Odin (small-scale port)

- Adopt **adaptive domain sampling** when we do multi-domain RLVR
- Adopt **difficulty-aware length penalty** — free win, no cost
- Full 5-domain setup is overkill at 122M; start with math + code

## ZAYA1-8B validated RL cascade (May 2026)

Zyphra's ZAYA1-8B technical report ships a **concrete end-to-end RL cocktail**
used to train a production 760M-active reasoning MoE. Recipe below; full
context + Odin applicability in
[zaya1_8b_findings_2026.md](zaya1_8b_findings_2026.md).

### The algorithmic spine

| Component | Value | Rationale |
|-----------|-------|-----------|
| Trust region | **DPPO Binary-TV**, δ=0.1 | Replaces PPO ratio-clip; binary mask on tokens where divergence > δ. Tune δ against the largest value that still constrains reward-growth vs an unconstrained baseline |
| Loss aggregation | **Dr-GRPO SMTSN** (sequence-mean over token-sum-norm) | Removes GRPO's implicit length normalization bias toward long responses |
| Advantage | **MaxRL**: `Â_i = (r_i − r̄) / r̄` | Divide by per-prompt **mean**, not stddev. Unbiased for truncated-MLE objective, stronger gradient on hard prompts |
| KL in reward | **None** | See "Length-bias trap" below |
| Optimizer | **Momentum-free Muon** on matrix weights; AdamW on embeds + LM head | Each RL update depends only on the current rollout batch; no cross-batch momentum averaging incompatible gradient directions. Also saves first-moment buffer memory |
| Async | PipelineRL, 2–5× rollout workers/trainer, weight sync every 2 iters, 2-update staleness bound | Standard async setup |
| Router replay | vLLM writes per-token/layer expert indices; trainer pins to same assignments | **MoE-only** — not applicable to dense Odin |

### Length-bias trap (negative result worth memorising)

Combining **signed K1 log-ratio KL in reward + sequence-level aggregation +
broadcast to all tokens + PipelineRL stale mixed-policy rollouts** creates a
**length-dependent positive reward offset** unrelated to task quality. Long
completions accumulate more negative `l_t` terms; subtracted from reward they
become a length bonus.

**Mitigation used in production:** remove KL-in-reward entirely; trust region
is DPPO Binary-TV alone. **Adopt the same default** when we build RL.

Alternatives Zyphra list but don't deploy:
- Chunk-local signed-log-ratio isolation (aggregate per-chunk not whole-seq).
- Staleness rescaling by `g(Δ_c) = max(1, Δ_c)` (Bartoldson 2026 first-order
  EMA-reference approximation).

### Reward gating via content canaries

- **Streaming LZ77** (zlib wbits=−10, level-1, Z_SYNC_FLUSH) — flag rollouts
  with any chunk `r_c < 0.05`; **zero task reward before advantage computation**,
  even if verifier accepts the answer.
- **Rare-token fraction** (top-10 % of tokenizer ID range) — logged per batch.

### Curriculum: RLVE-Gym Thompson/IRT

ZAYA1 uses **Thompson sampling over a Gaussian-prior pool of logistic IRT
parameters** `(μ, s)` to calibrate each of 400 verifiable environments to a
0.5 solve rate — the maximum Fisher information point of the logistic model —
with ε-greedy around target and allowed regressions. Weighted environment
sampling favours the least-sampled envs. This is a generic RLVR pattern and
should be the default when we do multi-environment RLVR.

### Where this sits relative to the variants above

ZAYA1's cascade is GRPO-family (Dr-GRPO + MaxRL advantage) with **DPPO
Binary-TV in place of PPO clip**. It does not overlap with F-GRPO / Scaf-GRPO
/ GRPO-SG / f-GRPO / Apriel — those are orthogonal ideas (focal weighting,
scaffolding hints, sharpness guidance, divergence unification, difficulty-
aware length penalty) and could be composed on top of this spine.

For Odin post-training: **ZAYA1 spine + F-GRPO focal advantage + Scaf-GRPO
scaffolding** is a reasonable "pick the best of each" stack when we get there.

## Recommended stack for Odin post-training


Given zero existing infra, build this in order:

### Phase 1: SFT foundation (no RL)
- Data: SmolTalk2 or similar public instruction dataset
- Next-token loss masked on assistant turns only
- 1-2 epochs at LR 2e-5

### Phase 2: Preference alignment (single-stage)
- **ORPO** — simplest, no reference model, no separate SFT phase
- Alternative: **APO + model merging** (SmolLM3 recipe)
- Data: public preference mixtures (Tulu 3)

### Phase 3: RLVR for reasoning
**Default recommended stack:**
1. **F-GRPO** (drop-in upgrade over vanilla GRPO)
2. **+ Scaf-GRPO scaffolding** (essential at small scale)
3. **+ Apriel-Reasoner difficulty-aware length penalty** (free)

Alternative path if we want theoretical unification:
1. **f-GRPO** (covers both RLVR and PA in one implementation)

### Phase 4: Generalization refinement
- GRPO-SG (sharpness-guided token weighting) — if we see overfitting

## Implementation cost estimates

| Component | LoC estimate | Dependencies |
|-----------|-------------:|--------------|
| GRPO core | ~400 | Rollout infrastructure, verifier |
| F-GRPO add-on | ~30 | GRPO implemented |
| Scaf-GRPO scaffolding | ~200 | GRPO + hint dataset |
| GRPO-SG token weighting | ~50 | GRPO |
| Adaptive domain sampling | ~100 | Multi-domain dataset |
| Difficulty-aware length penalty | ~30 | GRPO |

Full stack: ~800 LoC to implement everything. **Rollout + verifier
infrastructure is the bulk of the work** (~60% of effort). The GRPO variants
themselves are small additions.

## Open questions (to answer before implementation)

1. **Does scaffolding transfer to non-reasoning domains?** Paper is math-only.
   Worth probing on function-calling or instruction-following.
2. **What's minimum group size for GRPO at our scale?** 7B paper uses G=8-16.
   At 122M we may need bigger groups (more noise).
3. **Is online rollout compute feasible on 2× Strix Halo?** Rollouts are
   memory-bound (KV cache) and single-machine; DDP doesn't help rollouts,
   only the policy update step.

## See also

- `knowledge/training/instruct_alignment_techniques_2025_2026.md` — earlier
  survey (SimPO, ORPO, KTO, RePO, AlphaPO) — still valid; this doc adds the
  GRPO/RLVR dimension
- `knowledge/training/alignment_implementation_details.md` — implementation
  reference for SFT + DPO
- `docs/research/broad-research-synthesis-2026-05-06.md` Part 5 — full
  discussion of post-training in context
