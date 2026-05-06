---
title: "Scaling Laws 2025-2026: T² Scaling, Architecture Conditioning, Chinchilla Biases"
domain: training
type: reference
status: active
tags: [scaling-laws, chinchilla, overtraining, t2-scaling, inference-cost, compute-optimal]
related:
  - cpt_best_practices_2026.md
  - ../architectures/reliable_small_lm_insights.md
  - ../../docs/research/broad-research-synthesis-2026-05-06.md
---

# Scaling Laws 2025-2026: The Ground Has Shifted

## Most important new result: T² Scaling

**Paper:** "Test-Time Scaling Makes Overtraining Compute-Optimal" —
arXiv:2604.01411, April 2026.

### Thesis

Traditional scaling laws (Chinchilla) optimize (model-size, training-tokens)
for fixed training compute. They **do not account for inference cost**, and
modern LLMs scale at test time (repeated sampling, pass@k).

Train-to-Test (T²) scaling jointly optimizes:
- Pretraining model size
- Pretraining token count
- Test-time inference sample count (k)

under a **fixed end-to-end compute budget** (training + inference).

### Key finding

When inference cost is included, the compute-optimal pretraining point shifts
**radically into the overtraining regime**, well outside the range standard
pretraining scaling laws suggest.

**Empirically validated:** Pretraining heavily overtrained models in the
T²-predicted-optimal region yields substantially stronger downstream
performance vs Chinchilla-optimal baselines at matched end-to-end compute.

**Post-training robustness:** Findings survive the post-training stage —
meaning T² matters for deployed systems, not just raw base models.

### Implications for Odin (122M)

Chinchilla-optimal: ~20× tokens/params = **~2.4B tokens**
T²-optimal for deployed: 50-200× = **~6-25B tokens**

Our current OdinFlat trajectory: ~950M cumulative tokens = **~7.8× ratio**.
By Chinchilla we're somewhat under-trained; by T² we're **drastically
under-trained** if we plan to deploy with any test-time scaling.

**dolma-10b-odin32k.bin (6.9B tokens, already on Machine A)** would take us
to ~57× ratio — solidly in the T²-optimal range for inference-deployed
small models.

### For project strategy

Before planning another "architecture improvement" experiment, consider that
our single highest-impact compute spend is likely **running the existing
architecture on more tokens**. The gap between "7× ratio" and "57× ratio"
is larger than any architectural gain we could plausibly engineer.

## Chinchilla Approach 2 has systematic biases

**Paper:** "Problems with Chinchilla Approach 2: Systematic Biases in
IsoFLOP Parabola Fits" — arXiv:2603.22339, March 2026.

### Thesis

Chinchilla Approach 2 (IsoFLOP parabola fitting) — the most widely used
method for fitting scaling laws — has biases even on noise-free synthetic
data.

Applied to published **Llama 3 IsoFLOP data** at frontier scales: these
biases imply **6.5% parameter under-allocation**, or ~$1.4M in unnecessary
compute at 50% H100 MFU on a $3.8e25 FLOP training budget.

### Three sources of error

1. **IsoFLOP sampling grid width**: Taylor approximation accuracy around the minimum
2. **Uncentered IsoFLOP sampling**: grids are rarely centered on the true optimum
3. **Loss-surface asymmetry** (α ≠ β): the standard parabolic assumption fails

### Fix

**Chinchilla Approach 3** (full loss-surface fit) largely eliminates these
biases. Previously regarded as less data-efficient, numerically unstable,
prone to local minima, hard to implement. The paper shows each concern is
unfounded or addressable via **Variable Projection** — which exploits the
partially linear objective structure to reduce to a well-conditioned
2-dimensional optimization.

### Implications for us

We don't fit scaling laws at our scale (compute budget doesn't justify
dedicated scaling studies), but:
- **Don't trust published "compute-optimal" mixes** that use Approach 2
- Our Phase 3 shape-sweep findings (±3% noise band) implicitly accept this
  level of noise — consistent with the paper's warnings
- If we ever do run a scaling study (30M → 122M μP probe, for example),
  use Approach 3 with Variable Projection, not Approach 2

## Architecture-conditioned scaling laws

**Paper:** "Scaling Laws Meet Model Architecture: Toward Inference-Efficient
LLMs" — arXiv:2510.18245, October 2025.

### Thesis

Architecture parameters (hidden size, MLP/attention ratio, GQA groups) change
the training-efficiency / inference-efficiency trade-off. Standard scaling
laws ignore this.

### Methodology

Trained **>200 models** spanning **80M - 3B params, 8B - 100B training
tokens**. Fit a conditional scaling law that augments Chinchilla with
architectural information.

### Findings

At matched training compute, optimal architectures achieve:
- **+2.1% accuracy** (vs LLaMA-3.2)
- **+42% inference throughput**

### Key architectural knobs (from the paper)

1. **Hidden size** — balance against depth
2. **MLP-to-attention ratio** — tune based on expected seq length at deployment
3. **GQA group count** — inference-memory vs capacity trade-off

### For Odin-specific architecture decisions

Our OdinFlat has:
- hidden = 768
- MLP = 2816 (ratio 3.67×) — **heavy on MLP**
- GQA groups = 4 (standard)

The paper's Pareto-front architectures tend to have MLP/attn ratio around
2.5-3.0× for inference-efficient configurations. Our 3.67× is somewhat
MLP-heavy; we'd pay in inference throughput.

**Recommendation:** If we ever redesign for a deployment target, pull down
the ffn_inner ratio to ~3.0× and reallocate params to depth (more layers).
But this is an architectural-redesign-level change; don't do it mid-stream.

## Meta-insight: the three scaling law papers together

Combined takeaway:

1. **Train more tokens than Chinchilla suggests** (T² Scaling)
2. **Don't trust published "optimal" allocations** (Chinchilla Approach 2 biases)
3. **Architectural choices shift the Pareto frontier** (Architecture-
   conditioned scaling)

For a practical 122M model today:
- Budget: target 50-100× tokens/params, not 20×
- Use WSD scheduler with decay phase (concentrates training on high-quality data)
- Don't over-optimize model shape without matching deployment target
- Evaluation on inference-cost-relevant benchmarks (pass@k, latency)

## Open questions for our project

1. **What's our actual deployment target?**
   Per the reliable-small-lm-insights doc, we have no defined deployment
   envelope. T² scaling depends on whether we plan to use repeated sampling
   at inference. If yes → overtrain aggressively. If no → Chinchilla ~20× is fine.

2. **Should we re-scope OdinFlat to 60M with more training?**
   At 60M / 6B tokens (100× ratio) we'd likely beat our current 122M / 1B
   (8× ratio) on deployment metrics. Worth a modeling study.

3. **Is dolma-10b enough?** For 122M at 50-100× ratio, we need 6-12B tokens.
   dolma-10b-odin32k.bin is 6.9B — right at the edge. Two epochs of dolma
   gets us to ~100×.

## See also

- `docs/research/broad-research-synthesis-2026-05-06.md` Part 1 — full scaling
  laws discussion
- `knowledge/training/imu1_recipe_2026.md` — a 430M / 72B token (168× ratio)
  success story validating the T² "overtrain" finding empirically
- `knowledge/architectures/reliable_small_lm_insights.md` — our existing
  discussion of deployment envelope (GAP 1)
