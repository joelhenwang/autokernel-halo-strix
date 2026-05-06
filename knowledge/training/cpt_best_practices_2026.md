---
title: "Continued Pretraining Best Practices (2025-2026)"
domain: training
type: reference
status: active
tags: [cpt, continued-pretraining, domain-adaptation, catastrophic-forgetting, token-replay, small-lm]
related:
  - imu1_recipe_2026.md
  - ../architectures/reliable_small_lm_insights.md
  - ../../docs/research/broad-research-synthesis-2026-05-06.md
---

# Continued Pretraining Best Practices (2025-2026)

Consolidated guidance from three primary papers on CPT at various scales,
plus our own empirical trajectory with OdinFlat/OdinHalo (wikitext →
gpt-training-small → stem-crawl-solo).

## Primary sources

- **Qalb** — arXiv:2601.08141 (Jan 2026): LLaMA-3.1 8B → Urdu, 1.97B CPT tokens
- **Domain-Adaptive CPT of Small LMs** — arXiv:2504.09687 (April 2025): 125M
  model, 400M→1B tokens CPT on educational domain (**directly applicable to
  OdinFlat's 122M**)
- **TriMix / logit fusion** — arXiv:2604.18106 (April 2026): test-time CPT
  alternative via logit fusion
- **SmolLM3 blueprint** — HuggingFace blog (July 2025): 3-stage staged
  pretraining with data-mixture transitions
- **IMU-1** — arXiv:2602.02522 (Jan 2026): WSD + post-hoc EMA for cross-stage
  polish

## Key findings (cross-paper)

### 1. Token replay prevents catastrophic forgetting

**Empirical sweet spot: ~7-10% of original pretraining distribution** mixed
into CPT corpus.

- Qalb: **140M English Wikipedia tokens** mixed with **1.84B Urdu tokens**
  (7.1% replay) → +44.64 points over base LLaMA-3.1, zero observed forgetting
  on English benchmarks
- General rule: below 5% replay → detectable drop on original domain; above
  15% → CPT progress slows materially

**Our current practice:** 0% replay. Next CPT run should include 7-10% wikitext
mixed into stem-crawl or dolma.

### 2. LR rewind to 30-50% of original peak

**Best practice:** Start CPT at **30-50% of the peak LR** used during original
pretraining. Higher risks undoing pretraining progress; lower slows domain
adaptation.

Our trajectory:
- Original (wikitext): 8e-4 peak
- CPT runs: 6e-4 (75% of original) — **slightly too high** by this rule
- Recommended: 3-4e-4 (37-50%) for next CPT stage

### 3. Short warmup relative to CPT duration

5-10% of CPT step count, NOT pretraining-length warmup.

- For a 1000-step CPT run: 50-100 warmup steps
- For a 5000-step CPT run: 250-500 warmup steps

Our current runs use warmup=500 universally, which is ~10-40% of the run
depending on step count. At short runs (~1000 steps) we're massively
over-warming.

### 4. Stage order matters — quality gradient

Progress from broad → domain-specific → high-quality polish:

```
Stage 1 (Stable):    Broad web / general mixture
Stage 2 (Stable):    Domain data + broad replay
Stage 3 (Decay):     High-quality curated subset, LR → 0, EMA checkpoints
```

Our current order (wikitext → gpt-small → stem-crawl) is ad-hoc. stem-crawl
is domain-specific (STEM); if we'd planned this as CPT we would have done:
broad foundations → domain-focus → high-quality polish. The missing polish
phase is what the T² scaling paper predicts will give the biggest
quality-per-token bump at this late stage.

### 5. Per-stage capability evaluation (not just loss)

Every CPT paper measures domain-relevant benchmarks at stage boundaries, not
just aggregate validation loss. Standard set:
- **MMLU** (knowledge): strongest CPT signal; +8.1% on the 125M model at 400M→1B
- **HellaSwag** (common-sense reasoning): +7.6% on same
- **BLiMP** (grammatical competence): zero-shot, cheap
- **Domain-specific benchmarks**: e.g., Urdu-specific scores for Qalb, AIME for math CPT

Our trajectory has zero capability evaluation. We see only training loss. Every
future CPT stage should finish with a benchmark pass.

## Empirical CPT recipe (recommended defaults for Odin)

Based on the cross-paper findings, applied to our 58M-122M scale:

```bash
# Example: CPT from OdinFlat-stem-crawl-final onto dolma-10b-odin32k
MODEL=models/odin_flat.py CLASS=OdinFlat \
  DATASET=datasets/dolma-10b-odin32k.bin \
  CKPT_DIR=checkpoints/odin-flat-dolma-ddp \
  EPOCHS=1 \
  LR=3e-4 \                             # 37% of original 8e-4
  WARMUP_STEPS=300 \                    # ~5% of typical 6k-step run
  MAX_GRAD_NORM=0.8 \
  bash scripts/launch_ddp.sh checkpoints/odin-flat-stem-crawl-ddp/step_FINAL.pt
```

**With replay:** wire `scripts/generate_replay_mixture.py` to interleave 7%
wikitext tokens with dolma before training. The `halo_training/mixture_data.py`
module already has the `MixtureDataset` abstraction — we just need to build the
config.

**Decay phase after main CPT:** 10% of CPT steps at decay LR=0.1× peak, on a
curated high-quality subset (e.g., wikipedia validation + curated stem
papers). Save checkpoints every 10% of decay. Post-hoc EMA β=0.8 over final 10.

## Alternatives and less-common approaches

### Logit fusion (no CPT at all) — TriMix

If the goal is task-specific capability (not general domain adaptation), logit
fusion at test time can substitute for weight updates:

```
output_logits = α·small_domain_model + β·task_aligned_model + γ·large_scaling_model
```

The TriMix paper (arXiv:2604.18106) validates this at low-resource language
adaptation: **prioritizing the small specialized model's logits is crucial**
(challenges the "large-model-dominant" assumption).

For Odin: not immediately useful (we don't have multiple complementary models),
but worth parking if we end up with both OdinFlat and a bigger reference.

### Tokenizer extension during CPT — continued BPE

If the CPT corpus has significant OOV vocabulary (e.g., scientific notation,
code syntax unique to a language), **continued BPE training** (arXiv:2512.03989)
extends the tokenizer without breaking existing token IDs.

For Odin on stem-crawl: odin-32k probably under-covers scientific Greek
letters, unit abbreviations, and latex markers. If we see >5% UNK rate on
stem-crawl validation, extend the tokenizer.

## Anti-patterns (things to avoid)

1. **No replay at all** — catastrophic forgetting by step 500 on small models
2. **Same peak LR as original pretraining** — undoes prior progress
3. **Ignoring per-stage capability eval** — you only learn about forgetting
   after 6+ hours of wasted compute
4. **Full-length warmup in short CPT** — wastes 30-40% of the training budget
5. **Random stage order** (our current practice) — quality-gradient order
   gives better final polish

## Our trajectory assessed against these rules

| Rule | Our OdinFlat practice | Grade |
|------|----------------------|:-----:|
| Token replay 7-10% | 0% | ❌ |
| LR rewind to 30-50% | 75% (8e-4 → 6e-4) | ⚠️ |
| Short warmup 5-10% | 45% (500 / 1128) | ❌ |
| Quality-gradient stage order | Ad-hoc | ⚠️ |
| Per-stage eval | None | ❌ |

Quick wins for next run:
- Add 7% wikitext replay to next dolma/stem run
- Lower LR to 3-4e-4 on next resume
- Reduce warmup to 5-10% of total steps
- Build minimal eval scorecard before next CPT stage

## See also

- `docs/research/broad-research-synthesis-2026-05-06.md` Part 4 — full CPT
  discussion with citations
- `knowledge/training/imu1_recipe_2026.md` — related pretraining stack
- `halo_training/mixture_data.py` — our existing MixtureDataset code
