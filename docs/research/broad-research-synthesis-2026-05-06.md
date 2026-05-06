# Broad Research Synthesis — 2026-05-06

Comprehensive deep-dive across **efficient pretraining, continued pretraining,
post-training, and new architectures** for small-to-medium language models,
synthesizing 240+ recent papers (2025-01 through 2026-04) from Hugging Face
Papers. Companion to the earlier focused synthesis at
`docs/research/small-lm-research-2026-05-06.md`; this one is broader and draws
on more primary sources.

**Scope:** From scaling laws that redefine "compute-optimal" to new optimizers
to RLVR algorithms, at multiple model scales (we include 15B findings where
they transfer down to our 58M/122M).

**Methodology:** 30+ targeted searches via `hf papers search`, 456 unique papers
matched, 241 from 2025+. Read the 17 highest-signal summaries, deep-read 6 key
papers (IMU-1, SPECTRA, Newton-Muon, T² scaling, MoDA, architecture-aware
scaling).

---

## Executive summary — 10 findings that matter most for Odin

| # | Finding | Paper / source | Action for Odin |
|---|---------|----------------|-----------------|
| 1 | **T² scaling laws redefine compute-optimal.** Including inference cost shifts optimal training deep into overtraining regime. Chinchilla 20× is UNDER-trained for deployed models. | [T² Scaling, 2604.01411] | Justify 20-50× token:param ratio. Our ~7× for OdinFlat is significantly under-trained. |
| 2 | **NorMuon + Cautious Weight Decay beats AdamW by 3.85%** on small-LM pretraining at 430M scale. | [IMU-1, 2602.02522] | **P0 replacement for our fused AdamW.** 7-step Newton-Schulz, ~3% overhead. |
| 3 | **Architectural interventions compound:** QK-norm + per-head gating + value residuals + LayerNorm scaling each give measurable gains. Most additive. | [IMU-1, 2602.02522] | OdinFlat already has QK-norm (in GQA). Add value residuals + LayerNorm scaling. |
| 4 | **MoDA (cross-LAYER) is different from our MoDA (cross-iteration).** New paper's Mixture-of-Depths Attention gives +2.11% with 3.7% FLOP overhead at 1.5B. | [MoDA, 2603.15619] | Try adding cross-layer depth KVs to OdinFlat — orthogonal to OdinHalo's cross-iter MoDA. |
| 5 | **SPECTRA spectral clipping** is orthogonal to optimizer choice; works on AdamW, Signum, AdEMAMix, yielding SOTA. | [SPECTRA, 2603.14315] | Apply on top of whichever optimizer wins. Cheap via Newton-Schulz. |
| 6 | **F-GRPO / f-GRPO** Focal-loss-inspired GRPO variants give +6.2 pass@256 at no extra compute. Drop-in to any group-relative RLVR algo. | [F-GRPO 2602.06717; f-GRPO 2602.05946] | Start here when we build RLVR; preferred over vanilla GRPO. |
| 7 | **Scaf-GRPO scaffolds hard problems** when the policy plateaus. +44.3% relative pass@1 on AIME24. | [Scaf-GRPO, 2510.19807] | Essential for learning-cliff avoidance in small-LM RL. |
| 8 | **Low-Rank Clone distillation: 20B tokens matches SOTA** trained on trillions (~1000× efficiency) at small scale. | [LRC, 2505.12781] | If we ever need a strong small base, distill from Qwen3/Gemma3 instead of pretraining from scratch. |
| 9 | **Token replay in CPT (10% original data) prevents catastrophic forgetting.** Confirmed across Urdu, educational, coding domains. | [Qalb 2601.08141; Domain CPT 2504.09687] | Add 10% wikitext replay to future stem-crawl + dolma runs. |
| 10 | **Continued BPE training** adapts tokenizers for new domains without breaking existing tokens. | [Teaching Tokenizers, 2512.03989] | Optional: extend odin-32k for stem/code if we move that direction. |

---

## Part 1 — Scaling laws: the ground has shifted

### T² Scaling (arXiv:2604.01411, April 2026) — most important new result

**Claim:** Pretraining scaling laws like Chinchilla do not account for test-time
compute (repeated sampling). When you jointly optimize over:
- pretraining tokens
- model size
- number of inference samples (k in pass@k)

the optimal allocation shifts radically. For deployed LLMs that use test-time
scaling, optimal pretraining is in the **overtraining regime**, well outside
the range Chinchilla suggests.

**Empirical validation:** They pretrained heavily overtrained models in the
predicted-optimal region. Overtrained models substantially outperform
Chinchilla-optimal models when inference cost is included.

**Survives post-training:** Their findings hold through the post-training stage.

**Implication for Odin (122M / ~1B cumulative tokens = ~8× ratio):**
- At 8× we're close to *old* Chinchilla-optimal (~20×) but deeply under-trained
  in T² terms. Optimal for inference-deployed models likely >100×.
- Our 122M model should ideally see 10-20B+ tokens, not ~1B.
- dolma-10b at 6.9B tokens would get us to ~57×. Much more aligned with T²
  recommendations.

### Chinchilla Approach 2 biases (arXiv:2603.22339)

**Claim:** Standard IsoFLOP parabola fits (used by everyone) have systematic
biases. On published Llama-3 data, these biases imply 6.5% parameter
under-allocation — an M unnecessary compute at frontier scale.

**Fix:** Chinchilla Approach 3 (full loss surface) is better; previously thought
less stable, but Variable Projection makes it well-conditioned.

**Implication for Odin:** When thinking about our own "compute-optimal" choices,
don't trust parabola fits of sparse ablations. Our existing architecture
ablation studies are small enough that this is probably noise-dominated anyway.

### Scaling laws meet architecture (arXiv:2510.18245)

**Claim:** Key arch factors (hidden size, MLP/attn ratio, GQA) shift inference
efficiency materially. Trained 200 models 80M-3B, fit conditional scaling law.

**Findings:** At matched training compute, optimal architectures give:
- Up to **+2.1% accuracy**
- Up to **+42% inference throughput**
vs LLaMA-3.2.

**Implication for Odin:** Our `hidden=768, ffn=2816 (3.67×)` is heavy on MLP
vs their finding. Don't retune unless we're post-pretraining-complete and
focused on inference. But remember this for future model designs.

---

## Part 2 — Optimizers: Muon family is the new Pareto frontier

### NorMuon + Cautious Weight Decay (IMU-1, arXiv:2602.02522, Jan 2026)

**Headline:** A 430M model trained on 72B tokens approaches the benchmark
performance of models trained on 56× more data (~4T tokens).

**Full recipe:**

```
Architecture:
  - QK-Norm attention  (stability)
  - Per-head gated attention  (expressivity, attention-sink mitigation)
  - Value residual connections  (gradient flow)
  - LayerNorm scaling  (depth pathology mitigation)

Optimizer:
  - NorMuon with neuron-wise normalization
  - Cautious Weight Decay (selective regularization via gradient-weight alignment)
  - 7 Newton-Schulz iterations with Polar Express constants
  - Dion Triton kernel for efficiency (~3% overhead vs AdamW)
  - lr = 0.0235 for 2D params (matrices)
  - lr = 0.007 for 1D params (biases, LN gains, embeddings, lm_head — use AdamW for these)

Parametrization:
  - μP for hyperparameter transfer across scales

Schedule:
  - 3-stage training (Stable + Stable + Decay)
  - WSD, 20% decay fraction (comparable to cosine)
  - Post-hoc EMA: β=0.8 over final 10 checkpoints (saved every 5k steps during decay)
```

**Ablation gains (each additive):**
- NorMuon vs AdamW: −2.88% relative
- + Cautious WD: −0.97% additional
- Total optimizer gain: −3.85%
- 3-stage training progression: 0.461 → 0.522 → 0.560 average
- + EMA: +0.014 final

**Throughput:** 1.8M tok/s peak; 26-36% MFU depending on stage.

**For Odin:**
- Replace our fused AdamW with NorMuon + CWD for next training run.
- We already have QK-norm in OdinFlat (NoPECodaAttention).
- Add value residuals to all block types.
- Add LayerNorm scaling at deeper blocks.
- Port μP parametrization to enable transferable hyperparameters.
- Post-hoc EMA in decay phase.

### Newton-Muon (arXiv:2604.01472, April 2026)

**Claim:** Muon can be interpreted as a quadratic surrogate Newton method that
*neglects* right-preconditioning by the input second-moment. Adding this back:

```
W ← W - η · msgn(G · (Z Zᵀ)⁻¹)
```

where msgn(X) = UVᵀ from compact SVD.

**Result:** On Modded-NanoGPT speedrun config, Newton-Muon reaches target
validation loss in **6% fewer iterations**, reducing wall-clock by ~4%.

**For Odin:** Marginal over plain Muon but free gain. Worth adopting after
NorMuon baseline is established.

### SPECTRA — Spectral Clipping (arXiv:2603.14315, March 2026)

**Claim:** Optimizer updates can have large spectral norms that destabilize
training. Stochastic gradient noise can have sparse spectral spikes (few
dominant singular values).

**Fix:**
- Post-spectral clipping of updates (bounds spectral norm)
- Optional pre-spectral clipping of gradients (suppresses noise spikes)
- Efficient via Newton-Schulz (no expensive SVD)
- Acts as composite Frank-Wolfe with spectral-norm constraints

**Results:** SOTA on AdamW, Signum, AdEMAMix when applied. Models have smaller
weight norms → better generalization.

**For Odin:** **Orthogonal to optimizer choice.** Apply on top of NorMuon/Muon/
AdamW. Cheap implementation (Newton-Schulz we already have for Muon). Run as
ablation once the main optimizer-swap is landed.

### Variance-Adaptive Muon (arXiv:2601.14603, Jan 2026)

**Claim:** Adam ≈ variance-adaptive sign update. Apply variance-adaptive
normalization to momentum *before* orthogonalization in Muon.

**Variants:**
- Muon-NSR (noise-to-signal ratio modulation)
- Muon-VS (variance-based scaling, no extra hyperparameters)

**Result:** LLaMA-1.2B: **1.36× fewer iterations** to target validation loss vs
well-tuned Muon. Consistent lower validation loss than both AdamW and Muon.

**For Odin:** Muon-VS is hyperparameter-free. Drop-in replacement.

### Optimizer summary for Odin

| Method | Expected gain over AdamW | Extra compute | Complexity |
|--------|------------------------:|--------------:|-----------|
| Muon (baseline) | −1-2% | +3% | Moderate |
| NorMuon | −2.88% | +3% | Moderate |
| **NorMuon + Cautious WD** | **−3.85%** | +3% | Moderate |
| Newton-Muon | −4-5% (est) | +3-5% | Moderate |
| Muon-VS | −4-5% (est, 1.36× speedup) | +3% | Low (HP-free) |
| SPECTRA on any | Additive on top, 0.5-1% | +1% | Low |

**Recommendation:** Start with **NorMuon + Cautious WD + SPECTRA post-clipping**.
Expected total: ~−4.5% to −5% relative loss improvement over our current fused
AdamW, for ~4-5% throughput cost.

---

## Part 3 — Efficient pretraining recipes

### Key insight: multi-stage training is now table-stakes

Every new small-LM release uses some form of staged pretraining:

| Model | Scale | Stages | Key feature |
|-------|------:|-------:|-------------|
| SmolLM3 | 3B | 3 | Stable → Quality-up → Decay |
| IMU-1 | 430M | 3 | Warmup/Stable/Decay with distinct data |
| Nanbeige4-3B | 3B | 3 | FG-WSD (fine-grained WSD) with progressive data mixtures |
| Baguettotron | <1B | — | Synthetic-data-only |

**Common elements:**
1. WSD scheduler (or FG-WSD with per-stage data evolution)
2. Upsample quality data (code, math) in later stages
3. Decay phase with LR → 0 on best data
4. EMA of checkpoints during decay
5. Intra-document attention masking

### Where to Begin — subnetwork selection (arXiv:2510.07227, Oct 2025)

**Claim:** Structurally sparse sub-network initializations outperform random
initialization under matched compute. Evolutionary search finds good sub-nets.

**Results:** Their best model (Whittle) matches Pythia perplexity with **9.2×
fewer pretraining tokens** by using:
- Sparse LLM-weight-based initialization
- Evolutionary search for optimal sub-network structure
- Knowledge distillation from larger teacher

**For Odin:** If we want a 58M / 122M from scratch with minimal data, consider
distilling/initializing from Qwen3-0.6B or Gemma-3-270M instead of Xavier init.

### Low-Rank Clone (arXiv:2505.12781, May 2025)

**Claim:** 20B tokens with LRC distillation matches SOTA models trained on
trillions of tokens.

**Mechanism:** Low-rank projection matrices that jointly:
- Soft-prune teacher weights (compression)
- Align student activations with teacher (behavioral equivalence)
- Align FFN activations (not just hidden states)

**For Odin:** If we have access to Qwen3-3B or similar, LRC gives us a
shortcut to a strong 122M base in ~20B tokens.

### FG-WSD (Nanbeige4-3B, arXiv:2512.06266, Dec 2025)

**Claim:** Fine-Grained Warmup-Stable-Decay scheduler progressively refines
data mixtures across stages.

**For Odin:** Already doing ad-hoc version (wikitext → gpt-small → stem-crawl).
Formalize as FG-WSD with data-mixture transitions tied to LR phases.

### Data mixing findings

From FineInstructions (arXiv:2601.22146) and the broader literature:
- Synthetic instruction data at pretraining scale (≥100M instructions) helps
  small models more than large ones.
- Diversity matters more than volume at this scale.

---

## Part 4 — Continued pretraining (CPT)

### Qalb — Urdu LLaMA CPT (arXiv:2601.08141, Jan 2026)

**Most relevant CPT paper for our work.** LLaMA-3.1 8B → Urdu via:
- **1.97B tokens CPT**
- **140M tokens English Wikipedia (~7% replay)** to prevent catastrophic forgetting
- **Alif instruction fine-tuning** after

**Result:** +44.64 points over base LLaMA-3.1 8B on Urdu benchmarks.
90.34 weighted avg, beats prior SOTA (87.1).

**Transferable insight:** Token replay ratio of ~7-10% is the empirical sweet
spot. Our current 0% replay risks drift.

### Domain-Adaptive CPT for Small LMs (arXiv:2504.09687, April 2025)

**Directly relevant:** 125M parameter model (very close to OdinFlat's 122M).

**Protocol:** 400M → 1B tokens CPT on educational domain.

**Results:**
- **+8.1% MMLU**
- **+7.6% HellaSwag**

**Key insight:** Memory-optimized training configurations matter more at small
scale. The paper emphasizes thoughtful preprocessing and staged training.

**For Odin:** Our incremental CPT (wikitext→gpt-small→stem-crawl) is basically
doing this pattern. The paper suggests we should see similar ~8% benchmark
gains once we eval properly.

### Efficient Low-Resource Language Adaptation (arXiv:2604.18106, April 2026)

**Novel insight: Logit fusion instead of weight finetuning.**

TriMix fuses logits from three sources at test time:
- Small model continually pretrained on target domain
- High-resource LM fine-tuned on task
- Scaling benefit of a large model

**Key finding:** "Prioritizing small specialized model's logits is crucial" —
challenges the "large-model-dominant" assumption.

**For Odin:** Provocative idea — instead of SFT-ing a 122M Odin into a good
instruct model, consider serving it alongside a bigger-model logit mixer at
inference time. Not immediately actionable but worth parking.

### CPT best practices — consolidated

Pulled from 3+ recent CPT papers:

1. **Replay ratio:** 5-10% of original pretraining data mixed in
2. **LR rewind:** Start CPT at 30-50% of original pretraining peak LR
3. **Warmup:** 5-10% of CPT steps, not pretraining-length warmup
4. **Stage order:** domain data first, high-quality data last (quality decay)
5. **Eval as you go:** per-stage capability probe (MMLU, HellaSwag, etc.)

Our current recipe is partially aligned:
- ✓ LR rewind (8e-4 → 6e-4 = 75%, could go 40-50% = 3-4e-4)
- ✓ Warmup (500/1128 ~45% — too long for final phases; 10-15% better)
- ✗ No replay (0% instead of 5-10%)
- ✗ Stage order ad-hoc (wikitext, gpt-small, stem-crawl) not quality-gradient
- ✗ Per-stage eval missing

---

## Part 5 — Post-training: GRPO family is the new standard

Post-R1 era (Jan 2025+): Group Relative Policy Optimization (GRPO) dominates
RLVR. A tidal wave of variants in the last 6 months.

### GRPO variants (comprehensive)

#### F-GRPO (arXiv:2602.06717, Feb 2026)

**Problem:** Small groups miss rare-correct trajectories. Even as total correct
mass grows, unsampled-correct mass can shrink.

**Fix:** Focal-loss-inspired difficulty-aware advantage scaling — downweight
high-success prompts.

**Drop-in to:** GRPO, DAPO, CISPO.

**Results on Qwen2.5-7B:** pass@256 improvements:
- GRPO: 64.1 → 70.3
- DAPO: 69.3 → 72.5
- CISPO: 73.2 → 76.8

No extra compute or group-size increase.

#### f-GRPO (arXiv:2602.05946, Feb 2026)

Theoretical framework: preference alignment = divergence estimation.

Proposes f-divergence-based policy updates. Extends beyond RLVR to preference
alignment (PA) with f-HAL (Hybrid Alignment Loss: on+off policy mix).

**For Odin:** The theoretical framing matters because once we have an
alignment/reasoning pipeline, f-GRPO's generalization across RLVR + PA is
a unifying choice.

#### Scaf-GRPO (arXiv:2510.19807, Oct 2025)

**Problem:** "Learning cliff" — when problems are beyond the model's current
capability, rewards are always zero, advantages collapse, no gradient signal.

**Fix:** Scaffold with tiered in-prompt hints when plateau detected. Hints
range from abstract concepts → concrete steps.

**Result:** Qwen2.5-Math-7B on AIME24: **+44.3% relative pass@1** vs vanilla
GRPO.

**For Odin:** Essential for any reasoning-RL on a small model. Small LMs hit
the learning cliff fast; scaffolded hints are necessary.

#### GRPO-SG (arXiv:2511.00066, Oct 2025)

Sharpness-guided token-weighted GRPO. Downweights tokens with large
gradient-norm contributions.

**For Odin:** Generalization-focused; use if we see overfitting during RLVR.

### Apriel-Reasoner — full 15B RL recipe (arXiv:2604.02007, April 2026)

**Scale:** 15B Apriel-Base, five domains (math, code, instruction following,
logic puzzles, function calling).

**Novel contributions:**
1. **Adaptive domain sampling:** Preserves target domain ratios despite
   heterogeneous rollout lengths.
2. **Difficulty-aware length penalty:** Longer reasoning for hard problems,
   shorter for easy ones. No extra training cost.

**Results:** Trained with 16K-token budget, generalizes to 32K at inference.
Beats Apriel-Base on AIME/GPQA/MMLU-Pro/LiveCodeBench. **30-50% shorter
reasoning traces.**

**For Odin (down-scaled):** The adaptive domain sampling + difficulty-aware
length penalty are recipe ingredients that transfer. Not urgent — we have no
reasoning base yet — but worth implementing in a halo_training/rlvr.py when
we get there.

### Post-training recommendation stack for Odin

Given our complete lack of post-training infrastructure:

**Phase 1 (SFT):**
- Use SmolTalk2 + stem-crawl-derived instructions
- Simple next-token loss on assistant turns, masked on user
- 1-2 epochs, LR 2e-5
- Expected 2-3 hours DDP

**Phase 2 (Preference Alignment):**
- **ORPO first** (no separate SFT stage, no reference model, simplest)
- Then try **APO + model merging** (from SmolLM3 recipe) if ORPO plateaus

**Phase 3 (RLVR for reasoning — later):**
- **F-GRPO** as default (simple drop-in over GRPO)
- **+ Scaf-GRPO scaffolding** for hard problems (essential at small scale)
- **+ Difficulty-aware length penalty** (Apriel-Reasoner)
- Math/code/function-calling datasets at small scale (~5K each)

---

## Part 6 — Architectures worth knowing

### MoDA — Mixture-of-Depths Attention (arXiv:2603.15619, March 2026)

**Critical naming collision:** This MoDA is different from our Parcae's MoDA.
- **Our MoDA (OdinHalo):** Cross-*iteration* depth KVs (same layer, prior iter)
- **New MoDA:** Cross-*layer* depth KVs (prior layers, same iter)

**Mechanism:** Each attention head attends to current-layer KVs + depth KVs
from preceding layers. Hardware-efficient algorithm resolves non-contiguous
memory access patterns (97.3% of FlashAttention-2 efficiency at 64K seq).

**Results at 1.5B:**
- +0.2 perplexity improvement across 10 validation benchmarks
- +2.11% on 10 downstream tasks
- **3.7% FLOPs overhead** only

**Bonus finding:** Post-norm + MoDA outperforms pre-norm + MoDA.

**For Odin:** Orthogonal to our cross-iteration MoDA. Could combine:
- OdinFlat + new MoDA → richer cross-layer interactions (no iterations)
- OdinHalo + new MoDA → both cross-iter AND cross-layer depth attention

Worth ablating; 3.7% cost is very affordable.

### Higher-order Linear Attention (HLA) (arXiv:2510.27258, Oct 2025)

**Claim:** Linear attention + SSMs are usually first-order. HLA realizes 2nd+
order interactions with compact prefix sufficient statistics.

**Second-order HLA:**
- Constant-size state
- Linear per-token computation
- No n×n matrices materialized
- Chunk-parallel training via associative scans

**For Odin:** Interesting as a next-generation replacement for our current
NoPECodaAttention. Higher-order interactions at linear cost would be a big
win on the looped variant. Parking lot until we want to revisit attention.

### InfoMamba (arXiv:2603.18031, March 2026)

**Claim:** Diagonal short-memory SSMs have structural gaps in approximating
causal attention. Their analysis characterizes what's missing.

**Fix:** Replace self-attention with concept-bottleneck linear filtering +
information-maximizing fusion (IMF) with SSM.

**For Odin:** Attention-free hybrid. Too invasive to swap into OdinFlat at this
stage. Relevant if we build a new model family.

### SeeDNorm (arXiv:2510.22777, Oct 2025)

**Claim:** RMSNorm discards input-norm information. Static γ insufficient for
distributional shifts (especially zero-shot).

**Fix:** Dynamic input-dependent γ. Data-dependent, self-rescaled.

**Results:** Consistently beats RMSNorm, LayerNorm, DyT across LLM pretrain
and computer vision tasks.

**For Odin:** Drop-in RMSNorm replacement, minimal extra cost. Easy ablation.

### Architectures summary

| Arch change | Cost | Expected gain | Maturity |
|-------------|-----:|-------------:|:--------:|
| Add value residuals | +0.5% params | +0.5-1% loss | IMU-1 validated |
| Add LayerNorm scaling | negligible | +0.3-0.5% loss | IMU-1 validated |
| Add per-head gating | +0.5% params | +0.3-0.5% | IMU-1 validated |
| Replace RMSNorm with SeeDNorm | negligible | +0.2% | Paper-only |
| Add cross-layer MoDA | +3.7% FLOPs | +2% tasks at 1.5B | Paper, 1 scale |
| Replace attention with HLA | similar | unknown transfer | Research-grade |
| Replace attention with InfoMamba | similar | unknown transfer | Research-grade |

---

## Part 7 — Tokenizer adaptation

### Teaching Old Tokenizers New Words (arXiv:2512.03989, Dec 2025)

**Two techniques:**

1. **Continued BPE training:** Extend pre-trained tokenizer by continuing BPE
   merge learning on new data. Avoids the common problem of appended tokens
   being unreachable/never used.

2. **Leaf-based vocabulary pruning:** Remove redundant tokens while preserving
   model quality.

**For Odin:** Our odin-32k is currently static. If we want to domain-adapt to
stem-crawl vocabulary (scientific terms), continued BPE is the right approach.
Not urgent — 32K is already domain-general — but possible future refinement.

---

## Part 8 — Test-time methods

### TEMPO (arXiv:2604.19295, April 2026) — important theoretical insight

**Problem with TTT:** Existing TTT methods plateau quickly. Self-generated
reward drifts without external calibration → performance plateau + diversity
collapse.

**Fix (TEMPO):** EM-based alternation:
- E-step: Policy refinement on unlabeled questions
- M-step: Critic recalibration on a labeled dataset

**Results:** OLMO3-7B on AIME 2024: **33.0% → 51.1%**. Qwen3-14B: 42.3% → 65.8%.

**For Odin:** Test-time training is still experimental at small scale. Parking
lot — relevant when we have a capable reasoning model.

---

## Part 9 — Concrete action queue for Odin (priority-ranked)

| Rank | Experiment | Cost | Expected impact | Dependencies |
|:----:|-----------|------|-----------------|-------|
| **1** | **Implement NorMuon + Cautious WD** as optimizer replacement | 2-3 days | **−3.85% loss** over AdamW on next training run | New optimizer file |
| **2** | Add SPECTRA spectral clipping on top of NorMuon | 1 day | −0.5-1% loss, better generalization | #1 implemented |
| **3** | Remove weight decay from embeddings + 1D params | 5 min | Free win (IMU-1 + OLMo2) | None |
| **4** | Intra-document attention masking | 1 hr | Free stability win | None |
| **5** | Add value residual connections to all blocks | 2 hr | +0.3-0.5% loss | None |
| **6** | Add LayerNorm scaling (depth-dependent) | 1 hr | +0.2-0.5% | None |
| **7** | μP parametrization port for HP transfer | 4 hr | Enables cheap hyperparam tuning at 30-60M scale | None |
| **8** | Build minimal eval scorecard (per-domain BPB + BLiMP) | 1 day | Visibility into every future checkpoint | lm-eval-harness |
| **9** | SFT pipeline + SmolTalk2 fine-tune on OdinFlat | 2 days | First instruct-capable Odin | Eval scorecard |
| **10** | ORPO alignment on SFT checkpoint | 1 day | First aligned Odin | #9 done |
| **11** | APO + model merging (if ORPO plateaus) | 2 days | Cleaner alignment | #10 done |
| **12** | **Decay phase with EMA on curated mix** (after stem-crawl) | 4 hr | +0.1-0.2 loss, final polish | Stem-crawl done |
| **13** | F-GRPO RLVR on math/code (after ORPO) | 3 days | First reasoning Odin | #10 + verifiable data |
| **14** | Scaf-GRPO for hard-problem scaffolding | 1 day on top of #13 | Avoids learning cliff | #13 done |
| **15** | OdinHalo stem-crawl run (symmetric) | 4 hr | Verify looped catches up thesis | Machines free |
| **16** | **T²-optimal training plan** — use dolma-10b-odin32k (6.9B tokens) for a real ~50× ratio run | 6-8 hr cumulative | The "real" base model | Copy dataset to B |
| **17** | Cross-layer MoDA as orthogonal addition to OdinFlat | 2 days | +2% tasks potential | OdinFlat stable |
| **18** | SeeDNorm drop-in for RMSNorm | 1 hr | +0.2% loss, minimal cost | None |
| **19** | Continued BPE training on stem-crawl for vocab | 1 day | Better coverage of STEM terms | Optional |
| **20** | Low-Rank Clone distillation experiment | 1 week | Shortcut to strong 122M | Access to teacher weights |

### Reordered by "what to do first" (realistic sequence)

**Sprint 1 (1-2 days):** Foundation wins — #3, #4, #5, #6. Do all four as a
single architecture-refresh commit. Free ~+1% loss improvement across the
next run.

**Sprint 2 (3-4 days):** Optimizer upgrade — #1, #2, #7. Biggest single
impact (~−4%). Requires careful testing. Retain AdamW path as fallback.

**Sprint 3 (2-3 days):** Evaluation infrastructure — #8. Unblocks all
downstream decisions. Run retroactively on all existing checkpoints.

**Sprint 4 (1 week):** Post-training pipeline — #9, #10, optionally #11.
First instruct-capable Odin.

**Sprint 5 (1-2 weeks):** Reasoning — #13, #14, #13 on math + code data. Start
with small curated sets.

**Sprint 6 (optional, 1 week):** T²-optimal pretraining run — #16 with all
Sprint 1+2 improvements. This is "the real base model" — fp16, 6.9B tokens,
new optimizer, all architectural improvements.

---

## Part 10 — Open questions and research bets

### What I don't yet know and want to find out

1. **Does μP actually transfer from 30M → 122M at our specific architecture?**
   Plan: Train 3 small-scale (~30M) probes with NorMuon-CWD, fit scaling,
   extrapolate to 122M, verify.

2. **Does cross-layer MoDA combine additively with our cross-iter MoDA?**
   Both could interact unexpectedly. Run ablation.

3. **Is SPECTRA worth the extra Newton-Schulz pass?**
   Claim is generalization. Need to measure on held-out eval, not just train
   loss.

4. **What's our real compute-optimal ratio under T² scaling?**
   Depends on deployment plan. If we target pass@1 (no inference scaling),
   Chinchilla ~20× is OK. If we target pass@k>4, we should train 50-100×.

5. **Does Scaf-GRPO scaffolding transfer to non-reasoning domains?**
   Paper is all math/logic. Worth testing on instruction-following or
   function-calling.

6. **Can LRC distillation work from Qwen3-3B → OdinFlat?**
   Architectural mismatch (Qwen3 transformer vs our HyPE hybrid). LRC assumes
   similar width. Low priority unless we need a better base fast.

### Bets I'd take right now (if forced)

1. **Bet A:** NorMuon+CWD gives −3-5% loss improvement on next training run.
   Confidence: 80%. Risk: stability issue at our tiny scale.

2. **Bet B:** Intra-document masking + no-WD-embeddings gives +0.5% loss at
   zero cost. Confidence: 95%.

3. **Bet C:** F-GRPO is the right RLVR starting point. Confidence: 70%. Risk:
   pure GRPO or DAPO might be simpler to implement.

4. **Bet D:** Value residuals are additive and help. Confidence: 75%. Risk:
   could interact with our HyPE/Parcae structure.

5. **Bet E:** T²-optimal training on dolma (6.9B, ~57× ratio) produces a
   qualitatively more coherent 122M than our current trajectory. Confidence:
   90%. Risk: dolma's quality curation is different from wikitext's, may not
   match 57× expected gain.

---

## Sources (primary)

All 2025-2026 via `hf papers search` and `hf papers read`. Dates in YYYY-MM-DD.
For the full 241-paper list, see the internal research cache (not committed).

**Pretraining:**
- IMU-1: 2602.02522 (2026-01-25)
- Where to Begin / Whittle: 2510.07227 (2025-10-08)
- Low-Rank Clone: 2505.12781 (2025-05-19)
- Nanbeige4-3B: 2512.06266 (2025-12-06)

**Continued pretraining:**
- Qalb: 2601.08141 (2026-01-13)
- Domain-Adaptive CPT small LMs: 2504.09687 (2025-04-13)
- TriMix logit fusion: 2604.18106 (2026-04-20)
- Teaching Old Tokenizers: 2512.03989 (2025-12-03)

**Post-training / RLVR:**
- F-GRPO: 2602.06717 (2026-02-06)
- f-GRPO (divergence): 2602.05946 (2026-02-05)
- Scaf-GRPO: 2510.19807 (2025-10-22)
- GRPO-SG: 2511.00066 (2025-10-29)
- Apriel-Reasoner: 2604.02007 (2026-04-02)
- APO: 2408.06266 (2024-08, adopted by SmolLM3)

**Scaling laws:**
- T² scaling: 2604.01411 (2026-04-01)
- Chinchilla bias: 2603.22339 (2026-03-21)
- Scaling+Architecture: 2510.18245 (2025-10-21)

**Optimizers:**
- Newton-Muon: 2604.01472 (2026-04-01)
- SPECTRA: 2603.14315 (2026-03-15)
- Variance-Adaptive Muon: 2601.14603 (2026-01-21)
- MuonAll: 2511.06086 (2025-11-08)

**Architectures:**
- MoDA (cross-layer): 2603.15619 (2026-03-16)
- Higher-order Linear Attention: 2510.27258 (2025-10-31)
- InfoMamba: 2603.18031 (2026-03-08)
- SeeDNorm: 2510.22777 (2025-10-26)

**Test-time:**
- TEMPO: 2604.19295 (2026-04-21)

---

## Related prior syntheses in this repo

- `docs/research/small-lm-research-2026-05-06.md` — focused synthesis (earlier,
  today) with more SmolLM3 detail
- `knowledge/INDEX.md` — 36-file knowledge base index
- `knowledge/architectures/reliable_small_lm_insights.md` — Labonne guide
- `knowledge/training/instruct_alignment_techniques_2025_2026.md` — post-training
- `docs/adal/llm_novelties_2025_2026_report.md` — earlier novelties review (HALO, HyPE, Lightning)
