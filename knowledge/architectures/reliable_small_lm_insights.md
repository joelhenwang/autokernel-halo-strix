---
title: "Insights from Reliable Small LM Training Guide"
domain: architectures
type: reference
status: active
related:
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%small-lm, %best-practices, %stability]
---

# Insights from "Reliable Small LM Training Guide" Applied to Our Project

**Source:** `docs/reliable_small_language_model_training_guide.md` (Maxime Labonne, AI Engineer Europe 2026)
**Date:** 2026-04-10

---

## What We're Already Doing Right

| Guide Principle | Our Implementation |
|----------------|-------------------|
| Design for deployment hardware first | Entire project built around gfx1151 profiling (no MFMA, 240 GB/s, 6 MB L2) |
| Prefer cheap operators on target hardware | causal-conv1d 10x, Griffin element-wise "free", rocBLAS-friendly dims |
| On-device profiling as architecture design | `bench.py`, `profile.py`, roofline model, 20+ kernel benchmarks |
| Quantization-aware thinking | `dequantize_int4` (16.3x), int4 inference estimates in every hypothesis |
| Tied embeddings | All architectures use tied input/output embeddings |

## Gaps We Need to Close

### GAP 1: No defined deployment target (Sections 1, 3)
**Problem:** We optimize for "tok/s and BPB on BabyLM" — an engineering metric, not a product spec.
**Action:** Write a target envelope doc: what task, what latency, what memory budget, what quantization level.
**Impact:** Changes architecture ranking entirely. A coding model wants attention. A summarizer wants long recurrence. A classifier doesn't need generation at all.

### GAP 2: Embedding-heavy budget (Section 4.1)
**Problem:** 50257 × 1024 = 51.5M params in embeddings = 21% of 250M budget.
**Action:** Consider vocabulary reduction (30K-40K tokens) to free 10-20M params for transformation depth/width.
**Impact:** Could improve effective model capacity by 5-10%. Labonne's deck specifically calls out embedding-heavy designs as reducing effective model usefulness.

### GAP 3: Single-metric evaluation (Sections 8.3, 8.4, 9)
**Problem:** We track loss, BPB, tok/s, MFU. That's optimization health only.
**Missing metrics:**
- Per-domain perplexity (split BabyLM by source)
- Capability probes at checkpoints (BLiMP grammatical competence)
- Fixed sample packs for qualitative regression
- Repetition/doom-loop detection (n-gram metrics)
- Quantized quality (int4 BPB vs fp16 BPB)

### GAP 4: No post-training pipeline (Section 6)
**Problem:** 100% focused on pretraining. Guide says narrow task SFT matters disproportionately for small models.
**Action:** Build basic SFT infrastructure after pretraining converges.
**Priority:** After we have a trained base model with good BPB.

### GAP 5: No repetition/doom-loop defenses (Section 7)
**Problem:** Guide warns about "doom looping" — small models + reasoning traces + complex tasks → repetitive failure.
**Action:** Add n-gram repetition metric to training loop + evaluation. Create loop-specific test prompts.
**Priority:** Medium — only matters once we generate text, not during pretraining.

### GAP 6: No quantized evaluation (Section 4.4)
**Problem:** All our profiling is fp16. Guide says quantized profiling should happen early.
**Action:** Add int4/int8 quantization + eval to training pipeline. Compare BPB before/after quantization at checkpoints.
**Impact:** May invalidate some kernel speedup estimates. int4 changes the bottleneck calculus.

---

## Strategic Implications for Architecture Ranking

The guide suggests our ranking should weight **deployment efficiency** more heavily:

| Factor | Current Weight | Guide-Recommended Weight |
|--------|---------------|-------------------------|
| Training tok/s | HIGH | MEDIUM |
| Training loss/BPB | HIGH | HIGH |
| int4 decode tok/s | LOW | **HIGH** |
| Memory after quantization | LOW | **HIGH** |
| Task-specific quality | NONE | **HIGH** |
| Elastic inference (MatFormer) | LOW | **HIGH** |

This would promote:
- **MatFormer** (free elastic inference, already fastest in training)
- **TERNARY-REFLEX** and **OBSIDIAN** (L2-resident quantized weights)
- **RESONANT-LOOP** (fewest unique params → fastest decode)
- **PROMETHEUS** (global context via attention → better for retrieval tasks)

And demote:
- Complex architectures (ARCHON, CHIMERA) — deployment overhead
- Deep narrow designs (48L × d=512) — serial decode latency

---

## Immediate Action Items

| Priority | Action | Effort | When |
|----------|--------|--------|------|
| **P0** | Define target task envelope | 1 hour | Before next architecture decision |
| **P1** | Add per-domain BPB to training loop | 2 hours | Next training run |
| **P1** | Add n-gram repetition metric | 1 hour | Next training run |
| **P1** | Create fixed 20-30 prompt regression pack | 1 hour | This week |
| **P2** | Integrate lm-evaluation-harness | 4 hours | Next sprint |
| **P2** | Add int4 quantization + eval pipeline | 4 hours | Next sprint |
| **P2** | Vocabulary size experiment (50K vs 32K) | 8 hours | Next sprint |
| **P3** | Basic SFT infrastructure | 8 hours | After base model trained |
| **P3** | BLiMP linguistic probes | 2 hours | After base model trained |

---

## Key Quote

> "A small model project usually fails when the team asks the wrong question.
> Wrong: 'How do we make the smallest possible version of a general LLM?'
> Right: 'What recipe produces the best reliability for this task under this latency and memory budget?'"

We're currently asking the engineering question. The guide says we need to answer the product question first.

---

## Update: Maxime Labonne Talk (AI Engineer, May 2026)

**Source:** "Everything I Learned Training Frontier Small Models" — Maxime Labonne, Liquid AI (Head of Pre-Training)
**Date:** 2026-05-03 (video: youtube.com/watch?v=fLUtUkqYHnQ)

### New Findings Beyond the Written Guide

**1. 28T tokens on 350M params — far beyond Chinchilla, performance keeps growing**
- LFM 2.5 350M trained on 28 trillion tokens (80,000× Chinchilla optimal)
- Roberts et al (2026) new test-time scaling laws suggest even MORE tokens would be beneficial
- Implication for us: our 544M token dataset on 58.5M params is dramatically undertrained. Need orders of magnitude more data (dolma-10b is a step but still small by these standards)

**2. Embedding efficiency matters at small scale**
- Gemma 3 270M: 63% of params are embeddings (inefficient)
- Gemma 2.5 0.8B: 29% embeddings
- LFM 2: ~10% embeddings (90% effective params)
- Our TyrHaloLight: embed_rank=448 factorized — effective embedding cost is ~15% (factorization helps but 50257 vocab still heavy)
- Action: consider vocab reduction to 32K for more effective params

**3. LFM 2 architecture: gated short convolutions confirmed fastest**
- On-device profiling on AMD Ryzen AI Max+ 395 (our exact hardware!) and Samsung Galaxy S25 Ultra
- Short convs significantly faster than: sliding window attention, gated Delta Net, GLA, GQA
- This validates our ShortConvBlock-heavy design (4 conv + 2 GQA per block)
- LFM 2 architecture: hybrid short convolutions + GQA (same pattern as TyrHaloLight)

**4. Doom loop solution: concrete pipeline**
- Doom loop ratio: 15-16% after pretraining, barely moves after SFT
- DPO reduces significantly, RL with verifiable rewards reduces to near-zero
- Qwen 3.5 0.8B in reasoning mode: >50% doom loops (scaled-down = bad)
- **Anti-doom-loop pipeline:**
  1. Generate 5 rollouts with temperature sampling (diverse, some avoid loops)
  2. Generate 1 rollout with temperature=0 (expected to doom loop)
  3. LLM jury scores all rollouts
  4. Best score = chosen, worst score = rejected
  5. Train DPO on this data → model learns to avoid doom loops
  6. Follow with RL + n-gram repetition penalty → near-zero doom loops

**5. Cold start SFT data critical for small models**
- If RL task doesn't train well, likely missing cold-start SFT data for that task
- Solution: go back to SFT stage, add task-similar data, retrain
- Small models more sensitive to this than large models

**6. Task-specific > general-purpose**
- LFM 2.5 350M optimized for: data extraction + tool use
- Not trying to be best at code or math — users don't use 350M models that way
- "The more narrow you can design it, the better it is"
- Implication: when we post-train, pick 1-2 target tasks and optimize for them

**7. Agentic tools compensate for low knowledge capacity**
- Small models hallucinate due to low knowledge capacity
- Fix: give them web search / tool use capabilities
- "These tiny models are actually very good at agentic tasks"
- Tool use + good reasoning > raw knowledge
- Long context weakness also fixable via recursive language model environments + Python

**8. Small models are NOT scaled-down big models**
- Liquid's core philosophy: edge models are their own thing
- Different architecture decisions, different training recipes, different post-training
- Distillation creates embedding-heavy inefficient models (Gemma approach)
- Training from scratch with on-device profiling is better

### Relevance to Our Project

| Finding | Impact | Priority |
|---------|--------|----------|
| 28T tokens on 350M | Our 544M tokens is severely undertrained. Need 10B+ minimum | **HIGH** |
| Short conv fastest on Ryzen AI Max+ 395 | Validates our 4-conv + 2-GQA block design | Confirmed |
| Embedding efficiency | Our factorized embed is good but vocab reduction (50K→32K) would help | MEDIUM |
| Doom loop anti-pipeline | Must implement DPO + RL post-training after pretraining | P3 (after base model) |
| Cold start SFT | Build task-specific SFT data before RL stage | P3 |
| Task-specific focus | Pick 1-2 target tasks for post-training (tool use + data extraction recommended) | P2 |
| Agentic tools for small models | Plan tool-use SFT from the start | P2 |

---

## Update: GPT-X2-125M Model Card Analysis (May 2026)

**Source:** AxiomicLabs/GPT-X2-125M on HuggingFace
**Key result:** 125M model matches SmolLM2-135M (46.7% vs 47.6% avg benchmarks) using 27× fewer tokens (75B vs 2T). Trained on single RTX 3080 Ti in ~500 hours.

### Architecture: Depth > Width at Small Scale

30 layers × d=576 (was 27L in v1). Custom 32K BPE tokenizer (trained on FineWeb-Edu) saves ~9% compression vs GPT-2 50K vocab. Saved embedding params reinvested into 3 extra layers. GQA 9Q/3KV (3:1 ratio). QK-Norm (RMSNorm per head before RoPE). SwiGLU 2.67x ratio. RoPE theta=100K. No bias anywhere.

### Training Recipe

- **75B tokens**, 4-source progressive curriculum (FineWeb-Edu 55%, DCLM 38%, FineMath 4.5%, NPset-Python 2.5%)
- **WSD schedule:** 2K warmup → 80% stable at LR 1.5e-3 → 20% linear decay to 0 (not cosine)
- **Weight decay annealing:** 0.1 during stable phase → 0.01 during decay
- **z-loss (1e-4):** On logit magnitudes, used first 31B tokens only then disabled
- **Batch:** 524K tokens (micro=8, seq=1024, accum=64)
- **AdamW** betas=0.9/0.95

### Key Design Decisions and Why

1. **Custom 32K tokenizer** — 50K GPT-2 BPE wastes ~15% of params on embeddings at 125M scale. 32K custom BPE trained on domain data gives 9% better compression AND frees params for transformer layers.
2. **30 layers at d=576** — Depth is primary quality driver. Wider but shallower is worse.
3. **LR 1.5e-3** — v1 used 6e-4 (too conservative). SmolLM uses 3e-3. Higher LR + longer warmup works.
4. **WSD > cosine** — Linear decay to 0 during final 20% gives tighter convergence.
5. **z-loss early, disable late** — Prevents logit magnitude drift during high-LR phase. Unnecessary once model stabilizes.
6. **Weight decay annealing** — 0.1→0.01 during cooldown matches "less regularization during fine-grained convergence" intuition.
7. **Progressive curriculum** — Start with general text, ramp math/code during stable LR, taper during decay.
8. **AST code normalization** — 1.25x token compression on Python via AST→TinyDSL pseudocode.

### Relevance to Our Project

| Finding | Our Status | Action |
|---------|-----------|--------|
| Depth > width | Parcae loop gives depth via weight sharing (12 effective layers) | Validated |
| Custom 32K tokenizer | Using GPT-2 50K (50257). ~15% embed overhead | Vocab reduction would free ~5M params |
| WSD schedule | Using cosine decay | Consider WSD for next training run |
| z-loss early | Not implemented | Add z-loss (1e-4 on logit magnitudes) during warmup |
| Weight decay annealing | Constant wd | Anneal 0.1→0.01 during LR decay |
| 75B tokens at 125M | 544M tokens on 58.5M | We need 50-100× more data |
| Progressive curriculum | Single dataset | Build curriculum for dolma-10b |
| QK-Norm critical | Already using | Confirmed |

---

## Update: Model Card Analysis — InstructLM-500M, Baguettotron, whiff-mamba2-50M (May 2026)

### InstructLM-500M (Microsoft, arXiv 2406.14491)

**Method:** Instruction Pre-Training — augment pretraining corpus with synthesized instruction-response pairs generated by a fine-tuned Mistral-7B "instruction synthesizer." ~5 QA pairs per raw text, ~52 tokens each. Mixed with raw text → standard next-token prediction.

**Architecture:** Mistral 500M (16L × d=1536, 24 heads, SiLU, 32K vocab). Nothing novel architecturally — all gains from data.

**Key results:**
- 500M on 100B tokens (instruction-augmented) ≈ Pythia-1B on 300B tokens (vanilla). 2× param + 3× data efficiency.
- Models trained with instruction PT benefit MORE from subsequent instruction tuning than vanilla PT models.
- Domain-adaptive: Llama3-8B + instruction PT on domain text → beats Llama3-70B on biomedicine and finance.

**Data generation pipeline:**
1. Fine-tune Mistral-7B on 30+ context-based QA datasets (SQuAD, TriviaQA, PubMedQA, CommonsenseQA, etc.)
2. Input: `<CON> raw_text </CON>` → Output: `<QUE> question <ANS> answer </END>`
3. Multi-round synthesis (M=2 general, M=3 domain): each round uses previous outputs as few-shot examples
4. Quality: 77.5% accuracy, 92.9% relevance, 40+ task categories
5. Throughput: ~1 day on A100-80GB per 1B tokens

**Relevance:** Highest-impact data technique found. When scaling to dolma-10b, augmenting with instruction pairs could give 2-3× effective data efficiency. Requires 7B model to run synthesizer (pre-generate offline via cloud/API).

### Baguettotron (PleIAs, 321M)

**Architecture:** Llama-style, **80 layers × d=576**, GQA 9Q/3KV, SwiGLU 1536, 65K vocab, tied embeddings. Claims "deepest SLM in its size range."

**Training:** 200B fully synthetic tokens (SYNTH dataset) on 16×H100 (~20K H100-hours). Nanotron framework.

**Key findings:**
- 80 layers at d=576 works and outperforms shallower wider alternatives
- "Deeper architectures benefit MORE from dense reasoning data"
- Approaches Qwen-0.6B (2× params) on MMLU. 10-50× less data than comparable models
- MMLU signal emerges from step 9000 (~2 hours of training)

**Novel techniques:**
1. **Backreasoning** — give generation model the ground-truth answer, have it "simulate not knowing" to produce reasoning traces. Inverse of typical prompting.
2. **Simulated entropy tokens** — 18 levels (`⟨H≈0.1⟩` to `⟨H≈1.8⟩`) in tokenizer for inference control.
3. **Stenographic reasoning markers** — special tokens for logic flow (→, ↺), confidence (●, ◐, ○), verification (☐, ☑, ✓).
4. **SYNTH dataset** — 7 task categories, fully synthetic from Wikipedia seeds. Multi-stage pipelines with formalized checks + LLM-as-judge curation.
5. **Rolling thinking** — append `<think>` traces, discard past ones.

**Relevance:** Validates extreme depth at small scale (our Parcae loop approach). Backreasoning is a novel data generation technique worth adopting. Fully synthetic data competitive with 10-50× web data — important for data-constrained training.

### whiff-mamba2-50M (limloop, 50M)

**Architecture:** Mamba2, 20L × d=512, state_size=48 (vs standard 128), expand=1 (no internal expansion), conv_kernel=4. Phi-3.5-mini tokenizer (32K vocab). 50M params.

**Training:** 230M tokens bilingual (Russian + English). **CPU-only** (Ryzen 7 5700G, no GPU). 200-300 hours.

**Key technique — Progressive context + batch scheduling:**

| Stage | Context | Batch | Curriculum |
|-------|---------|-------|------------|
| 1-3 | 64→128→256 | 64 | 40%+ Wikipedia |
| 4 | 256 | 96 | Drop Wikipedia, add bilingual pairs |
| 5-8 | 512 | 96→512 | Progressive batch increase |

Could NOT start at full context — model wouldn't converge. Progressive scheduling was necessary.

**Novel decisions:** state_size=48 (CPU memory), expand=1 (param efficiency), bilingual parallel corpus prevents language confusion, SSM "attractor states" = doom loops in SSM form.

**Relevance:** Progressive context scheduling (start short, grow) could save 10-15% early compute. Progressive batch scheduling could improve gradient stability. SSM attractor states confirm doom loops are universal across architectures.

### Cross-Model Priority Actions

| Technique | Source | Effort | Expected Impact |
|-----------|--------|--------|-----------------|
| Instruction-augmented pretraining | InstructLM | HIGH (need 7B synth) | 2-3× data efficiency |
| Extreme depth validated | Baguettotron | Already doing (Parcae) | Consider mean=3 |
| Backreasoning data generation | Baguettotron | MEDIUM | Better reasoning traces |
| Fully synthetic data | Baguettotron | HIGH | Competitive with 10-50× web data |
| Progressive context scheduling | whiff-mamba2 | LOW (trivial) | Save early compute |
| Progressive batch scheduling | whiff-mamba2 | LOW | Better gradient stability |
