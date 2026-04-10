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
