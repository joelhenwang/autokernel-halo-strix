---
title: "META-ENGRAM"
domain: architectures
type: plan
status: stale
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %meta-engram]
---

# META ENGRAM

**Two-Tier Knowledge System: Always-On Meta Tokens + On-Demand Hash Table Lookup**

## Hypothesis

Combine two complementary knowledge mechanisms into the **knowledge-maximized** small model: (1) **Hymba meta tokens** (128 learned embeddings always in context — compressed world knowledge, always available, interaction-rich) and (2) **Engram hash tables** (millions of N-gram entries — on-demand factual lookup, O(1), context-gated). On a parallel hybrid backbone, this creates a model that KNOWS more per parameter than any dense model, because knowledge is stored in dedicated mechanisms rather than buried in weight matrices.

**Evidence:**
- Hymba: meta tokens add +1.4% accuracy at 300M for 131K params. They act as "backstop" for attention and encode compressed knowledge.
- Engram: doubles effective depth (Layer 5 → Layer 12 CKA alignment). Stores 56-71% of factual knowledge. 5× LR, zero WD for tables.
- Combining them is novel: meta tokens handle GENERAL knowledge (grammar, world facts, reasoning patterns), while Engram handles SPECIFIC knowledge (named entities, formulaic phrases, N-gram statistics).

---

## Architecture

```
[128 Meta Tokens] + Input Tokens → Embedding (d=1024, tied LM head)
  → 16 Parallel Hybrid Blocks (d=1024):
      RMSNorm → [Conv(12ch) || Griffin Rec(4ch)] → Concat → OutProj → +Residual
      + Engram injection (at layers 1, 4, 8, 12)    ← 4 Engram layers!
      RMSNorm → SwiGLU FFN → +Residual
  → Final RMSNorm → LM Head
```

### Two-Tier Knowledge System

**Tier 1: Meta Tokens (Always-On, General Knowledge)**

```python
# 128 learned embeddings, prepended to every input
meta = self.meta_tokens.expand(B, 128, 1024)    # ~131K params
x = concat(meta, embed(tokens))                   # (B, 128+T, 1024)
# Meta tokens interact with all real tokens through:
# - Conv channels: local patterns around meta tokens
# - Recurrence channels: meta info propagates through state
# - Attention (if any): meta tokens attend to/from all positions
```

**What meta tokens learn:** In Hymba's analysis, meta tokens absorb:
- Grammar/syntax patterns (always-useful context)
- Common reasoning templates
- Backstop for attention softmax (prevents "forced-to-attend" to irrelevant tokens)

**Tier 2: Engram Tables (On-Demand, Specific Knowledge)**

```python
# At 4 designated layers (1, 4, 8, 12):
bg = bigram_table[hash(token[t-1], token[t])]     # K=4 heads
tg = trigram_table[hash(token[t-2:t+1])]           # K=4 heads
fg = fourgram_table[hash(token[t-3:t+1])]          # K=4 heads (NEW: 4-grams)

alpha = sigmoid(rmsnorm(h) @ rmsnorm(W_K @ concat(bg, tg, fg)))
x = x + alpha * W_V @ concat(bg, tg, fg)
```

**What Engram tables learn:** From DeepSeek's analysis:
- Named entities ("Alexander the Great" → specific embedding)
- Formulaic phrases ("by the way", "in order to")
- Domain-specific N-grams
- Gating removes noise from hash collisions

### Synergy: Meta + Engram

The two tiers are complementary:

| Aspect | Meta Tokens | Engram Tables |
|--------|-------------|---------------|
| **When active** | Always (prepended) | On-demand (hash lookup) |
| **What they store** | General knowledge, patterns | Specific facts, N-grams |
| **Capacity** | 131K params (compressed) | ~50M params (expansive) |
| **Access pattern** | Interacts via conv/rec/attention | O(1) hash, context-gated |
| **Cost** | One-time computation at start | Per-token at Engram layers |

**Together:** Meta tokens provide a "knowledge foundation" that's always available. Engram tables provide "knowledge details" when specific N-gram patterns are encountered. The model can rely on meta tokens for general reasoning and Engram for factual precision.

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| d_conv | 768 (12 channels × 64) |
| d_rec | 256 (4 channels × 64) |
| n_layers | 16 |
| ffn_inner | 1536 (SwiGLU, 1.5× — reduced to fit large Engram budget) |
| meta_tokens | 128 |
| Engram layers | 4 (layers 1, 4, 8, 12) |
| Engram d_engram | 96 |
| Bigram table | 131,072 × 96 = 12.6M params |
| Trigram table | 65,536 × 96 = 6.3M params |
| 4-gram table | 32,768 × 96 = 3.1M params |
| Engram hash heads | 4 per order |
| **Total Engram params** | **~22M (×2 for 4 layers shared) + projections = ~50M** |

Wait — Engram tables are shared across layers (per Engram paper: one table, per-layer W_K). So:
- Tables: bigram 131K×96 + trigram 65K×96 + 4gram 32K×96 = ~22M
- Per Engram layer: W_K (1024→288) + W_V (288→1024) + gate proj = ~0.6M
- 4 Engram layers: 4 × 0.6M = 2.4M
- **Total Engram: ~24.4M**

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×1024, tied) | 51.5M |
| Meta tokens (128×1024) | 0.13M |
| Per layer: parallel hybrid (conv+rec+outproj) | ~5.2M |
| Per layer: SwiGLU FFN (1024→1536→1024) | ~4.7M |
| Per layer: RMSNorm ×2 | ~2K |
| Per layer total | ~9.9M |
| 16 layers | ~158.4M |
| Engram tables (shared) | 22.0M |
| Engram projections (4 layers) | 2.4M |
| Final RMSNorm + routing | ~0.1M |
| **TOTAL** | **~234.5M** ✓ |

**Knowledge ratio:** 24.4M Engram + 0.13M meta = **24.5M dedicated knowledge params** = 10.4% of model. The rest is "thinking" (projections, FFN, norms).

Compare: GPT-2 124M has 0% dedicated knowledge — everything is implicit in dense weights.

## Decode Speed (Strix Halo)

| Mode | Weight reads | Estimate |
|------|-------------|----------|
| fp16 | ~469 MB layers + Engram lookups (~KB) + LM head (103 MB) | ~3.4 ms = ~294 tok/s |
| int4 | ~117 MB + LM head (26 MB) | ~0.84 ms = ~1190 tok/s |

Speed is comparable to other d=1024 16-layer designs. The value proposition is QUALITY, not speed — more knowledge per param.

## Training

### Optimizer Groups

| Group | Optimizer | LR | Weight Decay |
|-------|-----------|-----|-------------|
| Backbone (conv, rec, FFN, norms) | AdamW | 8e-4 cosine → 8e-5 | 0.1 |
| Meta tokens | AdamW | 8e-4 (standard) | 0.01 (light) |
| Engram tables | **Adam** | **5× base = 4e-3** | **0** |
| Engram projections (W_K, W_V) | AdamW | 8e-4 | 0.1 |

### Phase Training

1. **Phase 1 (40%):** Backbone only (parallel hybrid + FFN). No Engram, no meta tokens. Learn basic language.
2. **Phase 2 (30%):** Add meta tokens + Engram. Both knowledge tiers activate. Meta tokens learn general knowledge; Engram tables learn N-gram patterns.
3. **Phase 3 (30%):** Full model. Fine-tune all components together. Engram gating sharpens.

### Knowledge Distillation (Optional)

Teacher (GPT-2 Medium 355M) provides logits. Meta tokens and Engram tables absorb teacher's knowledge faster than pure gradient descent on next-token prediction:
```
loss = 0.7 * CE(student, targets) + 0.3 * KL(student, teacher, T=2.0)
```

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Meta tokens + Engram = redundant knowledge storage | MEDIUM | Meta = general, Engram = specific. Different granularity. If redundant, meta tokens cost only 131K params — negligible. |
| 4 Engram layers too many (Engram paper used 2) | MEDIUM | Start with 2 (layers 1, 8), ablate adding layers 4, 12. Monitor gate activations. |
| ffn_inner=1536 too small (LFM2 uses 6656) | HIGH | Engram compensates for factual capacity. Conv channels add 3×768 projections. If quality poor, reduce Engram tables and increase FFN. |
| Meta tokens slow down prefill (128 extra positions) | LOW | At d=1024, 128 positions add ~5% to prefill compute. Negligible for decode (meta tokens are already processed). |

## Success Criteria

1. Loss < 4.5 in 15 min
2. Meta tokens show non-trivial attention patterns (not ignored)
3. Engram gating activates selectively (entities → high alpha, function words → low)
4. Removing Engram at inference degrades factual tasks > 30% (Engram IS storing knowledge)
5. Removing meta tokens degrades performance > 1% (meta tokens contribute)
6. Quality advantage over same-param model without meta+Engram (ablation)

## Implementation Roadmap

1. Implement parallel hybrid backbone (Parallel Caveman pattern, 16 layers)
2. Implement meta token prepending (128 × d_model)
3. Implement Engram module (3 N-gram orders, 4 hash heads, context-aware gating)
4. Integrate Engram at 4 layers with shared tables, per-layer W_K
5. Verify params ~234M
6. Phase 1: backbone only (40% budget)
7. Phase 2: add meta + Engram (30%)
8. Phase 3: full model fine-tuning (30%)
9. Ablation: meta only vs Engram only vs both
10. Compare factual recall vs non-knowledge baselines

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x), **silu_gate_mul** (1.6x), **cross_entropy** (1.8x)
- **prefix_scan** (8.4x) — adaptable for Griffin associative scan
- Apply via `autokernel.optimize(model, training=True)`
- Engram hash tables: no special kernel needed (hash + gather are fast on GPU)

### Griffin/SSM Scan: Use Chunked Linear Recurrence
**Do NOT use sequential loops or `torch.associative_scan`** — both yield only 1.3K tok/s (4% MFU) on gfx1151. Use **chunked linear recurrence** (chunk_size=64) for 5x speedup:
- Reference: `models/amadeus.py:selective_scan_chunked`
- Griffin operator `(a₂·a₁, a₂·b₁+b₂)` fits the same pattern
- The `sqrt(1-a²)` coupling term can be precomputed per chunk

### Throughput Estimates (from AMADEUS baseline)
- **Eager mode:** ~6-8K tok/s, 16-20% MFU for ~250M SSM model
- **With autokernel patterns:** ~7-9K tok/s, 18-22% MFU
- **Token budget:** 15 min = 5.4-7.2M | 45 min = 16-22M | 120 min = 43-58M

### MFU: 65-75% training
FFN dominates compute (weight-bandwidth-bound). Recurrence is element-wise (~95% MFU). Engram lookups are memory-bound but small.

---

## Possible Optimizations & Throughput Estimate

**Baseline (estimated):** ~5,500 tok/s eager (13% MFU)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +90% MFU — Engram lookups may cause graph breaks | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| `causal-conv1d` in GatedConv | 10x conv speedup | Available |
| Engram hash table optimization | Pre-compute hash indices; fuse lookup + gate | Possible |
| Meta token overhead | 128 meta tokens add ~3% to sequence length | By design |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~10,500 tok/s (25% MFU)
**Tokens in 45 min:** ~28.4M (1.8 BabyLM epochs)
**Ranking:** #14 of 22 architectures
