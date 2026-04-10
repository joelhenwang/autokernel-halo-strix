# GENIUS CAVEMAN

**Information-Weighted Dual-Path Architecture with Engram Reflexes**

## Hypothesis

Not all tokens deserve equal compute. A "genius caveman" architecture routes tokens by **information content**: low-information tokens (articles, prepositions, punctuation — the "glue" of language) are predicted by **Engram hash tables + a tiny L2-resident reflex path**, while high-information tokens (nouns, verbs, rare words — the "meaning" of language) get **full processing through a Mamba-3 complex SSM with mHC multi-branch residuals and adaptive depth**.

This combines:
- **Caveman insight**: 65% of tokens are predictable glue (JuliusBrussee/caveman)
- **Dual Cortex**: System 1 (L2-resident, cheap) / System 2 (DRAM, powerful) routing
- **Engram**: Hash-indexed N-gram tables predict glue words via O(1) lookup
- **Mamba-3**: Complex-valued SSM for content word reasoning
- **mHC**: 4-branch residual highway connecting both paths
- **Resonant Loop**: Adaptive iteration depth for especially hard content words

**The first-principles argument:** On 240 GB/s memory-bound hardware, every byte of weight read costs time. If 65% of tokens can be predicted by a 3MB lookup table living in L2 cache, those tokens cost ~0 ms each. Only the 35% of "genius" tokens pay DRAM costs. Weighted average decode time drops dramatically.

---

## Why This Makes Sense (From Caveman Repo Data)

The caveman skill achieves **65% token reduction** on LLM outputs while maintaining **100% technical accuracy**. A 2026 paper found constraining models to brief responses **improved accuracy by 26 percentage points**.

This means:
1. Most tokens in natural language are LOW ENTROPY — highly predictable from context
2. These tokens don't need deep neural processing — they follow N-gram statistics
3. A model that skips deep processing for these tokens loses nothing and GAINS speed
4. The "unnecessary" tokens ARE the ones Engram was designed to handle: formulaic N-gram patterns

**The mapping:**
- Caveman's "removed words" = articles, prepositions, hedging → **Engram reflex** (hash lookup)
- Caveman's "preserved words" = technical terms, code, content → **Genius path** (full SSM + MoE)

---

## Architecture

```
Token → Embedding (d=1024, tied LM head)
  │
  ├─ Information Classifier (dictionary + small MLP)
  │   → score ∈ [0, 1] per token (0 = pure glue, 1 = pure content)
  │
  ├─ mHC 4-Branch Highway [carries both paths' state]
  │
  ├─ REFLEX PATH (caveman brain):                    # ALL tokens enter here
  │   Engram lookup (bigram/trigram hash tables)
  │   + Tiny gated recurrence (d=256, 6 layers)
  │   → Reflex prediction (for glue tokens, this IS the output)
  │   → Entropy check: if confident → SKIP genius path
  │
  ├─ GENIUS PATH (content brain):                     # ~35% of tokens
  │   Project up: 256 → 1024
  │   Mamba-3 SISO complex SSM (d=1024, 8 layers)
  │   + Adaptive iteration (hard tokens get 2 passes via SCORE damping)
  │   → Project down: 1024 → 256
  │   → Additive correction to reflex state
  │
  ├─ mHC merge + Project up: 256 → 1024
  → Final RMSNorm → LM Head (1024 → 50257)
```

## Token Classification

### Fixed Dictionary (Zero Parameters)

A curated list of ~500 high-frequency, low-information token IDs:

```python
GLUE_TOKENS = {
    # Articles: the, a, an, The, A, An, ...
    # Prepositions: of, in, to, for, on, at, with, by, from, ...
    # Conjunctions: and, or, but, so, yet, ...
    # Auxiliaries: is, are, was, were, be, been, being, has, have, had, ...
    # Pronouns: it, he, she, they, we, I, you, ...
    # Determiners: this, that, these, those, some, any, ...
    # Punctuation: , . ; : ! ? " ' ( ) ...
    # Common filler: the, that, which, who, ...
}
# ~500 tokens covering ~60-65% of token occurrences in English text
```

These ~500 tokens (1% of vocab) account for ~60-65% of token OCCURRENCES in natural text (Zipf's law).

### Soft Scoring (Tiny MLP, Optional Enhancement)

```python
info_score = sigmoid(MLP_tiny(embed(token)))  # 1024 → 64 → 1
is_glue = (token_id in GLUE_TOKENS) or (info_score < theta)
```

The MLP catches context-dependent cases: "bank" is high-info after "river" (content) but lower-info after "the" (common collocation).

## Component Details

### Reflex Path (Caveman Brain)

**ALL tokens pass through here.** It's the minimum processing every token gets.

```python
# 1. Engram lookup (O(1) per token, from L2)
e = engram_lookup(token_ids, K=8, orders=[2,3])     # bigram + trigram
alpha = sigmoid(RMSNorm(h) @ RMSNorm(W_K @ e) / sqrt(d))
h_engram = alpha * W_V @ e                            # gated hash retrieval

# 2. Tiny gated recurrence (d=256, L2-resident)
for layer in reflex_layers:                            # 6 layers, d=256
    h = griffin_recurrence(h + h_engram)               # a·h + √(1-a²)·(i·v)
    h = swiglu_ffn(rmsnorm(h))                         # small FFN (256→512→256)
```

**Reflex path params:** ~5M (recurrence) + ~20M (Engram tables) = ~25M
**Reflex path size:** 50 MB fp16, **6.3 MB int4 → fits in L2!**

For glue tokens: the Engram lookup + tiny recurrence is enough. The bigram "of the" or trigram "in the morning" is directly retrieved. No deep processing needed.

### Genius Path (Content Brain)

**Only ~35% of tokens enter here** — those where the reflex path is uncertain.

```python
if not is_glue or entropy_high:
    h_up = proj_up(h_reflex)                          # 256 → 1024
    
    for layer in genius_layers:                        # 8 Mamba-3 + 8 MoE-FFN
        h_up = mamba3_siso(h_up)                       # complex-valued SSM
        h_up = moe_ffn(rmsnorm(h_up))                  # 8 experts, top-2
    
    # Optional: adaptive second pass for very hard tokens
    if delta_norm > theta_hard:                         # ~10% of tokens
        h_up = score_iterate(h_up, genius_layers)       # (1-d)·h + d·F(h)
    
    h_correction = proj_down(h_up)                     # 1024 → 256
    h_reflex = h_reflex + h_correction                 # additive correction
```

**Genius path params:** ~160M (Mamba-3 + MoE)
**Active per genius token:** ~60M (top-2 of 8 experts)

### mHC Highway (Connecting Both Paths)

The 4-branch mHC highway runs at d=256 (reflex dimension) and carries information between paths:

- Branch 0: Reflex state (always updated)
- Branch 1: Engram knowledge channel (always updated)
- Branch 2: Genius corrections (updated only when genius path fires)
- Branch 3: Context accumulator (smoothed running state)

```python
# mHC at reflex dimension (d=256, n=4)
# Sinkhorn H_res ensures stable cross-branch mixing
stream = H_res @ stream + H_post * sublayer_output    # 4×4 doubly stochastic
```

mHC overhead at d=256: ~6K params per sublayer. Negligible.

## Configuration

| Parameter | Reflex Path | Genius Path |
|-----------|-------------|-------------|
| d_model | 256 | 1024 |
| n_layers | 6 | 16 (8 Mamba-3 + 8 MoE-FFN) |
| Mixer | Griffin gated recurrence | Mamba-3 SISO complex SSM |
| FFN | SwiGLU (256→512→256) | MoE (8E, top-2, d_ffn=1024) |
| Engram | 2 bigram + 1 trigram tables | — |
| mHC branches | 4 (d=256) | — (operates on projected d=1024) |
| Activation | 100% of tokens | ~35% of tokens |
| Adaptive depth | — | Optional SCORE iteration for hardest ~10% |

| Global | Value |
|--------|-------|
| vocab_size | 50257 |
| d_embedding | 1024 |
| block_size | 1024 |
| Engram hash heads | 8 |
| Engram table rows | 262K (bigram ×2) + 131K (trigram) |
| Engram d_engram | 128 |

## Parameter Count

| Component | Params |
|-----------|--------|
| **Token embedding (50257 × 1024, tied)** | **51.5M** |
| Embed → reflex proj (1024 → 256) | 262K |
| **Reflex path (6 layers, d=256)** | **~5M** |
| **Engram tables + projections** | **~22M** |
| Reflex → genius proj (256 → 1024) | 262K |
| **Genius path:** | |
| — 8 Mamba-3 SISO layers (d=1024, d_inner=1536) | ~74M |
| — 8 MoE-FFN layers (8 experts, d_ffn=1024) | ~80M |
| Genius → reflex proj (1024 → 256) | 262K |
| Reflex → embed proj (256 → 1024) | 262K |
| mHC overhead (all layers) | ~0.2M |
| Info classifier MLP | 65K |
| Final RMSNorm | 1K |
| **TOTAL** | **~234M** |
| **Active per glue token** | **~27M** (reflex + Engram) |
| **Active per content token** | **~87M** (reflex + Engram + genius top-2 MoE) |

### Weight Sizes

| Component | fp16 | int4 |
|-----------|------|------|
| Reflex path + Engram | 54 MB | **~7 MB (L2 BORDERLINE!)** |
| Genius path | 308 MB | 77 MB |
| Embedding + LM head | 103 MB | 26 MB |

**Critical optimization:** Quantize Engram tables to int8 (not int4 — need precision for knowledge). Reflex recurrence in int4. Combined: ~6 MB. **Fits in L2.**

## Decode Speed (Strix Halo)

### fp16

| Token type | Fraction | DRAM reads | Time |
|-----------|----------|-----------|------|
| Glue (reflex only) | ~65% | Reflex from L2 (~0 MB) + LM head (103 MB) | ~0.81 ms |
| Content (reflex + genius) | ~35% | Reflex (L2) + Genius (174 MB active) + LM head (103 MB) | ~1.83 ms |
| **Weighted average** | | | **~1.17 ms = ~855 tok/s** |

### int4

| Token type | Fraction | DRAM reads | Time |
|-----------|----------|-----------|------|
| Glue (reflex only) | ~65% | Reflex (**L2!**) + LM head (26 MB) | ~0.30 ms |
| Content (reflex + genius) | ~35% | Reflex (L2) + Genius (44 MB active) + LM head (26 MB) | ~0.61 ms |
| **Weighted average** | | | **~0.41 ms = ~2439 tok/s** |

### With int4 LM head + aggressive Engram (more glue tokens handled by lookup)

If we push glue ratio to 75% (broader dictionary + MLP enhancement):

| Token type | Fraction | Time |
|-----------|----------|------|
| Glue | 75% | 0.30 ms |
| Content | 25% | 0.61 ms |
| **Weighted** | | **~0.38 ms = ~2632 tok/s** |

### Comparison

| Model | Params | Active (avg) | fp16 tok/s | int4 tok/s |
|-------|--------|-------------|-----------|-----------|
| GPT-2 124M | 124M | 124M | ~198 | — |
| Spectral Hydra | 244M | 244M | ~285 | ~714 |
| Resonant Loop | 59M unique | 59M | ~694 | ~1613 |
| Chimera Engram B | 231M | 63M | ~794 | ~2000 |
| Dual Cortex | 231M | 10-165M | ~641 | ~1961 |
| **Genius Caveman** | **234M** | **27-87M** | **~855** | **~2439** |

## Training

### Token Classification Setup

Pre-compute `is_glue` for all training tokens using the fixed dictionary. No learning needed for this — Zipf's law guarantees the top ~500 tokens cover ~65% of occurrences.

### Soft Gating for Training

During training, ALL tokens go through BOTH paths (for gradient flow), weighted by `info_score`:

```python
h_reflex = reflex_path(embed, engram_lookup)           # always
h_genius = genius_path(proj_up(h_reflex))              # always during training
info = info_score(token_ids)                           # 0=glue, 1=content
h_final = (1-info) * h_reflex + info * (h_reflex + proj_down(h_genius))
```

At inference: hard routing (skip genius path entirely for glue tokens).

### Phase Training

1. **Phase 1 (40%):** Train reflex path + Engram only (no genius path). The caveman brain learns the basics.
2. **Phase 2 (30%):** Add genius path, soft gating. Both paths learn to cooperate.
3. **Phase 3 (20%):** Add MoE to genius path (was dense FFN in Phase 2). Expert routing learns.
4. **Phase 4 (10%):** Enable adaptive depth (SCORE iteration) for hardest tokens. Fine-tune info threshold.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Dataset | OpenWebText ~100M tokens |
| Batch | 48 × 512 = 24K tok, grad_accum=2 (48K effective) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Engram tables | **Adam, 5× LR, zero weight decay** (per Engram paper) |
| Base LR | 8e-4 cosine → 8e-5 |
| mHC α init | 0.01 (per mHC paper) |
| Sinkhorn iters | 20 (per mHC paper) |
| Mamba-3 RoPE | **Data-dependent** (per Mamba-3 paper, non-negotiable) |
| Weight decay | 0.1 (backbone) |
| Grad clip | 1.0 |
| Precision | fp16 mixed + fp32 scan |
| Est. throughput | ~35M tokens in 15 min |

## HIP Kernels

**Reuse (7 existing kernels!):**
| Kernel | Usage |
|--------|-------|
| `fused_residual_add_rmsnorm` (6.6×) | Both paths |
| `silu_gate_mul` (1.6×) | Reflex FFN + MoE experts |
| `moe_gating` (3.5×) | Genius path routing |
| `rotary_embedding` (3.7×) | Mamba-3 data-dependent B,C rotation |
| `prefix_scan` (8.4×) | Adapt for both Griffin scan (reflex) + complex scan (genius) |
| `cross_entropy` (1.8×) | Output loss |
| `dequantize_int4` (16.3×) | All projections |

**New:**
1. **Fused Engram+Gate+Reflex** — hash lookup + gating + Griffin recurrence in one kernel. The reflex path's hot loop.
2. **Fused Mamba-3 Decode Step** — complex SSM element-wise ops. Genius path decode.
3. **Fused Info Router** — dictionary check + MLP + threshold → skip genius. Saves kernel launches for 65% of tokens.

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Fixed glue dictionary is too rigid | LOW | Dictionary covers top-500 tokens (~65% of occurrences by Zipf); MLP catches edge cases |
| Reflex path too weak for some "glue" tokens in complex contexts | MEDIUM | Engram provides context-aware gating (alpha→0 for misleading lookups); soft training lets both paths contribute |
| Genius path gradient starvation (only ~35% of tokens) | HIGH | Phase 1-2 trains all tokens through both paths; genius path fully trained before hard routing |
| Mamba-3 + MoE + Engram + mHC too many components | HIGH | Phase training decouples; each component validated independently in papers |
| Glue/content boundary is fuzzy | MEDIUM | Soft gating during training; hard routing only at inference; MLP refines dictionary-based classification |
| Adaptive depth adds complexity | LOW | Optional — disable if training budget too tight; model works without it |

## Success Criteria

1. Loss < 4.5 in 15 min
2. Glue tokens (65%+) route to reflex path at convergence
3. Engram gates activate on N-gram patterns (high alpha for "of the", "in the")
4. Content tokens route to genius path and show lower loss than reflex-only
5. Decode > 800 tok/s fp16, > 2000 tok/s int4 on Strix Halo
6. Per-token efficiency ≥ GPT-2 124M at equal tokens seen
7. **Caveman metric**: model achieves same perplexity on content words as a full-compute baseline, while spending 3× less total compute

## Implementation Roadmap

1. Curate glue token dictionary (~500 tokens from tiktoken GPT-2 vocab)
2. Implement reflex path: Engram + Griffin recurrence (d=256, 6 layers)
3. Implement genius path: Mamba-3 SISO + dense FFN (d=1024, 16 layers)
4. Implement mHC 4-branch highway at d=256
5. Implement soft info gating + projection bridges
6. Phase 1: train reflex + Engram only (40% of budget)
7. Phase 2: add genius path with soft gating (30%)
8. Phase 3: replace dense FFN with MoE in genius path (20%)
9. Phase 4: enable adaptive depth + tune info threshold (10%)
10. Fused Engram+Reflex HIP kernel
11. Fused Mamba-3 decode HIP kernel
12. Int4 quantization + L2 validation for reflex path
13. Decode benchmark: measure glue/content ratio, latency distribution
14. Compare to all previous hypotheses + GPT-2 baseline

---

*"Caveman no waste brain on 'the'. Caveman KNOW 'the'. Caveman save brain for big think. Caveman... genius."*

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
| `torch.compile(mode="default")` | +90% MFU — dual-path + routing may limit fusion | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x, moe_gating 3.5x | Available |
| `causal-conv1d` in GatedConv | 10x conv speedup | Available |
| `mamba-ssm` for Genius path | 5.6x scan speedup in Mamba layers | Available |
| Reflex path L2 partial fit | Tiny recurrence + Engram in reflex path | By design |
| mHC routing overhead | 4-branch routing adds ~2% compute | By design |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~10,500 tok/s (25% MFU)
**Tokens in 45 min:** ~28.4M (1.8 BabyLM epochs)
**Ranking:** #15 of 22 architectures
