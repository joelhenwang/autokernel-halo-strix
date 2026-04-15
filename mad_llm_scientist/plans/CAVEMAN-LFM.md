---
title: "CAVEMAN-LFM"
domain: architectures
type: plan
status: stale
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %caveman-lfm]
---

# CAVEMAN LFM

**LFM2-Validated Gated Conv + Griffin Recurrence + Engram Knowledge + Caveman Routing**

## Hypothesis

Take LFM2's hardware-search-validated architecture (gated short conv + sparse global mixing + large SwiGLU FFN), replace the 6 GQA attention layers (0.05× on Strix Halo) with Griffin gated recurrence, add Engram hash-indexed knowledge tables, and add information-weighted routing where ~65% of tokens (glue words) skip recurrence layers via a cheap bypass.

**Grounded in:** LFM2 technical report (2511.23404) — their hardware-in-the-loop search tested SSMs, linear attention, and complex hybrids, then rejected them all in favor of gated conv + attention. We keep the validated conv, replace the invalid-for-our-hardware attention.

**Our innovations over LFM2:**
1. Griffin recurrence replacing GQA (element-wise, no matmul for mixing)
2. Engram tables (12.8M) for O(1) N-gram knowledge lookup
3. Caveman routing: 65% of tokens skip recurrence via cheap bypass
4. Decay bias spectrum in Griffin dims (fast/medium/slow)

---

## Architecture

```
Tokens → Embedding (d=1024, tied LM head, vocab=50257)
  → 16 layers (LFM2-derived pattern):

  Layer:  C  C  R* C  C  R  C  C  R  C* R  C  R  C  R  C
          0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
          (C=GatedConv, R=GriffinRecurrence, *=+Engram)

  → Final RMSNorm → LM Head
```

- **10 gated short conv blocks**: ALL tokens flow through (LFM2's core mixer)
- **6 Griffin recurrence blocks**: content tokens (35%) get full recurrence; glue tokens (65%) get cheap linear bypass
- **Engram** at layers 2 and 9
- **SwiGLU FFN** at every layer

## Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| d_model | 1024 | LFM2-350M |
| ffn_inner | 2240 (SwiGLU, 2.19×) | Budget-constrained (LFM2 uses 6656/6.5×) |
| n_layers | 16 (10 conv + 6 recurrence) | LFM2-350M pattern |
| conv_kernel | 3 | LFM2-350M |
| vocab_size | 50257 | tiktoken GPT-2 constraint |
| block_size | 2048 | |
| rope_theta | 500,000 | |
| weight_tying | yes | LFM2 principle |
| Engram hash heads | 4 | |
| Engram d_engram | 64 | |
| Engram tables | bigram 65K×64, trigram 32K×64 (×2 layers) | |
| Glue dictionary | ~500 token IDs (~65% of occurrences) | Zipf's law |
| **Total params** | **~248M** | |

## Component Details

### Gated Short Conv Block (LFM2 Validated)

```python
# Pre-mixer RMSNorm + gated conv
x_norm = rmsnorm(x)
B, C, h_tilde = linear(x_norm, 1024 → 3072).chunk(3)  # three 1024-dim projections
y = B * h_tilde                    # element-wise gate
z = causal_conv1d(y, k=3)          # depthwise, 1024 channels
out = linear(C * z, 1024 → 1024)   # output gate + projection
x = x + out                        # residual

# Pre-FFN RMSNorm + SwiGLU
x = x + swiglu_ffn(rmsnorm(x))     # 1024 → 2240 → 1024
```

All element-wise + tiny conv. Zero matmul for mixing. Perfect for Strix Halo.

### Griffin Gated Recurrence Block (Replacing GQA)

```python
x_norm = rmsnorm(x)

# --- Token routing ---
is_glue = GLUE_DICT[token_ids]                        # fixed dictionary, ~65% True

# Glue bypass (cheap)
bypass = linear(x_norm, 1024 → 1024)                  # single projection

# Content: full Griffin recurrence (bounded by construction)
a = sigmoid(linear(x_norm, 1024 → 1024) + decay_bias) # decay gate + spectrum bias
i = sigmoid(linear(x_norm, 1024 → 1024))              # input gate
v = linear(x_norm, 1024 → 1024)                        # value
h = a * h_prev + sqrt(1 - a**2) * (i * v)             # Griffin coupling
rec_out = linear(h, 1024 → 1024)                       # output projection

# Merge
mixer_out = where(is_glue, bypass, rec_out)            # hard routing at inference
x = x + mixer_out

# SwiGLU FFN (ALL tokens)
x = x + swiglu_ffn(rmsnorm(x))
```

**Training:** Soft routing — `out = (1-info)*bypass + info*rec_out` where `info = sigmoid(MLP(embed))`.
**Inference:** Hard routing — glue tokens skip recurrence entirely.

**Decay bias spectrum** (per-dimension, 1024 dims):
- Dims 0–255: bias=-2.2 → σ≈0.10 (fast decay, local patterns)
- Dims 256–767: bias=0.0 → σ=0.50 (medium decay)
- Dims 768–1023: bias=+4.6 → σ≈0.99 (slow decay, long-range)

**Parallel scan for training:** Associative operator `(a₂·a₁, a₂·b₁+b₂)` on pairs `(a, √(1-a²)·i·v)`. Adapt `kernels/hip/prefix_scan.py`. fp32 accumulation.

### Engram Hash-Indexed Tables (at layers 2 and 9)

```python
# Per token: bigram + trigram lookup
bg = bigram_table[hash(token[t-1], token[t])]       # K=4 heads, d=64
tg = trigram_table[hash(token[t-2], token[t-1], token[t])]

# Context-aware gating
e = concat(bg.mean(0), tg.mean(0))                   # (128,)
alpha = sigmoid(linear(rmsnorm(h)) @ rmsnorm(linear(e)))
x = x + alpha * linear(e, 128 → 1024)
```

- **Training:** Tables use Adam, **5× base LR, zero weight decay** (per Engram paper)
- **Tables are learned** via backprop (hash is fixed, embeddings are differentiable)
- **Mechanistic effect:** Doubles effective depth (Layer 5 ≈ Layer 12 of baseline per CKA analysis in Engram paper)

### Glue Token Dictionary

~500 token IDs from tiktoken GPT-2 vocab: articles, prepositions, conjunctions, auxiliaries, pronouns, determiners, punctuation. By Zipf's law, covers ~60-65% of token occurrences.

Stored as a boolean tensor `(50257,)` — zero parameters, O(1) lookup.

## Parameter Budget

| Component | Params |
|-----------|--------|
| Embedding (50257×1024, tied) | 51.5M |
| 10 conv layers: gated conv projs + conv1d | 42.0M |
| 6 rec layers: Griffin projs + bypass + decay_bias | 31.5M |
| 16 SwiGLU FFN layers (1024→2240→1024) | 110.1M |
| Engram (2 layers: tables + projections) | 12.8M |
| Info classifier MLP + final RMSNorm | 0.1M |
| **TOTAL** | **~248M** |

## Training

### Phase Training

1. **Phase 1 (50% of steps):** Conv + FFN only. Recurrence uses bypass for all tokens. Engram off. Model learns basic patterns.
2. **Phase 2 (35%):** Enable Griffin recurrence with soft routing. Enable Engram. Both paths active.
3. **Phase 3 (15%):** Anneal routing temperature (1.0→0.1). Add λ_compute penalty. Target 65% glue ratio.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Dataset | OpenWebText ~100M tokens |
| Batch | 48×512, accum=2 (48K effective) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| LR | 8e-4 cosine → 8e-5, warmup 100 |
| Engram table LR | **5× base, Adam, zero weight decay** |
| decay_bias LR | 0.1× base (preserve spectrum) |
| Weight decay | 0.1 (backbone) |
| Grad clip | 1.0 |
| Precision | fp16 mixed + fp32 scan |
| Est. throughput | ~27M tokens in 15 min |

### Optional: Knowledge Distillation

If teacher available (GPT-2 Medium 355M pre-trained):
```
loss = 0.7 * CE(student, targets) + 0.3 * KL(student, teacher, T=2.0)
```

## Decode Speed (Strix Halo)

| Mode | Glue tokens (65%) | Content tokens (35%) | Weighted avg |
|------|-------------------|---------------------|-------------|
| fp16 | ~460 MB reads → 2.71 ms | ~496 MB → 2.92 ms | **~2.78 ms = ~360 tok/s** |
| int4 | ~115 MB → 0.68 ms | ~124 MB → 0.73 ms | **~0.70 ms = ~1429 tok/s** |

### Comparison

| Model | Params | fp16 tok/s | int4 tok/s |
|-------|--------|-----------|-----------|
| GPT-2 124M | 124M | ~198 | — |
| LFM2-350M on Strix Halo | 350M | **~120** (attention kills it) | ~300 |
| Spectral Hydra | 244M | ~285 | ~714 |
| Resonant Loop | 59M | ~694 | ~1613 |
| Genius Caveman | 234M | ~855 | ~2439 |
| **CAVEMAN LFM** | **248M** | **~360** | **~1429** |

**vs LFM2 on Strix Halo: 3× faster fp16, 4.8× faster int4** — because we eliminated the attention bottleneck.

**vs Genius Caveman: slower but MUCH simpler** — 3 component types (conv, recurrence, Engram) vs 6+. More likely to train successfully in 15 min.

## HIP Kernels

**Reuse (5 existing):**

| Kernel | Speedup | Usage |
|--------|---------|-------|
| `fused_residual_add_rmsnorm` | 6.6× | 32 uses (16 pre-mixer + 16 pre-FFN) |
| `silu_gate_mul` | 1.6× | 16 SwiGLU FFNs |
| `prefix_scan` | 8.4× | Adapt for Griffin associative scan |
| `cross_entropy` | 1.8× | Output loss |
| `dequantize_int4` | 16.3× | All projections in int4 mode |

**New (4 to write, priority order):**

1. **Fused Griffin Recurrence Step** (decode) — sigmoid(a+bias), sigmoid(i), sqrt(1-a²), mul, add. Adapt `silu_gate_mul.py`. Est. 6-10×.
2. **Griffin Associative Scan** (training) — adapt `prefix_scan.py` for `(a₂·a₁, a₂·b₁+b₂)`. Complex pairs in fp32 LDS.
3. **Fused Gated Conv Block** — B⊙h̃ + conv_step(k=3) + C⊙z in one kernel. Est. 4-8×.
4. **Fused Engram Lookup+Gate** — hash(int) + gather + sigmoid + weighted sum. Est. 3-5×.

**Strategy:** `torch.compile(mode="reduce-overhead")` for training. Custom HIP kernels for inference decode.

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| ffn_inner=2240 too small (LFM2 uses 6656) | HIGH | Conv blocks have own 3072-dim projections adding capacity. Reduce Engram to increase FFN if needed. |
| Griffin loses long-range vs attention | MEDIUM | Decay bias spectrum (slow dims ≈ 0.99 retention) + Engram for factual recall. Fallback: 1 sliding-window attention at layer 8. |
| 15 min too short for 248M model | HIGH | Focus on loss-vs-time trend. Optional knowledge distillation from GPT-2 Medium. |
| Glue routing hurts training gradients | MEDIUM | Phase 1-2 use soft routing (full gradient flow). Hard routing only at inference. |
| Engram hash collisions | LOW | 4 heads with distinct primes. Context-aware gating suppresses noise. |

## Success Criteria

1. Loss < 4.5 in 15 min (random ~10.8)
2. 60-70% of tokens route to glue bypass at convergence
3. Engram gates activate selectively (high alpha on N-gram patterns)
4. Decode > 300 tok/s fp16 on Strix Halo
5. Per-token efficiency ≥ GPT-2 124M at equal tokens seen
6. Griffin recurrence state shows temporal specialization (fast dims forget, slow dims retain)

## Implementation Roadmap

1. Curate glue token dictionary from tiktoken GPT-2 vocab (~500 tokens)
2. Implement GatedConvBlock (LFM2 pattern: B⊙h̃, conv_3, C⊙z)
3. Implement GriffinRecurrenceBlock (decay+input+value gates, √(1-a²) coupling, bypass)
4. Implement Engram module (hash tables, context-aware gating)
5. Implement CavemanLFM model class (16 layers, layer_types pattern, routing)
6. Verify param count = ~248M, test forward/backward pass
7. Phase 1 training: conv+FFN only (50% budget)
8. Phase 2: enable recurrence + Engram + soft routing (35%)
9. Phase 3: anneal routing, add compute penalty (15%)
10. Fused Griffin recurrence HIP kernel
11. Griffin associative scan HIP kernel
12. Int4 quantization + decode benchmark
13. Compare to GPT-2 + LFM2 baseline estimates

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x) — applies to every residual+norm pair (32 instances)
- **silu_gate_mul** (1.6x) — applies to SwiGLU FFN (16 instances)
- **cross_entropy** (1.8x) — applies to loss computation
- **prefix_scan** (8.4x) — adaptable for Griffin associative scan
- Apply all via `autokernel.optimize(model, training=True)`

### Griffin Recurrence: Use Chunked Linear Recurrence
The Griffin operator `(a₂·a₁, a₂·b₁+b₂)` is the same associative scan as Mamba SSM. **Do NOT use sequential loops or `torch.associative_scan`** — both are equally slow on gfx1151 (~1.3K tok/s). Use **chunked linear recurrence** (chunk_size=64):
- Reference implementation: `models/amadeus.py:selective_scan_chunked`
- The `sqrt(1-a²)` coupling term can be precomputed per chunk
- Expected: ~10ms/layer for d=1024, batch=8, seq=512

### Throughput Estimates (corrected from AMADEUS baseline)
- **Eager mode (no compile/autokernel):** ~6-8K tok/s, ~16-20% MFU
- **With autokernel patterns:** ~7-9K tok/s, ~18-22% MFU
- **Token budget:** 15 min = 5.4-7.2M tokens | 45 min = 16-22M tokens | 120 min = 43-58M tokens
- Previous estimate "~27M tokens in 15 min" was optimistic

### MFU Analysis
- SwiGLU FFN dominates (56% of params, ~80% MFU — weight-bandwidth-bound)
- Griffin recurrence is element-wise (~95% MFU)
- Caveman routing (`where(is_glue, bypass, rec_out)`) is negligible cost
- **Overall training MFU: 65-75%** (limited by FFN weight reads)

### External Kernel Integration (verified 2026-04-10)

- **GatedConv:** causal-conv1d (10x vs nn.Conv1d) — auto-used if installed, try/except fallback
- **SSM scan (if Mamba path used):** mamba-ssm selective_scan_fn (5.6x, 0.32ms) — drop-in upgrade
- **Griffin scan:** Chunked linear recurrence remains primary. FLA HGRN (0.40ms) as alternative.

---

## Possible Optimizations & Throughput Estimate

**Baseline (estimated):** ~5,500 tok/s eager (14% MFU)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +80% MFU — LFM2-style hybrid compiles reasonably | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| `causal-conv1d` in GatedConv | 10x conv speedup (10 conv layers) | Available |
| FLA HGRN for Griffin recurrence | 0.40ms Triton kernel (6 Griffin layers) | Available |
| Engram lookup optimization | Pre-hash N-gram indices | Possible |
| Caveman routing overhead | ~3% compute for routing decisions | By design |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~10,000 tok/s (25% MFU)
**Tokens in 45 min:** ~27.0M (1.7 BabyLM epochs)
**Ranking:** #16 of 22 architectures
