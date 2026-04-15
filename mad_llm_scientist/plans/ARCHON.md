---
title: "ARCHON"
domain: architectures
type: plan
status: stale
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %archon]
---

# ARCHON

**The Supreme Architecture — Quality-Maximized Convergence of 11 Hypotheses and 15+ Papers**

## Hypothesis

A 250M parameter language model that combines **every quality-maximizing ingredient** backed by paper evidence — parallel hybrid mixing, mHC multi-branch residuals, meta tokens, Engram knowledge tables, Griffin gated recurrence, multi-token prediction training, and knowledge distillation — can rival LFM2.5-350M-Base quality despite being 100M params smaller and attention-free.

Every ingredient is selected for its **quality impact evidence**, not speed or novelty. This is the rational convergence of 11 prior hypotheses and 15+ papers into one architecture.

---

## Ingredients and Evidence

| Ingredient | Quality Evidence | Source Paper |
|-----------|-----------------|-------------|
| Parallel hybrid (conv+rec in same layer) | +1.1% commonsense, +4.7% recall at 300M | Hymba, Falcon-H1, Meta Hybrid Analysis |
| mHC 4-branch residual (Sinkhorn) | +7% BBH at 27B, provably stable | mHC (2512.24880) |
| Meta tokens (128 learned embeddings) | +1.4% at 300M for 131K params | Hymba (2411.13676) |
| Engram N-gram tables | Doubles effective depth, stores 56-71% of factual knowledge | Engram (2601.07372) |
| Griffin gated recurrence (√(1-a²)) | Proven at 14B, bounded state norm | Griffin (2402.19427) |
| MTP training (4 prediction heads) | +12% HumanEval, improves backbone quality | Meta MTP (2404.19737) |
| Gated short conv (k=3) | Validated optimal by hardware-in-the-loop search | LFM2 (2511.23404) |
| Decay bias spectrum | Multi-scale temporal dynamics across dims | Original (Spectral Hydra) |
| Knowledge distillation | Tempered decoupled Top-K | LFM2 (2511.23404) |

---

## Architecture

```
[128 Meta Tokens] + Input Tokens → Embedding (d=1024, tied LM head, vocab=50257)
  │
  → mHC 4-Branch Expansion (1024 → 4×1024 residual stream)
  │
  → 16 Parallel Hybrid Blocks:
  │     mHC H_pre readout (Sinkhorn) → d=1024
  │     RMSNorm
  │     ┌──────────────────────────────────┐
  │     │ Conv Channels (12)  ║  Griffin   │
  │     │ B⊙h̃ → Conv₃ → C⊙z ║  Recurrence│
  │     │ d_conv=768          ║  (4 heads) │
  │     │                     ║  d_rec=256 │
  │     └──────────────────────────────────┘
  │     Concat (768+256=1024) → Output Proj (1024→1024)
  │     mHC H_post write + H_res cross-branch mixing
  │     + Residual
  │     RMSNorm → SwiGLU FFN (1024→1792→1024)
  │     + Residual
  │     (+ Engram injection at layers 2 and 9)
  │
  → mHC Final Readout (4×1024 → 1024)
  → Final RMSNorm
  → 4 MTP Heads → Shared LM Head (tied embedding)
```

## Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| d_model | 1024 | LFM2-350M |
| d_conv (conv channels) | 768 (12 × 64) | Falcon-H1 3:1 ratio |
| d_rec (recurrence channels) | 256 (4 × 64) | Falcon-H1 3:1 ratio |
| n_layers | 16 | LFM2-350M |
| ffn_inner | 1792 (SwiGLU, 1.75×) | Budget-constrained |
| conv_kernel | 3 | LFM2-350M |
| mHC branches | 4 | mHC paper |
| mHC Sinkhorn iterations | 20 | mHC paper |
| mHC α init | 0.01 | mHC paper |
| meta_tokens | 128 | Hymba |
| vocab_size | 50257 | tiktoken GPT-2 |
| block_size | 1024 | |
| RoPE theta | 500,000 | |
| weight_tying | yes | LFM2 |
| MTP heads | 4 (shared LM head + 3 adapters) | Meta MTP |
| Engram d_engram | 96 | |
| Engram hash heads | 4 per order | |
| Engram layers | 2 and 9 | Engram paper |
| Bigram table | 65,536 × 96 | |
| Trigram table | 32,768 × 96 | |

## Component Details

### 1. Parallel Hybrid Block (Every Layer)

Both conv and recurrence process the SAME input in parallel. Outputs concatenated.

```python
x_norm = rmsnorm(x)

# CONV CHANNELS (12 of 16, d=768) — LFM2 validated
B, C, h_tilde = linear(x_norm, 1024 → 3×768).chunk(3)
y = B * h_tilde                          # element-wise gate
z = causal_conv1d(y, k=3)                # depthwise conv
conv_out = C * z                          # (B, T, 768)

# RECURRENCE CHANNELS (4 of 16, d=256) — Griffin with decay spectrum
a = sigmoid(linear(x_norm, 1024→256) + decay_bias)   # per-dim bias for multi-scale
i = sigmoid(linear(x_norm, 1024→256))
v = linear(x_norm, 1024→256)
h = a * h_prev + sqrt(1 - a**2) * (i * v)            # bounded by construction
rec_out = h                                            # (B, T, 256)

# CONCAT + OUTPUT
out = linear(concat(conv_out, rec_out), 1024→1024)
```

### 2. mHC 4-Branch Residual (Wrapping Each Sublayer)

```python
x_bar = rmsnorm(flatten(stream))                            # (B, T, 4096) → normalized
H_pre  = sigmoid(0.01 * (x_bar @ phi_pre) + b_pre)         # readout weights (4,)
H_post = 2 * sigmoid(0.01 * (x_bar @ phi_post) + b_post)   # write-in weights (4,)
H_res  = sinkhorn(exp(0.01 * mat(x_bar @ phi_res) + b_res), iters=20)  # 4×4 doubly stochastic

x = H_pre @ stream                        # weighted sum: 4 branches → 1024
y = sublayer(x)                            # parallel hybrid or FFN
stream = H_res @ stream + H_post * y       # cross-branch mixing + write-in
```

**Critical from paper:** H_res provides MAJORITY of quality gain. Sinkhorn on exp(logits). α=0.01 init.

### 3. Meta Tokens (128 Learned Embeddings)

```python
meta = self.meta_tokens.expand(B, 128, 1024)    # 131K params
x = concat(meta, embed(tokens))                   # (B, 128+T, 1024)
# All layers process meta+tokens together
# Output: discard meta positions
logits = lm_head(output[:, 128:, :])
```

### 4. Engram (Layers 2 and 9)

```python
bg = bigram_table[hash(token[t-1], token[t])]     # K=4 heads, d=96
tg = trigram_table[hash(token[t-2:t+1])]           # K=4 heads, d=96
e = concat(bg.mean(0), tg.mean(0))                 # (192,)
alpha = sigmoid(rmsnorm(h) @ rmsnorm(W_K @ e) / sqrt(d))
x = x + alpha * W_V @ e                            # gated injection
```

**Training recipe from paper:** Adam, 5× base LR, zero weight decay. Conv zero-initialized.

### 5. Griffin Decay Bias Spectrum

Per-dimension initialization across the 256 recurrence dims:

| Dims | Bias | σ(bias) | Role |
|------|------|---------|------|
| 0–63 | -2.2 | ~0.10 | Fast decay: local patterns |
| 64–191 | 0.0 | 0.50 | Medium: clause-level |
| 192–255 | +4.6 | ~0.99 | Slow: entity/topic tracking |

**LR for decay_bias:** 0.1× base (preserve spectrum, from Spectral Hydra design).

### 6. MTP Training Heads

```python
# Head 1: standard next-token (no adapter)
logits_1 = lm_head(rmsnorm(h))

# Heads 2-4: small adapters + shared LM head
for k in range(1, 4):
    h_k = h + mlp_adapter_k(concat(h, embed(argmax(logits_prev))))
    logits_{k+1} = lm_head(rmsnorm(h_k))
```

**Training loss:**
```python
loss = sum(w[k] * CE(logits_k, targets[t+k]) for k in range(4))
# weights = [1.0, 0.5, 0.3, 0.2]
```

**Meta's finding:** MTP training improves the BACKBONE even for standard next-token prediction. The multi-step signal forces the model to encode more predictive information.

### 7. Knowledge Distillation

If teacher available (GPT-2 Medium 355M):
```python
loss_total = 0.7 * CE(student, targets) + 0.3 * KL(student_soft, teacher_soft, T=2.0)
```

From LFM2: tempered decoupled Top-K avoids support mismatch. Use Top-K=32 truncation on teacher logits.

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×1024, tied) | 51,463,168 |
| Meta tokens (128×1024) | 131,072 |
| **Per layer:** | |
| — Conv projections (1024→3×768 + 768→) | ~3,500,000 |
| — Griffin projections (4 × 1024→256 + outproj share) | ~1,300,000 |
| — Output projection (1024→1024) | ~1,049,600 |
| — SwiGLU FFN (1024→1792→1024) | ~5,505,024 |
| — mHC (phi_pre, phi_post, phi_res, biases) | ~50,000 |
| — RMSNorm × 2 + decay_bias | ~2,304 |
| — **Layer total** | **~11,406,928** |
| **16 layers** | **~182,510,848** |
| Engram tables (bigram 65K×96 + trigram 32K×96) | 9,437,184 |
| Engram projections (2 layers × W_K, W_V, gate) | ~2,500,000 |
| MTP adapters (3 × concat(1024,1024)→512→1024) | ~4,700,000 |
| mHC initial expansion + final readout | ~100,000 |
| Final RMSNorm | 1,024 |
| **GRAND TOTAL** | **~250,843,296 (~251M)** |

## Training

### Phase Training (15-minute budget, ~25M tokens)

| Phase | Budget | Components Active | What Learns |
|-------|--------|-------------------|-------------|
| 1 (35%) | 5.25 min | Backbone (conv+rec+FFN) + mHC. No Engram, no meta, no MTP. | Core language model |
| 2 (25%) | 3.75 min | + Meta tokens + Engram. Both knowledge tiers activate. | Knowledge injection |
| 3 (25%) | 3.75 min | + MTP heads (4). Multi-token prediction signal. | Backbone improvement from MTP |
| 4 (15%) | 2.25 min | + Knowledge distillation from teacher (if available). | Teacher knowledge transfer |

### Optimizer Groups

| Group | Optimizer | LR | Weight Decay |
|-------|-----------|-----|-------------|
| Backbone (conv, rec, FFN, norms, outproj) | AdamW | 8e-4 cosine → 8e-5 | 0.1 |
| mHC (phi, biases) | AdamW | 8e-4 | 0.1 |
| Meta tokens | AdamW | 8e-4 | 0.01 |
| Engram tables | **Adam** | **4e-3 (5× base)** | **0** |
| Engram projections (W_K, W_V) | AdamW | 8e-4 | 0.1 |
| Decay bias | AdamW | **8e-5 (0.1× base)** | 0 |
| MTP adapters | AdamW | 8e-4 | 0.1 |

### Other Hyperparams

| Parameter | Value |
|-----------|-------|
| Dataset | OpenWebText ~100M tokens |
| Batch | 32 × 512 = 16K, accum=4 (64K effective) |
| Warmup | 100 steps |
| Grad clip | 1.0 |
| Precision | fp16 mixed + fp32 for scans and Sinkhorn |
| Gradient checkpointing | Every 4 layers |
| KD teacher | GPT-2 Medium 355M (optional) |
| KD temperature | 2.0 |
| KD Top-K | 32 |

## Decode Speed (Strix Halo)

| Mode | Weight Reads | Kernel Overhead | Total | Throughput |
|------|-------------|-----------------|-------|------------|
| fp16 | ~502 MB | ~0.4 ms (16L × 6 kernels × 5μs) | ~3.4 ms | **~294 tok/s** |
| int4 | ~126 MB | ~0.4 ms | ~1.14 ms | **~877 tok/s** |
| int4 + MTP burst (avg 2×) | ~126 MB × 1.05 | ~0.4 ms | ~1.2 ms for 2 tok | **~1667 eff tok/s** |

Not the fastest design (that's Ternary Reflex at ~2500). But **the highest QUALITY design** at 250M.

## HIP Kernels

**Reuse (5):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), prefix_scan (8.4×), cross_entropy (1.8×), dequantize_int4 (16.3×)

**New (5):**
1. **Fused Parallel Hybrid Block** — conv gate + conv1d_step + rec gate + concat + outproj. The hot inner loop.
2. **Griffin Associative Scan** — adapt prefix_scan for `(a₂·a₁, a₂·b₁+b₂)`. fp32 in LDS.
3. **Fused mHC Read-Write** — Sinkhorn 4×4 (in registers) + sigmoid readout + write-in. Per sublayer.
4. **Fused Engram Lookup+Gate** — hash + gather + rmsnorm + sigmoid + weighted sum.
5. **Fused MTP Head Chain** — 3 sequential adapters, each L2-resident.

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Too many components → training instability | HIGH | Phase training isolates components. Each is proven independently. mHC Sinkhorn prevents residual instability. |
| ffn_inner=1792 too small | HIGH | Engram provides 12M dedicated factual capacity. Conv channels have own 3×768 projections. Meta tokens add compressed knowledge. Together compensate for smaller FFN. |
| 15 min insufficient for this complexity | HIGH | Phase training. MTP improves backbone efficiency per token. KD provides richer gradients. Focus on per-token learning curves, not absolute quality. |
| mHC + parallel hybrid + Engram interactions | MEDIUM | mHC and Engram synergize (ablation: multi-branch gating is Engram's #1 feature). Parallel hybrid is orthogonal to mHC (operates within each sublayer). |
| Meta tokens add 128 positions | LOW | <5% prefill overhead. Decode: meta tokens already processed. Hymba proved negligible cost. |

## Success Criteria

1. Loss < 4.0 in 15 min (ambitious, driven by MTP + KD + Engram)
2. Per-token learning efficiency > all previous 11 hypotheses at equal tokens
3. mHC branches show distinct H_pre patterns (branch specialization)
4. Engram gates activate on entities/formulaic phrases
5. MTP head 2 accuracy > 40%, head 3 > 25%, head 4 > 15%
6. Meta tokens show non-trivial contribution (ablation: removing them degrades > 1%)
7. Decode > 250 tok/s fp16 on Strix Halo

## What Makes ARCHON the Supreme Architecture

It's not the fastest. It's not the simplest. It's not the most novel.

It's the **most informed.** Every ingredient is here because paper evidence says it improves quality. Nothing is here for speed or novelty alone. The combination hasn't been built before, but each piece has been validated independently:

- Parallel hybrid: 3 papers at 300M-34B scale
- mHC: proven at 3B-27B by DeepSeek
- Meta tokens: proven at 300M-1.5B by NVIDIA
- Engram: proven at 27B by DeepSeek
- Griffin: proven at 100M-14B by Google
- MTP: proven at 7B-13B by Meta
- Gated conv: proven at 350M-2.6B by Liquid AI
- Knowledge distillation: proven across all scales by everyone

The question isn't whether each piece works. It's whether they **compose** at 250M in 15 minutes of training. Phase training is the answer: isolate, then combine, then enhance.

## Implementation Roadmap

1. Implement GatedConvChannels (LFM2 pattern, 12 channels)
2. Implement GriffinRecurrenceChannels (4 channels, decay bias spectrum)
3. Implement ParallelHybridBlock (conv || rec → concat → outproj)
4. Implement mHC module (4-branch, Sinkhorn H_res, sigmoid H_pre/H_post)
5. Implement MetaTokens (128 learned embeddings, prepend/strip)
6. Implement Engram (hash tables, context-aware gating, at layers 2, 9)
7. Implement MTPHeads (3 adapters + shared LM head)
8. Assemble ArchonModel, verify ~251M params
9. Phase 1 training: backbone + mHC only
10. Phase 2: + meta tokens + Engram
11. Phase 3: + MTP heads
12. Phase 4: + knowledge distillation
13. Griffin associative scan kernel
14. Fused parallel hybrid kernel
15. Int4 quantization + decode benchmark
16. Compare to ALL previous hypotheses + LFM2 baseline estimate

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

**Baseline (estimated):** ~4,500 tok/s eager (11% MFU)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +100% MFU — many components may cause graph breaks | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| `causal-conv1d` in GatedConv | 10x conv speedup | Available |
| mHC 4-branch routing | Complex routing adds ~4% overhead | By design |
| Engram + meta tokens | ~5% overhead from knowledge components | By design |
| MTP heads | Additional training compute (~4%) | By design |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~8,500 tok/s (22% MFU)
**Tokens in 45 min:** ~23.0M (1.4 BabyLM epochs)
**Ranking:** #20 of 22 architectures
**Note:** Lowest throughput tier but designed for maximum quality. Worth testing if quality gap justifies 2x slower training.
