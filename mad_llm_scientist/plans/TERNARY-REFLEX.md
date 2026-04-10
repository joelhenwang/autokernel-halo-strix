# TERNARY REFLEX

**MatMul-Free Reflex Path + Standard Genius Path with Caveman Routing**

## Hypothesis

Take Dual Cortex's dual-path architecture but make the fast reflex path **completely matmul-free** using ternary weights ({-1, 0, +1}) and element-wise GRU. On Strix Halo with NO matrix cores, ternary operations (additions and bit-flips) are the ultimate hardware-native compute. The reflex path processes 65% of tokens (glue words) with ZERO matmuls. The genius path uses standard weights via rocBLAS for the 35% hard tokens.

**Evidence:** MatMul-free LM paper (2406.02528) — competitive at 2.7B, 61% training memory reduction, 10× inference memory reduction. The gap narrows at larger scales.

---

## Architecture

```
Token → Embedding (d=1024, tied LM head)
  → Project down: 1024 → 320 (TERNARY projection!)
  │
  ├─ REFLEX PATH (matmul-free, 8 layers, d=320):
  │   For each layer:
  │     Ternary linear(320→960) → element-wise gate → ternary linear(960→320)
  │     + Element-wise GRU for sequence mixing (no matmul!)
  │     + RMSNorm
  │   → Entropy check: if confident → SKIP genius path
  │
  ├─ GENIUS PATH (standard, 10 layers, d=1024):
  │   Project up: 320 → 1024 (standard linear)
  │   Parallel hybrid blocks (conv+Griffin recurrence)
  │   SwiGLU FFN (standard weights)
  │   → Project down: 1024 → 320
  │   → Additive correction to reflex state
  │
  → Project up: 320 → 1024 → Final RMSNorm → LM Head
```

### MatMul-Free Reflex Layer

From the MatMul-free LM paper, replace all dense layers with ternary weights:

```python
# Ternary linear: weights ∈ {-1, 0, +1}
# Forward: y = TernaryLinear(x) = accumulate(x * ternary_weight)
# This is NOT a matmul — it's additions and sign flips
# Memory: 2 bits per weight (vs 16 bits for fp16)

# Element-wise GRU for sequence mixing (no QK^T attention!)
z = sigmoid(ternary_linear(x))           # update gate
r = sigmoid(ternary_linear(x))           # reset gate
h_tilde = tanh(ternary_linear(r * h_prev + x))  # candidate
h = (1 - z) * h_prev + z * h_tilde      # state update
# ALL operations: ternary accumulation + element-wise sigmoid/tanh/multiply
```

**Key:** The ternary linear computes `y[i] = Σ_j w[i,j] * x[j]` where w ∈ {-1,0,+1}. This is just: add x[j] if w=+1, subtract x[j] if w=-1, skip if w=0. No multiplication at all.

### Weight Storage

| Component | Standard (fp16) | Ternary (2-bit) | Savings |
|-----------|----------------|-----------------|---------|
| Reflex path (8 layers, d=320) | 10 MB | **1.25 MB** | 8× |
| Genius path (10 layers, d=1024) | 300 MB | 300 MB (standard) | — |
| Embedding + LM head | 103 MB | 103 MB (standard) | — |

**Reflex path in ternary: 1.25 MB → FITS IN L2 (6 MB) with massive headroom!**

## Configuration

| Parameter | Reflex (MatMul-Free) | Genius (Standard) |
|-----------|---------------------|-------------------|
| d_model | 320 | 1024 |
| n_layers | 8 | 10 |
| Weight type | Ternary {-1,0,+1} | fp16/int4 |
| Mixer | Element-wise GRU | Parallel hybrid (conv+Griffin) |
| FFN | Ternary linear + ReLU² | SwiGLU (standard) |
| Activation rate | 100% of tokens | ~35% (content only) |

| Global | Value |
|--------|-------|
| vocab_size | 50257 |
| d_embedding | 1024 |
| Entropy router | norm(h_new - h_old) > θ |
| Engram | On reflex path (bigram+trigram tables) |

## Parameter Count

| Component | Params | Storage |
|-----------|--------|---------|
| Embedding (50257×1024, tied) | 51.5M | 103 MB fp16 |
| Embed→reflex proj (ternary 1024→320) | 328K | 82 KB ternary |
| Reflex path (8 layers, d=320, ternary) | ~8M | **1.0 MB ternary** |
| Engram tables + projections | 12.8M | 25.6 MB fp16 |
| Reflex→genius proj (standard 320→1024) | 328K | 0.66 MB |
| Genius path (10 layers, d=1024, standard) | ~155M | 310 MB fp16 |
| Genius→reflex proj (standard 1024→320) | 328K | 0.66 MB |
| Reflex→embed proj (standard 320→1024) | 328K | 0.66 MB |
| Entropy MLP + norms | ~70K | 0.14 MB |
| **TOTAL** | **~228M** | — |

## Decode Speed (Strix Halo)

**The killer feature:** Reflex path reads 1 MB of ternary weights (in L2!) + Engram lookups (~6 KB). The "compute" is additions and bit-flips — effectively FREE on any hardware.

| Token type | Fraction | DRAM reads | Compute | Time |
|-----------|----------|-----------|---------|------|
| Glue (reflex only) | 65% | LM head (26 MB int4) + reflex (**L2!**) | Ternary accum (free) | ~0.19 ms |
| Content (reflex + genius) | 35% | LM head (26 MB) + genius (77.5 MB int4) + reflex (L2) | Standard + ternary | ~0.79 ms |
| **Weighted average** | | | | **~0.40 ms = ~2500 tok/s int4** |

### fp16

| Token type | Fraction | Time |
|-----------|----------|------|
| Glue | 65% | LM head (103 MB) + reflex (L2) = ~0.65 ms |
| Content | 35% | LM head (103 MB) + genius (310 MB) = ~2.48 ms |
| **Weighted** | | **~1.29 ms = ~775 tok/s** |

## Training

### Ternary Weight Training

From MatMul-free LM: use Straight-Through Estimator (STE) for ternary quantization during training:
```python
# Forward: quantize to {-1, 0, +1}
w_ternary = sign(w_latent) * (abs(w_latent) > threshold)
# Backward: gradient passes through as if continuous
grad_w_latent = grad_w_ternary  # STE
```

Maintain latent fp32 weights, quantize on forward pass, STE on backward.

### Phase Training

1. **Phase 1 (40%):** Train reflex path only (ternary) + Engram. All tokens through reflex.
2. **Phase 2 (30%):** Add genius path (standard). Soft routing. Both paths receive gradients.
3. **Phase 3 (20%):** Anneal routing. Push glue ratio to 65%.
4. **Phase 4 (10%):** Freeze ternary weights, fine-tune genius path + routing threshold.

### Hyperparameters

Standard Caveman LFM recipe. Ternary path uses higher LR (2× base) due to quantization noise.

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Ternary weights too weak at d=320 | HIGH | 8 layers of ternary + Engram tables compensate. MatMul-free LM was competitive at 2.7B. |
| STE training instability | MEDIUM | Standard technique, well-understood. Warmup with fp32, then enable ternary. |
| Gap between ternary and standard path confuses routing | MEDIUM | Phase training decouples. Reflex quality floor established before genius activates. |

## Success Criteria

1. Loss < 5.0 in 15 min (ternary path alone)
2. 65%+ tokens route to reflex at convergence
3. Reflex path achieves < 10% quality gap vs full-precision baseline
4. Decode > 2000 tok/s int4 on Strix Halo
5. Ternary weights fit in L2 (verified: 1 MB << 6 MB)

## Implementation Roadmap

1. Implement TernaryLinear module (latent fp32, forward quantize, STE backward)
2. Implement ElementWiseGRU (ternary linears + sigmoid/tanh)
3. Implement reflex path (8 ternary layers)
4. Port genius path from Caveman LFM (parallel hybrid)
5. Implement entropy router + projection bridges
6. Phase training
7. Benchmark ternary decode speed on Strix Halo

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x), **silu_gate_mul** (1.6x) — in genius path
- **cross_entropy** (1.8x) — loss computation
- Reflex path uses ternary ops (no existing kernel — needs new TernaryLinear HIP kernel)
- Apply genius path kernels via `autokernel.optimize(model, training=True)`

### L2 Cache Advantage (Key Hardware Win)
- Ternary reflex path: **1.25 MB → fits entirely in L2 (6 MB)**
- For glue tokens (65%), forward reads only LM head from DRAM — reflex weights cached
- **This is the fastest decode architecture for Strix Halo** (DRAM reads = LM head only for majority of tokens)
- Genius path (310 MB fp16 / 78 MB int4) must read from DRAM

### Scan Implementation
- Reflex path: Element-wise GRU (d=320) — sequential but small dimension. No chunked scan needed.
- Genius path: Griffin recurrence → **use chunked linear recurrence** (chunk_size=64). Reference: `models/amadeus.py:selective_scan_chunked`.

### STE Training Stability (CRITICAL)
Straight-Through Estimator for ternary weights requires very conservative training:
- **Max LR: 1e-4** (NOT the standard 8e-4 — 8x lower)
- **Warmup: 200 steps** (2x standard)
- **Grad clip: 0.05** (20x tighter than standard 1.0)
- **Label smoothing: 0.02-0.05** (softens gradients through ternary quantization)
- From COOKBOOK.md TernaryLinear section — these are non-negotiable

### MFU Estimate: ~88% weighted average
- Reflex path (65% of tokens): ~98% MFU (L2-cached ternary ops)
- Genius path (35% of tokens): ~75% MFU (weight-bandwidth-bound matmuls)
- Weighted: 0.65 × 98% + 0.35 × 75% = **88% MFU** (best dual-path design)

### Throughput Estimates
- **Training (eager):** ~5-7K tok/s (reflex is fast, genius is Caveman-speed)
- **Token budget:** 15 min = 4.5-6.3M | 45 min = 13.5-18.9M | 120 min = 36-50M
- **Decode (int4):** ~1800-2200 tok/s realistic (2500 only if genius path rarely activates)

---

## Possible Optimizations & Throughput Estimate

**Baseline (estimated):** ~7,000 tok/s eager (16% MFU)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +100% MFU — dual-path compiles independently | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| Ternary reflex path L2 residence | Ternary weights (1.58-bit) fit more in L2; 5-10x effective BW for reflex path | By design |
| `dequantize_int4` for ternary weights | 16.3x dequantization kernel at inference | Available |
| FLA HGRN for Griffin scan | 0.40ms Triton kernel for genius path recurrence | Available |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~14,000 tok/s (32% MFU)
**Tokens in 45 min:** ~37.8M (2.4 BabyLM epochs)
**Ranking:** #6 of 22 architectures
