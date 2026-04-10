# DEEP NARROW ORACLE

**48 Layers at d=512 with Parallel Hybrid + Engram + Burst Decoding**

## Hypothesis

Falcon-H1-1.5B-Deep (66 layers, d=1280) rivals 7-10B models — proving deep+narrow beats wide+shallow at small scale. Our version: **48 layers at d=512** with parallel hybrid (conv+recurrence), Engram tables, and multi-token burst decoding. Each layer is tiny (~3M params = 6 MB fp16 = **1.5 MB int4 — fits in L2!**), enabling Resonant-Loop-style L2 caching benefits even without weight sharing.

**Key insight:** With 48 tiny layers, consecutive layers' weights partially overlap in L2 cache. The last layer read evicts the first, but intermediate layers may still be cached. This gives a natural "sliding L2 window" effect.

---

## Architecture

```
[64 Meta Tokens] + Input Tokens → Embedding (d=512, tied LM head)
  → 48 Parallel Hybrid Blocks (d=512):
      RMSNorm → [Conv(8ch) || Recurrence(4ch)] → Concat → OutProj → +Residual
      RMSNorm → SwiGLU FFN(512→1024→512) → +Residual
      (+ Engram at blocks 4, 24)
  → Final RMSNorm → MTP Heads (4 heads for burst decoding) → LM Head
```

### Why 48 Layers at d=512?

| Config | Layers | d_model | Params per layer | Total (excl embed) |
|--------|--------|---------|-----------------|-------------------|
| Wide (Caveman LFM) | 16 | 1024 | ~13M | ~208M |
| **Deep narrow** | **48** | **512** | **~3.1M** | **~149M** |
| Falcon-H1-0.5B | 36 | 1024 | ~10M | ~360M |

Deep narrow uses fewer total params in layers but 3× more refinement passes. Each pass is cheap. At d=512, every projection is 4× smaller than d=1024.

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| d_conv (conv channels) | 384 (8 channels × 48) |
| d_rec (recurrence channels) | 128 (4 channels × 32) |
| n_layers | 48 |
| ffn_inner | 1024 (SwiGLU, 2× expansion) |
| conv_kernel | 3 |
| meta_tokens | 64 |
| vocab_size | 50257 |
| Engram | bigram 50K×64, trigram 25K×64, at layers 4, 24 |
| MTP heads | 4 (for burst decoding) |
| Glue dictionary | ~500 tokens |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×512, tied) | 25.7M |
| Meta tokens (64×512) | 33K |
| Per layer: conv projs (512→3×384 + part of outproj) | ~0.8M |
| Per layer: rec projs (4×512→128 + bypass) | ~0.3M |
| Per layer: output proj (512→512) | ~0.26M |
| Per layer: SwiGLU FFN (512→1024→512) | ~1.6M |
| Per layer: RMSNorm ×2 | ~1K |
| Per layer total | ~3.0M |
| 48 layers | ~144M |
| Engram (2 layers) | ~8M |
| MTP adapters (3 × 512→512 MLPs) | ~0.8M |
| Decay biases + routing | ~0.1M |
| **TOTAL** | **~179M** |

179M is well under 250M! Options:
- Increase d_model to 576: layers ~3.8M × 48 = 182M + 29.5M embed = 212M. Still fits.
- Add more Engram: 30M in tables → 209M. Fits.
- Increase ffn_inner to 1280: layers ~3.5M × 48 = 168M + 25.7M = 194M. Fits.

**Revised: d_model=576, ffn_inner=1152, Engram=20M:**
- Per layer: ~3.8M × 48 = 182M
- Embedding: 50257×576 = 28.9M
- Engram: 20M
- MTP + meta + routing: ~1.5M
- **TOTAL: ~233M** ✓

## L2 Cache Analysis

**Per layer in int4: ~3.8M × 0.5B = 1.9 MB**

L2 is 6 MB. That's **3 consecutive layers** in L2 simultaneously!

During decode, when processing layer N:
- Layer N weights: read from DRAM (or L2 if recently cached)
- Layer N-1 weights: might still be in L2 (LRU)
- Layer N-2 weights: might still be in L2

With 48 layers, the "sliding L2 window" covers ~6% of layers at any time. Not transformative, but provides ~5-10% cache hit rate on layer transitions.

## Decode Speed (Strix Halo)

Weight reads per token:
- 48 layers × 3.8M × 2B fp16 = 365 MB
- LM head: 50257 × 576 × 2B = 58 MB
- Total: ~423 MB fp16

| Mode | Weight reads | Kernel overhead (48L × 5 kernels × 5μs) | Total |
|------|-------------|------------------------------------------|-------|
| fp16 | 423 MB / 240 GB/s = 2.49 ms | 1.2 ms | **~3.7 ms = ~270 tok/s** |
| int4 | 106 MB / 240 GB/s = 0.62 ms | 1.2 ms | **~1.82 ms = ~549 tok/s** |

**Note:** Kernel launch overhead is significant at 48 layers! 240 kernel launches × 5μs = 1.2 ms. This is the deep-narrow penalty.

**With burst decoding (avg 2.3×):**
- int4: 549 × 2.3 = **~1263 tok/s effective**

**Optimization:** Fuse consecutive layers to reduce kernel launches. Fuse 3 layers into 1 mega-kernel: 16 launches instead of 48 → 0.4 ms overhead → 1.02 ms total → ~980 tok/s int4 → ~2254 burst.

## Training

### Advantages of Deep Narrow

- 48 layers of gradient flow → better feature learning per param
- Each layer's parameters are small → faster per-step (less weight to read)
- Can use gradient checkpointing every 8 layers (6 checkpoints)
- Falcon-H1 evidence: deep narrow trains stably with μP

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Dataset | OpenWebText ~100M tokens |
| Batch | 64×512, accum=2 (64K effective) — larger batch, smaller model |
| LR | 1e-3 cosine → 1e-4 (higher LR for small model, per Falcon-H1 μP findings) |
| Engram LR | 5× base, zero WD |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 8 layers |
| Precision | fp16 mixed + fp32 scan |
| Est. throughput | ~40M tokens in 15 min (small layers = fast steps) |

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| 48 layers = vanishing gradients | HIGH | RMSNorm at every layer; residual connections; μP initialization |
| Kernel launch overhead (1.2ms) | HIGH | Fuse 3 consecutive layers into mega-kernels; reduces to 0.4ms |
| d=512 lacks representational capacity | MEDIUM | 48 layers compensate with depth; Engram adds factual capacity |
| Training sequential depth = slow | MEDIUM | Small layers = fast per-step; gradient checkpointing every 8 |

## Success Criteria

1. Loss < 4.5 in 15 min (deeper model may learn faster per token)
2. Loss curve shows advantage over 16-layer d=1024 baseline at equal tokens
3. Decode > 500 tok/s int4 on Strix Halo (before burst)
4. Burst decoding adds ≥2× effective throughput
5. Per-layer int4 confirmed to fit 3 consecutive layers in L2

## Implementation Roadmap

1. Implement ParallelHybridBlock at d=576 (conv 8ch + rec 4ch)
2. Stack 48 layers with gradient checkpointing every 8
3. Add meta tokens (64) + Engram (layers 4, 24)
4. Add MTP heads (4) for burst decoding
5. Verify params ~233M
6. Train with higher LR (μP-inspired)
7. Implement fused 3-layer mega-kernel to reduce launch overhead
8. Benchmark: decode speed, burst ratio, quality vs wide baseline

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x), **silu_gate_mul** (1.6x), **cross_entropy** (1.8x)
- Apply via `autokernel.optimize(model, training=True)`

### 48 Layers × d=512: Serial Weight Read Concern
Deep-narrow means 48 serial matmul weight reads per forward pass. On memory-bound hardware (240 GB/s), each layer reads ~2 MB (d=512 weights) = 96 MB total. This is fast in absolute terms but the **serial depth limits parallelism** — the GPU can't overlap layer N+1 with layer N.

### Scan: Use Chunked Linear Recurrence
Same as all SSM plans. Reference: `models/amadeus.py:selective_scan_chunked`.

### L2 Cache Note
With d=512, per-layer weights (~2 MB) fit in L2 (6 MB). But 48 layers total (~96 MB) don't — only ~3 layers cached at once. The deep architecture repeatedly evicts L2. Wider+shallower (e.g., 16 layers × d=1024) would be more cache-friendly.

### Throughput: ~4-6K tok/s (48 serial layers hurt), MFU: 50-60%
The serial depth is the main throughput limiter. Consider reducing to 32 layers if throughput matters more than depth.

### External Kernel Integration (verified 2026-04-10)

- **GatedConv:** causal-conv1d (10x vs nn.Conv1d) — critical for 48 layers of conv
- **Throughput estimates updated:** With causal-conv1d, conv overhead drops significantly across all 48 layers
