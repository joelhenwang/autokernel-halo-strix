---
title: "CHIMERA-HALO Architecture Design"
domain: architectures
type: design
status: active
tags: [chimera-halo, parcae, xsa, factorized-embeddings, looped, hybrid, lfm2, nandi]
related:
  - parcae_stable_looped_models.md
  - hypothesis_buildout_results.md
  - ../../docs/superpowers/specs/2026-04-16-jormungandr-halo-design.md
---

# CHIMERA-HALO: Unified Looped Hybrid with Factorized Embeddings

**Date:** 2026-04-20
**File:** `models/chimera_halo.py`
**Status:** Implemented, awaiting training

## Motivation

Combines insights from 5 recent papers into a single architecture targeting ~150M effective params on Strix Halo:

| Paper | Key Insight Used |
|-------|-----------------|
| **LFM2** (Liquid, 2511.23404) | 75:25 conv:attention ratio, hardware-optimized hybrid blocks |
| **Parcae** (UCSD/Together, 2604.12946) | Spectral-stable looping, Poisson depth, scaling laws |
| **XSA** (Apple, 2603.09078) | Exclusive Self Attention — zero-param quality boost |
| **Nandi-150M** (Rta AI Labs) | Factorized embeddings + layer sharing for param efficiency |
| **Attention-to-Mamba** (Apple, 2604.14191) | Identity initialization for grafted components |

**Baselines to beat:**
- SmolLM2-135M: HellaSwag 42.1, ARC 43.9 (trained on 2T tokens)
- Nandi-Mini-150M: avg 25.63 (trained on 500B tokens)

## Architecture

```
FactorizedEmbedding(50260, rank=256, d=768)           ~13.3M
  |
Prelude: 1 XSAGQABlock(768, 12h/4kv)                  ~8.1M
  |
+----- SimpleParcaeInjection(768) -----+
|                                       |
|  SHARED BLOCK (8 unique layers):      |
|    L0-L2, L4-L5, L7: ShortConvBlock  |  6 conv (local mixing)
|    L3, L6: XSAGQABlock               |  2 attn (global mixing)
|                                       |
|  Repeat: 2x (16 effective layers)     |
|  Parcae loop: Poisson(mean=3), bptt=2 |
|  DepthMemoryCache over iterations     |
+---------------------------------------+
  |
Coda: 1 XSAGQABlock(768, 12h/4kv)                     ~8.1M
  |
RMSNorm -> FactorizedLMHead(768, rank=256)             ~0.2M
```

## Parameter Budget

| Component | Params |
|-----------|--------|
| Factorized input embedding (50260 x 256 + 256 x 768) | 13.07M |
| Factorized output proj (768 x 256, table shared) | 0.20M |
| Prelude XSAGQABlock | 8.06M |
| 6x ShortConvBlock (shared) | 48.39M |
| 2x XSAGQABlock (shared) | 16.12M |
| Coda XSAGQABlock | 8.06M |
| SimpleParcaeInjection | 0.002M |
| DepthMemoryCache | 0.05M |
| Norms | 0.002M |
| **TOTAL UNIQUE** | **93.9M** |
| **Effective (2x repeat)** | **158.5M** |

Standard embedding at vocab=50260, d=768 would cost 38.6M params. Factorized rank=256 costs 13.3M — **saving 25.3M params** (27% of total budget) freed for compute layers.

## Key Design Decisions

### 1. Uniform d=768 (not heterogeneous d=768/512)
JORMUNGANDR-HALO used d=512 core for L2 cache residency. CHIMERA-HALO uses uniform d=768:
- Enables Nandi-style layer sharing (same dims everywhere)
- No projection overhead between stages
- All matmuls 128-aligned for Tensile
- L2 benefit was marginal (22.8MB actual vs claimed 1.2MB)

### 2. Factorized Embeddings (rank=256)
From Nandi-150M: `Embedding(V, R) → Linear(R, D)` for input, `Linear(D, R) → matmul(embed.weight.T)` for output. Rank=256 chosen for 128-alignment (Tensile). Saves 25.3M params.

### 3. 75:25 Conv:Attention Ratio
LFM2 uses ~60:40 at 350M. At 150M scale, attention is proportionally more expensive. 6 conv + 2 GQA per 8-layer block gives 4 effective attention layers (with 2x repeat) — comparable to ARGUS-PRIME's 6 GQA in 16 layers.

### 4. XSA on All Attention Layers
XSA subtracts self-value projection: `z = y - (y·v/||v||²)·v`. Zero parameters, zero hyperparameters. Uses CodaAttention(exclusive=True) from JORMUNGANDR-HALO. Gains scale with model size and context length.

### 5. No TTT, No FiLM, No ValueEmbedding
- TTT: no benefit at ctx=256 (JORMUNGANDR ablations), adds 1.2-4.3M params
- FiLM: needs heterogeneous dims for cross-dim conditioning; marginal gain at this scale
- ValueEmbedding: small contribution, adds autokernel compatibility complexity
- All can be added later as staged activation if needed

### 6. 2x Layer Repeat + Poisson Loop
Nandi's deterministic 2x sharing + Parcae's stochastic depth. 8 unique layers executed twice per iteration, Poisson(mean=3) iterations with truncated BPTT(depth=2). Effective depth: ~50 layers.

## Autokernel Compatibility

| Pattern | Status | Notes |
|---------|--------|-------|
| FusedQKV | Compatible via CodaAttention wq/wk/wv/wo | XSA applied inside CodaAttention, FusedQKV skipped on XSA layers |
| FusedSwiGLU | Yes | All SwiGLU use w_gate_up + w_down |
| RMSNorm | Yes | All norms are RMSNorm class |
| FusedGatedConv | Yes | ShortConvBlocks use GatedConv |

## Variants

| Class | Config | Use Case |
|-------|--------|----------|
| `ChimeraHalo` | mean_recurrence=3, XSA, depth cache | Default training |
| `ChimeraHaloDeep` | mean_recurrence=5, bptt=3 | Maximum quality |
| `ChimeraHaloBare` | No XSA, no depth cache | Ablation baseline |
| `ChimeraHaloNoLoop` | Single pass (no Parcae) | Ablation: loop value |
| `ChimeraHaloMini` | ~0.7M, vocab=1000 | Smoke testing |

## Training Recipe (Recommended)

```bash
# Smoke test
python -m halo_training --model models/chimera_halo.py --class-name ChimeraHaloMini --smoke

# BabyLM validation
python -m halo_training --model models/chimera_halo.py --class-name ChimeraHalo \
  --dataset babylm --compile --optimize-kernels --muon \
  --block-size 1024 --batch-size 16 --lr 8e-4 --time-budget 45

# Dolma production
nohup python -m halo_training --model models/chimera_halo.py --class-name ChimeraHalo \
  --dataset dolma --compile --optimize-kernels --muon \
  --block-size 1024 --batch-size 16 --epochs 1 --time-budget 1440 \
  --checkpoint-dir checkpoints/chimera_dolma --checkpoint-interval 5000 &
```

## Comparison to Prior Architectures

| Model | Unique Params | Effective | Layers | Conv:Attn | tok/s (est) | Key Mechanism |
|-------|-------------|-----------|--------|-----------|-------------|---------------|
| ARGUS-PRIME | 168M | 168M | 16 | 10:6 | 18K | Surgical TTT |
| JORMUNGANDR-HALO | 104M | 152M | 3-core+4-coda | ~6:4 | 34K | L2-resident d=512 loop |
| **CHIMERA-HALO** | **94M** | **158M** | **8 shared x2** | **6:2** | **~18-24K** | **Factorized embed + uniform loop** |
| SmolLM2-135M | 135M | 135M | 30 | 0:30 | N/A | Deep narrow Llama |
| Nandi-150M | 150M | 150M (32 eff) | 16 x2 | 0:32 | N/A | Layer sharing + factorized embed |

## Paper References

1. LFM2 Technical Report — arxiv.org/abs/2511.23404
2. Parcae: Scaling Laws for Stable Looped Language Models — arxiv.org/abs/2604.12946
3. Exclusive Self Attention — arxiv.org/abs/2603.09078
4. Attention to Mamba: Cross-Architecture Distillation — arxiv.org/abs/2604.14191
5. Self-Improving Pretraining — arxiv.org/abs/2601.21343
