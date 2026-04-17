---
title: "HALO-PRIME: Mamba-Enhanced Looped Architecture for Strix Halo"
domain: architectures
type: spec
status: active
related:
  - docs/superpowers/specs/2026-04-16-jormungandr-halo-design.md
  - mad_llm_scientist/plans/JORMUNGANDR.md
  - knowledge/architectures/architecture_mix_matrix.md
  - knowledge/training/argus_prime_results.md
tags: [%halo-prime, %looped, %mamba, %strix-halo, %amadeus, %jormungandr, %xsa, %depth-mc]
---

# HALO-PRIME: Mamba-Enhanced Looped Architecture for Strix Halo

## Problem

JORMUNGANDR-HALO's core loop uses ShortConvBlocks (depthwise Conv1d, kernel=3) for sequence mixing. Each layer sees only 3 adjacent tokens. Over 3 layers × 4 iterations, the receptive field is ~36 tokens — just 3.5% of the sequence at ctx=1024. The Prelude/Coda GQA attention provides global context, but the iterated core (where most compute goes) is blind to long-range dependencies.

## Solution

Replace the core loop's ShortConvBlock with a parallel Conv+Mamba hybrid block (AMADEUS's proven mixer). Each loop iteration now has full-sequence receptive field via Mamba's selective scan, while retaining local pattern matching via GatedConv. Every other component (Prelude, Coda, XSA, Depth MC, TTT, FiLM, VE, Parcae) stays identical to JORMUNGANDR-HALO.

## Mix Matrix Classification

From `knowledge/architectures/architecture_mix_matrix.md`:

| Component | ID | Choice | Rationale |
|-----------|-----|--------|-----------|
| Structure | S7 | Hybrid (Prelude + shared loop + Coda) | Proven in JORMUNGANDR-HALO |
| Core Mixer | M4+M13 | Mamba-3 SISO + GatedConv (parallel) | AMADEUS quality winner (2.90 loss) |
| FFN | F1/F4 | SwiGLU (core) / SwiGLU+TTT (Coda) | TTT activates at ctx≥512 |
| Conditioning | C1 | FiLM (cross-dim, 512→768) | Proven in JORMUNGANDR-HALO |
| Enhancements | E1+E5+XSA | TTT + Depth MC + XSA | All individually validated |

**Risk: LOW** — every component is individually proven. Only novelty is Mamba inside the loop (required loop-specific stability fixes).

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  ▼
┌──────────────────────────────────────┐
│ PRELUDE (d=768, 2 unique layers)     │
│   Layer 1: ShortConvBlock(768, 512)  │  ← local patterns
│   Layer 2: GQABlock(768, n_kv=4)    │  ← global attention
└───────────┬──────────────────────────┘
            │ input_embed = h (saved, d=768)
            ▼
┌──────────────────────────────────────────────────────┐
│ FUSED ENTRY: Parcae injection + dim adapter           │
│   h_core = A * proj_down(h) + B * proj_down(embed)   │
│                                                       │
│ CORE LOOP (d=512, 3 CoreHybridBlock × T iterations)  │
│                                                       │
│   CoreHybridBlock (AMADEUS-style at d=512):           │
│     RMSNorm → (GatedConv(256) || Mamba3SISO(256))    │
│       → concat → proj → momentum → SwiGLU(1792)      │
│                                                       │
│   Per-iteration: Parcae re-inject → 3 layers → norm  │
│   Detached iters: no grad. Gradient iters: backprop.  │
│   FiLM fingerprint at last gradient iter.             │
│   LoopStableMamba3SISO: dt_max=0.1, pre-softplus     │
│   clamp for fp16 safety.                              │
│                                                       │
│   RMSNorm between iterations (prevents state growth)  │
└───────────┬──────────────────────────────────────────┘
            │ Depth MC → proj_up (512→768)
            ▼
┌──────────────────────────────────────┐
│ CODA (d=768, 4 unique layers)        │
│   Layer 1: ShortConvBlock + FiLM     │
│   Layer 2: CodaGQABlock + FiLM      │  ← XSA, n_kv=4
│   Layer 3: ShortConvBlock + FiLM     │
│   Layer 4: CodaGQABlock + FiLM      │  ← XSA + VE + TTT
└───────────┬──────────────────────────┘
            ▼
  Final RMSNorm → LM Head → Predictions
```

---

## Key Changes from JORMUNGANDR-HALO

### 1. CoreHybridBlock replaces ShortConvBlock in core loop

| Aspect | ShortConvBlock (old) | CoreHybridBlock (new) |
|--------|---------------------|----------------------|
| Local mixing | GatedConv (kernel=3) | GatedConv (d_conv=256) |
| Global mixing | None | Mamba-3 SISO (d_mamba=256) |
| Receptive field/layer | 3 tokens | Full sequence |
| Receptive field/iter | ~9 tokens | Full sequence |
| Architecture | RMSNorm → Conv → SwiGLU | RMSNorm → (Conv ∥ Mamba) → proj → SwiGLU |
| Params/layer | ~3.5M | ~3.8M (+9%) |

### 2. LoopStableMamba3SISO (loop-hardened variant)

Standard Mamba3SISO works in stacked architectures (AMADEUS) but the loop amplifies numerical instability:
- State accumulates over T iterations through the same weights
- Parcae injection's contraction (A ∈ (-1, 0)) can be overwhelmed by Mamba residuals
- fp16 dt_proj can produce inf gradients through softplus

**Fixes:**
- `dt_raw.clamp(min=-10, max=5)` BEFORE softplus — prevents gradient poison
- `dt.clamp(max=0.1)` vs standard `0.5` — gentler state updates for iterated use
- `core_iter_norm = RMSNorm(d_core)` between iterations — bounds state growth

### 3. No torch.compile

JORMUNGANDR-HALO used per-zone torch.compile for 3.07x speedup. HALO-PRIME cannot: autokernel's custom HIP backward for selective_scan conflicts with Dynamo tracing. Mamba's mamba-ssm library already provides optimized scan kernels, so compile provides less benefit here.

---

## Parameter Budget

| Component | Params | Notes |
|-----------|--------|-------|
| Embedding (50257×768, tied) | 38.6M | Same |
| Prelude: ShortConvBlock(768) | ~9.1M | Same |
| Prelude: GQABlock(768, n_kv=4) | ~8.8M | Same |
| Entry proj_down (768→512) | 0.4M | Same |
| Parcae A, B | 1K | Same |
| **Core: 3× CoreHybridBlock(512)** | **~11.4M** | **+2.7M from Mamba** |
| Core iter norm | 512 | New |
| Exit proj_up (512→768) | 0.4M | Same |
| Coda: 2× ShortConvBlock(768) | ~18.2M | Same |
| Coda: 2× CodaGQABlock(768) | ~17.6M | Same |
| FiLM conditioner (4 targets) | ~0.5M | Same |
| Depth Memory Cache (GRM) | ~0.03M | Same |
| XSA | 0 | Same |
| Value Embedding (d_ve=64) | ~3.2M | Same |
| TTT sniper (1-step) | ~1.2M | Same |
| **TOTAL UNIQUE** | **~107M** | +3M vs JORMUNGANDR-HALO |
| **Effective (4× loop)** | **~157M** | |

### Core Mixer Dimensions (Tensile-friendly)

| Matrix | Dims | Mult 128? |
|--------|------|-----------|
| GatedConv proj | 512 × 768 | ✓ |
| Mamba x_proj | 512 × 256 | ✓ |
| Mamba dt_proj | 512 × 256 | ✓ |
| Mamba B_proj | 512 × 64 | 64 (half) |
| Mamba C_proj | 512 × 64 | 64 (half) |
| Conv+Mamba out_proj | 512 × 512 | ✓ |
| Core SwiGLU gate_up | 512 × 3584 | ✓ |
| Core SwiGLU down | 1792 × 512 | ✓ |

---

## Results

### BabyLM (1 epoch, ctx=256, autokernel, Muon)

| Model | Loss | Δ vs J-H Bare | tok/s | Params |
|-------|------|---------------|-------|--------|
| JORMUNGANDR Bare | 6.028 | — | 33.7K | 99.2M |
| JORMUNGANDR +XSA | 5.973 | -0.9% | 33.7K | 99.2M |
| JORMUNGANDR +XSA+DC | 5.770 | -4.3% | 33.7K | 99.2M |
| JORMUNGANDR Full | 5.770 | -4.3% | 33.5K | 103.5M |
| **HALO-PRIME Full** | **5.650** | **-6.3%** | **28.5K** | **103.5M** |

**HALO-PRIME beats all JORMUNGANDR-HALO variants.** The -2.1% improvement over JORMUNGANDR Full comes from Mamba's global context in the core loop — even at ctx=256 where TTT/FiLM/VE don't help.

Throughput is 28.5K vs 33.5K (85%) because HALO-PRIME cannot use torch.compile (Mamba HIP backward incompatible with Dynamo). The quality gain outweighs the throughput cost.

### WikiText-103 (pending)

Running: 1 epoch, ctx=1024, lr=0.0004, from BabyLM checkpoint. Results will be added.

---

## Compile Incompatibility

JORMUNGANDR-HALO achieved 3.07x speedup from per-zone torch.compile (14K → 43K tok/s). HALO-PRIME cannot use this because:

1. `autokernel.selective_scan_backward` is a custom HIP kernel registered via `torch.library`
2. Dynamo traces into the custom op's backward function
3. The HIP kernel doesn't support Inductor's lazy allocation

**Workaround options (future):**
- Wrap selective_scan as `torch.library.custom_op` with opaque backward
- Use mamba-ssm's CUDA backend instead of autokernel's HIP scan
- Compile only the non-Mamba parts (conv, ffn, norm) via selective `torch.compile`

---

## Optimizer Groups

Same as JORMUNGANDR-HALO, with additions for Mamba params:

| Pattern | Optimizer | LR | Notes |
|---------|-----------|-----|-------|
| `core_layers.*.out_proj`, `core_layers.*.ffn.*` | Muon | 0.005 | 2D weights |
| `core_layers.*.ssm.*` | AdamW | 8e-4 | SSM params (forced) |
| `core_layers.*.conv.*` | AdamW | 8e-4 | Conv params (forced) |
| `core_layers.*.log_beta` | AdamW | 8e-4 | Momentum beta (forced) |
| `prelude.*`, `coda.*` (2D) | Muon | 0.005 | Standard Muon |
| `*film*`, `*ttt*`, `*norm*`, `*bias` | AdamW | 8e-4 | Standard |

**Muon ndim fix:** `split_params_for_muon` now uses `param.ndim == 2` (not `>= 2`) to prevent QK-Norm scales (shape `(n_heads, 1, 1)`) from entering Newton-Schulz.

---

## Loop Stability Design

Mamba in a loop is novel — the same weights process accumulated state T times. Three stability measures:

### 1. Per-iteration RMSNorm
```python
# After all 3 core layers per iteration:
h_core = self.core_iter_norm(h_core)
```
Without this, state norm grows ~60% per iteration → NaN by iteration 4. With it, growth bounded to ~19% (norm rescales but directional information preserved).

### 2. Tighter dt clamping
```python
# LoopStableMamba3SISO:
dt_raw = self.dt_proj(x_norm).clamp(min=-10.0, max=5.0)  # pre-softplus
dt = F.softplus(dt_raw).clamp(min=1e-4, max=0.1)         # tighter max
```
Standard Mamba uses `max=0.5`. In the loop, large dt → large state updates → amplification. `max=0.1` keeps updates gentle.

### 3. Parcae contraction
```python
# Each iteration starts with:
h_core = A * h_core + B * input_embed_down
# A = -exp(log_A) ∈ (-1, 0): contractive
```
Combined with the iter norm, this provides double contraction per iteration.

---

## Files

| File | Description |
|------|-------------|
| `models/halo_prime.py` | Model implementation (~350 lines) |
| `scripts/run_halo_prime_babylm.sh` | BabyLM training script |
| `scripts/run_halo_prime_wt103.sh` | WikiText-103 CPT script |

## Reused (no modification)

| File | What it provides |
|------|-----------------|
| `models/amadeus.py` | RMSNorm, SwiGLU, GatedConv, Mamba3SISO, _scan_dispatch |
| `models/argus_prime.py` | ShortConvBlock, GQABlock, Attention |
| `models/argus.py` | TTTSwiGLU, precompute_freqs_cis, apply_rotary_emb |
| `models/jormungandr_halo.py` | ParcaeInjection, ValueEmbedding, CodaAttention, CodaGQABlock, CrossDimFiLMConditioner, DepthMemoryCache |
| `halo_training/muon.py` | MuonAdamW (fixed: ndim == 2 for Newton-Schulz) |
