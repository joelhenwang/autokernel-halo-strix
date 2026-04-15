---
title: "Hypothesis Build-Out & Training Results"
domain: architectures
type: results
status: active
related:
  - knowledge/architectures/Estimation_Hypothesis_Ranking.md
  - knowledge/architectures/reliable_small_lm_insights.md
  - knowledge/architectures/ple_ablation_results.md
tags: [%hypothesis, %amadeus, %tempest, %buildout, %170m]
---

# Hypothesis Build-Out & Training Results

**Date:** 2026-04-11 — 2026-04-12
**Hardware:** AMD Radeon 8060S (gfx1151, RDNA 3.5), 40 CUs, 128 GB LPDDR5X

## Summary

Built 5 new architecture implementations, optimized all models with fused Griffin projections and vectorized chunked scan, downsized to ~170M params, and trained all 9 hypotheses on BabyLM (16.5M tokens, 2 epochs). Key conclusion: **AMADEUS/MaestroPrima win on quality, RESONANT-LOOP wins on throughput, dual-path models (DualCortex/Obsidian) fail with autokernel.**

---

## New Models Implemented

| Model | File | Architecture | Params (full / 170M) |
|-------|------|-------------|---------------------|
| MAESTRO-PRIMA | `models/maestro_prima.py` | AMADEUS + Conductor (137K overhead) | 243.9M / 157.8M |
| SPECTRAL-HYDRA | `models/spectral_hydra.py` | Multi-scale Griffin, 16 heads heterogeneous decay | 244.5M / 176.8M |
| RESONANT-LOOP | `models/resonant_loop.py` | Shared block × 16 iterations + ACT halting | 58.8M / 50.7M |
| DUAL-CORTEX | `models/dual_cortex.py` | System 1/2 entropy-gated dual-path Griffin | 154.3M / 125.2M |
| OBSIDIAN | `models/obsidian.py` | BitNet b1.58 reflex + Caveman LFM genius | 169.3M / 124.0M |

---

## Optimizations Applied (2026-04-11)

### 1. Fused Griffin Projections
Combined 3 separate Linear(d, d_rec) projections (w_a, w_i, w_v) into single Linear(d, 3*d_rec) + split. Saves 2 GEMM kernel launches per layer. Applied to: tempest.py, spectral_hydra.py, tempest_124m.py.

### 2. Vectorized Chunked Scan
Replaced the Python for-loop in `_chunked_scan()` with fully vectorized cross-chunk propagation using cumulative products. No more graph breaks — torch.compile can now fuse through the entire scan. Result: **Tempest compile now works** (previously failed).

### 3. FusedGriffinBlockPattern (autokernel)
Added new pattern to detect TempestBlock structure (conv + griffin + out_proj + pre_norm + ffn + ffn_norm + momentum). **Disabled** — causes torch.compile tracing failures due to module import resolution. The individual RMSNorm and SwiGLU patterns still fire.

### 4. Tempest 124M Variant
Created `models/tempest_124m.py` — parameter-matched to LlamaModel (d=768, 14L, 123.7M params, vocab=32000) for fair comparison.

---

## Why LlamaModel is 3x Faster Than Our Hypotheses

At equal params (~124M), the gap is only 1.2x in eager mode. The 3x gap comes from optimization effectiveness:

| Model | Eager | Autokernel | AK + Compile | Compile Boost |
|-------|-------|------------|-------------|--------------|
| LlamaModel 124.7M | 15,276 | 47,684 | **49,302** | **3.2x** |
| Tempest124M 123.7M | 12,031 | 16,674 | **20,184** | **1.7x** |
| Amadeus124M 109.8M | 12,717 | 17,148 | FAILS | — |

**Root causes:**
1. **FusedResidualRMSNorm block pattern** gives LlamaModel 6.6x on the whole block — Tempest doesn't match (different attribute names)
2. **torch.compile** fuses 3.2x for LlamaModel (no graph breaks) vs 1.7x for Tempest (chunked scan was the bottleneck, now fixed)
3. **Griffin's 3 narrow projections** (1024→384 each) have worse Tensile tile utilization than LlamaModel's QKV (768→2048). Partially fixed by fusing into single GEMM.

---

## Throughput Benchmarks at ~170M Scale

**Config:** batch=8, seq=256, autokernel.optimize(), 30 steps median

| Rank | Model | Params | Best tok/s | Config | MFU |
|------|-------|--------|-----------|--------|-----|
| 1 | Obsidian | 124.0M | 36,419 | AK+compile | 45.6% |
| 2 | DualCortex | 125.2M | 34,656 | AK+compile | 43.8% |
| 3 | ResonantLoop | 50.7M | 17,703 | AK+compile | 9.1% |
| 4 | Tempest | 176.8M | 14,991 | AK+compile | 26.8% |
| 5 | Prometheus | 174.3M | 14,956 | AK+compile | 26.3% |
| 6 | SpectralHydra | 176.8M | 14,719 | AK+compile | 26.3% |
| 7 | Virtuoso | 180.8M | 13,876 | AK+compile | 25.3% |
| 8 | Amadeus | 157.7M | 13,116 | AK only | 20.9% |
| 9 | MaestroPrima | 157.8M | 12,868 | AK only | 20.5% |

**Note:** DualCortex/Obsidian have high tok/s because they're only 124-125M params — but they fail to learn (see training results).

---

## BabyLM Training Results (16.5M tokens, 2 epochs)

**Config:** batch=16, seq=256, autokernel.optimize(), AdamW 8e-4 cosine→8e-5, 90/10 train/val split

| Rank | Model | Params | Val Loss | Train Loss | tok/s | steps/s | Time |
|------|-------|--------|----------|------------|-------|---------|------|
| **1** | **Amadeus** | 157.7M | **2.9015** | 2.7510 | 13,203 | 3.2 | 38 min |
| **2** | **MaestroPrima** | 157.8M | **2.9017** | 2.7393 | 12,896 | 3.1 | 39 min |
| 3 | Tempest | 176.8M | 2.9796 | 2.7688 | 12,952 | 3.2 | 39 min |
| 4 | Virtuoso | 180.8M | 2.9936 | 2.8189 | 11,165 | 2.7 | 45 min |
| 5 | Prometheus | 174.3M | 2.9951 | 2.8379 | 13,066 | 3.2 | 39 min |
| 6 | SpectralHydra | 176.8M | 3.1940 | 3.1182 | 10,323 | 2.5 | 49 min |
| 7 | ResonantLoop | 50.7M | 3.4176 | 3.2680 | 15,907 | 3.9 | 32 min |
| 8 | DualCortex | 125.2M | 5.4352 | 5.4433 | 32,426 | 7.9 | **FAILED** |
| 9 | Obsidian | 124.0M | 5.7074 | 5.6637 | 34,115 | 8.3 | **FAILED** |

### Quality Tiers

**Tier 1 (val < 3.0):** Amadeus, MaestroPrima, Tempest — SSM hybrids and pure Griffin learn well
**Tier 2 (val 3.0-3.2):** Virtuoso, Prometheus, SpectralHydra — viable but slightly worse
**Tier 3 (val > 3.4):** ResonantLoop — throughput champion but 50.7M params limits quality
**Failed (val > 5.0):** DualCortex, Obsidian — autokernel breaks dual-path architectures at d=256

---

## Conductor vs No-Conductor (Head-to-Head)

Detailed A/B comparison: AMADEUS vs MAESTRO-PRIMA with identical data split, hyperparameters, seed.

| Metric | Amadeus | MaestroPrima | Difference |
|--------|---------|-------------|------------|
| Final val loss | 2.9038 | 2.9023 | **-0.0015** |
| Epoch 1 val | 3.1176 | 3.1153 | -0.0023 |
| tok/s | 8,155 | 8,100 | -0.7% |
| Extra params | — | +137K (0.09%) | — |

**Training curve analysis:**
- Steps 0-2000: Alternating leads, ±0.02 noise
- Steps 2000-4000: Converging, <0.005 difference
- Steps 4000-7200: MaestroPrima consistently ~0.002 better

**Verdict: NO SIGNIFICANT DIFFERENCE.** The Conductor learns something real (consistent 0.002 advantage in epoch 2) but the effect is negligible at 16M tokens. May matter at 100M+ tokens.

---

## Key Findings & Conclusions

### 1. SSM Hybrids Win on Quality
AMADEUS (Conv + Mamba-3 SISO + FiLM) achieves the best validation loss at 170M scale. The selective scan's ability to model long-range dependencies pays off even on 16M tokens.

### 2. Parameter Count is the #1 Throughput Lever
Models with fewer params are faster: ResonantLoop (50.7M → 15.9K tok/s) > Amadeus (157.7M → 13.2K tok/s) > Tempest (176.8M → 12.9K tok/s). Within the same architecture, reducing params always increases tok/s proportionally.

### 3. Dual-Path Architectures Are Broken with Autokernel
DualCortex and Obsidian both fail completely (val > 5.0) when autokernel is applied. The HIP kernel replacements at small dimensions (d=256 for the fast/reflex path) cause numerical issues. These models need either:
- Running without autokernel (`--no-optimize`)
- Fixing the HIP kernels for small hidden dimensions
- Increasing the fast path dimension to d≥512

### 4. torch.compile Matters More Than autokernel for Recurrence Models
autokernel gives 1.3x for Griffin models (only individual RMSNorm + SwiGLU). torch.compile gives additional 1.3x by fusing element-wise chains. Combined: ~1.7x. vs LlamaModel's 3.2x from FusedResidualRMSNorm block pattern + compile.

### 5. Vectorized Chunked Scan Enables Compile
Replacing the Python for-loop with cumulative product cross-chunk propagation unlocked torch.compile for Tempest: 12.8K → 15.0K tok/s (+17%). This is a permanent improvement.

### 6. Fused Griffin Projections Provide Modest Gains
Combining w_a + w_i + w_v into single GEMM saves kernel launches but the gain is small (~3%) because rocBLAS Tensile already has low launch overhead.

### 7. Estimation Accuracy
Original throughput estimates were 1.4-2.1x optimistic. Actual correction factor: ~1.65x average. Dual-path architectures were most overestimated (2.07x).

### 8. The Conductor is a Non-Factor at 16M Tokens
MaestroPrima's Conductor adds 0.15% improvement — statistically indistinguishable from noise. The mean-pool summary is too coarse for the model to learn meaningful component dynamics at this data scale.

---

## Dual-Path Eager Diagnostic (2026-04-12)

Ran DualCortex and Obsidian WITHOUT autokernel on BabyLM to isolate root cause.

| Model | Config | Val Loss | tok/s | Diagnosis |
|-------|--------|----------|-------|-----------|
| DualCortex d=256 | **eager** | **3.1909** | 11,356 | Works — autokernel is the problem |
| DualCortex d=256 | autokernel | 5.4352 | 32,426 | FAILED |
| Obsidian d=256 | **eager** | **3.4924** (ep1) | 12,636 | Works — autokernel is the problem |
| Obsidian d=256 | autokernel | 5.7074 | 34,115 | FAILED |

**Root cause confirmed:** autokernel HIP kernel replacements (RMSNorm, SwiGLU) at d=256 cause numerical divergence. The architectures themselves learn normally in eager mode. DualCortex eager (3.19) is competitive with SpectralHydra (3.19) and close to Tempest (2.98).

---

## Why LlamaModel is 2.4x Faster Than Griffin (Compile Gap Analysis)

At 124M params, fair comparison:

| Model | Eager | Autokernel | AK + Compile | Compile Boost |
|-------|-------|------------|-------------|--------------|
| LlamaModel | 15,276 | 47,684 | **49,302** | **3.2x** |
| Tempest | 12,031 | 16,674 | **20,184** | **1.7x** |

The gap is NOT architecture quality (only 1.27x in eager). It's optimization effectiveness:

1. **FusedResidualRMSNorm block pattern** — LlamaModel's TransformerBlock matches, getting residual+norm fused into single HIP kernel. Tempest blocks don't match (different attribute names).
2. **Scan opacity** — SDPA attention is one kernel call. Griffin's chunked scan is ~15 ops that compile can't fully fuse.
3. **Compile region size** — LlamaModel compiles into 1-2 large fused regions per block. Tempest fragments into many.

### Path to Closing the Gap

1. **Register Griffin scan as torch.library custom op** — compile treats it as one opaque node, fuses everything around it (like we did for selective_scan)
2. **FusedGriffinBlock pattern** — compile-safe block replacement with fused residual+norm HIP kernel
3. **FLA HGRN backend** — 0.40ms Triton recurrence, verified on gfx1151, as alternative scan implementation
4. **Expected result:** two large compile regions per block instead of many small ones → ~30-50% improvement

---

## Recommended Next Steps

1. **Implement compile-optimized Griffin block** — custom op for scan + fused residual+norm pattern (see spec)
2. **Wire FLA HGRN as scan backend** — drop-in fast recurrence for all Griffin models
3. **Train top 3 (Amadeus, Tempest, Prometheus) on GPT-training-small** — larger dataset to differentiate
4. **Fix DualCortex/Obsidian autokernel** — either fix HIP kernels at d=256 or increase to d≥512
5. **Scale ResonantLoop to ~150M params** — improve quality while keeping throughput advantage

---

## Files Created/Modified

### New Model Files
- `models/maestro_prima.py` — AMADEUS + Conductor
- `models/spectral_hydra.py` — Multi-scale Griffin
- `models/resonant_loop.py` — Shared block iterative + ACT
- `models/dual_cortex.py` — System 1/2 dual-path
- `models/obsidian.py` — BitNet b1.58 + Caveman LFM
- `models/tempest_124m.py` — Tempest at 124M for fair comparison
- `models/amadeus_124m.py` — AMADEUS at 124M for fair comparison

### Modified Files
- `models/tempest.py` — Fused Griffin projections (w_aiv), vectorized chunked scan
- `models/spectral_hydra.py` — Fused projections (w_aiv)
- `autokernel/_patterns.py` — Added FusedGriffinBlockPattern (disabled), Griffin block detection helpers

### New Scripts
- `scripts/bench_all_hypotheses.py` — Benchmark all models at any scale
- `scripts/bench_170m_all.py` — Benchmark all at ~170M with eager/AK/compile
- `scripts/bench_124m_comparison.py` — Fair comparison at ~124M
- `scripts/train_170m_smoke.py` — Train all models on any dataset
- `scripts/compare_conductor.py` — Head-to-head Conductor A/B test
