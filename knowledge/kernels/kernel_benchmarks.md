---
title: "Kernel Benchmarks"
domain: kernels
type: results
status: active
related:
  - knowledge/kernels/fused_kernels.md
  - knowledge/kernels/external_kernels.md
  - knowledge/hardware/amd_rdna35_strix_halo.md
  - REPORT.md
tags: [%kernels, %benchmarks, %hip, %speedup, %gfx1151]
---

# Kernel Benchmarks (vs PyTorch on gfx1151)

## HIP Kernel Speeds

- **6-16x**: dequantize_int4 (16.3x), fused_residual_add_layernorm (10.7x), prefix_scan (8.4x), dequantize_int8 (8.1x), fused_residual_add_rmsnorm (6.6x)
- **1.5-4x**: rotary_embedding (3.7x), moe_gating (3.5x), rmsnorm (3.3x), fused_bias_silu/gelu (1.9x), cross_entropy (1.8x), silu_gate_mul (1.6x)
- **~1x**: reduce, layernorm, silu, gelu, softmax
- **<0.3x (skip)**: matmul, flash_attention (standard HIP build), fused_mlp, top_k_sampling

## End-to-End Speedups (LlamaModel7B, 5.9B)

- **1.189x** `autokernel.optimize(model, compile=True)` — best overall
- **1.162x** `--torch-compile` alone (not composable with manual kernel replacements)
- **1.053x** `--incremental --fused-qkv` (4 HIP kernels + fused QKV)
- **9.4 tok/s** decode (KV-cache, 106ms/token)

## Training Results

| Model | Config | tok/s | MFU |
|-------|--------|-------|-----|
| LlamaModel 124.7M | compile + autokernel | **43K** | **54%** |
| LlamaModel 124.7M | eager | 14.5K | 17% |
| AMADEUS 243.8M | autokernel + compile + HIP scan | **10.4K** | **26%** |
| AMADEUS 243.8M | eager, chunked scan | 6.5K | 16% |
| AMADEUS 243.8M | sequential scan | 1.3K | 4% |
| Tempest 244.5M | compile + autokernel | **11,049** | **27.3%** |
| Tempest 124.7M | compile + autokernel + fused block | **22,008** | **27.5%** |
| Tempest + MatFormer 244.5M | compile + autokernel | **8,166** | **20.2%** |

AMADEUS best loss: 12.18 (BPB 4.88) after 2 epochs on BabyLM (30.8M tokens, 79 min).

## Hypothesis Architecture Training (2026-04-12, ~170M params, BabyLM 2 epochs)

| Model | Params | Val Loss | tok/s | Notes |
|-------|--------|----------|-------|-------|
| **Amadeus** | 157.7M | **2.90** | 13,203 | Best quality |
| **MaestroPrima** | 157.8M | **2.90** | 12,896 | Conductor adds negligible benefit |
| Tempest | 176.8M | 2.98 | 12,952 | Best pure Griffin |
| Virtuoso | 180.8M | 2.99 | 11,165 | PLE+MatFormer no quality benefit at this scale |
| Prometheus | 174.3M | 3.00 | 13,066 | 2 attention layers don't help |
| SpectralHydra | 176.8M | 3.19 | 10,323 | Decay spectrum needs tuning |
| ResonantLoop | 50.7M | 3.42 | **15,907** | Throughput champion, quality-limited by params |
| DualCortex | 125.2M | 5.44 | 32,426 | **FAILED** — autokernel breaks small dims |
| Obsidian | 124.0M | 5.71 | 34,115 | **FAILED** — autokernel breaks small dims |

## LAZARUS TTT Fast Weights (2026-04-13)

| Variant | Params | tok/s | Notes |
|---------|--------|-------|-------|
| LAZARUS-A (full) | 247.5M | 8,309 | Too heavy — hit budget at 68% epoch 1 |
| **LazarusLite** | 160.2M | **12,150** | 92% of AMADEUS speed, 2 TTT layers, no momentum |
| AMADEUS (baseline) | 157.7M | 13,203 | Quality champion |

TTT overhead: ~8% throughput + 1.3GB memory.

See `knowledge/architectures/hypothesis_buildout_results.md` for full analysis.

## Muon Optimizer (2026-04-13)

**A/B comparison (10-min budget, BabyLM, batch=16, block=256, accum=4):**

| Model | Optimizer | tok/s | Steps | Best Loss | Memory |
|-------|-----------|-------|-------|-----------|--------|
| LlamaModel 124.7M | AdamW | **49,711** | 1,004 | 17.58 | 2.6 GB |
| LlamaModel 124.7M | Muon (lr=0.005) | 48,131 | 1,004 | **17.48** | 2.6 GB |
| AMADEUS 243.8M | AdamW | **9,304** | 340 | 14.93 | 9.2 GB |
| AMADEUS 243.8M | Muon (lr=0.005) | 8,888 | 325 | **14.77** | **9.0 GB** |

Note: displayed loss inflated 4x by accum_steps logging (real AMADEUS loss ≈ 3.7).

## Compile-Optimized Griffin Block (2026-04-12)

| Config | Before | After | Improvement |
|--------|--------|-------|-------------|
| Tempest124M AK+compile (fwd+bwd only) | 20,184 tok/s | **71,024 tok/s** | **3.52x** |
| Tempest124M AK only (fp16, fwd+bwd) | 16,674 tok/s | **59,500 tok/s** | **3.57x** |
| Tempest124M training (AMP+optim) | ~12,952 tok/s | **22,358 tok/s** | **1.73x** |

## PLE + MatFormer Ablation

- **PLE Path A:** Best quality (loss 22.65, -1.5% vs base) at 3% throughput cost
- **MatFormer:** Free (+0.2% tok/s), negligible quality cost, enables elastic inference
- **PLE Path B / A+B:** No benefit — drop
- See `knowledge/architectures/ple_ablation_results.md` for full results
