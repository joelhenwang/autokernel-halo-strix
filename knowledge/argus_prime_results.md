# ARGUS-PRIME — Full Results & Lessons Learned (2026-04-13)

## Architecture

ARGUS-PRIME evolves from ARGUS by aligning with LFM2.5's actual architecture:
- **10:6 ShortConv/GQA ratio** (16 layers: 10 GatedConv + 6 GQA attention)
- **GQA positions:** 0-indexed {2, 5, 7, 9, 12, 15} — global context every ~2.5 layers
- **QK-Norm** on attention (L2-normalize Q,K before attention, LFM2 technique)
- **d_model=768, d_conv=512, ffn_inner=2816** (3.7x expansion, LFM2-aligned)
- **Surgical TTT:** 1 layer at position 15 only ("sniper" — maximum leverage before LM head)
- **Momentum residual** inlined (compile-friendly, FusedGriffinBlock pattern)
- **Hybrid flash_sdpa attention** (8.9% faster: flash_attn forward + SDPA backward)
- **Removed:** Engram, MatFormer (unproven at 16M tokens, hurt throughput)

## 6 Ablation Variants

| Variant | TTT Config | FiLM | Status |
|---------|-----------|------|--------|
| **B0** | 1 layer (15), single-step | No | **STABLE, best throughput** |
| B1 | 1 layer (15), single-step | Yes | Stable (FiLM fixed: apply after layer) |
| B2 | 1 layer (15), multi-step 3x | No | **UNSTABLE — NaN at step 22** |
| B3 | 1 layer (15), multi-step 3x | Yes | **UNSTABLE — NaN at step 12** |
| B4 | 2 layers (7,15), single-step | No | Stable, slightly better loss |
| B5 | 2 layers (7,15), single-step | Yes | Not tested |

**Multi-step TTT (B2/B3) fails for from-scratch training.** The 3 iterative gradient steps amplify
errors when weights are random. Multi-step likely only works for fine-tuning pretrained models
(as in the ByteDance paper). Steps were unrolled (no for loop) for compile safety.

**FiLM collapse (B1, fixed):** Original placement applied FiLM BEFORE layer forward → amplified FFN
gradients → grads went to zero. Fix: apply FiLM AFTER layer forward.

## Throughput Optimization Journey

| Optimization | tok/s | Change | Verdict |
|-------------|-------|--------|---------|
| B0 original (d_conv=768) | 16,797 | baseline | — |
| **d_conv 768→512** | **17,968** | **+7.0%** | **Winner: less memory-bound conv** |
| 14 layers + ffn=3328 | 18,205 | +8.4% | Small gain, fewer layers |
| Fused forward HIP kernel | 17,863 | +0% | Forward not bottleneck |
| Inlined momentum+RMSNorm | ~17,000 | +1% | Small compile gain |
| Custom op + PyTorch fallback bwd | 17,335 | -3.5% | Slower than autograd native |
| Custom op + HIP backward kernel | 17,604 | -2.0% | Atomic overhead + recompute |
| Fused QKV projection | 17,367 | -3.5% | Breaks autokernel FusedQKV+RoPE |
| hipBLASLt + Stream-K env vars | 17,836 | -0.9% | No effect (Tensile optimal) |
| Wider FFN (3328) only | 16,967 | -5.6% | Higher MFU but slower tok/s |

**Best: 18.0K tok/s at d_conv=512, 30.5% MFU, ~155M params.**

## Key Lessons

### Why 50% MFU is impossible with ShortConv
ShortConv layers use depthwise Conv1d (per-channel, no cross-channel compute) + element-wise gating.
These are inherently memory-bound — near-zero arithmetic intensity. On gfx1151 (59.4 TFLOPS,
240 GB/s), MFU only counts useful FLOPs against peak. Memory-bound ops eat wall time without
contributing to MFU. With 10/16 layers being ShortConv, the MFU ceiling is ~30-35%.

LlamaModel hits 54% because ALL ops are dense matmuls (QKV, attention, FFN).

### Why fused backward kernels didn't help
1. **autograd + causal_conv1d backward is already fast** — causal_conv1d library has optimized backward
2. **Custom op overhead:** save/restore tensors + Python dispatch adds latency
3. **Recomputation cost:** HIP backward recomputes y,z from proj_out (6 extra FP ops/channel)
4. **Atomic contention:** grad_conv_weight accumulation via atomicAdd across 4096 rows

The lesson: custom backward kernels only help when the PyTorch default is clearly suboptimal
(like RMSNorm where 5 separate ops fuse into 1). For GatedConv, autograd already does the right thing.

### Why fused QKV hurts
autokernel's FusedQKVPattern matches `(wq, wk, wv, wo)` and fuses QKV projection + RoPE application
into one optimized graph region. Manual QKV fusion (one `wqkv` Linear) breaks this pattern match.
The loss of autokernel's combined QKV+RoPE optimization (-3.7x on RoPE) outweighs saving 2 GEMM launches.

### GEMM-level tuning is exhausted on gfx1151
- hipBLASLt: already default in PyTorch (`Cublaslt` backend)
- Stream-K: tested, no effect on 40-CU scalar FMA
- use_cu_efficiency: C API only, not accessible from Python
- Dimension alignment: already multiples of 128
- Tensile scalar FMA is near-optimal — cannot be improved

## Muon Hyperparameter Sweep (GPT-training-small)

| LR (base) | Block | Batch×Accum | Real Loss | tok/s |
|-----------|-------|-------------|-----------|-------|
| 0.0005 | 256 | 16×4 | 5.29 | 16.8K |
| **0.0008** | **256** | **16×4** | **5.20** | **16.9K** |
| **0.0012** | **256** | **16×4** | **5.18** | **16.8K** |
| **0.0015** | **256** | **16×4** | **5.18** | **16.8K** |
| 0.0012 | 512 | 16×4 | 5.32 | 17.0K |
| 0.0012 | 1024 | 16×4 | 5.81 | 16.4K |
| 0.0012 | 256 | 16×8 | 5.27 | 17.7K |

**Optimal: LR 0.0012, block=256, batch=16×4.** Muon LR scales to ~0.0075 internally.

- Higher LR (0.0012-0.0015) converges faster than default 0.0008
- Larger blocks hurt quality in fixed time (fewer optimizer steps)
- Larger batch is neutral — no quality benefit from 128 vs 64

## Profiling Data

**ARGUS-PRIME B0 (eager + autokernel, batch=16, block=256):**

| Phase | Time | % |
|-------|------|---|
| Forward | 87.3ms | 31.9% |
| **Backward** | **150.4ms** | **54.9%** |
| Optimizer | 17.4ms | 6.4% |
| Other | 18.6ms | 6.8% |
| **Total** | **273.7ms** | 100% |

Backward dropped from ARGUS's 70.8% to 54.9% by removing 3 TTT layers.

## Files

| File | Purpose |
|------|---------|
| `models/argus_prime.py` | All 6 variants + ArgusPrimeWide, ArgusPrime14L, ArgusPrimeMini |
| `models/argus.py` | Original ARGUS (predecessor) |
| `kernels/hip/fused_gated_conv.py` | Forward HIP kernel (used but no throughput gain) |
| `kernels/hip/fused_gated_conv_backward.py` | Backward HIP kernel (tested, slower than native) |
| `kernels/hip/_torch_ops.py` | Custom op registration with autograd |

## GPT-training-small Run (in progress)

Config: B0, d_conv=512, lr=0.0012, Muon, block=256, batch=16×4, compile+autokernel, 2 epochs.
Estimated: ~3.6 hours for 222M tokens at 16.9K tok/s. Checkpoint saved.
