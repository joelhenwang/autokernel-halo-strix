# Explore aiter Ops + rocBLAS Tuning for gfx1151

**Date:** 2026-04-10
**Status:** Design approved, pending implementation
**Workstream:** C (of A/B/C/D optimization roadmap)
**Depends on:** Workstream A (need baselines to measure improvement)

## Problem

aiter provides HIP-compiled ops beyond flash attention (fused RMSNorm, RoPE, quantization, MoE gating) that could supplement or replace our autokernel HIP kernels. rocBLAS offers per-problem kernel tuning and hipBLASLt features (epilogue fusion, Stream-K, grouped GEMM) we haven't tested. Both are potential free performance gains.

## Scope

### Part 1: Benchmark aiter's HIP ops vs our autokernel kernels

aiter's `module_aiter_core` (successfully compiled) exposes ops that overlap with our autokernel kernels:

| aiter op | Our autokernel equivalent | Our speedup | Worth testing? |
|----------|--------------------------|-------------|----------------|
| Fused RMSNorm | fused_residual_add_rmsnorm (6.6x) | High | Yes — compare |
| RoPE | rotary_embedding (3.7x) | Medium | Yes — compare |
| Fused bias+activation | fused_bias_silu (1.9x) | Medium | Yes — compare |
| MoE gating | moe_gating (3.5x) | Medium | Only if MoE architectures used |
| Quantization (int4/int8) | dequantize_int4 (16.3x) | High | Yes — for inference |

**Benchmark script:** Create `scripts/bench_aiter_ops.py` that:
1. Discovers which aiter ops are available (`torch.ops.aiter.*`)
2. Benchmarks each against our autokernel equivalent and PyTorch baseline
3. Reports: correctness (atol), speed, backward support

### Part 2: rocBLAS per-problem tuning

rocBLAS's `rocblas-gemm-tune` finds optimal Tensile kernels for specific GEMM shapes. Our models use a small set of fixed GEMM shapes:

| GEMM (M, N, K) | Where | Frequency |
|-----------------|-------|-----------|
| (B*T, 2*ffn, d_model) = (*, 5120, 1024) | SwiGLU gate+up | Every layer |
| (B*T, d_model, ffn) = (*, 1024, 2560) | SwiGLU down | Every layer |
| (B*T, d_model, d_model) = (*, 1024, 1024) | out_proj, QKV | Every layer |
| (B*T, vocab, d_model) = (*, 50257, 1024) | LM head | Once |

**Tuning procedure:**
1. Collect profile data: `ROCBLAS_LAYER=4 python -m halo_training --smoke`
2. Run tuner: `rocblas-gemm-tune --input profile.yaml --output tuned.yaml`
3. Apply: `export ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=tuned.yaml`
4. Re-benchmark

### Part 3: hipBLASLt features

Test availability and impact of:
- **Stream-K:** `TENSILE_SOLUTION_SELECTION_METHOD=2` — better CU utilization for non-square GEMMs
- **Epilogue fusion:** `ROCBLAS_USE_HIPBLASLT=1` — fuses bias+activation into GEMM
- Both are env-var toggles — zero code changes needed

### Part 4: torch.compile interaction

Test if any of the above compose with `torch.compile(mode="default")`. Some rocBLAS/hipBLASLt env vars may interfere with Inductor's own GEMM dispatch.

## Implementation

### File to create
| File | Purpose |
|------|---------|
| `scripts/bench_aiter_ops.py` | Benchmark aiter HIP ops vs autokernel |
| `scripts/tune_rocblas_gemm.sh` | rocBLAS GEMM tuning workflow |

### Execution order
1. `bench_aiter_ops.py` — discover what's available, benchmark
2. `tune_rocblas_gemm.sh` — collect + tune + apply
3. Test Stream-K and hipBLASLt env vars
4. Run full training with best combination

## Expected Impact

| Optimization | Expected gain | Effort |
|-------------|---------------|--------|
| aiter fused ops | 0-10% (may match autokernel) | Low (benchmark only) |
| rocBLAS tuning | 2-5% (GEMM-specific) | Low (tooling) |
| Stream-K | 1-3% (CU utilization) | Trivial (env var) |
| hipBLASLt epilogue | 1-5% (fewer kernel launches) | Trivial (env var) |

## Verification

Before/after training tok/s comparison on the same architecture (winner from Workstream A).
