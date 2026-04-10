# Explore aiter Ops + rocBLAS Tuning for gfx1151

**Date:** 2026-04-10
**Status:** Implemented and tested — see Results section
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

## Results (2026-04-10)

### Part 1: aiter HIP ops — BLOCKED on gfx1151

aiter's CK/HIP ops (RMSNorm, RoPE, activation, quantization) **do not build on gfx1151**. Root causes:
- **`mfma_adaptor` undefined**: aiter's "opus" framework (`csrc/include/opus/opus.hpp:2827`) references MFMA (Matrix FMA) instructions that don't exist on RDNA 3.5. This is a fundamental CDNA-only dependency, not a patchable issue.
- **Bare math functions** (`fabsf` in `activation_kernels.cu:370`): Same ROCm 7.12 issue we patched for CK headers, but spread across `csrc/` files too.
- **`module_rmsnorm` / `module_activation` JIT build fails**: Both modules fail to compile for gfx1151.

Only aiter's **Triton-based ops** work (flash_attn via `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`). The CK/HIP ops are CDNA-targeted.

**Conclusion:** Our autokernel HIP kernels remain the best option for fused ops on gfx1151. Benchmarked:
- autokernel RMSNorm: **48x** vs PyTorch (0.01ms vs 0.44ms)
- autokernel fused_bias_silu: **2.0x** vs PyTorch (0.19ms vs 0.38ms)

### Part 2: rocBLAS per-problem tuning — NOT AVAILABLE

`rocblas-bench` and `rocblas-gemm-tune` are not installed. They require the `rocblas-clients` package which is separate from the base ROCm install. Only the rocBLAS library itself is present at `/opt/rocm/core-7.12/lib/rocblas/library/`.

To install: `sudo apt install rocblas-clients` (if available for ROCm 7.12), or build from source.

### Part 3: hipBLASLt env vars — NO EFFECT

Tested on a (4096, 2560) × (2560, 1024) GEMM (SwiGLU shape):
- **Baseline**: 0.61ms
- **Stream-K** (`TENSILE_SOLUTION_SELECTION_METHOD=2`): 0.62ms (0.98x) — no improvement
- **hipBLASLt** (`ROCBLAS_USE_HIPBLASLT=1`): 0.63ms (0.98x) — no improvement
- **Both**: 0.63ms (0.97x) — no improvement

Tensile scalar FMA on gfx1151 is already near-optimal for these shapes. The env vars target CDNA/MFMA workloads.

### Part 4: torch.compile interaction — N/A

No aiter ops or rocBLAS tuning to compose with torch.compile.

## Verification

Before/after training tok/s comparison on the same architecture (winner from Workstream A).
