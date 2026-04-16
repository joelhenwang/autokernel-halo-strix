---
title: "Training Anti-patterns & Optimization Patterns"
domain: training
type: reference
status: active
related:
  - knowledge/hardware/amd_rdna35_strix_halo.md
  - knowledge/kernels/kernel_benchmarks.md
  - CLAUDE.md
tags: [%antipatterns, %optimization, %patterns, %rocblas, %rocm, %hip, %training, %gfx1151]
---

# Training Anti-patterns & Optimization Patterns

## Winning Optimization Patterns

1. **Kernel fusion** (6-16x): Fuse 3+ ops into one kernel. Each eliminated intermediate tensor saves 2 memory passes.
2. **Eliminate cast/allocation overhead** (8x): Fuse multi-dtype ops (int8->float->sub->mul->half) into single kernel.
3. **Online algorithms** (1.8x): Fused max+sum with rescaling eliminates a memory pass.
4. **Native fp16 intrinsics** (3.7x): `__hadd2`/`__hmul` match PyTorch rounding. fp32 intermediates + cast-back does not.
5. **Fused GEMM projections** (Griffin w_a+w_i+w_v -> single Linear): Saves 2 kernel launches/layer.
6. **Vectorized chunked scan** (no Python loops): Enables torch.compile fusion, +17% on Tempest.

## Anti-Patterns (don't repeat)

- **LDS caching for 2-pass ops**: L2 (6MB) already serves the second read. LDS only helps replacing a *separate kernel launch*.
- **`__shfl_down` when all lanes need result**: Use `__shfl_xor`. `__shfl_down` only gives lane 0 the result.
- **Binary search for top-k**: PyTorch radix sort is fundamentally better (0.25x). Don't retry.
- **Compute-bound kernels without MFMA**: matmul, flash_attention, fused_mlp — can't beat rocBLAS on scalar FMA.
- **fp32 add when reference adds fp16**: `__hadd2(x, r)` matches PyTorch, `__half2float` + add does not.
- **inf/NaN checks under `-ffast-math`**: `-ffinite-math-only` optimizes away `x != x`. Use bit-level: `(__half_as_ushort(h) & 0x7C00) == 0x7C00`.
- **fp32 topk on fp16 softmax**: Use `__hgt` (fp16 comparison) to match PyTorch tie-breaking.
- **`model.to(float16)` destroys complex buffers**: Casts complex64 `freqs_cis` to real. Save/restore complex buffers around dtype casts.
- **Sequential SSM scans**: Use chunked linear recurrence (chunk_size=64), not Python loops or `torch.associative_scan`. 5x faster. See `models/amadeus.py`.
- **autokernel on small hidden dims (d<=256)**: HIP kernel replacements cause training divergence for dual-path models with d_fast=256. Run without `--optimize-kernels` or increase to d>=512.
- **Python for-loops in chunked scan**: Causes torch.compile graph breaks. Use vectorized cross-chunk propagation via cumulative products instead (see `models/tempest.py`).
- **Adaptive softmax for training**: 3 tier matmuls is 4% slower than 1 large matmul on memory-bound hardware. Single LM head for training.
- **SSM state explosion**: Init: A_log=log(arange(1,N+1)), dt_proj bias=-4.0, dt clamped [1e-4, 0.5], B/C normalized by max(norm, 1.0).
- **torch.compile on full looped models**: Python loops with variable depth + no_grad/grad switching cause graph breaks. Compile gives 0% or negative benefit. **Fix:** compile each layer/zone independently (`model.compile_zones()`). JORMUNGANDR-HALO: 14K (eager) → 43K (AK + per-zone compile), 3.07x.
- **Passing kwargs to autokernel-replaced modules**: Autokernel's fused replacements don't accept extra kwargs (e.g., `value_bias`). Guard with `if value_bias is not None: attn(x, freqs, value_bias=value_bias) else: attn(x, freqs)`.
- **Checkpoints are always fp32**: AMP dtype (fp16/bf16) only exists transiently inside autocast. Checkpoints save fp32 master weights. Safe to load across fp16 (AMD) → bf16 (NVIDIA) training.

## rocBLAS / BLAS Optimization

rocBLAS uses Tensile scalar FMA on gfx1151. Can't beat it — shape workloads to help it:
- **Fewer, larger GEMMs.** Fuse QKV into `Linear(d, (nq+2nkv)*hd)`. Fuse gate+up in SwiGLU into `Linear(d, 2*ffn)`.
- **Pad dims to multiples of 128.** Tensile tiles: 64x64, 128x64, 128x128.
- **Strided batched > pointer-batched** for multi-head ops (less overhead on LPDDR5X).
- hipBLASLt env vars (`ROCBLAS_USE_HIPBLASLT=1`, `TENSILE_SOLUTION_SELECTION_METHOD=2`): **tested, no effect on gfx1151**. Tensile scalar FMA is already near-optimal.
- `rocblas-gemm-tune`: exists but ABI-incompatible with system rocBLAS. Needs rebuild against `/opt/rocm/core-7.12/lib/librocblas.so.5`.
- See `knowledge/hardware/amd_rdna35_strix_halo.md` section 6 for full rocBLAS/hipBLAS/hipBLASLt reference.

## DeepSpeed CPUAdam on ROCm

Offloads optimizer to CPU (AVX-512). Useful for Mode B (>2B models). Not needed for <2B — `AdamW(fused=True)` is faster.

```bash
export LIBRARY_PATH=/opt/rocm/core-7.12/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:$LD_LIBRARY_PATH
```

Setup: JIT build (not `DS_BUILD_CPU_ADAM=1`). Monkey-patch `strip_empty_entries` to filter `None` from `cxx_args()`.

## ROCm/HIP Development Reference

### Debugging Checklist
1. `__launch_bounds__(N)` — N <= 1024, smem fits 64KB LDS
2. Indexing overflow — `(long long)row * N` for >2B elements
3. Alignment — `half2` needs 4-byte, `float4` needs 16-byte. Scalar fallback for tails.
4. Dynamic smem — verify smem_bytes <= 65536
5. Dtype mismatch — kernel precision != reference precision -> large error

### Error Codes
- `hipErrorIllegalAddress` — OOB access. Check indexing near boundaries.
- `hipErrorLaunchFailure` — Kernel crash (assertion, illegal instruction, register overflow).
- `hipErrorInvalidValue` — Bad launch params (grid/block dims, smem size).
- `Timed out after 30s` — Pre-compile: `python -c "from kernels.hip.<name> import _get_module; _get_module()"`

### RDNA 3.5 vs CDNA 3
See `knowledge/hardware/amd_rdna35_strix_halo.md` section 1 for full comparison table. Key: wave32 (not 64), no MFMA, 240 GB/s (not 5.3 TB/s), 6 MB L2 (not 256 MB), 40 CUs (not 304).

### Profiling
```bash
rocprof --stats python bench.py --kernel <name>              # timing
rocprofv3 --hip-trace --hsa-trace -o trace.csv python bench.py  # counters
# FETCH_SIZE/time ~ 240 GB/s -> memory-bound. ALU_BUSY > 80% -> compute-bound.
```

### Compilation
- hipcc takes ~100s per file. Pre-compile before benchmarking. Hash-based caching in `_compile.py`.
- `-fno-fast-math -ffp-contract=off` when exact fp16 rounding matters.
- `_compile.py` auto-prepends ROCm 7.12 compat preamble to all HIP source: `__expf`->`__builtin_expf`, `sqrtf`->`__builtin_sqrtf`, `rsqrtf`/`fmaxf`/`fminf`/`__fdividef` device wrappers, `std::min`/`std::max`. No manual patching of individual kernel files needed.

### bf16 vs fp16
bf16 (bfloat16) is NOT recommended on gfx1151. AMADEUS bf16 is 24% slower (7.1K vs 9.3K tok/s), uses 32% more memory (12.1 vs 9.2 GB). bf16 + torch.compile crashes on LlamaModel (Inductor can't codegen complex RoPE ops). **Stick with fp16 + GradScaler.**
