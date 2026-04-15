---
title: "AMD RDNA 3.5 (Strix Halo, gfx1151) Optimization Reference"
domain: hardware
type: reference
status: active
related:
  - knowledge/hardware/workload_guidance.md
  - knowledge/kernels/backward_pass_optimization_research.md
tags: [%gfx1151, %rdna35, %strix-halo, %rocm, %rocblas, %lds, %wave32]
---

# AMD RDNA 3.5 (Strix Halo, gfx1151) Optimization Reference

> Use this reference when optimizing HIP C++ kernels on AMD Strix Halo APU.
> Source: https://rocm.docs.amd.com/projects/HIP/en/latest/index.html

---

## 1. Hardware Architecture

### Strix Halo Key Specs (Confirmed: Ryzen AI MAX+ 395)

| Parameter | Value |
|-----------|-------|
| Architecture | RDNA 3.5 (APU, integrated GPU) |
| ISA | gfx1151 |
| CPU | 16 Zen 5 cores / 32 threads, 3.0–5.1 GHz, 16 MB L2 + 64 MB L3, AVX-512 |
| GPU | Radeon 8060S |
| Compute Units | 40 CUs |
| Work Group Processors | 20 WGPs (2 CUs per WGP) |
| Memory | 128 GB soldered LPDDR5X, 256-bit bus (unified CPU+GPU) |
| GPU-Visible Memory | ~116 GB (reported by PyTorch, rest reserved for CPU/OS) |
| Memory Bandwidth | ~240 GB/s (LPDDR5X-7500) |
| FP16 Peak | ~59.4 TFLOPS |
| FP32 Peak | ~29.7 TFLOPS |
| LDS per CU | 64 KB |
| L2 Cache | ~6 MB |
| Wavefront Size | **32 threads** (wave32, preferred for compute) |
| Max VGPRs per SIMD | 1536 |
| SIMDs per CU | 2 |
| TDP | 45–120W configurable |
| NPU | XDNA 2, 50 TOPS INT8 |

**Verified on hardware:** PyTorch 2.10.0+rocm7.12.0, ROCm HIP 7.12, `rocminfo` confirms gfx1151.

### Key Differences from CDNA (MI300X)

| Feature | CDNA3 (MI300X) | RDNA 3.5 (Strix Halo) |
|---------|---------------|----------------------|
| GPU Type | Discrete (HBM3) | APU (LPDDR5X shared) |
| Wavefront Size | 64 | 32 (preferred) |
| Matrix Cores | MFMA (1307 TFLOPS FP16) | **None** (no MFMA, scalar FMA only) |
| Memory BW | 5300 GB/s | ~240 GB/s |
| L2 Cache | 256 MB | ~6 MB |
| CUs | 304 | 40 |
| LDS per CU | 64 KB | 64 KB |
| Power | 750W | 45-120W |

### Memory Hierarchy

```
LPDDR5X (shared with CPU, ~240 GB/s)
    ↓
L2 Cache (~6 MB, shared by all CUs)
    ↓
L1 Cache (per CU, 16-32 KB)
    ↓
LDS (64 KB per CU, explicitly managed in HIP via __shared__)
    ↓
Registers (VGPRs per SIMD, partitioned across active wavefronts)
```

**Critical:** Since bandwidth is ~240 GB/s (vs 5300 GB/s on MI300X), nearly ALL
kernels will be memory-bound. Optimization priority: minimize global memory traffic,
maximize LDS reuse, use vectorized loads (128-bit).

---

## 2. HIP C++ Programming Model

### Headers and Compilation

```cpp
#include <hip/hip_runtime.h>     // Core HIP API
#include <hip/hip_fp16.h>        // Half-precision support
#include <torch/extension.h>     // PyTorch C++ extension
```

Compile with: `hipcc -O3 -ffast-math -std=c++17 --offload-arch=gfx1151`

### Key HIP Intrinsics for RDNA 3.5

**Wavefront shuffle (wave32):**
```cpp
// No sync mask needed (unlike CUDA __shfl_*_sync)
float val = __shfl_down(val, offset);       // shift down
float val = __shfl_xor(val, offset);        // XOR shuffle
float val = __shfl(val, lane);              // broadcast from lane
```

**Fast math:**
```cpp
__expf(x)           // fast exponential
__fdividef(x, y)    // fast division
__sincosf(x, &s, &c)  // simultaneous sin/cos
rsqrtf(x)           // fast inverse sqrt
```

**Half precision:**
```cpp
__half2float(h)      // half -> float
__float2half(f)      // float -> half
__halves2half2(a, b) // pack two halves
// half2 vectorized operations available
```

### Wave32 Reduction Pattern

```cpp
constexpr int WARP_SIZE = 32;  // RDNA wave32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down(val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor(val, offset));
    return val;
}
```

### Block Reduction via LDS

```cpp
__shared__ float smem[32];  // one per wavefront

// After warp reduce:
if (lane_id == 0)
    smem[warp_id] = val;
__syncthreads();

// First wavefront reduces across all wavefronts:
if (warp_id == 0) {
    float v = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    v = warp_reduce_sum(v);
}
```

---

## 3. Optimization Strategy for Strix Halo

### Tier 1: Block/Thread Configuration
- Use **wave32** (32 threads per wavefront) as the default
- Block sizes: 128-256 threads (4-8 wavefronts per block)
- Ensure thread count is multiple of 32
- Use `__launch_bounds__(N)` to control register pressure

### Tier 2: Memory Access
- **Vectorized loads**: Use `float4` (128-bit) loads for maximum bandwidth
- **Coalesced access**: Adjacent threads access adjacent memory
- **LDS tiling**: 64 KB per CU — use smaller tiles than MI300X
- **LDS bank conflicts**: 32 banks, pad shared memory arrays (+8 elements)

### Tier 3: Compute Optimization
- All accumulation in float32 for numerical stability
- Use FMA intrinsics for multiply-accumulate
- Minimize register pressure (fewer VGPRs = more concurrent wavefronts)
- `#pragma unroll` for inner loops

### Tier 4: Occupancy
- Target 4-8 wavefronts per SIMD for latency hiding
- VGPRs per wavefront = registers_per_thread × 32
- Max concurrent wavefronts = 1536 / VGPRs_per_wavefront (per SIMD)
- Monitor with: `rocprof --stats`

### Tier 5: APU-Specific
- **No PCIe transfers**: GPU and CPU share the same memory
- **Bandwidth is the bottleneck**: ~240 GB/s vs 5300 GB/s on MI300X
- **Ridge point is very high**: nearly everything is memory-bound
- **Fusion is critical**: fuse operations to avoid extra global memory reads/writes
- **Prefetch**: Use computation to hide memory latency

### Tier 6: Kernel-Specific
- **matmul**: Shared memory tiling with double buffering. Scalar FMA accumulation.
  Tile sizes: 64x64x32 or smaller. At LLM sizes, will be memory-bound.
- **softmax**: Wavefront-per-row, half2 vectorized loads, 3-pass (max, exp+sum, normalize)
- **layernorm/rmsnorm**: Welford's algorithm, vectorized float4 loads, fused epilogue
- **flash_attention**: Tiled online softmax. Q/K/V blocked in LDS. Scalar dot products.
- **fused_mlp**: Fuse gate+up+silu+mul to avoid intermediate global writes

---

## 4. Profiling Tools

```bash
# Basic kernel timing
rocprof --stats python bench.py

# Detailed hardware counters
rocprof --hsa-trace python bench.py

# Memory bandwidth analysis
rocprof -i counters.txt python bench.py
# where counters.txt contains: pmc: TCC_HIT_sum, TCC_MISS_sum, ...

# System-level monitoring
rocm-smi --showuse --showmemuse
```

---

## 5. Common Pitfalls

1. **Wave32 vs Wave64**: RDNA defaults to wave32 for compute shaders. Do NOT use
   `0xffffffff` masks in shuffle ops (that's CUDA-specific). HIP shuffles are maskless.

2. **LDS capacity**: 64 KB per CU. Double-buffered 64x64 fp16 tiles = ~16 KB per buffer.
   Stay well under 64 KB total to allow occupancy.

3. **Low bandwidth**: Don't expect to match MI300X throughput. Focus on achieving
   high % of the ~240 GB/s peak for memory-bound kernels.

4. **No MFMA**: Don't try to use CDNA matrix instructions. Use scalar tiled GEMM
   with shared memory for matmul/attention.

5. **Unified memory**: No need for explicit host-device transfers. Tensors are
   already in shared LPDDR5X accessible by both CPU and GPU.

---

## 6. rocBLAS / hipBLAS / hipBLASLt Reference for gfx1151

> All `nn.Linear` layers in PyTorch call rocBLAS under the hood. Understanding
> rocBLAS behavior on gfx1151 is critical for architecture-level throughput
> optimization — you cannot beat rocBLAS with custom HIP matmuls on this hardware,
> but you CAN shape your workloads to make rocBLAS faster.

### 6.1 How rocBLAS Works on gfx1151

rocBLAS uses **Tensile** (auto-generated GEMM kernels) as its backend. On CDNA
GPUs (MI300X), Tensile selects MFMA-based kernels. On RDNA 3.5 (gfx1151),
**there are no MFMA instructions**, so Tensile falls back to **scalar FMA kernels**.
This is why custom matmul kernels cannot beat rocBLAS — they use the same scalar
FMA pipeline, but Tensile's generated code is already highly optimized for tiling,
register allocation, and memory access patterns.

**Backend selection:**
- **Tensile** (default): auto-generated GEMM kernels, always available
- **hipBLASLt**: higher-level library with epilogue fusion, grouped GEMM, Stream-K.
  Default for gfx12; availability on gfx1151 should be tested with `ROCBLAS_USE_HIPBLASLT=1`
- Control via env vars:
  - `ROCBLAS_USE_HIPBLASLT=0` → force Tensile
  - `ROCBLAS_USE_HIPBLASLT=1` → prefer hipBLASLt, Tensile fallback
  - `ROCBLAS_USE_HIPBLASLT_BATCHED=0` → Tensile for batched GEMM only

### 6.2 Available Mixed-Precision GEMM Types

Operation: `D = alpha * op(A) * op(B) + beta * C` (op = none, transpose, or conjugate transpose)

| Shorthand | A/B Type | C/D Type | Compute | Use Case |
|-----------|----------|----------|---------|----------|
| **HHS** | fp16 | fp16 | **fp32** | Training default (PyTorch autocast) |
| HSS | fp16 | fp32 | fp32 | Higher output precision |
| **BBS** | bf16 | bf16 | **fp32** | BF16 training |
| BSS | bf16 | fp32 | fp32 | BF16 with fp32 output |
| HHH | fp16 | fp16 | fp16 | Low-precision inference |
| SSS | fp32 | fp32 | fp32 | Full precision |
| I8II | int8 | int32 | int32 | Quantized inference |

**NOT available on gfx1151:**
- FP8 GEMM of any kind (requires gfx942/MI300X or gfx950/gfx12)
- FP4/FP6 GEMM (gfx950+ only)
- `rocblas_gemm_flags_fp16_alt_impl` (MI200-specific BF16 instruction path)
- XDL math mode (`rocblas_xf32_xdl_math_op`) — CDNA matrix core feature

### 6.3 GEMM Variants (Batched and Strided)

Three variants, in order of preference for multi-head operations:

1. **`gemm_strided_batched`** — All batches contiguous with fixed stride. Single base
   pointer + stride offsets. **Preferred for multi-head attention** (Q*K^T, attn*V)
   because memory access is sequential and predictable.

2. **`gemm_batched`** — Array of pointers to independent matrices. More flexible but
   adds pointer indirection overhead. Use when matrices are non-contiguous.

3. **`gemm`** — Single matrix multiply. Use for large fused projections (fused QKV).

**Why strided > pointer-batched on gfx1151:** With only ~240 GB/s LPDDR5X, every
byte of overhead matters. Strided batched avoids reading an array of pointers from
global memory, keeping the memory access pattern predictable for the prefetcher.

### 6.4 Matmul Shaping Rules for gfx1151

Since all GEMM on gfx1151 is memory-bound (no MFMA), the key optimization is
**reducing the number of GEMM calls and maximizing each call's size:**

1. **Fewer, larger GEMMs always win.** One GEMM of (M, N=3K, K) is faster than
   three GEMMs of (M, N=K, K). This is why fused QKV (concatenating wq/wk/wv
   projections) gives measurable speedup, and why adaptive softmax (3 tier
   matmuls) is 4% slower than one large LM head matmul.

2. **Column-major is native.** rocBLAS inherits BLAS column-major convention.
   PyTorch tensors are row-major by default. rocBLAS handles this via transpose
   flags, but be aware that `A @ B` in PyTorch becomes `B^T @ A^T` in rocBLAS
   column-major terms.

3. **Alignment matters.** Tensile's generated kernels prefer M, N, K dimensions
   that are multiples of the tile size. Common Tensile tiles on scalar FMA:
   - 64×64, 128×64, 64×128, 128×128
   - Padding d_model, head_dim, ffn_inner to multiples of 64 or 128 avoids
     tail-handling overhead

4. **The `use_cu_efficiency` flag.** `rocblas_gemm_flags_use_cu_efficiency` selects
   kernels optimized for fewer CUs — designed exactly for chips like gfx1151 (40 CUs).
   PyTorch does not set this by default. For direct rocBLAS calls or hipBLASLt, enable it.

5. **Stochastic rounding.** `rocblas_gemm_flags_stochastic_rounding` is available
   for bf16 training — may improve convergence for BF16 models on this hardware.

### 6.5 hipBLASLt Advanced Features

hipBLASLt wraps Tensile with a higher-level API. Key features beyond rocBLAS:

**Epilogue fusion (bias + activation in one launch):**
hipBLASLt can fuse post-GEMM operations into the matmul kernel:
- `D = activation(alpha * A * B + beta * C + bias)`
- Supported activations: ReLU, GELU, Swish/SiLU, Sigmoid, Clamp
- Supported epilogues: bias add, gradient computations (DGELU, bias grad)
- **Impact on gfx1151:** Eliminates a separate kernel launch for bias+activation
  after linear layers. For memory-bound hardware, fewer launches = less overhead.

**Grouped GEMM (MoE routing):**
Multiple GEMMs with different dimensions in a single launch:
- All problems must share same types and transpose operations
- Only M, N, K can vary across the group
- "Fixed MK" mode: M and K constant, only N varies (common in MoE)
- **Use case:** MoE expert routing — instead of launching N expert matmuls
  sequentially, group them into one dispatch

**Stream-K (work-centric decomposition):**
- Enable: `TENSILE_SOLUTION_SELECTION_METHOD=2`
- Distributes work evenly across CUs for non-square GEMM shapes
- Relevant for gfx1151: with only 40 CUs, load imbalance from tile-centric
  decomposition hurts more than on 304-CU MI300X
- Tuning: `TENSILE_STREAMK_DYNAMIC_GRID=6` (auto), `TENSILE_STREAMK_MAX_CUS`

### 6.6 Performance Tuning

**Kernel selection tuning (offline):**
```bash
# Step 1: Collect profile data
ROCBLAS_LAYER=4 python -c "import torch; a=torch.randn(1024,1024,device='cuda',dtype=torch.float16); torch.mm(a,a)"

# Step 2: Run the tuner on collected data
rocblas-gemm-tune --input profile_data.yaml --output tuned_solutions.yaml

# Step 3: Apply tuned solutions
export ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=tuned_solutions.yaml
```

**Solution index API (runtime tuning):**
```c
// Get all solutions for a specific problem
rocblas_gemm_ex_get_solutions(handle, transA, transB, m, n, k,
    alpha, a, a_type, lda, b, b_type, ldb,
    beta, c, c_type, ldc, d, d_type, ldd,
    compute_type, flags, solution_list, &num_solutions);

// Benchmark each, use the fastest
rocblas_gemm_ex(..., rocblas_gemm_algo_solution_index, best_solution_id, flags);
```
**Note:** Solution indices are architecture-specific and rocBLAS-version-specific.
Cannot reuse across different GPUs or library versions.

**Startup optimization:**
```c
// Call once at program start to pre-load Tensile kernels
// Avoids 100-500ms latency spike on first GEMM call
rocblas_initialize();
```

**HIP Graph capture for repeated GEMMs:**
```c
// Capture a GEMM into a HIP graph, replay it with zero launch overhead
hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
rocblas_gemm_ex(handle, ...);  // recorded, not executed
hipStreamEndCapture(stream, &graph);
hipGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
// Replay 1000x with no CPU-side overhead:
for (int i = 0; i < 1000; i++) hipGraphLaunch(graph_exec, stream);
```
All transformer layers have identical GEMM shapes → capture once, replay per layer.

### 6.7 Environment Variables Summary

| Variable | Purpose | Default |
|----------|---------|---------|
| `ROCBLAS_USE_HIPBLASLT` | Backend: 0=Tensile, 1=hipBLASLt | auto |
| `ROCBLAS_USE_HIPBLASLT_BATCHED` | Backend for batched: 0=Tensile | auto |
| `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH` | Tuned kernel selection file | none |
| `ROCBLAS_LAYER` | Logging: 2=args, 4=profile | 0 |
| `ROCBLAS_DEFAULT_ATOMICS_MODE` | 0=deterministic, 1=allow atomics | 0 |
| `ROCBLAS_CHECK_NUMERICS` | NaN/inf detection (1=full, 2=report, 4=error) | 0 |
| `TENSILE_SOLUTION_SELECTION_METHOD` | 2=Stream-K for better CU utilization | 0 |
| `TENSILE_STREAMK_DYNAMIC_GRID` | Stream-K workgroup count (6=auto) | 6 |

### 6.8 Practical Implications for Architecture Design

| Design Decision | Why | Example |
|----------------|-----|---------|
| **Fuse QKV into one projection** | 1 large GEMM > 3 small GEMMs | wqkv = Linear(d, (nq+2nkv)*hd) |
| **Fuse gate+up in SwiGLU** | 1 GEMM for [gate; up] > 2 separate | w_gate_up = Linear(d, 2*ffn) |
| **Pad dims to multiples of 128** | Aligns with Tensile tile sizes | d_model=1024, head_dim=128, ffn=2560 |
| **Prefer GQA over MHA** | Fewer KV heads = smaller batched GEMM for KV proj | 8 Q heads, 2 KV heads |
| **Single LM head > tiered softmax** | One large (B*T, V) GEMM is faster | Avoid adaptive softmax for training |
| **Use strided batched for attention** | Less overhead than pointer-batched | PyTorch SDPA already does this |
| **Consider hipBLASLt epilogue fusion** | Avoids separate bias+activation kernel | Linear + SiLU in one launch |
| **MoE: use grouped GEMM** | One dispatch for all experts | hipBLASLt GroupedGemm API |

### 6.9 Tested Results on gfx1151 (2026-04-10)

**hipBLASLt/Stream-K env vars: NO EFFECT.** Tested on SwiGLU-shaped GEMM (4096×2560)×(2560×1024):
- `ROCBLAS_USE_HIPBLASLT=1`: 0.98x (no change)
- `TENSILE_SOLUTION_SELECTION_METHOD=2` (Stream-K): 0.98x (no change)
- Both combined: 0.97x (no change)

Tensile scalar FMA is already near-optimal for these shapes on gfx1151. The env vars target CDNA/MFMA workloads.

**rocblas-gemm-tune: BLOCKED (ABI mismatch, tested twice).** Binary at `~/Desktop/ai_lab/rocm-libraries/projects/rocblas/build/release/clients/staging/rocblas-gemm-tune` links against system `/opt/rocm/core-7.12/lib/librocblas.so.5` but crashes with "Could not initialize Tensile host" — the Tensile host C++ code compiled into the client doesn't match the serialization format of the system's `.dat`/`.co` files (151 gfx1151 .hsaco files at `/opt/rocm/core-7.12/lib/rocblas/library/`). The system rocBLAS was installed via pip (`rocm-sdk-libraries-gfx1151 7.12.0`), not from the same source tree as the client. `ROCBLAS_TENSILE_LIBPATH` env var doesn't help — the failure is in Tensile host init, before kernel loading. **Fix:** Build the client from the exact same rocBLAS source+commit that produced the system `librocblas.so.5`, or find a matching `rocblas-clients` package for ROCm 7.12.

---

## 7. ROCm 7.12 Source Build Patching (gfx1151)

> Every external CUDA/HIP package that compiles device code needs patching for ROCm 7.12 on gfx1151. This section documents the systematic issue and fix pattern.

### 7.1 The Problem: Bare Math Functions in Device Code

ROCm 7.12's HIP compiler rejects bare C math functions (`expf`, `exp2f`, `powf`, `__logf`, `expm1f`, `sincosf`, `log1pf`, `fabsf`) in device code on gfx1151. These are treated as host-only functions. The fix: replace with `__builtin_` equivalents.

**Affected packages (verified):**
- **causal-conv1d** 1.6.1 — `csrc/` files. Script: `scripts/install_causal_conv1d_rocm.sh`
- **mamba-ssm** 2.3.0 — `csrc/` files. Script: `scripts/install_mamba_ssm_rocm.sh`
- **aiter** (CK headers) — 24 files in `3rdparty/composable_kernel/include/` and `csrc/`. Script: `scripts/patch_aiter_ck_rocm.sh`

### 7.2 The Patch Pattern

```bash
# Replace bare math functions with __builtin_ equivalents
# Use negative lookbehind to avoid double-patching (__builtin_amdgcn_exp2f etc.)
sed -i -E 's/(?<!builtin_)(?<!amdgcn_)\bexp2f\(/__builtin_exp2f(/g' "$file"
sed -i -E 's/(?<!builtin_)(?<!amdgcn_)\bexpf\(/__builtin_expf(/g' "$file"
sed -i -E 's/(?<!builtin_)\bpowf\(/__builtin_powf(/g' "$file"
sed -i -E 's/(?<!builtin_)\b__logf\(/__builtin___logf(/g' "$file"
sed -i -E 's/(?<!builtin_)\bexpm1f\(/__builtin_expm1f(/g' "$file"
sed -i -E 's/(?<!builtin_)\bsincosf\(/__builtin_sincosf(/g' "$file"
sed -i -E 's/(?<!builtin_)\blog1pf\(/__builtin_log1pf(/g' "$file"
```

**Also handle `std::expf(` → `__builtin_expf(`** (some CK headers use `std::` prefix).

After patching, always clear JIT build caches:
```bash
rm -rf ~/.triton/cache/
rm -rf aiter/aiter/jit/build/*/  # aiter JIT cache
```

### 7.3 hipcc Location

ROCm 7.12 installs hipcc at `/opt/rocm/core-7.12/bin/hipcc`, not `/opt/rocm/bin/hipcc`. Many packages hardcode the latter. Fix:
```bash
sudo ln -sf /opt/rocm/core-7.12/bin/hipcc /opt/rocm/bin/hipcc
```

### 7.4 Required Environment Variables

Add to `.venv/bin/activate` for all ROCm source builds:
```bash
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm/core-7.12
export HIP_PATH=/opt/rocm/core-7.12
export HIP_PLATFORM=amd
export CPLUS_INCLUDE_PATH=/opt/rocm/core-7.12/include:${CPLUS_INCLUDE_PATH:-}
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:/opt/rocm/core-7.12/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/opt/rocm/core-7.12/lib:/opt/rocm/core-7.12/lib64:${LIBRARY_PATH:-}
```

**Critical:** `CPLUS_INCLUDE_PATH` must include `/opt/rocm/core-7.12/include` — aiter's JIT build needs `rocprim/rocprim.hpp` from this path.

---

## 8. aiter (AMD AI Tensor Engine Runtime) on gfx1151

### 8.1 What Works

- **Triton-based flash_attn forward** (0.25ms, 4.2x vs SDPA) — via `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
- **Triton-based flash_attn backward** — works but 66% slower than SDPA backward on gfx11
- **`module_aiter_core`** — CK JIT builds successfully (after math builtin patching)

### 8.2 What Does NOT Work

- **CK attention backward** — fundamentally not supported on gfx11 architecture. aiter correctly falls back to Triton.
- **HIP ops (RMSNorm, RoPE, activation, quantization)** — `module_rmsnorm`, `module_activation` JIT build fails. Root cause: aiter's "opus" framework (`csrc/include/opus/opus.hpp`) references `mfma_adaptor` — a CDNA-only type. This is a fundamental architecture dependency, not a patchable issue.
- **`silu_and_mul`, `rms_norm`** etc via CK path — all depend on modules that fail to build.

### 8.3 The Hybrid Attention Breakthrough

Pure flash_attn fwd+bwd is 26% slower than SDPA for training. But we can combine the best of both:

| Backend | Forward | Backward | Fwd+Bwd | vs SDPA |
|---------|---------|----------|---------|---------|
| **hybrid_attention** | 0.25ms (flash) | 2.92ms (SDPA) | **3.50ms** | **8.9% faster** |
| SDPA | 1.07ms | 2.64ms | 3.84ms | baseline |
| flash_attn (Triton) | 0.25ms | 4.39ms | 4.84ms | 26% slower |

**How it works:** `kernels/hip/hybrid_attention.py` uses `_flash_attn_forward` for the fast forward pass, then passes the softmax logsumexp (`softmax_lse`) directly to `torch.ops.aten._flash_attention_backward` (SDPA's CK-based backward). The logsumexp tensor (B, H, T, float32) is the shared interface — both flash_attn and SDPA compute it identically, enabling zero-recompute backward.

**Gradient accuracy:** max_diff=0.002 (within fp16 tolerance).

**Triton backward is optimal at 32x32:** Exhaustive autotune tested 4 configs (32x32, 32x64, 64x64, 32x128). The original 32x32x32x32 RDNA config is optimal — RDNA 3.5 without MFMA can't exploit larger tiles. Split backward mode (3 kernel launches) is catastrophically slower (15.39ms). The `exp2` trick is already in use.

---

## 9. Unified Memory Training Characteristics

### 9.1 Data Loading is Negligible

On gfx1151, CPU and GPU share LPDDR5X. There is no PCIe transfer. Profiling shows data loading is **0.4% of training step time**. `pin_memory=True` is effectively a no-op (memory is already shared).

### 9.2 CPU/GPU Bandwidth Competition

Unlike discrete GPUs, heavy CPU activity during GPU compute steals memory bandwidth. This is a theoretical concern but hard to measure in practice. The profiler shows backward pass dominates (53% of step), with no visible data loading stalls.

### 9.3 Batch Size Sweet Spot

Tested on AMADEUS 243.8M (seq=256, eager):

| batch_size | tok/s | Notes |
|-----------|-------|-------|
| 4 | 5,374 | Low GPU utilization |
| 8 | 7,220 | Good scaling |
| **16** | **7,539** | **Peak** — L2 cache fits activations |
| 32 | 7,518 | Plateau — spilling to DRAM |
| 64 | 7,517 | No further gain |

Peak at batch=16 suggests L2 cache (6 MB) is the bottleneck: activations for batch=16 fit well, larger batches spill without gaining more parallelism from the 40 CUs.

### 9.4 Training Step Breakdown (AMADEUS 243.8M, eager)

| Phase | Time (ms) | % |
|-------|-----------|---|
| backward | 102.6 | 53% |
| forward | 37.9 | 20% |
| optimizer_step (fused AdamW) | 35.8 | 19% |
| grad_clip | 13.6 | 7% |
| loss (cross_entropy) | 2.5 | 1% |
| data_load | 0.8 | 0.4% |

**Key insight:** Backward dominates. The optimizer step is surprisingly expensive (19%) — fused AdamW still reads/writes all parameters. For >2B models, CPUAdam offloading (DeepSpeed) moves this to CPU AVX-512.
