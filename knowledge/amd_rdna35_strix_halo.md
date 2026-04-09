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
