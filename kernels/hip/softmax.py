"""
AutoKernel -- HIP C++ Softmax kernel.

Current kernel: Row-parallel numerically stable softmax with wavefront-level reductions.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Numerically stable softmax (subtract row max before exp)
  - Wavefront-level tree reductions via __shfl_xor for max and sum
  - half2 vectorized global memory loads for fp16 inputs
  - LDS caching in block-wide kernel: cache input as float during pass 1,
    read from LDS in pass 2 (eliminates second global memory read, ~33% BW saving)
  - Register caching in warp-per-row kernel: each lane caches ≤32 floats
  - __expf / __fdividef for fast math intrinsics
"""

KERNEL_TYPE = "softmax"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>

constexpr int WARP_SIZE = 32;  // RDNA wave32 mode

// =========================================================================
// Wavefront-level reduction helpers (wave32)
// =========================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor(val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

// =========================================================================
// Softmax kernel -- fp16 warp-per-row with register caching
// =========================================================================

// Online softmax with register caching: 1 global read instead of 2
// Pass 1: read input, cache in registers, compute running max AND exp-sum
// Pass 2: write output from cached registers (no global re-read)
// For n_cols < 1024, each lane handles at most ceil(512/32) = 16 half2 pairs
constexpr int MAX_PAIRS_PER_LANE = 16;

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
softmax_kernel_fp16(
    const half* __restrict__ input,
    half*       __restrict__ output,
    int n_rows,
    int n_cols
) {
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane        = threadIdx.x % WARP_SIZE;
    const int total_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

    const int n_pairs = n_cols / 2;
    const int n_tail  = n_cols % 2;

    // Can we cache in registers? Each lane processes ceil(n_pairs/32) pairs
    const bool use_cache = (n_pairs <= MAX_PAIRS_PER_LANE * WARP_SIZE);

    for (int row = warp_global; row < n_rows; row += total_warps) {
        const half* row_in  = input  + (long long)row * n_cols;
        half*       row_out = output + (long long)row * n_cols;

        const half2* row_in_v  = reinterpret_cast<const half2*>(row_in);
        half2*       row_out_v = reinterpret_cast<half2*>(row_out);

        float thread_max = -FLT_MAX;
        float thread_sum = 0.0f;

        // Register cache (only used when n_cols is small enough)
        float cached_lo[MAX_PAIRS_PER_LANE];
        float cached_hi[MAX_PAIRS_PER_LANE];
        int n_cached = 0;
        float tail_val = 0.0f;

        // ---- Pass 1: online softmax — find max + exp-sum ----
        for (int i = lane; i < n_pairs; i += WARP_SIZE) {
            half2 v = row_in_v[i];
            float lo = __half2float(v.x);
            float hi = __half2float(v.y);

            if (use_cache && n_cached < MAX_PAIRS_PER_LANE) {
                cached_lo[n_cached] = lo;
                cached_hi[n_cached] = hi;
                n_cached++;
            }

            float local_max = fmaxf(lo, hi);
            if (local_max > thread_max) {
                thread_sum *= __expf(thread_max - local_max);
                thread_max = local_max;
            }
            thread_sum += __expf(lo - thread_max) + __expf(hi - thread_max);
        }

        if (n_tail && lane == 0) {
            tail_val = __half2float(row_in[n_cols - 1]);
            if (tail_val > thread_max) {
                thread_sum *= __expf(thread_max - tail_val);
                thread_max = tail_val;
            }
            thread_sum += __expf(tail_val - thread_max);
        }

        float row_max = warp_reduce_max(thread_max);
        thread_sum *= __expf(thread_max - row_max);
        float row_sum = warp_reduce_sum(thread_sum);
        float inv_sum = __fdividef(1.0f, row_sum);

        // ---- Pass 2: write output ----
        if (use_cache) {
            // From cached registers (no global re-read)
            int cache_idx = 0;
            for (int i = lane; i < n_pairs; i += WARP_SIZE) {
                float lo = __expf(cached_lo[cache_idx] - row_max) * inv_sum;
                float hi = __expf(cached_hi[cache_idx] - row_max) * inv_sum;
                row_out_v[i] = __halves2half2(__float2half(lo), __float2half(hi));
                cache_idx++;
            }
        } else {
            // Re-read from global memory (2-pass fallback for wide rows)
            for (int i = lane; i < n_pairs; i += WARP_SIZE) {
                half2 v = row_in_v[i];
                float lo = __expf(__half2float(v.x) - row_max) * inv_sum;
                float hi = __expf(__half2float(v.y) - row_max) * inv_sum;
                row_out_v[i] = __halves2half2(__float2half(lo), __float2half(hi));
            }
        }
        if (n_tail && lane == 0) {
            float val = use_cache
                ? __expf(tail_val - row_max) * inv_sum
                : __expf(__half2float(row_in[n_cols - 1]) - row_max) * inv_sum;
            row_out[n_cols - 1] = __float2half(val);
        }
    }
}

// =========================================================================
// Block-wide softmax for wide rows (n_cols >= 1024)
// All 256 threads cooperate on one row via shared memory reductions
// LDS caching: input cached as float during pass 1, read from LDS in pass 2
// Eliminates second global memory read (~33% bandwidth saving)
// =========================================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
softmax_block_fp16(
    const half* __restrict__ input,
    half*       __restrict__ output,
    int n_rows,
    int n_cols
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const half* row_in  = input  + (long long)row * n_cols;
    half*       row_out = output + (long long)row * n_cols;

    // Dynamic shared memory for caching input row as float
    // Layout: [n_cols floats for cached input] [WARPS_PER_BLOCK floats for max] [WARPS_PER_BLOCK floats for sum]
    extern __shared__ float s_data[];
    float* s_cached = s_data;  // n_cols floats
    float* smem_max = s_data + n_cols;  // WARPS_PER_BLOCK floats
    float* smem_sum = smem_max + WARPS_PER_BLOCK;  // WARPS_PER_BLOCK floats

    // Pass 1: online softmax — cache input in LDS + compute max+sum
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    const int n_pairs = n_cols / 2;
    const half2* row_in_v = reinterpret_cast<const half2*>(row_in);

    for (int i = tid; i < n_pairs; i += THREADS_PER_BLOCK) {
        half2 v = row_in_v[i];
        float lo = __half2float(v.x);
        float hi = __half2float(v.y);

        // Cache in LDS
        s_cached[2 * i]     = lo;
        s_cached[2 * i + 1] = hi;

        float local_max = fmaxf(lo, hi);
        if (local_max > thread_max) {
            thread_sum *= __expf(thread_max - local_max);
            thread_max = local_max;
        }
        thread_sum += __expf(lo - thread_max) + __expf(hi - thread_max);
    }
    // Odd tail
    if ((n_cols & 1) && tid == 0) {
        float val = __half2float(row_in[n_cols - 1]);
        s_cached[n_cols - 1] = val;
        if (val > thread_max) {
            thread_sum *= __expf(thread_max - val);
            thread_max = val;
        }
        thread_sum += __expf(val - thread_max);
    }

    // Warp-level reduce max
    float warp_max = warp_reduce_max(thread_max);
    thread_sum *= __expf(thread_max - warp_max);
    float warp_sum = warp_reduce_sum(thread_sum);

    // Block-level reduce via shared memory
    if (lane == 0) {
        smem_max[warp_id] = warp_max;
        smem_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    float row_max, row_sum;
    if (warp_id == 0) {
        float m = (lane < WARPS_PER_BLOCK) ? smem_max[lane] : -FLT_MAX;
        float s = (lane < WARPS_PER_BLOCK) ? smem_sum[lane] : 0.0f;
        float block_max = warp_reduce_max(m);
        s *= __expf(m - block_max);
        float block_sum = warp_reduce_sum(s);
        if (lane == 0) {
            smem_max[0] = block_max;
            smem_sum[0] = block_sum;
        }
    }
    __syncthreads();
    row_max = smem_max[0];
    row_sum = smem_sum[0];

    float inv_sum = __fdividef(1.0f, row_sum);

    // Pass 2: write output from LDS cache (no global re-read)
    half2* row_out_v = reinterpret_cast<half2*>(row_out);
    for (int i = tid; i < n_pairs; i += THREADS_PER_BLOCK) {
        float lo = __expf(s_cached[2 * i]     - row_max) * inv_sum;
        float hi = __expf(s_cached[2 * i + 1] - row_max) * inv_sum;
        row_out_v[i] = __halves2half2(__float2half(lo), __float2half(hi));
    }
    if ((n_cols & 1) && tid == 0) {
        float val = __expf(s_cached[n_cols - 1] - row_max) * inv_sum;
        row_out[n_cols - 1] = __float2half(val);
    }
}

// =========================================================================
// C++ launcher
// =========================================================================

torch::Tensor softmax_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a GPU tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const int ndim   = input.dim();
    TORCH_CHECK(ndim >= 1, "input must have at least 1 dimension");

    const int n_cols = input.size(ndim - 1);
    const int n_rows = input.numel() / n_cols;

    auto output = torch::empty_like(input);

    // LDS budget: n_cols * sizeof(float) must fit in ~48KB for LDS-cached block kernel
    constexpr size_t LDS_BUDGET = 48 * 1024;

    TORCH_CHECK(input.dtype() == torch::kFloat16, "softmax_hip requires float16 input");

    size_t smem_needed = (n_cols + 2 * WARPS_PER_BLOCK) * sizeof(float);
    if (n_cols >= 1024 && smem_needed <= LDS_BUDGET) {
        // Wide rows that fit in LDS: block-wide with LDS caching (1 global read)
        int grid = min(n_rows, 65535);
        softmax_block_fp16<<<grid, THREADS_PER_BLOCK, smem_needed>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            n_rows, n_cols
        );
    } else if (n_cols < 1024) {
        // Narrow rows: warp-per-row with register caching (1 global read)
        const int total_warps_needed = n_rows;
        const int blocks = (total_warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const int grid = min(blocks, 65535);
        softmax_kernel_fp16<<<grid, THREADS_PER_BLOCK>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            n_rows, n_cols
        );
    } else {
        // Very wide rows (>= 1024 but exceeds LDS): block-wide without LDS caching
        // Use the LDS-cached kernel but with smem_needed=0 trick won't work,
        // so we use the warp-per-row kernel in non-cached mode via grid-stride
        // Actually: just use block-wide with enough LDS by clamping to budget
        // FALLBACK: use warp kernel which handles any size (2 global reads)
        const int total_warps_needed = n_rows;
        const int blocks = (total_warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const int grid = min(blocks, 65535);
        softmax_kernel_fp16<<<grid, THREADS_PER_BLOCK>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            n_rows, n_cols
        );
    }

    return output;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "softmax_hip")
    return _module


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.softmax_ref signature."""
    assert x.is_cuda

    orig_shape = x.shape
    orig_dtype = x.dtype

    # Non-FP16 path: fall back to PyTorch
    if orig_dtype != torch.float16:
        return torch.softmax(x, dim=-1)

    x = x.contiguous()

    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    mod = _get_module()
    out = mod.softmax_hip(x)

    return out.view(orig_shape)
