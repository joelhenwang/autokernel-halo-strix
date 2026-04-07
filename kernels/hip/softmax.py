"""
AutoKernel -- HIP C++ Softmax kernel.

Current kernel: Row-parallel numerically stable softmax with wavefront-level reductions.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Numerically stable softmax (subtract row max before exp)
  - Wavefront-level tree reductions via __shfl_xor for max and sum
  - Grid-stride loop over rows for large row counts
  - half2 vectorized global memory loads for fp16 inputs
  - One wavefront per row; supports arbitrary (non-power-of-2) row lengths
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

// =========================================================================
// Softmax kernel: one wavefront per row, grid-stride over rows
// =========================================================================

constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
softmax_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n_rows,
    int n_cols
) {
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane        = threadIdx.x % WARP_SIZE;
    const int total_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

    for (int row = warp_global; row < n_rows; row += total_warps) {
        const float* row_in  = input  + (long long)row * n_cols;
        float*       row_out = output + (long long)row * n_cols;

        // ---- Pass 1: find row max (for numerical stability) ----
        float thread_max = -FLT_MAX;
        for (int col = lane; col < n_cols; col += WARP_SIZE) {
            thread_max = fmaxf(thread_max, row_in[col]);
        }
        float row_max = warp_reduce_max(thread_max);

        // ---- Pass 2: compute exp(x - max) and accumulate sum ----
        float thread_sum = 0.0f;
        for (int col = lane; col < n_cols; col += WARP_SIZE) {
            float val = __expf(row_in[col] - row_max);
            row_out[col] = val;
            thread_sum += val;
        }
        float row_sum = warp_reduce_sum(thread_sum);

        // ---- Pass 3: divide by sum ----
        float inv_sum = __fdividef(1.0f, row_sum);
        for (int col = lane; col < n_cols; col += WARP_SIZE) {
            row_out[col] *= inv_sum;
        }
    }
}

// =========================================================================
// Softmax kernel -- fp16 path with half2 vectorized loads
// =========================================================================

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

    for (int row = warp_global; row < n_rows; row += total_warps) {
        const half* row_in  = input  + (long long)row * n_cols;
        half*       row_out = output + (long long)row * n_cols;

        const half2* row_in_v  = reinterpret_cast<const half2*>(row_in);
        half2*       row_out_v = reinterpret_cast<half2*>(row_out);

        // ---- Pass 1: find row max ----
        float thread_max = -FLT_MAX;

        for (int i = lane; i < n_pairs; i += WARP_SIZE) {
            half2 v = row_in_v[i];
            float lo = __half2float(v.x);
            float hi = __half2float(v.y);
            thread_max = fmaxf(thread_max, fmaxf(lo, hi));
        }
        if (n_tail && lane == 0) {
            float last = __half2float(row_in[n_cols - 1]);
            thread_max = fmaxf(thread_max, last);
        }

        float row_max = warp_reduce_max(thread_max);

        // ---- Pass 2: exp(x - max) and accumulate sum ----
        float thread_sum = 0.0f;

        for (int i = lane; i < n_pairs; i += WARP_SIZE) {
            half2 v = row_in_v[i];
            float lo = __expf(__half2float(v.x) - row_max);
            float hi = __expf(__half2float(v.y) - row_max);
            row_out_v[i] = __halves2half2(__float2half(lo), __float2half(hi));
            thread_sum += lo + hi;
        }
        if (n_tail && lane == 0) {
            float val = __expf(__half2float(row_in[n_cols - 1]) - row_max);
            row_out[n_cols - 1] = __float2half(val);
            thread_sum += val;
        }

        float row_sum = warp_reduce_sum(thread_sum);

        // ---- Pass 3: divide by sum ----
        float inv_sum = __fdividef(1.0f, row_sum);

        for (int i = lane; i < n_pairs; i += WARP_SIZE) {
            half2 v = row_out_v[i];
            float lo = __half2float(v.x) * inv_sum;
            float hi = __half2float(v.y) * inv_sum;
            row_out_v[i] = __halves2half2(__float2half(lo), __float2half(hi));
        }
        if (n_tail && lane == 0) {
            float val = __half2float(row_out[n_cols - 1]) * inv_sum;
            row_out[n_cols - 1] = __float2half(val);
        }
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

    const int total_warps_needed = n_rows;
    const int blocks = (total_warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int grid = min(blocks, 65535);

    if (input.dtype() == torch::kFloat16) {
        softmax_kernel_fp16<<<grid, THREADS_PER_BLOCK>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            n_rows, n_cols
        );
    } else if (input.dtype() == torch::kBFloat16) {
        auto input_f32  = input.to(torch::kFloat32);
        auto output_f32 = torch::empty_like(input_f32);
        softmax_kernel<<<grid, THREADS_PER_BLOCK>>>(
            input_f32.data_ptr<float>(),
            output_f32.data_ptr<float>(),
            n_rows, n_cols
        );
        output = output_f32.to(torch::kBFloat16);
    } else {
        softmax_kernel<<<grid, THREADS_PER_BLOCK>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
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

    x = x.contiguous()

    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    needs_cast = orig_dtype not in (torch.float16, torch.bfloat16, torch.float32)
    if needs_cast:
        x = x.to(torch.float32)

    mod = _get_module()
    out = mod.softmax_hip(x)

    if needs_cast:
        out = out.to(orig_dtype)

    return out.view(orig_shape)
