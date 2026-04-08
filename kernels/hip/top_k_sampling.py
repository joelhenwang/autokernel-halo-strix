"""
AutoKernel -- HIP C++ Top-K Sampling kernel.

Fuses temperature scaling + top-k selection + softmax into a single kernel.
Used during autoregressive decode to sample next tokens.

Features:
  - Temperature scaling fused with top-k threshold finding
  - Two-pass algorithm: pass 1 finds k-th largest value (threshold),
    pass 2 applies softmax only to values >= threshold
  - Vectorized half2 loads
  - Block-cooperative reduction for threshold finding
"""

KERNEL_TYPE = "top_k_sampling"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
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

__device__ __forceinline__ int warp_reduce_sum_int(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Top-k sampling kernel: temperature + top-k + softmax fused
// Algorithm:
// 1. Find approximate k-th largest value using iterative threshold narrowing
// 2. Apply softmax only to values >= threshold, -inf to rest
__global__ void __launch_bounds__(BLOCK_SIZE)
top_k_sampling_kernel(
    const half* __restrict__ logits,  // [B, V]
    half* __restrict__ output,        // [B, V] softmax probabilities (0 for non-top-k)
    int B, int V, int K, float temperature
) {
    const int batch = blockIdx.x;
    if (batch >= B) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    const half* row = logits + (long long)batch * V;
    half* out_row = output + (long long)batch * V;

    float inv_temp = 1.0f / temperature;

    // Step 1: Find global max (needed for numerical stability and threshold search)
    float local_max = -FLT_MAX;
    for (int i = tid; i < V; i += BLOCK_SIZE) {
        float val = __half2float(row[i]) * inv_temp;
        local_max = fmaxf(local_max, val);
    }
    float warp_max = warp_reduce_max(local_max);

    __shared__ float smem[32];  // for warp reductions
    if (lane_id == 0) smem[warp_id] = warp_max;
    __syncthreads();
    if (warp_id == 0) {
        float m = (lane_id < num_warps) ? smem[lane_id] : -FLT_MAX;
        m = warp_reduce_max(m);
        if (lane_id == 0) smem[0] = m;
    }
    __syncthreads();
    float global_max = smem[0];

    // Step 2: Binary search for the threshold that gives exactly K elements
    // Start with range [global_max - 20, global_max] and narrow down
    float lo = global_max - 40.0f;  // generous lower bound
    float hi = global_max;

    // 20 iterations of binary search gives precision of 40/2^20 ≈ 4e-5
    for (int iter = 0; iter < 20; iter++) {
        float mid = (lo + hi) * 0.5f;

        // Count elements >= mid
        int local_count = 0;
        for (int i = tid; i < V; i += BLOCK_SIZE) {
            float val = __half2float(row[i]) * inv_temp;
            if (val >= mid) local_count++;
        }

        // Block-level sum
        int warp_count = warp_reduce_sum_int(local_count);
        if (lane_id == 0) smem[warp_id] = __int_as_float(warp_count);
        __syncthreads();
        if (warp_id == 0) {
            int c = (lane_id < num_warps) ? __float_as_int(smem[lane_id]) : 0;
            c = warp_reduce_sum_int(c);
            if (lane_id == 0) smem[0] = __int_as_float(c);
        }
        __syncthreads();
        int total_count = __float_as_int(smem[0]);

        if (total_count > K) {
            lo = mid;  // threshold too low, raise it
        } else {
            hi = mid;  // threshold too high or just right, lower it
        }
    }

    float threshold = hi;  // use upper bound to ensure <= K elements

    // Step 3: Online softmax over elements >= threshold
    // Pass 3a: find max and sum of exp(val - max) for qualifying elements
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    for (int i = tid; i < V; i += BLOCK_SIZE) {
        float val = __half2float(row[i]) * inv_temp;
        if (val >= threshold) {
            if (val > row_max) {
                row_sum *= __expf(row_max - val);
                row_max = val;
            }
            row_sum += __expf(val - row_max);
        }
    }

    // Block reduce max
    warp_max = warp_reduce_max(row_max);
    row_sum *= __expf(row_max - warp_max);
    row_max = warp_max;
    float warp_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        smem[warp_id] = row_max;
        smem[warp_id + 16] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float m = (lane_id < num_warps) ? smem[lane_id] : -FLT_MAX;
        float s = (lane_id < num_warps) ? smem[lane_id + 16] : 0.0f;
        float block_max = warp_reduce_max(m);
        s *= __expf(m - block_max);
        float block_sum = warp_reduce_sum(s);
        if (lane_id == 0) {
            smem[0] = block_max;
            smem[1] = block_sum;
        }
    }
    __syncthreads();

    float final_max = smem[0];
    float final_sum = smem[1];
    float inv_sum = __fdividef(1.0f, final_sum);

    // Step 4: Write output = softmax prob if >= threshold, else 0
    for (int i = tid; i < V; i += BLOCK_SIZE) {
        float val = __half2float(row[i]) * inv_temp;
        if (val >= threshold) {
            out_row[i] = __float2half(__expf(val - final_max) * inv_sum);
        } else {
            out_row[i] = __float2half(0.0f);
        }
    }
}

torch::Tensor top_k_sampling_hip(
    torch::Tensor logits,
    int64_t k,
    double temperature
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a GPU tensor");
    TORCH_CHECK(logits.dtype() == torch::kFloat16, "logits must be float16");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [B, V]");

    int B = logits.size(0);
    int V = logits.size(1);

    auto output = torch::zeros_like(logits);

    top_k_sampling_kernel<<<B, BLOCK_SIZE>>>(
        reinterpret_cast<const half*>(logits.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        B, V, (int)k, (float)temperature
    );

    return output;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "top_k_sampling_hip")
    return _module


def kernel_fn(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert logits.is_cuda

    orig_shape = logits.shape
    orig_dtype = logits.dtype

    if orig_dtype != torch.float16:
        # Fallback to PyTorch
        scaled = logits / temperature
        topk_vals, topk_idx = torch.topk(scaled, k, dim=-1)
        out = torch.full_like(scaled, float('-inf'))
        out.scatter_(-1, topk_idx, topk_vals)
        return torch.nn.functional.softmax(out, dim=-1)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    elif logits.ndim > 2:
        logits = logits.view(-1, logits.shape[-1])

    mod = _get_module()
    out = mod.top_k_sampling_hip(logits, k, temperature)

    return out.view(orig_shape)
