"""
AutoKernel -- HIP C++ Fused Residual Add + LayerNorm kernel.

Fuses x + residual with LayerNorm into a single kernel.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Single kernel for residual add + LayerNorm (eliminates intermediate tensor)
  - LDS caching of hidden = x + residual to avoid re-reading from global memory
  - Wavefront shuffle (__shfl_xor) reductions for fast intra-wavefront sum
  - Block-level reduction via shared memory across wavefronts
  - rsqrtf() for fast inverse square root
  - Vectorized half2 loads for maximum memory bandwidth
  - Fused normalize + scale + bias in one output pass from LDS
"""

KERNEL_TYPE = "fused_residual_add_layernorm"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr float EPS = 1e-5f;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Fused residual add + LayerNorm kernel
// hidden = x + residual, then output = (hidden - mean) / sqrt(var + eps) * weight + bias
// Uses dynamic shared memory to cache hidden row as float.
__global__ void __launch_bounds__(1024)
fused_residual_add_layernorm_kernel(
    const half* __restrict__ X,
    const half* __restrict__ R,    // residual
    const half* __restrict__ W,    // weight (gamma)
    const half* __restrict__ B,    // bias (beta)
    half* __restrict__ OUT,
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockSize / WARP_SIZE;

    const half* x_row = X + (long long)row * N;
    const half* r_row = R + (long long)row * N;
    half* out_row = OUT + (long long)row * N;

    // Dynamic shared memory layout:
    // [N floats: cached hidden] [32 floats: warp sums] [32 floats: warp sum_sqs]
    extern __shared__ float s_data[];
    float* s_hidden = s_data;
    float* s_warp_sum = s_data + N;
    float* s_warp_sq = s_warp_sum + 32;

    // Phase 1: Compute hidden = x + residual in fp16 (match PyTorch),
    // cache in LDS as float, accumulate sum and sum_sq
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    int idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 xval = *reinterpret_cast<const half2*>(x_row + idx);
        half2 rval = *reinterpret_cast<const half2*>(r_row + idx);
        // Add in fp16 to match PyTorch's per-op rounding
        half2 hval = __hadd2(xval, rval);
        float h0 = __half2float(hval.x);
        float h1 = __half2float(hval.y);
        s_hidden[idx]     = h0;
        s_hidden[idx + 1] = h1;
        local_sum += h0 + h1;
        local_sum_sq += h0 * h0 + h1 * h1;
    }
    if (idx < N) {
        half h_half = __hadd(x_row[idx], r_row[idx]);
        float h = __half2float(h_half);
        s_hidden[idx] = h;
        local_sum += h;
        local_sum_sq += h * h;
    }

    // Wavefront-level reduction
    local_sum = warp_reduce_sum(local_sum);
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    // Block-level reduction via shared memory
    if (lane_id == 0) {
        s_warp_sum[warp_id] = local_sum;
        s_warp_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float s = (lane_id < num_warps) ? s_warp_sum[lane_id] : 0.0f;
        float sq = (lane_id < num_warps) ? s_warp_sq[lane_id] : 0.0f;
        s = warp_reduce_sum(s);
        sq = warp_reduce_sum(sq);
        if (lane_id == 0) {
            float mean = s / (float)N;
            float variance = sq / (float)N - mean * mean;
            s_warp_sum[0] = mean;
            s_warp_sum[1] = rsqrtf(variance + EPS);
        }
    }
    __syncthreads();

    float row_mean = s_warp_sum[0];
    float row_inv_std = s_warp_sum[1];

    // Phase 2: Normalize from LDS cache + scale by weight + add bias
    idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 wval = *reinterpret_cast<const half2*>(W + idx);
        half2 bval = *reinterpret_cast<const half2*>(B + idx);
        float n0 = (s_hidden[idx]     - row_mean) * row_inv_std;
        float n1 = (s_hidden[idx + 1] - row_mean) * row_inv_std;
        float o0 = n0 * __half2float(wval.x) + __half2float(bval.x);
        float o1 = n1 * __half2float(wval.y) + __half2float(bval.y);

        half2 result;
        result.x = __float2half(o0);
        result.y = __float2half(o1);
        *reinterpret_cast<half2*>(out_row + idx) = result;
    }
    if (idx < N) {
        float n0 = (s_hidden[idx] - row_mean) * row_inv_std;
        float wv = __half2float(W[idx]);
        float bv = __half2float(B[idx]);
        out_row[idx] = __float2half(n0 * wv + bv);
    }
}

torch::Tensor fused_residual_add_layernorm_hip(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a GPU tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a GPU tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a GPU tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a GPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");

    int M = x.size(0);
    int N = x.size(1);

    auto out = torch::empty_like(x);

    int threads = min(1024, max(32, ((N + 1) / 2 + 31) / 32 * 32));

    dim3 grid(M);
    dim3 block(threads);

    // Dynamic shared memory: N floats (hidden cache) + 64 floats (warp scratch)
    size_t smem_bytes = (N + 64) * sizeof(float);

    fused_residual_add_layernorm_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N
    );

    return out;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "fused_residual_add_layernorm_hip")
    return _module


def kernel_fn(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert x.is_cuda and residual.is_cuda and weight.is_cuda and bias.is_cuda

    orig_shape = x.shape

    # Non-FP16 path: fall back to PyTorch
    if x.dtype != torch.float16:
        hidden = x + residual
        return torch.nn.functional.layer_norm(hidden, [hidden.shape[-1]], weight, bias)

    if x.ndim == 1:
        x = x.unsqueeze(0)
        residual = residual.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])
        residual = residual.view(-1, residual.shape[-1])

    if residual.dtype != torch.float16:
        residual = residual.to(torch.float16)
    if weight.dtype != torch.float16:
        weight = weight.to(torch.float16)
    if bias.dtype != torch.float16:
        bias = bias.to(torch.float16)

    mod = _get_module()
    out = mod.fused_residual_add_layernorm_hip(x, residual, weight, bias)

    return out.view(orig_shape)
