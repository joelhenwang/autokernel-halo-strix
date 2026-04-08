"""
AutoKernel -- HIP C++ RMSNorm kernel.

Current kernel: Fused RMSNorm with wavefront-shuffle reduction and vectorized loads.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Single-pass RMS computation: rms = sqrt(mean(x^2) + eps)
  - Wavefront shuffle cascade (__shfl_down) for fast intra-wavefront sum reduction
  - Block-level reduction via shared memory across wavefronts
  - rsqrtf() for fast inverse square root
  - Vectorized half2 loads for maximum memory bandwidth
  - Fused normalize + scale in one output pass
"""

KERNEL_TYPE = "rmsnorm"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr float EPS = 1e-6f;
constexpr int WARP_SIZE = 32;  // RDNA wave32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void rmsnorm_kernel(
    const half* __restrict__ X,
    const half* __restrict__ W,
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

    const half* x_row = X + row * N;
    half* out_row = OUT + row * N;

    // Phase 1: Compute sum of squares using vectorized loads
    float local_sum_sq = 0.0f;

    int idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 val = *reinterpret_cast<const half2*>(x_row + idx);
        float lo = __half2float(val.x);
        float hi = __half2float(val.y);
        local_sum_sq += lo * lo + hi * hi;
    }
    if (idx < N) {
        float val = __half2float(x_row[idx]);
        local_sum_sq += val * val;
    }

    // Wavefront-level reduction
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    // Block-level reduction via shared memory
    __shared__ float shared_sums[32];

    if (lane_id == 0) {
        shared_sums[warp_id] = local_sum_sq;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane_id < num_warps) ? shared_sums[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }

    __shared__ float s_rms_inv;
    if (tid == 0) {
        float mean_sq = block_sum / static_cast<float>(N);
        s_rms_inv = rsqrtf(mean_sq + EPS);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;

    // Phase 2: Fused normalize + scale, vectorized stores
    idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 xval = *reinterpret_cast<const half2*>(x_row + idx);
        half2 wval = *reinterpret_cast<const half2*>(W + idx);

        float x0 = __half2float(xval.x) * rms_inv * __half2float(wval.x);
        float x1 = __half2float(xval.y) * rms_inv * __half2float(wval.y);

        half2 result;
        result.x = __float2half(x0);
        result.y = __float2half(x1);
        *reinterpret_cast<half2*>(out_row + idx) = result;
    }
    if (idx < N) {
        float xv = __half2float(x_row[idx]);
        float wv = __half2float(W[idx]);
        out_row[idx] = __float2half(xv * rms_inv * wv);
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda(), "x must be a GPU tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a GPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "weight must be float16");

    int M = x.size(0);
    int N = x.size(1);

    auto out = torch::empty_like(x);

    int threads = min(1024, max(32, ((N + 1) / 2 + 31) / 32 * 32));

    dim3 grid(M);
    dim3 block(threads);

    rmsnorm_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
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
        _module = compile_hip(HIP_SRC, "rmsnorm_hip")
    return _module


def kernel_fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.rmsnorm_ref signature."""
    assert x.is_cuda and weight.is_cuda

    # FP32 path: fall back to PyTorch (our HIP kernel is FP16-only)
    if x.dtype == torch.float32:
        rms = (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        return x * rms * weight

    orig_dtype = x.dtype
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    if weight.dtype != torch.float16:
        weight = weight.to(torch.float16)

    mod = _get_module()
    out = mod.rmsnorm_hip(x, weight)

    if orig_dtype != torch.float16:
        out = out.to(orig_dtype)

    return out
