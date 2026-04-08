"""
AutoKernel -- HIP C++ Fused Residual Add + RMSNorm kernel.

Current kernel: Fuses x + residual with RMSNorm into a single kernel.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Single kernel for residual add + RMSNorm (eliminates intermediate tensor)
  - LDS caching of hidden = x + residual to avoid re-reading from global memory
  - Wavefront shuffle (__shfl_xor) reductions for fast intra-wavefront sum
  - Block-level reduction via shared memory across wavefronts
  - rsqrtf() for fast inverse square root
  - Vectorized half2 loads for maximum memory bandwidth
  - Fused normalize + scale in one output pass from LDS
"""

KERNEL_TYPE = "fused_residual_add_rmsnorm"
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
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Fused residual add + RMSNorm kernel
// hidden = x + residual, then output = hidden / rms(hidden) * weight
// Uses dynamic shared memory to cache hidden row, avoiding a second global read.
__global__ void __launch_bounds__(1024)
fused_residual_add_rmsnorm_kernel(
    const half* __restrict__ X,
    const half* __restrict__ R,    // residual
    const half* __restrict__ W,    // weight
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

    // Dynamic shared memory for caching hidden = x + residual as float
    extern __shared__ float s_hidden[];

    // Phase 1: Compute hidden = x + residual in fp16 (match PyTorch rounding),
    // cache in LDS as float, accumulate sum_sq
    float local_sum_sq = 0.0f;

    int idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 xval = *reinterpret_cast<const half2*>(x_row + idx);
        half2 rval = *reinterpret_cast<const half2*>(r_row + idx);
        // Add in fp16 to match PyTorch's per-op rounding, then promote to fp32
        half2 hval = __hadd2(xval, rval);
        float h0 = __half2float(hval.x);
        float h1 = __half2float(hval.y);
        s_hidden[idx]     = h0;
        s_hidden[idx + 1] = h1;
        local_sum_sq += h0 * h0 + h1 * h1;
    }
    if (idx < N) {
        half h_half = __hadd(x_row[idx], r_row[idx]);
        float h = __half2float(h_half);
        s_hidden[idx] = h;
        local_sum_sq += h * h;
    }

    // Wavefront-level reduction
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    // Block-level reduction via shared memory
    // Use a small region at the end of s_hidden for warp sums
    // We need num_warps floats; s_hidden has N floats, and N >= blockSize typically
    __shared__ float shared_sums[32];

    if (lane_id == 0) {
        shared_sums[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        // lane 0 has the final sum, but shfl_xor gives all lanes the result
        if (lane_id == 0) {
            // Store rms_inv for all threads
            float mean_sq = val / static_cast<float>(N);
            shared_sums[0] = rsqrtf(mean_sq + EPS);
        }
    }
    __syncthreads();

    float rms_inv = shared_sums[0];

    // Phase 2: Normalize from LDS cache + scale by weight, vectorized writes
    idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 wval = *reinterpret_cast<const half2*>(W + idx);
        float h0 = s_hidden[idx]     * rms_inv * __half2float(wval.x);
        float h1 = s_hidden[idx + 1] * rms_inv * __half2float(wval.y);

        half2 result;
        result.x = __float2half(h0);
        result.y = __float2half(h1);
        *reinterpret_cast<half2*>(out_row + idx) = result;
    }
    if (idx < N) {
        float wv = __half2float(W[idx]);
        out_row[idx] = __float2half(s_hidden[idx] * rms_inv * wv);
    }
}

torch::Tensor fused_residual_add_rmsnorm_hip(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight
) {
    TORCH_CHECK(x.is_cuda(), "x must be a GPU tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a GPU tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a GPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(residual.dtype() == torch::kFloat16, "residual must be float16");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "weight must be float16");

    int M = x.size(0);
    int N = x.size(1);

    auto out = torch::empty_like(x);

    int threads = min(1024, max(32, ((N + 1) / 2 + 31) / 32 * 32));

    dim3 grid(M);
    dim3 block(threads);

    // Dynamic shared memory: N floats for caching hidden row
    size_t smem_bytes = N * sizeof(float);

    fused_residual_add_rmsnorm_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N
    );

    return out;
}
"""

HIP_SRC_DUAL = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr float EPS_DUAL = 1e-6f;
constexpr int WARP_SIZE_DUAL = 32;  // RDNA wave32

__device__ __forceinline__ float warp_reduce_sum_dual(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE_DUAL / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Dual-output fused residual add + RMSNorm kernel.
// Returns BOTH hidden = x + residual AND normalized = rmsnorm(hidden) * weight.
__global__ void __launch_bounds__(1024)
fused_residual_add_rmsnorm_dual_kernel(
    const half* __restrict__ X,
    const half* __restrict__ R,       // residual
    const half* __restrict__ W,       // weight
    half* __restrict__ HIDDEN_OUT,    // output: x + residual
    half* __restrict__ NORM_OUT,      // output: rmsnorm(x + residual) * weight
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid / WARP_SIZE_DUAL;
    const int lane_id = tid % WARP_SIZE_DUAL;
    const int num_warps = blockSize / WARP_SIZE_DUAL;

    const half* x_row = X + (long long)row * N;
    const half* r_row = R + (long long)row * N;
    half* hidden_row = HIDDEN_OUT + (long long)row * N;
    half* norm_row = NORM_OUT + (long long)row * N;

    // Dynamic shared memory for caching hidden = x + residual as float
    extern __shared__ float s_hidden[];

    // Phase 1: Compute hidden = x + residual in fp16, write to HIDDEN_OUT,
    // cache in LDS as float, accumulate sum_sq
    float local_sum_sq = 0.0f;

    int idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 xval = *reinterpret_cast<const half2*>(x_row + idx);
        half2 rval = *reinterpret_cast<const half2*>(r_row + idx);
        half2 hval = __hadd2(xval, rval);

        // Write hidden to global memory
        *reinterpret_cast<half2*>(hidden_row + idx) = hval;

        float h0 = __half2float(hval.x);
        float h1 = __half2float(hval.y);
        s_hidden[idx]     = h0;
        s_hidden[idx + 1] = h1;
        local_sum_sq += h0 * h0 + h1 * h1;
    }
    if (idx < N) {
        half h_half = __hadd(x_row[idx], r_row[idx]);
        hidden_row[idx] = h_half;
        float h = __half2float(h_half);
        s_hidden[idx] = h;
        local_sum_sq += h * h;
    }

    // Wavefront-level reduction
    local_sum_sq = warp_reduce_sum_dual(local_sum_sq);

    // Block-level reduction via shared memory
    __shared__ float shared_sums[32];

    if (lane_id == 0) {
        shared_sums[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_sums[lane_id] : 0.0f;
        val = warp_reduce_sum_dual(val);
        if (lane_id == 0) {
            float mean_sq = val / static_cast<float>(N);
            shared_sums[0] = rsqrtf(mean_sq + EPS_DUAL);
        }
    }
    __syncthreads();

    float rms_inv = shared_sums[0];

    // Phase 2: Normalize from LDS cache + scale by weight, write to NORM_OUT
    idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 wval = *reinterpret_cast<const half2*>(W + idx);
        float h0 = s_hidden[idx]     * rms_inv * __half2float(wval.x);
        float h1 = s_hidden[idx + 1] * rms_inv * __half2float(wval.y);

        half2 result;
        result.x = __float2half(h0);
        result.y = __float2half(h1);
        *reinterpret_cast<half2*>(norm_row + idx) = result;
    }
    if (idx < N) {
        float wv = __half2float(W[idx]);
        norm_row[idx] = __float2half(s_hidden[idx] * rms_inv * wv);
    }
}

std::vector<torch::Tensor> fused_residual_add_rmsnorm_dual_hip(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight
) {
    TORCH_CHECK(x.is_cuda(), "x must be a GPU tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a GPU tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a GPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(residual.dtype() == torch::kFloat16, "residual must be float16");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "weight must be float16");

    int M = x.size(0);
    int N = x.size(1);

    auto hidden = torch::empty_like(x);
    auto normalized = torch::empty_like(x);

    int threads = min(1024, max(32, ((N + 1) / 2 + 31) / 32 * 32));

    dim3 grid(M);
    dim3 block(threads);

    size_t smem_bytes = N * sizeof(float);

    fused_residual_add_rmsnorm_dual_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
        reinterpret_cast<half*>(normalized.data_ptr<at::Half>()),
        M, N
    );

    return {hidden, normalized};
}
"""

_module = None
_module_dual = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "fused_residual_add_rmsnorm_hip")
    return _module


def _get_module_dual():
    global _module_dual
    if _module_dual is None:
        _module_dual = compile_hip(HIP_SRC_DUAL, "fused_residual_add_rmsnorm_dual_hip")
    return _module_dual


def kernel_fn(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.fused_residual_add_rmsnorm_ref signature."""
    assert x.is_cuda and residual.is_cuda and weight.is_cuda

    orig_shape = x.shape

    # Non-FP16 path: fall back to PyTorch
    if x.dtype != torch.float16:
        hidden = x + residual
        rms = (hidden.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        return hidden * rms * weight

    orig_dtype = x.dtype

    if x.ndim == 1:
        x = x.unsqueeze(0)
        residual = residual.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])
        residual = residual.view(-1, residual.shape[-1])

    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    if residual.dtype != torch.float16:
        residual = residual.to(torch.float16)
    if weight.dtype != torch.float16:
        weight = weight.to(torch.float16)

    mod = _get_module()
    out = mod.fused_residual_add_rmsnorm_hip(x, residual, weight)

    if orig_dtype != torch.float16:
        out = out.to(orig_dtype)

    return out.view(orig_shape)


def kernel_fn_dual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple:
    """Dual-output variant for verify.py. Returns (hidden, normalized)."""
    assert x.is_cuda and residual.is_cuda and weight.is_cuda

    orig_shape = x.shape

    # Non-FP16 path: fall back to PyTorch
    if x.dtype != torch.float16:
        hidden = x + residual
        hidden_f = hidden.float()
        rms = (hidden_f.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        normalized = (hidden_f * rms * weight.float()).to(x.dtype)
        return hidden, normalized

    orig_dtype = x.dtype

    if x.ndim == 1:
        x = x.unsqueeze(0)
        residual = residual.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])
        residual = residual.view(-1, residual.shape[-1])

    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    if residual.dtype != torch.float16:
        residual = residual.to(torch.float16)
    if weight.dtype != torch.float16:
        weight = weight.to(torch.float16)

    mod = _get_module_dual()
    results = mod.fused_residual_add_rmsnorm_dual_hip(x, residual, weight)
    hidden = results[0]
    normalized = results[1]

    if orig_dtype != torch.float16:
        hidden = hidden.to(orig_dtype)
        normalized = normalized.to(orig_dtype)

    return hidden.view(orig_shape), normalized.view(orig_shape)
