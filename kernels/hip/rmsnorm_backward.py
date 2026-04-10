"""
AutoKernel -- HIP C++ Fused RMSNorm Backward kernel.

Fuses the entire RMSNorm backward pass into a single kernel:
  1. Recompute rms_inv from saved input x
  2. Compute normed = x * rms_inv
  3. Warp+block reduction for inner_sum = sum(grad_output * weight * normed)
  4. grad_x = rms_inv * (grad_output * weight - normed * inner_sum / D)
  5. Atomic accumulate grad_weight += grad_output * normed

Replaces 5 separate PyTorch ops (fp32 cast, recompute, reduction, chain rule,
weight grad) with a single fused HIP kernel, eliminating 4 intermediate tensors.

Input:  x           (M, N)  fp16  -- original input (saved from forward)
        weight      (N,)    fp16  -- RMSNorm weight
        grad_output (M, N)  fp16  -- gradient from upstream

Output: grad_x      (M, N)  fp16  -- gradient w.r.t. input
        grad_weight (N,)    fp32  -- gradient w.r.t. weight (accumulated across rows)
"""

KERNEL_TYPE = "rmsnorm_backward"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr float EPS = 1e-6f;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level sum reduction: warp reduce -> shared memory -> warp 0 reduce
__device__ __forceinline__ float block_reduce_sum(
    float val, int tid, int warp_id, int lane_id, int num_warps,
    float* shared_sums
) {
    val = warp_reduce_sum(val);
    if (lane_id == 0) shared_sums[warp_id] = val;
    __syncthreads();
    float result = 0.0f;
    if (warp_id == 0) {
        result = (lane_id < num_warps) ? shared_sums[lane_id] : 0.0f;
        result = warp_reduce_sum(result);
    }
    return result;
}

__global__ void rmsnorm_backward_kernel(
    const half* __restrict__ X,          // (M, N) input
    const half* __restrict__ W,          // (N,) weight
    const half* __restrict__ GRAD_OUT,   // (M, N) upstream gradient
    half* __restrict__ GRAD_X,           // (M, N) output gradient
    float* __restrict__ GRAD_W,          // (N,) weight gradient (fp32, atomically accumulated)
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
    const half* g_row = GRAD_OUT + row * N;
    half* gx_row = GRAD_X + row * N;

    // Use dynamic shared memory:
    // [0, N) floats: cached x values
    // [N, N+32) floats: shared reduction workspace
    extern __shared__ float smem[];
    float* x_cache = smem;
    float* shared_sums = smem + N;

    // ---- Phase 1: Load x, compute sum_sq for rms_inv ----
    float local_sum_sq = 0.0f;

    int idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 xval = *reinterpret_cast<const half2*>(x_row + idx);
        float x0 = __half2float(xval.x);
        float x1 = __half2float(xval.y);
        x_cache[idx] = x0;
        x_cache[idx + 1] = x1;
        local_sum_sq += x0 * x0 + x1 * x1;
    }
    if (idx < N) {
        float xv = __half2float(x_row[idx]);
        x_cache[idx] = xv;
        local_sum_sq += xv * xv;
    }

    // Block reduction for sum_sq
    float total_sum_sq = block_reduce_sum(local_sum_sq, tid, warp_id, lane_id, num_warps, shared_sums);

    __shared__ float s_rms_inv;
    if (tid == 0) {
        float mean_sq = total_sum_sq / static_cast<float>(N);
        s_rms_inv = rsqrtf(mean_sq + EPS);
    }
    __syncthreads();
    float rms_inv = s_rms_inv;

    // ---- Phase 2: Compute inner_sum = sum(grad_out * weight * normed) ----
    // normed = x * rms_inv
    float local_inner = 0.0f;

    idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 gval = *reinterpret_cast<const half2*>(g_row + idx);
        half2 wval = *reinterpret_cast<const half2*>(W + idx);

        float g0 = __half2float(gval.x);
        float g1 = __half2float(gval.y);
        float w0 = __half2float(wval.x);
        float w1 = __half2float(wval.y);
        float n0 = x_cache[idx] * rms_inv;
        float n1 = x_cache[idx + 1] * rms_inv;

        local_inner += g0 * w0 * n0 + g1 * w1 * n1;
    }
    if (idx < N) {
        float gv = __half2float(g_row[idx]);
        float wv = __half2float(W[idx]);
        float nv = x_cache[idx] * rms_inv;
        local_inner += gv * wv * nv;
    }

    // Block reduction for inner_sum
    __syncthreads();  // reuse shared_sums
    float inner_sum = block_reduce_sum(local_inner, tid, warp_id, lane_id, num_warps, shared_sums);

    __shared__ float s_inner_sum;
    if (tid == 0) {
        s_inner_sum = inner_sum / static_cast<float>(N);
    }
    __syncthreads();
    float inner_over_D = s_inner_sum;

    // ---- Phase 3: Compute grad_x and accumulate grad_weight ----
    idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 gval = *reinterpret_cast<const half2*>(g_row + idx);
        half2 wval = *reinterpret_cast<const half2*>(W + idx);

        float g0 = __half2float(gval.x);
        float g1 = __half2float(gval.y);
        float w0 = __half2float(wval.x);
        float w1 = __half2float(wval.y);
        float n0 = x_cache[idx] * rms_inv;
        float n1 = x_cache[idx + 1] * rms_inv;

        // grad_x = rms_inv * (grad_out * weight - normed * inner_sum / D)
        float gx0 = rms_inv * (g0 * w0 - n0 * inner_over_D);
        float gx1 = rms_inv * (g1 * w1 - n1 * inner_over_D);

        half2 result;
        result.x = __float2half(gx0);
        result.y = __float2half(gx1);
        *reinterpret_cast<half2*>(gx_row + idx) = result;

        // Atomic accumulate grad_weight (fp32)
        atomicAdd(&GRAD_W[idx], g0 * n0);
        atomicAdd(&GRAD_W[idx + 1], g1 * n1);
    }
    if (idx < N) {
        float gv = __half2float(g_row[idx]);
        float wv = __half2float(W[idx]);
        float nv = x_cache[idx] * rms_inv;

        float gxv = rms_inv * (gv * wv - nv * inner_over_D);
        gx_row[idx] = __float2half(gxv);

        atomicAdd(&GRAD_W[idx], gv * nv);
    }
}

std::vector<torch::Tensor> rmsnorm_backward_hip(
    torch::Tensor x, torch::Tensor weight, torch::Tensor grad_output
) {
    TORCH_CHECK(x.is_cuda(), "x must be a GPU tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a GPU tensor");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a GPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");

    int M = x.size(0);
    int N = x.size(1);

    auto grad_x = torch::empty_like(x);
    auto grad_weight = torch::zeros({N}, torch::dtype(torch::kFloat32).device(x.device()));

    int threads = min(1024, max(32, ((N + 1) / 2 + 31) / 32 * 32));
    int smem_bytes = (N + 32) * sizeof(float);  // x_cache + shared_sums

    dim3 grid(M);
    dim3 block(threads);

    rmsnorm_backward_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(grad_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(grad_x.data_ptr<at::Half>()),
        grad_weight.data_ptr<float>(),
        M, N
    );

    return {grad_x, grad_weight};
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        cpp_src = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> rmsnorm_backward_hip(
    torch::Tensor, torch::Tensor, torch::Tensor);
"""
        _module = compile_hip(HIP_SRC, "rmsnorm_backward_hip", cpp_src=cpp_src)
    return _module


def kernel_fn(
    x: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor
) -> tuple:
    """Fused RMSNorm backward.

    Args:
        x: (M, N) or (..., N) original input, fp16
        weight: (N,) RMSNorm weight, fp16
        grad_output: (M, N) or (..., N) upstream gradient, fp16

    Returns:
        (grad_x, grad_weight): grad_x same shape/dtype as x, grad_weight fp32
    """
    assert x.is_cuda and weight.is_cuda and grad_output.is_cuda

    # FP32 fallback — use PyTorch ops
    if x.dtype != torch.float16:
        x_f = x.float()
        w_f = weight.float()
        g_f = grad_output.float()
        rms_sq = x_f.pow(2).mean(-1, keepdim=True) + 1e-6
        rms_inv = rms_sq.rsqrt()
        normed = x_f * rms_inv
        D = x_f.shape[-1]
        grad_weight = (g_f * normed).sum(dim=tuple(range(g_f.ndim - 1)))
        grad_normed = g_f * w_f
        grad_x = grad_normed * rms_inv - normed * (grad_normed * normed).sum(-1, keepdim=True) / D
        return grad_x.to(x.dtype), grad_weight.to(weight.dtype)

    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, x.shape[-1])
    g_2d = grad_output.contiguous().view(-1, grad_output.shape[-1])
    w = weight.contiguous()

    if w.dtype != torch.float16:
        w = w.to(torch.float16)

    mod = _get_module()
    results = mod.rmsnorm_backward_hip(x_2d, w, g_2d)
    grad_x = results[0].view(orig_shape)
    grad_weight = results[1]  # fp32

    return grad_x, grad_weight
