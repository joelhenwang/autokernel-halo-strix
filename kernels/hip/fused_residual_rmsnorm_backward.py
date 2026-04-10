"""
AutoKernel -- HIP C++ Fused Residual Add + RMSNorm Backward kernel.

Fuses the dual-output backward (grad from hidden path + grad from normed path)
with the RMSNorm backward into a single kernel:
  1. Recompute rms_inv from saved hidden (= x + residual)
  2. Compute grad through RMSNorm for the normed output path
  3. Merge: total_grad = grad_hidden + grad_from_norm
  4. grad_x = grad_residual = total_grad (same gradient to both inputs)
  5. Atomic accumulate grad_weight from normed path

Replaces 8 separate PyTorch ops with a single fused HIP kernel.

Input:  hidden      (M, N)  fp16  -- saved hidden = x + residual from forward
        weight      (N,)    fp16  -- RMSNorm weight
        grad_hidden (M, N)  fp16  -- gradient from hidden output
        grad_normed (M, N)  fp16  -- gradient from normed output

Output: grad_x      (M, N)  fp16  -- gradient w.r.t. x input
        grad_residual (M, N) fp16 -- gradient w.r.t. residual (same tensor as grad_x)
        grad_weight (N,)    fp32  -- gradient w.r.t. weight
"""

KERNEL_TYPE = "fused_residual_rmsnorm_backward"
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

__global__ void fused_residual_rmsnorm_backward_kernel(
    const half* __restrict__ HIDDEN,      // (M, N) saved hidden from forward
    const half* __restrict__ W,           // (N,) RMSNorm weight
    const half* __restrict__ GRAD_H,      // (M, N) grad from hidden output
    const half* __restrict__ GRAD_N,      // (M, N) grad from normed output
    half* __restrict__ GRAD_X,            // (M, N) grad for x input
    float* __restrict__ GRAD_W,           // (N,) weight gradient (fp32, atomic)
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockSize / WARP_SIZE;

    const half* h_row = HIDDEN + row * N;
    const half* gh_row = GRAD_H + row * N;
    const half* gn_row = GRAD_N + row * N;
    half* gx_row = GRAD_X + row * N;

    // Dynamic shared memory: [0..N) = hidden cache, [N..N+32) = reduction workspace
    extern __shared__ float smem[];
    float* h_cache = smem;
    float* shared_sums = smem + N;

    // ---- Phase 1: Load hidden, compute rms_inv ----
    float local_sum_sq = 0.0f;

    int idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 hval = *reinterpret_cast<const half2*>(h_row + idx);
        float h0 = __half2float(hval.x);
        float h1 = __half2float(hval.y);
        h_cache[idx] = h0;
        h_cache[idx + 1] = h1;
        local_sum_sq += h0 * h0 + h1 * h1;
    }
    if (idx < N) {
        float hv = __half2float(h_row[idx]);
        h_cache[idx] = hv;
        local_sum_sq += hv * hv;
    }

    float total_sum_sq = block_reduce_sum(local_sum_sq, tid, warp_id, lane_id, num_warps, shared_sums);

    __shared__ float s_rms_inv;
    if (tid == 0) {
        float mean_sq = total_sum_sq / static_cast<float>(N);
        s_rms_inv = rsqrtf(mean_sq + EPS);
    }
    __syncthreads();
    float rms_inv = s_rms_inv;

    // ---- Phase 2: Compute inner_sum for RMSNorm backward ----
    // inner_sum = sum(grad_normed * weight * normed_h) where normed_h = hidden * rms_inv
    float local_inner = 0.0f;

    idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 gnval = *reinterpret_cast<const half2*>(gn_row + idx);
        half2 wval = *reinterpret_cast<const half2*>(W + idx);

        float gn0 = __half2float(gnval.x);
        float gn1 = __half2float(gnval.y);
        float w0 = __half2float(wval.x);
        float w1 = __half2float(wval.y);
        float n0 = h_cache[idx] * rms_inv;
        float n1 = h_cache[idx + 1] * rms_inv;

        local_inner += gn0 * w0 * n0 + gn1 * w1 * n1;
    }
    if (idx < N) {
        float gnv = __half2float(gn_row[idx]);
        float wv = __half2float(W[idx]);
        float nv = h_cache[idx] * rms_inv;
        local_inner += gnv * wv * nv;
    }

    __syncthreads();
    float inner_sum = block_reduce_sum(local_inner, tid, warp_id, lane_id, num_warps, shared_sums);

    __shared__ float s_inner_over_D;
    if (tid == 0) {
        s_inner_over_D = inner_sum / static_cast<float>(N);
    }
    __syncthreads();
    float inner_over_D = s_inner_over_D;

    // ---- Phase 3: Compute total grad and write outputs ----
    idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 ghval = *reinterpret_cast<const half2*>(gh_row + idx);
        half2 gnval = *reinterpret_cast<const half2*>(gn_row + idx);
        half2 wval = *reinterpret_cast<const half2*>(W + idx);

        float gh0 = __half2float(ghval.x);
        float gh1 = __half2float(ghval.y);
        float gn0 = __half2float(gnval.x);
        float gn1 = __half2float(gnval.y);
        float w0 = __half2float(wval.x);
        float w1 = __half2float(wval.y);
        float n0 = h_cache[idx] * rms_inv;
        float n1 = h_cache[idx + 1] * rms_inv;

        // RMSNorm backward: grad_h_from_norm
        float ghn0 = rms_inv * (gn0 * w0 - n0 * inner_over_D);
        float ghn1 = rms_inv * (gn1 * w1 - n1 * inner_over_D);

        // Total grad = grad_hidden + grad_h_from_norm
        float total0 = gh0 + ghn0;
        float total1 = gh1 + ghn1;

        half2 result;
        result.x = __float2half(total0);
        result.y = __float2half(total1);
        *reinterpret_cast<half2*>(gx_row + idx) = result;

        // Accumulate grad_weight
        atomicAdd(&GRAD_W[idx], gn0 * n0);
        atomicAdd(&GRAD_W[idx + 1], gn1 * n1);
    }
    if (idx < N) {
        float ghv = __half2float(gh_row[idx]);
        float gnv = __half2float(gn_row[idx]);
        float wv = __half2float(W[idx]);
        float nv = h_cache[idx] * rms_inv;

        float ghnv = rms_inv * (gnv * wv - nv * inner_over_D);
        float totalv = ghv + ghnv;

        gx_row[idx] = __float2half(totalv);
        atomicAdd(&GRAD_W[idx], gnv * nv);
    }
}

std::vector<torch::Tensor> fused_residual_rmsnorm_backward_hip(
    torch::Tensor hidden, torch::Tensor weight,
    torch::Tensor grad_hidden, torch::Tensor grad_normed
) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be a GPU tensor");
    TORCH_CHECK(hidden.dtype() == torch::kFloat16, "hidden must be float16");

    int M = hidden.size(0);
    int N = hidden.size(1);

    auto grad_x = torch::empty_like(hidden);
    auto grad_weight = torch::zeros({N}, torch::dtype(torch::kFloat32).device(hidden.device()));

    int threads = min(1024, max(32, ((N + 1) / 2 + 31) / 32 * 32));
    int smem_bytes = (N + 32) * sizeof(float);

    dim3 grid(M);
    dim3 block(threads);

    fused_residual_rmsnorm_backward_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(hidden.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(grad_hidden.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(grad_normed.data_ptr<at::Half>()),
        reinterpret_cast<half*>(grad_x.data_ptr<at::Half>()),
        grad_weight.data_ptr<float>(),
        M, N
    );

    // grad_x and grad_residual are the same tensor
    return {grad_x, grad_x, grad_weight};
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        cpp_src = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> fused_residual_rmsnorm_backward_hip(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
"""
        _module = compile_hip(HIP_SRC, "fused_residual_rmsnorm_backward_hip", cpp_src=cpp_src)
    return _module


def kernel_fn(
    hidden: torch.Tensor, weight: torch.Tensor,
    grad_hidden: torch.Tensor, grad_normed: torch.Tensor,
) -> tuple:
    """Fused residual+RMSNorm backward.

    Args:
        hidden: (M, N) or (..., N) saved hidden = x + residual, fp16
        weight: (N,) RMSNorm weight, fp16
        grad_hidden: gradient from hidden output path, fp16
        grad_normed: gradient from normed output path, fp16

    Returns:
        (grad_x, grad_residual, grad_weight)
    """
    assert hidden.is_cuda

    if hidden.dtype != torch.float16:
        h_f = hidden.float()
        w_f = weight.float()
        gn_f = grad_normed.float()

        rms_sq = h_f.pow(2).mean(-1, keepdim=True) + 1e-6
        rms_inv = rms_sq.rsqrt()
        normed_h = h_f * rms_inv
        D = h_f.shape[-1]

        grad_weight = (gn_f * normed_h).sum(dim=tuple(range(gn_f.ndim - 1)))
        grad_normed_scaled = gn_f * w_f
        grad_h_from_norm = grad_normed_scaled * rms_inv - normed_h * (grad_normed_scaled * normed_h).sum(-1, keepdim=True) / D

        total_grad_h = grad_hidden.float() + grad_h_from_norm
        grad_x = total_grad_h.to(hidden.dtype)
        return grad_x, grad_x, grad_weight.to(weight.dtype)

    orig_shape = hidden.shape
    h_2d = hidden.contiguous().view(-1, hidden.shape[-1])
    w = weight.contiguous()
    gh_2d = grad_hidden.contiguous().view(-1, grad_hidden.shape[-1])
    gn_2d = grad_normed.contiguous().view(-1, grad_normed.shape[-1])

    if w.dtype != torch.float16:
        w = w.to(torch.float16)

    mod = _get_module()
    results = mod.fused_residual_rmsnorm_backward_hip(h_2d, w, gh_2d, gn_2d)

    grad_x = results[0].view(orig_shape)
    grad_weight = results[2]  # fp32

    return grad_x, grad_x, grad_weight
