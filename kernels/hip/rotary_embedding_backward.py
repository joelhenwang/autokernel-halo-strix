"""
AutoKernel -- HIP C++ Fused Rotary Embedding Backward kernel.

Rotary embedding is an orthogonal transform. The backward is simply applying
the rotation with negated sin:
  grad_x[d]   = grad_out[d]   * cos + grad_out[d+1] * sin
  grad_x[d+1] = grad_out[d+1] * cos - grad_out[d]   * sin

Replaces 3 PyTorch ops (chunk + rotate_half via view+neg+cat + multiply)
with a single fused kernel using fp32 intermediate arithmetic.

Input:  grad_output (B, H, N, D)  fp16
        cos_cache   (N, D/2)      fp32
        sin_cache   (N, D/2)      fp32

Output: grad_x      (B, H, N, D)  fp16
"""

KERNEL_TYPE = "rotary_embedding_backward"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;

__global__ void rotary_embedding_backward_kernel(
    const half* __restrict__ GRAD_OUT,
    half* __restrict__ GRAD_X,
    const float* __restrict__ cos_full,  // [N, D] or broadcastable — full cos values
    const float* __restrict__ sin_full,  // [N, D] or broadcastable — full sin values
    int cos_stride_n,                    // stride for N dim in cos/sin
    int B, int H, int N, int D
) {
    const int total = B * H * N * (D / 2);
    const int half_D = D / 2;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        int d_pair = idx % half_D;
        int remainder = idx / half_D;
        int n = remainder % N;
        remainder = remainder / N;
        int h = remainder % H;
        int b = remainder / H;

        int base_idx = ((b * H + h) * N + n) * D + d_pair * 2;
        float g0 = __half2float(GRAD_OUT[base_idx]);
        float g1 = __half2float(GRAD_OUT[base_idx + 1]);

        // cos/sin indexed at the FIRST element of each pair
        int cs_idx = n * cos_stride_n + d_pair * 2;
        float c = cos_full[cs_idx];
        float s = sin_full[cs_idx];

        // Backward of rotary: apply with negated sin
        // Forward was: y0 = x0*c - x1*s, y1 = x0*s + x1*c
        // Backward:    gx0 = g0*c + g1*s, gx1 = -g0*s + g1*c
        GRAD_X[base_idx]     = __float2half(g0 * c + g1 * s);
        GRAD_X[base_idx + 1] = __float2half(g1 * c - g0 * s);
    }
}

torch::Tensor rotary_embedding_backward_hip(
    torch::Tensor grad_output, torch::Tensor cos_flat, torch::Tensor sin_flat,
    int64_t cos_stride_n
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be on GPU");
    TORCH_CHECK(grad_output.dim() == 4, "grad_output must be [B, H, N, D]");

    int B = grad_output.size(0);
    int H = grad_output.size(1);
    int N = grad_output.size(2);
    int D = grad_output.size(3);
    TORCH_CHECK(D % 2 == 0, "D must be even");

    auto grad_x = torch::empty_like(grad_output);

    int total = B * H * N * (D / 2);
    int blocks = min((total + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);

    rotary_embedding_backward_kernel<<<blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const half*>(grad_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(grad_x.data_ptr<at::Half>()),
        cos_flat.data_ptr<float>(),
        sin_flat.data_ptr<float>(),
        (int)cos_stride_n,
        B, H, N, D
    );

    return grad_x;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        cpp_src = r"""
#include <torch/extension.h>
torch::Tensor rotary_embedding_backward_hip(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
"""
        _module = compile_hip(
            HIP_SRC, "rotary_embedding_backward_hip",
            cpp_src=cpp_src,
            extra_hip_cflags=["-fno-fast-math", "-ffp-contract=off"],
        )
    return _module


def kernel_fn(
    grad_output: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Fused rotary embedding backward.

    Args:
        grad_output: (B, H, N, D) upstream gradient, fp16
        cos: (..., N, D) broadcastable, fp32 — full D (not D/2)
        sin: (..., N, D) broadcastable, fp32 — full D (not D/2)

    Returns:
        grad_x: same shape/dtype as grad_output
    """
    assert grad_output.is_cuda

    if grad_output.dtype != torch.float16:
        g = grad_output.float()
        c = cos.float()
        s = sin.float()

        def rotate_half(t):
            t1, t2 = t.chunk(2, dim=-1)
            return torch.cat((-t2, t1), dim=-1)

        grad_x = g * c + rotate_half(g) * (-s)
        return grad_x.to(grad_output.dtype)

    N = grad_output.shape[2]
    D = grad_output.shape[3]

    # Flatten cos/sin to (N, D) contiguous — squeeze batch dims
    cos_2d = cos.float().reshape(-1, cos.shape[-1])
    sin_2d = sin.float().reshape(-1, sin.shape[-1])
    # Take last N rows (in case of batch dims) and ensure D matches
    if cos_2d.shape[0] > N:
        cos_2d = cos_2d[:N]
        sin_2d = sin_2d[:N]
    cos_2d = cos_2d.contiguous()
    sin_2d = sin_2d.contiguous()
    cos_stride_n = cos_2d.stride(0)

    go = grad_output.contiguous()

    mod = _get_module()
    return mod.rotary_embedding_backward_hip(go, cos_2d, sin_2d, cos_stride_n)
