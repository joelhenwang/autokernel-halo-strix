"""AutoKernel -- Fused RoPE + Gate Multiply kernel for HyPEShortConvBlock.

Fuses the RoPE rotation on the gate signal with the element-wise multiply by h_tilde.
Eliminates the intermediate b_rope tensor (B x T x d_conv, ~2 MB per call).

Input:
    b          (M, D) — gate signal, fp16
    h_tilde    (M, D) — up signal, fp16
    freqs_cos  (T, R//2) — RoPE cos for each position pair, float
    freqs_sin  (T, R//2) — RoPE sin for each position pair, float
    T          int — seq_len
    D          int — d_conv
    H          int — rope_head_dim (must divide D evenly)

Output:
    y (M, D) — RoPE(b) * h_tilde, fp16

KERNEL_TYPE = "fused_rope_gate_mul"
BACKEND = "hip"
"""

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;

__global__ void __launch_bounds__(BLOCK_SIZE)
fused_rope_gate_mul_kernel(
    const half* __restrict__ b,
    const half* __restrict__ h_tilde,
    const float* __restrict__ freqs_cos,
    const float* __restrict__ freqs_sin,
    half* __restrict__ y,
    int B, int T, int D, int R_half
) {
    const int global_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int stride = gridDim.x * BLOCK_SIZE;
    const int total_elems = B * T * D;
    const int n_pairs = total_elems / 2;

    const half2* b_v = reinterpret_cast<const half2*>(b);
    const half2* h_v = reinterpret_cast<const half2*>(h_tilde);
    half2* y_v = reinterpret_cast<half2*>(y);

    // Each half2 covers 2 consecutive elements — naturally aligned with RoPE pairs
    for (int idx = global_idx; idx < n_pairs; idx += stride) {
        const int elem = idx * 2;
        const int d_idx = elem % D;
        const int t_idx = (elem / D) % T;

        // Pair index within head (cos/sin values repeat for each rope head)
        const int pair = (d_idx / 2) % R_half;

        float cos_val = freqs_cos[t_idx * R_half + pair];
        float sin_val = freqs_sin[t_idx * R_half + pair];

        half2 bv = b_v[idx];
        half2 hv = h_v[idx];

        float a = __half2float(bv.x);
        float bb = __half2float(bv.y);
        float h0 = __half2float(hv.x);
        float h1 = __half2float(hv.y);

        float rot_a = a * cos_val - bb * sin_val;
        float rot_b = a * sin_val + bb * cos_val;

        half2 result;
        result.x = __float2half(rot_a * h0);
        result.y = __float2half(rot_b * h1);
        y_v[idx] = result;
    }
}

torch::Tensor fused_rope_gate_mul(
    torch::Tensor b,
    torch::Tensor h_tilde,
    torch::Tensor freqs_cos,
    torch::Tensor freqs_sin,
    int64_t T,
    int64_t D,
    int64_t R_half
) {
    TORCH_CHECK(b.is_cuda(), "b must be on CUDA");
    TORCH_CHECK(h_tilde.is_cuda(), "h_tilde must be on CUDA");

    // Ensure freqs tensors are on the same device as b
    if (!freqs_cos.is_cuda()) {
        freqs_cos = freqs_cos.to(b.device());
    }
    if (!freqs_sin.is_cuda()) {
        freqs_sin = freqs_sin.to(b.device());
    }

    int B = b.size(0) / T;
    int total_elems = B * T * D;
    int n_pairs = total_elems / 2;

    auto y = torch::empty({B * T, D}, b.options());

    int grid_size = (n_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fused_rope_gate_mul_kernel<<<grid_size, BLOCK_SIZE>>>(
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(h_tilde.data_ptr<at::Half>()),
        freqs_cos.data_ptr<float>(),
        freqs_sin.data_ptr<float>(),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        B, T, D, R_half
    );

    return y;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "fused_rope_gate_mul")
    return _module


@torch.compiler.disable
def kernel_fn(
    b: torch.Tensor,
    h_tilde: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    T: int,
    D: int,
    R_half: int,
) -> torch.Tensor:
    mod = _get_module()
    return mod.fused_rope_gate_mul(b, h_tilde, freqs_cos, freqs_sin, T, D, R_half)