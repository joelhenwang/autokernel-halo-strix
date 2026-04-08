"""
AutoKernel -- HIP C++ Fused Bias Add + SiLU Activation kernel.

Computes: output = SiLU(x + bias) = (x + bias) / (1 + exp(-(x + bias)))
Common pattern after linear projections with bias.

Features:
  - Fuses bias addition with SiLU activation (eliminates intermediate tensor)
  - Per-column bias broadcast
  - Vectorized half2 loads/stores
  - fp32 intermediate computation
  - Grid-stride loop for arbitrary tensor sizes
"""

KERNEL_TYPE = "fused_bias_silu"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;

__global__ void __launch_bounds__(BLOCK_SIZE)
fused_bias_silu_kernel(
    const half* __restrict__ input,   // [M, N]
    const half* __restrict__ bias,    // [N]
    half* __restrict__ output,        // [M, N]
    int M, int N
) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int stride = gridDim.x * BLOCK_SIZE;
    const int total_pairs = (M * N) / 2;

    const half2* in_v = reinterpret_cast<const half2*>(input);
    half2* out_v = reinterpret_cast<half2*>(output);

    for (int i = tid; i < total_pairs; i += stride) {
        // Compute column indices for bias lookup
        int flat_idx = i * 2;
        int col = flat_idx % N;

        half2 x = in_v[i];
        half2 b = *reinterpret_cast<const half2*>(bias + col);

        float x0 = __half2float(x.x) + __half2float(b.x);
        float x1 = __half2float(x.y) + __half2float(b.y);

        // SiLU(x) = x / (1 + exp(-x))
        float y0 = x0 / (1.0f + __expf(-x0));
        float y1 = x1 / (1.0f + __expf(-x1));

        half2 result;
        result.x = __float2half(y0);
        result.y = __float2half(y1);
        out_v[i] = result;
    }

    // Odd tail
    int total = M * N;
    if (tid == 0 && (total & 1)) {
        int idx = total - 1;
        int col = idx % N;
        float x = __half2float(input[idx]) + __half2float(bias[col]);
        output[idx] = __float2half(x / (1.0f + __expf(-x)));
    }
}

torch::Tensor fused_bias_silu_hip(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a GPU tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a GPU tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(bias.dtype() == torch::kFloat16, "bias must be float16");

    int M = input.size(0);
    int N = input.size(1);

    auto output = torch::empty_like(input);

    int total_pairs = (M * N) / 2;
    int blocks = min((total_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);

    fused_bias_silu_kernel<<<blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N
    );

    return output;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "fused_bias_silu_hip")
    return _module


def kernel_fn(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert x.is_cuda and bias.is_cuda

    orig_shape = x.shape
    orig_dtype = x.dtype

    if orig_dtype != torch.float16:
        return torch.nn.functional.silu(x + bias)

    x = x.contiguous()

    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    if bias.dtype != torch.float16:
        bias = bias.to(torch.float16)

    mod = _get_module()
    out = mod.fused_bias_silu_hip(x, bias)

    return out.view(orig_shape)
