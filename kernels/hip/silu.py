"""
AutoKernel -- HIP C++ SiLU (Swish) activation kernel.

SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
Purely element-wise, trivially memory-bound.

Features:
  - Vectorized half2 loads/stores for maximum memory bandwidth
  - fp32 intermediate computation for accuracy
  - __expf fast intrinsic for sigmoid
  - Grid-stride loop for arbitrary tensor sizes
"""

KERNEL_TYPE = "silu"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;

__global__ void __launch_bounds__(BLOCK_SIZE)
silu_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int N
) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int stride = gridDim.x * BLOCK_SIZE;

    // Vectorized half2 path
    const half2* in_v = reinterpret_cast<const half2*>(input);
    half2* out_v = reinterpret_cast<half2*>(output);
    const int n_pairs = N / 2;

    for (int i = tid; i < n_pairs; i += stride) {
        half2 v = in_v[i];
        float x0 = __half2float(v.x);
        float x1 = __half2float(v.y);

        // SiLU(x) = x / (1 + exp(-x))
        float y0 = x0 / (1.0f + __expf(-x0));
        float y1 = x1 / (1.0f + __expf(-x1));

        half2 result;
        result.x = __float2half(y0);
        result.y = __float2half(y1);
        out_v[i] = result;
    }

    // Odd tail
    if (tid == 0 && (N & 1)) {
        float x = __half2float(input[N - 1]);
        output[N - 1] = __float2half(x / (1.0f + __expf(-x)));
    }
}

torch::Tensor silu_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a GPU tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be float16");

    auto output = torch::empty_like(input);
    int N = input.numel();

    int n_pairs = N / 2;
    int blocks = min((n_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);

    silu_kernel<<<blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N
    );

    return output;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "silu_hip")
    return _module


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.silu_ref signature."""
    assert x.is_cuda

    orig_shape = x.shape
    orig_dtype = x.dtype

    # Non-FP16 path: fall back to PyTorch
    if orig_dtype != torch.float16:
        return torch.nn.functional.silu(x)

    x = x.contiguous()

    mod = _get_module()
    out = mod.silu_hip(x)

    return out.view(orig_shape)
