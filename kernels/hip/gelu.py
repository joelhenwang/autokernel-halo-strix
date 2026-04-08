"""
AutoKernel -- HIP C++ GELU activation kernel.

GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
Purely element-wise, memory-bound.

Features:
  - Vectorized half2 loads/stores for maximum memory bandwidth
  - fp32 intermediate computation with erff() intrinsic
  - Grid-stride loop for arbitrary tensor sizes
"""

KERNEL_TYPE = "gelu"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

constexpr int BLOCK_SIZE = 256;
constexpr float SQRT_2_INV = 0.7071067811865475f;  // 1/sqrt(2)

__device__ __forceinline__ float gelu_f32(float x) {
    return x * 0.5f * (1.0f + erff(x * SQRT_2_INV));
}

__global__ void __launch_bounds__(BLOCK_SIZE)
gelu_kernel(
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

        half2 result;
        result.x = __float2half(gelu_f32(x0));
        result.y = __float2half(gelu_f32(x1));
        out_v[i] = result;
    }

    // Odd tail
    if (tid == 0 && (N & 1)) {
        float x = __half2float(input[N - 1]);
        output[N - 1] = __float2half(gelu_f32(x));
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a GPU tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be float16");

    auto output = torch::empty_like(input);
    int N = input.numel();

    int n_pairs = N / 2;
    int blocks = min((n_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);

    gelu_kernel<<<blocks, BLOCK_SIZE>>>(
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
        _module = compile_hip(HIP_SRC, "gelu_hip")
    return _module


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.gelu_ref signature."""
    assert x.is_cuda

    orig_shape = x.shape
    orig_dtype = x.dtype

    # Non-FP16 path: fall back to PyTorch
    if orig_dtype != torch.float16:
        return torch.nn.functional.gelu(x)

    x = x.contiguous()

    mod = _get_module()
    out = mod.gelu_hip(x)

    return out.view(orig_shape)
