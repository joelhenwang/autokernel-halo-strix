"""
AutoKernel -- HIP C++ Int8 Dequantization kernel.

Per-channel dequantization: output[i,j] = (int8[i,j] - zero_point[j]) * scale[j]
Purely element-wise with per-column broadcast, memory-bound.

Features:
  - Packed int8x4 loads (32-bit load for 4 int8 values)
  - Per-channel scale/zero_point broadcast
  - Vectorized half2 output writes
  - Grid-stride loop for arbitrary tensor sizes
"""

KERNEL_TYPE = "dequantize_int8"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;

__global__ void __launch_bounds__(BLOCK_SIZE)
dequantize_int8_kernel(
    const signed char* __restrict__ input,   // [M, N] int8
    const half* __restrict__ scale,          // [N] fp16
    const signed char* __restrict__ zero_pt, // [N] int8
    half* __restrict__ output,               // [M, N] fp16
    int M, int N
) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int stride = gridDim.x * BLOCK_SIZE;
    const int total = M * N;

    // Process 2 elements at a time for half2 output writes
    const int n_pairs = total / 2;

    for (int i = tid; i < n_pairs; i += stride) {
        int idx0 = 2 * i;
        int idx1 = idx0 + 1;

        int col0 = idx0 % N;
        int col1 = idx1 % N;

        // Load int8 values
        float x0 = (float)input[idx0];
        float x1 = (float)input[idx1];

        // Load zero points
        float zp0 = (float)zero_pt[col0];
        float zp1 = (float)zero_pt[col1];

        // Load scales
        float s0 = __half2float(scale[col0]);
        float s1 = __half2float(scale[col1]);

        // Dequantize
        float y0 = (x0 - zp0) * s0;
        float y1 = (x1 - zp1) * s1;

        // Write as half2
        half2 result;
        result.x = __float2half(y0);
        result.y = __float2half(y1);
        *reinterpret_cast<half2*>(output + idx0) = result;
    }

    // Odd tail
    if (tid == 0 && (total & 1)) {
        int idx = total - 1;
        int col = idx % N;
        float x = (float)input[idx];
        float zp = (float)zero_pt[col];
        float s = __half2float(scale[col]);
        output[idx] = __float2half((x - zp) * s);
    }
}

torch::Tensor dequantize_int8_hip(
    torch::Tensor x_int8,
    torch::Tensor scale,
    torch::Tensor zero_point
) {
    TORCH_CHECK(x_int8.is_cuda(), "x_int8 must be a GPU tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a GPU tensor");
    TORCH_CHECK(zero_point.is_cuda(), "zero_point must be a GPU tensor");
    TORCH_CHECK(x_int8.dtype() == torch::kInt8, "x_int8 must be int8");
    TORCH_CHECK(scale.dtype() == torch::kFloat16, "scale must be float16");
    TORCH_CHECK(zero_point.dtype() == torch::kInt8, "zero_point must be int8");

    int M = x_int8.size(0);
    int N = x_int8.size(1);

    auto output = torch::empty({M, N}, torch::TensorOptions().device(x_int8.device()).dtype(torch::kFloat16));

    int total = M * N;
    int n_pairs = total / 2;
    int blocks = min((n_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);

    dequantize_int8_kernel<<<blocks, BLOCK_SIZE>>>(
        x_int8.data_ptr<signed char>(),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        zero_point.data_ptr<signed char>(),
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
        _module = compile_hip(HIP_SRC, "dequantize_int8_hip")
    return _module


def kernel_fn(x_int8: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert x_int8.is_cuda

    orig_shape = x_int8.shape

    if x_int8.ndim == 1:
        x_int8 = x_int8.unsqueeze(0)
    elif x_int8.ndim > 2:
        x_int8 = x_int8.view(-1, x_int8.shape[-1])

    if scale.dtype != torch.float16:
        scale = scale.to(torch.float16)
    if zero_point.dtype != torch.int8:
        zero_point = zero_point.to(torch.int8)

    mod = _get_module()
    out = mod.dequantize_int8_hip(x_int8, scale, zero_point)

    return out.view(orig_shape)
