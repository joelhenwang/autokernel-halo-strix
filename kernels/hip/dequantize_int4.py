"""
AutoKernel -- HIP C++ Int4 Dequantization kernel.

Per-channel int4 dequantization: unpack 2 int4 values per byte, apply scale and zero_point.
Output layout: even columns from low nibble, odd columns from high nibble.

Features:
  - Packed uint8 loads (1 byte = 2 int4 values)
  - Nibble extraction via bitwise ops
  - Per-channel scale/zero_point broadcast
  - Vectorized half2 output writes (naturally 2 outputs per packed byte)
  - Grid-stride loop for arbitrary tensor sizes
"""

KERNEL_TYPE = "dequantize_int4"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;

__global__ void __launch_bounds__(BLOCK_SIZE)
dequantize_int4_kernel(
    const unsigned char* __restrict__ packed,  // [M, N_packed] uint8, 2 int4 per byte
    const half* __restrict__ scale,            // [N_packed] fp16
    const unsigned char* __restrict__ zero_pt, // [N_packed] uint8
    half* __restrict__ output,                 // [M, N_out] fp16, N_out = N_packed * 2
    int M, int N_packed
) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int stride = gridDim.x * BLOCK_SIZE;
    const int total = M * N_packed;  // one packed byte per element

    for (int i = tid; i < total; i += stride) {
        int row = i / N_packed;
        int col = i % N_packed;

        unsigned char byte = packed[i];
        float lo = (float)(byte & 0x0F);
        float hi = (float)((byte >> 4) & 0x0F);

        float zp = (float)zero_pt[col];
        float s = __half2float(scale[col]);

        float y_lo = (lo - zp) * s;
        float y_hi = (hi - zp) * s;

        // Write as half2: even col = lo, odd col = hi
        int out_idx = row * N_packed * 2 + col * 2;
        half2 result;
        result.x = __float2half(y_lo);
        result.y = __float2half(y_hi);
        *reinterpret_cast<half2*>(output + out_idx) = result;
    }
}

torch::Tensor dequantize_int4_hip(
    torch::Tensor x_packed,
    torch::Tensor scale,
    torch::Tensor zero_point
) {
    TORCH_CHECK(x_packed.is_cuda(), "x_packed must be a GPU tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a GPU tensor");
    TORCH_CHECK(zero_point.is_cuda(), "zero_point must be a GPU tensor");
    TORCH_CHECK(x_packed.dtype() == torch::kUInt8, "x_packed must be uint8");
    TORCH_CHECK(scale.dtype() == torch::kFloat16, "scale must be float16");
    TORCH_CHECK(zero_point.dtype() == torch::kUInt8, "zero_point must be uint8");

    int M = x_packed.size(0);
    int N_packed = x_packed.size(1);
    int N_out = N_packed * 2;

    auto output = torch::empty({M, N_out}, torch::TensorOptions().device(x_packed.device()).dtype(torch::kFloat16));

    int total = M * N_packed;
    int blocks = min((total + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);

    dequantize_int4_kernel<<<blocks, BLOCK_SIZE>>>(
        x_packed.data_ptr<unsigned char>(),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        zero_point.data_ptr<unsigned char>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N_packed
    );

    return output;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "dequantize_int4_hip")
    return _module


def kernel_fn(x_packed: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert x_packed.is_cuda

    if x_packed.ndim == 1:
        x_packed = x_packed.unsqueeze(0)
    elif x_packed.ndim > 2:
        x_packed = x_packed.view(-1, x_packed.shape[-1])

    if scale.dtype != torch.float16:
        scale = scale.to(torch.float16)

    mod = _get_module()
    out = mod.dequantize_int4_hip(x_packed, scale, zero_point)

    return out
