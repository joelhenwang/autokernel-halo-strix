"""
AutoKernel -- HIP C++ Fused SiLU-Gate-Multiply (SwiGLU activation) kernel.

Computes: output = SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
Used in every FFN block of LLaMA/Mistral/Mixtral (2x per transformer layer).

Features:
  - Fuses SiLU activation with elementwise multiply (eliminates intermediate tensor)
  - Vectorized half2 loads/stores for both gate and up tensors
  - fp32 intermediate computation for accuracy
  - Grid-stride loop for arbitrary tensor sizes
"""

KERNEL_TYPE = "silu_gate_mul"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;

__global__ void __launch_bounds__(BLOCK_SIZE)
silu_gate_mul_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int N
) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int stride = gridDim.x * BLOCK_SIZE;

    const half2* gate_v = reinterpret_cast<const half2*>(gate);
    const half2* up_v = reinterpret_cast<const half2*>(up);
    half2* out_v = reinterpret_cast<half2*>(output);
    const int n_pairs = N / 2;

    for (int i = tid; i < n_pairs; i += stride) {
        half2 g = gate_v[i];
        half2 u = up_v[i];

        float g0 = __half2float(g.x);
        float g1 = __half2float(g.y);
        float u0 = __half2float(u.x);
        float u1 = __half2float(u.y);

        // SiLU(g) * u = g / (1 + exp(-g)) * u
        float y0 = (g0 / (1.0f + __expf(-g0))) * u0;
        float y1 = (g1 / (1.0f + __expf(-g1))) * u1;

        half2 result;
        result.x = __float2half(y0);
        result.y = __float2half(y1);
        out_v[i] = result;
    }

    // Odd tail
    if (tid == 0 && (N & 1)) {
        float g = __half2float(gate[N - 1]);
        float u = __half2float(up[N - 1]);
        output[N - 1] = __float2half((g / (1.0f + __expf(-g))) * u);
    }
}

torch::Tensor silu_gate_mul_hip(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda(), "gate must be a GPU tensor");
    TORCH_CHECK(up.is_cuda(), "up must be a GPU tensor");
    TORCH_CHECK(gate.dtype() == torch::kFloat16, "gate must be float16");
    TORCH_CHECK(up.dtype() == torch::kFloat16, "up must be float16");

    auto output = torch::empty_like(gate);
    int N = gate.numel();

    int n_pairs = N / 2;
    int blocks = min((n_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);

    silu_gate_mul_kernel<<<blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
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
        _module = compile_hip(HIP_SRC, "silu_gate_mul_hip")
    return _module


def kernel_fn(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert gate.is_cuda and up.is_cuda

    orig_shape = gate.shape
    orig_dtype = gate.dtype

    if orig_dtype != torch.float16:
        return torch.nn.functional.silu(gate) * up

    gate = gate.contiguous()
    up = up.contiguous()

    mod = _get_module()
    out = mod.silu_gate_mul_hip(gate, up)

    return out.view(orig_shape)
