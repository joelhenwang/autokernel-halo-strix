"""
AutoKernel -- HIP C++ Fused SiLU Gate Multiply Backward kernel.

Fuses the SwiGLU backward into a single element-wise kernel:
  sigmoid = 1 / (1 + exp(-gate))
  d_silu = sigmoid * (1 + gate * (1 - sigmoid))
  grad_gate = grad_output * up * d_silu
  grad_up = grad_output * gate * sigmoid

Replaces 3 separate PyTorch ops (sigmoid, derivative, multiply) with one fused
kernel using half2 vectorized loads/stores.

Input:  gate        (..., N)  fp16
        up          (..., N)  fp16
        grad_output (..., N)  fp16

Output: grad_gate   (..., N)  fp16
        grad_up     (..., N)  fp16
"""

KERNEL_TYPE = "silu_gate_mul_backward"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;

__global__ void silu_gate_mul_backward_kernel(
    const half* __restrict__ GATE,
    const half* __restrict__ UP,
    const half* __restrict__ GRAD_OUT,
    half* __restrict__ GRAD_GATE,
    half* __restrict__ GRAD_UP,
    int N
) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int stride = gridDim.x * BLOCK_SIZE;

    // Process half2 pairs for vectorized memory access
    const int n_pairs = N / 2;

    for (int i = tid; i < n_pairs; i += stride) {
        int base = i * 2;
        half2 gv = *reinterpret_cast<const half2*>(GATE + base);
        half2 uv = *reinterpret_cast<const half2*>(UP + base);
        half2 gov = *reinterpret_cast<const half2*>(GRAD_OUT + base);

        // Process both elements in fp32
        float g0 = __half2float(gv.x);
        float g1 = __half2float(gv.y);
        float u0 = __half2float(uv.x);
        float u1 = __half2float(uv.y);
        float go0 = __half2float(gov.x);
        float go1 = __half2float(gov.y);

        // sigmoid
        float sig0 = 1.0f / (1.0f + __expf(-g0));
        float sig1 = 1.0f / (1.0f + __expf(-g1));

        // d_silu = sig * (1 + gate * (1 - sig))
        float dsilu0 = sig0 * (1.0f + g0 * (1.0f - sig0));
        float dsilu1 = sig1 * (1.0f + g1 * (1.0f - sig1));

        // grad_gate = grad_out * up * d_silu
        float gg0 = go0 * u0 * dsilu0;
        float gg1 = go1 * u1 * dsilu1;

        // grad_up = grad_out * silu(gate) = grad_out * gate * sigmoid
        float gu0 = go0 * g0 * sig0;
        float gu1 = go1 * g1 * sig1;

        half2 rgg, rgu;
        rgg.x = __float2half(gg0);
        rgg.y = __float2half(gg1);
        rgu.x = __float2half(gu0);
        rgu.y = __float2half(gu1);

        *reinterpret_cast<half2*>(GRAD_GATE + base) = rgg;
        *reinterpret_cast<half2*>(GRAD_UP + base) = rgu;
    }

    // Handle odd tail element
    if (N % 2 != 0 && tid == 0) {
        int last = N - 1;
        float g = __half2float(GATE[last]);
        float u = __half2float(UP[last]);
        float go = __half2float(GRAD_OUT[last]);

        float sig = 1.0f / (1.0f + __expf(-g));
        float dsilu = sig * (1.0f + g * (1.0f - sig));

        GRAD_GATE[last] = __float2half(go * u * dsilu);
        GRAD_UP[last] = __float2half(go * g * sig);
    }
}

std::vector<torch::Tensor> silu_gate_mul_backward_hip(
    torch::Tensor gate, torch::Tensor up, torch::Tensor grad_output
) {
    TORCH_CHECK(gate.is_cuda(), "gate must be a GPU tensor");
    TORCH_CHECK(gate.dtype() == torch::kFloat16, "gate must be float16");

    int N = gate.numel();
    auto grad_gate = torch::empty_like(gate);
    auto grad_up = torch::empty_like(up);

    int n_pairs = N / 2;
    int blocks = min((n_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
    if (blocks == 0) blocks = 1;

    silu_gate_mul_backward_kernel<<<blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(grad_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(grad_gate.data_ptr<at::Half>()),
        reinterpret_cast<half*>(grad_up.data_ptr<at::Half>()),
        N
    );

    return {grad_gate, grad_up};
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        cpp_src = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> silu_gate_mul_backward_hip(
    torch::Tensor, torch::Tensor, torch::Tensor);
"""
        _module = compile_hip(HIP_SRC, "silu_gate_mul_backward_hip", cpp_src=cpp_src)
    return _module


def kernel_fn(
    gate: torch.Tensor, up: torch.Tensor, grad_output: torch.Tensor
) -> tuple:
    """Fused SiLU Gate Mul backward.

    Args:
        gate: (..., N) gate input, fp16
        up: (..., N) up input, fp16
        grad_output: (..., N) upstream gradient, fp16

    Returns:
        (grad_gate, grad_up): same shape/dtype as inputs
    """
    assert gate.is_cuda and up.is_cuda and grad_output.is_cuda

    if gate.dtype != torch.float16:
        g_f = gate.float()
        u_f = up.float()
        go_f = grad_output.float()
        sig = torch.sigmoid(g_f)
        d_silu = sig * (1.0 + g_f * (1.0 - sig))
        grad_gate = go_f * u_f * d_silu
        grad_up = go_f * g_f * sig
        return grad_gate.to(gate.dtype), grad_up.to(up.dtype)

    orig_shape = gate.shape
    gate_flat = gate.contiguous().view(-1)
    up_flat = up.contiguous().view(-1)
    grad_flat = grad_output.contiguous().view(-1)

    mod = _get_module()
    results = mod.silu_gate_mul_backward_hip(gate_flat, up_flat, grad_flat)
    return results[0].view(orig_shape), results[1].view(orig_shape)
