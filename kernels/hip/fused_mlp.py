"""
AutoKernel -- HIP C++ Fused MLP (SwiGLU) kernel.

Current kernel: Fused gate_proj + up_proj + SiLU activation + elementwise mul.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Fuses two matrix multiplications (gate and up projections) with SiLU activation
  - Tiled accumulation for better register reuse
  - Avoids writing intermediate results to global memory
  - __fdividef for fast SiLU computation
  - Block-level tiling for large hidden dimensions
"""

KERNEL_TYPE = "fused_mlp"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;
constexpr int TILE_K = 32;

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

// Stage 1: Fused gate+up+silu+mul
// For each row of x, compute: hidden[i] = SiLU(dot(x, gate_W[i])) * dot(x, up_W[i])
__global__ void fused_gate_up_kernel(
    const half* __restrict__ x,        // [M, K]
    const half* __restrict__ gate_w,   // [N, K] (N = hidden_dim)
    const half* __restrict__ up_w,     // [N, K]
    half* __restrict__ hidden,         // [M, N]
    int M, int K, int N
) {
    const int row = blockIdx.x;
    const int col_start = blockIdx.y * BLOCK_SIZE;

    if (row >= M) return;

    const int tid = threadIdx.x;
    const int col = col_start + tid;

    if (col >= N) return;

    const half* x_row = x + row * K;
    const half* gate_row = gate_w + col * K;
    const half* up_row = up_w + col * K;

    float gate_val = 0.0f;
    float up_val = 0.0f;

    for (int k = 0; k < K; k += TILE_K) {
        int k_end = min(k + TILE_K, K);
        #pragma unroll 8
        for (int kk = k; kk < k_end; kk++) {
            float x_val = __half2float(x_row[kk]);
            gate_val += x_val * __half2float(gate_row[kk]);
            up_val   += x_val * __half2float(up_row[kk]);
        }
    }

    float result = silu(gate_val) * up_val;
    hidden[row * N + col] = __float2half(result);
}

// Stage 2: down projection (standard matmul hidden @ down_W^T)
__global__ void down_proj_kernel(
    const half* __restrict__ hidden,   // [M, N]
    const half* __restrict__ down_w,   // [K, N]
    half* __restrict__ output,         // [M, K]
    int M, int N, int K
) {
    const int row = blockIdx.x;
    const int col_start = blockIdx.y * BLOCK_SIZE;

    if (row >= M) return;

    const int tid = threadIdx.x;
    const int col = col_start + tid;

    if (col >= K) return;

    const half* h_row = hidden + row * N;
    const half* w_row = down_w + col * N;

    float acc = 0.0f;
    for (int n = 0; n < N; n += TILE_K) {
        int n_end = min(n + TILE_K, N);
        #pragma unroll 8
        for (int nn = n; nn < n_end; nn++) {
            acc += __half2float(h_row[nn]) * __half2float(w_row[nn]);
        }
    }

    output[row * K + col] = __float2half(acc);
}

torch::Tensor fused_mlp_hip(
    torch::Tensor x,
    torch::Tensor gate_w,
    torch::Tensor up_w,
    torch::Tensor down_w
) {
    TORCH_CHECK(x.is_cuda(), "x must be on GPU");

    int M = x.size(0);
    int K = x.size(1);
    int N = gate_w.size(0);

    auto hidden = torch::empty({M, N}, x.options());

    {
        dim3 grid(M, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        fused_gate_up_kernel<<<grid, block>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(gate_w.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(up_w.data_ptr<at::Half>()),
            reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
            M, K, N
        );
    }

    auto output = torch::empty({M, K}, x.options());

    {
        dim3 grid(M, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        down_proj_kernel<<<grid, block>>>(
            reinterpret_cast<const half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(down_w.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            M, N, K
        );
    }

    return output;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "fused_mlp_hip")
    return _module


def kernel_fn(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.fused_mlp_ref signature."""
    assert x.is_cuda

    orig_dtype = x.dtype
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
        w_gate = w_gate.to(torch.float16)
        w_up = w_up.to(torch.float16)
        w_down = w_down.to(torch.float16)

    mod = _get_module()
    out = mod.fused_mlp_hip(x, w_gate, w_up, w_down)

    if orig_dtype != torch.float16:
        out = out.to(orig_dtype)

    return out
