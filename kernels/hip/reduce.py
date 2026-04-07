"""
AutoKernel -- HIP C++ Reduce (sum along last dim) kernel.

Current kernel: Hierarchical reduction with warp shuffle + shared memory.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - __shfl_down cascade for intra-wavefront reduction (wave32 on RDNA 3.5)
  - Shared memory for inter-wavefront reduction within a block
  - Vectorized loads for memory bandwidth
  - One row per block for simplicity and correctness
"""

KERNEL_TYPE = "reduce"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;  // RDNA wave32 mode

// Wavefront-level sum reduction (wave32)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down(val, offset);
    return val;
}

// One block per row. Each block cooperatively reduces one row of [M, N] to one scalar.
__global__ void reduce_sum_kernel(
    const half* __restrict__ input,  // [M, N]
    half* __restrict__ output,       // [M]
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    const half* row_ptr = input + row * N;

    // Phase 1: thread-local accumulation via grid-stride
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        local_sum += __half2float(row_ptr[i]);
    }

    // Phase 2: wavefront-level reduction
    local_sum = warp_reduce_sum(local_sum);

    // Phase 3: block-level reduction via shared memory
    __shared__ float smem[WARP_SIZE];  // one per wavefront

    if (lane_id == 0) {
        smem[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by first wavefront
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            output[row] = __float2half(val);
        }
    }
}

torch::Tensor reduce_sum_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be on GPU");
    TORCH_CHECK(input.dim() == 2, "input must be [M, N]");

    int M = input.size(0);
    int N = input.size(1);

    auto output = torch::empty({M}, input.options());

    dim3 grid(M);
    dim3 block(BLOCK_SIZE);

    reduce_sum_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
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
        _module = compile_hip(HIP_SRC, "reduce_sum_hip")
    return _module


def kernel_fn(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.reduce_sum_ref signature."""
    assert x.is_cuda

    # Normalize dim
    if dim < 0:
        dim = x.ndim + dim
    assert 0 <= dim < x.ndim

    orig_dtype = x.dtype

    # For the common case of reducing the last dimension on a 2D tensor
    if dim == x.ndim - 1:
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        # Ensure 2D
        orig_shape = list(x.shape)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        mod = _get_module()
        out = mod.reduce_sum_hip(x)

        if orig_dtype != torch.float16:
            out = out.to(orig_dtype)

        # Restore output shape (input shape minus last dim)
        out_shape = orig_shape[:-1]
        if not out_shape:
            out_shape = [1]
        return out.view(out_shape)
    else:
        # General case: move reduction dim to last, then reduce
        perm = list(range(x.ndim))
        perm.pop(dim)
        perm.append(dim)
        x = x.permute(*perm).contiguous()
        return kernel_fn(x, dim=-1)
