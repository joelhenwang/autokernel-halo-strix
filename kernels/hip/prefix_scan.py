"""
AutoKernel -- HIP C++ Parallel Prefix Scan (Inclusive Cumulative Sum) kernel.

Core operation for Mamba/SSM architectures. Inclusive prefix sum along last dim.

Features:
  - Warp-level scan via __shfl_up (Hillis-Steele within wave32)
  - Block-level tree via shared memory for inter-warp communication
  - One block per row, supports N up to BLOCK_SIZE * elements_per_thread
  - fp32 accumulation for numerical stability
"""

KERNEL_TYPE = "prefix_scan"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

// Warp-level inclusive scan via __shfl_up (Hillis-Steele)
__device__ __forceinline__ float warp_inclusive_scan(float val) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up(val, offset);
        if ((threadIdx.x % WARP_SIZE) >= offset) val += n;
    }
    return val;
}

// Prefix scan kernel: one block per row
// Each thread handles ceil(N / BLOCK_SIZE) elements sequentially,
// then we do a block-wide scan to combine partial sums.
// Uses LDS to store fp32 intermediates (avoids fp16 overflow in cumsum).
__global__ void __launch_bounds__(BLOCK_SIZE)
prefix_scan_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    const half* row_in = input + (long long)row * N;
    half* row_out = output + (long long)row * N;

    // Dynamic shared memory for fp32 prefix sums (avoids fp16 overflow)
    extern __shared__ float s_data[];
    float* s_prefix = s_data;          // N floats for intermediate prefix sums
    float* warp_totals = s_data + N;   // 32 floats
    float* warp_offsets = warp_totals + 32;  // 32 floats

    // Each thread processes a contiguous chunk of elements
    int chunk_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, N);

    // Step 1: Sequential scan within each thread's chunk — store in LDS as fp32
    float running_sum = 0.0f;
    for (int i = start; i < end; i++) {
        running_sum += __half2float(row_in[i]);
        s_prefix[i] = running_sum;  // fp32 in LDS, no overflow
    }

    // Step 2: Block-wide exclusive scan of per-thread totals
    float scanned = warp_inclusive_scan(running_sum);

    if (lane_id == WARP_SIZE - 1) {
        warp_totals[warp_id] = scanned;
    }
    __syncthreads();

    if (warp_id == 0) {
        float wt = (lane_id < num_warps) ? warp_totals[lane_id] : 0.0f;
        float ws = warp_inclusive_scan(wt);
        if (lane_id < num_warps) {
            warp_offsets[lane_id] = ws;
        }
    }
    __syncthreads();

    float warp_offset = (warp_id > 0) ? warp_offsets[warp_id - 1] : 0.0f;
    float thread_offset = warp_offset + scanned - running_sum;

    // Step 3: Read from LDS, add offset in fp32, write final fp16 to global
    for (int i = start; i < end; i++) {
        float val = s_prefix[i] + thread_offset;
        row_out[i] = __float2half(val);
    }
}

torch::Tensor prefix_scan_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a GPU tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, N]");

    int M = input.size(0);
    int N = input.size(1);

    auto output = torch::empty_like(input);

    // Dynamic shared memory: N floats (prefix sums) + 64 floats (warp scratch)
    size_t smem_bytes = (N + 64) * sizeof(float);

    prefix_scan_kernel<<<M, BLOCK_SIZE, smem_bytes>>>(
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
        _module = compile_hip(HIP_SRC, "prefix_scan_hip")
    return _module


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert x.is_cuda

    orig_shape = x.shape
    orig_dtype = x.dtype

    # Non-FP16: fall back to PyTorch
    if orig_dtype != torch.float16:
        return torch.cumsum(x, dim=-1)

    x = x.contiguous()

    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    mod = _get_module()
    out = mod.prefix_scan_hip(x)

    return out.view(orig_shape)
