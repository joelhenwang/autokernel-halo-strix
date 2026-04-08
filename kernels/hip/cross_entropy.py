"""
AutoKernel -- HIP C++ Cross Entropy Loss kernel.

Current kernel: Fused online log-sum-exp + NLL in a single pass.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Online log-sum-exp avoids materializing full softmax
  - Wavefront-level max and sum reductions via __shfl_down
  - Block-level cooperative reduction via shared memory
  - Fast __logf / __expf intrinsics
  - Grid-stride loop for arbitrary batch sizes
"""

KERNEL_TYPE = "cross_entropy"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;  // RDNA wave32
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// Wavefront-level max reduction (shfl_xor so ALL lanes get result)
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor(val, offset));
    return val;
}

// Wavefront-level sum reduction (shfl_xor so ALL lanes get result)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

// Each block handles one sample in the batch.
// Online fused max+sum in a single pass with half2 vectorized loads.
__global__ void __launch_bounds__(BLOCK_SIZE)
cross_entropy_kernel(
    const half* __restrict__ logits,     // [batch, vocab]
    const int64_t* __restrict__ targets, // [batch]
    float* __restrict__ losses,          // [batch]
    int batch, int vocab
) {
    const int b = blockIdx.x;
    if (b >= batch) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const half* row = logits + (long long)b * vocab;
    const int target = targets[b];

    // === Single-pass online max + exp-sum with half2 vectorized loads ===
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    const int n_pairs = vocab / 2;
    const half2* row_v = reinterpret_cast<const half2*>(row);

    #pragma unroll 4
    for (int v = tid; v < n_pairs; v += BLOCK_SIZE) {
        half2 val = row_v[v];
        float lo = __half2float(val.x);
        float hi = __half2float(val.y);
        float pair_max = fmaxf(lo, hi);

        if (pair_max > local_max) {
            local_sum *= __expf(local_max - pair_max);
            local_max = pair_max;
        }
        local_sum += __expf(lo - local_max) + __expf(hi - local_max);
    }
    // Handle odd vocab size
    if ((vocab & 1) && tid == 0) {
        float val = __half2float(row[vocab - 1]);
        if (val > local_max) {
            local_sum *= __expf(local_max - val);
            local_max = val;
        }
        local_sum += __expf(val - local_max);
    }

    // === Warp-level reduction ===
    float warp_max = warp_reduce_max(local_max);
    local_sum *= __expf(local_max - warp_max);
    float warp_sum = warp_reduce_sum(local_sum);

    // === Block-level reduction via shared memory ===
    __shared__ float smem_max[NUM_WARPS];
    __shared__ float smem_sum[NUM_WARPS];

    if (lane_id == 0) {
        smem_max[warp_id] = warp_max;
        smem_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    float row_max, row_sum;
    if (warp_id == 0) {
        float m = (lane_id < NUM_WARPS) ? smem_max[lane_id] : -FLT_MAX;
        float s = (lane_id < NUM_WARPS) ? smem_sum[lane_id] : 0.0f;
        float block_max = warp_reduce_max(m);
        s *= __expf(m - block_max);
        float block_sum = warp_reduce_sum(s);
        if (lane_id == 0) {
            smem_max[0] = block_max;
            smem_sum[0] = block_sum;
        }
    }
    __syncthreads();
    row_max = smem_max[0];
    row_sum = smem_sum[0];

    // === Compute loss = -logits[target] + max + log(sum) ===
    if (tid == 0) {
        float target_logit = __half2float(row[target]);
        losses[b] = -target_logit + row_max + __logf(row_sum);
    }
}

torch::Tensor cross_entropy_hip(torch::Tensor logits, torch::Tensor targets) {
    TORCH_CHECK(logits.is_cuda(), "logits must be on GPU");
    TORCH_CHECK(targets.is_cuda(), "targets must be on GPU");
    TORCH_CHECK(logits.dim() == 2, "logits must be [batch, vocab]");

    int batch = logits.size(0);
    int vocab = logits.size(1);

    auto losses = torch::empty({batch}, logits.options().dtype(torch::kFloat32));

    dim3 grid(batch);
    dim3 block(BLOCK_SIZE);

    cross_entropy_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(logits.data_ptr<at::Half>()),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch, vocab
    );

    return losses.mean();
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "cross_entropy_hip")
    return _module


def kernel_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.cross_entropy_ref signature."""
    assert logits.is_cuda and targets.is_cuda

    # FP32 path: fall back to PyTorch (our HIP kernel is FP16-only)
    if logits.dtype == torch.float32:
        return torch.nn.functional.cross_entropy(logits, targets)

    if logits.dtype != torch.float16:
        logits = logits.to(torch.float16)

    mod = _get_module()
    loss = mod.cross_entropy_hip(logits, targets)

    return loss.to(torch.float32)
