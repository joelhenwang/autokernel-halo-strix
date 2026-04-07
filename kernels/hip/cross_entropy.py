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

// Wavefront-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down(val, offset));
    return val;
}

// Wavefront-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down(val, offset);
    return val;
}

// Each block handles one sample in the batch
__global__ void cross_entropy_kernel(
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
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    const half* row = logits + b * vocab;
    const int target = targets[b];

    // Phase 1: Find row max
    float local_max = -FLT_MAX;
    for (int v = tid; v < vocab; v += BLOCK_SIZE) {
        local_max = fmaxf(local_max, __half2float(row[v]));
    }

    local_max = warp_reduce_max(local_max);

    __shared__ float smem_max[32];
    if (lane_id == 0) smem_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem_max[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_id == 0) smem_max[0] = val;
    }
    __syncthreads();
    float row_max = smem_max[0];

    // Phase 2: Compute sum of exp(logits - max)
    float local_sum = 0.0f;
    for (int v = tid; v < vocab; v += BLOCK_SIZE) {
        local_sum += __expf(__half2float(row[v]) - row_max);
    }

    local_sum = warp_reduce_sum(local_sum);

    __shared__ float smem_sum[32];
    if (lane_id == 0) smem_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem_sum[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) smem_sum[0] = val;
    }
    __syncthreads();
    float row_sum = smem_sum[0];

    // Phase 3: Compute loss = -logits[target] + max + log(sum)
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

    if logits.dtype != torch.float16:
        logits = logits.to(torch.float16)

    mod = _get_module()
    loss = mod.cross_entropy_hip(logits, targets)

    return loss.to(torch.float32)
