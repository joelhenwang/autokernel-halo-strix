"""
AutoKernel -- HIP C++ Fused PLE Gate kernel (Per-Layer Embeddings Path A).

Fuses: Linear(d_model -> ple_dim) -> GELU -> elementwise_mul -> Linear(ple_dim -> d_model) -> RMSNorm
into a single kernel, avoiding materializing the ple_dim intermediate to global memory.

At ple_dim=64 this is purely memory-bound: the intermediate is only 64 floats per position.
A single kernel saves 2 round-trips to DRAM for the bottleneck tensor.

Input:  h           (M, d_model)   -- hidden state, fp16
        W_down      (ple_dim, d_model) -- down projection weights, fp16
        W_up        (d_model, ple_dim) -- up projection weights, fp16
        norm_weight (d_model,)     -- RMSNorm weight, fp16

Output: (M, d_model) -- PLE contribution to add to residual stream, fp16

Each thread block handles one row (one position in the sequence).
Phase 1: Compute h @ W_down^T -> bottleneck (ple_dim values, held in registers/LDS)
Phase 2: Apply GELU activation in-place
Phase 3: Compute bottleneck @ W_up^T -> output (d_model values)
Phase 4: RMSNorm the output
"""

KERNEL_TYPE = "fused_ple_gate"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr float EPS = 1e-6f;
constexpr int WARP_SIZE = 32;
// PLE_DIM is small (64 by default). We keep the bottleneck in LDS.
constexpr int MAX_PLE_DIM = 128;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float gelu_approx(float x) {
    // Fast GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c = 0.7978845608f;  // sqrt(2/pi)
    const float k = 0.044715f;
    float x3 = x * x * x;
    float inner = c * (x + k * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Fused PLE Path A kernel:
//   bottleneck = GELU(h @ W_down^T)      -- (M, ple_dim)
//   raw_out = bottleneck @ W_up^T         -- (M, d_model)
//   output = RMSNorm(raw_out, norm_w)     -- (M, d_model)
//
// One block per row (position). Thread count = max(d_model/2, 32) capped at 1024.
// Bottleneck stored in shared memory (ple_dim * sizeof(float) per block).

__global__ void fused_ple_gate_kernel(
    const half* __restrict__ H,          // (M, d_model)
    const half* __restrict__ W_down,     // (ple_dim, d_model)
    const half* __restrict__ W_up,       // (d_model, ple_dim)
    const half* __restrict__ norm_weight, // (d_model,)
    half* __restrict__ OUT,               // (M, d_model)
    int M,        // number of rows (B*T)
    int D,        // d_model
    int P         // ple_dim
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockSize / WARP_SIZE;

    const half* h_row = H + (long long)row * D;
    half* out_row = OUT + (long long)row * D;

    // Shared memory for bottleneck values (ple_dim floats)
    extern __shared__ float smem[];
    float* bottleneck = smem;  // [P] floats

    // =============================================
    // Phase 1: h @ W_down^T -> bottleneck (P values)
    // Each thread computes partial dot products for assigned bottleneck dims
    // =============================================
    for (int p = tid; p < P; p += blockSize) {
        const half* w_row = W_down + (long long)p * D;
        float acc = 0.0f;

        // Vectorized dot product h_row . w_row
        int d = 0;
        for (; d + 1 < D; d += 2) {
            half2 hv = *reinterpret_cast<const half2*>(h_row + d);
            half2 wv = *reinterpret_cast<const half2*>(w_row + d);
            acc += __half2float(hv.x) * __half2float(wv.x);
            acc += __half2float(hv.y) * __half2float(wv.y);
        }
        if (d < D) {
            acc += __half2float(h_row[d]) * __half2float(w_row[d]);
        }

        // Phase 2: GELU activation (fused)
        bottleneck[p] = gelu_approx(acc);
    }
    __syncthreads();

    // =============================================
    // Phase 3: bottleneck @ W_up^T -> raw_out (D values)
    // Phase 4: RMSNorm(raw_out)
    // =============================================

    // First compute raw output and accumulate sum-of-squares for RMSNorm
    // Each thread handles multiple output dimensions
    float local_sum_sq = 0.0f;

    // We need the raw output values for RMSNorm, store temporarily
    // Use a second region of shared memory for partial results
    // Actually, we compute in two passes:
    //   Pass A: compute raw_out values and sum_sq (thread-local storage via registers + LDS)
    //   Pass B: normalize and write

    // Since D can be large (1024), each thread computes a few output dims
    // We'll do it with grid-stride over output dimensions

    // Pass A: compute raw_out and accumulate sum_sq
    // We store raw_out values back to the output buffer temporarily in fp32
    // Actually, let's use the output buffer as scratch (write fp16, re-read for norm)

    for (int d = tid; d < D; d += blockSize) {
        const half* w_col = W_up + (long long)d * P;  // W_up is (D, P), row d
        float acc = 0.0f;

        for (int p = 0; p < P; p++) {
            acc += bottleneck[p] * __half2float(w_col[p]);
        }

        // Store raw value temporarily
        out_row[d] = __float2half(acc);
        local_sum_sq += acc * acc;
    }

    // Reduce sum_sq across block for RMSNorm
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    // Re-use part of smem for cross-warp reduction (after bottleneck is no longer needed)
    __syncthreads();
    float* shared_sums = smem;  // reuse bottleneck memory

    if (lane_id == 0) {
        shared_sums[warp_id] = local_sum_sq;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane_id < num_warps) ? shared_sums[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }

    __shared__ float s_rms_inv;
    if (tid == 0) {
        float mean_sq = block_sum / static_cast<float>(D);
        s_rms_inv = rsqrtf(mean_sq + EPS);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;

    // Pass B: Read back raw values, apply RMSNorm, write final output
    for (int d = tid * 2; d + 1 < D; d += blockSize * 2) {
        half2 raw_val = *reinterpret_cast<const half2*>(out_row + d);
        half2 w_val = *reinterpret_cast<const half2*>(norm_weight + d);

        float r0 = __half2float(raw_val.x) * rms_inv * __half2float(w_val.x);
        float r1 = __half2float(raw_val.y) * rms_inv * __half2float(w_val.y);

        half2 result;
        result.x = __float2half(r0);
        result.y = __float2half(r1);
        *reinterpret_cast<half2*>(out_row + d) = result;
    }
    // Handle odd tail
    {
        int d = ((D / 2) * 2);  // start of potential tail
        d += (tid == 0 && (D & 1)) ? 0 : D;  // only tid 0 handles tail
        if (d < D) {
            float raw = __half2float(out_row[d]);
            float w = __half2float(norm_weight[d]);
            out_row[d] = __float2half(raw * rms_inv * w);
        }
    }
}

torch::Tensor fused_ple_gate_hip(
    torch::Tensor h,
    torch::Tensor w_down,
    torch::Tensor w_up,
    torch::Tensor norm_weight
) {
    TORCH_CHECK(h.is_cuda(), "h must be a GPU tensor");
    TORCH_CHECK(w_down.is_cuda(), "w_down must be a GPU tensor");
    TORCH_CHECK(w_up.is_cuda(), "w_up must be a GPU tensor");
    TORCH_CHECK(norm_weight.is_cuda(), "norm_weight must be a GPU tensor");
    TORCH_CHECK(h.dtype() == torch::kFloat16, "h must be float16");
    TORCH_CHECK(w_down.dtype() == torch::kFloat16, "w_down must be float16");

    int M = h.size(0);  // B*T
    int D = h.size(1);  // d_model
    int P = w_down.size(0);  // ple_dim

    TORCH_CHECK(w_down.size(1) == D, "w_down must be (ple_dim, d_model)");
    TORCH_CHECK(w_up.size(0) == D, "w_up must be (d_model, ple_dim)");
    TORCH_CHECK(w_up.size(1) == P, "w_up dim mismatch");
    TORCH_CHECK(norm_weight.size(0) == D, "norm_weight must be (d_model,)");

    auto out = torch::empty_like(h);

    // Thread count: enough to cover D/2 for vectorized loads, capped
    int threads = min(1024, max(32, ((D + 1) / 2 + 31) / 32 * 32));

    // Shared memory: max(P, num_warps) floats for bottleneck + reduction
    int num_warps = threads / WARP_SIZE;
    int smem_bytes = max(P, num_warps) * sizeof(float);

    dim3 grid(M);
    dim3 block(threads);

    fused_ple_gate_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w_down.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w_up.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(norm_weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, D, P
    );

    return out;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "fused_ple_gate_hip")
    return _module


def kernel_fn(
    h: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Fused PLE Path A: Linear->GELU->Linear->RMSNorm.

    Args:
        h: (B*T, d_model) or (B, T, d_model) hidden state, fp16
        w_down: (ple_dim, d_model) down projection weight, fp16
        w_up: (d_model, ple_dim) up projection weight, fp16
        norm_weight: (d_model,) RMSNorm weight, fp16

    Returns:
        (same shape as h) PLE contribution to add to residual
    """
    assert h.is_cuda
    orig_shape = h.shape
    orig_dtype = h.dtype

    # Fallback for non-fp16
    if orig_dtype != torch.float16:
        bottleneck = torch.nn.functional.gelu(h @ w_down.t())
        raw = bottleneck @ w_up.t()
        rms = (raw.float().pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        return (raw.float() * rms * norm_weight.float()).to(orig_dtype)

    # Flatten to 2D
    h_2d = h.reshape(-1, h.shape[-1]).contiguous()
    w_down = w_down.contiguous()
    w_up = w_up.contiguous()
    norm_weight = norm_weight.contiguous()

    mod = _get_module()
    out = mod.fused_ple_gate_hip(h_2d, w_down, w_up, norm_weight)

    return out.view(orig_shape)


# --- PyTorch reference for correctness testing ---

def reference_fn(
    h: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    bottleneck = torch.nn.functional.gelu(h @ w_down.t())
    raw = bottleneck @ w_up.t()
    rms = (raw.float().pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
    return (raw.float() * rms * norm_weight.float()).to(h.dtype)
