"""
AutoKernel -- HIP C++ Fused Engram Gate + Value + Conv kernel (Variant B).

Fuses: dot-product gate (DeepSeek magnitude-preserving) + gated value + depthwise conv1d
into a single kernel, eliminating 3 intermediate tensors.

Operations fused (per position):
  gate_raw = dot(query, key) / sqrt(D)
  gate = sigmoid(abs(gate_raw).clamp(1e-6).sqrt() * sign(gate_raw))
  gated_value = gate * value
  conv_value = depthwise_conv1d(value, k=3)  # causal padding
  output = gated_value + conv_value

Input:  query       (M, D)   -- RMSNorm'd hidden state, fp16
        key         (M, D)   -- projected engram key, fp16
        value       (M, D)   -- projected engram value, fp16
        conv_weight (D, K)   -- depthwise conv weights, fp16 (K=3)
        conv_bias   (D,)     -- depthwise conv bias, fp16
        seq_len     int      -- sequence length T (to avoid conv across batch boundaries)

Output: (M, D)  -- engram output to add to residual, fp16

Each thread block handles one row (one position). Holds gate scalar in register,
loads value neighbors for conv from global memory (L2-cached for adjacent positions).
"""

KERNEL_TYPE = "fused_engram_gate_conv"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Fused Engram gate + value + conv kernel
// One block per row (position in flattened B*T)
__global__ void fused_engram_gate_conv_kernel(
    const half* __restrict__ Q,          // (M, D) query (rmsnorm'd hidden)
    const half* __restrict__ K,          // (M, D) key (projected engram)
    const half* __restrict__ V,          // (M, D) value (projected engram)
    const half* __restrict__ conv_w,     // (D, conv_k) depthwise conv weights
    const half* __restrict__ conv_b,     // (D,) conv bias
    half* __restrict__ OUT,              // (M, D) output
    int M,           // total rows (B*T)
    int D,           // d_model
    int T,           // seq_len (to detect batch boundaries)
    int conv_k       // conv kernel size (3)
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockSize / WARP_SIZE;

    const half* q_row = Q + (int64_t)row * D;
    const half* k_row = K + (int64_t)row * D;
    const half* v_row = V + (int64_t)row * D;
    half* out_row = OUT + (int64_t)row * D;

    // Position within the current sequence (for causal conv boundary)
    int pos_in_seq = row % T;

    // =============================================
    // Phase 1: Compute gate scalar
    //   gate_raw = dot(query, key) / sqrt(D)
    //   gate = sigmoid(abs(gate_raw).sqrt() * sign(gate_raw))
    // =============================================

    float local_dot = 0.0f;

    // Vectorized dot product
    int d = tid * 2;
    for (; d + 1 < D; d += blockSize * 2) {
        half2 qv = *reinterpret_cast<const half2*>(q_row + d);
        half2 kv = *reinterpret_cast<const half2*>(k_row + d);
        local_dot += __half2float(qv.x) * __half2float(kv.x);
        local_dot += __half2float(qv.y) * __half2float(kv.y);
    }
    if (d < D) {
        local_dot += __half2float(q_row[d]) * __half2float(k_row[d]);
    }

    // Warp reduction
    local_dot = warp_reduce_sum(local_dot);

    // Block reduction via shared memory
    __shared__ float shared_sums[32];
    if (lane_id == 0) {
        shared_sums[warp_id] = local_dot;
    }
    __syncthreads();

    float block_dot = 0.0f;
    if (warp_id == 0) {
        block_dot = (lane_id < num_warps) ? shared_sums[lane_id] : 0.0f;
        block_dot = warp_reduce_sum(block_dot);
    }

    // Broadcast gate to all threads
    __shared__ float s_gate;
    if (tid == 0) {
        float gate_raw = block_dot / sqrtf((float)D);
        // DeepSeek magnitude-preserving gate
        float abs_gate = (gate_raw >= 0.0f) ? gate_raw : -gate_raw;
        abs_gate = (abs_gate < 1e-6f) ? 1e-6f : abs_gate;  // clamp
        float sign_gate = (gate_raw >= 0.0f) ? 1.0f : -1.0f;
        float mp_gate = sqrtf(abs_gate) * sign_gate;
        // Sigmoid
        s_gate = 1.0f / (1.0f + __expf(-mp_gate));
    }
    __syncthreads();

    float gate = s_gate;

    // =============================================
    // Phase 2: Gated value + depthwise conv1d
    //   gated_value = gate * value[pos]
    //   conv_value = sum_j(value[pos-j] * conv_w[d,j]) + conv_b[d]
    //   output = gated_value + conv_value
    // =============================================

    // Vectorized output: process 2 channels at a time
    for (d = tid * 2; d + 1 < D; d += blockSize * 2) {
        // Gated value at current position
        half2 vv = *reinterpret_cast<const half2*>(v_row + d);
        float gv0 = gate * __half2float(vv.x);
        float gv1 = gate * __half2float(vv.y);

        // Depthwise conv1d (causal, k=conv_k)
        // conv_out[d] = sum_{j=0}^{conv_k-1} value[pos-j, d] * weight[d, j] + bias[d]
        float cv0 = __half2float(conv_b[d]);
        float cv1 = __half2float(conv_b[d + 1]);

        for (int j = 0; j < conv_k; j++) {
            int src_pos_in_seq = pos_in_seq - j;
            if (src_pos_in_seq >= 0) {
                // Same batch, valid position
                int src_row = row - j;
                const half* src_v = V + (int64_t)src_row * D;
                half2 sv = *reinterpret_cast<const half2*>(src_v + d);

                // Weight layout: (D, conv_k) — channel d, kernel position j
                float w0 = __half2float(conv_w[d * conv_k + j]);
                float w1 = __half2float(conv_w[(d + 1) * conv_k + j]);

                cv0 += __half2float(sv.x) * w0;
                cv1 += __half2float(sv.y) * w1;
            }
            // else: zero padding (causal, no future positions)
        }

        // Output = gated_value + conv_value
        half2 result;
        result.x = __float2half(gv0 + cv0);
        result.y = __float2half(gv1 + cv1);
        *reinterpret_cast<half2*>(out_row + d) = result;
    }

    // Handle odd tail
    if (tid == 0 && (D & 1)) {
        d = D - 1;
        float gv = gate * __half2float(v_row[d]);
        float cv = __half2float(conv_b[d]);
        for (int j = 0; j < conv_k; j++) {
            int src_pos = pos_in_seq - j;
            if (src_pos >= 0) {
                int src_row = row - j;
                cv += __half2float(V[(int64_t)src_row * D + d]) * __half2float(conv_w[d * conv_k + j]);
            }
        }
        out_row[d] = __float2half(gv + cv);
    }
}

torch::Tensor fused_engram_gate_conv_hip(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    int64_t seq_len
) {
    TORCH_CHECK(query.is_cuda(), "query must be a GPU tensor");
    TORCH_CHECK(query.dtype() == torch::kFloat16, "query must be float16");
    TORCH_CHECK(key.dtype() == torch::kFloat16, "key must be float16");
    TORCH_CHECK(value.dtype() == torch::kFloat16, "value must be float16");

    int M = query.size(0);  // B*T
    int D = query.size(1);  // d_model
    int T = seq_len;
    int K = conv_weight.size(1);  // conv kernel size

    auto out = torch::empty_like(query);

    int threads = min(1024, max(32, ((D + 1) / 2 + 31) / 32 * 32));

    fused_engram_gate_conv_kernel<<<M, threads>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(conv_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(conv_bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, D, T, K
    );

    return out;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "fused_engram_gate_conv_hip")
    return _module


def kernel_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Fused Engram gate + value + conv.

    Args:
        query: (B*T, D) or (B, T, D) RMSNorm'd hidden state, fp16
        key: (B*T, D) or (B, T, D) projected engram key, fp16
        value: (B*T, D) or (B, T, D) projected engram value, fp16
        conv_weight: (D, K) depthwise conv weights, fp16
        conv_bias: (D,) depthwise conv bias, fp16
        seq_len: int, sequence length T

    Returns:
        (same shape as query) engram output
    """
    assert query.is_cuda
    orig_shape = query.shape
    orig_dtype = query.dtype

    if orig_dtype != torch.float16:
        return reference_fn(query, key, value, conv_weight, conv_bias, seq_len)

    q_2d = query.reshape(-1, query.shape[-1]).contiguous()
    k_2d = key.reshape(-1, key.shape[-1]).contiguous()
    v_2d = value.reshape(-1, value.shape[-1]).contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()

    mod = _get_module()
    out = mod.fused_engram_gate_conv_hip(q_2d, k_2d, v_2d, conv_weight, conv_bias, seq_len)

    return out.view(orig_shape)


def reference_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    D = query.shape[-1]
    # Gate
    gate_raw = (query * key).sum(dim=-1, keepdim=True) / (D ** 0.5)
    gate = gate_raw.abs().clamp(min=1e-6).sqrt() * gate_raw.sign()
    gate = torch.sigmoid(gate)
    # Gated value
    gated_value = gate * value
    # Depthwise conv1d (causal)
    B_T = value.shape[0] if value.dim() == 2 else value.shape[0] * value.shape[1]
    v_3d = value.reshape(-1, seq_len, D)  # (B, T, D)
    v_conv = v_3d.transpose(1, 2)  # (B, D, T)
    K = conv_weight.shape[1]
    conv_out = torch.nn.functional.conv1d(
        v_conv, conv_weight.unsqueeze(1), conv_bias,
        padding=K - 1, groups=D
    )[:, :, :seq_len]  # causal: trim future
    conv_value = conv_out.transpose(1, 2).reshape_as(value)  # back to (B*T, D) or (B, T, D)
    return gated_value + conv_value
