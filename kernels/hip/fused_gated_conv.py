"""
AutoKernel -- HIP Fused GatedConv: gate_mul + depthwise_conv + output_gate.

Fuses the post-projection operations of GatedConv into fewer kernel launches:
  Phase 1: y = b * h_tilde (element-wise input gating)
  Phase 2: z = depthwise_conv1d(y); output = c * z (fused conv + output gate)

Eliminates the intermediate `z` tensor between conv and output gating.
Phase 1 writes `y` to global memory (needed for conv lookback across positions).

Input:  proj_out    (M, 3*D) -- projection output [b|c|h] concatenated, fp16
        conv_weight (D, K)   -- depthwise conv weights, fp16
        conv_bias   (D,)     -- depthwise conv bias, fp16
        seq_len     int      -- sequence length T

Output: (M, D) -- GatedConv output, fp16

Compared to 4 separate PyTorch ops, this saves 2 intermediate tensor round-trips
(the `z` tensor between conv and output gate is never materialized).
"""

KERNEL_TYPE = "fused_gated_conv"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Phase 1: Compute y = b * h_tilde (element-wise input gating)
// Input:  proj_out (M, 3*D) — [b | c | h_tilde] concatenated
// Output: y (M, D)
__global__ void gated_conv_phase1_kernel(
    const half* __restrict__ proj_out,   // (M, 3*D)
    half* __restrict__ Y,                // (M, D)
    int M,
    int D
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int three_D = 3 * D;

    const half* proj_row = proj_out + (int64_t)row * three_D;
    // b is at offset 0, h_tilde is at offset 2*D
    const half* b_ptr = proj_row;
    const half* h_ptr = proj_row + 2 * D;
    half* y_row = Y + (int64_t)row * D;

    // Vectorized: 2 channels at a time
    for (int d = tid * 2; d + 1 < D; d += stride * 2) {
        half2 bv = *reinterpret_cast<const half2*>(b_ptr + d);
        half2 hv = *reinterpret_cast<const half2*>(h_ptr + d);
        half2 yv;
        yv.x = __float2half(__half2float(bv.x) * __half2float(hv.x));
        yv.y = __float2half(__half2float(bv.y) * __half2float(hv.y));
        *reinterpret_cast<half2*>(y_row + d) = yv;
    }
    // Odd tail
    int d = (D / 2) * 2 + tid;
    if (d < D && d == D - 1 && tid == 0) {
        y_row[d] = __float2half(__half2float(b_ptr[d]) * __half2float(h_ptr[d]));
    }
}

// Phase 2: Fused depthwise conv1d + output gating
// z = conv(Y) + bias; output = c * z
// Input:  Y (M, D), proj_out (M, 3*D) — c at offset D
// Output: out (M, D)
__global__ void gated_conv_phase2_kernel(
    const half* __restrict__ Y,            // (M, D) — gated input from phase 1
    const half* __restrict__ proj_out,     // (M, 3*D) — for accessing c at offset D
    const half* __restrict__ conv_w,       // (D, K) — depthwise conv weights
    const half* __restrict__ conv_b,       // (D,) — conv bias
    half* __restrict__ OUT,                // (M, D) — output
    int M,
    int D,
    int T,          // seq_len for causal boundary
    int conv_k      // kernel size (3)
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int three_D = 3 * D;

    const half* y_row = Y + (int64_t)row * D;
    // c is at offset D in proj_out
    const half* c_ptr = proj_out + (int64_t)row * three_D + D;
    half* out_row = OUT + (int64_t)row * D;

    int pos_in_seq = row % T;

    // Vectorized: 2 channels at a time
    for (int d = tid * 2; d + 1 < D; d += stride * 2) {
        // Depthwise conv1d (causal)
        float cv0 = __half2float(conv_b[d]);
        float cv1 = __half2float(conv_b[d + 1]);

        for (int j = 0; j < conv_k; j++) {
            int src_pos = pos_in_seq - j;
            if (src_pos >= 0) {
                int src_row = row - j;
                const half* src_y = Y + (int64_t)src_row * D;
                half2 sv = *reinterpret_cast<const half2*>(src_y + d);
                float w0 = __half2float(conv_w[d * conv_k + j]);
                float w1 = __half2float(conv_w[(d + 1) * conv_k + j]);
                cv0 += __half2float(sv.x) * w0;
                cv1 += __half2float(sv.y) * w1;
            }
        }

        // Output gating: output = c * conv_result
        half2 cv = *reinterpret_cast<const half2*>(c_ptr + d);
        half2 result;
        result.x = __float2half(__half2float(cv.x) * cv0);
        result.y = __float2half(__half2float(cv.y) * cv1);
        *reinterpret_cast<half2*>(out_row + d) = result;
    }

    // Handle odd tail
    if (tid == 0 && (D & 1)) {
        int d = D - 1;
        float cv = __half2float(conv_b[d]);
        for (int j = 0; j < conv_k; j++) {
            int src_pos = pos_in_seq - j;
            if (src_pos >= 0) {
                int src_row = row - j;
                cv += __half2float(Y[(int64_t)src_row * D + d]) *
                      __half2float(conv_w[d * conv_k + j]);
            }
        }
        out_row[d] = __float2half(__half2float(c_ptr[d]) * cv);
    }
}

// Launcher: runs both phases
torch::Tensor fused_gated_conv_hip(
    torch::Tensor proj_out,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    int64_t seq_len
) {
    TORCH_CHECK(proj_out.is_cuda(), "proj_out must be a GPU tensor");
    TORCH_CHECK(proj_out.dtype() == torch::kFloat16, "proj_out must be float16");

    int M = proj_out.size(0);          // B*T
    int three_D = proj_out.size(1);    // 3*D
    int D = three_D / 3;
    int T = seq_len;
    int K = conv_weight.size(1);       // conv kernel size

    // Temporary for y = b * h_tilde
    auto y = torch::empty({M, D}, proj_out.options());
    auto out = torch::empty({M, D}, proj_out.options());

    int threads = min(1024, max(32, ((D + 1) / 2 + 31) / 32 * 32));

    // Phase 1: input gating
    gated_conv_phase1_kernel<<<M, threads>>>(
        reinterpret_cast<const half*>(proj_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        M, D
    );

    // Phase 2: conv + output gating
    gated_conv_phase2_kernel<<<M, threads>>>(
        reinterpret_cast<const half*>(y.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(proj_out.data_ptr<at::Half>()),
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
        _module = compile_hip(HIP_SRC, "fused_gated_conv_hip")
    return _module


def kernel_fn(
    proj_out: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Fused GatedConv: gate_mul + depthwise_conv + output_gate.

    Args:
        proj_out: (B*T, 3*D) or (B, T, 3*D) — projection output [b|c|h] concatenated, fp16
        conv_weight: (D, K) — depthwise conv weights, fp16
        conv_bias: (D,) — conv bias, fp16
        seq_len: sequence length T

    Returns:
        (B*T, D) or (B, T, D) — GatedConv output, fp16
    """
    orig_shape = proj_out.shape
    three_D = orig_shape[-1]
    D = three_D // 3

    if proj_out.dim() == 3:
        B, T, _ = orig_shape
        proj_flat = proj_out.reshape(-1, three_D)
    else:
        proj_flat = proj_out
        T = seq_len

    mod = _get_module()
    out = mod.fused_gated_conv_hip(proj_flat.contiguous(), conv_weight, conv_bias, T)

    if len(orig_shape) == 3:
        return out.reshape(B, T, D)
    return out
