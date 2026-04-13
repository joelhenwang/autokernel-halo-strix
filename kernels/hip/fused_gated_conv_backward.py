"""
AutoKernel -- HIP Fused GatedConv Backward kernel.

Fuses the entire GatedConv backward into a single kernel:
  Forward: output = c * conv(b * h_tilde) where [b,c,h] = proj_out.chunk(3)
  Backward: recomputes y=b*h and z=conv(y) from saved proj_out, then computes
  all gradients in one pass per position.

Eliminates 4 intermediate tensors vs PyTorch autograd's default chain.

Input:  proj_out    (M, 3*D) fp16 -- saved from forward [b|c|h_tilde]
        conv_weight (D, K)   fp16 -- depthwise conv weights
        conv_bias   (D,)     fp16 -- conv bias
        grad_output (M, D)   fp16 -- upstream gradient

Output: grad_proj_out    (M, 3*D) fp16  -- [grad_b|grad_c|grad_h]
        grad_conv_weight (D, K)   fp32  -- atomic accumulated
        grad_conv_bias   (D,)     fp32  -- atomic accumulated
"""

KERNEL_TYPE = "fused_gated_conv_backward"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Fused GatedConv backward kernel
// One block per position (row in flattened B*T)
// Each thread handles 2 channels (half2 vectorized)
__global__ void gated_conv_backward_kernel(
    const half* __restrict__ PROJ,       // (M, 3*D) [b|c|h] concatenated
    const half* __restrict__ CONV_W,     // (D, K) depthwise conv weights
    const half* __restrict__ CONV_B,     // (D,) conv bias
    const half* __restrict__ GRAD_OUT,   // (M, D) upstream gradient
    half* __restrict__ GRAD_PROJ,        // (M, 3*D) output [grad_b|grad_c|grad_h]
    float* __restrict__ GRAD_CW,        // (D, K) conv weight gradient (fp32 atomics)
    float* __restrict__ GRAD_CB,        // (D,) conv bias gradient (fp32 atomics)
    int M, int D, int T, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int three_D = 3 * D;
    const int pos_in_seq = row % T;

    const half* proj_row = PROJ + (int64_t)row * three_D;
    const half* go_row = GRAD_OUT + (int64_t)row * D;
    half* gp_row = GRAD_PROJ + (int64_t)row * three_D;

    // Process 2 channels at a time (half2 vectorized)
    for (int d = tid * 2; d + 1 < D; d += stride * 2) {
        // --- Load current position data ---
        half2 bv = *reinterpret_cast<const half2*>(proj_row + d);
        half2 cv = *reinterpret_cast<const half2*>(proj_row + D + d);
        half2 hv = *reinterpret_cast<const half2*>(proj_row + 2 * D + d);
        half2 gov = *reinterpret_cast<const half2*>(go_row + d);

        float b0 = __half2float(bv.x), b1 = __half2float(bv.y);
        float c0 = __half2float(cv.x), c1 = __half2float(cv.y);
        float h0 = __half2float(hv.x), h1 = __half2float(hv.y);
        float go0 = __half2float(gov.x), go1 = __half2float(gov.y);

        // --- Recompute z = conv(b*h) + bias ---
        float z0 = __half2float(CONV_B[d]);
        float z1 = __half2float(CONV_B[d + 1]);
        for (int j = 0; j < K; j++) {
            int src_pos = pos_in_seq - j;
            if (src_pos >= 0) {
                int src_row = row - j;
                const half* src_proj = PROJ + (int64_t)src_row * three_D;
                // y_prev = b_prev * h_prev
                float bp0 = __half2float(src_proj[d]);
                float bp1 = __half2float(src_proj[d + 1]);
                float hp0 = __half2float(src_proj[2 * D + d]);
                float hp1 = __half2float(src_proj[2 * D + d + 1]);
                float w0 = __half2float(CONV_W[d * K + j]);
                float w1 = __half2float(CONV_W[(d + 1) * K + j]);
                z0 += (bp0 * hp0) * w0;
                z1 += (bp1 * hp1) * w1;
            }
        }

        // --- Backward through output gate: output = c * z ---
        float grad_c0 = go0 * z0;
        float grad_c1 = go1 * z1;
        float grad_z0 = go0 * c0;
        float grad_z1 = go1 * c1;

        // --- Backward through conv (transpose: look FORWARD) ---
        float grad_y0 = 0.0f, grad_y1 = 0.0f;
        for (int j = 0; j < K; j++) {
            int dst_pos = pos_in_seq + j;
            if (dst_pos < T) {
                int dst_row = row + j;
                if (dst_row < M) {
                    // grad_z at future position = grad_output[dst] * c[dst]
                    float go_f0 = __half2float(GRAD_OUT[(int64_t)dst_row * D + d]);
                    float go_f1 = __half2float(GRAD_OUT[(int64_t)dst_row * D + d + 1]);
                    float c_f0 = __half2float(PROJ[(int64_t)dst_row * three_D + D + d]);
                    float c_f1 = __half2float(PROJ[(int64_t)dst_row * three_D + D + d + 1]);
                    float w0 = __half2float(CONV_W[d * K + j]);
                    float w1 = __half2float(CONV_W[(d + 1) * K + j]);
                    grad_y0 += w0 * (go_f0 * c_f0);
                    grad_y1 += w1 * (go_f1 * c_f1);
                }
            }
        }

        // --- Backward through input gate: y = b * h ---
        float grad_b0 = grad_y0 * h0;
        float grad_b1 = grad_y1 * h1;
        float grad_h0 = grad_y0 * b0;
        float grad_h1 = grad_y1 * b1;

        // --- Write grad_proj_out = [grad_b | grad_c | grad_h] ---
        half2 gb_out, gc_out, gh_out;
        gb_out.x = __float2half(grad_b0); gb_out.y = __float2half(grad_b1);
        gc_out.x = __float2half(grad_c0); gc_out.y = __float2half(grad_c1);
        gh_out.x = __float2half(grad_h0); gh_out.y = __float2half(grad_h1);
        *reinterpret_cast<half2*>(gp_row + d) = gb_out;
        *reinterpret_cast<half2*>(gp_row + D + d) = gc_out;
        *reinterpret_cast<half2*>(gp_row + 2 * D + d) = gh_out;

        // --- Atomic accumulate conv weight/bias gradients ---
        for (int j = 0; j < K; j++) {
            int src_pos = pos_in_seq - j;
            if (src_pos >= 0) {
                int src_row = row - j;
                const half* src_proj = PROJ + (int64_t)src_row * three_D;
                float yp0 = __half2float(src_proj[d]) * __half2float(src_proj[2 * D + d]);
                float yp1 = __half2float(src_proj[d + 1]) * __half2float(src_proj[2 * D + d + 1]);
                atomicAdd(&GRAD_CW[d * K + j], yp0 * grad_z0);
                atomicAdd(&GRAD_CW[(d + 1) * K + j], yp1 * grad_z1);
            }
        }
        atomicAdd(&GRAD_CB[d], grad_z0);
        atomicAdd(&GRAD_CB[d + 1], grad_z1);
    }

    // Handle odd tail
    if (tid == 0 && (D & 1)) {
        int d = D - 1;
        float b_val = __half2float(proj_row[d]);
        float c_val = __half2float(proj_row[D + d]);
        float h_val = __half2float(proj_row[2 * D + d]);
        float go_val = __half2float(go_row[d]);

        // Recompute z
        float z_val = __half2float(CONV_B[d]);
        for (int j = 0; j < K; j++) {
            int src_pos = pos_in_seq - j;
            if (src_pos >= 0) {
                int src_row = row - j;
                float bp = __half2float(PROJ[(int64_t)src_row * three_D + d]);
                float hp = __half2float(PROJ[(int64_t)src_row * three_D + 2 * D + d]);
                z_val += (bp * hp) * __half2float(CONV_W[d * K + j]);
            }
        }

        float gc = go_val * z_val;
        float gz = go_val * c_val;

        // Conv transpose
        float gy = 0.0f;
        for (int j = 0; j < K; j++) {
            int dst_pos = pos_in_seq + j;
            if (dst_pos < T && row + j < M) {
                float go_f = __half2float(GRAD_OUT[(int64_t)(row + j) * D + d]);
                float c_f = __half2float(PROJ[(int64_t)(row + j) * three_D + D + d]);
                gy += __half2float(CONV_W[d * K + j]) * (go_f * c_f);
            }
        }

        gp_row[d] = __float2half(gy * h_val);         // grad_b
        gp_row[D + d] = __float2half(gc);              // grad_c
        gp_row[2 * D + d] = __float2half(gy * b_val);  // grad_h

        for (int j = 0; j < K; j++) {
            int src_pos = pos_in_seq - j;
            if (src_pos >= 0) {
                int sr = row - j;
                float yp = __half2float(PROJ[(int64_t)sr * three_D + d]) *
                           __half2float(PROJ[(int64_t)sr * three_D + 2 * D + d]);
                atomicAdd(&GRAD_CW[d * K + j], yp * gz);
            }
        }
        atomicAdd(&GRAD_CB[d], gz);
    }
}

std::vector<torch::Tensor> fused_gated_conv_backward_hip(
    torch::Tensor proj_out,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor grad_output,
    int64_t seq_len
) {
    TORCH_CHECK(proj_out.is_cuda(), "proj_out must be GPU");
    TORCH_CHECK(proj_out.dtype() == torch::kFloat16, "proj_out must be fp16");

    int M = proj_out.size(0);
    int three_D = proj_out.size(1);
    int D = three_D / 3;
    int K = conv_weight.size(1);
    int T = seq_len;

    auto grad_proj = torch::empty_like(proj_out);
    auto grad_cw = torch::zeros({D, K}, torch::dtype(torch::kFloat32).device(proj_out.device()));
    auto grad_cb = torch::zeros({D}, torch::dtype(torch::kFloat32).device(proj_out.device()));

    int threads = min(512, max(32, ((D + 1) / 2 + 31) / 32 * 32));

    gated_conv_backward_kernel<<<M, threads>>>(
        reinterpret_cast<const half*>(proj_out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(conv_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(conv_bias.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(grad_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(grad_proj.data_ptr<at::Half>()),
        grad_cw.data_ptr<float>(),
        grad_cb.data_ptr<float>(),
        M, D, T, K
    );

    return {grad_proj, grad_cw, grad_cb};
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        cpp_src = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> fused_gated_conv_backward_hip(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
"""
        _module = compile_hip(HIP_SRC, "fused_gated_conv_backward_hip", cpp_src=cpp_src)
    return _module


def kernel_fn(
    proj_out: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    grad_output: torch.Tensor,
    seq_len: int,
) -> tuple:
    """Fused GatedConv backward.

    Args:
        proj_out: (M, 3*D) or (B, T, 3*D) — saved [b|c|h] from forward, fp16
        conv_weight: (D, K) — depthwise conv weights, fp16
        conv_bias: (D,) — conv bias, fp16
        grad_output: (M, D) or (B, T, D) — upstream gradient, fp16
        seq_len: int — sequence length T

    Returns:
        (grad_proj_out, grad_conv_weight, grad_conv_bias)
    """
    orig_shape = proj_out.shape
    three_D = orig_shape[-1]
    D = three_D // 3

    proj_flat = proj_out.contiguous().view(-1, three_D)
    grad_flat = grad_output.contiguous().view(-1, D)
    cw = conv_weight.contiguous()
    cb = conv_bias.contiguous()

    if cw.dtype != torch.float16:
        cw = cw.to(torch.float16)
    if cb.dtype != torch.float16:
        cb = cb.to(torch.float16)

    mod = _get_module()
    results = mod.fused_gated_conv_backward_hip(proj_flat, cw, cb, grad_flat, seq_len)

    grad_proj = results[0].view(orig_shape)
    grad_cw = results[1]  # fp32
    grad_cb = results[2]  # fp32

    return grad_proj, grad_cw, grad_cb
