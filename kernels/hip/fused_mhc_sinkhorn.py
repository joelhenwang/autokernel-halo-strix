"""
AutoKernel -- HIP C++ Fused mHC (multi-Head Cache) Sinkhorn Kernel.

Fuses the entire mHC routing computation into a single kernel:
  1. Three small projections: phi_pre (d→4), phi_post (d→4), phi_res (d→16)
  2. Sigmoid activations for H_pre, H_post
  3. Exp + Sinkhorn normalization (20 iters) for H_res (4x4 matrix)
  4. All in registers — 4x4 Sinkhorn fits in 16 floats

Per-token computation (one thread per token):
  x_bar = mean(stream, dim=0)         -- mean over 4 branches
  H_pre  = sigmoid(0.01 * (x_bar @ phi_pre) + b_pre)        -- (4,)
  H_post = 2 * sigmoid(0.01 * (x_bar @ phi_post) + b_post)  -- (4,)
  logits = 0.01 * reshape(x_bar @ phi_res + b_res, 4, 4)    -- (4,4)
  H_res  = sinkhorn(exp(logits), iters=20)                   -- (4,4) doubly stochastic

Input:  x_bar       (M, D)   -- mean of 4 branch streams, fp16
        phi_pre     (D, 4)   -- readout projection, fp16
        b_pre       (4,)     -- readout bias, fp16
        phi_post    (D, 4)   -- write-in projection, fp16
        b_post      (4,)     -- write-in bias, fp16
        phi_res     (D, 16)  -- mixing projection, fp16
        b_res       (16,)    -- mixing bias, fp16

Output: H_pre       (M, 4)   -- readout weights, fp32
        H_post      (M, 4)   -- write-in weights, fp32
        H_res       (M, 4, 4) -- mixing matrix (doubly stochastic), fp32
"""

KERNEL_TYPE = "fused_mhc_sinkhorn"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int N_BRANCHES = 4;
constexpr int SINKHORN_ITERS = 20;
constexpr float ALPHA = 0.01f;

// One thread per token — all mHC computation fits in registers
__global__ void fused_mhc_sinkhorn_kernel(
    const half* __restrict__ X_BAR,    // (M, D) mean of branch streams
    const half* __restrict__ PHI_PRE,  // (D, 4) readout projection
    const half* __restrict__ B_PRE,    // (4,) readout bias
    const half* __restrict__ PHI_POST, // (D, 4) write-in projection
    const half* __restrict__ B_POST,   // (4,) write-in bias
    const half* __restrict__ PHI_RES,  // (D, 16) mixing projection
    const half* __restrict__ B_RES,    // (16,) mixing bias
    float* __restrict__ H_PRE_OUT,     // (M, 4) readout weights
    float* __restrict__ H_POST_OUT,    // (M, 4) write-in weights
    float* __restrict__ H_RES_OUT,     // (M, 16) mixing matrix flattened
    int M,    // number of tokens
    int D     // hidden dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    const half* x_row = X_BAR + (int64_t)tid * D;

    // =============================================
    // Phase 1: Three mat-vec products (x_bar @ phi_*)
    // phi_pre: (D, 4), phi_post: (D, 4), phi_res: (D, 16)
    // All small enough to compute serially per thread
    // =============================================

    float pre[N_BRANCHES], post[N_BRANCHES], res[N_BRANCHES * N_BRANCHES];

    // Initialize with biases
    for (int i = 0; i < N_BRANCHES; i++) {
        pre[i] = __half2float(B_PRE[i]);
        post[i] = __half2float(B_POST[i]);
    }
    for (int i = 0; i < N_BRANCHES * N_BRANCHES; i++) {
        res[i] = __half2float(B_RES[i]);
    }

    // Accumulate dot products
    for (int d = 0; d < D; d++) {
        float x = __half2float(x_row[d]);

        for (int i = 0; i < N_BRANCHES; i++) {
            pre[i] += x * __half2float(PHI_PRE[d * N_BRANCHES + i]);
            post[i] += x * __half2float(PHI_POST[d * N_BRANCHES + i]);
        }
        for (int i = 0; i < N_BRANCHES * N_BRANCHES; i++) {
            res[i] += x * __half2float(PHI_RES[d * N_BRANCHES * N_BRANCHES + i]);
        }
    }

    // =============================================
    // Phase 2: Activations
    //   H_pre  = sigmoid(alpha * pre)
    //   H_post = 2 * sigmoid(alpha * post)
    // =============================================

    for (int i = 0; i < N_BRANCHES; i++) {
        pre[i] = 1.0f / (1.0f + __expf(-ALPHA * pre[i]));
        post[i] = 2.0f / (1.0f + __expf(-ALPHA * post[i]));
    }

    // =============================================
    // Phase 3: Sinkhorn normalization of H_res
    //   logits = alpha * res (already scaled)
    //   M = exp(logits)
    //   for 20 iters: row_normalize → col_normalize
    // =============================================

    // Exp of scaled logits
    float sinkhorn[N_BRANCHES * N_BRANCHES];
    for (int i = 0; i < N_BRANCHES * N_BRANCHES; i++) {
        sinkhorn[i] = __expf(ALPHA * res[i]);
    }

    // Sinkhorn iterations (alternating row and column normalization)
    for (int iter = 0; iter < SINKHORN_ITERS; iter++) {
        // Row normalization: each row sums to 1
        for (int r = 0; r < N_BRANCHES; r++) {
            float row_sum = 0.0f;
            for (int c = 0; c < N_BRANCHES; c++) {
                row_sum += sinkhorn[r * N_BRANCHES + c];
            }
            float inv = (row_sum > 1e-8f) ? (1.0f / row_sum) : 0.0f;
            for (int c = 0; c < N_BRANCHES; c++) {
                sinkhorn[r * N_BRANCHES + c] *= inv;
            }
        }

        // Column normalization: each column sums to 1
        for (int c = 0; c < N_BRANCHES; c++) {
            float col_sum = 0.0f;
            for (int r = 0; r < N_BRANCHES; r++) {
                col_sum += sinkhorn[r * N_BRANCHES + c];
            }
            float inv = (col_sum > 1e-8f) ? (1.0f / col_sum) : 0.0f;
            for (int r = 0; r < N_BRANCHES; r++) {
                sinkhorn[r * N_BRANCHES + c] *= inv;
            }
        }
    }

    // =============================================
    // Phase 4: Write outputs
    // =============================================

    for (int i = 0; i < N_BRANCHES; i++) {
        H_PRE_OUT[tid * N_BRANCHES + i] = pre[i];
        H_POST_OUT[tid * N_BRANCHES + i] = post[i];
    }
    for (int i = 0; i < N_BRANCHES * N_BRANCHES; i++) {
        H_RES_OUT[tid * N_BRANCHES * N_BRANCHES + i] = sinkhorn[i];
    }
}

std::vector<torch::Tensor> fused_mhc_sinkhorn_hip(
    torch::Tensor x_bar,
    torch::Tensor phi_pre, torch::Tensor b_pre,
    torch::Tensor phi_post, torch::Tensor b_post,
    torch::Tensor phi_res, torch::Tensor b_res
) {
    TORCH_CHECK(x_bar.is_cuda(), "x_bar must be GPU tensor");

    int M = x_bar.size(0);
    int D = x_bar.size(1);

    auto opts_f32 = torch::TensorOptions().device(x_bar.device()).dtype(torch::kFloat32);
    auto h_pre = torch::empty({M, 4}, opts_f32);
    auto h_post = torch::empty({M, 4}, opts_f32);
    auto h_res = torch::empty({M, 16}, opts_f32);

    int threads = 256;
    int blocks = (M + threads - 1) / threads;

    fused_mhc_sinkhorn_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(x_bar.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(phi_pre.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b_pre.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(phi_post.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b_post.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(phi_res.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b_res.data_ptr<at::Half>()),
        h_pre.data_ptr<float>(),
        h_post.data_ptr<float>(),
        h_res.data_ptr<float>(),
        M, D
    );

    return {h_pre, h_post, h_res.view({M, 4, 4})};
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        cpp_src = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> fused_mhc_sinkhorn_hip(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
"""
        _module = compile_hip(HIP_SRC, "fused_mhc_sinkhorn_hip", cpp_src=cpp_src)
    return _module


def kernel_fn(
    x_bar: torch.Tensor,
    phi_pre: torch.Tensor, b_pre: torch.Tensor,
    phi_post: torch.Tensor, b_post: torch.Tensor,
    phi_res: torch.Tensor, b_res: torch.Tensor,
) -> tuple:
    """Fused mHC Sinkhorn: projections + activations + 20-iter Sinkhorn.

    Args:
        x_bar: (M, D) mean of 4 branch streams, fp16
        phi_pre: (D, 4) readout projection, fp16
        b_pre: (4,) readout bias, fp16
        phi_post: (D, 4) write-in projection, fp16
        b_post: (4,) write-in bias, fp16
        phi_res: (D, 16) mixing projection, fp16
        b_res: (16,) mixing bias, fp16

    Returns:
        (H_pre, H_post, H_res): (M,4), (M,4), (M,4,4) all fp32
    """
    assert x_bar.is_cuda

    if x_bar.dtype != torch.float16:
        return reference_fn(x_bar, phi_pre, b_pre, phi_post, b_post, phi_res, b_res)

    x_2d = x_bar.reshape(-1, x_bar.shape[-1]).contiguous()

    mod = _get_module()
    results = mod.fused_mhc_sinkhorn_hip(
        x_2d,
        phi_pre.contiguous(), b_pre.contiguous(),
        phi_post.contiguous(), b_post.contiguous(),
        phi_res.contiguous(), b_res.contiguous(),
    )
    return results[0], results[1], results[2]


def reference_fn(x_bar, phi_pre, b_pre, phi_post, b_post, phi_res, b_res):
    """Pure PyTorch reference."""
    ALPHA = 0.01
    N = 4

    x = x_bar.float().reshape(-1, x_bar.shape[-1])

    # Projections
    pre_logits = x @ phi_pre.float() + b_pre.float()
    post_logits = x @ phi_post.float() + b_post.float()
    res_logits = x @ phi_res.float() + b_res.float()

    # Activations
    H_pre = torch.sigmoid(ALPHA * pre_logits)
    H_post = 2.0 * torch.sigmoid(ALPHA * post_logits)

    # Sinkhorn
    M_mat = torch.exp(ALPHA * res_logits.reshape(-1, N, N))
    for _ in range(20):
        M_mat = M_mat / M_mat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        M_mat = M_mat / M_mat.sum(dim=-2, keepdim=True).clamp(min=1e-8)
    H_res = M_mat

    return H_pre, H_post, H_res
