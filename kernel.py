"""
AutoKernel -- Optimized cross_entropy kernel (v4: online softmax, fused mode default).

Op type: cross_entropy
Model shape: batch=4096, vocab=32768

Design:
  * Single-pass online softmax (Milakov & Gimelshein 2018) -- max and sum
    computed jointly via associative combine: (m_a,s_a) ⊕ (m_b,s_b) =
    (max(m_a,m_b), s_a·exp(m_a-m)+s_b·exp(m_b-m)).

  * Forward emits per-row: loss, logsumexp (z-loss side-output), grad_logits
    (fused mode) OR row_max+row_sum (tiny mode).

  * Fused mode (default): backward is a no-op identity (just returns saved
    grad_logits * grad_output). Trades 67 MB extra saved tensor @ prod for
    saving one full-pass backward kernel.

  * Tiny mode: saves only row_max+row_sum; backward recomputes softmax.

  * Features baked into kernel (runtime branches, invariant-hoisted):
    - logit_softcap: softcap*tanh(x/softcap) applied in-register before online
      update; backward folds (1 - tanh²(x/softcap)) into grad.
    - z_loss side output: per-row logsumexp emitted free (we already compute it).
    - ignore_index (default -100): rows with target==ignore zero grad + loss.
    - label_smoothing α: loss = (1-α)(LSE - x_y) + α(LSE - mean(x)); grad adds
      -α/V uniform term.
"""

KERNEL_TYPE = "cross_entropy"
BACKEND = "hip"

MODEL_SHAPES = {'batch': 4096, 'vocab': 32768}

TEST_SIZES = [
    ("model_primary", {'batch': 4096, 'vocab': 32768}),
    ("model_half", {'batch': 2048, 'vocab': 16384}),
    ("model_double", {'batch': 8192, 'vocab': 65536}),
]

TOLERANCES = {'float16': {'atol': 0.01, 'rtol': 0.01}, 'bfloat16': {'atol': 0.02, 'rtol': 0.02}, 'float32': {'atol': 1e-05, 'rtol': 1e-05}}


def FLOPS_FN(s):
    return 4 * s["batch"] * s["vocab"]


def BYTES_FN(s, dt_bytes):
    return (s["batch"] * s["vocab"] + s["batch"]) * dt_bytes


import torch
from kernels.hip._compile import compile_hip

# =============================================================================
# Forward kernel: online softmax + optional fused grad_logits writeback
# =============================================================================
HIP_FWD_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>
#include <math.h>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// Device-side tanh implemented via exp (tanhf stdlib not available in device code on gfx1151).
// tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x)) for x >= 0
//         = -tanh(-x)                        for x < 0
// Saturates cleanly and is numerically stable for large |x|.
__device__ __forceinline__ float dev_tanhf(float x) {
    float ax = (x < 0.0f) ? -x : x;
    float e = __expf(-2.0f * ax);
    float r = (1.0f - e) / (1.0f + e);
    return (x < 0.0f) ? -r : r;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor(val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

// Online softmax combine: given thread-local (m, s) where s is already
// normalized against local m, combine across warp using max-then-rescale.
__device__ __forceinline__ void warp_reduce_online(float& m, float& s) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float m_other = __shfl_xor(m, offset);
        float s_other = __shfl_xor(s, offset);
        float m_new = fmaxf(m, m_other);
        // Guard against -inf overflow: if m_new == -inf both sides are -inf.
        float factor_self  = (m_new == -FLT_MAX) ? 0.0f : __expf(m - m_new);
        float factor_other = (m_new == -FLT_MAX) ? 0.0f : __expf(m_other - m_new);
        s = s * factor_self + s_other * factor_other;
        m = m_new;
    }
}

// Forward kernel. One block per row. Online softmax in single pass over logits.
// Writes:
//   - losses[b]       : per-row loss (scalar)
//   - logsumexp_out[b]: per-row LSE (used for z_loss, always emitted)
//   - row_maxes_out[b], row_sums_out[b]: kept for tiny-mode backward (optional)
//   - grad_logits[b,*]: if write_grad != 0, emit softmax-based grad (fused mode)
//
// Handles: logit_softcap, ignore_index, label_smoothing (all runtime).
// scale_grad is the per-row factor applied to grad (typically 1/N_valid for mean).
__global__ void __launch_bounds__(BLOCK_SIZE)
cross_entropy_fwd_kernel(
    const half* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    float* __restrict__ logsumexp_out,
    float* __restrict__ row_maxes_out,
    float* __restrict__ row_sums_out,
    half* __restrict__ grad_logits,
    int B, int vocab,
    float softcap,
    int ignore_index,
    float label_smoothing,
    float scale_grad,
    int write_grad
) {
    const int b = blockIdx.x;
    if (b >= B) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const half* row = logits + (long long)b * vocab;
    const int target = (int)targets[b];
    const bool is_ignored = (target == ignore_index);

    // Precompute invariants outside loop
    const bool use_softcap = (softcap > 0.0f);
    const float inv_softcap = use_softcap ? (1.0f / softcap) : 0.0f;
    const bool use_ls = (label_smoothing > 0.0f);
    const float ls_alpha = label_smoothing;
    const float ls_one_minus = 1.0f - label_smoothing;
    const float ls_uniform = label_smoothing / (float)vocab;

    // ---- Phase 1: online softmax over logits (reads once) ----
    // Also accumulate sum_logits for label smoothing (uniform KL term).
    float local_m = -FLT_MAX;
    float local_s = 0.0f;
    float local_sum_x = 0.0f;  // only used if use_ls

    // uint4 loads = 8 half values per transaction
    const uint4* row_u4 = reinterpret_cast<const uint4*>(row);
    const int n_vec8 = vocab / 8;

    #pragma unroll 2
    for (int v = tid; v < n_vec8; v += BLOCK_SIZE) {
        uint4 pack = row_u4[v];
        const half* h = reinterpret_cast<const half*>(&pack);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float x = __half2float(h[k]);
            if (use_softcap) {
                x = softcap * dev_tanhf(x * inv_softcap);
            }
            if (use_ls) local_sum_x += x;
            // online update
            float m_new = fmaxf(local_m, x);
            float factor = (local_m == -FLT_MAX) ? 0.0f : __expf(local_m - m_new);
            local_s = local_s * factor + __expf(x - m_new);
            local_m = m_new;
        }
    }
    // Tail: vocab % 8
    const int tail_start = n_vec8 * 8;
    for (int i = tail_start + tid; i < vocab; i += BLOCK_SIZE) {
        float x = __half2float(row[i]);
        if (use_softcap) {
            x = softcap * dev_tanhf(x * inv_softcap);
        }
        if (use_ls) local_sum_x += x;
        float m_new = fmaxf(local_m, x);
        float factor = (local_m == -FLT_MAX) ? 0.0f : __expf(local_m - m_new);
        local_s = local_s * factor + __expf(x - m_new);
        local_m = m_new;
    }

    // ---- Warp-level online reduce ----
    warp_reduce_online(local_m, local_s);
    // sum_x is a regular sum
    float warp_sum_x = use_ls ? warp_reduce_sum(local_sum_x) : 0.0f;

    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_s[NUM_WARPS];
    __shared__ float smem_sx[NUM_WARPS];

    if (lane_id == 0) {
        smem_m[warp_id] = local_m;
        smem_s[warp_id] = local_s;
        if (use_ls) smem_sx[warp_id] = warp_sum_x;
    }
    __syncthreads();

    // ---- Block-level reduce (warp 0) ----
    if (warp_id == 0) {
        float m = (lane_id < NUM_WARPS) ? smem_m[lane_id] : -FLT_MAX;
        float s = (lane_id < NUM_WARPS) ? smem_s[lane_id] : 0.0f;
        warp_reduce_online(m, s);
        float sx = use_ls ?
            ((lane_id < NUM_WARPS) ? smem_sx[lane_id] : 0.0f) : 0.0f;
        if (use_ls) sx = warp_reduce_sum(sx);
        if (lane_id == 0) {
            smem_m[0] = m;
            smem_s[0] = s;
            if (use_ls) smem_sx[0] = sx;
        }
    }
    __syncthreads();

    const float row_max = smem_m[0];
    const float row_sum = smem_s[0];
    const float row_sum_x = use_ls ? smem_sx[0] : 0.0f;
    const float lse = row_max + __logf(row_sum);  // logsumexp

    // ---- Phase 2: write loss, LSE, (optional) row_max/row_sum ----
    if (tid == 0) {
        logsumexp_out[b] = lse;
        if (row_maxes_out != nullptr) row_maxes_out[b] = row_max;
        if (row_sums_out  != nullptr) row_sums_out[b]  = row_sum;

        if (is_ignored) {
            losses[b] = 0.0f;
        } else {
            // Fetch target logit (with softcap if enabled)
            float x_target = __half2float(row[target]);
            if (use_softcap) x_target = softcap * dev_tanhf(x_target * inv_softcap);

            // loss = lse - x_target  (standard CE, scaled per-row by 1)
            // with label smoothing: loss = lse - (1-α)*x_target - α*mean(x)
            float loss;
            if (use_ls) {
                float mean_x = row_sum_x / (float)vocab;
                loss = lse - ls_one_minus * x_target - ls_alpha * mean_x;
            } else {
                loss = lse - x_target;
            }
            losses[b] = loss;
        }
    }

    // ---- Phase 3 (fused mode only): write grad_logits in second pass ----
    // grad_i = (softmax(x)_i - one_hot_i) * scale_grad  [basic]
    // with ls: grad_i = ((1-α)*softmax - (1-α)*[i==y] + α*(softmax - 1/V)) * scale_grad
    //         = (softmax - (1-α)*[i==y] - α/V) * scale_grad
    // with softcap: multiply by (1 - tanh²(x_orig/softcap)) on top
    // ignore_index: grad = 0 for entire row
    if (write_grad && grad_logits != nullptr) {
        half* grad_row = grad_logits + (long long)b * vocab;
        if (is_ignored) {
            // Zero the entire row
            uint4* grad_u4 = reinterpret_cast<uint4*>(grad_row);
            const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
            for (int v = tid; v < n_vec8; v += BLOCK_SIZE) {
                grad_u4[v] = zero;
            }
            for (int i = tail_start + tid; i < vocab; i += BLOCK_SIZE) {
                grad_row[i] = __float2half(0.0f);
            }
            return;
        }

        const float inv_row_sum = 1.0f / row_sum;
        // scale with label smoothing baked in
        const float target_coeff = use_ls ? ls_one_minus : 1.0f;

        uint4* grad_u4 = reinterpret_cast<uint4*>(grad_row);
        for (int v = tid; v < n_vec8; v += BLOCK_SIZE) {
            uint4 pack = row_u4[v];
            const half* hin = reinterpret_cast<const half*>(&pack);
            const int base_idx = v * 8;

            uint4 out;
            half* hout = reinterpret_cast<half*>(&out);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float x_orig = __half2float(hin[k]);
                float x = use_softcap ? (softcap * dev_tanhf(x_orig * inv_softcap)) : x_orig;
                float p = __expf(x - row_max) * inv_row_sum;
                int idx = base_idx + k;
                float g = p;
                if (idx == target) g -= target_coeff;
                if (use_ls) g -= ls_uniform;
                if (use_softcap) {
                    // d/dx [softcap*tanh(x/softcap)] = 1 - tanh²(x_orig/softcap)
                    float t = x * inv_softcap;  // == tanh(x_orig/softcap)
                    g *= (1.0f - t * t);
                }
                g *= scale_grad;
                hout[k] = __float2half(g);
            }
            grad_u4[v] = out;
        }
        for (int i = tail_start + tid; i < vocab; i += BLOCK_SIZE) {
            float x_orig = __half2float(row[i]);
            float x = use_softcap ? (softcap * dev_tanhf(x_orig * inv_softcap)) : x_orig;
            float p = __expf(x - row_max) * inv_row_sum;
            float g = p;
            if (i == target) g -= target_coeff;
            if (use_ls) g -= ls_uniform;
            if (use_softcap) {
                float t = x * inv_softcap;
                g *= (1.0f - t * t);
            }
            g *= scale_grad;
            grad_row[i] = __float2half(g);
        }
    }
}

// Host entry. Accepts scale_grad from caller (= 1/N_valid) so forward writes
// pre-scaled grad, and backward can return saved tensor directly (no multiply pass).
std::vector<torch::Tensor> cross_entropy_fwd_hip(
    torch::Tensor logits,
    torch::Tensor targets,
    double softcap,
    int64_t ignore_index,
    double label_smoothing,
    int64_t write_grad,
    int64_t save_max_sum,
    double scale_grad_in
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be on GPU");
    TORCH_CHECK(targets.is_cuda(), "targets must be on GPU");
    TORCH_CHECK(logits.dim() == 2, "logits must be [batch, vocab]");
    TORCH_CHECK(logits.dtype() == torch::kFloat16, "logits must be fp16");

    int B = logits.size(0);
    int vocab = logits.size(1);

    auto losses     = torch::empty({B}, logits.options().dtype(torch::kFloat32));
    auto logsumexp  = torch::empty({B}, logits.options().dtype(torch::kFloat32));
    torch::Tensor row_max = save_max_sum ?
        torch::empty({B}, logits.options().dtype(torch::kFloat32)) : torch::Tensor();
    torch::Tensor row_sum = save_max_sum ?
        torch::empty({B}, logits.options().dtype(torch::kFloat32)) : torch::Tensor();
    torch::Tensor grad_logits = write_grad ?
        torch::empty_like(logits) : torch::Tensor();

    float scale_grad = (float)scale_grad_in;

    dim3 grid(B);
    dim3 block(256);

    cross_entropy_fwd_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(logits.data_ptr<at::Half>()),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        logsumexp.data_ptr<float>(),
        save_max_sum ? row_max.data_ptr<float>() : nullptr,
        save_max_sum ? row_sum.data_ptr<float>() : nullptr,
        write_grad   ? reinterpret_cast<half*>(grad_logits.data_ptr<at::Half>()) : nullptr,
        B, vocab,
        (float)softcap,
        (int)ignore_index,
        (float)label_smoothing,
        scale_grad,
        (int)write_grad
    );

    return {losses, logsumexp, row_max, row_sum, grad_logits};
}
"""

# =============================================================================
# Tiny-mode backward kernel (saves only row_max+row_sum; used when memory tight)
# =============================================================================
HIP_BWD_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>
#include <math.h>

constexpr int BWD_BLOCK_SIZE = 256;

// Device-side tanh (tanhf stdlib not available in device code on gfx1151).
__device__ __forceinline__ float dev_tanhf(float x) {
    float ax = (x < 0.0f) ? -x : x;
    float e = __expf(-2.0f * ax);
    float r = (1.0f - e) / (1.0f + e);
    return (x < 0.0f) ? -r : r;
}

__global__ void __launch_bounds__(BWD_BLOCK_SIZE)
cross_entropy_bwd_tiny_kernel(
    const half* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ row_maxes,
    const float* __restrict__ row_sums,
    half* __restrict__ grad_logits,
    int B, int vocab,
    float softcap,
    int ignore_index,
    float label_smoothing,
    float scale_grad
) {
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;

    const half* row = logits + (long long)b * vocab;
    half* grad_row = grad_logits + (long long)b * vocab;
    const int target = (int)targets[b];
    const bool is_ignored = (target == ignore_index);
    const float row_max = row_maxes[b];
    const float inv_row_sum = 1.0f / row_sums[b];
    const bool use_softcap = (softcap > 0.0f);
    const float inv_softcap = use_softcap ? 1.0f / softcap : 0.0f;
    const bool use_ls = (label_smoothing > 0.0f);
    const float ls_uniform = label_smoothing / (float)vocab;
    const float target_coeff = use_ls ? (1.0f - label_smoothing) : 1.0f;

    const uint4* row_u4 = reinterpret_cast<const uint4*>(row);
    uint4* grad_u4 = reinterpret_cast<uint4*>(grad_row);
    const int n_vec8 = vocab / 8;

    if (is_ignored) {
        const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
        for (int v = tid; v < n_vec8; v += BWD_BLOCK_SIZE) grad_u4[v] = zero;
        const int tail_start = n_vec8 * 8;
        for (int i = tail_start + tid; i < vocab; i += BWD_BLOCK_SIZE)
            grad_row[i] = __float2half(0.0f);
        return;
    }

    #pragma unroll 2
    for (int v = tid; v < n_vec8; v += BWD_BLOCK_SIZE) {
        uint4 pack = row_u4[v];
        const half* hin = reinterpret_cast<const half*>(&pack);
        const int base_idx = v * 8;

        uint4 out;
        half* hout = reinterpret_cast<half*>(&out);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float x_orig = __half2float(hin[k]);
            float x = use_softcap ? (softcap * dev_tanhf(x_orig * inv_softcap)) : x_orig;
            float p = __expf(x - row_max) * inv_row_sum;
            int idx = base_idx + k;
            float g = p;
            if (idx == target) g -= target_coeff;
            if (use_ls) g -= ls_uniform;
            if (use_softcap) {
                float t = x * inv_softcap;
                g *= (1.0f - t * t);
            }
            g *= scale_grad;
            hout[k] = __float2half(g);
        }
        grad_u4[v] = out;
    }
    const int tail_start = n_vec8 * 8;
    for (int i = tail_start + tid; i < vocab; i += BWD_BLOCK_SIZE) {
        float x_orig = __half2float(row[i]);
        float x = use_softcap ? (softcap * dev_tanhf(x_orig * inv_softcap)) : x_orig;
        float p = __expf(x - row_max) * inv_row_sum;
        float g = p;
        if (i == target) g -= target_coeff;
        if (use_ls) g -= ls_uniform;
        if (use_softcap) {
            float t = x * inv_softcap;
            g *= (1.0f - t * t);
        }
        g *= scale_grad;
        grad_row[i] = __float2half(g);
    }
}

torch::Tensor cross_entropy_bwd_tiny_hip(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor row_maxes,
    torch::Tensor row_sums,
    double softcap,
    int64_t ignore_index,
    double label_smoothing,
    double scale_grad
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
    int B = logits.size(0);
    int vocab = logits.size(1);
    auto grad_logits = torch::empty_like(logits);

    dim3 grid(B);
    dim3 block(256);

    cross_entropy_bwd_tiny_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(logits.data_ptr<at::Half>()),
        targets.data_ptr<int64_t>(),
        row_maxes.data_ptr<float>(),
        row_sums.data_ptr<float>(),
        reinterpret_cast<half*>(grad_logits.data_ptr<at::Half>()),
        B, vocab,
        (float)softcap,
        (int)ignore_index,
        (float)label_smoothing,
        (float)scale_grad
    );

    return grad_logits;
}
"""

_fwd_module = None
_bwd_module = None


def _get_fwd_module():
    global _fwd_module
    if _fwd_module is None:
        cpp_src = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> cross_entropy_fwd_hip(
    torch::Tensor, torch::Tensor, double, int64_t, double, int64_t, int64_t, double);
"""
        _fwd_module = compile_hip(HIP_FWD_SRC, "cross_entropy_fwd_hip",
                                  cpp_src=cpp_src)
    return _fwd_module


def _get_bwd_module():
    global _bwd_module
    if _bwd_module is None:
        cpp_src = r"""
#include <torch/extension.h>
torch::Tensor cross_entropy_bwd_tiny_hip(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    double, int64_t, double, double);
"""
        _bwd_module = compile_hip(HIP_BWD_SRC, "cross_entropy_bwd_tiny_hip",
                                  cpp_src=cpp_src)
    return _bwd_module


class _CrossEntropyHIP(torch.autograd.Function):
    """Custom autograd Function for CE (online softmax + optional fused grad).

    Args:
        logits: [B, V] fp16
        targets: [B] int64
        softcap: logit softcap (0 = off)
        ignore_index: target val to mask (default -100)
        label_smoothing: alpha (0 = off)
        mode: "fused" (default) or "tiny". fused saves grad_logits in forward
            (backward is ~no-op); tiny saves only row_max+sum (lower memory).
        return_z: if True, also return per-row logsumexp for z-loss.

    Returns:
        loss scalar (mean over non-ignored rows).
        If return_z: (loss, logsumexp_per_row).
    """

    @staticmethod
    def forward(ctx, logits, targets, softcap, ignore_index,
                label_smoothing, mode, return_z):
        mod = _get_fwd_module()
        B = logits.size(0)
        write_grad = 1 if mode == "fused" else 0
        save_max_sum = 1 if mode == "tiny" else 0

        # Compute n_valid (accounting for ignore_index). For fused mode we pre-scale
        # grad_logits in the kernel so backward doesn't need a multiply pass.
        if ignore_index is not None:
            valid_mask = (targets != ignore_index)
            n_valid_t = valid_mask.sum().to(torch.float32).clamp_(min=1.0)
        else:
            valid_mask = None
            n_valid_t = torch.tensor(float(B), device=logits.device, dtype=torch.float32)

        # For fused mode, we need scale_grad as a Python float (passed as double to kernel).
        # This .item() forces a sync but only when fused mode is actually used.
        if write_grad:
            n_valid_scalar = n_valid_t.item()
            scale_grad = 1.0 / n_valid_scalar
        else:
            n_valid_scalar = float(B)  # will be computed via tensor for loss mean
            scale_grad = 1.0  # unused in tiny mode

        losses, lse, row_max, row_sum, grad_logits = mod.cross_entropy_fwd_hip(
            logits, targets,
            float(softcap), int(ignore_index),
            float(label_smoothing),
            int(write_grad), int(save_max_sum),
            float(scale_grad),
        )

        # Scalar loss = mean over non-ignored rows
        if valid_mask is not None:
            loss = (losses * valid_mask.to(torch.float32)).sum() / n_valid_t
        else:
            loss = losses.mean()

        # Save for backward
        ctx.mode = mode
        ctx.softcap = float(softcap)
        ctx.ignore_index = int(ignore_index)
        ctx.label_smoothing = float(label_smoothing)
        ctx.return_z = bool(return_z)
        ctx.n_valid_item = n_valid_scalar
        ctx.scale_grad = scale_grad  # already baked into saved grad in fused mode

        if mode == "fused":
            ctx.save_for_backward(grad_logits)
            ctx.has_tiny = False
        else:
            ctx.save_for_backward(logits, targets, row_max, row_sum)
            ctx.has_tiny = True

        if return_z:
            return loss, lse
        return loss

    @staticmethod
    def backward(ctx, grad_loss, *rest):
        if ctx.has_tiny:
            logits, targets, row_max, row_sum = ctx.saved_tensors
            mod = _get_bwd_module()
            # Combine grad_loss scalar with 1/n_valid in one multiply
            scale_t = grad_loss / ctx.n_valid_item if grad_loss.dim() == 0 else grad_loss
            scale = scale_t.item() if scale_t.dim() == 0 else (grad_loss.item() / ctx.n_valid_item)
            grad_logits = mod.cross_entropy_bwd_tiny_hip(
                logits, targets, row_max, row_sum,
                ctx.softcap, ctx.ignore_index,
                ctx.label_smoothing, scale,
            )
            return grad_logits, None, None, None, None, None, None
        else:
            # Fused: grad already pre-scaled by 1/n_valid in forward. If grad_loss==1
            # (standard loss.backward() path), return saved tensor directly (zero-copy).
            (saved_grad,) = ctx.saved_tensors
            # grad_loss is a scalar fp32 tensor. Check if it's ~1.0 for zero-copy path.
            if grad_loss.dim() == 0 and grad_loss.item() == 1.0:
                return saved_grad, None, None, None, None, None, None
            # General case: multiply (full-tensor pass, but rare)
            return saved_grad * grad_loss.to(saved_grad.dtype), None, None, None, None, None, None


# =============================================================================
# Public entry points
# =============================================================================

def kernel_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Entry point matching reference.cross_entropy_ref signature (for bench.py)."""
    assert logits.is_cuda and targets.is_cuda

    if logits.dtype == torch.float32:
        return torch.nn.functional.cross_entropy(logits, targets)

    if logits.dtype != torch.float16:
        logits = logits.to(torch.float16)

    # Fast path for forward-only (bench.py) — skip grad writeback.
    if not (logits.requires_grad and torch.is_grad_enabled()):
        mod = _get_fwd_module()
        losses, _, _, _, _ = mod.cross_entropy_fwd_hip(
            logits, targets, 0.0, -100, 0.0, 0, 0, 1.0  # write_grad=0, save_max_sum=0, scale_grad unused
        )
        # Mean over non-ignored rows (matches PyTorch CE behavior)
        valid = (targets != -100).to(torch.float32)
        n_valid = valid.sum().clamp(min=1.0)
        return (losses * valid).sum() / n_valid

    # Training path: fused mode (default), saves grad_logits for ~no-op backward.
    return _CrossEntropyHIP.apply(logits, targets, 0.0, -100, 0.0, "fused", False)


def ce_full(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    softcap: float = 0.0,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    mode: str = "fused",
    return_z: bool = False,
):
    """Full-featured CE entry point.

    Args:
        logits: [B, V] fp16 (fp32 fallback uses PyTorch CE)
        targets: [B] int64
        softcap: logit softcap (0 = off). Applies softcap*tanh(x/softcap) before CE.
        ignore_index: target value to mask (default -100).
        label_smoothing: alpha (0 = off).
        mode: "fused" (default) or "tiny".
        return_z: if True, also return per-row logsumexp tensor.

    Returns:
        loss (scalar) or (loss, lse) if return_z.
    """
    assert logits.is_cuda and targets.is_cuda
    if logits.dtype == torch.float32:
        # fp32 fallback: no HIP kernel, use PyTorch
        if softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)
        loss = torch.nn.functional.cross_entropy(
            logits, targets,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        if return_z:
            lse = torch.logsumexp(logits, dim=-1)
            return loss, lse
        return loss

    if logits.dtype != torch.float16:
        logits = logits.to(torch.float16)

    return _CrossEntropyHIP.apply(
        logits, targets, softcap, ignore_index, label_smoothing, mode, return_z,
    )
