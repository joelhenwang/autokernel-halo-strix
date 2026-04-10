"""
AutoKernel -- HIP C++ Parallel Selective Scan Backward kernel.

Replaces the sequential Python loops in _torch_ops.py (1024 serial steps each
for forward recompute + reverse gradient scan) with parallel prefix scans.

Forward recurrence:
    state[t] = dA[t] * state[t-1] + dBx[t]
    y[t] = C[t] * state[t] + D * x[t]

Backward:
    grad_C[t] = grad_y[t] * state[t+1]
    grad_state += grad_y[t] * C[t]
    grad_dA[t] = grad_state * state[t]
    grad_dBx[t] = grad_state
    grad_state = grad_state * dA[t]   (propagate backward)
    grad_D = sum(grad_y * x)
    grad_x = grad_y * D

Two sub-kernels:
  A: Forward state recompute via parallel prefix scan (reuse forward algorithm)
  B: Reverse gradient propagation via parallel reverse prefix scan

Expected speedup: 8-16x over sequential Python loops.
"""

KERNEL_TYPE = "selective_scan_backward"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

// --- Forward scan helpers (same as selective_scan.py) ---

__device__ __forceinline__ void warp_ssm_scan_up(float& decay, float& value) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float d_other = __shfl_up(decay, offset);
        float v_other = __shfl_up(value, offset);
        if ((threadIdx.x % WARP_SIZE) >= offset) {
            value = decay * v_other + value;
            decay = decay * d_other;
        }
    }
}

// --- Reverse scan helper: propagate (decay, value) from high lanes to low ---
// Reverse associative operator for backward:
// When propagating grad_state backward: gs[t] = gs[t+1] * dA[t+1] + local_grad[t]
// This is the same SSM operator but scanned in reverse direction.
__device__ __forceinline__ void warp_ssm_scan_down(float& decay, float& value) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float d_other = __shfl_down(decay, offset);
        float v_other = __shfl_down(value, offset);
        if ((threadIdx.x % WARP_SIZE) + offset < WARP_SIZE) {
            value = d_other * value + v_other;
            decay = d_other * decay;
        }
    }
}

// Sub-kernel A: Forward state recompute
// Produces states[0..seq_len] for each (batch, dim)
__global__ void __launch_bounds__(BLOCK_SIZE)
forward_recompute_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ dBx,
    float* __restrict__ states,   // (batch, seq_len+1, d_inner) output
    int batch_size, int seq_len, int d_inner
) {
    const int bd_idx = blockIdx.x;
    const int b = bd_idx / d_inner;
    const int d = bd_idx % d_inner;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    extern __shared__ float s_data[];
    float* warp_decays = s_data;
    float* warp_values = s_data + num_warps;
    float* warp_offset_decays = s_data + 2 * num_warps;
    float* warp_offset_values = s_data + 3 * num_warps;

    const long long base_in = (long long)b * seq_len * d_inner + d;
    const long long base_st = (long long)b * (seq_len + 1) * d_inner + d;

    int chunk_size = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int t_start = tid * chunk_size;
    int t_end = min(t_start + chunk_size, seq_len);

    // Pass 1: local sequential scan
    float total_decay = 1.0f;
    float total_value = 0.0f;
    for (int t = t_start; t < t_end; t++) {
        long long idx = base_in + (long long)t * d_inner;
        float da = dA[idx];
        float dbx = dBx[idx];
        total_value = da * total_value + dbx;
        total_decay = da * total_decay;
    }

    // Pass 2: warp-level inclusive scan
    float scan_decay = total_decay;
    float scan_value = total_value;
    warp_ssm_scan_up(scan_decay, scan_value);

    if (lane_id == WARP_SIZE - 1) {
        warp_decays[warp_id] = scan_decay;
        warp_values[warp_id] = scan_value;
    }
    __syncthreads();

    // Cross-warp scan
    if (warp_id == 0) {
        float wd = (lane_id < num_warps) ? warp_decays[lane_id] : 1.0f;
        float wv = (lane_id < num_warps) ? warp_values[lane_id] : 0.0f;
        warp_ssm_scan_up(wd, wv);
        if (lane_id < num_warps) {
            warp_offset_decays[lane_id] = wd;
            warp_offset_values[lane_id] = wv;
        }
    }
    __syncthreads();

    // Compute exclusive prefix
    float offset_value;
    if (tid == 0) {
        offset_value = 0.0f;
    } else {
        float prev_scan_d = __shfl_up(scan_decay, 1);
        float prev_scan_v = __shfl_up(scan_value, 1);
        if (lane_id == 0) {
            offset_value = warp_offset_values[warp_id - 1];
        } else {
            if (warp_id == 0) {
                offset_value = prev_scan_v;
            } else {
                float prior_v = warp_offset_values[warp_id - 1];
                offset_value = prev_scan_d * prior_v + prev_scan_v;
            }
        }
    }

    // Pass 3: Recompute with offset, write ALL states including state[0] = 0
    if (tid == 0) {
        states[base_st] = 0.0f;  // state[0] = 0
    }

    float state = offset_value;
    for (int t = t_start; t < t_end; t++) {
        long long idx = base_in + (long long)t * d_inner;
        float da = dA[idx];
        float dbx = dBx[idx];
        state = da * state + dbx;
        states[base_st + (long long)(t + 1) * d_inner] = state;
    }
}

// Sub-kernel B: Reverse gradient scan
// Given states[0..T] from sub-kernel A, compute gradients via reverse scan
__global__ void __launch_bounds__(BLOCK_SIZE)
reverse_gradient_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ dBx,
    const float* __restrict__ C,
    const float* __restrict__ D,
    const float* __restrict__ x,
    const float* __restrict__ grad_y,
    const float* __restrict__ states,  // (batch, seq_len+1, d_inner)
    float* __restrict__ grad_dA,
    float* __restrict__ grad_dBx,
    float* __restrict__ grad_C,
    float* __restrict__ grad_x,
    float* __restrict__ grad_D_partial, // (batch * d_inner) partial sums
    int batch_size, int seq_len, int d_inner
) {
    const int bd_idx = blockIdx.x;
    const int b = bd_idx / d_inner;
    const int d = bd_idx % d_inner;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    extern __shared__ float s_data[];
    float* warp_decays = s_data;
    float* warp_values = s_data + num_warps;
    float* warp_offset_decays = s_data + 2 * num_warps;
    float* warp_offset_values = s_data + 3 * num_warps;

    const long long base_in = (long long)b * seq_len * d_inner + d;
    const long long base_st = (long long)b * (seq_len + 1) * d_inner + d;
    float D_val = D[d];

    int chunk_size = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Reverse assignment: thread 0 gets the LAST chunk, thread N-1 gets the first
    // This way high-tid threads process early timesteps, enabling reverse scan
    int rev_tid = BLOCK_SIZE - 1 - tid;
    int t_start_rev = rev_tid * chunk_size;
    int t_end_rev = min(t_start_rev + chunk_size, seq_len);

    // Pass 1: Local reverse scan — compute boundary (decay, value) pair
    // Reverse recurrence: gs_new = dA[t] * gs_old + dA[t] * gy[t] * C[t]
    // Associative operator: (a, b) where gs = a * gs_incoming + b
    // Per timestep: a = dA[t], b = dA[t] * gy[t] * C[t]
    // Compose: (a2, b2) ⊕ (a1, b1) = (a2*a1, a2*b1 + b2) — same SSM operator
    float chunk_decay = 1.0f;
    float chunk_value = 0.0f;
    local_grad_D_sum = 0.0f;

    for (int t = t_end_rev - 1; t >= t_start_rev; t--) {
        long long idx = base_in + (long long)t * d_inner;
        float gy = grad_y[idx];
        float c_val = C[idx];
        float da = dA[idx];
        float x_val = x[idx];

        // SSM reverse step: gs = da * gs + da * gy * c
        float b_val = da * gy * c_val;
        chunk_value = da * chunk_value + b_val;
        chunk_decay = da * chunk_decay;

        local_grad_D_sum += gy * x_val;
    }

    // Accumulate grad_D across all threads in this block (each thread has a chunk)
    atomicAdd(&grad_D_partial[bd_idx], local_grad_D_sum);

    // Pass 2: Block-wide reverse scan of (chunk_decay, chunk_value) pairs
    // Note: thread 0 has the LAST chunk (highest t), thread BLOCK_SIZE-1 has the FIRST
    // So we do a normal "forward" scan from tid 0 to tid BLOCK_SIZE-1,
    // which corresponds to reverse in time (from t=T-1 down to t=0)
    float scan_decay = chunk_decay;
    float scan_value = chunk_value;
    warp_ssm_scan_up(scan_decay, scan_value);

    if (lane_id == WARP_SIZE - 1) {
        warp_decays[warp_id] = scan_decay;
        warp_values[warp_id] = scan_value;
    }
    __syncthreads();

    if (warp_id == 0) {
        float wd = (lane_id < num_warps) ? warp_decays[lane_id] : 1.0f;
        float wv = (lane_id < num_warps) ? warp_values[lane_id] : 0.0f;
        warp_ssm_scan_up(wd, wv);
        if (lane_id < num_warps) {
            warp_offset_decays[lane_id] = wd;
            warp_offset_values[lane_id] = wv;
        }
    }
    __syncthreads();

    // Compute exclusive prefix for this thread's incoming grad_state
    float incoming_gs;
    if (tid == 0) {
        incoming_gs = 0.0f;  // No gradient flowing in from after sequence end
    } else {
        float prev_scan_d = __shfl_up(scan_decay, 1);
        float prev_scan_v = __shfl_up(scan_value, 1);
        if (lane_id == 0) {
            incoming_gs = warp_offset_values[warp_id - 1];
        } else {
            if (warp_id == 0) {
                incoming_gs = prev_scan_v;
            } else {
                float prior_v = warp_offset_values[warp_id - 1];
                incoming_gs = prev_scan_d * prior_v + prev_scan_v;
            }
        }
    }

    // Pass 3: Recompute reverse scan with correct offset, write final gradients
    float grad_state = incoming_gs;
    for (int t = t_end_rev - 1; t >= t_start_rev; t--) {
        long long idx = base_in + (long long)t * d_inner;
        long long st_idx_t = base_st + (long long)t * d_inner;       // states[t]
        long long st_idx_t1 = base_st + (long long)(t + 1) * d_inner; // states[t+1]

        float gy = grad_y[idx];
        float c_val = C[idx];
        float da = dA[idx];
        float x_val = x[idx];

        // y[t] = C[t] * state[t+1] + D * x[t]
        grad_C[idx] = gy * states[st_idx_t1];
        grad_state += gy * c_val;

        // state[t+1] = dA[t] * state[t] + dBx[t]
        grad_dA[idx] = grad_state * states[st_idx_t];
        grad_dBx[idx] = grad_state;
        grad_state *= da;

        // grad_x from D*x term
        grad_x[idx] = gy * D_val;
    }
}

std::vector<torch::Tensor> selective_scan_backward_hip(
    torch::Tensor dA, torch::Tensor dBx, torch::Tensor C,
    torch::Tensor D, torch::Tensor x, torch::Tensor grad_y
) {
    TORCH_CHECK(dA.is_cuda() && grad_y.is_cuda(), "inputs must be GPU tensors");
    TORCH_CHECK(dA.dtype() == torch::kFloat32, "dA must be float32");
    TORCH_CHECK(dA.dim() == 3, "dA must be 3D [batch, seq, d_inner]");

    int batch_size = dA.size(0);
    int seq_len = dA.size(1);
    int d_inner = dA.size(2);

    // Allocate states buffer for forward recompute
    auto states = torch::zeros({batch_size, seq_len + 1, d_inner},
                               torch::dtype(torch::kFloat32).device(dA.device()));

    // Allocate output gradients
    auto grad_dA_out = torch::empty_like(dA);
    auto grad_dBx_out = torch::empty_like(dBx);
    auto grad_C_out = torch::empty_like(C);
    auto grad_x_out = torch::empty_like(x);
    auto grad_D_partial = torch::zeros({batch_size * d_inner},
                                        torch::dtype(torch::kFloat32).device(dA.device()));

    int num_warps = BLOCK_SIZE / WARP_SIZE;
    size_t smem_bytes = 4 * num_warps * sizeof(float);
    int total_blocks = batch_size * d_inner;

    dim3 grid(total_blocks);
    dim3 block(BLOCK_SIZE);

    // Sub-kernel A: Forward state recompute
    forward_recompute_kernel<<<grid, block, smem_bytes>>>(
        dA.data_ptr<float>(),
        dBx.data_ptr<float>(),
        states.data_ptr<float>(),
        batch_size, seq_len, d_inner
    );

    // Sub-kernel B: Reverse gradient scan
    reverse_gradient_kernel<<<grid, block, smem_bytes>>>(
        dA.data_ptr<float>(),
        dBx.data_ptr<float>(),
        C.data_ptr<float>(),
        D.data_ptr<float>(),
        x.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        states.data_ptr<float>(),
        grad_dA_out.data_ptr<float>(),
        grad_dBx_out.data_ptr<float>(),
        grad_C_out.data_ptr<float>(),
        grad_x_out.data_ptr<float>(),
        grad_D_partial.data_ptr<float>(),
        batch_size, seq_len, d_inner
    );

    // Reduce grad_D_partial: (batch*d_inner,) -> (d_inner,)
    auto grad_D_out = grad_D_partial.view({batch_size, d_inner}).sum(0);

    return {grad_dA_out, grad_dBx_out, grad_C_out, grad_D_out, grad_x_out};
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        cpp_src = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> selective_scan_backward_hip(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor);
"""
        _module = compile_hip(HIP_SRC, "selective_scan_backward_hip", cpp_src=cpp_src)
    return _module


def kernel_fn(
    dA: torch.Tensor, dBx: torch.Tensor, C: torch.Tensor,
    D: torch.Tensor, x: torch.Tensor, grad_y: torch.Tensor,
) -> tuple:
    """Parallel selective scan backward.

    Args:
        dA: (batch, seq, d_inner) fp32
        dBx: (batch, seq, d_inner) fp32
        C: (batch, seq, d_inner) fp32
        D: (d_inner,) fp32
        x: (batch, seq, d_inner) fp32
        grad_y: (batch, seq, d_inner) fp32

    Returns:
        (grad_dA, grad_dBx, grad_C, grad_D, grad_x) all fp32
    """
    assert dA.is_cuda and dA.dtype == torch.float32

    dA = dA.contiguous()
    dBx = dBx.contiguous()
    C = C.contiguous()
    D = D.contiguous()
    x = x.contiguous()
    grad_y = grad_y.contiguous()

    mod = _get_module()
    results = mod.selective_scan_backward_hip(dA, dBx, C, D, x, grad_y)
    return results[0], results[1], results[2], results[3], results[4]
