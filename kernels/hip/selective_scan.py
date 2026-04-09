"""
AutoKernel — HIP C++ Fused Selective Scan kernel for diagonal SSMs.

Computes the linear recurrence:
    state[t] = dA[t] * state[t-1] + dBx[t]
    y[t] = C[t] * state[t] + D * x[t]

where dA, dBx, C are (batch, seq, d_inner) and D is (d_inner,).

Uses the same three-stage parallel scan as prefix_scan.py but with the
associative operator (a₂·a₁, a₂·b₁+b₂) instead of addition.

Features:
  - Warp-level scan via __shfl_up (wave32)
  - Block-level tree via shared memory
  - One block per (batch, d_chunk) — each block scans the full sequence for BLOCK_D dims
  - fp32 throughout for SSM numerical stability
"""

KERNEL_TYPE = "selective_scan"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

// Warp-level inclusive scan with the SSM associative operator:
//   (a2, b2) ⊕ (a1, b1) = (a2*a1, a2*b1 + b2)
// Each lane holds (decay, value). After scan, each lane holds the
// cumulative (decay_product, weighted_sum) from lane 0 to this lane.
__device__ __forceinline__ void warp_ssm_scan(float& decay, float& value) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float d_other = __shfl_up(decay, offset);
        float v_other = __shfl_up(value, offset);
        if ((threadIdx.x % WARP_SIZE) >= offset) {
            // (decay, value) ⊕ (d_other, v_other) = (decay*d_other, decay*v_other + value)
            value = decay * v_other + value;
            decay = decay * d_other;
        }
    }
}

// Selective scan kernel: one block per (batch_idx, dimension).
// Threads within the block split the sequence across timesteps.
// Grid: (batch * d_inner, 1)
// Each thread handles ceil(seq_len / BLOCK_SIZE) timesteps.
__global__ void __launch_bounds__(BLOCK_SIZE)
selective_scan_kernel(
    const float* __restrict__ dA,      // (batch, seq, d_inner) — row-major
    const float* __restrict__ dBx,     // (batch, seq, d_inner)
    const float* __restrict__ C,       // (batch, seq, d_inner)
    const float* __restrict__ D,       // (d_inner,)
    const float* __restrict__ x,       // (batch, seq, d_inner) — for D*x skip
    float* __restrict__ y,             // (batch, seq, d_inner) — output
    int batch_size, int seq_len, int d_inner
) {
    const int bd_idx = blockIdx.x;       // linear index into (batch, d_inner)
    const int b = bd_idx / d_inner;      // batch index
    const int d = bd_idx % d_inner;      // dimension index

    if (b >= batch_size) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    // Shared memory for inter-warp scan
    extern __shared__ float s_data[];
    float* warp_decays = s_data;                          // num_warps
    float* warp_values = s_data + num_warps;              // num_warps
    float* warp_offset_decays = s_data + 2 * num_warps;   // num_warps
    float* warp_offset_values = s_data + 3 * num_warps;   // num_warps

    // Stride: data[b, t, d] = data[b * seq_len * d_inner + t * d_inner + d]
    const long long base = (long long)b * seq_len * d_inner + d;
    float D_val = D[d];

    // Each thread handles a contiguous chunk of timesteps
    int chunk_size = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int t_start = tid * chunk_size;
    int t_end = min(t_start + chunk_size, seq_len);

    // Pass 1: Sequential scan within thread's chunk to get boundary (decay, value)
    float total_decay = 1.0f;
    float total_value = 0.0f;
    for (int t = t_start; t < t_end; t++) {
        long long idx = base + (long long)t * d_inner;
        float da = dA[idx];
        float dbx = dBx[idx];
        total_value = da * total_value + dbx;
        total_decay = da * total_decay;
    }

    // Pass 2: Block-wide inclusive scan of (decay, value) pairs
    float scan_decay = total_decay;
    float scan_value = total_value;
    warp_ssm_scan(scan_decay, scan_value);

    if (lane_id == WARP_SIZE - 1) {
        warp_decays[warp_id] = scan_decay;
        warp_values[warp_id] = scan_value;
    }
    __syncthreads();

    // Cross-warp scan
    if (warp_id == 0) {
        float wd = (lane_id < num_warps) ? warp_decays[lane_id] : 1.0f;
        float wv = (lane_id < num_warps) ? warp_values[lane_id] : 0.0f;
        warp_ssm_scan(wd, wv);
        if (lane_id < num_warps) {
            warp_offset_decays[lane_id] = wd;
            warp_offset_values[lane_id] = wv;
        }
    }
    __syncthreads();

    // Compute exclusive prefix for this thread
    float offset_value;
    if (tid == 0) {
        offset_value = 0.0f;
    } else {
        // Exclusive = inclusive of previous thread
        float prev_scan_d = __shfl_up(scan_decay, 1);
        float prev_scan_v = __shfl_up(scan_value, 1);
        if (lane_id == 0) {
            // First lane in warp > 0: take from warp_offset of previous warp
            offset_value = warp_offset_values[warp_id - 1];
        } else {
            // Combine prior warps' total with intra-warp exclusive
            if (warp_id == 0) {
                offset_value = prev_scan_v;
            } else {
                float prior_v = warp_offset_values[warp_id - 1];
                float prior_d_unused = warp_offset_decays[warp_id - 1];
                // prev_scan is inclusive of (lane_id-1) within this warp
                // We need to combine with prior warps:
                // offset = prev_scan ⊕ prior = (prev_scan_d * prior_d, prev_scan_d * prior_v + prev_scan_v)
                offset_value = prev_scan_d * prior_v + prev_scan_v;
            }
        }
    }

    // Pass 3: Recompute local scan with correct offset, write output
    float state = offset_value;
    for (int t = t_start; t < t_end; t++) {
        long long idx = base + (long long)t * d_inner;
        float da = dA[idx];
        float dbx = dBx[idx];
        state = da * state + dbx;

        float c_val = C[idx];
        float x_val = x[idx];
        y[idx] = c_val * state + D_val * x_val;
    }
}

torch::Tensor selective_scan_hip(
    torch::Tensor dA,
    torch::Tensor dBx,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor x
) {
    TORCH_CHECK(dA.is_cuda() && dBx.is_cuda() && C.is_cuda() && D.is_cuda() && x.is_cuda(),
                "all inputs must be GPU tensors");
    TORCH_CHECK(dA.dtype() == torch::kFloat32, "dA must be float32");
    TORCH_CHECK(dA.dim() == 3, "dA must be 3D [batch, seq, d_inner]");

    int batch_size = dA.size(0);
    int seq_len = dA.size(1);
    int d_inner = dA.size(2);

    auto y = torch::empty_like(dA);

    int num_warps = BLOCK_SIZE / WARP_SIZE;
    size_t smem_bytes = 4 * num_warps * sizeof(float);

    int total_blocks = batch_size * d_inner;  // one block per (batch, dim)
    dim3 grid(total_blocks);
    dim3 block(BLOCK_SIZE);

    selective_scan_kernel<<<grid, block, smem_bytes>>>(
        dA.data_ptr<float>(),
        dBx.data_ptr<float>(),
        C.data_ptr<float>(),
        D.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size, seq_len, d_inner
    );

    return y;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "selective_scan_hip")
    return _module


def kernel_fn(
    dA: torch.Tensor,
    dBx: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Fused selective scan for diagonal SSMs.

    Args:
        dA: (batch, seq, d_inner) — per-timestep decay (exp(dt * A))
        dBx: (batch, seq, d_inner) — per-timestep input (dt * B * x)
        C: (batch, seq, d_inner) — output projection
        D: (d_inner,) — skip connection
        x: (batch, seq, d_inner) — original input for D*x term

    Returns:
        y: (batch, seq, d_inner) — scan output
    """
    assert dA.is_cuda and dA.dtype == torch.float32
    dA = dA.contiguous()
    dBx = dBx.contiguous()
    C = C.contiguous()
    D = D.contiguous()
    x = x.contiguous()

    mod = _get_module()
    return mod.selective_scan_hip(dA, dBx, C, D, x)
