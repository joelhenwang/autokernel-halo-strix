"""
AutoKernel -- HIP C++ Matrix Multiplication kernel.

Current kernel: Tiled GEMM with double-buffered shared memory for RDNA 3.5.
Target metric: throughput_tflops (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Shared memory tiling with double buffering for compute/load overlap
  - Bank-conflict-free shared memory layout (padding)
  - Thread-level accumulation in float32 for numerical stability
  - Vectorized loads where possible
  - __launch_bounds__ for register pressure control

Note: RDNA 3.5 (gfx1151) does not have CDNA-style MFMA. This kernel uses
scalar FMA operations. The agent should optimize tile sizes and memory access
patterns for the RDNA 3.5 memory hierarchy (LDS, L1, L2, LPDDR5X).
"""

KERNEL_TYPE = "matmul"
BACKEND = "hip"

import torch
from kernels.hip._compile import compile_hip

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Tile dimensions -- tuned for RDNA 3.5 LDS capacity
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;

// Thread block: 16x16 = 256 threads
// Each thread computes a 4x4 tile of the output (BLOCK_M/16 x BLOCK_N/16)
constexpr int THREAD_M = BLOCK_M / 16;  // 4
constexpr int THREAD_N = BLOCK_N / 16;  // 4

// Shared memory padding to avoid bank conflicts
constexpr int SMEM_PAD = 8;

__global__ void __launch_bounds__(256)
matmul_kernel(
    const half* __restrict__ A,   // [M, K]
    const half* __restrict__ B,   // [K, N]
    half* __restrict__ C,         // [M, N]
    int M, int N, int K
) {
    // Block position in the output grid
    const int bx = blockIdx.x;  // M dimension
    const int by = blockIdx.y;  // N dimension

    // Thread position within the 16x16 thread block
    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;

    // Global output base for this block
    const int block_row = bx * BLOCK_M;
    const int block_col = by * BLOCK_N;

    // Double-buffered shared memory
    __shared__ half smem_A[2][BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ half smem_B[2][BLOCK_K][BLOCK_N + SMEM_PAD];

    // Per-thread accumulator: THREAD_M x THREAD_N floats
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    const int tid = threadIdx.x;
    const int k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    // Load first tile into buffer 0
    // A tile: BLOCK_M * BLOCK_K = 64*32 = 2048 elements, 256 threads -> 8 per thread
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid * 8 + i;
        int row = idx / BLOCK_K;
        int col = idx % BLOCK_K;
        int g_row = block_row + row;
        int g_col = col;
        if (g_row < M && g_col < K)
            smem_A[0][row][col] = A[g_row * K + g_col];
        else
            smem_A[0][row][col] = __float2half(0.0f);
    }
    // B tile: BLOCK_K * BLOCK_N = 32*64 = 2048 elements -> 8 per thread
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid * 8 + i;
        int row = idx / BLOCK_N;
        int col = idx % BLOCK_N;
        int g_row = row;
        int g_col = block_col + col;
        if (g_row < K && g_col < N)
            smem_B[0][row][col] = B[g_row * N + g_col];
        else
            smem_B[0][row][col] = __float2half(0.0f);
    }
    __syncthreads();

    // Main loop with double buffering
    for (int k = 0; k < k_tiles; k++) {
        int cur_buf = k % 2;
        int next_buf = 1 - cur_buf;

        // Prefetch next tile (if not last)
        if (k + 1 < k_tiles) {
            int k_next = (k + 1) * BLOCK_K;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = tid * 8 + i;
                int row = idx / BLOCK_K;
                int col = idx % BLOCK_K;
                int g_row = block_row + row;
                int g_col = k_next + col;
                if (g_row < M && g_col < K)
                    smem_A[next_buf][row][col] = A[g_row * K + g_col];
                else
                    smem_A[next_buf][row][col] = __float2half(0.0f);
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = tid * 8 + i;
                int row = idx / BLOCK_N;
                int col = idx % BLOCK_N;
                int g_row = k_next + row;
                int g_col = block_col + col;
                if (g_row < K && g_col < N)
                    smem_B[next_buf][row][col] = B[g_row * N + g_col];
                else
                    smem_B[next_buf][row][col] = __float2half(0.0f);
            }
        }

        // Compute: each thread accumulates its THREAD_M x THREAD_N tile
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk++) {
            // Load THREAD_M elements from A column kk
            float a_reg[THREAD_M];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                a_reg[i] = __half2float(smem_A[cur_buf][ty * THREAD_M + i][kk]);
            }
            // Load THREAD_N elements from B row kk
            float b_reg[THREAD_N];
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                b_reg[j] = __half2float(smem_B[cur_buf][kk][tx * THREAD_N + j]);
            }
            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++)
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    // Store results to global memory
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        int g_row = block_row + ty * THREAD_M + i;
        if (g_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int g_col = block_col + tx * THREAD_N + j;
            if (g_col < N) {
                C[g_row * N + g_col] = __float2half(acc[i][j]);
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a GPU tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a GPU tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(256);  // 16x16 threads

    matmul_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_hip(HIP_SRC, "matmul_hip")
    return _module


def kernel_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.matmul_ref signature."""
    assert A.is_cuda and B.is_cuda

    orig_dtype = A.dtype
    if A.dtype != torch.float16:
        A = A.to(torch.float16)
    if B.dtype != torch.float16:
        B = B.to(torch.float16)

    mod = _get_module()
    C = mod.matmul_hip(A, B)

    if orig_dtype != torch.float16:
        C = C.to(orig_dtype)

    return C
