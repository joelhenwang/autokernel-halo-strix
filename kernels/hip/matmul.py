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

// Tile dimensions -- 128x128 output with BLOCK_K=32
// More work per block = fewer blocks = less launch overhead
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

// Thread block: 256 threads arranged as 16x16
// Each thread computes an 8x8 output tile
constexpr int THREAD_TILE_M = 8;  // BLOCK_M / 16
constexpr int THREAD_TILE_N = 8;  // BLOCK_N / 16

constexpr int SMEM_PAD = 8;

__global__ void __launch_bounds__(256)
matmul_kernel(
    const half* __restrict__ A,   // [M, K]
    const half* __restrict__ B,   // [K, N]
    half* __restrict__ C,         // [M, N]
    int M, int N, int K
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;

    const int block_row = bx * BLOCK_M;
    const int block_col = by * BLOCK_N;

    // Double-buffered shared memory
    // A: 128*32 = 4096 halfs per buffer = ~8.5 KB, B: 32*128 = 4096 halfs = ~8.5 KB
    // Total: 2 * (8.5 + 8.5) = ~34 KB — fits in 64 KB LDS
    __shared__ half smem_A[2][BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ half smem_B[2][BLOCK_K][BLOCK_N + SMEM_PAD];

    float acc[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++)
            acc[i][j] = 0.0f;

    const int k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    // Helper: cooperative tile load
    // 128*32 = 4096 elements, 256 threads -> 16 per thread
    auto load_tile = [&](int buf, int k_base) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = tid * 16 + i;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int g_row = block_row + row;
            int g_col = k_base + col;
            smem_A[buf][row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : __float2half(0.0f);
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = tid * 16 + i;
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int g_row = k_base + row;
            int g_col = block_col + col;
            smem_B[buf][row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : __float2half(0.0f);
        }
    };

    // Load first tile
    load_tile(0, 0);
    __syncthreads();

    // Main loop with double buffering
    for (int kt = 0; kt < k_tiles; kt++) {
        int cur = kt % 2;
        int nxt = 1 - cur;

        // Prefetch next tile
        if (kt + 1 < k_tiles) {
            load_tile(nxt, (kt + 1) * BLOCK_K);
        }

        // Compute: 8x8 outer product per thread
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk++) {
            float a_reg[THREAD_TILE_M];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++)
                a_reg[i] = __half2float(smem_A[cur][ty * THREAD_TILE_M + i][kk]);

            float b_reg[THREAD_TILE_N];
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++)
                b_reg[j] = __half2float(smem_B[cur][kk][tx * THREAD_TILE_N + j]);

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++)
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    // Store with half2 vectorization
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        int g_row = block_row + ty * THREAD_TILE_M + i;
        if (g_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j += 2) {
            int g_col = block_col + tx * THREAD_TILE_N + j;
            if (g_col + 1 < N) {
                half2 val = __halves2half2(__float2half(acc[i][j]), __float2half(acc[i][j+1]));
                *reinterpret_cast<half2*>(&C[g_row * N + g_col]) = val;
            } else if (g_col < N) {
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

    # FP32 path: fall back to PyTorch (our HIP kernel is FP16-only)
    if A.dtype == torch.float32:
        return torch.mm(A, B)

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
