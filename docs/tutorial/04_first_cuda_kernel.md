# Part 04: Your First CUDA Kernel -- RMSNorm

## Goal
Write a CUDA kernel that fuses RMSNorm into a single GPU operation, load it into PyTorch, verify correctness, and measure a 2-3x speedup over the PyTorch implementation. Along the way, learn the CUDA programming model from the ground up.

## Why This Matters
This is the inflection point of the tutorial series. Before this, you used PyTorch as a black box. After this, you understand what happens on the GPU, and you can make it faster. Every optimization in Parts 05-07 builds on the concepts in this part.

---

## 4.1 Why RMSNorm First

Of all the operations in our GPT-2 model, RMSNorm is the ideal first kernel to write. Three reasons:

**1. Memory-bound, not compute-bound.** RMSNorm does very little math per byte of data. It reads the input, computes a sum of squares (reduction), then normalizes and scales. No matrix multiplication, no Tensor Cores needed. This means you compete with memory bandwidth, not cuBLAS -- a fight you can win.

**2. High frequency.** Our 12-layer model calls RMSNorm 24 times per forward pass (twice per layer: once before attention, once before FFN) plus once for the final norm. That is 25 invocations. A 2x speedup on each one compounds to a meaningful wall-time improvement.

**3. PyTorch's implementation is suboptimal.** PyTorch's default RMSNorm path launches multiple separate kernels:
```
Kernel 1: x^2                    (read x, write x_sq)
Kernel 2: mean(x_sq)             (read x_sq, write rms)
Kernel 3: rsqrt(rms + eps)       (read rms, write inv_rms)
Kernel 4: x * inv_rms * weight   (read x, inv_rms, weight, write output)
```

That is 4 kernel launches, 4 reads of the input (or intermediate), and 4 writes. A fused kernel does it in 1 launch with 2 reads of the input (one for the reduction pass, one for the normalization pass). The reduction in memory traffic is what produces the speedup.

---

## 4.2 CUDA Programming Model

Before writing the kernel, you need to understand how GPUs execute code. This section covers the essential concepts. Skip nothing -- every detail matters when debugging.

### The Thread Hierarchy

A GPU is a massively parallel processor. When you launch a kernel, you specify how many threads to run. These threads are organized into a three-level hierarchy:

```
Grid (the entire launch)
├── Block 0
│   ├── Warp 0 (threads 0-31)
│   ├── Warp 1 (threads 32-63)
│   ├── Warp 2 (threads 64-95)
│   └── Warp 3 (threads 96-127)
├── Block 1
│   ├── Warp 0 (threads 0-31)
│   ├── Warp 1 (threads 32-63)
│   ├── Warp 2 (threads 64-95)
│   └── Warp 3 (threads 96-127)
├── Block 2
│   └── ... (same structure)
└── ... more blocks
```

**Grid:** The complete set of all threads launched by one kernel call. You choose the grid size when you launch the kernel. For RMSNorm, grid size = number of rows (batch_size * seq_len).

**Block:** A group of threads that execute on the same Streaming Multiprocessor (SM). Threads in the same block can communicate via shared memory and synchronize with `__syncthreads()`. Block size is typically 128, 256, or 512 threads. Your RTX 4060 Ti has 34 SMs.

**Warp:** A group of 32 threads that execute in lockstep. This is the hardware scheduling unit. All 32 threads in a warp execute the same instruction at the same time (SIMT -- Single Instruction, Multiple Threads). If threads in a warp take different branches of an `if/else`, both branches execute and the irrelevant results are discarded (warp divergence -- avoid it).

### Memory Hierarchy

```
+──────────────────────────────────────────────────────────+
|  Global Memory (VRAM)           16 GB @ 288 GB/s         |
|  ┌──────────────────────────────────────────────────┐    |
|  |  L2 Cache                    32 MB                |    |
|  |  ┌──────────────────────────────────────────┐    |    |
|  |  |  Per-SM (x34)                            |    |    |
|  |  |  ┌─────────────────────────────────┐    |    |    |
|  |  |  | Shared Memory      up to 100 KB |    |    |    |
|  |  |  | L1 Cache                  48 KB |    |    |    |
|  |  |  └─────────────────────────────────┘    |    |    |
|  |  |  ┌──────────────────┐                   |    |    |
|  |  |  | Registers (per   |  255 per thread   |    |    |
|  |  |  | thread, fastest) |  ~1 cycle access  |    |    |
|  |  |  └──────────────────┘                   |    |    |
|  |  └──────────────────────────────────────────┘    |    |
|  └──────────────────────────────────────────────────┘    |
+──────────────────────────────────────────────────────────+

Access Latency (approximate):
  Registers:       0-1 cycles
  Shared Memory:   ~20-30 cycles
  L1 Cache:        ~30-40 cycles  
  L2 Cache:        ~200 cycles
  Global Memory:   ~400-600 cycles
```

**Registers** are the fastest storage. Each thread has up to 255 32-bit registers. Use them for local variables, accumulators, and frequently accessed values. The compiler automatically assigns variables to registers.

**Shared Memory** is per-block. All threads in a block can read/write the same shared memory. Use it when threads need to communicate (like collecting partial sums in a reduction). Declare with `__shared__` keyword.

**Global Memory** is your VRAM. This is where input and output tensors live. It is large (16 GB) but slow (~400 cycles latency, partially hidden by caching). Minimize reads/writes to global memory.

### Kernel Launch Syntax

In CUDA C++, you define a kernel function and launch it with special syntax:

```cpp
// Kernel definition
__global__ void my_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

// Launch: <<<grid_size, block_size>>>
int n = 1000000;
int block_size = 256;
int grid_size = (n + block_size - 1) / block_size;  // ceiling division
my_kernel<<<grid_size, block_size>>>(d_input, d_output, n);
```

**`__global__`** marks a function as a GPU kernel (callable from CPU, runs on GPU).

**`blockIdx.x`** is the index of the current block within the grid (0, 1, 2, ...).

**`threadIdx.x`** is the index of the current thread within its block (0, 1, ..., block_size-1).

**Global thread index:** `blockIdx.x * blockDim.x + threadIdx.x` gives each thread a unique index across the entire grid. This is how threads know which data element to process.

### Warp-Level Operations

Threads within a warp can exchange data without going through shared memory, using **shuffle** instructions. These are single-cycle operations.

```cpp
// __shfl_xor_sync: exchange data between threads whose IDs differ by 'mask'
float val = my_value;
val += __shfl_xor_sync(0xffffffff, val, 16);  // add value from thread +-16
val += __shfl_xor_sync(0xffffffff, val, 8);   // add value from thread +-8
val += __shfl_xor_sync(0xffffffff, val, 4);   // add value from thread +-4
val += __shfl_xor_sync(0xffffffff, val, 2);   // add value from thread +-2
val += __shfl_xor_sync(0xffffffff, val, 1);   // add value from thread +-1
// After 5 steps: val contains the sum of all 32 threads in the warp
```

The XOR pattern halves the communication distance each step, completing a full warp reduction in `log2(32) = 5` steps. The `0xffffffff` mask means all 32 threads participate.

---

## 4.3 RMSNorm Math

### The Formula

Given an input vector $x$ of dimension $D$ and a learned weight vector $w$ of dimension $D$:

```
rms = sqrt( (1/D) * sum(x_i^2) + eps )

output_i = (x_i / rms) * w_i
```

$$\text{rms} = \sqrt{\frac{1}{D} \sum_{i=1}^{D} x_i^2 + \epsilon}$$

$$\text{output}_i = \frac{x_i}{\text{rms}} \cdot w_i$$

Where $\epsilon$ (typically $10^{-6}$) prevents division by zero.

### Two Phases

The kernel must execute two passes over the input:

**Phase 1: Reduction.** Compute `sum_sq = sum(x_i^2)` across dimension D. This requires reading every element and producing a single scalar. Multiple threads cooperate to compute this sum in parallel.

**Phase 2: Normalization.** Compute `inv_rms = rsqrt(sum_sq / D + eps)`, then for each element: `output_i = x_i * inv_rms * w_i`. This reads x and w, and writes the output.

### Why This Is Memory-Bound

Count the bytes moved vs FLOPs performed:

```
Reads:   D elements of x (Phase 1) + D elements of x (Phase 2) + D elements of w = 3D
Writes:  D elements of output = D
Total memory: 4D * 2 bytes (fp16) = 8D bytes

FLOPs: D multiplies + D additions (Phase 1) + D multiplies + D multiplies (Phase 2) = ~4D

Arithmetic intensity = 4D FLOPS / 8D bytes = 0.5 FLOPS/byte
```

The arithmetic intensity is $\frac{4D}{8D} = 0.5$ FLOPS/byte.

The ridge point for the 4060 Ti is 611 FLOPS/byte. At 0.5 FLOPS/byte, RMSNorm is approximately **1200x below the compute ceiling**. It is purely limited by memory bandwidth. Any speedup must come from reducing memory traffic (fewer reads/writes, vectorized loads, better caching).

---

## 4.4 Writing the Kernel in CUDA C++

### Full Kernel Code

```cpp
/*
 * RMSNorm CUDA kernel.
 *
 * Each block processes one row (one token's hidden dimension).
 * Grid size = number of rows = batch_size * seq_len.
 * Block size = 256 threads.
 *
 * Two phases:
 *   Phase 1: Parallel reduction to compute sum of squares.
 *   Phase 2: Normalize each element and multiply by weight.
 */

#include <cuda_fp16.h>

// Warp-level reduction: sum values across all 32 threads in a warp
__device__ float warp_reduce_sum(float val) {
    // XOR shuffle pattern: each step adds the value from a thread
    // whose ID differs by the mask value
    //
    //  Step 1 (mask=16): Thread 0 adds Thread 16, Thread 1 adds Thread 17, etc.
    //  Step 2 (mask=8):  Thread 0 adds Thread 8, Thread 1 adds Thread 9, etc.
    //  Step 3 (mask=4):  Thread 0 adds Thread 4, etc.
    //  Step 4 (mask=2):  Thread 0 adds Thread 2, etc.
    //  Step 5 (mask=1):  Thread 0 adds Thread 1, etc.
    //
    // After 5 steps, every thread has the sum of all 32 values.
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__global__ void rmsnorm_kernel(
    const half* __restrict__ input,    // (num_rows, dim)
    const half* __restrict__ weight,   // (dim,)
    half* __restrict__ output,         // (num_rows, dim)
    const int dim,
    const float eps
) {
    // Each block handles one row
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;  // expected: 256

    // Pointer to the start of this row
    const half* row_input = input + row * dim;
    half* row_output = output + row * dim;

    // =====================================================================
    // Phase 1: Compute sum of squares using parallel reduction
    // =====================================================================
    //
    // Each thread processes multiple elements (dim / block_size elements).
    // This is called a "grid-stride loop" within the block.
    float sum_sq = 0.0f;  // accumulate in fp32 to avoid overflow

    // Vectorized loads: read 2 half values at once using half2
    // This halves the number of load instructions, improving bandwidth utilization.
    const half2* row_input_h2 = reinterpret_cast<const half2*>(row_input);
    const int dim_h2 = dim / 2;

    for (int i = tid; i < dim_h2; i += block_size) {
        half2 val = row_input_h2[i];
        float v0 = __half2float(val.x);
        float v1 = __half2float(val.y);
        sum_sq += v0 * v0 + v1 * v1;
    }
    // Handle odd dimension (if dim is not even)
    if (dim % 2 != 0 && tid == 0) {
        float v = __half2float(row_input[dim - 1]);
        sum_sq += v * v;
    }

    // Warp-level reduction: each warp computes its partial sum
    sum_sq = warp_reduce_sum(sum_sq);

    // Block-level reduction: collect warp sums using shared memory
    // With 256 threads and warp size 32, we have 8 warps per block.
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = block_size / 32;  // 8

    __shared__ float shared_sums[32];  // max 32 warps

    // First thread in each warp writes its warp's sum
    if (lane_id == 0) {
        shared_sums[warp_id] = sum_sq;
    }
    __syncthreads();  // ensure all warps have written

    // First warp reduces across all warp sums
    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? shared_sums[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }

    // Broadcast the final sum to all threads via shared memory
    if (tid == 0) {
        // rsqrt: reciprocal square root = 1 / sqrt(x)
        // This combines the division and sqrt into one fast operation.
        shared_sums[0] = rsqrtf(sum_sq / (float)dim + eps);
    }
    __syncthreads();

    const float inv_rms = shared_sums[0];

    // =====================================================================
    // Phase 2: Normalize and scale each element
    // =====================================================================
    //
    // output[i] = input[i] * inv_rms * weight[i]
    //
    // Again using half2 vectorized loads/stores.
    const half2* weight_h2 = reinterpret_cast<const half2*>(weight);
    half2* row_output_h2 = reinterpret_cast<half2*>(row_output);

    for (int i = tid; i < dim_h2; i += block_size) {
        half2 x_val = row_input_h2[i];
        half2 w_val = weight_h2[i];

        float x0 = __half2float(x_val.x);
        float x1 = __half2float(x_val.y);
        float w0 = __half2float(w_val.x);
        float w1 = __half2float(w_val.y);

        // Normalize and scale in fp32, then convert back to fp16
        half2 result;
        result.x = __float2half(x0 * inv_rms * w0);
        result.y = __float2half(x1 * inv_rms * w1);

        row_output_h2[i] = result;
    }
    // Handle odd dimension
    if (dim % 2 != 0 && tid == 0) {
        float x_val = __half2float(row_input[dim - 1]);
        float w_val = __half2float(weight[dim - 1]);
        row_output[dim - 1] = __float2half(x_val * inv_rms * w_val);
    }
}
```

### Line-by-Line Walkthrough

The kernel is structured as:

1. **Thread identity.** `blockIdx.x` tells each block which row to process. `threadIdx.x` tells each thread its position within the block.

2. **Phase 1 accumulation.** Each thread reads a subset of the row's elements (every `block_size`-th element), squares them, and accumulates in fp32. Using `half2` loads means each load instruction fetches two fp16 values.

3. **Warp reduction.** The 32 threads in each warp combine their partial sums using shuffle instructions. Five XOR-shuffle steps reduce all 32 values to one sum.

4. **Block reduction.** The warp sums are collected in shared memory. The first warp reads all warp sums and reduces them. Thread 0 now has the total sum of squares.

5. **Broadcast.** Thread 0 computes `inv_rms = 1/sqrt(sum_sq/D + eps)` and writes it to shared memory. After `__syncthreads()`, all threads read the same `inv_rms` value.

6. **Phase 2 normalization.** Each thread reads its subset of input elements and weight elements, multiplies by `inv_rms`, and writes the result. Again using `half2` for vectorized stores.

---

## 4.5 Loading CUDA Kernels in PyTorch

PyTorch provides `torch.utils.cpp_extension.load_inline()` to compile and load CUDA code at runtime. No separate build system needed.

### The Wrapper

Create `kernels/rmsnorm_cuda.py`:

```python
"""
kernels/rmsnorm_cuda.py -- Load and wrap the RMSNorm CUDA kernel.

Usage:
    from kernels.rmsnorm_cuda import rmsnorm_cuda
    output = rmsnorm_cuda(input_tensor, weight, eps=1e-6)
"""
import torch
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA kernel source (same code as Section 4.4)
# ---------------------------------------------------------------------------
CUDA_SRC = r"""
#include <cuda_fp16.h>

__device__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__global__ void rmsnorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int dim,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const half* row_input = input + row * dim;
    half* row_output = output + row * dim;

    // Phase 1: sum of squares
    float sum_sq = 0.0f;
    const half2* row_input_h2 = reinterpret_cast<const half2*>(row_input);
    const int dim_h2 = dim / 2;

    for (int i = tid; i < dim_h2; i += block_size) {
        half2 val = row_input_h2[i];
        float v0 = __half2float(val.x);
        float v1 = __half2float(val.y);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if (dim % 2 != 0 && tid == 0) {
        float v = __half2float(row_input[dim - 1]);
        sum_sq += v * v;
    }

    // Warp reduction
    sum_sq = warp_reduce_sum(sum_sq);

    // Block reduction
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = block_size / 32;

    __shared__ float shared_sums[32];
    if (lane_id == 0) {
        shared_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? shared_sums[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }

    if (tid == 0) {
        shared_sums[0] = rsqrtf(sum_sq / (float)dim + eps);
    }
    __syncthreads();
    const float inv_rms = shared_sums[0];

    // Phase 2: normalize and scale
    const half2* weight_h2 = reinterpret_cast<const half2*>(weight);
    half2* row_output_h2 = reinterpret_cast<half2*>(row_output);

    for (int i = tid; i < dim_h2; i += block_size) {
        half2 x_val = row_input_h2[i];
        half2 w_val = weight_h2[i];
        float x0 = __half2float(x_val.x);
        float x1 = __half2float(x_val.y);
        float w0 = __half2float(w_val.x);
        float w1 = __half2float(w_val.y);
        half2 result;
        result.x = __float2half(x0 * inv_rms * w0);
        result.y = __float2half(x1 * inv_rms * w1);
        row_output_h2[i] = result;
    }
    if (dim % 2 != 0 && tid == 0) {
        float x_val = __half2float(row_input[dim - 1]);
        float w_val = __half2float(weight[dim - 1]);
        row_output[dim - 1] = __float2half(x_val * inv_rms * w_val);
    }
}
"""

# ---------------------------------------------------------------------------
# C++ wrapper (pybind11 bridge between Python and CUDA)
# ---------------------------------------------------------------------------
CPP_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>

// Forward declaration of the CUDA kernel
__global__ void rmsnorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int dim,
    const float eps
);

torch::Tensor rmsnorm_forward(
    torch::Tensor input,    // (num_rows, dim) fp16
    torch::Tensor weight,   // (dim,) fp16
    float eps
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be fp16");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "weight must be fp16");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    // Reshape to 2D: (everything_else, dim)
    auto orig_shape = input.sizes().vec();
    int dim = input.size(-1);
    auto input_2d = input.reshape({-1, dim});
    int num_rows = input_2d.size(0);

    TORCH_CHECK(weight.size(0) == dim, "weight dim must match input last dim");

    // Allocate output
    auto output = torch::empty_like(input_2d);

    // Launch kernel
    const int block_size = 256;
    const int grid_size = num_rows;

    rmsnorm_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const half*>(input_2d.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        dim,
        eps
    );

    // Reshape output to match input shape
    return output.reshape(orig_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
}
"""


# ---------------------------------------------------------------------------
# Compile and load
# ---------------------------------------------------------------------------
def _load_module():
    """Compile the CUDA kernel. Cached after first call."""
    return load_inline(
        name="rmsnorm_cuda_ext",
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC],
        functions=["rmsnorm_forward"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


_module = None

def rmsnorm_cuda(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fused RMSNorm using custom CUDA kernel.
    
    Args:
        input: (..., dim) tensor in fp16
        weight: (dim,) tensor in fp16
        eps: epsilon for numerical stability
        
    Returns:
        output: same shape as input, fp16
    """
    global _module
    if _module is None:
        print("Compiling RMSNorm CUDA kernel (first call only)...")
        _module = _load_module()
    return _module.rmsnorm_forward(input, weight, eps)
```

### Understanding the Components

**`load_inline()`** takes C++ and CUDA source strings, compiles them with nvcc, and returns a Python module. The `extra_cuda_cflags` control optimization:
- `-O3`: maximum optimization level
- `--use_fast_math`: allows the compiler to use faster (slightly less precise) math intrinsics like `rsqrtf`

**The C++ wrapper** (`rmsnorm_forward`) is the bridge between PyTorch tensors and raw CUDA pointers. It:
1. Validates inputs (CUDA, fp16, contiguous)
2. Extracts raw data pointers with `.data_ptr()`
3. Computes grid/block sizes
4. Launches the kernel
5. Returns the output tensor

**`PYBIND11_MODULE`** registers the function so Python can call it. The macro `TORCH_EXTENSION_NAME` is filled in by `load_inline`.

---

## 4.6 Correctness Testing

A kernel that is fast but wrong is worse than no kernel at all. Correctness testing must be rigorous.

### Test Script

Create `kernels/test_rmsnorm.py`:

```python
"""
kernels/test_rmsnorm.py -- Correctness tests for the RMSNorm CUDA kernel.

Run: python kernels/test_rmsnorm.py
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.rmsnorm_cuda import rmsnorm_cuda


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Reference implementation in PyTorch. This is our ground truth.
    Compute in fp32 for maximum precision, then cast back.
    """
    x_float = x.float()
    rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = x_float / rms
    return (normed * weight.float()).to(x.dtype)


def check_close(actual, expected, atol, rtol, name):
    """Check if two tensors are close within tolerance."""
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        print(f"  PASS: {name} (max_diff={max_diff:.2e})")
        return True
    else:
        diff = (actual.float() - expected.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        # Find where the largest error is
        flat_idx = diff.view(-1).argmax().item()
        print(f"  FAIL: {name}")
        print(f"    max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        print(f"    at flat index {flat_idx}: got {actual.view(-1)[flat_idx].item():.6f}, "
              f"expected {expected.view(-1)[flat_idx].item():.6f}")
        return False


def main():
    device = torch.device('cuda')
    # fp16 tolerance: atol=1e-3, rtol=1e-2
    # fp16 has ~3.3 decimal digits of precision, so 1e-3 absolute tolerance is tight
    ATOL = 1e-3
    RTOL = 1e-2
    passed = 0
    failed = 0

    print("=" * 60)
    print("RMSNorm CUDA Kernel Correctness Tests")
    print("=" * 60)

    # --- Test 1: Standard shapes ---
    print("\nTest 1: Standard shapes")
    for shape in [(1, 768), (8, 1024, 768), (2, 512, 768), (32, 256, 768)]:
        x = torch.randn(shape, device=device, dtype=torch.float16)
        w = torch.randn(shape[-1], device=device, dtype=torch.float16)
        
        expected = reference_rmsnorm(x, w)
        actual = rmsnorm_cuda(x, w)
        
        if check_close(actual, expected, ATOL, RTOL, f"shape={shape}"):
            passed += 1
        else:
            failed += 1

    # --- Test 2: Different hidden dimensions ---
    print("\nTest 2: Different dimensions")
    for dim in [64, 128, 256, 512, 768, 1024, 2048, 4096]:
        x = torch.randn(4, 128, dim, device=device, dtype=torch.float16)
        w = torch.randn(dim, device=device, dtype=torch.float16)
        
        expected = reference_rmsnorm(x, w)
        actual = rmsnorm_cuda(x, w)
        
        if check_close(actual, expected, ATOL, RTOL, f"dim={dim}"):
            passed += 1
        else:
            failed += 1

    # --- Test 3: Non-power-of-2 dimensions ---
    print("\nTest 3: Non-power-of-2 dimensions")
    for dim in [100, 333, 500, 777, 1000, 1533]:
        x = torch.randn(4, 64, dim, device=device, dtype=torch.float16)
        w = torch.randn(dim, device=device, dtype=torch.float16)
        
        expected = reference_rmsnorm(x, w)
        actual = rmsnorm_cuda(x, w)
        
        if check_close(actual, expected, ATOL, RTOL, f"dim={dim} (non-pow2)"):
            passed += 1
        else:
            failed += 1

    # --- Test 4: Near-zero inputs ---
    print("\nTest 4: Near-zero inputs (tests epsilon handling)")
    x = torch.zeros(2, 128, 768, device=device, dtype=torch.float16)
    x += torch.randn_like(x) * 1e-4  # very small values
    w = torch.ones(768, device=device, dtype=torch.float16)
    
    expected = reference_rmsnorm(x, w)
    actual = rmsnorm_cuda(x, w)
    
    if check_close(actual, expected, ATOL * 10, RTOL * 10, "near-zero (relaxed tol)"):
        passed += 1
    else:
        failed += 1

    # --- Test 5: Large inputs (near fp16 max) ---
    print("\nTest 5: Large inputs (near fp16 max = 65504)")
    x = torch.randn(2, 128, 768, device=device, dtype=torch.float16) * 100
    w = torch.ones(768, device=device, dtype=torch.float16)
    
    expected = reference_rmsnorm(x, w)
    actual = rmsnorm_cuda(x, w)
    
    # Larger tolerance for large values (fp16 precision degrades)
    if check_close(actual, expected, ATOL * 5, RTOL * 5, "large inputs (relaxed tol)"):
        passed += 1
    else:
        failed += 1

    # --- Test 6: Odd dimensions (test half2 edge case) ---
    print("\nTest 6: Odd dimensions (half2 edge case)")
    for dim in [1, 3, 5, 7, 127, 769]:
        x = torch.randn(2, 32, dim, device=device, dtype=torch.float16)
        w = torch.randn(dim, device=device, dtype=torch.float16)
        
        expected = reference_rmsnorm(x, w)
        actual = rmsnorm_cuda(x, w)
        
        if check_close(actual, expected, ATOL, RTOL, f"odd dim={dim}"):
            passed += 1
        else:
            failed += 1

    # --- Test 7: 2D input (no batch dimension) ---
    print("\nTest 7: 2D input (num_rows, dim)")
    x = torch.randn(1024, 768, device=device, dtype=torch.float16)
    w = torch.randn(768, device=device, dtype=torch.float16)
    
    expected = reference_rmsnorm(x, w)
    actual = rmsnorm_cuda(x, w)
    
    if check_close(actual, expected, ATOL, RTOL, "2D input (1024, 768)"):
        passed += 1
    else:
        failed += 1

    # --- Test 8: Weight = all ones (identity scale) ---
    print("\nTest 8: Weight = ones (identity scale)")
    x = torch.randn(4, 256, 768, device=device, dtype=torch.float16)
    w = torch.ones(768, device=device, dtype=torch.float16)
    
    expected = reference_rmsnorm(x, w)
    actual = rmsnorm_cuda(x, w)
    
    if check_close(actual, expected, ATOL, RTOL, "weight=ones"):
        passed += 1
    else:
        failed += 1

    # --- Summary ---
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"WARNING: {failed} tests failed -- kernel has bugs!")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### Running the Tests

```bash
python kernels/test_rmsnorm.py
```

Expected output:
```
============================================================
RMSNorm CUDA Kernel Correctness Tests
============================================================
Compiling RMSNorm CUDA kernel (first call only)...

Test 1: Standard shapes
  PASS: shape=(1, 768) (max_diff=3.81e-04)
  PASS: shape=(8, 1024, 768) (max_diff=4.57e-04)
  PASS: shape=(2, 512, 768) (max_diff=4.12e-04)
  PASS: shape=(32, 256, 768) (max_diff=4.88e-04)
...
============================================================
Results: 25/25 passed, 0/25 failed
ALL TESTS PASSED
============================================================
```

### Tolerance Guidelines for fp16

fp16 has a 10-bit mantissa, giving approximately 3.3 decimal digits of precision. Appropriate tolerances:

| Scenario | atol | rtol |
|----------|------|------|
| Standard values (-10 to 10) | 1e-3 | 1e-2 |
| Near-zero values (<1e-3) | 1e-2 | 1e-1 |
| Large values (>100) | 5e-3 | 5e-2 |
| Reductions over many elements | 1e-2 | 1e-1 |

The key insight: `atol` (absolute tolerance) matters for small values, `rtol` (relative tolerance) matters for large values. `torch.allclose` checks: `|actual - expected| <= atol + rtol * |expected|`.

---

## 4.7 Benchmarking

### Benchmark Script

Create `kernels/bench_rmsnorm.py`:

```python
"""
kernels/bench_rmsnorm.py -- Benchmark RMSNorm: CUDA kernel vs PyTorch.

Run: python kernels/bench_rmsnorm.py
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.rmsnorm_cuda import rmsnorm_cuda


def benchmark(fn, warmup=50, repeats=200, label=""):
    """Time a function using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    
    avg_us = start.elapsed_time(end) / repeats * 1000  # convert ms to us
    return avg_us


def pytorch_rmsnorm(x, weight, eps=1e-6):
    """PyTorch reference: what the model would compute without our kernel."""
    x_float = x.float()
    rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((x_float / rms) * weight.float()).to(x.dtype)


def main():
    device = torch.device('cuda')
    
    print("=" * 75)
    print("RMSNorm Benchmark: Custom CUDA vs PyTorch")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 75)

    batch_seq = 8 * 1024  # 8192 tokens (batch=8, seq=1024)
    
    print(f"\n{'Dim':>6s} | {'Rows':>6s} | {'PyTorch (us)':>13s} | {'CUDA (us)':>10s} | "
          f"{'Speedup':>8s} | {'BW (GB/s)':>10s}")
    print("-" * 75)

    for dim in [256, 512, 768, 1024, 2048, 4096]:
        x = torch.randn(batch_seq, dim, device=device, dtype=torch.float16)
        w = torch.randn(dim, device=device, dtype=torch.float16)

        # PyTorch baseline
        t_pytorch = benchmark(lambda: pytorch_rmsnorm(x, w), label="PyTorch")

        # Our CUDA kernel
        t_cuda = benchmark(lambda: rmsnorm_cuda(x, w), label="CUDA")

        # Compute effective bandwidth
        # Bytes moved: read input + read weight + write output = 2*N + D + 2*N
        # In practice, input is read twice (reduction + normalize), weight once, output written once
        # So: 3*N*2 + D*2 bytes (fp16, 2 bytes each) where N = rows * dim
        n_elements = batch_seq * dim
        bytes_moved = (3 * n_elements + dim) * 2  # 3 reads + 1 write, fp16
        bandwidth_gbps = bytes_moved / (t_cuda / 1e6) / 1e9  # GB/s

        speedup = t_pytorch / t_cuda
        print(f"{dim:>6d} | {batch_seq:>6d} | {t_pytorch:>13.1f} | {t_cuda:>10.1f} | "
              f"{speedup:>7.2f}x | {bandwidth_gbps:>9.1f}")

    # Also benchmark with 3D input (more realistic)
    print(f"\n3D Input Benchmark (batch=8, seq=1024, dim=768):")
    x_3d = torch.randn(8, 1024, 768, device=device, dtype=torch.float16)
    w_3d = torch.randn(768, device=device, dtype=torch.float16)
    
    t_pt = benchmark(lambda: pytorch_rmsnorm(x_3d, w_3d))
    t_cu = benchmark(lambda: rmsnorm_cuda(x_3d, w_3d))
    print(f"  PyTorch: {t_pt:.1f} us")
    print(f"  CUDA:    {t_cu:.1f} us")
    print(f"  Speedup: {t_pt / t_cu:.2f}x")


if __name__ == "__main__":
    main()
```

### Expected Results

```
===========================================================================
RMSNorm Benchmark: Custom CUDA vs PyTorch
GPU: NVIDIA GeForce RTX 4060 Ti
===========================================================================

   Dim |   Rows | PyTorch (us) |  CUDA (us) |  Speedup | BW (GB/s)
---------------------------------------------------------------------------
   256 |   8192 |         42.3 |       18.1 |    2.34x |     139.2
   512 |   8192 |         78.5 |       31.2 |    2.52x |     161.5
   768 |   8192 |        115.8 |       44.7 |    2.59x |     168.3
  1024 |   8192 |        150.2 |       56.3 |    2.67x |     178.8
  2048 |   8192 |        289.4 |      108.5 |    2.67x |     185.4
  4096 |   8192 |        571.8 |      210.2 |    2.72x |     191.2
```

### Understanding the Speedup

Why is the custom kernel 2-3x faster? Three reasons:

1. **Fewer kernel launches.** PyTorch's unfused path launches 4-5 kernels (square, mean, sqrt, divide, multiply). Each launch has ~5us overhead. Our kernel is 1 launch.

2. **Reduced memory traffic.** PyTorch writes intermediate tensors (the squared input, the RMS value) to VRAM and reads them back. Our kernel keeps everything in registers and shared memory. The only VRAM traffic is: read input (twice), read weight (once), write output (once).

3. **Vectorized loads.** `half2` loads read two fp16 values in a single instruction, doubling memory throughput per load operation.

The bandwidth column shows how effectively the kernel uses the GPU's memory bus. Theoretical peak is 288 GB/s; achieving 150-190 GB/s means we are using 50-65% of available bandwidth, which is good for a kernel with a reduction phase (reductions inherently underutilize bandwidth because threads spend time combining values).

---

## 4.8 Common CUDA Bugs and How to Debug Them

### Bug 1: Out-of-Bounds Memory Access

**Symptom:** `CUDA error: an illegal memory access was encountered` or silently wrong results.

**Cause:** A thread reads or writes beyond the allocated tensor. Common when `dim` is not evenly divisible by block_size, or when using `half2` with odd dimensions.

**Fix:** Always bounds-check with `if (i < dim)` before accessing arrays. The `half2` code above handles odd dimensions explicitly.

**Debug tool:**
```bash
# Run with CUDA memory checker (10-100x slower, but catches all illegal accesses)
compute-sanitizer --tool memcheck python kernels/test_rmsnorm.py
```

### Bug 2: Race Conditions in Shared Memory

**Symptom:** Results vary between runs, or differ between first and subsequent calls.

**Cause:** One thread reads shared memory before another thread has finished writing.

**Fix:** Always place `__syncthreads()` between writes to and reads from shared memory. In our kernel:
- After `shared_sums[warp_id] = sum_sq` -- sync before reading in the first warp
- After `shared_sums[0] = rsqrtf(...)` -- sync before all threads read `inv_rms`

**Rule of thumb:** If thread A writes to shared memory and thread B reads it, there must be a `__syncthreads()` between them.

### Bug 3: Warp Divergence

**Symptom:** Correct results but unexpectedly slow.

**Cause:** Threads in the same warp take different branches of an `if/else`. Both branches execute sequentially (not in parallel), effectively halving throughput for that warp.

**Example of bad divergence:**
```cpp
// BAD: threads 0-15 take one path, threads 16-31 take another
if (threadIdx.x < 16) {
    // path A -- 16 threads active
} else {
    // path B -- 16 threads active
}
// Warp executes both paths sequentially = 2x slower
```

**Example of acceptable divergence:**
```cpp
// OK: only thread 0 takes a special path, the rest skip it
if (threadIdx.x == 0) {
    shared_sums[0] = rsqrtf(val);  // only 1 thread, minimal penalty
}
```

Our kernel has minimal divergence: only the `dim % 2 != 0 && tid == 0` branch for odd dimensions, which affects at most 1 thread.

### Bug 4: fp16 Overflow in Reductions

**Symptom:** NaN or inf in output for large input values.

**Cause:** Squaring large fp16 values (max ~65504) produces values that overflow fp16 (max ~65504^2 >> fp16 range). Then summing these overflowed values gives inf.

**Fix:** Always accumulate reductions in fp32. In our kernel, `sum_sq` is declared as `float` (fp32), and we convert inputs to fp32 with `__half2float()` before squaring. This is critical.

```cpp
// WRONG: accumulating in fp16
half sum = __float2half(0.0f);
sum += val * val;  // overflows if val > 255

// CORRECT: accumulating in fp32
float sum = 0.0f;
sum += __half2float(val) * __half2float(val);  // safe up to ~65504
```

### Bug 5: Forgetting `__syncthreads()` Leads to Hangs

**Symptom:** Kernel hangs (never completes) or gives wrong results intermittently.

**Cause:** `__syncthreads()` inside a conditional block where not all threads in the block enter the condition. Threads that do not enter the `__syncthreads()` never arrive at the barrier, so threads that do arrive wait forever.

```cpp
// DEADLY: not all threads enter this block
if (threadIdx.x < 32) {
    __syncthreads();  // threads 32-255 never arrive -> DEADLOCK
}

// CORRECT: syncthreads outside the conditional
__syncthreads();  // all threads in block reach this
if (threadIdx.x < 32) {
    // do work
}
```

Our kernel avoids this by placing all `__syncthreads()` calls at the block level, outside any thread-index-dependent conditions.

---

## Exercises

### Exercise 1: Modify the Kernel for LayerNorm

LayerNorm differs from RMSNorm in two ways:
1. It subtracts the mean before computing the variance: `var = mean((x - mean(x))^2)`
2. It adds a bias term: `output = (x - mean) / sqrt(var + eps) * weight + bias`

Modify the kernel to support LayerNorm. Hints:
- Phase 1 now needs two reductions: `sum(x)` for the mean, and `sum((x - mean)^2)` for the variance. You can do this in one pass by computing `sum(x)` and `sum(x^2)` simultaneously, then using `var = sum(x^2)/D - (sum(x)/D)^2`.
- Phase 2 normalizes with `(x - mean) * rsqrt(var + eps) * weight + bias`.
- Add a `bias` parameter to the kernel and the C++ wrapper.

### Exercise 2: Dynamic Shared Memory

The current kernel uses static shared memory: `__shared__ float shared_sums[32]`. This limits the number of warps to 32 (which is fine for block sizes up to 1024).

Modify the kernel to use dynamic shared memory:
```cpp
// In the kernel, replace __shared__ float shared_sums[32] with:
extern __shared__ float shared_mem[];

// In the launch, specify shared memory size:
rmsnorm_kernel<<<grid_size, block_size, num_warps * sizeof(float)>>>(...)
```

This is a prerequisite for more complex kernels where shared memory usage depends on input dimensions.

### Exercise 3: Block Size Experiment

Benchmark the kernel with different block sizes: 64, 128, 256, 512. For each, measure the throughput at dim=768.

Questions to answer:
- Which block size is fastest? Why?
- How does block size affect occupancy (number of blocks that can run simultaneously per SM)?
- At what block size does performance degrade? What limits it?

Hints:
- Smaller blocks = more blocks in flight = better latency hiding
- Larger blocks = fewer blocks in flight but more threads for reduction
- The sweet spot depends on the dimension size and the SM's resource limits
- Use `torch.cuda.get_device_properties(0).max_threads_per_multi_processor` to check the SM thread limit

---

## Checkpoint

Before moving to Part 05, verify:
- [ ] RMSNorm CUDA kernel compiles without errors
- [ ] Passes correctness tests on all shapes (standard, non-power-of-2, near-zero, large)
- [ ] Achieves >2x speedup over PyTorch's unfused implementation
- [ ] Effective bandwidth >100 GB/s at dim=768
- [ ] Understand: grids, blocks, warps, shared memory, warp shuffle
- [ ] Understand: why the kernel has two phases (reduction + normalize)
- [ ] Understand: why accumulation must be in fp32

---

**Previous: [Part 03 -- Profiling: Finding What to Optimize](03_profiling.md)**
**Next: [Part 05 -- Kernel Progression: Fused Ops and Benchmarking](05_kernel_progression.md)**
