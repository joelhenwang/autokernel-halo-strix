# Part 05: Kernel Progression -- Fusion, Correctness, Benchmarking

## Goal
Write four production-quality fused CUDA kernels, build a reusable correctness harness, and assemble a benchmark suite that reports speedup, bandwidth utilization, and roofline position. By the end, you will have the kernel library that Parts 06-07 plug into a real model.

## Why This Matters
Part 04 taught you how to write one kernel. That is like learning to swing a hammer once. This part teaches you to build an entire toolkit -- and more importantly, to *verify* that every tool actually works before you trust it in a training run. A 2x faster kernel that is wrong 0.1% of the time will silently corrupt your model.

---

## 5.1 The Kernel Fusion Principle

### Why Fusion Wins

Consider a typical transformer FFN residual path in PyTorch:

```python
# Three separate PyTorch operations
hidden = x + residual          # op A: reads x, residual; writes hidden
norm_out = rmsnorm(hidden)     # op B: reads hidden; writes norm_out
ffn_out = ffn(norm_out)        # op C: reads norm_out (we won't fuse this -- it's matmul)
```

Each operation launches a separate CUDA kernel. Each kernel must:
1. Read its inputs from global memory (VRAM).
2. Compute.
3. Write its outputs back to global memory.

Between op A and op B, the `hidden` tensor lives in VRAM. The GPU writes it (~288 GB/s write), then immediately reads it back (~288 GB/s read). That round-trip is pure waste -- the data was just computed and is still warm in registers or L1 cache.

### The Memory Traffic Accounting

For a tensor of size `[B, T, D]` in fp16 (2 bytes per element), with `N = B * T * D` elements:

**Unfused (op A + op B separately):**
| Operation | Reads | Writes |
|-----------|-------|--------|
| Add (A) | `x` (2N bytes) + `residual` (2N bytes) | `hidden` (2N bytes) |
| RMSNorm (B) | `hidden` (2N bytes) + `weight` (2D bytes) | `output` (2N bytes) |
| **Total** | **8N + 2D bytes** | **4N bytes** |

**Fused (A+B in one kernel):**
| Operation | Reads | Writes |
|-----------|-------|--------|
| FusedResNorm | `x` (2N bytes) + `residual` (2N bytes) + `weight` (2D bytes) | `output` (2N bytes) |
| **Total** | **4N + 2D bytes** | **2N bytes** |

The fused kernel does **half the memory traffic**. On a bandwidth-limited GPU like the RTX 4060 Ti (288 GB/s), halving memory traffic means roughly 2x speedup -- and that is before accounting for kernel launch overhead savings.

### The General Rule

```
Fused kernel memory traffic = read(first inputs) + write(last output)
Unfused kernel memory traffic = sum of all intermediate reads and writes
```

The biggest wins come from fusing 3 or more sequential element-wise operations, because you eliminate 2 or more intermediate tensors. The residual + RMSNorm fusion eliminates one intermediate (`hidden`), giving ~2x on memory traffic. A hypothetical residual + RMSNorm + SiLU fusion would eliminate two intermediates, giving closer to 3x.

### When Fusion Does NOT Help

Fusion only helps when the operations are **memory-bound** (limited by how fast you can read/write data, not by how fast you can compute). Matrix multiplications are **compute-bound** -- they do O(N^3) work on O(N^2) data. Fusing a matmul with an element-wise op saves negligible time because the matmul dominates. Let cuBLAS handle matmuls.

---

## 5.2 Fused Residual + RMSNorm Kernel

### The Pattern

This appears in every transformer block:

```python
def transformer_block(x, residual, attn_output, weight, eps=1e-6):
    # After attention
    hidden = x + attn_output       # residual connection
    # Before FFN
    rms = (hidden.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    output = hidden * rms * weight  # RMSNorm
    return hidden, output           # hidden needed for next residual
```

PyTorch executes this as two separate kernels with an intermediate `hidden` tensor in VRAM.

### Fused CUDA Kernel

```cuda
// fused_residual_rmsnorm.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr float EPS = 1e-6f;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    // __shfl_xor broadcasts the result to ALL lanes in the warp.
    // This is critical: every thread needs the sum for normalization.
    // __shfl_down would only give the result to lane 0.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// One block per row (token). Dynamic shared memory caches the hidden row.
__global__ void __launch_bounds__(1024)
fused_residual_rmsnorm_kernel(
    const half* __restrict__ X,       // [M, N] -- input
    const half* __restrict__ R,       // [M, N] -- residual
    const half* __restrict__ W,       // [N]    -- RMSNorm weight
    half* __restrict__ OUT,           // [M, N] -- normalized output
    half* __restrict__ HIDDEN,        // [M, N] -- x + residual (for next residual)
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockSize / WARP_SIZE;

    const half* x_row = X + (long long)row * N;
    const half* r_row = R + (long long)row * N;

    // Dynamic shared memory: cache hidden row + warp-level partial sums
    extern __shared__ char smem[];
    float* s_hidden = reinterpret_cast<float*>(smem);              // N floats
    float* s_warp_sums = s_hidden + N;                             // num_warps floats

    // Step 1: Compute hidden = x + residual, store in shared memory.
    //         Simultaneously accumulate sum-of-squares for RMSNorm.
    float thread_sq_sum = 0.0f;
    for (int i = tid; i < N; i += blockSize) {
        float xi = __half2float(x_row[i]);
        float ri = __half2float(r_row[i]);
        float hi = xi + ri;
        s_hidden[i] = hi;
        thread_sq_sum += hi * hi;
    }

    // Step 2: Reduce sum-of-squares across the block.
    //         Two-level reduction: warp-level shuffle, then cross-warp via shared memory.
    float warp_sum = warp_reduce_sum(thread_sq_sum);
    if (lane_id == 0) {
        s_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final reduction across warps (only first warp participates)
    float total_sq = 0.0f;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0f;
        total_sq = warp_reduce_sum(val);
    }
    // Broadcast total_sq to all threads via shared memory
    if (tid == 0) {
        s_warp_sums[0] = total_sq;
    }
    __syncthreads();
    total_sq = s_warp_sums[0];

    // Step 3: Compute RMSNorm scale factor
    float rms_inv = rsqrtf(total_sq / (float)N + EPS);

    // Step 4: Normalize and write outputs.
    //         Read hidden from shared memory (NOT from global memory).
    half* out_row = OUT + (long long)row * N;
    half* hid_row = HIDDEN + (long long)row * N;
    for (int i = tid; i < N; i += blockSize) {
        float hi = s_hidden[i];
        float wi = __half2float(W[i]);
        float normed = hi * rms_inv * wi;
        out_row[i] = __float2half(normed);
        hid_row[i] = __float2half(hi);   // write hidden for next residual
    }
}
```

### Python Wrapper

```python
import torch
from torch.utils.cpp_extension import load_inline

# Load the kernel (load_inline compiles on first call, caches afterward)
cuda_src = open("fused_residual_rmsnorm.cu").read()
fused_module = load_inline(
    name="fused_residual_rmsnorm",
    cpp_sources="",  # no C++ wrapper needed
    cuda_sources=cuda_src,
    functions=["fused_residual_rmsnorm_kernel"],  # exported name
    verbose=False,
)

def fused_residual_rmsnorm(x: torch.Tensor, residual: torch.Tensor,
                            weight: torch.Tensor) -> tuple:
    """Fused residual add + RMSNorm.

    Args:
        x: [B, T, D] or [M, N] input tensor (fp16)
        residual: same shape as x (fp16)
        weight: [D] or [N] RMSNorm weight (fp16)

    Returns:
        (hidden, normed) -- both same shape as x
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1]).contiguous()
    r_flat = residual.reshape(-1, residual.shape[-1]).contiguous()
    M, N = x_flat.shape

    out = torch.empty_like(x_flat)
    hidden = torch.empty_like(x_flat)

    # Thread block: up to 1024, must be multiple of 32, at least N threads if N <= 1024
    threads = min(1024, ((N + 31) // 32) * 32)
    # Shared memory: N floats for hidden cache + (threads/32) floats for warp sums
    num_warps = threads // 32
    smem_bytes = (N + num_warps) * 4  # 4 bytes per float

    fused_module.fused_residual_rmsnorm_kernel(
        x_flat, r_flat, weight, out, hidden,
        M, N,
        grid=(M,), block=(threads,), shared_mem=smem_bytes,
    )
    return hidden.view(orig_shape), out.view(orig_shape)
```

### Why Shared Memory Matters Here

Without shared memory, we would need to:
1. Read `x` and `residual` from global memory, compute `hidden`, write `hidden` to global memory.
2. Read `hidden` back from global memory, compute RMSNorm, write output.

With shared memory, step 2 reads `hidden` from LDS (shared memory) instead of global memory. LDS bandwidth is roughly 10-20x higher than global memory bandwidth. The shared memory acts as a programmer-controlled cache that we explicitly load and read from.

### Expected Speedup

On RTX 4060 Ti with D=768, B=8, T=512: **4-6x** over unfused PyTorch, because:
- Eliminated one global memory round-trip for `hidden` (saves ~50% of traffic)
- Eliminated one kernel launch overhead (~5 microseconds saved)
- Shared memory reads are ~10x faster than global memory reads

---

## 5.3 SwiGLU Activation Kernel

### The Pattern

SwiGLU is the activation function used in LLaMA, Mistral, and most modern LLMs:

```python
def swiglu(gate, up):
    """gate and up are outputs of two separate linear projections."""
    return torch.silu(gate) * up
    # silu(x) = x * sigmoid(x)
```

PyTorch runs this as two kernels:
1. `silu(gate)` -- reads `gate`, computes sigmoid, multiplies, writes temp tensor
2. `temp * up` -- reads temp and `up`, multiplies, writes output

The intermediate `temp` tensor is pure waste.

### Fused CUDA Kernel

```cuda
// fused_swiglu.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int BLOCK_SIZE = 256;

__global__ void __launch_bounds__(BLOCK_SIZE)
fused_swiglu_kernel(
    const half* __restrict__ gate,    // [N] flattened
    const half* __restrict__ up,      // [N] flattened
    half* __restrict__ output,        // [N] flattened
    int N
) {
    // Process two fp16 values at a time via half2 for 2x memory throughput
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int stride = gridDim.x * BLOCK_SIZE;

    const half2* gate_v = reinterpret_cast<const half2*>(gate);
    const half2* up_v = reinterpret_cast<const half2*>(up);
    half2* out_v = reinterpret_cast<half2*>(output);
    const int n_pairs = N / 2;

    for (int i = tid; i < n_pairs; i += stride) {
        half2 g = gate_v[i];
        half2 u = up_v[i];

        // Compute in fp32 for numerical stability
        float g0 = __half2float(g.x);
        float g1 = __half2float(g.y);
        float u0 = __half2float(u.x);
        float u1 = __half2float(u.y);

        // SiLU(g) * u = g * sigmoid(g) * u
        // __expf is a fast exponential intrinsic (less precise than expf,
        // but sufficient for fp16 output)
        float y0 = (g0 / (1.0f + __expf(-g0))) * u0;
        float y1 = (g1 / (1.0f + __expf(-g1))) * u1;

        half2 result;
        result.x = __float2half(y0);
        result.y = __float2half(y1);
        out_v[i] = result;
    }

    // Handle odd last element
    if (tid == 0 && (N % 2 != 0)) {
        float g = __half2float(gate[N - 1]);
        float u = __half2float(up[N - 1]);
        output[N - 1] = __float2half((g / (1.0f + __expf(-g))) * u);
    }
}
```

### Python Wrapper

```python
def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up.

    Args:
        gate: any shape, fp16, contiguous
        up: same shape as gate, fp16, contiguous

    Returns:
        output: same shape, fp16
    """
    assert gate.shape == up.shape
    assert gate.dtype == torch.float16
    N = gate.numel()
    output = torch.empty_like(gate)

    threads = BLOCK_SIZE
    blocks = min((N // 2 + threads - 1) // threads, 65535)

    fused_module.fused_swiglu_kernel(
        gate.contiguous(), up.contiguous(), output,
        N,
        grid=(blocks,), block=(threads,),
    )
    return output
```

### Key Design Decisions

1. **half2 vectorized loads**: Each thread loads two fp16 values at once. This doubles memory throughput because the memory controller can service 32-bit aligned loads more efficiently than 16-bit loads.

2. **fp32 intermediate**: We promote to fp32 for the sigmoid/exp computation. Computing `exp(-g)` in fp16 would overflow for `|g| > 11` (fp16 max is 65504). The final result goes back to fp16.

3. **Grid-stride loop**: `for (int i = tid; i < n_pairs; i += stride)` lets one block handle more data than its thread count. This is better than launching exactly one thread per element because it amortizes kernel launch cost and gives better occupancy for small tensors.

### Expected Speedup

On RTX 4060 Ti: **1.5-2x** over PyTorch's separate `silu` + `mul` kernels, primarily from eliminating the intermediate tensor.

---

## 5.4 Rotary Embedding Kernel

### The Math

Rotary Position Embedding (RoPE) encodes position information by rotating pairs of elements in the query and key vectors:

```
For each pair (x[2i], x[2i+1]) at position t:
    x_new[2i]   = x[2i]   * cos(t * freq_i) - x[2i+1] * sin(t * freq_i)
    x_new[2i+1] = x[2i+1] * cos(t * freq_i) + x[2i]   * sin(t * freq_i)

where freq_i = 1 / (10000^(2i / d))
```

This is a 2D rotation matrix applied independently to each pair of dimensions:

```
[x_new_0]   [cos(theta)  -sin(theta)] [x_0]
[x_new_1] = [sin(theta)   cos(theta)] [x_1]
```

### PyTorch Reference

```python
def apply_rotary_emb_pytorch(x, cos, sin):
    """
    x: [B, H, T, D]  (batch, heads, seq_len, head_dim)
    cos: [T, D//2]
    sin: [T, D//2]
    """
    x0, x1 = x[..., ::2], x[..., 1::2]  # even and odd elements
    # cos/sin need broadcasting: [1, 1, T, D//2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out0 = x0 * cos - x1 * sin
    out1 = x1 * cos + x0 * sin
    return torch.stack([out0, out1], dim=-1).flatten(-2)
```

This creates multiple intermediate tensors (`x0`, `x1`, `out0`, `out1`, and the stacked result).

### Precision Path Warning

There are two common RoPE implementations:

1. **fp16 throughout** (HuggingFace default): cos/sin stay in fp16, rotation in fp16.
2. **fp32 intermediate** (LLaMA original): promote to fp32 before rotation, cast back to fp16.

Your CUDA kernel **must match** whichever precision path your model uses. If the model trains with fp32 intermediates but your kernel uses fp16, you will get different gradients during training, leading to subtly wrong results that pass basic correctness tests but diverge over thousands of training steps.

### Fused CUDA Kernel (fp32 intermediate path)

```cuda
// rotary_embedding.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int BLOCK_SIZE = 256;

// x: [B, H, T, D], cos_cache: [T, D/2], sin_cache: [T, D/2]
// Each thread handles one (even, odd) pair.
__global__ void __launch_bounds__(BLOCK_SIZE)
rotary_embedding_kernel(
    const half* __restrict__ x,
    half* __restrict__ output,
    const half* __restrict__ cos_cache,
    const half* __restrict__ sin_cache,
    int B, int H, int T, int D
) {
    const int half_D = D / 2;
    const int total = B * H * T * half_D;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {

        // Decode linear index into (b, h, t, pair)
        int pair = idx % half_D;
        int remainder = idx / half_D;
        int t = remainder % T;
        remainder = remainder / T;
        int h = remainder % H;
        int b = remainder / H;

        // Indices into x: the pair (x[..., 2*pair], x[..., 2*pair+1])
        int base = ((b * H + h) * T + t) * D + pair * 2;
        float x0 = __half2float(x[base]);
        float x1 = __half2float(x[base + 1]);

        // Indices into cos/sin cache: [t, pair]
        int cache_idx = t * half_D + pair;
        float c = __half2float(cos_cache[cache_idx]);
        float s = __half2float(sin_cache[cache_idx]);

        // 2D rotation in fp32
        float out0 = x0 * c - x1 * s;
        float out1 = x1 * c + x0 * s;

        output[base]     = __float2half(out0);
        output[base + 1] = __float2half(out1);
    }
}
```

### Python Wrapper

```python
def fused_rotary_embedding(x: torch.Tensor, cos: torch.Tensor,
                            sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x.

    Args:
        x: [B, H, T, D] fp16 tensor
        cos: [T, D//2] fp16 precomputed cosines
        sin: [T, D//2] fp16 precomputed sines

    Returns:
        [B, H, T, D] fp16 tensor with rotary embeddings applied
    """
    B, H, T, D = x.shape
    assert D % 2 == 0
    output = torch.empty_like(x)

    total = B * H * T * (D // 2)
    threads = BLOCK_SIZE
    blocks = min((total + threads - 1) // threads, 65535)

    fused_module.rotary_embedding_kernel(
        x.contiguous(), output, cos.contiguous(), sin.contiguous(),
        B, H, T, D,
        grid=(blocks,), block=(threads,),
    )
    return output
```

### Why This Kernel Helps

The PyTorch version creates 5 intermediate tensors (even slice, odd slice, two products, stack output). Our kernel reads `x` once, reads cos/sin once, writes output once. For a typical setup (B=8, H=12, T=512, D=64), this eliminates ~6 MB of intermediate memory traffic.

### Expected Speedup

On RTX 4060 Ti: **2-3x** over PyTorch, depending on sequence length and head dimension.

---

## 5.5 Cross-Entropy Loss Kernel

### The Algorithm: Online Softmax

The standard cross-entropy computation is:

```
loss = -log(softmax(logits)[target])
     = -logits[target] + log(sum(exp(logits)))
```

The naive approach computes this in three passes:
1. Find `max(logits)` (for numerical stability)
2. Compute `sum(exp(logits - max))`
3. Compute `loss = -logits[target] + max + log(sum)`

The **online softmax** algorithm (Milakov & Gimelshein, 2018) fuses steps 1 and 2 into a single pass:

```
// Single pass: track running max and running sum simultaneously
max_val = -inf
sum_exp = 0
for each logit:
    if logit > max_val:
        // New maximum found. Correct all previously accumulated exp values.
        sum_exp = sum_exp * exp(old_max - new_max)
        max_val = logit
    sum_exp += exp(logit - max_val)
```

This halves the number of passes over the logits vector (which is `vocab_size` wide -- typically 32K-128K elements).

### Warp Shuffle: `__shfl_xor` vs `__shfl_down`

Both are warp-level primitives that let threads exchange register values without going through shared memory.

- **`__shfl_down_sync(mask, val, offset)`**: Thread `i` gets value from thread `i + offset`. Only the lower-numbered threads get useful results. Thread 31 gets garbage. Use this when only one thread (lane 0) needs the final result.

- **`__shfl_xor_sync(mask, val, offset)`**: Thread `i` gets value from thread `i XOR offset`. ALL threads get useful results (the XOR pattern pairs every thread symmetrically). Use this when **every thread** needs the final result.

For cross-entropy, every thread needs the max and sum to compute its contribution to the loss. So we use `__shfl_xor`:

```cuda
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;  // ALL 32 threads now hold the maximum
}
```

### Fused CUDA Kernel

```cuda
// cross_entropy.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// One block per sample in the batch.
// Uses online softmax: single pass for max + sum_exp.
__global__ void __launch_bounds__(BLOCK_SIZE)
cross_entropy_kernel(
    const half* __restrict__ logits,     // [batch, vocab]
    const int64_t* __restrict__ targets, // [batch]
    float* __restrict__ losses,          // [batch]
    int batch, int vocab
) {
    const int b = blockIdx.x;
    if (b >= batch) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const half* row = logits + (long long)b * vocab;
    const int target = targets[b];

    // Online max + sum_exp in a single pass
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    for (int i = tid; i < vocab; i += BLOCK_SIZE) {
        float val = __half2float(row[i]);
        if (val > thread_max) {
            // Correct previously accumulated exponents for new max
            thread_sum *= __expf(thread_max - val);
            thread_max = val;
        }
        thread_sum += __expf(val - thread_max);
    }

    // Warp-level reduction of (max, sum) pairs
    // First reduce max across warp
    float warp_max = warp_reduce_max(thread_max);
    // Correct each thread's sum to use warp_max
    thread_sum *= __expf(thread_max - warp_max);
    float warp_sum = warp_reduce_sum(thread_sum);

    // Cross-warp reduction via shared memory
    __shared__ float s_max[NUM_WARPS];
    __shared__ float s_sum[NUM_WARPS];

    if (lane_id == 0) {
        s_max[warp_id] = warp_max;
        s_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float m = (lane_id < NUM_WARPS) ? s_max[lane_id] : -FLT_MAX;
        float s = (lane_id < NUM_WARPS) ? s_sum[lane_id] : 0.0f;

        float block_max = warp_reduce_max(m);
        s *= __expf(m - block_max);
        float block_sum = warp_reduce_sum(s);

        if (lane_id == 0) {
            float target_logit = __half2float(row[target]);
            // loss = -target_logit + max + log(sum_exp)
            losses[b] = -target_logit + block_max + __logf(block_sum);
        }
    }
}
```

### Python Wrapper

```python
def fused_cross_entropy(logits: torch.Tensor,
                        targets: torch.Tensor) -> torch.Tensor:
    """Fused cross-entropy loss.

    Args:
        logits: [B, V] fp16 logits (B = batch, V = vocab_size)
        targets: [B] int64 target indices

    Returns:
        [B] fp32 per-sample losses
    """
    B, V = logits.shape
    losses = torch.empty(B, dtype=torch.float32, device=logits.device)

    fused_module.cross_entropy_kernel(
        logits.contiguous(), targets.contiguous(), losses,
        B, V,
        grid=(B,), block=(BLOCK_SIZE,),
    )
    return losses
```

### Expected Speedup

On RTX 4060 Ti with vocab=50257: **1.5-2x** over PyTorch's `F.cross_entropy`, primarily from eliminating the full softmax materialization.

---

## 5.6 Building a Correctness Harness

### The 5-Stage Testing Protocol

Every kernel must pass all five stages before you trust it in training:

**Stage 1: Basic Shapes**
Test with the exact shapes your model uses (e.g., B=8, T=512, D=768 for GPT-2 124M).

**Stage 2: Edge Cases**
- Minimum size: B=1, T=1, D=64
- Non-power-of-2 dimensions: D=768 (not a power of 2)
- Maximum batch: fill 80% of VRAM
- D not divisible by warp size: D=100

**Stage 3: Numerical Stability**
- Very large inputs: `x = torch.randn(...) * 100`
- Very small inputs: `x = torch.randn(...) * 0.001`
- Mix of large and small: `x[:, :, :384] *= 100; x[:, :, 384:] *= 0.001`
- All zeros (degenerate case)

**Stage 4: Determinism**
Run the kernel 10 times on the same input. Results must be bit-identical.

**Stage 5: Large Scale**
Test with production-sized inputs over 1000 iterations. Compare accumulated error (not just per-call error).

### Tolerance Selection

```python
# fp16 has ~3.3 decimal digits of precision (mantissa = 10 bits)
# fp32 has ~7.2 decimal digits of precision (mantissa = 23 bits)

FP16_ATOL = 1e-3   # absolute tolerance for fp16 outputs
FP16_RTOL = 1e-2   # relative tolerance for fp16 outputs
FP32_ATOL = 1e-6   # absolute tolerance for fp32 outputs
FP32_RTOL = 1e-5   # relative tolerance for fp32 outputs
```

Why these specific values? fp16 can represent values with ~0.1% relative error. Setting `rtol=1e-2` gives 10x headroom for accumulated rounding from multiple operations. Setting `atol=1e-3` handles values near zero where relative error is meaningless.

### Reusable Correctness Framework

```python
"""correctness.py -- Reusable kernel correctness testing framework."""
import torch
import sys
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

@dataclass
class TestResult:
    name: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    details: str = ""

class KernelCorrectnessTest:
    """Test a CUDA kernel against a PyTorch reference implementation."""

    def __init__(self, kernel_fn: Callable, reference_fn: Callable,
                 name: str, dtype: torch.dtype = torch.float16):
        self.kernel_fn = kernel_fn
        self.reference_fn = reference_fn
        self.name = name
        self.dtype = dtype

        if dtype == torch.float16:
            self.atol = 1e-3
            self.rtol = 1e-2
        else:
            self.atol = 1e-6
            self.rtol = 1e-5

    def _compare(self, result: torch.Tensor, expected: torch.Tensor,
                 test_name: str) -> TestResult:
        """Compare kernel output against reference."""
        # Cast both to float32 for comparison
        r = result.float()
        e = expected.float()

        abs_err = (r - e).abs()
        max_abs = abs_err.max().item()

        # Relative error: |r - e| / max(|e|, 1e-8)
        rel_err = abs_err / (e.abs().clamp(min=1e-8))
        max_rel = rel_err.max().item()

        passed = torch.allclose(r, e, atol=self.atol, rtol=self.rtol)

        return TestResult(
            name=f"{self.name}/{test_name}",
            passed=passed,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            details=f"atol={self.atol}, rtol={self.rtol}"
        )

    def run_all(self, input_generator: Callable,
                shapes: List[Tuple]) -> List[TestResult]:
        """Run all 5 test stages.

        Args:
            input_generator: fn(shape, dtype, device) -> tuple of input tensors
            shapes: list of shapes to test (stage 1 uses first, stage 2 uses all)
        """
        results = []
        device = 'cuda'

        # Stage 1: Basic shapes
        for shape in shapes[:2]:
            inputs = input_generator(shape, self.dtype, device)
            kernel_out = self.kernel_fn(*inputs)
            ref_out = self.reference_fn(*inputs)
            results.append(self._compare(kernel_out, ref_out,
                                         f"basic_{shape}"))

        # Stage 2: Edge cases
        edge_shapes = [(1, 1, 64), (1, 1, 100), (1, 1, 768)]
        for shape in edge_shapes:
            inputs = input_generator(shape, self.dtype, device)
            kernel_out = self.kernel_fn(*inputs)
            ref_out = self.reference_fn(*inputs)
            results.append(self._compare(kernel_out, ref_out,
                                         f"edge_{shape}"))

        # Stage 3: Numerical stability
        shape = shapes[0]
        for scale_name, scale in [("large", 100.0), ("small", 0.001),
                                   ("zeros", 0.0)]:
            inputs = input_generator(shape, self.dtype, device)
            if scale == 0.0:
                inputs = tuple(torch.zeros_like(t) for t in inputs
                              if t.is_floating_point())
            else:
                inputs = tuple(t * scale if t.is_floating_point() else t
                              for t in inputs)
            kernel_out = self.kernel_fn(*inputs)
            ref_out = self.reference_fn(*inputs)
            results.append(self._compare(kernel_out, ref_out,
                                         f"stability_{scale_name}"))

        # Stage 4: Determinism
        inputs = input_generator(shapes[0], self.dtype, device)
        first_out = self.kernel_fn(*inputs)
        deterministic = True
        for _ in range(9):
            repeat_out = self.kernel_fn(*inputs)
            if not torch.equal(first_out, repeat_out):
                deterministic = False
                break
        results.append(TestResult(
            name=f"{self.name}/determinism",
            passed=deterministic,
            max_abs_error=0.0 if deterministic else float('inf'),
            max_rel_error=0.0 if deterministic else float('inf'),
            details="10 identical runs"
        ))

        # Stage 5: Large-scale accumulated error
        shape = shapes[0]
        total_abs_error = 0.0
        n_iters = 100
        for _ in range(n_iters):
            inputs = input_generator(shape, self.dtype, device)
            kernel_out = self.kernel_fn(*inputs)
            ref_out = self.reference_fn(*inputs)
            total_abs_error += (kernel_out.float() - ref_out.float()).abs().mean().item()
        avg_error = total_abs_error / n_iters
        results.append(TestResult(
            name=f"{self.name}/large_scale_{n_iters}iters",
            passed=avg_error < self.atol * 10,
            max_abs_error=avg_error,
            max_rel_error=0.0,
            details=f"avg abs error over {n_iters} random inputs"
        ))

        return results


def print_results(results: List[TestResult]) -> bool:
    """Print test results and return True if all passed."""
    all_passed = True
    print(f"\n{'='*70}")
    print(f"{'Test':<40} {'Status':<8} {'Max Abs Err':<15} {'Max Rel Err':<15}")
    print(f"{'='*70}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if not r.passed:
            all_passed = False
        print(f"{r.name:<40} {status:<8} {r.max_abs_error:<15.2e} {r.max_rel_error:<15.2e}")
    print(f"{'='*70}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed
```

### Example: Testing the SwiGLU Kernel

```python
def swiglu_input_generator(shape, dtype, device):
    B, T, D = shape
    gate = torch.randn(B, T, D, dtype=dtype, device=device)
    up = torch.randn(B, T, D, dtype=dtype, device=device)
    return (gate, up)

def swiglu_reference(gate, up):
    return torch.silu(gate.float()).to(gate.dtype) * up

def swiglu_kernel(gate, up):
    return fused_swiglu(gate, up)

test = KernelCorrectnessTest(swiglu_kernel, swiglu_reference, "SwiGLU")
results = test.run_all(
    swiglu_input_generator,
    shapes=[(8, 512, 2048), (4, 256, 768)]
)
all_ok = print_results(results)
assert all_ok, "SwiGLU correctness tests failed!"
```

---

## 5.7 Building a Benchmark Suite

### Standardized Timing Protocol

Every benchmark must follow this protocol to get reproducible results:

```python
"""benchmark.py -- Standardized kernel benchmark suite."""
import torch
import time
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class BenchmarkResult:
    name: str
    kernel_time_ms: float
    reference_time_ms: float
    speedup: float
    bandwidth_gb_s: float
    bandwidth_utilization: float  # fraction of peak
    flops: Optional[float] = None
    flops_utilization: Optional[float] = None

# RTX 4060 Ti specs (update for your hardware)
PEAK_BANDWIDTH_GB_S = 288.0
PEAK_FP16_TFLOPS = 176.0
PEAK_FP32_TFLOPS = 22.0


def benchmark_kernel(kernel_fn: Callable,
                     reference_fn: Callable,
                     inputs: tuple,
                     name: str,
                     bytes_accessed: int,
                     flops: Optional[int] = None,
                     warmup: int = 50,
                     iterations: int = 200) -> BenchmarkResult:
    """Benchmark a kernel against a reference.

    Args:
        kernel_fn: the CUDA kernel wrapper
        reference_fn: the PyTorch reference
        inputs: tuple of input tensors
        name: human-readable name
        bytes_accessed: total bytes read + written by the kernel
        flops: total floating-point operations (None for memory-bound ops)
        warmup: number of warmup iterations (not timed)
        iterations: number of timed iterations
    """
    # Warmup: fill caches, trigger JIT compilation, stabilize clocks
    for _ in range(warmup):
        kernel_fn(*inputs)
        reference_fn(*inputs)
    torch.cuda.synchronize()

    # Time the kernel
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        kernel_fn(*inputs)
    torch.cuda.synchronize()
    kernel_time = (time.perf_counter() - start) / iterations

    # Time the reference
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        reference_fn(*inputs)
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - start) / iterations

    # Compute metrics
    kernel_time_ms = kernel_time * 1000
    ref_time_ms = ref_time * 1000
    speedup = ref_time / kernel_time if kernel_time > 0 else float('inf')

    # Bandwidth: bytes / time
    bw_gb_s = (bytes_accessed / 1e9) / kernel_time if kernel_time > 0 else 0
    bw_util = bw_gb_s / PEAK_BANDWIDTH_GB_S

    # FLOPS (if compute-bound)
    flops_util = None
    if flops is not None and kernel_time > 0:
        tflops = (flops / 1e12) / kernel_time
        flops_util = tflops / PEAK_FP16_TFLOPS

    return BenchmarkResult(
        name=name,
        kernel_time_ms=kernel_time_ms,
        reference_time_ms=ref_time_ms,
        speedup=speedup,
        bandwidth_gb_s=bw_gb_s,
        bandwidth_utilization=bw_util,
        flops=flops,
        flops_utilization=flops_util,
    )


def roofline_analysis(result: BenchmarkResult, arithmetic_intensity: float):
    """Print roofline analysis for a kernel.

    Args:
        result: benchmark result
        arithmetic_intensity: FLOPS / bytes_accessed
    """
    # Ridge point: where compute ceiling meets bandwidth ceiling
    ridge_point = PEAK_FP16_TFLOPS * 1e12 / (PEAK_BANDWIDTH_GB_S * 1e9)

    print(f"\n--- Roofline Analysis: {result.name} ---")
    print(f"Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPS/byte")
    print(f"Ridge Point:          {ridge_point:.2f} FLOPS/byte")

    if arithmetic_intensity < ridge_point:
        print(f"Classification:       MEMORY BOUND")
        print(f"Bandwidth Used:       {result.bandwidth_gb_s:.1f} / {PEAK_BANDWIDTH_GB_S:.1f} GB/s "
              f"({result.bandwidth_utilization*100:.1f}%)")
        print(f"Optimization target:  Reduce memory traffic (fusion, caching)")
    else:
        print(f"Classification:       COMPUTE BOUND")
        if result.flops_utilization is not None:
            print(f"FLOPS Used:           {result.flops_utilization*100:.1f}% of peak")
        print(f"Optimization target:  Use tensor cores, reduce instruction count")


def print_benchmark_table(results: list):
    """Print a formatted benchmark comparison table."""
    print(f"\n{'='*90}")
    print(f"{'Kernel':<25} {'Kernel(ms)':<12} {'PyTorch(ms)':<12} "
          f"{'Speedup':<10} {'BW(GB/s)':<12} {'BW Util':<10}")
    print(f"{'='*90}")
    for r in results:
        print(f"{r.name:<25} {r.kernel_time_ms:<12.3f} {r.reference_time_ms:<12.3f} "
              f"{r.speedup:<10.2f}x {r.bandwidth_gb_s:<12.1f} {r.bandwidth_utilization*100:<10.1f}%")
    print(f"{'='*90}")
```

### Example: Benchmarking All Kernels

```python
"""run_benchmarks.py -- Benchmark all custom kernels."""
import torch

# Configuration matching GPT-2 124M
B, T, D = 8, 512, 768
V = 50257
H = 12
HEAD_DIM = D // H

results = []

# 1. Fused Residual + RMSNorm
x = torch.randn(B, T, D, dtype=torch.float16, device='cuda')
residual = torch.randn_like(x)
weight = torch.randn(D, dtype=torch.float16, device='cuda')

def ref_res_rmsnorm(x, r, w):
    h = x + r
    rms = (h.float().pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
    return h, (h.float() * rms * w.float()).half()

# Bytes: read x (2*N) + residual (2*N) + weight (2*D) + write hidden (2*N) + output (2*N)
N = B * T * D
bytes_total = 2 * N * 4 + 2 * D + 2 * N * 2  # simplified
results.append(benchmark_kernel(
    lambda x, r, w: fused_residual_rmsnorm(x, r, w),
    lambda x, r, w: ref_res_rmsnorm(x, r, w),
    (x, residual, weight),
    "FusedResRMSNorm",
    bytes_accessed=bytes_total,
))

# 2. SwiGLU
gate = torch.randn(B, T, D * 4, dtype=torch.float16, device='cuda')
up = torch.randn_like(gate)
N_swiglu = gate.numel()
results.append(benchmark_kernel(
    lambda g, u: fused_swiglu(g, u),
    lambda g, u: torch.silu(g) * u,
    (gate, up),
    "FusedSwiGLU",
    bytes_accessed=N_swiglu * 2 * 3,  # read gate + up, write output, 2 bytes each
))

# 3. Rotary Embedding
q = torch.randn(B, H, T, HEAD_DIM, dtype=torch.float16, device='cuda')
cos = torch.randn(T, HEAD_DIM // 2, dtype=torch.float16, device='cuda')
sin = torch.randn(T, HEAD_DIM // 2, dtype=torch.float16, device='cuda')
N_rope = q.numel()
results.append(benchmark_kernel(
    lambda q, c, s: fused_rotary_embedding(q, c, s),
    lambda q, c, s: apply_rotary_emb_pytorch(q, c, s),
    (q, cos, sin),
    "FusedRoPE",
    bytes_accessed=N_rope * 2 * 2 + T * HEAD_DIM * 2,  # read x + write x + cos/sin
))

# 4. Cross-Entropy
logits = torch.randn(B * T, V, dtype=torch.float16, device='cuda')
targets = torch.randint(0, V, (B * T,), device='cuda')
results.append(benchmark_kernel(
    lambda l, t: fused_cross_entropy(l, t),
    lambda l, t: torch.nn.functional.cross_entropy(l.float(), t, reduction='none'),
    (logits, targets),
    "FusedCrossEntropy",
    bytes_accessed=B * T * V * 2 + B * T * 8 + B * T * 4,  # logits + targets + losses
))

print_benchmark_table(results)

# Roofline analysis for each
for r in results:
    # Approximate arithmetic intensity (FLOPS / bytes)
    # Memory-bound ops have low AI
    roofline_analysis(r, arithmetic_intensity=0.5)  # all our kernels are memory-bound
```

### Expected Output

```
==========================================================================================
Kernel                    Kernel(ms)   PyTorch(ms)  Speedup   BW(GB/s)     BW Util
==========================================================================================
FusedResRMSNorm           0.042        0.215        5.12x     198.4        68.9%
FusedSwiGLU               0.031        0.058        1.87x     210.2        73.0%
FusedRoPE                 0.018        0.049        2.72x     185.6        64.4%
FusedCrossEntropy          0.089        0.152        1.71x     178.3        61.9%
==========================================================================================
```

---

## Exercises

### Exercise 1: Fused Bias + GELU Kernel

Write a kernel that computes `GELU(x + bias)` in a single pass. The GELU approximation is:

```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Hints:
- Use the tanh approximation, not the erf version (it is faster on GPU).
- `bias` has shape `[D]` and broadcasts across `[B, T, D]`.
- Use `__tanhf()` for the fast tanh intrinsic.
- Read `x` and `bias` once, write output once.

Verify with the correctness harness from Section 5.6.

### Exercise 2: Fused Bias + SiLU Kernel

Write a kernel that computes `SiLU(x + bias)` in a single pass:

```
SiLU(x + bias) = (x + bias) * sigmoid(x + bias)
```

This appears in models that use SiLU activation with a bias term. Same structure as Exercise 1 but with a different activation function.

### Exercise 3: Backward Pass for SwiGLU

Add a backward pass for the SwiGLU kernel using PyTorch's autograd `Function`:

```python
class FusedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return fused_swiglu(gate, up)  # your CUDA kernel

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        # Compute gradients in PyTorch (not CUDA)
        # d/d(gate) [silu(gate) * up] = up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
        # d/d(up) [silu(gate) * up] = silu(gate)
        sig = torch.sigmoid(gate.float())
        silu_gate = gate.float() * sig
        d_silu = sig * (1.0 + gate.float() * (1.0 - sig))
        grad_gate = (grad_output.float() * up.float() * d_silu).to(gate.dtype)
        grad_up = (grad_output.float() * silu_gate).to(up.dtype)
        return grad_gate, grad_up
```

Test by training a small MLP with your fused SwiGLU for 100 steps and verifying that the loss decreases.

---

## Checkpoint

Before moving to Part 06, verify:

1. **4+ kernels passing correctness**: Fused Residual+RMSNorm, SwiGLU, RoPE, and Cross-Entropy all pass the 5-stage correctness harness.

2. **All kernels show speedup**: Every kernel is faster than the PyTorch reference. If one is not, investigate whether it is memory-bound or compute-bound and whether your implementation is hitting the bandwidth ceiling.

3. **Benchmark suite runs cleanly**: `run_benchmarks.py` produces a table with speedup and bandwidth utilization for all kernels.

4. **You understand roofline position**: For each kernel, you can explain whether it is memory-bound or compute-bound, and what the theoretical maximum speedup is.

```python
# Quick verification script
assert fused_res_rmsnorm_passes_correctness, "Fused Residual+RMSNorm failed"
assert fused_swiglu_passes_correctness, "SwiGLU failed"
assert fused_rope_passes_correctness, "RoPE failed"
assert fused_ce_passes_correctness, "Cross-Entropy failed"
assert fused_res_rmsnorm_speedup > 1.0, "Fused Residual+RMSNorm not faster"
assert fused_swiglu_speedup > 1.0, "SwiGLU not faster"
assert fused_rope_speedup > 1.0, "RoPE not faster"
assert fused_ce_speedup > 1.0, "Cross-Entropy not faster"
print("Part 05 complete. Ready for Part 06: Autokernel.")
```

---

**Next: [Part 06: Autokernel -- Pattern Matching & Kernel Replacement](06_autokernel.md)**
