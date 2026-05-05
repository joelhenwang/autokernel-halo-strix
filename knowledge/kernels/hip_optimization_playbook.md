---
title: "HIP C++ Optimization Playbook"
domain: kernels
type: reference
status: active
tags: [%hip, %optimization, %playbook, %rdna35]
---

## Optimization Playbook

Work through these tiers roughly in order. Earlier tiers give larger gains with less risk.
Later tiers require more expertise but can unlock the final 10-20%.

### Tier 1: Block Size Tuning

The single most impactful change for most kernels. Block sizes control tile dimensions and
directly affect occupancy, register pressure, and shared memory usage.

**What to try:**
- Sweep BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K through powers of 2: 16, 32, 64, 128, 256.
- For matmul-like kernels, try rectangular tiles (e.g., 128x64 instead of 64x64).
- Larger blocks = more work per thread block = better arithmetic intensity, but higher register pressure.
- Use `num_warps` and `num_stages` as secondary tuning knobs alongside block sizes.

**Typical gains**: 10-50% from finding the right block size vs the default.

### Tier 2: Memory Access Optimization

Once block sizes are tuned, memory is usually the bottleneck.

**Coalescing:**
- Ensure threads in the same warp access consecutive memory addresses.
- For matmul, this means loading along the contiguous dimension (stride-1).
- Transpose one operand if needed to make both loads coalesced.

**Prefetching:**
- Use `tl.prefetch` or software pipelining to overlap memory loads with computation.
- Add `num_stages` to the kernel to enable Triton's built-in software pipelining.
- Typical: `num_stages=3` or `num_stages=4` for matmul.

**L2 Cache Swizzling:**
- Reorder tile indices so neighboring thread blocks access nearby memory.
- Group tiles along the K dimension to maximize L2 cache reuse.

**Shared Memory Bank Conflicts:**
- 32 banks, 4 bytes wide on NVIDIA GPUs. Add 1 element of padding per row.

**Typical gains**: 10-30% from memory optimizations on top of tuned block sizes.

### Tier 3: Compute Optimization

**TF32 and Mixed Precision:**
- Use `tl.dot(a, b, allow_tf32=True)` for matmul accumulation with TF32 inputs.
- Keep accumulators in fp32 for numerical stability.
- Cast results to output dtype only at the end.

**Fused Operations:**
- Fuse elementwise operations (bias add, activation, scaling) into the kernel epilogue.
- Avoid writing intermediate results to global memory.

**Instruction-Level Optimization:**
- Minimize operations in the inner loop. Hoist invariant computations outside.
- Use `tl.where` instead of branches where possible.

**Typical gains**: 5-15% from compute optimizations.

### Tier 4: Advanced Techniques

**Split-K:**
- Decompose the K dimension across multiple thread blocks.
- Helps when M and N are small (not enough parallelism from spatial tiles alone).

**Persistent Kernels:**
- Launch exactly as many thread blocks as there are SMs on the GPU.
- Each block loops over multiple tiles instead of processing just one.
- Eliminates launch overhead and improves L2 cache utilization.

**Autotune:**
- Use `@triton.autotune` with multiple `triton.Config` configurations.
- Let Triton search over block sizes, num_warps, and num_stages.

**Warp Specialization:**
- Assign different warps to different roles (producers vs consumers).

**Register Tiling:**
- Manually control register allocation via constexpr tile sizes.
- Larger register tiles increase ILP but can cause register spilling.

**Typical gains**: 5-20% from advanced techniques, but higher risk.

### Tier 5: Architecture-Specific Optimizations

**H100 (Hopper, SM90):**
- TMA (Tensor Memory Accelerator): hardware-accelerated bulk copies.
- WGMMA (Warp Group Matrix Multiply Accumulate): next-gen tensor core instructions.
- Cluster-level shared memory.

**A100 (Ampere, SM80):**
**AMD Strix Halo (RDNA 3.5, gfx1151):**
- ~40 CUs, wavefront size = 32 (wave32 mode). All wavefront-level reasoning uses 32-wide.
- LPDDR5X: ~120 GB/s unified bandwidth. Nearly ALL kernels are memory-bound.
- No MFMA. Scalar tiled GEMM with LDS reuse for matrix operations.
- LDS: 64 KB per CU. Use smaller tiles than CDNA. Stay under ~32 KB per double-buffered tile pair.
- Block sizes: 128-256 threads (4-8 wavefronts) is the sweet spot.
- `__shfl_down(val, offset)` and `__shfl_xor(val, offset)` -- no mask argument (unlike CUDA).
- Profile with `rocprof --stats` or `rocprofv3`. Key counters: `TCC_HIT/TCC_REQ` (L2 hit rate),
  `SQ_WAVE_CYCLES` (occupancy), `TCP_PENDING_STALL_CYCLES` (mem stall).
- See `knowledge/hardware/amd_rdna35_strix_halo.md` for full hardware reference.

**Typical gains**: 5-15% from architecture-specific tuning.

### Tier 6: Kernel-Specific Tricks

**Matrix Multiplication (matmul):**
- Swizzle tile ordering for L2 reuse.
- Epilogue fusion (bias, activation, scaling).
- Split-K for tall-skinny matrices.

**Softmax:**
- Two-pass online softmax (track running max and sum in one pass).
- Multi-row processing: process multiple rows per thread block.

**LayerNorm / RMSNorm:**
- Welford's online algorithm for numerically stable variance.
- Fuse weight and bias application into the kernel.
- Multi-row processing for better occupancy.

**Flash Attention:**
- Online softmax with running statistics.
- Block-sparse patterns for long sequences.
- Causal masking with early termination.

**Cross Entropy:**
- Online log-sum-exp for numerical stability.
- Fuse with label indexing to avoid materializing the full logit tensor.

**Rotary Embeddings (RoPE):**
- Fuse with Q/K projection.
- Vectorized sin/cos computation.
- Precompute and cache frequency tables.

### Multi-Kernel Optimization Additions

When optimizing multiple kernels in sequence, you gain cross-kernel insights:

- **Shared block sizes**: If BLOCK_SIZE=128 works well for matmul, try 128 for layernorm and attention too.
- **Data layout awareness**: If you change memory layout for one kernel, consider downstream impact.
- **Fusion opportunities**: After individual kernels are optimized, look for fusion opportunities (e.g., matmul + layernorm).
- **Consistent precision strategy**: Use the same precision across kernels to avoid numerical drift.

### Anti-Patterns (Things That Usually Do Not Work)

- **Extremely large block sizes** (512+): Register spill destroys performance.
- **Too many `num_stages`** (>5): Shared memory overflow.
- **Unnecessary `tl.debug_barrier`**: Memory fences serialize execution.
- **Manual unrolling when Triton already unrolls**: Triton's compiler handles constexpr loop unrolling.
- **Premature use of `atomic_add`**: Only use for split-K reductions.
- **Ignoring alignment**: Misaligned loads waste half the bandwidth.
- **Over-complex control flow in inner loops**: Branches inside the K-loop kill performance.

---

## HIP C++ Optimization Playbook (RDNA 3.5 / Strix Halo)

The agent edits raw HIP C++ source embedded as a Python string in `kernel.py`. The `HIP_SRC`
variable contains the kernel code, compiled at runtime via `torch.utils.cpp_extension.load_inline()`
with hipcc on ROCm.

**The kernel contract**: `kernel.py` exports `KERNEL_TYPE`, `BACKEND = "hip"`,
and `kernel_fn()` with the same signature. `bench.py` runs identically.

**Target hardware**: AMD Strix Halo (gfx1151, RDNA 3.5 APU, ~40 CUs, ~120 GB/s LPDDR5X).
See `knowledge/hardware/amd_rdna35_strix_halo.md` for full hardware reference.

**Key constraint**: Nearly ALL kernels are memory-bound on Strix Halo due to ~120 GB/s
bandwidth (vs 5300 GB/s on MI300X). Optimization priority: minimize global memory traffic.

### HIP Tier 1: Thread/Block Configuration

The most impactful change. Thread count and block dimensions directly control occupancy.

**What to try:**
- RDNA 3.5 uses **wave32** (32 threads per wavefront). Block sizes should be multiples of 32.
- Sweep `blockDim.x` through 128, 256, 512. More threads = better latency hiding but higher register pressure.
- Use `dim3` for 2D/3D thread blocks when the kernel has spatial structure (e.g., matmul tiles).
- Set `__launch_bounds__(maxThreadsPerBlock)` to control register allocation.

**Typical gains**: 10-50% from finding the right thread/block config.

### HIP Tier 2: Shared Memory (LDS) Tiling

**Tiling strategy:**
- Load tiles of input from global memory into `__shared__` memory (maps to LDS, 64 KB per CU).
- Process the tile using fast LDS reads.
- Bank-conflict-free layout: add padding (e.g., `__shared__ float tile[32][33]` instead of `[32][32]`).
- Double buffering: use two LDS buffers, load next tile while computing on current.

**Vectorized loads:**
- Use `float4` for 128-bit loads (4x throughput vs scalar `float`).
- Align memory accesses to 128 bytes for maximum coalescing.
- `half2` vectorized loads for FP16 data.

**Typical gains**: 20-40% from LDS tiling over naive global memory access.

### HIP Tier 3: Compute Optimization

RDNA 3.5 does NOT have MFMA matrix cores. Use scalar FMA operations.

**What to try:**
- Register-level tiling: each thread accumulates a 4x4 or 8x8 output tile in registers.
- Keep accumulator in float32, convert to fp16 only at epilogue.
- `#pragma unroll` for inner loops.
- Minimize register pressure to maximize wavefront occupancy.

**Typical gains**: 10-30% from compute optimization.

### HIP Tier 4: Advanced Techniques

**Persistent kernels:**
- Launch exactly `CU_COUNT` thread blocks. Each loops over multiple tiles.
- Eliminates kernel launch overhead and improves L2 reuse.
- Use `atomicAdd` for global synchronization when needed.

**Occupancy tuning:**
- VGPRs per SIMD: 1536. Each wavefront uses (registers_per_thread × 32) VGPRs.
- Max concurrent wavefronts = 1536 / VGPRs_per_wavefront (per SIMD, 2 SIMDs per CU).
- Target 4-8 wavefronts per SIMD for good latency hiding.
- `__launch_bounds__(256)` helps compiler control register allocation.

**Typical gains**: 10-20% from advanced techniques.

### HIP Tier 5: RDNA 3.5 Architecture-Specific

**AMD Strix Halo (gfx1151, RDNA 3.5 APU):**
- Compile: `hipcc -O3 -ffast-math -std=c++17 --offload-arch=gfx1151`
- Wavefront = 32 threads (wave32 mode). All `__shfl_*` use 32-wide, no mask needed.
- No MFMA instructions. Scalar tiled GEMM with LDS for matrix multiply.
- LDS: 64 KB per CU. Use `__launch_bounds__(256)` for occupancy.
- Memory: LPDDR5X unified (shared with CPU), ~120 GB/s peak.
- Nearly everything is memory-bound. Fusion and LDS reuse are critical.
- `__restrict__` on all pointer args to enable compiler optimization.
- Profile: `rocprof --stats`, `rocprofv3`.

**Typical gains**: 5-15% from arch-specific tuning.

### HIP Tier 6: Kernel-Specific Tricks

**Matrix multiplication:**
- Tiled GEMM with double-buffered LDS. Tile sizes: 64x64x32 or smaller.
- Register-level tiling: each thread accumulates a 4x4 output tile.
- Epilogue fusion: add bias, apply activation -- all in registers before writing.

**Softmax:**
- Wavefront-level `__shfl_xor` tree reduction (5 steps for 32 threads).
- Multi-row processing: each thread block handles multiple rows for better occupancy.
- `__expf()` and `__fdividef()` fast math intrinsics.

**LayerNorm / RMSNorm:**
- Welford's online algorithm for single-pass mean+variance.
- Wavefront shuffle cascade (`__shfl_down`) for partial statistics.
- `rsqrtf()` for fast inverse square root.
- Vectorized `float4`/`half2` loads.

**Flash Attention:**
- Double-buffered LDS for Q/K/V tiles.
- Online softmax with running max and sum rescaling.
- Causal mask with early tile termination (skip tiles where all positions are masked).
- Scalar dot products for Q@K^T and attn@V (no matrix cores).

**Cross Entropy:**
- Fused online log-sum-exp: single pass for max, exp-sum, and target lookup.
- Wavefront-level max + sum reductions avoid LDS overhead.
- `__logf()` / `__expf()` fast intrinsics.

**Rotary Embeddings:**
- `__sincosf()` for fused sin/cos computation.
- Vectorized `half2` read-modify-write.

### HIP Anti-Patterns

- **Wavefront divergence in inner loops**: All threads in a wavefront must take the same branch.
- **Uncoalesced global memory access**: Threads in a wavefront must access consecutive addresses.
- **LDS bank conflicts**: 32 banks, 4 bytes each. Pad or swizzle to avoid.
- **Register spilling**: Too many local variables cause spills to slow local memory. Use `__launch_bounds__`.
- **Excessive `__syncthreads()`**: Each sync stalls the entire block. Minimize sync points.
- **Global atomics in hot paths**: Use wavefront-level reduction first, then one atomic per wavefront.
- **Forgetting `__restrict__`**: Without it, the compiler assumes pointers may alias, blocking optimizations.
- **Using `printf` in production kernels**: Serializes execution.
- **Using CUDA sync masks**: HIP shuffles (`__shfl_down`, `__shfl_xor`) do NOT take a mask argument.