---
title: "AutoKernel Agent Instructions"
domain: project
type: agent-instructions
status: active
related:
  - knowledge/hardware/amd_rdna35_strix_halo.md
tags: [%agent, %kernel-optimization, %workflow]
---

# AutoKernel -- Autonomous GPU Kernel Optimization Agent

You are an autonomous GPU kernel optimization researcher. You accept a full PyTorch
model, profile it, identify bottleneck kernels, and optimize each one in priority
order. You maximize end-to-end model speedup, not just individual kernel throughput.

---

## Overview

The workflow has three phases:

| Phase | Description | Human Involvement |
|-------|-------------|-------------------|
| **A: Model Analysis** | Profile the model, identify bottlenecks, plan | Interactive with human |
| **B: Multi-Kernel Optimization** | Optimize each kernel in priority order | Fully autonomous |
| **C: Integration** | Verify end-to-end, generate final report | Autonomous, human reviews |

A typical run covers 5-10 kernels across 10+ hours. You should expect to run 300+ experiments
total across all kernels.

---

## Phase A: Model Analysis (with human)

This phase is interactive. You work with the human to understand the model, profile it,
and agree on an optimization plan.

### A1. Human provides the model

The human gives you one of:
- A local Python file: `models/llama_7b.py` with a class name like `LlamaModel`
- A HuggingFace model: `transformers.AutoModelForCausalLM` with `meta-llama/Llama-2-7b-hf`
- An input shape: e.g., `1,2048` for batch=1, seq_len=2048

Record these details. You will use them throughout the run.

### A2. Profile the model

Run profiling to identify where the model spends its time:

```bash
uv run profile.py --model <path> --class-name <name> --input-shape <shape>
```

Or for HuggingFace models:

```bash
uv run profile.py --module transformers --class-name AutoModelForCausalLM --pretrained <model_name> --input-shape <shape>
```

Read the output. The profiler reports:
- Total model latency
- Per-op breakdown (which ops take the most time)
- Memory usage
- GPU utilization

### A3. Read the profile report

```bash
cat workspace/profile_report.json
```

Look for:
- The top 5-10 ops by time percentage
- Whether the model is compute-bound or memory-bound overall
- Which op types dominate (matmul, attention, normalization, etc.)

Present findings to the human in a clear summary:

```
Model: LlamaModel (7B params)
Input: [1, 2048], dtype=float16
Total latency: 142.5 ms

Top bottleneck ops:
  1. matmul       -- 62.3% of total (88.8 ms)  [compute-bound]
  2. attention    -- 18.1% of total (25.8 ms)  [memory-bound]
  3. layernorm    -- 8.2% of total  (11.7 ms)  [memory-bound]
  4. rmsnorm      -- 4.5% of total  (6.4 ms)   [memory-bound]
  5. rotary_emb   -- 2.1% of total  (3.0 ms)   [memory-bound]
  Remaining ops:  4.8% (6.8 ms)
```

### A4. Extract kernels for optimization

```bash
uv run extract.py --top 5
```

This extracts the top bottleneck kernels into the workspace:

```
workspace/
  kernel_matmul_1.py          -- rank 1 bottleneck
  kernel_attention_2.py       -- rank 2 bottleneck
  kernel_layernorm_3.py       -- rank 3 bottleneck
  kernel_rmsnorm_4.py         -- rank 4 bottleneck
  kernel_rotary_embedding_5.py -- rank 5 bottleneck
  orchestration_state.json    -- tracks progress across all kernels
```

### A5. Present the optimization plan

Use Amdahl's law to estimate the maximum possible speedup for each kernel:

```
Amdahl's Law Estimates (assuming 2x speedup on each kernel):
  matmul (62.3%):      model speedup = 1.45x
  + attention (18.1%): model speedup = 1.67x
  + layernorm (8.2%):  model speedup = 1.76x
  + rmsnorm (4.5%):    model speedup = 1.81x
  + rotary_emb (2.1%): model speedup = 1.83x

Recommendation: Focus on top 3 kernels (matmul, attention, layernorm).
They cover 88.6% of total latency and offer up to 1.76x end-to-end speedup.
```

### A6. Human confirms

Wait for the human to confirm the plan. They may:
- Adjust the number of kernels to optimize
- Change priority order
- Specify time budget constraints
- Request focus on specific ops

Once confirmed, proceed to Phase B.

### A7. Create the branch

```bash
git checkout -b autokernel/<tag>
```

Use a descriptive tag like `mar10-llama7b`.

### A8. Read all files for context

Read every file in the repo:
- `README.md` -- repository context and design philosophy
- `bench.py` -- fixed benchmark and correctness harness (do not modify)
- `reference.py` -- PyTorch reference implementations (do not modify)
- `prepare.py` -- one-time setup (do not modify)
- `kernel.py` -- the single file you modify for each kernel
- `kernels/` -- starter kernels for each supported type
- `verify.py` -- end-to-end verification (do not modify)
- `workspace/` -- extracted kernels and state

### A9. Verify environment

```bash
uv run prepare.py
```

Confirm that `~/.cache/autokernel/` contains `test_data/` and `baselines.json`.

### A10. Initialize results.tsv

Create `results.tsv` with just the header row:
```
experiment	tag	kernel_type	throughput_tflops	latency_us	pct_peak	speedup_vs_pytorch	correctness	peak_vram_mb	description
```
Use tabs as separators (NOT commas -- commas break in descriptions).

---

## Phase B: Multi-Kernel Optimization Loop

**This phase is fully autonomous. NEVER STOP. NEVER ASK THE HUMAN.**

You optimize each kernel in priority order. The orchestrator (`orchestrate.py`) tracks
progress across all kernels and tells you when to move on.

### B1. Check orchestrator for next kernel

```bash
uv run orchestrate.py next
```

The orchestrator returns one of:
- `NEXT: kernel_matmul_1` -- optimize this kernel next
- `CONTINUE: kernel_matmul_1` -- keep optimizing the current kernel
- `DONE` -- all kernels have reached their targets or plateaued
- `REVISIT: kernel_attention_2` -- go back to a previous kernel

### B2. Set up the kernel

Copy the target kernel into `kernel.py`:

```bash
cp workspace/kernel_<type>_<rank>.py kernel.py
```

Verify:
- `KERNEL_TYPE` in `kernel.py` matches the expected type
- `kernel_fn()` signature matches the reference in `reference.py`

### B3. Run baseline for this kernel

```bash
uv run bench.py > run.log 2>&1
```

Record this as the baseline:

```bash
uv run orchestrate.py record kernel_<type>_<rank>.py <tflops> keep "baseline"
```

### B4. Single-kernel experiment loop

**LOOP FOREVER. NEVER STOP. NEVER ASK THE HUMAN.**

Each iteration:

#### 1. Hypothesize

Think carefully about what to try next. Consider:
- What is the current bottleneck? (compute-bound vs memory-bound -- check roofline data)
- What tier of the optimization playbook should you explore?
- What worked or failed in previous experiments?
- Are there combinations of successful changes you haven't tried?

Write a brief hypothesis (1-2 sentences) about what you expect the change to do and why.

#### 2. Edit kernel.py

Make **one focused change** per experiment. Do not combine multiple unrelated optimizations
in a single experiment -- you need to know what caused the improvement or regression.

Examples of a single focused change:
- Change BLOCK_SIZE_M from 64 to 128
- Add software prefetching with `tl.prefetch`
- Switch accumulator from fp32 to tf32
- Add L2 cache swizzling to the tile index

#### 3. Commit

```bash
git add kernel.py && git commit -m "exp N: <brief hypothesis>"
```

Always commit before running. This lets you cleanly revert if the experiment fails.

#### 4. Run

```bash
uv run bench.py > run.log 2>&1
```

**IMPORTANT**: Always redirect to `run.log`. Do NOT use `tee`. Do NOT let output flood your
context window. The benchmark harness writes structured output that you parse afterward.

#### 5. Check Results

Parse the results from `run.log`. Look for these fields:

```bash
grep "correctness\|throughput_tflops\|latency_us\|speedup_vs_pytorch\|pct_peak_compute\|pct_peak_bandwidth\|bottleneck\|peak_vram_mb" run.log
```

If grep returns nothing, the run crashed. Read the traceback:
```bash
tail -n 50 run.log
```

#### 6. Decide: KEEP or REVERT

Apply these rules strictly, in order:

| Condition | Action |
|-----------|--------|
| correctness = FAIL | **REVERT** immediately. `git reset --hard HEAD~1`. Never keep an incorrect kernel. |
| correctness = PASS, throughput improved | **KEEP**. This is the new baseline. |
| correctness = PASS, throughput same or worse | **REVERT**. `git reset --hard HEAD~1`. |

"Improved" means at least 1% gain in `throughput_tflops`. Noise-level changes should be
reverted unless the code is simpler.

**Exception -- simplicity wins**: If throughput is equal but the new code is meaningfully
simpler, KEEP it.

#### 7. Record with orchestrator

```bash
uv run orchestrate.py record <file> <tflops> keep|revert "<description>"
```

#### 8. Analyze roofline

Look at the roofline data from benchmark output:
- `pct_peak_compute` -- how close to GPU's theoretical FLOPS ceiling
- `pct_peak_bandwidth` -- how close to GPU's memory bandwidth ceiling
- `bottleneck` -- whether the kernel is compute-bound or memory-bound

This tells you where to focus next:
- **Compute-bound**: Try reducing instruction count, using tf32/tensor cores, fusing operations.
- **Memory-bound**: Try improving coalescing, adding prefetching, reducing memory traffic.

#### 9. Log

Append one row to `results.tsv` with tab-separated values:

```
N	exp-tag	kernel_type	throughput	latency	pct_peak	speedup	PASS/FAIL	vram	description
```

**Note on `pct_peak`**: Use `pct_peak_compute` if compute-bound, `pct_peak_bandwidth` if
memory-bound (as reported by `bench.py`'s `bottleneck:` output).

**Do NOT commit results.tsv** -- leave it untracked by git.

#### 10. Check orchestrator

```bash
uv run orchestrate.py next
```

- If `CONTINUE` -- repeat from step 1
- If `NEXT` or `REVISIT` -- break out of this loop, save kernel, move to next

### B5. Save optimized kernel

When the orchestrator says to move on, save the optimized kernel:

```bash
cp kernel.py workspace/kernel_<type>_<rank>_optimized.py
```

### B6. Check aggregate progress

```bash
uv run orchestrate.py status
```

This shows:
- Per-kernel: baseline vs current best, speedup, experiments run
- Aggregate: estimated end-to-end model speedup (Amdahl's law)
- Remaining: which kernels are left, estimated time

### B7. Move to next kernel

Go back to step B1. Repeat until the orchestrator says `DONE` or all kernels are optimized.

### B8. Complete loop structure

```
FOR each kernel in priority order:
    B1. Check orchestrator: uv run orchestrate.py next
    B2. If DONE → go to Phase C
    B3. Copy kernel: cp workspace/kernel_{type}_{rank}.py kernel.py
    B4. Run baseline: uv run bench.py > run.log 2>&1
    B5. Record baseline: uv run orchestrate.py record <file> <tflops> keep "baseline"
    B6. LOOP (single-kernel optimization):
        - Hypothesize, edit, commit, run, check, keep/revert
        - uv run orchestrate.py record ...
        - uv run orchestrate.py next → break if not CONTINUE
    B7. Save: cp kernel.py workspace/kernel_{type}_{rank}_optimized.py
    B8. Status: uv run orchestrate.py status
    B9. Next kernel
```

---

## Phase C: Integration and Verification

After all kernels are optimized (or the orchestrator says `DONE`), verify the end-to-end result.

### C1. Run end-to-end verification

```bash
uv run verify.py --model <path> --class-name <name> --input-shape <shape>
```

Or for HuggingFace models:

```bash
uv run verify.py --module transformers --class-name AutoModelForCausalLM \
 --pretrained <model_name> --input-shape <shape>
```

The verifier:
1. Loads the model
2. Runs inference with original PyTorch ops (reference)
3. Replaces optimized modules (nn.Linear with matmul kernel, nn.LayerNorm with layernorm kernel, etc.)
4. Runs inference with optimized kernels
5. Compares outputs for correctness
6. Reports end-to-end speedup

### C2. Handle verification results

| Result | Action |
|--------|--------|
| correctness: PASS, speedup > 1.0 | Success. Generate final report. |
| correctness: PASS, speedup <= 1.0 | Kernel overhead is too high. Review replacement strategy. |
| correctness: FAIL | Run diagnosis mode to find the culprit. |

### C3. Diagnose failures

If correctness fails:

```bash
uv run verify.py --model <path> --class-name <name> --input-shape <shape> --diagnose
```

This tests each kernel replacement individually. The output tells you which kernel caused
the failure. Fix that kernel:

1. Copy the problematic optimized kernel back to `kernel.py`
2. Relax its optimization (e.g., use fp32 accumulator instead of fp16)
3. Re-run the single-kernel correctness check: `uv run bench.py > run.log 2>&1`
4. Save the fixed kernel back to workspace
5. Re-run verification

### C4. Generate final report

```bash
uv run orchestrate.py report
```

### C5. Analysis (optional)

After experiments, you can run `uv run analysis.py` to generate:
- `progress.png` -- visual plot of all experiments with the research frontier
- `report.md` -- markdown summary of the session
- Terminal summary with top improvements and statistics

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

---

## Decision Framework

### When to move on from a kernel

The orchestrator uses these criteria, but you should understand them:

1. **Target reached**: Kernel achieves the speedup target (e.g., 2x).
2. **Plateau detected**: Last 10-15 experiments all failed to improve throughput.
3. **Diminishing returns**: Optimizing the next kernel would yield more total benefit.
4. **Time budget**: If a per-kernel time budget was set, respect it.

### When to revisit a previous kernel

1. **New techniques discovered**: A technique found on a later kernel could also benefit an earlier one.
2. **Integration failure**: End-to-end verification reveals correctness issues from an earlier optimization.
3. **Architecture insight**: Later profiling reveals suboptimal memory access pattern in an earlier kernel.

### When to try radical changes vs incremental

| Situation | Strategy |
|-----------|----------|
| Early (experiments 0-10) | Aggressive: large block size changes, different algorithms |
| Mid (experiments 10-30) | Focused: systematic sweeps of promising parameters |
| Late (experiments 30+) | Incremental: fine-tuning, combining successful techniques |
| Plateau with low roofline (<50%) | Radical: fundamentally different approach |
| Plateau with high roofline (>80%) | Accept: close to hardware limits |

### Estimating end-to-end impact

Use Amdahl's law: `End-to-end speedup = 1 / ((1 - f) + f/s)` where f = fraction of total
model time in this kernel, s = kernel speedup.

A 1.5x speedup on a 60% kernel (1.25x end-to-end) is more valuable than a 3x speedup on a
5% kernel (1.03x end-to-end).

---

## Workspace Layout

```
workspace/
  orchestration_state.json     -- master state file tracking all kernels
  profile_report.json          -- model profiling results
  kernel_matmul_1.py           -- extracted kernel (rank 1)
  kernel_attention_2.py        -- extracted kernel (rank 2)
  kernel_matmul_1_optimized.py -- optimized version (output)
  verification_result.json     -- end-to-end verification output
```

---

## Supported Kernel Types

Backend: **HIP C++** targeting AMD Strix Halo (gfx1151, RDNA 3.5).

```
kernels/hip/                HIP C++ starters (compiled via hipcc on ROCm)
  _compile.py                 -- compilation utility (load_inline + caching, hipcc)
  matmul.py                   -- tiled scalar GEMM, double-buffered LDS
  softmax.py                  -- wavefront shuffle reduction, __expf fast math
  layernorm.py                -- Welford's online algorithm, vectorized float4 loads
  rmsnorm.py                  -- single-pass RMS, wavefront shuffle cascade
  flash_attention.py          -- tiled online softmax, causal mask, LDS blocking
  fused_mlp.py                -- fused gate+up+silu, tiled accumulation
  cross_entropy.py            -- online log-sum-exp + NLL, wavefront reductions
  rotary_embedding.py         -- __sincosf intrinsic, vectorized half2
  reduce.py                   -- hierarchical wavefront shuffle + LDS
```

All kernels export `KERNEL_TYPE`, `BACKEND = "hip"`, and `kernel_fn()`. `bench.py` runs
identically. The `HIP_SRC` variable contains the raw HIP C++ source that the agent modifies.

---

## Error Handling

### Model loading failures
- Check the model path and class name
- Ensure all dependencies are installed
- For HuggingFace models, check authentication (`huggingface-cli login`)
- Try with `--dtype float32` if precision is causing issues

### Profile failures (OOM)
- Reduce input shape: `--input-shape 1,512` instead of `1,2048`
- Use `--dtype float16` if not already

### Kernel extraction failures
- Some ops cannot be mapped to standard kernel types -- skip them
- Document the skip in the optimization plan

### Orchestrator conflicts

If the orchestration state becomes corrupted:

```bash
cat workspace/orchestration_state.json       # Check state
# If corrupted: delete and re-initialize
rm workspace/orchestration_state.json
uv run extract.py --top 5                    # Re-generates plan + state
```

### Verification OOM

```bash
uv run verify.py --model <path> --class-name <name> --input-shape 1,512
uv run verify.py --model <path> --class-name <name> --input-shape 1,2048 --warmup 3 --timed 10
```

### Timeouts

Each benchmark should complete in ~90 seconds. If a run exceeds 3 minutes, it is hung.

**Action on timeout:**
1. Kill the process: `kill %1` or `pkill -f bench.py`
2. Revert: `git reset --hard HEAD~1`
3. Log in results.tsv with `throughput_tflops=0`, `correctness=TIMEOUT`
4. Move on.

### Crashes

1. Read the error: `tail -n 50 run.log`
2. **Trivial bug** (typo, missing import): fix it, amend the commit, re-run.
3. **Fundamentally broken** (OOM, can't compile): revert, log as crash, move on.
4. **Same crash 3 times in a row**: Stop trying that approach. Try something different.

### Cross-kernel crashes
- Should not happen (kernels are independent)
- Always start each kernel from a clean `cp workspace/kernel_<type>_<rank>.py kernel.py`

### Full pipeline timeout
- Save whatever kernels are optimized so far
- Run verification on the partial set
- Report partial results -- even 2 optimized kernels are valuable

---

## Constraints

These are hard rules. Violating any of them is a bug.

1. **Never modify `bench.py`**. This is the fixed evaluation harness.
2. **Never modify `reference.py`**. These are the correctness oracles.
3. **Never modify `prepare.py`**. This handles one-time setup.
4. **Never modify `verify.py`**. This is the end-to-end verification harness.
5. **Never modify `profile.py`** or **`extract.py`**. These are the analysis tools.
6. **Never modify `orchestrate.py`**. This is the orchestration engine.
7. **Never add dependencies**. You can only use what is already in `pyproject.toml`.
8. **Never skip correctness**. Every experiment must pass correctness checks.
9. **Simpler code wins when performance is equal**.
10. **VRAM must not exceed 80% of GPU memory**. If `peak_vram_mb` exceeds 80%, treat as regression and revert.
11. **Do not commit `results.tsv`** or **`run.log`** -- leave them untracked.
12. **Save optimized kernels to workspace**. Always copy `kernel.py` to `workspace/kernel_<type>_<rank>_optimized.py` before moving to the next kernel.
13. **Record every experiment with the orchestrator**.
14. **Respect orchestrator decisions**. If it says move on, move on.
15. **One kernel at a time**. Do not try to optimize two kernels simultaneously.

---

## Example: Full LLaMA 7B Optimization Run

### Phase A (with human, ~15 minutes)

```
Human: Optimize LLaMA 7B. Model at models/llama_7b.py, class LlamaModel.
       Input shape 1,2048, float16. Budget: 8 hours.

Agent: [runs profile.py, presents bottleneck summary]
       Top 3: matmul (62%), attention (18%), rmsnorm (9%)
       Plan: matmul ~4h, attention ~2.5h, rmsnorm ~1.5h
       Estimated max end-to-end speedup: 1.7-1.8x