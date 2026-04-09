# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoKernel is an autonomous AI agent system for GPU kernel optimization and model training on AMD Strix Halo. The kernel agent modifies one kernel file at a time, runs a fixed evaluation harness, and keeps or reverts changes. This loop runs continuously (40+ experiments/hour) across bottleneck kernels identified via profiling. The `halo_training/` package then uses these optimized kernels for fast pretraining (3.05x speedup, 54% MFU).

Target: AMD Strix Halo (gfx1151, RDNA 3.5 APU) on ROCm 7.12. Backend: HIP C++ compiled via hipcc.

## Setup & Commands

```bash
# Install (uses uv package manager, Python 3.10+)
# PyTorch 2.10.0+rocm7.12
uv sync                          # core dependencies
uv sync --extra models           # + HuggingFace model support
uv sync --extra hip              # + ninja for faster HIP compilation
uv sync --extra kernelbench      # + KernelBench 250+ problems

# Validate environment (ROCm, hipcc, gfx1151 detection)
uv run python scripts/validate_env.py

# Run the benchmark harness on a kernel (the core evaluation loop)
uv run python bench.py --kernel kernels/hip/reduce.py --reference reference.py

# Profile a model to find bottleneck ops
uv run python profile.py --model models/llama_7b.py

# Extract top-N kernels as optimization targets
uv run python extract.py

# Multi-kernel orchestration (Amdahl's law scheduler)
uv run python orchestrate.py

# End-to-end verification after optimization
uv run python verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512

# Incremental kernel-by-kernel verification (applies kernels one at a time, logs cumulative speedup)
# Default order: fused_residual_add_rmsnorm, rmsnorm, silu_gate_mul, rotary_embedding
uv run python verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512 --incremental --atol 0.01 --rtol 0.01

# With fused QKV projection (concatenates wq/wk/wv for better rocBLAS utilization)
uv run python verify.py --model models/llama_7b.py --class-name LlamaModel7B --input-shape 1,512 --incremental --fused-qkv --atol 0.01 --rtol 0.01

# With torch.compile (Inductor backend — 1.16x speedup on 7B, NOT composable with kernel replacements)
uv run python verify.py --model models/llama_7b.py --class-name LlamaModel7B --input-shape 1,512 --torch-compile

# Compile with custom ops (verify.py path)
uv run python verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512 --compile-with-kernels --atol 0.01 --rtol 0.01

# Library API (recommended — best performance, simplest usage)
python -c "
import autokernel
from models.llama_7b import LlamaModel
model = autokernel.optimize(LlamaModel(), compile=True)
# model is now optimized: 1.34x on 170M, 1.19x on 7B
"

# Decode benchmark (autoregressive generation with KV-cache)
uv run python verify.py --model models/llama_7b.py --class-name LlamaModel7B --input-shape 1,128 --decode-benchmark --decode-tokens 32

# Generate progress chart from historical log
uv run python verify.py --chart

# Visualize experiment progress
uv run python analysis.py
```

## Training Commands

```bash
# Train with halo_training (Mode A auto-selected for <2B models, Mode B for >2B)
python -m halo_training --model models/llama_7b.py --class-name LlamaModel --dataset babylm

# With torch.compile + autokernel optimized kernels (best performance: 3.05x speedup)
python -m halo_training --model models/llama_7b.py --class-name LlamaModel \
    --compile --optimize-kernels

# Smoke test (200 steps: checks loss convergence, grad norms, memory, throughput)
python -m halo_training --model models/llama_7b.py --class-name LlamaModel --smoke

# Large model training (Mode B: layer-streaming + activation checkpointing)
python -m halo_training --model models/llama_7b.py --class-name LlamaModel7B \
    --mode B --batch-size 4 --time-budget 90

# Library API (recommended)
python -c "
from halo_training import train
from models.llama_7b import LlamaModel
stats = train(LlamaModel(), dataset='babylm', compile=True, optimize_kernels=True)
"
```

## Three-Phase Workflow

**Phase A (Interactive):** Human provides a model -> `profile.py` identifies bottleneck ops -> `extract.py` creates starter kernels -> human confirms optimization plan.

**Phase B (Autonomous):** Agent iteratively modifies `kernel.py` (single file), `bench.py` evaluates with 5-stage correctness harness (smoke, shapes, stability, determinism, edge cases) + roofline performance. `orchestrate.py` schedules kernels by Amdahl's law impact. Results logged to TSV.

**Phase C (Autonomous):** `verify.py` plugs optimized kernels back into the original model, runs end-to-end correctness + benchmark, reports total speedup.

**Phase D (Training):** `halo_training/` uses optimized kernels for pretraining. Mode A (whole-model compile, <2B) or Mode B (layer-streaming + checkpointing, >2B) auto-selected. Integrates `autokernel.optimize(model, training=True)` for 3.05x training speedup.

## Architecture Constraints

- **`kernel.py`** is the ONLY file the agent modifies during optimization. Contains `HIP_SRC` raw string with HIP C++ source. One kernel at a time, clean diffs, easy reverts.
- **`reference.py`** is IMMUTABLE - contains PyTorch reference implementations (ground truth for correctness).
- **`bench.py`** is IMMUTABLE - fixed evaluation harness, never modified during optimization.
- **`program.md`** contains the full agent instructions for autonomous operation (3-phase workflow, HIP C++ optimization playbook, crash recovery).

## Key Directories

- `kernels/hip/` - 20 HIP C++ kernel types + utilities
  - Original 9: matmul, softmax, layernorm, rmsnorm, flash_attention, fused_mlp, cross_entropy, rotary_embedding, reduce
  - Fusion kernels: fused_residual_add_rmsnorm, fused_residual_add_layernorm, silu_gate_mul, fused_bias_silu
  - Activation kernels: silu, gelu
  - Quantization: dequantize_int8, dequantize_int4
  - Specialized: prefix_scan, moe_gating, top_k_sampling
  - Utilities: `_compile.py` (HIP compilation), `_torch_ops.py` (torch.library custom op registrations for torch.compile)
- `models/` - Self-contained model definitions (LLaMA, GPT-2, BERT) requiring no external deps
- `halo_training/` - Composable training stack (Mode A/B)
  - `trainer.py` (main loop), `streaming.py` (Mode B layer-streaming), `optimizer.py` (COOKBOOK.md param groups + DeepSpeed CPUAdam)
  - `callbacks.py` (PhaseScheduler, MemoryMonitor, StateNormMonitor, PerParamGradMonitor)
  - `metrics.py` (BPB, MFU, TrainingLogger), `smoke.py` (200-step validation), `evaluate.py` (val BPB, decode benchmark)
  - `data.py` (BabyLMDataset), `memory.py` (MemoryBudget, auto mode selection), `model_utils.py` (layer discovery)
  - `cli.py` + `__main__.py` (CLI entry point: `python -m halo_training`)
- `autokernel/` - Library API (`autokernel.optimize()`) with `_registry.py` pattern matching + kernel replacement
- `knowledge/` - Hardware reference docs (RDNA 3.5 Strix Halo architecture, workload guidance)
- `workspace/` - Runtime artifacts (gitignored): profile reports, optimization plans, orchestration state, result logs

## HIP C++ Kernel Pattern

All kernels follow this pattern — the agent modifies the `HIP_SRC` string:
```python
KERNEL_TYPE = "kernel_name"
BACKEND = "hip"
HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
// ... kernel code ...
"""
```
Compiled at runtime via `kernels/hip/_compile.py` which uses `torch.utils.cpp_extension.load_inline()` (hipcc on ROCm).

## Hardware Details

- **GPU**: AMD Radeon 8060S (gfx1151), RDNA 3.5 APU, 40 CUs, 2.9 GHz, ~59.4 TFLOPS FP16
- **CPU**: 16 Zen 5 cores / 32 threads, 5.1 GHz boost, AVX-512, 64 MB L3
- **Memory**: 128 GB soldered LPDDR5X (~240 GB/s), unified with CPU (no PCIe transfers), ~116 GB GPU-visible
- **Wavefront**: 32 threads (wave32 mode)
- **LDS**: 64 KB per CU
- **TDP**: 45–120W configurable
- **Key constraint**: No MFMA matrix cores — nearly all kernels are memory-bound
- See `knowledge/amd_rdna35_strix_halo.md` for full reference

## Benchmark Output

`bench.py` produces greppable plaintext + JSON. Performance is reported as throughput (TFLOPS or GB/s) and percentage of GPU theoretical peak (roofline). Results are appended to `results.tsv` (experiment ID, kernel type, throughput, speedup, correctness status).

## Git Conventions

- Feature branches: `autokernel/<tag>` (e.g., `autokernel/apr7-strix-halo`)
- `results.tsv`, `workspace/`, `CLAUDE.md` are gitignored
- Experiment results documented in `experiments/` directory

## Optimization Results & Lessons (see REPORT.md for full details)

### Current Kernel Speeds (vs PyTorch)
- **16.3x** dequantize_int4, **10.7x** fused_residual_add_layernorm, **8.4x** prefix_scan, **8.1x** dequantize_int8
- **6.6x** fused_residual_add_rmsnorm, **3.7x** rotary_embedding, **3.5x** moe_gating, **3.3x** rmsnorm
- **1.9x** fused_bias_silu, **1.9x** fused_bias_gelu, **1.8x** cross_entropy, **1.6x** silu_gate_mul
- ~1x: reduce, layernorm, silu, gelu, softmax
- <0.3x: matmul, flash_attention (standard build), fused_mlp, top_k_sampling (compute-bound or wrong algorithm — skip these)
- **Note:** Aule-Attention (`pip install aule-attention`, Triton-based) and AOTriton custom builds report 20-30x speedups for attention on gfx1151. See `mad_llm_scientist/COOKBOOK.md` §1.5b for details.

### End-to-End Model Speedups (LlamaModel7B, 5.9B params)
- **1.189x** `autokernel.optimize(model, compile=True)` (library API — best overall)
- **1.161x** `--compile-with-kernels` (verify.py torch.compile + custom ops)
- **1.162x** `--torch-compile` (Inductor alone — not composable with manual kernel replacements)
- **1.053x** `--incremental --fused-qkv` (4 HIP kernels + fused QKV projection)
- **9.4 tok/s** decode throughput (`--decode-benchmark`, KV-cache, 106ms/token)

### Training Results (halo_training, LlamaModel 124.7M)
- **43K tok/s** Mode A + torch.compile + autokernel (3.05x speedup, 54% MFU)
- **14.5K tok/s** Mode A baseline (eager, no autokernel, 17% MFU)
- **853 tok/s** Mode B (2.09B model, layer-streaming, 34.5 GB memory)
- **103 tok/s** decode throughput (7B, KV-cache, prefill 10ms)

### Training Results (AMADEUS 243.8M, SSM hybrid)
- **10.4K tok/s** autokernel + torch.compile(default) + HIP scan kernel (1.75x over eager, 25.6% MFU)
- **7.6K tok/s** autokernel patterns only (1.29x, RMSNorm 3.3x + SwiGLU 1.6x)
- **6.5K tok/s** eager baseline with chunked scan (15.9% MFU, 12.7 GB)
- **1.3K tok/s** sequential/associative scan (4% MFU — DO NOT USE)
- Best loss: 12.18 (BPB 4.88) after 2 epochs on BabyLM (30.8M tokens, 79 min)

### Winning Optimization Patterns (apply in this order)
1. **Kernel fusion** (6-16x): Fuse 3+ PyTorch ops into one kernel. Each eliminated intermediate tensor saves 2 memory passes. This is the #1 lever.
2. **Eliminate cast/allocation overhead** (8x): Operations with multiple dtype casts (int8→float→sub→mul→half) benefit enormously from single-kernel fusion.
3. **Online algorithms** (1.8x): Fused max+sum with rescaling (online softmax) eliminates a memory pass.
4. **Native fp16 intrinsics** (3.7x): `__hadd2`, `__hmul`, `__hsub` match PyTorch rounding exactly. Using fp32 intermediates + cast-back produces different rounding — use native fp16 for element-wise ops.

### Known Anti-Patterns (don't repeat)
- **LDS caching for single-kernel 2-pass ops**: L2 cache (6MB) already serves the second read. LDS caching only helps when it replaces a *separate kernel launch* (fusion), not when both passes are in the same kernel. (Softmax/layernorm LDS attempts regressed or were neutral.)
- **`__shfl_down` when all lanes need the result**: Use `__shfl_xor` instead. `__shfl_down` only gives lane 0 the result — causes silent correctness bugs in online softmax/rescaling algorithms.
- **Binary search for top-k selection**: PyTorch's radix sort is fundamentally better. Our binary search approach was 0.25x. Don't retry.
- **Compute-bound kernels without MFMA**: matmul, flash_attention, fused_mlp cannot beat rocBLAS on scalar FMA. Skip.
- **fp32 addition when reference adds in fp16**: `__hadd2(x, r)` matches PyTorch, `__half2float(x) + __half2float(r)` does not.
- **Float inf/NaN checks under `-ffast-math`**: `-ffast-math` implies `-ffinite-math-only`, which optimizes away `x != x` (NaN check) and `x == INFINITY`. Use **fp16 bit-level checks** instead: `(__half_as_ushort(h) & 0x7C00) == 0x7C00` to detect inf/NaN in inputs. Similarly, use `__float_as_uint` for fp32 bit-level checks.
- **fp32 topk on fp16 softmax output**: `torch.topk` operates on fp16 values. Our fp32 softmax gives tiny differences between experts that are equal in fp16. Use `__hgt` (fp16 comparison) for topk selection to match PyTorch's tie-breaking (smallest index wins for equal fp16 values).
- **`model.to(dtype=float16)` destroys complex buffers**: Casts complex64 (e.g. `freqs_cis` for RoPE) to real fp16, silently discarding the imaginary part (sin values). Save and restore complex buffers around dtype casts. verify.py now handles this automatically.
- **Sequential scan loops for SSMs**: A Python `for t in range(seq_len)` loop for Mamba/SSM scans launches thousands of tiny GPU kernels (~8 per timestep). On gfx1151, this yields only 1.3K tok/s at 4% MFU. PyTorch's `torch._higher_order_ops.associative_scan` is equally slow. Use **chunked linear recurrence** instead (chunk_size=64, cumprod+cumsum within chunks, 8 serial inter-chunk steps for T=512) — gives 5x speedup (6.5K tok/s, 16% MFU). A fused HIP scan kernel (`kernels/hip/selective_scan.py`) gives another 5.9x on the scan component. See `models/amadeus.py` for reference.
- **Adaptive softmax / tiered LM head for training**: Splitting the vocab into frequency tiers (8K full-rank + 16K/26K low-rank) with chunked cross-entropy saves memory (9.7 vs 12.7 GB) but is **4% slower** than a single large matmul for training. On memory-bound hardware without MFMA, rocBLAS handles one large matmul more efficiently than three smaller ones — the tier routing + masked gather overhead dominates. May still help decode (early exit skips weight reads) but not training throughput.
- **SSM state explosion from bad init**: Mamba-style SSMs need careful initialization: A_log=log(arange(1,N+1)), dt_proj bias=-4.0 (softplus(-4)≈0.018), dt clamped to [1e-4, 0.5], B/C normalized by max(norm, 1.0). Without this, state norms grow → NaN within 100 steps.

## DeepSpeed CPUAdam on ROCm (gfx1151)

DeepSpeed CPUAdam offloads optimizer state to CPU using AVX-512. Useful for Mode B (>2B models) where GPU memory is tight. On Strix Halo's unified memory, it trades GPU compute for CPU SIMD.

```bash
# Required environment (add to .bashrc on remote)
export LIBRARY_PATH=/opt/rocm/core-7.12/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:$LD_LIBRARY_PATH
```

**Setup notes:**
- `DS_BUILD_CPU_ADAM=1 pip install deepspeed` does NOT work on this setup. Use JIT build instead (first call takes ~18s, then cached at `~/.cache/torch_extensions/`).
- Monkey-patch required: `deepspeed.ops.op_builder.builder.OpBuilder.strip_empty_entries` must filter `None` values from `cxx_args()` — ROCm builder produces a trailing `None`.
- AVX-512 on Zen 5 is double-pumped (256-bit datapath) but still faster than AVX2. DeepSpeed auto-detects `avx512f`.
- Not needed for <2B models — PyTorch's `AdamW(fused=True)` is faster when optimizer state fits in GPU memory.

## ROCm/HIP Development Reference (adapted from amdpilot-skills for gfx1151)

### HIP Kernel Debugging Checklist
When a HIP kernel crashes or produces wrong results:
1. **Check launch bounds**: `__launch_bounds__(N)` — verify N ≤ 1024, and total shared memory fits in 64KB LDS
2. **Check indexing overflow**: Use `(long long)row * N` for large tensors (int overflow at >2B elements)
3. **Check alignment**: `half2` requires 4-byte alignment, `float4` requires 16-byte alignment. Use scalar fallback for unaligned tails.
4. **Check shared memory sizing**: Dynamic smem (`<<<grid, block, smem_bytes>>>`) — verify smem_bytes ≤ 65536
5. **Verify dtype matching**: If bench.py says max_abs_error >> tolerance, check if kernel computes in different precision than reference

### ROCm Error Codes (common in kernel development)
- `hipErrorIllegalAddress` — Out-of-bounds memory access. Check array indexing, especially with vectorized loads near tensor boundaries.
- `hipErrorLaunchFailure` — Kernel crashed. Usually an assertion, illegal instruction, or stack overflow from too many registers.
- `hipErrorInvalidValue` — Bad kernel launch params. Check grid/block dims, shared memory size.
- `Timed out after 30s` — hipcc compilation timeout in bench.py. Pre-compile with `python -c "from kernels.hip.<name> import _get_module; _get_module()"` before benchmarking.

### RDNA 3.5 (gfx1151) vs CDNA 3 (gfx942) — Key Differences
Most ROCm optimization guides target CDNA (MI300X). Here's what's different for RDNA 3.5:

| Feature | CDNA 3 (MI300X) | RDNA 3.5 (gfx1151) | Impact |
|---------|-----------------|---------------------|--------|
| Wavefront width | 64 threads (wave64) | **32 threads (wave32)** | Warp shuffles use WARP_SIZE=32, warp masks are 32-bit |
| Matrix cores | MFMA (1307 TFLOPS fp16) | **None** | Compute-bound kernels can't compete with rocBLAS |
| Memory | HBM3 (5.3 TB/s) | **LPDDR5X (~240 GB/s)** | 22x less bandwidth — nearly everything is memory-bound |
| L2 cache | 256 MB | **6 MB** | L2 serves repeated reads for data < 4MB; LDS caching only helps beyond this |
| LDS per CU | 64 KB | **64 KB** | Same — LDS caching patterns transfer |
| CU count | 304 | **20** | Much less parallelism — avoid persistent kernels, prefer fewer blocks |
| `__shfl_xor`/`__shfl_down` | 64-lane shuffles | **32-lane shuffles** | Same API, different width |
| Tensor core intrinsics | `__builtin_amdgcn_mfma_*` | **Not available** | Cannot use MFMA; use scalar FMA only |

### Profiling HIP Kernels on gfx1151
```bash
# Basic kernel timing (rocprof v1)
rocprof --stats python bench.py --kernel <name>

# Counter collection (rocprofv3 if available)
rocprofv3 --hip-trace --hsa-trace -o trace.csv python bench.py --kernel <name>

# Key counters for memory-bound analysis:
# - FETCH_SIZE: bytes fetched from memory
# - WRITE_SIZE: bytes written to memory  
# - ALU_BUSY: fraction of ALU cycles active
# - FETCH_LATENCY: average memory fetch latency
# If FETCH_SIZE/kernel_time ≈ peak bandwidth → memory-bound (most of our kernels)
# If ALU_BUSY > 80% → compute-bound (matmul/attention)
```

### Compilation Notes
- hipcc on gfx1151 takes ~100s per kernel file. Always pre-compile before benchmarking.
- bench.py has a 30s timeout that kills compilation. Use: `python -c "from kernels.hip.<name> import _get_module; _get_module()"` first.
- Hash-based caching in `_compile.py` — recompiles only when source changes.
- Use `--offload-arch=gfx1151` (auto-detected by `_compile.py`).
- `-fno-fast-math -ffp-contract=off` when exact fp16 rounding matters (e.g., rotary_embedding).
