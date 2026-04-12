# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoKernel is an autonomous AI agent system for GPU kernel optimization and model training on AMD Strix Halo (gfx1151, RDNA 3.5 APU, ROCm 7.12). The kernel agent modifies one kernel file at a time, runs a fixed evaluation harness, and keeps or reverts changes. The `halo_training/` package uses these optimized kernels for fast pretraining (3.05x speedup, 54% MFU).

## Commands
### Note: If the Halo Strix machine is the remote machine, use `run_remote.sh` to run commands on the remote machine with the Halo Strix hardware. Quickly go through the script to understand the workspace path and venv in the remote machine.

```bash
# Setup (uv package manager, Python 3.10+, PyTorch 2.10.0+rocm7.12)
uv sync                          # core deps
uv sync --extra models           # + HuggingFace
uv sync --extra hip              # + ninja for HIP compilation
uv run python scripts/validate_env.py  # validate ROCm + gfx1151

# Kernel optimization loop
uv run python bench.py --kernel kernels/hip/reduce.py --reference reference.py
uv run python profile.py --model models/llama_7b.py
uv run python extract.py
uv run python orchestrate.py

# End-to-end verification (flags: --incremental, --fused-qkv, --torch-compile,
#   --compile-with-kernels, --decode-benchmark, --decode-tokens N, --chart)
uv run python verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512

# Training (Mode A auto for <2B, Mode B for >2B)
python -m halo_training --model models/llama_7b.py --class-name LlamaModel --dataset babylm
python -m halo_training --model models/llama_7b.py --class-name LlamaModel --compile --optimize-kernels
python -m halo_training --model models/llama_7b.py --class-name LlamaModel --smoke  # 200-step validation
```

## Workflow

**Phase A:** Human provides model → `profile.py` → `extract.py` → confirm plan.
**Phase B:** Agent modifies `kernel.py` (single file, HIP_SRC string) → `bench.py` evaluates (5-stage correctness + roofline). `orchestrate.py` schedules by Amdahl's law.
**Phase C:** `verify.py` plugs kernels into model, runs e2e correctness + benchmark.
**Phase D:** `halo_training/` trains with optimized kernels. Mode A (compile, <2B) or Mode B (streaming + checkpointing, >2B).

## Architecture Constraints

- **`kernel.py`** is the ONLY file modified during optimization. Contains `HIP_SRC` string.
- **`reference.py`** and **`bench.py`** are IMMUTABLE.
- **`program.md`** contains full agent instructions for autonomous operation.

## Key Directories

- `kernels/hip/` — 20+ HIP kernel types + `_compile.py` (compilation) + `_torch_ops.py` (torch.library custom ops)
- `models/` — Self-contained model definitions (LLaMA, GPT-2, AMADEUS, TEMPEST, PROMETHEUS, SPECTRAL-HYDRA, RESONANT-LOOP, MAESTRO-PRIMA, DUAL-CORTEX, OBSIDIAN, VIRTUOSO)
- `halo_training/` — Composable training stack (Mode A/B), CLI: `python -m halo_training`
- `autokernel/` — Library API (`autokernel.optimize()`) with pattern matching + kernel replacement
- `knowledge/` — Hardware reference docs (RDNA 3.5 architecture, rocBLAS/hipBLAS reference)

## Hardware

GPU: Radeon 8060S (gfx1151), 40 CUs, wave32, **no MFMA**, ~59.4 TFLOPS FP16. Memory: 128 GB LPDDR5X (~240 GB/s), unified CPU+GPU. CPU: 16 Zen 5 cores, AVX-512. LDS: 64 KB/CU. L2: 6 MB. See `knowledge/amd_rdna35_strix_halo.md` for full reference including rocBLAS/hipBLAS/hipBLASLt (§6).

## Optimization Results (see REPORT.md for full details)

### Kernel Speeds (vs PyTorch)
- **6-16x**: dequantize_int4 (16.3x), fused_residual_add_layernorm (10.7x), prefix_scan (8.4x), dequantize_int8 (8.1x), fused_residual_add_rmsnorm (6.6x)
- **1.5-4x**: rotary_embedding (3.7x), moe_gating (3.5x), rmsnorm (3.3x), fused_bias_silu/gelu (1.9x), cross_entropy (1.8x), silu_gate_mul (1.6x)
- **~1x**: reduce, layernorm, silu, gelu, softmax
- **<0.3x (skip)**: matmul, flash_attention (standard HIP build), fused_mlp, top_k_sampling

### External Kernel Libraries (verified on gfx1151)

| Package | Op | Time | vs Baseline | Backward | Use Case |
|---------|-----|------|------------|----------|----------|
| **causal-conv1d** 1.6.1 | depthwise conv1d | **0.02ms** | 10x vs nn.Conv1d | Yes | All GatedConv in all architectures |
| **mamba-ssm** 2.3.0 | selective scan | **0.32ms** | 5.6x vs our HIP kernel | Yes | Drop-in upgrade for AMADEUS |
| **hybrid_attention** | flash fwd + SDPA bwd | **3.50ms** fwd+bwd | **8.9% faster** than SDPA | Yes | **Best for training** |
| **flash_attn** 2.8.4 (aiter) | attention fwd only | **0.25ms** | 4.2x vs SDPA | Triton bwd slower | Inference/decode |
| **FLA** 0.4.2 | GLA | **1.28ms** | — | Yes | Griffin alternative (Triton) |
| **FLA** 0.4.2 | Retention | **0.77ms** | — | Yes | Fastest FLA recurrence |
| **FLA** 0.4.2 | HGRN | **0.40ms** | — | Yes | Per-dim recurrence (B,T,D) |
| **FLA** 0.4.2 | DeltaNet | **1.60ms** | — | Yes | Most expressive, slowest |
| **scattermoe** 0.3.0 | fused MoE (Triton) | fwd+bwd OK | — | Yes | MoE architectures (CHIMERA, GENIUS-CAVEMAN) |

**Attention backend selection (gfx1151):**
- **Training:** Use `hybrid_flash_sdpa_attention` from `kernels/hip/hybrid_attention.py` — flash_attn forward (0.25ms) + SDPA aten backward with shared logsumexp (2.92ms) = **3.50ms**, 8.9% faster than pure SDPA (3.84ms). Gradient accuracy: max_diff=0.002 (fp16 tolerance).
- **Inference/decode:** Use flash_attn directly (0.25ms forward, 4.2x faster than SDPA).
- **Avoid:** Pure flash_attn for training (Triton backward is 66% slower than SDPA on gfx11).

**Installation:** All packages require source builds with ROCm patches. See `INSTALL_CAUSAL_CONV1D.md`, `INSTALL_MAMBA_SSM.md`, `INSTALL_AITER.md`. Key: replace bare `expf`/`exp2f`/`powf`/`__logf` with `__builtin_` equivalents in device code. Scripts: `scripts/install_causal_conv1d_rocm.sh`, `scripts/install_mamba_ssm_rocm.sh`, `scripts/patch_aiter_ck_rocm.sh`.

**Liger-Kernel 0.7.0:** Tested 2026-04-10 — mostly **incompatible with gfx1151**. FusedLinearCrossEntropyLoss crashes (hipErrorIllegalAddress). RMSNorm crashes (API mismatch with PyTorch 2.10). Only SwiGLU works but our HIP kernel beats it (1.7x vs 1.6x). Not recommended for gfx1151.

### New Fused Kernels (2026-04-10)

| Kernel | Speedup | File | Notes |
|--------|---------|------|-------|
| **fused_mhc_sinkhorn** | **28.5x** | `kernels/hip/fused_mhc_sinkhorn.py` | 3 projections + 20-iter Sinkhorn 4x4 in registers. Exact correctness. |
| **fused_engram_gate_conv** | **7.4x** | `kernels/hip/fused_engram_gate_conv.py` | Gate + gated value + depthwise conv1d. Wired into `models/engram.py`. |
| **fused_ple_gate** | ~3-5x (est.) | `kernels/hip/fused_ple_gate.py` | Linear->GELU->Linear->RMSNorm. Wired into `models/ple.py` (mode "a" only). |
| **chunked_linear_cross_entropy** | Memory opt | `kernels/hip/chunked_linear_cross_entropy.py` | Saves 2-12 GB by chunking LM head+CE. 25% slower backward (recomputes logits). |

**Anti-pattern confirmed:** Engram Variants A (hash+gather+gate) and C (full fusion) were 5-7x SLOWER than PyTorch because they put matmuls inside HIP kernels. On gfx1151 without MFMA, never put matmuls in HIP kernels — let rocBLAS handle them.

### End-to-End Speedups (LlamaModel7B, 5.9B)
- **1.189x** `autokernel.optimize(model, compile=True)` — best overall
- **1.162x** `--torch-compile` alone (not composable with manual kernel replacements)
- **1.053x** `--incremental --fused-qkv` (4 HIP kernels + fused QKV)
- **9.4 tok/s** decode (KV-cache, 106ms/token)

### Training Results
| Model | Config | tok/s | MFU |
|-------|--------|-------|-----|
| LlamaModel 124.7M | compile + autokernel | **43K** | **54%** |
| LlamaModel 124.7M | eager | 14.5K | 17% |
| AMADEUS 243.8M | autokernel + compile + HIP scan | **10.4K** | **26%** |
| AMADEUS 243.8M | eager, chunked scan | 6.5K | 16% |
| AMADEUS 243.8M | sequential scan | 1.3K | 4% |
| Tempest 244.5M | compile + autokernel | **11,049** | **27.3%** |
| Tempest 124.7M | compile + autokernel + fused block | **22,008** | **27.5%** |
| Tempest + MatFormer 244.5M | compile + autokernel | **8,166** | **20.2%** |

AMADEUS best loss: 12.18 (BPB 4.88) after 2 epochs on BabyLM (30.8M tokens, 79 min).

### Hypothesis Architecture Training (2026-04-12, ~170M params, BabyLM 2 epochs)

| Model | Params | Val Loss | tok/s | Notes |
|-------|--------|----------|-------|-------|
| **Amadeus** | 157.7M | **2.90** | 13,203 | Best quality |
| **MaestroPrima** | 157.8M | **2.90** | 12,896 | Conductor adds negligible benefit |
| Tempest | 176.8M | 2.98 | 12,952 | Best pure Griffin |
| Virtuoso | 180.8M | 2.99 | 11,165 | PLE+MatFormer no quality benefit at this scale |
| Prometheus | 174.3M | 3.00 | 13,066 | 2 attention layers don't help |
| SpectralHydra | 176.8M | 3.19 | 10,323 | Decay spectrum needs tuning |
| ResonantLoop | 50.7M | 3.42 | **15,907** | Throughput champion, quality-limited by params |
| DualCortex | 125.2M | 5.44 | 32,426 | **FAILED** — autokernel breaks small dims |
| Obsidian | 124.0M | 5.71 | 34,115 | **FAILED** — autokernel breaks small dims |

See `knowledge/hypothesis_buildout_results.md` for full analysis.

### Compile-Optimized Griffin Block (2026-04-12)
**FusedGriffinBlockPattern** now enabled — replaces TempestBlock with compile-friendly forward:
- Inlines momentum residual (avoids module boundary for compile)
- Plain PyTorch RMSNorm (Inductor fuses element-wise ops)
- Griffin scan via original module (correct autograd through log-domain scan)
- `_use_hip_backward()` guard skips HIP backward during compile tracing

| Config | Before | After | Improvement |
|--------|--------|-------|-------------|
| Tempest124M AK+compile (fwd+bwd only) | 20,184 tok/s | **71,024 tok/s** | **3.52x** |
| Tempest124M AK only (fp16, fwd+bwd) | 16,674 tok/s | **59,500 tok/s** | **3.57x** |
| Tempest124M training (AMP+optim) | ~12,952 tok/s | **22,358 tok/s** | **1.73x** |

**Real training pipeline comparison (AK+compile, BabyLM, batch=16, seq=256):**
| Model | Training tok/s | MFU | Memory |
|-------|---------------|-----|--------|
| LlamaModel 124.7M | **47,864** | 60.3% | 2.6 GB |
| Tempest124M 123.7M | **22,358** | 27.9% | 4.9 GB |

Griffin training gap narrowed from ~3x to **2.14x** vs LlamaModel. Remaining gap: SDPA attention = 1 kernel vs scan = many ops, higher memory from scan intermediates.
See `docs/superpowers/specs/2026-04-12-compile-optimized-griffin-design.md` for design spec.

### PLE + MatFormer Ablation (Tempest base, 10-min runs)
- **PLE Path A:** Best quality (loss 22.65, -1.5% vs base) at 3% throughput cost
- **MatFormer:** Free (+0.2% tok/s), negligible quality cost, enables elastic inference
- **PLE Path B / A+B:** No benefit — drop
- See `knowledge/ple_ablation_results.md` for full results

### Data / Tokenization
- EOS token (50256, `<|endoftext|>`) inserted between documents in `halo_training/data.py`

### Winning Optimization Patterns
1. **Kernel fusion** (6-16x): Fuse 3+ ops into one kernel. Each eliminated intermediate tensor saves 2 memory passes.
2. **Eliminate cast/allocation overhead** (8x): Fuse multi-dtype ops (int8→float→sub→mul→half) into single kernel.
3. **Online algorithms** (1.8x): Fused max+sum with rescaling eliminates a memory pass.
4. **Native fp16 intrinsics** (3.7x): `__hadd2`/`__hmul` match PyTorch rounding. fp32 intermediates + cast-back does not.
5. **Fused GEMM projections** (Griffin w_a+w_i+w_v → single Linear): Saves 2 kernel launches/layer.
6. **Vectorized chunked scan** (no Python loops): Enables torch.compile fusion, +17% on Tempest.

### Anti-Patterns (don't repeat)
- **LDS caching for 2-pass ops**: L2 (6MB) already serves the second read. LDS only helps replacing a *separate kernel launch*.
- **`__shfl_down` when all lanes need result**: Use `__shfl_xor`. `__shfl_down` only gives lane 0 the result.
- **Binary search for top-k**: PyTorch radix sort is fundamentally better (0.25x). Don't retry.
- **Compute-bound kernels without MFMA**: matmul, flash_attention, fused_mlp — can't beat rocBLAS on scalar FMA.
- **fp32 add when reference adds fp16**: `__hadd2(x, r)` matches PyTorch, `__half2float` + add does not.
- **inf/NaN checks under `-ffast-math`**: `-ffinite-math-only` optimizes away `x != x`. Use bit-level: `(__half_as_ushort(h) & 0x7C00) == 0x7C00`.
- **fp32 topk on fp16 softmax**: Use `__hgt` (fp16 comparison) to match PyTorch tie-breaking.
- **`model.to(float16)` destroys complex buffers**: Casts complex64 `freqs_cis` to real. Save/restore complex buffers around dtype casts.
- **Sequential SSM scans**: Use chunked linear recurrence (chunk_size=64), not Python loops or `torch.associative_scan`. 5x faster. See `models/amadeus.py`.
- **autokernel on small hidden dims (d≤256)**: HIP kernel replacements cause training divergence for dual-path models with d_fast=256. Confirmed: both DualCortex and Obsidian train normally in eager mode (val 3.19, 3.49) but diverge completely with autokernel (val >5.4). Run without `--optimize-kernels` or increase to d≥512.
- **Python for-loops in chunked scan**: Causes torch.compile graph breaks. Use vectorized cross-chunk propagation via cumulative products instead (see `models/tempest.py`).
- **Adaptive softmax for training**: 3 tier matmuls is 4% slower than 1 large matmul on memory-bound hardware. Single LM head for training.
- **SSM state explosion**: Init: A_log=log(arange(1,N+1)), dt_proj bias=-4.0, dt clamped [1e-4, 0.5], B/C normalized by max(norm, 1.0).

### rocBLAS / BLAS Optimization

rocBLAS uses Tensile scalar FMA on gfx1151. Can't beat it — shape workloads to help it:
- **Fewer, larger GEMMs.** Fuse QKV into `Linear(d, (nq+2nkv)*hd)`. Fuse gate+up in SwiGLU into `Linear(d, 2*ffn)`.
- **Pad dims to multiples of 128.** Tensile tiles: 64×64, 128×64, 128×128.
- **Strided batched > pointer-batched** for multi-head ops (less overhead on LPDDR5X).
- hipBLASLt env vars (`ROCBLAS_USE_HIPBLASLT=1`, `TENSILE_SOLUTION_SELECTION_METHOD=2`): **tested, no effect on gfx1151**. Tensile scalar FMA is already near-optimal.
- `rocblas-gemm-tune`: exists but ABI-incompatible with system rocBLAS (Tensile init crash). Needs rebuild against `/opt/rocm/core-7.12/lib/librocblas.so.5`.
- See `knowledge/amd_rdna35_strix_halo.md` §6 for full rocBLAS/hipBLAS/hipBLASLt reference.

### aiter HIP Ops on gfx1151

aiter's CK/HIP ops (RMSNorm, RoPE, activation, quantization) **do not build on gfx1151** — their "opus" framework depends on `mfma_adaptor` (CDNA-only). Only aiter's **Triton-based ops** work (flash_attn). Our autokernel HIP kernels remain the best option for fused ops.

## DeepSpeed CPUAdam on ROCm

Offloads optimizer to CPU (AVX-512). Useful for Mode B (>2B models). Not needed for <2B — `AdamW(fused=True)` is faster.

```bash
export LIBRARY_PATH=/opt/rocm/core-7.12/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:$LD_LIBRARY_PATH
```

Setup: JIT build (not `DS_BUILD_CPU_ADAM=1`). Monkey-patch `strip_empty_entries` to filter `None` from `cxx_args()`.

## ROCm/HIP Development Reference

### Debugging Checklist
1. `__launch_bounds__(N)` — N ≤ 1024, smem fits 64KB LDS
2. Indexing overflow — `(long long)row * N` for >2B elements
3. Alignment — `half2` needs 4-byte, `float4` needs 16-byte. Scalar fallback for tails.
4. Dynamic smem — verify smem_bytes ≤ 65536
5. Dtype mismatch — kernel precision ≠ reference precision → large error

### Error Codes
- `hipErrorIllegalAddress` — OOB access. Check indexing near boundaries.
- `hipErrorLaunchFailure` — Kernel crash (assertion, illegal instruction, register overflow).
- `hipErrorInvalidValue` — Bad launch params (grid/block dims, smem size).
- `Timed out after 30s` — Pre-compile: `python -c "from kernels.hip.<name> import _get_module; _get_module()"`

### RDNA 3.5 vs CDNA 3
See `knowledge/amd_rdna35_strix_halo.md` §1 for full comparison table. Key: wave32 (not 64), no MFMA, 240 GB/s (not 5.3 TB/s), 6 MB L2 (not 256 MB), 40 CUs (not 304).

### Profiling
```bash
rocprof --stats python bench.py --kernel <name>              # timing
rocprofv3 --hip-trace --hsa-trace -o trace.csv python bench.py  # counters
# FETCH_SIZE/time ≈ 240 GB/s → memory-bound. ALU_BUSY > 80% → compute-bound.
```

### Compilation
- hipcc takes ~100s per file. Pre-compile before benchmarking. Hash-based caching in `_compile.py`.
- `-fno-fast-math -ffp-contract=off` when exact fp16 rounding matters.
- `_compile.py` auto-prepends ROCm 7.12 compat preamble to all HIP source: `__expf`->`__builtin_expf`, `sqrtf`->`__builtin_sqrtf`, `rsqrtf`/`fmaxf`/`fminf`/`__fdividef` device wrappers, `std::min`/`std::max`. No manual patching of individual kernel files needed.

### Training Target
- **Phase 1:** Beat LFM2.5-350M on standard benchmarks (HellaSwag, ARC, MMLU)
- **Phase 2:** Instruction-tune for on-device Strix Halo assistant
- **Dataset funnel:** smoke -> BabyLM -> GPT-training-small -> Dolma 10B -> Dolma 100B
- See `docs/superpowers/specs/2026-04-10-training-evolution-design.md` for full 5-stage pipeline
