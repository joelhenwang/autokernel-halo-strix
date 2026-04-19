---
title: "Training Anti-patterns & Optimization Patterns"
domain: training
type: reference
status: active
related:
  - knowledge/hardware/amd_rdna35_strix_halo.md
  - knowledge/kernels/kernel_benchmarks.md
  - CLAUDE.md
tags: [%antipatterns, %optimization, %patterns, %rocblas, %rocm, %hip, %training, %gfx1151]
---

# Training Anti-patterns & Optimization Patterns

## Winning Optimization Patterns

1. **Kernel fusion** (6-16x): Fuse 3+ ops into one kernel. Each eliminated intermediate tensor saves 2 memory passes.
2. **Eliminate cast/allocation overhead** (8x): Fuse multi-dtype ops (int8->float->sub->mul->half) into single kernel.
3. **Online algorithms** (1.8x): Fused max+sum with rescaling eliminates a memory pass.
4. **Native fp16 intrinsics** (3.7x): `__hadd2`/`__hmul` match PyTorch rounding. fp32 intermediates + cast-back does not.
5. **Fused GEMM projections** (Griffin w_a+w_i+w_v -> single Linear): Saves 2 kernel launches/layer.
6. **Vectorized chunked scan** (no Python loops): Enables torch.compile fusion, +17% on Tempest.

## Anti-Patterns (don't repeat)

- **LDS caching for 2-pass ops**: L2 (6MB) already serves the second read. LDS only helps replacing a *separate kernel launch*.
- **`__shfl_down` when all lanes need result**: Use `__shfl_xor`. `__shfl_down` only gives lane 0 the result.
- **Binary search for top-k**: PyTorch radix sort is fundamentally better (0.25x). Don't retry.
- **Compute-bound kernels without MFMA**: matmul, flash_attention, fused_mlp — can't beat rocBLAS on scalar FMA.
- **fp32 add when reference adds fp16**: `__hadd2(x, r)` matches PyTorch, `__half2float` + add does not.
- **inf/NaN checks under `-ffast-math`**: `-ffinite-math-only` optimizes away `x != x`. Use bit-level: `(__half_as_ushort(h) & 0x7C00) == 0x7C00`.
- **fp32 topk on fp16 softmax**: Use `__hgt` (fp16 comparison) to match PyTorch tie-breaking.
- **`model.to(float16)` destroys complex buffers**: Casts complex64 `freqs_cis` to real. Save/restore complex buffers around dtype casts.
- **Sequential SSM scans**: Use chunked linear recurrence (chunk_size=64), not Python loops or `torch.associative_scan`. 5x faster. See `models/amadeus.py`.
- **autokernel on small hidden dims (d<=256)**: HIP kernel replacements cause training divergence for dual-path models with d_fast=256. Run without `--optimize-kernels` or increase to d>=512.
- **Python for-loops in chunked scan**: Causes torch.compile graph breaks. Use vectorized cross-chunk propagation via cumulative products instead (see `models/tempest.py`).
- **Adaptive softmax for training**: 3 tier matmuls is 4% slower than 1 large matmul on memory-bound hardware. Single LM head for training.
- **SSM state explosion**: Init: A_log=log(arange(1,N+1)), dt_proj bias=-4.0, dt clamped [1e-4, 0.5], B/C normalized by max(norm, 1.0).
- **torch.compile on full looped models**: Python loops with variable depth + no_grad/grad switching cause graph breaks. Compile gives 0% or negative benefit. **Fix:** compile each layer/zone independently (`model.compile_zones()`). JORMUNGANDR-HALO: 14K (eager) → 43K (AK + per-zone compile), 3.07x.
- **Passing kwargs to autokernel-replaced modules**: Autokernel's fused replacements don't accept extra kwargs (e.g., `value_bias`). Guard with `if value_bias is not None and not hasattr(attn, 'w_qkv'): attn(x, freqs, value_bias=value_bias) else: attn(x, freqs)`. Check `hasattr(self.attn, 'w_qkv')` to detect fused replacement.
- **Checkpoints are always fp32**: AMP dtype (fp16/bf16) only exists transiently inside autocast. Checkpoints save fp32 master weights. Safe to load across fp16 (AMD) → bf16 (NVIDIA) training.
- **FiLM cross-dimension mismatch**: Don't apply FiLM modulation (d_target=768) to core loop states (d=512). Compute FiLM context from the core loop, but only apply to Coda layers (d=768). n_film_targets = 4 (Coda only), not 6.
- **Loss reporting with gradient accumulation**: `halo_training` reports loss summed over accum_steps, not averaged. Divide reported loss by accum_steps for actual per-token CE. E.g., reported 24.0 with accum_steps=4 → actual loss 6.0.
- **Muon + QK-Norm ndim crash**: `split_params_for_muon` must use `param.ndim == 2` (not `>= 2`). QK-Norm scales have shape `(n_heads, 1, 1)` — ndim=3 passes `>= 2` check but crashes Newton-Schulz which requires exactly 2D.
- **Don't bypass causal_conv1d_fn under compile**: F.conv1d fallback is slower than the opaque causal_conv1d kernel even with the graph break. The 10x kernel speed outweighs fusion benefit. Tested: -8.1% tok/s.
- **Mamba in loop needs LoopStableMamba3SISO**: Standard Mamba3SISO causes NaN after ~60 steps in a looped architecture. Fix: clamp dt_proj output BEFORE softplus (`clamp(-10, 5)`), tighter dt_max (0.1 vs 0.5), per-iteration RMSNorm.
- **Mamba breaks torch.compile**: autokernel's selective_scan_backward HIP kernel is incompatible with Dynamo tracing. Use compile for Griffin/Conv architectures, not Mamba.
- **d=768 vs d=512 dominates all other factors**: In GRIFFIN-HALO sweep, d=768 core achieves loss ~3.2 while d=512 achieves ~6.1 — nearly 2x difference. Mixer type (Griffin vs GQA), aggregation (DMC vs AttnRes), and adaptive depth all fall within ~3% run-to-run variance at d=768.
- **AttnRes + DMC don't synergize**: Tested in GRIFFIN-HALO sweep. DMC alone: 3.286, AttnRes alone: 3.213, both together: 3.261. Combined is worse than AttnRes alone.
- **CodaAttnRes hurts**: Cross-stage AttnRes in the Coda (3.382) is worse than standard residuals (3.193). Don't add AttnRes to every connection — it only helps for depth aggregation over iteration outputs.
- **Compile fusion doesn't scale with d**: At d=768, compile gives only +18% (vs +140% at d=512 for JORMUNGANDR-HALO). Reason: d=768 GEMMs dominate; compile fuses element-wise ops between GEMMs, which are a smaller fraction of total compute at larger d.
- **Progressive narrowing bridges quality-throughput gap**: One d=768 Griffin iteration → proj_down → d=512 ShortConv×3 refinement iters achieves 33.1K tok/s at loss 5.545 (beats JORMUNGANDR-HALO's 5.770 at 33.7K). The wide first iteration provides global context that makes cheap narrow iterations more effective.
- **Lean architecture viable at d=768**: Cutting Prelude from 2→1 layers and Coda from 4→2 layers gives +21% throughput (25.8K) at only +10.7% quality cost (3.535 vs 3.193). The unique layer overhead (Prelude+Coda) accounts for ~52% of d=768 forward time.

## XSA + Depth Memory Cache (Ablation Results)

Tested on BabyLM 1 epoch, compile + autokernel, Muon, batch=16, block=256:

| Config | Loss | Δ vs Bare | Params added |
|--------|------|-----------|-------------|
| Bare (baseline) | 6.028 | — | — |
| +XSA | 5.973 | -0.9% | 0 |
| +Depth MC (GRM) | 5.879 | -2.5% | 32K |
| +XSA+DC | 5.770 | -4.3% | 32K |
| Full (FiLM+VE+TTT+XSA+DC) | 5.770 | -4.3% | 4.3M |

**XSA** (Exclusive Self Attention, Zhai 2026): removes self-value projection from attention output. Zero params, zero compute. -0.9% alone.
**Depth MC** (Memory Caching GRM, Behrouz et al. 2026): caches loop iteration states, content-dependent gated aggregation. 32K params. -2.5% alone.
**Additive**: XSA+DC combined = -4.3%, nearly exact sum of individual effects. No interference.
**FiLM+VE+TTT add no further benefit** on BabyLM or WikiText-103 at ctx=256 (see below).

### WikiText-103 Comparison (ctx=256, 1 epoch, 119M tokens, resumed from BabyLM)

| Config | Start Loss | Final Loss | BPB | tok/s |
|--------|-----------|------------|-----|-------|
| XSA+DC (99.2M) | 10.529 | **6.563** | 10.520 | 33,722 |
| Full (103.5M) | 12.044 | 6.797 | 10.896 | 33,456 |

XSA+DC wins by -0.234 loss (-3.4%) on WikiText-103 at ctx=256. Full's extra 4.3M params (FiLM+VE+TTT) don't help at short context.

### WikiText-103 Context Length Ablation (1 epoch, 119M tokens, from BabyLM checkpoints)

| Context | Config | Start Loss | Final Loss | tok/s |
|---------|--------|-----------|------------|-------|
| 256 | XSA+DC (99.2M) | 10.529 | **6.563** | 33,722 |
| 256 | Full (103.5M) | 12.044 | 6.797 | 33,456 |
| 1024 | XSA+DC (99.2M) | 10.713 | 6.852 | 34,650 |
| 1024 | Full (103.5M) | 12.057 | **6.805** | 33,880 |

**TTT/FiLM/VE crossover between ctx=256 and ctx=1024.** At 256, XSA+DC wins by -3.4%. At 1024, Full wins by -0.7%. TTT needs sufficient context to adapt — at 256 tokens it's dead weight, at 1024 it has enough material. **Recommendation:** use Full config for ctx≥512, XSA+DC for ctx≤256.

### GPT-Training-Small (2 epochs, ctx=1024, lr=0.0004, 585M tokens)

| Config | Start Loss | Final Loss | tok/s |
|--------|-----------|------------|-------|
| Full (103.5M) | 11.126 | **7.021** | 34,059 |
| XSA+DC (99.2M) | 11.147 | 7.036 | 34,766 |

Full wins by -0.2% — consistent with ctx=1024 trend. Both models at loss ~7.0, still too high for coherent text (need ~3.5-4.0). Total tokens seen: ~730M. ARGUS-PRIME needed 4.7B tokens to reach loss 3.8.

**CPT LR note:** lr=0.0012 caused early plateau at ~7.1 (1 epoch). Reducing to lr=0.0004 + 2 epochs pushed through to 7.0. For continued pre-training, use 3-5x lower LR than pretraining base.

### GriffinHaloProgressive (110.1M, post Parcae injection fix)

Fixed critical bug: `SimpleParcaeInjection` output zero on first loop iteration (A+B=0 when h==input_embed). All prior GRIFFIN-HALO/JORMUNGANDR-HALO runs had dead core loops.

| Stage | Dataset | Tokens | Final CE | tok/s |
|-------|---------|--------|----------|-------|
| BabyLM 1ep | BabyLM | 16M | 6.35 | 34K |
| GPT-small 2ep | GPT-training-small | 585M | 7.06 | 34K |
| WT103 2ep | WikiText-103 | 238M | 6.72 | 35K |

GPT-small CE=7.06 matches JORMUNGANDR Full (7.02) — validates fix. Still above coherence threshold. Needs billions more tokens.

## rocBLAS / BLAS Optimization

rocBLAS uses Tensile scalar FMA on gfx1151. Can't beat it — shape workloads to help it:
- **Fewer, larger GEMMs.** Fuse QKV into `Linear(d, (nq+2nkv)*hd)`. Fuse gate+up in SwiGLU into `Linear(d, 2*ffn)`.
- **Pad dims to multiples of 128.** Tensile tiles: 64x64, 128x64, 128x128.
- **Strided batched > pointer-batched** for multi-head ops (less overhead on LPDDR5X).
- hipBLASLt env vars (`ROCBLAS_USE_HIPBLASLT=1`, `TENSILE_SOLUTION_SELECTION_METHOD=2`): **tested, no effect on gfx1151**. Tensile scalar FMA is already near-optimal.
- `rocblas-gemm-tune`: exists but ABI-incompatible with system rocBLAS. Needs rebuild against `/opt/rocm/core-7.12/lib/librocblas.so.5`.
- See `knowledge/hardware/amd_rdna35_strix_halo.md` section 6 for full rocBLAS/hipBLAS/hipBLASLt reference.

## DeepSpeed CPUAdam on ROCm

Offloads optimizer to CPU (AVX-512). Useful for Mode B (>2B models). Not needed for <2B — `AdamW(fused=True)` is faster.

```bash
export LIBRARY_PATH=/opt/rocm/core-7.12/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:$LD_LIBRARY_PATH
```

Setup: JIT build (not `DS_BUILD_CPU_ADAM=1`). Monkey-patch `strip_empty_entries` to filter `None` from `cxx_args()`.

## ROCm/HIP Development Reference

### Debugging Checklist
1. `__launch_bounds__(N)` — N <= 1024, smem fits 64KB LDS
2. Indexing overflow — `(long long)row * N` for >2B elements
3. Alignment — `half2` needs 4-byte, `float4` needs 16-byte. Scalar fallback for tails.
4. Dynamic smem — verify smem_bytes <= 65536
5. Dtype mismatch — kernel precision != reference precision -> large error

### Error Codes
- `hipErrorIllegalAddress` — OOB access. Check indexing near boundaries.
- `hipErrorLaunchFailure` — Kernel crash (assertion, illegal instruction, register overflow).
- `hipErrorInvalidValue` — Bad launch params (grid/block dims, smem size).
- `Timed out after 30s` — Pre-compile: `python -c "from kernels.hip.<name> import _get_module; _get_module()"`

### RDNA 3.5 vs CDNA 3
See `knowledge/hardware/amd_rdna35_strix_halo.md` section 1 for full comparison table. Key: wave32 (not 64), no MFMA, 240 GB/s (not 5.3 TB/s), 6 MB L2 (not 256 MB), 40 CUs (not 304).

### Profiling
```bash
rocprof --stats python bench.py --kernel <name>              # timing
rocprofv3 --hip-trace --hsa-trace -o trace.csv python bench.py  # counters
# FETCH_SIZE/time ~ 240 GB/s -> memory-bound. ALU_BUSY > 80% -> compute-bound.
```

### Compilation
- hipcc takes ~100s per file. Pre-compile before benchmarking. Hash-based caching in `_compile.py`.
- `-fno-fast-math -ffp-contract=off` when exact fp16 rounding matters.
- `_compile.py` auto-prepends ROCm 7.12 compat preamble to all HIP source: `__expf`->`__builtin_expf`, `sqrtf`->`__builtin_sqrtf`, `rsqrtf`/`fmaxf`/`fminf`/`__fdividef` device wrappers, `std::min`/`std::max`. No manual patching of individual kernel files needed.

### bf16 vs fp16
bf16 (bfloat16) is NOT recommended on gfx1151. AMADEUS bf16 is 24% slower (7.1K vs 9.3K tok/s), uses 32% more memory (12.1 vs 9.2 GB). bf16 + torch.compile crashes on LlamaModel (Inductor can't codegen complex RoPE ops). **Stick with fp16 + GradScaler.**
