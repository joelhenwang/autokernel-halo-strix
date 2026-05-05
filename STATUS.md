# Current Training Status

> **Read this file before every training launch or decision.**
> Update after every run completes or fails.

---

## CE Kernel Optimization Stack (2026-05-05)

Two-phase rewrite of cross-entropy path on OdinHalo (V=32768, B=4, T=256).

**Phase 1 (`kernel.py`):** Online softmax single-pass CE kernel with fused/tiny modes.
  - Features baked in-kernel: `logit_softcap`, `z_loss`, `ignore_index`, `label_smoothing`
  - `kernel_fn` fast path (forward-only) for bench.py compatibility
  - `ce_full()` entry point with kwargs
  - **Isolated perf (B=4096, V=32768)**: fwd 2.32×, bwd 1.54×, **fwd+bwd 1.66×** vs PyTorch
  - Correctness: 5-stage bench + 16 feature tests all PASS

**Phase 2 (`kernels/hip/chunked_linear_cross_entropy.py` rewritten):**
Avoids materializing `[N, V]` logits tensor by chunked linear+CE+grad flow.
  - Uses Phase 1 HIP kernel per chunk with pre-scaled grad (no bwd multiply pass)
  - fp16 matmul (drops fp32 from prior impl)
  - Supports softcap, label_smoothing, ignore_index, z_loss
  - Tied weight grads handled via PyTorch autograd accumulation
  - CLI flag: `--chunked-ce`
  - Model integration: `FactorizedLMHead.use_chunked_ce` → returns `h_low` in training
  - Gradient parity vs fp32 reference: loss_rel ~1e-7, grad_rel ~1e-3 (fp16 matmul floor)

**Production OdinHalo (batch=4, T=256, V=32768) ablation:**

| Config                                    | tok/s  | Memory  | Speedup |
|-------------------------------------------|--------|---------|---------|
| Baseline (PyTorch CE)                     | 9,807  | 1.93 GB | 1.000×  |
| + Phase 1 HIP CE                          | 9,857  | 1.74 GB | 1.005×  |
| + Phase 2 Chunked CE                      | 11,455 | 1.56 GB | 1.168×  |
| + RoPE fusion (PyTorch CE)                | 11,295 | 1.96 GB | 1.152×  |
| + RoPE + HIP CE  (**best tok/s**)         | 11,704 | 1.76 GB | **1.193×** |
| + RoPE + Chunked CE (**best memory**)     | 11,366 | 1.59 GB | 1.159× (-343 MB) |

**Files:**
- `kernel.py` — rewritten Phase 1 kernel (online softmax + all features)
- `kernels/hip/chunked_linear_cross_entropy.py` — Phase 2 chunked linear+CE
- `models/components/embeddings.py` — FactorizedLMHead gained `forward_hlow()` + `use_chunked_ce` flag
- `models/odin_halo.py` — conditional h_low return + `logit_softcap` attribute
- `halo_training/trainer.py` — chunked_ce wired with softcap/z_loss/label_smoothing passthrough
- `halo_training/cli.py` — `--chunked-ce` opt-in flag, auto-propagates `use_chunked_ce=True`
- `scripts/test_ce_features.py` — 16 feature correctness cases
- `scripts/test_chunked_ce.py` — gradient parity + memory test
- `scripts/test_odin_chunked.py` — end-to-end OdinHalo integration test
- `scripts/ablation_full.py` — full stack ablation
- `scripts/ablation_modes.py` — fused vs tiny mode comparison

---


## Compile × Kernel Ablation + RoPE Bug Fix (2026-05-05 later)

Critical bug fix + comprehensive compile ablation.

### RoPE non-contiguous bug (fixed in `models/components/conv_blocks.py`)
`freqs_cis.real[:T, :pairs].float()` — `.real` on a complex tensor returns a
**non-contiguous view** (complex memory is interleaved [real, imag] fp32 pairs,
so `.real` has stride 2). The HIP `fused_rope_gate_mul` kernel reads this as
contiguous, effectively reading imag values as cos at odd positions → garbled
RoPE rotation. Silent miscompute; all halo models using this path were training
with WRONG positional encoding. Added `.contiguous()` in both HIP call sites.

Isolated RoPE+gate output diff (fixed vs buggy HIP): **max_err = 13.7** (not noise).
Isolated RoPE+gate output diff (fixed HIP vs native): max_err = 1e-3 (fp16 noise).

### Compile investigation results

OdinHalo has 4 graph breaks per HyPEShortConvBlock at default compile:
1. HIP `fused_rope_gate_mul` (wrapped with `@torch.compiler.disable`)
2-4. `causal_conv1d_fn` calls to DaoAILab C++ extension (non-contiguous `out=` tensor)

Added **compile-friendly path** (`HyPEShortConvBlock._compile_friendly` flag) with:
- Native PyTorch RoPE + gate multiply (no custom kernel)
- Manual causal conv via `F.conv1d` (no DaoAILab extension)
- Result: 0 graph breaks, compiles with `fullgraph=True`
- Accessible via `model.compile_zones_friendly()`

**However, 0 breaks did NOT materially speed up compile** — compile-friendly
path runs ~ the same as default compile (both ~1.08× eager at batch=4). Root
cause: HIP kernels are faster than the Inductor-generated triton for their
specific operations, even with graph breaks.

### Batch size sensitivity — MAJOR finding

Compile lift grows dramatically with batch size. At small batches, kernel launch
overhead dominates and compile can't help much. At larger batches, the overhead
is amortized and Inductor's fusion + matmul autotuning pays off.

### Full stack ablation (OdinHalo V=32768, 400 steps, 200 warmup, fused AdamW)

| Config | batch=4 tok/s | batch=16 tok/s | Lift (bs=16) |
|--------|---------------|----------------|--------------|
| Baseline (PyTorch CE, no fusion)            |  9,793 | 11,145 | 1.000× |
| + HIP CE (tiny)                             | 10,053 | 11,549 | 1.036× |
| + HIP CE + RoPE fusion                      | 10,058 | 11,203 | 1.005× |
| + HIP CE + RoPE + Chunked CE                |  9,790 | 10,959 | 0.983× |
| compile + PyTorch CE                        | 10,745 | 14,108 | 1.266× |
| **compile + HIP CE (best tok/s)**           | **11,066** | **14,682** | **1.317×** |
| compile + HIP CE + RoPE fusion              | 11,047 | 14,506 | 1.302× |
| **compile + HIP CE + RoPE + Chunked**       | 10,690 | 14,228 | 1.277× |
| *(best mem at batch=16)*                    | 1.95 GB | **3.89 GB** | (vs 6.60 GB) |

### reduce-overhead mode (CUDA graphs) — best memory configurations

| batch | Config | tok/s | Peak mem | Notes |
|-------|--------|-------|----------|-------|
| 16 | compile default + HIP CE               | 14,682 | 4.83 GB | best tok/s at bs=16 |
| 16 | compile reduce-overhead + HIP CE       | 14,425 | **2.14 GB** | -2.7 GB for 1.8% tok/s loss |
| 16 | compile reduce-overhead + Chunked CE   | 13,933 | **1.67 GB** | absolute lowest memory |
| 32 | compile default + HIP CE               | 13,967 | 9.72 GB | throughput plateaus |
| 32 | compile reduce-overhead + Chunked CE   | 12,736 | 3.65 GB | largest effective batch at low mem |

Throughput plateaus at batch≥16; going to batch=32 doesn't help (GPU saturated).

### reduce-overhead limitations (IMPORTANT)

`TORCH_COMPILE_MODE=reduce-overhead` is **NOT supported with looped models**
(HALO family with `compile_zones`). CUDA graph buffer reuse across Parcae
iterations invalidates saved activations for backward. The trainer detects this
and auto-falls-back to default mode with a warning:

```
WARNING: reduce-overhead is incompatible with looped models
  (buffer reuse across Parcae iterations). Falling back to default.
```

Similarly, `reduce-overhead + --chunked-ce` is unsupported (auto-disabled with
warning). Use case: memory-savings are available via isolated-benchmark reduce-
overhead testing, but for production training, use default compile mode.

For non-looped models (e.g., plain Llama), reduce-overhead works fine.

### Chunk size tuning for ChunkedLinearCrossEntropyLoss

| chunk_size | tok/s (compiled) | Peak mem | Note |
|-----------:|-----------------:|---------:|:-----|
| 128 | 13,731 | 3.86 GB | more python overhead |
| 256 | 14,117 | 3.89 GB | prior default |
| **512** | **14,303** | 3.96 GB | **new default, sweet spot** |
| 1024 | 12,397 | 4.09 GB | matmul shapes start to hurt |
| 4096 | 12,358 | 4.90 GB | loses memory benefit |

## Phase 1 Quick-Wins Results (2026-05-05)

Phase 1 spec: `docs/superpowers/specs/2026-05-05-phase1-quick-wins-design.md`.
Plan: `docs/superpowers/plans/2026-05-05-phase1-quick-wins-plan.md`.

### Summary

| Work item | Status | Measured effect |
|-----------|:------:|:----------------|
| W1 Deep profile tooling | ✓ shipped | `docs/perf/odinhalo-profile-2026-05-05-compile/profiler.md` generated (drives Phase 2) |
| W5 Residual dedup in HyPEShortConvBlock | ✓ shipped | Inductor already CSE'd under compile — neutral throughput |
| W6 DataLoader: `--num-workers` + `non_blocking=True` | ✓ shipped | Neutral (Strix unified memory; pinning+non_blocking ~no-op) |
| W2 Lion optimizer + `--lion` CLI | ✓ shipped | Opt-in flag; smoke test passes |
| W3 Optimizer shootout (4-way) | ✓ shipped | See table below; AdamW wins tok/s, Muon wins final loss |
| W4 `compile(optimizer.step)` experiment | ✓ shipped | No benefit (0.997×); fused AdamW already single-kernel |
| CLion (Cautious Lion, arXiv:2604.14587) | ✓ shipped | Added as part of W3 optimizer suite; per-coord gate default |

### Shootout: OdinHalo V=32768 batch=16, 400 steps (200 warmup, 200 measured)

| Optimizer | tok/s | Peak GB | Init loss | Final loss | Δ loss |
|-----------|------:|--------:|----------:|-----------:|-------:|
| **AdamW (fused)** | **13,991** | 5.67 | 4.751 | **4.181** | −0.570 |
| Muon | 3,958 | 5.47 | 4.474 | **4.072** | −0.402 |
| Lion | 13,695 | 5.44 | 5.314 | 4.311 | −1.003 |
| CLion (per_coord, ν=1e-6) | 13,431 | 5.44 | 5.291 | 4.313 | −0.977 |

**Winner:** AdamW (highest tok/s + reasonable loss).
**Lowest loss:** Muon, but 3.5× slower step (Newton-Schulz iteration).
**Largest loss reduction:** Lion, but higher starting loss (sign-update at LR=3e-5 is coarse).
**CLion vs Lion:** essentially identical at ν=1e-6 (per-coord gate rarely triggers identity
path at this threshold). Paper's claimed generalization advantage needs longer runs + test-set
eval to observe.

### CLion implementation notes (arXiv:2604.14587)

Paper's Algorithm 2 specifies whole-tensor gating: use `sign(c)` iff the minimum non-zero
absolute value of `c` exceeds ν. At scale (OdinHalo 57.6M params, median |c|≈1e-5 per
tensor), **this gate almost never fires** — any single tiny gradient component fails the
check, forcing every tensor through the identity path at Lion's tiny LR → no learning.

Figure 1(d) of the paper illustrates CLion as a per-coordinate active function (identity
for small |c|, sign for larger). We default to this interpretation (`gate_mode="per_coord"`)
and offer `gate_mode="per_tensor"` for paper-faithfulness. The per-coord interpretation is
both useful in practice and consistent with the figure.

Default ν for OdinHalo-scale training: **1e-6** (below typical |c|≈1e-5 so sign fires for
~90%+ of coords). Paper's Theorem 2 threshold of 1.0 is only appropriate if gradients are
well-scaled to O(1); it is not the right value for modern LLM training with GradScaler.

### Regression check (post-Phase-1)

| Config | batch=16 tok/s | Peak GB | vs pre-Phase-1 |
|--------|---------------:|--------:|---------------:|
| compile + HIP CE | 14,577 | 4.83 | −0.7% (within noise) |
| compile + HIP CE + Chunked CE | 14,163 | 3.89 | −0.5% (within noise) |

All 8 halo model variants still pass single-step training test with `--chunked-ce`.

### Next steps

Phase 1 did NOT deliver a throughput lift because the shootout winner (AdamW) was already
the default. The value Phase 1 delivered:

1. **Profile artifact** (W1) for Phase 2 fusion investigation.
2. **Optimizer options** (Lion, CLion) as opt-in flags for future experiments.
3. **Confirmed no regression** from refactors.
4. **Confirmed `compile(optimizer.step)` has no benefit** — document as deferred.

Phase 2 (fusion investigation) should use the W1 profile as its starting point.

---

## Phase 2 Fusion Investigation Results (2026-05-05)

Phase 2 spec: `docs/superpowers/specs/2026-05-05-phase2-fusion-investigation-design.md`.
Plan: `docs/superpowers/plans/2026-05-05-phase2-fusion-investigation-plan.md`.
Summary: `docs/perf/phase2-summary-2026-05-05.md`.

### Summary

Six work items evaluated in ~4 hours. **Zero fusions shipped. Zero regressions.**
Phase 1 already captured all attainable wins on this stack.

| WI | Target (% of Phase 1 wall) | Outcome |
|----|----------------------------|:--------|
| WI1 | `triton_poi_fused__to_copy_mul_transpose_view_{7,8}` (9.1%) | CLOSED — already optimal Inductor fusion of 5 ops (RoPE+cast+QKV gather). Memory-bandwidth-bound. |
| WI2 | `aten::add_` + `aten::copy_` (9.3%) | CLOSED — 67% add_ is autograd weight-grad accumulation; 90% copy_ is input H2D upload. |
| WI3 | `aten::embedding_dense_backward` (4.1%) | DEFERRED — already near 1.3 TB/s bandwidth limit; tied-embedding autograd fusion too risky. |
| WI4 | `Memset (Device)` (4.1%) | CLOSED — framework-internal (rocBLAS scratch, fused_adamw, GradScaler); no user-reachable source. |
| WI5 | `Memcpy HtoD` (4.0%) | CLOSED — all 4 H2D strategies slower than baseline on unified memory. |
| WI6 | Inductor fusion catalog | SHIPPED — 92 unique triton kernels documented (up to 24 ops fused per kernel). |

### Key findings

1. **Inductor fuses aggressively under `compile_zones`:** 92 triton kernels cover
   nearly every elementwise chain in the model. `mul` appears in 81 kernels, `add` in 33.
   Writing custom HIP kernels for patterns already in the catalog would yield no speedup.

2. **Unified-memory H2D wisdom is inverted on Strix Halo.** Prefetching, pinned memory, and
   non-blocking copies all REGRESS throughput by 1-2%. Current `pin_memory=False +
   non_blocking=True` is the local optimum.

3. **Gradient lifecycle alternatives regress.** `set_to_none=True` beats `set_to_none=False`
   and pre-allocated foreach_zero_ by ~2%. Caching allocator already handles this well.

4. **The 4.1% Memsets are invisible to user code.** Every `aten::zero_` reports 0 μs.
   The actual Memsets come from rocBLAS/fused_adamw/GradScaler internals.

5. **Post-Phase-1 the stack is memory-bandwidth-limited at nearly every hot op.**
   Each of the 5% profile entries is already near its shape's theoretical bandwidth ceiling.
   Further gains require CUDA graphs (Phase 3) or batch-size scaling, not more fusion.

### Baseline confirmation

Post-Phase-2 ablation matches Phase 1 (no regressions):

| Config | Batch=16 tok/s | Peak GB |
|--------|--------------:|--------:|
| **compile + HIP CE (best throughput)** | **14,708** | 4.83 |
| compile + HIP CE + RoPE + Chunked CE (best memory) | 14,152 | **3.89** |

### Permanent artifacts delivered

- `docs/perf/inductor-fusion-catalog.md` + `.json` — 92-kernel structured catalog.
- `docs/perf/wi{1,2,3,4,5}-*.md` — per-WI analyses with closure rationales.
- `docs/perf/kernel-bodies-c1.txt` — triton source for the 9.1% kernel.
- `docs/perf/phase2-summary-2026-05-05.md` — consolidated summary.

Reusable tooling:
- `scripts/dump_inductor_output.py`, `parse_inductor_cache.py`, `extract_kernel_body.py`.
- `scripts/profile_shape_calls.py` — shape-annotated per-op profile.
- `scripts/bench_h2d_strategies.py`, `bench_zero_grad.py` — ablation harnesses.

### Next step: Phase 3 (CUDA graphs through Parcae)

Phase 3 starts from confirmed baseline **14,708 tok/s** at `compile + HIP CE`.
Expected lift: 5-15% by eliminating HIP launch overhead via graph capture.

---




### Chunked CE extension to all halo models (session 2)

Extended `use_chunked_ce=True` ctor arg to: VidarHalo, FenrirHalo, TyrHalo, BaldrHalo, ChimeraHalo
(previously only OdinHalo). All 6 halo families + mini variants verified in
`scripts/test_all_models_chunked_train.py`.

### When to use Chunked CE — model-size dependent

Chunked CE overhead (16 chunks × 4 GEMMs = 64 launches per step) is fixed cost.
It becomes worthwhile only when:
- Rest-of-model compute dominates (larger models, more Parcae iterations)
- Memory headroom is tight (bigger batches, longer sequences)

**Measured at batch=16, block=256:**

| Model | PyTorch CE | Chunked CE | Δ tok/s | Δ mem | Recommend |
|-------|-----------:|-----------:|--------:|------:|:----------|
| **OdinHalo** (57M, 3 iter)  | 14,108 | 14,228 | +0.8% | -1.7 GB | ✓ use chunked |
| **VidarHalo** (47M, 2 iter) | 27,334 | 11,207 | **-59%** | -1.4 GB | ✗ skip chunked |

The rule: if your model is already GPU-saturated on compute (OdinHalo at bs=16),
chunked CE frees up memory at ~0% cost. If your model is memory-bound and the
CE path is proportionally large (VidarHalo), chunked CE's Python loop overhead
dominates the savings.

### Key takeaways
1. **For production training use batch=16 + compile + HIP CE (tiny)**.
2. **For memory-constrained setups** (larger batches, longer sequences): add
   `TORCH_COMPILE_MODE=reduce-overhead` for 50%+ memory savings at <2% throughput cost.
3. **For extreme memory savings** (1.67 GB): combine reduce-overhead + `--chunked-ce`.
4. **RoPE HIP fusion and RoPE Inductor fusion are functionally equivalent** when
   compile is on; no need to force HIP fusion under compile.
5. **Fused AdamW** (`torch.optim.AdamW(..., fused=True)`) is +12% at batch=4.
   Already default in `halo_training/optimizer.py`.
6. **Trainer now auto-uses `compile_zones`** for looped models when `--compile` set.
7. **`TORCH_COMPILE_MODE` env var** switches between default/reduce-overhead/max-autotune.

### Scripts (this session)
- `scripts/ablation_final.py` — comprehensive compile × kernel × batch_size
- `scripts/ablation_compile.py` — compile-friendly vs default compile
- `scripts/ablation_compile_modes.py` — mode={default, reduce-overhead, max-autotune}
- `scripts/ablation_optimizer.py` — fused AdamW + grad_clip + batch size
- `scripts/diag_compile.py`, `scripts/diag_compile_v2.py` — graph-break diagnosis
- `scripts/test_compile_friendly_parity.py` — output parity tests
- `scripts/trace_rope_in_block.py` — proved RoPE non-contig bug
- `scripts/mini_rope_repro.py` — isolated RoPE math verification
- `scripts/profile_step.py` — torch.profiler dump showing matmul=68%, AdamW=19%

---


## Active Model

**ODIN-HALO** (`models/odin_halo.py`, class `OdinHalo`) ← **NEW**
- 57.6M unique / ~156M effective params
- d=768, 6 shared layers (5 HyPEShortConv + 1 NoPE-GQA) × 3 Parcae iterations
- HyPE: NoPE attention (content-only) + RoPE on conv gate
- No momentum, iteration skip connections, logit softcap=30
- Tokenizer: `tokenizers/odin-32k/tokenizer.json` (EOS=0, PAD=1, vocab=32768)
- Dataset: `datasets/dolma-10b-odin32k.bin` (6.8B tokens, 13.7 GB, Machine A only)

**VIDAR-HALO** (`models/vidar_halo.py`, class `VidarHalo`)
- 47.0M unique / 95M effective params
- d=768, 4 shared layers (3 ShortConv + 1 MoDA-GQA) × 2 Parcae iterations
- No momentum (direct residuals) — 22% faster than momentum variants
- Tokenizer: `tokenizers/vidar-32k/tokenizer.json` (EOS=0, PAD=1, vocab=32000)
- Dataset: `datasets/dolma-10b-vidar32k.bin` (6.9B tokens, 13.8 GB, both machines)

**Variants:**
- `VidarHalo` — 32K vocab (production)
- `VidarHaloGPT2` — 50257 vocab (for GPT-2 .bin files like stem-crawl-solo.bin)
- `VidarHaloMini` — d=128 smoke test (~1.7M params)

**Secondary: FENRIR-HALO** (`models/fenrir_halo.py`, class `FenrirHalo`)
- 80.8M unique, d=640, 10 shared layers × Parcae mean=3
- Velocity clamp ±8.0 added (fix for NaN at step 30800)
- GPT-2 tokenizer (50257 vocab)

---

## Active Training Runs

| Machine | Model | Dataset | Config | Step | Loss | tok/s | Started | Status |
|---------|-------|---------|--------|------|------|-------|---------|--------|
| — | — | — | — | — | — | — | — | idle |

## Latest Training Run (2026-05-04)

VIDAR-HALO DDP, 32K tokenizer, compile + autokernel, AdamW, lr=0.001, warmup=300, stem-crawl-vidar32k:

| Step | Loss | BPB | tok/s (instantaneous) | Memory |
|------|------|-----|----------------------|--------|
| 400 | 8.50 | 3.41 | ~26,000 | 7.1 GB |

Pre-compile required: `python scripts/precompile_kernels.py --model models/vidar_halo.py --class-name VidarHalo` on each machine before DDP launch.

---

## Best Checkpoints (verified clean)

| Model | Path | Loss/BPB | Dataset | Tokens Seen | Notes |
|-------|------|----------|---------|-------------|-------|
| FenrirHalo | `fenrir_halo_babylm/step_2566.pt` | 3.16 / 1.26 | BabyLM (16M) | ~16M | Clean, well-converged |
| FenrirHalo | `fenrir_halo_dolma/step_15000.pt` | ~4.2 / ~1.7 | dolma-10b (7B) | ~500M | Last clean before NaN zone |
| TyrHaloLight | `tyr_light_e1_v3/step_8299.pt` | 5.08 / 2.10 | stem-crawl (544M) | 544M | Epoch 1 winner vs BALDR |
| BaldrHalo | `baldr_halo_e1_lr002/step_8299.pt` | 6.87 / 2.75 | stem-crawl (544M) | 544M | Lost to TyrHaloLight |
| VidarHalo | `vidar_smoke_32k/step_300.pt` | ~25.8 / ~10.4 | dolma-10b-vidar32k | ~10M | Smoke test only (300 steps, eager) |

---

## Corrupted / Unusable Checkpoints

| Path | Reason |
|------|--------|
| `fenrir_halo_dolma/step_25000.pt` | Trained without velocity clamp; NaN on resume |
| `fenrir_halo_dolma/step_30000.pt` | Same — weights damaged by unclamped velocity |
| `fenrir_halo_dolma/step_105000.pt` | Deep NaN; all steps after 30800 are corrupted |
| `fenrir_halo_dolma_r2/*` | Grad NaN at step 3200; short run, not useful |

---

## Known Issues & Hard Rules

1. **Always use `--optimize-kernels`** for real training. Without it: 10K tok/s. With it: 35K tok/s. The difference is autokernel fusing RMSNorm+SwiGLU+QKV.
2. **Always use .sh scripts** for remote commands: `run_remote.sh` (Machine A), `run_remote_b.sh` (Machine B). Never raw SSH.
3. **bf16 is NOT supported** on gfx1151. Always fp16 or fp32. bf16 is 24% slower, compile crashes.
4. **DDP smoke tests**: `--max-steps 300 --time-budget 20`. Never launch full epoch for smoke testing.
5. **Single-machine smoke tests**: `python -m halo_training ... --smoke` (200 steps built-in).
6. **Large .bin files use memmap** — no bulk np.fromfile for datasets >1GB. Both `data.py` and `train_ddp.py` use zero-copy memmap.
7. **FENRIR checkpoints after step_25000 are corrupted** — trained without velocity clamp. Use step_15000 or earlier.
8. **Custom tokenizer EOS=0** (not 50256). Check tokenizer config before hardcoding EOS.

---

## Throughput Reference

| Config | Model | tok/s | Notes |
|--------|-------|-------|-------|
| Single eager | VidarHalo (47M) | 18,929 | No compile, no autokernel |
| Single compiled (fwd+bwd only) | VidarHalo | 31,362 | No optimizer overhead |
| Single AdamW+compile | VidarHalo | ~16,800 | CE on vocab=50257 was bottleneck |
| DDP AdamW (no autokernel) | VidarHaloGPT2 | 34,541 global | 2 machines, TB4, vocab=50257 |
| DDP AdamW+compile, 32K tok | VidarHalo | ~26,000 global | 2 machines, TB4, no autokernel |
| DDP AdamW+compile+autokernel, 32K | VidarHalo | ~26,000 global | 2 machines, TB4, 7.1GB mem |
| Single compile+autokernel | VidarHalo | 7,100 | Per-machine instantaneous (isolated bench) |
| DDP Muon+compile+autokernel | FenrirHalo (80M) | ~26,000 global\* | Original dolma run (\*35K was inflated cumulative avg) |

**Note:** Prior 63K and 41K numbers were cumulative averages inflated by Inductor warmup ramp. Real instantaneous throughput is ~13K per machine, ~26K global DDP. Metric fixed to instantaneous in train_ddp.py.

### Ablation Throughput (VidarHaloAblation, d=768, 2L×2iter, 30M, single machine)

| Config | tok/s | Memory | Notes |
|--------|-------|--------|-------|
| bs=32 accum=2 eager (no AK) | **10,333** | 7.9 GB | **Use for Tier S** — zero startup cost |
| bs=32 accum=2 eager+AK | 7,200 | 16.9 GB | AK adds chunked CE (30+ min compile) |
| bs=32 accum=2 compiled+AK | 11,104 | 7.9 GB | +7% but 25+ min Inductor compile |

**Tier S config: eager, no compile, no AK. 10.3K tok/s, BabyLM 1ep ≈ 27 min.**

Key findings:
- MTP dropped: 45% throughput cost, no quality evidence at sub-100M scale
- torch.rms_norm replaces HIP RMSNorm: 15.5ms→2.1ms per call (7.5x)
- Depth-reduced (2L) not width-reduced (d=384): same GEMM shapes as production
- ~10K ceiling for d=768 single machine eager — bandwidth bound at 240 GB/s
- Compile/AK only worth it for runs >1 hour (Tier M/V) where compile cost amortizes

### First Ablation Results (Tier S, BabyLM 1ep, d=768 2L×2iter)

| Config | Final Loss | BPB | tok/s | Notes |
|--------|-----------|-----|-------|-------|
| Baseline | 6.73 | 2.698 | 7.2K | With AK (first run) |
| P1a (Polar-Express NS) | 6.72 | 2.693 | 7.2K | -0.01 loss (noise) |

**P1a verdict:** No meaningful gain at screening scale. Keep for Tier M test (may help more with longer training + larger matrices).

---

## Machine Info

| Machine | SSH | Venv | Project | TB4 IP | GPU |
|---------|-----|------|---------|--------|-----|
| A (rank 0) | `run_remote.sh` | `~/Desktop/ai_lab/.venv/` | `~/Desktop/ai_lab/autokernel-halo-strix/` | 10.77.0.1 | gfx1151 |
| B (rank 1) | `run_remote_b.sh` | `~/Desktop/comfyui-rocm7.12/.venv/` | `~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/` | 10.77.0.2 | gfx1151 |

DDP: `GLOO_SOCKET_IFNAME=thunderbolt0`, `MASTER_ADDR=10.77.0.1`, backend=gloo.

---

*Last updated: 2026-05-05*
