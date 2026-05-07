# Hardware & Training Constraints

> Single source of truth for all hard constraints. ALWAYS check before writing HIP kernels or launching training.
> Extracted from CLAUDE.md (was 22 bullets buried in a 144L file). Machine-parseable checklist.

## Hardware (gfx1151 / RDNA 3.5)

- [ ] **No MFMA** — can't beat rocBLAS for matmul/attention. Don't put matmuls in HIP kernels.
- [ ] **wave32** not wave64
- [ ] **LDS: 64 KB/CU** — tile sizes must respect this
- [ ] **L2: 6 MB** — data < 4 MB gets near-free repeated reads (fits in L2)

## Data Types & Compute

- [ ] **fp16 + GradScaler** only — NOT bf16 (24% slower, compile crashes)
- [ ] **Unified memory** — gloo matches NCCL for DDP (no GPU-CPU copy penalty)

## Training

- [ ] **Always `--epochs N`, minimum 1** — epochs, not time budgets
- [ ] **torch.compile model only**, never the optimizer (29GB memory blowup)
- [ ] **Autokernel breaks d<=256** — skip `--optimize-kernels` for small hidden dims
- [ ] **EMA (`--ema`)** — free generalization gain (+7.5%), decay 0.999
- [ ] **Checkpoints are fp32** — safe to load across hardware (AMD→NVIDIA)
- [ ] **Final checkpoint always saved** — trainer auto-saves at end of training
- [ ] **Check `train_log.jsonl`** for progress, don't rely on SSH stdout for long runs

## Autokernel / Kernel Fusion

- [ ] **Don't break FusedQKV+RoPE pattern** — manual QKV fusion loses 3.7x speedup
- [ ] **Momentum blocks break FusedResidualRMSNorm** — blocks with `log_beta` have `forward(x, velocity) → (x, velocity)`, incompatible with FusedResidualRMSNorm. Autokernel auto-skips these.
- [ ] **_skip_autokernel flag** — momentum-free blocks set `_skip_autokernel = True`. Sub-module optimizations (FusedQKV, FusedSwiGLU) still apply.
- [ ] **Autokernel before checkpoint load** — fused QKV keys must exist before `load_state_dict()`
- [ ] **Autokernel + value_bias** — detect fused replacement with `hasattr(self.attn, 'w_qkv')` before passing extra kwargs

## Looped Models (Parcae)

- [ ] **Velocity clamp required for fp16 Parcae loops** — `velocity.clamp(-8, 8)` prevents fp16 overflow. Costs ~22% tok/s but prevents NaN.
- [ ] **iter_scales clamp at forward** (2026-05-07+) — OdinHalo/VidarHalo `_apply_iter_norm` clamps `iter_scales[i]` to ±4.0 at forward time. Prevents geometric-compounding drift across Parcae iterations. Parameter is not mutated; checkpoint-compatible both ways.
- [ ] **Hyperloop HC streams too expensive on Strix Halo** — 35-41% throughput cost due to 240 GB/s bandwidth limit. Use `use_mhc=False` (default).
- [ ] **Paracae injection skip first iteration** — `SimpleParcaeInjection`/`ParcaeInjection` cancel to zero when `h == input_embed`. Skip injection on iteration 0.
- [ ] **Per-zone compile for looped models** — compile each ShortConvBlock independently, not the full model. Use `model.compile_zones()`.

## fp16 Stability (long-horizon / multi-epoch / resumed runs)

- [ ] **`--z-loss 1e-4`** recommended for runs > 1 epoch on looped models. Regularizes `logsumexp(logits)²` to prevent logit-magnitude drift. Active first 40% of training.
- [ ] **`--attn-softcap 50.0`** considered for long-context (T ≥ 512) looped runs. ~3% throughput cost; bounds attention scores pre-softmax to prevent fp16 overflow (>65504).
- [ ] **`--activation-monitor`** enabled for runs > 5K steps. Samples per-layer maxabs every 100 steps to `$CKPT_DIR/activation_stats.jsonl`; fp16_headroom < 2 signals imminent overflow.
- [ ] **`--max-grad-norm 0.8`** auto-applied when `--resume-from` is set (can override). Resumed models have larger weights → larger grads.
- [ ] **GradScaler growth_interval = 500** (was 2000, 2026-05-07+). Slows scale runaway across long runs; less chance of outlier grad overflow.
- [ ] **NaN forensics dump** — when `StabilityGuard` triggers, `$CKPT_DIR/nan_dump_step_N.pt` is saved before rollback: offending batch, scaler state, grad history, weight maxabs, activation stats. See `knowledge/training/fp16_stability_gfx1151.md` for diagnostic playbook.
- [ ] **Rollback tightens scaler** (2026-05-07+) — `StabilityGuard.rollback` halves `scaler.growth_interval` in addition to halving LR. Lowers chance of immediate re-overflow.

## Data

- [ ] **EOS token** — GPT-2: id 50256; Vidar-32K: id 0. Check tokenizer config, don't hardcode.
- [ ] **Large .bin datasets use memmap** — `np.memmap` with per-batch `.astype(np.int64)` in `__getitem__`. No bulk memory allocation.

## Monitoring

- [ ] **Loss reporting** — logged loss is accumulated over `accum_steps`. Divide by accum_steps for actual per-token CE.
- [ ] **DDP smoke tests:** `--max-steps 300 --time-budget 20`. Never launch full epoch as a smoke test.
- [ ] **Single-machine smoke tests:** `python -m halo_training ... --smoke` (built-in 200-step test).
- [ ] **After every run completes or fails:** Update STATUS.md.

## Decision Protocol

1. Before ANY training launch: check STATUS.md
2. Before resuming from checkpoint: verify it's listed as "clean" in STATUS.md
3. Always use `--optimize-kernels` for real training (35K vs 10K tok/s)
4. Always use .sh scripts for remote commands, never raw SSH

## Hardware Reference
- Full specs: `knowledge/hardware/amd_rdna35_strix_halo.md`
- Kernel benchmarks: `knowledge/kernels/kernel_benchmarks.md`
- Training anti-patterns: `knowledge/training/training_antipatterns.md`