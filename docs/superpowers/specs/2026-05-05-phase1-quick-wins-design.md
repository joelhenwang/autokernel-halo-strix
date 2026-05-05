# Phase 1: OdinHalo Quick-Wins Throughput Optimization

**Status:** Design
**Date:** 2026-05-05
**Target model:** `models/odin_halo.py` (OdinHalo, 57.6M params, 156M effective)
**Baseline:** 14,682 tok/s at batch=16, 4.83 GB peak (`compile + HIP CE`, per STATUS.md)
**Target after Phase 1:** ~16,500–17,000 tok/s (+10–15%)

## Goal

Extract throughput wins that require no new HIP kernels and no architectural change, while simultaneously producing the profile artifact that feeds Phase 2's fusion investigation.

## Scope

### In-scope

1. **Deep profile** — generate a categorized performance report (`docs/perf/odinhalo-profile-*.md`) that breaks down step time into matmul / norm / elementwise / copy / optimizer / other. Drives Phase 2.
2. **Lion optimizer** — add a new Lion implementation alongside existing AdamW and Muon.
3. **Optimizer shootout** — benchmark AdamW (fused) vs Muon vs Lion on OdinHalo at batch=16, 400 steps. Record throughput, peak memory, and loss trajectory.
4. **compile(optimizer.step) experiment** — measure whether wrapping `optimizer.step` with `torch.compile` improves throughput on PyTorch ≥ 2.5.
5. **Residual dedup** — refactor `HyPEShortConvBlock.forward` to compute `x + conv_out` once rather than twice.
6. **DataLoader tuning** — enable `num_workers=2`, `pin_memory=True`, `non_blocking=True` on H2D copies. Expose `--num-workers` CLI flag.

### Out-of-scope (deferred to later phases)

- New HIP kernels (Phase 2).
- CUDA graphs through Parcae loop (Phase 3).
- Architectural changes to `mean_recurrence`, `backprop_depth`, `d_conv`, or layer count.
- Loss/quality tuning beyond what the optimizer shootout requires.

## Architecture

Phase 1 lands as six independent, individually-measurable changes. Each change is an opt-in flag or a leaf refactor, so regressions can be bisected cleanly. No change is default-on until its isolated ablation shows ≥ 0.5% improvement without quality regression.

### Components and boundaries

**Deep profile tooling** (`scripts/profile_step_deep.py`, new)
- Runs 25 warmup + 5 measured training steps on OdinHalo at batch=16, block=256.
- Uses `torch.profiler` with activities `CPU+CUDA`, schedule `wait=1 warmup=1 active=3 repeat=1`.
- Emits two outputs:
  - Raw `torch.profiler` key-averages table (CPU + CUDA).
  - Categorized Markdown report: each op bucketed by heuristic name match into `matmul`, `norm`, `elementwise`, `copy`, `optimizer`, `attention`, `other`.
- Second script (`scripts/profile_step_rocprof.sh`, new) wraps `rocprof --stats` around the same 30-step run, capturing HIP kernel-level timing.
- Combined output directory: `docs/perf/odinhalo-profile-<date>/` containing `profiler.md`, `rocprof.csv`, `rocprof.md` (generated summary).
- Consumer: Phase 2 investigation framework reads the categorized Markdown to identify fusion candidates.

**Lion optimizer** (`halo_training/lion.py`, new)
- Single file, ~60 lines including docstrings.
- Class `Lion(torch.optim.Optimizer)` with signature `(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)`.
- One exponential-moving-average buffer per parameter (vs AdamW's two).
- Update rule: `update = sign(β1 · m + (1 − β1) · g)`; `p -= lr · (update + weight_decay · p)`; `m = β2 · m + (1 − β2) · g`.
- Integration: `halo_training/optimizer.py::build_optimizer()` gains a `use_lion: bool = False` argument. CLI exposes `--lion`.
- Default LR for Lion in `build_optimizer`: `base_lr * 0.3` (Lion converges with ~3× smaller LR than AdamW). Overridable.

**Optimizer shootout** (`scripts/shootout_optimizers.py`, new)
- Runs three 400-step training sessions (200 warmup, 200 measured) on OdinHalo at batch=16 with the same seed, dataset, and schedule.
- One session each for `--adamw` (baseline), `--muon`, `--lion`.
- Captures per-step loss, tok/s (steady-state), peak memory.
- Emits `docs/perf/optimizer-shootout-<date>.md` with a table and a "winner" paragraph.
- Does **not** change the default optimizer. Changing defaults is a follow-up decision after the shootout.

**`compile(optimizer.step)` experiment**
- Single optional code path in `halo_training/trainer.py`, gated on an env var `TORCH_COMPILE_OPTIMIZER=1` for initial testing.
- Wraps `optimizer.step = torch.compile(optimizer.step, fullgraph=False)` after optimizer construction.
- Required PyTorch version check (`torch.__version__ >= '2.5'`); skip with warning on older versions.
- Measured standalone at batch=16 vs uncompiled fused AdamW.

**Residual dedup** (edit `models/components/conv_blocks.py::HyPEShortConvBlock.forward`)
- Before:
  ```python
  conv_out = self.out_proj(c * z)
  ffn_out = self.ffn(self.ffn_norm(x + conv_out))
  return x + conv_out + ffn_out
  ```
- After:
  ```python
  conv_out = self.out_proj(c * z)
  residual = x + conv_out
  ffn_out = self.ffn(self.ffn_norm(residual))
  return residual + ffn_out
  ```
- Inductor may CSE this under compile, but eager benefits from the explicit reuse and the code reads as intended.

**DataLoader tuning** (edit `halo_training/data.py::build_dataloader`)
- Accept `num_workers: int = 2`, `pin_memory: bool = True` arguments (existing signature already has `num_workers` parameter with default 0).
- Change default `num_workers=0 → 2`.
- Set `pin_memory=True` in the `DataLoader(...)` call.
- `halo_training/trainer.py` input-transfer sites already structured as `batch = next(it); input_ids = input_ids.to(device)` — change to `.to(device, non_blocking=True)` at all 2 sites.
- CLI flag: `--num-workers <int>` exposed via `halo_training/cli.py`.

## Data flow

### Profile report production
```
OdinHalo forward+backward+step (30 iters)
    ├─ torch.profiler → raw event stream → key_averages().table()
    │     → categorize_ops() → docs/perf/odinhalo-profile-<date>/profiler.md
    └─ rocprof --stats (outer wrapper, same run)
          → rocprof_stats.csv
          → summarize_rocprof.py → docs/perf/odinhalo-profile-<date>/rocprof.md
```

### Optimizer shootout
```
for opt in [adamw, muon, lion]:
    instantiate OdinHalo(same_seed)
    run 400 training steps (200 warmup, 200 measured)
    capture tok/s, peak_gb, loss_trajectory
    append row to results table
emit docs/perf/optimizer-shootout-<date>.md
```

## Error handling

- **rocprof absent:** `profile_step_rocprof.sh` checks `command -v rocprof` and exits with a clear "rocprof not installed; skipping kernel-level profile" message. PyTorch-level profile still runs.
- **Lion numeric instability:** if the shootout loss trajectory exceeds 1.5× AdamW's initial loss, mark Lion as "failed" in the shootout report with a loss plot. Do not set it as default.
- **`compile(optimizer.step)` incompatible:** if PyTorch < 2.5 or the compile call raises, skip with a warning. Training continues with uncompiled optimizer.
- **`num_workers > 0` + Windows:** `num_workers > 0` can deadlock on Windows under some Python setups. Document in `--help` that `num_workers=0` is the safe fallback.

## Testing strategy

### Correctness
- **Residual dedup parity**: smoke test `OdinHaloMini` with 200 steps before and after change; assert final loss within 0.1% absolute.
- **Lion correctness**: unit test in `scripts/test_lion.py` that runs 100 steps on a tiny MLP and verifies loss decreases.
- **`num_workers` correctness**: smoke test that 200 steps produce identical loss trajectory with `num_workers=0` vs `num_workers=2` (given `shuffle=False`).

### Performance gates
- Residual dedup must not regress tok/s at batch=16 (allow ±1% noise).
- `compile(optimizer.step)` reported separately; does not block unless it crashes.
- DataLoader change must show ≥ 0% impact on tok/s (not a regression).

## Success criteria

Phase 1 succeeds if **all** of the following hold:

1. `docs/perf/odinhalo-profile-<date>/profiler.md` and `rocprof.md` exist, both summarize ops by category, and totals match measured step time within 5%.
2. `docs/perf/optimizer-shootout-<date>.md` names a winner (or ties AdamW).
3. Residual dedup lands with correctness parity and no tok/s regression.
4. DataLoader tuning lands with no tok/s regression.
5. `compile(optimizer.step)` experiment has a measurement published (outcome can be "no benefit").

End-to-end tok/s improvement is **not** a gating criterion for Phase 1 because the optimizer choice depends on the shootout winner. If AdamW wins, Phase 1 contributes only residual dedup + DataLoader (<2% expected). If Muon or Lion wins, Phase 1 contributes +7–10%.

## Deliverables

**Code:**
- `halo_training/lion.py` (new)
- `halo_training/optimizer.py` (add `use_lion` support)
- `halo_training/trainer.py` (add `TORCH_COMPILE_OPTIMIZER` env var, `.to(non_blocking=True)`)
- `halo_training/cli.py` (add `--lion`, `--num-workers`)
- `halo_training/data.py` (accept `pin_memory`, default `num_workers=2`)
- `models/components/conv_blocks.py` (residual dedup in `HyPEShortConvBlock.forward`)

**Scripts (new):**
- `scripts/profile_step_deep.py`
- `scripts/profile_step_rocprof.sh`
- `scripts/summarize_rocprof.py`
- `scripts/shootout_optimizers.py`
- `scripts/test_lion.py`

**Docs:**
- `docs/perf/odinhalo-profile-<date>/profiler.md`
- `docs/perf/odinhalo-profile-<date>/rocprof.md`
- `docs/perf/optimizer-shootout-<date>.md`
- `STATUS.md` addendum with Phase 1 results table

## Open questions

None at spec time. Decisions deferred to measurements:

- Whether to change default optimizer — decided after shootout.
- Whether to enable `TORCH_COMPILE_OPTIMIZER` by default — decided after measurement.
- Whether to tune the Lion LR ratio further — defer to a follow-up if Lion wins.

## Handoff to Phase 2

Phase 2's investigation framework reads `docs/perf/odinhalo-profile-<date>/profiler.md` as its primary input. If the optimizer shootout shifts the AdamW/Muon/Lion choice, Phase 2 re-uses the profile under the same optimizer; it does not re-profile.
