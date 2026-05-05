# Phase 1 Implementation Plan

**Companion to:** `2026-05-05-phase1-quick-wins-design.md`
**Target:** OdinHalo quick-wins throughput optimization
**Estimated effort:** 2 days (16 engineering hours)

## Approach

Six work items. Five are independent and can land in any order; one (optimizer shootout) depends on Lion being implemented. Each work item lands as its own commit with parity tests passing. Nothing is default-on until its isolated ablation clears the threshold.

## Work items with dependencies

```
W1 Deep profile tooling  ─┐
W2 Lion optimizer        ─┤
W3 Optimizer shootout    ─┘  (needs W2)
W4 compile(optimizer)     (independent)
W5 Residual dedup         (independent)
W6 DataLoader tuning      (independent)
```

Recommended execution order: **W1 → W5 → W6 → W2 → W3 → W4** (profile first because we benchmark everything else against it; W5/W6 are cheapest; W2/W3 are the biggest; W4 is the experimental wildcard).

---

## W1: Deep profile tooling

**Files:**
- `scripts/profile_step_deep.py` (new)
- `scripts/profile_step_rocprof.sh` (new)
- `scripts/summarize_rocprof.py` (new)
- `docs/perf/odinhalo-profile-2026-05-05/` (output directory, created by scripts)

**Steps:**

1. **Create `scripts/profile_step_deep.py`**
   - Builds OdinHalo (full, not Mini) on CUDA.
   - Creates fused AdamW, GradScaler, BabyLMDataset, dataloader batch=16 block=256.
   - Warms 25 steps (matches existing `profile_step.py` pattern).
   - Profiles 5 steps with `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=schedule(wait=1, warmup=1, active=3, repeat=1))`.
   - Emits `docs/perf/odinhalo-profile-2026-05-05/profiler.md` with two sections:
     - Raw `key_averages().table(sort_by="cuda_time_total", row_limit=40)`.
     - Categorized breakdown: a function `categorize_op(name) -> str` maps op names into buckets {matmul, norm, elementwise, copy, optimizer, attention, other} via substring heuristics (`"mm"` → matmul, `"rms"`|`"norm"` → norm, `"mul"`|`"add"`|`"sigmoid"`|`"silu"` → elementwise, `"copy"` → copy, `"adamw"`|`"foreach"` → optimizer, `"flash"`|`"sdpa"`|`"softmax"` → attention, else → other). Sums `self_cuda_time_total` by bucket, renders table with % of wall time.
   - Sanity check at end: category totals must match raw total within 5%.

2. **Create `scripts/profile_step_rocprof.sh`**
   - Bash script. Checks `command -v rocprof` and exits with "rocprof not installed; skipping kernel-level profile" if absent.
   - Otherwise runs `rocprof --stats --hip-trace python scripts/profile_step_deep.py --rocprof-subrun` producing `rocprof_stats.csv`.
   - The `--rocprof-subrun` flag in `profile_step_deep.py` runs without the `torch.profiler` wrapper (to avoid double-profiling overhead).

3. **Create `scripts/summarize_rocprof.py`**
   - Reads `rocprof_stats.csv`.
   - Groups by kernel name prefix (e.g., `Cijk_*` = rocBLAS matmul, `_ZN*cross_entropy*` = our CE, etc.).
   - Emits `docs/perf/odinhalo-profile-2026-05-05/rocprof.md` with top-20 kernels by total time and grouped totals.

4. **Run the tools and commit the profile output**
   - `bash run_remote.sh "cd ... && python scripts/profile_step_deep.py"`
   - `bash run_remote.sh "cd ... && bash scripts/profile_step_rocprof.sh"`
   - Pull generated files back, commit to git.

**Acceptance (W1):**
- Both `.md` files exist and render on GitHub.
- Category totals sum to within 5% of wall time.
- Top 5 op categories named explicitly (matmul %, norm %, etc.).

**Estimated: 3 hours.**

---

## W2: Lion optimizer

**Files:**
- `halo_training/lion.py` (new)
- `halo_training/optimizer.py` (add `use_lion` kwarg + dispatch)
- `halo_training/cli.py` (add `--lion` flag)
- `scripts/test_lion.py` (new, 100-step MLP convergence test)

**Steps:**

1. **Implement `halo_training/lion.py`**
   - Class `Lion(torch.optim.Optimizer)` with `__init__(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)`.
   - `@torch.no_grad()` `step(closure=None)` method:
     ```python
     update = torch.sign(β1 · state['exp_avg'] + (1 − β1) · p.grad)
     p.mul_(1 − lr · wd).add_(update, alpha=-lr)
     state['exp_avg'].mul_(β2).add_(p.grad, alpha=1 − β2)
     ```
   - Support param groups with per-group lr/betas/weight_decay.

2. **Extend `halo_training/optimizer.py::build_optimizer`**
   - Add `use_lion: bool = False, lion_lr_ratio: float = 0.3` kwargs.
   - Before the AdamW dispatch, branch: `if use_lion: return Lion(groups, lr=base_lr * lion_lr_ratio, betas=(0.9, 0.99), weight_decay=weight_decay)`.
   - Mutually exclusive with `use_muon`; raise `ValueError` if both True.

3. **Add CLI flag in `halo_training/cli.py`**
   - `parser.add_argument("--lion", action="store_true", help="Use Lion optimizer (lr scaled by 0.3 vs base_lr)")`.
   - Thread through to `build_optimizer(use_lion=args.lion, ...)`.
   - Update trainer to pass `use_lion=...`.

4. **Write `scripts/test_lion.py`**
   - Tiny MLP (784 → 128 → 10), random batch, MSE loss.
   - Run 100 steps with Lion; assert final loss < 0.1 × initial loss.
   - Run same with AdamW for sanity; assert Lion converges within same order of magnitude.

**Acceptance (W2):**
- `python scripts/test_lion.py` passes.
- `python -m halo_training --model ... --lion --smoke` runs without error and smoke criteria pass.

**Estimated: 2.5 hours.**

---

## W3: Optimizer shootout

**Files:**
- `scripts/shootout_optimizers.py` (new)
- `docs/perf/optimizer-shootout-2026-05-05.md` (output)

**Steps:**

1. **Create `scripts/shootout_optimizers.py`**
   - Shared setup: `torch.manual_seed(42)`, OdinHalo, BabyLMDataset babylm-odin32k, batch=16, block=256, fp16 autocast, compile_zones applied.
   - Three sessions: AdamW (fused=True, lr=1e-4), Muon (default), Lion (lr=3e-5).
   - Each session: 200 warmup + 200 measured steps. Capture `tok/s` (last 100 of measured), `peak_gb`, `loss_per_step[]`.
   - Emit `docs/perf/optimizer-shootout-2026-05-05.md` with:
     - Table: optimizer | tok/s | peak_gb | initial_loss | final_loss | loss_delta.
     - Chart-less loss trajectory for each (first 10 + last 10 values suffice).
     - "Winner" paragraph: selects highest tok/s that also achieves `final_loss < 0.9 × initial_loss` (sanity convergence check).

2. **Run on Machine A**
   - `bash run_remote.sh "cd ... && python scripts/shootout_optimizers.py"`.
   - Pull results file.

3. **Do NOT change default optimizer as part of this PR**
   - The winner becomes the recommended config in STATUS.md but AdamW remains `halo_training/optimizer.py`'s default. Changing the default is a follow-up decision.

**Acceptance (W3):**
- Results file committed with a named winner.
- STATUS.md `Throughput Reference` section gains a row for the winner at OdinHalo batch=16.

**Estimated: 2 hours (wall clock longer due to three training runs).**

---

## W4: compile(optimizer.step) experiment

**Files:**
- `halo_training/trainer.py` (gate on `TORCH_COMPILE_OPTIMIZER` env var)
- `scripts/ablation_compile_optimizer.py` (new)

**Steps:**

1. **Edit `halo_training/trainer.py`**
   - After `optimizer = build_optimizer(...)`, add:
     ```python
     if os.environ.get("TORCH_COMPILE_OPTIMIZER") == "1":
         if torch.__version__ >= "2.5":
             optimizer.step = torch.compile(optimizer.step, fullgraph=False)
             print("Compiled optimizer.step (TORCH_COMPILE_OPTIMIZER=1)")
         else:
             print(f"WARNING: TORCH_COMPILE_OPTIMIZER requires PyTorch >= 2.5 (got {torch.__version__})")
     ```

2. **Create `scripts/ablation_compile_optimizer.py`**
   - Same setup as `ablation_final.py`. Two rows: `baseline compile + HIP CE` vs `same + TORCH_COMPILE_OPTIMIZER=1`.
   - 400 steps (200 warmup). Print tok/s delta.

3. **Run and commit results**
   - Add a row to STATUS.md's Compile × Kernel Ablation table recording the measured delta.

**Acceptance (W4):**
- Env var gate works: unset → default path; set → compiled path logged.
- Measurement published, even if delta is ≤ 0. No decision to make default-on as part of Phase 1.

**Estimated: 1.5 hours.**

---

## W5: Residual dedup in HyPEShortConvBlock

**Files:**
- `models/components/conv_blocks.py` (edit `HyPEShortConvBlock.forward`)

**Steps:**

1. **Read the current body of `forward` after the conv path** and replace:
   ```python
   conv_out = self.out_proj(c * z)
   ffn_out = self.ffn(self.ffn_norm(x + conv_out))
   return x + conv_out + ffn_out
   ```
   with:
   ```python
   conv_out = self.out_proj(c * z)
   residual = x + conv_out
   ffn_out = self.ffn(self.ffn_norm(residual))
   return residual + ffn_out
   ```

2. **Correctness verification**
   - Run `scripts/test_compile_friendly_parity.py` — must still pass.
   - Run `scripts/test_odin_chunked.py` — final loss match to existing baseline within 1e-5 (this is a pure algebraic refactor; numerics must be identical barring fp rounding).
   - Run OdinHaloMini smoke: `python -m halo_training --model models/odin_halo.py --class-name OdinHaloMini --smoke`. Loss trajectory identical to pre-change.

**Acceptance (W5):**
- All three tests pass.
- Ablation at batch=16 shows no regression (±1% noise tolerance).

**Estimated: 1 hour.**

---

## W6: DataLoader tuning

**Files:**
- `halo_training/data.py` (default changes, add `pin_memory`)
- `halo_training/trainer.py` (`non_blocking=True` on H2D copies)
- `halo_training/cli.py` (add `--num-workers` flag)

**Steps:**

1. **Edit `halo_training/data.py::build_dataloader`**
   - Signature: add `pin_memory: bool = True`.
   - Change default `num_workers` from 0 to 2.
   - Pass `pin_memory` to the `DataLoader(...)` call.

2. **Edit `halo_training/trainer.py`**
   - Find both sites (train loop + validation if any) where `input_ids.to(device)` / `targets.to(device)` happen.
   - Change to `.to(device, non_blocking=True)`.

3. **Add `--num-workers` CLI flag** in `halo_training/cli.py`, default 2, thread to train call, thread to build_dataloader.

4. **Smoke test**
   - `python -m halo_training ... --smoke` runs clean.
   - `python -m halo_training ... --smoke --num-workers 0` still works (regression guard).

**Acceptance (W6):**
- Both smoke runs pass.
- Short 50-step training on OdinHalo shows tok/s ≥ previous baseline (tolerance ±2%).

**Estimated: 1 hour.**

---

## Final regression test

After all 6 work items land:

1. Run `scripts/ablation_final.py` at batch=16 with the current stack (compile + HIP CE, plus all Phase 1 items enabled). Compare to the pre-Phase-1 baseline of 14,682 tok/s.
2. Run `scripts/test_all_models_chunked_train.py` to confirm no halo variants broke.
3. Append a post-Phase-1 row to STATUS.md's Throughput Reference section.

**Expected post-Phase-1 tok/s:**
- If optimizer shootout winner is Muon or Lion: ~16,000–17,000 tok/s (+9 to +16%).
- If AdamW wins: ~14,900–15,100 tok/s (+1.5 to +3% from dedup + data pipeline).

## Rollback plan

Each work item is its own commit. Revert commits selectively if any later turns out to regress throughput or cause training instability. The optimizer shootout is read-only (no default change); nothing to roll back there.

## Hand-off artifacts

- `docs/perf/odinhalo-profile-2026-05-05/` — drives Phase 2.
- `docs/perf/optimizer-shootout-2026-05-05.md` — decision input for future default-optimizer change.
- Updated STATUS.md — current baseline for Phase 2/Phase 3 planning.
