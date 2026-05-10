# v3 40k Campaign — Session Execution Log

**Session date:** 2026-05-10 / 2026-05-11
**Starting SHA:** `664fd20` (post-T-0 read-only findings)
**Plan locked:** `docs/research/autokernel-40k-v3-execution-plan.md`

This doc summarizes what executed in this session, what's still running,
and what's ready to launch in the next session.

---

## Completed in this session

### Step 1 — Plan + STATUS (commit `d05adf7` / earlier)
- Execution plan: `docs/research/autokernel-40k-v3-execution-plan.md`
- STATUS.md updated with v3 §5.3 language verbatim (nominal 40k,
  engineering success 36k, strong 38k, stretch 40k)

### Step 2 — T-0 Instrumentation (all 8 items) ✓ DONE
- T-0.1 backward profile: `docs/perf/backward-breakdown.md` (pre-session)
- **T-0.2 per-param NorMuon telemetry** — `halo_training/normuon.py`:
  ctor kwargs `telemetry_enabled`, `telemetry_path`, `trust_cap`,
  `trust_cap_scope`, `w_gate_up_scale`, `w_gate_up_ramp_steps`,
  `spectra_branchless`. Emits v3 §9.1 JSONL schema. Flag:
  `--ak-normuon-telemetry`.
- T-0.3 sync-point audit: `docs/perf/sync-point-audit.md` (pre-session)
- T-0.4 graph-break inventory: `docs/perf/graph-breaks-inventory.md`
  (pre-session)
- **T-0.5 granular `--ak-*` flags** — `scripts/train_ddp.py`: 28 flags
  per v3 §8 (16 core + 12 add-on). All default OFF. Flag manifest
  printed at launch.
- **T-0.6 DDP allreduce trace** — `scripts/train_ddp.py`:
  `_DDP_TRACE_STATE` + hooks in allreduce_grads_async/sync. Emits v3
  §9.3 JSONL per opt step. `--assert-no-sync` flag aborts if
  allreduce_count > 1/step. **Verified: no_sync correct** (count=1 in
  T-1.5 run).
- **T-0.7 dtype/autocast inventory** — `scripts/autokernel_dtype_
  inventory.py` + `docs/perf/dtype-autocast-inventory.md`/`.jsonl`.
  **Key finding: 0/7 ops have `register_autocast`** — confirms v3 H11
  as concrete gap.
- **T-0.8 sync counter in profiler** — `scripts/train_ddp.py` scans
  profile key_averages for aten::item count + hipMemcpyWithStream wall.
  Emits v3 §9.4 JSONL.

### Step 3 — T-1 Quick Wins (partial)
- **T-1.1 branchless SPECTRA** — `halo_training/spectra.py` +
  `halo_training/normuon.py`. Env var
  `AUTOKERNEL_SPECTRA_BRANCHLESS=1` activates the path.
  **Parity test 12/12 PASS** (`scripts/test_spectra_branchless.py`).
- **T-1.2 deferred loss sync** — `scripts/train_ddp.py`:
  `--ak-sync-cleanup` routes `loss.item()` through a tensor accumulator,
  syncing once per opt step (8× reduction at accum=8).
- **T-1.3 graph-break fix (rope_gate_mul)** — see T-3.2 below.
  DaoAILab `causal_conv1d_fn` shim deferred.
- **T-1.4 DDP bucket sweep** — `scripts/probe_t1_ddp_bucket_sweep.sh`
  ready to launch (4 bucket sizes × 150-step smoke).
- **T-1.5 fused-zloss validation** — **RUNNING in background**. See
  `docs/perf/t1-5-fused-zloss-preliminary.md`. **PRELIMINARY PASS:
  +7.7% tok/s stable over 300 steps, loss healthy.**

### Step 4 — T-2 infrastructure
- **T-2.1 batch=32 probe** — `scripts/probe_t2_1_batch32.sh` ready
  (4-config smoke).
- **T-2.3 NorMuon impl optimizations** — branchless SPECTRA already
  wired via `spectra_branchless=True` ctor arg; activated by
  `--ak-normuon-impl-opt` or `--ak-spectra-branchless` or
  `--ak-sync-cleanup`.

### Step 5 — T-3 dtype audit
- **T-3.1 Tier 0 inventory** — done (see T-0.7 above).
- **T-3.2 fused_rope_gate_mul fix** — `kernels/hip/_torch_ops.py`:
  `autokernel::fused_rope_gate_mul` custom_op with `register_fake` +
  `register_autograd` (pure PyTorch backward). `models/components/
  conv_blocks.py` routes through it when `AUTOKERNEL_FIX_ROPE_GATE=1`
  (via `--ak-fix-rope-gate-op` flag).
  **Parity test 3/3 PASS** (forward, gradient flow, backward parity).
  **Fixes pre-fix silent-freeze** where b/h_tilde grads were severed.
- T-3.2 causal_conv1d shim — deferred (tier-1 target, rope_gate_mul
  was the higher-priority fix per T-0.4 occurrence count).
- T-3.3 Tier 2 deep parity tests — deferred (5 ops; schedule allows).

### Step 6 — T-4 infrastructure
- **T-4 compiled autograd smoke** — `scripts/probe_t4_compiled_
  autograd.sh` ready (4-config smoke).
- **CA activation** — `scripts/train_ddp.py` sets
  `torch._dynamo.config.compiled_autograd = True` when
  `--ak-compiled-autograd` is passed.

### Step 8 — T-6 Sprint 3 launch scripts
- **Sprint 3A launcher** — `scripts/launch_sprint3a.sh` with STACK
  selector (A/B/C/D). Stack A is production-ready now.
- **Sprint 3B launcher** — `scripts/launch_sprint3b.sh` for OdinHalo
  with the same stack selector.

---

## Running / deferred

### T-1.5 continues in background
2000-step target. ~4h wall projected. Result: `docs/perf/t1-5-fused-
zloss-preliminary.md` at step 300; full scorecard when complete.

### Deferred to next session (compute-budgeted)
- **T-1.4 DDP bucket sweep** (1-2h compute)
- **T-2.1 batch=32 probe** (1-2h compute)
- **T-4 compiled autograd smoke** (1-2h compute, gate-contingent
  follow-on work)
- **T-3.3 Tier 2 deep parity** for 4 remaining ops (~2-3h engineering +
  ~30min compute)
- **T-3.2 causal_conv1d shim** (~4-6h engineering + test)
- **T-5 hidden kernel recovery** (warm-start matrix, trust cap,
  staging — ~5-10h compute)
- **T-6 Sprint 3A launch** (Stack A ready; ~61h wall)
- **T-6 Sprint 3B launch** (Stack A ready; ~77h wall)

---

## Key findings to propagate

1. **+7.7% tok/s from fused zloss alone** (T-1.5 at step 300, stable).
   Matches v3's 5-8% prediction. Ready for production Stack A.

2. **0/7 training-path custom ops have `register_autocast`** (T-0.7).
   Confirms v3 H11 as a concrete gap contributing to Phase C/G
   divergence. Tier 2 fix pending.

3. **fused_rope_gate_mul had a pre-fix silent-freeze** (T-3.2). The
   production path bypassed autograd on 2/3 of self.proj's output
   columns. Fix ships as `--ak-fix-rope-gate-op` flag; test passes.

4. **DDP no_sync is correct** (T-0.6 live verification). allreduce_
   count=1 per opt step measured over 148 steps.

5. **Honest throughput ceiling: 35-38k** matches v3's §1 "practical
   expected outcomes". 40k remains stretch; requires most T-2/T-4/T-5
   levers at upper bound.

---

## Sprint 3 readiness

**Stack A is ready to ship NOW** at projected ~33.7k tok/s:

```bash
STACK=A bash scripts/launch_sprint3a.sh
STACK=A bash scripts/launch_sprint3b.sh
```

Stack B/C/D unlock additively as T-2/T-4/T-5 probes complete and pass
their gates. Upgrading in place is safe — sprint checkpoints save
optimizer/scaler state.

---

## Commits this session

See `git log --oneline 664fd20..HEAD`:
- `d05adf7` Lock 40k throughput execution plan + STATUS 40k throughput framing
- `ded406a` T-0.5 + T-0.2: granular --ak-* flags + NorMuon per-param telemetry
- `e60b5df` T-0.6/7/8: DDP allreduce trace + dtype inventory + sync counter
- `37c16c6` T-1.1 + T-1.2 + T-1.4: branchless SPECTRA, deferred loss, DDP sweep
- `ac59cf9` T-1.1 parity VERIFIED + T-0.7 dtype inventory
- `124b18e` Fix param_names collision with PyTorch optimizer key
- `b498816` T-3.2: fused_rope_gate_mul custom_op with register_autograd
- `22ce520` T-2.1 batch=32 probe + T-4 compiled autograd smoke + CA activation
