# Execution Plan: Stage 1 → Sprint 1.5 → Sprint 3A → Sprint 3B

**Date:** 2026-05-07
**Purpose:** Resume-safe guide for a context-compacted session. This doc captures the full execution path from "right now" (just after the Sprint 3 smoke + StabilityGuard scale-collapse fix shipped) through to completion of Sprint 3B.
**Owner:** Assistant, self-directed with per-stage user checkpoints noted inline.
**Read before executing any step.**

---

## 0. Context snapshot (must re-read on resume)

### 0.1 What shipped recently
- **commit `007486b`** — fp16 stability hardening (z-loss, iter_scales clamp, growth_interval=500, resumed-run grad-norm, attn-softcap, activation-monitor, NaN forensics dump, rollback tightens growth_interval, scaler scale in log)
- **commit `7eb0188`** — fix: replace `scaler.step` try/except with `backwards_in_cycle` counter in `scripts/train_ddp.py`
- **commit `cf9f45a`** — Sprint 3 smoke (OdinHalo dolma-10B, 1000 steps) + `StabilityGuard.check_scaler` fix for scale-collapse gap discovered in the smoke

### 0.2 Key measurements from Sprint 3 smoke (`docs/perf/sprint3-smoke-findings.md`)
- OdinHalo + dolma-10B DDP: **23,302 tok/s** at `lr_2d=5e-3` (smoke baseline; too hot — scale collapse by step 950)
- Expected post-monitor-fix: **~26-28K tok/s** (~10-15% recovery from removing `.item()` sync per forward per layer)
- Loss minimum was step 650 at `loss=3.25`, then climbed monotonically — LR too high for OdinHalo/dolma
- `shared_layers.5` maxabs grew geometrically: 38 → 289 → 1237 → 3247 → 5061 → **9117** over 500 steps. fp16 headroom collapsed 1700× → 7.2× → NaN.
- Eval scorecards exist at `docs/perf/eval-scorecards/sprint3-smoke-dolma-step-{250,500,750,1000}.json`

### 0.3 Current test status
All suites pass (verified 2026-05-07):
- `scripts/test_fp16_stability.py` — 16/16
- `scripts/test_sprint1_1_normuon.py` — 8/8
- `scripts/test_sprint1_1_profile.py` — 3/3
- `scripts/test_sprint1_1_bench.py` — 1/1
- `scripts/test_sprint1_phase1.py` — 15/15
- `scripts/test_sprint1_phase2.py` — 15/15

### 0.4 User decisions locked in
| Decision | Answer |
|---|---|
| Sprint 3 model order | OdinFlat first (Sprint 3A, ~50h), OdinHalo second (Sprint 3B, budget TBD) |
| Sprint 1.5 execution | Full 9-day plan as designed in existing spec/plan |
| Sprint 1.5 timing | Before Sprint 3A so both machines can be used for 3A's full DDP run |
| Sprint 1.5 code ownership | Assistant implements, user reviews each phase |
| Stage 1 iter step count | **400 steps per iter** (S1.3 gets 700; S1.4 is 400 × 2) |
| Stage 1 iter 1 LR | `lr_2d=2e-3` (conservative) |
| Stage 1 includes optimize-kernels A/B | Yes (iter 4) |
| Stage 1 includes OdinFlat dolma sanity | Yes (iter 5) |
| Sprint 3B token budget | Decide AFTER Stage 1 measured throughput |
| Activation monitor fix first | Yes (S1.0) |

### 0.5 Both machines state (as of smoke complete)
- Machine A (`joelwang-ai-2@10.77.0.1`, venv `~/Desktop/ai_lab/autokernel-halo-strix/.venv`, clean no aiter): has `dolma-10b-odin32k.bin` (13.7 GB)
- Machine B (`joelwang-ai-1@10.77.0.2`, venv `~/Desktop/comfyui-rocm7.12/.venv`, has aiter): has `dolma-10b-odin32k.bin` copied from A (13.7 GB, 1.04 GB/s over TB4)
- Both idle; previous smoke run terminated cleanly

### 0.6 Critical reminder: sync to BOTH machines
`bash sync_remote.sh` goes to Machine A only. For DDP runs (which SSH into Machine B to start rank 1), ALWAYS also run `bash sync_remote_b.sh`. I lost 30 minutes on the first smoke attempt from a stale Machine B.

---

## 1. Stage 1 — OdinHalo LR tuning + OdinFlat dolma sanity (today, ~3.5 hrs compute)

All DDP runs on both machines. Iterations sequential (each needs exclusive machine time).

### 1.0 — Pre-iter: Fix activation monitor performance (~10 min dev, 0 compute)

**Problem:** `halo_training/activation_monitor.py::_make_hook` runs `.item()` (GPU→CPU sync) on every forward of every tracked layer, independent of sample_every. On OdinHalo that's 18 effective layers × 8 microsteps/opt-step × 2 ranks = 288 syncs per opt step. Estimated cost: 10-15% throughput.

**Design of fix:**
```python
# halo_training/activation_monitor.py

class ActivationMonitor:
    def __init__(...):
        ...
        # Trainer calls ``set_step`` each opt step so the hook can no-op
        # on non-sampling steps. Removes ~10-15% throughput overhead on
        # looped models.
        self._current_step: int = -1
        self._should_sample: bool = False

    def set_step(self, global_step: int) -> None:
        """Called by trainer each opt step BEFORE the forward. Decides
        whether this step's forward should compute maxabs."""
        self._current_step = global_step
        self._should_sample = (global_step % self.sample_every == 0)

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            if not self._should_sample:
                return  # no-op on non-sampling steps
            t = _first_tensor(output)
            if t is None:
                return
            with torch.no_grad():
                # Store tensor ref, not the .item() scalar — compute
                # and sync happens only once per sample-step inside step().
                self._current_stats[name] = {
                    "maxabs_tensor": t.detach().abs().max(),  # still on GPU
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "shape": list(t.shape),
                }
        return hook

    def step(self, global_step: int) -> None:
        # If this isn't a sampling step, nothing to do.
        if global_step % self.sample_every != 0:
            return
        if not self._current_stats:
            return
        # Batch all .item() syncs into a single barrier-ish pass at commit time.
        # Sort first so serialization order is stable per step.
        rows = []
        for layer, stats in sorted(self._current_stats.items()):
            maxabs_t = stats.get("maxabs_tensor")
            if maxabs_t is None:
                continue
            maxabs = float(maxabs_t.item())  # GPU sync here, once per layer per sample step
            headroom = _FP16_MAX / maxabs if maxabs > 0 else float("inf")
            rows.append({
                "step": int(global_step),
                "layer": layer,
                "maxabs": maxabs,
                "fp16_headroom": headroom,
                "dtype": stats["dtype"],
                "shape": stats["shape"],
            })
        if rows and self.output_path:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            with open(self.output_path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    def current_stats(self) -> Dict[str, Dict]:
        """Used by NaN forensics dump. Force .item() on any pending tensors."""
        out = {}
        for k, v in self._current_stats.items():
            d = dict(v)
            t = d.pop("maxabs_tensor", None)
            if t is not None:
                try:
                    d["maxabs"] = float(t.item())
                except Exception:
                    d["maxabs"] = None
            out[k] = d
        return out
```

**Trainer plumbing** — `scripts/train_ddp.py`, in the microstep loop ONCE per opt step (not per microstep). Place before the first microstep in the cycle OR after global_step increments. Current location: right after `global_step += 1` in the async path (line ~970).

Simplest: call `monitor.set_step(global_step)` right after `if monitor is not None: monitor.step(global_step)`. The set_step applies to the NEXT opt step's forwards.

But wait — the hook fires during the CURRENT microstep's forward. So we need to set `_should_sample` BEFORE the first microstep of the next opt step, not after. Currently `set_step` would be called between opt steps. The flag applies to all forwards between this `set_step` and the next one. That's microsteps of the upcoming cycle.

Trainer change: after `monitor.step(global_step)`, also call `monitor.set_step(global_step + 1)` — so forwards in the next accum cycle sample on step `global_step + 1` if that's a sample step.

Alternative clean design: maintain current_step inside ActivationMonitor, call `monitor.set_step(global_step + 1)` right after `global_step += 1`. Then first forward of next cycle sees `_should_sample = ((global_step+1) % sample_every == 0)`. Commit happens when `monitor.step(global_step+1)` is called at end of that step.

**Tests to add:**
- `test_monitor_no_op_on_non_sample_step`: attach monitor with sample_every=100, call set_step(5), run forward, verify `_current_stats` is empty AND no `.item()` was synced
- `test_monitor_records_on_sample_step`: set_step(100), run forward, verify `_current_stats` has `maxabs_tensor` entries; call step(100), verify JSONL written
- Update existing `test_activation_monitor_writes_jsonl` to use the new flow

**Exit criteria:**
- All 16/16 fp16_stability tests still pass + 2 new pass
- Throughput regression test: run `scripts/profile_step.py` with `--activation-monitor` on vs off at sample_every=100 — delta < 1% (was 10-15%)

**Commit:** `fp16 stability: gate activation monitor .item() sync behind sample step`

### 1.1 — OdinHalo dolma iter 1 at lr_2d=2e-3 (~35 min)

**Setup:**
```bash
# Kill any stragglers
bash run_remote.sh "pkill -f torchrun 2>/dev/null; sleep 3"
bash run_remote_b.sh "pkill -f torchrun 2>/dev/null; sleep 3"

# Sync to BOTH machines
bash sync_remote.sh && bash sync_remote_b.sh
```

**Launch** (via `scripts/launch_ddp.sh`, wrapped in a poll loop):
```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
  rm -rf checkpoints/sprint3-iter1-lr2e3 && mkdir -p checkpoints/sprint3-iter1-lr2e3 && \
  CKPT_DIR=checkpoints/sprint3-iter1-lr2e3 \
  MODEL=models/odin_halo.py CLASS=OdinHalo \
  DATASET=datasets/dolma-10b-odin32k.bin \
  EPOCHS=1 LR=8e-4 \
  BLOCK=512 BATCH=16 ACCUM=8 \
  CHECKPOINT_INTERVAL=200 \
  EXTRA_FLAGS='--max-steps 400 --imu1-groups --normuon --lr-2d 2e-3 --lr-1d 8e-4 --no-muon \
               --intra-doc-mask --value-residuals --head-gating \
               --z-loss 1e-4 --z-loss-fraction 1.0 \
               --attn-softcap 50.0 \
               --activation-monitor --activation-monitor-interval 50 \
               --max-grad-norm 0.8 \
               --auto-eval' \
  bash scripts/launch_ddp.sh > checkpoints/sprint3-iter1-lr2e3/launch.log 2>&1"

# Poll for Done
bash run_remote.sh "while true; do
    if grep -q '^Done:' ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/sprint3-iter1-lr2e3/rank0.log 2>/dev/null; then
      echo DONE; break
    fi
    if grep -qE '^Traceback|^RuntimeError|unrecoverable' ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/sprint3-iter1-lr2e3/rank0.log 2>/dev/null; then
      echo FATAL; tail -30 ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/sprint3-iter1-lr2e3/rank0.log; break
    fi
    sleep 60
  done"
```

**Pull results:**
```bash
scp joelwang-ai-2@192.168.1.140:~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/sprint3-iter1-lr2e3/activation_stats.jsonl docs/perf/sprint3-iter1-activation-stats.jsonl
scp "joelwang-ai-2@192.168.1.140:~/Desktop/ai_lab/autokernel-halo-strix/docs/perf/eval-scorecards/sprint3-iter1-lr2e3-step-*.json" docs/perf/eval-scorecards/
python scripts/analyze_activation_stats.py docs/perf/sprint3-iter1-activation-stats.jsonl
```

**Pass criteria (ALL must hold):**
1. **Loss monotonically decreasing** from step 50 through step 400 (allow small wobbles; trend must be down)
2. **shared_layers.5 maxabs at step 400** < 100 (smoke had 99.6 at step 400 with lr 5e-3; we expect 2e-3 to show ~20-40 maxabs at step 400)
3. **fp16_headroom at step 400** > 500× for all layers
4. **Zero scaler backoffs** (scale stays at init_scale=1024 through warmup, may grow to 2048 after 500 steps of clean training — 400-step iter shouldn't see growth)
5. **No NaN microstep messages** in rank0.log
6. **`grad=<finite value>`** in final log line (not `grad=nan`)

**Decision tree:**
- ALL criteria pass → proceed to S1.2 (push to lr_2d=3e-3)
- Fails (1) or (2) or (3) → lr_2d=2e-3 is still too hot for OdinHalo/dolma; halve again → iter 1b at lr_2d=1e-3
- Fails (4) only → investigate before continuing (scale change during 400 steps is unusual unless very long warmup elapsed)

**Commit after iter 1:** `Sprint 3 Stage 1 iter 1: OdinHalo dolma lr_2d=2e-3 {PASS|FAIL}` with scorecard + activation stats + rank0.log summary.

### 1.2 — OdinHalo dolma iter 2 at lr_2d=3e-3 (~35 min; conditional)

Same template as 1.1 but `--lr-2d 3e-3` and `CKPT_DIR=checkpoints/sprint3-iter2-lr3e3`.

**Rationale:** iter 1 at 2e-3 is conservative; if clean, 3e-3 gets faster loss descent. Prefer higher-LR winner if both pass.

**Pass criteria:** same as 1.1.

**Decision tree:**
- Both 1.1 and 1.2 pass → pick **iter 2 (3e-3)** as winner for 1.3 (faster loss descent)
- Only 1.1 passes → **iter 1 (2e-3)** winner
- Neither passes → iter 1b at 1e-3 (and skip 1.2); if still fails, escalate to user

**Commit:** `Sprint 3 Stage 1 iter 2: OdinHalo dolma lr_2d=3e-3 {PASS|FAIL}`

### 1.3 — OdinHalo extended confirmation at winner LR (~60 min, 700 steps)

Config same as winning iter but:
- `--max-steps 700`
- `CHECKPOINT_INTERVAL=200` (3-4 intermediate checkpoints + final)
- `CKPT_DIR=checkpoints/sprint3-iter3-confirm`

**Rationale:** smoke's crisis zone was steps 650-950. 700-step extended run confirms stability through that window.

**Pass criteria:**
- All 1.1 criteria PLUS:
- **shared_layers.5 maxabs at step 700** < 200 (smoke had 3247 at step 700 with lr 5e-3; winner should be dramatically less)
- **fp16_headroom at step 700** > 300×
- **Loss at step 700** lower than loss at step 400 (keeps monotonic descent)

**Decision:** if iter 3 passes, winner LR is locked. If fails, drop 1 step down LR and retry iter 3.

**Commit:** `Sprint 3 Stage 1 iter 3: OdinHalo dolma 700-step confirmation {PASS|FAIL}`

### 1.4 — optimize-kernels A/B (~70 min, 400 steps × 2)

Two runs at winner LR, identical except for `--optimize-kernels` flag:

**Run A (off):** `CKPT_DIR=checkpoints/sprint3-iter4A-nokernel`, no `--optimize-kernels`
**Run B (on):** `CKPT_DIR=checkpoints/sprint3-iter4B-kernel`, add `--optimize-kernels`

**Pass criteria:**
- Both runs complete without crash (autokernel can fail; handle gracefully — commit says "on" if B crashed we just skip it)
- Step-400 loss within 0.5% between A and B
- Throughput: record both; keep the faster if quality matches

**Decision:**
- B faster AND quality parity → use `--optimize-kernels` for Sprint 3A and 3B
- B crashes or quality diverges → keep OFF
- A faster (unlikely given +3× claim) → keep OFF

**Commit:** `Sprint 3 Stage 1 iter 4: optimize-kernels A/B {winner=on|off}`

### 1.5 — OdinFlat dolma sanity at lr_2d=5e-3 (~20 min, 400 steps)

Last iter validates OdinFlat on dolma (never tested — Sprint 1 only did wikitext). Since OdinFlat is non-looped, no iter_scales, and Sprint 1.1 Run 2b shipped it cleanly at 5e-3 on wikitext, this should PASS on dolma.

**Config:**
```bash
CKPT_DIR=checkpoints/sprint3-iter5-odinflat-sanity \
MODEL=models/odin_flat.py CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 LR=8e-4 \
BLOCK=512 BATCH=16 ACCUM=8 \
CHECKPOINT_INTERVAL=200 \
EXTRA_FLAGS='--max-steps 400 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --no-muon \
             --intra-doc-mask --value-residuals --head-gating \
             --z-loss 1e-4 --z-loss-fraction 1.0 \
             --attn-softcap 50.0 \
             --activation-monitor --activation-monitor-interval 50 \
             --max-grad-norm 1.0 \
             --auto-eval'
```

Note: max-grad-norm=1.0 (standard; OdinFlat is non-looped so doesn't need 0.8 tightening). `--optimize-kernels` per iter 4 decision.

**Pass criteria:**
- Same as 1.1 but more lenient: fp16_headroom > 100× (OdinFlat has deeper layers so more growth is OK at same LR)
- Loss monotonically decreasing through step 400
- Zero NaN microsteps, zero scaler backoffs

**Decision:**
- Pass → Sprint 3A locks lr_2d=5e-3 for full epoch
- Fail → unexpected; drop OdinFlat to lr_2d=3e-3 and retry, report to user before Sprint 3A

### 1.6 — Stage 1 wrap-up commit

Update `STATUS.md` with:
- Final OdinHalo lr_2d winner (from iter 3)
- OdinFlat dolma sanity result
- optimize-kernels decision
- Measured post-monitor-fix tok/s for both models
- **Sprint 3B token budget** decision (e.g. "50h wall @ 27K tok/s = 4.86B tokens ≈ 70% of T²-optimal")
- Throughput projections for both Sprint 3A and 3B

Commit: `Sprint 3 Stage 1 complete: LR + kernel config locked in`

---

## 2. Stage 2 — Sprint 1.5 (SPECTRA + μP, ~9 elapsed days)

**Spec:** `docs/superpowers/specs/2026-05-06-sprint1.5-spectra-mup-design.md`
**Plan:** `docs/superpowers/plans/2026-05-06-sprint1.5-spectra-mup-plan.md`

I implement per the plan. Phases A-F as spec'd. Key milestones:

### Phase A (days 1-3, 0 compute, ~790 LoC across 11 files)
Key tasks (from plan §Phase A):
- A.1 `halo_training/spectra.py` — SPECTRA post-clipping via existing Newton-Schulz from `halo_training/muon.py`
- A.2 Integrate SPECTRA into `NorMuon.step()` via `spectra_post` / `spectra_clip_norm` kwargs
- A.3 `halo_training/mup.py` — μP init scaling + 3-way param groups (embed, hidden, readout)
- A.4 Wire `use_mup` into `build_optimizer`
- A.5 Apply μP init in `OdinFlatBase._init_weights` (triggered by model flag)
- A.6 `models/odin_flat_30m.py` — 30M proportional probe
- A.7 CLI flag plumbing (`--spectra-post`, `--mup`, `--mup-base-width`, `--spectra-clip-norm`, etc.)
- A.8 Smoke tests: all-flags-default must match Sprint 1 baseline exactly

**Exit gate for Phase A:** unit tests all pass; `--help` lists new flags; default-off smoke matches baseline within fp16 noise.

### Phase B (day 4, ~2 hrs compute)
- B1: clip_norm sweep `{0.5, 1.0, 2.0}` × 200 steps on full OdinFlat (DDP, uses both machines)
- B2: 30M μP LR probe `{0.015, 0.020, 0.0235, 0.030}` × 500 steps (SINGLE NODE on Machine A; Machine B idle)

### Phase C (days 4-5, ~6 hrs DDP)
Factorial cells C1/C2/C3 at 122M OdinFlat, wikitext-103, 2 epochs each:
- C1: SPECTRA-post only (clip_norm from B1 winner) + Sprint 1 recipe
- C2: μP-partial only + Sprint 1 recipe
- C3: SPECTRA-post + μP-partial combined

Baseline: `checkpoints/odin-flat-wikitext-ddp-sprint1/step_1869.pt` (per plan §Phase C pre-req). Might need to re-resume from Run 2b's final checkpoint `checkpoints/sprint1-run2b/step_936.pt` — **user approval needed** if baseline checkpoint path differs from plan assumption.

### Phase D (days 6-7, 2d dev + ~4 hrs compute)
- Implement SPECTRA pre-clipping (gradient spectral clip after backward, before optimizer.step)
- Cells D1 (post+pre, no μP), D2 (post+pre + μP-partial)

### Phase E (days 8-9, 2d dev + ~4 hrs compute)
- Implement full μP (1/d_head attention scaling; partial μP + this)
- Cells E1 (post+pre + full-μP), E2 + E3 (transfer sanity — LR at winner × {0.8, 1.2})

### Phase F (day 9, 0.5d cleanup)
- Flag defaults decision: which of SPECTRA-post, SPECTRA-pre, μP-partial, μP-full (if any) to flip ON by default based on factorial winners
- Update knowledge base: `knowledge/training/spectra_mup_results_2026.md` with factorial attribution
- AGENTS.md + CONSTRAINTS.md entries
- Single atomic Sprint 1.5 commit (all 9 days of work)

### User checkpoints during Sprint 1.5
Per user's "I implement" answer, they review each phase. Propose 5 checkpoint interactions:
1. **After Phase A.8** (unit tests green, smoke matches baseline) — confirm code before compute burns start
2. **After Phase B** (clip_norm winner, 30M LR winner) — review probe results before committing to factorial
3. **After Phase C** (C1/C2/C3 scorecards) — decide if pre-clip (D) is worth the compute
4. **After Phase E2/E3** (transfer sanity) — confirm μP transfer claim holds before flipping defaults
5. **Before Phase F commit** — review consolidated doc + flag decisions

**How I'll signal:** at each checkpoint, update STATUS.md + push individual commits, then ask user to review before proceeding.

---

## 3. Stage 3 — Sprint 3A: OdinFlat dolma 1 epoch (~50h DDP)

**Pre-reqs:** Stage 1 + Sprint 1.5 complete. Both machines idle. Sprint 1.5 baseline checkpoint preserved.

### 3.1 Launch config

```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
  rm -rf checkpoints/sprint3a-odinflat-dolma && \
  mkdir -p checkpoints/sprint3a-odinflat-dolma && \
  CKPT_DIR=checkpoints/sprint3a-odinflat-dolma \
  MODEL=models/odin_flat.py CLASS=OdinFlat \
  DATASET=datasets/dolma-10b-odin32k.bin \
  EPOCHS=1 LR=8e-4 \
  BLOCK=512 BATCH=16 ACCUM=8 \
  WARMUP_STEPS=300 \
  CHECKPOINT_INTERVAL=1000 \
  MAX_GRAD_NORM=1.0 \
  EXTRA_FLAGS='--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --no-muon \
               --intra-doc-mask --value-residuals --head-gating \
               --z-loss 1e-4 --z-loss-fraction 1.0 \
               --attn-softcap 50.0 \
               --activation-monitor --activation-monitor-interval 500 \
               --auto-eval \
               {optimize-kernels flag per Stage 1.4} \
               {Sprint 1.5 winner flags: --spectra-post ? --mup ? etc.}' \
  bash scripts/launch_ddp.sh"
```

### 3.2 Expected metrics
- Steps: 6.9B tokens / 131K tokens per step = **~52,700 opt steps**
- Wall time: 6.9B / 38K tok/s (OdinFlat) = **~50 hrs**
- Checkpoints: every 1000 steps = 52 checkpoints → 52 auto-eval scorecards
- If Sprint 1.5 included SPECTRA/μP, adjust flag set accordingly

### 3.3 Monitoring
- Daily: poll `tail -50 $CKPT_DIR/rank0.log` and `tail -10 $CKPT_DIR/train_log.jsonl`
- Watch for: scale warnings, NaN microstep messages, `grad=nan`, rollback events
- Check `activation_stats.jsonl` every ~6 hours; all layers should maintain headroom > 100×

### 3.4 Success criteria
- Completes 1 epoch (6.9B tokens, ~52.7K steps)
- Final `avg BPB` < wikitext 1-epoch baseline (Run 2b: 2.812) — dolma is a different distribution so this is approximate
- No rollbacks (or recoverable ones — max 5)
- Zero corrupted checkpoints

### 3.5 If it fails partway
- Rollback already handled by StabilityGuard (all 4 detection mechanisms in place)
- If training dies beyond rollback: `$CKPT_DIR/nan_dump_step_N.pt` has forensics — read per `knowledge/training/fp16_stability_gfx1151.md` playbook
- Last good checkpoint is always recoverable for resume via `--resume-from`

---

## 4. Stage 4 — Sprint 3B: OdinHalo dolma (~40-50h DDP)

**Pre-reqs:** Sprint 3A complete. Stage 1 LR tuning complete. Token budget decided in Stage 1.6 based on measured throughput.

### 4.1 Launch config

```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
  rm -rf checkpoints/sprint3b-odinhalo-dolma && \
  mkdir -p checkpoints/sprint3b-odinhalo-dolma && \
  CKPT_DIR=checkpoints/sprint3b-odinhalo-dolma \
  MODEL=models/odin_halo.py CLASS=OdinHalo \
  DATASET=datasets/dolma-10b-odin32k.bin \
  EPOCHS=1 LR=8e-4 \
  BLOCK=512 BATCH=16 ACCUM=8 \
  WARMUP_STEPS=300 \
  CHECKPOINT_INTERVAL=1000 \
  MAX_GRAD_NORM=0.8 \  # OdinHalo looped — tighter per AGENTS.md
  EXTRA_FLAGS='--max-steps {from Stage 1.6 decision} \
               --imu1-groups --normuon \
               --lr-2d {Stage 1 iter 3 winner} --lr-1d 8e-4 --no-muon \
               --intra-doc-mask --value-residuals --head-gating \
               --z-loss 1e-4 --z-loss-fraction 1.0 \
               --attn-softcap 50.0 \
               --activation-monitor --activation-monitor-interval 500 \
               --auto-eval \
               {optimize-kernels flag per Stage 1.4} \
               {Sprint 1.5 winner flags IFF OdinHalo port validated}' \
  bash scripts/launch_ddp.sh"
```

### 4.2 Key differences from Sprint 3A
- OdinHalo (not OdinFlat) — looped model, ~23-27K tok/s
- `--max-steps` explicit based on token budget (from Stage 1.6)
- `MAX_GRAD_NORM=0.8` for looped (existing convention per AGENTS.md)
- Sprint 1.5 flags only if they were validated on OdinHalo — spec says "OdinHalo port deferred". User decision needed before enabling.

### 4.3 Success criteria
- Completes the target token count (set by Stage 1.6)
- No rollbacks (ideally); if rollback fires, StabilityGuard handles per the 4-mechanism path
- Activation headroom stays > 50× throughout (tighter than 3A because looped)

---

## 5. Operational notes

### 5.1 Sync protocol (critical)

**Always run BOTH sync scripts before any DDP launch:**
```bash
bash sync_remote.sh && bash sync_remote_b.sh
```

Machine B runs rank 1 via SSH from `launch_ddp.sh`, so stale code on B breaks the entire run. This cost me 30 minutes on the first Sprint 3 smoke.

### 5.2 Running on remote machines

- `bash run_remote.sh "<command>"` — Machine A (DDP rank 0)
- `bash run_remote_b.sh "<command>"` — Machine B (DDP rank 1, standalone for single-node tests)

Never use raw SSH; these scripts set the venv + HSA_OVERRIDE_GFX_VERSION correctly.

### 5.3 PowerShell quirks

`run_remote.sh` invocations with complex multi-line bash (especially ones with `"`, `\`, `$`) get mangled by PowerShell. Preferred pattern:
1. Write the shell command to a file: `scripts/run_sprintX_iterY.sh`
2. Sync via `bash sync_remote.sh`
3. Execute: `bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && bash scripts/run_sprintX_iterY.sh"`

### 5.4 Commit hygiene

- One commit per iter or per phase — small, reviewable units
- Commit message files go in `C:\Users\z00517bz\AppData\Local\Temp\opencode\*.txt` when the message has special chars that break PowerShell (backticks, `$`, etc.)
- Always `git push origin main` after commit unless doing a multi-step work-in-progress (rare)
- Never commit WIP directories that aren't related to current task (e.g. `docs/research/zaya1_8b_technical_report/`, `docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md` — these are unrelated WIP)

### 5.5 Test suites to verify before each major commit

```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
  .venv/bin/python scripts/test_fp16_stability.py && \
  .venv/bin/python scripts/test_sprint1_1_normuon.py && \
  .venv/bin/python scripts/test_sprint1_1_profile.py && \
  .venv/bin/python scripts/test_sprint1_1_bench.py && \
  .venv/bin/python scripts/test_sprint1_phase1.py && \
  .venv/bin/python scripts/test_sprint1_phase2.py"
```

All must pass. Sprint 1.5 adds new tests; add them to the list.

### 5.6 Critical file locations

| What | Where |
|---|---|
| Trainer | `scripts/train_ddp.py` |
| Launch script | `scripts/launch_ddp.sh` |
| NorMuon | `halo_training/normuon.py` |
| Optimizer factory | `halo_training/optimizer.py` |
| Activation monitor | `halo_training/activation_monitor.py` |
| OdinHalo model | `models/odin_halo.py` |
| OdinFlat model | `models/odin_flat.py` |
| Attention components | `models/components/attention.py` (note `_attention_core` helper) |
| fp16 stability knowledge | `knowledge/training/fp16_stability_gfx1151.md` |
| Agent guide | `AGENTS.md` |
| Hard constraints | `CONSTRAINTS.md` |
| Current training status | `STATUS.md` |

### 5.7 Dataset sizes on both machines (verify before DDP)
- `datasets/dolma-10b-odin32k.bin`: 13,678,718,182 bytes on both
- `datasets/wikitext-103-odin32k.bin`: 245,964,976 bytes on both

Transfer over TB4 (10.77.0.x subnet): ~1 GB/s; 13.7 GB takes ~13 seconds.

### 5.8 Training-run naming convention

- `checkpoints/sprint3-iter1-lr2e3/` — Stage 1 iter 1
- `checkpoints/sprint3-iter2-lr3e3/` — Stage 1 iter 2
- `checkpoints/sprint3-iter3-confirm/` — Stage 1 iter 3 (700-step)
- `checkpoints/sprint3-iter4A-nokernel/`, `checkpoints/sprint3-iter4B-kernel/` — Stage 1 iter 4 A/B
- `checkpoints/sprint3-iter5-odinflat-sanity/` — Stage 1 iter 5
- `checkpoints/sprint15-phaseB-clip-{0.5,1.0,2.0}/` — Sprint 1.5 Phase B clip sweep
- `checkpoints/sprint15-phaseB-mup30m-lr-{0.015,0.020,0.0235,0.030}/` — Sprint 1.5 Phase B μP probe
- `checkpoints/sprint15-{C1,C2,C3,D1,D2,E1,E2,E3}/` — Sprint 1.5 factorial cells
- `checkpoints/sprint3a-odinflat-dolma/` — Sprint 3A full run
- `checkpoints/sprint3b-odinhalo-dolma/` — Sprint 3B full run

Scorecard name follows checkpoint dir: e.g. `sprint3-iter1-lr2e3-step-400.json`.

---

## 6. Open decisions requiring future user input

| Decision | When needed | Default if user unavailable |
|---|---|---|
| Sprint 3B exact token budget | After Stage 1.6 | 50h wall (~4.2B tokens for OdinHalo) |
| Sprint 1.5 baseline checkpoint path | Before Phase C | Use Run 2b's step_936; re-train Sprint 1 Run 2 2-epoch if user insists on plan's "step_1869" |
| Sprint 1.5 OdinHalo port for 3B | Before Sprint 3B launch | Leave OFF (plan explicitly defers OdinHalo port to post-Sprint-3) |
| Abort criteria for Sprint 3A/B | If rollbacks exceed 3 | Surface to user; don't auto-abort prematurely |

---

## 7. Rollback paths (if something goes wrong)

### 7.1 Activation monitor fix breaks something
- Symptom: any test fails after S1.0
- Revert: `git revert <S1.0 commit>`; drop the `set_step`/`_should_sample` additions
- Fallback: disable `--activation-monitor` during Stage 1 iters; use old smoke's activation_stats as reference baseline

### 7.2 Stage 1 iter fails unexpectedly
- Any iter failing pass criteria → document in commit, try next lower LR, OR escalate to user if two iters in a row fail
- If infrastructure crash (not LR-related): check rank0.log + Machine B rank1.log; common causes are stale B sync, network hiccup on TB4, disk full

### 7.3 Sprint 1.5 Phase A dev breaks existing suites
- Must run all 6 test suites before each Phase A sub-commit
- If a suite fails, revert the offending commit, investigate, retry

### 7.4 Sprint 3A NaN / rollback storm
- StabilityGuard handles up to 5 rollbacks automatically (LR halves, scaler growth halves each time)
- After 5: `RuntimeError("StabilityGuard: unrecoverable instability")` → read nan_dump, adjust LR or flags, resume via `--resume-from <best_checkpoint>`
- If 3+ rollbacks occur → surface to user before 4th triggers (STATUS.md update + pause)

---

## 8. How to resume this plan after context compaction

1. Read this document in full.
2. Check `STATUS.md` for the most recent "COMPLETE" or "IN PROGRESS" entry.
3. Check `git log --oneline -20` for the last commits — they'll indicate the latest stage.
4. Verify both machines' state: `bash run_remote.sh "ps aux | grep torchrun"` and `bash run_remote_b.sh "ps aux | grep torchrun"`.
5. If any training is RUNNING, poll it until done before starting a new iter.
6. Continue from the first incomplete task in this plan.

**Current execution position (at plan write time):** About to execute **Stage 1.0 — activation monitor fix**. Nothing has been changed yet from the "post-commit-cf9f45a" state. When you see "Build mode" confirmed in the session after compact, proceed with S1.0.

---

## 9. Summary timeline

| Stage | Duration | Compute | User touchpoints |
|:-----:|---------:|--------:|---|
| S1.0 activation monitor fix | ~10 min dev | 0 | 1 commit review (optional) |
| S1.1 OdinHalo lr_2d=2e-3 | ~35 min | 35 min | Scorecard review |
| S1.2 OdinHalo lr_2d=3e-3 | ~35 min | 35 min | Scorecard review |
| S1.3 OdinHalo extended | ~60 min | 60 min | Scorecard review + LR winner confirmation |
| S1.4 optimize-kernels A/B | ~70 min | 70 min | Kernel decision |
| S1.5 OdinFlat dolma sanity | ~20 min | 20 min | Scorecard review |
| S1.6 Stage 1 commit | ~15 min | 0 | Review final STATUS.md |
| **Stage 1 total** | **~4 hrs** | **~3.5 hrs** | ~6 review points |
| Sprint 1.5 Phase A | 3 days dev | 0 | Phase A.8 checkpoint |
| Sprint 1.5 Phase B | 4 hrs | 2 hrs | Probe winners checkpoint |
| Sprint 1.5 Phase C | 6 hrs | 6 hrs | C1/C2/C3 scorecards checkpoint |
| Sprint 1.5 Phase D | 2 days dev + 4 hrs | 4 hrs | Pre-clip value check |
| Sprint 1.5 Phase E | 2 days dev + 4 hrs | 4 hrs | Transfer claim checkpoint |
| Sprint 1.5 Phase F | 0.5 day | 0 | Final commit review |
| **Sprint 1.5 total** | **~9 days** | **~16 hrs** | ~5 review points |
| Sprint 3A OdinFlat dolma | ~50 hrs | 50 hrs | Daily pulse check |
| Sprint 3B OdinHalo dolma | ~40-50 hrs | 40-50 hrs | Daily pulse check |
| **GRAND TOTAL** | **~14-15 days** | **~110 hrs** | |

**Begin execution with S1.0 upon context-compact confirmation.**
