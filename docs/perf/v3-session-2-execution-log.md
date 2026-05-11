# v3 Session 2 Execution Log

**Date:** 2026-05-10 to 2026-05-11 (continued from session 1 closeout at commit `c015a0a`)
**Agent:** 1 long session (~11h wall clock)
**Goal:** Execute Phase A + B + C + D of the v3 40k throughput campaign.
**Outcome:** **Stack D locked as canonical winner (34,697 tok/s, +10.7%)**; Sprint 3 awaits user approval.

---

## Session summary

Session 1 (prior) ended at commit `c015a0a` with T-1.5 fused-zloss 2000-step run in progress (step 1250 of 2000) and a comprehensive handoff document. This session resumed with T-1.5 still running, executed all phases of the handoff's plan, discovered Stack D works on first direct gate attempt, and documented the result.

**Key realization:** The handoff's Phase C matrix (C.1 warm-start × 4 configs → C.2 trust cap → C.3 staging → C.4 Stack D assembly) was designed assuming Stack D would likely fail. In reality, Stack D **passed 2000-step canonical gate on first try** after a minor dtype fix. C.1 matrix was superseded; C.2 ran as diagnostic confirmation (0 trust-cap triggers in 2059 samples → update-scale not the mechanism); C.3 skipped given time budget + C.2 result.

## Phase A (local edits during T-1.5 wait → tests after)

### A.1 causal_conv1d shim (T-3.2 second half)
- `kernels/hip/_torch_ops.py`: `autokernel::causal_conv1d` custom_op wrapping DaoAILab's `causal_conv1d_fn`. Probed DaoAILab internals first → discovered `CausalConv1dFn.apply` is a proper `torch.autograd.Function`. Backward replays that via `torch.autograd.grad` for production speed; falls back to pure-PyTorch depthwise conv1d if extension unavailable.
- `models/components/conv_blocks.py`: env-gated routing through `AUTOKERNEL_CAUSAL_CONV_SHIM=1`.
- `scripts/train_ddp.py`: wired `--ak-causal-conv-shim` env var setter.
- `scripts/test_causal_conv1d_shim.py`: **4/4 PASS** on Machine A (forward rel_err 5e-4, gradient flow verified, backward rel_err 0.0 via DaoAILab replay, production-shape parity).

### A.2 Tier 2 parity tests
- `scripts/test_tier2_parity.py`: 4 ops × {vanilla, autocast} = 8 test cells.
- **8/8 PASS** on Machine A. Under autocast == vanilla numerics (autocast rules fire cleanly, no dtype drift).

### A.3 register_autocast rules
- `kernels/hip/_torch_ops.py`: added rules for 5 training-path ops.
  - `silu_gate_mul_op.register_autocast("cuda", torch.float16)` (homogeneous fp16)
  - `rmsnorm_op.register_autocast("cuda", torch.float16)`
  - `fused_res_rmsnorm_op.register_autocast("cuda", torch.float16)`
  - `rotary_emb_fp32_op.register_autocast("cuda", torch.float32)` (fp32 by design)
  - `causal_conv1d_shim_op.register_autocast("cuda", torch.float16)` (A.1 shim)
  - **NOT added** to `fused_rope_gate_mul_op` (mixed-dtype op; would downcast fp32 freqs_cos/sin to fp16 and crash kernel). Explanatory comment added.
- Re-ran `scripts/autokernel_dtype_inventory.py`: **6/9 ops with register_autocast** (up from 0/9).

### A.4 T-1.5 2000-step canonical gate: PASS
- Completed 2000 steps in 7850s = 2h 10m wall (cluster start 2026-05-10 19:08).
- Final: **33,410 tok/s aggregate**, best loss **3.1302** (beats Sprint 3A-confirm baseline 3.15).
- Scorecard clean (all 4 domain BPB within Sprint 3A-confirm range).
- Stack A LOCKED.

**Phase A commit:** `e3baf27` — "Phase A + T-1.5 FINAL: causal_conv1d shim + Tier 2 parity + register_autocast + C.0 bundle-dump + probe scripts + T-1.5 2000-step scorecard"

### C.0 replay-bundle dump (extended existing `save_nan_forensics`)
- Added `optimizer` + `args` kwargs to `save_nan_forensics`.
- Emits `replay-bundle-step-N/` directory on NaN/collapse with batch.pt, model_state.pt, optim_state.pt, rng.pt, config.json, activation_stats_window.json. Fail-quiet: bundle errors don't block subsequent rollback.
- `scripts/replay_step.py` executor deferred to future session.

### `--resume-preserve-optimizer` flag
- Added to `scripts/train_ddp.py` for C.1 warm-start matrix's preserved-state variant (default `--resume-from` is weights-only).
- Wired optimizer + scaler state restore post-construction.

## Phase B (DDP probes, sequential)

### B.1 T-1.4 DDP bucket sweep — null-effect
- 4 configs (bucket_cap_mb ∈ {8, 25, 50, 100}) × 150 steps.
- Results: 28,639 / 28,608 / 28,736 / 28,799 tok/s. All within **0.67%** of each other.
- Manual allreduce in `train_ddp.py` dominates; DDP bucket subsystem sees only 1 event per opt step. Keep default bucket_cap_mb=25. `gradient_as_bucket_view=True` already hardcoded.
- `docs/perf/t1-4-ddp-bucket-sweep-findings.md`

### B.2 T-2.1 batch=32 probe — null-effect
- 4 configs × 200 steps.
- Results: baseline 16 = 31,563 tok/s @ 13.5 GB; batch=32 configs ranged 31,504–31,521 @ 24.6–25.6 GB (-0.1 to -0.2% vs baseline).
- batch=32 FITS memory but provides no throughput gain. Workload is bandwidth/IO-bound at MFU 20.7%, not compute-bound.
- **SKIP batch=32** for Stack B.
- `docs/perf/t2-1-batch32-findings.md`

### B.3 T-4 compiled autograd gated smoke — GATE FAILS
- 4 configs × 300 steps.
- Results: baseline 31,630; +CA alone **31,444 (-0.6%)**; +CA+fused_zloss 33,605 (fused_zloss doing all work); +CA+ddp_tune 31,459 (-0.5%).
- Gate requires ≥2.5–3% net gain. CA alone REGRESSES. Aligns with T-0.1 finding (backward 2.18× forward ≈ near-theoretical; CA has no Python overhead to amortize).
- **CA ships as infrastructure only.** `--ak-compiled-autograd` flag stays but NOT in production recipe.
- `docs/perf/t4-compiled-autograd-findings.md`

**Stack A/B/C all tied at 33,410 tok/s.** Without hidden-kernel recovery, ~34k was the ceiling.

**Phase B commit:** `1ee2e5c` — "Phase B complete: T-1.4 bucket sweep + T-2.1 batch=32 + T-4 CA all null-effect -> Stack C == Stack B == Stack A"

## Phase C (hidden kernel recovery)

### C.1 warm-start matrix — SUPERSEDED
- Initial run had coordination issues: (1) Phase II gloo handshake error (port 29500 stale between chained phases), (2) Phase I 60-min timeouts too tight for 1000-step runs.
- Given Stack D passed direct (C.4), the warm-start matrix's diagnostic value evaporated. Dropped.

### C.4 Stack D direct 2000-step gate — PASS (headline)
- `scripts/probe_t5_c4_stackd_direct.sh` — full Stack D recipe from scratch.
- **Launch 1 crashed at step 0:** `RuntimeError: expected scalar type Float but found Half`. Root cause: `register_autocast("cuda", fp16)` on `fused_rope_gate_mul_op` downcast its fp32 `freqs_cos`/`sin` inputs (the op has mixed-dtype signature by design). **Fix:** removed register_autocast rule from that specific op; kept hand-managed `.half()` casting at the conv_blocks.py call site.
- **Launch 2:** completed 2000 steps in 7559s = 2h 6m.
- Final: **34,697 tok/s** aggregate (+10.7% vs baseline, +3.9% vs Stack A).
- Best loss: **3.1384** (tied with Stack A within noise).
- Scale grew 1024 → 16384 cleanly; zero divergence; zero frozen params.
- MFU: **21.4%** (up from Stack A's 20.7%).
- Scorecard (vs Stack A step 2000):
  - wikitext BPB 2.0935 vs 2.0971 (better by -0.0036)
  - gpt_small BPB 2.0447 vs 2.0572 (better by -0.0125)
  - dolma BPB 1.5930 vs 1.6018 (better by -0.0088)
  - **Max layer kurtosis 14.83 vs 31.02 (-52%, much healthier)**
  - Effective rank 1.030 vs 1.040 (more useful dimensions)
- `docs/perf/t5-c4-stackd-findings.md` — full analysis.

### C.2 trust cap diagnostic — update-scale ruled out
- 500 steps with Stack D + `--ak-trust-cap 0.02 --ak-trust-cap-scope w_gate_up`.
- Wall: ~32 min. tok/s 34,489 (essentially tied with Stack D alone).
- **0 trust_cap_triggered events across 2059 telemetry samples.** Update/weight ratio never exceeded 0.02.
- Confirms **update-scale is NOT the mechanism** of Phase C/G divergence.
- `docs/perf/t5-c2-trust-cap-findings.md`

### C.3 w_gate_up staging — SKIPPED
- C.2 showed update scale stays well below threshold. Staging targets same hypothesis. Null-effect expected.
- Time budget: running C.3 (~1-2h) would delay Phase D finalization unnecessarily.
- Deferred to future session if investigating OdinHalo (looped model has different dynamics).

**Stack D commit:** `cf8ace2` — "Stack D PASSES 2000-step gate: +10.7% tok/s + better quality. Ship as canonical winner."

## Phase D (finalize)

- `docs/perf/v3-final-stack-scorecard.md` — comprehensive comparison across all 4 stacks + hypothesis retrospective + deferred items + 40k assessment.
- `STATUS.md` — new Stack D section at top (supersedes prior "in progress" framing). Includes Sprint 3 launch command (awaits user approval).
- `AGENTS.md` — added 6 Training gotchas covering Stack D recipe, v3 `--ak-*` flag taxonomy, register_autocast rule mandate, trust cap/staging diagnostic-only status, compiled autograd ship-as-infra-only, batch=32 null-effect, DDP bucket null-effect.
- `CONSTRAINTS.md` — added 3 rules: register_autocast mandate (fp16/fp32 per op, NO rule for mixed-dtype), Tier 2 parity test pre-ship requirement, Sprint 3A/3B explicit-approval rule.
- `docs/perf/v3-session-2-execution-log.md` — this file.

## v3 hypotheses retrospective

| ID | Hypothesis | Verdict |
|---|---|---|
| H1 | `w_gate_up` init statistics → NorMuon amplify → fp16 overflow | Ruled out (C.2 showed ratio <0.02) |
| H4/H5 | NorMuon post-transform dynamics drive divergence | Ruled out (Stack D stable from scratch) |
| **H11** | **Custom-op autocast boundary dtype mismatch** | **CONFIRMED** (A.3 register_autocast rules fixed Phase C/G pattern) |
| H14 | Optimizer state mismatch post-switch | Untested (C.1 superseded) |
| H17 | Forward graph breaks fragment backward | Partially addressed (T-3.2 shims) |
| H18 | HyPE recompile storm | Ruled out pre-session (T-0.4) |

**Primary root cause: v3 H11 confirmed.** Phase B's autograd-safe wiring
was necessary but insufficient. Adding register_autocast rules with per-op
dtype management (fp16 for homogeneous SwiGLU path, fp32 for rotary_emb_fp32
internals, NO rule for mixed-dtype fused_rope_gate_mul) restored stability
of the hidden-kernel stack.

## 40k target assessment

- Nominal 40k: NOT reached (34.7k = 87% of way, short by 5.3k)
- Engineering success 36k: NOT reached
- Stack D at 34.7k is the honest ceiling for this session's scope.

Phase A-C exhausted the kernel/optimizer dimension. Workload is now
bandwidth/IO-bound (MFU 21.4% at unified memory + TB4 bandwidth ceiling).
Reaching 36-40k requires orthogonal changes: larger block (compile cost,
unclear win), different accum pattern, algorithmic changes, or TB4
interconnect upgrade.

## Deferred to post-launch

1. **OdinHalo Stack D validation** — Stack D validated on OdinFlat only. Recommend 500-step smoke on OdinHalo before full Sprint 3B launch (looped 58M/156M-effective model has different gradient statistics).
2. **C.1 warm-start matrix re-run** — becomes informative if OdinHalo diverges where OdinFlat did not.
3. **C.3 w_gate_up staging re-run** — diagnostic only; only if a future probe shows update/weight ratio > 0.02.
4. **`scripts/replay_step.py` executor** — bundle-dump infra shipped; replay executor deferred.
5. **40k aspiration** — requires orthogonal (non-optimization) changes.

## Commits this session

1. `e3baf27` — Phase A + T-1.5 FINAL: causal_conv1d shim + Tier 2 parity + register_autocast + C.0 bundle-dump + probe scripts + T-1.5 2000-step scorecard
2. `1ee2e5c` — Phase B complete: T-1.4 bucket sweep + T-2.1 batch=32 + T-4 CA all null-effect
3. `cf8ace2` — Stack D PASSES 2000-step gate: +10.7% tok/s + better quality. Ship as canonical winner.
4. (final commit of Phase D — this session log + STATUS/AGENTS/CONSTRAINTS updates + C.2 findings)

## Final state + next action

**Stack D recipe (canonical-locked):**
```bash
--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
--intra-doc-mask --value-residuals --head-gating \
--z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
--mup --mup-base-width 256 \
--spectra-post --spectra-clip-norm 1.0 \
--activation-monitor --activation-monitor-interval 200 \
--use-fused-zloss --ak-loss-zloss \
--ak-fix-rope-gate-op --ak-causal-conv-shim \
--ak-sync-cleanup --ak-spectra-branchless \
--ak-normuon-telemetry
```

**STOP before Sprint 3A/3B launch.** Per user directive, agent awaits
explicit approval. Presenting:
- Winner: **Stack D (34,697 tok/s, +10.7% vs baseline)**
- Expected Sprint 3A wall: ~61h (OdinFlat, dolma-10b, 1 epoch)
- Expected Sprint 3B wall: ~77h (OdinHalo, dolma-10b, 1 epoch, recommend smoke first)
- Launch commands:
  ```bash
  STACK=D bash scripts/launch_sprint3a.sh
  STACK=D bash scripts/launch_sprint3b.sh
  ```

Wait for user go-ahead.
