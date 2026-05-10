# AutoKernel 40k Throughput Campaign — Locked Execution Plan

**Campaign origin:** Response to research engineer v3 (`docs/research/autokernel_40k_v3_research_engineering_response.md`, 2026-05-11)
**Locked on:** 2026-05-11 after v1/response/v2/addendum/v3 exchange
**Repo SHA at lock:** `98f2f39` (main, post-v3-addendum commit)
**User directives:**
- Accept v3's P0→P5 reordering (SPECTRA first, compiled autograd as gated smoke)
- Adopt v3 §5.3 STATUS.md language verbatim (nominal=40k, success=36k, strong=38k, stretch=40k)
- Accept v3's three-tier autocast audit structure
- Launch Sprint 3A/3B only after T-6 stack integration completes (Week 3)

**Single source of truth** for the throughput campaign. All per-phase details consolidated here. Do not re-derive from v1/v2/v3 — use this doc.

---

## 1. Target framing

```text
Nominal target:        40k aggregate tok/s across 2× Strix Halo systems
Engineering success:   ≥36k stable aggregate tok/s
Strong success:        ≥38k stable aggregate tok/s
Stretch outcome:       40k stable aggregate tok/s
Non-negotiable:        no new frozen params, no GradScaler collapse, clean 2000-step loss parity
```

Speedups from frozen params, severed gradients, or unstable fp16 behavior do not count.

Blanket `--optimize-kernels` remains disabled for production until granular kernel flags pass per-feature throughput, gradient-flow, dtype, and 2000-step stability gates.

---

## 2. Baseline and arithmetic

```text
baseline aggregate throughput    ≈ 31,331 tok/s
tokens per optimizer step        = 16 × 512 × 8 × 2 = 131,072
baseline step time               ≈ 4.18 s
target step time (40k)           ≈ 3.28 s
required step-time reduction     ≈ 21.8%
```

Intermediate targets:

| tok/s | Required step-wall reduction |
|---:|---:|
| 33.5k | 6.5% |
| 35k | 10.0% |
| 36k | 12.9% |
| 37k | 15.3% |
| 38k | 17.6% |
| 40k | 21.8% |

---

## 3. Phase execution plan (T-0 → T-6)

### T-0: Instrumentation (Days 0-1)

| Item | Purpose | Source |
|---|---|---|
| T-0.1 Backward profile | baseline backward breakdown | DONE: `docs/perf/backward-breakdown.md` |
| T-0.2 Per-param NorMuon telemetry | catch optimizer-side divergence | v3 §9.1 schema |
| T-0.3 Sync-point audit | static grep for `.item()` | DONE: `docs/perf/sync-point-audit.md` |
| T-0.4 Graph-break inventory | compile-time barrier catalog | DONE: `docs/perf/graph-breaks-inventory.md` |
| T-0.5 Granular `--ak-*` flags | replace monolithic `--optimize-kernels` | v3 §8.1 + §8.2 |
| T-0.6 DDP allreduce trace | verify no_sync, measure overlap | v3 §9.3 schema |
| T-0.7 Dtype/autocast trace | catch AMP mismatches | v3 §9.2 schema |
| T-0.8 Sync counter in profiler | track `.item()` elimination progress | v3 §9.4 schema |

**Exit gate:** all instrumentation live; DDP no_sync verified correct (or fixed).

### T-1: Quick safe wins (Days 1-2)

| Item | Expected tok/s gain | Gate |
|---|---:|---|
| T-1.1 Branchless SPECTRA | +1-4% | ≥1% + update parity + 500-step smoke |
| T-1.2 Other hot-path `.item()` removal | +1-3% | no regression |
| T-1.3 DDP no_sync verification | 1-5% (huge if broken) | allreduce count correct |
| T-1.4 DDP bucket sweep + grad_as_bucket_view | +1-3% | no learning change |
| T-1.5 Fused z-loss 2000-step validation | +3-8% | 2000-step parity |

**Exit gate:** ≥33.5k tok/s stable.

### T-2: Batch + optimizer path (Days 2-4)

| Item | Expected tok/s gain |
|---|---:|
| T-2.1 batch=32/accum=4 probe | +0-5% |
| T-2.2 NorMuon telemetry analysis | informs T-2.3 |
| T-2.3 NorMuon implementation cleanup | +2-5% |
| T-2.4 NorMuon algorithmic experiments (ns_steps=3 vs 5) | 0-3% or stability |

**Exit gate:** ≥36k tok/s stable.

### T-3: Dtype/autocast audit (Days 4-6)

| Item | Scope |
|---|---|
| T-3.1 Tier 0 inventory | all 7 training-path custom ops |
| T-3.2 Tier 1 graph-break fixes | fused_rope_gate_mul + causal_conv1d |
| T-3.3 Tier 2 deep parity | silu_gate_mul, RMSNorm, fused_res_rmsnorm, rope_gate_mul, causal_conv1d |
| T-3.4 Replay harness | `scripts/replay_step.py` for divergence-prone batches |

**Exit gate:** Tier 0 inventory documented; Tier 1 ops graph-break-free; Tier 2 cos ≥0.9999.

### T-4: Compiled autograd gated smoke (Days 6-7)

Budget: 1-2 days max.

| Item | Duration |
|---|---|
| T-4.1 Smoke: single-node native + compiled_autograd | 2h |
| T-4.2 Smoke: + fused zloss | 2h |
| T-4.3 Smoke: DDP + best bucket | 4h (includes DDP overlap measurement) |
| T-4.4 Gate decision | stop-or-continue |

**Continuation gate:** ≥2.5-3% net DDP tok/s AND no overlap regression AND no recompile churn.

**If gate fails:** compiled autograd ships as infrastructure only.

### T-5: Hidden kernel recovery (Days 7-12)

| Item | Notes |
|---|---|
| T-5.1 Warm-start matrix | 4 configs: 500/1000 × preserved/fresh optimizer |
| T-5.2 w_gate_up update-scale staging (0.25 → 1.0 ramp) | v3 §10 E8.4 |
| T-5.3 Post-NorMuon trust cap (τ=0.02) | v3 §7.2 composition order |
| T-5.4 triton_op SwiGLU retry | conditional on T-4 passing |

**Exit gate:** ≥1 `--ak-swiglu-*` config passes 2000-step gate.

### T-6: Stack integration + Sprint 3 launch (Days 12-18)

| Stack | Composition | Expected |
|---|---|---:|
| Stack A | branchless SPECTRA + fused zloss + DDP tuned + sync cleanup | 33.5-36k |
| Stack B | A + batch=32 + NorMuon cleanup | 36-38.5k |
| Stack C | B + compiled autograd (if T-4 passed) | 37-39k |
| Stack D | B/C + recovered hidden kernels | 38.5-40k+ |

T-6.x:
- Assemble Stack A/B/C/D launch scripts
- 2000-step gate on best-passing stack
- Update STATUS.md / AGENTS.md / CONSTRAINTS.md with final recipes
- Launch Sprint 3A (~61h) on best stack
- Launch Sprint 3B (~77h) after or overlapping 3A

---

## 4. Flag taxonomy (v3 §8, locked)

### 4.1 Core granular kernel/runtime flags

```
--ak-loss-ce                 Route logits through kernel.ce_full (safe CE path)
--ak-loss-zloss              Bake z-loss into fused CE (Phase B.5)
--ak-swiglu-fwd              Use HIP silu_gate_mul for forward
--ak-swiglu-bwd              Use HIP silu_gate_mul backward (vs PyTorch fallback)
--ak-rmsnorm                 Use HIP rmsnorm custom_op
--ak-res-rmsnorm             Use HIP fused_res_rmsnorm custom_op
--ak-rope                    Use HIP rotary_emb_fp32 custom_op
--ak-rope-gate               Use HIP fused_rope_gate_mul (requires T-3.2 fix)
--ak-causal-conv             Use DaoAILab causal_conv1d_fn (requires T-3.2 shim)
--ak-qkv                     Use fused QKV custom_op
--ak-ple-gate                Use HIP fused_ple_gate
--ak-normuon                 Use NorMuon optimizer (vs AdamW)
--ak-compiled-autograd       Enable torch._dynamo.config.compiled_autograd
--ak-triton-visible          Route through torch.library.triton_op (vs custom_op)
--ak-sync-cleanup            Branchless SPECTRA + deferred .item() aggregation
--ak-ddp-tune                gradient_as_bucket_view=True + tuned bucket_cap_mb
```

### 4.2 v3 add-on flags

```
--ak-spectra-branchless         Specific branchless SPECTRA patch
--ak-autocast-tier {none,t1,all} Which tier of autocast rules to apply
--ak-dtype-trace                Emit dtype trace JSONL
--ak-fix-rope-gate-op           Use fixed rope_gate_mul custom_op
--ak-causal-conv-shim           Use shimmed causal_conv1d custom_op
--ak-normuon-telemetry          Emit per-param update telemetry JSONL
--ak-normuon-impl-opt           Enable NorMuon implementation optimizations
--ak-batch32-probe              (not a flag; use BATCH=32 ACCUM=4 env in launch_ddp.sh)
--ak-trust-cap <float>          Post-NorMuon trust cap threshold (default 0.0=off)
--ak-trust-cap-scope {none,w_gate_up,spiking,all_2d}
--ak-w-gate-up-scale <float>    Initial w_gate_up update scale
--ak-w-gate-up-ramp-steps <int> Steps over which to ramp scale to 1.0
```

### 4.3 Flag rules

1. No flag implicitly enables all optimized kernels.
2. Every flag is independently measurable.
3. Every flag records itself in checkpoint/training metadata.
4. Every flag appears in run manifest + profiler summary + JSONL logs.
5. Hidden-kernel flags must pass gradient-flow + dtype audit before 2000-step gate.

---

## 5. Telemetry schemas (v3 §9, locked)

### 5.1 NorMuon per-param telemetry (JSONL, enabled by `--ak-normuon-telemetry`)

```json
{
  "step": 0, "param_name": "", "shape": [0,0],
  "dtype_param": "torch.float16", "dtype_grad": "torch.float16", "dtype_update": "torch.float32",
  "param_norm": 0.0, "raw_grad_norm": 0.0, "momentum_norm": 0.0,
  "post_ns_norm": 0.0, "spectra_sigma1": 0.0, "spectra_scale": 1.0,
  "effective_lr": 0.0,
  "update_norm_pre_trust": 0.0, "update_norm_post_trust": 0.0,
  "trust_ratio_pre": 0.0, "trust_ratio_post": 0.0, "trust_cap_triggered": false,
  "maxabs_param": 0.0, "maxabs_grad": 0.0, "maxabs_update": 0.0,
  "grad_isfinite": true, "update_isfinite": true
}
```

Sample frequency: every step for first 20, then every 50, then every 10 after first warning, always on non-finite.

### 5.2 Custom-op dtype trace (enabled by `--ak-dtype-trace`)

```json
{
  "step": 0, "op_name": "",
  "forward_input_dtypes": [], "forward_output_dtype": "",
  "backward_input_dtype": "", "backward_output_dtypes": [],
  "internal_accumulation": "unknown",
  "has_register_autocast": false, "has_register_autograd": true, "has_register_fake": true,
  "graph_break_source": false, "enabled_by_flags": []
}
```

Emit once at registration + once at first call site per op.

### 5.3 DDP communication telemetry (always-on per opt step)

```json
{
  "step": 0, "accum_steps": 8, "no_sync_expected_microsteps": 7,
  "allreduce_count": 0, "allreduce_total_ms": 0.0,
  "first_allreduce_start_ms_after_backward_start": 0.0,
  "last_allreduce_end_ms": 0.0, "overlap_ratio_estimate": 0.0,
  "bucket_cap_mb": 25, "gradient_as_bucket_view": true, "static_graph": false
}
```

### 5.4 Sync counter in profiler summary

```json
{
  "profile_window_steps": 10,
  "aten_item_count": 0, "hipMemcpyWithStream_cpu_wall_s": 0.0,
  "known_hot_syncs": {
    "spectra_sigma1_item": 0, "loss_item": 0,
    "valid_global_sum_item": 0, "jsonl_logging": 0
  }
}
```

---

## 6. Pre-committed interpretation table (v3 §14)

| Observation | Most likely meaning | Action |
|---|---|---|
| Branchless SPECTRA +2-4% | Sync audit found real critical-path stalls | Keep, make default |
| Branchless SPECTRA ~0% | CPU syncs mostly hidden by GPU work | Keep if no cost, don't overcount |
| Fused zloss stable +5-8% | Loss path is safe pillar | Stack everywhere |
| Fused zloss diverges | denominator/masking/dtype bug | Fix CE/zloss parity first |
| DDP no_sync huge gain | accumulation communication bug | Fix immediately, rebaseline all |
| DDP bucket sweep +1-3% | overlap suboptimal | Keep tuned bucket |
| batch=32 fits +5% | clean utilization win | Promote early |
| batch=32 requires checkpoint + loses speed | memory tradeoff not worth it | Drop |
| NorMuon update_ratio spikes pre-divergence | optimizer-scale failure | trust cap / staging |
| no update spikes, activation explodes | forward/residual/fp16 numerical issue | fp32 islands / activation guard |
| autocast rule changes stability | dtype boundary mismatch | add rule + tests |
| compiled autograd + local speed, DDP flat | allreduce overlap regression | reject CA for production |
| compiled autograd +3% net DDP | useful infrastructure | continue cautiously |
| hidden kernels stable only after step 1000 | warmup-local instability | delayed production enable possible |
| hidden kernels fail immediately after enable | kernel/dtype/backward mismatch | return to replay harness |

---

## 7. Mandatory quality gates

1. No newly frozen params
2. No param with unexpected `grad_fn=None` path
3. No GradScaler collapse
4. No nonfinite update events
5. No sustained activation/fp16-headroom danger trend
6. 500-step smoke before stacking
7. 2000-step learning parity before production
8. Per-feature tok/s measured against same baseline window
9. Compile time separated from steady-state throughput
10. DDP overlap measured whenever compiled autograd or graph changes are involved

## 8. Explicit reject conditions

A feature is rejected if it:
- increases tok/s by freezing or detaching gradients
- requires hiding a frozen-param audit failure in an allowlist
- improves local kernel microbench but not end-to-end tok/s
- speeds single-node but regresses DDP aggregate throughput
- requires lower LR that damages validated loss curve (unless user explicitly accepts slower convergence)
- passes 200 steps but fails before 2000 with scaler collapse or rising loss pathology

---

## 9. Outcome probabilities

| Outcome | Probability | tok/s |
|---|---:|---:|
| Ship Stack A only | 80% | 33.5-36k |
| Ship Stack B | 60% | 36-38k |
| Ship Stack C (+ compiled autograd if T-4 passes) | 25% | 37-39k |
| Ship Stack D (+ recovered hidden kernels) | 15% | 38.5-40k |
| Fall back to current 31.3k | 5% | 31k |

(Probabilities overlap — Stack B ship implies Stack A ship.)

---

## 10. Stop-and-consult-user conditions

I will stop execution and ask user if:
- T-1 fails to reach 33.5k (reassess T-2/T-3)
- T-0.6 reveals DDP `no_sync` is broken (rebaseline)
- T-4 smoke passes but DDP overlap regresses ≥2%
- T-5 warm-start matrix all fail (accept Stack B ceiling)
- Any 500-step smoke fails with pattern not covered by §6 table
- Timeline slips >3 days from 18-day plan

---

## 11. Cross-reference documents

| Document | Role |
|---|---|
| `docs/research/autokernel_halo_strix_40k_throughput_experiment_guide.md` | v1 engineer guide |
| `docs/research/autokernel-40k-guide-engineering-response.md` | our response to v1 |
| `docs/research/autokernel_40k_revised_engineering_handoff_v2.md` | v2 revised handoff |
| `docs/research/autokernel-40k-engineering-v3-addendum.md` | our v3 request |
| `docs/research/autokernel_40k_v3_research_engineering_response.md` | v3 final response |
| **`docs/research/autokernel-40k-v3-execution-plan.md`** | **this document (single source of truth)** |
| `docs/perf/backward-breakdown.md` | T-0.1 findings |
| `docs/perf/sync-point-audit.md` | T-0.3 findings |
| `docs/perf/graph-breaks-inventory.md` | T-0.4 findings |
| `docs/research/autokernel-stability-research-brief.md` | full 14K-word stability brief |
| `STATUS.md` | throughput campaign + current state |
| `CONSTRAINTS.md` | autograd-safety + hardware rules |
| `AGENTS.md` | training workflow reference |

---

*Plan locked 2026-05-11. Execution begins at T-0.5 (granular flags) + T-0.2 (NorMuon telemetry).*
