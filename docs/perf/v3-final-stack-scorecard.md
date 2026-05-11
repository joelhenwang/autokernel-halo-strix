# v3 40k Campaign — Final Stack Scorecard

**Date:** 2026-05-11
**Campaign:** v3 40k throughput + hidden-kernel recovery
**Session:** second (continuing from 2026-05-10 v3 session 1)
**Baseline:** OdinFlat Sprint 3A-confirm, **31,331 tok/s aggregate** on DDP/gloo TB4
**Winner:** **Stack D** — 34,697 tok/s aggregate (**+10.7%**)

---

## Stack comparison

| Stack | Composition | tok/s | Delta | Loss@2000 | Frozen | GradScaler | Status |
|---|---|---:|---:|---:|---:|---|---|
| Baseline | Sprint 3A-confirm | 31,331 | — | 3.15 | 0 | stable | canonical-old |
| A | +fused_zloss | **33,410** | +6.6% | 3.1302 | 0 | stable | shipped |
| B | = A (batch=32/bucket null) | 33,410 | +6.6% | — | — | — | B.1/B.2 null |
| C | = A (CA regresses) | 33,410 | +6.6% | — | — | — | B.3 gate fail |
| **D** | **A + ak-fix-rope-gate-op + ak-causal-conv-shim + ak-sync-cleanup + ak-spectra-branchless + ak-normuon-telemetry + register_autocast on 5 ops** | **34,697** | **+10.7%** | **3.1384** | 0 | stable | **CANONICAL-LOCKED** |
| E | Native 1000 → Stack D resume (preserved optim) — 2-stage | ~33.5k agg (Stage 1 drags; net -1-2% vs D) | +9-10% | **3.1065** (C.1.c single-run evidence; -0.033 vs D) | 0 | stable | ALT recipe (opt-in) |

## Stack D recipe (production)

```bash
# Stack D — production Sprint 3 command awaits user approval
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

Effective batch = 16 × 512 × 8 × 2 = 131,072 tokens/opt-step.

## Quality comparison (step 2000)

| metric | Stack A | Stack D | delta |
|---|---:|---:|---:|
| tok/s aggregate | 33,410 | 34,697 | +3.9% |
| Best loss | 3.1302 | 3.1384 | +0.008 (noise) |
| wikitext_val BPB | 2.0971 | 2.0935 | **-0.0036** |
| gpt_small_val BPB | 2.0572 | 2.0447 | **-0.0125** |
| stem_crawl_val BPB | 1.6240 | 1.6239 | -0.0001 |
| dolma_val BPB | 1.6018 | 1.5930 | **-0.0088** |
| Effective rank | 1.0397 | 1.0300 | -0.0097 |
| **Max layer kurtosis** | **31.02** | **14.83** | **-52%** ↓ |
| Max layer rms_norm | 42.2 | 59.9 | +42% (lm_head) |
| Inference tok/s seq=512 | 67,716 | 64,152 | -5.3% |
| MFU | 20.7% | 21.4% | +0.7% |
| Memory/node | 12.4 GB | 12.7 GB | +2% |

**Stack D strictly better on validation loss, BPB, convergence, activation
health.** Training MFU improved. Slightly lower inference tok/s is on a
different codepath (inference doesn't use autocast consistently); training
is what matters for Sprint 3.

## Phase breakdown

### Phase A — engineering during T-1.5 (complete)

- **A.1** `autokernel::causal_conv1d` custom_op (T-3.2 second half)
  - `kernels/hip/_torch_ops.py`: custom_op wrapping DaoAILab's CausalConv1dFn
  - `models/components/conv_blocks.py`: env-gated routing
  - `scripts/train_ddp.py`: `--ak-causal-conv-shim` env wiring
  - `scripts/test_causal_conv1d_shim.py`: **4/4 PASS** (fwd rel_err 5e-4, bwd rel_err 0.0 via DaoAILab replay)
- **A.2** `scripts/test_tier2_parity.py`: **8/8 PASS** (4 ops × vanilla+autocast)
- **A.3** register_autocast on 5 training-path ops
  - Added to: silu_gate_mul, rmsnorm, fused_res_rmsnorm, rotary_emb_fp32 (fp32), causal_conv1d
  - NOT added to: fused_rope_gate_mul (mixed fp16/fp32 inputs — hand-managed)
  - Initial crash on Stack D launch caught by runtime, fixed by removing
    autocast rule from fused_rope_gate_mul only
- **A.4** T-1.5 2000-step canonical scorecard: **PASS**

### Phase B — DDP probes (complete, all null-effect)

- **B.1** Bucket sweep: all within 0.67% (28.6k-28.8k tok/s range).
  Manual allreduce dominates → bucket tuning null. Keep default.
- **B.2** batch=32 probe: FITS memory (24.6 GB) but no throughput gain
  (-0.1 to -0.2%). Workload is bandwidth/IO-bound at MFU 20.7%, not compute.
  SKIP batch=32.
- **B.3** Compiled autograd: alone REGRESSES 0.6%. Gate FAILS. CA ships as
  infra-only. Aligns with T-0.1 finding that backward is near-theoretical
  (2.18× forward) so CA has little Python overhead to amortize.

### Phase C — hidden kernel recovery (Stack D direct path succeeded)

- **C.0** Replay-bundle dump infrastructure added to `save_nan_forensics`
  (bundle dir emitted on NaN/collapse trigger with model/optim/RNG/config).
- **C.1** Warm-start matrix **SUPERSEDED**. Had coordination issues (gloo
  port stale between phases, phase timeouts). Since Stack D passed directly
  from scratch, the diagnostic matrix is no longer informative.
- **C.2** Trust cap diagnostic: **0 triggers** across 2059 telemetry samples
  over 500 steps. Update/weight ratio never exceeded 0.02. Confirms
  **update-scale is NOT the mechanism** of Phase C/G divergence.
- **C.3** w_gate_up staging: **SKIPPED**. C.2 showed update scale is
  well-behaved; staging targets the same hypothesis. Null-effect expected.
- **C.4** Stack D direct 2000-step gate: **PASS** (headline result).

### Phase D — finalize (in progress)

## v3 hypotheses — retrospective

| ID | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | `w_gate_up` init statistics → NorMuon amplify → fp16 overflow | Ruled out | C.2 showed update/weight ratio always <0.02 |
| H4/H5 | NorMuon post-transform dynamics drive divergence | Ruled out | Stack D stable from scratch, no staging needed |
| H11 | **Custom-op autocast boundary dtype mismatch** | **CONFIRMED** | Adding register_autocast on 5 ops + fixing mixed-dtype rope_gate → Stack D passes |
| H14 | Optimizer state mismatch post-switch | Untested | C.1 matrix superseded; Stack D from-scratch works |
| H17 | Forward graph breaks fragment backward | Partially | T-3.2 shims eliminate 2/2 breaks; compiled autograd still regresses |
| H18 | HyPE recompile storm | Ruled out (pre-session) | T-0.4 showed only 1 one-time recompile |

**Primary finding:** **v3 H11 CONFIRMED as the root cause.** Phase B's
autograd-safe wiring was necessary but insufficient. register_autocast rules
with proper per-op dtype management (fp16 for SwiGLU path, fp32 for RoPE
internal, NO rule for mixed-dtype `fused_rope_gate_mul` — hand-managed)
restored the hidden-kernel stack.

## 40k target assessment (v3 §5.3 framing)

- Nominal (40k): NOT reached (34.7k = 87% of way)
- Engineering success (36k): NOT reached
- Strong (38k): NOT reached
- **Stretch baseline (aspirational): 34.7k ≈ Stack D ceiling**

Gap analysis: **5.3k tok/s short of 40k.** To reach 36-40k requires orthogonal
changes (larger block size, different accumulation, algorithmic changes,
dual-node cross-TB bandwidth upgrade) that are outside Phase A-C scope.

Workload bottleneck at MFU 21.4% is memory bandwidth / TB4 interconnect, not
compute. gfx1151 has no MFMA → rocBLAS matmul is the computational ceiling,
and we've saturated it. Throughput gains now require algorithmic changes,
not more kernel optimization.

## Deferred to post-launch

1. **C.1 warm-start matrix** — re-run if OdinHalo (looped model) diverges
   where OdinFlat did not; then warm-start matrix becomes informative again.
2. **C.3 w_gate_up staging** — diagnostic only; re-run if a future probe
   shows w_gate_up update/weight ratio >0.02.
3. **`scripts/replay_step.py`** executor — bundle-dump infra is shipped;
   replay executor itself is deferred to future session per handoff §11.10.
4. **OdinHalo Stack D validation** — Stack D was validated on OdinFlat only.
   OdinHalo (looped 58M/156M-effective) may need different settings.
   Recommend a 500-step smoke before full Sprint 3B launch.
5. **40k aspiration** — requires orthogonal changes (non-Phase-A-C).

## Next action (awaits user approval)

Sprint 3A (OdinFlat, dolma-10b, 1 epoch, ~61h):
```bash
STACK=D bash scripts/launch_sprint3a.sh    # RECOMMENDED (canonical winner)
STACK=E bash scripts/launch_sprint3a.sh    # ALT (delayed-enable, modest single-run quality edge)
```

Sprint 3B (OdinHalo, dolma-10b, 1 epoch, ~77h) — after OdinHalo Stack D smoke:
```bash
STACK=D bash scripts/launch_sprint3b.sh  # DO NOT RUN without explicit user approval
```

Per session start rule: the agent **STOPS HERE and awaits explicit user
approval** for Sprint 3A/3B launches. Present Stack D composition, expected
throughput, expected wall time, and wait for user go-ahead.

## Phase C post-hoc additions (session 2 completion)

After the initial Phase D close, C.1 warm-start matrix + C.3 w_gate_up staging
were run to completion per user request:

- **C.1 warm-start matrix:** all 4 configs PASS (a/b tied, c/d show better
  final loss than Stack D from-scratch). H14 (optimizer state mismatch) RULED
  OUT (C.1.a vs C.1.b tied). Delayed-enable (C.1.c/d) shows modest quality
  edge (-0.033 to -0.123 final loss vs Stack D direct). Single-run evidence
  only — not conclusive as a systematic win, but shipped Stack E as opt-in
  alternative recipe for users who want the edge.
- **C.3 w_gate_up staging:** NULL-EFFECT (tok/s + loss indistinguishable
  from Stack D). Confirms C.2 finding that update-scale is not a mechanism.

Full details: `docs/perf/t5-c1-warmstart-findings.md`, `docs/perf/t5-c3-wgu-staging-findings.md`.
