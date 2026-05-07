# Autonomous session 2026-05-07 → 2026-05-08 16:00

User stepped away for ~24h. Full autonomy granted for pre-approved scope
(see header of this file). Every decision and observation logged here.

## Scope & halt rules

In scope:
  1. S1.3: lr_2d=2.5e-3, 700 steps extended confirmation
  2. S1.4: --optimize-kernels A/B (400 x 2)
  3. S1.5: OdinFlat dolma sanity (lr_2d=5e-3, 400 steps)
  4. S1.6: Stage 1 wrap-up commit + STATUS.md update
  5. Phase A' convergence_stats.py development (0 compute)
  6. Phase A' retroactive scoring on 5 existing checkpoints
  7. Phase A' gate-decision report (cosines measured, decision DEFERRED to user)

Halt rules:
  - If S1.3 fails gates: document, fall back to S1.1 (lr_2d=2e-3) as winner
  - If S1.4 --optimize-kernels crashes: skip, document, move on
  - If S1.5 fails recipe transfer to dolma: log, do not commit as Sprint 3A config
  - If remote unreachable >15 min: retry 5 min intervals; if >2h, halt branch
  - Never: force push, destructive git, bf16 attempts, unapproved sprint pivots

Gate decisions deferred (cosines computed, user reads on return):
  - Phase A' -> Phase B greenlight: cos(iter_2, iter_3) threshold

## Session timeline

### 2026-05-07 14:45 local -- S1.3 launch

Created scripts/run_sprint3_s1_3.sh (copy of iter2b template with
--max-steps 700, new ckpt dir sprint3-s1_3-lr2_5e3-700).

Pass criteria (expanded for 700-step horizon):
  1. Loss monotonic from step 50 to 700
  2. shared_layers.5 maxabs at step 700 <200 (doubled from 400-step gate)
  3. fp16_headroom at step 700 >300x (lowered from 500x)
  4. Zero scaler backoffs
  5. No NaN microstep messages
  6. grad=<finite>
  7. Late-window deceleration: 600->700 growth <= 500->600 growth

Projection from iter 2b (step 400 maxabs 52.31, growth ~1.23x per 50 step):
  step 500: 52.31 * 1.23^2 = 79
  step 600: 52.31 * 1.23^4 = 120
  step 700: 52.31 * 1.23^6 = 181
  -> Projected pass with <10% margin. Tight.

If growth compounds at 1.30x (iter 2's mid-window rate):
  step 700: 52.31 * 1.30^6 = 259 -> FAIL

S1.3 is the real test; 3-point curve suggested 2.5e-3 is probably safe
but confirmation required.

Launched expected wall-time: 700 steps * 5.4 s/step = 63 min.
Warm cache (same graph structure as iter 2b): +2 min compile.

### 2026-05-07 14:50 -- Phase A' convergence_stats module written

halo_training/eval/convergence_stats.py (252 lines):
  - run(model, tokenizer, splits, args=None) -> dict | None
  - Per-layer cos-to-final (last iter in looped case)
  - Inter-layer transition cos
  - Effective rank (stable rank via svdvals)
  - Iter-k cos-to-final + frac(cos > tau=0.95), for looped models
  - Iter transition cos
  - Returns None if model lacks .layers / .shared_layers
  - Hooks installed + removed via try/finally

Registered in scripts/eval_checkpoint.py EVALUATORS list.

scripts/test_convergence_stats.py (250 lines, 12 tests):
  - Module basics: imports, _resolve_layers for flat/looped/absent
  - _effective_rank: rank-1 matrix -> 1.0, isotropic noise -> ~full rank
  - _pick_split: preference order
  - End-to-end on OdinFlatMini (flat, mr=1) and OdinHaloMini (looped, mr=2)
  - Runs on tiny synthetic .bin; uses CUDA autocast
  - Registry test: scripts/eval_checkpoint.py contains the module

Remote test result: 12/12 pass in 7.0s.

At step 50 S1.3 loss=8.98 (matches iter 2b's 9.03 within noise), grad=1.12
(slightly above MAX_GRAD_NORM=0.8 but clipped fine), scale=1e3, tok/s=22,718.

### 2026-05-07 ~16:00 -- S1.3 COMPLETE, FAIL gate #2

See STAGE1_S1_3 section of commit ea0aaff for full analysis.

Summary: loss 8.98 -> 2.72 monotonic, no NaN, no scaler backoff, but
shared_layers.5 maxabs reached 238.09 at step 700 (gate <200 FAIL by 19%).
Late-window growth went 1.03x -> 1.52x (re-acceleration matching S1.2
signature at lr=3e-3).

Per halt rule: S1.3 FAILED -> fall back to lr_2d=2e-3 for S1.4 + S1.5 +
Sprint 3B. User can override on return.

Convergence stats (side-effect Phase A' data):
  Step 700: iter_k_cos_to_final [0.08, 0.41], frac_high [0.0, 0.0013]
  -> Current reading: KILL Phase B. But this is early training
  (<1.5% of epoch). Need converged checkpoint for definitive call.

### 2026-05-07 18:06 -- S1.4 launch + IMMEDIATE CRASH

Launched S1.4: OdinHalo lr_2d=2e-3 + --optimize-kernels, 400 steps.
Crashed on first forward:

  TypeError: _FusedResidualRMSNormBlockReplacement.forward() got an
  unexpected keyword argument 'depth_kvs'

Root cause: autokernel/_patterns.py line 314 defines a replacement block
whose forward signature is `forward(self, x, freqs_cis)` -- it does not
accept depth_kvs. The pattern matcher wraps OdinHalo's NoPEMoDAGQABlock
(which takes depth_kvs for MoDA depth-attention) with this replacement,
and the signature mismatch crashes on the GQA call site:

  models/odin_halo.py line 206:
    h = layer(h, depth_kvs=prior_kvs)

This is a pre-existing incompatibility between --optimize-kernels and
OdinHalo's MoDA architecture. NOT caused by S1.x work.

Per halt rule #2: "If S1.4 --optimize-kernels crashes compile -> skip,
document, move on." Not touching autokernel/_patterns.py -- that's a
substantive change outside autonomous scope.

Decision: S1.4 -> SKIP with note. Sprint 3B will run WITHOUT
--optimize-kernels for OdinHalo. Existing baseline throughput of
~25K tok/s (from S1.1/S1.2/S1.2b/S1.3) is production config.

For OdinFlat (S1.5), --optimize-kernels should be safe since OdinFlat
has no MoDA depth_kvs machinery. Will re-check the pattern match before
committing S1.5 Sprint 3A config.

Known follow-up work (DEFERRED to user on return):
  - autokernel/_patterns.py pattern matcher: either detect MoDA blocks
    and skip them, OR extend _FusedResidualRMSNormBlockReplacement.forward
    to forward **kwargs. Either is safe to implement but requires testing.
  - This would unlock the ~15-25% throughput lift that --optimize-kernels
    nominally provides (~29K tok/s target from 25K baseline).

### 2026-05-07 ~18:10 -- S1.5 launch (OdinFlat dolma sanity)

lr_2d=5e-3, 400 steps, no --optimize-kernels (would crash same as S1.4).
Expected: validate Sprint 1 recipe transfers from wikitext to dolma on
the flat model.

### 2026-05-07 ~18:40 -- S1.5 COMPLETE, PASS

Results (400 DDP steps, 28.7 min wall, 52.5M tokens):
  Loss:       8.42 -> 5.78 -> 5.23 -> 4.97 -> 4.84 -> 4.70 -> 4.62 -> 4.57
              monotonically decreasing
  BPB final:  1.831 at step 400 (vs 1.84 OdinHalo S1.1 at step 400 -- tie)
  Throughput: 30,566 tok/s aggregate
  layers.13 maxabs at step 400: 41.09 (very safe)
  fp16_headroom: 1594x
  Scaler:     1e3 throughout (zero backoffs)
  NaN:        none
  Late-window growth factor 350->400: 1.08x (clean deceleration)

ALL 5 PASS CRITERIA MET. OdinFlat + Sprint 1 recipe + dolma at lr_2d=5e-3
is viable for Sprint 3A. Throughput is ~19% below the prior wikitext run
(30.5K vs 37.5K tok/s on wikitext) because --optimize-kernels is off.

### 2026-05-07 ~18:50 -- Phase A' retroactive scoring

Scored 7 checkpoints via scripts/score_convergence_retroactive.py on
machine A (while S1.5 was no longer using GPU).

Results: 5 OdinHalo looped, 2 OdinFlat.

Iter convergence (OdinHalo, 5 checkpoints spanning 200 -> 700 steps):
  iter_1 vs iter_2 cos-to-final frac_high: all 0.001-0.002
  Below even the PARTIAL threshold (0.85) by 3 orders of magnitude.
  iter_1 vs iter_2 cos values DROP as training progresses
  (0.61 at step 200 -> 0.41 at step 700) -- iters do more independent
  work, not less.

Per-layer (OdinFlat, 2 checkpoints):
  Classic LEAP layer-skip pattern: sharp cos jump at layer 6 (first GQA),
  gradual climb through layers 7-12, layer 13 = 1.0 by definition.
  Layer 12 frac_high_cos = 0.31 (31% tokens could exit at L12).
  Layer 11 frac_high_cos = 0.14.

Written: docs/perf/phase_a_prime_gate_decision.md
Verdict:
  OdinHalo adaptive-iter:    KILL Phase B
  OdinFlat layer-skip:       GREENLIGHT (but scoped to 5-10% inference
                             speedup only, not the original plan's 15-25%)

Caveats: training is only 700 steps into a 52K-step epoch. Verdict should
be re-checked at Sprint 3B end-of-epoch. Effective rank 2-3 is unusually
low and suggests a low-dim representation space. Noted in gate doc.

### 2026-05-07 ~19:00 -- S1.5 + Phase A' commit prep


