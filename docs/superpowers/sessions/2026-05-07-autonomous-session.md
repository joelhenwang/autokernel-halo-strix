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

