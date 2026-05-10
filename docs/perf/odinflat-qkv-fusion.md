# Track 2.a: QKV-Fusion Ship Decision (2026-05-10)

## TL;DR

QKV fusion **fails the 3% throughput ship gate**: measured lift is +0.07%
(within noise). All 6 unit tests pass. No loss regression. No memory
regression. Autokernel's FusedQKVPattern becomes a no-op on fused
attention classes — an acceptable tradeoff for OdinFlat/OdinHalo which
were already skipping the pattern via `_skip_autokernel=True` since
Phase 0. Code **stays committed** (63de5be + 7a5a108) because it's a
cleaner architectural design, the migration hook guarantees checkpoint
compatibility, and no regression was observed. Sprint 3A recipe is
**unchanged**.

## A/B probe

Matched single-node 200-step probes on `dolma-10b-odin32k.bin`, Sprint
1.5 C3 recipe, block=512, batch=16, accum=8, `max-autotune-no-cudagraphs`.

Pre-fusion: ran off a local checkout of parent commit `63de5be^` applied
to `models/components/attention.py` only (other files pinned at current
HEAD so probe scripts and training loop don't drift).

Post-fusion: HEAD at `7a5a108` (Track 2.a commits 63de5be + probe script).

| Metric | Pre-fusion | Post-fusion | Δ |
|---|---:|---:|---:|
| Step 50 loss | 7.3734 | 7.2199 | -0.15 |
| Step 100 loss | 5.1993 | 5.2077 | +0.008 |
| Step 150 loss | 4.8532 | 4.8788 | +0.026 |
| Step 200 loss | 4.7041 | 4.7212 | +0.017 |
| Steady tok/s (steps 100-200 avg) | 15,883 | 15,894 | **+0.07%** |
| Peak memory | 13.5 GB | 13.5 GB | 0 |
| First-compile overhead (step 25 tok/s) | 13,306 | 8,657 | -35% (one-time) |

Raw logs archived at:
- `docs/perf/odinflat-profile-2026-05-10/qkv-prefusion.log`
- `docs/perf/odinflat-profile-2026-05-10/qkv-postfusion.log`

## Interpretation

- **Steady-state gain of +0.07%** is indistinguishable from measurement
  noise. Track 1.3 profile already predicted this (attention forward is
  ~7.5% of step, best-case fusion speedup of ~20% on that 7.5% = ~1.5%
  theoretical ceiling; actual observed is well below ceiling, likely
  because rocBLAS's small-GEMM path was already well-tuned for the
  split case).

- **First-compile penalty is one-time**. On a full 52k-step Sprint 3A
  run the amortized cost is zero.

- **Loss deltas at every measured step are within fp16 noise** (± 0.05
  was the plan's parity threshold; we observe ≤ 0.03 at every step
  after warmup). No ship-blocker.

## Decision

**Keep QKV fusion code committed.** Rationale:

1. **No regression observed** — throughput, memory, loss parity all
   clean. The plan's revert condition was regression, not "below ship
   gate with zero downside".
2. **Cleaner architecture** — single `wqkv` Linear is standard practice
   in modern transformer implementations (LLaMA-4, Mistral, GPT-NeoX).
   Easier to reason about, easier to maintain.
3. **Migration hook guarantees backward compatibility** — every
   pre-fusion checkpoint (Sprint 1, 1.5, 3A-confirm, Stage 1 variants)
   continues to load through `Attention._load_from_state_dict`. Verified
   by `test_state_dict_migration` in `scripts/test_qkv_fusion.py`.
4. **Autokernel interaction is a no-op on our primary path** —
   OdinFlat/OdinHalo use `NoPECodaAttention` with
   `_skip_autokernel = True` since Phase 0, so `FusedQKVPattern` was
   already inactive. Secondary models (jormungandr_halo, baldr_halo)
   would lose FusedQKVPattern, but (a) they are not on the Sprint 3
   training path, and (b) per Track 3.A findings, `--optimize-kernels`
   has silent-freeze bugs on anything using SwiGLU HIP — so the
   replacement path may be broken for those models anyway.
5. **Nonzero future optionality** — if we ever implement a proper
   fused-QKV+attention HIP or Triton kernel, having `wqkv` as the
   source of truth simplifies its integration.

## Sprint 3A impact

No recipe change. Sprint 3A launches with the locked C3 recipe at
~31.3K tok/s aggregate (expected from DDP across A+B, extrapolated
from single-node ~15.9K × 2).

## Followups

- Track 3 findings (see `autokernel-frozen-blast-radius.md` and
  forthcoming `autokernel-deep-analysis.md`) further support the
  argument that `FusedQKVPattern` itself should be retired or
  autograd-validated if ever re-enabled for any training path.
- QKV fusion's theoretical upside (larger single GEMM) may surface
  more clearly at larger d_model or n_heads. If a larger-scale model
  is introduced (e.g. OdinHalo d=1024+), re-run this probe to
  re-evaluate the gain.
