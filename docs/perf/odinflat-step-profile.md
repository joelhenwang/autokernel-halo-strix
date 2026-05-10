# OdinFlat Step Profile (Track 1.3, 2026-05-10)

**Probe:** 50 opt steps, profile capture steps 30:40 (10 steps).
**Config:** Sprint 1.5 C3 recipe (μP + SPECTRA + NorMuon, no --optimize-kernels),
dolma-10b-odin32k.bin, batch=16, block=512, accum=8, TB4 DDP across A+B.
**Script:** `scripts/profile_odinflat_step.sh`.
**Artifacts:**
- `docs/perf/odinflat-profile-2026-05-10/profile-summary.txt` (26 KB flat op table)
- Full Chrome trace at `checkpoints/odinflat-profile/profile.json` on Machine A
  (1.5 GB, not committed)
- `docs/perf/odinflat-profile-2026-05-10/rank0.log`

## Headline numbers

Profile span: 10 opt steps = 80 microsteps (10 × accum=8), CUDA time total 40.244 s.

| Section | CUDA time | % of step (40.244s = 100%) |
|---|---:|---:|
| aten::mm (all matmuls, fwd+bwd+NorMuon) | 15.234 s | **37.85 %** |
| NorMuon optimizer step | 5.05 s | **12.5 %** |
| aten::logsumexp (z-loss forward, active) | 4.456 s | **11.1 %** |
| LogsumexpBackward | 2.256 s | 5.6 % |
| CausalConv1dFnBackward (HyPE conv bwd) | 1.831 s | 4.54 % |
| CausalConv1dFn (HyPE conv fwd, compiled) | ~0.4 s | 1.0 % |
| Compiled forward graph (CompiledFunction CUDA) | 6.523 s | 16.2 % |
| Compiled backward graph (CompiledFunctionBackward CUDA) | 14.213 s | 35.3 % |
| DDP allreduce (est. residual) | — | < 5 % |

Totals above overlap (matmuls appear under both `aten::mm` and inside
CompiledFunction regions). The self-CUDA accounting does not cleanly
separate forward/backward/optimizer because Inductor has fused many
ops into compiled regions.

## Attention forward share (derived from FLOP ratio)

Per transformer layer (d=768, n_heads=12, n_kv_heads=4, block=512, batch=16):

| Projection | FLOPs (GFLOP, forward only) |
|---|---:|
| Q (wq, 768→768) | 4.83 |
| K (wk, 768→256) | 1.61 |
| V (wv, 768→256) | 1.61 |
| wo (768→768) | 4.83 |
| Attention total | **12.88** |
| FFN w_gate_up (768 → 2×2048) | 25.77 |
| FFN w_down (2048 → 768) | 12.88 |
| FFN total | **38.65** |
| Layer total | **51.53** |

- **Attention forward = 25 % of per-layer forward FLOP**
- Layer forward wall ≈ 30 % of step (rough, from CompiledFunction / total)
- **→ Attention forward ≈ 25 % × 30 % = 7.5 % of total step wall**

Best-case QKV fusion speedup on that 7.5 %: 20 % (rocBLAS prefers one
larger matmul over three split GEMMs at the GQA 4/4/4 ratio). That is
**~1.5 % total wall time saved**.

## Track 1 GATE decision

Plan §3 gate matrix:

| Observed | Decision |
|---|---|
| Attention forward > 25 % of step | Proceed with QKV fusion as planned |
| Optimizer > 40 % of step | Pivot Track 2 to NorMuon NS caching |
| **Forward ≈ 40 %, Backward ≈ 35 %, Optimizer ≈ 20 %, other ≈ 5 %** | **Proceed with QKV fusion; plan NorMuon as future work** |
| Something surprising dominates (e.g. DDP, logsumexp) | Stop, report, re-plan |

Observed reality is **the fourth row**: z-loss `logsumexp` + its backward
together account for **16.7 % of step wall time**, which was not in any
row of the original gate matrix. This is larger than the QKV-fusion
upside (1.5 %) by an order of magnitude.

### Recommendation to user

Two materially different Track 2 scopes are now on the table:

1. **Original plan (QKV fusion):** 2-4 h dev, 1.5 % wall gain → **BELOW
   the 3 % ship gate**. Code and tests are already drafted locally (6/6
   tests pass) so "do nothing" is cheap. Ship or shelve per gate.

2. **New opportunistic target (z-loss fusion):** z-loss currently computes
   a separate `logsumexp(logits)^2` on top of the already-computed CE
   loss. The existing `kernel.py` `ce_full(mode=...)` path already bakes
   z-loss into the fused online-softmax CE kernel (per AGENTS.md §"CE +
   chunked-CE stack"). Routing OdinFlat through `--chunked-ce` (or
   wiring the fused z-loss into the non-chunked path) would **eliminate
   the 4.456 s `aten::logsumexp` forward** and its 2.256 s backward. That
   is **~16 % of step wall**, potentially reclaimable. Not in the
   original plan.

3. **No Track 2, go straight to Track 3 + Sprint 3A launch at baseline.**
   Simplest path. Accepts 31.3K tok/s.

## Secondary observations

- **NorMuon at 12.5 %** is non-trivial but in line with Sprint 1.1 (which
  measured 3.5 % cost vs AdamW on wikitext via `--ns-dtype fp16`; the
  absolute NS cost remains). Caching / fusing NS across microsteps is a
  future track.
- **HyPE causal_conv1d backward at 4.5 %**: the DaoAILab C++ extension
  backward kernel. `_compile_friendly` path would replace it with
  Inductor-native, previously measured as zero gain.
- **hipMemcpyWithStream dominates CPU wall at 79 %** (32.547 s CPU vs
  40.244 s CUDA). CPU is ~idle most of the time, waiting on sync points
  (`aten::item` fires 870 times across 10 opt steps, ~11 per forward).
  Expected — Strix Halo's unified memory means every `.item()` call
  forces a stream sync, but CPU wall is not the binding constraint.

## What's NOT in the profile

The probe DOES NOT reveal:
- Where `aten::mm` time is spent (QKV vs FFN vs NorMuon NS) — the flat
  table doesn't tag by call-site. The Chrome trace could answer this
  via `with_stack=True` attribution; not analyzed in this pass.
- Inductor's fused-triton-kernel vs rocBLAS-GEMM split — some matmuls
  are inside `Torch-Compiled Region` entries, not counted under
  `aten::mm`.

A follow-up profile pass with shape-tagged analysis (à la Phase 2 WI6's
`scripts/profile_shape_calls.py`) would resolve the ambiguity.

## Conclusion

Report findings to the user and confirm Track 2 scope before executing.
