# Phase 2: OdinHalo Fusion Investigation Framework

**Status:** Design
**Date:** 2026-05-05
**Depends on:** Phase 1 profile report (`docs/perf/odinhalo-profile-<date>/`)
**Target model:** `models/odin_halo.py` (OdinHalo)
**Expected gain:** +3–8% on top of post-Phase-1 baseline (data-driven; could be 0% if Inductor already fuses everything)

## Goal

Systematically identify, scope, implement, and measure the fusion bands around the 68%-matmul time that are **not** already captured by Inductor under `torch.compile`. Land only fusions that pay their cost.

## Rationale

Inductor's fusion is strong under `compile_zones` (per Phase 1's 0 graph breaks in the compile-friendly path). Writing custom HIP kernels that duplicate what Inductor already does is code burden without throughput. This phase decides which fusions are still worth custom effort after Inductor has done its work.

## Hardware-imposed constraints

- **No MFMA on gfx1151** → in-kernel tile matmul runs ~5 TFLOPS vs rocBLAS ~15–20 TFLOPS. Every fusion must keep large matmuls in rocBLAS. Custom HIP kernels may do matmul *prologue* (inputs to a matmul) or *epilogue* (outputs of a matmul) but not the matmul itself.
- **64 KB LDS / CU, 6 MB L2** → intermediate activations at batch=16, T=256, d=768 are ~6 MB per tensor. Cannot keep all of a block's activations on-chip. Fusion boundaries follow tensor reuse locality.

## Scope

### In-scope

1. Classify every non-matmul op in the Phase 1 profile that consumes ≥ 2% of wall time.
2. For each candidate, determine whether Inductor already fuses it under compile.
3. For remaining candidates, implement fusion as: wire existing HIP kernel (if available) or write a new HIP kernel.
4. Measure each fusion in isolation; land only those clearing the +0.5% end-to-end threshold.
5. Record rejected fusions in a "deferred" log to avoid rework later.

### Out-of-scope

- Matmul replacements (covered under hardware constraint).
- Architectural changes.
- Anything consuming < 2% of wall time (too noisy to measure).

## Architecture

### Investigation workflow

```
Phase 1 profile ──┐
                  ▼
          enumerate ops ≥ 2%
                  │
                  ▼
     ┌── does Inductor fuse it? ──┐
     │                            │
    yes                           no
     │                            │
     ▼                            ▼
  skip + log                 gain-feasibility check
                                  │
                         ┌────────┴────────┐
                         │                 │
                  ≥ 1% expected      < 1% expected
                         │                 │
                         ▼                 ▼
                   implement+measure   defer + log
                         │
                  ┌──────┴──────┐
                  │             │
              ≥ 0.5% gained   < 0.5%
                  │             │
                  ▼             ▼
               ship         revert + defer + log
```

### Components

**Op classifier** (`scripts/classify_ops.py`, new)
- Input: `docs/perf/odinhalo-profile-<date>/profiler.md` (raw torch.profiler table).
- Output: `docs/perf/phase2-candidates-<date>.md` — table of ops sorted by descending % wall time, with columns: op name, self-CUDA %, self-CPU %, call count, inferred category, and an empty "Inductor fused?" column for manual verification.
- Runs as a pure Python script over the Markdown table; no GPU required.

**Inductor-fusion checker** (`scripts/check_inductor_fusion.py`, new)
- Captures `TORCH_LOGS=output_code` output when running `model.compile_zones()` forward on OdinHalo.
- For each candidate op from the classifier, searches the generated triton source for kernels whose body contains the op pattern (e.g., `rsqrt` + `mul` for RMSNorm, `sigmoid` + `mul` for SiLU gate).
- Annotates the candidates table with verdicts: `Inductor-fused` / `Standalone` / `Unclear`.

**Gain-feasibility estimator**
- For each `Standalone` candidate, compute an upper-bound expected gain: `candidate_pct * 0.8` (assume HIP kernel removes 80% of the op's memory traffic).
- Example: if `aten::mul` is 5% of wall time and isn't Inductor-fused, upper-bound gain from fusing it is ~4%.
- Candidates with upper-bound gain < 1% are marked `deferred` without implementation.

**Kernel selection**
- For each `implement` candidate, choose between:
  - **Wire existing** — an existing `kernels/hip/*.py` kernel matches the pattern. Cost: custom `torch.autograd.Function` wrapper + parity test. Example: `fused_residual_add_rmsnorm` already handles the residual+norm pattern.
  - **Write new** — no existing kernel. Requires HIP source, forward + backward, autograd wrapper, tests.
- Existing kernel inventory (verified at spec time):
  - `fused_mlp` — fuses gate+up+silu+mul (forward-only, needs custom backward). Likely candidate for SwiGLU prologue.
  - `fused_residual_add_rmsnorm` — fuses x+residual+rmsnorm. Wire into block residual points.
  - `fused_residual_add_layernorm` — unused for OdinHalo (uses RMSNorm not LayerNorm).
  - `fused_rope_gate_mul` — already wired.
  - `fused_bias_silu`, `fused_bias_gelu` — standalone activations, low-value for OdinHalo.

**Per-fusion deliverable template**

Each selected fusion produces:
1. HIP source at `kernels/hip/<name>.py` (if new; existing kernels are re-used).
2. `torch.autograd.Function` wrapper handling both forward and backward.
3. Parity test at `scripts/test_<name>_parity.py` verifying:
   - Loss within 1e-3 relative of PyTorch eager reference.
   - Gradient L2 norm within 5e-3 relative.
   - Tested at OdinHalo production shapes (batch=16, T=256, d=768, d_conv=512).
4. Isolation ablation run recording tok/s delta vs pre-fusion baseline.
5. STATUS.md update with the measured delta.
6. If the isolation ablation shows < +0.5% end-to-end tok/s, the fusion is reverted and moved to the deferred log with a reason.

## Data flow

```
Phase 1 profile (profiler.md)
        │
        ▼
classify_ops.py ──► candidates-<date>.md (ops ≥ 2%)
        │
        ▼
check_inductor_fusion.py ──► candidates annotated
        │
        ▼
manual gain estimation ──► implement-list vs defer-list
        │                              │
        ▼                              ▼
implement + measure per fusion    fusion-deferred.md
        │
        ▼
fusion-results-<date>.md (per-fusion tok/s delta)
        │
        ▼
STATUS.md addendum
```

## Error handling

- **Inductor codegen dump empty**: `TORCH_LOGS=output_code` sometimes produces no output when compile is already cached. Force a cache-cold run by setting `TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-phase2` before capture.
- **Parity test fails for a wired kernel**: revert the wrapper, log the failure mode in `fusion-deferred.md`, move to the next candidate. A parity failure is usually an autograd pitfall (non-contiguous inputs, fp32/fp16 mismatch).
- **Ablation shows regression** (negative tok/s): revert. Do not investigate "why" in this phase — log and defer.

## Testing strategy

### Per-fusion gates (sequential, per candidate)

1. **Unit parity**: output matches PyTorch eager within tolerance on random inputs.
2. **Shape parity**: output matches on actual OdinHalo production shapes at batch=16.
3. **Gradient parity**: backward matches within 5e-3 relative on random inputs.
4. **Gradient shape parity**: backward matches at production shapes.
5. **End-to-end parity**: 5-step training loop with fusion vs without produces identical loss trajectory (tolerance 1e-3 relative per step).
6. **Ablation gate**: ≥ +0.5% tok/s at batch=16.

A fusion ships only after all 6 pass.

### Global regression gate (once all selected fusions ship)

- Full 400-step (200 warmup, 200 measured) ablation on OdinHalo at batch=16 with all Phase 2 fusions enabled vs post-Phase-1 baseline. Must show strictly positive tok/s delta. If zero/negative, the combined set has an interaction and requires debugging before landing.

## Success criteria

Phase 2 is complete when **all** of:

1. `docs/perf/phase2-candidates-<date>.md` exists and enumerates every op ≥ 2% of post-Phase-1 wall time.
2. Each candidate has a verdict (Inductor-fused / implemented / deferred).
3. Every `implemented` candidate has shipped with parity + ablation passing, OR is moved to `fusion-deferred.md` with a reason.
4. `docs/perf/fusion-deferred.md` contains entries for every rejected candidate with: reason, expected gain, cost-to-implement.
5. STATUS.md contains a post-Phase-2 ablation row.

**Phase 2 may land with zero fusions shipped** if all candidates are Inductor-fused or below threshold. That is a valid outcome — we stop investing.

### Stop conditions

- 5 working days have elapsed since start.
- All candidates from the initial classification have been evaluated.
- Whichever comes first.

## Deliverables

**Scripts (new):**
- `scripts/classify_ops.py`
- `scripts/check_inductor_fusion.py`

**Per-fusion (variable count):**
- `kernels/hip/<name>.py` (if new)
- `scripts/test_<name>_parity.py`
- Integration changes to `models/components/*.py` or `models/odin_halo.py`

**Docs:**
- `docs/perf/phase2-candidates-<date>.md`
- `docs/perf/fusion-deferred.md`
- `docs/perf/phase2-results-<date>.md`
- STATUS.md addendum

## Open questions

Deferred to the investigation itself:

- Which specific ops will exceed 2%? (Depends on Phase 1 profile.)
- Will Inductor fuse RMSNorm+linear prologue, SiLU+mul, residual+norm? (Unknown until `check_inductor_fusion.py` runs.)
- Will custom HIP kernels beat Inductor-generated triton on fp16 element-wise ops? (Measured per-fusion.)

## Handoff to Phase 3

Phase 3 (CUDA graphs through Parcae) is independent of Phase 2's kernel work. Phase 2's post-ablation profile serves as Phase 3's baseline if Phase 3 runs after Phase 2.
