# Phase 3: CUDA Graphs Through the Parcae Loop

**Status:** Design
**Date:** 2026-05-05
**Target model:** `models/odin_halo.py` (OdinHalo and all HALO-family looped models)
**Depends on:** nothing (can run in parallel with Phase 2)
**Expected gain:** +10–15% on top of whatever stack is in place when it lands; bigger memory win (–30–50% peak) than tok/s win

## Goal

Enable `TORCH_COMPILE_MODE=reduce-overhead` (or an equivalent CUDA-graph benefit) for OdinHalo's Parcae-looped forward pass. Currently the trainer auto-falls-back to `default` mode for looped models because per-layer CUDA graphs reuse buffers across Parcae iterations and corrupt saved-for-backward activations.

## Current state

Per STATUS.md (Compile × Kernel Ablation):

```
compile reduce-overhead + HIP CE   at bs=16: 14,425 tok/s, 2.14 GB peak (NON-LOOPED hypothetical)
compile default + HIP CE           at bs=16: 14,682 tok/s, 4.83 GB peak (ACTUAL, looped)
```

The memory delta (2.14 vs 4.83 GB) shows the potential ceiling: CUDA graphs could cut peak memory by more than half. Throughput is a wash or mild positive.

The issue is architectural: `compile_zones()` compiles each `model.shared_layers[i]` as an individual CUDA-graph-backed function. Parcae calls `shared_layers[i]` three times (iters 0, 1, 2). Each invocation of a given compiled function replays its single CUDA graph, which reuses the graph's input/output buffers. Autograd saves references to those buffers for the backward pass, but by the time backward runs, later iterations have overwritten the buffer contents.

## Scope

### In-scope

1. Evaluate three candidate approaches (X, Y, Z) via time-boxed POCs.
2. Implement the winning approach end-to-end.
3. Wire it into `halo_training/trainer.py` so `TORCH_COMPILE_MODE=reduce-overhead` no longer auto-falls-back for looped models.
4. Document the approach, known limitations, and any opt-out paths in STATUS.md and AGENTS.md.

### Out-of-scope

- Non-looped models (they already work with reduce-overhead; no action needed).
- Compile-mode combinations with `--chunked-ce` if those remain incompatible; document but do not fix in this phase.
- Generalising to all halo models (VidarHalo, FenrirHalo, etc.). This phase lands OdinHalo only; other looped halo models adopt the same pattern in follow-up work.

## Candidate approaches

### Option X — Clone-at-boundaries

Insert explicit `h = h.clone()` calls at every point where a compiled-function output feeds a subsequent compiled-function call or a non-compiled autograd-saving operation.

**Required clones (for OdinHalo):**
- Before each `self.shared_layers[i](h, ...)` call inside the Parcae iteration loop.
- After the final iteration, before `self.final_norm(h)`.

**Memory cost:** +N × activation_size per step, where N = number of clone sites. For OdinHalo at bs=16, T=256, d=768 fp16: ~6 MB per clone × (mean_recurrence × n_shared_layers + 1) = ~6 × (3×6+1) = ~114 MB. Well under the 2.7 GB memory head-room expected after reduce-overhead.

**Throughput cost:** each clone is one memcpy of an on-GPU fp16 tensor. At bs=16, one clone ≈ 30 μs. 19 clones/step ≈ 570 μs/step overhead. At current 68 ms/step, this is ~0.8% throughput cost.

**POC time budget:** 1 day.

### Option Y — Unrolled compile

Wrap the entire Parcae forward body (injection + skip + run_shared_block × mean_recurrence) in a single `torch.compile`'d function, unrolling the iteration loop at compile time. Each iteration's call to `shared_layers[i]` becomes a distinct node in the graph, not a reuse of the same compiled function. No buffer aliasing because no reuse.

**Trade-off:** the compiled graph grows by `mean_recurrence`×. For mean_recurrence=3, graph size triples. Inductor + Triton compile time scales with graph size; expect 2–5× longer first-compile warmup (60s → 120–300s).

**POC time budget:** 2 days.

### Option Z — Manual CUDA graphs

Bypass `torch.compile` for the Parcae region. Use `torch.cuda.CUDAGraph` directly: pre-allocate static input/output tensors, capture the forward, replay on each step. Autograd integration requires manually registering saved tensors with explicit `.clone()` on boundaries (similar to X, but without Inductor's help).

**Trade-off:** most code; most deterministic. Breaks independence from compile. Useful only if X and Y both fail acceptance.

**POC time budget:** 2 days.

## Architecture

### Investigation workflow

```
Day 1: Option X POC
    ├─ insert clones as described
    ├─ run ablation at bs=16: measure tok/s, peak_gb, first-compile warmup
    ├─ run 200-step smoke-quality test: loss trajectory within 1% of baseline
    └─ acceptance check (below)
         │
         ├─ PASS ──► skip to Implementation phase (winner = X)
         └─ FAIL ──► continue to Option Y

Day 2-3: Option Y POC
    ├─ rewrite _forward_unrolled to call a single compile'd parcae_body
    ├─ measure same metrics
    └─ acceptance check
         │
         ├─ PASS ──► skip to Implementation (winner = Y)
         └─ FAIL ──► continue to Option Z

Day 4-5: Option Z POC (if needed)
    ├─ manual CUDAGraph capture+replay
    ├─ measure + acceptance
    └─ PASS ──► Implementation (winner = Z)
       FAIL ──► document blocker, exit without landing

Day 6-7: Implementation of winner
    ├─ production-ready code path
    ├─ trainer auto-upgrade: remove reduce-overhead fallback for OdinHalo
    ├─ ablation + STATUS.md update
    └─ AGENTS.md documentation
```

### Acceptance criteria per POC

A POC advances to implementation if **all** of:

1. **Throughput gain:** ≥ +5% tok/s at batch=16 vs current `compile + HIP CE` baseline (14,682 tok/s → ≥ 15,420 tok/s).
2. **Memory:** peak memory at batch=16 ≤ 3.5 GB (vs current 4.83 GB) AND no regression larger than +200 MB over the "expected" reduce-overhead ceiling (~2.14 GB).
3. **Quality:** 200-step smoke-test loss trajectory within 1% relative of the default-compile baseline at every checkpoint.
4. **Warmup:** first-compile time ≤ 120 s (double the current 52 s; preserves usability for development runs).

If a POC fails exactly one criterion by < 10%, it can be "conditionally accepted" with a documented trade-off and the user decides before landing.

### Components (winner-independent)

**`TORCH_COMPILE_MODE=reduce-overhead` auto-upgrade** (edit `halo_training/trainer.py`)
- Current logic: if `compile_mode == "reduce-overhead" and hasattr(model, "compile_zones")`: fall back to default.
- New logic: for models on an allowlist (initially `{OdinHalo, OdinHaloBase, OdinHaloAblation, OdinHaloMini}`), invoke the winner's entrypoint (e.g., `model.compile_zones_graphed()`) instead of falling back.
- The allowlist lives in a module-level constant so it's grep-able when we generalize.

**Winner-specific implementation** (one of):

- **X:** `models/odin_halo.py::_forward_unrolled` gains clone sites. A new method `compile_zones_graphed()` triggers `compile_zones()` plus env-guard for the reduce-overhead mode.

- **Y:** `models/odin_halo.py` gains a `_parcae_body(h, input_embed, freqs_cis, depth_kv_buffer)` method that runs the entire iteration loop unrolled. `_forward_unrolled` calls a `torch.compile`'d wrapper of `_parcae_body`. `compile_zones_graphed()` sets the compile path.

- **Z:** new file `models/components/cuda_graph_wrap.py` contains a `CUDAGraphedParcae` class with capture+replay+backward integration. `OdinHalo.forward` dispatches to it when `self._graphed_parcae is not None`.

## Data flow

```
input_ids
    │
    ▼
tok_embeddings (eager, not in graph)
    │
    ▼
[Parcae region: captured in CUDA graph(s) via winner approach]
    │
    ▼ (h output, cloned if X; fresh if Y/Z)
final_norm (eager)
    │
    ▼
lm_head (eager) → logits
```

Gradient flows backward through the graph-captured region. Backward saves activations; winner approach ensures those saves see non-aliased memory.

## Error handling

- **PyTorch < 2.5**: reduce-overhead mode was less stable. Require PyTorch ≥ 2.5 at runtime via version check; fall back to default on older versions with a clear warning. Print installed PyTorch version in the warning.
- **OOM during graph capture**: reduce-overhead needs extra memory for static input/output buffers at capture time. If `torch.cuda.OutOfMemoryError` raised during warmup, fall back to default mode and surface the actual OOM stack. Do not silently succeed.
- **Gradient NaN first 10 steps**: can happen if a clone site is missed. Runtime monitor: first 10 training steps check `torch.isfinite(loss)`. If any is non-finite AND reduce-overhead is active AND this is not the user's fault (e.g., already-diverged model), auto-fall-back to default with a clear error message.
- **Quality regression at scale**: if the 200-step smoke test shows > 1% loss divergence from default compile, abort the POC and document the delta in `docs/perf/phase3-poc-X-results.md` (or Y/Z).

## Testing strategy

### Per-POC tests

1. **Gradient parity** (mandatory): 5-step training loop with winner approach vs default compile. Parameter L2 norm at step 5 matches within 0.1% relative.
2. **Loss trajectory parity**: 200-step smoke test; max relative deviation across all steps ≤ 1%.
3. **Peak memory check**: measured at bs=16 over 200 steps. Must be ≤ 3.5 GB.
4. **Throughput check**: steady-state tok/s averaged over last 100 of 200 measured steps. Must be ≥ 15,420.
5. **Warmup check**: time from process start to first measured step. Must be ≤ 120 s with cold Inductor cache.

### Post-implementation regression suite

- Rerun `scripts/ablation_final.py` at batch=16 to confirm landed win.
- Rerun `scripts/test_all_models_chunked_train.py` to confirm other HALO models unaffected.
- Rerun OdinHaloMini smoke test to confirm small-model path still works.

## Success criteria

Phase 3 succeeds if **all**:

1. A winner (X, Y, or Z) clears acceptance.
2. Winner is landed in `models/odin_halo.py` and `halo_training/trainer.py` with opt-in via `TORCH_COMPILE_MODE=reduce-overhead`.
3. At least one of: `--chunked-ce` fallback removed, OR documented limitation that reduce-overhead with chunked-ce remains incompatible and why.
4. STATUS.md has a post-Phase-3 ablation row showing new tok/s and peak mem numbers.
5. AGENTS.md compile-strategy section updated with the recommended configuration.

Phase 3 **may** exit without landing if all three POCs fail acceptance. In that case, produce `docs/perf/phase3-blocked.md` with what was tried, measurements, and root-cause hypotheses. This exits successfully without pushing broken code.

## Deliverables (if winner lands)

**Code (winner-dependent, but all three share these):**
- `models/odin_halo.py` — new `compile_zones_graphed()` method, updated `_forward_unrolled` if applicable.
- `halo_training/trainer.py` — allowlist for reduce-overhead auto-upgrade.

**Winner-specific additions:**
- X only: clone site insertions; no new files.
- Y only: `_parcae_body` method restructure; no new files.
- Z only: `models/components/cuda_graph_wrap.py` new file.

**Scripts (new):**
- `scripts/ablation_phase3.py` — measures each POC under identical conditions and emits a comparison table.

**Docs:**
- `docs/perf/phase3-poc-X-results.md` (always)
- `docs/perf/phase3-poc-Y-results.md` (if X fails)
- `docs/perf/phase3-poc-Z-results.md` (if X and Y fail)
- `docs/perf/phase3-blocked.md` (only if all fail)
- STATUS.md addendum
- AGENTS.md section update

## Open questions

Deferred to POC measurement:

- Exact memory cost of Option X clones at batch=32 (documented if relevant).
- Whether `TORCH_COMPILE_MODE=max-autotune` composes with the winning approach (possibly a follow-up after landing).
- Whether generalizing to VidarHalo/FenrirHalo is a simple copy-paste or requires per-model tuning (deferred to a follow-up spec).

## Relationship to other phases

- **Phase 1**: provides the baseline tok/s and memory numbers that acceptance criteria are measured against. Phase 3 baselines on the post-Phase-1 stack.
- **Phase 2**: independent. Can land before, after, or in parallel with Phase 3. If Phase 2 lands first, Phase 3 baselines on the post-Phase-2 stack.
- **Future**: generalization to other HALO models is a follow-up spec that reuses this infrastructure.
