# T-0.4 Inductor graph-break inventory (2026-05-10)

**Scope:** catalog every graph break and recompile in the current OdinFlat compile path. Inform T-2.1 compiled-autograd feasibility and T-1.3 break-fix scope.

**Method:** single-node smoke with `TORCH_LOGS='graph_breaks,recompiles'`:
```
torchrun --nproc_per_node=1 scripts/train_ddp.py \
  --model models/odin_flat.py --class-name OdinFlat \
  --dataset datasets/dolma-10b-odin32k.bin \
  --block-size 512 --batch-size 4 --accum-steps 2 \
  --compile --no-muon --lr 8e-4 \
  --warmup-steps 5 --max-steps 10 --num-workers 2 \
  [full Sprint 1.5 C3 recipe flags]
```
Executed on Machine A, 2026-05-10 17:07 local. Full log: `checkpoints/graph-break-probe/full.log` (401 lines, 75 s runtime).

---

## Headline result

**Only 2 distinct graph break locations. Only 1 recompile across 10 opt steps.**

This is dramatically less than the "4 breaks per HyPEShortConvBlock × 14 blocks = 56 breaks" estimate in STATUS.md / AGENTS.md. The actual picture is much cleaner than the prior documentation suggested.

---

## Graph break inventory

| # | Location | File:Line | Count | Cause |
|---|---|---|---:|---|
| 1 | `models/components/conv_blocks.py:284` | fused_rope_mul call | 1 | `@torch.compiler.disable`'d HIP `kernel_fn` from `kernels/hip/fused_rope_gate_mul.py` |
| 2 | `torch/_library/custom_ops.py:698` (from `conv_blocks.py:296`) | custom_op trace barrier | 4 | Likely DaoAILab `causal_conv1d_fn` registered as older-style custom op; Dynamo can't trace through |

### Break #1 detail

From the log:
```
Graph Break Reason: Encountered graph break when attempting to trace CALL
Skip calling `torch.compiler.disable()`d function
  Explanation: Skip calling function `<function kernel_fn at 0x...>` since
  it was wrapped with `torch.compiler.disable` (reason: None)
  Hint: Remove the `torch.compiler.disable` call
```

**Source:** `kernels/hip/fused_rope_gate_mul.py` decorates the Python wrapper with `@torch.compiler.disable` to prevent Dynamo from tracing into the HIP pybind call (which would sever gradient flow pre-Phase-B).

**Fix options:**

A. **Convert to `torch.library.custom_op`** with `register_autograd` and `register_fake` (same pattern as other Phase B fixes). Removes `@torch.compiler.disable` because the op becomes opcheck-compatible. Estimated cost: 2-4 hours.

B. **Use `_compile_friendly=True` path** (already exists, `conv_blocks.py:289`). Replaces HIP kernel with native PyTorch equivalent. Per AGENTS.md: "0 graph breaks vs 4 default. But NOT faster."

C. **Accept the break.** The break fires once per forward per layer. Dynamo correctly continues compiling on either side of the break. The cost is compile overhead + inability to fuse across the boundary. Estimated cost: marginal at runtime (kernel work dominates).

**Recommendation:** Fix A for consistency with Phase B principle (every training-path HIP kernel uses `register_autograd`). Also enables compiled-autograd to see through the boundary, which matters for T-2.1.

### Break #2 detail

4 occurrences at `torch/_library/custom_ops.py:698` triggered from `conv_blocks.py:296`. The physical call site is either line 293 (`causal_conv1d_fn` from DaoAILab) or line 297 (fallback `self.conv(...)`).

Most likely: **DaoAILab's `causal_conv1d_fn` uses the older `torch._custom_op.impl.custom_op` decorator** (pre-`torch.library.custom_op`), which has different Dynamo-tracing semantics and always breaks.

**Fix options:**

A. **Wrap `causal_conv1d_fn` in a new `torch.library.custom_op`** with explicit `register_autograd`. Gives us control over tracing. Estimated cost: 4-6 hours + parity tests.

B. **Replace with Inductor-generated conv1d.** Would require implementing a `torch.nn.functional.conv1d`-based equivalent. Likely slower (measured `_compile_friendly`: "NOT faster"). Rejected.

C. **Accept the break.** Same reasoning as #1. Since DaoAILab's kernel IS fast (profiled at 4.54% of step for backward; forward is even cheaper), the break overhead is small.

**Recommendation:** Fix A, same reasoning as #1. Enables compiled-autograd through the conv boundary.

---

## Recompile inventory

**Total recompiles across 10 opt steps: 1.**

| # | Function | File:Line | Cache path | Trigger guard |
|---|---|---|---|---|
| 1 | `forward` | `models/odin_flat.py:64` | `[8/1]` | `v_prev is None` (transition from None → tensor on step 2) |

This is the **value residual** path (`models/components/attention.py:326 in forward`). On step 1, `v_prev = None`. On step 2+, `v_prev` is the tensor from the previous layer's value projection. Dynamo specialized the step-1 graph on `v_prev is None` and recompiles once when the guard fails.

**Impact:** one-time cost at step 2 (compile overhead). No steady-state cost after warmup.

**Fix options:**

A. **Initialize `v_prev` as a zero tensor instead of None.** Eliminates the branch, no recompile. But changes initial arithmetic slightly (layer 0 adds zeros instead of skipping). Would need numerical-equivalence validation.

B. **Mark `v_prev` as a dynamic-shape input.** Dynamo supports optional-tensor inputs but requires explicit decoration.

C. **Accept the one-time recompile.** Cost is amortized to zero across a 2000-step run.

**Recommendation:** C (accept). Fixing this saves ~2-3 seconds of compile time per run, not worth the engineering effort or numerical-equivalence risk.

---

## Autotune activity

The log shows ~25 Inductor autotune events during compile:
```
SingleProcess AUTOTUNE benchmarking takes 0.9098 seconds and 0.0002 seconds precompiling for 29 choices
SingleProcess AUTOTUNE benchmarking takes 1.0613 seconds and 0.0002 seconds precompiling for 31 choices
[... ~25 total]
```

These are `max-autotune-no-cudagraphs` searching for the best Triton config for each matmul shape. Total autotune time: ~20 seconds during first-compile. Cached in subsequent runs.

**Notable autotune winners (top-ranked by wall time for given shape):**
- `triton_mm_713`: 0.180 ms, `BLOCK_K=128, BLOCK_M=64, BLOCK_N=32, num_warps=4`
- `triton_mm_721`: 0.182 ms, `BLOCK_K=64, BLOCK_M=128, BLOCK_N=32, num_warps=4`
- `triton_mm_722`: 0.184 ms, `BLOCK_K=32, BLOCK_M=128, BLOCK_N=64, num_warps=4, waves_per_eu=2`

These match patterns known to work on RDNA 3.5 (wave32, no MFMA). No anomalies observed.

---

## Implications for T-2.1 compiled autograd

### Positive signals

1. **Graph-break surface is small.** Compiled autograd has only 2 break-point types to work around. This is a much better starting point than expected.
2. **Recompile rate is near-zero after warmup.** Compiled-autograd's cache invalidation won't thrash.
3. **Autotune infrastructure works.** RDNA 3.5 Triton configs autotune successfully; compiled-autograd will use the same tuning path.

### Caveats

1. **Both break points are inside the backward-relevant region** (rope + conv1d both have backward passes). Compiled autograd may choke on `@torch.compiler.disable` and the DaoAILab custom op.
2. **Fix A for both breaks (register as `torch.library.custom_op`) is a prerequisite** for compiled-autograd to have clean traces through the conv blocks.

### Revised T-2.1 plan

**Phase 1 (prerequisite, ~1 day):**
- Convert `fused_rope_gate_mul.kernel_fn` to `torch.library.custom_op` + `register_autograd`
- Wrap DaoAILab `causal_conv1d_fn` in a `torch.library.custom_op` shim with explicit autograd

**Phase 2 (compiled autograd, ~4-5 days):**
- Enable `torch._dynamo.config.compiled_autograd = True`
- Run 200-step smoke
- Debug any custom_op + compiled-autograd incompatibilities
- 2000-step stability gate

Total T-2.1: still within the 5-6 day budget but with more concrete breakdown.

---

## Implications for T-1.3 break fixes

Ship gate for T-1.3:
- Fixing break #1 (fused_rope_gate_mul): expected 0-1% tok/s direct gain (break cost was small), but **prerequisite for T-2.1**. Ship.
- Fixing break #2 (causal_conv1d_fn): same as #1. Ship.
- Accepting recompile #1 (v_prev): no fix needed.

Both break fixes are effectively **infrastructure for T-2.1**, not throughput wins in themselves. If T-2.1 is descoped, these fixes deliver near-zero direct benefit and shouldn't be prioritized.

---

## Revised T-1 exit gate

Prior assumption: T-1 delivers ~3-5% tok/s from break fixes + sync audit + fused-zloss validation.

Revised given this data:
- T-1.1 fused-zloss: +5-8% (unchanged)
- T-1.2 sync points: +2-4% (per `sync-point-audit.md`)
- T-1.3 graph breaks: ~+0-1% direct, serves as T-2.1 prerequisite

**Realistic T-1 exit tok/s:** 32.5-34k (vs prior estimate 32.5k).

---

## Artifacts

- Full log: `checkpoints/graph-break-probe/full.log` on Machine A (401 lines, 16.8 KB)
- Checkpoint: `checkpoints/graph-break-probe/step_10.pt` on Machine A
- This document: `docs/perf/graph-breaks-inventory.md`

Not committing the full log (verbose TORCH_LOGS output, 16 KB of compiler chatter). Key excerpts are in this document.
