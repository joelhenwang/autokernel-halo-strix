---
title: "OdinHalo Fullgraph Compile Sprint — plan"
domain: plans
type: plan
status: ready-to-execute
sprint: fullgraph-compile
tags: [torch.compile, fullgraph, custom-op, cuda-graphs, parcae, odinhalo, gfx1151]
related:
  - ../specs/2026-05-07-odinhalo-fullgraph-compile-design.md
  - 2026-05-07-v3-experiment-roadmap.md
---

# OdinHalo Fullgraph Compile Sprint — plan (2026-05-07)

## Trigger

**Starts after v3.1 distillation Sprint lands** (~6 weeks from 2026-05-07).
Not before — E1 distillation is higher ROI and sequentially ordered.

## Owner

Single engineer, ~3 weeks of focused time. If multiple engineers are
available, only one should own this Sprint; its phases are sequential.

## Phase A — Diagnostic (1 day)

**Goal**: ground-truth Dynamo break report for `OdinHalo.forward`.

### A.1 Pre-flight

```bash
bash sync_remote.sh                    # sync code to Machine A
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
    python -c 'import torch; print(torch.__version__)'"
```

Note the exact PyTorch version. `torch.library.custom_op` requires
2.4+; `register_fake` requires 2.4+. Confirm both present.

### A.2 Execute diagnostic

```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
    python scripts/diagnose_fullgraph.py --model odin_halo --class OdinHalo \
        --output docs/perf/fullgraph-diagnostic-2026-05-07.md"
```

Script captures:

1. Break count under `fullgraph=True`.
2. Each break's file:line + reason_code + user_stack_summary.
3. Categorization (HIP kernel / list mutation / detach / other).
4. Recommended path (S1 / S2 / S3) based on distribution.

### A.3 Gate A — path decision

| Diagnostic says | Commit to |
|---|---|
| ≤ 6 breaks, concentrated in HIP kernels + depth_kv_buffer | **S2** (default) |
| ≤ 3 breaks, all simple | S1 minimal |
| ≥ 10 breaks, scattered | S3 scan-based |

Write the decision in `docs/perf/fullgraph-diagnostic-2026-05-07.md`
before Phase B starts.

## Phase B — Break fixes (~10–12 days)

### B.1 HIP kernel wrapping (~4–5 days)

Target: `fused_rope_gate_mul` (1 call) and `causal_conv1d_fn` (1 call
site; 3 breaks reported by Dynamo at that site).

#### B.1.1 `fused_rope_gate_mul` (~1.5 days)

File: `kernels/hip/fused_rope_gate_mul.py`.

Add a `torch.library.custom_op` wrapper alongside the existing
`kernel_fn`. Keep `kernel_fn` for non-compile callers:

```python
@torch.library.custom_op("halo::fused_rope_gate_mul", mutates_args=())
def fused_rope_gate_mul_op(
    b: Tensor, h_tilde: Tensor,
    freqs_cos: Tensor, freqs_sin: Tensor,
    T: int, d_conv: int, rope_pairs: int,
) -> Tensor:
    return kernel_fn(b, h_tilde, freqs_cos, freqs_sin, T, d_conv, rope_pairs)

@fused_rope_gate_mul_op.register_fake
def _(b, h_tilde, freqs_cos, freqs_sin, T, d_conv, rope_pairs):
    return torch.empty_like(b, dtype=torch.float32)

@fused_rope_gate_mul_op.register_autograd
def _backward(ctx, grad_output):
    # Reconstruct backward from saved tensors. Gradient of RoPE rotation
    # is the conjugate rotation; gradient of gate mul is the usual
    # element-wise chain rule. Detailed implementation lives here.
    ...
```

Tests in `scripts/test_compile_safe_kernels.py`:

- `test_rope_gate_mul_numeric_parity_fp16` — output matches raw
  `kernel_fn` within `1e-3` abs tolerance.
- `test_rope_gate_mul_gradcheck_fp32` — `torch.autograd.gradcheck`
  passes on fp32 proxy at small shape (B=2, T=8, d_conv=16).

#### B.1.2 `causal_conv1d_fn` (~2 days)

File: create a thin adapter `kernels/causal_conv1d/compile_safe.py`
that wraps the DaoAILab `causal_conv1d_fn` in a `custom_op`.

Similar pattern. If `register_autograd` proves unreliable because the
DaoAILab extension has its own backward path through CUDA that
doesn't play well with our adapter, **fall back**:

```python
@torch.compiler.allow_in_graph
def causal_conv1d_compile_safe(x, weight, bias):
    from causal_conv1d import causal_conv1d_fn
    return causal_conv1d_fn(x, weight, bias)
```

`allow_in_graph` preserves the break-free graph without requiring
backward registration; Dynamo treats the call as opaque but traces
around it.

Tests: same pattern as B.1.1.

#### B.1.3 `HyPEShortConvBlock` feature flag (~0.5 days)

File: `models/components/conv_blocks.py`.

Add `use_compile_safe_kernels: bool = False` to
`HyPEShortConvBlock.__init__`. In `forward`, branch on the flag:

```python
if getattr(self, "use_compile_safe_kernels", False):
    y = fused_rope_gate_mul_op(b_flat, h_tilde_flat, freqs_cos, freqs_sin,
                               T, self.d_conv, self.rope_head_dim // 2)
    # ... same for conv via causal_conv1d_compile_safe
elif getattr(self, "_compile_friendly", False):
    # existing compile-friendly path (native PyTorch)
    ...
else:
    # existing @torch.compiler.disable path (current default)
    ...
```

Default `False` preserves every other halo model's behavior.

### B.2 Static depth_kv_buffer (~3 days)

#### B.2.1 Forward changes (~1 day)

File: `models/odin_halo.py`.

Replace the list-of-dicts in `_forward_unrolled` with a preallocated
tensor:

```python
# In _forward_unrolled, after input_embed:
n_moda = len(self.gqa_positions)
depth_kv_buf = h.new_zeros(
    self.mean_recurrence, n_moda, B, 2, self.n_kv_heads, T, self.head_dim,
)
depth_kv_valid = torch.zeros(
    self.mean_recurrence, dtype=torch.bool, device=h.device,
)

# Pass to _run_shared_block, which writes via indexed assignment
h, depth_kv_buf, depth_kv_valid = self._run_shared_block_static(
    h, freqs_cis, depth_kv_buf, depth_kv_valid, iter_idx=0,
)
```

Update `_run_shared_block` to `_run_shared_block_static` that
accepts + returns the tensor triple. Inside, MoDA writes become:

```python
if is_gqa and self.use_moda:
    layer_ordinal = self._gqa_layer_ordinal[idx]   # 0-indexed within gqa_positions
    k, v = layer.compute_depth_kv(h.detach())
    depth_kv_buf[iter_idx, layer_ordinal, :, 0] = k  # K
    depth_kv_buf[iter_idx, layer_ordinal, :, 1] = v  # V
    depth_kv_valid[iter_idx] = True
```

`NoPEMoDAGQABlock.forward` reads prior KVs by slicing the tensor
across `iter` dim up to current iteration.

#### B.2.2 MoDA attention update (~1 day)

File: `models/odin_halo.py::NoPEMoDAGQABlock` and
`models/components/attention.py::NoPECodaAttention` (only the MoDA
depth_kvs handling path).

Change the `depth_kvs` parameter from `List[Tuple[K, V]]` to a
tensor slice. Inside, iterate the valid prior iterations via the
valid-mask tensor rather than a dynamic list length.

#### B.2.3 Parity tests (~1 day)

Add to `scripts/test_compile_safe_kernels.py`:

- `test_depth_kv_buffer_parity_at_step_10` — forward parity with
  old list-based path at step 10.
- `test_depth_kv_buffer_parity_at_step_100` — same at step 100.
- `test_depth_kv_buffer_parity_at_step_500` — same at step 500 with
  optimizer updates applied.

All three must pass before integration.

### B.3 Loop unroll (~1 day)

File: `models/odin_halo.py`.

Manual 3-iteration unroll of `_forward_unrolled` behind the
`use_compile_safe_kernels=True` path. Dynamic variant preserved as
`_forward_dynamic` for future E2 iteration-warmup curriculum.

### B.4 CLI + trainer integration (~2 days)

#### B.4.1 CLI flag

File: `halo_training/cli.py`.

```python
parser.add_argument("--fullgraph-compile", action="store_true",
    help="OdinHalo-only: compile forward as a single graph via "
         "use_compile_safe_kernels + static depth_kv_buffer. "
         "Requires preconditions documented in "
         "docs/superpowers/specs/2026-05-07-odinhalo-fullgraph-compile-design.md.")
```

When set, auto-passes `use_compile_safe_kernels=True` to OdinHalo's
`HyPEShortConvBlock` children and calls `compile_zones(fullgraph=True)`.

#### B.4.2 `compile_zones` update

File: `models/odin_halo.py::OdinHaloBase.compile_zones`.

```python
def compile_zones(self, mode: str = None, fullgraph: bool = False):
    if fullgraph:
        # Single-graph compile of the forward
        self._compiled_forward = torch.compile(
            self._forward_unrolled, fullgraph=True, mode=mode or "default",
        )
        return
    # existing per-layer compile path
    ...
```

`forward()` dispatches to `_compiled_forward` when set.

#### B.4.3 Trainer wire-up

File: `halo_training/trainer.py`.

Read `args.fullgraph_compile`, set `use_compile_safe_kernels=True` on
all HyPE blocks before `compile_zones(fullgraph=True)` call. No other
changes.

#### B.4.4 Integration tests

Extend `scripts/test_compile_safe_kernels.py`:

- `test_fullgraph_compile_no_breaks` — smoke-instantiate OdinHalo
  with `use_compile_safe_kernels=True`, run `compile_zones(
  fullgraph=True)`, execute forward, assert no
  `torch._dynamo.exc.Unsupported`.
- `test_fullgraph_loss_parity_100_steps` — 100-step training parity
  vs `compile_zones` path, max|Δloss| < 0.25.

## Phase C — Validation (~3 days)

### C.1 Correctness (1 day)

500-step training run on babylm smoke, `use_compile_safe_kernels=True
+ fullgraph compile`, compare to current `compile_zones` baseline
run with same seed.

Gate: `max|Δloss| < 0.25` over all 500 steps.

If gate fails: log failing step's loss delta, weights maxabs,
activation maxabs. Likely root cause is a `custom_op` fake impl
mismatch.

### C.2 Throughput default mode (1 day)

1-epoch wikitext-103 DDP run with `--fullgraph-compile` and
`TORCH_COMPILE_MODE` unset.

Gate: aggregate tok/s ≥ `compile_zones` baseline (14,018 tok/s per
WI-B3 non-max-autotune).

### C.3 Throughput max-autotune (0.5 days)

Same as C.2 but with `TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`.

Gate: aggregate tok/s ≥ 14,742 tok/s (current WI-B3 max-autotune
baseline). Celebrate if > 15,500.

### C.4 CUDA graph capture retry (0.5 days)

Same as C.2 but with `TORCH_COMPILE_MODE=reduce-overhead`.

Report outcome either way. If HIP's graph backend emits "empty graph"
warnings still, document the specific failure mode in
`docs/perf/odinhalo-fullgraph-results-2026-05-07.md` — useful info for
future PyTorch / ROCm upgrade evaluation.

Success here is the 5–15% win; expected but not guaranteed.

## Phase D — Decision & ship (~1 day)

### D.1 Outcome matrix

| Scenario | Action |
|---|---|
| C.1 fail | Loop back to B. Investigate `custom_op` fake impl; fix; re-run C.1. If unfixable in 2 days, archive and document as negative result. |
| C.1 pass, C.2/C.3 regress > 2% | Archive; ship nothing; document findings. |
| C.1 pass, C.2/C.3 parity (±2%) | Ship `--fullgraph-compile` opt-in; document as "foundation for A2, not a throughput play." Leave per-layer as production default. |
| C.1 pass, C.2/C.3 +2% or better | Ship opt-in flag; update AGENTS.md production recommendations to mention fullgraph option; gather 1 week production hours before considering default flip. |
| C.4 CUDA graphs succeed (+5% or more) | Ship `--fullgraph-compile --cuda-graphs` combination; STATUS.md highlight. |

### D.2 Docs update

- `AGENTS.md`: add fullgraph section under compile strategy if
  shipping. Include benchmark numbers.
- `STATUS.md`: Sprint outcome entry.
- `docs/perf/odinhalo-fullgraph-results-2026-05-07.md`: full
  results doc with all four measurements, break before/after
  counts, CUDA graph report, recommendation.
- `knowledge/architectures/a2_reversible_parcae_audit_2026.md`:
  update §5.3 with measured compile cost (likely revise down from
  5–8% estimate).

### D.3 Commit

Single commit at Phase D end. Message format:

```
fullgraph compile on OdinHalo: <outcome summary>

- Phase A: <break count> breaks reduced to <after count>
- Phase B: wrapped <kernels>; static depth_kv_buffer
- Phase C: tok/s <before> → <after>; CUDA graphs <status>
- Phase D: shipped as <opt-in flag | default | archived>

See docs/perf/odinhalo-fullgraph-results-2026-05-07.md.
```

## Kill gates during sprint

Each phase has a hard kill gate that ends the sprint immediately:

- **After Phase A**: if diagnostic shows 20+ breaks scattered
  throughout `_forward_unrolled` (not just HIP + buffer), abandon.
  Engineering cost exceeds the 3-week budget.
- **After Phase B.1**: if `custom_op` backward for
  `causal_conv1d_fn` cannot be registered and `allow_in_graph`
  fallback also breaks under `fullgraph=True`, abandon. Document
  as "DaoAILab extension boundary fundamentally incompatible."
- **After Phase B.2**: if depth_kv_buffer parity tests fail and the
  MoDA attention math proves harder to preserve than expected,
  abandon.
- **After Phase C.1**: if loss parity fails and two days of
  debugging don't identify the cause, abandon.

Each kill produces a written negative-result note. This is the
minimum useful output even in failure.

## Dependencies & prerequisites

- **Must complete before starting**: v3.1 distillation Sprint (so
  E1 doesn't compete for engineering time).
- **Must NOT complete before starting**: A2 reversible Parcae Sprint
  (A2 depends on this Sprint's outcome for its cost model).
- **No code dependencies on in-flight work**: this Sprint can be
  picked up independently by whoever owns v3.2 phase.

## Budget summary

| Phase | Days | Key deliverable |
|---|---|---|
| A | 1 | Diagnostic report + path decision |
| B.1 | 5 | HIP kernels wrapped with custom_op |
| B.2 | 3 | Static depth_kv_buffer + parity tests |
| B.3 | 1 | Loop unroll |
| B.4 | 2 | CLI + trainer wiring |
| C | 3 | 4 validation measurements |
| D | 1 | Results doc + decision + commit |
| **Total** | **16** | ~3 weeks calendar (with buffer) |

## Related docs

- `docs/superpowers/specs/2026-05-07-odinhalo-fullgraph-compile-design.md` — Sprint spec.
- `docs/superpowers/plans/2026-05-07-v3-experiment-roadmap.md` — where this Sprint sits (post-v3.1 infrastructure milestone).
- `knowledge/architectures/a2_reversible_parcae_audit_2026.md` §5.3 — downstream beneficiary.
- `AGENTS.md` compile strategy section — current state.
- `STATUS.md` Phase 3 WI-A0/B2/B3 — measurements this Sprint builds on.
