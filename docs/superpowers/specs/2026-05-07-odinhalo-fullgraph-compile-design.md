---
title: "OdinHalo Fullgraph Compile Sprint — design"
domain: specs
type: spec
status: design-ready
sprint: fullgraph-compile
tags: [torch.compile, fullgraph, custom-op, cuda-graphs, parcae, odinhalo, gfx1151]
related:
  - ../plans/2026-05-07-odinhalo-fullgraph-compile-plan.md
  - ../../../knowledge/architectures/a2_reversible_parcae_audit_2026.md
  - ../../../AGENTS.md
---

# OdinHalo Fullgraph Compile Sprint — design (2026-05-07)

## Status

**Design-ready. No code yet.** Sprint slots in after v3.1 distillation
lands (~6 weeks out). ~3 weeks single-engineer budget.

## 1. Motivation

Two goals, honestly priced:

1. **Throughput win** — measured ceiling ~2–5% without CUDA graphs,
   up to 15% if HIP CUDA graph capture starts working. Phase 3 WI-B2
   measured whole-model compile at +1.3% over per-layer `compile_zones`
   at our current graph-break boundaries; removing the boundaries is
   the work in scope here.

2. **Enable downstream compile work** — A2 reversible Parcae (see the
   audit) currently prices in a ~5–8% compile regression because
   coupling blocks disturb `compile_zones`'s per-layer granularity. A
   clean fullgraph foundation removes that tax. Same for any future
   `autograd.Function` patterns or `torch.export` paths.

**Not in scope** (explicit non-goals):

- Other halo variants (Vidar, Tyr, Baldr, Fenrir, Chimera, OdinFlat).
  They stay on `compile_zones` until this proves out on OdinHalo.
- Rewriting HIP kernels as Triton (`fused_rope_gate_mul` and
  `causal_conv1d_fn` stay as-is; we wrap them, not replace them).
- Any training-behavior changes. Loss trajectory must match the
  current `compile_zones` path within fp16 noise.

## 2. Root cause recap

Under `torch.compile(model, fullgraph=True)` on `OdinHalo.forward`
today, the graph breaks come from three sources identified by reading
the code in `models/odin_halo.py::_forward_unrolled`:

1. **HIP kernels with `@torch.compiler.disable`** in
   `HyPEShortConvBlock.forward` (`models/components/conv_blocks.py:262-298`):
   - `fused_rope_gate_mul` — 1 break per block invocation.
   - `causal_conv1d_fn` (DaoAILab extension) — 3 breaks per
     invocation (extension entry + out-tensor handling + exit).
   - Total across 5 HyPE blocks × 3 Parcae iterations =
     **60 breaks per forward**.

2. **Python list mutation on `depth_kv_buffer`**
   (`models/odin_halo.py:230, 237, 249`):
   ```python
   depth_kv_buffer: List[Dict[int, Tuple[Tensor, Tensor]]] = []
   ...
   depth_kv_buffer.append(current_kvs)    # mutating state
   prior_kvs = [buf[idx] for buf in depth_kv_buffer if idx in buf]   # iterating it
   ```
   List append + dict-indexed lookup is a data-dependent pattern
   Dynamo traces with overhead and may break on `reduce-overhead`.

3. **`h.detach()` in the MoDA depth-KV write**
   (`models/odin_halo.py:208`) — fine inside a graph (just an op) but
   an explicit CUDA-graph-capture blocker because it signals gradient
   severance at an unstable boundary.

The Python `for i in range(1, self.mean_recurrence)` **is not** a
break source. Dynamo unrolls static-bound loops. This is the
misconception that motivated the user's question and the AGENTS.md
clarification committed in `f275bdd`.

## 3. Design overview

Four sequential phases. Each has a clear gate before advancing.

### Phase A — Diagnostic (1 day)

Confirm the break hypothesis before committing to a fix path.

**Deliverable**: `scripts/diagnose_fullgraph.py` (ships at Sprint
start, usable immediately). Runs OdinHalo forward under
`torch.compile(fullgraph=True, dynamic=False)` with `torch._dynamo.config.verbose = True`. Captures the break report to
stdout + `docs/perf/fullgraph-diagnostic-2026-05-07.md`.

**Decision gate A**:

| Diagnostic outcome | Path |
|---|---|
| 3–6 breaks concentrated in HIP kernels + depth_kv_buffer | **S2** (default) |
| Few breaks, simple fixes only | S1 minimal |
| Many breaks (10+) scattered | S3 scan-based |

Expected: S2. We'll commit to it unless the diagnostic surprises us.

### Phase B — Break fixes (~2 weeks)

#### B.1 HIP kernel wrapping via `torch.library.custom_op`

Target files:

- `models/components/conv_blocks.py::HyPEShortConvBlock.forward`
  (lines 262–298).
- `kernels/hip/fused_rope_gate_mul.py`.
- `kernels/causal_conv1d/` — DaoAILab extension adapter.

Pattern (applied to each kernel):

```python
@torch.library.custom_op(
    "halo::fused_rope_gate_mul", mutates_args=()
)
def fused_rope_gate_mul(
    b: Tensor, h_tilde: Tensor,
    freqs_cos: Tensor, freqs_sin: Tensor,
    T: int, d_conv: int, rope_pairs: int,
) -> Tensor:
    # Delegates to the existing HIP implementation
    from kernels.hip.fused_rope_gate_mul import kernel_fn
    return kernel_fn(b, h_tilde, freqs_cos, freqs_sin, T, d_conv, rope_pairs)

@fused_rope_gate_mul.register_fake
def _(b, h_tilde, freqs_cos, freqs_sin, T, d_conv, rope_pairs):
    # Shape inference for Inductor
    return torch.empty_like(b, dtype=torch.float32)

@fused_rope_gate_mul.register_autograd
def _(ctx, grad_output):
    # Backward delegates to the existing gradient path
    ...
```

**Behind a feature flag**. `HyPEShortConvBlock.__init__` gains a
`use_compile_safe_kernels: bool = False` parameter. Default preserves
current behavior for other halo models; OdinHalo overrides when
`--fullgraph-compile` is passed.

**Fallback**: if `register_autograd` proves difficult for
`causal_conv1d_fn` (DaoAILab's extension has custom gradient handling
that may not cleanly adapt), fall back to
`@torch.compiler.allow_in_graph` for that specific kernel. Still
fixes the graph break, loses a small amount of fusion scheduling
flexibility.

#### B.2 Static `depth_kv_buffer`

Target: `models/odin_halo.py::OdinHaloBase._forward_unrolled` and
`_run_shared_block`.

Change:

```python
# Before — dynamic list of dicts
depth_kv_buffer: List[Dict[int, Tuple[Tensor, Tensor]]] = []
depth_kv_buffer.append(current_kvs)

# After — preallocated static tensor (K and V stacked along dim 3)
# Shape: [mean_recurrence, n_moda_layers, B, 2, n_kv_heads, T, head_dim]
n_moda = len(self.gqa_positions)
depth_kv_buf = torch.zeros(
    self.mean_recurrence, n_moda, B, 2, self.n_kv_heads, T, self.head_dim,
    device=h.device, dtype=h.dtype,
)
depth_kv_valid = torch.zeros(
    self.mean_recurrence, dtype=torch.bool, device=h.device,
)
# Indexed write during iteration — static shape, Dynamo-friendly
depth_kv_buf[iter_idx, layer_ordinal] = torch.stack([k, v], dim=1)
depth_kv_valid[iter_idx] = True
```

`_run_shared_block` signature updates to accept the tensor +
valid-mask; `NoPEMoDAGQABlock.forward` updates to slice the tensor
instead of iterating a list.

Memory impact: at OdinHalo's config (`mean_recurrence=3`,
`n_moda=1`, `B=16`, `n_kv_heads=4`, `T=512`, `head_dim=64`):
`3 × 1 × 16 × 2 × 4 × 512 × 64 × 2 bytes ≈ 25 MB`. Identical to
what the list-of-dicts currently consumes; no regression.

#### B.3 Loop structure

Manual unroll of `_forward_unrolled` to three explicit iteration
blocks. Rationale: Dynamo unrolls static loops anyway, but the
unrolled form is easier to debug and produces marginally faster
trace (no loop-variable handling).

Dynamic-iteration variant (for future E2 iteration-warmup
curriculum compatibility) kept under a separate method
`_forward_dynamic` that falls back to `compile_zones` per-layer
when fullgraph is incompatible.

#### B.4 CLI + trainer integration

- `halo_training/cli.py`: new flag `--fullgraph-compile`.
- `models/odin_halo.py::OdinHaloBase.compile_zones`: accepts
  `fullgraph: bool = False` kwarg. When True, bypasses per-layer
  compile and compiles `_forward_unrolled` as a single graph.
- `halo_training/trainer.py`: reads flag, picks path. Default path
  (no flag) preserves current behavior byte-for-byte.

### Phase C — Validation (~3 days)

Four measurements on Machine A, batch=16 block=256, OdinHalo
production config:

1. **Correctness**: 500-step loss parity vs `compile_zones` baseline.
   Gate: `max|Δloss| < 0.25` (within documented fp16 noise band).
2. **Throughput (default compile)**: tok/s comparison to per-layer.
   Gate: ≥ per-layer baseline (no regression).
3. **Throughput (max-autotune)**: same, with
   `TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`.
   Gate: ≥ current production baseline (14,742 tok/s per WI-B3).
4. **CUDA graph capture retry**: re-run with
   `TORCH_COMPILE_MODE=reduce-overhead`. Measure whether HIP graph
   capture now succeeds. Report outcome either way — this is the
   biggest theoretical lever and the WI-A0 open question.

### Phase D — Decision & ship (~1 day)

| Outcome | Action |
|---|---|
| Fullgraph ≥ +2% over per-layer AND/OR CUDA graphs succeed | Ship `--fullgraph-compile` opt-in; update AGENTS.md; STATUS.md entry. Do NOT flip default; gather production hours first. |
| Fullgraph within ±2% of per-layer (no CUDA graph win) | Ship opt-in flag; document as "foundation for A2 reversible, not a throughput play." Per-layer remains production default. |
| Fullgraph regresses >2% | Archive. Write negative-result note in `docs/perf/`. Revisit when PyTorch/ROCm updates. |

## 4. Risk register

| Risk | Severity | Likelihood | Mitigation |
|---|---:|---:|---|
| `torch.library.custom_op` backward registration mishandles fp16 grad for `causal_conv1d_fn` | Med | Med | fp32 `gradcheck` gate before fp16 deploy. If register_autograd fails, fall back to `allow_in_graph` for that op. |
| Static depth_kv_buffer breaks subtle MoDA attention invariant | Med | Low | Step-by-step parity tests at 10/100/500 steps. |
| `causal_conv1d_fn` wrapping requires patching DaoAILab extension | Med | Med | Adapter layer insulates from upstream changes. `allow_in_graph` fallback preserves usability. |
| HIP CUDA graphs still fail with static shapes | Med | High | Expected; ship fullgraph default mode regardless. Document as negative result. |
| Changes touch shared `conv_blocks.py` — other halo variants regress | High | Low | `use_compile_safe_kernels` feature flag; default False preserves every other model. |
| Compile time balloons with max-autotune + fullgraph | Low | Med | Measure; worst case ~4 min first compile, ~15s warm. |
| Diagnostic reveals breaks I didn't anticipate | Med | Med | Decision gate A reroutes to S1 or S3. Budget absorbs one replan. |

## 5. Rejected alternatives

- **S1 minimal (`allow_in_graph` only, no `custom_op`)** — viable as
  fallback but loses Inductor's ability to schedule around the HIP
  call. A2 readiness is weaker. We'll use this path only if S2
  proves blocked.
- **S3 scan-based (`torch.ops.higher_order.scan`)** — cleanest
  long-term architecture, but stability varies by PyTorch version
  and rewrite blast radius exceeds the sprint budget. Revisit if E2
  iteration-warmup curriculum actually needs dynamic iteration count
  under compile.
- **S4 `_compile_friendly=True`** — already shipped; swaps HIP for
  native PyTorch; 0 breaks but NOT faster (HIP beats Inductor
  triton for these ops). Useful as debug tool, not production path.

## 6. Deliverables

Ship-at-Sprint-start (this spec + plan + tools):

1. `docs/superpowers/specs/2026-05-07-odinhalo-fullgraph-compile-design.md` (this doc).
2. `docs/superpowers/plans/2026-05-07-odinhalo-fullgraph-compile-plan.md` (companion plan).
3. `scripts/diagnose_fullgraph.py` (Phase A tool).
4. `scripts/test_compile_safe_kernels.py` (Phase B test scaffold with
   skip markers until kernels are wrapped).

Ship-at-Sprint-end (Phase D outputs):

5. Wrapped HIP kernels in `models/components/conv_blocks.py`
   (feature-flagged).
6. Static depth_kv_buffer in `models/odin_halo.py` (feature-flagged).
7. `--fullgraph-compile` CLI flag.
8. `docs/perf/odinhalo-fullgraph-results-2026-05-07.md` validation
   report.
9. STATUS.md entry with outcome.

## 7. Downstream impacts

- **A2 Reversible Parcae** (see
  `knowledge/architectures/a2_reversible_parcae_audit_2026.md` §5.3):
  A2's "compile regression 5–8%" estimate is under current
  per-layer compile. If this Sprint succeeds, A2 rebases to
  "compile parity or small win." This strengthens A2's decision
  case at its 2026-09-01 kill date.
- **E1 Self-distillation**: teacher uses its own `compile_zones`
  path. Student could opt into `--fullgraph-compile`; expected
  marginal gain (~1–3% tok/s on student side). Not worth coupling
  Sprint B and E1 Sprint; run E1 on per-layer first.
- **FrankenMoE-Loop v2**: MoE dispatch has its own compile story
  (ScatterMoE). Orthogonal; this Sprint doesn't touch it.

## 8. Timeline

| Phase | Duration | Gate |
|---|---|---|
| Pre-sprint (this spec + plan) | Ships now | Approved |
| Phase A diagnostic | 1 day | Break report published, S1/S2/S3 chosen |
| Phase B fixes | 10–12 days | All tests green, no regression on default path |
| Phase C validation | 3 days | 4 measurements complete |
| Phase D decision + ship | 1 day | STATUS + AGENTS updated |
| **Total** | **~3 weeks** | — |

## 9. Related docs

- `docs/superpowers/plans/2026-05-07-odinhalo-fullgraph-compile-plan.md` — companion plan doc.
- `knowledge/architectures/a2_reversible_parcae_audit_2026.md` — A2 audit; benefits from this Sprint's foundation.
- `AGENTS.md` (compile strategy section, updated `f275bdd`) — the
  why-per-layer explanation that motivated this Sprint.
- `STATUS.md` Phase 3 WI-A0 / WI-B2 — the measurements this Sprint
  builds on (+1.3% whole-model ceiling, reduce-overhead HIP failure).
- `models/odin_halo.py` — the forward path being restructured.
- `models/components/conv_blocks.py` — shared block under kernel wrapping.
