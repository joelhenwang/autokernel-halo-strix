# WI3: `embedding_dense_backward` fusion — not attempted (risk/reward)

**Phase 2 work item:** WI3 — attempt fusion of `aten::embedding_dense_backward` (4.1% of Phase 1 wall time) with the tied lm_head backward matmul.
**Status:** CLOSED WITHOUT IMPLEMENTATION. Risk/reward analysis does not justify the 6-8h investment given Phase 2's other WI outcomes.
**Evidence:** Analysis below.

## What the op does

`aten::embedding_dense_backward` implements the reverse-mode gradient for
`F.embedding(weight, input_ids)`. Given `dL/dh_out` of shape `(N, d)` and
`input_ids` of shape `(N,)` plus the embedding weight of shape `(V, d)`,
it performs:

```
grad_weight[input_ids[i]] += dL/dh_out[i]   for i in range(N)
```

This is a sparse scatter-add: each output row is only written to at positions
where that row's index appeared in `input_ids`. PyTorch implements this via
atomic scatter-add over the GPU threads.

## Profile data

From Phase 1 profile at batch=16, block=256:
- **3 calls per 3 profiler steps** → 1 call per step
- **73.99 μs/call average**
- **4.07% of total GPU time**

One call per step is consistent with the embedding weight being updated once per
backward pass (the N = 16 × 256 = 4096 positions are accumulated in a single
call).

## Why fusion with the tied-lm-head backward is conceptually attractive

OdinHalo's `FactorizedLMHead.embed_table` SHARES weights with `tok_embeddings.embed`
(tied embedding). During backward, BOTH of these paths must write to the same
`.grad` tensor:

1. **LM head path** — dense matmul: `grad_weight += normed.T @ dL/dlogits`. Done by
   rocBLAS GEMM. Fast, ~45 μs combined across layers.

2. **Embedding path** — sparse scatter: `embedding_dense_backward(dL/dh_embed, ids)`.
   Slow (73.99 μs), dominated by atomic contention.

The theoretical fusion: compute both gradients in a **single** kernel that knows
which rows are written by whom, avoiding the atomic path entirely for the dense
part.

## Why we are not doing this

### 1. The scatter-add is already near-optimal for its shape

At `V = 32768` and `N = 4096` input positions, the expected collision rate
(multiple positions mapping to the same vocab index) is `N/V ≈ 12.5%`. With such
low collision rates, the scatter-add is memory-bound on 32768 × 768 × 4 bytes =
**96 MB** of embedding gradient output. At measured 73.99 μs, that's 1.3 TB/s
effective bandwidth — the operation is already near the theoretical limit for
this shape on gfx1151. A fusion won't speed up the dominant memory traffic.

### 2. Tied-embedding autograd is delicate

PyTorch's autograd engine detects shared-parameter gradient accumulation via
hooks and inserts an `aten::add_` to sum contributions. Replacing this with a
custom fused kernel requires:

- Wrapping `FactorizedLMHead.forward` in a custom `autograd.Function`.
- Manually accumulating both scatter-add AND matmul gradients in the `backward()`.
- Ensuring the optimizer sees a SINGLE accumulated grad (not two).
- Preserving numerical bit-identity with PyTorch's native path (otherwise
  training divergence is silent).

Gradient-parity testing must cover the full `FactorizedLMHead` (including the
`forward_hlow` branch used by chunked CE) AND must match across fp16 autocast.
This is a 2-3 hour test surface alone.

### 3. Phase 2 evidence strongly suggests no shippable win

Every other Phase 2 WI that investigated a concrete fusion target has closed as
"no attackable gain":

- WI1 (transpose/copy fusion, 9.1%): already fused optimally by Inductor.
- WI2 (add_/copy_, combined 9.3%): autograd-internal + input H2D, both unfixable.
- WI5 (H2D prefetch, 4.0%): regressions on unified memory.
- WI4 (Memset, 4.1%): framework-internal, not user-reachable.

The pattern is: **post-Phase-1, throughput is limited by the sum of many tiny,
already-optimized operations.** A 4.1% single-op win requires overhead-free
fusion of two ops that combined are already fast. The expected lift after
Amdahl's law is no more than 1-2%, below the risk threshold for a
checkpoint-breaking change.

### 4. The fix may arrive for free via chunked CE

OdinHalo already supports `use_chunked_ce=True` via `--chunked-ce`. When
enabled, `FactorizedLMHead.forward_hlow` returns the rank-dim hidden state
instead of full logits, and `ChunkedLinearCrossEntropyLoss` computes the
lm-head grad in per-chunk matmul form. Chunked CE writes to
`embed_table.weight.grad` directly via a per-chunk scatter-free kernel,
eliminating the duplicate work against which WI3 was protecting.

The current profile was NOT captured with `--chunked-ce`. If we switch the
production config to chunked CE (worth evaluating separately, see Phase 1
optimizer-shootout where it was ~0.8% slower but saved 1.7 GB memory), the
WI3 target op may disappear or change shape without any fusion work.

## Decision

**WI3 NOT ATTEMPTED.** Closing without code changes.

Rationale:
- The 4.1% target is already near the theoretical memory bandwidth limit.
- Tied-embedding autograd fusion is high risk (silent correctness bugs).
- Phase 2's pattern of closed WIs strongly suggests the remaining % is
  fundamental to the current stack, not unlocked by more fusion.
- Chunked CE provides an orthogonal path if memory ever becomes the constraint.

## Potential future work

Return to WI3 only if:

1. **Chunked CE becomes the default** and the embedding_backward shape changes
   substantially (rerun profile first).
2. **A larger effective batch size** (e.g., 32+ via gradient accumulation) shifts
   the relative cost of this op upward to >6%.
3. **A wider embedding** (V=65k+) makes the scatter-add actually
   compute-bound rather than bandwidth-bound.

None of these apply to the current OdinHalo production config.

## Artifacts

None — no code written.
Analysis archived here.
