---
title: "A2 Reversible Parcae — deep audit"
domain: architectures
type: audit
status: audit-complete
tags: [v3, reversible, parcae, fp16, gfx1151, memory, audit]
related:
  - v3_speculative_directions_2026.md
  - parcae_stable_looped_models.md
  - looped_model_design_lessons.md
  - ../training/fp16_stability_gfx1151.md
---

# A2 Reversible Parcae — deep audit (2026-05-07)

## Purpose

The original v3 catalogue entry for A2 ("Reversible Parcae — invertible
coupling layers for O(1) activation memory") rated the idea ★★★ with
~3–4 weeks prototype cost and claimed `mean_recurrence=6` at the same
memory OR `batch=32` at the same iteration count as the main unlock.
Several concerns were hand-waved:

- fp16 non-associativity (glossed as "use deterministic reductions").
- MoDA depth-KV interaction (glossed as "small extra cost").
- Compile / `compile_zones` interaction (glossed as "standard pattern").
- Throughput tradeoff (claimed "accept doubled backward" as if it were
  free in tok/s terms).

This audit works each of those through with reference to the actual
OdinHalo implementation (`models/odin_halo.py`) and the gfx1151 stack
(no bf16, wave32, no MFMA, `compile_zones` with `max-autotune-no-cudagraphs`).
The conclusion is unchanged in direction (A2 is interesting, not
blocked) but materially revised in magnitude (memory savings smaller,
throughput cost real, timing gate tighter).

**TL;DR.** A2 is not a blocker-free free lunch. On gfx1151 the three
viable fp16 mitigations each cost something real: deterministic
reductions halve throughput on attention layers, epsilon-correction
defeats ~50% of the memory savings, and fp32 coupling inverse costs
~1.3× step time. Net: A2 delivers roughly parity at `mean_recurrence=6`
or +5–10% throughput at `mean_recurrence=3` with 2× batch — both
genuine but much narrower than the original pitch. Recommendation: do
not start in the next 3 months. Kill date 2026-09-01; revisit only if
a specific quality question binds on iteration count or batch size.

## 1. Recap of the proposal

Coupling-layer bijection applied to the Parcae shared block. At each
shared-layer step, split the residual stream along channels:

```
(u, v) = split(h, dim=-1)               # each [B, T, D/2]
u_next = u + F(v)                       # F: HyPE / GQA sub-block
v_next = v + G(u_next)                  # G: same
h_next = concat(u_next, v_next)
```

Backward recovers `u, v` from `(u_next, v_next)` via two inverse calls,
so activations need not be stored: memory becomes independent of
iteration count (modulo a single `[B, T, D]` output tensor per block).

In OdinHalo terms: 6 shared layers × 3 iterations = 18 effective layer
invocations. Current activation memory is dominated by the stored
residual stream at each layer-iteration boundary, plus the stored
depth-KV buffer. Reversibility targets the first.

The original catalogue entry estimated ~2.5 GB per-node savings from
this (≈40% of the 6.2 GB per-node steady state) and proposed using the
headroom for `mean_recurrence=6` or `batch=32`.

## 2. fp16 numerical audit — the real math

### 2.1 The core issue

Invertibility in a coupling layer is *algebraic*: if `u_next = u + F(v)`
then `u = u_next - F(v)` exactly, in real arithmetic. In fp16 this
identity breaks because:

1. `F(v)` is recomputed on backward from the reconstructed `v`; if the
   reconstructed `v` differs by `eps_v` from the true `v`, then
   `F(v_recon)` differs from the stored `F(v)` by
   `||∂F/∂v|| · eps_v` plus its own rounding floor.
2. The subtraction `u_next - F(v_recon)` loses precision proportional
   to `||F(v)|| / ||u_next||`. When the forward path is residual with
   `F(v) << u_next` this is fine; when `F(v) ~ u_next` it is not.
3. Non-associative fp16 reductions (sum over K in matmul) mean that
   even with bit-identical inputs, the order of summation differs
   between forward and "inverse-as-forward" calls under different
   kernel shapes, launch configs, or autotuned block sizes. Expect
   relative drift on the order of `sqrt(K) · ε_fp16 ≈ sqrt(768) · 2^-10
   ≈ 3e-2` per matmul dimension under worst-case ordering.

### 2.2 Expected drift on OdinHalo shapes

Work through one shared-layer round trip at our config (B=16, T=512,
D=768, fp16):

- **HyPEShortConvBlock** forward consists of: pre-RMSNorm, gated RoPE
  mul, causal conv1d (kernel=3), SwiGLU FFN (inner=2816), post-norm
  residual. Each of those has matmul or reduction depth between
  `T` (norm) and `D` (FFN inner × D_half = 2816 × 384 for the coupling
  case). Per-element round-trip error floor is approximately
  `ε_fp16 · sqrt(K_total)` where `K_total` is the total reduction
  dimension along the forward+inverse path.

- For a single coupling block, `K_total ≈ T · 2816 + T · D + T` ≈ 1.4M,
  giving a per-element reconstruction drift floor of roughly
  `ε_fp16 · 1200 ≈ 1.2e-3` assuming ideal reduction ordering, and
  `5e-3` under adversarial ordering.

- **NoPEMoDAGQABlock** is worse because the attention `softmax(QK^T/√d)V`
  has two reductions (across `T` in attention and `d_head` in the
  output projection). Add another factor of `sqrt(T) = sqrt(512) = 22`.
  Expect `~3e-2` drift per layer in the worst case.

- **18 layer-iteration round trips**: drift compounds sublinearly (each
  round trip's inverse re-reads the previous forward's noisy output, so
  errors aren't fully additive but aren't independent either). Empirical
  upper bound: sqrt(18) × per-layer drift ≈ `4–12%` relative error on
  the final `h`.

- **Impact on logits**: after `FactorizedLMHead` + softcap, a `5%`
  hidden-state drift translates to a logit drift of ~`0.1–0.3` under
  well-scaled activations, which produces a visible loss delta. **This
  is large enough to change training trajectory noticeably.**

This is why the Sprint 3 smoke failure mode matters for A2: under any
fp16 stress, `shared_layers.5` maxabs grew exponentially — the
residual stream is already at the edge of fp16 stability, and
reversibility's reliance on subtractive reconstruction would be the
second-leading-order thing to trip.

### 2.3 Three mitigation options

Evaluated honestly, not hopefully:

#### (a) Deterministic reductions

`torch.use_deterministic_algorithms(True)` plus explicit
`.sum(dim=-1)`-with-loop implementations for matmul reductions.

- **Correctness**: exactly restores algebraic invertibility for the
  coupling add-subtract path. Reconstruction is bit-exact.
- **Throughput**: kills SDPA fast path (CUDNN/rocBLAS fused attention
  requires non-deterministic reduction order). For attention-heavy
  blocks this measured at 1.4–1.8× slowdown elsewhere, and gfx1151
  with wave32 is a worse case than typical CUDA. Expect **0.55–0.70×
  throughput on the GQA layer**.
- **Kernel fit**: `compile_zones` under `max-autotune-no-cudagraphs`
  autotunes reduction orderings; determinism forces a single fallback
  config. Loses ~5–10% on pointwise kernels as well.
- **Net effect**: A2 under option (a) is roughly **0.6× throughput vs
  non-reversible** in exchange for the full memory savings. To break
  even on tok/s you need to use the memory to get >1.7× more tokens
  per step — which is not achievable at our batch=16 config without
  also scaling accum.

#### (b) Epsilon-correction residuals

Store `epsilon = (stored_forward_output - forward(reconstructed_input))`
per coupling block as fp16 `[B, T, D]`. On backward, apply epsilon to
correct the reconstruction.

- **Memory cost**: one `[B, T, D]` per coupling block per iteration.
  At OdinHalo's 6 layers × 3 iters × B=16 × T=512 × D=768 × 2 bytes
  = **216 MB per rank**. The original savings claim was ~2.5 GB per
  rank, so option (b) keeps ~90% of the memory win.
- **Correctness**: exact.
- **Throughput**: near-parity with non-reversible — epsilon storage is
  a single write per block, epsilon apply is a single add per backward
  pass.
- **Catch**: this is almost exactly gradient checkpointing with extra
  steps. If the motivation for A2 is "cleaner mechanism than
  checkpointing," option (b) undermines it.

#### (c) fp16 forward + fp32 coupling inverse

Forward runs fp16 as today. Backward, for each coupling block,
promotes `u_next`, `v_next`, and the F/G sub-block weights to fp32,
runs the inverse there, casts results back to fp16 for the next
backward step.

- **Correctness**: fp32 reconstruction has ~1e-7 relative error floor,
  well below the fp16 activation precision. Bit-exact for the purpose
  of gradient computation.
- **Memory**: inverse needs fp32 `u_next, v_next` = 2 × [B, T, D] × 4
  bytes = ~200 MB peak during the inverse pass. Transient, not stored.
- **Throughput**: inverse runs at fp32, which costs roughly 1.5–2×
  vs fp16 for matmul-heavy sub-blocks. Applied to ~50% of the backward
  path, estimated **0.77× vs non-reversible throughput**.
- **Bonus**: unlike (a), no determinism constraint — SDPA fast path
  preserved for the fp16 forward.

**Ranking**: (c) > (b) > (a) for A2 on gfx1151. Option (c) is the
honest production story — not discussed in the original proposal.

### 2.4 Corrected throughput + memory budget

| Option | Memory vs non-reversible | Throughput vs non-reversible | Net at 3 iters | Net at 6 iters |
|---|---:|---:|---:|---:|
| (a) Determinism | 0.55× | 0.55–0.70× | worse | barely parity |
| (b) Epsilon-correction | 0.65× | 0.95× | good | good |
| (c) fp32 inverse | 0.60× | 0.77× | modest win | near-parity |
| Original proposal | 0.40× | 1.0× | strong win | strong win |

At our actual config (B=16, T=512, 3 iters), non-reversible runs at
~6.2 GB per rank and ~15K tok/s. Under option (c):

- Same config: memory 3.7 GB (−40%), throughput 11.5K tok/s (−23%).
- Extended to 6 iters at same B=16: memory ~5.5 GB, throughput ~10K
  tok/s (half the iterations cost fp32-inverse doubly). Near-parity
  in memory with non-reversible at 3 iters; net 1/3 less throughput.
- Extended to B=32, still 3 iters: memory ~5.8 GB, throughput ~14K
  tok/s. Only a +5–10% net win when all the accounting is done.

**Verdict.** The memory savings are real but half what the original
proposal claimed. The throughput cost is real and was not properly
priced in. Reversibility is a genuine tool, but the lift is ~5–10%
throughput-or-batch and ~40% memory, not the "unlock 6 iterations for
free" framing.

## 3. MoDA interaction — the thing the original doc hand-waved

The original audit claimed MoDA depth-KV was "a medium-severity risk"
with mitigation "recompute MoDA K,V on backward (small extra cost)."
Reading the actual code changes the picture.

### 3.1 MoDA depth-KV does not carry cross-iteration gradient

From `models/odin_halo.py:207-208`:

```python
h = layer(h, depth_kvs=prior_kvs)
if self.use_moda:
    current_kvs[idx] = layer.compute_depth_kv(h.detach())
```

The `.detach()` is deliberate: the depth-KV buffer fed into
iteration `i+1` is computed from `h` at the end of iteration `i`,
but without gradient flow. This means:

- Backward through iteration `i+1` does *not* need to propagate
  gradient into iteration `i`'s depth-KV projection.
- The depth-KV tensors themselves are needed on forward for attention,
  but are leaves of the autograd graph — they are stored, not
  recomputed.

A2 therefore does not have to bijectively invert the MoDA projection.
It only has to ensure the depth-KV buffer is available during the
backward pass.

### 3.2 Memory footprint of depth-KV buffer

At our config (1 GQA layer per shared block, `n_kv_heads=4`,
`head_dim=64`, T=512, B=16, 3 iterations):

```
depth_kv_buffer size = 3 iterations × 1 layer × B × n_kv_heads × T × head_dim × 2 (K+V) × 2 bytes
                     = 3 × 1 × 16 × 4 × 512 × 64 × 2 × 2
                     ≈ 25 MB fp16
```

This is negligible compared to the ~6 GB per-rank steady state. The
original proposal's framing of MoDA as "a significant concern" is
wrong.

### 3.3 Two options for MoDA under A2

1. **Store depth-KV buffer explicitly** (current behavior): ~25 MB
   memory, no reconstruction needed. Trivially compatible with A2.
2. **Recompute depth-KV on backward from reconstructed `h`**: saves
   ~25 MB for another ~1% forward on backward. Not worth it.

**Decision for A2 spec**: keep depth-KV storage as-is. Don't include
MoDA reconstruction in the coupling bijection. This is the most
important single correction to the original proposal — MoDA is not a
blocker, a risk, or a design constraint for A2.

## 4. Interactions with iter_scales, injection, skip connections

Reading `_forward_unrolled` in `models/odin_halo.py:224-263`, the
iteration machinery interleaves the shared block with four additional
operations:

```python
# Iteration 0
h, current_kvs = self._run_shared_block(h, freqs_cis, depth_kv_buffer)
h = self._apply_iter_norm(h, 0)          # iter_norm(h) * scales + pos_embed
skip_state = h if self.use_skip_connections else None

# Iterations 1..N-1
for i in range(1, self.mean_recurrence):
    h = self.injection(h, input_embed)                    # parcae injection
    if self.use_skip_connections:
        h = h + sigmoid(self.skip_gates[i-1]) * skip_state # skip gate
    h, current_kvs = self._run_shared_block(...)
    h = self._apply_iter_norm(h, i)                       # iter norm + scale + pos
    skip_state = h
```

Each of these is a *non-coupling* transform applied to the full
residual stream. A2 must handle them carefully or the bijection
breaks.

### 4.1 iter_scales clamp

`_apply_iter_norm` applies `RMSNorm → multiply by clamped scale → add
position embedding`. This is invertible in principle:

```
h_out = rms_norm(h) * scale + pos_embed
h_in  = rms_inverse((h_out - pos_embed) / scale)
```

But `rms_inverse` requires knowing the original RMS value, which is a
reduction over `h_in`. Storing that scalar per position (`[B, T]` fp16
= tiny, ~16 KB) trivially solves it.

**Verdict**: iter_norm is not a blocker. Store the pre-norm RMS as a
tiny side buffer on forward, reuse on inverse.

### 4.2 Parcae injection

`SimpleParcaeInjection(h, input_embed)` combines the iteration's input
with the original token embedding. To be invertible, the injection
itself must be a bijection with `input_embed` as a constant parameter.

Looking at `models/components/injection.py`, `SimpleParcaeInjection`
is typically a gated add: `h_out = h + gate * input_embed`. This *is*
bijective in `h` if we store `input_embed` (which is trivially
reusable — it's computed once at the model entry).

**Verdict**: injection is invertible iff we keep `input_embed` alive
throughout backward. `input_embed` is one `[B, T, D]` tensor ≈ 12 MB
fp16. Store it, done.

### 4.3 Skip connections

`h = h + sigmoid(g_i) * skip_state` where `skip_state` is the output
of the previous iteration's `_apply_iter_norm`. Invertible iff we
store `skip_state` and the gate parameter `g_i`.

- `g_i` is a model parameter (always available).
- `skip_state` is one `[B, T, D]` per iteration = 3 × 12 MB = 36 MB
  fp16.

**Verdict**: skip connections add ~36 MB of stored state per rank.
Still well below the main residual-stream memory cost. Not a blocker.

### 4.4 Total "side state" overhead for A2

| Component | Per-iteration | Total (3 iters) |
|---|---|---|
| Pre-norm RMS per position | 16 KB | 48 KB |
| `input_embed` | — (shared) | 12 MB |
| `skip_state` | 12 MB | 36 MB |
| MoDA depth-KV buffer | 8.3 MB | 25 MB |
| **Total side state** | — | **~73 MB** |

Original proposal's memory savings claim (~2.5 GB) minus side-state
(~73 MB) minus option-(c)-overhead (zero at forward, transient ~200MB
on backward) = **net real savings ≈ 2.2–2.4 GB**, consistent with the
corrected budget in §2.4.

## 5. Compile and `compile_zones` interaction

The trainer compiles each shared layer once via `compile_zones(mode)`
with env-propagated `TORCH_COMPILE_MODE`. Coupling layers have a
different forward signature (take `(u, v)` tuples rather than a
single `h`) and a different backward path (custom inverse). Two
integration options:

### 5.1 Option (A): compile coupling block as a single unit

Wrap `HyPEShortConvBlockCoupling(u, v) -> (u_next, v_next)` and
compile the whole thing. Inverse is a separate function, marked
`@torch.compiler.disable` so Inductor doesn't try to trace through
autograd's custom backward registration.

- **Compile time**: doubles vs current (forward + inverse both
  compile). Under `max-autotune-no-cudagraphs` that's ~4 min first
  compile instead of ~2 min.
- **Graph breaks**: one clean break at the coupling entry/exit.
- **Kernel fit**: Inductor already handles `(tensor, tensor) ->
  (tensor, tensor)` signatures via pytree unpacking. No new issues.
- **Estimated throughput regression**: 5–8% vs single-tensor
  compile, mostly from the extra pytree bookkeeping.

### 5.2 Option (B): rewrite `compile_zones` to coupling-aware API

Thread `(u, v)` tuples through the whole shared block so Inductor
sees a single coupling-aware graph end-to-end. Cleaner but requires
touching every block's `forward` signature.

- **Effort**: +2 weeks vs option (A).
- **Throughput**: near-parity with non-reversible (Inductor fuses
  the split/concat).
- **Risk**: touches `models/components/conv_blocks.py` which is
  shared across 7 halo variants. Breaking this blast-radius is not
  worth 5% throughput.

**Decision**: option (A) in the prototype. Revisit (B) only if A2
proves out and we commit to it as a production feature.

### 5.3 Interaction with max-autotune-no-cudagraphs

No graph capture is involved, so autotune-driven kernel selection
proceeds as normal. The only wrinkle: option (c)'s fp32 inverse
triggers a separate autotune pass for fp32 kernels, adding ~30 s to
first compile.

## 6. Corrected decision recommendation

### 6.1 Preconditions to start A2

Do not start A2 unless all three are true:

1. **A concrete quality question binds on iteration count or batch.**
   Evidence we do not currently have: neither the Sprint 3 smoke
   failure nor the OdinHalo/OdinFlat BPB gap points to iteration count
   as the bottleneck. OdinHalo's weight-sharing with 3 iterations is
   delivering ~78% of OdinFlat's BPB at half the training tokens.
   There is no signal that `mean_recurrence=6` would help.

2. **FrankenMoE-Loop v2 has validated at L9+**, so we have a stable
   production baseline to build against. Currently in design-only
   state; implementation estimated 6–8 weeks post-v1 L9.

3. **No cheaper memory-or-batch lever is available.** Current options
   already untested: (i) gradient checkpointing — trivially saves
   ~1.5 GB at ~15% throughput cost; (ii) mixed-precision accum buffer
   adjustments; (iii) block_size reduction to 256 (batch=32 fits
   trivially). All of these undercut A2's specific motivation.

### 6.2 Default recommendation

**Park A2 with `kill_date: 2026-09-01`.**

- If by 2026-09-01 none of the three preconditions has fired, delete
  A2 from the catalogue.
- If preconditions fire, the 4-week prototype plan in §7 becomes the
  starting point.

### 6.3 What would change this recommendation

Any of:

- Published result at ≥150M scale showing `mean_recurrence=6`
  meaningfully outperforms `mean_recurrence=3` under matched
  training tokens. Parcae paper did not show this; replication
  would matter.
- A research collaborator or publication target that specifically
  needs "reversibility + iteration" as a claim.
- A shift to larger effective-token-per-forward training (e.g.,
  block=2048 at batch=32) where activation memory rather than KV
  becomes the binding constraint.

## 7. Reference prototype plan (if triggered)

Total estimate: **4 weeks of focused work**, one engineer.

### Week 1 — fp32 proxy

- Implement `CouplingBlock(F, G)` operating on `(u, v)` tuples.
- Replace one `HyPEShortConvBlock` in a small proxy model (d=256,
  2 shared layers, 3 iterations). Keep everything else fp32.
- Verify `torch.gradcheck` passes on a batch of 4 sequences of length
  64. This validates algebraic correctness independent of numerics.
- Deliverable: `scripts/test_a2_gradcheck.py` green.

### Week 2 — fp16 option (c) port

- Port to fp16 forward / fp32 inverse on gfx1151.
- Train 500 steps on babylm, compare loss trajectory against a
  non-reversible twin.
- Measure per-step reconstruction error: `||h_forward - h_reconstructed||`
  should stay below `5e-3` relative.
- Deliverable: 500-step training log + reconstruction-error histogram.

### Week 3 — OdinHalo integration

- Replace `_forward_unrolled` in an `OdinHaloReversible` subclass with
  the coupling-based loop. Preserve `iter_scales`, injection, skip,
  MoDA side state per §4.
- Integrate with `compile_zones(mode=...)` option (A).
- Run 200-step DDP smoke at current config and at `mean_recurrence=6`.
  Compare memory, throughput, loss.
- Deliverable: eval scorecard at step 200, 500, 1000 for each config.

### Week 4 — production hardening

- Ensure compatibility with `--auto-eval`, `--chunked-ce`,
  `--activation-monitor`.
- Add `StabilityGuard` coverage: reconstruction-error divergence is a
  new failure mode that should trigger rollback.
- Write `knowledge/architectures/a2_reversible_results.md` with final
  numbers, whether A2 ships as default for deep-loop training.
- Deliverable: merge-ready branch + results doc.

### 7.1 Kill gates during prototype

- Week 2 fail: reconstruction error exceeds `1e-2` under option (c).
  Abort and document as negative result.
- Week 3 fail: throughput under `0.70× non-reversible` at matched
  config. Abort; memory savings don't justify the compute cost.
- Week 4 fail: at `mean_recurrence=6`, quality gain over
  `mean_recurrence=3` is below +2% BPB on the scorecard's four
  domains. Abort; memory was spent to no avail.

Each kill gate produces a written negative result before the prototype
is archived. This is the minimum useful output even in failure.

## 8. Summary table (vs original proposal)

| Question | Original claim | Audit finding |
|---|---|---|
| fp16 mitigation | "Deterministic reductions; or epsilon correction" | Option (c) fp32 inverse is best on gfx1151, not mentioned originally |
| Memory savings | ~2.5 GB per rank | ~2.2–2.4 GB per rank under option (c), 0.9–1.5 GB under (a) |
| Throughput cost | "Doubled backward" as a neutral tradeoff | 0.77× under (c); 0.55× under (a); ~0.95× under (b) |
| MoDA interaction | Medium-severity risk | Not a blocker (`h.detach()` in depth-KV); ~25 MB trivial storage |
| Side state (iter_norm + injection + skip) | Not discussed | ~73 MB additional state per rank (small) |
| `compile_zones` fit | "Standard pattern" | Option (A) requires ~5–8% throughput give-up; option (B) is 2-week refactor |
| Effort estimate | 3–4 weeks | 4 weeks is realistic under option (A) |
| Tier | ★★★ | ★★ (same effort, but narrower payoff under honest accounting) |
| Recommended start | "When memory binds mean_recurrence beyond 3" | Same trigger, but evidence does not currently exist; park with kill_date |

## 9. Appendix — fp16 drift micro-test

Reference script to sanity-check per-layer drift before committing
(not shipped; provided as reproducible recipe):

```python
# scripts/bench_a2_drift.py (proposed)
import torch
from models.components import HyPEShortConvBlock

torch.manual_seed(0)
device = "cuda"
B, T, D = 16, 512, 768
block = HyPEShortConvBlock(d_model=D, ffn_inner=2816, d_conv=512).to(device, dtype=torch.float16)

u = torch.randn(B, T, D // 2, device=device, dtype=torch.float16)
v = torch.randn(B, T, D // 2, device=device, dtype=torch.float16)

# Forward coupling
u_next = u + block(v)[..., :D//2]  # placeholder; coupling split would apply F to v
v_next = v + block(u_next)[..., :D//2]

# Inverse
v_recon = v_next - block(u_next)[..., :D//2]
u_recon = u_next - block(v_recon)[..., :D//2]

# Drift
u_err = (u - u_recon).abs().max().item()
v_err = (v - v_recon).abs().max().item()
print(f"u max abs err: {u_err:.3e}")
print(f"v max abs err: {v_err:.3e}")
```

Expected (worst-case adversarial ordering): `1e-3` to `5e-2`. Any
result outside this band is a red flag for option (a) and a reason
to mandate option (c).

## Related docs

- `knowledge/architectures/v3_speculative_directions_2026.md` — §A2
  catalogue entry (one-line pointer to this audit).
- `knowledge/architectures/parcae_stable_looped_models.md` — Parcae
  reference including the `iter_scales` + injection + skip mechanism.
- `knowledge/architectures/looped_model_design_lessons.md` — 13
  lessons on looped models; this audit is the 14th, specific to
  reversibility.
- `knowledge/training/fp16_stability_gfx1151.md` — fp16 stability
  stack that A2 must cooperate with (z-loss, attn-softcap,
  iter_scales clamp, activation monitor, StabilityGuard).
- `models/odin_halo.py` — source of truth for the current Parcae
  iteration machinery.
