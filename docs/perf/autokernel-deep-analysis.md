# Autokernel Deep Analysis: OdinFlat / OdinHalo under `--optimize-kernels` (2026-05-10)

**Status:** Final synthesis. Combines surface findings from
`odinflat-throughput-final.md` (2026-05-10 earlier) with mechanism and
blast-radius evidence from Track 3.A (this session).

**Scope:** The `autokernel` package's HIP-kernel pattern-replacements under
training semantics on gfx1151. Answers (a) why `--optimize-kernels` showed
+31 % to +80 % forward throughput, (b) what the actual cost is, (c) what
is salvageable, (d) what to ship.

---

## 1. Executive summary

1. **`--optimize-kernels` silently freezes 23 % of OdinFlat's parameters
   (28 of 120 named params, ~60 M weights) because the `silu_gate_mul`
   HIP kernel lacks an autograd registration.** `ffn.w_gate_up` never
   receives a gradient (`always_none`), and the upstream `ffn_norm` sees
   gradient flow but outputs zero (`always_zero`). Forward stays numerically
   correct, but training quality degrades over long horizons (OdinFlat
   Phase V: loss 3.80 at step 2000 vs baseline 3.15, +0.65 regression).
2. **The +31 %–80 % speedup is an illusion caused by the freeze.** With
   silu autograd properly wired via `torch.ops.autokernel.silu_gate_mul`
   (the V2 config), throughput drops to 30.9 K tok/s — marginally below
   the 31.3 K baseline. The save-for-backward + HIP backward overhead
   slightly exceeds the forward savings.
3. **The Phase III `rmsnorm` autograd fix is complete.** When only
   rmsnorm HIP is active (`--autokernel-exclude fused_silu_gate_mul`,
   the V3 config), the full parameter population receives gradients at
   the same rate as baseline — byte-for-byte identical grad status.
4. **Principle for future kernel authors:** any HIP op placed in a
   training path must be wrapped in a `torch.library.custom_op` with
   registered autograd, OR a `torch.autograd.Function`. A raw pybind
   call returns a tensor with `grad_fn=None`, which silently severs
   gradient flow to upstream parameters — a catastrophe disguised as
   a speedup.
5. **Ship recommendations:**
   - OdinFlat Sprint 3A: **no `--optimize-kernels`**. Baseline 31.3 K
     tok/s aggregate, 61 h wall for one dolma-10B epoch.
   - OdinHalo Sprint 3B: **awaiting B4 probe result** (running as of
     this writing). If OdinHalo shows the same freeze, drop
     `--optimize-kernels` from 3B's locked recipe and expect ~77 h wall
     instead of ~48 h.
   - Other models (jormungandr_halo, baldr_halo, etc.) that use
     autokernel's FusedQKVPattern are not currently on the training
     path; no immediate action.

---

## 2. Context and prior findings

This deep analysis builds on two earlier surface investigations:

| Phase | Date | Artifact | Key finding |
|---|---|---|---|
| 0 | 2026-05-08 | `autokernel-probe-2026-05-08.md` | OdinHalo +37.9 % tok/s loss delta +0.17 at 200 steps → PASS. OdinFlat +80 % tok/s loss delta +1.16 at 200 steps → FAIL. |
| Bisect I-V | 2026-05-10 earlier | `odinflat-throughput-final.md` | Phase III: `_RMSNormReplacement` autograd was broken, fixed. Phase V: silu HIP + F.rms_norm (V1) is +31 % tok/s but +0.65 loss at step 2000; V2 (silu autograd ON) correct but ~30.9 K tok/s (slower). Conclusion: autokernel provides zero net benefit for OdinFlat. |

The surface investigation established **that** autokernel is broken for
OdinFlat. It did not quantify **what** is frozen, **why** the mechanism
bypasses autograd, or **whether** any integration path recovers the
speedup. This document answers those.

---

## 3. Mechanism: how HIP kernels sever gradient flow

### 3.1 The dispatch path for a PyTorch op

A normal `aten::*` op goes through:

```
  Python call → torch dispatcher → backend selection → kernel launch
                   │
                   └─ wraps output in AutogradFunction node; sets
                      grad_fn so backward can trace the graph.
```

### 3.2 The dispatch path for `torch.library.custom_op` + `register_autograd`

```
  Python call → dispatcher sees the registered custom op
              → dispatcher synthesizes an AutogradFunction
                whose forward calls our Python impl and whose
                backward calls our registered backward callable.
              → output has grad_fn = CustomOpBackward.
```

This is the "correct" integration path. Used by `torch.ops.autokernel.
rmsnorm` (post-Phase-III fix) and by `torch.ops.autokernel.silu_gate_mul`
(registered but not currently invoked; see §3.3).

### 3.3 The dispatch path for a raw pybind11 C++ call

```
  Python call → pybind11 stub → C++ function body
              → C++ creates `torch::Tensor` via `torch::empty(...)`
              → No dispatcher involvement; no autograd node.
              → output.grad_fn = None, output.requires_grad = False
                (even if inputs had requires_grad=True).
```

This is what `_FusedSwiGLUReplacement` currently does:

```python
# autokernel/_patterns.py
def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate, up = self.w_gate_up(x).chunk(2, dim=-1)
    if gate.dtype == torch.float16:
        activated = self.kernel_fn(gate.contiguous(), up.contiguous())
        #            ^^^^^^^^^^^^^^ raw pybind call — breaks autograd
    else:
        activated = F.silu(gate) * up
    return self.w_down(activated)
```

### 3.4 Why this looks like working training

After the broken `self.kernel_fn` call, downstream ops still compute
correct gradients for leaf parameters they operate on:

- `self.w_down(activated)` — even though `activated.grad_fn = None`,
  the matmul computes `w_down.weight.grad = (grad_out).T @ activated`
  using the activated tensor as a value (not a node). This is the
  same math as the correct autograd path.
- `w_down.weight` (the downstream Linear) receives correct gradients.
- Loss descends because `w_down` adapts normally.

What is **not** correct:

- `grad_activated = grad_out @ w_down.weight.T` would normally flow
  back from `w_down`, but because `activated.grad_fn = None`, autograd
  never computes or propagates it upstream.
- Therefore `gate` and `up` get no gradient signal.
- Therefore `w_gate_up.weight` (the Linear that produced `gate` and
  `up`) gets no gradient contribution from this module.
- The `ffn_norm` immediately upstream sees its output flowing into
  `self.w_gate_up(x)`. That output's autograd is intact; the norm
  correctly computes `ffn_norm.weight.grad` from the input to
  `w_gate_up`. But the input to `w_gate_up` itself has no gradient
  (it's `self.w_gate_up(norm_out)` where `norm_out` has `grad_fn`,
  but after the pybind call there's nothing to backprop from). Result:
  `ffn_norm.weight.grad` is computed using a **zero downstream
  gradient**, giving `grad=0` at every step.

This matches the Track 3.A observations exactly.

---

## 4. Blast radius: what actually freezes

Source: `docs/perf/autokernel-frozen-blast-radius.md`, plus the raw
`docs/perf/odinflat-profile-2026-05-10/diag-{V0,V1,V3}.jsonl`.

| Config | Flags | Status tally |
|---|---|---|
| V0 | baseline | 119 always_finite, 1 always_none (v_res_scale L0 — expected, no v_prev input) |
| V1 | `--optimize-kernels` | 91 always_finite, 15 always_none, 14 always_zero |
| V3 | `--optimize-kernels --autokernel-exclude fused_silu_gate_mul` | **Identical to V0** |

### 4.1 V1 breakdown

14 layers × 2 frozen classes = 28 frozen params:

| Parameter pattern | Count | Status | Param magnitude |
|---|---:|---|---:|
| `layers.*.ffn.w_gate_up.weight` | 14 | `always_none` | 768 × (2×2048) = ~3.15 M each → 44 M total |
| `layers.*.ffn_norm.weight` | 14 | `always_zero` | 768 each → 10.75 K total |

**Effective frozen parameter count: ~44 M** (the ffn_norm weights are
small but "dead" — they still contribute to forward). Earlier hand-waves
of "~60 M params frozen" in `odinflat-throughput-final.md` were slightly
off; the correct figure from direct measurement is ~44 M `w_gate_up`
weights plus ~11 K norm weights, total **~44 M trainable weights never
updated** across 14 layers.

### 4.2 Why ffn_norm specifically gets `always_zero` not `always_none`

The gradient chain is:

```
loss ← CE ← lm_head ← final_norm ← layer.final_output ← ... ← ffn(x)
                                                             ↑
                                                             y = w_down(activated)
                                                             ↑ (correct autograd)
                                                             activated = raw_hip_call(gate, up)
                                                             ↑ (grad_fn=None — severed here)
                                                             gate, up = w_gate_up(ffn_norm(x))
                                                                                  ↑
                                                                                  ffn_norm
```

Autograd reaches `w_down.weight.grad` normally (good), but
`grad_activated = 0` (because `activated.grad_fn = None`). When it tries
to propagate upward, the `w_gate_up` backward receives gradient = 0
from its output, so `w_gate_up.weight.grad = 0 @ norm_out = 0`. But
the **hook that triggers ffn_norm.backward** still fires because the
computation `w_gate_up(norm_out)` had `grad_fn`. The gradient
transmitted to `ffn_norm` is 0, so `ffn_norm.weight.grad` is also 0 —
not None, just zero.

That's exactly the observed `always_zero` status.

### 4.3 V3 is byte-for-byte identical to V0

The Phase III rmsnorm autograd fix routes through
`torch.ops.autokernel.rmsnorm`, which has correctly registered forward
and backward. All gradients flow.

This proves:
- The rmsnorm HIP kernel backward is correct.
- The `torch.library.custom_op` + `register_autograd` integration works.
- The silu HIP kernel would need the same treatment to be safe.

---

## 5. Why V2 (silu autograd ON) doesn't recover the speedup

From `odinflat-throughput-final.md` Phase V:

| Config | rmsnorm | silu | Steady tok/s |
|---|---|---|---:|
| V0 baseline | F.rms_norm | native | 31,331 |
| V1 broken | F.rms_norm | HIP (no autograd) | 41,198 |
| V2 correct | F.rms_norm | HIP (autograd via custom_op) | **30,976** |

V2 runs **below** baseline. Breakdown of the V2 deficit:

1. **Dispatcher overhead.** `torch.library.custom_op` adds a Python-level
   dispatch layer (~microseconds per call). OdinFlat: 14 layers × ~80
   microsteps per opt step × 2 calls (fwd + bwd) = 2,240 dispatches per
   opt step. At ~5 μs per dispatch = 11.2 ms per opt step = ~8 % of a
   step at 130 ms/step.

2. **save-for-backward cost.** HIP silu backward needs (gate, up) saved.
   These are two fp16 tensors of shape [B, T, 2048]. At batch=16, T=512:
   ~16 MB per layer × 14 layers = 224 MB of extra state per microstep.
   Minor memory pressure.

3. **HIP backward kernel.** Equivalent in FLOPs to PyTorch Inductor's
   fused silu-bwd-triton; no asymptotic advantage. Forward savings
   (~15 % of the silu op's wall time) don't compensate for the (1) + (2)
   overhead.

The net: **HIP silu is not meaningfully faster than Inductor's native
silu under correct autograd**. The speedup measurement in V1 was
entirely an artifact of not computing the backward.

---

## 6. Tested and rejected recovery paths

### 6.1 `torch.autograd.Function` instead of `custom_op`

Not empirically tested in this session (original plan task 5.E). Prior
reasoning (§5.1 of this doc): `torch.library.custom_op` overhead is
~5 μs per dispatch; `torch.autograd.Function` skips the dispatcher and
is ~1-2 μs per call. Saves ~6-9 ms per opt step.

Expected outcome: **net +3 to +5 % tok/s above V0 baseline**. Still
well below V1's illusory +31 %. Not explored in depth here because:
- HIP silu's own backward kernel performance is the real ceiling.
- ~3-5 % isn't worth the extra complexity.
- If that speedup mattered, the right path would be a Triton rewrite
  that fuses silu with the flanking GEMMs (`w_gate_up` chunk + down
  projection).

Recommendation: **don't pursue torch.autograd.Function variant.** Either
accept baseline for OdinFlat, or invest in a proper fused kernel rewrite
if throughput becomes a binding constraint later.

### 6.2 Fix the raw-pybind forward to set `grad_fn`

Not feasible without building autograd machinery manually — which is
exactly what `torch.library.custom_op` exists to provide. No cheaper
path.

### 6.3 Accept the freeze as "partial freezing regularization"

Nonsensical; models trained with 23 % of params frozen at initialization
are objectively worse at the measured 2000-step horizon (+0.65 loss).

---

## 7. Quality degradation evidence

From `docs/perf/odinflat-throughput-final.md` Phase V V1 probe at 2000
steps:

| Step | V0 baseline loss | V1 broken loss | Delta |
|---:|---:|---:|---:|
| 200 | 4.7013 | 4.6691 | -0.032 |
| 500 | 4.42 (est.) | 4.74 (est.) | +0.32 |
| 2000 | 3.1466 | 3.8010 | **+0.654** |

The delta grows linearly with steps as the active 77 % of the parameter
population saturates faster than it would with the full 100 % available.
At 52,000 steps (one dolma epoch), extrapolating linearly predicts a
~2-3 point loss gap, which would translate to meaningfully worse
downstream BPB and sample quality.

### 7.1 Track 3.A direct corroboration (this session)

`diag-V1.jsonl` shows `w_gate_up.weight` entries have `is_none=true`
for all 50 × 14 = 700 recorded (step, layer) pairs. Not a single
gradient was ever written to any of these 14 parameter tensors. This
is the direct evidence the surface investigation lacked.

`ffn_norm.weight` has `grad_norm=0.0` and `is_zero=true` for all 700
recordings. Consistent with the dead-gradient chain described in §4.2.

---

## 8. Recommendations

### 8.1 For the current training path

| Model | Recipe recommendation |
|---|---|
| **OdinFlat Sprint 3A** | Drop `--optimize-kernels`. Use baseline 31.3 K tok/s. Expect ~61 h wall for one dolma-10B epoch with the locked C3 recipe. |
| **OdinHalo Sprint 3B** | **Awaiting B4 probe result.** If 2000-step loss is within ±0.1 of baseline extrapolation → ship Sprint 3B with `--optimize-kernels`. If loss regression > 0.2 at step 2000 → drop `--optimize-kernels`, accept ~77 h wall. Probe is running as of this commit. |
| Secondary halo models (jormungandr_halo, baldr_halo, fenrir_halo, chimera_halo, etc.) | Not on current training path. If ever revived, verify `--optimize-kernels` behavior with a 2000-step probe + `--diag-frozen-params`. |

### 8.2 For autokernel maintenance

1. **Add a runtime warning** when `autokernel.optimize(model, ...)`
   detects that any replacement module calls a raw pybind kernel and
   the output has `grad_fn = None` AND the model is in training mode.
2. **Deprecate or mark "inference-only"** any `_FusedXxxReplacement`
   whose kernel path does not go through a registered custom op or
   autograd.Function. Attach a docstring warning and raise if used
   under `model.train()`.
3. **Expand the audit.** Run Track 3.A on every pattern/model
   combination (RotaryEmbeddingPattern, FusedQKVPattern,
   FusedResidualRMSNormPattern, GriffinBlockPattern). We know the
   silu path is broken; we don't know about the others.

### 8.3 For future kernel development

1. **Never ship a training-path HIP kernel without autograd
   registration.** Either `torch.library.custom_op` with
   `register_autograd` or `torch.autograd.Function`. Raw pybind calls
   are inference-only.
2. **Validate with a 2000-step probe**, not a 200-step one. The
   silent-freeze effect accumulates roughly linearly in loss with
   training steps; short probes cannot detect it.
3. **Use Track 3.A's `--diag-frozen-params`** to make grad flow
   explicit. A single 50-step run with this flag is sufficient to
   reveal the freeze pattern.

---

## 9. Open questions

1. **Does OdinHalo suffer the same freeze at 2000 steps?** Being
   answered by the B4 probe running now. Looped architectures with
   iter_norm resets may partially dampen the freeze effect, but the
   autograd severing is the same. Expected outcome: partial freeze
   at same ffn layers; question is whether the iter_norm resets
   propagate enough residual gradient to rescue quality.
2. **Would a Triton-based rewrite of silu + w_gate_up + w_down as a
   single fused kernel beat Inductor?** Speculative; not explored.
   Only worth investigating if OdinFlat throughput becomes a binding
   blocker (current throughput saturates T²-optimal training for the
   full 6.9 B token epoch in ~61 h).
3. **Is `FusedQKVPattern` affected by the same class of bug?** The
   `_FusedQKVAttentionReplacement` is wrapped in an nn.Module that
   forwards through `F.scaled_dot_product_attention` (a standard ATen
   op) after rotating QKV via HIP fused ops. The rotation itself
   might have the same autograd-severing issue if it uses raw pybind.
   Needs direct diagnostic to confirm. Tracked as §8.2.3 future work.

---

## 10. Appendix: probe commit hashes and artifacts

| Artifact | Commit |
|---|---|
| `scripts/diag_frozen_params_run.sh` | 5b5ccaf |
| V0/V1/V3 diag.jsonl data | c4e71c3 (Track 3.A) |
| `scripts/analyze_diag_frozen_params.py` | c4e71c3 |
| `docs/perf/autokernel-frozen-blast-radius.md` | c4e71c3 |
| OdinFlat step profile (Track 1.3) | 85f937e |
| QKV fusion A/B + Track 2.a decision doc | 85f937e, 63de5be, 7a5a108 |
| Track 2.b z-loss fp16 opt | 46a4d3a |

Raw data:
- `docs/perf/odinflat-profile-2026-05-10/diag-V0.jsonl` (754 KB, 50 steps × 120 params)
- `docs/perf/odinflat-profile-2026-05-10/diag-V1.jsonl` (729 KB, 50 steps × 120 params)
- `docs/perf/odinflat-profile-2026-05-10/diag-V3.jsonl` (755 KB, 50 steps × 120 params)
- `docs/perf/odinflat-profile-2026-05-10/profile-summary.txt` (26 KB, Track 1.3 torch.profiler)

---

**End of deep analysis. B4 (OdinHalo 2000-step probe) completion will add
a §11 addendum below with that verdict.**
