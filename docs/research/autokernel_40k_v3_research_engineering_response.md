# AutoKernel Halo Strix 40k Throughput: v3 Research Response and Engineering Handoff

**Date:** 2026-05-10 / 2026-05-11 context  
**Repo context:** `joelhenwang/autokernel-halo-strix`, v3 addendum referencing SHA `664fd20`  
**Primary objective:** help the engineering agent decide what to do next to increase validated pretraining throughput toward the nominal **40k aggregate tok/s** target without repeating the earlier fake-speed failure mode where kernels accelerated training by silently freezing important parameters.

---

## 0. Executive conclusion

The v3 addendum changes the priority order.

The strongest near-term plan is **not** to spend the next block of time trying to make compiled autograd the main 40k unlock. The T-0.1 backward breakdown shows the backward path is already close to the expected dense-transformer cost ratio: `CompiledFunctionBackward` is large, but the measured backward/forward ratio is **2.18×**, close to the theoretical **2×** dense-matmul backward cost. That means the large backward bucket is mostly real work, not obvious graph-fragmentation waste. Compiled autograd remains useful infrastructure, but its expected throughput value should be demoted to a short gated smoke unless the smoke proves a real net DDP gain.

The new near-term leading candidates are:

1. **Branchless SPECTRA / hot-path sync removal.**  
   T-0.3 found **870 `aten::item` calls across 10 optimizer steps**, with `sigma1.item()` in SPECTRA accounting for about **50 syncs/step**. This is a concrete, low-risk patch.

2. **Fused z-loss / chunked CE+zloss validation on the native hidden path.**  
   This is still one of the safest throughput levers because it does not alter hidden-layer representation learning. The profile says the separate z-loss pass is costly, but the expected net win should be treated as **3–8%**, not guaranteed 16.7%.

3. **Batch=32 / accum=4 probe.**  
   If it fits without unacceptable recompute overhead, it is a clean utilization/microstep-overhead win and should be tested earlier than compiled autograd.

4. **NorMuon implementation cleanup and telemetry.**  
   NorMuon remains both a throughput target and a stability-risk amplifier. Per-parameter update telemetry should land before hidden-kernel recovery.

5. **Tiered custom-op autocast/dtype audit.**  
   Graph-break fixes should narrow to the two unfixed ops, but dtype/autocast sanity checks should touch all training-path custom ops. A Phase-B op can compile cleanly and still have an AMP/dtype mismatch that matters for fp16 stability.

6. **Compiled autograd smoke only.**  
   Run a short smoke after prerequisite graph-break cleanup and the DDP/no-sync baseline are known. Continue only if it produces at least **2.5–3% net steady-state DDP tok/s** without recompile churn or DDP-overlap regression.

7. **Hidden-kernel recovery only after the above.**  
   The old 40k-class result is not a valid target by itself because it was entangled with silent frozen parameters. Hidden kernels should return through delayed enable, `w_gate_up` staging, post-NorMuon trust caps, dtype parity, and a 2000-step stability gate.

The corrected target framing should be:

```text
Nominal target:        40k aggregate tok/s across 2× Strix Halo systems
Engineering success:   ≥36k stable aggregate tok/s
Strong success:        ≥38k stable aggregate tok/s
Stretch outcome:       40k stable aggregate tok/s
Non-negotiable:        no new frozen params, no GradScaler collapse, clean 2000-step loss parity
```

Treating **40k as the only pass/fail line** would create the wrong incentive: it would push the team back toward the kind of speedup that already looked good on tok/s and bad on learning.

---

## 1. Inputs reviewed

This response is based on:

- The uploaded v3 addendum: `autokernel-40k-engineering-v3-addendum.md`.
- The repo’s current `STATUS.md`, which documents the original `--optimize-kernels` frozen-parameter failure, Phase-B autograd fixes, Phase-C/G divergence, and locked decision to drop blanket `--optimize-kernels` for Sprint 3A/3B.[^status]
- T-0.1 backward breakdown: backward is close to theoretical dense-transformer cost; `CompiledFunctionBackward` is not obviously wasteful.[^backward]
- T-0.4 graph-break inventory: only two distinct graph-break locations and one steady-state-irrelevant recompile were found.[^graph]
- T-0.3 sync-point audit: SPECTRA `sigma1.item()` is the largest hot-path sync offender.[^sync]
- PyTorch compiled autograd documentation: compiled autograd can capture a larger backward graph, but it adds runtime overhead, is more prone to recompiles, and is still under active development.[^compiled_autograd]
- PyTorch `torch.library` documentation: `custom_op`, `triton_op`, `register_autocast`, `register_autograd`, and their compiler/autocast semantics.[^torch_library]
- PyTorch custom-op tutorial: training custom ops should use `register_autograd`, and `opcheck` tests registration but not mathematical gradient correctness.[^custom_ops]
- PyTorch DDP docs and notes: `no_sync`, bucket sizing, `gradient_as_bucket_view`, DDP overlap, and the risk that whole-backward compilation can interfere with allreduce overlap.[^ddp_api][^ddp_notes]
- PyTorch AMP examples and recipe: unscale before clipping/inspection, unscale only after full accumulation, and avoid `.item()`-style CPU-GPU synchronization.[^amp_examples][^amp_recipe]
- PyTorch HIP docs: ROCm/HIP intentionally reuses CUDA interfaces, so `device_type="cuda"` remains the expected PyTorch API surface on HIP.[^hip]
- AMD/ROCm Strix Halo/RDNA3.5 docs: Ryzen AI Max+ 395 uses 256-bit LPDDR5x-8000 and RDNA3.5 APU memory is physically shared/GPUVM-mapped rather than discrete HBM-like VRAM.[^amd_395][^rdna35]

---

## 2. Why the v3 data changes the plan

### 2.1 The backward bucket is not the big free lunch

The earlier v2 plan ranked compiled autograd high because the profile exposed a large `CompiledFunctionBackward` bucket. That was a reasonable first hypothesis. T-0.1 makes the picture clearer.

Key data from T-0.1:

```text
CompiledFunction forward:   6.523 s over 10 opt steps
CompiledFunctionBackward:  14.213 s over 10 opt steps
Backward/forward ratio:       2.18×
Theoretical dense ratio:      2.00×
```

Dense transformer backward naturally needs roughly two matmuls per forward matmul: one input-gradient matmul and one weight-gradient matmul. A 2.18× observed wall ratio does not look like a fragmented 5× backward path. It looks like a mostly efficient backward path with limited overhead.

T-0.1 also reports:

```text
Backward-specific rocBLAS transposed GEMMs: 17.49% of step wall
Inductor fused SwiGLU backward:              2.92% of step wall
LogsumexpBackward:                           5.6% of step wall
CausalConv1dFnBackward:                      4.54% of step wall
MmBackward0, likely NorMuon NS:              5.2% of step wall
```

The key implication: **compiled autograd cannot remove rocBLAS GEMM work**, cannot magically eliminate z-loss unless the loss path is rewritten, and cannot trace through an external C++ extension unless it has compatible wrapper semantics. It may still reduce wrapper/dispatch fragmentation and possibly pull some non-compiled backward regions into a larger graph, but the budget is smaller than initially hoped.

### 2.2 The graph-break surface is smaller than expected

T-0.4 found only two distinct graph-break locations:

| Break | Location | Cause | Direct speed value | Main value |
|---|---|---|---:|---|
| #1 | `fused_rope_gate_mul.kernel_fn` | wrapper uses `@torch.compiler.disable` | probably 0–1% | compiled-autograd prerequisite, Phase-B consistency |
| #2 | DaoAILab `causal_conv1d_fn` | older custom-op semantics / trace barrier | probably 0–1% | compiled-autograd prerequisite, dtype/autocast control |

The important conclusion is not “ignore graph breaks.” The conclusion is: **do not expect graph-break cleanup alone to recover large throughput.** It is infrastructure, not a direct 40k lever.

T-0.4 also weakens the old hypothesis that HyPE/depth-KV list mutation is causing repeated recompile storms. There was one recompile across 10 optimizer steps, caused by `v_prev is None` changing to tensor on step 2. That is not a steady-state throughput issue.

### 2.3 SPECTRA syncs are concrete and patchable

T-0.3 is the highest-EV immediate patch because it found a real hot-path pattern:

```text
870 aten::item calls / 10 opt steps = 87 syncs per optimizer step
SPECTRA sigma1.item(): ~50 syncs per optimizer step
```

The proposed branchless SPECTRA rewrite removes the CPU scalar branch and keeps the decision tensor-side. This is exactly the sort of issue PyTorch’s AMP/performance guidance warns about: avoid excessive CPU-GPU synchronization such as `.item()` calls when chasing GPU throughput.

This is a better near-term bet than compiled autograd because:

- It is simple.
- It is localized.
- It should not change learning semantics beyond tiny floating-point differences.
- It has a clear before/after measurement: `.item()` count, CPU wall, CUDA wall, update parity, and tok/s.

---

## 3. Direct answer to v3 Question 1: compiled autograd expected value

### 3.1 Revised expected value

Given T-0.1, I would revise compiled-autograd upside to:

```text
Expected steady-state step-wall gain: 1–4%
Good outcome:                        4–6%
Exceptional outcome:                 6–8%
>8% outcome:                         unlikely on current evidence
```

The T-0.1 document itself still lists a broader upper bound, but the stronger interpretation is that the upper bound is not a planning assumption. The observed backward/forward ratio being close to 2× means the large backward bucket is mostly unavoidable matmul work. Compiled autograd can optimize around the edges; it probably cannot reclassify the run from 31k to 40k.

### 3.2 Is there a mechanism for recovering more than ~8%?

Possible, but narrow. Compiled autograd could exceed 8% only if one or more of these second-order mechanisms appears in the actual smoke:

1. **It pulls non-compiled NorMuon-related backward/NS work into a more efficient compiled region.**  
   T-0.1 reports `MmBackward0` at about 5.2% of step wall, likely related to NorMuon NS matmuls. If compiled-autograd or a compiled-optimizer path materially improves this, the gain could exceed the pure wrapper-overhead estimate. But this is not guaranteed and may belong under NorMuon implementation work rather than compiled autograd.

2. **It enables fusion across currently separate backward boundaries.**  
   PyTorch says compiled autograd can capture a larger backward graph than AOTAutograd, including cases where forward graph breaks would otherwise become backward graph breaks. That is a real mechanism. But T-0.4 says the forward graph-break surface is tiny, so the amount of recoverable fragmentation is probably modest.

3. **It reduces Python/autograd-engine scheduling overhead in a way the profile does not expose cleanly.**  
   This is possible in principle, but T-0.1 shows `autograd::engine::evaluate_function` overhead itself is tiny. So this cannot be the main route to 40k.

4. **It improves DDP hook timing without harming overlap.**  
   This is speculative. PyTorch’s DDP notes explicitly warn that whole-forward/whole-backward compilation can prevent allreduce overlap because hooks fire only after the optimized backward finishes unless DDPOptimizer breaks graphs at bucket boundaries. Therefore compiled autograd could also regress end-to-end DDP throughput even if local backward kernels look faster.

The practical answer: **yes, there are mechanisms, but none are strong enough to justify a 5–6 day compiled-autograd push before lower-risk patches unless a smoke test proves real net gain.**

### 3.3 Should T-2.2 move after NorMuon implementation optimization?

Yes. Full compiled-autograd engineering should move **after**:

```text
1. Branchless SPECTRA / sync cleanup
2. Fused-zloss validation on native hidden path
3. Batch=32/accum=4 feasibility probe
4. NorMuon telemetry and obvious implementation cleanup
5. DDP no_sync/bucket baseline verification
```

But do not delete compiled autograd. Replace the 5–6 day block with this gated plan:

```text
CA-0 prerequisite cleanup, 0.5–1 day:
  - Fix fused_rope_gate_mul wrapper or add compile-friendly custom_op route.
  - Wrap DaoAILab causal_conv1d through a torch.library custom_op shim if feasible.
  - Add register_fake/register_autograd/register_autocast as needed.

CA-1 smoke, 0.5–1 day:
  - Native baseline + compiled_autograd=True.
  - Native + fused zloss + compiled_autograd=True.
  - DDP with best known no_sync/bucket config.
  - Collect recompile count, unique backward graphs, DDP overlap, steady tok/s, loss/scaler health.

Continue only if:
  - ≥2.5–3% net steady-state DDP tok/s improvement, and
  - no recompile churn, and
  - no loss/scaler regression, and
  - no DDP allreduce-overlap regression.
```

### 3.4 Do not confuse infrastructure value with immediate throughput value

Compiled autograd may be worth having long-term. It can make future custom-op and backward-path optimization easier. But the near-term question is not “is it good infrastructure?” The question is: **what is the fastest safe path to validated throughput?** T-0.1 says compiled autograd is no longer the highest-EV first move.

---

## 4. Direct answer to v3 Question 2: autocast scope

### 4.1 Split graph-break work from dtype/stability work

The v3 addendum asks whether autocast/fp32-island testing should narrow to the two unfixed HIP ops because T-0.4 found only those two graph breaks.

My answer:

```text
Narrow compiler/graph-break fixes to the two unfixed ops.
Do not narrow dtype/autocast auditing only to the two unfixed ops.
```

These are different problems.

A graph break is a compiler-visibility issue. An autocast mismatch is a numerical/stability issue. A custom op can compile cleanly and still run at the wrong dtype boundary under AMP.

PyTorch’s `register_autocast` exists for exactly this reason: it defines how a custom op behaves inside an autocast-enabled region. Its documented behavior is to cast incoming floating-point tensors to the target dtype and then run the custom op with autocast disabled. On ROCm/HIP, PyTorch intentionally reuses CUDA interfaces, so `device_type="cuda"` is still the right logical device type in standard PyTorch APIs.

### 4.2 Recommended tiered scope

#### Tier 0: cheap dtype/autocast inventory across all training-path custom ops

Run this across all seven relevant ops:

```text
silu_gate_mul
rmsnorm
fused_res_rmsnorm
rotary_emb_fp32
fused_ple_gate
fused_rope_gate_mul
causal_conv1d shim/wrapper
```

For each op, record:

```text
op_name
has_register_fake
has_register_autograd
has_register_autocast
forward_input_dtypes under torch.autocast("cuda", dtype=torch.float16)
forward_output_dtype
backward_input_dtype
backward_output_dtypes
internal_accumulation dtype, if known
whether op is a graph-break source
whether op is used in --optimize-kernels training path
```

This is not a full parity matrix. It is a cheap inventory and should not block T-0.2/T-0.5.

#### Tier 1: fix the two compiler-boundary ops

```text
1. fused_rope_gate_mul.kernel_fn
2. DaoAILab causal_conv1d_fn wrapper/shim
```

For both:

```text
- remove @torch.compiler.disable where feasible
- wrap through torch.library.custom_op
- add register_fake
- add register_autograd if training path requires gradients
- add register_autocast("namespace::op", "cuda", torch.float16)
- add opcheck
- add fwd/bwd parity tests against native/reference path
```

This is a compiled-autograd prerequisite more than a direct throughput patch.

#### Tier 2: deep parity on stability-dangerous ops

Run deeper parity on:

```text
silu_gate_mul
rmsnorm
fused_res_rmsnorm
fused_rope_gate_mul
causal_conv1d wrapper
```

Why include `silu_gate_mul` even though it compiles cleanly? Because the original 40k-class result was directly entangled with the SwiGLU/FFN path and frozen `w_gate_up`. A clean graph does not guarantee correct dtype behavior, correct fp16 headroom, or a NorMuon-safe gradient distribution.

### 4.3 What to test in Tier 2

For each Tier-2 op, compare:

```text
native eager reference
native compiled / Inductor reference
custom op without explicit register_autocast
custom op with register_autocast("cuda", torch.float16)
custom op with fp32 internal reductions or fp32 sensitive math, where applicable
```

Metrics:

```text
forward max_abs_err / rel_err
backward max_abs_err / rel_err
gradient cosine
post-NorMuon update cosine
maxabs activation drift by layer
GradScaler scale over replayed batch sequence
loss over 200/500-step smoke if op is enabled in training path
```

Precommit interpretations:

| Observation | Likely meaning | Next action |
|---|---|---|
| Graph breaks fixed but divergence unchanged | Compiler visibility was not the root cause | Continue dtype/NorMuon work |
| register_autocast changes training stability | dtype boundary mismatch was real | Keep autocast rule and add parity gates |
| fp32 internal math stabilizes op | fp16 internal math was too aggressive | Keep fp32 island if speed cost is acceptable |
| fwd parity good but update cosine poor | NorMuon amplifies small backward differences | Trust cap / update telemetry / staging |
| custom op clean but native still faster | compiler opacity or dispatch overhead remains | avoid that op as a throughput lever |

### 4.4 Minimal register_autocast patch pattern

Use this as a conceptual pattern. Exact names need to match repo namespaces.

```python
import torch

# After defining/registering the custom op and its autograd formula:
torch.library.register_autocast(
    "autokernel::silu_gate_mul",
    "cuda",               # PyTorch HIP builds still use cuda logical device APIs
    torch.float16,
)
```

For ops with fp32-sensitive internal math, do not blindly force all math to fp16. Use explicit internal casts where needed:

```python
def reference_sensitive_math(x: torch.Tensor) -> torch.Tensor:
    # conceptual example
    x32 = x.float()
    y32 = stable_reduction_or_exp_like_math(x32)
    return y32.to(x.dtype)
```

The test should decide whether this fp32 island is worth the speed cost.

---

## 5. Direct answer to v3 Question 3: 40k realism

### 5.1 Target math

Validated baseline:

```text
baseline aggregate throughput ≈ 31,331 tok/s
nominal target               = 40,000 tok/s
required throughput lift      = 40,000 / 31,331 = 1.277× ≈ +27.7%
```

With the known effective step size:

```text
tokens_per_optimizer_step = 16 batch × 512 block × 8 accum × 2 ranks
                          = 131,072 tokens

baseline_step_time = 131,072 / 31,331
                   ≈ 4.18 s

target_step_time = 131,072 / 40,000
                 ≈ 3.28 s

needed_step_saving ≈ 0.91 s / optimizer step
needed_step_wall_reduction ≈ 21.7–21.8%
```

For intermediate targets:

| Throughput | Throughput lift vs 31,331 | Step time | Step-wall reduction vs baseline |
|---:|---:|---:|---:|
| 35k | +11.7% | 3.745 s | ~10.0% |
| 36k | +14.9% | 3.641 s | ~12.9% |
| 37k | +18.1% | 3.542 s | ~15.3% |
| 38k | +21.3% | 3.449 s | ~17.6% |
| 40k | +27.7% | 3.277 s | ~21.8% |

This is why 40k is hard. A +27.7% throughput gain is only possible if several optimizations land and their wall-time savings do not heavily overlap.

### 5.2 Honest estimate

I agree with the v3 addendum’s revised calibration:

```text
35–36k: realistic near-term if branchless SPECTRA + fused loss + small DDP/sync wins land.
36–37k: good engineering outcome and a reasonable Sprint success criterion.
37–38.5k: strong outcome; likely requires batch=32/accum=4 or NorMuon implementation work.
38.5–40k: stretch; requires most levers to hit upper-bound estimates with low overlap.
40k+: possible, but not something to promise from current evidence.
```

My probability-style framing:

| Outcome | Current plausibility | Conditions |
|---|---:|---|
| 35k stable | high | branchless sync cleanup + some loss/CE gain |
| 36k stable | medium-high | loss path and SPECTRA both land cleanly; no DDP regression |
| 38k stable | medium | batch=32 or NorMuon cleanup adds a real 3–5% |
| 40k stable | low-to-medium, about 10–20% | fused loss + sync + batch + NorMuon + DDP + maybe compiled autograd all land near top end |
| 40k via hidden kernels alone | low | current evidence says correct-training SwiGLU is near ceiling and old speed was partly fake |

### 5.3 Recommended STATUS.md language

Use target language that keeps ambition but avoids bad incentives:

```markdown
### AutoKernel 40k throughput effort

Nominal target: 40k aggregate tok/s across 2× Strix Halo systems.

Engineering success criterion: ≥36k aggregate tok/s with clean 2000-step loss parity,
no newly frozen params, no GradScaler collapse, no nonfinite update events, and no
post-NorMuon update-ratio pathology.

Strong success: ≥38k aggregate tok/s under the same stability and learning gates.

Stretch: 40k aggregate tok/s. Do not treat failure to reach 40k as failure if ≥36k
is stable and learning-equivalent. Speedups from frozen params, severed gradients,
or unstable fp16 behavior do not count.

Blanket --optimize-kernels remains disabled for production training until granular
kernel flags pass per-feature throughput, gradient-flow, dtype, and 2000-step stability gates.
```

---

## 6. Updated hypothesis ranking

| ID | Hypothesis | Current rank | Reason | Next experiment |
|---|---|---:|---|---|
| H1 | SPECTRA hot-path syncs cost real step wall | Very high | T-0.3 found ~50 syncs/step from `sigma1.item()` | Branchless SPECTRA patch |
| H2 | Fused z-loss/chunked CE can save safe loss-path overhead | High | z-loss path is expensive and does not alter hidden representations | Native hidden path + fused zloss |
| H3 | Batch=32/accum=4 improves utilization / reduces microstep overhead | High | Reported +5% in prior sweep, if memory fits | batch32 probe early |
| H4 | NorMuon implementation overhead is addressable | High | `MmBackward0`/NS and optimizer path are visible costs | telemetry then implementation cleanup |
| H5 | NorMuon amplifies small custom-op numerical differences | High for stability | Phase C/G divergence pattern fits update-scale/fp16 edge | per-param update telemetry |
| H6 | Custom-op autocast/dtype mismatch drives Phase C/G divergence | High for stability | Phase-B ops may compile but lack explicit autocast rules | tiered dtype audit |
| H7 | DDP no_sync/bucket tuning can save 1–3%, huge only if broken | Medium-high | DDP cost appears bounded, but accumulation mistakes are expensive | no_sync/bucket trace |
| H8 | Compiled autograd saves meaningful backward overhead | Medium-low | T-0.1 shows near-theoretical backward ratio | short gated smoke only |
| H9 | Two graph-break fixes unlock direct throughput | Low direct, medium infra | T-0.4 says only two breaks; runtime cost likely small | fix only as infra/prereq |
| H10 | Hidden SwiGLU kernels can recover 40k safely | Low-to-medium | isolated HIP/Triton already near ceiling; old 40k had frozen params | delayed enable + trust cap later |
| H11 | Fused zloss denominator/masking bug is root divergence cause | Low | Phase C also diverged without fused zloss | parity test but not main bet |
| H12 | HyPE/depth-KV mutation causes recompile storm | Drop | T-0.4 found one one-time recompile only | no further work unless new evidence |
| H13 | Optimizer-state mismatch after native→optimized switch | Medium | directly testable and could explain delayed divergence | warm-start preserved-vs-fresh |
| H14 | DDP overlap regresses under compiled autograd | Medium risk | PyTorch DDP notes warn whole-backward compilation can harm overlap | collect overlap in CA smoke |

---

## 7. Revised priority order

### P0 — unblock measurement and safe feature isolation

These should proceed immediately.

```text
P0.1 Implement granular --ak-* flags.
P0.2 Implement NorMuon per-parameter telemetry.
P0.3 Add dtype/autocast trace fields.
P0.4 Add DDP communication counters / no_sync verification.
P0.5 Add item/sync counter in profiler summary.
```

### P1 — low-risk throughput patches

```text
P1.1 Branchless SPECTRA, remove sigma1.item().
P1.2 Fused z-loss / chunked CE+zloss validation on native hidden path.
P1.3 Remove or defer hot-path .item() calls not needed for control flow.
P1.4 Verify DDP no_sync and run bucket settings sweep.
```

### P2 — batch and optimizer path

```text
P2.1 batch=32/accum=4 probe.
P2.2 NorMuon implementation cleanup after telemetry.
P2.3 Group/batch NS work where shapes permit.
P2.4 Avoid CPU scalar reads inside optimizer and diagnostics.
```

### P3 — stability-critical dtype work

```text
P3.1 Tier-0 dtype/autocast inventory across all training-path custom ops.
P3.2 Fix fused_rope_gate_mul and causal_conv1d graph-break/custom-op wrappers.
P3.3 Deep parity on silu_gate_mul, RMSNorm/residual RMSNorm, rope_gate_mul, causal_conv1d.
```

### P4 — conditional compiled autograd

```text
P4.1 Run short compiled-autograd smoke.
P4.2 Continue only if ≥2.5–3% net DDP tok/s gain and no overlap/recompile regression.
```

### P5 — hidden-kernel recovery

```text
P5.1 delayed enable native → optimized
P5.2 w_gate_up update-scale staging
P5.3 post-NorMuon trust cap
P5.4 per-kernel replay harness
P5.5 2000-step stability gate
```

---

## 8. Required flags

Keep the v2 granular flag taxonomy, but add the v3-specific controls.

### 8.1 Core granular kernel/runtime flags

```text
--ak-loss-ce
--ak-loss-zloss
--ak-swiglu-fwd
--ak-swiglu-bwd
--ak-rmsnorm
--ak-res-rmsnorm
--ak-rope
--ak-rope-gate
--ak-causal-conv
--ak-qkv
--ak-ple-gate
--ak-normuon
--ak-compiled-autograd
--ak-triton-visible
--ak-sync-cleanup
--ak-ddp-tune
```

### 8.2 v3 add-on flags

```text
--ak-spectra-branchless
--ak-autocast-tier {none,tier1,all}
--ak-dtype-trace
--ak-fix-rope-gate-op
--ak-causal-conv-shim
--ak-normuon-telemetry
--ak-normuon-impl-opt
--ak-batch32-probe
--ak-trust-cap <float>
--ak-trust-cap-scope {none,w_gate_up,spiking,all_2d}
--ak-w-gate-up-scale <float>
--ak-w-gate-up-ramp-steps <int>
```

### 8.3 Rules for flags

1. No flag should implicitly enable all optimized kernels.
2. Every flag should be independently measurable.
3. Every flag should record itself in the checkpoint/training metadata.
4. Every flag should appear in the run manifest, profiler summary, and JSONL logs.
5. Hidden-kernel flags must run through gradient-flow audit and dtype parity before a 2000-step training gate.

---

## 9. Telemetry schema refinements

Do not overbuild telemetry before running experiments. Add enough to catch update-scale and dtype failures without reintroducing hot-path syncs.

### 9.1 NorMuon per-parameter update telemetry

Log periodically, not every microstep. Keep scalars tensor-side until sampled.

```json
{
  "step": 0,
  "param_name": "layers.0.ffn.w_gate_up.weight",
  "shape": [0, 0],
  "dtype_param": "torch.float16",
  "dtype_grad": "torch.float16",
  "dtype_update": "torch.float32_or_float16",
  "param_norm": 0.0,
  "raw_grad_norm": 0.0,
  "momentum_norm": 0.0,
  "post_ns_norm": 0.0,
  "spectra_sigma1": 0.0,
  "spectra_scale": 1.0,
  "effective_lr": 0.0,
  "update_norm_pre_trust": 0.0,
  "update_norm_post_trust": 0.0,
  "trust_ratio_pre": 0.0,
  "trust_ratio_post": 0.0,
  "trust_cap_triggered": false,
  "maxabs_param": 0.0,
  "maxabs_grad": 0.0,
  "maxabs_update": 0.0,
  "grad_isfinite": true,
  "update_isfinite": true
}
```

### 9.2 Custom-op dtype trace

```json
{
  "step": 0,
  "op_name": "autokernel::silu_gate_mul",
  "forward_input_dtypes": ["torch.float16", "torch.float16"],
  "forward_output_dtype": "torch.float16",
  "backward_input_dtype": "torch.float16",
  "backward_output_dtypes": ["torch.float16", "torch.float16"],
  "internal_accumulation": "fp16_or_fp32_or_unknown",
  "has_register_autocast": true,
  "has_register_autograd": true,
  "has_register_fake": true,
  "graph_break_source": false,
  "enabled_by_flags": ["--ak-swiglu-fwd", "--ak-swiglu-bwd"]
}
```

### 9.3 DDP communication telemetry

```json
{
  "step": 0,
  "accum_steps": 8,
  "no_sync_expected_microsteps": 7,
  "allreduce_count": 0,
  "allreduce_total_ms": 0.0,
  "first_allreduce_start_ms_after_backward_start": 0.0,
  "last_allreduce_end_ms": 0.0,
  "overlap_ratio_estimate": 0.0,
  "bucket_cap_mb": 25,
  "gradient_as_bucket_view": true,
  "static_graph": false
}
```

### 9.4 Sync telemetry

```json
{
  "profile_window_steps": 10,
  "aten_item_count": 0,
  "hipMemcpyWithStream_cpu_wall_s": 0.0,
  "known_hot_syncs": {
    "spectra_sigma1_item": 0,
    "loss_item": 0,
    "valid_global_sum_item": 0,
    "jsonl_logging": 0
  }
}
```

---

## 10. Experiment cards

The following cards are written so an engineering agent can execute without re-deriving the plan.

---

### Experiment E1 — Branchless SPECTRA sync removal

**Purpose:** Remove the largest known hot-path CPU-GPU synchronization source.

**Hypothesis:** Replacing `sigma1.item()` with tensor-side branchless scaling saves **1–4%** step wall with no learning regression.

**Patch sketch:**

```python
# Before: CPU scalar sync + branch
sigma1_val = sigma1.item()
if sigma1_val * (1.0 / safety_margin) <= clip_norm:
    return M
scale = (clip_norm * safety_margin) / max(sigma1_val, 1e-12)
return M * scale

# After: tensor-side branchless scale
scale = torch.clamp(
    clip_norm * safety_margin / torch.clamp(sigma1, min=1e-12),
    max=1.0,
)
return M * scale
```

**Run matrix:**

```text
E1.0 baseline current SPECTRA
E1.1 branchless SPECTRA
E1.2 branchless SPECTRA + reduced nonessential logging .item() calls
```

**Metrics:**

```text
tok/s
CUDA wall per optimizer step
CPU wall per optimizer step
aten::item count
hipMemcpyWithStream CPU wall
update parity vs baseline on fixed mini-batch
loss@200/500
GradScaler scale
```

**Pass gate:**

```text
≥1% tok/s improvement or meaningful sync reduction with no regression.
Update tensors match baseline within tolerance.
No loss/scaler regression at 500-step smoke.
```

**Interpretation:**

| Result | Meaning | Next action |
|---|---|---|
| +2–4% stable | T-0.3 estimate confirmed | Keep always-on |
| +0–1% but item count drops | syncs were not CUDA-critical | Keep if no overhead, but do not overcount |
| negative speed | extra multiply/kernel cost exceeds sync savings | specialize fast path or sample less often |
| update parity fails | patch changed SPECTRA semantics | fix before continuing |

---

### Experiment E2 — Fused z-loss / chunked CE+zloss on native hidden path

**Purpose:** Recover loss-path overhead without altering hidden-layer model math.

**Hypothesis:** Native hidden path plus fused z-loss or chunked CE+zloss yields **3–8%** tok/s gain and stable loss parity.

**Run matrix:**

```text
E2.0 native baseline, no fused zloss
E2.1 native hidden path + --use-fused-zloss only
E2.2 native hidden path + ChunkedLinearCrossEntropyLoss(z_loss_weight)
E2.3 native hidden path + chunked CE without zloss, control
```

**Minimum gate:**

```text
500 steps for throughput/stability smoke
2000 steps for learning parity if E2.1 or E2.2 looks promising
```

**Metrics:**

```text
tok/s
loss@200, 500, 1000, 2000
CE scalar parity
z-loss scalar parity
dlogits parity
valid-token denominator parity
ignore_index parity
softcap parity
GradScaler scale
nonfinite count
```

**Pass gate:**

```text
≥3% tok/s improvement, preferably ≥5%.
Loss curve within seed/run variance through 2000 steps.
No GradScaler collapse.
No frozen-param regression.
```

**Interpretation:**

| Result | Meaning | Next action |
|---|---|---|
| +5–8% stable | loss path is a core pillar | keep in all stacks |
| +3–5% stable | still useful but not enough alone | keep and stack |
| speed flat | loss implementation not reducing real wall | profile CE/logits materialization |
| divergence only in fused path | zloss gradient/dtype/denominator bug | inspect valid-token denominator, masking, softcap |
| both fused and non-fused diverge | not a zloss-specific issue | return to NorMuon/fp16 stability |

---

### Experiment E3 — DDP accumulation and bucket tuning

**Purpose:** Verify gradient accumulation is not allreducing every microstep and tune overlap.

**Hypothesis:** If `no_sync()` is already correct, DDP gains are probably **1–3%**. If `no_sync()` is broken, the gain could be much larger.

**Run matrix:**

```text
E3.0 current DDP config
E3.1 force no_sync() for microsteps 0..accum-2
E3.2 gradient_as_bucket_view=True
E3.3 bucket_cap_mb sweep: 8, 16, 25, 50, 100, 200
E3.4 static_graph=True only if unused-param behavior allows it
```

**Important implementation note:** PyTorch documents that the forward pass should be inside the `no_sync()` context; otherwise gradients may still synchronize.

**Metrics:**

```text
tok/s
allreduce count per optimizer step
allreduce total time
first allreduce start relative to backward start
overlap ratio
bucket count
peak memory
loss parity
```

**Pass gate:**

```text
No learning change.
Keep any config with ≥1% stable tok/s gain or memory reduction enabling batch=32.
```

**Interpretation:**

| Observation | Meaning | Next action |
|---|---|---|
| E3.1 huge gain | accumulation allreduce bug | fix immediately |
| bucket sweep +2–3% | overlap was suboptimal | keep tuned bucket |
| gradient_as_bucket_view no speed but saves memory | useful if it enables batch=32 | keep conditionally |
| compiled autograd later reduces overlap | reject compiled autograd unless net tok/s improves | preserve DDP overlap |

---

### Experiment E4 — Batch=32 / accum=4 probe

**Purpose:** Test whether larger microbatches reduce overhead and improve utilization without memory failure or learning drift.

**Hypothesis:** batch=32/accum=4 can yield **0–5%**, and prior sweep suggests about +5% if it fits.

**Run matrix:**

```text
E4.0 batch=16 accum=8 baseline
E4.1 batch=32 accum=4 no checkpoint, if fits
E4.2 batch=32 accum=4 + DDP gradient_as_bucket_view=True
E4.3 batch=32 accum=4 + selective checkpoint only if memory requires it
E4.4 batch=32 accum=4 + memory-reduction kernels only, no hidden backward changes
```

**Metrics:**

```text
tok/s
peak memory
allocation failures
step-time variance
loss@500/2000
GradScaler scale
activation maxabs/fp16 headroom
```

**Pass gate:**

```text
≥3% tok/s gain with clean 500-step smoke.
2000-step parity before adding to production stack.
```

**Interpretation:**

| Result | Meaning | Next action |
|---|---|---|
| fits and +5% | take it early | stack with E1/E2 |
| fits but speed flat | baseline already utilized | skip unless stability improves |
| requires checkpoint and loses speed | not worth it | revert |
| memory-reduction kernel enables +5% | hidden kernels useful for memory, not math | keep only memory-safe variants |

---

### Experiment E5 — NorMuon telemetry and implementation cleanup

**Purpose:** Identify optimizer update spikes and reduce NorMuon overhead without changing model math.

**Hypotheses:**

1. NorMuon update ratios may spike before fp16/GradScaler failure.
2. NorMuon implementation overhead can save **2–5%** if NS/SPECTRA/update apply is cleaned up.

**Run matrix:**

```text
E5.0 baseline NorMuon + telemetry only
E5.1 branchless SPECTRA from E1
E5.2 ns_steps=3, same LR, telemetry enabled
E5.3 ns_steps=3 + adjusted update scaling
E5.4 grouped/batched same-shape NS kernels
E5.5 fused SPECTRA scale + effective LR + update apply
```

**Metrics:**

```text
tok/s
optimizer step wall
raw_grad_norm
momentum_norm
post_NS_norm
spectra_sigma1
update_norm_pre/post_trust
actual_update_norm / param_norm
trigger rate for SPECTRA/trust cap
loss@500/2000
GradScaler scale
```

**Pass gate:**

```text
Throughput gain ≥2% with 2000-step learning parity, or telemetry proves a stability mechanism.
```

**Interpretation:**

| Observation | Meaning | Next action |
|---|---|---|
| update_ratio spikes before divergence | optimizer-scale failure | trust cap / w_gate_up staging |
| ns_steps=3 stable | easy optimizer speed win | keep and tune scaling |
| ns_steps=3 destabilizes | NS quality matters | keep 5, optimize implementation |
| no update spikes but activations explode | forward numerics/residual dynamics | fp32 islands/activation guard |
| optimizer cleanup +3–5% | core 40k pillar | stack with E1/E2/E4 |

---

### Experiment E6 — Tiered custom-op dtype/autocast audit

**Purpose:** Eliminate dtype-boundary and fp16 internal-math mismatches as a Phase C/G divergence mechanism.

**Hypothesis:** Some training-path custom ops lack explicit autocast behavior or fp32-sensitive internal math, causing small gradient differences that NorMuon/fp16 amplifies.

**Run matrix:**

```text
E6.0 Tier-0 dtype inventory across all seven ops
E6.1 Tier-1 graph-break op fixes: rope_gate_mul + causal_conv1d shim
E6.2 register_autocast on Tier-1 only
E6.3 register_autocast on all training-path custom ops
E6.4 fp32 internal math variants for silu/RMSNorm/rope/conv where applicable
E6.5 replay divergence-prone batch sequence through native vs custom paths
```

**Metrics:**

```text
fwd/bwd parity
grad cosine
post-NorMuon update cosine
dtype trace
activation maxabs drift
GradScaler scale
loss over fixed replay sequence
```

**Pass gate:**

```text
No dtype surprises.
Any register_autocast/fp32-island variant must pass parity and 500-step smoke before being stacked.
```

**Interpretation:**

| Observation | Meaning | Next action |
|---|---|---|
| missing autocast on active op | concrete bug/coverage gap | add rule + test |
| autocast rule improves stability | dtype mismatch likely cause | keep and expand tests |
| fp32 internal variant stable but slower | stability/speed tradeoff | use only in dangerous op |
| no dtype issues found | deprioritize H6 | focus NorMuon/batch/loss |

---

### Experiment E7 — Compiled autograd smoke

**Purpose:** Check whether compiled autograd has real net DDP value after T-0 data lowered expectations.

**Hypothesis:** Compiled autograd may save **1–4%**, good case **4–6%**, but may regress DDP overlap.

**Prerequisites:**

```text
- E3 DDP no_sync baseline known
- graph-break fixes or accepted break behavior documented
- dtype trace available
```

**Run matrix:**

```text
E7.0 native baseline, compiled_autograd=False
E7.1 native baseline, compiled_autograd=True
E7.2 native + fused zloss, compiled_autograd=True
E7.3 native + best DDP bucket/no_sync, compiled_autograd=True
E7.4 optional: Tier-1 custom-op fixes + compiled_autograd=True
```

**Implementation sketch:**

```python
import torch

torch._dynamo.config.compiled_autograd = True

# or narrow scope:
with torch._dynamo.compiled_autograd.enable(torch.compile(fullgraph=False)):
    loss.backward()
```

**Metrics:**

```text
steady-state tok/s after compile warmup
unique backward graphs
recompile count
compiled_autograd log summary
DDP allreduce overlap ratio
bucket timing
loss/scaler behavior
compile time separately from steady-state time
```

**Continue gate:**

```text
Continue beyond smoke only if net DDP tok/s improves by ≥2.5–3%,
with no overlap regression, no recompile churn, and clean 500-step smoke.
```

**Reject condition:**

```text
Reject compiled autograd if local backward is faster but end-to-end DDP tok/s is flat or worse because allreduce overlap regresses.
```

---

### Experiment E8 — Hidden-kernel recovery with delayed enable and trust controls

**Purpose:** Try to recover the remaining 5–7% possible hidden-kernel speed without repeating frozen-gradient behavior.

**Hypothesis:** Hidden kernels may be stable if enabled after early training and constrained by update-ratio controls.

**Prerequisites:**

```text
- E1/E2/E3/E5/E6 complete or understood
- no monolithic --optimize-kernels
- per-kernel flags available
- frozen-param audit clean
- dtype/autocast audit clean for enabled ops
```

**Run matrix:**

```text
E8.0 native baseline 1000 steps
E8.1 native 500 steps -> enable selected hidden kernels, preserve optimizer/scaler
E8.2 native 500 steps -> enable selected hidden kernels, fresh optimizer/scaler
E8.3 native 1000 steps -> enable selected hidden kernels, preserve optimizer/scaler
E8.4 enable only SwiGLU fwd/bwd with w_gate_up scale 0.25 -> 1.0 ramp
E8.5 same + post-NorMuon trust cap
E8.6 selected layers only, if replay localizes instability
```

**Trust cap patch sketch:**

```python
def apply_trust_cap_(update, param, effective_lr, tau=0.02, eps=1e-12):
    # update is post-NS and post-SPECTRA, before param.add_
    update_norm = torch.linalg.vector_norm(update.float())
    param_norm = torch.linalg.vector_norm(param.float())
    ratio = effective_lr * update_norm / (param_norm + eps)
    scale = torch.clamp(tau / (ratio + eps), max=1.0)
    update.mul_(scale.to(update.dtype))
    return ratio
```

**Correct composition order:**

```text
raw grad
→ momentum/Nesterov
→ Newton-Schulz
→ SPECTRA absolute cap
→ effective LR / shape scaling / param multiplier
→ trust-ratio cap
→ decoupled weight decay handling
→ param update
```

**Pass gate:**

```text
≥3% incremental speed beyond the safe stack.
No new frozen params.
No GradScaler collapse.
No sustained update-ratio spikes.
Clean 2000-step loss parity.
```

**Interpretation:**

| Observation | Meaning | Next action |
|---|---|---|
| delayed enable stable | warmup-local instability | production can enable after warmup |
| preserved state fails, fresh state works | optimizer-state mismatch | reset/transform state on enable |
| both fail shortly after enable | kernel numerics/dtype issue | return to E6 replay |
| trust cap fires before divergence and stabilizes | update-scale root cause | tune cap/staging |
| trust cap never fires but divergence remains | activation/logit overflow or kernel math | fp32 islands/residual guard |

---

## 11. Proposed execution schedule

### Day 0–1: unblock instrumentation and flags

```text
- Implement --ak-* granular flags.
- Add branchless SPECTRA flag.
- Add dtype trace scaffolding.
- Add NorMuon telemetry skeleton.
- Add DDP comm counters/no_sync verification.
```

### Day 1–2: immediate safe throughput patches

```text
- Run E1 branchless SPECTRA.
- Run E2 fused zloss/chunked CE smoke.
- Run quick E3 no_sync verification.
```

### Day 2–4: batch and optimizer path

```text
- Run E4 batch=32/accum=4.
- Run E5 telemetry-only.
- Patch obvious NorMuon sync/implementation issues.
```

### Day 4–6: dtype/autocast and graph-break infrastructure

```text
- Run E6 Tier-0 inventory.
- Fix rope_gate_mul and causal_conv1d wrappers if still needed.
- Add register_autocast rules.
- Run Tier-1/Tier-2 parity where needed.
```

### Day 6–7: compiled-autograd smoke

```text
- Run E7 only as a gated smoke.
- If <2.5–3% net DDP gain, stop.
- If gain is real and stable, schedule deeper CA work.
```

### Day 7+: stack and hidden-kernel recovery

```text
- Build best no-hidden-kernel stack.
- Run 2000-step stability gate.
- Only then attempt E8 hidden-kernel recovery.
```

---

## 12. Candidate throughput stacks

### Stack A — safest near-term stack

```text
native hidden path
+ branchless SPECTRA
+ fused z-loss / chunked CE+zloss if stable
+ DDP no_sync/bucket sanity
+ remove nonessential hot-path .item()
```

Expected result:

```text
33.5k–36k aggregate tok/s
```

This is the most likely stack to improve throughput without changing learning dynamics.

### Stack B — likely strong result

```text
Stack A
+ batch=32/accum=4 if fits
+ NorMuon implementation cleanup
```

Expected result:

```text
36k–38.5k aggregate tok/s
```

This is the stack I would treat as the realistic engineering target.

### Stack C — conditional compiled-autograd stack

```text
Stack B
+ compiled autograd only if E7 smoke shows ≥2.5–3% net DDP gain
```

Expected result:

```text
37k–39k if compiled autograd lands well
```

Do not budget this as guaranteed.

### Stack D — 40k stretch stack

```text
Stack B or C
+ hidden-kernel recovery through delayed enable
+ w_gate_up staging
+ post-NorMuon trust cap
+ dtype/autocast-clean training-path custom ops
```

Expected result:

```text
38.5k–40k+ only if several upper-bound wins stack with low overlap
```

This is the stretch path, not the mainline.

---

## 13. Stop conditions and non-negotiable quality gates

A throughput improvement does not count unless all quality gates pass.

### 13.1 Mandatory gates

```text
1. No newly frozen params.
2. No param with unexpected grad_fn=None path.
3. No GradScaler collapse.
4. No nonfinite update events.
5. No sustained activation/fp16-headroom danger trend.
6. 500-step smoke before stacking.
7. 2000-step learning parity before production recommendation.
8. Per-feature tok/s measured against the same baseline window.
9. Compile time separated from steady-state throughput.
10. DDP overlap measured whenever compiled autograd or graph changes are involved.
```

### 13.2 Explicit reject conditions

```text
Reject a feature if:
  - it increases tok/s by freezing or detaching gradients;
  - it requires hiding a frozen-param audit failure in an allowlist;
  - it improves local kernel microbench but does not improve end-to-end tok/s;
  - it speeds single-node but regresses DDP aggregate throughput;
  - it requires a lower LR that damages the validated loss curve unless the user explicitly accepts slower convergence;
  - it passes 200 steps but fails before 2000 with scaler collapse or rising loss pathology.
```

---

## 14. Precommitted interpretation table

| Observation | Most likely meaning | Action |
|---|---|---|
| Branchless SPECTRA gives +2–4% | Sync audit found real critical-path stalls | Keep, make default |
| Branchless SPECTRA gives ~0% | CPU syncs were mostly hidden by GPU work | Keep if no cost, but do not count |
| Fused zloss stable +5–8% | Loss path is safe pillar | Stack everywhere |
| Fused zloss diverges | denominator/masking/dtype bug | Fix CE/zloss parity before use |
| DDP no_sync huge gain | accumulation communication bug | Fix immediately |
| DDP bucket sweep +1–3% | overlap suboptimal | Keep tuned bucket |
| batch=32 fits +5% | clean utilization win | Promote early |
| batch=32 requires checkpoint and loses speed | memory tradeoff not worth it | Drop or use memory kernels |
| NorMuon update_ratio spikes pre-divergence | optimizer-scale failure | trust cap / staging |
| no update spikes but activation maxabs explodes | forward/residual/fp16 numerical issue | fp32 islands / activation guard |
| autocast rule changes stability | dtype boundary mismatch | add rule + tests |
| compiled autograd + local speed, DDP flat | allreduce overlap regression | reject CA for production |
| compiled autograd +3% net DDP | useful infrastructure | continue cautiously |
| hidden kernels stable only after step 1000 | warmup-local instability | delayed production enable possible |
| hidden kernels fail immediately after enable | kernel/dtype/backward mismatch | return to replay harness |

---

## 15. Engineering notes on Strix Halo constraints

The hardware context matters. Ryzen AI Max+ 395 / Strix Halo is an APU with Radeon 8060S-class integrated graphics, 40 graphics cores, 256-bit LPDDR5x memory, and LPDDR5x-8000 support. That gives roughly 256 GB/s theoretical memory bandwidth per machine before real-world efficiency losses. It is not an MI300X-style HBM system.

ROCm’s RDNA3.5 APU documentation also says memory access is GPUVM/GTT-backed over physically shared system memory rather than a separate discrete VRAM pool. That supports the same strategic conclusion as before:

```text
Do not spend time trying to beat rocBLAS on dense GEMMs.
Do spend time reducing extra passes, syncs, optimizer overhead, and memory traffic.
```

For this repo, the most addressable classes are:

```text
- loss path reductions / logsumexp / CE fusion
- optimizer path and SPECTRA/NS overhead
- CPU-GPU sync removal
- DDP overlap correctness
- batch/microstep utilization
- dtype/autocast stability for existing custom ops
```

The least attractive near-term target remains custom dense matmul.

---

## 16. Final recommendation

The next engineering sprint should be framed as:

```text
Recover 0.5–0.9 seconds per optimizer step through a stack of safe, measured deltas.
Do not attempt to make blanket --optimize-kernels production-safe as the primary objective.
```

The most plausible path is:

```text
31.3k baseline
→ 33.5–35k from SPECTRA sync cleanup + fused loss + small DDP cleanup
→ 36–38.5k from batch=32 and/or NorMuon implementation cleanup
→ 39–40k only if compiled autograd or hidden-kernel recovery lands cleanly
```

The order should be:

```text
1. Implement granular flags and telemetry.
2. Patch branchless SPECTRA.
3. Validate fused zloss/chunked CE on native hidden path.
4. Verify DDP no_sync and bucket behavior.
5. Probe batch=32/accum=4.
6. Use NorMuon telemetry to guide implementation cleanup and trust caps.
7. Run dtype/autocast inventory and Tier-1 graph-break fixes.
8. Run compiled-autograd smoke only; continue only if it earns its keep.
9. Attempt hidden-kernel recovery only after the safe stack is established.
```

The target remains 40k. The honest engineering success criterion should be **≥36k stable**, with **≥38k** as a strong result and **40k** as stretch. That framing keeps the team ambitious without rewarding invalid speedups.

---

## 17. Source notes

[^status]: Repo `STATUS.md`, documenting the autokernel remediation, Phase-B fixes, Phase-C/G divergence, and locked Sprint 3A/3B decisions: <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/main/STATUS.md>

[^backward]: T-0.1 backward breakdown: <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/main/docs/perf/backward-breakdown.md>

[^graph]: T-0.4 graph-break inventory: <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/main/docs/perf/graph-breaks-inventory.md>

[^sync]: T-0.3 sync-point audit: <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/main/docs/perf/sync-point-audit.md>

[^throughput]: OdinFlat throughput final investigation, including frozen-gradient speed artifacts: <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/main/docs/perf/odinflat-throughput-final.md>

[^compiled_autograd]: PyTorch compiled autograd tutorial, including its purpose and limitations: <https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html>

[^torch_library]: PyTorch `torch.library` docs for `custom_op`, `triton_op`, `register_autocast`, and `register_autograd`: <https://docs.pytorch.org/docs/2.11/library.html>

[^custom_ops]: PyTorch custom Python operators tutorial, including `register_autograd` and `opcheck` guidance: <https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html>

[^ddp_api]: PyTorch `DistributedDataParallel` API docs, including `no_sync`, `bucket_cap_mb`, and `gradient_as_bucket_view`: <https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html>

[^ddp_notes]: PyTorch DDP notes on allreduce/backward overlap and TorchDynamo DDPOptimizer: <https://docs.pytorch.org/docs/2.11/notes/ddp.html>

[^amp_examples]: PyTorch AMP examples, including unscale-before-clipping and accumulation guidance: <https://docs.pytorch.org/docs/2.11/notes/amp_examples.html>

[^amp_recipe]: PyTorch AMP recipe, including guidance to avoid `.item()` calls and other CPU-GPU synchronization when speedup is minor: <https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>

[^hip]: PyTorch HIP semantics, documenting that HIP reuses `torch.cuda` interfaces and `torch.device('cuda')`: <https://docs.pytorch.org/docs/2.11/notes/hip.html>

[^amd_395]: AMD Ryzen AI Max+ 395 product page, including Strix Halo, 256-bit LPDDR5x, LPDDR5x-8000, and Radeon 8060S specs: <https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html>

[^rdna35]: ROCm RDNA3.5 system optimization docs, including GPUVM/GTT shared-memory behavior for gfx1150/gfx1151/gfx1152 APUs: <https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html>
