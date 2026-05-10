# AutoKernel Halo Strix: 40k tok/s Throughput Recovery Guide

**Artifact type:** engineering handoff report and experiment guide  
**Target repo:** `joelhenwang/autokernel-halo-strix`  
**Primary target:** recover real pretraining throughput toward **~40k tok/s aggregate** without freezing parameters or degrading learning  
**Audience:** engineering agent starting fresh from the repo and this report  
**Prepared:** 2026-05-10 Europe/Lisbon time. Some repo source documents carry 2026-05-11 timestamps; this guide treats those as repo document labels, not as local-current-date assertions.

---

## 0. Read this first

The current problem is not “the kernels are broken” in a simple sense. It is more specific:

1. The original fast path produced attractive throughput partly because it silently removed gradient work from large parameter groups. That cannot ship.
2. The post-remediation autograd-safe path restores correct gradient flow, but then either:
   - loses most or all end-to-end throughput advantage, or
   - becomes numerically unstable under the current NorMuon + fp16 + production LR recipe.
3. The route to 40k tok/s should not be a single all-or-nothing `--optimize-kernels` switch. It should be a sequence of narrowly isolated, throughput-positive, learning-preserving optimizations.

The strongest recommendation in this report is:

> **Treat 40k tok/s as a step-time budget problem. First reclaim the loss path and optimizer path, then reintroduce hidden-layer kernels through isolated toggles only if they pass full training gates.**

For OdinFlat-like baseline throughput of ~31.3k tok/s, reaching 40k tok/s requires reducing step wall time to about `31.3 / 40.0 = 78.25%` of baseline, i.e. a **21.75% step-time reduction**. The repo step profile already identifies two large, plausible targets:

- z-loss `logsumexp` forward + backward: **~16.7% of step wall**.
- NorMuon optimizer work: **~12.5% of step wall**.

That means a real 40k path can exist without betting the whole project on SwiGLU replacement. A realistic stack is:

```text
native hidden path
+ fused / factorized CE + z-loss
+ NorMuon update optimization or update-level overhead reduction
+ reduced synchronization/logging overhead
+ optionally selective, delayed, or compiler-visible hidden-layer kernels
```

This is safer than:

```text
--optimize-kernels
```

because the all-in flag mixes autograd safety, kernel numerics, compiler visibility, optimizer dynamics, and fp16 stability into one uncontrolled variable.

---

## 1. Source map and evidence base

The report is based on the supplied repo and external documentation. The engineering agent should not need to re-read every source before starting, but these are the key references.

### 1.1 Repo sources

| Area | Source |
|---|---|
| Stability brief | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/research/autokernel-stability-research-brief.md> |
| Current status | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/STATUS.md> |
| OdinFlat final throughput investigation | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/odinflat-throughput-final.md> |
| Step profile | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/odinflat-step-profile.md> |
| Phase C divergence | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/phase-c-final-analysis.md> |
| Phase G OdinHalo divergence | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/phase-g-findings.md> |
| Post-fix audit synthesis | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/autokernel-audit-2026-05-11-synthesis.md> |
| Autokernel deep analysis | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/autokernel-deep-analysis.md> |
| Triton SwiGLU ship gate | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/triton-swiglu-ship-gate-bench.md> |
| HIP custom op registration | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/kernels/hip/_torch_ops.py> |
| Triton fused SwiGLU implementation | <https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/kernels/triton/fused_swiglu.py> |

### 1.2 External technical references

| Area | Source |
|---|---|
| PyTorch custom ops / `torch.library` / autograd / autocast / `triton_op` | <https://docs.pytorch.org/docs/2.11/library.html> |
| User-defined Triton kernels with `torch.compile` | <https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html> |
| PyTorch AMP and GradScaler usage | <https://docs.pytorch.org/docs/2.11/notes/amp_examples.html> |
| PyTorch Compiled Autograd | <https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html> |
| PyTorch Muon optimizer docs | <https://docs.pytorch.org/docs/2.11/generated/torch.optim.Muon.html> |
| Keller Jordan Muon explanation | <https://kellerjordan.github.io/posts/muon/> |
| Liger Kernel docs | <https://linkedin.github.io/Liger-Kernel/> |
| Liger fused linear CE implementation | <https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py> |
| PyTorch blog on Liger FLCE + compile | <https://pytorch.org/blog/peak-performance-minimized-memory/> |
| AMD Ryzen AI Max+ 395 official specs | <https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html> |
| AMD RDNA3.5 APU optimization docs | <https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html> |
| AMD hipBLASLt epilogue docs | <https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.1.0/reference/datatypes.html> |
| LARS / layer-wise trust ratio motivation | <https://openreview.net/forum?id=rJ4uaX2aW> |
| ULMFiT gradual unfreezing / discriminative fine-tuning | <https://arxiv.org/abs/1801.06146> |

---

## 2. Current state in one page

### 2.1 Hardware and system constraints

The project is targeting AMD Strix Halo / gfx1151 / RDNA 3.5 APUs. The stability brief describes the working machine as Ryzen AI Max+ 395-class hardware with:

- Radeon 8060S integrated GPU-class graphics.
- 40 CUs.
- unified LPDDR5X memory.
- no MFMA / matrix cores.
- small L2 relative to datacenter GPUs.
- fp16 + GradScaler as the practical training dtype stack.

AMD's official Ryzen AI Max+ 395 page lists Radeon 8060S graphics, 40 graphics cores, 256-bit LPDDR5x memory, up to 128 GB memory, and LPDDR5x-8000 memory speed. AMD's RDNA3.5 optimization docs describe Ryzen APUs with integrated RDNA3.5 graphics and LPDDR5X/DDR5 memory as a distinct APU optimization target.

**Consequences for kernel strategy:**

1. Do not expect hand-written transformer GEMMs to beat rocBLAS/Inductor paths. The target lacks tensor/matrix cores and is memory-bandwidth constrained.
2. Prioritize memory-traffic reduction, pointwise/reduction fusion, loss fusion, optimizer overhead, and avoiding synchronization.
3. Treat MI300X/H100-style assumptions skeptically. Many published kernel wins rely on HBM bandwidth, larger caches, tensor cores, or CUDA-specific libraries.

### 2.2 Model and training context

The repo is training Odin-family hybrid LMs:

- **OdinFlat:** about 122M parameters, 14-layer flat hybrid, production block size 512, FFN-heavy / sparse attention.
- **OdinHalo:** about 58M parameters, looped/shared structure, lower LR recipe, block size likely shorter in relevant probes.
- Training stack includes fp16 AMP + GradScaler, NorMuon, SPECTRA, μP for OdinFlat, z-loss, activation monitor, attn softcap, DDP across two Strix Halo machines.

The exact flags vary by sprint, but the key pattern is:

```text
native full-parameter path = learns, but ~25k–31k tok/s depending model
pre-fix optimized path = often much faster, but freezes parameters
post-fix optimized path = correct grad flow, but either no speedup or divergence
```

### 2.3 Silent-freeze incident

The original speed path used raw pybind HIP calls from module `forward()` functions. Those calls returned tensors with no autograd node. Result:

- Forward loss looked meaningful.
- Downstream weights still trained.
- Upstream parameters feeding the custom op silently stopped receiving gradients.
- The throughput increase was partly caused by not computing backward work for a large chunk of the model.

For OdinFlat, the postmortem identifies `ffn.w_gate_up.weight` as the main blast radius. The final throughput investigation table says:

- Baseline: about **31,331 tok/s**, loss **3.1466 @ step 2000**.
- Broken native + HIP SiLU: about **41,198 tok/s**, but loss **3.8010 @ step 2000**, a +0.65 loss regression.
- Correct autograd SiLU path: about **30,976 tok/s**, i.e. essentially baseline or slightly worse.

The deeper analysis corrects the frozen-weight estimate to about **44M `w_gate_up` weights** plus small norm weights, roughly **23%** of OdinFlat's trainable parameter population.

### 2.4 Phase B remediation

Phase B rewired five previously unsafe replacement paths through autograd-safe routes such as `torch.library.custom_op` plus `register_autograd`, or native PyTorch fallbacks.

Post-fix audit synthesis says production Odin-family probes now show zero newly frozen parameters under `--optimize-kernels`; remaining always-none/zero params are architecturally legitimate cases such as inactive `v_res_scale` or `head_gate`.

**Important:** this means the silent-freeze bug was addressed. The current problem is not primarily “gradients are missing.” The current problem is throughput and stability under correct full-parameter training.

### 2.5 Post-fix divergence

The fixed path diverges under current production recipes:

- **OdinFlat Phase C:** all three post-fix attempts diverged around step 200–250 at `lr_2d=5e-3`, including a DDP run with the same batch as the validated native Sprint 3A-confirm recipe.
- **OdinHalo Phase G:** at `lr_2d=2e-3`, the post-fix optimized run tracked **better** than the pre-fix frozen run through step 700, then diverged around step 750–800 with grad/scaler collapse.

This is crucial: the post-fix path can learn better before it blows up. That strongly suggests the fix is directionally correct, but the recipe crosses a numerical/optimizer stability boundary.

### 2.6 Current production decision

The repo status locks current production sprint decisions:

```text
Sprint 3A OdinFlat: no --optimize-kernels
Sprint 3B OdinHalo: no --optimize-kernels
--use-fused-zloss: opt-in, not validated end-to-end
Triton fused SwiGLU: kept as infrastructure, failed ship gate vs HIP
```

This guide is not arguing to overturn that immediately. It is a plan for post-sprint or focused recovery work to reach real 40k tok/s.

---

## 3. Problem restatement for the engineering agent

### 3.1 Goal

Reach or exceed approximately **40k tok/s aggregate** for pretraining while preserving model learning quality.

### 3.2 Non-negotiable constraints

A candidate throughput path is invalid if it:

- freezes parameters unintentionally;
- changes the training objective without explicit approval;
- passes only short 200-step probes but fails 1000–2000-step training gates;
- causes GradScaler collapse, repeated skipped steps, or fp16 headroom violation;
- regresses validation/BPB beyond normal seed variance;
- produces speed by bypassing backward or optimizer work.

### 3.3 The key performance math

For an OdinFlat-like baseline of ~31.3k tok/s:

```text
target throughput = 40.0k tok/s
baseline throughput = 31.3k tok/s
required throughput multiplier = 40.0 / 31.3 ≈ 1.278x
required step-time multiplier = 31.3 / 40.0 ≈ 0.7825
required step-time reduction ≈ 21.75%
```

If z-loss forward+backward is really **16.7%** of step wall and can be mostly eliminated without destabilizing training:

```text
new throughput ≈ 31.3 / (1 - 0.167) ≈ 37.6k tok/s
```

Then only another ~6–7% step-time improvement is needed:

```text
40.0 / 37.6 ≈ 1.064x throughput
```

That can plausibly come from optimizer overhead, reduced syncs, compiled backward, or small safe kernels. This is why the recommended route is loss-first.

For an OdinHalo-like baseline of ~25k tok/s, 40k is harder:

```text
25 / 40 = 0.625 step-time multiplier
required step-time reduction ≈ 37.5%
```

OdinHalo may need both loss/optimizer wins and a stable subset of hidden-layer optimization. But the same discipline applies: stack isolated safe wins rather than enabling all replacements.

---

## 4. Root-cause assessment

### 4.1 What is already proven

#### Proven fact A: raw pybind training-path HIP ops are unsafe

A raw pybind HIP call bypasses PyTorch dispatcher/autograd registration. If used in a training `forward()`, its output can have `grad_fn=None`, severing gradient propagation. The repo directly observed this in the original `_FusedSwiGLUReplacement` path.

**Action:** never reintroduce a raw pybind op into a training path. It may be inference-only, but not training.

#### Proven fact B: the old speedup was partly fake

The final OdinFlat table shows the broken path at ~41.2k tok/s but worse long-horizon loss, while the autograd-correct SiLU path drops back to ~31.0k tok/s. That means the raw speed result cannot be used as an estimate of safe speed.

#### Proven fact C: Phase B fixes restore gradient flow

Post-fix dynamic probes show zero newly frozen params across successful production Odin models. Runtime preflight also catches missing gradient flow before training starts.

#### Proven fact D: the fixed path is unstable at current production LRs

Phase C and Phase G divergence documents show that `--optimize-kernels` under Phase B fixed code diverges at current tuned LRs.

#### Proven fact E: z-loss and NorMuon are large performance targets

The step profile identifies:

```text
z-loss logsumexp forward: 11.1%
z-loss backward: 5.6%
NorMuon optimizer: 12.5%
```

That makes them better first-order targets than QKV fusion or isolated SwiGLU.

### 4.2 Most likely causal explanation

The most likely current failure mechanism is:

```text
post-fix path restores full-parameter training
+ custom-kernel numerics differ slightly from native Inductor path
+ NorMuon orthogonalizes/scales 2D updates aggressively
+ fp16 GradScaler has limited overflow margin
+ current LR/warmup was tuned near a stability boundary
= delayed divergence once update/activation statistics cross fp16 headroom
```

This is not the same as “the kernels cannot learn.” Phase G learned better than the pre-fix frozen run through step 700. The current issue is that the optimized full-parameter path is unstable under the current recipe.

### 4.3 Why “just fix SwiGLU” is unlikely to be enough

SwiGLU has three separate issues:

1. **Autograd safety:** fixed by custom op or autograd function.
2. **End-to-end performance:** the safe HIP path loses isolated speed because of dispatch, saved tensors, backward overhead, and compiler opacity.
3. **Training stability:** any small difference in fwd/bwd or dtype handling can interact with NorMuon/fp16.

The Triton SwiGLU kernel achieved about 1.43x vs eager in isolated fwd+bwd, but only about 0.986x vs the autograd-safe HIP path and failed the repo's ship gate. That does not mean Triton is useless; it means **this particular isolated elementwise SwiGLU is not the main 40k path**.

---

## 5. Ranked hypotheses

The following hypotheses are ordered by expected value: probability of being true × expected speedup × safety.

### H1. The safest 40k path is fused/factorized loss, not hidden-layer kernels

**Assessment:** highest priority.

**Evidence:**

- z-loss `logsumexp` forward + backward is measured at ~16.7% of step wall.
- It is outside the hidden representation path, so it is less likely to disturb layer dynamics than replacing MLP internals.
- Phase C divergence occurred both with and without fused z-loss variants, so fused z-loss is not cleanly established as the root cause.
- Liger Kernel and PyTorch/torchtune work show that fused linear CE / chunked CE is a major practical optimization target for LLM training, often with exact loss computation and reduced memory traffic.

**Hypothesis:** native hidden path + fused CE/z-loss can recover most of the required step-time reduction while preserving learning.

**Predicted outcome:**

```text
baseline 31.3k → 36k–38k with loss-only fusion
```

**Main risk:** denominator/masking/softcap/label-smoothing mismatch in the loss gradient, especially with `ignore_index`, z-loss fraction, or chunking.

**Experiment family:** E1, E2, E3 below.

---

### H2. `torch.library.custom_op` opacity and dispatch overhead erase the SwiGLU win

**Assessment:** high priority for diagnosis, medium priority for speed.

PyTorch docs state that `torch.library.custom_op` treats custom ops as opaque to `torch.compile`/export, while `torch.library.triton_op` makes Triton implementations visible to compiler subsystems. The current HIP custom-op path may prevent Inductor/AOTAutograd from fusing around or into the operation well enough to beat the native `F.silu(gate) * up` path.

**Hypothesis:** the autograd-safe HIP path is semantically correct but too opaque/fragmented; a compiler-visible Triton op or direct in-graph implementation may recover part of the speed.

**Predicted outcome:**

```text
custom_op SwiGLU: ~baseline or slightly slower
triton_op/direct Triton: maybe +2% to +5% total step if compiler integration works
```

**Main risk:** not enough upside. Isolated fwd+bwd speed is not enough; end-to-end training may still be baseline.

**Experiment family:** E4, E5.

---

### H3. Custom-op autocast/dtype behavior differs from native Inductor and destabilizes fp16

**Assessment:** high priority for stability.

PyTorch exposes `torch.library.register_autocast()` for custom ops. A custom op that does not match the native autocast/dtype behavior can be numerically close on toy tests but different enough to affect fp16 stability under NorMuon.

**Hypothesis:** the fixed path diverges partly because one or more kernels use slightly different fp16/fp32 intermediate behavior than native PyTorch/Inductor.

**Predicted outcome:**

```text
fp32 internal sigmoid/silu or registered autocast variant improves update cosine and delays/prevents divergence
```

**Main risk:** precision fixes reduce speed.

**Experiment family:** E6.

---

### H4. NorMuon amplifies small gradient differences into unstable updates

**Assessment:** high priority.

Muon-style optimizers compute 2D hidden-layer updates using momentum followed by Newton-Schulz orthogonalization. PyTorch's Muon docs note 2D hidden-layer focus, NS steps, and shape/RMS adjustment options. That makes raw gradient norm insufficient as the only safety metric.

**Hypothesis:** the optimized kernels perturb gradients enough that NorMuon post-processing yields update spikes, especially in `w_gate_up`, causing activation growth and scaler collapse.

**Predicted outcome:**

```text
raw grad norms may look acceptable
post-NS/post-SPECTRA update ratios spike before divergence
post-update trust caps or w_gate_up update scaling stabilize the run
```

**Main risk:** stabilizers reduce learning speed or change the tuned recipe.

**Experiment family:** E7, E8, E9.

---

### H5. Enabling hidden-layer kernels after warmup avoids the fragile early window

**Assessment:** medium-high priority because it is cheap to test.

Phase C divergence happens near the warmup/LR boundary for OdinFlat; Phase G diverges later for OdinHalo. Delayed activation may avoid early high-curvature instability and still recover most full-run wall-clock time.

**Hypothesis:** native training through step 500–1000 followed by optimized kernels is stable, or at least diverges less quickly.

**Predicted outcome:**

```text
if instability is warmup-specific: delayed enable succeeds
if instability is kernel/optimizer-intrinsic: delayed enable diverges shortly after switch
```

**Main risk:** if switching changes graph compilation or optimizer state unexpectedly, results are noisy.

**Experiment family:** E10.

---

### H6. Selective layer enablement can recover partial speed without crossing stability threshold

**Assessment:** medium priority.

The all-layer switch is too blunt. Divergence may localize to specific layers, attention-adjacent blocks, loop iterations, or high-activation layers.

**Hypothesis:** enabling SwiGLU or RMSNorm replacement only on safe layer subsets gives a partial speedup and avoids runaway layers.

**Predicted outcome:**

```text
half-layer kernel enable + fused loss may reach target for OdinFlat-like runs
one or two layers may be responsible for most stability failures
```

**Main risk:** partial hidden-layer speedup alone is likely too small; only useful when stacked with loss/optimizer wins.

**Experiment family:** E11.

---

### H7. Optimizing NorMuon itself is safer than replacing hidden-layer math

**Assessment:** medium-high priority.

NorMuon is measured at ~12.5% of step wall. It can be validated by comparing exact update tensors before applying them. This is safer than hidden-layer replacement because model forward/backward math remains native.

**Hypothesis:** fused/grouped NorMuon operations, fewer CPU syncs, fewer dtype conversions, or fewer NS steps with matching scale can recover enough speed to stack with loss fusion.

**Predicted outcome:**

```text
+3% to +8% total step-time improvement with less learning risk than hidden kernels
```

**Main risk:** optimizer implementation complexity; incorrect update math can silently alter training.

**Experiment family:** E12, E13.

---

### H8. Host/device synchronization overhead is a small but real final-mile target

**Assessment:** medium priority.

The step profile notes hundreds of `aten::item` calls across the profile window. On an APU with unified memory, the CPU may not be the primary bottleneck, but `.item()` still forces stream synchronization.

**Hypothesis:** reducing frequent CPU scalar reads and logging syncs gives 1–3% speedup and improves stability telemetry quality.

**Predicted outcome:**

```text
small speedup; useful as final-mile after major wins
```

**Main risk:** losing observability if logging is reduced too aggressively.

**Experiment family:** E14.

---

### H9. hipBLASLt epilogues may help, but probably not full SwiGLU

**Assessment:** speculative.

AMD hipBLASLt supports epilogues such as GELU, ReLU, Swish/SiLU, and bias variants. However, a transformer SwiGLU MLP needs a pairwise `silu(gate) * up` operation over the two halves of a projection output, then a second down-projection GEMM. Simple single-output activation epilogues are not a full SwiGLU fusion.

**Hypothesis:** hipBLASLt epilogues could help for some activation/bias subcases, but likely do not solve Odin's full SwiGLU MLP without a more specialized fused GEMM+SwiGLU epilogue path.

**Predicted outcome:**

```text
maybe helpful for GeGLU/GELU or simple SiLU layers; limited for current SwiGLU
```

**Main risk:** time sink with little payoff on gfx1151.

**Experiment family:** E15 only if earlier phases fail.

---

### H10. Full custom GEMM fusion is the wrong first bet on this hardware

**Assessment:** high confidence.

Because gfx1151 lacks matrix cores and rocBLAS is the tuned GEMM implementation, writing custom HIP/Triton matmuls is unlikely to beat vendor GEMM. Replacing `w_gate_up` and `w_down` GEMMs with homemade kernels is likely to regress throughput unless the epilogue fusion saves enough memory traffic to compensate. That is a major research project, not the first sprint to 40k.

**Action:** do not start here.

---

## 6. Required instrumentation before more long runs

Before running more 1000–2000-step ablations, add telemetry that can distinguish:

```text
wrong gradient
correct gradient but bad update scaling
activation overflow
loss-kernel mismatch
compiler/runtime overhead
```

### 6.1 Per-parameter update telemetry

Global grad norm is insufficient with NorMuon. Add per-2D-parameter logging at the optimizer step, after each transformation stage.

For each relevant parameter group:

```text
step
param_name
shape
param_norm
raw_grad_norm
raw_grad_maxabs
momentum_norm
post_ns_norm
post_ns_maxabs
post_spectra_norm
actual_update_norm = ||lr * update||
actual_update_maxabs
update_to_weight_ratio = ||lr * update|| / (||weight|| + eps)
cos(raw_grad, final_update)
cos(native_update, optimized_update) when running paired replay
```

Do not log every parameter every step in production. For experiments, log every optimizer step for the first 300 steps, then every 10 or 20 steps. Always log spikes.

**Tripwire suggestions:**

```text
update_to_weight_ratio > 0.02 for any large hidden matrix: warning
update_to_weight_ratio > 0.05: save forensics and optionally stop
post_ns_norm/raw_grad_norm grows by >10x on same param: warning
cos(native_update, optimized_update) < 0.99 on major FFN matrix: investigate
```

These thresholds are starting points. Tune them from native baseline distributions.

### 6.2 Activation telemetry

Log per-layer activation maxabs and fp16 headroom around:

```text
input to ffn_norm
output of ffn_norm
gate half of w_gate_up
up half of w_gate_up
silu(gate)
silu(gate) * up
w_down output
post-residual hidden
attention logits if applicable
final logits maxabs
```

For overhead control, make this a debug mode and sample only selected layers by default:

```text
layers: first, middle, attention-adjacent, last
for OdinHalo: per shared layer and per loop iteration if feasible
```

### 6.3 GradScaler and overflow telemetry

PyTorch AMP guidance says gradients produced by `scaler.scale(loss).backward()` are scaled, so any clipping or inspection of `.grad` must happen after `scaler.unscale_(optimizer)` and only after all accumulation microsteps for the effective batch are complete.

Log:

```text
scaler_scale
was_step_skipped
num_nonfinite_grads
first_nonfinite_param
unscaled_global_grad_norm
scale_growth/backoff events
consecutive skipped steps
```

### 6.4 Loss-path parity telemetry

For any fused CE/z-loss candidate, log:

```text
ce_loss_native
ce_loss_fused
z_loss_native
z_loss_fused
total_loss_native
total_loss_fused
valid_token_count_native
valid_token_count_fused
ignore_index count
label_smoothing flag/value
softcap flag/value
z_loss_fraction/effective z_weight
```

For a replay batch, compare gradients:

```text
d_logits cosine / rel_err if logits materialized
d_hidden cosine / rel_err
d_lm_head or factorized head gradient cosine / rel_err
```

### 6.5 Replay bundle on first failure

On first scaler collapse, nonfinite grad, or update-ratio spike, save:

```text
batch token ids / labels / doc mask
model state pre-step
optimizer state pre-step
scaler state
activation stats
per-param update stats
feature flags / environment
compile mode
git SHA
```

Then replay the same batch under:

```text
native
native + fused loss
optimized + PyTorch fallback backward
optimized + HIP backward
optimized + lower w_gate_up update scale
```

This converts “run diverged” into a local, reproducible diagnosis.

---

## 7. Experiment matrix overview

The following experiment families are designed to be run in order. Do not run the entire matrix blindly. Each phase has a decision gate.

| ID | Theme | First run | Primary decision |
|---|---|---|---|
| E0 | Baseline + telemetry | native known-good recipe | establish update/activation baseline |
| E1 | Native + fused z-loss only | no hidden kernels | is loss-only path stable and faster? |
| E2 | Native + chunked/factorized fused CE+z-loss | no hidden kernels | can loss path alone get high 30k? |
| E3 | Loss parity stress tests | unit/replay | is loss gradient exactly correct? |
| E4 | SwiGLU integration path comparison | native vs custom_op vs triton_op | is compiler opacity the issue? |
| E5 | End-to-end hidden-kernel micro-stack | only one replacement family | does any hidden kernel produce real e2e speed? |
| E6 | Autocast/precision variants | SwiGLU/RMSNorm/rotary | is dtype behavior causing instability? |
| E7 | NorMuon update telemetry | native vs optimized | where do updates diverge? |
| E8 | Post-update trust caps | optimized path | can we stabilize without killing learning? |
| E9 | `w_gate_up` LR/update staging | optimized path | is FFN gate path the unstable group? |
| E10 | Delayed enable | native first, kernels later | is instability warmup-local? |
| E11 | Layer-selective enable | subsets | can partial kernels stack safely? |
| E12 | NorMuon performance optimization | native hidden path | can optimizer speed provide final 3–8%? |
| E13 | NS steps / scale matching | native and optimized | can optimizer speed/stability improve together? |
| E14 | Sync reduction | native and fused loss | final-mile speed |
| E15 | hipBLASLt epilogue exploration | isolated prototype | only if earlier routes fail |

---

## 8. Experiment E0 — Baseline with full telemetry

### Purpose

Establish native distributions for update ratios, activation maxabs, scaler behavior, and throughput. Without this, the new telemetry has no reference scale.

### Configuration

Use the known-good native recipe for the target model.

For OdinFlat Sprint 3A-style baseline, the repo's locked recipe is approximately:

```bash
EXTRA_FLAGS='--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --auto-eval'
# No --optimize-kernels.
```

### Minimum run length

```text
1000 steps minimum
2000 steps preferred
```

### Metrics

```text
tok/s median after warmup
loss curve
validation/BPB if available
activation maxabs and fp16 headroom
GradScaler scale / skipped steps
per-param update_to_weight_ratio distribution
```

### Gate

This is the control. If it is unstable, stop and fix baseline first. Do not interpret kernel results against a drifting baseline.

---

## 9. Experiment E1 — Native + fused z-loss only

### Purpose

Test the highest expected-value speed path without modifying hidden-layer math.

### Configuration

```text
native model path
no --optimize-kernels
only enable fused z-loss or fused CE z-loss path
```

Potential flag if current repo supports it:

```bash
--use-fused-zloss
```

But verify that this flag does not implicitly enable other autokernel replacements. It should be a loss-only change.

### Why this matters

The step profile says separate z-loss `logsumexp` forward + backward is ~16.7% of step wall. Eliminating most of this could move OdinFlat-like throughput from ~31.3k to ~37.6k.

### Run length

```text
200-step smoke for immediate gross errors
1000-step stability check
2000-step ship-quality gate
```

Do not accept a 200-step pass as proof. The silent-freeze failure looked acceptable early.

### Success gate

```text
tok/s: ≥36k sustained on OdinFlat-like baseline, or proportional improvement for target model
loss@1000 and loss@2000: within normal baseline seed variance
GradScaler: no collapse, no repeated skipped steps
frozen-param audit: clean
loss parity: native vs fused CE/z-loss within tolerance on replay batches
```

### Failure interpretation

| Observation | Interpretation |
|---|---|
| Throughput unchanged | fused path not actually eliminating logsumexp or bottleneck shifted |
| Loss mismatch immediately | formula/masking/denominator bug |
| Loss ok but GradScaler issues | z-loss gradient scale or dtype issue |
| Stable but only +3% | profile attribution was misleading or fusion incomplete |

---

## 10. Experiment E2 — Factorized fused CE + z-loss

### Purpose

Build a more ambitious loss-path kernel that fuses the LM head projection, CE, and z-loss, especially useful if Odin's output head is factorized/rank-based.

### Design sketch

Flatten hidden states:

```text
h: [N, H_or_rank]
targets: [N]
weight/head: [V, H_or_rank] or factorized equivalent
```

Stream vocab tiles without materializing the full logits tensor:

```text
for each token row i:
    online max over vocab tiles
    online logsumexp over vocab tiles
    target logit
    CE = lse - target_logit
    z = z_weight * lse^2
```

Backward:

```text
softmax_j = exp(logit_j - lse)
grad_logit_j = softmax_j / denom
if j == target: grad_logit_j -= 1 / denom
z_loss contribution: grad_logit_j += 2 * z_weight * lse * softmax_j / denom
accumulate d_h += grad_logit_j * W_j
accumulate d_W_j += grad_logit_j * h_i
```

For factorized heads, adapt this to the actual repo architecture. Start with a chunked PyTorch reference implementation before writing HIP/Triton.

### Why this route is attractive

Liger Kernel's fused linear CE avoids materializing large logits and computes gradients in a chunked/fused manner. Liger docs describe exact computation, unit tests, and convergence testing; the source implementation supports options such as `ignore_index`, `lse_square_scale`, label smoothing, softcap, and accumulation dtype. This is directly relevant to the repo's z-loss and CE path.

### Implementation steps

1. Add a pure PyTorch chunked reference:

```text
loss = chunked_factorized_ce_zloss_reference(...)
```

2. Compare against current native CE+z-loss:

```text
loss rel_err
d_hidden cosine
d_head_weight cosine
valid token denominator
```

3. Only after reference parity, implement Triton/HIP.

4. Add production-shape tests:

```text
block sizes: 256, 512
vocab: 32768
batch/accum equivalent to production
ignore_index rows
softcap on/off
z_loss_fraction on/off
label smoothing if applicable
```

### Success gate

```text
native hidden path unchanged
loss parity exact enough for fp16 training
2000-step learning parity
tok/s ≥37k on OdinFlat-like baseline, or proportional gain
```

### Failure interpretation

| Observation | Interpretation |
|---|---|
| Unit parity fails | fix denominator/masking/softcap before training |
| Unit parity passes but training drifts | compare d_hidden and d_head update under real batches |
| Speed poor | memory tiling or head factorization not exploiting shape; profile tile occupancy |

---

## 11. Experiment E3 — Loss parity stress suite

### Purpose

Prevent a subtle fused loss bug from being mistaken for a model/kernel instability.

### Required cases

```text
1. normal labels, no ignore_index
2. random ignore_index positions
3. entire sequence ignored
4. mixed valid/ignored within same batch
5. z_loss_weight = 0
6. z_loss_weight = 1e-4
7. z_loss_fraction active/inactive
8. softcap off
9. softcap on
10. label smoothing off/on if supported
11. reduction = mean, and denominator exactly checked
12. fp32 reference vs fp16/autocast execution
```

### Metrics

```text
loss_abs_err
loss_rel_err
d_hidden_cosine
d_hidden_max_abs_err
d_head_cosine
d_head_max_abs_err
valid_token_count
```

### Acceptance

For production fp16, do not rely only on relative error. Use:

```text
cosine ≥ 0.9999 for major gradients
no systematic bias in loss across random batches
no denominator mismatch under ignore_index
no NaN/Inf in extreme-logit tests
```

---

## 12. Experiment E4 — SwiGLU integration path comparison

### Purpose

Determine whether the current correct SwiGLU path is slow because of compiler opacity, autograd overhead, or unavoidable backward cost.

### Variants

Compare these in the same harness:

```text
A. native: F.silu(gate) * up under current compile_zones
B. HIP custom_op + register_autograd, PyTorch fallback backward
C. HIP custom_op + register_autograd, HIP backward
D. Triton torch.autograd.Function implementation
E. torch.library.triton_op + wrap_triton implementation
F. direct Triton call inside compiled zone, if possible
```

The PyTorch docs recommend `torch.library.triton_op` for Triton-backed custom operators because `custom_op` is opaque to `torch.compile`, while `triton_op` is visible to compiler subsystems. They also recommend registering autograd for training support; if the backward calls Triton kernels, those kernels should also be wrapped as `triton_op`.

### Metrics

Do not stop at fwd+bwd microbench. Record:

```text
isolated fwd latency
isolated fwd+bwd latency
end-to-end tok/s
unique compile graphs
graph breaks
recompile count
loss@200/1000/2000
grad cosine vs native
post-NorMuon update cosine vs native
```

### Gate

```text
isolated speedup is necessary but not sufficient
end-to-end speedup ≥5% total step wall for hidden-kernel work to matter
loss parity through 1000–2000 steps
no update-ratio spikes vs native baseline
```

### Expected outcome

The most likely outcome is that native Inductor is already competitive for correct SwiGLU. If so, stop trying to recover 40k from SwiGLU alone and return to loss/optimizer targets.

---

## 13. Experiment E5 — One replacement family at a time

### Purpose

Replace the all-or-nothing `--optimize-kernels` flag with granular toggles.

### Add flags

Implement or expose flags like:

```bash
--ak-rmsnorm {native,hip_custom_op,triton_op}
--ak-swiglu {native,hip_fwd_torch_bwd,hip_fwd_hip_bwd,triton_fn,triton_op}
--ak-rope {native,hip_custom_op,triton_op}
--ak-qkv {native,fused}
--ak-loss {native,fused_zloss,fused_ce_zloss,factorized_ce_zloss}
--ak-normuon {native,fused,debug}
```

Also add environment toggles where flags are hard to plumb:

```bash
AUTOKERNEL_NO_BWD_HIP=1
AUTOKERNEL_SWIGLU_MODE=triton_op
AUTOKERNEL_LOSS_MODE=fused_zloss
```

### Run pattern

For each family:

```text
native baseline
+ one replacement only
+ same replacement with PyTorch fallback backward
+ same replacement with fused/HIP backward
```

### Gate

A replacement family must satisfy:

```text
no newly frozen params
unit fwd+bwd parity
end-to-end tok/s improvement
loss parity through 1000–2000 steps
no GradScaler instability
```

### Do not stack until single-family tests pass

Stacking unvalidated replacements produces uninterpretable failures.

---

## 14. Experiment E6 — Autocast and precision variants

### Purpose

Find whether dtype/autocast differences explain post-fix instability.

### Variants for each custom op

```text
1. no explicit register_autocast
2. register_autocast(device_type="cuda", cast_inputs=torch.float16)
3. fp16 input/output, fp32 internal sigmoid/silu math
4. fp16 forward, fp32 backward intermediates
5. PyTorch fallback backward with fp32 intermediates
6. HIP/Triton backward with fp32 intermediates where practical
```

Although the hardware is ROCm/HIP, PyTorch custom op registration often uses `device_type` values matching PyTorch device semantics; confirm whether the relevant tensor device type appears as `cuda` under ROCm PyTorch or needs a ROCm-specific path. Do not guess in code; inspect `tensor.device.type` in the runtime.

### Metrics

```text
fwd rel_err vs native
grad rel_err and cosine vs native
post-NorMuon update cosine vs native
activation maxabs trend
GradScaler trend
throughput
```

### Gate

If a precision variant stabilizes training with only a small speed cost, keep it. A slightly slower stable kernel may still stack with fused loss to reach 40k.

---

## 15. Experiment E7 — NorMuon update bisect

### Purpose

Determine whether divergence is driven by raw gradients or by the optimizer transformation.

### Paired replay

Use the same batch and same model state. Run:

```text
native forward/backward -> collect raw grads -> run NorMuon update without applying
optimized forward/backward -> collect raw grads -> run NorMuon update without applying
```

Compare:

```text
raw_grad cosine per param
post_momentum cosine
post_NS cosine
post_SPECTRA cosine
final_update cosine
update_to_weight_ratio difference
```

### Interpretation

| Finding | Meaning |
|---|---|
| raw grads differ significantly | kernel backward/numerics issue |
| raw grads close, post-NS diverges | NorMuon sensitivity / scaling issue |
| final update ratio spikes only after SPECTRA | SPECTRA interaction |
| only `w_gate_up` problematic | use param-specific staging/scaling |
| many layers problematic | lower LR / trust cap / precision issue |

---

## 16. Experiment E8 — Post-NorMuon trust cap

### Purpose

Prevent optimizer-transformed updates from exceeding a safe layer-wise/parameter-wise ratio.

### Motivation

Layer-wise adaptive rate methods such as LARS were introduced because large-batch training with global LR and warmup can diverge; LARS uses layer-wise scaling based on norms. Here, the exact optimizer is different, but the safety problem is similar: the final update norm relative to the weight norm matters.

### Patch sketch

Inside NorMuon step, after the final update tensor is computed and before applying it:

```python
# Pseudocode only; adapt to actual optimizer implementation.
update = final_update
lr = group["lr"]
eps = 1e-12

param_norm = p.data.float().norm()
update_norm = (lr * update.float()).norm()
ratio = update_norm / (param_norm + eps)

trust_cap = group.get("trust_cap", None)
if trust_cap is not None and torch.isfinite(ratio) and ratio > trust_cap:
    scale = trust_cap / (ratio + eps)
    update.mul_(scale.to(update.dtype) if torch.is_tensor(scale) else scale)
```

### Suggested grid

```text
trust_cap = None
trust_cap = 0.05
trust_cap = 0.02
trust_cap = 0.01
```

Apply variants:

```text
all hidden 2D params
only w_gate_up
only layers with observed spikes
```

### Success gate

```text
optimized path stable through 1000–2000 steps
loss does not lag native by more than normal variance
throughput remains improved
update-ratio spikes disappear
```

### Failure interpretation

| Observation | Interpretation |
|---|---|
| cap stabilizes but loss stalls | cap too tight or target too broad |
| cap does not affect divergence | raw kernel gradient/numerics issue likely |
| only `w_gate_up` cap works | specific FFN gate path instability |

---

## 17. Experiment E9 — `w_gate_up` LR/update staging

### Purpose

Test whether the large FFN gate/up projection group is the unstable parameter family after autograd restoration.

### Rationale

The silent-freeze blast radius concentrated on `w_gate_up`. The fixed path correctly trains it. That is good, but it changes the effective dynamics versus the recipe that appeared stable under the frozen path. Borrow the spirit of discriminative learning rates and gradual unfreezing: stage the sensitive group rather than reducing the entire model LR.

### Schedule variants

```text
A. w_gate_up_lr_mult = 0.25 for steps 0–1000, ramp 0.25→1.0 over steps 1000–2000
B. w_gate_up_lr_mult = 0.10 for steps 0–1000, ramp 0.10→1.0 over steps 1000–3000
C. w_gate_up_update_scale = 0.25 post-NorMuon, ramp to 1.0
D. only attention-adjacent layers scaled
E. only layers with update-ratio spikes scaled
```

### Implementation detail

Do not only scale the raw gradient if NorMuon later normalizes/orthogonalizes it in a scale-invariant way. Scaling may need to be applied to:

```text
learning rate group
post-NorMuon final update
or both, depending actual optimizer math
```

### Gate

```text
Phase C/G divergence window passed
loss trajectory remains competitive
update_to_weight_ratio is controlled
activation maxabs does not grow exponentially
```

---

## 18. Experiment E10 — Delayed kernel enable

### Purpose

Determine whether optimized kernels are unstable only during early warmup/catch-up dynamics.

### Plan

Run native first, then resume with selected kernels:

```text
Run native to step 500.
Save checkpoint including model, optimizer, scaler, RNG, dataloader state.
Resume with only fused loss.
Resume with fused loss + one hidden replacement.
Resume with fused loss + hidden replacement + trust cap.
```

Then repeat with delay step 1000.

### Candidate commands

Exact scripts will depend on current repo launchers. Conceptually:

```bash
# Stage A: native warmup/control
python scripts/train_ddp.py ... --max-steps 1000 [baseline flags] --save-every 500

# Stage B: resume with selected feature
python scripts/train_ddp.py ... --resume-from checkpoints/.../step_1000.pt \
  [baseline flags] \
  --use-fused-zloss \
  --ak-swiglu triton_op \
  --ak-trust-cap 0.02
```

### Gate

```text
if divergence occurs shortly after enabling, issue is not purely early warmup
if stable, delayed enable is production-useful even if early steps remain native
```

### Caution

Make sure compile graph warmup does not pollute throughput measurement immediately after resume. Exclude first N steps after switch from tok/s statistics.

---

## 19. Experiment E11 — Layer-selective hidden kernels

### Purpose

Find whether the unstable effect localizes by depth or block type.

### Add flag

```bash
--ak-swiglu-layers 0,1,2,3
--ak-swiglu-layers 7,8,9,10,11,12,13
--ak-swiglu-layers all-but-attention-adjacent
--ak-swiglu-layers first-half
--ak-swiglu-layers second-half
```

For OdinHalo, allow shared-layer and iteration-aware toggles if the forward structure supports it:

```bash
--ak-swiglu-shared-layers 0,1,2
--ak-swiglu-iters 0
```

### Test plan

```text
native + fused loss
native + fused loss + first-half SwiGLU
native + fused loss + second-half SwiGLU
native + fused loss + all except known runaway layers
```

### Metrics

```text
tok/s
loss
per-layer activation maxabs
per-layer update ratios
GradScaler scale
```

### Gate

A partial enable is useful if:

```text
it adds ≥2–3% step-time improvement on top of fused loss
it passes 1000–2000 steps
it does not produce localized activation growth
```

---

## 20. Experiment E12 — NorMuon performance optimization

### Purpose

Recover safe speed from optimizer overhead while preserving native model math.

### Opportunities

The repo already improved NorMuon by moving NS to fp16 in an earlier session, but step profile still reports ~12.5% optimizer cost. Investigate:

```text
1. group similar 2D matrices and batch NS operations where shapes match
2. fuse normalization + NS polynomial steps if memory traffic dominates
3. fuse SPECTRA post-processing and update apply
4. avoid CPU .item() for optimizer diagnostics
5. reduce unnecessary dtype casts / contiguous copies
6. avoid repeated allocation of scratch tensors
7. preallocate work buffers per param shape
```

### Validation

Optimizer changes must compare exact or near-exact update tensors:

```text
same params, same grads
old optimizer update
new optimizer update
cosine ≥ 0.9999 or explain intended difference
max rel_err bounded
training loss parity through 1000–2000 steps
```

### Success gate

```text
≥3% total step-time gain with no learning regression
```

If ≥5–6% is achieved, then fused loss + NorMuon optimization probably reaches 40k for OdinFlat-like baseline.

---

## 21. Experiment E13 — NS steps and update scale

### Purpose

Find a faster and possibly more stable NorMuon setting.

### Grid

```text
ns_steps = 5 current
ns_steps = 4
ns_steps = 3
adjust_lr_fn = current/original
adjust_lr_fn = match_rms_adamw if supported or implement equivalent
extra_update_scale = 1.0, 0.5, 0.25 for selected params
```

PyTorch's Muon docs describe `ns_steps`, Newton-Schulz coefficients, and learning-rate adjustment options including an RMS-matching style. The details of NorMuon may differ, but the idea is relevant: orthogonalized update scaling is not a fixed law; it is a tunable part of the optimizer.

### Gate

```text
short unit update comparison
then 1000-step learning parity
then 2000-step stability
```

Do not accept speed if validation/BPB degrades materially.

---

## 22. Experiment E14 — Synchronization/logging reduction

### Purpose

Remove avoidable stream synchronizations, especially `.item()` calls inside hot paths.

### Audit list

Search for:

```text
.item()
float(tensor)
int(tensor)
tensor.cpu()
tensor.tolist()
print(tensor)
logging of tensor scalars
GradScaler scale reads every step
activation maxabs reads every step
optimizer diagnostics reads every step
```

### Strategy

```text
log every N optimizer steps
keep tensor counters on device when possible
defer CPU readback
batch scalar reads into one logging point
use debug mode for detailed telemetry
```

### Gate

```text
≥1% speedup is worthwhile as a final-mile optimization
must preserve failure detection for scaler collapse and nonfinite grads
```

---

## 23. Experiment E15 — hipBLASLt epilogue exploration

### Purpose

Check whether vendor GEMM epilogues can provide a safe activation fusion without custom matmul.

### Context

AMD hipBLASLt docs list epilogues such as ReLU, GELU, Swish/SiLU, and bias variants. However, current SwiGLU requires:

```text
w_gate_up output -> split into gate/up halves -> silu(gate) * up -> w_down
```

A plain Swish epilogue on the whole GEMM output does not compute the pairwise product with the up half. Therefore this is speculative.

### What to test

```text
1. Is hipBLASLt available and performant on gfx1151 in this environment?
2. Can it apply Swish to only one half of the output or use aux output cheaply?
3. Does PyTorch/Inductor already use hipBLASLt epilogues for relevant shapes?
4. Is there an exposed Python route without large integration cost?
```

### Gate

Only continue if an isolated benchmark shows a plausible end-to-end path. Otherwise, do not spend a sprint here.

---

## 24. Recommended execution order

### Phase A — make experiments interpretable

1. Add granular optimization flags.
2. Add update telemetry.
3. Add replay bundle on failure.
4. Run E0 native baseline with telemetry.

**Exit gate:** native baseline distributions are known.

### Phase B — loss path first

1. Run E1 native + fused z-loss only.
2. If speed is inadequate, prototype E2 factorized fused CE+z-loss.
3. Run E3 parity stress suite.

**Exit gate:** either:

```text
loss path gives ≥36k–38k and passes 2000 steps
```

or:

```text
loss path is not the expected speed target; return to profile and reassess
```

### Phase C — optimizer path

1. Run E12 NorMuon performance profiling/optimization.
2. Run E13 NS-step/scale variants if profiling suggests it.
3. Keep native hidden model path.

**Exit gate:** fused loss + optimizer wins approach or exceed 40k without hidden-layer replacements.

### Phase D — hidden kernels only if still needed

1. Run E4 integration comparison.
2. Run E6 precision/autocast variants.
3. Run E5 one replacement family at a time.
4. Run E11 layer-selective if all-layer fails.
5. Run E10 delayed enable if early-window instability persists.

**Exit gate:** a hidden kernel enters the production stack only after 1000–2000-step loss/stability pass.

### Phase E — stack only proven wins

Potential stack:

```text
native hidden path
+ fused/factorized CE+z-loss
+ optimized NorMuon update path
+ reduced sync/logging
+ optional selective/delayed compiler-visible SwiGLU
```

Run final 2000-step gate and an eval scorecard before full pretraining.

---

## 25. Concrete candidate stacks

### Stack 1 — Loss-first, lowest risk

```text
native hidden path
+ fused z-loss / fused CE+z-loss
+ no hidden autokernel replacements
```

Expected:

```text
31.3k → 36k–38k for OdinFlat-like baseline
```

This should be the first serious route.

### Stack 2 — Loss + optimizer

```text
native hidden path
+ fused/factorized CE+z-loss
+ NorMuon grouped/fused/update optimization
+ reduced sync logging
```

Expected:

```text
31.3k → 39k–41k if z-loss target and optimizer target both yield partial wins
```

This is the most plausible stable 40k stack.

### Stack 3 — Loss + selective hidden kernels

```text
native hidden path mostly
+ fused/factorized CE+z-loss
+ SwiGLU optimized only on safe layer subset
+ post-update trust cap on affected params
```

Expected:

```text
useful if optimizer work underdelivers
```

Risk:

```text
more moving parts; needs careful layer telemetry
```

### Stack 4 — Delayed optimized hidden path

```text
native steps 0–500 or 0–1000
then enable fused loss + selected hidden kernel
with trust cap or w_gate_up staging
```

Expected:

```text
production-useful if instability is warmup-local
```

Risk:

```text
checkpoint/resume and compile transition complexity
```

### Stack 5 — Full autokernel replacement

```text
--optimize-kernels full stack
```

Current recommendation:

```text
avoid until every component has passed isolated and stacked gates
```

---

## 26. Acceptance gates

### 26.1 Kernel/unit gates

Every training-path kernel must pass:

```text
no raw pybind training path
custom op or autograd function registered
opcheck where applicable
gradcheck or finite-difference check where feasible
fwd+bwd parity at production shapes
fp16/autocast tests
ignore_index/masking tests for loss kernels
DDP-compatible shape tests
```

### 26.2 Training gates

A candidate stack must pass:

```text
200-step smoke: no gross issues
1000-step stability: no scaler collapse, no activation runaway
2000-step quality: loss and eval scorecard within seed variance
frozen-param audit: zero newly frozen params
throughput: sustained median after compile/warmup, not peak
```

Suggested throughput measurement:

```text
exclude first 100 optimizer steps
exclude first 20 steps after feature switch or resume
report median tok/s over last 200 steps
report p10/p90 tok/s to catch stalls
```

### 26.3 Stability gates

Fail immediately or dump forensics if:

```text
GradScaler scale < 1.0 or repeated scale backoffs
nonfinite grad norm
any large hidden parameter update_to_weight_ratio > configured hard cap
fp16_headroom < 10x sustained
activation maxabs grows exponentially across intervals
loss spike > predefined threshold vs baseline window
```

### 26.4 Quality gates

Do not use only training loss. Include:

```text
validation BPB / scorecard at comparable checkpoints
loss curve shape
gradient/update health
parameter freeze audit
sample sanity if normal for project
```

---

## 27. Implementation notes for granular flags

### 27.1 Do not overload `--optimize-kernels`

Keep `--optimize-kernels` as a legacy/all-in flag if needed, but add explicit subflags. Suggested design:

```python
parser.add_argument("--ak-loss", choices=["native", "fused_zloss", "fused_ce_zloss", "factorized_ce_zloss"], default="native")
parser.add_argument("--ak-swiglu", choices=["native", "hip_torch_bwd", "hip_hip_bwd", "triton_fn", "triton_op"], default="native")
parser.add_argument("--ak-rmsnorm", choices=["native", "hip", "triton"], default="native")
parser.add_argument("--ak-rope", choices=["native", "hip", "triton"], default="native")
parser.add_argument("--ak-swiglu-layers", type=str, default="")
parser.add_argument("--ak-enable-after-step", type=int, default=0)
parser.add_argument("--ak-trust-cap", type=float, default=0.0)
parser.add_argument("--ak-w-gate-up-lr-mult", type=float, default=1.0)
parser.add_argument("--ak-update-telemetry", action="store_true")
parser.add_argument("--ak-replay-on-failure", action="store_true")
```

### 27.2 Preserve reproducibility in feature switches

If enabling kernels mid-run:

```text
save and restore RNG states
save dataloader position
save optimizer and scaler states
record compile mode and graph settings
clear or warm compile caches deterministically
```

### 27.3 Make feature state visible in logs

Every log header should print:

```text
git SHA
model
compile mode
ak-loss mode
ak-swiglu mode
ak-rmsnorm mode
ak-rope mode
ak-layer mask
HIP backward enabled?
trust cap
w_gate_up scaling
GradScaler config
```

---

## 28. Specific notes on `_torch_ops.py`

The current `kernels/hip/_torch_ops.py` is useful because it already provides a key A/B hook:

```text
AUTOKERNEL_NO_BWD_HIP=1
```

It also states that backward falls back to PyTorch during compile tracing and that disabling HIP backward may let Inductor fuse PyTorch fallback backward ops. Use this as an explicit experiment axis:

```text
HIP forward + PyTorch fallback backward
HIP forward + HIP backward
```

For each custom op, ensure:

```text
register_fake exists
register_autograd exists
register_autocast considered/tested
fallback backward uses fp32 intermediates where native path does
HIP backward has production-shape parity tests
```

---

## 29. Specific notes on Triton SwiGLU

The current Triton fused SwiGLU implementation is a valid harness piece but not a production win yet.

Issues to examine:

1. It uses `torch.autograd.Function`; PyTorch docs warn `register_autograd` with `triton_op` is preferable for composability with `torch.compile`.
2. It makes inputs contiguous. That copy may matter end-to-end.
3. It only fuses the elementwise activation/product, not the surrounding GEMMs.
4. It stores outputs/gradients as fp16 after fp32 intermediate math. Need parity against native autocast behavior.
5. Ship-gate bench says it does not beat autograd-safe HIP by enough.

Recommended next implementation:

```text
rewrite as torch.library.triton_op + wrap_triton
register autograd formula
wrap backward Triton kernels as triton_op too
compare graph breaks and end-to-end training
```

Do not assume this will be the 40k path. Treat it as a diagnostic for compiler visibility.

---

## 30. Why 200-step probes are not enough

The repo already has the answer:

- The broken OdinFlat path looked plausible at step 200.
- The loss gap widened by step 2000.
- Phase G was better through step 700 and then diverged.

Therefore:

```text
200 steps = syntax/gross smoke only
1000 steps = initial stability
2000 steps = minimum learning gate for this class of bug
full pretraining = only after 2000-step gate passes
```

Use shorter runs only to kill bad candidates quickly, not to approve good ones.

---

## 31. Things not to spend much time on initially

### 31.1 Full custom matmul kernels

Do not start by replacing dense GEMMs. gfx1151 lacks matrix cores; rocBLAS/Inductor is the default benchmark to beat. Custom matmul fusion may become a research project, but it is not the fastest route to a stable 40k.

### 31.2 QKV fusion as the main speed target

The step profile estimates attention forward at ~7.5% of total step wall and QKV fusion upside around ~1.5% total wall. That is below the scale needed for 40k.

### 31.3 Isolated kernel microbenchmarks as ship evidence

They are necessary but insufficient. The repo has already shown isolated elementwise speed can disappear end-to-end under correct autograd.

### 31.4 Reverting Phase B

Do not ship a model with a known silent freeze to recover throughput. It is not a valid solution.

---

## 32. Expected failure modes and diagnosis table

| Symptom | Likely cause | First diagnostic |
|---|---|---|
| Great tok/s, loss drifts after 1000+ steps | hidden freeze or wrong gradient | frozen-param audit + gradient parity |
| No speedup from safe kernel | custom-op overhead/compiler opacity | compare `triton_op`, graph breaks, dispatch count |
| Divergence near warmup completion | LR/update scale + fp16 boundary | update telemetry + delayed enable |
| Divergence shortly after enabling kernels from checkpoint | kernel numerics or optimizer interaction | replay same batch native vs optimized |
| GradScaler collapse without huge raw grad norm | post-NorMuon update spike or activation overflow | update_to_weight_ratio + activation maxabs |
| Fused loss immediate loss mismatch | denominator/masking/softcap bug | loss stress suite |
| Fused loss stable but no speed | fusion not on critical path | profiler with shape/call attribution |
| Layer-specific activation explosion | unstable layer subset | layer-selective disable/enable |
| update cosine poor only after NS | NorMuon sensitivity | NS steps/scale/trust cap |

---

## 33. Recommended first five engineering tasks

### Task 1 — Add experiment flags and logging header

Create granular flags for loss, SwiGLU, RMSNorm, RoPE, backward mode, layer mask, delayed enable, trust cap, and update telemetry.

**Deliverable:** one PR/commit with CLI parsing, logging, and no behavior change by default.

### Task 2 — Add update telemetry to NorMuon

Add debug logging for raw grad, post-NS, post-SPECTRA, final update, and update/weight ratio.

**Deliverable:** native baseline E0 report with distributions.

### Task 3 — Run native + fused z-loss only

No hidden kernels. Use current fused z-loss if available.

**Deliverable:** 1000-step and preferably 2000-step comparison:

```text
native vs native+fused_zloss
tok/s
loss
GradScaler
activation stats
update ratios
```

### Task 4 — Build loss parity stress suite

Test every masking/softcap/z-loss edge case.

**Deliverable:** test file with native vs fused comparisons and production shapes.

### Task 5 — Compare custom_op vs triton_op SwiGLU

Rewrite or prototype SwiGLU as `torch.library.triton_op` and compare graph behavior/end-to-end throughput.

**Deliverable:** end-to-end 500-step smoke plus microbench, not a ship decision yet.

---

## 34. What a good result would look like

A strong outcome after the first cycle:

```text
E0 native baseline:
  31.3k tok/s, stable, update-ratio distribution known

E1 native + fused zloss:
  36.5k–38.0k tok/s, stable through 2000 steps

E12 NorMuon/sync optimization:
  +3% to +5% step-time improvement on top

Final stack:
  39.5k–41.0k tok/s, no hidden kernels, learning parity
```

That would be the cleanest 40k solution.

A weaker but still useful outcome:

```text
fused loss reaches 36k
NorMuon optimization reaches 38k
selective/delayed Triton SwiGLU adds final 2k
```

A bad outcome:

```text
only full --optimize-kernels reaches 40k
but it diverges or degrades loss
```

Do not ship the bad outcome.

---

## 35. Notes on documentation cleanup

The repo has fast-moving docs and some source docs may conflict by date or phase. For engineering clarity, update docs after the next experiment cycle:

1. Make `STATUS.md` the source of truth for production run flags.
2. Mark old `--optimize-kernels` examples as historical if they predate Phase B or Phase G.
3. Add a section called “40k throughput recovery plan” with this guide's experiment order.
4. Add a rule: no full-run launch with `--optimize-kernels` unless each replacement family in the stack has passed the 2000-step gate.

---

## 36. Final assessment

The 40k tok/s target is not unrealistic. But the path probably does **not** look like the old `--optimize-kernels` path. The old path's headline speed was contaminated by missing backward work. The autograd-safe version is the correct foundation, but it exposes the true challenge: the system has to be fast under full-parameter learning.

The best route is:

```text
1. Reclaim the loss path.
2. Optimize NorMuon/update overhead.
3. Reduce sync overhead.
4. Only then add hidden-layer kernels through granular, compiler-visible, parity-tested routes.
5. Stabilize with post-update trust caps or parameter-specific staging if needed.
```

The first real experiment should be boring and decisive:

```text
native hidden path + fused z-loss only + full telemetry + 2000-step gate
```

If that works, the project is already close to 40k without touching the fragile hidden-layer replacements. If it fails, the failure will still be informative because it isolates the loss path from the rest of autokernel.

---

## Appendix A — Minimal runbook

### A.1 Before running

```text
1. git status clean
2. record git SHA
3. confirm current STATUS.md ship decision
4. confirm model recipe and dataset
5. enable telemetry flags
6. run frozen-param preflight
7. run loss parity unit tests if using fused loss
8. warm compile separately if needed
```

### A.2 During run

Watch:

```text
tok/s after warmup
loss vs baseline window
GradScaler scale
nonfinite grad events
update_to_weight_ratio spikes
activation fp16 headroom
```

### A.3 On failure

```text
1. save replay bundle
2. stop if scaler collapse or nonfinite persists
3. replay same batch native vs optimized
4. compare raw grads and post-NorMuon updates
5. classify failure using table in section 32
```

### A.4 After run

Publish a short result document with:

```text
configuration
feature flags
git SHA
throughput median/p10/p90
loss table at 100/200/500/1000/2000
GradScaler summary
activation summary
update-ratio summary
frozen-param audit
verdict
next action
```

---

## Appendix B — Example result template

```markdown
# Experiment: native + fused_zloss only

## Config
- git SHA:
- model:
- dataset:
- block size:
- batch / accum / DDP:
- compile mode:
- flags:
- AK modes:

## Throughput
| window | median tok/s | p10 | p90 |
|---|---:|---:|---:|
| steps 100-300 | | | |
| steps 300-1000 | | | |
| steps 1000-2000 | | | |

## Loss
| step | native baseline | candidate | delta |
|---:|---:|---:|---:|
| 100 | | | |
| 200 | | | |
| 500 | | | |
| 1000 | | | |
| 2000 | | | |

## Stability
- GradScaler min/max:
- skipped steps:
- nonfinite grad events:
- fp16 headroom min:
- largest activation maxabs:
- largest update_to_weight_ratio:

## Gradient / update parity
- frozen params:
- d_hidden cosine:
- d_head cosine:
- worst param update cosine:

## Verdict
PASS / FAIL / INCONCLUSIVE

## Next action
```

---

## Appendix C — Quick hypothesis-to-experiment map

| Hypothesis | Experiments |
|---|---|
| loss path is main route | E1, E2, E3 |
| custom op opacity erases speed | E4, E5 |
| autocast/precision mismatch destabilizes | E6 |
| NorMuon amplifies differences | E7, E8, E9, E13 |
| warmup-local instability | E10 |
| layer-local instability | E11 |
| optimizer overhead can recover speed safely | E12, E13 |
| sync overhead is final-mile | E14 |
| vendor epilogues might help | E15 |

---

## Appendix D — Safety rules for engineering agent

1. Never use raw pybind HIP outputs in a training path unless wrapped in autograd machinery.
2. Never infer learning quality from 200-step loss only.
3. Never treat tok/s as valid if frozen-param audit fails.
4. Never stack unvalidated replacements.
5. Always inspect unscaled gradients after AMP unscale, not scaled gradients.
6. Always compare post-optimizer updates, not just raw gradients, when using NorMuon.
7. Always keep native baseline runs close in time to candidate runs because compile/cache/system conditions can drift.
8. Always record exact feature flags and git SHA.
9. Prefer exact loss/optimizer fusions over hidden-representation approximations.
10. Treat 40k as a budget target, not as permission to ship degraded learning.
