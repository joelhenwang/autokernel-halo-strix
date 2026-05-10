# Autokernel Halo Strix — Revised 40k tok/s Engineering Handoff

**Document version:** v2 revised assessment  
**Date:** 2026-05-10  
**Prepared for:** engineering agent working on `joelhenwang/autokernel-halo-strix`  
**Primary objective:** reach approximately **40k aggregate tok/s** across the two Strix Halo machines while preserving full-parameter learning quality.  
**Secondary objective:** turn the current ambiguous `--optimize-kernels` situation into a granular, measurable, reproducible performance/stability program.

---

## 0. Executive summary

The goal is not simply to make `--optimize-kernels` pass. The goal is:

> **40k aggregate tok/s with a loss curve, GradScaler behavior, activation statistics, frozen-parameter audit, and per-parameter update dynamics indistinguishable from the validated native full-parameter training recipe.**

The old 40k-class throughput result is not sufficient evidence that the platform can safely train at 40k, because the fast path was entangled with the silent-freeze bug: raw pybind HIP calls returned tensors with `grad_fn=None`, severing autograd and freezing a large fraction of parameters. The repo's current status correctly locks Sprint 3A/Sprint 3B to **drop blanket `--optimize-kernels`** until a correct stack passes long-horizon gates. That decision should remain in force until a replacement stack passes the gates in this document.

The revised assessment after reading the engineer-agent response is:

1. **The plan should start with measurement and backward/DDP profiling, not with loss fusion alone.**  
   The earlier loss-first recommendation was directionally useful but underweighted the engineer team's correction: the largest ambiguous bucket is compiled backward / backward graph structure, not just z-loss.

2. **40k requires about a 27.8% throughput lift from the validated 31.3k baseline.**  
   With 131,072 tokens per optimizer step, 31.3k tok/s is about 4.19 seconds per optimizer step. 40k tok/s is about 3.28 seconds. The team must recover roughly **0.91 seconds per optimizer step**, or about **21.8% step-time reduction**.

3. **No single safe lever is likely to give the full improvement.**  
   The plausible path is a stack:

   ```text
   baseline native full-parameter path
   + fused z-loss / chunked CE z-loss, if stable
   + DDP accumulation/overlap fixes
   + graph-break and scalar-sync cleanup
   + compiled autograd or backward graph improvement, if feasible
   + batch=32/accum=4 if memory can be made to fit
   + NorMuon implementation/update optimization
   + only then a compiler-visible hidden-layer kernel path
   ```

4. **The mainline 40k route should be “safe-stack first, hidden-kernel recovery second.”**  
   Do not let SwiGLU kernel work dominate the project. The repo's own isolated benchmark shows Triton and autograd-safe HIP SwiGLU are already near parity: HIP is 1.45x eager fwd+bwd, Triton is 1.43x eager fwd+bwd, and Triton is 0.986x of HIP. That means the standalone elementwise kernel is already near its local ceiling. The remaining value of Triton is **compiler visibility**, not raw kernel tuning.

5. **Compiled autograd belongs in the first serious experiment tranche.**  
   PyTorch's compiled autograd can capture a larger runtime backward graph than normal AOTAutograd, especially when forward graph breaks fragment the backward graph. It also has real risks: cache lookup overhead, recompiles, graph breaks, and incomplete feature compatibility. Treat it as a feasibility gate, not an assumed win.

6. **DDP overlap and accumulation correctness must be verified immediately.**  
   With two Strix Halo machines over Thunderbolt 4 and gradient accumulation of 8, accidental allreduce on every microstep would be a serious performance bug. Even if `no_sync()` is already correct, bucket sizing and `gradient_as_bucket_view=True` are cheap stackable opportunities.

7. **NorMuon update telemetry is mandatory before blaming kernels.**  
   Global grad norm is not enough. NorMuon transforms raw gradients through momentum, Newton-Schulz orthogonalization, SPECTRA, shape scaling, and LR multiplication. The failure may be visible only in the **post-transform update/weight ratio**.

8. **The correct hidden-kernel recovery path is delayed/controlled enablement, not blanket enablement.**  
   If the optimized path is worth recovering, test it through warm-start, dtype/autocast parity, PyTorch-backward fallback, post-NorMuon trust caps, and `w_gate_up` staging.

### Practical expected outcomes

| Stack | Probability | Expected aggregate tok/s | Risk profile |
|---|---:|---:|---|
| Current validated native baseline | already proven | ~31.3k | stable but slow |
| Safe quick stack: fused z-loss + DDP/sync fixes | high | ~33k-35k | low model risk |
| Safe main stack: quick stack + compiled autograd or batch=32 | medium | ~36k-38.5k | moderate runtime risk |
| Full 40k stack: main stack + NorMuon optimization or recovered compiler-visible hidden kernels | medium-low | ~39k-41k | higher engineering/stability risk |
| Blanket `--optimize-kernels` without new gates | unacceptable | unknown | already failed Phase C/G |

---

## 1. Source context and current state

### 1.1 Hardware and topology

The target hardware is AMD Strix Halo / Ryzen AI Max+ 395-class APU hardware. AMD lists this SKU as **Strix Halo**, with 16 Zen 5 CPU cores, Radeon 8060S Graphics, 40 graphics cores, up to 128 GB memory, and **256-bit LPDDR5x-8000** system memory. On this class of APU, memory is physically shared between CPU and GPU; ROCm's RDNA3.5 APU documentation describes gfx1150/gfx1151/gfx1152 memory through GPU Virtual Memory rather than a separate discrete VRAM pool.

Correct practical implication:

```text
256-bit bus = 32 bytes per transfer
LPDDR5x-8000 ≈ 8e9 transfers/s
32 bytes × 8e9 transfers/s ≈ 256 GB/s theoretical per machine
```

The repo's internal brief cites approximately 240 GB/s effective/observed bandwidth, which is consistent with this order of magnitude. Use **~240-256 GB/s per machine** as the practical bandwidth frame. If quoting both machines together, ~480-512 GB/s aggregate theoretical is reasonable, but do not use 512 GB/s as the single-machine roof.

Hardware consequences:

- No MFMA / no matrix cores for this gfx1151 target, so custom HIP GEMMs are unlikely to beat rocBLAS for transformer GEMM shapes.
- Unified LPDDR5x memory is large but much lower bandwidth than datacenter HBM.
- The 6 MB-ish L2 cache and shared-memory topology make extra tensor passes expensive.
- Pointwise/reduction fusion is still valuable because it reduces LPDDR traffic.
- Dense matmul replacement is a poor target unless it is vendor-backed or extremely specialized.

### 1.2 Model and training context

The repo is training custom hybrid language models, primarily:

- **OdinFlat**: about 122M parameters, validated Sprint 3A baseline around **31.3k aggregate tok/s** without `--optimize-kernels`.
- **OdinHalo**: smaller looped/halo variant, separate stability behavior but same broad issue.

The training stack includes:

- fp16 + GradScaler only; bf16 is not a practical path on this SKU per repo constraints.
- NorMuon for 2D hidden parameters.
- SPECTRA post-clip.
- μP for the OdinFlat recipe.
- z-loss and attention/logit softcaps.
- two-machine DDP over Thunderbolt 4 / gloo.
- `TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` already part of the baseline.

### 1.3 Autokernel silent-freeze incident

The project discovered that the original `--optimize-kernels` path used raw pybind11 HIP calls inside module forward passes. Those calls returned tensors with no autograd history (`grad_fn=None`), causing upstream parameter groups to be silently frozen.

The repo's stability brief and status docs describe the core incident:

- OdinFlat froze roughly 23% of parameters in the affected path.
- A large affected area was the FFN gate/up path, especially `w_gate_up`.
- The fast path produced plausible descending loss early, but long-horizon quality regressed.
- Phase B rewired the unsafe replacements through autograd-safe paths such as `torch.library.custom_op` plus `register_autograd` or native PyTorch fallback.
- Static/dynamic audits now validate gradient flow at probe scale.

This is the most important safety lesson:

> **Never count a throughput improvement unless the frozen-parameter audit, gradient-flow audit, and long-horizon learning curve pass.**

### 1.4 Post-fix divergence

After Phase B, the optimized path became autograd-correct but unstable at production recipes.

Observed failures:

- **OdinFlat Phase C:** three post-fix `--optimize-kernels` attempts diverged around step 200-250, including a DDP run at the production effective batch.
- **OdinHalo Phase G:** post-fix optimized path tracked better than the pre-fix frozen path through step 700, then diverged around step 750-800 at lower lr_2d.

This is not evidence that Phase B should be reverted. It is evidence that the old speedup partly came from training less of the model, and the correct path now exposes a genuine optimizer/numerics/fp16 stability boundary.

### 1.5 Current production decision

Current repo status locks these decisions:

```text
Sprint 3A OdinFlat: do not use blanket --optimize-kernels.
Sprint 3B OdinHalo: do not use blanket --optimize-kernels.
--use-fused-zloss: opt-in, not yet validated end-to-end.
Triton fused SwiGLU: infrastructure, not a ship feature yet.
```

This document does not overturn those locked decisions. It gives the engineering agent a plan for finding a safe higher-throughput stack.

---

## 2. Revised performance arithmetic

### 2.1 Baseline and target

Validated OdinFlat baseline:

```text
aggregate throughput ≈ 31.3k tok/s
microbatch = 16
block = 512
accum_steps = 8
world_size = 2
```

Tokens per optimizer step:

```text
16 × 512 × 8 × 2 = 131,072 tokens / optimizer step
```

Baseline step time:

```text
131,072 / 31,300 ≈ 4.19 seconds / optimizer step
```

Target step time at 40k tok/s:

```text
131,072 / 40,000 ≈ 3.28 seconds / optimizer step
```

Required savings:

```text
4.19 - 3.28 ≈ 0.91 seconds / optimizer step
1 - 31.3/40 ≈ 21.8% step-time reduction
40/31.3 ≈ 1.278x throughput lift
```

### 2.2 Step-profile interpretation

The engineering response quotes the current profile roughly as:

| Bucket | Approximate share | Interpretation |
|---|---:|---|
| `aten::mm` across fwd/bwd/NorMuon | ~37.85% | largely real GEMM work; not easily addressable on gfx1151 |
| compiled backward region | ~35.3% | major ambiguous bucket; must be call-site attributed |
| compiled forward region | ~16.2% | already compiled baseline |
| NorMuon optimizer step | ~12.5% | addressable through implementation or ns-step experiments |
| separate z-loss `logsumexp` forward | ~11.1% | redundant with CE-style pass, but not all removable in net terms |
| `LogsumexpBackward` | ~5.6% | z-loss backward target |
| HyPE causal-conv backward | ~4.5% | possible niche target after attribution |
| DDP allreduce estimate | ~4-5% | addressable through accumulation correctness and overlap |

Do not add these percentages naively. The profile buckets overlap. For example, `aten::mm` can appear inside compiled forward/backward regions.

### 2.3 What is really addressable?

Non-addressable or hard-to-address:

- standard GEMMs already handled by rocBLAS / Inductor.
- architectural FLOPs necessary for training.
- some compiled backward time that is just real matmul/backward math.

Addressable:

- redundant z-loss pass.
- graph-break-induced fragmentation.
- DDP communication scheduling/copy overhead.
- hot-path scalar syncs.
- optimizer Python/dispatch overhead.
- NorMuon NS/SPECTRA/update apply implementation.
- batch/microstep inefficiency.
- custom-op opacity around hidden kernels.

The engineering implication is sharp:

> **40k is possible only if several addressable deltas stack. It is not plausible from one isolated kernel improvement.**

---

## 3. Revised priority order

The revised order is:

```text
Phase 0: Measurement and isolation
  0.1 Backward profile with call-site attribution
  0.2 DDP communication trace and no_sync verification
  0.3 Graph-break + recompile inventory
  0.4 Sync-point audit
  0.5 NorMuon per-parameter update telemetry
  0.6 Granular --ak-* flags

Phase 1: Quick safe wins
  1.1 Native hidden path + fused z-loss only
  1.2 Native hidden path + chunked CE z-loss path
  1.3 DDP bucket/no_sync/gradient_as_bucket_view sweep
  1.4 Hot-path scalar-sync cleanup
  1.5 Quick graph-break fixes where obvious

Phase 2: Big runtime/compiler levers
  2.1 Compiled autograd feasibility
  2.2 torch.library.triton_op SwiGLU integration test
  2.3 NorMuon implementation optimization / compiled optimizer exploration
  2.4 batch=32/accum=4 enablement if memory permits

Phase 3: Divergence recovery for optimized hidden path
  3.1 Replay bundle and replay_step.py
  3.2 Warm-start native -> optimized path
  3.3 Post-NorMuon trust cap
  3.4 w_gate_up update-scale staging
  3.5 custom-op autocast / fp32-island parity tests
  3.6 layer-selective kernels only if replay localizes the failure

Phase 4: Stack integration and ship gate
  4.1 Assemble best no-hidden-kernel stack
  4.2 Assemble best hidden-kernel recovery stack, if any
  4.3 2000-step gate minimum
  4.4 Update STATUS.md, CONSTRAINTS.md, AGENTS.md
```

This order deliberately moves **compiled autograd, DDP, and graph-break profiling earlier** than the previous guide.

---

## 4. Hypothesis ranking

### H1 — Backward graph fragmentation / compiled backward inefficiency is a major stackable performance lever

**Claim:** A significant part of the missing throughput is in the backward graph structure, not the forward kernels. Forward graph breaks, custom-op boundaries, hooks, Python state, and runtime autograd fragmentation may be reducing the effectiveness of the compiled training step.

**Evidence:** The step profile has a large compiled-backward bucket. PyTorch's compiled autograd documentation says ordinary `torch.compile` captures backward only partially, and forward graph breaks can lead to backward graph breaks. Compiled autograd can capture a larger backward graph at runtime, but may introduce overhead and recompiles.

**Experiments:**

- Backward call-site attribution.
- `TORCH_LOGS=graph_breaks,recompiles,compiled_autograd`.
- Native baseline with and without compiled autograd.
- Native + fused zloss with and without compiled autograd.
- DDP native + compiled autograd.
- Triton-op SwiGLU + compiled autograd.
- custom-op HIP path + compiled autograd.

**Expected gain:** 0-8%. Treat >10% as possible but not expected.

**Failure predictions:**

| Observation | Interpretation |
|---|---|
| compiled autograd improves single-node but not DDP | DDP hooks/buckets conflict or prevent useful capture |
| many recompiles | dynamic state, list mutation, variable shapes, or hooks need fixing first |
| no gain and compiled backward is mostly GEMM | backward bucket is real compute, not fragmentation |
| custom-op path fails but native succeeds | custom-op/register_autograd opacity or incompatibility |
| triton_op succeeds where custom_op fails | compiler visibility matters; migrate training path toward triton_op |

### H2 — DDP accumulation and communication overlap can recover a nontrivial stackable 2-5%

**Claim:** With two machines over TB4/gloo and accumulation=8, communication scheduling is a stackable opportunity. The first mandatory check is that allreduce occurs only on the final accumulation microstep.

**Experiments:**

- Verify `no_sync()` on microsteps 0 through accum_steps-2.
- Count allreduces per optimizer step.
- Sweep `bucket_cap_mb`: 8, 16, 25, 50, 100, 200.
- Enable `gradient_as_bucket_view=True`.
- Test `static_graph=True` only if unused-param behavior permits.
- Re-benchmark gloo vs RCCL if RCCL is available and reliable.

**Expected gain:**

- Huge if `no_sync()` is missing or broken.
- 1-5% if buckets/copies are suboptimal.
- Near-zero if communication is already fully hidden.

**Failure predictions:**

| Observation | Interpretation |
|---|---|
| allreduce count equals accumulation steps | serious accumulation bug; fix immediately |
| smaller buckets improve speed | overlap was delayed by large buckets |
| larger buckets improve speed | too many tiny reductions / overhead |
| `gradient_as_bucket_view` only reduces memory | still valuable if it enables batch=32 |
| no DDP setting changes speed | communication already hidden or below noise |

### H3 — Fused z-loss / chunked CE z-loss is a safe but limited win

**Claim:** The loss path is a low-risk performance pillar, but the engineer response correctly reduces expectations. CE requires a logsumexp-like computation; the target is eliminating the **second** z-loss pass, not all CE softmax work.

**Experiments:**

- Native baseline, no fused z-loss.
- Native hidden path + `--use-fused-zloss` only.
- Native hidden path + chunked CE with integrated `z_loss_weight`.
- Native hidden path + chunked CE without z-loss for control.

**Expected gain:** 3-8% likely, 8-14% possible if the measured separate pass was particularly expensive. Do not expect 16.7% net throughput.

**Failure predictions:**

| Observation | Interpretation |
|---|---|
| stable + 5-8% | keep as a safe pillar |
| stable + 1-3% | keep only if it does not complicate stack |
| divergence only with fused path | z-loss gradient, denominator, mask, softcap, or dtype mismatch |
| CE scalar parity but gradient drift | backward formula or denominator mismatch |

### H4 — NorMuon post-transform update dynamics, not raw gradients, drive the divergence

**Claim:** The divergence may not be visible in raw grad norm. NorMuon transforms gradients through momentum, Newton-Schulz orthogonalization, SPECTRA, shape/LR scaling, and final update. The problematic signal may be `||actual_update|| / ||weight||`, especially for `w_gate_up`.

**Experiments:**

- Add per-parameter telemetry before new long runs.
- Compare native vs optimized path update cosines and update/weight ratios.
- Add post-NorMuon trust cap.
- Add `w_gate_up` update scale ramp.
- Try `ns_steps=3` vs `ns_steps=5` only after telemetry.
- Try Muon-like `match_rms_adamw` scaling if compatible with repo NorMuon.

**Expected gain:** Stabilization rather than speed directly; possible optimizer speed if fewer NS steps pass parity.

**Failure predictions:**

| Observation | Interpretation |
|---|---|
| update ratio spikes before GradScaler collapse | trust cap / w_gate_up staging likely helps |
| raw grad norm normal but update ratio spikes | NorMuon transform is amplifying geometry |
| trust cap fires constantly from step 1 | LR/SPECTRA/scale is globally too aggressive |
| trust cap never fires but divergence persists | likely activation/logit overflow or kernel dtype issue |
| ns_steps=3 stable | optimizer speed win possible |
| ns_steps=3 destabilizes | optimize implementation, not algorithmic steps |

### H5 — custom-op dtype/autocast boundaries differ from native Inductor

**Claim:** The post-fix custom ops can be autograd-correct but numerically different from native Inductor. In fp16, tiny differences in the SwiGLU gate or backward can be magnified by NorMuon.

**Experiments:**

- Check `torch.library.register_autocast` behavior for custom ops.
- Compare custom op no-autocast rule vs explicit fp16 autocast rule.
- Test fp32 internal sigmoid/SiLU in forward.
- Test fp32 intermediate computation in backward.
- Compare native, HIP custom_op, Triton AutogradFunction, and `torch.library.triton_op`.
- Compare post-NorMuon update cosine, not only fwd/bwd relative error.

**Expected gain:** Stabilization of hidden-kernel path; maybe no speed.

**Failure predictions:**

| Observation | Interpretation |
|---|---|
| fwd parity good but update cosine poor | small gradient drift amplified by optimizer |
| explicit autocast changes training | dtype boundary mismatch was real |
| fp32 internal SiLU stabilizes | fp16 custom-op math was too close to overflow/rounding boundary |
| triton_op matches native better | prefer compiler-visible Triton path |

### H6 — warmup-local instability can be avoided by delayed kernel enablement

**Claim:** The optimized path may be unstable during LR warmup or early `w_gate_up` catch-up, but usable after native training has moved the model into a safer region.

**Experiments:**

- Native 500 steps -> resume optimized with preserved optimizer/scaler.
- Native 500 steps -> resume optimized weights-only with fresh optimizer/scaler.
- Native 1000 steps -> resume optimized preserved state.
- Native 1000 steps -> enable only loss path first, then SwiGLU later.

**Expected gain:** Production-useful if stable after early window; loses only first 500-1000 steps of speed.

**Failure predictions:**

| Observation | Interpretation |
|---|---|
| preserved-state fails, fresh-state works | optimizer state mismatch with kernel path |
| both fail quickly after enable | kernel dtype/backward/numerics issue independent of warmup |
| step-1000 enable works, step-500 fails | warmup-local instability; delayed enable viable |
| divergence happens same absolute step regardless of enable time | global model/optimizer/fp16 stability threshold |

### H7 — batch=32/accum=4 is a clean utilization win if memory can be made to fit

**Claim:** Larger microbatch can improve throughput by reducing per-microstep overhead and improving utilization. The engineering response says batch=32 gave roughly +5% but doubles memory. This is valuable if memory can be recovered safely.

**Experiments:**

- batch=32/accum=4 without checkpointing, if it fits.
- `gradient_as_bucket_view=True` to reduce memory.
- selective activation checkpointing with non-reentrant checkpoint.
- save GEMM outputs, recompute cheap pointwise only.
- avoid recomputing expensive matmuls.

**Expected gain:** 0-5% if feasible.

**Failure predictions:**

| Observation | Interpretation |
|---|---|
| batch=32 fits and +5% | take immediately |
| checkpointing loses more than it saves | not worth it |
| selective checkpoint with saved mm works | use selective policy |
| batch=32 changes loss dynamics | verify effective batch/LR/accum semantics |

### H8 — hot-path scalar syncs cost 1-3%

**Claim:** Frequent `.item()` or CPU scalar reads inside optimizer/microstep loops can serialize GPU work. The profile shows many `aten::item` calls over 10 optimizer steps.

**Experiments:**

- grep audit for `.item()`, `float(tensor)`, `bool(tensor)`, `get_scale()`, CPU reads.
- classify hot-path vs logging-only.
- convert optimizer decisions to tensor-side predicates where possible.
- sample telemetry every N steps without changing critical path.

**Expected gain:** 1-3% if hot-path syncs exist.

**Failure predictions:**

| Observation | Interpretation |
|---|---|
| sync count high inside optimizer | fix likely useful |
| sync only in every-200-step logging | negligible speed gain; keep observability |
| removing sync changes behavior | code relied on CPU scalar branch; rewrite carefully |

### H9 — hidden-layer kernels need compiler visibility, not more raw kernel tuning

**Claim:** Standalone SwiGLU HIP/Triton kernels are already near the local memory-bandwidth ceiling. The remaining useful experiment is whether `torch.library.triton_op` lets Inductor see through the boundary enough to improve end-to-end training.

**Experiments:**

- Native Inductor `F.silu(gate) * up`.
- HIP custom_op path.
- Triton AutogradFunction path.
- `torch.library.triton_op` + `wrap_triton` path.
- direct Triton call inside compiled region if feasible.
- all tested end-to-end, not only microbench.

**Expected gain:** 0-5% if compiler visibility helps; likely no pure kernel speedup.

**Failure predictions:**

| Observation | Interpretation |
|---|---|
| triton_op improves e2e but not microbench | graph integration is the win |
| triton_op no e2e gain | stop chasing SwiGLU standalone |
| custom_op slower/equally fast but less stable | avoid opaque custom_op in train path |

### H10 — factorized fused CE+z-loss may be a medium-term loss/head win

**Claim:** If the factorized LM head materializes full logits or repeats vocab passes, a streaming fused linear CE + z-loss can save memory traffic and improve speed without touching hidden representation learning.

**Design:**

```text
Inputs:
  h_low: [N, rank]
  E: [V, rank]
  target: [N]

Forward:
  stream vocab tiles
  online max/logsumexp
  target logit
  CE + z_loss

Backward:
  tilewise softmax contribution
  accumulate d_h_low
  accumulate d_E
```

**Expected gain:** unknown; 0-8% depending on current head/loss implementation. Do not start here until profiling proves the head path remains expensive after existing chunked CE/fused zloss.

### H11 — HyPE causal-conv backward may be a niche target

**Claim:** The profile shows HyPE causal-conv backward around 4.5%. If call-site attribution confirms this is a clean standalone bucket, it may be worth optimizing.

**Experiments:**

- Attribute exact conv backward cost by layer.
- Check if graph breaks isolate it.
- Test native/compiled/autograd behavior.
- Only write custom kernels if the implementation is simple and parity is easy.

**Expected gain:** probably 1-3%, maybe more if graph fragmentation is tied to it.

### H12 — activation checkpointing is a memory enabler, not a speed feature

**Claim:** Checkpointing trades compute for memory. Its value here is enabling batch=32/accum=4 or lower memory pressure, not directly speeding training.

Use non-reentrant checkpointing where possible. Be careful with detached tensors and mutable state. Prefer selective checkpointing policies that save expensive matmuls and recompute cheap pointwise ops.

---

## 5. Required instrumentation before more ablations

### 5.1 Granular flags

Replace the monolithic `--optimize-kernels` decision surface with explicit flags:

```text
--ak-enable-all                       # only for test matrix convenience; never production default
--ak-loss-ce
--ak-loss-zloss
--ak-loss-factorized-ce
--ak-swiglu-fwd
--ak-swiglu-bwd
--ak-swiglu-triton-op
--ak-rmsnorm
--ak-rope
--ak-qkv
--ak-hype-conv
--ak-normuon
--ak-compiled-autograd
--ak-compiled-optimizer
--ak-ddp-tuned
--ak-batch32
--ak-no-hot-syncs
```

Each flag should log:

```json
{
  "ak_flags": {
    "loss_zloss": true,
    "swiglu_fwd": false,
    "swiglu_bwd": false,
    "compiled_autograd": true
  },
  "git_sha": "...",
  "torch_version": "...",
  "rocm_version": "...",
  "compile_mode": "max-autotune-no-cudagraphs"
}
```

### 5.2 Backward profile with call-site attribution

The current `CompiledFunctionBackward` bucket is too broad. The engineering agent should produce a table like:

| Backward call-site | CUDA time | % step | inside compiled graph? | mm share | notes |
|---|---:|---:|---:|---:|---|
| FFN `w_gate_up` grad | | | | | |
| FFN `w_down` grad | | | | | |
| SwiGLU backward | | | | | |
| HyPE conv backward | | | | | |
| RMSNorm backward | | | | | |
| CE/zloss backward | | | | | |
| NorMuon NS mm | | | | | |
| DDP wait/allreduce | | | | | |

Success condition:

```text
No experiment proceeds to compiled-autograd optimization until the team knows what is actually inside the backward bucket.
```

### 5.3 DDP trace

Minimum fields per optimizer step:

```json
{
  "step": 123,
  "accum_steps": 8,
  "allreduce_count": 1,
  "allreduce_bytes": 244000000,
  "first_allreduce_ms_from_backward_start": 841.2,
  "last_allreduce_end_ms": 4052.7,
  "ddp_wait_ms": 37.4,
  "bucket_cap_mb": 25,
  "gradient_as_bucket_view": true
}
```

The hard red flag is `allreduce_count > 1` for an accumulation window unless the design intentionally buckets multiple allreduces only on the final microstep. The agent should distinguish bucket count from microstep count.

### 5.4 Graph-break inventory

Run:

```bash
TORCH_LOGS=graph_breaks,recompiles,compiled_autograd \
TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
bash scripts/launch_ddp.sh ...
```

Classify breaks:

| Break source | Expected severity | Action |
|---|---|---|
| `@torch.compiler.disable` around hot HIP kernels | high | replace, narrow, or move outside hot graph |
| Python list mutation in depth/KV buffers | high if repeated | convert to tensor/ring buffer or static structure |
| intentional `h.detach()` | semantic, not necessarily costly | verify no repeated graph fragmentation |
| custom_op boundary | medium | use triton_op or native expression where useful |
| logging/print/scalar extraction | low unless hot-path | sample/defer |
| dynamic shapes | high if causing recompiles | staticize shapes |

### 5.5 NorMuon telemetry

The telemetry should record raw gradient and each optimizer transformation stage. Do not rely on global grad norm.

Per 2D parameter:

```json
{
  "step": 750,
  "name": "layers.7.ffn.w_gate_up.weight",
  "shape": [1536, 768],
  "param_norm": 42.31,
  "param_maxabs": 0.31,
  "raw_grad_norm": 0.82,
  "raw_grad_maxabs": 0.006,
  "momentum_norm": 1.91,
  "post_ns_norm": 18.44,
  "post_spectra_norm": 1.00,
  "effective_lr": 0.002,
  "actual_update_norm": 0.002,
  "update_to_weight_ratio": 0.000047,
  "trust_cap_scale": 1.0,
  "spectra_clipped": true,
  "grad_finite": true,
  "update_finite": true
}
```

Log at:

```text
first 20 optimizer steps
then every 50 steps
then every 10 steps after first warning
always on non-finite/scale-collapse trigger
```

Avoid `.item()` in the hot path. Store tensors or sample sparsely if necessary.

### 5.6 Replay bundle

At the first non-finite grad, GradScaler collapse, activation headroom warning, or update-ratio spike, save:

```text
model weights
optimizer state
GradScaler state
LR scheduler state
random seeds
batch token IDs and document boundaries
loss config
ak flag config
activation stats
per-param update stats
graph-break/recompile counters
DDP bucket config
```

Then replay with:

```text
native path
native + fused zloss
optimized HIP fwd + PyTorch fallback backward
optimized HIP fwd + HIP backward
triton_op path
lower w_gate_up scale
trust cap on/off
```

---

## 6. Detailed experiment cards

### E1 — Backward profile and call-site attribution

**Purpose:** determine whether compiled-autograd/graph work can actually reduce the large backward bucket.

**Run matrix:**

```text
E1-A native baseline, current compile mode
E1-B native + profiler stack/call-site attribution
E1-C native + graph-break logging
E1-D native + one-rank single-node control
E1-E DDP profile with communication timeline
```

**Metrics:**

```text
tok/s
step wall
compiled forward time
compiled backward time
aten::mm by call-site
HyPE conv backward time
loss/zloss backward time
DDP wait time
recompile count
graph-break count
```

**Success output:** `docs/perf/backward-breakdown.md` with a ranked table and optimization recommendations.

---

### E2 — DDP accumulation and overlap

**Purpose:** recover communication/copy overhead and verify accumulation correctness.

**Run matrix:**

```text
E2-A baseline DDP as-is
E2-B assert no_sync on microsteps 0..accum-2
E2-C gradient_as_bucket_view=True
E2-D bucket_cap_mb=8
E2-E bucket_cap_mb=16
E2-F bucket_cap_mb=25
E2-G bucket_cap_mb=50
E2-H bucket_cap_mb=100
E2-I static_graph=True if safe
```

**Success gate:**

```text
allreduce count is correct
no learning difference
speed improves or memory drops enough to enable batch=32
```

**Interpretation:**

- If `no_sync()` was missing, fix immediately and rebaseline everything.
- If bucket tuning gives >2%, keep tuned config in all future runs.
- If only memory improves, use it for batch=32 experiments.

---

### E3 — Fused z-loss and chunked CE in isolation

**Purpose:** capture the safest loss-path win without touching hidden-layer learning.

**Run matrix:**

```text
E3-A native baseline, no fused zloss
E3-B native + --use-fused-zloss only
E3-C native + chunked CE with z_loss_weight
E3-D native + chunked CE without zloss control
```

**Parity tests:**

```text
CE scalar
zloss scalar
dlogits parity
valid-token denominator
ignore_index
softcap on/off
label smoothing if used
z_loss_fraction behavior
packed/intra-doc mask interaction
```

**Training gate:**

```text
500-step smoke for speed/stability
2000-step gate before shipping
```

**Success:** +3-8% with 2000-step loss parity.

---

### E4 — Compiled autograd feasibility

**Purpose:** test whether runtime backward capture improves the large backward bucket.

**Run matrix:**

```text
E4-A native no compiled autograd
E4-B native + compiled autograd
E4-C native + fused zloss + compiled autograd
E4-D DDP native + compiled autograd
E4-E DDP + best DDP bucket config + compiled autograd
E4-F triton_op SwiGLU only + compiled autograd
E4-G custom_op HIP path + compiled autograd
```

**Example toggle:**

```python
import torch

torch._dynamo.config.compiled_autograd = True

@torch.compile(mode="max-autotune-no-cudagraphs")
def train_step(...):
    ...
```

or narrow context if the repo prefers:

```python
with torch._dynamo.compiled_autograd.enable(torch.compile(fullgraph=False)):
    loss.backward()
```

**Metrics:**

```text
steady-state tok/s after warmup
compiled backward time
unique compiled graphs
recompile count
graph-break count
DDP bucket/hook behavior
compile time
step-time variance
loss/GradScaler parity
```

**Stop conditions:**

```text
frequent recompiles after warmup
DDP hook failures
loss/gradient mismatch
>5% regression in steady state
```

---

### E5 — Graph-break cleanup

**Purpose:** remove or contain graph breaks that fragment forward/backward and prevent compiler wins.

**Procedure:**

1. Run graph-break inventory.
2. Rank breaks by frequency and location.
3. Fix only hot-path breaks first.
4. Reprofile after each fix; do not combine many graph-break fixes before measuring.

**Likely fixes:**

```text
replace Python lists with tensor-backed buffers where possible
move semantic detach outside compiled region if possible
narrow @torch.compiler.disable regions
use torch.library.triton_op for Triton kernels
avoid Python control flow on tensor values
avoid CPU scalar extraction in compiled region
```

---

### E6 — Sync-point audit

**Purpose:** remove hot-path CPU/GPU synchronizations.

**Audit patterns:**

```bash
grep -R "\.item()\|float(.*Tensor\|bool(.*Tensor\|get_scale\|cpu()\|numpy()" \
  halo_training models autokernel kernels scripts -n
```

Classify:

```text
microstep hot path
optimizer hot path
DDP hot path
every-step logging
every-N-step diagnostics
rare forensics path
```

**Rule:** remove hot-path syncs; preserve sparse observability.

---

### E7 — NorMuon telemetry and optimization

**Purpose:** determine whether the failure is raw gradient, optimizer transform, SPECTRA, update application, or activation dynamics.

**Run matrix:**

```text
E7-A native baseline telemetry only
E7-B post-fix optimized telemetry only, short run
E7-C native + fused zloss telemetry
E7-D optimized with PyTorch fallback backward
E7-E optimized with HIP backward
```

**Optimization experiments after telemetry:**

```text
ns_steps=3 vs 5
fused NS/SPECTRA/update apply
grouped same-shape NS operations
no CPU scalar reads inside SPECTRA
match_rms_adamw-style scaling if compatible
```

---

### E8 — SPECTRA + trust-cap composition

**Purpose:** stabilize post-NorMuon updates without globally lowering LR.

Let:

```text
U_ns     = post-Newton-Schulz update direction
U_spec   = SPECTRA(U_ns), capped by absolute norm c
eta_eff  = lr × shape_adjust × param_group_multiplier
delta    = eta_eff × U_spec
trust    = ||delta|| / (||W|| + eps)
```

SPECTRA enforces:

```text
||U_spec|| <= c
```

Trust cap enforces:

```text
||eta_eff × U_spec|| / ||W|| <= τ
```

Equivalent cap on `U_spec`:

```text
||U_spec|| <= τ × ||W|| / eta_eff
```

Together:

```text
||U_final|| <= min(c, τ × ||W|| / eta_eff)
```

They are not redundant:

- SPECTRA caps absolute update norm.
- trust cap caps relative step size vs parameter norm.
- trust cap matters most for small-norm parameters or high effective LR groups.

Recommended order:

```text
raw grad
→ momentum / Nesterov
→ Newton-Schulz
→ SPECTRA absolute norm cap
→ effective LR / shape scaling / param multiplier
→ trust-ratio cap
→ decoupled weight decay handling, if not already applied
→ parameter update
```

Implementation sketch:

```python
def apply_trust_cap_(update, param, effective_lr, tau=0.02, eps=1e-12):
    # update is post-NS/post-SPECTRA and before param.add_
    update_f32 = update.float()
    param_f32 = param.float()

    update_norm = torch.linalg.vector_norm(update_f32)
    param_norm = torch.linalg.vector_norm(param_f32)

    ratio = effective_lr * update_norm / (param_norm + eps)
    scale = torch.clamp(tau / (ratio + eps), max=1.0)
    update.mul_(scale.to(update.dtype))
    return ratio, scale
```

Start with:

```text
τ = 0.02 diagnostic
τ = 0.05 less intrusive
apply to w_gate_up only first
then only layers whose update_ratio spikes
```

---

### E9 — Warm-start / delayed kernel enable

**Purpose:** test whether optimized-kernel instability is warmup-local or intrinsic.

**Run matrix:**

```text
E9-A native 1000-step baseline
E9-B native 500 -> optimized, preserving optimizer/scaler
E9-C native 500 -> optimized, fresh optimizer/scaler
E9-D native 1000 -> optimized, preserving optimizer/scaler
E9-E native 1000 -> loss kernels first -> SwiGLU later
```

**Interpretation:**

| Observation | Meaning |
|---|---|
| preserved fails, fresh works | optimizer-state/path interaction |
| both fail soon after enable | kernel numeric issue likely |
| 1000 works, 500 fails | warmup-local instability |
| divergence tied to absolute step | global stability threshold |

---

### E10 — Custom-op autocast and fp32 islands

**Purpose:** ensure custom ops match native autocast/precision behavior.

**Run matrix:**

```text
E10-A native Inductor F.silu(gate) * up
E10-B custom_op with no explicit autocast rule
E10-C custom_op with register_autocast(..., torch.float16)
E10-D custom_op with fp32 internal sigmoid/SiLU, fp16 output
E10-E custom_op backward with fp32 intermediates
E10-F triton_op with fp32 internal sigmoid/SiLU
```

**Metrics:**

```text
forward rel_err
backward rel_err
grad cosine per layer
post-NorMuon update cosine
activation maxabs drift
loss over replayed batch sequence
```

---

### E11 — batch=32 / accum=4

**Purpose:** reduce microstep overhead and improve utilization.

**Run matrix:**

```text
E11-A batch=16 accum=8 baseline
E11-B batch=32 accum=4, no checkpoint, if fits
E11-C batch=32 accum=4 + gradient_as_bucket_view
E11-D batch=32 accum=4 + selective checkpoint, save mm
E11-E batch=32 accum=4 + recompute cheap pointwise only
```

**Caution:** activation checkpointing can silently change behavior if the recomputed function is not equivalent. Use non-reentrant checkpointing and explicit determinism checks.

---

### E12 — factorized fused CE + z-loss

**Purpose:** medium-term head/loss optimization if existing fused z-loss is insufficient.

**Prototype requirements:**

```text
no full logits allocation if avoidable
fp32 online max/logsumexp
valid-token denominator parity
ignore_index parity
softcap parity
z_loss_weight parity
d_h and d_E gradient parity
```

**Do not start until:** E3 proves loss path remains a real bottleneck after existing fused/chunked CE.

---

### E13 — `torch.library.triton_op` SwiGLU integration

**Purpose:** test compiler visibility, not raw standalone kernel speed.

PyTorch's `torch.library.triton_op` documentation says `custom_op` is opaque to `torch.compile` and export, while `triton_op` can make Triton kernels visible to compiler subsystems when wrapped correctly with `wrap_triton`.

**Run matrix:**

```text
E13-A native F.silu(gate) * up
E13-B HIP custom_op
E13-C Triton AutogradFunction
E13-D torch.library.triton_op + wrap_triton
E13-E E13-D + compiled autograd
```

**Success:** end-to-end speedup and 2000-step stability. Isolated microbench improvement alone does not count.

---

### E14 — HyPE causal-conv backward

**Purpose:** optional follow-up if call-site attribution confirms ~4-5% clean bucket.

**Run matrix:**

```text
E14-A current HyPE backward profile
E14-B compiled autograd effect on HyPE backward
E14-C graph-break cleanup around HyPE
E14-D custom backward only if parity is straightforward
```

Do not start before E1/E5.

---

## 7. Answers to the engineer-agent's eight questions

### 7.1 Compiled autograd + custom_op + ROCm

I do not know of a public guarantee that compiled autograd + ROCm + DDP + `torch.library.custom_op(register_autograd)` works reliably for this exact stack. PyTorch documents compiled autograd as active development and not compatible with all existing features. That makes it a high-priority **feasibility gate**, not an assumed win.

Recommended order:

```text
single-node native
single-node native + compiled autograd
DDP native + compiled autograd
DDP native + fused zloss + compiled autograd
DDP triton_op SwiGLU + compiled autograd
DDP custom_op HIP + compiled autograd
```

Expectation:

```text
native + compiled autograd: plausible
triton_op + compiled autograd: plausible
custom_op + compiled autograd: possible but less likely to expose optimization through the op
DDP + compiled autograd: uncertain; hooks and bucket behavior are the risk
```

### 7.2 NorMuon + trust cap + SPECTRA

They are orthogonal but can double-clip if thresholds are too tight. SPECTRA caps absolute update norm after Newton-Schulz. Trust cap limits the final LR-scaled update relative to the parameter norm.

Correct order:

```text
NS → SPECTRA → effective LR/shape scaling → trust cap → parameter update
```

SPECTRA dominates when:

```text
spectra_clip <= trust_cap * ||W|| / effective_lr
```

Trust cap dominates when:

```text
spectra_clip > trust_cap * ||W|| / effective_lr
```

Start with `w_gate_up` only and log trigger rate.

### 7.3 Alternative divergence hypotheses

Add these to the repo's existing H-list:

```text
H11: custom-op autocast boundary mismatch.
Prediction: explicit register_autocast or fp32 internal SiLU changes stability.

H12: post-NorMuon update-ratio spikes, not raw grad spikes.
Prediction: update/weight ratio rises before GradScaler collapse.

H13: LR warmup completion aligns with custom-op numerical drift.
Prediction: longer warmup or delayed enable shifts divergence.

H14: optimizer state mismatch after native->optimized switch.
Prediction: preserved-state warm-start behaves differently from fresh-state warm-start.

H15: DDP bucket/allreduce timing changes scaler or update timing.
Prediction: single-node and DDP divergence steps differ after equal effective batch/LR.

H16: fused z-loss denominator/masking mismatch.
Prediction: fused-zloss-only run diverges while native zloss does not.

H17: forward graph breaks fragment backward enough to lose Inductor fusion and alter reduction ordering.
Prediction: compiled autograd or triton_op improves both speed and stability.

H18: HyPE conv/depth-KV buffer mutation causes repeated recompiles.
Prediction: graph-break inventory finds recurrent breaks in hot loop.
```

### 7.4 Warm-start experiment design

Native 500-step -> optimized resume is good but incomplete. Run both preserved optimizer/scaler and fresh optimizer/scaler. Also test step 1000, because Phase G divergence happened after step 700.

Better matrix:

```text
native 500 -> optimized preserved state
native 500 -> optimized fresh optimizer/scaler
native 1000 -> optimized preserved state
native 1000 -> optimized fresh optimizer/scaler
native 1000 -> enable loss kernels only -> enable hidden kernels after another 500
```

Do not warm-start only FFN gate weights unless replay localizes the issue specifically to those weights. Whole-model warm-start better tests production feasibility.

### 7.5 Chunked fused CE+z-loss extensions

First validate the existing `ChunkedLinearCrossEntropyLoss(z_loss_weight)` and fused CE path. Then extend only if profiling says head/loss remains expensive.

Concrete extensions:

1. strict parity harness for ignore_index, softcap, z_loss_fraction, label smoothing, packed/intra-doc masks, and valid-token denominator.
2. gradient-in-forward mode that computes CE and dlogits together.
3. factorized-head streaming CE to avoid full logits materialization.
4. fp32 online max/logsumexp accumulation.
5. chunk-size autotune over vocab tile, token tile, and rank tile.

Liger Kernel is a useful design reference because it includes CrossEntropy and FusedLinearCrossEntropy kernels and emphasizes exact forward/backward testing, but its reported A100/bf16/FSDP results should not be treated as directly transferable to Strix Halo.

### 7.6 fp16-only stability alternatives

Highest-value additions:

```text
1. fp32 reductions in custom kernels:
   logsumexp, RMSNorm/RMS variance, SPECTRA norms, NS norms.

2. explicit custom-op autocast rules:
   avoid accidental dtype drift versus native path.

3. post-update trust cap:
   especially w_gate_up or any layer with update-ratio spikes.

4. w_gate_up staging:
   update scale 0.25 for first 500-1000 steps after enabling optimized path, then ramp to 1.0.

5. overflow-triggered rollback and local scale reduction:
   if scaler falls below threshold, rollback and resume with lower w_gate_up scale/trust cap.

6. fp32 optimizer state where memory permits:
   momentum/NS buffers in fp32 for sensitive groups.

7. activation/logit headroom telemetry:
   catch layer-local maxabs growth before scaler collapse.
```

Lower priority:

```text
stochastic rounding: hard to implement/validate, possible overhead
Kahan summation: possibly useful in optimizer reductions, lower priority than fp32 reductions
global LR reduction: stabilizes but may hurt tuned recipe quality
```

AMP diagnostics must unscale gradients only after all accumulation microsteps for the effective batch are complete.

### 7.7 Realistic gfx1151 throughput ceiling

40k aggregate is aspirational but not absurd. It is hard because:

- per-node target rises from about 15.65k tok/s to 20k tok/s.
- GEMMs are a large real-compute bucket and not easily addressable.
- custom SwiGLU is already near its standalone bandwidth ceiling.
- DDP over TB4 adds communication overhead.
- fp16 stability blocks the old fast path.

Back-of-envelope view:

```text
36-38k aggregate: plausible with disciplined safe-stack work.
40k aggregate: plausible only if compiled autograd, DDP, loss fusion, batch/optimizer work stack.
>40k aggregate: likely requires recovered hidden-kernel path or batch=32/accum=4.
```

### 7.8 Triton-on-ROCm ceiling

For this SwiGLU elementwise op, trust the repo's own data over general lore: HIP and Triton are already within ~1-2% of each other and both are around 1.43-1.45x eager in isolated fwd+bwd. That means no large pure-kernel tuning win remains for standalone SwiGLU.

Triton remains valuable for:

- `torch.library.triton_op` compiler visibility.
- CE/reduction kernels.
- factorized fused loss/head kernels.
- easier parity/autotune infrastructure.

It should not be treated as a magic route to beat HIP on every pointwise op.

---

## 8. Success gates and stop conditions

### 8.1 Speed gates

Use steady-state tok/s after warmup, not initial compilation steps.

| Gate | Required result |
|---|---:|
| quick safe stack | >=32.5k tok/s |
| main safe stack | >=36k tok/s |
| near-target stack | >=38.5k tok/s |
| target stack | >=40k tok/s |

### 8.2 Learning/stability gates

Every candidate stack must pass:

```text
no newly frozen params beyond documented architectural allowlist
loss@500 within expected baseline variance
loss@1000 within expected baseline variance
loss@2000 within expected baseline variance
GradScaler scale does not collapse
no repeated non-finite gradient steps
activation maxabs headroom remains above safety threshold
per-param update/weight ratio does not spike without explanation
no silent graph recompilation storm
```

### 8.3 Kernel parity gates

For each training-path kernel:

```text
forward parity at production shapes
backward parity at production shapes
autocast behavior parity
grad cosine per affected parameter
post-NorMuon update cosine
replay-batch loss parity
```

### 8.4 Stop conditions

Stop a run and save replay bundle on:

```text
first non-finite gradient
GradScaler scale below configured floor
activation fp16 headroom below safety threshold
loss jump > configured threshold
update_to_weight_ratio spike over threshold
unexpected graph recompile storm
DDP allreduce count anomaly
```

---

## 9. Concrete stack candidates

### Stack A — lowest risk quick stack

```text
native hidden path
+ fused z-loss or chunked CE z-loss if stable
+ DDP no_sync/bucket/gradient_as_bucket_view best config
+ hot-path scalar sync cleanup
```

Expected: **33k-35k**.

This probably does not reach 40k, but it should be the first production-safe improvement.

### Stack B — realistic near-term main stack

```text
Stack A
+ compiled autograd if recompiles are controlled
+ graph-break cleanup needed for compiled autograd
+ batch=32/accum=4 if memory fits or can be enabled safely
```

Expected: **36k-38.5k**.

This is the most realistic no-hidden-kernel route.

### Stack C — possible 40k without blanket hidden kernels

```text
Stack B
+ NorMuon implementation optimization
+ ns_steps=3 only if parity holds
+ fused SPECTRA/update apply
+ factorized fused CE if profiling still supports it
```

Expected: **38k-41k** depending on compiled autograd and batch result.

### Stack D — possible 40k+ with recovered hidden kernels

```text
Stack B or C
+ triton_op SwiGLU, not opaque custom_op
+ delayed enable at step 1000
+ w_gate_up update scale 0.25 -> 1.0 ramp
+ post-NorMuon trust cap
+ fp32 internal custom-op reductions/islands
```

Expected: **39k-41k**, but highest stability risk.

---

## 10. What not to spend time on

### 10.1 Do not pursue custom HIP GEMMs

The hardware lacks MFMA/matrix cores, and rocBLAS is the right path for dense matmuls on this target. Custom GEMM work is unlikely to beat vendor libraries enough to justify the risk.

### 10.2 Drop hipBLASLt epilogue work for SwiGLU

Standard GEMM epilogues do not naturally express:

```text
linear -> split gate/up -> SiLU(gate) * up -> next linear
```

A full fused MLP kernel would be a multi-week custom project and likely lose vendor GEMM advantages on gfx1151.

### 10.3 Do not search layer-selective SwiGLU blindly

Layer masks create a combinatorial search space. Only run layer-selective experiments if replay or telemetry localizes instability to specific layers.

### 10.4 Do not count microbenchmarks as ship evidence

The repo already learned this: isolated SwiGLU speed does not guarantee end-to-end throughput or learning stability.

### 10.5 Do not lower global LR as the first fix

Lower global LR may stabilize but risks sacrificing the tuned native recipe. Prefer targeted update controls first:

```text
w_gate_up scale
post-update trust cap
fp32 custom-op islands
longer warmup only if evidence supports it
```

---

## 11. Pre-committed interpretation table

| Observation | Most likely meaning | Next action |
|---|---|---|
| fused zloss-only stable and +5-8% | loss path is safe pillar | keep in all stacks |
| fused zloss-only diverges | zloss gradient/dtype/mask bug | parity harness for denominator/ignore_index/softcap |
| compiled autograd gives no speed | backward bucket is real compute/GEMM or recompiles | stop expecting compiler-only 40k |
| compiled autograd recompiles often | dynamic state/list mutation/hook issue | fix graph breaks first |
| DDP `no_sync()` gives large speedup | accumulation allreduce bug | fix and rebaseline all experiments |
| DDP bucket sweep gives +3-5% | communication overlap was suboptimal | keep tuned bucket settings |
| update ratio spikes before divergence | NorMuon/SPECTRA/update-scale issue | trust cap + w_gate_up staging |
| activation maxabs spikes but update ratio normal | forward numerics/residual dynamics | fp32 islands, softcaps, residual scale investigation |
| warm-start preserved fails, fresh works | optimizer state mismatch | reset/transform optimizer state on kernel enable |
| warm-start at 1000 succeeds but 500 fails | warmup-local instability | delayed enable is viable |
| triton_op improves e2e but not microbench | compiler visibility is the win | prefer triton_op over custom_op |
| triton_op no e2e gain | SwiGLU boundary not bottleneck | stop chasing standalone SwiGLU |
| batch=32 fits and +5% | clean utilization win | adopt after 2000-step parity |
| checkpointing enables batch=32 but erases speed | memory solution not worth compute | drop checkpointing path |

---

## 12. Immediate engineering checklist

### Day 0-2 checklist

```text
[ ] Add granular --ak-* flag plumbing or at least env toggles.
[ ] Produce backward call-site profile.
[ ] Produce DDP communication/allreduce trace.
[ ] Verify no_sync semantics under accum=8.
[ ] Run graph-break/recompile inventory.
[ ] Run sync-point grep/audit.
[ ] Add NorMuon telemetry schema behind low-frequency sampling flag.
```

### Day 3-5 checklist

```text
[ ] Run fused zloss-only native hidden path.
[ ] Run chunked CE zloss native hidden path.
[ ] Run DDP bucket + gradient_as_bucket_view sweep.
[ ] Remove obvious hot-path syncs.
[ ] Fix high-impact graph breaks only if profile supports.
```

### Day 6-10 checklist

```text
[ ] Run compiled autograd single-node native.
[ ] Run compiled autograd DDP native.
[ ] Run compiled autograd + best DDP config.
[ ] Run triton_op SwiGLU integration experiment.
[ ] Test batch=32/accum=4 memory path.
```

### Day 11-16 checklist

```text
[ ] Build replay_step.py.
[ ] Run native->optimized warm-start matrix.
[ ] Run PyTorch-backward fallback vs HIP-backward localization.
[ ] Add trust cap and w_gate_up staging if telemetry supports.
[ ] Run custom-op autocast/fp32-island parity matrix.
```

### Final gate

```text
[ ] Best no-hidden-kernel stack passes 2000 steps.
[ ] Best hidden-kernel stack, if any, passes 2000 steps.
[ ] STATUS.md, CONSTRAINTS.md, AGENTS.md updated to match results.
[ ] Any production command excludes blanket --optimize-kernels unless the granular stack passed.
```

---

## 13. Recommended production command philosophy

Do not ship commands like:

```bash
--optimize-kernels
```

Ship commands like:

```bash
--ak-loss-zloss \
--ak-ddp-tuned \
--ak-compiled-autograd \
--ak-no-hot-syncs
```

Only add hidden kernels when individually proven:

```bash
--ak-swiglu-triton-op \
--ak-swiglu-enable-step 1000 \
--ak-w-gate-up-scale-start 0.25 \
--ak-w-gate-up-scale-ramp-steps 1000 \
--ak-trust-cap 0.02
```

This makes the performance stack auditable and reversible.

---

## 14. Reference sources

The following sources were used to prepare this handoff. Verify current repo state before launching experiments, because the repo is actively changing.

### Repo sources

- `STATUS.md` — current ship decisions, Phase B/C/G/I summary, fused z-loss status:  
  https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/STATUS.md

- `CONSTRAINTS.md` — hardware/software constraints and fp16/Strix Halo rules:  
  https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/CONSTRAINTS.md

- OdinFlat step profile:  
  https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/odinflat-step-profile.md

- OdinFlat throughput final report:  
  https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/odinflat-throughput-final.md

- Phase C final analysis:  
  https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/phase-c-final-analysis.md

- Phase G findings:  
  https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/phase-g-findings.md

- Triton SwiGLU ship-gate bench:  
  https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/perf/triton-swiglu-ship-gate-bench.md

- Autokernel stability research brief:  
  https://raw.githubusercontent.com/joelhenwang/autokernel-halo-strix/refs/heads/main/docs/research/autokernel-stability-research-brief.md

### PyTorch / ROCm / hardware sources

- PyTorch `torch.library` docs, including `register_autograd`, `register_autocast`, `triton_op`, and `wrap_triton`:  
  https://docs.pytorch.org/docs/2.11/library.html

- PyTorch compiled autograd tutorial:  
  https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html

- PyTorch DistributedDataParallel docs, including `bucket_cap_mb` and `gradient_as_bucket_view`:  
  https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html

- PyTorch AMP examples, including `unscale_` and accumulation rules:  
  https://docs.pytorch.org/docs/2.11/notes/amp_examples.html

- PyTorch activation checkpointing docs:  
  https://docs.pytorch.org/docs/2.11/checkpoint.html

- PyTorch user-defined Triton kernels with `torch.compile`:  
  https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html

- PyTorch Muon docs:  
  https://docs.pytorch.org/docs/2.11/generated/torch.optim.Muon.html

- AMD Ryzen AI Max+ 395 specs:  
  https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html

- ROCm RDNA3.5 system optimization docs:  
  https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html

- Liger Kernel docs, used as design reference for exact fused CE/FusedLinearCE style kernels:  
  https://linkedin.github.io/Liger-Kernel/

---

## 15. Final recommendation

The engineering team should not try to recover 40k by resurrecting a monolithic optimized-kernel path. That path already failed the core requirement: speed and learning must coexist.

The best path is:

```text
1. Measure backward/DDP/graph breaks first.
2. Capture safe loss and DDP/sync wins.
3. Test compiled autograd as a feasibility gate.
4. Use batch=32/accum=4 if memory can be made to fit without losing speed.
5. Add NorMuon telemetry and trust controls before hidden-kernel recovery.
6. Convert hidden kernels to compiler-visible triton_op only if end-to-end evidence supports it.
7. Stack only deltas that individually pass speed, parity, and 2000-step learning gates.
```

The honest target ladder is:

```text
31.3k -> 33-35k: likely with safe quick wins.
33-35k -> 36-38.5k: plausible with compiled autograd, DDP, and/or batch=32.
36-38.5k -> 40k: requires NorMuon optimization or recovered compiler-visible hidden kernels.
```

The hard line remains:

> **No throughput number counts unless the model still learns with all intended parameters.**
