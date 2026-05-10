# Engineering response to the 40k tok/s throughput guide

**Source doc:** `docs/research/autokernel_halo_strix_40k_throughput_experiment_guide.md`
**Response author:** engineering team operating the repo
**Repo SHA at time of writing:** `4a367ea` (main)
**Purpose:** give the guide author the operator context they didn't have, so their v2 can refine priorities, drop stale items, and surface hypotheses we're missing
**Tone:** collegial, data-forward, not adversarial. The guide is the best synthesis we've received. This response is a handoff, not a rejection.

---

## 0. What this document is and isn't

**Is:**
- Specific data corrections to the guide, each with evidence from the repo
- A list of topics the guide missed that we think should be in v2
- A ranked list of priority disagreements and the reasons
- Our locked 3-week execution plan (T-0 through T-4), fully transparent
- Eight specific questions where your expertise would most improve our plan
- An appendix of raw data you can use to re-reason

**Isn't:**
- A rejection of the guide's framework. The granular-flag / loss-first / telemetry-first approach is correct.
- A complete re-derivation. We want you to revise your v1, not re-write from scratch.
- Blocking. We're executing observation-only T-0 items in parallel with your v2. When v2 arrives we'll fold it into T-0.2 onward.

**What we want from v2:**
1. Revised priority ranking given the corrected numbers in §3
2. Additional hypotheses (H11+) we haven't considered, especially around the divergence mechanism given §4's missing topics
3. Answers (or disagreements) on the eight questions in §7
4. Sharper failure-mode predictions — "if experiment X diverges at step N, it means Y" — so we can pre-commit to interpretation

---

## 1. Acknowledgment — what your guide gets right

Five items we're adopting directly:

1. **Granular `--ak-*` flags instead of monolithic `--optimize-kernels`.** The all-in switch was our biggest tactical mistake. Phase C v1/v2/v3 and Phase G all diverged ambiguously because we couldn't A/B one axis at a time. Your §27 flag design is going into our T-0.5.
2. **Loss-path before hidden-layer sequencing.** We agree the loss path has a better safety/speed ratio than SwiGLU replacement. Our Phase I (Triton fused SwiGLU) empirically ruled out a pure-kernel win on this hardware.
3. **`torch.library.triton_op` vs `custom_op` for compile visibility.** This is a genuine gap. Our current Triton harness uses `torch.autograd.Function` which has the same opacity problem as `custom_op`. We're adding this to T-2.2.
4. **Per-parameter update telemetry (your §6.1, our E7).** We have global grad norm and activation maxabs, but nothing between raw grad and final applied update. With NorMuon + Newton-Schulz + SPECTRA, that's three transformations we can't see through. T-0.2.
5. **Delayed kernel enable (E10) as the cheap warmup-local-instability test.** Our Phase G showed the optimized path tracked *better* than pre-fix through step 700 before diverging at step 750. That pattern begs the warm-start experiment. T-3.2.

---

## 2. Corrected data — where your guide is numerically off

This is the most important section. Each correction includes evidence.

### 2.1 Fused z-loss upside is 5-8%, not 16.7%

**Your claim:** §0 and §5 E1 imply that fusing z-loss eliminates 16.7% of step wall (the 11.1% `logsumexp` forward + 5.6% backward), taking throughput 31.3k → 37.6k.

**Correction:** `logsumexp` is intrinsic to CE. You cannot eliminate it. What you can eliminate is the *second* `logsumexp` pass when z-loss is computed separately from CE.

**Evidence:**

- `docs/perf/odinflat-step-profile.md:19-27` shows the 16.7% measurement. It is the cost of a **redundant** pass, not the cost of logsumexp itself. CE already computes logsumexp internally; our training currently runs CE *and then* a separate `logsumexp(logits)²` for z-loss on top. Fusing means doing both in one pass.

- `kernels/hip/chunked_linear_cross_entropy.py:32-140` already supports `z_loss_weight` built into the chunked CE kernel. When `chunked_ce_handles_zloss = (z_weight > 0)` in `halo_training/trainer.py:302`, the z-loss path runs inside the fused kernel with no second pass.

- `kernel.py` has a fused `ce_full(z_loss_weight=...)` path (Phase B.5, 2026-05-11). `--use-fused-zloss` routes logits through it.

**Realistic upside:** Saving the *second* logsumexp = ~5-8% step-wall reduction, not 17%. The first logsumexp stays either way.

**Implication for your execution order:** Your Phase B exit gate ("loss path gives ≥36k–38k and passes 2000 steps") is probably unachievable from z-loss fusion alone. The loss path alone will hit ~33-34k, not 37-38k.

---

### 2.2 The largest untouched bucket is backward, not loss

**Your guide barely mentions backward.** Your §4.3 SwiGLU discussion touches it ("backward overhead negates forward savings") but there's no experiment family for backward compilation or optimization.

**Reality:**

| Bucket | % of step | Your guide priority |
|---|---:|---|
| aten::mm (all GEMMs, fwd+bwd+NorMuon NS) | 37.85% | not addressable (rocBLAS) |
| **Compiled backward (CompiledFunctionBackward CUDA)** | **35.3%** | **not discussed** |
| Compiled forward (CompiledFunction CUDA) | 16.2% | not discussed |
| z-loss logsumexp fwd | 11.1% | HIGH priority (E1/E2) |
| NorMuon optimizer step | 12.5% | MEDIUM priority (E12/E13) |
| z-loss logsumexp bwd | 5.6% | included in E1/E2 |
| HyPE causal_conv1d bwd | 4.5% | not discussed |

Source: `docs/perf/odinflat-step-profile.md:19-27`.

(Totals >100% because `aten::mm` overlaps with CompiledFunction; not clean additive.)

**Implication:** Compiled autograd is likely the single biggest lever we have. PyTorch 2.4+ supports `torch._dynamo.config.compiled_autograd = True` which compiles backward as one graph. If it works with our `register_autograd` custom_ops, it targets the 35.3% bucket directly.

**This is missing from your v1 entirely.** Please consider adding an experiment family E-bwd (or fold into E4) specifically for compiled-autograd feasibility on our stack.

---

### 2.3 `aten::mm` = 37.85%, not "62% is addressable"

**Your §3.3 math:**
```
baseline 31.3k → 40k requires -21.75% step time
```

**Our reality:** of that 100% step wall, ~38% is rocBLAS-backed `aten::mm` on a chip with no MFMA. We don't have a realistic path to beat rocBLAS.

**Revised addressable budget:**
- Non-addressable (rocBLAS GEMMs + allocator + dispatcher + Python overhead): ~40-45%
- Addressable (backward compile, loss fusion, optimizer opt, DDP overlap, sync removal): ~25-30%
- Already captured (max-autotune-no-cudagraphs, fp16 NS, fused AdamW): ~baseline

**Implication:** 40k from 31k (+28%) is at the *edge* of the addressable budget. 35-37k (+13-18%) is a more honest recovery target given the silent-freeze bug was inflating the pre-fix baseline by ~15-20%.

**Request for v2:** please sanity-check your 40k framing. Is this target from the hardware spec, from a datacenter-GPU comparison, or from the broken `--optimize-kernels` number (~41k, now known to be partially fake)?

---

### 2.4 DDP allreduce overhead is missing from your analysis

**Your guide doesn't mention DDP.** Your E14 "sync reduction" covers `.item()` calls but not allreduce.

**Our topology:**
- Two Strix Halo machines over Thunderbolt 4
- gloo backend (we've confirmed gloo matches RCCL on unified-memory gfx1151)
- `GLOO_SOCKET_IFNAME=thunderbolt0`
- Measured TB4 raw throughput: ~1.04 GB/s

**Gradient volume per step:**
- OdinFlat 122M params × fp16 = 244 MB
- Trainable 2D params (NorMuon scope) × fp16 ≈ 200 MB
- At 1 GB/s = ~200-250 ms of raw allreduce

**Step wall time at 31.3k tok/s (batch=16, block=512, accum=8, DDP=2):**
- Effective batch = 16 × 8 × 2 × 512 = 131,072 tokens
- Step wall = 131,072 / 31,300 ≈ 4.2 s
- Allreduce = 200ms / 4200ms = ~4-5% of step wall

But this is with default `gradient_as_bucket_view=False` and default bucket sizing. Tighter bucket control + overlap could recover most of this.

**Suggestion for v2:** add an E-ddp experiment family covering `gradient_as_bucket_view`, `bucket_cap_mb` tuning, `no_sync()` on non-final microsteps at accum=8 (1 allreduce/step instead of 8), and gloo-vs-RCCL re-benchmark on unified memory.

---

### 2.5 `max-autotune-no-cudagraphs` is already on — not a free lever

**Your guide doesn't acknowledge the current compile mode.**

**Reality:** Sprint 3A-confirm runs with `TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`. Phase 3 WI-B3 measured +5.17% tok/s vs `default`. This is our **baseline**, not an unused knob.

Evidence: `AGENTS.md` under "Compile strategy"; `scripts/launch_ddp.sh:21`.

**Notes:**
- `max-autotune` (with cudagraphs) crashes with `accum_steps > 1`. This is a hard constraint on Strix Halo.
- `reduce-overhead` doesn't crash but HIP's CUDA-graph backend produces empty-graph warnings and runs eagerly (Phase 3 WI-A0). Net −1.8%. Trainer auto-redirects to `default` with a NOTE.

**Implication:** any compile-mode experiment in v2 should start from `max-autotune-no-cudagraphs` as the floor. There's no "turn on max-autotune" win left to capture.

---

### 2.6 Triton fused SwiGLU hits HIP parity, not underperforms

**Your §4.3 reads:** "Triton SwiGLU kernel achieved about 1.43× vs eager in isolated fwd+bwd, but only about 0.986× vs the autograd-safe HIP path."

**Missing context you need:** our autograd-safe HIP is **1.45× eager**. So Triton at 0.99× HIP ≈ 1.43× eager ≈ the same ceiling.

**Evidence:** `docs/perf/triton-swiglu-ship-gate-bench.md:7-11`:

| Comparison | fwd speedup | fwd+bwd speedup |
|---|---:|---:|
| eager-vs-autograd-hip | 1.693× | 1.450× |
| eager-vs-triton | 1.634× | 1.434× |
| autograd-hip-vs-triton | 0.977× | 0.986× |

**Implication:** Triton isn't being outperformed by HIP. Both are hitting a **kernel-level ceiling** determined by memory bandwidth on gfx1151. There is no pure-kernel win left to extract from SwiGLU elementwise.

**The only remaining Triton lever is compile visibility** — whether `triton_op` integration allows Inductor to fuse around the SwiGLU boundary (e.g. into the preceding `w_gate_up` output or the following `w_down` input). This is what your E4 should test, but v1 doesn't make this reframing explicit.

**Suggestion for v2:** explicitly recharacterize Triton SwiGLU. It's not "a kernel that doesn't beat HIP yet." It's "both kernels are at the bandwidth ceiling; the only remaining lever is compiler integration."

---

## 3. Missing topics we want added to v2

Five topics the guide doesn't cover that we think should be experiment families.

### 3.1 Compiled autograd

**What:** `torch._dynamo.config.compiled_autograd = True` (PyTorch 2.4+) compiles the backward pass as a single graph.

**Why:** targets the 35.3% CompiledFunctionBackward bucket (§2.2).

**Risks:**
- ROCm compatibility uncertain
- Interaction with `register_autograd` custom_ops unknown
- DDP + compiled autograd may have bucket-hook conflicts

**Budget:** 5-6 days (per our user, this is our highest-priority T-2 item despite risk).

**Request for v2:** do you have experience with compiled autograd on ROCm? Known pitfalls? Compatible Triton/custom_op patterns?

---

### 3.2 Inductor graph-break inventory

**What:** run `TORCH_LOGS=graph_breaks,recompiles` to catalog every graph break in the current compile_zones path.

**Why:** our HyPEShortConvBlock has known breaks:
- `@torch.compiler.disable` on HIP kernels: 4 breaks per block × 14 blocks = potentially 56 break points
- `h.detach()` in MoDA depth-KV write (intentional — severs cross-iter gradient)
- Python list mutation on `depth_kv_buffer`

We don't know which are cheap (compiler elides) vs expensive (graph recompile overhead).

**Budget:** 1 hour to catalog, variable to fix.

**Request for v2:** given your understanding of Dynamo/Inductor, which of these break patterns typically have non-zero runtime cost vs which are compile-time-only concerns?

---

### 3.3 Activation checkpointing / selective recompute

**What:** `torch.utils.checkpoint.checkpoint_sequential` or `torch.ops.aten._scaled_dot_product_*` selective recompute.

**Why:** our DDP sweep showed batch=32 (vs our current batch=16) gives +5% tok/s but doubles memory. Selective recompute could enable batch=32 at current memory. Combined with T-1/T-2 wins could stack to 40k.

**Request for v2:** which recompute strategies are most cost-effective on attention-light hybrid architectures? Our OdinFlat is 2 attention layers + 12 conv layers; the conventional "recompute attention" doesn't apply cleanly.

---

### 3.4 SPECTRA + proposed trust-cap composition

**What:** your E8 proposes a post-NorMuon trust cap on `||lr·update|| / ||weight||`. We already apply SPECTRA post-clip at `clip_norm=1.0` (per-tensor norm cap after NS).

**Question:** these are orthogonal (SPECTRA caps norm, trust-cap caps ratio) but applied sequentially they may double-clip. Which order? Do they compose cleanly? Does SPECTRA subsume trust-cap at high-norm params?

**Implementation note:** `halo_training/spectra.py` currently runs after NS, before the LR multiplication. Your trust-cap would need to run after LR multiplication, just before param update.

**Request for v2:** mathematical analysis of composition. Ideally code-level.

---

### 3.5 fp16-only stability techniques beyond our current stack

**Hard constraint:** bf16 is not supported on gfx1151 (we measured 24% slower + compile crashes).

**Our current fp16 stability stack:**
- `GradScaler(growth_interval=500)`
- `logit_softcap=30` (post-logit cap)
- `attn_softcap=50` (pre-softmax)
- `z_loss=1e-4` (logsumexp² regularization)
- `--activation-monitor` (per-layer maxabs JSONL)
- `iter_scales.clamp(-4, 4)` inside looped-model forward
- `max_grad_norm=1.0` (tightened to 0.8 on resume)
- SPECTRA post-clip at 1.0
- StabilityGuard with NaN forensics dump + rollback

**Question:** given fp16 is the only option, what additional techniques would you recommend? Stochastic rounding? Kahan summation in optimizer? Mixed fp16/fp32 momentum? Two-pass fp32 reductions at specific layers?

This is particularly relevant to Phase C/G divergence since fp16 headroom is our primary constraint.

---

## 4. Priority disagreements

Where we'd rank differently than your §24 execution order.

### 4.1 Backward profile + compiled autograd should be Phase A (earlier)

**Your order:** Phase A instrumentation → Phase B loss → Phase C optimizer → Phase D hidden kernels.

**Our view:** the biggest single lever (compiled autograd, 35.3% bucket) is unclassified in your plan. Add it to Phase A as a feasibility gate. If compiled autograd works, it alone may get us most of the way to target without touching loss or hidden kernels.

**Proposed reordering:**
- Phase A: instrumentation + **backward profile + compiled autograd feasibility**
- Phase B: loss path (if still needed after A)
- Phase C: optimizer + DDP overlap
- Phase D: hidden kernels (only if A-C underdeliver)

---

### 4.2 Drop E15 (hipBLASLt)

**Your priority:** "only if earlier phases fail."

**Our view:** drop entirely. hipBLASLt epilogues (GELU, SiLU, bias) operate on full GEMM output. SwiGLU requires splitting the output in half and doing a pairwise product — not expressible as a standard epilogue. A specialized `w_gate_up + split + silu*up` fused kernel would be a multi-week custom-kernel project on gfx1151, which we've established we shouldn't do.

**Request for v2:** either drop E15 or specify the exact epilogue you think is applicable and show a 2-pager sketch of how it maps to SwiGLU.

---

### 4.3 Gate E11 (layer-selective) behind replay findings

**Your priority:** medium, parallel experiment.

**Our view:** 14 layers × 3 kernel types × on/off per layer = combinatorial explosion. Without a prior (a specific layer showing instability), searching this space is ~40 hours of DDP compute.

**Proposed:** gate E11 behind T-3.1 replay harness. Only run E11 if the replay harness localizes divergence to specific layers. Otherwise not worth it.

---

### 4.4 Your E14 (sync reduction) should be split

**Your E14 covers both `.item()` removal and logging reduction.**

Our view: these have different ROIs.
- `.item()` removal on hot paths (optimizer step, per-param telemetry): +1-3%
- Logging reduction (every N steps instead of every step): negligible, arguably bad for observability

**Proposed:** rename to "sync-point audit" and explicitly scope it to hot-path scalar reads, not logging frequency.

---

## 5. Our locked plan (T-0 → T-4)

Full transparency per user direction. Here's what we're executing.

**Scope:** ~3 weeks of engineering before Sprint 3A launches. Payoff: Sprint 3A (~61h) + Sprint 3B (~77h) both run ~20-30% faster, saving 25-50h of compute and unlocking `--optimize-kernels` long-term.

### Phase T-0 — Instrumentation foundation (2-3 days)

- **T-0.1** Backward-pass profiler (read-only) → `docs/perf/backward-breakdown.md`
- **T-0.2** Per-parameter NorMuon telemetry (code addition, deferred until v2)
- **T-0.3** Sync-point audit (read-only grep) → `docs/perf/sync-point-audit.md`
- **T-0.4** Graph-break inventory (read-only `TORCH_LOGS`) → `docs/perf/graph-breaks-inventory.md`
- **T-0.5** Granular `--ak-*` flags (code addition, deferred until v2 for flag taxonomy guidance)

**Currently executing:** T-0.1, T-0.3, T-0.4 in parallel with your review of this document.

### Phase T-1 — Quick wins (1-2 days)

- **T-1.1** Validate `--use-fused-zloss` on native hidden path at 2000 steps
- **T-1.2** Remove identified sync points (from T-0.3 findings)
- **T-1.3** Fix high-impact graph breaks (from T-0.4 findings)

**Exit gate:** tok/s ≥ 32.5k (3-5% improvement).

### Phase T-2 — Big levers (4-6 days)

- **T-2.1** Compiled autograd — **5-6 day budget per user direction, highest priority**
- **T-2.2** `triton_op` rewrite of fused_swiglu
- **T-2.3** DDP allreduce overlap (`gradient_as_bucket_view`, bucket tuning, `no_sync()`)

**Exit gate:** tok/s ≥ 36k.

### Phase T-3 — Replay + divergence recovery (4-6 days)

- **T-3.1** Replay bundle + `scripts/replay_step.py`
- **T-3.2** Warm-start `--optimize-kernels` experiment (research-brief priority 6 + your E10)
- **T-3.3** Post-NorMuon trust cap + `w_gate_up` staging (conditional on T-0.2 telemetry)
- **T-3.4** `triton_op` SwiGLU re-test in full integration

**Exit gate:** `--optimize-kernels` passes 2000-step gate for at least one config.

### Phase T-4 — Stack integration (2-3 days)

- **T-4.1** Best-stack assembly (Stack A: loss-only; Stack B: + kernels if T-3 succeeds)
- **T-4.2** 2000-step gate on best stack
- **T-4.3** Update STATUS.md + CONSTRAINTS.md + AGENTS.md

### Phase T-5 — Launch Sprint 3A/3B

With whatever stack passes T-4.2.

### Risk-ranked outcomes

| Outcome | Probability | tok/s |
|---|---:|---:|
| Stack A (loss + compiled autograd + DDP overlap, no hidden kernels) | 70% | 36-38k |
| Stack B (+ warm-start or trust-cap recovery of `--optimize-kernels`) | 25% | 38-41k |
| Fall back to current locked recipe | 5% | 31k |

---

## 6. Questions where your v2 would most improve the plan

Eight concrete asks. Free-form answers welcome — we'd rather have opinionated prose than a checkbox.

### 6.1 Compiled autograd + custom_op + ROCm
Our biggest expected lever. What's your experience with `torch._dynamo.config.compiled_autograd = True` + `torch.library.custom_op` (with `register_autograd`)? Known failure modes? Does it work on ROCm 7.12 specifically? Any documented incompatibility with DDP bucket hooks?

### 6.2 NorMuon + trust-cap + SPECTRA mathematical composition
Given SPECTRA already caps per-tensor norm at 1.0 after NS but before LR multiplication, and your proposed trust-cap runs after LR multiplication on `||update||/||weight||` ratio — are these orthogonal safety mechanisms or do they double-clip? What's the mathematically correct order? Does SPECTRA dominate trust-cap above a critical norm?

### 6.3 Alternative divergence hypotheses
Given §2's corrected numbers (backward 35%, loss 5-8%, DDP 5%, matmul non-addressable), your H1-H10 priority ranking should shift. Given what you know about Phase C (step 250 divergence, grad 272 at DDP scale, lr_2d=5e-3) and Phase G (step 750, lr_2d=2e-3), what additional hypotheses should we consider? We have H4-H8 in our research brief but are open to new ideas.

### 6.4 Warm-start experiment design
T-3.2 plans: native train 500 steps, checkpoint, resume with `--optimize-kernels`. Tests your H5 (warmup-local). Does this design test what you think it tests? Better designs? (e.g. should we instead *warm-start only the FFN gate weights* rather than all params?)

### 6.5 Chunked fused CE+z-loss extensions
Our `ChunkedLinearCrossEntropyLoss` already supports `z_loss_weight`. Your E2 proposes "factorized fused CE+z-loss." Given we have the primitives, what specific extensions would you recommend? Reference implementations beyond Liger?

### 6.6 fp16-only stability alternatives
bf16 is unavailable on gfx1151. Given our existing stack (GradScaler, softcap×2, z-loss, activation monitor, SPECTRA, StabilityGuard), what additional fp16-only techniques would you add? Stochastic rounding? Kahan summation in NorMuon momentum? Per-layer fp32 fallback zones?

### 6.7 Realistic gfx1151 throughput ceiling
Given no MFMA, 40 CUs, 256-bit LPDDR5X-8000, 6 MB L2, 64 KB LDS/CU, on a 122M model at batch=16/block=512 — is 40k aspirational or achievable? What's your back-of-envelope roof from memory bandwidth alone? Ours: 256b × 8 GT/s × 2 channels ≈ 512 GB/s peak, ~300 GB/s effective; at fp16 SwiGLU 2×2048 FFN this should support ~X tok/s ceiling but we haven't done this math carefully.

### 6.8 Triton-on-ROCm ceiling
Our Triton fused_swiglu hits HIP parity (0.99×) but not beyond. In your experience, does Triton on ROCm 7.12 / gfx1151 generally plateau at rocBLAS/HIP levels, or can it exceed with the right tuning configs? Are there specific autotune configs (BLOCK_M, BLOCK_N, num_warps, num_stages) that tend to work on RDNA 3.5 but not on datacenter GPUs?

---

## 7. Closing + response ask

### What we want from v2

- Revise your §4 (root-cause assessment) and §5 (hypotheses) given §2's corrected numbers
- Recompute §24 execution order priorities
- Drop E15 unless you can show a concrete epilogue→SwiGLU mapping
- Add E-compiled-autograd and E-ddp-overlap families (or tell us why not)
- Answer the 8 questions in §6
- Provide sharper failure-mode predictions — what observations would disprove your top hypothesis?

### What we'll do in parallel

We're executing T-0.1, T-0.3, T-0.4 now (observation-only, findings will be in the repo before your v2 lands). T-0.2 and T-0.5 (code-touching) are deferred to incorporate your v2 guidance on telemetry taxonomy and flag design.

### Timing

Not blocking. Take the time you need. When v2 arrives we'll read it against this document and update our plan. If v2 materially shifts priorities we'll re-consult our user before committing to the new order.

### Open invitation

If §2's corrected numbers change your core conclusions — if your H1 is no longer priority 1 given that backward is the elephant, or if you now think compiled autograd is the real 40k path — tell us explicitly. We'd rather have an honest pivot than a polished-but-wrong plan.

---

## Appendix A — Raw step profile (for your reference)

From `docs/perf/odinflat-step-profile.md`, commit SHA at time of measurement: `[earlier commit]`. Measured on Sprint 1.5 C3 recipe, native path (no `--optimize-kernels`), dolma-10b-odin32k.bin, batch=16, block=512, accum=8, DDP across 2 Strix Halo machines over TB4.

| Section | CUDA time (10 opt steps = 80 microsteps, 40.244s total) | % of step |
|---|---:|---:|
| aten::mm (all matmuls, fwd+bwd+NorMuon) | 15.234 s | 37.85% |
| Compiled forward graph (CompiledFunction CUDA) | 6.523 s | 16.2% |
| Compiled backward graph (CompiledFunctionBackward CUDA) | **14.213 s** | **35.3%** |
| NorMuon optimizer step | 5.05 s | 12.5% |
| aten::logsumexp (z-loss fwd, active) | 4.456 s | 11.1% |
| LogsumexpBackward | 2.256 s | 5.6% |
| CausalConv1dFnBackward (HyPE conv bwd) | 1.831 s | 4.54% |
| CausalConv1dFn (HyPE conv fwd) | ~0.4 s | 1.0% |
| DDP allreduce (estimated, gloo over TB4) | ~0.2 s | ~5% |

Notes:
- Totals overlap because `aten::mm` appears under both its own bucket and inside CompiledFunction regions.
- Attention forward derived from FLOP ratio: ~25% per-layer × ~30% layer-forward-wall = **~7.5% of step wall**. QKV fusion upside ~1.5%, below our 3% ship gate.
- Per-layer FLOPs (d=768, n_heads=12, n_kv_heads=4, block=512, batch=16): attention 12.88 GFLOP, FFN 38.65 GFLOP. FFN dominates — which is why SwiGLU was the target and why the silent-freeze of `w_gate_up` had such a blast radius.

---

## Appendix B — Phase I Triton ship-gate bench

From `docs/perf/triton-swiglu-ship-gate-bench.md`.

Production shape: `(16, 512, 2048)`, dtype: `torch.float16`. 200 timed iters + 50 warmup per config.

| Comparison | fwd speedup | fwd+bwd speedup |
|---|---:|---:|
| eager-vs-autograd-hip | 1.693× | 1.450× |
| eager-vs-triton | 1.634× | 1.434× |
| autograd-hip-vs-triton | 0.977× | 0.986× |

Raw numbers:
- eager fwd: 731.4 µs; autograd-HIP fwd: 432.0 µs; Triton fwd: 448.4 µs
- eager fwd+bwd: 2730.6 µs; autograd-HIP fwd+bwd: 1882.9 µs; Triton fwd+bwd: 1906.6 µs

Both HIP and Triton sit at the memory-bandwidth ceiling for this shape. No kernel-level win available.

---

## Appendix C — Phase B remediation summary (what's in the safe path today)

Six fixes shipped 2026-05-11, validated by 21-probe empirical audit (0 frozen params across 5 Odin models at 50-step scale):

| Fix | Component | Mechanism |
|---|---|---|
| B.1 | `_FusedSwiGLUReplacement` | wired to `torch.library.custom_op silu_gate_mul` + `register_autograd` |
| B.2 | `_SiluGateMulReplacement` | same custom_op backend |
| B.3 | `_LayerNormReplacement` | uses `F.rms_norm` (native PyTorch) |
| B.4 | `_FusedQKVAttentionReplacement` | wired to QKV fusion custom_op with autograd |
| B.4b | `_FusedResidualRMSNormBlockReplacement` | wired to fused_res_rmsnorm custom_op |
| B.5 | `_CrossEntropyHIP` | extended to bake z-loss gradient into backward (enables `--use-fused-zloss`) |
| B.6 | 7 new autograd-safety tests | regression coverage |

Eight registered custom_ops in `kernels/hip/_torch_ops.py`:
- rmsnorm, rotary_emb_fp32, silu_gate_mul, fused_res_rmsnorm, selective_scan, griffin_scan, fused_ple_gate, fused_gated_conv

Runtime preflight (`_autokernel_autograd_preflight` in `scripts/train_ddp.py`) validates autograd flow on every launch, aborts training if any pattern severs gradients. Confirmed firing on 4 separate Phase C/G launches.

---

## Appendix D — Locked Sprint 3A recipe (your reference point for "baseline" training)

```bash
EXTRA_FLAGS='--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --auto-eval'
# No --optimize-kernels. No --use-fused-zloss (opt-in, unvalidated).

TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
  bash scripts/launch_ddp.sh
```

Measured: ~31.3k tok/s aggregate, loss 3.15 at step 2000 on dolma-10b-odin32k, Sprint-3A-confirm (2000 steps). This is the "31.3k baseline" we're starting from.

---

## Appendix E — Key file index

### Docs (perf)
- `docs/perf/odinflat-step-profile.md` — the profile breakdown cited throughout §2
- `docs/perf/triton-swiglu-ship-gate-bench.md` — Phase I result
- `docs/perf/phase-c-final-analysis.md` — Phase C divergence (OdinFlat)
- `docs/perf/phase-g-findings.md` — Phase G divergence (OdinHalo)
- `docs/perf/autokernel-audit-2026-05-11-synthesis.md` — 21-probe empirical audit
- `docs/perf/autokernel-deep-analysis.md` — root-cause synthesis for silent-freeze

### Docs (research)
- `docs/research/autokernel-stability-research-brief.md` — our full 12.3K-word brief for external agents
- `docs/research/autokernel_halo_strix_40k_throughput_experiment_guide.md` — your v1 guide (the subject of this response)

### Code (authoritative)
- `kernels/hip/_torch_ops.py` — 8 registered custom_ops with autograd
- `kernels/hip/chunked_linear_cross_entropy.py` — `ChunkedLinearCrossEntropyLoss` with z_loss_weight
- `kernels/triton/fused_swiglu.py` — current Triton SwiGLU (torch.autograd.Function-wrapped)
- `autokernel/_patterns.py` — 7 Replacement classes, all post-Phase-B autograd-safe
- `autokernel/triton_base.py` — `TritonAutogradFunction` base class (candidate for triton_op migration)
- `halo_training/normuon.py` — NorMuon optimizer
- `halo_training/spectra.py` — SPECTRA post-clip
- `halo_training/trainer.py` — training loop including chunked_ce_handles_zloss path
- `scripts/train_ddp.py` — DDP trainer entry point with autograd preflight
- `scripts/launch_ddp.sh` — production DDP launcher

### Knowledge
- `knowledge/training/autograd_safety_hip_kernels.md` — principle doc for kernel authoring rules
- `knowledge/kernels/triton_author_guide.md` — Triton authoring workflow
- `knowledge/hardware/amd_rdna35_strix_halo.md` — gfx1151 hardware reference
- `knowledge/training/fp16_stability_gfx1151.md` — our fp16 stability stack
- `knowledge/training/normuon_throughput_gfx1151.md` — NorMuon NS fp16 win (Sprint 1.1)

---

*End of response. Awaiting v2.*
