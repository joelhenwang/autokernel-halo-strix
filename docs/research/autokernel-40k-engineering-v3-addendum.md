# v3 request addendum to research engineer

**Source exchange:**
- v1: `docs/research/autokernel_halo_strix_40k_throughput_experiment_guide.md` (engineer's initial guide)
- engineering response: `docs/research/autokernel-40k-guide-engineering-response.md` (our reply with 6 data corrections + 8 questions)
- v2: `docs/research/autokernel_40k_revised_engineering_handoff_v2.md` (engineer's revised handoff)
- **this doc (v3 request):** addendum with T-0 data that v2 didn't have access to + three targeted questions

**Repo SHA at time of writing:** `664fd20` (main, post-T-0 findings)
**Purpose:** give engineer the T-0 data we gathered after they wrote v2, and request targeted re-assessment on three specific areas. Not a re-write of the engagement.
**Blocking:** we are holding T-1 execution until v3 arrives.

---

## 1. Meta — what this is and isn't

**Is:**
- A focused addendum with ~3 pages of new empirical data
- Three specific questions where your v2 expected-values should be recalibrated
- A scope-limiter naming the v2 content we've already adopted (so v3 doesn't waste effort re-deriving those)

**Isn't:**
- A rejection of v2. v2 is a substantial improvement over v1 and we're adopting most of it verbatim.
- A full re-briefing. You have sufficient context.

**What we want from v3:**
- Revised expected-value for compiled autograd (your H1) given data showing backward is near-theoretical
- Revised scope for E10 custom-op autocast testing given our 2-break-location inventory
- Honest engineering view on 40k realism given our T-0.1 ceiling data
- (optional) any additional hypotheses or reframings surfaced by this data

**What we're doing while we wait:**
- T-0.2 (per-param NorMuon telemetry) and T-0.5 (granular `--ak-*` flags) are blocked pending v3 because we want to incorporate your feedback on flag taxonomy and telemetry schema
- T-1 onward is blocked pending v3
- Research brief hypothesis list updated with H11/H14/H17 from your v2

---

## 2. T-0 data you didn't have access to

Three observation-only investigations completed between your v2 and this addendum. Each has a committed findings doc in the repo. Key results:

### 2.1 Backward is near-theoretical (T-0.1)

**Source:** `docs/perf/backward-breakdown.md` (re-analysis of the committed `profile-summary.txt` with attention to backward-tagged operations)

**Headline:**
- CompiledFunction forward: 6.523 s of 10-opt-step CUDA wall (16.2%)
- CompiledFunctionBackward: 14.213 s (35.3%)
- **Backward/forward ratio: 2.18×**
- Theoretical minimum backward/forward ratio for a dense transformer: 2.00× (every forward matmul needs two backward matmuls — input grad and weight grad)
- Observed 2.18× implies **~9% overhead above theoretical minimum**

**Backward decomposed:**
- rocBLAS transposed-shape GEMMs (`Cijk_Alik_Bljk` / `Cijk_Ailk_Bjlk`, HHS_BH): 17.49% of step wall — the `w.T @ g` and `g.T @ x` gradient matmuls
- Inductor-generated `triton_poi_fused__unsafe_view_cat_mul_silu_silu_back` (SwiGLU backward): 2.92% — **Inductor is already fusing SwiGLU backward correctly**
- LogsumexpBackward (z-loss derivative): 5.6%
- CausalConv1dFnBackward (DaoAILab): 4.54%
- MmBackward0 (non-compiled, likely NorMuon NS internal matmuls): 5.2%

**Implication for your v2 H1:**
Your H1 says "backward graph fragmentation / compiled backward inefficiency is a major stackable performance lever." Our data shows the inefficiency is ~9% of backward wall, or ~3% of step wall. Compiled autograd's upper bound on this machine is probably 2-8%, not 5-15%.

**Implication for your §7.7:**
"40k aggregate is aspirational but not absurd" — we think this understates the difficulty. If compiled autograd is capped at ~3-5% real gain, loss path is 5-8%, sync removal is 2-4%, DDP overlap is 1-3%, batch=32 is 0-5%, NorMuon optimization is 2-5% — the sum even at upper bound is 13-30%, and these are not fully additive (overlapping wall-time categories). 40k requires +28%. It's at or beyond the edge.

**Our revised honest estimate:** 35-37k is realistic; 40k is the upper-bound outcome probability ~15%.

### 2.2 Graph-break surface is tiny (T-0.4)

**Source:** `docs/perf/graph-breaks-inventory.md` (single-node smoke with `TORCH_LOGS='graph_breaks,recompiles'`, 10 opt steps on OdinFlat full Sprint 1.5 C3 recipe)

**Headline:**
- **Only 2 distinct graph break locations across the entire model forward path**
- **Only 1 recompile across 10 opt steps**
- 401 log lines total (mostly Inductor autotune output)

**The 2 break locations:**

| # | Location | Cause | Fix path |
|---|---|---|---|
| 1 | `models/components/conv_blocks.py:284` | `@torch.compiler.disable`'d HIP `kernel_fn` from `kernels/hip/fused_rope_gate_mul.py` | Convert to `torch.library.custom_op` + `register_autograd` (same pattern as Phase B ops) |
| 2 | `torch/_library/custom_ops.py:698` triggered from `conv_blocks.py:296` | DaoAILab `causal_conv1d_fn` using older-style custom op registration | Wrap DaoAILab's call in our own `torch.library.custom_op` shim |

Occurrence counts: break #1 fires once (conv_blocks.py), break #2 fires 4 times (across compile cache entries [1/0]-[4/0]).

**The 1 recompile:**
`odin_flat.py:64 forward` recompiles at step 2 because guard `v_prev is None` fails on the first transition to a non-None v_prev tensor. One-time cost, amortizes to zero across a 2000-step run.

**Implication for your v2 H18 ("HyPE conv/depth-KV buffer mutation causes repeated recompiles"):**
Data doesn't support. We see 1 recompile in 10 opt steps, for the v_prev transition (not HyPE-specific). Depth-KV list mutation — if it's happening inside compile zones — isn't producing recompile storms that the logger sees. H18 can be deprioritized or dropped.

**Implication for your v2 E10 (custom-op autocast and fp32 islands):**
v2 §6.E10 proposes autocast/precision variants on "each custom op" (implicitly all 7 we have). But the graph-break inventory shows only these 2 ops are at the compiler-visibility boundary. The 5 Phase-B-fixed ops (`silu_gate_mul`, `rmsnorm`, `fused_res_rmsnorm`, `rotary_emb_fp32`, `fused_ple_gate`) compile cleanly — they may still have autocast issues, but they're not graph-break-causing.

**Our revised E10 scope proposal:** test autocast on these 2 ops first, expand to the Phase-B ops only if Phase C/G divergence persists after fixing the 2 obvious candidates.

### 2.3 SPECTRA is the biggest sync offender (T-0.3)

**Source:** `docs/perf/sync-point-audit.md` (static grep + profile cross-reference)

**Headline:**
- Profile shows 870 `aten::item` calls across 10 opt steps = **87 syncs/step**
- CPU wall: 79% (32.547s of 41.174s) in `hipMemcpyWithStream` — the cost of these syncs
- CPU wall ≠ CUDA wall on APU (CPU mostly idle during GPU work), but each sync serializes critical-path latency

**Distribution of syncs:**

| File:Line | Frequency | Purpose | Sync count/step |
|---|---|---|---:|
| `spectra.py:88` `sigma1.item()` | per-2D-param-per-opt-step | gate clip fast path | **~50** |
| `chunked_linear_cross_entropy.py:73` `valid_global.sum().item()` | per forward | loss denominator | 8 |
| `trainer.py:443` `loss.item()` | per microstep | running_loss aggregation | 8 |
| `train_ddp.py:1468/1502/1550` | per opt step | JSONL logging | 3 |
| other | various | various | ~18 |

**Proposed fix for SPECTRA (eliminates ~50 syncs/step):**

```python
# before (spectra.py:88)
sigma1_val = sigma1.item()
if sigma1_val * (1.0 / safety_margin) <= clip_norm:
    return M
scale = (clip_norm * safety_margin) / max(sigma1_val, 1e-12)
return M * scale

# after (branchless, no sync, same semantics)
scale = torch.clamp(clip_norm * safety_margin / torch.clamp(sigma1, min=1e-12), max=1.0)
return M * scale
```

When no clip is needed, `scale = 1.0` and `M * scale` is a no-op kernel dispatch. Marginal extra kernel call cost, zero sync.

**Estimated total T-1.2 savings: 2-4% step wall.**

---

## 3. Implications for v2 priority ranking

Summarizing how the T-0 data should shift v2's hypothesis ranking:

| v2 Hypothesis | Your v2 priority | T-0 evidence | Suggested v3 revision |
|---|---|---|---|
| H1 backward fragmentation | **major stackable** | backward at 2.18× fwd for 2× FLOPs (9% overhead above theoretical) | Demote expected gain to 2-8%, not 5-15%. Still worth 5-6 day T-2.2 budget as infrastructure. |
| H2 DDP overlap | 1-5% or "huge if no_sync broken" | profile shows DDP ≤5% of step (estimated residual) | Cap realistic expected gain at <5%. Keep "huge if broken" tripwire. |
| H3 fused z-loss | 3-8% likely | our profile math (eliminating redundant pass, not logsumexp itself) | Accept your 3-8% range |
| H4 NorMuon update dynamics | stabilization | untested pending T-0.2 telemetry | Unchanged; co-priority with H11 |
| H5 custom-op dtype drift | stabilization | superseded by H11 | Fold into H11 |
| H6 warmup-local | medium | tested by T-3.2 warm-start matrix | Unchanged |
| H7 batch=32 | 0-5% | our DDP sweep showed +5% at batch=32 | Accept 0-5%; T-2.1 early test per user direction |
| H8 hot-path syncs | 1-3% | our T-0.3 shows ~2-4% | Accept your range, slight revision upward |
| H9 hidden kernels | compiler visibility only | our T-0.4 confirms only 2 ops break; Phase I confirmed SwiGLU at HIP parity | Keep as-is |
| H10 factorized fused CE | 0-8%, uncertain | we have ChunkedLinearCrossEntropyLoss w/ z_loss_weight already | Depend on E3 profiling results |
| H11 **custom-op autocast** | mentioned | **promoted — most probable Phase C/G mechanism** | **Elevate to H1-H4 priority level; scope to 2 unfixed ops first** |
| H12 update-ratio spikes | mentioned | untested pending T-0.2 | Keep as-is; co-depends on H4 |
| H13 LR warmup alignment | mentioned | testable by T-3.2 | Keep as-is |
| H14 optimizer state mismatch | mentioned | directly tested by T-3.2 preserved-vs-fresh | Keep, sharpen prediction |
| H15 DDP bucket timing | mentioned | DDP profile pending | Keep as-is |
| H16 fused z-loss denominator bug | mentioned | Phase C v2 (no fused zloss) also diverged, weakening this | Demote |
| H17 forward breaks fragment backward | mentioned | our T-0.4 shows only 2 breaks; T-0.1 shows backward near-theoretical | Demote; data weakens mechanism |
| H18 HyPE recompile storm | mentioned | **1 recompile / 10 opt steps (one-time v_prev transition)** | **Drop; data disproves** |

**Most important priority shift:** H11 (custom-op autocast) should co-priority with H4 (NorMuon amplification). Our Phase B ops don't use `register_autocast` — this is a concrete, fixable gap that matches the Phase C/G divergence pattern (correct learning through ~step 700, then accumulated error breaks through fp16 headroom).

---

## 4. Three targeted v3 questions

These are the only questions we need answered for v3. The rest of v2 we're adopting as-is.

### Question 1: Compiled autograd expected-value given backward-near-theoretical

Our T-0.1 data shows:
- Backward/forward wall ratio: 2.18×
- Theoretical minimum (pure GEMM): 2.00×
- Observed overhead above theoretical: ~9% of backward, ~3% of step wall
- Inductor already fusing SwiGLU backward into `triton_poi_fused__unsafe_view_cat_mul_silu_silu_back` (2.92%)

**Question:** given backward appears to be running at near-theoretical efficiency (2.18× forward for 2× forward FLOPs), what is your revised expected-value for compiled autograd in this repo? Specifically:

- **(a)** Is there a mechanism by which compiled autograd recovers more than ~8% of step wall given this data? If yes, what's the mechanism?
- **(b)** Does your experience with `torch._dynamo.config.compiled_autograd = True` on other workloads show speedups beyond 8% when the baseline backward is already at 2.0-2.2× forward?
- **(c)** Should we move T-2.2 (compiled autograd, 5-6 day budget) **after** T-2.4 (NorMuon implementation optimization, which has a clearer data-supported ~3-5% target)? Or is compiled autograd still your top pick for T-2 given the infrastructure value beyond throughput?

### Question 2: E10 autocast scope — narrow to 2 unfixed HIP ops?

Our T-0.4 shows only 2 HIP ops cause graph breaks:
1. `fused_rope_gate_mul.kernel_fn` — uses `@torch.compiler.disable`, not Phase-B-fixed
2. DaoAILab `causal_conv1d_fn` — uses old custom_op semantics, external extension

The 5 Phase-B-fixed ops (`silu_gate_mul`, `rmsnorm`, `fused_res_rmsnorm`, `rotary_emb_fp32`, `fused_ple_gate`) compile cleanly — no graph break visible to Dynamo.

**Question:** given this, should v2's E10 (custom-op autocast + fp32 islands) narrow to:

- **Tier 1:** the 2 unfixed HIP ops (rope_gate_mul + causal_conv1d). Fix both by converting to proper `torch.library.custom_op` + `register_autograd` + `register_autocast(device_type="cuda", cast_inputs=torch.float16)`. These are prerequisites for T-2.2 compiled autograd anyway.
- **Tier 2:** the 5 Phase-B-fixed ops. Add `register_autocast` decorators. Only run full parity matrix if Phase C/G divergence persists after Tier 1.

Or do you see reasons to audit all 7 ops in one pass regardless of graph-break visibility? (We're worried about missing a stability-relevant autocast mismatch in a Phase-B op that doesn't happen to break the graph.)

### Question 3: 40k realism

User direction is to keep 40k as the nominal target regardless of the realistic ceiling. But for internal calibration and for STATUS.md framing, we want your honest engineering view.

Summing plausible stackable upside on T-0 data:
- fused z-loss: 5-8%
- sync removal: 2-4%
- DDP overlap (if no_sync OK): 1-3%
- batch=32 if fits: 0-5%
- compiled autograd: 2-8% (per Q1)
- NorMuon implementation: 2-5%
- hidden-kernel recovery (Stack D): 0-3% end-to-end beyond the above

Upper bound sum: ~35%. But these categories overlap in wall-time attribution (e.g., compiled autograd might subsume some sync-removal savings). Realistic non-overlapping sum: ~20-25%.

31.3k × 1.25 = 39.1k. 31.3k × 1.20 = 37.6k.

**Question:** do you agree with:
- **Realistic ceiling: 35-37k** (based on plausible upper-bound stacks with realistic overlap discount)
- **40k: possible but requires every lever at upper-bound AND low overlap-discount**
- **STATUS.md should commit to "target 40k, success criterion 36k+" rather than "success = 40k"?**

Or do you see a specific combination where 40k is confidently attainable given this data?

---

## 5. What we've already adopted from v2 (scope-limiter)

So v3 doesn't waste effort re-deriving these:

- **§5.1 flag taxonomy** — 16 `--ak-*` flags, adopting as-is for T-0.5
- **§5.5 NorMuon telemetry schema** — JSON schema, adopting as-is for T-0.2
- **§7.2 SPECTRA + trust-cap composition math** — adopting for T-3.3
  - Order: NS → SPECTRA → effective LR → trust cap → update
  - Dominance conditions correctly derived
- **§11 pre-committed interpretation table** — adopting as decision tree for T-1 through T-4
- **§13 production command philosophy** — no monolithic `--optimize-kernels` in Sprint 3 launch; use granular flags
- **Phase 0 → Phase 4 execution order** — adopting with our internal T-0 → T-4 numbering, reordered per user direction (batch=32 earlier in T-2)
- **H11, H14, H17 hypotheses** — added to `docs/research/autokernel-stability-research-brief.md §10.5`
- **H18 dropped** — T-0.4 data doesn't support

---

## 6. Our plan while blocking on v3

**Not executing:**
- T-0.2 (NorMuon telemetry) — waiting for your answer on schema refinements
- T-0.5 (granular flags) — waiting for your answer on Q1 (which may shift flag priorities)
- T-1 onward — blocked

**Already completed (visible to you now):**
- T-0.1 backward profile → `docs/perf/backward-breakdown.md`
- T-0.3 sync-point audit → `docs/perf/sync-point-audit.md`
- T-0.4 graph-break inventory → `docs/perf/graph-breaks-inventory.md`
- Research brief hypothesis list expanded with H11/H14/H17

**Expected timeline:**
- If v3 lands within 3-5 days: Sprint 3 launches ~3 weeks from now
- If v3 takes longer: we may unblock T-0.2/T-0.5 with best-available info and proceed, consulting user first

---

## 7. Files delivered since v2

For your review:

**New since v2 (direct links):**
- `https://github.com/joelhenwang/autokernel-halo-strix/blob/main/docs/perf/backward-breakdown.md` (T-0.1)
- `https://github.com/joelhenwang/autokernel-halo-strix/blob/main/docs/perf/sync-point-audit.md` (T-0.3)
- `https://github.com/joelhenwang/autokernel-halo-strix/blob/main/docs/perf/graph-breaks-inventory.md` (T-0.4)
- `https://github.com/joelhenwang/autokernel-halo-strix/blob/main/docs/research/autokernel-40k-guide-engineering-response.md` (our response to v1)
- `https://github.com/joelhenwang/autokernel-halo-strix/blob/main/docs/research/autokernel-40k-engineering-v3-addendum.md` (this document)

**Updated:**
- `https://github.com/joelhenwang/autokernel-halo-strix/blob/main/docs/research/autokernel-stability-research-brief.md` §10.5 (added H11/H14/H17)

---

*End of addendum. Awaiting v3 with targeted responses to §4 questions 1-3.*
