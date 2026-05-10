---
title: "ZAYA1-8B: Applicable findings for Odin"
domain: training
type: reference
status: active
tags: [zaya1, moe, reasoning, rl, rlvr, post-training, ttc, markovian-rsa, ap-trimming, ccga, router-replay, dppo, dr-grpo, maxrl, reasoning-aware-pretraining, amd, mi300x]
paper: "ZAYA1-8B Technical Report (Zyphra, May 2026)"
paper_local: "../../docs/research/zaya1_8b_technical_report/zaya1_8b_technical_report.md"
companion_arxiv:
  - "2511.17127 — Training Foundation Models on a Full-Stack AMD Platform (ZAYA1-base)"
  - "2510.04476 — Compressed Convolutional Attention (CCA/CCGQA)"
model_card: "https://huggingface.co/Zyphra/ZAYA1-8B"
related:
  - imu1_recipe_2026.md
  - grpo_family_2026.md
  - fp16_stability_gfx1151.md
  - instruct_alignment_techniques_2025_2026.md
  - sft_pipeline.md
  - ap_trimming_recipe.md
  - ../architectures/small_lm_arch_interventions_2026.md
  - ../architectures/parcae_stable_looped_models.md
---

# ZAYA1-8B — Applicable Findings for Odin

## TL;DR

ZAYA1-8B is a **760M-active / 8.4B-total reasoning MoE** trained end-to-end on
AMD MI300X + Pollara. It matches or exceeds DeepSeek-R1-0528 and GPT-5-High on
**HMMT'25 with TTC** at ~1/50th the active-parameter count. Architecturally it
is a different regime from Odin (dense small LM, no MoE, no RL infra yet); but
the **training methodology, RL spine, stability engineering, and TTC harness
contain several high-value items that do transfer**. This note extracts those,
rates each for Odin applicability (dense / small / gfx1151 / no-MoE), and
flags concrete next steps.

**One-line recommendation.** Do not copy ZAYA1's architecture. Do port, in
approximate order of ROI:

1. **Answer-Preserving (AP) trimming** for any reasoning-data inclusion at our
   short block sizes — standalone recipe in
   [ap_trimming_recipe.md](ap_trimming_recipe.md).
2. **Streaming LZ77 + rare-token canaries** as a reward/loss-path sanity gate;
   supplements existing `StabilityGuard`.
3. **Learned residual scaling** (α·x + β at each residual site) — 4·L·D
   params, negligible FLOPs, addresses residual-norm growth with depth. Port
   candidate for OdinFlat and (especially) OdinHalo where the Parcae loop
   re-traverses depth 3× per step.
4. **Best-fit-decreasing bin packing** instead of fixed-boundary truncation
   when we build SFT infrastructure.
5. **Half-RoPE** (50 % of head channels RoPE'd, 50 % position-free) as an
   ablation for OdinFlat's `NoPEGQABlock` — mid-risk, easy to try.
6. If/when we add RLVR: **MaxRL advantage + Dr-GRPO SMTSN aggregation + DPPO
   Binary-TV trust region + no-KL-in-reward + momentum-free Muon** is the
   validated cocktail. Documented below, not yet needed.

Not applicable at our current scale: MoE router designs, PID bias-balancing,
router replay, CCA/CCGQA (we are dense and already use GQA + factorized
embed/head).

---

## 1. What ZAYA1-8B is

| Property | Value |
|----------|-------|
| Architecture | Decoder-only MoE, 40 layers, d=2048 |
| Active / total | **0.76B / 8.4B** (top-1 MoE, 16 experts/layer) |
| Attention | **CCGQA + CCA preconditioner**, 8 Q-heads, 2 KV heads, head_dim=128, 2× Q compression, 8× KV-cache compression vs MHA |
| Position | **50 % RoPE on each head** (half-RoPE, half NoPE) |
| Router | MLP + EDA (depth-weighted averaging), PID-controlled bias balancing |
| Residual | Learned scaling `αx + β` at residual and layer output |
| Tokenizer | Gemma-3 (vocab 262,272) |
| Hardware | 1,024 × AMD MI300X + Pensando Pollara 400 |
| License | Apache-2.0; weights on HF |

Training (ZAYA1-base → midtrain → SFT → 4-stage RL):

| Phase | Context | RoPE base | Tokens | Notes |
|-------|--------:|----------:|-------:|-------|
| Pretrain 1 | 4K | 10K | 8T | Broad web + code + math + multilingual |
| Pretrain 2 | 4K | 10K | 4T | Upweight code/math/reasoning/IF |
| Midtrain | 32K | 1M | 1.2T | **86 % long-CoT** |
| SFT | 131K | 5M | 660B | **75 % long-CoT** + chat/IF/code/math/TTC |
| RL cascade | — | — | — | 232 + 400 + 384 + 464 + 384 steps |

Optimizer: Muon with AdamW RMS matching during pretrain/midtrain/SFT;
**momentum-free Muon** during RL.

---

## 2. Findings × Odin applicability matrix

| # | Finding | Zyphra relevance | **Odin fit** | Priority |
|---|---------|-----------------:|-------------:|---------:|
| A | AP trimming for CoT at short context | High | **HIGH — block=256/512 vs CoT ≥10K** | P0 |
| B | Streaming LZ77 + rare-token canaries | High | **HIGH — drop-in** | P0 |
| C | Learned residual scaling (αx+β per block) | Medium | **HIGH** for OdinHalo looped depth | P1 |
| D | Best-fit-decreasing bin packing at SFT | Medium | **MEDIUM** (no SFT infra yet) | P2 (when SFT) |
| E | Half-RoPE on GQA heads | Medium | **MEDIUM** — ablate on OdinFlat `NoPEGQABlock` | P2 |
| F | Reasoning-aware pretraining (CoT from pretrain on) | High | **MEDIUM** if we target reasoning Odin | P2 |
| G | RL spine (DPPO Binary-TV + Dr-GRPO SMTSN + MaxRL + no-KL-in-reward + momentum-free Muon) | Core | **DEFERRED** — no RLVR infra yet; capture recipe | P3 |
| H | Router replay | Core for MoE RL | **N/A** — Odin is dense | — |
| I | RLVE-Gym + Thompson/IRT difficulty calibration | Medium | **MEDIUM** future RLVR pattern | P3 |
| J | Length-reward design (difficulty-gated, group-relative) | Medium | **MEDIUM** future RLVR pattern | P3 |
| K | Markovian RSA TTC harness | Core | **LOW–MEDIUM** at our scale; interesting for reasoning-Odin | P3 |
| L | Precision matching: BF16 + matched FP32 op set for engine/trainer agreement | Core for RL | **LOW** — we're fp16-only on gfx1151; inverse direction | informational |
| M | Length-bias derivation: signed K1 KL-in-reward + pipeline RL | Core for RL | **LOW** — note and avoid if we do RL | informational |
| N | CCA / CCGQA compressed-latent attention | Core | **N/A** — GQA + factorized already | — |
| O | MLP router + PID bias balancing | MoE | **N/A** | — |
| P | Top-1 MoE, no residual expert | MoE | **N/A** | — |
| Q | AMD MI300X + Pollara training stack proof | Supporting | **N/A** — gfx1151 is a different SKU; note only | informational |

---

## 3. P0: port immediately

### A. Answer-Preserving (AP) trimming

**Problem.** Teacher CoT traces run 10K–30K+ tokens. Our current block sizes
are 256–1024. Naive truncation either drops the answer (training on reasoning
that never concludes) or drops the example (losing reasoning signal). Neither
is good.

**ZAYA1 procedure** (Report §III-A, applied offline to training data, re-run
at each new context length 4K → 32K → 131K):

1. If full conversation fits C — keep.
2. Else trim the **tail** of the final reasoning block just before `</think>`
   and the answer section; keep the head of the CoT and the whole answer.
3. Else drop earlier turns' `<think>` blocks but preserve their answers;
   re-apply step 2.
4. Else drop sample.

**Rationale.** The head of a CoT carries decomposition and planning; the tail
consolidates into the answer. Tail-trimming preserves decomposition + aligned
answer, which trains short-context models on coherent partial reasoning.

**Why it matters for us.** STATUS.md lists "Zero post-training infrastructure"
as the biggest gap. Any reasoning-dataset inclusion — whether in pretraining
(Akter et al. 2025 finding) or SFT — runs headlong into this without
AP-trimming. This is the cheapest of the P0 wins.

**Stand-alone recipe + pseudocode:** [ap_trimming_recipe.md](ap_trimming_recipe.md).

### B. Streaming LZ77 + rare-token canaries

**Purpose.** Reward-path / loss-path sanity gate that catches degenerate
repetition and gibberish regardless of whether the verifier or loss accepts
the output. Flagged rollouts (during RL) or training batches (during
pretraining) have their signal zeroed / logged before the optimizer sees them.

**Spec** (Report §IV-F):

- **Compression canary**: zlib, wbits=−10 (1024-byte LZ77 window), level-1
  deflate, `Z_SYNC_FLUSH` between chunks. Per-chunk ratio
  `r_c = (compressed − flush_overhead) / raw`. Flag if any chunk has
  `r_c < τ_repeat = 0.05`.
- **Rare-token canary**: fraction of tokens whose token-ID falls in top
  {10, 5, 2, 1} % of the tokenizer ID range. Rising fraction precedes other
  failures, cheap to compute.

**Fit for us.** Complements `StabilityGuard` + `--activation-monitor` already
in `halo_training/activation_monitor.py`. Current stack catches NaN and
grad-norm explosions; LZ77 catches *semantically* degenerate generation
(loops, near-verbatim repetition) that still flows valid gradients.

**Port sketch.** Small module `halo_training/content_canaries.py` exposing:

```python
def lz77_min_ratio(token_ids: np.ndarray, chunk_tokens: int = 256) -> float: ...
def rare_token_frac(token_ids: np.ndarray, top_pct: float = 0.10) -> float: ...
```

Wire into the periodic eval loop during pretraining and (future) the RL
rollout pipeline. Emit to the same JSONL as activation stats.

**Mapped to Odin today:** useful during the `sprint3` dolma-10B run (per
STATUS.md) — would have caught the NaN-before-NaN repetitive spans the fp16
incident looked for but didn't isolate.

---

## 4. P1: port soon — learned residual scaling

> **FrankenMoE composition update (2026-05-07):** the learned residual
> scaling port has been promoted into two composite designs covered by
> dedicated specs:
>
> - **FrankenMoE-Flat v1** (TBD spec) — 75M active / 175M total, OdinFlat
>   backbone, MoE + learned residual scaling + half-RoPE + N1 γ. Pretraining
>   knobs only; no SFT/RL.
> - **FrankenMoE-Loop v2** — [spec](../../docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md)
>   — OdinHalo backbone, R2 sticky routing + E1 shared experts + N1
>   γ_{e,i} + Sched-A 2.5-iteration schedule + M3 narrow FFN at iter 2.
>   Implementation blocked on v1 L9.
> - Companion design notes:
>   [knowledge/architectures/looped_moe_design_2026.md](../architectures/looped_moe_design_2026.md)
>   — the "what MoE means in a Parcae loop" question with our committed
>   answers and their rationale.
>
> The standalone learned-residual-scaling port described below remains
> valid for any halo model that doesn't want MoE; FrankenMoE v1/v2 subsume
> it when MoE is desired.

**Spec** (Report §II-A-3):

```
Res-scale(x) = α ⊙ x + β                  # α ∈ R^D, β ∈ R^D, per-site learned
x_{l+1} = Res-scale_res(x_l) + Res-scale_out(Layer(RMSNorm(x_l)))
```

Init α = 1, β = 0 (identity). Total added params: **4 · L · D** — same order
as LayerNorm scales. Zyphra describes it as a cheaper alternative to Qwen's
attention-gating matrix, with the same depth-norm-control benefit.

**Why it fits OdinHalo especially.** In a looped Parcae model with
`mean_recurrence=3` and 6 shared layers, the residual stream traverses 18
"effective" layers worth of cumulative additions per forward pass. Residual
norm growth compounds across iterations — our existing `iter_scales.clamp(±4)`
already concedes this. Learned residual scaling would give the model a
principled, trainable way to regulate that growth without a hard clamp.

**Port sketch.**

- New tensor pair `(alpha, beta)` per residual site in
  `models/components/conv_blocks.py` (HyPEShortConvBlock) and `NoPEGQABlock` /
  `NoPECodaAttention` consumers, or add as a wrapper in `_components.py` and
  gate with a model-level `use_residual_scaling` flag.
- For OdinHalo with shared weights across iterations, the (α, β) pairs should
  still be **per-iteration**, not shared (otherwise we lose the ability to
  regulate per-iteration norm growth). This is the one important design
  choice: do **not** share residual-scale params across Parcae iters.
- Keep `iter_scales.clamp(±4)` — the scaling is learned but fp16 overflow
  prevention remains insurance.

**Validation plan.** 200-step DDP smoke on wikitext-103 OdinFlat first (no
iteration complication), then 200-step OdinHalo. Gate: no regression at step
200; scorecard `avg_bpb` within noise.

---

## 5. P2: worth trying

### D. Best-fit-decreasing bin packing at SFT

From Report §IV-A: SFT is trained at 131K context but examples are variable
length. Zyphra explicitly calls out that "packing strategy mattered" and
attributes **hallucination artifacts** to fixed-boundary truncation of packed
sequences. They use BFD (Ding et al. 2024) — fill each 131K window with whole
examples when possible, handle over-length examples in dataset-specific
preprocessing (via AP-trimming).

**Fit.** We do not yet have an SFT pipeline. When we build one, BFD packing
should be the default rather than a streaming truncator. Tagged for the
Sprint-N SFT infra work; not actionable now.

### E. Half-RoPE on GQA heads

Zyphra applies **RoPE to only 50 % of channels in each head**, leaving the
other half without position embeddings. This gives the model positional and
position-free capacity per head rather than partitioning by head (as we do
with HyPE: all conv heads are RoPE'd, all NoPEGQABlock heads are position-free).

**Candidate ablation.** In `models/components/attention.py::NoPECodaAttention`,
apply RoPE to only the first `head_dim // 2` dims of Q and K. Cheap change;
run the scorecard on a 200-step OdinFlat wikitext-103 config and compare to
full-NoPE baseline.

**Expected magnitude.** Unknown at our scale. Zyphra do not ablate half-RoPE
separately. Low confidence, low cost.

### F. Reasoning-aware pretraining

Zyphra cite Akter et al. 2025: including long-CoT data **from pretraining
onward** produces gains that post-training alone does not recover. Their
midtrain mix is **86 % long-CoT**, their SFT is **75 %**. At our scale with
dolma-10B + wikitext-103 this is a direction-of-travel question, not a
today-change. Depends on whether the target "what is Odin for?" is
reasoning-specialized or a general small-LM.

Deferred; worth revisiting when STATUS's Sprint 3 dolma-10B completes.

---

## 6. P3: capture the RL recipe for later

We have no RLVR infrastructure. When we add it — per the STATUS.md "biggest
gap" — this is the validated ZAYA1 cocktail, for direct lift into a design
doc.

### G.1 RL algorithmic spine

| Component | Value | Why |
|-----------|-------|-----|
| Trust region | **DPPO Binary-TV**, δ=0.1 | Replaces PPO ratio-clip with binary mask on tokens where divergence > δ; tune δ as largest value that still constrains reward-growth vs unconstrained baseline |
| Loss aggregation | **Dr-GRPO SMTSN** (sequence-mean over token-sum-norm) | Removes GRPO's implicit length normalization which biases toward longer responses |
| Advantage | **MaxRL**: `Â_i = (r_i − r̄) / r̄` | Divide by per-prompt **mean** not stddev; unbiased for truncated-MLE objective, stronger gradient on hard prompts |
| KL in reward | **None** | Trust region is DPPO alone; see §6.3 below for why |
| Optimizer | **Momentum-free Muon** on matrix weights, AdamW on embeds + LM head | Each RL update only depends on the current rollout batch; no cross-batch momentum averaging incompatible gradient directions |
| Async | PipelineRL, 2–5× rollout workers per trainer, weight sync every 2 iters, 2-update staleness bound | |
| Packing | Variable-length attention, fixed token budget (131,072) per GPU per microbatch, cross-rank pack rebalancing | |

Append these into [grpo_family_2026.md](grpo_family_2026.md) as the
ZAYA1-cascade reference stack.

### G.2 Reward gates (supplements to §3.B)

- **Streaming LZ77 repetition canary** with `τ_repeat = 0.05`: flagged rollouts
  have **task reward zeroed before advantage computation**, regardless of
  verifier outcome.
- **Rare-token fraction** (top-10 % IDs) logged per batch, not used as a gate.
- Purpose: prevent reward-hacking where a long degenerate trace ends with a
  (technically correct) answer the verifier accepts.

### G.3 Length-bias trap (Report §IV-F, NEGATIVE RESULT)

Combining **(signed K1-estimator log-ratio KL in reward) + (sequence-level
aggregation) + (broadcast back to all tokens) + (PipelineRL stale mixed-policy
rollouts)** produces a **length-dependent positive reward offset** unrelated to
task quality. Long completions accumulate more negative `l_t` terms which, when
subtracted from reward, become a bonus for being long.

**Mitigations** (Zyphra list but do **not** implement in production):

- Chunk-local signed-log-ratio isolation: aggregate per-chunk, not whole-sequence.
- Staleness rescaling: `g(Δ_c) = max(1, Δ_c)` divisor per chunk (from
  Bartoldson 2026 first-order EMA-reference approximation).

**ZAYA1 fix:** remove KL-in-reward entirely; DPPO Binary-TV handles trust
region. **Adopt the same default** when we add RL.

### G.4 Curriculum design

From §IV-B-4: RLVE-Gym with **Thompson sampling + IRT logistic calibration**
to hold each of 400 environments near a 0.5 solve rate (maximum Fisher
information under the logistic IRT model). ε-greedy around target with
allowed regressions, least-sampled-env weighting. This is a generic pattern
worth capturing independent of ZAYA1 — reusable for any future verifiable-
reward training.

### G.5 TTC harness: Markovian RSA

Formal spec, decisions, and why PaCoRe's hybrid variant beat vanilla on LCB-v6
are all in the source report (Report §VI). Useful if we ever train Odin to be
a reasoning model with TTC. Not actionable at current scope.

**Serving-profile property worth remembering:** with `(N, C, T, β, τ) =
(16, 4, 2, 40K, 4K)`, aggregation prefill is bounded by `|q| + C·τ + O(1)`
and per-candidate decode by `β`, **independent of total reasoning**. This is
a nice decoupling of "how hard the model thinks" from "how much context the
server must hold."

---

## 7. Explicitly NOT applicable

- **CCA / CCGQA** — we already use GQA + factorized embed/head. CCA gains are
  relative to full MHA, not GQA; at 32K+ contexts and MI300X scale. Our gfx1151
  attention path is different (HIP + flash-attn). No port.
- **MLP router + PID bias balancing + EDA** — MoE only; Odin is dense.
- **Top-1 MoE, no residual expert** — MoE only.
- **Router replay** — MoE-RL combination we do not have.
- **BF16 + matched FP32 op set on engine + trainer** — our stack is fp16 + fp32
  selective (because bf16 is broken on gfx1151, per STATUS's `fp16 Stability`
  work). This is the opposite direction. The engine/trainer-agreement *idea*
  applies if we add a rollout engine; the specific precision configuration does
  not.
- **AMD MI300X + Pollara 400 platform validation** — different hardware SKU
  (gfx942 vs our gfx1151). Their cluster/networking characterization is
  interesting reading but does not transfer directly. See arXiv:2511.17127 for
  the companion systems paper.

---

## 8. Open questions this raises for us

1. **Should OdinHalo get learned residual scaling?** (High-confidence yes; needs
   200-step A/B to confirm at fp16.)
2. **Should we include even a small fraction of long-CoT data in the dolma-10B
   pretraining run?** Akter 2025 + ZAYA1 both argue pretraining-time inclusion
   is irrecoverable later. Low-cost experiment: 5–10 % CoT in the next
   continued-pretraining phase, AP-trimmed at our block size.
3. **When we build SFT, do we use BFD packing from day 1?** (Yes — cheaper
   than later retrofit.)
4. **Half-RoPE vs full-NoPE on `NoPECodaAttention`**: worth one scorecard.
5. **Does the Markovian-RSA shape of "batched candidates with bounded
   carry-forward" have any utility at our scale for evaluation-time
   self-consistency?** Orthogonal to training; could be a post-hoc harness.

---

## 9. Companion papers worth reading alongside

- **Report itself**: [../../docs/research/zaya1_8b_technical_report/zaya1_8b_technical_report.md](../../docs/research/zaya1_8b_technical_report/zaya1_8b_technical_report.md)
- **arXiv:2511.17127** (ZAYA1-base systems paper) — AMD MI300X + Pollara
  cluster characterization; mostly of HW-interest.
- **arXiv:2510.04476** (CCA / CCGQA) — compressed-latent attention; architectural
  background for CCGQA. Not a port candidate for us.
- **arXiv:2510.06557** (Markovian Thinker, Aghajohari et al.) — the
  bounded-workspace recurrence idea that Markovian RSA extends.
- **arXiv:2026-01 Akter et al.** (cited in Report §III-A) — reasoning-aware
  pretraining finding. The evidence base for why long-CoT belongs in
  pretraining, not just post-training.

## 10. What this does NOT include

- **SubQ / SSA (Subquadratic, May 2026)** — no technical paper published yet,
  only marketing posts. Architectural claims (content-dependent selection,
  linear-in-context attention, 52× B200 prefill at 1M) are interesting but
  not yet independently verifiable or implementable. See
  [../../docs/research/subq_ssa_watchlist_2026.md](../../docs/research/subq_ssa_watchlist_2026.md)
  for the watch-list note; revisit when a paper drops.

## See also

- [ap_trimming_recipe.md](ap_trimming_recipe.md) — P0 implementation recipe
- [grpo_family_2026.md](grpo_family_2026.md) — RLVR family; ZAYA1 cascade is
  captured there too
- [fp16_stability_gfx1151.md](fp16_stability_gfx1151.md) — our stability stack;
  LZ77 + rare-token canaries extend it
- [imu1_recipe_2026.md](imu1_recipe_2026.md) — current pretraining recipe
- [../../docs/research/broad-research-synthesis-2026-05-06.md](../../docs/research/broad-research-synthesis-2026-05-06.md) — 2025-2026 literature synthesis
- [../../docs/research/zaya1_8b_technical_report/zaya1_8b_technical_report.md](../../docs/research/zaya1_8b_technical_report/zaya1_8b_technical_report.md) — full source report
