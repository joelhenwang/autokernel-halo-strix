# Deep Literature Review: 12 May-2026 Papers

**Date:** 2026-05-13
**Method:** 4 parallel explore agents reading the full saved arxiv HTML for each paper (except δ-mem which has no HTML — source code at `declare-lab/delta-Mem` was used instead).
**Length of each group report:** 8,000–15,000 words each with page-level citations.
**Confidence:** HIGH for technical claims (verified against source), MEDIUM for cross-paper synthesis.

This replaces the earlier shallow summary (commit 1bb81d5) which was based on abstracts and partial webfetch output. That earlier summary was directionally correct but missed specific numbers, shaky claims, and cross-paper tensions. This document is the authoritative record.

---

## Tier A — Highest Relevance to OdinPhantomLoop / Production

### A1. LoopUS (arxiv 2605.11011) — Park, Lee, Kim, Bae (Pusan / Changwon Nat'l Univ)

**What it does:** Converts pretrained LLM (Qwen3-1.7B/4B/8B, Phi-4-14B, TinyLlama) into a looped latent-refinement model via block decomposition + selective gate + random deep supervision + confidence head.

**Core equations** (precise):

```
Block decomposition:  E (encoder, first 0–5 layers) | M (reasoning, middle) | D (decoder, last 1 layer)
Forward:              h^(0) = E(x);   h^(b+1) = G(M, h^(b));   ℓ^(B) = D(h^(B))

Selective gate:
  δ^(b) = M(h^(b)) − h^(b)                                    (residual proposal)
  Δ^(b) = softplus(W_Δ · δ^(b) + b_Δ)                        (positive step size)
  α^(b) = exp(Δ^(b) ⊙ A),   A ∈ R^d_{<0}                     (learned, elementwise decay)
  h^(b+1) = α^(b) ⊙ M(h^(b)) + (1 − α^(b)) ⊙ h^(b)          (damped mix)

Equivalent as forward-Euler:
  h^(b+1) = h^(b) + P^(b) · (M(h^(b)) − h^(b)),   P^(b) = diag(α^(b))

Confidence head:       q̃^(b) = q_φ(h^(b)),   q^(b) = σ(q̃^(b));   stop when q^(b) ≥ q_th
```

**Loss** (per sampled depth b ∈ S, |S|=K):
```
L^(b) = L_LM^(b) + L_mono^(b) + L_Q^(b)
L_mono^(b) = SiLU(L_LM^(b) − L_LM^(b−1))       # monotonicity; SiLU chosen after ablation
L_Q^(b) = BCE(q̃^(b+1), fraction of tokens correctly predicted at step b)
```

**Training recipe** (all from Appendix A):

| Item | Value |
|---|---|
| Data | FineWeb-Edu CC-MAIN-2025-26, streaming |
| Tokens | 3 B (1 epoch) |
| Context | 1024 |
| Batch | 64 (per-device) |
| Optimizer | AdamW, LR **5e-5**, cosine, 300 warmup, bf16 + FlashAttn-2 |
| B (max loop steps) | 20 |
| K (supervised per batch) | 5 |
| Stop threshold (train) | 0.55 |
| Stop threshold (infer) | 0.6, B_max=8 |
| Hardware | L40S / RTX PRO 6000 / H200 depending on backbone |

**Key results** (Table 1):

| Backbone | WikiText ppl w/o→w | AVG acc 7 tasks w/o→w | Δ |
|---|---|---|---|
| Qwen3-1.7B | 21 → 16.9 | 53.7 → 55.3 | +1.6 |
| Qwen3-4B | 16.4 → 13.9 | 60.3 → 62.1 | +1.8 |
| Qwen3-8B | 12.2 → 10.3 | 63.2 → 65.4 | +2.2 |
| Phi-4-14B | 9.59 → 7.75 | 67.0 → 68.6 | +1.6 |

Gains concentrate on ARC-C, OBQA, PIQA, WG. MMLU and HellaSwag mostly flat. **+3.0% avg zero-shot** across backbones, **17.4% WikiText ppl reduction**, **21.3% LAMBADA ppl reduction**.

Inference: at q_th=0.6, Qwen3-4B halts at **3.39 iterations average** (out of B_max=8). Stable at unseen depths up to 100 iterations.

**Ablations** (Figure 9, on L_LM):
- (a) Remove selective gate → higher L_LM (drift-controlled refinement lost)
- (b) Remove E/D decomposition (loop all layers) → higher L_LM
- (c) Remove random deep supervision → similar L_LM but slower convergence
- (d) Sigmoid gate instead of decay gate → higher L_LM
- (e) SiLU > SoftPlus > ReLU ≈ SELU for monotonicity activation
- (f) TBPTT worse than random deep supervision

**Novelty vs prior:**
- vs McLeish et al. 2025 (retrofitted recurrence): LoopUS +6.3 AVG on TinyLlama with **17-20× fewer training tokens** (3B vs 52B)
- vs Bae et al. 2025a (relaxed recursive transformers w/ LoRA): +6.3 vs +3.5 AVG with 20× fewer tokens
- The random deep supervision idea traces back to Hierarchical Reasoning Model (Wang et al. 2024); LoopUS's contribution is the end-to-end combination with selective gate

**Shaky points:**
- Table 2 prior-method comparisons are copied from those papers, not re-run (authors flag this)
- No peer review (May 2026 preprint)
- Block decomposition is heuristic ("where the cosine-similarity profile changes most abruptly"); reproducer must do the analysis per backbone
- Assumes pretrained backbone has staged-layer geometry; untested on MoE/SSM/hybrid

**Code:** `thrillcrazyer.github.io/LoopUS` (referenced, not inspected).

---

### A2. MELT (arxiv 2605.07721) — Qualcomm AI Research

**What it does:** Memory-efficient retrofit of looped transformers (Ouro-1.4B-Thinking → MELT-1.6B). Replaces per-loop KV cache (which grows linearly with reasoning depth T) with a single per-layer KV cache shared across loops and updated via GRU-style gating.

**Core equations** (precise):

```
Baseline LoopLM:      F^(T)(·) = lmhead ∘ (M ∘ … ∘ M) [T times] ∘ emb(·)
                      KV per layer: O(L·T)                                   [grows with depth]

MELT gated update (per token t, per layer l):
  z_t^(l) = σ(x_t^(l) W_z + h_{t-1}^(l) U_z + b_z)                    (Eq. 1)
  h_t^(l) = z_t^(l) ⊙ h_{t-1}^(l) + (1 − z_t^(l)) ⊙ x_t^(l)
  k_t^(l) = h_t^(l) W_K,  v_t^(l) = h_t^(l) W_V
  K_t^(l) = [K^(l), k_t^(l)],  V_t^(l) = [V^(l), v_t^(l)]
  x_attn^(l) = Attn(q^(l), K_t^(l), V_t^(l)) + x_t^(l)
  x_{t}^(l+1) = FFN(x_attn^(l)) + x_attn^(l)

Memory per layer: O(L)                                                        [constant in T]
```

**Critical subtlety:** `t` in Eq. 1 is the **token index along the sequence**, NOT the loop index. The gate drives a recurrence along tokens for a single layer. Per-loop KV "growth" that Ouro suffers is replaced by in-place update *across loops*; the new latent at token `t` already integrates information from all preceding loops because the previous token's latent feeds Eq. 1.

**Training recipe** (Table 6):

| Item | Value |
|---|---|
| Base | Ouro-1.4B-Thinking |
| Layers / d | 24 / 2048 |
| Loops T | 4 |
| Gate params | ≈ 0.2 B |
| Chunk size | 500 tokens |
| Sequence length | 10,000 |
| Batch | 320,000 tokens |
| LR base | **8e-6** |
| LR gate | **5e-4** (63× base) |
| Phase 1 (interpolation α: 0→1) | 500 steps, 160 M tokens |
| Phase 2 (attn-aligned distillation, β=0.1) | 300 steps, 96 M tokens |
| Compute | 1,040 H100-GPU-h main / 20,000 H100-GPU-h project total |

**Three required training tricks** (ablation Table 4 shows each is essential):
1. **Chunk-wise training**: parallel within 500-token chunks, sequential across chunks. Removing it → 0% on every benchmark (total failure)
2. **Interpolated transition**: `KV = α·KV_MELT + (1−α)·KV_base`, α: 0→1 over 500 steps. Removing it → 10-point drop
3. **Attention-aligned distillation**:
   ```
   L = L_KD + β · (1/(N·T)) · Σ_{l,t} ‖o^(l,t)_MELT − sg(o^(l,t)_LoopLM)‖²
   ```
   Removing it → 2-3 point drop

**Key results** (Table 1, 6 math benchmarks, avg pass@1):

| Model | Avg pass@1 |
|---|---|
| Qwen3-1.7B | 43.1 |
| MELT-1.6B | 59.9 |
| Ouro-1.4B-Thinking (parent) | 62.3 |

MELT beats similar-sized non-looped by ~3 points, 2.4 points behind its Ouro parent.

**Memory** (Table 2, 32K-token gen, vLLM):

| Model | KV/token | Total 32K mem | vs Ouro |
|---|---|---|---|
| MELT-1.6B | 0.197 MB | 9.49 GB | **−3× KV, −3× total** |
| Ouro-1.4B-Thinking | 0.786 MB | 27.97 GB | — |
| Qwen3-1.7B (MQA) | 0.115 MB | 7.07 GB | better (MQA) |

**Gate variant ablation** (Phase-1 only, Table 3):
- Element-wise gate (MELT): 77.7 AMC23 pass@1
- Scalar single-gate: 66.9
- Mean: 68.8
- EMA-0.2: 68.6
- Last-loop only: 69.7

Element-wise state-dependent gating dominates by 8-10 points.

**Shaky points:**
- Proposition E.1 ("gradient superhighway") only analyzes the token axis, not the loop axis. At gate saturation z→1, Jacobian J_t→I, which in Error Control's taxonomy is exactly the degenerate case where contraction vanishes. **No theoretical analysis of loop-axis error dynamics.**
- Appendix F acknowledges: could NOT reproduce Ouro's reported numbers from public artifacts; Ouro's "early-exit gate" is actually disabled by default and doesn't reduce compute even when triggered.
- No MQA variant; remaining memory gap to Qwen3-1.7B (~2.5 GB at 32K) is acknowledged as future work.
- Chunk-wise training is a workaround not a solution; scaling past 1.6B untested.
- "First architecture to exceed standard models at same memory footprint" — narrow claim, true only vs chosen baselines.

**Code:** "Will be released soon" — not released at paper submission.

---

### A3. Error Control Dynamics (arxiv 2605.07755) — Chung, Choi, Kim (Yonsei)

**What it does:** Formal theory of when recurrent architectures can maintain stable state tracking. Proves that affine recurrences (SSMs, Linear Attention) **cannot correct errors along state-separating subspaces** once they preserve state representations exactly.

**Taxonomy** (Def. 1, Table 1). All recurrent layers fit:
```
h_t = φ(g(h_{t-1}, x_t) ⊙ (A(x_t) · h_{t-1}) + b(x_t))
y_t = dec(h_t, x_t)
```

- **Affine class** (state Jacobian ∂h_t/∂h_{t-1} doesn't depend on h_{t-1}): S4, Mamba, Mamba-3, AUSSM, Simple AUSSM, Negative Mamba, Linear RNN, Token-gated RNN
- **State-dependent class** (Jacobian depends on h_{t-1}): tanh RNN, State-gated RNN, LSTM, GRU

**Theorem 1 (Affine neutrality on symbolic subspace):** Let G be finite symbolic states with per-state representations c_g ∈ F^d. The symbolic subspace is:
```
U = span{c_g − c_{g'} : g, g' ∈ G}
```
For any state-preserving sequence s with affine return map F_s(h) = A_s · h + b_s, if F_s(c_g) = c_g ∀g ∈ G with non-degenerate representations:
```
A_s|_U = I   (identity on symbolic subspace)
⇒ F_s(c_g + δ) − F_s(c_g) = δ  for all δ ∈ U
```
**Errors along state-separating directions are preserved. Exact tracking and error correction are incompatible for affine recurrences.**

**Corollary 1 (Finite-horizon):** crossing time when distinguishability ratio q(t)=R(t)/M(t) passes threshold τ:
```
T_cross ≈ τ · M / ‖W_out · η̄‖
```
where η̄ is mean projected residual drift, M is between-class separation.

**Per-operator Jacobian analysis (Appendix E.1):**
- **Pointwise activations (tanh, ReLU):** Jacobian = diag(φ'(p)) · W_h. State-dependent → **can contract U**. ✓
- **Pair operators (max, min, GroupSort):** state-dependent permutation, **can contract**. ✓
- **Whole-vector normalizations (LayerNorm, sphere projection):** rank-1 state-dep + isotropic 1/σ scaling. **Cannot discriminate between symbolic directions.** ✗

This is why Table 3 shows LayerNorm and sphere-projection RNN variants FAIL S3 state tracking.

**Experimental results** (Table 2, max-passing length where accuracy ≥90%, trained to length 60):

| Model | Dynamics | C2 L1/L2 | C6 L1/L2 | S3 L1/L2 |
|---|---|---|---|---|
| Mamba | Affine | ✗/60 | ✗/60 | ✗/✗ |
| Mamba-3 | Affine | 200/300 | 100/100 | ✗/60 |
| AUSSM | Affine | 1000/✗ | 200/100 | ✗/✗ |
| Negative Mamba | Affine | 1000/1000 | 100/200 | 100/200 |
| Token-gated RNN | Affine | 1000/700 | 300/400 | 500/1000 |
| **tanh RNN** | State-dep | **1000/1000** | **1000/1000** | **1000/1000** |
| **State-gated RNN** | State-dep | **1000/1000** | **1000/1000** | **1000/1000** |

State-dependent models cap the eval range; affine models collapse at or just above training length (60) on nontrivial groups.

**T_cross correlates with empirical max-passing length at Pearson r = 0.87 on log-log (Fig 4).**

**Critical caveat:** Appendix E.2 shows parity (C2) is solvable by affine **neutral oscillation** — `F_a(h) = −h + (c_0 + c_1)` satisfies `F_a² = id` even without error correction. **Don't benchmark loop stability with parity or ≤2-class tasks.** Use ≥3 classes with sequences ≥10× unroll depth.

**Training recipe** (~2,000 GPU-h total, A6000/3090 — easiest of all 12 papers to reproduce):

| Item | Value |
|---|---|
| d_model | 698 (parameter-budget matched) |
| Depth | {1, 2} |
| Optimizer | AdamW, wd=0.01, batch 256 |
| Curriculum | starts T=2, doubles at ≥95% for 5 consecutive epochs, cap L_max=60 |
| Grid | d_state ∈ {32,64,128}, lr ∈ {1e-4, 5e-4, 1e-3}, 3 schedulers, 3 seeds (81 cells/group) |
| Eval | lengths {100,...,1000} |

**Code:** Anonymized supplementary material; public release on acceptance.

**Novelty:** Distinct from Merrill et al. 2024 "Illusion of state" (which argues via circuit complexity: TC⁰ SSMs can't solve NC¹-hard tasks). Error Control shows that **even when expressivity is sufficient**, affine models still fail due to missing error-correction dynamics.

---

## Tier B — Useful for Future Infrastructure

### B1. SlimSpec (arxiv 2605.10453) — Plaksin et al. (Nebius)

**What it does:** Replace drafter LM-head `W_full ∈ R^{V×d}` with low-rank factorization `W_up · W_down` (rank r). Preserves full vocabulary.

**Equation:**
```
Standard:  z = W_full · h,        W_full ∈ R^{V×d},   cost O(V·d)
SlimSpec:  z = W_up · W_down · h, W_down ∈ R^{r×d}, W_up ∈ R^{V×r},
                                   cost O(r·d + V·r) ≈ O(V·r) when V ≫ d
```

No bias, no non-linearity between W_down and W_up. Mathematically equivalent to forcing W_full to have effective rank ≤ r.

**Acceptance–cost framework** (Eq. 4):
```
ρ_TPS(ν, ρ_τ; κ) = ρ_τ · (1 + κ) / (1 + ν·κ)

where:
  ρ_τ = τ_M / τ_Full       (acceptance preservation)
  ν = T_head^M / T_head^Full (head-latency factor)
  κ = T_head^Full / T_{non-head} (head-dominance ratio)
```

Break-even condition: `ρ_τ > (1+νκ)/(1+κ)`.

**κ empirical values:**
- Llama-3.1-8B, T=0, B=1: κ ≈ 0.25
- Qwen3-30B-A3B: much smaller (bigger verify cost)

**Results** (Table 2, speedup vs no-speculator baseline, T=0, avg over 3 benchmarks):

| Target | Method | B=1 avg | B=64 avg |
|---|---|---|---|
| Llama-3.1-8B | Full Vocab baseline | 2.48 | 1.40 |
| Llama-3.1-8B | VocabTrim-T V_tr=64K | 2.70 | 1.41 |
| Llama-3.1-8B | SpecVocab r=d/8 | 2.86 | 1.46 |
| Llama-3.1-8B | **SlimSpec r=d/8** | **2.94** | **1.52** |
| Qwen3-30B-A3B | Full Vocab | 1.91 | 1.34 |
| Qwen3-30B-A3B | SlimSpec r=d/8 | 2.05 | 1.40 |

**Rank sweet spot** (Table 3): r=d/8 wins. r=d/16 gains ν but loses acceptance.

| Config | ρ_τ | ν | Speedup |
|---|---|---|---|
| VocabTrim-T V_tr=64K | 1.00 | 0.58 | 1.09 |
| SpecVocab r=d/8 | 1.01 | 0.46 | 1.16 |
| **SlimSpec r=d/8** | **0.99** | **0.21** | **1.19** |
| SlimSpec r=d/16 | 0.98 | 0.14 | 1.18 |

**Novelty:** Low-rank LM-head factorization is NOT new (Adaptive Softmax, ALBERT, GroupReduce). The contribution is (a) applying it to drafter head in speculative decoding, (b) the acceptance-cost framework.

**Training recipe** (Appendix A): EAGLE-3 drafter, forward KL loss, AdamW, LR 4e-4, no weight decay, cosine, 10 epochs, batch 64, grad clip 0.5, 128 GPU-hrs on H200.

**Shaky points:**
- "8-9% improvement over baselines" only on Llama-3.1-8B and GPT-OSS-20B; on Qwen3-30B gap shrinks to 1-2%
- Not tested on >70B targets (framework predicts tiny gains — verify dominates)
- No automatic rank selection
- Applicable ONLY to drafter, not primary LM-head

**Relevance to us:** We already use FactorizedLMHead with rank=256 (r=d/3 for d=768). SlimSpec's sweet spot of r=d/8 would be rank=96 for us. Could test rank reduction for primary LM-head (different context from their drafter result but same mathematical structure).

### B2. UniPrefill (arxiv 2605.06221) — CASIA + WeChat

**What it does:** Token-level prefill acceleration. Scores token importance at full-attention layers via block-wise top-p selection, then drops tokens from ALL subsequent layers (FFN, linear attn, SWA). Works on hybrid architectures.

**Algorithm:**
1. At each full attention layer: compute partial attention `S = Q[N-n:N] K^T ∈ R^{n×N}` (only last n=128 query positions)
2. Online softmax → per-token scores → block-reduce (G=64) → s̄_g^(b)
3. Top-p selection (p=0.99 default): retain smallest block set with cumulative score ≥ p
4. Sparsity propagates through ALL subsequent sublayers in the block

**Complexity:** Saves `ΔFLOPs = (1−ρ)·(L−ℓ₁)·O(N·d²)` per drop — linear in remaining layers, not just attention.

**Key results** (Table 1, RULER benchmark, 128K context TTFT speedup):

| Architecture | Avg RULER (baseline → UniPrefill) | 128K speedup |
|---|---|---|
| Llama-3.1-8B (full attn) | 90.36 → 90.45 | **2.26×** |
| Qwen3-Next-80B-A3B (linear/full 3:1) | 94.76 → 93.94 | 1.68× |
| Gemma-3-12B (SWA/full 5:1) | 79.99 → 78.87 | 1.49× |

Throughput (Table 2): up to +109% on Llama at BSZ=16, +68% on Qwen3-Next, +42% on Gemma. Speedup scales with context length AND batch size.

**Shaky points:**
- Error bound (Eq. 12) only asymptotic, no empirical Lipschitz measurement
- Table 5 "seed variance" shows identical numbers across 3 seeds — either deterministic algorithm or single-run dressed as multi-seed
- Short context shows regression (−5% at 4K on Qwen3-Next-80B-A3B)
- Inference-only, no training support
- TileLang/Triton kernels are Nvidia-focused; ROCm port non-trivial

**Relevance to us:** Not applicable now (T=512 too short). Worth revisiting IF/WHEN we deploy at context ≥32K. Direct cost to port: ~2 engineer-weeks for HIP/Triton kernels.

### B3. MISA (arxiv 2605.07363) — PKU MuLab

**What it does:** Routes DSA indexer heads via MoE-style selection. Reduces per-query cost from O(H^I · L) to O(hL + H^I · M).

**Core equations:**
```
Block pooling:     k̃^I_b = Pool({k^I_s : s ∈ B_b})
Router affinity:   A_{t,j,b} = w^I_{t,j} · ReLU(q^I_{t,j} · k̃^I_b)
Head importance:   E_{t,j} = (1/M) Σ_b |A_{t,j,b}|
Active heads:      H_t = TopK_j(E_{t,j}, h)           # h=8 ≪ H^I=64

Scoring:           Î_{t,s} = Σ_{j∈H_t} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
Top-k tokens:      T_t = TopK_s(Î_{t,:}, k=2048)
```

**Results** (Table 1, LongBench):
- DeepSeek-V3.2 (H^I=64): DSA=51.05 vs MISA=50.85 (8× fewer heads)
- GLM-5 (H^I=32): DSA=46.01 vs MISA=46.43 (4× fewer heads, improvement)

Kernel: 3.82× speedup over DSA's indexer kernel on H200 (TileLang).

**Relevance to us:** Only applicable to DSA-style architectures. OdinFlat uses GQA, not DSA. Zero applicability unless we retrofit DSA entirely (weeks of work for zero current gain).

### B4. δ-mem (arxiv 2605.12357) — Declare-Lab / MindLab

**What it does:** Lightweight memory adapter. Augments frozen full-attention backbone (Qwen3/SmolLM3) with 8×8 online state matrix updated by delta rule. State readout generates low-rank corrections to Q/K/V/O projections.

**State update** (from source code `deltamem/core/delta_impl.py`):
```
keep_t, erase_t, write_t ← per-token gates
S_t = keep_t · S_{t-1} − erase_t · (S_{t-1} k_t) ⊗ k_t + write_t · v_t ⊗ k_t
r_t = S · q^M_t                                       (readout)
```

Corrections applied with α/r = 2 scaling (LoRA-style):
```
q_corrected = W_q h + (α/r) U_q · r_t
```

**Training:** SFT on QASPER-episode data. Backbone frozen. Rank r=8, α=16. Auxiliary losses (contrast, KL, causal, anchor, recovery, sparsity). 1 epoch, LR 2e-4, 8-GPU DeepSpeed ZeRO-2.

**Results** (abstract): 1.10× avg, **1.31× on MemoryAgentBench**, 1.20× on LoCoMo.

**Relevance to us:** Not architectural — memory adapter for agent/assistant use case. Full attention preserved. Not applicable to LM pretraining. Revisit only if we build agent/assistant with persistent state.

---

## Tier C — Methodological Lessons (Limited Direct Transfer)

### C1. TextLDM (arxiv 2605.07748) — Jiang et al.

**What it does:** Transfers visual DiT recipe (VAE + DiT + flow-matching + CFG) to language modeling with minimal changes.

**Key novelty — REPA for text VAE:** cosine-similarity alignment between VAE encoder and frozen Qwen3-1.7B:
```
L_REPA = -(1/N) Σ_i cos(h^enc_i, sg(h^LLM_i))
```

**Critical ablation (Table 3):** All VAE configs reach ≥99.6% reconstruction but generation quality varies substantially. REPA improves the **geometry** of the latent manifold, not round-trip fidelity.

**Results:** 768M TextLDM matches/beats reproduced GPT-2-medium (459M) on ROUGE+MAUVE across WikiSource/Wikipedia/TinyStories; lags on One Billion Words (length-sampling bias).

**Transferable insight for our AR project:** Add a REPA-like alignment loss during training. Freeze a larger LLM (Qwen3-1.7B), compute cosine similarity between an intermediate layer of our model and the 3rd-to-last layer of the frozen LLM. ~100 lines, 1-day experiment. Could improve OOD generalization (where TextLDM's biggest REPA gains appeared).

**Shaky:** Only 2 diffusion baselines (SSD-LM, Block Diffusion). Excluded PLAID/COSMOS/CALM/LLaDA for semi-reasonable reasons. BERTScore gap is real and acknowledged in main text but glossed over in intro.

### C2. ThinC (arxiv 2605.07237) — Korea U

**What it does:** Training framework teaching LLM to "think in code" — single NL planning step then all reasoning via code blocks. 12,200 distilled trajectories + SFT + GRPO with DAPO modifications.

**Training recipe:**
- SFT: LR 7e-6, cosine, batch 16, 3 epochs, 8×H200
- RL: DAPO-Math-17k, 3-stage curriculum (280+120+rest steps), G=8 rollouts/prompt
- DAPO modifications: token-level normalization across group, asymmetric clipping (ε_low=0.20, ε_high=0.28), NO KL penalty

**Results** (5 benchmarks, avg@16 under 32K):

| Model | Average |
|---|---|
| ThinC-1.7B | 42.8% (vs base 32.2%, +10.6) |
| **ThinC-4B** | **78.1%** |
| Qwen3-235B-A22B-Thinking | 75.2% |
| ASTER-4B (matched base + similar RL) | 74.0% |

**Most surprising finding:** Recovery@k robustness — SFT data contains NO code errors but model recovers from 5 consecutive code failures at 33% rate, 2× any interleaved baseline. Format itself confers robustness.

**Transferable insight:** DAPO recipe is the best-documented open RL-with-verifiable-rewards configuration. Apply if/when we move to RL post-training.

**Shaky:** Post-SFT scores below base models (SFT teaches format, not accuracy; RL does all lifting). Anyone with smaller RL compute budget won't reproduce 4B > 235B result.

### C3. RigidFormer (arxiv 2605.09196) — MIT + Meta

**What it does:** Object-centric Transformer for rigid-body physics. One token per object, 4 anchors per object, Verlet integration + differentiable Kabsch SVD alignment for rigidity-by-construction.

**Key novelty — ARoPE (Anchor-based RoPE):**
```
ψ_ω(x_k) = ⊕_{j=1..3} [{ω_l · x_{k,j}}_{l=1..16}; {ω_l · x_{k,j}}_{l=1..16}]   (96-D)
ARoPE(A_i) = (1/N_a) Σ_k ψ_ω(x_k)                                           (mean pool)
```

Mean-pooling over anchors gives invariance to anchor reindexing; world-frame positions preserve shape/centroid. Ablation Table 7: PCA-based PE → 1.548 m RMSE (catastrophic). ARoPE → 0.161 (best).

**Transferable insight:** Methodological. The honest symmetry scoping (Appendix B explicitly states model is NOT SE(3)-equivariant because gravity is in world frame) is a template for how to claim architectural properties without overstating.

**Direct relevance to us:** Low. Rigid-body physics ≠ language modeling. ARoPE for 3D positions doesn't transfer to 1D sequence positions.

### C4. SlimQwen (arxiv 2605.08738) — Alibaba Qwen + MBZUAI

**What it does:** Compression recipe for Qwen3-Next-80A3B → 23A2B (4× params, 1.5× active) via depth+width+expert pruning and 400B-token continual pretraining.

**Key novelties:**
1. **Partial-preservation expert merging**: keep top ⌊Ñ/2⌋ untouched, merge remaining into next ⌊Ñ/2⌋ based on cosine similarity + importance scores
2. **MTP knowledge distillation (Eq. 11):**
   ```
   L_MTP-KD = -(1/D) Σ_k (1/(T-k)) Σ_i Σ_v q_{i+k}[v] · log p^k_{i+k}[v]
   L_total = (1-λ)L_LM + λL_KD + β·[(1-λ)L_MTP-LM + λL_MTP-KD]
   ```
   with λ: 1→0.75 linear, β: 0.3→0.1 cosine

**Key findings:**
- Pruned init beats scratch: +11.79 AVG with same 120B budget
- All one-shot expert compression methods (frequency/soft-logits/REAP/merging) converge to similar final performance after 400B tokens — **no single method dominates**
- Partial-preservation gives ~0.5-1 point consistent improvement
- LM loss + KD beats pure KD on knowledge benchmarks
- MTP-KD improves speculative decoding acc_k (longer acceptance > shorter)
- Progressive pruning beats one-shot (depth-first best at 40B+360B split)
- Compute: ~400B tokens on unspecified (proprietary Alibaba) cluster

**Transferable insight for our project:** MTP-KD as auxiliary objective. Structure:
```python
# At each MTP depth k ∈ {1, 2}
h^k = TRM_k([RMSNorm(h^{k-1}); RMSNorm(Emb(t_{i+k}))] · M_k)
p^k_{i+k} = OutHead(h^k)
# Train against teacher's MTP distribution
```

Even WITHOUT a teacher, MTP-LM alone improves backbone quality. If we add a teacher (e.g., distill larger model → ours) later, MTP-KD is the recipe.

**Shaky:** "All one-shot methods converge" based on ~1-point differences (within noise, no confidence intervals). Data corpus not disclosed. No code release.

---

## Cross-Cutting Theoretical Insight

**The most important finding from this deep study:** there's a tension between MELT's gate and Error Control's theorem that neither paper discusses.

1. **MELT's gate is state-dependent** in Error Control's taxonomy (z reads h through U_z), so Theorem 1 doesn't directly indict it.
2. **BUT** MELT's Proposition E.1 argues gradient preservation under gate saturation (z→1), at which point J_t→I — which is Error Control's exact degenerate case where contraction vanishes.
3. So MELT retains error-correction capability only when the gate is NOT saturated. Multi-epoch training could drive saturation; MELT doesn't analyze loop-axis dynamics.

**LoopUS's gate is safer:** α = exp(Δ ⊙ A) with Δ > 0 and A < 0 gives α ∈ (0,1) **strictly** — cannot saturate to 1. This is the theoretically cleanest gate formulation of the three relevant papers.

**Shared pattern:** All three Tier-A papers agree that gates matter AND they must be state-dependent. Scalar/global/isotropic aggregation fails empirically (MELT Table 3) and theoretically (Error Control Jacobian analysis).

---

## Revised Recommendations for OdinPhantomLoop

Based on deep study (not previous surface-level summary):

### R1 (highest priority) — Upgrade OdinPhantomLoop pos_bias to LoopUS-style selective gate

Current design: `h = core_i(h, freqs_cis) + pos_bias[i]` — pure additive bias at each position. In Error Control's taxonomy this is an **affine** operator at the iteration boundary (since pos_bias doesn't enter the state-Jacobian).

Proposed upgrade at iteration boundaries (positions 4, 8):
```python
delta = h_after_block - h_iter_start
Delta = F.softplus(self.W_Delta(delta) + self.b_Delta)      # [d_core]
alpha = torch.exp(Delta * self.A)                            # A learned, sign-clamped <0
h_iter_end = alpha * h_after_block + (1 - alpha) * h_iter_start
```

**Why this is the right choice:**
- Strictly bounded α ∈ (0,1) (cannot saturate to 1 — the failure mode MELT doesn't discuss)
- Per-token, per-channel adaptive damping (not scalar — MELT ablation shows this matters by 10+ points)
- Residual-scaled (Δ = softplus(W · (M(h)−h)) — data-dependent step size)
- Pointwise — zero graph breaks under torch.compile
- Tiny overhead: +1.2M params for 2 boundaries × (768² + 768) ≈ 1.18M

**Expected effect:** Better multi-epoch stability (fewer NaN dumps), small quality lift. Bandwidth cost is O(L·d) vs block cost O(L·d²) — below noise floor on gfx1151.

**Verification before shipping:** compare smoke loss at 500 steps vs baseline OdinPhantomLoop on same seed.

### R2 — Add monotonicity loss at iteration boundaries

LoopUS Figure 9(e) ablation: SiLU > SoftPlus > ReLU > SELU for `L_mono = SiLU(L_LM^b − L_LM^{b-1})`.

In OdinPhantomLoop, add auxiliary loss at positions 4, 8, 11:
```python
L_mono = F.silu(ce_at(p=8) - ce_at(p=4)) + F.silu(ce_at(p=11) - ce_at(p=8))
```
Requires computing LMHead at intermediate positions (2 extra LMHead calls per step).

**Cost:** ~2-3% tok/s degradation.
**Benefit:** Monotonic per-iteration refinement (LoopUS Fig 6 shows this is the default behavior, monotonicity loss enforces it).

### R3 — Add REPA-style alignment loss (TextLDM-inspired)

During training, compute cosine similarity between our model's intermediate hidden state and frozen Qwen3-1.7B's 3rd-to-last layer:
```python
L_REPA = -F.cosine_similarity(h_ours_layer_k, sg(h_qwen_layer_N-3), dim=-1).mean()
L_total = L_LM + λ * L_REPA      # λ = 1 per TextLDM
```

Runs Qwen3-1.7B forward once per batch. ~500ms/step extra on current hardware (could be prefetched on separate GPU or cached if tokens repeat).

**Expected effect:** Improved OOD generalization (TextLDM's biggest REPA gains were on Wikipedia/WikiSource OOD).

**Risk:** Layer choice matters (TextLDM Table 2d: last layer worse than 3rd-to-last). Would need ablation.

### R4 — Don't add random deep supervision (breaks compile)

LoopUS uses K=5 random gradients out of B=20 total iterations, running the rest under torch.no_grad(). This introduces runtime grad/no-grad branching that breaks torch.compile's flat-graph assumption. Our OdinPhantomLoop is deliberately flat-unrolled with 3 iterations × 4 layers = 12 positions, all in autograd. Memory is not our constraint (bandwidth is). Skip.

### R5 — If we ever warm-start from OdinHalo checkpoint, use MELT's split LR

Phase 1 of MELT uses 63× higher LR on new gate params vs preserved base weights (5e-4 vs 8e-6). Our `halo_training/optimizer.py` already supports parameter groups — pass two groups on resume.

### R6 — Diagnose loops with Error Control's distinguishability ratio

Add to `scripts/eval_checkpoint.py` scorecard:
```python
# At each of the 12 positions p in phantom loop:
#   c_g(p) = per-class centroid of hidden state (class = ground-truth next token)
#   R(p) = within-class spread (E ‖h − c_g‖²)
#   M(p) = min pairwise centroid distance
#   q(p) = R(p) / M(p)
# Report q(p) for p ∈ {0, 4, 8, 11}
```
If q(p) increases monotonically across positions, we're oscillating (not refining). If q(p) decreases, we're converging.

Cost: one eval batch with intermediate hooks. Benefit: mechanism-level diagnostic (ThinC Recovery@k methodological pattern applied to our architecture).

### R7 — MTP head as auxiliary objective

Add D=1 MTP head during training (SlimQwen Eq. 9 structure):
```python
h^1 = TRM_1([RMSNorm(h^0); RMSNorm(Emb(t_{i+1}))] · M_1)
p^1_{i+1} = OutHead(h^1)
L_MTP-LM = CE(p^1_{i+1}, t_{i+1})
L_total = L_LM + β · L_MTP-LM      # β starts 0.3, decays to 0.1 cosine
```

SlimQwen: MTP improves backbone quality on knowledge benchmarks even without teacher (no KD needed). Sets up speculative decoding substrate for later.

**Cost:** ~1 extra transformer block + shared LMHead = ~5M params, ~5-10% tok/s.
**Benefit:** Free quality per SlimQwen §4.2; speculative decoding readiness.

### R8 — Skip: BLT-D, MISA, δ-mem, RigidFormer for now

- BLT-D: byte-level tokenization opposes our 32K BPE direction
- MISA: requires DSA-style architecture we don't have
- δ-mem: agent/memory use case we don't have
- RigidFormer: physics simulation, not LM
- SlimSpec: tests a draft-head rank; we have primary-head rank. Could test rank reduction (rank=256→128 or 96) as a separate optimization but unrelated to main recommendations.
- UniPrefill: only applicable when we serve inference at ≥32K context

---

## Appendix: Source Locations

| Paper | Local file | Size |
|---|---|---|
| MELT | `tool_e21350e06001OfBVOOOaOQD21N` | 120,887 B |
| LoopUS | `tool_e2135128e001g9B2w7OeuVHNoJ` | 130,992 B |
| Error Control | `tool_e21350f07001viIBIfXGUJmYTU` | 93,136 B |
| UniPrefill | (re-fetched complete) | — |
| MISA | `tool_e21350e09001S4oseuIJMeE9LC` | 102,280 B |
| δ-mem | (source code `declare-lab/delta-Mem`) | — |
| BLT-D | `tool_e21350e5900169fCEG2tK9gV77` | 110,761 B |
| SlimSpec | `tool_e2135112f001o6RORSCQ2zLdPu` | 73,892 B |
| SlimQwen | `tool_e21350fe5001IioVJP3HoHmTay` | 66,256 B |
| TextLDM | `tool_e21350dfe001hl2NQEqGKwxymS` | 65,166 B |
| ThinC | `tool_e21350e34001X52Npvg11NwnVn` | 56,851 B |
| RigidFormer | `tool_e21350e950012B4fi9pEMirQDN` | 120,644 B |

All raw files located at `C:\Users\z00517bz\.local\share\opencode\tool-output\`.

## Appendix: Methodological Notes

Each of the 4 explore-agent reports is 8,000-15,000 words. They contain additional per-paper detail not fully reproduced here (e.g., BLT-D attention mask diagrams, MELT's gate-variant sweeps, ThinC's Recovery@k curves). Refer to the raw agent outputs in the opencode session log (`ses_1dec75d4affeF7dNDue8MykeNi`, `ses_1dec713dbffeKDXTW3JdicajfD`, `ses_1dec6efb6ffeH5CC7HpGk1tkmN`, `ses_1dec6d981ffeecpwoFc19WHU6E`) for complete technical depth.

This document is the consolidated summary with precise recommendations for OdinPhantomLoop. Ship this; keep the raw reports for deeper technical dives.
