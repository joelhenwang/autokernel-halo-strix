# FrankenMoE-Loop v2 — Looped MoE + 2.5-iteration scheduling

**Date:** 2026-05-07
**Status:** Design approved, implementation blocked on FrankenMoE-Flat v1 L9
**Parent:** `knowledge/training/zaya1_8b_findings_2026.md` (P1 port track)
**Depends on:** FrankenMoE-Flat v1 (not yet specced — architecture + pretraining knobs, 75M/175M flat MoE Odin)
**Companion knowledge:** `knowledge/architectures/looped_moe_design_2026.md` (generalizable findings)

---

## 1. Goal

Produce the first Parcae-looped MoE model: an OdinHalo-backboned mixture-of-experts architecture that exploits the Parcae loop's weight-sharing premise **together with** MoE's expert-specialization premise, tuned for gfx1151's fp16-only / RDNA-3.5 / 2-node TB4-DDP environment.

Ship two architectural innovations beyond the baseline ZAYA1 × Odin combination:

1. **R2 sticky routing with E1 shared experts** — route once at iteration 0, replay expert assignments at iterations 1-2. Each chosen expert refines each of its tokens three times.
2. **2.5-iteration aggressive scheduling (Sched-A + M3)** — iteration 2 runs only the two NoPE-GQA blocks with narrowed FFN width. ~30% throughput win over full 3-iteration baseline.

**Research bet:** looped MoE is not just "MoE on a looped backbone" — the combination has architectural choices (routing policy, expert sharing, iteration schedule) with no canonical answer in the literature. This spec commits to specific answers and provides an ablation path (R1 via L11, Sched-B via L8.8) to validate them empirically.

## 2. Scope

### In scope

- Architectural design for FrankenMoE-Loop v2 with all decisions locked
- 2.5-iteration scheduling extension (v2.5 functionality) merged into the same spec
- Training curriculum (annealed hard schedule)
- MoE-specific scorecard metrics additions
- Rollout L0-L11 with pass criteria and ablations
- Risk register and mitigations

### Out of scope (deferred)

- SFT / RL cascade — ZAYA1 post-training recipe captured in `knowledge/training/zaya1_8b_findings_2026.md` §G; needs separate spec when we build RLVR infra
- Markovian RSA TTC harness — inference-side, post-train concern
- Per-token dynamic early exit (S4 / MoD-style) — kills compile on HIP; needs dedicated HIP kernel design
- M2 (converge-test batch-level depth gate) — listed as v3 candidate
- N5 (dual-route Parcae) — v3 candidate
- CCA / CCGQA attention — we're GQA already
- Scale beyond 75M active / 175M total

### Prerequisite

FrankenMoE-Flat v1 must complete L9 (full-epoch wikitext-103 DDP with quality ≥ baseline) before this spec's implementation starts. Rationale: v1 delivers battle-tested implementations of ScatterMoE dispatch, ZAYA1 MLP router, PID bias balancing, per-expert γ scaling, LZ77 + rare-token canaries, and MoE-aware scorecard metrics. v2 then becomes a focused additive effort on proven components rather than a full architectural research project.

## 3. Locked architectural decisions

### 3.1 Backbone

OdinHalo shared-layer structure, unchanged:

| Pos | Type | Notes |
|----:|------|------|
| 0 | HyPEShortConvBlock | Dense SwiGLU |
| 1 | HyPEShortConvBlock | **MoE SwiGLU** (4 experts) |
| 2 | NoPE-GQA block | Dense SwiGLU |
| 3 | HyPEShortConvBlock | Dense SwiGLU |
| 4 | HyPEShortConvBlock | **MoE SwiGLU** (4 experts) |
| 5 | NoPE-GQA block | Dense SwiGLU |

- `d_model = 768`, head_dim = 64, 12 heads, 4 KV heads
- `mean_recurrence = 3`, `backprop_depth = 3`
- 2 MoE layers at positions {1, 4} — sandwich the GQA blocks
- Factorized embed + head (rank 256), logit softcap 30
- Half-RoPE on NoPE-GQA blocks (ablation carried forward from v1-flat L2)

### 3.2 MoE configuration

- **Experts per MoE layer:** 4 (scaled down from ZAYA1's 16 for 75M-active budget; R2+E1 lets each expert process each token 3× which compensates)
- **Routing:** top-1, no residual expert
- **FFN width per expert:** `ffn_inner = 1536` (expansion factor 2)
- **Dispatch kernel:** ScatterMoE wrapped with `@torch.compiler.disable`

### 3.3 Routing policy: R2 sticky (locked)

```python
# Iteration 0: route all MoE layers, cache decisions
expert_assignments = {}  # {layer_idx: Tensor[B, T] of expert indices}
for layer_idx in moe_positions:
    scores = router(h_at_layer_idx)
    expert_assignments[layer_idx] = scores.argmax(-1)  # top-1
    h = dispatch_experts(h, expert_assignments[layer_idx])

# Iterations 1, 2: replay cached assignments
for iter_idx in [1, 2]:
    h = injection(h, input_embed)
    h = skip_gated(h, skip_state, iter_idx)
    for layer_idx in moe_positions:
        h = dispatch_experts(h, expert_assignments[layer_idx])  # same experts, different h
```

Consequences:
- **6 routing events per forward** instead of 18 (R1's total)
- **Load balancing** operates on the same per-(layer, token) unit as dense MoE — PID balancer port works unchanged
- **Expert specialization signal** intact — each expert sees a consistent token subset at 3 iteration stages
- **Router replay** generalizes ZAYA1's RL-side router replay (rollout→trainer consistency) to iteration-side (iter-0→iter-{1,2} consistency)
- **Gradient:** each expert accumulates gradient from 3 forward passes per token — an implicit 3× signal with strong per-token correlation

### 3.4 Expert sharing: E1 (locked)

Expert weights are literally the same tensor at iterations 0, 1, 2 — exactly mirroring how `shared_layers[i]` is reused for attention/dense blocks across iterations.

Fallback on empirical failure (per-expert gradient-norm diverging across iterations, visible in scorecard): upgrade to E3 = shared backbone + per-iteration LoRA adapter. Not built in v2; mitigation path only.

### 3.5 Router: ZAYA1 MLP + EDA + PID bias balancing

Port from ZAYA1 report §II-A-2:

```
r_l = W_down @ x_l                    # R = 256
r_l = r_l + γ · r_{l-1}               # EDA within-iteration only
s_l = softmax_fp32(MLP_3layer_GeLU(RMSNorm(r_l)))
e_idx = topk(s_l + b_l, k=1)
```

- **PID bias balancing:** separate AdamW optimizer, per-expert bias `b_{l,e}`, gradient `∇b_{l,e} = p_{l,e} − 1/E` where `p_{l,e}` is the empirical routing probability in the global batch
- **Router softmax in FP32** (extends the existing FP32-promotion list)
- **EDA scope:** within-iteration only. Iter-boundary resets `r_{l-1}` to zero. (Rationale: routing only happens at iter 0 anyway under R2.)

Router param count: `R·D + 3·R² + E·R = 256·768 + 3·256² + 4·256 ≈ 396K per MoE layer, ≈0.8M total`. Rounding error vs expert weights.

### 3.6 Novel: N1 per-expert per-iteration output scaling γ_{e,i}

```python
# Per MoE layer: (num_experts, mean_recurrence, d_model) learned tensor
self.expert_iter_gamma = nn.Parameter(
    torch.ones(num_experts, mean_recurrence, d_model)
)

# Forward at MoE layer:
expert_out = self.experts.dispatch(h, expert_assignments[layer_idx])
# Gather γ for (chosen expert, current iter) per token
gamma = self.expert_iter_gamma[expert_assignments[layer_idx], iter_idx]  # [B, T, D]
gamma = gamma.clamp(-4.0, 4.0)   # same ±4 clamp as iter_scales (fp16 safety)
expert_out = gamma * expert_out
h = h + expert_out  # residual
```

- **Parameters added:** `E · mean_recurrence · D = 4 · 3 · 768 = 9,216` per MoE layer, `~18K total` across 2 MoE layers. Rounding error.
- **Inductor compiles clean** (gather + elementwise multiply)
- **Addresses C6** (per-expert residual-norm drift across iterations)
- **Init:** ones; behaves as identity at start of training

### 3.7 Residual scaling: per-iteration (α_i, β_i)

Applied at every residual site (attention output residual + FFN output residual) at every iteration:

```python
# Per block: (mean_recurrence, d_model) for each of α and β
self.alpha = nn.Parameter(torch.ones(mean_recurrence, d_model))
self.beta = nn.Parameter(torch.zeros(mean_recurrence, d_model))

# Forward:
h_next = alpha[iter_idx] * h + Layer(x) + beta[iter_idx]
```

Per-site params: `2 · mean_recurrence · D = 2 · 3 · 768 ≈ 4.6K`. Across 6 layers × 2 sites = 12 sites → `~55K total`. Still rounding error.

### 3.8 2.5-iteration schedule: Sched-A + M3 (locked shipping path)

**Layer schedule (Sched-A, "GQA-polish"):**

```python
SCHEDULE = {
    0: (0, 1, 2, 3, 4, 5),      # full pass, 6 layer-forwards
    1: (0, 1, 2, 3, 4, 5),      # full pass, 6 layer-forwards
    2: (2, 5),                  # GQA-only polish, 2 layer-forwards
}
# Total: 14 layer-forwards vs baseline 18 = 2.33 equivalent iterations
```

Iteration 2 runs **only the two NoPE-GQA blocks**. Skipped layers (0, 1, 3, 4) — including both MoE layers — are elided from the Python loop in production mode.

**Narrow FFN at iter 2 (M3):** the two GQA blocks that run at iter 2 use their dense SwiGLU FFN at half-width:

```python
# At iter 2, dense FFN uses narrowed weights:
if iter_idx == 2 and self.m3_active:
    inner_half = self.ffn_inner // 2
    gate = x @ self.ffn.W_gate[:, :inner_half]
    up = x @ self.ffn.W_up[:, :inner_half]
    out = (F.silu(gate) * up) @ self.ffn.W_down[:inner_half, :]
else:
    out = self.ffn(x)  # full width
```

- No separate weights — single `.narrow()` view over existing tensors
- Backward gradient flows only through the narrowed slice
- **Expected wall-clock improvement:** ~30-35% vs v2 baseline (`14/18 × 50%-width FFN at iter 2`)

### 3.9 Training curriculum (annealed schedule)

Cold-start with the skip schedule in place risks unstable routing and under-trained γ_{e,i}. Three-phase curriculum:

| Phase | Step fraction | Schedule active | M3 active | Notes |
|-------|:-------------:|:---------------:|:---------:|-------|
| **Warmup** | 0 – 20 % | No (full 3×6) | No (full FFN) | Standard Parcae-MoE training; routing + PID converge |
| **Anneal** | 20 – 40 % | Soft via sigmoid gates λ_{l, i=2} | Soft via W_down scalar `m ∈ [1.0, 0.5]` | Gradient flows through "dying" paths; rest of network compensates |
| **Production** | 40 – 100 % | Hard elision in Python loop | Hard `.narrow()` slice | Full wall-clock savings realized |

**Soft-gate anneal (phase b):**
```python
# One learnable scalar per scheduled-skip layer, init high
self.skip_gate_logit = nn.Parameter(torch.full((len(skip_layers),), 3.0))

# Forward at iter 2 during anneal:
for skip_idx, layer_idx in enumerate(skip_layers):
    g = torch.sigmoid(self.skip_gate_logit[skip_idx])
    layer_out = self.shared_layers[layer_idx](h, ...)
    h = h + g * layer_out   # g → 0 turns layer off

# M3 anneal:
self.m3_scale = nn.Parameter(torch.tensor(1.0))
# clamped to [0.5, 1.0] via sigmoid-based parameterization
# At iter 2 dense FFN: out = ffn(x, width_mult=self.m3_scale)
```

**Hard-switch gate (phase b → c):**
- Only transition layers with `σ(skip_gate_logit) < 0.05`
- Layers with residual gate > 0.05 get extended anneal (another 5 % of steps)
- For M3: wait until `m3_scale < 0.55` before committing to half-width

### 3.10 Stack preservation (unchanged from OdinHalo)

- `SimpleParcaeInjection` at start of each iteration
- `skip_gates` carrying iter-to-iter state
- `loop_pos_embeds[iter_idx]` added after `iter_norm`
- `iter_scales[iter_idx].clamp(-4, 4)` applied after `iter_norm`
- `MoDA` depth-KV buffer across iterations (compatible with scheduled skips — layers not run at iter 2 contribute no iter-2 KV entries; downstream MoDA reads remain robust per existing `_run_shared_block` pattern)

### 3.11 Creative additions (zero-risk throughput + quality)

Five low-risk innovations beyond the core R2+E1+N1+Sched-A+M3 stack. Each is opt-in via a CLI flag, default-off during initial L0-L8 validation, default-on at L9 after confirming no regression.

#### T7 — R2 permutation caching ★ throughput, zero quality impact

Under R2 sticky routing, the ScatterMoE dispatch permutation (token → expert-bucket mapping, including the radix-sort of expert indices) is **identical at iter 0, 1, 2** for every MoE layer. Compute once at iter 0, cache in the same dict as expert assignments, reuse at iter 1 and iter 2.

```python
# Iter 0 (routing pass) per MoE layer:
expert_assignments[layer_idx] = scores.argmax(-1)              # [B, T]
permutation[layer_idx] = scatter_moe_sort(expert_assignments[layer_idx])  # cache

# Iter 1, 2 (replay) per MoE layer:
out = scatter_moe_dispatch(
    h, experts, permutation=permutation[layer_idx],   # bypass sort
)
```

- **Throughput win:** one radix sort + gather per (MoE layer × iter > 0). At 2 MoE layers × 2 iters = 4 saved sorts per forward. Single-digit % on the MoE path; infrastructure win.
- **Memory:** one extra `Tensor[B, T]` int32 per MoE layer. Negligible.
- **Correctness invariant:** `expert_assignments[layer_idx]` unchanged across iters (already enforced by `test_router_replay_determinism.py`), therefore permutation is unchanged — no new invariant to test.
- **CLI:** `--moe-cache-permutation` (default on once validated).

#### T11 — Iter-2 KV reuse via MoDA buffer ★ throughput + potential quality upside

Under Sched-A, iter 2 runs only the 2 NoPE-GQA blocks (positions 2, 5). MoDA already caches `(K, V)` for those layers at iter 0 and iter 1 in `depth_kv_buffer`. At iter 2 we **skip the K and V projections entirely** and reuse iter-1 KV; only Q is projected from iter-2's refined `h`.

```python
# Inside NoPE-GQA block forward at iter 2 under Sched-A + T11:
Q = self.w_q(h_iter2)                         # project Q fresh
if self.t11_kv_reuse and iter_idx == 2 and depth_kv_buffer:
    K_prev, V_prev = depth_kv_buffer[-1][self.layer_idx]   # iter-1 KV
    K, V = K_prev, V_prev
else:
    K, V = self.w_k(h_iter2), self.w_v(h_iter2)
# Attention with fresh Q, prior KV
attn_out = F.scaled_dot_product_attention(Q, K, V, ...)
```

- **Compute saved:** 2 matmuls per GQA block at iter 2, i.e. `2 × 2 = 4` matmuls per forward. Each projection is `d_model × d_kv = 768 × 256` ≈ 200K params worth of compute per block.
- **Overall throughput:** small absolute (~2 %), but **free** — zero new code (MoDA buffer exists), zero new parameters.
- **Quality hypothesis:** the most-refined Q (iter-2 `h`) attending to well-integrated KV (iter-1 output after full 6-layer pass) may be *better* than Q and KV both from a partially-converged iter-2 state. Refined query, mature keys. To be verified empirically.
- **Interaction with MoDA:** MoDA at later layers within iter 2 reads its own depth-KV buffer, which now would contain an iter-2 entry that was actually iter-1 KV reused. To avoid MoDA reading a stale buffer as if it were fresh, **T11 tags reused entries** so MoDA de-duplicates. Tagging is a boolean mask on the buffer, no compute cost.
- **CLI:** `--iter2-kv-reuse` (off during anneal phase since KV tensors may change shape if soft gates partially skip; on in production phase).

#### Q1 — Expert stochastic depth across iterations ★ quality, training-only

At training time, with probability `p_drop = 0.10`, replace an expert's output at a random (expert, iter) pair with identity for the token batch. Inference is untouched — zero latency impact at deploy.

```python
# Inside MoE layer forward during training:
for iter_idx in range(mean_recurrence):
    if training and torch.rand(1) < self.p_expert_dropout:
        # Drop this expert's contribution at this iteration
        expert_out = torch.zeros_like(x)
    else:
        expert_out = self.experts(x, ...)
    h = h + gamma_ei[iter_idx] * expert_out
```

- **Regularization intuition:** forces each (expert, iter) combination to be individually meaningful rather than each iteration freeloading on neighbours. Complements the per-expert per-iter γ scaling by pushing γ to learn real contribution weights instead of compensating for over-reliance.
- **Compatible with stochastic-depth literature** (Huang 2016); novel in the expert × iteration grain.
- **FLOP impact:** training sees 10% fewer expert forwards on average, slight training-time throughput *gain* (~3%). Inference unchanged.
- **CLI:** `--expert-iter-dropout 0.1` (default 0 until L5 validation; then enable).

#### Q10 — Routing temperature annealing ★ quality, zero cost

Softmax temperature `τ` in the router starts at 2.0 (softer distribution → more exploration) and anneals to 1.0 (standard) linearly over the first 10 % of training steps.

```python
# In the router:
tau = self.temperature_schedule(global_step)   # 2.0 → 1.0 over 10% of steps
scores = F.softmax(logits / tau, dim=-1)
e_idx = topk(scores + b, k=1)
```

- **Rationale:** at early training, expert weights are near-identical (xavier init), so `softmax(logits)` is flat. Temperature 2.0 keeps the gradient signal distributed across experts; as differentiation emerges, cooling to 1.0 sharpens the commitment. Standard MoE warmup trick documented in multiple papers (Switch Transformer, DeepSeek-MoE).
- **Cost:** one scalar schedule evaluation per forward. Free.
- **Composability with PID balancing:** neutral. PID operates on empirical routing probabilities, which are sharper at lower temperature; the schedule doesn't change the balancing target.
- **CLI:** `--router-temp-anneal` (default on; schedule hardcoded 2.0→1.0 over 10% of total steps).

#### Q3 — Iteration-varying RoPE base for conv blocks ★ quality, zero cost

Conv blocks at iter 0 use RoPE base `θ₀ = 10,000` (the OdinHalo default — local positional sensitivity). Iter 1 uses `θ₁ = 30,000`. Iter 2 uses `θ₂ = 100,000`. Progressive broadening of the effective positional window across iterations.

```python
# At init, precompute three freqs_cis tables:
self.register_buffer("freqs_cos_iter", torch.stack([
    precompute_freqs_cis(head_dim, max_seq_len, base=theta)
    for theta in [10_000, 30_000, 100_000]
]).real.float(), persistent=False)
# (similar for freqs_sin)

# In conv-block forward at iteration iter_idx:
freqs_cis = torch.polar(
    self.freqs_cos_iter[iter_idx][:T],
    self.freqs_sin_iter[iter_idx][:T],
)
```

- **Rationale:** Parcae iterations re-apply the same weights with progressively more context. If the inductive bias is "each iteration integrates further," then the *positional* scale should match — iter 0 sees local structure, iter 2 sees long-range structure. This mirrors how different RoPE bases (10K, 100K, 1M) generalize to different context lengths in published work (Llama-3, Qwen3); we compress that "context-length curriculum" into iteration stages.
- **Parameter cost:** three freqs_cis tables instead of one. Precomputed at init, stored as non-persistent buffers. ~3× the RoPE cache memory (`3 × head_dim × max_seq_len × 4 bytes` ≈ 500 KB for our config). Rounding error.
- **FLOP cost:** zero — same number of RoPE applications, just different θ.
- **Compile:** buffer indexing by iter_idx is a tensor gather; Inductor specializes cleanly if the index is a Python int (it is — `iter_idx` is a Python loop variable).
- **CLI:** `--iter-varying-rope` (default off initially; enable after L3 if no regression).

### 3.12 Further candidates (v3, not in v2 scope)

Flagged for later, documented here so they're not re-invented:

- **T6: Windowed attention at iter 2 (W=128).** At Sched-A iter 2, switch both GQA blocks to sliding-window attention. Potentially large throughput win at block ≥ 512 (70 % attention-FLOP reduction at iter 2 → 10-15 % overall), but breaks iter 2's "global integration" inductive bias. Ablate as an L8.8 alternate schedule if Sched-A's quality is strong enough to spare.
- **Q6: Iteration-consistency auxiliary CE loss.** Compute CE at iter 1's output with weight 0.1. Training-only. Candidate if L9 quality gap vs OdinHalo is borderline.
- **N-Combo-3: Attention-scale schedule across iterations.** Local at iter 0, windowed at iter 1, full at iter 2 (or the reverse). Radical reframing of Parcae as a spatial-scale progression. v3 research direction.
- **Expert capacity factor modulation.** Let capacity factor `c` vary by iteration (higher at iter 0 when routing is least certain, lower at iter 1-2 under sticky replay). Requires non-trivial ScatterMoE config; defer.
- **Delta-activation iter 2.** Compute only `Δh = Layer(h_iter1) - h_iter1` at iter 2, add to stored `h_iter1`. Math-wise similar to T11 but for full layers. Requires careful numerical analysis for fp16 stability. v3.

## 4. FP32 promotion list (superset of ZAYA1)

Operations that run in FP32 on both (future) rollout engine and trainer — total list for FrankenMoE-Loop:

| Op | Source |
|----|--------|
| LM-head matmul + fused CE | existing (Sprint 2) |
| QK-norm, QK-mean | existing |
| RMSNorm | existing |
| Residual stream additions | existing |
| `iter_scales[iter_idx]` multiply | existing |
| **Router softmax** | NEW (ZAYA1) |
| **Load-balance loss** | NEW (MoE) |
| **PID bias update** | NEW (ZAYA1) |
| **Per-iter (α_i, β_i) application** | NEW |
| **Per-expert per-iter γ_{e,i} application** | NEW (N1) |
| **M3 FFN half-width matmul** | NEW (optional, reuse LM-head precedent) |
| **Soft skip-gate sigmoid (anneal only)** | NEW |
| **Router temperature division (`logits / τ`)** | NEW (Q10) |
| **T11 reused-KV scaled_dot_product_attention** | NEW (Q-fp32-at-iter2 only; KV stays fp16 from MoDA buffer) |

Existing `--z-loss`, `--attn-softcap`, `--activation-monitor`, `GradScaler growth_interval=500`, resume-tightened `--max-grad-norm 0.8` all continue to apply.

## 5. Compile strategy

- **Dense backbone layers** (attention blocks, dense SwiGLU, RMSNorm, iter_norm, injection): compiled via existing `compile_zones(mode=max-autotune-no-cudagraphs)`
- **MoE FFN layers:** wrap ScatterMoE dispatch with `@torch.compiler.disable` — same pattern as `fused_rope_gate_mul` per STATUS.md Phase 3 WI-A0
- **Hard schedule (production phase):** `SCHEDULE` is a Python-constant tuple-of-tuples; `for layer_idx in SCHEDULE[iter_idx]:` unrolls cleanly under Inductor specialization
- **Soft schedule (anneal phase):** sigmoid gate introduces a `.item()` or tensor-scalar multiply; compile-safe via constant propagation. Gate parameter is never sharded.
- **M3 `.narrow()`:** compiles cleanly as a constant-stride slice. Inductor specializes on the half-width shape.
- **Expert assignment cache:** Python dict keyed on layer index, values are `Tensor[B, T]`. Dict access at the top of each MoE forward is not inside a compiled region (ScatterMoE wrapper is `@torch.compiler.disable`).
- **T7 permutation cache:** same pattern — dict mutated outside compiled region, read-only inside.
- **T11 KV reuse:** the `if iter_idx == 2 and t11_kv_reuse` branch is a Python-level constant at forward time (iter_idx is a loop variable, t11_kv_reuse is a module attribute). Inductor specializes on each branch; both compile clean.
- **Q3 iter-varying freqs_cis:** `self.freqs_cos_iter[iter_idx]` is a constant-index tensor gather. Compiles clean.
- **Q1 expert dropout:** wrapped in `if self.training` + `torch.rand(1) < p`. The random branch is data-dependent but at the batch-grain, not per-token — Inductor handles this via a conditional subgraph. If it graph-breaks in practice, wrap the dropped-vs-full paths in `@torch.compiler.disable`; overhead is 0.1 probability × one guard check.

**Reject list:** `reduce-overhead` mode (HIP graph capture empty-graph bug from Phase 3 WI-A0).

## 6. MoE-specific scorecard metrics (new module)

Add `halo_training/eval/moe_stats.py`, invoked by `eval_checkpoint.py` when the model reports `has_moe_layers = True`:

### Per-MoE-layer metrics

- `router_entropy_mean` — mean per-layer Shannon entropy of routing probabilities, nats
- `router_entropy_by_iter` — only populated under R1 ablation; same but (layer, iter) axis
- `tokens_per_expert_histogram` — list of expert-load fractions over the full eval split
- `dead_expert_count` — experts receiving < 0.1 % of tokens over eval
- `expert_imbalance_ratio` — max/min expert-load ratio per layer
- `grad_norm_per_expert` — captured during final training microbatch (forensic; opt-in)

### Per-iteration metrics

- `gamma_ei_histogram` — flattened `expert_iter_gamma` distribution per layer at each iter
- `gamma_ei_clipped_count` — fraction of γ values hitting the ±4 clamp
- `residual_norm_by_iter` — L2 norm of residual stream at end of each iteration (forensic for C6)
- `skip_gate_lambda_by_layer` — during anneal only, current σ(λ) for each scheduled-skip layer
- `m3_scale_value` — current anneal value of the FFN half-width scalar

### Creative-addition metrics

- `router_temperature` — current τ value (Q10); should converge to 1.0 after 10 % of training
- `expert_iter_dropout_events_per_step` — count of stochastic-depth drops applied (Q1; training only)
- `t11_kv_reuse_active_iters` — list of iterations where KV reuse was engaged (should be exactly `[2]` under Sched-A + T11)
- `permutation_cache_hit_rate` — should be 1.0 after iter 0 under R2 + T7 (invariant check)
- `rope_base_by_iter` — emits `[10_000, 30_000, 100_000]` or `[10_000]×3` depending on whether Q3 is enabled

### Sanity assertions

- Under R2 sticky: per-token per-layer expert assignment must be **identical** across iterations within a single forward. Unit test `test_router_replay_determinism.py`.
- Under hard schedule: layers in `skip_layers` must not appear in profile trace for iter 2. Unit test `test_schedule_elision.py`.
- Under E1 shared: expert weight tensors must be the same `id()` across iterations. Unit test `test_expert_weight_sharing.py`.

## 7. Parameter budget

Target: 75M active / 175M total (per earlier design decision).

| Component | Unique params |
|-----------|--------------:|
| Factorized embed + head (shared weights) | ~10 M |
| Dense backbone unique: 4 conv blocks + 2 GQA blocks + 4 dense SwiGLU FFN | ~40 M |
| MoE layers: 2 × 4 experts × ~3.5 M per expert | ~28 M |
| Router MLP + EDA + PID biases (2 layers) | ~0.8 M |
| Per-iter (α_i, β_i) at 12 residual sites | ~55 K |
| Per-expert per-iter γ_{e, i} (N1) at 2 MoE layers | ~18 K |
| Skip-gate logits + M3 scale | ~10 scalars |
| **Unique total** | **~79 M** |
| Active FLOPs per iter (dense + 1 expert × 2 layers) | ~45 M-equivalent |
| Effective FLOPs (baseline 3 iters × 6 layers) | ~135 M-equiv |
| Effective FLOPs (Sched-A + M3 production phase) | ~95 M-equiv |

"Effective total" in OdinHalo's sense ≈ 180-200 M. Sits in the 75M active / 175M total target.

## 8. Rollout plan

Ordered 12-step descent. Each step is 200 DDP steps + eval scorecard unless noted. Fallback = revert the last-enabled flag.

| Step | Change | New scorecard metrics | Pass criterion |
|------|--------|-----------------------|----------------|
| **L0** | Baseline: OdinHalo 58M, sweep defaults | existing | reproduce step_1869 loss 4.71, BPB 1.89 |
| **L1** | + Learned per-iter `(α_i, β_i)` at 6×2=12 sites | `alpha_by_iter_distribution` | BPB ≤ L0 + 0.5 %, tok/s ≥ −2 % |
| **L2** | + Half-RoPE on NoPE-GQA blocks | — | BPB ≤ L1 + 0.5 %, tok/s ≥ −1 % |
| **L3** | + Single MoE layer (pos 1), 4 experts, linear router, **R1 naive per-iter** | entropy, per-expert tokens, per-expert grad-norm | trains without NaN; entropy > 1.3 nats; no dead experts |
| **L4** | + Switch routing to **R2 sticky** (iter 0 → replay) | expert-assignment determinism test | BPB ≤ L3, tok/s ≥ +5 % (fewer routing ops) |
| **L4.5** | + **T7 permutation caching** under R2 | `permutation_cache_hit_rate` | BPB identical to L4 (invariant); tok/s ≥ +1-3 % |
| **L5** | + Replace linear router with ZAYA1 MLP + EDA + PID balancing | balance histogram entropy | PID settles in < 500 steps; entropy stays > 1.3 |
| **L5.5** | + **Q10 routing temperature annealing** (τ: 2.0 → 1.0 over 10 %) | `router_temperature` curve | BPB ≤ L5; entropy curve smoother at early steps |
| **L6** | + N1 γ_{e, i} (per-expert per-iter scaling) | γ distribution, γ-clamp-hit-rate | γ stays in ±4; BPB ≤ L5 |
| **L6.5** | + **Q1 expert stochastic depth** (`p=0.1`) | `expert_iter_dropout_events_per_step` | BPB ≤ L6 + 0.5 %; training-tok/s ≥ L6 (dropout reduces compute) |
| **L7** | + Second MoE layer (pos 4) — full FrankenMoE-Loop v2 baseline | all-of-above | BPB ≤ OdinFlat gpt-small (5.07) on comparable resume |
| **L7.5** | + **Q3 iter-varying RoPE base** (10K/30K/100K) | `rope_base_by_iter` | BPB ≤ L7 + 0.5 %; tok/s within ±1 % |
| **L8** | + LZ77 + rare-token canaries | canary hit-rate | diagnostic only |
| **L8.5** | + Soft schedule gates λ_{l, i=2} at init g = 0.95 + M3 scale `m = 1.0` | `skip_gate_lambda_by_layer`, `m3_scale_value` | trains unchanged vs L8; gates + m stable |
| **L8.6** | Anneal λ toward **Sched-A** target + M3 scale toward 0.5 | anneal trajectory visible | converges; BPB ≤ L8 + 0.5 % |
| **L8.7** | Hard-switch: elide scheduled-skip layers + use narrow FFN at iter 2 | throughput delta | tok/s +25–33 %; BPB regression < 1 % |
| **L8.75** | + **T11 iter-2 KV reuse via MoDA buffer** (only meaningful after hard-switch; needs production phase) | `t11_kv_reuse_active_iters == [2]` | BPB ≤ L8.7 + 0.5 % (expect parity or improvement); tok/s ≥ +1-2 % |
| **L8.8** | Ablation: repeat L8.5-L8.75 from fresh L8 checkpoint with **Sched-B** `[3, 4, 5]` | same | compare BPB vs throughput tradeoff, pick winner as production |
| **L9** | Full-epoch wikitext-103 DDP + scorecard + machine parity | full scorecard | beat OdinHalo step_1869 on ≥ 2 of 4 held-out BPB domains |
| **L10** | Resume on gpt-training-small → stem-crawl → dolma-10B | existing trajectories | loss trajectory ≤ OdinHalo at each checkpoint |
| **L11** | Ablation: run **R1 + N3 (stickiness bonus) + N4 (iter-conditioned router input)** from fresh L5 checkpoint to full epoch | router entropy per (layer, iter), iter-to-iter switching rate | if R1 ≥ R2 at full epoch, R2 choice is validated; if R1 > R2, promote R1 as v3 candidate |

### Step cost estimates

| Range | Wall-clock | Notes |
|-------|------------|-------|
| L0-L7.5 | ~1.5-2.5 weeks | Stage-wise MoE buildup + T7/Q10/Q1/Q3 creative additions interleaved; debugging-heavy at L3 (first MoE + Parcae interaction) |
| L8-L8.8 | ~1.5-2.5 weeks | Anneal + schedule + M3 + T11 KV reuse + Sched-A/B ablation |
| L9 | 1-2 days | Full epoch wikitext-103 (~1 hour DDP + scorecard) |
| L10 | months | Maps to existing Sprint 3 dolma-10B trajectory |
| L11 | ~1 week | Single full-epoch ablation run |

Total L0-L9: **~4-6 weeks** once triggered.
Total L0-L11: **~6-8 weeks**.

## 9. Risk register

| ID | Risk | Severity | Likelihood | Mitigation |
|----|------|---------:|-----------:|------------|
| RL-1 | R2 sticky routing causes gradient conflict across iterations (expert sees different "refinement stages" of same tokens) | High | Med | N1 γ_{e,i} is the primary mitigation; E3 fallback (per-iter LoRA adapter) documented, not built |
| RL-2 | PID balancing fails to converge at 4 experts (too few for statistical balance at our DDP batch size) | Med | Med | Start L3 with 2 experts; add auxiliary load-balance loss; fall back to 8 experts per MoE layer |
| RL-3 | ScatterMoE + Parcae loop produces compile errors beyond the `@torch.compiler.disable` pattern | Med | Med | Keep MoE FFNs entirely eager; compile only dense path; quantify cost vs dense-only run |
| RL-4 | MoDA depth-KV × MoE routing has unforeseen interaction | Med | Low | MoDA on attention out, MoE on FFN out — orthogonal paths; MoDA-off ablation at L7 validates |
| RL-5 | fp16 overflow in expert output accumulation across 3 iterations (even with γ clamp) | Low | Med | `halo_training/eval/moe_stats.py::residual_norm_by_iter`; z-loss already in place; extend activation-monitor |
| RL-6 | R2 wins because it's easy to train, not because it's capability-superior | Low | Med | L11 R1 + N3 + N4 ablation explicitly measures this |
| RL-7 | Anneal phase leaves `skip_gate_logit` stuck in intermediate range at hard-switch | Med | Med | Only transition layers with σ(λ) < 0.05; extended-anneal budget of additional 5 % steps |
| RL-8 | Hard-switch (L8.7) introduces discontinuity spike in training loss | Low | Med | Switch at epoch boundary + warmup restart (LR 6e-4, warmup 500 steps), matches resumed-run defaults |
| RL-9 | M3 `.narrow()` interacts with NorMuon Newton-Schulz (half-width shape may trip Polar Express init constants) | Low | Med | NS operates on `W_gate`, `W_up`, `W_down` full tensors during optimizer step — `.narrow()` is forward-only. Verify in unit test |
| RL-10 | Sched-A (iter 2 = GQA only) is too aggressive → BPB regression > 2 % | Med | Med | L8.8 Sched-B runs in parallel as fallback; production switches to whichever wins |
| RL-11 | At 58M unique params, MoE gives no measurable gain over well-trained dense | Low | Med | L11 R1 ablation + dense baseline comparison; if confirmed, publishable negative result "MoE in Parcae at this scale" |
| RL-12 | Training time dominated by MoE-FFN-eager overhead, not compute | Low | Med | Accept; OdinHalo's compile lift is already modest (5.17% from Phase 3); MoE dispatch cost at 4 experts × 2 layers is bounded |
| RL-13 | T7 permutation cache stale if `expert_assignments` mutates between iters (bug in replay) | Low | Low | `test_router_replay_determinism.py` asserts invariance; cache assertion added to forward |
| RL-14 | T11 iter-2 KV reuse makes iter-2 attention fundamentally different from iter 0/1 (quality drop or gain) | Med | Med | Default off in L8.75; ablate explicitly; quality hypothesis is reused-KV ≥ fresh-KV, verify empirically |
| RL-15 | T11 + MoDA interaction: MoDA downstream of an iter-2 reused-KV block reads what it believes is fresh iter-2 KV but is iter-1 | Med | Med | Tag reused entries in `depth_kv_buffer`; MoDA de-duplicates. Covered by new unit test `test_t11_moda_dedup.py` |
| RL-16 | Q1 expert stochastic depth causes γ_{e,i} to drift (compensating for dropped events) | Low | Med | γ-distribution scorecard captures this; drop rate 0.1 is low enough that compensation is minor |
| RL-17 | Q10 temperature annealing fights PID balancing (high τ = softer routing = harder to balance) | Low | Med | PID has AdamW inner loop that adapts to gradient scale; monitor `expert_imbalance_ratio` during anneal window |
| RL-18 | Q3 iter-varying RoPE breaks positional consistency across the loop (same position encodes differently per iter) | Low | Med | Hypothesis is this is a feature, not a bug (progressive broadening). If regresses, disable — zero cost |

## 10. Deliverables on implementation

### New files

- `models/franken_moe_loop.py` — new model class, inherits from refactored `OdinHaloBase` with MoE-FFN support hooks
- `models/components/moe_ffn.py` — `ZAYA1Router`, `PIDBalancer`, `ExpertPool`, `MoESwiGLU` with ScatterMoE dispatch
- `models/components/residual_scaling.py` — `PerIterResidualScale(α_i, β_i)` utility
- `halo_training/content_canaries.py` — `lz77_min_ratio`, `rare_token_frac` (shared between pretraining and future RL)
- `halo_training/eval/moe_stats.py` — MoE-specific scorecard evaluator
- `halo_training/schedule.py` — `LayerSchedule` helper + soft-gate anneal + hard-switch controller
- `scripts/test_router_replay_determinism.py` — R2 sticky invariant check (also validates T7 permutation-cache invariant)
- `scripts/test_schedule_elision.py` — hard-skip correctness check
- `scripts/test_expert_weight_sharing.py` — E1 invariant check
- `scripts/test_t11_moda_dedup.py` — asserts MoDA doesn't double-count reused-KV entries (RL-15)

### Modified files

- `models/odin_halo.py` — refactor `OdinHaloBase._run_shared_block` to accept a layer schedule; add optional MoE-layer hooks
- `halo_training/cli.py` — flags: `--moe-experts`, `--moe-layers`, `--routing-policy {r1,r2}`, `--expert-iter-gamma`, `--layer-schedule {full,sched-a,sched-b}`, `--m3-narrow-ffn`, `--schedule-anneal-fraction`, **plus creative-addition flags**: `--moe-cache-permutation` (T7), `--iter2-kv-reuse` (T11), `--expert-iter-dropout FLOAT` (Q1), `--router-temp-anneal` (Q10), `--iter-varying-rope` (Q3)
- `halo_training/trainer.py` — anneal → hard-switch lifecycle; schedule-aware compile region
- `scripts/eval_checkpoint.py` — invoke `moe_stats` evaluator when model declares `has_moe_layers = True`
- `knowledge/INDEX.md` — link spec + companion knowledge doc
- `STATUS.md` — spec-approved status entry
- `AGENTS.md` — add training-gotchas entry for looped MoE compile-disable pattern

## 11. Out-of-scope (punt list)

Items explicitly deferred from v2 scope and their trigger conditions for future work:

| Punt | Trigger to revisit |
|------|---------------------|
| **S4: per-token early exit (MoD-style)** | If L8.7 throughput gain is insufficient and we want another +20 % — needs HIP kernel for masked dispatch |
| **M2: converge-test batch-level depth gate** | If L8.8 shows batch-to-batch variance in "which schedule works best" |
| **N2: refinement-gated skip (per-token confidence)** | v3+; subsumes M2 at per-token grain |
| **N5: dual-route Parcae** | v3+ research direction; only pursue if L11 shows R2/R1 dichotomy is the bottleneck |
| **E3: per-iteration LoRA adapter on shared experts** | L7 scorecard shows per-iter grad-norm divergence > 2× across iterations |
| **Agentic / tool RL** | Separate spec entirely; blocked on SFT + RL infra |
| **Markovian RSA inference harness** | After post-training infra lands; captured in `knowledge/training/zaya1_8b_findings_2026.md` §G.5 |
| **Scale beyond 75M active** | After dolma-10B L10 completes and we have a token-efficiency curve to justify |

## 12. Open questions (flag for implementation)

1. **Should routing happen at iter 0 before the first layer of each MoE position, or once globally at the top of forward?** Locked answer: once per MoE layer at iter 0, using that layer's input representation. Allows each MoE layer's router to see position-appropriate features.
2. **Does the expert_assignments dict sit on GPU or pinned CPU?** Locked answer: GPU, same device as the model, `Tensor[B, T]` of int32. Memory cost trivial (batch=16 × block=256 × 2 layers × 4B = 32 KB).
3. **Does the anneal-to-hard-switch commit happen mid-batch or at an epoch boundary?** Locked answer: epoch boundary, followed by 500-step LR-warmup restart at 6e-4 (matches resumed-training defaults per AGENTS.md DDP knobs).
4. **Do we auto-eval during L8.6 anneal phase?** Locked answer: yes, every 500 steps to catch degenerate trajectories before hard-switch.

## 13. Related docs

- `docs/research/zaya1_8b_technical_report/zaya1_8b_technical_report.md` — source architecture report
- `knowledge/training/zaya1_8b_findings_2026.md` — applied-findings synthesis
- `knowledge/architectures/looped_moe_design_2026.md` — generalizable design notes (companion to this spec)
- `knowledge/training/imu1_recipe_2026.md` — NorMuon + CWD recipe, unchanged in this design
- `knowledge/training/fp16_stability_gfx1151.md` — fp16 hardening; extends with new FP32-promoted ops
- `knowledge/architectures/parcae_stable_looped_models.md` — Parcae reference, unchanged foundation
- `STATUS.md` Phase 3 WI-A0 — documents `reduce-overhead` failure on HIP (why we stay on `max-autotune-no-cudagraphs`)
- `STATUS.md` Sprint 1.1 — NorMuon fp16-NS configuration that FrankenMoE-Loop inherits
