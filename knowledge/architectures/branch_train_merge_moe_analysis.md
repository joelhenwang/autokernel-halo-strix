---
title: "Branch-Train-Merge MoE from Dense Checkpoints — Feasibility Analysis"
domain: architectures
type: reference
status: active
related:
  - knowledge/architectures/paper_deep_dive_2026_05.md
  - knowledge/architectures/reliable_small_lm_insights.md
  - docs/superpowers/specs/2026-05-03-vidar-halo-design.md
tags: [%moe, %btx, %bar, %sparse-upcycling, %model-merging, %domain-experts]
---

# Branch-Train-Merge MoE from Dense Checkpoints

**Date:** 2026-05-04
**Question:** Can we pretrain a 150M dense LLM, fork into domain-specific checkpoints, post-train each, and merge into an MoE of small LMs?
**Answer:** Yes — this is exactly the BTX/BAR pipeline, validated at 7B scale. Untested at 150M. Capacity-per-expert is the key risk.

---

## Literature Summary

### Branch-Train-Merge (BTM) — Meta/FAIR, 2022

arXiv:2208.03306. Fork seed LM → train 64 domain experts independently → ensemble at inference (weighted by domain classifier).

- 22.4B total (64×350M experts)
- Matches LM trained with 2.5× more compute
- Random splits fail — meaningful domain boundaries required
- No learned token-level routing (coarse domain selection)

### Branch-Train-MiX (BTX) — Meta, 2024

arXiv:2403.07816. Fork Llama-2 7B → train domain branches (math 201B tokens, code 210B, wiki 42B) → merge FFNs into MoE + average attention → fine-tune router on 80B mixed tokens.

- 4×7B = 28B total, 11.1B active (top-2)
- Beats Llama-2 13B (47.9 vs 45.4 avg)
- Beats BTM ensemble (47.9 vs 43.4) — router learning is critical
- Beats sparse upcycling (47.9 vs 46.3) — domain specialization helps
- 77% compute on async expert training, 23% on MoE fine-tuning
- Load balancing critical to prevent dead experts

### BAR (Branch-Adapt-Route) — Allen AI, April 2026

arXiv:2604.18473. Fork OLMo 2 7B → train 4 domain experts through full post-training pipeline (mid-train + SFT + RLVR) → frozen anchor expert + domain FFN experts + trained router.

- 5×7B = 35B total, ~14B active
- BAR 49.1 vs full retraining 50.5 (only 1.4 points behind)
- **Dense merging catastrophically fails (6.5/100)** after mid-training divergence
- Modular upgrades: code v1→v2 = +16.5 code, zero interference elsewhere
- Progressive unfreezing: FFN-only → embeddings/head → attention during RLVR
- Domain-only SFT destroys general capabilities — must mix domain + general data
- Router trained on 5% stratified SFT data, all non-router weights frozen
- Anchor expert (frozen post-trained FFN) essential for preserving base capabilities

### Sparse Upcycling — Google, 2022

arXiv:2212.05055. Dense checkpoint → replicate FFN N times → add router → continue MoE training.

- Upcycled models beat MoE trained from scratch at matched compute
- Qwen1.5-MoE: 1.8B → 64 experts, 2.7B active, matches Qwen-7B (75% training cost reduction)
- LLaMA-MoE: 7B → partition FFN → 3.5B active, beats same-size dense
- Random initialization noise speeds convergence

### Model Merging Techniques

| Method | Key Idea | When to Use |
|--------|----------|-------------|
| Task Arithmetic (2023) | task_vector = finetuned - pretrained; add/negate | Simple capability composition |
| Model Soups (2022) | Average weights of different fine-tunes | Same-architecture, same-base models |
| TIES-Merging (2023) | Trim noise, resolve sign conflicts, merge aligned | Reducing interference across merges |
| DARE (2024) | Drop 90-99% of deltas, rescale remainder | Large models, sparsification before merge |
| DELLA (2024) | Magnitude-aware pruning before merge | Best general merger (+3.6 over TIES) |
| Evolutionary (2025, Sakana AI) | Evolve merge recipes in param + data flow space | Cross-domain, zero additional training |

**Critical finding:** Dense merging (TIES/DARE/soups) fails catastrophically after mid-training divergence (BAR: 6.5/100). Must use routing, not averaging, for experts that have diverged significantly.

### LoRA-to-MoE Approaches

- **LoRAMoE (2023):** Multiple LoRA adapters + router = plugin MoE. Routes knowledge-preservation vs task LoRAs.
- **PHATGOOSE (2024):** Post-hoc per-token per-layer routing among PEFT modules. No simultaneous data access needed.
- **X-LoRA (2024):** Deep layer-wise token-level mixing of LoRA adapters via gating on hidden states.

---

## Two Paradigms: Split-Model MoE vs Full-Model MoE

The original question has two distinct interpretations:

### Paradigm 1: Split-Model MoE (BTX/BAR standard approach)

```
1× 150M model → split FFN layers into domain experts
Total: ~160M, Active: ~77M per token
Expert = one FFN layer (~17M params)
```

### Paradigm 2: Full-Model MoE (proposed approach)

```
N× 150M models → each is a complete domain-specialized model
Total: N×150M, Active: ~150M per token (top-1) or ~300M (top-2)
Expert = entire 150M model with full attention + FFN + embeddings
```

Paradigm 2 is the core vision: **train N complete 150M models, each post-trained on a different domain, then compose them via routing into a unified system.** This eliminates the capacity-per-expert concern entirely — each expert has the full 150M budget for its domain.

---

## Paradigm 2: Full-Model MoE — Architecture Options

### Common Pipeline (all options share this)

```
Pretrain: 1× VidarHalo ~150M effective on dolma-10b (general base)
    │
    ├──→ Fork 1: Full 150M copy, post-train on Code data
    ├──→ Fork 2: Full 150M copy, post-train on STEM/Math data
    ├──→ Fork 3: Full 150M copy, post-train on Tool Use data
    ├──→ Anchor: Original post-trained general model (frozen)
    │
    └── Combine via one of the routing options below
```

With 4 experts: **600M total params.** On Strix Halo (128GB unified memory) this is trivial.

---

### Option A: Per-Token Per-Layer Routing (finest grain)

```python
for layer_idx in range(n_layers):
    for token_idx in range(seq_len):
        expert_id = router[layer_idx](hidden[token_idx])  # top-1 or top-2
        hidden[token_idx] = experts[expert_id].layers[layer_idx](hidden[token_idx])
```

Every layer independently routes each token to one of N expert layer implementations. Attention AND FFN differ per expert.

| Property | Value |
|----------|-------|
| Granularity | Per-token, per-layer |
| Total params | N × 150M |
| Active params (top-1) | 150M |
| Active params (top-2) | 300M |
| Router params | N × d_model per layer |

**Pros:**
- Maximum flexibility — different tokens routed to different experts within one sequence
- A code snippet inside a math document routes to code expert for those tokens
- Each expert's attention patterns fully preserved (no averaging)

**Cons:**
- **Hidden state coherence:** Token t processed by code expert, token t+1 by math expert. Their hidden states evolved through different attention weights. When token t+1 attends to token t's KV cache, the representations may be incompatible.
- **KV cache explosion:** Each expert produces different K/V projections. With top-1 routing changing per-token, KV cache can't be shared across experts.
- **Parcae loop chaos:** MoDA depth KVs from iteration 0 (routed to expert A) consumed in iteration 1 (potentially routed to expert B). Mismatched projections.
- **Router training complexity:** Must learn N × n_layers routing decisions simultaneously.

**Mitigation for hidden state coherence:**
- **Shared embeddings + shared LM head:** All experts share input/output embeddings (averaged). Only transformer layers are routed. Tokens enter and exit through the same representation space.
- **Shared attention, routed FFN only:** Reduces to BTX-style. Loses the domain-attention benefit but fixes coherence. See Option C.
- **Sticky routing:** Force all tokens in a sequence through the same expert per layer. See Option D.

---

### Option B: Per-Sequence Model Selection (coarsest grain)

```python
expert_id = sequence_router(mean_pool(embeddings))  # one decision per input
output = experts[expert_id].forward(input_ids)
```

Classify entire input, route to one complete model.

| Property | Value |
|----------|-------|
| Granularity | Per-sequence |
| Total params | N × 150M |
| Active params | 150M (always top-1) |
| Router params | N × d_model (single classifier) |

**Pros:**
- Simplest. Each model runs as-is — no architectural modification needed.
- No hidden state coherence issues. No KV cache complications.
- Each expert independently deployable (load only code model for code tasks).
- Router is a simple classifier, trainable on labeled domain data.

**Cons:**
- **Coarse routing.** A "write Python to solve this integral" is both code AND math. Which expert?
- **BTM proved this inferior:** Per-sequence ensemble (BTM) = 43.4 avg vs per-token MoE (BTX) = 47.9. Token-level routing wins by 4.5 points.
- **No blending.** Can't combine code knowledge and math knowledge within one response.
- **Wasted capacity.** Three experts sit idle per sequence.

**When it works anyway:**
- **Task-specific deployment.** You KNOW the user wants code → load code model. No router needed. This is what Liquid AI recommends: "The more narrow you design it, the better."
- **Agentic routing.** An orchestrator LLM selects which expert to call per turn. More flexible than a learned router for multi-step tasks.

---

### Option C: Shared Attention + Routed FFN (BTX-style, proven)

```python
# Shared attention (averaged across all forks)
shared_attn = average(expert_0.attn, expert_1.attn, ..., expert_N.attn)

for layer_idx in range(n_layers):
    attn_out = shared_attn[layer_idx](hidden, freqs_cis)
    expert_id = router[layer_idx](attn_out)  # per-token
    ffn_out = experts[expert_id].ffn[layer_idx](attn_out)
    hidden = attn_out + ffn_out
```

Average attention weights across forks. Route only FFN layers. Proven at 7B in BTX and BAR.

| Property | Value |
|----------|-------|
| Granularity | Per-token FFN routing, shared attention |
| Total params | 1× attn (~30% of 150M) + N× FFN (~70% of 150M) |
| Active params (top-1) | ~150M (shared attn + 1 FFN) |
| Active params (top-2) | ~255M (shared attn + 2 FFNs) |
| Router params | N × d_model per layer |

With 4 experts + 1 anchor: ~45M shared attn + 5×105M FFN = **~570M total, ~150M active.**

**Pros:**
- **Proven.** BTX 47.9 beats Llama-2 13B. BAR 49.1 near full-retrain ceiling.
- **No hidden state coherence issue.** All tokens go through same attention → consistent KV cache. Only FFN varies.
- **Parcae loop safe.** MoDA depth KVs produced by shared attention → consistent across iterations. Only FFN output differs.
- **Parameter-efficient.** Attention (~30% of model) shared, not duplicated.
- **BAR anchor pattern.** Frozen anchor FFN preserves general capabilities.

**Cons:**
- **Attention averaging loses domain signal.** Code models learn different attention patterns than math models (e.g., bracket matching vs equation structure). Averaging these is lossy.
- **BAR found this acceptable at 7B** but at 150M, each attention head matters more. Less redundancy → more averaging loss.
- **Requires post-merge router training.** 5% of SFT data, frozen experts, 2 epochs (cheap but required).

**Attention averaging quality:**
- BAR: averaging diverged shared params (attention, embeddings) causes "little to no measurable performance loss"
- BTX: attention layers are "less domain-specialized" than FFN
- GenDistill: beam search for attention layer placement found only 7/28 attention layers matter for domain tasks
- Retrieval-Aware: only 2% of attention heads are retrieval-critical; rest are generic

Evidence strongly suggests attention averaging is safe even at small scale.

---

### Option D: Sticky Layer-Expert Routing (novel hybrid)

```python
# Per-layer expert selection, but CONSISTENT within a sequence
expert_assignments = []
for layer_idx in range(n_layers):
    scores = router[layer_idx](prompt_embedding)  # route on prompt, not per-token
    expert_assignments.append(top_k(scores, k=1))

# Forward pass uses assigned experts consistently
for layer_idx, expert_id in enumerate(expert_assignments):
    hidden = experts[expert_id].layers[layer_idx](hidden)
```

Each layer selects one expert based on prompt context. All tokens in the sequence pass through that same expert for that layer. Different layers can pick different experts.

| Property | Value |
|----------|-------|
| Granularity | Per-layer (consistent within sequence) |
| Total params | N × 150M |
| Active params | 150M (one expert per layer, but potentially different experts per layer) |
| Router params | N × d_model per layer |

**Pros:**
- **Full expert specialization.** Both attention and FFN from the chosen expert. No averaging loss.
- **Sequence coherence.** All tokens processed by same expert per layer → consistent KV cache.
- **Layer-specific blending.** Early layers might route to general expert (feature extraction), later layers to domain expert (task specialization). Emergent specialization by depth.
- **Parcae loop compatible.** Same expert selected for both iterations (prompt doesn't change).
- **Inference = just running one model** per layer. No MoE overhead per token.

**Cons:**
- **No intra-sequence mixing.** Can't blend code + math within one response at token level.
- **Coarser than Option A.** Per-layer routing, not per-token.
- **Router sees prompt only.** If the domain shifts mid-sequence, routing can't adapt.
- **Novel.** Not directly tested in literature. Closest: PHATGOOSE (per-module routing) and Hymba (per-layer SSM/attention selection).

**Interesting property:** This reduces to Option B if all layers select the same expert. If layers select different experts, it creates chimeric models — e.g., layers 0-1 from general expert (broad features), layer 2 GQA from code expert (code-specific attention), layer 3 from math expert (math post-processing). Whether this chimeric routing actually helps is an empirical question.

---

### Option E: LoRA-Expert Routing (lightest weight)

```python
# Base model (frozen) + domain LoRA adapters routed per-token
base_out = frozen_base.layers[layer_idx](hidden)
expert_id = router[layer_idx](hidden)
lora_delta = lora_experts[expert_id][layer_idx](hidden)  # rank-16 or rank-32
hidden = base_out + lora_delta
```

Don't fork full models. Train LoRA adapters per domain on the frozen base. Route which LoRA is applied per token.

| Property | Value |
|----------|-------|
| Granularity | Per-token, per-layer |
| Total params | 150M base + N × ~5M LoRA = ~170M |
| Active params | 155M (base + 1 LoRA) |
| Router params | N × d_model per layer |

With 4 domain LoRAs (rank-32): **~170M total, ~155M active.**

**Pros:**
- **Lightest.** Only ~13% parameter overhead for 4 experts.
- **No hidden state coherence issue.** Base model provides consistent representations. LoRA only adds domain deltas.
- **LoRAMoE, PHATGOOSE, X-LoRA all validate this.** Proven approach.
- **Trivial to add new domains.** Train one LoRA, add to router. No retraining needed.
- **Edge-friendly.** 170M total vs 600M for full-model experts.

**Cons:**
- **LoRA capacity limited.** Rank-32 LoRA = ~5M params per domain. Less specialization than full 150M fork.
- **Can't do deep mid-training.** LoRA adapts behavior, not knowledge. BAR's 50B-token mid-training creates deep knowledge shifts that LoRA can't match.
- **Base model quality ceiling.** If base model doesn't know code well, LoRA can't add code knowledge — only adjust behavior.

**When to use:** Post-training specialization (SFT, tool use, safety). Not for knowledge-intensive domains that need mid-training.

---

## Comparison Matrix

| | A: Per-Token | B: Per-Sequence | C: Shared Attn | D: Sticky Layer | E: LoRA-Expert |
|---|---|---|---|---|---|
| **Total params** | N×150M | N×150M | ~570M | N×150M | ~170M |
| **Active params** | 150-300M | 150M | 150-255M | 150M | 155M |
| **Attention** | Per-expert | Per-expert | Averaged | Per-expert | Shared base |
| **FFN** | Per-expert | Per-expert | Per-expert | Per-expert | Base + LoRA |
| **Routing grain** | Token×Layer | Sequence | Token×Layer (FFN) | Layer (prompt) | Token×Layer |
| **KV cache** | Problematic | Clean | Clean | Clean | Clean |
| **Parcae safe** | Risky | Yes | Yes | Yes | Yes |
| **Hidden coherence** | Risky | Perfect | Good | Good | Good |
| **Domain depth** | Full | Full | Full FFN, avg attn | Full | LoRA-limited |
| **Proven at scale** | No | BTM (weak) | **BTX/BAR (strong)** | No | LoRAMoE/PHATGOOSE |
| **Edge deployable** | Heavy | Heavy | Medium | Heavy | **Light** |
| **Modular upgrades** | Yes | Yes | Yes | Yes | **Easiest** |

---

## Challenges and Solutions

### Challenge 1: Hidden State Coherence Across Experts

**Problem:** When different tokens route to different experts (Options A, E), their hidden states evolve through different weight matrices. When token t+1 attends to token t, the key/value representations may be in different subspaces.

**Impact:** Attention scores become meaningless. Model produces incoherent output.

**Solutions by severity:**

| Solution | Approach | Applies to |
|----------|----------|-----------|
| Shared attention (Option C) | All tokens same attention weights → consistent KV | C only |
| Sticky routing (Option D) | Same expert per sequence per layer → consistent within sequence | D only |
| Shared embeddings + norms | Input/output representation space is shared even if internals differ | A, E |
| Expert interpolation | `output = α·expert_i(x) + (1-α)·expert_j(x)` with soft routing | A (modified) |
| Post-routing projection | Shared projection after expert output normalizes to common space | A, E |

**Recommendation:** For first implementation, use Option C or D. They avoid the problem architecturally rather than patching it.

### Challenge 2: Parcae Loop + MoE Interaction

**Problem:** Vidar uses shared layers iterated 2×. MoDA depth KVs from iteration 0 are consumed in iteration 1. If routing changes between iterations, KV representations are inconsistent.

**Impact:** MoDA attention attends to keys/values from a different expert's projection space.

**Solutions:**

| Solution | Description | Cost |
|----------|-------------|------|
| Consistent iteration routing | Same expert selected for both iterations (prompt-based routing) | Zero — natural for D, forced for A/C |
| Shared depth_kv_proj | Average depth_kv projection weights across experts. Domain-agnostic depth signal | Slight quality loss on cross-iteration info |
| Disable MoDA in MoE mode | Remove cross-iteration depth KVs. Loop position embeds carry iter info instead | Lose 2.1% quality gain from MoDA |
| Expert-aware depth KVs | depth_kv_proj is shared but reads from the routed expert's hidden state | Clean — hidden state already domain-specialized |

**Recommendation:** Option "Expert-aware depth KVs" — keep depth_kv_proj shared (averaged), but it reads from the routed expert's layer output. Cross-iteration info carries domain signal through the hidden state, not through diverged projection weights.

### Challenge 3: Router Training Data Requirements

**Problem:** Router must learn which expert handles which input. Requires labeled or self-labeled domain data covering all expert domains.

**Impact:** Bad routing = wrong expert selected = worse than dense model.

**Solutions:**

| Approach | Data needed | Quality |
|----------|-------------|---------|
| Domain-labeled data | Classification labels per sequence/token | Best but expensive to label |
| Self-labeled via loss | Route each training example to expert with lowest loss, use as label | Automatic, no manual labels |
| Balanced random + load balancing loss | Start random, let load balancing + gradient push toward specialization | Slowest convergence, may not specialize |
| Expert-generated routing labels | Each expert scores each training batch; expert with lowest perplexity "claims" that batch | Automatic, high quality, requires N forward passes |
| Distillation from per-sequence routing | Train per-sequence router (Option B, cheap) → distill into per-token router (Option A/C) | Progressive refinement |

**BAR approach (proven):** 5% stratified sample of SFT data from all domains. All non-router weights frozen. 2 epochs. LR 1e-4. Cheap and effective.

**Our approach:** Use the same data splits used for domain post-training as router training labels. Code data → label "code expert." Math data → label "math expert." General data → label "anchor."

### Challenge 4: Top-K Routing Decisions

**Problem:** Top-1 is cheapest (150M active). Top-2 doubles active compute (300M active) but may improve quality. Which K?

**Analysis:**

| K | Active params | Benefit | Cost |
|---|-------------|---------|------|
| Top-1 | 150M | Single expert, fastest inference | May miss cross-domain tokens |
| Top-2 | 300M | Blends two experts per token, better coverage | 2× FFN compute, need weighted combination |
| Top-1 + anchor always | 150M + anchor FFN | Domain expert + general fallback | ~255M active in Option C, 300M in others |

**BTX uses top-2.** BAR uses top-1 + frozen anchor (effectively top-2 with one always being the anchor). Laguna XS.2 uses top-k out of 256 experts.

**Recommendation for our scale:**

- **Option C (shared attn + routed FFN):** Top-1 + frozen anchor (BAR pattern). Domain FFN + anchor FFN per token. Active: ~255M.
- **Option D (sticky layer):** Top-1 per layer. Active: 150M. Clean, fast.
- **Option E (LoRA):** Top-1 per token. Active: 155M. Light.

**Soft routing alternative:** Instead of hard top-K, use soft weighted combination:
```python
weights = softmax(router(hidden))  # [N_experts]
output = sum(w_i * expert_i(hidden) for w_i, expert_i in zip(weights, experts))
```
All experts contribute, weighted by relevance. Active = N×150M (expensive) but can be approximated by top-2 soft weights.

### Challenge 5: Load Balancing and Dead Experts

**Problem:** Without explicit balancing, router may converge to always selecting one expert. Other experts become "dead" — never used, never updated.

**Impact:** Wastes parameters. MoE degenerates to dense model.

**Solutions:**

| Method | Description | Used by |
|--------|-------------|---------|
| Auxiliary load balancing loss | Penalize uneven expert utilization across batch | Switch Transformer, BTX, DSV4 |
| Expert capacity factor | Hard cap on tokens per expert per batch | Switch Transformer |
| Jitter noise | Add noise to router logits during training | GShard |
| Hash routing (token ID based) | Deterministic, guaranteed balance | DSV4 (first layers) |
| Normalized sigmoid router | Sqrt(softplus) activation with adaptive bias | Laguna XS.2 |

**Recommendation:** Auxiliary load balancing loss (standard, `aux_loss_coeff=0.01`) + router jitter noise during training. Simple and proven.

### Challenge 6: Shared vs Separate Embeddings and LM Head

**Problem:** Domain experts may develop different embedding spaces during post-training. Averaging may lose domain-specific token representations.

**Analysis:** BAR found embedding averaging causes "little to no measurable performance loss." Reason: embeddings are pretrained on general data and change minimally during domain post-training (especially if frozen during early post-training stages).

**However:** Tool use requires new special tokens (`<tool_call>`, `<tool_result>`). These tokens don't exist in the general pretrained embedding. BAR found unfreezing embeddings/LM head for tool use is critical (20.3 → 46.4).

**Solution:** 
- Average embeddings from all experts
- If any expert added new special tokens, keep those embeddings from that expert (no averaging for new tokens)
- LM head: average for shared vocab, keep expert-specific for new tokens

### Challenge 7: Inference Cost and Edge Deployment

**Problem:** Full-model MoE (Options A, B, D) = N×150M total params in memory. At 4 experts = 600M. In fp16 = 1.2GB. Fits Strix Halo easily but may not fit all edge targets.

| Target | Memory | 600M fp16 fits? | 170M fp16 fits? |
|--------|--------|----------------|----------------|
| Strix Halo (128GB) | Trivial | Yes | Yes |
| Laptop (16GB) | Yes | Yes | Yes |
| Phone (8GB) | Tight | Yes (with quantization) | Yes |
| Phone (4GB) | No | **Int4 needed (300MB)** | Yes |

**Solutions for constrained deployment:**
- **Option E (LoRA):** Only 170M total. Fits anywhere.
- **Expert offloading:** Keep anchor + 1 expert in VRAM/RAM, others on disk. Load on demand.
- **Quantization:** Int4 MoE = 600M × 0.5 bytes = 300MB. Fits on all targets.
- **Deploy single expert:** For known-domain applications, just ship the code expert. 150M = 300MB fp16.
- **Progressive loading:** Router runs on CPU, selects expert, loads expert weights from disk/flash storage.

---

## Recommended Path

### Phase 1: Validate MoE at 150M (sparse upcycling gate experiment)

Test if MoE structure helps at all at this scale:

1. Take trained Vidar checkpoint
2. Replicate each FFN 4× (identical init + small noise)
3. Add per-layer router (zero-init)
4. Continue training on mixed data with load balancing loss
5. Compare MoE Vidar (~160M total, ~77M active) vs dense Vidar (~47.5M)

**If MoE wins → proceed. If MoE loses → stop here (150M too small for specialization).**

### Phase 2: Train domain-specialized forks

1. Fork pretrained Vidar checkpoint 4× (code, STEM, tool use, general anchor)
2. Post-train each independently:
   - Code: mid-train on code subset of dolma-10b + SFT on code tasks
   - STEM: mid-train on math/science subset + SFT on math tasks
   - Tool use: SFT only on tool-use data (BAR pattern)
   - Anchor: standard post-training on general data (frozen after post-training)
3. Mix domain + general SFT data (BAR finding: domain-only destroys general capabilities)

### Phase 3: Compose into MoE

**Start with Option C** (shared attention + routed FFN) — proven, safe:
1. Average attention weights across all 4 forks
2. Keep each fork's FFN as a domain expert + anchor FFN
3. Train router on 5% stratified sample of all-domain SFT data (frozen experts)

**Then test Option D** (sticky layer routing) — novel, potentially better:
1. No averaging — full expert layers preserved
2. Per-layer prompt-based routing
3. Compare quality vs Option C

**Later explore Option E** (LoRA experts) — lightest, best for edge:
1. Train domain LoRAs on frozen base
2. Router selects LoRA per token
3. Compare quality/efficiency tradeoff

### Phase 4: Evaluate modular upgradability

The key differentiator of this approach:
1. Replace code expert with improved version → measure: code quality up, other domains unchanged?
2. Add new domain expert (e.g., medical) → measure: new capability without retraining?
3. Compare vs full retraining from scratch → measure: how much quality gap?

---

## Key Takeaways

1. **The pipeline works at 7B.** BTX beats Llama-2 13B. BAR is within 1.4 points of full retraining.
2. **Dense merging fails.** Must use MoE routing after significant divergence (BAR: 6.5/100 with averaging).
3. **Frozen anchor expert is essential.** Preserves base capabilities while domain experts specialize.
4. **Domain + general data mixing required.** Domain-only SFT destroys general capabilities (BAR: 25.8 → 36.8 with general data).
5. **Router training is cheap.** 5% of SFT data, frozen experts, 2 epochs.
6. **Full-model experts (Paradigm 2) avoid the capacity problem** that plagues FFN-only experts at 150M.
7. **Option C (shared attn + routed FFN) is the safest first step.** Proven at scale, Parcae-compatible.
8. **Option D (sticky layer routing) is the most interesting.** Novel, preserves full expert attention, but untested.
9. **Option E (LoRA routing) is the most practical for edge.** Lightest weight, easiest to upgrade.
10. **Sparse upcycling gate experiment (1 day) decides whether to pursue any of this.**
11. **Liquid AI validates the direction.** "The more narrow you design it, the better" — MoE of specialists is the answer for edge models.
