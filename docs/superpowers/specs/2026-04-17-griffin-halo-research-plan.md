---
title: "GRIFFIN-HALO Research Plan: Pareto Frontier Factor Sweep"
domain: architectures
type: spec
status: active
related:
  - docs/superpowers/specs/2026-04-16-jormungandr-halo-design.md
  - docs/superpowers/specs/2026-04-17-halo-prime-design.md
  - knowledge/architectures/architecture_mix_matrix.md
  - knowledge/training/training_antipatterns.md
tags: [%griffin-halo, %research-plan, %ablation, %pareto, %attnres, %dmc, %adaptive-depth]
---

# GRIFFIN-HALO Research Plan: Pareto Frontier Factor Sweep

## Goal

Map the **quality-vs-throughput Pareto frontier** for looped hybrid architectures on Strix Halo. Identify which components are worth their cost, which aren't, and build the best overall architecture from the evidence.

## Method

Orthogonal factor sweep (Approach B). Sequential runs on a single GPU. Reuse existing JORMUNGANDR-HALO and HALO-PRIME baselines. BabyLM 1-epoch as the fast evaluation protocol, top configs validated on WikiText-103.

---

## Existing Baselines (No Runs Needed)

All from prior experiments: BabyLM 1 epoch, ctx=256, Muon, compile+autokernel where applicable.

| Model | Type | Loss | tok/s | Params | Notes |
|-------|------|------|-------|--------|-------|
| JORMUNGANDR Bare | Looped, ShortConv, no extras | 6.028 | 33.7K | 99.2M | Local-only core |
| JORMUNGANDR +XSA | Looped + XSA | 5.973 | 33.7K | 99.2M | -0.9% from XSA |
| JORMUNGANDR +DMC | Looped + Depth MC | 5.879 | 33.7K | 99.2M | -2.5% from DMC |
| JORMUNGANDR +XSA+DMC | Looped + both | 5.770 | 33.7K | 99.2M | Additive: -4.3% |
| JORMUNGANDR Full | Looped + all (FiLM/VE/TTT/XSA/DC) | 5.770 | 33.5K | 103.5M | FiLM/VE/TTT add nothing at ctx=256 |
| HALO-PRIME Full | Looped, Mamba core, all extras | 5.650 | 28.5K | 103.5M | Best quality, no compile (Mamba breaks Dynamo) |

---

## Design Axes

### Axis 1: Loop Mixer — What provides global context in the core?

| Level | Block | Global Mixing | Compile? | Source |
|-------|-------|---------------|----------|--------|
| **A) Conv-only** | ShortConvBlock (kernel=3) | None (local) | Yes | JORMUNGANDR-HALO baseline |
| **B) Griffin+Conv** | GatedConv(384) ∥ GriffinRecurrence(384) | Recurrence (O(T)) | Yes | Tempest proven (2.98 loss) |
| **C) GQA+Conv** | GatedConv(512) + GQA(768, n_kv=2) | Attention (O(T²)) | Yes | LFM2 validated minimal pair |

LFM2's hardware-in-the-loop search found GatedConv+GQA is the minimal winning pair — SSMs/linear attention added no quality benefit in stacked architectures. However, our architecture is looped, where the core block runs 2-4x with no GQA inside. Griffin and GQA-in-loop are both untested in this regime.

### Axis 2: Depth Aggregation — How are loop iteration outputs combined?

| Level | Method | Granularity | Evidence |
|-------|--------|-------------|----------|
| **A) Last-only** | Use final iteration output | N/A | Standard baseline |
| **B) DMC** | Content-dependent gated mix (GRM) | Per-position, per-iteration | Proven -2.5% on JORMUNGANDR-HALO |
| **C) AttnRes-iter** | Softmax attention over iteration outputs | Per-layer (same for all positions) | AttnRes paper: 1.25x compute advantage |
| **D) DMC+AttnRes** | DMC aggregates per-position, AttnRes selects per-Coda-layer | Both granularities | Novel synergy — untested |

DMC and AttnRes are orthogonal:
- **DMC**: "Which iteration depth is best for THIS token position?" (per-position)
- **AttnRes**: "Which source block is best for THIS Coda layer?" (per-layer)

### Axis 3: Coda Residuals — How does Coda receive information?

| Level | Method | What Coda Layers Access |
|-------|--------|------------------------|
| **A) Standard** | `h = h + layer_output` | Only previous layer's accumulated state |
| **B) AttnRes cross-stage** | `h = AttnRes(w_l, [b_0, b_prelude, b_loop, partial])` | Embedding, Prelude output, loop output, Coda partial — each layer selects independently |

AttnRes in Coda replaces FiLM's function (communicating loop context to Coda) with a more principled mechanism. Cost: one learned query vector `w_l` (768 dims) + RMSNorm per source per layer. Negligible.

### Axis 4: Adaptive Depth — Do easy tokens skip iterations?

| Level | Method | Throughput Impact |
|-------|--------|-------------------|
| **A) Fixed T=4** | Poisson-sampled during training, fixed at inference | Baseline |
| **B) MoE Router** | top-k selection, capacity_factor=0.5. Phase 1 (all tokens, 2 iters) → Phase 2 (hard tokens only, 2 iters) | Est. 1.3-1.6x throughput |

Phase 2 uses Conv-only blocks (no recurrence) since global context is already established in Phase 1. Hard tokens are gathered into a compact tensor for smaller GEMMs.

### Axis 5: Dimensions — Core loop width

| Level | Core dim | Prelude/Coda dim | Adapters | Rationale |
|-------|----------|-----------------|----------|-----------|
| **A) d=768 uniform** | 768 | 768 | None | Simpler, no proj_down/up, each iteration costs 2.25x more |
| **B) d=512 heterogeneous** | 512 | 768 | proj_down + proj_up | Proven in JORMUNGANDR-HALO, cheaper iterations |

---

## The 10 Runs

Sequential. Each run is BabyLM, 1 epoch, ctx=256, Muon, compile+autokernel.
Config: batch=16, block=256, accum=4, lr=0.0012 (matching JORMUNGANDR-HALO ablation protocol).

Runs are **adaptive** — the winner of early runs determines the config for later runs. This maximizes information per run.

| Run | Mixer | Depth Agg | Coda Res | Adaptive | Dim | Primary Question |
|-----|-------|-----------|----------|----------|-----|------------------|
| **1** | Griffin+Conv | Last-only | Standard | Fixed | 768 | Griffin base: compiles? tok/s? loss? |
| **2** | GQA+Conv | Last-only | Standard | Fixed | 768 | GQA-in-loop vs Griffin (LFM2 hypothesis) |
| **3** | *winner(1,2)* | DMC | Standard | Fixed | 768 | Does DMC help with a global mixer? |
| **4** | *winner* | AttnRes-iter | Standard | Fixed | 768 | AttnRes vs DMC for depth aggregation |
| **5** | *winner* | DMC+AttnRes | Standard | Fixed | 768 | DMC+AttnRes synergy |
| **6** | *winner* | *best agg* | AttnRes | Fixed | 768 | AttnRes in Coda — additional value? |
| **7** | *winner* | *best agg* | *best res* | MoE 0.5 | 768 | Adaptive depth: throughput vs quality |
| **8** | *winner* | *best agg* | *best res* | Fixed | 512 | d=512 vs d=768 comparison |
| **9** | Best full config | | | | | Confirmation of Pareto-optimal config |
| **10** | ARGUS-PRIME B0 | — | — | — | 768 | Stacked non-looped baseline on BabyLM |

### Go/No-Go Gate: Runs 1-2

If BOTH Run 1 and Run 2 produce worse loss AND worse tok/s than JORMUNGANDR-HALO XSA+DMC (5.770 loss, 33.7K tok/s), stop and diagnose before proceeding. The base mixer must prove itself first.

### Winner Selection Rule

When selecting the "winner" between runs to carry forward:
- If one config dominates (better loss AND better/equal tok/s): pick it.
- If there's a tradeoff (one has better loss, other has better tok/s): pick the one with better loss, since throughput improvements come from later axes (adaptive depth).
- If results are within 1% on both metrics: pick the simpler config (fewer components).

### Run Dependencies

```
Runs 1, 2 → independent (parallel if 2 GPUs, sequential here)
Run 3 → depends on winner of 1 vs 2
Runs 4, 5 → depend on Run 3's mixer choice
Run 6 → depends on best aggregation from 3-5
Run 7 → depends on best residual from 6
Run 8 → depends on best config from 3-7
Run 9 → depends on all prior
Run 10 → independent (can run anytime)
```

---

## Fixed Components (Not Variables)

These are always on — individually justified by prior evidence or zero-cost:

| Component | Justification |
|-----------|---------------|
| XSA (Exclusive Self Attention) | Zero params, zero compute, proven -0.9% |
| Parcae injection | Loop stability guarantee, proven |
| Per-iteration RMSNorm | Prevents state growth (learned from HALO-PRIME NaN debugging) |
| TTT on last Coda GQA | Proven benefit at ctx>=512, low cost |
| Muon optimizer | Proven 2x token efficiency |
| 1 block per iteration | Griffin/GQA provide global context; loop provides depth |
| SwiGLU FFN (2048 inner at d=768) | Tensile-aligned, standard |

## Cut Components (Not Tested)

| Component | Reason |
|-----------|--------|
| FiLM conditioning | Zero benefit at ctx=256, marginal at 1024. AttnRes subsumes its function. |
| Value Embeddings | Never isolated in ablation. 3.2M params for unknown gain. |
| Momentum residuals | Inherited from Tempest, never ablated independently. Adds complexity. |
| Multi-step TTT | Causes NaN from scratch (proven anti-pattern). |
| Extra TTT locations | Test separately after the main sweep if warranted. |

---

## Architecture Details Per Config

### Run 1: Griffin+Conv Base (d=768)

```
Prelude (d=768): ShortConvBlock(768, 512, 2816) + GQABlock(768, n_kv=4)

Core Loop (d=768, 1 block × 4 iterations):
  GriffinConvBlock:
    RMSNorm → (GatedConv(384) ∥ GriffinRecurrence(384)) → concat → proj(768) → SwiGLU(2048)
  Per-iter: Parcae re-inject → block → iter_norm

Coda (d=768): ShortConvBlock + CodaGQABlock(XSA) + ShortConvBlock + CodaGQABlock(XSA+TTT)

Output: RMSNorm → tied LM head
```

Griffin from `tempest.py`: fully vectorized chunked scan, multi-scale decay bias spectrum, zero Python loops. torch.compile compatible.

### Run 2: GQA+Conv Base (d=768, LFM2-inspired)

```
Core Loop (d=768, 1 block × 4 iterations):
  GQAConvBlock:
    RMSNorm → GatedConv(512) → residual
    RMSNorm → GQA(768, n_kv=2, QK-Norm) → residual
    RMSNorm → SwiGLU(2048) → residual
```

GQA-in-loop uses only 2 KV heads (lightweight, ~2M extra params for QKV projections). Needs RoPE frequencies passed into the loop. SDPA compiles natively. Cost: O(T^2) per iteration, acceptable at ctx=256, monitor at ctx=1024.

### Runs 3-5: Depth Aggregation Variants

**DMC (Run 3):** Existing `DepthMemoryCache` module from jormungandr_halo.py. Linear(768, 64) projection, content-dependent gating, per-position weighted mix.

**AttnRes-iter (Run 4):** New module. Each Coda layer gets a learned query `w_l` that attends over iteration outputs via softmax:
```python
class IterationAttnRes(nn.Module):
    def __init__(self, d_model, n_consumers):
        self.queries = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(n_consumers)])
        self.norm = RMSNorm(d_model)
    
    def forward(self, iteration_states, consumer_idx):
        V = torch.stack(iteration_states)          # (N_iters, B, T, d)
        K = self.norm(V)
        q = self.queries[consumer_idx]
        logits = torch.einsum('d, n b t d -> n b t', q, K)
        weights = logits.softmax(dim=0)             # (N_iters, B, T)
        return torch.einsum('n b t, n b t d -> b t d', weights, V)
```

Zero-init queries → starts as uniform average (paper recommendation).

**DMC+AttnRes (Run 5):** DMC aggregates iteration outputs per-position into `b_loop`. AttnRes in Coda selects from `[b_0, b_prelude, b_loop, partial]` per-layer.

### Run 6: Coda AttnRes

Replace standard residuals in Coda with Block AttnRes. Sources:
- `b_0`: token embedding
- `b_prelude`: after Prelude (GQA contextual representation)
- `b_loop`: DMC-aggregated loop output (or AttnRes-aggregated, depending on Run 3-5 winner)
- `partial`: accumulated Coda output so far

Each Coda layer attends to these 4 sources with its own learned query, selecting the most useful source for its specific function.

### Run 7: Adaptive Depth

MoE-style router after Phase 1 (2 iterations):
```python
class DepthRouter(nn.Module):
    def __init__(self, d_model, capacity_factor=0.5):
        self.gate = nn.Linear(d_model, 1)
        self.capacity_factor = capacity_factor
        nn.init.zeros_(self.gate.weight)
    
    def forward(self, h):
        scores = self.gate(h).squeeze(-1)
        k = int(h.shape[1] * self.capacity_factor)
        top_vals, top_idx = scores.topk(k, dim=1)
        return top_idx, torch.sigmoid(top_vals), scores
```

Phase 2 uses Conv+SwiGLU blocks (no recurrence — global context already established). Hard tokens gathered into compact tensor `(B, k, d)` for smaller GEMMs. Auxiliary load balancing loss prevents router collapse.

Curriculum: capacity_factor starts at 1.0 (all tokens → Phase 2) for first 20% of training, then drops to 0.5.

### Run 8: d=512 Comparison

Same as Run 9's best config but with d_core=512, d_prelude=d_coda=768, proj_down/proj_up adapters. Tests whether cheaper iterations (2.25x less FLOP per iter) outweigh the adapter overhead and dimension mismatch.

---

## Metrics Per Run

| Metric | Source | Threshold |
|--------|--------|-----------|
| Actual per-token CE loss | reported_loss / accum_steps | Lower is better |
| tok/s | train_log.jsonl | Higher is better |
| MFU | train_log.jsonl | Informational |
| Peak memory (GB) | train_log.jsonl | Must fit in 128GB |
| Compile success | Did torch.compile work? | Must pass for throughput configs |
| Max grad norm | train_log.jsonl | < 10.0 (stability) |
| NaN/Inf count | train_log.jsonl | Must be 0 |
| State norm ratio | StateNormMonitor | < 1.5 (loop stability) |

---

## Validation Phase

Top 2-3 configs from the BabyLM sweep → WikiText-103 (ctx=1024, 1 epoch, lr=0.0004, from BabyLM checkpoint):

- Tests quality transfer to larger data and longer context
- Tests TTT crossover at ctx>=512
- Tests throughput at ctx=1024 (different compute profile)
- Directly comparable to JORMUNGANDR-HALO WikiText results (XSA+DC: 6.852, Full: 6.805)

---

## Expected Outcome

A Pareto frontier plot with loss on Y-axis and tok/s on X-axis, showing every tested config plus the existing baselines. The "winner" is the config furthest into the upper-right quadrant (best quality at highest throughput), or the set of configs forming the efficient frontier if the tradeoff is smooth.

```
Quality (loss ↓ better)
    │
5.6 │              * HALO-PRIME (Mamba, 28.5K)
    │         ?  
5.7 │    ? ? ?    * JORMUNGANDR XSA+DMC (33.7K)
    │       ?         ?
5.8 │                      ?
    │
5.9 │                              ?
    │
6.0 │  * JORMUNGANDR Bare (33.7K)        * ARGUS-PRIME (18K)
    └──────────────────────────────────────────────
        18K   28K   34K   42K   53K   67K  tok/s →
        
    * = existing data point
    ? = new GRIFFIN-HALO config to test
```

---

## Literature Informing This Plan

| Paper | Key Insight Applied |
|-------|-------------------|
| Attention Residuals (Kimi, 2603.15031) | AttnRes replaces FiLM + provides cross-stage depth selection. Zero-init queries. Softmax > sigmoid. |
| Memory Caching GRM (Behrouz, 2602.24281) | DMC for per-position iteration depth selection. Proven -2.5%. |
| Exclusive Self Attention (Zhai, 2603.09078) | XSA removes self-value projection. Zero cost, proven -0.9%. |
| LFM2 (Liquid AI, 2511.23404) | GatedConv+GQA is the minimal pair. SSMs don't help in stacked archs. Motivates GQA-in-loop option. |
| MoDA (Zhu, 2603.15619) | Cross-depth KV sharing validates AttnRes approach. Hardware-efficient kernel available. |
| STILL (Meng, 2602.02180) | Self-Saliency Score for token routing. Could inform router design. |
| HySparse (Gao, 2602.03560) | Oracle attention layer identifies important tokens. Prelude GQA as oracle. |
| Stateful Token Reduction (Jiang, 2603.00198) | Progressive reduction schedule. Validates Phase 1 (all) → Phase 2 (hard) split. |
| MaBERT (Kim, 2603.03001) | Padding-safe SSM masking. Implement for Griffin recurrence training correctness. |

---

## Implementation Order

1. Build `GriffinConvBlock` (reuse Griffin from tempest.py + GatedConv from amadeus.py)
2. Build `GQAConvBlock` (reuse GQA from argus_prime.py + GatedConv)
3. Build `IterationAttnRes` module
4. Build `CodaAttnRes` module
5. Build `DepthRouter` + Phase 2 Conv block
6. Build `GriffinHaloBase` model with config flags for all axes
7. Build variant classes for each of the 10 runs
8. Build `GriffinHaloMini` for smoke testing
9. Training scripts for sequential BabyLM runs
10. Analysis script for Pareto frontier plotting

---

## Files to Create

| File | Description |
|------|-------------|
| `models/griffin_halo.py` | Main model with all axis configurations |
| `scripts/run_griffin_halo_sweep.sh` | Sequential BabyLM ablation runner |
| `scripts/run_griffin_halo_validate.sh` | WikiText-103 validation for top configs |
| `scripts/plot_pareto.py` | Pareto frontier visualization |
