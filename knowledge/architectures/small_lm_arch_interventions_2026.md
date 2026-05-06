---
title: "Small-LM Architectural Interventions (2025-2026) — Consolidated Playbook"
domain: architectures
type: reference
status: active
tags: [small-lm, architecture, qk-norm, value-residuals, layernorm-scaling, per-head-gating, seednorm, attention-sink]
related:
  - parcae_stable_looped_models.md
  - moda_cross_layer_2026.md
  - reliable_small_lm_insights.md
  - ../training/imu1_recipe_2026.md
  - ../../docs/research/broad-research-synthesis-2026-05-06.md
---

# Small-LM Architectural Interventions — Consolidated Playbook

A set of relatively-cheap architecture tweaks that have emerged as consensus
across multiple 2025-2026 small-LM releases (IMU-1, SmolLM3, Nanbeige4,
OLMo 2, LFM2, Granite-4). Each is small in isolation; cumulatively they
account for much of the "sample efficiency" gap between 2024-era and
2026-era small LMs.

## Quick reference table

| Intervention | Cost | Gain (at ~430M) | Adopted by | Fit for Odin? |
|--------------|-----:|---------------:|------------|:-----:|
| QK-Norm attention | trivial | stability | IMU-1, SmolLM3 | ✓ already done |
| Value residual connections | +0.1% params | +0.3-0.5% loss | IMU-1 | **Yes, port** |
| LayerNorm scaling | negligible | +0.2-0.5% loss | IMU-1 | **Yes, port** |
| Per-head gated attention | +0.5% params | +0.3-0.5% loss | IMU-1 | **Yes, port** |
| No WD on embeddings / 1D params | negligible | +0.1-0.3% loss | OLMo 2, SmolLM3, IMU-1 | **Yes, easy** |
| Intra-document attention masking | ~0 at train | +0.05-0.2% loss, stability | Llama-3, SmolLM3 | **Yes, easy** |
| NoPE-every-4th-layer | negligible | long context +several% | SmolLM3 | Alternative to HyPE |
| Tied embeddings | saves params | neutral | Everyone | ✓ already done |
| GQA with 4 groups | cheaper decode | matches full attn | Everyone | ✓ already done |
| SeeDNorm (dynamic RMSNorm γ) | negligible | +0.2% | Paper-only | Worth ablating |
| Grouped-query attention GQA=4 | cheaper KV | matches MHA | Everyone | ✓ already done |

## Interventions in detail

### QK-Norm attention (Henry et al., 2020)

**What:** Apply RMSNorm to Q and K *before* the dot product.

**Why:** Keeps attention logits in a bounded range, preventing softmax
saturation and associated training instability. Lowers activation kurtosis.

**In our codebase:** Already implemented in `NoPECodaAttention` (see
`models/components/attention.py`). OdinFlat and OdinHalo both use it by
default. ✓ done.

### Value residual connections (Zhou et al., 2024)

**What:** Add a learned residual on the attention's value path:
```python
v_res = v + alpha * v_prev    # where v_prev is from previous attn layer
```

**Why:** Improves gradient flow, especially in deeper stacks. Prior layers'
value information stays accessible in deeper layers.

**In our codebase:** Not implemented. Port needed in
`models/components/attention.py::NoPECodaAttention.forward`.

**Implementation sketch:**

```python
class NoPECodaAttention(Attention):
    def __init__(self, ...):
        super().__init__(...)
        self.v_res_scale = nn.Parameter(torch.zeros(1))  # init 0 → pure baseline

    def forward(self, x, v_prev=None, ...):
        ...
        v = self.wv(x).view(...)
        if v_prev is not None:
            v = v + self.v_res_scale * v_prev
        ...
        return out, v  # return v for next layer's v_prev
```

The model must thread `v_prev` between consecutive attention layers (only
GQA layers in our case — positions 6 and 13 in OdinFlat).

**Priority:** High. Cheap to implement, validated at adjacent scale.

### LayerNorm scaling

**What:** Scale each LayerNorm / RMSNorm's learned γ by a depth-dependent
factor:
```python
gamma_i = gamma_i_base / sqrt(layer_idx + 1)
```

(Exact formula varies across sources; IMU-1 uses depth-scaled γ during
initialization.)

**Why:** Deeper layers get proportionally smaller initial activation
magnitudes, preventing the "runaway amplification" common in deep stacks.

**Implementation:** Single-line change in `_init_weights` in each model.
Multiply each RMSNorm weight by `1/sqrt(i+1)` during init.

**Priority:** High. 5-minute change.

### Per-head gated attention (Qiu et al., 2025)

**What:** Gate each attention head's output by a learned sigmoid:
```python
attn_out = sigmoid(per_head_gate) * attn_out  # per_head_gate: [n_heads, 1, 1]
```

**Why:**
1. Improves expressivity (each head can be dialed down).
2. **Mitigates attention sinks** — the pathological early tokens that absorb
   most of the attention mass.

**In our codebase:** Not implemented. Port into both `Attention` and
`NoPECodaAttention` classes.

**Implementation:**

```python
class NoPECodaAttention(Attention):
    def __init__(self, ..., n_heads):
        super().__init__(...)
        self.head_gate = nn.Parameter(torch.ones(n_heads))  # init open

    def forward(self, x, ...):
        ...
        attn_out = ...  # [B, n_heads, T, head_dim]
        attn_out = attn_out * torch.sigmoid(self.head_gate).view(1, -1, 1, 1)
        ...
```

**Priority:** High. Simple, additive, validated.

### No weight decay on embeddings and 1D parameters

**What:** Split parameter groups in the optimizer:
- 2D weight matrices → WD = 0.1 (standard)
- Embeddings, LN gains, biases, lm_head → WD = 0.0

**Why:** OLMo 2 found that applying WD to embeddings causes drift in embedding
norms over training. Without WD, they stabilize at healthier values.

**In our codebase:** Currently we apply uniform WD to all params via fused
AdamW. Fix in `halo_training/trainer.py` and `scripts/train_ddp.py`:

```python
no_decay_params = [p for n, p in model.named_parameters()
                   if "embed" in n or "norm" in n or p.dim() < 2]
decay_params = [p for n, p in model.named_parameters() if p not in no_decay_params]
optimizer = AdamW([
    {"params": decay_params, "weight_decay": 0.1},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=lr, betas=(0.9, 0.95), fused=True)
```

**Priority:** Highest. 5-minute change. Free gain.

### Intra-document attention masking

**What:** Multiple documents get packed into a single training sequence
(our block_size=512 usually contains 2-3 documents separated by `<|endoftext|>`).
Standard causal attention lets tokens from document A attend to tokens from
document B, creating spurious cross-document dependencies.

**Fix:** Build a per-token `doc_id` tensor; in attention, mask cross-doc
positions:
```python
attn_mask = (doc_id[:, None] == doc_id[None, :]) & causal_mask
```

**Why:** Llama 3, SmolLM3, and OLMo 2 all use it and report cleaner gradient
dynamics + slight loss improvement, especially at longer context.

**In our codebase:** Not implemented. Needs:
1. Dataloader: build `doc_id` from EOS positions per sample
2. Model forward: derive attention mask from `doc_id`
3. Apply in `Attention`, `NoPECodaAttention`, `HyPEShortConvBlock` (for conv?
   conv is causal 1D and doesn't directly see tokens from other docs through
   kernel windows at kernel=3, so may not need)

**Priority:** High. 1-hour implementation. Universally useful.

### NoPE-every-4th-layer (Yang et al., arXiv:2501.18795, 2025)

**What:** Selectively remove rotary position embeddings from every 4th layer,
keeping RoPE on the other 3/4.

**Why:** Improves long-context performance without hurting short-context
capabilities. The NoPE layers let the model learn content-only attention
patterns that generalize better beyond training length.

**Relevance to Odin:** Our current HyPE design does something similar but
different:
- Our HyPE: RoPE on conv gate, NoPE on GQA attention (all GQA layers have NoPE)
- SmolLM3: RoPE on all attention, NoPE on every 4th layer

Both approaches separate positional and content handling, but at different
granularities. Not urgent — we already have a positional strategy that works.

**Priority:** Low. Investigate only if we rebuild the model family.

### SeeDNorm (arXiv:2510.22777, Oct 2025)

**What:** Input-dependent RMSNorm γ. Standard RMSNorm uses a static learned γ;
SeeDNorm computes γ dynamically as a function of the input norm.

**Why:** RMSNorm discards input-norm information in the forward pass.
Input-dependent γ preserves it, handling distributional shifts better
(especially zero-shot scenarios).

**Results:** Consistent wins over RMSNorm / LayerNorm / DyT across LLM
pretraining and computer vision tasks.

**In our codebase:** Drop-in replacement in `models/_components.py::RMSNorm`.

**Priority:** Medium. Cheap ablation, small expected gain.

## Recommended adoption order for Odin

**Sprint 1 (foundation — 2-4 hours):**
1. No WD on embeddings + 1D params (highest ROI, 5 min)
2. Intra-document attention masking (1 hour, universal)
3. LayerNorm scaling (5 min)

**Sprint 2 (substantive — 4-8 hours):**
4. Value residual connections in GQA layers (2-3 hours)
5. Per-head gated attention (2-3 hours)

**Sprint 3 (optional/experimental — 4-8 hours):**
6. SeeDNorm ablation
7. NoPE-every-4th variant (as OdinFlat-v2 alongside current HyPE)

## Combined expected gain

From IMU-1's ablation tables, the cumulative loss improvement from all
architectural interventions (QK-norm + per-head gating + value residuals + LN
scaling, each ~0.5-1%) was:
- Baseline: loss X
- + All four interventions: X - 2 to 3%

We already have QK-Norm, so expected marginal gain from adding the other three:
**~1.5-2% relative loss improvement** at OdinFlat scale, with <1% throughput cost.

## Interactions to watch

- **Value residuals + our Parcae looping** (OdinHalo): v_prev from previous
  iteration is different from v_prev from previous layer. Need to specify
  which meaning. Likely: per-iteration, resetting at loop entry.
- **Per-head gating + MoDA**: The new cross-layer MoDA also attends cross-
  layer; gated outputs might interact with depth KVs. Ablate separately before
  combining.
- **Intra-doc masking + NoPE attention**: Our NoPECodaAttention is already
  mask-aware via the causal path; adding intra-doc mask is an intersection,
  not a conflict.

## See also

- `knowledge/training/imu1_recipe_2026.md` — the IMU-1 full recipe that
  validates most of these interventions empirically
- `knowledge/architectures/moda_cross_layer_2026.md` — the other new
  architectural component (cross-layer depth KV attention)
- `knowledge/architectures/parcae_stable_looped_models.md` — our looped
  model baseline
- `docs/research/broad-research-synthesis-2026-05-06.md` Part 6 — full
  architecture landscape with alternatives (HLA, InfoMamba, etc.)
