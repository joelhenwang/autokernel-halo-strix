---
title: "MAESTRO-PRIMA"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %maestro-prima]
---

# MAESTRO-PRIMA

**Movement I: Component Dynamics — The Conductor Controls the Volume**

*The conductor raises a hand. The strings swell. The brass falls silent. Each section plays at the volume the conductor demands.*

## Hypothesis

AMADEUS (the composer) writes a fixed score — every layer processes every input identically. MAESTRO-PRIMA (the conductor) adds a tiny network (~137K params, 0.06% of model) that reads the input and decides HOW LOUD each instrument should play at each layer. Gated Conv, Mamba-3, and SwiGLU FFN are the three sections of the orchestra. The conductor controls their volumes dynamically.

**Base:** AMADEUS architecture (see AMADEUS.md). Gated Conv + Mamba-3 SISO + SwiGLU FFN, 16 layers, d=1024, ~242M params.

---

## The Conductor

```python
class Conductor(nn.Module):
    def __init__(self, d_model=1024, d_cond=128, n_layers=16, n_signals=3):
        self.input_proj = nn.Linear(d_model, d_cond)         # read the sequence
        self.score_proj = nn.Linear(d_cond, n_layers * n_signals)  # produce the score
    
    def forward(self, embed):  # embed: (B, T, d)
        summary = embed.mean(dim=1)                           # (B, d) — sequence summary
        cond = F.relu(self.input_proj(summary))               # (B, 128)
        score = torch.sigmoid(self.score_proj(cond)) * 2.0    # (B, 48) → range [0, 2]
        return score.view(-1, 16, 3)                          # (B, 16, 3): conv, mamba, ffn
```

**Signals per layer (3):**

| Signal | Range | Controls | Init |
|--------|-------|----------|------|
| conv_scale | [0, 2] | Gated Conv output volume | ~1.0 (sigmoid(0)×2) |
| mamba_scale | [0, 2] | Mamba-3 SISO output volume | ~1.0 |
| ffn_scale | [0, 2] | SwiGLU FFN output volume | ~1.0 |

**Init at 1.0 = identity.** Untrained conductor = standard AMADEUS. The conductor can only HELP, never hurt at init.

## Conducted Layer

```python
def conducted_forward(self, h, score_i):
    # score_i: (B, 3) — conv_scale, mamba_scale, ffn_scale
    h_norm = rmsnorm(h)
    
    # Parallel hybrid with conductor scaling
    conv_out = self.gated_conv(h_norm) * score_i[:, 0:1].unsqueeze(1)   # (B, T, 640) × (B, 1, 1)
    mamba_out = self.mamba3(h_norm) * score_i[:, 1:2].unsqueeze(1)      # (B, T, 384) × (B, 1, 1)
    h = h + self.outproj(torch.cat([conv_out, mamba_out], dim=-1))
    
    # FFN with conductor scaling
    h = h + self.ffn(rmsnorm(h)) * score_i[:, 2:3].unsqueeze(1)        # (B, T, d) × (B, 1, 1)
    return h
```

## Parameter Count

| Component | Params |
|-----------|--------|
| AMADEUS base | 241.6M |
| Conductor input_proj (1024→128) | 131K |
| Conductor score_proj (128→48) | 6K |
| **Conductor total** | **137K** |
| **GRAND TOTAL** | **~241.7M** |

## Training

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (50%) | 7.5 min | Backbone only. Conductor OFF (scores fixed at 1.0). | Learn language |
| 2 (30%) | 4.5 min | + Conductor (backbone FROZEN, train conductor only) | Conductor learns to read the orchestra |
| 3 (20%) | 3 min | Everything trainable. | Joint optimization |

## What The Conductor Learns

| Input type | Expected conductor behavior |
|-----------|---------------------------|
| Factual text ("The capital of France...") | Boost Mamba (long-range context matters) |
| Formulaic text ("Dear Sir/Madam...") | Boost Conv (local patterns suffice) |
| Reasoning ("If X then Y because...") | Boost FFN (transformation/reasoning) |
| Mixed content | Balanced scores |

## Risks & Success Criteria

| Risk | Mitigation |
|------|------------|
| Conductor collapses to uniform signals | Init at sigmoid(0)×2=1.0. Monitor score variance. |
| Mean-pool too coarse | For PRIMA, accept this. VIVACE fixes it. |

**Success:** Score variance > 0.1 across input types. Loss improvement > 0.3% vs AMADEUS.

## Implementation Roadmap

1. Build AMADEUS base (see AMADEUS.md roadmap)
2. Add Conductor class (input_proj + score_proj, ~10 lines)
3. Modify layer forward to accept score_i (multiply outputs)
4. Phase 1: train base
5. Phase 2: freeze base, train conductor
6. Phase 3: unfreeze all
7. Visualize: score heatmap (16 layers × 3 signals) per input type

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Built on AMADEUS — Same Kernel Reuse
This architecture extends AMADEUS (gated conv + Mamba-3 SISO + SwiGLU). All AMADEUS optimizations apply:
- **fused_residual_add_rmsnorm** (6.6x), **silu_gate_mul** (1.6x), **cross_entropy** (1.8x)
- Apply via `autokernel.optimize(model, training=True)`
- Mamba-3 scan: **use chunked linear recurrence** (chunk_size=64). Reference: `models/amadeus.py:selective_scan_chunked`
- Do NOT use sequential loops or `torch.associative_scan` — both are 5x slower

### Conductor/Extension Overhead
The conductor/PLE additions are element-wise ops (linear projections, sigmoid, multiply) — negligible overhead (<1% of forward time). The AMADEUS base dominates compute.

### Throughput Baseline
- **AMADEUS measured:** 6,400 tok/s, 15.9% MFU, 12.7 GB memory (eager, 243M params)
- This variant adds <5M params → expect similar throughput
- **Token budget:** 15 min = ~5.8M | 45 min = ~17M | 120 min = ~46M

---

## Possible Optimizations & Throughput Estimate

**Baseline (estimated):** ~6,200 tok/s eager (15% MFU) — AMADEUS base + 137K conductor

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +60% MFU (same as AMADEUS base) | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| `causal-conv1d` in GatedConv | 10x conv speedup | Available |
| `mamba-ssm` selective_scan_fn | 5.6x scan speedup (0.32ms) | Available |
| Conductor overhead minimal | FiLM-style: 137K params, <0.5% of total compute | By design |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~12,000 tok/s (29% MFU)
**Tokens in 45 min:** ~32.4M (2.0 BabyLM epochs)
**Ranking:** #9 of 22 architectures
