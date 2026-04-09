# MAESTRO-FORTE

**Movement II: Emotional Arc — Crescendo and Diminuendo Across Depth**

*The conductor doesn't just control volume — they control ENERGY. A crescendo builds. A diminuendo fades. The emotional arc shapes the entire performance.*

## Hypothesis

PRIMA controls which instruments play louder. FORTE adds control over the RESIDUAL STREAM — the lifeblood that carries information between layers. The conductor creates crescendos (amplify residual = "let this through!") and diminuendos (attenuate residual = "suppress this noise").

**Extends:** MAESTRO-PRIMA. Same base (AMADEUS), same component dynamics, PLUS residual gating.

---

## The Conductor (Extended)

```python
class ConductorForte(nn.Module):
    def __init__(self, d_model=1024, d_cond=128, n_layers=16):
        self.input_proj = nn.Linear(d_model, d_cond)
        self.score_proj = nn.Linear(d_cond, n_layers * 4)  # 3 volumes + 1 residual gate
    
    def forward(self, embed):
        summary = embed.mean(dim=1)
        cond = F.relu(self.input_proj(summary))
        raw_score = self.score_proj(cond).view(-1, 16, 4)
        
        volumes = torch.sigmoid(raw_score[:, :, :3]) * 2.0   # [0, 2] — conv, mamba, ffn
        gates = torch.sigmoid(raw_score[:, :, 3:4])           # [0, 1] — residual gate
        return torch.cat([volumes, gates], dim=-1)             # (B, 16, 4)
```

**New signal: residual_gate (per layer)**

| Signal | Range | Musical Analog |
|--------|-------|---------------|
| residual_gate → 1.0 | Full pass-through | **Fortissimo** — this layer's contribution MATTERS |
| residual_gate → 0.5 | Half contribution | **Mezzo** — moderate |
| residual_gate → 0.0 | Suppress layer output | **Pianissimo** — this layer is noise, skip it |

## Conducted Layer (with Emotional Arc)

```python
def conducted_forward(self, h, score_i):
    h_prev = h  # save for gated residual
    h_norm = rmsnorm(h)
    
    # Component dynamics (same as PRIMA)
    conv_out = self.gated_conv(h_norm) * score_i[:, 0:1].unsqueeze(1)
    mamba_out = self.mamba3(h_norm) * score_i[:, 1:2].unsqueeze(1)
    mixer_out = self.outproj(torch.cat([conv_out, mamba_out], dim=-1))
    ffn_out = self.ffn(rmsnorm(h + mixer_out)) * score_i[:, 2:3].unsqueeze(1)
    
    # Emotional arc: gated residual
    h_new = h + mixer_out + ffn_out
    gate = score_i[:, 3:4].unsqueeze(1)  # (B, 1, 1)
    h = gate * h_new + (1 - gate) * h_prev  # crescendo/diminuendo
    return h
```

## Parameter Count

| Component | Params |
|-----------|--------|
| AMADEUS base | 241.6M |
| Conductor input_proj (1024→128) | 131K |
| Conductor score_proj (128→64) | 8K |
| **Conductor total** | **139K** |
| **GRAND TOTAL** | **~241.7M** |

## What The Conductor Learns (Emotional Arc)

```
Layer:  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
Gate:  0.9 0.8 0.7 0.9 0.6 0.8 0.9 1.0 0.8 0.5 0.7 0.9 0.6 0.8 0.9 1.0
       ─── building ──→ ★peak★ ─── middle ──→ ─── building ──→ ★finale★
```

The conductor learns that some layers are CRITICAL (gate→1) and others are redundant (gate→0.5). This is like learned layer pruning but SOFT and INPUT-DEPENDENT.

## Training, Risks, Success Criteria

Same as PRIMA (3-phase training, same risks). Additional success criterion: residual gates show variance > 0.15 across layers (the arc is NOT flat).

## Implementation Roadmap

1. Start from MAESTRO-PRIMA implementation
2. Add 4th signal (residual_gate) to score_proj output
3. Modify conducted_forward to use gated residual
4. Visualize: residual gate pattern across layers (should show "arc" shape)

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
