# MAESTRO-FINALE

**Movement IV: The Full Symphony — Component Dynamics + Emotional Arc + Evolving Interpretation**

*The finale. Every technique the conductor has mastered — volume control, emotional dynamics, evolving interpretation — combined into one transcendent performance. This is the conductor at their peak.*

## Hypothesis

FINALE combines all three movements: the conductor controls instrument volumes (PRIMA), shapes the emotional arc via residual gating (FORTE), and evolves its interpretation mid-performance via a recurrent state (VIVACE). A tiny 160K-param meta-controller that orchestrates a 242M-param model.

**Base:** AMADEUS architecture (see AMADEUS.md). **Conductor:** All of PRIMA + FORTE + VIVACE combined.

---

## The Full Conductor

```python
class ConductorFinale(nn.Module):
    def __init__(self, d_model=1024, d_cond=64, n_groups=4):
        self.initial_proj = nn.Linear(d_model, d_cond)       # first reading
        self.orchestra_proj = nn.Linear(d_model, d_cond)     # re-read projection
        self.gru = nn.GRUCell(d_cond, d_cond)                # evolving state
        self.score_projs = nn.ModuleList([
            nn.Linear(d_cond, 4 * 4) for _ in range(n_groups)  # 4 layers × 4 signals
        ])
    
    def forward(self, embed, get_checkpoint_fn):
        state = F.relu(self.initial_proj(embed.mean(dim=1)))  # (B, 64)
        all_scores = []
        
        for group_idx in range(4):
            if group_idx > 0:
                checkpoint = get_checkpoint_fn(group_idx - 1)  # (B, T, d)
                reading = F.relu(self.orchestra_proj(checkpoint.mean(dim=1)))
                state = self.gru(reading, state)  # EVOLVE interpretation
            
            raw = self.score_projs[group_idx](state).view(-1, 4, 4)  # (B, 4, 4)
            volumes = torch.sigmoid(raw[:, :, :3]) * 2.0    # [0,2] — conv, mamba, ffn
            gates = torch.sigmoid(raw[:, :, 3:4])            # [0,1] — residual gate
            all_scores.append(torch.cat([volumes, gates], dim=-1))
        
        return torch.cat(all_scores, dim=1)  # (B, 16, 4)
```

## The 4 Signals

| Signal | Range | From | Musical Analog |
|--------|-------|------|---------------|
| conv_scale | [0, 2] | PRIMA | Strings volume |
| mamba_scale | [0, 2] | PRIMA | Winds volume |
| ffn_scale | [0, 2] | PRIMA | Brass volume |
| residual_gate | [0, 1] | FORTE | Crescendo / Diminuendo |
| *conductor GRU* | *hidden state* | VIVACE | *The conductor's evolving feel* |

## Conducted Layer (Full Symphony)

```python
def conducted_forward(self, h, score_i):
    h_prev = h
    h_norm = rmsnorm(h)
    
    # PRIMA: Component dynamics
    conv_out = self.gated_conv(h_norm) * score_i[:, 0:1].unsqueeze(1)
    mamba_out = self.mamba3(h_norm) * score_i[:, 1:2].unsqueeze(1)
    mixer_out = self.outproj(torch.cat([conv_out, mamba_out], dim=-1))
    h = h + mixer_out
    ffn_out = self.ffn(rmsnorm(h)) * score_i[:, 2:3].unsqueeze(1)
    h_new = h + ffn_out
    
    # FORTE: Emotional arc
    gate = score_i[:, 3:4].unsqueeze(1)
    h = gate * h_new + (1 - gate) * h_prev
    return h
```

## Full Forward Pass

```python
def forward(self, tokens):
    h = self.embed(tokens)
    conductor_state = self.conductor.initial_proj(h.mean(dim=1))
    
    for group_idx in range(4):
        # VIVACE: Re-read and evolve
        if group_idx > 0:
            reading = self.conductor.orchestra_proj(h.mean(dim=1))
            conductor_state = self.conductor.gru(reading, conductor_state)
        
        # Generate score for this group
        raw = self.conductor.score_projs[group_idx](conductor_state).view(-1, 4, 4)
        volumes = torch.sigmoid(raw[:, :, :3]) * 2.0
        gates = torch.sigmoid(raw[:, :, 3:4])
        group_score = torch.cat([volumes, gates], dim=-1)
        
        # Conduct 4 layers
        for i in range(4):
            layer_idx = group_idx * 4 + i
            h = self.layers[layer_idx].conducted_forward(h, group_score[:, i])
    
    return self.lm_head(self.final_norm(h))
```

## Parameter Count

| Component | Params |
|-----------|--------|
| AMADEUS base | 241.6M |
| initial_proj (1024→64) | 65K |
| orchestra_proj (1024→64) | 65K |
| GRU (d=64) | 24K |
| 4 score_projs (64→16) | 4K |
| **Conductor total** | **~158K** |
| **GRAND TOTAL** | **~241.8M** |

**158K params control 241.6M params.** The conductor is 0.065% of the model. A TINY brain orchestrating a massive body.

## The Full Performance

```
Input: "The theory of relativity, proposed by Einstein in 1905..."

Conductor reads input → "Factual, scientific, long entities"

Group 1 (layers 1-4):
  conv=1.2 mamba=1.8 ffn=1.0 gate=0.9  → Heavy Mamba (need context for "Einstein")
  
Checkpoint: "Good entity tracking, but FFN overprocessing common words"

Group 2 (layers 5-8): GRU evolves →
  conv=1.0 mamba=1.5 ffn=0.7 gate=0.8  → Reduce FFN, maintain Mamba
  
Checkpoint: "Clean representations. Confident on factual content"

Group 3 (layers 9-12): GRU evolves →
  conv=0.8 mamba=1.2 ffn=1.3 gate=0.9  → Now boost FFN for synthesis

Checkpoint: "Ready for output. Need precise language generation"

Group 4 (layers 13-16): GRU evolves →
  conv=1.0 mamba=1.0 ffn=1.5 gate=1.0  → Full FFN for output quality ★FINALE★
```

## Training

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (50%) | 7.5 min | Backbone only. Conductor OFF. | Learn language |
| 2 (30%) | 4.5 min | + Full conductor (backbone FROZEN) | Conductor learns dynamics + arc + evolution |
| 3 (20%) | 3 min | Everything trainable | Joint optimization |

## Comparison: All 4 Movements

| | PRIMA | FORTE | VIVACE | **FINALE** |
|---|---|---|---|---|
| Component dynamics | YES | YES | YES | **YES** |
| Emotional arc | — | YES | — | **YES** |
| Evolving interpretation | — | — | YES | **YES** |
| Conductor params | 137K | 139K | 157K | **158K** |
| Signals per layer | 3 | 4 | 3 | **4** |
| Re-reads orchestra | — | — | 3× | **3×** |
| Complexity | LOW | LOW | MED | **MED** |

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| 4 signals + GRU = overfitting conductor | LOW | 158K params can't overfit. GRU has 24K params. |
| Score collapse (all signals → 1.0) | LOW | Init at sigmoid(0)×2=1.0 is already the "do nothing" state. The model can only improve from there. |
| Phase 2 too short for GRU to learn | MEDIUM | GRU is tiny. 4.5 min × ~1.5M tok/min = 6.75M tokens of conductor training. Sufficient for 158K params. |

## Success Criteria

1. AMADEUS base: loss < 4.5, decode > 300 tok/s
2. FINALE > PRIMA by > 0.2% loss (the extra signals help)
3. Score patterns differ by input type (visualizable)
4. GRU state evolves meaningfully across groups (cosine sim < 0.9)
5. Residual gates show non-uniform arc (some layers dampened, some amplified)
6. The 158K conductor params justify their existence vs AMADEUS baseline

## Implementation Roadmap

1. Build AMADEUS base (AMADEUS.md roadmap)
2. Implement PRIMA (component dynamics)
3. Test PRIMA alone (does basic conducting help?)
4. Add FORTE (residual gating)
5. Add VIVACE (GRU evolution)
6. Combine into FINALE
7. Ablation: AMADEUS vs PRIMA vs FORTE vs VIVACE vs FINALE
8. Visualize: full conductor score heatmap per input type
9. Visualize: conductor GRU state trajectory across depth
10. Visualize: residual gate "emotional arc" curves

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

**Baseline (estimated):** ~6,000 tok/s eager (15% MFU) — AMADEUS base + full 4-signal conductor

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +60% MFU (same as AMADEUS base) | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| `causal-conv1d` in GatedConv | 10x conv speedup | Available |
| `mamba-ssm` selective_scan_fn | 5.6x scan speedup (0.32ms) | Available |
| 4-signal conductor adds ~2% overhead | Most expressive conductor variant | By design |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~11,500 tok/s (28% MFU)
**Tokens in 45 min:** ~31.1M (1.9 BabyLM epochs)
**Ranking:** #13 of 22 architectures
