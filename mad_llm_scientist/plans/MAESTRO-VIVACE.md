---
title: "MAESTRO-VIVACE"
domain: architectures
type: plan
status: stale
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %maestro-vivace]
---

# MAESTRO-VIVACE

**Movement III: Evolving Interpretation — The Conductor Comes Alive**

*A great conductor doesn't decide everything before the first note. They LISTEN. They FEEL how the orchestra is responding. They ADJUST mid-performance. The interpretation EVOLVES.*

## Hypothesis

PRIMA reads the input ONCE and produces a fixed score. But a real conductor is ALIVE — they re-read the orchestra at checkpoints and adjust. VIVACE adds a tiny GRU that gives the conductor a RECURRENT STATE evolving across the depth of the model. Every 4 layers, the conductor re-reads the hidden states and updates its interpretation.

**Extends:** MAESTRO-PRIMA. Same base (AMADEUS), same component dynamics, PLUS evolving interpretation via conductor GRU.

---

## The Conductor (Evolving)

```python
class ConductorVivace(nn.Module):
    def __init__(self, d_model=1024, d_cond=64, n_groups=4, signals_per_layer=3):
        self.initial_proj = nn.Linear(d_model, d_cond)                    # first reading
        self.orchestra_proj = nn.Linear(d_model, d_cond)                  # re-read projection
        self.gru = nn.GRUCell(d_cond, d_cond)                            # evolving state
        self.score_projs = nn.ModuleList([
            nn.Linear(d_cond, 4 * signals_per_layer) for _ in range(n_groups)  # 4 layers × 3 signals
        ])
    
    def forward(self, embed, hidden_states_at_checkpoints):
        # Initial reading from input
        state = F.relu(self.initial_proj(embed.mean(dim=1)))  # (B, 64)
        
        all_scores = []
        for group_idx in range(4):
            # Re-read orchestra at checkpoint (every 4 layers)
            if group_idx > 0:
                orchestra = hidden_states_at_checkpoints[group_idx - 1]  # (B, T, d)
                reading = F.relu(self.orchestra_proj(orchestra.mean(dim=1)))  # (B, 64)
                state = self.gru(reading, state)  # UPDATE interpretation
            
            # Produce score for this group
            group_score = torch.sigmoid(self.score_projs[group_idx](state)) * 2.0
            all_scores.append(group_score.view(-1, 4, 3))  # (B, 4, 3)
        
        return torch.cat(all_scores, dim=1)  # (B, 16, 3)
```

**The key difference from PRIMA:** The conductor reads the orchestra's state at layers 4, 8, 12 and ADJUSTS its score. If layers 1-4 produced confused representations, the conductor changes strategy for layers 5-8.

## Forward Pass (with Checkpoints)

```python
def forward(self, tokens):
    h = self.embed(tokens)
    
    # Initial conductor reading
    conductor_state = self.conductor.initial_proj(h.mean(dim=1))
    
    checkpoints = []
    for group_idx in range(4):
        # Get score for this group
        if group_idx > 0:
            reading = self.conductor.orchestra_proj(h.mean(dim=1))
            conductor_state = self.conductor.gru(reading, conductor_state)
        
        group_score = sigmoid(self.conductor.score_projs[group_idx](conductor_state)) * 2.0
        group_score = group_score.view(-1, 4, 3)
        
        # Process 4 layers with this group's score
        for i in range(4):
            layer_idx = group_idx * 4 + i
            h = self.layers[layer_idx].conducted_forward(h, group_score[:, i])
        
        checkpoints.append(h.detach())  # save for next re-read
    
    return self.lm_head(self.final_norm(h))
```

## Parameter Count

| Component | Params |
|-----------|--------|
| AMADEUS base | 241.6M |
| initial_proj (1024→64) | 65K |
| orchestra_proj (1024→64) | 65K |
| GRU (d=64) | 3 × (64×64 + 64×64) = 24K |
| 4 score_projs (64→12) | 4 × 0.8K = 3K |
| **Conductor total** | **~157K** |
| **GRAND TOTAL** | **~241.8M** |

## What The Conductor Learns (Evolving)

```
Group 1 (layers 1-4): Initial interpretation based on input embeddings
  → Checkpoint: "The orchestra sounds confused on entity tracking"

Group 2 (layers 5-8): GRU update → boost Mamba for context recovery  
  → Checkpoint: "Better! But FFN is overshooting on common patterns"

Group 3 (layers 9-12): GRU update → reduce FFN, maintain Mamba
  → Checkpoint: "Clean representations. Ready for final push"

Group 4 (layers 13-16): GRU update → balanced for output quality
```

The conductor's interpretation EVOLVES. Same input, but the score at layer 12 is DIFFERENT from the score at layer 1 — because the conductor learned from watching layers 1-11.

## Training

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (50%) | 7.5 min | Backbone only. Conductor OFF (scores=1.0). | Learn language |
| 2 (30%) | 4.5 min | + Conductor (backbone FROZEN). | Conductor learns to read AND re-read |
| 3 (20%) | 3 min | Everything trainable. | Joint optimization |

**Phase 2 trains the GRU.** The conductor must learn: (a) what the orchestra sounds like at each checkpoint, (b) how to adjust its signals in response.

## Risks & Success Criteria

| Risk | Mitigation |
|------|------------|
| GRU adds sequential dependency across depth | d=64, 4 calls. < 0.01ms overhead. |
| Conductor can't learn meaningful evolution in 4.5 min | GRU is TINY (24K params). Learns fast. |
| Checkpoints add memory overhead | Use `.detach()` — no gradient through checkpoints to conductor reads |

**Success:** Conductor GRU state changes meaningfully between groups (cosine sim < 0.9). Group 4 scores differ from Group 1 scores.

## Implementation Roadmap

1. Start from MAESTRO-PRIMA implementation
2. Add GRU cell (d=64) + orchestra_proj + 4 score_projs
3. Modify forward to process in 4 groups with checkpoint re-reads
4. Phase 2: freeze backbone, train conductor (including GRU dynamics)
5. Visualize: conductor state trajectory across 4 groups (t-SNE or PCA)

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

**Baseline (estimated):** ~6,000 tok/s eager (15% MFU) — AMADEUS base + GRU conductor evolution

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +60% MFU (same as AMADEUS base) | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| `causal-conv1d` in GatedConv | 10x conv speedup | Available |
| `mamba-ssm` selective_scan_fn | 5.6x scan speedup (0.32ms) | Available |
| GRU conductor adds ~1% overhead | Small recurrent state evolution per layer | By design |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~11,500 tok/s (28% MFU)
**Tokens in 45 min:** ~31.1M (1.9 BabyLM epochs)
**Ranking:** #12 of 22 architectures
