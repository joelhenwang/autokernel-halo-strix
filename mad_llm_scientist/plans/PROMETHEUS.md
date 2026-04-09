# PROMETHEUS

**Steal Fire from the Gods: 2 Flash-Attn + 14 Griffin + Residual Momentum**

*Prometheus stole fire from the gods and gave it to humanity. We steal ATTENTION from transformers — just 2 layers — and give it to our Griffin architecture. The rest stays efficient.*

## Hypothesis

AMADEUS proves SSM hybrids work but achieves only 16% MFU (6.4K tok/s). Transformers + autokernel hit 54% MFU (43K tok/s). The gap isn't about architecture — it's about THROUGHPUT. What if we use MOSTLY Griffin (element-wise, ~85% MFU) but steal 2 flash-attn layers for the global context that Griffin can't provide?

14 Griffin layers run fast. 2 attention layers (flash-attn on ROCm) provide direct position-to-position context. Residual momentum adds the secret sauce.

---

## Architecture

```
Tokens → Embedding (d=1024, tied LM head, vocab=50257)
  velocity = 0  (residual momentum state)
  │
  → 16 Layers:
  │   Layers 1-3:   Griffin hybrid (GatedConv||Griffin) + SwiGLU  ← fast, local
  │   ★ Layer 4:    flash-attn GQA + SwiGLU                      ← FIRE: global context
  │   Layers 5-11:  Griffin hybrid + SwiGLU                       ← fast, local+medium
  │   ★ Layer 12:   flash-attn GQA + SwiGLU                      ← FIRE: late refinement
  │   Layers 13-16: Griffin hybrid + SwiGLU                       ← fast, output prep
  │
  │   ALL layers use Residual MOMENTUM (not standard residual)
  │
  → Final RMSNorm → LM Head
```

## Griffin Hybrid Layer (14 of 16)

```python
def griffin_layer_forward(self, h, velocity, token_ids):
    # Residual momentum injection
    h_norm = rmsnorm(h)

    # Parallel: gated conv (local) || Griffin recurrence (medium-range)
    conv_out = self.gated_conv(h_norm)                    # (B, T, 640)
    griffin_out = self.griffin(h_norm)                     # (B, T, 384)
    mixer_out = self.outproj(cat([conv_out, griffin_out])) # (B, T, 1024)

    # Momentum residual (NOT standard h = h + mixer_out)
    velocity = self.beta * velocity + mixer_out
    h = h + velocity

    # FFN with standard residual (momentum only on mixer)
    h = h + self.ffn(rmsnorm(h))
    return h, velocity
```

## Attention Layer (2 of 16 — layers 4 and 12)

```python
def attention_layer_forward(self, h, velocity):
    h_norm = rmsnorm(h)

    # GQA: 8 query heads, 2 KV heads (4:1 ratio)
    q = self.wq(h_norm).view(B, T, 8, 128).transpose(1, 2)
    k = self.wk(h_norm).view(B, T, 2, 128).transpose(1, 2)
    v = self.wv(h_norm).view(B, T, 2, 128).transpose(1, 2)

    # flash-attn (ROCm build available on training machine)
    attn_out = flash_attn_func(q, k, v, causal=True)  # O(T) memory
    attn_out = self.wo(attn_out.reshape(B, T, 1024))

    # Momentum residual
    velocity = self.beta * velocity + attn_out
    h = h + velocity

    h = h + self.ffn(rmsnorm(h))
    return h, velocity
```

**Why layers 4 and 12:**
- Layer 4: Early global context. The model has basic token representations from layers 1-3. Attention at layer 4 creates GLOBAL dependencies before Griffin refines them.
- Layer 12: Late refinement. After 11 layers of processing, attention at layer 12 allows the model to do final GLOBAL reasoning before output.
- This matches Falcon-H1/Hymba/LFM2's finding: a few attention layers at strategic positions ≈ full attention quality.

## Residual Momentum (Secret Ingredient)

```python
class MomentumResidual(nn.Module):
    def __init__(self, beta_init=0.5):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init / (1 - beta_init))))

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)  # (0, 1)

    def forward(self, h, layer_output, velocity):
        velocity = self.beta * velocity + layer_output
        return h + velocity, velocity
```

Applied to the MIXER output (conv/Griffin/attention), NOT to FFN. This gives the mixer's signal momentum across depth while keeping FFN updates local.

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| d_conv | 640 (10 × 64) |
| d_griffin | 384 (6 × 64) |
| n_layers | 16 (14 Griffin + 2 attention) |
| n_attn_heads | 8 (at layers 4, 12) |
| n_kv_heads | 2 (GQA 4:1 ratio) |
| head_dim | 128 |
| ffn_inner | 2560 (SwiGLU) |
| momentum_beta_init | 0.5 |
| vocab_size | 50257 |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×1024, tied) | 51.5M |
| 14 Griffin layers × ~10.0M (conv+griffin+outproj+FFN+norms) | 140.0M |
| 2 Attention layers × ~12.5M (QKV+WO+FFN+norms) | 25.0M |
| Momentum params (16 scalars) | 16 |
| Final RMSNorm | 1K |
| **TOTAL** | **~216.5M** |

Under 250M with room. Options: increase ffn_inner, add PLE, add Engram.

## Hardware Optimization Notes

### Throughput Estimate
- 14 Griffin layers at ~85% MFU: 14/16 × 85% = 74% contribution
- 2 flash-attn layers at ~40-54% MFU: 2/16 × 47% = 6% contribution
- Blended MFU: ~**70-80%**
- At 75% MFU: ~**20-30K tok/s** for 216M params

### Token Budget

| Time | Tokens (at 25K tok/s) |
|------|----------------------|
| 45 min | **67.5M** (4.2 BabyLM epochs) |
| 120 min | **180M** |

vs AMADEUS: 17M in 45 min. **PROMETHEUS sees 4x more data.**

### Kernel Reuse
- fused_residual_add_rmsnorm (6.6x) — all 16 layers
- silu_gate_mul (1.6x) — all 16 SwiGLU FFNs
- cross_entropy (1.8x) — output loss
- flash-attn (ROCm build) — layers 4, 12
- Griffin scan: chunked linear recurrence (chunk_size=64)
- Apply via `autokernel.optimize(model, training=True)`

### Scan: Griffin is SIMPLER than Mamba-3
Griffin scan operator: `(a₂·a₁, a₂·b₁+b₂)` — real-valued scalars per dim.
Mamba-3 scan: complex data-dependent A,B,C with RoPE.
Griffin scan should be **faster per step** → higher MFU than AMADEUS.

## Training

| Phase | Budget | Active |
|-------|--------|--------|
| 1 (60%) | 27 min | Full model. All layers active. Momentum ON from start. |
| 2 (40%) | 18 min | + optional PLE/FiLM/Engram additions |

**Simpler than AMADEUS:** No phased component activation needed. Griffin and attention are both standard — train everything together from step 1.

**Optimizer:** AdamW, 8e-4 → 8e-5 cosine, 100 warmup. Momentum beta gets 1x LR.

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| flash-attn on ROCm crashes | MEDIUM | Fallback: torch SDPA (slower but works). Or: remove attn layers → TEMPEST. |
| 2 attn layers dominate wall-clock | LOW | flash-attn is efficient. If >20% wall-clock, reduce to 1 attn layer. |
| Momentum destabilizes training | LOW | beta init 0.5, learned. If loss spikes, set beta→0 (= standard residual). |
| Griffin quality < Mamba-3 | MEDIUM | The 4x more data exposure may compensate. Ablation: PROMETHEUS vs AMADEUS at equal wall-clock time. |

## Success Criteria

1. Throughput > 20K tok/s (3x AMADEUS)
2. Loss at 45 min < AMADEUS loss at 45 min (despite simpler architecture)
3. Momentum beta converges to non-zero value (momentum IS being used)
4. flash-attn layers stable on ROCm (no crashes in 45 min)
5. The 2 attention layers show qualitatively different attention patterns (not redundant)

## Implementation Roadmap

1. Implement Griffin recurrence (gated, bounded, with decay spectrum)
2. Implement GatedConv (reuse from AMADEUS)
3. Implement GQA attention layer with flash-attn
4. Implement MomentumResidual (one scalar, one multiply-add)
5. Assemble: 14 Griffin + 2 attn + momentum, verify ~216M params
6. Smoke test (d=128, 4 layers)
7. Full training on BabyLM
8. Ablation: momentum ON vs OFF
9. Ablation: 2 attn layers vs 0 (= TEMPEST) vs 4
