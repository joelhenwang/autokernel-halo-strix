# TEMPEST

**Pure Element-Wise Storm: 16 Griffin Layers + Residual Momentum**

*A tempest doesn't need complexity. It needs raw, sustained FORCE. Every operation is element-wise. Maximum MFU. Maximum data exposure. Quality through QUANTITY.*

## Hypothesis

On Strix Halo without MFMA, the fastest possible architecture is one where EVERY non-FFN operation is element-wise. Griffin recurrence is exactly this: `h = a*h + sqrt(1-a²)*(i*v)` — sigmoid, multiply, sqrt, multiply, add. All element-wise. All FREE (hidden behind memory latency).

The ONLY matmuls are in the linear projections and FFN (handled by rocBLAS). Everything else rides for free. Target: **85-90% MFU**, seeing 2-3x more tokens than AMADEUS.

Residual momentum compensates for Griffin's quality gap vs Mamba-3 by giving the model depth-wise inertia.

---

## Architecture

```
Tokens → Embedding (d=1024, tied LM head, vocab=50257)
  velocity = 0
  │
  → 16 Parallel Hybrid Blocks (ALL identical):
  │     RMSNorm
  │     [GatedConv(d=640, k=3) || Griffin(d=384, decay spectrum)]
  │     Concat → OutProj(1024→1024)
  │     ★ Momentum Residual ★
  │     RMSNorm → SwiGLU FFN (1024→2560→1024)
  │     + Standard Residual (FFN)
  │
  → Final RMSNorm → LM Head
```

**ONE block type.** Repeated 16 times. No attention. No SSM scan with complex state. No data-dependent A/B/C. Just conv + Griffin + FFN + momentum.

---

## Griffin Recurrence (Pure Element-Wise)

```python
class GriffinRecurrence(nn.Module):
    def __init__(self, d_model, d_rec=384):
        super().__init__()
        self.w_a = nn.Linear(d_model, d_rec)     # decay gate projection
        self.w_i = nn.Linear(d_model, d_rec)     # input gate projection
        self.w_v = nn.Linear(d_model, d_rec)     # value projection
        # Decay bias spectrum: multi-scale temporal dynamics
        self.decay_bias = nn.Parameter(torch.zeros(d_rec))
        # Init spectrum: fast(-2.2), medium(0.0), slow(+4.6)
        with torch.no_grad():
            self.decay_bias[:96].fill_(-2.2)      # fast: local patterns
            self.decay_bias[96:288].fill_(0.0)    # medium: clause structure
            self.decay_bias[288:].fill_(4.6)      # slow: topic tracking

    def forward(self, x, h_prev=None):
        a = torch.sigmoid(self.w_a(x) + self.decay_bias)   # decay gate (0,1)
        i = torch.sigmoid(self.w_i(x))                      # input gate (0,1)
        v = self.w_v(x)                                      # value

        # Bounded recurrence (Griffin coupling)
        input_signal = torch.sqrt(1 - a**2) * (i * v)

        if self.training:
            # Chunked linear recurrence for parallel training
            # Operator: (a₂·a₁, a₂·b₁+b₂) — REAL scalars, simpler than Mamba-3
            h = chunked_linear_recurrence(a, input_signal, chunk_size=64)
        else:
            # Sequential for inference (trivial)
            h = a * h_prev + input_signal

        return h  # (B, T, d_rec)
```

**Why Griffin is faster than Mamba-3:**

| Property | Griffin | Mamba-3 SISO |
|----------|--------|-------------|
| Gate computation | sigmoid(Wx + bias) | sigmoid(Wx) + data-dependent RoPE |
| State dimension | d_rec (= d_rec) | d_rec × dstate (= d_rec × 64) |
| Scan operator | real scalar multiply | complex multiply with dstate inner loop |
| Memory per state | d_rec values | d_rec × dstate values |
| FLOPs per step | O(d_rec) | O(d_rec × dstate) |

Griffin is **simpler, faster, less memory**. The trade-off: no data-dependent dynamics (A,B,C are input-independent in Griffin). Residual momentum helps compensate.

---

## Residual Momentum

```python
# Per model (shared beta, or per-layer beta)
velocity = torch.zeros_like(h)

for layer in self.layers:
    mixer_out = layer.mixer(rmsnorm(h))  # conv || Griffin
    velocity = beta * velocity + mixer_out
    h = h + velocity
    h = h + layer.ffn(rmsnorm(h))  # FFN uses standard residual
```

Momentum on MIXER only. FFN stays standard. This gives the convolution+recurrence signal depth-wise inertia while keeping FFN transformations local.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| d_conv | 640 (10 × 64) |
| d_griffin | 384 (6 × 64) |
| n_layers | 16 |
| ffn_inner | 2560 (SwiGLU, 2.5×) |
| conv_kernel | 3 |
| decay_bias | spectrum: fast(-2.2), med(0.0), slow(+4.6) |
| momentum_beta | 0.5 (learned) |
| vocab_size | 50257 |
| block_size | 1024 |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×1024, tied) | 51.5M |
| Per layer: gated conv (1024→3×640 + conv₃) | ~2.0M |
| Per layer: Griffin (3 × 1024→384 + decay_bias) | ~1.2M |
| Per layer: output proj (1024→1024) | ~1.05M |
| Per layer: SwiGLU FFN (1024→2560→1024) | ~7.86M |
| Per layer: RMSNorm ×2 | ~2K |
| **Per layer total** | **~12.11M** |
| **16 layers** | **193.8M** |
| Momentum (1 scalar) | 1 |
| Final RMSNorm | 1K |
| **TOTAL** | **~245.3M** |

## Hardware Optimization Notes

### MFU Analysis

| Operation | % of layer FLOPs | MFU | Notes |
|-----------|-----------------|-----|-------|
| FFN matmuls (gate_up + down) | ~65% | ~50-60% | rocBLAS, bandwidth-bound |
| Conv/Griffin projections | ~25% | ~50-60% | rocBLAS, smaller matmuls |
| Griffin recurrence | ~5% | ~95% | Element-wise = free |
| Conv1d | ~3% | ~90% | Depthwise, cheap |
| Norms, gates, momentum | ~2% | ~100% | Element-wise = free |

**Blended MFU estimate: 65-75%.** Higher than AMADEUS (16%) because no complex SSM scan.

### Throughput Estimate

| Mode | Est. MFU | Est. tok/s | 45-min tokens |
|------|----------|-----------|---------------|
| Eager | 65-75% | **12-18K** | 32-49M |
| + torch.compile | 75-85% | **16-22K** | 43-59M |
| + autokernel | 78-88% | **18-25K** | 49-67M |

**vs AMADEUS (6.4K):** 2-4x faster. **vs Transformer+autokernel (43K):** 0.4-0.6x. The gap narrows significantly.

### Kernel Reuse
- fused_residual_add_rmsnorm (6.6x) — all 16 layers
- silu_gate_mul (1.6x) — all 16 SwiGLU FFNs
- cross_entropy (1.8x) — loss
- Chunked linear recurrence (from amadeus.py, simplified for real-valued Griffin)
- Apply via `autokernel.optimize(model, training=True)`

---

## Training

**Single-phase.** One block type, no components to phase in. Train everything from step 1.

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (fused=True) |
| LR | 8e-4 → 8e-5 cosine, 100 warmup |
| Batch | 48 × 512 = 24K tokens/step |
| Precision | fp16 mixed, fp32 for scan accumulation |
| Grad clip | 1.0 |
| Griffin decay_bias LR | 0.1× base (preserve spectrum) |
| Momentum beta LR | 1× base |

**The simplest training recipe in the lab.** No phase transitions. No special components. Just train.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Griffin quality < Mamba-3 (no data-dependent dynamics) | HIGH | Residual momentum adds expressiveness. 2-4x more data exposure compensates. Ablate: equal-time comparison. |
| Momentum beta collapses to 0 | LOW | Init 0.5, learned. Monitor. If collapses: fix at 0.3. |
| Still slower than transformer+autokernel | MEDIUM | At 18K vs 43K, transformer sees 2.4x more data. Quality-per-token must be 2.4x better. |
| Chunked Griffin scan still slow | LOW | Griffin scan is SIMPLER than Mamba-3 — real scalars, no complex, no dstate. Should be faster. |

## Success Criteria

1. Throughput > 12K tok/s (2x AMADEUS minimum)
2. MFU > 60% (vs AMADEUS 16%)
3. Loss at equal wall-clock time < AMADEUS loss
4. Momentum beta stays non-zero (contributing)
5. Decay spectrum preserved (fast/medium/slow heads remain distinct)
6. Implementation < 300 lines

## Implementation Roadmap

1. Implement GriffinRecurrence with decay spectrum
2. Implement GatedConv (reuse from AMADEUS)
3. Implement MomentumResidual
4. Implement chunked_linear_recurrence for Griffin (simplify from Mamba-3 version)
5. Assemble 16 identical blocks, verify ~245M params
6. Smoke test (d=128, 4 layers)
7. Measure MFU: target >60%
8. Full BabyLM training (45 min)
9. Compare vs AMADEUS at equal wall-clock time
