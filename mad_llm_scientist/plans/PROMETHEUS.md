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

    # Attention backend priority:
    # 1. aule-attention (Triton, hardware-agnostic, GQA native, pip install)
    #    `from aule_attention import flash_attention`
    #    attn_out = flash_attention(q, k, v, causal=True)  # handles GQA natively
    # 2. F.scaled_dot_product_attention (PyTorch SDPA, Inductor backend)
    # 3. flash-attn ROCm (0.05x on gfx1151 — LAST resort)
    #
    # Aule-Attention uses Triton kernels → generates code for gfx1151 directly.
    # MIT license, pip install aule-attention, head_dim<=128, full backward.
    # github.com/AuleTechnologies/Aule-Attention
    k = k.repeat_interleave(4, dim=1)  # only if using SDPA; aule handles GQA natively
    v = v.repeat_interleave(4, dim=1)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_out = self.wo(attn_out.transpose(1, 2).reshape(B, T, 1024))

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

## Hardware Optimization Notes (Strix Halo gfx1151)

> Updated 2026-04-09 with AMADEUS measured baselines and flash-attn reality check.

### Attention Backend on gfx1151: Verified Results (2026-04-10)

**Winner: hybrid_flash_sdpa_attention** — flash_attn forward (0.25ms) + SDPA backward (2.92ms) = **3.50ms fwd+bwd, 8.9% faster than pure SDPA** (3.84ms). Wired into `models/prometheus.py` with auto-detection.

| Backend | Fwd+Bwd | vs SDPA | Status |
|---------|---------|---------|--------|
| **hybrid_attention** | **3.50ms** | **8.9% faster** | **Active** (auto-detected) |
| PyTorch SDPA | 3.84ms | baseline | Fallback |
| flash_attn (pure Triton) | 4.84ms | 26% slower | Avoid for training |

**How it works:** `kernels/hip/hybrid_attention.py` uses flash_attn's fast forward, then passes the softmax logsumexp directly to `torch.ops.aten._flash_attention_backward` (SDPA's fast CK backward). No recompute overhead. Gradient accuracy: max_diff=0.002.

**Estimated attention cost per layer:**

| Backend | Per layer fwd+bwd | 2 layers total | Impact on throughput |
|---------|-------------------|---------------|---------------------|
| **hybrid_attention** | ~3.5ms | ~7ms | Negligible — PROMETHEUS ≈ TEMPEST speed |
| PyTorch SDPA | ~3.8ms | ~7.7ms | ~5% slower than hybrid |

### Realistic Throughput Estimate (updated with hybrid_attention)

Using AMADEUS component profiling as baseline (batch=8, seq=512):
- 14 Griffin layers × ~7.1ms each = ~100ms
- 2 hybrid attention layers × ~3.5ms fwd+bwd = ~7ms
- LM Head: ~13ms
- Overhead: ~5ms

**With hybrid_attention (verified):**
- Forward: ~100 + ~3.5 + 13 + 5 = **~122ms** → fwd+bwd ~366ms → **~11.2K tok/s eager**
- With autokernel + compile: **~14-16K tok/s**
- PROMETHEUS ≈ TEMPEST throughput. Attention is near-free with hybrid backend.

| Mode | Throughput | vs AMADEUS (10.4K) |
|------|-----------|-------------------|
| Eager | **10-12K** | ~1.0x |
| + autokernel + compile | **14-16K** | ~1.3-1.5x |

**Key insight:** hybrid_attention makes the 2 attention layers cost only ~7ms total (fwd+bwd), which is <5% of total wall-clock. PROMETHEUS gets global context essentially for free.

### Token Budget (corrected)

| Time | Tokens (at 11K tok/s) | BabyLM epochs |
|------|----------------------|---------------|
| 45 min | **30M** | 1.9 |
| 120 min | **79M** | 5.0 |

vs AMADEUS optimized: 28M in 45 min. **Similar data exposure, not 4x more.**

### Kernel Reuse
- **fused_residual_add_rmsnorm** (6.6x) — all 16 layers ✓
- **silu_gate_mul** (1.6x) — all 16 SwiGLU FFNs ✓ (FusedSwiGLUPattern)
- **cross_entropy** (1.8x) — loss ✓
- **hybrid_flash_sdpa_attention** — layers 4, 12 (8.9% faster than SDPA, auto-detected)
- **Chunked linear recurrence / HIP selective_scan** — 14 Griffin layers
- Apply via `autokernel.optimize(model, training=True)`

### Griffin Scan: Simpler Than Mamba-3
Griffin scan operator: `(a₂·a₁, a₂·b₁+b₂)` — real-valued scalars per dim.
No complex state, no dstate inner loop. Existing HIP `selective_scan` kernel works directly.
Expected ~0.5ms per Griffin layer (vs 1.8ms for Mamba-3 with HIP kernel).

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
| Attention backend performance | **HIGH** | Try Aule-Attention first (`pip install aule-attention`). If fast on gfx1151 → attention is ~free. If not → fall back to SDPA (~10ms/layer). Do NOT use standard flash-attn (0.05x). |
| 2 attn layers dominate wall-clock (SDPA scenario) | **MEDIUM** | Only applies if Aule-Attention doesn't work. At ~10ms SDPA each, they're ~15% of forward. If >25% wall-clock, reduce to 1 attn layer (at layer 8 only). |
| Throughput slower than TEMPEST (SDPA scenario) | **MEDIUM** | ~10-15% penalty with SDPA. If quality gain < 5% vs TEMPEST, drop attention → become TEMPEST. With Aule-Attention this risk disappears. |
| Momentum destabilizes training | LOW | beta init 0.5, learned via sigmoid(log_beta). If loss spikes, set beta→0 (= standard residual). |
| Griffin quality < Mamba-3 | MEDIUM | At comparable throughput (~11K vs 10.4K), data exposure is similar. Quality comparison is fair: architecture vs architecture, not throughput vs throughput. |

## Success Criteria

1. **Loss at 45 min < AMADEUS loss at 45 min** (the KEY metric — architecture quality at equal time)
2. **Loss at 45 min < TEMPEST loss at 45 min** (attention layers must justify their cost)
3. Throughput > 10K tok/s with autokernel+compile (competitive with AMADEUS optimized)
4. Momentum beta converges to non-zero value
5. SDPA attention layers stable (no crashes in 45 min)
6. Ablation: PROMETHEUS vs TEMPEST (2 attn layers vs 0) — attention must improve loss by >3%

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

### External Kernel Integration (verified 2026-04-10)

- **GatedConv:** causal-conv1d (10x vs nn.Conv1d) — imported from `models/tempest.py` which has `try/except` integration
- **Attention:** hybrid_flash_sdpa_attention (8.9% faster than SDPA) — auto-detected in `_detect_attn_backend()`, combines flash_attn forward + SDPA backward
- **Griffin scan:** FLA HGRN (0.40ms) available as alternative per-dim recurrence
- **Expected throughput with external kernels:** 10-14K tok/s eager, 14-16K with autokernel+compile
