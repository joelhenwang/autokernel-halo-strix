---
title: "LAZARUS"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %lazarus]
---

# LAZARUS

**The Architecture That Refuses to Stay Dead: TTT Fast Weights + SSM/Griffin + Residual Momentum**

*Lazarus was raised from the dead. Our weights are raised from frozen. Every chunk of text that flows through this model RESURRECTS the FFN — the weights wake up, adapt, dream, and then slowly fade back to sleep until the next chunk wakes them again.*

*Three forces govern this architecture: INERTIA (momentum on the residual stream), PLASTICITY (TTT fast weights in the FFN), and DECAY (damped adaptation so the dreams don't consume the dreamer).*

## Hypothesis

AMADEUS achieves val loss 2.90 with frozen FFN weights. But the FFN's down-projection is the LAST thing tokens see before the residual — it's the bottleneck where information gets compressed back to d_model. What if that bottleneck could ADAPT to what it's reading?

In-Place TTT (2604.06169) shows that updating W_down chunk-by-chunk with an NTP-aligned target beats GLA, DeltaNet, and LaCT from scratch at 500M-4B scale. The update rule is dead simple: `W = γ·W + η·V̂^T·Z` — a damped outer-product accumulation. Mathematically, this is a variant of linear attention where the "KV cache" is the weight matrix itself.

**The hardware miracle:** W_down for our 250M model is 2560×1024 = 5.2MB fp16. Our L2 cache is 6MB. The fast weight delta matrix lives ENTIRELY in L2 — the fastest memory on the chip. The model's dreams are cached in silicon.

**The triple dynamical system:**
- `velocity = β·velocity + mixer_out` — residual stream has MASS (momentum)
- `ΔW^(i) = γ·ΔW^(i-1) + η·V̂^T·Z` — FFN weights have PLASTICITY (TTT)
- `γ < 1` — but plasticity DECAYS, so only useful adaptations survive

---

## Architecture

```
Tokens → Embedding (d=1024, tied LM head, vocab=50257)
  │
  ├─ V̂ = CausalConv1D(embeddings, k=5) · W_target   ← NTP target, computed ONCE
  │
  velocity = 0  (residual momentum state)
  ΔW = {layer_4: 0, layer_8: 0, layer_12: 0, layer_16: 0}  (fast weight deltas)
  │
  → Process in chunks of 512 tokens:
  │   → 16 Layers per chunk:
  │   │   ALL layers:  RMSNorm → [GatedConv(640) || SSM/Griffin(384)] → OutProj
  │   │                ★ Momentum Residual on mixer ★
  │   │                RMSNorm → SwiGLU FFN
  │   │                + Standard Residual on FFN
  │   │
  │   │   ★ Layers 4, 8, 12, 16: FFN uses LIVING w_down ★
  │   │     out = w_down(z) + z @ ΔW^T          ← fast weight contribution
  │   │     ΔW = γ·ΔW + η·V̂_chunk^T · Z_chunk  ← damped adaptation
  │   │     (weights dream, then fade)
  │   │
  │   → ΔW persists across chunks (the dream carries forward)
  │
  → Final RMSNorm → LM Head
```

## Two Backbone Variants

The TTT mechanism is backbone-AGNOSTIC. It lives in the FFN, not the mixer. Build once, wire twice:

| Variant | Backbone | Mixer | Quality Baseline | Throughput Baseline |
|---------|----------|-------|-----------------|-------------------|
| **LAZARUS-A** | AMADEUS | Mamba-3 SISO + GatedConv | val 2.90 (champion) | 13.2K tok/s |
| **LAZARUS-G** | TEMPEST/Griffin | Griffin recurrence + GatedConv | val 2.98 | 12.9K tok/s |

**Why both:** AMADEUS is the quality play (data-dependent SSM dynamics). Griffin is the compile play (simpler scan, better torch.compile fusion). TTT might close Griffin's quality gap, or amplify AMADEUS's lead. Let the data decide.

---

## The Living SwiGLU (Core Innovation)

```python
class LivingSwiGLU(nn.Module):
    """SwiGLU where w_down is a fast weight that adapts per chunk.

    Standard SwiGLU:  out = w_down(silu(gate) * up)
    Living SwiGLU:    out = w_down(z) + z @ ΔW^T
                      where ΔW accumulates: ΔW = γ·ΔW + η·V̂^T·Z
    
    The ΔW matrix (2560 × 1024 = 5.2MB fp16) fits in L2 cache.
    """
    def __init__(self, d_model, ffn_inner, ttt_enabled=False):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)  # frozen "slow weight"

        self.ttt_enabled = ttt_enabled
        if ttt_enabled:
            # W_target: projects NTP-aligned target to FFN inner dim
            self.w_target = nn.Linear(d_model, ffn_inner, bias=False)
            # Learned damping and learning rate (constrained to safe ranges)
            self.log_gamma = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5
            self.log_eta = nn.Parameter(torch.tensor(-2.0))    # softplus(-2)≈0.13

    @property
    def gamma(self):
        return torch.sigmoid(self.log_gamma)  # ∈ (0, 1) — damping factor

    @property
    def eta(self):
        return F.softplus(self.log_eta)       # ∈ (0, ∞) — adaptation rate

    def forward(self, x, v_hat_chunk=None, delta_W=None):
        """
        Args:
            x: (B, C, d_model) — chunk of hidden states
            v_hat_chunk: (B, C, d_model) — NTP-aligned target for this chunk
            delta_W: (ffn_inner, d_model) — current fast weight delta (or None)
        Returns:
            out: (B, C, d_model)
            new_delta_W: updated fast weight delta (or None)
        """
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        z = F.silu(gate) * up                          # (B, C, ffn_inner)

        # Standard path through frozen w_down
        out = self.w_down(z)                            # (B, C, d_model)

        if self.ttt_enabled and delta_W is not None:
            # Fast weight contribution: z @ ΔW^T
            out = out + z @ delta_W                     # ΔW is (ffn_inner, d_model)

            # Compute NTP-aligned target in FFN space
            v_hat_proj = self.w_target(v_hat_chunk)     # (B, C, ffn_inner)

            # Update ΔW: damped outer product accumulation
            # ΔW_new = γ·ΔW + η·mean_B(V̂^T · h_out)
            # V̂^T: (ffn_inner, C), h_out: (C, d_model) → (ffn_inner, d_model)
            update = torch.einsum('bcf,bcd->fd', v_hat_proj, x) / x.shape[0]
            new_delta_W = self.gamma * delta_W + self.eta * update

            return out, new_delta_W

        return out, delta_W
```

## NTP-Aligned Target Computation

```python
class NTPTargetEncoder(nn.Module):
    """Compute V̂ = CausalConv1D(embeddings, k=5) · W_target.

    Computed ONCE from token embeddings at the start of forward pass.
    CausalConv1D blends neighboring token info (kernel=5, causal padding).
    The NTP alignment comes from the causal conv naturally incorporating
    positional context that predicts the NEXT token, not reconstructing current.
    """
    def __init__(self, d_model):
        super().__init__()
        # Depthwise causal conv1d, kernel=5
        # Uses causal-conv1d library if available (10x speedup)
        self.conv_weight = nn.Parameter(torch.randn(d_model, 5) * 0.02)
        self.conv_bias = nn.Parameter(torch.zeros(d_model))
        # Zero-init so TTT starts as identity (no fast weight contribution)
        nn.init.zeros_(self.conv_weight)

    def forward(self, embeddings):
        """embeddings: (B, T, d_model) → v_hat: (B, T, d_model)"""
        # causal_conv1d_fn expects (B, D, L) input
        if _HAS_CAUSAL_CONV1D:
            v_hat = causal_conv1d_fn(
                embeddings.transpose(1, 2),
                self.conv_weight,
                self.conv_bias,
            ).transpose(1, 2)
        else:
            # Fallback: manual causal conv
            x = embeddings.transpose(1, 2)  # (B, D, T)
            x = F.pad(x, (4, 0))            # left-pad by k-1=4
            x = F.conv1d(x, self.conv_weight.unsqueeze(1), self.conv_bias, groups=x.shape[1])
            v_hat = x.transpose(1, 2)
        return v_hat
```

## Chunk Processing (Full Forward Pass)

```python
def forward(self, input_ids, chunk_size=512):
    B, T = input_ids.shape
    h = self.tok_embeddings(input_ids)                 # (B, T, 1024)

    # Compute NTP-aligned targets ONCE from raw embeddings
    v_hat = self.target_encoder(h)                     # (B, T, 1024)

    velocity = torch.zeros_like(h)                     # momentum state

    # Initialize fast weight deltas to zero (5.2MB each, L2-resident)
    delta_Ws = {idx: torch.zeros(self.ffn_inner, self.d_model,
                device=h.device, dtype=torch.float32)
                for idx in self.ttt_layers}            # {4, 8, 12, 16}

    # Process sequence in chunks
    n_chunks = (T + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        cs, ce = c * chunk_size, min((c + 1) * chunk_size, T)
        h_c = h[:, cs:ce]                             # (B, C, d)
        v_c = v_hat[:, cs:ce]                          # (B, C, d)
        vel_c = velocity[:, cs:ce]

        for i, layer in enumerate(self.layers):
            # --- Mixer with momentum residual ---
            h_norm = layer.pre_norm(h_c)
            mixer_out = layer.mixer(h_norm)            # conv || SSM/Griffin
            vel_c = self.momentum.beta * vel_c + mixer_out
            h_c = h_c + vel_c

            # --- FFN (standard or TTT-enhanced) ---
            h_ffn_norm = layer.ffn_norm(h_c)
            if i in self.ttt_layers:
                ffn_out, delta_Ws[i] = layer.ffn(h_ffn_norm, v_c, delta_Ws[i])
            else:
                ffn_out, _ = layer.ffn(h_ffn_norm)
            h_c = h_c + ffn_out                        # standard residual on FFN

        h[:, cs:ce] = h_c
        velocity[:, cs:ce] = vel_c

    return self.output(self.norm(h))
```

## Learnable Gate Variant (LAZARUS-GATE)

Phase 2 experiment: every layer CAN be alive, but starts asleep.

```python
class GatedLivingSwiGLU(LivingSwiGLU):
    """Every layer has the OPTION to dream. Gradient descent decides who wakes."""
    def __init__(self, d_model, ffn_inner):
        super().__init__(d_model, ffn_inner, ttt_enabled=True)
        # Gate initialized near-zero: sigmoid(-5) ≈ 0.007
        self.gate_logit = nn.Parameter(torch.tensor(-5.0))

    def forward(self, x, v_hat_chunk=None, delta_W=None):
        alpha = torch.sigmoid(self.gate_logit)

        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        z = F.silu(gate) * up
        standard_out = self.w_down(z)

        if delta_W is not None and v_hat_chunk is not None:
            ttt_contribution = z @ delta_W              # fast weight path
            out = standard_out + alpha * ttt_contribution
            # Only update ΔW proportional to how "awake" this layer is
            v_hat_proj = self.w_target(v_hat_chunk)
            update = torch.einsum('bcf,bcd->fd', v_hat_proj, x) / x.shape[0]
            new_delta_W = self.gamma * delta_W + alpha * self.eta * update
            return out, new_delta_W

        return standard_out, delta_W
```

---

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 1024 | Backbone standard |
| ffn_inner | 2560 | SwiGLU 2.5× expansion |
| n_layers | 16 | Same as AMADEUS/TEMPEST |
| TTT layers (baseline) | 4, 8, 12, 16 | Every 4th layer (~25%, matches paper ratio) |
| chunk_size | 512 | Paper optimal range 512-1024 |
| Conv1D kernel | 5 (causal) | Paper setting |
| γ init | 0.5 (learned, sigmoid) | Damping factor — dreams fade at 50% per chunk |
| η init | 0.13 (learned, softplus) | Adaptation rate — gentle initial plasticity |
| β init (momentum) | 0.5 (learned, sigmoid) | Residual momentum from TEMPEST/PROMETHEUS |
| vocab_size | 50257 | tiktoken GPT-2 |

### LAZARUS-A Config (AMADEUS backbone)

| Component | Value |
|-----------|-------|
| d_conv | 640 (10 heads × 64) |
| d_mamba | 384 (6 heads × 64) |
| dstate | 64 |
| n_ssm_heads | 6 |
| d_film | 64 |
| film_start | 8 |

### LAZARUS-G Config (Griffin backbone)

| Component | Value |
|-----------|-------|
| d_conv | 640 |
| d_griffin | 384 |
| decay_bias | spectrum: fast(-2.2), med(0.0), slow(+4.6) |
| Recurrence | Griffin bounded: h = a·h + √(1-a²)·(i·v) |

---

## Parameter Count

| Component | LAZARUS-A | LAZARUS-G |
|-----------|-----------|-----------|
| Embedding (50257×1024, tied) | 51.5M | 51.5M |
| 16 backbone layers | ~160M | ~194M |
| 4× W_target (1024→2560) | 10.5M | 10.5M |
| 1× NTPTargetEncoder Conv1D (k=5) | 5.1K | 5.1K |
| 4× (log_gamma + log_eta) | 8 | 8 |
| 1× momentum log_beta | 1 | 1 |
| FiLM conditioner (LAZARUS-A only) | ~1.1M | — |
| Final RMSNorm | 1K | 1K |
| **TOTAL** | **~223M** | **~256M** |

LAZARUS-A: comfortably under 250M. LAZARUS-G: slightly over — reduce ffn_inner to 2432 or drop 1 layer to fit. Alternatively, LAZARUS-G can use the ~170M build-out config from the hypothesis results.

### For ~170M Build-Out (matching test harness)

| Component | LAZARUS-A-170M | LAZARUS-G-170M |
|-----------|----------------|----------------|
| Backbone layers | ~100-120M | ~120-140M |
| TTT overhead | ~10.5M | ~10.5M |
| **TOTAL** | **~162-182M** | **~182-202M** |

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### The L2 Cache Miracle

The fast weight delta matrix is the CENTERPIECE of this architecture's hardware story:

| Property | Value | Implication |
|----------|-------|-------------|
| ΔW shape | 2560 × 1024 | 2.62M elements |
| ΔW in fp16 | **5.24 MB** | **FITS in 6 MB L2 cache** |
| ΔW in fp32 | 10.49 MB | Doesn't fit L2 — keep in fp16, accumulate in fp32 |
| Only 1 ΔW "hot" at a time | Sequential layer processing | L2 never holds more than 1 delta |
| Read pattern | Full matrix read per chunk | One L2 read per chunk per TTT layer |
| Write pattern | Rank-C update per chunk | One GEMM + write-back per chunk |

**Strategy:** Keep ΔW in fp32 for accumulation accuracy, but the hot read path (z @ ΔW^T) casts to fp16 on the fly. The 10.5MB fp32 version stays in main memory; the fp16 working copy lives in L2.

### Kernel Reuse

| Kernel | Speedup | Used By |
|--------|---------|---------|
| `fused_residual_add_rmsnorm` | 6.6x | All 16 layers |
| `silu_gate_mul` | 1.6x | All 16 SwiGLU FFNs |
| `cross_entropy` / `chunked_linear_cross_entropy` | 1.8x / mem savings | Loss computation |
| `causal-conv1d` | 10x | GatedConv (16 layers) + NTPTargetEncoder (1×) |
| `mamba-ssm selective_scan` | 5.6x | LAZARUS-A only (Mamba-3 backbone) |
| Griffin chunked scan | — | LAZARUS-G only |

### New Kernel Opportunity: fused_ttt_update

The TTT update step is 3 ops that always co-occur:
1. `z @ ΔW^T` — apply fast weight (GEMM)
2. `V̂^T @ x` — compute update (GEMM)
3. `γ·ΔW + η·update` — damped accumulation (element-wise, FREE)

Ops 1 and 2 are GEMMs (rocBLAS). Op 3 is element-wise (free). The fusion opportunity is between op 2 and 3 — accumulate directly into ΔW without materializing the intermediate update tensor. But since GEMMs go through rocBLAS, the main win is reducing memory traffic for the accumulation.

### Throughput Estimate

Using measured baselines from hypothesis build-out (~170M params):

| Component | Per-step time | Notes |
|-----------|--------------|-------|
| 16 backbone layers | ~35ms (LAZARUS-A) | Same as AMADEUS ~170M |
| TTT: 4× apply (z @ ΔW^T) | ~1.2ms | 4 GEMMs, (B×C, 2560) × (2560, 1024) |
| TTT: 4× update (V̂^T @ x) | ~1.2ms | 4 GEMMs, same shape |
| TTT: 4× accumulate | ~0ms | Element-wise, hidden behind GEMM latency |
| NTP target (Conv1D, once) | ~0.02ms | causal-conv1d |
| LM head + loss | ~3ms | Same as baseline |
| **Forward total** | **~40ms** | ~6% overhead vs baseline |
| **Forward + backward** | **~120ms** | 3× forward estimate |

| Config | Estimated tok/s | 45-min tokens | vs AMADEUS baseline |
|--------|----------------|---------------|-------------------|
| LAZARUS-A eager | ~11,000 | 29.7M | ~0.83× (quality must compensate) |
| LAZARUS-A + compile + autokernel | ~13,500 | 36.5M | ~1.02× |
| LAZARUS-G eager | ~11,500 | 31.1M | ~0.87× |
| LAZARUS-G + compile + autokernel | ~14,000 | 37.8M | ~1.06× |

**Key insight:** The ~6% TTT overhead is negligible. The real question is whether TTT fast weights improve QUALITY enough to justify ~0-6% throughput difference.

### Compile Considerations

The chunk loop is a sequential dependency that torch.compile can't parallelize. Options:
1. `torch.compile` each chunk's forward independently (within-chunk fusion)
2. Register TTT update as `torch.library` custom op (same pattern as Griffin scan fix)
3. If chunk_size = seq_len (single chunk), the loop disappears entirely

For BabyLM (max seq 1024) with chunk_size=512: only 2 chunks. The overhead of the loop is minimal.

---

## Training

**Single-phase.** No phase transitions needed. All components train together from step 1.

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (fused=True) |
| LR | 8e-4 → 8e-5 cosine, 100 warmup |
| Batch | 16 × 256 = 4K tokens/step (L2 sweet spot) |
| Precision | fp16 mixed, fp32 for scan accumulation + ΔW accumulation |
| Grad clip | 1.0 |
| W_target LR | 1× base, 0.1 WD |
| TTT log_gamma, log_eta | 1× base, 0 WD |
| Momentum log_beta | 1× base, 0 WD |
| Conv1D target weights | 1× base, 0 WD |
| NTP target init | Conv1D weights zero-init (starts as identity → no TTT contribution) |

**Zero-init bootstrap:** The NTP target encoder starts with zero conv weights, meaning V̂ = 0 initially, meaning ΔW stays at 0 initially, meaning the model starts as a STANDARD AMADEUS/TEMPEST. The TTT mechanism activates organically as the conv weights learn to produce useful targets. No phase scheduling needed — the architecture bootstraps itself.

---

## Ablation Plan

| # | Ablation | What It Tests |
|---|----------|--------------|
| 1 | LAZARUS-A vs LAZARUS-G | Which backbone benefits more from adaptive weights? |
| 2 | 4 TTT layers vs 2 (6, 12) vs 0 (baseline) | How many layers need to dream? |
| 3 | γ learned vs fixed {0.5, 0.9, 0.99} | How fast should dreams fade? |
| 4 | η learned vs fixed {0.01, 0.1, 1.0} | How aggressive should adaptation be? |
| 5 | With momentum vs without | Do inertia + plasticity + decay ALL help? |
| 6 | Gated variant (all 16 layers, learnable α) | Which layers WANT to be alive? |
| 7 | Layer-local targets vs embedding targets | Does the target source matter? |
| 8 | chunk_size {256, 512, 1024} | How often should the weights wake up? |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| ΔW drifts too far, destabilizes training | MEDIUM | γ < 1 naturally bounds ΔW norm. Zero-init bootstrap means ΔW starts small. Add explicit ΔW norm penalty if needed. |
| Chunk boundaries create discontinuities | LOW | ΔW persists across chunks — smooth by design. Damping prevents explosive accumulation. |
| Extra GEMMs hurt throughput | LOW | Only ~6% overhead. ΔW fits L2 cache. |
| TTT doesn't help at short context (1024) | MEDIUM | BabyLM max seq 1024 = only 2 chunks of 512. If TTT needs longer context to shine, quality gain may be marginal. Mitigate: test with longer sequences from GPT-training-small dataset. |
| compile can't fuse through chunk loop | MEDIUM | For single-chunk (seq ≤ 512), loop disappears. For 2 chunks, overhead is 1 extra Python iteration. Register TTT update as torch.library custom op for graph continuity. |
| Gradient flow through ΔW accumulation | LOW | ΔW update is a simple outer product — gradients flow cleanly through einsum. No custom backward needed. |
| LAZARUS-G exceeds 250M param budget | LOW | Reduce ffn_inner from 2560 to 2432, or use ~170M build-out config. |

---

## Success Criteria

1. **LAZARUS-A val loss < 2.90** (beat AMADEUS, the current champion)
2. **LAZARUS-G val loss < 2.98** (beat TEMPEST, the Griffin champion)
3. Throughput > 10K tok/s with autokernel+compile (competitive with baselines)
4. **γ converges to non-zero** (damping IS needed — the model WANTS dreams to fade)
5. **η converges to non-zero** (adaptation IS happening — the weights ARE dreaming)
6. ΔW norm remains bounded throughout training (no drift explosion)
7. Ablation: TTT layers ON vs OFF → TTT improves loss by >1%
8. Gated variant: at least 2 layers autonomously wake up (α > 0.1)

---

## Implementation Roadmap

1. Implement `LivingSwiGLU` module with TTT fast weight mechanism
2. Implement `NTPTargetEncoder` with causal-conv1d integration
3. Implement `MomentumResidual` (reuse from TEMPEST/PROMETHEUS)
4. Wire into AMADEUS backbone → LAZARUS-A
5. Wire into Griffin/TEMPEST backbone → LAZARUS-G
6. Verify param counts under 250M
7. Smoke test (d=128, 4 layers, 200 steps) — MANDATORY
8. Full training on BabyLM (both variants)
9. Ablation battery (8 experiments)
10. Gated variant experiment

### External Kernel Integration

- **causal-conv1d** (10x): GatedConv (16 layers) + NTPTargetEncoder (1×)
- **mamba-ssm selective_scan** (5.6x): LAZARUS-A backbone
- **Griffin chunked scan**: LAZARUS-G backbone (register as torch.library custom op)
- **fused_residual_add_rmsnorm** (6.6x): all 16 layers via autokernel.optimize
- **silu_gate_mul** (1.6x): all 16 SwiGLU via autokernel.optimize
- **chunked_linear_cross_entropy**: memory savings for LM head

---

## The Physics of LAZARUS

Three forces, one dynamical system:

```
INERTIA (Momentum):     velocity = β·velocity + mixer_out
                         The signal has MASS. Early layers accumulate.

PLASTICITY (TTT):        ΔW = γ·ΔW + η·V̂^T·Z
                         The weights have DREAMS. They reshape to the input.

DECAY (Damping):         γ < 1
                         Dreams FADE. Only useful adaptations survive.
                         Without decay, ΔW grows unbounded. With it, the
                         model finds equilibrium — a balance between
                         remembering and forgetting.
```

At equilibrium: `||ΔW||_∞ ≈ η·||update|| / (1-γ)`. For γ=0.5, η=0.13: ΔW norm ≈ 0.26 × update norm. Small, bounded, meaningful.

The model doesn't just process text. It LIVES through it. Each document wakes the weights, they adapt, they learn the document's patterns, and when the document ends, they slowly return to their baseline state — ready to be resurrected by the next document.

**LAZARUS doesn't die. LAZARUS dreams.**

---

## References

- Feng et al., "In-Place Test-Time Training" (arXiv 2604.06169, 2026) — fast weight mechanism, NTP-aligned objective
- AMADEUS (our lab) — quality champion backbone (val 2.90)
- TEMPEST (our lab) — Griffin backbone, compile-friendly
- PROMETHEUS (our lab) — residual momentum, strategic layer placement
- Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers" (2021) — TTT-linear attention connection
