---
title: "ARGUS-PRIME"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %argus-prime]
---

# ARGUS-PRIME

**The Streamlined LFM2.5: Strip the Fat, Upgrade the Chassis, Find the Michelin Recipe**

*ARGUS had 6 mechanisms and 15.3K tok/s. LFM2.5 has 2 mechanisms and dominates benchmarks. Sometimes the greatest innovation is knowing what to REMOVE. ARGUS-PRIME is the disciplined evolution: same LFM2.5 skeleton, fewer distractions, faster chassis, surgical ablation to find the minimum winning recipe.*

## Motivation

ARGUS (156M, 15.3K tok/s compile+AK) was our first LFM2.5-inspired build. Profiling revealed:
- **Backward pass = 70.8% of training time** (306ms / 432ms)
- TTT on all 4 GQA layers creates expensive einsum chains in backward
- d_model=768 underutilizes rocBLAS Tensile tiles
- Engram + MatFormer add complexity for unproven quality gains at 16M tokens

Meanwhile, compile-optimized Tempest hits **22.3K tok/s** and LlamaModel hits **47.8K tok/s**. ARGUS is leaving speed on the table.

**The question:** What is the MINIMUM recipe that gives us LFM2.5's quality at maximum throughput?

**The sniper hypothesis:** Instead of spraying TTT across 4 layers (shotgun), place ONE maximally-powerful TTT at layer 16 (the final FFN before the LM head) with multi-step gradient updates. FiLM provides air support — cheap context modulation across the upper half. One deep surgical strike at the point of maximum impact, supported by context awareness everywhere else.

A 2×3 ablation will tell us: TTT (1-single/1-multi/2-standard layers) × FiLM (yes/no).

---

## What Changes From ARGUS (informed by LFM2 Technical Report)

The LFM2 technical report (arXiv 2511.23404) revealed key differences from our assumptions:
- **6 GQA layers in 16** (not 4) — ratio is 10:6 (5:3), not 3:1
- **FFN is 4.5× d_model** (4608 at d=1024) — much bigger than our 2.5×
- **8 KV heads** (not 4) — finer-grained KV sharing
- **QK-Norm** on attention — stability technique we lacked
- **Hardware-in-the-loop search** — architecture optimized under real device constraints

### Scaling to ~175M (from LFM2-350M)

LFM2-350M has d=1024, ffn=4608, 350M params. Halving to 175M while keeping the 16-layer structure and 10:6 ratio → **d=768, ffn=2816 (3.7× expansion)**.

| Change | ARGUS (current) | ARGUS-PRIME | Why |
|--------|----------------|-------------|-----|
| d_model | 768 | **768** | Keeps LFM2 scaling (350M→175M = d 1024→768) |
| ffn_inner | 2048 (2.7×) | **2816 (3.7×)** | Closer to LFM2's 4.5× ratio — FFN is where capacity lives |
| n_layers | 16 | **16** | Same as LFM2-350M |
| Conv:GQA ratio | 12:4 (3:1) | **10:6 (5:3)** | Matches LFM2's actual ratio (6 GQA in 16 layers) |
| GQA positions | 4,8,12,16 | **3,6,8,10,13,16** | 6 attention layers evenly distributed |
| n_heads / n_kv | 12 / 4 | **12 / 8** | 8 KV heads matches LFM2 (n_heads scales with d: 12 at d=768) |
| head_dim | 64 | **64** | Same as LFM2 |
| Attention | SDPA | **hybrid_flash_sdpa + QK-Norm** | LFM2 uses QK-Norm; hybrid is 8.9% faster |
| Engram | YES | **REMOVED** | Unproven at 16M tokens |
| MatFormer | YES | **REMOVED for training** | Branching hurts Inductor |
| TTT layers | 4 (all GQA) | **1 or 2 (ablated)** | Sniper approach: minimum effective dose |
| FiLM | NO | **Ablated: yes or no** | Steal AMADEUS's context awareness |
| d_conv | 768 | **768** (= d_model) | Full-width conv like LFM2 |

**What stays the same:**
- RoPE on GQA layers (LFM2 confirmed)
- Momentum residual (free, our addition beyond LFM2)
- Inlined momentum + plain RMSNorm (compile-friendly)
- Tied embeddings + single LM head
- causal-conv1d (10x) for GatedConv
- Conv kernel = 3 (LFM2 confirmed)

---

## The 2×3 Ablation Matrix — The Sniper Study

TTT is MANDATORY (In-Place TTT is a core thesis). The question is: **how many layers and how deep?**

| Variant | TTT Config | FiLM | Expected tok/s | What It Tests |
|---------|-----------|------|----------------|--------------|
| **B0** | 1 layer (16), single-step | NO | ~24-27K | Minimal sniper — one shot, no support |
| **B1** | 1 layer (16), single-step | YES | ~23-26K | Sniper with air support — does FiLM help the single TTT? |
| **B2** | 1 layer (16), **multi-step (3 steps)** | NO | ~23-26K | Deep sniper — 3 gradient steps at layer 16 |
| **B3** | 1 layer (16), **multi-step (3 steps)** | YES | ~22-25K | **THE HYPOTHESIS: deep sniper + air support** |
| **B4** | 2 layers (8, 16), single-step | NO | ~22-25K | Standard bracket — the NEXUS approach |
| **B5** | 2 layers (8, 16), single-step | YES | ~21-24K | Bracket with air support |

### Why Layer 16 for the Single TTT

Layer 16 is the LAST GQA block — the final FFN before the LM head. This is the bottleneck where hidden representations get compressed into predictions. Adapting THIS layer directly shapes EVERY output token. Maximum leverage from minimum investment.

The other 3 GQA layers (4, 8, 12) use standard SwiGLU. They still provide global context via attention — they just don't adapt their FFN weights.

### Multi-Step TTT (B2, B3)

Standard TTT does 1 outer-product update per chunk. Multi-step does **3 updates per chunk** — iterating the gradient step:

```python
# Single-step (B0, B1): 1 update per chunk
delta_W = lr * outer_product(h_chunk, target_chunk)
w_adapted = w_down + cumsum(delta_W)

# Multi-step (B2, B3): 3 updates per chunk (inner loop)
for step in range(3):
    output = h_chunk @ w_current.T
    grad = outer_product(h_chunk, target - output)
    w_current = w_current + lr * grad
```

This is more expensive per layer (~3× the TTT compute at layer 16), but:
- We only have 1 TTT layer (not 4), so total TTT cost is still ~75% LESS than ARGUS's 4 single-step layers
- The multi-step lets the single layer adapt MORE deeply to the document
- The paper (2604.06169) showed multi-step helps at longer contexts
- 3 steps × 1 layer ≈ cost of 1.5 single-step layers — well within budget

### What Each Comparison Tells Us

| Comparison | Question Answered |
|-----------|-----------------|
| B0 vs B2 | Does multi-step help for single TTT layer? (depth vs breadth) |
| B0 vs B1 | Does FiLM complement single TTT? |
| B2 vs B3 | Does FiLM complement deep single TTT? |
| B3 vs B5 | Is 1 deep TTT + FiLM as good as 2 standard TTT + FiLM? |
| B1 vs B5 | Does the 2nd TTT layer add value when FiLM is present? |
| B0 vs B4 | Is 1 sniper as good as 2 standard? (minimum dose) |
| Best variant vs AMADEUS (2.90) | Did we beat the champion? |
| Best variant vs Tempest (2.98) | Did we beat the speed king? |

---

## Architecture (Shared Across All Variants)

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  velocity = 0  (residual momentum state)
  │
  → 16 Layers (10:6 ShortConv/GQA — LFM2 ratio):
  │
  │   Layers 1-2:   ShortConvBlock (GatedConv + Momentum + SwiGLU)
  │   ★ Layer 3:    GQABlock (GQA+QK-Norm + Momentum + SwiGLU)
  │   Layers 4-5:   ShortConvBlock
  │   ★ Layer 6:    GQABlock
  │   Layer 7:      ShortConvBlock
  │   ★ Layer 8:    GQABlock  [+ FiLM fingerprint computed here if enabled]
  │   Layer 9:      ShortConvBlock  [+ FiLM if enabled]
  │   ★ Layer 10:   GQABlock  [+ FiLM if enabled]
  │   Layers 11-12: ShortConvBlock  [+ FiLM if enabled]
  │   ★ Layer 13:   GQABlock  [+ FiLM if enabled]
  │   Layers 14-15: ShortConvBlock  [+ FiLM if enabled]
  │   ★ Layer 16:   GQABlock  [+ FiLM if enabled] [+ TTT on FFN: THE SNIPER]
  │
  → Final RMSNorm → LM Head

  Layer 16 (the sniper):
    GQA Attention + QK-Norm → Momentum Residual
    → FiLM modulation (if enabled — carries context from layer 8)
    → RMSNorm → TTTSwiGLU (1 or 3 gradient steps per chunk)
    → Residual → output to LM head
```

**6 GQA layers** at positions 3, 6, 8, 10, 13, 16 — roughly every 2-3 layers, matching LFM2's 10:6 ratio. Global context is available every ~2.5 layers instead of every 4 (ARGUS) or never (Tempest).

**The logic:** Layers 1-15 build the representation with frequent global attention. Layer 16's attention gathers final global context. Then the LIVING FFN adapts its projection to the specific document — the last transform before vocabulary prediction. One surgical adaptation at the point of maximum leverage.

**For B4/B5 (2-TTT variants):** Layer 8's GQA block ALSO gets TTTSwiGLU. Layer 8 = mid-document adaptation (at the FiLM fingerprint point). Layer 16 = final refinement.

### ShortConvBlock (12 layers — same for all variants)

```python
class ShortConvBlock(nn.Module):
    """GatedConv mixer + momentum residual + SwiGLU FFN.
    
    Compile-friendly: momentum inlined, RMSNorm plain PyTorch.
    """
    def __init__(self, cfg, film_enabled=False):
        self.pre_norm = RMSNorm(cfg.d_model)
        self.conv = GatedConv(cfg.d_model, cfg.d_conv, cfg.conv_kernel)
        self.out_proj = nn.Linear(cfg.d_conv, cfg.d_model, bias=False)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)
        self.film_enabled = film_enabled

    def forward(self, x, velocity, beta, film_gamma=None, film_beta=None):
        mixer_out = self.out_proj(self.conv(self.pre_norm(x)))
        
        # Inlined momentum (fewer autograd nodes → Inductor fuses)
        velocity = beta * velocity + mixer_out
        x = x + velocity
        
        # Optional FiLM before FFN
        if self.film_enabled and film_gamma is not None:
            x = film_gamma.unsqueeze(1) * x + film_beta.unsqueeze(1)
        
        x = x + self.ffn(self.ffn_norm(x))
        return x, velocity
```

### GQABlock (6 layers — TTT on layer 16 only, optional on layer 8)

```python
class GQABlock(nn.Module):
    """GQA Attention with QK-Norm + momentum + SwiGLU/TTTSwiGLU FFN.
    
    Uses hybrid_flash_sdpa_attention (8.9% faster).
    QK-Norm: L2-normalize Q and K before attention (LFM2 technique).
    12 query heads, 8 KV heads, head_dim=64 at d=768.
    """
    def __init__(self, cfg, ttt_enabled=False, ttt_multi_step=False,
                 film_enabled=False):
        self.pre_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                              qk_norm=True)  # LFM2: QK-Norm
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.film_enabled = film_enabled
        
        if ttt_enabled and ttt_multi_step:
            self.ffn = MultiStepTTTSwiGLU(cfg.d_model, cfg.ffn_inner,
                                           cfg.ttt_chunk, cfg.ttt_lr_init,
                                           ttt_steps=3)
        elif ttt_enabled:
            self.ffn = TTTSwiGLU(cfg.d_model, cfg.ffn_inner,
                                  cfg.ttt_chunk, cfg.ttt_lr_init)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)

    def forward(self, x, velocity, beta, freqs_cis, 
                film_gamma=None, film_beta=None, ttt_target=None):
        attn_out = self.attn(self.pre_norm(x), freqs_cis)
        
        # Inlined momentum
        velocity = beta * velocity + attn_out
        x = x + velocity
        
        # Optional FiLM before FFN
        if self.film_enabled and film_gamma is not None:
            x = film_gamma.unsqueeze(1) * x + film_beta.unsqueeze(1)
        
        h_ffn = self.ffn_norm(x)
        if isinstance(self.ffn, (TTTSwiGLU, MultiStepTTTSwiGLU)) and ttt_target is not None:
            x = x + self.ffn(h_ffn, ttt_target=ttt_target)
        else:
            x = x + self.ffn(h_ffn)
        
        return x, velocity
```

**QK-Norm implementation** (from LFM2):
```python
# Inside Attention.forward, after projecting Q and K:
q = F.normalize(q, dim=-1) * self.q_scale  # learned scale per head
k = F.normalize(k, dim=-1) * self.k_scale  # learned scale per head
# Then apply RoPE, then attention as normal
```

### FiLM Conditioner (from AMADEUS — enabled in A1, A3, A5)

```python
class FiLMConditioner(nn.Module):
    """Context fingerprint at GQA layer 8, modulates layers 9-16."""
    
    def __init__(self, d_model, d_film=64, n_conditioned_layers=8):
        self.context_proj = nn.Linear(d_model, d_film)
        self.gamma_projs = nn.ModuleList([
            nn.Linear(d_film, d_model, bias=True)
            for _ in range(n_conditioned_layers)
        ])
        self.beta_projs = nn.ModuleList([
            nn.Linear(d_film, d_model, bias=True)
            for _ in range(n_conditioned_layers)
        ])
        # Zero-init: starts as identity (gamma=1, beta=0)
```

FiLM fingerprint computed at **GQA layer 8** (= layer 8 in 1-indexed, the 2nd attention layer). This is a natural fingerprint point: the model has seen 7 layers of processing + 1 attention layer. Modulates all subsequent layers (9-16).

### TTTSwiGLU (from ARGUS — enabled on layer 16, optionally layer 8)

Same ByteDance cumsum approach as current ARGUS, upgraded to d=1024, with new multi-step variant:

- Chunked outer-product updates on w_down
- `detach()` on cumsum for memory savings
- `ttt_target = h` (current hidden state as target)
- ttt_lr = 0.01 (fixed buffer, not learned)
- ttt_chunk = 512 (larger = fewer serial steps)

**Single-step (B0, B1, B4, B5):** Standard — 1 outer-product gradient update per chunk. Same as ARGUS.

**Multi-step (B2, B3):** 3 inner gradient steps per chunk on the single layer 16 TTT. The w_down adapts MORE deeply to each chunk before moving on. Implementation:

```python
class MultiStepTTTSwiGLU(TTTSwiGLU):
    """TTTSwiGLU with multiple gradient steps per chunk.
    
    3 steps × 1 layer ≈ cost of 1.5 single-step layers.
    Much cheaper than 4 single-step layers (ARGUS original).
    """
    def __init__(self, d_model, ffn_inner, ttt_chunk=512,
                 ttt_lr_init=0.01, ttt_steps=3, ttt_conv_kernel=5):
        super().__init__(d_model, ffn_inner, ttt_chunk, ttt_lr_init, ttt_conv_kernel)
        self.ttt_steps = ttt_steps

    def forward(self, x, ttt_target=None):
        B, T, _ = x.shape
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        h = F.silu(gate) * up  # (B, T, ffn_inner)

        if ttt_target is None:
            return self.w_down(h)

        # Smooth target via causal conv1d
        t = self.ttt_conv(ttt_target.transpose(1, 2))[:, :, :T].transpose(1, 2)

        C = self.ttt_chunk
        h_chunked = self._pad_and_chunk(h, C)
        t_chunked = self._pad_and_chunk(t, C)
        nc = h_chunked.shape[1]

        # Start from original w_down
        w_current = self.w_down.weight.clone()  # (d_model, ffn_inner)

        outputs = []
        for chunk_idx in range(nc):
            h_c = h_chunked[:, chunk_idx]       # (B, C, ffn_inner)
            t_c = t_chunked[:, chunk_idx]       # (B, C, d_model)

            # Multi-step inner loop on this chunk
            for step in range(self.ttt_steps):
                out_c = h_c @ w_current.T       # (B, C, d_model)
                t_proj = F.linear(t_c, self.ttt_proj.weight)
                # Gradient of reconstruction loss
                grad = torch.einsum('bch,bcd->dh', h_c, t_proj - out_c) / (B * C)
                w_current = w_current + self.ttt_lr * grad.detach()

            # Final output with adapted weight
            outputs.append(h_c @ w_current.T)

        return torch.cat(outputs, dim=1)[:, :T]
```

**Cost comparison:**

| Config | TTT ops per forward | vs ARGUS (4 single-step) |
|--------|-------------------|-------------------------|
| B0/B1: 1 layer, 1 step | 1 | **75% cheaper** |
| B2/B3: 1 layer, 3 steps | 3 | **25% cheaper** |
| B4/B5: 2 layers, 1 step | 2 | **50% cheaper** |
| ARGUS original: 4 layers, 1 step | 4 | baseline |

**For B4/B5 (2-TTT variants):** Standard single-step TTT on GQA layers 8 and 16. Layer 8 = mid-document adaptation. Layer 16 = final refinement.

---

## Configuration (LFM2-aligned at ~175M)

| Parameter | Value | LFM2-350M | Notes |
|-----------|-------|-----------|-------|
| d_model | 768 | 1024 | Scaled ½ for ~175M param budget |
| n_layers | 16 | 16 | Same |
| d_conv | 768 | 1024 | Full-width conv (= d_model) |
| ffn_inner | 2816 (3.7×) | 4608 (4.5×) | Scaled, closer to LFM2's big FFN ratio |
| conv_kernel | 3 | 3 | Same |
| n_heads | 12 | 16 | head_dim=64, scaled with d |
| n_kv_heads | 8 | 8 | Matches LFM2 exactly |
| head_dim | 64 | 64 | Same |
| gqa_layers | {3,6,8,10,13,16} | 6 of 16 | 10:6 conv:GQA ratio like LFM2 |
| QK-Norm | YES | YES | Scaling-invariant Q/K normalization |
| ttt_chunk | 512 | — | Larger than ARGUS (256) for speed |
| ttt_lr | 0.01 (fixed) | — | Conservative for from-scratch |
| d_film | 64 | — | FiLM fingerprint dimension (our addition) |
| film_start | layer 8 (1-indexed) | — | 3rd GQA layer = natural fingerprint point |
| momentum β init | 0.5 | — | Learned via sigmoid (our addition) |
| vocab_size | 50257 | 65536 | We keep tiktoken GPT-2 |
| max_seq_len | 1024 | 4096+ | Limited by BabyLM for now |

---

## Parameter Count

| Component | Params | Notes |
|-----------|--------|-------|
| Embedding (50257×768, tied) | 38.6M | Tied with LM head |
| 10 ShortConvBlocks | ~90.6M | Conv(768) + SwiGLU(2816) per block: ~9.06M/layer |
| 6 GQABlocks (no TTT) | ~52.7M | Attn(12h/8kv) + SwiGLU(2816) per block: ~8.78M/layer |
| +1 TTTSwiGLU overhead (B0-B3) | +1.2M | ttt_proj(768,768) + ttt_conv(768,5) on layer 16 |
| +2 TTTSwiGLU overhead (B4/B5) | +2.4M | Layers 8 + 16 |
| FiLM (B1/B3/B5) | +0.8M | context_proj(768,64) + 8×gamma/beta(64,768) |
| Momentum log_beta | 1 | Single scalar |
| QK-Norm scales | ~1.5K | Per-head scaling factors |
| RoPE freqs_cis | — | Buffer, not parameter |

| Variant | Total Params | vs 175M target |
|---------|-------------|----------------|
| B0 (1 TTT) | ~183M | +4.6% |
| B1 (1 TTT + FiLM) | ~184M | +5.1% |
| B2 (1 TTT multi-step) | ~183M | +4.6% |
| B3 (1 TTT multi-step + FiLM) | ~184M | +5.1% |
| B4 (2 TTT) | ~184M | +5.1% |
| B5 (2 TTT + FiLM) | ~185M | +5.7% |

All within ~175-185M range. Multi-step TTT adds zero extra parameters — same weights, iterated more. Close to our ~170M test harness baselines for fair comparison.

---

## Throughput Estimates

### Why ARGUS-PRIME Should Be Much Faster Than ARGUS

| Factor | ARGUS (15.3K) | ARGUS-PRIME B0 |
|--------|--------------|----------------|
| Scan ops | None (good) | None (same) |
| Engram | Hash + gate + proj | **REMOVED** |
| MatFormer | Branched FFN | **REMOVED** |
| d_model | 768 | **768** (same, but bigger FFN: 2816 vs 2048) |
| GQA layers | 4 | **6** (LFM2-aligned, more global context) |
| Attention | SDPA | **hybrid_flash_sdpa + QK-Norm** (-8.9% + stability) |
| TTT layers | 4 (heavy bwd) | **1** (layer 16 only — THE sniper) |

**The key insight:** ShortConv + GQA attention = ALL single-kernel ops. No scan fragmentation. torch.compile should achieve **LlamaModel-tier fusion** (3x+ boost). With 6 GQA layers (vs 4), we get more global context. With only 1 TTT layer (vs 4), the backward bottleneck is dramatically reduced. Bigger FFN (2816 vs 2048) adds capacity where it matters.

**Key difference from ARGUS:** 6 attention layers means more compute per forward pass BUT also means the model can attend globally every ~2.5 layers instead of every 4. LFM2 proved this ratio works — and hybrid_flash_sdpa makes each attention layer only 3.5ms.

LlamaModel at 124M gets 47.8K tok/s with compile. ARGUS-PRIME at ~183M has ~1.5× more params. Scaling: 47.8K / 1.5 ≈ **~32K tok/s** theoretical ceiling. But 6 GQA layers add ~21ms (6×3.5ms), and FFN is bigger. Conservative:

| Variant | Est. tok/s | vs AMADEUS (13.2K) | vs Tempest (22.3K) |
|---------|-----------|-------------------|-------------------|
| B0 (1 TTT single) | 20-24K | **1.5-1.8×** | **0.9-1.1×** |
| B1 (1 TTT + FiLM) | 19-23K | 1.4-1.7× | 0.9-1.0× |
| B2 (1 TTT multi-step) | 19-23K | 1.4-1.7× | 0.9-1.0× |
| B3 (1 TTT multi + FiLM) | 18-22K | 1.4-1.7× | 0.8-1.0× |
| B4 (2 TTT) | 18-22K | 1.4-1.7× | 0.8-1.0× |
| B5 (2 TTT + FiLM) | 17-21K | 1.3-1.6× | 0.8-0.9× |

**Note:** The extra 2 GQA layers (6 vs 4) reduce throughput vs our previous d=1024 estimate. But they add SIGNIFICANT quality — LFM2 proved the 10:6 ratio. The trade is: ~2-4K tok/s for much better global context coverage.

### 45-Minute Token Budget

| Variant | Est. tok/s | Tokens in 45 min |
|---------|-----------|-----------------|
| B0 | ~22K | **59M** (3.7 BabyLM epochs) |
| B3 | ~20K | 54M (3.4 epochs) |
| AMADEUS | 13.2K | 35.6M (2.2 epochs) |
| Tempest | 22.3K | 60M (3.7 epochs) |

Even the heaviest variant (B5) sees **1.3× more tokens than AMADEUS** in the same wall-clock budget.

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Compile Story

This is where ARGUS-PRIME should SHINE vs Griffin-based architectures:

| Op Type | Compile behavior | ARGUS-PRIME | Tempest |
|---------|-----------------|-------------|---------|
| GatedConv | Single conv1d kernel → Inductor fuses | 10 layers | 16 layers |
| GQA Attention | SDPA = 1 kernel | 6 layers | 0 layers |
| Griffin scan | ~15 ops, fragments graph | **0 layers** | 16 layers |
| RMSNorm | Element-wise → Inductor fuses | 32 (inlined) | 32 (inlined) |
| SwiGLU | silu+mul+matmul → Inductor fuses | 12-16 | 16 |
| TTTSwiGLU | einsum chains, harder to fuse | 0-4 | 0 |

**No scan = no compile fragmentation.** ARGUS-PRIME A0 should compile as cleanly as LlamaModel.

### Kernel Reuse

| Kernel | Speedup | Where |
|--------|---------|-------|
| fused_residual_add_rmsnorm | 6.6x | All 16 layers (autokernel) |
| silu_gate_mul | 1.6x | Standard SwiGLU layers (autokernel) |
| causal-conv1d | 10x | 10 GatedConv layers |
| hybrid_flash_sdpa_attention | 8.9% vs SDPA | 6 GQA layers |
| cross_entropy / chunked_linear_CE | 1.8x / mem savings | Loss |
| rotary_embedding | 3.7x | 6 GQA layers (autokernel) |

### L2 Cache (TTT)

| Variant | ΔW per layer | L2 fit? |
|---------|-------------|---------|
| B0-B3 (1 TTT at layer 16) | 2560×1024 = 5.2MB fp16 | **YES** — single ΔW, hot only at layer 16 |
| B4/B5 (2 TTT at layers 8,16) | 5.2MB each, sequential | **YES** — only 1 hot at a time |

With only 1 TTT layer, the ΔW matrix occupies L2 for just 1/16 of the forward pass. The other 15 layers have FULL L2 available for normal compute.

### Backward Pass Optimization

The profiling showed 70.8% backward for ARGUS. ARGUS-PRIME attacks this from every angle:

| Change | Backward impact |
|--------|----------------|
| Remove Engram | Eliminates hash+gate backward |
| Remove MatFormer | Eliminates branching backward |
| TTT 4→1 layers (B0-B3) | **75% less TTT backward** |
| TTT 4→2 layers (B4/B5) | 50% less TTT backward |
| hybrid_flash_sdpa | Faster attention backward |
| d=1024 | Better Tensile utilization in backward GEMMs |

**B0 should have backward ≈ 58-62%** — approaching LlamaModel's profile (vs ARGUS's 70.8%). The 1 TTT layer adds minimal backward cost.

---

## Training Protocol

### Shared Settings (All 6 Variants)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (fused=True) |
| LR | 8e-4 → 8e-5 cosine, 100 warmup |
| Batch | 16 × 256 = 4096 tokens/step |
| Precision | fp16 mixed (AMP + GradScaler) |
| Grad clip | 1.0 |
| Dataset | BabyLM (~16.5M tokens) |
| Epochs | 2 (same as hypothesis build-out) |
| Eval | 90/10 train/val split |

### Variant-Specific

| Parameter | B0 | B1 | B2 | B3 | B4 | B5 |
|-----------|----|----|----|----|----|----|
| TTT layers | 16 | 16 | 16 | 16 | 8,16 | 8,16 |
| TTT steps | 1 | 1 | **3** | **3** | 1 | 1 |
| FiLM | — | YES | — | YES | — | YES |
| ttt_lr | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| d_film | — | 64 | — | 64 | — | 64 |

### Running the Ablation

Each variant is a single flag configuration on the SAME model file:

```bash
# B0: 1 TTT single-step (sniper, no support)
python -m halo_training --model models/argus_prime.py --class-name ArgusPrime \
    --compile --optimize-kernels --dataset babylm

# B1: 1 TTT single-step + FiLM (sniper with air support)
python -m halo_training --model models/argus_prime.py --class-name ArgusPrimeFiLM \
    --compile --optimize-kernels --dataset babylm

# B2: 1 TTT multi-step (deep sniper)
python -m halo_training --model models/argus_prime.py --class-name ArgusPrimeMulti \
    --compile --optimize-kernels --dataset babylm

# B3: 1 TTT multi-step + FiLM (THE HYPOTHESIS: deep sniper + air support)
python -m halo_training --model models/argus_prime.py --class-name ArgusPrimeMultiFiLM \
    --compile --optimize-kernels --dataset babylm

# B4: 2 TTT single-step (bracket)
python -m halo_training --model models/argus_prime.py --class-name ArgusPrimeTTT2 \
    --compile --optimize-kernels --dataset babylm

# B5: 2 TTT single-step + FiLM (bracket + air support)
python -m halo_training --model models/argus_prime.py --class-name ArgusPrimeTTT2FiLM \
    --compile --optimize-kernels --dataset babylm
```

**Time budget:** 6 variants × ~30-45 min each = 3-4.5 hours total. Can parallelize 2 at a time (each ~4-5GB, total ~10GB of 116GB).

**Priority pair:** Run **B3** (the hypothesis) and **B0** (the minimum) first. B3 tests the full sniper+FiLM idea. B0 tests whether a single minimal TTT at layer 16 is enough on its own.

---

## Ablation Decision Tree

```
Start: Run B0 (1 TTT minimal) and B3 (1 TTT multi + FiLM) first
  │
  ├─ If B3 quality > B0 by >2%:
  │   → Multi-step + FiLM helps! Run B1 and B2 to isolate.
  │   ├─ If B1 ≈ B3: FiLM is doing the work, multi-step is noise → ship B1
  │   ├─ If B2 ≈ B3: Multi-step is doing the work, FiLM is noise → ship B2
  │   └─ If B3 > both B1 and B2: SYNERGY! Ship B3.
  │
  ├─ If B0 ≈ B3 (1 TTT is enough either way):
  │   → Single-step is sufficient. Run B1 to check if FiLM helps.
  │   → Ship B0 (simplest) or B1 (if FiLM helps) at ~25K tok/s.
  │
  └─ If both B0 and B3 lose to AMADEUS (2.90):
      → Run B4/B5 (2 TTT layers) to check if MORE TTT helps.
      → If B5 > B3: The bracket (2 TTT) is better than 1 deep sniper.
      → If still losing: ShortConv/GQA skeleton at ~197M may need
        more params (scale to 244M) or longer sequences.
```

**Priority pair:** B0 first (the minimum), B3 second (the hypothesis). Only run B1/B2/B4/B5 if needed to isolate variables.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| d=1024 makes model too big | LOW | ~195-205M, well under 250M budget |
| 4 GQA layers too expensive | MEDIUM | hybrid_flash_sdpa = 3.5ms each. 4 × 3.5ms = 14ms on ~100ms forward. ~14% overhead. Acceptable for global context. |
| TTT backward still dominates | LOW | Only 1 TTT layer (75% less backward than ARGUS). Multi-step adds compute but zero extra autograd nodes (inner loop is detached). |
| FiLM fingerprint at GQA layer 8 is worse than AMADEUS's layer 8 | LOW | AMADEUS's layer 8 had SSM context. ARGUS-PRIME layer 8 has GQA (global attention) — arguably RICHER context for fingerprint. |
| ShortConv (no recurrence) can't model long-range without attention | LOW | 4 GQA layers every 4th = global context every 3 local layers. LFM2.5 proves this is sufficient at 350M. |
| Compile doesn't fuse as well as hoped | MEDIUM | Smoke test compile first. ShortConv + SDPA should be fully fusable — no scan to fragment the graph. |

---

## Success Criteria

1. **B0 val loss < 2.98** — minimal sniper beats Tempest (proves the skeleton + 1 TTT works)
2. **Best variant val loss < 2.90** — beats AMADEUS (the quality champion)
3. **B0 throughput > 22K tok/s** — faster than compile-optimized Tempest
4. **B0 throughput > 25K tok/s** — compile-gap closure (no scan fragmentation)
5. **Multi-step ablation:** B2 < B0 (multi-step helps for single layer)
6. **FiLM ablation:** B1 < B0 and/or B3 < B2 (FiLM measurably helps quality)
7. **Sniper vs bracket:** B3 ≈ B5 (1 deep TTT + FiLM matches 2 standard TTT + FiLM)
8. **Backward pass < 63%** of training time for B0 (down from ARGUS's 70.8%)

---

## Implementation Roadmap

1. Create `models/argus_prime.py` — shared base with config flags for TTT/FiLM/multi-step
2. Import `TTTSwiGLU` from `models/argus.py` (already proven)
3. Create `MultiStepTTTSwiGLU` extending TTTSwiGLU (3-step inner loop)
4. Import `FiLMConditioner` from `models/amadeus.py` (already proven)
5. Wire `hybrid_flash_sdpa_attention` into GQA layers
6. Create class variants: ArgusPrime (B0), ArgusPrimeFiLM (B1), ArgusPrimeMulti (B2), ArgusPrimeMultiFiLM (B3), ArgusPrimeTTT2 (B4), ArgusPrimeTTT2FiLM (B5)
7. Smoke test ALL 6 variants (d=128, 4 layers, 200 steps) — MANDATORY
8. Compile test ALL 6 variants
9. Run **B0 + B3** first (priority pair)
10. Analyze results → decision tree → run remaining variants as needed
11. Profile backward pass % for B0 vs ARGUS

---

## The Philosophy

LFM2.5 taught us: **the simplest architecture wins.** No SSM, no Griffin, no complex recurrence. Just convolutions (cheap) + attention (strategic) + good engineering.

ARGUS tried to throw everything at the wall. ARGUS-PRIME asks: **what's the minimum that actually matters?**

The ablation will tell us one of three stories:
- **B0 is enough:** One minimal TTT sniper at layer 16 is all you need. Ship at ~25K tok/s.
- **B3 wins:** The deep sniper + FiLM air support is the Michelin recipe. Ship at ~23K tok/s.
- **Nothing beats AMADEUS:** The SSM hybrid genuinely captures something convolutions + sparse attention can't. Go back to NEXUS.

Every outcome is a win. We either find a faster architecture OR we learn that AMADEUS's quality comes from Mamba-3, not just parameter count. Either way, we stop guessing and start KNOWING.

**ARGUS-PRIME: one sniper, one truth.**

---

## References

- LFM2.5-350M (Liquid AI) — 3:1 ShortConv/GQA architecture, target to beat
- ARGUS (our lab) — 156M, 15.3K tok/s, 6 mechanisms, profiled backward = 70.8%
- AMADEUS (our lab) — val 2.90, quality champion, FiLM source
- TEMPEST (our lab) — 22.3K tok/s compile-optimized, throughput baseline
- LAZARUS/LazarusLite (our lab) — TTT proven mechanically, 2 layers sufficient
- In-Place TTT (arXiv 2604.06169) — ByteDance cumsum approach
- Compile-Optimized Griffin (our lab) — FusedGriffinBlock 3.52x, inline momentum pattern
