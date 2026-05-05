# ODIN-HALO Design Spec

**Date:** 2026-05-04
**Status:** APPROVED
**Replaces:** VIDAR-HALO as next architecture target
**Purpose:** ~123M effective param general base model, architecturally pre-shaped for BTX/BAR MoE forking

---

## Summary

ODIN-HALO is a 58M unique / ~156M effective param looped hybrid LM. It synthesizes all validated findings from this project's 30+ architecture experiments, 20+ paper deep-dives, and parameter-golf analysis into a single design optimized for:

1. **Quality per FLOP** on Strix Halo (bandwidth-limited, no MFMA)
2. **Future MoE composability** via BTX/BAR domain expert forking
3. **Length generalization** via HyPE positional encoding

Key architectural decisions: d=768 (quality-dominant), ×3 Parcae iterations (TRM/param-golf sweet spot), 5 shared layers (4C+1A, LFM2.5-aligned), HyPE (NoPE attention, RoPE conv), aggressive embed factorization (rank=256, maximizes transformer budget).

---

## Design Decisions and Evidence

| Decision | Value | Evidence | Source |
|----------|-------|----------|--------|
| d=768 | Quality-dominant factor | 3.2 vs 6.1 loss (d=768 vs d=512) | GRIFFIN 10-run sweep |
| ×3 iterations | Effective depth sweet spot | param-golf ×3 on layers 3-5; TRM "2L+more recursions > 4L+fewer" | Parameter-golf PR#1855, TRM (Samsung) |
| 4C+1A layout | Conv-heavy, minimal attention | LFM2.5 HW search: conv beats SSMs on Ryzen AI Max+ | Liquid AI tech report |
| GQA at center | Balanced local→global→local flow | GenDistill: mid-to-late attention placement best | GenDistill beam search, Vidar proven |
| HyPE (NoPE attn) | Length generalization from day 1 | 99.8% NIAH at 256K from 4K training | HypeNet ICML 2026 |
| Conv kernel=3 | Smallest sufficient receptive field | LFM2.5 uses k=3 across all layers | Liquid AI tech report |
| Embed rank=256 | Maximize transformer budget | 8.6M embed vs 12.6M at rank=384, saves 4M for FFN | GPT-X2 insight: depth > embeddings |
| Custom 32K tokenizer | -12% tokens, -36% vocab vs GPT-2 | Pre-trained on dolma-10b, pretokenized .bin exists | Vidar tokenizer (already trained) |
| Polar-Express NS | Free optimizer upgrade | 4× better orthogonalization, zero extra FLOP | Parameter-golf #1855, arXiv:2505.16932 |
| WSD schedule | Beats cosine at small scale | GPT-X2-125M matched SmolLM2 with 27× fewer tokens | GPT-X2-125M |
| MIN_LR = 10% | Don't decay to zero | Convergent: DSV4, GPT-X2, Poolside, param-golf | 4 independent validations |
| EMA 0.999 | +7.5% generalization, zero cost | TRM ablation: −7.5% without EMA | TRM (Samsung SAIL) |
| MoDA depth-attention | Cross-iteration info via attention | +2.11% quality in TyrHaloLight | MoDA paper + our ablation |
| Iter skip connections | Direct feature bypass between iterations | U-Net pattern from param-golf, zero-init = safe | Parameter-golf + BAR skip pattern |
| No momentum/velocity | Recovers throughput, MoDA+loop_pos carry info | +22% tok/s, no quality loss with MoDA | TyrHaloLight → Vidar lesson |
| Logit softcap=30 | fp16 stability, bounds output logits | Gemma2 validated, complements z-loss | Gemma2 + parameter-golf |
| Identity init | Free training stability | GenDistill: smoother startup for conv/gates | GenDistill/KDA (Huawei) |

---

## Architecture

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| vocab_size | 32768 | Custom BPE trained on dolma-10b |
| d_model | 768 | Quality-dominant factor |
| embed_rank | 256 | Aggressive factorization (saves 4M vs rank=384) |
| n_shared_layers | 6 | 5 ShortConv + 1 MoDA-GQA |
| gqa_positions | (3,) | Center position: Conv→Conv→Conv→GQA→Conv→Conv |
| mean_recurrence | 3 | Deterministic, unrolled, full backprop |
| backprop_depth | 3 | Full gradient through all iterations |
| n_heads | 12 | head_dim = 64 |
| n_kv_heads | 4 | 3:1 GQA ratio |
| ffn_inner | 2816 | 3.67× d, padded to 256-multiple for Tensile tiles |
| d_conv | 512 | 0.67× d, autokernel-safe (>256) |
| conv_kernel | 3 | Causal conv1d (LFM2.5 validated) |
| max_seq_len | 2048 | Training context (HyPE enables inference beyond) |

### Layer Layout (shared, iterated ×3)

```
Position 0: ShortConvBlock  — local pattern extraction (RoPE in gate)
Position 1: ShortConvBlock  — local refinement (RoPE in gate)
Position 2: ShortConvBlock  — deeper local features (RoPE in gate)
Position 3: MoDA-GQA        — global attention (NoPE) + cross-iteration depth KVs
Position 4: ShortConvBlock  — post-attention mixing (RoPE in gate)
Position 5: ShortConvBlock  — final local processing (RoPE in gate)
```

6 layers × 3 iterations = **18 effective layers.** More depth validated by LFM2 (16L at 350M) and Baguettotron (80L at 321M).

### HyPE Positional Encoding

| Component | Positional Strategy | Rationale |
|-----------|-------------------|-----------|
| ShortConv blocks | RoPE applied to conv gate signal | Local positional awareness for sequential processing |
| GQA block | **NoPE** (no positional encoding on Q/K) | Content-only attention → length generalization |
| QK-Norm | Mandatory (RMSNorm per head before attention) | Prevents exploding logits without positional anchor |
| MoDA depth KVs | No RoPE (positional info from source hidden state) | Cross-iteration KVs should be position-agnostic |

### Parameter Budget

```
Component                          Params
─────────────────────────────────────────
FactorizedEmbedding (32768×256)    8.39M
  proj_up (256×768)                0.20M
LM Head proj_down (768×256, tied)  0.20M
ShortConvBlock ×5                 40.40M
  conv.proj: 768×(3×512)           1.18M each
  out_proj: 512×768                0.39M each
  SwiGLU: 768×5632 + 2816×768     6.48M each
  norms                            <0.01M each
MoDA-GQA ×1                       8.47M
  wq: 768×768                      0.59M
  wk: 768×256                      0.20M
  wv: 768×256                      0.20M
  wo: 768×768                      0.59M
  depth_kv_proj: 768×512           0.39M
  SwiGLU: same                     6.48M
  norms                            <0.01M
MTP head (768×256)                 0.20M
Parcae injection                   ~1.2K
Loop pos embeds (3×768)            2.3K
Iter scales (3)                    3
Skip gates (2×768)                 1.5K
iter_norm + final_norm             ~1.5K
─────────────────────────────────────────
UNIQUE TOTAL:                     ~58.1M
ITERATED (6 layers × 3):         ~146.6M
NON-ITERATED (embed/head/mtp):   ~9.2M
FULL EFFECTIVE:                   ~155.8M
```

### Forward Pass (Unrolled, 3 Iterations)

```python
def forward(x):
    h = embed(x)                                    # (B, T) → (B, T, 768)
    freqs = precompute_freqs(T)                     # For RoPE in conv blocks
    input_embed = h
    depth_kv_buffer = []
    
    # Iteration 0 (no injection on first iter — Parcae A+B=0)
    h, kvs_0 = shared_block(h, freqs, depth_kvs=[])
    h = iter_norm(h) * iter_scales[0] + loop_pos_embeds[0]
    h0_skip = h                                     # save for skip connection
    depth_kv_buffer.append(kvs_0)
    
    # Iteration 1
    h = parcae_inject(h, input_embed)               # re-inject backbone
    h = h + sigmoid(skip_gate_0) * h0_skip          # iter skip 0→1
    h, kvs_1 = shared_block(h, freqs, depth_kvs=depth_kv_buffer)
    h = iter_norm(h) * iter_scales[1] + loop_pos_embeds[1]
    h1_skip = h                                     # save for skip connection
    depth_kv_buffer.append(kvs_1)
    
    # Iteration 2
    h = parcae_inject(h, input_embed)               # re-inject backbone
    h = h + sigmoid(skip_gate_1) * h1_skip          # iter skip 1→2
    h, kvs_2 = shared_block(h, freqs, depth_kvs=depth_kv_buffer)
    h = iter_norm(h) * iter_scales[2] + loop_pos_embeds[2]
    
    # Output
    logits = lm_head(final_norm(h))
    logits = 30.0 * tanh(logits / 30.0)            # softcap
    
    # z-loss (first 40% of training)
    if training and step < 0.4 * total_steps:
        loss += 1e-4 * logits.float().pow(2).mean()
    
    # MTP (training only)
    if training:
        mtp_loss = mtp_head(h, targets)
        loss += 0.1 * mtp_loss
    
    return logits
```

### Features Enabled (baked in)

| Feature | Params | Compute | Source |
|---------|--------|---------|--------|
| HyPE (RoPE conv, NoPE attn) | 0 | 0 | HypeNet ICML 2026 |
| XSA (exclusive self-attention) | 0 | 0 | Zhai 2026 |
| MoDA depth-attention | 0.39M | 1 extra KV concat per iter | MoDA paper |
| Iteration skip connections | 1.5K | 2 gated adds | Parameter-golf U-Net pattern |
| Loop position embeddings | 2.3K | 1 add per iter | Hyperloop |
| Per-iteration learned scales | 3 | 1 mul per iter | Parameter-golf layerwise scale |
| Logit softcap=30 | 0 | 1 tanh | Gemma2 |
| MTP depth=1 | 0.20M | Training only | DeepSeek-V4 |
| EMA 0.999 | 0 (shadow copy) | 0 throughput | TRM |
| Identity init (conv/gates) | 0 | 0 | GenDistill |
| Polar-Express NS (Muon) | 0 | 0 (same FLOP, better coefficients) | arXiv:2505.16932 |

### Features Disabled

| Feature | Why Off | Throughput Saved |
|---------|---------|-----------------|
| Momentum/velocity | MoDA + loop pos + skip carry cross-iter info | +22% |
| HC streams (Hyperloop) | 35-41% cost on 240 GB/s bandwidth | +35-41% |
| Prelude/Coda blocks | Save params + latency for the loop | +10-15% |
| bf16 | 24% slower on gfx1151, compile crashes | Use fp16+GradScaler |

### Features for Ablation (not baked in)

| Feature | Why Ablate | Expected Impact |
|---------|-----------|-----------------|
| Delayed recurrence (flat → ×3 at 25-35%) | High impact but activation timing needs tuning | −0.005 to −0.015 BPB |
| Parallel residuals (2-lane GQA) | Throughput impact unknown at d=768 | −0.003 to −0.010 BPB |
| Curriculum warm-start (easy→hard data) | Needs difficulty scoring infra | −0.01 to −0.03 BPB |

---

## Training Recipe

### Schedule: WSD (Warmup-Stable-Decay)

```
Phase 1: Warmup     (steps 0 → 300)      LR: 0 → 0.002
Phase 2: Stable     (steps 300 → 80%)     LR: 0.002 (constant)
Phase 3: Decay      (80% → 100%)          LR: 0.002 → 0.0002 (linear, MIN_LR=10%)
```

### Optimizer

- **Polar-Express Muon** for 2D params (5-step minimax NS in fp32)
- **AdamW** for 1D params (betas=0.9/0.95, fused=True)
- Weight decay: 0.1 stable → 0.01 decay phase (WD annealing)
- Grad clip: 1.0

### Polar-Express NS Coefficients

```python
_PE_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
```

### Precision

- fp16 + GradScaler(init_scale=1024, backoff_factor=0.25)
- No velocity clamp needed (no momentum)
- NS iterations in fp32 (not bf16 — hardware constraint)

### Data

- **Tokenizer:** Custom 32K BPE (`tokenizers/vidar-32k/`, already trained on dolma-10b)
- **Pretraining:** dolma-10b-vidar32k.bin (6.9B tokens, 13.8 GB, already pretokenized on Machine A)
- **Progressive context scheduling:**
  - 0-20%: seq_len=256
  - 20-60%: seq_len=512
  - 60-100%: seq_len=1024

### z-loss

```python
if step < 0.4 * total_steps:
    z_loss = 1e-4 * logits.float().pow(2).mean()
    loss += z_loss
```

---

## Target Performance

| Metric | Target | Basis |
|--------|--------|-------|
| tok/s (compiled) | ~15-18K | d=768, 6L×3 iterations, heavier than Vidar (4L×2) |
| BPB (stem-crawl 1ep) | < 1.95 | Beat TyrHaloLight (2.10), +50% effective depth |
| BPB (dolma-10b 1ep) | < 1.80 | Scaling with 13× more data |
| Memory | < 50 GB | Fits single Strix Halo (128 GB unified) |
| Compile | Full model, one graph | Unrolled forward, no Python loops |

---

## Future MoE Path (BTX/BAR)

ODIN-HALO is explicitly designed as the base for domain expert forking:

### Pre-shaped for Option C (Shared Attention + Routed FFN)

| Component | In MoE | Params |
|-----------|--------|--------|
| Embeddings | Shared (averaged across forks) | 8.6M |
| LM Head | Shared (averaged) | 0.2M |
| Attention (GQA block, position 3) | Shared (averaged) | ~2.0M |
| FFN in GQA block | **Expert** (per-domain) | ~6.5M per expert |
| FFN in ShortConv blocks ×5 | **Expert** (per-domain) | ~32.4M per expert |
| Anchor FFN (frozen, post-trained base) | Always active | ~38.9M |

With 4 domain experts + 1 anchor: **~205M total FFN, ~78M active (anchor + 1 expert top-1).**

Full MoE total: ~215M params, ~87M active per token.

### Why this architecture is MoE-friendly

1. **Wide FFN (2816):** Each expert retains 6.5M FFN per layer — enough capacity for domain specialization
2. **Single GQA block:** Only 1 attention layer to average → minimal averaging loss
3. **Conv blocks independent:** 5 ShortConv blocks share no cross-attention — each FFN can specialize independently
4. **HyPE NoPE attention:** Shared attention doesn't depend on position → router decisions don't break positional coherence
5. **Parcae loop + shared routing:** Same expert selected across all 3 iterations (router on prompt) → consistent depth-KVs
6. **6 layers × 5 conv = 30 expert FFNs** when forked — fine-grained domain specialization per layer

---

## Relationship to Prior Architectures

```
TyrHaloLight (58.5M, d=640, 6L×2, BPB 2.10)
  → Widen d=640→768 (quality dominant)
  → Keep 6 layers (depth > width, LFM2/Baguettotron validated)
  → Increase iterations 2→3 (TRM/param-golf sweet spot)
  → Custom 32K tokenizer (saves embed params)
  → HyPE (NoPE attention → length generalization)
  → Aggressive embed rank 384→256 (maximize transformer budget)
  → Polar-Express NS, MIN_LR, iter scales, skip connections, softcap
  = ODIN-HALO (58M unique / ~156M effective)

VIDAR-HALO (47.5M, d=768, 4L×2, target BPB < 2.10)
  → Same 6 layers (matches TyrHaloLight depth), +1 iteration (2→3)
  → Embed rank 384→256 (reallocate to transformer budget)
  → HyPE positional (NoPE attention)
  → Iteration skip connections
  → Polar-Express NS
  = ODIN-HALO (58M unique / ~156M effective)

Future:
  ODIN-HALO (base) → fork → domain experts → BTX/BAR MoE (~215M total, ~87M active)
```

---

## Implementation

### New files
- `models/odin_halo.py` — model definition

### Modified files
- `halo_training/muon.py` — Polar-Express NS coefficients
- `halo_training/trainer.py` — MIN_LR floor in WSD scheduler

### Variant classes

```python
class OdinHalo(OdinHaloBase):
    """Production: d=768, 6 layers × 3 iters, ~58M unique, ~156M effective.
    (48.9M iterated ×3 = 146.6M + 9.2M non-iterated = 155.8M)"""
    pass

class OdinHaloAblation(OdinHaloBase):
    """Ablation variant: d=384, 6 layers, ~20M unique. Tier S screening."""
    def __init__(self, **kw):
        kw.setdefault("d_model", 384)
        kw.setdefault("embed_rank", 128)
        kw.setdefault("n_heads", 6)
        kw.setdefault("n_kv_heads", 2)
        kw.setdefault("ffn_inner", 1408)
        kw.setdefault("d_conv", 320)
        super().__init__(**kw)

class OdinHaloMini(OdinHaloBase):
    """Smoke test: d=128, ~2M params."""
    def __init__(self, **kw):
        kw.setdefault("d_model", 128)
        kw.setdefault("embed_rank", 64)
        kw.setdefault("n_heads", 4)
        kw.setdefault("n_kv_heads", 2)
        kw.setdefault("ffn_inner", 256)
        kw.setdefault("d_conv", 128)
        kw.setdefault("mean_recurrence", 2)
        super().__init__(**kw)
```

---

## Verification Plan

1. **Smoke test:** OdinHaloMini on BabyLM, 100 steps, loss decreasing, no NaN
2. **Param count:** Verify unique=~47M, effective=~123M (37.8M×3 + 9.2M)
3. **Compile test:** Verify single compile graph (no graph breaks) on full OdinHalo
4. **HyPE verification:** Confirm NoPE in GQA (no freqs_cis passed to attention), RoPE in conv gate
5. **Throughput:** Measure tok/s compiled, target ≥18K
6. **Ablation baseline:** Run OdinHaloAblation on BabyLM, establish Tier S baseline
7. **Tier M baseline:** Run OdinHalo on wikitext-103, establish Tier M baseline
8. **Tier V race:** OdinHalo 1-epoch on stem-crawl-solo, compare BPB vs TyrHaloLight (2.10)
9. **Length extrapolation:** Eval at 512, 1024, 2048, 4096 context (HyPE benefit test)
10. **Generation eval:** Sample text at checkpoints (GenDistill warning: don't trust BPB alone)
