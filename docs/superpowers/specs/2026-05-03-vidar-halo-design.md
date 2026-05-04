# VIDAR-HALO Design Spec

**Date:** 2026-05-03
**Status:** ACTIVE
**Replaces:** TyrHaloLight as primary training target on Strix Halo

## Summary

VIDAR-HALO is a lean looped hybrid LM targeting ~25K tok/s on Strix Halo with ~47.5M unique / ~95M effective parameters. Key insight: d=768 dominates quality (GRIFFIN-HALO sweep: loss 3.2 vs 6.1 at d=512), Parcae loop ×2 dominates param efficiency (TyrHaloLight BPB 2.10 beat flat BALDR 2.75). By dropping momentum (recovers 22% throughput), using fewer but wider layers, and training a custom 32K tokenizer, VIDAR-HALO maximizes quality per FLOP on bandwidth-limited hardware.

## Design Decisions and Evidence

| Decision | Evidence | Source |
|----------|----------|--------|
| d=768 | 3.2 vs 6.1 loss (d=768 vs d=512) | GRIFFIN-HALO 10-run sweep |
| Loop ×2 | BPB 2.10 (58M looped) beat 2.75 (118M flat) | Epoch 1 race, stem-crawl-solo |
| No momentum | Recovers 22% throughput, MoDA+loop_pos carry cross-iter info | Velocity clamp profiling |
| Custom 32K tokenizer | +9% compression, −15% embed params → reinvest in depth | GPT-X2-125M (27× data efficiency) |
| WSD schedule | Beats cosine at small scale | GPT-X2-125M |
| z-loss early | Prevents logit drift during high-LR warmup | GPT-X2-125M |
| XSA | -0.9% loss, 0 params, 0 compute | Zhai 2026, our ablation |
| EMA 0.999 | +7.5% generalization, 0 throughput cost | TRM (Samsung SAIL) |
| Identity init for conv/gates | Free stability, smoother startup | GenDistill/KDA (Huawei) |
| No HC streams | 35-41% throughput cost on 240 GB/s bandwidth | Strix Halo measurement |
| Full backprop (not 1-step) | 1-step is −30.9% on recursive models | TRM ablation |
| Unrolled forward | Compile-friendly, no .item()/Poisson/branches | TyrHaloLight proven |
| 4 layers (not 6) | Budget-constrained by d=768 + 50M unique target | FLOP math |

## Architecture

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| vocab_size | 32000 | Custom BPE trained on dolma-10b |
| d_model | 768 | Dominant quality factor |
| embed_rank | 384 | Factorized embedding (saves ~14M vs full) |
| n_shared_layers | 4 | 3 ShortConv + 1 MoDA-GQA |
| mean_recurrence | 2 | Deterministic, unrolled |
| backprop_depth | 2 | Full backprop through all iterations |
| n_heads | 12 | head_dim = 64 |
| n_kv_heads | 4 | 3:1 GQA ratio |
| ffn_inner | 2816 | 3.67× d, padded to 128-multiple for Tensile tiles |
| d_conv | 640 | 0.83× d, gated conv channel dim |
| conv_kernel | 3 | Causal conv1d |

### Layer Layout (shared, iterated ×2)

```
Position 0: ShortConvBlock  — local pattern extraction
Position 1: ShortConvBlock  — local refinement
Position 2: MoDA-GQA        — global attention + cross-iteration depth KVs
Position 3: ShortConvBlock  — post-attention mixing
```

GQA at position 2 (center): depth KVs from iteration 0 flow into iteration 1 midway through each pass. Prepend depth KVs (not append) for flash-compatible `is_causal=True`.

### Parameter Budget

```
Component                          Params
─────────────────────────────────────────
FactorizedEmbedding (32K×384)     12.29M
  proj_up (384×768)                0.30M
LM Head proj_down (768×384, tied)  0.30M
ShortConvBlock ×3                 25.47M
  conv.proj: 768×(3×640)           1.47M each
  out_proj: 640×768                0.49M each
  SwiGLU: 768×5632 + 2816×768     6.34M each
  norms                            <0.01M each
MoDA-GQA ×1                       8.54M
  wq: 768×768                      0.59M
  wk: 768×256                      0.20M
  wv: 768×256                      0.20M
  wo: 768×768                      0.59M
  depth_kv_proj: 768×512            0.39M
  SwiGLU: same                     6.34M
  norms                            <0.01M
MTP head (768×384)                 0.30M
Loop pos embeds (2×768)            <0.01M
iter_norm (RMSNorm)                <0.01M
─────────────────────────────────────────
UNIQUE TOTAL:                     ~47.5M
EFFECTIVE (×2 iterations):        ~95.0M
```

### Forward Pass (Unrolled)

```python
def forward(x, targets):
    h = embed(x)                              # (B, T) → (B, T, 768)
    freqs = precompute_freqs(T)
    depth_kv_buffer = []
    
    # Iteration 0
    h = parcae_inject(h, input_embed=h)       # skip on iter 0 (A+B=0)
    h, depth_kvs_0 = shared_block(h, freqs, depth_kvs=[])
    h = iter_norm(h) + loop_pos_embeds[0]
    depth_kv_buffer.append(depth_kvs_0)
    
    # Iteration 1
    h = parcae_inject(h, input_embed=embed(x)) # re-inject backbone
    h, depth_kvs_1 = shared_block(h, freqs, depth_kvs=depth_kv_buffer)
    h = iter_norm(h) + loop_pos_embeds[1]
    
    logits = lm_head(h)
    loss = cross_entropy(logits, targets)
    
    # z-loss (first 40% of training)
    if step < 0.4 * total_steps:
        loss += 1e-4 * logits.float().pow(2).mean()
    
    # MTP auxiliary loss
    if training:
        mtp_loss = mtp_head(h, targets)
        loss += 0.1 * mtp_loss
    
    return loss
```

### Features Enabled

| Feature | Params | Compute | Source |
|---------|--------|---------|--------|
| XSA (exclusive self-attention) | 0 | 0 | Zhai 2026 |
| MoDA depth-attention | 0.39M | 1 extra KV concat | MoDA paper |
| Loop position embeddings | 1.5K | 1 add per iter | Hyperloop |
| MTP depth=1 | 0.30M | Training only | TyrHaloLight |
| EMA 0.999 | 0 (shadow copy) | 0 throughput | TRM |
| Identity init conv/gates | 0 | 0 | GenDistill |

### Features Disabled

| Feature | Why Off | Throughput Saved |
|---------|---------|-----------------|
| Momentum/velocity | MoDA + loop pos embeds carry cross-iter info | +22% |
| HC streams | 35-41% cost on 240 GB/s bandwidth | +35-41% |
| Prelude/Coda | Saves ~8M params + latency | +10-15% |

## Tokenizer

Train custom 32K BPE tokenizer on dolma-10b using `tokenizers` library (HuggingFace).

**Spec:**
- Vocab size: 32000
- Algorithm: BPE (byte-fallback)
- Training data: dolma-10b (sample ~1B tokens for tokenizer training)
- Special tokens: `<|endoftext|>` (EOS, id=0), `<|pad|>` (id=1)
- Normalization: NFC unicode, no lowercasing
- Pre-tokenizer: byte-level with whitespace splitting (GPT-2 style)

**Why:** GPT-X2-125M showed custom 32K BPE trained on domain data gives 9% compression improvement over GPT-2 50K. At our scale, saves ~10M embed params reinvested in transformer depth.

**Implementation:** `scripts/train_tokenizer.py` — reads dolma-10b JSONL, trains BPE, saves to `tokenizers/vidar-32k/`. Takes ~8 min (2M docs).

**Results (2026-05-04):** Tokenizer trained on dolma-10b-sample (19.5M docs). Pretokenized to `datasets/dolma-10b-vidar32k.bin` (6.9B tokens, 13.8 GB) using `scripts/pretokenize.py --workers 14`.

**Compression comparison vs GPT-2:**
- General text: identical token count
- Code: **-33%** tokens (dolma code data improves BPE merges)
- Technical jargon: +11% (rare terms split more with smaller vocab)
- **Overall: -12.3% fewer tokens** across test suite
- Vocab reduction: 50,257 → 32,000 (-36%) → embedding layer 36% smaller

## Training Recipe

### Schedule: WSD (Warmup-Stable-Decay)

```
Phase 1: Warmup     (steps 0 → 300)      LR: 0 → 0.002
Phase 2: Stable     (steps 300 → 80%)     LR: 0.002 (constant)
Phase 3: Decay      (80% → 100%)          LR: 0.002 → 0 (linear)
```

### Optimizer

- **Muon** for 2D params (Newton-Schulz in fp32)
- **AdamW** for 1D params (betas=0.9/0.95, fused=True)
- Weight decay: 0.1 during stable → 0.01 during decay phase
- Grad clip: 1.0

### Precision

- fp16 + GradScaler(init_scale=1024, backoff_factor=0.25)
- No velocity clamp needed (no momentum)

### Progressive Context Scheduling

| Phase | Context | When | Rationale |
|-------|---------|------|-----------|
| 1 | 256 | First 20% | Fast convergence on short sequences |
| 2 | 512 | 20-60% | Intermediate context |
| 3 | 1024 | 60-100% | Full target context |

Batch size adjusted to maintain ~constant tokens/batch.

### Data

- **Tokenizer training:** dolma-10b sample (~1B tokens)
- **Pretraining:** dolma-10b (pre-tokenized to .bin with custom tokenizer)
- **Progressive curriculum** (future, when scaling beyond dolma-10b):
  - Start: general web text dominant
  - Stable phase: ramp math/code
  - Decay phase: taper specialized data

### z-loss

```python
if step < 0.4 * total_steps:
    z_loss = 1e-4 * logits.float().pow(2).mean()
    loss += z_loss
```

Prevents logit magnitude drift during high-LR warmup. Disable after 40% to avoid constraining fine-grained convergence.

## Target Performance

| Metric | Target | Basis |
|--------|--------|-------|
| tok/s | ~25K | TyrHaloLight 18K + 22% (no velocity clamp) + d=768 GEMM scaling |
| BPB | < 2.10 | Beat TyrHaloLight (stem-crawl-solo 1-epoch) |
| Memory | < 40 GB | Fits single Strix Halo (128 GB unified) |
| Compile | Full model, one graph | Unrolled forward, no Python loops |

## Implementation

### New files
- `models/vidar_halo.py` — model definition
- `scripts/train_tokenizer.py` — BPE tokenizer training
- `tokenizers/vidar-32k/` — trained tokenizer artifacts

### Modified files
- `halo_training/cli.py` — WSD schedule, z-loss, progressive context, weight decay annealing
- `halo_training/trainer.py` — WSD scheduler, z-loss integration, context scheduling
- `halo_training/data.py` — custom tokenizer loading, context length switching
- `scripts/pretokenize.py` — support custom tokenizer

### Variant classes

```python
class VidarHalo(VidarHaloBase):
    """Default: mean=2, full features."""
    pass

class VidarHaloMini(VidarHaloBase):
    """Smoke test: d=128, 4 layers, ~2M params."""
    pass
```

## Verification Plan

1. **Tokenizer:** Train on dolma-10b, verify compression ratio vs GPT-2 50K
2. **Smoke test:** VidarHaloMini on BabyLM, 100 steps, check loss decreasing
3. **Compile test:** Verify single compile graph (no graph breaks)
4. **Throughput:** Measure tok/s, target ≥25K
5. **1-epoch race:** stem-crawl-solo, compare BPB vs TyrHaloLight (2.10)
6. **Generation eval:** Sample text at checkpoints (GenDistill warning: don't trust BPB alone)

## Relationship to Other Architectures

```
TyrHaloLight (58.5M, 18K tok/s, BPB 2.10)
  → Drop momentum (+22% speed)
  → Widen d=640→768 (quality dominant factor)
  → Fewer layers 6→4 (budget-constrained)
  → Custom 32K tokenizer (saves embed params)
  → WSD + z-loss + progressive context (GPT-X2 recipe)
  = VIDAR-HALO (47.5M unique, ~25K tok/s target)

BALDR-HALO (118M flat, 20K tok/s, BPB 2.75) — retired, loop wins
CHIMERA-HALO (94M/158M) — retired, too complex
FENRIR-HALO (80M/160M) — retired, replaced by TYR-HALO
```
