# OBSIDIAN

**Evolution of Ternary Reflex: BitNet b1.58 + Griffin + Caveman LFM Genius + Engram**

## Hypothesis

Ternary Reflex proved the L2-resident matmul-free path is a genuine hardware innovation for Strix Halo. It failed due to TRAINING (STE + fp16 + 50K vocab), not design. OBSIDIAN replaces every weak component with a proven one:

| Ternary Reflex v1 | OBSIDIAN (v2) | Evidence |
|----|---------------|---------|
| Raw STE (identity backward) | **BitNet b1.58 absmax quantization** | Proven stable at 3B+ scale (2402.17764). Same {-1,0,+1} weights. |
| Element-wise GRU | **Griffin gated recurrence** (ternary gates) | Proven in 6 of our architectures. √(1-a²) coupling is bounded. |
| Custom genius path | **Caveman LFM backbone** (Tier 1, proven) | Build genius first, add reflex on top. If reflex fails, genius still works. |
| No factual knowledge in reflex | **Engram hash tables** at layers 2, 6 | O(1) factual lookup without matmuls. Reflex becomes SMART. |
| Full 50K vocab from step 0 | **6-phase graduated vocab** | COOKBOOK Section 4.5 + TERNARY-REFLEX-RESPONSE stabilization. |

---

## Architecture

```
Tokens → Embedding (d=1024, tied LM head, vocab=50257)
  │
  ├─ REFLEX PATH (BitNet b1.58, 8 layers, d=320, L2-RESIDENT):
  │     BitNet project: 1024 → 320
  │     8× Reflex Blocks:
  │       RMSNorm → BitNetLinear(320→960) → SiLU gate → BitNetLinear(960→320)
  │       + Griffin recurrence (d=320, BitNet gates, decay spectrum)
  │       + RMSNorm
  │     + Engram injection (layers 2, 6)
  │     Confidence: if reflex entropy < θ → SKIP genius
  │
  ├─ GENIUS PATH (Caveman LFM, 10 layers, d=1024):
  │     Project up: 320 → 1024 (fp16)
  │     10× Caveman LFM blocks (from CAVEMAN-LFM.md):
  │       Layer pattern: C C R C C R C R C R
  │       C=GatedConv(k=3), R=GriffinRec (fp16 standard)
  │       SwiGLU FFN (1024→2240→1024)
  │     Project down: 1024 → 320
  │     Additive correction to reflex state
  │
  → Project up: 320 → 1024 → Final RMSNorm → LM Head
```

---

## Component Details

### BitNet b1.58 Quantization (Replacing STE)

```python
class BitNetLinear(nn.Module):
    def __init__(self, in_d, out_d):
        self.weight = nn.Parameter(torch.randn(out_d, in_d) * 0.02)  # fp32 latent

    def forward(self, x):
        # Activation quantization: absmax → 8-bit
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quant = (x / x_scale * 127).round().clamp(-128, 127)

        # Weight quantization: centralize + absmax → {-1, 0, +1}
        w_mean = self.weight.mean()
        w_centered = self.weight - w_mean
        w_scale = w_centered.abs().mean().clamp(min=1e-5)
        w_quant = (w_centered / w_scale).round().clamp(-1, 1)

        # Forward: scaled ternary matmul (additions + sign flips)
        y = F.linear(x_quant.float(), w_quant) * (w_scale * x_scale / 127)
        return y
```

**Why stable:** Gradient flows through `w_scale = mean(|w - mean(w)|)` — smooth and bounded. No STE passthrough. The scaling factor normalizes gradients automatically.

### Griffin Recurrence on Reflex Path (BitNet Gates)

```python
# BitNet-quantized gates + standard Griffin coupling
a = sigmoid(BitNetLinear(x_norm, 320 → 320) + decay_bias)  # quantized gate
i = sigmoid(BitNetLinear(x_norm, 320 → 320))               # quantized input gate
v = BitNetLinear(x_norm, 320 → 320)                         # quantized value
h = a * h_prev + sqrt(1 - a**2) * (i * v)                  # standard coupling

# Parallel scan for training (fp32 accumulation)
# Sequential for inference (trivial)
```

Decay bias spectrum for d=320:
- Dims 0–79: bias=-2.2 (fast, local)
- Dims 80–239: bias=0.0 (medium, clause)
- Dims 240–319: bias=+4.6 (slow, topic)

### Genius Path = Caveman LFM

Reuse CAVEMAN-LFM.md exactly. 10 layers, d=1024:
- 6 gated conv blocks (C), 4 Griffin recurrence blocks (R)
- Pattern: `C C R C C R C R C R`
- SwiGLU FFN (1024→2240→1024) at every layer
- Standard fp16 weights, standard training

### Engram on Reflex Path (Layers 2, 6)

```python
# Same Engram as COOKBOOK Section 1.6, but on d=320 hidden state
bg = bigram_table[hash(token[t-1], token[t])]   # K=4 heads, d=64
tg = trigram_table[hash(token[t-2:t+1])]         # K=4 heads, d=64
e = concat(bg.mean(0), tg.mean(0))               # (128,)
alpha = sigmoid(rmsnorm(h) @ rmsnorm(W_K @ e) / sqrt(d))
x = x + alpha * W_V @ e                          # gated injection
```

**Engram training:** Adam, 5x base LR, zero WD (per COOKBOOK).

### Routing (Glue Dictionary + Entropy)

```python
# Primary: glue dictionary (~500 tokens, 65% by Zipf's law)
is_glue = GLUE_DICT[token_ids]

# Secondary: entropy check on reflex logits (for content tokens)
reflex_logits = lm_head(project_up(reflex_state))
entropy = -(F.softmax(reflex_logits, -1) * F.log_softmax(reflex_logits, -1)).sum(-1)
skip_genius = is_glue | (entropy < theta)

# Training: soft routing
gate = sigmoid((delta_norm - theta) / temperature)
out = (1-gate) * reflex_out + gate * (reflex_out + genius_correction)

# Inference: hard routing
out = reflex_out if skip_genius else reflex_out + genius_correction
```

---

## Configuration

| Parameter | Reflex Path | Genius Path |
|-----------|------------|-------------|
| d_model | 320 | 1024 |
| n_layers | 8 | 10 |
| Weight type | BitNet {-1,0,+1} | fp16 standard |
| Mixer | Griffin (BitNet gates) | GatedConv + Griffin (fp16) |
| FFN | BitNet SiLU gate | SwiGLU (1024→2240→1024) |
| Engram | Layers 2, 6 | None (genius uses standard weights) |
| Activation | 100% of tokens | ~35% (content tokens via routing) |

## Parameter Count

| Component | Params | Storage |
|-----------|--------|---------|
| Embedding (50257×1024, tied) | 51.5M | 103 MB fp16 |
| Projection bridges (4×) | 1.3M | 2.6 MB fp16 |
| Reflex path (8L, d=320, BitNet) | ~8M | **~1.0 MB ternary** |
| Engram tables + projections | 8M | 16 MB fp16 |
| Genius path (Caveman LFM, 10L) | ~155M | 310 MB fp16 |
| Routing + norms | ~70K | 0.14 MB |
| **TOTAL** | **~224M** | |

Budget remaining: ~26M. Options: increase reflex d to 384 (+3M), add Engram to genius (+5M), increase ffn_inner, or add MTP heads.

---

## Decode Speed (Strix Halo)

| Token type | Fraction | DRAM reads | Time | Throughput |
|-----------|----------|-----------|------|------------|
| Glue (reflex only) | 65% | LM head (26 MB int4) + reflex (**L2!**) | ~0.20 ms | ~5000 tok/s |
| Content (both paths) | 35% | LM head (26 MB) + genius (77 MB int4) + reflex (L2) | ~0.75 ms | ~1333 tok/s |
| **Weighted average** | | | **~0.39 ms** | **~2564 tok/s int4** |

fp16: ~775 tok/s weighted average.

---

## Training — 6-Phase Graduated

**Each phase has a validation gate. Do NOT proceed if gate fails.**

| Phase | Budget | Active | Vocab | LR | Gate |
|-------|--------|--------|-------|-----|------|
| 0: Smoke test | 200 steps | Mini config (d=128) | 1K | 1e-3 | Loss < 7, no NaN, grad_norm < 100 |
| 1: Genius warmup | 40% | Caveman LFM genius path ONLY | 50K | 8e-4 | Loss monotonically decreasing |
| 2: Reflex warmup | 20% | BitNet reflex + embeddings. Genius frozen. | 1K | 3e-4 | grad_norm < 10, max_logit < 30 |
| 3: Vocab expand | 10% | Reflex only | 50K | 3e-4 | grad_norm < 50 for 500 steps |
| 4: Both paths | 20% | Full model, soft routing | 50K | genius 8e-4, reflex 3e-4 | Loss < 6.0 |
| 5: Anneal routing | 10% | Full model, anneal temp 1.0→0.1 | 50K | cosine decay | 60-70% glue ratio |

**Phase 1 is Caveman LFM.** If Phase 2 fails, you still have a working genius path. The reflex is an ADD-ON, not a requirement. This is the key engineering insight vs Ternary Reflex v1.

### Optimizer Groups

| Group | Optimizer | LR | WD |
|-------|-----------|-----|-----|
| Genius backbone (conv, rec, FFN) | AdamW | 8e-4 → 8e-5 | 0.1 |
| Reflex BitNet weights | AdamW | 3e-4 → 3e-5 | 0.01 |
| Engram tables | **Adam** | **4e-3 (5×)** | **0** |
| Decay biases (both paths) | AdamW | 8e-5 (0.1×) | 0 |
| Routing MLP | AdamW | 8e-4 | 0.1 |

---

## HIP Kernels

**Reuse (5):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), prefix_scan (8.4×), cross_entropy (1.8×), dequantize_int4 (16.3×)

**New (3):**
1. **BitNet Ternary Accumulation** — {-1,0,+1} weight × int8 activation. Pure add/sub. L2-resident. Est. 10-20× vs fp16 matmul.
2. **Griffin Associative Scan** — Adapt prefix_scan for `(a₂·a₁, a₂·b₁+b₂)`. fp32 accumulation.
3. **Fused Engram Lookup+Gate** — hash + gather + sigmoid + weighted sum.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| BitNet noise hurts 250M quality | MEDIUM | Proven at 3B+. Genius path compensates. If reflex degrades quality, disable it (genius still works). |
| Griffin with ternary gates too noisy | MEDIUM | Sigmoid compresses noise to (0,1). Fallback: fp16 gates, keep only FFN as BitNet. |
| Reflex d=320 too narrow | MEDIUM | Engram adds factual capacity. Budget for d=384 (+3M params). |
| Training instability at vocab expand (Phase 3) | LOW | BitNet is fundamentally more stable than STE. 6-phase graduated + validation gates. |
| Two-path complexity | LOW | Genius = Caveman LFM (Tier 1, already proven). Reflex is incremental. |

## Success Criteria

1. Genius path alone matches Caveman LFM baseline quality (Phase 1 gate)
2. Adding reflex IMPROVES loss (not just speed)
3. Loss < 5.0 in 15 min (combined model)
4. 65%+ tokens route to reflex at convergence
5. Reflex weights confirmed 1.0 MB (fits in L2 with 5MB headroom)
6. Decode > 2000 tok/s int4 on Strix Halo
7. NO gradient explosion at any phase (all validation gates pass)
8. BitNet weights show clear {-1,0,+1} clustering after training

## Implementation Roadmap

1. **Build Caveman LFM first** (Tier 1 — modules: GatedConv, Griffin, SwiGLU, Engram, GlueDict)
2. Verify Caveman LFM trains successfully (Phase 1 validation)
3. Implement BitNetLinear (absmax quantization, no STE)
4. Implement reflex block (BitNet FFN + BitNet Griffin + Engram)
5. Implement routing (glue dictionary + entropy check)
6. Implement projection bridges (320↔1024)
7. Phase 2-3: train reflex with graduated vocab
8. Phase 4-5: combine paths, anneal routing
9. BitNet ternary accumulation HIP kernel
10. Decode benchmark: verify L2 residence + 2000+ tok/s

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x), **silu_gate_mul** (1.6x), **cross_entropy** (1.8x)
- **prefix_scan** (8.4x) — adaptable for Griffin associative scan
- Apply via `autokernel.optimize(model, training=True)`
- Engram hash tables: no special kernel needed (hash + gather are fast on GPU)

### Griffin/SSM Scan: Use Chunked Linear Recurrence
**Do NOT use sequential loops or `torch.associative_scan`** — both yield only 1.3K tok/s (4% MFU) on gfx1151. Use **chunked linear recurrence** (chunk_size=64) for 5x speedup:
- Reference: `models/amadeus.py:selective_scan_chunked`
- Griffin operator `(a₂·a₁, a₂·b₁+b₂)` fits the same pattern
- The `sqrt(1-a²)` coupling term can be precomputed per chunk

### Throughput Estimates (from AMADEUS baseline)
- **Eager mode:** ~6-8K tok/s, 16-20% MFU for ~250M SSM model
- **With autokernel patterns:** ~7-9K tok/s, 18-22% MFU
- **Token budget:** 15 min = 5.4-7.2M | 45 min = 16-22M | 120 min = 43-58M

### MFU: 65-75% training
FFN dominates compute (weight-bandwidth-bound). Recurrence is element-wise (~95% MFU). Engram lookups are memory-bound but small.

---

## Possible Optimizations & Throughput Estimate

**Baseline (estimated):** ~7,000 tok/s eager (16% MFU)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +100% MFU — BitNet ternary ops compile to efficient element-wise | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| BitNet ternary weights L2 residence | 1.58-bit weights: 8L × d=320 reflex path ≈ 5.2M params ≈ 1.3 MB → fits L2 | By design |
| `dequantize_int4` kernel | 16.3x for inference-time ternary dequantization | Available |
| Graduated vocab training | Smaller effective vocab early → faster initial training steps | By design |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~14,000 tok/s (32% MFU)
**Tokens in 45 min:** ~37.8M (2.4 BabyLM epochs)
**Ranking:** #7 of 22 architectures
