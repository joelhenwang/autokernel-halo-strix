# PARALLEL CAVEMAN

**Falcon-H1-Style Parallel Hybrid Conv+Recurrence with Meta Tokens and Caveman Routing**

## Hypothesis

Upgrade Caveman LFM from alternating conv/recurrence layers to **parallel hybrid** in every layer: gated conv channels + Griffin recurrence channels process the same input simultaneously, outputs concatenated. Add Hymba-style **meta tokens** (128 learned embeddings). Keep Engram and caveman routing.

**Evidence:** Three independent papers (Hymba, Falcon-H1, Meta Hybrid Analysis) converge: parallel hybrid > sequential/alternating. At 300M: +1.1% commonsense, +4.7% recall vs sequential. Falcon-H1-0.5B with parallel hybrid matches 7B models from 2024.

---

## Architecture

```
[128 Meta Tokens] + Input Tokens → Embedding (d=1024, tied LM head)
  → 16 Parallel Hybrid Blocks:
      RMSNorm → [Conv Channels (12) || Recurrence Channels (4)] → Concat → Output Proj → +Residual
      RMSNorm → SwiGLU FFN → +Residual
      (+ Engram at blocks 2, 9)
      (+ Caveman routing: glue tokens skip recurrence channels)
  → Final RMSNorm → LM Head
```

### Parallel Hybrid Block

```python
x_norm = rmsnorm(x)

# --- PARALLEL: Conv channels + Recurrence channels ---

# Conv channels (12 of 16 total, d_conv=768)
B, C, h_tilde = linear(x_norm, 1024 → 3*768).chunk(3)   # gated conv projections
y = B * h_tilde
z = causal_conv1d(y, k=3)
conv_out = C * z                                           # (B, T, 768)

# Recurrence channels (4 of 16 total, d_rec=256)
a = sigmoid(linear(x_norm, 1024 → 256) + decay_bias)     # decay gate
i = sigmoid(linear(x_norm, 1024 → 256))                   # input gate
v = linear(x_norm, 1024 → 256)                             # value
h = a * h_prev + sqrt(1 - a**2) * (i * v)                 # Griffin coupling
rec_out = h                                                 # (B, T, 256)
# Caveman: glue tokens use bypass instead of recurrence
rec_out = where(is_glue, linear(x_norm, 1024 → 256), rec_out)

# --- CONCAT + OUTPUT ---
combined = concat(conv_out, rec_out)                       # (B, T, 1024)
out = linear(combined, 1024 → 1024)                        # output projection
x = x + out

# SwiGLU FFN (all tokens)
x = x + swiglu_ffn(rmsnorm(x))
```

**Key difference from Caveman LFM:** Every layer has BOTH conv and recurrence. No alternating. The 12:4 channel ratio means 75% conv (cheap local mixing) + 25% recurrence (expensive global context) — matching Falcon-H1's finding that a small share of attention/SSM handles precision while conv does bulk work.

### Meta Tokens (from Hymba)

128 learned embeddings prepended to every input sequence:
```python
meta = self.meta_tokens.expand(B, 128, 1024)   # (B, 128, 1024), ~131K params
x = concat(meta, embed(tokens))                  # (B, 128+T, 1024)
# Process through all layers, then discard meta positions at output
logits = lm_head(x[:, 128:, :])                 # only real token positions
```

**Purpose:** Compressed world knowledge + attention backstop. Hymba ablation: +1.4% accuracy at 300M for 131K params. Essentially free.

## Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| d_model | 1024 | LFM2 |
| d_conv (conv channels) | 768 (12 channels × 64) | Falcon-H1 ratio |
| d_rec (recurrence channels) | 256 (4 channels × 64) | Falcon-H1 ratio |
| n_layers | 16 | LFM2 |
| ffn_inner | 2240 (SwiGLU) | Budget-constrained |
| conv_kernel | 3 | LFM2 |
| meta_tokens | 128 | Hymba |
| vocab_size | 50257 | tiktoken GPT-2 |
| Engram | bigram 65K×64 + trigram 32K×64, layers 2,9 | Engram paper |
| Glue dictionary | ~500 tokens (~65% occurrences) | Caveman |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×1024, tied) | 51.5M |
| Meta tokens (128×1024) | 0.13M |
| Per layer: conv projections (1024→3×768 + 768→part of 1024) | ~3.9M |
| Per layer: recurrence projs (4 × 1024→256 + bypass) | ~1.3M |
| Per layer: output projection (1024→1024) | ~1.05M |
| Per layer: SwiGLU FFN (1024→2240→1024) | ~6.9M |
| Per layer: RMSNorm ×2 | ~2K |
| Per layer total | ~13.2M |
| 16 layers | ~211M |
| Engram (2 layers) | ~12.8M |
| **TOTAL** | **~275M** — over budget! |

**Fix:** Reduce ffn_inner to 1792 (1.75× expansion):
- FFN per layer: 3 × 1024 × 1792 = 5.5M
- Layer total: ~11.7M
- 16 layers: ~187M
- Total: 51.5 + 0.13 + 187 + 12.8 = **~251M** ✓

**Revised ffn_inner = 1792**

## Decode Speed (Strix Halo)

Key advantage: ALL 16 layers run conv+recurrence in parallel. No sequential alternating. The recurrence channels are only 256-dim (tiny state).

| Mode | Weight reads | Estimate |
|------|-------------|----------|
| fp16 | ~502 MB, glue skip 256-dim rec (save ~24 MB) | ~2.8 ms = ~357 tok/s |
| int4 | ~126 MB | ~0.74 ms = ~1351 tok/s |

Comparable to Caveman LFM but with BETTER quality (parallel hybrid evidence).

## Training

Same phase training as Caveman LFM. Engram: 5× LR, zero WD. Meta tokens: standard LR.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Parallel hybrid untested at 250M on our hardware | Hymba tested at 300M; Falcon-H1 at 520M; well-validated |
| ffn_inner=1792 small | Conv channels have own 3×768 projections adding capacity |
| Meta tokens add 128 positions to sequence | Negligible compute cost; can reduce to 64 if memory-tight |

## Implementation Roadmap

1. Implement ParallelHybridBlock (conv channels || rec channels → concat → output proj)
2. Implement meta token prepending
3. Port Engram + caveman routing from Caveman LFM
4. Verify params ~251M
5. Phase training (conv-only → +recurrence → +Engram)
6. Benchmark: compare to Caveman LFM (alternating) at equal tokens

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
Same as Caveman LFM: fused_res_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), prefix_scan (8.4x). Apply via `autokernel.optimize(model, training=True)`.

### Scan Implementation
Parallel hybrid runs conv and recurrence on the same input — no extra serial cost vs alternating. Griffin recurrence uses chunked linear recurrence (NOT sequential/associative_scan). Reference: `models/amadeus.py:selective_scan_chunked`.

### Throughput Estimates
- **Eager:** ~6-8K tok/s, ~16-20% MFU (similar to Caveman LFM)
- **Token budget:** 15 min = 5.4-7.2M | 45 min = 16-22M | 120 min = 43-58M
- Parallel hybrid adds one concat+proj matmul per layer vs alternating — ~2% overhead

### FFN Capacity Warning
`ffn_inner=1792` (1.75x) is reduced from Caveman's 2240 (2.19x) to fit 250M budget. This saves ~15% memory bandwidth per layer but may hurt quality. Benchmark both 1792 and 2240 before committing.

### MFU: 65-75% training (same as Caveman LFM)

### External Kernel Integration (verified 2026-04-10)

- **GatedConv:** causal-conv1d (10x vs nn.Conv1d) — auto-used if installed, try/except fallback
- **SSM scan (if Mamba path used):** mamba-ssm selective_scan_fn (5.6x, 0.32ms) — drop-in upgrade
- **Griffin scan:** Chunked linear recurrence remains primary. FLA HGRN (0.40ms) as alternative.
