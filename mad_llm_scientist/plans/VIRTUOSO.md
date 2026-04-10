# VIRTUOSO

> **Superseded** by composable PLE+MatFormer modules. See `models/ple.py`, `models/matformer.py`, `models/virtuoso.py` and `docs/superpowers/specs/2026-04-10-ple-matformer-design.md`.

**The Master Performer: AMADEUS Base + Per-Layer Embeddings (PLE) + FiLM Conditioning**

*A virtuoso doesn't just play the notes — they bring unique perception and interpretation to every phrase. Each layer of VIRTUOSO has its own eyes (PLE) and its own feeling for the piece (FiLM).*

## Hypothesis

Gemma 4 (Google, 2025) proved that **Per-Layer Embeddings** — giving each decoder layer its own embedding table for every token — dramatically improves parameter efficiency in small models. VIRTUOSO combines PLE with AMADEUS's FiLM conditioning on a parallel hybrid backbone. Each layer knows BOTH what each token IS (PLE) and what the sequence IS ABOUT (FiLM). Nobody has combined these two techniques.

**Evidence:**
- Gemma 4 E2B: 2.3B effective params, 5.1B with PLE. PLE is the majority of params but zero FLOPs (lookups only).
- Config: `hidden_size_per_layer_input: 256`, `vocab_size_per_layer_input: 262144` at `hidden_size: 1536`.
- PLE solves the "frontloading problem" — standard transformers pack ALL token info into one initial embedding. PLE gives each layer FRESH token-level information.

---

## Architecture

```
Tokens → Main Embedding (d=1024, tied LM head, vocab=50257)
  │
  → 16 Parallel Hybrid Blocks:
  │     ★ PLE injection: h = h + ple_proj[i](ple_table[i][token_ids]) ★
  │     RMSNorm
  │     [GatedConv(d=640) || Mamba-3 SISO(d=384)] → Concat → OutProj → +Residual
  │     RMSNorm → SwiGLU FFN (1024→2240→1024) → +Residual
  │
  │     ★ LAYER 8: Context Fingerprint (FiLM) ★
  │     ★ LAYERS 9-16: FiLM Conditioned ★
  │
  → Final RMSNorm → LM Head
```

---

## Per-Layer Embeddings (PLE)

```python
class PerLayerEmbeddings(nn.Module):
    def __init__(self, vocab_size=50257, d_ple=32, d_model=1024, n_layers=16):
        super().__init__()
        # Each layer has its OWN embedding table
        self.tables = nn.ModuleList([
            nn.Embedding(vocab_size, d_ple) for _ in range(n_layers)
        ])
        # Each layer has a projection from d_ple to d_model
        self.projections = nn.ModuleList([
            nn.Linear(d_ple, d_model, bias=False) for _ in range(n_layers)
        ])
    
    def get_layer_embedding(self, token_ids, layer_idx):
        """Get per-layer token embedding for injection into residual stream."""
        ple = self.tables[layer_idx](token_ids)          # (B, T, d_ple=32)
        return self.projections[layer_idx](ple)           # (B, T, d_model=1024)
```

**Integration per layer:**
```python
def forward_layer(self, h, token_ids, layer_idx, film_score=None):
    # 1. PLE injection — FRESH token identity for this layer
    h = h + self.ple.get_layer_embedding(token_ids, layer_idx)
    
    # 2. Standard parallel hybrid block
    h_norm = rmsnorm(h)
    conv_out = self.gated_conv(h_norm)                    # (B, T, 640)
    mamba_out = self.mamba3(h_norm)                        # (B, T, 384)
    h = h + self.outproj(torch.cat([conv_out, mamba_out], dim=-1))
    h = h + self.ffn(rmsnorm(h))
    
    # 3. FiLM conditioning (layers 9-16 only)
    if film_score is not None:
        gamma = self.gamma_proj(film_score) + 1.0
        beta = self.beta_proj(film_score)
        h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)
    
    return h
```

**What each layer receives:**

| Signal | Source | What It Provides | Cost |
|--------|--------|-----------------|------|
| Residual stream `h` | Previous layer | Accumulated representations | 0 (already there) |
| PLE embedding | Per-layer table lookup | **FRESH token identity** (what IS this token?) | ~0 FLOPs (lookup) |
| FiLM modulation | Midpoint fingerprint | **Sequence context** (what is this text ABOUT?) | ~0 FLOPs (mul+add) |

**vs standard transformer:** Only gets residual stream. Token identity degrades with depth ("representation drift"). No explicit sequence-level context signal.

---

## Why PLE + FiLM is Novel

| Technique | What It Provides | Who Uses It |
|-----------|-----------------|-------------|
| PLE alone | Per-layer token identity | Gemma 4 (Google, 2025) |
| FiLM alone | Sequence-level context conditioning | AMADEUS (ours) |
| **PLE + FiLM** | **Per-layer token identity + sequence context** | **VIRTUOSO (nobody else)** |

Each layer has TWO conditioning signals:
1. **PLE:** "This token is 'Einstein'" (identity, per-token)
2. **FiLM:** "This text is about physics" (context, per-sequence)

Together: Layer 12 knows both WHAT it's processing and WHY.

---

## Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 1024 | Standard |
| d_conv | 640 (10 × 64) | LFM2 proven |
| d_mamba | 384 (6 × 64) | Mamba-3 SISO |
| dstate | 64 | Mamba-3 |
| n_layers | 16 | Standard |
| ffn_inner | 2240 (SwiGLU) | Reduced from 2560 to fit PLE budget |
| d_ple | 32 | Per-layer embedding dim |
| d_film | 64 | FiLM bottleneck |
| vocab_size | 50257 | tiktoken GPT-2 |

## Parameter Count

| Component | Params |
|-----------|--------|
| Main embedding (50257×1024, tied) | 51.5M |
| Per layer: gated conv | ~2.0M |
| Per layer: Mamba-3 SISO | ~0.9M |
| Per layer: output proj | ~1.05M |
| Per layer: SwiGLU FFN (1024→2240→1024) | ~6.88M |
| Per layer: RMSNorm ×2 | ~2K |
| **Per layer total** | **~10.83M** |
| **16 layers** | **173.3M** |
| PLE tables (50257 × 32 × 16) | 25.7M |
| PLE projections (32→1024 × 16) | 0.5M |
| FiLM conditioning | 1.1M |
| Final RMSNorm | 1K |
| **GRAND TOTAL** | **~252.1M** |

**Note:** If over 250M budget, reduce d_ple to 24 → PLE=19.3M → total=245.7M.

**PLE insight from Gemma 4:** These 26.2M PLE params cost ZERO FLOPs. They're embedding lookups. On Strix Halo with 128GB unified memory, the extra 52MB of tables (fp16) is negligible. The model gains per-layer token awareness for FREE compute cost.

---

## Decode Speed (Strix Halo)

| Mode | Weight Reads | Overhead | Total | Throughput |
|------|-------------|----------|-------|------------|
| fp16 | 504 MB / 240 GB/s = 2.10ms | 0.48ms | ~2.58ms | **~388 tok/s** |
| int4 | ~240 MB / 240 GB/s = 1.00ms | 0.48ms | ~1.48ms | **~676 tok/s** |

PLE adds 16 embedding lookups per token (~52MB total tables). On unified memory with 240 GB/s, reading 52MB = 0.22ms. But PLE entries per token per layer = 32 values = 64 bytes — fits in cache line. Effectively free.

---

## Training

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (50%) | 7.5 min | Backbone (conv+Mamba-3+FFN) only. No PLE, no FiLM. | Learn language basics |
| 2 (30%) | 4.5 min | + PLE (all 16 layers). | Each layer learns unique token perception |
| 3 (20%) | 3 min | + FiLM + full fine-tuning | Context conditioning + joint optimization |

**Optimizer:** Single AdamW group. PLE tables use standard 1x LR, 0 WD (they're embedding tables, same treatment as main embedding).

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| PLE adds 26M params for "just lookups" | LOW | Gemma 4 proved it works. Zero FLOPs. Quality gain > parameter cost. |
| PLE tables overfit on small data (BabyLM, 16M tokens) | MEDIUM | d_ple=32 is small. 25.7M lookup params with 16M tokens = ~0.6 tokens per param. May need regularization or smaller d_ple. |
| PLE + FiLM interaction unclear | LOW | They're orthogonal: PLE is per-token-per-layer, FiLM is per-sequence-per-layer. No interference expected. |
| Over 250M budget | LOW | Reduce d_ple from 32 to 24 (saves 6.4M). |

## Success Criteria

1. Loss < 4.3 in 15 min (PLE should improve over AMADEUS's 4.5 target)
2. PLE improves loss > 1% vs no-PLE ablation
3. FiLM improves loss > 0.5% vs no-FiLM ablation
4. PLE + FiLM together > either alone (synergy)
5. Per-layer PLE tables show distinct patterns (layer specialization)
6. Decode speed unaffected (< 1% overhead from PLE lookups)

## Implementation Roadmap

1. Build AMADEUS base (see AMADEUS.md roadmap)
2. Implement PerLayerEmbeddings class (16 tables + 16 projections)
3. Integrate PLE injection at start of each layer
4. Phase 1: train backbone
5. Phase 2: add PLE, train
6. Phase 3: add FiLM, full fine-tuning
7. Ablation: base vs +PLE vs +FiLM vs +both
8. Visualize: per-layer embedding similarity (t-SNE) to confirm layer specialization

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

### External Kernel Integration (verified 2026-04-10)

- **GatedConv:** causal-conv1d (10x vs nn.Conv1d) — auto-used if installed, try/except fallback
- **Mamba-3 scan:** mamba-ssm selective_scan_fn (5.6x, 0.32ms) — drop-in upgrade for AMADEUS base
- **Conductor/PLE ops:** Element-wise, no external kernel needed
