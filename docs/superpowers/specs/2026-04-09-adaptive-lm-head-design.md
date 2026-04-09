# Adaptive LM Head + Chunked Cross-Entropy

**Date:** 2026-04-09
**Status:** Draft
**Goal:** Reduce LM head cost from 13.25ms (33% of forward) via frequency-tiered vocabulary projection + memory-efficient chunked loss computation.

---

## Problem

After Tier 1+2 optimizations (autokernel patterns + HIP scan kernel), the LM head matmul is the single largest remaining bottleneck:

| Component | Time | % of Forward |
|-----------|------|-------------|
| **LM head** (1024 @ 50257) | **13.25ms** | **33%** |
| SwiGLU FFN ×16 | 3.7ms × 16 | ~15% each layer |
| Mamba3SISO ×16 | 1.8ms × 16 | ~7% each layer |
| RMSNorm, Conv, etc. | ~2ms | ~5% |

The LM head is a single rocBLAS matmul: `(4096, 1024) @ (1024, 50257)`. We cannot beat rocBLAS at matmul (no MFMA). Instead, we reduce the work.

**Training:** The 412 MB output logit tensor dominates memory bandwidth.
**Inference:** The 103 MB weight read per decode step dominates latency.

---

## Design: Three-Tier Adaptive Softmax with Chunked Loss

### Vocabulary Tiers

Split vocab_size=50257 into 3 frequency tiers. Token frequencies from BabyLM/OpenWebText follow a Zipf distribution — the top 8K tokens cover ~90% of text.

| Tier | Tokens | Size | Projection | Params | % of Training Tokens |
|------|--------|------|-----------|--------|---------------------|
| 0 (common) | 0–8,191 | 8,192 | Full-rank: Linear(1024, 8192) | 8.4M | ~65% |
| 1 (medium) | 8,192–24,575 | 16,384 | Low-rank: Linear(1024, 256) → Linear(256, 16384) | 4.5M | ~25% |
| 2 (rare) | 24,576–50,256 | 25,681 | Low-rank: Linear(1024, 128) → Linear(128, 25681) | 3.4M | ~10% |
| **Total** | | **50,257** | | **16.3M** | |

vs. standard LM head: Linear(1024, 50257) = **51.5M params** (3.2x reduction).

**Tier boundaries** are set by sorting tiktoken GPT-2 vocab by frequency in BabyLM. Tier 0 is the most common 8192 tokens, etc.

### Forward Pass (Training)

```python
class AdaptiveLMHead(nn.Module):
    def __init__(self, d_model=1024, vocab_size=50257,
                 tier_sizes=(8192, 16384, 25681),
                 tier_ranks=(1024, 256, 128)):
        # Tier 0: full-rank
        self.head_0 = nn.Linear(d_model, tier_sizes[0], bias=False)
        # Tier 1: low-rank
        self.proj_1 = nn.Linear(d_model, tier_ranks[1], bias=False)
        self.head_1 = nn.Linear(tier_ranks[1], tier_sizes[1], bias=False)
        # Tier 2: low-rank
        self.proj_2 = nn.Linear(d_model, tier_ranks[2], bias=False)
        self.head_2 = nn.Linear(tier_ranks[2], tier_sizes[2], bias=False)
        # Token-to-tier mapping
        self.register_buffer('token_tier', ...)  # (vocab_size,) int8
        self.register_buffer('token_idx', ...)   # (vocab_size,) int32 — index within tier

    def forward(self, h, targets=None):
        """If targets provided, compute chunked CE loss. Else return full logits."""
        if targets is not None:
            return self._chunked_ce_loss(h, targets)
        return self._full_logits(h)
```

### Chunked Cross-Entropy (Memory-Efficient Loss)

During training, we never materialize the full (B*T, 50257) logit tensor. Instead:

1. Compute Tier 0 logits: `(4096, 1024) @ (1024, 8192) = (4096, 8192)` — 67 MB
2. Compute target logit for each token (gather from whichever tier the target is in)
3. Accumulate log-sum-exp across tiers: `log_Z = logaddexp(lse_0, lse_1, lse_2)`
4. Loss = `target_logit - log_Z`

**Peak memory:** max(67 MB, 134 MB, 105 MB) = **134 MB** (vs 412 MB standard) — 3x reduction.

```python
def _chunked_ce_loss(self, h, targets):
    B_T, D = h.shape[0] * h.shape[1], h.shape[-1]
    h_flat = h.reshape(B_T, D)

    # Gather target info
    target_tier = self.token_tier[targets.reshape(-1)]   # which tier
    target_idx = self.token_idx[targets.reshape(-1)]     # index within tier

    # Accumulate log-sum-exp across tiers
    log_Z = torch.full((B_T,), float('-inf'), device=h.device)
    target_logit = torch.zeros(B_T, device=h.device)

    for tier, (proj, head, size) in enumerate(self.tiers):
        if proj is not None:
            logits_tier = head(proj(h_flat))  # low-rank
        else:
            logits_tier = head(h_flat)        # full-rank
        
        # Update log-sum-exp
        log_Z = torch.logaddexp(log_Z, logits_tier.logsumexp(dim=-1))
        
        # Gather target logits from this tier
        mask = (target_tier == tier)
        if mask.any():
            idx = target_idx[mask]
            target_logit[mask] = logits_tier[mask].gather(1, idx.unsqueeze(1)).squeeze(1)

    loss = (log_Z - target_logit).mean()
    return loss
```

### Forward Pass (Inference)

During inference (decode), we want the top-K tokens for sampling:

1. Compute Tier 0 logits only (8192 tokens, ~8.4 MB weight read)
2. Find top-K candidates in Tier 0
3. **Early exit:** If top-1 confidence > threshold (e.g., softmax > 0.9), skip Tiers 1+2
4. Otherwise, compute Tier 1+2 logits and merge

**Expected:** ~65% of decode steps exit after Tier 0 only. Weight read drops from 103 MB to ~17 MB (8.4M × 2 bytes) for those steps.

### Decode Speed Estimate

| Scenario | Weight Read | Time (240 GB/s) | Probability |
|----------|------------|-----------------|-------------|
| Tier 0 only (early exit) | 17 MB | 0.07ms | ~65% |
| Tier 0 + Tier 1 | 17 + 9 = 26 MB | 0.11ms | ~25% |
| All tiers | 17 + 9 + 7 = 33 MB | 0.14ms | ~10% |
| **Weighted average** | **~20 MB** | **~0.08ms** | |

vs. standard LM head: 103 MB → 0.43ms. That's a **5x decode speedup** on the LM head alone.

---

## Training Speed Estimate

| Operation | Standard | Adaptive + Chunked |
|-----------|---------|-------------------|
| LM head matmul | (4096, 1024) @ (1024, 50257) = 13.25ms | Tier 0: ~2.3ms + Tier 1: ~1.1ms + Tier 2: ~0.6ms = **~4.0ms** |
| Output tensor | 412 MB | max 134 MB (Tier 1 is largest) |
| **Total LM head** | **13.25ms** | **~4-5ms** |

Net improvement: 13.25 → 4-5ms = **~8ms saved per forward**, which at batch=8×512 tokens should add ~15-20% more tok/s on top of the current 10.4K.

---

## Weight Tying

Standard weight tying (`output.weight = tok_embeddings.weight`) doesn't work directly with adaptive softmax because the tiers have different shapes. Two options:

**Option A (recommended): Tie Tier 0 only.** The embedding table is (50257, 1024). Tier 0's head is (8192, 1024) — use the first 8192 rows of the embedding. Tiers 1+2 have their own low-rank projections (not tied).

**Option B: No tying.** Separate embedding (50257 × 1024 = 51.5M) and adaptive head (16.3M). Total: 67.8M vs 51.5M (shared). Adds 16.3M params but stays under 250M budget.

---

## Integration Points

| File | Changes |
|------|---------|
| `models/amadeus.py` | Replace `self.output = nn.Linear(vocab, d, bias=False)` with `AdaptiveLMHead` |
| `models/amadeus.py` | Modify `forward()` to pass targets for chunked CE during training |
| `halo_training/trainer.py` | Adjust loss computation: if model returns loss, use it directly |
| New: `models/adaptive_head.py` (or inline) | `AdaptiveLMHead` class with tier projections + chunked CE |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Tier boundaries don't match actual BabyLM frequency | Pre-compute token frequencies from dataset, set boundaries empirically |
| Low-rank tiers lose quality on rare tokens | Start with rank=256/128, increase if rare-token loss is significantly higher |
| Chunked CE has different gradient flow | Mathematically equivalent to standard CE — no gradient difference |
| Early exit in decode misses rare tokens | Only skip when confidence > 0.9; always compute all tiers for low-confidence steps |
| Adds complexity to the forward pass | Contained in one class; the rest of AMADEUS is unchanged |

---

## Success Criteria

1. Training: LM head time drops from 13.25ms to <5ms
2. Training: Final loss within 5% of standard LM head baseline (12.18 ± 0.6)
3. Inference: Decode speed improves by >2x on the LM head component
4. Memory: Peak logit tensor drops from 412 MB to <150 MB
5. Params: Total model stays under 250M
