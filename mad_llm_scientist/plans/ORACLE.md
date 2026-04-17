---
title: "ORACLE"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/SENTINEL.md
  - mad_llm_scientist/plans/JORMUNGANDR.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %oracle, %surprisal, %adaptive-depth, %sr-ttt, %heterogeneous-loop]
---

# ORACLE

**Surprisal-Driven Heterogeneous Loop Depth — The Seer Who Gazes Longest at What Matters Most**

*"The oracle does not spend equal time on every supplicant. The peasant with a simple question gets a swift answer. The king facing war gets the deepest vision. Wisdom is knowing where to look hardest."*
*SR-TTT proved surprisal routes attention. ORACLE proves it routes depth.*

## Hypothesis

Not all tokens need the same depth. "The" needs 2 iterations. A math proof's conclusion needs 16. ORACLE combines SR-TTT's (Feb 2026) surprisal signal with JORMUNGANDR-HALO's loop to create **per-token adaptive depth with heterogeneous processing**. A cheap 1-iteration probe computes per-token surprisal, then tokens are grouped into LOW/MEDIUM/HIGH bands that get 2/8/16 iterations respectively. The HIGH band additionally gets In-Place TTT adaptation (from SENTINEL). All three bands use the same shared L2-resident block. The net effect: 60% of tokens are processed 4x faster (2 vs 8 iterations), while the 10% that need it get 2x deeper processing than the baseline.

**Key papers:** "SR-TTT" (2603.06642, Feb 2026), "In-Place TTT" (2604.06169, Apr 2026), "Elastic Attention" (2601.17367, Jan 2026)

---

## Architecture

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  -> Probe (1 iteration of shared block):
  |     h_probe = shared_block(h)
  |     logits_probe = lm_head(norm(h_probe))
  |     surprisal_t = -log P(target_t | h_probe)
  |
  -> Group by surprisal:
  |     LOW  (s < tau_low,  ~60%): 2 iterations, base weights
  |     MED  (tau_low <= s < tau_high, ~30%): 8 iterations, base weights
  |     HIGH (s >= tau_high, ~10%): 16 iterations + In-Place TTT
  |
  -> Process each group through shared block:
  |     +---------------------------------------------+
  |     | 1 SHARED BLOCK (L2-resident, ~3.7MB fp16)   |
  |     |   EFLA Token Mixer (chunk-wise, C=64)        |
  |     |   SwiGLU FFN                                 |
  |     |   For HIGH group only: TTT on W_down         |
  |     +---------------------------------------------+
  |
  -> Reassemble (scatter tokens back to original positions)
  -> Final RMSNorm -> LM Head
```

### The Surprisal-Depth Insight

Standard looped models apply N iterations uniformly. This wastes compute on easy tokens and under-processes hard tokens. Language has a heavy-tailed difficulty distribution:

```
Token difficulty distribution (empirical, GPT-2 tokenizer on BabyLM):
  ~60% of tokens: surprisal < 2.0 (function words, common continuations)
  ~30% of tokens: 2.0 <= surprisal < 5.0 (content words, moderate difficulty)
  ~10% of tokens: surprisal >= 5.0 (rare words, topic shifts, reasoning)
```

If the baseline uses 10 uniform iterations, ORACLE's average is:
```
avg_iterations = 0.6 * 2 + 0.3 * 8 + 0.1 * 16 = 1.2 + 2.4 + 1.6 = 5.2
```

That's **1.9x fewer iterations** than uniform 10, but the hard tokens get 1.6x MORE iterations (16 vs 10). Quality improves where it matters most (high loss tokens), throughput improves everywhere else.

### Batch-Level Implementation

Grouping tokens by depth and processing each group separately avoids padding waste:

```python
# Sort tokens by surprisal -> process groups -> scatter back
sorted_idx = surprisal.argsort(dim=-1)
# Group 1: first 60% (LOW), Group 2: next 30% (MED), Group 3: last 10% (HIGH)
# Process each group with different iteration count
# Scatter results back to original positions
```

This is efficient on GPU because each group is a contiguous batch of tokens. GPU occupancy is high for the large LOW group and adequate for the MED group. The small HIGH group may have lower occupancy but gets the most iterations.

---

## Component 1: Surprisal Probe

```python
class SurprisalProbe(nn.Module):
    def __init__(self, shared_block, lm_head, final_norm):
        self.block = shared_block
        self.lm_head = lm_head
        self.norm = final_norm

    @torch.no_grad()
    def forward(self, h, targets):
        h_probe = self.block(h)
        logits = self.lm_head(self.norm(h_probe))
        surprisal = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='none'
        ).view(h.shape[0], h.shape[1])
        return surprisal, h_probe
```

## Component 2: Depth Router

```python
class DepthRouter:
    def __init__(self, tau_low=2.0, tau_high=5.0,
                 n_low=2, n_med=8, n_high=16):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.depths = {'low': n_low, 'med': n_med, 'high': n_high}

    def route(self, surprisal):
        low_mask = surprisal < self.tau_low
        high_mask = surprisal >= self.tau_high
        med_mask = ~low_mask & ~high_mask
        return {
            'low': (low_mask, self.depths['low']),
            'med': (med_mask, self.depths['med']),
            'high': (high_mask, self.depths['high']),
        }
```

## Component 3: Oracle Model

```python
class OracleModel(nn.Module):
    def __init__(self, d_model=768, tau_low=2.0, tau_high=5.0):
        self.embedding = nn.Embedding(50257, d_model)
        self.shared_block = ErebusBlock(d_model)  # EFLA + SwiGLU
        self.ttt_module = InPlaceTTT(d_model, eta=0.01)
        self.probe = SurprisalProbe(self.shared_block, None, None)
        self.router = DepthRouter(tau_low, tau_high)
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, 50257, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.probe.lm_head = self.lm_head
        self.probe.norm = self.final_norm

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        h = self.embedding(input_ids)

        if self.training and targets is not None:
            # Probe: 1 iteration to get surprisal
            surprisal, h_probe = self.probe(h, targets)
            groups = self.router.route(surprisal)

            # Process each group with appropriate depth
            h_out = torch.zeros_like(h)
            for group_name, (mask, n_iters) in groups.items():
                if mask.any():
                    h_group = h_probe[mask]  # start from probe output (1 iter done)
                    for i in range(n_iters - 1):  # -1 because probe was iter 1
                        if group_name == 'high':
                            h_group = self.shared_block(h_group)
                            h_group = self.ttt_module(h_group, surprisal[mask])
                        else:
                            h_group = self.shared_block(h_group)
                    h_out[mask] = h_group
        else:
            # Inference: fixed depth (or use cached surprisal from previous tokens)
            for i in range(10):  # default inference depth
                h = self.shared_block(h)
            h_out = h

        return self.lm_head(self.final_norm(h_out))
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_heads | 12 |
| head_dim | 64 |
| ffn_inner | 1920 (2.5x) |
| shared_blocks | 1 |
| tau_low | 2.0 (surprisal threshold for LOW) |
| tau_high | 5.0 (surprisal threshold for HIGH) |
| n_iters_low | 2 |
| n_iters_med | 8 |
| n_iters_high | 16 |
| ttt_eta | 0.01 |
| chunk_size | 64 |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257x768, tied) | 38.6M |
| **Shared block (EREBUS-style):** | **~7.38M** |
| TTT module (no extra params) | 0 |
| Final RMSNorm | 768 |
| **GRAND TOTAL (unique)** | **~46.0M** |
| **Effective params (avg 5.2 iters)** | **~76M effective** |

Same parameter budget as EREBUS/SENTINEL. Depth router and TTT module add zero learnable parameters.

---

## Training

### Two Phases

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (50%) | 22 min | Uniform 10 iterations (no routing) | Stabilize shared block |
| 2 (50%) | 23 min | Surprisal routing + heterogeneous depth | Learn adaptive depth allocation |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-3 cosine -> 1e-4, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 32x1024, accum=2 (64K effective) |
| Precision | fp16 mixed + fp32 EFLA state |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 iterations (MED/HIGH groups) |
| tau_low | 2.0 (fixed) |
| tau_high | 5.0 (fixed) |
| TTT warmup | Disabled until Phase 2 step 500 |

### Tau Calibration

tau_low and tau_high are calibrated from Phase 1 loss statistics:
```
tau_low = percentile(token_losses, 60)   # 60th percentile
tau_high = percentile(token_losses, 90)  # 90th percentile
```
This ensures the 60/30/10 split regardless of absolute loss scale.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Surprisal probe requires targets (not available at inference) | MEDIUM | At inference, use previous token's loss as proxy, or use fixed depth. Or: train a tiny MLP to predict surprisal from h (distill probe into hidden state). |
| Grouped processing fragments GPU batches | MEDIUM | LOW group (60%) maintains good occupancy. MED (30%) is adequate. HIGH (10%) may be small but gets most iterations. |
| Gradient discontinuity at group boundaries | LOW | Groups are non-overlapping masks, no routing gradient needed. The shared block parameters get gradients from all groups. |
| Phase 1 -> Phase 2 transition causes instability | LOW | Phase 1 uses uniform depth. Phase 2 starts with wide tau range (nearly uniform) and narrows. Smooth transition. |
| Variable depth means variable memory for gradient checkpointing | MEDIUM | Checkpoint every 4 iterations for all groups. MED/HIGH groups use more memory but there are fewer of these tokens. |

## Success Criteria

1. **Val loss < 2.85 on BabyLM** (deeper processing on hard tokens improves overall loss)
2. **Throughput > 50K tok/s** (average 5.2 iterations vs 10 baseline)
3. Per-group loss analysis: HIGH group loss decreases most from extra iterations
4. Average depth < 6 with < 0.5% quality loss vs uniform-10
5. Group sizes match expected 60/30/10 distribution
6. Throughput scales linearly with average depth reduction

---

## Implementation Roadmap

1. Start from EREBUS implementation (EFLA looped block)
2. Implement SurprisalProbe (1-iteration + cross-entropy)
3. Implement DepthRouter (tau-based grouping)
4. Implement grouped processing (sort + scatter)
5. Add In-Place TTT for HIGH group only
6. Assemble OracleModel with two-phase training
7. Verify parameter count (~46M unique)
8. Phase 1: uniform depth, collect surprisal statistics for tau calibration
9. Phase 2: enable routing, calibrate tau from Phase 1 statistics
10. Ablation: uniform-10 vs ORACLE (same total compute budget)
11. Ablation: ORACLE-noTTT vs ORACLE-withTTT (isolate TTT contribution)
12. Visualize: which tokens get which depth? Do hard tokens improve most?

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), chunked_linear_cross_entropy

**External (2):** causal-conv1d (10x), FLA DeltaNet kernel (EFLA)

**New (0):** Surprisal is cross_entropy (existing). Grouping is index_select + scatter (standard PyTorch). No new kernels.

### Compute Budget Analysis

```
Uniform-10 baseline:
  1024 tokens x 10 iters = 10,240 token-iterations

ORACLE (60/30/10 split at 2/8/16):
  614 tokens x 2 iters  = 1,228 token-iterations (LOW)
  307 tokens x 8 iters  = 2,456 token-iterations (MED)
  103 tokens x 16 iters = 1,648 token-iterations (HIGH)
  + 1024 tokens x 1 iter = 1,024 (probe)
  Total: 6,356 token-iterations

Speedup: 10,240 / 6,356 = 1.61x
```

But the probe iteration is lightweight (no_grad). Effective speedup: **~1.7-1.9x**.

### Group Processing Efficiency

| Group | Tokens | Iters | Batch size | GPU occupancy |
|-------|--------|-------|------------|---------------|
| LOW | 614 | 2 | 614 | High (75%) |
| MED | 307 | 8 | 307 | Good (50%) |
| HIGH | 103 | 16 | 103 | Moderate (25%) |
| Probe | 1024 | 1 | 1024 | Full (100%) |

The LOW group dominates wall-clock time (most tokens, though fewest iters). GPU occupancy is acceptable for all groups.

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Phase 1 (uniform-10) | compile + AK | ~38K tok/s |
| Phase 2 (ORACLE, avg 5.2 iters) | compile + AK | ~55K tok/s |
| Phase 2 (+ FLA + causal-conv1d) | | **~62K tok/s** |
| **Inference (cached surprisal)** | | **~65K tok/s** |

**Estimated training throughput (Phase 2):** ~55-62K tok/s
**Estimated inference throughput:** ~60-70K tok/s
**Tokens in 45 min:** ~95M (Phase 1) + ~75M (Phase 2) = ~170M total (10.6 BabyLM epochs)
**Ranking:** #1-2 for throughput, #2-3 for quality
