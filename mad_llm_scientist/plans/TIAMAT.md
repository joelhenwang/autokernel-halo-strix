---
title: "TIAMAT"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %tiamat, %gsa, %memory-caching, %unified-memory]
---

# TIAMAT

**Gated Slot Attention + Unified Memory Caching — The Sea Dragon With Growing Memory**

*"The sea dragon doesn't forget — it stores each memory in a pearl, and the ocean floor holds millions."*
*GSA's compact state achieves Transformer-level recall. Unified memory makes caching those states free.*

## Hypothesis

Gated Slot Attention (GSA, Zhang & Yang Sep 2024) provides the best recall-memory tradeoff among linear attention variants via its two-pass GLA architecture linked by softmax (exponential memory capacity from Hopfield network theory). Memory Caching (MC, Behrouz et al. Feb 2026) enhances any RNN by checkpointing hidden states at regular intervals, growing effective memory from O(1) to O(√L). On discrete GPUs, MC's checkpoints incur PCIe transfer cost. On Strix Halo's unified LPDDR5X (128GB), **MC checkpoints are free pointer operations** — no data movement required. TIAMAT combines GSA + MC to create a recurrent model with unbounded effective context at O(L) training cost.

**Key papers:** "Gated Slot Attention" (2409.07146), "Memory Caching: RNNs with Growing Memory" (2602.24281)

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  → 16 GSA + MC Blocks:
  │     RMSNorm
  │     ┌──────────────────────────────────────────────┐
  │     │ Gated Slot Attention (GSA)                   │
  │     │   Pass 1: GLA(q, k, 1-α, α, 1) → o'        │
  │     │   softmax(o') → attention weights over slots │
  │     │   Pass 2: GLA(softmax(o'), 1-α, v, 1, α) → o│
  │     │   m=64 slots, H=4 heads                     │
  │     │                                              │
  │     │ Memory Cache (MC)                            │
  │     │   Checkpoint GSA state every C=64 tokens     │
  │     │   Cache in unified LPDDR5X (zero copy)       │
  │     │   Gated retrieval: top-k=4 nearest states    │
  │     │   Aggregate via learned gating               │
  │     └──────────────────────────────────────────────┘
  │     +Residual
  │     RMSNorm → SwiGLU FFN (768→1920→768) → +Residual
  │
  → Final RMSNorm → LM Head
```

---

## Component 1: Gated Slot Attention (GSA)

GSA is a two-pass GLA with softmax link. The key insight: `m` memory slots (much fewer than `d`) are updated with gated forgetting, and queries attend to slots via softmax — preserving the sharpness that linear attention loses.

```python
class GatedSlotAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=4, n_slots=64, head_dim=192):
        self.n_heads = n_heads
        self.n_slots = n_slots
        self.head_dim = head_dim  # d_model / n_heads
        # Projections
        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        # Forget gate: sigmoid with damping
        self.w_alpha = nn.Linear(d_model, n_heads * n_slots, bias=True)
        self.damping = 8.0  # τ from GSA paper
        # Output: Swish + RMSNorm (following GSA paper)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q = F.silu(self.w_q(x)).view(B, T, self.n_heads, self.head_dim)
        k = F.silu(self.w_k(x)).view(B, T, self.n_heads, self.head_dim)
        v = F.silu(self.w_v(x)).view(B, T, self.n_heads, self.head_dim)

        # Forget gate with damping (element-wise, free)
        alpha = torch.sigmoid(self.w_alpha(x) / self.damping)
        alpha = alpha.view(B, T, self.n_heads, self.n_slots)

        # Pass 1: GLA with slot routing → attention weights over slots
        # o'_t = GLA(q_t, k_t, 1-α_t, α_t, 1)  ∈ R^{n_slots}
        slot_logits = gla_pass1(q, k, alpha)  # (B, T, H, n_slots)

        # Softmax over slots (sharp retrieval, element-wise)
        slot_weights = F.softmax(slot_logits, dim=-1)

        # Pass 2: GLA with slot-weighted value accumulation
        # o_t = GLA(softmax(o'), 1-α, v, 1, α)
        o = gla_pass2(slot_weights, alpha, v)  # (B, T, H, head_dim)

        # RMSNorm + Swish gate + output projection
        o = self.w_o(F.silu(o).flatten(-2, -1))
        return o, self._get_state()  # Return state for MC
```

**State size per layer:** 2 × (n_slots × head_dim) × n_heads = 2 × (64 × 192) × 4 = 98K params. **Much smaller** than GLA (256 × d) or RetNet (512 × d). The softmax link exponentially increases effective capacity.

**Training:** Uses FLA's hardware-efficient chunkwise algorithm for both GLA passes.

## Component 2: Memory Cache (MC)

```python
class MemoryCache(nn.Module):
    """Checkpoint GSA states at intervals, retrieve via gated aggregation."""
    def __init__(self, state_dim, n_heads=4, top_k=4, checkpoint_interval=64):
        self.checkpoint_interval = checkpoint_interval
        self.top_k = top_k
        # Gated aggregation
        self.gate_proj = nn.Linear(state_dim, top_k, bias=True)
        # Cache is a list of (state, position) tuples in unified memory
        # NO data movement needed on Strix Halo!

    def checkpoint(self, state, position):
        """Store state checkpoint. On unified memory this is a pointer append."""
        if position % self.checkpoint_interval == 0:
            self.cache.append((state.detach(), position))

    def retrieve(self, query_state, current_pos):
        """Retrieve top-k nearest cached states via gated aggregation."""
        if len(self.cache) == 0:
            return torch.zeros_like(query_state)

        # Compute similarity to cached states
        cached_states = torch.stack([s for s, _ in self.cache])  # (N_cache, ...)
        similarities = torch.cosine_similarity(
            query_state.unsqueeze(0), cached_states, dim=-1
        )
        # Top-k retrieval
        topk_idx = similarities.topk(min(self.top_k, len(self.cache))).indices
        topk_states = cached_states[topk_idx]

        # Gated aggregation (learned weighted sum)
        gate = F.softmax(self.gate_proj(query_state.flatten())[:len(topk_idx)], dim=0)
        aggregated = (gate.unsqueeze(-1).unsqueeze(-1) * topk_states).sum(0)
        return aggregated

    def augment_state(self, gsa_state, gsa_output, current_pos):
        """Augment GSA output with memory cache retrieval."""
        self.checkpoint(gsa_state, current_pos)
        cached = self.retrieve(gsa_state, current_pos)
        # Residual blend: current state + cached context
        blend_gate = torch.sigmoid(self.blend_proj(gsa_output))
        return gsa_output + blend_gate * self.cache_proj(cached.flatten())
```

**Why this is free on Strix Halo:** On discrete GPUs, `cached_states` would live in CPU DRAM, requiring PCIe transfers for `torch.stack` and `cosine_similarity`. On Strix Halo, GPU and CPU share LPDDR5X — the stack operation is a pointer dereference, similarity is a GPU kernel on shared memory. **Zero data movement overhead.**

**Memory budget:** Each checkpoint = ~98K × 2B = 196KB. With 128GB unified memory, we can store ~650K checkpoints = ~41M tokens of context. **Effectively unlimited for our training runs.**

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_layers | 16 |
| GSA n_heads | 4 |
| GSA head_dim | 192 (768/4) |
| GSA n_slots | 64 |
| GSA damping τ | 8.0 |
| MC checkpoint_interval | 64 tokens |
| MC top_k | 4 |
| ffn_inner | 1920 (2.5×) |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×768, tied) | 38.6M |
| **Per GSA layer:** | |
|   w_q, w_k, w_v (768→768 each) | 1.77M |
|   w_o (768→768) | 0.59M |
|   w_alpha (768→256) | 0.20M |
|   **GSA subtotal** | **2.56M** |
| **Per MC module:** | |
|   gate_proj (state_dim→4) | ~2K |
|   blend_proj + cache_proj | ~1.2M |
|   **MC subtotal** | **~1.2M** |
| **Per SwiGLU FFN:** | 4.42M |
| **Per RMSNorm ×2:** | 1.5K |
| **Per block total** | **~8.20M** |
| **16 blocks** | **131.2M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **~169.8M** |

---

## Training

### 2 Phases

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (80%) | 36 min | GSA only (MC disabled) | Learn language with gated slot attention |
| 2 (20%) | 9 min | GSA + MC enabled | Learn memory retrieval and caching |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 8e-4 cosine → 8e-5, 150-step warmup |
| Weight decay | 0.1 |
| Batch | 24×1024, accum=2 (48K effective) |
| Precision | fp16 mixed + fp32 GSA state |
| Grad clip | 1.0 |
| MC cache limit | 1024 checkpoints per sequence (64K tokens lookback) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| GSA two-pass GLA is 2× cost of single-pass | MEDIUM | Each GLA pass uses compact 64-slot state, so both passes combined still cheaper than one full-state GLA |
| MC retrieval overhead at large cache sizes | LOW | Top-k on 1024 cached states = 1024 cosine similarities. Trivial on GPU. |
| MC gradients through cached states | MEDIUM | Detach cached states (no gradient). MC learns via gate_proj and blend_proj only. |
| FLA GSA kernel not optimized for gfx1151 | MEDIUM | GSA is two GLA passes. FLA GLA kernel works on gfx1151 (0.40ms verified for HGRN). |
| 64 slots too few for complex tasks | LOW | GSA paper shows 64 slots matches 256-dim GLA on commonsense. Softmax exponentially increases capacity. |

## Success Criteria

1. Val loss < 2.95 on BabyLM
2. Throughput > 10K tok/s (compile + autokernel)
3. MC improves long-context metrics (copying, retrieval tasks) by > 5%
4. Ablation: GSA outperforms equivalent GLA at same state size
5. Cache utilization > 50% (most queries retrieve non-trivial cached states)

---

## Implementation Roadmap

1. Implement GLA two-pass forward (adapt FLA library's GLA chunkwise)
2. Implement GSA module: two-pass GLA + softmax link + gating
3. Implement MemoryCache: checkpoint, retrieve, augment_state
4. Assemble TiamatBlock (GSA + MC + SwiGLU FFN)
5. Assemble TiamatModel (16 blocks), verify ~170M params
6. Phase 1: train GSA-only, 36 min
7. Phase 2: enable MC, train 9 min
8. Register GLA chunkwise as torch.library custom op
9. Benchmark: compare GSA vs GLA vs Mamba at same state budget
10. Long-context eval: needle-in-haystack at 2K, 4K, 8K contexts with MC

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), cross_entropy (1.8×), chunked_linear_cross_entropy

**External (1):** FLA GLA kernel (both passes of GSA, 0.40ms per pass verified)

**New (1):** MC retrieval kernel — cosine similarity + top-k on cached states. Could be HIP kernel but torch operations on shared memory are already fast.

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Eager fp16 | GSA only | ~7K tok/s |
| + autokernel | GSA only | ~9K tok/s |
| + compile | GSA only | ~12K tok/s |
| + FLA kernels | GSA only | ~13K tok/s |
| + MC enabled | GSA + MC | ~11K tok/s |

**GSA cost analysis:** Each layer = 2 × GLA pass (0.40ms each) + softmax (free) + projections (rocBLAS) + FFN (rocBLAS). Total per layer ~2.5ms. 16 layers = 40ms/step. At seq=1024: 1024/40ms = 25.6K tokens/step.

**MC overhead:** Top-4 retrieval from 1024-entry cache adds ~0.1ms per layer (cosine similarity on shared memory). Negligible vs GLA cost.

**Estimated optimized throughput:** ~11-13K tok/s (compile + autokernel + FLA)
**Tokens in 45 min:** ~30-35M (1.9-2.2 BabyLM epochs)
**Ranking:** #14 of 31 architectures
