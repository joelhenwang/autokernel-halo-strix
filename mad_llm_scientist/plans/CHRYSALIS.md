---
title: "CHRYSALIS"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/JORMUNGANDR.md
  - mad_llm_scientist/plans/RESONANT-LOOP.md
  - knowledge/architectures/hypothesis_buildout_results.md
  - knowledge/kernels/mHC_MoE_Engram_optimizations.md
tags: [%hypothesis, %plan, %chrysalis, %efla, %looped, %moe, %triple-efficiency]
---

# CHRYSALIS

**Looped EFLA + Scatter-MoE: The Triple Efficiency Stack — Transformation Through Layered Economy**

*"Inside the chrysalis, three forces conspire: compression (looping), precision (EFLA), and selection (MoE). What emerges is greater than their sum."*
*Each technique addresses a different bottleneck. Together they multiply.*

## Hypothesis

Three orthogonal efficiency techniques, when composed, **multiply** rather than add their benefits:

1. **Looped architecture** (Parcae-style) → fewer unique parameters → L2 cache reuse
2. **Error-Free Linear Attention** → exact dynamics → stable across loop iterations
3. **Scatter-MoE FFN** → sparse activation → fewer memory reads per token in the FFN (which is 2/3 of model params)

On Strix Halo where bandwidth is the #1 bottleneck, this triple stack minimizes memory reads: looping means fewer unique weights to read (L2-cached), MoE means only 2/8 experts are read per token, and EFLA means the attention mechanism is element-wise (free). This should push throughput beyond JORMUNGANDR's 43K tok/s while maintaining competitive quality through EFLA's stability and MoE's capacity.

**Key papers:** "Error-Free Linear Attention" (2512.12602), ScatterMoE (verified on gfx1151), Parcae looped training

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  → 1 SHARED BLOCK × 10 iterations:
  │     RMSNorm
  │     ┌──────────────────────────────────────────┐
  │     │ EFLA Token Mixer (same as EREBUS)         │
  │     │   12 heads, head_dim=64                  │
  │     │   ShortConv + Swish + L2Norm             │
  │     │   Exact ODE solution (α_t formula)       │
  │     │   Chunk-wise parallel (C=64)             │
  │     └──────────────────────────────────────────┘
  │     +Residual
  │     RMSNorm
  │     ┌──────────────────────────────────────────┐
  │     │ Scatter-MoE FFN                          │
  │     │   8 experts, top-2 routing               │
  │     │   Each expert: SwiGLU(768 → 640 → 768)  │
  │     │   Capacity factor: 1.25                  │
  │     │   Load balancing: aux loss 0.01          │
  │     │   ★ Only 2/8 experts loaded per token ★  │
  │     └──────────────────────────────────────────┘
  │     +Residual
  │
  │     ★ Iteration-aware: layer_id = iteration_idx ★
  │     ★ Stochastic depth: Poisson(λ=10) ★
  │
  → Final RMSNorm → LM Head
```

### The Triple Efficiency Multiplication

| Technique | What It Saves | Multiplicative Factor |
|-----------|--------------|----------------------|
| Looping (10 iter) | Unique params: 175M → ~17M per block | ~5× param read reduction (L2 cached) |
| MoE (8 exp, top-2) | FFN reads: 100% → 25% per token | ~4× FFN bandwidth reduction |
| EFLA | Attention compute: matmul → element-wise | ~∞× for non-matmul ops (free) |

**Combined:** For a 1024-token sequence, weight reads = ~17M × 2B = 34MB (shared block, L2-cached after iter 1) + 2 experts × 1.3MB = 2.6MB per token's FFN. EFLA adds zero bandwidth cost. Effective read: ~37MB per iteration. **vs standard 175M model: 350MB per pass.** That's ~9.5× less bandwidth.

---

## Component 1: EFLA Token Mixer

Same as EREBUS — exact ODE solution for delta rule. See EREBUS.md for full implementation.

Key parameters: 12 heads, head_dim=64, ShortConv(k=3), L2Norm, output gate, chunk_size=64.

## Component 2: Scatter-MoE FFN

```python
# Using scattermoe 0.3.0 (verified on gfx1151)
from scattermoe.mlp import MLP as ScatterMLP

class MoEFFN(nn.Module):
    def __init__(self, d_model=768, n_experts=8, top_k=2, expert_dim=640):
        self.n_experts = n_experts
        self.top_k = top_k
        # Router
        self.router = nn.Linear(d_model, n_experts, bias=False)
        # ScatterMoE: fused dispatch + expert forward + gather
        self.experts = ScatterMLP(
            input_size=d_model,
            hidden_size=expert_dim,
            num_experts=n_experts,
            top_k=top_k,
            activation=F.silu  # SwiGLU-like
        )

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B*T, D)

        # Router logits
        logits = self.router(x_flat)  # (B*T, n_experts)
        # Top-k routing with softmax
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # ScatterMoE: groups tokens by expert, processes contiguously
        out = self.experts(x_flat, weights, indices)
        return out.view(B, T, D)

    def aux_loss(self, logits):
        """Load balancing auxiliary loss."""
        # Encourage uniform expert utilization
        router_probs = F.softmax(logits, dim=-1)
        avg_probs = router_probs.mean(dim=0)
        return (avg_probs * torch.log(avg_probs + 1e-8)).sum() * self.n_experts
```

**Why Scatter-MoE on Strix Halo:** ScatterMoE groups tokens by expert assignment before computing, ensuring contiguous memory access. On a bandwidth-limited chip like Strix Halo, this is critical — scattered reads waste bandwidth. Each expert is a small SwiGLU (768→640→768) = 1.28M params = 2.56MB. Top-2 routing reads 5.12MB of expert weights per forward pass instead of 10.24MB (4 experts) or 40.96MB (all 8).

## Component 3: Iteration-Aware Block

```python
class ChrysalisBlock(nn.Module):
    def __init__(self, d_model=768, n_iterations=10):
        self.efla = EFLAMixer(d_model, n_heads=12, head_dim=64)
        self.moe_ffn = MoEFFN(d_model, n_experts=8, top_k=2, expert_dim=640)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        # Iteration embedding (tells the block which iteration it is)
        self.iter_embed = nn.Embedding(n_iterations, d_model)

    def forward(self, x, iteration_idx, state=None):
        # Add iteration embedding (element-wise, free)
        iter_bias = self.iter_embed(torch.tensor(iteration_idx, device=x.device))
        x_biased = x + iter_bias.unsqueeze(0).unsqueeze(0)  # Broadcast

        h = x + self.efla(self.norm1(x_biased), state=state)
        h = h + self.moe_ffn(self.norm2(h))
        return h
```

The iteration embedding lets the shared block behave differently at each iteration (like positional encoding for depth). Cost: one embedding lookup + element-wise add (free).

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_iterations | 10 |
| shared_blocks | 1 |
| EFLA n_heads | 12 |
| EFLA head_dim | 64 |
| MoE n_experts | 8 |
| MoE top_k | 2 |
| expert_dim (FFN inner) | 640 |
| conv_kernel | 3 |
| chunk_size | 64 |
| iter_embed_dim | 768 |
| capacity_factor | 1.25 |
| aux_loss_weight | 0.01 |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×768, tied) | 38.6M |
| **Shared block:** | |
|   EFLA mixer (w_qkv, w_o, conv, beta, gate, norm) | ~3.0M |
|   MoE router (768→8) | 6.1K |
|   MoE experts: 8 × SwiGLU(768→640→768) | 8 × 1.28M = **10.24M** |
|   Iteration embedding (10×768) | 7.7K |
|   RMSNorm ×2 | 1.5K |
|   **Block total** | **~13.25M** |
| **×1 block (shared across 10 iterations)** | **13.25M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL (unique)** | **~51.9M** |
| **Effective params (10 iterations, top-2/8)** | **~130M effective** |
| **Active params per token** | **~6.3M** (EFLA mixer + 2 experts + overhead) |

**The key insight:** Only 51.9M unique parameters. The core block is ~13.25M = **26.5MB in fp16**. This doesn't fit entirely in L2 (6MB), but:
- EFLA mixer (6MB) ≈ L2 boundary
- Active expert pair (5.1MB) ≈ L2 boundary
- After iteration 1, hot paths are L2-cached

---

## Training

### Single Phase

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Full model, 10 iterations, MoE active | Learn language with triple efficiency |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-3 cosine → 1e-4, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 32×1024, accum=2 (64K effective) |
| Precision | fp16 mixed + fp32 EFLA state |
| Grad clip | 1.0 |
| MoE aux loss weight | 0.01 |
| Stochastic depth | Poisson(λ=10), clamped [6, 14] |
| Expert dropout | 0.1 (during training, one random expert masked) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| MoE load imbalance across experts | MEDIUM | Aux loss + expert dropout. Monitor per-expert utilization. |
| ScatterMoE + torch.compile interaction | MEDIUM | ScatterMoE is Triton-based. May need to wrap as custom op for compile compatibility. |
| Small expert dim (640) limits capacity | LOW | 8 experts × 640 = effective 5120-dim FFN (vs 1920 dense). Net 2.7× more capacity with same active params. |
| Iteration embedding overfits to specific iterations | LOW | Stochastic depth varies iteration count. Embedding generalizes. |
| Total unique params (52M) too few for quality | MEDIUM | 10 iterations + MoE gives 130M effective. If quality insufficient, increase to 12 iterations. |

## Success Criteria

1. Throughput > 40K tok/s (compile + autokernel) — surpassing JORMUNGANDR
2. Val loss < 3.15 on BabyLM (respectable quality at extreme throughput)
3. Expert utilization balanced within 2× of mean
4. Ablation: MoE + loop > dense + loop at same unique params
5. Ablation: EFLA stability across 10-14 iterations (no quality degradation)

---

## Implementation Roadmap

1. Implement EFLA mixer (reuse from EREBUS)
2. Integrate ScatterMoE (scattermoe 0.3.0, verified on gfx1151)
3. Implement MoE router with aux loss
4. Implement ChrysalisBlock with iteration embedding
5. Implement ChrysalisModel with Parcae-style loop
6. Verify parameter count (~52M unique)
7. Smoke test: 10 min, verify > 35K tok/s + MoE load balance
8. Wrap ScatterMoE as torch.library custom op for compile compatibility
9. Full training: BabyLM 2 epochs
10. Ablation: dense FFN vs MoE FFN (same unique params)
11. Ablation: iteration count [6, 8, 10, 12] quality scaling

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), cross_entropy (1.8×), chunked_linear_cross_entropy

**External (3):** causal-conv1d (10×), FLA DeltaNet (EFLA chunkwise), scattermoe 0.3.0 (fused MoE)

**New (0):** No custom kernels needed. All components use existing optimized implementations.

### Bandwidth Analysis (The Real Optimization)

| Component | Unique Weights | Reads per Forward | Notes |
|-----------|---------------|-------------------|-------|
| Embedding | 38.6M (77.2MB) | 1× | Streamed, not L2-cacheable |
| EFLA mixer | 3.0M (6.0MB) | 10× (L2-cached after iter 1) | L2 boundary — partially cached |
| MoE router | 6.1K (12KB) | 10× (L2-cached) | Trivially L2-cached |
| Active experts (top-2) | 2.56M (5.1MB) | 10× per token pair | Different experts per token |
| Inactive experts | 7.68M (15.4MB) | 0× | Never read! Bandwidth saved. |
| RMSNorm + misc | ~5K (10KB) | 10× (L2-cached) | Trivial |

**Total bandwidth per forward pass:**
- Embedding: 77.2MB (1×)
- EFLA: 6.0MB × 1 (cached) + element-wise (free)
- MoE: 5.1MB × 10 = 51MB (varies by token routing)
- Total: ~130MB ← vs standard 175M model: 350MB

**Bandwidth savings: ~2.7×** from loop caching + MoE sparsity.

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Eager fp16 | | ~18K tok/s |
| + autokernel | | ~25K tok/s |
| + compile | | ~38K tok/s |
| + ScatterMoE + FLA + causal-conv1d | | **~45K tok/s** |

**Estimated optimized throughput:** ~42-48K tok/s (compile + autokernel + all external kernels)
**Tokens in 45 min:** ~113-130M (7.1-8.1 BabyLM epochs)
**Ranking:** #1 of 31 architectures (highest throughput hypothesis)
