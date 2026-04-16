---
title: "BIFROST"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/PROMETHEUS.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %bifrost, %kda, %flux-routing, %hybrid]
---

# BIFROST

**Bandwidth-Optimal KDA with Dynamic Flux Routing — The Rainbow Bridge Between Linear and Full Attention**

*"Not every layer needs the same attention. The bridge adapts its form to the weight of who crosses."*
*Kimi Linear proved KDA beats full attention. Flux Attention proved routing is better than static ratios.*

## Hypothesis

Kimi Delta Attention (KDA, Kimi Linear Oct 2025) with fine-grained channel-wise gating outperforms full attention even on short-context tasks. Instead of Kimi Linear's static 3:1 KDA-to-attention ratio, BIFROST uses Flux Attention's (Apr 2026) lightweight Layer Router to dynamically decide per-layer and per-input whether to use KDA or sliding-window attention. On Strix Halo, KDA is ~90% element-wise operations (all free), making it dramatically cheaper than attention while being equally expressive. The Flux Router adds negligible overhead (one Linear(d, 1) per layer) and lets easy inputs skip expensive attention entirely.

**Key papers:** "Kimi Linear" (2510.26692), "Flux Attention" (2604.07394)

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  → 16 Adaptive Hybrid Blocks:
  │     RMSNorm
  │     ┌──────────────────────────────────────────────┐
  │     │ Flux Router: σ(Linear(pool(h), 1))           │
  │     │   │                                          │
  │     │   ├─ route=KDA (p > 0.5):                    │
  │     │   │   KDA Token Mixer                        │
  │     │   │   Channel-wise gating: α_t ∈ [0,1]^{d_k}│
  │     │   │   Delta rule: β_t ∈ [0,1]               │
  │     │   │   ShortConv(k=3) + Swish + L2Norm       │
  │     │   │   Chunk-wise parallel (C=64)             │
  │     │   │                                          │
  │     │   └─ route=SWA (p ≤ 0.5):                   │
  │     │       Sliding Window Attention (w=256)       │
  │     │       NoPE (no positional encoding)          │
  │     │       GQA: 8Q/2KV heads                      │
  │     └──────────────────────────────────────────────┘
  │     +Residual
  │     RMSNorm → SwiGLU FFN (768→1920→768) → +Residual
  │
  → Final RMSNorm → LM Head
```

### Why Dynamic Routing

Kimi Linear's ablation shows:
- 0:1 (all attention): PPL 5.77
- 3:1 (75% KDA): PPL **5.65** (best)
- 7:1 (87.5% KDA): PPL 5.67 (close)
- Pure KDA: PPL 5.71

The optimal ratio depends on content. Factual retrieval needs attention. Fluent generation needs KDA's smooth decay. Flux Attention showed a trained router converges to content-dependent ratios that outperform any fixed schedule.

On Strix Halo, KDA layers cost ~0.3ms (element-wise dominant) while SWA layers cost ~3.5ms (matmul + softmax). If the router allocates ~12/16 layers to KDA and ~4/16 to SWA, effective cost is: 12×0.3 + 4×3.5 = 17.6ms vs 16×3.5 = 56ms for all-attention. **3.2× faster** at similar quality.

---

## Component 1: KDA Token Mixer (Kimi Delta Attention)

```python
class KDAMixer(nn.Module):
    """Kimi Delta Attention with fine-grained channel-wise gating."""
    def __init__(self, d_model=768, n_heads=12, head_dim=64, conv_kernel=3):
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Fused QKV (rocBLAS-friendly)
        self.w_qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        # ShortConv per head dim (following KDA paper)
        self.short_conv_q = nn.Conv1d(n_heads*head_dim, n_heads*head_dim,
                                       kernel_size=conv_kernel,
                                       groups=n_heads*head_dim,
                                       padding=conv_kernel-1)
        self.short_conv_k = nn.Conv1d(n_heads*head_dim, n_heads*head_dim,
                                       kernel_size=conv_kernel,
                                       groups=n_heads*head_dim,
                                       padding=conv_kernel-1)
        self.short_conv_v = nn.Conv1d(n_heads*head_dim, n_heads*head_dim,
                                       kernel_size=conv_kernel,
                                       groups=n_heads*head_dim,
                                       padding=conv_kernel-1)
        # Channel-wise decay: low-rank projection (rank=head_dim)
        self.w_alpha_down = nn.Linear(d_model, head_dim, bias=False)
        self.w_alpha_up = nn.Linear(head_dim, n_heads * head_dim, bias=False)
        # Scalar learning rate per head
        self.w_beta = nn.Linear(d_model, n_heads, bias=True)
        # Output gate (low-rank)
        self.w_gate_down = nn.Linear(d_model, head_dim, bias=False)
        self.w_gate_up = nn.Linear(head_dim, n_heads * head_dim, bias=False)
        # Head-wise RMSNorm
        self.head_norm = nn.GroupNorm(n_heads, n_heads * head_dim, affine=True)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # ShortConv + Swish (element-wise, free on Strix Halo)
        q = F.silu(self.short_conv_q(q.transpose(1,2))[:,:,:T].transpose(1,2))
        k = F.silu(self.short_conv_k(k.transpose(1,2))[:,:,:T].transpose(1,2))
        v = F.silu(self.short_conv_v(v.transpose(1,2))[:,:,:T].transpose(1,2))

        # L2Norm on Q/K (element-wise, free)
        q = F.normalize(q.view(B,T,self.n_heads,self.head_dim), dim=-1)
        k = F.normalize(k.view(B,T,self.n_heads,self.head_dim), dim=-1)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # Channel-wise decay α ∈ [0,1]^{d_k} (THE key innovation from KDA)
        alpha = torch.sigmoid(self.w_alpha_up(F.silu(self.w_alpha_down(x))))
        alpha = alpha.view(B, T, self.n_heads, self.head_dim)

        # Scalar learning rate β ∈ [0,1]
        beta = torch.sigmoid(self.w_beta(x)).unsqueeze(-1)

        # KDA chunk-wise (uses FLA DeltaNet-style kernel)
        o = kda_chunkwise(q, k, v, alpha, beta, chunk_size=64)

        # Output gate + head norm (element-wise, free)
        gate = torch.sigmoid(self.w_gate_up(F.silu(self.w_gate_down(x))))
        gate = gate.view(B, T, self.n_heads, self.head_dim)
        o = gate * self.head_norm(o.flatten(-2,-1).transpose(1,2)).transpose(1,2).view(B,T,self.n_heads,self.head_dim)
        return self.w_o(o.flatten(-2, -1))
```

**Element-wise op count per token:** ShortConv (free), Swish (free), L2Norm (free), sigmoid for α (free), sigmoid for β (free), sigmoid for gate (free), GroupNorm (free). The ONLY expensive op is the QKV projection matmul and the chunk-wise Q·K^T within KDA.

## Component 2: Sliding Window Attention (SWA)

```python
class SWAMixer(nn.Module):
    """Sliding window attention with NoPE — GQA 8Q/2KV."""
    def __init__(self, d_model=768, n_q_heads=8, n_kv_heads=2, head_dim=64, window=256):
        self.w_qkv = nn.Linear(d_model, (n_q_heads + 2*n_kv_heads) * head_dim, bias=False)
        self.w_o = nn.Linear(n_q_heads * head_dim, d_model, bias=False)
        self.window = window
        # NO positional encoding (NoPE) — KDA layers handle position

    def forward(self, x):
        # Standard GQA with sliding window mask, no RoPE
        # Uses hybrid_flash_sdpa_attention for training
        ...
```

**NoPE on SWA layers:** Following Kimi Linear's insight — KDA's channel-wise decay already encodes position (it acts as a data-dependent, learnable positional encoding). SWA layers with NoPE can be converted to pure MQA at inference for efficiency.

## Component 3: Flux Router

```python
class FluxRouter(nn.Module):
    """Lightweight per-layer router: KDA or SWA."""
    def __init__(self, d_model=768):
        self.proj = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.proj.bias, 1.0)  # Default to KDA

    def forward(self, x):
        # Pool over sequence dimension
        pooled = x.mean(dim=1)  # (B, D)
        logit = self.proj(pooled).squeeze(-1)  # (B,)
        if self.training:
            # Gumbel-sigmoid for differentiable routing
            noise = -torch.log(-torch.log(torch.rand_like(logit) + 1e-8) + 1e-8)
            route = torch.sigmoid((logit + noise) / 0.5)
        else:
            route = (logit > 0).float()
        return route  # 1.0 = KDA, 0.0 = SWA
```

**Training:** Gumbel-sigmoid with temperature annealing (0.5 → 0.1) enables gradient flow through the discrete routing decision. Aux loss encourages ~75% KDA usage (matching Kimi Linear's optimal ratio).

**Inference:** Hard routing (threshold at 0). Each layer deterministically picks KDA or SWA.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_layers | 16 |
| KDA heads | 12 (head_dim=64) |
| SWA: n_q_heads / n_kv_heads | 8 / 2 (GQA) |
| SWA window | 256 |
| ffn_inner | 1920 (2.5×) |
| conv_kernel | 3 |
| chunk_size | 64 |
| alpha_rank | 64 (low-rank decay projection) |
| gate_rank | 64 (low-rank output gate) |
| default_route | KDA (router bias=1.0) |
| router_aux_loss_weight | 0.01 |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×768, tied) | 38.6M |
| **Per KDA layer:** | |
|   w_qkv (768→2304) | 1.77M |
|   w_o (768→768) | 0.59M |
|   ShortConv ×3 (k=3, 768ch each) | 6.9K |
|   w_alpha (768→64→768) | 98K |
|   w_beta (768→12) | 9.2K |
|   w_gate (768→64→768) | 98K |
|   head_norm | 1.5K |
|   **KDA layer subtotal** | **~2.57M** |
| **Per SWA layer:** | |
|   w_qkv (768→640) | 0.49M |
|   w_o (512→768) | 0.39M |
|   **SWA layer subtotal** | **~0.88M** |
| **Per block (KDA + SWA + router + FFN):** | |
|   KDA mixer | 2.57M |
|   SWA mixer | 0.88M |
|   Flux Router (768→1) | 769 |
|   SwiGLU FFN | 4.42M |
|   RMSNorm ×2 | 1.5K |
|   **Block total** | **~7.88M** |
| **16 blocks** | **126.1M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **~164.7M** |

Note: Both KDA and SWA weights exist in every layer, but only one executes per forward pass. The router decides which path is active. Total parameter reads per step depend on routing decisions (~75% KDA typical).

---

## Training

### 2 Phases

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (70%) | 31 min | All layers as KDA only (router disabled) | Learn language with KDA |
| 2 (30%) | 14 min | Router enabled, both paths active | Learn routing + fine-tune SWA |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 8e-4 cosine → 8e-5, 150-step warmup |
| Weight decay | 0.1 |
| Batch | 24×1024, accum=2 (48K effective) |
| Precision | fp16 mixed + fp32 KDA state |
| Grad clip | 1.0 |
| Router aux loss | 0.01 × |KDA_ratio - 0.75|² |
| Gumbel temperature | 0.5 → 0.1 (linear anneal in Phase 2) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| KDA chunk-wise kernel availability | MEDIUM | FLA library has GLA/DeltaNet kernels; KDA is DeltaNet + channel-wise α, adaptable |
| Router collapses to one path | LOW | Aux loss on KDA ratio + initial bias toward KDA. Gumbel noise ensures exploration. |
| SWA layers unused waste parameters | LOW | 0.88M per SWA layer × 16 = 14M overhead. Acceptable even if rarely used. |
| Phase 2 too short for router convergence | MEDIUM | Phase 1 pre-trains both KDA and SWA (SWA gets gradient even if not routed, via straight-through estimator) |
| NoPE on SWA limits positional awareness | LOW | KDA layers handle all positional encoding via channel-wise decay. Validated by Kimi Linear at 3B scale. |

## Success Criteria

1. Val loss < 2.95 on BabyLM (2 epochs) — matching AMADEUS quality
2. Throughput > 14K tok/s (compile + autokernel)
3. Router learns non-trivial routing (not all-KDA or all-SWA)
4. Ablation: dynamic routing outperforms static 3:1 ratio by > 0.5%
5. KDA-only (Phase 1) already competitive with AMADEUS

---

## Implementation Roadmap

1. Implement KDAMixer with channel-wise alpha, ShortConv, L2Norm, output gate
2. Implement SWAMixer with GQA (8Q/2KV), sliding window, NoPE
3. Implement FluxRouter with Gumbel-sigmoid training, hard routing inference
4. Assemble BifrostBlock (router → KDA|SWA → FFN)
5. Assemble BifrostModel (16 blocks), verify ~165M params
6. Phase 1: train with router disabled (all KDA), 31 min
7. Phase 2: enable router, add aux loss, train 14 min
8. Register KDA chunk-wise as torch.library custom op for compile
9. Ablation: compare dynamic routing vs static 3:1 vs pure KDA
10. Measure per-layer routing decisions — visualize which layers prefer attention

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), cross_entropy (1.8×), chunked_linear_cross_entropy

**External (3):** causal-conv1d (10× ShortConv), FLA DeltaNet kernel (chunk-wise KDA), hybrid_flash_sdpa_attention (SWA training)

**New (0):** No new HIP kernels needed. KDA α_t computation is element-wise (PyTorch handles it). Router is a single Linear.

### Throughput Estimate

| Mode | Config | KDA layers | SWA layers | Throughput |
|------|--------|-----------|-----------|------------|
| Phase 1 (all KDA) | eager | 16 | 0 | ~9K tok/s |
| Phase 1 (all KDA) | compile+AK | 16 | 0 | ~15K tok/s |
| Phase 2 (dynamic ~12:4) | compile+AK | ~12 | ~4 | ~14K tok/s |
| Phase 2 (dynamic ~14:2) | compile+AK | ~14 | ~2 | ~16K tok/s |

**Effective bandwidth analysis:** KDA layers are dominated by the QKV projection (768→2304 = 1.77M × 2B = 3.5MB read). All gating/decay/delta-rule ops are element-wise (free). SWA layers additionally compute Q·K^T attention matrix (~3.5ms with hybrid_flash_sdpa). Dynamic routing to ~75% KDA saves ~40% wall-clock vs all-attention.

**Estimated optimized throughput:** ~15K tok/s (compile + autokernel + external kernels)
**Tokens in 45 min:** ~40.5M (2.5 BabyLM epochs)
**Ranking:** #10 of 31 architectures
