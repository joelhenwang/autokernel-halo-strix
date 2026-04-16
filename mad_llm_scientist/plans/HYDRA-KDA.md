---
title: "HYDRA-KDA"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/SPECTRAL-HYDRA.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %hydra-kda, %multi-resolution, %temporal-heads, %kda]
---

# HYDRA-KDA

**Multi-Resolution Temporal Heads with Entropy-Gated Routing — Many Heads, Many Timescales**

*"The Hydra's strength is not in any single head, but that each head sees a different horizon."*
*Language has multi-scale temporal structure. One decay rate fits none.*

## Hypothesis

Kimi Delta Attention (KDA) uses uniform heads with learned but undifferentiated channel-wise gating. Language processing inherently requires multiple temporal resolutions: function words need local context (fast decay), entity tracking needs long memory (slow decay), and in-context learning needs precise retrieval (strong delta updates). HYDRA-KDA initializes four distinct head groups with different temporal biases and couples them via an entropy-based cross-group router. Every operation — gating, decay, routing, entropy computation — is element-wise and therefore **free on Strix Halo**. This is a zero-cost expressivity upgrade over uniform-head KDA.

**Key papers:** "Kimi Linear" (2510.26692), "Hybrid Architectures Systematic Analysis" (2510.04800)

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  → 16 Multi-Resolution KDA Blocks:
  │     RMSNorm
  │     ┌──────────────────────────────────────────────────┐
  │     │ 4 Head Groups (3 heads each, 12 total):         │
  │     │                                                  │
  │     │ ⚡ REFLEX (3h): α₀~0.1, β₀~0.9                 │
  │     │   Fast decay, aggressive updates                 │
  │     │   "What just happened?"                          │
  │     │                                                  │
  │     │ 🧠 MEMORY (3h): α₀~0.95, β₀~0.3               │
  │     │   Slow decay, gentle updates                     │
  │     │   "What's the topic?"                            │
  │     │                                                  │
  │     │ 🔍 RETRIEVAL (3h): standard KDA, learned α/β    │
  │     │   Balanced decay, standard delta rule            │
  │     │   "Find the relevant context"                    │
  │     │                                                  │
  │     │ 🛡️ SENTINEL (3h): k=(1-α), HGRN2-style         │
  │     │   Coupled input-forget gates                     │
  │     │   "Selective attention to important tokens"      │
  │     │                                                  │
  │     │ ──── Entropy-Gated Cross-Group Router ────       │
  │     │ H(group_output) → softmax → weighted combine    │
  │     └──────────────────────────────────────────────────┘
  │     +Residual
  │     RMSNorm → SwiGLU FFN (768→1920→768) → +Residual
  │
  → Final RMSNorm → LM Head
```

---

## Component 1: Multi-Resolution Head Groups

All four groups share the same QKV projection but have **independent decay and learning rate parameters** with different initializations.

```python
class MultiResKDA(nn.Module):
    def __init__(self, d_model=768, n_groups=4, heads_per_group=3, head_dim=64):
        self.n_groups = n_groups
        self.n_heads = n_groups * heads_per_group  # 12 total
        self.head_dim = head_dim

        # Shared fused QKV projection (rocBLAS-friendly)
        self.w_qkv = nn.Linear(d_model, 3 * self.n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(self.n_heads * head_dim, d_model, bias=False)

        # Per-group ShortConv
        self.short_conv = nn.Conv1d(
            self.n_heads * head_dim, self.n_heads * head_dim,
            kernel_size=3, groups=self.n_heads * head_dim, padding=2
        )

        # Per-group channel-wise decay (low-rank, independent initialization)
        self.alpha_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, head_dim, bias=False),
                nn.SiLU(),
                nn.Linear(head_dim, heads_per_group * head_dim, bias=True)
            ) for _ in range(n_groups)
        ])

        # Per-group learning rate
        self.beta_projs = nn.ModuleList([
            nn.Linear(d_model, heads_per_group, bias=True)
            for _ in range(n_groups)
        ])

        # Cross-group entropy router
        self.router = EntropyRouter(d_model, n_groups)

        # Head-wise output norm + gate
        self.head_norm = nn.GroupNorm(self.n_heads, self.n_heads * head_dim)
        self.w_gate = nn.Linear(d_model, self.n_heads * head_dim, bias=False)

        # Initialize temporal biases
        self._init_temporal_biases()

    def _init_temporal_biases(self):
        """Initialize each group with distinct temporal behavior."""
        # Group 0 (REFLEX): fast decay, aggressive updates
        nn.init.constant_(self.beta_projs[0].bias, 2.0)   # sigmoid(2)≈0.88
        nn.init.constant_(self.alpha_projs[0][-1].bias, -2.0)  # sigmoid(-2)≈0.12

        # Group 1 (MEMORY): slow decay, gentle updates
        nn.init.constant_(self.beta_projs[1].bias, -1.0)  # sigmoid(-1)≈0.27
        nn.init.constant_(self.alpha_projs[1][-1].bias, 3.0)   # sigmoid(3)≈0.95

        # Group 2 (RETRIEVAL): balanced (standard KDA init)
        nn.init.constant_(self.beta_projs[2].bias, 0.0)   # sigmoid(0)=0.50
        nn.init.constant_(self.alpha_projs[2][-1].bias, 0.0)

        # Group 3 (SENTINEL): coupled gates (HGRN2-style)
        nn.init.constant_(self.beta_projs[3].bias, 0.5)
        nn.init.constant_(self.alpha_projs[3][-1].bias, 0.5)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # ShortConv + Swish (element-wise, free)
        q = F.silu(self.short_conv(q.transpose(1,2))[:,:,:T].transpose(1,2))
        k = F.silu(self.short_conv(k.transpose(1,2))[:,:,:T].transpose(1,2))
        v = F.silu(self.short_conv(v.transpose(1,2))[:,:,:T].transpose(1,2))

        # L2Norm Q/K
        q = F.normalize(q.view(B,T,self.n_heads,self.head_dim), dim=-1)
        k = F.normalize(k.view(B,T,self.n_heads,self.head_dim), dim=-1)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # Compute per-group alpha and beta
        group_outputs = []
        hpg = 3  # heads_per_group
        for g in range(self.n_groups):
            h_start, h_end = g * hpg, (g + 1) * hpg
            alpha_g = torch.sigmoid(self.alpha_projs[g](x)).view(B,T,hpg,self.head_dim)
            beta_g = torch.sigmoid(self.beta_projs[g](x)).unsqueeze(-1)

            # SENTINEL group: couple input and forget gates (k = 1 - α)
            if g == 3:
                k_g = (1 - alpha_g) * k[:,:,h_start:h_end]
            else:
                k_g = k[:,:,h_start:h_end]

            o_g = kda_chunkwise(
                q[:,:,h_start:h_end], k_g,
                v[:,:,h_start:h_end], alpha_g, beta_g,
                chunk_size=64
            )
            group_outputs.append(o_g)

        # Entropy-gated cross-group routing (element-wise, free)
        route_weights = self.router(x, group_outputs)  # (B, T, n_groups)
        o = torch.cat([
            route_weights[:,:,g:g+1].unsqueeze(-1) * group_outputs[g]
            for g in range(self.n_groups)
        ], dim=2)  # (B, T, n_heads, head_dim)

        # Output gate + norm
        gate = torch.sigmoid(self.w_gate(x)).view(B,T,self.n_heads,self.head_dim)
        o = gate * self.head_norm(o.flatten(-2,-1).transpose(1,2)).transpose(1,2).view(B,T,self.n_heads,self.head_dim)
        return self.w_o(o.flatten(-2, -1))
```

## Component 2: Entropy-Gated Cross-Group Router

```python
class EntropyRouter(nn.Module):
    """Route based on output entropy of each head group."""
    def __init__(self, d_model, n_groups):
        self.n_groups = n_groups
        self.proj = nn.Linear(d_model, n_groups, bias=True)
        nn.init.constant_(self.proj.bias, 0.0)  # Equal routing initially

    def forward(self, x, group_outputs):
        # Base routing from input
        base_route = self.proj(x)  # (B, T, n_groups)

        # Entropy signal from each group's output
        for g, o_g in enumerate(group_outputs):
            # Compute entropy of output distribution (element-wise, free)
            o_norm = F.softmax(o_g.flatten(-2,-1), dim=-1)
            entropy = -(o_norm * (o_norm + 1e-8).log()).sum(dim=-1, keepdim=True)
            base_route[:,:,g:g+1] = base_route[:,:,g:g+1] + entropy

        return F.softmax(base_route, dim=-1)  # (B, T, n_groups)
```

The router considers both the input representation (learned) and the **entropy of each group's output** — high-entropy outputs are "uncertain" and get downweighted. This is entirely element-wise.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_layers | 16 |
| n_groups | 4 |
| heads_per_group | 3 |
| n_heads (total) | 12 |
| head_dim | 64 |
| ffn_inner | 1920 (2.5×) |
| conv_kernel | 3 |
| chunk_size | 64 |
| alpha_rank | 64 |
| REFLEX init | α≈0.12, β≈0.88 |
| MEMORY init | α≈0.95, β≈0.27 |
| RETRIEVAL init | α≈0.50, β≈0.50 |
| SENTINEL init | α≈0.62, β≈0.62, coupled gates |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×768, tied) | 38.6M |
| **Per block:** | |
|   w_qkv (768→2304) | 1.77M |
|   w_o (768→768) | 0.59M |
|   ShortConv (k=3, 768ch) | 2.3K |
|   alpha_projs ×4 (768→64→192 each) | 4×61K = 0.24M |
|   beta_projs ×4 (768→3 each) | 4×2.3K = 9.2K |
|   EntropyRouter (768→4) | 3.1K |
|   w_gate (768→768) | 0.59M |
|   head_norm | 1.5K |
|   SwiGLU FFN (768→1920→768) | 4.42M |
|   RMSNorm ×2 | 1.5K |
|   **Block total** | **~7.63M** |
| **16 blocks** | **122.1M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **~160.7M** |

---

## Training

### Single Phase (the groups differentiate via initialization, not phasing)

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Full model | Groups self-specialize from distinct initializations |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 8e-4 cosine → 8e-5, 150-step warmup |
| Weight decay | 0.1 |
| Batch | 24×1024, accum=2 (48K effective) |
| Precision | fp16 mixed + fp32 KDA state |
| Grad clip | 1.0 |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Groups converge to uniform behavior | MEDIUM | Distinct initialization biases + per-group independent decay projections. Monitor α/β distributions per group during training. |
| Entropy router adds no benefit | LOW | Router is 3.1K params. Ablation: remove router, use uniform weighting. Zero-risk overhead. |
| SENTINEL coupled gates hurt quality | LOW | HGRN2 validated this parameterization. If underperforms, relax coupling. |
| 4-group KDA is 4× the alpha projection cost | LOW | Alpha projections are low-rank (768→64→192). 0.24M total vs 0.098M for single-group. Negligible vs FFN. |
| FLA kernel doesn't support per-group alpha | LOW | Each group calls kda_chunkwise independently. 4 calls × 3 heads = same cost as 1 call × 12 heads (FLA is head-parallel). |

## Success Criteria

1. Val loss < 2.90 on BabyLM (outperform AMADEUS)
2. Throughput > 13K tok/s (within 10% of uniform-head KDA)
3. Groups develop distinct α/β distributions (measured at convergence)
4. Ablation: multi-resolution > uniform initialization by > 1%
5. RETRIEVAL heads show higher activation on factual passages

---

## Implementation Roadmap

1. Implement KDA chunk-wise forward (adapt FLA DeltaNet kernel)
2. Implement per-group alpha/beta projections with distinct initializations
3. Implement SENTINEL group with coupled gates (k = 1 - α)
4. Implement EntropyRouter
5. Assemble MultiResKDA module
6. Assemble HydraBlock + HydraModel (16 layers), verify ~161M params
7. Smoke test: 10 min, verify loss decreasing + groups maintaining distinct distributions
8. Full training: BabyLM 2 epochs
9. Analysis: plot per-group α/β distributions at steps [100, 500, 1000, final]
10. Ablation: uniform init vs multi-resolution init (same architecture, different init)
11. Ablation: with vs without entropy router

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), cross_entropy (1.8×), chunked_linear_cross_entropy

**External (2):** causal-conv1d (10× ShortConv), FLA DeltaNet kernel (chunk-wise KDA)

**New (0):** All new operations (alpha projection, entropy computation, routing) are element-wise or small Linear layers — no custom kernels needed.

### Why This Is Free on Strix Halo

The ONLY difference from uniform-head KDA is:
1. **4 independent alpha projections** instead of 1 shared: 4 small matmuls (768→64→192) vs 1 (768→64→768). Net cost difference: ~0.15ms per layer. **Negligible.**
2. **Entropy computation per group**: softmax + log + sum. **Element-wise, free.**
3. **Routing weights**: one Linear(768→4) + softmax. **Trivial.**

All the temporal specialization happens through initialization and gating — operations that are already computed in standard KDA and cost nothing on this hardware.

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Eager fp16 | | ~9K tok/s |
| + autokernel | | ~12K tok/s |
| + compile | | ~14.5K tok/s |
| + FLA + causal-conv1d | | **~15K tok/s** |

**Cost overhead vs uniform KDA:** < 2% (0.6ms per forward pass from 4× alpha projections, negligible vs ~35ms total forward).

**Estimated optimized throughput:** ~14-15K tok/s (compile + autokernel + external kernels)
**Tokens in 45 min:** ~38-41M (2.4-2.5 BabyLM epochs)
**Ranking:** #9 of 31 architectures
