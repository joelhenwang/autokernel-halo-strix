---
title: "VALKYRIE"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/TEMPEST.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %valkyrie, %rwkv7, %kda, %efla, %best-of-breed]
---

# VALKYRIE

**RWKV-7 + KDA Gating + EFLA Exactness + Dynamic Layer Skipping — The Chooser of the Slain**

*"The Valkyrie chooses not the strongest, but the worthiest. From three architectures, she takes only their best qualities."*
*RWKV-7's expressivity. KDA's fine-grained gating. EFLA's exact dynamics. Each proven. Combined: unstoppable.*

## Hypothesis

Rather than inventing a new architecture, VALKYRIE takes the **proven best** from three recent breakthroughs and combines them into a single, clean design:

1. **From RWKV-7:** Generalized delta rule backbone, token shift, ReLU² FFN, decoupled add/remove keys. Proven SoTA at 3B scale for recurrent models.
2. **From KDA (Kimi Linear):** Fine-grained channel-wise gating (α ∈ [0,1]^{d_k}). Upgrades RWKV-7's vector-valued but dimension-matched gating to truly independent per-channel decay.
3. **From EFLA:** Exact ODE solution replacing Euler discretization. Zero error accumulation. Drop-in replacement of the α computation.

Plus **Flux-style dynamic layer skipping** for throughput: a tiny router per layer skips the entire token mixer when input entropy is low (only FFN executes). This is especially effective on Strix Halo where the FFN dominates bandwidth and the token mixer is mostly element-wise (free).

**This is the "safe bet" hypothesis — lowest risk, highest expected value.**

**Key papers:** "RWKV-7 Goose" (2503.14456), "Kimi Linear" (2510.26692), "Error-Free Linear Attention" (2512.12602), "Flux Attention" (2604.07394)

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  → 16 Valkyrie Blocks:
  │     RMSNorm
  │     ┌──────────────────────────────────────────────┐
  │     │ Skip Router: σ(Linear(pool(h), 1))           │
  │     │   │                                          │
  │     │   ├─ route=FULL (p > 0.3):                   │
  │     │   │   Token Shift (RWKV-7 lerp mixing)       │
  │     │   │   VALKYRIE Token Mixer:                   │
  │     │   │     RWKV-7 generalized delta rule         │
  │     │   │     + KDA channel-wise α ∈ [0,1]^{d_k}  │
  │     │   │     + EFLA exact ODE solution             │
  │     │   │     Chunk-wise parallel (C=64)            │
  │     │   │                                          │
  │     │   └─ route=SKIP (p ≤ 0.3):                  │
  │     │       Pass-through (identity)                 │
  │     └──────────────────────────────────────────────┘
  │     +Residual
  │     RMSNorm → ReLU² FFN (768→1920→768) → +Residual
  │
  → Final RMSNorm → LM Head
```

### The Three Upgrades Explained

**Upgrade 1: RWKV-7 → KDA channel-wise gating**

RWKV-7 state evolution:
```
S_t = S_{t-1} · (diag(w_t) - k̂ᵀ · (a_t ⊙ k̂_t)) + vᵀ · k_t
```
where w_t ∈ R^{d_k} is vector-valued but matched to the key dimension.

KDA upgrade: Replace w_t with truly independent per-channel decay α_t, computed via a low-rank projection:
```
α_t = sigmoid(W_α_up · SiLU(W_α_down · x_t))  ∈ [0,1]^{d_k}
```
This gives each feature dimension its own forgetting rate — finer control than RWKV-7's coupled decay.

**Upgrade 2: Euler → EFLA exact solution**

Both RWKV-7 and KDA use Euler discretization for the delta rule step. Replace with EFLA's exact closed form:
```python
lambda_t = (k_hat * k_hat).sum(dim=-1, keepdim=True)
efla_alpha = (1 - torch.exp(-beta_t * lambda_t)) / (lambda_t + 1e-6)
```
This is a scalar computation per token per head — trivial overhead, massive stability gain.

**Upgrade 3: Dynamic layer skipping**

Not every layer needs to process every token. Low-entropy inputs (predictable next tokens) can skip the expensive token mixer, keeping only the FFN for feature transformation. A tiny router (768→1 linear) decides per-layer.

---

## Component 1: VALKYRIE Token Mixer

```python
class ValkyrieTokenMixer(nn.Module):
    """RWKV-7 backbone + KDA channel-wise gating + EFLA exact dynamics."""
    def __init__(self, d_model=768, n_heads=12, head_dim=64):
        self.n_heads = n_heads
        self.head_dim = head_dim

        # RWKV-7 style projections (fused for rocBLAS)
        self.w_qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # RWKV-7: decoupled keys for add/remove
        self.w_k_hat = nn.Linear(d_model, n_heads * head_dim, bias=False)

        # KDA: channel-wise decay (low-rank projection)
        self.w_alpha_down = nn.Linear(d_model, head_dim, bias=False)
        self.w_alpha_up = nn.Linear(head_dim, n_heads * head_dim, bias=True)

        # RWKV-7: in-context learning rate (vector-valued per head)
        self.w_beta = nn.Linear(d_model, n_heads * head_dim, bias=True)

        # Token shift parameters (RWKV-7 lerp)
        self.token_shift = nn.Parameter(torch.zeros(5, d_model))

        # Output gate + head norm
        self.w_gate = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.head_norm = nn.GroupNorm(n_heads, n_heads * head_dim)

        # RWKV-7 bonus term
        self.bonus = nn.Parameter(torch.zeros(n_heads, head_dim))

    def forward(self, x, state=None):
        B, T, D = x.shape

        # Token shift (RWKV-7 lerp mixing, element-wise, free)
        x_shifted = lerp_token_shift(x, self.token_shift)

        # Projections (rocBLAS)
        qkv = self.w_qkv(x_shifted)
        q, k, v = qkv.chunk(3, dim=-1)
        k_hat = self.w_k_hat(x_shifted)  # Decoupled removal key

        # Reshape
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        k_hat = F.normalize(k_hat.view(B, T, self.n_heads, self.head_dim), dim=-1)

        # KDA channel-wise decay (element-wise, free)
        alpha = torch.sigmoid(self.w_alpha_up(F.silu(self.w_alpha_down(x_shifted))))
        alpha = alpha.view(B, T, self.n_heads, self.head_dim)

        # RWKV-7 vector-valued learning rate (element-wise, free)
        beta = torch.sigmoid(self.w_beta(x_shifted))
        beta = beta.view(B, T, self.n_heads, self.head_dim)

        # EFLA exact ODE solution (element-wise, free)
        lambda_t = (k_hat * k_hat).sum(dim=-1, keepdim=True)  # scalar per head
        efla_factor = (1 - torch.exp(-beta.mean(dim=-1, keepdim=True) * lambda_t)) / (lambda_t + 1e-6)

        # Generalized delta rule with EFLA exactness
        # S_t = S_{t-1} · (diag(α_t) - efla_factor · k̂ᵀ · (β_t ⊙ k̂_t)) + vᵀ · k_t
        o = valkyrie_chunkwise(q, k, v, k_hat, alpha, beta, efla_factor,
                                self.bonus, chunk_size=64, state=state)

        # Output gate + head norm (element-wise, free)
        gate = torch.sigmoid(self.w_gate(x))
        gate = gate.view(B, T, self.n_heads, self.head_dim)
        o = gate * self.head_norm(o.flatten(-2,-1).transpose(1,2)).transpose(1,2).view(B,T,self.n_heads,self.head_dim)
        return self.w_o(o.flatten(-2, -1))
```

## Component 2: ReLU² FFN (RWKV-7 style)

```python
class ReluSquaredFFN(nn.Module):
    """RWKV-7 uses ReLU² instead of SwiGLU. Slightly cheaper, competitive quality."""
    def __init__(self, d_model=768, ffn_inner=1920):
        self.w_up = nn.Linear(d_model, ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.relu(self.w_up(x)).square())
```

**Note:** ReLU² has fewer parameters than SwiGLU (no gate projection). This saves ~1.5M params per layer. The squared activation provides sufficient nonlinearity.

## Component 3: Skip Router

```python
class SkipRouter(nn.Module):
    """Per-layer decision to skip token mixer. Default: active."""
    def __init__(self, d_model=768, skip_threshold=0.3):
        self.proj = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.proj.bias, 2.0)  # Default: don't skip (sigmoid(2)=0.88)
        self.threshold = skip_threshold

    def forward(self, x):
        pooled = x.mean(dim=1)  # (B, D)
        logit = self.proj(pooled)
        if self.training:
            # Gumbel noise for differentiable routing
            noise = -torch.log(-torch.log(torch.rand_like(logit) + 1e-8) + 1e-8)
            return torch.sigmoid((logit + noise) / 0.5)
        else:
            return (logit > 0).float()
```

**Expected behavior:** ~80% of layers execute the full token mixer, ~20% skip. Saves ~20% of token mixer computation at minimal quality cost.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_layers | 16 |
| n_heads | 12 |
| head_dim | 64 |
| ffn_inner | 1920 (ReLU², no gate — 2.5×) |
| conv_kernel | — (RWKV-7 uses token shift, not conv) |
| chunk_size | 64 |
| alpha_rank | 64 (KDA low-rank decay) |
| token_shift_groups | 5 (RWKV-7) |
| skip_threshold | 0.3 |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×768, tied) | 38.6M |
| **Per block:** | |
|   w_qkv (768→2304) | 1.77M |
|   w_k_hat (768→768) | 0.59M |
|   w_o (768→768) | 0.59M |
|   w_alpha (768→64→768) | 0.10M |
|   w_beta (768→768) | 0.59M |
|   w_gate (768→768) | 0.59M |
|   token_shift (5×768) | 3.8K |
|   bonus (12×64) | 768 |
|   head_norm | 1.5K |
|   SkipRouter (768→1) | 769 |
|   ReLU² FFN (768→1920→768) | 2.95M |
|   RMSNorm ×2 | 1.5K |
|   **Block total** | **~7.19M** |
| **16 blocks** | **115.0M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **~153.6M** |

Lean model: 153.6M params. ReLU² FFN saves ~1.5M/layer vs SwiGLU. Well under the 175M target.

---

## Training

### Single Phase

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Full model, skip router active from start | Learn language with best-of-breed recurrence |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 8e-4 cosine → 8e-5, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 24×1024, accum=2 (48K effective) |
| Precision | fp16 mixed + fp32 state evolution |
| Grad clip | 1.0 |
| Skip router: Gumbel temp | 0.5 → 0.1 (anneal) |

**RWKV-7 specific inits (from paper, Appendix E):**
- A_log: -1.0 (stable eigenvalues)
- w_beta bias: -3.0 (conservative learning rate)
- bonus: zeros (no initial attention sink compensation)
- token_shift: zeros (equal mixing initially)

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| RWKV-7 + KDA + EFLA integration complexity | MEDIUM | Each upgrade is a drop-in modification. KDA = replace w with per-channel α. EFLA = replace β with exact formula. Test incrementally. |
| EFLA factor interacts poorly with RWKV-7's decoupled keys | MEDIUM | EFLA factor applies to the removal term (k̂ branch). If unstable, fall back to standard beta. Ablation: EFLA vs Euler. |
| ReLU² FFN underperforms SwiGLU | LOW | RWKV-7 paper validates ReLU² at scale. If needed, switch to SwiGLU (adds ~1.5M/layer). |
| Skip router collapses to always-active | LOW | Bias initialized high (don't skip). If no skipping emerges, router adds ~0.8K overhead. Zero risk. |
| FLA kernel needs modification for generalized delta rule | MEDIUM | RWKV-7 chunkwise is supported in FLA (RWKV kernel). KDA gating can be pre-applied before kernel call. |

## Success Criteria

1. Val loss < 2.85 on BabyLM (**beat AMADEUS**)
2. Throughput > 14K tok/s (compile + autokernel)
3. EFLA ablation: exact > Euler by > 0.5% loss improvement
4. KDA ablation: channel-wise > vector-valued gating by > 0.3%
5. Skip router ablation: skipping saves > 15% compute with < 0.5% quality loss

---

## Implementation Roadmap

1. Implement RWKV-7 base: token shift, generalized delta rule, bonus term
2. Upgrade: replace vector-valued w with KDA channel-wise α (low-rank projection)
3. Upgrade: replace Euler β with EFLA exact factor
4. Implement ReLU² FFN
5. Implement SkipRouter
6. Assemble ValkyrieBlock + ValkyrieModel (16 layers), verify ~154M params
7. Smoke test: 10 min, verify loss decreasing + state stability
8. Register valkyrie_chunkwise as torch.library custom op (adapt FLA RWKV kernel)
9. Full training: BabyLM 2 epochs
10. Ablation cascade: RWKV-7 base → +KDA → +EFLA → +SkipRouter (measure each upgrade)
11. Compare against AMADEUS, TEMPEST, PROMETHEUS on quality and throughput

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (3):** fused_residual_add_rmsnorm (6.6×), cross_entropy (1.8×), chunked_linear_cross_entropy

**External (1):** FLA RWKV kernel (chunk-wise generalized delta rule, verified on gfx1151)

**New (0):** ReLU² is `F.relu(x).square()` — two element-wise ops (free). All KDA/EFLA additions are element-wise.

### Why Every Upgrade Is Free on Strix Halo

| Upgrade | Operations Added | Cost on Strix Halo |
|---------|-----------------|-------------------|
| KDA α projection | 768→64 + SiLU + 64→768 | ~0.05ms (tiny matmuls) |
| KDA sigmoid(α) | element-wise sigmoid | **Free** |
| EFLA factor | sum, exp, division | **Free** |
| Token shift lerp | element-wise multiply+add | **Free** |
| Bonus term | element-wise add | **Free** |
| Skip router | 768→1 + sigmoid | **Free** |
| **Total overhead vs plain linear attention** | | **~0.05ms/layer ≈ 0.8ms total** |

VALKYRIE adds ~0.8ms to a forward pass that takes ~35ms. **2.3% overhead for three major expressivity upgrades.**

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Eager fp16 | | ~9K tok/s |
| + autokernel | | ~12K tok/s |
| + compile | | ~16K tok/s |
| + FLA RWKV kernel | | ~17K tok/s |
| + layer skipping (~20% skip) | | **~19K tok/s** |

**Estimated optimized throughput:** ~17-19K tok/s (compile + autokernel + FLA + skip)
**Tokens in 45 min:** ~46-51M (2.9-3.2 BabyLM epochs)
**Ranking:** #5 of 31 architectures

**This is the recommended first implementation — lowest risk, highest expected quality, competitive throughput.**
