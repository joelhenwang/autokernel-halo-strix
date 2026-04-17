---
title: "AEGIS"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/EREBUS.md
  - mad_llm_scientist/plans/VALKYRIE.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %aegis, %kalman, %bayesian, %adaptive-depth, %looped]
---

# AEGIS

**Kalman-Filtered Recurrence with Uncertainty-Gated Adaptive Depth — The Shield That Knows Its Blind Spots**

*"The aegis does not protect equally against all threats — it concentrates its power where uncertainty is greatest. What the shield doesn't know, it guards against most fiercely."*
*Kalman Linear Attention proved Bayesian filtering is parallelizable. AEGIS proves uncertainty is the optimal depth router.*

## Hypothesis

Kalman Linear Attention (Feb 2026) reframes sequence modeling as Bayesian filtering — the Kalman filter in *information form* is parallelizable via associative scan AND produces explicit per-token uncertainty estimates as a FREE byproduct. In a looped architecture, this uncertainty signal is the **optimal routing criterion for adaptive depth**: high-uncertainty tokens get more iterations, low-uncertainty tokens get fewer. Unlike learned routers (Flux, ACT) that train a separate signal from scratch, AEGIS gets its routing signal for free from the Kalman posterior — a principled, information-theoretically grounded measure of what the model doesn't yet know.

**Key paper:** "Kalman Linear Attention" (2602.10743, February 2026)

---

## Architecture

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  -> 1 SHARED BLOCK x adaptive iterations:
  |     RMSNorm
  |     +---------------------------------------------+
  |     | Kalman Linear Attention (KLA)                |
  |     |   State: (mu_t, Sigma_t) belief state        |
  |     |   Information form: (J_t, h_t) = (Sigma^-1,  |
  |     |                                  Sigma^-1*mu)|
  |     |   Associative scan (FLA-compatible)           |
  |     |   FREE: uncertainty = diag(Sigma_t)           |
  |     +---------------------------------------------+
  |     +Residual
  |     RMSNorm -> SwiGLU FFN (768->1920->768) -> +Residual
  |
  |     * Per-token adaptive depth: *
  |     *   uncertainty_t = mean(diag(Sigma_t))         *
  |     *   n_iters_t = clip(ceil(uncertainty/tau),2,16)*
  |     *   Low uncertainty: 2-3 iters (60% of tokens)  *
  |     *   Medium: 6-8 iters (30%)                     *
  |     *   High: 12-16 iters (10%)                     *
  |
  -> Final RMSNorm -> LM Head
```

### Why Kalman Filtering for Language Modeling

Standard linear attention computes a weighted sum over state. Kalman filtering interprets this as **Bayesian inference**: the state S_t is a posterior belief about the sequence, updated with each new observation (token). The key advantage: the posterior ALSO encodes uncertainty — how confident the model is about its current representation.

**Standard DeltaNet:**
```
S_t = (I - beta * k @ k.T) @ S_{t-1} + beta * k @ v.T
output = q.T @ S_t
```

**Kalman Linear Attention:**
```
# Prediction step (prior)
J_pred = decay * J_{t-1}
h_pred = decay * h_{t-1}

# Update step (posterior, incorporating new observation)
K_t = J_pred @ k / (k.T @ J_pred @ k + sigma_obs^2)  # Kalman gain
J_t = J_pred + k @ k.T / sigma_obs^2                   # precision update
h_t = h_pred + k * v / sigma_obs^2                      # information update

# Output
mu_t = J_t^{-1} @ h_t        # posterior mean (= standard output)
Sigma_t = J_t^{-1}            # posterior covariance (= FREE uncertainty!)
```

In information form (J, h), the update is additive and **parallelizable via associative scan**.

### Uncertainty as Depth Router

The diagonal of Sigma_t gives per-dimension uncertainty. Averaging across dimensions yields a scalar uncertainty score per token. This is **information-theoretically optimal** for deciding computation allocation:

- **Low uncertainty** (e.g., "the cat sat on the ___"): The model is confident. 2-3 iterations suffice.
- **High uncertainty** (e.g., first token of a new topic): The model needs more refinement. 12-16 iterations.

No other adaptive-depth scheme has this property. ACT learns a halting signal from scratch. Flux routes at the layer level. AEGIS uses the model's own Bayesian confidence — it literally knows what it doesn't know.

---

## Component 1: Kalman Linear Attention

```python
class KalmanLinearAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, head_dim=64, sigma_obs=1.0):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.sigma_obs_sq = sigma_obs ** 2
        self.w_qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.w_decay = nn.Linear(d_model, n_heads, bias=True)
        self.w_gate = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.head_norm = nn.GroupNorm(n_heads, n_heads * head_dim)

    def forward(self, x, state=None):
        B, T, D = x.shape
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = F.normalize(k.view(B, T, self.n_heads, self.head_dim), dim=-1)
        v = v.view(B, T, self.n_heads, self.head_dim)

        decay = torch.sigmoid(self.w_decay(x)).unsqueeze(-1)

        # Information form update (parallelizable)
        # J_update = k @ k.T / sigma^2  (precision increment)
        # h_update = k * v / sigma^2     (information increment)
        precision_inc = (k.unsqueeze(-1) * k.unsqueeze(-2)) / self.sigma_obs_sq
        info_inc = k * v / self.sigma_obs_sq

        # Chunk-wise associative scan in information form
        o, uncertainty = kalman_chunkwise(
            q, k, v, decay, precision_inc, info_inc,
            self.sigma_obs_sq, chunk_size=64, state=state
        )
        # uncertainty: (B, T, n_heads) — mean diagonal of Sigma_t

        gate = torch.sigmoid(self.w_gate(x)).view(B, T, self.n_heads, self.head_dim)
        o = gate * self.head_norm(
            o.flatten(-2, -1).transpose(1, 2)
        ).transpose(1, 2).view(B, T, self.n_heads, self.head_dim)
        return self.w_o(o.flatten(-2, -1)), uncertainty
```

## Component 2: Adaptive Depth Controller

```python
class AdaptiveDepthController:
    def __init__(self, tau=0.5, min_iters=2, max_iters=16):
        self.tau = tau
        self.min_iters = min_iters
        self.max_iters = max_iters

    def compute_depths(self, uncertainty):
        """Convert per-token uncertainty to iteration counts.
        
        Args:
            uncertainty: (B, T) mean uncertainty scores
        Returns:
            depths: (B, T) integer iteration counts
            groups: list of (mask, n_iters) for batched execution
        """
        depths = torch.clamp(
            torch.ceil(uncertainty / self.tau).int(),
            self.min_iters, self.max_iters
        )
        # Group tokens by depth for efficient batched execution
        unique_depths = depths.unique().tolist()
        groups = []
        for d in sorted(unique_depths):
            mask = (depths == d)
            groups.append((mask, d))
        return depths, groups
```

## Component 3: AEGIS Looped Model

```python
class AegisModel(nn.Module):
    def __init__(self, d_model=768, base_iterations=12):
        self.embedding = nn.Embedding(50257, d_model)
        self.shared_block = AegisBlock(d_model)
        self.depth_controller = AdaptiveDepthController(tau=0.5)
        self.base_iterations = base_iterations
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, 50257, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        h = self.embedding(input_ids)

        # Initial pass to get uncertainty estimate
        h_probe, uncertainty = self.shared_block(h, return_uncertainty=True)

        if self.training:
            # Fixed depth during training (simpler, more stable)
            n_iter = min(max(
                torch.poisson(torch.tensor(float(self.base_iterations))).int().item(),
                8), 16)
            for i in range(n_iter):
                h, uncertainty = self.shared_block(h, return_uncertainty=True)
        else:
            # Adaptive depth during inference
            mean_uncertainty = uncertainty.mean(dim=-1)  # (B, T)
            depths, groups = self.depth_controller.compute_depths(mean_uncertainty)
            for mask, n_iters in groups:
                h_subset = h[mask]
                for i in range(n_iters):
                    h_subset, _ = self.shared_block(h_subset)
                h[mask] = h_subset

        return self.lm_head(self.final_norm(h))
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_heads | 12 |
| head_dim | 64 |
| ffn_inner | 1920 (2.5x) |
| n_iterations | 12 (train: Poisson, eval: adaptive 2-16) |
| shared_blocks | 1 |
| sigma_obs | 1.0 (observation noise, learnable) |
| tau | 0.5 (depth threshold) |
| min_iters | 2 |
| max_iters | 16 |
| chunk_size | 64 |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257x768, tied) | 38.6M |
| **Shared block:** | |
|   KLA: w_qkv (768->2304) | 1.77M |
|   KLA: w_o (768->768) | 0.59M |
|   KLA: w_decay (768->12) | 9.2K |
|   KLA: w_gate (768->768) | 0.59M |
|   KLA: head_norm | 768 |
|   SwiGLU FFN: w_gate_up (768->3840) | 2.95M |
|   SwiGLU FFN: w_down (1920->768) | 1.47M |
|   RMSNorm x2 | 1.5K |
|   **Block total** | **~7.38M** |
| **x1 block (shared across 12 iterations)** | **7.38M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL (unique)** | **~46.0M** |
| **Effective params (12 avg iterations)** | **~127M effective** |

Same parameter budget as EREBUS. The Kalman machinery adds zero parameters — uncertainty is extracted from existing state.

---

## Training

### Single Phase (fixed depth, adaptive only at inference)

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Fixed 12 iterations (Poisson) | Learn Kalman dynamics + stable training |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-3 cosine -> 1e-4, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 32x1024, accum=2 (64K effective) |
| Precision | fp16 mixed + fp32 Kalman state |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 iterations |
| sigma_obs | 1.0 (fixed, not learned during training) |

**Training simplification:** Adaptive depth is inference-only. During training, all tokens get 12 iterations (Poisson-sampled). This avoids gradient routing complexity while still learning the Kalman uncertainty representation.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Kalman precision matrix J_t becomes ill-conditioned | MEDIUM | Add diagonal regularization: J_t = J_t + epsilon*I. Use fp32 for J_t (only 64x64 per head = 4K floats). |
| Uncertainty signal is not meaningful at 170M scale | MEDIUM | Kalman uncertainty is mathematically guaranteed to be meaningful (it's the Bayesian posterior). If signal is weak, increase sigma_obs to amplify uncertainty variation. |
| Information-form associative scan differs from FLA kernels | HIGH | Must implement custom Kalman scan. Can adapt FLA GLA kernel by augmenting state to carry (J, h) jointly. |
| Fixed training depth + adaptive inference = distribution shift | MEDIUM | At inference, tau is tuned to match training's average depth (~12). Tokens getting 2 iters saw 12 during training — may be over-processed but not under-processed. |
| Precision matrix inversion for uncertainty extraction | LOW | Only need diagonal of Sigma = J^{-1}. For diagonal J (which is approximately true), this is element-wise 1/J. Full inversion not needed. |

## Success Criteria

1. Val loss < 2.90 on BabyLM (**match AMADEUS**)
2. Throughput > 35K tok/s (train) / > 50K tok/s (inference with adaptive depth)
3. Uncertainty correlates with token difficulty (measured by per-token loss)
4. Adaptive depth: average inference iterations < 8 (vs 12 training) with < 0.5% quality drop
5. Kalman posterior is well-calibrated (uncertainty predicts actual error magnitude)

---

## Implementation Roadmap

1. Implement Kalman update in information form (J_t, h_t)
2. Implement chunk-wise associative scan for Kalman state
3. Extract uncertainty from diagonal of J_t^{-1}
4. Implement KalmanLinearAttention module
5. Implement AdaptiveDepthController
6. Assemble AegisBlock + AegisModel (shared looped)
7. Verify parameter count (~46M unique)
8. Smoke test: 10 min, verify loss decreasing + uncertainty varies across tokens
9. Register kalman_chunkwise as torch.library custom op
10. Full training: BabyLM 2 epochs (fixed depth)
11. Inference: tune tau for adaptive depth, measure quality vs compute tradeoff
12. Visualize: uncertainty heatmaps over text, correlation with token rarity

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), chunked_linear_cross_entropy

**External (1):** FLA GLA kernel (adapted for Kalman scan)

**New (1):** Kalman precision update — outer product k@k.T is element-wise on diagonal (fast). Full outer product needed for off-diagonal but chunk-wise amortizes this.

### Kalman Overhead Analysis

| Operation | Cost | On Strix Halo |
|-----------|------|---------------|
| Precision increment: k@k.T/sigma^2 | Outer product per token | Element-wise for diagonal = **free** |
| Information increment: k*v/sigma^2 | Element-wise multiply | **Free** |
| Decay application: decay * J | Element-wise multiply | **Free** |
| Uncertainty extraction: 1/diag(J) | Element-wise reciprocal | **Free** |
| Associative scan over (J, h) | 2x state vs standard | ~0.8ms (doubled GLA) |
| **Total Kalman overhead** | | **~0.8ms/layer/iter** |

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Eager fp16 | 12 iterations | ~18K tok/s |
| + autokernel | | ~25K tok/s |
| + compile | | ~34K tok/s |
| + FLA adapted kernel | | **~38K tok/s** |
| **Inference (adaptive, avg 7 iters)** | | **~58K tok/s** |

**Estimated training throughput:** ~34-38K tok/s
**Estimated inference throughput:** ~55-65K tok/s (adaptive depth saves ~40% compute)
**Tokens in 45 min:** ~92-103M (5.7-6.4 BabyLM epochs)
**Ranking:** #3-4 of all architectures
