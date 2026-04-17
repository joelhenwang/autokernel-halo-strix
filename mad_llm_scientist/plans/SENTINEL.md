---
title: "SENTINEL"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/JORMUNGANDR.md
  - mad_llm_scientist/plans/EREBUS.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %sentinel, %ttt, %looped, %in-place-ttt, %surprisal]
---

# SENTINEL

**In-Place TTT on Shared Looped Blocks — The Watchful Guardian That Learns While It Looks**

*"The sentinel does not merely observe — it adapts its vigil to what it sees. Each pass through the watchtower reveals new threats, and the guardian reshapes its gaze accordingly."*
*JORMUNGANDR proved looping at 43K tok/s. In-Place TTT proves adaptation needs zero extra parameters.*

## Hypothesis

In-Place TTT (ByteDance, April 2026) uses the MLP's final projection matrix as "fast weights" that adapt per-token during the forward pass — requiring zero extra parameters. In a looped architecture (JORMUNGANDR-HALO, 43K tok/s), TTT makes each loop iteration see **effectively different weights** despite sharing the same physical block. The TTT update is a rank-1 outer product (element-wise on Strix Halo = free). SR-TTT's surprisal gate (Feb 2026) filters ~80% of easy tokens, so only hard tokens pay for adaptation. The net effect: shared-block quality approaching multi-block architectures, at shared-block throughput.

**Key papers:** "In-Place Test-Time Training" (2604.06169, April 2026), "SR-TTT: Surprisal-Aware Residual TTT" (2603.06642, Feb 2026)

---

## Architecture

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  -> 1 SHARED BLOCK x N iterations (N=12 train, Poisson):
  |     RMSNorm
  |     +---------------------------------------------+
  |     | EFLA Token Mixer                            |
  |     |   ShortConv(k=3) on Q/K/V                  |
  |     |   Swish activation + L2Norm on Q/K          |
  |     |   EFLA exact delta rule (chunk=64)           |
  |     +---------------------------------------------+
  |     +Residual
  |     RMSNorm
  |     +---------------------------------------------+
  |     | SwiGLU FFN with In-Place TTT:               |
  |     |   gate, up = Linear(x, 768->2x1920).chunk(2)|
  |     |   h = silu(gate) * up                       |
  |     |   y = (W_down + Delta_t) @ h                |
  |     |                                              |
  |     |   Delta_t computation (surprisal-gated):     |
  |     |     s_t = -log P(y_{t-1} | context)          |
  |     |     if s_t > tau:                            |
  |     |       error = y_target - y_pred              |
  |     |       Delta_t = -eta * outer(error, h)       |
  |     |     else:                                    |
  |     |       Delta_t = 0  (skip, 80% of tokens)    |
  |     +---------------------------------------------+
  |     +Residual
  |
  |     * Stochastic depth: Poisson(lambda=12), clamp [8,16] *
  |     * Delta_t reset to zero at start of each sequence *
  |
  -> Final RMSNorm -> LM Head
```

### Why In-Place TTT on Looped Blocks

Standard looped models (JORMUNGANDR, RESONANT-LOOP, EREBUS) reuse identical weights across iterations. The token mixer evolves hidden state across iterations, but the FFN transformation is literally the same function applied repeatedly. In-Place TTT changes this: the FFN's output projection W_down is perturbed by a rank-1 update Delta_t that depends on the current token's difficulty.

In a 12-iteration loop:
```
Iteration 1:  W_down + Delta_1  (first refinement)
Iteration 2:  W_down + Delta_2  (Delta_2 computed from iteration 1's output)
...
Iteration 12: W_down + Delta_12 (final refinement)
```

Each iteration effectively sees DIFFERENT weights — the shared block becomes 12 distinct blocks at zero parameter cost. The TTT signal provides an additional NTP-aligned training gradient at every iteration (not just the final output), creating a richer optimization landscape.

### Surprisal Gating (SR-TTT)

Not all tokens need adaptation. "The" always predicts well; a math proof's conclusion doesn't. SR-TTT gates the TTT update on surprisal:

```python
s_t = cross_entropy(predicted_t, target_t)  # surprisal
if s_t > tau:  # ~20% of tokens
    Delta_t = -eta * torch.outer(error, h)   # rank-1, element-wise
else:
    Delta_t = 0  # skip, no overhead
```

The 80% skip rate means the TTT overhead is amortized to <5% of total compute.

---

## Component 1: EFLA Token Mixer (from EREBUS)

```python
class EFLAMixer(nn.Module):
    def __init__(self, d_model=768, n_heads=12, head_dim=64, conv_kernel=3):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.w_qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.short_conv = nn.Conv1d(
            n_heads * head_dim, n_heads * head_dim,
            kernel_size=conv_kernel, groups=n_heads * head_dim,
            padding=conv_kernel - 1
        )
        self.w_beta = nn.Linear(d_model, n_heads, bias=True)
        self.w_gate = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.head_norm = nn.GroupNorm(n_heads, n_heads * head_dim)

    def forward(self, x, state=None):
        B, T, D = x.shape
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = F.silu(self.short_conv(q.transpose(1, 2))[:, :, :T].transpose(1, 2))
        k = F.silu(self.short_conv(k.transpose(1, 2))[:, :, :T].transpose(1, 2))
        v = F.silu(self.short_conv(v.transpose(1, 2))[:, :, :T].transpose(1, 2))
        q = F.normalize(q.view(B, T, self.n_heads, self.head_dim), dim=-1)
        k = F.normalize(k.view(B, T, self.n_heads, self.head_dim), dim=-1)
        v = v.view(B, T, self.n_heads, self.head_dim)
        beta = torch.sigmoid(self.w_beta(x)).unsqueeze(-1)
        lambda_t = (k * k).sum(dim=-1, keepdim=True)
        alpha_t = (1 - torch.exp(-beta * lambda_t)) / (lambda_t + 1e-6)
        o = efla_chunkwise(q, k, v, alpha_t, chunk_size=64, state=state)
        gate = torch.sigmoid(self.w_gate(x)).view(B, T, self.n_heads, self.head_dim)
        o = gate * self.head_norm(
            o.flatten(-2, -1).transpose(1, 2)
        ).transpose(1, 2).view(B, T, self.n_heads, self.head_dim)
        return self.w_o(o.flatten(-2, -1))
```

## Component 2: SwiGLU FFN with In-Place TTT

```python
class InPlaceTTTFFN(nn.Module):
    def __init__(self, d_model=768, ffn_inner=1920, eta=0.01, tau=2.0):
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)
        self.eta = eta
        self.tau = tau

    def forward(self, x, surprisal=None):
        B, T, D = x.shape
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up  # (B, T, ffn_inner)

        y = F.linear(h, self.w_down.weight)  # base output

        if self.training and surprisal is not None:
            mask = (surprisal > self.tau).unsqueeze(-1).float()  # (B, T, 1)
            error = x - y  # NTP-aligned reconstruction target
            # rank-1 update: Delta = -eta * outer(error, h), applied per-token
            # Efficient: Delta @ h = -eta * error * (h @ h) for each token
            h_norm_sq = (h * h).sum(dim=-1, keepdim=True)  # scalar
            correction = -self.eta * error * h_norm_sq  # (B, T, D)
            y = y + mask * correction

        return y
```

**Training dynamics:** The TTT signal (correction term) provides gradient to W_down beyond the standard NTP loss. This is equivalent to an inner-loop optimization step, making the model a meta-learner. The surprisal mask ensures only informative tokens contribute to adaptation.

## Component 3: Sentinel Looped Model

```python
class SentinelModel(nn.Module):
    def __init__(self, d_model=768, n_iterations=12):
        self.embedding = nn.Embedding(50257, d_model)
        self.shared_block = SentinelBlock(d_model)
        self.n_iterations = n_iterations
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, 50257, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, targets=None):
        h = self.embedding(input_ids)

        n_iter = self.n_iterations
        if self.training:
            n_iter = min(max(
                torch.poisson(torch.tensor(float(self.n_iterations))).int().item(),
                8), 16)

        surprisal = None
        for i in range(n_iter):
            if self.training and i > 0:
                with torch.no_grad():
                    probe_logits = self.lm_head(self.final_norm(h))
                    if targets is not None:
                        surprisal = F.cross_entropy(
                            probe_logits.view(-1, 50257),
                            targets.view(-1), reduction='none'
                        ).view(h.shape[0], h.shape[1])
            h = self.shared_block(h, surprisal=surprisal, iteration=i)

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
| n_iterations | 12 (train: Poisson [8,16], eval: fixed 12) |
| shared_blocks | 1 |
| conv_kernel | 3 |
| chunk_size | 64 |
| ttt_eta | 0.01 (inner learning rate) |
| ttt_tau | 2.0 (surprisal threshold) |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257x768, tied) | 38.6M |
| **Shared block:** | |
|   EFLA mixer: w_qkv (768->2304) | 1.77M |
|   EFLA mixer: w_o (768->768) | 0.59M |
|   EFLA mixer: ShortConv (k=3, 768ch) | 2.3K |
|   EFLA mixer: w_beta (768->12) | 9.2K |
|   EFLA mixer: w_gate (768->768) | 0.59M |
|   EFLA mixer: head_norm | 768 |
|   SwiGLU FFN: w_gate_up (768->3840) | 2.95M |
|   SwiGLU FFN: w_down (1920->768) | 1.47M |
|   RMSNorm x2 | 1.5K |
|   **Block total** | **~7.38M** |
| **x1 block (shared across 12 iterations)** | **7.38M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL (unique)** | **~46.0M** |
| **Effective params (12 iterations)** | **~127M effective** |

Core block is ~3.7MB in fp16. Fits comfortably in 6MB L2 cache. TTT adds ZERO parameters — Delta_t is computed on-the-fly and never stored as a weight.

---

## Training

### Single Phase

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Full model, TTT active from step 1000 | Learn language + per-token adaptation |

### TTT Warmup

TTT is disabled for the first 1000 steps (surprisal threshold set to infinity). This lets the base model stabilize before adaptation kicks in. After warmup, tau anneals from 4.0 -> 2.0 over 2000 steps.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-3 cosine -> 1e-4, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 32x1024, accum=2 (64K effective) |
| Precision | fp16 mixed + fp32 EFLA state |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 iterations |
| Stochastic depth | Poisson(lambda=12), clamped [8, 16] |
| TTT eta | 0.01 (fixed, not learned) |
| TTT tau warmup | inf -> 4.0 -> 2.0 (first 3000 steps) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| TTT inner gradient destabilizes training | MEDIUM | Surprisal gate + small eta (0.01) + warmup. TTT paper shows stability at 4B scale. |
| Surprisal probe adds overhead per iteration | MEDIUM | Probe uses existing LM head (no extra params). Skip probe on iterations 1 (use previous surprisal). |
| Rank-1 Delta_t too weak for meaningful adaptation | LOW | In-Place TTT paper shows rank-1 suffices at 4B. At 170M, each outer product covers more of the weight space. |
| TTT correction conflicts with standard NTP gradient | LOW | TTT objective IS NTP — they're aligned by construction. No conflicting gradients. |
| Stochastic depth + variable TTT = training instability | LOW | EREBUS validated stochastic depth. TTT is additive (Delta=0 is valid). |

## Success Criteria

1. Val loss < 2.88 on BabyLM (**beat AMADEUS 2.90**)
2. Throughput > 35K tok/s (compile + autokernel + FLA)
3. TTT ablation: TTT-on vs TTT-off > 1.5% loss improvement
4. Surprisal gate filters > 70% of tokens (confirming efficiency)
5. Quality improves with iteration count (ablation: 4, 8, 12, 16)
6. No NaN/Inf across all iterations

---

## Implementation Roadmap

1. Start from EREBUS implementation (EFLA looped block)
2. Add In-Place TTT to SwiGLU FFN's W_down
3. Implement surprisal computation (probe + cross_entropy)
4. Implement surprisal gating with warmup schedule
5. Implement SentinelModel with Parcae-style loop
6. Verify parameter count (~46M unique, ~127M effective)
7. Smoke test: 10 min, confirm > 30K tok/s + loss decreasing
8. Register EFLA chunk-wise as torch.library custom op
9. Full training: BabyLM 2 epochs
10. Ablation cascade: base loop -> +TTT -> +surprisal gate -> +stochastic depth
11. Compare vs EREBUS (same throughput class) and AMADEUS (quality target)
12. Visualize: which tokens trigger TTT? Do they correlate with loss reduction?

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), chunked_linear_cross_entropy

**External (2):** causal-conv1d (10x conv speedup), FLA DeltaNet kernel (chunk-wise EFLA)

**New (0):** TTT outer product is `torch.outer(error, h)` — standard PyTorch element-wise. Surprisal is `F.cross_entropy` — already have kernel.

### TTT Compute Cost Analysis

| Operation | Per-token cost | Frequency | Net cost |
|-----------|---------------|-----------|----------|
| Surprisal probe (LM head forward) | ~0.5ms | Every iteration after 1st | 0.5ms x 11 = 5.5ms |
| Cross-entropy for surprisal | ~0.01ms | Same | negligible |
| Outer product (error x h) | ~0.001ms | 20% of tokens | negligible |
| Mask application | ~0.001ms | All tokens | negligible |
| **Total TTT overhead** | | | **~5.5ms/sequence** |

The surprisal probe dominates. Optimization: cache probe logits from previous iteration (stale by 1 iteration but still useful). This reduces probes from 11 to 1.

**With cached surprisal:** TTT overhead drops to ~0.5ms/sequence (**<2% overhead**).

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Eager fp16 (no TTT) | | ~20K tok/s |
| + autokernel | | ~28K tok/s |
| + compile | | ~38K tok/s |
| + TTT (cached surprisal) | | ~37K tok/s |
| + causal-conv1d + FLA | | **~41K tok/s** |

**Estimated optimized throughput:** ~38-41K tok/s (compile + autokernel + TTT + external)
**Tokens in 45 min:** ~103-111M (6.4-6.9 BabyLM epochs)
**Ranking:** #2-3 of all architectures (near JORMUNGANDR, better quality)
