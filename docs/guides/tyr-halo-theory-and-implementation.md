# TYR-HALO: Theory, Design, and PyTorch Implementation Guide

A comprehensive guide covering the theoretical foundations, architectural decisions, PyTorch implementation details, and optimization strategies behind TYR-HALO — a 58M-parameter small language model designed for maximum efficiency on limited hardware.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Architecture Walkthrough](#3-architecture-walkthrough)
4. [PyTorch Implementation](#4-pytorch-implementation)
5. [Training Optimizations](#5-training-optimizations)
6. [Inference Optimizations](#6-inference-optimizations)
7. [Kernel-Level Optimizations](#7-kernel-level-optimizations)
8. [Data Optimizations](#8-data-optimizations)
9. [Post-Training Pipeline](#9-post-training-pipeline)
10. [Reference Papers](#10-reference-papers)

---

## 1. Design Philosophy

### The Core Problem

Training competitive small LLMs requires solving a token-efficiency problem. SmolLM2-135M was trained on 2 trillion tokens. We have a budget of 12 billion tokens — 166x fewer. To compete, every mechanism must either improve quality per token or increase throughput.

TYR-HALO solves this through three principles:

1. **Throughput through simplicity.** Fewer layers (6 vs FENRIR's 10), shallower loop (mean=2 vs 3), smaller parameter count (58M vs 80M). Target: 60K+ tok/s.

2. **Quality through mechanisms.** Novel components (MoDA depth-attention, mHC Sinkhorn residuals, MTP auxiliary loss) each contribute 2-5% quality improvement with minimal FLOPs overhead.

3. **Quality through data.** Synthetic pretraining (Tufa Labs methodology) provides 3-6x token efficiency. Better data compensates for fewer tokens.

### Why Not Just Scale Down a Transformer?

A 58M standard transformer (e.g., 6-layer LLaMA) achieves ~25% HellaSwag. Our target is >30%. The gap comes from:

- **Depth starvation.** 6 layers is insufficient for complex language modeling. Standard transformers need 12+ layers for competitive quality.
- **Parameter waste.** Embedding tables consume 13M+ params (22% of budget) that don't contribute to transformation depth.
- **No information reuse.** Each layer processes independently — shallow layers can't benefit from deep layer representations.

TYR-HALO addresses all three:
- Parcae loop: 6 physical layers × mean 2 iterations = 12 effective layers
- Factorized embeddings: saves ~19M params vs standard embeddings
- MoDA depth-attention: each iteration accesses representations from all prior iterations

---

## 2. Theoretical Foundations

### 2.1 Parcae Looping (Weight Sharing with Stochastic Depth)

**Source:** "Parcae: Scalable Weight-Tying for Stable Deep Transformers" (2024)

The key insight is that running the same set of layers multiple times (with re-injection of the original input) approximates a much deeper network with far fewer parameters.

**Mathematical formulation:**

Given a shared block F with parameters θ, the Parcae loop computes:

```
h_0 = Prelude(embed(x))
h_1 = F_θ(h_0)                              # iteration 1
h_2 = F_θ(A * h_1 + B * h_0)               # iteration 2 (re-injection)
h_3 = F_θ(A * h_2 + B * h_0)               # iteration 3 (re-injection)
...
```

Where A and B are the Parcae injection parameters:
```
A = -exp(log_A)    → eigenvalues in (-1, 0), contractive
B = exp(log_B)     → positive, controls input re-injection strength
```

**Why this is stable:** The spectral constraint on A ensures that repeated application is contractive — the system cannot diverge. Each iteration refines the representation while the B*h_0 term re-anchors to the original input, preventing mode collapse.

**Stochastic depth via Poisson sampling:** During training, the number of loop iterations is sampled from a Poisson distribution:
```
n_iters ~ Poisson(mean_recurrence)
```

This regularizes the model, forcing it to produce useful representations at every iteration depth, not just the final one. The 1-sqrt curriculum gradually increases effective depth during training:
```
effective_progress = 1 - sqrt(1 - step/curriculum_steps)
```

**BPTT truncation:** Only the last `backprop_depth` iterations receive gradients. Earlier iterations are detached (`torch.no_grad()`). This bounds memory usage while allowing the model to learn from deep iteration chains.

**First iteration skip:** SimpleParcaeInjection outputs zero when h == input_embed (because A + B = 0 at that point). Therefore, injection is skipped on iteration 0 — only applied on re-entries.

### 2.2 MoDA Depth-Attention (Cross-Iteration Information Retrieval)

**Source:** "Mixture-of-Depths Attention" (Zhu et al., ByteDance, March 2026)

**The problem:** In deep networks, informative features from shallow layers get diluted through repeated residual updates. Standard residual connections compress all historical information into a single additive stream — later layers cannot selectively retrieve early-layer information.

In a looped architecture, this problem is amplified: iteration 2 cannot access specific features from iteration 0's computation unless they survived through the residual stream.

**The MoDA solution:** Each attention head attends to both:
1. **Sequence KVs** — standard within-layer context (with causal masking + RoPE)
2. **Depth KVs** — representations from the same token position in prior iterations (no RoPE, fully visible)

Mechanically, depth KVs are concatenated along the sequence dimension before SDPA:

```
K_full = cat([K_sequence, K_depth_iter0, K_depth_iter1, ...], dim=seq)
V_full = cat([V_sequence, V_depth_iter0, V_depth_iter1, ...], dim=seq)
```

The attention mask is constructed so that:
- Sequence portion: standard causal (lower triangular)
- Depth portion: fully visible (all ones)

This means every query token can attend to any depth KV from any prior iteration, but only to causally preceding sequence tokens.

**Depth KV production:** Each MoDA-GQA layer has an FFN-side projection:
```
depth_kv_proj = Linear(d_model, n_kv_heads * head_dim * 2)
```

After processing, the layer projects its hidden state into a (K, V) pair that future iterations can attend to. No RoPE is applied because depth KVs represent same-position cross-layer information, not different sequence positions.

**Why FFN-side (not attention-side):** The MoDA paper's ablation shows FFN-side KV projection gives the best accuracy/parameter ratio. The FFN captures post-attention refined features, which are more informative as depth keys.

**Cost:** ~3-5% FLOPs overhead (concatenating depth KVs increases attention computation). ~0.16M extra params per GQA layer. Expected benefit: +2% downstream improvement (MoDA paper: +2.11% avg across 10 benchmarks).

### 2.3 mHC (Manifold-Constrained Hyper-Connections)

**Source:** DeepSeek-V4 (April 2026)

**The problem:** Standard residual connections (`h = h + F(h)`) are additive — information can only be added, never selectively mixed or routed. SimpleParcaeInjection (`h = A*h + B*input`) is contractive but rank-1 per dimension.

**The mHC solution:** Expand the residual stream from 1 branch to N=4 branches. Each branch carries d-dimensional features. Between iterations, a learned Sinkhorn-constrained mixing matrix routes information across branches.

**Mathematical formulation:**

Given N=4 branches, the residual stream is a tensor `streams ∈ R^{B × N × T × d}`.

**Read (before each layer):** Compute a weighted sum of branches to produce single input:
```
x_bar = mean(streams, dim=1)                    # (B, T, d)
H_pre = sigmoid(0.01 * phi_pre(x_bar))          # (B, T, 4)
h_input = sum(streams * H_pre, dim=1)            # weighted sum → (B, T, d)
```

**Write (after each layer):** Mix existing branches and inject new output:
```
H_post = 2 * sigmoid(0.01 * phi_post(x_bar))    # (B, T, 4)
logits = 0.01 * phi_res(x_bar)                   # (B, T, 16) → reshape to (B, T, 4, 4)
H_res = sinkhorn(exp(logits), iters=20)           # (B, T, 4, 4) doubly stochastic
mixed = H_res @ streams + H_post * block_output   # mix + write
```

**Sinkhorn normalization:** The mixing matrix H_res is constrained to the Birkhoff polytope (doubly stochastic matrices) via Sinkhorn-Knopp iteration:
```
for _ in range(20):
    log_alpha = log_alpha - logsumexp(log_alpha, dim=-1)  # row normalize
    log_alpha = log_alpha - logsumexp(log_alpha, dim=-2)  # col normalize
H_res = exp(log_alpha)
```

This ensures:
- Each row sums to 1 (information is conserved, not amplified)
- Each column sums to 1 (no branch is starved or flooded)
- Spectral norm ≤ 1 (contractive, stable for looped architectures)

**Why 4x4?** DeepSeek-V4 uses n_hc=4. At 4x4, Sinkhorn is ~200 FLOPs per token — negligible even in pure PyTorch. A HIP kernel exists (28.5x speedup) but is unnecessary at this size.

**Initialization:** All phi projections initialized to zeros → near-identity at startup. The model starts as if mHC doesn't exist, then gradually learns to use the branches.

**0.01 scaling factor:** Prevents early-training explosions. The raw projection outputs are scaled by 0.01 before sigmoid/exp, keeping initial mixing matrices close to uniform.

### 2.4 Multi-Token Prediction (MTP)

**Source:** Meta "Better & Faster Large Language Models via Multi-token Prediction" (2024), DeepSeek-V4 (2026)

**The theory:** Standard next-token prediction only requires h_L to encode information about token t+1. MTP forces h_L to also encode information about token t+2 (and beyond), creating richer representations.

**Mathematical formulation:**

Given hidden state h at the final layer:
```
logits_main = LM_head(h)                    # predict t+1
logits_mtp  = MTP_head(h[:, :-2])           # predict t+2 from position t
```

The total loss combines both:
```
L = CE(logits_main, targets) + λ * CE(logits_mtp, targets[:, 2:])
```

where λ = 0.3 (DeepSeek V4 value, reduced to 0.1 during LR decay).

**Why this improves the backbone:** The gradient from the MTP loss flows back through the entire network, updating all layers to encode longer-range dependencies. The backbone learns to predict not just "what comes next" but "what comes next and after that."

**Implementation detail:** The MTP head shares the embedding table (tied weights) with the main head. This is done via factorized projection:
```
MTP_head: h → Linear(d, rank) → matmul(embed_table.T) → logits
```

This is identical to the main FactorizedLMHead architecture, ensuring the MTP head operates in the same representation space.

**Training only:** The MTP head is discarded after training. Its sole purpose is to improve backbone quality during pretraining.

### 2.5 Factorized Embeddings

**Source:** Nandi-150M (2025), SmolLM2

**The problem:** For vocab=50257 and d=640, a standard embedding table is 50257 × 640 = 32.2M parameters — 55% of a 58M budget. This is catastrophic for small models.

**The solution:** Factorize via low-rank decomposition:
```
Input:  Embedding(50257, rank=256) → Linear(256, 640)    # 13.03M params
Output: Linear(640, 256) → matmul(Embedding.weight.T)     # 0.16M params (tied)
```

Total: 13.19M vs 32.2M for standard. Saves 19M params (59% reduction) that are reallocated to transformation depth.

The rank=256 is chosen to be 128-aligned for GPU matrix multiplication efficiency (Tensile tile sizes on AMD, cuBLAS tile sizes on NVIDIA).

### 2.6 Exclusive Self Attention (XSA)

**Source:** "Exclusive Self Attention" (Zhai, 2026)

Standard self-attention computes:
```
y_i = sum_j(alpha_ij * v_j)
```

This includes the self-value projection: when i==j, the attention output contains v_i scaled by alpha_ii. This is redundant with the FFN layer, which also processes the current token's features.

XSA removes this redundancy:
```
z_i = y_i - proj(y_i, v_i)
    = y_i - (dot(y_i, v_i) / norm(v_i)^2) * v_i
```

This forces attention to capture information from OTHER tokens only, while the FFN handles the current token's transformation. Zero extra parameters. Measured -0.9% loss improvement in JORMUNGANDR-HALO experiments.

### 2.7 The Linear Probing Theorem and Draft Heads

**Theoretical foundation for speculative decoding from intermediate layers.**

A linear probe at layer L-k achieves approximately (1 - k/L)² of the final layer's next-token accuracy. For TYR-HALO with 14 effective layers (6 shared × 2 iterations + prelude + coda):

```
h_iter0 (after 6 layers): ~(1 - 6/14)² ≈ 33% accuracy loss → ~67% of final accuracy
```

But this is pessimistic — the formula assumes independent layers. With Parcae looping, iter0 sees a richer input (mHC branches + depth KVs from prelude), so empirical accuracy is higher: ~85-90% for easy tokens.

**Jacobian approximation:** The last k layers are locally approximately linear:
```
h_L ≈ J_k @ h_{L-k} + b_k
```

Therefore:
```
logits ≈ W_head @ J_k @ h_{L-k} + W_head @ b_k
       = W_draft @ h_{L-k} + bias_draft
```

A "draft head" is exactly W_draft = W_head @ J_k collapsed into a single linear projection. Each draft head learns this collapsed mapping during post-training distillation from the MTP head.

**DFlash validation (Chen et al. 2026):** Empirically confirmed that target-model intermediate features provide strong drafting signal:
- Without KV injection (our draft heads case): 2.83x speedup
- With KV injection: 5.15x speedup

### 2.8 DS2D Forecast Embeddings

**Source:** "Dynamic Self-Speculative Decoding" (Samsung, 2026)

Learned prefix embeddings (m=4) appended to the input tell the model to expect multi-token prediction. Trained via prefix-tuning on frozen backbone.

```
input_to_model = [token_embeds, forecast_embed_1, ..., forecast_embed_m]
```

The model learns that positions occupied by forecast embeddings should produce logits for future tokens rather than the standard next-token prediction. This is complementary to draft heads:

- **Forecast embeddings (static):** Prime the model's attention patterns for multi-token slots
- **Draft heads (dynamic):** Use actual h_iter0 to produce draft tokens

Together, forecast embeddings improve the quality of h_iter0 for draft head consumption.

### 2.9 Concurrent Token Generation (CTG)

**Source:** Samsung (2026)

Multiple independent generation streams from the same prefix, processed in a single batched forward pass. Each stream has its own KV-cache partition and sampling state.

**Why this matters for training:** ES alignment (EGGROLL) requires K=16 rollouts per prompt. Without CTG: 16 sequential generations. With CTG at batch=8: 2 batched passes. 8x speedup for the most expensive phase of post-training.

---

## 3. Architecture Walkthrough

### 3.1 Full Data Flow

```
Input: token IDs (B, T)
  ↓
FactorizedEmbedding: (B, T) → (B, T, 256) → (B, T, 640)
  ↓
Prelude: MoDA-GQA block (unshared, XSA)
  Input:  h (B, T, 640), velocity=zeros
  Output: h (B, T, 640), velocity (B, T, 640)
  ↓
Save input_embed = h
  ↓
mHC Init: h → streams (B, 4, T, 640) [clone h into 4 branches]
  ↓
╔══════════════════════════════════════════════════╗
║  PARCAE LOOP (sampled iterations, mean=2)        ║
║                                                  ║
║  For each iteration i:                           ║
║    1. mHC Read: streams → h (weighted sum)       ║
║    2. Re-inject (skip iter 0):                   ║
║       h = A * h + B * input_embed                ║
║    3. Run 6 shared layers:                       ║
║       L0: ShortConvBlock (conv + momentum + FFN) ║
║       L1: ShortConvBlock                         ║
║       L2: MoDA-GQA (attn + depth KVs + FFN)     ║
║       L3: ShortConvBlock                         ║
║       L4: ShortConvBlock                         ║
║       L5: MoDA-GQA (attn + depth KVs + FFN)     ║
║    4. iter_norm(h)                               ║
║    5. mHC Write: h → update streams              ║
║    6. Store depth KVs for future iterations      ║
╚══════════════════════════════════════════════════╝
  ↓
Coda: MoDA-GQA block (unshared, XSA)
  ↓
RMSNorm → FactorizedLMHead → logits (B, T, 50257)
  ↓ (training only)
MTP Head → mtp1_logits (B, T-2, 50257)
```

### 3.2 ShortConvBlock Internals

```
Input: x (B, T, 640), velocity (B, T, 640)
  ↓
pre_norm = RMSNorm(x)
  ↓
GatedConv:
  proj(pre_norm) → chunk into b, c, h_tilde (each B, T, 512)
  y = b * h_tilde
  z = causal_conv1d(y, kernel=3)
  conv_out = c * z                              # (B, T, 512)
  ↓
out_proj: Linear(512, 640)                       # back to d_model
  ↓
Momentum: velocity = sigmoid(log_beta) * velocity + out_proj
  ↓
Residual: x = x + velocity
  ↓
FFN norm + SwiGLU:
  normed = RMSNorm(x)
  gate, up = w_gate_up(normed).chunk(2)          # (B, T, 2304) → 2 × (B, T, 1152)
  x = x + w_down(silu(gate) * up)               # (B, T, 640)
  ↓
Output: x (B, T, 640), velocity (B, T, 640)
```

### 3.3 MoDA-GQA Block Internals

```
Input: x (B, T, 640), velocity (B, T, 640), freqs_cis, depth_kvs
  ↓
pre_norm = RMSNorm(x)
  ↓
Attention (CodaAttention with depth KV support):
  Q = wq(pre_norm) → (B, 10, T, 64)             # 10 heads
  K = wk(pre_norm) → (B, 2, T, 64)              # 2 KV heads
  V = wv(pre_norm) → (B, 2, T, 64)
  ↓
  apply_rotary_emb(Q, K, freqs_cis)              # RoPE
  QK-Norm: Q = normalize(Q) * q_scale
           K = normalize(K) * k_scale
  K, V = repeat_interleave(5)                    # 2 → 10 heads (GQA expansion)
  ↓
  if depth_kvs:                                   # MoDA: concat depth KVs
    for dk, dv in depth_kvs:
      K = cat([K, dk], dim=seq)                  # no RoPE on depth portion
      V = cat([V, dv], dim=seq)
    mask = [causal | fully_visible_depth]         # custom attention mask
    y = SDPA(Q, K, V, mask)
  else:
    y = SDPA(Q, K, V, is_causal=True)
  ↓
  if exclusive (XSA):
    y = y - proj(y, V_seq)                        # remove self-value
  ↓
  attn_out = wo(y)                                # (B, T, 640)
  ↓
Momentum: velocity = sigmoid(log_beta) * velocity + attn_out
Residual: x = x + velocity
FFN: x = x + SwiGLU(RMSNorm(x))
  ↓
Depth KV production:
  kv = depth_kv_proj(x)                          # Linear(640, 2*2*64) = (B, T, 256)
  k_depth, v_depth = chunk → (B, 2, T, 64) each
  ↓
Output: x, velocity, (k_depth, v_depth)
```

---

## 4. PyTorch Implementation

### 4.1 Key Implementation Files

| File | Purpose |
|------|---------|
| `models/tyr_halo.py` | Full model: TyrHaloBase + all variants + inference utilities |
| `models/jormungandr_halo.py` | CodaAttention with depth_kvs support |
| `models/argus_prime.py` | Attention (GQA), ShortConvBlock |
| `models/amadeus.py` | RMSNorm, SwiGLU, GatedConv |
| `models/chimera_halo.py` | FactorizedEmbedding, FactorizedLMHead |
| `models/griffin_halo.py` | SimpleParcaeInjection |
| `halo_training/mtp_loss.py` | MTP loss function |
| `halo_training/cli.py` | CLI with --mtp flag |

### 4.2 Implementing mHC from Scratch

The mHC Branch Manager is the most novel component. Here's the step-by-step PyTorch implementation with commentary:

```python
class mHCBranchManager(nn.Module):
    def __init__(self, d_model: int, n_branches: int = 4, sinkhorn_iters: int = 20):
        super().__init__()
        self.n_branches = n_branches
        self.sinkhorn_iters = sinkhorn_iters

        # Three small projections generate dynamic parameters per token
        # phi_pre: controls how branches are read (weighted sum)
        # phi_post: controls how new output is written back
        # phi_res: generates the mixing matrix (n_branches x n_branches)
        self.phi_pre = nn.Linear(d_model, n_branches, bias=True)
        self.phi_post = nn.Linear(d_model, n_branches, bias=True)
        self.phi_res = nn.Linear(d_model, n_branches * n_branches, bias=True)

        # Zero init = near-identity startup (model behaves as if mHC doesn't exist)
        for m in [self.phi_pre, self.phi_post, self.phi_res]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
```

**Why zero init matters:** DeepSeek-V4 uses zero-init with small gating factors so mHC starts as a no-op. The model first learns standard representations, then gradually discovers how to use the branch structure. Without this, early training is destabilized by random mixing.

```python
    def read(self, streams: torch.Tensor) -> torch.Tensor:
        # streams: (B, n_branches, T, d)
        x_bar = streams.mean(dim=1)  # mean across branches: (B, T, d)

        # 0.01 scaling prevents explosion: sigmoid(0.01 * 0) = 0.5 = uniform read
        H_pre = torch.sigmoid(0.01 * self.phi_pre(x_bar))  # (B, T, 4)

        # Weighted sum across branches. H_pre.unsqueeze(-1) broadcasts over d.
        # transpose to align branch dim: (B, 4, T, 1) * (B, 4, T, d) → sum → (B, T, d)
        return (streams * H_pre.unsqueeze(-1).transpose(1, 2)).sum(dim=1)
```

```python
    def write(self, streams: torch.Tensor, block_output: torch.Tensor) -> torch.Tensor:
        x_bar = streams.mean(dim=1)
        nb = self.n_branches

        # Write-in weights: how much of block_output goes to each branch
        # Factor of 2: ensures output can be amplified, not just attenuated
        H_post = 2.0 * torch.sigmoid(0.01 * self.phi_post(x_bar))  # (B, T, 4)

        # Mixing matrix: (B, T, 4, 4) doubly stochastic
        logits = 0.01 * self.phi_res(x_bar)
        logits = logits.view(*x_bar.shape[:-1], nb, nb)
        H_res = sinkhorn_normalize(logits, self.sinkhorn_iters)

        # Apply mixing: rotate branches, then add new output
        st = streams.permute(0, 2, 1, 3)  # (B, T, 4, d)
        mixed = torch.einsum('btij,btjd->btid', H_res, st)  # (B, T, 4, d)
        mixed = mixed + H_post.unsqueeze(-1) * block_output.unsqueeze(2)
        return mixed.permute(0, 2, 1, 3)  # back to (B, 4, T, d)
```

**Sinkhorn normalization:**

```python
def sinkhorn_normalize(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Convert log-space matrix to doubly stochastic via Sinkhorn-Knopp.

    For 4x4 matrices: ~200 FLOPs per token. Pure PyTorch is sufficient.
    A HIP fused kernel exists (28.5x speedup) but only matters for large matrices.
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()
```

**Why log-space:** Computing Sinkhorn in log-space (logsumexp instead of normalize-then-log) is numerically stable. The exp() at the end produces the final doubly stochastic matrix.

### 4.3 Implementing MoDA Depth-Attention

The key modification is in CodaAttention.forward(). After computing Q, K, V and applying RoPE, depth KVs from prior iterations are concatenated:

```python
def forward(self, x, freqs_cis, value_bias=None, depth_kvs=None):
    B, T, _ = x.shape
    q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
    k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
    v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

    # RoPE on sequence KVs only
    q, k = apply_rotary_emb(q, k, freqs_cis)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # QK-Norm (DeepSeek-V4 confirms this eliminates need for QK-Clip)
    if self.qk_norm:
        q = F.normalize(q, dim=-1) * self.q_scale
        k = F.normalize(k, dim=-1) * self.k_scale

    # GQA expansion: 2 KV heads → 10 query heads
    if self.n_rep > 1:
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

    # MoDA: concatenate depth KVs (no RoPE — position-independent)
    has_depth = depth_kvs is not None and len(depth_kvs) > 0
    if has_depth:
        for dk, dv in depth_kvs:
            # Expand depth KVs to match query head count
            if dk.shape[1] != k.shape[1]:
                dk = dk.repeat_interleave(k.shape[1] // dk.shape[1], dim=1)
                dv = dv.repeat_interleave(v.shape[1] // dv.shape[1], dim=1)
            k = torch.cat([k, dk], dim=2)  # concat along sequence dim
            v = torch.cat([v, dv], dim=2)

    # Attention with appropriate masking
    if has_depth:
        n_depth = k.shape[2] - T
        # Causal for sequence, fully visible for depth
        mask = torch.ones(T, T + n_depth, dtype=torch.bool, device=q.device)
        mask[:, :T] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # XSA: remove self-value projection (sequence V only, not depth V)
    if self.exclusive:
        v_seq = v[:, :, :T, :]
        dot = (y * v_seq).sum(dim=-1, keepdim=True)
        v_norm_sq = (v_seq * v_seq).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        y = y - (dot / v_norm_sq) * v_seq

    return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))
```

**Critical detail:** XSA projection uses only the sequence portion of V (v_seq = v[:, :, :T, :]), not the depth KVs. This prevents the XSA correction from removing depth-sourced information.

### 4.4 Implementing the MTP Loss

```python
def build_mtp_loss_fn(mtp_weight: float = 0.3):
    def mtp_loss_fn(output, batch):
        _, targets = batch

        if isinstance(output, dict):
            logits = output["logits"]
            V = logits.shape[-1]
            loss_main = F.cross_entropy(logits.view(-1, V), targets.view(-1))

            total = loss_main
            if "mtp1" in output:
                mtp1 = output["mtp1"]
                # MTP predicts t+2: targets shifted by 2
                mtp_targets = targets[:, 2:].reshape(-1)
                loss_mtp = F.cross_entropy(mtp1.reshape(-1, V), mtp_targets)
                total = total + mtp_weight * loss_mtp
            return total
        else:
            # Fallback for non-MTP models
            V = output.shape[-1]
            return F.cross_entropy(output.view(-1, V), targets.view(-1))

    return mtp_loss_fn
```

**Why targets[:, 2:]:** The MTP head at position t predicts token at position t+2. Since `targets` is already shifted by 1 from `input_ids` (standard LM setup), MTP targets need an additional shift of 1, totaling shift=2 from the original `targets` tensor.

### 4.5 Implementing Draft Heads

```python
class DraftHeads(nn.Module):
    def __init__(self, d_model, embed_rank, embed_table, n_drafts=4):
        super().__init__()
        self.n_drafts = n_drafts
        # Each head: independent Linear probe on intermediate hidden state
        self.probes = nn.ModuleList([
            nn.Linear(d_model, embed_rank, bias=False) for _ in range(n_drafts)
        ])
        self.embed_table = embed_table  # shared with main LM head

    def forward(self, h_iter0):
        """Parallel draft logits from h_iter0's last position."""
        last_h = h_iter0[:, -1:, :]  # only need last token for decode
        return [F.linear(probe(last_h), self.embed_table.weight)
                for probe in self.probes]

    def draft_tokens(self, h_iter0):
        """Greedy-decode K draft tokens in parallel."""
        logits_list = self.forward(h_iter0)
        return torch.cat([lg.argmax(dim=-1) for lg in logits_list], dim=-1)
```

**Why shared embed_table:** Draft heads use the same embedding projection as the main LM head. This means they operate in the same token-space, making their predictions directly comparable during verification.

### 4.6 Speculative Decode Loop

The `speculative_decode()` method on TyrHaloBase implements Parcae-native speculation:

```
1. Run iter0 → h_iter0
2. Draft K=4 tokens from draft_heads(h_iter0)
3. Append draft tokens to input sequence
4. Run full model (iter0 + iter1) on extended sequence
5. Verify: compare model's logits at each draft position
6. Accept longest correct prefix
7. Sample one token from verified position
```

This reuses the natural loop structure — iter0 is already computed for drafting, iter1 provides the verification signal. No wasted compute.

---

## 5. Training Optimizations

### 5.1 Muon Optimizer

TYR-HALO uses Muon for 2D weight matrices (Linear layers) and AdamW for everything else. Muon applies Newton-Schulz orthogonalization (10 iterations) to the gradient, producing updates that respect the weight manifold geometry.

**Parameter groups:**

| Group | Optimizer | LR | Rationale |
|-------|-----------|-----|-----------|
| Core block 2D weights | Muon | 0.015 | 0.75x base — loop accumulates gradients across iterations |
| Prelude/Coda 2D weights | Muon | 0.02 | Standard — single-pass layers |
| mHC projections | AdamW | 8e-4 | Small modules, need Adam's adaptive rates |
| MTP head | AdamW | 8e-4 | Auxiliary head, separate learning dynamics |
| Depth KV projections | AdamW | 8e-4 | Small projections, AdamW stability |
| Norms, biases, embeddings | AdamW | 8e-4 | Standard — 1D params always use AdamW |

**Why lower LR for core block:** The Parcae loop causes the same parameters to receive gradients from multiple iterations. With BPTT=2, each core weight gets gradient contributions from 2 loop iterations. A lower LR compensates for this implicit gradient amplification.

### 5.2 Per-Zone Compilation

`torch.compile` cannot trace through Python loops with dynamic trip counts (Poisson-sampled depth). Solution: compile each layer independently, leave the loop as Python:

```python
def compile_zones(self):
    self.prelude = torch.compile(self.prelude, mode="default")
    for i in range(len(self.shared_layers)):
        self.shared_layers[i] = torch.compile(self.shared_layers[i], mode="default")
    self.coda = torch.compile(self.coda, mode="default")
```

Each compiled zone becomes a fused kernel graph. Inductor fuses:
- ShortConvBlock: conv → momentum (element-wise) → residual → RMSNorm → SwiGLU into 1-2 regions
- MoDA-GQA: QKV → RoPE → SDPA → output → momentum → norm → SwiGLU into 1-2 regions

Expected: ~2.5-3x compile boost per layer (JORMUNGANDR-HALO measured 3.07x).

### 5.3 Precision Strategy

**NVIDIA (RTX 4060 Ti):** bf16 natively. No GradScaler needed. Ada Lovelace tensor cores handle bf16 at full speed.

**AMD (Strix Halo gfx1151):** fp16 + GradScaler. bf16 is 24% slower on RDNA 3.5 and crashes compile with RoPE complex buffers.

The model auto-detects platform:
```python
is_amd = hasattr(torch.version, 'hip') and torch.version.hip is not None
if dtype == torch.bfloat16 and is_amd:
    raise RuntimeError("bf16 is 24% slower on gfx1151. Use fp16.")
```

### 5.4 Gradient Accumulation for Effective Batch Size

At batch=32, ctx=1024, accum_steps=8: effective batch = 262K tokens/step.

The accum_steps=8 is chosen so that on 2-machine DDP (Strix Halo), the allreduce communication overlaps with compute. At accum_steps=8, sync overhead is ~3-6%.

### 5.5 Depth-Scaled Weight Initialization

Output projections (wo, w_down, out_proj) are scaled by 1/sqrt(2*n_layers):

```python
if "wo." in name or "w_down." in name or "out_proj." in name:
    p.div_(math.sqrt(2 * n_layers))
```

This prevents signal amplification in deep residual networks. With 6 layers × 2 iterations = 12 effective layers, each residual contribution is scaled by ~1/4.9, keeping the total signal magnitude stable.

---

## 6. Inference Optimizations

### 6.1 Phase 0: Draft Heads + DS2D (Single GPU, Free)

**Setup:**
```python
model = TyrHalo(use_draft_heads=True, use_forecast=True)
# Load trained weights...
model.eval()
output = model.speculative_decode(input_ids, max_new_tokens=256)
```

**Expected speedup:** ~2.5-3x decode over autoregressive baseline.

**How it works internally:**
1. Forward iter0 → h_iter0 (6 layers, ~2.5ms)
2. draft_heads(h_iter0) → 4 draft tokens (parallel, ~0.05ms)
3. Forward iter0+iter1 on [input + drafts] → verify_logits (~5ms)
4. Accept 2-3 correct drafts on average
5. Total: ~7.5ms for 3-4 tokens vs ~10ms autoregressive (2 × 5ms)

### 6.2 Phase 1: DFlash Drafter on Machine B (2 machines)

**Setup:** Target on Machine A (192.168.1.140), DFlash drafter on Machine B (192.168.1.145). Thunderbolt 4 interconnect.

**Speedup:** ~4-5x. DFlash generates 16 tokens in parallel with ~85% acceptance.

### 6.3 Phase 2: DFlash + SSD Caching (2 machines)

**Setup:** Same as Phase 1, plus Saguaro outcome caching on Machine B.

**Speedup:** ~6-8x. Cache hit ~90% eliminates drafting latency for most cycles.

### 6.4 CTG for Batched Generation

For ES alignment or any scenario requiring multiple completions from the same prompt:

```python
from models.tyr_halo import concurrent_generate

completions = concurrent_generate(
    model, prompt_ids,
    n_streams=16,          # 16 parallel rollouts
    max_new_tokens=128,
    temperature=1.0,
)
# completions: (16, T + 128)
```

Speedup: ~8x over sequential generation (16 streams in 2 batched passes at batch=8).

---

## 7. Kernel-Level Optimizations

### 7.1 AMD Strix Halo (gfx1151) — HIP Custom Ops

When running on AMD, autokernel provides fused HIP kernels registered as `torch.library.custom_op`:

| Custom Op | Speedup | What It Fuses |
|-----------|---------|---------------|
| `autokernel::fused_res_rmsnorm` | 6.6x | residual add + RMSNorm into single pass |
| `autokernel::rmsnorm` | 3.3x | standalone RMSNorm |
| `autokernel::silu_gate_mul` | 1.6x | SiLU(gate) * up (SwiGLU activation) |
| `autokernel::rotary_emb_fp32` | 3.7x | RoPE embedding |
| `autokernel::mhc_sinkhorn` | 28.5x | Full mHC computation (projections + Sinkhorn) |

**FusedResidualRMSNorm for ShortConvBlock:** The momentum pattern `x = x + velocity` followed by RMSNorm can use the existing fused kernel:
```python
hidden, normed = torch.ops.autokernel.fused_res_rmsnorm(velocity, x, ffn_norm.weight)
# hidden = velocity + x = updated x
# normed = rmsnorm(hidden)
```

This saves one memory pass per ShortConvBlock (4 blocks × 2 iterations = 8 fusions per forward pass).

**Pattern matching:** autokernel's pattern detector automatically matches:
- FusedQKVPattern on all 4 Attention modules (prelude + 2 core GQA + coda)
- RMSNormPattern on all standalone RMSNorm instances
- SiluGateMulPattern / FusedSwiGLUPattern on all FFN modules

### 7.2 NVIDIA — torch.compile + CUDA Graphs

On NVIDIA, skip autokernel entirely. Use torch.compile:

```python
# Full model (preferred, if Poisson sampling doesn't break tracing)
model = torch.compile(model, mode="reduce-overhead")

# Fallback: per-zone (same as AMD approach)
model.compile_zones()
```

`mode="reduce-overhead"` enables CUDA graphs, which capture the entire kernel launch sequence and replay it. This eliminates Python overhead and CPU-GPU synchronization. Effective when the same computation graph is repeated (which it is for each Parcae iteration).

### 7.3 Memory-Bandwidth Optimization Strategy

TYR-HALO is memory-bandwidth-bound (small model, 58M params). The optimization hierarchy:

1. **Kernel fusion** (eliminate intermediate tensors) — highest impact (6-10x per fused op)
2. **Vectorized loads** (half2 on AMD, bf16 tensor cores on NVIDIA) — 2x throughput
3. **L2 cache utilization** — core block ~4.8MB fits in 6MB L2 on Strix Halo → iteration 2 is L2-hot
4. **Compile graph fusion** — Inductor fuses element-wise chains across layers (2.5-3x)
5. **Avoid matmul custom kernels** — rocBLAS/cuBLAS are already optimal for GEMMs

---

## 8. Data Optimizations

### 8.1 Synthetic Pretraining (Tufa Labs, 2026)

The biggest quality lever. Generate augmented training data offline using a same-size generator model.

**Three augmentation modes:**

| Mode | Prompt | Token Expansion | Best For |
|------|--------|----------------|----------|
| **TPT** | "Simulate an expert's in-depth thought process" | 3.13x | Math, reasoning |
| **First Principles** | "Break down to fundamental concepts" | 3.53x | Science, definitions |
| **Rephrasing** | "Elaborate, clarify key steps, reveal complexities" | 1.75x | Code, general |

**Generator:** Qwen-3.5-0.8B (same-size works per Tufa Labs finding). Temperature=1, top_p=0.95, top_k=20, max 3072 tokens per augmentation.

**Key finding:** 3-6x token efficiency. Models trained on synthetic data match originals using 3-6x fewer training tokens. On 12B synthetic tokens, this is equivalent to 36-72B standard tokens.

### 8.2 CLIMB Data Mixture Pipeline

Existing pipeline (scripts/datamix/) extended for TYR-HALO:

1. **Phase 0-2:** Sample, embed (MiniLM), cluster (FAISS K=16)
2. **Phase 3:** Proxy search with TyrHaloMini
3. **Phase 3.5 (new):** Synthetic augmentation of top clusters
4. **Phase 4:** API quality scoring
5. **Phase 5:** Assemble pre-mixed .bin

Target mix: 40% math (TPT), 20% science (First Principles), 15% code (Rephrasing), 25% general (raw).

### 8.3 Pre-tokenized Binary Format

Training data stored as pre-tokenized uint16 .bin files. BabyLMDataset reads these directly — no tokenization at training time. EOS token (50256) inserted between documents during preprocessing.

---

## 9. Post-Training Pipeline

### 9.1 EGGROLL ES Alignment

Replace gradient-based alignment (DPO/SimPO) with Evolution Strategies:

1. Sample rank-r perturbation: ε = σ(ε₂ε₁ᵀ), rank=4
2. Generate K=16 rollouts (via CTG: 2 batched passes, 8x faster)
3. Score with external judge (Llama-3-8B-Instruct)
4. ES gradient estimate from fitness-weighted perturbations
5. Update model parameters (no backprop)

**Why ES over DPO:** No reference model needed (saves 50% memory). Works on non-differentiable objectives (tool-use success, judge scores). Population noise-reuse amortizes rollout cost.

### 9.2 Draft Head Training

After base model training, train draft heads via distillation:

1. Run base model on held-out data, cache (h_iter0, correct_next_token) pairs
2. Train each draft head to predict correct_next_token from h_iter0
3. Loss: cross-entropy per head, independent training
4. ~1000 steps sufficient (small linear probes converge fast)

### 9.3 Forecast Embedding Training

DS2D forecast embeddings trained via prefix-tuning:

1. Freeze entire backbone
2. Only optimize the m=4 forecast embedding parameters
3. Modified causal mask: forecast positions cannot attend to prompt (prevents information leakage)
4. Loss: next-token prediction at forecast positions
5. ~5000 steps (more optimization needed since embeddings must learn without backbone adaptation)

---

## 10. Reference Papers

| # | Paper | Year | Key Contribution to TYR-HALO |
|---|-------|------|------------------------------|
| 1 | DeepSeek-V4 | 2026 | mHC Sinkhorn residuals, MTP w/ 0.3 loss weight, Muon validation at 1.6T params |
| 2 | MoDA | 2026 | Depth-attention via KV concatenation, +2.11% at 3.7% FLOPs, fused kernel at 97.3% FA2 speed |
| 3 | EGGROLL | 2025 | Rank-r ES for alignment, int8-only training feasibility, 91% inference throughput |
| 4 | Self-Improving Pretraining | 2026 | Online DPO during pretraining, 3.2x reasoning (deferred — too expensive) |
| 5 | PTP | 2025 | Auxiliary variable parallel decode, 2.4x speedup (superseded by DFlash) |
| 6 | EBT | 2025 | Energy-based heads, 35% better scaling rate (deferred — Hessian cost) |
| 7 | Tufa Labs | 2026 | Synthetic pretraining: 3-6x token efficiency, same-size generator |
| 8 | DFlash | 2026 | Block diffusion drafter, 4.9x speedup, validates draft-from-middle theory |
| 9 | SSD/Saguaro | 2026 | Async speculation caching, +30% on any drafter, 90% cache hit |
| 10 | Samsung DS2D+CTG | 2026 | Forecast embeddings for self-speculation, CTG for batched generation |
| 11 | Learning Mechanics | 2026 | Theoretical framework for DL theory (no direct architecture impact) |
| 12 | Parcae | 2024 | Stable looping via spectral constraint, Poisson depth, BPTT truncation |
| 13 | Meta MTP | 2024 | Multi-token prediction improves backbone representations |
| 14 | XSA | 2026 | Exclusive Self Attention removes self-value redundancy, -0.9% loss |
| 15 | Nandi-150M | 2025 | Factorized embeddings save 59% embedding parameters |
| 16 | LFM2 | 2025 | 75:25 conv:attention ratio optimal for hybrid models |

---

## Appendix: Complete Model Instantiation

```python
from models.tyr_halo import TyrHalo

# Default: all mechanisms enabled
model = TyrHalo()
# TyrHalo: 58.6M parameters

# With draft heads and forecast embeddings for inference
model = TyrHalo(use_draft_heads=True, use_forecast=True)

# Training
model.train()
output = model(input_ids)
# output = {"logits": (B, T, 50257), "mtp1": (B, T-2, 50257)}

# MTP loss
from halo_training.mtp_loss import build_mtp_loss_fn
loss_fn = build_mtp_loss_fn(mtp_weight=0.3)
loss = loss_fn(output, (input_ids, targets))
loss.backward()

# Inference with speculative decoding
model.eval()
model.use_draft_heads = True
model.draft_heads = DraftHeads(640, 256, model.tok_embeddings.embed, n_drafts=4)
output_ids = model.speculative_decode(prompt_ids, max_new_tokens=256)

# Concurrent generation for ES alignment
from models.tyr_halo import concurrent_generate
rollouts = concurrent_generate(model, prompt_ids, n_streams=16, max_new_tokens=128)
```
