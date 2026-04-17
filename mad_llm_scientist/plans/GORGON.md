---
title: "GORGON"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/VALKYRIE.md
  - mad_llm_scientist/plans/EREBUS.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %gorgon, %deltaproduct, %householder, %efla, %quality]
---

# GORGON

**DeltaProduct Householder Cascade with EFLA — The Petrifying Gaze That Sees Through Stone**

*"Where DeltaNet sees with one eye (rank-1), the Gorgon opens many — each Householder reflection a new perspective that pierces through the veil of information compression."*
*DeltaProduct proved multi-step updates beat single-step. EFLA proves each step can be exact.*

## Hypothesis

Standard DeltaNet takes ONE gradient descent step per token (rank-1 state transition). DeltaProduct (Feb 2025) takes **n_h steps** using products of generalized Householder reflections, yielding diagonal + rank-n_h transitions. Each reflection H_i = I - 2v_iv_i^T/(v_i^Tv_i) is an outer product — entirely element-wise on Strix Halo. The paper demonstrates dramatic improvements on state tracking and length extrapolation. EFLA's exact ODE solution applies to EACH Householder step (exploiting the same rank-1 structure of v_i*v_i^T), eliminating discretization error accumulation across the n_h sub-steps. This is the **strongest quality hypothesis**: rank-4 state transitions approach attention-level expressivity at O(1) per-token cost.

**Key papers:** "DeltaProduct: Improving State-Tracking via Householder Products" (2502.10297, Feb 2025), "EFLA" (2512.12602)

---

## Architecture

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  -> 16 DeltaProduct Blocks:
  |     RMSNorm
  |     +---------------------------------------------+
  |     | DeltaProduct Token Mixer                     |
  |     |   ShortConv(k=3) on Q/K/V + Swish + L2Norm  |
  |     |   For j in 1..n_h (n_h=4 Householder steps): |
  |     |     v_j = Project(x, W_j)  (d_head dim)      |
  |     |     lambda_j = v_j^T @ v_j  (scalar)         |
  |     |     alpha_j = (1-exp(-beta*lambda_j))/lambda_j|
  |     |                                (EFLA exact)   |
  |     |     S = (I - alpha_j * v_j@v_j^T)*S           |
  |     |         + alpha_j * v_j @ value_j^T           |
  |     |   Chunk-wise parallel (C=64)                  |
  |     +---------------------------------------------+
  |     +Residual
  |     RMSNorm -> SwiGLU FFN (768->1920->768) -> +Residual
  |
  -> Final RMSNorm -> LM Head
```

### Why DeltaProduct Beats DeltaNet

**DeltaNet (rank-1):** Each token updates the state matrix S by rank-1:
```
S_t = (I - alpha * k @ k.T) @ S_{t-1} + alpha * k @ v.T
```
This is one gradient descent step of an online linear regression. The state can only erase information along ONE direction per token.

**DeltaProduct (rank-n_h):** Each token updates via a PRODUCT of n_h Householder reflections:
```
H_j = I - 2 * v_j @ v_j.T / (v_j.T @ v_j)
S_t = (H_1 @ H_2 @ ... @ H_{n_h}) @ S_{t-1} + update
```
This is n_h gradient descent steps — the state can erase/modify information along n_h independent directions per token.

**State tracking consequence:** Standard DeltaNet can track ONE state variable per step (e.g., the most recently opened bracket). DeltaProduct at n_h=4 can track FOUR independent state variables simultaneously (e.g., nested brackets, counters, flags).

### EFLA on Each Householder Step

Each Householder reflection involves the matrix `v_j @ v_j.T`, which has rank 1. EFLA's exact ODE solution exploits exactly this rank-1 structure:
```
lambda_j = v_j.T @ v_j           # scalar (element-wise sum)
alpha_j = (1 - exp(-beta * lambda_j)) / lambda_j  # exact
```
Without EFLA, n_h=4 sub-steps accumulate 4x the Euler discretization error per token. With EFLA, ALL sub-steps are exact — zero error regardless of n_h.

---

## Component 1: DeltaProduct Token Mixer

```python
class DeltaProductMixer(nn.Module):
    def __init__(self, d_model=768, n_heads=12, head_dim=64, n_householder=4):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_h = n_householder

        self.w_qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Householder direction projections (one per step)
        self.w_v = nn.ModuleList([
            nn.Linear(d_model, n_heads * head_dim, bias=False)
            for _ in range(n_householder)
        ])
        # Value projections (one per step, for additive update)
        self.w_val = nn.ModuleList([
            nn.Linear(d_model, n_heads * head_dim, bias=False)
            for _ in range(n_householder)
        ])

        # ShortConv on Q/K/V
        self.short_conv = nn.Conv1d(
            n_heads * head_dim, n_heads * head_dim,
            kernel_size=3, groups=n_heads * head_dim, padding=2
        )

        # Shared beta (learning rate)
        self.w_beta = nn.Linear(d_model, n_heads, bias=True)

        # Output gate + head norm
        self.w_gate = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.head_norm = nn.GroupNorm(n_heads, n_heads * head_dim)

    def forward(self, x, state=None):
        B, T, D = x.shape
        qkv = self.w_qkv(x)
        q, k, v_base = qkv.chunk(3, dim=-1)

        q = F.silu(self.short_conv(q.transpose(1, 2))[:, :, :T].transpose(1, 2))
        k = F.silu(self.short_conv(k.transpose(1, 2))[:, :, :T].transpose(1, 2))
        q = F.normalize(q.view(B, T, self.n_heads, self.head_dim), dim=-1)
        k = F.normalize(k.view(B, T, self.n_heads, self.head_dim), dim=-1)

        beta = torch.sigmoid(self.w_beta(x)).unsqueeze(-1)  # (B, T, H, 1)

        # Compute n_h Householder directions + values
        v_dirs = []
        v_vals = []
        for j in range(self.n_h):
            v_j = F.normalize(
                self.w_v[j](x).view(B, T, self.n_heads, self.head_dim), dim=-1
            )
            val_j = self.w_val[j](x).view(B, T, self.n_heads, self.head_dim)
            # EFLA exact coefficient for this Householder step
            lambda_j = (v_j * v_j).sum(dim=-1, keepdim=True)
            alpha_j = (1 - torch.exp(-beta * lambda_j)) / (lambda_j + 1e-6)
            v_dirs.append((v_j, alpha_j))
            v_vals.append(val_j)

        # Chunk-wise DeltaProduct scan
        o = deltaproduct_chunkwise(
            q, k, v_dirs, v_vals, chunk_size=64, state=state
        )

        gate = torch.sigmoid(self.w_gate(x)).view(B, T, self.n_heads, self.head_dim)
        o = gate * self.head_norm(
            o.flatten(-2, -1).transpose(1, 2)
        ).transpose(1, 2).view(B, T, self.n_heads, self.head_dim)
        return self.w_o(o.flatten(-2, -1))
```

## Component 2: DeltaProduct Chunk-Wise Kernel

The chunk-wise computation follows the WY representation used in DeltaNet/FLA. The key difference: within each chunk, n_h Householder products are applied sequentially per token. Between chunks, the state carries the accumulated product.

```python
def deltaproduct_chunkwise(q, k, v_dirs, v_vals, chunk_size=64, state=None):
    """Chunk-wise parallel DeltaProduct scan.
    
    Within each chunk: sequential over tokens, n_h products per token.
    Between chunks: parallel via WY representation.
    """
    B, T, H, D = q.shape
    n_chunks = T // chunk_size
    outputs = []

    for c in range(n_chunks):
        start = c * chunk_size
        end = start + chunk_size
        q_c = q[:, start:end]
        
        # Apply n_h Householder products for each token in chunk
        # This is where EFLA exactness matters most
        chunk_out = _process_chunk_deltaproduct(
            q_c, v_dirs, v_vals, state, start, end
        )
        outputs.append(chunk_out)

    return torch.cat(outputs, dim=1)
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_layers | 16 |
| n_heads | 12 |
| head_dim | 64 |
| n_householder | 4 (ablation: 1, 2, 4, 8) |
| ffn_inner | 1920 (2.5x) |
| conv_kernel | 3 |
| chunk_size | 64 |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257x768, tied) | 38.6M |
| **Per block:** | |
|   w_qkv (768->2304) | 1.77M |
|   w_o (768->768) | 0.59M |
|   ShortConv (k=3, 768ch) | 2.3K |
|   w_v x4 (768->768 each) | 2.36M |
|   w_val x4 (768->768 each) | 2.36M |
|   w_beta (768->12) | 9.2K |
|   w_gate (768->768) | 0.59M |
|   head_norm | 768 |
|   SwiGLU FFN: w_gate_up (768->3840) | 2.95M |
|   SwiGLU FFN: w_down (1920->768) | 1.47M |
|   RMSNorm x2 | 1.5K |
|   **Block total** | **~12.1M** |
| **16 blocks** | **193.6M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **OVER BUDGET (193.6M > 175M)** |

### Parameter Adjustment

At n_h=4 with separate projections, the model exceeds 175M. Options:
- **Option A:** Reduce n_h to 2 (saves 2.36M/layer = 37.8M). Total: ~155.8M. **Recommended.**
- **Option B:** Reduce d_model to 640. Total: ~152M.
- **Option C:** Share w_v projections across pairs (n_h/2 projections). Total: ~174.7M.
- **Option D:** Reduce layers to 14. Total: ~169.4M.

**Selected: Option A (n_h=2)** — rank-2 is the sweet spot per DeltaProduct paper's ablation. Rank-4 is diminishing returns for the parameter cost.

| Component (n_h=2) | Params |
|-----------|--------|
| w_v x2 + w_val x2 | 1.18M + 1.18M = 2.36M/layer |
| **Revised block total** | **~9.74M** |
| **16 blocks** | **155.8M** |
| **GRAND TOTAL** | **~155.8M** |

---

## Training

### Single Phase

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Full model, n_h=2 | Learn Householder dynamics + EFLA exactness |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 8e-4 cosine -> 8e-5, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 24x1024, accum=2 (48K effective) |
| Precision | fp16 mixed + fp32 state |
| Grad clip | 1.0 |
| w_v init | orthogonal (Householder directions should be diverse) |
| w_beta bias | -3.0 (conservative learning rate) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| DeltaProduct chunk-wise kernel doesn't exist in FLA | HIGH | Must implement custom. Structure is similar to DeltaNet WY but with multi-step products. Start with sequential Python, then optimize. |
| n_h=2 projections add significant compute | MEDIUM | Each w_v, w_val is a full 768->768 matmul. 4 extra matmuls/layer = ~20% more FLOP. But on Strix Halo, throughput = bandwidth, and these are fused GEMMs. |
| Householder orthogonality not maintained during training | LOW | L2-normalize v_j before use. Exact orthogonality not required — the product is well-defined for any non-zero v_j. |
| Quality improvement doesn't justify parameter cost | MEDIUM | Ablation n_h={1,2} directly. If n_h=1 matches n_h=2, we've rediscovered DeltaNet. Paper shows n_h=2 gives significant gains on state tracking. |
| Sequential n_h steps within chunk slow down parallel scan | MEDIUM | n_h=2 means 2 sequential steps per token within chunks. This doubles the intra-chunk work but chunks are parallelized. Net impact: ~1.5x slowdown vs DeltaNet. |

## Success Criteria

1. **Val loss < 2.82 on BabyLM** (beat AMADEUS by > 3% — this is the quality-first hypothesis)
2. Throughput > 10K tok/s (compile + autokernel)
3. n_h=2 > n_h=1 (DeltaNet) by > 2% loss improvement
4. State tracking tasks: perfect accuracy on bracket matching, counter tracking
5. Length extrapolation: maintain quality at 2x training context (2048 tokens)

---

## Implementation Roadmap

1. Implement DeltaProductMixer with n_h Householder steps + EFLA exact coefficients
2. Implement sequential DeltaProduct scan (Python, for correctness verification)
3. Implement chunk-wise DeltaProduct scan (WY adaptation)
4. Assemble GorgonBlock + GorgonModel (16 layers)
5. Verify parameter count (~156M at n_h=2)
6. Smoke test: 10 min, verify loss decreasing + state tracking accuracy
7. Register deltaproduct_chunkwise as torch.library custom op
8. Full training: BabyLM 2 epochs
9. Ablation: n_h = {1, 2} (with and without EFLA)
10. State tracking eval: bracket matching, counter tracking, copying tasks
11. Length extrapolation eval: train on 1024, eval on 2048

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), chunked_linear_cross_entropy

**External (1):** causal-conv1d (10x ShortConv)

**New (1):** deltaproduct_chunkwise — multi-step Householder product scan. Most complex kernel in this proposal. Can start with FLA DeltaNet kernel + Python loop over n_h.

### Householder Cost Analysis

Each Householder step per token:
```
v_j = W_v @ x        -> 768x768 matmul (rocBLAS, ~0.15ms)
L2Norm(v_j)          -> element-wise (free)
lambda_j = dot(v,v)  -> reduction (free)  
alpha_j = EFLA       -> element-wise (free)
State update         -> outer product (free)
```

For n_h=2: 4 extra matmuls/layer (2 for v, 2 for val) at 768x768 = ~0.6ms extra.
For 16 layers: ~9.6ms additional per forward pass.

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Eager fp16 | 16L, n_h=2 | ~7K tok/s |
| + autokernel | | ~10K tok/s |
| + compile | | ~13K tok/s |
| + causal-conv1d + custom scan | | **~14K tok/s** |

**Estimated optimized throughput:** ~12-14K tok/s
**Tokens in 45 min:** ~32-38M (2.0-2.4 BabyLM epochs)
**Ranking:** #8-10 of all architectures (quality-focused, not throughput-optimized)

This is deliberately a quality play. GORGON trades throughput for the most expressive recurrence in our portfolio.
