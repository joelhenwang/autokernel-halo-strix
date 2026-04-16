---
title: "EREBUS"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/JORMUNGANDR.md
  - mad_llm_scientist/plans/RESONANT-LOOP.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %erebus, %efla, %looped, %error-free]
---

# EREBUS

**Error-Free Looped Linear Attention — The Void That Remembers Perfectly**

*"From the primordial darkness, exact forms crystallize. Each iteration refines, never distorts."*
*JORMUNGANDR proved looping works (43K tok/s). EFLA proves iteration can be exact.*

## Hypothesis

Parcae-style block sharing (JORMUNGANDR's 43K tok/s throughput) combined with Error-Free Linear Attention (EFLA, Lei et al. Dec 2025) as the primary sequence mixer eliminates the fundamental instability of looped recurrent models — discretization error accumulation across iterations. EFLA solves the delta-rule ODE exactly via the rank-1 matrix exponential closed form, meaning 12-16 loop iterations produce **zero compounding error**. On Strix Halo, EFLA's core operations are entirely element-wise (free), and shared block parameters (~3MB) are L2-resident after iteration 1.

**Key paper:** "Error-Free Linear Attention is a Free Lunch" (2512.12602) — exact ODE solution for delta rule at same linear-time complexity.

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  → 1 SHARED BLOCK × 12 iterations:
  │     RMSNorm
  │     ┌──────────────────────────────────────────┐
  │     │ EFLA Token Mixer                         │
  │     │   ShortConv(k=3) on Q/K/V               │
  │     │   Swish activation                       │
  │     │   L2Norm on Q/K                          │
  │     │   α_t = (1-exp(-β_t·λ_t))/λ_t  (EXACT) │
  │     │   S_t = (I - α_t·k·kᵀ)S_{t-1} + α·k·vᵀ│
  │     │   Chunk-wise parallel (C=64)              │
  │     └──────────────────────────────────────────┘
  │     +Residual
  │     RMSNorm → SwiGLU FFN (768→1920→768) → +Residual
  │
  │     ★ Spectral norm constraint: ρ(transition) < 1 ★
  │     ★ Stochastic depth: Poisson(λ=12) iterations ★
  │
  → Final RMSNorm → LM Head
```

### Why EFLA, Not Standard Delta Rule

Standard DeltaNet uses Euler discretization:
```
S_t = (I - β_t·k_t·k_tᵀ)·S_{t-1} + β_t·k_t·v_tᵀ    (1st order, O(β²) error)
```

EFLA derives the exact closed-form solution by exploiting the rank-1 structure of A_t = k_t·k_tᵀ:
```
A^n = λ^{n-1}·A  where λ = k_tᵀ·k_t
exp(-β·A) = I - ((1 - exp(-β·λ))/λ)·A
```

This collapses the matrix exponential to a **scalar exponential + element-wise ops**. The update becomes:
```python
lambda_t = (k * k).sum(dim=-1)                    # scalar per token
alpha_t = (1 - torch.exp(-beta_t * lambda_t)) / lambda_t  # exact decay
S_t = (I - alpha_t * k @ k.T) @ S_{t-1} + alpha_t * k @ v.T
```

**All new ops (exp, division, sum) are element-wise → free on Strix Halo.**

In a looped architecture with 12 iterations, Euler error compounds as O(12·β²) per token. EFLA: **zero error regardless of iteration count**.

---

## Component 1: EFLA Token Mixer

```python
class EFLAMixer(nn.Module):
    def __init__(self, d_model=768, n_heads=12, head_dim=64, conv_kernel=3):
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Fused QKV projection (rocBLAS-friendly)
        self.w_qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        # ShortConv per head (following KDA parameterization)
        self.short_conv = nn.Conv1d(
            n_heads * head_dim, n_heads * head_dim,
            kernel_size=conv_kernel, groups=n_heads * head_dim,
            padding=conv_kernel - 1
        )
        # Beta (learning rate) projection
        self.w_beta = nn.Linear(d_model, n_heads, bias=True)
        # Output gate (low-rank, following KDA)
        self.w_gate = nn.Linear(d_model, n_heads * head_dim, bias=False)
        # Head-wise RMSNorm on output
        self.head_norm = nn.GroupNorm(n_heads, n_heads * head_dim)

    def forward(self, x, state=None):
        B, T, D = x.shape
        qkv = self.w_qkv(x)  # (B, T, 3*H*d)
        q, k, v = qkv.chunk(3, dim=-1)

        # ShortConv + Swish (element-wise, free)
        q = F.silu(self.short_conv(q.transpose(1,2))[:,:,:T].transpose(1,2))
        k = F.silu(self.short_conv(k.transpose(1,2))[:,:,:T].transpose(1,2))
        v = F.silu(self.short_conv(v.transpose(1,2))[:,:,:T].transpose(1,2))

        # L2Norm on Q/K (element-wise, free)
        q = F.normalize(q.view(B,T,self.n_heads,self.head_dim), dim=-1)
        k = F.normalize(k.view(B,T,self.n_heads,self.head_dim), dim=-1)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # Beta: data-dependent learning rate
        beta = torch.sigmoid(self.w_beta(x)).unsqueeze(-1)  # (B,T,H,1)

        # EFLA exact update (element-wise dominant)
        lambda_t = (k * k).sum(dim=-1, keepdim=True)  # (B,T,H,1)
        alpha_t = (1 - torch.exp(-beta * lambda_t)) / (lambda_t + 1e-6)

        # Chunk-wise parallel computation (same structure as DeltaNet)
        o = efla_chunkwise(q, k, v, alpha_t, chunk_size=64, state=state)

        # Output gate + head norm
        gate = torch.sigmoid(self.w_gate(x)).view(B, T, self.n_heads, self.head_dim)
        o = self.w_o((gate * self.head_norm(o.flatten(-2,-1).transpose(1,2)).transpose(1,2).view(B,T,self.n_heads,self.head_dim)).flatten(-2,-1))
        return o
```

**EFLA chunk-wise parallel**: Identical algebraic structure to DeltaNet's WY representation. The only change is replacing `β_t` with `α_t = (1 - exp(-β_t·λ_t)) / λ_t`. This means we can directly use the FLA library's DeltaNet kernel with a pre-computed `alpha_t`.

## Component 2: Parcae-Style Loop with Spectral Stability

```python
class ErebusModel(nn.Module):
    def __init__(self, d_model=768, n_iterations=12):
        self.embedding = nn.Embedding(50257, d_model)
        self.shared_block = ErebusBlock(d_model)  # EFLA + SwiGLU
        self.n_iterations = n_iterations
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, 50257, bias=False)
        self.lm_head.weight = self.embedding.weight  # tied

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        # Stochastic depth: sample iteration count from Poisson
        n_iter = self.n_iterations if not self.training else \
                 min(max(torch.poisson(torch.tensor(float(self.n_iterations))).int().item(), 8), 16)
        for i in range(n_iter):
            h = self.shared_block(h, iteration=i)
        return self.lm_head(self.final_norm(h))
```

**Spectral constraint:** After each EFLA state update, the transition matrix eigenvalues are bounded by `|1 - α_t·λ_t| ≤ 1` because `α_t·λ_t = 1 - exp(-β_t·λ_t) ∈ [0, 1)`. **EFLA is inherently stable** — no explicit spectral norm constraint needed (unlike JORMUNGANDR's SCORE damping).

## Component 3: SwiGLU FFN

```python
gate, up = linear(x_norm, 768 → 2×1920).chunk(2)
out = linear(silu(gate) * up, 1920 → 768)
```

Standard 2.5× expansion. Fused gate+up projection (1 large GEMM).

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_heads | 12 |
| head_dim | 64 |
| d_state (per head) | 64×64 = 4096 |
| ffn_inner | 1920 (2.5×) |
| n_iterations | 12 (train: Poisson, eval: fixed) |
| shared_blocks | 1 |
| conv_kernel | 3 |
| chunk_size | 64 |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×768, tied) | 38.6M |
| EFLA mixer: w_qkv (768→2304) | 1.77M |
| EFLA mixer: w_o (768→768) | 0.59M |
| EFLA mixer: ShortConv (k=3, 768ch) | 2.3K |
| EFLA mixer: w_beta (768→12) | 9.2K |
| EFLA mixer: w_gate (768→768) | 0.59M |
| EFLA mixer: head_norm | 768 |
| SwiGLU FFN: gate+up (768→3840) | 2.95M |
| SwiGLU FFN: down (1920→768) | 1.47M |
| RMSNorm ×2 | 1.5K |
| **Per shared block total** | **~7.38M** |
| **×1 block (shared across 12 iterations)** | **7.38M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **~46.0M unique** |
| **Effective params (12 iterations)** | **~127M effective** |

Only **46M unique parameters** — the core block is ~7.38M, which is **~3.7MB in fp16**. This fits comfortably in the 6MB L2 cache, enabling near-free repeated reads on iterations 2-12.

---

## Training

### Single Phase (simplest possible)

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Full model, 12 iterations | Learn language with error-free recurrence |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-3 cosine → 1e-4, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 32×1024, accum=2 (64K effective) |
| Precision | fp16 mixed + fp32 EFLA state |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 iterations |
| Stochastic depth | Poisson(λ=12), clamped [8, 16] |

**Note:** Higher LR than typical (1e-3 vs 8e-4) because EFLA's exact dynamics are more stable than Euler approximations. EFLA paper shows faster convergence at higher LR.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| EFLA chunk-wise kernel not available for gfx1151 | MEDIUM | Use FLA DeltaNet kernel with pre-computed α_t. Same algebraic structure. |
| 46M unique params too few for quality | MEDIUM | 12 iterations × 7.38M = 127M effective. Parcae showed 2.2× param efficiency. Scale iterations to 16 if needed. |
| Stochastic depth hurts convergence | LOW | Clamp minimum iterations to 8. EFLA is stable regardless of iteration count. |
| EFLA fp32 state precision needed | LOW | State matrix is 64×64 per head = 4K floats. fp32 state adds only 192KB per iteration. |
| ShortConv without causal-conv1d is slow | LOW | causal-conv1d already wired in; falls back to nn.Conv1d gracefully. |

## Success Criteria

1. Throughput > 35K tok/s (with compile + autokernel)
2. Val loss < 3.1 on BabyLM (2 epochs)
3. Zero NaN/Inf across all iterations (EFLA stability)
4. Quality improves monotonically with iteration count (ablation: 4, 8, 12, 16)
5. Core block parameters confirmed L2-resident (< 4MB fp16)

---

## Implementation Roadmap

1. Implement EFLA update rule: exact α_t computation from β_t and λ_t
2. Implement EFLA chunk-wise parallel (adapt FLA DeltaNet chunkwise kernel)
3. Implement EFLAMixer with ShortConv + L2Norm + output gate
4. Implement shared ErebusBlock (EFLA + SwiGLU FFN + residual + RMSNorm)
5. Implement ErebusModel with Parcae-style loop (stochastic depth)
6. Verify parameter count (~46M unique, ~127M effective)
7. Smoke test: 10 min on smoke-test-dataset, confirm > 30K tok/s + loss decreasing
8. Register EFLA chunk-wise scan as torch.library custom op for compile compatibility
9. Full training: BabyLM 2 epochs with compile + autokernel
10. Ablation: iteration count [4, 8, 12, 16] — verify quality scales with iterations
11. Compare vs JORMUNGANDR (same throughput class) and AMADEUS (quality target)

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), cross_entropy (1.8×), chunked_linear_cross_entropy (memory savings)

**External (2):** causal-conv1d (10× conv speedup), FLA DeltaNet kernel (chunk-wise EFLA)

**New (1):** EFLA α_t pre-computation kernel — fused exp + division + elementwise. Trivial to implement as HIP kernel, but PyTorch element-wise is already free.

### Throughput Estimate

| Mode | Unique Weights | Iterations | Effective Read | Throughput |
|------|---------------|------------|----------------|------------|
| Eager fp16 | 46M (92MB) | 12 | 92MB (L2 cached after iter 1) | ~20K tok/s |
| + autokernel | 46M | 12 | 92MB | ~28K tok/s |
| + torch.compile | 46M | 12 | 92MB (L2 dominant) | ~38K tok/s |
| + causal-conv1d + FLA | 46M | 12 | 92MB | **~42K tok/s** |

**L2 cache analysis:** Core block = 7.38M params × 2 bytes = 14.8MB. This does NOT fit L2 (6MB) in one pass, but the **hot path** (EFLA state matrices 12×64×64×2 = 96KB, RMSNorm 768×2 = 1.5KB, conv weights 3×768×2 = 4.6KB) totals ~100KB — well within L2. The large projections (QKV, FFN) stream through L2 but benefit from iteration reuse.

**Effective bandwidth multiplier from looping:** With 12 iterations, QKV+FFN weights are read 12× from LPDDR5X (~92MB × 12 = 1.1GB) but the element-wise state ops between iterations are L2-cached. Net: ~85% of time is weight streaming (bandwidth-limited), ~15% is L2-fast element-wise.

### Expected Performance

**Estimated throughput:** ~38-42K tok/s (compile + autokernel + external kernels)
**Tokens in 45 min:** ~103-113M (6.4-7.1 BabyLM epochs)
**Ranking:** #2 of 31 architectures (behind only LlamaModel 43K, tied with JORMUNGANDR)
