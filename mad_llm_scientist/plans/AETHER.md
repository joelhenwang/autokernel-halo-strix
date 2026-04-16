---
title: "AETHER"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/ARGUS-PRIME.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %aether, %titans, %efla, %three-tier-memory]
---

# AETHER

**Three-Tier Memory with EFLA-Backed Neural Memory — The Ether That Permeates All Scales**

*"The ancient Greeks believed aether filled all space between earth and stars. This model fills all timescales between token and epoch."*
*Titans proved three-tier memory works. EFLA makes the long-term tier exact. Strix Halo's memory hierarchy makes it natural.*

## Hypothesis

Titans (Google, Dec 2024) introduced a three-tier memory architecture: sliding-window (short-term), attention (medium-term), and neural memory (long-term) that scales to 2M+ context. AETHER adapts this for Strix Halo by: (1) replacing medium-term attention with Error-Free Linear Attention (EFLA) for exact dynamics at linear cost, (2) implementing the neural memory gradient as element-wise ops (free on Strix Halo), and (3) mapping the three tiers to the chip's actual memory hierarchy (LDS → L2 → LPDDR5X). The result is a model that captures all temporal scales while keeping computation bandwidth-optimal.

**Key papers:** "Titans: Learning to Memorize at Test Time" (2501.00663), "Error-Free Linear Attention" (2512.12602)

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  → 16 Three-Tier Memory Blocks:
  │     RMSNorm
  │     ┌──────────────────────────────────────────────────┐
  │     │ TIER 1: REFLEX (Sliding Window, w=64)           │
  │     │   GQA: 4Q/2KV, head_dim=64, window=64          │
  │     │   Maps to: LDS (64KB per CU)                    │
  │     │   Cost: small rocBLAS GEMM, bounded              │
  │     │   Purpose: immediate token dependencies          │
  │     │                                                  │
  │     │ TIER 2: WORKING (EFLA, linear complexity)       │
  │     │   12 heads, head_dim=64, chunk_size=64           │
  │     │   Maps to: L2 cache (state matrices ~96KB)       │
  │     │   Cost: element-wise dominant                    │
  │     │   Purpose: medium-range context compression      │
  │     │                                                  │
  │     │ TIER 3: DEEP (Neural Memory, gradient-based)    │
  │     │   Memory module: d=768, surprise-gated           │
  │     │   Maps to: LPDDR5X (persistent state)            │
  │     │   Cost: element-wise gradient + outer product    │
  │     │   Purpose: long-term knowledge accumulation      │
  │     │                                                  │
  │     │ ──── 3-Way Gated Combination ────                │
  │     │ gate_r·o_reflex + gate_w·o_working + gate_d·o_deep│
  │     └──────────────────────────────────────────────────┘
  │     +Residual
  │     RMSNorm → SwiGLU FFN (768→1920→768) → +Residual
  │
  │     ★ + 64 PERSISTENT MEMORY TOKENS (learned, prepended) ★
  │
  → Final RMSNorm → LM Head
```

---

## Component 1: Reflex Tier (Sliding Window Attention)

```python
class ReflexTier(nn.Module):
    """Tiny sliding window for immediate local dependencies."""
    def __init__(self, d_model=768, n_q_heads=4, n_kv_heads=2, head_dim=64, window=64):
        self.w_qkv = nn.Linear(d_model, (n_q_heads + 2*n_kv_heads)*head_dim, bias=False)
        self.w_o = nn.Linear(n_q_heads * head_dim, d_model, bias=False)
        self.window = window
        # RoPE for local position awareness
        self.rotary = RotaryEmbedding(head_dim)

    def forward(self, x):
        # GQA with sliding window mask, RoPE
        # Uses SDPA with custom mask for training
        # Window=64 means KV cache is bounded to 64 tokens
        ...
        return o  # (B, T, d_model)
```

Cost: Very small. Window=64 means each query attends to at most 64 keys. The Q·K^T matrix is 64×64 per head — fits in LDS.

## Component 2: Working Tier (EFLA)

```python
class WorkingTier(nn.Module):
    """EFLA for medium-range context with exact dynamics."""
    def __init__(self, d_model=768, n_heads=6, head_dim=64):
        self.w_qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.short_conv = CausalConv1d(n_heads*head_dim, kernel_size=3)
        self.w_beta = nn.Linear(d_model, n_heads, bias=True)

    def forward(self, x, state=None):
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        q = F.silu(self.short_conv(q))
        k = F.silu(self.short_conv(k))
        v = F.silu(self.short_conv(v))
        q = F.normalize(q.view(B,T,self.n_heads,64), dim=-1)
        k = F.normalize(k.view(B,T,self.n_heads,64), dim=-1)

        beta = torch.sigmoid(self.w_beta(x)).unsqueeze(-1)
        lambda_t = (k * k).sum(dim=-1, keepdim=True)
        alpha_t = (1 - torch.exp(-beta * lambda_t)) / (lambda_t + 1e-6)

        o = efla_chunkwise(q, k, v.view(B,T,self.n_heads,64), alpha_t, chunk_size=64, state=state)
        return self.w_o(o.flatten(-2,-1))
```

State matrices (6 heads × 64×64) = 24K floats = 48KB → fits in L2.

## Component 3: Deep Tier (Neural Memory, Titans-style)

```python
class DeepTier(nn.Module):
    """Surprise-gated neural memory that learns to memorize at test time."""
    def __init__(self, d_model=768, mem_dim=768):
        self.mem_dim = mem_dim
        # Memory module parameters (learnable associative memory)
        self.w_k = nn.Linear(d_model, mem_dim, bias=False)
        self.w_v = nn.Linear(d_model, mem_dim, bias=False)
        self.w_q = nn.Linear(d_model, mem_dim, bias=False)
        self.w_o = nn.Linear(mem_dim, d_model, bias=False)
        # Surprise gate: controls when to memorize
        self.w_surprise = nn.Linear(d_model, 1, bias=True)
        # Memory learning rate (per-token, data-dependent)
        self.w_lr = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.w_lr.bias, -3.0)  # Small initial LR

    def forward(self, x, memory_state=None):
        B, T, D = x.shape
        if memory_state is None:
            memory_state = torch.zeros(B, self.mem_dim, self.mem_dim,
                                       device=x.device, dtype=torch.float32)

        k = self.w_k(x)  # (B, T, mem_dim)
        v = self.w_v(x)
        q = self.w_q(x)

        outputs = []
        for t in range(T):
            k_t = k[:, t]  # (B, mem_dim)
            v_t = v[:, t]
            q_t = q[:, t]

            # Read from memory
            o_t = memory_state @ q_t.unsqueeze(-1)  # (B, mem_dim, 1)
            o_t = o_t.squeeze(-1)

            # Surprise: how different is actual v from predicted v?
            predicted_v = memory_state @ k_t.unsqueeze(-1)
            surprise = F.mse_loss(predicted_v.squeeze(-1), v_t, reduction='none').mean(-1, keepdim=True)
            surprise_gate = torch.sigmoid(self.w_surprise(x[:, t]) + surprise)

            # Gradient-based memory update (element-wise dominant)
            lr = torch.sigmoid(self.w_lr(x[:, t]))  # (B, 1)
            # Loss: 0.5 * ||M·k - v||²
            # Gradient: (M·k - v)·kᵀ
            error = predicted_v.squeeze(-1) - v_t  # (B, mem_dim)
            grad = error.unsqueeze(-1) @ k_t.unsqueeze(-2)  # (B, mem_dim, mem_dim)

            # Surprise-gated update (element-wise, free on Strix Halo)
            memory_state = memory_state - lr.unsqueeze(-1) * surprise_gate.unsqueeze(-1) * grad
            outputs.append(o_t)

        o = torch.stack(outputs, dim=1)  # (B, T, mem_dim)
        return self.w_o(o), memory_state
```

**Why this is cheap on Strix Halo:** The gradient computation is an outer product (element-wise) + element-wise multiply. The memory read is a matmul but on a small matrix (768×768 = 0.6M params = 1.2MB). The surprise gate is element-wise (free). The sequential loop is the bottleneck — use chunked processing (chunk_size=64) with accumulated gradient for speedup.

## Component 4: Three-Way Gate

```python
class ThreeWayGate(nn.Module):
    def __init__(self, d_model=768):
        self.gate_proj = nn.Linear(d_model, 3, bias=True)
        nn.init.constant_(self.gate_proj.bias, torch.tensor([1.0, 1.0, 0.5]))

    def forward(self, x, o_reflex, o_working, o_deep):
        gates = F.softmax(self.gate_proj(x), dim=-1)  # (B, T, 3)
        return (gates[:,:,0:1] * o_reflex +
                gates[:,:,1:2] * o_working +
                gates[:,:,2:3] * o_deep)
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_layers | 16 |
| Reflex: n_q/n_kv/hd | 4/2/64, window=64 |
| Working: n_heads/hd | 6/64, EFLA chunk=64 |
| Deep: mem_dim | 768 |
| ffn_inner | 1920 (2.5×) |
| persistent_tokens | 64 (learned, prepended) |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×768, tied) | 38.6M |
| Persistent tokens (64×768) | 49K |
| **Per block:** | |
|   Reflex: w_qkv + w_o + rotary | 0.74M |
|   Working: w_qkv + conv + beta + w_o | 1.47M |
|   Deep: w_k + w_v + w_q + w_o + surprise + lr | 2.37M |
|   ThreeWayGate (768→3) | 2.3K |
|   SwiGLU FFN (768→1920→768) | 4.42M |
|   RMSNorm ×2 | 1.5K |
|   **Block total** | **~9.01M** |
| **16 blocks** | **144.1M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **~182.8M** |

**Note:** Slightly over 175M target. Can trim by reducing Deep tier to mem_dim=512 (saves ~25M) or reducing Reflex to 2Q/1KV heads.

---

## Training

### 3 Phases (progressive tier activation)

| Phase | Budget | Active Tiers | Purpose |
|-------|--------|-------------|---------|
| 1 (50%) | 22 min | Working (EFLA) only | Learn core language modeling with linear attention |
| 2 (30%) | 14 min | Working + Reflex | Add local precision from sliding window |
| 3 (20%) | 9 min | All three tiers + gate | Learn long-term memory and tier routing |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 6e-4 cosine → 6e-5, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 16×1024, accum=3 (48K effective) |
| Precision | fp16 mixed + fp32 neural memory state |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 layers |
| Deep tier chunk_size | 64 (accumulated gradient) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Deep tier sequential loop kills throughput | HIGH | Chunk the loop: accumulate gradients over 64 tokens, apply once. Reduces serial steps 16×. |
| Neural memory state explosion | MEDIUM | Memory state is fp32, 768×768 = 2.4MB per layer × 16 = 38MB total. Fits in memory. Clamp state values. |
| Three tiers + FFN = too many params | MEDIUM | Reduce Deep mem_dim to 512 or share Deep parameters across layers (every 4 layers). |
| Phase transitions destabilize training | LOW | Smooth activation: new tiers start with near-zero gate weights. Progressive warmup. |
| EFLA + SWA + Neural Memory = 3 different codepaths | MEDIUM | Each tier is independently tested. Integration via simple gated sum. |

## Success Criteria

1. Val loss < 2.90 on BabyLM (quality parity with AMADEUS)
2. Throughput > 8K tok/s (compile + autokernel)
3. Long-context tasks: needle-in-haystack accuracy > 80% at 4K context
4. Deep tier surprise gate activates on novel/rare tokens (measured)
5. Ablation: three-tier > two-tier > single-tier

---

## Implementation Roadmap

1. Implement EFLA Working tier (adapt from EREBUS)
2. Implement SWA Reflex tier with GQA and RoPE
3. Implement Titans-style Deep tier with surprise gating
4. Implement ThreeWayGate with progressive activation
5. Implement persistent memory tokens (64 learned tokens prepended)
6. Assemble AetherBlock + AetherModel, verify ~175M params (may need trimming)
7. Phase 1: EFLA-only training, 22 min
8. Phase 2: + Reflex, 14 min
9. Phase 3: + Deep + gate, 9 min
10. Ablation: single-tier, two-tier, three-tier comparisons
11. Long-context eval at 2K, 4K context lengths

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Memory Hierarchy Mapping

| Tier | Size | Maps To | Effective BW |
|------|------|---------|-------------|
| Reflex (SWA w=64) | Q·K^T: 64×64×4h = 16KB | LDS (64KB/CU) | ~10 TB/s |
| Working (EFLA state) | 6h×64×64×4B = 96KB | L2 (6MB) | ~2 TB/s |
| Deep (memory matrix) | 768×768×4B = 2.4MB per layer | L2 (6MB, partial) | ~500 GB/s |
| FFN weights | ~4.4M×2B = 8.8MB per layer | LPDDR5X | 240 GB/s |

The three tiers naturally exploit the three levels of the Strix Halo memory hierarchy. This is not a coincidence — the tier sizes were chosen to match the hardware.

### Kernel Reuse

**Reuse (5):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), cross_entropy (1.8×), chunked_linear_cross_entropy, hybrid_flash_sdpa_attention (8.9% faster, for Reflex tier)

**External (2):** causal-conv1d (10× Working tier), FLA DeltaNet kernel (EFLA)

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Eager fp16 | All tiers | ~5K tok/s |
| + autokernel + compile | All tiers | ~8K tok/s |
| + external kernels | All tiers | ~10K tok/s |
| Phase 1 only (EFLA) | Working tier only | ~14K tok/s |

**Bottleneck analysis:** Deep tier sequential loop dominates. With chunk_size=64: 1024/64 = 16 serial gradient accumulation steps per layer. Each step ~0.3ms (768×768 outer product). Total deep tier: ~0.3 × 16 × 16layers = 77ms. **This is the limiting factor.** Consider sharing Deep tier across every 4 layers (4× speedup → ~19ms).

**Estimated optimized throughput:** ~8-10K tok/s (compile + autokernel + external + chunked deep tier)
**Tokens in 45 min:** ~22-27M (1.4-1.7 BabyLM epochs)
**Ranking:** #18 of 31 architectures (quality-focused, not throughput-focused)
