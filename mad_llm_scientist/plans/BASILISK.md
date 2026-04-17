---
title: "BASILISK"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/BIFROST.md
  - mad_llm_scientist/plans\AMADEUS.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %basilisk, %hybrid-head, %falcon-h1, %mamba3, %kda, %gated-attention]
---

# BASILISK

**Hybrid-Head Dynamic Mode Selection — The Serpent King Whose Each Scale Sees Differently**

*"The basilisk does not see with one eye — each scale is a different sense. Some scales taste heat, others feel vibration, others see light. The creature that combines all senses survives."*
*Falcon-H1 proved hybrid-heads beat hybrid-layers. Gated Attention proved one sigmoid is a free lunch.*

## Hypothesis

Falcon-H1 (TII, July 2025, 71 upvotes) proved that placing different head types (SSM + attention) **within the same layer** outperforms interleaving them across layers. BASILISK extends this to THREE head types — KDA channel-wise recurrence, Mamba-3 complex MIMO SSM, and gated softmax attention — with a learned per-head mode gate that dynamically allocates head types based on input. Gated Attention (Qwen team, May 2025) adds a sigmoid gate after SDPA that consistently improves performance across 30 model variants — a confirmed free lunch. On Strix Halo, KDA and Mamba-3 heads are element-wise (free); only attention heads cost matmuls. If the model learns 25% attention allocation, the architecture is 4x cheaper than full attention at near-full quality.

**Key papers:** "Falcon-H1" (2507.22448, Jul 2025), "Mamba-3" (2603.15569, Mar 2026), "Kimi Linear/KDA" (2510.26692), "Gated Attention" (2505.06708, May 2025)

---

## Architecture

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  -> 16 Basilisk Blocks:
  |     RMSNorm
  |     +---------------------------------------------+
  |     | Shared QKV Projection (768 -> 3*768)         |
  |     | Split into 12 heads (head_dim=64)            |
  |     |                                              |
  |     | Per-head mode gate:                          |
  |     |   g_h = MLP(pool(x)) -> R^3                  |
  |     |   weights = softmax(g_h / tau)               |
  |     |   [w_kda, w_mamba3, w_attn]                  |
  |     |                                              |
  |     | Head types:                                  |
  |     |   Type A (KDA): channel-wise gated recurrence|
  |     |     alpha_t in [0,1]^{d_k}, delta rule       |
  |     |     Element-wise -> FREE                     |
  |     |                                              |
  |     |   Type B (Mamba-3): complex MIMO SSM         |
  |     |     Complex state, d_state=16 complex        |
  |     |     Element-wise -> FREE                     |
  |     |                                              |
  |     |   Type C (Gated Attn): SDPA + sigmoid gate   |
  |     |     sigma(gate) * softmax(QK^T/sqrt(d))V    |
  |     |     Matmul-based -> costs compute            |
  |     |                                              |
  |     | Training: soft mix (all 3 computed)           |
  |     | Inference: hard argmax (only dominant type)   |
  |     +---------------------------------------------+
  |     +Residual
  |     RMSNorm -> SwiGLU FFN (768->1920->768) -> +Residual
  |
  -> Final RMSNorm -> LM Head
```

### Why Hybrid-Head, Not Hybrid-Layer

AMADEUS uses **hybrid-pathway** (Conv + Mamba in parallel as separate wide paths). BIFROST uses **hybrid-layer** (KDA layers interleaved with attention layers). BASILISK uses **hybrid-head** (different head types within the same multi-head structure).

Advantages of hybrid-head:
1. **Shared QKV projection** — one fused matmul serves all head types (rocBLAS-efficient)
2. **Per-head specialization** — different tokens naturally need different processing
3. **Finer granularity** — 12 heads give 12 independent decisions vs 16 layers for hybrid-layer
4. **Proven at scale** — Falcon-H1 validated hybrid-head at 1.5B-34B parameters

### The Three Head Types

**Type A — KDA (Kimi Delta Attention):**
- Channel-wise decay: alpha_t ∈ [0,1]^{d_k} per head
- Delta rule with EFLA exact coefficient
- Best for: smooth continuation, language modeling, gradual state evolution
- Cost: element-wise = **free**

**Type B — Mamba-3 Complex MIMO:**
- Complex-valued state (d_state=16 complex per head)
- Magnitude decay + phase rotation
- Best for: periodic patterns, long-range dependency, position encoding
- Cost: element-wise complex multiply = **free**

**Type C — Gated Softmax Attention:**
- Standard SDPA + learned sigmoid gate (Qwen's finding)
- The gate provides non-linearity on attention output, eliminates attention sinks
- Best for: factual retrieval, in-context learning, copying
- Cost: matmul-based (only expensive head type)

### Gated Attention Free Lunch

The Qwen team tested a simple modification across 30 variants of 15B MoE models:
```python
attn_output = sigmoid(gate) * scaled_dot_product_attention(q, k, v)
```
This consistently improved performance by:
- Introducing non-linearity on the low-rank attention mapping
- Creating query-dependent sparse gating (gate near 0 → skip that head's contribution)
- Eliminating attention sinks (no more wasted capacity on BOS tokens)
- Improving length extrapolation

The gate is one sigmoid (free on Strix Halo). All attention heads in BASILISK use this.

---

## Component 1: Per-Head Mode Gate

```python
class HeadModeGate(nn.Module):
    def __init__(self, d_model=768, n_heads=12, n_modes=3, init_temp=1.0):
        self.gate_proj = nn.Linear(d_model, n_heads * n_modes, bias=True)
        self.n_heads = n_heads
        self.n_modes = n_modes
        self.temperature = init_temp
        # Initialize: bias toward KDA (mode 0)
        nn.init.zeros_(self.gate_proj.weight)
        bias = torch.zeros(n_heads * n_modes)
        bias[0::n_modes] = 1.0  # KDA bias
        self.gate_proj.bias = nn.Parameter(bias)

    def forward(self, x):
        B, T, D = x.shape
        pooled = x.mean(dim=1)  # (B, D)
        logits = self.gate_proj(pooled).view(B, self.n_heads, self.n_modes)
        if self.training:
            weights = F.softmax(logits / self.temperature, dim=-1)
        else:
            weights = F.one_hot(logits.argmax(dim=-1), self.n_modes).float()
        return weights  # (B, n_heads, 3)
```

## Component 2: KDA Head

```python
class KDAHead(nn.Module):
    def __init__(self, d_model=768, head_dim=64):
        self.w_alpha_down = nn.Linear(d_model, head_dim // 4, bias=False)
        self.w_alpha_up = nn.Linear(head_dim // 4, head_dim, bias=False)
        self.w_beta = nn.Linear(d_model, 1, bias=True)

    def forward(self, q, k, v, x_for_gate, state=None):
        alpha = torch.sigmoid(self.w_alpha_up(F.silu(self.w_alpha_down(x_for_gate))))
        beta = torch.sigmoid(self.w_beta(x_for_gate))
        lambda_t = (k * k).sum(dim=-1, keepdim=True)
        efla_alpha = (1 - torch.exp(-beta * lambda_t)) / (lambda_t + 1e-6)
        return kda_head_chunkwise(q, k, v, alpha, efla_alpha, chunk_size=64, state=state)
```

## Component 3: Mamba-3 Complex Head

```python
class Mamba3ComplexHead(nn.Module):
    def __init__(self, head_dim=64, d_state=16):
        self.d_state = d_state
        self.A_log_mag = nn.Parameter(-torch.ones(d_state))
        self.A_phase = nn.Parameter(torch.linspace(0, 6.28, d_state))
        self.w_B = nn.Linear(head_dim, 2 * d_state, bias=False)
        self.w_C = nn.Linear(head_dim, 2 * d_state, bias=False)
        self.w_dt = nn.Linear(head_dim, 1, bias=True)
        nn.init.constant_(self.w_dt.bias, -4.0)

    def forward(self, q, k, v, state=None):
        dt = F.softplus(self.w_dt(v))
        A_mag = torch.exp(-torch.exp(self.A_log_mag) * dt)
        A_real = A_mag * torch.cos(self.A_phase * dt)
        A_imag = A_mag * torch.sin(self.A_phase * dt)
        B_complex = self.w_B(k)
        C_complex = self.w_C(q)
        return complex_ssm_head(A_real, A_imag, B_complex, C_complex, v, state=state)
```

## Component 4: Gated Attention Head

```python
class GatedAttentionHead(nn.Module):
    def __init__(self, head_dim=64):
        self.gate_param = nn.Parameter(torch.zeros(head_dim))

    def forward(self, q, k, v):
        attn_out = F.scaled_dot_product_attention(
            q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2),
            is_causal=True
        ).squeeze(2)
        gate = torch.sigmoid(self.gate_param)
        return gate * attn_out
```

## Component 5: Basilisk Block Assembly

```python
class BasiliskBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12, head_dim=64):
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.w_qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.mode_gate = HeadModeGate(d_model, n_heads)
        self.kda_heads = nn.ModuleList([KDAHead(d_model, head_dim) for _ in range(n_heads)])
        self.mamba3_heads = nn.ModuleList([Mamba3ComplexHead(head_dim) for _ in range(n_heads)])
        self.attn_heads = nn.ModuleList([GatedAttentionHead(head_dim) for _ in range(n_heads)])
        self.head_norm = nn.GroupNorm(n_heads, n_heads * head_dim)
        self.ffn = SwiGLUFFN(d_model, 1920)

    def forward(self, x):
        h = self.norm1(x)
        qkv = self.w_qkv(h)
        q, k, v = [t.view(*t.shape[:2], 12, 64) for t in qkv.chunk(3, dim=-1)]
        mode_weights = self.mode_gate(h)  # (B, 12, 3)

        outputs = []
        for i in range(12):
            w_kda, w_m3, w_attn = mode_weights[:, i, 0], mode_weights[:, i, 1], mode_weights[:, i, 2]
            o_kda = self.kda_heads[i](q[:,:,i], k[:,:,i], v[:,:,i], h)
            o_m3 = self.mamba3_heads[i](q[:,:,i], k[:,:,i], v[:,:,i])
            o_attn = self.attn_heads[i](q[:,:,i], k[:,:,i], v[:,:,i])
            # Soft mix during training
            o_head = (w_kda.view(-1,1,1)*o_kda + w_m3.view(-1,1,1)*o_m3
                      + w_attn.view(-1,1,1)*o_attn)
            outputs.append(o_head)

        o = torch.stack(outputs, dim=2)  # (B, T, 12, 64)
        o = self.head_norm(o.flatten(-2,-1).transpose(1,2)).transpose(1,2)
        o = self.w_o(o)
        x = x + o
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_layers | 16 |
| n_heads | 12 |
| head_dim | 64 |
| ffn_inner | 1920 (2.5x) |
| n_modes | 3 (KDA, Mamba-3, GatedAttn) |
| gate_temperature | 1.0 -> 0.1 (anneal) |
| mamba3_d_state | 16 (complex) |
| conv_kernel | 3 (for KDA ShortConv) |
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
|   Mode gate (768->36) | 28K |
|   KDA heads x12: alpha proj + beta | 12 x (12.3K + 769) = 157K |
|   Mamba3 heads x12: A + B + C + dt | 12 x (32 + 2K + 2K + 65) = 49K |
|   GatedAttn heads x12: gate | 12 x 64 = 768 |
|   head_norm | 1.5K |
|   SwiGLU FFN | 4.42M |
|   RMSNorm x2 | 1.5K |
|   **Block total** | **~7.02M** |
| **16 blocks** | **112.3M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **~150.9M** |

Lean at 151M — well under 175M budget. The per-head mode machinery adds only ~235K params per block (3.3% overhead).

---

## Training

### Two Phases

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (70%) | 31 min | All heads compute all 3 modes (soft gate, tau=1.0) | Learn all head types + routing |
| 2 (30%) | 14 min | Anneal tau to 0.1, approaching hard routing | Specialize heads |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 8e-4 cosine -> 8e-5, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 24x1024, accum=2 (48K effective) |
| Precision | fp16 mixed + fp32 state for KDA/Mamba3 |
| Grad clip | 1.0 |
| Gate temperature | 1.0 -> 0.1 (linear anneal in Phase 2) |
| Gate diversity loss | 0.01 * entropy(mode_distribution_per_layer) |

**Diversity loss:** Penalizes uniform mode allocation. Encourages each layer to develop a distinctive head-type profile (e.g., early layers mostly KDA, late layers mostly attention).

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| All heads collapse to one mode | MEDIUM | Diversity loss + initial KDA bias + temperature annealing. If collapse still happens, assign modes manually (4 KDA, 4 Mamba3, 4 Attn). |
| Soft mixing 3 types per head is expensive during training | HIGH | All 3 must be computed during Phase 1. Mitigation: Phase 1 is 70%, Phase 2 with hard routing is cheaper. Post-training: only dominant type computed. |
| Per-head complex SSM (Mamba-3) has too few params | LOW | d_state=16 complex is small but each head is independent. 12 heads x 16 dims = 192 total complex states. |
| Mode gate routing is per-sequence, not per-token | MEDIUM | Per-sequence pooling is coarse. Could upgrade to per-token routing but that adds significant complexity. Per-sequence is the Flux Attention approach (proven). |
| Sequential per-head processing is slow | HIGH | During training, batch all heads of same type together. Use scatter/gather to route heads. Inference with hard routing: only compute selected type. |

## Success Criteria

1. Val loss < 2.88 on BabyLM (**beat AMADEUS**)
2. Throughput > 12K tok/s (compile + autokernel)
3. Mode specialization emerges: different heads/layers prefer different types
4. Gated attention ablation: gated > ungated by measurable margin
5. Post-convergence hard routing: < 0.5% quality loss vs soft routing
6. Inference throughput improves 1.5-2x with hard routing (skip unused head types)

---

## Implementation Roadmap

1. Implement KDAHead, Mamba3ComplexHead, GatedAttentionHead
2. Implement HeadModeGate with temperature annealing
3. Assemble BasiliskBlock with shared QKV + per-head routing
4. Assemble BasiliskModel (16 layers)
5. Verify parameter count (~151M)
6. Smoke test: 10 min, verify all 3 head types get gradient
7. Phase 1: train with soft gates (tau=1.0)
8. Phase 2: anneal temperature, observe head specialization
9. Visualize mode allocation per layer (heatmap: layer x head -> dominant mode)
10. Inference: switch to hard routing, measure speedup + quality impact
11. Ablation: gated vs ungated attention, 2 modes vs 3 modes

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), chunked_linear_cross_entropy

**External (3):** FLA DeltaNet kernel (KDA heads), hybrid_flash_sdpa_attention (attn heads), causal-conv1d (KDA ShortConv)

**New (0):** Mamba-3 complex head uses element-wise complex ops (PyTorch handles it). Mode gate is one Linear.

### Training Cost (Soft Mix)

During Phase 1, all 3 head types are computed for all 12 heads:
```
Per head, per layer:
  KDA:    ~0.05ms (element-wise + tiny matmuls)
  Mamba3: ~0.03ms (element-wise complex)
  GAttn:  ~0.3ms (SDPA matmul)
  Total:  ~0.38ms per head

Per layer: 12 x 0.38ms = 4.56ms
16 layers: 73ms per forward pass
+ FFN: ~25ms
Total forward: ~98ms -> ~10K tok/s
```

### Inference Cost (Hard Routing, ~3 KDA + 5 Mamba3 + 4 Attn learned ratio)

```
KDA: 3 heads x 0.05ms = 0.15ms
Mamba3: 5 heads x 0.03ms = 0.15ms  
GAttn: 4 heads x 0.3ms = 1.2ms
Total: 1.5ms per layer (vs 4.56ms soft mix)
16 layers: 24ms + FFN 25ms = 49ms -> ~20K tok/s
```

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Training (soft, Phase 1) | eager | ~7K tok/s |
| Training (soft, Phase 1) | compile + AK | ~10K tok/s |
| Training (near-hard, Phase 2) | compile + AK | ~14K tok/s |
| **Inference (hard routing)** | compile + AK | **~20K tok/s** |

**Estimated training throughput:** ~10-14K tok/s
**Estimated inference throughput:** ~18-22K tok/s
**Tokens in 45 min:** ~27-38M (1.7-2.4 BabyLM epochs)
**Ranking:** #7-9 training, #5-6 inference
