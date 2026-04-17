---
title: "CHIMERA"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/EREBUS.md
  - mad_llm_scientist/plans/BASILISK.md
  - mad_llm_scientist/plans/SENTINEL.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %chimera, %morphing, %gated-attention, %per-iteration, %looped]
---

# CHIMERA

**Per-Iteration Architecture Morphing — The Beast That Changes Shape at Every Depth**

*"The chimera is not one creature but many fused into one body — lion head for strength, goat body for endurance, serpent tail for cunning. At each depth of the loop, a different nature emerges."*
*Gated Attention proved one sigmoid is a free lunch. CHIMERA proves loops can morph their architecture per iteration.*

## Hypothesis

In a looped model, all iterations use the same architecture — but different depths benefit from different architectures. Early iterations should do local processing (fast recurrence), middle iterations should build global state (SSM), late iterations should do correction (attention). CHIMERA learns this depth-varying architecture via differentiable per-iteration mode gates. The gate for each iteration is a FIXED learnable parameter (not input-dependent), so the model discovers the optimal architecture-depth profile during training. Gated Attention (Qwen team, May 2025, confirmed across 30 model variants) provides a free lunch on all attention components. Post-convergence, the dominant pathway per iteration is hardened for inference speed.

**Key papers:** "Gated Attention for LLMs" (2505.06708, May 2025, Qwen/Alibaba), "Flux Attention" (2604.07394, Apr 2026), "Elastic Attention" (2601.17367, Jan 2026)

---

## Architecture

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  -> 1 SHARED BLOCK x 12 iterations:
  |     For iteration i:
  |       RMSNorm
  |       +---------------------------------------------+
  |       | Three pathways (shared weights):             |
  |       |   Path A: KDA/EFLA recurrence (element-wise) |
  |       |   Path B: Mamba-3 complex SSM (element-wise) |
  |       |   Path C: Gated Attention (matmul + sigmoid) |
  |       |                                              |
  |       | Per-iteration gate:                          |
  |       |   g_i in R^3 = learnable param for iter i    |
  |       |   weights_i = softmax(g_i / tau)             |
  |       |   output = w_A*PathA + w_B*PathB + w_C*PathC |
  |       +---------------------------------------------+
  |       +Residual
  |       RMSNorm -> SwiGLU FFN (768->1920->768) -> +Residual
  |
  |     Expected learned pattern:
  |       Iter 1-3:  [0.8, 0.2, 0.0]  -> mostly KDA (syntax)
  |       Iter 4-8:  [0.2, 0.7, 0.1]  -> mostly SSM (semantics)
  |       Iter 9-12: [0.1, 0.3, 0.6]  -> mostly attention (correction)
  |
  -> Final RMSNorm -> LM Head
```

### Why Per-Iteration Morphing

Hybrid models choose architecture per LAYER (BIFROST) or per HEAD (BASILISK). Both have fixed assignments that don't change with processing depth. But in a looped model, depth IS the variable — and different depths have fundamentally different roles:

**Early iterations (1-3):** The hidden state is close to the embedding. Processing is mainly syntactic — recognizing word boundaries, morphology, local n-grams. Fast recurrence (KDA) excels here.

**Middle iterations (4-8):** The hidden state has absorbed local context. Processing shifts to semantic — building phrase representations, tracking entities, maintaining coherence. Complex SSM (Mamba-3) with rotational dynamics excels here.

**Late iterations (9-12):** The hidden state is near-final. Processing is correction — resolving ambiguities, attending to distant context, final prediction refinement. Attention excels here because it can directly look up specific positions.

CHIMERA doesn't ASSUME this pattern — it DISCOVERS it via learned gates. The gates are per-iteration parameters (not input-dependent), so they converge to fixed architecture profiles. This is a form of **meta-architecture search within the training loop**.

### Gated Attention Free Lunch

Every attention head uses the Qwen team's sigmoid gate:
```python
attn_output = sigmoid(gate) * F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

This is a proven free lunch (tested across 30 variants at 15B scale): eliminates attention sinks, improves length extrapolation, provides query-dependent sparsity. Cost: one sigmoid (free on Strix Halo).

---

## Component 1: Shared Multi-Pathway Block

```python
class ChimeraBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12, head_dim=64, max_iterations=16):
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        # Shared QKV for all pathways
        self.w_qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Path A: KDA/EFLA recurrence
        self.kda_alpha_down = nn.Linear(d_model, head_dim, bias=False)
        self.kda_alpha_up = nn.Linear(head_dim, n_heads * head_dim, bias=False)
        self.kda_beta = nn.Linear(d_model, n_heads, bias=True)

        # Path B: Mamba-3 complex SSM
        self.mamba3_A_log_mag = nn.Parameter(-torch.ones(n_heads, 16))
        self.mamba3_A_phase = nn.Parameter(
            torch.linspace(0, 6.28, 16).unsqueeze(0).expand(n_heads, -1)
        )
        self.mamba3_B = nn.Linear(d_model, 2 * n_heads * 16, bias=False)
        self.mamba3_C = nn.Linear(d_model, 2 * n_heads * 16, bias=False)
        self.mamba3_dt = nn.Linear(d_model, n_heads, bias=True)

        # Path C: Gated Attention
        self.attn_gate = nn.Parameter(torch.zeros(n_heads, head_dim))

        # Per-iteration mode gates
        self.mode_logits = nn.Parameter(torch.zeros(max_iterations, 3))
        # Initialize: slight bias toward KDA for all iterations
        nn.init.constant_(self.mode_logits[:, 0], 0.5)

        # Head norm
        self.head_norm = nn.GroupNorm(n_heads, n_heads * head_dim)

        # SwiGLU FFN
        self.ffn_gate_up = nn.Linear(d_model, 2 * 1920, bias=False)
        self.ffn_down = nn.Linear(1920, d_model, bias=False)

    def forward(self, x, iteration=0, kda_state=None, mamba_state=None):
        h = self.norm1(x)
        B, T, D = h.shape
        qkv = self.w_qkv(h)
        q, k, v = [t.view(B, T, 12, 64) for t in qkv.chunk(3, dim=-1)]

        # Get mode weights for this iteration
        weights = F.softmax(self.mode_logits[iteration] / self.temperature, dim=-1)
        w_kda, w_mamba3, w_attn = weights[0], weights[1], weights[2]

        # Path A: KDA/EFLA
        if w_kda > 0.01:  # skip if near-zero
            alpha = torch.sigmoid(self.kda_alpha_up(F.silu(self.kda_alpha_down(h))))
            alpha = alpha.view(B, T, 12, 64)
            beta = torch.sigmoid(self.kda_beta(h)).unsqueeze(-1)
            lam = (k * k).sum(dim=-1, keepdim=True)
            efla_alpha = (1 - torch.exp(-beta * lam)) / (lam + 1e-6)
            o_kda = kda_chunkwise(q, k, v, alpha, efla_alpha, chunk_size=64)
        else:
            o_kda = torch.zeros_like(q)

        # Path B: Mamba-3 complex SSM
        if w_mamba3 > 0.01:
            dt = F.softplus(self.mamba3_dt(h)).unsqueeze(-1)
            o_mamba3 = complex_ssm_forward(
                q, v, self.mamba3_A_log_mag, self.mamba3_A_phase,
                self.mamba3_B, self.mamba3_C, dt, h
            )
        else:
            o_mamba3 = torch.zeros_like(q)

        # Path C: Gated Attention
        if w_attn > 0.01:
            attn_out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2),
                v.transpose(1, 2), is_causal=True
            ).transpose(1, 2)
            gate = torch.sigmoid(self.attn_gate)
            o_attn = gate * attn_out
        else:
            o_attn = torch.zeros_like(q)

        # Weighted combination
        o = w_kda * o_kda + w_mamba3 * o_mamba3 + w_attn * o_attn
        o = self.head_norm(o.flatten(-2, -1).transpose(1, 2)).transpose(1, 2)
        o = self.w_o(o.view(B, T, -1))
        x = x + o

        # FFN
        h2 = self.norm2(x)
        gate, up = self.ffn_gate_up(h2).chunk(2, dim=-1)
        x = x + self.ffn_down(F.silu(gate) * up)
        return x
```

## Component 2: Chimera Model

```python
class ChimeraModel(nn.Module):
    def __init__(self, d_model=768, n_iterations=12):
        self.embedding = nn.Embedding(50257, d_model)
        self.shared_block = ChimeraBlock(d_model, max_iterations=20)
        self.shared_block.temperature = 1.0  # anneal during training
        self.n_iterations = n_iterations
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, 50257, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        n_iter = self.n_iterations
        if self.training:
            n_iter = min(max(
                torch.poisson(torch.tensor(float(self.n_iterations))).int().item(),
                8), 16)
        for i in range(n_iter):
            h = self.shared_block(h, iteration=i)
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
| n_iterations | 12 (train: Poisson [8,16], eval: fixed) |
| shared_blocks | 1 |
| n_pathways | 3 (KDA, Mamba-3, GatedAttn) |
| mamba3_d_state | 16 (complex) |
| gate_temperature | 1.0 -> 0.1 (anneal) |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257x768, tied) | 38.6M |
| **Shared block:** | |
|   w_qkv (768->2304) | 1.77M |
|   w_o (768->768) | 0.59M |
|   KDA: alpha proj (768->64->768) | 0.10M |
|   KDA: beta (768->12) | 9.2K |
|   Mamba-3: A params (12x16x2) | 384 |
|   Mamba-3: B + C (768->384 each) | 0.59M |
|   Mamba-3: dt (768->12) | 9.2K |
|   Attn gate (12x64) | 768 |
|   Mode logits (16x3) | 48 |
|   head_norm | 1.5K |
|   SwiGLU FFN | 4.42M |
|   RMSNorm x2 | 1.5K |
|   **Block total** | **~7.49M** |
| **x1 shared block** | **7.49M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL (unique)** | **~46.1M** |
| **Effective params (12 iterations)** | **~127M effective** |

Same budget as EREBUS/SENTINEL. The mode logits add 48 parameters total (negligible). All three pathways share the QKV projection.

---

## Training

### Two Phases

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (60%) | 27 min | All paths computed, tau=1.0 (soft) | Learn pathway weights + KDA/Mamba3/Attn |
| 2 (40%) | 18 min | Anneal tau -> 0.1 (near-hard) | Specialize iteration profiles |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-3 cosine -> 1e-4, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 32x1024, accum=2 (64K effective) |
| Precision | fp16 mixed + fp32 states |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 iterations |
| Temperature | 1.0 -> 0.1 (cosine anneal in Phase 2) |
| Diversity loss | 0.005 * var(mode_logits[:, 0]) (encourage variation across iterations) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| All iterations collapse to same mode | MEDIUM | Diversity loss encourages variation. Different initialization per iteration slot. Visualize mode profile during training. |
| Soft mixing 3 paths during Phase 1 is 3x compute | HIGH | Threshold: skip path if weight < 0.01. During Phase 1, most iterations use all 3. Phase 2 pruning reduces to 1-2. |
| Mode logits overfit to Poisson-sampled iteration counts | LOW | During training, iterations vary [8,16]. Mode logits for iterations 1-8 always trained; 9-16 trained ~50% of time. Initialize 9-16 logits same as 8. |
| Gated attention path dominates (simplest gradient) | MEDIUM | Initialize KDA bias slightly higher. Monitor per-path gradient norms. If attention dominates, increase its temperature. |
| Post-convergence hard routing loses quality | LOW | Temperature annealing in Phase 2 gradually transitions. By tau=0.1, mixture is already near-hard. |

## Success Criteria

1. Val loss < 2.88 on BabyLM (**beat AMADEUS**)
2. Training throughput > 20K tok/s (Phase 2 with near-hard routing)
3. Non-trivial iteration-depth profile emerges (different modes at different iterations)
4. Post-convergence hard routing: < 0.5% quality loss
5. Inference throughput > 40K tok/s (hard routing + L2-resident block)
6. Ablation: CHIMERA > fixed-mode loop (same total compute)

---

## Implementation Roadmap

1. Implement ChimeraBlock with three pathways + per-iteration mode logits
2. Implement gated attention (sigmoid gate on SDPA output)
3. Implement KDA + EFLA pathway (reuse from BIFROST/EREBUS)
4. Implement Mamba-3 complex pathway (reuse from PHENIX)
5. Assemble ChimeraModel with looped execution
6. Verify parameter count (~46M unique)
7. Phase 1: train with soft gates, visualize mode evolution
8. Phase 2: anneal temperature, observe specialization
9. Visualize: mode allocation heatmap (iteration x mode weight)
10. Post-convergence: extract hard mode profile, measure speedup
11. Ablation: 3 modes vs 2 modes, fixed-profile vs learned-profile

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), chunked_linear_cross_entropy

**External (2):** FLA DeltaNet kernel (KDA path), hybrid_flash_sdpa_attention (attention path)

**New (0):** Complex SSM path uses element-wise ops. Mode gating is softmax over 3 values (negligible).

### Compute Cost by Phase

**Phase 1 (soft mixing, all 3 paths):**
```
Per iteration:
  KDA: ~0.3ms (element-wise + chunk scan)
  Mamba3: ~0.2ms (element-wise complex)
  GAttn: ~3.5ms (SDPA matmul)
  FFN: ~2ms
  Total: ~6ms per iteration
12 iterations: ~72ms per forward -> ~14K tok/s
```

**Phase 2 (near-hard, typically 1 path dominant):**
```
Typical profile at convergence:
  Iter 1-4: KDA dominant -> 0.3ms each = 1.2ms
  Iter 5-8: Mamba3 dominant -> 0.2ms each = 0.8ms
  Iter 9-12: GAttn dominant -> 3.5ms each = 14ms
  FFN all: 2ms x 12 = 24ms
  Total: ~40ms -> ~25K tok/s
```

**Inference (hard routing):**
```
Skip non-dominant paths entirely:
  Iter 1-4: KDA only -> 0.3ms x 4 = 1.2ms
  Iter 5-8: Mamba3 only -> 0.2ms x 4 = 0.8ms
  Iter 9-12: GAttn only -> 3.5ms x 4 = 14ms
  FFN: 2ms x 12 = 24ms
  Total: ~40ms -> ~25K tok/s
  (same as Phase 2 — hard routing = near-hard)
```

But if profile converges to FEWER attention iterations (e.g., only iters 11-12):
```
  Iter 1-6: KDA -> 0.3ms x 6 = 1.8ms
  Iter 7-10: Mamba3 -> 0.2ms x 4 = 0.8ms
  Iter 11-12: GAttn -> 3.5ms x 2 = 7ms
  FFN: 24ms
  Total: ~33.6ms -> ~30K tok/s
```

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Phase 1 (soft, all paths) | compile + AK | ~14K tok/s |
| Phase 2 (near-hard) | compile + AK | ~25K tok/s |
| Inference (hard, optimal profile) | compile + AK | ~28-35K tok/s |
| Inference (hard, + FLA + flash) | compile + AK + ext | **~32-40K tok/s** |

**Estimated training throughput:** ~14K (Phase 1) -> ~25K (Phase 2)
**Estimated inference throughput:** ~30-40K tok/s
**Tokens in 45 min:** ~38M (Phase 1) + ~27M (Phase 2) = ~65M total (4.1 BabyLM epochs)
**Ranking:** Training #5-7, Inference #3-5 of all architectures

The key value is the DISCOVERED architecture profile — understanding what processing type each depth requires. This insight can be applied to all future looped architectures.
