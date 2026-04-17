---
title: "PHENIX"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/EREBUS.md
  - mad_llm_scientist/plans/JORMUNGANDR.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %phenix, %mamba3, %complex, %mimo, %looped]
---

# PHENIX

**Complex-Valued MIMO Looped Blocks — The Flame That Burns in Two Dimensions**

*"The phoenix does not merely live again — it lives in a richer space. Where others see one dimension, the phoenix sees two: magnitude and phase, amplitude and rotation."*
*Mamba-3 proved complex states halve memory at equal quality. PHENIX proves half-state loops fit L2 better.*

## Hypothesis

Mamba-3 (Gu & Dao, March 2026) introduces complex-valued state updates + MIMO (multi-input multi-output). Complex states encode phase information — rotational dynamics that real-valued states cannot represent — yielding richer state evolution for the same memory footprint. At 1.5B scale, Mamba-3 achieves equivalent perplexity to Mamba-2 with **half the state size**. In a looped architecture on Strix Halo, halving the state means tighter L2 residency, faster iteration throughput, and more effective iterations within the same wall-clock budget. Complex arithmetic (4 FMAs per multiply) remains element-wise — completely free on bandwidth-bound hardware.

**Key paper:** "Mamba-3: Improved Sequence Modeling using State Space Principles" (2603.15569, March 2026, Gu & Dao)

---

## Architecture

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  -> 1 SHARED BLOCK x 14 iterations (more than EREBUS due to smaller state):
  |     RMSNorm
  |     +---------------------------------------------+
  |     | Complex MIMO SSM Mixer                       |
  |     |   Input: x_real -> project to complex z      |
  |     |   z = x_r + i*x_i  (via learned split)      |
  |     |                                              |
  |     |   State: S_t in C^{n_heads x d_state}        |
  |     |   d_state = 32 complex (= 64 real equivalent)|
  |     |                                              |
  |     |   A_t in C^{d_state}: complex diagonal       |
  |     |     |A_t| < 1 (magnitude), angle free (phase)|
  |     |   S_t = A_t * S_{t-1} + B_t * x_t  (MIMO)   |
  |     |   y_t = Re(C_t * S_t)  (project to real)     |
  |     |                                              |
  |     |   MIMO: 4 input channels, 4 output channels  |
  |     |   sharing same state (multi-in multi-out)     |
  |     +---------------------------------------------+
  |     +Residual
  |     RMSNorm -> SwiGLU FFN (768->1920->768) -> +Residual
  |
  |     * Stochastic depth: Poisson(lambda=14), clamp [10,18] *
  |
  -> Final RMSNorm -> LM Head
```

### Why Complex-Valued States

Real-valued SSM state evolution: `S_t = a * S_{t-1} + b * x_t` where a is a real decay.

Complex-valued: `S_t = (a_r + i*a_i) * S_{t-1} + (b_r + i*b_i) * x_t`

The complex multiplication encodes BOTH decay (magnitude) AND rotation (phase):
- `|A| < 1`: state decays over time (forgetting)
- `angle(A)`: state rotates in 2D plane (phase tracking)

Phase tracking is crucial for:
- **Positional encoding**: oscillating A encodes position naturally (like RoPE but learned)
- **Periodic patterns**: rhythm, meter, repetition in language
- **Interference**: two signals constructively/destructively interfere based on phase alignment

A real-valued d_state=64 requires 64 parameters to represent 64 independent decay rates.
A complex-valued d_state=32 uses 64 parameters (32 real + 32 imaginary) but encodes 32 independent decay rates AND 32 independent phase rotations — **strictly more information**.

### MIMO (Multi-Input Multi-Output)

Standard SSM: one input channel x_t, one output y_t per state.
MIMO: n_in input channels share the same state, n_out output channels read from the same state.

```
Standard: x -> B -> S -> C -> y    (rank-1 update)
MIMO:     [x1,x2,x3,x4] -> [B1,B2,B3,B4] -> S -> [C1,C2,C3,C4] -> [y1,y2,y3,y4]
```

MIMO with 4 I/O channels achieves rank-4 updates per step — same expressivity as 4 separate SSM heads but with shared state (less memory, more parameter-efficient).

---

## Component 1: Complex MIMO SSM Mixer

```python
class ComplexMIMOSSM(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_state=32, n_io=4):
        self.n_heads = n_heads
        self.d_state = d_state  # complex dims
        self.n_io = n_io
        self.head_dim = d_model // n_heads  # 64

        # Project input to complex domain
        self.w_in = nn.Linear(d_model, 2 * n_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Complex A: learned magnitude + phase
        self.A_log_mag = nn.Parameter(-torch.ones(n_heads, d_state))
        self.A_phase = nn.Parameter(torch.linspace(0, 2 * 3.14159, d_state).unsqueeze(0).expand(n_heads, -1))

        # MIMO B and C: n_io channels, complex
        self.w_B = nn.Linear(d_model, 2 * n_heads * d_state * n_io, bias=False)
        self.w_C = nn.Linear(d_model, 2 * n_heads * d_state * n_io, bias=False)

        # Data-dependent discretization (dt)
        self.w_dt = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.w_dt.bias, -4.0)

        # Output gate
        self.w_gate = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.head_norm = nn.GroupNorm(n_heads, n_heads * self.head_dim)

    def forward(self, x, state=None):
        B, T, D = x.shape

        # Input to complex domain: split into real + imaginary
        z = self.w_in(x)  # (B, T, 2 * n_heads * head_dim)
        z_r, z_i = z.chunk(2, dim=-1)
        # z_complex conceptually = z_r + i*z_i

        # Complex A via magnitude * exp(i*phase)
        dt = F.softplus(self.w_dt(x)).unsqueeze(-1)  # (B, T, H, 1)
        A_mag = torch.exp(-torch.exp(self.A_log_mag) * dt.squeeze(-1).unsqueeze(-1))
        A_phase_t = self.A_phase.unsqueeze(0).unsqueeze(0) * dt
        # A = A_mag * exp(i * A_phase_t)
        A_real = A_mag * torch.cos(A_phase_t)
        A_imag = A_mag * torch.sin(A_phase_t)

        # MIMO B, C projections (complex)
        B_flat = self.w_B(x).view(B, T, self.n_heads, self.d_state, self.n_io, 2)
        C_flat = self.w_C(x).view(B, T, self.n_heads, self.d_state, self.n_io, 2)
        B_r, B_i = B_flat[..., 0], B_flat[..., 1]
        C_r, C_i = C_flat[..., 0], C_flat[..., 1]

        # Chunk-wise complex MIMO scan
        o = complex_mimo_chunkwise(
            z_r, z_i, A_real, A_imag, B_r, B_i, C_r, C_i,
            self.n_io, chunk_size=64, state=state
        )
        # o is real-valued after Re() projection: (B, T, n_heads, head_dim)

        gate = torch.sigmoid(self.w_gate(x)).view(B, T, self.n_heads, self.head_dim)
        o = gate * self.head_norm(
            o.flatten(-2, -1).transpose(1, 2)
        ).transpose(1, 2).view(B, T, self.n_heads, self.head_dim)
        return self.w_o(o.flatten(-2, -1))
```

## Component 2: Phenix Looped Model

```python
class PhenixModel(nn.Module):
    def __init__(self, d_model=768, n_iterations=14):
        self.embedding = nn.Embedding(50257, d_model)
        self.shared_block = PhenixBlock(d_model)
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
                10), 18)
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
| d_state | 32 (complex = 64 real equivalent) |
| n_io | 4 (MIMO channels) |
| ffn_inner | 1920 (2.5x) |
| n_iterations | 14 (train: Poisson [10,18], eval: fixed 14) |
| shared_blocks | 1 |
| chunk_size | 64 |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257x768, tied) | 38.6M |
| **Shared block:** | |
|   Complex MIMO: w_in (768->1536) | 1.18M |
|   Complex MIMO: w_o (768->768) | 0.59M |
|   Complex MIMO: A params (12x32x2) | 768 |
|   Complex MIMO: w_B (768->12x32x4x2) | 2.36M |
|   Complex MIMO: w_C (768->12x32x4x2) | 2.36M |
|   Complex MIMO: w_dt (768->12) | 9.2K |
|   Complex MIMO: w_gate (768->768) | 0.59M |
|   Complex MIMO: head_norm | 768 |
|   SwiGLU FFN: w_gate_up (768->3840) | 2.95M |
|   SwiGLU FFN: w_down (1920->768) | 1.47M |
|   RMSNorm x2 | 1.5K |
|   **Block total** | **~11.5M** |
| **x1 block (shared across 14 iterations)** | **11.5M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL (unique)** | **~50.1M** |
| **Effective params (14 iterations)** | **~199M effective** |

Complex MIMO adds ~4M params over EREBUS's EFLA mixer (for B, C projections), but MIMO's rank-4 updates are strictly more expressive. The block at ~5.8MB in fp16 fits inside the 6MB L2 cache for iterations 2-14.

---

## Training

### Single Phase

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Full model, 14 iterations | Learn complex dynamics + MIMO interactions |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 8e-4 cosine -> 8e-5, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 32x1024, accum=2 (64K effective) |
| Precision | fp16 mixed + fp32 complex state |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 iterations |
| A_log_mag init | -1.0 (unit decay) |
| A_phase init | linspace(0, 2*pi, d_state) per head |
| dt bias init | -4.0 (conservative discretization) |

**Complex-specific init:** A_phase initialized to evenly spaced angles around the unit circle. This ensures initial state captures all rotational frequencies (like a Fourier basis). A_log_mag at -1.0 gives moderate decay.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Complex arithmetic has no native PyTorch fp16 support | MEDIUM | Implement as real-valued pairs: (a_r, a_i). All ops are element-wise FMAs. torch.compile handles this well. |
| MIMO B/C projections too large (2.36M each) | LOW | At d_state=32 and n_io=4, these are moderate. Can reduce n_io to 2 if too heavy. |
| Phase initialization sensitive to frequency range | MEDIUM | Linspace(0, 2*pi) covers full frequency range. Ablation: random phase vs structured phase. |
| Complex gradients may be unstable | LOW | Complex SSMs are well-studied (S4, S4D). Magnitude constraint |A|<1 guarantees BIBO stability. |
| Chunk-wise scan for complex MIMO not in FLA | HIGH | Must implement custom. Can adapt mamba-ssm scan to complex dtype or implement as real-valued 2x scan. |

## Success Criteria

1. Val loss < 2.90 on BabyLM (**match or beat AMADEUS**)
2. Throughput > 40K tok/s (compile + autokernel)
3. Complex state tracks meaningful phase information (visualize A rotations)
4. MIMO ablation: n_io=4 > n_io=1 by > 1% loss improvement
5. 14 iterations outperform EREBUS at 12 (validates L2 advantage of smaller state)
6. Block fp16 size confirmed < 6MB (L2-resident)

---

## Implementation Roadmap

1. Implement complex arithmetic helpers (complex_mul, complex_conj, etc.) as real-valued pairs
2. Implement ComplexMIMOSSM with learned A (magnitude+phase), B, C projections
3. Implement complex MIMO chunk-wise scan (adapt mamba-ssm or write custom)
4. Implement PhenixBlock (ComplexMIMO + SwiGLU FFN + residual + RMSNorm)
5. Implement PhenixModel with Parcae-style loop (14 iterations)
6. Verify parameter count (~50M unique, ~199M effective)
7. Smoke test: verify complex gradients flow correctly, loss decreasing
8. Register complex_mimo_chunkwise as torch.library custom op
9. Full training: BabyLM 2 epochs
10. Ablation: complex vs real state (same parameter budget), n_io={1,2,4}
11. Ablation: iteration count {8, 10, 12, 14, 16}
12. Visualize phase dynamics: do heads learn different rotational frequencies?

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), chunked_linear_cross_entropy

**External (1):** mamba-ssm scan (5.6x, if adaptable to complex dtype)

**New (1):** complex_mimo_chunkwise — complex multiply + accumulate. Could implement as HIP kernel (element-wise complex mul = 4 FMAs, well within element-wise "free" regime).

### Complex Arithmetic Cost

```
Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
  = 4 real FMAs + 2 real adds
  Per-element cost: 6 FLOP (vs 2 for real multiply)
  
On Strix Halo (bandwidth-bound):
  Real multiply reads: 2 floats (8 bytes), computes 2 FLOP, AI = 0.25
  Complex multiply reads: 4 floats (16 bytes), computes 6 FLOP, AI = 0.375
  Both FAR below crossover (62.5 FLOP/byte) -> BOTH bandwidth-bound -> BOTH "free"
```

Complex arithmetic is 3x more FLOPs but reads 2x more data. Net: ~50% more bandwidth per element. But since element-wise ops are hidden behind matmul memory traffic, this cost is invisible.

### State Memory

| State Type | Size per head per token | Total for 12 heads |
|------------|------------------------|-------------------|
| EREBUS (real d=64) | 64 x 64 x 2B = 8KB | 96KB |
| PHENIX (complex d=32) | 32 x 32 x 4B = 4KB | 48KB |

**PHENIX state is 2x smaller.** This means the state working set during iteration is tighter, improving L2 hit rates.

### Throughput Estimate

| Mode | Iterations | Throughput |
|------|------------|------------|
| Eager fp16 | 14 | ~18K tok/s |
| + autokernel | 14 | ~26K tok/s |
| + compile | 14 | ~36K tok/s |
| + mamba-ssm (complex adapted) | 14 | **~42K tok/s** |

**Estimated optimized throughput:** ~38-42K tok/s
**Tokens in 45 min:** ~103-113M (6.4-7.1 BabyLM epochs)
**Ranking:** #2-3 of all architectures (tied with EREBUS/SENTINEL throughput class)
