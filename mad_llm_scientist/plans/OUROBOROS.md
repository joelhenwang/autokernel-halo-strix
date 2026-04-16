# OUROBOROS

**The Self-Devouring Loop: Parcae-Stable Looped ShortConv + GQA Coda + TTT Sniper + FiLM Mid-Loop + Momentum Coherence**

*The Ouroboros — the serpent that eats its own tail. An ancient symbol of eternal recursion, self-renewal, and the cycle of creation. Our model loops through itself, each iteration devouring the last, refining and refining until the serpent is ready to strike.*

*Six innovations from six sources, fused into one tiny beast:*
- **Parcae's stable loop** (SSM A-matrix constraint) — the serpent doesn't explode
- **LFM2.5's ShortConv** (proven cheapest operator) — the serpent's body
- **Momentum across loops** (our invention) — the serpent remembers its trajectory
- **FiLM at loop midpoint** (from AMADEUS) — the serpent WATCHES itself think
- **TTT sniper at the final layer** (from LAZARUS/ARGUS-PRIME) — the serpent's fangs
- **GQA in Coda** (from LFM2) — the serpent's eyes see globally before striking

**Design decisions documented in:** `plans/PRE_OUROBOROS.md`
**Research assessment of Parcae:** `plans/PARCAE.md`

---

## Hypothesis

Parcae proves that looped architectures achieve 2× parameter efficiency with SSM-style stability constraints. LFM2.5 proves that ShortConv + sparse GQA is the optimal block design. Our lab proved that FiLM (AMADEUS, val 2.90), momentum (Tempest), and TTT (LAZARUS/ARGUS) each contribute measurable quality improvements.

**OUROBOROS combines all of these:** a tiny core block of 3 ShortConv layers (~27M params) that loops 8 times through L2 cache with Parcae's stability guarantee, bracketed by a thin Prelude (2 unique layers) and an armed Coda (4 unique layers with GQA + TTT sniper). FiLM fingerprints the loop at its midpoint — the model watches its own thinking evolve.

**The bet:** 121M unique parameters with 338M effective capacity (3× param efficiency), running at 19-23K tok/s on Strix Halo, matching or beating AMADEUS (val 2.90) at 157M params.

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  ▼
┌──────────────────────────────────┐
│ PRELUDE (2 unique layers)        │
│   Layer 1: ShortConvBlock        │  ← Local pattern establishment
│   Layer 2: GQABlock + QK-Norm    │  ← One global attention pass
└──────────┬───────────────────────┘
           │ input_embed = h (saved for re-injection)
           ▼
┌──────────────────────────────────────────────────────────────┐
│ CORE LOOP (3 ShortConv layers × 8 iterations)                │
│                                                              │
│   velocity = 0                                               │
│                                                              │
│   for t in 1..8:                                             │
│     ┌─────────────────────────────────────────┐              │
│     │ Parcae injection:                       │              │
│     │   h = A·h + B·input_embed               │              │
│     │   (A = -exp(log_A), guaranteed stable)  │              │
│     │                                         │              │
│     │ Core block (3 ShortConv layers):        │              │
│     │   → RMSNorm → GatedConv → SwiGLU       │              │
│     │   → RMSNorm → GatedConv → SwiGLU       │              │
│     │   → RMSNorm → GatedConv → SwiGLU       │              │
│     │                                         │              │
│     │ Momentum coherence:                     │              │
│     │   velocity = β·velocity + block_output  │              │
│     │   h = h + velocity                      │              │
│     └─────────────────────────────────────────┘              │
│                                                              │
│     ★ t=4: FiLM fingerprint (mid-loop introspection)        │
│     ★ t=5..8: FiLM modulates ShortConv layers               │
│                                                              │
│   TRAINING: t=1..5 detached, t=6..8 with gradients          │
└──────────┬───────────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────┐
│ CODA (4 unique layers)           │
│   Layer 1: ShortConvBlock + FiLM │  ← Local refinement
│   Layer 2: GQABlock + FiLM       │  ← Global context
│   Layer 3: ShortConvBlock + FiLM │  ← Local refinement
│   Layer 4: GQABlock + FiLM       │  ← Global context
│            + TTT Sniper (3-step) │     THE FANGS
└──────────┬───────────────────────┘
           ▼
  Final RMSNorm → LM Head → Predictions
```

---

## The Physics

Three forces govern the loop:

```
PARCAE STABILITY (per iteration):
    h = A·h + B·e + CoreBlock(h, e)
    A = diag(-exp(log_A))  →  |eigenvalues| < 1  →  GUARANTEED CONVERGENCE
    The spring constant: pulls h back toward equilibrium.

MOMENTUM COHERENCE (across iterations):
    velocity = β·velocity + block_output
    h = h + velocity
    The mass: iterations that agree AMPLIFY, oscillations DAMPEN.

FiLM INTROSPECTION (at midpoint):
    context = fingerprint(h at iteration 4)
    Iterations 5-8 + Coda: modulated by context
    The mirror: the serpent WATCHES itself think and adjusts.

TOGETHER: DAMPED HARMONIC OSCILLATOR
    Spring constant = (1 - |A|)  →  restoring force toward equilibrium
    Mass = 1/(1 - β)             →  inertia of the trajectory
    The loop CONVERGES to a fixed point. Smoothly. Predictably.
```

---

## Core Components

### 1. Parcae Injection

```python
class ParcaeInjection(nn.Module):
    """SSM-style stable injection: h = A·h + B·e.
    A = -exp(log_A) guarantees |eigenvalues| < 1 (stability by construction).
    B = exp(log_B) controls input injection strength."""

    def __init__(self, d_model):
        super().__init__()
        self.log_A = nn.Parameter(torch.full((d_model,), -0.7))  # A ≈ -0.5
        self.log_B = nn.Parameter(torch.full((d_model,), -0.7))  # B ≈ 0.5

    def forward(self, h, input_embed):
        A = -torch.exp(self.log_A)     # ∈ (-1, 0) per dimension
        B = torch.exp(self.log_B)      # ∈ (0, ∞) per dimension
        return A * h + B * input_embed  # element-wise, FREE on Strix Halo
```

### 2. ShortConvBlock (shared in core loop, unique in Prelude/Coda)

```python
class ShortConvBlock(nn.Module):
    """GatedConv + SwiGLU with optional FiLM. LFM2.5-style.
    Compile-friendly: inlined residuals, plain PyTorch RMSNorm."""

    def __init__(self, d_model, ffn_inner, film_enabled=False):
        self.pre_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_model, kernel_size=3)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)
        self.film_enabled = film_enabled

    def forward(self, h, film_gamma=None, film_beta=None):
        h = h + self.out_proj(self.conv(self.pre_norm(h)))
        if self.film_enabled and film_gamma is not None:
            h = film_gamma.unsqueeze(1) * h + film_beta.unsqueeze(1)
        h = h + self.ffn(self.ffn_norm(h))
        return h
```

### 3. GQABlock (Prelude + Coda only, never inside loop)

```python
class GQABlock(nn.Module):
    """GQA + QK-Norm + hybrid_flash_sdpa + optional TTT sniper.
    n_heads=12, n_kv=8, head_dim=64 at d=768. LFM2 config."""

    def __init__(self, d_model, n_heads, n_kv, ffn_inner,
                 ttt_enabled=False, film_enabled=False):
        self.pre_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv, qk_norm=True)
        self.ffn_norm = RMSNorm(d_model)
        self.film_enabled = film_enabled
        if ttt_enabled:
            self.ffn = MultiStepTTTSwiGLU(d_model, ffn_inner, ttt_steps=3)
        else:
            self.ffn = SwiGLU(d_model, ffn_inner)
```

### 4. Loop FiLM Conditioner (fingerprints at iteration 4)

```python
class LoopFiLMConditioner(nn.Module):
    """Mid-loop introspection. Fingerprints h at iteration 4.
    Modulates iterations 5-8 (4 targets) + Coda layers (3 targets with FiLM) = 7 total.
    Zero-init: starts as identity."""

    def __init__(self, d_model, d_film=64, n_conditioned=7):
        self.context_proj = nn.Linear(d_model, d_film)
        self.gamma_projs = nn.ModuleList([
            nn.Linear(d_film, d_model, bias=True) for _ in range(n_conditioned)
        ])
        self.beta_projs = nn.ModuleList([
            nn.Linear(d_film, d_model, bias=True) for _ in range(n_conditioned)
        ])
        self._init_identity()  # zero-init all gamma/beta projections
```

### 5. Loop Momentum

```python
class LoopMomentum(nn.Module):
    def __init__(self, beta_init=0.5):
        super().__init__()
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(beta_init / (1 - beta_init)))
        )

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)  # ∈ (0, 1)
```

### 6. TTT Sniper (Multi-Step, last Coda layer only)

```python
class MultiStepTTTSwiGLU(nn.Module):
    """SwiGLU with 3 inner gradient steps on w_down per chunk.
    Applied ONLY at the last Coda GQA layer (layer 6/6 overall).
    3 steps × 1 layer ≈ cost of 1.5 single-step layers."""

    def __init__(self, d_model, ffn_inner, ttt_steps=3, ttt_lr=0.01, ttt_chunk=512):
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)
        self.ttt_proj = nn.Linear(d_model, d_model, bias=False)
        self.ttt_conv = nn.Conv1d(d_model, d_model, 5, padding=4, groups=d_model)
        self.ttt_steps = ttt_steps
        self.register_buffer('ttt_lr', torch.tensor(ttt_lr))
```

---

## Forward Pass

```python
class Ouroboros(nn.Module):
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)               # (B, T, 768)
        freqs_cis = self.freqs_cis[:T]

        # === PRELUDE (run once) ===
        h = self.prelude_conv(h)                          # ShortConvBlock
        h = self.prelude_attn(h, freqs_cis)               # GQABlock + QK-Norm
        input_embed = h.detach() if self.detach_embed else h

        # === CORE LOOP (3 ShortConv × 8 iterations) ===
        velocity = torch.zeros_like(h)
        beta = self.momentum.beta
        context = None
        film_idx = 0

        for t in range(self.n_loops):
            use_grad = not self.training or t >= self.n_no_grad

            with torch.no_grad() if not use_grad else nullcontext():
                # Parcae injection: stable state update
                h = self.injection(h, input_embed)

                # 3 shared ShortConv layers
                block_out = h
                for layer in self.core_block:
                    fg, fb = None, None
                    if context is not None and t >= self.film_start_iter:
                        fg, fb = self.film.get_gamma_beta(context, film_idx)
                        film_idx += 1
                    block_out = layer(block_out, fg, fb)

                # Momentum across iterations
                velocity = beta * velocity + (block_out - h)
                h = h + velocity

                # FiLM fingerprint at midpoint
                if t == self.film_iter:
                    context = self.film.compute_context(h)
                    film_idx = 0

        # === CODA (run once, 4 unique layers) ===
        for i, layer in enumerate(self.coda):
            fg, fb = self.film.get_gamma_beta(context, film_idx + i) if context else (None, None)
            if hasattr(layer, 'attn'):
                h = layer(h, freqs_cis, film_gamma=fg, film_beta=fb,
                          ttt_target=h if hasattr(layer.ffn, 'ttt_steps') else None)
            else:
                h = layer(h, fg, fb)

        return self.output(self.norm(h))
```

---

## Configuration

### OUROBOROS (Standard, d=768)

| Parameter | Value | Source |
|-----------|-------|--------|
| d_model | 768 | LFM2 scaled to ~175M |
| ffn_inner | 2816 (3.7×) | LFM2 ratio |
| n_heads | 12 | head_dim=64 |
| n_kv_heads | 8 | LFM2 |
| head_dim | 64 | LFM2 |
| QK-Norm | YES | LFM2 |
| conv_kernel | 3 | LFM2 |
| **Prelude** | 1 ShortConv + 1 GQA | 2 unique layers |
| **Core loop** | 3 ShortConv × 8 iters | 3 shared layers |
| **Coda** | 2 ShortConv + 2 GQA (last has TTT) | 4 unique layers |
| n_loops | 8 | 5 detached + 3 with grad |
| Parcae A init | log_A = -0.7 (A ≈ -0.5) | Moderate decay |
| Parcae B init | log_B = -0.7 (B ≈ 0.5) | Moderate injection |
| Momentum β init | 0.5 | Our standard |
| FiLM fingerprint | Iteration 4 | Mid-loop |
| FiLM targets | Iters 5-8 + 3 Coda layers = 7 | |
| TTT sniper | Last Coda GQA, 3 steps | Multi-step |
| d_film | 64 | AMADEUS |
| ttt_lr | 0.01 (fixed) | ARGUS |
| vocab_size | 50257 | tiktoken GPT-2 |

### OUROBOROS-MINI (d=512, L2-Cached Core)

| Parameter | Value | Diff from Standard |
|-----------|-------|-------------------|
| d_model | **512** | Smaller — core block fits L2! |
| ffn_inner | **1792 (3.5×)** | Scaled proportionally |
| n_heads | **8** | head_dim=64 |
| n_kv_heads | **4** | Scaled |
| Prelude | 1 ShortConv + 1 GQA | Same structure |
| Core loop | 3 ShortConv × 8 iters | **Core block ~1.8MB → fully L2-resident** |
| Coda | 2 ShortConv + 2 GQA + TTT | Same structure |
| All else | Same | Same loop/FiLM/momentum/TTT settings |

---

## Parameter Count

### OUROBOROS (d=768)

| Component | Params | Notes |
|-----------|--------|-------|
| Embedding (50257×768, tied) | 38.6M | Tied with LM head |
| Prelude: 1 ShortConv | ~9.1M | Unique |
| Prelude: 1 GQA | ~8.8M | Unique |
| **Core: 3 ShortConv (SHARED)** | **~27.2M** | **Shared across 8 iterations** |
| Parcae injection (A, B) | 1.5K | 768 + 768 |
| Loop momentum | 1 | log_beta |
| Coda: 2 ShortConv | ~18.2M | Unique |
| Coda: 2 GQA (1 with TTT) | ~18.8M | Unique (+1.2M TTT overhead) |
| FiLM conditioner | ~0.8M | 7 targets |
| **TOTAL UNIQUE** | **~121M** | |
| **Effective (8× loop)** | **~338M equivalent** | **2.8× param efficiency** |

### OUROBOROS-MINI (d=512)

| Component | Params | Notes |
|-----------|--------|-------|
| Embedding (50257×512, tied) | 25.7M | Tied |
| Prelude: 1 ShortConv + 1 GQA | ~8.8M | Unique |
| **Core: 3 ShortConv (SHARED)** | **~12.1M** | **~1.8MB fp16 — FITS L2 (6MB)!** |
| Coda: 2 ShortConv + 2 GQA + TTT | ~18.4M | Unique |
| FiLM + Parcae + momentum | ~0.5M | |
| **TOTAL UNIQUE** | **~65M** | |
| **Effective (8× loop)** | **~162M equivalent** | **2.5× param efficiency** |

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### L2 Cache Analysis

| Variant | Core Block Weight Size (fp16) | Fits L2 (6MB)? | Implication |
|---------|------------------------------|-----------------|-------------|
| **OUROBOROS (d=768)** | ~2.8MB | **YES with 3.2MB room** | Iterations 2-8 read from L2 at ~10× effective BW |
| **OUROBOROS-MINI (d=512)** | ~1.8MB | **YES with 4.2MB room** | Iters 2-8 from L2 + room for activations |

Both variants have L2-resident core blocks. The first iteration loads from DRAM (240 GB/s). Iterations 2-8 read from L2 cache (~2.4 TB/s effective). **The loop is essentially FREE from a memory-bandwidth perspective after iteration 1.**

### Throughput Estimate (OUROBOROS d=768)

| Component | Time | Notes |
|-----------|------|-------|
| Prelude (2 layers) | ~3ms | 1 ShortConv + 1 GQA (hybrid_flash) |
| Loop iter 1 (DRAM) | ~4ms | 3 ShortConv, cold weights |
| Loop iters 2-8 (L2) | 7 × ~1.5ms ≈ 10ms | L2-cached, ~2.5× faster per iter |
| Parcae + momentum | ~0ms | Element-wise, fused with loop |
| Coda (4 layers) | ~7ms | 2 ShortConv + 2 GQA + TTT |
| LM head + loss | ~2ms | Standard |
| **Forward** | **~26ms** | |
| **Forward + backward** | **~85ms** | Detached iters save ~62% of loop backward |

| Metric | OUROBOROS | OUROBOROS-MINI |
|--------|----------|----------------|
| Est. tok/s | **19-23K** | **25-32K** |
| vs AMADEUS (13.2K) | 1.4-1.7× | 1.9-2.4× |
| vs Tempest (22.3K) | 0.85-1.0× | 1.1-1.4× |
| 45-min tokens | 51-62M | 67-86M |
| Unique params | 121M | 65M |

**OUROBOROS-MINI at 25-32K tok/s would be the FASTEST model in the lab** while having only 65M unique parameters. If quality matches ~170M models, it's a breakthrough in parameter efficiency for edge deployment.

### Why Backward Is Cheap

Parcae's detached iteration trick: only 3 of 8 iterations track gradients.

| Training phase | Iterations | Backward cost |
|---------------|------------|---------------|
| Detached (t=1..5) | 5 × 3 ShortConv = 15 layer-equivalents | **ZERO** — no gradient |
| Gradient (t=6..8) | 3 × 3 ShortConv = 9 layer-equivalents | Standard backward |
| Coda | 4 layers | Standard backward |
| Prelude | 2 layers | Standard backward |
| **Total backward** | **15 layer-equivalents** | vs 30 effective forward |

Backward covers only 50% of the effective depth. Combined with the core block being only ~27M params (shared), this should reduce backward from ARGUS's 70.8% to closer to **55-60%**.

### Compile Compatibility

| Component | Compile-safe? | Notes |
|-----------|--------------|-------|
| Core ShortConv blocks | **YES** | Same block × 8 = uniform, Inductor loves this |
| Parcae injection | **YES** | Element-wise multiply + add |
| Momentum | **YES** | Element-wise |
| FiLM | **YES** | Element-wise |
| GQA + hybrid_flash_sdpa | **YES** | SDPA = single kernel |
| TTT sniper | **YES** | `_skip_autokernel=True`, einsum is compile-friendly |
| `torch.no_grad()` switch | **CAUTION** | Compile core block separately, call from Python loop |

**Strategy:** `torch.compile(self.core_block, mode="default")` — compile the 3 ShortConv layers as a unit. The Python loop (8 iterations) is not compiled, but each call to the compiled block fuses perfectly.

### Kernel Reuse

| Kernel | Speedup | Where |
|--------|---------|-------|
| fused_residual_add_rmsnorm | 6.6x | All layers + all loop iterations (autokernel) |
| silu_gate_mul | 1.6x | All SwiGLU instances (autokernel) |
| causal-conv1d | 10x | All GatedConv (Prelude + loop + Coda) |
| hybrid_flash_sdpa_attention | 8.9% | 3 GQA layers (1 Prelude + 2 Coda) |
| rotary_embedding | 3.7x | 3 GQA layers (autokernel) |
| cross_entropy | 1.8x | Loss |

---

## Training

### Protocol

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (fused=True) |
| LR | 8e-4 → 8e-5 cosine, 100 warmup |
| Batch | 16 × 256 = 4096 tokens/step |
| Precision | fp16 mixed (AMP + GradScaler) |
| Grad clip | 1.0 |
| Loop iterations | 8 (5 detached + 3 with grad) |
| Dataset | BabyLM (~16.5M tokens) |
| Epochs | 2 |

### Optimizer Groups

| Pattern | LR mult | WD | Notes |
|---------|---------|-----|-------|
| `*log_A*`, `*log_B*` | 0.1× | 0 | Parcae stability — slow learning |
| `*log_beta*` | 1× | 0 | Momentum coefficient |
| `*film*` | 1× | 0.1 | FiLM conditioner |
| `*ttt*`, `*w_target*` | 1× | 0.1 | TTT sniper |
| `*norm*`, `*bias` | 1× | 0 | Standard |
| Everything else | 1× | 0.1 | Backbone |

### Curriculum for Loop Depth

```
Steps 0-20%:     2 loops (0 detached + 2 with grad)
Steps 20-40%:    4 loops (1 detached + 3 with grad)
Steps 40-60%:    6 loops (3 detached + 3 with grad)
Steps 60-100%:   8 loops (5 detached + 3 with grad)
```

### Zero-Init Bootstrap

- **FiLM:** gamma/beta projections zero-init → identity transform
- **TTT:** ttt_conv zero-init → no fast weight contribution
- **Parcae A, B:** initialized to A≈-0.5, B≈0.5 → moderate decay + moderate injection
- **Momentum β:** initialized to 0.5 → moderate inertia

Model starts as a simple 9-layer network (Prelude + 1 loop + Coda) with standard FFN. Loop depth, FiLM, and TTT activate organically during training.

---

## Ablation Plan

| # | Experiment | What It Tests |
|---|-----------|--------------|
| 1 | OUROBOROS vs ARGUS-PRIME B3 | Looped vs non-looped LFM2.5 at same effective capacity |
| 2 | OUROBOROS vs AMADEUS | 121M looped vs 157M standard — param efficiency |
| 3 | OUROBOROS vs Tempest | Speed + quality comparison vs compile-optimized Griffin |
| 4 | 8 loops vs 4 vs 2 | How much looping is needed? |
| 5 | With momentum vs without | Does loop momentum improve convergence? |
| 6 | With FiLM mid-loop vs without | Does mid-loop introspection help? |
| 7 | With TTT sniper vs without | Does adaptive Coda FFN help? |
| 8 | OUROBOROS vs OUROBOROS-MINI | d=768 vs d=512 quality/speed tradeoff |
| 9 | Parcae A learned vs fixed | Does the model learn non-trivial decay rates? |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Loop doesn't converge | LOW | A-matrix constraint is mathematically guaranteed. Curriculum starts from 2 loops. If still unstable → something else is wrong. |
| 121M unique params not enough quality | MEDIUM | Effective capacity is 338M. If insufficient, add 1 more ShortConv to core (4×8=32 effective layers, ~155M unique). |
| L2 cache doesn't help as estimated | LOW | Even without L2, 8 loops of 3 ShortConv are cheap (no attention). L2 is a bonus, not a dependency. |
| FiLM mid-loop is useless | LOW | Zero-init. If useless, gamma=1, beta=0 throughout. No harm. Ablation #6 tests. |
| Compile breaks on no_grad switch | MEDIUM | Compile core block independently. Python loop handles the iteration/gradient switch. |
| OUROBOROS-MINI (d=512) too small for quality | MEDIUM | d=512 is aggressive. If quality suffers, MINI is a throughput benchmark only. Standard (d=768) is the quality target. |
| Parcae A/B learning interferes with momentum | LOW | Different purposes: A/B control injection magnitude, momentum controls directional coherence. They're orthogonal. If they interfere → fix momentum first (it's the newer addition). |

---

## Success Criteria

1. **OUROBOROS val loss < 2.98** — beat Tempest at 121M unique params (vs 176.8M)
2. **OUROBOROS val loss < 2.90** — beat AMADEUS at 121M unique params (vs 157.7M)
3. **Throughput > 18K tok/s** (OUROBOROS) and **> 25K tok/s** (MINI)
4. **Parcae A converges** to non-trivial values (loop stability IS learned, not just default)
5. **Momentum β converges to non-zero** (loop coherence IS needed)
6. **Quality improves with more loops** (4→8 shows measurable gain)
7. **121M OUROBOROS ≈ 175M non-looped** quality (param efficiency confirmed)
8. **OUROBOROS-MINI at 65M** achieves quality ≈ 120M non-looped (L2 exploitation confirmed)

---

## Implementation Roadmap

1. Create `models/ouroboros.py` with shared config for standard + MINI
2. Implement `ParcaeInjection` (A-matrix constraint, B injection)
3. Implement `LoopFiLMConditioner` (mid-loop fingerprint at iteration 4)
4. Implement `LoopMomentum` (velocity across iterations)
5. Reuse `ShortConvBlock` and `GQABlock` patterns from ARGUS-PRIME
6. Reuse `MultiStepTTTSwiGLU` from ARGUS-PRIME (TTT sniper)
7. Wire the 3-zone forward pass: Prelude → Loop → Coda
8. Implement loop curriculum scheduler
9. Implement detached/gradient iteration split
10. Smoke test BOTH variants (d=128, 4 layers, 200 steps) — MANDATORY
11. Compile test (compile core block independently)
12. Full training on BabyLM
13. Run ablation battery

---

## References

- Parcae (Together AI, 2026) — stable looped transformers, SSM-style A-matrix constraint
- LFM2/LFM2.5 (Liquid AI, 2025/2026) — ShortConv + sparse GQA, 3:1 ratio, target to beat
- AMADEUS (our lab) — FiLM conditioning, val 2.90 quality champion
- TEMPEST (our lab) — compile-optimized Griffin, 22.3K tok/s, momentum
- LAZARUS (our lab) — TTT fast weights, damped accumulation
- ARGUS-PRIME (our lab) — LFM2.5-scaled ablation framework, TTT sniper concept
- RESONANT-LOOP (our lab) — shared block + ACT halting, val 3.42, throughput champion
- PRE_OUROBOROS (our lab) — full design decision log with alternatives
