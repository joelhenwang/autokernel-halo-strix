---
title: "NEXUS"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %nexus]
---

# NEXUS

**The Split-Mind Griffin: Compile-Optimized Griffin + TTT Fast Weights + FiLM + Residual Momentum**

*A nexus is where different pathways CONVERGE. Layer 5's subconscious adaptation meets layer 8's context fingerprint meets layer 13's conscious specialization. Three forms of intelligence, connected at the nexus of the model's depth.*

*Four ingredients stolen from four creations: Griffin's SPEED (Tempest, 22K tok/s), FiLM's AWARENESS (AMADEUS, val 2.90), TTT's DREAMS (LAZARUS), and Momentum's INERTIA (Tempest/Prometheus). Each proven individually. Never combined.*

## Hypothesis

The compile-optimized Griffin (Tempest) runs at 22K tok/s but loses to AMADEUS on quality (2.98 vs 2.90). AMADEUS has two things Tempest doesn't: (1) Mamba-3's data-dependent state transitions, and (2) FiLM conditioning. We can't bring Mamba-3 to Griffin without losing compile performance. But we CAN bring FiLM. And we can add something AMADEUS never had: **TTT fast weights that reshape the FFN to the document**.

**The split-mind design:** Place TTT at layers 5 and 13, bracketing the FiLM fingerprint at layer 8.
- **Layer 5 (subconscious):** Adapts w_down BEFORE the model knows what the document is about. Catches low-level patterns — syntax, register, domain vocabulary.
- **Layer 8 (fingerprint):** FiLM computes context fingerprint from the TTT-enriched hidden state. RICHER signal because layer 5 already injected document-level patterns.
- **Layer 13 (conscious):** Adapts w_down WITH full FiLM context. Specializes high-level projections — semantic roles, discourse patterns, prediction targets.

**The bet:** FiLM closes the context-awareness gap. TTT closes the adaptivity gap. Together on Griffin's compile-optimized chassis, we get AMADEUS quality at Tempest speed.

---

## What We Stole (and from whom)

| Ingredient | Source | Cost | Proven Result |
|-----------|--------|------|--------------|
| Compile-optimized Griffin | TEMPEST | Chassis | 22K tok/s (1.73x from compile) |
| Residual Momentum | TEMPEST/PROMETHEUS | FREE (element-wise) | Depth-wise inertia, proven stable |
| FiLM Conditioning | AMADEUS | ~1.1M params, ~0 FLOPs | Part of val 2.90 recipe |
| TTT Fast Weights (2 layers) | LAZARUS/LazarusLite | ~4% throughput, ~1GB mem | Mechanically proven, L2-resident ΔW |
| Fused Griffin projections | TEMPEST | Already in | 3% GEMM savings |
| Vectorized chunked scan | TEMPEST | Already in | Compile-safe, +17% throughput |
| Strategic layer placement | PROMETHEUS | Design principle | 1/3 and 2/3 marks for maximum coverage |
| Zero-init bootstrap | LAZARUS | Design principle | Model starts as vanilla, activates organically |

**What we deliberately LEFT behind:**
- Mamba-3 SISO (breaks compile) — from AMADEUS
- PLE Path A (26M params, 3% throughput cost) — from VIRTUOSO
- Conductor network (negligible benefit) — from MAESTRO-PRIMA
- Heterogeneous decay spectrum (didn't help quality) — from SPECTRAL-HYDRA
- Dual-path with d<512 (autokernel breaks it) — from DUAL-CORTEX/OBSIDIAN

---

## Architecture

```
Tokens → Embedding (d=1024, tied LM head, vocab=50257)
  │
  ├─ V̂ = CausalConv1D(embeddings, k=5)   ← NTP target, computed ONCE, zero-init
  │
  velocity = 0  (residual momentum state)
  ΔW_5 = 0, ΔW_13 = 0  (fast weight deltas, each 5.2MB, L2-resident)
  │
  → 16 Compile-Optimized Griffin Blocks:
  │
  │   Layers 1-4:   Griffin + Conv + Momentum + SwiGLU         [FOUNDATION]
  │   ★ Layer 5:    Griffin + Conv + Momentum + LIVING SwiGLU   [SUBCONSCIOUS TTT]
  │   Layers 6-7:   Griffin + Conv + Momentum + SwiGLU         [BRIDGE]
  │   ★ Layer 8:    Griffin + Conv + Momentum + SwiGLU         [FiLM FINGERPRINT COMPUTED HERE]
  │   Layers 9-12:  Griffin + Conv + Momentum + SwiGLU + FiLM  [MODULATED]
  │   ★ Layer 13:   Griffin + Conv + Momentum + LIVING SwiGLU + FiLM  [CONSCIOUS TTT]
  │   Layers 14-16: Griffin + Conv + Momentum + SwiGLU + FiLM  [FINAL]
  │
  → Final RMSNorm → LM Head
```

---

## Core Components

### 1. Griffin Block (Compile-Optimized, from Tempest)

Identical to Tempest's FusedGriffinBlockPattern:
- Inlined momentum residual (avoids module boundary for compile)
- Plain PyTorch RMSNorm (Inductor fuses element-wise ops)
- Vectorized chunked scan (no Python loops, compile-safe)
- Fused w_aiv projection (3 Griffin projections → single GEMM)
- `_use_hip_backward()` guard skips HIP backward during compile tracing

```python
class NexusBlock(nn.Module):
    """Griffin block with optional TTT-enhanced FFN and optional FiLM."""

    def __init__(self, cfg, ttt_enabled=False, film_enabled=False):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.d_model)
        self.conv = GatedConv(cfg.d_model, cfg.d_conv, cfg.conv_kernel)
        self.griffin = GriffinRecurrence(cfg.d_model, cfg.d_griffin)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ttt_enabled = ttt_enabled
        self.film_enabled = film_enabled

        if ttt_enabled:
            self.ffn = LivingSwiGLU(cfg.d_model, cfg.ffn_inner)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)

    def forward(self, h, velocity, beta, v_hat=None, delta_W=None,
                film_gamma=None, film_beta=None):
        # Mixer: parallel conv + Griffin
        h_norm = self.pre_norm(h)
        conv_out = self.conv(h_norm)                          # (B, T, 640)
        grif_out = self.griffin(h_norm)                        # (B, T, 384)
        mixed = self.out_proj(torch.cat([conv_out, grif_out], dim=-1))

        # Momentum residual on mixer
        velocity = beta * velocity + mixed
        h = h + velocity

        # Optional FiLM modulation BEFORE FFN norm
        if self.film_enabled and film_gamma is not None:
            h = film_gamma.unsqueeze(1) * h + film_beta.unsqueeze(1)

        # FFN (standard or TTT-enhanced)
        h_ffn = self.ffn_norm(h)
        if self.ttt_enabled and delta_W is not None:
            ffn_out, delta_W = self.ffn(h_ffn, v_hat, delta_W)
        else:
            ffn_out = self.ffn(h_ffn)
        h = h + ffn_out

        return h, velocity, delta_W
```

### 2. Griffin Recurrence (Element-Wise, Compile-Safe)

```python
class GriffinRecurrence(nn.Module):
    """h = a*h + sqrt(1-a^2) * (i*v) — bounded, element-wise."""

    def __init__(self, d_model, d_griffin):
        super().__init__()
        self.d_griffin = d_griffin
        # Fused a+i+v projection (single GEMM, 3% savings)
        self.w_aiv = nn.Linear(d_model, 3 * d_griffin, bias=False)
        self.out_proj = nn.Linear(d_griffin, d_griffin, bias=False)
        # Decay bias: uniform init (heterogeneous spectrum didn't help)
        self.decay_bias = nn.Parameter(torch.zeros(d_griffin))

    def forward(self, x_norm):
        a_raw, i_raw, v = self.w_aiv(x_norm).chunk(3, dim=-1)
        a = torch.sigmoid(a_raw + self.decay_bias)           # decay gate ∈ (0,1)
        i = torch.sigmoid(i_raw)                               # input gate
        coupling = torch.sqrt(1.0 - a * a + 1e-6)            # Griffin coupling

        # Vectorized chunked scan (compile-safe, no Python loops)
        h = _chunked_scan(a, coupling * i * v)                 # (B, T, d_griffin)
        return self.out_proj(h)
```

### 3. LivingSwiGLU — TTT Fast Weight FFN (2 instances: layers 5, 13)

From LAZARUS/LazarusLite. Zero-init bootstrap — starts as standard SwiGLU.

```python
class LivingSwiGLU(nn.Module):
    """SwiGLU where w_down is augmented with a fast weight that adapts per-sequence.

    Standard:  out = w_down(silu(gate) * up)
    Living:    out = w_down(z) + z @ ΔW
               ΔW = γ·ΔW + η·V̂ᵀ·x      (damped accumulation)

    ΔW shape: (ffn_inner, d_model) = (2560, 1024) = 5.2MB fp16 → fits in L2.
    """

    def __init__(self, d_model, ffn_inner):
        super().__init__()
        self.d_model = d_model
        self.ffn_inner = ffn_inner
        self._skip_autokernel = True   # prevent FusedSwiGLU pattern from replacing
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)  # frozen "slow weight"
        self.w_target = nn.Linear(d_model, ffn_inner, bias=False)
        self.log_gamma = nn.Parameter(torch.tensor(0.0))     # sigmoid(0) = 0.5
        self.log_eta = nn.Parameter(torch.tensor(-4.0))      # softplus(-4) ≈ 0.018

    @property
    def gamma(self):
        return torch.sigmoid(self.log_gamma)

    @property
    def eta(self):
        return F.softplus(self.log_eta)

    def forward(self, x, v_hat=None, delta_W=None):
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        z = F.silu(gate) * up                                 # (B, T, ffn_inner)
        out = self.w_down(z)                                   # (B, T, d_model)

        if delta_W is not None and v_hat is not None:
            # Fast weight contribution: z @ ΔW
            out = out + z @ delta_W                            # ΔW is (ffn_inner, d_model)

            # Compute update: V̂ projected to FFN space, outer product with input
            v_proj = self.w_target(v_hat)                      # (B, T, ffn_inner)
            update = torch.einsum('btf,btd->fd', v_proj, x) / x.shape[0]

            # Damped accumulation: dreams fade (γ < 1)
            delta_W = self.gamma * delta_W + self.eta * update

        return out, delta_W
```

### 4. NTP Target Encoder (from LAZARUS)

CausalConv1D on embeddings. Zero-init: starts as identity (no TTT contribution). TTT activates organically as conv weights learn useful targets.

```python
class NTPTargetEncoder(nn.Module):
    """V̂ = CausalConv1D(embeddings, k=5). Computed ONCE per forward pass."""

    def __init__(self, d_model, kernel_size=5):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv_weight = nn.Parameter(torch.zeros(d_model, kernel_size))  # zero-init!
        self.conv_bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, embeddings):
        """(B, T, d_model) → (B, T, d_model)"""
        if _HAS_CAUSAL_CONV1D:
            return causal_conv1d_fn(
                embeddings.transpose(1, 2),
                self.conv_weight, self.conv_bias,
            ).transpose(1, 2)
        # Fallback: manual causal conv
        x = embeddings.transpose(1, 2)                        # (B, D, T)
        x = F.pad(x, (self.kernel_size - 1, 0))              # left-pad
        x = F.conv1d(x, self.conv_weight.unsqueeze(1),
                     self.conv_bias, groups=self.d_model)
        return x.transpose(1, 2)
```

### 5. FiLM Conditioner (from AMADEUS)

Fingerprint at layer 8. Modulates layers 9-16. Zero-init: starts as identity (gamma=1, beta=0).

```python
class FiLMConditioner(nn.Module):
    def __init__(self, d_model, d_film, n_conditioned_layers):
        super().__init__()
        self.context_proj = nn.Linear(d_model, d_film, bias=True)
        self.gamma_projs = nn.ModuleList([
            nn.Linear(d_film, d_model, bias=True)
            for _ in range(n_conditioned_layers)
        ])
        self.beta_projs = nn.ModuleList([
            nn.Linear(d_film, d_model, bias=True)
            for _ in range(n_conditioned_layers)
        ])
        self._init_identity()

    def _init_identity(self):
        for proj in self.gamma_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        for proj in self.beta_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def compute_context(self, h):
        return self.context_proj(h.mean(dim=1))                # (B, d_film)

    def get_gamma_beta(self, context, layer_idx):
        gamma = self.gamma_projs[layer_idx](context) + 1.0    # starts at 1
        beta = self.beta_projs[layer_idx](context)             # starts at 0
        return gamma, beta
```

### 6. Residual Momentum (from TEMPEST)

Single learned scalar β, applied to mixer output only.

```python
class MomentumResidual(nn.Module):
    def __init__(self, beta_init=0.5):
        super().__init__()
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(beta_init / (1 - beta_init)))
        )

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)
```

---

## Full Forward Pass

```python
class Nexus(nn.Module):
    def __init__(self, ...):
        # ... (see Configuration below)
        self.ttt_layers = {4, 12}     # 0-indexed: layers 5 and 13 (1-indexed)
        self.film_start = 8           # 0-indexed: FiLM fingerprint at layer 9 (1-indexed)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)                     # (B, T, 1024)

        # NTP targets: computed ONCE from raw embeddings
        v_hat = self.target_encoder(h)                         # (B, T, 1024)

        # State initialization
        velocity = torch.zeros_like(h)
        delta_Ws = {
            idx: torch.zeros(self.ffn_inner, self.d_model,
                             device=h.device, dtype=torch.float32)
            for idx in self.ttt_layers
        }

        beta = self.momentum.beta
        context = None

        for i, layer in enumerate(self.layers):
            # FiLM fingerprint at layer 8 (0-indexed)
            if i == self.film_start:
                context = self.film.compute_context(h)

            # FiLM gamma/beta for modulated layers
            film_g, film_b = None, None
            if i >= self.film_start and context is not None:
                film_g, film_b = self.film.get_gamma_beta(context, i - self.film_start)

            # TTT routing
            dW = delta_Ws.get(i)
            v = v_hat if i in self.ttt_layers else None

            # Forward through block
            h, velocity, new_dW = layer(h, velocity, beta, v, dW, film_g, film_b)

            # Update fast weight delta
            if i in self.ttt_layers and new_dW is not None:
                delta_Ws[i] = new_dW

        h = self.norm(h)
        return self.output(h)
```

---

## Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| d_model | 1024 | Tempest (exact match) |
| n_layers | 16 | Tempest (exact match) |
| d_conv | 640 | Tempest (exact match) |
| d_griffin | 384 | Tempest (exact match) |
| ffn_inner | 2560 | Tempest (exact match) |
| conv_kernel | 3 | Tempest (exact match) |
| momentum β init | 0.5 | Tempest |
| film_start | 8 (0-indexed) | AMADEUS |
| d_film | 64 | AMADEUS |
| n_film_layers | 8 (layers 9-16) | AMADEUS |
| ttt_layers | {4, 12} (0-indexed) | NEW — bracket FiLM |
| TTT γ init | 0.5 (learned, sigmoid) | LAZARUS |
| TTT η init | 0.018 (learned, softplus) | LazarusLite (conservative start) |
| TTT Conv1D kernel | 5 (causal, zero-init) | LAZARUS |
| vocab_size | 50257 | Standard (tiktoken GPT-2) |
| max_seq_len | 1024 | Standard |

---

## Parameter Count

| Component | Params | Notes |
|-----------|--------|-------|
| Embedding (50257×1024, tied) | 51.5M | Weight-tied with LM head |
| 16 Griffin blocks (conv + griffin + out_proj + ffn_norm + ffn) | ~125.3M | Same as Tempest |
| FiLM conditioner (context_proj + 8×gamma + 8×beta) | 1.1M | Layers 9-16 |
| 2× LivingSwiGLU W_target (1024→2560) | 5.24M | Layers 5, 13 |
| 1× NTP target Conv1D (k=5, d=1024) | 5.1K | Zero-init |
| 2× (log_gamma + log_eta) | 4 | Learned TTT dynamics |
| 1× momentum log_beta | 1 | Learned inertia |
| Final RMSNorm | 1K | — |
| **TOTAL** | **~183.2M** | +6.3M over Tempest 176.8M |

Safely under 250M. A/B comparison vs Tempest: only 3.6% more parameters from TTT+FiLM.

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### The L2 Cache Story

| Asset | Size (fp16) | L2 fit? | Access Pattern |
|-------|------------|---------|---------------|
| ΔW layer 5 | 5.24 MB | YES | Read+update during layer 5, dormant layers 6-16 |
| ΔW layer 13 | 5.24 MB | YES | Read+update during layer 13, dormant layers 1-12, 14-16 |
| FiLM context (B×64) | 4 KB | YES | Computed once at layer 8, read 8 times |

Only ONE ΔW is "hot" at a time. Between layers 6-12: no ΔW in play, full L2 available for normal compute. The two TTT layers are PERFECTLY spaced to avoid L2 contention.

### Throughput Estimate

| Factor | tok/s Impact | Cumulative |
|--------|-------------|-----------|
| Baseline: compile-optimized Tempest ~170M | 22,358 | 22,358 |
| FiLM overhead (8 small Linear + 1 mean pool) | -1% | ~22,130 |
| TTT layer 5 (1× apply GEMM + 1× update einsum) | -2% | ~21,700 |
| TTT layer 13 (1× apply GEMM + 1× update einsum) | -2% | ~21,260 |
| NTP target encoder (1× causal conv1d) | ~0% | ~21,260 |
| **Estimated total** | | **~20-21K tok/s** |

### Competition Table

| Architecture | Params | tok/s | Val Loss | Notes |
|-------------|--------|-------|----------|-------|
| LlamaModel 124M | 124.7M | 47,864 | — | AK+compile, transformer |
| **NEXUS** | **~183M** | **~20-21K** | **Target: <2.90** | AK+compile |
| Tempest 176.8M | 176.8M | 22,358 | 2.98 | AK+compile, compile-optimized |
| Amadeus 157.7M | 157.7M | 13,203 | **2.90** | AK only (compile incompatible) |
| LazarusLite 160.2M | 160.2M | 12,150 | TBD | AK only |
| Prometheus 174.3M | 174.3M | 13,066 | 3.00 | AK only |

**NEXUS at 20K tok/s = 1.52x faster than AMADEUS.** If quality matches 2.90, this is the new champion. If quality BEATS 2.90, we've found the optimal point in the speed-quality tradeoff.

### Kernel Reuse

| Kernel | Speedup | Where |
|--------|---------|-------|
| FusedGriffinBlockPattern | 1.73x compile | All 16 blocks |
| fused_residual_add_rmsnorm | 6.6x | All 16 layers (autokernel) |
| silu_gate_mul | 1.6x | 14 standard SwiGLU layers (autokernel) |
| causal-conv1d | 10x | 16 GatedConv + 1 NTP encoder |
| cross_entropy / chunked_linear_CE | 1.8x / mem savings | Loss |
| Griffin vectorized chunked scan | compile-safe | All 16 Griffin recurrences |

**Note:** The 2 LivingSwiGLU layers have `_skip_autokernel = True` — autokernel won't replace them with FusedSwiGLU, preserving the TTT fast weight path.

### Compile Compatibility

| Component | Compile-safe? | Notes |
|-----------|--------------|-------|
| Griffin vectorized scan | YES | No Python loops, cumulative products |
| RMSNorm | YES | Inductor fuses element-wise |
| SwiGLU (14 standard layers) | YES | Inductor fuses silu+mul+matmul |
| LivingSwiGLU (2 TTT layers) | YES* | einsum + scalar ops are compile-friendly. ΔW passed explicitly, no global mutation |
| FiLM modulation | YES | Element-wise multiply + add |
| Momentum | YES | Scalar multiply + add |
| NTP target encoder | YES | Single conv1d, precomputed |

*The main concern is ΔW as a mutable tensor across the layer loop. Since we pass it explicitly through function arguments (not as a global or buffer), torch.compile should trace through cleanly. The `_skip_autokernel` flag prevents pattern matching interference.

---

## Training

Single-phase. No freezing, no phase transitions. Zero-init bootstrap ensures the model starts as vanilla Tempest and organically activates FiLM + TTT as training progresses.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW (fused=True) | Standard |
| Base LR | 8e-4 → 8e-5 cosine | 100-step warmup |
| Batch size | 16 × 256 = 4096 tokens/step | L2 sweet spot |
| Precision | fp16 mixed (AMP) | fp32 for scan accumulation + ΔW accumulation |
| Grad clip | 1.0 | Standard |
| GradScaler | YES (fp16) | Required |

### Optimizer Groups

| Parameter pattern | LR mult | Weight Decay | Notes |
|------------------|---------|-------------|-------|
| `*w_target*` | 1× | 0.1 | TTT target projection |
| `*log_gamma*`, `*log_eta*` | 1× | 0 | TTT dynamics (no decay — let them find equilibrium) |
| `*log_beta*` (momentum) | 1× | 0 | Momentum coefficient |
| `*film*`, `*context_proj*` | 1× | 0.1 | FiLM conditioner |
| `*target_encoder*` | 1× | 0 | NTP Conv1D (zero-init, needs freedom to learn) |
| `*norm*`, `*bias` | 1× | 0 | Standard: norms/biases no WD |
| Everything else | 1× | 0.1 | Backbone weights |

### Zero-Init Bootstrap

Both FiLM and TTT start as identity transforms:
- **FiLM:** gamma_proj weights = 0, bias = 0 → gamma = 1, beta = 0 → identity
- **TTT:** NTP conv weights = 0 → V̂ = 0 → update = 0 → ΔW stays at 0 → standard SwiGLU

The model trains as vanilla Tempest until gradient descent discovers useful FiLM/TTT patterns. No warmup phases, no freezing, no curriculum. The architecture bootstraps itself.

---

## Ablation Plan

| # | Experiment | What It Tests | Expected Outcome |
|---|-----------|--------------|-----------------|
| 1 | **NEXUS vs Tempest** | Does TTT + FiLM improve quality? | NEXUS < 2.98 (Tempest baseline) |
| 2 | **NEXUS vs Tempest+FiLM (no TTT)** | Does TTT add value beyond FiLM? | TTT should contribute >0.5% loss reduction |
| 3 | **NEXUS vs Griffin+TTT (no FiLM)** | Does FiLM add value beyond TTT? | FiLM should contribute (it was part of 2.90 recipe) |
| 4 | **TTT at {5,13} vs {4,7} vs {10,14}** | Does straddling FiLM matter? | Straddle should be best (maximum depth coverage) |
| 5 | **NEXUS vs AMADEUS at equal wall-clock** | The REAL question | NEXUS sees 1.5x more tokens — quality must match |

### What Each Ablation Costs

| Ablation | Modifications needed | Extra training runs |
|----------|---------------------|-------------------|
| #1 | Already have Tempest baseline | 0 (compare to existing) |
| #2 | Disable TTT (set ttt_layers={}) | 1 short run |
| #3 | Disable FiLM (set film_start=99) | 1 short run |
| #4 | Change ttt_layers set | 2 short runs |
| #5 | Already have AMADEUS baseline | 0 (compare to existing) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| FiLM + TTT redundant at layer 13 | MEDIUM | If redundant, TTT η→0 naturally (learned). Ablation #2 quantifies. If TTT adds <0.5% over FiLM, drop it → Tempest+FiLM at full 22K tok/s. |
| Compile breaks with ΔW state passing | MEDIUM | Explicit argument passing (no globals). `_skip_autokernel=True` prevents interference. Test compile on smoke model BEFORE full training. |
| BabyLM too short for TTT to shine | LOW | Zero-init bootstrap → if TTT doesn't help, ΔW ≈ 0 throughout. No harm, model degrades gracefully to Tempest+FiLM. GPT-training-small has longer documents — TTT may show more benefit there. |
| 5% throughput drop not worth it | LOW | FiLM alone is ~1% overhead. If TTT doesn't justify its 4%, disable it (ttt_layers={}) and keep Tempest+FiLM at ~22K tok/s. Graceful degradation by design. |
| Griffin quality ceiling at 170M | MEDIUM | If Griffin+FiLM+TTT can't reach 2.90 at 170M, try at full 244M (Tempest's original config). More params = more FFN capacity for TTT to exploit. |

---

## Success Criteria

1. **NEXUS val loss < 2.90** — beat AMADEUS (the quality champion)
2. **NEXUS val loss < 2.98** — beat Tempest (minimum bar, proves TTT+FiLM help)
3. **Throughput > 18K tok/s** with AK + compile (faster than AMADEUS 13.2K)
4. **Ablation #2:** TTT contributes measurable improvement over FiLM alone
5. **TTT η converges to non-zero** in BOTH layers (both layers ARE adapting)
6. **Layer 5 and layer 13 learn DIFFERENT γ/η** (different adaptation profiles for subconscious vs conscious)
7. **FiLM fingerprint quality:** layer 8 hidden state norm is richer (higher variance) than Tempest layer 8 — evidence that layer 5 TTT enriched it

---

## Implementation Roadmap

1. **Start from `models/tempest.py`** — copy as `models/nexus.py`
2. Import `LivingSwiGLU` and `NTPTargetEncoder` from `models/lazarus_lite.py`
3. Import `FiLMConditioner` from `models/amadeus.py`
4. Create `NexusBlock` with optional ttt_enabled and film_enabled flags
5. Wire forward pass: velocity + delta_Ws + FiLM context routing
6. Verify param count ~183M
7. **Smoke test** (d=128, 4 layers, 200 steps) — MANDATORY
8. Compile test: `torch.compile(model, mode="default")` on smoke config
9. Full training on BabyLM with autokernel + compile
10. Run ablation battery (#1-#5)

### External Kernel Integration

- **causal-conv1d** (10x): 16 GatedConv layers + 1 NTP target encoder
- **Griffin vectorized chunked scan**: all 16 layers (compile-safe)
- **fused_residual_add_rmsnorm** (6.6x): all 16 layers via autokernel.optimize
- **silu_gate_mul** (1.6x): 14 standard SwiGLU layers via autokernel.optimize
- **chunked_linear_cross_entropy**: memory savings for LM head

---

## The Physics of NEXUS

Three forms of intelligence, converging at one point in the architecture:

```
MOMENTUM (layers 1-16):    velocity = β·velocity + mixer_out
                            The signal has MASS through depth.

FiLM (layers 9-16):         h = γ(ctx)·h + β(ctx)
                            The representation is MODULATED by sequence context.

TTT (layers 5, 13):         ΔW = γ·ΔW + η·V̂ᵀ·x
                            The FUNCTION itself adapts to the document.
```

Layer 5's ΔW captures what the model sees BEFORE understanding. Layer 8's fingerprint captures the UNDERSTANDING. Layer 13's ΔW adapts WITH understanding. The split mind: subconscious + conscious, bracketing the moment of comprehension.

**NEXUS: where speed, adaptation, and awareness converge.**

---

## References

- TEMPEST (our lab) — compile-optimized Griffin chassis, 22K tok/s, val 2.98
- AMADEUS (our lab) — FiLM conditioning, val 2.90 quality champion
- LAZARUS/LazarusLite (our lab) — TTT fast weight mechanism, proven on gfx1151
- PROMETHEUS (our lab) — strategic layer placement (1/3, 2/3 marks)
- VIRTUOSO (our lab) — PLE ablation (left behind: too costly for this build)
- Feng et al., "In-Place Test-Time Training" (arXiv 2604.06169, 2026)
- Compile-Optimized Griffin spec: `docs/superpowers/specs/2026-04-12-compile-optimized-griffin-design.md`
