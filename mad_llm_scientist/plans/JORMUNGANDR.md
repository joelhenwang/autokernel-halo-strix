# JORMUNGANDR

**The World Serpent: Parcae-Stable Looped ShortConv + MuonAdamW + Poisson Depth + Value Embeddings + CompreSSM Monitoring + Staged Component Activation**

*In Norse mythology, Jormungandr is the Ouroboros grown to its final form — the World Serpent, so vast it encircles all of Midgard and grasps its own tail. Where Ouroboros was the concept, Jormungandr is the realization: every lesson learned, every instability tamed, every shortcut eliminated. The serpent that ate itself and emerged stronger.*

*JORMUNGANDR is OUROBOROS after digesting our assessment. Every recommendation applied, every risk mitigated, every optimization from Parcae's actual codebase integrated. Not a redesign — a maturation.*

**Predecessor:** `plans/OUROBOROS.md` (the original design)
**Design decisions:** `plans/PRE_OUROBOROS.md` (fork-by-fork log)
**Parcae assessment:** `plans/PARCAE.md` (research analysis)
**Assessment that produced this plan:** Claude's full cross-reference of Parcae source code, training script, blog post, and our knowledge base

---

## What Changed From OUROBOROS

| Component | OUROBOROS | JORMUNGANDR | Why |
|-----------|----------|-------------|-----|
| Optimizer | AdamW | **MuonAdamW** | Parcae's actual recipe; our Muon results: 2x token efficiency |
| Momentum | Built-in from start | **Disabled by default, staged activation** | Unproven + risks breaking Parcae stability guarantee |
| TTT | Multi-step (3) from start | **Single-step default, multi-step upgrade** | Multi-step caused NaN from scratch in ARGUS-PRIME |
| Loop depth sampling | Fixed curriculum only | **Poisson-sampled + curriculum** | Parcae's key regularization technique, prevents overfitting to fixed depth |
| Value Embeddings | Not included | **Added to last Coda GQA (d_ve=64, projected)** | Parcae technique; reduced from 3 layers to 1 to stay within param budget |
| FiLM fingerprint | Iteration 4 (detached) | **Iteration 6 (first gradient iteration)** | Ensures fingerprint receives direct gradients |
| State monitoring | Basic norms | **CompreSSM HSV monitoring on loop state** | Quantifies which loop dimensions are actually used |
| Throughput estimates | 19-23K tok/s (optimistic) | **15-19K tok/s (conservative)** | Accounts for L2 contention with optimizer |
| Dataset plan | BabyLM only | **BabyLM smoke, GPT-training-small full, WikiText-103 scale** | Loop scaling needs more data to manifest |
| Component activation | All active from step 0 (zero-init) | **Staged: bare loop first, add components at milestones** | Isolate regressions, prove each addition |
| Core block LR | Same as backbone | **0.5x backbone LR** | 8 gradient accumulations per step from loop iterations |
| Per-sequence depth | Not included | **Optional per-sequence Poisson at inference** | Adaptive compute — harder inputs get more iterations |
| Precision | fp16 + GradScaler | **fp16 + GradScaler (training), fp32 Coda eval** | Precision where it matters most |

---

## Hypothesis

OUROBOROS proposed combining six innovations simultaneously. JORMUNGANDR proposes the same architecture but with a **staged activation protocol** that proves each component's value before adding the next, **Parcae's actual training recipe** (MuonAdamW, Poisson depth, Value Embeddings), and **conservative risk management** based on every failure mode our lab has encountered.

**The bet is unchanged:** 121M unique parameters with 338M effective capacity, running at 15-19K tok/s on Strix Halo, targeting val < 2.90 (beating AMADEUS at 157M unique params).

**The difference:** JORMUNGANDR won't crash, because each component is proven before the next is added.

---

## Architecture

Identical to OUROBOROS in structure. The changes are in **training protocol**, **optimizer**, **monitoring**, and **component staging**.

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  v
+----------------------------------+
| PRELUDE (2 unique layers)        |
|   Layer 1: ShortConvBlock        |  <- Local pattern establishment
|   Layer 2: GQABlock + QK-Norm    |  <- One global attention pass (n_kv=4)
+----------+-----------------------+
           | input_embed = h (saved for re-injection)
           v
+--------------------------------------------------------------+
| CORE LOOP (3 ShortConv layers x T iterations)                |
|                                                              |
|   T ~ Poisson(mean_recurrence) during training               |
|   T = mean_recurrence at inference (or per-sequence adaptive)|
|                                                              |
|   for t in 1..T:                                             |
|     +------------------------------------------+             |
|     | Parcae injection:                        |             |
|     |   h = A*h + B*input_embed                |             |
|     |   (A = -exp(log_A), guaranteed stable)   |             |
|     |                                          |             |
|     | Core block (3 ShortConv layers):         |             |
|     |   -> RMSNorm -> GatedConv -> SwiGLU      |             |
|     |   -> RMSNorm -> GatedConv -> SwiGLU      |             |
|     |   -> RMSNorm -> GatedConv -> SwiGLU      |             |
|     |                                          |             |
|     | [STAGED] Momentum coherence:             |             |
|     |   velocity = beta*velocity + block_output|             |
|     |   h = h + velocity                       |             |
|     +------------------------------------------+             |
|                                                              |
|     * t=6: FiLM fingerprint (first gradient iteration)       |
|     * t=7..8: FiLM modulates ShortConv layers                |
|                                                              |
|   TRAINING: t=1..5 detached, t=6..8 with gradients          |
+----------+---------------------------------------------------+
           v
+----------------------------------+
| CODA (4 unique layers)           |
|   Layer 1: ShortConvBlock + FiLM |  <- Local refinement
|   Layer 2: GQABlock + FiLM       |  <- Global context (n_kv=4)
|   Layer 3: ShortConvBlock + FiLM |  <- Local refinement
|   Layer 4: GQABlock + FiLM       |  <- Global context (n_kv=4)
|            + Value Embedding     |     (d_ve=64, projected to kv_dim)
|            + TTT Sniper (1-step) |     THE FANGS (single-step default)
+----------+-----------------------+
           v
  Final RMSNorm -> LM Head -> Predictions
```

---

## Key Changes Explained

### 1. MuonAdamW (from Parcae's actual recipe)

OUROBOROS specified AdamW. Parcae's actual training script uses MuonAdamW with:

```python
# Parcae's actual optimizer config
muon_momentum = 0.95
muon_momentum_warmup_start = 0.85
muon_momentum_warmup_steps = 300
muon_weight_decay = linear_decay(base_wd, 0, max_steps)  # decays to 0
```

Our own Muon results showed 2x token efficiency on BabyLM. On a small token budget, this is the single highest-impact change.

```python
# JORMUNGANDR optimizer groups
param_groups = [
    # Muon group: 2D weight matrices (matmul weights)
    {"params": muon_params, "kind": "muon",
     "lr": 0.02, "momentum": 0.95, "weight_decay": 0.01},

    # AdamW group: everything else (1D, embeddings, norms)
    {"params": adamw_params, "kind": "adamw",
     "lr": 8e-4, "betas": (0.9, 0.95), "weight_decay": 0.1},

    # Slow group: Parcae stability parameters
    {"params": [log_A, log_B], "kind": "adamw",
     "lr": 8e-5, "betas": (0.9, 0.95), "weight_decay": 0.0},

    # Core block: reduced LR (gradient accumulation from 8 iterations)
    {"params": core_block_params, "kind": "muon",
     "lr": 0.01, "momentum": 0.95, "weight_decay": 0.01},
]
```

**Why reduced core block LR:** The shared core block receives gradients from all gradient-tracked iterations (3 per step). Unlike a normal layer that gets one gradient per step, the core block gets an effective 3x gradient accumulation. Reducing LR by 0.5x compensates and prevents the core block from dominating the optimization landscape.

### 2. Poisson-Sampled Loop Depth

OUROBOROS used only a fixed curriculum (2 -> 4 -> 6 -> 8 loops). Parcae's actual code samples from a Poisson distribution, which is a key regularization technique:

```python
def sample_loop_depth(self, step):
    """Parcae-style Poisson depth sampling with curriculum."""
    # Curriculum: ramp mean_recurrence over first 20% of training
    progress = min(step / self.curriculum_steps, 1.0)
    # 1-sqrt schedule (fast ramp early, slows later) — from Parcae
    effective_progress = 1 - math.sqrt(1 - progress)

    t_full = max(self.mean_recurrence - self.backprop_depth, 0)  # detached
    s_full = self.backprop_depth                                  # with grad

    t = max(math.ceil(effective_progress * t_full), 0)
    s = s_full  # always keep full backprop depth

    # Poisson sampling: each batch gets a DIFFERENT total depth
    n_detached = torch.poisson(torch.tensor([float(t)])).long().item()
    n_detached = min(n_detached, 2 * t)  # bounded Poisson (prevent extremes)
    n_grad = s

    return n_detached, n_grad
```

**Why this matters:** Fixed iteration counts cause the model to overfit to a specific depth. Poisson sampling forces the model to produce good outputs regardless of how many iterations it gets. At inference, you use the mean — but the model is robust to variation.

### 3. Value Embeddings (from Parcae's code)

Parcae's actual implementation includes per-layer learned embedding tables added to attention values. This is a non-trivial technique that OUROBOROS missed:

```python
class ValueEmbedding(nn.Module):
    """Per-layer vocabulary-aware bias for attention values.
    Inspired by Parcae's code. Applied to last Coda GQA layer only.

    REVISED: Original design used full kv_dim (512) per layer × 3 layers = ~77M params.
    Fix: use d_ve=64 + linear projection up to kv_dim. Cost: ~3.2M + 16K = ~3.2M total.
    """

    def __init__(self, vocab_size, d_ve, kv_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_ve)  # 50257 × 64 = 3.2M
        self.proj = nn.Linear(d_ve, kv_dim, bias=False)  # 64 × 256 = 16K
        nn.init.zeros_(self.embed.weight)  # start as no-op

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids))  # (B, T, kv_dim)
```

**REVISED:** Applied to last Coda GQA layer only (the TTT sniper layer). Uses d_ve=64 embedding + linear projection to kv_dim=256 (with n_kv_heads=4, head_dim=64). Zero-initialized so it starts as identity.

**Cost:** ~3.2M params (50257 × 64 embedding + 64 × 256 projection). Adds to unique param count: 121M → ~124M.

**Why only 1 layer:** Original design (3 layers × full kv_dim=512) would cost ~77M params — more than the entire core block. Projecting through d_ve=64 and applying to only the final GQA (where it helps most — the TTT layer that benefits from token-aware bias) keeps the cost reasonable.

### 4. FiLM Fingerprint Moved to Iteration 6

OUROBOROS fingerprinted at iteration 4. The problem: with 5 detached iterations (t=1..5), the fingerprint at t=4 is computed without gradients. The model can learn to USE the context (in gradient-tracked t=6..8), but can't directly optimize HOW the context is computed.

**JORMUNGANDR fix:** Fingerprint at iteration 6 (the first gradient-tracked iteration). This means:
- The fingerprint computation receives direct gradients
- 2 gradient-tracked iterations (t=7..8) + 4 Coda layers = 6 FiLM targets (vs OUROBOROS's 7)
- Slightly fewer modulation targets, but the fingerprint actually LEARNS properly

```python
# JORMUNGANDR FiLM schedule
film_fingerprint_iter = 6     # first gradient iteration (was 4 in OUROBOROS)
film_start_iter = 7           # modulation begins next iteration
film_targets = 2 + 4          # iters 7-8 (2) + Coda layers (4) = 6 targets
```

**Alternative considered:** Keep fingerprint at iteration 4 but use a straight-through estimator to flow gradients. This is more complex and less stable — moving the fingerprint is simpler.

### 5. CompreSSM-Style HSV Monitoring

We just studied CompreSSM (balanced truncation for SSM compression). The Hankel singular value analysis applies directly to JORMUNGANDR's loop state:

```python
@torch.no_grad()
def monitor_loop_hsv(self, loop_states):
    """Monitor which loop state dimensions are actually used.
    Computes Gramian-based importance scores per dimension.

    loop_states: list of h tensors from each iteration, shape (B, T, d_model)
    """
    # Stack iterations: (n_iters, B, T, d_model) -> treat as SSM trajectory
    H = torch.stack(loop_states, dim=0)  # (8, B, T, 768)

    # Per-dimension variance across iterations (proxy for controllability)
    iter_variance = H.var(dim=0).mean(dim=(0, 1))  # (768,)

    # Per-dimension contribution to output change (proxy for observability)
    delta = H[-1] - H[0]  # final - initial
    output_contribution = (delta ** 2).mean(dim=(0, 1))  # (768,)

    # Combined importance (product of controllability and observability)
    importance = (iter_variance * output_contribution).sqrt()
    importance_sorted, _ = importance.sort(descending=True)

    # Metrics
    total_energy = importance.sum()
    cumulative = importance_sorted.cumsum(dim=0) / total_energy

    metrics = {
        "hsv/top_10pct_energy": cumulative[int(0.1 * len(cumulative))].item(),
        "hsv/top_50pct_energy": cumulative[int(0.5 * len(cumulative))].item(),
        "hsv/effective_rank": (importance > 0.01 * importance.max()).sum().item(),
        "hsv/energy_concentration": (importance.max() / importance.mean()).item(),
    }
    return metrics
```

**What this tells us:**
- If `hsv/top_10pct_energy` > 0.9 early in training: most loop dimensions are wasted, consider reducing d_model or applying CompreSSM truncation
- If `hsv/effective_rank` < d_model/4: the loop is using < 25% of its capacity — the model is undertrained or the loop is too wide
- If `hsv/energy_concentration` > 100: a few dimensions dominate — possible representation collapse

**Future direction:** If monitoring shows low effective rank after convergence, apply CompreSSM balanced truncation to compress the loop state dimension mid-training. A 768-dim loop that only uses 400 effective dimensions can be truncated to 400, cutting loop compute by ~47%.

### 6. Staged Component Activation Protocol

OUROBOROS activated everything from step 0 (relying on zero-init). JORMUNGANDR uses explicit stages:

```
STAGE 1: BARE LOOP (Steps 0 - 15%)
    Active:   Prelude + Core Loop + Coda (no FiLM, no TTT, no momentum)
    Loop:     Poisson-sampled, curriculum 2 -> 4 iterations
    Goal:     Prove Parcae stability works. Establish baseline loss curve.
    Metric:   val loss should be decreasing. Loop state norms bounded.

STAGE 2: ADD FILM (Steps 15% - 30%)
    Active:   + FiLM conditioner (fingerprint at iteration 6)
    Loop:     Curriculum 4 -> 6 iterations
    Goal:     FiLM gammas/betas diverge from identity. Quality improves.
    Monitor:  film/gamma_std, film/beta_std — should increase from 0.

STAGE 3: ADD TTT (Steps 30% - 45%)
    Active:   + TTT sniper (single-step) at last Coda GQA
    Loop:     Curriculum 6 -> 8 iterations (full depth reached)
    Goal:     TTT loss contribution is non-zero. No NaN.
    Monitor:  ttt/delta_norm, loss stability.

STAGE 4: FULL TRAINING (Steps 45% - 100%)
    Active:   All components. Poisson depth sampling fully engaged.
    Loop:     8 iterations mean (Poisson-sampled around mean)
    Goal:     Converge to final quality. All metrics stable.

OPTIONAL STAGE 5: UPGRADE TTT (if Stage 4 stable for 20% of training)
    Active:   TTT sniper upgraded from 1-step to 3-step
    Goal:     Quality improvement without NaN
    Fallback: If NaN within 1000 steps, revert to 1-step, reload checkpoint

OPTIONAL STAGE 6: ADD MOMENTUM (if val loss plateaus)
    Active:   + Loop momentum (beta=0.3 initially, ramp to 0.5)
    Goal:     Break through plateau. Momentum provides directional coherence.
    Fallback: If loss INCREASES for 500 steps, disable momentum, reload checkpoint
    Monitor:  momentum/velocity_norm — should not grow unboundedly
```

**Why staging matters:** OUROBOROS has 6 novel components interacting simultaneously. If training fails, you don't know which one caused it. Staging isolates each component. If Stage 1 fails, the entire loop concept is broken. If Stage 3 fails, TTT is the problem. Each stage has a clear fallback.

### 7. Conservative Throughput Estimates

OUROBOROS estimated 19-23K tok/s. This assumed L2 cache provides 10x effective bandwidth for loop iterations 2-8.

**Reality check from our hardware knowledge:**
- 240 GB/s LPDDR5X bandwidth is SHARED between CPU and GPU
- During training, the optimizer (MuonAdamW) runs on CPU and accesses memory
- L2 (6MB) must hold core block weights (~2.8MB) + activations + gradient buffers
- Effective L2 benefit is more likely 3-5x, not 10x

**Revised estimates:**

| Component | OUROBOROS est. | JORMUNGANDR est. | Notes |
|-----------|---------------|-----------------|-------|
| Prelude (2 layers) | ~3ms | ~3ms | Same |
| Loop iter 1 (DRAM) | ~4ms | ~4ms | Same |
| Loop iters 2-8 (L2) | 7 x 1.5ms = 10ms | 7 x 2.2ms = 15ms | Conservative L2 benefit |
| Parcae injection | ~0ms | ~0ms | Element-wise |
| Coda (4 layers + VE) | ~7ms | ~8ms | Value Embeddings add lookup |
| LM head + loss | ~2ms | ~2ms | Same |
| **Forward total** | ~26ms | ~32ms | |
| **Fwd + Bwd** | ~85ms | ~100ms | |

| Metric | OUROBOROS (optimistic) | JORMUNGANDR (conservative) |
|--------|----------------------|---------------------------|
| Est. tok/s | 19-23K | **15-19K** |
| vs AMADEUS (13.2K) | 1.4-1.7x | **1.1-1.4x** |
| 45-min tokens | 51-62M | **40-51M** |

Even at the conservative estimate, JORMUNGANDR beats AMADEUS on throughput while using fewer unique parameters. The param efficiency is the real win, not raw tok/s.

---

## New Optimizations

### 8. Dual-Rate Learning for Shared vs Unique Layers

The core block is shared across 8 iterations. During backward (iterations 6-8), it receives gradients from 3 different forward passes. This is effectively 3x gradient accumulation for the core block, while Prelude/Coda layers receive gradients from only 1 forward pass each.

Without compensation, the core block learns 3x faster than the surrounding layers, creating an optimization imbalance.

```python
# JORMUNGANDR learning rate schedule
param_groups = {
    "prelude":    {"lr_mult": 1.0},    # standard rate
    "core_block": {"lr_mult": 0.5},    # halved — compensates for 3x grad accumulation
    "coda":       {"lr_mult": 1.0},    # standard rate
    "log_A":      {"lr_mult": 0.1},    # very slow — stability-critical
    "log_B":      {"lr_mult": 0.1},    # very slow — stability-critical
    "film":       {"lr_mult": 1.0},    # standard
    "ttt":        {"lr_mult": 1.0},    # standard
    "value_embed":{"lr_mult": 1.0},    # standard
}
```

### 9. Per-Sequence Adaptive Depth at Inference

Parcae's code supports per-sequence variable depth. At inference, instead of using fixed `mean_recurrence`, run an early-exit check:

```python
@torch.no_grad()
def adaptive_inference(self, input_ids, max_iters=12, convergence_threshold=0.01):
    """Run until the loop converges or max_iters reached.
    Harder inputs get more iterations automatically."""
    h = self.prelude(input_ids)
    input_embed = h

    for t in range(max_iters):
        h_prev = h
        h = self.injection(h, input_embed)
        for layer in self.core_block:
            h = layer(h)

        # Check convergence: relative change in h
        rel_change = (h - h_prev).norm(dim=-1).mean() / h.norm(dim=-1).mean()

        if rel_change < convergence_threshold and t >= 4:  # minimum 4 iters
            break

    return self.coda(h)
```

**Why this matters for deployment:** On Strix Halo as an edge device, most inputs are "easy" (common patterns, short context). Adaptive depth means:
- Easy inputs: 4-5 iterations (fast)
- Hard inputs: 8-12 iterations (thorough)
- Average case is FASTER than fixed 8 iterations

### 10. Gradient Checkpointing Strategy for Loop Iterations

The 3 gradient-tracked iterations (t=6,7,8) each run 3 ShortConv layers = 9 layer-equivalents of activation memory. With gradient checkpointing per-iteration:

```python
# In iterate_forward, gradient-tracked phase:
for step in range(n_grad):
    if self.gradient_checkpointing:
        x = torch.utils.checkpoint.checkpoint(
            self.update_recurrent_state,
            x, input_embeds, freqs_cis, mask,
            use_reentrant=False,  # required for compile compatibility
        )
    else:
        x = self.update_recurrent_state(x, input_embeds, freqs_cis, mask)
```

**Memory savings:** Checkpointing 3 iterations saves ~60% of loop activation memory at the cost of one extra forward pass per iteration. Since the forward pass is cheap (ShortConv, L2-cached), the time cost is minimal (~5ms) but the memory savings allow larger batch sizes.

### 11. Attention Fallback Inside Loop (Emergency Path)

If Stage 1 (bare loop) shows that pure ShortConv inside the loop produces val loss significantly worse than expected (e.g., > 3.2), the fallback is to add one GQA layer inside the loop:

```
EMERGENCY CORE BLOCK (if pure ShortConv insufficient):
    Layer 1: ShortConvBlock
    Layer 2: ShortConvBlock
    Layer 3: GQABlock (lightweight: n_heads=8, n_kv=4)

Cost: +28ms per forward pass (8 iterations x ~3.5ms per GQA)
Benefit: Global context every iteration instead of only Prelude/Coda
```

This is PRE-OUROBOROS Option (B) from Decision 2. Prepared but NOT default.

### 12. Muon Weight Decay Schedule (from Parcae's actual code)

Parcae's training script linearly decays Muon weight decay to 0 over training. This is important — constant weight decay on shared parameters fights the loop's convergence:

```python
def update_muon_wd(optimizer, step, max_steps, base_wd):
    """Linearly decay Muon weight decay to 0. From Parcae's actual code."""
    current_wd = base_wd * (1 - step / max_steps)
    for group in optimizer.param_groups:
        if group.get("kind") == "muon":
            group["weight_decay"] = max(0.0, current_wd)
```

### 13. Loss Guardrail (from Parcae's training script)

Parcae implements an automatic training termination if loss stays too high after sufficient tokens:

```python
# After 10B tokens (proportionally scaled for our budget):
# After ~5M tokens on BabyLM, loss > 6 -> terminate
GUARDRAIL_TOKEN_THRESHOLD = 5_000_000
GUARDRAIL_LOSS_THRESHOLD = 6.0

if total_tokens > GUARDRAIL_TOKEN_THRESHOLD and loss > GUARDRAIL_LOSS_THRESHOLD:
    print(f"Loss guardrail: loss={loss:.2f} after {total_tokens/1e6:.1f}M tokens. Stopping.")
    break
```

### 14. Randomized Position IDs (from Parcae's code)

Parcae randomly samples position IDs instead of using sequential 0..T-1. This prevents the model from memorizing absolute positions:

```python
if self.training and self.randomize_positions:
    position_ids = torch.sort(
        torch.randint(0, self.max_position, (seq_len,), device=device)
    )[0]
```

**Why this matters for looped models:** The same RoPE frequencies are applied at every loop iteration. If position IDs are always sequential, the model could learn to "count" iterations via positional patterns. Randomized positions prevent this degenerate shortcut.

---

## Configuration

### JORMUNGANDR (Standard, d=768)

| Parameter | Value | Source / Change from OUROBOROS |
|-----------|-------|-------------------------------|
| d_model | 768 | Same |
| ffn_inner | 2816 (3.7x) | Same |
| n_heads | 12 | Same |
| n_kv_heads | **4** | **FIXED: 8 gives n_rep=1.5 (not integer). 4 gives 3:1 GQA, proven in ARGUS-PRIME** |
| head_dim | 64 | Same |
| QK-Norm | YES | Same |
| conv_kernel | 3 | Same |
| **Prelude** | 1 ShortConv + 1 GQA | VE removed from Prelude (budget, moved to last Coda GQA only) |
| **Core loop** | 3 ShortConv x 8 iters (Poisson) | **Poisson-sampled depth** |
| **Coda** | 2 ShortConv + 2 GQA + VE(last) + TTT(1-step) | **+ 1 Value Embedding (d_ve=64) on last GQA, single-step TTT** |
| n_loops mean | 8 | Same |
| n_loops sampling | **Poisson(mean), bounded** | **NEW: Parcae regularization** |
| Parcae A init | log_A = -0.7 (A = -0.5) | Same |
| Parcae B init | log_B = -0.7 (B = 0.5) | Same |
| Momentum | **DISABLED by default** | **CHANGED: staged activation only** |
| FiLM fingerprint | **Iteration 6** | **CHANGED: first gradient iteration** |
| FiLM targets | Iters 7-8 + 4 Coda layers = 6 | Reduced from 7 |
| TTT sniper | Last Coda GQA, **1 step default** | **CHANGED: single-step safe default** |
| **Value Embeddings** | **1 GQA layer (last Coda), d_ve=64** | **REVISED: 3 layers @ full kv_dim = ~39M params (budget-breaking). 1 layer @ d_ve=64 = ~3.2M. Projected up inside attention.** |
| d_film | 64 | Same |
| ttt_lr | 0.01 (fixed) | Same |
| vocab_size | 50257 | Same |
| **Position randomization** | **Disabled by default (test in Stage 1)** | **CAUTION: Parcae uses learned positional embeddings, not RoPE. Randomizing positions with RoPE scrambles rotation frequencies. Test carefully — may only apply to ShortConv layers.** |
| **CompreSSM monitoring** | Every eval step | **NEW: loop state health check** |

### JORMUNGANDR-MINI (d=512, L2-Cached Core)

| Parameter | Value | Diff from Standard |
|-----------|-------|-------------------|
| d_model | **512** | Smaller — core block fits L2 |
| ffn_inner | **1792 (3.5x)** | Scaled |
| n_heads | **8** | head_dim=64 |
| n_kv_heads | **4** | Scaled |
| Core block | 3 ShortConv x 8 iters | **~1.8MB -> fully L2-resident** |
| Value Embedding | 1 GQA layer (last Coda), d_ve=64 | Same strategy, projected to kv_dim=256 |
| All else | Same | Same |

---

## Parameter Count

### JORMUNGANDR (d=768)

| Component | Params | Notes |
|-----------|--------|-------|
| Embedding (50257x768, tied) | 38.6M | Tied with LM head |
| Prelude: 1 ShortConv | ~9.1M | Unique |
| Prelude: 1 GQA (no VE) | ~8.8M | Unique (VE moved to Coda only) |
| **Core: 3 ShortConv (SHARED)** | **~27.2M** | **Shared across T iterations** |
| Parcae injection (A, B) | 1.5K | 768 + 768 |
| Coda: 2 ShortConv | ~18.2M | Unique |
| Coda: 2 GQA (1 with TTT) + 1 VE | ~19.2M | Unique (+3.2M VE on last GQA, +1.2M TTT) |
| FiLM conditioner | ~0.7M | 6 targets (reduced from 7) |
| **TOTAL UNIQUE** | **~124M** | +3M from OUROBOROS (Value Embedding on last Coda GQA) |
| **Effective (8x loop)** | **~341M equivalent** | **2.75x param efficiency** |

### JORMUNGANDR-MINI (d=512)

| Component | Params | Notes |
|-----------|--------|-------|
| Embedding (50257x512, tied) | 25.7M | Tied |
| Prelude + Coda | ~28.5M | Unique |
| **Core: 3 ShortConv (SHARED)** | **~12.1M** | **~1.8MB fp16 — FITS L2** |
| FiLM + Parcae + VE | ~1.5M | |
| **TOTAL UNIQUE** | **~68M** | |
| **Effective (8x loop)** | **~165M equivalent** | **2.4x param efficiency** |

---

## Training Protocol

### Optimizer: MuonAdamW

| Parameter | Value | Source |
|-----------|-------|--------|
| **Optimizer** | **MuonAdamW** | **Parcae's actual recipe** |
| Muon base LR | 0.02 | Parcae default |
| Muon momentum | 0.85 -> 0.95 (warmup 300 steps) | Parcae code |
| Muon weight decay | 0.01 -> 0 (linear decay) | Parcae code |
| AdamW base LR | 8e-4 | Our standard |
| AdamW betas | (0.9, 0.95) | Standard |
| AdamW weight decay | 0.1 | Standard |
| LR schedule | Cosine with warmup + cooldown | Parcae recipe |
| Warmup | 10% of max_steps | |
| Cooldown | 5% of max_steps | |
| Min LR | 8e-5 | 10% of base |
| Grad clip | 1.0 | Standard |
| Precision | fp16 + GradScaler | **NOT bf16** (24% slower on gfx1151) |
| Batch | 16 x 256 = 4096 tokens/step | Same as OUROBOROS |

### Optimizer Groups (Detailed)

| Pattern | Optimizer | LR | WD | Notes |
|---------|-----------|-----|-----|-------|
| `core_block.*` (2D) | Muon | 0.01 | decaying | **0.5x base Muon LR** (3x grad accumulation) |
| `prelude.*`, `coda.*` (2D) | Muon | 0.02 | decaying | Standard Muon |
| `*log_A*`, `*log_B*` | AdamW | 8e-5 | 0 | **0.1x base** — stability-critical, slow learning |
| `*film*` | AdamW | 8e-4 | 0.1 | Standard |
| `*ttt*` | AdamW | 8e-4 | 0.1 | Standard |
| `*value_embed*` | AdamW | 8e-4 | 0 | No weight decay on embeddings. Only 1 layer (last Coda GQA). |
| `*norm*`, `*bias` | AdamW | 8e-4 | 0 | Standard |
| `wte`, `lm_head` | AdamW | 8e-4 | 0 | Tied embeddings |

### Loop Depth Schedule

```
Training Progress    Mean Depth    Sampling           Detached / Grad
0% - 10%           2             Fixed              0 / 2
10% - 20%          4             Poisson(4), bounded   1 / 3
20% - 40%          6             Poisson(6), bounded   3 / 3
40% - 100%         8             Poisson(8), bounded   5 / 3
```

### Component Activation Schedule

```
Step 0:      Parcae injection + core loop + Prelude/Coda + Value Embeddings
Step 15%:    + FiLM conditioner (fingerprint at iter 6)
Step 30%:    + TTT sniper (1-step at last Coda GQA)
Step 45%:    Full training (all components active)
Step 65%:    (Optional) Upgrade TTT to 3-step if stable
Step 75%:    (Optional) Enable momentum if loss plateaued
```

### Dataset Strategy

| Phase | Dataset | Tokens | Purpose |
|-------|---------|--------|---------|
| **Smoke** | BabyLM (d=128, 4 layers) | ~200 steps | Validate forward/backward, shapes, gradients |
| **Phase 1** | BabyLM (full d=768) | ~33M (2 epochs) | Validate architecture. Compare to AMADEUS/Tempest. |
| **Phase 2** | GPT-training-small | ~100M | Test loop scaling with more data. Compare val loss. |
| **Phase 3** | WikiText-103 | ~100M | Long-context quality. Test adaptive inference. |
| **Phase 4** | Dolma 10B subset | ~1B | Full scale. Loop scaling laws should manifest. |

**Why this funnel matters:** Parcae's key finding is that "compute-optimal training requires increasing BOTH looping depth AND data together." BabyLM's 16.5M tokens may not be enough data to see the benefit of 8 iterations. If JORMUNGANDR shows quality gains only at Phase 2+, that's expected and not a failure of Phase 1.

---

## Monitoring Dashboard

### Per-Step Metrics

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| `loss` | Training loss | > 6.0 after 5M tokens -> terminate |
| `val_loss` | Validation loss | Track vs AMADEUS (2.90) |
| `grad_norm` | Gradient magnitude | > 10.0 -> investigate |
| `parcae/A_mean` | Mean of -exp(log_A) | Should be in (-0.8, -0.2) |
| `parcae/B_mean` | Mean of exp(log_B) | Should be in (0.2, 2.0) |
| `loop/state_norm` | h norm at loop exit | Unbounded growth -> instability |
| `loop/convergence_rate` | rel_change between last 2 iters | < 0.01 -> loop converged early, consider fewer iters |
| `loop/iteration_diversity` | Variance of h across iterations | Near-zero -> loop collapsed (all iters produce same h) |
| `film/gamma_std` | FiLM gamma deviation from 1.0 | Near-zero after Stage 2 -> FiLM not learning |
| `film/beta_std` | FiLM beta deviation from 0.0 | Near-zero after Stage 2 -> FiLM not learning |
| `ttt/delta_norm` | TTT weight update magnitude | NaN or > 100 -> TTT unstable |
| `hsv/effective_rank` | How many loop dims are used | < d_model/4 -> consider CompreSSM truncation |
| `hsv/energy_concentration` | Dimension dominance ratio | > 100 -> representation collapse |

### Per-Eval Metrics (Every Eval Step)

| Metric | Purpose |
|--------|---------|
| `val_loss_at_depth_2` | Quality with 2 loop iterations |
| `val_loss_at_depth_4` | Quality with 4 loop iterations |
| `val_loss_at_depth_8` | Quality with 8 loop iterations (full) |
| `val_loss_at_depth_12` | Quality with 12 iterations (over-compute) |
| `depth_scaling_slope` | log(val_loss) vs log(depth) — should be negative |

**Depth-scaling evaluation** (from Parcae's validation code): evaluate at multiple depths to see if more iterations actually help. If `val_loss_at_depth_8` is not meaningfully better than `val_loss_at_depth_4`, the model doesn't need 8 iterations — save compute.

---

## Ablation Plan (Revised)

| # | Experiment | What It Tests | Stage |
|---|-----------|---------------|-------|
| 0 | **Smoke test** (d=128, 4 layers, 200 steps) | Forward/backward works at all | Pre-Stage 1 |
| 1 | **Bare loop** (no FiLM, no TTT, no momentum) | Parcae stability on our hardware | Stage 1 |
| 2 | **Bare loop + FiLM** | Does mid-loop fingerprint help? | Stage 2 |
| 3 | **Bare loop + FiLM + TTT(1-step)** | Does TTT sniper help? | Stage 3 |
| 4 | **Full JORMUNGANDR** (all except momentum) | Full architecture quality | Stage 4 |
| 5 | **JORMUNGANDR + momentum** | Does momentum improve or destabilize? | Stage 6 |
| 6 | **8 loops vs 4 vs 2** | How much looping is needed? | Post-Stage 4 |
| 7 | **Poisson depth vs fixed depth** | Does Poisson regularization help? | Post-Stage 4 |
| 8 | **With VE vs without VE** | Do Value Embeddings contribute? | Post-Stage 4 |
| 9 | **MuonAdamW vs AdamW** | Optimizer impact on loop convergence | Post-Stage 4 |
| 10 | **JORMUNGANDR vs AMADEUS** | 124M looped vs 157M standard | Final |
| 11 | **JORMUNGANDR vs ARGUS-PRIME B3** | Looped vs non-looped LFM2.5 | Final |
| 12 | **JORMUNGANDR-MINI (d=512)** | Does L2-cached core deliver? | Final |
| 13 | **JORMUNGANDR + GQA inside loop** | Emergency fallback if quality needs global context | If needed |

---

## Risks & Mitigations (Revised)

| Risk | Severity | Mitigation | Differs from OUROBOROS? |
|------|----------|------------|------------------------|
| Loop doesn't converge | LOW | A-matrix constraint is proven. Staged activation isolates the cause. | Same |
| 124M unique params not enough quality | MEDIUM | 341M effective capacity. If insufficient, add 1 ShortConv to core. | Same |
| Multi-step TTT NaN | **HIGH** | **Start with single-step. Upgrade only after stability proven.** | **NEW: learned from ARGUS-PRIME** |
| Momentum destabilizes | **MEDIUM** | **Disabled by default. Only activate if loss plateaus.** | **NEW: risk identified in assessment** |
| FiLM fingerprint doesn't learn | LOW | **Moved to iter 6 for gradient flow. Monitor gamma/beta std.** | **NEW: gradient flow fix** |
| BabyLM too small for loop scaling | **MEDIUM** | **Multi-phase dataset strategy. GPT-training-small as true test.** | **NEW: data scaling concern** |
| L2 cache less effective than estimated | LOW | Conservative estimates (15-19K vs 19-23K). L2 is a bonus, not dependency. | **NEW: realistic estimates** |
| Core block LR too high (3x grad accumulation) | MEDIUM | **0.5x LR for core block params.** | **NEW: identified in assessment** |
| bf16 accidentally used | LOW | **Explicit fp16 + GradScaler. DO NOT copy Parcae's bf16 setting.** | **NEW: hardware constraint** |
| Compile breaks on no_grad switch | MEDIUM | Compile core block separately. Python loop handles iteration. | Same |

---

## Success Criteria (Revised)

### Must-Hit (Required for Success)

1. **Stage 1 (bare loop) converges** — val loss decreases, loop state norms bounded
2. **JORMUNGANDR val loss < 2.98** on BabyLM — beat Tempest at fewer unique params
3. **Throughput > 15K tok/s** — conservative target, still beats AMADEUS (13.2K)
4. **Quality improves with more loops** (4 -> 8 shows measurable gain)
5. **No training instability** (no NaN, no loss spikes, no divergence)

### Stretch Goals

6. **JORMUNGANDR val loss < 2.90** — beat AMADEUS with 124M vs 157M unique params
7. **Throughput > 18K tok/s** — upper range estimate
8. **Loop scaling confirmed at Phase 2+** — more data + more loops = better quality
9. **JORMUNGANDR-MINI at 68M unique** achieves quality near 120M non-looped
10. **CompreSSM monitoring reveals prunable dimensions** — future optimization path

### Things We'll Learn Regardless

- Whether Parcae's stability mechanism transfers to ShortConv blocks (not just transformer blocks)
- Whether Poisson-sampled depth is essential or just regularization sugar
- Whether FiLM mid-loop introspection works (never tested anywhere)
- Whether Value Embeddings help at our scale
- The actual L2 cache benefit for looped architectures on Strix Halo
- Whether MuonAdamW is strictly better than AdamW for looped models

---

## Implementation Roadmap

1. Create `models/jormungandr.py` — fork from OUROBOROS design
2. Implement `ParcaeInjection` (A-matrix constraint, B injection)
3. Implement `ValueEmbedding` for GQA layers
4. Implement `PoissonDepthSampler` with curriculum
5. Implement `LoopFiLMConditioner` with fingerprint at iteration 6
6. Implement `LoopHSVMonitor` (CompreSSM-style state analysis)
7. Reuse `ShortConvBlock` and `GQABlock` from ARGUS-PRIME
8. Implement `SingleStepTTTSwiGLU` (upgrade path to multi-step)
9. Wire 3-zone forward pass: Prelude -> Loop -> Coda
10. Implement staged activation scheduler
11. Configure MuonAdamW with dual-rate param groups
12. Smoke test (d=128, 4 layers, 200 steps) — **MANDATORY**
13. Compile test (compile core block independently)
14. Stage 1: Bare loop on BabyLM
15. Stage 2-4: Progressive component activation
16. Ablation battery on BabyLM
17. Phase 2: GPT-training-small full run
18. JORMUNGANDR-MINI variant

---

## Future Directions (Post-Validation)

### If JORMUNGANDR succeeds:
- **CompreSSM mid-training truncation:** If HSV monitoring shows low effective rank, apply balanced truncation to compress loop state dimension. A 768-dim loop using 400 effective dims can be truncated to 400 (-47% loop compute).
- **Distillation target:** Train JORMUNGANDR as teacher, distill into a non-looped student for deployment on hardware without L2 benefits.
- **Longer context:** Test with 2048+ sequence length. More iterations may help with longer contexts (Parcae's scaling law).
- **DeltaNet-style injection:** Replace additive A*h + B*e with outer-product injection h + delta_W * e. Combines looping with TTT-style adaptation at the injection point.

### If JORMUNGANDR partially succeeds:
- **Add GQA inside loop** (ablation #13) — if quality needs in-loop global context.
- **Heterogeneous d:** d=768 for Prelude/Coda, d=512 for core loop. Best of both worlds.
- **FLA HGRN inside loop:** 0.40ms per-dimension recurrence (Triton), cheaper than GQA. Adds long-range within the loop at minimal cost.

### If JORMUNGANDR fails:
- **Pure Parcae reproduction:** Fork Parcae's JAX code, port to PyTorch, run on our hardware. If their exact recipe works, the problem is our modifications.
- **Fall back to OUROBOROS-lite:** Remove all additions (no FiLM, no TTT, no VE). Pure Parcae loop + ShortConv. The simplest possible version.

---

## Lab Review (2026-04-16)

**Reviewer:** Engineering (cross-referenced against all empirical results in the lab)

**Verdict: BUILD IT.** Most defensively designed architecture in the repo. Staged activation makes it debuggable. MuonAdamW + Poisson depth from Parcae's proven recipe.

### Fixes Applied

| Issue | Fix | Source |
|-------|-----|--------|
| n_kv_heads=8 gives n_rep=1.5 (not integer) | **n_kv_heads=4** (3:1 GQA) | ARGUS-PRIME had same bug |
| Value Embeddings: 3 layers × full kv_dim = ~77M params | **1 layer (last Coda GQA), d_ve=64 + projection** (~3.2M) | Param budget would be broken |
| Position randomization conflicts with RoPE | **Disabled by default, test in Stage 1** | Parcae uses learned positions, not RoPE |

### Optimization Recommendations

**1. Compile strategy: compile core block independently**

The Python loop with `no_grad`/`grad` switching and Poisson-variable depth will cause torch.compile graph breaks. Don't compile the full model.

```python
# Compile each zone independently
self.prelude = torch.compile(self.prelude, mode="default")
self.core_block = torch.compile(self.core_block, mode="default")
self.coda = torch.compile(self.coda, mode="default")
# The Python loop connecting them is uncompilable — accept this.
```

Use `mode="default"` (not `reduce-overhead`) because CUDAGraphs require fixed tensor shapes, and Poisson depth changes the number of loop iterations per batch. RESONANT-LOOP (our existing looped model) runs at 15.9K tok/s without compile gains — zone-level compile should push this higher.

**2. Start with mean_recurrence=4, not 8**

8 iterations × 3 ShortConv = 24 layer-equivalents of sequential compute. Prelude(2) + Loop(24) + Coda(4) = 30 effective layers — far more than ARGUS-PRIME's 16 at 18K tok/s. Start conservative:

- Phase 1 (BabyLM): target mean_recurrence=4 (14 effective layers)
- Phase 2 (GPT-training-small): ramp to 6 if quality improves with depth
- Phase 3+: ramp to 8 only if depth-scaling evaluation shows clear benefit

The depth-scaling evaluation (`val_loss_at_depth_2/4/8/12`) will tell us the actual optimal depth. Don't assume 8 is needed.

**3. Autokernel QKV pattern: use separate wq/wk/wv/wo**

All GQA layers must use separate `self.wq`, `self.wk`, `self.wv`, `self.wo` attributes (not fused wqkv) for autokernel's `FusedQKVPattern` to match. This is the same pattern as ARGUS-PRIME's Attention class. Manual QKV fusion loses 3.7x speedup.

**4. bf16 guardrail in model code**

Don't rely on documentation alone — add an assertion:

```python
def forward(self, input_ids, targets=None):
    assert input_ids.device.type != 'cuda' or not torch.is_autocast_enabled() or \
           torch.get_autocast_gpu_dtype() != torch.bfloat16, \
           "bf16 is 24% slower on gfx1151 and crashes torch.compile with RoPE. Use fp16."
    ...
```

**5. FiLM modulation count is "4 unique + 2 repeated"**

Iterations 7-8 share the same core block weights, so FiLM modulates the same computation twice. The 4 Coda layers are truly unique. This isn't a problem, but the plan should acknowledge that effective FiLM diversity is 4 unique + 2 repeated, not 6 unique.

**6. L2 cache benefit: measure, don't assume**

The plan estimates 3-5x L2 benefit for loop iterations 2-8. Profile iteration 1 vs iteration 2 timing with `rocprof` to get the actual number. If L2 benefit is <2x, the throughput estimate drops to 12-15K tok/s — still viable but changes the Phase 1 go/no-go.

```bash
rocprofv3 --hip-trace -o trace.csv python -m halo_training \
    --model models/jormungandr.py --class-name Jormungandr --smoke
# Compare kernel times for iteration 1 (cold) vs iteration 2+ (warm)
```

**7. Emergency GQA fallback throughput cost**

Section 11 proposes adding GQA inside the loop as a fallback. At 8 iterations × ~3.5ms per GQA, this adds ~28ms to forward, dropping throughput to ~10K tok/s — below AMADEUS (13.2K). If this fallback is ever needed, reduce to 4 loop iterations first (14ms cost, ~14K tok/s) before committing to 8.

**8. Gradient checkpointing for loop: use `use_reentrant=False`**

Section 10's gradient checkpointing code correctly specifies `use_reentrant=False`. This is mandatory for torch.compile compatibility. Good.

### Risk Assessment Update

| Risk | Original | Revised | Why |
|------|----------|---------|-----|
| n_kv_heads bug | Not identified | **FIXED** | Would crash at GQA forward pass |
| VE param explosion | Not identified | **FIXED** | ~77M → 3.2M params |
| RoPE + position randomization | Not identified | **FLAGGED** | May corrupt positional encoding |
| torch.compile graph breaks | MEDIUM | **HIGH** | Python loop + variable depth + no_grad switch = no full-model compile |
| 30 effective layers → slow | Not identified | **MEDIUM** | Start with 4 iters (14 effective layers), not 8 (30) |

### Key Metrics to Track in Stage 1

If any of these fail, stop and diagnose before proceeding:

1. **Loop state norm bounded** — `h.norm()` should not grow iteration over iteration
2. **Loss decreasing within 500 steps** — if loss is flat after 500 steps, Parcae injection may not be working
3. **L2 cache benefit measured** — compare iteration 1 vs 2+ timing
4. **Throughput > 12K tok/s** (at 4 iterations) — below this, the loop overhead is too high
5. **No NaN** — if NaN appears in Stage 1 (no TTT, no momentum), the core loop stability is broken

---

## References

- Parcae (Together AI, 2026) — stable looped transformers, SSM-style A-matrix constraint, MuonAdamW recipe
- CompreSSM (Chahine et al., ICLR 2026) — balanced truncation for SSM state compression, HSV monitoring
- LFM2/LFM2.5 (Liquid AI, 2025/2026) — ShortConv + sparse GQA, 3:1 ratio
- AMADEUS (our lab) — FiLM conditioning, val 2.90
- TEMPEST (our lab) — compile-optimized Griffin, 22.3K tok/s
- LAZARUS (our lab) — TTT fast weights
- ARGUS-PRIME (our lab) — LFM2.5-scaled ablations, TTT sniper, multi-step NaN finding, n_kv bug
- RESONANT-LOOP (our lab) — shared block (val 3.42, SCORE damping insufficient), 15.9K tok/s
- OUROBOROS (our lab) — predecessor design, PRE_OUROBOROS decision log
- PARCAE.md (our lab) — research assessment of Parcae paper and code
