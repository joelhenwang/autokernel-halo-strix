# Universal Implementation Cookbook

**How to Build ANY of Our 13 Architectures — Module Library, Training Recipes, Stability Lessons**

## Purpose

This cookbook provides the **shared implementation recipe** for all 13 architectures in our laboratory. It tells the engineer WHAT to build and how to wire it. It does NOT repeat:

- **Token budget math** → see `PRETRAINING_CONCERNS.md`
- **Evaluation methodology** → see `EVALUATION_GUIDE.md`
- **MFU analysis** → see `BPB_MFU_ANALYSIS.md`
- **Training loop basics** → see `llm_engineer_agent/CLAUDE (1).md`

## Universal Constants

| Constant | Value |
|----------|-------|
| vocab_size | 50257 (tiktoken GPT-2) |
| tokenizer | `tiktoken.get_encoding("gpt2")` |
| max params | 250M |
| target hardware | AMD Strix Halo gfx1151 (RDNA 3.5, no MFMA, 240 GB/s, 6MB L2) |
| dataset (pretrain) | `datasets/babylm-strict-small/` (~16M tokens, multi-epoch mandatory) |
| dataset (smoke test) | `datasets/smoke-test-dataset/` |
| precision | fp16 mixed (default), fp32 for scans/Sinkhorn |
| compile | `torch.compile(model, mode="reduce-overhead")` |
| kernel integration | `autokernel.optimize(model, compile=True)` for inference |

---

## Section 1: Shared Module Library

Every module below appears in 1+ architecture plans. Interface, init recipe, and optimizer group are CONSENSUS values across all plans.

### 1.1 RMSNorm

| Field | Value |
|-------|-------|
| Interface | `RMSNorm(d: int)`, forward: `(B,T,d) → (B,T,d)` |
| Init | weight = ones(d), no bias |
| Optimizer | 1x LR, 0 WD |
| Used by | ALL 13 architectures |

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

### 1.2 SwiGLU FFN

| Field | Value |
|-------|-------|
| Interface | `SwiGLU(d: int, ffn_inner: int)`, forward: `(B,T,d) → (B,T,d)` |
| Init | Xavier uniform for Linear weights, zero bias |
| Optimizer | 1x LR, 0.1 WD |
| Used by | ALL 13 architectures |

```python
class SwiGLU(nn.Module):
    def __init__(self, d, ffn_inner):
        super().__init__()
        self.w_gate_up = nn.Linear(d, 2 * ffn_inner, bias=False)  # gate + up fused
        self.w_down = nn.Linear(ffn_inner, d, bias=False)
    def forward(self, x):
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)
```

### 1.3 Gated Short Conv (k=3)

| Field | Value |
|-------|-------|
| Interface | `GatedConv(d: int, k: int=3)`, forward: `(B,T,d) → (B,T,d)` |
| Init | B,C,h_tilde projections: normal(0, 0.02). Conv weights: normal(0, 0.02), zero bias |
| Optimizer | 1x LR, 0.1 WD |
| Used by | Caveman LFM, Parallel Caveman, ARCHON, Meta Engram, Harmonic Dreamer, Deep Narrow Oracle, Burst Oracle, Spectral Hydra (k=4), Genius Caveman |

```python
# Forward:
B, C, h_tilde = linear(x, d → 3*d).chunk(3)  # three d-dim projections
y = B * h_tilde                                # element-wise gate
z = causal_conv1d(y, kernel_size=k)            # depthwise, d channels
out = linear(C * z, d → d)                     # output gate + projection
```

### 1.4 Griffin Gated Recurrence

| Field | Value |
|-------|-------|
| Interface | `GriffinRecurrence(d: int, n_heads: int)`, forward: `(B,T,d), h_prev → (B,T,d), h_new` |
| Init | Gate/value projections: Xavier. **decay_bias: spectrum init (see below)**. |
| Optimizer | Projections: 1x LR, 0.1 WD. **decay_bias: 0.1x LR, 0 WD** |
| Used by | Caveman LFM, Parallel Caveman, Spectral Hydra, ARCHON, Meta Engram, Dual Cortex, Genius Caveman |

**Decay bias spectrum initialization (per-dim):**

| Dims | Bias | σ(bias) | Captures |
|------|------|---------|----------|
| 0–25% | -2.2 | ~0.10 | Fast decay: local patterns, N-grams |
| 25–75% | 0.0 | 0.50 | Medium decay: clause structure |
| 75–100% | +4.6 | ~0.99 | Slow decay: topic, entity tracking |

```python
# Forward (inference — sequential):
a = sigmoid(Wa(x) + decay_bias)        # decay gate with spectrum bias
i = sigmoid(Wi(x))                     # input gate
v = Wv(x)                              # value projection
h = a * h_prev + sqrt(1 - a**2) * (i * v)  # Griffin coupling (bounded!)

# Forward (training — parallel scan):
# Associative operator: (a2, b2) ⊕ (a1, b1) = (a2*a1, a2*b1 + b2)
# Pairs: (a_t, sqrt(1-a_t^2) * i_t * v_t)
# Requires fp32 accumulation
h = associative_scan(operator, pairs)
```

### 1.5 Associative Scan

| Field | Value |
|-------|-------|
| Interface | `associative_scan(operator, pairs) → output` |
| Precision | **fp32 REQUIRED** (accumulated products lose precision in fp16) |
| Used by | All recurrence-based architectures (Spectral Hydra, Caveman LFM, Parallel Caveman, ARCHON, etc.) |

**Implementation options (in order of preference):**
1. **Chunked linear recurrence** (chunk_size=64) — uses cumprod+cumsum within chunks, only T/64 serial inter-chunk steps. **5x faster than sequential on gfx1151.** See `models/amadeus.py:selective_scan_chunked` for working implementation.
2. Adapt `kernels/hip/prefix_scan.py` (8.4x kernel) — for fused HIP inference
3. Manual loop with `torch.compile` — fallback (same speed as sequential in practice)

> **WARNING (verified on gfx1151):** `torch.associative_scan()` from `torch._higher_order_ops` is equally slow as a sequential Python loop on this hardware (~1.3K tok/s for 243M SSM model). The chunked approach achieves 6.4K tok/s. Do NOT use associative_scan — go straight to chunked linear recurrence.

### 1.6 Engram Hash Tables

| Field | Value |
|-------|-------|
| Interface | `Engram(vocab_size, d_engram, n_hash_heads)` with per-layer `W_K, W_V, gate` |
| Init | Tables: normal(0, 0.02). Projections: Xavier. **Conv weights: zero (identity)** |
| Optimizer | **Tables: Adam (NOT AdamW), 5x base LR, 0 WD.** Projections: 1x LR, 0.1 WD |
| Used by | Caveman LFM, Parallel Caveman, ARCHON, Meta Engram, Genius Caveman, Burst Oracle, Harmonic Dreamer, Chimera Engram, Deep Narrow Oracle |

**CRITICAL:** Engram tables use Adam (no weight decay), 5x learning rate. This is from the Engram paper — verified by ablation. Using AdamW or standard LR dramatically hurts table learning.

```python
# Hash function: multi-head modular hashing
def hash_ngram(tokens, table_size, n_heads=4):
    primes = [31, 37, 41, 43]  # distinct primes per head
    hashes = []
    for head in range(n_heads):
        h = 0
        for t in tokens:
            h = (h * primes[head] + t) % table_size
        hashes.append(h)
    return hashes  # K entries per n-gram

# Context-aware gating (per Engram layer):
e = concat(bigram_lookup.mean(0), trigram_lookup.mean(0))  # (d_engram*2,)
alpha = sigmoid(rmsnorm(h) @ rmsnorm(W_K @ e) / sqrt(d))
x = x + alpha * W_V @ e
```

### 1.7 Meta Tokens

| Field | Value |
|-------|-------|
| Interface | `MetaTokens(n_tokens: int, d: int)` |
| Init | normal(0, 0.02) |
| Optimizer | 1x LR, 0.01 WD |
| Used by | Parallel Caveman (128), ARCHON (128), Meta Engram (128), Deep Narrow Oracle (64), Harmonic Dreamer (32 dynamic) |

```python
# Prepend:
meta = self.meta_tokens.expand(B, n_tokens, d)
x = torch.cat([meta, embed(tokens)], dim=1)  # (B, n_meta+T, d)

# Strip at output:
logits = lm_head(x[:, n_meta:, :])  # discard meta positions
```

### 1.8 MTP Heads (Multi-Token Prediction)

| Field | Value |
|-------|-------|
| Interface | `MTPHeads(d, n_heads, lm_head)`, forward: backbone output → K sets of logits |
| Init | Adapter MLPs: Xavier. Shared LM head: weight-tied with embedding |
| Optimizer | Adapters: 1x LR, 0.1 WD. LM head: tied with embedding (no separate group) |
| Loss weights | `[1.0, 0.5, 0.3, 0.2]` for heads 1-4 |
| Used by | ARCHON (4), Burst Oracle (4), Harmonic Dreamer (2), Deep Narrow Oracle (4) |

```python
# Head 1: standard next-token (no adapter)
logits_1 = lm_head(rmsnorm(h))

# Heads 2-K: adapter + shared LM head
for k in range(1, K):
    h_k = h + adapter_k(concat(h, embed(argmax(prev_logits))))
    logits_k = lm_head(rmsnorm(h_k))

loss = sum(w[k] * CE(logits_k, targets[:, t+k]) for k in range(K))
```

### 1.9 Sinkhorn mHC (4-Branch Residual)

| Field | Value |
|-------|-------|
| Interface | `mHC(d, n_branches=4)`, wraps sublayers |
| Init | **alpha=0.01 for all phi parameters.** Biases: zero |
| Precision | **fp32 required for Sinkhorn iterations** |
| Optimizer | 1x LR, 0.1 WD |
| Used by | ARCHON, Chimera Engram, Genius Caveman |

```python
# CRITICAL: exp(logits) BEFORE Sinkhorn (not raw logits!)
H_pre  = sigmoid(0.01 * (x_bar @ phi_pre) + b_pre)         # readout (4,)
H_post = 2 * sigmoid(0.01 * (x_bar @ phi_post) + b_post)   # write-in (4,)
H_res  = sinkhorn(exp(0.01 * mat(x_bar @ phi_res) + b_res), iters=20)  # 4×4 mixing

x = H_pre @ stream        # weighted sum: 4 branches → d
y = sublayer(x)            # conv/rec/FFN
stream = H_res @ stream + H_post * y  # cross-branch mixing + write-in
```

### 1.10 Glue Dictionary / Entropy Router

| Field | Value |
|-------|-------|
| Interface | `GlueDict(tokenizer)` → boolean tensor `(vocab_size,)` |
| Parameters | Zero (pure lookup). Optional MLP router for soft routing: ~70K params |
| Used by | Caveman LFM, Parallel Caveman, Genius Caveman, Dual Cortex, Burst Oracle, Ternary Reflex, Deep Narrow Oracle |

```python
# ~500 glue token IDs (articles, prepositions, conjunctions, auxiliaries, pronouns, punct)
GLUE_WORDS = ["the","a","an","of","in","to","for","on","at","with","by","and","or","but",
              "is","are","was","were","be","been","has","have","had","it","he","she","they",
              "we","I","you","this","that",",",".",";",":","!","?", ...]

# Build lookup: encode all cased/whitespace variants
enc = tiktoken.get_encoding("gpt2")
glue_ids = set()
for word in GLUE_WORDS:
    for variant in [word, " "+word, word.capitalize(), " "+word.capitalize()]:
        glue_ids.update(enc.encode(variant))
IS_GLUE = torch.zeros(50257, dtype=torch.bool)
IS_GLUE[list(glue_ids)] = True
```

### 1.11 TernaryLinear + Clipped STE

| Field | Value |
|-------|-------|
| Interface | `TernaryLinear(in_d, out_d)`, forward: quantize to {-1,0,+1}, backward: clipped STE |
| Init | Latent weights: normal(0, 0.02). **Master weights always fp32** |
| Optimizer | 1e-4 LR max (NOT standard 8e-4), 200-step warmup, 0 WD |
| Used by | Ternary Reflex only |

```python
class ClippedSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, thresh=0.5, clip_val=1.0):
        ctx.save_for_backward(w)
        ctx.clip_val = clip_val
        out = torch.zeros_like(w)
        out = torch.where(w > thresh, torch.ones_like(w), out)
        out = torch.where(w < -thresh, -torch.ones_like(w), out)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        (w,) = ctx.saved_tensors
        mask = (w.abs() <= ctx.clip_val).to(grad_out.dtype)  # BOUNDED backward
        return grad_out * mask, None, None
```

### 1.12 DHO Recurrence (Damped Harmonic Oscillator)

| Field | Value |
|-------|-------|
| Interface | `DHORecurrence(d, n_heads)`, forward: complex scan |
| Init | omega: softplus_inv(π/4, π/16, π/64 by head group). gamma: softplus_inv(0.3, 0.05, 0.01) |
| Precision | **fp32 complex for scan accumulation** |
| Optimizer | **omega, gamma: 0.125x LR, 0 WD.** Projections: 1x LR, 0.1 WD |
| Used by | Harmonic Dreamer only |

```python
r = exp(-softplus(gamma_param))     # decay magnitude ∈ (0, 1)
omega = softplus(omega_param)       # angular frequency ∈ (0, ∞)
pole = r * exp(1j * omega)          # complex pole

# Training: complex associative scan
z = complex_parallel_scan(pole.expand(T), x_complex)  # O(T log T), fp32 complex
h = z.real  # real-valued output
```

### 1.13 Per-Layer Embeddings (PLE)

| Field | Value |
|-------|-------|
| Interface | `PLE(vocab_size, d_ple, d_model, n_layers)` — per-layer embedding tables + projections |
| Init | Tables: normal(0, 0.02). Projections: Xavier uniform. |
| Optimizer | 1x LR, 0 WD (same treatment as main embedding) |
| Source | Gemma 4 (Google, 2025) — E2B/E4B models. Config: `hidden_size_per_layer_input: 256` |
| Used by | VIRTUOSO. Can be added to ANY architecture as a plug-in. |

Each layer has its own embedding table `(vocab_size × d_ple)` + linear projection `(d_ple → d_model)`. Injected via addition to residual stream BEFORE each layer processes.

```python
class PerLayerEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_ple, d_model, n_layers):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(vocab_size, d_ple) for _ in range(n_layers)])
        self.projections = nn.ModuleList([nn.Linear(d_ple, d_model, bias=False) for _ in range(n_layers)])

    def get(self, token_ids, layer_idx):
        return self.projections[layer_idx](self.tables[layer_idx](token_ids))

# Usage in layer forward:
# h = h + ple.get(token_ids, layer_idx)  # inject before layer processing
```

**Cost:** Zero FLOPs (embedding lookup only). For vocab=50257, d_ple=32, 16 layers: 25.7M params + 0.5M projections = 26.2M total. On Strix Halo's 128GB unified memory, the extra tables (~52MB fp16) are negligible.

**Why it works (from Gemma 4):** Standard transformers pack ALL token info into one initial embedding. Deep layers lose track of token identity ("representation drift"). PLE gives each layer FRESH token-level information — solving the "frontloading problem."

### 1.14 Residual Momentum

| Field | Value |
|-------|-------|
| Interface | `MomentumResidual(beta_init)`, forward: `(h, layer_output, velocity) → (h_new, velocity_new)` |
| Init | beta_init=0.5 (learned via sigmoid, kept in (0,1)). Velocity: zero-initialized. |
| Optimizer | 1x LR, 0 WD (single scalar parameter) |
| Source | Original invention (Mad Scientist 2.0, 2026). Nobody has published depth-dimension momentum. |
| Used by | PROMETHEUS, TEMPEST. Can be added to ANY architecture. |

Replaces the standard residual `h = h + f(h)` with a momentum formulation:

```python
class MomentumResidual(nn.Module):
    def __init__(self, beta_init=0.5):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init / (1 - beta_init))))

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)

    def forward(self, h, layer_output, velocity):
        velocity = self.beta * velocity + layer_output
        return h + velocity, velocity
```

**Apply to MIXER output only** (conv/recurrence/attention), not FFN. Initialize velocity to zeros at start of forward pass.

**Cost:** ONE multiply-add per element per layer. Element-wise = FREE on Strix Halo.

**Math:** Early layers contribute ~1/(1-beta) more to the final state. At beta=0.5: ~2x amplification for early layers. Creates a natural gradient highway — early layers get stronger gradients without any explicit mechanism.

**Why it works:** Standard residuals are memoryless — each layer adds independently. Momentum gives the residual stream INERTIA. If layers 1-5 all push in the same direction, the accumulated velocity carries that signal further. This is Newtonian mechanics applied to the depth dimension.

---

## Section 2: Optimizer Group Factory

Table-driven assignment. Scan `model.named_parameters()` and assign groups by name pattern:

| Name pattern | LR multiplier | Weight Decay | Optimizer | Rationale |
|-------------|--------------|-------------|-----------|-----------|
| `*engram*table*`, `*gram_table*` | **5.0** | **0** | **Adam** | Engram paper: 5x LR, zero WD, Adam not AdamW |
| `*decay_bias*` | 0.1 | 0 | AdamW | Preserve spectrum (Spectral Hydra) |
| `*omega_param*`, `*gamma_param*` | 0.125 | 0 | AdamW | Preserve DHO frequency spectrum |
| `*meta_token*` | 1.0 | 0.01 | AdamW | Light WD (Hymba) |
| `*ple_table*`, `*ple*embed*` | 1.0 | 0 | AdamW | PLE tables: same as main embedding (Gemma 4) |
| `*norm*`, `*bias` (not decay_bias) | 1.0 | 0 | AdamW | Standard: norms/biases no WD |
| everything else | 1.0 | 0.1 | AdamW | Standard backbone |

**Base LR:** 8e-4, cosine schedule → 8e-5, 100-step warmup.
**Exception:** Ternary path uses 1e-4 max LR with 200-step warmup.

```python
def build_optimizer_groups(model, base_lr=8e-4):
    groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "engram" in name and "table" in name:
            groups.append({"params": [param], "lr": base_lr * 5, "weight_decay": 0})
        elif "decay_bias" in name:
            groups.append({"params": [param], "lr": base_lr * 0.1, "weight_decay": 0})
        elif "omega_param" in name or "gamma_param" in name:
            groups.append({"params": [param], "lr": base_lr * 0.125, "weight_decay": 0})
        elif "meta_token" in name:
            groups.append({"params": [param], "lr": base_lr, "weight_decay": 0.01})
        elif "norm" in name or (name.endswith(".bias") and "decay" not in name):
            groups.append({"params": [param], "lr": base_lr, "weight_decay": 0})
        else:
            groups.append({"params": [param], "lr": base_lr, "weight_decay": 0.1})
    # Use Adam (not AdamW) for Engram tables — separate optimizer if needed
    return groups
```

---

## Section 3: Phase Training Framework

Every architecture uses multi-phase training. Generic scheduler:

```python
class PhaseScheduler:
    def __init__(self, phases, total_steps):
        """phases: [(fraction, set_of_module_prefixes, description)]"""
        self.phases = phases
        self.boundaries = []
        cumulative = 0
        for frac, _, desc in phases:
            cumulative += int(frac * total_steps)
            self.boundaries.append(cumulative)

    def get_phase(self, step):
        for i, boundary in enumerate(self.boundaries):
            if step < boundary:
                return i
        return len(self.phases) - 1

    def apply(self, model, step):
        phase_idx = self.get_phase(step)
        active_prefixes = set()
        for i in range(phase_idx + 1):
            active_prefixes.update(self.phases[i][1])
        for name, param in model.named_parameters():
            param.requires_grad_(any(name.startswith(p) for p in active_prefixes))
```

**Save checkpoint at every phase boundary.** Use `strict=False` when loading across phases.

---

## Section 4: Stability Lessons

### 4.1 Precision Decision Matrix

| Component | fp16 | bf16 | fp32 |
|-----------|------|------|------|
| Standard backbone (conv, FFN, norms) | **YES** | YES if no throughput penalty | NO |
| Associative scan accumulation | NO | NO | **YES** |
| Sinkhorn iterations (mHC) | NO | NO | **YES** |
| STE backward pass | RISKY | Safer | Fallback |
| GradScaler needed? | **YES** | NO | NO |

**Default: fp16 + GradScaler.** Verify bf16 throughput with `prepare.py` before committing. RDNA 3.5 may emulate bf16 at reduced speed.

### 4.2 Gradient Clipping Strategy

```python
# EVERY STEP:
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if not torch.isfinite(total_norm):
    optimizer.zero_grad(set_to_none=True)  # skip step
    log_warning(f"Non-finite grad norm at step {step}")
```

| Architecture tier | max_norm | Notes |
|------------------|----------|-------|
| Tier 1 (standard) | 1.0 | Default |
| Tier 2 (medium) | 1.0 | Default |
| Tier 3 (complex) | 0.5 | Extra caution |
| Tier 4 + STE | **0.05** | Aggressive clipping required |

### 4.3 Label Smoothing

| Situation | Smoothing value |
|-----------|----------------|
| Standard architectures | 0.0 (no smoothing) |
| Large vocab instability observed | 0.01-0.02 |
| STE architectures (Ternary Reflex) | **0.02-0.05** |

### 4.4 Progressive Unfreezing (Tier 3-4 Fallback)

If phase training produces instability, use this more conservative sequence:

1. Freeze everything except embeddings + LM head. Train 500 steps.
2. Unfreeze backbone (conv/FFN). Train to phase 1 end.
3. Unfreeze recurrence/SSM. Train to phase 2 end.
4. Unfreeze Engram/mHC/MTP. Train to phase 3 end.

### 4.5 Small-Scale Validation Protocol (MANDATORY)

**Before ANY full-scale run, run this smoke test:**

```python
# Mini config
mini_config = dict(d_model=128, n_layers=4, ffn_inner=256, vocab_size=1000, block_size=128)
model = ArchitectureClass(**mini_config)
# Train 200 steps on random data
for step in range(200):
    x = torch.randint(0, 1000, (4, 128)).cuda()
    loss = model(x).loss
    loss.backward()
    optimizer.step()
    # ASSERTIONS:
    assert not torch.isnan(loss), f"NaN loss at step {step}"
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    assert grad_norm < 100, f"Exploding gradients: {grad_norm} at step {step}"
    assert loss.item() < 50, f"Loss not decreasing: {loss.item()} at step {step}"
```

**If ANY assertion fails: DO NOT proceed to full scale. Fix first.**

This is THE lesson from the Ternary Reflex failure. Small scale (d=128, vocab=10K) worked. Full scale (d=320, vocab=50K) exploded to 10^36 gradients.

---

## Section 5: Architecture Build Order

| Tier | Architecture | Risk | New Modules Needed | Build After |
|------|-------------|------|--------------------|-------------|
| 1 | **Caveman LFM** | LOW | GatedConv, Griffin, Engram, GlueDict | — (first!) |
| 1 | **Spectral Hydra** | LOW | Griffin (reuse from Caveman LFM), AssocScan | Caveman LFM |
| 1 | **Parallel Caveman** | LOW | MetaTokens (+ reuse all Caveman LFM modules) | Caveman LFM |
| 2 | **Resonant Loop** | MED | SCORE damping, ACT halting | Tier 1 |
| 2 | **Meta Engram** | MED | 4-gram Engram, MetaTokens | Tier 1 |
| 2 | **Deep Narrow Oracle** | MED | MTP heads (+ 48 layers, gradient flow risk) | Tier 1 |
| 3 | **Dual Cortex** | HIGH | Entropy router, dual-width projections | Tier 1-2 |
| 3 | **Burst Oracle** | HIGH | MTP + Engram verification (complex inference) | Tier 1-2 |
| 3 | **Harmonic Dreamer** | HIGH | DHO recurrence (novel!), dynamic scratchpad | Tier 1-2 |
| 4 | **ARCHON** | V.HIGH | mHC Sinkhorn + ALL above | Tier 1-3 |
| 4 | **Chimera Engram** | V.HIGH | Mamba-3 complex SSM + MoE + mHC | Tier 1-3 |
| 4 | **Genius Caveman** | V.HIGH | Dual path + Mamba-3 + MoE + mHC | Tier 1-3 |
| 4 | **Ternary Reflex** | V.HIGH | TernaryLinear + Clipped STE (known instability) | Tier 1-3 |

**Recommendation:** Build **Caveman LFM first**. It establishes 80% of the shared module library (GatedConv, Griffin, Engram, GlueDict, SwiGLU, RMSNorm). Spectral Hydra and Parallel Caveman are then incremental. After Tier 1, every subsequent architecture reuses modules rather than building from scratch.

**Training strategies** (Self-Curriculum, Lottery Forge) apply ON TOP of any Tier 1-3 architecture. Build the architecture first, then layer the training strategy.

---

## Section 6: Ternary Reflex Addendum

The external review identified gradient explosion at full scale. Root cause: **triple interaction of identity STE + fp16 overflow + 50K vocab softmax**.

**The architecture is SOUND.** 30/30 unit tests pass. Small scale trains. The problem is purely the training recipe.

See `reviews/TERNARY-REFLEX-RESPONSE.md` for the full corrected recipe with:
- Clipped STE (bounded backward pass)
- Lower LR (1e-4 max, not 1.6e-3)
- Label smoothing (0.02)
- Restricted vocab in Phase 1 (1K, not 50K)
- 6-phase stabilization sequence with validation gates

**Universal takeaway:** Section 4.5 (Small-Scale Validation) is MANDATORY for all Tier 3-4 architectures. The Ternary Reflex failure would have been caught by a 200-step smoke test at d=128 with full 50K vocab.

---

## Verified Training Baselines (2026-04-09)

Real measurements from `halo_training/` on AMD Ryzen AI MAX+ 395 (gfx1151). Use these to calibrate throughput estimates in all plans.

### Architecture Baselines

| Architecture | Params | Config | tok/s | MFU | Memory | Notes |
|-------------|--------|--------|-------|-----|--------|-------|
| LlamaModel (transformer) | 124.7M | eager | 14,500 | 17% | ~17 GB | Baseline |
| LlamaModel (transformer) | 124.7M | compile + autokernel | **43,000** | **54%** | ~17 GB | 3.05x speedup |
| AMADEUS (SSM hybrid) | 243.8M | eager, chunked scan | 6,400 | 16% | 12.7 GB | Baseline SSM |
| AMADEUS (SSM hybrid) | 243.8M | autokernel patterns only | 7,638 | 19% | 12.7 GB | Tier 1: RMSNorm + SwiGLU |
| AMADEUS (SSM hybrid) | 243.8M | **autokernel + compile + HIP scan** | **10,400** | **26%** | **12.7 GB** | **Best SSM: 1.75x** |
| AMADEUS (adaptive head) | 260.1M | autokernel + compile | 9,983 | 25% | 9.7 GB | **4% slower** — 3 matmuls > 1 |

### Token Budget Calculator

| Throughput | 15 min | 45 min | 120 min |
|-----------|--------|--------|---------|
| 6.4K tok/s (SSM eager) | 5.8M | 17.3M | 46.1M |
| 10.4K tok/s (SSM optimized) | 9.4M | 28.1M | 74.9M |
| 14.5K tok/s (transformer eager) | 13.1M | 39.2M | 104.4M |
| 43K tok/s (transformer + autokernel) | 38.7M | 116.1M | 309.6M |

### Key Findings

**Scan implementation matters 5x:**
- Sequential Python loop: 1.3K tok/s (4% MFU) — **DO NOT USE**
- `torch.associative_scan`: 1.3K tok/s (4% MFU) — **equally slow, DO NOT USE**
- **Chunked linear recurrence** (chunk_size=64): **6.4K tok/s (16% MFU)** — use this
- **Fused HIP scan kernel** (`kernels/hip/selective_scan.py`): scan component 5.9x faster (10.6ms → 1.8ms)
- Reference: `models/amadeus.py`

**torch.compile mode for SSMs:** `mode="default"` and `mode="reduce-overhead"` give identical throughput (8,258 vs 8,278 tok/s). CUDAGraphs don't help when scan dominates. Use `mode="default"` to avoid CUDAGraph conflicts with autokernel.

**Adaptive softmax hurts training:** 3-tier vocab split (8K/16K/26K) saves memory (9.7 vs 12.7 GB) but is 4% slower. One large rocBLAS matmul beats three smaller ones on memory-bound hardware. Don't use for training.
