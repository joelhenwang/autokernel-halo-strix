# Part 11: Architecture Design -- From Papers to Novel Models

## Goal
Design, implement, and validate a novel LLM architecture by synthesizing ideas from multiple research papers. You will build CHIMERA -- a looped hybrid that combines factorized embeddings, conv/attention mixing, stable recurrence, and exclusive self-attention into a single coherent model.

## Why This Matters
There are hundreds of architecture papers published each year. The skill is not reading them -- it is deciding which pieces to combine, how to budget parameters, and how to verify that your Frankenstein creation actually works. This part teaches the synthesis process end-to-end.

## Prerequisites
You should have completed Parts 01-10. Specifically:
- Part 04-05: You can write and benchmark CUDA kernels.
- Part 06-07: You understand autokernel pattern matching and torch.compile.
- Part 09: You have an evaluation harness.
- Part 10: You can read the math in architecture papers (SSMs, spectral norms, attention formulas).

---

## 11.1 The Design Space

Every transformer-like language model is built from the same menu of components. The art is in the selection and combination.

### Attention Mechanisms

**Multi-Head Attention (MHA):** The original. Every head has its own Q, K, V projections. For `n_heads=12, d_model=768`, each head has `d_head=64`. Full attention is O(T^2) in sequence length.

```python
# MHA: each head independent
q = wq(x).view(B, T, n_heads, d_head)      # (B, T, 12, 64)
k = wk(x).view(B, T, n_heads, d_head)      # (B, T, 12, 64)
v = wv(x).view(B, T, n_heads, d_head)      # (B, T, 12, 64)
# Q, K, V all have n_heads=12 independent heads
```

**Grouped Query Attention (GQA):** Share K, V across groups of heads. With `n_heads=12, n_kv_heads=4`, you have 3 query heads per KV head. Saves 2/3 of KV parameters and KV-cache memory.

```python
# GQA: fewer KV heads, queries grouped
q = wq(x).view(B, T, n_heads, d_head)      # (B, T, 12, 64) -- 12 query heads
k = wk(x).view(B, T, n_kv_heads, d_head)   # (B, T, 4, 64)  -- 4 KV heads
v = wv(x).view(B, T, n_kv_heads, d_head)   # (B, T, 4, 64)
# Each KV head serves 3 query heads (n_heads / n_kv_heads = 3)
```

**Multi-Query Attention (MQA):** The extreme case -- 1 KV head shared by all query heads. Maximum KV-cache savings but sometimes hurts quality.

**Exclusive Self Attention (XSA):** A zero-parameter modification to any attention mechanism. After computing attention output, remove the component that points along the value direction for each token's own position. This forces attention to capture information *orthogonal* to what the FFN already sees, reducing redundancy.

```python
# XSA: subtract self-value projection after attention
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Remove self-value component: z_i = y_i - proj(y_i, v_i)
dot = (y * v).sum(dim=-1, keepdim=True)
v_norm_sq = (v * v).sum(dim=-1, keepdim=True).clamp(min=1e-8)
y = y - (dot / v_norm_sq) * v  # zero extra parameters, ~1% quality gain
```

**Linear Attention:** Replace softmax with a kernel function (e.g., ELU+1) to get O(T) complexity. Quality degrades for long sequences but can work as a "fast path" in hybrid models.

### Local Mixing

**Gated Convolution:** A depthwise conv1d with a sigmoid gate. Captures local patterns (n-grams) cheaply. Most modern hybrid models use these as 60-75% of their layers.

```python
class GatedConv(nn.Module):
    def __init__(self, d_model, d_conv, kernel_size=3):
        super().__init__()
        self.proj_in = nn.Linear(d_model, 2 * d_conv, bias=False)
        self.conv = nn.Conv1d(d_conv, d_conv, kernel_size,
                              padding=kernel_size-1, groups=d_conv)
        self.proj_out = nn.Linear(d_conv, d_model, bias=False)

    def forward(self, x):
        gate, val = self.proj_in(x).chunk(2, dim=-1)
        val = val.transpose(1, 2)  # (B, d_conv, T) for Conv1d
        val = self.conv(val)[:, :, :x.size(1)]  # causal trim
        val = val.transpose(1, 2)
        return self.proj_out(F.silu(gate) * val)
```

**Short Convolution Block:** Wraps a gated conv with RMSNorm, momentum residual, and SwiGLU FFN. This is the workhorse layer in LFM2-style architectures.

**Depthwise Convolution:** Conv applied independently per channel. Used in Mamba's input projection and many hybrid models. Very memory-bandwidth efficient.

### Recurrence

**State Space Models (SSMs):** Mamba, S4, Griffin. Process sequences through a recurrent state `h[t] = A * h[t-1] + B * x[t]`. O(T) complexity, constant memory per step at inference. The challenge is initializing A correctly -- see Part 10.

**Parcae Looping:** Instead of stacking 48 unique layers, use 8 shared layers repeated 6 times. Each repetition re-injects the input via a contractive injection:

```python
# Parcae injection: A*h + B*input guarantees contraction
# A = -exp(log_A) forces eigenvalues into (-1, 0)
A = -torch.exp(self.log_A)  # strictly negative, so |A| < 1
B = torch.exp(self.log_B)   # positive injection weight
h = A * h + B * input_embed  # contractive: ||h|| shrinks each iteration
```

The key insight from Parcae is that spectral radius rho(A) < 1 guarantees the loop converges rather than diverging. This gives you 2x parameter efficiency -- 8 layers behave like 16.

**Test-Time Training (TTT):** The model updates its own weights during inference. A small MLP inside the model is trained on the current context using a self-supervised loss. Expensive but powerful for adaptation.

### FFN Variants

**SwiGLU:** The modern standard. Two parallel projections (gate and up), SiLU activation on the gate, element-wise multiply, project down. 2.67x expansion is typical.

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, ffn_inner):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)

    def forward(self, x):
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)
```

**GeGLU:** Same structure but uses GELU activation instead of SiLU. Marginally different in practice.

**Standard ReLU FFN:** The original transformer FFN. Two linear layers with ReLU. Still works fine but SwiGLU is strictly better at the same parameter count.

### Normalization

**RMSNorm:** The standard for modern LLMs. Cheaper than LayerNorm (no mean subtraction), works just as well.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
```

**LayerNorm:** Original transformer normalization. Subtracts mean, divides by std, then applies learned scale and bias. The mean subtraction is wasted compute for LLM-scale models.

**QK-Norm:** Normalize Q and K vectors before computing attention scores. Prevents attention logits from growing unboundedly with depth. Critical for deep models and Parcae-style looping.

```python
# QK-Norm: L2-normalize then scale by learnable parameter
if self.qk_norm:
    q = F.normalize(q, dim=-1) * self.q_scale  # q_scale ~ sqrt(d_head)
    k = F.normalize(k, dim=-1) * self.k_scale
```

### Position Encoding

**RoPE (Rotary Position Embedding):** Rotates Q, K vectors by position-dependent angles in 2D subspaces. Relative position information is preserved through the dot product. The standard choice for modern LLMs.

```python
def precompute_freqs_cis(dim, max_seq_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex exponentials
```

**Learned Position Embeddings:** Absolute positions added to token embeddings. Simple but does not extrapolate to unseen lengths.

**ALiBi:** Adds a linear bias to attention scores based on distance. No learned parameters. Good for extrapolation but slightly worse than RoPE for in-distribution lengths.

### How to Read the Menu

When designing a model, you pick one item from each category:

| Component | Safe Default | Aggressive Choice | Avoid |
|-----------|-------------|-------------------|-------|
| Attention | GQA (4 KV heads) | XSA + GQA | MQA at <350M |
| Local Mix | ShortConv block | Gated Conv + SSM | Pure attention |
| Recurrence | None (standard stack) | Parcae loop | Multi-step TTT from scratch |
| FFN | SwiGLU (2.67x) | SwiGLU (3.7x) | Standard ReLU |
| Norm | RMSNorm | RMSNorm + QK-Norm | LayerNorm |
| Position | RoPE | RoPE | Learned (no extrapolation) |

The "safe default" column gives you a solid baseline. The "aggressive" column is where research happens. The "avoid" column lists things that empirically hurt at our model scale.

---

## 11.2 Reading Papers into Architecture Decisions

Each paper contributes one key insight. Your job is to extract the actionable piece and ignore the rest.

### LFM2 Insight: 60-75% Conv, 25-40% Attention

LFM2 (Liquid Foundation Models 2) found that the optimal hardware-performance tradeoff for hybrid architectures uses mostly local mixing layers with a minority of attention layers. For a 20-layer model, that is roughly 14-15 ShortConv blocks and 5-6 GQA blocks.

Why this works: convolution layers are cheap (no quadratic attention cost, no KV-cache at inference) and capture most local patterns. Attention is expensive but necessary for long-range dependencies. The 75:25 split gets 95% of full-attention quality at 40-50% of the compute cost.

**Actionable decision:** In an 8-layer shared block, place GQA at positions 3 and 6 (25% attention). The other 6 positions get ShortConv blocks.

### Parcae Insight: Stable Looping for 2x Parameter Efficiency

The Parcae paper proves that shared-weight loops with contractive injection converge if the spectral radius of the injection matrix A satisfies rho(A) < 1. The construction `A = -exp(log_A)` guarantees this because exp(x) > 0, so -exp(x) is always in (-inf, 0), and typical initialization at log_A = -0.7 gives A approximately -0.5.

**Actionable decision:** Use 8 unique layers, repeat them 2 times per Parcae iteration. With 3 mean Parcae iterations (Poisson-sampled), you get 8 * 2 * 3 = 48 effective layer applications from 8 unique layers.

**Critical warning:** Never start training with multi-step TTT inside a Parcae loop. The TTT gradient signal through multiple loop iterations causes NaN within the first 100 steps. Always start with single-step TTT or no TTT, then optionally add multi-step after the model has converged.

### XSA Insight: Remove Self-Value Projection (Free Quality)

Exclusive Self Attention removes the component of the attention output that lies along the value direction for each token's own position. The mathematical operation costs nearly nothing (a dot product and subtraction per head per position) but reduces redundancy between the attention and FFN pathways.

**Actionable decision:** Add XSA to all GQA blocks. Zero additional parameters, approximately 1% quality improvement.

### Nandi Insight: Factorized Embeddings Save 25%+ Parameters

The standard embedding table for GPT-2's vocabulary (50,257 tokens) at d=768 is 38.6M parameters. That is 25-40% of a 150M model spent on a lookup table! Nandi's factorized approach uses a low-rank decomposition:

```
Standard:    Embedding(50257, 768) = 38.6M params
Factorized:  Embedding(50257, 256) + Linear(256, 768) = 12.9M + 0.2M = 13.1M params
Savings:     25.5M parameters (66% reduction in embedding cost)
```

The output head mirrors this: `Linear(768, 256)` followed by matmul with the embedding table's transpose.

**Actionable decision:** Use `embed_rank=256` for a d=768 model. This frees 25M parameters to spend on more layers or wider FFNs.

### How Each Paper Contributes One Piece

| Paper | Key Piece We Take | What We Ignore |
|-------|-------------------|----------------|
| LFM2 | 75:25 conv:attention ratio | Their specific conv implementation |
| Parcae | Contractive loop with A = -exp(log_A) | Their curriculum schedule details |
| XSA | Subtract self-value projection in attention | Their theoretical analysis |
| Nandi | Factorized embedding + tied output | Their MoE routing strategy |
| Attention-to-Mamba | Zero-init new components as no-op | Their SSM-specific transfer |

This is the core skill: extract one actionable idea per paper, combine them into a coherent whole.

---

## 11.3 Parameter Budget Planning

### The Budget Equation

```
Total params = Embedding + N_layers * LayerParams + Head
```

For a factorized embedding model:

```
Embedding:
  embed_table = V * R           (V=50257, R=256 -> 12.9M)
  proj_up     = R * D           (R=256, D=768 -> 0.2M)

Per Conv Layer:
  pre_norm    = D               (768 -> 0.8K)
  proj_in     = D * 2*D_conv    (768 * 1024 -> 0.8M)
  conv1d      = D_conv * K      (512 * 3 -> 1.5K)
  proj_out    = D_conv * D      (512 * 768 -> 0.4M)
  ffn_norm    = D               (768 -> 0.8K)
  ffn_gate_up = D * 2*FFN       (768 * 5632 -> 4.3M)
  ffn_down    = FFN * D         (2816 * 768 -> 2.2M)
  Total per conv layer:         ~7.7M

Per GQA Layer:
  pre_norm    = D               (0.8K)
  wq          = D * D           (768 * 768 -> 0.6M)
  wk          = D * KV_dim      (768 * 256 -> 0.2M)
  wv          = D * KV_dim      (768 * 256 -> 0.2M)
  wo          = D * D           (768 * 768 -> 0.6M)
  ffn (same as conv layer FFN): ~6.5M
  Total per GQA layer:          ~8.1M

Head:
  norm        = D               (0.8K)
  proj_down   = D * R           (768 * 256 -> 0.2M)
  (embed table reused -- tied weights, no extra params)
```

### Budget Example: 150M Target

Let us design for approximately 94M unique parameters that become approximately 158M effective with Parcae looping.

```
Factorized Embedding:           13.1M
Prelude GQA block:               8.1M (unshared entry)
Shared block (6 conv + 2 GQA):
  6 * 7.7M + 2 * 8.1M =        62.4M
Parcae injection:                0.002M (just A and B vectors)
Depth memory cache:              0.1M
Coda GQA block:                  8.1M (unshared exit)
Head (factorized):               0.2M
Iter norm:                       0.001M
------------------------------------------
Total unique:                   ~92M
Effective (2x repeat, 3 iters): ~158M
```

### Alignment Constraints for Tensor Cores

NVIDIA Tensor Cores require matrix dimensions to be multiples of specific values for peak throughput:

| Data Type | Required Alignment | Recommended |
|-----------|-------------------|-------------|
| FP16 | Multiple of 8 | Multiple of 128 |
| INT8 | Multiple of 16 | Multiple of 128 |
| TF32 | Multiple of 8 | Multiple of 64 |

Practical rules:
- **d_model:** Use 512, 768, 1024, or 1536. Not 700 or 900.
- **FFN inner dimension:** Use multiples of 128. Not 2800 -- use 2816.
- **Number of heads:** d_model / n_heads must be a power of 2 (typically 64 or 128).
- **KV heads:** n_heads must be divisible by n_kv_heads.

```python
# Verify alignment
def check_alignment(d_model, n_heads, n_kv_heads, ffn_inner, embed_rank):
    head_dim = d_model // n_heads
    kv_dim = n_kv_heads * head_dim

    assert d_model % 128 == 0, f"d_model={d_model} not aligned to 128"
    assert ffn_inner % 128 == 0, f"ffn_inner={ffn_inner} not aligned to 128"
    assert head_dim in (32, 64, 128), f"head_dim={head_dim} not a power of 2"
    assert n_heads % n_kv_heads == 0, f"n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
    assert embed_rank % 64 == 0, f"embed_rank={embed_rank} not aligned to 64"
    print("All alignment checks passed.")

check_alignment(768, 12, 4, 2816, 256)
```

---

## 11.4 Designing CHIMERA (Walkthrough)

We will now walk through designing a concrete architecture from scratch. We call it CHIMERA because it is a hybrid of multiple research ideas stitched together.

### Step 1: Constraints

- **Hardware:** RTX 4060 Ti, 16GB VRAM, CUDA 12.x
- **Parameter budget:** ~94M unique, ~150-160M effective
- **Target:** Beat SmolLM2-135M (HellaSwag 42.1, ARC 43.9)
- **Must support:** torch.compile, custom CUDA kernels via autokernel

### Step 2: Pick Components

Based on the papers from Section 11.2:

| Component | Choice | Source |
|-----------|--------|--------|
| Embedding | Factorized (rank=256) | Nandi |
| Layer mix | 6 ShortConv + 2 GQA per block | LFM2 (75:25 ratio) |
| Attention | GQA with QK-Norm + XSA | GQA standard + XSA paper |
| Looping | Parcae (mean=3 iterations) | Parcae |
| Depth aggregation | Gated DepthMemoryCache | Memory Caching paper |
| FFN | SwiGLU (2816 inner = 3.67x) | Standard |
| Norm | RMSNorm + QK-Norm in GQA | Standard |
| Position | RoPE | Standard |

### Step 3: Calculate Dimensions

```python
# Model configuration
vocab_size = 50257
d_model = 768
embed_rank = 256
n_heads = 12           # d_head = 64
n_kv_heads = 4         # 3 query heads per KV head
ffn_inner = 2816       # 3.67x expansion
d_conv = 512           # conv channel dimension
n_shared_layers = 8    # 6 conv + 2 GQA
n_repeat = 2           # repeat shared block 2x per iteration
mean_recurrence = 3    # Poisson mean for Parcae loop
max_seq_len = 1024
```

### Step 4: Verify Parameter Budget

```python
"""calculate_params.py -- Verify parameter budget for CHIMERA."""
import torch
import torch.nn as nn

V, R, D = 50257, 256, 768
FFN = 2816
D_CONV = 512
N_HEADS, N_KV_HEADS = 12, 4
HEAD_DIM = D // N_HEADS  # 64
KV_DIM = N_KV_HEADS * HEAD_DIM  # 256
CONV_K = 3

def count(name, params):
    print(f"  {name}: {params:,.0f} ({params/1e6:.2f}M)")
    return params

total = 0

print("=== Factorized Embedding ===")
total += count("embed_table", V * R)
total += count("proj_up", R * D)

print("\n=== Prelude GQA Block ===")
total += count("pre_norm", D)
total += count("wq", D * D)
total += count("wk", D * KV_DIM)
total += count("wv", D * KV_DIM)
total += count("wo", D * D)
total += count("q_scale + k_scale", N_HEADS + N_KV_HEADS)
total += count("log_beta", 1)
total += count("ffn_norm", D)
total += count("ffn_gate_up", D * 2 * FFN)
total += count("ffn_down", FFN * D)

print("\n=== Shared Block (6 conv + 2 GQA) ===")
conv_params = 0
conv_params += D  # pre_norm
conv_params += D * 2 * D_CONV  # proj_in
conv_params += D_CONV * CONV_K  # conv1d
conv_params += D_CONV * D  # proj_out
conv_params += 1  # log_beta
conv_params += D  # ffn_norm
conv_params += D * 2 * FFN  # ffn_gate_up
conv_params += FFN * D  # ffn_down
total += count("6 conv layers", 6 * conv_params)

gqa_params = 0
gqa_params += D + D * D + D * KV_DIM + D * KV_DIM + D * D  # norm + Q/K/V/O
gqa_params += N_HEADS + N_KV_HEADS  # scales
gqa_params += 1  # log_beta
gqa_params += D + D * 2 * FFN + FFN * D  # ffn
total += count("2 GQA layers", 2 * gqa_params)

print("\n=== Other Components ===")
total += count("parcae_injection", 2 * D)  # log_A + log_B
total += count("depth_cache", D * 64)      # W_u projection
total += count("iter_norm", D)
total += count("coda_gqa", gqa_params)     # unshared exit

print("\n=== Factorized Head ===")
total += count("proj_down", D * R)
# embed_table reused (tied) -- no extra params

print(f"\n{'='*40}")
print(f"TOTAL UNIQUE: {total:,.0f} ({total/1e6:.1f}M)")
print(f"EFFECTIVE (2x repeat, 3 iters): ~{total * 1.7 / 1e6:.0f}M")
```

Running this should give approximately 92-94M unique parameters.

### Step 5: Autokernel Compatibility Check

Autokernel performs pattern-matching to replace standard PyTorch modules with fused CUDA kernels. Your architecture must follow specific patterns for this to work.

**FusedQKV Pattern:** Autokernel looks for `self.wq`, `self.wk`, `self.wv` as separate `nn.Linear` modules and fuses them into a single `w_qkv` projection + RoPE application. If you manually create a combined `w_qkv`, autokernel cannot match the pattern and you lose the 3.7x speedup.

```python
# CORRECT: separate projections (autokernel matches this)
class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

# WRONG: manual fusion breaks autokernel pattern matching
class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.w_qkv = nn.Linear(dim, (n_heads + 2*n_kv_heads) * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
```

**SwiGLU Pattern:** Autokernel matches `self.w_gate_up` (fused gate+up) followed by `self.w_down`. Keep this naming convention.

**Autokernel breaks at small dimensions:** If d_model <= 256, HIP/CUDA custom kernels may diverge from reference implementations. Only use autokernel for d_model >= 512.

---

## 11.5 Implementation Patterns

### Reusing Existing Components

Never reimplement what already exists. Import base classes and compose them:

```python
"""chimera.py -- CHIMERA architecture implementation."""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse proven components from other models
from models.amadeus import RMSNorm, SwiGLU, GatedConv
from models.argus import precompute_freqs_cis, apply_rotary_emb
from models.argus_prime import Attention, ShortConvBlock
```

### Factorized Embedding Implementation

```python
class FactorizedEmbedding(nn.Module):
    """Low-rank embedding: Embedding(V, R) -> Linear(R, D).

    Saves (V * D - V * R - R * D) params. With V=50260, R=256, D=768:
    saves approximately 25M params vs standard Embedding(V, D).
    """

    def __init__(self, vocab_size: int, rank: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank)
        self.proj_up = nn.Linear(rank, d_model, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj_up(self.embed(input_ids))


class FactorizedLMHead(nn.Module):
    """Factorized output: Linear(D, R) -> matmul with embed table transpose.

    Shares the embedding table from FactorizedEmbedding (tied weights).
    """

    def __init__(self, d_model: int, rank: int, embed_table: nn.Embedding):
        super().__init__()
        self.proj_down = nn.Linear(d_model, rank, bias=False)
        self.embed_table = embed_table  # tied reference, not a copy

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_low = self.proj_down(h)
        return F.linear(h_low, self.embed_table.weight)
```

### Parcae Injection

The injection module mixes the current hidden state with the original input embedding at each loop re-entry. The key guarantee is contraction: `|A| < 1` ensures the loop does not diverge.

```python
class SimpleParcaeInjection(nn.Module):
    """Stable loop injection: h = A*h + B*input_embed.

    A = -exp(log_A) guarantees eigenvalues in (-1, 0) -- always contractive.
    B = exp(log_B) controls input re-injection strength.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Initialize at log(-0.5) approx -0.7 for A approx -0.5
        self.log_A = nn.Parameter(torch.full((d_model,), -0.7))
        self.log_B = nn.Parameter(torch.full((d_model,), -0.7))

    def forward(self, h: torch.Tensor, input_embed: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.log_A)  # always negative -> |A| < 1 if log_A > -inf
        B = torch.exp(self.log_B)   # always positive injection
        return A * h + B * input_embed
```

**Important:** The first loop iteration skips injection because `h == input_embed` at that point, so `A*h + B*h = (A+B)*h` would just scale the input. All Parcae implementations must handle this:

```python
# First iteration: run shared block WITHOUT injection
h, velocity = self._run_shared_block(h, velocity, freqs_cis)

# Subsequent iterations: re-inject then run
for t in range(n_remaining_iterations):
    h = self.injection(h, input_embed)  # contract + re-inject
    h, velocity = self._run_shared_block(h, velocity, freqs_cis)
```

### Depth Memory Cache

Instead of using only the final loop state, cache `h` at each iteration and let the model select its own weighted mix of depths. Some positions converge after 2 iterations; others need all 4.

```python
class DepthMemoryCache(nn.Module):
    """Content-dependent gated aggregation over cached loop iteration states."""

    def __init__(self, d_model: int, d_gate: int = 64):
        super().__init__()
        self.W_u = nn.Linear(d_model, d_gate, bias=False)

    def forward(self, cached_states: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate cached loop states via content-dependent gating.

        Args:
            cached_states: List of (B, T, d_model) tensors, one per iteration.
        Returns:
            Aggregated state (B, T, d_model).
        """
        if len(cached_states) == 1:
            return cached_states[0]

        # Query from the last iteration (highest-depth representation)
        u = self.W_u(cached_states[-1])  # (B, T, d_gate)

        # Gate each cached state by similarity to its mean-pooled fingerprint
        gates = []
        for state in cached_states:
            key = self.W_u(state).mean(dim=1)  # (B, d_gate)
            gate = (u * key.unsqueeze(1)).sum(dim=-1)  # (B, T)
            gates.append(gate)

        gates = torch.softmax(torch.stack(gates, dim=-1), dim=-1)  # (B, T, N)

        # Weighted aggregation
        stacked = torch.stack(cached_states, dim=-1)  # (B, T, d_model, N)
        return (stacked * gates.unsqueeze(2)).sum(dim=-1)  # (B, T, d_model)
```

### XSA GQA Block

The full block combining GQA + XSA + momentum residual + SwiGLU:

```python
class XSAGQABlock(nn.Module):
    """GQA block with Exclusive Self Attention, momentum residual, SwiGLU FFN."""

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4,
                 momentum_beta: float = 0.5, use_xsa: bool = True):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.use_xsa = use_xsa

        # Standard Attention with separate wq/wk/wv for autokernel compatibility
        self.attn = Attention(d_model, n_heads, n_kv_heads, qk_norm=True)

        # Learnable momentum coefficient (initialized to 0.5)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta / (1 - momentum_beta)))
        )

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, velocity: torch.Tensor,
                freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm attention
        attn_out = self.attn(self.pre_norm(x), freqs_cis)

        # Momentum residual: velocity = beta * velocity + attn_out
        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + attn_out

        # Residual connection
        x = x + velocity

        # Pre-norm FFN (inlined RMSNorm for fusion opportunity)
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight

        x = x + self.ffn(normed)
        return x, velocity
```

### compile_zones() for Looped Models

Python loops break `torch.compile` because the compiler cannot handle dynamic control flow. The solution: compile each layer independently, leave the loop uncompiled.

```python
def compile_zones(self):
    """Compile each layer independently for per-zone fusion.

    Call AFTER autokernel.optimize() and BEFORE training.
    The Python loop stays uncompiled; each layer = fused kernel.
    """
    if self.use_prelude:
        self.prelude = torch.compile(self.prelude, mode="default")

    for i in range(len(self.shared_layers)):
        self.shared_layers[i] = torch.compile(
            self.shared_layers[i], mode="default"
        )

    if self.use_coda:
        self.coda = torch.compile(self.coda, mode="default")

    return self
```

**Critical:** Never compile the full model when it contains Python loops. Use `model.compile_zones()` instead. The speedup is similar (3x on JORMUNGANDR-HALO) without the compile failures.

### The Full Forward Pass

Here is the complete forward pass for CHIMERA, bringing all components together:

```python
def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
    B, T = input_ids.shape
    h = self.tok_embeddings(input_ids)       # factorized embed
    freqs_cis = self.freqs_cis[:T]           # RoPE frequencies

    # === PRELUDE (unshared entry GQA) ===
    velocity = torch.zeros_like(h)
    if self.use_prelude:
        h, velocity = self.prelude(h, velocity, freqs_cis)
    input_embed = h  # save for Parcae re-injection

    # === PARCAE LOOP ===
    if self.training:
        step = self.step_counter.item()
        n_detached, n_grad = self.sample_loop_depth(step)
        self.step_counter += 1
    else:
        n_detached = 0
        n_grad = self.mean_recurrence

    cached_states = []

    # First iteration: no re-injection (h == input_embed)
    h, velocity = self._run_shared_block(h, velocity, freqs_cis)
    h = self.iter_norm(h)

    # Detached iterations (no gradient -- saves memory)
    for _ in range(n_detached):
        with torch.no_grad():
            h = self.injection(h, input_embed)
            h, velocity = self._run_shared_block(h, velocity, freqs_cis)
            h = self.iter_norm(h)
            cached_states.append(h.detach())

    # Gradient-tracked iterations (backprop flows through these)
    for _ in range(n_grad):
        h = self.injection(h, input_embed)
        h, velocity = self._run_shared_block(h, velocity, freqs_cis)
        h = self.iter_norm(h)
        cached_states.append(h)

    # Depth Memory Cache: gated aggregation
    if self.depth_cache is not None and len(cached_states) > 1:
        h = self.depth_cache(cached_states)

    # === CODA (unshared exit GQA) ===
    velocity = torch.zeros_like(h)
    if self.use_coda:
        h, velocity = self.coda(h, velocity, freqs_cis)

    return self.lm_head(self.norm(h))
```

### Poisson Depth Sampling

The Parcae loop samples its depth from a Poisson distribution, with a curriculum that starts shallow and increases:

```python
def sample_loop_depth(self, step: int) -> Tuple[int, int]:
    """Parcae-style Poisson depth sampling with 1-sqrt curriculum."""
    progress = min(step / max(self.curriculum_steps, 1), 1.0)
    effective_progress = 1 - math.sqrt(1 - progress)

    t_full = max(self.mean_recurrence - self.backprop_depth, 0)
    t = max(math.ceil(effective_progress * t_full), 0)

    # Sample detached iterations from Poisson
    n_detached = torch.poisson(torch.tensor([float(t)])).long().item()
    n_detached = min(n_detached, 2 * max(t, 1))  # cap at 2x mean
    n_grad = self.backprop_depth  # always this many gradient iterations

    return n_detached, n_grad
```

The 1-sqrt curriculum means depth increases slowly at first, then accelerates. After `curriculum_steps`, the model trains at full depth.

---

## 11.6 The Ablation Discipline

### Always Build a "Bare" Variant

Before testing the full model, you need a baseline to compare against. The "Bare" variant strips all novel components:

```python
class ChimeraBare(ChimeraBase):
    """Ablation: no XSA, no depth cache, no Prelude/Coda."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=3,
            use_xsa=False,
            use_depth_cache=False,
            use_prelude=False,
            use_coda=False,
            **kw,
        )
```

### Always Build a "Mini" Variant for Smoke Testing

Before running an 8-hour training job, verify the model works at tiny scale:

```python
class ChimeraMini(ChimeraBase):
    """Tiny config for smoke testing (~3M params)."""
    def __init__(self):
        super().__init__(
            vocab_size=1000,
            d_model=128,
            embed_rank=32,
            n_shared_layers=4,
            gqa_positions=(1, 3),
            n_repeat=2,
            d_conv=128,
            ffn_inner=256,
            n_heads=4,
            n_kv_heads=2,
            conv_kernel=3,
            mean_recurrence=2,
            backprop_depth=2,
            curriculum_steps=100,
            use_xsa=False,        # XSA needs d >= 512
            use_depth_cache=False,
            momentum_beta_init=0.5,
            max_seq_len=512,
            use_prelude=False,
            use_coda=False,
        )
```

The Mini variant must:
- Have approximately 3M parameters (fast to train)
- Not use autokernel (d=128 is too small)
- Complete a smoke test in under 60 seconds
- Exercise the full forward pass including looping

### One Component at a Time

The correct ablation sequence:

1. **Bare:** No XSA, no depth cache, no Prelude, no Coda. Just the Parcae loop with ShortConv + GQA blocks.
2. **+Prelude:** Add the entry GQA block. Measure: does it help?
3. **+Coda:** Add the exit GQA block. Measure: does it help?
4. **+XSA:** Enable exclusive self-attention. Measure.
5. **+DepthCache:** Enable depth memory cache. Measure.
6. **Full CHIMERA:** All components enabled.

If any component does not help, **remove it**. Do not keep complexity for its own sake.

### Example Ablation Results Table

```
| Variant          | Val Loss | HellaSwag | Tok/s | Notes              |
|------------------|----------|-----------|-------|--------------------|
| Bare             | 3.82     | 29.1      | 45K   | Baseline           |
| +Prelude         | 3.71     | 30.5      | 42K   | -0.11 loss, worth  |
| +Coda            | 3.65     | 31.2      | 40K   | -0.06 loss, worth  |
| +XSA             | 3.62     | 31.8      | 39K   | -0.03 loss, free   |
| +DepthCache      | 3.58     | 32.5      | 38K   | -0.04 loss, worth  |
| Full             | 3.55     | 33.1      | 37K   | Best quality       |
```

If "+DepthCache" showed *no* improvement, you would drop it and ship Bare+Prelude+Coda+XSA.

### Variant Classes for Clean Ablation

Define all variants as separate classes in the same file, inheriting from the base:

```python
class Chimera(ChimeraBase):
    """Default: mean_recurrence=3, XSA on, depth cache on."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=3, **kw)


class ChimeraDeep(ChimeraBase):
    """Deep: 5 iterations for maximum effective capacity."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=5, backprop_depth=3, **kw)


class ChimeraBare(ChimeraBase):
    """Ablation: no XSA, no depth cache."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=3,
            use_xsa=False, use_depth_cache=False, **kw,
        )


class ChimeraNoLoop(ChimeraBase):
    """Ablation: no Parcae loop (single pass through shared block x2)."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=1, backprop_depth=1,
            use_depth_cache=False, **kw,
        )
```

Each variant is one class, one constructor, one `--class-name` flag in the training CLI. This makes experiments reproducible and composable.

---

## 11.7 Putting It All Together: The Implementation File

Here is the complete structure of a model file ready for training:

```python
"""
CHIMERA: Unified Looped Hybrid with Factorized Embeddings.

Combines insights from 5 papers:
  - LFM2: 75:25 conv:attention hybrid ratio
  - Parcae: Stable looping via spectral constraint rho(A)<1
  - XSA: Exclusive Self Attention (zero-param quality boost)
  - Nandi: Factorized embeddings + layer sharing
  - Attention-to-Mamba: Identity initialization for new components

Usage:
    python -m halo_training --model models/chimera.py --class-name ChimeraMini --smoke
    python -m halo_training --model models/chimera.py --class-name Chimera \\
        --dataset babylm --compile --optimize-kernels --muon
"""

# ... imports ...
# ... FactorizedEmbedding, FactorizedLMHead ...
# ... SimpleParcaeInjection, DepthMemoryCache ...
# ... XSAGQABlock ...
# ... ChimeraBase (with full forward pass) ...
# ... Chimera, ChimeraDeep, ChimeraBare, ChimeraNoLoop, ChimeraMini ...
```

The training command:
```bash
# Smoke test (30 seconds)
python -m halo_training --model models/chimera.py --class-name ChimeraMini --smoke

# Full training with all optimizations
python -m halo_training --model models/chimera.py --class-name Chimera \
    --dataset babylm --compile --optimize-kernels --muon

# Ablation
python -m halo_training --model models/chimera.py --class-name ChimeraBare \
    --dataset babylm --compile --optimize-kernels --muon
```

---

## Exercises

### Exercise 1: Design a 100M Architecture

Using the component menu from Section 11.1, design a 100M parameter architecture that does NOT use Parcae looping. Instead, use a standard deep stack.

Requirements:
- Factorized embeddings (rank=192)
- d_model=512 (smaller but deeper)
- At least 16 layers
- 75% conv, 25% GQA
- SwiGLU FFN

Write out the full parameter budget calculation showing approximately 100M total.

### Exercise 2: Calculate Parameter Budget for Your Design

Take your design from Exercise 1 and verify the budget programmatically:

```python
# Implement your model
model = YourModel()
n_params = sum(p.numel() for p in model.parameters())
print(f"Total: {n_params / 1e6:.1f}M")

# Verify each component
for name, module in model.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {n / 1e6:.2f}M")
```

### Exercise 3: Implement with Mini Variant

Create a complete model file with:
- Your full architecture class
- A Mini variant (approximately 3M params, vocab=1000, d=128)
- A Bare variant (strip the most novel component)

Run the smoke test:
```bash
python -m halo_training --model models/your_model.py --class-name YourModelMini --smoke
```

Verify:
- Smoke test completes without errors
- Loss decreases from step 1 to step 50
- No NaN values in loss or gradients

---

## Checkpoint

Before moving to Part 12, verify:

- [ ] You understand the component menu and can justify each choice
- [ ] You can extract one actionable idea from a research paper
- [ ] You have calculated a parameter budget for a 150M model
- [ ] Your model file has: Base, Default, Bare, NoLoop, Deep, and Mini variants
- [ ] Smoke test passes (loss decreases, no NaN)
- [ ] You have defined at least 3 ablation variants ready for systematic comparison

**Expected time:** 8 hours. If you finish faster, go deeper on the ablation experiments -- the discipline of measuring every component individually is the most valuable skill in this entire series.

---

## Quick Reference: Component Import Map

```python
# All reusable components and where they live
from models.amadeus import RMSNorm, SwiGLU, GatedConv, FiLMConditioner
from models.argus import precompute_freqs_cis, apply_rotary_emb, TTTSwiGLU
from models.argus_prime import Attention, ShortConvBlock, GQABlock
from models.griffin_halo import SimpleParcaeInjection
from models.jormungandr_halo import (
    ParcaeInjection,        # with dimension adapter
    DepthMemoryCache,       # gated iteration aggregation
    CodaAttention,          # GQA + XSA + value bias
    CodaGQABlock,           # full block with TTT option
    ValueEmbedding,         # per-layer vocab-aware value bias
)
from models.chimera_halo import (
    FactorizedEmbedding,    # low-rank embed
    FactorizedLMHead,       # tied factorized output
    XSAGQABlock,            # GQA + XSA + momentum
)
```
