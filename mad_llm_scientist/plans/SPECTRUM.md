---
title: "SPECTRUM"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/TEMPEST.md
  - mad_llm_scientist/plans/EREBUS.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %spectrum, %fft, %lti, %selective-correction, %parallel]
---

# SPECTRUM

**FFT-Based LTI Backbone + Selective Correction — The Prism That Splits Light to See Further**

*"White light reveals nothing of itself. Only when split into a spectrum do we see the hidden wavelengths. SPECTRUM decomposes the sequence into parallel frequencies, then corrects only what needs correction."*
*S4 proved FFT-based parallelism works. Mamba proved selectivity matters. SPECTRUM proves you can have both.*

## Hypothesis

S4's original FFT-based parallel scan was abandoned because Mamba's input-dependent selectivity broke the convolution structure. But language has a DUAL nature: ~80% of information is captured by input-INDEPENDENT patterns (grammar, syntax, common phrases) and ~20% requires input-DEPENDENT selectivity (semantic disambiguation, coreference). SPECTRUM decomposes the recurrence into: (1) a fast LTI (Linear Time-Invariant) backbone via FFT that captures the 80% (O(N log N), embarrassingly parallel), plus (2) a small selective correction layer that adds the 20% input-dependence (O(N) element-wise). The sequential bottleneck is 5x smaller than full Mamba, enabling much higher parallelism on Strix Halo.

**Key papers:** "CAT: Circular-Convolutional Attention" (2504.06704, Apr 2025), "S4" (Gu et al., original), "Mamba-3" (2603.15569)

---

## Architecture

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  -> 16 Spectrum Blocks:
  |     RMSNorm
  |     +---------------------------------------------+
  |     | LTI Backbone (via rocFFT):                   |
  |     |   x -> FFT -> H(w) * X(w) -> IFFT -> y_lti  |
  |     |   H(w): learned transfer function per head   |
  |     |   O(N log N), embarrassingly parallel         |
  |     |                                              |
  |     | Selective Correction:                         |
  |     |   Small EFLA (d_state=16, minimal)            |
  |     |   Input-dependent gating on LTI output        |
  |     |   correction = EFLA(x) * sigmoid(gate)        |
  |     |   y = y_lti + alpha * correction              |
  |     |   O(N) sequential but tiny state              |
  |     +---------------------------------------------+
  |     +Residual
  |     RMSNorm -> SwiGLU FFN (768->1920->768) -> +Residual
  |
  -> Final RMSNorm -> LM Head
```

### The LTI-Selective Decomposition

**Standard Mamba:** ALL computation is input-dependent (A, B, C, dt all depend on x). This prevents FFT parallelization. The scan is O(N) sequential.

**SPECTRUM:** Decompose into LTI + correction.
```
y = y_LTI + alpha * correction

y_LTI:      Fixed A, B, C (not input-dependent)
             -> Can be computed as convolution
             -> FFT(x) * H(w) -> IFFT
             -> O(N log N) parallel
             
correction:  Small EFLA with d_state=16
             -> Input-dependent gates
             -> O(N) sequential but 4x smaller state
             -> Captures only what LTI misses
```

**Why this works:** In a linear system, the LTI part captures all structural patterns — grammar, syntax, n-gram statistics, position encoding (via learned frequencies in H(w)). The selective correction handles disambiguation and semantic context — the part that actually REQUIRES input-dependence.

### Transfer Function H(w)

Each head learns a transfer function H(w) in the frequency domain:
```python
H_real = nn.Parameter(torch.randn(n_heads, T//2 + 1) * 0.01)
H_imag = nn.Parameter(torch.randn(n_heads, T//2 + 1) * 0.01)
H = torch.complex(H_real, H_imag)  # learnable frequency response
```

H(w) directly encodes which frequencies the model attends to. Low-frequency components capture global structure (topic, style). High-frequency components capture local patterns (morphology, syntax).

---

## Component 1: LTI Backbone via FFT

```python
class LTIBackbone(nn.Module):
    def __init__(self, d_model=768, n_heads=12, head_dim=64, max_len=1024):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.w_in = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_out = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Learnable transfer function per head (frequency domain)
        n_freq = max_len // 2 + 1
        self.H_real = nn.Parameter(torch.randn(n_heads, n_freq, head_dim) * 0.01)
        self.H_imag = nn.Parameter(torch.randn(n_heads, n_freq, head_dim) * 0.01)

        # Causal mask in frequency domain (learned decay)
        self.decay_log = nn.Parameter(torch.linspace(-1, -4, n_freq).unsqueeze(0).expand(n_heads, -1))

    def forward(self, x):
        B, T, D = x.shape
        h = self.w_in(x).view(B, T, self.n_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3)  # (B, H, T, d)

        # FFT along sequence dimension
        H_freq = h.float()
        X_freq = torch.fft.rfft(H_freq, dim=2)  # (B, H, T//2+1, d)

        # Apply learned transfer function + causal decay
        H = torch.complex(self.H_real, self.H_imag)[:, :X_freq.size(2)]
        decay = torch.exp(self.decay_log)[:, :X_freq.size(2)]
        H_causal = H * decay.unsqueeze(-1)

        Y_freq = X_freq * H_causal.unsqueeze(0)  # element-wise in freq domain

        # IFFT back to time domain
        y = torch.fft.irfft(Y_freq, n=T, dim=2)  # (B, H, T, d)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, -1).half()

        return self.w_out(y)
```

## Component 2: Selective Correction (Tiny EFLA)

```python
class SelectiveCorrection(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_state=16):
        self.n_heads = n_heads
        self.d_state = d_state
        head_dim = d_model // n_heads

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads, bias=True)
        self.w_gate = nn.Linear(d_model, d_model, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # correction strength

    def forward(self, x, y_lti):
        B, T, D = x.shape
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, -1)
        k = F.normalize(k.view(B, T, self.n_heads, -1), dim=-1)
        v = v.view(B, T, self.n_heads, -1)

        beta = torch.sigmoid(self.w_beta(x)).unsqueeze(-1)
        lambda_t = (k * k).sum(dim=-1, keepdim=True)
        efla_alpha = (1 - torch.exp(-beta * lambda_t)) / (lambda_t + 1e-6)

        correction = efla_chunkwise(q, k, v, efla_alpha, chunk_size=64)
        correction = correction.flatten(-2, -1)

        gate = torch.sigmoid(self.w_gate(x))
        return y_lti + self.alpha * gate * correction
```

## Component 3: Spectrum Block

```python
class SpectrumBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.lti = LTIBackbone(d_model, n_heads)
        self.correction = SelectiveCorrection(d_model, n_heads)
        self.ffn_gate_up = nn.Linear(d_model, 2 * 1920, bias=False)
        self.ffn_down = nn.Linear(1920, d_model, bias=False)

    def forward(self, x):
        h = self.norm1(x)
        y_lti = self.lti(h)
        y = self.correction(h, y_lti)
        x = x + y
        h2 = self.norm2(x)
        gate, up = self.ffn_gate_up(h2).chunk(2, dim=-1)
        x = x + self.ffn_down(F.silu(gate) * up)
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
| lti_max_len | 1024 |
| correction_d_state | 16 (tiny) |
| alpha_init | 0.1 (correction strength) |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257x768, tied) | 38.6M |
| **Per block:** | |
|   LTI: w_in (768->768) | 0.59M |
|   LTI: w_out (768->768) | 0.59M |
|   LTI: H_real + H_imag (12 x 513 x 64 x 2) | 0.79M |
|   LTI: decay_log (12 x 513) | 6.2K |
|   Correction: w_qkv (768->2304) | 1.77M |
|   Correction: w_beta (768->12) | 9.2K |
|   Correction: w_gate (768->768) | 0.59M |
|   Correction: alpha | 1 |
|   SwiGLU FFN | 4.42M |
|   RMSNorm x2 | 1.5K |
|   **Block total** | **~8.75M** |
| **16 blocks** | **140.0M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **~178.6M** |

### Over budget — adjust

At 178.6M, slightly over 175M. Options:
- **Option A:** Reduce to 14 layers. Total: ~161.1M. (**Selected**)
- **Option B:** Reduce ffn_inner to 1536. Total: ~155.4M.

**Selected: Option A (14 layers)** — keeps architectural width, reduces serial depth.

| Revised | Params |
|---------|--------|
| 14 blocks | 122.5M |
| **GRAND TOTAL** | **~161.1M** |

---

## Training

### Single Phase

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Full model (LTI + correction + FFN) | Learn frequency-domain + selective patterns |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 8e-4 cosine -> 8e-5, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 24x1024, accum=2 (48K effective) |
| Precision | fp16 mixed (fp32 for FFT operations) |
| Grad clip | 1.0 |
| H init | small random (N(0, 0.01)) |
| decay_log init | linspace(-1, -4) (moderate to strong decay) |
| alpha init | 0.1 (correction starts weak) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| FFT requires fp32 (no fp16 FFT in PyTorch) | MEDIUM | FFT is O(N log N) = 10K FLOPs for N=1024. Even in fp32, this is tiny vs matmuls. Cast input/output at boundaries. |
| LTI backbone too weak for language (not periodic enough) | HIGH | The correction layer handles non-periodic patterns. LTI captures structure; correction captures meaning. If LTI useless, alpha grows to dominate and SPECTRUM degenerates to pure EFLA. |
| Learned transfer function H(w) overfits to sequence length | MEDIUM | H(w) is defined in frequency space which is length-agnostic. For variable lengths, zero-pad to max_len before FFT. |
| rocFFT performance on gfx1151 unknown | MEDIUM | rocFFT is part of ROCm stack and should work. Size 1024 is standard. If slow, batch FFTs across heads. |
| Causal masking in frequency domain is approximate | HIGH | True causal convolution requires one-sided kernel. The decay factor approximates this. For exact causality, use causal convolution in time domain (but loses FFT advantage). |

## Success Criteria

1. Val loss < 2.95 on BabyLM (competitive with TEMPEST)
2. Throughput > 18K tok/s (FFT parallelism advantage)
3. LTI backbone captures meaningful patterns (visualize H(w) — should show structure)
4. alpha grows from 0.1 to meaningful value (correction is useful, not dominant)
5. Length generalization: maintain quality at 2x training length (frequency-domain advantage)
6. Ablation: SPECTRUM > pure EFLA (same params) — validates LTI+correction decomposition

---

## Implementation Roadmap

1. Implement LTIBackbone with learnable H(w) and causal decay
2. Implement SelectiveCorrection with tiny EFLA (d_state=16)
3. Assemble SpectrumBlock (LTI + correction + SwiGLU FFN)
4. Assemble SpectrumModel (14 layers)
5. Verify parameter count (~161M)
6. Smoke test: 10 min, verify FFT + EFLA work together, loss decreasing
7. Profile: rocFFT performance on gfx1151 at size 1024
8. Full training: BabyLM 2 epochs
9. Visualize H(w): what frequencies does each head learn?
10. Ablation: LTI-only vs correction-only vs both
11. Length generalization: train on 512, eval on 1024 and 2048

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), chunked_linear_cross_entropy

**External (2):** FLA DeltaNet kernel (correction EFLA), rocFFT (system library)

**New (0):** All LTI operations are standard torch.fft.rfft/irfft + element-wise multiply. No custom kernels needed.

### FFT Performance Estimate

```
rocFFT for N=1024, batch=B*H=24*12=288:
  FFT: O(N log N) = 1024 * 10 = 10K complex FLOPs per sequence per head
  Total: 288 * 10K = 2.88M FLOPs
  At 59.4 TFLOPS: ~0.05 microseconds (negligible)
  
  Actual wall-clock (memory-bound): ~0.1ms per FFT batch
  Plus IFFT: ~0.1ms
  Total FFT overhead: ~0.2ms per layer
```

FFT is negligible compared to matmuls (~3ms per layer for QKV + FFN).

### Parallelism Advantage

```
Standard Mamba/EFLA: O(N) sequential scan per layer
  With chunk_size=64: 1024/64 = 16 chunks, but intra-chunk is sequential
  Bottleneck: sequential portion = O(chunk_size) per chunk

SPECTRUM LTI: O(N log N) fully parallel (FFT)
  No sequential bottleneck
  Correction: O(N) but tiny d_state=16 (4x smaller than EFLA d_state=64)
  Correction sequential portion: ~25% of standard EFLA
  
Net: ~75% of mixer computation is fully parallel (LTI via FFT)
     ~25% is sequential (tiny correction)
```

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Eager fp16 | 14L | ~10K tok/s |
| + autokernel | | ~14K tok/s |
| + compile | | ~19K tok/s |
| + FLA (correction) | | **~21K tok/s** |

**Estimated optimized throughput:** ~18-21K tok/s
**Tokens in 45 min:** ~49-57M (3.0-3.5 BabyLM epochs)
**Ranking:** #5-6 of all architectures

Moderate throughput, but the true value is the HIGHEST PARALLELISM of any recurrent design — important as we scale to larger batches and longer sequences.
