# RESONANT LOOP

**Cache-Resident Iterative Shared Block with Adaptive Halting**

## Hypothesis

A single shared transformer-like block (7.3M params), iterated 8–16 times per token with **per-token adaptive halting** and **SCORE-style damped residuals**, can match GPT-2 124M quality at ~59M unique parameters (~140M effective). In int4, the block (3.7 MB) fits entirely in L2 cache (6 MB) — iterations 2–16 are free from a memory-bandwidth perspective.

**Inspired by:** Universal Transformers (weight sharing), SCORE (ODE residual h=(1-d)·h+d·F(h)), AdaPonderLM (adaptive depth), Mixture-of-Depths (token-dependent compute).

**Novel contribution:** Weight sharing + adaptive halting + L2 cache exploitation on memory-bound hardware. No attention, no recurrent state.

---

## Architecture

```
Tokens → Embedding (tied LM head)
  → Resonant Loop (iterate 1..N):
      h += iter_emb[i]                    # depth conditioning
      h_out = SharedBlock(h)              # conv + mixer + SwiGLU
      h = (1-d)·h + d·h_out              # SCORE damped residual
      if cumulative_halt ≥ 1.0: break     # ACT early exit (inference only)
  → Final RMSNorm → LM Head

SharedBlock:
    RMSNorm → Depthwise Conv1d(k=4) → Channel Mixer(1024→1024) → +Residual
    RMSNorm → SwiGLU FFN(1024→2048→1024) → +Residual
```

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| ffn_inner | 2048 (SwiGLU, 2× ratio) |
| conv_kernel | 4 (depthwise causal) |
| max_iterations | 16 |
| SCORE d_init | sigmoid(-2.2) ≈ 0.1 (learned) |
| λ_ponder | 0.01 (ACT regularization) |
| vocab_size | 50257 |
| block_size | 1024 |
| **Unique params** | **~58.8M** |
| **Effective params (16 iter)** | **~168M** |
| **Model size int4** | **~37 MB** |

## Parameter Breakdown

| Shared Block (single copy) | Params |
|---------------------------|--------|
| RMSNorm (pre-conv) | 1,024 |
| Depthwise conv1d (k=4) + bias | 5,120 |
| Channel mixer (1024×1024) + bias | 1,049,600 |
| RMSNorm (pre-FFN) | 1,024 |
| SwiGLU (gate + up + down) | 6,292,768 |
| **Block total** | **7,349,248 (~7.3M)** |

| Per-iteration (NOT shared) | Params |
|---------------------------|--------|
| Iteration embeddings (16 × 1024) | 16,384 |
| Halting gates (16 × (1024+1)) | 16,400 |
| **Subtotal** | **32,784** |

| Global | Params |
|--------|--------|
| Embedding (50257 × 1024, tied) | 51,463,168 |
| SCORE damping scalar | 1 |
| Final RMSNorm | 1,024 |
| **Grand total** | **~58.8M** |

## Key Mechanisms

### SCORE Damped Residual

```python
d = sigmoid(self.damping)   # learned, init ≈ 0.1
h = (1 - d) * h + d * block(h + iter_emb[i])
```

At init, each iteration changes h by only 10%. Prevents instability. The model learns to increase d if larger steps help.

### Adaptive Halting (ACT)

Per-iteration halting gates (NOT shared — each iteration needs different stopping criteria):

```python
p_halt = sigmoid(W_halt[i] @ h + b_halt[i])   # per token
cumulative_halt += p_halt
if cumulative_halt >= 1.0: break               # inference: real early exit
```

**Halting bias init:** iterations 0–3: b=-2.0 (unlikely halt), 4–7: b=-0.5, 8–11: b=0.0, 12–15: b=+1.0.

**Training:** Always run all 16 iterations. Use ACT weighted output combination + ponder cost: `loss += λ · mean(cumulative_halt)`.

**Inference:** Actually break early. Easy tokens halt at 4–6 iterations, hard tokens use all 16.

### Iterated Conv Receptive Field

Conv1d k=4, iterated N times through residual: effective receptive field = 3N+1. At N=16: 49 tokens. The residual stream propagates information beyond direct conv reach.

## L2 Cache Analysis

| Precision | Block Size | Fits in L2 (6MB)? |
|-----------|-----------|-------------------|
| fp16 | 14.7 MB | No (41% cache hits on iter 2+) |
| int4 | **3.7 MB** | **Yes (0.61× L2)** |

**Int4 decode:** Iteration 1 reads 3.7 MB from DRAM → block enters L2 → iterations 2–16 read 0 bytes from DRAM.

**Critical:** The LM head (50257×1024) reads ~25.7 MB int4 / ~103 MB fp16 every token regardless. This dominates DRAM traffic in int4 mode.

## Training

### Phase Training

1. **Phase 1 (70% of steps):** Fixed 12 iterations, no halting. Block learns text processing.
2. **Phase 2 (30% of steps):** Enable ACT. Halting gates learn when to stop.

### Pseudocode

```python
h = embed(tokens)
d = sigmoid(self.damping)
output, cumulative_halt = zeros_like(h), zeros(B, T)

for i in range(16):
    h = (1-d)*h + d*shared_block(h + iter_emb[i])
    p_halt = sigmoid(halt_gates[i](h))
    # ACT accumulation: weighted output based on halting probabilities
    ...
logits = lm_head(final_rmsnorm(output))
loss = CE(logits, targets) + λ_ponder * cumulative_halt.mean()
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Dataset | OpenWebText ~100M tokens |
| Batch | 48 × 512 = 24K tok, grad_accum=2 (48K effective) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| LR | 8e-4 cosine → 8e-5, warmup 100 steps |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 iterations |
| Est. throughput | ~60M tokens in 15 min |

## Decode Speed (Strix Halo)

| Mode | 16 iterations | Adaptive (avg 10) |
|------|--------------|-------------------|
| fp16 | ~526 tok/s | **~694 tok/s** |
| int4 | ~1299 tok/s | **~1613 tok/s** |

## HIP Kernels

**Reuse:** `fused_residual_add_rmsnorm` (6.6×), `silu_gate_mul` (1.6×), `cross_entropy` (1.8×), `dequantize_int4` (16.3×).

**New (priority order):**
1. **Fused SCORE Residual + RMSNorm** — adapt `fused_residual_add_rmsnorm`: replace `h+residual` with `(1-d)*h + d*h_out`. HIGH priority.
2. **Fused Conv1d Step + Channel Mixer** (inference) — ring buffer update + conv + linear. MEDIUM.
3. **Halting Gate + Early Exit** (inference) — projection + sigmoid + comparison + break. LOW.

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Conv receptive field too limited (49 tok) | HIGH | Residual stream propagation; fallback: add 1 local attention layer after all iterations |
| Shared weights receive 16× gradients | HIGH | SCORE damping limits influence; grad clip 1.0; phase training |
| Halting doesn't learn meaningful stopping | MEDIUM | Tune λ_ponder; halting bias init; phase training decouples learning |
| 7.3M per "effective layer" capacity | MEDIUM | 16 iterations = 117M effective; increase ffn_inner to 2560 if needed |
| Sequential training (no pipeline) | MEDIUM | Block is tiny = fast per-iter; gradient checkpointing every 4 |

## Success Criteria

1. Loss < 4.5 in 15 min
2. ≥30% of tokens halt before iteration 12
3. Decode > 1000 tok/s fp16 on Strix Halo
4. Per-token efficiency ≥ GPT-2 124M

## Implementation Roadmap

1. PyTorch model: SharedBlock + ResonantLoop + ACT bookkeeping
2. Verify param count, gradient flow, shapes
3. Train with fixed 12 iterations (no ACT), log loss
4. Add ACT (phase 2), monitor halting distributions
5. Fused SCORE residual + RMSNorm HIP kernel
6. Decode benchmark (fixed + adaptive)
7. Int4 quantization + L2 cache benchmark
8. Compare to GPT-2 and Spectral Hydra

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x) — adaptable for SCORE damping: replace `h+residual` with `(1-d)*h + d*h_out`
- **silu_gate_mul** (1.6x) — applies to SwiGLU FFN in shared block
- **cross_entropy** (1.8x) — loss computation
- Apply via `autokernel.optimize(model, training=True)`

### No Scan Needed
SCORE damping is a simple lerp: `(1-d)*h + d*block(h)`. The depthwise conv1d (k=4) iterated N times gives effective receptive field = 3N+1 = 49 tokens at N=16. No associative scan or recurrence scan required — this is a key advantage.

### L2 Cache Advantage (Unique to This Architecture)
- Shared block: 7.3M params in int4 = **3.7 MB → fits in L2 (6 MB)**
- After iteration 1, subsequent iterations read weights from L2 cache (~10x faster than DRAM)
- This is the only plan that can achieve near-100% L2 cache hit rate for weight reads
- `torch.compile` should fuse the iteration loop for maximum benefit

### Throughput Estimates (CORRECTED)
- Previous estimate "~60M tokens in 15 min" is unrealistic
- With 16 iterations through the shared block, each forward pass is ~16x a single-block forward
- **Eager mode:** ~3-4K tok/s → 2.7-3.6M tokens in 15 min
- **With torch.compile** (fuses iteration loop): potentially 8-10K tok/s → 7.2-9M in 15 min
- **torch.compile is critical for this architecture** — without it, the Python iteration loop dominates

### Effective Parameter Count (CORRECTED)
- "~168M effective" is misleading. Weight sharing means each parameter gets 16x gradient signal, but representational capacity is still ~59M
- **Correct framing:** "59M unique params with 16x gradient amplification" — NOT "168M effective"
- Quality will be closer to a 70-90M model than a 168M model

### Decode Speed (CORRECTED)
- LM head (103 MB fp16 / 26 MB int4) dominates decode — block is L2-cached
- **fp16:** 103 MB / 240 GB/s + 16 × ~0.01ms (cached block) ≈ 0.59ms → **~1700 tok/s**
- **int4 (adaptive, avg 10 iter):** 26 MB / 240 GB/s + 10 × ~0.003ms ≈ 0.14ms → **~7000 tok/s**
- Previous estimates (526/1299) were too conservative — LM head latency was overestimated

### MFU: 70-80% training (high due to L2 caching after first iteration)
