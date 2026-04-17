---
title: "MEDUSA"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/JORMUNGANDR.md
  - mad_llm_scientist/plans/EREBUS.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %medusa, %diffusion, %block-diffusion, %looped, %parallel-decode]
---

# MEDUSA

**Block Diffusion via Looped Denoiser — The Gaze That Reveals Many Futures at Once**

*"Medusa's gaze turns one moment into many — where autoregression sees one token, the denoiser sees a block of futures crystallizing in parallel. Each loop iteration sharpens the vision."*
*Block Diffusion proved parallel generation viable. Looped-as-Diffusion proved loops ARE denoising.*

## Hypothesis

"Efficient Parallel Samplers for Recurrent-Depth Models" (Oct 2025) proved that **looped transformers are naturally continuous causal diffusion LMs** — loop iterations correspond to denoising steps. This means JORMUNGANDR-HALO's loop IS already a diffusion process, we just haven't exploited it. By adding a block-diffusion training objective alongside NTP, we can generate 4-8 tokens per forward pass at inference without any architectural change. Block Diffusion (March 2025, 77 upvotes, 995 GitHub stars) provides the practical framework. The dual NTP + denoising objective creates a richer training signal that should improve quality even in standard autoregressive mode.

**Key papers:** "Block Diffusion" (2503.09573, March 2025), "Efficient Parallel Samplers for Recurrent-Depth Models" (2510.14961, Oct 2025), "dLLM" (2602.22661, Feb 2026)

---

## Architecture

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  -> 1 SHARED BLOCK x N iterations:
  |     RMSNorm
  |     +---------------------------------------------+
  |     | EFLA Token Mixer (from EREBUS)               |
  |     |   Chunk-wise parallel delta rule (C=64)      |
  |     +---------------------------------------------+
  |     +Residual
  |     RMSNorm -> SwiGLU FFN (768->1920->768) -> +Residual
  |
  |     Training: dual objective at each iteration:
  |       Objective 1: Standard NTP (final iteration only)
  |       Objective 2: Block denoising (all iterations):
  |         noise_level = (1 - i/N) * sigma_max
  |         noised = clean_block + noise * epsilon
  |         loss += MSE(predicted_block, clean_block) * w_denoise
  |
  |     Inference: parallel block generation
  |       1. Initialize: corrupt next B=8 positions with noise
  |       2. Loop N iterations (denoising)
  |       3. Accept tokens with confidence > threshold
  |       4. Advance by accepted count (variable block size)
  |
  -> Final RMSNorm -> LM Head (NTP) + Denoise Head (block prediction)
```

### The Loop-Diffusion Connection

A looped model with N iterations naturally maps to a diffusion process:
```
Iteration 1:  heavy noise -> rough block prediction  (t=T, maximal noise)
Iteration 4:  medium noise -> refined prediction     (t=T/2)
Iteration 8:  light noise -> near-final prediction   (t=0, minimal noise)
Iteration 12: clean -> standard NTP prediction       (denoised)
```

This is not a metaphor — it's mathematically exact. The paper proves that the implicit score function of the looped model satisfies the diffusion SDE.

### Block Diffusion Adaptation

Block Diffusion generates fixed-size blocks of B tokens via discrete denoising. We adapt this for our continuous-latent setting:

1. **During training:** At each iteration i, take the next B=8 positions, add noise proportional to `(1 - i/N)`, and predict the clean tokens. The noise schedule is cosine (following Block Diffusion).

2. **During inference:** Initialize next B positions with Gaussian noise in embedding space. Run N iterations. Each iteration refines the predictions. Accept tokens whose softmax confidence exceeds a threshold.

3. **Effective throughput:** If average accepted block size is K tokens, inference throughput = `base_throughput * K`.

---

## Component 1: Dual-Objective Training Head

```python
class DualHead(nn.Module):
    def __init__(self, d_model=768, block_size=8):
        self.ntp_head = nn.Linear(d_model, 50257, bias=False)
        self.denoise_head = nn.Linear(d_model, d_model, bias=False)
        self.block_size = block_size
        self.noise_schedule = CosineNoiseSchedule()

    def training_loss(self, h, targets, embeddings, iteration, max_iterations):
        ntp_loss = F.cross_entropy(
            self.ntp_head(h).view(-1, 50257), targets.view(-1)
        )

        noise_level = self.noise_schedule(1.0 - iteration / max_iterations)
        B, T, D = h.shape
        if T > self.block_size:
            clean_block = embeddings[:, 1:self.block_size+1]
            noise = torch.randn_like(clean_block) * noise_level
            noised_block = clean_block + noise
            predicted_clean = self.denoise_head(h[:, :self.block_size])
            denoise_loss = F.mse_loss(predicted_clean, clean_block)
        else:
            denoise_loss = torch.tensor(0.0, device=h.device)

        return ntp_loss + 0.1 * denoise_loss

    def inference_generate(self, h, n_tokens=8, confidence_threshold=0.8):
        logits = self.ntp_head(h[:, -1:])
        probs = F.softmax(logits, dim=-1)
        confidence, tokens = probs.max(dim=-1)
        accepted = confidence > confidence_threshold
        return tokens, accepted
```

## Component 2: Block Denoiser

```python
class BlockDenoiser(nn.Module):
    def __init__(self, d_model=768, block_size=8):
        self.block_size = block_size
        self.noise_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model)
        )

    def add_noise(self, clean_embeddings, noise_level):
        noise = torch.randn_like(clean_embeddings) * noise_level
        return clean_embeddings + noise

    def forward(self, h, noise_level):
        noise_cond = self.noise_embed(noise_level.unsqueeze(-1))
        return h + noise_cond
```

## Component 3: Medusa Looped Model

```python
class MedusaModel(nn.Module):
    def __init__(self, d_model=768, n_iterations=12, block_size=8):
        self.embedding = nn.Embedding(50257, d_model)
        self.shared_block = ErebusBlock(d_model)  # Reuse EREBUS block
        self.block_denoiser = BlockDenoiser(d_model, block_size)
        self.dual_head = DualHead(d_model, block_size)
        self.n_iterations = n_iterations
        self.block_size = block_size
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = self.dual_head.ntp_head
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, targets=None):
        h = self.embedding(input_ids)
        embeddings = h.clone()

        n_iter = self.n_iterations
        if self.training:
            n_iter = min(max(
                torch.poisson(torch.tensor(float(self.n_iterations))).int().item(),
                8), 16)

        total_loss = 0.0
        for i in range(n_iter):
            if self.training:
                noise_level = (1.0 - i / n_iter)
                h = self.block_denoiser(h, torch.tensor(noise_level, device=h.device))
            h = self.shared_block(h, iteration=i)

            if self.training and targets is not None:
                iter_loss = self.dual_head.training_loss(
                    self.final_norm(h), targets, embeddings, i, n_iter
                )
                total_loss = total_loss + iter_loss / n_iter

        logits = self.lm_head(self.final_norm(h))
        if self.training:
            return logits, total_loss
        return logits
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
| block_size_denoise | 8 (tokens per denoising block) |
| noise_schedule | cosine |
| denoise_loss_weight | 0.1 |
| confidence_threshold | 0.8 (inference acceptance) |
| chunk_size | 64 |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257x768, tied) | 38.6M |
| **Shared block (EREBUS-style):** | **~7.38M** |
| Block denoiser: noise_embed | 0.15M |
| Dual head: denoise_head (768->768) | 0.59M |
| Final RMSNorm | 768 |
| **GRAND TOTAL (unique)** | **~46.7M** |
| **Effective params (12 iterations)** | **~127M effective** |

Only ~0.74M params added over EREBUS for the denoising machinery. The core looped block is identical.

---

## Training

### Two Phases

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (60%) | 27 min | NTP only (standard looped training) | Stabilize base model |
| 2 (40%) | 18 min | NTP + denoise dual objective | Learn block denoising |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-3 cosine -> 1e-4, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 32x1024, accum=2 (64K effective) |
| Precision | fp16 mixed + fp32 EFLA state |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 iterations |
| denoise_weight | 0.0 (Phase 1) -> 0.1 (Phase 2, linear ramp) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Dual objective hurts NTP quality | HIGH | Phase 1 trains NTP-only first. denoise_weight=0.1 is conservative. Ablation: denoise weight sweep. |
| Block diffusion at 170M too small for quality parallel decode | HIGH | SDLM shows quality improves with scale. At 170M, block_size=4 may be more realistic than 8. |
| Continuous-latent denoising less effective than discrete | MEDIUM | Block Diffusion works in discrete space. Our continuous-latent variant may be less sharp. Mitigation: project to logits at each iteration. |
| Inference block acceptance rate too low | MEDIUM | If confidence threshold too high, effective block_size=1 (standard AR). Tune threshold down to 0.6 for higher acceptance. |
| Noise conditioning interacts poorly with loop state | MEDIUM | Block denoiser adds noise_level to hidden state. If disruptive, switch to FiLM modulation (scale+shift). |

## Success Criteria

1. Val loss < 2.95 on BabyLM (NTP quality preserved despite dual objective)
2. Training throughput > 32K tok/s (dual objective adds ~15% overhead)
3. Inference: average accepted block size > 3 tokens (3x effective speedup)
4. Inference effective throughput > 100K tok/s (base 38K x 3 block)
5. Dual-objective model >= NTP-only model on val loss (denoising helps!)
6. No quality regression on standard benchmarks when using AR-only inference

---

## Implementation Roadmap

1. Start from EREBUS implementation (EFLA looped block)
2. Implement BlockDenoiser with cosine noise schedule
3. Implement DualHead with NTP + denoise objectives
4. Assemble MedusaModel with dual-phase training
5. Verify parameter count (~47M unique)
6. Phase 1: train NTP-only, verify EREBUS-level quality
7. Phase 2: enable dual objective, tune denoise_weight
8. Implement block-parallel inference with confidence thresholding
9. Measure: average accepted block size, effective inference throughput
10. Ablation: block_size {4, 8, 16}, denoise_weight {0.05, 0.1, 0.2}
11. Compare: AR-only inference vs block-parallel inference (quality + speed)

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), chunked_linear_cross_entropy

**External (2):** causal-conv1d (10x), FLA DeltaNet kernel (EFLA)

**New (0):** Noise conditioning is element-wise. MSE loss is standard. No new kernels needed.

### Training Overhead

| Component | Additional cost vs EREBUS |
|-----------|--------------------------|
| Block denoiser noise conditioning | ~0.01ms (element-wise, free) |
| Denoise head forward (768->768) | ~0.1ms (one small matmul) |
| MSE loss computation | ~0.01ms (element-wise, free) |
| **Total additional per iteration** | **~0.12ms** |
| **Total additional for 12 iterations** | **~1.4ms** |
| **Relative overhead** | **~3-5%** |

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| Training (NTP-only Phase 1) | compile + AK | ~38K tok/s |
| Training (dual Phase 2) | compile + AK | ~35K tok/s |
| Inference (AR) | compile + AK | ~38K tok/s |
| **Inference (block, avg K=4)** | compile + AK | **~150K effective tok/s** |
| **Inference (block, avg K=8)** | compile + AK | **~300K effective tok/s** |

**Estimated training throughput:** ~35-38K tok/s
**Estimated inference throughput:** ~150-300K effective tok/s (block-parallel, **the highest of any hypothesis**)
**Tokens in 45 min:** ~95-103M (5.9-6.4 BabyLM epochs)
**Ranking:** Training #3-4, Inference #1 by a wide margin
