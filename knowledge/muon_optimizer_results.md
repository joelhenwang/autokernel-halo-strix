# Muon Optimizer — Results & Configuration (2026-04-13)

## What is Muon?

Muon (MomentUm Orthogonalized by Newton-schulz) replaces AdamW's second moment
with gradient orthogonalization via 5 Newton-Schulz iterations. Published by
KellerJordan (2024) and scaled by Moonshot AI (Moonlight paper, 2025). Kimi K2
used MuonClip to pretrain 1T params on 15.5T tokens with zero loss spikes.

## Algorithm

For each 2D weight matrix per step:
1. Momentum: `buf = 0.95 * buf + grad`
2. Nesterov: `g = grad + 0.95 * buf`
3. Newton-Schulz (5 iterations in bf16): `g = UV^T` (nearest semi-orthogonal matrix)
4. Scale: `g *= sqrt(max(m,n)) * 0.2` (built-in muP)
5. Update: `W -= lr * (g + wd * W)` (decoupled weight decay)

Non-2D params (embeddings, norms, biases, SSM params) use standard AdamW internally.

## Parameter Routing on gfx1151

Only standard MLP/FFN weights go to Muon. SSM/conv/special params stay on AdamW.

**Muon-eligible:** Linear weights in FFN (w_gate_up, w_down), attention projections (wq, wk, wv, wo), output projections
**AdamW-forced patterns:** ssm, mamba, conv, scan, A_log, dt_, D_param, target, film, embedding, embed, output.weight, log_gamma, log_eta, log_beta, omega, gamma_param, decay, conductor, engram, meta_token

### Parameter split examples:
- LlamaModel 124.7M: 60 Muon + 27 AdamW
- AMADEUS 243.8M: 48 Muon + 244 AdamW

## A/B Comparison (10-min budget, BabyLM, batch=16, block=256, accum=4)

### LlamaModel 124.7M (compile + autokernel)

| Optimizer | tok/s | MFU | Steps | Best Loss | Memory |
|-----------|-------|-----|-------|-----------|--------|
| AdamW (lr=8e-4) | **49,711** | 62.6% | 1,004 | 17.58 | 2.6 GB |
| Muon (lr=5e-3) | 48,131 | 60.6% | 1,004 | **17.48** | 2.6 GB |

- Throughput overhead: **3.2%** (Newton-Schulz on 60 params)
- Loss improvement: **0.6%** (marginal after 1 epoch)

### AMADEUS 243.8M (autokernel, no compile)

| Optimizer | tok/s | MFU | Steps | Best Loss | Memory |
|-----------|-------|-----|-------|-----------|--------|
| AdamW (lr=8e-4) | **9,304** | 22.9% | 340 | 14.93 | 9.2 GB |
| Muon (lr=5e-3) | 8,888 | 21.9% | 325 | **14.77** | **9.0 GB** |

- Throughput overhead: **4.5%** (NS on 48 params, larger model)
- Loss improvement: **1.1%** (despite 4.4% fewer steps!)
- Memory savings: **0.2 GB** (fewer optimizer state buffers)
- 2 non-finite grad skips (steps 224, 278) — minor instability

Note: displayed loss is inflated 4x by accum_steps=4 logging artifact. Real loss ≈ displayed / 4.

## Critical Configuration Notes

### LR Scale
- AdamW: lr=8e-4 (standard)
- Muon: lr=0.005 (NOT the standard 0.02 from the paper)
- Standard Muon lr=0.02 causes divergence on SSM models (state norms explode to 2.7x)
- Lower lr=0.005 stabilizes training while preserving convergence advantage

### Do NOT torch.compile the Newton-Schulz function
Each unique weight shape triggers a separate compilation graph. With ~50-85 unique shapes,
this causes 29 GB memory blowup (vs 9 GB baseline). The NS function is only 5 small matmuls
per param — eager overhead is <1% of total training time.

### Memory overhead
- NS creates bf16 temporaries per param (largest: ~10 MB for 1024×5120 weight)
- Momentum buffers: 1 per Muon param (vs 2 for AdamW) — net savings
- Total: slightly lower than AdamW for same model

### GradScaler compatibility
Muon works with PyTorch's GradScaler (fp16 mixed precision):
1. scaler.unscale_(optimizer) — unscales all gradients
2. clip_grad_norm_ — clips all gradients  
3. scaler.step(optimizer) — calls Muon.step() which routes internally

### LR Scheduler compatibility
LambdaLR cosine schedule works across all param groups. Muon groups get
`0.005 * lambda(step)`, AdamW groups get `8e-4 * lambda(step)`. Warmup and
cosine decay apply identically to both.

## Implementation Files

| File | Purpose |
|------|---------|
| `halo_training/muon.py` | Muon optimizer class (~200 lines) |
| `halo_training/optimizer.py` | Factory: `build_optimizer(use_muon=True)` |
| `halo_training/cli.py` | `--muon` flag |
| `halo_training/trainer.py` | Passes `use_muon` through |
| `halo_training/smoke.py` | Smoke test support |

## What We Skipped

**MuonClip (QK weight clipping):** Only needed at multi-billion scale for attention logit
explosion. At 160-250M with SSM models, not relevant.

**Distributed Muon:** Our single-GPU setup doesn't need the all-reduce variant.

## Next Steps

- Test with bf16 (eliminates GradScaler, may fix transient grad norm issues)
- Longer training runs (2 epochs on BabyLM) to quantify convergence advantage
- GPT-training-small (111M tokens) where token-efficiency matters more
- LR sweep (0.003, 0.005, 0.01) to find optimal Muon LR for SSM models
