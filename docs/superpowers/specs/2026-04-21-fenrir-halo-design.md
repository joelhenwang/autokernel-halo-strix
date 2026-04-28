# FENRIR-HALO: Clean-Sheet Parcae Architecture for Strix Halo DDP

**Date:** 2026-04-21
**Status:** Design
**Target:** Beat Portimbria-150M (HellaSwag 27.5%, ARC-E 33.8%) and compete with SmolLM2-135M

---

## Architecture

```
FactorizedEmbed(50257, rank=256, d=640)
  |
Prelude: 1 XSA-GQA(640, 10h/2kv, head_dim=64, ffn=2304)
  |
+--- SimpleParcaeInjection(640) ---+
|                                   |
|  CORE BLOCK (10 unique layers):   |
|    L0-L3: ShortConvBlock(640)     |  4 conv (local)
|    L4:    XSA-GQA(640, 10h/2kv)  |  1 attn (global @ 40%)
|    L5-L8: ShortConvBlock(640)     |  4 conv (local)
|    L9:    XSA-GQA(640, 10h/2kv)  |  1 attn (global @ 100%)
|                                   |
|  Loop: Poisson(mean=3), bptt=2    |
|  DepthMemoryCache over iterations |
|  Per-zone compile per layer       |
+-----------------------------------+
  |
Coda: 1 XSA-GQA(640, 10h/2kv, head_dim=64, ffn=2304)
  |
RMSNorm(640) -> FactorizedLMHead(640, rank=256)
```

### Dimensions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 640 | 5x128, Tensile-aligned |
| n_heads | 10 | head_dim=64 |
| n_kv | 2 | 5:1 GQA, aggressive KV compression |
| ffn_inner | 2304 | 3.6x d, 9x256 aligned |
| d_conv | 4 | Short conv kernel size |
| vocab | 50257 | GPT-2 tokenizer |
| embed_rank | 256 | Factorized embedding |
| ctx | 1024 | Production context |
| rope_theta | 10000 | Standard RoPE |

### Parameter Budget

| Component | Params |
|-----------|--------|
| FactorizedEmbed(50257, 256, 640) | 13.03M |
| FactorizedLMHead(640, 256) | 0.16M |
| Prelude: 1 XSA-GQA | 5.40M |
| Core: 8 ShortConvBlock(640, 2304) | 45.28M |
| Core: 2 XSA-GQA(640) | 10.80M |
| Coda: 1 XSA-GQA | 5.40M |
| SimpleParcaeInjection(640) | ~1K |
| DepthMemoryCache | ~32K |
| Norms | ~1K |
| **Total unique** | **~80M** |
| **Parcae-equivalent (2x)** | **~160M** |

### Mechanisms (Proven Only)

| Mechanism | Effect | Cost | Source |
|-----------|--------|------|--------|
| XSA | -0.9% loss | 0 params | Apple, measured in JORMUNGANDR-HALO |
| Depth Memory Cache | -2.5% loss | 32K params | Measured in JORMUNGANDR-HALO |
| Factorized Embeddings | Saves ~19M params | Tiny compute | Nandi-150M |
| Parcae Loop (Poisson) | 2x param efficiency | Loop overhead | Parcae paper |
| 75:25 Conv:Attn | Efficient hybrid | N/A | LFM2 |

**Excluded (no measured benefit at ctx<=1024):** FiLM, Value Embeddings, TTT, multi-step TTT.

---

## Training Configuration

### Optimizer

| Group | Optimizer | LR | WD | Notes |
|-------|-----------|-----|-----|-------|
| core_block.* (2D) | Muon | 0.01 | decaying | 0.5x base (loop accumulates grads) |
| prelude.*, coda.* (2D) | Muon | 0.02 | decaying | Standard |
| *log_A*, *log_B* | AdamW | 8e-5 | 0 | 0.1x, stability-critical |
| *depth_cache* | AdamW | 8e-4 | 0 | Small module |
| *norm*, *bias | AdamW | 8e-4 | 0 | Standard |
| *embed* | AdamW | 8e-4 | 0 | Factorized embed weights |

### Schedule

| Setting | Value |
|---------|-------|
| LR schedule | Cosine with linear warmup |
| Warmup | 2% of total steps |
| Muon momentum | 0.85 -> 0.95 (warmup 300 steps) |
| WD schedule | Linear decay to 0 |
| Batch | 16 seqs x 1024 ctx = 16K tokens/step |
| Accum steps | 8 |
| Effective batch | 16 x 8 x 2 machines = 256 seqs = 262K tokens/step |
| Grad clip | 1.0 |

### Parcae Loop

| Setting | Value |
|---------|-------|
| Depth sampling | Poisson(mean=3) |
| BPTT depth | 2 (last 2 iters get gradients) |
| Curriculum | `1 - sqrt(1 - progress)` ramp |
| Detached iters | All but last 2 use torch.no_grad() |
| Injection | `h = A*h + B*input_embed` (skipped at iter 0) |
| A constraint | `A = -exp(log_A)`, eigenvalues in (-1, 0) |

### Precision & Compile

| Setting | Value |
|---------|-------|
| Precision | fp16 + GradScaler (NOT bf16) |
| Compile | Per-zone: model.compile_zones() |
| Autokernel | Yes (d=640 > 256 threshold) |
| Mode | compile mode="default" (not reduce-overhead) |

---

## Data Pipeline

**CLIMB-optimized mixture** from Dolma/FineWeb:

1. Stream + sample from Dolma 100B
2. Embed with MiniLM (CPU)
3. Cluster K=16 (FAISS)
4. Proxy search with FENRIR-HALO-Mini (tiny variant)
5. Assemble pre-mixed .bin with optimal weights

Target: 12B tokens, pre-tokenized uint16 .bin files.

---

## DDP Training

| Metric | Single Machine | DDP (2x) |
|--------|---------------|----------|
| tok/s (est.) | ~15-17K | ~28-32K |
| Time for 12B tokens | ~8 days | ~4 days |
| MFU (est.) | ~28-32% | ~29-33% |

### DDP Config

- Backend: gloo (optimal for unified memory)
- Interface: Thunderbolt 4 (9 Gbps measured)
- Async overlap: manual allreduce after microsteps
- FP16 gradient compression
- Sync overhead: ~3-6% at accum_steps=8

```bash
# Machine 0 (master)
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --master_addr=10.77.0.1 --master_port=29500 \
    -m halo_training --model models/fenrir_halo.py --class-name FenrirHalo \
    --dataset datasets/datamix_state/phase5_final/train.bin \
    --compile --optimize-kernels --muon \
    --block-size 1024 --batch-size 16 --accum-steps 8 \
    --lr 0.02 --time-budget 5760

# Machine 1 (worker)
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=10.77.0.1 --master_port=29500 \
    -m halo_training --model models/fenrir_halo.py --class-name FenrirHalo \
    --dataset datasets/datamix_state/phase5_final/train.bin \
    --compile --optimize-kernels --muon \
    --block-size 1024 --batch-size 16 --accum-steps 8 \
    --lr 0.02 --time-budget 5760
```

---

## Key Differences from Prior Architectures

| Aspect | JORMUNGANDR-HALO | CHIMERA-HALO | FENRIR-HALO |
|--------|------------------|--------------|-------------|
| Core dim | d=512 (hetero) | d=768 (uniform) | d=640 (uniform) |
| Dim transitions | proj_down/up | None | None |
| Core layers | 3 ShortConv | 8 (6 conv + 2 GQA) | 10 (8 conv + 2 GQA) |
| Attn in core | None | 2 GQA | 2 GQA |
| Layer repeat | None | 2x | None |
| FiLM/TTT/VE | Yes (Full) | No | No |
| Factorized embed | No | Yes (rank=256) | Yes (rank=256) |
| Unique params | 104M | 94M | 80M |
| Parcae-equiv | ~152M | ~158M | ~160M |
| Target tokens | 585M (actual) | Untrained | 12B |
| DDP | No | No | Yes, 2-machine |

---

## Variants

| Class | Config | Use Case |
|-------|--------|----------|
| FenrirHalo | mean_recurrence=3, bptt=2 | Default production training |
| FenrirHaloDeep | mean_recurrence=5, bptt=3 | Quality push |
| FenrirHaloBare | No XSA, no depth cache | Ablation baseline |
| FenrirHaloNoLoop | Single pass, no Parcae | Ablation: loop value |
| FenrirHaloMini | d=128, 4 layers, vocab=50257 | Smoke test + CLIMB proxy |

---

## Success Criteria

### Must Beat (Portimbria-150M, 6B tokens)

| Benchmark | Portimbria | Target |
|-----------|-----------|--------|
| HellaSwag | 27.46% | >30% |
| PIQA | 57.62% | >58% |
| ARC-Easy | 33.80% | >36% |
| Winogrande | 52.72% | >53% |

### Stretch Goal (SmolLM2-135M, 2T tokens)

| Benchmark | SmolLM2 | Target |
|-----------|---------|--------|
| HellaSwag | 42.1% | >35% |
| ARC-Easy | 48.99% | >42% |

---

## Implementation Checklist

1. [ ] Create `models/fenrir_halo.py` (all variants)
2. [ ] Verify forward/backward pass locally
3. [ ] Smoke test on remote (single machine)
4. [ ] Add compile_zones() support
5. [ ] Verify autokernel compatibility
6. [ ] Scale CLIMB pipeline to 12B tokens (Dolma)
7. [ ] Run Phase 4 (API quality scoring)
8. [ ] Assemble pre-mixed .bin
9. [ ] DDP smoke test (2 machines)
10. [ ] Production training run (~4 days)
11. [ ] Eval with lm-evaluation-harness
12. [ ] Compare benchmarks to Portimbria + SmolLM2
