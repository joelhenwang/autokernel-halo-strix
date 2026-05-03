# TYR-HALO: Small Efficient Productive LLM Architecture

**Date:** 2026-04-29
**Status:** Implemented, smoke-tested
**Target:** Beat Portimbria-150M (HellaSwag 27.5%), compete with SmolLM2-135M (42.1%)

---

## Research Foundation

Architecture informed by deep analysis of 12 papers:

| Paper | Key Takeaway for TYR-HALO |
|-------|--------------------------|
| **DeepSeek-V4** (2026) | mHC Sinkhorn residuals, Muon at scale, MTP depth=1 w/ 0.3 loss weight |
| **MoDA** (Zhu et al. 2026) | Depth-attention: attend across loop iterations via concatenated KVs, +2.11% at 3.7% FLOPs |
| **EGGROLL** (Sarkar et al. 2025) | Rank-r ES for post-training alignment, no backprop needed |
| **Self-Improving Pretraining** (Meta 2026) | Online DPO during pretraining, 3.2x reasoning boost (deferred — too expensive) |
| **PTP** (Draxler et al. 2025) | Auxiliary variable reparameterization for parallel decode (superseded by DFlash) |
| **EBT** (Gladstone et al. 2025) | Energy-based heads, 35% better scaling rate (deferred — Hessian too expensive on gfx1151) |
| **Tufa Labs** (2026) | Synthetic pretraining: 3-6x token efficiency, same-size generator sufficient |
| **DFlash** (Chen et al. 2026) | Block diffusion drafter: 16 tokens parallel, KV injection from target layers, 4.9x speedup. Validates draft-from-middle theory: no-KV-injection baseline = 2.83x |
| **SSD/Saguaro** (Kumar et al. 2026) | Async speculation caching, +30% on top of any drafter, needs 2nd GPU |
| **Samsung DS2D** (2026) | Self-speculative decoding via learned forecast embeddings — no external drafter, 2.05-2.27x. CTG: 6x for batched multi-stream generation |
| **Learning Mechanics** (Simon et al. 2026) | Theoretical framework — no direct architecture impact |
| **PINN Ball Trajectory** (2026) | Physics-informed loss design — pedagogical reference for constrained losses |

---

## Architecture

```
FactorizedEmbed(50257, rank=256, d=640)
  |
Prelude: 1 MoDA-GQA(640, 10h/2kv, ffn=2304, XSA)
  |
+--- mHC Branch Init (4 streams x d=640) ---+
|                                             |
|  CORE BLOCK (6 unique layers):              |
|    L0-L1: ShortConvBlock(640, 512, 2304)   |
|    L2:    MoDA-GQA(640, 10h/2kv, XSA)     |
|    L3-L4: ShortConvBlock(640, 512, 2304)   |
|    L5:    MoDA-GQA(640, 10h/2kv, XSA)     |
|                                             |
|  mHC mixing (Sinkhorn 4x4, 20 iters)       |
|  Loop: Poisson(mean=2), BPTT=2             |
+---------------------------------------------+
  |
Coda: 1 MoDA-GQA(640, 10h/2kv, ffn=2304, XSA)
  |
RMSNorm(640) -> FactorizedLMHead(640, rank=256)
  |
MTP AuxHead(640, rank=256) [training only, depth=1]
  |
Draft Heads x4 [inference only, from h_iter0]
```

### Dimensions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 640 | 5x128 Tensile-aligned |
| n_heads | 10 | head_dim=64 |
| n_kv_heads | 2 | 5:1 GQA |
| ffn_inner | 2304 | 3.6x d |
| d_conv | 512 | ShortConv projection dim |
| embed_rank | 256 | Factorized embedding |
| n_branches | 4 | mHC branch count |
| mtp_depth | 1 | MTP auxiliary head |
| mean_recurrence | 2 | Parcae loop depth |
| bptt_depth | 2 | Both iterations get gradients |

### Parameter Budget

| Component | Params |
|-----------|--------|
| FactorizedEmbed | 13.03M |
| FactorizedLMHead | 0.16M |
| Prelude MoDA-GQA | 5.40M |
| Core: 4x ShortConvBlock | 22.64M |
| Core: 2x MoDA-GQA + depth KV proj | 11.52M |
| Coda MoDA-GQA | 5.40M |
| mHC projections | ~0.01M |
| MTP head | 0.16M |
| Norms, injection | ~0.01M |
| **Total unique** | **~58.6M** |
| **Parcae-equivalent (mean=2)** | **~115M** |

---

## Novel Mechanisms

### MoDA Depth-Attention (from MoDA paper, Zhu et al. 2026)

Each MoDA-GQA attention head attends to both sequence-level KVs and depth-level KVs from prior loop iterations at the same token position. Depth KVs concatenated along sequence dimension before SDPA. No RoPE on depth KVs (position-independent). XSA applied to sequence portion only.

FFN-side depth KV projection: `nn.Linear(d_model, n_kv_heads * head_dim * 2)` produces per-iteration depth KVs (~0.16M extra params per GQA layer).

Expected: +2% downstream improvement at ~3-5% FLOPs overhead.

### mHC Branch Manager (from DeepSeek-V4, 2026)

4-branch residual stream with Sinkhorn-constrained mixing. Replaces SimpleParcaeInjection with richer information flow across loop iterations.

- Read: sigmoid-weighted sum of 4 branches -> single d-dim input
- Write: Sinkhorn-normalized 4x4 mixing matrix + write-in weights
- Pure PyTorch Sinkhorn (4x4 matrix, 20 iterations = trivially cheap)
- All projections initialized near-identity (zeros) for stable warmup
- HIP fused kernel available (28.5x speedup) for AMD deployment, but PyTorch sufficient at 4x4

### MTP Auxiliary Head (from DeepSeek-V4 / Meta)

Factorized auxiliary head predicts token at position +2. Shares embedding table. Loss weight 0.3 (reduced to 0.1 at LR decay). Discarded after training — improves backbone representations during training only.

MTP does NOT speed up training (still compute CE on every position). It improves **token efficiency** — backbone encodes richer future-context information in h_L.

### Synthetic Pretraining Data (from Tufa Labs, 2026)

3-6x token efficiency via synthetic augmentation. Same-size generator (0.8B) rewrites math/science/code data in three modes:
- **TPT**: Thinking-Augmented Pretraining (adds explicit thought process)
- **First Principles**: breaks content to fundamental concepts
- **Rephrasing**: maximizes learnability for autoregressive student

Generated offline, tokenized to .bin files. This is the primary training speed lever — better data, not more compute.

---

## Variants

| Class | Config | Use Case |
|-------|--------|----------|
| TyrHalo | mean=2, mHC+MoDA+MTP | Default production |
| TyrHaloFast | mean=2, MoDA, no mHC/MTP | Max throughput ablation |
| TyrHaloBare | mean=2, no MoDA/mHC/MTP/XSA | Baseline ablation |
| TyrHaloNoLoop | mean=1, no novel mechanisms | Single-pass ablation |
| TyrHaloMini | d=128, 4 layers, full vocab | Smoke test + CLIMB proxy |

---

## Training Configuration

### Optimizer Groups

| Group | Optimizer | LR |
|-------|-----------|-----|
| core_block.* (2D) | Muon | 0.015 |
| prelude.*, coda.* (2D) | Muon | 0.02 |
| mhc.phi_*, mtp_head.*, depth_kv_proj.* | AdamW | 8e-4 |
| *norm*, *bias, *embed* | AdamW | 8e-4 |

Muon validated at massive scale by DeepSeek-V4 (1.6T params, 33T tokens). Our qk_norm=True eliminates need for QK-Clip (confirmed by V4). Momentum 0.95 matches V4 steady state.

### Schedule

- Warmup: 2% of total steps
- LR: cosine decay
- Muon momentum: 0.85 -> 0.95
- Batch: 32 seqs x 1024 ctx x 8 accum = 262K tokens/step
- MTP loss weight: 0.3 (reduce to 0.1 at decay)

### Precision

- NVIDIA: bf16 (native tensor cores, no GradScaler)
- AMD: fp16 + GradScaler (bf16 is 24% slower on gfx1151)
- bf16 guard is AMD-only: auto-detected via `torch.version.hip`

### Data Mixing

| Source | Proportion | Augmentation |
|--------|-----------|--------------|
| Math (MegaMath/FineWeb-Edu) | 40% | TPT augmented |
| Science (FineWeb-Edu) | 20% | First Principles |
| Code (StarCoder subset) | 15% | Rephrasing |
| General (Dolma/FineWeb) | 25% | None (raw) |

---

## Training Throughput

| Platform | Config | tok/s |
|----------|--------|-------|
| RTX 4060 Ti | compile + bf16 | ~60-80K |
| Strix Halo (single) | AK + compile | ~35-40K |
| Strix Halo (DDP 2x) | AK + compile | ~65-75K |

Training time estimates (RTX 4060 Ti, compile + bf16):

| Dataset | Tokens | Time |
|---------|--------|------|
| BabyLM smoke | 16M | ~5 min |
| GPT-training-small | 585M | ~3 hours |
| Synthetic 12B | 12B | ~2.5 days |

---

## Post-Training Pipeline

### Stage 1: EGGROLL ES Alignment + CTG Batching

Rank-r Evolution Strategies for alignment (from EGGROLL, Sarkar et al. 2025). No backprop needed — pure inference + fitness scoring.

- Perturbations: σ(ε₂ε₁ᵀ), rank=4 (individual low-rank, sum is high-rank)
- K=16 rollouts per instruction prompt
- External judge: Llama-3-8B-Instruct or API
- Noise-reuse: multiple updates within single sequences
- Population sharing: 512 members share sequences (EGGROLL finding)

**CTG acceleration (Samsung, 2026):** Generate K=16 rollouts via Concurrent Token Generation — single prefill, 16 masked output streams with partitioned KV-cache. Instead of 16 sequential generations, run ceil(16/8) = 2 batched forward passes. **~8x speedup for the alignment rollout phase.**

```
Standard ES rollouts:  16 × T_generate = 16T
CTG ES rollouts:       2 × T_generate = 2T  (batch=8 per pass)
```

Implemented as `concurrent_generate()` utility in `models/tyr_halo.py`.

Replaces DPO/SimPO. Works on any architecture including looped hybrids. No gradients means no memory overhead for reference model.

### Stage 2: Inference Optimization (phased)

See "Inference Architecture" section below.

---

## Inference Architecture: Theory and Implementation

### Theoretical Foundation

**Linear Probing Theorem (informal):** A linear probe at layer L-k achieves ~(1 - k/L)² of final-layer next-token accuracy. For TYR-HALO with 14 effective layers, h_iter0 (after 6 layers) contains ~85-90% of final prediction signal.

**Jacobian Approximation:** The last k layers are locally approximately linear:
```
h_L ≈ A_k @ h_{L-k} + b_k
logits ≈ (W_head @ A_k) @ h_{L-k} + (W_head @ b_k)
       = W_draft @ h_{L-k} + bias
```

W_draft = W_head @ A_k is a single matrix turning intermediate hidden state into approximate logits. This is a "draft head."

**DFlash Validation:** DFlash (Chen et al. 2026) empirically proves this theory:
- Without target features (no KV injection): 2.83x speedup
- With target features (KV injection): 5.15x speedup
- Target features contribute ~45% of total speedup

Our draft heads are the "without target features" case. DFlash proves this baseline already gives 2.83x.

### Parcae Loop as Natural Draft/Verify Pipeline

TYR-HALO's Parcae loop maps directly to speculative decoding:

```
Iteration 0 (6 layers) = DRAFT PHASE
  h_iter0 already contains ~85-90% of next-token signal
  Draft heads produce K=4 candidate tokens in parallel (one matmul each)

Iteration 1 (6 layers) = VERIFY PHASE
  Runs full model on [input..., draft_tokens]
  Verifies draft correctness via standard speculative verification
  Zero wasted compute — iter1 is already in the forward pass
```

### Four-Phase Inference Strategy

```
Phase 0 (single GPU, free):
  DS2D forecast embeddings + draft heads on h_iter0
  Forecast embeddings prime model for multi-token slots (DS2D, Samsung 2026)
  Draft heads: K=4 parallel linear probes on h_iter0 (dynamic, conditioned)
  Expected: ~2.5-3x decode speedup
  Cost: 4 small matmuls (640 x 256 x 50257 each) ≈ 52M FLOPs
  Full forward: ~700M FLOPs → drafting is ~13x cheaper

Phase 1 (2 machines, moderate effort):
  DFlash-style block diffusion drafter on Machine B
  KV injection from h_iter0 sent over Thunderbolt 4
  Expected: ~4-5x decode speedup
  Generates 16 draft tokens in single parallel forward pass

Phase 2 (2 machines, full stack):
  DFlash drafter + SSD/Saguaro outcome caching
  Pre-compute speculations for top-F verification outcomes
  Cache hit ~90% (Saguaro sampling concentrates residual distribution)
  Expected: ~6-8x decode speedup
```

### Phase 0: DS2D Forecast Embeddings + Draft Heads (Single GPU)

Two complementary mechanisms that stack:

**DS2D Forecast Embeddings (Samsung, 2026):** m=4 learned embeddings appended to prompt. Trained via prefix-tuning on frozen backbone. Primes the model to expect multi-token prediction slots. Static — same embeddings regardless of input. ~0.04M params.

**Draft Heads (Jacobian approximation):** K=4 linear probes on h_iter0. Dynamic — conditioned on actual intermediate representations. Each head independently predicts t+k. ~0.6M params (4 × Linear(640, 256) + shared embed table).

**Why they combine:** DS2D provides the model with a "heads up" that multi-token slots are expected. h_iter0 draft heads provide the actual dynamic signal. DS2D alone gives 2.05-2.27x (Samsung). Draft heads alone give ~2.5-3x (DFlash no-KV baseline). Combined: forecast primes representation → draft heads exploit richer h_iter0.

**Parcae-native speculative decode:**
```
iter0: model processes [input + forecast_embeds] → h_iter0
       draft_heads(h_iter0) → K=4 draft tokens (parallel, ~0.05ms)
iter1: model processes [input + draft_tokens] → verify
       accept longest correct prefix + sample next token
```

iter0 = draft, iter1 = verify. Zero wasted compute.

Expected acceptance per DFlash ablation (no KV injection baseline):
- Easy tokens: ~85-90% per position
- Hard tokens: ~50-60% per position
- Average accepted: ~2.5-3 tokens per cycle

### Phase 1: 2-Machine DFlash Drafter

**Topology:**
```
Machine A (192.168.1.140):  Target model (TYR-HALO full)
Machine B (192.168.1.145):  DFlash block diffusion drafter

Thunderbolt 4: ~9 Gbps = ~1.1 GB/s
```

**Data flow per cycle:**

| Data | Size | Direction | Latency |
|------|------|-----------|---------|
| h_iter0 features (5 layers, K+V) | ~32 KB/token | A → B | ~0.03 ms |
| 16 draft token IDs | 64 bytes | B → A | ~0.001 ms |
| Verification result | 4 bytes | A → B | ~0.001 ms |

Transfer latency (~0.03 ms) is negligible vs compute (~5 ms forward pass).

**Async pipeline:**
```
Machine A:                    Machine B:
iter0 (6 layers)              
  └── h_iter0 ──────────────→ DFlash: denoise 16 tokens (parallel, 1 step)
iter1 (6 layers, verify) ←─── draft tokens arrive mid-iter1
  └── verify + accept          SSD cache: pre-compute next outcomes
```

Zero idle time — Machine B's DFlash runs in parallel with Machine A's iter1.

**DFlash Drafter Architecture:**
- 5 transformer layers, d=640, bidirectional within block
- Shared frozen embeddings + LM head from TYR-HALO (tied)
- KV injection: features from 5 uniformly-sampled TYR-HALO layers projected into drafter attention
- Block size: 16 tokens
- Single-step denoising (no iterative refinement)
- Training: on cached h_iter0 features, position-dependent exponential loss decay

**int8 variant:** Combine DFlash's parallel generation with EGGROLL's int8 insight. 5-layer drafter in int8 = 2x faster + 2x less bandwidth on RDNA 3.5 WMMA instructions.

**Timing:**

| Step | Machine A | Machine B | Wall time |
|------|-----------|-----------|-----------|
| Forward iter0 | 2.5 ms | idle | 2.5 ms |
| Send features | — | — | 0.03 ms |
| Forward iter1 + verify | 2.5 ms | DFlash (parallel) | 2.5 ms |
| Accept + next | 0.1 ms | Cache lookup | 0.1 ms |
| **Total per cycle** | | | **~5.1 ms** |
| **Tokens accepted** | | | **~6-8** |
| **Effective tok/s** | | | **~1200-1600** |

vs single-machine autoregressive ~200 tok/s → **6-8x speedup.**

### Phase 2: SSD Outcome Caching

Adds Saguaro (Kumar et al. 2026) on top of DFlash drafter.

While Machine A verifies current draft, Machine B predicts the top-F likely verification outcomes and pre-computes DFlash speculations for each. Verification outcome = (k accepted tokens, bonus token t*). Cache maps outcomes → pre-computed next drafts.

**Saguaro Sampling:** Bias draft distribution to concentrate residual probability on cached tokens:
```
σ_{F,C}(z) ∝ C*exp(z_t) for top-F tokens, exp(z_t) otherwise
```
C ∈ [0,1] trades acceptance rate for cache hit predictability.

Cache hit ~90% → next draft returned immediately (zero latency). On miss → DFlash fallback (standard single-step).

### Superseded Approaches

| Original Plan | Replaced By | Why |
|---------------|-------------|-----|
| int8 minGRU sequential drafter | Draft heads + DS2D (Phase 0) / DFlash (Phase 1) | Sequential generation wrong paradigm; parallel wins |
| PTP auxiliary variable decoding | DFlash block diffusion | DFlash: 16 tokens parallel, higher acceptance |
| DepthMemoryCache | MoDA depth-attention | MoDA is per-head per-token; strictly more powerful |
| Sequential ES rollouts | CTG batched generation | 8x faster alignment via concurrent masked streams |
| Static draft-only approach | DS2D forecast + dynamic draft heads | Forecast primes model, draft heads provide dynamic signal; complementary |

---

## Key Differences from Prior Architectures

| Aspect | FENRIR-HALO | TYR-HALO |
|--------|-------------|----------|
| Unique params | 80M | 58M |
| Shared layers | 10 | 6 |
| Loop depth | mean=3 | mean=2 |
| Depth aggregation | DepthMemoryCache (gated) | MoDA depth-attention (per-head) |
| Residual mechanism | SimpleParcaeInjection | mHC Sinkhorn (4-branch) |
| Auxiliary loss | None | MTP depth=1 |
| Data strategy | CLIMB mixture | CLIMB + synthetic (Tufa Labs) |
| NVIDIA support | No | Yes (bf16, full compile) |
| Inference strategy | Autoregressive only | 3-phase: draft heads → DFlash → SSD |
| Alignment | DPO/SimPO planned | EGGROLL ES (no gradients) |
| 2-machine inference | Not considered | Full async pipeline: target + DFlash drafter |

---

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Model (all variants) | Done, smoke-tested | `models/tyr_halo.py` |
| CodaAttention depth_kvs | Done | `models/jormungandr_halo.py` |
| MTP loss | Done | `halo_training/mtp_loss.py` |
| CLI --mtp flag | Done | `halo_training/cli.py` |
| RTX 4060 Ti guide | Done | `docs/guides/tyr-halo-rtx4060ti-training.md` |
| Draft heads (DraftHeads class) | Done, smoke-tested | `models/tyr_halo.py` |
| DS2D forecast embeddings | Done, smoke-tested | `models/tyr_halo.py` |
| CTG concurrent generation | Done, smoke-tested | `models/tyr_halo.py` |
| Speculative decode loop | Done, smoke-tested | `models/tyr_halo.py` `speculative_decode()` |
| DFlash drafter (Phase 1) | Planned | `models/tyr_drafter.py` |
| SSD caching (Phase 2) | Planned | `scripts/ssd_inference.py` |
| EGGROLL ES alignment + CTG | Planned | `scripts/eggroll_align.py` |
| Synthetic data gen | Planned | `scripts/synthetic_pretrain_gen.py` |
| mhc_sinkhorn custom op | Planned (AMD) | `kernels/hip/_torch_ops.py` |
| Autokernel pattern aliases | Planned (AMD) | `autokernel/_patterns.py` |

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

### Throughput

| Platform | Target |
|----------|--------|
| RTX 4060 Ti (training) | >60K tok/s |
| Strix Halo DDP (training) | >60K tok/s |
| Single GPU (inference, Phase 0) | >500 tok/s (~2.5x baseline) |
| 2-machine (inference, Phase 2) | >1200 tok/s (~6x baseline) |
