---
title: "Paper Deep-Dive: May 2026"
domain: architectures
type: reference
status: active
tags: [%paper-review, %hyperloop, %trm, %hrm, %distillation, %looped-transformers, %hyper-connections, %sessa, %infomamba, %gendistill, %mtp]
related:
  - knowledge/architectures/reliable_small_lm_insights.md
  - docs/superpowers/specs/2026-04-29-tyr-halo-design.md
  - knowledge/training/instruct_alignment_techniques_2025_2026.md
---

# Paper Deep-Dive: May 2026

9 papers analyzed. Ordered by relevance to our project.

---

## 1. Hyperloop Transformers (2604.21254v2) — MIT

**Authors:** Zeitoun, Torroba-Hennigen, Kim (MIT)

**Core idea:** Partition Transformer into begin/middle/end blocks. Middle block loops with shared weights. At each loop boundary, apply simplified hyper-connections: diagonal sigmoid gating of n=4 parallel residual streams, learned loop position embeddings, stream read/write projections.

### Architecture

```
Begin block (non-looped, ~25% params)
  → Expand to n=4 parallel streams (copy)
  → Middle block (looped 3x, shared weights, ~50% params)
    → At each loop boundary:
        z = RMSNorm(flatten(y))  # y: [T, n, C] → z: [T, nC]
        H_pre  = sigmoid(alpha_pre * (W_pre @ z) + b_pre)    # [1, n] — read
        H_post = 2*sigmoid(alpha_post * (W_post @ z) + b_post) # [n, 1] — write
        H_res  = diag(sigmoid(alpha_res * (W_res @ z) + b_res)) # [n, n] — gate
        input = H_pre @ y          # mix n streams → 1 C-dim
        output = MiddleBlock(input) + e_l  # + loop position embedding
        y = H_res @ y + H_post @ output    # gate streams + write back
  → Contract (average across n streams)
  → End block (non-looped, ~25% params)
```

### Key Results

| Scale | Transformer PPL | Hyperloop PPL | Hyperloop Params | Param Savings |
|-------|----------------|---------------|-----------------|---------------|
| 240M | 14.65 | **14.40** | 135.7M | 43% |
| 1B | 10.19 | **9.65** | 579.7M | 41% |
| 2B | 8.60 | **8.49** | 990.8M | 51% |

Beats full-size Transformer at every scale. <5% throughput penalty.

### Critical Ablations

- **Diagonal gating beats Sinkhorn** (14.40 vs 14.59) — simpler is better
- **HC at loop boundaries only beats per-layer** (14.40 vs 14.45) — sparse placement optimal
- **LoRA across loops barely helps** (14.77 at rank 32, +7.5M params) — Hyperloop 14.40 with +0.2M
- **4 streams is sweet spot** (n=2: 14.43, n=4: 14.40, n=10: 14.35)
- **Loop position embeddings distinguish iterations** without per-layer params
- **INT4 quantization:** Hyperloop degrades less than vanilla looped (14.68 vs 15.18)
- **Overtraining (100B on 136M):** Hyperloop 12.19 vs Looped 12.56 — resilient

### Training Setup

AdamW, lr=4e-4, warmup 1-2K steps, cosine decay to 4e-5, batch=256×2048=524K tokens, BF16, grad clip 1.0, 8×H100. Weight decay 0.1. SwiGLU with 2.75x expansion. RoPE base 10000. Llama-2 tokenizer.

### Relevance: **VERY HIGH — direct upgrade to TyrHaloLight**

Our mHC uses Sinkhorn; paper proves diagonal gating is both simpler and better. We're missing loop position embeddings (free quality gain). Our begin/end blocks (prelude/coda) were dropped for speed but paper shows they matter.

---

## 2. Tiny Recursion Models / TRM (2510.04871v1) — Samsung SAIL

**Author:** Jolicoeur-Martineau (Samsung SAIL Montreal)

**Core idea:** Simplify HRM into a single 2-layer network (5-7M params) that recurses T=3 cycles × n=6 latent steps = 42 effective depth. Two latent variables: y (current answer) and z (reasoning state). Deep supervision with N=16 segments.

### Architecture

```python
def latent_recursion(x, y, z, n=6):
    for i in range(n):
        z = net(x, y, z)   # latent reasoning
    y = net(y, z)           # refine answer
    return y, z

def deep_recursion(x, y, z, n=6, T=3):
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(x, y, z, n)
    y, z = latent_recursion(x, y, z, n)  # WITH grad
    return y, z
```

### Key Results

| Task | DeepSeek R1 (671B) | o3-mini-high | HRM (27M) | **TRM (5-7M)** |
|------|-------------------|-------------|-----------|----------------|
| Sudoku-Extreme | 0% | 0% | 55.0% | **87.4%** |
| ARC-AGI-1 | 15.8% | 34.5% | 40.3% | **44.6%** |

### Critical Ablations (Sudoku-Extreme)

| Change | Accuracy | Delta |
|--------|----------|-------|
| Full TRM (T=3, n=6) | 87.4% | — |
| 1-step gradient (DEQ-style) | 56.5% | **−30.9%** |
| Too few recursions (T=2, n=2) | 73.7% | −13.7% |
| Self-attention on small grids | 74.7% | −12.7% |
| 4 layers instead of 2 | 79.5% | −7.9% |
| No EMA (0.999) | 79.9% | −7.5% |
| Separate networks | 82.4% | −5.0% |
| MoE | massive drop | — |

### Training Setup

AdamW, lr=1e-4, beta=(0.9, 0.95), batch=768, hidden=512, EMA=0.999. Stable-max loss. Weight decay 1.0 (Sudoku/Maze), 0.1 (ARC). 1×L40S <36h (Sudoku), 4×H100 ~3d (ARC).

### Relevance: **MEDIUM-HIGH (conceptual validation)**

- Validates "small + deep recursion > large flat" = our TyrHaloLight result
- **1-step gradient is catastrophic** — confirms our full backprop choice
- **EMA (0.999) gives free gains** — we should add this
- **2 layers + more recursions > 4 layers + fewer** — our 6-layer × 2 iters might benefit from fewer layers × more iters
- **MoE kills generalization** — avoid MoE in recursive architectures
- Non-autoregressive (puzzles), so architecture doesn't directly transfer to LM pretraining

---

## 3. Hierarchical Reasoning Model / HRM (2506.21734v3) — Sapient Intelligence

**Authors:** Wang et al. (Sapient Intelligence, Singapore + Tsinghua)

**Core idea:** Two-level recurrent architecture. Low-level f_L (fast, detailed) iterates T times, then high-level f_H (slow, abstract) updates once. This creates "hierarchical convergence" — L-module converges to local equilibrium, H-module shifts the target, resetting convergence. 1-step gradient approximation (no BPTT). Q-learning halting (ACT).

### Architecture

```
z_L^i = f_L(z_L^{i-1}, z_H^{i-1}, x)   [every step]
z_H^i = f_H(z_H^{i-1}, z_L^{i-1})       [every T steps]
```

N=2 H-cycles × T=2 L-cycles per segment. M=16 segments with deep supervision. 27M params, 4-layer blocks each, Post-Norm RMSNorm, bidirectional attention.

### Key Results

- ARC-AGI-1: 40.3% (beats o3-mini-high 34.5%, Claude 3.7 21.2%)
- Sudoku-Extreme: ~perfect, Maze-Hard: ~perfect
- Direct 8-layer Transformer (same size): 0% on Sudoku/Maze

### Key Mechanism: Hierarchical Convergence

Standard RNNs: residuals → 0 quickly (premature convergence, limited effective depth).
HRM: L-module converges, then H-module resets target → saw-tooth residual pattern, maintaining computational activity. This is exactly what our Parcae injection does.

### Training Setup

Adam-atan2 (scale-invariant), lr=1e-4, warmup=2000, batch=768, hidden=512. Post-Norm. Stablemax loss. Q-head with sigmoid halting. EMA of weights. 8×GPU, ~24h for ARC.

### ACT (Adaptive Computation Time) via Q-Learning

Q_halt, Q_continue from linear head on z_H. Halt when Q_halt > Q_continue. Trained with BCE + bootstrapped Q-targets. No replay buffer needed (Post-Norm bounds activations).

### Relevance: **MEDIUM (we already do most of this)**

- Our SimpleParcaeInjection = their z_H → z_L reset mechanism
- 1-step gradient works for HRM but TRM Paper 2 shows full backprop is far better
- **ACT for inference** — model halts early on easy tokens, useful for speculative decoding
- Post-Norm is interesting but harder to train; not worth switching from our Pre-Norm

---

## 4. TIDE — Cross-Architecture Distillation for Diffusion LLMs (2604.26951v1) — Peking University

**Authors:** Zhang et al. (Peking/Zhejiang)

**Core idea:** Distill large diffusion LLMs (8-16B) into tiny 0.6B diffusion LLM. Three components:

1. **TIDAL:** Dual-axis lambda modulation — weight distillation by diffusion timestep (unreliable at high noise) and training progress (prevent early collapse).
2. **CompDemo:** Split masked positions into two complementary sets, run two teacher passes with complementary context. Enriches teacher signal under heavy masking.
3. **Reverse CALM:** Cross-tokenizer distillation via byte-aligned chunk-level Bernoulli KL. Bounded gradients (unlike Forward CALM which diverges).

### Key Results

0.6B student: 34.2% avg benchmarks (vs 32.7% undistilled, vs 40.9% same-size AR). Excels at code (HumanEval 49.4% vs 32.3% AR). 22× memory reduction vs 16B teacher.

### Reverse CALM Gradient Property

```
dL_rev/dθ = -Σ_c (dp_s^c/dθ) · log(p_t^c / (1 - p_t^c))
```
Bounded: coefficient depends only on fixed teacher. Dual-end filtering: poorly aligned chunks (p_t ≈ 0.5) zero the coefficient. Equivalent to Bernoulli KL (mode-seeking).

TIDAL + Reverse CALM are incompatible — TIDAL suppresses the gradient that makes Reverse CALM work.

### Training Setup

LR=5e-5, 10 epochs, seq_len=512, block_size=32, BF16, DDP, flex_attention. Datasets: Tulu-3 SFT + SmolTalk + OpenCoder-SFT (Python).

### Relevance: **LOW**

We don't use diffusion LLMs. Reverse CALM's bounded gradient analysis is elegant — save for future distillation work. Not actionable now.

---

## 5. GenDistill / Hybrid-KDA (2603.26556) — Huawei Zurich

**Authors:** Kostelec, Wang, Laborieux, Sourmpis, Qinghai (Huawei Zurich / ACS Lab)

**Core idea:** Expose critical evaluation gap: perplexity overestimates distilled hybrid model quality vs generation eval. A 7B distilled model appeared within 0.2pp of teacher on log-likelihood but fell 20.8pp behind on generation tasks. Propose GenDistill pipeline + Hybrid-KDA architecture (replace 21/28 attention layers with linear attention using channel-wise diagonal gating, short convolutions, SiLU output gating).

### Architecture: Hybrid-KDA

Starting from Qwen3-0.6B (596M), retains 7 frozen attention layers (selected via beam search on associative recall), replaces 21 with KDA layers. KDA uses per-dimension diagonal gates for selective memory retention, identity-initialized convolutions, L2-normalized QK (no learnable scales). Total: 613M params.

### GenDistill Pipeline

- Stage 1 (30M tokens): Q/K projection alignment via L2 hidden-state distance
- Stage 2 (30M tokens): Block-wise hidden state alignment
- Stage 3a (500M tokens): End-to-end KD on pretraining data
- Stage 3b (250M tokens): Instruction-tuning KD with frozen attention layers

### Key Results

86-90% teacher accuracy retained on knowledge benchmarks, 75% KV cache reduction, 2-4x TTFT speedup at 128K context. Critical ablation: dataset selection has largest impact (26.2% to 46.9% average). Completion-only loss masking beats full-sequence. Freezing attention during post-training gives 5-8pp generation improvement invisible to perplexity.

### Relevance: **MEDIUM — two actionable findings**

1. **Identity initialization for conv/gates** — free stability improvement. Start convolutions and gating at identity → smooth training startup. Applicable to our ShortConvBlocks.
2. **Don't trust BPB alone** — perplexity systematically overestimates hybrid model quality. When comparing loop architectures vs flat, need generation-based evaluation. Our BPB 2.10 vs 2.75 comparison is valid (same architecture family), but cross-family comparisons need generation eval.
3. **Channel-wise diagonal gating** — validates our existing finding that diagonal gating beats Sinkhorn.
4. **Beam search for attention layer placement** — at 4-6 shared layers, manually testing which positions get attention vs conv may find better placement than our current (2, 5) positions.

---

## 6. InfoMamba (2603.18031) — Renmin/CSU/Zhejiang

**Authors:** Wang, Fu, Zhou et al. (Renmin University, Central South U, Zhejiang U)

**Core idea:** Parallel SSM + concept-bottleneck global filtering with information-maximizing fusion. Two streams: (1) concept-bottleneck linear filtering layer (O(nk) complexity, replaces quadratic attention), (2) selective SSM recurrent stream. Fused via IMF injection into SSM dynamics with redundancy-reduction loss.

### IMF Fusion (Key Portable Idea)

```python
# Inject global context h from filtering layer into SSM recurrence:
s_t = Lambda * s_{t-1} + B * x_t + P * h_t     # SSM with global injection
y_t = C * s_t + F * h_t                          # Output with global skip
```

### Cross-Covariance Redundancy Loss

```python
# Training-only loss, zero inference cost:
L_red = ||off_diag(C_stream1_stream2)||^2_F
```

Forces dual streams to learn complementary (not redundant) features.

### Key Results

ImageNet 89.0% (SOTA for SSM-class), IMDb 85.1%, AG-News 89.1%. +37.6% throughput vs ViT-S. Removing global path: -5.9 points. Removing MI loss: -1.6 points.

### Relevance: **LOW-MEDIUM — portable training losses**

- IMF fusion equation applicable to any dual-stream architecture, but our hybrid blocks already use simpler gated fusion that works
- **Cross-covariance loss** is the most portable piece: trivial to add between ShortConv and GQA streams, zero inference cost, encourages complementary representations
- Vision-focused, no LM experiments — can't directly compare BPB
- Concept-bottleneck replaces attention but our models need attention for MoDA depth KVs

---

## 7. Embedding-Space Probing MTP (2603.17942) — ICML 2026

**Core idea:** Training-free multi-token prediction for any frozen LLM. Synthesize "mask tokens" in embedding space (mean of prompt embeddings), inject alongside input with tree attention masks. Decoder layers progressively align mask-token states with valid next-token states. Predictions organized into speculative token tree, verified losslessly.

### Key Results

- 12% higher acceptance length on LLaMA3, 8-12% on Qwen3 vs SOTA training-free baselines
- 15-19% throughput improvement (tokens/second)
- Up to 40% fewer forward model calls at BC=30/60
- Robust to mask initialization (even 5-10 sigma from mean only slightly degrades)

### Architecture

Tested on LLaMA3.2-3B, LLaMA3.1-8B, Qwen3-8B, Qwen3-32B. 1-2 mask tokens. Dynamic update rule: `m_i[s+1] = m_i[s] + lambda * (e_{t+s} - m_i[s])`, lambda=0.1. Tree pruning by cumulative probability.

### Relevance: **LOW — inference only**

Not a training technique. Validates that MTP structure exists latently in trained models — confirms our MTP head is architecturally sound. For deployment on Strix Halo (bandwidth-bound), reducing forward passes via embedding probing is attractive. Zero memory overhead. File for post-training inference optimization.

---

## 8. DSKD-KQ (2603.22056) — Cross-Tokenizer Distillation

**Core idea:** Knowledge distillation between LLMs with different tokenizers/vocabularies. Extends DSKD-CMA with adversarial (GAN-style) and optimal transport methods to align teacher K and student Q distributions across tokenizer boundaries.

### Methods

1. **DSKD-CMA-GA** (adversarial): discriminator distinguishes teacher keys from student queries, student projector acts as generator
2. **DSKD-CMA-CT** (conditional transport): optimal transport with trainable critic

### Key Results

GA achieves 18.81 avg ROUGE-L (best cross-tokenizer), +0.37 over baseline DSKD-CMA. Cross-tokenizer KD now matches same-tokenizer in some benchmarks. Student: GPT-2 124M, Teacher: Qwen1.5 1.8B (150K vocab).

### Relevance: **LOW — Phase 2 reference**

Not applicable to pretraining. Useful if we ever distill from a larger model (Qwen/LLaMA) into our custom architectures with different tokenizers. The adversarial K-Q alignment is general enough to apply to any cross-attention between heterogeneous representations. File for instruction-tuning phase.

---

## 9. Sessa — Selective State Space Attention (2604.18580)

**Core idea:** Place attention inside a recurrent feedback loop. Lower-triangular system solve `(I - B_fb)s = f` where B_fb is scalar feedback matrix with input-dependent gating. Yields power-law memory decay O(l^{-beta}) — slower than exponential (Mamba) and 1/l dilution (Transformers).

### Architecture

Dual-path per block: (1) forward attention with RoPE (standard causal), (2) feedback attention over strict past (no positional encoding) with scalar gain gamma_t in (-1,1) via tanh. Triangular solve = sequential recurrence.

### Key Results

- Long-context: +8.6% over Transformer on SymbolSoup, +26% on Diffuse MQAR
- **Short-context: -8% perplexity vs Transformer/Mamba2** (8.37 vs 7.67 on SimpleStories)
- Mamba2 fails entirely on long-context associative recall tasks

### Relevance: **NOT ACTIONABLE**

The -8% perplexity penalty on short context is disqualifying for our 1K-2K training regime. O(T^2) triangular solve offers no throughput advantage over flash attention. Only portable nugget: scalar feedback gain (gamma_t) could strengthen Parcae velocity gating, but we're dropping momentum entirely. Interesting theory, wrong regime.

---

## Cross-Paper Synthesis: What Matters for TyrHaloLight

| Finding | Source | Confidence | Action |
|---------|--------|------------|--------|
| Diagonal gating > Sinkhorn for stream mixing | Hyperloop | HIGH (3 scales) | Replace mHC Sinkhorn |
| Loop position embeddings help | Hyperloop | HIGH | Add `e_l` per iteration |
| HC at loop boundaries only (not per-layer) | Hyperloop | HIGH | Already doing this |
| 4 parallel streams is optimal | Hyperloop | HIGH | Keep n=4 |
| Begin/end non-looped blocks matter | Hyperloop | HIGH | Add lightweight begin/end |
| Full backprop through iterations >> 1-step gradient | TRM | HIGH (30.9% gap) | Already doing full backprop |
| EMA (0.999) gives free generalization | TRM | HIGH (7.5% gap) | Add EMA to trainer |
| Smaller network + more recursions > larger + fewer | TRM | HIGH | Validate with mean=3 vs mean=2 |
| MoE kills generalization in recursive models | TRM | HIGH | Avoid MoE |
| ACT for variable-depth inference | HRM | MEDIUM | Future: speculative decode |
| Hierarchical convergence = Parcae injection | HRM | HIGH | Already implemented |
| Identity init for conv/gates — free stability | GenDistill | HIGH | Adopt in ShortConvBlocks |
| Don't trust BPB alone — need generation eval | GenDistill | HIGH | Add generation eval alongside BPB |
| Cross-covariance loss for dual streams | InfoMamba | MEDIUM | Training-only, zero inference cost |
| Beam search for attention layer placement | GenDistill | MEDIUM | Test (1,3) vs (2,5) GQA positions |
| Embedding probing for inference MTP | Embed-MTP | LOW | Post-training deployment |
| Sessa feedback gain — short context penalty | Sessa | LOW | Not actionable at 1K-2K ctx |

---

## Implementation Results (2026-05-03)

All 4 improvements implemented and benchmarked on Strix Halo (gfx1151, 240 GB/s).

### What was implemented

1. **HyperloopHC** — Diagonal sigmoid gating replaces mHC Sinkhorn. `HyperloopHC` class in `models/tyr_halo.py`. Available via `use_mhc=True`. n=2 or n=4 parallel streams, RMSNorm on flattened streams, 3 small projections (W_pre, W_post, W_res) per loop boundary.

2. **Loop position embeddings** — `self.loop_pos_embeds = nn.Parameter(torch.zeros(mean_recurrence, d_model))`. Added after `iter_norm(h)` at each iteration. Always active. Near-zero parameter cost.

3. **EMA** — `--ema` CLI flag, decay=0.999. Uses `torch.optim.swa_utils.AveragedModel`. EMA state saved in checkpoints (`ema_state_dict` key). ~0% throughput cost.

4. **Velocity clamp ±8.0** — In ShortConvBlock + MoDAGQABlock. Prevents fp16 overflow in Parcae loop.

### Throughput impact on Strix Halo

| Config | tok/s | Delta | Notes |
|--------|-------|-------|-------|
| Old TyrHaloLight (no fixes) | 23,615 | — | Unstable: NaN at step 790 with lr=0.005 |
| + velocity clamp + GradScaler(1024) | ~18,450 | −22% | Stable for full epoch (BPB 2.10) |
| + loop pos embeds (fresh compile) | 16,004 | −32% | Stable. Compile cache mismatch may inflate gap |
| + Hyperloop HC (n=2 streams) | 14,505 | −39% | HC stream ops too expensive on bandwidth-limited HW |
| + Hyperloop HC (n=4 streams) | 13,980 | −41% | Diminishing returns, more memory |
| + EMA | ~same | 0% | Free |

### Key finding: Hyperloop HC is too expensive on our hardware

The Hyperloop paper benchmarks on H100 (3.35 TB/s bandwidth, 989 TFLOPS). Our Strix Halo has 240 GB/s bandwidth — 14× less. The HC operations (RMSNorm on expanded n×C tensor, 3 linear projections, stream expand/contract) are memory-bandwidth-bound, not compute-bound. On H100, the overhead is <5%. On Strix Halo, it's 35-41%.

**Decision: HC streams disabled by default** (`use_mhc=False` in TyrHaloLight). Loop position embeddings kept (free). EMA kept (free). Velocity clamp kept (stability required).

### What to use on high-bandwidth hardware

On NVIDIA H100/A100 or future AMD with higher bandwidth: enable HC via `use_mhc=True`. Use n=4 streams (paper's sweet spot). The quality gain from Hyperloop HC is real (14.40 vs 14.85 PPL at 240M scale) — we just can't afford it on our current hardware.

### Updated architecture: TyrHaloLight (production)

```
TyrHaloLight: 58.5M unique, ~117M Parcae-equivalent
  - embed_rank=448, d_model=640, n_heads=8, n_kv_heads=4
  - 6 shared layers (4 ShortConv + 2 MoDA-GQA), Parcae loop mean=2
  - Loop position embeddings (per-iteration learned vector)
  - Velocity clamp ±8.0 (fp16 stability)
  - No prelude/coda, no HC streams
  - MTP depth=1 (training only)
  - MoDA depth-attention (prepend KVs, flash-compatible)
  - EMA decay=0.999 (evaluation/inference)
```

---

## Batch 3: Cross-Architecture Distillation Papers + LFM2.5 Analysis (2026-05-03)

4 distillation papers analyzed + LFM2.5-350M model card + LFM2 technical report. Focus: feasibility of converting LFM2.5-350M to hybrid Mamba.

---

### 10. Attention to Mamba (2604.14191) — Apple/Meta

**Authors:** Moudgil, Huang, Dhekane, Rodriguez, Zappella, Danieli

**Core idea:** Two-stage distillation bridges Transformer→pure Mamba gap. Direct distillation fails; linearized attention as intermediate stepping stone.

**Pipeline:**
- Stage 1: Transformer → Linearized Attention (kernel trick approximation of softmax)
- Stage 2: Linearized Attention → Pure Mamba (principled weight init from stage 1)

**Results at 1B/10B tokens:** PPL 14.11 vs 13.86 teacher (Pythia-1B). Gap = 0.25. Near-lossless.

**Key insight:** Architectural gap too large for single-step KD. Linearized attention creates traversable bridge. Robust across architecture variants, scales with model size and tokens.

**Relevance: MEDIUM — proves pure-SSM distillation possible, but we don't need it**

The two-stage bridge idea is elegant. If we ever want to create a pure-Mamba version of a trained Transformer, this is the recipe. But at 350M scale, the loss from even 0.25 PPL gap is proportionally larger. Better for 1B+.

---

### 11. HALO / HypeNet — Hybrid Linear Attention Done Right (2601.22156) — Tsinghua, ICML 2026

**Authors:** Chen, Thai, Zhou, Zhang, Shen, Wang, Xiao, Han, Liu (Tsinghua/OpenBMB)

**Core idea:** Convert pretrained Transformers to RNN-attention hybrids with only **2.3B tokens** (<0.01% of pretraining data). 10-100× more efficient than prior methods.

**HALO 4-Stage Pipeline:**

| Stage | What | Data |
|-------|------|------|
| 1 | Hidden-state alignment (MSE, layer-wise) | ~0.3B tokens |
| 2 | Attention layer selection (recall vs commonsense tradeoff) | — |
| 3 | End-to-end KL distillation | 1B tokens |
| 4 | Long-context finetuning | 1B tokens |

**HypeNet architecture:**
- Lightning Attention as RNN mixer (data-independent forget gates — prevents catastrophic forgetting)
- **HyPE positional encoding:** RoPE in RNN layers, NoPE in attention layers → length generalization to 256K from 4K training
- QK-normalization critical (removing it: NIAH@128K drops from 79.9% to 17.3%)
- GQA→MHA expansion in RNN layers for expressivity
- Output gating on RNN layers

**Results:**
- 99.8% NIAH at 256K (teacher Qwen3-1.7B: 17% at 128K!)
- 3× decoding speedup, 3.4× prefill speedup
- Only 25% attention layers retained
- Lightning Attention > Mamba2 > GLA for this use case

**Relevance: HIGH — most data-efficient distillation method, HyPE is portable**

HyPE (RoPE in recurrent, NoPE in attention) is a simple trick with outsized impact on length generalization. Lightning Attention's data-independent forget gates outperforming Mamba2 is surprising and important — selective state isn't always better. The 4-stage pipeline is the most practical recipe for Transformer→hybrid conversion.

---

### 12. Retrieval-Aware Distillation (2602.11374) — Bick, Xing, Gu (CMU/UCSD)

**Authors:** Aviv Bick, Eric P. Xing, Albert Gu

**Core idea:** SSMs fail specifically at in-context retrieval (Gather-and-Aggregate heads). Identify and preserve only the ~2% of attention heads critical for retrieval; replace everything else with SSM.

**Method:**
1. Ablation on synthetic retrieval tasks identifies G&A heads
2. Non-uniform, sparse placement across layers (not every-4th-layer)
3. Replace non-critical attention with SSM
4. Distill from full Transformer teacher

**Results:**
- **2% attention heads** (~10 in 1B model) → >95% teacher performance on retrieval
- **8× SSM state dimension reduction** (retrieval handled by attention, so SSM state can be tiny)
- 5-6× memory efficiency vs hybrids retaining 25% attention

**Key insight:** Once a few attention heads handle retrieval, SSMs don't need large states. The SSM's job simplifies to local/sequential processing.

**Relevance: HIGH — principled attention placement + SSM state sizing**

This is the theoretical foundation for why 2-6 GQA blocks + many conv/SSM blocks works. LFM2.5 uses 6/16 attention blocks (37.5%) — probably generous, could work with fewer. The 8× state reduction finding suggests our SSM state sizes could be smaller.

---

### 13. Effective Distillation to Hybrid xLSTM (2603.15590) — Hochreiter group

**Authors:** Schmidinger, Schmied, Hartl, Stap, Hoedt, Beck, Böck, Klambauer, Hochreiter

**Core idea:** 3-stage modular linearization + decentralized expert training + linear weight merging.

**Architecture:** `h_t = o_t * mLSTM(q_t) + (1 - o_t) * SWA(q_t)` — per-head sigmoid gate fuses mLSTM and sliding window attention.

**Pipeline:**
- Stage I: Layer-wise hidden-state alignment (MSE, 655M tokens)
- Stage II: Sparse KD (CE + top-k=256 KL, 5-20B tokens, offline teacher logits)
- Stage III: Expert merging via linear weight averaging

**Results:**
- alpha* = 0.0 (teacher parity on 50%+ benchmarks) for xLSTM-Llama3.1-8B
- 2× latency reduction (batch=1, 65K), 4× throughput (batch=8)
- Beats LoLCATs, QRWKV6, Mamba-in-Llama

**Critical ablations:** Pure mLSTM >> linear attention. mLSTM + SWA = striking synergy. Full fine-tuning >> LoRA. No normalization layers (degrade student-teacher alignment). Attention sinks essential.

**Relevance: MEDIUM — expert merging is portable, mLSTM+SWA validates hybrid**

The decentralized expert training + merging pipeline is interesting for post-training: train domain specialists independently, merge via weight averaging. The learned scalar gate `o_t` for stream fusion is simpler than our diagonal gating approach and might work for conv+attention fusion.

---

### 14. LFM2 Technical Report (2511.23404) — Liquid AI

**Authors:** Amini, Labonne, Hasani, Lechner, Rus et al. (Liquid AI)

**LFM2-350M Architecture:**

| Property | Value |
|----------|-------|
| Layers | 16 |
| d_model | 1024 |
| FF dim | 4608 (SwiGLU) |
| Heads | 16Q / 8KV / 64 head_size |
| Attention blocks | 6 of 16 (37.5%) |
| Conv blocks | 10 of 16 (62.5%) |
| Conv kernel | **3** |
| Vocab | 65,536 |
| Context | 32K |
| Pretraining tokens | 10T (LFM2) → 28T (LFM2.5) |

**Gated Short Convolution Block:**
```python
(B, C, h̃) = Linear(h)        # ℝ^d → ℝ^(3d)
y = B ⊙ h̃                    # element-wise gating
z = Conv_k=3(y)               # 1D depthwise conv, kernel 3
o = Linear_out(C ⊙ z)         # output gating + projection
```

**Hardware-in-the-loop search conclusion:** Liquid AI explicitly tested S4, Mamba, Mamba2, CfC, linear attention, sliding window attention. **Verdict: "Most benefits of complex hybrid SSM/linear-attention blocks can be captured by gated short convolutions + a small number of GQA layers."**

**Decoupled Top-K KD:** Binary term (L_B matches total mass in teacher's top-K set) + conditional top-K term (temp-scaled KL within set). Teacher: LFM1-7B, top-K=32. Novel contribution for small-model training from larger teacher.

**Post-training:** 3-stage pipeline (SFT with curriculum → preference alignment via length-normalized DPO → model merging via soup/TIES/DARE). Doom loop fix: 5 temp rollouts + 1 greedy → LLM jury → DPO → RL with n-gram penalty.

**MoE variant (8.3B/1.5B active):** 32 experts, top-k=4, normalized sigmoid router. First 2 layers dense. Matches 2.6B dense quality at 1.5B active.

---

### LFM2.5→Mamba Conversion Feasibility Analysis

**Verdict: Technically feasible. Strategically backwards.**

**Why it's technically possible:**
- All 4 distillation papers prove cross-architecture transfer works at 1B+ scale
- Attention-to-Mamba achieves near-lossless pure-SSM via 2-stage bridging
- HALO achieves it in only 2.3B tokens
- LFM2.5's LIV blocks are structurally close to Mamba (both: Linear → conv1d → gate → Linear)

**Why it's a bad idea:**

1. **Liquid AI already tried Mamba and rejected it.** Hardware-in-the-loop search on Ryzen AI Max+ 395 (our exact hardware) showed gated conv k=3 is fastest. Converting back to Mamba reverses their finding.

2. **Conv1d k=3 is simpler and faster than selective scan at inference.** Conv1d = 3-element shift register. Mamba = recurrent state update with data-dependent gates. On bandwidth-limited hardware (240 GB/s), the state management overhead is real.

3. **350M too small for distillation losses.** All papers work at 1B-8B. At 350M, less redundancy = each head matters more. The 28T-token knowledge is fragile at this scale.

4. **LIV internals partially opaque.** Hidden-state alignment (most data-efficient stage) requires exact intermediate dimensions. LIV block code is available via HF but proprietary architecture decisions may not align with standard SSM assumptions.

5. **You'd lose the 28T-token advantage.** Distillation transfers knowledge imperfectly. The conversion process + 2-20B retraining tokens < the 28T already baked in.

**What makes sense instead:**
- Train Mamba hybrid from scratch with insights from all 4 papers (our TyrHaloLight approach)
- Scale training data (28T proves returns keep growing at 350M)
- Use Decoupled Top-K KD from a larger teacher into our architectures
- Apply HyPE (RoPE in recurrent, NoPE in attention) for length generalization

---

### Cross-Paper Synthesis: New Findings for Our Project

| Finding | Source | Confidence | Action |
|---------|--------|------------|--------|
| Two-stage bridge (linearized attention) enables pure-SSM distillation | Attention-to-Mamba | HIGH | File for future teacher→SSM pipeline |
| Only 2.3B tokens needed for Transformer→hybrid conversion | HALO | HIGH | Most data-efficient recipe known |
| HyPE: RoPE in recurrent + NoPE in attention → 64× length generalization | HALO | HIGH | Test in TyrHaloLight |
| Lightning Attention > Mamba2 for distilled hybrids | HALO | HIGH | Consider as alternative to ShortConv |
| QK-Norm critical for long-context hybrid (79.9% → 17.3% NIAH without) | HALO | HIGH | Already using QK-Norm |
| Only 2% attention heads needed for retrieval (G&A heads) | Retrieval-Aware | HIGH | Our 33% GQA may be generous |
| 8× SSM state reduction once retrieval handled by attention | Retrieval-Aware | HIGH | Can use smaller state_size |
| mLSTM >> linear attention for distilled students | xLSTM | HIGH | Avoid linear attention variants |
| mLSTM + SWA = striking synergy (learned sigmoid gate) | xLSTM | HIGH | Simpler fusion than diagonal gating |
| Full fine-tuning >> LoRA for architecture conversion | xLSTM | HIGH | Don't use LoRA for hybrid conversion |
| No normalization in distilled students (degrades alignment) | xLSTM | MEDIUM | Context-dependent, may conflict with stability |
| Expert merging via linear weight averaging works | xLSTM | MEDIUM | Portable to post-training specialist merging |
| Conv k=3 beats SSMs on edge hardware (Liquid AI HW search) | LFM2 tech report | VERY HIGH | Conv blocks validated, no need to replace |
| Decoupled Top-K KD (binary + conditional within top-K) | LFM2 tech report | HIGH | Novel KD technique for small models |
| 28T tokens on 350M — returns still growing | LFM2.5 | HIGH | We need orders of magnitude more data |
| 65K vocab with ~10% embedding overhead | LFM2 | MEDIUM | Our 50K vocab = ~15% overhead, consider reduction |

---

## Batch 4: Poolside Laguna — Model Factory + XS.2 / M.1 (2026-05-03)

4 sources analyzed: XS.2-M.1 intro blog, Laguna deeper dive, Titan infrastructure blog, HuggingFace model card.

---

### 15. Laguna XS.2 — Architecture (poolside/Laguna-XS.2)

| Property | Value |
|----------|-------|
| Total params | 33B |
| Active params/token | 3B (9.1% activation) |
| Experts | 256 + 1 shared |
| Layers | 40 (30 SWA + 10 global attention) |
| SWA window | 512 tokens |
| SWA:Global ratio | 3:1 |
| Gating | Sigmoid, per-head, per-layer rotary scales |
| Context | 128K |
| Precision | BF16 train / FP8 inference |
| Training tokens | 30T+ |
| License | Apache 2.0 |

256 experts is unusual — most MoE use 8-64. Each expert ~117M params. Shared expert ensures baseline capability. 3B/33B activation ratio sits between DeepSeek-V4-Flash (4.6%) and Mixtral (27.6%).

Mixed attention: 30 SWA (window=512) + 10 global. Same pattern as HALO paper and xLSTM distillation — local attention cheap, global only where needed. 512-token window is emerging consensus (Poolside, xLSTM, Gemma3).

### 16. Laguna M.1 — Architecture

225B total / 23B active. MoE. API-only, internals not disclosed. Trained on same 30T+ tokens. XS.2 achieves 94% of M.1's SWE-bench Verified score at 13% active params — strong diminishing returns above 3B active for code tasks.

---

### 17. Training Pipeline — Muon, AutoMixer, Async RL (Deeper Dive)

**Distributed Muon at 6,144 H200s:**
- 15% fewer steps to match AdamW loss
- 1 state/param vs AdamW's 2 → lower memory
- Newton-Schulz distributed across ranks via GPUDirect RDMA
- CUDA graphs for Newton-Schulz procedure
- **<1% overhead** of total training step time
- AdamW still used for embeddings + prediction head (same as DSV4)

Largest public Muon validation. Confirms our choice in `halo_training/muon.py`.

**Data: AutoMixer proxy-model ensemble:**
- ~60 proxy models trained on different data mixes
- Surrogate regression maps mix → downstream performance
- Regularized toward baseline mix (prevents degenerate solutions)
- **Key finding:** code quality driven by synthetic + curated code; math benefits from diverse web math
- ~13% synthetic data in XS.2 final mix (~4.4T synthetic across series)

**Deduplication insight — global dedup harmful:**
- Global dedup disproportionately removes high-quality data (appears in multiple sources)
- Snapshot-level dedup with quality-distribution matching instead
- Result: **~2× more unique tokens** maintaining performance
- Counterintuitive, important for our dolma processing

**Asynchronous Agent RL with CISPO:**
- Fully async: actors and trainer run independently
- Solves sync RL problems for long-horizon code tasks:
  - Sync RL: GPUs idle during long rollouts
  - Sync RL: long trajectories systematically under-sampled
- **CISPO algorithm variant** for off-policy stability
- Token-in, token-out: preserve token IDs end-to-end (no re-tokenization drift)
- Weight transfer via GPUDirect RDMA: 5 seconds for full M.1 BF16 weights
- Sandboxed code execution containers for trajectory rollouts
- Iceberg tables for trajectory storage

**Off-policy drift sources identified:** stale weights, non-deterministic kernels, precision mismatches, re-tokenization errors. CISPO + token preservation addresses all.

---

### 18. Titan: Model Factory Infrastructure

**Stack:** TorchTitan (distributed training) + Dagster (orchestration/lineage) + Neptune (metrics) + incident.io (monitoring) + GPUDirect RDMA + CUDA graphs + Sentry/FlightRecorder (debugging).

**Hardware:** 10K H200 GPU cluster. M.1 trained on 6,144 interconnected Hopper GPUs. Kubernetes + torchrun.

**Automation:**
- Pre-flight node stress tests
- Auto-recovery from faulty nodes/stalls (configurable 10-min timeout)
- Escalation: 30-min soft ping → 60-min hard ping → auto Slack channel
- **Periodic weight hash checks** for silent data corruption (SDC)
- First model (Malibu v1) required weeks of manual babysitting; now fully automated

**Disaggregated evaluation:** Checkpoint evals scheduled independently from training. Sometimes every few hundred steps. Separate compute.

**XS.2 pre-training: 5 weeks** start to full post-training.

---

### Benchmark Results

| Benchmark | M.1 (225B/23B) | XS.2 (33B/3B) | Devstral Small 2 (24B) | Claude Haiku 4.5 |
|-----------|----------------|----------------|------------------------|------------------|
| SWE-bench Verified | **72.5%** | **68.2%** | 68.0% | 73.3% |
| SWE-bench Multilingual | **67.3%** | **62.4%** | 55.7% | — |
| SWE-bench Pro | **46.9%** | **44.5%** | — | 39.5% |
| Terminal-Bench 2.0 | **40.7%** | **30.1%** | 22.5% | — |

Eval: Harbor Framework, max 500 steps, sandboxed, temp=0.7, top_k=20, 3-7 averaged runs.

---

### Cross-Source Synthesis: Portable Findings

| Finding | Source | Confidence | Action |
|---------|--------|------------|--------|
| Muon 15% fewer steps, <1% overhead at 6K GPUs | Deeper dive | VERY HIGH | Already using. Validated at production scale |
| Snapshot dedup > global dedup (2× more unique tokens) | Deeper dive | HIGH | Revisit dolma data processing pipeline |
| AutoMixer ~60 proxy models for data mix | Deeper dive | HIGH | Scale CLIMB pipeline concept |
| SWA window=512 + sparse global attention | Model card | HIGH | Matches our 4-conv + 2-GQA pattern |
| Async RL mandatory for agentic code tasks | Deeper dive | HIGH | File for post-training RL stage |
| CISPO for off-policy stability in agent RL | Deeper dive | HIGH | Better than GRPO/PPO for long-horizon |
| Token-in, token-out prevents re-tokenization drift | Deeper dive | HIGH | Apply to agent.py runtime |
| 256 experts works at 33B scale | Model card | MEDIUM | Not applicable at 58-170M |
| Per-head sigmoid gating on attention layers | Model card | MEDIUM | Simple, portable gating mechanism |
| FP8 KV cache quantization production-ready | Model card | MEDIUM | File for inference deployment |
| Global dedup removes best data preferentially | Deeper dive | HIGH | Counterintuitive, verify on our data |
| Disaggregated eval from training | Titan blog | LOW | Already doing this |
| Periodic weight hashing for SDC | Titan blog | LOW | Only at >1K GPU scale |
| XS.2 = 94% of M.1 at 13% active params | Benchmarks | HIGH | Extreme MoE sparsity works for code |

---

## Batch 5: OpenAI Parameter Golf — Competition Analysis (2026-05-04)

44 entries analyzed from OpenAI's "Parameter Golf" competition (March-April 2026). Goal: best BPB in 16MB artifact, 10 min on 8×H100. Pure L(N) optimization.

**Winner:** 1.0611 BPB (from 1.2244 baseline). Transformers dominate; SSM hybrids competitive but ~86 mBPB behind best transformer.

---

### Competition Setup

| Constraint | Limit |
|------------|-------|
| Artifact size | 16,000,000 bytes (code + compressed model) |
| Training time | 10 minutes on 8×H100 SXM |
| Eval time | 10 minutes additional |
| Metric | BPB on FineWeb validation |
| Architecture | Unconstrained |

Baseline: 9L, 512d, 1024 vocab, tied embeddings. Score: 1.2244.

---

### Winning Architecture (1.0611 BPB)

11L × 512d × 8H/4KV GQA. SentencePiece 8192 vocab. Tied embeddings. LeakyReLU(0.5)² MLP 4×. Partial RoPE (16/64 dims). Logit softcap=30.

**Key components stacked:**
1. Depth recurrence (layers 3-5 looped 3×, activated at 35% of training)
2. SmearGate (13-param bigram mixing, BOS masking)
3. Parallel residuals (2-lane from layer 8+, learned mixing)
4. Sparse attention gate (96 params/layer vs 4096)
5. U-Net skip connections with sigmoid gates
6. Layerwise LN scale: `1/sqrt(layer+1)`
7. Polar-Express NS coefficients (per-iteration minimax-optimized)
8. MIN_LR = 0.1 × peak (warmdown floor)
9. EMA decay 0.9965
10. GPTQ int6 + LQER rank-4 correction + lrzip ZPAQ compression
11. CaseOps (lossless case factoring before tokenization)
12. Phased TTT (3-phase score-first LoRA adaptation at eval)

---

### Key Techniques Explained

**Depth Recurrence:** Loop layers 3-5 three times. 11 physical → 17 virtual layers. Zero parameter cost. Critical: delayed activation at 35% of training (train flat first, then enable looping).

**SmearGate:** `x_t = x_t + λ·σ(W·x_t[:12])·x_{t-1}`. 13 parameters (12-dim linear + scalar). Zero-init (transparent at start). BOS masking prevents cross-document leakage. Input-dependent 1-token causal lookback.

**Parallel Residuals:** 2-lane residual from layer 7+. Attention reads lane 0, MLP reads lane 1. Both write to both with learned scalars. Simplified 2-stream Hyperloop. Finding: MLP barely writes to attention lane in deep layers.

**Polar-Express NS:** Replace Muon's fixed `(3.4445, -4.775, 2.0315)` with 5 per-iteration minimax-optimized tuples. Zero cost, better polar factor quality.

**CaseOps:** Bijective transform factors capitalization from content before BPE. `"The NASA"` → `"TITLE the ALLCAPS nasa"`. BPE merges on content not case variants. ~5 mBPB free gain.

**LQER:** Post-GPTQ error correction. Rank-4 SVD on quantization residuals for top-3 tensors. Stored at int4. ~280 KB savings over brute-force.

**Phased TTT:** 3-phase eval-time adaptation. Score prefix docs → global SGD on scored docs → score more → SGD again. Per-document LoRA rank 80 with reset between docs.

---

### SSM Hybrid Results (Non-Record Track)

| Entry | BPB | Architecture |
|-------|-----|-------------|
| Top transformer | 1.0611 | 11L full attention |
| Hymba-8L | 1.1470 | Mamba + SWA parallel per-layer, sigmoid gate |
| Mamba-3 Hybrid | 1.1473 | 7 Mamba-3 + 1 attention |
| JEPA + Mamba-2 | 1.2064 | Pure Mamba-2 + JEPA aux loss |
| Universal Transformer | 1.2249 | 3 blocks × 4 iterations |

SSM hybrids ~86 mBPB behind best transformer. Pure Mamba-3 10.7 mBPB worse than hybrid (single attention layer essential). Hymba parallel fusion (SSM+attention per layer with sigmoid gate) most competitive.

---

### Portable Findings for TyrHaloLight

| Finding | Priority | Effort | Expected Impact |
|---------|----------|--------|-----------------|
| **Polar-Express NS coefficients** | HIGH | 30 min | Free Muon upgrade, better convergence |
| **MIN_LR = 0.1 × peak** | HIGH | 5 min | Better warmdown, convergent with DSV4/GPT-X2/Poolside |
| **Layerwise LN scale 1/sqrt(l+1)** | HIGH | 15 min | Stability, prevents deep layer dominance |
| **Delayed recurrence (35% of training)** | HIGH | 30 min | Train flat first, then enable Parcae loop |
| **Logit softcap=30** | MEDIUM | 15 min | fp16 stability, Gemma2-validated |
| **2-lane parallel residuals** | MEDIUM | 2 hr | Lighter than Hyperloop HC, learned mixing |
| **Parcae iteration skip connections** | MEDIUM | 1 hr | Gated skip between loop iterations |
| **JEPA aux loss** | MEDIUM | 2 hr | Zero inference cost, representation shaping |
| **CaseOps tokenizer** | LOW | 4 hr | Only useful with custom tokenizer |
| **Sparse attention gate** | LOW | 1 hr | Not param-limited at 58M |
| **LQER post-quant correction** | LOW | 2 hr | File for deployment |
| **Untied FFN per iteration** | RESEARCH | 2 hr | Needs ablation |
| **Hymba parallel fusion** | RESEARCH | 4 hr | Different from our sequential pattern |
| **Progressive recurrence** | RESEARCH | 1 hr | Start mean=1, increase mid-training |
