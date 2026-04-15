---
title: "Optimization Candidates for mHC, MoE, and Engram Components"
domain: kernels
type: results
status: active
related:
  - knowledge/kernels/backward_pass_optimization_research.md
  - knowledge/kernels/backward_pass_optimization_results.md
tags: [%engram, %moe, %scattermoe, %fla, %fusion]
---

# Optimization Candidates for mHC, MoE, and Engram Components

**Date:** 2026-04-10
**Hardware:** AMD Strix Halo gfx1151 (RDNA 3.5, ROCm 7.12, no MFMA)
**Context:** These three components appear in 8+ architecture hypotheses and are the primary throughput bottleneck for Tier 3-4 architectures.

---

## 1. Engram (Hash-Based N-gram Knowledge Tables)

**What it does:** Hash-indexes token N-grams, looks up learned embeddings from multi-head tables, gates the result with hidden state, injects via residual.

**Bottleneck:** Irregular gather from embedding tables (hash → index → gather). Hash computation itself is O(1) XOR.

**Current implementation:** `models/engram.py` (218 lines) — pure PyTorch, uses `nn.Embedding` gather.

### Primary: FBGEMM_GPU Table Batched Embeddings (TBE)

- **What:** Meta's industry-standard GPU kernel for batched embedding lookups. Powers all of Meta's recommendation models (billions of embeddings). Highly optimized memory access patterns for irregular gather.
- **URL:** https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu
- **ROCm support:** Yes — 3.9% of codebase is HIP code, has ROCm CI workflows
- **gfx1151 status:** Untested. Has HIP code but CI targets MI200/MI300 (CDNA). May need math builtin patching for ROCm 7.12 (see knowledge §7).
- **Integration:** Replace `MultiHeadEmbedding.forward()` gather with `fbgemm_gpu.split_table_batched_embeddings_ops`. Input: hash indices (B, T, n_heads) → output: embeddings (B, T, n_heads, d_embed).
- **Expected speedup:** 2-5x on embedding lookup portion. Main gain is from optimized memory coalescing and prefetching.

### Alternative 1: hipCollections (AMD GPU Hash Tables)

- **What:** AMD's ROCm port of NVIDIA's cuCollections. Header-only C++ library for GPU-accelerated concurrent data structures (static_map, static_set).
- **URL:** https://github.com/ROCm/hipCollections
- **ROCm support:** Yes, requires HIP 7.0.2+
- **gfx1151 status:** Untested. Tested on MI200/MI300 (CDNA), no explicit RDNA mention.
- **When to use:** If the hash computation (not the gather) becomes the bottleneck, or if you need dynamic hash table resizing. Currently our hash is simple XOR (fast), so this is lower priority.
- **Integration:** C++ header-only, needs custom PyTorch extension wrapper.

### Primary (VERIFIED): Custom Fused Engram Gate+Conv Kernel (HIP)

- **What:** Fuses gate computation (dot-product → DeepSeek magnitude-preserving → sigmoid) + gated value multiply + depthwise conv1d into one HIP kernel
- **File:** `kernels/hip/fused_engram_gate_conv.py` (Variant B)
- **Speedup:** **7.4x** (0.129ms vs 0.950ms PyTorch reference) — verified on gfx1151, 2026-04-10
- **Wired into:** `models/engram.py` via try/except import, auto-used when fp16
- **Note:** Variants A (hash+gather+gate) and C (full fusion) were 5-7x SLOWER than PyTorch because they do matmuls in the kernel instead of letting rocBLAS handle them. Confirmed anti-pattern: never do matmuls in HIP kernels on gfx1151.

### Fallback: Current PyTorch Implementation

The existing `models/engram.py` is clean and correct. On gfx1151, the Engram tables are sized to fit L2 cache (~50MB fp16, ~6.3MB int4). The PyTorch `nn.Embedding` gather is already reasonably fast for small table sizes. If no external library works, this is fine — the Engram overhead is ~2-5% of total training step.

---

## 2. MoE (Mixture of Experts)

**What it does:** Top-K softmax routing → scatter tokens to K selected experts → expert forward (small SwiGLU FFN) → gather and combine results.

**Bottleneck:** Token scatter/gather for expert dispatch + small-GEMM utilization on gfx1151 (small GEMMs < fewer large GEMMs on Tensile scalar FMA).

**Current implementation:** `kernels/hip/moe_gating.py` (186 lines) — HIP kernel for routing/gating only (3.5x vs PyTorch). Expert forward uses standard `nn.Linear`.

### Primary: ScatterMoE (Triton)

- **What:** ~700-line pure Triton implementation of Sparse MoE. Fuses expert linear transforms with scatter/gather reordering. Avoids padding and unnecessary copies. Supports FSDP.
- **URL:** https://github.com/shawntan/scattermoe
- **Paper:** arXiv:2403.08245
- **ROCm support:** Pure Triton (`@triton.jit` with `tl.dot`) — should work on any Triton-supported backend
- **gfx1151 status:** Untested but promising. FLA (also pure Triton) works perfectly on gfx1151. ScatterMoE uses similar Triton patterns.
- **Integration:** Replace expert dispatch + forward + gather with `scattermoe.ParallelExperts`. Input: tokens (B*T, D) + expert indices (B*T, K) → output: combined (B*T, D).
- **Expected speedup:** 2-4x on the full MoE forward pass (fuses dispatch + expert matmul + gather). Paper reports 40% faster than Tutel.
- **Installation:** `pip install scattermoe` or build from source. May need ROCm 7.12 patching.

### Alternative 1: vLLM Fused MoE (Triton path)

- **What:** vLLM's comprehensive fused MoE with multiple backends. The Triton-based kernels fuse top-k gating + expert dispatch + FC1→activation→FC2 + result gathering.
- **URL:** https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe
- **ROCm support:** Has `rocm_aiter_fused_moe.py` but that targets gfx942/950 only. The **Triton path** (`fused_moe_triton.py`) may work on gfx1151.
- **gfx1151 status:** Untested. The Triton kernels are promising but extracting them from vLLM is complex (many dependencies).
- **When to use:** If ScatterMoE doesn't work. vLLM's Triton MoE is more battle-tested but harder to extract.
- **Integration:** Extract `fused_moe_triton.py` + dependencies. Significant effort (~2-4 hours).

### Alternative 2: Grouped GEMM via torch.compile

- **What:** Instead of external libraries, use `torch.compile` to fuse the expert forward passes. Group tokens by expert, run batched matmuls.
- **When to use:** If no external MoE library works on gfx1151. This is the simplest approach.
- **Expected speedup:** 1.5-2x (compile eliminates launch overhead but can't fuse scatter/gather with matmuls).

### Skip: Libraries That Won't Work

| Library | Why Skip |
|---------|----------|
| **MegaBlocks** (github.com/databricks/megablocks) | NVIDIA-only, open ROCm issue since March 2024 |
| **aiter FusedMoE** (github.com/ROCm/aiter) | gfx942/gfx950 only — MFMA dependency |
| **grouped_gemm** (github.com/tgale96/grouped_gemm) | NVIDIA CUTLASS only, no ROCm |

### Fallback: Current Implementation

Our `moe_gating.py` HIP kernel (3.5x for routing) + standard `nn.Linear` for expert forward is adequate. The main MoE overhead is the scatter/gather, which is ~2-5% of step time for 8 experts with top-2 routing. On gfx1151 with unified memory, scatter/gather is less severe than on discrete GPUs.

---

## 3. mHC (multi-Head Cache / multi-Head Controller)

**What it does:** 4-branch residual highway with Sinkhorn-normalized cross-branch mixing. Per sublayer: readout (weighted branch sum), process, write-in (weighted branch update), cross-branch mixing (4x4 doubly-stochastic matrix).

**Bottleneck:** Sinkhorn normalization (20 iterations on 4x4 matrices, requires fp32). The 3 projections (d_model → 4) are tiny matmuls.

**Current implementation:** None — only described in plan files (ARCHON, GENIUS-CAVEMAN, CHIMERA-ENGRAM).

### Primary (VERIFIED): Custom Fused mHC Sinkhorn Kernel (HIP)

- **What:** Fuses 3 small projections (d→4, d→4, d→16) + sigmoid/exp activations + 20-iteration Sinkhorn 4×4 normalization into one HIP kernel. One thread per token, entire 4×4 matrix in registers.
- **File:** `kernels/hip/fused_mhc_sinkhorn.py`
- **Speedup:** **28.5x** (7.8ms vs 221.9ms PyTorch reference) — verified on gfx1151, 2026-04-10
- **Correctness:** Exact match (pre=0.0, post=0.0, res=0.0). H_res rows sum to 1.0000 (doubly stochastic).
- **Why so fast:** PyTorch needs 40+ kernel launches (20 Sinkhorn iters × row_norm + col_norm). Our kernel does everything in one launch with the 4×4 matrix in 16 float registers.
- **Design:**
  ```
  Per token (one thread):
    1. Load x_bar (mean of 4 branch streams, d_model)
    2. Compute 3 matmuls: x_bar @ φ_pre(4), x_bar @ φ_post(4), x_bar @ φ_res(16)
    3. Apply activations: sigmoid(0.01*logits), exp(0.01*logits)
    4. Reshape φ_res to 4x4, run 20 Sinkhorn iterations in registers:
       for i in range(20): row_norm → col_norm (alternating)
    5. Output: H_pre(4), H_post(4), H_res(4x4) per token
  ```

### Alternative 1: Adapt ScatterMoE Dispatch Pattern

- **What:** ScatterMoE's dispatch/gather is architecturally similar to mHC (compute scores → weighted dispatch → compute → weighted combine). Adapt `ParallelLinear` for 4 fixed branches instead of dynamic experts.
- **URL:** https://github.com/shawntan/scattermoe
- **When to use:** If writing a custom HIP kernel is too much effort. ScatterMoE handles the dispatch/combine; you'd still need Sinkhorn separately.
- **Limitation:** Doesn't fuse Sinkhorn — just optimizes the dispatch/gather part.

### Alternative 2: PyTorch + torch.compile

- **What:** Implement mHC in pure PyTorch and let `torch.compile` fuse what it can. The 4x4 Sinkhorn loop may compile to efficient code if the iteration count is static.
- **When to use:** As a first implementation before writing a custom kernel. Good enough for prototyping.
- **Expected overhead:** ~0.5ms/layer (measured estimate from plan files). With compile: ~0.2-0.3ms/layer.
- **Limitation:** torch.compile may not fully fuse the Sinkhorn loop (20 iterations with alternating row/col normalization may cause graph breaks).

### Fallback: Simplified mHC (no Sinkhorn)

If Sinkhorn proves too expensive, replace the doubly-stochastic mixing with simple softmax over 4 branches:
- `H_res = softmax(logits, dim=-1)` instead of Sinkhorn
- Loses the doubly-stochastic guarantee but saves ~80% of mHC compute
- The mHC paper (2512.24880) shows H_res provides the majority of quality gain, but simple softmax may be "good enough"

---

## 4. FLA Library Updates (New Recurrence Kernels)

**What:** Flash Linear Attention now has 33 operation modules. We use 4 (HGRN, GLA, Retention, DeltaNet). New kernels could replace or improve recurrence in multiple architectures.

**URL:** https://github.com/sustcsonglin/flash-linear-attention
**ROCm support:** Yes — pure Triton, works on gfx1151 (verified with existing 4 kernels)

### New Kernels Worth Testing

| Kernel | Time (est.) | Description | Useful For |
|--------|-------------|-------------|------------|
| **RWKV7** | ~0.5ms | Latest RWKV with improved recurrence | Alternative to Griffin/HGRN in any architecture |
| **Gated Delta Rule** | ~1.0ms | More expressive than DeltaNet with gating, 6 variants | CHIMERA-ENGRAM, any delta-rule architecture |
| **FoX (forgetting_attn)** | ~0.8ms | Selective memory decay (forgetting transformer) | Novel recurrence for SPECTRAL-HYDRA decay spectrum |
| **TTT** | ~1.5ms | Test-Time Training — learns at test time | Novel: could replace Engram for online adaptation |
| **Titans** | ~1.2ms | Google's memory-augmented architecture | Alternative to Engram knowledge injection |
| **Comba** | ~1.3ms | 340M recurrence from Comba paper | Referenced in SPECTRAL-HYDRA as inspiration |

### Integration

All FLA kernels share the same interface pattern:
```python
from fla.ops.<name> import chunk_<name>, fused_recurrent_<name>
# chunk_* for training (parallel), fused_recurrent_* for inference (sequential)
output = chunk_<name>(q, k, v, ...)  # (B, T, H, D) → (B, T, H, D)
```

Update FLA: `pip install -U flash-linear-attention` (or build from source for ROCm patches).

---

## 5. Liger-Kernel (General Training Optimizations)

**What:** LinkedIn's Triton kernel collection for LLM training. Overlaps with some autokernel HIP kernels.

- **URL:** https://github.com/linkedin/Liger-Kernel
- **ROCm support:** Officially supported (ROCm 6.3+), but **mostly incompatible with gfx1151 + ROCm 7.12 + PyTorch 2.10**
- **gfx1151 status:** **TESTED 2026-04-10 — mostly broken**

### Benchmark Results (2026-04-10)

| Kernel | Liger (Triton) | AutoKernel (HIP) | PyTorch | Winner |
|--------|---------------|-------------------|---------|--------|
| **FusedLinearCE** | **CRASH** (`hipErrorIllegalAddress`) | N/A | baseline | **Not usable on gfx1151** |
| **RMSNorm** | **CRASH** (`torch.distributed.tensor.DTensor` missing) | 3.3x | baseline | **HIP** |
| **SwiGLU** | 0.301ms (1.6x) | **0.283ms (1.7x)** | 0.471ms | **HIP wins by 1.1x** |
| **CrossEntropy** | **CRASH** (API mismatch) | 1.8x | baseline | **HIP** |

**Conclusion:** Liger-Kernel is not viable on gfx1151. Only SwiGLU worked, and our HIP kernel beats it. The FusedLinearCrossEntropyLoss — the highest-value kernel — crashes with illegal memory access on both forward and backward. Root cause: Liger's Triton kernels are tested on MI250/MI300 (CDNA), not RDNA 3.5 consumer APUs.

**If we want fused linear+CE, we need to write it ourselves as a HIP kernel.**

---

## 6. ScatterMoE (Verified on gfx1151)

**Status: WORKS (2026-04-10)**

- **URL:** https://github.com/shawntan/scattermoe
- **Version:** 0.3.0 (installed via `pip install -e .`)
- **Forward + backward:** Both work on gfx1151
- **API:** `scattermoe.mlp.MLP(input_size, hidden_size, num_experts, top_k, activation=F.silu)`

```python
from scattermoe.mlp import MLP
moe = MLP(d_model, ffn, n_experts, top_k, activation=F.silu).cuda().half()
out = moe(x, topk_weights, topk_indices.int())
```

Ready to wire into CHIMERA-ENGRAM and GENIUS-CAVEMAN architectures.

---

## 7. DeepSpeed (Assessed 2026-04-10)

**Status: Not useful for our setup** (single GPU, 250M model, 128 GB unified memory, RDNA 3.5)

| Feature | Verdict | Reason |
|---------|---------|--------|
| ZeRO Stage 1/2 | Skip | Single GPU, nothing to partition |
| Activation checkpointing | Skip | Use `torch.utils.checkpoint` natively |
| FusedAdam | Skip | PyTorch `AdamW(fused=True)` is equivalent |
| Sparse attention | Skip | NVIDIA V100/A100 only |
| Flops Profiler | Low priority | Per-module FLOP breakdown, mildly useful |

DeepSpeed is already installed on remote but only the Flops Profiler has marginal value.

---

## 8. Flashback (Noted for Future)

**URL:** https://github.com/lengstrom/flashback
**What:** Fused backwards-over-backwards kernels for softmax/sigmoid attention. Enables differentiating through training steps (meta-learning, architecture search, data poisoning detection).
**Status:** JAX/Pallas only, no PyTorch, no ROCm. Not actionable now.
**Future use:** If we do neural architecture search or meta-learning, Flashback's approach to second-order derivatives through attention is relevant.

---

## 9. Top-K Sampling Bucket Sort (Deferred)

**Source:** https://codepointer.substack.com/p/llamacpp-accelerate-top-k-sampling
**What:** Replace our 20-pass binary search with 2-3 pass histogram-based threshold finding. llama.cpp achieved 2.9x speedup at k=8000.
**Our kernel:** `kernels/hip/top_k_sampling.py` uses iterative bisection (20 passes). Bucket sort needs only 2 passes (histogram + scatter).
**Status:** Deferred to inference phase. Not needed for training.

---

## Priority Summary (Updated 2026-04-10)

| Priority | Component | Best Option | Status | Expected Impact |
|----------|-----------|-------------|--------|-----------------|
| **HIGH** | MoE | **ScatterMoE** (Triton) | **VERIFIED on gfx1151** | 2-4x MoE forward |
| **HIGH** | Engram | Custom fused HIP kernel (3 variants written) | Ready to benchmark | 3-6x fused ops |
| **HIGH** | FLA update | pip install -U fla | Available | New recurrence options |
| **MEDIUM** | mHC | Custom fused HIP kernel | Not yet written | 5-10x Sinkhorn overhead |
| **LOW** | Liger-Kernel | **BROKEN on gfx1151** | Tested, not usable | N/A |
| **LOW** | Chunked Linear+CE | `kernels/hip/chunked_linear_cross_entropy.py` | **VERIFIED** — memory opt, not speed | Saves 2-12 GB, 25% slower bwd |
| **DEFERRED** | Top-K bucket sort | Custom HIP kernel | Deferred to inference | 7-10x sampling |

---

## Installation Notes

All external packages on gfx1151 need ROCm 7.12 math builtin patching (`expf` → `__builtin_expf`, etc.). See `knowledge/hardware/amd_rdna35_strix_halo.md` §7 for the pattern. Budget 1-2 hours per package for source builds.

```bash
# ScatterMoE (Triton-based, likely works)
pip install scattermoe  # or: git clone + pip install -e .

# FBGEMM_GPU (has HIP code, needs ROCm build)
git clone https://github.com/pytorch/FBGEMM.git
cd FBGEMM/fbgemm_gpu
# Follow ROCm build instructions in fbgemm_gpu/docs/

# FLA update
pip install -U flash-linear-attention

# Liger-Kernel
pip install liger-kernel
```
