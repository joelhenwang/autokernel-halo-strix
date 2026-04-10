# Optimization Libraries Integration + Fused Engram Kernel

**Date:** 2026-04-10
**Status:** Design approved

---

## Context

We have identified several optimization opportunities from `knowledge/mHC_MoE_Engram_optimizations.md` and external research. This spec covers installing/benchmarking external libraries and writing a custom fused Engram kernel (3 variants) to accelerate the complex architecture components (Engram, MoE) that currently limit Tier 3-4 hypothesis throughput.

Key findings driving this work:
- Engram hash+gather+gate+conv has 3+ intermediate tensors that can be eliminated via fusion
- ScatterMoE (pure Triton) is the best MoE candidate for gfx1151
- Liger-Kernel's `FusedLinearCrossEntropyLoss` eliminates the full (B*T, vocab) logits tensor (~1.6 GB saved)
- DeepSpeed offers almost nothing for single-GPU 250M on RDNA 3.5 (only Flops Profiler is mildly useful)
- Flashback is JAX-only (backwards-over-backwards for meta-learning) — noted for future reference
- Top-K bucket sort (llama.cpp technique) deferred to inference phase

---

## Part 1: Fused Engram Kernel (3 Variants)

### Current Implementation

`models/engram.py` — `EngramLayer.forward()`:
```
1. hash_indices = hash_map(input_ids)           # XOR hash, (B,T,n_heads)
2. embs = embeddings(hash_indices)              # gather, (B,T,n_heads,d_head)
3. embs_flat = embs.reshape(B,T,-1)             # (B,T,d_engram)
4. key = key_proj(embs_flat)                    # Linear, (B,T,d_model)
5. value = value_proj(embs_flat)                # Linear, (B,T,d_model)
6. query = query_norm(hidden_states)            # RMSNorm, (B,T,d_model)
7. gate_raw = (query * key).sum(-1,keepdim=True) / sqrt(D)
8. gate = sigmoid(abs(gate_raw).sqrt() * sign(gate_raw))  # DeepSeek gating
9. gated_value = gate * value                   # (B,T,d_model)
10. conv_value = short_conv(value)              # depthwise conv1d k=3
11. output = gated_value + conv_value           # (B,T,d_model)
```

### Variant A: Hash + Gather + Gate

**Fuses:** Steps 1-8 (hash → gather → project key → dot-product gate → sigmoid)
**Input:** `hidden_states (B,T,D)`, `input_ids (B,T)`, hash multipliers, embedding table weight, key_proj weight, norm weight
**Output:** `gate (B,T,1)`, `embs_flat (B,T,d_engram)` (needed for value path)
**File:** `kernels/hip/fused_engram_hash_gate.py`

Design: One block per (batch, position). Each block:
1. Compute XOR hash indices for all n-gram sizes × n_heads in registers
2. Gather from embedding table (irregular but L2-cached)
3. Flatten multi-head embeddings
4. Matrix-vector product for key projection (d_engram × d_model, held in LDS)
5. RMSNorm of hidden state (in LDS)
6. Dot product query*key, DeepSeek gating, sigmoid

**Challenge:** Irregular gather pattern. Embedding table is ~50MB fp16, larger than L2 (6 MB). But per-position access is only n_heads × n_orders entries — small working set per block.

### Variant B: Gate + Value + Conv

**Fuses:** Steps 7-11 (dot-product gate → DeepSeek gating → sigmoid → gated multiply → depthwise conv1d)
**Input:** `query (B,T,D)`, `key (B,T,D)`, `value (B,T,D)`, conv weight (D,k), conv bias (D,)
**Output:** `engram_output (B,T,D)`
**File:** `kernels/hip/fused_engram_gate_conv.py`

Design: One block per (batch, position). Each block:
1. Load query, key vectors for this position
2. Dot product → scale by 1/sqrt(D) → abs().sqrt()*sign() → sigmoid = gate scalar
3. Load value vector, multiply by gate
4. Load value vectors for positions [t-2, t-1, t] for depthwise conv1d (k=3)
5. Compute conv: sum(value[t-j] * conv_weight[j]) + bias for each channel
6. Add gated_value + conv_value → output

**Advantage:** All regular memory access. No irregular gathers. Pure element-wise + small conv.

### Variant C: Full Fusion

**Fuses:** Steps 1-11 (everything)
**Input:** `hidden_states (B,T,D)`, `input_ids (B,T)`, all weights
**Output:** `engram_output (B,T,D)`
**File:** `kernels/hip/fused_engram_full.py`

Design: Combines A + B. One block per (batch, position). Holds all intermediates in registers/LDS. No global memory writes until final output.

**Challenge:** Large register pressure. Many weights to load (embedding table, key_proj, value_proj, conv, norm). May need to split into sub-kernels within a single launch.

### Benchmark

**File:** `scripts/bench_engram_kernels.py`

Compare all 3 variants + PyTorch baseline:
- Shapes: B=16, T=256, D=1024, d_engram=512, n_heads=8, ngram_sizes=[2,3], table_size=65536
- Measure: forward time, backward time (PyTorch autograd for all), correctness (atol vs reference)
- Report: speedup vs PyTorch, memory savings

---

## Part 2: ScatterMoE Install + Test

**What:** Pure Triton MoE that fuses expert dispatch + forward + gather.
**URL:** https://github.com/shawntan/scattermoe
**Why:** Our `moe_gating.py` only accelerates routing (3.5x). ScatterMoE fuses the full MoE forward.

### Steps
1. Clone on remote: `git clone https://github.com/shawntan/scattermoe.git`
2. Install: `cd scattermoe && pip install -e .`
3. Test import: `python -c "from scattermoe import ParallelExperts; print('OK')"`
4. Benchmark: Compare vs standard `nn.Linear` experts + our `moe_gating` on shapes matching CHIMERA-ENGRAM (8 experts, top-2, d=1024, ffn=1024)
5. If works: document integration pattern for CHIMERA-ENGRAM and GENIUS-CAVEMAN

**Risk:** Triton on gfx1151 is less mature than CDNA. May need ROCm 7.12 patching.

---

## Part 3: Liger-Kernel — FusedLinearCrossEntropyLoss

**What:** Fused LM head matmul + cross-entropy loss. Avoids materializing full (B*T, vocab) logits.
**URL:** https://github.com/linkedin/Liger-Kernel
**ROCm:** Officially supported since v0.4.0

### Why This Is the Big Win

For a 250M model with vocab=50257, batch=16, seq=1024:
- Standard: `logits = lm_head(hidden)` → (16384, 50257) tensor = **1.6 GB** fp16
- Fused: computes loss in chunks, never materializes full logits tensor
- Saves: ~1.6 GB memory + one full DRAM round-trip

### Steps
1. Install on remote: `pip install liger-kernel`
2. Test import: `python -c "from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; print('OK')"`
3. Benchmark: Compare fused vs standard `nn.Linear` + `nn.CrossEntropyLoss` on our shapes
4. Integrate into training: replace loss computation in `halo_training/trainer.py`
5. Also benchmark Liger's RMSNorm, SwiGLU, RoPE, fused_add_rmsnorm against our HIP kernels — use winner for each

### Integration Pattern
```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

loss_fn = LigerFusedLinearCrossEntropyLoss()
# Instead of: logits = lm_head(hidden); loss = F.cross_entropy(logits, targets)
# Do:         loss = loss_fn(hidden, lm_head.weight, targets)
```

---

## Part 4: DeepSpeed (Minimal)

**Assessment:** Almost nothing useful for single-GPU 250M on RDNA 3.5.

| Feature | Verdict | Reason |
|---------|---------|--------|
| ZeRO Stage 1/2 | Skip | Single GPU, nothing to partition |
| Activation checkpointing | Skip | Use `torch.utils.checkpoint` natively if needed |
| FusedAdam | Skip | PyTorch `AdamW(fused=True)` is equivalent |
| Sparse attention | Skip | NVIDIA-only (V100/A100) |
| Curriculum learning | Skip | Not implemented in DeepSpeed |
| **Flops Profiler** | **Low priority** | Per-module FLOP breakdown, mildly useful |

### Only Action
- Try `deepspeed.profiling.flops_profiler.FlopsProfiler` on one model to get per-module FLOP breakdown
- Document results in knowledge base
- No ongoing integration needed

---

## Part 5: Knowledge Capture

Update `knowledge/mHC_MoE_Engram_optimizations.md` with:
- Flashback: JAX-only, backwards-over-backwards for meta-learning/arch search. Not actionable now. Reference for future.
- DeepSpeed: Not useful for our setup (single GPU, 250M, RDNA 3.5). Only Flops Profiler has marginal value.
- Top-K bucket sort: llama.cpp technique — histogram-based threshold finding in 2-3 passes vs our 20-pass binary search. 7-10x potential speedup. Deferred to inference phase.
- Liger-Kernel: FusedLinearCrossEntropyLoss verified as high-value. Also has 28 Triton kernels, ROCm supported.

---

## Files to Create

| File | Purpose |
|------|---------|
| `kernels/hip/fused_engram_hash_gate.py` | Variant A: hash + gather + gate |
| `kernels/hip/fused_engram_gate_conv.py` | Variant B: gate + value + conv |
| `kernels/hip/fused_engram_full.py` | Variant C: full fusion |
| `scripts/bench_engram_kernels.py` | Benchmark all 3 variants + PyTorch baseline |
| `scripts/bench_liger_kernels.py` | Benchmark Liger vs autokernel HIP kernels |

## Files to Modify

| File | Change |
|------|--------|
| `kernels/hip/_torch_ops.py` | Register winning Engram variant with autograd |
| `models/engram.py` | Wire winning variant via try/except import |
| `halo_training/trainer.py` | Integrate FusedLinearCrossEntropyLoss (if Liger works) |
| `knowledge/mHC_MoE_Engram_optimizations.md` | Update with all findings |

---

## Verification

1. All 3 Engram kernel variants compile on gfx1151
2. Correctness: max_diff < 0.01 vs PyTorch reference for all variants
3. Benchmark table shows speedup for each variant
4. ScatterMoE imports and runs a forward pass on gfx1151
5. Liger `FusedLinearCrossEntropyLoss` produces correct loss on our vocab/batch shapes
6. Liger kernel head-to-head vs autokernel HIP: documented winner for each op
