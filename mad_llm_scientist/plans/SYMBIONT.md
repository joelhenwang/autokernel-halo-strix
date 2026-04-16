---
title: "SYMBIONT"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/hardware/amd_rdna35_strix_halo.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %symbiont, %cpu-gpu, %rwkv7, %apu-exploit]
---

# SYMBIONT

**CPU-GPU Cooperative Recurrence via Unified Memory — Two Processors, One Organism**

*"The lichen thrives not because the alga or fungus is strong alone, but because they share everything."*
*Strix Halo's CPU and GPU share the same memory. Why not share the same model?*

## Hypothesis

The Strix Halo APU uniquely combines a powerful GPU (40 CUs, RDNA 3.5) with a powerful CPU (16 Zen 5 cores, AVX-512) sharing unified LPDDR5X memory. No other hardware platform offers this combination. SYMBIONT exploits this by splitting the RWKV-7 architecture across both processors: the GPU handles all matrix multiplications (projections, FFN) via rocBLAS, while the CPU handles all state evolution operations (delta rule updates, gating, element-wise) via AVX-512. Communication is zero-copy through shared memory pointers. This eliminates the GPU's bottleneck (element-wise state ops competing with matmuls for CU time) and the CPU's bottleneck (matmuls too slow without matrix accelerators).

**Key papers:** "RWKV-7 Goose" (2503.14456), Strix Halo unified memory architecture

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  → 16 Cooperative Blocks (GPU + CPU pipeline):
  │
  │  ┌─── GPU (40 CUs, rocBLAS) ──────────────────────┐
  │  │ RMSNorm (autokernel HIP)                        │
  │  │ QKV projection: Linear(768 → 2304) via rocBLAS  │
  │  │ Gate projections: w, a via rocBLAS               │
  │  │ ──────── signal CPU with Q,K,V,w,a ─────────── │
  │  │                                                  │
  │  │ (while CPU computes state evolution...)          │
  │  │ Prefetch next layer's weights into L2            │
  │  │ ──────── wait for CPU output o ────────────────│
  │  │                                                  │
  │  │ Output projection: Linear(768 → 768) via rocBLAS│
  │  │ +Residual                                        │
  │  │ RMSNorm → SwiGLU FFN (768→1920→768) → +Residual│
  │  └─────────────────────────────────────────────────┘
  │
  │  ┌─── CPU (16 Zen 5, AVX-512) ────────────────────┐
  │  │ Receive Q,K,V,w,a from shared memory            │
  │  │ RWKV-7 state evolution (generalized delta rule):│
  │  │   S_t = S_{t-1}·(diag(w_t) - k̂ᵀ·(a⊙k̂)) + vᵀ·k│
  │  │ All element-wise + outer products               │
  │  │ 16 threads × AVX-512 = 256 parallel fp16 ops   │
  │  │ Write output o to shared memory                 │
  │  └─────────────────────────────────────────────────┘
  │
  │  ═══════════ SHARED LPDDR5X (128 GB) ═══════════
  │
  → Final RMSNorm → LM Head
```

### The CPU-GPU Split

**Why this works on Strix Halo and nowhere else:**

| Operation | On GPU (CUs) | On CPU (Zen 5) | Bottleneck |
|-----------|-------------|----------------|------------|
| Linear projections | **rocBLAS 3.5ms** | ~40ms (no matrix accel) | Memory BW |
| RWKV-7 state evolution | ~1.2ms (CU element-wise) | **~0.8ms (AVX-512)** | Compute |
| RMSNorm | **0.05ms (HIP kernel)** | ~0.2ms | Trivial |
| SwiGLU FFN | **rocBLAS 2.8ms** | ~35ms | Memory BW |

Key insight: RWKV-7 state evolution is **element-wise ops + outer products** — exactly what AVX-512 excels at. Moving it to CPU frees GPU CUs for the matmul pipeline. With double-buffering, GPU and CPU overlap:

```
GPU Layer N:  [proj]──[wait]──[out_proj + FFN]
CPU Layer N:       ──[state]──
GPU Layer N+1:              ──[proj]──[wait]──[out_proj + FFN]
CPU Layer N+1:                     ──[state]──
```

---

## Component 1: RWKV-7 State Evolution (CPU)

```python
# This runs on CPU via AVX-512 (torch with MKL backend)
def rwkv7_state_update_cpu(S, k, v, w, a, k_hat):
    """
    RWKV-7 generalized delta rule on CPU.
    S: (n_heads, head_dim, head_dim) state matrix
    k, v: (n_heads, head_dim) key/value vectors
    w: (n_heads, head_dim) vector-valued decay gate
    a: (n_heads, head_dim) in-context learning rate
    k_hat: (n_heads, head_dim) decoupled key for removal
    """
    # diag(w_t) - k̂ᵀ·(a ⊙ k̂)  — all element-wise
    transition = torch.diag_embed(w) - k_hat.unsqueeze(-1) * (a * k_hat).unsqueeze(-2)
    # S_t = S_{t-1} · transition + vᵀ · k
    S = torch.bmm(S, transition) + v.unsqueeze(-1) * k.unsqueeze(-2)
    # Output: o = Sᵀ · q  (q computed on GPU)
    return S
```

**CPU performance estimate:** State matrix per head = 64×64 = 4096 floats = 8KB. With 12 heads = 96KB total. This fits in Zen 5's L1 cache (64KB per core, with L2 spillover). AVX-512 processes 32 fp16 ops per instruction. State update = ~200 AVX-512 instructions per head × 12 heads = 2400 instructions. At 2.9 GHz = ~0.8μs per token. **For 1024 tokens: ~0.8ms total.**

## Component 2: GPU Projection Pipeline

```python
class SymbiontBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12, head_dim=64):
        # All projections on GPU (rocBLAS)
        self.w_qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.w_wa = nn.Linear(d_model, 2 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, 1920)
        # Token shift (RWKV-7 lerp-based temporal mixing)
        self.token_shift = nn.Parameter(torch.zeros(5, d_model))

    def forward(self, x, state):
        # GPU: projections
        x_norm = self.norm1(x)
        # RWKV-7 token shift (element-wise, free)
        x_shifted = lerp_token_shift(x_norm, self.token_shift)
        qkv = self.w_qkv(x_shifted)
        q, k, v = qkv.chunk(3, dim=-1)
        wa = self.w_wa(x_shifted)
        w, a = wa.chunk(2, dim=-1)
        w = torch.sigmoid(w)  # decay gate
        a = torch.sigmoid(a)  # learning rate

        # CPU: state evolution (zero-copy via shared memory)
        o, new_state = cpu_rwkv7_forward(q, k, v, w, a, state)

        # GPU: output projection + FFN
        h = x + self.w_o(o)
        h = h + self.ffn(self.norm2(h))
        return h, new_state
```

## Component 3: Double-Buffered Synchronization

```python
class DoubleBuffer:
    """Lock-free double buffer for CPU-GPU state exchange."""
    def __init__(self):
        self.buffer_a = None  # GPU writes, CPU reads
        self.buffer_b = None  # CPU writes, GPU reads
        self.fence = threading.Event()

    def gpu_write(self, data):
        self.buffer_a = data  # Zero-copy: pointer to shared memory
        self.fence.set()

    def cpu_read_and_process(self, state_fn):
        self.fence.wait()
        result = state_fn(self.buffer_a)
        self.buffer_b = result
        self.fence.clear()

    def gpu_read(self):
        return self.buffer_b
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_layers | 16 |
| n_heads | 12 |
| head_dim | 64 |
| d_state (per head) | 64×64 |
| ffn_inner | 1920 (2.5×) |
| token_shift_groups | 5 (RWKV-7 style) |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |
| CPU threads | 12 (leaving 4 for data loading) |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×768, tied) | 38.6M |
| **Per block:** | |
|   w_qkv (768→2304) | 1.77M |
|   w_wa (768→1536) | 1.18M |
|   w_o (768→768) | 0.59M |
|   token_shift (5×768) | 3.8K |
|   SwiGLU FFN (768→1920→768) | 4.42M |
|   RMSNorm ×2 | 1.5K |
|   **Block total** | **~7.97M** |
| **16 blocks** | **127.5M** |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **~166.1M** |

---

## Training

### Single Phase

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Full model with CPU-GPU pipeline | Learn language with cooperative recurrence |

**Fallback:** If CPU-GPU synchronization proves too complex during training, fall back to standard GPU-only RWKV-7 (all operations on GPU). The CPU optimization is an inference-time enhancement.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 6e-4 cosine → 6e-5, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 24×1024, accum=2 (48K effective) |
| Precision | fp16 mixed (GPU), fp32 state (CPU) |
| Grad clip | 1.0 |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| CPU-GPU synchronization overhead | HIGH | Double buffering + layer pipelining. Worst case: run everything on GPU (standard RWKV-7). |
| PyTorch CPU-GPU interop limitations | HIGH | Use torch tensors on shared memory (pin_memory=False on unified). May need custom C++ extension for tight synchronization. |
| Backward pass through CPU operations | HIGH | Use torch autograd normally — CPU tensors participate in autograd graph. Backward state gradients computed on CPU. |
| RWKV-7 training stability | MEDIUM | Follow RWKV-7 paper's init: A_log=-1, w_init with exp scheduling, token_shift lerp ratios. |
| CPU contention with data loading | LOW | Reserve 4 CPU cores for data loading, 12 for state evolution. Strix Halo has 16 cores. |

## Success Criteria

1. CPU-GPU overlap achieves > 50% utilization on both processors simultaneously
2. Throughput > 15K tok/s (with CPU-GPU pipelining)
3. Val loss < 3.0 on BabyLM (matching RWKV-7 quality at scale)
4. Ablation: CPU-GPU split vs GPU-only shows > 20% throughput gain
5. No synchronization deadlocks across 45 min training run

---

## Implementation Roadmap

1. Implement RWKV-7 state evolution in pure PyTorch (GPU-only first)
2. Verify RWKV-7 forward/backward correctness on GPU
3. Implement CPU state evolution using torch CPU tensors
4. Implement DoubleBuffer synchronization primitive
5. Implement SymbiontBlock with CPU-GPU pipeline
6. Profile: measure GPU idle time waiting for CPU, and vice versa
7. Tune: balance pipeline (e.g., move token_shift to CPU, move output gate to GPU)
8. Smoke test: 10 min, verify > 12K tok/s with pipeline
9. Full training: BabyLM 2 epochs
10. Ablation: GPU-only vs CPU-GPU split throughput comparison
11. If pipeline overhead too high: fall back to GPU-only RWKV-7 with FLA kernels

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), cross_entropy (1.8×), chunked_linear_cross_entropy

**External (1):** FLA RWKV/HGRN kernel (fallback if CPU pipeline fails)

**New (1):** CPU state evolution — may need custom C++/AVX-512 kernel for tight performance. PyTorch MKL handles AVX-512 transparently for standard ops.

### CPU Performance Model

| Operation | Per Token | Per Sequence (1024) | Notes |
|-----------|-----------|-------------------|-------|
| diag_embed(w) | 0.05μs | 51μs | Element-wise |
| k_hat outer product | 0.1μs | 102μs | AVX-512 native |
| bmm(S, transition) | 0.3μs | 307μs | 12 heads × 64×64 matmul |
| v outer product | 0.1μs | 102μs | AVX-512 native |
| **Total state evolution** | **0.55μs** | **~0.56ms** | Per layer |
| **16 layers** | | **~9ms** | All state work |

**GPU computation per step (excluding state):** projections + FFN ≈ 16 × (3.5 + 2.8)ms = ~100ms.
**CPU computation per step:** 16 × 0.56ms = ~9ms.
**Pipeline efficiency:** CPU finishes 11× faster than GPU. CPU is never the bottleneck.

### Throughput Estimate

| Mode | Config | Throughput |
|------|--------|------------|
| GPU-only RWKV-7, eager | All on GPU | ~9K tok/s |
| GPU-only + compile + AK | All on GPU | ~14K tok/s |
| CPU-GPU pipeline + compile | Split execution | ~18K tok/s |
| CPU-GPU + FLA fallback kernels | Hybrid | ~16K tok/s |

**Estimated optimized throughput:** ~16-18K tok/s (CPU-GPU pipeline + compile)
**Tokens in 45 min:** ~43-49M (2.7-3.0 BabyLM epochs)
**Ranking:** #6 of 31 architectures

**Note:** This is the highest-risk hypothesis but potentially the most interesting — it demonstrates a capability unique to APU hardware that no discrete GPU can replicate.
