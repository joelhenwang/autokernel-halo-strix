---
title: "HELIX"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - mad_llm_scientist/plans/SYMBIONT.md
  - mad_llm_scientist/plans/CHRONOS.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %helix, %cpu-gpu, %plstm, %unified-memory, %dual-clock, %apu-native]
---

# HELIX

**CPU pLSTM Slow-Track + GPU Fast Recurrence — The Double Helix of Dual-Clock Processing**

*"DNA's power comes from two strands twisted together — neither strand alone carries the message, but together they encode life. HELIX intertwines CPU and GPU into one organism."*
*pLSTM proved parallelizable LSTM is practical. SmallThinker proved architecture-for-hardware design wins.*

## Hypothesis

Strix Halo's 128GB unified LPDDR5X is shared between CPU (16 Zen 5 cores, AVX-512) and GPU (40 CUs) at the same 240 GB/s. **No other consumer GPU has this.** HELIX is the first architecture that treats CPU and GPU as co-equal co-processors running DIFFERENT recurrences in TRUE parallelism on the same data — not a fallback, but a design choice. The GPU runs fast token-by-token KDA recurrence (capturing local patterns). The CPU runs a pLSTM (parallelizable LSTM) on 32-token summaries (capturing document-level structure). Both tracks share the same LPDDR5X — no data movement. The CPU track is "free" additional computation that runs in parallel with GPU, providing a dual-timescale inductive bias that improves quality at zero throughput cost.

**Key papers:** "pLSTM: Parallelizable Linear Source Transition Mark Networks" (2506.11997, Jun 2025), "SmallThinker" (2507.20984, Jul 2025)

---

## Architecture

```
Tokens -> Embedding (d=768, tied LM head, vocab=50257)
  |
  -> Fork into two tracks (both read from same LPDDR5X):
  |
  | GPU Track (fast, every token):               CPU Track (slow, every 32 tokens):
  | 12 KDA Layers (d=768)                        4 pLSTM Layers (d=384)
  |   KDA channel-wise recurrence                  pLSTM on chunk summaries
  |   + EFLA exact dynamics                        Learned compression (32 tok -> 1)
  |   + SwiGLU FFN                                 ReLU^2 FFN (lighter)
  |   Process: every token                         Process: 1 summary per 32 tokens
  |   Latency: ~0.05ms/token                       Latency: ~0.4ms/summary
  |   Captures: local syntax, morphology           Captures: topic, coherence, structure
  |
  -> Fusion Point (every 32 tokens):
  |   CPU summary -> FiLM modulation on GPU hidden state
  |   h_gpu = gamma * h_gpu + beta
  |   where (gamma, beta) = Linear(h_cpu)
  |   Zero-copy: both read/write same LPDDR5X
  |
  -> Final RMSNorm -> LM Head
```

### Why CPU + GPU Cooperative

**On discrete GPU (NVIDIA):**
```
GPU (HBM) <--PCIe 16x--> CPU (DDR5)
   ~3 TB/s                  ~100 GB/s
   Bottleneck: PCIe = 64 GB/s bidirectional
   CPU track adds latency, not free compute
```

**On Strix Halo (APU):**
```
GPU (40 CUs) --- shared LPDDR5X (240 GB/s) --- CPU (16 Zen 5)
   Both access same physical memory
   Zero-copy tensors: cpu_tensor.data_ptr() == gpu_tensor.data_ptr()
   CPU compute runs IN PARALLEL with GPU compute
   CPU track is genuinely free additional processing
```

This is the **only architecture in our portfolio that requires unified memory** and cannot be replicated on discrete GPUs.

### Why pLSTM, Not RWKV-7

SYMBIONT and CHRONOS in existing plans use RWKV-7 on CPU. Problems:
1. RWKV-7's generalized delta rule is sequential per-token — CPU must process tokens one-by-one
2. CPU at ~5 GHz processes ~0.1ms per RWKV-7 step — too slow for per-token processing

pLSTM is fundamentally different:
1. **Parallelizable across sequence** — processes a batch of 32 summaries simultaneously
2. **Gate operations use FMA** — Zen 5 AVX-512 is excellent at FMA (double-pumped, ~60 GFLOPS)
3. **LSTM gates are naturally vectorizable** — perfect for SIMD
4. **Works on summaries, not raw tokens** — 32x fewer steps than per-token RWKV-7

---

## Component 1: GPU Track (KDA + EFLA)

```python
class GPUTrack(nn.Module):
    def __init__(self, d_model=768, n_layers=12):
        self.layers = nn.ModuleList([
            KDABlock(d_model) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

Standard KDA layers from BIFROST, running on GPU. 12 layers x d=768.

## Component 2: Chunk Summarizer

```python
class ChunkSummarizer(nn.Module):
    """Compress 32 tokens into 1 summary vector for CPU track."""
    def __init__(self, d_model=768, d_cpu=384, chunk_size=32):
        self.chunk_size = chunk_size
        self.compress = nn.Linear(d_model * chunk_size, d_cpu, bias=False)
        # Alternatively: learned cross-attention query
        self.query = nn.Parameter(torch.randn(1, 1, d_cpu) * 0.02)
        self.k_proj = nn.Linear(d_model, d_cpu, bias=False)
        self.v_proj = nn.Linear(d_model, d_cpu, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        n_chunks = T // self.chunk_size
        x_chunks = x[:, :n_chunks * self.chunk_size].view(B, n_chunks, self.chunk_size, D)
        # Cross-attention: one query attends to 32 tokens
        k = self.k_proj(x_chunks)  # (B, n_chunks, 32, d_cpu)
        v = self.v_proj(x_chunks)
        q = self.query.expand(B, n_chunks, -1, -1)  # (B, n_chunks, 1, d_cpu)
        attn = F.scaled_dot_product_attention(q, k, v).squeeze(2)  # (B, n_chunks, d_cpu)
        return attn
```

## Component 3: CPU Track (pLSTM)

```python
class CPUTrack(nn.Module):
    """Runs on CPU via torch.device('cpu'). Processes chunk summaries."""
    def __init__(self, d_cpu=384, n_layers=4, ffn_inner=768):
        self.layers = nn.ModuleList([
            pLSTMBlock(d_cpu, ffn_inner) for _ in range(n_layers)
        ])

    def forward(self, summaries):
        # summaries: (B, n_chunks, d_cpu) — already on CPU via zero-copy
        h = summaries
        for layer in self.layers:
            h = layer(h)
        return h  # (B, n_chunks, d_cpu)


class pLSTMBlock(nn.Module):
    """Parallelizable LSTM block for CPU execution."""
    def __init__(self, d_cpu=384, ffn_inner=768):
        self.norm1 = nn.LayerNorm(d_cpu)
        self.norm2 = nn.LayerNorm(d_cpu)
        # pLSTM: parallel gates via chunked computation
        self.w_gates = nn.Linear(d_cpu, 4 * d_cpu, bias=True)  # i, f, o, g
        self.ffn_up = nn.Linear(d_cpu, ffn_inner, bias=False)
        self.ffn_down = nn.Linear(ffn_inner, d_cpu, bias=False)

    def forward(self, x):
        h = self.norm1(x)
        gates = self.w_gates(h)
        i, f, o, g = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        # Parallel scan for LSTM cell state
        c = parallel_lstm_scan(f, i * g)
        h_out = o * torch.tanh(c)
        x = x + h_out
        h2 = self.norm2(x)
        x = x + self.ffn_down(F.relu(self.ffn_up(h2)).square())
        return x
```

## Component 4: FiLM Fusion

```python
class FiLMFusion(nn.Module):
    """Modulate GPU hidden state with CPU summary."""
    def __init__(self, d_gpu=768, d_cpu=384):
        self.proj = nn.Linear(d_cpu, 2 * d_gpu, bias=True)
        # Initialize as identity: gamma=1, beta=0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:d_gpu] = 1.0  # gamma init

    def forward(self, h_gpu, h_cpu, chunk_size=32):
        B, T, D = h_gpu.shape
        n_chunks = T // chunk_size
        # Expand CPU summary to match GPU token resolution
        gamma_beta = self.proj(h_cpu)  # (B, n_chunks, 2*d_gpu)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # Repeat for each token in chunk
        gamma = gamma.unsqueeze(2).expand(-1, -1, chunk_size, -1).reshape(B, T, D)
        beta = beta.unsqueeze(2).expand(-1, -1, chunk_size, -1).reshape(B, T, D)
        return gamma * h_gpu + beta
```

## Component 5: Helix Model Assembly

```python
class HelixModel(nn.Module):
    def __init__(self, d_model=768, d_cpu=384, chunk_size=32):
        self.embedding = nn.Embedding(50257, d_model)
        self.summarizer = ChunkSummarizer(d_model, d_cpu, chunk_size)
        self.gpu_track = GPUTrack(d_model, n_layers=12)
        self.cpu_track = CPUTrack(d_cpu, n_layers=4)
        self.fusion = FiLMFusion(d_model, d_cpu)
        self.chunk_size = chunk_size
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, 50257, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        h = self.embedding(input_ids)

        # CPU track: compress + process (runs in parallel with GPU)
        summaries = self.summarizer(h)  # (B, T//32, 384)
        # In practice, launch CPU track as async, overlap with GPU track
        cpu_output = self.cpu_track(summaries)

        # GPU track: full token processing
        h = self.gpu_track(h)

        # Fusion: CPU modulates GPU
        h = self.fusion(h, cpu_output, self.chunk_size)

        return self.lm_head(self.final_norm(h))
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model (GPU) | 768 |
| d_cpu | 384 |
| n_layers_gpu | 12 |
| n_layers_cpu | 4 |
| n_heads (GPU) | 12 |
| head_dim | 64 |
| ffn_inner_gpu | 1920 (2.5x) |
| ffn_inner_cpu | 768 (2x) |
| chunk_size | 32 (summary compression ratio) |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257x768, tied) | 38.6M |
| **GPU Track (12 KDA layers):** | |
|   Per KDA block: ~7.02M x 12 | 84.2M |
| **Chunk Summarizer:** | |
|   k_proj + v_proj (768->384 each) | 0.59M |
|   query param | 384 |
| **CPU Track (4 pLSTM layers):** | |
|   Per pLSTM block: w_gates (384->1536) + FFN (384->768->384) | |
|   = 0.59M + 0.59M = 1.18M x 4 | 4.72M |
| **FiLM Fusion:** | |
|   proj (384->1536) | 0.59M |
| Final RMSNorm | 768 |
| **GRAND TOTAL** | **~128.7M** |

Lean at 129M. The CPU track adds only ~5.3M params. Well under 175M budget.

---

## Training

### Single Phase (with CPU-GPU overlap)

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (100%) | 45 min | Both tracks, FiLM fusion | Learn dual-clock dynamics |

### CPU-GPU Overlap Strategy

```python
# Pseudocode for training step with overlap
def training_step(batch):
    h = embedding(batch)
    
    # Launch CPU track asynchronously
    summaries = summarizer(h)
    cpu_future = torch.jit.fork(cpu_track, summaries.cpu())
    
    # GPU track runs in parallel
    h = gpu_track(h)
    
    # Wait for CPU track, fuse
    cpu_output = torch.jit.wait(cpu_future).to(h.device)
    h = fusion(h, cpu_output)
    
    logits = lm_head(final_norm(h))
    return loss(logits, targets)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (both tracks) |
| LR | 8e-4 cosine -> 8e-5, 200-step warmup |
| Weight decay | 0.1 |
| Batch | 24x1024, accum=2 (48K effective) |
| Precision | fp16 mixed (GPU), fp32 (CPU track — AVX-512 is fp32-native) |
| Grad clip | 1.0 |
| FiLM init | identity (gamma=1, beta=0) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| PyTorch autograd spanning CPU+GPU is untested on ROCm | HIGH | Start with sequential (CPU after GPU, no overlap). Add async overlap in Phase 2 if sequential works. |
| torch.compile breaks on mixed CPU/GPU graphs | HIGH | Compile GPU track and CPU track separately. Fusion point is simple element-wise (compile-friendly). |
| CPU track too slow (bottlenecks GPU) | MEDIUM | pLSTM on 32 summaries with 4 layers: ~2ms. GPU processes 1024 tokens in ~25ms. CPU finishes 12x faster. |
| FiLM fusion is too weak (CPU info doesn't transfer) | MEDIUM | Ablation: FiLM vs cross-attention fusion. If FiLM too weak, add a lightweight cross-attention at fusion point. |
| Gradient through CPU track is slow | MEDIUM | CPU backward is fp32 on Zen 5 — slower than GPU. But CPU track is 4% of total params — small gradient. |

## Success Criteria

1. Val loss < 2.88 on BabyLM (**beat AMADEUS** — dual-timescale advantage)
2. GPU throughput > 15K tok/s (CPU track truly runs in parallel, zero overhead)
3. Ablation: HELIX > GPU-only by > 1.5% loss improvement
4. CPU track utilization > 80% (not idle waiting for GPU)
5. FiLM modulation is non-trivial (gamma != 1, beta != 0 after training)
6. Paragraph-level coherence improved (qualitative eval)

---

## Implementation Roadmap

1. Implement ChunkSummarizer (cross-attention compression)
2. Implement pLSTMBlock with parallel scan for LSTM cell state
3. Implement CPUTrack (4 pLSTM layers)
4. Implement GPUTrack (12 KDA layers, reuse from BIFROST)
5. Implement FiLMFusion with identity initialization
6. Assemble HelixModel with sequential execution (no overlap first)
7. Verify: CPU track processes 32 summaries in < 3ms
8. Verify: parameter count ~129M
9. Smoke test: 10 min, verify both tracks get gradient
10. Add async CPU-GPU overlap via torch.jit.fork
11. Full training: BabyLM 2 epochs
12. Ablation: HELIX vs GPU-only (disable CPU track) vs CPU-only

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Kernel Reuse

**Reuse (4):** fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x), cross_entropy (1.8x), chunked_linear_cross_entropy

**External (2):** FLA DeltaNet kernel (KDA heads), causal-conv1d (ShortConv)

**New (0):** pLSTM on CPU uses standard PyTorch ops + AVX-512. FiLM fusion is element-wise.

### CPU Compute Budget

```
Zen 5 @ 5 GHz, 16 cores, AVX-512 (double-pumped):
  FP32 peak: ~60 GFLOPS (AVX-512 = 16 FP32 FMA/cycle, double-pumped)
  
pLSTM per summary (d=384):
  Gates: 384 x 1536 matmul = 1.18M FLOPs
  FFN: 384 x 768 + 768 x 384 = 0.59M FLOPs
  Total per layer: ~1.77M FLOPs
  4 layers: 7.08M FLOPs
  32 summaries: 226M FLOPs

Time at 60 GFLOPS: 226M / 60G = 3.8ms

GPU processes 1024 tokens in ~25ms -> CPU finishes 6.6x faster
CPU is never the bottleneck.
```

### Unified Memory Advantage

```
NVIDIA (discrete):
  GPU writes summaries to HBM (1 us)
  DMA to CPU RAM via PCIe (64 GB/s -> 0.05ms for 3MB)
  CPU processes
  DMA back to GPU (0.05ms)
  Total overhead: ~0.1ms per fusion point
  
Strix Halo (unified):
  GPU writes summaries to LPDDR5X (same as CPU memory)
  CPU reads from LPDDR5X (zero-copy, 0 latency)
  CPU processes
  GPU reads result (zero-copy, 0 latency)
  Total overhead: 0ms
```

### Throughput Estimate

| Mode | Config | GPU tok/s | CPU overhead |
|------|--------|-----------|-------------|
| Sequential (no overlap) | eager | ~9K | ~5% slowdown |
| Sequential | compile + AK | ~14K | ~3% slowdown |
| **Overlapped** | compile + AK | **~15K** | **0% (parallel)** |
| Overlapped | + FLA kernels | **~17K** | 0% |

**Estimated optimized throughput:** ~15-17K tok/s (CPU truly parallel)
**Tokens in 45 min:** ~41-46M (2.5-2.9 BabyLM epochs)
**Ranking:** #6-7 of all architectures

Note: throughput is modest (12 GPU layers, not looped) but the CPU track provides quality improvement at zero throughput cost. The quality-per-parameter is the metric that matters here.
