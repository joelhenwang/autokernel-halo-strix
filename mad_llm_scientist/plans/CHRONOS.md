---
title: "CHRONOS"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/hardware/amd_rdna35_strix_halo.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %chronos, %dual-clock, %cpu-world-model, %apu-moonshot]
---

# CHRONOS

**Dual-Clock CPU/GPU Architecture — The God of Time Who Sees All Scales**

*"Chronos moves at two speeds: the swift hand of seconds (GPU) and the slow hand of ages (CPU). Together they tell perfect time."*
*This architecture is physically impossible on any hardware except an APU with unified memory.*

## Hypothesis

The most radical hypothesis in the portfolio. CHRONOS runs **two models simultaneously** on Strix Halo's two processors, sharing unified LPDDR5X memory:

1. **GPU model (fast clock):** Small, fast autoregressive transformer with sliding-window attention (512 tokens). Handles next-token prediction at high throughput. This is the "working memory."
2. **CPU model (slow clock):** Persistent RWKV-7 world model running on 16 Zen 5 cores with AVX-512. Compresses the **entire conversation history** into a fixed-size state matrix. This is the "long-term memory."

Every 4 layers, the GPU model cross-attends to the CPU world model's current state (stored in shared LPDDR5X — zero copy). The CPU model updates asynchronously, always lagging a few tokens behind the GPU.

This architecture exploits the **unique APU property** of simultaneous CPU-GPU execution with shared memory. On discrete GPUs (NVIDIA, AMD MI), the PCIe bus makes this prohibitively slow. On other APUs, the CPU is too weak (no AVX-512, fewer cores).

**Key inspiration:** Hippocampus (fast, episodic, GPU) + Neocortex (slow, semantic, CPU) dual-system theory of memory.

---

## Architecture

```
═══════════════════ SHARED LPDDR5X (128 GB) ═══════════════════

GPU MODEL (Fast Clock — 40 CUs, RDNA 3.5):
  Tokens → Embedding (d=512, tied LM head, vocab=50257)
    │
    → 12 Fast Transformer Blocks:
    │     RMSNorm → SWA (w=512, GQA 6Q/2KV, hd=64) + RoPE
    │     +Residual → RMSNorm → SwiGLU FFN (512→1280→512)
    │     +Residual
    │
    │     ★ Every 4th layer (layers 4, 8, 12):
    │     ★ Cross-Attention to CPU World State
    │     ★ Q from GPU hidden, KV from world_state matrix
    │
    → Final RMSNorm → LM Head

CPU WORLD MODEL (Slow Clock — 16 Zen 5 cores, AVX-512):
  Receives: GPU hidden states (buffered, delayed ~4 tokens)
    │
    → 6 RWKV-7 Layers (d=512, 8 heads, hd=64):
    │     Token shift → QKV projection
    │     Generalized delta rule state evolution
    │     ReLU² FFN (512→1280→512)
    │
    → World State Matrix: (512 × 512) = 1MB
    │ Written to shared memory after each token
    │
    ★ GPU reads world_state at will (zero copy) ★

═══════════════════════════════════════════════════════════════
```

### Timing Diagram

```
Time →  token 1    token 2    token 3    token 4    token 5
GPU:    [gen]      [gen]      [gen]      [gen+xattn] [gen]
                                          ↑ reads world_state
CPU:    [                update world model from tokens 1-4          ]
                                                        [update 5-8]
Memory: ═══════ world_state_A (written by CPU) ══════════════════════
```

The GPU generates tokens at full speed. The CPU processes them in batches (4 tokens at a time) to update the world state. Cross-attention happens at GPU layers 4, 8, 12 — reading whatever world state is currently available.

---

## Component 1: GPU Fast Transformer

```python
class FastTransformerBlock(nn.Module):
    """Small, fast transformer block with sliding window attention."""
    def __init__(self, d_model=512, n_q_heads=6, n_kv_heads=2, head_dim=64, window=512):
        self.norm1 = nn.RMSNorm(d_model)
        self.w_qkv = nn.Linear(d_model, (n_q_heads + 2*n_kv_heads) * head_dim, bias=False)
        self.w_o = nn.Linear(n_q_heads * head_dim, d_model, bias=False)
        self.rotary = RotaryEmbedding(head_dim)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, 1280)
        self.window = window

    def forward(self, x, world_state=None, use_cross_attn=False):
        # Self-attention with sliding window
        h = x + self.self_attn(self.norm1(x))

        # Cross-attention to world state (every 4th layer)
        if use_cross_attn and world_state is not None:
            h = h + self.cross_attn(h, world_state)

        h = h + self.ffn(self.norm2(h))
        return h
```

## Component 2: Cross-Attention to World State

```python
class WorldStateCrossAttention(nn.Module):
    """GPU queries attend to CPU world model's state matrix."""
    def __init__(self, d_model=512, n_heads=4, head_dim=64, world_dim=512):
        self.norm = nn.RMSNorm(d_model)
        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_kv = nn.Linear(world_dim, 2 * n_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.gate = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.gate.bias, -2.0)  # Initially nearly closed

    def forward(self, x, world_state):
        """
        x: (B, T, d_model) — GPU hidden states
        world_state: (B, world_dim, world_dim) — CPU world model state
        """
        B, T, D = x.shape
        x_norm = self.norm(x)

        q = self.w_q(x_norm).view(B, T, 4, 64)  # (B, T, H, hd)

        # Treat world_state rows as key-value pairs
        # world_state: (B, 512, 512) → treat 512 rows as 512 "memory tokens"
        ws_flat = world_state  # (B, 512, 512)
        kv = self.w_kv(ws_flat)  # (B, 512, 2*4*64)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, 512, 4, 64)
        v = v.view(B, 512, 4, 64)

        # Standard cross-attention (Q from GPU, KV from CPU world state)
        # Uses SDPA — 512 KV tokens is small, fast
        attn_out = F.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        ).transpose(1,2).flatten(-2,-1)

        # Gated residual (starts near-zero, learns to open)
        g = torch.sigmoid(self.gate(x))
        return g * self.w_o(attn_out)
```

## Component 3: CPU World Model (RWKV-7)

```python
class CPUWorldModel(nn.Module):
    """RWKV-7 running on CPU, updating world state asynchronously."""
    def __init__(self, d_model=512, n_layers=6, n_heads=8, head_dim=64):
        self.layers = nn.ModuleList([
            RWKV7Block(d_model, n_heads, head_dim) for _ in range(n_layers)
        ])
        self.world_proj = nn.Linear(d_model, d_model, bias=False)
        # State: accumulated world knowledge
        self.register_buffer('world_state', torch.zeros(1, d_model, d_model))
        self.decay = nn.Parameter(torch.tensor(0.99))

    def update(self, tokens_hidden):
        """
        Process a batch of token hidden states to update world model.
        tokens_hidden: (B, chunk_size, d_model) from GPU's last hidden layer
        """
        h = tokens_hidden.to('cpu')  # Zero-copy on unified memory!
        states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            h, states[i] = layer(h, states[i])

        # Update world state: exponential moving average
        new_info = self.world_proj(h.mean(dim=1))  # (B, d_model)
        world_update = new_info.unsqueeze(-1) @ new_info.unsqueeze(-2)  # outer product
        self.world_state = self.decay * self.world_state + (1 - self.decay) * world_update

        return self.world_state
```

## Component 4: Asynchronous Coordinator

```python
import threading

class AsyncCoordinator:
    """Manages asynchronous CPU-GPU execution."""
    def __init__(self, cpu_model):
        self.cpu_model = cpu_model
        self.buffer_queue = []
        self.world_state_cache = None
        self.lock = threading.Lock()
        self.cpu_thread = None

    def submit_tokens(self, hidden_states):
        """GPU submits hidden states for CPU processing."""
        self.buffer_queue.append(hidden_states.detach())
        if len(self.buffer_queue) >= 4:  # Process in chunks of 4 tokens
            batch = torch.cat(self.buffer_queue, dim=1)
            self.buffer_queue = []
            # Launch CPU update in background thread
            self.cpu_thread = threading.Thread(
                target=self._cpu_update, args=(batch,)
            )
            self.cpu_thread.start()

    def _cpu_update(self, batch):
        with torch.no_grad():  # CPU model doesn't need training gradients
            new_state = self.cpu_model.update(batch)
            with self.lock:
                self.world_state_cache = new_state

    def get_world_state(self):
        """GPU reads latest world state (non-blocking)."""
        with self.lock:
            return self.world_state_cache
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| **GPU Model:** | |
| d_model | 512 |
| n_layers | 12 |
| SWA: n_q/n_kv/hd | 6/2/64, window=512 |
| cross_attn_layers | [4, 8, 12] |
| ffn_inner | 1280 (2.5×) |
| **CPU World Model:** | |
| d_model | 512 |
| n_layers | 6 |
| RWKV-7 heads | 8, hd=64 |
| world_state_dim | 512×512 |
| update_chunk | 4 tokens |
| **Shared:** | |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes (GPU only) |

## Parameter Count

| Component | Params |
|-----------|--------|
| **GPU Model:** | |
| Embedding (50257×512, tied) | 25.7M |
| Per SWA block (w_qkv + w_o + FFN + norms) | ~4.0M |
| Per cross-attn module (w_q + w_kv + w_o + gate) | ~0.66M |
| 12 SWA blocks | 48.0M |
| 3 cross-attn modules | 2.0M |
| Final RMSNorm | 512 |
| **GPU subtotal** | **~75.7M** |
| **CPU World Model:** | |
| Per RWKV-7 block (qkv + wa + o + FFN + shifts) | ~4.2M |
| 6 RWKV-7 blocks | 25.2M |
| world_proj (512→512) | 0.26M |
| **CPU subtotal** | **~25.5M** |
| **GRAND TOTAL** | **~101.2M** |

Extremely lean: 101M total params. GPU reads ~75M (150MB fp16) per forward. CPU reads ~25M (50MB) per update.

---

## Training

### 3 Phases

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (50%) | 22 min | GPU model only (SWA + FFN, no cross-attn) | Learn standard language modeling |
| 2 (30%) | 14 min | + CPU world model (frozen GPU, train CPU) | Learn world state compression |
| 3 (20%) | 9 min | Full pipeline: GPU + CPU + cross-attn | Learn to use world state for generation |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| GPU Optimizer | AdamW, LR 1e-3 cosine → 1e-4 |
| CPU Optimizer | AdamW, LR 3e-4 cosine → 3e-5 |
| Weight decay | 0.1 |
| GPU Batch | 32×1024, accum=1 (32K effective) |
| CPU Batch | 32×4 (chunk of 4 tokens processed per update) |
| GPU Precision | fp16 mixed |
| CPU Precision | fp32 (Zen 5 native) |
| Grad clip | 1.0 |
| Cross-attn gate init | bias=-2.0 (nearly closed) |
| World state decay | 0.99 (learned) |

### Training Strategy

**Phase 1:** Standard transformer training. GPU model should achieve ~3.2 val loss (competitive for 75M params).

**Phase 2:** Freeze GPU model. Train CPU world model to compress hidden states. Loss: auxiliary "world state prediction" loss — can the world state reconstruct the hidden state statistics?

**Phase 3:** Unfreeze everything. Joint training. GPU model learns to leverage world state via cross-attention. Loss: standard CE + 0.1× world prediction aux loss.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| CPU-GPU synchronization complexity | VERY HIGH | Start with synchronous execution (no threading). Add async only after correctness verified. |
| PyTorch CPU-GPU tensor interop | HIGH | All tensors on same device (cuda). CPU model uses .to('cpu') which is zero-copy on unified memory. Test this assumption. |
| CPU world model too slow to keep up | MEDIUM | CPU processes 4 tokens while GPU generates next 4. CPU takes ~2ms for 4 tokens (6 layers × 0.3ms). GPU takes ~20ms for 4 tokens. CPU is 10× faster. |
| Cross-attention to stale world state | LOW | World state is an EMA — naturally smooth. 4-token lag is negligible for long-term memory. |
| Small GPU model (75M) limits quality | MEDIUM | 75M transformer at 512d is GPT-2 small class. With world state context: effective capacity much higher. |
| Threading + CUDA + PyTorch interaction | HIGH | Use torch.no_grad() for CPU model (no autograd through async boundary). GPU trains end-to-end. |
| Training backward pass through cross-attn | MEDIUM | Detach world_state from CPU graph. GPU gradients flow through cross-attn Q/O projections only. CPU trained with separate loss. |

## Success Criteria

1. GPU throughput > 25K tok/s (without CPU overhead)
2. Full pipeline throughput > 20K tok/s (with CPU async)
3. Val loss < 3.0 on BabyLM (world state improves over GPU-only)
4. Cross-attn gate opens > 0.3 average (world state is being used)
5. Ablation: with world model > without world model by > 2% loss improvement
6. Zero deadlocks/crashes across 45 min run

---

## Implementation Roadmap

1. Implement GPU Fast Transformer (12-layer SWA + SwiGLU) — standard, well-understood
2. Train GPU model alone (Phase 1), verify ~3.2 val loss, ~30K tok/s
3. Implement CPU RWKV-7 World Model (6-layer, d=512)
4. Implement WorldStateCrossAttention module
5. Test: GPU reads CPU world_state tensor — verify zero-copy on unified memory
6. Implement synchronous pipeline (no threading) — verify correctness
7. Phase 2: train CPU world model (frozen GPU)
8. Phase 3: joint training with cross-attention
9. Implement AsyncCoordinator with threading (if synchronous is too slow)
10. Measure: CPU utilization, GPU utilization, world_state read latency
11. Ablation: with vs without world model on val loss

---

## Hardware Optimization Notes (Strix Halo gfx1151)

### Why This Only Works on Strix Halo

| Capability | Strix Halo | Discrete GPU (RTX 4090) | Other APU (M3 Max) |
|-----------|------------|------------------------|-------------------|
| Unified memory | **128GB LPDDR5X** | 24GB VRAM + PCIe | 96GB unified |
| CPU cores | **16 Zen 5** | Host CPU (separate) | 12 cores (no AVX-512) |
| AVX-512 | **Yes (double-pumped)** | N/A (host CPU varies) | **No** |
| Zero-copy GPU→CPU | **Yes** | **No** (PCIe 4.0 ~25 GB/s) | Yes (but weaker CPU) |
| GPU TFLOPS FP16 | 59.4 | 330 | 27 |
| Combined CPU+GPU | **Simultaneous** | Serial (PCIe bottleneck) | Simultaneous (weaker) |

**The key differentiator:** Strix Halo's CPU is strong enough (AVX-512) to run a meaningful model, AND shares memory with the GPU at full bandwidth. No other consumer hardware has this combination.

### Kernel Reuse

**GPU:** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), cross_entropy (1.8×), hybrid_flash_sdpa_attention (SWA training)

**CPU:** PyTorch MKL backend (automatic AVX-512 utilization for matmuls and element-wise)

### Throughput Estimate

| Component | Time per step (1024 tokens) |
|-----------|---------------------------|
| GPU: 12 SWA layers | ~24ms |
| GPU: 3 cross-attn layers | ~6ms |
| GPU: FFN (12 layers) | ~15ms |
| GPU: Embedding + LM head | ~3ms |
| **GPU total** | **~48ms** |
| CPU: 6 RWKV-7 layers (4 tokens) | ~2ms |
| CPU: world_state update | ~0.5ms |
| **CPU total** | **~2.5ms** (10× faster than GPU) |

| Mode | Config | Throughput |
|------|--------|------------|
| GPU only (no world model) | compile + AK | ~30K tok/s |
| + sync CPU world model | compile + AK | ~25K tok/s |
| + async CPU world model | compile + AK | **~28K tok/s** |

**CPU is never the bottleneck.** The 2.5ms CPU cycle completes while GPU is still processing the next batch. Async overhead is minimal (~2ms synchronization per batch of 4 tokens).

**Estimated optimized throughput:** ~25-30K tok/s (compile + autokernel + async CPU)
**Tokens in 45 min:** ~68-81M (4.2-5.1 BabyLM epochs)
**Ranking:** #3 of 31 architectures

**Risk-adjusted ranking:** #20 of 31 (high risk of implementation failure offsets high throughput potential). **This is the moonshot.** If it works, it demonstrates a new paradigm for APU-native AI architectures.
