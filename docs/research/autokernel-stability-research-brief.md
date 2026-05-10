# Autokernel Stability on AMD Strix Halo — External Research Brief

**Project:** `autokernel-halo-strix` — autonomous GPU kernel optimization + hybrid LM training on AMD Strix Halo APU.
**Repo:** https://github.com/joelhenwang/autokernel-halo-strix
**Date:** 2026-05-11
**Target audience:** External research agent with no prior project context. Only the GitHub URL.
**Ask:** Deep research, independent hypothesis ranking, proposed experiments, second opinion on our locked decisions.

---

## Table of Contents

1. Executive summary
2. Hardware context
3. Software stack
4. Model architecture
5. Training recipe
6. The autokernel system
7. The silent-freeze incident (Phase V, 2026-05-10)
8. Phase B remediation (5 replacement fixes + z-loss extension)
9. Post-fix divergence (Phase C + Phase G)
10. Hypotheses
11. What we tried (full ablation table)
12. What we considered but skipped
13. Current production state
14. Questions for the research agent
- Appendix A: Math derivations
- Appendix B: Code snippets
- Appendix C: Commit timeline
- Appendix D: File index

---

## 1. Executive summary

We are training custom hybrid-architecture language models (OdinFlat 122M, OdinHalo 58M) on an AMD Strix Halo APU (gfx1151, RDNA 3.5, ROCm 7.12). The platform has two major constraints that shape every kernel choice: **no matrix cores (MFMA) and unified memory with ~240 GB/s bandwidth**. Our in-repo "autokernel" package pattern-matches PyTorch modules and swaps them with HIP C++ kernels.

**The problem we want outside help on:**

Our autokernel system had a class of bug we call the **silent-freeze bug**: HIP kernels were called via raw pybind11 in module forward passes. The returned tensors had `grad_fn=None`, severing autograd's gradient propagation. 23% of OdinFlat's parameters (44M weights) were silently frozen at init during training. Forward passes produced sensible losses; loss even descended, because downstream parameters still trained. The bug was invisible until long-horizon training (~2000 steps) showed a +0.65 loss regression.

**Phase B of this remediation wired all five affected Replacement classes through autograd-safe paths** (`torch.library.custom_op` with `register_autograd`). Static and runtime audits confirm gradient flow is now correct — 5 of 5 production Odin models show zero newly-frozen parameters in a 21-probe diagnostic sweep.

**But: when we enable `--optimize-kernels` (post-fix) on real training runs, they diverge.**

- OdinFlat (`lr_2d=5e-3`): diverges at step 200-250 across three different configurations (single-node batch=128 with/without fused-zloss, DDP batch=256).
- OdinHalo (`lr_2d=2e-3`): diverges at step 750, after tracking 0.76 BETTER loss than the pre-fix baseline through step 700.

The divergence is NOT a new autograd bug (tests + preflight confirm grad flow is clean). It appears to be a **training-dynamics issue** created by correctly unfreezing ~44M parameters mid-recipe that were tuned under the partially-frozen regime. The divergence step scales with LR (`1/LR` roughly), consistent with accumulated out-of-equilibrium gradient statistics from the catch-up phase.

**What we shipped anyway:**
Phase B fixes are kept as future-proofing. Production Sprint 3A (OdinFlat) and Sprint 3B (OdinHalo) launch **without** `--optimize-kernels`, using the previously-validated vanilla recipe (Sprint 3A-confirm reached loss 3.15 at step 2000). This costs ~29 hours of wall time on Sprint 3B vs. the theoretical `--optimize-kernels` path, but guarantees correctness.

**What we want:**
1. Independent diagnosis of the Phase C/G divergence. Is our "unfrozen-param gradient statistics" hypothesis correct?
2. Literature references for fine-tuning that unfreezes parameters gracefully.
3. Experiments we haven't tried (we enumerate what we skipped in §12).
4. Second-opinion on our hypothesis ranking.
5. Potentially: bug-finding in our Phase B fixes that we missed.
6. Kernel-design ideas that exploit gfx1151's no-MFMA constraint beyond what we've tried.

---

## 2. Hardware context

### 2.1 Strix Halo (gfx1151, RDNA 3.5) key specs

Source: `knowledge/hardware/amd_rdna35_strix_halo.md`. Verified on hardware (Ryzen AI MAX+ 395).

| Parameter | Value |
|---|---|
| Architecture | RDNA 3.5 (APU, integrated GPU) |
| ISA | gfx1151 |
| CPU | 16 Zen 5 cores / 32 threads, 3.0–5.1 GHz |
| GPU | Radeon 8060S, 40 CUs, 20 WGPs |
| Memory | 128 GB soldered LPDDR5X, 256-bit bus (unified CPU+GPU+NPU) |
| GPU-visible memory | ~116 GB (PyTorch sees rest reserved for CPU/OS) |
| Memory bandwidth | ~240 GB/s (LPDDR5X-7500) |
| FP16 peak (compute) | ~59.4 TFLOPS |
| FP32 peak | ~29.7 TFLOPS |
| LDS per CU | 64 KB |
| L2 cache | ~6 MB |
| Wavefront size | **32 threads (wave32, preferred)** |
| Matrix cores (MFMA) | **NONE** — scalar FMA only |
| Max VGPRs per SIMD | 1536 |
| TDP | 45-120 W configurable |

### 2.2 Critical consequences for kernel design

**No MFMA.** This is the single most consequential constraint. Strix Halo has no tensor / matrix cores; all compute goes through scalar FMAs at wave32 granularity. rocBLAS is heavily tuned for this and typically beats any hand-written HIP matmul kernel for the standard transformer shapes. **Corollary: putting matmuls inside custom HIP kernels loses to calling rocBLAS directly.**

**Unified memory.** CPU, GPU, and NPU share the same 128 GB LPDDR5X pool. No discrete HBM, no PCIe copies, no HtoD/DtoH penalty. But bandwidth is **~240 GB/s** — an order of magnitude below discrete GPUs (MI300X has 5300 GB/s). Every kernel is effectively memory-bandwidth bound at this scale.

**Low L2 (6 MB).** Datasets > ~4 MB don't fit; repeated reads from larger tensors go to LPDDR5X. This limits the "free re-read" strategies that work well on MI300X (256 MB L2).

**Wave32, not wave64.** Our kernels can't naively port from MI300X CDNA code; tile sizes and occupancy math differ.

**fp16 + GradScaler only.** bf16 is 24% slower on this SKU and causes torch.compile crashes. This is empirically measured and codified in `CONSTRAINTS.md`:

```
- [ ] **fp16 + GradScaler** only — NOT bf16 (24% slower, compile crashes)
```

### 2.3 Two-machine topology

Two physically identical Strix Halo machines connected via **Thunderbolt 4**. We use DDP across them with the gloo backend. TB4 bandwidth (~40 Gbps) would be a bottleneck for high-param-count models, but at OdinFlat's 122M and OdinHalo's 58M it's manageable. Empirically gloo matches NCCL performance on this topology because the unified-memory model eliminates the usual CPU↔GPU staging overhead.

| Machine | Host | NIC | Role |
|---|---|---|---|
| A (master) | joelwang-ai-2 | thunderbolt0 @ 10.77.0.1 | rank 0, launches Machine B via SSH |
| B (worker) | joelwang-ai-1 | thunderbolt0 @ 10.77.0.2 | rank 1 |

Phase C v3 (DDP batch=16×8×2=256) runs at ~31K tok/s aggregate post-B-fix during its ~200 steps before diverging. B4 (OdinHalo pre-fix, 2000 steps) ran at ~32.5K tok/s steady.

### 2.4 Prior finding: HIP kernels for pointwise ops DO win on this hardware

See `docs/perf/odinflat-throughput-final.md` and our Phase I ship-gate bench `docs/perf/triton-swiglu-ship-gate-bench.md`. At the OdinFlat SwiGLU production shape (B=16, T=512, H=2048, fp16):

| Implementation | fwd+bwd wall time | Speedup vs eager |
|---|---:|---:|
| Eager PyTorch (`F.silu(g) * u`) | 2730 μs | 1.00× (baseline) |
| Autograd-safe HIP (`torch.ops.autokernel.silu_gate_mul`) | 1881 μs | **1.45×** |
| Triton fused SwiGLU (`kernels.triton.fused_swiglu`) | 1907 μs | 1.43× (0.99× vs HIP) |

So HIP pointwise kernels ARE materially faster than Inductor-fused eager PyTorch at this shape. We lose this ~5-7% total step wall speedup (from the SwiGLU slice alone) when we ship without `--optimize-kernels`.

---

## 3. Software stack

### 3.1 Versions

- **ROCm 7.12** (toolchain, HIP runtime, rocBLAS, rocFFT, MIOpen).
- **PyTorch ≥ 2.4** built against ROCm 7.12. We use the AMD wheel index:
  `https://download.pytorch.org/whl/rocm7.12`
- **Python ≥ 3.10**
- **Triton for ROCm** bundled with the PyTorch ROCm build.
- HIP C++ extensions via **pybind11**.
- `torch.library.custom_op` + `register_autograd` (modern autograd integration).

From `pyproject.toml`:

```toml
requires-python = ">=3.10"
dependencies = [
    "torch>=2.4.0",
    ...
]
[tool.uv.sources]
torch = [{ index = "pytorch-rocm712" }]
[[tool.uv.index]]
name = "pytorch-rocm712"
url = "https://download.pytorch.org/whl/rocm7.12"
```

### 3.2 Compile modes (relevant to this investigation)

`AGENTS.md` codifies several empirical findings:

- **`TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`** is the production default. Plain `max-autotune` crashes when `accum_steps > 1` (CUDA graph buffer overwrite). The `-no-cudagraphs` variant keeps autotuning but skips graph capture.
- **`reduce-overhead` mode is incompatible with looped models** (HIP CUDA-graph backend produces "empty graph" warnings and runs eagerly). Trainer auto-redirects to `default` with a note.
- **Per-zone compile for looped models.** Trainer uses `model.compile_zones(...)` to compile each layer independently rather than the whole model, sidestepping graph breaks from HIP extensions and Python-list mutation.

### 3.3 Inductor behavior on gfx1151

Inductor's Triton codegen fuses most elementwise chains aggressively. From the prior Phase 2 investigation (`docs/perf/phase2-summary-2026-05-05.md`):

- 92 unique triton kernels covering nearly every elementwise chain under `compile_zones`
- `mul` appears in 81, `add` in 33
- rocBLAS still wins for matmul (no triton_mm variant beats it)

This sets the baseline that any kernel-optimization effort has to beat. Our Phase I bench shows HIP beats Inductor on SwiGLU elementwise (1.45×) but Triton does not beat HIP (0.99×).

---

## 4. Model architecture (detailed)

This section is deliberately extensive because kernel optimization interacts with every layer structure. External agent should be able to reason about blast radius and kernel-applicability from this alone.

### 4.1 OdinFlat — 14-layer flat hybrid (primary Sprint 3A target)

**File:** `models/odin_flat.py:93-378`
**Parameter count:** 121.7M (all unique, non-looped)
**Production block size:** 512 tokens
**Tokenizer:** custom 32K BPE at `tokenizers/vidar-32k/tokenizer.json` (EOS=0)

Constructor defaults (from `models/odin_flat.py:113-135`):

```python
class OdinFlatBase(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 768,
        embed_rank: int = 256,
        n_layers: int = 14,
        gqa_positions: Tuple[int, ...] = (6, 13),
        n_heads: int = 12,
        n_kv_heads: int = 4,
        ffn_inner: int = 2816,
        d_conv: int = 512,
        conv_kernel: int = 3,
        max_seq_len: int = 2048,
        use_xsa: bool = True,
        use_softcap: bool = True,
        ...
    ):
```

**Layer composition:** 14 layers total, of which **only 2 are attention** (indices 6 and 13) and the remaining 12 are HyPE conv blocks. This is atypical — most transformer-family models are attention-dominant. Our models are conv-dominant with sparse attention.

- `HyPEShortConvBlock` (layers 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12) — 12 layers
- `NoPEGQABlock` (layers 6, 13) — 2 layers
- Each block has a `SwiGLU` FFN (conv block has it after conv; GQA block has it after attention)

**Component details:**

#### 4.1.1 `FactorizedEmbedding` and `FactorizedLMHead`

File: `models/components/embeddings.py`

Instead of `nn.Embedding(V, D)`, we factorize into:
- `self.embed: nn.Embedding(V, rank)` — lookup into rank-dim space
- `self.proj_up: nn.Linear(rank, D, bias=False)` — project to d_model

Parameter count: `V × rank + rank × D` instead of `V × D`.

For V=32768, D=768, rank=256:
- Factorized: 32768·256 + 256·768 = 8.39M + 0.20M = 8.59M
- Dense: 32768·768 = 25.17M

Saves ~16.58M params (≈13% of model).

LM head is tied via `FactorizedLMHead(d_model, rank, self.tok_embeddings.embed)` — reuses the same embed matrix.

Optional `use_chunked_ce` flag: when training, `forward()` returns the rank-dim `h_low` tensor instead of full logits, and a `ChunkedLinearCrossEntropyLoss` does the final `h_low @ embed.T` chunk-by-chunk to avoid materializing the full `[B·T, V]` logits tensor.

#### 4.1.2 `NoPECodaAttention` (content-only GQA)

File: `models/components/attention.py:194-336`

```python
class NoPECodaAttention(Attention):
    """Content-only CodaAttention — no RoPE on Q/K, QK-Norm mandatory.

    Designed for HyPE (ODIN-HALO): attention is purely content-based, enabling
    length generalization without positional encoding bias. QK-Norm prevents
    exploding logits without the positional anchor that RoPE provides.
    ...
    """

    # Phase 0 (2026-05-08): opt out of autokernel's FusedQKV pattern.
    _skip_autokernel = True
```

Key departures from standard GQA:
- **No RoPE applied.** Purely content-based attention. HyPE positional information comes from the conv blocks, not the attention.
- **QK-Norm mandatory.** Without a positional anchor, logits can blow up; QK-norm (L2-normalize Q and K along head_dim, then rescale by learned `q_scale`/`k_scale` of shape `[n_heads, 1, 1]`) prevents this.
- **XSA (Exclusive Self-Attention).** After computing attention, removes the self-value projection: `y = y - (y·v_seq / |v_seq|²) * v_seq`. Forces attention to capture content orthogonal to the token's own value.
- **Sprint 1 additions:** `v_res_scale` (scalar nn.Parameter, init 0.0) and `head_gate` (per-head nn.Parameter of shape `[n_heads]`, init 1.0 → sigmoid 0.731 effective). Both optional via `v_prev=...` and `head_gate_active=True` in the forward call.
- **MoDA depth-KV support:** accepts optional `depth_kvs` list for cross-iteration attention (used by looped models).
- **`_skip_autokernel = True`** class attribute: opts out of autokernel's FusedQKV pattern because our forward signature (`doc_mask`, `v_prev`, `head_gate_active`, `return_v` kwargs) is incompatible with the FusedQKVAttentionReplacement's `forward(x, freqs_cis, **kwargs)` assumption.

Shape after Track 2.a QKV fusion (2026-05-10, commit `63de5be`):
- `self.wqkv: nn.Linear(dim, q_dim + 2*kv_dim, bias=False)` — for dim=768, n_heads=12, n_kv_heads=4, head_dim=64: `wqkv.weight.shape = (1280, 768)`.

Forward signature:

```python
def forward(self, x: torch.Tensor,
            depth_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            doc_mask: Optional[torch.Tensor] = None,
            v_prev: Optional[torch.Tensor] = None,
            head_gate_active: bool = False,
            return_v: bool = False,
            ):
    B, T, _ = x.shape
    q, k, v = self._split_qkv(x, B, T)  # wqkv projection then chunk
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    v_raw = v
    if v_prev is not None:
        v = v + self.v_res_scale * v_prev

    q = F.normalize(q, dim=-1) * self.q_scale
    k = F.normalize(k, dim=-1) * self.k_scale

    if self.n_rep > 1:
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

    # ... depth-KV prepend, intra-doc mask, attention_core, head-gating, XSA, wo
```

#### 4.1.3 `HyPEShortConvBlock` (causal conv + positional)

File: `models/components/conv_blocks.py:HyPEShortConvBlock`

The "Hy"brid "P"ositional "E"ncoding block. Uses `causal_conv1d_fn` (DaoAILab's C++ extension) for depth-wise causal conv plus a RoPE-like gated step. Fused via HIP kernel `fused_rope_gate_mul` under `--optimize-kernels` when it's enabled. Otherwise uses native PyTorch + `F.conv1d`.

The block has an `_compile_friendly = True` toggle (`models/components/conv_blocks.py`) which swaps HIP kernels for native equivalents, achieving 0 graph breaks under `torch.compile(fullgraph=True)`. Not faster — Inductor and HIP are within ~10% at this shape — but needed for full-graph compilation when requested.

#### 4.1.4 `SwiGLU` FFN (the central kernel target)

From `models/_components.py` (SwiGLU class):

```python
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        # Fused w_gate_up: input dim → 2·hidden (gate + up concatenated)
        self.w_gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.w_down   = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)
```

For OdinFlat: `dim=768`, `hidden=2816` (SwiGLU inner). Per-layer FFN params:
- `w_gate_up`: `dim × 2·hidden = 768 × 5632 = 4.32M`
- `w_down`: `hidden × dim = 2816 × 768 = 2.16M`
- Total per FFN: **6.49M params**
- Across 14 layers: **90.9M params** (~75% of OdinFlat's total)

**This is the kernel-optimization gold target.** SwiGLU FFN dominates parameter count and arithmetic intensity. It's also where the silent-freeze bug hit — `_FusedSwiGLUReplacement` was the primary broken Replacement class.

### 4.2 OdinHalo — 6 shared layers × 3 iterations (primary Sprint 3B target)

**File:** `models/odin_halo.py:83-391`
**Parameter count:** 57.6M unique (156M effective via 3-iteration loop)
**Production block size:** 256 tokens
**Architecture:** Parcae-looped hybrid.

Key differences from OdinFlat:
- **6 shared layers**, each invoked 3 times (total 18 virtual layers).
- `shared_layers` (nn.ModuleList) vs OdinFlat's `layers`.
- Each layer has `iter_norm` reset between iterations (re-normalizes the hidden state to prevent drift).
- **MoDA depth-KV cache:** previous iteration's K,V are prepended to the current iteration's K,V, letting each token attend to itself at the prior iteration.
- **`iter_scales`** per-layer learned scalar (shape `[n_shared_layers, 3]`) that weights the residual contribution per iteration. Clamped to `[-4, 4]` at forward time for fp16 stability (2026-05-07+).
- Compile strategy: `compile_zones()` compiles each shared layer independently. Dynamo does NOT unroll the `for i in range(3)` Python loop at graph-construction time; it compiles one instance and re-dispatches.

Layer composition: 6 shared layers, of which the 4th (`shared_layers[3]`) is a `NoPEMoDAGQABlock` (the only attention layer), rest are `HyPEShortConvBlock`.

B4 probe (2026-05-11) showed that with pre-fix `--optimize-kernels`, OdinHalo trained to loss 2.51 at step 2000 despite having 14 of its 61 named parameters frozen (`shared_layers.{0..5}.ffn.w_gate_up.weight` + `ffn_norm.weight`, plus layer-3-specific `attn.v_res_scale` and `attn.head_gate`). The iter_norm resets apparently damped the silent-freeze damage.

### 4.3 Non-trivial Sprint-1 features that affect gradient flow

Added in Sprint 1 / 1.5 (`docs/perf/sprint1*.md`). All opt-in via training flags but ON in production Sprint 3A/3B recipes.

- `--intra-doc-mask`: additional causal mask that prevents attention across document boundaries within a packed batch. Uses `doc_ids` tensor threaded through the dataloader.
- `--value-residuals`: `v = v + v_res_scale * v_prev` where `v_prev` is the pre-residual V from a prior GQA layer. Carries value information through the stack.
- `--head-gating`: per-head scalar gate `sigmoid(head_gate)` multiplied into attention output.
- `--imu1-groups`: optimizer 2-way split (2D params → NorMuon, 1D params → AdamW). See §5.

These all produce additional parameters that must be handled in the autograd-safety audit. See §7.

### 4.4 Factorized CE loss path (ChunkedLinearCE)

`kernels/hip/chunked_linear_cross_entropy.py` (`ChunkedLinearCrossEntropyLoss`).

When `use_chunked_ce=True`, the model's `forward()` returns the low-rank hidden state `h_low: [B, T, rank]` instead of full logits. The trainer then calls:

```python
loss = chunked_ce_fn(h_low, embed_table.weight, targets)
```

Chunked CE processes `[chunk_size, V]` slices rather than the full `[B·T, V]` logits tensor. Memory peak savings at V=32768, B=16, T=512: ~1-3 GB.

This has a separate autograd path from `F.cross_entropy` which matters because our Phase B.5 extended `_CrossEntropyHIP` to accept `z_loss_weight` in its forward — but `ChunkedLinearCrossEntropyLoss` calls into `_CrossEntropyHIP` per chunk and would need analogous z-loss plumbing. Not done in Phase B.5 (opt-in `--use-fused-zloss` only covers the non-chunked path).

### 4.5 Parameter tally (OdinFlat, for reference)

| Group | Param count | Share |
|---|---:|---:|
| Embedding + LM head (factorized, tied) | ~8.6M | 7.1% |
| Attention (2 layers × Q/K/V/wo + QK-Norm scales + head_gate + v_res_scale) | ~3.3M | 2.7% |
| Conv layers (12 × conv weights + RoPE) | ~18.8M | 15.5% |
| SwiGLU FFN (14 × w_gate_up + w_down) | ~90.9M | 74.7% |
| Norms and misc | ~0.1M | 0.1% |
| **Total** | **~121.7M** | |

**SwiGLU w_gate_up alone** (the silent-freeze victim): 14 × 4.32M = 60.5M params. When it was frozen, ~50% of the model's trainable parameter mass was stuck at init.

_(Continues in subsequent sections: training recipe, autokernel system, the silent-freeze incident, Phase B remediation, post-fix divergence, hypotheses, ablations, what we skipped, production state, questions, appendices.)_

---

## 5. Training recipe

### 5.1 Optimizer stack: NorMuon + AdamW via IMU-1 grouping

File: `halo_training/normuon.py`, `halo_training/optimizer.py`

We use **IMU-1-style parameter grouping** (from the IMU-1 recipe paper referenced in our Sprint 1 specs):

- **2D parameters** (linear weight matrices, embedding weight, `w_gate_up`, `w_down`, `wqkv`, `wo`, `q_scale`, `k_scale`) → **NorMuon**.
- **1D parameters** (norms, biases, scalars like `head_gate`, `v_res_scale`) → **fused AdamW** (`torch.optim.AdamW(..., fused=True)`).

With μP active (Sprint 1.5 C3, commit `ef32915`), the 2D group is further split into three by role:

| μP group | Example params | LR (for OdinFlat at lr_2d=5e-3) |
|---|---|---:|
| `embedding` | `tok_embeddings.embed` | 5.0e-3 |
| `hidden` | all middle-layer 2D weights | 1.67e-3 |
| `readout` | `lm_head.proj_down` | 5.56e-4 |
| `1D` (AdamW) | norms, scalars | 8.0e-4 |

The 1D LR is fixed at `lr_1d=8e-4` by default.

### 5.2 NorMuon: Muon + neuron-wise norm + Cautious WD

NorMuon combines three ideas:
1. **Muon update** (Keller Jordan): Newton-Schulz iteration to orthogonalize the raw update matrix.
2. **Neuron-wise L2 normalization** of the orthogonalized update.
3. **Cautious weight decay**: flip WD sign if it would reduce gradient magnitude, else keep.

File: `halo_training/normuon.py`. Key steps per 2D param on each opt step:

```python
def _normuon_step(p, g, state, ...):
    # 1. Add momentum (β1)
    state["momentum"].mul_(β1).add_(g)
    m = state["momentum"]

    # 2. Newton-Schulz orthogonalization (5 iterations)
    u = newton_schulz(m, steps=5, dtype=torch.float16)   # fp16 for speed

    # 3. Neuron-wise normalization: scale each row by 1/||row||
    if p.ndim == 2 and p.shape[0] >= neuron_norm_min_dim:
        row_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        u = u / row_norm

    # 4. Cautious weight decay: if g·u < 0, skip WD this step
    if cautious_wd and (g * u).sum().item() < 0:
        wd_effective = 0
    else:
        wd_effective = wd

    # 5. Update
    p.data.mul_(1 - lr * wd_effective).add_(u, alpha=-lr)
```

**Newton-Schulz polynomial** (5 iterations, see `newton_schulz_5iter`):

$$
U_{k+1} = a\, U_k + b\, U_k (U_k^T U_k) + c\, U_k (U_k^T U_k)^2
$$

with `(a, b, c) = (3.4445, -4.7750, 2.0315)` tuned for 5-iter convergence. This runs **three fp16 matmuls per iteration × 5 iterations = 15 matmuls per 2D param per opt step**. On OdinFlat with ~90M 2D params batched per optimizer step, this is one of the dominant compute consumers: Track 1.3 profile (`docs/perf/odinflat-step-profile.md`) showed **NorMuon.step at 12.5% of step wall time**.

**Sprint 1.1** (2026-05-07, commit in earlier session) made NS use fp16 matmul (`HHS_BH_` rocBLAS kernel) and flipped `--ns-dtype fp16` to default. +17.5% tok/s on Run 2b.

### 5.3 SPECTRA post-clip

File: `halo_training/spectra.py`

After NorMuon computes the update and before it's applied, SPECTRA clips the update's spectral norm via power iteration:

```python
# Power iteration to estimate ||U||_2 (largest singular value)
v = torch.randn(U.shape[1])
for _ in range(ns_iter):        # 5 iterations default
    v = (U.T @ (U @ v))
    v = v / v.norm()
sigma_max = (U @ v).norm()

# Clip if over threshold
if sigma_max > clip_norm * 1.02:    # 2% safety margin
    U.mul_(clip_norm / sigma_max)
```

Sprint 1.5 Phase C found combined μP + SPECTRA wins by -0.38 loss at step 400 vs baseline, at 5% throughput cost. Production config uses `--spectra-post --spectra-clip-norm 1.0`.

### 5.4 LR schedule: WSD (warmup-stable-decay)

File: `halo_training/schedules.py`

```
     lr
      |
peak  |         _____________________
      |        /                     \
      |       /                       \
      |      /                         \
      |     /                           \___
      |    /                                \
      |   /                                  \____
 min  |__/___________________________________________
      0  warmup_steps           decay_start       total_steps
```

Production Sprint 3A: `warmup_steps=300`, `lr_2d=5e-3`, `min_lr_ratio=0.1`. Decay is linear from `decay_start` to final `min_lr = peak_lr * min_lr_ratio`.

For 1-epoch dolma-10B at batch=256, total_steps ≈ 52,000. Warmup is ~0.6% of training.

### 5.5 fp16 stability stack (added 2026-05-07, post-NaN incident)

After a 2-epoch OdinHalo run NaN'd mid-training (2026-05-07), we added a **prevention + detection + response** stack. All active in Sprint 3A/3B recipes.

**Prevention:**
- `--z-loss 1e-4 --z-loss-fraction 1.0` — penalizes `mean_i(lse(logits_i)^2)` (z-loss, aka log-partition-function regularization). Keeps logits from drifting in magnitude.
- `--attn-softcap 50.0` — pre-softmax `scores = 50 * tanh(scores / 50)` bounds attention scores in [-50, 50] to prevent fp16 overflow (max 65504).
- `iter_scales.clamp(-4, 4)` at forward time in looped models.
- GradScaler `growth_interval=500` (down from default 2000).

**Detection:**
- `--activation-monitor` emits per-layer `maxabs` and `fp16_headroom` to `$CKPT_DIR/activation_stats.jsonl` every N steps.
- `StabilityGuard` with 4 failure modes: NaN loss, NaN grad, inf param, **scale collapse** (`scaler.get_scale() < 1.0`).

**Response:**
- `StabilityGuard.rollback()`: load last good checkpoint, halve LR, halve `scaler.growth_interval`, emit forensics dump to `$CKPT_DIR/nan_dump_step_N.pt`.

Knowledge doc: `knowledge/training/fp16_stability_gfx1151.md`.

### 5.6 Data pipeline

- **Dataset:** `datasets/dolma-10b-odin32k.bin` — 6.9B tokens of Dolma-10B, pre-tokenized with our 32K BPE tokenizer, stored as `np.int32` array on disk (13.8 GB).
- **Loader:** `np.memmap` zero-copy, per-batch `.astype(np.int64)` in `__getitem__`.
- **Block structure:** sequences of `block_size=512` (OdinFlat) or `256` (OdinHalo).
- **Document boundaries:** optional `doc_ids` tensor (same shape as `input_ids`) for intra-doc attention masking.
- **Parallelism:** `num_workers=12` per rank (sweet spot from 2026-05-06 sweep).

### 5.7 Production training command (Sprint 3A-confirm, validated)

```bash
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=10.77.0.1 --master_port=29500 \
  scripts/train_ddp.py \
  --model models/odin_flat.py --class-name OdinFlat \
  --dataset datasets/dolma-10b-odin32k.bin --epochs 1 \
  --block-size 512 --batch-size 16 --accum-steps 8 \
  --compile --no-muon --lr 8e-4 --backend gloo \
  --warmup-steps 300 --num-workers 12 \
  --max-grad-norm 1.0 \
  --checkpoint-dir $CKPT --checkpoint-interval 500 --log-interval 50 \
  --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --auto-eval
```

Note the absence of `--optimize-kernels`. That flag existed in earlier recipes; after this remediation we are shipping without it.

---

## 6. The autokernel system

### 6.1 What it is

`autokernel/` is an in-repo package that walks an `nn.Module` tree after model construction and swaps specific sub-modules with HIP-kernel-backed equivalents. Entry point:

```python
import autokernel
model = autokernel.optimize(model, training=True)   # swaps modules in place
report = autokernel.report(model)                   # diagnostic
```

Training command flag `--optimize-kernels` triggers this.

### 6.2 Architecture

Two layers:

**Layer A — Pattern classes** (`autokernel/_patterns.py`): 8 `Pattern` subclasses that each:
- Implement `matches(name, module, model) -> bool` to detect candidate modules by attribute shape (e.g. has `w_gate_up` and `w_down` that are both Linear → SwiGLU pattern).
- Implement `replace(module) -> nn.Module` to construct a replacement.

Patterns defined:
- `RMSNormPattern`, `LayerNormPattern`, `SiluGateMulPattern`, `FusedSwiGLUPattern`, `FusedQKVPattern`, `RotaryEmbeddingPattern`, `FusedResidualRMSNormPattern`, `FusedGriffinBlockPattern`.

**Layer B — Replacement classes** (`autokernel/_patterns.py`): 7 `_*Replacement(nn.Module)` classes that each override `forward()` to call HIP kernels. These hold references to the original module's parameters and implement the forward.

### 6.3 Two dispatch paths for HIP kernels

**UNSAFE path (raw pybind11):**

```python
def forward(self, x):
    return self.kernel_fn(x, self.weight)   # self.kernel_fn is a pybind11-wrapped C++ op
```

Output tensor comes from a `torch::empty()` call inside the C++ extension. `requires_grad=False` by default, `grad_fn=None`. No autograd node is inserted. **Breaks gradient flow** — see §7.

**SAFE path (`torch.library.custom_op` + `register_autograd`):**

File: `kernels/hip/_torch_ops.py`. Example for silu_gate_mul:

```python
@torch.library.custom_op("autokernel::silu_gate_mul", mutates_args=())
def silu_gate_mul_op(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    from kernels.hip.silu_gate_mul import kernel_fn
    return kernel_fn(gate, up)

@silu_gate_mul_op.register_fake
def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return gate.new_empty(gate.shape)

def _silu_gate_mul_setup(ctx, inputs, output):
    gate, up = inputs
    ctx.save_for_backward(gate, up)

def _silu_gate_mul_backward(ctx, grad_output):
    gate, up = ctx.saved_tensors
    if _use_hip_backward() and gate.dtype == torch.float16 and gate.is_cuda:
        from kernels.hip.silu_gate_mul_backward import kernel_fn as silu_bwd_fn
        return silu_bwd_fn(gate, up, grad_output)
    # fp32 fallback: explicit chain rule
    g = grad_output.float()
    gate_f = gate.float()
    up_f = up.float()
    sig = torch.sigmoid(gate_f)
    silu_gate = gate_f * sig
    d_silu = sig * (1.0 + gate_f * (1.0 - sig))
    grad_gate = g * up_f * d_silu
    grad_up = g * silu_gate
    return grad_gate.to(gate.dtype), grad_up.to(up.dtype)

silu_gate_mul_op.register_autograd(
    _silu_gate_mul_backward, setup_context=_silu_gate_mul_setup
)
```

### 6.4 Current state of the 7 Replacement classes (post-Phase-B)

| # | Class | Kernel path | Verdict (static audit) |
|---|---|---|---|
| 1 | `_RMSNormReplacement` | `torch.ops.autokernel.rmsnorm` (Phase III fix, 2026-05-09) + `self.kernel_fn` fallback | **CONDITIONAL-SAFE** |
| 2 | `_LayerNormReplacement` | `F.layer_norm` (no HIP backward exists; Phase B.3 fix) | **SAFE** |
| 3 | `_SiluGateMulReplacement` | `torch.ops.autokernel.silu_gate_mul` (Phase B.2) | **SAFE** |
| 4 | `_FusedSwiGLUReplacement` | `torch.ops.autokernel.silu_gate_mul` (Phase B.1 — the primary fix) | **SAFE** |
| 5 | `_FusedQKVAttentionReplacement` | `torch.ops.autokernel.rotary_emb_fp32` (Phase B.4) + raw fallback | **CONDITIONAL-SAFE** |
| 6 | `_FusedResidualRMSNormBlockReplacement` | `torch.ops.autokernel.fused_res_rmsnorm` (Phase B.4b) | **SAFE** |
| 7 | `_FusedGriffinBlockReplacement` | Plain PyTorch ops (uses `self.griffin(...)` module) | **SAFE** |

### 6.5 Registered custom ops (`kernels/hip/_torch_ops.py`)

| # | Op name | Has autograd backward? |
|---|---|:---:|
| 1 | `autokernel::rmsnorm` | ✓ |
| 2 | `autokernel::rotary_emb_fp32` | ✓ |
| 3 | `autokernel::silu_gate_mul` | ✓ |
| 4 | `autokernel::fused_res_rmsnorm` | ✓ |
| 5 | `autokernel::selective_scan` | ✓ |
| 6 | `autokernel::griffin_scan` | ✓ |
| 7 | `autokernel::fused_ple_gate` | ✓ |
| 8 | `autokernel::fused_gated_conv` | ✓ |

All 8 have `register_autograd` — the Phase B fixes leveraged the already-registered ops that were simply NOT being invoked by the Replacement forwards.

### 6.6 Pattern matching on OdinFlat in production

From `scripts/diag_autokernel_patterns.py` output on OdinFlat (μP on):
- `rmsnorm`: 29 modules replaced (one per pre/post-norm across 14 layers + final_norm + per-FFN norms)
- `fused_silu_gate_mul`: 14 modules replaced (one per layer FFN)

Other patterns don't fire on OdinFlat because:
- `NoPECodaAttention._skip_autokernel = True` blocks `FusedQKVPattern`.
- `HyPEShortConvBlock` doesn't match the `FusedResidualRMSNormBlock` alias (different forward signature).
- No LayerNorm modules (OdinFlat uses RMSNorm).

So in OdinFlat production, the primary HIP pathway is the **14 `_FusedSwiGLUReplacement` modules**. This is why the Phase C divergence tracks to changes in that specific replacement.

---

## 7. The silent-freeze incident (Phase V, 2026-05-10)

### 7.1 Discovery

During Sprint 3A preparation we ran a 2000-step OdinFlat probe to compare pre-/post-RoPE-fix training quality. Under `--optimize-kernels` the model diverged slightly from baseline. Investigation (`docs/perf/odinflat-throughput-final.md`):

| Config | Steady tok/s | Loss @ step 200 | Loss @ step 2000 | Verdict |
|---|---:|---:|---:|---|
| P0 baseline (no `--optimize-kernels`) | 31,331 | 4.70 | **3.15** | reference |
| V1 (silu HIP, raw pybind) | 41,198 | 4.67 | **3.80** | +0.65 loss regression |
| V2 (silu HIP, autograd-registered) | 30,976 | 4.71 | ? | training correct, tok/s slightly BELOW baseline |

So the V1 "+31% throughput" was real on the wall clock, but accompanied by a significant quality regression visible only at long horizon.

### 7.2 Root cause: raw pybind11 returns tensor with `grad_fn=None`

The `_FusedSwiGLUReplacement.forward()` was calling `self.kernel_fn(gate, up)` directly:

```python
# autokernel/_patterns.py, pre-Phase-B (buggy)
def forward(self, x):
    gate, up = self.w_gate_up(x).chunk(2, dim=-1)
    if gate.dtype == torch.float16:
        activated = self.kernel_fn(gate.contiguous(), up.contiguous())  # RAW pybind call
    else:
        activated = F.silu(gate) * up
    return self.w_down(activated)
```

`self.kernel_fn` is a pybind11-wrapped C++ function that internally calls `torch::empty(...)` to allocate the output, then fills it via a HIP kernel launch. The returned tensor has:
- `requires_grad = False` (default for `torch::empty`)
- `grad_fn = None` (no autograd node created; torch dispatcher was not involved)

Downstream ops see `activated` as a value (not a node). `self.w_down(activated)` still computes a correct forward. On backward, `w_down.weight.grad = grad_out.T @ activated` (leaf-weight gradient) is computed via `activated`'s value alone — does not require a grad_fn to traverse.

**But** `grad_activated = grad_out @ w_down.weight.T` would normally propagate UP through `activated.grad_fn` back to `gate` and `up`. Since `activated.grad_fn = None`, autograd's backward traversal stops there. `gate.grad` and `up.grad` are never written. Therefore `w_gate_up.weight.grad` (one stage further up) is also never computed.

Worse: immediately upstream of `w_gate_up` is an `ffn_norm` (RMSNorm). The norm's forward IS autograd-tracked (it's a regular `nn.Module`), so autograd traces back to it — but with `grad_out_from_norm = 0` (because the chain died at `activated`). `ffn_norm.weight.grad = 0 · (whatever)` = literally `grad=0`, not `grad=None`.

### 7.3 Empirical blast radius (Track 3.A, 2026-05-10)

We added `--diag-frozen-params` to `scripts/train_ddp.py`. This flag writes a JSONL file at every optimizer step recording `{step, params: [{name, grad_norm, is_none, is_zero}, ...]}`. Post-run analysis classifies each parameter's trajectory across the run as `always_finite`, `always_zero`, `always_none`, or `occasionally_finite`.

Ran three 50-step single-node probes on OdinFlat with the Sprint 1.5 C3 recipe:

- **V0** baseline (no `--optimize-kernels`)
- **V1** `--optimize-kernels` (pre-Phase-B code, all patterns active)
- **V3** `--optimize-kernels --autokernel-exclude fused_silu_gate_mul` (only rmsnorm HIP with Phase III fix)

Results (`docs/perf/autokernel-frozen-blast-radius.md`):

| Config | always_finite | always_none | always_zero | Interpretation |
|---|---:|---:|---:|---|
| V0 | 119 | 1 (v_res_scale L0) | 0 | baseline clean |
| V1 | 91 | **15** | **14** | **~60M params frozen** |
| V3 | 119 | 1 | 0 | identical to V0 |

V1 frozen params (grouped):
- 14 × `layers.*.ffn.w_gate_up.weight` (all FFN layers) — `always_none`, ~60M weights total
- 14 × `layers.*.ffn_norm.weight` — `always_zero` (dead-chain upstream)
- 1 × attention v_res_scale (first attention layer, no v_prev input; expected allowed-unused)

So under pre-fix `--optimize-kernels`, **23% of OdinFlat's named parameters and ~50% of its trainable weight mass were frozen at init during training**.

V3 being IDENTICAL to V0 also proved that the Phase III RMSNorm autograd fix (2026-05-09) was complete — `_RMSNormReplacement` routes through `torch.ops.autokernel.rmsnorm` correctly.

### 7.4 Why training still looked "healthy" pre-fix

Three factors masked the bug until long-horizon training:
1. **Downstream params compensated.** `w_down` adapted to work with a frozen `w_gate_up`. Loss still descended.
2. **Throughput appeared HIGHER** because backward computation was skipped for the frozen params. +31% was ~50% "fake" (not computing 50M params' backward) + ~50% "real" (HIP fwd is faster than eager).
3. **Short-horizon losses look similar.** At step 200 the frozen vs correct models differ by only ~0.03 loss. By step 2000 the gap grows to +0.65.

### 7.5 Original "+31% throughput" was an artifact of the bug, not a feature

This is the meta-finding the whole investigation pivoted on:

> Any HIP kernel integration measured at "+X% throughput" where X > 0 either broke autograd (skipping backward work for params frozen by the bug) OR the reference baseline was sub-optimal (Inductor could have matched the HIP speed).

Phase V V2 (autograd-correct silu HIP) measured at 30.9K tok/s vs baseline 31.3K. The **real** HIP-vs-Inductor throughput advantage on OdinFlat's SwiGLU block is approximately zero after you compute the backward correctly.

This is one of the big findings we want the research agent to validate. Our Phase I bench (§9) complicates the picture because in-isolation HIP shows 1.45× speedup vs eager. The discrepancy is between "this single op is 45% faster in isolation" and "the model-level step is the same speed" — consistent with Amdahl's law (SwiGLU is ~10-15% of step wall at production shape).

---

## 8. Phase B remediation

### 8.1 Scope

Five Replacement classes had raw pybind paths in their forward. Phase B wired all five through registered custom ops (see §6.3 for the SAFE pattern).

| # | Fix | Old behavior | New behavior |
|---|---|---|---|
| B.1 | `_FusedSwiGLUReplacement` | `self.kernel_fn(gate, up)` | `torch.ops.autokernel.silu_gate_mul(gate, up)` (autograd-registered) |
| B.2 | `_SiluGateMulReplacement` (split-Linear variant) | `self.kernel_fn(gate, up)` | same as B.1 |
| B.3 | `_LayerNormReplacement` | `self.kernel_fn(x, w, b)` | `F.layer_norm(x, ...)` (no HIP LayerNorm backward exists; fallback to Inductor-fused eager) |
| B.4 | `_FusedQKVAttentionReplacement` | `self.rotary_fn(q, cos, sin)` (raw pybind for RoPE) | `torch.ops.autokernel.rotary_emb_fp32(q, cos, sin)` |
| B.4b | `_FusedResidualRMSNormBlockReplacement` | `self.kernel_fn_dual(attn_out, x, w)` | `torch.ops.autokernel.fused_res_rmsnorm(...)` |

All five commit `5ebe594`.

### 8.2 Phase B.5: z-loss gradient extension in `_CrossEntropyHIP`

The training loop's z-loss (`args.z_loss * logits.logsumexp(dim=-1).pow(2).mean()`) was a separate PyTorch-side pass that materialized a [B·T, V] fp32 tensor and did a second logsumexp over the vocab axis. Track 1.3 profile showed this as ~16.7% of step wall.

Phase B.5 (commit `f24d8dd`) extended `_CrossEntropyHIP` to optionally bake the z-loss into the HIP CE kernel's forward AND backward:

```python
@staticmethod
def forward(ctx, logits, targets, softcap, ignore_index,
            label_smoothing, mode, return_z, z_loss_weight=0.0):
    z_loss_weight = float(z_loss_weight)
    if z_loss_weight > 0.0 and mode != "tiny":
        mode = "tiny"   # need logits saved for backward softmax recompute
    # ... existing fused CE forward ...
    # Phase B.5: add z_loss to scalar loss
    if z_loss_weight > 0.0:
        z_loss = z_loss_weight * (lse_valid * lse_valid).sum() / n_valid_t
        loss = loss + z_loss
    # ... save ctx.z_loss_weight ...

@staticmethod
def backward(ctx, grad_loss, *rest):
    if ctx.has_tiny:
        logits, targets, row_max, row_sum = ctx.saved_tensors
        grad_logits = mod.cross_entropy_bwd_tiny_hip(...)
        # Phase B.5: add z-loss gradient contribution to grad_logits.
        # d/d(logits[i,j]) of (z_weight * lse[i]^2 / N) =
        #   (2 * z_weight / N) * lse[i] * softmax[i,j]
        if ctx.z_loss_weight > 0.0:
            with torch.no_grad():
                shifted = logits.float() - row_max.unsqueeze(-1)
                softmax = torch.exp(shifted) / row_sum.unsqueeze(-1)
                lse_vec = row_max + torch.log(row_sum)
                coef = (2.0 * ctx.z_loss_weight / ctx.n_valid_item) * grad_loss_scalar
                z_grad = coef * lse_vec.unsqueeze(-1) * softmax
                if ctx.ignore_index is not None:
                    valid = (targets != ctx.ignore_index).float().unsqueeze(-1)
                    z_grad = z_grad * valid
                grad_logits = grad_logits + z_grad.to(grad_logits.dtype)
        return grad_logits, None, None, None, None, None, None, None
```

Exposed to trainer via `--use-fused-zloss` (default OFF pending validation).

### 8.3 Tests added (Phase B.6)

File: `scripts/test_phase_b_autograd_safety.py` — 7 tests:

1. `test_fused_swiglu_replacement_grad_flows` — construct a minimal module with the replacement, run forward+backward, assert every leaf param has finite non-None grad.
2. `test_silu_gate_mul_replacement_grad_flows` — same for split variant.
3. `test_layernorm_replacement_grad_flows` — same for LayerNorm.
4. `test_ce_full_zloss_grad_parity_vs_eager` — gradient parity vs eager reference:
   ```python
   loss_e = F.cross_entropy(le, targets) + 1e-4 * le.logsumexp(dim=-1).pow(2).mean()
   loss_f = _ce_k.ce_full(logits_fused, targets, mode="tiny", z_loss_weight=1e-4)
   # assert relative error < 5% fp16 tolerance
   ```
5. `test_ce_full_zloss_value_matches_eager` — scalar loss includes z-loss contribution.
6. `test_ce_full_zloss_disabled_matches_no_z` — z_loss_weight=0 produces identical loss.
7. `test_no_unsafe_replacements` (CPU-only) — reads the static audit JSON and asserts no `UNSAFE` verdicts remain.

### 8.4 Empirical audit (Phase A.3, 2026-05-11)

Ran the `--diag-frozen-params` probe across 7 Odin-family models × 3 configs (V0 baseline, V1 `--optimize-kernels`, V3 exclude-silu). 21 probes total, 15 successful (6 mini-variant failures due to probe-config incompatibility, not autograd issues — see §11.3).

Post-Phase-B results (`docs/perf/autokernel-audit-2026-05-11-synthesis.md`):

| Model | V0 params | V1 newly frozen vs V0 | V3 newly frozen vs V0 |
|---|---:|---:|---:|
| odin_flat (122M) | 120 | **0** | **0** |
| odin_flat_30m (33M) | 72 | **0** | **0** |
| odin_flat_ablation (68M) | 70 | **0** | **0** |
| odin_halo (58M) | 61 | **0** | **0** |
| odin_halo_ablation | 61 | **0** | **0** |

Phase B fixes empirically validated at probe scale.

### 8.5 Phase D.A Triton harness + D.B Triton fused_swiglu

In parallel with Phase B we built infrastructure for future Triton kernels:

- `autokernel/triton_base.py` — `TritonAutogradFunction` base class (lower per-call overhead than `torch.library.custom_op`).
- `autokernel/triton_autotune.py` — shape + git-SHA keyed autotune cache.
- `scripts/kernel_parity_harness.py` — fwd+bwd parity across dtype/shape panel.
- `scripts/kernel_bench_harness.py` — isolated throughput bench.
- `knowledge/kernels/triton_author_guide.md` — authoring manual.
- `kernels/triton/fused_swiglu.py` — first Triton kernel (silu·up elementwise).

Triton kernel pair:

```python
@triton.jit
def _fused_swiglu_fwd_kernel(OUT_ptr, GATE_ptr, UP_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    gate = tl.load(GATE_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    silu_gate = gate * sig
    tl.store(OUT_ptr + offs, (silu_gate * up).to(tl.float16), mask=mask)

@triton.jit
def _fused_swiglu_bwd_kernel(GRAD_GATE_ptr, GRAD_UP_ptr, GATE_ptr, UP_ptr, GRAD_OUT_ptr, N, BLOCK: tl.constexpr):
    # d/dg = up * sigmoid(g) * (1 + g * (1 - sigmoid(g)))
    # d/du = g * sigmoid(g) = silu(g)
    ...
```

Phase I ship-gate bench (`docs/perf/triton-swiglu-ship-gate-bench.md`), at production shape B=16 T=512 H=2048 fp16:

| Implementation | fwd+bwd μs | speedup vs eager | speedup vs HIP |
|---|---:|---:|---:|
| Eager `F.silu(g)*u` | 2730 | 1.00× | — |
| Autograd-safe HIP | 1881 | **1.45×** | 1.00× |
| Triton fused_swiglu | 1907 | 1.43× | **0.99×** |

Ship gate (Triton ≥ 1.05× HIP): **FAIL**. Triton and HIP are numerically tied at this shape on gfx1151. Phase H Sprint 3A stability bisect skipped per locked plan.

### 8.6 Phase E runtime guardrails

To prevent regression of the silent-freeze class of bug:

**E.3 preflight in `train_ddp.py`:** after `autokernel.optimize(model, training=True)` and before the training loop begins, dispatch one forward+backward on a dummy batch. Iterate all `requires_grad=True` parameters. Raise `RuntimeError` if any has `grad=None` (excluding documented always-unused params like `v_res_scale` on the first layer and `head_gate` when `head_gate_active=False`).

Confirmed working on Phase C and Phase G launches:

```
  [autokernel] preflight OK: all parameters received gradients after dummy forward+backward
```

**E.2 CI smoke test** (`scripts/test_autokernel_autograd_safety.py`): one-step OdinFlatMini training with `--optimize-kernels`, asserts all grads flow.

**E.4 `CONSTRAINTS.md` rules:** codified as a must-check list:

```
- [ ] **HIP kernels in training paths MUST use `torch.library.custom_op` + `register_autograd`**
      OR `torch.autograd.Function`. Raw pybind calls (`self.kernel_fn(...)`) in
      `forward()` return tensors with `grad_fn=None` which silently severs gradient
      flow to upstream parameters. Caused OdinFlat's +0.65 loss regression at step 2000.
      Pre-merge gate: run `scripts/audit_autokernel_replacements.py`; any `UNSAFE`
      verdict blocks merge.
```

---

## 9. Post-fix divergence (the thing we want research help on)

### 9.1 Symptoms

After Phase B, three variants of a 2000-step OdinFlat verification probe (Phase C) and one 1000-step OdinHalo probe (Phase G) all **diverged**.

### 9.2 Phase C attempts (OdinFlat post-B, lr_2d=5e-3)

From `docs/perf/phase-c-final-analysis.md`:

| Probe | Config | Batch | Step 200 | Step 250 | Step 300 | Outcome |
|---|---|---:|---:|---:|---:|---|
| v1 | `--optimize-kernels --use-fused-zloss`, single-node | 128 | loss 5.38, grad 10.8 | loss 7.07, **grad 79353** | loss 11.29, scaler 1.6e-2 | DIVERGED |
| v2 | `--optimize-kernels`, single-node | 128 | loss 5.29, grad 0.95 | loss 5.84, **grad inf** | — | DIVERGED |
| v3 | `--optimize-kernels`, DDP | 256 | loss 5.59, **grad 272** | loss 9.40, scaler 1.6e-2 | — | DIVERGED |

v3 matched the EXACT config of Sprint 3A-confirm (which reached loss 3.15 at step 2000 cleanly), except with `--optimize-kernels` enabled. Same divergence.

### 9.3 Phase G (OdinHalo post-B, lr_2d=2e-3)

From `docs/perf/phase-g-findings.md`:

| Step | Phase G (post-fix) | B4 (pre-fix) | Delta |
|---:|---:|---:|---:|
| 100 | 7.34 | 7.16 | +0.19 |
| 200 | 5.58 | 5.82 | −0.24 |
| 500 | 4.74 | 4.87 | −0.13 |
| 700 | **3.68** | 4.44 | **−0.76** |
| 750 | 3.43 (grad 18.4) | 4.10 | −0.67 |
| 800 | **4.37 (grad inf)** | 3.88 | DIVERGED |
| 850 | 9.27 | 3.71 | DIVERGED |

Phase G was tracking **substantially better than pre-fix B4** (−0.76 loss at step 700) before diverging.

### 9.4 Divergence signature (uniform across all 4 probes)

1. A single batch produces a high but finite grad (10-300).
2. Next few microsteps produce `Non-finite grad norm, skipping step`.
3. Scaler halves itself on each skip; by step `+50` it's at 1e-2 or smaller.
4. LR-scaled update per step approaches 0; model can't learn.
5. Subsequent loss either oscillates (Phase C v1/v2) or balloons (Phase C v3, Phase G).

### 9.5 LR-scaling observation

| Model | LR | Divergence step |
|---|---:|---:|
| OdinFlat Phase C | 5e-3 | ~250 |
| OdinHalo Phase G | 2e-3 | ~750 |

Ratio: 2.5× LR difference, 3× divergence-step difference. Consistent with a mechanism where accumulated gradient-magnitude drift is approximately linear in LR × steps.

### 9.6 Ruled-out causes

- **Not `--use-fused-zloss`.** v2 removed it, still diverged.
- **Not batch size.** v3 matched production DDP batch=256, still diverged.
- **Not a new autograd bug.** Preflight passed on all 4 launches. 50-step audits showed zero newly-frozen params.
- **Not a model-specific artifact.** Both OdinFlat and OdinHalo show it with the same structure.

### 9.7 The problem the research agent should help with

We've concluded (tentatively) that Phase B fixes change the **numerical and statistical properties of gradients** in a way that interacts badly with our LR/optimizer/fp16 setup. Specifically:

- Pre-fix: ~44M params frozen → effective model size ~50% smaller → NorMuon updates on a smaller active subnetwork → stable at lr_2d=5e-3.
- Post-fix: all params training → gradient flow through every layer → update magnitudes larger → fp16 headroom eroded → overflow at ~step 250-750 depending on LR.

But we haven't proven this mechanism. The divergence may instead be:
- A subtle numerical drift in the HIP silu backward that accumulates
- An interaction between NorMuon's Newton-Schulz and a specific gradient distribution
- An fp16-specific scaling issue we're missing
- Something else entirely

See §10 for our hypothesis ranking.

---

## 10. Hypotheses

We rank three hypotheses by our current confidence, with explicit predictions each makes. The research agent is invited to challenge this ranking.

### H1 (primary, highest confidence): Unfrozen-parameter gradient statistics out of equilibrium

**Mechanism sketch:**

Pre-Phase-B, for 2000 training steps, ~44M `w_gate_up` weights sat at their initial values (He/Kaiming init, small-magnitude). The downstream `w_down` adapted to the frozen `silu(w_gate_up·x)` as if it were a fixed random feature extractor.

Post-Phase-B, the same `w_gate_up` weights suddenly start receiving gradients from the first step. Their initial gradient magnitudes are NOT at equilibrium — they're large because the downstream `w_down` was adapted to an "incorrect" frozen version and the new gradients reflect the discrepancy.

NorMuon's neuron-wise normalization then amplifies these large gradient components (row-wise scaling makes even small-magnitude rows contribute). Newton-Schulz iteration (5 steps of matmul, orthogonalization) compounds. LR is ramping up (warmup 300 steps) so the effective per-step update magnitude is increasing.

By step ~250 (at lr_2d=5e-3) or ~750 (at lr_2d=2e-3), the accumulated perturbation magnifies one of the layers' activations past fp16 range (max 65504). `attn-softcap=50` bounds attention logits but doesn't help FFN activations.

**Predictions:**

- If H1 is correct: lowering lr_2d should delay or eliminate the divergence (linear in LR).
- Extending warmup_steps should also help (gives unfrozen params more time to reach equilibrium at lower effective update magnitude).
- Initial values for `w_gate_up` should matter: starting close to the pre-fix frozen values (init) vs close to the trained values should change divergence onset.
- Layer-wise LR or per-parameter LR multipliers on `w_gate_up` specifically should fix it.

**Supporting evidence:**
- Divergence step scales with 1/LR empirically (2.5× LR ratio → 3× step ratio).
- Phase G was outperforming pre-fix training (−0.76 loss) before diverging — consistent with "correct training that's doing too much too fast."
- The signature is a single-batch grad spike, not a gradual drift — consistent with a catastrophic interaction rather than steady accumulation.

**Weakness:**
- We haven't actually tested the predicted interventions (lowered LR on OdinFlat, extended warmup). These are the primary skipped options in §12.

### H2 (secondary): HIP vs Inductor numerical drift

**Mechanism sketch:**

Autograd-safe HIP silu_gate_mul produces numerically different forward outputs than Inductor-fused `F.silu(gate) * up` (different FMA ordering, different fp16 rounding). Per-kernel tests (`test_phase_b_autograd_safety.py::test_ce_full_zloss_grad_parity_vs_eager`) show parity within 5% relative error at fp16 — but in production, the compounded difference across 14 layers × 2000 steps could drift the model into a region where the vanilla recipe is unstable.

**Predictions:**

- If H2 is correct: the divergence would be independent of LR (would happen at the same step regardless of lr_2d). We observe linear-in-LR scaling, which contradicts H2.
- Post-fix model weights at step ~200 should differ from what Sprint 3A-confirm produced at the same step if tested on identical data.
- The divergence would occur under all lr_2d values if the drift is the cause.

**Supporting evidence:**
- Phase I bench measures autograd-safe HIP at 1.45× eager — so HIP IS doing different work. Could produce slightly different gradients.

**Weakness:**
- Our LR-scaling observation strongly counters H2. If the bug were purely numerical drift, divergence step should be invariant to LR.
- Per-kernel numerical tests pass.

### H3 (tertiary): LR recipe tuned for pre-fix frozen-subset training

**Mechanism sketch:**

Variant of H1. lr_2d=5e-3 was tuned empirically against a pre-fix training regime where ~50% of the model weights were frozen. The effective LR applied to the ACTIVE 50% is approximately normal, but the full-model LR (what we think we're setting) is effectively ~2× lower than what's actually applied to the trainable subnetwork.

When we flip to full-active-model post-fix, the same nominal lr_2d=5e-3 is now distributed across 100% of the weights, so the effective per-neuron update magnitude is different. But in which direction? Arguably the SAME update per weight (the optimizer applies lr_2d per-parameter), so this model shouldn't predict a change in stability...

**Predictions:**
- If H3 is correct: the recipe has to be re-tuned, not just lowered. Specifically: lr_2d warmup schedule would need redesign.

**Weakness:**
- NorMuon's neuron-wise normalization theoretically produces updates that are invariant to the active-subset size. So H3's mechanism is weak.
- Similar predictions to H1 without a clearly distinguishing test.

### 10.4 Ranking and open questions

| # | Confidence | Predicts LR-scaling? | Simplest test |
|---|---|---:|---|
| H1 | 70% | Yes (matches obs) | Lower lr_2d on OdinFlat |
| H2 | 20% | No (contradicts obs) | Numerical comparison at fixed step |
| H3 | 10% | Unclear | LR search with different schedules |

We haven't executed the H1 test (OdinFlat with lr_2d=2e-3 + `--optimize-kernels`) because Sprint 3A currently ships without `--optimize-kernels` and the bisect was scoped out when Phase I showed Triton ties HIP (making the stability work throughput-neutral regardless).

### 10.5 Alternate hypotheses the agent should consider

We are NOT confident H1-H3 exhaust the space. Some alternatives we surfaced in discussion but didn't develop:

**H4: NorMuon's Newton-Schulz amplifies specific gradient spectra.** NS orthogonalization applied to a gradient matrix with unusual singular-value distribution (e.g. freshly-unfrozen `w_gate_up` with nearly-uniform gradients) could produce an update matrix with a much larger spectral norm than expected. SPECTRA post-clip catches >clip_norm×1.02, but if the polynomial approximation error grows with the specific distribution, the clipped value may still be too large.

**H5: fp16 loss-scaling interaction.** `GradScaler` scales the loss UP (default 2^16) so backward produces grads ×2^16. If a single HIP silu backward produces a slightly-too-large output, the scaled grad overflows fp16 (cap 65504). Unscaled grad would be safe; scaled grad is not. Our scaler starts at 2^16 and grows — when it hits a max (typically 2^16+) and encounters a batch with unusual activation distribution, overflow.

**H6: Specific layer localization.** Only 2 of 14 OdinFlat layers are attention. The 12 conv layers each have SwiGLU. If the divergence is localized to one specific layer's w_gate_up (e.g. layer 6 or 13 which sit after attention), per-layer LR or init fixes could work while global interventions don't. We haven't inspected per-layer activation stats during Phase C divergence.

**H7: Dolma-10B data distribution hot spot.** At step 250 (~32K tokens in at batch=128, block=512), we're hitting a specific region of dolma that may have gradient-pathological content. Pre-fix training was more robust because frozen params couldn't be destabilized by it. A reproducibility test (same seed, same shuffle) would test this.

**H8: Implicit autograd-graph size explosion.** `torch.library.custom_op` creates an autograd node per call. With 14 FFN layers × 8 microsteps/opt step = 112 custom_op dispatches per step. If each creates an autograd node that stays alive for the backward, the graph could grow much larger than pre-fix. Unclear how this would cause divergence but worth verifying graph structure.

---

## 11. What we tried (full ablation table)

| Phase | Config | Hardware | Result | Reference |
|---|---|---|---|---|
| **V0** (2026-05-10) | Baseline, no `--optimize-kernels`, `lr_2d=5e-3` | DDP | Stable. Loss 3.15 at step 2000. | `docs/perf/odinflat-throughput-final.md` |
| **V1** (2026-05-10) | `--optimize-kernels` pre-Phase-B (raw pybind silu) | DDP | +31% tok/s illusion, +0.65 loss regression at step 2000. | same |
| **V2** (2026-05-10) | `--optimize-kernels` pre-Phase-B with silu autograd wired manually | DDP | Correct training, tok/s 30.9K (slightly BELOW baseline). | same |
| **B4** (2026-05-11) | `--optimize-kernels` **pre-Phase-B** on OdinHalo, lr_2d=2e-3 | DDP | Stable. Loss 2.51 at step 2000. But 14 of 61 params silently frozen. | `docs/perf/odinhalo-b4-findings.md` |
| **Phase A.3** (2026-05-11) | 7 Odin models × V0/V1/V3, 50-step probes, post-Phase-B | single-node | V1/V3: 0 newly-frozen params across 5 successful models. Phase B empirically validated at probe scale. | `docs/perf/autokernel-audit-2026-05-11-synthesis.md` |
| **Phase C v1** (2026-05-11) | Post-Phase-B `--optimize-kernels --use-fused-zloss`, single-node batch=128, lr_2d=5e-3 | single-node | **DIVERGED step 250** (grad 79353). | `docs/perf/phase-c-divergence-analysis.md` |
| **Phase C v2** (2026-05-11) | Post-Phase-B `--optimize-kernels` only, single-node batch=128, lr_2d=5e-3 | single-node | **DIVERGED step 250** (grad inf). | same |
| **Phase C v3** (2026-05-11) | Post-Phase-B `--optimize-kernels`, DDP batch=256, lr_2d=5e-3 | DDP | **DIVERGED step 250** (grad 272). | `docs/perf/phase-c-final-analysis.md` |
| **Phase G** (2026-05-11) | Post-Phase-B `--optimize-kernels`, OdinHalo, DDP batch=256, lr_2d=2e-3, 1000 steps | DDP | Tracked −0.76 loss vs B4 through step 700. **DIVERGED step 750**. | `docs/perf/phase-g-findings.md` |
| **Phase I** (2026-05-11) | Isolated bench: Triton fused_swiglu vs autograd-safe HIP at production shape | single-node | **FAIL ship gate** — Triton 0.99× HIP. HIP IS 1.45× eager. | `docs/perf/triton-swiglu-ship-gate-bench.md` |

### 11.1 Non-divergence-related findings in this session

- **QKV fusion (Track 2.a, commit `63de5be`)**: consolidated `wq`/`wk`/`wv` into single `wqkv` Linear. Passed the 6-test unit suite. 200-step probe showed +0.07% throughput (within noise). Shipped because it's cleaner architecture with zero regression.
- **Track 1.3 profile**: z-loss (aten::logsumexp + backward) = 16.7% of step wall. Motivated Phase B.5.
- **Phase B.5 `--use-fused-zloss`**: extends `_CrossEntropyHIP` to bake z-loss into forward + backward. Gradient parity with eager tested within 5% rel error. Not shipped as default (Phase C v1/v2 both failed for unrelated reasons; not conclusively validated).
- **Static audit tool** (`scripts/audit_autokernel_replacements.py`): AST-based classifier. Pre-Phase-B: 5 UNSAFE + 1 CONDITIONAL-SAFE + 1 UNKNOWN. Post-Phase-B: 0 UNSAFE + 2 CONDITIONAL-SAFE + 5 SAFE.

### 11.2 Diagnostic tools built

- `scripts/audit_autokernel_replacements.py` — static AST audit.
- `scripts/autokernel_coverage_matrix.py` — pattern×model firing matrix.
- `scripts/audit_phase_a3_batch.sh` — 7-model × 3-config diagnostic batch runner.
- `scripts/analyze_audit_phase_a3.py` — aggregate frozen-params analyzer.
- `scripts/analyze_diag_frozen_params.py` — single-file analyzer (V0/V1/V3 comparison).
- `scripts/phase_i_triton_ship_gate.py` — ship-gate bench.
- `scripts/kernel_parity_harness.py` — fwd+bwd parity across shapes/dtypes.
- `scripts/kernel_bench_harness.py` — isolated throughput.
- `--diag-frozen-params PATH` flag on `scripts/train_ddp.py` — per-step per-param JSONL grad-norm recorder.
- `--profile-steps START:END` flag — torch.profiler over opt-step range.

### 11.3 Known A.3 probe limitations

`odin_flat_mini` and `odin_halo_mini` failed at probe startup (exit code 1, HIP SIGABRT). Root cause not isolated — likely an architectural incompatibility between the probe's `batch=4 block=256` config and the Mini variants' small hidden dims. Not pursued because:
- Mini variants are not on the production path.
- The 5 successful Odin probes cover the same code paths in `autokernel/_patterns.py`.

---

## 12. What we considered but skipped

This table captures things we discussed but did not execute, for the research agent to weigh in on.

| # | Option | Reason for skip | Expected effect (our guess) | Cost if tried |
|---|---|---|---|---|
| 1 | **OdinFlat + lr_2d=2e-3 + `--optimize-kernels`** | Phase I ship-gate failed (Triton ties HIP), so `--optimize-kernels` has no throughput upside on OdinFlat. Sprint 3A ships without it regardless. | Would likely pass stability (predicted by H1's linear-in-LR scaling). Zero throughput gain. | ~1h DDP probe. |
| 2 | **OdinFlat + warmup_steps=600 + `--optimize-kernels`** | Same reason as #1. | Would delay divergence by ~2× per H1. | ~1h DDP probe. |
| 3 | **Combined lowered LR + extended warmup** | Same reason as #1. | Belt-and-suspenders. H1 predicts stable. | ~1h DDP probe. |
| 4 | **max-grad-norm=0.5** | Grad clip acts AFTER overflow already happened in GradScaler; doesn't help inf grads. | Minimal effect. | ~1h DDP probe. |
| 5 | **Drop SPECTRA post-clip** | Removes one "intensity" knob but Phase C observed in configs WITH SPECTRA — not proven to be the culprit. Small quality cost (-0.04 loss per Phase 1.5 C). | Unclear. | ~1h DDP probe + quality regression risk. |
| 6 | **Warm-start `w_gate_up` from a Sprint-3A-confirm checkpoint** | Would test H1 directly (seed unfrozen params from already-trained values, skip the catch-up phase). Requires re-engineering of checkpoint loading to load ONLY w_gate_up. | Strong H1 test. Would stabilize if H1 is correct. | ~2h engineering + 1h probe. |
| 7 | **Two-stage training** (start frozen, gradual thaw) | Radical recipe change. Would require new trainer code. | Would definitely work if H1 is correct but overkill. | 4-8h engineering. |
| 8 | **Per-parameter LR multipliers** (lower lr on `w_gate_up` specifically) | Simpler than #7. Would target H1 directly. Not in our μP framework. | Direct H1 test. | 2-3h engineering + 1h probe. |
| 9 | **NorMuon momentum adjustments** (β₁ lower, e.g. 0.85) | Would reduce amplification of large gradient moments. Couples with NS-iteration intensity. | Might delay or fix divergence; unclear if quality impact. | 1h probe. |
| 10 | **NorMuon NS iterations** (5 → 3) | Less aggressive orthogonalization. Could reduce gradient amplification. | Similar to #9. | 1h probe + NS-convergence test. |
| 11 | **Initial weight scale** for `w_gate_up` (Kaiming vs smaller init) | If H1 is correct, smaller init → less catch-up → more stable. | Could work. | 1h probe. |
| 12 | **Gradient noise injection during warmup** | Regularizes magnitude of updates. Unusual but viable. | Speculative. | 1h probe. |
| 13 | **StabilityGuard rollback enabled** | Would catch the divergence and rollback + halve LR. Doesn't fix root cause but could let training recover. | Would produce a half-LR model that continues training. | 1h probe + checkpoint engineering. |
| 14 | **Layer-wise LR** (deeper layers slower) | Standard stability trick. Not straightforward with our μP 3-way grouping. | Speculative. | 2h engineering. |
| 15 | **Full-fusion Triton SwiGLU (w_gate_up + silu + w_down)** | Would replace 2 GEMMs + elementwise with 1 custom kernel. Mentioned in author guide but not built. | Could beat HIP ship gate. Would unlock `--optimize-kernels` path with real throughput gain. | 1-2 weeks kernel development. |
| 16 | **Revert Phase B for silu specifically** | Would re-introduce the silent-freeze. Defeats remediation purpose. | — | — (rejected) |
| 17 | **Deeper Phase I bench at other shapes** | Triton tied HIP at production shape. Might win at smaller shapes. | Unclear ROI. | 1-2h bench. |

**Agent, these are the experiments we think might work but didn't run.**

### 12.1 Skipped per user decision

After Phase I failed the ship gate, we explicitly skipped the Phase H bisect (options 1-5 above). Rationale: even if we stabilize `--optimize-kernels`, the throughput gain is neutral (HIP 1.45× eager on the SwiGLU op ≈ +5-7% total step wall, but this is offset by the autograd-overhead we observed in V2 measurements). The stability-work-for-zero-throughput trade-off didn't justify multi-hour probes.

### 12.2 Stability test we ran and didn't (yet)

- Didn't retry Phase C variants with `--no-spectra-post` to isolate whether SPECTRA contributes to instability.
- Didn't try disabling `--use-fused-zloss` + also disabling `--z-loss` entirely (would test whether z-loss interacts with post-fix HIP silu).
- Didn't profile step-by-step which layer's activations first cross fp16 threshold (would localize the instability to a specific layer).

---

## 13. Current production state

### 13.1 Sprint 3A (OdinFlat)

Recipe (locked):

```
--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4
--intra-doc-mask --value-residuals --head-gating
--z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0
--activation-monitor --activation-monitor-interval 200
--mup --mup-base-width 256
--spectra-post --spectra-clip-norm 1.0
--auto-eval
```

No `--optimize-kernels`. No `--use-fused-zloss`. Matches Sprint 3A-confirm (validated at loss 3.15 @ step 2000).

Expected: ~61 hours wall for 1 epoch dolma-10B (6.9B tokens) at ~31.3K tok/s aggregate. Target loss at epoch end: ~2.5-3.0 extrapolated from step-2000 trajectory.

### 13.2 Sprint 3B (OdinHalo)

Recipe (locked):

```
lr_2d=2e-3, warmup=300, block=256, batch=16×8×2=256 (DDP)
  + same Sprint 1 features as 3A
```

No `--optimize-kernels`. Expected: ~77 hours wall (vs ~48 hours with `--optimize-kernels`). Trade-off: +29h wall for correct full-parameter training. Pre-fix buggy code (48h, 14 frozen params) was rejected as not shippable.

### 13.3 What stays as future-proofing

Phase B fixes are kept in the codebase. `autokernel.optimize(model, training=True)` now produces a correctly-training model. Future use cases:
- **If Triton throughput improves** (better kernels, shape changes, ROCm updates), `--optimize-kernels` becomes a net win and we re-enable.
- **If recipe tuning succeeds** (per §12 options 1-14), `--optimize-kernels` ships on a future sprint.
- **If any new model** trains at sufficiently low LR that doesn't trigger the divergence, it can enable `--optimize-kernels` immediately.

The runtime preflight ensures that if any Replacement regresses to raw pybind in the future, training aborts with an actionable error before compute is wasted.

### 13.4 Guardrails active in production

- **CI smoke test** `scripts/test_autokernel_autograd_safety.py` — 2 tests, CUDA-only. Fails hard if V0 or V1 leaves any leaf param without finite grad.
- **Static audit** `scripts/audit_autokernel_replacements.py` — AST scan. Pre-merge gate.
- **Runtime preflight** in `train_ddp.py` — dummy batch, aborts on grad severance.
- **CONSTRAINTS.md** rules — author-facing docs with must-check list.
- **Knowledge article** `knowledge/training/autograd_safety_hip_kernels.md` — principle + workflow + debugging checklist.

---

## 14. Questions for the research agent

### 14.1 Hypothesis validation

1. **Rank our three hypotheses.** Do you agree H1 (unfrozen-param gradient statistics) is the primary cause? Can you identify a distinguishing experiment that cleanly separates H1 from H2/H3?

2. **Is the LR-scaling observation (1/LR × divergence step) diagnostic of a specific class of training instability?** Does this appear in the literature under a specific name?

3. **Is there a known phenomenon of "thawing-layer instability"** or similar when parameters transition from frozen to trainable during training? What are the standard mitigation recipes?

4. **Could this be a variant of the "linear warmup instability" phenomenon** observed in large-batch training? We use 300-step linear warmup; is a cosine or geometric warmup more stable in cases like ours?

### 14.2 Hypothesis ranking / bug-finding

5. **Is there a subtle bug we missed in the Phase B autograd registration?** Specifically:
   - `_silu_gate_mul_backward` uses a HIP backward kernel for fp16+CUDA inputs, fp32 fallback otherwise. Test_phase_b_autograd_safety passes parity at 5% rel error. Could accumulated fp16 rounding across 14 layers × 250 steps produce the divergence?
   - Phase B.5's z-loss gradient: we compute `(2*z_w/N)*lse[i]*softmax[i,j]` via recomputed softmax. Does this match what autograd would produce via the chain rule if z-loss were computed in pure PyTorch?

6. **Is our `_autokernel_autograd_preflight` check sufficient?** It runs one dummy forward+backward. Could a buggy replacement pass preflight but fail over long horizons?

### 14.3 Recipe / experiment proposals

7. **What experiments would you prioritize** from our skipped list (§12)? Which single intervention is most likely to stabilize post-fix `--optimize-kernels`?

8. **Two-stage training feasibility.** If we add a "frozen phase" where `w_gate_up` is explicitly frozen for N warmup steps, then unfrozen gradually, is there a principled way to choose N and the thawing schedule?

9. **Is layer-wise LR** (lower LR for deeper/earlier layers) a sensible response here? We have μP 3-way grouping but not layer-wise.

10. **Could changing the NorMuon β₁ momentum or NS iteration count** produce stability without destroying quality gains? What's the expected loss cost of β₁=0.85 vs 0.95 or NS iters 3 vs 5?

### 14.4 Hardware / kernel opportunities

11. **Given gfx1151's no-MFMA constraint**, are there kernel fusion opportunities we've missed beyond SwiGLU / z-loss / RMSNorm? What's the highest-ROI target for a hand-optimized kernel on this SKU?

12. **Phase I showed Triton ties autograd-safe HIP at production shape (0.99×).** Are there ROCm-specific Triton tiling / LDS / num_warps configurations we should try that might make Triton beat HIP?

13. **Would a full-fusion SwiGLU kernel** (fuse `w_gate_up` GEMM + silu·up elementwise + `w_down` GEMM into a single Triton dispatch) beat our current approach? Does the memory-bound nature of gfx1151 favor this?

14. **Is there value in exploring a lower-level HIP kernel** (avoiding Triton) for silu_gate_mul that uses specific LDS tiling for the 64 KB/CU budget? Or has rocBLAS already optimized this class of kernel?

### 14.5 Process / second opinion

15. **Our ship decision (drop `--optimize-kernels`, accept +29h wall on Sprint 3B).** Was this the right trade-off given the info we had? What would you have done differently?

### 14.6 Open invitation

Beyond these 15 questions, **please surface anything we didn't think to ask about.** Specifically:
- Is there a bug we missed?
- Is there a literature reference that would change our framing?
- Is there a kernel opportunity we're blind to?
- Is our fp16-stability stack over- or under-engineered?
- Are there tools or debugging approaches we haven't used (e.g. torch.profile for grad-magnitude tracking)?

### 14.7 Prioritized experiment menu (if you want to suggest what to run next)

Rank-ordered by our perceived "evidence yield per compute-hour":

| # | Experiment | Probe cost | What it tests |
|---|---|---|---|
| 1 | OdinFlat `--optimize-kernels` at lr_2d=2e-3, warmup=300 | ~1h DDP | H1: can lower LR stabilize? |
| 2 | Diff checkpoints at step 100 between post-fix and pre-fix on identical seeds/data | ~2h | H2: quantify numerical drift |
| 3 | Record per-layer activation maxabs JSONL during a Phase C run; see which layer spikes first | ~1h | H6: layer localization |
| 4 | OdinHalo `--optimize-kernels` at lr_2d=1e-3 | ~1h DDP | H1: extreme lr reduction |
| 5 | OdinFlat `--optimize-kernels` with NorMuon β1=0.85 (from default 0.95) | ~1h DDP | H4: reduce NS amplification |
| 6 | Warm-start w_gate_up from Sprint 3A-confirm step-500 checkpoint, continue with `--optimize-kernels` | ~2h engineering + 1h probe | H1: skip catch-up phase |
| 7 | OdinFlat with `--optimize-kernels --no-spectra-post` at lr_2d=5e-3 | ~1h DDP | H4: SPECTRA contribution |
| 8 | Layer-wise LR (outer layers faster, inner slower) | ~2h engineering + 1h probe | H6: localization |
| 9 | Initial scale experiments on w_gate_up | ~1h probe | H1 + H7 |
| 10 | Profile a Phase C run with torch.profiler to confirm where compute time goes post-fix | ~45min | Sanity check vs Track 1.3 |

**If you could only run ONE experiment:** OdinFlat at lr_2d=2e-3 with `--optimize-kernels`. Strong H1 test; ~1 hour wall; most information-dense per compute.

**Highest-risk-highest-reward experiment:** full-fusion Triton SwiGLU kernel (fuse w_gate_up GEMM + silu*up elementwise + w_down GEMM into one dispatch). If it beats HIP by >10% on production shape, enables `--optimize-kernels` to ship with real throughput gain. Estimated 1-2 weeks engineering + validation.

---

## Appendix A: Math derivations

### A.1 SwiGLU forward and backward

Forward:
$$
y = \text{SiLU}(g) \cdot u = (g \cdot \sigma(g)) \cdot u
$$

where $\sigma$ is the logistic sigmoid.

Backward:

For the output scalar $L$ and upstream gradient $\partial L / \partial y$, we need $\partial L / \partial g$ and $\partial L / \partial u$.

$$
\frac{\partial L}{\partial u} = \frac{\partial L}{\partial y} \cdot g \cdot \sigma(g) = \frac{\partial L}{\partial y} \cdot \text{SiLU}(g)
$$

$$
\frac{\partial L}{\partial g} = \frac{\partial L}{\partial y} \cdot u \cdot \sigma(g) \cdot \left(1 + g \cdot (1 - \sigma(g))\right)
$$

Plain text: `dy/dg = u * sigmoid(g) * (1 + g * (1 - sigmoid(g)))` and `dy/du = silu(g)`.

Triton kernel implementation (from `kernels/triton/fused_swiglu.py:78-97`):

```python
@triton.jit
def _fused_swiglu_bwd_kernel(
    GRAD_GATE_ptr, GRAD_UP_ptr,
    GATE_ptr, UP_ptr, GRAD_OUT_ptr,
    N, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    gate = tl.load(GATE_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(GRAD_OUT_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    silu_gate = gate * sig
    d_silu = sig * (1.0 + gate * (1.0 - sig))
    grad_gate = grad_out * up * d_silu
    grad_up = grad_out * silu_gate
    tl.store(GRAD_GATE_ptr + offs, grad_gate.to(tl.float16), mask=mask)
    tl.store(GRAD_UP_ptr + offs, grad_up.to(tl.float16), mask=mask)
```

### A.2 Z-loss forward, backward, and contribution to grad_logits

Forward:
$$
\mathcal{L}_z = w_z \cdot \text{mean}_i\left(\text{lse}(l_i)^2\right)
$$

where $\text{lse}(l_i) = \log \sum_j \exp(l_{ij})$ for logits row $i$.

Total loss:
$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_z
$$

Backward contribution to logits from $\mathcal{L}_z$:

$$
\frac{\partial \mathcal{L}_z}{\partial l_{ij}} = w_z \cdot \frac{\partial}{\partial l_{ij}}\left(\frac{1}{N}\sum_i \text{lse}(l_i)^2\right) = \frac{2 w_z}{N} \cdot \text{lse}(l_i) \cdot \frac{\partial \text{lse}(l_i)}{\partial l_{ij}}
$$

And:
$$
\frac{\partial \text{lse}(l_i)}{\partial l_{ij}} = \text{softmax}(l_i)_j
$$

So:
$$
\frac{\partial \mathcal{L}_z}{\partial l_{ij}} = \frac{2 w_z}{N} \cdot \text{lse}(l_i) \cdot \text{softmax}(l_i)_j
$$

This is what we add to `grad_logits` in Phase B.5's `_CrossEntropyHIP.backward`. Implementation (from `kernel.py:634-660`):

```python
if ctx.z_loss_weight > 0.0:
    with torch.no_grad():
        grad_loss_scalar = float(grad_loss.item()) if grad_loss.dim() == 0 else 1.0
        shifted = logits.float() - row_max.unsqueeze(-1)
        softmax = torch.exp(shifted) / row_sum.unsqueeze(-1)
        lse_vec = row_max + torch.log(row_sum)
        coef = (2.0 * ctx.z_loss_weight / ctx.n_valid_item) * grad_loss_scalar
        z_grad = coef * lse_vec.unsqueeze(-1) * softmax
        if ctx.ignore_index is not None:
            valid = (targets != ctx.ignore_index).float().unsqueeze(-1)
            z_grad = z_grad * valid
        grad_logits = grad_logits + z_grad.to(grad_logits.dtype)
```

### A.3 Newton-Schulz polynomial iteration

NorMuon's orthogonalization step approximates `U V^T` where `U, S, V = svd(raw_update)` via a 5-step polynomial iteration (Keller Jordan's formulation):

$$
U_{k+1} = a\, U_k + b\, U_k \left(U_k^T U_k\right) + c\, U_k \left(U_k^T U_k\right)^2
$$

with $(a, b, c) = (3.4445, -4.7750, 2.0315)$ optimized for convergence over 5 iterations given normalized input.

Cost per 2D param per opt step: 3 fp16 matmuls × 5 iterations = 15 matmuls. For OdinFlat's ~90M 2D params this is the dominant optimizer cost (~12.5% of step wall per Track 1.3).

### A.4 Autograd severance: formal mechanism

Let $y = f_{\text{pybind}}(x)$ where $f_{\text{pybind}}$ is a raw C++ extension call that does `torch::empty()` + in-place fill. Then:
- $y.\text{requires\_grad} = \text{False}$
- $y.\text{grad\_fn} = \text{None}$

Consider a downstream op $z = g(y, w)$ where $w$ is a leaf parameter with `requires_grad=True`.

During `z.backward()`, autograd traverses the graph from $z$. Because $y$ has `grad_fn=None`, traversal stops at $y$. The Jacobian-vector product $\frac{\partial z}{\partial y}^T \nabla_z L$ is not propagated upstream.

BUT: `w.grad` is still correctly computed because autograd computes it from the value of $y$ (not needing to traverse through it). Specifically for a linear op $z = y W^T$:
- $\nabla_W L = \nabla_z L \cdot y^T$ — correct
- $\nabla_y L = \nabla_z L \cdot W$ — conceptually should be computed but never gets to $x$'s ancestors

So the bug is asymmetric: the leaf parameter "right after" the severed node trains correctly, but any leaf parameters "upstream" (whose gradient would have to come through the severed node) get zero or None gradients.

---

## Appendix B: Key code snippets

### B.1 The bug (pre-Phase-B)

`autokernel/_patterns.py:582-588` (pre-fix):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate, up = self.w_gate_up(x).chunk(2, dim=-1)
    if gate.dtype == torch.float16:
        activated = self.kernel_fn(gate.contiguous(), up.contiguous())  # UNSAFE raw pybind
    else:
        activated = F.silu(gate) * up
    return self.w_down(activated)
```

### B.2 The fix (Phase B.1)

`autokernel/_patterns.py:554-617` (post-fix):

```python
class _FusedSwiGLUReplacement(nn.Module):
    def __init__(self, original: nn.Module, kernel_fn: Callable):
        super().__init__()
        self.w_gate_up = original.w_gate_up
        self.w_down = original.w_down
        self.kernel_fn = kernel_fn   # retained for inference/debug only
        try:
            import kernels.hip._torch_ops   # side-effect: registers custom ops
            self._autograd_op = torch.ops.autokernel.silu_gate_mul
        except Exception:
            self._autograd_op = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        if gate.dtype == torch.float16 and self._autograd_op is not None:
            activated = self._autograd_op(gate.contiguous(), up.contiguous())
        else:
            activated = F.silu(gate) * up
        return self.w_down(activated)
```

### B.3 Custom-op registration (kernels/hip/_torch_ops.py:161-201)

```python
@torch.library.custom_op("autokernel::silu_gate_mul", mutates_args=())
def silu_gate_mul_op(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    from kernels.hip.silu_gate_mul import kernel_fn
    return kernel_fn(gate, up)

@silu_gate_mul_op.register_fake
def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return gate.new_empty(gate.shape)

def _silu_gate_mul_setup(ctx, inputs, output):
    gate, up = inputs
    ctx.save_for_backward(gate, up)

def _silu_gate_mul_backward(ctx, grad_output):
    gate, up = ctx.saved_tensors
    if _use_hip_backward() and gate.dtype == torch.float16 and gate.is_cuda:
        from kernels.hip.silu_gate_mul_backward import kernel_fn as silu_bwd_fn
        return silu_bwd_fn(gate, up, grad_output)
    g = grad_output.float()
    gate_f = gate.float()
    up_f = up.float()
    sig = torch.sigmoid(gate_f)
    silu_gate = gate_f * sig
    d_silu = sig * (1.0 + gate_f * (1.0 - sig))
    grad_gate = g * up_f * d_silu
    grad_up = g * silu_gate
    return grad_gate.to(gate.dtype), grad_up.to(up.dtype)

silu_gate_mul_op.register_autograd(
    _silu_gate_mul_backward, setup_context=_silu_gate_mul_setup
)
```

### B.4 Runtime preflight (scripts/train_ddp.py)

```python
def _autokernel_autograd_preflight(model, device, args) -> tuple[bool, str]:
    """Dispatch one forward+backward on a dummy batch. Assert every
    requires_grad=True param receives a finite, non-None gradient.
    """
    import torch
    # ... construct dummy batch ...
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    model.train()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(x, targets=t) if _model_accepts_kwargs(model, "targets") else model(x)
        # ... compute loss ...
    loss.backward()

    ALLOWED_ZERO_PATTERNS = {
        "v_res_scale",  # first-layer, no v_prev
        "head_gate",    # only active when head_gate_active=True
    }
    offenders, none_offenders = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            if any(allowed in name for allowed in ALLOWED_ZERO_PATTERNS):
                continue
            none_offenders.append(name)
            continue
        gnorm = float(p.grad.detach().float().abs().sum().item())
        if not torch.isfinite(p.grad).all():
            offenders.append(f"{name} (non-finite grad)")
        elif gnorm == 0.0 and not any(allowed in name for allowed in ALLOWED_ZERO_PATTERNS):
            offenders.append(f"{name} (grad all zero)")

    if none_offenders:
        return False, f"{len(none_offenders)} parameters received grad=None: {', '.join(none_offenders[:5])}"
    if offenders:
        return False, f"{len(offenders)} parameter grads malformed: {'; '.join(offenders[:5])}"
    return True, ""
```

### B.5 Diagnostic: `--diag-frozen-params`

From `scripts/train_ddp.py` (closure captured at train-loop init):

```python
def diag_writer(m):
    rec = {"step": global_step, "params": []}
    for name, p in m.named_parameters():
        if p.grad is None:
            rec["params"].append({"name": name, "grad_norm": None,
                                   "is_none": True, "is_zero": False})
        else:
            g = p.grad.detach()
            gn = float(g.norm().item())
            rec["params"].append({"name": name, "grad_norm": gn,
                                   "is_none": False, "is_zero": (gn == 0.0)})
    _diag_fh.write(json.dumps(rec) + "\n")
    _diag_fh.flush()
```

The writer is invoked by `_complete_step` after `clip_grad_norm_` and before `zero_grad`, so grads are populated.

### B.6 Phase I ship-gate bench at production shape

From `scripts/phase_i_triton_ship_gate.py`:

```python
shape = (16, 512, 2048)
dtype = torch.float16

def eager_swiglu(gate, up):
    return F.silu(gate) * up

def autograd_hip_swiglu(gate, up):
    return torch.ops.autokernel.silu_gate_mul(gate.contiguous(), up.contiguous())

# HIP vs Triton (SHIP GATE)
r = bench_kernel_fwd_bwd(
    name="autograd-hip-vs-triton",
    triton_fn=fused_swiglu,
    reference_fn=autograd_hip_swiglu,
    shape=shape, dtype=dtype, input_count=2,
    warmup=50, iters=200,
)
# r["speedup_fwd_bwd"] = 0.99x at production shape → FAIL gate of 1.05x
```

---

## Appendix C: Commit timeline (this session)

Phase-labeled commits 2026-05-10 through 2026-05-11, most recent first:

```
add4ef9  Phase G: OdinHalo 1000-step verification DIVERGED at step 750
784ab4f  Phase G: launcher
d690711  Phase I: Triton 0.99x vs HIP (SHIP GATE FAIL)
40797f8  Phase I: ship-gate bench script
00090ee  STATUS.md: Sprint 3A/3B ship decisions locked (no --optimize-kernels)
bc0796a  Phase C final: v3 DDP diverged too
331732e  Session execution log addendum
d08bef8  Phase A.4: synthesis — Phase B fixes empirically validated
ac5bd91  Phase C v3: DDP launcher
e044e33  Phase A.3: rescope to Odin family (21 probes not 42)
68bffe4  Phase E: preflight allows head_gate=None (documented unused)
0919ff1  Phase C v2: drop --use-fused-zloss to isolate
aef44ed  Phase A.3: pass Sprint 1 flags to avoid preflight false positives
8f1330f  Session docs: master record + execution log
fca7cf3  Phase C v1: launcher
093bd3d  B4 findings: OdinHalo silent-freeze confirmed pre-fix
82e0655  Phase F.1: STATUS.md remediation summary
2a4dcb4  Phase F: knowledge/autograd_safety doc + AGENTS.md update
404b140  Phase E: runtime guardrails + CI smoke + CONSTRAINTS.md
2501dd9  Phase D.B: Triton fused SwiGLU kernel
eaccbd4  Phase D.A: Triton harness shipped
f24d8dd  Phase B.5+6: fused z-loss in _CrossEntropyHIP + tests
5ebe594  Phase B.1-4: wire autograd-safe paths for all UNSAFE replacements
3dcd43e  Phase A.2+3 tooling: coverage matrix + batch runner + analyzer
1f54ec0  Phase A.1: static audit tool
1c53c03  Track 3.F synthesis: docs/perf/autokernel-deep-analysis.md (pre-session)
24844a0  Track 2.a ship + 2.b z-loss fp16 opt (pre-session)
52bcf92  Track 3.A: frozen-params blast radius (pre-session)
63de5be  Track 2.a: QKV fusion (pre-session)
85f937e  Track 1.2-3: OdinFlat step profile (pre-session)
5b5ccaf  Track 1.1 + 3.A flags (pre-session)
0b5b99c  Execution plan (pre-session)
0e4c23b  Throughput investigation FINAL: +31% was artifact of broken autograd (pre-session)
```

---

## Appendix D: File index (what the research agent should read)

### D.1 Priority 1 (start here)

- **This document** (`docs/research/autokernel-stability-research-brief.md`) — self-contained brief.
- `docs/perf/autokernel-deep-analysis.md` — the root-cause synthesis from the prior session.
- `docs/perf/phase-c-final-analysis.md` — Phase C 3-variant divergence analysis.
- `docs/perf/phase-g-findings.md` — OdinHalo Phase G verification divergence.
- `docs/perf/session-2026-05-11-autokernel-remediation.md` — session summary.
- `docs/perf/session-2026-05-11-execution-log.md` — chronology + gotchas.

### D.2 Priority 2 (empirical data)

- `docs/perf/autokernel-frozen-blast-radius.md` — Track 3.A V0/V1/V3.
- `docs/perf/autokernel-audit-2026-05-11-synthesis.md` — Phase A.3/4 post-fix audit (21 probes).
- `docs/perf/autokernel-static-audit.md` + `.json` — static AST audit.
- `docs/perf/triton-swiglu-ship-gate-bench.md` — Phase I ship-gate data.
- `docs/perf/odinhalo-b4-findings.md` — pre-fix OdinHalo silent-freeze.
- `docs/perf/odinflat-throughput-final.md` — pre-session V0/V1/V2 throughput summary.
- `docs/perf/odinflat-step-profile.md` — Track 1.3 profile (where time goes).
- `docs/perf/phase-c-divergence-analysis.md` — Phase C v1/v2 (pre-v3).

### D.3 Priority 3 (principle docs)

- `knowledge/training/autograd_safety_hip_kernels.md` — principle doc.
- `knowledge/kernels/triton_author_guide.md` — Triton authoring.
- `knowledge/hardware/amd_rdna35_strix_halo.md` — hardware reference.
- `knowledge/training/fp16_stability_gfx1151.md` — fp16 stability stack.
- `knowledge/training/normuon_throughput_gfx1151.md` — NorMuon on gfx1151.
- `CONSTRAINTS.md` — checklist of rules.
- `AGENTS.md` — training commands, gotchas.
- `STATUS.md` — current training status + decisions.

### D.4 Priority 4 (source code)

- `autokernel/_patterns.py` — the 7 Replacement classes.
- `kernels/hip/_torch_ops.py` — 8 registered custom ops.
- `kernels/triton/fused_swiglu.py` — Phase D.B Triton kernel.
- `kernel.py` — HIP CE kernel + Phase B.5 z-loss extension.
- `models/odin_flat.py` — OdinFlat architecture.
- `models/odin_halo.py` — OdinHalo looped architecture.
- `models/components/attention.py` — NoPECodaAttention.
- `models/components/_components.py` — SwiGLU, RMSNorm.
- `halo_training/normuon.py` — NorMuon optimizer.
- `halo_training/spectra.py` — SPECTRA post-clip.
- `halo_training/mup.py` — μP init + param groups.
- `scripts/train_ddp.py` — DDP trainer with preflight.
- `scripts/audit_autokernel_replacements.py` — static AST audit.
- `scripts/test_phase_b_autograd_safety.py` — Phase B tests.
- `scripts/test_autokernel_autograd_safety.py` — CI smoke.

### D.5 Priority 5 (raw logs and data)

- `docs/perf/odinhalo-b4-rank0.log` — full B4 training log.
- `docs/perf/odinhalo-b4-diag.jsonl` — per-step per-param grad norms (pre-fix).
- `docs/perf/odinflat-profile-2026-05-10/diag-{V0,V1,V3}.jsonl` — Track 3.A probes.
- `docs/perf/odinflat-profile-2026-05-10/profile-summary.txt` — Track 1.3 profile.
- `docs/perf/phase-g-rank0.log` — Phase G full log (divergence visible).

---

**End of research brief. Agent, welcome to the problem. Looking forward to your analysis.**


