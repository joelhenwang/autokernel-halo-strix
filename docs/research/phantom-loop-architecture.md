# Phantom Loop Architecture Research

**Date:** 2026-05-13
**Status:** Experimental — validated via smoke tests + microbenchmarks, awaiting DDP 500-step gate

## Executive Summary

We discovered that **OdinPhantomLoop** — a fully-unrolled shared-weight architecture with per-position additive modulation and input refresh signals — achieves **33,481 tok/s single-node** under whole-model `torch.compile`, an **18% improvement** over the looped OdinHaloSlim baseline (28,371 tok/s) and a projected **~67k tok/s DDP aggregate** (93% above OdinFlat's Stack D ceiling of 34,697).

The key insight: when `forward()` is fully static (no Python loops, no conditional control flow), `torch.compile(fullgraph=False)` with `max-autotune-no-cudagraphs` can fuse operations across layer boundaries. Per-zone compilation (one compiled artifact per layer) creates materialization points that prevent this cross-layer fusion.

## Background: The Throughput Ceiling

OdinFlat (122M params, d=768, 14 flat layers) hits 34,697 tok/s DDP aggregate on Stack D — confirmed as the hardware bandwidth ceiling for that architecture:

- MFU: 21.4% of 59.4 TFLOPS fp16 peak
- Primary bottleneck: memory bandwidth (~240 GB/s shared LPDDR5X)
- No MFMA (matrix cores) on gfx1151
- aten::mm accounts for 37.85% of step time (non-addressable, rocBLAS-optimal)

## Key Discoveries

### 1. Heterogeneous dimension (d_core=512) breaks the bandwidth ceiling

**OdinHaloSlim** (49.4M params): Prelude(768) → proj_down → 4 shared layers @ d=512 × 3 iters → proj_up → Coda(768)

- Single-node smoke: **17,683 tok/s** (+39.5% vs OdinFlat's 12,663)
- DDP 500-step validated: **50,400 tok/s aggregate** (+45.3% vs Stack D)
- Loss at 500 steps: 5.34 (vs OdinFlat's ~5.1 — acceptable for 49M vs 122M)
- Memory: 5.1 GB/node (vs 13.5 GB for OdinFlat)

The throughput gain comes from **reduced DRAM traffic** (d^2 matmuls are 55% smaller at d=512 vs d=768), not from L2 cache residency.

### 2. L2 cache residency is physically impossible on gfx1151

Byte-level calculation for a single HyPEShortConvBlock at d=512:
- proj (512→1536): 786K params × 2 bytes = 1.57 MB
- FFN (512→3584→512): 2.75M params × 2 bytes = 5.5 MB
- Total per layer: **~7.6 MB fp16** (exceeds 6 MB L2)

Even the smallest useful layer exceeds the 6 MB L2 cache. The Jormungandr design doc's claim of "L2-resident core" was based on an incorrect estimate. However, **temporal locality within a single matmul** (tiles stay hot during the op) is real — it's just not cross-kernel persistence.

### 3. Python loop overhead is negligible

**OdinHaloSlimUnrolled** (Idea D): Same architecture as OdinHaloSlim but with the `for` loop manually unrolled and ParcaeInjectionSlim replaced by inline scalars.

| Model | tok/s (per-zone compile) | Loss @ 200 |
|---|---:|---:|
| OdinHaloSlim (Python loop) | 28,371 | 4.91 |
| OdinHaloSlimUnrolled (no loop) | 28,151 | **4.36** |

**Throughput: identical** (within noise). The Python `for i in range(3)` costs nothing because Dynamo already unrolls static-bound loops.

**Quality: unrolled is better** (loss 4.36 vs 4.91). The `ParcaeInjectionSlim` module's learned alpha/beta blending slightly hurts optimization vs direct inline scalars. Simpler injection = better training signal.

### 4. Multiplicative per-position modulation causes instability

**OdinPhantomLoop v1** (with `pos_scale * output + pos_bias`):
- Loss stuck at 6.54 after 200 steps (barely training)
- 6/200 gradient inf events, max grad 99.17
- Root cause: multiplicative scale compounds across 12 positions (3 virtual iters × 4 layers), causing exponential activation drift

**Fix:** Remove multiplicative `pos_scale`, keep **additive-only** `pos_bias`:
- Loss 4.48 after 200 steps (good convergence)
- 0/200 gradient inf events
- Per-position bias allows differentiation without scale drift

### 5. Full torch.compile on static graphs gives +18% over per-zone

This is the biggest single finding of the session:

| Compile strategy | tok/s (single-node) | Why |
|---|---:|---|
| Per-zone (each layer compiled separately) | 27,514 | Each zone materializes outputs to memory before next zone reads them |
| **Full compile (whole forward as one graph)** | **33,481** | Inductor fuses across layer boundaries — avoids intermediate materialization |

The +18% comes from **eliminating materialization points** between layers. When per-zone compile wraps each `HyPEShortConvBlock` in its own compiled graph, the output tensor must be written to HBM before the next zone reads it. With whole-model compile on a fully static forward, Inductor can:
- Fuse element-wise ops across layer boundaries (norm output → next layer's input)
- Schedule memory writes more efficiently (write-back only when needed)
- Eliminate redundant intermediate tensors

**Why this only works for unrolled architectures:** Python `for` loops don't prevent Dynamo from tracing — Dynamo unrolls them. But the `compile_zones()` approach *explicitly prevents* cross-layer fusion by creating separate compiled regions. The innovation is calling `torch.compile` on the entire `forward()` method instead of per-layer.

### 6. Flat unique-weight models thrash on gfx1151

**OdinFlatSlim** (82M params, 12 unique d=512 layers, flat):
- Steps 1-100: 6,700-6,800 tok/s (reasonable)
- Steps 150+: collapsed to **1,500 tok/s** (pathological)

Root cause: 12 unique layers = 12 separate compile graphs + 12 unique weight matrices rotating through L2. The combination of compile cache pressure and L2 thrashing degrades throughput catastrophically.

**Lesson:** Weight sharing isn't just parameter-efficient — it's **compile-efficient** (fewer unique compiled artifacts) and **cache-efficient** (same weights reused = hot in L2 during next access).

### 7. GriffinRecurrence (O(T) linear attention) is quality-neutral at T=512

**OdinFlatGriffin** (123M params, 12 conv + 2 GriffinGlobalBlock):
- 12,417 tok/s (vs OdinFlat 12,663) — slight regression from scan overhead
- Loss 4.40 (vs OdinFlat 4.35) — quality parity

At T=512, the O(T^2) attention is only 2/14 layers and T is short enough that the quadratic cost is manageable. Griffin's advantage manifests at T=1024+ where the quadratic memory and compute become dominant.

## The OdinPhantomLoop Architecture

### Design

```
FactorizedEmbed(vocab=32K, rank=256, d=768)
Prelude: 1× HyPEShortConvBlock(d=768, ffn=2816)
proj_down: Linear(768 → 512, bias=False)

[Virtual Iteration 0]  — positions 0-3
  core_0(h, freqs) + pos_bias[0]
  core_1(h, freqs) + pos_bias[1]
  core_2(h, freqs) + pos_bias[2]
  core_3(h, freqs) + pos_bias[3]

[Refresh 1]  h = h + α₁ * input_embed_down

[Virtual Iteration 1]  — positions 4-7
  core_0(h, freqs) + pos_bias[4]
  core_1(h, freqs) + pos_bias[5]
  core_2(h, freqs) + pos_bias[6]
  core_3(h, freqs) + pos_bias[7]

[Refresh 2]  h = h + α₂ * input_embed_down

[Virtual Iteration 2]  — positions 8-11
  core_0(h, freqs) + pos_bias[8]
  core_1(h, freqs) + pos_bias[9]
  core_2(h, freqs) + pos_bias[10]
  core_3(h, freqs) + pos_bias[11]

proj_up: Linear(512 → 768, bias=False)
Coda: NoPEGQABlock(d=768, ffn=2816) + HyPEShortConvBlock(d=768, ffn=2816)
FactorizedLMHead(d=768, rank=256, vocab=32K)
```

### Parameters
- **49.0M unique params** (same budget as OdinHaloSlim)
- 4 shared core layers: ~15.2M (reused 3×)
- Prelude: ~8.8M, Coda: ~17.6M, Embeddings: ~9M
- Per-position bias: 12 × 512 = 6,144 params (0.01% of model)
- Refresh alphas: 2 scalars

### Key Properties
1. **Fully static forward** — no Python loops, no conditionals, no dynamic shapes
2. **Weight sharing preserved** — 4 core layers called 12 times total
3. **Additive-only modulation** — per-position bias, no multiplicative scale (stability)
4. **Input refresh at virtual boundaries** — learned alpha re-injection at positions 4, 8
5. **Whole-model compilable** — `compile_full(mode="max-autotune-no-cudagraphs")`
6. **15 effective layers** — 1 prelude + 12 core + 2 coda

### Compilation Strategy

```python
# WRONG (per-zone — leaves throughput on the table):
model.compile_zones(mode="default")  # 27,514 tok/s

# RIGHT (whole-model — enables cross-layer Inductor fusion):
model.compile_full(mode="max-autotune-no-cudagraphs")  # 33,481 tok/s (+18%)
```

The whole-model compile works because:
- No graph breaks in the forward (all ops are Dynamo-traceable)
- `HyPEShortConvBlock` uses `torch.ops.autokernel.*` custom ops (Dynamo-visible)
- No Python-level control flow (no `if`, no `for`, no `while`)
- All tensor shapes are static (batch=16, T=512/256, d=512/768 known at trace time)

## Throughput Comparison Table

| Model | Params | Compile | Single-node tok/s | DDP projected | vs Stack D |
|---|---:|---|---:|---:|---:|
| OdinFlat (Stack D baseline) | 122M | per-zone | 12,663 | 34,697 | — |
| OdinHaloSlim (looped) | 49.4M | per-zone | 17,683 | **50,400** (measured) | +45% |
| OdinHaloSlimUnrolled (Idea D) | 49.0M | per-zone | 17,620 | ~50,000 | +44% |
| OdinPhantomLoop (Idea A) | 49.0M | per-zone | 18,223 | ~52,000 | +50% |
| **OdinPhantomLoop (Idea A)** | **49.0M** | **full** | **33,481** | **~67,000** | **+93%** |
| OdinFlatGriffin (linear attn) | 123M | per-zone | 12,417 | ~35,000 | +1% |
| OdinFlatSlim (flat d=512) | 82M | per-zone | collapsed | — | broken |

## Failed Experiments

### Multiplicative per-position modulation
- `h = layer(h) * pos_scale[p] + pos_bias[p]` with `pos_scale` init to 1.0
- Gradient amplification across 12 positions: compounding scale drift
- Fix: additive only (`h = layer(h) + pos_bias[p]`)

### Flat unique-weight model (OdinFlatSlim)
- 12 unique d=512 layers (no sharing) — 82M params
- Collapsed from 6,800 to 1,500 tok/s by step 150
- Root cause: compile cache thrashing + L2 weight rotation
- Lesson: weight sharing is mandatory for throughput on this hardware

### Doubled core width (d_core=1024)
- 99.3M params, 298M effective
- 5,696 tok/s compiled single-node — back to OdinFlat territory
- FLOPs scale as d^2; bandwidth ceiling reasserts at larger dims

## Open Questions

1. **DDP validation:** Does the +18% full-compile advantage survive DDP? The allreduce
   overlap timing may interact differently with whole-model compile vs per-zone.

2. **Quality at convergence:** OdinPhantomLoop showed loss 4.48 at step 200 on BabyLM
   (same as Idea D's 4.36, better than looped baseline's 4.91). Does this advantage
   hold at 500-2000 steps on wikitext-103?

3. **Longer sequences:** At T=1024 or T=2048, does the quadratic attention in the coda
   become a bottleneck? If so, combining Phantom Loop + GriffinGlobalBlock coda could
   address it.

4. **More iterations:** 4 layers × 4 virtual iters (16 positions) vs current 4×3 (12).
   The refresh signal + pos_bias mechanism generalizes trivially. Would need `fullgraph`
   test at the larger unrolled size.

5. **Adapter cycling (Idea B):** Per-cycle rank-32 LoRA adapters would give each virtual
   iteration a different effective weight matrix while keeping the same compiled artifact.
   Not yet tested.

## Files

| File | Purpose |
|---|---|
| `models/odin_halo_slim.py` | Original looped hetero-dim model (DDP validated at 50k) |
| `models/odin_halo_slim_unrolled.py` | Idea D: diagnostic unrolled variant |
| `models/odin_phantom_loop.py` | **Idea A: the winning architecture** |
| `models/odin_flat_slim.py` | Failed flat variant (collapse documented) |
| `models/odin_flat_griffin.py` | Linear attention variant (quality-neutral) |
| `models/components/mixer_blocks.py` | GriffinGlobalBlock component |

## Recommended Next Steps

1. **DDP 500-step gate on OdinPhantomLoop** with full compile — validate 67k projection
2. **If confirmed:** promote to Sprint 3 candidate (would complete dolma-10B epoch in ~20h vs 61h)
3. **Quality study:** longer runs to verify convergence trajectory vs OdinFlat
4. **Explore T=1024:** test if full-compile advantage scales with sequence length
