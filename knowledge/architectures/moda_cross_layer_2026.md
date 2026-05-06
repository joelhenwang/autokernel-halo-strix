---
title: "MoDA (Cross-Layer Depth Attention) — Different From Our Parcae MoDA"
domain: architectures
type: reference
status: active
tags: [moda, mixture-of-depths, cross-layer-attention, depth-scaling, naming-collision]
paper: "Mixture-of-Depths Attention (arXiv:2603.15619, 2026-03-16)"
code: "https://github.com/hustvl/MoDA"
related:
  - parcae_stable_looped_models.md
  - small_lm_arch_interventions_2026.md
  - ../../docs/research/broad-research-synthesis-2026-05-06.md
---

# MoDA (Cross-Layer) — Mixture-of-Depths Attention

## Naming collision warning

**Two different things share the name "MoDA":**

| Flavor | Our usage | New paper (2603.15619) |
|--------|-----------|------------------------|
| **Parcae MoDA** | Cross-**iteration** depth KVs (same layer, different iterations of Parcae loop) | — |
| **Paper MoDA** | Cross-**layer** depth KVs (different layers, same forward pass) | ✓ |

OdinHalo uses Parcae MoDA (see `knowledge/architectures/parcae_stable_looped_models.md`).
This doc is about the *new* cross-layer MoDA, which is orthogonal — they could
potentially be combined.

## What the new MoDA does

**Problem addressed:** As LLMs grow deeper, informative features formed in
shallow layers get diluted by repeated residual updates. By the deep layers,
the information is hard to recover.

**Mechanism:** Each attention head attends to **BOTH**:
1. Current-layer sequence KV pairs (standard self-attention)
2. **Depth KV pairs from preceding layers** (the novel part)

In other words, a head at layer L can directly attend to KV values computed at
layers 0..L-1 — bypassing the dilution via residual summation.

## Implementation — hardware-efficient algorithm

The paper describes an algorithm that resolves non-contiguous memory access
patterns inherent in cross-layer attention. Key result: **97.3% of
FlashAttention-2's efficiency** at 64K sequence length.

This matters because the naive implementation (store all prior-layer KVs, gather
them at each layer) has terrible memory locality. The paper's technique batches
and linearizes the accesses.

## Results at 1.5B scale

| Metric | Improvement | Cost |
|--------|------------:|-----:|
| Avg perplexity across 10 benchmarks | −0.2 | — |
| Avg downstream accuracy across 10 tasks | **+2.11%** | — |
| FLOPs overhead | — | **3.7%** |

Additional finding: **Post-norm + MoDA > Pre-norm + MoDA**. This is interesting
because modern LLMs universally use pre-norm; MoDA appears to reverse that
preference.

## Relevance to our models

### OdinFlat (122M, flat non-looped)

**Good fit.** OdinFlat's 14 layers have the depth-dilution problem this paper
solves. Adding cross-layer MoDA would:
- Let layer 13 directly see KV from layer 0 (the embedding output)
- Preserve fine-grained embeddings all the way through
- Cost ~3.7% more FLOPs (per-node throughput ~38K → ~36.5K tok/s)

### OdinHalo (58M unique / 157M effective, looped)

**Interesting but orthogonal.** OdinHalo already uses Parcae MoDA (cross-
iteration). Adding cross-layer MoDA on top would give both cross-iter AND
cross-layer depth attention.

But the looped architecture already partly addresses depth dilution by
re-running the same block (which forces shallow→deep information preservation).
Could be redundant.

### Implementation cost estimate

New component in `models/components/`:
- `attention_moda.py` — new `MoDAAttention` class wrapping our existing
  `NoPECodaAttention`, adds a prior-layer-KV cache and extends the attention
  key/value space
- Modify the forward path of the host model to thread the prior-layer-KV
  cache through all layers
- Memory: +K × (seq_len × n_kv_heads × head_dim × 2) bytes where K = number
  of prior layers cached. Choose K=4 as in the paper for negligible extra
  memory at our block=512.

Rough estimate: ~2 days to port + ablate.

## Experimental plan (when we're ready)

1. **Port minimal cross-layer MoDA** into `models/components/attention_moda.py`
2. **Add `--moda-cross-layer` flag** to model ctors (OdinFlat + OdinHalo variants)
3. **Baseline ablation:** train OdinFlat 500 steps with vs without, same data,
   same seed. Measure loss delta.
4. **Memory check:** verify memory overhead is the predicted 3.7% FLOPs +
   small KV cache growth
5. **If wins:** extend to OdinHalo for cross-iter + cross-layer combination

## Caveats

- **Tested only at 1.5B** in the paper. Transfer to 58-122M is unverified.
  Paper claims are specific to that scale.
- **Best paired with post-norm**; our current models use pre-norm. Would need
  to ablate post-norm + MoDA vs our current pre-norm + no-MoDA.
- **3.7% FLOP overhead is the measurement at 1.5B**. At smaller scales the
  overhead ratio may differ (typically smaller — less attention compute
  relative to MLP).
- **May interact with HyPE positional encoding** in non-obvious ways; our
  NoPE-in-attention design depends on content-only attention; adding
  prior-layer keys may re-introduce the position-dependence we avoided.

## Alternatives considered in the space

- **ResidualAttn** (older): Simpler cross-layer attention without the
  hardware optimization. Doesn't reach the 97.3% efficiency threshold.
- **Parcae MoDA** (what OdinHalo uses): cross-ITER, not cross-LAYER. Different
  mechanism for different dilution problem.

## Priority

**Medium-low.** The paper's +2.11% is real but small vs our other untapped
gains (NorMuon+CWD at −3.85%, more training tokens via dolma). MoDA is a
potential future architectural refinement but should come AFTER:

1. Optimizer upgrade to NorMuon+CWD
2. Basic arch interventions (value residuals, LN scaling)
3. Bigger training budget via dolma

Only revisit MoDA if those earlier wins are already landed and we're looking
for the next 1-2% loss improvement.

## See also

- `knowledge/architectures/parcae_stable_looped_models.md` — the OTHER MoDA
  (cross-iter, what OdinHalo actually uses)
- `knowledge/architectures/small_lm_arch_interventions_2026.md` — broader
  list of arch tricks to consider
- `docs/research/broad-research-synthesis-2026-05-06.md` Part 6 — MoDA in
  context of other architecture alternatives
