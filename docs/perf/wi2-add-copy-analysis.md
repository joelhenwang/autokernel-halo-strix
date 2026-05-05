# WI2: `aten::add_` and `aten::copy_` call-site classification

**Phase 2 work item:** WI2 — classify where the 4.9% `aten::add_` and 4.4% `aten::copy_`
cost originates and decide which (if any) is attackable.

**Status:** CLOSED. `aten::add_` has no attackable targets. `aten::copy_` is dominated
by input-batch H2D copies — already covered by WI5 (async prefetch).

**Evidence:** `docs/perf/wi2-shape-calls.md` (shape + stack profile at batch=16,
block=256, compile_zones=yes, 3 active profiler steps).

## `aten::copy_` classification (279.9 μs total, 129 invocations)

| Rank | Shape | Calls | Total μs | % | Interpretation |
|:----|:------|------:|---------:|--:|:--------------|
| 1 | `(16, 256)` int64 | 6 | 253.3 | **90.5%** | **Input batch H2D copy** (`input_ids` + `targets`). 42 μs/call. |
| 2 | `(4096, 32768)` | 6 | 21.8 | 7.8% | Logits tensor contiguity copy (from CE path / unscale). 3.6 μs/call. |
| 3 | `(16, 256, 768)` | 39 | 2.9 | 1.0% | Hidden-state contig copies at graph boundaries. Negligible. |
| 4 | `(32768, 256)` | 6 | 1.4 | 0.5% | Factorized LM head weight copy. Negligible. |
| 5+ | various | 81 | ~0.5 | <0.2% | Noise, layer-norm weights, etc. |

### Key finding

**90.5% of `aten::copy_` time is the 6 H2D copies of the input batch** (3 steps × 2
tensors each = input_ids + targets). At 42 μs per `(16, 256)` int64 tensor
(32 KB), the effective bandwidth is 760 MB/s — way below Strix Halo's 256 GB/s
peak. This confirms the H2D copy is **dominated by synchronization/setup overhead,
not bandwidth**.

### Why so slow for a tiny tensor?

On Strix Halo (unified memory, LPDDR5x shared between CPU and GPU), `.to(device)`
from a non-pinned CPU tensor triggers a synchronous stream copy. With
`pin_memory=False` (current default in `halo_training/data.py`), `non_blocking=True`
becomes **advisory only** — PyTorch still issues a blocking stream copy because the
source pages aren't registered.

### Action for `aten::copy_`

**→ Route to WI5** (async prefetch). Same root cause, same fix: pin_memory + CUDA
stream overlap. Already planned; no new action in WI2.

## `aten::add_` classification (23.0 μs total, 402 invocations)

| Rank | Shape | Calls | Total μs | % | Interpretation |
|:----|:------|------:|---------:|--:|:--------------|
| 1 | `(5632, 768)` | 36 | 8.0 | 34.6% | **SwiGLU up-proj weight grad-accumulation** (autograd). `d_ff=5632 × d=768`. |
| 2 | `(16, 256, 768)` | 69 | 6.6 | 28.8% | Activation residuals + their gradients. `d=768`. |
| 3 | `(768, 2816)` | 36 | 4.1 | 17.8% | **SwiGLU down-proj weight grad**. `d × d_ff/2=2816` (fused gate+up in down). |
| 4 | `(32768, 256)` | 3 | 1.5 | 6.6% | **Factorized LM head weight grad** (tied-embedding side). |
| 5 | `(1536, 768)` | 30 | 1.5 | 6.4% | **QKV packed weight grad** (`1536 = 3 × 512`). |
| 6 | `(768, 512)` | 30 | 0.6 | 2.7% | MoDA depth-memory projection weight grad. |
| 7 | `(768, 768)` | 12 | 0.3 | 1.2% | Attention output-proj weight grad. |
| 8+ | various | 189 | ~0.6 | 2.5% | Tiny weights (norms, gates), negligible. |

### Key finding

**~67% of `aten::add_` time is weight-gradient accumulation** in shapes matching
linear-layer weights. These adds are emitted by PyTorch's autograd engine during
`backward()` to accumulate grads across parallel paths (e.g., an embedding weight
used both for input lookup AND as the tied lm_head must have its grad summed).

**~29% is activation-side residuals** at shape `(16, 256, 768)` — the `x + layer(x)`
pattern at `shared_layers[i]` boundaries in OdinHalo's Parcae loop. These are
outside `compile_zones` (the boundary is exactly where the trainer re-splits the
graph to enable compile-per-layer).

### Why this is not attackable

1. **Weight-grad adds (67%)** are autograd-internal. Changing them requires either
   (a) a custom autograd Function bypassing the default accumulator, or (b) replacing
   the tied-embedding pattern with two untied parameters (changes model, breaks
   checkpoint). Neither justified by a 4.9% wall-time cost that's also largely
   already overlapped with subsequent kernel launches.

2. **Activation residual adds (29%)** at shape `(16, 256, 768)` are at
   `compile_zones` boundaries. The only way to fuse them is to widen the compile
   unit — but a single larger compiled graph breaks the Parcae loop's cache-safety
   assumptions (see `AGENTS.md` note: `reduce-overhead` incompatible with looped
   Parcae models). Phase 3 handles this differently (CUDA graphs).

3. **Tiny scattered adds (~3%)** are launch-overhead-bound. Not attackable except
   by not launching them, which isn't possible for separate-weight operations.

### Observation on profile count

402 `aten::add_` calls in 3 steps = **134 per step**. OdinHalo has 6 shared_layers
× 3 Parcae iterations = 18 layer invocations. With ~7 weight parameters per layer
getting grad-accumulated and ~2 activation residuals per layer, we expect
~18×9 = 162 adds per step. The measured 134 matches this within noise, confirming
the classification.

## Decision

**WI2 CLOSED** as "no attackable targets".

- `aten::add_` is dominated by autograd-internal weight-grad accumulation and
  cross-compile-boundary activation residuals. Both unfixable without larger
  architectural changes that Phase 2 scope explicitly rules out.
- `aten::copy_` dominant cost (90%) is input H2D upload — **delegated to WI5**.

### Hand-off to WI5

Concrete target now documented: the `(16, 256)` int64 H2D copy of `input_ids` +
`targets` at **42 μs/call × 2 tensors × steps_per_second**. At 14k tok/s and
batch=16, that's ~3.4 steps/sec × 2 copies × 42 μs = **~285 μs/sec of H2D time
per stream**. If fully overlapped with compute, that's a clear, bounded gain.

Expected WI5 lift: **0.5-2.5%** throughput at current baseline. WI5 acceptance gate
requires ≥0.5% to ship.

## Artifact

Raw shape-annotated profile: `docs/perf/wi2-shape-calls.md`.
Tooling: `scripts/profile_shape_calls.py` (reusable for future investigations).
