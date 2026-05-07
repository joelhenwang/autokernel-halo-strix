# NorMuon Profile Summary — Sprint 1.1 Phase A

**Measurement date:** 2026-05-07
**Hardware:** 2× AMD Strix Halo (gfx1151) — Radeon 8060S
**Configuration:** single-node `scripts/profile_step.py`, 50 warmup + 100 measured steps, `batch=16 block=512 accum=1`
**Model:** `models/odin_flat.py::OdinFlat` (121.7M params)
**Dataset:** `datasets/wikitext-103-odin32k.bin`
**PyTorch:** torch+ROCm 7.12
**Compile:** `torch.compile(mode="default")` (per profile_step.py, applied before optimizer)

---

## Step time attribution

Absolute numbers include `torch.profiler` overhead which inflates
NorMuon cost vs production training (where NorMuon is measured at 17.8%
of step time for Run 2 full recipe). The **attribution ratios** within
each config are what Phase B targets.

| Config | Total CUDA time (100 steps) | CUDA/step | `aten::mm` | NorMuon.step | NS matmul delta |
|---|---:|---:|---:|---:|---:|
| **P-AdamW** (baseline)      | 45.3 s  | 453 ms | 25.5 s (56%) | — | — |
| **P-NorMuon** (no v_res/head_gate) | 107.9 s | 1079 ms | 82.5 s (76%) | 63.8 s (**59%**) | +57.0 s |
| **P-Full** (Run 2 config)   | 118.3 s | 1183 ms | 92.5 s (78%) | 73.9 s (**62%**) | +67.0 s |

**Delta interpretation:**
- `P-NorMuon mm − P-AdamW mm = 57.0 s` = Newton-Schulz matmul cost
- `P-Full NorMuon.step − P-NorMuon NorMuon.step = 10.0 s` = cost added by
  value_residuals + head_gating (both route matmuls through the 2D NorMuon
  group; NS matmul count rises accordingly)
- `NorMuon.step − NS matmul delta` = non-matmul optimizer overhead (Python
  loop + neuron-norm + cautious-WD mask + momentum update)
  - P-NorMuon: 63.8 − 57.0 = **6.8 s** (6% of total step)
  - P-Full: 73.9 − 67.0 = **6.9 s** (same; v_res + head_gate add mm, not Python)

---

## Within NorMuon.step() breakdown (P-Full, estimated)

Top matmul kernels observed:

| rocBLAS kernel | Self CUDA | % step | Notes |
|---|---:|---:|---|
| `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_MT16x16x16…` | 22.2 s | 20.6% | fp32 NS @ medium shape |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_MT8x8x8…`   | 31.5 s | 26.7% | fp32 NS @ small shape |
| `Cijk_Ailk_Bjlk_HHS_BH_Bias_HA_S_SAV_MT64x96…`| 12.1 s | 10.3% | fp16 forward (unrelated to NS) |

`S_B_` prefix = **fp32 input, fp32 compute**. `HHS_BH_` = fp16 input, fp16→fp32 accumulate.
NS is routing through fp32 rocBLAS kernels because `NorMuon.__init__(ns_dtype=None)`
defaults to fp32 cast (see `halo_training/normuon.py:122`).

Non-matmul NorMuon cost (~6-7% of step):
- `aten::mul` (scale terms): ~1.9%
- `aten::add_` / `aten::mul_` (momentum buffer update): ~2.5%
- `aten::div` (neuron-wise normalize): ~1.1%
- cautious-WD mask ops: ~0.5%
- Python loop dispatch: <0.5%

Cautious-WD and neuron-norm are both cheap. The dominant cost is the
**5-step Newton-Schulz iteration at fp32** for every 2D parameter.

---

## torch.compile events

`docs/perf/normuon-compile-log.txt` (119 lines, with TORCH_LOGS=graph_breaks,recompiles).

**Graph breaks: 2, both intentional and pre-existing:**
1. `fused_rope_mul` in `models/components/conv_blocks.py:284` — wrapped with
   `torch.compiler.disable` (our HIP kernel; correct behavior).
2. `_causal_conv1d_fwd_cpp` in `CausalConv1dFn.apply` — aiter's conv1d op,
   dynamo cannot trace.

**Recompiles: 0.** No recompiles attributable to value_residuals or head_gating.

**NorMuon-attributable breaks: 0.** The optimizer `step()` is never inside
`torch.compile`'s wrapper — it runs from the trainer's Python scope.
Graph-break attribution is clean.

---

## Newton-Schulz micro-benchmark

`scripts/bench_newton_schulz.py`, 500 iters per config, both machines.

**Per-shape timing (Machine A, `docs/perf/normuon-ns-benchmark.json`):**

| Shape | fp32 (ms) | fp16 (ms) | Speedup |
|---|---:|---:|---:|
| (768, 768)   | 6.96   | 0.65  | **10.68×** |
| (768, 2816)  | 22.21  | 1.69  | **13.12×** |
| (2816, 768)  | 23.15  | 2.70  | **8.58×**  |
| (768, 128)   | 0.89   | 0.22  | 4.07×  |
| (128, 768)   | 0.84   | 0.22  | 3.85×  |
| (256, 768)   | 0.84   | 0.22  | 3.79×  |

**Machine B parity (`docs/perf/normuon-ns-benchmark-machineB.json`):**
- fp16 total: 123.2 ms (A: 122.3 ms) — **0.8% delta**, within noise floor
- fp32 total: 1752.6 ms (A: 1338.2 ms) — **31% delta**, Machine B fp32 is
  slower but fp16 is bit-identical. Makes the fp16 optimization MORE
  attractive on B (93% reduction vs 91% on A).

**Estimated per-step NS cost (Machine A, 84 2D params of mixed shapes):**
- fp32: 1338 ms/step
- fp16: 122 ms/step
- **Savings: 1216 ms/step (90.9% reduction)**

This far exceeds the observed 57s/100-step delta from profiler (570 ms/step),
because the micro-benchmark runs NS back-to-back with no forward/backward
interleaving. Real savings depend on overlap with other ops; the profiler
attribution gives a more realistic upper bound on Phase B1's win.

---

## Conclusions — Phase B ordering

1. **B1 (fp16 NS) is the dominant win.** NS matmuls are 53% of step time and
   fp16 cuts them 8-13× on the dominant SwiGLU shapes. Even with overlap,
   this should cut total NorMuon overhead from 17.8% to <5%.

2. **B2 (size-gated neuron-norm) is a minor polish.** Neuron-norm contributes
   ~1% of step time across all 2D params. Gating it at min_dim=512 skips
   only 3 params (tok_embeddings.projection 256×768, and 2 factorized head
   projections 768×128 / 128×768). Expected savings: <0.5% of step. Include
   for completeness but don't expect visible throughput impact.

3. **B3 (disable cautious-WD) is minor polish.** Mask compute is ~0.5% of
   step. Primary value is if cautious-WD is discovered to hurt quality at
   our scale (orthogonal concern).

4. **Batched NS (Phase C1) is viable.** 14 layers × similar shapes means
   many same-shape params can be stacked. Expected to add 10-20% on top of
   B1 fp16 via amortized kernel launch overhead.

5. **HIP NS kernel (Phase D) is NOT needed** unless fp16 NS + batched still
   leaves cost > 10% at Run 2b. Given the 90% reduction projected from fp16
   alone, Phase D is very likely unnecessary.

**Phase B execution order:** B0 (reference) → B1 (fp16 NS, biggest win) →
B2 (size gate, cheap) → B3 (no-CWD, quality check) → B4 (combo of winners).

Expected post-B1 throughput: 37,000-38,500 tok/s (vs Run 2 baseline 32,478
and AdamW reference 39,538). If achieved, Phase B passes the 7% cost gate
(target: ≤ 35,600 tok/s = 10% cost; well above floor 33,000 = 16.5% cost).

---

## Artifacts committed

```
docs/perf/normuon-profile-AdamW.txt           — Phase A.1 P-AdamW profile
docs/perf/normuon-profile-NorMuon.txt         — Phase A.1 P-NorMuon profile
docs/perf/normuon-profile-Full.txt            — Phase A.1 P-Full (Run 2) profile
docs/perf/normuon-compile-log.txt             — Phase A.5.1 dynamo events
docs/perf/normuon-ns-benchmark.json           — Phase A.5.2 NS bench (Machine A)
docs/perf/normuon-ns-benchmark-machineB.json  — Phase A.5.2 NS bench (Machine B)
docs/perf/normuon-profile-summary.md          — this file
scripts/profile_step.py                       — new: Sprint 1 flag-aware profiler
scripts/bench_newton_schulz.py                — new: standalone NS benchmark
scripts/test_sprint1_1_profile.py             — new: profile_step --help test
scripts/test_sprint1_1_bench.py               — new: NS bench smoke test
```
