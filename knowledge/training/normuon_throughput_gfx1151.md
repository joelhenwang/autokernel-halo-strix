---
title: "NorMuon Throughput on gfx1151: Empirical Results + Sprint 1.1 Fix"
domain: training
type: reference
status: active
tags: [normuon, muon, optimizer, throughput, gfx1151, strix-halo, newton-schulz, sprint1, sprint1.1]
paper: "IMU-1 (arXiv:2602.02522) — recipe reference"
related:
  - imu1_recipe_2026.md
  - muon_optimizer_results.md
  - ../kernels/fused_kernels.md
  - ../../docs/superpowers/specs/2026-05-06-sprint1-foundation-wins-design.md
  - ../../docs/superpowers/specs/2026-05-07-sprint1.1-normuon-throughput-design.md
---

# NorMuon Throughput on gfx1151 (122M OdinFlat, DDP)

Empirical results from Sprint 1 Phase 3 (2026-05-06), Sprint 1.1 profile
+ fix (2026-05-07), and the IMU-1 paper's claims compared side-by-side.

## TL;DR

- **Pre-Sprint-1.1 (fp32 NS):** 17.8% cost vs AdamW. NS fp32 matmuls
  dominated (53% of step time).
- **Post-Sprint-1.1 (fp16 NS, shipped as default):** **3.5% cost** vs
  AdamW, no quality regression. fp16 NS in rocBLAS runs 8-13× faster on
  SwiGLU-shape matrices.
- **IMU-1 paper's 3% claim** is now within reach on gfx1151 once you
  flip one dtype. No custom Triton or HIP kernel required.

## Before Sprint 1.1 — observed throughput cost

All runs OdinFlat (121.7M params), DDP over TB4 (2× Strix Halo),
block=512, batch=16, accum=8 (eff_batch=256), fp16 autocast,
`max-autotune-no-cudagraphs` compile.

| Run | Optimizer | Other features | tok/s | Cost vs baseline |
|-----|-----------|----------------|------:|-----------------:|
| Baseline (AdamW, fused, single group) | AdamW | — | 39,538 | — |
| Run 1b (AdamW two-group + free wins) | AdamW | intra-doc-mask + IMU-1 grouping (no LR split) + LN scaling + no-WD-on-embed | 39,104 | 1.1% |
| Probe A | NorMuon lr_2d=8e-4 lr_1d=8e-4 | + free wins | 34,363 | 13.1% |
| Probe B | NorMuon lr_2d=2e-3 lr_1d=8e-4 | + free wins | 34,596 | 12.5% |
| Probe C | NorMuon lr_2d=5e-3 lr_1d=8e-4 | + free wins | 33,234 | 15.9% |
| Run 2 (full recipe, fp32 NS) | NorMuon lr_2d=5e-3 lr_1d=8e-4 | + free wins + value residuals + head gating | 32,478 | 17.8% |

## After Sprint 1.1 — same config, fp16 NS

| Run         | tok/s    | Cost vs baseline | Δ loss vs Run 2 | Δ wiki_bpb vs Run 2 |
|-------------|---------:|-----------------:|----------------:|--------------------:|
| **Run 2b (full recipe, fp16 NS)** | **38,162** | **3.5%** | +0.01% | +0.17% |

Single-flag difference from Run 2: `--ns-dtype fp16` (now the default).
Quality within noise floor; throughput +17.5% over Run 2.

## Root cause: NorMuon was running fp32 matmuls in Newton-Schulz

Sprint 1.1 Phase A profile (`docs/perf/normuon-profile-summary.md`):

| Config | CUDA/step | aten::mm | NorMuon.step | NS rocBLAS prefix |
|---|---:|---:|---:|---|
| P-AdamW (baseline)          | 453 ms  | 56% | —   | `HHS_BH_` (fp16) |
| P-NorMuon (fp32 NS default) | 1079 ms | 76% | 59% | `S_B_` (fp32)    |
| P-Full (Run 2 config)       | 1183 ms | 78% | 62% | `S_B_` (fp32)    |

**Why fp32 was the default:** `halo_training/normuon.py::NorMuon.__init__`
has `ns_dtype: Optional[torch.dtype] = None`, and `build_imu1_optimizer`
never set it explicitly. The NS iteration internally does
`X = X.to(dtype or torch.float32)` — so absent an explicit dtype, fp32.

**Why fp16 works:** gfx1151 has no MFMA, but it DOES have native fp16
WMMA. rocBLAS dispatches `Cijk_..._HHS_BH_...` half-precision kernels
for 2D matmuls sized like OdinFlat's SwiGLU (768×2816) and square
projections (768×768). Those kernels are 8-13× faster than the
single-precision equivalents (Phase A bench, n=500):

| Shape         | fp32 (ms) | fp16 (ms) | Speedup |
|---------------|---------:|---------:|--------:|
| (768, 768)    | 6.96     | 0.65     | 10.7×   |
| (768, 2816)   | 22.21    | 1.69     | **13.1×** |
| (2816, 768)   | 23.15    | 2.70     | 8.6×    |
| (768, 128)    | 0.89     | 0.22     | 4.1×    |
| (128, 768)    | 0.84     | 0.22     | 3.9×    |
| (256, 768)    | 0.84     | 0.22     | 3.8×    |

Machine B parity: fp16 total within 0.8% of Machine A, confirming
deterministic behavior. fp32 is 31% slower on Machine B, which made the
fp16 switch MORE attractive there (93% NS reduction vs 91% on A).

## Quality gain justifying NorMuon at all (Run 2b results)

| Metric | Run 2b | Baseline (AdamW) | Δ |
|---|---:|---:|---:|
| Final train loss | **4.4741** | 4.7975 | **−6.7%** |
| wikitext_val BPB | **1.8962** | 1.9214 | −1.3% |
| gpt_small_val BPB | 2.8277 | 2.8634 | −1.2% |
| stem_crawl_val BPB | 3.4420 | 3.5361 | −2.7% |
| dolma_val BPB | 3.0822 | 3.1094 | −0.9% |
| avg BPB | 2.8120 | 2.861 | −1.7% |
| Memory | 10.1 GB | 10.3 GB | −1.9% |

Post Sprint 1.1 the full recipe delivers −6.7% training loss, beats
baseline on all 4 domains, **AND** achieves this with only 3.5%
throughput cost. The recipe is now Pareto-dominant.

## Other knobs that didn't make it

Sprint 1.1 Phase B tested two other throughput toggles; both have
been shipped as opt-in CLI flags but not adopted as default:

- **`--neuron-norm-min-dim 512`** — skip neuron-wise norm on matrices
  with `min(rows, cols) < 512`. +6.6% tok/s at step 200 but wiki_bpb
  drifted to 2.2242 (above the 2.22 gate by 0.09%). Rejected.
- **`--no-cautious-wd`** — remove the sign-mask cautious WD gate. +2.8%
  tok/s at step 200 (noise-floor). Clean quality. Kept available but
  not default.

Neither knob was needed; fp16 NS alone solved the throughput gate
with substantial margin.

## Interaction with torch.compile

Confirmed clean via `TORCH_LOGS=graph_breaks,recompiles`:

- **NorMuon.step is NOT compiled** (Python-side optimizer loop, correct).
- **2 graph breaks in the model forward:** `fused_rope_mul` and
  `causal_conv1d_fwd_cpp`. Both intentional (torch.compiler.disable
  wrappers for our HIP kernel and aiter's op respectively). Zero breaks
  attributable to NorMuon / value-residuals / head-gating.
- **0 recompiles.**

So compile was never the issue. The fp32 NS matmul was.

## Why the IMU-1 paper reported ~3% overhead

IMU-1 used a custom Triton kernel for the 5-step Newton-Schulz. At 430M
the per-param overhead amortizes better and the Triton kernel presumably
routes through fp16 tensor cores on NVIDIA. We arrive at the same
effective cost (3.5% vs their 3%) by simply asking rocBLAS to dispatch
the fp16 kernels instead of fp32. No custom kernel needed.

## How to use

- **Default path (2026-05-07+):** fp16 NS is ON automatically when you
  use `--normuon`. No extra flag needed.
- **Restore Phase 2 behavior (fp32 NS):** pass `--ns-dtype fp32`.
- **Debug:** `scripts/bench_newton_schulz.py` runs the 5-step NS in
  isolation per shape; useful when sanity-checking new param shapes
  or a different hardware target.

## Future work (not needed near-term)

The original Sprint 1.1 plan included Phase C (batched NS across same-
shape params) and Phase D (fused HIP NS kernel). Both were gated
behind "Phase B achieves cost > 7%". Phase B achieved 5.97% cost with
B1 alone and 4.26% with the B4 combo. Neither C nor D ran.

If a future model scales beyond 1B params or uses a different shape
distribution (e.g. much larger FFN), re-measure. Batched NS could add
~10-20% if there's significant shape clustering. HIP NS kernel would
save the 6% Python + neuron-norm + cautious-WD overhead.

## See also

- `imu1_recipe_2026.md` — the recipe we're implementing
- `muon_optimizer_results.md` — earlier Muon cost analysis (pre-NorMuon)
- `../../docs/superpowers/specs/2026-05-07-sprint1.1-normuon-throughput-design.md` — spec
- `../../docs/superpowers/plans/2026-05-07-sprint1.1-normuon-throughput-plan.md` — plan
- `../../docs/perf/sprint1.1-summary.md` — final summary
- `../../docs/perf/normuon-profile-summary.md` — Phase A profile details
- `../../STATUS.md` — Sprint 1 + 1.1 run-by-run results
