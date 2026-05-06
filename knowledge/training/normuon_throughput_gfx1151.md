---
title: "NorMuon Throughput on gfx1151: Empirical Results + Optimization Plan"
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

Empirical results from Sprint 1 Phase 3 (2026-05-06) plus optimization
plan from Sprint 1.1 (2026-05-07). Written against the backdrop of
`imu1_recipe_2026.md`'s claim that "NorMuon has only ~3% step overhead."
Reality at our scale is different.

## Observed throughput cost

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
| Run 2 (full recipe) | NorMuon lr_2d=5e-3 lr_1d=8e-4 | + free wins + value residuals + head gating | **32,478** | **17.8%** |

**Key disagreement with IMU-1 paper:** NorMuon's step overhead is
**~13-16% at our scale**, not the ~3% they claim. Reasons:

1. **IMU-1 uses Triton** kernel for Newton-Schulz. We use PyTorch
   ops because gfx1151 doesn't run upstream Triton reliably. Each NS
   iteration is 3 matmuls as separate kernel launches; 5 iterations ×
   ~60 2D matrices × 3 matmuls = ~900 kernel launches per optimizer step.

2. **IMU-1 is at 430M with ~3× our param count**. The per-param Python
   overhead amortizes better at higher parameter counts; at 122M the
   Python dispatch is a larger fraction of step time.

3. **gfx1151 has no MFMA**. Matmul throughput is ~60% of rocBLAS peak
   with the small matrix sizes NS runs (768×768 and similar). The
   quadratic-in-matmul NS suffers disproportionately.

4. **Not compiled**. The NS function explicitly disables torch.compile
   (per `halo_training/muon.py` comment — shape-specific recompile
   blowup). Means we miss Inductor fusion on the 5-iter path.

## Quality gain that justifies the cost (Run 2 results)

| Metric | Run 2 (NorMuon full) | Baseline (AdamW) | Δ |
|---|---:|---:|---:|
| Final train loss | **4.4736** | 4.7975 | **−6.8%** |
| wikitext_val BPB | **1.893** | 1.9214 | −1.5% |
| gpt_small_val BPB | 2.8327 | 2.8634 | −1.1% |
| stem_crawl_val BPB | 3.4314 | 3.5361 | −3.0% |
| dolma_val BPB | 3.0883 | 3.1094 | −0.7% |
| avg BPB | 2.810 | 2.861 | −1.8% |
| Memory | 10.1 GB | 10.3 GB | −1.9% |

The full recipe (NorMuon + value residuals + head gating) delivers
−6.8% training loss and beats baseline on ALL 4 held-out domains.
Quality improvement is real; the only gate failure is throughput.

## Where the cost goes (informed guesses pending Phase A profile)

The Sprint 1.1 profile will give us concrete numbers; prior-art guesses:

| Likely cost center | Guess | Evidence |
|---|---:|---|
| Newton-Schulz matmuls (5 iter × 3 matmuls × ~60 params) | 60-70% | 900 small-matrix launches dominate on gfx1151 |
| Python per-param for-loop overhead | 10-15% | `for p in group["params"]: ...` with ~60 iterations |
| Neuron-wise normalization (L2 norm per row + div) | 5-10% | ~60 reductions + divisions |
| Cautious-WD sign mask + elementwise select | 5-10% | Per-param sign compute + elementwise |
| Momentum buffer copy / mul / add | 5-10% | Standard optimizer overhead |

Sprint 1.1 Phase A will run `scripts/profile_step.py` on three configs
(AdamW, NorMuon-only, full recipe) to attribute this concretely. See
`docs/superpowers/specs/2026-05-07-sprint1.1-normuon-throughput-design.md`.

## Optimization candidates (Sprint 1.1 design)

### Cheap wins (Phase B)

- **`ns_dtype=torch.float16`** — gfx1151 has native fp16 WMMA; NS in fp16
  should roughly 1.5-2× faster. Already supported via `ns_dtype` kwarg in
  `halo_training/muon.py::zeropower_via_polar_express`. Risk: numerical
  stability at edges of orthogonality.

- **Size-gated neuron-norm** (`neuron_norm_min_dim=512`) — skip neuron-wise
  norm for matrices where `min(rows, cols) < 512`. In OdinFlat this only
  skips the factorized embedding projection (256×768), preserves all
  attention + FFN. Saves the reduction + division for that one param.

- **Disable cautious WD** — removes the sign-mask elementwise op per
  param. Quality risk is small (CWD is a small refinement); reverts to
  standard decoupled WD.

### Structural wins (Phase C)

- **Batched Newton-Schulz** — stack same-shape 2D parameters into a 3D
  tensor and run NS once on the batch. OdinFlat has 12 identical
  HyPEShortConv SwiGLU inputs, 12 identical SwiGLU outputs, 12 identical
  gate matrices. Reduces ~60 per-param NS invocations to ~10 batched ones.
  Algorithmically equivalent.

- **Reduce NS iterations** 5 → 4 or 3. Linear cost reduction. Quality
  risk at 3 iters (orthogonality err grows from 0.08 to 0.25).

### Heavy lift (Phase D, conditional)

- **Fused HIP Newton-Schulz kernel** — if profile shows matmuls dominate,
  write a single HIP kernel that does the 5-iter NS in one launch with
  half2 loads. Matches the pattern used for `fused_rope_gate_mul`. Only
  attempted if Phase B + C don't hit the <10% cost target.

## Interaction with torch.compile

The NorMuon step is NOT compiled (explicit design choice — shape-specific
recompile blowup). But the MODEL forward (which includes value_residuals
and head_gating) IS compiled via `compile_zones()`. Adding value_residuals
and head_gating in Run 2 cost +2 pp over Probe C; likely a combination of:

- Value residual `v = v + v_res_scale * v_prev` adds a residual path in
  the compiled graph, potentially introducing a graph break at the
  layer boundary
- Head gating `attn_out *= sigmoid(head_gate).view(1, n, 1, 1)` is a
  simple broadcast multiply; probably cheap in isolation

Phase A.5 will analyze `TORCH_LOGS=graph_breaks,recompiles` output to
confirm or refute.

## Revision of IMU-1 recipe expectations at our scale

`knowledge/training/imu1_recipe_2026.md` predicts "~3% step overhead" for
NorMuon. Our measurements show **~13-16% for NorMuon alone on gfx1151 /
122M / PyTorch ops**. Expected delta after Sprint 1.1 optimizations:

| Target | Path |
|--------|------|
| 10% cost | Phase B + maybe Phase C1 (batched NS) |
| 7% cost (original gate) | Phase C1 + C2 (reduce iters) or Phase D (HIP) |
| 3% cost (paper claim) | Would require custom Triton/HIP kernel matching their implementation |

Pragmatic goal: ≤10% cost at <1% quality regression. Documented trade-off.

## See also

- `imu1_recipe_2026.md` — the recipe we're implementing (now cross-referenced)
- `muon_optimizer_results.md` — earlier Muon cost analysis (pre-NorMuon)
- `../../docs/superpowers/specs/2026-05-07-sprint1.1-normuon-throughput-design.md` — the optimization spec
- `../../STATUS.md` — Sprint 1 Phase 3 run-by-run results
