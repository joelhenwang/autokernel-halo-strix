---
title: "IMU-1 Recipe: NorMuon + Cautious WD + μP for Sample-Efficient Small LM Pretraining"
domain: training
type: reference
status: active
tags: [imu1, normuon, cautious-wd, mup, small-lm, optimizer, pretraining-recipe, wsd, ema]
paper: "IMU-1: Sample-Efficient Pre-training of Small Language Models (arXiv:2602.02522, 2026-01-25)"
model: "https://huggingface.co/thepowerfuldeez/imu1_base"
related:
  - muon_optimizer_results.md
  - ../architectures/small_lm_arch_interventions_2026.md
  - cpt_best_practices_2026.md
  - ../../docs/research/broad-research-synthesis-2026-05-06.md
---

# IMU-1 Recipe — the New Small-LM Pretraining Default

## What it is

IMU-1 is a 430M-parameter model trained on 72B tokens that **approaches the
benchmark performance of models trained on 56× more data (~4T tokens)**. Its
value to us is not the final model but the **validated recipe** it ships with,
which is directly applicable to Odin (58M OdinHalo, 122M OdinFlat) on
gfx1151 DDP.

Relevance to our project: IMU-1's 430M scale is only 3.5× larger than OdinFlat
and uses similar architectural primitives (transformer decoder, tied embeds).
The recipe transfers without major adaptation.

## Recipe components (all additive in ablations)

### Architectural interventions (cumulative gains)

| Component | Purpose | Source |
|-----------|---------|--------|
| **QK-Norm attention** | Stability; lower activation kurtosis | Henry et al., 2020 |
| **Per-head gated attention** | Expressivity + attention-sink mitigation | Qiu et al., 2025 |
| **Value residual connections** | Better gradient flow | Zhou et al., 2024 |
| **LayerNorm scaling** | Depth-related training pathologies | IMU-1 |

OdinFlat already has QK-Norm (in `NoPECodaAttention`). The other three are net-new
additions we should port.

### Optimization stack

#### NorMuon — normalized Muon with neuron-wise scaling

Standard Muon orthogonalizes momentum matrices; NorMuon additionally normalizes
rows/columns to address post-orthogonalization norm imbalance.

Implementation notes:
- **7 Newton-Schulz iterations** with Polar Express constants (Amsel et al. 2025)
- Dion Triton kernel (Ahn et al. 2025) for orthogonalization efficiency
- ~3% overhead vs AdamW

#### Cautious Weight Decay (CWD)

Selective regularization: apply WD only to parameters whose gradient-weight
alignment satisfies a cautious criterion. Source: Chen et al. 2025.

#### Parameter grouping (critical detail)

| Group | Parameters | Optimizer | LR |
|-------|-----------|-----------|---:|
| 2D matrices | `ndim >= 2` weight matrices | NorMuon | **0.0235** |
| 1D / special | biases, LN gains, tok_embeddings, lm_head | AdamW | **0.007** |

Muon's implicit preconditioning tolerates the much higher LR; the split is
critical — applying Muon to embeddings/lm_head is known to destabilize training.

#### μP parametrization (Yang et al. 2022)

Enables hyperparameter transfer across scales. Train a 30M probe, fit optimal
LR/WD, transfer to 122M/430M without re-tuning.

### Schedule — 3-stage WSD with EMA

```
Stage 1 (Stable):    Warmup → peak LR → stable training on broad-mix data
Stage 2 (Stable):    continued stable on quality-upsampled data (math/code)
Stage 3 (Decay):     linear decay to 0 on highest-quality mix, save checkpoints every 5k steps

Post-hoc EMA:        β=0.8 over final 10 checkpoints (collected during decay)
```

WSD with 20% decay fraction matches cosine performance while allowing
checkpoint re-use for multi-stage training. EMA adds **+0.014 average benchmark
score** over the best single checkpoint.

## Measured ablation gains (IMU-1 Table 6)

| Change | Relative loss improvement |
|--------|:-:|
| AdamW → NorMuon | **−2.88%** |
| + Cautious Weight Decay | −0.97% additional |
| **NorMuon + CWD total over AdamW** | **−3.85%** |

Three-stage training progression (avg benchmark):
- Stage 1 only: 0.461
- + Stage 2: 0.522
- + Stage 3 decay: 0.560
- + EMA: 0.574

## Throughput cost

- NorMuon vs AdamW: **~3% per-step overhead** (7 Newton-Schulz iterations)
- Peak throughput reached: 1.8M tok/s (large cluster)
- MFU: **26-36%** depending on stage (drops at longer context)

At our 2× Strix Halo DDP with ~40K tok/s AdamW baseline, expect ~38K tok/s with
NorMuon+CWD. ~5% wall-clock cost for ~4% loss improvement is the trade.

## Adoption path for Odin

### Minimal port (1-2 days)

1. **Add `halo_training/normuon.py`**: new optimizer combining NorMuon +
   Cautious WD. Reference implementation uses PyTorch matrix primitives;
   our existing Muon code in `halo_training/muon.py` is ~80% of the work.
2. **Expose `--normuon` flag in `halo_training/cli.py`** and `scripts/train_ddp.py`.
3. **Parameter grouping logic**: 2D → NorMuon, 1D/embed/lm_head → AdamW.
   Reuse `halo_training/muon.py::split_params_for_muon` as the splitter.
4. **LR schedule**: 0.0235 for NorMuon group, 0.007 for AdamW group.
   Single `--lr` CLI arg with internal scaling, or two flags.

### Architecture additions (1-2 hrs each)

- **Value residuals**: simple `x = x + self.attn(...)` → `x = x + v_res(self.attn(...))`
  where `v_res = nn.Linear(d, d, bias=False)` + init small. Port into
  `HyPEShortConvBlock` and `NoPEGQABlock` forward paths.
- **LayerNorm scaling**: scale the learnable γ at layer index i by `f(i/depth)`.
  Usually `γ_i ← γ_i × (1 / sqrt(layer_idx + 1))` or similar.
- **Per-head gated attention**: gate per-head output with sigmoid of a
  per-head learned vector: `attn_out = head_gates * attn_out`.

### μP port (3-5 hrs)

- Depends on careful init rewriting. Defer to a dedicated experiment after
  NorMuon is landed.

### Verification plan

1. Train OdinFlat from scratch on wikitext-103 baseline (current): AdamW run
   → record final loss.
2. Same config with NorMuon + CWD: measure delta.
3. Same + architectural additions: measure delta.
4. Expected: cumulative **~5% loss improvement** at ~5% wall-clock cost.

## Caveats

- **Tiny models (<50M) may not see the full gain**: Muon-family optimizers are
  known to help more as model size grows. At 58M (OdinHalo unique) expect
  ~half the headline gain; at 122M OdinFlat expect closer to full gain.
- **Need careful LR tuning on first run**: Muon's higher LR tolerance can
  backfire at our scale. Start with 0.015 (not 0.0235) and back off on
  instability.
- **NorMuon's Newton-Schulz iteration** interacts with our existing
  `disable_hip_backward()` logic in `compile_zones`. Need to verify it doesn't
  regress under `TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`.

## Comparison to our existing optimizer knowledge

From `knowledge/training/muon_optimizer_results.md`:
- Basic Muon at 80M: 2× token efficiency vs AdamW but **3.5× slower step**
- We measured Muon as NOT worth the throughput hit for pretraining

IMU-1 finding changes this:
- **NorMuon has only ~3% step overhead, not 3.5×** — the speed gap closes dramatically
- Net effect: NorMuon is faster in wall-clock AND gives better loss

The change: (1) neuron-wise normalization (NorMuon vs plain Muon), (2) Triton
kernel (Dion), (3) Polar Express constants for Newton-Schulz. Together these
turn Muon from "slow but good" into "fast and good."

## See also

- `knowledge/training/muon_optimizer_results.md` — earlier Muon investigation
- `knowledge/architectures/small_lm_arch_interventions_2026.md` — architecture
  tricks in the IMU-1 recipe, consolidated with other 2025-2026 findings
- `docs/research/broad-research-synthesis-2026-05-06.md` Part 2 — optimizer
  family summary table
