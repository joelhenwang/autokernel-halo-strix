# ARCHIVED: Mad LLM Scientist

> **FROZEN: 2026-04-17** — This directory is archived.
> The 55 plans represent the ideation phase of the project (March-April 2026).
> Active architecture work has moved to `docs/superpowers/specs/` and `knowledge/architectures/`.

## What moved where

| Original | New canonical location |
|----------|----------------------|
| `COOKBOOK.md` | `knowledge/architectures/cookbook.md` |
| `EVALUATION_GUIDE.md` | `knowledge/training/evaluation_guide.md` |
| `BPB_MFU_ANALYSIS.md` | `knowledge/training/bpb_mfu_analysis.md` |
| `PRETRAINING_CONCERNS.md` | `knowledge/training/pretraining_concerns.md` |

The 55 plan files in `plans/` are historical — they were written by the "Mad LLM Scientist" persona
as creative hypothesis briefs. Only ~8 were implemented as actual models. The rest remain as
a reference catalog of ideas.

For current architecture designs, see:
- `docs/superpowers/specs/` — design specs with implementation details
- `knowledge/architectures/` — architecture rankings, results, design docs

For hardware constraints and kernel optimization, see:
- `CONSTRAINTS.md` (root) — all hard constraints
- `knowledge/kernels/hip_optimization_playbook.md` — full optimization playbook
- `knowledge/hardware/amd_rdna35_strix_halo.md` — full hardware reference