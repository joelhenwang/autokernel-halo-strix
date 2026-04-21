---
title: "CLIMB + Self-Improving Data Mixture Pipeline"
domain: training
type: design
status: active
tags: [climb, data-mixture, self-improving, proxy-search, quality-filtering]
related:
  - ../architectures/chimera_halo_design.md
  - muon_optimizer_results.md
---

# CLIMB + Self-Improving Data Mixture Pipeline

**Date:** 2026-04-21
**Scripts:** `scripts/datamix/phase0-5`
**Status:** Verified end-to-end on 10K FineWeb-Edu test run

## What It Does

Combines two papers into a modular data optimization pipeline:
- **CLIMB (NVIDIA, 2504.13161)**: Cluster corpus, search optimal mixture weights via proxy model
- **Self-Improving Pretraining (Meta, 2601.21343)**: Score/filter data quality via LLM judge

CLIMB picks the right **proportions**. Self-Improving cleans the **content**. No overlap.

## Pipeline

```
Phase 0: Stream & sample docs from HF dataset (CPU, ~30 min)
Phase 1: Embed with sentence-transformers (CPU, ~17 min)
Phase 2: FAISS K-means clustering (CPU, ~2 min)
Phase 3: CLIMB proxy model search (GPU, ~40 min)
Phase 4: API quality scoring + classifier (CPU + API, ~15 min)
Phase 5: Assemble final .bin dataset (CPU, ~5 min)
```

Total: ~1.5 hours + ~$1.30 API cost.

## Model-Agnostic Proxy Search

Any `(model_path, class_name)` pair works via `load_model_from_file()`. Tested with ChimeraHalo (94M). CLIMB paper shows 62M proxy loses only 0.3% vs 350M — architecture match matters more than proxy size.

## Test Run Results (10K FineWeb-Edu, 8 clusters, ChimeraHalo proxy)

| Trial | Source | Best val_loss | Notes |
|-------|--------|--------------|-------|
| Random (8 trials) | Dirichlet sampling | 10.47 | Broad exploration |
| **Surrogate (4 trials)** | **LightGBM-guided** | **9.86** | **6.2% improvement** |

Best mixture weights:
| Cluster | Topic | Weight |
|---------|-------|--------|
| 0 | Education/learning | 39.1% |
| 2 | General science | 20.8% |
| 5 | Health/maternal | 15.0% |
| 7 | News/opinion | 8.3% |
| 3 | Wikipedia | 6.4% |
| 1 | Science/environment | 5.2% |
| 4 | Biodiversity/library | 4.5% |
| 6 | Academic journals | 0.7% |

Educational content dominates — consistent with FineWeb-Edu being education-filtered.

## Usage

```bash
# Full pipeline (CPU phases local, GPU phase remote)
python scripts/datamix/phase0_sample.py --dataset allenai/dolma --max-docs 2000000
python scripts/datamix/phase1_embed.py
python scripts/datamix/phase2_cluster.py --n-clusters 16
python scripts/datamix/phase3_proxy_search.py --model models/chimera_halo.py --class-name ChimeraHalo \
    --val-dataset path/to/val.bin --n-rounds 3 --trials-per-round 20,10,5
python scripts/datamix/phase4_score.py --api-provider anthropic --samples-per-cluster 1000
python scripts/datamix/phase5_assemble.py --target-tokens 1000000000

# Train on assembled dataset
python -m halo_training --model models/chimera_halo.py --class-name ChimeraHalo \
    --dataset datasets/datamix_state/phase5_final/train.bin --compile --optimize-kernels --muon
```

## Key Design Decisions

- **all-MiniLM-L6-v2** for embeddings (384d, 6x faster than stella_en_400M, sufficient for K=16 clustering)
- **K=16 clusters** default (CLIMB paper sweet spot, enough granularity without overfitting)
- **LightGBM surrogate** maps mixture weights → val_loss (94% Spearman correlation per paper)
- **Sample-then-classify** for quality scoring: 1000 API calls per cluster → logistic regression → predict all (~$1.30 total)
- **Pre-mixed .bin output** loads directly via existing `BabyLMDataset` — zero trainer changes needed
- **Resumable**: all phases save state to `pipeline_state.json`

## Dependencies

```
sentence-transformers  # Phase 1
faiss-cpu             # Phase 2
lightgbm              # Phase 3
anthropic / openai    # Phase 4
scikit-learn          # Phase 4
```

## Gotchas

- `profile.py` in repo root shadows Python stdlib `profile` module → rename before running phases that import sentence_transformers/torch._dynamo
- ChimeraHaloMini (vocab=1000) can't be used as proxy on real data (tokens > 1000 → GPU crash). Use ChimeraHalo (vocab=50257)
- sentence_transformers >= 5.0 pulls in torchcodec which breaks on ROCm. Pin to `<4`
