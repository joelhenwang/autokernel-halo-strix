---
title: "Knowledge Graph Documentation Reorganization"
domain: project
type: spec
status: active
related:
  - KNOWLEDGE_GRAPH.md
  - CLAUDE.md
  - REPORT.md
tags: [%documentation, %knowledge-graph, %organization, %frontmatter]
---

# Knowledge Graph Documentation Reorganization

## Problem

The repo has 100 .md files (~27,600 lines) across 9 directories with:
- No master index connecting them
- Duplicate files (`llm_engineer_agent/` mirrors `mad_llm_scientist/`)
- Triple coverage (DDP setup in 3 places)
- Overloaded CLAUDE.md (391 lines mixing instructions, results, and reference)
- Orphaned stubs (`MatFormer.md`, `PLE.md` at root)
- Flat `knowledge/` directory (14 files, no sub-organization)

Both human and AI assistant need a way to quickly navigate to the right document without loading everything into context.

## Design

### 1. Frontmatter Schema

Every `.md` file gets a YAML frontmatter block:

```yaml
---
title: "DDP Setup Guide"
domain: training             # hardware | kernels | training | architectures | design-specs | operations | project
type: guide                  # guide | reference | results | spec | plan | review | cookbook | agent-instructions
status: active               # active | stale | archived
related:
  - knowledge/training/rccl_build_gfx1151_guide.md
  - scripts/ddp_env.sh
tags: [%ddp, %thunderbolt4, %gloo, %2-machine, %strix-halo]
---
```

**Fields:**
- **domain** — which cluster the doc belongs to (routing)
- **type** — what kind of document (filtering)
- **status** — lifecycle state (active, stale, archived)
- **related** — explicit graph edges to other files (navigation)
- **tags** — `%`-prefixed keywords for grep-based discovery (`grep "%ddp"`)

**Tag prefix:** `%` chosen because it has no YAML quoting issues and no regex special meaning, making `grep "%tagname"` unambiguous across the repo.

**Domain definitions:**
- **hardware** — Strix Halo specs, ROCm, gfx1151 quirks
- **kernels** — HIP optimization results, external kernel libraries, fused ops, anti-patterns
- **training** — training stack, DDP, distributed, Muon optimizer, pipelines, RCCL
- **architectures** — hypothesis plans, buildout results, rankings, model-specific results
- **design-specs** — dated design decisions (already-made choices)
- **operations** — commands, checklists, runbooks, monitoring guides
- **project** — README, CHANGELOG, REPORT, blog

**Type definitions:**
- **guide** — how to do something (DDP setup, RCCL build, installs)
- **reference** — factual lookup (hardware specs, workload guidance)
- **results** — measured outcomes (benchmarks, training runs, ablations)
- **spec** — design decision document
- **plan** — architecture hypothesis
- **review** — critique of a plan
- **cookbook** — reusable implementation patterns
- **agent-instructions** — CLAUDE.md, program.md, kernelbench program

### 2. Directory Restructure

```
autokernel-halo-strix/
├── CLAUDE.md                          # slimmed to ~120 lines, routing doc
├── KNOWLEDGE_GRAPH.md                 # NEW — master index
├── README.md                          # stays
├── REPORT.md                          # stays
├── CHANGELOG.md                       # stays
├── program.md                         # stays
│
├── knowledge/
│   ├── hardware/
│   │   ├── amd_rdna35_strix_halo.md   # moved from knowledge/
│   │   └── workload_guidance.md       # moved from knowledge/
│   ├── kernels/
│   │   ├── kernel_benchmarks.md       # NEW — extracted from CLAUDE.md
│   │   ├── external_kernels.md        # NEW — extracted from CLAUDE.md
│   │   ├── fused_kernels.md           # NEW — extracted from CLAUDE.md
│   │   ├── mHC_MoE_Engram_optimizations.md  # moved
│   │   ├── backward_pass_optimization_research.md  # moved
│   │   └── backward_pass_optimization_results.md   # moved
│   ├── training/
│   │   ├── ddp_setup_guide.md         # merged with ddp_dual_strix_halo_tb4.md
│   │   ├── rccl_build_gfx1151_guide.md  # moved
│   │   ├── muon_optimizer_results.md  # moved
│   │   ├── argus_prime_results.md     # moved
│   │   └── training_antipatterns.md   # NEW — extracted from CLAUDE.md
│   ├── architectures/
│   │   ├── hypothesis_buildout_results.md  # moved
│   │   ├── Estimation_Hypothesis_Ranking.md  # moved
│   │   ├── reliable_small_lm_insights.md  # moved
│   │   └── ple_ablation_results.md    # moved
│   └── INDEX.md                       # NEW — knowledge/ sub-index
│
├── docs/                              # untouched internally
│   ├── superpowers/specs/             # 13 dated specs (+ this one)
│   ├── halo-strix-apu/               # 6 operational guides
│   ├── reliable_small_language_model_training_guide.md
│   ├── possible_techniques_bwd_improv.md
│   └── blog-post.md
│
├── mad_llm_scientist/                 # untouched
│   ├── CLAUDE.md
│   ├── COOKBOOK.md
│   ├── EVALUATION_GUIDE.md
│   ├── BPB_MFU_ANALYSIS.md
│   ├── PRETRAINING_CONCERNS.md
│   ├── plans/                         # 30 hypothesis files
│   └── reviews/
│
├── INSTALL_CAUSAL_CONV1D.md           # stays
├── INSTALL_MAMBA_SSM.md               # stays
├── INSTALL_AITER.md                   # stays
│
├── experiments/                       # stays
├── kernelbench/                       # stays
├── external/                          # stays
└── halo_training/                     # stays
```

### 3. Deletions

| File/Directory | Reason |
|---------------|--------|
| `llm_engineer_agent/` (entire dir, 4 files) | Duplicates of `mad_llm_scientist/` |
| `MatFormer.md` (root, 2 lines) | Stub, covered by `2026-04-10-ple-matformer-design.md` |
| `PLE.md` (root, 25 lines) | Stub, covered by `2026-04-10-ple-matformer-design.md` |
| `knowledge/ddp_dual_strix_halo_tb4.md` | Merged into `knowledge/training/ddp_setup_guide.md` |

### 4. New Files Created

| File | Source | Content |
|------|--------|---------|
| `KNOWLEDGE_GRAPH.md` | New | Master index, domain-grouped, all docs listed with 1-line descriptions |
| `knowledge/INDEX.md` | New | Sub-index for knowledge/ subdirectories |
| `knowledge/kernels/kernel_benchmarks.md` | Extracted from CLAUDE.md | Kernel speed tables (6-16x, 1.5-4x, ~1x, <0.3x categories) |
| `knowledge/kernels/external_kernels.md` | Extracted from CLAUDE.md | External kernel library table + attention backend selection + install notes |
| `knowledge/kernels/fused_kernels.md` | Extracted from CLAUDE.md | 4 fused kernel results (mhc_sinkhorn, engram, ple, chunked CE) |
| `knowledge/training/training_antipatterns.md` | Extracted from CLAUDE.md | Anti-patterns list + winning patterns + ROCm dev reference |

### 5. CLAUDE.md Restructure

Slimmed from 391 lines to ~120 lines. Keeps:

- **Project Overview** (5 lines)
- **Quick Hardware Reference** (3 lines: GPU, CPU, memory)
- **Commands** (~20 lines: setup, kernels, training, DDP)
- **Workflow** (10 lines: Phase A→D)
- **Key Constraints** (~15 lines: things to always remember)
  - No MFMA, unified memory, wave32, fp16 not bf16
  - autokernel breaks d≤256
  - Don't put matmuls in HIP kernels
  - Don't break autokernel FusedQKV+RoPE pattern
  - torch.compile model only, never optimizer
  - Check train_log.jsonl, don't rely on SSH stdout
- **Key Directories** (5 lines: pointers)
- **Where to Find Everything Else** (~10 lines: pointers to KNOWLEDGE_GRAPH.md and knowledge/ subdirs)

Everything else (result tables, architecture comparisons, detailed anti-pattern explanations, rocBLAS section, ROCm dev reference) moves to knowledge/ files.

### 6. KNOWLEDGE_GRAPH.md Structure

Domain-grouped index with one line per file:

```
## Hardware
- [AMD RDNA 3.5 Strix Halo Reference](knowledge/hardware/amd_rdna35_strix_halo.md) — gfx1151 full specs

## Kernels
- [Kernel Benchmarks](knowledge/kernels/kernel_benchmarks.md) — HIP kernel speedups vs PyTorch
...

## Training & Distributed
...

## Architectures
...

## Design Specs
...

## Operations
...

## Project
...

## Research
...
```

Each entry is a link + 1-line description. The file serves as the first thing to read when navigating the repo.

### 7. Frontmatter on Existing Files

All ~90 remaining .md files get frontmatter added. The `related` field creates the graph edges between documents. Tags use `%` prefix for grep-based discovery.

Priority files for rich `related` fields (high connectivity):
- `CLAUDE.md` → points to all knowledge/ subdirs
- `knowledge/training/ddp_setup_guide.md` → RCCL build, argus_prime_results, ddp_env.sh
- `knowledge/hardware/amd_rdna35_strix_halo.md` → workload_guidance, kernel_benchmarks
- `mad_llm_scientist/COOKBOOK.md` → all 30 plans, EVALUATION_GUIDE
- `REPORT.md` → kernel_benchmarks, hypothesis_buildout_results, argus_prime_results

### 8. Net Change Summary

| Metric | Before | After |
|--------|--------|-------|
| Total .md files | 100 | ~95 |
| Files with frontmatter | 0 | ~95 |
| Duplicate files | 4 | 0 |
| CLAUDE.md lines | 391 | ~120 |
| knowledge/ subdirectories | 0 | 4 |
| Master index files | 0 | 2 (KNOWLEDGE_GRAPH.md + knowledge/INDEX.md) |
| Orphaned stubs | 2 | 0 |
| Triple-covered topics | 1 (DDP) | 0 |

### 9. Implementation Notes

- Use `git mv` for all moves to preserve history
- Add frontmatter to files in batches by domain (hardware first, then kernels, etc.)
- Update any internal cross-references that break from file moves
- Update MEMORY.md entries that point to old paths
- The `mad_llm_scientist/plans/` files (30) get minimal frontmatter (title, domain, type, status, tags) — no need for rich `related` fields on hypotheses that were never built
