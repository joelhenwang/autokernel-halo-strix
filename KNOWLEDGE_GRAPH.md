---
title: "Knowledge Graph"
domain: project
type: reference
status: active
tags: [%index, %navigation, %knowledge-graph]
---

# Knowledge Graph

> Master index for all documentation. Each file has YAML frontmatter with domain, type, status, tags, and related docs.
> Use `grep "%tag-name"` across the repo for tag-based discovery.

---

## Hardware

- [AMD RDNA 3.5 Strix Halo Reference](knowledge/hardware/amd_rdna35_strix_halo.md) — gfx1151 full specs, wave32, LDS, rocBLAS, profiling, compilation
- [Workload Guidance](knowledge/hardware/workload_guidance.md) — optimization strategy per workload type

## Kernels

- [Kernel Benchmarks](knowledge/kernels/kernel_benchmarks.md) — all HIP kernel speedups vs PyTorch, training results, architecture comparisons
- [External Kernel Libraries](knowledge/kernels/external_kernels.md) — causal-conv1d, mamba-ssm, flash_attn, FLA, scattermoe, attention backend selection
- [Fused Kernels](knowledge/kernels/fused_kernels.md) — mhc_sinkhorn (28.5x), engram (7.4x), ple_gate, chunked CE
- [mHC/MoE/Engram Optimizations](knowledge/kernels/mHC_MoE_Engram_optimizations.md) — Engram fusion results, ScatterMoE, FLA updates
- [Backward Pass Research](knowledge/kernels/backward_pass_optimization_research.md) — approaches surveyed (FP8, sampled softmax, IO-aware)
- [Backward Pass Results](knowledge/kernels/backward_pass_optimization_results.md) — empirical measurements from backward fusion

## Training & Distributed

- [DDP Setup Guide](knowledge/training/ddp_setup_guide.md) — 2x Strix Halo via TB4, 35K tok/s, full walkthrough
- [RCCL Build for gfx1151](knowledge/training/rccl_build_gfx1151_guide.md) — source build with patches for gfx1151
- [Muon Optimizer Results](knowledge/training/muon_optimizer_results.md) — 2x token efficiency, 50% less optimizer memory
- [ARGUS-PRIME Results](knowledge/training/argus_prime_results.md) — 18K tok/s single, 35K DDP, WikiText-103 CPT, Common Crawl
- [CLIMB Data Mixture Pipeline](knowledge/training/climb_data_mixture.md) — CLIMB + Self-Improving: cluster, proxy search, quality filter, assemble
- [Training Anti-patterns & Patterns](knowledge/training/training_antipatterns.md) — optimization patterns, anti-patterns, rocBLAS, ROCm dev reference

## Architectures

- [Hypothesis Buildout Results](knowledge/architectures/hypothesis_buildout_results.md) — 13 architectures trained at 170M, AMADEUS wins quality
- [Hypothesis Ranking](knowledge/architectures/Estimation_Hypothesis_Ranking.md) — all 30 hypotheses ranked by estimated MFU and feasibility
- [Reliable Small LM Insights](knowledge/architectures/reliable_small_lm_insights.md) — 6 gaps, Liquid AI talk, GPT-X2, InstructLM, Baguettotron, whiff-mamba2 analyses
- [PLE Ablation](knowledge/architectures/ple_ablation_results.md) — Path A wins quality, MatFormer free, B/AB no benefit
- [CHIMERA-HALO Design](knowledge/architectures/chimera_halo_design.md) — factorized embeddings + Parcae loop + LFM2 hybrid + XSA, 94M/158M
- [FENRIR-HALO Design](docs/superpowers/specs/2026-04-21-fenrir-halo-design.md) — clean-sheet Parcae loop, d=640, 80M/160M, DDP-ready, targets Portimbria-150M
- [**VIDAR-HALO Design**](docs/superpowers/specs/2026-05-03-vidar-halo-design.md) — **ACTIVE**: 47.5M/95M, d=768, 4L×2iter, no momentum, custom 32K tokenizer, WSD+z-loss, ~25K tok/s target
- [TYR-HALO Design](docs/superpowers/specs/2026-04-29-tyr-halo-design.md) — 58M/115M, MoDA+mHC+MTP+DraftHeads+DS2D+CTG, 12 papers, 4-phase inference
- [BALDR-HALO Design](docs/superpowers/specs/2026-04-29-baldr-halo-design.md) — 118M flat hybrid, 12 layers, MoDA+XSA+MTP, no loop/mHC, max speed target
- [TYR-HALO Theory & Implementation Guide](docs/guides/tyr-halo-theory-and-implementation.md) — full theory, PyTorch impl, kernel/compile/data optimizations
- [TYR-HALO RTX 4060 Ti Training Guide](docs/guides/tyr-halo-rtx4060ti-training.md) — NVIDIA bf16 training, memory budget, compile strategy
- [Architecture Plans](mad_llm_scientist/plans/) — 30+ hypothesis designs (AMADEUS, TEMPEST, ARGUS-PRIME, JORMUNGANDR, etc.)
- [JORMUNGANDR Plan](mad_llm_scientist/plans/JORMUNGANDR.md) — Parcae-stable looped ShortConv, staged activation, Poisson depth
- [COOKBOOK](mad_llm_scientist/COOKBOOK.md) — shared modules, implementation patterns for all architectures

## Design Specs (Decisions Made)

- [Halo Training Stack](docs/superpowers/specs/2026-04-08-halo-training-stack-design.md) — Mode A/B architecture
- [Autokernel Library API](docs/superpowers/specs/2026-04-08-autokernel-library-api-design.md) — pattern matching + kernel replacement
- [torch.compile Custom Ops](docs/superpowers/specs/2026-04-08-torch-compile-custom-ops-design.md) — HIP ops registration for compile
- [Adaptive LM Head](docs/superpowers/specs/2026-04-09-adaptive-lm-head-design.md) — adaptive softmax experiments
- [PLE + MatFormer](docs/superpowers/specs/2026-04-10-ple-matformer-design.md) — parametric layer ensemble ablation
- [External Kernels Integration](docs/superpowers/specs/2026-04-10-wire-external-kernels-design.md) — causal-conv1d, mamba-ssm wiring
- [Update Hypotheses](docs/superpowers/specs/2026-04-10-update-hypotheses-design.md) — mid-training architecture updates
- [Training Pipeline Optimization](docs/superpowers/specs/2026-04-10-training-pipeline-optimization-design.md) — optimizer groups, schedules
- [aiter/rocBLAS Optimization](docs/superpowers/specs/2026-04-10-aiter-rocblas-optimization-design.md) — BLAS tuning on gfx1151
- [Training Evolution](docs/superpowers/specs/2026-04-10-training-evolution-design.md) — 5-stage screening pipeline
- [Optimization Libraries](docs/superpowers/specs/2026-04-10-optimization-libraries-design.md) — external library selection
- [Backward Pass Optimization](docs/superpowers/specs/2026-04-10-backward-pass-optimization-design.md) — backward fusion, chunked CE
- [Compile-Optimized Griffin](docs/superpowers/specs/2026-04-12-compile-optimized-griffin-design.md) — FusedGriffinBlock, 3.52x improvement
- [Knowledge Graph Reorg](docs/superpowers/specs/2026-04-15-knowledge-graph-reorg-design.md) — this reorganization
- [Training Monitor](docs/superpowers/specs/2026-04-15-training-monitor-design.md) — live dashboard + checkpoint analyzer + callbacks
- [JORMUNGANDR-HALO](docs/superpowers/specs/2026-04-16-jormungandr-halo-design.md) — L2-resident d=512 core loop, 43K tok/s
- [FENRIR-HALO](docs/superpowers/specs/2026-04-21-fenrir-halo-design.md) — clean-sheet Parcae, d=640, 80M/160M
- [**TYR-HALO**](docs/superpowers/specs/2026-04-29-tyr-halo-design.md) — **ACTIVE**: MoDA+mHC+MTP, 58M/115M, 12-paper synthesis
- [**BALDR-HALO**](docs/superpowers/specs/2026-04-29-baldr-halo-design.md) — **ACTIVE**: flat 118M, MoDA+XSA+MTP, racing TYR-HALO on stem-crawl-solo

## Tokenizer

- [Vidar-32K BPE](tokenizers/vidar-32k/tokenizer.json) — custom 32K tokenizer trained on dolma-10b, -12.3% tokens vs GPT-2, -33% on code
- [Tokenizer Training Script](scripts/train_tokenizer.py) — HuggingFace `tokenizers` BPE trainer
- [Pretokenize Script](scripts/pretokenize.py) — multiprocessing + sharding, --workers N, --shard-id/--num-shards

## Alignment & Post-Training

- [Instruct Alignment Survey](knowledge/training/instruct_alignment_techniques_2025_2026.md) — ORPO, SimPO, KTO, EGGROLL ES, SmolLM recipe, Magpie data
- [Alignment Implementation](halo_training/alignment.py) — alignment trainer code

## Operations

- [Platform & ROCm Setup](docs/halo-strix-apu/01_platform_and_rocm.md) — ROCm 7.12, GPU detection, environment validation
- [Performance Analysis](docs/halo-strix-apu/02_training_performance_analysis.md) — profiling methodology, throughput measurement
- [CUDA/OcuLink Strategy](docs/halo-strix-apu/03_cuda_oculink_strategy.md) — multi-machine connectivity planning
- [Two-Machine DDP](docs/halo-strix-apu/04_two_machine_distributed_training.md) — operational DDP guide
- [100B Token Runbook](docs/halo-strix-apu/05_100b_token_runbook.md) — step-by-step checklist for large runs
- [Commands & Checklists](docs/halo-strix-apu/06_commands_and_checklists.md) — reference commands, troubleshooting
- [Install: causal-conv1d](INSTALL_CAUSAL_CONV1D.md) | [mamba-ssm](INSTALL_MAMBA_SSM.md) | [aiter](INSTALL_AITER.md)

## Project

- [README](README.md) — project overview, quick start
- [REPORT](REPORT.md) — canonical results summary (all kernel speeds, training results, architecture comparisons)
- [CHANGELOG](CHANGELOG.md) — version history
- [Blog Post](docs/blog-post.md) — narrative writeup

## Research

- [Paper Deep-Dive May 2026](knowledge/architectures/paper_deep_dive_2026_05.md) — 9 papers: Hyperloop, TRM, HRM, TIDE, GenDistill/KDA, InfoMamba, Embed-MTP, DSKD-KQ, Sessa
- [Mad LLM Scientist](mad_llm_scientist/CLAUDE.md) — researcher agent guidance, creativity constraints, hardware truth
- [COOKBOOK](mad_llm_scientist/COOKBOOK.md) — universal implementation recipe, shared module library
- [Evaluation Guide](mad_llm_scientist/EVALUATION_GUIDE.md) — benchmarks, methodology, checkpoint selection
- [BPB/MFU Analysis](mad_llm_scientist/BPB_MFU_ANALYSIS.md) — bits-per-byte and MFU framework
- [Pretraining Concerns](mad_llm_scientist/PRETRAINING_CONCERNS.md) — token budgets, convergence, stability
- [Small LM Training Guide](docs/reliable_small_language_model_training_guide.md) — comprehensive 100M-250M training guide
- [Backward Pass Techniques](docs/possible_techniques_bwd_improv.md) — survey of backward optimization approaches
