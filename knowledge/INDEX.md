---
title: "Knowledge Index"
domain: project
type: reference
status: active
tags: [%index, %knowledge]
---

# Knowledge Base Index

## hardware/

| File | Description |
|------|-------------|
| [amd_rdna35_strix_halo.md](hardware/amd_rdna35_strix_halo.md) | Full hardware reference: gfx1151 specs, wave32, LDS tiling, rocBLAS, profiling, compilation |
| [workload_guidance.md](hardware/workload_guidance.md) | Optimization strategy framework per workload type |

## kernels/

| File | Description |
|------|-------------|
| [kernel_benchmarks.md](kernels/kernel_benchmarks.md) | All HIP kernel speedups, training results, architecture comparisons |
| [external_kernels.md](kernels/external_kernels.md) | External libraries (causal-conv1d, mamba-ssm, flash_attn, FLA, scattermoe) |
| [fused_kernels.md](kernels/fused_kernels.md) | Custom fused kernels (mhc_sinkhorn 28.5x, engram 7.4x, ple_gate, chunked CE) |
| [mHC_MoE_Engram_optimizations.md](kernels/mHC_MoE_Engram_optimizations.md) | Engram fusion results, ScatterMoE, FLA updates |
| [backward_pass_optimization_research.md](kernels/backward_pass_optimization_research.md) | Research on backward pass optimization approaches |
| [backward_pass_optimization_results.md](kernels/backward_pass_optimization_results.md) | Empirical backward fusion measurements |

## training/

| File | Description |
|------|-------------|
| [ddp_setup_guide.md](training/ddp_setup_guide.md) | 2x Strix Halo DDP via TB4, 35K tok/s, full walkthrough |
| [rccl_build_gfx1151_guide.md](training/rccl_build_gfx1151_guide.md) | Build RCCL from source with gfx1151 patches |
| [muon_optimizer_results.md](training/muon_optimizer_results.md) | Muon vs AdamW: 2x token efficiency, 50% less memory |
| [argus_prime_results.md](training/argus_prime_results.md) | ARGUS-PRIME: 18K single, 35K DDP, WikiText-103 CPT |
| [training_antipatterns.md](training/training_antipatterns.md) | Optimization patterns, anti-patterns, rocBLAS, ROCm dev reference |
| [compressm_in_training_ssm_compression.md](training/compressm_in_training_ssm_compression.md) | CompreSSM (ICLR 2026): balanced truncation for in-training SSM state compression, math + PyTorch code |
| [sft_pipeline.md](training/sft_pipeline.md) | SFT pipeline: EOS warm-up, ChatML, staged instruction tuning (C→A→B), decoding params |

## architectures/

| File | Description |
|------|-------------|
| [hypothesis_buildout_results.md](architectures/hypothesis_buildout_results.md) | 13 architectures trained at 170M params, AMADEUS wins quality |
| [Estimation_Hypothesis_Ranking.md](architectures/Estimation_Hypothesis_Ranking.md) | All 30 hypotheses ranked by MFU and feasibility |
| [reliable_small_lm_insights.md](architectures/reliable_small_lm_insights.md) | 6 gaps in small model training, best practices |
| [ple_ablation_results.md](architectures/ple_ablation_results.md) | PLE Path A wins, MatFormer free, B/AB no benefit |
| [parcae_stable_looped_models.md](architectures/parcae_stable_looped_models.md) | Parcae (Together AI, 2026): stable looped transformers, 2x param efficiency, MuonAdamW recipe |
| [looped_model_design_lessons.md](architectures/looped_model_design_lessons.md) | 13 lessons from OUROBOROS->JORMUNGANDR: momentum risk, staged activation, L2 estimates |

## research/

| File | Description |
|------|-------------|
| [../docs/research/small-lm-research-2026-05-06.md](../docs/research/small-lm-research-2026-05-06.md) | Deep-dive synthesis of efficient pretraining / CPT / post-training / new architectures, with Odin-specific action queue |
