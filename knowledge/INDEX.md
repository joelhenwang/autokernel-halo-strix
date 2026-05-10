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
| [instruct_alignment_techniques_2025_2026.md](training/instruct_alignment_techniques_2025_2026.md) | Survey: SimPO, ORPO, KTO, RePO, AlphaPO, MIWV, DataFlow |
| [alignment_implementation_details.md](training/alignment_implementation_details.md) | Implementation reference: SFT + DPO practicalities |
| [climb_data_mixture.md](training/climb_data_mixture.md) | CLIMB data mixture recipes |
| [swe_specialization.md](training/swe_specialization.md) | Software-engineering specialization notes |
| [pretraining_concerns.md](training/pretraining_concerns.md) | Known concerns during pretraining |
| [bpb_mfu_analysis.md](training/bpb_mfu_analysis.md) | BPB + MFU analysis |
| [evaluation_guide.md](training/evaluation_guide.md) | Evaluation guide |
| [imu1_recipe_2026.md](training/imu1_recipe_2026.md) | **IMU-1 recipe (arXiv:2602.02522): NorMuon + Cautious WD + μP for small LM pretraining. −3.85% loss over AdamW at 430M** |
| [normuon_throughput_gfx1151.md](training/normuon_throughput_gfx1151.md) | **NorMuon empirical throughput on gfx1151 (Sprint 1 + 1.1): fp16 NS reduces cost from 17.8% to 3.5%. Matches paper claim without custom kernel.** |
| [fp16_stability_gfx1151.md](training/fp16_stability_gfx1151.md) | **fp16 prevention + forensics + response for long/multi-epoch/resumed runs. Root causes, --z-loss / --attn-softcap / --activation-monitor, NaN dump schema + diagnostic playbook** |
| [grpo_family_2026.md](training/grpo_family_2026.md) | **GRPO variant family: F-GRPO, Scaf-GRPO, GRPO-SG, f-GRPO, Apriel-Reasoner. Post-R1 RLVR landscape** |
| [cpt_best_practices_2026.md](training/cpt_best_practices_2026.md) | **Continued pretraining recipes: token replay 7-10%, LR rewind 30-50%, stage order, eval schedule. Consolidated from 2025-2026 CPT papers** |
| [scaling_laws_t2_2026.md](training/scaling_laws_t2_2026.md) | **T² scaling laws, Chinchilla Approach 2 biases, architecture-conditional scaling. "Overtrain is compute-optimal" when inference cost included** |
| [zaya1_8b_findings_2026.md](training/zaya1_8b_findings_2026.md) | **ZAYA1-8B technical report — applicable-findings synthesis + Odin applicability matrix. AP-trimming, LZ77 canary, learned residual scaling, RL cocktail (DPPO Binary-TV + Dr-GRPO SMTSN + MaxRL + no-KL-in-reward + momentum-free Muon)** |
| [ap_trimming_recipe.md](training/ap_trimming_recipe.md) | **Answer-Preserving trimming — standalone recipe for using long-CoT data at short context. Source: ZAYA1 §III-A** |

## architectures/

| File | Description |
|------|-------------|
| [hypothesis_buildout_results.md](architectures/hypothesis_buildout_results.md) | 13 architectures trained at 170M params, AMADEUS wins quality |
| [Estimation_Hypothesis_Ranking.md](architectures/Estimation_Hypothesis_Ranking.md) | All 30 hypotheses ranked by MFU and feasibility |
| [reliable_small_lm_insights.md](architectures/reliable_small_lm_insights.md) | 6 gaps in small model training, best practices |
| [ple_ablation_results.md](architectures/ple_ablation_results.md) | PLE Path A wins, MatFormer free, B/AB no benefit |
| [parcae_stable_looped_models.md](architectures/parcae_stable_looped_models.md) | Parcae (Together AI, 2026): stable looped transformers, 2x param efficiency, MuonAdamW recipe |
| [looped_model_design_lessons.md](architectures/looped_model_design_lessons.md) | 13 lessons from OUROBOROS->JORMUNGANDR: momentum risk, staged activation, L2 estimates |
| [moda_cross_layer_2026.md](architectures/moda_cross_layer_2026.md) | **Cross-layer MoDA (arXiv:2603.15619): different from our Parcae MoDA. +2.11% tasks at 1.5B, 3.7% FLOP overhead. Naming-collision warning** |
| [small_lm_arch_interventions_2026.md](architectures/small_lm_arch_interventions_2026.md) | **Consolidated playbook: QK-Norm, value residuals, LayerNorm scaling, per-head gating, no-WD-on-embeds, intra-doc masking, SeeDNorm. Rough port checklist for OdinFlat** |
| [looped_moe_design_2026.md](architectures/looped_moe_design_2026.md) | **Looped MoE (Parcae + ZAYA1 MoE) design: R2 sticky routing, E1 shared experts, N1 per-expert per-iter \u03b3_{e,i}, 2.5-iter Sched-A + M3. Companion to FrankenMoE-Loop v2 spec** |
| [v3_speculative_directions_2026.md](architectures/v3_speculative_directions_2026.md) | **18-idea v3 research catalogue: complex MoE, reversible Parcae, hidden-state diffusion, shared workspace, Kolmogorov routing, mycelial graph, path superposition, entropy conservation, forward-forward, momentum teacher + 8 more. Compatibility matrix + dream-stack + if-only-one ranking. Spec-quality depth per idea; design-only research menu** |

## research/

| File | Description |
|------|-------------|
| [../docs/research/small-lm-research-2026-05-06.md](../docs/research/small-lm-research-2026-05-06.md) | Focused synthesis: SmolLM3 recipe + APO + Odin-specific action queue |
| [../docs/research/broad-research-synthesis-2026-05-06.md](../docs/research/broad-research-synthesis-2026-05-06.md) | **Broad synthesis: 240+ recent papers across pretraining / CPT / RLVR / optimizers / new architectures, 20-item ranked action queue, bets list** |
| [../docs/research/zaya1_8b_technical_report/zaya1_8b_technical_report.md](../docs/research/zaya1_8b_technical_report/zaya1_8b_technical_report.md) | **Full ZAYA1-8B technical report (source). Applied-findings synthesis at [training/zaya1_8b_findings_2026.md](training/zaya1_8b_findings_2026.md)** |
| [../docs/research/subq_ssa_watchlist_2026.md](../docs/research/subq_ssa_watchlist_2026.md) | SubQ / SSA — watch-list note. No technical paper published yet; revisit when one drops |

## Active design specs (not yet implemented)

| File | Description |
|------|-------------|
| [../docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md](../docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md) | **FrankenMoE-Loop v2 + v2.5 spec. Looped MoE on OdinHalo backbone with R2 sticky routing + E1 shared experts + N1 per-expert per-iter γ + Sched-A 2.5-iter + M3 narrow FFN. Implementation blocked on FrankenMoE-Flat v1 L9.** |
