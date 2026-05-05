# Scripts Catalog

> Status: ACTIVE = production use | DEPRECATED = superseded | EXPERIMENTAL = not yet validated

## Training

| Script | Status | Purpose | Hardware |
|--------|--------|---------|----------|
| `train_ddp.py` | ACTIVE | Multi-machine DDP training | gfx1151 x2 |
| `train_170m_smoke.py` | DEPRECATED | Superseded by `halo_training --smoke` | gfx1151 |
| `ddp_env.sh` | ACTIVE | DDP environment variables | gfx1151 x2 |

## Kernel Benchmarking

| Script | Status | Purpose |
|--------|--------|---------|
| `bench_aiter_ops.py` | EXPERIMENTAL | aiter op benchmarks (CDNA ops don't build on gfx1151) |
| `bench_all_hypotheses.py` | HISTORICAL | Hypothesis buildout benchmark |
| `bench_attention_backward.py` | EXPERIMENTAL | Attention backward pass profiling |
| `bench_backward_optimizations.py` | EXPERIMENTAL | Backward pass optimization comparison |
| `bench_before_after.py` | EXPERIMENTAL | Before/after kernel comparison |
| `bench_chunked_ce.py` | ACTIVE | Chunked cross-entropy benchmark |
| `bench_engram_kernels.py` | EXPERIMENTAL | Engram kernel variants benchmark |
| `bench_external_kernels.py` | ACTIVE | External library kernel benchmarks |
| `bench_griffin_compile.py` | EXPERIMENTAL | Griffin compile optimization bench |
| `bench_liger_kernels.py` | EXPERIMENTAL | Liger kernel benchmarks (mostly broken on gfx1151) |
| `bench_124m_comparison.py` | HISTORICAL | 124M architecture comparison |
| `bench_170m_all.py` | HISTORICAL | 170M architecture benchmark |
| `run_kernel_bench.sh` | ACTIVE | Kernel benchmark launcher |

## Infrastructure

| Script | Status | Purpose |
|--------|--------|---------|
| `validate_env.py` | ACTIVE | Validate ROCm + gfx1151 environment |
| `precompile_kernels.py` | ACTIVE | Warm autokernel HIP cache before DDP |
| `pretokenize.py` | ACTIVE | Multiprocessing tokenizer pretokenization |
| `train_tokenizer.py` | ACTIVE | HuggingFace BPE tokenizer training |
| `training_dashboard.py` | ACTIVE | Live training monitor dashboard |
| `analyze_checkpoint.py` | ACTIVE | Checkpoint analysis utility |

## Data & Generation

| Script | Status | Purpose |
|--------|--------|---------|
| `generate_cached.py` | EXPERIMENTAL | Cached text generation |
| `generate_dpo_data.py` | EXPERIMENTAL | DPO preference data generation |
| `generate_text.py` | EXPERIMENTAL | Standard text generation |
| `magpie_generate.py` | EXPERIMENTAL | Magpie data generation |
| `prepare_code_cpt.py` | EXPERIMENTAL | Code CPT data preparation |
| `prepare_swe_data.py` | EXPERIMENTAL | SWE data preparation |
| `prepare_tool_data.py` | EXPERIMENTAL | Tool use data preparation |

## Tuning & Profiling

| Script | Status | Purpose |
|--------|--------|---------|
| `tune_flash_attn_bwd.py` | EXPERIMENTAL | Flash attention backward tuning |
| `tune_rocblas_gemm.sh` | ACTIVE | rocBLAS GEMM autotuning |
| `tune_training_pipeline.py` | EXPERIMENTAL | Training pipeline profiling |
| `profile_backward_breakdown.py` | EXPERIMENTAL | Backward pass breakdown |
| `profile_training_step.py` | EXPERIMENTAL | Training step profiling |
| `compare_conductor.py` | EXPERIMENTAL | Conductor comparison analyzer |
| `patch_aiter_bwd_rdna.py` | EXPERIMENTAL | aiter backward patch for RDNA |
| `patch_aiter_ck_rocm.sh` | EXPERIMENTAL | aiter/ComposableKernel ROCm patch |
| `fix_dualpath.py` | EXPERIMENTAL | Dual-path architecture fix utility |
| `agent.py` | EXPERIMENTAL | Agent experiment runner |

## Architecture-Specific Runners (Shell Scripts)

All shell scripts below are EXPERIMENTAL or HISTORICAL.

| Script | Status | Purpose |
|--------|--------|---------|
| `run_ablation.sh` | EXPERIMENTAL | Ablation runner |
| `run_ablation_remaining.sh` | EXPERIMENTAL | Remaining ablation runs |
| `run_gpt_small_2ep.sh` | HISTORICAL | GPT-2 small training |
| `run_gpt_small_xsadc.sh` | HISTORICAL | GPT-2 with XSA/DC |
| `run_griffin_halo_bridge.sh` | EXPERIMENTAL | Griffin HALO bridge |
| `run_griffin_halo_compile_fixes.sh` | EXPERIMENTAL | Griffin compile fixes |
| `run_griffin_halo_opt.sh` | EXPERIMENTAL | Griffin HALO optimization |
| `run_griffin_halo_sweep.sh` | EXPERIMENTAL | Griffin HALO parameter sweep |
| `run_halo_prime_babylm.sh` | EXPERIMENTAL | HALO-Prime BabyLM |
| `run_halo_prime_wt103.sh` | HISTORICAL | HALO-Prime WikiText-103 |
| `run_kernel_bench.sh` | ACTIVE | Kernel benchmarking |
| `run_ple_ablation.sh` | EXPERIMENTAL | PLE ablation runs |
| `run_progressive_cc.sh` | HISTORICAL | Progressive CommonCrawl |
| `run_progressive_wt103.sh` | HISTORICAL | Progressive WikiText-103 |
| `run_swe_training.sh` | EXPERIMENTAL | SWE training |
| `run_vidar_ablation.sh` | ACTIVE | VIDAR ablation runs |
| `run_wikitext103_comparison.sh` | HISTORICAL | WikiText-103 comparison |
| `run_wt103_ctx1024.sh` | HISTORICAL | WikiText-103 ctx=1024 |
| `run_wt103_ctx512.sh` | HISTORICAL | WikiText-103 ctx=512 |
| `run_wt103_then_1024.sh` | HISTORICAL | WikiText-103 staged |
| `parse_vidar_ablation.sh` | ACTIVE | VIDAR ablation log parser |

## Installation Helpers

| Script | Status | Purpose |
|--------|--------|---------|
| `install_causal_conv1d_rocm.sh` | ACTIVE | Build causal-conv1d from source |
| `install_mamba_ssm_rocm.sh` | ACTIVE | Build mamba-ssm from source |

## Subdirectories

| Dir | Purpose |
|-----|---------|
| `datamix/` | CLIMB + Self-Improving data mixture pipeline |
| `nvidia/` | Standalone NVIDIA GPU training (no ROCm deps) |