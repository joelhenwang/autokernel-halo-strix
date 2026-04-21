# GPU Kernel Optimization & LLM Training from Scratch

A 15-part hands-on tutorial series for building a complete LLM training system on consumer hardware.

**Target Hardware:** Linux, Ryzen 9 (Zen 3), RTX 4060 Ti (16GB VRAM), CUDA
**Prerequisites:** Python, basic PyTorch, git. No ML/DL theory required — we teach it.
**End Goal:** Train a novel architecture that beats SmolLM2-135M on standard benchmarks.

---

## Series Overview

| Part | Title | What You Build | Key Skill |
|------|-------|---------------|-----------|
| 01 | [Environment & Hardware](01_environment_and_hardware.md) | Validated dev environment + hardware profile | Setup, CUDA basics |
| 02 | [Training Stack](02_training_stack.md) | GPT-2 124M training on BabyLM | PyTorch training loop |
| 03 | [Profiling](03_profiling.md) | Bottleneck map of your training | torch.profiler, nsight |
| 04 | [First CUDA Kernel](04_first_cuda_kernel.md) | RMSNorm CUDA kernel | CUDA programming model |
| 05 | [Kernel Progression](05_kernel_progression.md) | 5+ fused kernels + benchmark suite | Fusion, correctness testing |
| 06 | [Autokernel](06_autokernel.md) | Pattern-matching kernel replacement library | AST walking, module replacement |
| 07 | [Autokernel + Compile](07_autokernel_and_compile.md) | Custom ops registered with torch.compile | torch.library, Inductor |
| 08 | [Data Pipeline](08_data_pipeline.md) | CLIMB mixture optimizer + quality filter | Clustering, proxy search |
| 09 | [Train-Eval-Benchmark](09_train_eval_benchmark.md) | Automated experiment framework | Experiment tracking, eval harness |
| 10 | [Math Foundations](10_math_foundations.md) | Paper-reading toolkit | Linear algebra, SSMs, control theory |
| 11 | [Architecture Design](11_architecture_design.md) | Novel architecture from papers | Research synthesis |
| 12 | [SFT & Alignment](12_sft_and_alignment.md) | Chat model with tool-use | SFT, DPO, ChatML |
| 13 | [Inference & Deployment](13_inference_and_deployment.md) | Fast generation server | KV-cache, quantization |
| 14 | [Putting It All Together](14_putting_it_together.md) | Full pipeline: data → train → eval → deploy | Integration |
| 15 | [Agent Automation](15_agent_automation.md) | Autonomous experiment pipeline | Orchestration, guardrails, LLM agents |

---

## How to Use This Tutorial

**Read linearly.** Each part builds on the previous. Don't skip ahead — Part 05's kernel fusion requires Part 04's CUDA basics, Part 11's architecture requires Part 10's math.

**Build everything yourself.** Don't copy-paste. Type every line. The muscle memory matters.

**Verify at each checkpoint.** Every part ends with concrete verification steps. If your output doesn't match, debug before moving on.

**Hardware-specific.** All benchmarks, memory budgets, and kernel configs are tuned for RTX 4060 Ti (16GB). If you have different hardware, the concepts transfer but numbers will differ.

---

## Time Estimates

| Part | Estimated Time | Difficulty |
|------|---------------|------------|
| 01 | 2 hours | Beginner |
| 02 | 4 hours | Beginner |
| 03 | 3 hours | Beginner |
| 04 | 6 hours | Intermediate |
| 05 | 8 hours | Intermediate |
| 06 | 6 hours | Intermediate |
| 07 | 4 hours | Intermediate |
| 08 | 4 hours | Intermediate |
| 09 | 6 hours | Intermediate |
| 10 | 10 hours | Theory-heavy |
| 11 | 8 hours | Advanced |
| 12 | 6 hours | Intermediate |
| 13 | 6 hours | Intermediate |
| 14 | 4 hours | Integration |
| 15 | 6 hours | Advanced |
| **Total** | **~83 hours** | |

Roughly 3 weeks at 4-5 hours/day, or 6 weeks at 2 hours/day.
