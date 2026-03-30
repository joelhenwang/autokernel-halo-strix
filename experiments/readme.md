# AutoKernel MI300 适配与优化闭环

基于 AutoKernel 项目在 AMD MI300 GPU 上的实际落地，梳理 skills-only 模式下的完整工作流。

---

## 1. AutoKernel 架构总览

```text
                 profile.py              extract.py           bench.py (loop)         verify.py
Any PyTorch  ──>  Rank kernels  ──>  Generate baseline  ──>  Optimize each  ──>  End-to-end
   model          by GPU time       Triton/CUDA kernels     kernel (agent)       verification
```

核心设计：Agent 只修改 `kernel.py` 一个文件，`bench.py` 是固定的 5-stage 评估器（smoke → shape sweep → numerical stability → determinism → edge cases + roofline performance），orchestrator 基于 Amdahl's Law 决定优化优先级。

## 2. MI300 适配：本地 Cursor ↔ 远端 GPU 的数据流

在 skills-only 模式下，Cursor IDE 中的 AI Agent 充当 AutoKernel 的"自主研究员"，不需要额外的 LLM API：

```text
┌──────────── Cursor IDE（本地 Windows）────────────┐
│                                                    │
│  AI Agent（Cursor 内置）                           │
│  ├─ 读 program.md（900 行优化指南，静态不修改）     │
│  ├─ 编辑 kernel.py（唯一被修改的文件）              │
│  ├─ 通过 SSH/SCP 与远端 GPU 交互                   │
│  └─ 分析 bench.py 输出 → 决策 keep / revert        │
│                                                    │
│  数据流：                                          │
│  kernel.py ──SCP──▶ 远端                           │
│  bench 结果 ◀──SSH stdout── 远端                   │
│                                                    │
│  不需要 git push/pull 同步                          │
│  不需要外部 LLM API                                │
│  不需要修改 program.md / bench.py / reference.py   │
└────────────────────────────────────────────────────┘
          │ SSH + SCP
          ▼
┌──────────── MI300 远端 Docker ─────────────────────┐
│  rocm/pytorch:rocm7.2 + Triton 3.4.0 (HIP/gfx942) │
│                                                     │
│  python bench.py --kernel <type>                    │
│  ├─ Stage 1: Smoke Test                             │
│  ├─ Stage 2: Shape Sweep (all sizes × all dtypes)   │
│  ├─ Stage 3: Numerical Stability (adversarial)      │
│  ├─ Stage 4: Determinism (3-run bitwise check)      │
│  ├─ Stage 5: Edge Cases (non-power-of-2)            │
│  └─ Performance: TFLOPS / GB/s / % peak / roofline  │
│                                                     │
│  输出: correctness + throughput_tflops + speedup     │
└─────────────────────────────────────────────────────┘
```

## 3. 单 Kernel 优化循环（Phase B 核心）

每轮 ~90 秒，一夜可跑 ~320 次迭代：

```text
┌─────────────────────────────────────────────────────────┐
│  FOR 每个 kernel（按 Amdahl's Law 优先级排序）：          │
│                                                         │
│    LOOP:                                                │
│    ┌──────────────────────────────────────────────────┐  │
│    │ 1. 假设  Agent 提出优化假设                       │  │
│    │    例："BLOCK_SIZE_M 64→128 提升 tiling 效率"      │  │
│    │                                                  │  │
│    │ 2. 编辑  修改 kernel.py（一次只改一处）            │  │
│    │                                                  │  │
│    │ 3. 执行  SCP → 远端 → bench.py → 读结果           │  │
│    │                                                  │  │
│    │ 4. 决策                                          │  │
│    │    ├─ correctness=FAIL → REVERT（绝不保留错误）   │  │
│    │    ├─ TFLOPS 提升 ≥1%  → KEEP（新 baseline）      │  │
│    │    └─ TFLOPS 持平或下降 → REVERT                  │  │
│    │                                                  │  │
│    │ 5. 分析 roofline                                 │  │
│    │    ├─ compute-bound → 尝试 TC/fused epilogue      │  │
│    │    └─ memory-bound  → 尝试 prefetch/coalescing    │  │
│    │                                                  │  │
│    │ 6. orchestrator 判断：CONTINUE / NEXT / DONE      │  │
│    └──────────────────────────────────────────────────┘  │
│                                                         │
│    保存优化后 kernel → 进入下一个 kernel                  │
└─────────────────────────────────────────────────────────┘
```

## 4. MI300 适配实测结果（Baseline / 未经优化循环）

环境：AMD Instinct MI308XHF (gfx942), PyTorch 2.8.0+rocm7.2, Triton 3.4.0 HIP backend。

以下是 9 个 starter kernel 的 **Phase 0 验证**（未经 Agent 迭代优化）：

| Kernel | 类型 | Correctness | Perf | vs PyTorch | 说明 |
|--------|------|-------------|------|-----------|------|
| matmul | 计算密集 | fp16/bf16 PASS | 72.96 TFLOPS | 0.50x | PyTorch 用 rocBLAS，starter 未优化 |
| softmax | 访存密集 | ALL PASS (24/24) | 887.9 GB/s | **1.16x** | Triton fused 天然优于多 op |
| layernorm | 访存密集 | fp16/fp32 PASS | 1076 GB/s | **1.58x** | bf16 有容忍度边界问题 |
| rmsnorm | 访存密集 | ALL PASS (8/8) | 1154 GB/s | **2.71x** | 优秀 |
| flash_attention | 计算密集 | ALL PASS (16/16) | 8.715 TFLOPS | 0.50x | tl.dot + tl.trans 在 HIP 上正确 |
| fused_mlp | 计算密集 | fp16 PASS (修复后) | 119.4 TFLOPS | 0.92x | **需修复** tl.math.tanh→sigmoid |
| cross_entropy | 访存密集 | ALL PASS (21/21) | 950 GB/s | **2.90x** | 优秀 |
| rotary_embedding | 访存密集 | 边界 FAIL | 214.4 GB/s | 0.82x | 容忍度过严 |
| reduce | 访存密集 | ALL PASS (8/8) | 2521 GB/s | **1.04x** | 饱和带宽 |

**关键发现**：
- 计算密集型 kernel（matmul, flash_attn）的 0.50x 是 starter vs rocBLAS/SDPA 的差距，Phase B 优化循环预期可提升至 0.80-0.95x
- 访存密集型 kernel 天然优于 PyTorch（fused kernel 减少 memory round-trip）
- 唯一需要代码修改的是 `tl.math.tanh`（HIP 不可用 → 用 sigmoid 恒等变换替代）

## 5. AMD 优化知识库（sourced from GEAK）

Phase B 优化循环前，已将 [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) 项目中的关键优化知识整合到 AutoKernel：

| 文件 | 内容 | 来源 |
|------|------|------|
| [`knowledge/amd_cdna3_optimization.md`](../knowledge/amd_cdna3_optimization.md) | MI300X 硬件架构、Triton-on-ROCm 差异、autotune configs、occupancy、coalescing、perf counters | GEAK KB layers 1/3/5/6 |
| [`knowledge/workload_guidance.md`](../knowledge/workload_guidance.md) | 基于 bottleneck 的优化策略框架（Prefer First / Consider / Deprioritize） | GEAK `workload_guidance.py` |
| [`program.md` Tier 5](../program.md) | 新增 MI300X (CDNA3, gfx942) 和 HIP 后端的 architecture-specific 优化指南 | GEAK KB + 实测经验 |
| [`bench.py` GPU detection](../bench.py) | 修复 MI308XHF 检测，gcnArchName fallback | MI300 porting 实测 |

### 关键要点（Agent 优化时参考）

1. **wavefront = 64**（非 NVIDIA 的 warp = 32），`num_warps=4` 即 256 线程
2. **BLOCK_SIZE = 256** 通常优于 128（CDNA 架构特性）
3. **`waves_per_eu`** 是 MI300 特有的 occupancy 调优参数，可加入 `triton.Config`
4. **`tl.math.tanh`** 不可用，已用 `2*tl.sigmoid(2*x) - 1` 替代
5. **LDS = 64 KB/CU**（H100 的 228 KB/SM），tile size 需更保守
6. **L2 = 256 MB**（H100 的 50 MB），可以更多依赖 L2 cache
7. 优化策略优先级：**kernel-body 改写 > autotune 参数扫描 > launch config 调优**

## 6. 详细实验记录

见 [mi300_porting.md](mi300_porting.md) -- 包含每个 kernel 的完整实验设计、结果、分析和结论。

## 7. Next Steps: Phase B 优化循环

Phase 0（适配验证）已完成。下一步进入 Phase B 自主优化：

| 优先级 | Kernel | 当前 vs PyTorch | 瓶颈 | 目标 | 策略 |
|--------|--------|----------------|------|------|------|
| 1 | matmul | 0.50x | compute | 0.80-0.95x | 增大 tile size, Split-K, MFMA-friendly layout |
| 2 | flash_attention | 0.50x | compute | 0.70-0.85x | 优化 tl.dot 利用率, 调整 BLOCK_M/N |
| 3 | fused_mlp | 0.92x | compute | 1.0-1.1x | 消除 sigmoid 额外开销, epilogue fusion |
| 4 | rotary_embedding | 0.82x (tolerance) | memory | 1.0x + fix tol | 修复容忍度, 优化 vectorized access |
| 5 | softmax | 1.16x | memory | 1.3-1.5x | 增大 BLOCK_SIZE, 减少 bank conflict |

执行方式：
```bash
# 在本地 Cursor 中：
# 1. 读 program.md + knowledge/amd_cdna3_optimization.md + knowledge/workload_guidance.md
# 2. cp kernels/<type>.py kernel.py
# 3. 编辑 kernel.py（一次改一处）
# 4. SCP kernel.py 到远端 → docker exec python bench.py --kernel <type>
# 5. 分析结果 → keep / revert → 下一轮
```
