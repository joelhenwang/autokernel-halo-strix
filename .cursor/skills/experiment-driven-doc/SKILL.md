---
name: experiment-driven-doc
description: >-
  实验驱动的文档追溯流程。在设计假设验证实验前，先将假设、实验方案、预期结果写入文档；
  实验/测试完成后，将实际结果、分析、结论、next-step 写回文档。确保每一轮实验可追溯。
  Use when running hypothesis-driven experiments, ablation studies, debugging
  investigations, or any iterative test-analyze-iterate workflow.
---

# Experiment-Driven Documentation

将"假设 → 实验设计 → 执行 → 结果 → 分析 → 下一步"全流程记录到项目文档中，
做到任何人（包括未来的自己）都能复现思路和决策依据。

## Workflow

### Phase 0: Problem Formulation 检查（必须最先做）

在设计任何实验之前，必须先回答以下问题。如果 Phase 0 不通过，**禁止进入 Phase 1**。

**核心问题：`π(observation) → action` 是单值函数吗？**

Checklist：

1. **观测完备性**：observation 是否包含决定 action 所需的**全部信息**？
   - 如果 action 依赖于某个变量（目标位置、任务 ID、环境参数等），该变量**必须**在 observation 中
   - 反例：cube 位置随机化但 observation 只有 joint state → `π(s)` 不是单值函数 → MSE 收敛到无意义的均值
2. **映射唯一性**：对于 observation 空间中的任意一个点，是否存在唯一的正确 action？
   - 如果同一个 observation 在不同 episode / 不同 context 对应不同 action → 问题不可解
   - 检查方式：列出所有随机化变量（goal position、object pose、task type 等），确认它们要么固定、要么在 observation 中
3. **数据-评估一致性**：训练数据中的 `(obs, action)` 与评估时的 `(obs → predict → execute)` 流程是否匹配？
   - observation 的构成和顺序在数据采集与评估中是否一致？
   - action 的语义（absolute / delta / target）在两端是否一致？

**如果发现 observation 不完备：**
- 方案 A：将缺失变量加入 observation（最直接）
- 方案 B：固定该变量（用于诊断，证明问题可解后再推广）
- 方案 C：用 vision 等富信息模态替代（长期方案）

> **教训来源**：在 Franka pick-cube 实验中，cube 位置每 episode 随机化但未加入 observation，
> 导致 π(s)→a 不是单值函数。历经 10+ 组实验（DART、闭环 IK、noise aug、history、时序修复）
> 全部 0% 成功率，最终才追溯到 problem formulation 层面的缺陷。
> 所有后续调优都是在一个**不可解的问题**上做无用功。

---

### Phase 1: Before Experiment — 写入实验设计

在编写代码或启动实验之前，先更新实验文档：

1. **确认 Phase 0 通过**：引用 Phase 0 checklist 的结论
2. **明确假设**：用一句话写清要验证什么
3. **实验方案**：脚本路径、关键参数、变量与控制
4. **预期结果**：如果假设成立 / 不成立，分别期望看到什么
5. **评估标准**：量化指标（success rate、loss、lift 等）

模板：

```markdown
## Exp-<ID>: <简短标题>

### Phase 0 确认
- 观测完备性: <observation 包含哪些信息，是否覆盖所有影响 action 的变量>
- 随机化变量: <哪些变量被随机化，是否在 observation 中>

### 假设
<一句话描述假设>

### 实验方案
- 脚本: `<path>`
- 关键参数: <table or list>
- 对照组: <baseline>
- 变量: <what changes>

### 预期
- 假设成立: <expected metrics>
- 假设不成立: <expected metrics>

### 结果
（待实验）

### 分析
（待实验）

### 结论与 Next Step
（待实验）
```

### Phase 1.5: Smoke Test — 快速验证脚本可运行

当实验预估运行时间 **> 10 分钟** 时，先用极小参数跑一次 smoke test：

1. **缩小规模**：`n_episodes=2, max_steps=50, n_steps=10` 等，保证 1-2 分钟内完成
2. **验证目标**：脚本无 crash、输入输出格式正确、关键计数（replan、gt_inject 等）符合预期
3. **不关注指标**：smoke test 的 success rate 没有意义，只验证管线畅通
4. **立即检查输出**：确认 log 中无 error/warning、输出文件已生成

```bash
# 示例：全量实验前的 smoke test
python 03_eval_act.py \
  --n-episodes 2 --max-steps 50 \
  --save /output/smoke_test
# 检查：exit code=0, eval_summary.json 存在, 关键字段合理
```

只有 smoke test 通过后，才启动全量实验。如果 smoke test 失败，
在文档调试表格中记录问题并修复，**避免浪费长时间等待**。

### Phase 2: During Experiment — 实时记录

- 如果实验中出现 error / 意外行为，立即在文档中记录调试过程
- 多轮迭代（r1, r2, r3...）用表格追踪每轮的问题、修复、结果

```markdown
| 轮次 | 问题 | 修复 | loss | eval 结果 |
|---|---|---|---|---|
| r1 | ... | ... | ... | ... |
| r2 | ... | ... | ... | ... |
```

### Phase 3: After Experiment — 写回结果与分析

实验完成后，必须回填文档中的"待实验"部分：

1. **结果**：原始数据表格（success rate、各 episode 详情等）
2. **分析**：
   - 假设是否成立？用数据说话
   - 与之前实验的对比（delta 表格）
   - 意外发现
3. **结论**：一句话总结
4. **Next Step**：基于结论推导下一步实验，按优先级排列

### Phase 4: 维护总览表

在文档顶部或专门章节维护一个实验总览表，方便快速回顾：

```markdown
| Exp | 假设 | 状态 | 关键结果 | 结论 |
|---|---|---|---|---|
| F1 | IK 数据 100% 成功 | ✅ done | 200/200=100% | Franka IK 稳定 |
| F3 | BC E1 N=5 > SO-101 | ✅ done | 90% vs 0% | 7-DOF 冗余有效 |
| F4 | noise aug 改善 shift | ✅ done | 全部下降 | noise 有害 |
```

## Commit Convention

每次文档更新时，commit message 包含实验 ID：

- 实验前：`F4 noise sweep: experiment design and hypothesis`
- 实验后：`F4 noise sweep results: noise aug harmful on Franka, all sigma degrade E1`

## Key Principles

1. **先审后做**：任何实验之前必须通过 Phase 0 problem formulation 检查 — observation 是否完备、映射是否唯一
2. **先写后跑**：不要跑完实验再补文档，先写设计再执行
3. **数据说话**：结论必须有量化证据支撑，避免"感觉上"
4. **对比呈现**：每个实验结果都与 baseline 和前序实验做 delta 对比
5. **决策可追溯**：next-step 必须从当前结论推导，形成清晰的因果链
6. **不删历史**：失败实验同样保留（标注 ✅ done / ❌ disproved），它们排除了假设空间
