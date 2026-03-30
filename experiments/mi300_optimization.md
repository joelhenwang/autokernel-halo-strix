# MI300 Phase B: Closed-Loop Kernel Optimization

环境：AMD Instinct MI308X (gfx942), PyTorch 2.6.0+rocm6.4, Triton 3.2.0 HIP backend.
节点：smc300x-clt-r4c11-02.cs-clt.dcgpu (8× MI308X)
方法：experiment-driven-doc skill

---

## 实验总览

| Exp | Kernel | 假设 | 状态 | 关键结果 | 结论 |
|-----|--------|------|------|----------|------|
| M1 | matmul | 增大 tile 128×128 + autotune | 🔄 running | — | — |
| M2 | matmul | 增加 num_warps=8 + swizzle | pending | — | — |
| M3 | matmul | Split-K for deep_k shapes | pending | — | — |

---

## Exp-M1: matmul tile size 64→128 + autotune

### Phase 0 确认
- 观测完备性: bench.py 输出 correctness + TFLOPS + % peak，包含所有决策所需信息
- 随机化变量: 无（确定性 benchmark，manual_seed=42）
- 映射唯一性: 每组 (kernel code, input shape, dtype) → 唯一的 (correctness, TFLOPS) ✅

### 假设
当前 starter kernel 使用 BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, 未使用 autotune。
MI300X 有 304 CUs 和 64KB LDS/CU。增大 tile 到 128×128×32 可以：
1. 提高数据复用率（arithmetic intensity 从 64→128）
2. 减少 grid 总 blocks 数（更好的 L2 利用）
3. 配合 num_warps=4（256 threads = 4 wavefronts）提供足够 occupancy

### 实验方案
- 脚本: `kernel.py` (修改 matmul_kernel)
- 关键参数: BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, num_warps=4
- 对照组: starter kernel (64×64×32, 72.8 TFLOPS, 0.443x)
- 变量: tile size 从 64→128

### 预期
- 假设成立: TFLOPS > 100, speedup > 0.60x, correctness PASS (fp16/bf16)
- 假设不成立: TFLOPS ≈ 72 或 correctness FAIL（LDS overflow 或 register spill）

### 结果
（待实验）

### 分析
（待实验）

### 结论与 Next Step
（待实验）

---
