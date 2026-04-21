# Part 01: Environment & Hardware Validation

## What You Will Learn
- How to set up a Python + CUDA development environment from scratch
- Your GPU's memory hierarchy and why it determines which optimizations work
- How to measure real bandwidth and compute throughput (not just read spec sheets)
- The roofline model: how to predict whether an operation is memory-bound or compute-bound
- Memory budget planning: how to calculate if a model fits in your VRAM before training

## Goal
Set up a complete development environment and understand your hardware's capabilities and limits before writing a single line of training code.

## Why This Matters
Every optimization decision depends on knowing your hardware. Writing a kernel without knowing your bandwidth ceiling is like driving without knowing the speed limit — you can't tell if you're fast or slow.

---

## 1.1 System Setup

### Linux Environment
```bash
# Check your hardware
lscpu | grep "Model name"          # Ryzen 9 (Zen 3)
nvidia-smi                          # RTX 4060 Ti, 16GB
cat /proc/meminfo | head -1         # System RAM
```

### Python Environment
```bash
# Create isolated project
mkdir gpu-kernel-lab && cd gpu-kernel-lab
python3 -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install tiktoken numpy matplotlib

# Verify CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name()}')"
```

### CUDA Toolkit
```bash
# Check nvcc (needed for custom kernels later)
nvcc --version
# If missing: sudo apt install nvidia-cuda-toolkit
# Or install CUDA toolkit from NVIDIA website (match your driver version)
```

**Checkpoint:** `torch.cuda.is_available()` returns `True`, `nvcc --version` shows CUDA 12.x.

---

## 1.2 Know Your GPU

### Memory Hierarchy

Note: The RTX 4060 Ti comes in different variants. The 16GB model uses the AD106 die. Your exact SM count may be 32 or 34 depending on the SKU — run `nvidia-smi -q | grep "Multiprocessor"` to check yours. The numbers below use 34 SMs; adjust if yours differs.

```
RTX 4060 Ti 16GB (AD106, Ada Lovelace)
├── 16 GB GDDR6 @ 288 GB/s          ← Your VRAM (main bottleneck for LLM training)
├── 32 MB L2 cache                   ← Shared across all SMs (huge — helps repeated reads)
├── 128 KB L1 / Shared Memory per SM ← Fast local storage (you control the split)
├── 34 SMs × 128 CUDA cores = 4352  ← Scalar compute (FP32)
├── 34 SMs × 4 Tensor Cores = 136   ← Matrix multiply hardware (FP16/INT8/INT4)
└── Warp size: 32 threads            ← Minimum execution unit (ALL 32 execute same instruction)
```

Think of it as a hierarchy of speed vs size:
```
Registers     (~256 KB total)  ← Fastest, per-thread, gone when thread finishes
Shared Memory (~128 KB per SM) ← Fast, shared within a block, you manage it manually
L1 Cache      (part of above)  ← Automatic, per-SM
L2 Cache      (32 MB)          ← Automatic, shared across all SMs
VRAM          (16 GB)          ← Slowest, but largest
```

Every CUDA kernel optimization is about moving data into the faster levels of this hierarchy and keeping it there as long as possible.

### Key Numbers to Remember
| Metric | Value | Why It Matters |
|--------|-------|---------------|
| VRAM | 16 GB | Model + optimizer + activations must fit |
| Bandwidth | 288 GB/s | Speed limit for memory-bound ops |
| FP16 Tensor Core TFLOPS | 176 | Speed limit for compute-bound ops |
| FP32 TFLOPS | 22 | Non-tensor-core compute |
| L2 Cache | 32 MB | Data < 32MB gets "free" second reads |
| Warp size | 32 | Your thread block MUST be a multiple of 32 |

### Measure Actual Bandwidth
```python
"""benchmark_bandwidth.py — Measure your GPU's actual memory bandwidth."""
import torch
import time

def measure_bandwidth(size_mb=256, iterations=100):
    size = size_mb * 1024 * 1024 // 4  # float32 elements
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.empty_like(a)
    
    # Warmup
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        b.copy_(a)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    bytes_moved = size * 4 * 2 * iterations  # read + write
    bandwidth = bytes_moved / elapsed / 1e9
    print(f"Measured bandwidth: {bandwidth:.1f} GB/s (theoretical: 288 GB/s)")
    print(f"Efficiency: {bandwidth / 288 * 100:.1f}%")

measure_bandwidth()
```

**Expected:** ~250-270 GB/s (85-95% of theoretical). If under 200, something is wrong with your driver/PCIe config.

### Measure Tensor Core Throughput
```python
"""benchmark_tensorcore.py — Measure FP16 matmul TFLOPS."""
import torch
import time

def measure_tflops(M=4096, N=4096, K=4096, iterations=100):
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    flops = 2 * M * N * K * iterations
    tflops = flops / elapsed / 1e12
    print(f"FP16 matmul: {tflops:.1f} TFLOPS (theoretical: 176 TFLOPS)")
    print(f"Efficiency: {tflops / 176 * 100:.1f}%")

measure_tflops()
```

**Expected:** ~100-140 TFLOPS (60-80% of theoretical). cuBLAS handles large matmuls well.

---

## 1.3 Memory Budget Planning

Your 16GB VRAM must hold:
```
Model weights (fp16):        params × 2 bytes
Optimizer state (AdamW):     params × 8 bytes (fp32 copy + momentum + variance)
Gradients (fp16):            params × 2 bytes
Activations:                 depends on batch_size × seq_len × layers
─────────────────────────────────────────────────
Total ≈ params × 12 bytes + activations
```

| Model Size | Weights | Optimizer | Gradients | Activations (B=8, T=1024) | Total | Fits? |
|-----------|---------|-----------|-----------|---------------------------|-------|-------|
| 124M (GPT-2) | 0.25 GB | 1.0 GB | 0.25 GB | ~2 GB | ~3.5 GB | YES |
| 350M | 0.7 GB | 2.8 GB | 0.7 GB | ~4 GB | ~8.2 GB | YES |
| 770M | 1.5 GB | 6.2 GB | 1.5 GB | ~6 GB | ~15.2 GB | TIGHT |
| 1.3B | 2.6 GB | 10.4 GB | 2.6 GB | ~8 GB | ~23.6 GB | NO |

**Your sweet spot: 124M-350M.** 770M possible with gradient checkpointing.

### Verify Memory Limits
```python
"""check_memory.py — Verify your model fits in VRAM."""
import torch
import torch.nn as nn

# Simulate 350M model
param_count = 350_000_000
dummy = nn.Linear(1024, 1024).cuda().half()  # just to init CUDA

# Allocate model-sized tensor
model_mem = param_count * 2  # fp16
opt_mem = param_count * 8    # AdamW fp32
grad_mem = param_count * 2   # fp16

total_gb = (model_mem + opt_mem + grad_mem) / 1e9
free_gb = torch.cuda.mem_get_info()[0] / 1e9
total_vram = torch.cuda.mem_get_info()[1] / 1e9

print(f"VRAM: {total_vram:.1f} GB total, {free_gb:.1f} GB free")
print(f"Estimated usage for 350M model: {total_gb:.1f} GB (weights + optimizer + gradients)")
print(f"Remaining for activations: {free_gb - total_gb:.1f} GB")
```

---

## 1.4 The Roofline Model

Before optimizing anything, understand what limits your operation:

```
                     Compute
                     Bound
                       │
TFLOPS  ─────────────/─│─────────── Peak (176 TFLOPS)
                     /  │
                    /   │
                   /    │
                  /     │
                 /      │
                /       │
               /        │
Bandwidth ────/─────────│
Bound        /          │
            /           │
           /            │
          /             │
   Arithmetic Intensity (FLOPS / Byte)
```

- **Memory-bound ops** (left of ridge): RMSNorm, softmax, activation functions. Speedup = reduce memory traffic.
- **Compute-bound ops** (right of ridge): Matmul, attention. Speedup = use Tensor Cores.
- **Ridge point** for 4060 Ti: 176 TFLOPS / 288 GB/s ≈ **611 FLOPS/byte**

If an op does less than 611 FLOPS per byte of data moved, it's memory-bound. Most non-matmul ops are heavily memory-bound.

---

## 1.5 Project Structure
```bash
gpu-kernel-lab/
├── models/           # Model definitions
├── kernels/          # CUDA kernels
├── training/         # Training loop, data, optimizers
├── autokernel/       # Pattern matching + kernel replacement
├── scripts/          # Data prep, evaluation, utilities
├── benchmarks/       # Kernel benchmarks + profiling results
├── checkpoints/      # Saved model weights
├── datasets/         # Training data
└── docs/             # Notes, results, learnings
```

```bash
mkdir -p models kernels training autokernel scripts benchmarks checkpoints datasets docs
```

---

## Exercises

1. **Bandwidth at different sizes.** Run `measure_bandwidth()` with sizes 1 MB, 16 MB, 64 MB, 256 MB, 1024 MB. Plot bandwidth vs size. At what size does bandwidth plateau? (This tells you the minimum transfer size needed to saturate the bus.)

2. **Tensor Core efficiency.** Run `measure_tflops()` with matrix sizes (512, 512, 512), (1024, 1024, 1024), (4096, 4096, 4096). At what size do Tensor Cores reach >50% efficiency? (This tells you the minimum matmul size worth optimizing.)

3. **Memory arithmetic.** Calculate by hand: if you train a 200M parameter model with AdamW in fp16, using batch_size=16 and block_size=1024, how much VRAM do you need? (Model weights + optimizer + gradients + estimated activations.) Check your answer against `torch.cuda.max_memory_allocated()` after one training step.

4. **Roofline classification.** For each operation below, calculate its arithmetic intensity (FLOPS per byte of data moved) and classify it as memory-bound or compute-bound on your 4060 Ti (ridge point = 611 FLOPS/byte):
   - RMSNorm on a (8, 1024, 768) tensor in fp16
   - Matrix multiply (8*1024, 768) @ (768, 2048) in fp16
   - Element-wise SiLU on a (8, 1024, 2048) tensor in fp16

---

## Checkpoint

Before moving to Part 02, verify:
- [ ] `torch.cuda.is_available()` → True
- [ ] `nvcc --version` → CUDA 12.x
- [ ] Measured bandwidth > 200 GB/s
- [ ] Measured FP16 TFLOPS > 80
- [ ] Memory budget calculated for your target model size
- [ ] Project directory structure created
- [ ] Exercises 1-4 completed with written answers

---

## What Comes Next

You now know your hardware's limits: how fast it can move data (bandwidth), how fast it can compute (TFLOPS), and how much it can hold (16 GB). In Part 02, you will build a complete training stack (model + data + training loop) and run it on this hardware. The performance numbers you measured here become your reference — when Part 03 profiles the training loop, you will know exactly whether each operation is hitting the hardware ceiling or leaving performance on the table.

**Next: [Part 02 — Training Stack: GPT-2 on BabyLM](02_training_stack.md)**
