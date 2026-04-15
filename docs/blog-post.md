---
title: "We Tried to Make LLaMA Faster on an AMD Laptop GPU"
domain: project
type: reference
status: active
related:
  - docs/halo-strix-apu/README.md
  - docs/superpowers/specs/2026-04-08-autokernel-library-api-design.md
tags: [%blog, %narrative, %autokernel]
---

# We Tried to Make LLaMA Faster on an AMD Laptop GPU. Here's What Actually Worked.

*An honest look at GPU kernel optimization on bandwidth-limited hardware — what we learned about fusion, Amdahl's law, and why torch.compile beat 95 hand-tuned experiments.*

---

## The Setup

We had one AMD Radeon 8060S — a laptop GPU built into the new Strix Halo APU. It has 20 compute units, shares its memory with the CPU (no dedicated VRAM), and runs at about 170 GB/s of memory bandwidth. For comparison, an NVIDIA A100 does 2,000 GB/s. We were working with about 8% of the bandwidth.

The question: **how much faster can we make LLaMA inference on this thing?**

We built an autonomous agent that writes, tests, and keeps-or-reverts GPU kernel optimizations in a loop. Over 95 experiments, it found 21 custom HIP C++ kernels. 15 of them beat PyTorch. Some by 16x.

But the end-to-end story? That's where it gets interesting.

## What Even Is a GPU Kernel?

When PyTorch runs `x + residual`, it launches a small program on the GPU — a "kernel" — that adds two arrays element by element. When it then runs `rmsnorm(result)`, that's a different kernel: read the result back from memory, compute the normalization, write the answer.

The problem is that intermediate result. It gets written to global memory (slow), then immediately read back (slow again). If the GPU could just do both operations in one shot — add AND normalize without the round-trip — that's a **kernel fusion**, and it's the single biggest optimization lever on bandwidth-limited hardware.

## Phase 1: Writing 21 Custom Kernels

Our agent wrote HIP C++ kernels (AMD's CUDA equivalent) for every operation in the LLaMA model. Each kernel was benchmarked against PyTorch's version on identical inputs. Here's what we found:

### The Winners (sorted by speedup)

| Kernel | Speedup | Why |
|--------|---------|-----|
| dequantize_int4 | **16.3x** | Fuses 10+ PyTorch ops into 1 |
| fused_residual_add_layernorm | **10.7x** | Eliminates intermediate tensor |
| prefix_scan | **8.4x** | Parallel scan in shared memory |
| dequantize_int8 | **8.1x** | Eliminates 3 dtype casts |
| fused_residual_add_rmsnorm | **6.6x** | Eliminates intermediate tensor |
| rotary_embedding | **3.7x** | Native fp16 intrinsics |
| moe_gating | **3.5x** | Fused softmax + top-k |
| rmsnorm | **3.3x** | Single-pass with shared memory |

### The Losers

| Kernel | Speedup | Why |
|--------|---------|-----|
| matmul | **0.24x** | No matrix cores on this GPU. PyTorch uses rocBLAS which has years of tuning. |
| flash_attention | **0.05x** | Same problem — attention is mostly matmul |
| fused_mlp | **0.02x** | Two large matmuls. Can't compete. |

**The pattern was clear:** we could beat PyTorch on memory-bound operations (where the bottleneck is reading/writing data), but not on compute-bound ones (where the bottleneck is math). And the biggest wins came from fusing 3+ operations into one, eliminating intermediate memory traffic.

## Phase 2: Plugging Kernels Into the Actual Model

Individual kernel benchmarks are exciting. But what happens when you plug them into LLaMA end-to-end?

We built a verification system that swaps PyTorch ops for our custom kernels inside the live model, then measures both correctness and speed. We applied them one at a time:

| Step | Added | End-to-End Speedup |
|------|-------|-------------------|
| Baseline (PyTorch) | — | 1.000x |
| +fused_residual_add_rmsnorm | 6.6x kernel | 1.018x |
| +rmsnorm | 3.3x kernel | 1.033x |
| +silu_gate_mul | 1.6x kernel | 1.052x |
| +rotary_embedding | 3.7x kernel | 1.053x |

**Wait. A 6.6x kernel only gave 1.8% end-to-end improvement?**

Yes. Welcome to Amdahl's law.

## The Amdahl's Law Reality Check

We profiled every GEMM (matrix multiply) operation in LlamaModel7B:

| Operation | Time per Layer | Layers | Total |
|-----------|---------------|--------|-------|
| wq projection | 0.546 ms | x32 | 17.5 ms |
| wk projection | 0.142 ms | x32 | 4.5 ms |
| wv projection | 0.141 ms | x32 | 4.5 ms |
| wo projection | 0.548 ms | x32 | 17.5 ms |
| FFN gate (w1) | 1.493 ms | x32 | 47.8 ms |
| FFN up (w3) | 1.487 ms | x32 | 47.6 ms |
| FFN down (w2) | 1.528 ms | x32 | 48.9 ms |
| Output projection | 4.587 ms | x1 | 4.6 ms |
| **Total GEMM** | | | **192.9 ms (43%)** |

Out of 448ms total, **193ms is matrix multiplication** — and we can't make that faster (no matrix cores). Our kernels optimize the remaining 57%, but those operations are small individually. Even a 6.6x speedup on a 0.5ms operation saves only 0.4ms out of 448ms.

The lesson: **op-level speedup multiplied by Amdahl fraction equals actual impact.** A 6.6x kernel on 1% of the workload ≈ 1% end-to-end improvement.

## The Breakthrough: torch.compile

After exhausting manual kernel optimization at 1.053x, we tried something different: `torch.compile(model, backend="inductor")`.

Result: **1.162x.** One line of code beat 95 experiments.

Why? torch.compile doesn't optimize individual operations — it optimizes the **graph**. It fuses embedding lookups with the first layer's normalization. It combines tensor transposes with attention softmax. It eliminates Python overhead between every operation. It sees the entire computation as one unit and optimizes globally.

Our kernels optimize the 5% of time in RMSNorm and SiLU. torch.compile optimizes the other 55% that's NOT matmul — all the small operations between our kernels that we never thought to fuse.

## Combining Both: The Best of Both Worlds

The problem: torch.compile can't see our HIP kernels. They're dynamically loaded C++ extensions — invisible to the graph tracer.

The solution: register our kernels as `torch.library.custom_op`. This tells the compiler "here's an opaque operation — don't try to fuse INTO it, but fuse everything AROUND it."

```python
@torch.library.custom_op("autokernel::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from kernels.hip.rmsnorm import kernel_fn
    return kernel_fn(x, weight)
```

Now Inductor can optimize the full graph while our hand-tuned kernels handle the operations they're best at.

**Result: 1.189x on LlamaModel7B.** The best of both worlds.

We wrapped it all into a library:

```python
import autokernel
model = autokernel.optimize(model, compile=True)
# That's it. 1.19x faster.
```

## The Decode Story

Everything above is about "prefill" — processing a full sequence of tokens at once. But real LLM inference has two phases:

1. **Prefill**: Process all input tokens (matmul-dominated, our optimizations help)
2. **Decode**: Generate one token at a time (weight-read-dominated, different bottleneck)

We built a KV-cache decode benchmark and measured:

| Model | Decode Latency | Tokens/sec |
|-------|---------------|------------|
| LlamaModel (170M) | 5.05 ms/token | 198 tok/s |
| LlamaModel7B (5.9B) | 106.93 ms/token | 9.4 tok/s |

For the 7B model, each token requires reading ALL 12 GB of model weights through memory. At 170 GB/s, that takes a minimum of 70ms — no kernel optimization can change that. The only options are reducing the weights (quantization, pruning) or generating multiple tokens per pass (speculative decoding).

We investigated writing a custom decode-attention kernel, but attention is less than 1% of decode time. The bottleneck is the weight reads, not the attention computation.

## Things That Didn't Work

Science is about what fails too. Here's what we tried and why it didn't help:

- **hipBLASLt** (AMD's alternative BLAS library): Slightly *slower* than default rocBLAS on our GPU
- **Combined FFN projections** (fusing w1+w3 into one matmul): No speedup — individual GEMMs already large enough for good utilization
- **LDS caching for normalization**: The L2 cache already serves the second read. Shared memory caching only helps when it replaces a *separate kernel launch*
- **Cross-block kernel fusion**: Benchmarked at 0.16% improvement. Not worth the complexity
- **CUDA graphs via `reduce-overhead` mode**: Neutral on gfx1151. ROCm's graph support doesn't add pipelining benefits
- **Binary search for top-k selection**: PyTorch's radix sort is fundamentally better (0.25x)

## Bugs That Taught Us Something

### The Invisible Sine

`model.to(dtype=torch.float16)` silently destroys complex-valued buffers. LLaMA stores rotary position frequencies as complex64 (`cos + j*sin`). After the dtype cast, only the cosine survives — the sine is discarded. The model still runs (it just rotates in one axis instead of two), producing subtly wrong results.

We spent hours debugging why our rotary kernel "failed correctness" before realizing the REFERENCE output was wrong. The fix: save complex buffers before casting, restore after.

### The -ffast-math Trap

HIP's `-ffast-math` flag implies `-ffinite-math-only`, which tells the compiler "assume no infinities or NaNs." The compiler then optimizes away `x != x` (the standard NaN check) and `x == INFINITY`. If your kernel checks for special values — and a benchmark test feeds you infinity — your check silently becomes `false` and your kernel produces garbage.

Fix: use bit-level checks on the fp16 representation: `(__half_as_ushort(h) & 0x7C00) == 0x7C00`.

### The fp16 Rounding Mismatch

PyTorch's `apply_rotary_emb` promotes to fp32 before computing, then casts back. Our kernel uses native fp16 intrinsics (which is WHY it's 3.7x faster). Both are correct — but they produce different fp16 rounding, and over 32 transformer layers the differences compound to 0.13 maximum error.

We added a `kernel_fn_fp32` variant that computes in fp32 for model-level precision matching, while keeping the fp16 variant for standalone kernel benchmarks.

## The Final Scorecard

| Approach | LlamaModel7B Speedup | Effort |
|----------|---------------------|--------|
| PyTorch baseline | 1.000x | — |
| 95 HIP kernel experiments | 1.053x | Very high |
| torch.compile (one line) | 1.162x | Trivial |
| **autokernel.optimize(compile=True)** | **1.189x** | One function call |
| Theoretical max (all non-matmul at 0ms) | ~1.75x | Impossible |

The honest conclusion: **on bandwidth-limited hardware without matrix cores, graph-level compilation (torch.compile) provides more end-to-end benefit than hand-tuned kernel optimization.** Our 95 experiments found 15 kernels that beat PyTorch by up to 16.3x individually — but they cover only 5% of the workload. Inductor covers the other 55%.

The ideal approach combines both: custom kernels as opaque custom ops inside a compiled graph. That's what `autokernel.optimize(compile=True)` does.

## What Would Change on Better Hardware?

On MI300X (data center GPU, 5.3 TB/s bandwidth, matrix cores):
- Matmul goes from 60% peak to 90%+ peak with MFMA
- Our fusion patterns still apply (10.7x fused_residual_add_layernorm isn't hardware-specific)
- The Amdahl fraction shifts — fusion covers a LARGER share of total time
- Expected end-to-end speedup: significantly higher than 1.19x

The patterns we discovered are hardware-agnostic. The specific numbers aren't.

## Key Takeaways

1. **Kernel fusion >> bandwidth optimization.** Eliminating intermediate tensors gives 6-16x. Optimizing memory access patterns gives 1-2x.

2. **Amdahl's law is brutal.** A 16x kernel on 1% of the workload = 1% end-to-end improvement.

3. **torch.compile is surprisingly good.** It fuses operations across the entire graph that manual optimization would never consider.

4. **Custom ops + Inductor = best of both worlds.** `torch.library.custom_op` lets your hand-tuned kernels coexist with graph compilation.

5. **Profile before optimizing.** We spent zero time on matmul optimization after one profiling run showed it was at 60% peak TFLOPS — already near ceiling.

6. **Decode is a different problem.** Prefill is compute-bound (matmul). Decode is memory-bound (weight reads). Different bottlenecks need different solutions.

---

*Built with AutoKernel on AMD Radeon 8060S (Strix Halo, gfx1151). All results on LlamaModel7B (5.9B parameters, fp16) with ROCm 7.12 and PyTorch 2.11.*
