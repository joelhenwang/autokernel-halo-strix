---
title: "Training Performance Analysis: Llama vs Custom Models"
domain: operations
type: guide
status: active
related:
  - docs/halo-strix-apu/01_platform_and_rocm.md
  - docs/superpowers/specs/2026-04-12-compile-optimized-griffin-design.md
  - docs/superpowers/specs/2026-04-10-training-pipeline-optimization-design.md
tags: [%profiling, %throughput, %mfu]
---

# 02 — Why Llama-170M Reaches ~40k tok/s but Custom Models Sit Around ~12k tok/s

## 1. The observed behavior

From the conversation:

- a **Llama-170M-like model** reached roughly **40k tok/s**
- this happened after:
  - successful `torch.compile`
  - substantial kernel optimization
  - fusion of most or all of the Transformer block

Meanwhile:

- **custom architectures** with non-standard components only reached around **12k tok/s**
- even after similar optimization effort

That gap is too large to blame on “one missing hardware feature” alone.

## 2. The most likely explanation: you are falling off the good path

The fast case likely has all of these properties:

- static or stable shapes
- few graph breaks
- dense GEMM-dominated compute
- regular attention/MLP patterns
- library-friendly linears
- large fused compiler regions

The slow case likely has one or more of:

- graph breaks
- dynamic control flow
- many small ops
- irregular reshapes/transposes
- extra reductions
- custom recurrences or scans
- gather/scatter/index-heavy work
- custom backward graphs that are much uglier than the forward

PyTorch’s `torch.compile` docs explicitly warn that code which is difficult to trace results in **graph breaks**, which are lost optimization opportunities.  
Source: PyTorch `torch.compile` tutorial  
<https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>

That is a clean match to your situation.

## 3. Why graph breaks matter so much here

A graph break usually means:

- less fusion
- more kernel launches
- more intermediate tensors
- more memory traffic
- worse scheduling opportunities

On a smaller model like ~170M parameters, the cost of these overheads becomes more visible because the model is not so huge that it automatically drowns them out with gigantic compute kernels.

In other words:

> the smaller and more irregular the workload, the more painful it is to lose fusion.

## 4. Why “custom architecture” often means “memory-bound architecture”

A vanilla Transformer is incredibly hardware-friendly:

- large linears
- large batched GEMMs
- predictable tensor shapes
- well-understood backwards
- lots of mature kernels in libraries

Custom blocks often degrade arithmetic intensity by replacing a few big dense operations with many smaller operations and extra tensor movement.

That means the bottleneck shifts toward:

- memory bandwidth
- kernel-launch overhead
- reductions
- temporary tensor materialization
- non-fused pointwise chains

The result is exactly the kind of collapse from ~40k tok/s to ~12k tok/s that you saw.

## 5. Why PyTorch’s library path can beat hand-optimization on gfx1151

AMD’s docs say PyTorch on ROCm can use **TunableOp** to pick the best GEMM kernel from **rocBLAS** and **hipBLASLt**, and specifically mention substitution of `torch.nn.functional.linear(...)` with a better-performing kernel at runtime.  
Source: AMD model acceleration libraries doc  
<https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html>

That is why the following pattern is common on your platform:

- a hand-written or experimental custom path underperforms
- a plain `F.linear` or `nn.Linear` path wins because it lands on a more mature backend

So the right question becomes:

> How do I redesign the custom model so more of its hot path becomes “library-visible” dense GEMM?

## 6. Why the backward pass may be much worse than the forward

Custom architectures often look manageable in forward but explode in backward because the autograd graph introduces:

- extra reductions
- transposes
- saved activations
- gradient accumulation kernels
- more opportunities for graph fragmentation

This is one of the easiest ways to underestimate the performance cost of a novel block.

The model may not be “compute-poor” in theory, but the actual training graph becomes:

- more fragmented
- more memory-heavy
- less library-friendly

## 7. A rough quantitative interpretation of your throughput

A very rough back-of-the-envelope training estimate for dense decoder models is about:

- **~6 × parameter_count FLOPs per token**

For a **170M** model:

- **40k tok/s** is roughly **40.8 TFLOP/s effective**
- **12k tok/s** is roughly **12.2 TFLOP/s effective**

That means the custom path is only delivering around **30% of the effective training throughput** of the optimized Transformer path.

That is not a “small inefficiency.” It is a sign that the compute graph is fundamentally landing on a much worse execution path.

## 8. The best candidate reasons for not reaching 30k tok/s

The most likely reasons are:

### A. Too many graph breaks
Any data-dependent branching or tracing-unfriendly Python logic can fragment the compiled graph.  
Source: PyTorch `torch.compile` tutorial  
<https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>

### B. Too many small kernels
Even if each kernel is “correct,” many tiny kernels usually lose badly to a few large kernels.

### C. The model is no longer dominated by large dense GEMMs
If the custom architecture replaces big linears with many small projections, scans, or irregular tensor operations, the best rocBLAS paths become less central.

### D. Backward is uglier than expected
Novel blocks often pay a much larger penalty in backward than people first assume.

### E. The model shape is not backend-friendly
Bad alignment, awkward layouts, or unfriendly aspect ratios can all reduce the effectiveness of GEMM backends.

### F. The compiler is not seeing enough stable structure
If the compiler cannot keep large regions intact, you lose the advantage that made Llama-170M fast.

## 9. What to do first

### Step 1: Profile one full training step
Do not optimize blindly.

Use:
- `torch.profiler`
- GPU kernel traces if needed
- timing broken down by operator class

You want to answer:

- what are the top 10 ops by time?
- what fraction is in `mm/addmm/bmm/matmul`?
- what fraction is in norms, reductions, pointwise ops, indexing, transposes, or custom kernels?

### Step 2: Look for graph breaks
PyTorch’s compiler docs explicitly say graph breaks are lost optimization opportunities.  
Source: PyTorch `torch.compile` tutorial  
<https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>

Any custom architecture work should include checking whether the fast path still compiles into large regions.

### Step 3: Test backend selection
Use rocBLAS / hipBLASLt selection and TunableOp. This is one of the highest-leverage experiments on your platform.

### Step 4: Refactor the block
You want to convert:

- many small weird ops

into:

- fewer large dense ops plus fuseable pointwise work

## 10. Practical design rules for custom blocks on Halo Strix

If the target is to get closer to **30k tok/s**, bias toward:

- fewer, larger linears
- batched dense operations
- fewer temporary tensors
- fewer materializing transposes
- simpler control flow
- stable shapes
- less Python-side branching
- avoiding tiny per-head or per-expert matmuls when possible
- minimizing operations that defeat fusion

In one sentence:

> make the custom block look more like a compiler-friendly GEMM sandwich.

## 11. What not to do

- do not assume a novel block is fast because the math looks elegant on paper
- do not spend weeks on custom low-level matmul kernels before confirming rocBLAS is actually the bottleneck
- do not use two machines to scale a bad single-node kernel path
- do not assume forward timings tell the full training story

## 12. Recommended debugging order

1. **Get a top-ops profile**
2. **Check graph breaks**
3. **Benchmark rocBLAS vs hipBLASLt vs TunableOp**
4. **Inspect where small kernels explode**
5. **Refactor the custom block**
6. **Only then** consider moving the whole model to CUDA if needed

## References

- PyTorch `torch.compile` tutorial:  
  <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>
- AMD model acceleration libraries doc:  
  <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html>
