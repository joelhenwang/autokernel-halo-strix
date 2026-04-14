# Halo Strix LLM Training Notes

This documentation pack consolidates the main engineering conclusions from our discussion about training small language models on **Ryzen AI Max+ 395 / Strix Halo (gfx1151)** systems.

It is organized by domain so you can read the parts independently:

- `01_platform_and_rocm.md` — hardware/software reality of gfx1151, ROCm support, rocBLAS, hipBLASLt, rocWMMA, AOTriton
- `02_training_performance_analysis.md` — why the fused Llama-170M path reaches ~40k tok/s while custom blocks sit around ~12k tok/s
- `03_cuda_oculink_strategy.md` — when an RTX 4060 Ti over OCuLink helps, and when it does not
- `04_two_machine_distributed_training.md` — how to think about 2-node training with two identical Halo Strix boxes
- `05_100b_token_runbook.md` — a concrete playbook for the future 100B-token run
- `06_commands_and_checklists.md` — commands, environment variables, and benchmarking checklist

## Executive summary

### 1) The chip is not “missing matmul”
The most important correction is conceptual:

- **gfx1151 can do matrix multiplication**
- ROCm officially supports **Ryzen AI Max+ 395 / gfx1151** on Linux
- PyTorch on ROCm can route GEMMs through **rocBLAS** and **hipBLASLt**
- but some of the more aggressive or specialized fast paths on gfx1151 are still **less mature** than on CUDA or on better-covered AMD targets

So the real issue is usually:

> **not** “the APU cannot do matmul”  
> **but** “the optimized kernel/backend path I wanted is missing, experimental, or weak on gfx1151.”

### 2) Your 40k tok/s result is probably a best-case path
Your reported **~40k tok/s** on a Llama-170M-style model is likely benefiting from:

- a very regular decoder-only Transformer structure
- few graph breaks under `torch.compile`
- large dense GEMMs
- large fused regions
- mature library-backed kernels

That is exactly the kind of architecture most likely to map well to current ROCm/PyTorch tooling.

### 3) The ~12k tok/s custom-model result is probably a graph-and-kernel problem
The much slower custom architectures are very likely suffering from a combination of:

- graph breaks
- weaker fusion
- many small kernels instead of a few large ones
- more memory traffic
- less friendly backward graphs
- falling off the best rocBLAS / hipBLASLt paths

### 4) The safest optimization stance on gfx1151 is:
- **lean into rocBLAS-backed GEMMs**
- use **standard `nn.Linear` / `F.linear` / `torch.matmul` / `torch.addmm` patterns**
- use **TunableOp**
- benchmark **rocBLAS vs hipBLASLt**
- avoid betting heavily on custom low-level matrix kernels unless profiling proves they win

### 5) Do not split individual ops between AMD and NVIDIA
Using an **RTX 4060 Ti 16GB over OCuLink** can make sense if:

- the **whole model** lives on CUDA, or
- an entire coarse stage of the workload lives on CUDA

It is usually **not worth it** to do fine-grained per-op ping-pong between:

- Halo Strix ROCm on one side, and
- the RTX 4060 Ti over OCuLink on the other

The transfer link is just too small relative to local memory bandwidth.

### 6) Two Halo Strix machines can help, but only with coarse data parallelism
If you use both machines, the realistic strategy is:

- **2-node DDP**
- one full model replica per machine
- local token shards on each machine
- gradient accumulation with `no_sync()` to reduce synchronization frequency

What is **not** attractive here:

- tensor parallel across the machines
- chatty model parallelism
- FSDP unless memory forces it

For a ~170M model, communication overhead matters more quickly than people expect.

## Fast decision guide

### If the goal is: maximize throughput on one box
Use the Halo Strix machine alone and focus on:
- compiler friendliness
- rocBLAS/hipBLASLt selection
- fusing large regions
- turning custom blocks into fewer, larger GEMMs

### If the goal is: rescue a custom architecture that maps badly to gfx1151
Try the **entire model on the RTX 4060 Ti**, not selective op offload.

### If the goal is: shorten a long 100B-token run
Try **2-node DDP** across the two Halo Strix boxes, but only after the single-node path is already healthy.

## Numbers that matter

Using the throughputs discussed in the conversation:

- **12k tok/s** → ~96.5 days for 100B tokens
- **30k tok/s** → ~38.6 days for 100B tokens
- **40k tok/s** → ~28.9 days for 100B tokens
- **60k tok/s** → ~19.3 days for 100B tokens
- **70k tok/s** → ~16.5 days for 100B tokens

That is why it is worth solving the performance problem before scaling to the 100B-token dataset.

## Source notes

This pack mixes:
- conclusions from our conversation,
- your reported measurements and hardware setup,
- and supporting external references from AMD, PyTorch, Linux kernel docs, NVIDIA, and the llama.cpp issue tracker.

The external references are embedded in each file where they matter.
