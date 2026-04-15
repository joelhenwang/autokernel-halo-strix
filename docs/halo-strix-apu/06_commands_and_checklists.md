---
title: "Commands, Environment Variables, and Benchmarking Checklist"
domain: operations
type: guide
status: active
related:
  - docs/halo-strix-apu/README.md
  - docs/halo-strix-apu/01_platform_and_rocm.md
tags: [%commands, %checklists, %troubleshooting]
---

# 06 — Commands, Environment Variables, and Benchmarking Checklist

## 1. GEMM backend experiments on ROCm

These are the highest-value experiments to run when you suspect GEMM backend selection is the problem.

### Baseline
```bash
python train.py
```

### Prefer hipBLASLt where possible
```bash
export TORCH_BLAS_PREFER_HIPBLASLT=1
python train.py
```

### Turn on TunableOp
```bash
export PYTORCH_TUNABLEOP_ENABLED=1
python train.py
```

### Warm-up / profiling with TunableOp, then use the selected kernels
```bash
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
python train.py

export PYTORCH_TUNABLEOP_TUNING=0
python train.py
```

### Force rocBLAS away from hipBLASLt
```bash
export ROCBLAS_USE_HIPBLASLT=0
python train.py
```

### Prefer hipBLASLt through rocBLAS dispatch
```bash
export ROCBLAS_USE_HIPBLASLT=1
python train.py
```

### Control batched behavior separately
```bash
export ROCBLAS_USE_HIPBLASLT=1
export ROCBLAS_USE_HIPBLASLT_BATCHED=0
python train.py
```

## 2. rocBLAS / hipBLASLt visibility

If you need to understand where the GEMMs are going, enable logging.

### rocBLAS logging
```bash
export ROCBLAS_LAYER=1
python train.py
```

If you need more detail, test stronger logging levels depending on what your version supports.

### hipBLASLt logging
```bash
export HIPBLASLT_LOG_LEVEL=1
python train.py
```

Use the logs to answer:
- are GEMMs actually going through rocBLAS?
- are hipBLASLt kernels being selected?
- are there obvious fallbacks?

## 3. `torch.compile` debugging

To inspect what compilation is doing:

```bash
export TORCH_COMPILE_DEBUG=1
python train.py
```

Things to inspect:
- number of compiled regions
- whether the custom block causes many graph fragments
- whether the Llama-like model produces fewer/larger compiled regions

PyTorch’s `torch.compile` tutorial is explicit that graph breaks are lost optimization opportunities.  
Source: <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>

## 4. Profiling checklist for the custom models

When profiling one training step, answer these questions:

### Operator breakdown
- what are the top 10 operators by total time?
- what percentage is in:
  - `mm`
  - `addmm`
  - `bmm`
  - `matmul`
  - norms
  - softmax
  - pointwise ops
  - transposes/views
  - indexing/scatter/gather
  - custom ops

### Compiler behavior
- are there graph breaks?
- are there many small kernels?
- is backward much worse than forward?

### Memory behavior
- how much temporary tensor creation happens?
- are there many materializing layout conversions?
- do small ops dominate instead of large dense kernels?

## 5. DDP gradient accumulation pattern

When using gradient accumulation with DDP, use `no_sync()` for the first `N-1` accumulation steps.

Conceptually:

```python
for step, batch in enumerate(loader):
    is_sync_step = ((step + 1) % grad_accum_steps == 0)

    if not is_sync_step:
        with ddp_model.no_sync():
            loss = ddp_model(batch).loss / grad_accum_steps
            loss.backward()
    else:
        loss = ddp_model(batch).loss / grad_accum_steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

PyTorch documents that `no_sync()` disables gradient synchronization temporarily and the next step outside the context performs the synchronization.  
Source: <https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>

PyTorch’s tuning guide recommends exactly this pattern when using gradient accumulation with DDP.  
Source: <https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html>

## 6. 2-node DDP launch sketch

A minimal pattern looks like this.

### Node 0
```bash
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=2
export NODE_RANK=0

torchrun   --nnodes=2   --nproc-per-node=1   --node-rank=${NODE_RANK}   --master-addr=${MASTER_ADDR}   --master-port=${MASTER_PORT}   train.py
```

### Node 1
```bash
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=2
export NODE_RANK=1

torchrun   --nnodes=2   --nproc-per-node=1   --node-rank=${NODE_RANK}   --master-addr=${MASTER_ADDR}   --master-port=${MASTER_PORT}   train.py
```

Then use:
- `DistributedSampler`
- one model replica per node
- local dataset shards

## 7. Thunderbolt networking quick notes

Linux kernel docs say Thunderbolt networking can expose a virtual Ethernet interface such as `thunderbolt0`.

One host can load:
```bash
sudo modprobe thunderbolt-net
```

Then check:
```bash
ip link
```

You should see a `thunderbolt*` interface when the connection is live.

Source: Linux kernel Thunderbolt docs  
<https://docs.kernel.org/admin-guide/thunderbolt.html>

## 8. Network benchmarking checklist

Before trusting distributed training, benchmark:

### Basic link visibility
```bash
ip addr
ip route
ping <peer-ip>
```

### Throughput
Use your preferred network benchmark tool on:
- 10GbE direct link
- Thunderbolt link

### Collective benchmark
Run an RCCL-oriented benchmark or equivalent communication test, not just raw network throughput.

The reason is simple:
- training performance depends on collectives
- not just on one-way bulk copy speed

## 9. Architecture refactor checklist

Use this checklist for custom blocks that underperform badly.

### Keep
- large `Linear` layers
- batched dense math
- stable shapes
- fuseable pointwise chains

### Reduce
- tiny matmuls
- many transposes
- repeated reshapes that materialize
- data-dependent branching
- indexing-heavy hot paths
- custom ops that break compilation
- fragmented backward graphs

### Ask
- can this become fewer, larger GEMMs?
- can this become one compiled region?
- can this become more like `Linear -> pointwise -> Linear`?

## 10. Final practical checklist

### Before using 2 nodes
- [ ] single-node profiler run completed
- [ ] graph breaks understood
- [ ] rocBLAS / hipBLASLt / TunableOp compared
- [ ] tokenized local shards prepared
- [ ] checkpoint save/resume tested

### Before using the 4060 Ti
- [ ] benchmarked as a whole-model target
- [ ] not planning fine-grained per-op offload
- [ ] model fits with optimizer state and sequence/batch settings

### Before starting the 100B-token run
- [ ] input pipeline stable
- [ ] logging stable
- [ ] restart path tested
- [ ] end-to-end tok/s verified
- [ ] validation cadence decided
- [ ] failure/recovery plan ready

## References

- AMD model acceleration libraries doc:  
  <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html>
- PyTorch DDP docs:  
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>
- PyTorch tuning guide:  
  <https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
- PyTorch `torch.compile` tutorial:  
  <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>
- Linux kernel Thunderbolt docs:  
  <https://docs.kernel.org/admin-guide/thunderbolt.html>
