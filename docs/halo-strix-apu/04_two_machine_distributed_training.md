---
title: "Two Halo Strix Machines: Distributed Training, Thunderbolt, and Ethernet"
domain: operations
type: guide
status: active
related:
  - docs/halo-strix-apu/03_cuda_oculink_strategy.md
  - docs/halo-strix-apu/05_100b_token_runbook.md
tags: [%ddp, %distributed, %2-machine]
---

# 04 — Two Halo Strix Machines: Distributed Training, Thunderbolt, and Ethernet

## 1. The hardware situation

You have:

- **two identical Ryzen AI Max+ 395 machines**
- each with **128 GB unified memory**

The realistic interconnect options you identified were:

- **10 Gbps Ethernet**
- **Thunderbolt 4 host-to-host networking** (`thunderbolt-net` / TCP/IP over Thunderbolt)

That is the correct framing.

## 2. What “inter-node” actually means

When we discussed RCCL and “inter-node,” the important meaning was:

> communication between separate computers

It does **not** specifically mean “you need a switch.”

AMD’s RCCL docs say RCCL uses:

- **PCIe and xGMI** for intra-node communication
- **InfiniBand, RoCE, and TCP/IP** for inter-node communication

Source: RCCL documentation  
<https://rocm.docs.amd.com/projects/rccl/en/docs-6.3.0/>

So these are all valid inter-node situations for your two-machine setup:

- direct host-to-host Ethernet
- Ethernet through a switch
- Thunderbolt host-to-host networking, which appears to Linux as a network interface

## 3. Thunderbolt networking is real on Linux

The Linux kernel docs describe host-to-host Thunderbolt networking through the `thunderbolt-net` driver. When enabled, Linux creates a virtual Ethernet interface such as `thunderbolt0`.

Source: Linux kernel Thunderbolt docs  
<https://docs.kernel.org/admin-guide/thunderbolt.html>

That means your idea of “IP/TCP over Thunderbolt” is basically right.

It should be treated as:

- a network link
- not a magical accelerator fabric
- still subject to networking overhead and software stack behavior

## 4. The bandwidth reality

### 10 GbE
Theoretical raw throughput:
- **10 Gbps ≈ 1.25 GB/s**

### Thunderbolt 4
Intel markets Thunderbolt 4 at:
- **40 Gbps bidirectional bandwidth**

Source: Intel Thunderbolt 4 overview  
<https://www.intel.com/content/www/us/en/gaming/resources/upgrade-gaming-accessories-thunderbolt-4.html>

However, you should not assume training will see a clean 4× improvement over 10 GbE just because the headline number is 40 vs 10. Real-world results depend on:

- PCIe tunneling behavior
- protocol overhead
- implementation quality
- driver behavior
- traffic pattern during training

So Thunderbolt networking is best viewed as:

- potentially better than 10 GbE
- not guaranteed to deliver the full marketing ratio in training

## 5. What distribution strategy makes sense for your model

For a ~170M-parameter model, the best multi-machine strategy is almost certainly:

- **DDP**
- one full model replica per machine
- each machine processes a shard of the batch
- gradients are synchronized across the two ranks

PyTorch’s DDP docs state that DDP synchronizes gradients across model replicas and that the user is responsible for partitioning the input, typically with a `DistributedSampler`.  
Source: PyTorch DDP docs  
<https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>

That is exactly the pattern you want.

## 6. Why DDP is the right choice here

A ~170M model is small enough that:

- it fits comfortably on one machine
- you do not need sharding just to make it fit
- communication overhead can matter quickly if you choose a chatty distributed method

DDP is attractive because it is the simplest coarse-grained scheme.

## 7. Why FSDP is probably the wrong starting point

PyTorch’s FSDP tutorial explains that FSDP:

- shards parameters, gradients, and optimizer states
- **all-gathers parameters before forward/backward**
- **reduce-scatters gradients during backward**

Sources:  
<https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html>  
<https://docs.pytorch.org/docs/stable/fsdp.html>

That is useful when you need the memory savings.

But for your case:

- the model is only ~170M
- memory is not the immediate problem
- communication is precious across a modest inter-node link

So FSDP is usually not the first thing to reach for here.

## 8. Why tensor parallel / chatty model parallel is unattractive

Across a modest host-to-host link, strategies that require frequent activation exchange between machines are usually a bad trade for a model this size.

You do **not** want:

- tensor-parallel matmuls split across the two boxes
- pipeline stages that force lots of activation traffic
- a setup that tries to make the two machines behave like one big GPU

The interconnect is simply too weak for that to be attractive compared with local compute.

## 9. The most important communication optimization: fewer synchronizations

PyTorch documents `DistributedDataParallel.no_sync()` as a context manager that disables gradient synchronization temporarily so gradients can accumulate locally, and then synchronize on the first step outside the context.  
Source: PyTorch DDP docs  
<https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>

The PyTorch tuning guide explicitly recommends using `no_sync()` for the first `N-1` accumulation steps when training with gradient accumulation, since all-reduce is only necessary on the final accumulation step before the optimizer step.  
Source: PyTorch tuning guide  
<https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html>

This is probably the single most important distributed-training knob for your setup.

## 10. A rough communication estimate

For a 170M-parameter model with bf16 gradients:

- parameters ≈ **170M**
- gradients ≈ **170M × 2 bytes ≈ 340 MB**

A lower-bound intuition for the cost of moving that much data per sync is:

### On 10 GbE
- `340 MB / 1.25 GB/s ≈ 0.27 s`

### On a perfect 40 Gbps link
- `340 MB / 5 GB/s ≈ 0.07 s`

Real training will do worse than those bounds, but they are useful intuition.

The conclusion is:

> You need enough compute per synchronization event, or the network cost becomes too visible.

That is why gradient accumulation is so important.

## 11. Should you use one machine or both for the 100B-token run?

### During development
Use **one machine**.

Reason:
- kernel tuning is easier
- debugging is easier
- distributed inefficiencies can hide the real bottleneck

### For the long 100B-token run
Using **both machines** can make sense if:
- the single-node path is already healthy
- you use **DDP**
- you reduce sync frequency with accumulation
- each machine has its own local copy of the training shards

## 12. Switch or direct cable?

For only two machines:

### Direct host-to-host Ethernet
Good option if:
- both machines have compatible ports
- you want minimal hardware

### Ethernet through a switch
Good option if:
- you want a cleaner network topology
- you already have or plan to use 10GbE switching
- you want easier expansion or monitoring

### Thunderbolt host-to-host
Good option if:
- it works cleanly on both machines
- you want to see whether it outperforms 10GbE in practice

Important:
- a switch is **not required** for RCCL to work in an inter-node configuration
- the “RCCL” part is the software collective layer, not a specific physical topology requirement

## 13. What to benchmark before committing

Before trusting a two-node setup, benchmark in this order:

1. raw network throughput and latency
2. RCCL collective performance
3. end-to-end tok/s on the real training job

Do **not** trust only the marketing number of the interconnect.

## 14. Best practical recommendation

For your exact situation:

### First choice
- optimize single-node training first

### Then, for multi-node
- use **2-node DDP**
- start with **10 GbE** because it is simpler and easier to debug
- treat **Thunderbolt networking** as an experiment that might be faster

### Avoid
- FSDP unless memory forces it
- tensor/model parallel across the two boxes
- streaming compressed training data over the network during training

## References

- RCCL documentation:  
  <https://rocm.docs.amd.com/projects/rccl/en/docs-6.3.0/>
- Linux kernel Thunderbolt docs:  
  <https://docs.kernel.org/admin-guide/thunderbolt.html>
- Intel Thunderbolt 4 overview:  
  <https://www.intel.com/content/www/us/en/gaming/resources/upgrade-gaming-accessories-thunderbolt-4.html>
- PyTorch DDP docs:  
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>
- PyTorch tuning guide:  
  <https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
- PyTorch FSDP tutorial:  
  <https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html>
- PyTorch FSDP docs:  
  <https://docs.pytorch.org/docs/stable/fsdp.html>
