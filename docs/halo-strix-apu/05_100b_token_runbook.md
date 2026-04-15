---
title: "Runbook for the 100B-Token Training Plan"
domain: operations
type: guide
status: active
related:
  - docs/halo-strix-apu/04_two_machine_distributed_training.md
  - docs/superpowers/specs/2026-04-10-training-evolution-design.md
tags: [%runbook, %100b-tokens, %dolma]
---

# 05 — Runbook for the 100B-Token Training Plan

## 1. Dataset size is not the real scaling problem

You mentioned that the **100B-token dataset** corresponds to roughly **180 GB of `.jsonl.zst` files**.

That storage number matters operationally, but it is **not** the main reason to choose one machine or two.

The real deciding factor is:

- **single-node training throughput**
- **distributed communication overhead**
- **how often you synchronize**
- **whether the architecture is already efficient on one node**

## 2. Training-time implications of your current throughput

Using the throughputs discussed in our conversation:

- **12k tok/s** → about **96.5 days**
- **30k tok/s** → about **38.6 days**
- **40k tok/s** → about **28.9 days**
- **60k tok/s** → about **19.3 days**
- **70k tok/s** → about **16.5 days**

for a **100B-token** run.

This is why you should treat the current throughput issue as a major engineering priority.

## 3. Recommended data pipeline shape

Do **not** train directly from a giant pile of compressed `.jsonl.zst` files over a shared network path if you can avoid it.

A better pipeline is:

1. convert raw `.jsonl.zst` into larger local intermediate files if needed
2. tokenize once
3. write **token shards**
4. place a local copy of the token shards on each training machine

That reduces:
- CPU decompression during training
- filesystem overhead
- network dependence during the critical path
- training variability caused by the input pipeline

## 4. The best staging strategy for your case

### Stage A — One-machine optimization
Before doing anything distributed, get the best possible numbers on a single box.

Your goals here:
- confirm whether the architecture can be made compiler-friendly
- confirm whether rocBLAS / hipBLASLt / TunableOp help
- identify top bottlenecks with a profiler
- confirm that the input pipeline is not the limiting factor

### Stage B — Dry run on 2-node DDP
Only after Stage A looks healthy:
- run the exact model on both Halo Strix machines
- use DDP
- add gradient accumulation
- compare tok/s scaling against single-node

### Stage C — Long 100B-token run
Only when:
- the 2-node setup is stable
- checkpointing is tested
- restart and resume are tested
- throughput is repeatable

## 5. When to stay on one machine

Stay on one machine if any of these are true:

- the custom architecture is still stuck on a poor kernel path
- the single-node implementation still has obvious graph-break or fusion problems
- the input pipeline is unstable
- distributed scaling is poor because the step is too communication-heavy

In that case, two nodes do not fix the real issue.

## 6. When to use both machines

Use both machines when:

- single-node throughput is already decent
- you have enough work per optimization step
- gradient accumulation reduces communication frequency
- end-to-end tok/s actually scales in practice

For your long 100B-token run, even a modest scaling gain can save many days.

## 7. Recommended distributed method for the long run

The method I would recommend for the long run is:

- **2 nodes**
- **1 process / 1 GPU per node**
- **DDP**
- **local token shards on each machine**
- **gradient accumulation with `no_sync()`**
- **simple, robust checkpointing**

Why:
- minimal complexity
- predictable communication
- no unnecessary sharding traffic
- easiest setup to debug and resume

## 8. Checkpointing advice

Before the long run, verify all of these:

- save checkpoint
- resume from checkpoint
- resume after forced interruption
- verify optimizer state is restored
- verify LR scheduler state is restored
- verify data loader / shard position recovery if you rely on it
- verify rank-safe checkpoint writing in distributed mode

The longer the run, the more important restart hygiene becomes.

## 9. Monitoring advice

For the 100B-token run, log at least:

- step
- tokens processed
- effective tok/s
- loss
- LR
- grad norm
- GPU memory usage
- host RAM usage
- data loading time
- step time
- checkpoint time
- validation loss/perplexity when relevant

If the architecture is novel, also log the architecture-specific internal signals that matter to you.

## 10. A sensible decision tree

### If single-node custom model is still near 12k tok/s
Do **not** rush into distributed training.
Fix:
- graph breaks
- kernel shape
- backend selection
- memory traffic

### If single-node custom model rises toward ~20–30k tok/s
Then 2-node DDP becomes much more attractive.

### If the whole custom model runs much better on the RTX 4060 Ti
Then compare:
- one Halo Strix
- two Halo Strix nodes with DDP
- one 4060 Ti
and pick the fastest **whole-workload** solution.

## 11. My recommended order of attack

1. stabilize the single-node training graph
2. benchmark rocBLAS / hipBLASLt / TunableOp
3. standardize the input pipeline into token shards
4. benchmark 2-node DDP over 10GbE
5. benchmark 2-node DDP over Thunderbolt networking if available
6. only then start the long 100B-token run

## 12. Final recommendation

For the **real 100B-token run**:

- **Yes, using both Halo Strix machines is worth exploring**
- **No, do not let distributed setup distract you from fixing the single-node bottlenecks first**
- **Store local training shards on each machine**
- **Use DDP, not model-parallel tricks**
- **Treat the network as a synchronization path, not a shared-memory substitute**
