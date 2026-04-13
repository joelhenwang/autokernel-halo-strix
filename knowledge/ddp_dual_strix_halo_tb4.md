# 2× Strix Halo DDP Training via Thunderbolt 4

## Overview

Two identical Ryzen AI MAX+ 395 machines connected via Thunderbolt 4 can train together using PyTorch Distributed Data Parallel (DDP). Each machine holds a full model copy, processes different data batches, and synchronizes gradients over TB4. For ARGUS-PRIME (168M params), this yields ~1.7-1.8× speedup — cutting Dolma 10B training from 7 days to 4 days.

## Hardware Setup

```
┌─────────────────────┐     Thunderbolt 4      ┌─────────────────────┐
│  Machine 0 (master)  │◄──── 32 Gbps ────►│  Machine 1 (worker)  │
│  Ryzen AI MAX+ 395   │     (4 GB/s)          │  Ryzen AI MAX+ 395   │
│  Radeon 8060S (40CU) │                       │  Radeon 8060S (40CU) │
│  128 GB LPDDR5X      │                       │  128 GB LPDDR5X      │
│  192.168.1.140       │                       │  192.168.1.XXX       │
└─────────────────────┘                       └─────────────────────┘
```

**Requirements:**
- Both machines on same network (TB4 bridge or Ethernet fallback)
- SSH access between machines (passwordless for torchrun)
- Identical Python environment (.venv with same packages)
- Identical codebase (rsync or shared NFS mount)
- Same ROCm version (7.12)

## Why DDP Works Well for 168M Models

| Factor | Value | Impact |
|--------|-------|--------|
| Model params | 168M | Small gradient payload |
| Gradient size (fp32) | 672 MB | Syncs in 168ms over TB4 |
| Gradient size (fp16 compressed) | 336 MB | Syncs in 84ms |
| Step time (1 machine) | 273ms | Sync is 31-61% of step time |
| With accum_steps=4 | 1,092ms per optimizer step | Sync amortized to 8-15% overhead |
| Effective speedup | 1.7-1.8× | 2× data throughput minus sync cost |

The key: gradient sync happens **once per optimizer step** (not per microstep). With `accum_steps=4`, each machine runs 4 independent microsteps before syncing, amortizing the TB4 cost.

## Architecture Choice: DDP (Not FSDP/Pipeline)

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **DDP** | Simple, no model changes, overlapped sync | Full model copy on each machine | **Best for 168M** |
| FSDP (ZeRO) | Shards params + optimizer state | Overhead > benefit at 168M | Overkill |
| Pipeline Parallel | Splits layers across machines | Bubble overhead, complex | Not worth it |
| DeepSpeed ZeRO-1 | Shards optimizer only | Marginal memory savings | Optional add-on |

At 168M params, the full model + optimizer fits easily in 128 GB. DDP is the simplest and most efficient approach.

## Implementation Plan

### Step 1: Network Configuration

```bash
# On both machines: verify TB4 connectivity
ping 192.168.1.XXX  # other machine's IP

# Ensure SSH works without password (for torchrun)
ssh-copy-id joelwang-ai-2@192.168.1.XXX

# Test bandwidth
iperf3 -s  # on machine 0
iperf3 -c 192.168.1.140  # on machine 1, expect ~3-4 Gbps
```

### Step 2: Environment Sync

```bash
# Sync codebase to both machines
rsync -avz ~/Desktop/ai_lab/autokernel-halo-strix/ \
  joelwang-ai-2@192.168.1.XXX:~/Desktop/ai_lab/autokernel-halo-strix/

# Verify identical environments
ssh joelwang-ai-2@192.168.1.XXX "source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate && python -c 'import torch; print(torch.__version__)'"
```

### Step 3: NCCL/RCCL Configuration

ROCm uses RCCL (ROCm Collective Communication Library) as the NCCL backend. Over TB4 (no RDMA), it falls back to TCP sockets.

```bash
# Environment variables for RCCL over TCP (set on BOTH machines)
export NCCL_SOCKET_IFNAME=eth0          # or enp* — the TB4 network interface
export NCCL_DEBUG=INFO                   # verbose logging (remove after debugging)
export NCCL_P2P_DISABLE=1               # no GPU-direct (unified memory, not discrete)
export NCCL_IB_DISABLE=1                # no InfiniBand
export NCCL_SOCKET_NTHREADS=4           # more threads for TCP
export NCCL_NSOCKS_PERTHREAD=4          # more sockets per thread
export MASTER_ADDR=192.168.1.140        # machine 0 IP
export MASTER_PORT=29500                # any free port
```

### Step 4: DDP Training Script

Create `scripts/train_ddp.py` — a wrapper around the existing trainer:

```python
"""Distributed Data Parallel training across 2 Strix Halo machines.

Usage (run on BOTH machines simultaneously):
  # Machine 0 (master):
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.140 --master_port=29500 \
    scripts/train_ddp.py --model models/argus_prime.py --class-name ArgusPrime ...

  # Machine 1 (worker):
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=192.168.1.140 --master_port=29500 \
    scripts/train_ddp.py --model models/argus_prime.py --class-name ArgusPrime ...
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup():
    dist.init_process_group(backend="nccl")  # RCCL on ROCm
    torch.cuda.set_device(0)  # 1 GPU per machine

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Load model (same on both machines)
    model = load_model(...)
    model = model.cuda()
    model = DDP(model, device_ids=[0])

    # Distributed sampler splits data across machines
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # Optional: fp16 gradient compression to halve sync cost
    from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
    model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Training loop (standard — DDP handles gradient sync automatically)
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # shuffle differently each epoch
        for batch in dataloader:
            loss = model(batch)
            loss.backward()       # DDP syncs gradients here (overlapped)
            optimizer.step()
            optimizer.zero_grad()

    cleanup()
```

### Step 5: Launch Commands

```bash
# Helper script: run_ddp.sh
# Launches training on both machines simultaneously

MASTER=192.168.1.140
WORKER=192.168.1.XXX
PORT=29500

# Machine 0 (master) — run locally
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=$MASTER --master_port=$PORT \
  scripts/train_ddp.py \
  --model models/argus_prime.py --class-name ArgusPrime \
  --dataset datasets/dolma-10b --epochs 2 \
  --optimize-kernels --compile --muon --lr 0.0012 \
  --batch-size 16 --block-size 256 --accum-steps 4 &

# Machine 1 (worker) — run via SSH
ssh joelwang-ai-2@$WORKER "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
  source .venv/bin/activate && \
  NCCL_SOCKET_IFNAME=eth0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=$MASTER --master_port=$PORT \
    scripts/train_ddp.py \
    --model models/argus_prime.py --class-name ArgusPrime \
    --dataset datasets/dolma-10b --epochs 2 \
    --optimize-kernels --compile --muon --lr 0.0012 \
    --batch-size 16 --block-size 256 --accum-steps 4"

wait
```

## Key Considerations

### Gradient Compression

fp16 gradient compression halves the sync payload (672 MB → 336 MB) with minimal quality impact:

```python
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
```

### Muon Optimizer Compatibility

Muon works natively with DDP:
1. DDP allreduces gradients after backward
2. Muon's `step()` receives the averaged gradients
3. Newton-Schulz orthogonalization operates on local (already-synced) gradients
4. No special handling needed — Muon sees the same averaged gradients on both machines

### Data Loading

Both machines must see the **same dataset** but **different batches**:
- `DistributedSampler` handles this automatically
- Dataset must be accessible on both machines (copy or NFS)
- `sampler.set_epoch(epoch)` ensures different shuffling each epoch

### Checkpointing

Only save on rank 0 to avoid duplicate writes:

```python
if rank == 0:
    torch.save({
        'model_state_dict': model.module.state_dict(),  # .module unwraps DDP
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, checkpoint_path)
dist.barrier()  # wait for save before continuing
```

### torch.compile + DDP

Works but compile happens independently on each machine (no shared compilation cache). First-step compilation overhead is ~5-10 min per machine (same as single-machine). After first step, compiled graphs are cached.

### Monitoring

```bash
# Check GPU utilization on both machines
watch -n 1 rocm-smi  # on each machine

# Check RCCL communication
NCCL_DEBUG=INFO torchrun ...  # shows allreduce timing
```

## Expected Performance

| Metric | 1 Machine | 2 Machines (DDP) |
|--------|-----------|-----------------|
| tok/s | 16,700 | ~29,000 |
| Effective batch | 64 | 128 |
| Gradient sync | — | ~84ms (fp16) per optimizer step |
| Memory per machine | 6.1 GB | 6.1 GB (same — full model copy) |
| Dolma 10B, 1 epoch | 6.9 days | ~4 days |
| Dolma 10B, 2 epochs | 13.9 days | ~8 days |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| TB4 disconnects during long training | Checkpoint every 1000 steps, resume from latest |
| RCCL timeout on slow sync | `NCCL_TIMEOUT=1800` (30 min timeout, default is 30s) |
| Different data order on restart | Save sampler state alongside checkpoint |
| One machine crashes | Kill both, resume from checkpoint on single machine |
| TB4 bandwidth saturated | Increase accum_steps (8 instead of 4) to amortize sync |
| Compile mismatch between machines | Same codebase + venv — `rsync` before each run |

## Not Recommended (for 168M model)

- **FSDP/ZeRO:** Model fits in memory; sharding adds complexity for zero benefit
- **Pipeline Parallel:** 16 layers split across 2 machines = bubble overhead + complex scheduling
- **Tensor Parallel:** Splits individual ops across machines — TB4 latency kills this
- **Gradient accumulation >8:** Diminishing returns — Muon works best with moderate batch size
