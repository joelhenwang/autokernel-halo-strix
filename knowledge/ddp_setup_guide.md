# DDP Setup Guide: 2x Strix Halo via Thunderbolt 4

Step-by-step guide to connect two Ryzen AI MAX+ 395 machines for distributed training of ARGUS-PRIME.

**Expected result:** ~29-32K tok/s (vs 16.7K single machine), ~1.85-1.9x speedup.

---

## Our Setup

```
┌──────────────────────────┐     Thunderbolt 4      ┌──────────────────────────┐
│  Machine 0 (master)       │◄──── 9 Gbps ─────►│  Machine 1 (worker)       │
│  Ryzen AI MAX+ 395        │     (measured)        │  Ryzen AI MAX+ 395        │
│  Radeon 8060S (40CU)      │                       │  Radeon 8060S (40CU)      │
│  128 GB LPDDR5X           │                       │  128 GB LPDDR5X           │
│  IP: 10.77.0.1            │                       │  IP: 10.77.0.2            │
│  Interface: tb-ddp        │                       │  Interface: tb-ddp        │
│                            │                       │                            │
│  Venv: ~/Desktop/ai_lab/  │                       │  Venv: ~/Desktop/          │
│    autokernel-halo-strix/  │                       │    comfyui-rocm7.12/.venv/ │
│    .venv/                  │                       │                            │
│  Codebase: ~/Desktop/     │                       │  Codebase: ~/Desktop/      │
│    ai_lab/autokernel-     │                       │    comfyui-rocm7.12/       │
│    halo-strix/            │                       │    autokernel-halo-strix/  │
└──────────────────────────┘                       └──────────────────────────┘
```

**TB4 bandwidth measured:** 9 Gbps (iperf3, 10 GB transfer) — 3x better than estimated.

| Metric | Value |
|--------|-------|
| Gradient sync (fp16 compressed, 336 MB) | ~33ms |
| Sync overhead (accum_steps=4) | ~3-6% |
| Effective speedup | ~1.85-1.9x |

---

## Step 1: Physical Connection

Plug a Thunderbolt 4 cable between the two machines.

---

## Step 2: Network

TB4 interface configured as `tb-ddp` with static IPs on a /30 subnet:

```bash
# Machine 0:
sudo ip addr add 10.77.0.1/30 dev tb-ddp
sudo ip link set tb-ddp up

# Machine 1:
sudo ip addr add 10.77.0.2/30 dev tb-ddp
sudo ip link set tb-ddp up
```

Verify: `ping 10.77.0.2` from Machine 0, `ping 10.77.0.1` from Machine 1.

---

## Step 3: Passwordless SSH

```bash
# From Machine 0:
ssh-copy-id joelwang-ai-2@10.77.0.2

# From Machine 1:
ssh-copy-id joelwang-ai-2@10.77.0.1
```

---

## Step 4: Sync Files to Machine 1

Machine 1 does NOT need a full copy of Machine 0's codebase. It needs:
- The model code (`models/argus_prime.py` + dependencies `models/amadeus.py`, `models/argus.py`)
- The training script (`scripts/train_ddp.py`)
- The env script (`scripts/ddp_env.sh`)
- The dataset (`.bin` file)
- The checkpoint (if resuming)
- Matching PyTorch version (both have same version confirmed)

```bash
# From Machine 0 — copy scripts
scp scripts/ddp_env.sh scripts/train_ddp.py \
    joelwang-ai-2@10.77.0.2:~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/scripts/

# Copy dataset (4.7 GB, ~4 sec over TB4)
rsync -avP datasets/common_crawl_sample.bin \
    joelwang-ai-2@10.77.0.2:~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/datasets/

# Copy checkpoint (if resuming)
rsync -avP checkpoints/argus_prime_cc/step_36000.pt \
    joelwang-ai-2@10.77.0.2:~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/checkpoints/argus_prime_cc/
```

---

## Step 5: Test Bandwidth

```bash
# Machine 0:
iperf3 -s

# Machine 1:
iperf3 -c 10.77.0.1
# Result: ~9 Gbps
```

---

## Step 6: RCCL Environment

The `scripts/ddp_env.sh` file is already configured:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export NCCL_SOCKET_IFNAME=thunderbol0
export GLOO_SOCKET_IFNAME=thunderbol0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_TIMEOUT=1800
export MASTER_ADDR=10.77.0.1
export MASTER_PORT=29500
```

Source it on both machines before launching: `source scripts/ddp_env.sh`

For first-run debugging, add `export NCCL_DEBUG=INFO` to see RCCL ring formation logs. Remove after confirming it works.

---

## Step 7: Launch DDP Training

Start Machine 0 first, then Machine 1 within ~30 seconds.

**Machine 0 (master):**
```bash
source ~/Desktop/ai_lab/autokernel-halo-strix/scripts/ddp_env.sh
source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --master_addr=10.77.0.1 --master_port=29500 \
    scripts/train_ddp.py \
    --model models/argus_prime.py --class-name ArgusPrime \
    --resume-from checkpoints/argus_prime_cc/step_36000.pt \
    --dataset datasets/common_crawl_sample.bin \
    --epochs 2 --compile --optimize-kernels --lr 0.0012 \
    --batch-size 16 --block-size 256 --accum-steps 8 \
    --checkpoint-dir checkpoints/argus_prime_cc_ddp_v2 \
    --checkpoint-interval 9000 --log-interval 100
```

**Machine 1 (worker):**
```bash
source ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/scripts/ddp_env.sh
source ~/Desktop/comfyui-rocm7.12/.venv/bin/activate
cd ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=10.77.0.1 --master_port=29500 \
    scripts/train_ddp.py \
    --model models/argus_prime.py --class-name ArgusPrime \
    --resume-from checkpoints/argus_prime_cc/step_36000.pt \
    --dataset datasets/common_crawl_sample.bin \
    --epochs 2 --compile --optimize-kernels --lr 0.0012 \
    --batch-size 16 --block-size 256 --accum-steps 8 \
    --checkpoint-dir checkpoints/argus_prime_cc_ddp_v2 \
    --checkpoint-interval 9000 --log-interval 100
```

Note: each machine uses its own venv and cwd but the same relative paths.
Async overlap + fp16 compression are on by default. Use `--no-async --no-fp16-compress` to disable.

---

## Step 8: Monitor Training

```bash
# From Machine 0:
tail -5 checkpoints/argus_prime_cc_ddp/train_log.jsonl

# Check GPU utilization on each machine:
watch -n 1 rocm-smi

# Check if still running:
ps aux | grep torchrun | grep -v grep
```

---

## Backend & Optimization: gloo + async overlap + fp16 compression

**Key finding:** On Strix Halo with unified memory, `gloo` matches or beats `nccl` for DDP.
Both backends give identical throughput because the bottleneck is TB4 TCP, not the allreduce implementation.

**Why gloo works well on unified memory:**
- No GPU↔CPU copy penalty — both access the same LPDDR5X
- No GPU kernel launch overhead or HIP stream sync barriers
- Zen 5 CPU (16 cores, AVX-512) handles reductions efficiently
- CPU allreduce runs concurrently on separate cores while GPU computes

**Optimization history (measured):**

| Version | Technique | accum_steps | tok/s | Real speedup | MFU |
|---------|-----------|-------------|-------|-------------|-----|
| v1 | DDP built-in sync | 16 | 31K | 1.58x | 26.8% |
| **v2** | **async overlap + fp16 compress** | **8** | **35K** | **2.1x** | **29.8%** |

**v2 optimizations (current):**
- `model.no_sync()` on ALL microsteps — DDP never triggers its own allreduce
- Manual async allreduce: grads compressed to fp16, `dist.all_reduce(async_op=True)` launched after last microstep
- Overlap: next batch's microsteps run on GPU while allreduce transfers over TCP in background
- `scaler.unscale_()` + `clip` + `step` + `update` done together after allreduce completes — keeps GradScaler state consistent
- Default `--accum-steps 8` (effective batch=256), down from 16 — async overlap hides the sync latency

**RCCL/NCCL status:** Bundled RCCL in pip PyTorch has `invalid kernel file` for gfx1151. We built RCCL from source with gfx1151 kernels (see `knowledge/rccl_build_gfx1151_guide.md`). It works, but offers no advantage over gloo on unified memory + TB4 TCP. Use gloo.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| NCCL `invalid kernel file` | Use `--backend gloo` (recommended) or build RCCL from source |
| `find_unused_parameters` error | Already set in train_ddp.py, needed for TTT + compile |
| `static_graph=True` crash | Don't use — incompatible with torch.compile on ROCm |
| `Connection refused` | Machine 0 must start first. Check `MASTER_ADDR=10.77.0.1` |
| Wrong interface | Verify `NCCL_SOCKET_IFNAME=thunderbol0` matches `ip addr` output |
| One machine crashes | Kill both, resume from latest checkpoint |
| TB4 disconnects | Checkpoint every 1/4 epoch. Resume from last checkpoint |
| Low tok/s (~22K) without async | Use default async overlap, or increase `--accum-steps 16` |
| GradScaler assertion error | `unscale_`/`step`/`update` must be together after allreduce completes |
| Different venv paths | OK — each machine uses its own venv. PyTorch version must match |
| Checkpoint loads with high loss (~70) | Apply autokernel BEFORE loading checkpoint (fused QKV keys) |

---

## Performance (Measured)

| Metric | 1 Machine | 2 Machines (DDP v2) |
|--------|-----------|---------------------|
| tok/s | 16,700 | **35,000** |
| MFU | 28.5% | **29.8%** |
| Real speedup | — | **2.1×** |
| Effective batch | 64 | 256 (batch=16 × accum=8 × 2 machines) |
| Memory per machine | 6.1 GB | 6.8 GB |
| TB4 bandwidth (measured) | — | 9 Gbps |
| Common Crawl 2.4B, 2 epochs | ~79 hrs | **~37 hrs (1.5 days)** |

Steady-state: 35K tok/s ± 0.2% over 3600+ steps. Loss converging: 4.27 → 4.14.
