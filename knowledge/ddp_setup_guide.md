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
    --batch-size 16 --block-size 256 --accum-steps 16 \
    --checkpoint-dir checkpoints/argus_prime_cc_ddp \
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
    --batch-size 16 --block-size 256 --accum-steps 16 \
    --checkpoint-dir checkpoints/argus_prime_cc_ddp \
    --checkpoint-interval 9000 --log-interval 100
```

Note: each machine uses its own venv and cwd but the same relative paths.
Note: `--accum-steps 16` is critical — see "Backend Selection" below.

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

## Backend Selection: gloo (Recommended)

**Key finding:** On Strix Halo with unified memory, `gloo` matches or beats `nccl` for DDP.

| Backend | accum_steps=4 | accum_steps=16 |
|---------|--------------|----------------|
| gloo | ~22K tok/s | **31K tok/s** |
| nccl | ~22K tok/s | ~31K tok/s |

**Why gloo wins on unified memory:**
- No GPU↔CPU copy penalty — both access the same LPDDR5X
- No GPU kernel launch overhead or HIP stream sync barriers
- Zen 5 CPU (16 cores, AVX-512) handles fp32 reductions efficiently
- CPU allreduce runs concurrently on separate cores while GPU computes next microstep

**Why accum_steps=16 is critical:**
- Allreduce happens once per optimizer step, not per microstep (`model.no_sync()` skips non-final steps)
- At accum_steps=4: sync every ~1.1s, overhead ~56% → 22K tok/s
- At accum_steps=16: sync every ~4.4s, overhead ~12% → 31K tok/s (93% scaling efficiency)
- Effective batch goes from 128 to 512, acceptable for 2.4B+ token datasets

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
| Low tok/s at accum_steps=4 | Increase to 16 — amortizes allreduce over more compute |
| Different venv paths | OK — each machine uses its own venv. PyTorch version must match |
| Checkpoint loads with high loss (~70) | Apply autokernel BEFORE loading checkpoint (fused QKV keys) |

---

## Performance (Measured)

| Metric | 1 Machine | 2 Machines (DDP gloo) |
|--------|-----------|----------------------|
| tok/s | 16,700 | **31,000** |
| MFU | 28.5% | 26.8% |
| Effective batch | 64 | 512 (batch=16 × accum=16 × 2 machines) |
| Scaling efficiency | — | **93%** (31K / 33.4K theoretical) |
| Memory per machine | 6.1 GB | 6.1 GB (same) |
| TB4 bandwidth (measured) | — | 9 Gbps |
| Common Crawl 2.4B, 2 epochs | ~79 hrs | **~42 hrs (1.8 days)** |
