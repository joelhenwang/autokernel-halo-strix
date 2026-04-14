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
export NCCL_SOCKET_IFNAME=tb-ddp
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
    --batch-size 16 --block-size 256 --accum-steps 4 \
    --checkpoint-dir checkpoints/argus_prime_cc_ddp \
    --checkpoint-interval 18000 --log-interval 100
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
    --batch-size 16 --block-size 256 --accum-steps 4 \
    --checkpoint-dir checkpoints/argus_prime_cc_ddp \
    --checkpoint-interval 18000 --log-interval 100
```

Note: each machine uses its own venv and cwd but the same relative paths.

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

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NCCL timeout` | Increase `NCCL_TIMEOUT=3600`. Check firewall: `sudo ufw allow 29500` |
| `Connection refused` | Machine 0 must start first. Check `MASTER_ADDR=10.77.0.1` is correct |
| Wrong interface | Verify `NCCL_SOCKET_IFNAME=tb-ddp` matches `ip addr` output |
| One machine crashes | Kill both, resume from latest checkpoint |
| TB4 disconnects | Checkpoint every 1/4 epoch. Resume from last checkpoint |
| Slow sync | Increase `--accum-steps` to 8 to amortize sync further |
| Different venv paths | OK — each machine uses its own venv. PyTorch version must match |

---

## Performance Reference

| Metric | 1 Machine | 2 Machines (DDP) |
|--------|-----------|-----------------|
| tok/s | 16,700 | ~29,000-32,000 |
| Effective batch | 64 | 128 |
| Gradient sync (fp16) | — | ~33ms per optimizer step |
| Memory per machine | 6.1 GB | 6.1 GB (same) |
| Common Crawl 2.4B, 2 epochs | ~79 hrs | ~42-45 hrs |
