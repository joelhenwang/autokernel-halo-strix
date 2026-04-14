# DDP Setup Guide: 2x Strix Halo via Thunderbolt 4

Step-by-step guide to connect two Ryzen AI MAX+ 395 machines for distributed training of ARGUS-PRIME.

**Expected result:** ~29K tok/s (vs 16.7K single machine), ~1.7-1.8x speedup.

---

## Step 1: Physical Connection

Plug a **Thunderbolt 4 cable** between the two machines. TB4 creates a direct network bridge (32 Gbps / 4 GB/s).

---

## Step 2: Network — Find Each Other

After plugging in the TB4 cable, figure out the network interface and IPs.

**On both machines:**
```bash
ip addr | grep -E "enp|thunderbolt|inet 192|inet 10"
```

You're looking for:
- The **TB4 network interface name** (something like `enp6s0`, `thunderbolt0`, or `enpXsY`)
- The **IP address** of each machine on that interface

If TB4 creates a direct link without DHCP, manually assign IPs:
```bash
# Machine 0 (master):
sudo ip addr add 10.77.0.1/30 dev <tb4_interface>
sudo ip link set <tb4_interface> up

# Machine 1 (worker):
sudo ip addr add 10.77.0.2/30 dev <tb4_interface>
sudo ip link set <tb4_interface> up
```

Verify connectivity:
```bash
# From Machine 0:
ping 10.77.0.1    # or whatever Machine 1's IP is

# From Machine 1:
ping 10.77.0.2
```

**Record these values — you'll need them later:**
- Machine 0 IP: `10.77.0.1/30` (e.g., 192.168.1.140 or 10.0.0.1)
- Machine 1 IP: `10.77.0.2/30` (e.g., 192.168.1.XXX or 10.0.0.2)
- TB4 interface name: `tb-ddp` (e.g., enp6s0)

---

## Step 3: Passwordless SSH

`torchrun` needs to SSH between machines without a password prompt.

**From Machine 0:**
```bash
ssh-copy-id joelwang-ai-2@<machine_1_ip>
```

**Verify it works without a password:**
```bash
ssh joelwang-ai-2@<machine_1_ip> "hostname && echo OK"
```

**From Machine 1 (also needed for bidirectional torchrun):**
```bash
ssh-copy-id joelwang-ai-2@<machine_0_ip>
```

---

## Step 4: Sync Environment on Machine 1

Machine 1 needs an identical codebase, venv, and dataset.

### 4a: Sync codebase
```bash
rsync -avz ~/Desktop/ai_lab/autokernel-halo-strix/ \
    joelwang-ai-2@<machine_1_ip>:~/Desktop/ai_lab/autokernel-halo-strix/
```

### 4b: Verify Python environment matches
```bash
ssh joelwang-ai-2@<machine_1_ip> \
    "source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate && \
     python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
```

Both machines must show the same PyTorch version and `True` for CUDA.

### 4c: Copy the dataset
```bash
rsync -avP ~/Desktop/ai_lab/autokernel-halo-strix/datasets/common_crawl_sample.bin \
    joelwang-ai-2@<machine_1_ip>:~/Desktop/ai_lab/autokernel-halo-strix/datasets/
```

This is 4.7 GB — takes ~2 seconds over TB4.

### 4d: Copy the checkpoint (if resuming)
```bash
rsync -avP ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/argus_prime_cc/step_36000.pt \
    joelwang-ai-2@<machine_1_ip>:~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/argus_prime_cc/
```

---

## Step 5: Test Bandwidth

```bash
# On Machine 0: start iperf server
iperf3 -s

# On Machine 1 (separate terminal):
iperf3 -c <machine_0_ip>
```

**Expected:** ~3-4 Gbps over TB4. If significantly lower, check the cable and interface.

Stop iperf3 after testing (Ctrl+C on Machine 0).

---

## Step 6: RCCL/NCCL Environment Variables

ROCm uses RCCL (ROCm Collective Communication Library) as the NCCL backend. Over TB4 (no RDMA), it falls back to TCP sockets.

Create a file `~/ddp_env.sh` on **both machines**:

```bash
cat > ~/ddp_env.sh << 'EOF'
export NCCL_SOCKET_IFNAME=<tb4_interface>   # e.g., enp6s0
export NCCL_P2P_DISABLE=1                   # no GPU-direct (unified memory, not discrete)
export NCCL_IB_DISABLE=1                    # no InfiniBand
export NCCL_SOCKET_NTHREADS=4               # more threads for TCP
export NCCL_NSOCKS_PERTHREAD=4              # more sockets per thread
export NCCL_TIMEOUT=1800                    # 30 min timeout (default 30s is too short)
export MASTER_ADDR=<machine_0_ip>           # master machine IP
export MASTER_PORT=29500                    # any free port
EOF
```

**Replace** `<tb4_interface>` and `<machine_0_ip>` with your actual values.

---

## Step 7: Launch DDP Training

The DDP training script (`scripts/train_ddp.py`) needs to be created first — tell Claude the IPs and interface name from Step 2, and it will generate the script.

**Launch pattern (run simultaneously on both machines):**

```bash
# Machine 0 (master):
source ~/ddp_env.sh
source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    scripts/train_ddp.py \
    --model models/argus_prime.py --class-name ArgusPrime \
    --resume-from checkpoints/argus_prime_cc/step_36000.pt \
    --dataset datasets/common_crawl_sample.bin \
    --epochs 2 --compile --optimize-kernels --muon --lr 0.0012 \
    --batch-size 16 --block-size 256 --accum-steps 4 \
    --checkpoint-dir checkpoints/argus_prime_cc_ddp \
    --checkpoint-interval 18000
```

```bash
# Machine 1 (worker):
source ~/ddp_env.sh
source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    scripts/train_ddp.py \
    --model models/argus_prime.py --class-name ArgusPrime \
    --resume-from checkpoints/argus_prime_cc/step_36000.pt \
    --dataset datasets/common_crawl_sample.bin \
    --epochs 2 --compile --optimize-kernels --muon --lr 0.0012 \
    --batch-size 16 --block-size 256 --accum-steps 4 \
    --checkpoint-dir checkpoints/argus_prime_cc_ddp \
    --checkpoint-interval 18000
```

**Note:** `--node_rank=0` on master, `--node_rank=1` on worker. Everything else is identical.

---

## Step 8: Monitor Training

**Check progress (from either machine):**
```bash
tail -5 checkpoints/argus_prime_cc_ddp/train_log.jsonl
```

**Check RCCL communication (first run only):**
Add `export NCCL_DEBUG=INFO` to `ddp_env.sh` temporarily. Look for `NCCL INFO` lines showing the allreduce ring formation. Remove after confirming it works (adds log noise).

**Check GPU utilization:**
```bash
watch -n 1 rocm-smi    # on each machine
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NCCL timeout` | Increase `NCCL_TIMEOUT=3600`. Check firewall: `sudo ufw allow 29500` |
| `Connection refused` | Machine 0 must start first (or within ~30s). Check MASTER_ADDR is correct |
| Wrong interface | `NCCL_SOCKET_IFNAME` must match the TB4 interface exactly. Check with `ip addr` |
| One machine crashes | Kill both, resume from latest checkpoint on single machine |
| TB4 disconnects mid-training | Checkpoint every 1/4 epoch. Resume from last checkpoint |
| Slow sync | Increase `--accum-steps` to 8 (amortizes sync further, but larger effective batch) |

---

## Performance Reference

| Metric | 1 Machine | 2 Machines (DDP) |
|--------|-----------|-----------------|
| tok/s | 16,700 | ~29,000 |
| Effective batch | 64 | 128 |
| Gradient sync | — | ~84ms (fp16 compressed) per optimizer step |
| Memory per machine | 6.1 GB | 6.1 GB (same — full model copy) |
| Common Crawl 2.4B, 2 epochs | ~79 hrs | ~45 hrs |
