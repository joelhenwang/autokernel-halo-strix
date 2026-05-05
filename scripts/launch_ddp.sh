#!/bin/bash
# Launch DDP training on both machines simultaneously.
# Run this from Machine A only.
# Usage: bash scripts/launch_ddp.sh

set -e

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
export GLOO_SOCKET_IFNAME=thunderbolt0
export NCCL_SOCKET_IFNAME=thunderbolt0
export MASTER_ADDR=10.77.0.1
export MASTER_PORT=29500

MODEL="models/odin_flat.py"
CLASS="OdinFlat"
DATASET="datasets/wikitext-103-odin32k.bin"
CKPT_DIR="checkpoints/odin-flat-wikitext-ddp"
EPOCHS=1
BLOCK=256
BATCH=16
ACCUM=8
LR="8e-4"

mkdir -p "$CKPT_DIR"

echo "=== Launching DDP training: $CLASS on wikitext-103 ==="
echo "  Master: $MASTER_ADDR:$MASTER_PORT (thunderbolt0)"
echo "  Model: $MODEL ($CLASS)"
echo "  Dataset: $DATASET"
echo "  Config: batch=${BATCH}x${ACCUM}x2=$((BATCH*ACCUM*2)), block=$BLOCK, lr=$LR"
echo ""

# Launch rank 1 (Machine B) first via SSH — it will wait for master
echo "[1/2] Launching rank 1 on Machine B (10.77.0.2)..."
ssh joelwang-ai-1@10.77.0.2 "
  source ~/Desktop/comfyui-rocm7.12/.venv/bin/activate
  cd ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix
  export HSA_OVERRIDE_GFX_VERSION=11.5.1
  export TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
  export GLOO_SOCKET_IFNAME=thunderbolt0
  export MASTER_ADDR=10.77.0.1
  export MASTER_PORT=29500
  mkdir -p $CKPT_DIR
  nohup torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT \
    scripts/train_ddp.py \
    --model $MODEL --class-name $CLASS \
    --dataset $DATASET --epochs $EPOCHS \
    --block-size $BLOCK --batch-size $BATCH --accum-steps $ACCUM \
    --compile --no-muon --lr $LR --backend gloo \
    --checkpoint-dir $CKPT_DIR --checkpoint-interval 500 --log-interval 50 \
    > $CKPT_DIR/rank1.log 2>&1 &
  echo \$!
" &
RANK1_SSH_PID=$!

sleep 3  # Give rank 1 time to start and begin waiting for rendezvous

# Launch rank 0 (this machine) — this is the master
echo "[2/2] Launching rank 0 on Machine A (10.77.0.1)..."
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  scripts/train_ddp.py \
  --model $MODEL --class-name $CLASS \
  --dataset $DATASET --epochs $EPOCHS \
  --block-size $BLOCK --batch-size $BATCH --accum-steps $ACCUM \
  --compile --no-muon --lr $LR --backend gloo \
  --checkpoint-dir $CKPT_DIR --checkpoint-interval 500 --log-interval 50 \
  2>&1 | tee $CKPT_DIR/rank0.log

echo ""
echo "=== DDP Training Complete ==="
wait $RANK1_SSH_PID 2>/dev/null
