#!/bin/bash
# Launch DDP training on both machines simultaneously, fully detached.
# Run this from Machine A only.
# Usage:
#   bash scripts/launch_ddp.sh              # fresh run (defaults: OdinFlat on wikitext)
#   bash scripts/launch_ddp.sh <resume.pt>  # resume from checkpoint
#
# Override model/config via env vars:
#   MODEL=models/odin_halo.py CLASS=OdinHalo \
#     CKPT_DIR=checkpoints/odin-halo-wikitext-ddp \
#     bash scripts/launch_ddp.sh
#
# Both ranks run under nohup + setsid; safe to disconnect SSH after launch.
# Monitor via: tail -f $CKPT_DIR/rank0.log

set -e

RESUME_CKPT="${1:-}"

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune-no-cudagraphs}"
export GLOO_SOCKET_IFNAME=thunderbolt0
export NCCL_SOCKET_IFNAME=thunderbolt0
export MASTER_ADDR=10.77.0.1
export MASTER_PORT="${MASTER_PORT:-29500}"

MODEL="${MODEL:-models/odin_flat.py}"
CLASS="${CLASS:-OdinFlat}"
DATASET="${DATASET:-datasets/wikitext-103-odin32k.bin}"
CKPT_DIR="${CKPT_DIR:-checkpoints/odin-flat-wikitext-ddp}"
EPOCHS="${EPOCHS:-1}"
BLOCK="${BLOCK:-256}"
BATCH="${BATCH:-16}"
ACCUM="${ACCUM:-8}"
LR="${LR:-8e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

mkdir -p "$CKPT_DIR"

RESUME_ARG=""
if [ -n "$RESUME_CKPT" ]; then
  RESUME_ARG="--resume-from $RESUME_CKPT"
  echo "  Resume: $RESUME_CKPT"
fi

echo "=== Launching DDP training: $CLASS on wikitext-103 ==="
echo "  Master: $MASTER_ADDR:$MASTER_PORT (thunderbolt0)"
echo "  Model: $MODEL ($CLASS)"
echo "  Dataset: $DATASET"
echo "  Config: batch=${BATCH}x${ACCUM}x2=$((BATCH*ACCUM*2)), block=$BLOCK, lr=$LR"
echo "  Warmup: $WARMUP_STEPS steps | ckpt every: $CHECKPOINT_INTERVAL | workers: $NUM_WORKERS | grad_clip: $MAX_GRAD_NORM"
echo ""

# Launch rank 1 (Machine B) via SSH — fully detached with nohup + setsid
echo "[1/2] Launching rank 1 on Machine B (10.77.0.2)..."
ssh joelwang-ai-1@10.77.0.2 "
  source ~/Desktop/comfyui-rocm7.12/.venv/bin/activate
  cd ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix
  export HSA_OVERRIDE_GFX_VERSION=11.5.1
  export TORCH_COMPILE_MODE=$TORCH_COMPILE_MODE
  export GLOO_SOCKET_IFNAME=thunderbolt0
  export MASTER_ADDR=10.77.0.1
  export MASTER_PORT=$MASTER_PORT
  mkdir -p $CKPT_DIR
  setsid nohup torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT \
    scripts/train_ddp.py \
    --model $MODEL --class-name $CLASS \
    --dataset $DATASET --epochs $EPOCHS \
    --block-size $BLOCK --batch-size $BATCH --accum-steps $ACCUM \
    --compile --no-muon --lr $LR --backend gloo \
    --warmup-steps $WARMUP_STEPS --num-workers $NUM_WORKERS \
    --max-grad-norm $MAX_GRAD_NORM \
    --checkpoint-dir $CKPT_DIR --checkpoint-interval $CHECKPOINT_INTERVAL --log-interval 50 \
    $RESUME_ARG \
    > $CKPT_DIR/rank1.log 2>&1 < /dev/null &
  disown
  echo LAUNCHED_B_pid=\$!
"

sleep 3  # Let rank 1 start and enter rendezvous wait

# Launch rank 0 (this machine) — fully detached
echo "[2/2] Launching rank 0 on Machine A (10.77.0.1)..."
setsid nohup torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  scripts/train_ddp.py \
  --model $MODEL --class-name $CLASS \
  --dataset $DATASET --epochs $EPOCHS \
  --block-size $BLOCK --batch-size $BATCH --accum-steps $ACCUM \
  --compile --no-muon --lr $LR --backend gloo \
  --warmup-steps $WARMUP_STEPS --num-workers $NUM_WORKERS \
  --max-grad-norm $MAX_GRAD_NORM \
  --checkpoint-dir $CKPT_DIR --checkpoint-interval $CHECKPOINT_INTERVAL --log-interval 50 \
  $RESUME_ARG \
  > $CKPT_DIR/rank0.log 2>&1 < /dev/null &
disown

echo "LAUNCHED_A_pid=$!"
echo ""
echo "=== Both ranks launched (detached) ==="
echo "Monitor: bash run_remote.sh 'tail -f $CKPT_DIR/rank0.log'"
echo "Killsig: bash run_remote.sh 'pkill -f torchrun' && bash run_remote_b.sh 'pkill -f torchrun'"
