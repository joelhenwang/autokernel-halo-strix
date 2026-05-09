#!/bin/bash
# Sprint 1.5 Phase B.3 + B.4: μP 30M LR probe on OdinFlat30M.
# 4 single-node torchrun jobs with lr_2d ∈ {0.015, 0.020, 0.0235, 0.030}
# for 500 steps each, using scripts/train_ddp.py (which has Sprint 1.5 flags).
# Single-node means nnodes=1; gloo backend; dolma-10B or wikitext dataset.
# Use wikitext (smaller) since this is a short probe.
#
# Runtime: ~8 min per run × 4 = ~32 min.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

LRS=(0.015 0.020 0.0235 0.030)
MAX_STEPS=500

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600

for LR in "${LRS[@]}"; do
    CKPT="checkpoints/sprint1.5-B-mup30m-lr${LR}"
    rm -rf "$CKPT"
    mkdir -p "$CKPT"

    echo
    echo "=== μP 30M probe, lr_2d = ${LR}  |  ${CKPT} ==="
    echo

    # Single-node torchrun (nnodes=1)
    torchrun --nnodes=1 --nproc_per_node=1 \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        scripts/train_ddp.py \
        --model models/odin_flat_30m.py --class-name OdinFlat30M \
        --dataset datasets/wikitext-103-odin32k.bin \
        --epochs 1 \
        --block-size 512 --batch-size 16 --accum-steps 4 \
        --compile --no-muon --lr 8e-4 --backend gloo \
        --warmup-steps 50 --num-workers 4 --max-grad-norm 1.0 \
        --checkpoint-dir "$CKPT" --checkpoint-interval 9999 --log-interval 25 \
        --max-steps ${MAX_STEPS} \
        --imu1-groups --normuon \
        --mup --mup-base-width 256 \
        --lr-2d ${LR} --lr-1d 8e-4 \
        --intra-doc-mask --value-residuals --head-gating \
        > "$CKPT/run.log" 2>&1 || echo "  WARNING: run for lr=${LR} exited non-zero"
    echo "  done."
done

echo
echo "=== Results ==="
echo
BEST_LR=0.0235
BEST_LOSS=999
for LR in "${LRS[@]}"; do
    CKPT="checkpoints/sprint1.5-B-mup30m-lr${LR}"
    LOG="$CKPT/run.log"
    if [ -f "$LOG" ]; then
        LAST_LOSS=$(grep -E '^\[step' "$LOG" | tail -1 | sed -E 's/.*loss=([0-9.]+).*/\1/' 2>/dev/null || echo "NAN")
        if [ "$LAST_LOSS" != "NAN" ] && (( $(echo "$LAST_LOSS < $BEST_LOSS" | bc -l 2>/dev/null) )); then
            BEST_LOSS=$LAST_LOSS
            BEST_LR=$LR
        fi
        echo "lr_2d=${LR}: loss_last=${LAST_LOSS}"
    else
        echo "lr_2d=${LR}: LOG MISSING"
    fi
done
echo
echo "WINNER: lr_2d=${BEST_LR} (loss=${BEST_LOSS})"
