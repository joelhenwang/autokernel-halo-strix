#!/bin/bash
# Sprint 1.5 Phase B.4 fallback: smaller-LR probe for μP on OdinFlat30M.
# Primary probe diverged on all 4 LRs in {0.015, 0.020, 0.0235, 0.030}.
# This fallback sweeps {0.005, 0.008, 0.010, 0.012} for 300 steps each.
# Runtime: ~6 min per run × 4 = ~25 min.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

LRS=(0.005 0.008 0.010 0.012)
MAX_STEPS=300

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600

for LR in "${LRS[@]}"; do
    CKPT="checkpoints/sprint1.5-B-mup30m-fallback-lr${LR}"
    rm -rf "$CKPT"
    mkdir -p "$CKPT"

    echo
    echo "=== μP 30M probe FALLBACK, lr_2d = ${LR}  |  ${CKPT} ==="
    echo

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
printf "%-8s %-12s %-12s %-8s\n" "LR" "last_loss" "final_scale" "alive?"
BEST_LR=0.005
BEST_LOSS=999
for LR in "${LRS[@]}"; do
    CKPT="checkpoints/sprint1.5-B-mup30m-fallback-lr${LR}"
    LOG="$CKPT/run.log"
    if [ -f "$LOG" ]; then
        LAST_LINE=$(grep -E '^\[step' "$LOG" | tail -1)
        LAST_LOSS=$(echo "$LAST_LINE" | sed -E 's/.*loss=([0-9.]+).*/\1/')
        LAST_SCALE=$(echo "$LAST_LINE" | sed -E 's/.*scale=([^ ]+).*/\1/')
        # Alive = scale not zero
        if [ "$LAST_SCALE" = "0.0e+00" ] || [ -z "$LAST_SCALE" ]; then
            ALIVE="DEAD"
        else
            ALIVE="ok"
        fi
        printf "%-8s %-12s %-12s %-8s\n" "$LR" "$LAST_LOSS" "$LAST_SCALE" "$ALIVE"
        if [ "$ALIVE" = "ok" ] && (( $(echo "$LAST_LOSS < $BEST_LOSS" | bc -l) )); then
            BEST_LOSS=$LAST_LOSS
            BEST_LR=$LR
        fi
    fi
done
echo
echo "WINNER (alive runs only): lr_2d=${BEST_LR} (loss=${BEST_LOSS})"
