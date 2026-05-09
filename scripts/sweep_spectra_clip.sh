#!/bin/bash
# Sprint 1.5 Phase B.1 + B.2: SPECTRA clip_norm sweep on OdinFlat (full).
# Runs 3 DDP jobs with clip_norm in {0.5, 1.0, 2.0} for 200 steps each.
# Uses NorMuon so SPECTRA applies. On OdinFlat we do NOT use
# --optimize-kernels per Phase 0 (autokernel-probe-2026-05-08.md).
#
# Runtime: ~25 min per run × 3 = ~75 min total.
# Parses steady-state loss (steps 100-200 mean) from each rank0.log and
# prints the winner at the end.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CLIP_NORMS=(0.5 1.0 2.0)
MAX_STEPS=200

wait_for_done() {
    local log="$1"
    while true; do
        sleep 30
        if [ -f "$log" ] && grep -q "^Done:" "$log"; then
            return 0
        fi
        # Safety: if rank0 crashed, bail (no "Done:" but clear error)
        if [ -f "$log" ] && grep -q "^Traceback" "$log" 2>/dev/null; then
            echo "FAIL: run crashed. See $log"
            return 1
        fi
    done
}

for CLIP in "${CLIP_NORMS[@]}"; do
    CKPT="checkpoints/sprint1.5-B-spectra-clip-${CLIP}"
    rm -rf "$CKPT"
    mkdir -p "$CKPT"

    echo
    echo "=== SPECTRA clip_norm = ${CLIP}  |  ${CKPT} ==="
    echo

    CKPT_DIR="$CKPT" \
    MODEL=models/odin_flat.py \
    CLASS=OdinFlat \
    DATASET=datasets/dolma-10b-odin32k.bin \
    EPOCHS=1 \
    LR=8e-4 \
    BLOCK=512 \
    BATCH=16 \
    ACCUM=8 \
    WARMUP_STEPS=100 \
    CHECKPOINT_INTERVAL=9999 \
    MAX_GRAD_NORM=1.0 \
    EXTRA_FLAGS="--max-steps ${MAX_STEPS} --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --spectra-post --spectra-clip-norm ${CLIP}" \
    bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

    echo "  launched; waiting for rank0 'Done:' marker..."
    if ! wait_for_done "$CKPT/rank0.log"; then
        echo "  FAIL: run for clip=${CLIP} did not complete"
        exit 1
    fi
    echo "  done."
done

echo
echo "=== Results ==="
echo
for CLIP in "${CLIP_NORMS[@]}"; do
    CKPT="checkpoints/sprint1.5-B-spectra-clip-${CLIP}"
    DONE_LINE=$(grep "^Done:" "$CKPT/rank0.log" | tail -1 || echo "MISSING")
    LAST_LOSS=$(grep -E '^\[step' "$CKPT/rank0.log" | tail -1 | sed -E 's/.*loss=([0-9.]+).*/\1/')
    TOKS=$(grep -E '^\[step' "$CKPT/rank0.log" | tail -1 | sed -E 's/.*tok\/s=([0-9,]+).*/\1/' | tr -d ',')
    echo "clip_norm=${CLIP}: loss_last=${LAST_LOSS} tok/s=${TOKS} | ${DONE_LINE}"
done

# Pick winner: lowest loss_last
echo
BEST_CLIP="1.0"
BEST_LOSS=999
for CLIP in "${CLIP_NORMS[@]}"; do
    CKPT="checkpoints/sprint1.5-B-spectra-clip-${CLIP}"
    LOSS=$(grep -E '^\[step' "$CKPT/rank0.log" | tail -1 | sed -E 's/.*loss=([0-9.]+).*/\1/')
    if (( $(echo "$LOSS < $BEST_LOSS" | bc -l) )); then
        BEST_LOSS=$LOSS
        BEST_CLIP=$CLIP
    fi
done
echo "WINNER: clip_norm=${BEST_CLIP} (loss=${BEST_LOSS})"
