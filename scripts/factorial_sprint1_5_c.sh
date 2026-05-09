#!/bin/bash
# Sprint 1.5 Phase C compact factorial — C1 (SPECTRA), C2 (μP), C3 (both).
# 500 steps per config on OdinFlat 122M dolma-10B via DDP.
# Each run: ~36 min wall (plus ~10 min compile warmup for first).
# Total: ~2h.
#
# Baseline reference: S1.5 at step 400 was loss 4.57 with the Sprint 1
# recipe (no SPECTRA, no μP). Phase C comparison is at step 400 and 500.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

MAX_STEPS=500

wait_for_done() {
    local log="$1"
    while true; do
        sleep 30
        if [ -f "$log" ] && grep -q "^Done:" "$log"; then
            return 0
        fi
        if [ -f "$log" ] && grep -q "^Traceback" "$log" 2>/dev/null; then
            echo "FAIL: run crashed. See $log"
            return 1
        fi
    done
}

launch_config() {
    local name="$1"
    local extra="$2"
    local ckpt="checkpoints/sprint1.5-C-${name}"
    rm -rf "$ckpt"
    mkdir -p "$ckpt"

    echo
    echo "=== Phase C: ${name}  |  ${ckpt} ==="
    echo "  extra flags: ${extra}"
    echo

    CKPT_DIR="$ckpt" \
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
    EXTRA_FLAGS="--max-steps ${MAX_STEPS} --imu1-groups --normuon --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 100 ${extra}" \
    bash scripts/launch_ddp.sh > "$ckpt/launch.log" 2>&1

    echo "  launched; waiting for rank0 'Done:' marker..."
    if ! wait_for_done "$ckpt/rank0.log"; then
        echo "  FAIL: ${name} did not complete"
        return 1
    fi
    echo "  ${name} done."
}

# C1: SPECTRA-post alone (at Sprint 1 winning LR lr_2d=5e-3)
launch_config "C1-spectra-only" \
    "--lr-2d 5e-3 --spectra-post --spectra-clip-norm 1.0"

# C2: μP-partial alone (at probe winner lr_2d=5e-3; probe established
# 5e-3 is the highest stable LR at 30M and the μP prescription for 122M
# is unchanged embedding-LR / scaled-down hidden-LR which is derived
# automatically by build_mup_param_groups)
launch_config "C2-mup-only" \
    "--lr-2d 5e-3 --mup --mup-base-width 256"

# C3: Combined SPECTRA + μP at the same LR
launch_config "C3-combined" \
    "--lr-2d 5e-3 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0"

echo
echo "=== Phase C Summary ==="
echo
printf "%-20s %-14s %-14s %-14s\n" "config" "loss_step_400" "loss_step_500" "tok/s (agg)"
for name in C1-spectra-only C2-mup-only C3-combined; do
    ckpt="checkpoints/sprint1.5-C-${name}"
    log="$ckpt/rank0.log"
    LOSS_400=$(grep -E '^\[step    400\]' "$log" | sed -E 's/.*loss=([0-9.]+).*/\1/')
    LOSS_500=$(grep -E '^\[step    500\]' "$log" | sed -E 's/.*loss=([0-9.]+).*/\1/')
    TOKS=$(grep -E '^Done:' "$log" | sed -E 's/.*in [0-9]+s \(([0-9,]+) tok.*/\1/' | tr -d ',')
    printf "%-20s %-14s %-14s %-14s\n" "$name" "$LOSS_400" "$LOSS_500" "$TOKS"
done
echo
echo "S1.5 baseline reference: step 400 loss=4.57 (no SPECTRA, no μP)"
