#!/bin/bash
# Sprint 3A: OdinFlat dolma-10B 1 full epoch with the LOCKED C3 recipe
# (Sprint 1 baseline + Sprint 1.5 μP + SPECTRA post-clip).
#
# Expected wall: ~61h (31.2K tok/s steady × 52K steps × 131K tokens/step).
# Pre-launch confirmation (2000 steps, ALL GATES PASS) at 00f1d82.
#
# Monitoring:
#   bash run_remote.sh 'tail -f checkpoints/sprint3a-odinflat-dolma/rank0.log'
#
# Kill signal (both machines):
#   bash run_remote.sh 'pkill -f torchrun' && bash run_remote_b.sh 'pkill -f torchrun'

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint3a-odinflat-dolma

# Resume support: if a checkpoint exists, pass it as $1 to resume.
RESUME="${1:-}"
if [ -z "$RESUME" ]; then
    if [ -d "$CKPT" ]; then
        # Fresh-start guard: don't silently overwrite an existing Sprint 3A dir.
        LATEST=$(ls "$CKPT"/step_*.pt 2>/dev/null | sort -V | tail -1 || true)
        if [ -n "$LATEST" ]; then
            echo "WARN: existing checkpoint $LATEST found."
            echo "      Pass it as arg1 to resume, or 'rm -rf $CKPT' to restart."
            exit 2
        fi
    fi
    mkdir -p "$CKPT"
fi

echo "=== Sprint 3A: OdinFlat dolma-10B 1 epoch ==="
echo "Config: C3 combined (μP + SPECTRA) on OdinFlat + dolma-10B via DDP"
echo "Checkpoint dir: $CKPT"
echo "Expected wall: ~61 hours"
[ -n "$RESUME" ] && echo "Resume from: $RESUME"
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
WARMUP_STEPS=300 \
CHECKPOINT_INTERVAL=2000 \
MAX_GRAD_NORM=1.0 \
EXTRA_FLAGS='--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 200 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 --auto-eval' \
bash scripts/launch_ddp.sh "$RESUME" > "$CKPT/launch.log" 2>&1

echo "Launch returned; tail of launch.log:"
tail -5 "$CKPT/launch.log" || true
