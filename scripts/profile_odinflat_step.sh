#!/bin/bash
# Track 1.2: Profile OdinFlat optimizer step under Sprint 1.5 C3 recipe.
#
# Launches a 50-step DDP probe with torch.profiler wrapping opt steps 30:40.
# Writes profile.json (Chrome trace) + profile-summary.txt to checkpoint dir.
# Must be launched from Machine A (it SSHs into Machine B for rank 1).
#
# Plan: docs/superpowers/plans/2026-05-10-odinflat-throughput-investigation-plan.md
# Sprint 1.5 C3 recipe per docs/perf/sprint1.5-phase-C-findings.md.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/odinflat-profile
rm -rf "$CKPT"
mkdir -p "$CKPT"

# Also mkdir on Machine B so rank1 can write its log.
ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$CKPT"

MODEL=models/odin_flat.py \
CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 BLOCK=512 BATCH=16 ACCUM=8 \
WARMUP_STEPS=100 CHECKPOINT_INTERVAL=9999 MAX_GRAD_NORM=1.0 \
NUM_WORKERS=12 \
CKPT_DIR="$CKPT" \
EXTRA_FLAGS='--max-steps 50 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 100 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 --profile-steps 30:40' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launched; tailing rank0.log..."

# Wait for Done / fail.
SECS=0
while true; do
    sleep 15
    SECS=$((SECS + 15))
    if grep -q '^Done:' "$CKPT/rank0.log" 2>/dev/null; then
        echo "DONE after ~${SECS}s"
        break
    fi
    if grep -qiE '(Traceback|Error|FAILED)' "$CKPT/rank0.log" 2>/dev/null; then
        echo "FAIL after ~${SECS}s — tail:"
        tail -30 "$CKPT/rank0.log"
        exit 1
    fi
    if [ $SECS -gt 1800 ]; then
        echo "TIMEOUT at 30min. rank0 tail:"
        tail -30 "$CKPT/rank0.log"
        echo "--- killing ---"
        pkill -f "train_ddp.py" || true
        ssh joelwang-ai-1@10.77.0.2 "pkill -f 'train_ddp.py'" || true
        exit 1
    fi
done

echo "=== Artifacts in $CKPT ==="
ls -la "$CKPT"
echo ""
echo "=== Profile summary (top 40 by cuda_time_total) ==="
head -50 "$CKPT/profile-summary.txt" || echo "(profile-summary.txt not found)"
