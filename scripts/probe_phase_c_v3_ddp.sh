#!/bin/bash
# Phase C v3: OdinFlat 2000-step DDP verification probe.
#
# History: v1 (with --use-fused-zloss) and v2 (without) both diverged at
# step 250 under single-node batch=128. Per phase-c-divergence-analysis.md,
# the root cause is batch-size + LR interaction — not Phase B fixes.
#
# v3 matches Sprint 3A-confirm's validated config: DDP batch=16x8x2=256
# at lr_2d=5e-3. Expected to train cleanly to loss ~3.15 at step 2000.
#
# Ship gate:
#   - loss@step_2000 within +/- 0.05 of 3.15 baseline
#   - tok/s aggregate within +/- 10% of 31.3K
#   - diag-frozen-params shows ZERO always_none/always_zero params (the
#     Phase B fix is validated empirically at scale).
#
# Launches BOTH machines via launch_ddp.sh (rank 1 on Machine B via SSH).
#
# Run from Machine A only.

set -eo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix
source .venv/bin/activate

CKPT="${CKPT:-checkpoints/phase-c-v3-ddp}"
rm -rf "$CKPT"
mkdir -p "$CKPT"

# Also pre-create the checkpoint dir on Machine B.
ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$CKPT"

echo "=== Phase C v3: OdinFlat 2000-step DDP (both machines) ==="
echo "  CKPT: $CKPT"
echo "  commit: $(git log --oneline -1)"
echo "  target: loss<=3.20 @ step 2000, tok/s>=28K aggregate"
echo ""

MODEL=models/odin_flat.py \
CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 BLOCK=512 BATCH=16 ACCUM=8 \
WARMUP_STEPS=300 CHECKPOINT_INTERVAL=500 MAX_GRAD_NORM=1.0 \
NUM_WORKERS=12 \
CKPT_DIR="$CKPT" \
MASTER_PORT=29521 \
EXTRA_FLAGS='--max-steps 2000 --optimize-kernels --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 200 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 --diag-frozen-params '"$CKPT"'/diag.jsonl --auto-eval' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launched; rank0 log: $CKPT/rank0.log"
echo "Monitor: bash run_remote.sh 'tail -f $CKPT/rank0.log'"
