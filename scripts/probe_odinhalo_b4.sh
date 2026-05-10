#!/bin/bash
# Track B4 (plan §6.2 promoted to critical path): OdinHalo 2000-step
# probe with --optimize-kernels under Sprint 3B locked recipe.
#
# Verifies whether OdinHalo suffers the same silent-freeze bug as OdinFlat
# at 2000-step horizon. Per Phase 0.3 probe, OdinHalo at 200 steps showed
# +38% tok/s with loss parity — but Track 3.A on OdinFlat showed the silent
# freeze only becomes measurable at 2000+ steps. If OdinHalo has the bug,
# Sprint 3B's locked recipe (--optimize-kernels ON) is a disaster.
#
# DDP across Machine A + B. Sprint 3B's locked recipe:
#   - OdinHalo, lr_2d=2e-3 (from S1.3b lock)
#   - block=256 (HALO models use shorter blocks)
#   - batch=16, accum=8
#   - --optimize-kernels ON (the question under test)
#   - --auto-eval for checkpoint scorecards at 500/1000/1500/2000
#
# Gate (applies to loss @ 2000):
#   - step_2000 train loss < 3.8 → PASS; Sprint 3B ships as locked
#   - step_2000 train loss \$\\ge\$ 3.8 → FAIL; drop --optimize-kernels for 3B
#
# Reference: Sprint 3A-confirm V1 (OdinFlat) hit loss 3.80 at step 2000
# with the silent-freeze active. V0 was 3.15. An OdinHalo silent-freeze
# would produce a similar gap.
#
# Wall: ~90 min compute + ~15 min eval overhead at checkpoints.
#
# Usage (from Machine A):
#   bash scripts/probe_odinhalo_b4.sh

set -eo pipefail  # no -u: venv's activate script references unbound vars

cd ~/Desktop/ai_lab/autokernel-halo-strix
# Activate venv so launch_ddp.sh's rank 0 torchrun command resolves on PATH.
source .venv/bin/activate

CKPT=checkpoints/odinhalo-b4-probe
rm -rf "$CKPT"
mkdir -p "$CKPT"

# Also mkdir on Machine B so rank1 can write its log.
ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$CKPT"

MODEL=models/odin_halo.py \
CLASS=OdinHalo \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=1e-3 BLOCK=256 BATCH=16 ACCUM=8 \
WARMUP_STEPS=300 CHECKPOINT_INTERVAL=500 MAX_GRAD_NORM=0.8 \
NUM_WORKERS=12 \
CKPT_DIR="$CKPT" \
MASTER_PORT=29511 \
EXTRA_FLAGS='--max-steps 2000 --optimize-kernels --imu1-groups --normuon --lr-2d 2e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 200 --auto-eval --diag-frozen-params checkpoints/odinhalo-b4-probe/diag.jsonl' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launched; rank0 log: $CKPT/rank0.log"
echo "Monitor: bash run_remote.sh 'tail -f $CKPT/rank0.log'"
