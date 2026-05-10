#!/bin/bash
# Track 2.a: QKV-fusion single-node throughput + loss-parity probe.
#
# Runs 200 opt steps on a single node (no DDP overhead) under the Sprint
# 1.5 C3 recipe. Logs steady-state tok/s and loss@50/100/150/200 to
# checkpoints/qkv-fusion-probe-{post,pre}/rank0.log.
#
# This script launches the POST-fusion configuration (current code).
# To compare against PRE-fusion, checkout the parent commit of the
# Track 2.a commit first and re-run with a different CKPT_DIR.
#
# Usage (from Machine A or B, single-node 1-GPU):
#   bash scripts/probe_qkv_fusion.sh
#
# Plan: docs/superpowers/plans/2026-05-10-odinflat-throughput-investigation-plan.md §4.2.6

set -euo pipefail

# Path resolver + venv activation (handles both Machine A and B layouts).
if [ -d ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix ]; then
    cd ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix
    source ~/Desktop/comfyui-rocm7.12/.venv/bin/activate
elif [ -d ~/Desktop/ai_lab/autokernel-halo-strix ]; then
    cd ~/Desktop/ai_lab/autokernel-halo-strix
    source .venv/bin/activate
fi

CKPT="${CKPT:-checkpoints/qkv-fusion-probe}"
rm -rf "$CKPT"
mkdir -p "$CKPT"

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune-no-cudagraphs}"

echo "=== QKV-fusion probe: 200-step OdinFlat single-node ==="
echo "  CKPT_DIR=$CKPT"
echo "  Commit: $(git log --oneline -1)"
echo ""

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29502 \
  scripts/train_ddp.py \
  --model models/odin_flat.py --class-name OdinFlat \
  --dataset datasets/dolma-10b-odin32k.bin --epochs 1 \
  --block-size 512 --batch-size 16 --accum-steps 8 \
  --compile --no-muon --lr 8e-4 --backend gloo \
  --warmup-steps 100 --num-workers 12 \
  --max-grad-norm 1.0 \
  --checkpoint-dir "$CKPT" --checkpoint-interval 9999 --log-interval 25 \
  --max-steps 200 \
  --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 100 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  2>&1 | tee "$CKPT/rank0.log"

echo ""
echo "=== Tail ==="
tail -15 "$CKPT/rank0.log"
