#!/bin/bash
# Track 3.A: frozen-params diagnostic runner.
#
# Launches a single-node single-GPU 50-step OdinFlat run with
# --diag-frozen-params. Three configs supported via CONFIG env var:
#
#   CONFIG=V0  → no --optimize-kernels (baseline; all grads should flow)
#   CONFIG=V1  → --optimize-kernels (rmsnorm HIP autograd-fixed + silu HIP
#                                    with broken autograd → w_gate_up freeze)
#   CONFIG=V3  → --optimize-kernels --autokernel-exclude fused_silu_gate_mul
#                (only rmsnorm HIP; should show full gradient flow if the
#                Phase III rmsnorm autograd fix is complete)
#
# Intentionally single-node (1 GPU) because grad-flow detection does not
# depend on DDP all-reduce. Cheap (~2 min compute each). Run sequentially
# on Machine B while Machine A does other work.
#
# Usage:
#   CONFIG=V0 bash scripts/diag_frozen_params_run.sh
#   CONFIG=V1 bash scripts/diag_frozen_params_run.sh
#   CONFIG=V3 bash scripts/diag_frozen_params_run.sh
#
# Plan: docs/superpowers/plans/2026-05-10-odinflat-throughput-investigation-plan.md

set -euo pipefail

CONFIG="${CONFIG:?CONFIG must be V0, V1, or V3}"

# The host may invoke this under run_remote_b.sh (chdir already done) or
# directly on Machine A (chdir required). Resolve either.
if [ -d ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix ]; then
    cd ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix
elif [ -d ~/Desktop/ai_lab/autokernel-halo-strix ]; then
    cd ~/Desktop/ai_lab/autokernel-halo-strix
fi

CKPT="checkpoints/diag-frozen-$CONFIG"
rm -rf "$CKPT"
mkdir -p "$CKPT"

case "$CONFIG" in
  V0)
    CONFIG_FLAGS=""
    ;;
  V1)
    CONFIG_FLAGS="--optimize-kernels"
    ;;
  V3)
    CONFIG_FLAGS="--optimize-kernels --autokernel-exclude fused_silu_gate_mul"
    ;;
  *)
    echo "Unknown CONFIG=$CONFIG (expected V0, V1, V3)" >&2
    exit 2
    ;;
esac

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune-no-cudagraphs}"

echo "=== diag-frozen-$CONFIG: 50-step single-node OdinFlat probe ==="
echo "  Config flags: $CONFIG_FLAGS"
echo "  Output: $CKPT/diag.jsonl"
echo ""

# Use torchrun with nproc=1,nnodes=1 so DDP init still works (train_ddp uses it).
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29501 \
  scripts/train_ddp.py \
  --model models/odin_flat.py --class-name OdinFlat \
  --dataset datasets/dolma-10b-odin32k.bin --epochs 1 \
  --block-size 512 --batch-size 16 --accum-steps 8 \
  --compile --no-muon --lr 8e-4 --backend gloo \
  --warmup-steps 100 --num-workers 12 \
  --max-grad-norm 1.0 \
  --checkpoint-dir "$CKPT" --checkpoint-interval 9999 --log-interval 10 \
  --max-steps 50 \
  --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 50 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --diag-frozen-params "$CKPT/diag.jsonl" \
  $CONFIG_FLAGS \
  2>&1 | tee "$CKPT/run.log"

echo ""
echo "=== diag.jsonl lines ==="
wc -l "$CKPT/diag.jsonl"
echo "=== first line ==="
head -1 "$CKPT/diag.jsonl" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); print(f'step={d[\"step\"]}, n_params={len(d[\"params\"])}')"
