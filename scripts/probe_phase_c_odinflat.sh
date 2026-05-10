#!/bin/bash
# Phase C verification: OdinFlat 2000-step single-node probe with
# Phase B post-fix --optimize-kernels + --use-fused-zloss.
#
# Ship gate: loss@2000 within +/- 0.05 of OdinFlat V0 baseline (3.15 was
# the step-2000 loss per docs/perf/sprint3a-confirm-findings.md / Phase V).
# Tok/s should match or exceed the 31.3K baseline.
#
# Runs on whichever machine has a free GPU (A or B, single-node).

set -eo pipefail

if [ -d ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix ]; then
    cd ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix
    source ~/Desktop/comfyui-rocm7.12/.venv/bin/activate
elif [ -d ~/Desktop/ai_lab/autokernel-halo-strix ]; then
    cd ~/Desktop/ai_lab/autokernel-halo-strix
    source .venv/bin/activate
fi

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune-no-cudagraphs}"

CKPT="${CKPT:-checkpoints/phase-c-odinflat-postfix-v2}"
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "=== Phase C v2: OdinFlat 2000-step, post-B-fix, --optimize-kernels ONLY (no --use-fused-zloss) ==="
echo "  CKPT: $CKPT"
echo "  commit: $(git log --oneline -1)"
echo "  V1 (original Phase C with --use-fused-zloss) diverged at step 250. Isolating cause."
echo ""

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29520 \
  scripts/train_ddp.py \
  --model models/odin_flat.py --class-name OdinFlat \
  --dataset datasets/dolma-10b-odin32k.bin --epochs 1 \
  --block-size 512 --batch-size 16 --accum-steps 8 \
  --compile --no-muon --lr 8e-4 --backend gloo \
  --warmup-steps 300 --num-workers 12 \
  --max-grad-norm 1.0 \
  --checkpoint-dir "$CKPT" --checkpoint-interval 500 --log-interval 50 \
  --max-steps 2000 \
  --optimize-kernels \
  --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --diag-frozen-params "$CKPT/diag.jsonl" \
  2>&1 | tee "$CKPT/rank0.log"

echo ""
echo "=== Tail ==="
grep -E '^\[step|^Done:' "$CKPT/rank0.log" | tail -10
