#!/bin/bash
# T-1.4: DDP bucket_cap_mb + gradient_as_bucket_view sweep.
# Run on Machine A; SSHs into Machine B for rank 1.
#
# Sweeps:
#   bucket_cap_mb ∈ {8, 16, 25, 50, 100, 200}
#   gradient_as_bucket_view ∈ {True}  (always; memory win is free)
#
# NB: this repo uses manual allreduce (see allreduce_grads_async in
# scripts/train_ddp.py) so bucket_cap_mb only affects DDP's internal
# bucket sizing when fallback to DDP hooks occurs. The primary signal
# here is whether gradient_as_bucket_view reduces memory enough to
# enable batch=32, and whether tuned bucket behavior changes anything.

set -e

cd ~/Desktop/ai_lab/autokernel-halo-strix

mkdir -p docs/perf/t1-ddp-bucket-sweep

for BUCKET_MB in 8 25 50 100; do
  CKPT="checkpoints/t1-ddp-sweep-b${BUCKET_MB}"
  rm -rf "$CKPT"
  mkdir -p "$CKPT"
  ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$CKPT"

  echo "=== bucket_cap_mb=$BUCKET_MB ==="

  MODEL=models/odin_flat.py \
  CLASS=OdinFlat \
  DATASET=datasets/dolma-10b-odin32k.bin \
  EPOCHS=1 \
  LR=8e-4 BLOCK=512 BATCH=16 ACCUM=8 \
  WARMUP_STEPS=50 CHECKPOINT_INTERVAL=9999 MAX_GRAD_NORM=1.0 \
  NUM_WORKERS=12 \
  CKPT_DIR="$CKPT" \
  EXTRA_FLAGS="--max-steps 150 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 --assert-no-sync" \
  bash scripts/launch_ddp.sh

  # Wait for completion (rough timeout 10 min / run)
  SECS=0
  while pgrep -f "train_ddp.py" > /dev/null; do
    sleep 5
    SECS=$((SECS + 5))
    if [ $SECS -gt 600 ]; then
      echo "TIMEOUT at 10min for bucket=$BUCKET_MB"
      pkill -f "train_ddp.py" || true
      ssh joelwang-ai-1@10.77.0.2 "pkill -f 'train_ddp.py'" || true
      break
    fi
  done

  # Extract tok/s median from log
  if [ -f "$CKPT/rank0.log" ]; then
    TOK_S=$(grep -oE 'tok/s=[0-9,]+' "$CKPT/rank0.log" | tail -50 \
            | sed 's/tok\/s=//;s/,//g' | sort -n \
            | awk '{a[NR]=$1} END{if(NR>0) print a[int(NR/2)]; else print "0"}')
    echo "bucket_cap_mb=$BUCKET_MB => median tok/s=$TOK_S"
    echo "$BUCKET_MB $TOK_S" >> docs/perf/t1-ddp-bucket-sweep/results.txt
  fi
done

echo "=== Sweep done ==="
cat docs/perf/t1-ddp-bucket-sweep/results.txt
