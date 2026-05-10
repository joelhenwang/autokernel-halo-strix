#!/bin/bash
# T-2.1: batch=32/accum=4 memory feasibility probe.
#
# Tests whether OdinFlat fits at batch=32 accum=4 (effective batch=256 same
# as production). Per DDP sweep (2026-05-06), batch=32 gave +5% throughput
# but doubled memory to ~10 GB/node. Combined with gradient_as_bucket_view
# this may be free.
#
# Run matrix:
#   E4-A baseline batch=16 accum=8 (control)
#   E4-B batch=32 accum=4, no checkpoint, if fits
#   E4-C batch=32 accum=4 + --ak-ddp-tune (gradient_as_bucket_view)
#   E4-D batch=32 accum=4 + --ak-sync-cleanup (includes spectra_branchless
#        + deferred loss accumulator + env var spectra branchless)
#
# 200-step smoke per config. Exit gate: ≥3% tok/s gain with clean
# GradScaler behavior.

set -e

cd ~/Desktop/ai_lab/autokernel-halo-strix

mkdir -p docs/perf/t2-1-batch32-probe

RESULTS=docs/perf/t2-1-batch32-probe/results.md
echo "# T-2.1 batch=32/accum=4 probe" > "$RESULTS"
echo "" >> "$RESULTS"
echo "| Config | batch | accum | tok/s | peak_mem_gb | scale | status |" >> "$RESULTS"
echo "|---|---:|---:|---:|---:|---:|---|" >> "$RESULTS"

run_config() {
  local name="$1"
  local batch="$2"
  local accum="$3"
  local extra_flags="$4"
  local ckpt="checkpoints/t2-1-${name}"
  rm -rf "$ckpt"
  mkdir -p "$ckpt"
  ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$ckpt"

  echo "=== ${name}: batch=${batch} accum=${accum} ${extra_flags} ==="

  MODEL=models/odin_flat.py \
  CLASS=OdinFlat \
  DATASET=datasets/dolma-10b-odin32k.bin \
  EPOCHS=1 \
  LR=8e-4 BLOCK=512 BATCH="$batch" ACCUM="$accum" \
  WARMUP_STEPS=50 CHECKPOINT_INTERVAL=9999 MAX_GRAD_NORM=1.0 \
  NUM_WORKERS=12 \
  CKPT_DIR="$ckpt" \
  EXTRA_FLAGS="--max-steps 200 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 ${extra_flags}" \
  bash scripts/launch_ddp.sh >/dev/null 2>&1

  # Wait for completion
  local SECS=0
  while pgrep -f "train_ddp.py" > /dev/null; do
    sleep 5
    SECS=$((SECS + 5))
    if [ $SECS -gt 900 ]; then
      echo "TIMEOUT at 15min for ${name}"
      pkill -f "train_ddp.py" || true
      ssh joelwang-ai-1@10.77.0.2 "pkill -f 'train_ddp.py'" || true
      break
    fi
  done

  # Extract median tok/s from last 10 log lines
  if [ -f "$ckpt/rank0.log" ]; then
    local TOK_S=$(grep -oE 'tok/s=[0-9,]+' "$ckpt/rank0.log" | tail -10 \
                  | sed 's/tok\/s=//;s/,//g' | sort -n \
                  | awk '{a[NR]=$1} END{if(NR>0) print a[int(NR/2)]; else print "FAIL"}')
    local MEM=$(grep -oE 'mem=[0-9.]+GB' "$ckpt/rank0.log" | tail -1 | sed 's/mem=//;s/GB//')
    local SCALE=$(grep -oE 'scale=[0-9.e+-]+' "$ckpt/rank0.log" | tail -1 | sed 's/scale=//')
    local STATUS="PASS"
    if grep -q "OOM\|nonfinite\|RuntimeError" "$ckpt/rank0.log" 2>/dev/null; then
      STATUS="FAIL"
    fi
    echo "| ${name} | ${batch} | ${accum} | ${TOK_S} | ${MEM} | ${SCALE} | ${STATUS} |" >> "$RESULTS"
    echo "Result: tok/s=${TOK_S} mem=${MEM}GB status=${STATUS}"
  else
    echo "| ${name} | ${batch} | ${accum} | NO_LOG | - | - | FAIL |" >> "$RESULTS"
  fi
}

run_config "baseline"           16  8  ""
run_config "batch32_plain"      32  4  ""
run_config "batch32_ddp_tune"   32  4  "--ak-ddp-tune"
run_config "batch32_sync_clean" 32  4  "--ak-ddp-tune --ak-sync-cleanup"

echo "=== T-2.1 DONE ==="
cat "$RESULTS"
