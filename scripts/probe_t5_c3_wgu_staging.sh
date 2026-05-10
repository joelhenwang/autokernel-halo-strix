#!/bin/bash
# T-5 C.3: w_gate_up update staging probe (0.25 -> 1.0 ramp over 1000 steps).
#
# Goal: test whether graduated application of NorMuon updates to w_gate_up
# parameters prevents the fp16 overflow cascade. These 44M parameters at
# fresh init produce out-of-equilibrium gradient statistics; NorMuon
# amplifies large updates which can overflow fp16 through SwiGLU chain.
#
# Mechanism: multiply NorMuon's update to w_gate_up params by `scale` where
# `scale` starts at 0.25 and ramps linearly to 1.0 over 1000 steps.
# Implemented in halo_training/normuon.py via w_gate_up_scale/ramp_steps kwargs.
#
# 1500-step DDP run from scratch with full optimized stack + staging.
#
# Decision (v3 pre-committed):
#   passes smoothly 1500 steps             -> staging works, fold into Stack D
#   passes 500 then diverges after ramp    -> ramp needs longer or cap stays permanently lower
#   fails early (step 200-400)             -> mechanism is NOT ramp alone

set -e
cd ~/Desktop/ai_lab/autokernel-halo-strix

mkdir -p docs/perf/t5-c3-wgu-staging
RESULTS=docs/perf/t5-c3-wgu-staging/results.md
echo "# T-5 C.3: w_gate_up staging (0.25 -> 1.0 over 1000 steps)" > "$RESULTS"
echo "" >> "$RESULTS"

CKPT=checkpoints/t5-c3-wgu-staging
rm -rf "$CKPT"
mkdir -p "$CKPT"
ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$CKPT"

BASE_FLAGS="--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0"
OPT_STACK="--use-fused-zloss --ak-loss-zloss --ak-fix-rope-gate-op --ak-causal-conv-shim --ak-spectra-branchless --ak-sync-cleanup --ak-normuon-telemetry"
STAGE_FLAGS="--ak-w-gate-up-scale 0.25 --ak-w-gate-up-ramp-steps 1000"

echo "=== T-5 C.3: full opt stack + w_gate_up staging (1500 steps) ==="

MODEL=models/odin_flat.py \
CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 BLOCK=512 BATCH=16 ACCUM=8 \
WARMUP_STEPS=300 CHECKPOINT_INTERVAL=500 MAX_GRAD_NORM=1.0 \
NUM_WORKERS=12 \
CKPT_DIR="$CKPT" \
EXTRA_FLAGS="--max-steps 1500 ${BASE_FLAGS} ${OPT_STACK} ${STAGE_FLAGS}" \
bash scripts/launch_ddp.sh >/dev/null 2>&1 &

SECS=0
while pgrep -f "train_ddp.py" > /dev/null; do
  sleep 15
  SECS=$((SECS + 15))
  if [ $SECS -gt 7200 ]; then
    echo "TIMEOUT at 120min"
    pkill -f "train_ddp.py" || true
    ssh joelwang-ai-1@10.77.0.2 "pkill -f 'train_ddp.py'" || true
    break
  fi
done

if [ -f "$CKPT/rank0.log" ]; then
  TOK_S=$(grep -oE 'tok/s=[0-9,]+' "$CKPT/rank0.log" | tail -10 \
          | sed 's/tok\/s=//;s/,//g' | sort -n \
          | awk '{a[NR]=$1} END{if(NR>0) print a[int(NR/2)]; else print "FAIL"}')
  LOSS=$(grep -oE 'loss=[0-9.]+' "$CKPT/rank0.log" | tail -1 | sed 's/loss=//')
  DIV_STEP=$(grep -oE '(StabilityGuard.*step [0-9]+|nonfinite.*step [0-9]+)' "$CKPT/rank0.log" \
            | head -1 | grep -oE '[0-9]+' | head -1 || echo "none")

  STATUS="PASS"
  CLASSIFY="smooth 1500 steps"
  if [ "$DIV_STEP" != "none" ] && [ -n "$DIV_STEP" ]; then
    STATUS="DIVERGED"
    if [ "$DIV_STEP" -lt 500 ]; then
      CLASSIFY="early divergence (step <500) — mechanism is NOT ramp alone"
    elif [ "$DIV_STEP" -lt 1000 ]; then
      CLASSIFY="mid-ramp divergence — ramp needs longer or cap permanent"
    else
      CLASSIFY="post-ramp divergence (step >=1000) — 0.25 mask hiding problem; full-scale update still unsafe"
    fi
  fi

  cat >> "$RESULTS" <<EOF
## Results

| metric | value |
|---|---|
| tok/s (median last 10) | ${TOK_S} |
| loss (final) | ${LOSS} |
| divergence step | ${DIV_STEP} |
| status | ${STATUS} |
| classification | ${CLASSIFY} |

## Interpretation

EOF
  if [ "$STATUS" = "PASS" ]; then
    echo "Staging WORKS. Fold into Stack D." >> "$RESULTS"
  else
    echo "Staging insufficient: ${CLASSIFY}." >> "$RESULTS"
  fi
  cat "$RESULTS"
else
  echo "FAIL: no rank0.log" >> "$RESULTS"
  echo "FAIL: no rank0.log"
fi

echo "=== T-5 C.3 DONE ==="
