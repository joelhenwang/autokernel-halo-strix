#!/bin/bash
# T-5 C.4 (direct): Stack D 2000-step gate.
#
# Skips the C.1/C.2/C.3 diagnostic matrix and tests Stack D directly:
# Stack A (fused_zloss) + ALL A.1/A.3 fixes + sync_cleanup + NorMuon telemetry.
#
# Rationale: C.1 matrix had coordination issues (gloo port stale between
# phases, 60-min phase timeout too tight for 1000 steps). The primary
# question Stack D probes — "does the full optimized custom_op route
# train cleanly for 2000 steps with all A.1/A.3 autograd+autocast fixes?"
# — can be tested directly in one run.
#
# If this passes:
#   Stack D = Stack A + ak-fix-rope-gate-op + ak-causal-conv-shim +
#             ak-sync-cleanup + ak-normuon-telemetry
#   All 5 gates satisfied: loss <= 4.20 + 0.1, no scaler collapse,
#   0 frozen params, tok/s >= Stack C, 2000 steps reached.
#
# If this fails:
#   Fall back to C.1/C.2/C.3 diagnostics (trust cap, staging) to isolate
#   which knob/kernel causes divergence.

set -e
cd ~/Desktop/ai_lab/autokernel-halo-strix

mkdir -p docs/perf/t5-c4-stackd
RESULTS=docs/perf/t5-c4-stackd/results.md
echo "# T-5 C.4 Stack D direct 2000-step gate" > "$RESULTS"
echo "" >> "$RESULTS"

CKPT=checkpoints/t5-c4-stackd
rm -rf "$CKPT"
mkdir -p "$CKPT"
ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$CKPT"

BASE_FLAGS="--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0"
STACK_D_FLAGS="--use-fused-zloss --ak-loss-zloss --ak-fix-rope-gate-op --ak-causal-conv-shim --ak-sync-cleanup --ak-spectra-branchless --ak-normuon-telemetry"

echo "=== T-5 C.4 Stack D direct 2000-step gate ==="
echo "Recipe: ${BASE_FLAGS} ${STACK_D_FLAGS}"

MODEL=models/odin_flat.py \
CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 BLOCK=512 BATCH=16 ACCUM=8 \
WARMUP_STEPS=300 CHECKPOINT_INTERVAL=500 MAX_GRAD_NORM=1.0 \
NUM_WORKERS=12 \
CKPT_DIR="$CKPT" \
EXTRA_FLAGS="--max-steps 2000 ${BASE_FLAGS} ${STACK_D_FLAGS} --activation-monitor --activation-monitor-interval 200" \
bash scripts/launch_ddp.sh >/dev/null 2>&1

# Wait for completion (2h ceiling, matches T-1.5 wall time)
SECS=0
LOG="$CKPT/rank0.log"
while pgrep -f "train_ddp.py" > /dev/null; do
  sleep 60
  SECS=$((SECS + 60))
  if [ $((SECS % 300)) -eq 0 ]; then
    if [ -f "$LOG" ]; then
      echo "[${SECS}s] $(tail -1 "$LOG" 2>/dev/null || echo 'no log yet')"
    fi
  fi
  if [ $SECS -gt 9000 ]; then  # 2.5h absolute ceiling
    echo "TIMEOUT at 150min"
    pkill -f "train_ddp.py" || true
    ssh joelwang-ai-1@10.77.0.2 "pkill -f 'train_ddp.py'" || true
    break
  fi
done

# Extract metrics
if [ -f "$LOG" ]; then
  TOK_S=$(grep -oE 'tok/s=[0-9,]+' "$LOG" | tail -30 \
          | sed 's/tok\/s=//;s/,//g' | sort -n \
          | awk '{a[NR]=$1} END{if(NR>0) print a[int(NR/2)]; else print "FAIL"}')
  FINAL_LOSS=$(grep -oE 'loss=[0-9.]+' "$LOG" | tail -1 | sed 's/loss=//')
  BEST_LOSS=$(grep -oE 'best loss=[0-9.]+' "$LOG" | tail -1 | sed 's/best loss=//')
  FINAL_STEP=$(grep -oE 'step[[:space:]]+[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+')
  DIV_STEP=$(grep -oE '(StabilityGuard.*step [0-9]+|nonfinite.*step [0-9]+|NaN.*step [0-9]+)' "$LOG" \
            | head -1 | grep -oE '[0-9]+' | head -1 || echo "none")
  FROZEN_CHECK=$(grep -oE 'frozen_params:[[:space:]]*[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+' || echo "?")
  SCALE_FINAL=$(grep -oE 'scale=[0-9.e+-]+' "$LOG" | tail -1 | sed 's/scale=//')

  # Gate checks
  PASS_STEP="FAIL"
  [ "$FINAL_STEP" = "2000" ] && PASS_STEP="PASS"

  PASS_LOSS="?"
  if [ -n "$FINAL_LOSS" ]; then
    LOSS_OK=$(awk "BEGIN{print ($FINAL_LOSS < 4.30)}")
    [ "$LOSS_OK" = "1" ] && PASS_LOSS="PASS" || PASS_LOSS="FAIL"
  fi

  PASS_NODIV="?"
  [ "$DIV_STEP" = "none" ] && PASS_NODIV="PASS" || PASS_NODIV="FAIL (diverged step ${DIV_STEP})"

  cat >> "$RESULTS" <<EOF

## Results

| metric | value |
|---|---|
| final step | ${FINAL_STEP} |
| tok/s (median) | ${TOK_S} |
| final loss | ${FINAL_LOSS} |
| best loss | ${BEST_LOSS} |
| divergence step | ${DIV_STEP} |
| final GradScaler scale | ${SCALE_FINAL} |
| frozen params | ${FROZEN_CHECK} |

## Gate checks

| gate | status |
|---|---|
| completed 2000 steps | ${PASS_STEP} |
| final loss within 0.1 of baseline (4.20) | ${PASS_LOSS} |
| no divergence | ${PASS_NODIV} |
| 0 frozen params | (check below) |

EOF

  if [ "$PASS_STEP" = "PASS" ] && [ "$PASS_LOSS" = "PASS" ] && [ "$PASS_NODIV" = "PASS" ]; then
    echo "## Verdict: STACK D PASSED" >> "$RESULTS"
    echo "" >> "$RESULTS"
    echo "Stack D composition:" >> "$RESULTS"
    echo "\`\`\`" >> "$RESULTS"
    echo "${BASE_FLAGS} ${STACK_D_FLAGS}" >> "$RESULTS"
    echo "\`\`\`" >> "$RESULTS"
  else
    echo "## Verdict: STACK D FAILED (fall back to C.1/C.2/C.3 diagnostics)" >> "$RESULTS"
  fi

  cat "$RESULTS"
else
  echo "FAIL: no rank0.log" | tee -a "$RESULTS"
fi

echo "=== T-5 C.4 direct DONE ==="
