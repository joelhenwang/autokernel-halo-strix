#!/bin/bash
# T-5 C.2: post-NorMuon trust cap probe (tau=0.02 on w_gate_up).
#
# Goal: test whether capping the post-NorMuon update/weight ratio at 0.02
# on w_gate_up parameters prevents Phase C/G-style divergence in the
# optimized hidden-kernel stack.
#
# Mechanism: after NorMuon produces its (whitened + shape-scaled) update,
# compute ||update|| / ||weight|| elementwise; clamp to <=0.02; rescale.
# Implemented in halo_training/normuon.py::apply_trust_cap_.
#
# 500-step DDP run from scratch with full optimized stack + trust cap.
#
# Decision (v3 pre-committed):
#   passes + triggers visible in telem -> update-scale mechanism, fold into Stack D
#   passes but no triggers                -> mechanism isn't update scale, check H11 instead
#   fails same Phase C pattern            -> trust cap alone is insufficient

set -e
cd ~/Desktop/ai_lab/autokernel-halo-strix

mkdir -p docs/perf/t5-c2-trust-cap
RESULTS=docs/perf/t5-c2-trust-cap/results.md
echo "# T-5 C.2: post-NorMuon trust cap (tau=0.02 on w_gate_up)" > "$RESULTS"
echo "" >> "$RESULTS"

CKPT=checkpoints/t5-c2-trust-cap
rm -rf "$CKPT"
mkdir -p "$CKPT"
ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$CKPT"

BASE_FLAGS="--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0"
OPT_STACK="--use-fused-zloss --ak-loss-zloss --ak-fix-rope-gate-op --ak-causal-conv-shim --ak-spectra-branchless --ak-sync-cleanup --ak-normuon-telemetry"
TRUST_CAP_FLAGS="--ak-trust-cap 0.02 --ak-trust-cap-scope w_gate_up"

echo "=== T-5 C.2: full opt stack + trust cap (500 steps) ==="

MODEL=models/odin_flat.py \
CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 BLOCK=512 BATCH=16 ACCUM=8 \
WARMUP_STEPS=300 CHECKPOINT_INTERVAL=500 MAX_GRAD_NORM=1.0 \
NUM_WORKERS=12 \
CKPT_DIR="$CKPT" \
EXTRA_FLAGS="--max-steps 500 ${BASE_FLAGS} ${OPT_STACK} ${TRUST_CAP_FLAGS}" \
bash scripts/launch_ddp.sh >/dev/null 2>&1 &

# Wait for completion
SECS=0
while pgrep -f "train_ddp.py" > /dev/null; do
  sleep 15
  SECS=$((SECS + 15))
  if [ $SECS -gt 3600 ]; then
    echo "TIMEOUT at 60min"
    pkill -f "train_ddp.py" || true
    ssh joelwang-ai-1@10.77.0.2 "pkill -f 'train_ddp.py'" || true
    break
  fi
done

# Extract metrics
if [ -f "$CKPT/rank0.log" ]; then
  TOK_S=$(grep -oE 'tok/s=[0-9,]+' "$CKPT/rank0.log" | tail -10 \
                | sed 's/tok\/s=//;s/,//g' | sort -n \
                | awk '{a[NR]=$1} END{if(NR>0) print a[int(NR/2)]; else print "FAIL"}')
  LOSS=$(grep -oE 'loss=[0-9.]+' "$CKPT/rank0.log" | tail -1 | sed 's/loss=//')
  DIV_STEP=$(grep -oE '(StabilityGuard.*step [0-9]+|nonfinite.*step [0-9]+)' "$CKPT/rank0.log" \
            | head -1 | grep -oE '[0-9]+' | head -1 || echo "none")

  # Trust-cap triggers visible in normuon telemetry?
  TELEM_FILE="$CKPT/normuon_telem_rank0.jsonl"
  CAP_TRIGGERS="-"
  if [ -f "$TELEM_FILE" ]; then
    CAP_TRIGGERS=$(grep -oE '"trust_cap_triggered":\s*true' "$TELEM_FILE" | wc -l)
  fi

  STATUS="PASS"
  [ "$DIV_STEP" != "none" ] && [ -n "$DIV_STEP" ] && STATUS="DIVERGED"

  cat >> "$RESULTS" <<EOF
## Results

| metric | value |
|---|---|
| tok/s (median last 10) | ${TOK_S} |
| loss (final) | ${LOSS} |
| divergence step | ${DIV_STEP} |
| trust-cap triggers | ${CAP_TRIGGERS} |
| status | ${STATUS} |

## Interpretation

EOF
  if [ "$STATUS" = "PASS" ] && [ "$CAP_TRIGGERS" != "0" ] && [ "$CAP_TRIGGERS" != "-" ]; then
    echo "update-scale mechanism CONFIRMED (trust cap fired, training stable). Fold into Stack D." >> "$RESULTS"
  elif [ "$STATUS" = "PASS" ] && ([ "$CAP_TRIGGERS" = "0" ] || [ "$CAP_TRIGGERS" = "-" ]); then
    echo "Training stable but trust cap never fired. Update-ratio never exceeded 0.02 in 500 steps. Mechanism is NOT primarily update-scale — look at H11 (autocast boundary) or H14 (optimizer state) instead." >> "$RESULTS"
  else
    echo "Training diverged same as Phase C. Trust cap alone is INSUFFICIENT — combine with C.3 staging and/or full register_autocast rules (A.3)." >> "$RESULTS"
  fi
  cat "$RESULTS"
else
  echo "| NO_LOG | - | - | - | - | FAIL |" >> "$RESULTS"
  echo "FAIL: no rank0.log"
fi

echo "=== T-5 C.2 DONE ==="
