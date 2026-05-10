#!/bin/bash
# T-5 C.1: warm-start matrix probe.
#
# Goal: test v3 hypothesis H14 (optimizer state mismatch) + warmup-local
# instability (v3 H5) by training native first, saving checkpoint, and
# resuming with the full optimized kernel stack under 4 variations.
#
# The optimized kernel stack (applied on RESUME, not from step 0):
#   --use-fused-zloss --ak-loss-zloss
#   --ak-fix-rope-gate-op --ak-causal-conv-shim
#   --ak-spectra-branchless --ak-sync-cleanup
#   --ak-normuon-telemetry
#
# (We cannot use --ak-autocast-tier all because it's reserved in the taxonomy;
# the register_autocast rules are always-on after A.3 ships.)
#
# Configs:
#   C.1.a  native 500 steps -> optimized resume with preserved optimizer state (+1000 steps)
#   C.1.b  native 500 steps -> optimized resume with FRESH optimizer state (+1000 steps)
#   C.1.c  native 1000 steps -> optimized resume with preserved state (+1000 steps)
#   C.1.d  native 1000 steps -> loss-only kernels (fused_zloss only) for 500 ->
#          then add hidden kernels for final 500 steps  [graduated]
#
# Per user directive: RUN ALL 4 unconditionally regardless of intermediate
# failures. Record per config whether it diverges, at what step, and
# whether trust-cap/staging-equivalent mechanisms fire.

set -e
cd ~/Desktop/ai_lab/autokernel-halo-strix

mkdir -p docs/perf/t5-c1-warmstart
RESULTS=docs/perf/t5-c1-warmstart/results.md
echo "# T-5 C.1 warm-start matrix" > "$RESULTS"
echo "" >> "$RESULTS"
echo "| Config | Phase I steps | Phase II steps | Final tok/s | Loss@end | Divergence step | Trigger | Status |" >> "$RESULTS"
echo "|---|---:|---:|---:|---:|---:|---|---|" >> "$RESULTS"

BASE_FLAGS="--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0"
OPT_STACK="--use-fused-zloss --ak-loss-zloss --ak-fix-rope-gate-op --ak-causal-conv-shim --ak-spectra-branchless --ak-sync-cleanup --ak-normuon-telemetry"
LOSS_ONLY_STACK="--use-fused-zloss --ak-loss-zloss --ak-sync-cleanup"

wait_for_ddp() {
  local SECS=0
  local NAME="$1"
  while pgrep -f "train_ddp.py" > /dev/null; do
    sleep 10
    SECS=$((SECS + 10))
    if [ $SECS -gt 3600 ]; then
      echo "TIMEOUT at 60min for ${NAME}"
      pkill -f "train_ddp.py" || true
      ssh joelwang-ai-1@10.77.0.2 "pkill -f 'train_ddp.py'" || true
      break
    fi
  done
}

extract_metrics() {
  local LOG="$1"
  if [ -f "$LOG" ]; then
    local TOK_S=$(grep -oE 'tok/s=[0-9,]+' "$LOG" | tail -10 \
                  | sed 's/tok\/s=//;s/,//g' | sort -n \
                  | awk '{a[NR]=$1} END{if(NR>0) print a[int(NR/2)]; else print "FAIL"}')
    local LOSS=$(grep -oE 'loss=[0-9.]+' "$LOG" | tail -1 | sed 's/loss=//')
    # Divergence = first occurrence of NaN/nonfinite/StabilityGuard rollback
    local DIV_STEP=$(grep -oE '(StabilityGuard.*step [0-9]+|nonfinite.*step [0-9]+|NaN.*step [0-9]+)' "$LOG" \
                    | head -1 | grep -oE '[0-9]+' | head -1 || echo "none")
    local TRIGGER=$(grep -oE '(nan_loss|loss_spike|param_nan|scale_collapse|grad_skips)' "$LOG" | head -1 || echo "-")
    echo "${TOK_S}|${LOSS}|${DIV_STEP}|${TRIGGER}"
  else
    echo "FAIL|FAIL|-|-"
  fi
}

setup_remote_ckpt_dir() {
  local CKPT="$1"
  rm -rf "$CKPT"
  mkdir -p "$CKPT"
  ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$CKPT"
}

run_phase() {
  local CKPT="$1"
  local MAX_STEPS="$2"
  local EXTRA="$3"
  local RESUME="$4"

  local RESUME_FLAG=""
  if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume-from $RESUME"
  fi

  MODEL=models/odin_flat.py \
  CLASS=OdinFlat \
  DATASET=datasets/dolma-10b-odin32k.bin \
  EPOCHS=1 \
  LR=8e-4 BLOCK=512 BATCH=16 ACCUM=8 \
  WARMUP_STEPS=300 CHECKPOINT_INTERVAL="$MAX_STEPS" MAX_GRAD_NORM=1.0 \
  NUM_WORKERS=12 \
  CKPT_DIR="$CKPT" \
  EXTRA_FLAGS="--max-steps $MAX_STEPS ${BASE_FLAGS} ${EXTRA} ${RESUME_FLAG}" \
  bash scripts/launch_ddp.sh >/dev/null 2>&1
  wait_for_ddp "$CKPT"
}

# ==========================================================================
# C.1.a: native 500 -> optimized + preserved state + 1000 more
# ==========================================================================
echo "=== C.1.a: native 500 -> opt+preserved 1000 ==="
CKPT_A_I=checkpoints/t5-c1a-phase1
CKPT_A_II=checkpoints/t5-c1a-phase2
setup_remote_ckpt_dir "$CKPT_A_I"
setup_remote_ckpt_dir "$CKPT_A_II"

run_phase "$CKPT_A_I" 500 "" ""
if [ -f "$CKPT_A_I/step_500.pt" ]; then
  run_phase "$CKPT_A_II" 1500 "$OPT_STACK --resume-preserve-optimizer" "$CKPT_A_I/step_500.pt"
fi
IFS='|' read -r TOK LOSS DIV TRG <<< "$(extract_metrics "$CKPT_A_II/rank0.log")"
STATUS="PASS"; [ "$DIV" != "none" ] && [ -n "$DIV" ] && STATUS="DIVERGED"
echo "| C.1.a preserved | 500 | 1000 | ${TOK} | ${LOSS} | ${DIV} | ${TRG} | ${STATUS} |" >> "$RESULTS"

# ==========================================================================
# C.1.b: native 500 -> optimized + FRESH optimizer + 1000 more
# ==========================================================================
echo "=== C.1.b: native 500 -> opt+fresh 1000 ==="
CKPT_B_I=checkpoints/t5-c1b-phase1
CKPT_B_II=checkpoints/t5-c1b-phase2
setup_remote_ckpt_dir "$CKPT_B_I"
setup_remote_ckpt_dir "$CKPT_B_II"

run_phase "$CKPT_B_I" 500 "" ""
if [ -f "$CKPT_B_I/step_500.pt" ]; then
  # Default --resume-from is weights-only -> fresh optimizer (exactly what C.1.b needs)
  run_phase "$CKPT_B_II" 1500 "$OPT_STACK" "$CKPT_B_I/step_500.pt"
fi
IFS='|' read -r TOK LOSS DIV TRG <<< "$(extract_metrics "$CKPT_B_II/rank0.log")"
STATUS="PASS"; [ "$DIV" != "none" ] && [ -n "$DIV" ] && STATUS="DIVERGED"
echo "| C.1.b fresh | 500 | 1000 | ${TOK} | ${LOSS} | ${DIV} | ${TRG} | ${STATUS} |" >> "$RESULTS"

# ==========================================================================
# C.1.c: native 1000 -> optimized + preserved + 1000 more
# ==========================================================================
echo "=== C.1.c: native 1000 -> opt+preserved 1000 ==="
CKPT_C_I=checkpoints/t5-c1c-phase1
CKPT_C_II=checkpoints/t5-c1c-phase2
setup_remote_ckpt_dir "$CKPT_C_I"
setup_remote_ckpt_dir "$CKPT_C_II"

run_phase "$CKPT_C_I" 1000 "" ""
if [ -f "$CKPT_C_I/step_1000.pt" ]; then
  run_phase "$CKPT_C_II" 2000 "$OPT_STACK --resume-preserve-optimizer" "$CKPT_C_I/step_1000.pt"
fi
IFS='|' read -r TOK LOSS DIV TRG <<< "$(extract_metrics "$CKPT_C_II/rank0.log")"
STATUS="PASS"; [ "$DIV" != "none" ] && [ -n "$DIV" ] && STATUS="DIVERGED"
echo "| C.1.c delayed | 1000 | 1000 | ${TOK} | ${LOSS} | ${DIV} | ${TRG} | ${STATUS} |" >> "$RESULTS"

# ==========================================================================
# C.1.d: native 1000 -> loss-only 500 -> full hidden 500 (graduated)
# ==========================================================================
echo "=== C.1.d: native 1000 -> loss-only 500 -> full 500 (graduated) ==="
CKPT_D_I=checkpoints/t5-c1d-phase1
CKPT_D_II=checkpoints/t5-c1d-phase2
CKPT_D_III=checkpoints/t5-c1d-phase3
setup_remote_ckpt_dir "$CKPT_D_I"
setup_remote_ckpt_dir "$CKPT_D_II"
setup_remote_ckpt_dir "$CKPT_D_III"

run_phase "$CKPT_D_I" 1000 "" ""
if [ -f "$CKPT_D_I/step_1000.pt" ]; then
  run_phase "$CKPT_D_II" 1500 "$LOSS_ONLY_STACK --resume-preserve-optimizer" "$CKPT_D_I/step_1000.pt"
fi
if [ -f "$CKPT_D_II/step_1500.pt" ]; then
  run_phase "$CKPT_D_III" 2000 "$OPT_STACK --resume-preserve-optimizer" "$CKPT_D_II/step_1500.pt"
fi
IFS='|' read -r TOK LOSS DIV TRG <<< "$(extract_metrics "$CKPT_D_III/rank0.log")"
STATUS="PASS"; [ "$DIV" != "none" ] && [ -n "$DIV" ] && STATUS="DIVERGED"
echo "| C.1.d graduated | 1000 | 500+500 | ${TOK} | ${LOSS} | ${DIV} | ${TRG} | ${STATUS} |" >> "$RESULTS"

echo "=== T-5 C.1 DONE ==="
cat "$RESULTS"
