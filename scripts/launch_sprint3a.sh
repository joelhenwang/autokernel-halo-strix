#!/bin/bash
# T-6 Sprint 3A launch: OdinFlat 1 epoch on dolma-10B.
#
# Stack lookup (2026-05-11+, v3 campaign results):
#   STACK=A   Fused z-loss only (T-1.5 canonical; +6.6% tok/s vs baseline)
#   STACK=B   SAME as A (batch=32 null-effect per T-2.1, DDP bucket null per T-1.4)
#   STACK=C   SAME as A (compiled autograd regresses per T-4)
#   STACK=D   A + ak-fix-rope-gate-op + ak-causal-conv-shim + ak-sync-cleanup
#             + ak-spectra-branchless + ak-normuon-telemetry
#             (+10.7% tok/s + better quality; T-5 C.4 canonical winner)
#   STACK=E   Delayed-enable: run scripts/launch_sprint3a_stackE.sh (two-stage)
#             Stage 1: native 1000 steps, Stage 2: Stack D resume (preserve optim)
#             T-5 C.1.c evidence: -0.033 better loss at matching total step than
#             Stack D from-scratch. Slightly more operator overhead.
#
# C.2 showed trust cap + w_gate_up staging have 0 effect (update-scale ruled out
# as a divergence mechanism). They were removed from Stack D production recipe.
#
# Launch from Machine A:
#   STACK=D bash scripts/launch_sprint3a.sh   # recommended
#   STACK=E bash scripts/launch_sprint3a.sh   # delayed-enable; alt recipe

set -e

# Stack selection via env var; default to Stack D (T-5 canonical winner).
STACK="${STACK:-D}"

# Stack E is implemented as a separate two-stage script — delegate to it.
if [ "$STACK" = "E" ]; then
  echo "Sprint 3A Stack E: delegating to launch_sprint3a_stackE.sh"
  exec bash "$(dirname "$0")/launch_sprint3a_stackE.sh"
fi

EXTRA_FLAGS_BASE="--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --ema --auto-eval \
  --use-fused-zloss --ak-loss-zloss"

BATCH_OVERRIDE="16"
ACCUM_OVERRIDE="8"

case "$STACK" in
  A)
    # T-1.5 canonical: fused zloss only. +6.6% vs Sprint 3A-confirm baseline.
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE"
    echo "Sprint 3A Stack A: fused z-loss only (T-1.5 canonical, +6.6%)"
    ;;
  B)
    # Stack B is not a thing — batch=32 + DDP bucket both null-effect per
    # T-2.1 + T-1.4. Fall through to Stack A recipe (identical throughput).
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE --ak-sync-cleanup --ak-spectra-branchless"
    echo "Sprint 3A Stack B (= A + sync-cleanup; batch=32 + bucket null-effect per B.1/B.2)"
    ;;
  C)
    # Stack C = Stack B (compiled autograd regresses per T-4)
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE --ak-sync-cleanup --ak-spectra-branchless"
    echo "Sprint 3A Stack C (= B; compiled autograd regresses per T-4)"
    ;;
  D)
    # T-5 C.4 canonical winner: 34,697 tok/s (+10.7%), strictly better
    # quality than Stack A (BPB, kurtosis, effective rank all improved).
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE \
      --ak-fix-rope-gate-op --ak-causal-conv-shim \
      --ak-sync-cleanup --ak-spectra-branchless \
      --ak-normuon-telemetry"
    echo "Sprint 3A Stack D: CANONICAL WINNER (+10.7%, T-5 C.4 validated)"
    ;;
  *)
    echo "Unknown STACK=$STACK (must be A, B, C, D, or E)"
    echo "Recommended: STACK=D (canonical winner)"
    echo "Alt:         STACK=E (delayed-enable; slight quality edge per C.1.c/d)"
    exit 1
    ;;
esac

echo "  batch=${BATCH_OVERRIDE} accum=${ACCUM_OVERRIDE} effective_batch=$((BATCH_OVERRIDE * ACCUM_OVERRIDE * 2))"
echo ""

CKPT_DIR="${CKPT_DIR:-checkpoints/sprint3a-stack-${STACK}-dolma10b}"

MODEL=models/odin_flat.py \
CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 \
BLOCK=512 \
BATCH="$BATCH_OVERRIDE" \
ACCUM="$ACCUM_OVERRIDE" \
WARMUP_STEPS=300 \
CHECKPOINT_INTERVAL=500 \
MAX_GRAD_NORM=1.0 \
NUM_WORKERS=12 \
CKPT_DIR="$CKPT_DIR" \
EXTRA_FLAGS="$EXTRA_FLAGS" \
bash scripts/launch_ddp.sh
