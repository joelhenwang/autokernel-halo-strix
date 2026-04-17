#!/bin/bash
# GRIFFIN-HALO Factor Sweep — Sequential BabyLM ablation runs
# Matching JORMUNGANDR-HALO ablation protocol:
#   BabyLM 1 epoch, ctx=256, Muon, compile+autokernel
#   batch=16, block=256, accum=4, lr=0.0012

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

COMMON="--model models/griffin_halo.py \
    --dataset babylm \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --accum-steps 4 \
    --checkpoint-interval 999999 \
    --time-budget 60"

run_config() {
    local NAME=$1
    local CLASS=$2
    echo "========================================"
    echo "Run: $NAME ($CLASS) at $(date)"
    echo "========================================"
    rm -rf checkpoints/griffin_halo_${NAME}
    python3 -m halo_training \
        --class-name $CLASS \
        --checkpoint-dir checkpoints/griffin_halo_${NAME} \
        $COMMON
    echo "Finished $NAME at $(date)"
    echo ""
    # Print final loss for quick comparison
    tail -1 checkpoints/griffin_halo_${NAME}/train_log.jsonl 2>/dev/null
    echo ""
}

# ==============================================================
# GATE RUNS: These two determine the core mixer for all later runs
# ==============================================================

run_config "r1_griffin" "GriffinHaloR1"
run_config "r2_gqa" "GriffinHaloR2"

echo "========================================"
echo "GATE RUNS COMPLETE. Check results above."
echo "Compare R1 (Griffin) vs R2 (GQA) loss and tok/s."
echo "Then run the remaining configs for the winner."
echo "========================================"

# ==============================================================
# REMAINING RUNS — Update class names based on R1 vs R2 winner
# Default assumes Griffin wins (R3_Griffin_DMC etc.)
# If GQA wins, change to R3_GQA_DMC variants
# ==============================================================

run_config "r3_dmc" "GriffinHaloR3_Griffin_DMC"
run_config "r4_attnres" "GriffinHaloR4_AttnRes"
run_config "r5_dmc_attnres" "GriffinHaloR5_DMC_AttnRes"
run_config "r6_coda_attnres" "GriffinHaloR6_CodaAttnRes"
run_config "r7_adaptive" "GriffinHaloR7_Adaptive"
run_config "r8_d512" "GriffinHaloR8_d512"

# Run 9: Confirmation — update class after analyzing R3-R8
run_config "r9_best" "GriffinHaloR9"

# Run 10: ARGUS-PRIME stacked baseline
echo "========================================"
echo "Run 10: ARGUS-PRIME B0 (stacked baseline) at $(date)"
echo "========================================"
rm -rf checkpoints/griffin_halo_r10_argus
python3 -m halo_training \
    --model models/argus_prime.py \
    --class-name ArgusPrime \
    --dataset babylm \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --accum-steps 4 \
    --checkpoint-dir checkpoints/griffin_halo_r10_argus \
    --checkpoint-interval 999999 \
    --time-budget 60
echo "Finished ARGUS-PRIME at $(date)"

echo "========================================"
echo "ALL SWEEP RUNS COMPLETE at $(date)"
echo "========================================"
