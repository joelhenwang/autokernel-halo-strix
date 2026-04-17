#!/bin/bash
# GRIFFIN-HALO Optimization Ablation — 3 isolated tests
# Baseline: R9 (Griffin+DMC, d=768, 4 iters, ffn=2048, compile)
# Each run changes ONE variable from R9.

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

# ==============================================================
# Test A: 3 iterations (vs R9's 4)
# ==============================================================
echo "========================================"
echo "Opt A: 3 iterations at $(date)"
echo "========================================"
rm -rf checkpoints/griffin_halo_opt_3iter
python3 -m halo_training \
    --model models/griffin_halo.py \
    --class-name GriffinHaloOpt_3iter \
    --dataset babylm \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --accum-steps 4 \
    --checkpoint-dir checkpoints/griffin_halo_opt_3iter \
    --checkpoint-interval 999999 \
    --time-budget 60
echo "Finished Opt A at $(date)"
tail -1 checkpoints/griffin_halo_opt_3iter/train_log.jsonl
echo ""

# ==============================================================
# Test B: FFN 1792 (vs R9's 2048)
# ==============================================================
echo "========================================"
echo "Opt B: FFN 1792 at $(date)"
echo "========================================"
rm -rf checkpoints/griffin_halo_opt_ffn1792
python3 -m halo_training \
    --model models/griffin_halo.py \
    --class-name GriffinHaloOpt_ffn1792 \
    --dataset babylm \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --accum-steps 4 \
    --checkpoint-dir checkpoints/griffin_halo_opt_ffn1792 \
    --checkpoint-interval 999999 \
    --time-budget 60
echo "Finished Opt B at $(date)"
tail -1 checkpoints/griffin_halo_opt_ffn1792/train_log.jsonl
echo ""

# ==============================================================
# Test C: Per-zone compile (vs R9's whole-model compile)
# NOTE: NO --compile flag. Model uses compile_zones() internally.
# We still use --optimize-kernels for autokernel.
# The trainer needs to call model.compile_zones() — since the CLI
# doesn't support this, we use a wrapper script approach:
# We run WITHOUT --compile (eager mode) to measure the baseline,
# then the per-zone compile benefit can be estimated.
# ==============================================================
echo "========================================"
echo "Opt C: No compile (eager + autokernel only) at $(date)"
echo "========================================"
rm -rf checkpoints/griffin_halo_opt_nocompile
python3 -m halo_training \
    --model models/griffin_halo.py \
    --class-name GriffinHaloOpt_zones \
    --dataset babylm \
    --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --accum-steps 4 \
    --checkpoint-dir checkpoints/griffin_halo_opt_nocompile \
    --checkpoint-interval 999999 \
    --time-budget 60
echo "Finished Opt C at $(date)"
tail -1 checkpoints/griffin_halo_opt_nocompile/train_log.jsonl
echo ""

echo "========================================"
echo "ALL OPTIMIZATION TESTS COMPLETE at $(date)"
echo "========================================"
echo "Compare against R9 baseline: loss=12.770 (actual 3.193), tok/s=21274"
