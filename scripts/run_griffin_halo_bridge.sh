#!/bin/bash
# GRIFFIN-HALO Bridge Tests — Targeting ≥34K tok/s at ≤3.2 loss
# Baseline: R9 (3.193 loss, 21.3K tok/s) and JORMUNGANDR XSA+DMC (5.770, 33.7K)

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

# ==============================================================
# Idea 1: Lean — 1 Prelude GQA + 3 Griffin iters + 2 Coda layers
# Target: ~33-37K tok/s at d=768 quality
# ==============================================================
echo "========================================"
echo "Idea 1: Lean GRIFFIN-HALO at $(date)"
echo "========================================"
rm -rf checkpoints/griffin_halo_lean
python3 -m halo_training \
    --class-name GriffinHaloLean \
    --checkpoint-dir checkpoints/griffin_halo_lean \
    $COMMON
echo "Finished Lean at $(date)"
tail -1 checkpoints/griffin_halo_lean/train_log.jsonl
echo ""

# ==============================================================
# Idea 2: Progressive — d=768 iter 1 (Griffin) → d=512 iters 2-4
# Target: ~31-34K tok/s bridging quality gap
# ==============================================================
echo "========================================"
echo "Idea 2: Progressive Narrowing at $(date)"
echo "========================================"
rm -rf checkpoints/griffin_halo_progressive
python3 -m halo_training \
    --class-name GriffinHaloProgressive \
    --checkpoint-dir checkpoints/griffin_halo_progressive \
    $COMMON
echo "Finished Progressive at $(date)"
tail -1 checkpoints/griffin_halo_progressive/train_log.jsonl
echo ""

echo "========================================"
echo "BRIDGE TESTS COMPLETE at $(date)"
echo "Compare:"
echo "  R9 baseline: loss=12.770 (actual 3.193), tok/s=21274"
echo "  JORMUNGANDR XSA+DMC: loss=5.770, tok/s=33700"
echo "========================================"
