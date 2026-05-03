# BALDR-HALO: Fast Flat Hybrid LLM, 118M

**Date:** 2026-04-29
**Status:** Design approved
**Target:** Higher tok/s than TYR-HALO at 2x params. Same data (stem-crawl-solo).

---

## Design Principle

Keep all novelty that costs <10% overhead. Drop everything >50% overhead. Flat 12-layer hybrid — no loop, no mHC, no SSM. MoDA depth-attention across GQA layers provides cross-layer information retrieval without Parcae loop cost.

## Architecture

```
FactorizedEmbed(50257, rank=256, d=768)
  |
L0:  ShortConvBlock(768, 512, 3072, momentum)
L1:  ShortConvBlock(768, 512, 3072, momentum)
L2:  ShortConvBlock(768, 512, 3072, momentum)
L3:  MoDA-GQA(768, 12h/4kv, 3072, XSA+QK-Norm)  ← first GQA
L4:  ShortConvBlock(768, 512, 3072, momentum)
L5:  ShortConvBlock(768, 512, 3072, momentum)
L6:  ShortConvBlock(768, 512, 3072, momentum)
L7:  MoDA-GQA(768, 12h/4kv, 3072, XSA+QK-Norm)  ← depth KVs from L3
L8:  ShortConvBlock(768, 512, 3072, momentum)
L9:  ShortConvBlock(768, 512, 3072, momentum)
L10: ShortConvBlock(768, 512, 3072, momentum)
L11: MoDA-GQA(768, 12h/4kv, 3072, XSA+QK-Norm)  ← depth KVs from L3+L7
  |
RMSNorm(768) → FactorizedLMHead(768, rank=256)
  |
MTP AuxHead(768, rank=256) [training only, depth=1]
```

## Dimensions

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_heads | 12 |
| n_kv_heads | 4 |
| head_dim | 64 |
| ffn_inner | 3072 (4x) |
| d_conv | 512 |
| embed_rank | 256 |
| n_layers | 12 (9 conv + 3 GQA) |
| GQA positions | 3, 7, 11 |
| MTP depth | 1 |
| Total params | ~118M |

## Novel Mechanisms (all <10% overhead)

1. **MoDA depth-attention:** GQA layers attend to prior GQA KVs (prepended as prefix, flash-compatible)
2. **XSA:** Removes self-value projection, zero extra params
3. **QK-Norm:** L2-normalize + learnable scale
4. **MTP depth=1:** Auxiliary t+2 prediction, weight 0.3
5. **Momentum residuals:** velocity = β * velocity + output; x = x + velocity
6. **Factorized embeddings:** rank=256, saves ~20M params

## Dropped (all >50% overhead)

- Parcae loop, mHC Sinkhorn, SSM/TTT, Prelude/Coda

## Speed Advantage

| Factor | TYR-HALO | BALDR-HALO |
|--------|----------|------------|
| Forward passes | 2 (loop) | 1 |
| Compile boost | ~1.5x (per-zone) | ~3x (full model) |
| Autokernel | Partial | Full (all patterns) |
| mHC overhead | ~13% | 0% |
| Estimated tok/s | 18K | 30-40K |

## Training

```bash
python -m halo_training --model models/baldr_halo.py --class-name BaldrHalo \
    --dataset datasets/stem-crawl-solo.bin \
    --compile --optimize-kernels --muon --mtp \
    --batch-size 8 --block-size 1024 --accum-steps 8 \
    --lr 0.02 --epochs 2 --checkpoint-dir checkpoints/baldr_halo
```

## Race

Machine A: TYR-HALO 58M (loop + mHC + MoDA), 18K tok/s
Machine B: BALDR-HALO 118M (flat + MoDA + MTP), 30-40K tok/s (est.)

Compare val loss at matched wall-clock time on stem-crawl-solo.
