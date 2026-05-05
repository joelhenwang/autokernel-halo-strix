# WI-B1: Shape sweep for OdinHalo — 2026-05-05

**Baseline:** batch=16, block=256 @ 13,997 tok/s

**Config:** compile_zones default, fp16 autocast, fused AdamW, warmup=20, measured=100


## Token budget: 2048 tokens/step

| Batch | Block | Status | Warmup (s) | tok/s | Peak GB | vs 16×256 |
|------:|------:|--------|-----------:|------:|--------:|----------:|
| 8 | 256 | OK | 3.2 | 12,808 | 3.20 | -8.49% |
| 16 | 128 | OK | 3.2 | 12,683 | 3.20 | -9.39% |

## Token budget: 4096 tokens/step

| Batch | Block | Status | Warmup (s) | tok/s | Peak GB | vs 16×256 |
|------:|------:|--------|-----------:|------:|--------:|----------:|
| 32 | 128 | OK | 32.1 | 14,303 | 5.28 | +2.18% |
| 16 | 256 | OK | 5.9 | 13,997 | 5.28 | +0.00% |
| 8 | 512 | OK | 6.0 | 13,923 | 5.28 | -0.53% |

## Token budget: 4608 tokens/step

| Batch | Block | Status | Warmup (s) | tok/s | Peak GB | vs 16×256 |
|------:|------:|--------|-----------:|------:|--------:|----------:|
| 24 | 192 | OK | 45.1 | 14,403 | 5.82 | +2.90% |
| 12 | 384 | OK | 6.6 | 14,218 | 5.83 | +1.58% |

## Token budget: 8192 tokens/step

| Batch | Block | Status | Warmup (s) | tok/s | Peak GB | vs 16×256 |
|------:|------:|--------|-----------:|------:|--------:|----------:|
| 64 | 128 | OK | 28.6 | 13,536 | 9.63 | -3.29% |
| 32 | 256 | OK | 12.3 | 13,487 | 9.63 | -3.64% |
| 16 | 512 | OK | 12.5 | 13,370 | 9.63 | -4.48% |

## Top 5 throughput shapes

| Rank | Batch | Block | Tokens/step | tok/s | Peak GB | vs baseline |
|-----:|------:|------:|------------:|------:|--------:|------------:|
| 1 | 24 | 192 | 4608 | 14,403 | 5.82 | +2.90% |
| 2 | 32 | 128 | 4096 | 14,303 | 5.28 | +2.18% |
| 3 | 12 | 384 | 4608 | 14,218 | 5.83 | +1.58% |
| 4 | 16 | 256 | 4096 | 13,997 | 5.28 | +0.00% |
| 5 | 8 | 512 | 4096 | 13,923 | 5.28 | -0.53% |

## Verdict

Best shape: **batch=24, block=192** @ **14,403 tok/s** (+2.90% vs baseline), peak 5.82 GB.

Shipping-gate eligible (+2.90% ≥ -2%) but does not clear stretch (+5%).