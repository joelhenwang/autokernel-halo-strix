# WI-B2: Compile scope sweep — 2026-05-05

**Config:** OdinHalo batch=16 block=256 warmup=20 measured=100 repeat=3

## Summary

| Strategy | Median tok/s | Stdev | Peak GB | Warmup (s) | Final loss | vs A |
|----------|-------------:|------:|--------:|-----------:|-----------:|-----:|
| A: compile_zones (per-layer) | 14,133 | 85 | 5.28 | 5.9 | 5.8148 | +0.00% |
| B: compile _run_shared_block (per-iter-body) | 13,973 | 28 | 5.74 | 6.0 | 5.8216 | -1.13% |
| C: compile _forward_unrolled (whole-model) | 14,317 | 71 | 5.62 | 5.8 | 5.7916 | +1.30% |

## Runs

### A: compile_zones (per-layer)

- run 1: 14,133 tok/s, 5.28 GB, warmup 8.9s, loss 5.8148
- run 2: 14,154 tok/s, 5.28 GB, warmup 5.9s, loss 5.9241
- run 3: 13,998 tok/s, 5.28 GB, warmup 5.9s, loss 5.8062
### B: compile _run_shared_block (per-iter-body)

- run 1: 13,973 tok/s, 5.30 GB, warmup 6.3s, loss 5.8216
- run 2: 13,954 tok/s, 5.74 GB, warmup 6.0s, loss 5.8240
- run 3: 14,010 tok/s, 6.17 GB, warmup 5.9s, loss 5.7764
### C: compile _forward_unrolled (whole-model)

- run 1: 14,353 tok/s, 5.19 GB, warmup 7.8s, loss 5.7911
- run 2: 14,216 tok/s, 5.62 GB, warmup 5.8s, loss 5.7916
- run 3: 14,317 tok/s, 6.06 GB, warmup 5.7s, loss 5.7997