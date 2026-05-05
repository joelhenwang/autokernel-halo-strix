# WI-B3: max-autotune compile mode — 2026-05-05

**Config:** OdinHalo batch=16 block=256 warmup=30 measured=100 repeat=3

## Summary

| Strategy | Median tok/s | Stdev | Peak GB | Warmup (s) | Final loss | vs A |
|----------|-------------:|------:|--------:|-----------:|-----------:|-----:|
| A: compile_zones (default mode, baseline) | 14,002 | 48 | 5.28 | 8.9 | 5.4089 | +0.00% |
| B: per-layer max-autotune | 14,638 | 71 | 5.23 | 9.7 | 5.4134 | +4.54% |
| C: per-layer max-autotune-no-cudagraphs | 14,276 | 24 | 6.65 | 8.6 | 5.3837 | +1.96% |