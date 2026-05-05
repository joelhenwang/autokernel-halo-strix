# Phase 3 WI-A0: reduce-overhead probe findings — 2026-05-05

**Config:** OdinHalo batch=16 block=256 warmup=15 measured=50

## Summary table

| Experiment | Status | Warmup (s) | tok/s | Peak GB | Final loss |
|-----------|--------|-----------:|------:|--------:|-----------:|
| E0: baseline compile_zones default | OK | 7.2 | 14,472 | 5.28 | 6.1921 |
| E1: compile_zones reduce-overhead, no clones | OK | 13.7 | 14,193 | 5.23 | 6.2904 |
| E1c: compile_zones reduce-overhead + layer-boundary clone | OK | 5.0 | 14,088 | 5.23 | 6.2652 |
| E3: no compile (eager) | OK | 5.8 | 10,978 | 7.07 | 6.1839 |

## Per-experiment detail

### E0: baseline compile_zones default

- **Status:** `OK`
- **compile_mode:** `compile_zones_default`
- **insert_clones:** `False`
- **chunked_ce:** `False`
- **Warmup:** 7.23 s
- **tok/s (steady):** 14,471.7
- **Peak memory:** 5.279 GB
- **Loss trajectory:** first=9.5160, last=6.1921, min=6.0288, max=9.5160, len=50

### E1: compile_zones reduce-overhead, no clones

- **Status:** `OK`
- **compile_mode:** `compile_zones_reduce_overhead`
- **insert_clones:** `False`
- **chunked_ce:** `False`
- **Warmup:** 13.71 s
- **tok/s (steady):** 14,193.2
- **Peak memory:** 5.232 GB
- **Loss trajectory:** first=9.5155, last=6.2904, min=6.0895, max=9.5155, len=50

### E1c: compile_zones reduce-overhead + layer-boundary clone

- **Status:** `OK`
- **compile_mode:** `compile_zones_reduce_overhead`
- **insert_clones:** `True`
- **chunked_ce:** `False`
- **Warmup:** 5.00 s
- **tok/s (steady):** 14,088.2
- **Peak memory:** 5.233 GB
- **Loss trajectory:** first=9.5147, last=6.2652, min=6.0989, max=9.5147, len=50

### E3: no compile (eager)

- **Status:** `OK`
- **compile_mode:** `none`
- **insert_clones:** `False`
- **chunked_ce:** `False`
- **Warmup:** 5.79 s
- **tok/s (steady):** 10,978.1
- **Peak memory:** 7.069 GB
- **Loss trajectory:** first=9.5295, last=6.1839, min=6.0017, max=9.5295, len=50


## Conclusions

(Filled in after reviewing experiments)
