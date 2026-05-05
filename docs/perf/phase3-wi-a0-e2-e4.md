# Phase 3 WI-A0: reduce-overhead probe findings — 2026-05-05

**Config:** OdinHalo batch=16 block=256 warmup=15 measured=50

## Summary table

| Experiment | Status | Warmup (s) | tok/s | Peak GB | Final loss |
|-----------|--------|-----------:|------:|--------:|-----------:|
| E2: whole-model reduce-overhead | OK | 14.6 | 14,419 | 5.06 | 6.2140 |
| E4: compile_zones reduce-overhead + chunked_ce | OK | 6.9 | 10,085 | 3.64 | 6.4972 |

## Per-experiment detail

### E2: whole-model reduce-overhead

- **Status:** `OK`
- **compile_mode:** `whole_model_reduce_overhead`
- **insert_clones:** `False`
- **chunked_ce:** `False`
- **Warmup:** 14.56 s
- **tok/s (steady):** 14,419.2
- **Peak memory:** 5.058 GB
- **Loss trajectory:** first=9.5037, last=6.2140, min=6.0565, max=9.5037, len=50

### E4: compile_zones reduce-overhead + chunked_ce

- **Status:** `OK`
- **compile_mode:** `compile_zones_reduce_overhead`
- **insert_clones:** `False`
- **chunked_ce:** `True`
- **Warmup:** 6.92 s
- **tok/s (steady):** 10,085.4
- **Peak memory:** 3.641 GB
- **Loss trajectory:** first=9.9244, last=6.4972, min=6.4342, max=9.9244, len=50


## Conclusions

(Filled in after reviewing experiments)
