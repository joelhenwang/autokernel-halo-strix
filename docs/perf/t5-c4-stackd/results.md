# T-5 C.4 Stack D direct 2000-step gate


## Results

| metric | value |
|---|---|
| final step | 2000 |
| tok/s (median) | 34848 |
| final loss | 3.1384 |
| best loss | 3.1384 |
| divergence step |  |
| final GradScaler scale | 1.6e+04 |
| frozen params | ? |

## Gate checks

| gate | status |
|---|---|
| completed 2000 steps | PASS |
| final loss within 0.1 of baseline (4.20) | PASS |
| no divergence | FAIL (diverged step ) |
| 0 frozen params | (check below) |

## Verdict: STACK D FAILED (fall back to C.1/C.2/C.3 diagnostics)
