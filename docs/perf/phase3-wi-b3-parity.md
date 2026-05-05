# WI-B3 parity: max-autotune loss divergence investigation — 2026-05-05

**Config:** OdinHalo batch=16 block=256, 200 training steps, same seed

## Summary

- **Baseline mode:** compile_zones (default)
- **Candidate:** per-layer max-autotune
- **Max |loss delta|:** 0.2085 at step 68
- **Mean |loss delta|:** 0.0862
- **Final loss baseline:** 4.5037
- **Final loss autotune:** 4.6052
- **Final |delta|:** 0.1015

## Throughput (steady-state, skip first 50 steps)

| Metric | Baseline | max-autotune | Δ |
|--------|---------:|-------------:|--:|
| Steady tok/s | 14,018 | 14,742 | +5.17% |
| Peak GB | 5.28 | 5.23 | - |

## Loss trajectory (every 10 steps)

| Step | Baseline | max-autotune | Δ |
|-----:|---------:|-------------:|--:|
| 1 | 10.4056 | 10.4056 | +0.0000 |
| 11 | 10.2069 | 10.2066 | -0.0003 |
| 21 | 9.0166 | 9.0141 | -0.0025 |
| 31 | 7.6239 | 7.6156 | -0.0083 |
| 41 | 6.8027 | 6.7932 | -0.0095 |
| 51 | 6.1824 | 6.3456 | +0.1632 |
| 61 | 6.3194 | 6.3567 | +0.0372 |
| 71 | 6.6168 | 6.6722 | +0.0554 |
| 81 | 5.9057 | 6.0445 | +0.1389 |
| 91 | 6.4223 | 6.4907 | +0.0684 |
| 101 | 6.0454 | 6.1266 | +0.0812 |
| 111 | 5.1250 | 5.2261 | +0.1011 |
| 121 | 5.0801 | 5.1983 | +0.1183 |
| 131 | 5.6154 | 5.7562 | +0.1408 |
| 141 | 5.0204 | 5.2031 | +0.1827 |
| 151 | 5.3843 | 5.5578 | +0.1735 |
| 161 | 5.0227 | 5.1641 | +0.1414 |
| 171 | 5.5677 | 5.6952 | +0.1275 |
| 181 | 5.2779 | 5.4075 | +0.1296 |
| 191 | 5.2855 | 5.3896 | +0.1041 |
| 200 | 4.5037 | 4.6052 | +0.1015 |

## Verdict

**MARGINAL**: max |delta| = 0.2085. Likely fp16-accumulation-order variance but worth a longer training run to confirm convergence.