# Phase II: autokernel bisect — `rmsnorm` is the culprit (2026-05-09)

## Method

4 DDP probes on OdinFlat + Sprint 1.5 C3 recipe + dolma-10B, 200 steps
each, ~13 min per probe. 2×2 matrix of {rmsnorm on/off} × {fused_silu_gate_mul on/off}.

## Results at step 200

| Probe | Steady tok/s | Aggregate tok/s | Loss | maxabs layers.13 | fp16 headroom |
|---|---:|---:|---:|---:|---:|
| **P0** baseline (no HIP) | 31,331 | 30,686 | **4.7013** | 27.09 | 2,418× |
| **P1** `rmsnorm` only | 60,639 | 58,707 | **7.0220** | **0.752** | 87,055× |
| **P2** `fused_silu_gate_mul` only | 39,962 | 33,758 | **4.6747** | 2.697 | 24,285× |
| **P3** both (Phase 0.4 repro) | 58,238 | 56,869 | 7.0305 | 0.868 | 75,494× |

## Interpretation

- **`rmsnorm` HIP kernel breaks OdinFlat training.** P1 and P3 show loss
  regression of +2.33 vs baseline (far past our 0.5 gate). Layer-13
  activation magnitude drops 36× (27.09 → 0.75) at just 200 steps —
  matches the 65× collapse observed in Phase 0.4 at 500 steps.

- **`fused_silu_gate_mul` is clean.** P2 matches P0 loss within 0.03
  (within-run noise), with +28% steady-state throughput and −10× layer-13
  activation magnitude. The magnitude shift is real but doesn't hurt
  training.

- **rmsnorm dominates when combined with silu.** P3 ≈ P1 in both loss
  (7.03 vs 7.02) and maxabs (0.87 vs 0.75). The silu kernel doesn't
  amplify rmsnorm's damage; it just runs alongside.

## Activation trajectory (key evidence)

Layer-13 maxabs across bisect probes:

```
step    P0         P1          P2         P3
  50    0.691      0.197       0.283      0.214
 100    3.576      0.445       0.857      0.437
 150    12.859     0.647       1.604      0.602
 200    27.094     0.752       2.697      0.868
```

P0 shows normal growth (27× over 200 steps). P1 and P3 show stalled
growth (only 4× over 200 steps) — activations are being systematically
under-scaled by the rmsnorm kernel. P2 shows intermediate growth (10×)
with healthy loss.

## Throughput implications

If Sprint 3A ships with:
- **Current config** (no HIP): 31.2K tok/s → ~61h wall
- **Silu-only** (ship immediately, no fix needed): 40K tok/s → ~48h wall (**saves ~13h**)
- **Silu + fixed rmsnorm** (Phase III target): 60K tok/s → ~32h wall (**saves ~29h**)

The silu-only path is the "free lunch" checkpoint: +28% throughput at no
quality cost, ready to ship today if Phase III takes longer than
expected.

## Next step (Phase III)

Root-cause the `rmsnorm` HIP kernel. Static reading in Phase I found the
math equivalent to the reference on paper; the drift must come from:

1. **fp16 accumulation order** — the kernel upcasts to fp32 for sum but
   stores back as fp16; subtle rounding could compound over 29 sequential
   invocations.
2. **Weight application precision** — `__half2float(w)` then cast back
   could systematically round toward zero.
3. **Block-reduce ordering** — the wavefront reduction order differs
   from PyTorch's reduction tree; ULP-level differences compound.

Phase III harness: random inputs, iterate the kernel 14 times in a
chain (simulating OdinFlat's depth), compare output magnitude to
reference. If rate of drift matches the 36× observation, we've found
the cause.

## Artifacts

```
scripts/bisect_autokernel.sh
scripts/_bisect_summary.sh
docs/perf/phase2-bisect-logs.tgz
checkpoints/sprint15-bisect-{P0,P1,P2,P3}-* (200-step each)
```
