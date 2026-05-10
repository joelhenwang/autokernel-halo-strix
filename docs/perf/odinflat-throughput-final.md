# OdinFlat throughput investigation — FINAL CONCLUSIONS (2026-05-10)

**TL;DR: autokernel provides zero net benefit for OdinFlat. Ship vanilla
C3 (no `--optimize-kernels`). The +31% throughput from Phase III/IV was
an artifact of broken autograd that silently froze 60M `w_gate_up`
params — revealed at 2000 steps by a +0.65 loss regression.**

## The complete result table

Each probe: OdinFlat + Sprint 1.5 C3 + dolma. 200 steps unless noted.

| # | rmsnorm | silu | Steady tok/s | Loss @ 200 | Loss @ 2000 | Verdict |
|---|---|---|---:|---:|---:|---|
| P0 baseline | manual | native | 31,331 | 4.7013 | **3.1466** | reference |
| P3-broken | HIP (no autograd) | HIP (no autograd) | 58,238 | 7.03 ⚠ | — | broken (gradients blocked everywhere) |
| P3-FIXED | HIP autograd | HIP (no autograd) | 40,962 | 4.8184 | ? | rmsnorm trained, w_gate_up frozen |
| C2 no-HIP-bwd | HIP autograd + PyT bwd | HIP (no autograd) | 40,932 | 4.7776 | ? | same as P3-FIXED |
| V0 native no-HIP | F.rms_norm | native | 31,736 | 4.7269 | ? | baseline-equivalent |
| **V1 native + silu HIP** | **F.rms_norm** | **HIP (no autograd)** | **41,198** | **4.6691** | **3.8010** | +31% tok/s BUT +0.65 loss @ 2000 |
| **V2 native + silu autograd** | **F.rms_norm** | **HIP (autograd)** | **30,976** | **4.7142** | ? | correct training, NO speedup |

## The investigative arc

1. **Phase 0.4 observation:** --optimize-kernels on OdinFlat gave +80%
   throughput and +1.16 loss (badly broken).
2. **Phase I audit:** 2 patterns active: `rmsnorm` + `fused_silu_gate_mul`.
3. **Phase II bisect:** `rmsnorm` caused the 65× collapse (loss +2.33).
4. **Phase III diagnosis:** `_RMSNormReplacement` missing autograd.
   Fixed. Measured +31% throughput at loss +0.12 (step 200).
5. **Phase IV compile study:** HIP backward = PyTorch fallback. +31%
   is the ceiling.
6. **Option 3:** switch `_components.RMSNorm` to `F.rms_norm`. V1 gave
   +31% with clean code.
7. **Phase V 2000-step validation:** V1 loss at step 2000 is 3.80 vs
   baseline 3.15. **+0.65 loss delta exceeds our 0.5 gate.**
8. **Phase V V2 probe:** Same config but with silu autograd re-enabled.
   Loss 4.7142 at step 200 (matches baseline 4.7013 exactly). But
   throughput drops to 30,976 — BELOW baseline. The autograd overhead
   plus HIP backward exceeds the forward savings.

## Why the +31% was fake

`_FusedSwiGLUReplacement` calls `self.kernel_fn(gate, up)` — a raw
pybind HIP op. Its output has `requires_grad=False` and `grad_fn=None`.
This **silently blocks gradient flow to `w_gate_up`** (60M params ×
14 layers = ~60M SwiGLU input-projection weights). With those params
frozen at init:

- Forward passes produce meaningful-looking loss because `w_down` still
  trains (it operates on the output of the frozen forward).
- But training quality degrades over long horizons because 60M of 121M
  total hidden params never update.

At step 200 the effect is small (loss +0.03 vs baseline).  
At step 500 the gap widens to +0.32.  
At step 2000 it's +0.65 — clearly unacceptable.

The "fast" +31% throughput comes from (a) faster silu forward (via HIP
kernel), and (b) not having to compute gradients for `w_gate_up` (since
autograd never traces into that path).

## Why enabling silu autograd doesn't recover the speedup

V2 shows that when the autograd IS properly registered, the HIP
backward kernel (which is numerically correct — confirmed by
`test_silu_gate_mul_backward.py`) plus save-for-backward overhead
slightly exceeds the forward savings. Net throughput: ~30.9K tok/s,
slightly BELOW the 31.3K baseline.

In other words: under correct training semantics, the HIP `silu_gate_mul`
kernel is NOT faster than PyTorch's native `F.silu(x) * up` fused by
Inductor. The forward speed advantage is real in isolation, but the
backward cost and the overhead of torch.library custom op registration
wipe it out.

## Why rmsnorm HIP is similarly no help

Phase V V0 showed `F.rms_norm` (native, Inductor-fused) runs at 31.7K
tok/s — same as manual `rsqrt(mean(x^2))`. Adding HIP rmsnorm (with
autograd fix) gets to 41K only because `silu_gate_mul` sneaks into that
config too with autograd broken. Truly isolated rmsnorm HIP: ~baseline.

## Recommended production config for Sprint 3A

```bash
# No autokernel. Full native/Inductor-fused path.
EXTRA_FLAGS='--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --auto-eval'
# (no --optimize-kernels)
TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
```

**Expected:** 31.3K tok/s steady → Sprint 3A wall ~61h at the **correct**
loss trajectory (3.15 at step 2000, extrapolated to ~4.0 loss at full
epoch end).

## Ceiling implications for Sprint 3A/3B

- **OdinFlat Sprint 3A: NO --optimize-kernels.** 61h wall at C3 baseline.
- **OdinHalo Sprint 3B: --optimize-kernels ON** per Phase 0.3 probe.
  OdinHalo's +38% throughput measurement was less prone to this failure
  mode because iter_norm resets mask the silent-freeze effect every 6
  layers. Confirmed stable at Phase 0.3 probe's step 200 loss parity.
  **This MAY have the same hidden cost at longer horizons.** Worth
  re-validating at 2000 steps before committing Sprint 3B.

## Follow-up work

1. **Triton RMSNorm rewrite (Track 4):** given V0 already matches the
   HIP kernel's isolated speed, the upside is minimal. Skipping.
2. **Triton SwiGLU kernel with proper autograd:** potentially faster
   than HIP silu + custom op backward. 1-2 weeks dev. Speculative.
3. **Verify OdinHalo's --optimize-kernels isn't hiding the same bug:**
   run a 2000-step OdinHalo probe at Sprint 3B config. IF loss diverges
   from C3 baseline, Sprint 3B is also affected. ~90 min compute.

## Artifacts (final set)

```
autokernel/_patterns.py                                 _RMSNormReplacement fixed; _FusedSwiGLUReplacement annotated
models/_components.py                                   RMSNorm uses F.rms_norm
scripts/test_rmsnorm_numerics.py                        HIP fwd equivalent to reference
scripts/test_rmsnorm_autograd.py                        confirmed grad_fn=None was the bug
scripts/test_silu_gate_mul_backward.py                  HIP bwd is numerically correct
scripts/diag_silu_raw_vs_autograd.py                    minimal grad_fn diagnostic
scripts/diag_autokernel_patterns.py                     Phase I pattern audit
scripts/bisect_autokernel.sh                            Phase II bisect orchestrator
scripts/verify_rmsnorm_fix.sh                           Phase III training verifier
scripts/probe_compile_only.sh                           Phase IV compile variants
scripts/probe_option3_native_rmsnorm.sh                 Option 3 V0/V1 probe
scripts/probe_option3_v2_silu_autograd.sh               V2 probe (final evidence)
scripts/run_sprint3a_confirm_v1.sh                      2000-step V1 validation
docs/perf/odinflat-autokernel-inventory.{json,md}       Phase I inventory
docs/perf/odinflat-bisect-findings.md                   Phase II findings
docs/perf/odinflat-rmsnorm-fix.md                       Phase III writeup
docs/perf/odinflat-compile-study.md                     Phase IV writeup
docs/perf/odinflat-option3-native-rmsnorm.md            Option 3 writeup
docs/perf/odinflat-throughput-final.md                  this file
docs/perf/phase{2,3,4,5}-*.tgz                          probe logs
```
