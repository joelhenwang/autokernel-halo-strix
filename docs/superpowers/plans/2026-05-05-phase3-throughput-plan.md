# Phase 3 Implementation Plan — Throughput-Primary, Broadened Scope

**Goal:** Find any path to **+5% tok/s** over the Phase 2-confirmed baseline of
**14,682 tok/s** (OdinHalo, batch=16, block=256, `compile + HIP CE`).

**Shipping gate (realistic):** throughput ≥ −2% (≥ 14,400 tok/s) + memory or
batch-unlock benefit.
**Stretch gate (aspirational):** throughput ≥ +5% (≥ 15,420 tok/s).
**Quality gate:** 200-step loss parity within 1% relative of the `compile + HIP CE`
baseline.

**Companion docs:**
- Spec: `docs/superpowers/specs/2026-05-05-phase3-cuda-graphs-parcae-design.md`
- Phase 2 summary: `docs/perf/phase2-summary-2026-05-05.md`
- Inductor catalog: `docs/perf/inductor-fusion-catalog.md`

Phase 3 is structured in **two tracks** evaluated sequentially. Track A pursues
the originally-specified CUDA graph approach. Track B pivots to broader throughput
levers if Track A cannot clear the stretch gate.

---

## Key changes from original spec

| Aspect | Original spec | Revised plan |
|--------|---------------|-------------|
| Expected gain claim | "+10–15% tok/s" | Evidence shows reduce-overhead is −1.8% tok/s alone; +5% is a stretch target, not promise |
| Acceptance gates | +5% tok/s + mem ≤ 3.5 GB + 120s warmup | Ship: ≥−2% tok/s + benefit. Stretch: +5% |
| Candidate order | X → Y → Z | X → Z → Y (Y cost/benefit is worst given Phase 2) |
| Scope | CUDA graphs only | Track A (CUDA graphs) + Track B (broader throughput levers) |
| Time budget | 5-7 days | Work items with gates, no day counts |
| Day 0 | not specified | Required WI-A0 investigation before A1 |
| Clone cost math | 0.8% (spec's 570 μs / 68 ms) | 0.24% (actual 570 μs / 279 ms) |

---

## Track A — CUDA graphs through Parcae

### WI-A0 — Investigation: reproduce STATUS row 113

STATUS.md claims `compile reduce-overhead + HIP CE = 14,425 tok/s / 2.14 GB` at
batch=16 for what may or may not be a looped OdinHalo run. The trainer's own
narrative says reduce-overhead auto-falls-back. Resolve the ambiguity before
committing to WI-A1.

**Approach:**
1. Instrument trainer: run OdinHalo with `TORCH_COMPILE_MODE=reduce-overhead` and
   print what compile mode actually applied.
2. If auto-fallback fires: temporarily bypass it with an env override and measure
   what actually breaks (NaN loss, CUDA error, silent wrong-answer?). Classify the
   failure mode.
3. Also test `reduce-overhead + --chunked-ce` (STATUS rows 114, 116 claim this
   works; narrative says it doesn't).
4. Write `docs/perf/phase3-wi-a0-findings.md` with:
   - Measured tok/s / memory for reduce-overhead + current forward
   - Actual failure mode of reduce-overhead on looped model
   - Whether STATUS row 113 is reproducible, not reproducible, or was never real

**Exit gates:**
- If reduce-overhead "just works" at ≥ 14,400 tok/s without custom glue: skip
  WI-A1/A2/A3, jump to landing.
- If it fails predictably (buffer aliasing, confirmed by NaN loss in first few
  steps): WI-A1 is justified.
- If it fails unpredictably (new failure mode we don't understand): deep-dive
  before touching model code.

### WI-A1 — Option X: clone-at-boundaries

**Approach:**
1. Identify every compile-zone output that feeds a subsequent compile-zone input
   or a non-compiled autograd save. From `_forward_unrolled`:
   - After each `layer(h, ...)` inside `_run_shared_block`
   - After each `_apply_iter_norm` call
   - After each `injection(h, input_embed)` call
   - After final-iteration `h` before `final_norm`
2. Insert `h = h.clone()` at all identified sites (approximate: 6 layers × 3
   iters + 3 iter_norms + 2 injections + 1 final ≈ 24 clone sites).
3. Add `compile_zones_graphed(self)` method on `OdinHaloBase` that calls
   `compile_zones()` then sets `self._graphed = True` — actual graph mode comes
   from env `TORCH_COMPILE_MODE=reduce-overhead`.
4. Remove the auto-fallback in `halo_training/trainer.py` for models with
   `compile_zones_graphed()` attribute.

**Clone-cost math (corrected):** at 14,682 tok/s × 4096 tokens/step =
**step time ≈ 279 ms/step**. Each clone of `(16, 256, 768)` fp16 = 6 MB ×
256 GB/s ≈ 23 μs + 5 μs launch = 28 μs. 24 clones × 28 μs =
**672 μs ≈ 0.24% overhead**. Well within budget.

**Measurement:** `scripts/ablation_phase3.py` (new) — 200-step measured run,
records tok/s, peak_gb, warmup_s, loss every 25 steps.

**Gate:** tok/s ≥ 14,400 AND 200-step loss within 1%.

### WI-A2 — Option Z: manual CUDA graph (only if WI-A1 fails)

**Approach:**
1. New file `models/components/cuda_graph_wrap.py` with `CUDAGraphedParcae` class.
2. Pre-allocate static input/output buffers sized for OdinHalo production config.
3. Capture forward via `torch.cuda.CUDAGraph.capture_begin/end`.
4. Custom `torch.autograd.Function` for the graphed region with explicit
   saved-tensor handling.
5. Same replay-and-measure harness as WI-A1.

**Risk:** autograd integration is delicate. GradScaler interacts with backward
in ways that may not compose cleanly with graph replay. NaN in first 10 steps
with `scaler.unscale_` is a known hazard.

**Gate:** same as WI-A1.

### WI-A3 — Option Y: unrolled compile (only if A1 AND A2 fail)

**Approach:**
1. Rewrite `_forward_unrolled` to call a single `torch.compile`'d `_parcae_body`
   that unrolls the `mean_recurrence` loop.
2. Expected compile-time inflation: 6 layers × 3 iters = 18 FX graphs vs current 6.

**Rationale per Phase 2:** Inductor catalog shows 92 fused kernels. Unrolling
triples the graph count without changing the fused-kernel structure. Included
only because user directed "keep as final fallback".

**Gate:** tok/s ≥ 14,400 AND 200-step loss within 1% AND warmup ≤ 180 s.

### Track A exit

- One of A1/A2/A3 passes gate: land winner, skip Track B, proceed to Landing.
- All three fail shipping gate but A1 or A2 passes memory gate only: land memory
  win, continue to Track B for throughput.
- All three fail all gates: close Track A with `docs/perf/phase3-track-a-blocked.md`,
  proceed to Track B.

---

## Track B — Broader throughput levers

Entered only if Track A doesn't clear the +5% stretch gate. Each WI is independent;
can be done in any order or stopped mid-way once gate is met.

### WI-B1 — Shape sweep (block × batch at constant token budget)

rocBLAS performance is shape-sensitive. Current (batch=16, block=256) = 4096
tokens/step might not be tok/s-optimal even at the same total token budget.

**Matrix:** {block=128, 192, 256, 384, 512} × {batch=8, 16, 24, 32} subject to
total_tokens ≤ 4096 per step AND memory ≤ 5 GB.

**Deliverable:** `docs/perf/phase3-wi-b1-shape-sweep.md`. If any shape shows ≥+5%
tok/s with equivalent tokens/step, winner.

**Gate:** +5% tok/s with quality parity.

### WI-B2 — Compile-per-Parcae-iteration

Middle ground between current `compile_zones` (6 graphs) and Option Y (18 graphs):
one graph per iter spanning all shared_layers in that iter (3 graphs). Captures
cross-layer residual adds that are currently eager (the 29% of `aten::add_` that
Phase 2 classified as cross-boundary residuals).

**Approach:** rewrite `_run_shared_block` as a `torch.compile`'d function, compile
3× (one per iter to let injection + skip_gate code run eager between iters).

**Gate:** +3% tok/s minimum (covers overhead of larger compile units).

### WI-B3 — max-autotune mode

Set `TORCH_COMPILE_MODE=max-autotune` on the current compile_zones path.

**Gate:** +3% tok/s. Warmup cost documented but does not fail the gate.

### WI-B4 — aiter flash attention on Machine B

Machine B has `aiter` installed (comfyui venv). aiter's flash_attn typically
1.2-1.5× faster than PyTorch's efficient attention for GQA-flavored shapes.

**Approach:**
1. Sync OdinHalo to Machine B via `sync_remote_b.sh`.
2. Toggle `_HAS_HYBRID_ATTN = True` in `models/components/attention.py` (per
   AGENTS.md currently forced False).
3. Verify correctness via 1-step parity vs Machine A baseline.
4. Measure tok/s.

**Gate:** +3% tok/s. If win, decide whether to port aiter to Machine A's venv or
keep production training on Machine B.

### WI-B5 — rocBLAS kernel selection audit

Inductor profile's mm-summary (from WI6 cache dump) showed the rocBLAS kernels
selected for each matmul shape. Audit whether the chosen `Cijk_*` tiles are the
best available for gfx1151.

**Approach:**
1. Extract rocBLAS tile selections from the Inductor log we already captured.
2. Compare against rocBLAS's published gfx1151 tune database (if available).
3. If mismatches found: set `ROCBLAS_LAYER=4` + `ROCBLAS_LOG_PATH` to collect
   runtime selection data and match.

**Gate:** +2% tok/s. Low ceiling but cheap to try.

### Track B exit

- Any WI passes stretch gate: land winner, exit.
- No WI passes stretch gate but shipping gate met via multiple tiny wins: land
  combined stack, document.
- Nothing clears shipping gate: close Phase 3 with `docs/perf/phase3-blocked.md`.

---

## Landing phase

Triggered when any WI wins its gate.

1. Production-readiness of winner code:
   - Remove opt-in debug flags
   - Document known incompatibilities (e.g., chunked-CE / smoke test)
   - Re-verify across all 8 HALO model variants via
     `scripts/test_all_models_chunked_train.py`
2. Combined ablation via `scripts/ablation_final.py` for STATUS.md row.
3. Quality verification: extended 1000-step smoke test (not 200) with loss logged
   every 50 steps.
4. Documentation:
   - STATUS.md row
   - AGENTS.md compile-strategy section update
   - `docs/perf/phase3-summary-YYYY-MM-DD.md`

---

## Stop conditions

- Shipping gate met: land winner, end.
- All WIs evaluated without meeting shipping gate: close with
  `phase3-blocked.md` documenting every attempted lever. Parallels Phase 2's
  valid-no-action outcome.
- Quality gate failure during any WI: abort that WI, proceed to next.

---

## Out-of-scope (requires further user input before attempting)

- Change `mean_recurrence`, `d_model`, or any model-architecture knob (training-
  quality changes, not Phase 3 scope).
- Port OdinHalo to Machine B permanently (WI-B4 measures; moving production
  needs user decision).
- Chase +5% via numerical precision changes (fp8, lossy rounding). Not reachable
  on gfx1151 anyway.

---

## Execution order

1. WI-A0 (investigation) — start immediately
2. Based on A0 outcome: WI-A1 or skip to landing
3. If A1 fails: WI-A2 (Option Z)
4. If A2 fails: WI-A3 (Option Y)
5. If Track A doesn't stretch: WI-B1 → WI-B2 → WI-B3 → WI-B4 → WI-B5 in that order
6. Landing phase when winner found OR close Phase 3 when exhausted
