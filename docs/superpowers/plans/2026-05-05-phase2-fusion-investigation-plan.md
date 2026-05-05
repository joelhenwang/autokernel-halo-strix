# Phase 2 Implementation Plan: Fusion Investigation

**Companion to:** `2026-05-05-phase2-fusion-investigation-design.md`
**Input artifact:** `docs/perf/odinhalo-profile-2026-05-05-compile/profiler.md`
**Estimated effort:** 2-5 days depending on how many candidates clear the gain threshold
**Expected gain:** **+0-5%** on top of post-Phase-1 baseline (14,577 tok/s). Realistic, because Inductor already fuses most of the obvious patterns under `compile_zones`.

## Candidate enumeration from Phase 1 profile

After subtracting profiler/compile-infrastructure overhead (`ProfilerStep*`, `Call CompiledFxGraph ...`, which sum to ~24% and are measurement artifacts, not real cost), the real GPU-time breakdown at batch=16 is approximately:

| Category | ~% of real time | Notes |
|----------|----------------:|:------|
| matmul (rocBLAS) | 27% | `aten::mm` + `Cijk_*` tiles. Cannot beat rocBLAS. Skip. |
| copy | 28% | Dominated by `triton_poi_fused__to_copy_mul_transpose_view_8` (already Inductor-fused). |
| elementwise | 25% | `aten::add_`, `aten::embedding_dense_backward`, various `triton_poi_fused_*`. |
| conv | 2.4% | `causal_conv1d` (DaoAILab extension). Below threshold. |
| optimizer | 2% | `_fused_adamw_`. Single kernel. |
| attention | 1.1% | log_softmax. Below threshold. |
| other | 14% | `Memset`, `Memcpy HtoD`. |

### Candidates enumerated (ops > 2% of measured wall time)

| # | Op | % | Decision path |
|--:|----|--:|:--------------|
| C1 | `triton_poi_fused__to_copy_mul_transpose_view_8` (two instances) | 9.1% | **Already Inductor-fused.** Check if transpose is avoidable by dataflow restructure. |
| C2 | `aten::mm` (eager) | 8.5% | rocBLAS. No action. |
| C3 | `aten::add_` | 4.9% | Residual/grad add in autograd accumulation. Likely Inductor-uncatchable (crosses graph boundaries). Investigate. |
| C4 | `aten::copy_` | 4.4% | Autocast dtype casts + contiguity ops. Investigate specific call sites. |
| C5 | `aten::embedding_dense_backward` | 4.1% | Tied embedding gradient (74 μs/call × 3 = huge per-call). Fusable with lm_head backward. |
| C6 | `Memset (Device)` | 4.1% | Buffer zero-init (grad_weight accumulator in chunked CE?). Possibly eliminable. |
| C7 | `Memcpy HtoD` | 4.0% | Input batch upload. Can be hidden by async prefetch. |
| C8 | `Cijk_*` rocBLAS matmuls | 8.4% combined | No action (rocBLAS). |

Candidates below 2% are logged to `docs/perf/fusion-deferred.md` at start without investigation.

## Work items in priority order

Priority = (expected gain) × (confidence) / (effort). Ranked.

### WI1 — Characterize C1 (triton transpose-copy) [investigation-only]

Goal: understand whether the 9% spent in `triton_poi_fused__to_copy_mul_transpose_view_8` represents unavoidable memory movement or a dataflow redundancy we introduced.

**Steps:**
1. Dump Inductor triton output with `TORCH_LOGS=output_code` for the HyPEShortConvBlock and find the specific source line that emits this fused kernel.
2. Identify the upstream ops: what `.to()`, `.mul(...)`, `.transpose(...)`, and `.view()` chain produces it? Likely the `causal_conv1d` interface (which uses channel-last layout, requiring `.transpose(1,2)` before and after the HIP kernel).
3. If cause is confirmed to be `causal_conv1d`'s transpose plumbing: document as unavoidable and close.
4. If cause is avoidable (e.g., an autocast dtype cast that can be moved), scope a targeted fix.

**Acceptance:** either a root-cause writeup in `docs/perf/c1-transpose-copy-analysis.md` or a proposed fix (new WI added).

**Estimated:** 3 hours.

### WI2 — Profile `aten::add_` call sites (C3) [investigation]

`aten::add_` at 4.9% (402 calls × 0.66 μs/call) is suspicious — this is likely happening OUTSIDE compile_zones at layer boundaries (the `residual + layer_out` pattern at the top level of OdinHalo's forward, or gradient accumulation in backward).

**Steps:**
1. Add targeted `torch.profiler` stacks: set `record_shapes=True` and inspect where the 402 `aten::add_` calls originate.
2. Classify: (a) autograd accumulation (cannot be fused), (b) residuals inside compiled regions (already Inductor-fused; this count is for eager portions only), (c) something else.
3. For (b), check whether the compile boundary between shared_layers can be moved to include more residuals.

**Acceptance:** call-site classification; proposed fix if any category has >1% expected gain.

**Estimated:** 2 hours.

### WI3 — Attempt fusion: embedding_dense_backward + lm_head backward (C5)

The `aten::embedding_dense_backward` takes 74 μs/call × 3 = ~4% per step. It implements the sparse gradient accumulation for `tok_embeddings.embed` when targets index into the embedding table. Because OdinHalo uses a tied lm_head (`FactorizedLMHead.embed_table` shares weights with `tok_embeddings.embed`), BOTH the embedding-lookup grad AND the lm_head-matmul grad must land in the same parameter's `.grad`.

Investigation:
1. Check whether chunked CE's backward currently writes to `embed_table.weight.grad` directly (via matmul) AND then `embedding_dense_backward` re-runs for the input-lookup path. If yes, the two could be merged.
2. Alternative: replace `embedding_dense_backward` with a manual scatter-add using precomputed per-position gradients from chunked CE.

**Risk:** tied-embedding autograd is subtle; numerical drift could cost us the chunked CE gradient-parity guarantee. Requires strict parity test.

**Acceptance gate:** parity test (gradient L2 match to 1e-3 relative of baseline) + isolated ablation ≥ +0.5% end-to-end.

**Estimated:** 6-8 hours if doable, or 2 hours to abort if infeasible.

### WI4 — Eliminate `Memset` zero-init (C6)

`Memset (Device)` at 4.06% (16 calls × 14 μs) is likely the `torch.zeros_like(weight)` allocations inside `ChunkedLinearCrossEntropy.forward` (`grad_weight` and possibly intermediate fp32 accumulators). These happen every step.

**Steps:**
1. Confirm by reading `kernels/hip/chunked_linear_cross_entropy.py` and identifying `torch.zeros_like` / `torch.empty_like` sites.
2. Replace per-step allocations with persistent buffers owned by the `ChunkedLinearCrossEntropyLoss` module (allocated on first call, reused thereafter). On backward completion, zero them in-place via `grad_buffer.zero_()` (still a memset but at least it's one predictable site).
3. If some memsets come from PyTorch autograd internals (unavoidable), document and close.

**Acceptance:** isolated measurement showing Memset ratio drops. Ablation ≥ +0.5% end-to-end to ship.

**Estimated:** 3 hours.

### WI5 — Async data prefetch to hide `Memcpy HtoD` (C7)

`Memcpy HtoD` at 3.95% (6 calls × 36 μs per step) is the input batch upload. With `non_blocking=True` already set (W6), this CAN be overlapped with compute — but only if there's a pre-fetched next-batch on CPU when the GPU is ready. Currently the loop does `batch = next(it); batch.to(device)` synchronously.

**Steps:**
1. Implement a simple `CUDAPrefetcher` wrapper in `halo_training/data.py`:
   - On construction, grabs a batch and issues `.to(device, non_blocking=True)` into pinned-memory staging tensors.
   - `next()` returns the current GPU batch and kicks off upload of the next one.
   - Uses `torch.cuda.Stream` for the upload stream to overlap with compute.
2. Wire into trainer via opt-in flag `--prefetch` initially; promote to default if no regression.

**Risk:** unified memory on Strix Halo may make `non_blocking=True` a no-op; prefetching via explicit streams may or may not help. Quick measurement decides.

**Acceptance:** isolated measurement at batch=16 shows `Memcpy HtoD` time decreases. Ablation ≥ +0.5% end-to-end.

**Estimated:** 3 hours.

### WI6 — Inductor-fusion output dump and diff [investigation]

Goal: make a canonical record of what Inductor is fusing so future agents know what NOT to re-implement.

**Steps:**
1. Run OdinHalo forward with `TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-phase2 TORCH_LOGS=output_code python scripts/profile_step_deep.py --compile`.
2. Capture stdout to `docs/perf/inductor-triton-dump.log`.
3. Extract top 20 `triton_poi_fused_*` and `triton_per_fused_*` kernels, list the ops each fuses.
4. Write `docs/perf/inductor-fusion-catalog.md` summarizing what's fused vs what's separate.

**Acceptance:** catalog committed. Informs future kernel decisions permanently.

**Estimated:** 2 hours.

## Sequencing and gates

```
Day 1: WI6 Inductor catalog (2h) ──► gives concrete evidence of what's already fused
       WI1 C1 transpose analysis (3h) ─► likely shows the 9% is causal_conv1d plumbing
       WI2 add_ call-site analysis (2h) ─► likely mostly autograd-internal, close
                                           |
Day 2: WI4 Memset elimination (3h) ─ if shows ≥ 0.5% gain, ship
       WI5 Async prefetch (3h)      ─ if shows ≥ 0.5% gain, ship
                                           |
Day 3+: WI3 embedding_backward fusion (6-8h) ─ highest risk, highest payoff
                                                └─ parity test mandatory before merge
                                           |
Day N (end): Combined ablation + STATUS.md addendum
```

Each shipped fusion goes through the per-fusion gate list from the spec:
1. Unit parity
2. Shape parity at OdinHalo production shapes
3. Gradient parity
4. Gradient shape parity
5. End-to-end 5-step training parity (loss trajectory match)
6. Ablation gate ≥ +0.5% tok/s

## Stop conditions

- 5 working days elapsed, OR
- All enumerated candidates (WI1-WI6) have been evaluated.

Phase 2 **may ship with zero fusions** if Inductor already captures everything profitable. That is a valid outcome — codified in the design spec.

## Hand-off to Phase 3

Phase 3 (CUDA graphs through Parcae) is independent and can start in parallel. When Phase 3 baselines tok/s, it uses whatever stack is current (post-Phase-2 or not).

## Scripts to create

- `scripts/classify_ops.py` — reads `profiler.md`, emits candidates CSV (from Phase 2 spec).
- `scripts/check_inductor_fusion.py` — runs compile with `TORCH_LOGS=output_code`, annotates candidates.
- `scripts/ablation_phase2.py` — per-fusion before/after tok/s measurement at batch=16.
- Per-fusion parity tests as needed: `scripts/test_<name>_parity.py`.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Inductor already fuses everything worth fusing — Phase 2 ships nothing | Design spec explicitly allows this outcome. Still produces the Inductor-catalog (WI6) as useful artifact. |
| Embedding-backward fusion (WI3) breaks tied-weight autograd accumulation | Strict parity test; revert and log if fails. |
| Async prefetch (WI5) shows no gain on unified memory | Document and close; 3h loss at worst. |
| Profile is stale after future codebase changes | Re-run W1 profile tooling any time; it's a reusable script. |

## Hand-off artifacts

- `docs/perf/fusion-deferred.md` — every considered-and-rejected candidate with reason.
- `docs/perf/phase2-results-<date>.md` — per-fusion ablation results.
- `docs/perf/inductor-fusion-catalog.md` — what Inductor fuses (permanent reference).
- STATUS.md post-Phase-2 row.

## Execution start

Ready to execute. Recommended first step: WI6 Inductor catalog (2h, lowest risk, produces evidence that informs every subsequent decision).
