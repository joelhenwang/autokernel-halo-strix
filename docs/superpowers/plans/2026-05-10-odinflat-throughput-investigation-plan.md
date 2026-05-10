# OdinFlat Throughput Investigation — Execution Plan

**Date:** 2026-05-10
**Author:** opencode session (Sonnet 4.7)
**Status:** LOCKED, ready to execute
**Sequence:** Track 1 (profile) → Track 2 (QKV fusion) → Track 3 (deep autokernel investigation, 6 sub-phases) → synthesis doc
**Wall time estimate:** ~2 days (12-15h dev + ~4h compute)
**Blocks:** Sprint 3A launch until complete
**Prior context:**
- `docs/perf/odinflat-throughput-final.md` (investigation summary)
- `docs/perf/odinflat-rmsnorm-fix.md` (Phase III rmsnorm autograd fix)
- `docs/perf/odinflat-bisect-findings.md` (Phase II bisect)
- `docs/perf/odinflat-autokernel-inventory.md` (Phase I audit)
- `docs/perf/odinflat-compile-study.md` (Phase IV compile variants)
- `docs/perf/odinflat-option3-native-rmsnorm.md` (F.rms_norm swap)

---

## 1. Motivation and prior-art context

### 1.1 The ambient question

OdinFlat currently trains at ~31,331 tok/s steady under the Sprint 1.5 C3
recipe (μP + SPECTRA, no `--optimize-kernels`). A full-epoch Sprint 3A
run on dolma-10B at this throughput takes ~61 hours. The user's intuition
was that `--optimize-kernels` should unlock more throughput — OdinHalo's
Phase 0.3 probe showed +38% steady-state tok/s with loss parity. A prior
investigation (2026-05-10, committed `d31ed4e` through `0e4c23b`) drove
through Phases I-V of an autokernel bisect and reached a hard conclusion:

> Autokernel provides zero net benefit for OdinFlat under correct
> training semantics. The "+31% throughput" measurements were
> artifacts of silently broken autograd that froze ~60M `w_gate_up`
> parameters at init. V2 with autograd properly registered ran
> **slower** than baseline (30,976 vs 31,331 tok/s) because the
> save-for-backward and HIP backward kernel overhead exceeds the
> forward savings.

This conclusion is correct for the specific path investigated but
**does not establish 31.3K as a hard throughput ceiling**. Three
categories of unexplored optimization remain:

1. **Measurement-first:** we never profiled the step to see where
   time actually goes. Optimizer overhead (NorMuon Newton-Schulz
   on every 2D parameter) could be 20-30% of step time and is
   a different kind of target than kernel fusion.
2. **Architecture-level fusion:** QKV projections are split into 3
   separate Linears (`wq`, `wk`, `wv`). Collapsing them into one
   `wqkv` matmul is a standard optimization with expected 5-15%
   attention-forward speedup on rocBLAS.
3. **Deep understanding:** the "why" of the autokernel quality
   regression was established at surface level (missing autograd)
   but not mechanistically explained. What exactly freezes, how
   far the blast radius spreads, why V2 is slower, whether any
   HIP speedup is recoverable via a different integration path
   — all unanswered.

### 1.2 Why this plan exists

The user asked "is OdinFlat's 31 tok/s a ceiling?" The honest answer
was "31 K is the ceiling via autokernel/compile, but we haven't
explored other levers." They approved this three-track investigation
with:

- Sequential execution: Track 1 → Track 2 → Track 3
- Full depth on Track 3 (all sub-phases 3.A-3.F)
- Single synthesis doc as final deliverable
- Sprint 3A blocked until complete

### 1.3 What success looks like

By the end of this plan we will have:

1. **Quantitative per-section step breakdown** (Track 1) — we know
   exactly where wall time goes. No more guessing. Directs any
   further optimization work.
2. **QKV-fusion module** (Track 2) — if the profile shows attention-forward
   is material, a standard, safe throughput win (5-15% expected).
   If the profile shows optimizer dominates, Track 2 pivots or is
   skipped in favor of NorMuon caching.
3. **Definitive understanding of autokernel's failure mode on
   OdinFlat** (Track 3) — mechanistic explanation, blast radius
   evidence, recoverability assessment. Either a working alternative
   integration (Triton or autograd.Function) or a definitive "HIP
   kernels cannot help here" conclusion with reasons.
4. **Single synthesis document** future engineers can read to avoid
   repeating this investigation.

---

## 2. Global decisions already locked

Do not re-ask the user about these:

| Decision | Value | Rationale |
|---|---|---|
| Execution order | **Track 1 → 2 → 3 → synthesis** | Profile informs Track 2 scope; Track 2 is cheap win; Track 3 is deep understanding |
| Track 3 depth | **Full 3.A-3.F** | Definitive root-cause; user explicitly requested "deep" |
| Track 3 deliverable | **Single `docs/perf/autokernel-deep-analysis.md`** | One synthesis doc future engineers can read top-to-bottom |
| Sprint 3A timing | **BLOCKED until complete** | User explicit: no 3A launch until investigation closes |
| Sprint 3B | **Still planned with `--optimize-kernels`** per Phase 0.3 | BUT flagged for potential same-bug verification (see §6.1 follow-ups) |

---

## 3. Track 1 — Profile the training step

**Goal:** produce a quantitative per-section wall-time breakdown for
a single optimizer step under the Sprint 1.5 C3 recipe. Direct all
downstream optimization effort based on the result.

**Wall estimate:** ~45 minutes total (30 min dev + 15 min compute).

### Task 1.1 — Add `--profile-steps` flag to `train_ddp.py`

**File:** `scripts/train_ddp.py`

**Change:** add a new argparse flag:

```python
parser.add_argument("--profile-steps", type=str, default="",
                    help="Profile a step range in the training loop. "
                         "Format: 'start:end' (e.g. '30:40' profiles "
                         "steps 30 through 39 inclusive). Writes Chrome "
                         "trace to $CKPT_DIR/profile.json and flat op "
                         "table to $CKPT_DIR/profile-summary.txt. "
                         "Empty = no profiling.")
```

In the training loop, wrap the relevant step range with
`torch.profiler.profile(activities=[ProfilerActivity.CPU,
ProfilerActivity.CUDA], record_shapes=True, with_stack=True)`.

On step `end`, emit:
- `profile.json` (Chrome trace format via `prof.export_chrome_trace`)
- `profile-summary.txt` via `prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)`

**Exit criterion:** `python scripts/train_ddp.py --help` lists the
flag; setting `--profile-steps 5:10` on a short run produces both
output files under the checkpoint directory.

**Test:** `scripts/test_profile_flag.py` — spawn a 20-step OdinFlatMini
run with `--profile-steps 10:15`, assert `profile.json` and
`profile-summary.txt` exist in the ckpt dir with non-zero size.

**Rollback:** remove the argparse entry + the profiler wrapping block;
no other state changes.

### Task 1.2 — Write `scripts/profile_odinflat_step.sh`

**File:** NEW

**Change:** bash orchestrator that launches a 50-step OdinFlat DDP run
on dolma with `--profile-steps 30:40`. Config identical to the C3
baseline (Sprint 1.5 C3 recipe).

```bash
#!/bin/bash
set -euo pipefail
cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/odinflat-profile
rm -rf "$CKPT"; mkdir -p "$CKPT"

CKPT_DIR="$CKPT" \
MODEL=models/odin_flat.py \
CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 BLOCK=512 BATCH=16 ACCUM=8 \
WARMUP_STEPS=100 CHECKPOINT_INTERVAL=9999 MAX_GRAD_NORM=1.0 \
EXTRA_FLAGS='--max-steps 50 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 --profile-steps 30:40' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

# Wait for Done:
while true; do
    sleep 15
    grep -q '^Done:' "$CKPT/rank0.log" 2>/dev/null && break
    grep -q '^Traceback' "$CKPT/rank0.log" 2>/dev/null && { echo FAIL; tail -10 "$CKPT/rank0.log"; exit 1; }
done

echo "=== Profile summary ==="
head -40 "$CKPT/profile-summary.txt"
```

**Exit criterion:** produces `$CKPT/profile.json` (>1 MB) and
`$CKPT/profile-summary.txt` (at least 40 ops listed).

**Rollback:** delete the script and checkpoint directory.

### Task 1.3 — Analyze and write breakdown table

**Files:**
- `scripts/analyze_profile.py` (NEW) — helper that parses the
  profile summary and outputs a per-category roll-up
- `docs/perf/odinflat-step-profile.md` (NEW) — human-readable writeup

**Categorization rules** (apply in `scripts/analyze_profile.py`):

| Category | Op name patterns (substring) |
|---|---|
| Forward: attention | `aten::bmm`, `aten::matmul`, `aten::scaled_dot_product_attention`, `SDPA`, `softmax`, `dropout` within `NoPEGQABlock` |
| Forward: SwiGLU | `aten::silu`, `aten::mul` on FFN tensors, `aten::linear` within `SwiGLU` |
| Forward: RMSNorm | `aten::rms_norm`, `aten::pow`, `aten::rsqrt`, `aten::mean` |
| Forward: embedding / head | `aten::embedding`, `aten::linear` on `tok_embeddings` or `lm_head` |
| Backward | any `*Backward` variant |
| Optimizer: NorMuon | `NewtonSchulz`, matmuls inside `halo_training.normuon` |
| Optimizer: AdamW + SPECTRA | ops inside `apply_post_clip`, `AdamW.step`, `fused_adam` |
| DDP allreduce | `c10d::allreduce`, `ncclAllReduce`, `ProcessGroupGloo` |
| Misc | everything else |

Produce a table:

| Section | Wall % | CUDA time (ms) | Top 3 ops |
|---|---:|---:|---|
| Forward: attention | ? | ? | ? |
| ... | | | |

**Exit criterion:** `docs/perf/odinflat-step-profile.md` exists with
filled table + 3-5 bullet-point observations (e.g. "optimizer takes
22% — QKV fusion upside capped at ~10% total wall"). Chrome trace
JSON archived under `docs/perf/` for future inspection.

**Test:** manual — inspect the summary table for sanity (percentages
sum to 95-105%; no single op > 40% unless expected like attention SDPA).

**Rollback:** delete the script + doc; profile data can be kept for
later re-analysis.

### Track 1 GATE

**Decision point after Task 1.3:**

| Profile outcome | Track 2 decision |
|---|---|
| Attention forward > 25% of step | Proceed with QKV fusion as planned |
| Optimizer > 40% of step | Pivot Track 2: NorMuon NS caching instead of QKV fusion |
| Forward ≈ 40%, Backward ≈ 35%, Optimizer ≈ 20%, other ≈ 5% (expected) | Proceed with QKV fusion; plan NorMuon as future work |
| Something surprising (e.g. DDP allreduce dominates) | Stop, report to user, re-plan |

**Do NOT silently pivot** — always report the finding and confirm
scope with the user before executing Track 2.

---

## 4. Track 2 — QKV fusion

**Goal:** collapse `wq`, `wk`, `wv` in `NoPECodaAttention` (and its parent
`CodaAttention` / `Attention`) into a single `wqkv` Linear. Produces a
larger but equally-hot matmul that runs faster on rocBLAS than three
separate smaller matmuls.

**Wall estimate:** 2-4 hours dev + ~15 min compute.

**Precondition:** Track 1 profile shows attention forward ≥ 25% of step
wall time. If not, revisit with user.

### Task 2.1 — Inspect current attention layout

**Files to read (no edits):**
- `models/components/attention.py` — three classes: `Attention`,
  `CodaAttention` (extends Attention, adds XSA + MoDA),
  `NoPECodaAttention` (extends Attention, no RoPE, Sprint 1 kwargs)
- `models/components/conv_blocks.py` — check whether HyPEShortConvBlock
  uses any attention (it doesn't, it's all conv; skip)

**Observations to record** in a comment at the top of `attention.py`:
- Current layout: `self.wq = nn.Linear(dim, n_heads * head_dim)`,
  `self.wk = nn.Linear(dim, n_kv_heads * head_dim)`,
  `self.wv = nn.Linear(dim, n_kv_heads * head_dim)`, all `bias=False`
- GQA sizes: for OdinFlat d=768, `n_heads=12`, `n_kv_heads=4`,
  `head_dim=64`; so output sizes are `12*64=768`, `4*64=256`, `4*64=256`
- Post-projection shape manipulation: `q.view(B, T, n_heads, head_dim)`,
  etc.

### Task 2.2 — Introduce `self.wqkv` in base `Attention` class

**File:** `models/components/attention.py`

**Change:** replace the three Linear declarations with a single one.
Inside `__init__`:

```python
# Fused QKV projection (2026-05-10): single matmul replaces three.
# Output layout: [q_dim, k_dim, v_dim] along last axis, where
# q_dim = n_heads * head_dim, k_dim = v_dim = n_kv_heads * head_dim.
self._q_dim = n_heads * self.head_dim
self._kv_dim = n_kv_heads * self.head_dim
self.wqkv = nn.Linear(dim, self._q_dim + 2 * self._kv_dim, bias=False)
self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
```

In the forward path of each subclass, replace:

```python
q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
```

with:

```python
qkv = self.wqkv(x)
q, k, v = qkv.split(
    [self._q_dim, self._kv_dim, self._kv_dim], dim=-1)
q = q.view(B, T, self.n_heads, self.head_dim)
k = k.view(B, T, self.n_kv_heads, self.head_dim)
v = v.view(B, T, self.n_kv_heads, self.head_dim)
```

**Exit criterion:** all three attention classes use `self.wqkv`;
`self.wq`, `self.wk`, `self.wv` no longer exist.

**Test:** unit test in `scripts/test_qkv_fusion.py` (see Task 2.4).

**Rollback:** revert `models/components/attention.py` from git.

### Task 2.3 — Add state-dict migration hook

**File:** `models/components/attention.py` (inside `Attention`)

**Change:** add `_load_from_state_dict` override that detects old
split-style keys and fuses them:

```python
def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                          strict, missing_keys, unexpected_keys, error_msgs):
    # Sprint 3A/earlier checkpoints have wq.weight / wk.weight / wv.weight
    # instead of wqkv.weight. Fuse them on load to preserve compatibility.
    wq_key = prefix + "wq.weight"
    wk_key = prefix + "wk.weight"
    wv_key = prefix + "wv.weight"
    wqkv_key = prefix + "wqkv.weight"
    if wqkv_key not in state_dict and all(
            k in state_dict for k in (wq_key, wk_key, wv_key)):
        wq = state_dict.pop(wq_key)
        wk = state_dict.pop(wk_key)
        wv = state_dict.pop(wv_key)
        state_dict[wqkv_key] = torch.cat([wq, wk, wv], dim=0)
    super()._load_from_state_dict(
        state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs)
```

**Exit criterion:** loading a pre-fusion checkpoint (e.g.
`checkpoints/sprint3a-confirm-v1-2000/step_2000.pt`) succeeds with
the new code, and a forward pass on the loaded model produces the
same loss on a fixed batch as the pre-fusion code path did.

**Test:** `scripts/test_qkv_fusion.py::test_state_dict_migration` —
see Task 2.4.

**Rollback:** remove the hook; users would need to re-train from scratch.

### Task 2.4 — Write `scripts/test_qkv_fusion.py`

**File:** NEW

**Tests:**

1. `test_forward_numerical_equivalence` — instantiate an old-style and
   new-style `NoPECodaAttention`, copy weights (via the migration hook
   semantics in reverse), run forward on the same input under autocast
   fp16, assert outputs match within fp16 precision.

2. `test_state_dict_migration` — build an OdinFlat from the pre-fusion
   checkpoint `sprint3a-confirm-v1-2000/step_2000.pt`, load into the
   new-fused OdinFlat, forward one batch, compare loss to the same
   forward done with pre-fusion code.

3. `test_wqkv_shape_conventions` — for OdinFlat's 12-head 4-kv-head
   config, assert `wqkv.weight.shape == (768+256+256, 768)`.

4. `test_split_reassembles` — dummy weights, run `qkv = wqkv(x)`,
   split, then reassemble via cat, verify exact bitwise identity.

**Exit criterion:** all 4 tests pass on remote with CUDA.

**Rollback:** delete `scripts/test_qkv_fusion.py`; no other effect.

### Task 2.5 — Verify autokernel interaction

**Files to inspect:** `autokernel/_patterns.py`

autokernel's `FusedQKVPattern` looks for the split-style attrs
(`wq`, `wk`, `wv`, `wo`). After fusion, that pattern won't match
(no `wq`), so it becomes a no-op on OdinFlat's attention. Confirm
this by either:

- Running `scripts/diag_autokernel_patterns.py` (already exists from
  Phase I) on the fused OdinFlat. Expect `fused_qkv_attention` to
  report 0 modules replaced.
- If the pattern still tries to match via a heuristic, add
  `_skip_autokernel = True` on the fused `Attention` class as a
  backstop (consistent with Phase 0 escape hatch).

**Exit criterion:** `diag_autokernel_patterns.py` on fused OdinFlat
reports the same pattern set as pre-fusion (rmsnorm + fused_silu_gate_mul)
with no new patterns triggered unexpectedly.

### Task 2.6 — Throughput probe

**File:** `scripts/probe_qkv_fusion.sh` (NEW)

**Change:** 200-step OdinFlat DDP probe on dolma, Sprint 1.5 C3 recipe,
no `--optimize-kernels` (to isolate the fusion effect). Mirror the
format of `scripts/bisect_autokernel.sh` probes.

**Exit criterion:** produces a `rank0.log` with clean `Done:` line.

### Task 2.7 — Analysis and ship decision

**Deliverable:** `docs/perf/odinflat-qkv-fusion.md` with:

- Throughput before/after table (steady tok/s, aggregate tok/s)
- Loss @ step 50, 100, 150, 200 comparison
- Gate: ship if **tok/s lift ≥ 3% AND loss delta within ±0.05 at
  step 200**.
- If gate passes: mark module change as production-ready. Keep
  committed.
- If gate fails: revert the attention.py change, keep the migration
  hook + unit tests for future. Document why (e.g. "fusion gave
  +1% which was within noise; attention isn't the bottleneck").

**Rollback plan:** migration hook remains even on failure so no
user is stranded if we later re-enable fusion.

---

## 5. Track 3 — Deep autokernel quality investigation

**Goal:** definitive mechanistic understanding of why autokernel
degrades training quality on OdinFlat, even after the Phase III
autograd fix. Produce a generalizable principle for future kernel
authors and determine whether any HIP speedup is recoverable.

**Wall estimate:** 1-1.5 days (~10-13 hours dev + ~2 hours compute).

### 5.A — Quantify exactly what freezes (~2h)

**Files:**
- `scripts/diag_frozen_params.py` (NEW) — per-param `.grad.norm()`
  recorder
- `docs/perf/autokernel-frozen-blast-radius.md` (NEW)

**Method:**

1. Run 50 training steps in **three configurations**, hooking every
   parameter's `.grad` immediately after `scaler.scale(loss).backward()`:
   - V0: no `--optimize-kernels` (baseline; all params should get
     non-zero grads)
   - V1: `--optimize-kernels` with both rmsnorm and silu HIP
     replacements active (expected: ~60M `w_gate_up.weight` frozen)
   - V3 (new): `--optimize-kernels --autokernel-exclude
     fused_silu_gate_mul` — ONLY rmsnorm HIP with the Phase III
     autograd fix. Should show full gradient flow if fix is complete.

2. Record `.grad.norm()` per parameter at steps 5, 10, 25, 50.
   Output: per-parameter heatmap showing (step × parameter) with
   grad norm color-coded.

3. Identify:
   - Which parameters have `grad=None` (never touched by autograd)
   - Which have `grad=0.0` (touched but zero gradient)
   - Which have finite nonzero grads
   - Trajectory over 50 steps

**Exit criterion:** heatmap rendered to PNG (or ASCII table) +
summary in markdown. Shows exactly which params are frozen per config.

**Hypotheses tested:**
- H1: V1 freezes `w_gate_up` across all 14 layers → confirms
  silu autograd disconnect
- H2: V1 also freezes `attn.wq/wk/wv` (upstream of blocked SwiGLU
  via residual path) or does NOT freeze them (they still get grad
  via the attention path, which is separate from SwiGLU's output)
- H3: V3 (rmsnorm-autograd-fixed, silu excluded) has full gradient
  flow → confirms Phase III fix is correct

### 5.B — Mechanism writeup: pybind raw vs custom_op autograd (~2h)

**Files:** `docs/perf/autokernel-autograd-mechanism.md` (NEW)

**Content outline:**

1. **The dispatch path for a "normal" PyTorch op:**
   - Python call → torch dispatcher → ATen backend → kernel
   - During tracing, dispatcher wraps with `AutogradFunction` node
   - Backward uses the inverse dispatch path

2. **The dispatch path for `torch.library.custom_op` + `register_autograd`:**
   - Python call → dispatcher recognizes the custom op
   - Dispatcher wraps with a synthesized `AutogradFunction` whose
     forward calls our Python impl and whose backward calls our
     registered backward callable
   - Graph node created; `grad_fn` set on output

3. **The dispatch path for a raw pybind C++ call (e.g.
   `kernels.hip.rmsnorm.kernel_fn`):**
   - Python call → pybind11 stub → C++ function body
   - C++ creates `torch::Tensor` via `torch::empty(...)`
   - No dispatcher involvement; no autograd node registration
   - Output tensor has `requires_grad=False` even if inputs did;
     no `grad_fn`
   - Subsequent ops that use this tensor as input lose gradient
     flow from inputs

4. **Why this isn't a crash — why "broken training" looks fine short-term:**
   - Downstream ops (e.g. `w_down(activated)`) still compute
     `w_down.weight.grad = grad_out.T @ activated` correctly, because
     `w_down.weight` is a leaf parameter and that gradient formula
     doesn't need `activated` to have `grad_fn`
   - Only the UPSTREAM gradient to `activated` (and from there to
     `gate`, `up`, `w_gate_up`) is lost
   - Loss still descends because `w_down` can adapt; but 60M params
     never update

5. **The "+31% throughput" illusion:**
   - Forward skipping 60M params' backward pass is free speed
   - Correct autograd must compute that backward, bringing throughput
     back to or below baseline
   - Therefore: any HIP kernel integration measured at "+X% throughput"
     where X > 0 either broke autograd OR the reference baseline was
     sub-optimal (Inductor could match the HIP speed)

6. **Principle for future kernel authors:**
   - NEVER use raw pybind kernel_fn calls in a training path
   - ALWAYS wrap via `torch.library.custom_op` + `register_autograd`
   - If the wrapping overhead dominates, the kernel isn't faster
     than PyTorch native under Inductor; stop and reconsider
   - Forward-only kernels (inference paths) are fine with raw
     pybind, but guard against accidental training use

**Exit criterion:** doc committed, reviewable as standalone reading.

### 5.C — V2 overhead decomposition (~1.5h)

**Files:**
- `scripts/diag_v2_overhead.py` (NEW) — torch.profiler on V2 config
- Addition to `docs/perf/autokernel-deep-analysis.md` (§C) with the
  results

**Method:**

1. Run 50-step OdinFlat with `--optimize-kernels
   --autokernel-exclude rmsnorm`. This is the V2 config where silu
   HIP is autograd-registered via `torch.ops.autokernel.silu_gate_mul`.
2. Wrap steps 30-40 in profiler with `with_stack=True`.
3. Analyze the profile for the silu forward + backward path:
   - `autokernel::silu_gate_mul` dispatcher time
   - `save_for_backward` save cost
   - `silu_gate_mul_hip` kernel launch overhead
   - `silu_gate_mul_backward_hip` kernel launch overhead
   - Compare to pure-PyTorch silu forward + backward (run V0 config
     and profile same range)
4. Identify the largest overhead source. Expected suspects:
   - Python-level dispatcher overhead (~microseconds per op, but
     multiplied by 14 layers × many backward calls)
   - HIP kernel launch latency vs Inductor-fused launch
   - Save-for-backward creating extra tensor refs

**Exit criterion:** quantified overhead table. Answer to "is there a
path where HIP silu can actually speed training up?":
- If dispatcher dominates → try `autograd.Function` direct (Task 5.E)
- If backward kernel latency dominates → HIP is architecturally bad
- If save-for-backward dominates → we could rematerialize in backward
  instead of saving (trade memory for time)

### 5.D — Quality degradation mechanism (~1.5h)

**Files:**
- `scripts/diag_weight_activations_v1.py` (NEW)
- `docs/perf/odinflat-v1-degeneracy-analysis.md` or addition to main
  synthesis doc

**Method:**

1. Take V1's step 2000 checkpoint: `sprint3a-confirm-v1-2000/step_2000.pt`
2. Direct weight inspection:
   - For every layer's `w_gate_up.weight`, compute:
     - Norm (should be ≈ init if frozen)
     - Cosine similarity to a fresh init (should be ≈ 1.0 if frozen)
     - Per-row norm distribution
   - For every layer's `w_down.weight`, same metrics (expected to
     have trained normally)
   - For attention Q/K/V and output projections: same
3. Activation statistics on a batch:
   - Per-layer gate/up tensor magnitudes
   - Per-layer attention output magnitudes
   - Compare to baseline checkpoint at same step (if available) or
     to a freshly-initialized model
4. Load-into-baseline sanity check:
   - Take V1 step-2000 checkpoint, load into an OdinFlat configured
     without `--optimize-kernels`
   - Run one forward pass with a deterministic batch
   - Compare loss to the V1 rank0.log's reported step-2000 loss
   - If they match: V1's checkpoint is legitimate (just trained with
     frozen params)
   - If they differ significantly: there's a numerical drift issue
     beyond just the freeze

**Exit criterion:** confirms or refutes the "silent freeze" hypothesis
with direct weight/activation evidence.

**Expected outcome:**
- `w_gate_up` cosine-to-init ≈ 1.0 (frozen): freeze hypothesis confirmed
- `w_down` cosine-to-init < 0.8 (trained): freeze hypothesis is
  partial — only upstream-of-broken-node freezes
- Loss matches between V1 log and re-load: no numerical drift beyond
  freeze

### 5.E — Recovery attempt: `torch.autograd.Function` variant (~2-3h)

**Files:**
- `autokernel/_patterns.py` — experimental alternative replacement
  class `_FusedSwiGLUReplacementAutogradFunc` (do not replace the
  existing class; make it opt-in via a kwarg)
- `scripts/probe_qkv_fusion.sh`-style `scripts/probe_autograd_func_silu.sh` (NEW)

**Rationale:** `torch.library.custom_op` has non-trivial Python
dispatcher overhead. The older `torch.autograd.Function` API has
lower overhead because it directly constructs the autograd graph
node in Python without going through the dispatcher.

**Change (experimental):**

```python
class _SiluGateMulAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, fwd_fn, bwd_fn):
        ctx.save_for_backward(gate, up)
        ctx.bwd_fn = bwd_fn
        return fwd_fn(gate.contiguous(), up.contiguous())

    @staticmethod
    def backward(ctx, grad_out):
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = ctx.bwd_fn(gate, up, grad_out)
        return grad_gate, grad_up, None, None


class _FusedSwiGLUReplacementFn(nn.Module):
    def __init__(self, original, fwd_fn, bwd_fn):
        super().__init__()
        self.w_gate_up = original.w_gate_up
        self.w_down = original.w_down
        self._fwd_fn = fwd_fn
        self._bwd_fn = bwd_fn

    def forward(self, x):
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        if gate.dtype == torch.float16:
            activated = _SiluGateMulAutogradFunc.apply(
                gate, up, self._fwd_fn, self._bwd_fn)
        else:
            activated = F.silu(gate) * up
        return self.w_down(activated)
```

With a constructor flag on `FusedSwiGLUPattern` to choose between
`_FusedSwiGLUReplacement` (current, broken) and
`_FusedSwiGLUReplacementFn` (new, experimental).

**Probe:** `scripts/probe_autograd_func_silu.sh` — same config as V2
but with this new replacement path. 200 steps, measure tok/s + loss
@ 200.

**Decision matrix:**

| Outcome | Action |
|---|---|
| tok/s > V2 (30.9K) + loss parity | Ship as new default; Sprint 3A gains real throughput |
| tok/s ≤ V2 + loss parity | HIP kernel is architecturally slow; no fix possible without Triton |
| tok/s arbitrary + loss regression | Autograd.Function is ALSO buggy somehow; investigate |

**Exit criterion:** quantitative answer to "is there a lower-overhead
way to keep HIP silu fast?". Either a working speedup (great) or a
definitive "no" with evidence.

**Rollback:** if inferior, keep the experimental class in the codebase
behind a flag (for future reference) but don't enable by default.

### 5.F — Synthesis document (~1h)

**File:** `docs/perf/autokernel-deep-analysis.md` (NEW, final deliverable)

**Structure:**

1. **Executive summary** (1 page)
   - The question, the bottom line, the three actionable conclusions

2. **Context and prior findings** (1/2 page)
   - Pointer to `odinflat-throughput-final.md` for the surface
     investigation
   - Why this deep dive was necessary

3. **Mechanism: how autograd breaks** (from 5.B)

4. **Blast radius: what actually freezes** (from 5.A)

5. **Quality degradation: what the frozen regime produces** (from 5.D)

6. **Why V2 overhead exceeds forward savings** (from 5.C)

7. **Recoverability: autograd.Function experiment** (from 5.E)

8. **Principle + recommendations**
   - Rule for future kernel authors: pybind raw is training-hostile
   - Rule for model/training code reviewers: flag any raw pybind call
     inside a module whose output feeds gradient-carrying ops
   - Recommendation for autokernel's current state:
     - `_RMSNormReplacement`: keep (Phase III fix); doesn't help
       throughput but doesn't hurt
     - `_FusedSwiGLUReplacement`: either document as
       inference-only, or remove, or fix per Task 5.E outcome
     - Consider: add a runtime assertion that raises a visible
       warning when `autokernel.optimize(model, training=True)`
       is called on a module whose output tensor has `grad_fn=None`
       post-replacement
   - Recommendation for Sprint 3B:
     - OdinHalo uses `--optimize-kernels` per Phase 0.3 probe
       (200 steps, loss parity at that scale). The same silent-freeze
       may affect it but be masked by iter_norm. **Strong
       recommendation: run 2000-step OdinHalo probe with the current
       config BEFORE committing 48h Sprint 3B.**
   - Whether to explore Triton as a follow-up: based on 5.E outcome

9. **Open questions for future work**

10. **Appendix:** commit hashes, probe logs, ancillary data

**Exit criterion:** doc committed; ~1500-3000 words; self-contained
(readable without context of previous docs).

### Track 3 GATE (exit)

After 5.F is committed:

| Condition | Next action |
|---|---|
| 5.E yielded a working speedup | Propose updating Sprint 3A recipe; validate with 2000-step probe (analogous to Phase V) |
| 5.E yielded no speedup but 5.A/5.B/5.D gave definitive understanding | Close investigation; launch Sprint 3A at baseline |
| 5.A showed OdinHalo is likely affected by same bug (via static reasoning) | Run 2000-step OdinHalo probe before Sprint 3B |
| Unexpected finding surfaces | Report, re-plan |

---

## 6. Post-completion actions (not part of this plan, but queued)

### 6.1 — Sprint 3A launch (after plan completes)

- If Track 2 QKV fusion shipped: re-baseline throughput, update
  `scripts/run_sprint3a.sh` if needed, run 2000-step confirmation,
  then launch full epoch
- If Track 2 skipped/failed: launch Sprint 3A at current baseline
  (31.3K tok/s, ~61h wall) with the recipe from
  `sprint3a-confirm-v1-2000` but WITHOUT `--optimize-kernels`

### 6.2 — Sprint 3B OdinHalo same-bug verification

Run a 2000-step OdinHalo probe with Sprint 3B config (includes
`--optimize-kernels`). Check step-2000 loss against a known-good
OdinHalo training trajectory (e.g. `sprint3-s1_3b-lr2e3-700/step_700.pt`
extrapolated). If the silent-freeze hypothesis is confirmed for
OdinHalo, either:
- Disable `--optimize-kernels` for Sprint 3B (accept ~77h wall)
- Apply the Track 5.E autograd.Function fix (if it worked for
  OdinFlat) and re-verify

### 6.3 — Update `STATUS.md` and `AGENTS.md`

- Add a section: "autokernel: --optimize-kernels is a quality hazard
  under training, do NOT enable without a 2000-step validation"
- Document the Phase I inventory tool as the canonical way to audit
  which patterns hit a given model

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|:---:|:---:|---|
| Track 1 profile reveals unexpected dominator (e.g. dataloader) | Low | Medium | Report to user; re-plan Track 2 scope |
| Track 2 QKV fusion breaks state-dict migration for all prior checkpoints | Low | High | Extensive unit test; roll back hook keeps compat |
| Track 2 fusion provides no speedup (<3%) | Medium | Low | Revert attention.py change; migration hook stays |
| Track 3 5.E autograd.Function has the same overhead as custom_op | Medium | Low | Documented as negative result; close the investigation |
| Track 3 5.E yields a speedup but with subtle numerical drift | Low | High | Phase V-style 2000-step validation before any ship decision |
| Sprint 3A gets blocked for > 2 days | Low | Medium | Can interrupt investigation at any gate and launch 3A |
| OdinHalo Sprint 3B ALSO has the silent-freeze bug (§6.2 discovers) | Medium | High | Follow-up verification task; may require Sprint 3B delay or recipe change |

---

## 8. Quick-start: resuming execution in the next session

When the next session opens this plan:

1. **First action:** verify current state matches this plan's starting
   assumptions. Read `STATUS.md` and check latest commits with
   `git log --oneline -10`. The most recent commit should be `0e4c23b`
   (throughput investigation FINAL) or later. If the tree has diverged,
   resolve before proceeding.

2. **Check Sprint 3A is not running:**
   `bash run_remote.sh "ps aux | grep torchrun | grep -v grep"` —
   should return empty.

3. **Sync both machines:**
   ```bash
   bash sync_remote.sh
   bash sync_remote_b.sh
   ```

4. **Execute Track 1 Task 1.1** (code change to `scripts/train_ddp.py`).
   See §3 for exact specification.

5. **After each task**: commit atomically. Commit messages should
   reference this plan (e.g. "Track 1.1: add --profile-steps flag
   per docs/superpowers/plans/2026-05-10-odinflat-throughput-investigation-plan.md").

6. **After each Track GATE**: report results to the user before
   proceeding. Do NOT silently pivot even if the profile says to.

7. **Final synthesis doc (5.F)** is the primary deliverable.
   `docs/perf/autokernel-deep-analysis.md` should be ~1500-3000 words,
   self-contained.

---

## 9. Glossary

| Term | Definition |
|---|---|
| **V0** | OdinFlat with `F.rms_norm` native + no HIP kernels. Baseline: ~31.7K tok/s, loss 4.73 @ step 200. |
| **V1** | OdinFlat with `F.rms_norm` + `--optimize-kernels --autokernel-exclude rmsnorm` (i.e. silu HIP only). Looks +31% faster but silently freezes `w_gate_up`. Loss +0.65 @ step 2000. |
| **V2** | V1 variant with silu autograd properly registered via `torch.ops.autokernel.silu_gate_mul`. Correct training, ~30.9K tok/s (slightly below baseline). |
| **C3** | Sprint 1.5 Phase C winner: μP + SPECTRA combined recipe. Baseline recipe for Sprint 3A. |
| **autokernel** | In-repo package at `autokernel/` that pattern-matches PyTorch modules and replaces them with HIP-kernel-backed equivalents. Historically buggy for training paths due to autograd plumbing. |
| **Phase 0-V** | Previous investigation phases (2026-05-08 through 2026-05-10). See `docs/perf/odinflat-throughput-final.md`. |
| **Track 1/2/3** | This plan's three tracks: profile, QKV fusion, deep investigation. |

---

## 10. Commit discipline

Expected commits for this plan, in order:

| # | Scope | Branch/tag |
|---|---|---|
| 1 | Track 1.1: `--profile-steps` flag | atomic |
| 2 | Track 1.2-1.3: profile run + analysis script + doc | atomic |
| 3 | Track 2.1-2.4: attention fusion + state-dict hook + tests | atomic |
| 4 | Track 2.5-2.7: autokernel verification + probe + ship decision doc | atomic |
| 5 | Track 3.A: frozen-params diagnostic + writeup | atomic |
| 6 | Track 3.B: autograd mechanism doc | atomic |
| 7 | Track 3.C: V2 overhead decomposition | atomic |
| 8 | Track 3.D: quality mechanism (weight+activation inspection) | atomic |
| 9 | Track 3.E: autograd.Function experiment + probe | atomic |
| 10 | Track 3.F: synthesis doc + STATUS.md update | atomic |

Each commit should be reviewable standalone. If any commit exceeds
500 lines of real code change, split it.

---

## 11. Explicit non-goals

This plan will NOT:

- Modify Sprint 3A or Sprint 3B recipes (except per §6 follow-up)
- Modify the Sprint 1.5 C3 stack (μP, SPECTRA)
- Attempt a Triton RMSNorm rewrite (evaluated and declined per
  `odinflat-option3-native-rmsnorm.md` §"Ceiling assessment")
- Investigate OdinHalo's throughput (separate future work)
- Fix or refactor the HIP kernel sources themselves (only the
  autokernel integration layer)
- Change NorMuon optimizer behavior (flagged as future work if
  Track 1 profile shows it's the bottleneck)
- Attempt bf16 (hardware-blocked on gfx1151 per `AGENTS.md`)

---

**END OF PLAN**

Status at time of writing: committed and ready to execute on next
session "go".
