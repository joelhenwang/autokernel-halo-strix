# Sprint 1 Implementation Plan — Foundation Wins

**Companion to:** `docs/superpowers/specs/2026-05-06-sprint1-foundation-wins-design.md`
**Date:** 2026-05-06
**Sequence:** Dev (Phase 1 → Phase 2) → Compute (Phase 3 → Phase 4 → Phase 5) → Cleanup (Phase 6)

---

## Work breakdown

Each phase's tasks are ordered to respect dependencies. Every task has:
- **Clear exit criterion** (what "done" looks like)
- **Test to write** (or "none" for trivial edits)
- **Rollback step** (what to revert if this task fails)

---

## Phase 1 — Free-wins infrastructure (1-2 days)

Goal: parameter grouping, doc_ids threading, LayerNorm scaling. All unit-tested.

### Task 1.1 — `doc_ids` in `PreTokenizedDataset`

**File:** `halo_training/data.py`
**Change:** `PreTokenizedDataset.__getitem__(idx)` returns `(input_ids, targets, doc_ids)` instead of `(input_ids, targets)`. Compute `doc_ids` as `cumsum(tokens == EOS_ID)` where `EOS_ID=0` (odin-32k convention per AGENTS.md).

**Exit criterion:**
```python
ds = PreTokenizedDataset("datasets/wikitext-103-odin32k.bin", block_size=32)
ids, tgts, doc_ids = ds[0]
assert doc_ids.shape == ids.shape
assert doc_ids.dtype == torch.int32
```

**Test:** `tests/test_data_doc_ids.py::test_doc_ids_cumsum_at_eos`
- Synthetic tokens with EOS at positions {5, 17, 25}
- Assert `doc_ids == [0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3, ...]`

**Rollback:** Revert `__getitem__` signature; callers using the old 2-tuple continue to work via backward-compat branch in the trainer.

### Task 1.2 — Backward-compat in `build_dataloader`

**File:** `halo_training/data.py`
**Change:** Ensure `build_dataloader` handles both 2-tuple and 3-tuple returns. If using `PreTokenizedDataset`, return 3-tuples; if using legacy `BabyLMDataset`, synthesize `doc_ids = zeros_like(tokens)` (treats whole batch as single doc).

**Exit criterion:** Both `BabyLMDataset` (smoke) and `PreTokenizedDataset` (DDP) paths produce tensors the trainer can consume.

**Test:** `tests/test_data_doc_ids.py::test_dataloader_returns_3tuple`
- Iterate one batch from each dataset type
- Assert yielded tuple length == 3

**Rollback:** Remove 3-tuple path; all datasets return 2-tuples; `doc_ids` defaults to `None` in model forward.

### Task 1.3 — Parameter grouping helper

**File:** `halo_training/optimizer.py`
**Change:** Add `split_params_2d_vs_1d(model)` per spec §5.2. Includes id-based de-dup for tied weights.

**Exit criterion:**
```python
group_2d, group_1d = split_params_2d_vs_1d(OdinFlat())
names_2d = {n for n, _ in group_2d}
names_1d = {n for n, _ in group_1d}
assert names_2d.isdisjoint(names_1d)
total_params = sum(p.numel() for _, p in group_2d) + sum(p.numel() for _, p in group_1d)
assert total_params == sum(p.numel() for p in OdinFlat().parameters() if p.requires_grad)
```

**Test:** `tests/test_param_grouping.py::test_complete_partition_odinflat`
- Assert every trainable param lands in exactly one group
- Assert `tok_embeddings.embed.weight` in group_1d
- Assert `tok_embeddings.projection.weight` in group_2d (if model uses FactorizedEmbedding with projection)
- Assert all biases and norms in group_1d

**Test:** `tests/test_param_grouping.py::test_tied_weights_counted_once`
- FactorizedLMHead ties to tok_embeddings.embed: assert this shared tensor appears only once in the splitter's output

**Rollback:** Leave both old (`split_params_for_muon`) and new helpers; optimizer construction uses a boolean switch.

### Task 1.4 — Update `build_optimizer` to consume new helper

**File:** `halo_training/optimizer.py`
**Change:** Refactor `build_optimizer` so both AdamW and NorMuon paths use `split_params_2d_vs_1d`:

```python
def build_optimizer(model, base_lr, optimizer="adamw", lr_2d=None, lr_1d=None, **kwargs):
    group_2d, group_1d = split_params_2d_vs_1d(model)

    if optimizer == "adamw":
        # No-WD on embeds + 1D params
        return torch.optim.AdamW([
            {"params": [p for _, p in group_2d], "weight_decay": 0.1},
            {"params": [p for _, p in group_1d], "weight_decay": 0.0},
        ], lr=base_lr, betas=(0.9, 0.95), fused=True)

    elif optimizer == "normuon":
        from halo_training.normuon import NorMuon
        return NorMuon(
            muon_params=[{"params": [p for _, p in group_2d], "lr": lr_2d or 0.0235}],
            adamw_params=[{"params": [p for _, p in group_1d], "lr": lr_1d or 0.007, "weight_decay": 0.0}],
            lr=lr_2d or 0.0235,
            weight_decay=0.1,
            cautious_wd=True,
        )
```

**Exit criterion:**
- Smoke test passes with `optimizer="adamw"` (regression baseline)
- Smoke test passes with `optimizer="normuon"` (requires Task 2.1 first for full test; Phase 1 verifies just the dispatch)

**Test:** `tests/test_optimizer_build.py::test_adamw_splits_wd`
- Verify `optimizer.param_groups[0]["weight_decay"] == 0.1` (group_2d)
- Verify `optimizer.param_groups[1]["weight_decay"] == 0.0` (group_1d)

**Rollback:** Keep legacy single-group AdamW as fallback; new path behind `optimizer="normuon"` flag only.

### Task 1.5 — LayerNorm scaling init

**File:** `models/_components.py` + `models/odin_flat.py::_init_weights`
**Change:** After Xavier init of 2D weights, scale each layer's RMSNorm γ by `1/√(layer_idx+1)`.

In `models/odin_flat.py::_init_weights`, after the existing init:

```python
# LayerNorm scaling (Sprint 1 — IMU-1 recipe)
import math
for layer_idx, layer in enumerate(self.layers):
    for submodule in layer.modules():
        if isinstance(submodule, RMSNorm):
            submodule.weight.data.mul_(1.0 / math.sqrt(layer_idx + 1))
```

**Exit criterion:** After instantiating `OdinFlat()`, `self.layers[0].pre_norm.weight` should be ≈ 1.0 (1/√1), `self.layers[13].pre_norm.weight` should be ≈ 0.267 (1/√14).

**Test:** `tests/test_ln_scaling.py::test_depth_scaled_gamma`
- Instantiate OdinFlat
- Assert `self.layers[i].pre_norm.weight[0] ≈ 1/sqrt(i+1)` for all i

**Rollback:** Remove the post-init loop; RMSNorm γ stays at its constructor default (1.0).

### Task 1.6 — Intra-document attention mask plumbing

**File:** `models/components/attention.py`
**Change:** `NoPECodaAttention.forward` accepts optional `doc_mask: [B, T, T]`. Inside:

```python
def forward(self, x, freqs_cis=None, depth_kvs=None, doc_mask=None, v_prev=None):
    # ... existing setup ...
    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale    # [B, H, T, T]

    # Apply causal mask (existing)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    attn_mask = causal
    if doc_mask is not None:
        attn_mask = attn_mask & doc_mask                          # intersect

    scores = scores.masked_fill(~attn_mask[None, None, :, :], float("-inf"))
    # ... rest unchanged ...
```

**File:** `models/odin_flat.py::forward`
**Change:** Build `doc_mask` from `doc_ids` at the top, pass to every attention call:

```python
def forward(self, input_ids, targets=None, doc_ids=None):
    B, T = input_ids.shape
    # ... existing embeddings + freqs_cis setup ...

    doc_mask = None
    if doc_ids is not None:
        doc_mask = (doc_ids[:, :, None] == doc_ids[:, None, :])   # [B, T, T]

    for layer, is_gqa in zip(self.layers, self._is_gqa):
        if is_gqa:
            h = layer(h, doc_mask=doc_mask)                        # NEW arg
        else:
            h = layer(h, freqs_cis)                                # conv ignores doc_mask
```

`HyPEShortConvBlock.forward` does NOT receive `doc_mask` — convs are local and short; cross-doc contamination via 3-token conv kernel is negligible at block=512.

**Exit criterion:** 1-step training with `doc_mask` enabled completes without error; no-op when `doc_mask=None`.

**Test:** `tests/test_doc_mask.py::test_mask_blocks_cross_doc`
- Batch of 2 samples, each with one EOS boundary
- Forward pass with `doc_ids` set
- Assert that attention weights between tokens with different `doc_ids` are zero (verify via register_forward_hook on `NoPECodaAttention`)

**Test:** `tests/test_doc_mask.py::test_mask_preserves_causal`
- With `doc_mask` enabled and all tokens in one doc
- Assert behavior matches no-mask baseline (causal-only)

**Rollback:** Remove `doc_mask` arg from attention forward; mask construction in model stays but becomes no-op.

### Task 1.7 — Thread `doc_ids` through trainer

**File:** `halo_training/trainer.py`
**Change:** Dataloader yields 3-tuple; pass `doc_ids` to `model(input_ids, targets=targets, doc_ids=doc_ids)`.

**File:** `scripts/train_ddp.py`
**Change:** Same, plus handle the DDP `no_sync()` context (just pass `doc_ids` through, no state).

**Exit criterion:** Full DDP smoke test runs (not a full epoch — just `--max-steps 10`) with new flags enabled.

**Test:** None (integration covered by Phase 3 Run 1).

**Rollback:** Pass `doc_ids=None` explicitly if new dataloader returns 3-tuple; model forward handles None gracefully.

### Task 1.8 — CLI flag plumbing (Phase 1 subset)

**File:** `halo_training/cli.py` + `scripts/train_ddp.py`
**Change:** Add flags:
- `--intra-doc-mask / --no-intra-doc-mask` (default: OFF for Phase 1; flip to ON in Phase 6)
- `--normuon / --no-normuon` (default: OFF; unused in Phase 1, but wire it now)
- `--lr-2d`, `--lr-1d` (default 0.0235, 0.007; unused in Phase 1)
- `--value-residuals / --no-value-residuals`
- `--head-gating / --no-head-gating`

Flag dispatch in trainer:
```python
if args.intra_doc_mask:
    # passed through from dataloader → model; no-op if this flag is off
    pass
if args.normuon:
    optimizer = build_optimizer(model, base_lr=args.lr, optimizer="normuon",
                                lr_2d=args.lr_2d, lr_1d=args.lr_1d)
```

**Exit criterion:** `python -m halo_training --help` lists all new flags.

**Test:** None (manual).

**Rollback:** Remove flags; trainer uses defaults.

### Task 1.9 — `launch_ddp.sh` `EXTRA_FLAGS` support — ALREADY COMPLETE

**Status:** Landed in Sprint 2 commit 413b4a6. `scripts/launch_ddp.sh` already
reads `EXTRA_FLAGS` env var and appends to both torchrun invocations. No work
needed for Sprint 1.

**Verification:** `EXTRA_FLAGS="--max-steps 10" bash scripts/launch_ddp.sh`
already runs 10 steps and exits cleanly (verified during Sprint 2 Phase 5
smoke test).

### Phase 1 exit gate

All 9 tasks complete (Task 1.9 inherited from Sprint 2). All unit tests pass.
Smoke test runs to completion with all new flags either absent (matches
baseline) or set to `--no-*` (matches baseline).

**Test file convention:** All new tests live in `scripts/test_*.py` to match
the existing repo pattern. The plan originally referenced `tests/*.py`; that
directory does not exist and we're not creating it.

---

## Phase 2 — NorMuon + arch additions (2-3 days)

### Task 2.1 — NorMuon optimizer

**File:** `halo_training/normuon.py` (NEW)
**Change:** Implement NorMuon class. Structure:

```python
import torch
from torch.optim.optimizer import Optimizer
from halo_training.muon import _newton_schulz_kernel  # reuse existing Newton-Schulz

class NorMuon(Optimizer):
    """Muon + neuron-wise norm + Cautious Weight Decay (IMU-1 recipe)."""

    def __init__(self, muon_params, adamw_params, lr, weight_decay=0.1,
                 betas=(0.9, 0.95), cautious_wd=True, ns_iterations=7):
        # ... standard Optimizer init with two param groups ...

    @torch.no_grad()
    def step(self, closure=None):
        # For muon params:
        #   1. Update momentum M = beta*M + G
        #   2. Orthogonalize: M_orth = newton_schulz(M, iterations=7)
        #   3. Neuron-wise normalize: M_orth = M_orth / M_orth.norm(dim=1, keepdim=True)
        #   4. Cautious WD: wd_mask = (sign(G) * sign(W) > 0)
        #   5. Update: W = W - lr * M_orth - lr * wd * W * wd_mask
        # For adamw params: standard fused AdamW update
```

Key implementation notes:
- **Newton-Schulz kernel:** reuse existing `halo_training/muon.py` orthogonalization function; it already has the 7-step Polar Express constants
- **Neuron-wise norm:** apply row-wise ℓ₂ normalize (or column-wise depending on shape orientation — the paper uses row-wise for the `in_features` dim)
- **Cautious WD:** elementwise mask `sign(G_t) * sign(W_t) > 0` as a `torch.where` in the update

**Exit criterion:** 1-step update on a toy 2-layer MLP produces finite, non-NaN weight changes.

**Test:** `tests/test_normuon.py::test_single_step_finite`
- 2-layer MLP, random data, 1 forward + 1 backward + 1 optimizer step
- Assert all weights remain finite

**Test:** `tests/test_normuon.py::test_newton_schulz_produces_orthogonal`
- Random matrix M
- Apply Newton-Schulz
- Assert `U U^T ≈ I` (within tolerance)

**Test:** `tests/test_normuon.py::test_cautious_wd_zeroes_on_sign_mismatch`
- Create synthetic gradient G and weight W with matched/mismatched signs
- Assert update applies WD only where `sign(G) * sign(W) > 0`

**Rollback:** Delete file; revert `build_optimizer` to always return AdamW.

### Task 2.2 — Wire NorMuon into `build_optimizer`

**File:** `halo_training/optimizer.py`
**Change:** The NorMuon path sketched in Task 1.4 becomes real; add the actual import and instantiation.

**Exit criterion:** `build_optimizer(model, optimizer="normuon")` returns a valid optimizer that takes a step without error.

**Test:** `tests/test_optimizer_build.py::test_normuon_step`
- Build optimizer for an OdinFlatMini
- Run 5 training steps on random data
- Assert loss is finite and decreases

**Rollback:** Same as Task 1.4.

### Task 2.3 — Value residual in `NoPECodaAttention`

**File:** `models/components/attention.py`
**Change:** Add `v_res_scale` parameter; accept `v_prev` kwarg; return `v` alongside output.

```python
class NoPECodaAttention(Attention):
    def __init__(self, dim, n_heads, n_kv_heads, exclusive=False):
        super().__init__(dim, n_heads, n_kv_heads, qk_norm=True)
        self.exclusive = exclusive
        # Value residual (Sprint 1)
        self.v_res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, freqs_cis=None, depth_kvs=None, doc_mask=None, v_prev=None):
        # ... existing Q,K,V projection ...
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        v_raw = v  # save before residual
        if v_prev is not None:
            v = v + self.v_res_scale * v_prev

        # ... rest of attention forward with v ...

        # Return (out, v_raw) — next GQA layer consumes v_raw as its v_prev
        return out, v_raw
```

**Exit criterion:** Forward pass without `v_prev` behaves identically to pre-Sprint-1. With `v_prev` and `v_res_scale=0`, behavior still matches (init safety).

**Test:** `tests/test_value_residual.py::test_init_zero_matches_baseline`
- Random input
- Forward with `v_prev=random_tensor` but `v_res_scale=0`
- Assert output == forward without `v_prev`

**Test:** `tests/test_value_residual.py::test_scale_learns_nonzero`
- After 10 training steps, assert `v_res_scale.abs() > 0` (grad flows)

**Rollback:** Remove `v_res_scale` parameter; `v_prev` ignored; return signature stays 1-tuple.

### Task 2.4 — Per-head gating

**File:** `models/components/attention.py`
**Change:** Add `head_gate` parameter; apply in forward.

```python
class NoPECodaAttention(Attention):
    def __init__(self, ..., n_heads):
        super().__init__(...)
        self.head_gate = nn.Parameter(torch.ones(n_heads))
        self.v_res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, ..., head_gate_active=True):
        # ... attention computation producing attn_out [B, H, T, D] ...
        if head_gate_active:
            attn_out = attn_out * torch.sigmoid(self.head_gate).view(1, -1, 1, 1)
        # ... project out via wo ...
```

The `head_gate_active` flag in forward lets the `--no-head-gating` CLI override disable it at runtime.

**Exit criterion:** At init, `torch.sigmoid(head_gate)` ≈ 0.73 per head.

**Test:** `tests/test_head_gate.py::test_init_open`
- Instantiate attention
- Assert `torch.sigmoid(self.head_gate).item() ≈ 0.73` for each head

**Test:** `tests/test_head_gate.py::test_grad_flows`
- After 10 steps, assert `head_gate.grad` non-zero

**Rollback:** Remove `head_gate` parameter; skip the multiplication.

### Task 2.5 — Thread `v_prev` in OdinFlat.forward

**File:** `models/odin_flat.py`
**Change:** Track `v_prev` between GQA layers.

```python
def forward(self, input_ids, targets=None, doc_ids=None):
    # ... existing embedding + freqs_cis + doc_mask setup ...

    v_prev = None
    for layer, is_gqa in zip(self.layers, self._is_gqa):
        if is_gqa:
            h, v = layer(h, doc_mask=doc_mask, v_prev=v_prev, head_gate_active=self.head_gate_active)
            v_prev = v
        else:
            h = layer(h, freqs_cis)

    # ... final_norm, lm_head, softcap ...
```

`head_gate_active` is set from the runtime flag (propagated from CLI). Default `True`; set to `False` by `--no-head-gating`.

**Exit criterion:** Integration test: 1-step forward with all features enabled produces valid output tensor shape `[B, T, vocab_size]`.

**Test:** `tests/test_odin_flat_integration.py::test_forward_all_features_on`
- All features enabled
- Forward pass on random input
- Assert output shape, no NaN, finite values

**Test:** `tests/test_odin_flat_integration.py::test_v_prev_threading`
- Register forward hooks on layers 6 and 13
- After forward, assert layer 13's attention received `v_prev == layer 6's v`

**Rollback:** Set `v_prev=None` always; attention ignores it.

### Task 2.6 — Backward-compat check

**File:** `models/odin_halo.py` (unchanged, but verify forward signature still works)
**Change:** None directly; just verify smoke test for OdinHalo still passes (it uses `forward(input_ids, targets)` without new kwargs — should be unaffected since changes are in OdinFlat only).

**Exit criterion:** `python -m halo_training --model models/odin_halo.py --class-name OdinHalo --smoke` passes.

**Test:** None (existing smoke test covers this).

**Rollback:** N/A.

### Phase 2 exit gate

All 6 tasks complete. All unit tests (13 total from Phase 1 + 2) pass. Smoke test with all flags ON passes.

---

## Phase 3 — Run 1 validation (~30 min compute)

### Task 3.1 — Launch Run 1 on free machines

**Prerequisite:** Stem-crawl OdinFlat run has completed OR has been stopped. Verify both machines are idle via `run_remote.sh "ps aux | grep train_ddp | grep -v grep | wc -l"` returns 0.

**Command:**
```bash
bash sync_remote.sh && bash sync_remote_b.sh
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
  MODEL=models/odin_flat.py CLASS=OdinFlat \
  DATASET=datasets/wikitext-103-odin32k.bin \
  CKPT_DIR=checkpoints/sprint1-run1 \
  EPOCHS=1 \
  EXTRA_FLAGS='--intra-doc-mask --no-normuon --no-value-residuals --no-head-gating' \
  bash scripts/launch_ddp.sh"
```

**Exit criterion:** Run completes 1 epoch (~1,869 steps); rank0.log has `Done:` line.

**Gate (A → proceed to Phase 4):**
- Final loss ≤ 4.40
- Zero NaN steps in rank0.log
- Steady-state throughput ≥ 37K tok/s (from steps 200+ log_interval entries)

**Gate failure response:**
- If loss regression: bisect by disabling `--intra-doc-mask`, then LayerNorm scaling (by reverting init), then no-WD-on-embeddings (by using old single-group AdamW)
- If NaN: inspect the grad_norm trajectory; typically intra-doc mask bug (zero tokens visible to early tokens in some samples)

**Rollback:** Delete `checkpoints/sprint1-run1/`; no other state change.

---

## Phase 4 — LR probe (~30 min compute)

### Task 4.1 — Run 3-config LR sweep

**Command:**
```bash
for LR2D in 0.015 0.0235 0.030; do
  bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
    MODEL=models/odin_flat.py CLASS=OdinFlat \
    DATASET=datasets/wikitext-103-odin32k.bin \
    CKPT_DIR=checkpoints/sprint1-lrprobe-lr2d-${LR2D} \
    EXTRA_FLAGS=\"--max-steps 200 --normuon --lr-2d $LR2D --lr-1d 0.007 \
                 --intra-doc-mask --no-value-residuals --no-head-gating\" \
    bash scripts/launch_ddp.sh"
done
```

Each run is ~7 min (200 steps at ~40K tok/s); total ~25 min for three runs plus relaunch overhead.

**Exit criterion:** Three completed runs with final logs and saved checkpoints.

**Winner selection:**
```python
# For each run, parse rank0.log
# Compute mean loss over steps 100-200 (post-warmup, pre-completion)
# Winner = config with lowest mean loss
```

**Gate (proceed to Phase 5):** Winner's mean-loss < 10 (finite). If all three diverge (loss > 10 or NaN), secondary probe at `lr_2d ∈ {0.008, 0.015, 0.0235}`.

**Rollback:** Delete probe checkpoints; use baseline AdamW for Run 2 via `--no-normuon`.

---

## Phase 5 — Run 2 validation (~1 hour compute)

### Task 5.1 — Launch Run 2 with full recipe (with --auto-eval)

**Command:**
```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
  MODEL=models/odin_flat.py CLASS=OdinFlat \
  DATASET=datasets/wikitext-103-odin32k.bin \
  CKPT_DIR=checkpoints/sprint1-run2 \
  EPOCHS=1 \
  EXTRA_FLAGS='--normuon --lr-2d <PROBE_WINNER> --lr-1d 0.007 \
              --intra-doc-mask --value-residuals --head-gating --auto-eval' \
  bash scripts/launch_ddp.sh"
```

`--auto-eval` (Sprint 2 feature) fires the scorecard in a detached subprocess
after every checkpoint save. The final step_1869 checkpoint auto-produces a
scorecard comparable to `docs/perf/eval-scorecards/odin-flat-wikitext-ddp-step-1869.json`.

**Exit criterion:** Run completes 1 epoch.

**Gate (A → B, proceed to Phase 6):**
- Final loss ≤ 4.30 (from rank0.log)
- Run 2 wikitext_val BPB ≤ 1.73 (from auto-eval scorecard, ~3.8% below 1.80 baseline)
- Zero NaN, max grad norm < 10
- Throughput ≥ 37K tok/s
- Memory ≤ 11.9 GB/node (15% above current 10.3 GB)
- `--no-normuon` fallback still trains to AdamW baseline parity
- `docs/perf/eval-scorecards/sprint1-run2-step-1869.json` exists and parseable

**Gate failure response per §6 table in design doc.**

### Task 5.2 — Qualitative sampling check

**Command:**
```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
  python3 scripts/ablate_odin_flat_sampling.py \
    --checkpoint checkpoints/sprint1-run2/step_1869.pt"
```

**Exit criterion:** Winning sampling config produces coherent-enough output (no regression vs existing OdinFlat wikitext step_1869 samples). Qualitative judgment.

---

## Phase 6 — Cleanup & commit (0.5 day)

### Task 6.1 — Flip default flag values

**File:** `halo_training/cli.py` and `scripts/train_ddp.py`
**Change:** Flip defaults to ON:
- `--normuon` (True by default)
- `--intra-doc-mask` (True by default)
- `--value-residuals` (True by default)
- `--head-gating` (True by default)

Keep the `--no-*` variants as opt-outs.

### Task 6.2 — Update STATUS.md

Add a new section documenting Sprint 1 results:
```
### Sprint 1: Foundation wins (2026-05-06 → 2026-05-XX)

Spec: docs/superpowers/specs/2026-05-06-sprint1-foundation-wins-design.md
Plan: docs/superpowers/plans/2026-05-06-sprint1-foundation-wins-plan.md

Final results (wikitext-103 1-epoch, OdinFlat, new recipe):
  Loss:         X.XX   (vs 4.47 baseline = -X.X%)
  Throughput:   X,XXX tok/s (vs 39K baseline = -X.X% cost)
  Memory:       X.X GB/node
  LR winner:    lr_2d = X.XXX
  ...
```

### Task 6.3 — Update AGENTS.md

Add to the "Training gotchas" section:
```
- **NorMuon + Cautious WD** is now the default optimizer (Sprint 1, May 2026).
  Override with `--no-normuon` for legacy AdamW path.
- **Parameter grouping** splits 2D matrices (NorMuon, lr_2d=0.0235) from
  1D/embeddings/lm_head (AdamW, lr_1d=0.007, no WD).
- **Intra-document attention masking** default-on; prevents cross-doc
  attention in packed sequences.
```

### Task 6.4 — Update knowledge base entries

- `knowledge/training/imu1_recipe_2026.md`: mark "Adoption path for Odin" section as ✓ implemented; link to Sprint 1 spec
- `knowledge/architectures/small_lm_arch_interventions_2026.md`: mark items 1-6 as ✓ shipped; link to Sprint 1 spec
- `docs/research/broad-research-synthesis-2026-05-06.md` action queue: mark items 1-7 as done

### Task 6.5 — Single cleanup commit

```
Sprint 1: Foundation wins — NorMuon+CWD, value residuals, per-head gating,
intra-doc masking, no-WD-on-embeddings

Delivers the IMU-1 recipe (arXiv:2602.02522) adapted for OdinFlat:
  - NorMuon + Cautious Weight Decay (replaces fused AdamW by default)
  - Parameter grouping: 2D matrices -> NorMuon (lr=0.0235);
    1D/embeddings/lm_head -> AdamW no-WD (lr=0.007)
  - LayerNorm scaling: init gamma = 1/sqrt(layer_idx+1)
  - Value residual connections: v from GQA-6 carried to GQA-13
    (init scale=0, model learns to use)
  - Per-head gated attention: sigmoid gate init at 1.0 (0.73 open)
  - Intra-document attention masking: prevents cross-doc attention
    in packed sequences

Validation on wikitext-103 (1 epoch, DDP):
  Run 1 (free wins only):  loss X.XX (vs 4.47 baseline)
  LR probe winner:         lr_2d = X.XXX
  Run 2 (full recipe):     loss X.XX (-X.X% vs baseline)
  Throughput cost:         X.X% (vs AdamW)
  Memory cost:             X.X% (vs AdamW)

13 unit tests added. Default flags flipped to new features ON.
Fallback available via --no-normuon / --no-value-residuals / etc.

Spec: docs/superpowers/specs/2026-05-06-sprint1-foundation-wins-design.md
Plan: docs/superpowers/plans/2026-05-06-sprint1-foundation-wins-plan.md
```

---

## Summary timeline

| Day | Phase | Activity |
|:---:|-------|----------|
| 1 | Phase 1 | Tasks 1.1-1.4 (data, params, optimizer dispatch) |
| 2 | Phase 1 | Tasks 1.5-1.9 (LN scaling, mask, flags, launcher) + unit tests |
| 3 | Phase 2 | Tasks 2.1-2.2 (NorMuon + wiring) |
| 4 | Phase 2 | Tasks 2.3-2.6 (arch additions + integration tests) |
| 5 | Phase 3-4 | Run 1 compute (30 min) + LR probe (30 min) + diagnosis |
| 5-6 | Phase 5 | Run 2 compute (1 hr) + qualitative check |
| 6 | Phase 6 | Docs, commit, tag |

**Total: 5-6 elapsed days, ~2 hours compute.**

## Dependencies external to Sprint 1

- **Stem-crawl OdinFlat run completes** (or is killed) before Phase 3-5 compute
- Baseline checkpoint `checkpoints/odin-flat-wikitext-ddp/step_1869.pt` remains intact
- Both machines idle during Phase 3-5
- No concurrent sweep work during Phase 3-5

## Handoff after Sprint 1

Sprint 1 completion unlocks:
- **Sprint 3** (Track B): T²-optimal dolma training uses the Sprint 1 recipe
  as its default, with `--auto-eval` enabled for per-checkpoint visibility
  during the ~50-hour run
- **Sprint 1.5** (conditional): SPECTRA + μP transfer study consumes the
  scorecard to measure probe vs main-model parity

Sprint 2 (evaluation scorecard) was completed BEFORE Sprint 1 per the
2026-05-06 re-sequencing decision; Sprint 1 validates its recipe against
the Sprint 2 scorecard baselines.

The Sprint 1 design doc and this implementation plan become the reference
for anyone asking "what's in our training stack?" — they capture the full
rationale behind each default.
