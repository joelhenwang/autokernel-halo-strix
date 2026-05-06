# Sprint 1 — Foundation Wins: NorMuon+CWD + Architectural Additions

**Date:** 2026-05-06
**Track:** A (of roadmap A → C → B → D)
**Status:** Design approved, awaiting implementation plan
**Related:**
- `docs/research/broad-research-synthesis-2026-05-06.md` (source research, action queue items 1-7)
- `knowledge/training/imu1_recipe_2026.md` (NorMuon+CWD recipe details)
- `knowledge/architectures/small_lm_arch_interventions_2026.md` (architectural additions playbook)

---

## 1. Goal

Deploy a validated "foundation wins" recipe upgrade to OdinFlat that becomes
the new default trainer configuration for all subsequent training runs
(Sprints 2, 3, 4). The upgrade bundles six orthogonal improvements from the
recent small-LM literature, adopted with appropriate adaptations for our
2× Strix Halo DDP + OdinFlat-122M scale.

## 2. Roadmap context

Sprint 1 is the first of four tracks in the A → C → B → D quarterly roadmap:

- **Sprint 1 (A):** Foundation wins (this spec)
- Sprint 2 (C): Evaluation scorecard infrastructure
- Sprint 3 (B): T²-optimal pretraining on dolma-10b (6.9B tokens, ~57× ratio)
- Sprint 4 (D): Post-training pipeline (SFT → ORPO → optional F-GRPO/Scaf-GRPO)

Sprint 1 delivers the recipe that Sprints 3 and 4 depend on: every downstream
training run benefits from the ~4-5% loss improvement baked in here.

**Gate to Sprint 2 (A → C):** Sprint 1 success criteria (§6) must be met
before Sprint 2 begins. If gate fails, re-plan Sprint 1.

## 3. Scope

### In scope

Six changes bundled into a single coordinated upgrade:

| # | Change | Source | Expected gain |
|---|--------|--------|-------------:|
| 1 | Intra-document attention masking | Llama-3, SmolLM3 | +0.05-0.2% loss, stability |
| 2 | LayerNorm scaling (init γ = 1/√(i+1)) | IMU-1 | +0.2-0.5% loss |
| 3 | No weight decay on embeddings + 1D params | OLMo 2, IMU-1 | +0.1-0.3% loss |
| 4 | NorMuon + Cautious Weight Decay optimizer | IMU-1 | −3.85% loss, ~3% throughput cost |
| 5 | Value residual connections (GQA-6 → GQA-13) | IMU-1 | +0.3-0.5% loss |
| 6 | Per-head gated attention | IMU-1 / Qiu et al. 2025 | +0.3-0.5% loss |

Items 1-3 are "free wins" (universal, validated, no/negligible cost). Items 4-6
are substantive (require careful implementation and validation).

### Out of scope (deferred)

- **SPECTRA spectral clipping** — Sprint 1.5
- **μP parametrization** — Sprint 1.5
- **Cross-layer MoDA** — future; see `knowledge/architectures/moda_cross_layer_2026.md`
- **SeeDNorm** — future ablation
- **OdinHalo port** — defer decision until after Sprint 3 (B) shows whether
  the flat variant remains the primary model at T²-optimal training budget

### Model scope

**OdinFlat only.** Our head-to-head at current budgets shows OdinFlat winning
on both throughput (+31%) and loss (final 4.47 vs 4.71 wikitext). Committing to
Sprint 3's T²-optimal dolma training will further favor flat (Chinchilla
advantage at higher token:param ratios is smaller; unique-parameter capacity
dominates). OdinHalo upgrade is a separate future decision.

## 4. Validation strategy

### Approach: direct OdinFlat wikitext re-baseline

We have a known-good reference: OdinFlat trained from scratch on wikitext-103
achieves **loss 4.47 at step 1,869** (DDP, block=256 in old config, or block=512
in new config which is the Sprint 3 production default). Sprint 1 validation
re-runs wikitext with the new recipe and compares against this baseline.

### Why this approach

- Matches our exact production scale (122M, DDP, gfx1151, gloo/TB4)
- Reuses existing dataset + tokenizer on both machines — zero setup cost
- Single run produces a direct A/B against trusted reference
- Signals any scale-attenuation of IMU-1's 430M findings

Alternative approaches (μP-style small-scale probe, multi-scale sweep) were
considered and deferred. Probe-first is only worth the extra compute if we
have low confidence in transfer; the IMU-1 paper's results at adjacent scale
provide enough confidence to go direct.

### Rollout sequencing: 2 grouped runs + LR probe

| Phase | Compute | What |
|-------|--------:|------|
| **Run 1** | ~30 min | Free wins only (items 1-3); NorMuon OFF, value residuals OFF, head gating OFF |
| **LR probe** | ~30 min | 3-config sweep of `lr_2d ∈ {0.015, 0.0235, 0.030}` (200 steps each) with full recipe minus architectural additions |
| **Run 2** | ~1 hr | Full recipe (items 1-6) with probe-winner `lr_2d`; 1 full epoch |

**Rationale:** Run 1 exercises parameter-grouping infrastructure (shared
prerequisite for items 3 and 4) before NorMuon stresses it. Bisection of
regressions becomes localized — if Run 2 fails after Run 1 passes, the
bug is in NorMuon or architectural additions, not the grouping itself.

## 5. Component design

### 5.1 Files touched

| File | Change | Lines (est.) |
|------|--------|-------------:|
| `halo_training/normuon.py` **NEW** | NorMuon optimizer (extends base Muon with neuron-wise norm + Cautious WD) | ~150 |
| `halo_training/optimizer.py` | Add `split_params_2d_vs_1d()` helper; extend `build_optimizer` to support NorMuon path; replace usage of legacy `split_params_for_muon` | +80 |
| `halo_training/cli.py` | Add `--normuon`, `--lr-2d`, `--lr-1d`, feature flags | +20 |
| `halo_training/data.py` | Build `doc_ids` tensor in `PreTokenizedDataset.__getitem__` | +15 |
| `halo_training/trainer.py` | Thread `doc_ids` through to model; param-group optimizer | +30 |
| `scripts/train_ddp.py` | Mirror trainer changes + CLI args | +30 |
| `scripts/launch_ddp.sh` | Add `EXTRA_FLAGS` env var pass-through (1-line flag append on both ranks) | +3 |
| `models/_components.py` | RMSNorm init supports depth-scaled γ (layer index hint) | +10 |
| `models/components/attention.py` | `NoPECodaAttention.forward` adds `v_prev`, `head_gate`, `doc_mask` | +40 |
| `models/components/conv_blocks.py` | `HyPEShortConvBlock` accepts/ignores threading params | +15 |
| `models/odin_flat.py` | Thread `v_prev` 6→13; `doc_mask` into all layers | +30 |

**Total: ~420 LoC across 9 files + 1 new file.**

### 5.2 Parameter grouping

Every parameter lands in exactly one of two groups:

**Group A — 2D weight matrices** (NorMuon, lr=0.0235):
- All `Linear.weight` with `ndim >= 2` that are NOT excluded below
- `FactorizedEmbedding.projection.weight` (the low-rank 256→768 up-projection; it is a standard Linear and can take NorMuon)

**Group B — 1D / embedding / output-head params** (AdamW, lr=0.007, weight_decay=0.0):
- All `nn.Parameter` with `ndim < 2` (biases, RMSNorm γ, logit scales, scalar gates)
- `FactorizedEmbedding.embed.weight` (the raw embedding table — IMU-1 convention)
- Any parameter under `lm_head.*` EXCEPT where it's a tied reference to the above
  (OdinFlat's `FactorizedLMHead` ties to `tok_embeddings.embed`, so no separate lm_head param exists at module-attribute level — `lm_head.weight` is `embed.weight`. The splitter walks `named_parameters()` which returns each tensor once, so no double-counting.)

**Implementation helper:**

```python
def split_params_2d_vs_1d(model):
    """Returns (group_2d, group_1d).

    group_2d: dense weight matrices -> NorMuon
    group_1d: embeddings, output head, biases, LN gains, scalar gates -> AdamW no-WD
    """
    group_2d, group_1d = [], []
    seen_ids = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen_ids:
            continue  # skip tied-weight duplicates
        seen_ids.add(id(p))

        is_1d = (
            p.ndim < 2 or
            name.startswith("tok_embeddings.embed.") or   # raw embedding table only
            name.startswith("lm_head.") or
            "_gate" in name or "scale" in name            # scalar gates, per-head gate, v_res_scale
        )
        (group_1d if is_1d else group_2d).append((name, p))
    return group_2d, group_1d
```

Note: `tok_embeddings.projection.weight` (the 256→768 Linear) is NOT excluded —
it falls through to Group A (NorMuon). This matches IMU-1's treatment of
post-embedding projections as normal 2D matrices.

**Relation to existing `halo_training/muon.py::split_params_for_muon`:**
that helper predates Sprint 1 and is used by the existing Muon code path.
We **replace** it with `split_params_2d_vs_1d`, which has a clearer
contract and handles tied weights explicitly. The old helper's one caller
(the `build_muon_optimizer` in `scripts/train_ddp.py`) is updated to use
the new one; base Muon continues to work as the fallback optimizer.

Unit test asserts every model parameter with `requires_grad=True` lands in
exactly one group (no double-counting, no orphans, no leftovers), validated
against OdinFlat + OdinFlatMini.

### 5.3 NorMuon optimizer

New file `halo_training/normuon.py`. Extends our existing `halo_training/muon.py::Muon`:

```python
class NorMuon(Muon):
    """Muon + neuron-wise normalization + Cautious Weight Decay.
    
    Adds to base Muon:
    1. Post-orthogonalization neuron-wise norm: after msgn(M), normalize rows
       (or columns, depending on shape orientation) to unit norm.
    2. Cautious Weight Decay: apply WD only when sign(gradient) · sign(weight) > 0
       (the cautious-WD condition from Chen et al. 2025).
    
    Newton-Schulz: 7 iterations with Polar Express constants (Amsel et al. 2025).
    """
```

Reference math:
- **Standard Muon update:** `W ← W - η · msgn(M)` where M is momentum, msgn is matrix-sign via Newton-Schulz
- **NorMuon:** `W ← W - η · normalize_rows(msgn(M))` where normalize_rows projects each row to unit ℓ₂ norm
- **Cautious WD:** `WD_effective = WD * (sign(G) · sign(W) > 0)` (elementwise)

Implementation strategy: reuse our existing `halo_training/muon.py` Newton-Schulz
kernel, wrap with normalization + CWD. Interface identical to `torch.optim.AdamW`
for drop-in use.

### 5.4 Intra-document masking data flow

```
┌─────────────────────────────┐
│ datasets/*.bin (memmap)     │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│ PreTokenizedDataset.__getitem__(idx) │
│   tokens  = memmap[idx:idx+T]        │
│   doc_ids = cumsum(tokens == EOS)    │ ← NEW
│   return tokens, doc_ids             │
└──────────────┬───────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ DataLoader yields (input_ids, targets, doc_ids) │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ trainer.py: model(input_ids, doc_ids, …) │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ OdinFlat.forward:                        │
│   doc_mask = (doc_ids[:,:,None] == doc_ids[:,None,:])  # [B, T, T] │
│   ... pass doc_mask to each attention layer …          │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ Attention.forward(…, doc_mask):          │
│   effective_mask = causal_mask & doc_mask │
│   attention_scores.masked_fill_(~effective_mask, -inf) │
└──────────────────────────────────────────┘
```

**EOS token:** `<|endoftext|>` = ID 0 in odin-32k tokenizer (per AGENTS.md).

**Optimization:** mask is shared across layers within a forward pass; build once
per batch, pass as parameter to all attention modules (both `NoPECodaAttention`
and the conv blocks — though conv doesn't attend, it threads the param).

**Memory cost:** `[B, T, T]` int8 tensor. At B=16, T=512 = **4 MB per batch**.
Negligible at our scale.

### 5.5 Value residual design

OdinFlat has GQA attention at layer indices 6 and 13 only; all other layers are
HyPE conv blocks. Value residual threads `v` from position 6 to position 13
through the intervening conv layers as an inert payload.

```
layer 0  (HyPEShortConv)    → output, v_prev=None pass-through
layer 1  (HyPEShortConv)    → output, v_prev=None pass-through
...
layer 6  (NoPEGQA)          → output, produces v_6 (stored in v_prev)
layer 7  (HyPEShortConv)    → output, v_prev=v_6 pass-through
...
layer 12 (HyPEShortConv)    → output, v_prev=v_6 pass-through
layer 13 (NoPEGQA)          → output, consumes v_prev=v_6, produces v_13 (unused further)
final_norm → lm_head → logits
```

**Mechanism inside `NoPECodaAttention.forward(..., v_prev=None)`:**

```python
v = self.wv(x).view(B, T, n_kv_heads, head_dim)
if v_prev is not None:
    v = v + self.v_res_scale * v_prev  # v_res_scale is scalar nn.Parameter, init ≈ 0
# … normal attention from here …
return out, v  # new v for next GQA layer to consume
```

**Init `v_res_scale = 0`** so initial training matches no-residual baseline;
the model learns whether to use v_prev. If the gain turns out to be zero at
our architecture (7 conv layers between GQAs may be too far), the parameter
stays near zero with no harm.

**Gap concern:** IMU-1 validates value residuals on contiguous attention
stacks. Our 7-layer gap is unusual. Mitigated by init-at-zero safety.

### 5.6 Per-head gated attention

```python
class NoPECodaAttention(Attention):
    def __init__(self, dim, n_heads, n_kv_heads, exclusive=False):
        super().__init__(...)
        self.head_gate = nn.Parameter(torch.ones(n_heads))  # init sigmoid(1) ≈ 0.73

    def forward(self, x, freqs_cis=None, v_prev=None, doc_mask=None):
        # ... standard attention producing attn_out shape [B, n_heads, T, head_dim] ...
        if self.head_gate_active:  # toggleable for ablation
            attn_out = attn_out * torch.sigmoid(self.head_gate).view(1, -1, 1, 1)
        # ... rest of forward ...
```

**Init at 1.0** → sigmoid(1.0) ≈ 0.73 (head mostly open, can close during training).

### 5.7 LayerNorm scaling

In each model's `_init_weights`, after Xavier init of 2D weights:

```python
for layer_idx, layer in enumerate(self.layers):
    for norm_module in layer.modules_with_rms_norm():
        norm_module.weight.data *= 1.0 / math.sqrt(layer_idx + 1)
```

Deeper layers start with proportionally smaller γ, preventing runaway
amplification. Affects `pre_norm`, `ffn_norm`, `final_norm` depending on placement.

### 5.8 CLI flags

New flags in `halo_training/cli.py` and `scripts/train_ddp.py`:

```
--normuon                   # (default TRUE after Sprint 1)
--no-normuon                # fallback to fused AdamW
--lr-2d <float>             # NorMuon group LR (default 0.0235)
--lr-1d <float>             # AdamW group LR (default 0.007)
--value-residuals           # (default TRUE after Sprint 1)
--no-value-residuals
--head-gating               # (default TRUE after Sprint 1)
--no-head-gating
--intra-doc-mask            # (default TRUE after Sprint 1)
--no-intra-doc-mask
```

Existing `--lr` flag continues to work: if user specifies `--lr` without
`--lr-2d/--lr-1d`, scale internally by IMU-1 ratio (lr_2d = lr × 3.36, lr_1d = lr).

**Default defaults (before Sprint 1 validates):** all new features OFF.
**After Sprint 1 success:** all new features ON, defaults flipped in a single commit.

## 6. Success criteria (Gate A → C)

Sprint 1 succeeds if ALL of the following are met on Run 2 (full recipe,
wikitext block=512, 1 epoch, DDP):

| Criterion | Target | Measured how |
|-----------|--------|--------------|
| **Final loss** | ≤ **4.30** (−3.8% vs 4.47 baseline) | Last `[step]` log line in `rank0.log` |
| **Stability** | Zero NaN steps AND max grad norm < 10 throughout | StabilityGuard counter + grad_norm log |
| **Throughput cost** | ≤ **7% aggregate tok/s reduction** vs AdamW baseline (~39K → ≥ 36K tok/s) | Steady-state tok/s (steps 200+) |
| **Memory cost** | ≤ **15% per-node increase** (~6.6 GB → ≤ 7.6 GB) | Per-step `mem=...GB` log |
| **Fallback works** | `--no-normuon` still trains to AdamW baseline parity | Manual smoke |

### Failure-response table

| Failure mode | Response |
|---|---|
| Run 1 loss ≥ 4.47 (regression) | Bisect free wins: rerun with `--no-intra-doc-mask`; if still bad, LN scaling is the bug |
| LR probe: all 3 configs diverge | Secondary probe at smaller LRs: `lr_2d ∈ {0.008, 0.015, 0.0235}` |
| Run 2 loss > 4.40 (ship-level regression) | Bisect: run without arch additions (`--no-value-residuals --no-head-gating`). If still bad → NorMuon bug. Fall back to AdamW + arch additions as alt-Sprint-1 |
| Run 2 throughput regresses > 15% | Profile Newton-Schulz in Inductor. Try reducing iterations 7 → 5. If still bad, disable NorMuon pending optimization |
| NorMuon + `compile_zones` incompatibility | Add `@torch.compiler.disable` around Newton-Schulz if needed |
| Value residual explodes / NaN | Clamp `v_res_scale` to `[-1, 1]` with tanh or hard clip |

## 7. Implementation phases

### Phase 1 — Free-wins infrastructure (1-2 days dev)

Implement and unit-test items 1-3:
- `doc_ids` in dataloader, threading through trainer
- Parameter grouping helper
- LayerNorm scaling init
- CLI flags (`--intra-doc-mask`, `--no-intra-doc-mask`, etc.)

**Exit:** All unit tests pass; smoke test (`--smoke`) completes without regression.

### Phase 2 — NorMuon + arch additions (2-3 days dev)

Implement:
- `halo_training/normuon.py` (extends existing Muon)
- Value residual threading in OdinFlat
- Per-head gated attention
- Integration tests: 1-step training with all features ON produces finite gradients

**Exit:** Unit tests + 100-step integration test pass.

### Phase 3 — Run 1 validation (~30 min compute)

**Launcher approach:** extend `scripts/launch_ddp.sh` to accept flags via a new
`EXTRA_FLAGS` env var (1-line addition: `$EXTRA_FLAGS` appended to the
torchrun command on both ranks). Keeps backward-compat with all existing
invocations.

Run 1 command:
```bash
MODEL=models/odin_flat.py CLASS=OdinFlat \
  DATASET=datasets/wikitext-103-odin32k.bin \
  CKPT_DIR=checkpoints/sprint1-run1 \
  EPOCHS=1 \
  EXTRA_FLAGS="--intra-doc-mask --no-normuon --no-value-residuals --no-head-gating" \
  bash scripts/launch_ddp.sh
```

Note: Run 1 includes the LayerNorm scaling init and no-WD-on-embeddings
(parameter grouping with AdamW both groups) — these are init-time / optimizer
construction choices, always active once implemented. Only the three explicit
`--no-*` flags above disable their respective features.

**Gate to Phase 4:** Run 1 final loss ≤ 4.40 AND throughput ≥ 37K tok/s.

### Phase 4 — LR probe (~30 min compute)

Three 200-step runs with full recipe except arch additions:
- `lr_2d ∈ {0.015, 0.0235, 0.030}` × `lr_1d = 0.007` constant

```bash
for LR2D in 0.015 0.0235 0.030; do
  MODEL=models/odin_flat.py CLASS=OdinFlat \
    CKPT_DIR=checkpoints/sprint1-lrprobe-lr2d-${LR2D} \
    EXTRA_FLAGS="--max-steps 200 --normuon --lr-2d $LR2D --lr-1d 0.007 \
                 --intra-doc-mask --no-value-residuals --no-head-gating" \
    bash scripts/launch_ddp.sh
done
```

**Winner selection:** lowest steady-state loss (mean of log entries at
steps 100-200). If all three diverge, secondary probe at smaller LRs.

**Gate to Phase 5:** Probe produces a finite-loss winner.

### Phase 5 — Run 2 (~1 hour compute)

Full recipe, 1 epoch wikitext. Expected 1,869 steps.

```bash
MODEL=models/odin_flat.py CLASS=OdinFlat \
  DATASET=datasets/wikitext-103-odin32k.bin \
  CKPT_DIR=checkpoints/sprint1-run2 \
  EPOCHS=1 \
  EXTRA_FLAGS="--normuon --lr-2d <PROBE_WINNER> --lr-1d 0.007 \
               --intra-doc-mask --value-residuals --head-gating" \
  bash scripts/launch_ddp.sh
```

**Gate to Phase 6 (success):** All 5 success criteria from §6 met.

### Phase 6 — Cleanup & commit (0.5 day)

- Update `STATUS.md` with Sprint 1 results table (target vs actual per metric)
- Update `AGENTS.md` "Training gotchas" section with NorMuon+CWD as new default
- Update `knowledge/training/imu1_recipe_2026.md` — mark `Adoption path` section as implemented
- Update `docs/research/broad-research-synthesis-2026-05-06.md` action-queue:
  mark items 1-7 as done
- Flip default flag values to ON in `halo_training/cli.py` and `scripts/train_ddp.py`
- Commit message: `"Sprint 1: Foundation wins — NorMuon+CWD, value residuals, per-head gating, intra-doc masking, no-WD-on-embeddings"`

## 8. Testing approach

### Unit tests (Phase 1 + Phase 2)

Target: 13 tests total (~1 hour to write).

| Test | Verifies |
|------|----------|
| `test_split_params_2d_vs_1d_complete` | Every parameter lands in exactly one group |
| `test_split_params_2d_vs_1d_embed_in_1d` | Embedding weights go to Group B |
| `test_split_params_2d_vs_1d_ln_in_1d` | LayerNorm γ goes to Group B |
| `test_doc_ids_builds_correctly` | EOS boundaries produce correct cumulative doc IDs |
| `test_doc_ids_mask_shape` | `doc_mask` is `[B, T, T]` bool |
| `test_doc_ids_mask_causal_preserved` | Causal pattern intact inside a single doc |
| `test_doc_ids_mask_blocks_cross_doc` | Cross-doc attention is zero |
| `test_layernorm_scaling_init` | γ values match expected `1/√(layer_idx+1)` |
| `test_normuon_newton_schulz` | Orthogonalization produces valid unit-norm columns |
| `test_normuon_cautious_wd_sign_check` | WD zero when sign(G)·sign(W) < 0 |
| `test_value_residual_init_zero` | `v_res_scale` starts at 0; output matches no-residual at step 0 |
| `test_value_residual_threading` | v_prev from layer 6 reaches layer 13 unchanged |
| `test_head_gate_init_partial` | `torch.sigmoid(head_gate)` ≈ 0.73 at init |

### Integration tests

- **Smoke test (existing):** `python -m halo_training --model models/odin_flat.py --class-name OdinFlat --smoke` passes with all new features enabled
- **100-step training:** single-node, all flags ON, produces finite gradient updates throughout

### Regression tests during validation runs

- Per-50-step NaN/inf counter (existing StabilityGuard)
- Grad norm histogram sampled at steps 50/100/.../1800
- Memory trace logged per step
- Post-run: qualitative sampling via `scripts/ablate_odin_flat_sampling.py`
  on the new checkpoint, confirm output quality not regressed

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| NorMuon unstable at 122M (vs validated 430M) | Medium | High | LR probe (Phase 4) catches this; fallback to safer LRs; ultimate fallback `--no-normuon` |
| NorMuon's Newton-Schulz breaks Inductor compile | Medium | Medium | Phase 2 integration test + targeted `@torch.compiler.disable`; known pattern from RoPE fusion |
| Value residual unstable across 7-layer gap | Low | Medium | Init at zero; Run 2 monitoring; clamp if needed |
| Per-head gating reduces expressivity too much | Low | Low | Init at sigmoid(1.0) = 0.73 (open); model can learn to open fully |
| Intra-doc masking breaks FlashAttention fast path | Medium | Low | Fallback: only enable for multi-doc samples; attention kernel cost offset |
| Parameter grouping misses a param | Low | Medium | Unit test enumerates all params; asserts every one lands somewhere |
| LR probe: all 3 diverge | Low | Medium | Secondary probe at smaller LRs; Sprint 1.5 redesign if persistent |
| Run 2 throughput regresses > 15% | Low | Medium | Profile Newton-Schulz; consider reducing 7 → 5 iterations; fallback to AdamW + architectural-only ship |
| OdinHalo port requested during Sprint 1 | Low (scoped out) | Low | Explicit out-of-scope; defer to post-Sprint-3 decision |
| Stem-crawl run interferes with Sprint 1 validation | High | Low | Wait for stem-crawl completion before Phase 3 (compute on free machines) |

## 10. Timeline

| Phase | Duration | Blocking? |
|-------|----------|:---------:|
| Phase 1 (dev) | 1-2 days | Can parallel with stem-crawl run |
| Phase 2 (dev) | 2-3 days | Can parallel |
| Phase 3 (Run 1 compute) | 30 min | Needs free machines |
| Phase 4 (LR probe compute) | 30 min | Needs free machines |
| Phase 5 (Run 2 compute) | 1 hour | Needs free machines |
| Phase 6 (cleanup + commit) | 0.5 day | No |

**Total elapsed:** **5-6 days** from start to successful gate A → C.

**Compute budget:** **~2 hours** of DDP time (plus unit/integration test cycles).

## 11. Dependencies

### External
- IMU-1 paper (arXiv:2602.02522) — reference recipe
- Existing `halo_training/muon.py` (reuse for NorMuon base)
- Existing `halo_training/optimizer.py::build_optimizer` (extend)
- Existing `halo_training/data.py::PreTokenizedDataset` (extend)
- Existing `scripts/launch_ddp.sh` with post-sweep defaults (block=512, num_workers=12)

### Internal
- Stem-crawl OdinFlat run should complete before Phase 3-5 compute (or coordinate machines)
- Baseline checkpoint `checkpoints/odin-flat-wikitext-ddp/step_1869.pt` used as
  comparison reference (don't overwrite this path)

## 12. Success = what we have at end of Sprint 1

1. **Recipe landed in production code:** all 6 changes implemented, tested,
   default-on, with opt-out flags for debugging
2. **Validated improvement:** Run 2 final loss ≤ 4.30 on wikitext (−3.8% vs
   4.47 baseline)
3. **Throughput impact known and bounded:** ≤ 7% cost measured; documented
4. **Tests passing:** 13 unit tests + 1 integration test + smoke test
5. **Documentation updated:** STATUS.md, AGENTS.md, knowledge base entries
   reflect shipped state
6. **Single clean commit** with the full Sprint 1 body, clear message,
   self-contained rollback by reverting it
7. **Gate A → C cleared** — ready to start Sprint 2 (evaluation scorecard)

## 13. Open questions / future decisions

These do NOT block Sprint 1 but inform future sprints:

1. **μP integration:** When should we port μP parametrization for
   hyperparameter transfer? Sprint 1.5 alongside SPECTRA, or a dedicated
   Sprint?
2. **OdinHalo upgrade:** Does it happen at all? Decision-point is after
   Sprint 3 sees whether OdinFlat + dolma maintains lead.
3. **Cross-layer MoDA:** Attractive but Paper MoDA vs our Parcae MoDA
   interact non-obviously. Park until post-Sprint-4.
4. **SPECTRA spectral clipping:** Likely Sprint 1.5 companion to μP. Need
   to ablate whether it works on top of NorMuon (orthogonal, should compose).

## Appendix A — Reference formulas

### NorMuon update

```
Standard Muon:
  M ← β · M + G                      (momentum accumulation)
  M_orth ← msgn(M)                   (Newton-Schulz orthogonalization)
  W ← W - η · M_orth                 (parameter update)

NorMuon adds:
  M_orth ← normalize_rows(msgn(M))   (neuron-wise unit ℓ₂ norm)
  W ← W - η · M_orth                 (same update form)
```

### Cautious Weight Decay

```
Standard WD:
  W ← W - η · λ · W

Cautious WD:
  cautious_mask = sign(G) · sign(W) > 0       (elementwise)
  W ← W - η · λ · (W ⊙ cautious_mask)         (selective)
```

### Newton-Schulz iteration (Polar Express constants)

```
for k in range(7):
    a, b, c = polar_express_coefs[k]   (paper-specific constants)
    X_next = a·X + b·(X @ X.T @ X) + c·(X @ X.T @ X @ X.T @ X)
    X = X_next
return X   (≈ orthogonal projection msgn(X_0))
```

Reference: IMU-1 Table 6 ablations validate this choice for 430M; Amsel et al.
2025 for Polar Express constants.

### LayerNorm scaling (IMU-1)

```
# At init
for layer_idx, layer in enumerate(self.layers):
    for norm in layer.modules:
        if isinstance(norm, RMSNorm):
            norm.weight.data *= 1.0 / math.sqrt(layer_idx + 1)
```

### Value residual (IMU-1)

```
v_prev = None
for layer_idx, layer in enumerate(self.layers):
    if isinstance(layer, NoPEGQABlock):
        out, v = layer(x, v_prev=v_prev)
        v_prev = v   # carry forward
    else:
        out = layer(x)
    x = out
```

Inside attention:
```
v = self.wv(x).view(B, T, n_kv_heads, head_dim)
if v_prev is not None:
    v = v + self.v_res_scale * v_prev   # v_res_scale is scalar, init 0
return out, v
```
