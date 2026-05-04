# VIDAR-HALO Ablation Plan

**Date:** 2026-05-04
**Status:** PLANNED
**Purpose:** Systematic ablation of parameter-golf and paper-derived techniques on VIDAR-HALO

---

## Why Vidar, Not a Fresh Design

Before planning ablations, the question: should we start fresh?

### Evidence for building on Vidar

| Decision | Evidence | Alternatives Tested |
|----------|----------|-------------------|
| d=768 | 3.2 vs 6.1 loss (GRIFFIN sweep, 10 configs) | d=512, d=640 implicit |
| Loop ×2 | BPB 2.10 (58M) beat 2.75 (118M flat) | 9 architectures at 170M |
| Conv + GQA hybrid | LFM2.5 HW search: conv beats SSMs on Ryzen AI Max+ | S4, Mamba, Mamba2, CfC, linear attn all tested by Liquid AI |
| 3:1 conv:GQA ratio | LFM2 uses 10:6, our ablations show 4:2 optimal for 6L | TyrHaloLight, CHIMERA, BALDR |
| Muon optimizer | Validated at 6K H200s (Poolside), DSV4, our training | AdamW (15% more steps needed) |
| WSD schedule | GPT-X2-125M: beats cosine at small scale | Cosine (prior default) |
| EMA 0.999 | TRM: +7.5% generalization, zero cost | No EMA (7.5% worse) |

### What the new findings add

The parameter-golf techniques and paper batch findings are **training refinements and architectural micro-improvements**, not paradigm shifts:

- Polar-Express NS: better optimizer coefficients (not a new optimizer)
- MIN_LR floor: better schedule (not a new schedule)
- Delayed recurrence: better loop activation timing (not a new loop mechanism)
- Parallel residuals: lighter version of HC streams we already tested
- Logit softcap: Gemma2 stability trick
- Skip connections: U-Net pattern between iterations

None of these require a different backbone. All are testable as incremental changes.

### What would justify a fresh design

- **SSMs overtaking convs on our hardware** — they haven't (LFM2.5 + parameter-golf both confirm conv is faster on edge)
- **Looping proven inferior to flat at equal compute** — opposite is true (TyrHaloLight 2.10 vs BALDR 2.75)
- **d=768 shown to be suboptimal** — no evidence, it's the dominant quality factor
- **A radically different architecture family** (e.g., diffusion LM, JEPA-only) — none have beaten autoregressive transformers at small scale

**Verdict: Vidar is the correct foundation. Ablate new techniques on top.**

---

## Ablation Protocol

### Two-Tier Design

Fast screening at reduced scale, then validation at full scale. Proven pattern in this project — BabyLM screening correctly ranked 9 architectures in the hypothesis buildout.

**Tier S (Screening):** `VidarHaloAblation` d=384, BabyLM, 1 epoch, ~12 min/run
**Tier V (Validation):** `VidarHaloGPT2` d=768, stem-crawl-solo, 1 epoch, ~8 hr/run

### Tier S: Screening Setup

| Parameter | Value |
|-----------|-------|
| Model | `VidarHaloAblation` (d=384, ~18M unique, ~26M effective) |
| Dataset | BabyLM (16.5M tokens, pre-tokenized .bin) |
| Seq len | 512 |
| Duration | 1 epoch (~504 steps) |
| Mode | Eager + autokernel (`--optimize-kernels`, no `--compile`) |
| Hardware | Machine A or B (Strix Halo, gfx1151) |
| Throughput | ~22K tok/s (d=384, seq=512, eager+AK) |
| Wall time | **~12 min per run** |
| Parallel | 2 machines → 2 configs per 12-min slot |
| LR | 0.003 (higher for smaller model, cf. GPT-X2 used 1.5e-3 at 125M) |
| Warmup | 50 steps |
| Seeds | 1 seed for screening (seed 42); 3-seed on finalists |
| Metrics | Final CE loss, tok/s, peak memory, NaN count |
| Checkpoints | None (`--checkpoint-interval 999999`) |

**VidarHaloAblation spec (to be created in `models/vidar_halo.py`):**

```python
class VidarHaloAblation(VidarHaloBase):
    """Ablation variant: d=384, 4 layers, ~18M params.
    d_conv=320 (>256 threshold) enables autokernel.
    Targets ~12 min for 1 epoch on BabyLM at seq=512."""
    def __init__(self, **kw):
        kw.setdefault("d_model", 384)
        kw.setdefault("embed_rank", 192)
        kw.setdefault("n_heads", 6)
        kw.setdefault("n_kv_heads", 2)
        kw.setdefault("ffn_inner", 1408)
        kw.setdefault("d_conv", 320)    # >256 — autokernel safe
        kw.setdefault("conv_kernel", 3)
        super().__init__(**kw)
```

**Why d=384, not d=128 or d=768:**

| Concern | d=128 (Mini) | d=384 (Ablation) | d=768 (Full) |
|---------|-------------|------------------|-------------|
| Capacity | Toy — can't generalize | Real generalization | Production |
| Muon NS behavior | Trivial matrices — NS overkill | Meaningful orthogonalization | Full-scale |
| Autokernel | Broken (d<256) | Works (d_conv=320>256) | Works |
| Eager 1ep BabyLM | ~5 min | **~12 min** | ~80 min |
| Technique sensitivity | Misses optimizer effects | Detects all categories | Detects all |

**Why BabyLM, not gpt-training-small or stem-crawl:**

- BabyLM (16.5M tokens) = 1 full epoch in 12 min at d=384. Full WSD cycle completes.
- gpt-training-small (292M) = only 6% coverage in 12 min. Loss still in early descent.
- stem-crawl (544M) = 3% coverage. Useless for comparison.
- BabyLM screening correctly ranked 9 architectures in prior hypothesis buildout. **Relative deltas transfer** even though absolute loss doesn't match web data.

**Screening command:**

```bash
python -m halo_training --model models/vidar_halo.py --class-name VidarHaloAblation \
    --dataset datasets/BabyLM-2026-Strict --optimize-kernels \
    --muon --ema --scheduler wsd --z-loss 1e-4 --lr 0.003 \
    --seq-len 512 --warmup 50 --epochs 1 --checkpoint-interval 999999
```

### Tier M: Scale Confirmation Setup

| Parameter | Value |
|-----------|-------|
| Model | `VidarHaloGPT2` (d=768, 50257 vocab, ~54M unique) |
| Dataset | wikitext-103-raw.bin (119M tokens, pre-tokenized .bin) |
| Seq len | 512 |
| Duration | 1 epoch (~1816 steps) |
| Mode | Compiled + autokernel (`--compile --optimize-kernels`) |
| Hardware | Machine A or B |
| Throughput | ~35K tok/s (d=768, seq=512, compiled) |
| Wall time | **~57 min per run** |
| Parallel | 2 machines → 2 configs per ~1hr slot |
| LR | 0.002 |
| Warmup | 150 steps (proportional) |
| Seeds | 1 seed (seed 42) |
| Metrics | Final CE loss, BPB, tok/s, peak memory, NaN count |
| Checkpoints | None |

**Why Tier M exists:** Tier S runs at d=384 — optimizer behavior, GEMM dynamics, and compile interactions differ at d=768. Tier M catches scale-dependent failures (e.g., Polar NS that helps at d=384 but hurts at d=768) before committing to 8-hour Tier V runs.

**Why wikitext-103:** 119M tokens = 1 full epoch in ~57 min at d=768 compiled. Complete WSD cycle (warmup→stable→decay). Already on Machine A (`datasets/wikitext-103-raw.bin`). General-domain English text — more representative than BabyLM, smaller than stem-crawl.

**Scale confirmation command:**

```bash
python -m halo_training --model models/vidar_halo.py --class-name VidarHaloGPT2 \
    --dataset datasets/wikitext-103-raw.bin --compile --optimize-kernels \
    --muon --ema --scheduler wsd --z-loss 1e-4 --lr 0.002 \
    --seq-len 512 --warmup 150 --epochs 1 --checkpoint-interval 999999
```

### Tier V: Final Validation Setup

| Parameter | Value |
|-----------|-------|
| Model | `VidarHaloGPT2` (d=768, 50257 vocab, ~54M unique) |
| Dataset | stem-crawl-solo (544M tokens, pre-tokenized .bin) |
| Seq len | 1024 |
| Duration | 1 epoch (8,299 steps) |
| Mode | Compiled + autokernel (`--compile --optimize-kernels`) |
| Hardware | Machine A or B |
| Throughput | ~25K tok/s |
| Wall time | **~8 hours per run** |
| LR | 0.002 |
| Warmup | 300 steps |
| Seeds | 1 seed for confirmation; 3-seed on final combo |
| Metrics | Final BPB, tok/s, peak memory, NaN count, generation quality (5 fixed prompts) |
| Checkpoints | Final only |

**Final validation command:**

```bash
python -m halo_training --model models/vidar_halo.py --class-name VidarHaloGPT2 \
    --dataset datasets/stem-crawl-solo.bin --compile --optimize-kernels \
    --muon --ema --scheduler wsd --z-loss 1e-4 --lr 0.002 --epochs 1
```

### What Each Tier Detects

| Technique | Tier S (d=384, 12 min) | Tier M (d=768, 1 hr) | Notes |
|-----------|----------------------|---------------------|-------|
| P1a Polar NS | YES (directional) | **DEFINITIVE** | NS quality matters more on larger matrices |
| P1b MIN_LR | YES | YES | Schedule effect, scale-independent |
| P1c Iter scales | YES | YES | Structural, 2 params |
| P2a Softcap | YES | YES | Logit bounding, scale-independent |
| P3a Delayed recurrence | YES (short loop phase) | **DEFINITIVE** | Tier M has 1184 looped steps vs 328 at Tier S |
| P4a Parallel residuals | YES | **CHECK throughput** | Tier M reveals bandwidth cost at d=768 |
| P5a Skip connection | YES | YES | Structural, scale-independent |
| Compile interaction | NO (eager) | **YES** | Tier M uses compiled mode |
| Autokernel interaction | Partial (d_conv=320) | **YES** (d_conv=576) | Full production AK pattern |

### Naming Convention

```
checkpoints/vidar_abl_{tier}_{phase}_{variant}/
  e.g., vidar_abl_s_p1a_polar_ns/       # Tier S screening
        vidar_abl_v_p3a_delayed_recur/   # Tier V validation
        vidar_abl_v_mx_p1ab_p3a/         # Tier V matrix combo
```

---

## Part 1: Sequential Ablation (One Change at a Time)

Each phase tested independently against baseline at **Tier S first**. Winners graduate to Tier V. This isolates the effect of each technique.

### Baseline (B0-S): Ablation Variant, BabyLM

```bash
python -m halo_training --model models/vidar_halo.py --class-name VidarHaloAblation \
    --dataset datasets/BabyLM-2026-Strict --optimize-kernels \
    --muon --ema --scheduler wsd --z-loss 1e-4 --lr 0.003 \
    --seq-len 512 --warmup 50 --epochs 1 --checkpoint-interval 999999
```

**Expected:** ~12 min, ~504 steps, final CE TBD (establishes ablation baseline)

### Baseline (B0-V): Full Vidar, stem-crawl-solo

```bash
python -m halo_training --model models/vidar_halo.py --class-name VidarHaloGPT2 \
    --dataset datasets/stem-crawl-solo.bin --compile --optimize-kernels \
    --muon --ema --scheduler wsd --z-loss 1e-4 --lr 0.002 --epochs 1
```

**Expected:** ~8 hr, ~8299 steps, ~2.10 BPB (pending first Vidar run)

---

### Phase 1a: Polar-Express Newton-Schulz Coefficients

**Change:** Replace Muon's 5×fixed NS coefficients with per-iteration minimax tuples.

```python
# OLD (muon.py:27):
a, b, c = (3.4445, -4.7750, 2.0315)  # fixed, applied 5×

# NEW (Polar-Express, per-iteration minimax-optimized):
_PE_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
# Source: PR #1787/#1855, arXiv:2505.16932, verified from train_gpt.py
```

**Hypothesis:** Better polar factor approximation → faster convergence. Same FLOP count.

| Metric | Predicted | Measurement |
|--------|-----------|-------------|
| BPB delta | −0.005 to −0.015 | |
| tok/s delta | ±0% | |
| Memory delta | 0 | |

**Risk:** LOW. Same algorithm, different constants. Cannot cause instability.

---

### Phase 1b: MIN_LR = 10% of Peak

**Change:** WSD decay floor from 0 to 0.0002 (10% of lr=0.002).

```python
# trainer.py WSD scheduler:
min_lr = 0.1 * lr  # was 0.0
```

**Hypothesis:** Non-zero floor keeps useful gradient updates in final 20% of training.

| Metric | Predicted | Measurement |
|--------|-----------|-------------|
| BPB delta | −0.005 to −0.010 | |
| tok/s delta | 0% | |
| Memory delta | 0 | |

**Risk:** LOW. Cosine-to-zero is the more extreme choice; a floor is conservative.

---

### Phase 1c: Per-Iteration Learned Output Scale

**Change:** Replace layerwise LN scale (see Q2 resolution — runtime per-layer scale redundant with our init depth_scale). Instead, add per-iteration learned scale factors.

```python
# New parameter:
self.iter_scales = nn.Parameter(torch.ones(mean_recurrence))

# In _forward_unrolled, replace:
#   h = self.iter_norm(h) + self.loop_pos_embeds[i]
# with:
#   h = self.iter_norm(h) * self.iter_scales[i] + self.loop_pos_embeds[i]
```

**Hypothesis:** Let model learn how much each iteration contributes to the final representation. Iteration 0 (foundation) may need higher weight than iteration 1 (refinement), or vice versa. 2 learned scalars.

| Metric | Predicted | Measurement |
|--------|-----------|-------------|
| BPB delta | −0.001 to −0.005 | |
| tok/s delta | 0% | |
| Params added | 2 | |
| Memory delta | 0 | |

**Risk:** LOW. Initialized at 1.0 (identity). Worst case: stays at 1.0, no effect. Cannot cause instability.

---

### Phase 2a: Logit Softcap

**Change:** `logits = 30.0 * tanh(logits / 30.0)` before cross-entropy.

```python
# In forward, after lm_head:
logits = 30.0 * torch.tanh(logits / 30.0)
```

**Hypothesis:** Bounds logit magnitudes. Complementary to z-loss (z-loss is additive, softcap is hard bound). Helps fp16 stability.

| Metric | Predicted | Measurement |
|--------|-----------|-------------|
| BPB delta | −0.002 to −0.005 | |
| tok/s delta | −1% | |
| Memory delta | 0 | |

**Risk:** LOW. Gemma2 validated. May slightly overlap with z-loss benefit.

---

### Phase 3a: Delayed Recurrence Activation

**Change:** Train as flat 4-layer model for first 35% of steps, then enable Parcae loop.

```python
# In _forward_unrolled:
flat_phase = (self._current_step < 0.35 * self._total_steps)
if flat_phase:
    # Single pass through shared block, no injection, no second iteration
    h, _ = self._run_shared_block(h, freqs_cis, [])
    h = self.iter_norm(h) + self.loop_pos_embeds[0]
else:
    # Full 2-iteration loop (existing code)
    ...
```

**Hypothesis:** Flat phase trains faster per step (skip 2nd iteration → ~1.5x tok/s), builds strong representations. Loop activation adds depth on solid foundation. Net training time savings ~10-15%.

| Metric | Predicted | Measurement |
|--------|-----------|-------------|
| BPB delta | −0.005 to −0.015 | |
| tok/s (flat phase) | ~35-38K | |
| tok/s (loop phase) | ~25K (normal) | |
| Net wall time | −10 to −15% | |
| Memory delta | 0 | |

**Risk:** MEDIUM. Loss may spike at activation boundary. Mitigation: ramp injection strength over 500 steps at boundary rather than hard switch.

**Variant 3a-soft:** Gradual activation with injection strength ramp.

```python
if step < activation_step:
    injection_strength = 0.0
elif step < activation_step + 500:
    injection_strength = (step - activation_step) / 500
else:
    injection_strength = 1.0
```

---

### Phase 4a: 2-Lane Parallel Residuals (GQA Block Only)

**Change:** In VidarMoDAGQABlock, split residual into 2 lanes for attention and FFN.

```python
class VidarMoDAGQABlock(nn.Module):
    def __init__(self, ...):
        ...
        # Learned mixing scalars (4 values: attn→lane0, attn→lane1, ffn→lane0, ffn→lane1)
        self.post_mix = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 1.0]))
        self.res_scale = nn.Parameter(torch.ones(2))
    
    def forward(self, x, freqs_cis, depth_kvs=None):
        lane0, lane1 = x, x
        attn_out = self.attn(self.pre_norm(lane0), freqs_cis, depth_kvs=depth_kvs)
        ffn_out = self.ffn(self.ffn_norm(lane1))
        
        mix = self.post_mix
        lane0 = self.res_scale[0] * lane0 + mix[0] * attn_out + mix[2] * ffn_out
        lane1 = self.res_scale[1] * lane1 + mix[1] * attn_out + mix[3] * ffn_out
        return 0.5 * (lane0 + lane1)
```

**Hypothesis:** Allows attention and FFN to specialize without competing for residual bandwidth. Parameter-golf: −2.2 mBPB at their scale. Lighter than Hyperloop HC (no RMSNorm on expanded tensor, no stream projections).

| Metric | Predicted | Measurement |
|--------|-----------|-------------|
| BPB delta | −0.003 to −0.010 | |
| tok/s delta | −2 to −5% | |
| Params added | 6 (4 mix + 2 scale) | |

**Risk:** MEDIUM. HC streams were 35-41% slower at d=640 on our hardware. This is much lighter (scalars not matrices), but any residual split increases memory traffic. Measure throughput carefully.

---

### Phase 5a: Iteration Skip Connection

**Change:** Gated skip from iteration 0 output to iteration 1 input.

```python
# New parameter:
self.skip_gate = nn.Parameter(torch.zeros(d_model))

# In _forward_unrolled, after iter 0:
h0_skip = h  # save iter 0 output

# Before iter 1 shared block:
g = torch.sigmoid(self.skip_gate)
h = h + g * h0_skip  # additive skip
```

**Hypothesis:** Direct feature bypass from iter 0 to iter 1 complements MoDA's attention-mediated cross-iteration info. MoDA is selective (attention-weighted); skip is direct (full bandwidth).

| Metric | Predicted | Measurement |
|--------|-----------|-------------|
| BPB delta | −0.003 to −0.008 | |
| tok/s delta | −1% | |
| Params added | 768 | |

**Risk:** LOW. Zero-init gate = transparent at start. Worst case: gate stays near zero, no effect.

---

### Phase 6a: JEPA Auxiliary Loss

**Change:** 2-layer MLP predicts next hidden state. MSE loss with stop-gradient on target. Training only.

```python
class JEPAPredictor(nn.Module):
    def __init__(self, d_model, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
        )
    
    def forward(self, h):
        pred = self.net(h[:, :-1])
        target = h[:, 1:].detach()
        return F.mse_loss(pred, target)

# In training loss:
jepa_loss = self.jepa_predictor(h)
loss += 0.1 * jepa_loss
```

**Hypothesis:** Auxiliary representation shaping signal. Forces hidden states to be locally predictive, improving feature geometry. Zero inference cost (predictor discarded).

| Metric | Predicted | Measurement |
|--------|-----------|-------------|
| BPB delta | −0.002 to −0.005 | |
| tok/s delta | −3 to −5% (training) | |
| Params added | ~1.2M (training only) | |

**Risk:** LOW-MEDIUM. May not help if CE + MTP already provide sufficient representation shaping.

---

### Phase 7a: Progressive Recurrence

**Change:** Three-phase depth schedule: flat → mean=1 → mean=2.

| Training % | Mode | Effective Depth | Est. tok/s |
|-----------|------|----------------|-----------|
| 0 - 25% | Flat (no loop) | 4L | ~35K |
| 25 - 50% | mean=1 (with injection) | 4L + injection | ~30K |
| 50 - 100% | mean=2 (full) | 8L effective | ~25K |

**Hypothesis:** Progressive depth lets model build representations at increasing complexity. Saves ~15-20% wall time vs full mean=2 from step 0.

| Metric | Predicted | Measurement |
|--------|-----------|-------------|
| BPB delta | −0.005 to −0.015 | |
| Net wall time | −15 to −20% | |
| Memory delta | 0 | |

**Risk:** MEDIUM. Two activation boundaries. Each could cause loss spikes. Need smooth transitions.

**Note:** This subsumes Phase 3a. If Phase 3a works, Phase 7a extends it further.

---

## Part 2: Matrix Ablation (Sensible Combinations)

After sequential ablation identifies individual winners, test combinations. Only combinations with theoretical synergy — not random mixing.

### Combination Logic

| Combo | Rationale | Interference Risk |
|-------|-----------|------------------|
| 1a + 1b | Optimizer + schedule, independent mechanisms | NONE |
| 1a + 1b + 1c | All zero-cost training improvements | LOW (LN scale interacts with depth_scale init) |
| 3a + 1a + 1b | Delayed recurrence + better optimizer + better schedule | LOW |
| 4a + 5a | Parallel residuals + skip connection = dual-path info flow | MEDIUM (both modify residual stream) |
| 3a + 7a | Delayed → Progressive recurrence (7a subsumes 3a) | NONE (sequential) |
| 2a + 1c | Softcap + LN scale = dual logit/activation control | LOW (complementary) |
| 6a + everything | JEPA on top of best combo | LOW (additive loss) |

### Matrix Design

**Tier 1: Zero-cost stack** (run first, ~30 min setup)

| ID | Config | Expected BPB | Notes |
|----|--------|-------------|-------|
| MX-01 | B0 + P1a | Baseline − 0.010 | Polar NS only |
| MX-02 | B0 + P1a + P1b | Baseline − 0.017 | + MIN_LR floor |
| MX-03 | B0 + P1a + P1b + P1c | Baseline − 0.019 | + iter scales (full Phase 1) |
| MX-04 | B0 + P1a + P1b + P2a | Baseline − 0.022 | Phase 1ab + softcap (skip iter scales) |

Compare MX-03 vs MX-04: does iter scaling or softcap add more on top of P1ab?

**Tier 2: Recurrence modifications** (run after Tier 1 winner identified)

| ID | Config | Expected BPB | Notes |
|----|--------|-------------|-------|
| MX-10 | BEST_T1 + P3a | T1 − 0.010 | + delayed recurrence |
| MX-11 | BEST_T1 + P7a | T1 − 0.012 | + progressive recurrence (subsumes 3a) |
| MX-12 | BEST_T1 + P3a-soft | T1 − 0.010 | + delayed recurrence (soft ramp) |

Compare MX-10 vs MX-12: hard switch vs soft ramp at activation boundary.

**Tier 3: Residual modifications** (run after Tier 2 winner identified)

| ID | Config | Expected BPB | Notes |
|----|--------|-------------|-------|
| MX-20 | BEST_T2 + P4a | T2 − 0.006 | + parallel residuals (GQA) |
| MX-21 | BEST_T2 + P5a | T2 − 0.005 | + iteration skip |
| MX-22 | BEST_T2 + P4a + P5a | T2 − 0.009 | Both residual mods |

Compare MX-20 vs MX-21: which residual modification matters more?
MX-22 tests for synergy vs interference.

**Tier 4: Auxiliary loss** (run on Tier 3 winner)

| ID | Config | Expected BPB | Notes |
|----|--------|-------------|-------|
| MX-30 | BEST_T3 + P6a | T3 − 0.003 | + JEPA aux loss |

**Tier 5: Full stack** (final combination)

| ID | Config | Expected BPB | Notes |
|----|--------|-------------|-------|
| MX-40 | All winners stacked | Baseline − 0.06 to −0.10 | Best from each tier |
| MX-41 | MX-40 3-seed | Same ± std | Statistical validation |

---

## Part 3: Throughput vs Quality Matrix

If any technique hurts tok/s > 5%, test with/without on throughput-sensitive configurations.

| Config | Target | Tradeoff |
|--------|--------|----------|
| VIDAR-SPEED | Max tok/s, BPB < 2.15 | Drop anything costing > 2% tok/s |
| VIDAR-QUALITY | Min BPB, tok/s > 20K | Accept up to 20% tok/s loss |
| VIDAR-BALANCED | BPB < 2.05, tok/s > 23K | Best of both |

---

## Execution Timeline

### Day 1: Tier S Screening — Phase 1 (optimizer + schedule)

All runs: VidarHaloAblation d=384, BabyLM 1 epoch, ~12 min each.

| Slot | Machine A | Machine B |
|------|-----------|-----------|
| 1 (0:00-0:15) | B0-S baseline | B0-S baseline (verify reproducibility) |
| 2 (0:15-0:30) | P1a (Polar NS) | P1b (MIN_LR) |
| 3 (0:30-0:45) | P1a+P1b combo | P1c (iter scales) |
| 4 (0:45-1:00) | P2a (softcap) | P1a+P1b+P1c |

**8 runs in 1 hour.** Analyze results, identify Phase 1 winners.

### Day 1 (cont): Tier S Screening — Phase 3+4+5 (architecture)

| Slot | Machine A | Machine B |
|------|-----------|-----------|
| 5 (1:00-1:15) | P3a (delayed recurrence) | P3a-soft (ramp variant) |
| 6 (1:15-1:30) | P4a (parallel residuals) | P5a (skip connection) |
| 7 (1:30-1:45) | P3a + P4a | P3a + P5a |
| 8 (1:45-2:00) | P4a + P5a | Best Phase 1 + best Phase 3 |

**16 runs in 2 hours total.** All individual phases + key combos screened.

### Day 1 (cont): Tier S Matrix — Best combos

| Slot | Machine A | Machine B |
|------|-----------|-----------|
| 9 (2:00-2:15) | MX-01 (all Phase 1 winners) | MX-10 (Phase 1 + Phase 3 winner) |
| 10 (2:15-2:30) | MX-20 (Phase 1+3 + residual winner) | MX-40 (full stack preview) |

**20 runs in 2.5 hours.** Full screening done in a single session.

### Day 2: Tier M — Scale Confirmation (~1 hr each)

Promote Tier S winners (CE delta ≥ −0.05) to d=768 on wikitext-103. Also run combos.

| Slot | Machine A (~1 hr each) | Machine B (~1 hr each) |
|------|------------------------|------------------------|
| 1 (0:00-1:00) | B0-M baseline (d=768, WT103) | Best single technique from Tier S |
| 2 (1:00-2:00) | 2nd best single technique | 3rd best single technique |
| 3 (2:00-3:00) | Best combo from Tier S | Best combo + runner-up technique |

**6 runs in 3 hours.** Now have d=768 rankings. Compare Tier S vs Tier M ranking:
- **Rankings match:** Screening protocol is valid. Proceed confidently.
- **Rankings diverge:** Scale-dependent effect found. Trust Tier M.

### Day 2 (cont) or Day 3: Tier V — Final Validation

Promote top 2 Tier M winners to stem-crawl-solo, 1 full epoch.

| Run | Machine A (8 hr) | Machine B (8 hr) |
|-----|-------------------|-------------------|
| Run 1 | B0-V (Vidar d=768, stem-crawl baseline) | Best Tier M combo |
| Run 2 | Second-best Tier M combo | (spare or alternative) |

**4 runs in ~16 hours** (2 serial runs per machine).

### Day 3 or 4: Tier V — 3-seed validation

| Run | Machine A | Machine B |
|-----|-----------|-----------|
| AM | Final combo, seed 42 | Final combo, seed 0 |
| PM | Final combo, seed 1234 | (spare if seeds diverge) |

### Decision Day

**3-seed Tier V mean determines the dolma-10b configuration.**

### Summary

| Phase | Runs | Wall Time | Compute |
|-------|------|-----------|---------|
| Tier S screening (d=384, BabyLM) | 20 | **2.5 hours** | 2 machines × 2.5 hr |
| Tier M confirmation (d=768, WT103) | 6 | **3 hours** | 2 machines × 3 hr |
| Tier V validation (d=768, stem-crawl) | 4 | **16 hours** | 2 machines × 2 × 8 hr |
| Tier V 3-seed | 3 | **16 hours** | 2 machines × 8 hr + 1 serial |
| **Total** | **~33** | **~3-4 days** | |

**Funnel: 20 configs → 6 survivors → 2 finalists → 1 winner (3-seed validated).**

---

## Success Criteria

### Tier S (Screening at d=384, BabyLM)

| Outcome | Action |
|---------|--------|
| CE delta ≥ −0.05 vs B0-S | **KEEP** — promote to Tier M |
| CE delta −0.02 to −0.05 vs B0-S | **MAYBE** — promote if combo with other winners helps |
| CE delta > −0.02 vs B0-S | **DROP** — within noise, not worth Tier M compute |

### Tier M (Scale Confirmation at d=768, WT103)

| Outcome | Action |
|---------|--------|
| CE delta ≥ −0.03 vs B0-M | **KEEP** — promote to Tier V |
| CE delta −0.01 to −0.03 vs B0-M | **MAYBE** — include in best combo only |
| CE delta > −0.01 vs B0-M | **DROP** — doesn't survive at scale |
| tok/s drop > 10% vs B0-M | **FLAG** — acceptable only if BPB gain justifies it |

### Tier V (Final Validation at d=768, stem-crawl-solo)

| Outcome | Action |
|---------|--------|
| Final combo BPB < 2.00 | Ship as VIDAR-HALO v2, move to dolma-10b |
| Final combo BPB 2.00-2.05 | Good improvement, proceed to dolma-10b |
| Final combo BPB 2.05-2.10 | Marginal; cherry-pick individual winners only |
| Final combo BPB > 2.10 | Techniques don't transfer at scale; revert to baseline |

### Transfer Validity Checks

**S→M correlation:** Tier S rankings must broadly match Tier M rankings. If a technique ranks #1 at Tier S but shows no gain at Tier M, that technique has a scale-dependent failure. Flag it, trust Tier M.

**M→V correlation:** If Tier M deltas don't predict Tier V deltas, the wikitext-103→stem-crawl domain gap is too large. Future ablations should use a stem-crawl subset for Tier M instead.

---

## Analysis Template (per run)

```markdown
## Run: [MX-XX] [descriptive name]
**Date:** YYYY-MM-DD
**Config:** [list all active techniques]
**Seed:** [42/0/1234]

### Results
| Metric | Value |
|--------|-------|
| Final BPB | |
| Final CE loss | |
| Mean tok/s (last 1K steps) | |
| Peak memory (GB) | |
| Wall time (seconds) | |
| NaN count | |
| Steps completed | |

### vs Baseline
| Metric | Delta | Significant? |
|--------|-------|-------------|
| BPB | | |
| tok/s | | |

### Loss Curve Notes
[Any spikes, instabilities, interesting patterns]

### Generation Samples (5 fixed prompts)
[Quality assessment]

### Verdict: [KEEP / DROP / CONDITIONAL]
[Reasoning]
```

---

## Open Questions — Resolution Log

### Q1: Polar-Express NS Coefficients — RESOLVED (2026-05-04)

**Problem:** Needed exact per-iteration minimax tuples from parameter-golf winner.

**Resolution:** Fetched from PR #1855 / PR #1787 `train_gpt.py`. Exact values:

```python
_PE_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),   # aggressive initial
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),   # taper
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),  # taper
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106), # taper
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),# stabilizer
)
```

Source uses bf16 + `@torch.compile` for NS — we keep **fp32 eager** (bf16 24% slower on gfx1151, compile causes 29GB memory blowup per unique weight shape). Drop-in replacement for `muon.py:27`. Referenced paper: arXiv:2505.16932 (originated PR #1344 by @Omrigotlieb).

### Q2: Layerwise LN Scale vs depth_scale Init — RESOLVED (2026-05-04)

**Problem:** Vidar already scales output projections by `1/sqrt(2 * n_layers * mean_recurrence)` at init. Adding runtime `1/sqrt(idx+1)` could over-shrink deep layers.

**Resolution:** Skip runtime LN scale. Init-only depth_scale already handles it.

Reasoning:
- Parameter-golf needs runtime scale because they use standard Xavier init (no depth-aware scaling). We already have µP-style init scaling.
- Stacking both = two mechanisms doing one job. One mechanism = one thing to debug.
- Parcae loop makes "layer index" ambiguous — same physical layer runs at two effective depths. Init scaling handles this cleanly; runtime `1/sqrt(idx+1)` would need iteration-aware control flow in the compiled graph.

**Replacement for P1c:** Per-iteration learned output scale (2 parameters, zero-cost):
```python
self.iter_scales = nn.Parameter(torch.ones(mean_recurrence))
# After each iteration:
h = self.iter_norm(h) * self.iter_scales[i] + self.loop_pos_embeds[i]
```

### Q3: Delayed Recurrence + torch.compile Cache — RESOLVED (2026-05-04)

**Problem:** Switching from flat (1 iter) to looped (2 iter) mid-training invalidates compiled graph.

**Resolution:** Call `torch._dynamo.reset()` at the activation boundary. Same pattern already used for context scheduling in `trainer.py:399`. Recompilation costs 5-10 min (~1% of 8-hour run). Disk cache stores both graphs after first run — subsequent runs instant.

```python
if step == activation_step:
    self._delayed_flat = False
    torch._dynamo.reset()
```

### Q4: Parallel Residuals + autokernel — RESOLVED (2026-05-04)

**Problem:** Would autokernel pattern-matching break with modified GQA block?

**Resolution:** Non-issue. `VidarMoDAGQABlock` already has `_skip_autokernel = True`. Autokernel never touches this block. Parallel residuals modify internal routing only — no external interface change.

### Q5: JEPA vs MTP Overlap — DEFERRED (2026-05-04)

**Problem:** Both are auxiliary representation-shaping losses. May be redundant.

**Resolution:** Defer to dolma-10b training phase. Reasoning:
- MTP validated at massive scale (DSV4). JEPA for text LM has one data point (parameter-golf non-record).
- 1-epoch ablation too short to distinguish auxiliary loss effects — need longer training.
- If Phases 1-5 deliver expected gains, JEPA's marginal −0.003 BPB is diminishing returns.
- When tested on dolma-10b: run 2 configs (MTP+JEPA vs MTP-only) to measure additive value.
