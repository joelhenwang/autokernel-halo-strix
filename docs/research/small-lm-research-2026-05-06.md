# Small LM Research Synthesis — 2026-05-06

Deep dive into efficient pretraining, continued pretraining, post-training, and
small-LM architectures. Synthesized from our existing knowledge base (17+ prior
research notes) plus targeted external research (HuggingFace SmolLM3 July 2025
blueprint, APO arXiv:2408.06266).

**Target audience:** decisions about what to do next with OdinFlat (122M) and
OdinHalo (58M unique / 157M effective) on 2× Strix Halo / gfx1151 / TB4 DDP.

**Current state (as of 2026-05-06):**
- OdinFlat: wikitext → gpt-small → stem-crawl (in progress), ~950M cumulative tokens
- OdinHalo: wikitext → gpt-small, ~420M cumulative tokens
- Loss gap collapsed 0.24 → 0.03 between wikitext and gpt-small — looped catching up
- Throughput defaults tuned via empirical sweep (block=512, num_workers=12)
- Zero post-training pipeline. Single-metric eval (loss/BPB only).

---

## TL;DR — top 5 recommendations ranked by ROI

| # | Action | Why now | Effort | Expected impact |
|:-:|--------|---------|-------:|-----------------|
| 1 | Adopt 3-stage pretraining recipe (stable → quality-upsample → decay) | SmolLM3 proves it at 3B; translates directly to our trajectory | 2 hours | +0.1-0.3 loss vs single-stage |
| 2 | Add **intra-document masking** to trainer | Nearly free training stability improvement; confirmed by Llama-3 + SmolLM3 | 1 hour | More stable long runs, cleaner gradients |
| 3 | Remove weight decay from embedding layers | OLMo 2 finding, adopted by SmolLM3; free win | 5 min | Better embedding norms, slight loss improvement |
| 4 | Build minimal SFT pipeline + APO/ORPO post-training | We have 0 post-training; the ablation doc is comprehensive but unused | ~2 days | Unlocks instruct-mode evaluation, enables all downstream work |
| 5 | Switch from cosine to **WSD scheduler** with `--min-lr-ratio 0.1` | AGENTS.md already notes this; matches SmolLM3 recipe | trivial | Cleaner final loss, better checkpoint quality for resume |

---

## Part 1 — What the knowledge base already has

Our existing docs (especially `knowledge/INDEX.md` + `docs/adal/llm_novelties_2025_2026_report.md`)
cover an impressive breadth. Items NOT duplicated below:

- **Architectures:** Parcae looped models, HyPE positional, HALO/HypeNet conversion,
  MatFormer, PLE, Lightning Attention, CompreSSM
- **Optimizers:** Muon (2× token efficiency, 3.5× slower step), Lion, CLion,
  fused AdamW baseline
- **Alignment:** SimPO, ORPO, KTO, RePO, AlphaPO, MIWV data selection,
  DataFlow (10K ≥ 1M)
- **Training:** DDP setup, RCCL gfx1151 build, SFT pipeline, EOS warmup,
  ChatML template, curriculum learning
- **Hardware:** Full gfx1151 reference, workload guidance, 20+ kernel benchmarks

The gaps this document addresses are **new-or-refreshed material** (SmolLM3 recipe
details, APO, staged pretraining formalism) and **Odin-specific integration**
(how to apply findings given our current trajectory).

---

## Part 2 — Efficient pretraining for small LMs

### Key finding: staged pretraining is the new default

SmolLM3's 3-stage recipe for 3B / 11T tokens scales down cleanly to our 122M
scale. The structure:

| Stage | Token fraction | Web | Code | Math | Purpose |
|-------|---------------:|----:|-----:|-----:|---------|
| 1 Stable | 72% (0 → 8T) | 85% | 12% | 3% | Broad foundations |
| 2 Stable | 18% (8T → 10T) | 75% | 15% | 10% | Upsample quality |
| 3 Decay | 10% (10T → 11.1T) | 63% | 24% | 13% | Cement high-quality, +reasoning data |

Critical details that generalize:

1. **WSD (Warmup-Stable-Decay) scheduler**, NOT cosine. Warmup 2000 steps, stable
   at peak LR through stages 1 + 2, linear decay to 0 in final 10% of total steps.
2. **LR=2e-4, AdamW (0.9, 0.95), wd=0.1, grad_clip=1.0**.
3. **Global batch = 2.36M tokens, sequence length 4096**. Batch scales with model size.
4. **Remove weight decay from embeddings** (OLMo 2 finding). Embeddings stabilize
   at healthier norms without it.
5. **Intra-document attention masking**: packed documents in same sequence don't
   attend to each other. Crucial for stability at long context.
6. **GQA** with 4 groups (matches full attention on quality, halves KV cache).
7. **Tied embeddings** (already our default).

### Mapping to our Odin trajectory

We've inadvertently built a 3-stage pipeline already:

| Our stage | Dataset | Tokens | Corresponds to SmolLM3 stage |
|-----------|---------|-------:|------------------------------|
| 1 | wikitext-103 | 123M | Weak analogue of Stable (narrow, not 85% web) |
| 2 | gpt-training-small | 296M | Weak analogue of Quality-upsampled stage |
| 3 (active) | stem-crawl-solo | 531M | Stem-heavy variant of Quality-upsampled |
| ??? | — | — | **Decay phase missing** |

We're missing the decay phase: a short final run (~5-10% of total compute) on
the **best** data at **decaying LR** to polish the model. For our ~950M
cumulative budget, that's a ~50-100M token epoch on a curated high-quality
subset with LR → 0.

**Actionable experiment:** After stem-crawl completes, run a decay phase of
~75M tokens on a hand-picked high-quality mix (e.g., 50% StackExchange + 30%
wikitext hand-curated + 20% code from existing corpora) with:
- `--scheduler wsd --min-lr-ratio 0.0` (decay to 0, not 10%)
- Peak LR 3e-4 (half of prior stage's 6e-4)
- 500-step warmup, then full decay

This is expected to improve final loss by 0.1-0.2 based on SmolLM3's observations.

### Intra-document masking — nearly-free stability win

**Problem:** Current trainer packs multiple documents into each sequence
(`block_size=512`). Tokens from document A attend to tokens from document B
via causal attention, treating the junk boundary as legitimate context.
Creates weird cross-document dependencies during gradient updates.

**Fix:** Build a document boundary mask during dataloader setup, passed to
attention. Prevents cross-document attention.

**Implementation cost for Odin:** ~1 hour. The HyPE attention path already
takes an optional mask parameter. Needs:
1. Dataloader: build `doc_id` tensor per token (from EOS positions).
2. In model forward: derive `attn_mask[i,j] = (doc_id[i] == doc_id[j])`, intersect
   with causal mask.
3. Apply in `NoPECodaAttention.forward` and `Attention.forward`.

**Risk:** None — documented-safe in Llama 3, OLMo 2, SmolLM3.

**Expected impact:** Marginal loss improvement (~0.02-0.05), cleaner training
dynamics, enables longer sequences without quality degradation.

### Remove weight decay from embeddings — 5-minute win

**Current:** Our fused AdamW applies `weight_decay=0.1` to all parameters,
including `tok_embeddings.embed.weight` (~8.6M params = 7% of model).

**Fix:** Split parameter groups:
```python
no_decay_params = [p for n, p in model.named_parameters() if "embed" in n or "norm" in n]
decay_params    = [p for n, p in model.named_parameters() if p not in no_decay_params]
optimizer = AdamW([
    {"params": decay_params,    "weight_decay": 0.1},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=lr, betas=(0.9, 0.95), fused=True)
```

**Implementation cost:** 5 minutes. Single edit in `train_ddp.py` and
`halo_training/optimizer.py`.

**Expected impact:** Slight reduction in embedding norm drift, marginal loss
improvement. Free upside.

---

## Part 3 — Continued pretraining

Our wikitext → gpt-small → stem-crawl chain IS continued pretraining, but
without the refinements the CPT literature suggests.

### Known-good CPT recipe elements

1. **Lower peak LR than original pretraining** (~30-50% of base LR).
   - We did this: 8e-4 → 6e-4 at each resume (25% lower). Could go further to
     4e-4 for more conservative continuation.
2. **Token replay buffer** — mix in 5-15% of the ORIGINAL pretraining data with
   the new domain. Prevents catastrophic forgetting.
   - We have NOT been doing this. Each resume is 100% new data.
3. **Fresh optimizer vs resumed optimizer**:
   - We use fresh (train_ddp.py resume-from loads weights only). This creates
     the observed warmup loss spike.
   - Resumed-optimizer variant would preserve AdamW momentum across datasets.
     Needs a new code path in train_ddp.py.
4. **Warmup proportional to remaining training** — our 500-step warmup for a
   4,046-step stem-crawl run is ~12% of training. Reasonable.

### Recommended Odin CPT refinements

**Experiment A: Token replay.** Mix 10% wikitext tokens into stem-crawl next
time. Requires a new `MixtureDataset` path. Already exists (see
`halo_training/mixture_data.py`), just need to wire it up.

**Experiment B: Resumed optimizer.** Add `--resume-optimizer` flag. When set,
load `optimizer_state_dict` from the checkpoint (already saved at
`train_ddp.py:104`). Avoids the loss spike on resume.

**Experiment C: Staged LR across resumes.** Each resume uses progressively
lower peak LR: 8e-4 (wikitext fresh) → 6e-4 (gpt-small) → 4e-4 (stem-crawl)
→ 2e-4 (decay phase). Models the full WSD schedule across the trajectory.

### Insight from cross-run comparison

The OdinFlat vs OdinHalo loss gap narrowing (0.24 → 0.03 after gpt-small)
is not an accident — it matches the "looped models benefit more from data" thesis
of the Parcae paper. **Prediction:** After stem-crawl, OdinHalo (if trained
symmetrically) should match or surpass OdinFlat on held-out loss. This is
worth verifying:
- Run OdinHalo on stem-crawl as a follow-up
- Evaluate both on a held-out slice (e.g., babylm validation) for proper
  comparison

---

## Part 4 — Post-training (where we have ZERO infrastructure)

Our knowledge base has the best post-training survey I've seen
(`knowledge/training/instruct_alignment_techniques_2025_2026.md`, 429 lines)
but **nothing is implemented**. This is the biggest capability gap.

### New material: Anchored Preference Optimization (APO)

Not in our existing survey. Paper: arXiv:2408.06266 (D'Oosterlinck et al., 2024,
v5 in Sep 2024; adopted by SmolLM3 in July 2025).

**What it is:** DPO variant with an anchor term that controls how far the
policy drifts from the reference model. More stable than DPO, competitive
with or better than ORPO/SimPO in empirical comparisons.

**Paired with CLAIR** (Contrastive Learning from AI Revisions) for data:
LLM revises its own outputs to create contrastive preference pairs. Works
with synthetic data generation from a larger teacher.

**Why it matters for Odin:** SmolLM3's recipe used APO for alignment and
model-merged with a mid-training checkpoint. At 3B this produced the SOTA
dual-mode reasoning model. For our 58-122M scale:
- APO's stability advantage likely matters more (tiny models are fragile).
- The CLAIR data-generation recipe is valuable: we don't have human preference
  labels but we can use Qwen3-32B or similar to revise our outputs.

**Alternative rankings for 80M-class models (from our existing survey):**
1. **ORPO** — single-stage (no SFT phase), no reference model. ~50% cheaper.
2. **SimPO** — no reference model, simple. Use if already have SFT data.
3. **APO** — more stable than DPO, worth trying if ORPO/SimPO plateau.
4. **KTO** — works with unpaired labels. Easiest data collection.

### Recommended minimum post-training stack

For Odin at 122M, a realistic post-training pipeline:

```
Step 1: Light SFT
  - Use SmolLM3-released SmolTalk2 dataset (filter to <2000 tokens for our seq len)
  - OR: generate instructions with Qwen3-32B via our stem-crawl topic seeds
  - 1 epoch, LR 2e-5, no masking on assistant turns
  - Expected: 2-3 hours on 2-machine DDP

Step 2: ORPO alignment  (recommended starting point — single stage)
  - Need ~5K preference pairs
  - Options for data:
    - Tulu 3 preference mixture (public, Llama-tuned but transferable)
    - Self-generate with temp=0.7 + temp=1.5 samples, keep low-temp as chosen
  - Single training phase, combines SFT + preference
  - Expected: 1-2 hours

Step 3: (optional) APO + model merging
  - Only if ORPO plateaus
  - Generate CLAIR-style data from larger teacher
  - Merge APO checkpoint with pre-APO checkpoint via linear interp (0.9/0.1)
```

**Deliverables needed:**
1. `halo_training/sft.py` — already scaffold exists per `knowledge/training/sft_pipeline.md`
2. `halo_training/orpo.py` — NEW, ~200 lines from scratch
3. `scripts/generate_sft_data.py` — pipe outputs from larger LM (requires
   API access or local Qwen3-32B)
4. Evaluation suite: MMLU, HellaSwag, BLiMP, MT-Bench — see
   `docs/tutorial/09_train_eval_benchmark.md` for lm-eval-harness integration

---

## Part 5 — Architectures and components worth tracking

### In our ecosystem (already on the radar)

- **Parcae looped** — we have this (OdinHalo).
- **HyPE** — we have this.
- **MatFormer** — not integrated; could swap into OdinFlat for elastic inference.
- **MoE at small scale** — OLMoE (1B active / 7B total), SmolMoE-style recipes.
  Not yet evaluated at our scale.

### Newer arrivals (from external research)

#### ATLAS (arXiv:2505.23735, May 2025)

Referenced in SmolLM3 comments by a community member. Google Research paper
proposing a "long-term memory module" that explicitly learns to memorize context,
not individual tokens. Uses an "Omega Rule" — memory updated based on a sliding
window of past tokens rather than online token-by-token.

Claims 10M context scaling on BABILong benchmark.

**Relevance to Odin:** If we want to push context beyond 2048, ATLAS is worth
reading. Not a near-term priority given our current 512-token training budget.

#### APO (covered in Part 4)

#### Mamba-3 / Hybrid Stacks

We haven't evaluated pure-SSM alternatives at our scale. OdinHalo's looped
hybrid is more parameter-efficient than pure Mamba-3 at matched FLOPs
(per our own hypothesis buildout), but Mamba-3's fast decode is attractive.

**Parking lot, not urgent:** Revisit if post-training highlights decode latency
as the main deployment issue.

#### SmolLM3 architecture specifics transferable to Odin

From the SmolLM3 blog:
- **NoPE on every 4th layer** (Yang et al., arXiv:2501.18795). Different from
  our HyPE (NoPE in GQA, RoPE in conv). Could test as an ablation.
- **GQA groups=4** (matches our current). ✓
- **Tied embeddings**. ✓
- **No WD on embeddings** (see Part 2).
- **Intra-document masking** (see Part 2).

### Things we've tried and ruled out

From our prior hypothesis buildout (`knowledge/architectures/hypothesis_buildout_results.md`):
- AMADEUS wins quality at 170M but complex deployment. Stick with simpler Odin.
- MoE at 170M — marginal. Skip until >500M total params.
- Deep-narrow (48L × d=512) — inference latency kills it.

---

## Part 6 — Evaluation — our biggest blind spot

Per `knowledge/architectures/reliable_small_lm_insights.md` GAP 3:
> "We track loss, BPB, tok/s, MFU. That's optimization health only."

Missing:
1. **Per-domain BPB** — split eval corpus by source (wikitext, stem-crawl,
   gpt-small subsets). Shows where continued pretraining helps/hurts.
2. **Capability probes** — BLiMP for grammatical competence (zero-shot, fast).
3. **Fixed sample packs** — 20-30 prompts with expected-style outputs for
   qualitative regression. We have `scripts/ablate_odin_flat_sampling.py` —
   could evolve into this.
4. **Repetition metrics** — distinct-2 (we measure), distinct-4, self-BLEU.
5. **Quantized BPB** — int4 vs fp16 on same validation. Matters for deployment.
6. **Standard benchmarks** — HellaSwag, ARC, MMLU-CF, PIQA. Via lm-evaluation-harness.

### Recommended evaluation deliverable

`scripts/eval_checkpoint.py`:
- Load checkpoint
- Compute per-domain BPB on 4 validation sources (babylm-hold, wikitext-hold,
  gpt-small-hold, stem-crawl-hold)
- Run our existing sampling ablation (already have it)
- Run BLiMP via lm-evaluation-harness (add dependency)
- Emit a JSON scorecard + append to `docs/perf/eval-scoreboard.jsonl`

This would be ~1 day to implement. Pays off immediately — every future
checkpoint gets a rich report card instead of just a loss number.

---

## Part 7 — Actionable experiment queue (ranked)

| Rank | Experiment | Cost | Value | Dependencies |
|:----:|-----------|------|-------|--------------|
| 1 | Remove WD from embeddings + intra-doc masking | 1 hr | Universal, permanent | None |
| 2 | Build minimal eval suite (`eval_checkpoint.py`) | 1 day | Every future checkpoint gets a scorecard | lm-eval-harness dep |
| 3 | Decay phase on curated mix after stem-crawl | 4 hr | +0.1-0.2 loss, cement quality | Stem-crawl finishing |
| 4 | Implement ORPO on Odin (+ SmolTalk2 data) | 2 days | First instruct-capable Odin | Eval suite exists |
| 5 | OdinHalo stem-crawl run (symmetric comparison) | 4 hr | Verify looped catches up thesis | Stem-crawl OdinFlat finished |
| 6 | APO + model merging | 1 day after ORPO | Higher-quality instruct model | ORPO working |
| 7 | Intra-document masking ablation | 1 day | Confirm the fix helps | Fix implemented |
| 8 | Token replay in CPT | 1 day | Prevents domain-drift in continued pretraining | MixtureDataset wiring |
| 9 | Move from cosine to WSD scheduler | 1 hr | Cleaner final loss | None |
| 10 | Test NoPE-every-4th as HyPE alternative | 2 days | May beat our current HyPE | New model variant |

### Priority reasoning

- **#1-2 are foundation** — both pay off immediately on every subsequent
  experiment. Do these first.
- **#3-4 unlock the next milestone** — a decay-polished base + an instruct
  variant = "deployable small LM" for the first time.
- **#5-6 are validation + refinement** — confirm hypotheses, bump quality.
- **#7-10 are research bets** — good to know, not critical path.

---

## Part 8 — What to do now

Given the active stem-crawl run (ETA ~3 hours) and our two-machine setup:

**Immediate (while stem-crawl runs):** Implement experiments #1 (WD + intra-doc
masking) so they're ready to apply to the next run. Build a basic eval
scorecard (#2) — can run on each completed checkpoint retroactively.

**After stem-crawl completes:**
1. Run the eval scorecard on all 4 OdinFlat checkpoints (wikitext final,
   gpt-small final, stem-crawl checkpoints, final).
2. Analyze the per-domain BPB trajectory to see which dataset helped most.
3. Decide: decay phase next, or symmetric OdinHalo-on-stem-crawl run first?

**Medium term (next session):**
1. First ORPO run (1-2 days).
2. First sample-set evaluation.

**Long term (multiple sessions):**
1. APO + merging for a dual-mode reasoning Odin (borrowing SmolLM3 recipe).
2. Evaluation suite expansion (MT-Bench, HumanEval).

---

## References

### Our existing knowledge base
- `knowledge/INDEX.md`
- `knowledge/architectures/reliable_small_lm_insights.md` — Labonne guide insights
- `knowledge/architectures/parcae_stable_looped_models.md` — Parcae paper
- `knowledge/training/instruct_alignment_techniques_2025_2026.md` — post-training survey
- `knowledge/training/sft_pipeline.md` — SFT scaffold
- `knowledge/training/muon_optimizer_results.md` — Muon findings
- `docs/adal/llm_novelties_2025_2026_report.md` — HALO, HypeNet, ATLAS etc.

### External research (fetched 2026-05-06)
- HuggingFace SmolLM3 blueprint (July 2025) — https://huggingface.co/blog/smollm3
- APO paper (arXiv:2408.06266, Sep 2024 v5) — D'Oosterlinck et al.
- NoPE-every-4th paper (arXiv:2501.18795, Yang et al., 2025)
- Intra-Document Masking (arXiv:2402.13991)
- OLMo 2 — no-WD-on-embeddings finding (public blog)
- ATLAS (arXiv:2505.23735) — long-term memory module

### Key data
- SmolLM3 training configs: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs
- SmolTalk2 SFT dataset: https://huggingface.co/datasets/HuggingFaceTB/smoltalk2
- Tulu 3 preferences: https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture
