---
title: "Answer-Preserving (AP) Trimming — recipe for CoT data at short context"
domain: training
type: recipe
status: active
tags: [ap-trimming, cot, reasoning-data, pretraining, sft, data-construction, zaya1]
paper: "ZAYA1-8B Technical Report §III-A; Akter et al. 2025 (reasoning-aware pretraining finding)"
related:
  - zaya1_8b_findings_2026.md
  - sft_pipeline.md
  - instruct_alignment_techniques_2025_2026.md
  - ../../docs/research/zaya1_8b_technical_report/zaya1_8b_technical_report.md
---

# Answer-Preserving (AP) Trimming

## The problem in one paragraph

Long chain-of-thought (CoT) traces from strong teachers routinely exceed 10K
tokens and tail to 30K+. Our pretraining and initial SFT block sizes are
**256 – 1024** (per the DDP sweep defaults). Every example over the context
budget has three options: **(i)** drop, losing reasoning signal; **(ii)**
naive truncation, which usually preserves the reasoning head and destroys the
answer — training the model on reasoning that never concludes; **(iii)**
preserve the answer and trim reasoning. AP trimming is option (iii), formalized
and re-applied at every context length the data passes through.

It matters because Akter et al. 2025 (cited in ZAYA1 report §III-A) show that
long-CoT exposure **in pretraining** gives gains that post-training cannot
recover, and ZAYA1 confirmed the effect at scale (86 % long-CoT in midtrain,
75 % in SFT). If we ever pull CoT data into dolma-mix phases or SFT, we need
a principled way to fit it into our context budget without breaking the
causal structure of reasoning → answer.

## Input assumptions

A **sample** is a multi-turn conversation. Each assistant turn has two parts:

- `<think> … </think>` reasoning block (optional but typical for reasoning data)
- Final-answer section (everything after `</think>` in that turn)

Messages are already in the chat template. Tokenization is deterministic and
lossless.

## Procedure

Given target context budget `C` (in tokens) and a sample `S`:

1. **Keep unchanged.** If `len_tokens(S) ≤ C`, return `S` as-is.
2. **Trim the tail of the last reasoning block.** Let `L = len_tokens(S) − C`.
   From the final assistant turn, truncate the **last `L` tokens of the
   reasoning block** (immediately before `</think>` or the answer section),
   leaving:
   - the full message history up to that point,
   - the start of the reasoning trace (head / planning / decomposition),
   - the full answer section of the final turn.
   The transition point becomes the "cut"; no artificial stop-token is
   inserted.
3. **Drop prior-turn reasoning.** If step 2 is insufficient (the retained
   reasoning block would go negative), drop the `<think>…</think>` blocks of
   **earlier** assistant turns one at a time (oldest first), preserving each
   turn's answer section. Re-apply step 2 after each drop.
4. **Drop sample.** If the answer sections alone exceed `C`, discard the
   sample.

**Never**: truncate inside the answer, truncate the head of a reasoning block,
truncate the system prompt, or truncate a user message.

## Why tail, not middle or head

- **Head** of a CoT: problem decomposition, search/planning, approach selection.
- **Tail** of a CoT: consolidation, final calculation, transition to answer.

Head-trimming destroys the causal chain that justifies the answer. Middle-
trimming creates a visible discontinuity inside a single logical chain.
Tail-trimming leaves a coherent partial chain: "here is how the model set up
the problem, and here is the final answer." The transition between head and
answer is distributionally artificial but preserves the causal skeleton.

ZAYA1 report explicitly checked pass-rate on reasoning benchmarks after
AP-trimmed pretraining+midtraining and **did not observe a truncation-specific
failure mode** in downstream evaluations.

## Stage-aware re-trimming

Re-run AP-trimming **offline per dataset per context length** as the pipeline
advances:

| Stage | Our block size | Re-trim against | Typical retention |
|-------|---------------:|----------------:|------------------:|
| Pretrain early | 256 / 512 | `C = 256` or `512` | Most aggressive; many steps 3/4 drops |
| Pretrain late / CPT | 512 / 1024 | `C = 1024` | Still aggressive |
| Midtrain (future) | 4K–8K | `C = 8192` | Moderate |
| SFT (future) | 16K–131K | `C = 131072` | Near-complete traces |

For dolma-mix CPT experiments at block=512, expect ~20–50 % of long-CoT
samples to hit steps 2/3, single-digit % to drop at step 4. Exact numbers
depend on the CoT source.

## Relation to other length-control methods

| Method | When applied | Modifies | Source |
|--------|-------------:|----------|--------|
| **AP trimming** (this doc) | **Offline, training-data construction** | Sample length + content | ZAYA1 §III-A |
| Answer-length filter | Offline, dataset selection | Sample inclusion (keep only long answers) | Akter et al. 2025 |
| Forced end-of-thinking | Online RL rollout | Inserts stop phrase when budget exceeded | Khatri et al. 2025 |
| Thinking-budget control | Inference | Caps generation at inference time | Yang et al. 2025 |
| Markovian Thinker chunking | Online (training + inference) | Chunks reasoning with bounded carry-forward | Aghajohari et al. 2025 |

AP-trimming is the **only** one that addresses the pretraining-context-
bottleneck case. The others either select data (length filter) or shape
generation (everything else). They are complementary.

## Port sketch for this repo

Target: a single utility callable from data-construction scripts (
`scripts/prepare_*.py`) and from any future SFT packer.

**File**: `halo_training/data/ap_trim.py`

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Turn:
    role: str                    # "system" | "user" | "assistant"
    reasoning: Optional[str]     # None if not an assistant turn or no <think>
    answer: str                  # user message OR assistant answer section

@dataclass
class TrimResult:
    sample: Optional[List[Turn]]   # None if dropped
    action: str                    # "keep" | "trim_tail" | "drop_prior_cot" | "drop"
    retained_tail_tokens: int      # 0 if action == "keep" or "drop"

def ap_trim(sample: List[Turn], budget_tokens: int,
            tokenizer, min_retained_reasoning: int = 64) -> TrimResult:
    """Apply AP-trimming to a single sample against a token budget.

    min_retained_reasoning: if tail-trim would leave fewer than this many
    reasoning tokens, escalate to step 3 (drop earlier CoT) or step 4.
    """
    ...
```

**Integration points** (not built yet — tag for the SFT pipeline work):

- `halo_training/data/streaming.py` — optional `ap_trim` preprocessing hook
- New CLI flag `--ap-trim` on `scripts/prepare_*.py` with `--ap-budget N`

**Unit tests to write alongside the module**:

1. Fits under budget → step 1, no-op.
2. Single-turn CoT overflow by K tokens → step 2 retains `len - K` reasoning
   tokens, answer intact.
3. Multi-turn, last turn can't fit even with full tail-trim → step 3 drops
   earlier `<think>`, re-runs step 2.
4. Answers alone exceed budget → step 4 drops sample.
5. Token-count round-trip: `len(tokenize(ap_trim(s))) ≤ budget` for all
   non-dropped outputs.
6. Edge case: `<think>` block is empty → tail-trim is a no-op, fall through
   to step 3.
7. Edge case: no `<think>` at all (non-reasoning data) → step 1 or step 4
   only; never step 2/3.

## When to actually ship this

Two triggers:

1. **Next CPT experiment that includes reasoning data**, even at a low mix
   (e.g., 5 % of dolma-mix is long-CoT). Without AP trimming we will either
   drop most of it (step 4) or train on answerless reasoning (naive truncate).
2. **First SFT stage** (currently P0 infra gap per STATUS.md). AP trimming is
   a prerequisite for BFD packing at any context shorter than the longest
   CoT in the source data.

If neither trigger is close, this doc is a reference-only note until it is.

## References

- ZAYA1-8B Technical Report §III-A — the source procedure and the
  empirical observation that tail-truncation does not produce visible
  downstream artifacts after pretraining+midtraining.
- Akter et al. 2025 (cited in ZAYA1 §III) — the motivation: reasoning data
  belongs in pretraining, and naive truncation is the reason it often doesn't
  help.
- Khatri et al. 2025 / Yang et al. 2025 — online length-control
  complements; different problem, same family.
- [zaya1_8b_findings_2026.md](zaya1_8b_findings_2026.md) — parent synthesis.
