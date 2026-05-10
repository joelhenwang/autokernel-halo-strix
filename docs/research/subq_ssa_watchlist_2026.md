---
title: "SubQ / SSA — watch-list note (no technical paper published)"
domain: research
type: watchlist
status: watching
tags: [subq, ssa, subquadratic, long-context, sparse-attention, content-dependent-selection, watchlist]
sources:
  - "https://subq.ai/introducing-subq (2026-05-05)"
  - "https://subq.ai/how-ssa-makes-long-context-practical (2026-05-05)"
related:
  - ../../knowledge/training/zaya1_8b_findings_2026.md
---

# SubQ / SSA — Watch-list Note

## Status

**No technical paper published yet.** Only two marketing posts (2026-05-05)
and third-party-verified benchmark claims. We are **not** going to act on SSA
until a paper drops. This doc is a placeholder so we can reassess quickly
once it does.

## Claims, briefly

- **`SubQ 1M-Preview`**: "first fully subquadratic LLM." Closed beta; no
  weights, no paper.
- **SSA (Subquadratic Sparse Attention)**: content-dependent selection —
  for each query, pick a subset of positions, compute *exact* attention
  over that subset. Linear-in-context-length compute and memory.
- Claimed speedups vs FlashAttention-2 on B200 (FlashAttention-3 reportedly
  didn't outperform FA-2 on B200):

  | Context | Speedup | Attention FLOPs reduction |
  |--------:|:-------:|:-------------------------:|
  | 128K | 7.2× | 8× |
  | 256K | 13.2× | — |
  | 512K | 23.0× | — |
  | 1M | **52.2×** | **62.5×** |

- Benchmarks (third-party-verified per SubQ's claim):
  - **RULER @128K: 95.0** (Opus 4.6 94.8)
  - **MRCR v2: 65.9 production / 83 research** (Opus 4.7 32.2, GPT-5.4 36.6,
    Gemini 3.1 Pro 26.3, GPT-5.5 74.0, Opus 4.6 78.3)
  - **SWE-Bench Verified: 81.8** (Opus 4.6 80.8)
- Research model claimed to function to **12M tokens**.
- Their published critique of alternative sparse-attention schemes:
  fixed-pattern, SSMs, hybrids (dense layers stay load-bearing), **DeepSeek
  Sparse Attention** (their critique: the "lightning indexer" is itself
  O(n²), so "complexity is moved, not removed").

## Why we are not acting

1. **No paper, no code, no weights.** Architectural claim (content-dependent
   selection that stays O(n)) cannot be verified, reproduced, or ported.
   Selector parameterization and asymptotic cost are not described.
2. **Decode latency not reported.** Prefill speedups are reported, not
   per-token decode cost. A long-context inference stack also cares about
   decode.
3. **Research vs production gap unexplained.** MRCR v2 drops from 83
   (research) to 65.9 (production) with no accompanying explanation. This
   gap is material to any adoption decision.
4. **Scale mismatch with Odin.** Our current models run block=256–1024. SSA's
   wins are at 128K+ tokens. Even if everything they claim is true, we have
   no workload today that exercises the regime it wins in.
5. **gfx1151 vs B200 mismatch.** All speed numbers are B200 vs FA-2 on B200.
   Porting or re-measuring on gfx1151 is a separate non-trivial project.

## When to revisit

Trigger any one of:

- A technical paper is published (arXiv, model card, or conference).
- Independent reimplementation (e.g., HF Transformers, vLLM, FLA) merges SSA.
- We start a workload that needs > 8K usable context (not just accept an 8K+
  prompt, but reliably retrieve over it). Current target is still ≤ 1024.
- A close-enough alternative (e.g., Native Sparse Attention, DeepSeek-SA v2,
  Mixture-of-Depths × attention-selection) publishes an architecturally clean
  content-dependent-selection mechanism with real code.

## What we would want to read when the paper drops

- Exact form of the **selector**: how positions are scored, what is
  parameterized, what the asymptotic cost is (truly O(n)? O(n·k) with small k?
  O(n log n)?). This is the single most important missing detail.
- **Decode** (not just prefill) latency with a full KV cache.
- **Training infrastructure**: sequence-parallelism strategy, memory scaling,
  how they achieved "linear memory scaling across the training pipeline" at
  1M+.
- Ablations against SSM / DSA / fixed-pattern **on the same pretraining
  data** — so the win isn't "better architecture" confounded with "better
  data."
- Why research → production drops MRCR v2 by 17 points.
- How the RL stage is structured if they explicitly trained to *use* long
  context rather than merely accept it.

## Comparison to ZAYA1-8B in one paragraph

SubQ attacks **capability per token of context** via an attention exponent
change; ZAYA1 attacks **capability per active parameter** via small-MoE,
heavy RL, and a bounded-workspace TTC harness. The techniques are mostly
orthogonal — CCA (ZAYA1's attention compression, a constant-factor O(n²) win)
and SSA (exponent change) could in principle be combined. Transparency is
asymmetric: ZAYA1 has weights + paper + two companion arXiv papers; SubQ has
marketing copy and verified benchmark numbers. For our purposes this means
ZAYA1 has yielded shippable findings (see
`knowledge/training/zaya1_8b_findings_2026.md`) and SubQ has not. Yet.

## See also

- [zaya1_8b_findings_2026.md](../../knowledge/training/zaya1_8b_findings_2026.md) — applied-findings companion
- [broad-research-synthesis-2026-05-06.md](broad-research-synthesis-2026-05-06.md)
