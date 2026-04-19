# SWE + Tool-Use Specialization for ARGUS-PRIME

## Overview

Specializing ARGUS-PRIME (168M params) as a **tool-using agent** for software
engineering tasks and general assistance. The model learns to call tools
(web search, calculator, code execution) and generate SEARCH/REPLACE code patches.

**Date:** 2026-04-18 – ongoing
**Model:** ARGUS-PRIME 168M (ShortConv/GQA + In-Place TTT + Engram + Momentum)
**Hardware:** Strix Halo Machine B (gfx1151, 128 GB, 15.7K tok/s with compile+autokernel)

## Training Pipeline

```
Dolma 10B CC pretrain (step_36000, loss 3.81)
    ↓
Code CPT v1: 103M Python tokens (step_3872, loss 6.65)
    ↓
Code CPT v2: 770M Python tokens, 3 epochs (RUNNING)
    ↓
Mixed SFT: 50K examples (tool-use + code repair + localize)
    ↓
DPO: 24K preference pairs for context grounding
    ↓
Agent runtime: scripts/agent.py with tool execution loop
```

## Data Sources

### 1. SWE Code Repair — Nemotron-Cascade-2

| Field | Value |
|-------|-------|
| Source | `nvidia/Nemotron-Cascade-2-SFT-Data` SWE agentless subset |
| Raw size | 23.9 GB JSONL, ~391K rows |
| Generator | DeepSeek-V3.2 / R1 |
| Format | 3-turn: system → issue+code → `<think>` + SEARCH/REPLACE |
| Preprocessing | `scripts/prepare_swe_data.py` — streams, extracts sub-tasks |
| Extraction rate | ~44% code repair, ~50% bug explain, ~40% localize |

**Sub-tasks extracted:**
- `swe_code_repair.jsonl` — 158K examples, avg 1094 tokens
- `swe_bug_explain.jsonl` — 19K examples, avg 1350 tokens
- `swe_localize.jsonl` — 155K examples, avg 247 tokens

### 2. Python Code — codeparrot/github-code-clean

| Field | Value |
|-------|-------|
| Source | `codeparrot/github-code-clean` Python-all subset |
| Size | 10 parquet files, 1.8 GB compressed → 4.4 GB text |
| Tokens | ~1.06B |
| Format | Raw Python source, EOS-separated .txt files |
| Preprocessing | `scripts/prepare_code_cpt.py` |

### 3. Tool-Use — Locutusque/function-calling-chatml

| Field | Value |
|-------|-------|
| Source | `Locutusque/function-calling-chatml` (reformatted Glaive) |
| Examples | 110,814 |
| Format | ChatML with system/human/gpt/function-call/function-response roles |
| Tools | Calculator, weather, news, currency, passwords, movies, etc. |
| Avg tokens | 620 |
| Preprocessing | `scripts/prepare_tool_data.py` — converts to our ChatML with `<tool_call>` markers |

### 4. DPO Preference Pairs — Synthetic

| Field | Value |
|-------|-------|
| Examples | 24,000 pairs |
| Type 1 | Tool name grounding: chosen=exact name from system prompt, rejected=memorized name (8K) |
| Type 2 | Value extraction: chosen=correct numbers from user msg, rejected=hallucinated numbers (8K) |
| Type 3 | Response grounding: chosen=quotes tool output, rejected=ignores tool output (8K) |
| Generator | `scripts/generate_dpo_data.py` — algorithmic, no reward model needed |

## Results

### SWE Repair v1 (from OpenHermes base, 2 epochs)
- Loss: 8.66 → 4.67
- Learned SEARCH/REPLACE format perfectly
- Hallucinated all code content — no grounding

### SWE Repair v2 (5 more epochs)
- Loss: 4.67 → 1.15
- Correct file identification (pex/version.py for version bumps)
- Correct variable targeting (__version__ for release issues)
- Still hallucinated exact values (wrong version numbers)

### Code CPT v1 (103M Python tokens)
- Loss: 12.4 → 6.65 (46% reduction)
- 380M tokens trained in 7.4 hours

### Mixed SFT (tool-use + SWE repair + localize)
- Loss: 10.8 → 5.46
- 142M tokens in 2.8 hours
- **Tool calling works:** model emits valid `<tool_call>` JSON
- **Tool selection works:** picks correct tool category
- **Tool refusal works:** refuses when tools don't match

### Agent Testing (scripts/agent.py)
- Full tool execution loop works end-to-end
- Calculator: model calls `calculate_tip` → aliased to calculator → result returned → model responds
- Time: model calls `get_current_time` → aliased to get_time → correct date returned
- Search: model attempts search but argument extraction weak
- **Core weakness:** model doesn't ground on context (memorized tool names, hallucinated numbers, ignored tool output)

## The Grounding Problem

At 168M params, the model learns **patterns** but not **copying**:

| Skill | Status | Explanation |
|-------|--------|-------------|
| Format generation | ✅ Works | Structural patterns are easy to memorize |
| Tool routing | ✅ Works | "math question → calculator" is a pattern |
| Tool refusal | ✅ Works | "no matching tool → refuse" is a pattern |
| Name copying | ❌ Fails | Can't read "web_search" from system prompt and reproduce it |
| Value extraction | ❌ Fails | Can't read "85 dollars" from user msg and put "85" in args |
| Result grounding | ❌ Fails | Can't read "11:12:48" from tool output and say "11:12" |

**Solutions being pursued:**
1. **More pretraining** (Code CPT v2, 770M tokens) — strengthens attention patterns for code
2. **DPO grounding** (24K preference pairs) — directly rewards correct copying, penalizes hallucination

## Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|im_start\|>` | 50257 | ChatML turn start |
| `<\|im_end\|>` | 50258 | ChatML turn end |
| `<\|pad\|>` | 50259 | Padding |
| `<tool_call>` | 50260 | Tool call start |
| `</tool_call>` | 50261 | Tool call end |

Use `build_tokenizer(phase="domain-sft")` for vocab 50262 (all tokens).
Use `--tool-use` flag in CLI to enable.

## Agent Runtime

`scripts/agent.py` implements a full tool-using agent loop:

1. User query → model generates response
2. If `<tool_call>` detected → parse JSON → execute tool
3. Feed tool result back as `tool` role message
4. Model generates final response grounded in tool output
5. Repeat up to 3 rounds

Built-in tools: `web_search` (DuckDuckGo), `calculator` (safe eval), `python_exec` (subprocess), `get_time`

Fuzzy alias mapping handles model's tendency to use memorized names (`calculate_tip` → `calculator`).

```bash
# Interactive mode
python scripts/agent.py --checkpoint checkpoints/argus_prime_mixed_sft/step_1446.pt

# Single query
python scripts/agent.py --checkpoint ... --query "What is 15% of 85?"
```

## Files

| File | Purpose |
|------|---------|
| `scripts/prepare_swe_data.py` | SWE data preprocessing (streams 23.9 GB) |
| `scripts/prepare_tool_data.py` | Tool-use data preparation from Glaive |
| `scripts/prepare_code_cpt.py` | Python code download + convert to training format |
| `scripts/generate_dpo_data.py` | DPO preference pair generation (synthetic) |
| `scripts/agent.py` | Agent runtime with tool execution loop |
| `scripts/run_swe_training.sh` | Training launcher (multi-stage) |
| `halo_training/dpo.py` | DPO loss, dataset, trainer |
| `halo_training/sft_data.py` | SFT dataset with tool token support |
| `halo_training/cli.py` | CLI with --phase dpo, --tool-use flags |
| `knowledge/training/swe_specialization.md` | This document |

## Anti-patterns

1. **Don't use TRL** — HF ecosystem coupling isn't worth it for a custom nn.Module. DPO loss is 15 lines.
2. **Don't use agentic SWE data** — 60K token conversations are useless at 2048 context
3. **Don't skip code CPT** — model can't ground code patches without Python token representations
4. **Don't train more than 1B tokens on 168M model** — diminishing returns, capacity-limited
5. **Don't expect 168M to memorize facts** — use tools for factual lookups, model is a routing layer
6. **Don't expect exact context copying without DPO** — SFT alone teaches format, DPO teaches grounding

## Evaluation Plan

### Agent evaluation
- Tool name accuracy: does model use exact name from system prompt?
- Argument extraction: does model extract correct values from user message?
- Response grounding: does model quote tool output in its response?

### SWE evaluation
- SEARCH/REPLACE format correctness (already ~100%)
- File path identification accuracy
- Patch content relevance (does SEARCH text appear in the input code?)
