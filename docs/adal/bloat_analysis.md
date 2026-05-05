# AutoKernel Codebase Bloat Analysis
> **Date:** 2026-05-04 | **Analyst:** AdaL (DeepSeek V4 Pro) | **Status:** COMPLETE

## TL;DR
Systematic deep dive into codebase bloat affecting coding agent reasoning quality.
**3 HIGH-severity, 5 MEDIUM, and 2 LOW findings identified.**
Top culprit: `program.md` at 893 lines — the main agent instruction file is a monolith.

---

## 1. High-Level Metrics

| Metric | Count |
|--------|-------|
| Total files in repo | 1,359 |
| Python files (.py) | 213 |
| Markdown files (.md) | 167 (54 in docs/ alone, ~20,843 lines) |
| Shell scripts (.sh) | 34 |
| Knowledge base .md files | 30 (~6,100 lines) |
| HIP kernels (kernels/hip/) | 40 .py files |
| Model definitions (models/) | 13+ .py files (200-971 lines each) |
| mad_llm_scientist/ | 67 files (plans, reviews, guides) |
| scripts/ | 73 files (50 Python, 23 Shell) |
| docs/tutorial/ | 15+ part curriculum |

---

## 2. Key Files for Agent Reasoning

| File | Lines | Role | Impact |
|------|-------|------|--------|
| `program.md` | 893 | Full agent instruction file | **CRITICAL** |
| `CLAUDE.md` | 144 | Claude Code guidance | HIGH |
| `REPORT.md` | 1,079 | Canonical kernel optimization results | MEDIUM |
| `kernel.py` | 141 | Target modification file | MEDIUM (lean) |
| `KNOWLEDGE_GRAPH.md` | 116 | Documentation index | MEDIUM |
| `models/griffin_halo.py` | 971 | Largest model definition | HIGH |
| `kernels/hip/_torch_ops.py` | 677 | PyTorch autograd integration | HIGH |
| `kernels/hip/_compile.py` | 369 | HIP compilation engine | MEDIUM |
| `halo_training/trainer.py` | 438 | Training engine (Mode A/B) | HIGH |
| `halo_training/cli.py` | 267 | CLI entry point | MEDIUM |
| `mad_llm_scientist/CLAUDE.md` | 290 | Researcher agent instructions | HIGH |
| `scripts/analyze_checkpoint.py` | 1,038 | Largest script file | LOW |
| `docs/tutorial/10_math_foundations.md` | 1,372 | Largest doc file | LOW |
| `kernelbench/program_kb.md` | -- | KernelBench agent instructions | MEDIUM |

---

## 3. Complete Findings Log

### F1: `program.md` is 893 lines — 🔴 HIGH
- **Issue:** The main agent instruction file is a monolith. Every coding agent (Claude Code, Codex, AdaL, Cursor) must parse this to understand the optimization loop.
- **Specific bloat:**
  - Full HIP C++ optimization playbook (6 tiers) inline
  - Detailed kernel-specific tricks for all 9 kernel types
  - Redundant error handling sections (crashes, timeouts, OOM)
  - Full example run (LLaMA 7B) with simulated dialogue
  - Decision framework with 4 sub-sections
  - Workspace layout, supported kernel types catalog
- **Agent impact:** 893 lines of context before any work begins. Each experiment iteration may re-reference it.
- **Suggestion:** Split into:
  1. **Core workflow** (< 100 lines): Phase A/B/C summary + decision rules
  2. **HIP optimization cookbook** → `knowledge/kernels/hip_optimization_playbook.md`
  3. **Error handling protocol** → `knowledge/training/error_handling.md`
  4. **Example runs** → `docs/tutorial/`

### F2: Massive Documentation Overlap — 🔴 HIGH
- **Issue:** README.md (332L), CLAUDE.md (144L), KNOWLEDGE_GRAPH.md (116L), STATUS.md (111L), REPORT.md (1,079L), knowledge/INDEX.md (50L), mad_llm_scientist/CLAUDE.md (290L) all serve overlapping navigation/overview roles.
- **Agent impact:** Agents typically read CLAUDE.md + README.md + KNOWLEDGE_GRAPH.md, getting overlapping content 3x. That's ~592 lines of context burned on redundant orientation.
- **Specific overlaps:**
  - Hardware specs: README, CLAUDE.md, knowledge/hardware/, mad_llm_scientist/CLAUDE.md
  - Training commands: README, CLAUDE.md, halo_training/README.md
  - Decision protocol: CLAUDE.md, STATUS.md, mad_llm_scientist/CLAUDE.md
  - Directory maps: README, CLAUDE.md, KNOWLEDGE_GRAPH.md
- **Suggestion:** Single source of truth pattern — CLAUDE.md becomes pure quick reference (< 50 lines), everything else delegates to knowledge/INDEX.md

### F3: Model Boilerplate Redundancy — 🔴 HIGH
- **Issue:** `RMSNorm` and `SwiGLU` are redefined in at least 9 different model files instead of being imported from a shared utility.
- **Files affected:** griffin_halo.py (971L), tyr_halo.py (752L), jormungandr_halo.py (556L), amadeus.py (459L), halo_prime.py (443L), chimera_halo.py, fenrir_halo.py, vidar_halo.py, baldr_halo.py
- **Why it hurts agents:**
  1. Agents must read more code to understand each model
  2. Redundant code paths make consistency bugs likely
  3. A fix to RMSNorm requires touching 9 files
  4. ~200 lines × 9 files = ~1,800 lines of copy-pasted code
- **Note:** Some files explicitly mention they "avoid name conflicts" by copying instead of importing — this is a code smell that should be fixed with explicit re-exports.
- **Suggestion:** Extract shared building blocks into `models/_components.py`:
  - `RMSNorm`, `SwiGLU`, RoPE `precompute_freqs_cis`
  - Common attention patterns, residual wrappers
  - Each model re-exports, e.g.: `from models._components import RMSNorm`

### F4: `CLAUDE.md` Mixed Concerns — 🟡 MEDIUM
- **Issue:** CLAUDE.md mixes quick-reference commands, hardware specs, training philosophy, ~40 lines of constraints, and directory maps into 144 lines.
- **Breakdown:** Commands (~30L), Architecture constraints (~15L), Training philosophy (~20L), Key constraints (~40L of bullet gotchas), Decision protocol (~10L), Directory map (~15L), graphify (~10L)
- **Agent impact:** The constraint list is valuable but buried. Agents scanning CLAUDE.md may miss the 15 specific gotchas (bf16 crashes, momentum blocks, MFMA absence, etc.).
- **Suggestion:** Extract constraints to `CONSTRAINTS.md` (machine-readable, machine-only). CLAUDE.md becomes: quick commands + "read STATUS.md first" + 5-line decision tree + pointer to constraints.

### F5: `models/griffin_halo.py` at 971 Lines — 🟡 MEDIUM
- **Issue:** Single model file nearly 1000 lines — largest Python file in the repo.
- **Root cause:** Implements 5-axis design sweep (core mixer, depth aggregation, residuals, value bias, activation checkpointing) all in one file.
- **Agent impact:** Agents implementing kernel optimizations or training this model must hold 971 lines of model logic in context.
- **Suggestion:** Refactor into `models/griffin/` package:
  - `mixer.py`, `aggregation.py`, `residuals.py`, `config.py`, `__init__.py`

### F6: `kernels/hip/_torch_ops.py` at 677 Lines — 🟡 MEDIUM
- **Issue:** Autograd logic for every kernel type lives in one monolithic file.
- **Agent impact:** Understanding the autograd for one kernel requires reading code for all kernels.
- **Suggestion:** Split per-kernel-type:
  - `_torch_ops/matmul.py`, `_torch_ops/rmsnorm.py`, `_torch_ops/attention.py`, etc.
  - `_torch_ops/__init__.py` re-exports all

### F8: `scripts/analyze_checkpoint.py` at 1,038 Lines — 🟡 MEDIUM
- **Issue:** Largest Python script — hierarchical heatmap exploration + weight/grad analysis.
- **Agent impact:** Single-purpose analysis tool, unlikely to be read frequently by coding agents. Low impact.
- **Suggestion:** Consider splitting analysis sub-commands into separate scripts.

### F9: Conceptual Overlap: `mad_llm_scientist/` vs `docs/superpowers/` — 🟡 MEDIUM
- **Issue:** Both directories contain architectural design docs. `mad_llm_scientist/plans/` are the creative blueprints; `docs/superpowers/specs/` are the finalized engineering decisions.
- **Agent impact:** Agents may read both, getting redundant context about the same architectures.
- **Suggestion:** Add a one-line pointer in each directory: "For final specs, see docs/superpowers/specs/" and vice versa.

### F7: Knowledge Base at ~6,100 Lines — 🟢 LOW
- **Issue:** 30 markdown files across 4 subdirectories, well-organized with consistent YAML frontmatter.
- **Good:** Properly indexed via KNOWLEDGE_GRAPH.md and knowledge/INDEX.md. Domain separation (hardware/kernels/training/architectures) is clean.
- **Concern:** Largest files at 580+ lines — agents may skip reading them entirely.
- **Suggestion:** Add ~10-line TL;DR summaries at the top of large knowledge files.

### F10: `docs/tutorial/` at ~20,843 Lines — 🟢 LOW
- **Issue:** Massive 15+ part curriculum, largest single file is 1,372 lines.
- **Agent impact:** Low — tutorials are reference materials, not required reading for agents. Only read on-demand.
- **Concern:** Some early parts reference 4060 Ti (project inception hardware), creating confusion with current Strix Halo focus.
- **Suggestion:** Add hardware-target badges to tutorial parts. Archive or update 4060 Ti references.

### F11: `experiments/` MI300X Content is Stale — 🟢 LOW
- **Issue:** The `experiments/` directory is almost entirely MI300X-focused (CDNA 3), which is no longer the active hardware target.
- **Agent impact:** Agents exploring the repo may waste time reading obsolete experiments.
- **Suggestion:** Add a README.md to `experiments/` noting "Archived MI300X experiments — active target is Strix Halo (gfx1151)."

### F12: Shell Script Proliferation — 🟡 MEDIUM
- **Issue:** 23 shell scripts in `scripts/`, many overlapping. `train_170m_smoke.py` superseded by `halo_training --smoke`. Multiple `run_wt103_*` scripts overlap with newer `run_progressive_*` series.
- **Agent impact:** Agents may struggle to identify the "correct" script to run.
- **Suggestion:** Audit and retire obsolete scripts. Add header comments to each .sh indicating its purpose and whether it's still maintained.

---

## 4. Architecture Diagram: Agent Reading Flow

```
                   ┌─────────────┐
                   │  program.md │  893 lines — ALWAYS read first
                   │  (CRITICAL) │
                   └──────┬──────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │CLAUDE.md │   │README.md │   │KNOWLEDGE_GR..│
    │  144L    │   │  332L    │   │    116L      │
    └────┬─────┘   └────┬─────┘   └──────┬───────┘
         │              │                │
         │   ~592 lines of OVERLAPPING content   │
         │         (F2 — wasted context)         │
         └──────────────┼────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────────┐
    │  models/ │  │kernels/  │  │halo_training/│
    │  13+ .py │  │hip/ 40py │  │   8 .py      │
    │RMSNormx9│  │_torch_ops │  │   438L max   │
    │  (F3)   │  │  677L(F6)│  │              │
    └──────────┘  └──────────┘  └──────────────┘
          │             │             │
          └─────────────┼─────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────────┐
    │knowledge/│  │  docs/   │  │mad_llm_sci/  │
    │ 6100L/30 │  │20843L/54 │  │  67 files    │
    │   (F7)   │  │  (F10)   │  │    (F9)      │
    └──────────┘  └──────────┘  └──────────────┘
```

---

## 5. Bloat Severity Summary

| # | Finding | Severity | Impact | Quick Fix? |
|---|---------|----------|--------|------------|
| F1 | program.md: agent instruction monolith (893L) | 🔴 HIGH | Every agent session | ✅ Split into 4 files |
| F2 | Documentation overlap (4+ files, ~600L) | 🔴 HIGH | Every agent session | ✅ Single-source-of-truth |
| F3 | Model boilerplate duplication (RMSNorm ×9) | 🔴 HIGH | Model comprehension | ⚠️ Extract _components.py |
| F5 | griffin_halo.py: 971-line monolith | 🟡 MEDIUM | Model work | ⚠️ Refactor to package |
| F4 | CLAUDE.md: mixed concerns (144L) | 🟡 MEDIUM | Quick reference | ✅ Extract CONSTRAINTS.md |
| F6 | _torch_ops.py: monolithic autograd (677L) | 🟡 MEDIUM | Kernel dev | ⚠️ Split per-kernel-type |
| F8 | analyze_checkpoint.py (1,038L) | 🟡 MEDIUM | Rarely read | 🔵 Low priority |
| F9 | mad_llm_scientist vs docs/superpowers overlap | 🟡 MEDIUM | Architecture work | ✅ Add cross-pointers |
| F12 | Shell script proliferation (23 .sh) | 🟡 MEDIUM | Training ops | ✅ Audit + header comments |
| F7 | Knowledge base size (6,100L) | 🟢 LOW | On-demand reads | 🔵 Add TL;DRs |
| F10 | docs/tutorial/ (20,843L) | 🟢 LOW | On-demand reads | 🔵 Add hardware badges |
| F11 | experiments/ MI300X stale content | 🟢 LOW | Wasted reads | ✅ Add README note |

**Legend:** ✅ Easy fix (< 1 hour) | ⚠️ Moderate effort (1-4 hours) | 🔵 Low priority

---

## 6. Recommended Action Plan (Priority Order)

### Immediate Wins (same-day, low risk)
1. **Split `program.md`** into core workflow (100L) + separate cookbooks → saves ~700 lines per agent session
2. **Slim `CLAUDE.md`** to quick-reference only (< 50L), extract constraints to `CONSTRAINTS.md`
3. **Add README.md note** to `experiments/` about MI300X being archived

### This Week (moderate effort)
4. **Extract `models/_components.py`** with shared RMSNorm, SwiGLU, RoPE — then refactor 9+ models
5. **Add TL;DRs** to top of knowledge files > 200 lines
6. **Audit shell scripts** — retire superseded ones, add header comments

### When Refactoring Models
7. **Split `griffin_halo.py`** into package
8. **Split `_torch_ops.py`** per kernel type

---

## 7. What's Working Well

- **`kernel.py`** at 141 lines is perfectly lean
- **`halo_training/`** is well-modularized (8 focused .py files, largest 438L)
- **`knowledge/`** YAML frontmatter + consistent domain separation
- **KNOWLEDGE_GRAPH.md** as a tag-based index is effective
- **STATUS.md** as single source of truth for training state
- **`kernels/hip/`** consistent boilerplate pattern (KERNEL_TYPE + BACKEND + HIP_SRC)
