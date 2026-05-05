# AutoKernel Codebase Debloating & Refactoring Guide
> **Date:** 2026-05-04 | **Analyst:** AdaL (DeepSeek V4 Pro) | **Status:** FINAL
> **Cross-reference:** docs/adal/bloat_analysis.md (12 findings)

## Executive Summary

This codebase is a sophisticated, research-heavy GPU kernel optimization system that has outgrown its
original structure. The core engine is solid but bloat comes from documentation sprawl, copy-pasted
model boilerplate, and monolithic files that force agents to burn thousands of context tokens.

**Goal: Restructure so agents spend fewer tokens on orientation and more on the actual task.**

| What | Current | Target |
|------|---------|--------|
| Agent pre-work reading burden | ~1,500 lines | ~250 lines |
| RMSNorm/SwiGLU definition sites | 9+ files | 1 file |
| Monolithic files > 600 lines | 4 files | 0 files |
| Documentation files with overlapping content | 7 files | 3 files (clear ownership) |

## Part 1: Structural Assessment

### Short Answer: Yes, but surgically, not wholesale.

The codebase has a clear evolutionary structure from its research origins. It grew organically as
hypotheses were tested. The structures that made sense at 3 models and 5 kernel types cause friction
at 13+ models and 40+ kernels.

### What's Healthy (Don't Touch)

| Area | Why It Works |
|------|-------------|
| `kernel.py` (141L) | Single target for agent loop - exactly right size |
| `halo_training/` (8 files, max 438L) | Well-modularized by concern |
| `knowledge/` (30 files, ~6,100L) | Clean domain separation + dual indexes |
| `kernels/hip/` boilerplate | Consistent KERNEL_TYPE + BACKEND + HIP_SRC template |
| `bench.py`, `reference.py`, `prepare.py` | Immutable harnesses - correct as standalone |
| `STATUS.md` | Single source of truth for training state |

### What Needs Refactoring (Priority-Ordered)

| Area | Problem | Severity |
|------|---------|----------|
| Root-level agent files | program.md (893L), CLAUDE.md, README.md, KNOWLEDGE_GRAPH all overlap | HIGH |
| `models/` | Monolithic files, boilerplate x9, tight cross-model coupling (73 imports) | HIGH |
| `kernels/hip/_torch_ops.py` | 677L monolith for all autograd | MEDIUM |
| `scripts/` | 73 files, overlapping functionality, obsolete not marked | MEDIUM |
| `docs/` vs `knowledge/` vs `mad_llm_scientist/` | Three doc hierarchies, no cross-pointers | MEDIUM |
| `experiments/` | Stale MI300X content | LOW |

### The Coupling Challenge
`tyr_halo.py` imports from 6 other model files. This is good architecture (composition) but means
extracting shared components must preserve import paths via re-exports. Strategy: move -> re-export
from original location -> deprecate gradually.

## Part 2: The 12 Fixes - Complete Implementation Guide

### FIX 1: Split `program.md` - The Agent Instruction Monolith (893L -> ~350L total)

**Impact:** HIGH | **Effort:** 1-2 hours | **Risk:** Low | **Token savings:** ~700 lines/session

Current: program.md contains Phase A/B/C workflow, full HIP C++ playbook (6 tiers, 200L), kernel
tricks for 9 types (150L), decision framework, error handling, LLaMA 7B example run with dialogue,
workspace layout, constraints.

Target Architecture:
- `program.md` (~120L): Core workflow + keep/revert/move-on table + constraints + pointers
- `knowledge/kernels/hip_optimization_playbook.md` (~200L): HIP Tier 1-6 + Anti-Patterns + RDNA 3.5
- `knowledge/training/error_handling_protocol.md` (~80L): Error handling + decision framework
- `docs/tutorial/06_example_llama7b_run.md` (~80L): Example run walkthrough

Steps:
1. Create 3 new files extracting content from program.md
2. Rewrite program.md to ~120L with pointers to new files
3. Update CLAUDE.md reference: "Read program.md for workflow. See knowledge/ for playbooks."

### FIX 2: Single-Source-of-Truth Documentation Architecture

**Impact:** HIGH | **Effort:** 1 hour | **Risk:** Low | **Token savings:** ~450-500 lines/session

The Overlap Matrix (7 files share hardware specs, commands, decision protocol, directory maps):

| Topic | README(332L) | CLAUDE(144L) | KNOWLEDGE(116L) | STATUS(111L) | mad_llm CLAUDE(290L) |
|-------|--------------|--------------|-----------------|--------------|---------------------|
| Hardware specs | Yes | Yes | - | - | Yes |
| Training commands | Yes | Yes | - | - | - |
| Decision protocol | - | Yes | - | Yes | Yes |
| Directory map | Yes | Yes | Yes | - | - |
| Constraints/gotchas | - | Yes(40L!) | - | Yes | - |

Target Architecture:
- `CLAUDE.md` (~45L): Agent entry - STATUS.md first -> commands -> constraints -> decision tree -> pointers
- `CONSTRAINTS.md` (NEW, ~60L): All 15 gotchas deduplicated in checklist format
- `README.md` (stays): Human entry, add 3-line agent preamble at top
- `STATUS.md` (stays): Training SSoT
- `knowledge/INDEX.md` (stays): Full doc index

### FIX 3: Extract Shared Model Components - End RMSNorm x9

**Impact:** HIGH | **Effort:** 3-4 hours | **Risk:** Medium | **Token savings:** ~1,800 lines total

RMSNorm, SwiGLU, and RoPE utilities are redefined in 9+ model files. Fixing one bug = touching 9 files.
Some files explicitly "avoid name conflicts" by copy-pasting.

Strategy: Move + Re-Export + Migrate (4 phases)

Phase 1: Create `models/_components.py` with RMSNorm, SwiGLU, precompute_freqs_cis, apply_rotary_emb,
GatedConv (if shared 3+), and any component duplicated in 3+ files.

Phase 2: Re-export from original locations (zero-breakage)
- In `models/amadeus.py`: `from models._components import RMSNorm, SwiGLU, GatedConv`
- In `models/argus.py`: `from models._components import precompute_freqs_cis, apply_rotary_emb`

Phase 3: Update direct consumers - remove local definitions, add imports. Test each.

Phase 4: Deprecate re-exports (future, after all consumers migrated).

Migration priority: halo_prime.py -> chimera_halo.py -> remaining halo variants -> core providers last.

### FIX 4: Slim CLAUDE.md + Extract CONSTRAINTS.md

**Impact:** MEDIUM | **Effort:** 30 min | **Risk:** Low

Covered in FIX 2. The 40-line "Key Constraints" section is buried in a 144L file.
Extracting to CONSTRAINTS.md makes it: machine-parseable checklist, always read, single source.

### FIX 5: Refactor griffin_halo.py into Package (971L -> ~7 files)

**Impact:** MEDIUM | **Effort:** 2-3 hours | **Risk:** Medium | **Token savings:** ~771 lines

Implements 5-axis design sweep in one file. Target structure:
```
models/griffin/
  __init__.py          # Re-exports GriffinHalo
  config.py            # GriffinConfig (~50L)
  mixer.py             # Core mixer variants (~250L)
  aggregation.py       # Parcae depth aggregation (~150L)
  residuals.py         # ShortConv + residual pathways (~200L)
  attention.py         # Attention + value_bias (~150L)
  model.py             # Top-level assembler (~200L)
```
Steps: git mv -> extract components -> __init__.py -> update imports -> deprecated re-export shim.

### FIX 6: Split _torch_ops.py into Package (677L -> ~12 files)

**Impact:** MEDIUM | **Effort:** 1-2 hours | **Risk:** Medium | **Token savings:** ~597 lines

Monolithic autograd registry. Target: per-kernel files under `kernels/hip/_torch_ops/` with
`__init__.py` re-exporting all ops. Backward-compatible.

### FIX 7: Audit & Organize scripts/ (73 Files)

**Impact:** MEDIUM | **Effort:** 1 hour | **Risk:** Low

Problems: train_170m_smoke.py superseded by halo_training --smoke. Multiple run_wt103_* overlap
with run_progressive_*. No status headers on 23 shell scripts.

Steps: Add standardized headers (Purpose/Status/Hardware). Mark obsolete as DEPRECATED.
Create scripts/README.md cataloging by category. Optionally archive to scripts/_archived/.

### FIX 8: Cross-Pointers Between mad_llm_scientist/ and docs/superpowers/

**Impact:** MEDIUM | **Effort:** 15 min | **Risk:** None

Both describe same architectures (Chimera, Fenrir, Tyr, Vidar, Jormungandr).
plans/ = blueprints, specs/ = finalized decisions. No cross-pointers.

Steps: Add pointer in mad_llm_scientist/CLAUDE.md. Add "Derived from" header to each spec.

### FIX 9: TL;DRs for Large Knowledge Files

**Impact:** LOW | **Effort:** 20 min | **Risk:** None

Add ~10-line TL;DR at top of files >200L. Priority: alignment_implementation_details.md (586L),
paper_deep_dive_2026_05.md (565L), amd_rdna35_strix_halo.md.

### FIX 10: Archive Stale experiments/ MI300X Content

**Impact:** LOW | **Effort:** 10 min | **Risk:** None

Create experiments/README.md: "Archived MI300X (CDNA 3) experiments. Active target: Strix Halo (gfx1151)."

### FIX 11: .gitignore Hygiene

**Impact:** LOW | **Effort:** 5 min | **Risk:** None

Add to .gitignore: datasets/, docs/*.pdf, docs/*.zip, *.pyc.
Currently: datasets/, PDFs in docs/, tutorial.zip, 52 stray .pyc files all show as untracked.

### FIX 12: Mark Obsolete Shell Scripts - Covered in FIX 7.

## Part 3: Implementation Roadmap

### Phase 1: Zero-Risk Papercuts (Day 1, ~2 hours)
Additive-only - no code changes, no breakage.

| Order | Action | Fix | Effort |
|-------|--------|-----|--------|
| 1.1 | Add .gitignore entries | F11 | 5 min |
| 1.2 | Create experiments/README.md | F10 | 10 min |
| 1.3 | Create CONSTRAINTS.md | F4 | 20 min |
| 1.4 | Create hip_optimization_playbook.md | F1 | 30 min |
| 1.5 | Create error_handling_protocol.md | F1 | 15 min |
| 1.6 | Create example_llama7b_run.md | F1 | 15 min |
| 1.7 | Add TL;DRs to knowledge files >200L | F9 | 15 min |
| 1.8 | Add mad_llm <-> docs cross-pointers | F8 | 10 min |

### Phase 2: Agent Entry Point Rewrite (Day 1-2, ~2 hours)
Modify existing docs only - no code changes.

| Order | Action | Fix | Effort |
|-------|--------|-----|--------|
| 2.1 | Rewrite program.md to ~120L | F1 | 20 min |
| 2.2 | Rewrite CLAUDE.md to ~45L | F2 | 30 min |
| 2.3 | Add agent preamble to README.md | F2 | 5 min |
| 2.4 | Update mad_llm_scientist/CLAUDE.md | F2 | 15 min |
| 2.5 | Add headers + create scripts/README.md | F7 | 45 min |

**Checkpoint:** Agent burden: ~1,500L -> ~250L.

### Phase 3: Code Restructuring (Day 2-4, ~4-6 hours)
Move code with backward-compatible re-exports. Run tests between each step.

| Order | Action | Fix | Effort |
|-------|--------|-----|--------|
| 3.1 | Create models/_components.py | F3 Ph1 | 30 min |
| 3.2 | Add re-exports to amadeus.py, argus.py | F3 Ph2 | 15 min |
| 3.3 | Migrate 3 simplest models | F3 Ph3 | 45 min |
| 3.4 | Migrate remaining models | F3 Ph3 | 1 hr |
| 3.5 | Split _torch_ops.py into package | F6 | 1 hr |
| 3.6 | Refactor griffin_halo.py into package | F5 | 1.5 hrs |

### Phase 4: Cleanup (Day 4-5, ~1 hour)
Remove deprecated re-exports, archive obsolete scripts, update hardware references in tutorials.

## Part 4: Before/After Comparison

### Agent Context Burn per Session

| Activity | Before | After | Savings |
|----------|--------|-------|---------|
| Orientation (program.md) | 893L | 120L | 773L |
| Reference docs (CLAUDE+README+KNOWLEDGE) | 592L | 95L | 497L |
| Understand one model (tyr_halo) | 752L | 552L | 200L |
| Debug one kernel autograd | 677L | 80L | 597L |
| Understand Griffin architecture | 971L | 200L | 771L |
| **Total per comprehensive session** | **~3,885L** | **~1,047L** | **~2,838L saved** |

## Part 5: Principles Going Forward

### Documentation
1. **One fact, one place.** Hardware -> knowledge/hardware/. Nowhere else.
2. **Agent files point, don't contain.** CLAUDE.md, program.md -> pointers to reference docs.
3. **TL;DR required.** Any markdown > 200 lines must have TL;DR at top.
4. **Status headers required.** Every script: ACTIVE / DEPRECATED / EXPERIMENTAL.

### Code
5. **Defined thrice? Extract once.** Any class/function in 3+ files -> shared module.
6. **Re-export, don't break.** When moving shared code, add re-export from original location.
7. **Models are packages at 500L.** Any model exceeding 500L -> becomes a package.
8. **Autograd is per-kernel.** New kernel types get their own file, not appended to monolith.

### Agent Workflow
9. **CLAUDE.md is the entry point.** Every agent reads it first. Must fit on one screen (~50L).
10. **STATUS.md is always current.** Before any training launch or optimization, read STATUS.md.

## Part 6: What NOT to Change

| Area | Why Keep It |
|------|------------|
| `kernel.py` at 141L | Exactly right for optimization loop target |
| `halo_training/` structure | Well-modularized, clear separation |
| `knowledge/` YAML frontmatter | Enables tag-based indexing |
| `kernels/hip/` boilerplate | Consistent template = safer agent editing |
| `models/` cross-imports | Composition over inheritance is correct |
| `REPORT.md` at 1,079L | Canonical results - on-demand reads |
| `bench.py`, `reference.py`, `prepare.py` | Immutable harness - correct as standalone |

## Appendices

### A: New Files to Create (22 files)

| File | Purpose | Phase |
|------|---------|-------|
| `CONSTRAINTS.md` | Hardware/training constraints checklist | 1 |
| `knowledge/kernels/hip_optimization_playbook.md` | From program.md | 1 |
| `knowledge/training/error_handling_protocol.md` | From program.md | 1 |
| `docs/tutorial/06_example_llama7b_run.md` | From program.md | 1 |
| `experiments/README.md` | MI300X archival notice | 1 |
| `scripts/README.md` | Script catalog | 2 |
| `models/_components.py` | Shared RMSNorm, SwiGLU, RoPE | 3 |
| `models/griffin/__init__.py, config.py, mixer.py, aggregation.py, residuals.py, attention.py, model.py` | Griffin package (7 files) | 3 |
| `kernels/hip/_torch_ops/__init__.py, _base.py, <type>.py` | Per-kernel autograd (11 files) | 3 |

### B: Files to Modify (14 files)

| File | Change | Phase |
|------|--------|-------|
| `program.md` | 893L -> ~120L | 2 |
| `CLAUDE.md` | 144L -> ~45L | 2 |
| `README.md` | Add agent preamble | 2 |
| `mad_llm_scientist/CLAUDE.md` | Cross-reference root CLAUDE.md | 2 |
| `.gitignore` | Add datasets/, docs/*.pdf, *.pyc | 1 |
| `models/amadeus.py` | Re-exports from _components | 3 |
| `models/argus.py` | Re-exports from _components | 3 |
| `models/griffin_halo.py` | Deprecated re-export shim | 3 |
| 9+ model files | Replace local RMSNorm/SwiGLU with imports | 3 |

---
*Guide generated by AdaL (DeepSeek V4 Pro) for the autokernel-halo-strix codebase.*
