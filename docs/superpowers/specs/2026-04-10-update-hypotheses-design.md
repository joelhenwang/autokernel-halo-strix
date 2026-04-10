# Update Scientist Hypotheses with Verified External Kernels

**Date:** 2026-04-10
**Status:** Design approved, pending implementation
**Workstream:** B (of A/B/C/D optimization roadmap)
**Depends on:** Workstream A (need final tok/s numbers from 15-min runs)

## Problem

The 29 architecture plans in `mad_llm_scientist/plans/` were written before we verified external kernel libraries. They reference our old baselines (10.4K tok/s AMADEUS, sequential scan warnings) but don't know about:

- causal-conv1d (10x conv speedup — every GatedConv is now near-free)
- mamba-ssm scan (5.6x — AMADEUS scan bottleneck eliminated)
- hybrid_attention (8.9% faster than SDPA — attention layers now viable for training)
- FLA ops (GLA 1.28ms, Retention 0.77ms, HGRN 0.40ms — new recurrence options)
- PLE + MatFormer composable modules (available for any architecture)

## What to Update

### 1. Scientist CLAUDE.md — already partially updated
The "External Libraries" table and attention guidance are current. Add:
- Revised MFU estimates based on Workstream A results
- Updated training baselines table with new tok/s from A

### 2. COOKBOOK.md — add new section
Add "§2: External Kernel Integration" covering:
- How to use causal-conv1d in any GatedConv (try/except pattern)
- How to use mamba-ssm scan in any SSM architecture
- How to use hybrid_attention for training attention layers
- FLA ops as drop-in recurrence alternatives

### 3. Plans that need updating (prioritized)

**High priority (architectures that directly benefit):**

| Plan | Update needed |
|------|--------------|
| AMADEUS.md | Add actual results with mamba-ssm + causal-conv1d |
| TEMPEST.md | Add causal-conv1d results |
| PROMETHEUS.md | Add hybrid_attention results, revise throughput estimates |
| VIRTUOSO.md | Already superseded — add reference to PLE+MatFormer modules |
| CAVEMAN-LFM.md | causal-conv1d speeds up conv path, mamba-ssm speeds up recurrence |
| PARALLEL-CAVEMAN.md | Same as CAVEMAN |

**Medium priority (architectures that gain new options):**

| Plan | Update needed |
|------|--------------|
| SPECTRAL-HYDRA.md | FLA HGRN (0.40ms) as alternative to custom element-wise recurrence |
| HARMONIC-DREAMER.md | FLA DeltaNet as alternative to DHO |
| DEEP-NARROW-ORACLE.md | causal-conv1d + updated throughput estimates |
| DUAL-CORTEX.md | hybrid_attention for the slow path |
| BURST-ORACLE.md | flash_attn forward for speculative decode verification |

**Low priority (strategy plans, not architecture-specific):**

| Plan | Update needed |
|------|--------------|
| PHOENIX.md | Add mamba-ssm + causal-conv1d to optimization roadmap |
| COLOSSEUM.md | Updated baselines |
| EVAL-FRAMEWORK.md | Add external kernel benchmarks to evaluation suite |

### 4. Each updated plan gets

A new "External Kernel Integration" subsection in "Hardware Optimization Notes":
```markdown
### External Kernel Integration
- **GatedConv:** causal-conv1d (10x) — automatically used if installed
- **SSM scan:** mamba-ssm selective_scan_fn (5.6x) — replaces chunked/HIP scan
- **Attention:** hybrid_flash_sdpa_attention (8.9% vs SDPA) — best for training
- **Expected throughput:** X tok/s (updated from Workstream A results)
```

## Implementation

A single script or manual pass through each plan file. For each:
1. Add the "External Kernel Integration" subsection
2. Update throughput estimates based on Workstream A results
3. Note which FLA ops could replace custom recurrence (where applicable)

## Files to Modify

| File | Change |
|------|--------|
| `mad_llm_scientist/CLAUDE.md` | Update baselines after Workstream A |
| `mad_llm_scientist/COOKBOOK.md` | Add §2: External Kernel Integration |
| `mad_llm_scientist/plans/*.md` (15+ files) | Add external kernel notes |

## Verification

Spot-check 3 plans to ensure the numbers are consistent with Workstream A results and the guidance is actionable for the scientist agent.
