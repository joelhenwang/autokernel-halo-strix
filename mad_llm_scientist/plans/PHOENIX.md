# PHOENIX

**AMADEUS Reborn: Close the MFU Gap Through Optimization**

*The phoenix doesn't create a new body. It rises from the ashes of the old one, reborn and stronger.*

## Hypothesis

AMADEUS achieves 16% MFU (6.4K tok/s). The architecture is sound (loss 42→15.5 in 560 steps). The problem is THROUGHPUT, not quality. This plan identifies and fixes the bottlenecks to reach 30-50% MFU (15-20K tok/s) WITHOUT changing the architecture.

This is NOT a new architecture. It's an **optimization roadmap** for `models/amadeus.py`.

---

## Current Baseline (Measured)

| Metric | Value |
|--------|-------|
| Model | AMADEUS 243.8M params |
| tok/s | 6,400 (eager, no compile) |
| MFU | 15.9% |
| Memory | 12.7 GB |
| Loss @ 560 steps | 15.5 |
| Scan | Chunked (chunk_size=64) |
| File | `models/amadeus.py` |

**Target:** 15-20K tok/s (2.5-3x speedup). This would make AMADEUS competitive with transformer throughput.

---

## Optimization Roadmap (Priority Order)

### Tier 1: Low-Hanging Fruit (expected 1.5-2x total)

**1.1 torch.compile**
```python
model = torch.compile(model, mode="default")
# NOT mode="reduce-overhead" (conflicts with autokernel)
# Expected: 1.3-1.8x (measured on transformer: 14.5K → 24K without autokernel)
```

**1.2 autokernel.optimize**
```python
model = autokernel.optimize(model, training=True)
# Replaces: RMSNorm (3.3x), fused_residual_add_rmsnorm (6.6x), silu_gate_mul (1.6x)
# Expected: +10-20% on top of compile
```

**1.3 Fused optimizer**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, fused=True)
# Reduces optimizer overhead. Works on ROCm.
```

**Expected after Tier 1:** 6.4K × 1.5 = **~10K tok/s** (conservative)

### Tier 2: Profile-Guided Optimization (expected additional 1.5-2x)

**2.1 Profile the actual bottleneck**
```bash
# Use rocprof to find where time is spent
rocprof --stats python -m halo_training --model models/amadeus.py --class-name Amadeus --smoke
```

Key questions to answer:
- What % of time is in the Mamba-3 scan?
- What % is in FFN matmuls (rocBLAS)?
- What % is in the gated conv?
- What % is overhead (kernel launch, Python dispatch)?

**2.2 Optimize chunked scan (if scan is bottleneck)**

Current: chunk_size=64, T/64=16 serial steps for seq_len=1024.

Options:
- Increase chunk_size to 128 or 256 (fewer serial steps, more memory)
- Fuse scan operations into a single HIP kernel (avoid intermediate tensors)
- Use fp16 scan (currently fp32) if precision permits

```python
# Current: 16 serial steps at chunk_size=64
# Option: 8 serial steps at chunk_size=128
# Option: 4 serial steps at chunk_size=256 (if memory allows)
```

**2.3 Reduce dstate (if scan is bottleneck)**

Current: dstate=64 (Mamba-3's inner state dimension).
Option: dstate=32 (halves scan compute, may lose quality).

```python
# config change only:
dstate: int = 32  # was 64
# This halves the work inside the scan: O(T × d_mamba × dstate)
```

Quality impact: unknown. Must ablate.

**2.4 FiLM removal (if FiLM isn't helping)**

FiLM adds 1.1M params and a mean-pool + projection per forward pass. At 16% MFU, any overhead matters. If FiLM doesn't improve loss, remove it.

```python
# Easy toggle:
film_start: int = 999  # disable FiLM (no layer reaches 999)
```

### Tier 3: Architectural Simplification (if needed)

**3.1 Replace Mamba-3 with Griffin**

If profiling shows the Mamba-3 scan is the dominant bottleneck (>50% wall-clock), consider replacing Mamba-3 with Griffin recurrence. This would make PHOENIX converge toward TEMPEST.

**3.2 Add residual momentum**

If Griffin replacement happens, add residual momentum to compensate for the quality loss of simpler recurrence.

---

## Expected Results

| Optimization | Est. tok/s | Multiplier | Cumulative |
|---|---|---|---|
| Baseline (current) | 6,400 | 1.0x | 6,400 |
| + torch.compile | ~9,000 | 1.4x | 9,000 |
| + autokernel | ~10,500 | 1.15x | 10,500 |
| + fused optimizer | ~11,000 | 1.05x | 11,000 |
| + scan optimization | ~14,000 | 1.3x | 14,000 |
| + reduce dstate 64→32 | ~17,000 | 1.2x | 17,000 |

**Realistic target: 12-17K tok/s.** At 15K tok/s, 45 min = 40.5M tokens (2.5 BabyLM epochs).

---

## Hardware Optimization Notes

### What's Already Optimized
- Chunked scan (5x vs sequential) — DONE
- RMSNorm, SwiGLU follow autokernel patterns — DONE
- Weight tying — DONE

### What's NOT Yet Optimized
- No torch.compile applied
- No autokernel.optimize applied
- Scan chunk_size may not be optimal
- dstate may be larger than needed
- FiLM overhead not measured

### Profiling Targets

| Metric | Tool | What to Look For |
|--------|------|-----------------|
| Per-op time | `rocprof --stats` | Which ops dominate wall-clock |
| Kernel launch overhead | `rocprof --hip-trace` | If Python dispatch is >10% |
| Memory bandwidth | `rocprof` counters | Are FFN matmuls bandwidth-bound? |
| Scan time | Manual `time.time()` | % of forward pass in scan |
| Compile speedup | Before/after `torch.compile` | Actual multiplier |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| torch.compile breaks custom scan | MEDIUM | Wrap scan in `torch.compiler.disable`. Compile everything ELSE. |
| Reducing dstate hurts quality | MEDIUM | Ablate: compare loss at dstate=64 vs 32 at equal steps. |
| Profiling shows fundamental bandwidth limit | HIGH | If FFN matmuls dominate (not scan), optimization ceiling is ~20K. Architecture change needed (→ TEMPEST). |

## Success Criteria

1. tok/s > 12K (2x current baseline)
2. MFU > 30%
3. Loss at equal tokens ≈ current AMADEUS loss (no quality regression from optimization)
4. Profiling data shared for future architecture decisions
5. Clear identification of THE bottleneck (scan? FFN? overhead?)

## Implementation Roadmap

1. Apply torch.compile + autokernel. Measure tok/s.
2. Profile with rocprof. Identify bottleneck.
3. If scan: optimize chunk_size, try dstate reduction.
4. If overhead: check Python dispatch, kernel launch count.
5. If FFN: this is the fundamental limit — consider TEMPEST.
6. Ablate FiLM: measure loss with/without.
7. Document ALL findings for future architecture decisions.

### External Kernel Integration (verified 2026-04-10)

- **mamba-ssm:** selective_scan_fn (5.6x, 0.32ms) — add to optimization roadmap for any Mamba-based architecture
- **causal-conv1d:** 10x conv speedup — add to optimization roadmap for all GatedConv architectures
- **hybrid_attention:** 8.9% faster than SDPA for training — add to optimization roadmap for attention layers
