# Hypothesis Throughput Ranking (Estimated)

**Date:** 2026-04-10
**Hardware:** AMD Strix Halo gfx1151 (Radeon 8060S, 40 CUs, 59.4 TFLOPS FP16, ~240 GB/s LPDDR5X)
**Pipeline:** 50 steps, batch=16, seq=256, 4096 tokens/step (204,800 tokens total)
**Formula:** `tok/s = (MFU × 59.4e12) / (6 × params)` — verified against AMADEUS (10.4K) and LlamaModel (43K)

---

## Optimization Pipeline (what "optimized" means)

| Optimization | Source | Impact |
|-------------|--------|--------|
| `torch.compile(mode="default")` | PyTorch Inductor | 1.5-3x MFU boost (fuses kernel launches, eliminates intermediates) |
| `autokernel.optimize(model, training=True)` | Our HIP kernels | RMSNorm 6.6x, SwiGLU 1.6x, RoPE 3.7x, cross_entropy 1.8x |
| `causal-conv1d` | External package | 10x vs nn.Conv1d for depthwise conv |
| `mamba-ssm` selective_scan | External package | 5.6x vs HIP kernel (0.32ms) |
| `hybrid_flash_sdpa_attention` | Our hybrid kernel | 8.9% faster than SDPA (3.50ms vs 3.84ms fwd+bwd) |
| FLA (Triton) | External package | HGRN 0.40ms, Retention 0.77ms, GLA 1.28ms |
| Batch=16, seq=256 | Pipeline tuning | L2 sweet spot (6 MB fits activations) |

---

## Ranking: Architecture Hypotheses by Estimated tok/s

### Tier 1: >15K tok/s (High Throughput)

| Rank | Hypothesis | Params | Type | Eager tok/s | Eager MFU | Optimized tok/s | Opt. MFU | Key Advantage |
|------|-----------|--------|------|-------------|-----------|-----------------|----------|---------------|
| **1** | **RESONANT-LOOP** | 59M unique / 168M eff. | Shared-block iterative | 10,000 | 17% | **22,000** | **37%** | L2 weight reuse on iterations 2-3; fewest unique params |
| **2** | **PROMETHEUS** | 216M | Griffin + 2 attention | 9,000 | 20% | **19,000** | **42%** | Smallest param count; hybrid_flash_sdpa for global context |
| **3** | **TEMPEST** | 245M | Pure Griffin element-wise | 8,000 | 20% | **18,000** | **45%** | All non-FFN ops element-wise; highest MFU potential |
| **4** | **VIRTUOSO** | 252M | TEMPEST + PLE + MatFormer | 7,500 | 19% | **16,000** | **41%** | PLE adds 0 FLOPs; MatFormer enables elastic inference |
| **5** | **SPECTRAL-HYDRA** | 245M | Multi-scale Griffin | 7,500 | 19% | **16,000** | **40%** | Heterogeneous decay spectrum; pure element-wise |

### Tier 2: 10-15K tok/s (Medium-High Throughput)

| Rank | Hypothesis | Params | Type | Eager tok/s | Eager MFU | Optimized tok/s | Opt. MFU | Key Advantage |
|------|-----------|--------|------|-------------|-----------|-----------------|----------|---------------|
| **6** | **TERNARY-REFLEX** | 228M | Dual-path ternary + Griffin | 7,000 | 16% | **14,000** | **32%** | Ternary path L2-resident; dual-path quality |
| **7** | **OBSIDIAN** | 224M | BitNet + Caveman | 7,000 | 16% | **14,000** | **32%** | BitNet 1.58 weights L2-friendly; graduated vocab |
| **8** | **AMADEUS** | 244M | Conv + Mamba-3 SISO | 6,400 | 16% | **13,000** | **32%** | **Verified baseline**; mamba-ssm scan 5.6x boost |
| **9** | **MAESTRO-PRIMA** | 242M | AMADEUS + conductor | 6,200 | 15% | **12,000** | **29%** | FiLM conductor only 137K params overhead |
| **10** | **MAESTRO-FORTE** | 242M | AMADEUS + residual gates | 6,200 | 15% | **12,000** | **29%** | Residual gates for conductor stability |
| **11** | **DUAL-CORTEX** | 231M | System 1/2 entropy-gated | 6,000 | 14% | **12,000** | **28%** | Entropy-adaptive fast/slow path routing |
| **12** | **MAESTRO-VIVACE** | 242M | AMADEUS + GRU conductor | 6,000 | 15% | **11,500** | **28%** | GRU conductor evolution adds small overhead |
| **13** | **MAESTRO-FINALE** | 242M | AMADEUS + full conductor | 6,000 | 15% | **11,500** | **28%** | 4-signal conductor (most expressive, most overhead) |

### Tier 3: 8-10K tok/s (Medium Throughput)

| Rank | Hypothesis | Params | Type | Eager tok/s | Eager MFU | Optimized tok/s | Opt. MFU | Key Advantage |
|------|-----------|--------|------|-------------|-----------|-----------------|----------|---------------|
| **14** | **META-ENGRAM** | 235M | Parallel hybrid + knowledge | 5,500 | 13% | **10,500** | **25%** | Meta tokens + Engram N-grams for retrieval |
| **15** | **GENIUS-CAVEMAN** | 234M | Dual-path info-routed | 5,500 | 13% | **10,500** | **25%** | Reflex+Genius paths with mHC routing |
| **16** | **CAVEMAN-LFM** | 248M | LFM2-validated hybrid | 5,500 | 14% | **10,000** | **25%** | Closest to LFM2.5 design; proven architecture class |
| **17** | **PARALLEL-CAVEMAN** | 251M | Caveman + parallel hybrid | 5,000 | 13% | **9,500** | **24%** | Parallel hybrid adds throughput vs serial Caveman |
| **18** | **BURST-ORACLE** | 251M | Caveman + MTP + Engram | 5,000 | 13% | **9,500** | **24%** | MTP burst decoding for inference speed |

### Tier 4: <8K tok/s (Lower Throughput, Higher Complexity)

| Rank | Hypothesis | Params | Type | Eager tok/s | Eager MFU | Optimized tok/s | Opt. MFU | Key Advantage |
|------|-----------|--------|------|-------------|-----------|-----------------|----------|---------------|
| **19** | **HARMONIC-DREAMER** | 241M | DHO + dynamic scratchpad | 4,500 | 11% | **9,000** | **22%** | 2nd-order oscillatory recurrence; novel dynamics |
| **20** | **ARCHON** | 251M | Quality-maximized hybrid | 4,500 | 11% | **8,500** | **22%** | mHC 4-branch + Engram + MTP; max quality |
| **21** | **CHIMERA-ENGRAM** | 243M | mHC + Mamba-3 + MoE | 4,500 | 11% | **8,500** | **21%** | Most components; complex routing + MoE scatter |
| **22** | **DEEP-NARROW-ORACLE** | 233M | 48L × d=512 | 4,000 | 9% | **8,000** | **19%** | Serial depth limits parallelism; 48 sequential launches |

---

## Reference Baselines (Verified)

| Architecture | Params | Config | tok/s | MFU | Status |
|-------------|--------|--------|-------|-----|--------|
| LlamaModel (transformer) | 124.7M | compile + autokernel | **43,000** | **54%** | Verified |
| LlamaModel (transformer) | 124.7M | eager | 14,500 | 17% | Verified |
| AMADEUS (SSM hybrid) | 243.8M | autokernel + compile + HIP scan | **10,400** | **26%** | Verified |
| AMADEUS (SSM hybrid) | 243.8M | eager, chunked scan | 6,400 | 16% | Verified |
| AMADEUS (SSM hybrid) | 243.8M | sequential scan | 1,300 | 4% | Verified |

### New Verified Baselines (2026-04-10 PLE Ablation)

| Architecture | Params | Config | tok/s | MFU | Status |
|-------------|--------|--------|-------|-----|--------|
| Tempest (Griffin) | 244.5M | compile + autokernel | **8,152** | **20.1%** | Verified |
| Tempest + MatFormer | 244.5M | compile + autokernel | **8,166** | **20.2%** | Verified |
| Tempest + PLE(a) | 246.6M | compile + autokernel | 7,936 | 19.7% | Verified |

### New Verified Baselines (2026-04-11 Hypothesis Build-Out)

| Architecture | Params | Config | tok/s | MFU | Status |
|-------------|--------|--------|-------|-----|--------|
| **RESONANT-LOOP** | 58.8M unique | eager | 10,056 | 6.0% | Verified |
| **RESONANT-LOOP** | 58.8M unique | autokernel | **13,344** | **7.9%** | Verified — **highest throughput** |
| **RESONANT-LOOP** | 58.8M unique | autokernel + compile | 13,076 | 7.8% | Compile no effect (graph breaks from 16-iter loop) |
| **MAESTRO-PRIMA** | 243.9M | eager | 6,648 | 16.4% | Verified |
| **MAESTRO-PRIMA** | 243.9M | autokernel | **8,848** | **21.8%** | Verified |
| **SPECTRAL-HYDRA** | 244.5M | eager | 5,976 | 14.8% | Verified |
| **SPECTRAL-HYDRA** | 244.5M | autokernel + compile | **10,204** | **25.2%** | Verified |
| **OBSIDIAN** | 169.3M | eager | **8,669** | **14.8%** | Verified — autokernel breaks loss |
| **DUAL-CORTEX v1** | 262.7M | eager | 5,837 | 15.5% | v1: exceeds 250M budget |
| **DUAL-CORTEX v2** | 154.3M | eager | **9,267** | **14.4%** | v2: reduced dims, fits budget |

**Note:** DualCortex v1 param count (262.7M) exceeded the 250M constraint. v2 reduced to 154.3M (d_fast=256, d_slow=1024, 8L slow). The slow path (d=1280, 10L) is larger than estimated. OBSIDIAN's autokernel optimization causes loss divergence — investigate before using. MAESTRO-PRIMA compile fails due to HIP scan backward incompatibility with torch.compile tracing.

**Note:** Estimated optimized throughput for Tempest was 18,000 tok/s (45% MFU). Actual measured is 8,152 tok/s (20.1% MFU). The formula's compile boost factor (2.25x for element-wise) was too optimistic — actual boost is ~1.28x (from 6,363 eager to 8,152 with autokernel). This means ALL estimates in the ranking above should be treated as upper bounds. Divide estimated "optimized tok/s" by ~2.2 for more realistic projections.

---

## Non-Architecture Plans (excluded from ranking)

| Plan | Type | Notes |
|------|------|-------|
| SELF-CURRICULUM | Training strategy | Applies to any architecture; ~1% overhead |
| LEVIATHAN | Distillation infrastructure | Train 1B teacher → distill to 250M student |
| LOTTERY-FORGE | Architecture search | 330M → pruned 250M; training strategy |
| MEGATRAIN-HALO | Porting guide | Unified memory training infrastructure |
| PHOENIX | Optimization roadmap | AMADEUS optimization path (not new arch) |
| COLOSSEUM | Training infrastructure | CPU optimizer, activation checkpointing |
| EVAL-FRAMEWORK | Evaluation suite | Architecture comparison tooling |

---

## Optimization Impact Analysis

### Which optimizations matter most per architecture type

| Architecture Type | torch.compile | autokernel | causal-conv1d | mamba-ssm | hybrid_attn | FLA |
|-------------------|--------------|------------|---------------|-----------|-------------|-----|
| Pure Griffin (TEMPEST, SPECTRAL-HYDRA) | **+100%** | +20% | **+30%** | N/A | N/A | Optional |
| Griffin + attn (PROMETHEUS) | **+100%** | +20% | **+30%** | N/A | **+9%** | Optional |
| Conv + SSM (AMADEUS, MAESTRO) | +60% | +20% | **+30%** | **+25%** | N/A | N/A |
| Dual-path (TERNARY, DUAL-CORTEX) | +80% | +20% | +20% | Varies | N/A | Optional |
| Complex hybrid (ARCHON, CHIMERA) | +50% | +15% | +15% | +15% | N/A | Optional |
| Shared-block (RESONANT-LOOP) | **+120%** | +20% | N/A | N/A | N/A | N/A |
| Deep narrow (DEEP-NARROW-ORACLE) | +60% | +15% | **+25%** | N/A | N/A | N/A |

### Tokens processed in a 50-step benchmark

| Rank | Hypothesis | Eager (50 steps) | Optimized (50 steps) | Improvement |
|------|-----------|------------------|----------------------|-------------|
| 1 | RESONANT-LOOP | 20.5s / 204K tok | 9.3s / 204K tok | **2.20x** |
| 2 | PROMETHEUS | 22.8s / 204K tok | 10.8s / 204K tok | **2.11x** |
| 3 | TEMPEST | 25.6s / 204K tok | 11.4s / 204K tok | **2.25x** |
| 4 | VIRTUOSO | 27.3s / 204K tok | 12.8s / 204K tok | **2.13x** |
| 5 | SPECTRAL-HYDRA | 27.3s / 204K tok | 12.8s / 204K tok | **2.13x** |
| 8 | AMADEUS (verified) | 32.0s / 204K tok | 15.8s / 204K tok | **2.03x** |
| 22 | DEEP-NARROW-ORACLE | 51.2s / 204K tok | 25.6s / 204K tok | **2.00x** |

---

## Backward Pass Optimization Impact (2026-04-10)

Measured backward kernel speedups on gfx1151. See `knowledge/backward_pass_optimization_results.md` for full details.

### Measured Backward Kernel Speedups

| Kernel | Speedup | Applicable To |
|--------|---------|---------------|
| rmsnorm_backward | **12.51x** | All architectures (32 calls/step for 16L) |
| silu_gate_mul_backward | **10.74x** | All with SwiGLU FFN (16 calls/step) |
| rotary_embedding_backward | **4.67x** | Models with attention (LLaMA, PROMETHEUS) |
| fused_residual_rmsnorm_backward | **~13x** (est.) | All architectures (16 calls/step) |
| selective_scan_backward | **21.15x** | SSM models (AMADEUS, MAESTRO variants) |

### Reranked Throughput with Backward Optimizations

Formula: `new_tok_s = old_tok_s / (1 - bwd_frac × (1 - 1/bwd_speedup))`
where bwd_frac = 0.53 (backward portion of step) and bwd_speedup is the weighted average across all backward ops.

| Rank | Hypothesis | Previous tok/s | Bwd Speedup (est.) | **New tok/s** | Change |
|------|-----------|---------------|-------------------|-------------|--------|
| **1** | **AMADEUS** | 13,000 | **~5x** (scan dominates) | **~26,000** | **+100%** |
| **2** | **RESONANT-LOOP** | 22,000 | ~2x (norm/activation) | **~33,000** | +50% |
| **3** | **PROMETHEUS** | 19,000 | ~2.5x (norm/rotary/attn) | **~31,000** | +63% |
| **4** | **TEMPEST** | 18,000 | ~2x (norm/activation) | **~27,000** | +50% |
| **5** | **VIRTUOSO** | 16,000 | ~2x (norm/activation) | **~24,000** | +50% |
| **6** | **SPECTRAL-HYDRA** | 16,000 | ~2x (norm/activation) | **~24,000** | +50% |
| **7** | **MAESTRO-PRIMA** | 12,000 | ~4x (scan + norm) | **~22,000** | +83% |
| **8** | **TERNARY-REFLEX** | 14,000 | ~2x (norm/activation) | **~21,000** | +50% |
| **9** | **OBSIDIAN** | 14,000 | ~2x (norm/activation) | **~21,000** | +50% |
| **10** | **MAESTRO-FORTE** | 12,000 | ~4x (scan + norm) | **~22,000** | +83% |

**Key shift:** AMADEUS jumps from rank 8 to rank 1-2 tier because the selective scan backward (21x) is so dominant in its backward pass. All MAESTRO variants similarly benefit from the scan speedup.

**Caveat:** These are extrapolated from isolated kernel benchmarks. Combined training speedup will be less than kernel speedup due to matmul backward (unchanged), optimizer step, and memory allocation overhead. The actual Tempest baseline shows estimates are ~2.2x too optimistic. Apply the same correction factor to these projections.

### Realistic Projections (÷2.2 correction) → Now Verified

| Hypothesis | Estimated tok/s | **Measured tok/s** | Config | Status |
|-----------|----------------|-------------------|--------|--------|
| RESONANT-LOOP | 22,000 | **13,344** | autokernel | ✓ #1 throughput |
| AMADEUS | 13,000 | **8,742** | autokernel + HIP bwd scan | ✓ 1.66x from scan bwd |
| SPECTRAL-HYDRA | 16,000 | **10,204** | autokernel + compile | ✓ |
| MAESTRO-PRIMA | 12,000 | **8,848** | autokernel | ✓ |
| OBSIDIAN | 14,000 | **8,669** | eager only | ✓ autokernel breaks loss |
| TEMPEST | 18,000 | **8,152** | autokernel + compile | ✓ baseline |
| PROMETHEUS | 19,000 | **11,087** | autokernel + compile | ✓ Fastest 240M+ model |
| DUAL-CORTEX v2 | 12,000 | **9,267** | eager | ✓ v2 reduced to 154.3M |

**Correction factor**: estimated ÷ measured ≈ 1.5-2.2x. Estimates are consistently optimistic.

---

## Key Insights

1. **Parameter count is the #1 throughput lever.** RESONANT-LOOP (168M effective) and PROMETHEUS (216M) lead because fewer params = fewer bytes/step. At our memory bandwidth (240 GB/s), every MB of weights costs ~4μs per pass.

2. **Element-wise ops are free.** Griffin recurrence adds zero measurable cost on gfx1151 — hidden behind memory latency. This is why TEMPEST/SPECTRAL-HYDRA achieve high MFU despite complex recurrence.

3. **torch.compile is the single biggest optimization.** Roughly 2x throughput for element-wise architectures (fuses launches, eliminates intermediates). SSM hybrid models benefit less (~1.6x) because scan patterns are harder to fuse.

4. **Complexity costs throughput, not quality.** ARCHON/CHIMERA-ENGRAM have the most sophisticated components (mHC, Engram, MoE, MTP) but rank lowest in throughput. The question is whether quality gains justify 2x slower training.

5. **The quality-throughput frontier.** The best designs balance both:
   - PROMETHEUS: fast (19K) + global context (2 attention layers)
   - TEMPEST: fastest pure recurrence (18K) but no retrieval capability
   - AMADEUS: proven quality (12.18 BPB) at moderate speed (13K with mamba-ssm)

6. **Within 45-min budget (at batch=16, seq=256):**

   | Hypothesis | Optimized tok/s | Tokens in 45 min | BabyLM epochs (~16M) |
   |-----------|----------------|------------------|---------------------|
   | RESONANT-LOOP | 22,000 | 59.4M | 3.7 |
   | PROMETHEUS | 19,000 | 51.3M | 3.2 |
   | TEMPEST | 18,000 | 48.6M | 3.0 |
   | AMADEUS | 13,000 | 35.1M | 2.2 |
   | DEEP-NARROW-ORACLE | 8,000 | 21.6M | 1.4 |

---

## Methodology Notes

- **Eager MFU** estimated from architecture similarity to verified AMADEUS (16%) and LlamaModel (17%) baselines
- **Optimized MFU** estimated by applying compile boost factor: 3x for transformers, 2.25x for pure element-wise, 1.6x for SSM hybrid, 1.8x for dual-path, 2.2x for shared-block
- **All estimates assume** batch=16, seq=256 (L2 sweet spot), warm compile cache (no JIT overhead in 50 steps)
- **Not estimated:** inference tok/s, int4 quantized tok/s (would change ranking significantly — L2-resident architectures like RESONANT-LOOP and TERNARY-REFLEX would dominate)
- **Disclaimer:** These are engineering estimates, not measurements. Actual throughput depends on implementation quality, kernel fusion patterns, and compile graph breaks. Verify with smoke tests.

---

## ACTUAL MEASURED RANKING (2026-04-11)

All measurements on gfx1151, batch=8, seq=256, median of 30 steps after 5 warmup.

### By Throughput (Best Config)

| Rank | Architecture | Params | Best Config | tok/s | MFU | Peak Mem | Notes |
|------|-------------|--------|-------------|-------|-----|----------|-------|
| **1** | **LlamaModel** | 124.7M | compile + autokernel | **43,000** | **54%** | ~4 GB | Transformer baseline, not SSM |
| **2** | **RESONANT-LOOP** | 58.8M | autokernel | **13,344** | 7.9% | 3.1 GB | Lowest params, highest throughput |
| **3** | **PROMETHEUS** | 241.3M | autokernel + compile | **11,087** | **27.0%** | 5.1 GB | **Fastest 240M+ model** |
| **4** | **SPECTRAL-HYDRA** | 244.5M | autokernel + compile | **10,204** | 25.2% | 5.2 GB | Multi-scale Griffin, compile helps |
| **5** | **DUAL-CORTEX v2** | 154.3M | eager | **9,267** | 14.4% | 5.6 GB | Reduced dims, fits budget |
| **6** | **MAESTRO-PRIMA** | 243.9M | autokernel | **8,848** | 21.8% | 5.8 GB | AMADEUS + Conductor |
| **7** | **AMADEUS** | 243.8M | autokernel + HIP bwd | **8,742** | 21.5% | ~7 GB | After scan backward integration |
| **8** | **OBSIDIAN** | 169.3M | eager | **8,669** | 14.8% | 6.0 GB | BitNet + Caveman, autokernel breaks loss |
| **9** | **TEMPEST** | 244.5M | autokernel + compile | **8,152** | 20.1% | ~5 GB | Pure Griffin baseline |

### Sorted by MFU (Efficiency)

| Rank | Architecture | MFU | Notes |
|------|-------------|-----|-------|
| 1 | LlamaModel | 54% | compile + autokernel |
| 2 | SPECTRAL-HYDRA | 25.2% | autokernel + compile |
| 3 | MAESTRO-PRIMA | 21.8% | autokernel |
| 4 | AMADEUS | 21.5% | autokernel + HIP bwd |
| 5 | TEMPEST | 20.1% | autokernel + compile |
| 6 | DUAL-CORTEX | 15.5% | eager |
| 7 | OBSIDIAN | 14.8% | eager |
| 8 | RESONANT-LOOP | 7.9% | Low MFU because 58.8M unique params → formula underweights |

### Key Findings (2026-04-11)

1. **RESONANT-LOOP is the throughput champion** at 13.3K tok/s. Only 58.8M unique params means minimal memory traffic per pass. The iterative shared block design pays off as hypothesized. torch.compile doesn't help (graph breaks from the 16-iteration loop).

2. **PROMETHEUS is the fastest 240M+ model** at 11.1K tok/s. The 14 Griffin + 2 attention hybrid design benefits from both autokernel (RMSNorm, SwiGLU, RoPE) and compile (element-wise fusion). Estimated 19K, measured 11.1K — 1.71x overestimate.

3. **SPECTRAL-HYDRA benefits most from compile** — 5,976 → 10,204 (1.71x). The pure element-wise architecture fuses well. This validates the "maximum MFU" design thesis.

4. **MAESTRO-PRIMA and AMADEUS are neck-and-neck** — the Conductor adds only 137K params overhead. The question is whether Conductor provides quality improvement (needs longer training to evaluate).

5. **OBSIDIAN is fast for its complexity** at 8.7K eager. The BitNet reflex path is lightweight. But autokernel.optimize breaks training stability — the ternary quantization doesn't play well with HIP kernel replacements. Needs investigation.

6. **DUAL-CORTEX v2** (154.3M) now fits budget after reducing d_slow from 1280→1024, d_fast from 320→256, and n_slow_layers from 10→8. Eager 9.3K tok/s.

7. **Estimated ÷ Measured correction factors:**
   - RESONANT-LOOP: 22K/13.3K = 1.65x
   - SPECTRAL-HYDRA: 16K/10.2K = 1.57x
   - MAESTRO-PRIMA: 12K/8.8K = 1.36x
   - OBSIDIAN: 14K/8.7K = 1.61x
   - DUAL-CORTEX: 12K/5.8K = 2.07x
   - Average: ~1.65x overestimate. Dual-path architectures are most overestimated.

### Tokens in 45-min Budget (Measured)

| Architecture | tok/s | Tokens/45min | BabyLM Epochs |
|-------------|-------|-------------|---------------|
| RESONANT-LOOP | 13,344 | 36.0M | 2.25 |
| PROMETHEUS | 11,087 | 29.9M | 1.87 |
| SPECTRAL-HYDRA | 10,204 | 27.6M | 1.72 |
| DUAL-CORTEX v2 | 9,267 | 25.0M | 1.56 |
| MAESTRO-PRIMA | 8,848 | 23.9M | 1.49 |
| AMADEUS | 8,742 | 23.6M | 1.48 |
| OBSIDIAN | 8,669 | 23.4M | 1.46 |
| TEMPEST | 8,152 | 22.0M | 1.38 |

---

## ACTUAL TRAINING RESULTS (2026-04-12)

BabyLM 16.5M tokens, 2 epochs, ~170M params, batch=16, seq=256, autokernel.optimize()

### By Quality (Val Loss, lower = better)

| Rank | Architecture | Params | Val Loss | Train Loss | tok/s | steps/s | Time |
|------|-------------|--------|----------|------------|-------|---------|------|
| **1** | **Amadeus** | 157.7M | **2.9015** | 2.7510 | 13,203 | 3.2 | 38 min |
| **2** | **MaestroPrima** | 157.8M | **2.9017** | 2.7393 | 12,896 | 3.1 | 39 min |
| 3 | Tempest | 176.8M | 2.9796 | 2.7688 | 12,952 | 3.2 | 39 min |
| 4 | Virtuoso | 180.8M | 2.9936 | 2.8189 | 11,165 | 2.7 | 45 min |
| 5 | Prometheus | 174.3M | 2.9951 | 2.8379 | 13,066 | 3.2 | 39 min |
| 6 | SpectralHydra | 176.8M | 3.1940 | 3.1182 | 10,323 | 2.5 | 49 min |
| 7 | ResonantLoop | 50.7M | 3.4176 | 3.2680 | 15,907 | 3.9 | 32 min |
| 8 | DualCortex | 125.2M | 5.4352 | 5.4433 | 32,426 | 7.9 | FAILED |
| 9 | Obsidian | 124.0M | 5.7074 | 5.6637 | 34,115 | 8.3 | FAILED |

### Conductor A/B Test (Amadeus vs MaestroPrima)

Head-to-head with identical data split, seed, hyperparameters:
- Amadeus final val: 2.9038
- MaestroPrima final val: 2.9023
- Difference: **-0.0015 (negligible)**
- Conductor consistently ~0.002 better in epoch 2, but statistically indistinguishable

### Dual-Path Eager Diagnostic (2026-04-12)

| Model | Eager (no AK) | With Autokernel | Diagnosis |
|-------|--------------|-----------------|-----------|
| DualCortex d=256 | **val=3.1909** (works) | val=5.4352 (failed) | autokernel is the problem |
| Obsidian d=256 | **val=3.4924** (ep1, works) | val=5.7074 (failed) | autokernel is the problem |

Both architectures learn normally in eager mode. DualCortex eager (3.19) is competitive with SpectralHydra.

### Key Training Insights

1. **SSM hybrids (Amadeus) win quality** — Mamba-3 SISO + gated conv + FiLM learns best at 16M tokens
2. **Conductor is a non-factor** at 16M tokens — 0.15% improvement, within noise
3. **Pure Griffin (Tempest) is the #3** — competitive quality, simpler architecture
4. **Dual-path models work in eager, fail with autokernel** — HIP kernels at d=256 cause divergence, not architecture
5. **ResonantLoop is throughput-optimal** (15.9K tok/s) but quality-limited at 50.7M params
6. **SpectralHydra underperforms** — heterogeneous decay spectrum initialization needs tuning
7. **torch.compile effectiveness is the #1 throughput lever** — 3.2x for LlamaModel vs 1.7x for Tempest. Closing this gap requires custom ops for scan + block-level fusion patterns
