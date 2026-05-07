---
title: "v3 Speculative Directions — 18-idea research catalogue"
domain: architectures
type: research-menu
status: speculative
tags: [v3, speculative, research-directions, moe, parcae, looped-models, novel-architecture, training-dynamics, inference, data-efficiency]
related:
  - looped_moe_design_2026.md
  - ../training/zaya1_8b_findings_2026.md
  - cookbook.md
  - hypothesis_buildout_results.md
  - parcae_stable_looped_models.md
  - small_lm_arch_interventions_2026.md
  - ../../docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md
---

# v3 Speculative Directions — Research Catalogue

## Status

**Speculative research menu, not implementation commitment.** This document captures 18 distinct architectural and training ideas that are:

1. Not already documented in our knowledge base (checked against cookbook, hypothesis-buildout, small-LM-interventions playbook, paper deep-dive, Parcae-stable-looped-models, and looped-moe-design)
2. Compatible in principle with gfx1151 hardware (fp16, no MFMA, 64 KB LDS, wave32, compile-friendly)
3. Evaluated **standalone** — not assumed to compose with FrankenMoE-Loop v2 unless explicitly noted

Each idea is specified at spec-quality depth: motivation, mechanism, novelty contrast, hardware fit, risk register, implementability tier, trigger condition, and standalone evaluation protocol. The **compatibility matrix** (Section 6) documents which pairs compose, conflict, or are orthogonal. Sections 7 and 8 rank ideas by "if-we-could-only-do-one" and sketch a "dream stack" composition.

**When to touch this doc:**
- After FrankenMoE-Loop v2 L9 lands and we want to pick a v3 experiment
- When a new paper triggers a "we should revisit that idea" moment
- When scorecard results suggest a gap the catalogue might fill

**When NOT to touch this doc:**
- During Sprint-N execution (commit to the sprint spec, not to speculation)
- Before v1/v2 validate (we need data to re-rank tiers, not more speculation)

## Reading guide

Each idea follows the uniform template:

```
### Idx. Name — one-line pitch

Motivation & theoretical grounding
Mechanism sketch
Why this is novel vs our knowledge base
Hardware fit on gfx1151
Risk register
Implementability (tier + effort)
Trigger condition
Standalone evaluation protocol
```

**Tier definitions:**

- **★★★★★** — implementable in ≤ 1 week, very clear theoretical basis, low failure risk. "Just do it when trigger fires."
- **★★★★** — 1-2 weeks, well-understood mechanism, standard literature precedent.
- **★★★** — 2-3 weeks, speculative but tractable, requires some infrastructure work.
- **★★** — 3-6 weeks, high novelty, significant unknowns, research-grade experiment.
- **★** — 6+ weeks or fundamentally uncertain. Labeled "exotic" throughout.

**Category layout:**

| Category | Theme | Ideas |
|----------|-------|-------|
| **A** | Structural novelty (what the network *does*) | A1–A5 |
| **B** | Training dynamics (how it *learns*) | B1–B5 |
| **C** | Exotic / reality-bending | C1–C5 |
| **D** | Meta paradigm (training / data / inference) | D1–D3 |

## Category A — Structural novelty

### A1. Resonant Expert Interference — complex-valued MoE

**One-line pitch.** Each expert emits a complex number `a_k · exp(i·θ_k)`; top-k combination is a vector sum in ℂ; magnitude is the output. Phase-aligned experts interfere constructively, disagreeing ones cancel.

**Motivation & theoretical grounding.**
Standard MoE aggregates expert outputs as a real-valued weighted sum. This has an underappreciated weakness: there is no mechanism for experts to *disagree* destructively. Two experts that produce opposite views of the same input get averaged into a mushy compromise, and the router has to rely solely on pre-routing to resolve contradictions. Allowing expert outputs to carry phase information gives the model a natural *consensus mechanism*: outputs that agree are reinforced; outputs that disagree partially cancel.

Complex-valued neural networks have a substantial literature (Trabelsi 2018 "Deep Complex Networks," Arjovsky 2016 unitary evolution RNNs, CoShNet for compact CNNs) but have never been deployed for MoE expert combination. The connection to quantum-inspired classification (Stoudenmire 2016 tensor networks) is conceptual rather than literal — we're not doing quantum computation, but we're exploiting the same "amplitudes add in superposition" mathematics for representational benefit.

**Mechanism sketch.**
For each MoE layer with `E` experts and top-k selection:

```python
# Expert output: (magnitude, phase) pair encoded as two real channels.
# expert_k(x) -> Tensor[B, T, 2, D]  where [:, :, 0] is magnitude, [:, :, 1] is phase-angle

a_k, theta_k = self.experts[k](x).chunk(2, dim=2)      # magnitude, phase
a_k = F.softplus(a_k)                                   # keep positive
# complex representation: (real, imag) = (a*cos, a*sin)
real_k, imag_k = a_k * torch.cos(theta_k), a_k * torch.sin(theta_k)

# Top-k combination in complex plane
real_total = sum(real_k for k in topk)
imag_total = sum(imag_k for k in topk)

# Final output: magnitude
out = torch.sqrt(real_total ** 2 + imag_total ** 2 + eps)   # fp32 promotion for sqrt
# Or: simpler alternative, Re(sum) if we want phase info to cancel without explicit magnitude
out = real_total
```

Phase-per-iteration variant (natural extension under Parcae): make `θ_k` depend on iteration index `i`:
```python
theta_k_i = theta_k + delta_k[i]     # delta_k[i] ∈ R^D, learned per-iter per-expert
```
A single expert becomes three phase-rotated versions across iterations. Composes with N1 (per-expert per-iter γ) — one controls magnitude, the other phase.

**Why this is novel vs our knowledge base.**
The cookbook documents MoE combination as weighted real sum (Section 1.9 Sinkhorn mHC is a related idea but works on different lines — 4-branch residuals, not phase). No complex-valued module exists in our stack except RoPE, which uses complex arithmetic only for positional rotation, never for output combination. The ZAYA1 router + PID balancing + γ_{e,i} apparatus from FrankenMoE-Loop is orthogonal — A1 changes *what* experts emit, not *how* they're chosen.

**Hardware fit on gfx1151.**
- Complex = 2 real fp16 channels. rocBLAS `HHS_BH_` matmul handles this.
- `sqrt` in fp16 has dynamic-range risk; promote the `sqrt(real² + imag²)` reduction to fp32 (joins the existing FP32 promotion list).
- `torch.cos` / `torch.sin` are fp16-safe if input is clamped to `[-π, π]`. Alternative: store `(real_weight, imag_weight)` directly and skip the trig.
- Compile: elementwise ops fuse cleanly. No LDS pressure concerns; adds one extra channel per expert output.
- Expected overhead: 2× expert output channels → 2× expert-output memory, 2× expert-output FLOPs for the combination step. At our scale (MoE FFN is ~40 % of compute), roughly 1.05× total cost.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Phase training dynamics collapse (θ converges to 0 or 2π uniformly) | High | Med | Phase warmup curriculum: start with θ frozen at 0 (reduces to standard MoE), unfreeze after 10% of training |
| Magnitude collapse under `abs(Σ)` bottleneck (gradient vanishes for some experts) | Med | Med | Stable `hypot` in fp32; gradient clipping on magnitude output |
| fp16 precision insufficient for phase differentiation | Med | Low | Promote θ tensor to bf16 — oh wait, bf16 unsupported on gfx1151; use fp32 for θ storage, cast to fp16 at forward |
| Top-1 routing (our R2 default) collapses complex benefit to trivial | High | High | **A1 requires top-k ≥ 2**; activate alongside R1 ablation or a dedicated top-2 variant |
| Interaction with N1 γ_{e,i} confuses training signal | Low | Med | Disable γ during phase warmup; re-enable after θ is producing diverse values |

**Implementability:** ★★★★ medium. ~2-3 weeks prototype. Requires (a) dual-output expert head, (b) complex-combination kernel (stock pytorch works), (c) phase warmup curriculum. Clear fallback: fix θ = 0 (recovers standard real MoE).

**Trigger condition.** Pick A1 if either:
- L11 R1 ablation validates top-k > 1 as viable at our scale (then A1 is the natural extension), or
- We hit a plateau on MoE-domain quality benchmarks and suspect "expert compromise" (averaging) is the root cause.

**Standalone evaluation protocol.**
1. Train a non-looped 4-expert top-2 baseline on wikitext-103 (5k steps) — this establishes the top-2 real-valued MoE floor.
2. Train the same architecture with A1 phase extension (warmup 10% real-only, then unfreeze θ) — 5k steps.
3. Metrics: BPB, router entropy, **θ distribution per expert** (is phase actually being used?), magnitude-distribution stability, per-expert utilization.
4. Pass criteria:
   - (a) BPB at step 5000 ≤ baseline BPB at step 5000 (parity or improvement).
   - (b) θ distribution shows non-uniform structure (KL from uniform > 0.1 nats).
   - (c) No magnitude collapse (min-expert magnitude > 10% of max-expert magnitude).
5. Fail path: revert to top-2 real; publish negative result in KB.

---

### A2. Reversible Parcae — invertible coupling layers for O(1) activation memory

**One-line pitch.** Replace standard residual blocks with coupling-layer invertible blocks (RevNet-style); backward reconstructs forward activations from output instead of storing them; activation memory becomes O(1) in iteration count.

**Motivation & theoretical grounding.**
Activation memory in Parcae scales linearly with `mean_recurrence` because each iteration's residual state must be stored for backward. For OdinHalo at `mean_recurrence=3` with `backprop_depth=3`, 3× activation memory. Reversible architectures (Gomez 2017 "RevNet: backpropagation without storing activations," Dinh 2014 "NICE," Dinh 2016 "Real NVP," Kitaev 2020 "Reformer," Jacobsen 2018 "i-RevNet") achieve memory O(1) in depth by making the forward bijective — backward can reconstruct activations by running forward-inverse.

Reversible networks have never been combined with Parcae-style weight-sharing iteration. The opportunity is specifically large here because (a) Parcae's memory cost scales with iteration count, and (b) we already have fp16 activation memory as a bottleneck at our batch sizes.

**Mechanism sketch.**
Standard coupling-layer formulation. At each shared-layer step, split the residual stream in half along channels: `(u, v)` each of dim `D/2`. Apply two sub-blocks `F` and `G`:

```python
# Forward
u_next = u + F(v)                     # F: half-channel nonlinear transform
v_next = v + G(u_next)                # G: same

# Backward (exact reconstruction from output, no stored activations)
v_recon = v_next - G(u_next)          # recover v
u_recon = u_next - F(v_recon)         # recover u
```

`F` and `G` are existing HyPE / GQA / SwiGLU sub-blocks operating on half-channels. Final output is concat of `(u, v)` through the iter_norm, injection, skip-gates, etc.

**Memory savings analysis.** Parcae with `mean_recurrence=3` stores 3 full `[B, T, D]` residual tensors (plus intermediate activations). Reversible Parcae stores 1 final output tensor + recomputes everything on backward. For our current OdinHalo DDP run at ~6.2 GB/node, ~40% is iteration activations. Savings ~2.5 GB per node → enables either `mean_recurrence=6` at the same memory or `batch=32` at the same iteration count.

**Why this is novel vs our knowledge base.**
Cookbook Section 1 doesn't mention reversibility. `knowledge/architectures/looped_model_design_lessons.md` discusses 13 lessons from Parcae looping; reversibility isn't among them. `parcae_stable_looped_models.md` (Together AI reference) uses standard stored-activation backward. Combining reversibility with weight-sharing iteration is the specific novelty.

**Hardware fit on gfx1151.**
- Coupling inverse is one extra forward call per layer on backward. Roughly doubles backward compute time (forward is run twice in effect: once original + once as "inverse reconstruction").
- fp16: invertibility is algebraic, not numerical. The exact reconstruction assumes bit-exact forward reproducibility — which is fp16-risky if we're unlucky with non-associative sums. Mitigation: use **deterministic** reductions during forward (set `torch.use_deterministic_algorithms(True)` for the coupling region, or use explicit loop-free reductions), or store a small "error correction" residual per layer (defeats some of the memory win but is safer).
- Compile: `@torch.compiler.disable` on the coupling inverse will be simplest first pass. Later, compile the coupling with `recompute` hooks via `torch.utils.checkpoint` — standard pattern.
- Enables mean_recurrence=6 (2× iteration count) at same memory, OR batch=32 at current iteration count.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| fp16 non-associativity breaks invertibility (reconstructed activations diverge from stored ones) | High | Med | Deterministic reductions in coupling; or store epsilon-correction per layer |
| Coupling expressivity constraint (each F, G sees only half channels) | Med | Med | Alternate which half is updated; use wider sub-blocks (ffn_inner stays full) |
| Backward-time doubles → training slowdown overall | Med | High | Accept as tradeoff for memory; enables 2× batch which recovers tokens/sec budget |
| Incompatibility with MoDA depth-KV buffer (MoDA stores K,V tensors that aren't part of the coupling bijection) | Med | Med | MoDA buffer lives outside the coupling; re-compute MoDA K,V on backward as well (small extra cost) |
| Gradient checkpointing interaction (we already do this) conflicts with coupling inverse | Low | Med | Disable gradient checkpointing on coupling layers; coupling inverse IS the memory-saving mechanism |
| fp16 gradcheck fails on small proxy model | Med | Low | Test on fp32 model first; verify algebraic correctness before fp16 deployment |

**Implementability:** ★★★ medium-high risk, ~3-4 weeks prototype.
- Week 1: fp32 coupling-layer block reimplementation of HyPE + GQA; verify `torch.gradcheck` passes.
- Week 2: integrate into OdinHaloBase, replace `_forward_unrolled` with reversible loop.
- Week 3: fp16 porting + stability hardening (deterministic reductions).
- Week 4: memory benchmarking; enable `mean_recurrence=6` or `batch=32` test.

**Trigger condition.** Pick A2 if:
- We need to scale `mean_recurrence` beyond 3 (the current Parcae default) and memory is the bottleneck, OR
- We want to scale batch size for better DDP utilization on current memory budget, OR
- A research collaborator wants to explore deep-loop Parcae (5–10 iterations) that's impractical with current activation storage.

**Standalone evaluation protocol.**
1. Build a small reversible proxy model (d=256, 2 shared layers, 3 iterations) in fp32. Verify `torch.gradcheck` passes.
2. Port to fp16, verify numerical stability over 500 training steps on babylm.
3. Scale to d=768 (production OdinHalo size), measure peak memory.
4. Compare throughput: reversible-fp16 vs non-reversible-fp16 at (a) same config, (b) doubled batch, (c) `mean_recurrence=6`.
5. Pass criteria:
   - (a) fp16 reconstruction error < 1e-3 relative, sustained 500 steps.
   - (b) Peak memory ≤ 60 % of non-reversible baseline.
   - (c) At doubled batch or mean_recurrence=6: BPB ≤ non-reversible-3-iter baseline + 1%.
6. Fail path: document the fp16 instability mode; recommend bf16 hardware path for future (inaccessible today on gfx1151).

---

### A3. Shared Latent Workspace — Global-Workspace-Theory + any iterative model

**One-line pitch.** Introduce a small learned tensor `W ∈ ℝ^{K×D}` (K = 16–32 "concept slots") shared across tokens and updated across iterations. Tokens read from workspace via cross-attention each iteration; workspace writes back via tokens' attention pooled. Complements — not replaces — MoDA.

**Motivation & theoretical grounding.**
Global Workspace Theory (Baars 1988, Dehaene 2014) posits a small "workspace" in the brain where specialized modules broadcast information for collective access. VanRullen 2021 ("A Deep Learning Framework for Neuroscience") proposed its use in deep networks. Perceiver (Jaegle 2021) implemented a similar idea via learned latents: a fixed-size latent array attends to the full input sequence and is attended back by outputs. Perceiver's latents are (a) per-example, (b) non-iterative (one round of cross-attention), (c) replace self-attention.

Our setting differs along all three axes: (a) the workspace is sequence-scoped (persists across tokens within a sequence but not across batch examples), (b) it's *iterative* across Parcae iterations, (c) it *augments* self-attention rather than replacing it. This specific combination — iterative sequence-scoped workspace on top of an autoregressive transformer — has no published analogue.

**Mechanism sketch.**

```python
# At model init:
self.workspace = nn.Parameter(torch.randn(K, D) * 0.02)   # K = 16-32

# At each Parcae iteration:
# Step 1 — tokens read from workspace
h = h + CrossAttn(Q=h, K=workspace_current, V=workspace_current)

# Step 2 — workspace reads from tokens (updates itself)
workspace_new = workspace_current + Attn(Q=workspace_current, K=h, V=h)
workspace_new = RMSNorm(workspace_new)

# Step 3 — standard shared-layer block runs
h = shared_block(h, ...)

# At end of forward: workspace discarded (reset per sequence)
```

Workspace state flows across iterations via `workspace_new` → `workspace_current`. Gradient flows through both cross-attentions, so both tokens and workspace are trained.

**Parameters added.** `K × D = 16 × 768 = ~12K` for the workspace itself, plus two cross-attention sets of projections per iteration. With shared cross-attention weights (follow Parcae philosophy), weights added ≈ `6 × D² = 3.5M` total. Roughly 4% of unique params at 80M scale.

**FLOPs added.** At each iteration, tokens→workspace is `O(B·T·K·D)` FLOPs; workspace→tokens is `O(B·K·T·D)`. With K=16, T=512, this is 3 % of a standard attention layer's cost.

**Why this is novel vs our knowledge base.**
Cookbook Section 1.7 has "Meta Tokens" but those are appended to the sequence (dim change) and participate in normal self-attention. A3's workspace is *separate from the token sequence*, has fixed size independent of sequence length, and is updated via dedicated cross-attentions. MoDA depth-KV gives each token access to its own prior-iteration state; A3 gives each token access to a sequence-level summary. Complementary, not overlapping.

**Hardware fit on gfx1151.**
- Workspace is small; stored once per sequence. Zero LDS pressure.
- Cross-attention ops are well-understood GEMMs; rocBLAS handles them.
- K=16 to 32 fits easily in shared memory for accelerated reductions.
- Compile: two extra cross-attentions per iteration. Inductor fuses normally. No graph breaks expected.
- Expected throughput overhead: +3 to 5 % per forward; roughly matches MoDA's cost profile.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Workspace collapse (all K slots become identical) | High | Med | Slot-diversity regularization: penalize cosine similarity between workspace rows |
| Workspace domination (all info flows through workspace, token residual underutilized) | High | Med | Gate workspace read with learnable scalar init at 0.1 |
| Fixed K hurts long sequences where K slots can't capture enough structure | Med | Med | K-scheduling: start small (K=8), grow to 32 by mid-training |
| Gradient through workspace makes training unstable at init | Med | Low | Initialize workspace small (0.02 × randn); use ≥ 500 step warmup |
| Incompatibility with MoDA (two cross-attention writes per iteration → interference) | Low | Low | A3 writes to workspace; MoDA reads/writes depth-KV buffer. Orthogonal tensors |
| Workspace doesn't help small models (< 100M) | Med | Med | Validate via ablation at 40M proxy scale before committing |

**Implementability:** ★★★★ ~2-3 weeks prototype. Architecturally clean addition.
- Week 1: implement `SharedWorkspace` module; unit tests for cross-attention read/write symmetry.
- Week 2: integrate into OdinHaloBase forward; 200-step scorecard vs baseline.
- Week 3: regularization experiments (slot diversity, warmup schedule); finalize defaults.

**Trigger condition.** Pick A3 if:
- Scorecard metrics show poor cross-sequence coherence (e.g., long-range dependency benchmarks regress), OR
- We want to explore longer-sequence training (block ≥ 2048) and suspect bounded workspace would help generalization, OR
- A research direction on explicit "concept tracking" becomes relevant.

**Standalone evaluation protocol.**
1. Train a 40M-proxy OdinHalo variant with A3 (K=16) on babylm for 2k steps vs matched baseline without workspace.
2. Metrics: BPB, per-position attention entropy (does workspace concentrate on specific positions?), workspace-slot-cosine-similarity matrix (is there slot diversity?), long-range probe task (prompt → continuation with dependency 200 tokens back).
3. Pass criteria:
   - (a) BPB parity at step 2000.
   - (b) Non-trivial slot usage: max-slot-usage / min-slot-usage < 10x (no winner-take-all collapse).
   - (c) Long-range probe accuracy ≥ baseline + 2 %.
4. Scale gate: if 40M shows promise, re-run at 120M OdinFlat proxy with K=32.
5. Fail path: document that sequence-level workspace adds cost without quality gain at our scale; archive for later scale-up.

---

### A4. Residual Path Superposition — quantum-inspired k-paths in the residual stream

**One-line pitch.** Residual stream is not `[B, T, D]` but `[B, T, P, D]` — `P` parallel "paths" with learned amplitudes. Periodic "interference layers" let paths attend to each other. At output, paths are collapsed to a single `[B, T, D]` via amplitude-weighted sum.

**Motivation & theoretical grounding.**
Transformers commit to a single representation per token per layer. This forces premature disambiguation — ambiguous tokens (lexical ambiguity, syntactic ambiguity) must be resolved locally before downstream layers see them, even though later context might prefer a different resolution. Mixture-of-Experts addresses this on the *weights* axis; path superposition addresses it on the *representation* axis.

The metaphor is quantum: the residual stream holds `P` simultaneous interpretations ("states"), each with a complex amplitude. Layers evolve each state independently (like non-interacting paths in a Feynman integral). Interference layers let states interact (like a beam splitter). Final decoding is a measurement that collapses the superposition to a classical output.

Complex-valued amplitudes are optional. A real-valued version treats paths as alternative hypotheses with non-negative weights (like an ensemble); a complex-valued version allows cancellation between paths (A1 at the residual-stream level). We describe the real-valued version here as the primary mechanism; complex-valued is a composable extension.

**Mechanism sketch.**

```python
# Residual stream reshape at start of model:
h = token_embeddings                                    # [B, T, D]
h_paths = h.unsqueeze(2).expand(-1, -1, P, -1).clone()  # [B, T, P, D]
amplitudes = nn.Parameter(torch.ones(P) / P)            # learned, init uniform

# At each layer, apply the layer to each path independently:
for p in range(P):
    h_paths[:, :, p, :] = Layer(h_paths[:, :, p, :])

# At interference layers (every N layers, e.g., at the GQA positions):
# Let paths attend to each other
h_interfere = InterferenceAttn(Q=h_paths, K=h_paths, V=h_paths, axis=-2)  # attn over P axis
h_paths = h_paths + h_interfere

# At end of model (before head):
h_out = (h_paths * amplitudes[None, None, :, None]).sum(dim=2)   # [B, T, D]

# Or: weighted sum via learned per-position weights
weights = softmax(self.path_logits(h_paths), dim=-2)  # [B, T, P, 1]
h_out = (h_paths * weights).sum(dim=2)
```

Path count `P = 2, 4`. Interference layers every 2–3 main layers. Final collapse learned via small MLP on per-position path weights.

**Parameters added.** Amplitudes: `P` scalars (tiny). Interference attention: one QKV projection set over a P-length sequence (≈ `3·D²` weights shared across interference points; or per-interference-point weights for richer mixing).
**FLOPs added.** Each layer forward is run `P` times (P-fold compute overhead in dense layers). Interference attention is cheap (attention over `P ≤ 4` positions). Total: P× main layer compute + small overhead. For P=2, roughly 2× compute. Aggressive but bounded.

**Memory.** Residual is `P×` larger. At P=2, doubles activation memory. At batch=16, block=256, P=2, fits within current per-node budget.

**Why this is novel vs our knowledge base.**
This is the most exotic idea in the catalogue. Nothing remotely similar exists in our KB. Cookbook Section 1.8 MTP heads predict *future* tokens (horizontal axis); A4 tracks *parallel interpretations* (orthogonal axis). Engram (Section 1.6) is a memory store, not a parallel representation. Meta Tokens (Section 1.7) augment the sequence, not the residual. Closest vague analogue: PLE (Section 1.13) per-layer embeddings but those are static per-layer-scoped lookups, not dynamically-evolving parallel hypotheses.

**Hardware fit on gfx1151.**
- P× memory cost. At P=2 and current batch=16 block=256, manageable; at P=4 need smaller batch.
- P× compute cost per dense layer. This is the dominant drawback.
- Path interference layer is `B·T·P·D` shape — P small, LDS friendly.
- Compile: path axis becomes a batch dim in effect (stack, run, unstack). Inductor handles this fine with static P.
- Expected throughput: roughly `1/P`× baseline at constant per-node batch. Halve batch to recover throughput → 0.5× effective tokens/step.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Paths collapse (all P paths become identical by late training) | High | High | Path-diversity loss: penalize cosine similarity between paths at final collapse layer |
| Path ensemble is worse than single-path-with-more-params (P× compute could instead fund 2× model size) | High | High | Ablation: compare A4 with P=2 against baseline + 2× FFN width at matched compute |
| At our scale (80M), P=2 offers no benefit (ensembles need scale) | High | Med | Validate at 40M proxy; abandon if no signal |
| Collapse layer choice (which mechanism combines paths) matters more than path evolution | Med | Med | Multiple collapse variants in ablation: uniform average, softmax-weighted, learned MLP |
| fp16 precision is insufficient for destructive interference (complex variant) | Med | Low | Real-valued variant is primary; complex only as follow-up |
| Training time explodes → impractical even at small scale | Med | Med | Limit to P=2; if throughput insufficient, abandon |

**Implementability:** ★★ high-risk / research-grade. ~4-6 weeks prototype.

**Trigger condition.** Pick A4 if:
- We're deliberately exploring "weird / magical" territory for publishable research novelty, OR
- We observe specific failure modes that suggest "the model is committing too early to one interpretation" (e.g., garden-path sentences, ambiguous pronouns), OR
- A paper publishes ensemble-in-single-model techniques with convincing results and we want to extend.

**Standalone evaluation protocol.**
1. Build 40M proxy with P=2, interference every 3 layers. Baseline: matched-FLOPs single-path 80M model.
2. Train on babylm for 2k steps each. Compare BPB, domain-specific benchmarks.
3. Additional metric: **path divergence** — measure `mean_pairwise_cos_sim(h_paths)` across paths at each interference layer over time.
4. Pass criteria:
   - (a) Path divergence remains < 0.95 (paths don't fully collapse).
   - (b) BPB beats matched-FLOPs single-path baseline by ≥ 2 %.
   - (c) Downstream probe tasks (ambiguity resolution if available) improve by ≥ 5 %.
5. If all three pass, promote to 80M at P=2; re-evaluate.
6. Fail path: this idea dies; useful negative result. A4's core assumption is "per-residual-stream disambiguation beats single-path with matched compute" — if false at our scale, documented for future.

---

### A5. Kolmogorov-Compressibility Routing + Heterogeneous-Capacity Experts

**One-line pitch.** Experts come in multiple sizes per MoE layer (tiny, medium, large). Router input augmented with local-context LZ77 compressibility signal from our canary infrastructure. Compressible tokens route to tiny experts; novel / surprising tokens route to large experts. Asymmetric PID balancing.

**Motivation & theoretical grounding.**
Standard MoE uses uniform-capacity experts; routing is token-content-based. This misses a dimension of adaptive compute: the *information content* of a token in its context. A token whose identity is near-deterministic given 128 tokens of preceding context (e.g., the second word of a repeated phrase) needs minimal computation. A rare token that introduces new information deserves full capacity.

Kolmogorov complexity is the theoretical measure of this (uncomputable in general), but **Lempel–Ziv compressibility** is a cheap, computable upper bound. Our training stack already computes LZ77 compressibility per chunk for the degenerate-repetition canary (from ZAYA1 findings port). The same signal can route tokens to capacity-appropriate experts at near-zero marginal cost.

Mixture-of-Depths (Raposo 2024) routes on token-content difficulty; A5 routes on *contextual novelty* — a genuinely different signal.

**Mechanism sketch.**

```python
# Router input augmentation with compressibility signal:
# For each token position t, compute local LZ77 ratio in a window ending at t.
# This is O(W log W) per position (W = window size, e.g., 128). Cheap.

compressibility = lz77_ratio(token_ids, window=128)   # [B, T], float in [0, 1]

# Router input: concat [h_token, compressibility]
router_input = torch.cat([h_layer_input, compressibility.unsqueeze(-1)], dim=-1)
scores = router_mlp(router_input)

# Heterogeneous experts per MoE layer:
# E.g., 8 experts total: 2 tiny (ffn_inner=384), 4 medium (ffn_inner=1024), 2 large (ffn_inner=2048)
# Total param count: 2·384·D + 4·1024·D + 2·2048·D = ... matched to uniform alternative
```

**Asymmetric PID balancing.** Standard PID balancing targets uniform `p_e = 1/E`. Here, targets are set by expert capacity:

```
p_e_target = size_e / sum(size_k for all k)
```

Tiny experts get smaller token allocations proportional to their compute budget.

**Why this is novel vs our knowledge base.**
Our existing MoE design (FrankenMoE-Flat, Loop v2) uses uniform-capacity experts. MoD (mentioned in paper deep-dive) routes on content-derived difficulty. **Compressibility-based routing is distinct**: difficulty is a learned signal about the model's own uncertainty; compressibility is an *external* signal about the sequence's information structure. The heterogeneous-capacity + compressibility combination is the specific novelty.

LZ77 canary exists in `knowledge/training/zaya1_8b_findings_2026.md §3B` for stability; extending it to *routing* is the additive move.

**Hardware fit on gfx1151.**
- LZ77 ratio is a CPU-side computation in the dataloader (near-zero marginal cost — already computed for canaries).
- Router input gains 1 extra channel; MLP router weight grows negligibly.
- Heterogeneous experts require ScatterMoE dispatch to handle variable-size outputs. Minor modification: pre-allocate max-size output buffer, mask outputs from smaller experts to zero in unused dims. Or: dispatch per expert-size class separately.
- fp16-safe throughout.
- Expected throughput: parity with uniform MoE (same FLOPs), possibly slightly better if tiny experts run fewer tokens.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Compressibility is a weak proxy for "token difficulty" | High | High | Validate on held-out data: correlation between LZ77 and per-token loss reduction |
| Tiny experts undertrained (few tokens → few gradients) | Med | High | PID targets scale with expert size; tiny experts get proportionally more training examples relative to their capacity |
| Heterogeneous dispatch complicates ScatterMoE | Med | Med | Separate dispatch calls per size class; accept modest overhead |
| Router learns to ignore compressibility signal (deep networks can route around cheap features) | Med | Med | Add small auxiliary loss encouraging routing-signal correlation |
| "Punctuation goes to tiny, rare words go to large" pathology | Low | Med | Document; if pathological, disable — doesn't harm (falls back to signal being ignored) |

**Implementability:** ★★★ medium. ~2-3 weeks.
- Week 1: extend `content_canaries.py` to emit per-position LZ77 ratio; wire into dataloader and router input.
- Week 2: heterogeneous-expert ScatterMoE dispatch; asymmetric PID.
- Week 3: training run + scorecard.

**Trigger condition.** Pick A5 if:
- Canary infrastructure is already shipped (prerequisite — FrankenMoE-Flat v1 L8 onward), AND
- We want to explore adaptive-compute at training time (not just inference-side MoD), AND
- A clear hypothesis emerges that tokens have systematically different compute needs (e.g., observed from per-token loss stratification on dev set).

**Standalone evaluation protocol.**
1. Pre-experiment: measure per-token loss distribution on wikitext-103 validation; bin by LZ77-compressibility quartile; compute mean loss per quartile. If loss-versus-compressibility is monotonic, A5 has signal.
2. Baseline: 4-expert uniform MoE.
3. A5: 4 experts with sizes [0.5×, 1×, 1×, 2×] matched to baseline total compute.
4. Metrics: BPB, expert utilization by size class, per-quartile-of-compressibility BPB.
5. Pass criteria:
   - (a) Overall BPB within 1% of baseline.
   - (b) Utilization roughly matches capacity ratios (tiny experts used ~0.5× as often as medium).
   - (c) Per-quartile analysis: tiny experts see higher proportion of high-compressibility tokens (signal is being used).
6. Fail path: A5 collapses to uniform allocation; document that LZ77-based routing has no signal at this scale.

## Category B — Training dynamics novelty

### B1. Hidden-State Diffusion — each Parcae iteration is a denoising step

**One-line pitch.** Add Gaussian noise to the residual stream at the start of each Parcae iteration with a decreasing schedule; the shared block becomes a denoiser. Anneal noise to zero in the last 10% of training. Iteration schedule IS the diffusion schedule.

**Motivation & theoretical grounding.**
Denoising Diffusion Probabilistic Models (Ho 2020) formalized the idea that iterative denoising of a signal with a controlled noise schedule is a powerful training paradigm. Recent continuous-diffusion LMs (SEDD, Diffusion-LM) apply diffusion to token representations, but always as a *generative* mechanism, never as a *training-dynamic regularizer* on autoregressive LMs.

Parcae's iterative refinement is already structurally similar to iterative denoising: each pass takes a representation and produces a better one. Making this structural similarity explicit — by injecting noise at iteration boundaries and training the shared block to denoise — turns the loop into a principled noise-schedule curriculum.

**Mechanism sketch.**

```python
# Noise schedule: decreasing sigma per iteration
sigmas = [0.3, 0.1, 0.0]   # final σ=0 at output, full-noise at iter-0 input
# Training-time: anneal all sigmas toward 0 in last 10% of training

def forward_with_diffusion(self, input_ids):
    h = self.tok_embeddings(input_ids)
    input_embed = h

    # Iteration 0
    if training and sigma_schedule.active:
        h = h + sigmas[0] * torch.randn_like(h)
    h, current_kvs = self._run_shared_block(h, freqs_cis, depth_kv_buffer)
    h = self._apply_iter_norm(h, 0)

    # Iterations 1..N-1
    for i in range(1, self.mean_recurrence):
        h = self.injection(h, input_embed)
        if training and sigma_schedule.active:
            h = h + sigmas[i] * torch.randn_like(h)
        h, current_kvs = self._run_shared_block(h, freqs_cis, depth_kv_buffer)
        h = self._apply_iter_norm(h, i)

    # Standard head (σ=0 at output)
    normed = self.final_norm(h)
    return self.lm_head(normed)
```

**Noise annealing schedule.** Linear decay from initial `σ_i = [0.3, 0.1, 0.0]` to all-zero over the last 10% of training steps. This prevents inference-time distribution shift — by the end of training, the model sees clean residual streams.

**Why this is novel vs our knowledge base.**
Cookbook Section 4 ("Stability Lessons") discusses gradient clipping and label smoothing, not residual-stream noise. `knowledge/training/training_antipatterns.md` doesn't mention denoising objectives. Stochastic Depth (Huang 2016) randomly drops layers — different mechanism. The noise-as-curriculum framing, with schedule aligned to Parcae iterations, is specific to this design.

**Hardware fit on gfx1151.**
- `torch.randn_like(h)` in fp16 is trivial.
- Adds one elementwise add + one RNG op per iteration.
- Zero LDS pressure; zero compile disruption.
- Expected throughput overhead: < 1 %.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Inference-time distribution shift (training saw noise, inference sees clean) | Med | High | Anneal σ to 0 in last 10% of training |
| Noise at fp16 overflows residual stream for some iterations | Low | Med | Clamp noise magnitude; z-loss remains active |
| Optimal σ schedule unknown; hand-tuned defaults may not transfer | Med | Med | Sweep σ ranges at small scale before committing |
| Noise interacts with γ_{e,i}: γ might compensate by shrinking (implicit noise attenuation) | Med | Med | Monitor γ distribution; alert if γ drifts systematically lower after B1 enabled |
| No benefit at small scale | Med | Med | 40M proxy ablation before 80M commit |

**Implementability:** ★★★★★ very high. ~1 week. Pure CLI addition + forward-pass noise injection.

**Trigger condition.** Pick B1 if:
- We want a free regularization mechanism that doesn't change architecture, OR
- The main model exhibits overfit symptoms (train loss falls but held-out BPB plateaus), OR
- We want to explore whether noise-scheduled training produces more "correctable" iteration trajectories (enabling other extensions like B4 entropy conservation).

**Standalone evaluation protocol.**
1. 40M proxy OdinHalo baseline: train 2k steps, record BPB.
2. B1 variant: same config + noise schedule `[0.3, 0.1, 0.0]` with last-10% anneal.
3. Metrics: BPB, train-val gap, γ_{e,i} distribution.
4. Pass criteria:
   - (a) BPB ≤ baseline.
   - (b) Train-val gap narrower (B1 should regularize).
   - (c) No inference-time degradation after σ anneal (BPB at step 2k matches extrapolated clean-model trajectory).
5. If pass, scale to 80M OdinHalo; if fail, document and archive.

---

### B2. Temporal-Contrastive Iteration Learning — self-supervised aux across iterations

**One-line pitch.** Contrastive loss: for each token, pull iter-`i` representation close to iter-`i+1` (positive pair) and push away from iter-`i` of a *different token at the same position* (negative pair). Forces iterations to produce distinct-but-related trajectories.

**Motivation & theoretical grounding.**
Parcae iterations refine a representation. Without explicit pressure, there's no guarantee that iterations produce meaningfully different states — they could converge too quickly (iterations 1-2 are redundant) or diverge (iterations 1-2 are noise). A contrastive objective across iterations provides:
- **Temporal consistency** — same token at consecutive iterations should be semantically close (positive pair)
- **Batch-level diversity** — different tokens at the same position should remain distinguishable across iterations (negative pair)

This is SimCLR-style self-supervision (Chen 2020) applied across the *iteration axis* instead of across *augmentation views*. InfoNCE-style loss (van den Oord 2018):

```
L_contrastive = -log(exp(sim(h_t_iter_i, h_t_iter_i+1) / τ) / Σ_neg exp(sim(h_t_iter_i, h_t'_iter_i') / τ))
```

where negatives are sampled within-batch at the same position but different tokens.

**Mechanism sketch.**

```python
# At training, after forward pass, extract per-iteration residual stream:
h_iter_list = [h_iter_0, h_iter_1, h_iter_2]     # each [B, T, D]

# Project to contrastive embedding space (small head):
z_iter_list = [self.contrastive_head(h_i) for h_i in h_iter_list]  # each [B, T, D_proj]

# Contrastive loss: positive pair (iter_i, iter_i+1 for same token);
# negatives: (iter_i, iter_i' for different tokens at same position in batch)
L_contrastive = infonce_loss(z_iter_list, temperature=0.1)

# Total loss: LM loss + λ * contrastive loss, λ = 0.1
loss = lm_loss + 0.1 * L_contrastive
```

**Contrastive head.** Small MLP projecting `D → D/4` (the "projection head" in SimCLR). Only used at training; dropped at inference. Minimal parameter overhead.

**Why this is novel vs our knowledge base.**
Cross-covariance redundancy loss (InfoMamba paper deep-dive) is related but applied to feature redundancy within a single representation, not across iterations. No contrastive objective in cookbook. SimSiam / BYOL (vision) use momentum teachers (see B5); B2 uses within-model iteration pairs which is distinct.

**Hardware fit on gfx1151.**
- Projection head: tiny MLP, trivial compute.
- InfoNCE is `O(B² · T · D_proj)` for negative sampling — at our batch=16, T=256, D_proj=192, this is cheap.
- Compile: standard PyTorch ops, fuses cleanly.
- Expected overhead: ~2 % forward time + projection-head gradient.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Contrastive loss dominates LM loss; model optimizes for iteration-consistency instead of next-token prediction | High | Med | λ = 0.1 start; sweep if needed |
| Temperature τ badly tuned → either all-positive or all-negative collapse | Med | Med | Start τ = 0.1 (SimCLR default); adjust based on scorecard |
| Positive-pair collapse: iter_i and iter_i+1 become identical (iterations contribute nothing) | High | Low | Anti-collapse regularization: penalize when pairwise cosine > 0.99 |
| Distraction-regularizer pathology: BPB improves at aux-loss cost of overall capability | Med | Low | Downstream eval (HellaSwag, ARC) as backstop; drop if downstream regresses |
| Batch-level negatives inadequate at small batch (our batch=16) | Med | Med | Use gradient-accumulation-wide batches for negative sampling |

**Implementability:** ★★★★ ~1-2 weeks.

**Trigger condition.** Pick B2 if:
- Observation that Parcae iterations are "underutilized" (later iterations contribute little to final loss), OR
- A paper publishes iteration-level self-supervision wins, OR
- We want to test whether contrastive aux objectives help at our small-scale regime.

**Standalone evaluation protocol.**
1. 40M proxy: baseline (no B2), B2 with λ = 0.1, B2 with λ = 0.3.
2. 2k-step training run each.
3. Metrics: BPB, per-iteration representation divergence (how different are iter-0, iter-1, iter-2 post-training?), downstream probe tasks.
4. Pass criteria:
   - (a) BPB within 1% of baseline (can slightly worsen; primary goal is representation quality).
   - (b) Per-iteration divergence increases (iter-0 and iter-2 more distinct) without collapsing to orthogonality.
   - (c) Downstream probe ≥ baseline.
5. Fail path: document, archive.

---

### B3. Forward-Forward Auxiliary Objectives — gradient-free sub-training

**One-line pitch.** Use Hinton's Forward-Forward algorithm (2022) for auxiliary branches. Main LM loss uses standard backprop; auxiliary objectives (e.g., next-next-token prediction at iter-0) are trained via FF's goodness-function contrast. No backward graph for auxiliaries → smaller memory, faster aux training.

**Motivation & theoretical grounding.**
Hinton's Forward-Forward (2022 paper) proposes an alternative to backpropagation: each layer is trained to distinguish positive (real) vs negative (fake) inputs via a "goodness" function (typically squared activation). Positive data pushes goodness up; negative data pushes it down. Training is entirely forward-pass based — no backward graph is constructed.

FF has been explored primarily in vision MLPs; application to language models is minimal. The opportunity here: auxiliary objectives are often discarded in mainstream LM training because their gradient graph doubles memory. FF-trained auxiliaries are free memory-wise.

**Mechanism sketch.**

```python
# Main LM loss: standard backprop through full model.

# Auxiliary branch: goodness-based training.
# Example auxiliary: at iter-0 output, predict next-token with a tiny head.

# Positive examples: actual (token, next-token) pairs.
# Negative examples: corrupted (token, random-replacement-of-next-token) pairs.

with torch.no_grad():   # FF doesn't need gradients on the input!
    h_iter_0 = self.get_iter_0_output(input_ids)

# FF layer operates on h_iter_0:
good_pos = self.ff_aux(h_iter_0, positive=True).pow(2).sum(-1)  # goodness
good_neg = self.ff_aux(h_iter_0, positive=False).pow(2).sum(-1)

# FF loss: push good_pos high, good_neg low
loss_ff = F.softplus(threshold - good_pos) + F.softplus(good_neg - threshold)

# Backprop only through ff_aux module (not through h_iter_0 construction).
# This saves memory — h_iter_0 is detached.
```

**Key property.** The `torch.no_grad()` wrapping `h_iter_0` means the full model's forward graph isn't retained for the aux objective. Only the FF aux module's ~few-MB parameters need gradient storage.

**Why this is novel vs our knowledge base.**
Cookbook Section 4 covers gradient clipping and precision; no mention of gradient-free training paradigms. FF algorithm isn't in any KB doc. The use of FF for LM auxiliaries — specifically as memory-saving training — is not documented anywhere we've surveyed.

**Hardware fit on gfx1151.**
- FF forward is standard GEMM; rocBLAS handles.
- No backward graph for FF → zero extra activation memory for auxiliary.
- fp16-safe.
- Compile: FF layer compiles normally; wrapped in `no_grad()` region.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| FF underperforms backprop on the aux task → aux never helps | High | Med | Ablate: FF aux vs backprop aux at small scale |
| Goodness function selection matters (squared activation is Hinton's default; may not transfer) | Med | Med | Test multiple goodness formulations |
| FF aux converges slowly → useless in the training budget | Med | Med | Monitor FF aux loss trajectory; if not dropping, disable |
| Negative-sample construction matters (random replacement vs shuffled vs masked) | Med | Med | Hinton's "corrupted data" recipe is standard; validate on small task |

**Implementability:** ★★★ ~2 weeks.
- Week 1: implement FF aux module with goodness-based training.
- Week 2: integrate into Parcae-based OdinHalo; scorecard.

**Trigger condition.** Pick B3 if:
- We want multiple auxiliary objectives (e.g., multi-token prediction at iter 0, iter 1, iter 2) but memory budget is tight, OR
- Research interest in gradient-free training paradigms becomes relevant, OR
- A FF-based LM publication shows competitive performance.

**Standalone evaluation protocol.**
1. Baseline: standard LM loss only.
2. B3-a: LM loss + FF aux (predicting next-next token at iter-0).
3. B3-b: LM loss + standard-backprop aux (same task, gradient graph retained).
4. Metrics: BPB, peak memory, aux task accuracy.
5. Pass criteria:
   - (a) B3-a matches or exceeds baseline BPB (aux helps or is neutral).
   - (b) B3-a peak memory < B3-b peak memory (main value of FF is memory).
   - (c) B3-a aux accuracy > 0.5 * B3-b aux accuracy (FF is learning something).
6. Fail path: FF is weaker than backprop; document, archive, prefer standard aux losses.

---

### B4. Entropy-Conservation Loss — iterations should collapse uncertainty

**One-line pitch.** Auxiliary loss penalizes Parcae iterations that increase representational entropy across iterations. Forces the loop to converge (uncertainty collapse) rather than diverge or stabilize at high entropy.

**Motivation & theoretical grounding.**
Parcae's semantic interpretation is "progressive refinement toward an answer." If this is correct, the representation at iter-2 should be *more certain* (lower entropy) than at iter-0. We can make this explicit via an entropy-conservation regularizer.

Information-bottleneck theory (Tishby 1999, 2015) motivates this: deep networks should compress representations toward task-relevant bits. For iterative models, compression should monotonically increase. Measuring representational entropy exactly is hard; using **variance**, **top-k softmax entropy**, or **spectral entropy** as proxies is cheap.

**Mechanism sketch.**

```python
def estimate_entropy(h):
    # Option A: variance-based
    return h.var(dim=(-1,)).mean()
    
    # Option B: spectral entropy via PCA eigenvalues (per batch)
    # More expensive; use only at scorecard time
    u, s, v = torch.pca_lowrank(h.reshape(-1, D), q=min(32, D))
    probs = s**2 / (s**2).sum()
    return -(probs * probs.log()).sum()
    
    # Option C: output-distribution entropy after virtual head projection
    logits = self.lm_head(self.final_norm(h))   # compute actual logits at each iter (expensive)
    return -(F.log_softmax(logits, dim=-1) * F.softmax(logits, dim=-1)).sum(-1).mean()

# During forward, collect h at each iteration:
h_iter = [h_after_iter_0, h_after_iter_1, h_after_iter_2]
entropies = [estimate_entropy(h_i) for h_i in h_iter]

# Conservation loss: penalize entropy that increases iteration-to-iteration
L_entropy = sum(max(0, entropies[i+1] - entropies[i]) for i in range(len(entropies)-1))

loss = lm_loss + λ * L_entropy    # λ = 0.01 - 0.1
```

**Why this is novel vs our knowledge base.**
Cookbook doesn't mention entropy regularization on hidden states. Information-bottleneck training variants exist in literature but are applied to single-forward networks. **Applying entropy conservation specifically to the Parcae iteration axis** is the novelty.

**Hardware fit on gfx1151.**
- Variance-based entropy: trivial compute, fp16-safe.
- Spectral entropy via PCA: done once per scorecard run, not per step.
- Compile: standard reduction ops.
- Expected overhead: < 0.5 %.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Variance is a poor entropy proxy → regularizer is noise | Med | Med | Compare variance vs spectral entropy on same samples; use whichever correlates better with downstream quality |
| Forcing entropy collapse hurts capacity (model can't represent ambiguity) | Med | Med | Hinge loss with tolerance: only penalize entropy *increase beyond threshold* |
| Optimal λ unclear | Med | Med | Sweep at small scale |
| Interaction with iter_scales: iter_scales can artificially reduce variance | Med | Low | Measure entropy *after* iter_scales; include iter_scales in the regularizer target |

**Implementability:** ★★★★★ very high. ~2 days.

**Trigger condition.** Pick B4 if:
- Scorecard shows `residual_norm_by_iter` not decreasing across iterations (iterations aren't converging), OR
- We want to explicitly test the "iterations = refinement" hypothesis, OR
- A cheap regularizer is desired.

**Standalone evaluation protocol.**
1. Baseline: OdinHalo, no B4.
2. B4-variance: variance-based entropy conservation, λ = 0.05.
3. B4-spectral: spectral entropy, λ = 0.05 (more expensive, smaller-scale only).
4. Metrics: BPB, per-iteration entropy trajectory, downstream probes.
5. Pass criteria:
   - (a) Entropy does decrease across iterations (regularizer is working).
   - (b) BPB ≤ baseline.
   - (c) Downstream probes ≥ baseline.
6. Fail path: document iteration-entropy behavior as a sanity check; archive.

---

### B5. Momentum-Teacher Self-Distillation — DINO for LMs

**One-line pitch.** Maintain an EMA-averaged copy of the model (`τ = 0.996`) as a momentum teacher. Main model distills toward the teacher on a cheap auxiliary head (e.g., predict teacher's attention weights at 1–2 layers). DINO-style self-distillation without labels.

**Motivation & theoretical grounding.**
DINO (Caron 2021) demonstrated that self-distillation with a momentum teacher produces high-quality representations without any explicit labels. The trick: the teacher is a slowly-updated version of the student, providing a stable-but-evolving target. Momentum coefficient `τ = 0.996` gives the teacher effective averaging over ~1000 steps.

For LMs, DINO-style self-distillation has been lightly explored (data2vec by Baevski 2022, continuous distillation). Our twist: use it as a *pretraining auxiliary*, not as the main objective. LM loss remains primary; DINO provides an auxiliary signal that shapes the attention patterns and internal representations.

**Mechanism sketch.**

```python
# At model init:
self.teacher = deepcopy(self.model)
for p in self.teacher.parameters():
    p.requires_grad = False

# Training step:
student_output = self.model(input_ids)
with torch.no_grad():
    teacher_output = self.teacher(input_ids)   # no gradient to teacher

# Aux distillation loss: match attention maps at specific layers
student_attn = student_output.attention_at_layer(5)   # [B, H, T, T]
teacher_attn = teacher_output.attention_at_layer(5)

# KL divergence on attention distributions
L_distill = F.kl_div(
    student_attn.log_softmax(dim=-1),
    teacher_attn.softmax(dim=-1),
    reduction='batchmean'
)

loss = lm_loss + 0.1 * L_distill

# After gradient step: EMA update
with torch.no_grad():
    for p_teacher, p_student in zip(self.teacher.parameters(), self.model.parameters()):
        p_teacher.data.mul_(0.996).add_(p_student.data, alpha=0.004)
```

**EMA schedule.** `τ` can warm up from 0.99 to 0.999 over training, mirroring BYOL (Grill 2020).

**Why this is novel vs our knowledge base.**
IMU-1 recipe (in KB) uses EMA for **checkpointing** (post-hoc averaging of final-phase checkpoints). Using EMA as a *training-time teacher signal* is distinct and undocumented. DINO-for-LM paradigm is not in KB.

**Hardware fit on gfx1151.**
- Teacher is a full copy of the model: 2× parameter memory.
- At our 80M scale, teacher weights ~160 MB fp16. Teacher activations need not be stored (runs in `no_grad`).
- Teacher forward doubles training-step FLOPs (approximately). Partial mitigation: run teacher every N steps and reuse stale outputs.
- Compile: teacher is compilable separately. Minor graph duplication.
- Expected overhead: ~1.8× training step time, ~1.5× memory.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Training slowdown too severe for value | Med | High | Teacher forward every 4 steps instead of every step |
| Attention-map distillation doesn't transfer well at small scale | Med | Med | Ablate on 40M proxy; fall back to hidden-state distillation |
| EMA too slow at the start → teacher is noise | Low | Med | Skip DINO for first 10% of training (warmup); teacher starts as SFT-quality after warmup |
| Teacher collapse (teacher and student become identical, aux loss → 0, no signal) | Med | Low | Temperature / centering as in DINO |
| Interaction with our fp16 GradScaler | Low | Med | Teacher stays in fp16; no scaler interaction |

**Implementability:** ★★★★ ~1-2 weeks.

**Trigger condition.** Pick B5 if:
- We want to improve representation quality at the cost of training throughput, OR
- Post-v2 we have a specific auxiliary task in mind (attention-pattern regularization, hidden-state smoothness), OR
- Research interest in self-distillation for small LMs.

**Standalone evaluation protocol.**
1. Baseline: OdinHalo no teacher.
2. B5: OdinHalo + EMA teacher + attention distillation at 2 layers, λ = 0.1.
3. Metrics: BPB, downstream probes, attention entropy (does distillation stabilize attention?).
4. Pass criteria:
   - (a) BPB ≤ baseline.
   - (b) Downstream probes improve by ≥ 2 %.
   - (c) Training time < 2× baseline.
5. Fail path: archive; distillation likely needs more scale than we have.

## Category C — Exotic / reality-bending

### C1. Mycelial Expert Graph — token walks through learned expert graph

**One-line pitch.** Experts are nodes in a directed graph with learned `E × E` edge weights. At each MoE layer, tokens walk through ~3 expert nodes (one per Parcae iteration). Entry via initial routing; subsequent nodes selected via edge-weight transitions conditioned on token state.

**Motivation & theoretical grounding.**
Standard MoE routing picks one expert per token per layer, independently each time. This misses the opportunity for expert sequences to have structure. If "expert A followed by expert B" is a meaningful compositional primitive, a graph-walk framework lets the model learn it. Biological metaphor: fungal mycelium networks where nutrients (tokens) traverse a network of hyphae (experts), with edge strengths learned via reinforcement.

Graph-neural-network inspired MoE (there are a few publications) operates on graphs *in the data*; C1 is MoE where *experts* form the graph. This specific direction is unpublished as far as we've surveyed.

Walk length = `mean_recurrence` = 3 creates a natural compositional alignment: each token visits exactly 3 experts across the Parcae iterations, tracing a path through the graph.

**Mechanism sketch.**

```python
# Per MoE layer:
self.edge_weights = nn.Parameter(torch.randn(E, E) * 0.01)   # [E, E] learned
self.start_router = MLP(D -> E)                              # picks entry expert

# Forward at iter-0:
start_logits = self.start_router(h)
expert_0_idx = start_logits.argmax(dim=-1)                   # [B, T]
# Apply expert
h = h + self.experts[expert_0_idx](h)

# Forward at iter-1: walk one edge
# Transition logits: based on current expert index + current h state
# Prev-expert identity acts as a "row selector" into edge_weights;
# h adds a content-conditional bias
transition_logits = self.edge_weights[expert_0_idx] + self.bias_head(h)  # [B, T, E]
expert_1_idx = transition_logits.argmax(dim=-1)
h = h + self.experts[expert_1_idx](h)

# Forward at iter-2: walk another edge
transition_logits = self.edge_weights[expert_1_idx] + self.bias_head(h)
expert_2_idx = transition_logits.argmax(dim=-1)
h = h + self.experts[expert_2_idx](h)
```

**Walk balancing.** Standard PID doesn't apply (it balances per-layer expert loads; here we have walk-endpoints to balance). Need new balancing: **node-visit balancing** (each expert visited roughly uniformly across batch × walk-step) + **edge-balance** (no single edge dominates transitions).

**Why this is novel vs our knowledge base.**
No graph-walk routing in KB. Mycelial structure + walk-based MoE is unpublished territory. Closest analogue in KB: Sinkhorn mHC (cookbook Section 1.9) uses 4-branch residuals but with parallel, not sequential, combination.

**Hardware fit on gfx1151.**
- Edge weight matrix: tiny (`E² = 16` scalars per MoE layer at E=4).
- Transition computation: gather + small bias add. Cheap.
- Walk structure changes our ScatterMoE dispatch: each iter visits a potentially different expert subset. Dispatch becomes per-iteration-conditional.
- fp16-safe.
- Expected overhead: similar to standard MoE per-iter; aggregated over 3 iters = ~3× dispatch cost (which was cheap to begin with).

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Walk collapse (all tokens converge to same walk after training) | High | High | Edge entropy regularization; revisit penalty |
| Load balancing fails (some experts never visited, some over-visited) | High | High | Node-visit balancing via modified PID |
| Edge weights don't separate (stay near-uniform) | Med | Med | Edge-diversity loss; careful initialization |
| Replaces R2 sticky routing — incompatible with v2 creative additions (T7, T11) | Med | Certain | This is standalone, not composable with v2 |
| Walk length fixed to mean_recurrence = 3 is arbitrary | Low | Med | Could make walk-length learnable via halt signal (additional complexity) |

**Implementability:** ★★ high-risk / high-reward. ~4-6 weeks.

**Trigger condition.** Pick C1 if:
- v2 R2 routing is validated but we observe it's too rigid (experts are "too specialized" and the model can't compose them), OR
- Research-direction interest in graph-structured MoE, OR
- We have bandwidth for a high-novelty research experiment.

**Standalone evaluation protocol.**
1. 40M proxy with 4 experts, 2 MoE layers, 3-step walks.
2. Baseline: standard R2 sticky routing with same experts.
3. Metrics: BPB, walk diversity (entropy of walk-index distribution over batch), per-expert visit count, edge-weight distribution.
4. Pass criteria:
   - (a) BPB parity or better.
   - (b) Walks diverse (entropy > 2.0 nats).
   - (c) All experts visited.
5. Fail path: document walk failure modes; archive.

---

### C2. Cross-Example Batch Attention — in-context learning during pretraining

**One-line pitch.** Within a training batch, tokens from *different examples* can attend to each other via gated cross-attention at selected layers. The model learns to transfer patterns across examples at pretraining time — implicit in-context learning as a training-time primitive.

**Motivation & theoretical grounding.**
In-context learning (ICL) is an emergent property of large LMs: given examples in-prompt, the model adapts to a new task. ICL is mysterious — no explicit training objective targets it. What if we make it explicit at pretraining?

C2 allows tokens in example `i` of a batch to attend to tokens in example `j` when they might contain transferable patterns. This intentionally breaks the "examples are independent" assumption of standard batching; the gamble is that this inductive bias accelerates ICL emergence.

**Mechanism sketch.**

```python
# Standard batch: [B, T, D]. Rearrange for cross-batch attention:
# Concat batch dim with time dim conditionally, gated by learned similarity.

# At selected layers (e.g., every 4th):
h_bt = h.view(B * T, D)   # flatten batch

# Gate: for each token, predict which other examples to attend to
# Small attention: token -> batch-of-tokens, top-k with k << B*T
gate_logits = self.cross_batch_gate(h_bt)        # [B*T, B*T] scores (or sparse approximation)
top_k_idx = gate_logits.topk(k=32, dim=-1).indices   # top-32 cross-batch partners

# Gather partners
partners = h_bt[top_k_idx]   # [B*T, 32, D]
cross_batch_attn = F.scaled_dot_product_attention(
    h_bt.unsqueeze(1), partners, partners
)   # [B*T, 1, D]

h = h + gate_scalar * cross_batch_attn.squeeze(1).view(B, T, D)
```

**Gating.** Learned scalar `gate_scalar`, init near 0. Layer can opt out of cross-batch attention by keeping gate low. Only a fraction of layers enable cross-batch (e.g., 2 of 14).

**Training batch design.** Normally examples in a batch are independent. C2 benefits from *intentional curation*: mix examples from related domains or tasks in the same batch (hard-batched curriculum). This is a data-side change that amplifies the architectural change.

**Why this is novel vs our knowledge base.**
Cookbook assumes within-example attention only. No cross-batch attention in KB. Prefix tuning and related ICL research operates at inference; C2 does it at pretraining.

**Hardware fit on gfx1151.**
- Cross-batch attention is `O((B·T)²)` for full connectivity — too expensive. Sparse top-k at k=32 gives `O(B·T·k·D)` which is manageable.
- DDP implication: cross-batch connectivity must stay *within* a single rank's batch (inter-rank cross-attention is impractical). Our batch=16 per rank gives reasonable scope.
- fp16-safe.
- Compile: top-k selection is data-dependent but Inductor handles with specialization.
- Expected overhead: ~5-10 % forward time at 2 of 14 layers enabled.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Gate stays at 0 (model learns to ignore cross-batch) → no effect | High | High | Bias gate init positive; or force small epsilon flow |
| At small batch (16 per rank) there's insufficient diversity for meaningful cross-attention | High | High | Build cross-batch partners from DDP all-gather (much larger effective batch, expensive) |
| Breaks independence assumption → gradient noise increase | Med | Med | Accept; measure |
| Cross-batch attention learns spurious correlations between unrelated examples | Med | Med | Intentional batch curation; or limit cross-attention to same-domain batches |
| Throughput overhead not recovered by quality gain | Med | High | Ablate rigorously before committing |

**Implementability:** ★★ speculative. ~4 weeks.

**Trigger condition.** Pick C2 if:
- We want to explore early ICL emergence as a research direction, OR
- A paper shows cross-batch attention gains, OR
- We're deliberately exploring the "fundamentally unusual" design space.

**Standalone evaluation protocol.**
1. Baseline: standard batching.
2. C2 at 40M proxy with 2 layers enabled for cross-batch attention.
3. Metrics: BPB, few-shot ICL probe (task inserted at inference with examples in-prompt), gate-activation trajectory.
4. Pass criteria:
   - (a) BPB within 2 % of baseline.
   - (b) ICL probe accuracy ≥ baseline + 5 %.
   - (c) Gate activates (> 0.1) on at least one enabled layer.
5. Fail path: document; cross-batch attention is likely ahead of its time at small scale.

---

### C3. Self-Chain-of-Thought Pretraining — iteration-output recycling

**One-line pitch.** At training, with probability p, feed iter-`N-1` output as *input* for a fresh iter-0 run (concatenated or replacing token embeddings). Teaches the model to continue its own refinement. Uses model-internal signals as synthetic training data.

**Motivation & theoretical grounding.**
Parcae iterates within a single forward pass. What if the model could iterate *across forward passes* — using one pass's output as another pass's input? This is the essence of chain-of-thought: generate partial thoughts, then generate more thoughts from those.

C3 makes this a training-time primitive. Mid-training, periodically:
1. Run a standard forward pass through all 3 Parcae iterations.
2. Take iter-2 output (pre-LM-head).
3. Re-inject it as if it were fresh input embeddings for a *second* forward pass.
4. Train the second pass to continue the first.

The model learns to consume its own iteration outputs as inputs. At inference, this enables arbitrary-depth chaining ("think longer" as needed).

**Mechanism sketch.**

```python
# Probability of applying C3 per batch:
p_self_cot = 0.1

if training and torch.rand(1) < p_self_cot:
    # Step 1: standard forward, detach output
    with torch.no_grad():
        h_first_pass = self.forward_through_iterations(input_ids)
        # h_first_pass: [B, T, D] — the iter-2 output, detached

    # Step 2: second forward using h_first_pass as input embedding
    # Replace the standard token-embeddings with the detached h_first_pass
    # Run all 3 iterations again
    h_second = self.run_iterations(input_embed=h_first_pass, input_ids_for_pos=input_ids)

    # Loss on second-pass output (predicting next token)
    loss = standard_lm_loss(h_second, next_tokens)
else:
    # Normal training
    loss = standard_lm_loss(self.forward(input_ids), next_tokens)
```

**Why this works.** The model learns to represent "continued refinement" in the same embedding space as initial tokens. At inference, this enables test-time-compute scaling via self-chaining (related to Markovian RSA but entirely unprompted).

**Why this is novel vs our knowledge base.**
AP trimming (KB) is training-data construction via teacher-model outputs. Markovian RSA (in ZAYA1-8B findings) is *inference-time* chaining with fixed-size tails. C3 is *training-time* self-chaining, teaching the model to consume its own outputs.

**Hardware fit on gfx1151.**
- Double the forward cost on C3-active batches (two full forwards).
- At `p = 0.1`, adds ~10 % training time.
- Memory: two forwards in same step → ~1.8× activation memory peak. Mitigation: use gradient checkpointing to offset.
- Compile-friendly if p is Python-constant.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Model learns to ignore second pass (treat as duplicate) | High | Med | Require different targets: first pass = teacher, second pass = student with larger target context |
| Second pass outputs are degenerate (close to input → no learning) | Med | Med | Require non-trivial transformation; reward difference between iter-2-first-pass and final-second-pass |
| Composability with Markovian RSA unclear | Low | Low | Can co-exist; Markovian RSA is inference-side, C3 is training-side |
| Second-pass gradient through iteration creates training instability | Med | Med | Start with `p = 0.05`, increase if stable |

**Implementability:** ★★★ ~1-2 weeks.

**Trigger condition.** Pick C3 if:
- We're building toward test-time-compute scaling (Markovian RSA adoption), OR
- Research interest in self-generated training data.

**Standalone evaluation protocol.**
1. Baseline: standard training.
2. C3 at p = 0.1.
3. Metrics: BPB, inference-time performance when self-chaining is applied (evaluate with 2-pass forward at inference).
4. Pass criteria:
   - (a) Training BPB ≤ baseline.
   - (b) Inference with self-chaining shows measurable improvement over single-pass.
5. Fail path: archive.

---

### C4. Hypernet-Generated Per-Iteration Weights

**One-line pitch.** A small hypernetwork generates layer weights at each Parcae iteration — weights are a *function* of iteration index and aggregated token statistics. Under Parcae's weight-sharing, this is the inverse move: weights vary per iteration, controlled by a meta-network.

**Motivation & theoretical grounding.**
HyperNetworks (Ha 2016) generate weights for a target network via a smaller network. Applied to Parcae: instead of sharing the same weights across iterations (strict), a hypernet generates iteration-specific weights from the iteration index and some conditioning signal.

This is the **anti-Parcae move**: introducing iteration-specific capacity. Benefit: the model can learn distinct behavior per iteration without explicit weight duplication. Cost: hypernet is a new parameter budget.

**Mechanism sketch.**

```python
# Hypernet input: iter_idx (one-hot or embedding) + aggregated token statistics
# Output: delta-weights for one layer

class Hypernet(nn.Module):
    def __init__(self, iter_dim, h_dim, base_weight_shape):
        super().__init__()
        self.iter_embed = nn.Embedding(3, iter_dim)
        self.mlp = nn.Sequential(
            nn.Linear(iter_dim + h_dim, 256),
            nn.GELU(),
            nn.Linear(256, reduce(mul, base_weight_shape))
        )
        self.base_weight_shape = base_weight_shape

    def forward(self, iter_idx, h_summary):
        # h_summary: [B, D] (mean over tokens, for instance)
        input = torch.cat([self.iter_embed(iter_idx), h_summary], dim=-1)
        delta = self.mlp(input).view(-1, *self.base_weight_shape)
        return delta

# In forward at each iteration:
base_W = self.shared_layer.weight   # [out, in]
delta_W = self.hypernet(iter_idx=i, h_summary=h.mean(dim=1))   # [B, out, in]
effective_W = base_W + delta_W   # per-batch-item per-iter weight

# Apply — needs per-batch-item matmul, i.e., bmm
out = torch.bmm(h.transpose(0, 1), effective_W.transpose(-1, -2))
```

**Why this is novel vs our knowledge base.**
PLE (cookbook Section 1.13) has per-layer embeddings but static. TTT Fast Weights (Section 1.15) has dynamic weights updated via fast-weight training; no iteration dimension. Hypernet-generated weights per Parcae iteration is a genuinely new construction.

**Hardware fit on gfx1151.**
- Per-batch weights require `bmm` instead of `mm` → higher memory traffic.
- Hypernet itself is cheap (small MLP).
- Deltas are small perturbations of the base weights; can be low-rank for efficiency.
- Expected overhead: substantial (2-3× matmul cost if naive; low-rank delta mitigates).

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Parameter explosion (hypernet output is full weight tensor) | High | Med | Low-rank delta: hypernet outputs two small matrices, base_W gets rank-k update |
| Training instability (weights change per-step per-batch) | Med | Med | Small learning rate on hypernet |
| Throughput overhead not recovered by quality | High | High | Ablate rigorously |

**Implementability:** ★★ ~4 weeks.

**Trigger condition.** Pick C4 if:
- Evidence that Parcae weight-sharing is too strict (later iterations need distinct capacity), OR
- Research interest in hypernet applications to LM training.

**Standalone evaluation protocol.**
1. Baseline: Parcae with shared weights.
2. C4 with low-rank (rank=8) hypernet-generated deltas.
3. Metrics: BPB, hypernet delta-norm trajectory (is the hypernet actually producing meaningful deltas?).
4. Pass criteria:
   - (a) BPB ≤ baseline + 1%.
   - (b) Hypernet delta norms non-trivial (not collapsed to zero).
5. Fail path: archive; Parcae weight-sharing is adequate.

---

### C5. Lyapunov-Stable Iterative Attention (DEQ / Hopfield-inspired)

**One-line pitch.** Define an explicit energy function `E(h)` and parameterize iteration as gradient descent on E. The loop is provably stable (descent-guaranteed) and has a clear fixed-point interpretation. Variant of Deep Equilibrium Models (DEQ) with fixed iteration count.

**Motivation & theoretical grounding.**
Deep Equilibrium Models (Bai 2019) run a single layer to convergence (or near-convergence) using fixed-point iteration. Hopfield networks (updated by Ramsauer 2020 to match transformer attention) define an energy function whose minimum is the stored pattern.

C5 unifies these: define an energy function for the residual stream; Parcae iterations are gradient descent on this energy. At fixed iteration count, we don't converge to exact minimum but descend toward it.

**Mechanism sketch.**

```python
# Energy function (simplified):
# E(h) = -½ h^T M h + V(h)
# where M is learned symmetric matrix, V is a learned nonlinear potential

def energy(self, h):
    quadratic = -0.5 * (h @ self.M @ h.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(-1)
    potential = self.potential_mlp(h).sum(-1)
    return quadratic + potential

# Iteration: symplectic gradient descent
for i in range(mean_recurrence):
    grad_E = torch.autograd.grad(energy(h).sum(), h, create_graph=True)[0]
    h = h - lr * grad_E
```

**Simplification.** Computing `grad_E` via autograd inside the forward is expensive. A more practical version: parameterize the iteration *as if* it were gradient descent on some energy, without explicitly computing the energy. The "shared block" is trained to act as `-∇E`.

**Why this is novel vs our knowledge base.**
DEQ / Hopfield framing not in KB. Lyapunov-stable iteration as design principle not documented.

**Hardware fit on gfx1151.**
- Autograd-in-forward doubles or triples compute.
- Without explicit energy: same cost as standard Parcae.
- fp16-safe for standard Parcae-like formulation.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Explicit energy formulation too expensive | High | High | Use implicit formulation (train layers to act as -∇E without computing E) |
| Lyapunov guarantee doesn't transfer to discrete-step, finite-iteration regime | Med | High | Accept as theoretical framing, not strict guarantee |
| Energy function choice (quadratic, Hopfield-like, other) affects outcome | Med | High | Experiment with multiple forms |

**Implementability:** ★★ ~4-6 weeks.

**Trigger condition.** Pick C5 if:
- Research interest in theoretical frameworks for iterative architectures, OR
- Our Parcae implementation exhibits instability that Lyapunov theory could explain/fix.

**Standalone evaluation protocol.**
1. Implicit C5: train Parcae where shared block structure is constrained to act as -∇E (via skip-gate + learned scale constraints).
2. Metrics: BPB, iteration trajectory convergence rate, ||h_{i+1} - h_i|| norms.
3. Pass criteria:
   - (a) BPB ≤ baseline.
   - (b) Iteration trajectory demonstrably converges (monotonic decrease in some energy proxy).
4. Fail path: archive.

## Category D — Meta paradigm

### D1. Static Prefill Cache — content-independent KV heads

**One-line pitch.** Profile which attention heads at which layers produce output depending only on absolute position (not content). Cache those once at model init. At inference, compute only the content-dependent residual. Throughput win on cold prefill.

**Motivation & theoretical grounding.**
Some attention patterns are content-independent: a head that always attends to the previous token, for example. Its K and V outputs depend only on position (via RoPE rotation of fixed initial K, V) — not on token content. These can be precomputed and cached.

This has been observed in interpretability research (Olsson 2022 "In-context learning and induction heads") but not systematically exploited as a compute optimization.

**Mechanism sketch.**

```python
# Profile: for each (layer, head), compute variance of K and V across content permutations.
# Heads with low variance are "position-only" heads.

# At model init (after training):
for (layer_idx, head_idx) in position_only_heads:
    static_K = compute_K_from_position_only(layer_idx, head_idx, max_seq_len)
    static_V = compute_V_from_position_only(layer_idx, head_idx, max_seq_len)
    register_as_buffer(f'static_kv_{layer_idx}_{head_idx}', (static_K, static_V))

# At inference prefill:
for layer in layers:
    for head in heads:
        if (layer_idx, head_idx) in position_only_heads:
            # Skip projection; use cached
            K, V = static_KV[(layer_idx, head_idx)][:T]
        else:
            K, V = compute_projections(layer, h)
```

**Why this is novel vs our knowledge base.**
Inference optimization is mostly not in KB (focus is training). This specific "profile heads, cache static ones" technique isn't documented.

**Hardware fit on gfx1151.**
- Pure inference-time optimization.
- Reduces prefill FLOPs on identified static heads.
- Memory cost: one K, V tensor per static head (bounded by max_seq_len).

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Few heads are truly position-only → small benefit | High | High | Profile first to assess potential |
| Position-only heads might actually be subtly content-dependent | Med | Med | Validate on held-out data |

**Implementability:** ★★★ ~2 weeks.

**Trigger condition.** Pick D1 if:
- Prefill latency is a bottleneck for deployment, OR
- Interpretability work reveals many static-position heads.

**Standalone evaluation protocol.**
1. Profile trained model: variance of K, V across content, per (layer, head).
2. Identify candidates (low-variance).
3. Measure prefill latency with / without static caching.
4. Pass criteria: latency reduction > 5 % with no quality regression.

---

### D2. Adaptive Data-Mixture Curriculum — real-time loss-gradient-based mix

**One-line pitch.** Dataset mix weights (web / code / math / multilingual) auto-tune in real-time based on per-domain loss gradient signal. Reactive version of DoReMi; mixture reshapes during training.

**Motivation & theoretical grounding.**
DoReMi (Xie 2023) computes optimal data-mixture weights via a proxy task, then trains with fixed weights. Real-time adaptation would respond to training dynamics: if math loss plateaus while code loss is still descending, upweight math.

**Mechanism sketch.**

```python
# Track per-domain loss history
domain_loss_history = {d: deque(maxlen=100) for d in domains}

# Every K steps:
for d in domains:
    recent_loss_slope = linear_fit(domain_loss_history[d])
    if recent_loss_slope > threshold:   # loss stagnant or rising
        mix_weights[d] *= 1.1
    elif recent_loss_slope < -steep_threshold:   # loss descending fast
        mix_weights[d] *= 0.95

mix_weights = softmax(mix_weights)
```

**Why this is novel vs our knowledge base.**
CLIMB (in KB) is static. Real-time adaptive version isn't documented.

**Hardware fit on gfx1151.**
- Pure dataloader change; no model change.
- Requires tracking per-domain loss.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Adaptive mix is noisy → worse than fixed | Med | Med | Bound weight changes; moving-average smoothing |
| Some domains require contrastive training (can't simply upweight) | Med | Low | Domain-specific overrides |

**Implementability:** ★★★ ~2 weeks.

**Trigger condition.** Pick D2 if:
- Multi-domain training (e.g., dolma-mix) shows uneven progress, OR
- Research interest in curriculum learning.

**Standalone evaluation protocol.**
1. Baseline: fixed CLIMB-optimized mix.
2. D2: adaptive mix, same domains, same total tokens.
3. Metrics: per-domain final BPB, total final BPB.
4. Pass criteria: D2 total BPB ≤ baseline + 1 %, per-domain variance reduced.

---

### D3. Gradient-Norm-Gated Example Dropping

**One-line pitch.** Examples whose gradient norm exceeds a threshold are demoted (downweighted or dropped). Filters outliers and memorization candidates dynamically during training.

**Motivation & theoretical grounding.**
High per-example gradient norm often indicates: (a) noisy / mislabeled examples, (b) examples the model is memorizing (rare n-grams with high gradient signal), or (c) truly hard examples that would benefit from extra training. The challenge is distinguishing these.

Heuristic: during early training, all are useful. During late training, high-gradient-norm examples are more likely memorization/noise than useful hard examples. D3 applies this heuristic dynamically.

**Mechanism sketch.**

```python
# Per-example gradient norm:
# (after loss.backward() but before optimizer.step(), on a per-microbatch basis)

for example in microbatch:
    ex_loss = compute_loss(example)
    ex_grad_norm = compute_grad_norm(ex_loss, model_params)
    if ex_grad_norm > threshold * moving_median:
        example.weight *= 0.5   # downweight

# Apply weighted loss
total_loss = sum(ex.weight * ex.loss for ex in microbatch)
total_loss.backward()
```

**Why this is novel vs our knowledge base.**
Gradient norm clipping is global. Per-example dynamic weighting based on grad-norm isn't in KB.

**Hardware fit on gfx1151.**
- Per-example gradient requires either microbatching or careful hook management.
- Minor overhead; standard PyTorch.

**Risk register.**

| Risk | Severity | Likelihood | Mitigation |
|------|---------:|-----------:|------------|
| Dropping too aggressively → data efficiency regresses | Med | Med | Only downweight, don't fully drop |
| Hard examples (genuinely informative) get dropped | Med | Med | Threshold tunes over time; early training preserves all |

**Implementability:** ★★★ ~1-2 weeks.

**Trigger condition.** Pick D3 if:
- Memorization symptoms observed in late training, OR
- Research interest in data-curriculum techniques.

**Standalone evaluation protocol.**
1. Baseline: no D3.
2. D3 with downweighting at 2σ threshold.
3. Metrics: BPB, memorization probe (exact-string continuation from training set).
4. Pass criteria: BPB ≤ baseline, memorization-probe accuracy reduced.

## 6. Compatibility matrix

Cross-reference of which ideas compose / are neutral / conflict. Legend:

- ✓ composes cleanly (benefits add)
- · neutral (independent, no interaction)
- ~ partial conflict (requires care, may dilute gains)
- ✗ conflicts (incompatible by design)

|     | A1 | A2 | A3 | A4 | A5 | B1 | B2 | B3 | B4 | B5 | C1 | C2 | C3 | C4 | C5 | D1 | D2 | D3 |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **A1** Complex MoE | — | ✓ | · | ~ | ✓ | ✓ | · | · | · | · | ✗ | · | · | ~ | · | · | · | · |
| **A2** Reversible Parcae | ✓ | — | ✓ | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ~ | ✓ | ~ | ✓ | · | · | · |
| **A3** Shared Workspace | · | ✓ | — | · | · | ✓ | ✓ | · | ✓ | ✓ | · | ~ | ✓ | · | ✓ | · | · | · |
| **A4** Path Superposition | ~ | ~ | · | — | · | ~ | · | · | ~ | · | ✗ | · | · | ~ | · | · | · | · |
| **A5** Kolmogorov + Het-MoE | ✓ | ✓ | · | · | — | ✓ | · | · | ✓ | · | ~ | · | · | · | · | · | ✓ | · |
| **B1** Hidden-State Diffusion | ✓ | ✓ | ✓ | ~ | ✓ | — | ✓ | · | ✓ | ~ | ✓ | · | ✓ | · | ✓ | · | · | · |
| **B2** Temporal-Contrastive | · | ✓ | ✓ | · | · | ✓ | — | ✓ | ✓ | ~ | ✓ | · | ✓ | · | ✓ | · | · | · |
| **B3** Forward-Forward | · | ✓ | · | · | · | · | ✓ | — | · | · | · | · | · | · | · | · | · | · |
| **B4** Entropy Conservation | · | ✓ | ✓ | ~ | ✓ | ✓ | ✓ | · | — | · | ✓ | · | ✓ | · | ✓ | · | · | · |
| **B5** Momentum Teacher | · | ✓ | ✓ | · | · | ~ | ~ | · | · | — | · | · | · | · | · | · | · | · |
| **C1** Mycelial Graph | ✗ | ~ | · | ✗ | ~ | ✓ | ✓ | · | ✓ | · | — | · | · | · | · | · | · | · |
| **C2** Cross-Example Attn | · | ~ | ~ | · | · | · | · | · | · | · | · | — | · | · | · | · | · | · |
| **C3** Self-CoT Recycling | · | ✓ | ✓ | · | · | ✓ | ✓ | · | ✓ | · | · | · | — | · | · | · | · | · |
| **C4** Hypernet Weights | ~ | ~ | · | ~ | · | · | · | · | · | · | · | · | · | — | · | · | · | · |
| **C5** Lyapunov Iteration | · | ✓ | ✓ | · | · | ✓ | ✓ | · | ✓ | · | · | · | · | · | — | · | · | · |
| **D1** Static Prefill | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | — | · | · |
| **D2** Adaptive Mix | · | · | · | · | ✓ | · | · | · | · | · | · | · | · | · | · | · | — | ✓ |
| **D3** Grad-Norm Gating | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | — |

**Notable conflicts:**

- **A1 × C1** (Complex MoE × Mycelial Graph): both redesign expert semantics, but in incompatible ways — complex combination assumes summable outputs, graph walks assume sequential application.
- **A4 × C1** (Path Superposition × Mycelial Graph): both introduce parallel/sequential structure on different axes; combining would require a 4D tensor (B, T, P, expert_in_walk) which blows up.
- **A1 × A4** (partial conflict): both use the complex-plane metaphor. A combined version could work but requires careful design to avoid double-counting phase information.

**Strongest compositions:**

- **A2 × B1 × B4**: Reversible Parcae + Hidden-State Diffusion + Entropy Conservation. All iteration-axis enhancements; memory-efficient, noise-regularized, entropy-converging. Could train 6-iteration Parcae with formal convergence properties.
- **A3 × B2 × B4**: Workspace + Temporal-Contrastive + Entropy Conservation. Three different flavors of "iterations should produce meaningful-but-convergent trajectories."
- **A5 × D2**: Heterogeneous-capacity experts + adaptive data mix. Both respond to per-token / per-domain signal; naturally complementary.

## 7. "If we could only do one" ranking

For a single v3 experiment post-v2, ranked by expected ROI (quality gain × implementability / effort):

1. **B4 Entropy Conservation** — ★★★★★ implementability, zero risk, ~2 days. Easiest win; either the regularizer helps or it doesn't, move on.
2. **B1 Hidden-State Diffusion** — ★★★★★ implementability, ~1 week. Strong theoretical motivation, clean fallback (σ=0).
3. **A2 Reversible Parcae** — the only idea that *unlocks a capability* (6+ iterations at current memory). Highest ceiling, highest risk.
4. **A3 Shared Workspace** — ~2-3 weeks, well-motivated, clean architecture.
5. **A1 Complex MoE** — intellectually interesting, requires top-k > 1 (adds complexity).
6. **B2 Temporal-Contrastive** — solid aux-loss bet.
7. **B5 Momentum Teacher** — throughput cost is the main concern.
8. **C3 Self-CoT Recycling** — natural precursor to TTC adoption.
9. **A5 Kolmogorov + Heterogeneous MoE** — infrastructure-heavy.
10. **B3 Forward-Forward** — novel paradigm, unclear payoff.
11. **D2 Adaptive Mix** — meta-level optimization.
12. **D3 Grad-Norm Gating** — specific remedy for specific symptoms.
13. **D1 Static Prefill** — inference-only optimization.
14. **C4 Hypernet Weights** — parameter explosion risk.
15. **C5 Lyapunov Iteration** — theoretical elegance, unclear practical gain.
16. **C2 Cross-Example Attention** — ahead-of-time for our scale.
17. **C1 Mycelial Graph** — high novelty, replaces established routing.
18. **A4 Path Superposition** — most magical, highest risk.

**My recommendation for first v3 experiment:** B4 + B1 together (they compose cleanly). ~1.5 weeks combined. Low risk, measurable signal, nice research story ("iterations as denoising with entropy conservation").

## 8. Dream-stack composition

If we had unlimited budget and wanted to build the most novel Parcae + MoE architecture that composes cleanly:

**Architecture.**

- **Backbone:** Reversible Parcae (A2) with `mean_recurrence = 6` (enabled by the memory savings)
- **Experts:** Heterogeneous-capacity MoE with Kolmogorov routing (A5)
- **Residual augmentation:** Shared Latent Workspace (A3) with K=24 slots
- **Routing policy:** R2 sticky (from v2), with permutation caching (T7 from v2)

**Training.**

- **Noise:** Hidden-State Diffusion (B1) with schedule `[0.3, 0.25, 0.2, 0.15, 0.1, 0.0]` across 6 iterations
- **Regularization:** Entropy Conservation loss (B4) at λ = 0.05
- **Self-supervision:** Temporal-Contrastive Iteration (B2) at λ = 0.1
- **Self-chaining:** Self-CoT Recycling (C3) with p = 0.1 in late training

**Meta.**

- **Data mix:** Adaptive Mix Curriculum (D2)
- **Outlier handling:** Grad-Norm Gating (D3)

**Expected cost vs v2 baseline.**

- Compute: ~2× (more iterations via reversibility, teacher-less)
- Memory: ~1.2× (reversibility nearly offsets 6 iterations)
- Wall-clock: ~2.5× due to compute + contrastive overhead

**Expected benefits.**

- 6 iterations of refinement (vs 3 in v2) → meaningfully deeper iterative reasoning
- Adaptive token-capacity matching via Kolmogorov routing
- Sequence-level structure via workspace
- Noise-regularized, entropy-converging training curriculum
- Self-chaining at inference for TTC

**Research novelty.** Very high — this composition has no direct analogue in the literature. Could anchor a paper.

**Realistic caveat.** This stack is intentionally ambitious. Most features would need to validate standalone before combining. A real rollout would be: validate B4 + B1 together (easiest combo) → add A2 (highest unlock) → add A3 (solid extension) → only then consider A5, B2, B3, etc.

## 9. Cross-cutting open questions

1. **Scale-appropriateness:** most ideas come from papers at 100M+ scale. Do they transfer to 80M? Hard to predict; our ablation protocol at 40M proxy gives signal.
2. **Interaction with fp16:** many ideas introduce new ops that might stress fp16 dynamic range. Each risk-register entry flags this where applicable.
3. **Training time budget:** aggressive stacking doubles or triples training time. Need explicit budget for v3 experiments (~1-2 weeks of DDP time per experiment).
4. **Reproducibility:** many of these ideas are fresh enough that external validation is minimal. First-mover in documenting results.
5. **Prioritization cadence:** ideas should be re-ranked quarterly based on new literature and internal experiment results.

## 10. Related docs

- `docs/superpowers/specs/2026-05-07-frankenmoe-loop-design.md` — FrankenMoE-Loop v2 spec (the architecture these ideas extend or alternate)
- `knowledge/architectures/looped_moe_design_2026.md` — generalizable findings on looped MoE
- `knowledge/architectures/cookbook.md` — primitive library (ensures we're not duplicating)
- `knowledge/architectures/hypothesis_buildout_results.md` — our earlier 13-architecture shootout
- `knowledge/architectures/parcae_stable_looped_models.md` — Parcae reference
- `knowledge/architectures/paper_deep_dive_2026_05.md` — literature contextualization
- `knowledge/architectures/small_lm_arch_interventions_2026.md` — established architectural interventions
- `knowledge/training/zaya1_8b_findings_2026.md` — ZAYA1 applied findings synthesis
