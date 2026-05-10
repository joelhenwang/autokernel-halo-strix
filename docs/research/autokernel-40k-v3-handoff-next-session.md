# v3 40k Throughput Campaign — Comprehensive Handoff for Next Session

**Version:** 1.0
**Created:** 2026-05-11
**Author:** Previous session agent (final commit `e9fae71` on `main`)
**For:** Next agent taking over after session compaction.
**Nature:** Self-contained. Assumes no prior context. Reads top-to-bottom as an executable plan.

---

## 0. READ FIRST — critical ground rules

**These rules override any inferred intent. Violate at your peril.**

1. **NEVER launch `scripts/launch_sprint3a.sh` or `scripts/launch_sprint3b.sh` without explicit user approval.** The user said verbatim: "From now on, never launch Sprint 3A or 3B without my saying first." This applies for all future sessions. The full campaign STOPS before the sprint launch; the scripts are prepared but not invoked.

2. **Baseline tok/s is 31,331 for OdinFlat on Sprint 3A-confirm recipe.** All throughput comparisons use this number.

3. **Hardware constraints (non-negotiable, from `CONSTRAINTS.md`):**
   - gfx1151 / Strix Halo / RDNA 3.5. **No MFMA.**
   - **bf16 is NOT supported** (24% slower + compile crashes). Always fp16 + GradScaler.
   - wave32 not wave64. LDS 64 KB/CU. L2 6 MB. Peak mem BW ~240-256 GB/s per machine.
   - `TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` required for `accum_steps > 1`.
   - DDP uses `gloo` (not NCCL) over Thunderbolt 4 — gloo matches RCCL on unified memory.

4. **Two-machine DDP topology:**
   - Machine A: `joelwang-ai-2@10.77.0.1` (also `@192.168.1.140` on LAN for scp). Venv: `~/Desktop/ai_lab/autokernel-halo-strix/.venv`. Clean, dedicated, no aiter.
   - Machine B: `joelwang-ai-1@10.77.0.2` (`@192.168.1.145`). Venv: `~/Desktop/comfyui-rocm7.12/.venv`. Has aiter (don't use for training).
   - TB4 interconnect via `thunderbolt0` interface, subnet `10.77.0.0/24`.
   - All DDP launches from Machine A; it SSHs to Machine B for rank 1.

5. **Sync commands before running:**
   - `bash sync_remote.sh` — sync to Machine A (required before every DDP or remote run on A)
   - `bash sync_remote_b.sh` — sync to Machine B (required before DDP)
   - `bash run_remote.sh "..."` — run command on Machine A
   - `bash run_remote_b.sh "..."` — run command on Machine B
   - **NEVER use raw SSH.** Always use these scripts.

6. **Current campaign state at session start:**
   - Commit: `e9fae71` on `main`
   - T-1.5 fused-zloss 2000-step DDP run is **RUNNING** in background on both machines. See §5 for how to check.
   - Stack A is validated at **+7.7% tok/s (33.7k)** at step 300-600 of T-1.5. Canonical gate at step 2000 is in progress.

---

## 1. What this document is

A comprehensive, self-contained execution plan for the remaining ~11-21 hours of work in the v3 40k throughput campaign.

If you have just been handed this document after a context reset:
- The **`docs/research/autokernel-40k-v3-execution-plan.md`** is the strategic plan (single source of truth for phase structure, decision gates, telemetry schemas).
- **This document** is the tactical/operational playbook for what to execute next. It duplicates critical context so you don't need to re-read every other file.
- **`docs/perf/v3-session-execution-log.md`** summarizes what the previous session did.

You are executing the continuation of a 6-round dialogue between engineering and an external research engineer. v1 (guide) → our response → v2 (revised guide) → our v3 addendum → v3 response (final) → execution. All six documents are in `docs/research/`.

---

## 2. Current state snapshot

### 2.1 Commit graph at handoff

```
e9fae71 Session closeout: T-6 launch scripts + T-1.5 preliminary findings + STATUS
22ce520 T-2.1 batch=32 probe + T-4 compiled autograd smoke + CA activation
b498816 T-3.2: fused_rope_gate_mul custom_op with register_autograd
124b18e Fix T-0.2 param_names collision with PyTorch optimizer key
ac59cf9 T-1.1 parity VERIFIED + T-0.7 dtype inventory
37c16c6 T-1.1 + T-1.2 + T-1.4: branchless SPECTRA, deferred loss sync
e60b5df T-0.6/7/8: DDP allreduce trace + dtype inventory + sync counter
ded406a T-0.5 + T-0.2: granular --ak-* flags + NorMuon telemetry
c44eb00 Lock 40k throughput execution plan + STATUS framing (post-v3)
98f2f39 v3 request addendum to research engineer + research brief expansion
664fd20 T-0 read-only findings: backward breakdown + sync audit + graph-breaks
08ee1b5 Engineering response to 40k throughput guide
4a367ea Research brief: autokernel stability on AMD Strix Halo
```

14 commits shipped this session. `main` is pushed to `origin`.

### 2.2 What's running in background

**T-1.5 DDP run** — as of handoff:
- Directory: `checkpoints/t1-5-fused-zloss/` (on Machine A)
- At step 600/2000 (as of session end), **+7.7% tok/s** sustained (33,656 vs 31,331 baseline)
- Loss trajectory: step 50 = 9.37, step 100 = 5.83, step 300 = 4.49, step 600 = 3.87 (healthy)
- DDP trace confirms `allreduce_count=1` per opt step (no_sync correct)
- No GradScaler events, no frozen params warning
- **Estimated completion: ~2 hours after session end** (1400 more steps at ~50/3min)

**How to check T-1.5 progress:**
```bash
bash run_remote.sh "grep 'step' ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/t1-5-fused-zloss/rank0.log | tail -10"
bash run_remote.sh "ls ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/t1-5-fused-zloss/step_*.pt 2>/dev/null"
bash run_remote.sh "ps aux | grep train_ddp | grep -v grep | head -2"
```

**If T-1.5 has NOT finished when you resume:**
- Wait for it (it's the canonical 2000-step gate). Do Phase A engineering meanwhile.

**If T-1.5 has finished:**
- Extract final scorecard, lock Stack A number, and begin Phase B immediately.

**If T-1.5 failed or crashed mid-run:**
- Check `checkpoints/t1-5-fused-zloss/rank0.log` for errors
- Relaunch per §6.A.4 instructions below
- Don't skip — T-1.5 result is needed for canonical Stack A gate

### 2.3 What's locked and shipped

| Item | Status | Artifact |
|---|---|---|
| v3 execution plan | LOCKED | `docs/research/autokernel-40k-v3-execution-plan.md` |
| STATUS.md 40k framing | LOCKED | `STATUS.md` top section |
| 28 granular `--ak-*` flags | SHIPPED | `scripts/train_ddp.py` |
| NorMuon per-param telemetry | SHIPPED | `halo_training/normuon.py` |
| DDP allreduce trace | SHIPPED + VERIFIED | `scripts/train_ddp.py` _DDP_TRACE_STATE |
| Dtype inventory tool + results | SHIPPED | `scripts/autokernel_dtype_inventory.py`, `docs/perf/dtype-autocast-inventory.md` |
| Sync counter in profiler | SHIPPED | `scripts/train_ddp.py` |
| Branchless SPECTRA | SHIPPED + PARITY VERIFIED | `halo_training/spectra.py`, `halo_training/normuon.py`, `scripts/test_spectra_branchless.py` |
| Deferred loss sync | SHIPPED | `scripts/train_ddp.py` (`--ak-sync-cleanup`) |
| Fused z-loss validation | RUNNING | T-1.5 DDP run |
| T-3.2 fused_rope_gate_mul fix | SHIPPED + 3/3 PARITY PASS | `kernels/hip/_torch_ops.py`, `scripts/test_fused_rope_gate_mul_custom_op.py` |
| Sprint 3A/3B launch scripts | SHIPPED (not launched) | `scripts/launch_sprint3a.sh`, `scripts/launch_sprint3b.sh` |
| Probe scripts (bucket, batch32, CA) | SHIPPED (not run) | `scripts/probe_t1_ddp_bucket_sweep.sh`, `scripts/probe_t2_1_batch32.sh`, `scripts/probe_t4_compiled_autograd.sh` |

### 2.4 Key empirical findings to remember

1. **Fused zloss delivers +7.7% tok/s** (T-1.5 measurement at step 300-600, stable). Within v3's 5-8% prediction.
2. **0/7 training-path custom ops have `register_autocast` rules** (T-0.7 inventory). This confirms v3's H11 hypothesis (custom-op autocast boundary mismatch) as a concrete gap. Phase C.A.3 below fixes this.
3. **fused_rope_gate_mul had a pre-fix silent-freeze bug.** Pre-fix: `@torch.compiler.disable` wrapper with no `register_autograd` severed grads to b/h_tilde. Now fixed via `autokernel::fused_rope_gate_mul` custom_op. Verified: b.grad and h_tilde.grad populate, parity within 5e-3 rel_err.
4. **DDP `no_sync` is correct.** `allreduce_count=1` per opt step observed over 148 steps of T-1.5. No accumulation communication bug.
5. **Backward is near-theoretical** (2.18× forward for 2× FLOPs). Compiled autograd realistic upside: 2-8%, not 5-15%.
6. **Only 2 distinct graph-break locations** pre-T-3.2: rope_gate_mul (FIXED) + causal_conv1d (Phase A.1 target).
7. **87 `.item()` calls per opt step** (~50 from SPECTRA, ~8 from chunked CE, ~3 from logging). SPECTRA branchless fix eliminates ~50/step.
8. **Hardware ceiling analysis:** summed upper-bound stacks reach 35% throughput gain max → 40k is aspirational, 35-38k is honest realistic ceiling.

### 2.5 Locked decisions from user (DO NOT re-ask)

From the planning dialogue:
- **Sprint 3A/3B launch timing:** "Never launch without my saying first" (universal rule).
- **T-5 hidden kernel recovery:** "Run unconditionally" (all C.1-C.4 sub-probes regardless of intermediate failures).
- **T-3.3 scope:** "Full Tier 2" — 4-op parity matrix AND register_autocast rule additions on all 5 training-path ops.
- **T-3.2 causal_conv1d shim:** "Write the shim" (full engineering, not deferred).
- **Session scope:** "One long session" (8-22h, everything except sprint launches).
- **Phase C failure handling:** "Run all C.1-C.4 regardless" (no early abort).
- **Throughput target:** 40k nominal, 36k engineering success, 38k strong, 40k stretch (per v3 §5.3 STATUS language).

---

## 3. Project background (minimum required context)

The autokernel-halo-strix project trains custom hybrid LMs (OdinFlat, OdinHalo, others) on AMD Strix Halo APUs (gfx1151) across two networked machines via Thunderbolt 4 / DDP/gloo. A previous `--optimize-kernels` feature silently froze 23% of OdinFlat's parameters because raw pybind HIP calls returned tensors with `grad_fn=None`. Phase B remediation (shipped 2026-05-11) rewired all HIP kernels through `torch.library.custom_op + register_autograd` or native PyTorch fallback. But the post-fix optimized path now diverges (Phase C at step 250, Phase G at step 750), revealing a genuine fp16 stability boundary.

The v3 throughput campaign addresses two things simultaneously:
1. **Short-term:** extract +5-10% safe throughput from sync cleanup, loss fusion, DDP tuning, batch=32, NorMuon implementation optimization, and compiled autograd (Stacks A/B/C).
2. **Long-term:** attempt recovery of the optimized hidden-kernel path through delayed enable, w_gate_up staging, post-NorMuon trust caps, dtype/autocast audit (Stack D).

Current production locks `--optimize-kernels` OFF; sprint runs (Sprint 3A OdinFlat ~61h, Sprint 3B OdinHalo ~77h) are queued awaiting a better stack.

### Models at play

- **OdinFlat:** 122M params, 14 layers, 768 hidden, 12 heads, 4 KV heads (GQA), hybrid (2 attention layers + 12 HyPEShortConvBlock conv layers), block=512, SwiGLU FFN 2×2048 hidden.
- **OdinHalo:** 58M unique / ~156M effective looped, 6 shared layers × 3 iterations with Parcae injection, lower LR (`lr_2d=2e-3`), requires `--polar-ns`.

### Training recipe (Sprint 3A-confirm, baseline)

```
--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --ema --auto-eval
# Effective batch = 16 × 512 × 8 × 2 = 131,072 tokens/opt-step
# Baseline ~31,331 tok/s aggregate
```

### Datasets

- `datasets/dolma-10b-odin32k.bin` — 6.9B tokens, 13.7 GB, on BOTH machines. Production.
- `datasets/wikitext-103-odin32k.bin` — smaller, for smoke tests.
- BabyLM — use `--smoke` or `--dataset babylm`.

### Tokenizer

- `tokenizers/vidar-32k/tokenizer.json` — 32K vocab. EOS=0 (not 50256).

---

## 4. File index (read these if stuck)

**MUST-READ before starting:**
- `STATUS.md` — current state, 40k framing, locked decisions
- `AGENTS.md` — training workflow, hardware constraints, kernel authoring rules
- `CONSTRAINTS.md` — 28-item machine-parseable checklist (autograd-safety, fp16-only, etc.)
- `docs/research/autokernel-40k-v3-execution-plan.md` — **THE strategic plan** (schemas, flag taxonomy, interpretation table)

**Strongly recommended:**
- `docs/research/autokernel_40k_v3_research_engineering_response.md` — v3 response (what research engineer said, why priorities changed)
- `docs/research/autokernel-stability-research-brief.md` — full 14K-word stability context (H1-H18 hypotheses, Phase C/G divergence analysis)
- `docs/perf/v3-session-execution-log.md` — previous session summary

**Reference as needed:**
- `docs/perf/backward-breakdown.md` — T-0.1 findings (backward is near-theoretical)
- `docs/perf/sync-point-audit.md` — T-0.3 findings (87 syncs/step inventory)
- `docs/perf/graph-breaks-inventory.md` — T-0.4 findings (2 break locations)
- `docs/perf/dtype-autocast-inventory.md` — T-0.7 findings (0/7 have register_autocast)
- `docs/perf/t1-5-fused-zloss-preliminary.md` — T-1.5 preliminary at step 300
- `docs/perf/odinflat-step-profile.md` — baseline step profile
- `docs/perf/triton-swiglu-ship-gate-bench.md` — Phase I result (both kernels at bandwidth ceiling)
- `docs/perf/phase-c-final-analysis.md` — Phase C/G divergence detail

**Key code:**
- `scripts/train_ddp.py` — DDP trainer. 2000+ lines. Contains all --ak-* flags, telemetry, DDP trace, profiler, StabilityGuard, autokernel preflight.
- `scripts/launch_ddp.sh` — DDP launcher (SSHs to Machine B for rank 1, detaches both via setsid+nohup)
- `halo_training/normuon.py` — NorMuon optimizer with v3 telemetry + trust cap + w_gate_up staging + branchless SPECTRA
- `halo_training/spectra.py` — SPECTRA post-clip (branchless path activated by env `AUTOKERNEL_SPECTRA_BRANCHLESS=1`)
- `halo_training/optimizer.py::build_imu1_optimizer` — entry point that threads all v3 kwargs into NorMuon
- `halo_training/mup.py::build_mup_param_groups` — μP 3-way split (embedding/hidden/readout)
- `kernels/hip/_torch_ops.py` — all torch.library.custom_op registrations (8 ops including T-3.2 fused_rope_gate_mul fix)
- `models/components/conv_blocks.py` — HyPEShortConvBlock (has AUTOKERNEL_FIX_ROPE_GATE env gate)
- `models/odin_flat.py::compile_zones` — per-layer torch.compile wrapper
- `models/odin_halo.py::compile_zones` — looped-model per-layer compile

**Scripts prepared but NOT run:**
- `scripts/probe_t1_ddp_bucket_sweep.sh` — T-1.4
- `scripts/probe_t2_1_batch32.sh` — T-2.1
- `scripts/probe_t4_compiled_autograd.sh` — T-4
- `scripts/launch_sprint3a.sh` — T-6 Sprint 3A (DO NOT RUN)
- `scripts/launch_sprint3b.sh` — T-6 Sprint 3B (DO NOT RUN)

**Tests (all passing where executed):**
- `scripts/test_spectra_branchless.py` — 12/12 PASS on Machine A (2026-05-11)
- `scripts/test_fused_rope_gate_mul_custom_op.py` — 3/3 PASS on Machine A (2026-05-11)
- `scripts/test_phase_b_autograd_safety.py` — 7 autograd-safety regression tests (pre-session)
- `scripts/test_triton_swiglu.py` — 5 Triton parity tests (pre-session)

---

## 5. v3 flag taxonomy (complete reference)

All 28 flags default to OFF/baseline. Pass via `EXTRA_FLAGS=` env var in `launch_ddp.sh` or directly to `scripts/train_ddp.py`.

### 5.1 Core granular kernel flags (`--ak-*`)

| Flag | Effect | Current wired? |
|---|---|---|
| `--ak-loss-ce` | Route logits through `kernel.ce_full` | Partial (reserved) |
| `--ak-loss-zloss` | Alias for `--use-fused-zloss` (bakes z-loss into fused CE via Phase B.5) | YES |
| `--ak-swiglu-fwd` | HIP silu_gate_mul forward | Reserved (patterns exist) |
| `--ak-swiglu-bwd` | HIP silu_gate_mul backward vs PyTorch fallback | Reserved |
| `--ak-rmsnorm` | HIP rmsnorm custom_op | Reserved |
| `--ak-res-rmsnorm` | HIP fused_res_rmsnorm custom_op | Reserved |
| `--ak-rope` | HIP rotary_emb_fp32 custom_op | Reserved |
| `--ak-rope-gate` | HIP fused_rope_gate_mul (legacy silent-freeze path) | Reserved — prefer --ak-fix-rope-gate-op |
| `--ak-causal-conv` | DaoAILab causal_conv1d_fn (needs shim from Phase A.1) | Reserved |
| `--ak-qkv` | Fused QKV custom_op | Reserved |
| `--ak-ple-gate` | HIP fused_ple_gate | Reserved |
| `--ak-compiled-autograd` | `torch._dynamo.config.compiled_autograd = True` | YES |
| `--ak-triton-visible` | Route Triton through `torch.library.triton_op` | Reserved |
| `--ak-sync-cleanup` | Branchless SPECTRA + deferred loss sync (aliases below) | YES |
| `--ak-ddp-tune` | `gradient_as_bucket_view=True` + tuned bucket_cap_mb | Partial (env var) |

### 5.2 v3 add-on flags

| Flag | Effect | Current wired? |
|---|---|---|
| `--ak-spectra-branchless` | Sets `AUTOKERNEL_SPECTRA_BRANCHLESS=1` | YES |
| `--ak-autocast-tier {none,tier1,all}` | Autocast rule application tier | Reserved (Phase A.3 activates) |
| `--ak-dtype-trace` | Emit dtype trace JSONL | Reserved (not hot-pathed yet) |
| `--ak-fix-rope-gate-op` | Sets `AUTOKERNEL_FIX_ROPE_GATE=1` → uses fixed rope_gate_mul custom_op | YES |
| `--ak-causal-conv-shim` | Sets `AUTOKERNEL_CAUSAL_CONV_SHIM=1` → uses Phase A.1 shim | Reserved (Phase A.1 wires) |
| `--ak-normuon-telemetry` | Emits v3 §5.1 JSONL | YES |
| `--ak-normuon-impl-opt` | Enables branchless SPECTRA | YES (via spectra_branchless) |
| `--ak-trust-cap <float>` | Post-NorMuon trust cap (0.0 = off, 0.02 diagnostic) | YES |
| `--ak-trust-cap-scope {none,w_gate_up,spiking,all_2d}` | Cap scope | YES |
| `--ak-w-gate-up-scale <float>` | Initial w_gate_up update scale (1.0 = no staging) | YES |
| `--ak-w-gate-up-ramp-steps <int>` | Ramp to 1.0 over N steps | YES |
| `--assert-no-sync` | Abort if allreduce_count > 1 per opt step | YES |

### 5.3 Flag interaction rules

- `--ak-sync-cleanup` implies `--ak-spectra-branchless` (both activate `AUTOKERNEL_SPECTRA_BRANCHLESS=1`)
- `--ak-sync-cleanup` implies deferred loss accumulator (tensor-side running_loss)
- `--ak-normuon-impl-opt` also activates branchless SPECTRA
- `--ak-loss-zloss` implies `--use-fused-zloss`
- Env vars override flag state if set directly (escape hatch)

---

## 6. Full execution plan (Phases A → B → C → D)

Read the entire section before executing.

**Estimated total time:** 11-22 hours.
**Phases must execute in order** due to DDP cluster being single-use.

### PHASE A — Engineering during T-1.5 wait (2-5h)

**Goal:** finish T-3.2 (causal_conv1d shim), T-3.3 (Tier 2 parity + register_autocast rules), capture T-1.5 scorecard.

**Precondition:** T-1.5 is running on DDP cluster (or completed). Machine A single-GPU work does NOT conflict with T-1.5 (T-1.5 uses one rank per machine, leaves one GPU free... actually, Strix Halo has ONE GPU per machine so T-1.5 occupies it during training steps, but single-node tests take <60s so you can interleave between T-1.5 log intervals).

**CRITICAL:** If Machine A GPU is fully occupied by T-1.5, your tests will block. Pragmatic approach: write all Phase A code locally (Windows), commit, sync, then run tests either:
- After T-1.5 finishes, OR
- During rebalance moments (Machine A GPU occupancy is not 100% — dataloader idle windows exist)
- On Machine B if GPU is free there (rank 1 also runs forward; but B's GPU is used by T-1.5 rank 1)

Best strategy: **do all Phase A code/write work locally (no GPU needed), batch all tests to run after T-1.5 completes**.

#### A.1 — T-3.2 causal_conv1d shim (~4-6h)

**Goal:** wrap DaoAILab's `causal_conv1d_fn` in a proper `torch.library.custom_op` with `register_autograd`, mirroring the T-3.2 fused_rope_gate_mul fix pattern. Eliminates the 2nd of 2 graph-break locations (per T-0.4 inventory) and enables clean compiled-autograd traces through conv blocks.

**Background:** DaoAILab's `causal_conv1d_fn` is an external C++ extension. It already has its own `torch.autograd.Function` wrapper internally, so backward correctness is not at risk — the issue is purely Dynamo tracing at custom_ops.py:698 (the compile barrier). Our wrapper just needs to make Dynamo see it as a clean custom_op boundary.

**Implementation steps:**

1. Add to end of `kernels/hip/_torch_ops.py` (after the existing T-3.2 fused_rope_gate_mul section):
```python
# ---------------------------------------------------------------------------
# causal_conv1d shim — T-3.2 second-half fix
# Wraps DaoAILab's causal_conv1d_fn for clean torch.compile + compiled-autograd.
# DaoAILab's existing internal autograd.Function handles the actual backward;
# we just provide a clean torch.library.custom_op boundary for Dynamo to see.
# ---------------------------------------------------------------------------

@torch.library.custom_op("autokernel::causal_conv1d", mutates_args=())
def causal_conv1d_shim_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Forward: x is (B, D, T), weight (D, K), bias (D,). Returns (B, D, T)."""
    from causal_conv1d import causal_conv1d_fn
    return causal_conv1d_fn(x, weight, bias)


@causal_conv1d_shim_op.register_fake
def _(x, weight, bias):
    return x.new_empty(x.shape)


def _causal_conv1d_shim_setup(ctx, inputs, output):
    x, weight, bias = inputs
    ctx.save_for_backward(x, weight, bias)


def _causal_conv1d_shim_backward(ctx, grad_out):
    x, weight, bias = ctx.saved_tensors
    # Call DaoAILab's internal backward by running the autograd.Function
    # with requires_grad=True and capturing the backward via autograd.grad.
    # Safer path: implement in pure PyTorch using F.conv1d backward.
    # Since DaoAILab's autograd.Function is not directly callable here,
    # we use a native fallback that matches the forward semantics.
    import torch.nn.functional as F
    D, K = weight.shape
    # Causal conv = conv1d with padding=K-1 on left only, then crop to T.
    # Forward was: y = causal_conv1d(x, weight, bias)  where weight is per-channel
    # (grouped conv with groups=D, kernel K).
    x_req = x.detach().requires_grad_(True)
    w_req = weight.detach().requires_grad_(True)
    b_req = bias.detach().requires_grad_(True) if bias is not None else None

    # Reproduce the forward with conv1d + groups=D (depthwise).
    with torch.enable_grad():
        x_padded = F.pad(x_req, (K - 1, 0))  # left-pad for causal
        y = F.conv1d(x_padded, w_req.unsqueeze(1), b_req, groups=D)
    grads = torch.autograd.grad(y, [x_req, w_req] + ([b_req] if b_req is not None else []),
                                 grad_outputs=grad_out, retain_graph=False)
    grad_x = grads[0]
    grad_w = grads[1]
    grad_b = grads[2] if b_req is not None else None
    return grad_x, grad_w, grad_b


causal_conv1d_shim_op.register_autograd(
    _causal_conv1d_shim_backward, setup_context=_causal_conv1d_shim_setup
)
```

2. Route conv_blocks.py through the shim when env var is set. Edit `models/components/conv_blocks.py` lines 292-295 (the existing `elif _HAS_CAUSAL_CONV1D:` branch):
```python
elif _HAS_CAUSAL_CONV1D:
    # v3 T-3.2 (2nd half): use shimmed custom_op when AUTOKERNEL_CAUSAL_CONV_SHIM=1.
    import os as _os
    if _os.environ.get("AUTOKERNEL_CAUSAL_CONV_SHIM", "0") in ("1", "true", "True"):
        import kernels.hip._torch_ops  # noqa: F401 trigger registration
        z = torch.ops.autokernel.causal_conv1d(
            y.transpose(1, 2), self.conv_weight, self.conv_bias
        ).transpose(1, 2)
    else:
        z = causal_conv1d_fn(
            y.transpose(1, 2), self.conv_weight, self.conv_bias
        ).transpose(1, 2)
```

3. Wire the flag in `scripts/train_ddp.py` after the existing rope-gate env var block (around the line where `AUTOKERNEL_FIX_ROPE_GATE` is set):
```python
if getattr(args, "ak_causal_conv_shim", False):
    os.environ["AUTOKERNEL_CAUSAL_CONV_SHIM"] = "1"
```

4. Write `scripts/test_causal_conv1d_shim.py` parity test (mirror `test_fused_rope_gate_mul_custom_op.py`):
   - Test 1 forward parity vs DaoAILab's direct call
   - Test 2 gradient flow (y.grad_fn is not None, x.grad populates, weight.grad populates)
   - Test 3 backward numerical parity vs pure-PyTorch conv1d reference (tolerance 5e-3 fp16)
   - Shapes: B=2, D=1536, T=128 (production-similar), also B=4, D=256, T=64 for regression.

5. Run test on Machine A after T-1.5 finishes:
```bash
bash sync_remote.sh
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && source .venv/bin/activate && export HSA_OVERRIDE_GFX_VERSION=11.5.1 && python scripts/test_causal_conv1d_shim.py"
```

6. Commit: `T-3.2 (2nd half): causal_conv1d shim with custom_op + register_autograd`

**Failure mode to watch for:** the pure-PyTorch conv1d fallback may be slightly slower than DaoAILab's fused kernel. This is ACCEPTABLE because backward speed matters less than compiled-autograd cleanness. If you find the shim makes training detectably slower (>2%), flag it and ask user.

**If DaoAILab's internal autograd.Function can be called directly** (check `from causal_conv1d import _causal_conv1d_fn_fn` or similar private API), prefer that to the pure-PyTorch fallback for better backward speed. Check `python -c "import causal_conv1d; print(dir(causal_conv1d))"` on Machine A.

#### A.2 — T-3.3 Tier 2 deep parity tests (~2h)

**Goal:** verify 4 training-path custom ops produce numerically-correct forward + backward under fp16 + autocast vs native PyTorch reference. Informs T-5 whether dtype drift is a real mechanism.

**Implementation:** Write `scripts/test_tier2_parity.py` with 4 test functions:

```python
def test_silu_gate_mul_parity():
    # Reference: F.silu(gate) * up under torch.autocast
    # Custom: torch.ops.autokernel.silu_gate_mul(gate, up) under same autocast
    # Check: forward rel_err < 5e-3, grad_gate rel_err < 5e-3, grad_up rel_err < 5e-3
    # Shapes: (16*512, 2048) production; (4*64, 256) smoke

def test_rmsnorm_parity():
    # Reference: F.rms_norm(x, (D,), weight, eps)
    # Custom: torch.ops.autokernel.rmsnorm(x, weight)
    # Shapes: (16*512, 768); (4*64, 256)

def test_fused_res_rmsnorm_parity():
    # Reference: x + residual, then F.rms_norm
    # Custom: torch.ops.autokernel.fused_res_rmsnorm(x, residual, weight)

def test_causal_conv1d_parity():
    # Reference: DaoAILab causal_conv1d_fn direct
    # Custom: torch.ops.autokernel.causal_conv1d (from A.1)
```

Each test uses `torch.manual_seed(42)`, fp16 dtype by default, on CUDA device. Both gradcheck-style gradient parity checks AND forward parity.

Run on Machine A:
```bash
bash sync_remote.sh
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && source .venv/bin/activate && export HSA_OVERRIDE_GFX_VERSION=11.5.1 && python scripts/test_tier2_parity.py"
```

Expected: 4/4 PASS. If any fails with rel_err > 5e-3, investigate before proceeding to T-5. Common failure: autocast dtype mismatch (caught by A.3 next).

#### A.3 — T-3.3 register_autocast rules on all 5 training-path ops (~1-2h)

**Goal:** fix v3's H11 (0/7 ops have register_autocast). Add explicit autocast rules so each op handles fp16 AMP correctly.

**Implementation:** Edit `kernels/hip/_torch_ops.py` to add `register_autocast` calls for these 5 ops:
1. `autokernel::silu_gate_mul`
2. `autokernel::rmsnorm`
3. `autokernel::fused_res_rmsnorm`
4. `autokernel::rotary_emb_fp32` (note: already fp32-internal per name, may need different treatment)
5. `autokernel::fused_rope_gate_mul` (from T-3.2 fix)
6. `autokernel::causal_conv1d` (from A.1)

For each, add after the existing `register_autograd` line:
```python
silu_gate_mul_op.register_autocast(
    "cuda", torch.float16
)
```

The semantics: when called under `torch.amp.autocast(device_type="cuda", dtype=torch.float16)`, the op receives fp16-cast inputs and runs with autocast disabled internally (standard PyTorch autocast pattern).

**Special case for `rotary_emb_fp32`:** this op's forward EXPECTS fp32 input (it does fp32 sincos math). Its `register_autocast` should cast to fp32 (not fp16) to preserve precision:
```python
rotary_emb_fp32_op.register_autocast(
    "cuda", torch.float32  # deliberately fp32 — the name says it all
)
```

**After adding all rules, re-run T-0.7 inventory to confirm:**
```bash
bash sync_remote.sh
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && source .venv/bin/activate && python scripts/autokernel_dtype_inventory.py --output docs/perf/dtype-autocast-inventory-post-A3.md --jsonl docs/perf/dtype-autocast-inventory-post-A3.jsonl"
```
Expected: now shows `6/7 have register_autocast` (the 7th is `fused_ple_gate` which may not be in training path).

**Then re-run Tier 2 parity tests under autocast:** Add to `scripts/test_tier2_parity.py` variants that wrap the reference in `torch.autocast("cuda", torch.float16)` and the custom_op call in the same. Expected: parity still holds (rel_err < 5e-3). If autocast changes the parity result, the autocast rule is wrong and needs revision.

Commit: `T-3.3: Tier 2 parity tests + register_autocast rules on 5 training-path ops`

#### A.4 — Monitor T-1.5 + capture scorecard

**Check-in cadence:** every 10-15 min using:
```bash
bash run_remote.sh "grep 'step' ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/t1-5-fused-zloss/rank0.log | tail -5"
```

**When T-1.5 reaches step 2000:**
1. Verify final checkpoint exists: `ls checkpoints/t1-5-fused-zloss/step_2000.pt`
2. Run eval scorecard:
```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && source .venv/bin/activate && EVAL_MACHINE=a python scripts/eval_checkpoint.py --checkpoint checkpoints/t1-5-fused-zloss/step_2000.pt --model models/odin_flat.py --class-name OdinFlat"
```
3. Pull scorecard JSON back to local:
```bash
scp joelwang-ai-2@192.168.1.140:~/Desktop/ai_lab/autokernel-halo-strix/docs/perf/eval-scorecards/*.json docs/perf/eval-scorecards/
```
4. Update `docs/perf/t1-5-fused-zloss-preliminary.md` to promote from "preliminary" to "final": rename file to `docs/perf/t1-5-fused-zloss-final.md`, add step-2000 numbers, add scorecard summary.
5. Commit: `T-1.5 FINAL: 2000-step canonical gate PASS + scorecard`

**Stop-and-ask if:**
- Final loss at step 2000 is >0.1 worse than Sprint 3A-confirm baseline (4.2 ± 0.05 expected)
- Any NaN / scaler collapse / frozen param warning in the log
- Scorecard regresses on any eval metric by >5%

**Otherwise:** proceed to Phase B.

### PHASE B — DDP probes (3-5h sequential)

**Goal:** run the 3 DDP probes that require the 2-machine cluster. Each is 30min-2h. All sequential (cluster is single-use).

**Precondition:** T-1.5 has released the cluster (step 2000 reached, both ranks exited cleanly).

**Safety check before EACH probe:**
```bash
bash run_remote.sh "ps aux | grep -E 'train_ddp|torchrun' | grep -v grep"
bash run_remote_b.sh "ps aux | grep -E 'train_ddp|torchrun' | grep -v grep"
```
Both machines must show NO train_ddp processes running. If they do, wait or investigate.

#### B.1 — T-1.4 DDP bucket sweep (~1-2h)

**Command:**
```bash
bash sync_remote.sh
bash sync_remote_b.sh
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && bash scripts/probe_t1_ddp_bucket_sweep.sh 2>&1 | tee -a docs/perf/t1-ddp-bucket-sweep/session.log"
```

The script runs 4 configs (bucket_cap_mb ∈ {8, 25, 50, 100}) × 150 steps each. Writes median tok/s per config to `docs/perf/t1-ddp-bucket-sweep/results.txt`.

**Expected outcome:** all 4 configs produce similar tok/s (33-34k range). DDP overlap in this repo uses manual allreduce (not DDP hooks), so bucket_cap_mb mainly affects the `gradient_as_bucket_view=True` memory reduction. If any config stands out as >1% better, lock it.

**Ship decision:**
- All within 1%: no change (lock default 25 MB). Add `gradient_as_bucket_view=True` regardless.
- One >1% better: lock that bucket size. Update `scripts/launch_sprint3a.sh` env vars.

Commit: `T-1.4: DDP bucket sweep results + lock decision`

**Write findings:** `docs/perf/t1-4-ddp-bucket-sweep-findings.md`

#### B.2 — T-2.1 batch=32 probe (~1-2h)

**Command:**
```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && bash scripts/probe_t2_1_batch32.sh 2>&1 | tee -a docs/perf/t2-1-batch32-probe/session.log"
```

4 configs × 200 steps:
1. baseline batch=16 accum=8 (control)
2. batch=32 accum=4 plain
3. batch=32 accum=4 + `--ak-ddp-tune`
4. batch=32 accum=4 + `--ak-ddp-tune --ak-sync-cleanup`

**Expected outcome per v3 prior sweep:** batch=32 gives +5% but doubles memory to ~10 GB/node. With `gradient_as_bucket_view=True`, memory may fit comfortably.

**Ship decision:**
- batch=32 fits AND +3% tok/s stable: **Stack B = A + batch=32**. Lock `BATCH=32 ACCUM=4` in `scripts/launch_sprint3a.sh`.
- batch=32 OOMs: **Stack B = A only**. Skip batch upgrade.
- batch=32 fits but throughput flat or worse: don't upgrade.

Commit: `T-2.1: batch=32 probe results + Stack B decision`

**Write findings:** `docs/perf/t2-1-batch32-findings.md`

#### B.3 — T-4 compiled autograd gated smoke (~1-2h)

**Command:**
```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && bash scripts/probe_t4_compiled_autograd.sh 2>&1 | tee -a docs/perf/t4-compiled-autograd-smoke/session.log"
```

4 configs × 300 steps:
1. baseline (no CA)
2. + compiled autograd
3. + CA + fused zloss
4. + CA + DDP tune

**Expected per T-0.1 data:** 1-4% realistic, 4-6% good, 6-8% exceptional, >8% unlikely. Primary risk: DDP overlap regression.

**Ship decision (gate from v3 §3.3):**
- ≥2.5-3% net DDP tok/s AND no overlap regression AND no recompile storm (>5 recompiles after warmup): **Stack C = B + CA**. Follow-on: run full 2000-step T-4 gate.
- Fails any condition: CA ships as infrastructure only. Stack C = Stack B.

Commit: `T-4: compiled autograd gated smoke results + Stack C decision`

**Write findings:** `docs/perf/t4-compiled-autograd-findings.md`

**Known limitation:** DDP trace `allreduce_total_ms` and `overlap_ratio_estimate` currently always show 0 because the repo does manual allreduce (only 1 call per opt step, not multiple). This doesn't affect the CA gate correctness — tok/s comparison is the primary signal. Mention this in findings doc.

**After B completes:** you have Stack A/B/C locked. Proceed to Phase C.

### PHASE C — T-5 hidden kernel recovery (5-10h, ALL sub-probes unconditional)

**Goal:** attempt recovery of the `--optimize-kernels` stack through delayed enable, warm-start, post-NorMuon trust caps, w_gate_up staging, and the T-3.2 custom_op fixes. User directed: **run C.1-C.4 unconditionally** regardless of intermediate failures.

**Precondition:** Phase A complete (shims + tests + autocast rules landed), Phase B complete.

**Safety mandate (v3 §13):**
- Frozen param audit MUST be clean (use `--diag-frozen-params`)
- GradScaler collapse → immediately STOP and forensics
- Any divergence before step 500 → STOP and replay
- Loss @ 2000 must be within 0.15 of baseline (tight parity)

#### C.0 — Replay bundle infrastructure (prerequisite, ~2-3h)

**Goal:** build `scripts/replay_step.py` that can re-run a single failing batch under 4-6 configurations for diagnosis.

**Pre-failure dump (add to StabilityGuard trigger path in `scripts/train_ddp.py`):**
When NaN / scaler collapse / nonfinite grad fires, dump:
```
<ckpt_dir>/replay-bundle-step-<N>/
  batch.pt          # input_ids, targets, doc_ids
  model_state.pt    # BEFORE the failing step
  optim_state.pt    # optimizer + scaler
  rng.pt            # torch + numpy RNG states
  config.json       # all --ak-* flags + compile mode + git SHA
  activation_stats.jsonl  # last N activation samples
```

**Replay script `scripts/replay_step.py`:**
```python
# Usage: python scripts/replay_step.py --bundle <path> --config <native|native_fused_zloss|opt_torch_bwd|opt_hip_bwd|lower_wgu|trust_cap>
# Loads bundle, overrides config flags, re-runs forward+backward+optimizer step.
# Emits: loss, grad_norm, scaler behavior, per-param update delta vs saved.
```

Write this if C.1 fails (to diagnose). If C.1 passes, defer replay infrastructure to future session.

**Abbreviated C.0:** if time is tight, skip full replay infra and build only the bundle DUMP path (not the replay script). A future session can add replay logic.

Commit: `T-5.C.0: replay bundle dump infrastructure`

#### C.1 — Warm-start matrix (~2-3h, 4 configs × 500-1000 steps each)

**Goal:** test v3 H14 (optimizer state mismatch) + warmup-local instability hypothesis.

**4 configs:**

1. **C.1.a** — native 500 steps, save checkpoint, resume with `--optimize-kernels` (via granular flags) + preserved optimizer/scaler state, run 1000 more steps (total 1500).
2. **C.1.b** — native 500 steps, save checkpoint, resume with same kernels but FRESH optimizer (re-init) + FRESH scaler, run 1000 more steps.
3. **C.1.c** — native 1000 steps, save checkpoint, resume with kernels + preserved state, 1000 more steps (total 2000).
4. **C.1.d** — native 1000 steps, save checkpoint, resume with loss-only kernels first (fused_zloss + ddp_tune) for 500 steps, then add hidden kernels for final 500 steps.

**Implementation:** write `scripts/probe_t5_c1_warmstart.sh` that launches each variant sequentially. Use `--max-steps` to cap each phase. Use `--resume-from` to load checkpoints.

**Kernel stack for "optimized" resume:**
```
--use-fused-zloss --ak-loss-zloss \
--ak-fix-rope-gate-op --ak-causal-conv-shim \
--ak-spectra-branchless --ak-sync-cleanup \
--ak-normuon-telemetry \
--ak-autocast-tier all
```

**Pre-committed interpretation (v3 §11):**

| Observation | Meaning | Next action |
|---|---|---|
| a fails, b works | optimizer-state mismatch (v3 H14 confirmed) | Ship: reset optimizer on switch |
| b fails, a works | preserved state carried stale momentum | Ship: preserved-state transition works |
| Both a+b fail shortly after enable | kernel numerics/dtype issue | Proceed to C.2/C.3 anyway; return to replay |
| c works, a fails | warmup-local instability (v3 H5 confirmed) | Ship: delayed enable at step 1000 |
| d works, abc fail | graduated enable is necessary | Ship: Stack D uses delayed enable |
| All 4 fail identically at same absolute step | global stability threshold | T-5 is hopeless; Stack D = Stack C |

**Record per config:** tok/s, loss @ final step, GradScaler scale history, update-ratio spikes (from telemetry), divergence step (if any), frozen param check.

**Time budget:** each config ~30-45 min. 4 configs = ~2-3h total.

Commit: `T-5.C.1: warm-start matrix results (all 4 configs)`

**Write findings:** `docs/perf/t5-c1-warmstart-findings.md`

**Per user direction: run all 4 regardless of failures.**

#### C.2 — Post-NorMuon trust cap probe (~1h)

**Goal:** test whether post-NorMuon update-ratio cap (τ=0.02 on w_gate_up) prevents the optimized-path divergence.

**Implementation:** `scripts/probe_t5_c2_trust_cap.sh` launches:
```
--use-fused-zloss --ak-loss-zloss \
--ak-fix-rope-gate-op --ak-causal-conv-shim \
--ak-spectra-branchless --ak-sync-cleanup \
--ak-normuon-telemetry \
--ak-autocast-tier all \
--ak-trust-cap 0.02 --ak-trust-cap-scope w_gate_up
```
500-step DDP run from scratch (no warm-start). Target: confirm tok/s stays in Stack A/B range, no divergence before step 500, trust-cap triggers visible in telemetry.

**Decision:**
- Passes + trust-cap triggers visible: confirms update-scale mechanism, fold into Stack D
- Passes but trust-cap never triggers: update-ratio never exceeded 0.02 → the mechanism isn't primarily update scale → look at H11 (autocast) more
- Fails same pattern as Phase C/G: trust cap alone is insufficient

Commit: `T-5.C.2: post-NorMuon trust cap probe results`

**Write findings:** `docs/perf/t5-c2-trust-cap-findings.md`

#### C.3 — w_gate_up staging probe (~1h)

**Goal:** test whether 0.25 → 1.0 ramp over 1000 steps on w_gate_up updates prevents the fp16 overflow cascade.

**Implementation:** `scripts/probe_t5_c3_wgu_staging.sh` launches:
```
--use-fused-zloss --ak-loss-zloss \
--ak-fix-rope-gate-op --ak-causal-conv-shim \
--ak-spectra-branchless --ak-sync-cleanup \
--ak-normuon-telemetry \
--ak-autocast-tier all \
--ak-w-gate-up-scale 0.25 --ak-w-gate-up-ramp-steps 1000
```
1500-step DDP run from scratch.

**Decision:**
- Passes without divergence: staging works, fold into Stack D
- Passes at step 500, diverges after ramp completes (step 1000+): ramp needs to be longer or scale stays permanently lower
- Fails early: ramp not the mechanism

Commit: `T-5.C.3: w_gate_up staging probe results`

#### C.4 — Stack D assembly + 2000-step gate (~2-3h)

**Goal:** combine passing ingredients from C.1-C.3 + A.1 shims + A.3 autocast rules into Stack D, run full 2000-step gate.

**Composition rules:**
- If C.1.c passed (native 1000 → optimized resume, preserved state): Stack D uses delayed-enable strategy. Launch in two phases:
  - Phase I: Stack B/C for 1000 steps (native hidden)
  - Phase II: resume with `--ak-fix-rope-gate-op --ak-causal-conv-shim --ak-autocast-tier all` + optional trust cap + optional staging
- If C.2 passed: include `--ak-trust-cap 0.02 --ak-trust-cap-scope w_gate_up`
- If C.3 passed: include `--ak-w-gate-up-scale 0.25 --ak-w-gate-up-ramp-steps 1000`
- If C.1 all failed but C.2 OR C.3 passed: try from-scratch Stack D with only the passing knob

**Gate criteria:**
- 2000 steps reach completion
- Loss @ 2000 within 0.1 of Sprint 3A-confirm baseline (4.2 expected)
- No GradScaler collapse
- Zero frozen params (preflight check)
- Tok/s ≥ Stack C + 2% (else infrastructure-only ship)

**If Stack D gate passes:** commit final Stack D recipe. Update `scripts/launch_sprint3a.sh` STACK=D variant.

**If Stack D gate fails:** drop Stack D. Note in final scorecard.

Commit: `T-5.C.4: Stack D 2000-step gate result`

**Write findings:** `docs/perf/t5-c4-stack-d-findings.md`

### PHASE D — Finalize (1-2h)

**Goal:** consolidate all findings, pick best-passing stack, update production docs, commit final state. **STOP before Sprint 3A/3B launch.**

#### D.1 — Best-stack pick logic

Walk the decision tree:
1. Does Stack D 2000-step gate pass AND tok/s > Stack C by ≥2%? → Ship D.
2. Else: does Stack C (+compiled autograd) gate pass AND tok/s > Stack B by ≥2%? → Ship C.
3. Else: does Stack B (+ batch=32) memory-fit AND pass 500-step smoke? → Ship B.
4. Else: ship Stack A (confirmed by T-1.5 +7.7%).

#### D.2 — Final scorecard document

Create `docs/perf/v3-final-stack-scorecard.md` with this structure:

```markdown
# v3 40k Campaign — Final Stack Scorecard

**Date:** [today]
**Baseline:** OdinFlat Sprint 3A-confirm, ~31,331 tok/s
**Winner:** Stack [A|B|C|D]

| Stack | Composition | tok/s | Delta | Loss@2000 | Frozen | GradScaler | Status |
|---|---|---:|---:|---:|---:|---|---|
| Baseline | native | 31,331 | — | 4.20 | 0 | stable | baseline |
| A | +fused_zloss +SPECTRA +sync | ~33,750 | +7.7% | [from T-1.5] | 0 | stable | SHIPPED |
| B | A + batch=32 + DDP tune | [T-2.1] | [T-2.1] | [...] | 0 | [...] | [DECISION] |
| C | B + compiled autograd | [T-4] | [T-4] | [...] | 0 | [...] | [DECISION] |
| D | C + hidden kernel recovery | [T-5] | [T-5] | [...] | 0 | [...] | [DECISION] |

## Recipe (winning stack)
[exact command including all --ak-* flags]

## Findings summary
- [one-liner per phase]

## Deferred to post-launch
- [items that needed more time]
```

#### D.3 — Update STATUS.md

Replace current throughput section:
```markdown
## AutoKernel 40k throughput effort — COMPLETE (Phase B-D)

Winner: Stack [X]
Final tok/s: [Y]k aggregate
Baseline: 31.3k aggregate

Production launch command:
```bash
STACK=[X] bash scripts/launch_sprint3a.sh
STACK=[X] bash scripts/launch_sprint3b.sh
```

Full scorecard: docs/perf/v3-final-stack-scorecard.md

Execution plan: docs/research/autokernel-40k-v3-execution-plan.md
Session logs: docs/perf/v3-session-execution-log.md, docs/perf/v3-session-2-execution-log.md
```

#### D.4 — Update AGENTS.md

Add a v3 flag taxonomy section with the 28 `--ak-*` flags, and note the launch-script convention (`STACK=A|B|C|D bash scripts/launch_sprint3a.sh`).

#### D.5 — Update CONSTRAINTS.md

Add these rules:
- "Any new HIP kernel in the training path MUST use `torch.library.custom_op + register_autograd + register_autocast`."
- "Tier 2 parity test required before shipping any new op to production training path."
- "Sprint 3A/3B launches require explicit user approval. Never auto-launch."

#### D.6 — Session 2 execution log

Write `docs/perf/v3-session-2-execution-log.md` (mirror `v3-session-execution-log.md` format). Include:
- Commits this session
- Findings per phase
- What was deferred to session 3 (if any)
- Recommended launch configuration

#### D.7 — Final commit + push

```bash
git add -A
git commit -m "v3 campaign final scorecard + STATUS/AGENTS/CONSTRAINTS updates (phase D complete, sprint launches await user approval)"
git push origin main
```

#### D.8 — STOP

**Do not launch Sprint 3A or 3B.** Wait for explicit user approval. Present the winning stack composition and expected throughput, and ask the user when to launch.

---

## 7. Stop-and-ask rules (pre-committed, do NOT deviate)

You WILL pause execution and ask the user if:

1. T-1.5 final loss at step 2000 is >0.1 worse than Sprint 3A-confirm baseline (4.20). Indicates fused z-loss parity concern.
2. Any Phase B probe (T-1.4 / T-2.1 / T-4) regresses tok/s >3% from baseline. Unexpected signal.
3. T-4 compiled autograd gate **passes with >5% net gain**. Confirms deeper follow-on integration scope — ask user whether to invest additional time.
4. Stack D 2000-step gate **passes** (unexpected positive outcome). Confirm ship decision before updating production docs.
5. Timeline exceeds 20h wall. Ask whether to continue or stop.
6. Any NEW frozen-params warning fires in any DDP run. Immediate stop — this indicates a ship-blocker regression.
7. GradScaler scale falls below 1.0 (collapse) in any probe. Immediate stop + forensics dump.
8. Machine A or Machine B disk usage exceeds 85%. Ask before continuing (checkpoints consume space fast).

You WILL NOT stop (pre-committed behavior):

- T-4 CA gate fails: ships as infra-only, continue to Phase C. (v3 default decision.)
- Phase C.1 warm-start all 4 configs diverge. Continue to C.2/C.3/C.4 regardless. (User directive.)
- Any individual Tier 2 parity test fails. Report + continue. Fix later if Stack D is chosen.
- Expected incremental findings (scorecards within predicted ranges).

---

## 8. Exact commands for each phase (copy-paste ready)

### 8.1 Setup at session start

```bash
# Navigate to repo
cd "C:\Users\z00517bz\Documents\dev\autokernel-halo-strix"

# Verify clean git state
git status
git log --oneline -5

# Check machine A status (T-1.5 should be running or done)
bash run_remote.sh "ps aux | grep train_ddp | grep -v grep | head -2"
bash run_remote.sh "grep 'step' ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/t1-5-fused-zloss/rank0.log | tail -5"
bash run_remote.sh "ls ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/t1-5-fused-zloss/step_*.pt 2>/dev/null"
```

### 8.2 Phase A.1 causal_conv1d shim commands

After writing the code edits described in §6.Phase.A.A.1:

```bash
# Syntax check locally
python -X utf8 -c "import ast; ast.parse(open('kernels/hip/_torch_ops.py', encoding='utf-8').read()); ast.parse(open('models/components/conv_blocks.py', encoding='utf-8').read()); ast.parse(open('scripts/train_ddp.py', encoding='utf-8').read()); print('all OK')"

# Sync to Machine A
bash sync_remote.sh

# Run parity test (wait until T-1.5 is done OR use separate gpu if available)
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && source .venv/bin/activate && export HSA_OVERRIDE_GFX_VERSION=11.5.1 && python scripts/test_causal_conv1d_shim.py 2>&1"

# If passes 3/3, commit
git add -A
git commit -m "T-3.2 (2nd half): causal_conv1d shim with custom_op + register_autograd"
git push origin main
```

### 8.3 Phase A.2/A.3 Tier 2 + autocast commands

```bash
# After writing test_tier2_parity.py and adding register_autocast to _torch_ops.py
python -X utf8 -c "import ast; ast.parse(open('kernels/hip/_torch_ops.py', encoding='utf-8').read()); ast.parse(open('scripts/test_tier2_parity.py', encoding='utf-8').read()); print('OK')"
bash sync_remote.sh
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && source .venv/bin/activate && export HSA_OVERRIDE_GFX_VERSION=11.5.1 && python scripts/test_tier2_parity.py 2>&1"

# Re-run dtype inventory to verify register_autocast is now detected
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && source .venv/bin/activate && python scripts/autokernel_dtype_inventory.py --output docs/perf/dtype-autocast-inventory-post-A3.md --jsonl docs/perf/dtype-autocast-inventory-post-A3.jsonl"

# Pull inventory back
scp joelwang-ai-2@192.168.1.140:~/Desktop/ai_lab/autokernel-halo-strix/docs/perf/dtype-autocast-inventory-post-A3.md docs/perf/
scp joelwang-ai-2@192.168.1.140:~/Desktop/ai_lab/autokernel-halo-strix/docs/perf/dtype-autocast-inventory-post-A3.jsonl docs/perf/

git add -A
git commit -m "T-3.3: Tier 2 parity tests + register_autocast rules on 5 ops"
git push origin main
```

### 8.4 Phase A.4 T-1.5 scorecard commands

```bash
# Wait until step 2000 reached
while true; do
  sleep 300
  STEP=$(bash run_remote.sh "grep 'step' ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/t1-5-fused-zloss/rank0.log | tail -1" | grep -oE 'step[ ]+[0-9]+' | grep -oE '[0-9]+')
  echo "current step: $STEP"
  if [ "$STEP" -ge 2000 ]; then break; fi
done

# Extract final metrics
bash run_remote.sh "grep 'step    ' ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/t1-5-fused-zloss/rank0.log | tail -3"
bash run_remote.sh "ls -la ~/Desktop/ai_lab/autokernel-halo-strix/checkpoints/t1-5-fused-zloss/step_2000.pt"

# Run scorecard
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && source .venv/bin/activate && EVAL_MACHINE=a python scripts/eval_checkpoint.py --checkpoint checkpoints/t1-5-fused-zloss/step_2000.pt --model models/odin_flat.py --class-name OdinFlat 2>&1 | tail -30"

# Pull scorecard JSON
scp joelwang-ai-2@192.168.1.140:~/Desktop/ai_lab/autokernel-halo-strix/docs/perf/eval-scorecards/*.json docs/perf/eval-scorecards/
# Also pull the JSONL index update
scp joelwang-ai-2@192.168.1.140:~/Desktop/ai_lab/autokernel-halo-strix/docs/perf/eval-scorecard.jsonl docs/perf/eval-scorecard.jsonl

# Update t1-5-fused-zloss-preliminary.md to final
# (Edit file locally to rename + add final step-2000 numbers + add scorecard summary)

git add -A
git commit -m "T-1.5 FINAL: 2000-step canonical gate PASS + scorecard"
git push origin main
```

### 8.5 Phase B.1 DDP bucket sweep commands

```bash
bash sync_remote.sh
bash sync_remote_b.sh

# Verify no DDP process active
bash run_remote.sh "ps aux | grep train_ddp | grep -v grep"
bash run_remote_b.sh "ps aux | grep train_ddp | grep -v grep"

# Launch sweep (blocks until all 4 configs complete)
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && bash scripts/probe_t1_ddp_bucket_sweep.sh 2>&1" | tee docs/perf/t1-4-sweep-session.log

# Pull results
scp joelwang-ai-2@192.168.1.140:~/Desktop/ai_lab/autokernel-halo-strix/docs/perf/t1-ddp-bucket-sweep/results.txt docs/perf/t1-ddp-bucket-sweep/

# Write findings
# Create docs/perf/t1-4-ddp-bucket-sweep-findings.md with analysis

git add -A
git commit -m "T-1.4: DDP bucket sweep results + ship decision"
git push origin main
```

### 8.6 Phase B.2 batch=32 probe commands

```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && bash scripts/probe_t2_1_batch32.sh 2>&1" | tee docs/perf/t2-1-batch32-session.log

# Pull results
scp joelwang-ai-2@192.168.1.140:~/Desktop/ai_lab/autokernel-halo-strix/docs/perf/t2-1-batch32-probe/results.md docs/perf/t2-1-batch32-probe/

# Analyze + findings doc
git add -A
git commit -m "T-2.1: batch=32 probe results + Stack B decision"
git push origin main
```

### 8.7 Phase B.3 compiled autograd commands

```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && bash scripts/probe_t4_compiled_autograd.sh 2>&1" | tee docs/perf/t4-ca-session.log

scp joelwang-ai-2@192.168.1.140:~/Desktop/ai_lab/autokernel-halo-strix/docs/perf/t4-compiled-autograd-smoke/results.md docs/perf/t4-compiled-autograd-smoke/

# Analyze — decide Stack C
git add -A
git commit -m "T-4: compiled autograd gated smoke results + Stack C decision"
git push origin main
```

### 8.8 Phase C warm-start matrix commands

Write `scripts/probe_t5_c1_warmstart.sh` (no existing script; template follows):

```bash
#!/bin/bash
set -e
cd ~/Desktop/ai_lab/autokernel-halo-strix

KERNEL_FLAGS="--use-fused-zloss --ak-loss-zloss --ak-fix-rope-gate-op --ak-causal-conv-shim --ak-spectra-branchless --ak-sync-cleanup --ak-normuon-telemetry --ak-autocast-tier all"
BASE_FLAGS="--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0"

# C.1.a: native 500 -> optimized preserved state
rm -rf checkpoints/t5-c1-native-phase
mkdir -p checkpoints/t5-c1-native-phase
MODEL=models/odin_flat.py CLASS=OdinFlat DATASET=datasets/dolma-10b-odin32k.bin \
  EPOCHS=1 LR=8e-4 BLOCK=512 BATCH=16 ACCUM=8 WARMUP_STEPS=300 CHECKPOINT_INTERVAL=500 \
  MAX_GRAD_NORM=1.0 NUM_WORKERS=12 CKPT_DIR=checkpoints/t5-c1-native-phase \
  EXTRA_FLAGS="--max-steps 500 $BASE_FLAGS" \
  bash scripts/launch_ddp.sh
# Wait...

# Then launch 4 variants resuming from step_500.pt
# Each with its own CKPT_DIR and flag composition
# (Full script left as exercise; mirror patterns from probe_t2_1_batch32.sh)
```

Run, gather results, analyze per the interpretation table in §6.Phase.C.C.1.

### 8.9 Phase D final commands

```bash
# After all probes and analysis, update production docs:
# - STATUS.md
# - AGENTS.md (v3 flag section)
# - CONSTRAINTS.md (register_autocast rule)
# - docs/perf/v3-final-stack-scorecard.md
# - docs/perf/v3-session-2-execution-log.md

git add -A
git commit -m "v3 campaign final: winning stack + scorecard + STATUS/AGENTS/CONSTRAINTS updates"
git push origin main

echo "=== SESSION COMPLETE ==="
echo "Winning stack: [A|B|C|D]"
echo "Production ready: STACK=[X] bash scripts/launch_sprint3a.sh"
echo ""
echo "STOP HERE. Await user approval for Sprint 3A/3B launch."
```

---

## 9. Expected outcomes and ranges

Use these to sanity-check probe results. If outside these ranges, something is wrong — stop and investigate.

### T-1.5 final (step 2000)

- Tok/s: 33.5-34k (matches prelim at step 600)
- Loss: 3.2-3.4 (baseline Sprint 3A-confirm was 3.15 at step 2000 without fused zloss)
- Loss difference from baseline: < 0.2 (fused z-loss adds small numerical difference; should be tight parity)
- GradScaler: stable, no collapse
- Frozen params: 0

### T-1.4 DDP bucket sweep

- All 4 bucket sizes within ±1% tok/s of each other (manual allreduce dominates)
- `gradient_as_bucket_view=True` reduces peak memory by ~0.5-1 GB per node
- Best case: 1-2% tok/s improvement from optimal bucket + view combo

### T-2.1 batch=32 probe

- Baseline batch=16: 33.7k tok/s (matches Stack A)
- batch=32 plain: expected +3-5% tok/s, ~+8 GB memory per node
- batch=32 + ddp_tune: same or slightly better, memory fits if <40 GB per node
- batch=32 + sync_cleanup: +5-7% (branchless SPECTRA stacking)

### T-4 compiled autograd

- Baseline: 33.7k (matches prior)
- + CA: uncertain — 1-4% likely, up to 8% possible, REGRESSION possible (DDP overlap)
- Recompile count: should be <5 after step 100 (warmup done). More = recompile storm, fail gate.
- Allreduce overlap measurement: CURRENTLY BROKEN (always shows 0 due to single-allreduce-per-step). Document as known issue; rely on tok/s for gate.

### T-5 Phase C probes

**C.1 warm-start (expected based on v3 H14):**
- If H14 confirmed (optimizer state mismatch): a fails, b works
- If warmup-local (H5): c works, a fails
- If neither: all 4 fail with same pattern as Phase C (divergence ~step 250)
- **Most likely outcome:** all 4 fail early (Stack D is hard). User directed to run anyway for complete diagnostic.

**C.2 trust cap (τ=0.02 on w_gate_up):**
- If update-scale is the mechanism: passes 500 steps, trust-cap triggers visible in telemetry
- If not: fails same Phase C pattern, no triggers visible

**C.3 w_gate_up staging (0.25 → 1.0 over 1000 steps):**
- If warmup-local: passes 1500 steps smoothly
- If fundamental: diverges either in ramp or shortly after reaching 1.0

**C.4 Stack D final:**
- Best-case: passes 2000-step gate at ~38-40k tok/s (Stack B + all recovery knobs)
- Likely-case: fails gate, Stack D ships as infrastructure only
- Worst-case: regression

### Phase D final stack winner

| Scenario | Winner | Expected tok/s |
|---|---|---:|
| Stack A only viable | A | 33.7k |
| Stack B works (batch=32 fits) | B | 35-37k |
| Stack C gate passes | C | 36-38k |
| Stack D gate passes (unlikely) | D | 38-40k |

**Most likely winner: Stack B.** 40k is stretch; 36-38k is honest realistic.

---

## 10. Environment details (reference)

### 10.1 Machine A (primary DDP + single-node tests)

- Hostname: `joelwang-ai-2-BeyondMax-Series`
- User: `joelwang-ai-2`
- SSH: `joelwang-ai-2@10.77.0.1` (TB4) or `@192.168.1.140` (LAN/scp)
- Repo path: `/home/joelwang-ai-2/Desktop/ai_lab/autokernel-halo-strix/`
- Venv: `.venv/bin/activate` (clean, no aiter)
- GPU: gfx1151, one per machine
- OS: Linux with ROCm 7.12

### 10.2 Machine B (DDP rank 1 only)

- Hostname: `joelwang-ai-1-BeyondMax-Series`
- User: `joelwang-ai-1`
- SSH: `@10.77.0.2` (TB4) or `@192.168.1.145` (LAN)
- Repo path: `/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/autokernel-halo-strix/`
- Venv: `/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/.venv/bin/activate` (has aiter — DO NOT use for training)
- Activation quirk: venv activate has unbound variable; may need `set +u` before sourcing

### 10.3 Network

- TB4 subnet: `10.77.0.0/24`
- Machine A: `10.77.0.1`
- Machine B: `10.77.0.2`
- Interface: `thunderbolt0`
- Measured raw throughput: ~1.04 GB/s
- DDP: `GLOO_SOCKET_IFNAME=thunderbolt0` + `MASTER_ADDR=10.77.0.1 MASTER_PORT=29500`

### 10.4 Required environment variables for DDP runs

Already set in `scripts/launch_ddp.sh` but critical to know:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1       # forces gfx1151 codepath
export TORCH_COMPILE_MODE="max-autotune-no-cudagraphs"  # required for accum > 1
export GLOO_SOCKET_IFNAME=thunderbolt0
export NCCL_SOCKET_IFNAME=thunderbolt0       # ignored but set for safety
export MASTER_ADDR=10.77.0.1
export MASTER_PORT=29500                     # use 29510/29511 if 29500 stale
```

### 10.5 Datasets on both machines

- `datasets/dolma-10b-odin32k.bin` — 6.9B tokens, 13.7 GB — on BOTH
- `datasets/wikitext-103-odin32k.bin` — smaller, smoke runs
- Loaded via `np.memmap` (zero-copy)

---

## 11. Known issues and gotchas

### 11.1 scp vs sync_remote.sh

`sync_remote.sh` uses scp under the hood but the initial runs of this session showed it occasionally didn't pick up tiny edits (possibly due to timestamp issues). If you see "file didn't update on remote," force-scp the specific file:
```bash
scp scripts/FILE.py joelwang-ai-2@192.168.1.140:/home/joelwang-ai-2/Desktop/ai_lab/autokernel-halo-strix/scripts/FILE.py
```

### 11.2 sync_remote_b.sh can timeout

First invocation can take >2 min if files are cold. Use `-timeout 300000` (5 min) for the `bash` tool call.

### 11.3 DDP trace allreduce_total_ms = 0

Known cosmetic bug in `scripts/train_ddp.py::_ddp_trace_emit`. When there's only 1 allreduce per opt step (our normal case), `first_allreduce_wall_ms == last_allreduce_wall_ms` so the diff is 0. Doesn't affect correctness; `allreduce_count` is what matters. Don't try to "fix" this unless you're specifically adding overlap measurement (requires hooking backward start/end separately).

### 11.4 param_names is a PyTorch-reserved optimizer key

Previous session hit `ValueError: all optimizer param groups should be with/without names`. Solution: use `_telem_param_names` (underscore-prefixed) in NorMuon and `_mup_param_names` in mup.py, translated by optimizer.py. Already fixed in commit `124b18e`. If you see this error again, check that you're not passing `param_names` as a key in an optimizer group dict.

### 11.5 rank 0 venv not auto-activated

`launch_ddp.sh` sources venv on rank 1 (Machine B's SSH command) but NOT for rank 0 (Machine A). Rank 0 relies on the caller's environment being already activated. This has been confirmed working in current setup but is a land mine if the venv path changes.

### 11.6 Compile warmup takes ~2-3 min

First forward on `max-autotune-no-cudagraphs` runs ~25-50 Inductor autotune searches (~20 s compile + 60-90 s first-run). Step 50 in logs typically shows compile-warmup throughput (~25k tok/s for T-1.5). Ignore step 50 from throughput averaging; use step 100+ as steady-state.

### 11.7 fp16 LayerNorm dtype warning

Harmless runtime warning: `Mismatch dtype between input and weight: input dtype = c10::Half, weight dtype = float, Cannot dispatch to fused implementation`. Appears in every training log. Inductor generates correct code; the warning is a C++ fallback path note. Ignore.

### 11.8 expandable_segments not supported

Harmless warning from HIP allocator: `expandable_segments not supported on this platform`. No action needed.

### 11.9 T-1.5 log_interval is 50

`log_interval=50` in default launch_ddp.sh. Steps appear in rank0.log only every 50 opt steps. `ddp_trace_rank0.jsonl` updates every opt step (higher resolution). Use JSONL for step-count verification during probes.

### 11.10 Deferred bundle-dump in replay infrastructure (C.0)

If C.0 replay infrastructure is time-constrained, implement ONLY the dump-on-failure path (not the replay executor script). A future session can add replay.py after seeing real failures.

---

## 12. Reference: complete flag dispatch logic

All flag effects flow through these code paths:

**Environment variables set by flags (in `scripts/train_ddp.py::main`):**

```python
if args.ak_fix_rope_gate_op:
    os.environ["AUTOKERNEL_FIX_ROPE_GATE"] = "1"

if args.ak_spectra_branchless or args.ak_sync_cleanup:
    os.environ["AUTOKERNEL_SPECTRA_BRANCHLESS"] = "1"

if args.ak_causal_conv_shim:  # Phase A.1 will add this
    os.environ["AUTOKERNEL_CAUSAL_CONV_SHIM"] = "1"

if args.ak_compiled_autograd:
    torch._dynamo.config.compiled_autograd = True
```

**NorMuon kwargs (from `halo_training/optimizer.py::build_imu1_optimizer`):**

```python
telemetry_enabled=args.ak_normuon_telemetry,
telemetry_path=<ckpt>/normuon_telem_rank{rank}.jsonl,
trust_cap=args.ak_trust_cap,
trust_cap_scope=args.ak_trust_cap_scope,
w_gate_up_scale=args.ak_w_gate_up_scale,
w_gate_up_ramp_steps=args.ak_w_gate_up_ramp_steps,
spectra_branchless=(args.ak_spectra_branchless or args.ak_sync_cleanup or args.ak_normuon_impl_opt),
```

**Training loop hot paths affected:**

```python
# v3 T-1.2: --ak-sync-cleanup replaces per-microstep loss.item() with tensor accumulator
if args.ak_sync_cleanup:
    _step_loss_t = _step_loss_t + loss.detach().float()
    loss_val = 0.0  # deferred
else:
    loss_val = loss.item()

# v3 T-0.6: DDP trace emits JSONL per opt step
_ddp_trace_emit(world_size, rank)

# v3 T-0.8: sync counter written after profiler window
# (inside profiler export block)
```

**conv_blocks.py branching:**

```python
# _compile_friendly: native PyTorch RoPE+gate (slow but clean)
# AUTOKERNEL_FIX_ROPE_GATE=1: autokernel::fused_rope_gate_mul (fast, autograd-safe)
# else: legacy kernel_fn @torch.compiler.disable (fast but silent-freeze!)

# Causal conv:
# _compile_friendly: _manual_causal_conv1d (native)
# _HAS_CAUSAL_CONV1D + AUTOKERNEL_CAUSAL_CONV_SHIM=1: autokernel::causal_conv1d (A.1 target)
# _HAS_CAUSAL_CONV1D (default): DaoAILab causal_conv1d_fn (graph-break, but correct backward)
# else: self.conv (native torch)
```

---

## 13. Required file edits for Phase A.1 (precise instructions)

Copy-paste these edits. File paths are absolute Windows paths as the agent will see them.

### Edit 1: `C:\...\kernels\hip\_torch_ops.py` (append at end)

Append the causal_conv1d shim block shown in §6.Phase.A.A.1 step 1. Keep the existing content intact; add at file end after the last `register_autograd(...)` line.

### Edit 2: `C:\...\models\components\conv_blocks.py`

Find the existing `elif _HAS_CAUSAL_CONV1D:` block (around line 292). Replace with the env-var-gated version shown in §6.Phase.A.A.1 step 2.

Leave the `if self._compile_friendly:` branch and the final `else:` branch unchanged.

### Edit 3: `C:\...\scripts\train_ddp.py`

Find this existing block in `main()` (lines ~905-917, the env-var setup section):
```python
if getattr(args, "ak_fix_rope_gate_op", False):
    os.environ["AUTOKERNEL_FIX_ROPE_GATE"] = "1"
if getattr(args, "ak_spectra_branchless", False) or getattr(args, "ak_sync_cleanup", False):
    os.environ["AUTOKERNEL_SPECTRA_BRANCHLESS"] = "1"
```

Add AFTER the second `if`:
```python
if getattr(args, "ak_causal_conv_shim", False):
    os.environ["AUTOKERNEL_CAUSAL_CONV_SHIM"] = "1"
```

The flag `--ak-causal-conv-shim` was already defined in commit `ded406a` — no need to add the argparse entry.

### Edit 4: create `C:\...\scripts\test_causal_conv1d_shim.py`

Mirror `scripts/test_fused_rope_gate_mul_custom_op.py`. Three test functions:
- `test_forward_parity()` — compare custom_op output to direct DaoAILab call
- `test_gradient_flow()` — assert x.grad and weight.grad populate, y.grad_fn is not None
- `test_backward_parity()` — compare custom_op backward to pure-PyTorch conv1d backward reference

Import pattern:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernels.hip._torch_ops  # trigger registration
from causal_conv1d import causal_conv1d_fn
import torch
import torch.nn.functional as F
```

Production-similar shapes: B=2, D=1536, T=128 (matches OdinFlat per-layer).

---

## 14. Required file edits for Phase A.2 + A.3 (precise instructions)

### Edit 5: `C:\...\kernels\hip\_torch_ops.py` — add register_autocast rules

After each existing `register_autograd` call for these ops:
- `silu_gate_mul_op`
- `rmsnorm_op`
- `fused_res_rmsnorm_op`
- `fused_rope_gate_mul_op` (from T-3.2 fix; at end of file currently)

Append:
```python
<op_name>.register_autocast("cuda", torch.float16)
```

Special case for `rotary_emb_fp32_op`:
```python
rotary_emb_fp32_op.register_autocast("cuda", torch.float32)  # fp32 internal; keep in fp32
```

For `causal_conv1d_shim_op` (after Edit 1):
```python
causal_conv1d_shim_op.register_autocast("cuda", torch.float16)
```

### Edit 6: create `C:\...\scripts\test_tier2_parity.py`

Structure (pseudocode, fill in exact imports + assertions):

```python
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernels.hip._torch_ops

device = "cuda"
torch.manual_seed(42)

def _rel(a, b):
    return (a.float() - b.float()).abs().max().item() / max(a.float().abs().max().item(), b.float().abs().max().item(), 1e-6)

def test_silu_gate_mul_parity():
    shape = (16 * 512, 2048)
    gate = torch.randn(shape, device=device, dtype=torch.float16, requires_grad=True)
    up = torch.randn(shape, device=device, dtype=torch.float16, requires_grad=True)

    # Reference
    gate_r = gate.detach().clone().requires_grad_(True)
    up_r = up.detach().clone().requires_grad_(True)
    y_ref = F.silu(gate_r) * up_r
    y_ref.sum().backward()

    # Custom
    y_hip = torch.ops.autokernel.silu_gate_mul(gate, up)
    y_hip.sum().backward()

    assert _rel(y_ref, y_hip) < 5e-3, f"forward: {_rel(y_ref, y_hip)}"
    assert _rel(gate.grad, gate_r.grad) < 5e-3
    assert _rel(up.grad, up_r.grad) < 5e-3
    print("  [PASS] silu_gate_mul")

def test_rmsnorm_parity():
    # shape (16*512, 768); weight (768,)
    ...

def test_fused_res_rmsnorm_parity():
    # x + residual, then RMSNorm
    ...

def test_causal_conv1d_parity():
    # Requires Phase A.1 shim
    ...

# Repeat each under torch.autocast to verify autocast parity
def test_all_under_autocast():
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        test_silu_gate_mul_parity()
        test_rmsnorm_parity()
        ...

if __name__ == "__main__":
    test_silu_gate_mul_parity()
    test_rmsnorm_parity()
    test_fused_res_rmsnorm_parity()
    test_causal_conv1d_parity()
    test_all_under_autocast()
    print("\n[PASS] T-3.3 Tier 2 parity (8 tests)")
```

---

## 15. Cross-session traceability

### 15.1 Prior sessions' key artifacts

- `docs/research/autokernel-stability-research-brief.md` — 14K-word stability context
- `docs/research/autokernel_halo_strix_40k_throughput_experiment_guide.md` — v1 from research engineer
- `docs/research/autokernel-40k-guide-engineering-response.md` — our response to v1
- `docs/research/autokernel_40k_revised_engineering_handoff_v2.md` — v2 from engineer
- `docs/research/autokernel-40k-engineering-v3-addendum.md` — our v3 request
- `docs/research/autokernel_40k_v3_research_engineering_response.md` — v3 final from engineer
- `docs/research/autokernel-40k-v3-execution-plan.md` — **the strategic plan (locked)**
- `docs/perf/v3-session-execution-log.md` — previous session (this handoff is follow-up)
- `docs/perf/t1-5-fused-zloss-preliminary.md` — T-1.5 at step 300 preliminary
- `docs/perf/backward-breakdown.md` — T-0.1 findings
- `docs/perf/sync-point-audit.md` — T-0.3 findings
- `docs/perf/graph-breaks-inventory.md` — T-0.4 findings
- `docs/perf/dtype-autocast-inventory.md` — T-0.7 findings (0/7 have register_autocast)

### 15.2 Decision trail (what was approved, when)

- **2026-05-11 early session:** user exited plan mode, told agent to "Start" executing v3 plan.
- **Mid-session:** user answered the 5-question planning block (accept v3 priority reordering, STATUS language verbatim, three-tier autocast, launch after T-6 Week 3).
- **Session close:** user asked "What about the other items?" then answered 5 more questions:
  - Never launch sprints without approval (universal rule).
  - T-5 unconditional.
  - T-3.3 full Tier 2.
  - T-3.2 causal_conv1d shim full engineering.
  - One long session for everything.
- **This handoff creation:** user explicitly requested a comprehensive handoff document for post-compaction continuity.

### 15.3 Commits by topic (if you need to revert anything)

| Commit | Topic | Safe to revert? |
|---|---|---|
| `d05adf7` (or similar `c44eb00`) | Lock execution plan + STATUS | NO — foundation |
| `ded406a` | Granular --ak-* flags | NO — foundation |
| `e60b5df` | DDP trace + dtype inventory + sync counter | NO — instrumentation |
| `37c16c6` | Branchless SPECTRA + deferred loss + bucket sweep | Low risk; parity verified |
| `ac59cf9` | Branchless SPECTRA parity VERIFIED + T-0.7 findings | NO — findings |
| `124b18e` | param_names collision fix | NO — required for NorMuon ctor |
| `b498816` | T-3.2 fused_rope_gate_mul fix | NO — fixes silent-freeze |
| `22ce520` | T-2.1/T-4 probe scripts + CA activation | Low risk; gated by flag |
| `e9fae71` | Session closeout + Sprint 3 launch scripts | NO — final state |

---

## 16. Final reminders for the agent picking this up

1. **Read §0 (ground rules) FIRST before doing anything.** The never-launch rule for Sprint 3 is inviolable.

2. **Check T-1.5 status IMMEDIATELY** after resuming. It may be finished (good), running (wait), or crashed (relaunch).

3. **Phase A can happen while T-1.5 runs** (single-node engineering, no DDP conflict).

4. **Phase B is STRICTLY sequential** — only one DDP probe at a time. Verify cluster is idle before each.

5. **Phase C is the big compute chunk** (5-10h). Plan for it. Use `tee` and `nohup` / `setsid` for detached runs.

6. **Commit frequently** — target 15-20 commits across all phases. Each atomic and revertible.

7. **Update this document if your plan diverges.** Add a `## Appendix: Session 2 amendments` section at the bottom rather than rewriting history.

8. **STOP before Sprint 3A/3B launch.** Present the winning stack and ask the user.

9. **Time budget:** ~11-22 hours. If you exceed 20h, stop and ask.

10. **When confused, read `docs/research/autokernel-40k-v3-execution-plan.md`** — it's the strategic North Star.

---

## 17. Version history of this document

- v1.0 (2026-05-11) — initial handoff from agent ending at commit `e9fae71`

---

*End of handoff document. Next agent: begin at §6 Phase A.*
