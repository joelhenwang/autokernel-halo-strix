# Session Execution Log — Autokernel Remediation (2026-05-10 → 2026-05-11)

**Purpose:** complement to `session-2026-05-11-autokernel-remediation.md`.
Captures the raw chronology of commands run, problems hit, and how they
were resolved. Future session can grep this for "gotcha" patterns and
avoid re-discovery.

---

## Chronology

### 09:11 — Initial setup + Track 1.1 / 3.A scaffolding

- `scripts/train_ddp.py` edited to add `--profile-steps` and
  `--diag-frozen-params` flags.
- Local tests ran (Windows, CPU).
- First `sync_remote.sh` run timed out at default 60s → retried with
  180s timeout. Went through.

### 09:19 — First profile probe launched

- Launched via `ssh ... setsid nohup bash scripts/profile_odinflat_step.sh`.
- Rank 0 + rank 1 both came up, began DDP rendezvous.
- Compile warmup took ~2 min (autotune cache miss).
- Probe completed 09:25 with profile.json (1.5 GB) + profile-summary.txt (26 KB).

### 09:25 — Profile analysis

- Fetched profile-summary.txt + rank0.log via scp.
- Computed FLOP-ratio estimate: attention forward ~7.5% of step.
- Identified z-loss at 16.7% as bigger target than QKV fusion.

### 09:35 — Track 2.a QKV fusion committed + probed

- Committed QKV fusion code (6 tests passing).
- Launched probe on Machine A for post-fusion throughput.
- Machine B ran Phase I: V0 diagnostic in parallel (single-node). **First parallelism success.**

### 09:35 — First "set -u" failure

- probe_qkv_fusion.sh uses `set -euo pipefail`.
- `source .venv/bin/activate` references CPLUS_INCLUDE_PATH → unbound var → script exit before torchrun.
- Error: `line 79: CPLUS_INCLUDE_PATH: unbound variable`
- **Fix**: `set -eo pipefail` (no -u).

### 09:41 — V0 diag succeeded

- 50 steps, loss 7.3261, tok/s 15K.
- All 120 params always_finite except 1 always_none (v_res_scale, expected).

### 09:43 — V1 diag launched on B

- First launch attempt: venv-sourcing issue stopped it immediately.
- After sync with fixed `set -eo pipefail`, V1 ran.
- Step 50: loss 7.32, tok/s 20.7K (looks faster — but later determined to be the silent-freeze).

### 09:47 — V1 analysis

- **14 x w_gate_up.weight: always_none (frozen)**
- **14 x ffn_norm.weight: always_zero (downstream dead grad)**
- Total: 28 params frozen. **OdinFlat blast radius confirmed.**

### 09:50 — Pre-fusion QKV baseline

- Reverted `attention.py` locally via `git show 63de5be~1:...`.
- Synced pre-fusion code to A.
- Ran pre-fusion baseline probe.
- Restored post-fusion locally + re-synced.
- Result: post-fusion +0.07% (noise floor).

### 09:56 — V3 diag completed

- Step 50: loss 7.74, tok/s 15.9K.
- Same as V0: 119 always_finite + 1 always_none. **Phase III rmsnorm fix confirmed complete.**

### 10:03 — Track 3.F synthesis doc committed

### 10:20 — User scope expansion

"Everything you reported should be documented and tackled."
→ Master plan negotiated, scope locked in 4 questions.

### 10:31 — First B4 probe attempt → silent failure

- Launched `scripts/probe_odinhalo_b4.sh` via detached nohup.
- Rank 1 started on B. Rank 0 logged `LAUNCHED_A_pid=3065996` but the process was gone 5 seconds later.
- rank0.log: 0 bytes.
- Diagnosis: `launch_ddp.sh` rank 0 invocation `setsid nohup torchrun ...` doesn't source venv → `torchrun` not on PATH → silent exit.
- `setsid nohup foo > err 2>&1 < /dev/null &` captures stderr to err, but if `foo` binary doesn't exist, err remains empty.

### 10:33 — Second B4 attempt with port change

- Suspected stale state on port 29500; changed to 29511.
- Same failure. Port wasn't the issue.
- Verified via manual foreground `torchrun ...` — works fine. Confirmed it's the venv activation pattern.

### 10:34 — Third B4 attempt with probe_odinhalo_b4.sh venv fix

- Edit: added `source .venv/bin/activate` at top of probe script.
- Rank 1 + rank 0 both came up.
- Compile warmup ~2 min. First `[step 50]` line at 10:40.

### 11:01 — Phase A.1 static audit first pass

- First run: `IndexError: list index out of range` on line splitting `torch.ops.autokernel.`
- Fix: guard the split.
- Second run: produced 7-class audit. Windows CP1252 crashed on ✓ / ✗ characters.
- Fix: `[Y]` / `[N]` ASCII. Third run succeeded.
- Finding: 5 UNSAFE (later refined with CONDITIONAL-SAFE).

### 11:02 — Audit refinement (CONDITIONAL-SAFE + AST Attribute)

- `_RMSNormReplacement` initially classified UNSAFE because the raw
  `self.kernel_fn` is in a conditional-fallback branch. Refined to
  CONDITIONAL-SAFE.
- `_FusedSwiGLUReplacement` showed spurious "torch.ops.autokernel.silu_gate_mul"
  references FROM COMMENTS. Rewrote extraction to use `ast.Attribute`
  chain matching (three-level: torch.ops.autokernel.<name>).

### 11:04 — Audit broadening (rotary_fn, kernel_fn_dual)

- `_FusedQKVAttentionReplacement` has `self.rotary_fn(q, cos, sin)` —
  the raw pybind path for RoPE.
- `_FusedResidualRMSNormBlockReplacement` has `self.kernel_fn_dual(...)`.
- Added both to UNSAFE pattern set. Re-ran audit. Now correctly flags
  5 pre-fix UNSAFE.

### 11:05 — Phase B.1-B.4b fixes

- Edit `_FusedSwiGLUReplacement` to use `torch.ops.autokernel.silu_gate_mul`.
- Edit `_SiluGateMulReplacement` same pattern.
- Edit `_LayerNormReplacement` to fall through to `F.layer_norm` (no HIP backward exists).
- Edit `_FusedQKVAttentionReplacement` to use `torch.ops.autokernel.rotary_emb_fp32`.
- Edit `_FusedResidualRMSNormBlockReplacement` to use `torch.ops.autokernel.fused_res_rmsnorm`.
- Added `_autograd_*` attribute naming convention SAFE pattern to audit.
- Added "no UNSAFE calls → SAFE by default" verdict rule.

### 11:20 — Phase B.5 z-loss extension

- Extended `_CrossEntropyHIP.forward` with `z_loss_weight` kwarg.
- Forced tiny mode when z_loss_weight > 0 (need logits saved for backward).
- Extended `.backward` to recompute softmax from saved row_max/row_sum,
  compute `(2*z_weight/N)*lse[i]*softmax[i,:]` and add to grad_logits.
- Updated `ce_full` signature. Updated fused-mode backward to return 8 None's instead of 7.
- Added `--use-fused-zloss` flag to train_ddp.py.

### 11:22 — Phase B.6 tests

- Wrote `scripts/test_phase_b_autograd_safety.py` with 7 tests.
- 1 CPU-only test (static audit has no UNSAFE) — PASSED.
- 6 CUDA tests — skipped on Windows, will run on Machine B post-sync.
- Also ran existing regression: 54 passed, 12 skipped. Zero regressions.

### 11:25 — Phase D.A harness

- Wrote `autokernel/triton_base.py` (TritonAutogradFunction + TritonModule).
- Wrote `autokernel/triton_autotune.py` (shape+git-SHA keyed cache).
- Wrote `scripts/kernel_parity_harness.py` (fwd+bwd parity).
- Wrote `scripts/kernel_bench_harness.py` (isolated throughput).
- Wrote `knowledge/kernels/triton_author_guide.md` (authoring manual).

### 11:28 — Phase D.B fused SwiGLU

- Wrote `kernels/triton/fused_swiglu.py`:
  - `_fused_swiglu_fwd_kernel`, `_fused_swiglu_bwd_kernel` (Triton jit)
  - `_FusedSwiGLUTritonFn(torch.autograd.Function)` wrapper
  - `fused_swiglu(gate, up)` public entry with eager fallback
  - `TritonFusedSwiGLUModule` nn.Module drop-in
- Wrote `scripts/test_triton_swiglu.py` (5 tests, CUDA only).

### 11:35 — Phase E guardrails

- Added `_autokernel_autograd_preflight` helper to `train_ddp.py`.
- Wired into main after `autokernel.optimize()`. Raises RuntimeError if
  any param fails the grad check.
- Wrote `scripts/test_autokernel_autograd_safety.py` CI smoke test (2 tests, CUDA).
- Updated `CONSTRAINTS.md` autokernel section with new rules.

### 11:40 — Phase F docs

- Wrote `knowledge/training/autograd_safety_hip_kernels.md`.
- Updated AGENTS.md training gotchas (3 revised/new bullets).
- Updated STATUS.md with remediation summary block.

### 11:50 — B4 probe completed

- 2000 steps, loss 2.5144 at step 2000.
- Fetched rank0.log + diag.jsonl (15 MB) + train_log.jsonl.
- Analyzed diag: 14 of 61 params frozen. **OdinHalo silent-freeze confirmed.**
- Wrote `docs/perf/odinhalo-b4-findings.md`.

### 11:51 — Phase A.3 launched on Machine B

- `bash scripts/audit_phase_a3_batch.sh` via detached nohup.
- Already processing odin_flat V0/V1/V3.
- 4 model variants (odin_flat_30m V3, odin_flat_30m_mini all) fail
  early; batch continues.

### 11:52 — Phase C launched on Machine A

- `bash scripts/probe_phase_c_odinflat.sh` via detached nohup.
- **Phase E.3 preflight PASSED**: "[autokernel] preflight OK".
- Compile warmup starting.

### 11:55 — Session documentation

- Wrote this execution log + master session record.

---

## Patterns worth remembering

### Launch reliability checklist

1. `source venv/bin/activate` always before torchrun
2. `set -eo pipefail` not `set -euo pipefail` (activate has unbound vars)
3. Default port 29500 sometimes has stale state; use 29510+ to be safe
4. `launch_ddp.sh` doesn't source venv for rank 0 — wrappers must do it
5. After `ssh setsid nohup ... &`, the outer ssh returns instantly; check with a second ssh `ps aux`

### Audit classification rules (final)

- **SAFE**: no UNSAFE calls in forward
- **CONDITIONAL-SAFE**: UNSAFE call gated behind a registered-autograd-op
  fallback check (e.g. `if self._autograd_op is not None: use safe path`)
- **UNSAFE**: raw `self.*_fn(`, `self.rotary_fn(`, `self.kernel_fn_dual(`
  in forward without fallback
- **UNKNOWN**: (should not occur post-fix; indicates a new pattern not yet
  recognized by the audit classifier)

### Diagnostic probe signature

A 50-step single-node probe with `--diag-frozen-params` is sufficient to
reveal silent-freeze bugs. Output: per-param per-step JSONL. Analysis:
per-param classification as always_none / always_zero / always_finite /
occasionally_finite.

### The "autograd-severed" signature

When you see forward output with `grad_fn=None` + upstream param
gradient = None, you have the bug. Downstream params still train
normally because they only need the output tensor VALUE for their own
gradient calculation. Training loss still descends. Only long-horizon
(1000+ step) probes reveal the quality degradation.

### What the preflight check catches

1. Missing `torch.library.custom_op` registration
2. Replacement calling raw `kernel_fn` in forward
3. Broken custom_op backward registration
4. Any future regression where a Replacement's output has
   `grad_fn=None` under training mode

### What it doesn't catch

1. Bugs in correctly-registered autograd backward (numerical errors)
2. Value-level gradient errors (wrong formula, off-by-one scale)
3. Non-leaf-parameter issues (e.g. activations that should have
   requires_grad but don't)

For value-level correctness, use `scripts/kernel_parity_harness.py`.

---

**End of execution log.**
