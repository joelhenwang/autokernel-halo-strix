# Autokernel Remediation Session — Full Record (2026-05-10 → 2026-05-11)

**Purpose:** Single-document account of everything attempted, shipped, learned,
and parked during the autokernel remediation work. Use this for post-session
review, onboarding a new session, and audit of decisions.

**Session origin:** Followed from `docs/perf/odinflat-throughput-final.md`
(2026-05-10 findings) → user-initiated scope expansion:

> "everything you reported should be documented and tackled"

Master plan accepted: full 14-model audit, fix every UNSAFE replacement,
ship a Triton kernel harness, add runtime+CI guardrails, document in
AGENTS.md / CONSTRAINTS.md / knowledge/.

---

## 1. Session timeline

### 1.1 Starting state

Prior-session deliverables (from `docs/perf/autokernel-deep-analysis.md`):

- Identified OdinFlat silent-freeze at 2000-step horizon (loss regression
  3.15 → 3.80).
- Phase III had already fixed `_RMSNormReplacement` autograd.
- Phase V had shown V2 (silu HIP with autograd correctly registered) ran
  slightly BELOW baseline (~30.9K vs 31.3K tok/s).
- Open questions (deferred in the earlier session):
  1. Triton rewrite of fused SwiGLU: feasible? worthwhile?
  2. Is `FusedQKVPattern` affected by the same class of bug?
  3. Are `_LayerNormReplacement`, `_SiluGateMulReplacement` broken the same way?
  4. Runtime guardrails to prevent recurrence

### 1.2 Scope decisions at session start (user-locked)

Four planning questions answered:

| Decision | Chosen | Alternative rejected |
|---|---|---|
| Audit breadth | **Full 14-model × 3-config matrix** | Production-path only |
| Triton investment depth | **Harness-first, kernels follow** | 2-day timeboxed, or full multi-week kernel portfolio |
| Sprint 3A/3B timing | **Block until Phase A+B+C complete** | Launch immediately, or partial |
| Guardrail strictness | **Warn in CI, hard error at train-launch** | Hard-fail CI, or manual only |

### 1.3 Execution phases (in order)

| Phase | Item | Outcome | Commit |
|---|---|---|---|
| prep | Track 1.1 `--profile-steps` flag | shipped, used by Track 1.3 | 5b5ccaf |
| prep | Track 3.A `--diag-frozen-params` flag | shipped, used throughout | 5b5ccaf |
| prep | Track 1.2/1.3 OdinFlat profile probe | attention forward ~7.5%, zloss 16.7% | 85f937e |
| prep | Track 2.a QKV fusion | shipped; failed 3% ship gate (+0.07%) but kept | 63de5be, 7a5a108 |
| prep | Track 2.b z-loss fp16 cast elimination | shipped | 24844a0 |
| prep | Track 3.A diag on OdinFlat V0/V1/V3 | V1 freezes 28 params; V3 clean | 52bcf92 |
| **A.1** | Static audit tool | 5 UNSAFE pre-fix → 0 UNSAFE post-fix | 1f54ec0 |
| **A.2** | 14-model coverage matrix tool | shipped, runs on Machine B post-B4 | 3dcd43e |
| **A.3** | Batch runner + analyzer | shipped, **RUNNING** on Machine B | 3dcd43e |
| **B4** | OdinHalo 2000-step probe | loss 2.51, 14/61 params frozen | 19e92c4, 093bd3d |
| **B.1-B.4b** | 5 UNSAFE replacements fixed | all SAFE / CONDITIONAL-SAFE | 5ebe594 |
| **B.5+B.6** | Fused z-loss + tests | shipped, 7 tests | f24d8dd |
| **D.A** | Triton kernel harness | 4 modules + author guide | eaccbd4 |
| **D.B** | Triton fused SwiGLU | shipped, 5 tests | 2501dd9 |
| **E** | Runtime guardrails + CI test | shipped, preflight **CONFIRMED** working | 404b140 |
| **F.2+3** | AGENTS.md + knowledge/autograd doc | shipped | 2a4dcb4 |
| **F.1** | STATUS.md remediation summary | shipped | 82e0655 |
| **C** | OdinFlat 2000-step post-fix probe | launcher shipped, **RUNNING** | fca7cf3 |

**Total commits this session: 14** (plus 4 prep commits for Track 1/2/3).

---

## 2. What was tried and did NOT work

### 2.1 launch_ddp.sh + probe orchestrator hybrid (failed twice)

First attempt at launching B4 used `bash scripts/probe_odinhalo_b4.sh`
which invoked `launch_ddp.sh` internally. Rank 1 would launch correctly
on Machine B, but rank 0 on Machine A exited silently — `rank0.log`
remained 0 bytes.

**Root cause**: `launch_ddp.sh` hardcodes rank 0's torchrun invocation
**without sourcing the venv**. When my orchestrator used `setsid nohup
bash ...` to detach, the child shell's `PATH` didn't include the venv's
`bin/` directory, so `torchrun` was not found. Rank 0 exited immediately
with "command not found" captured to /dev/null. rank0.log was empty
because torchrun itself produced no output.

Rank 1 worked because `launch_ddp.sh` explicitly `ssh`es into Machine B
with `source ~/Desktop/comfyui-rocm7.12/.venv/bin/activate` baked in.

**Second attempt** (different port to eliminate stale-state hypothesis):
same failure. Port wasn't the issue.

**Fix**: edit `probe_odinhalo_b4.sh` to source venv before calling
`launch_ddp.sh`. Workaround applied. Rank 0 launched correctly on third
attempt. (Commit `119cb78`.)

Longer-term fix deferred: `launch_ddp.sh` should source venv for rank 0
as it does for rank 1. Not touched this session to avoid scope creep.

### 2.2 `set -euo pipefail` broke every probe script

All my single-node probe scripts started with `set -euo pipefail` per
PowerShell/bash best practice. But `.venv/bin/activate` references
`CPLUS_INCLUDE_PATH` (and similar) which are unbound by default. With
`-u`, `source .venv/bin/activate` immediately dies.

**Fix**: changed all probe scripts to `set -eo pipefail` (no `-u`).
Documented in CONSTRAINTS.md as a recurring trap. (Commits `954ced2`,
`48ad46c`.)

### 2.3 Audit script false positives from comments

First version of `scripts/audit_autokernel_replacements.py` scanned
source as text. `_FusedSwiGLUReplacement`'s comments mentioned
"`torch.ops.autokernel.silu_gate_mul` has proper autograd" — the scanner
saw this as evidence the op WAS invoked, and flagged the class as
CONDITIONAL-SAFE when in reality the forward only called `self.kernel_fn`.

**Fix**: rewrite detection to walk `ast.Attribute` nodes directly. A
three-level `torch.ops.autokernel.<name>` chain in comments is not an
AST node, so comments no longer mislead the scanner.

### 2.4 Audit verdict logic bug: plain PyTorch forward = UNKNOWN

Second audit issue: `_FusedGriffinBlockReplacement`'s forward uses only
plain PyTorch ops (`torch.rsqrt`, `torch.sigmoid`, `self.pre_norm(...)`,
etc.). No SAFE-pattern matches and no UNSAFE matches either. Original
logic classified this as UNKNOWN.

**Fix**: revised verdict rule — absence of UNSAFE calls = SAFE by
default. Plain PyTorch ops don't need a SAFE-tag because they're
autograd-safe by construction. (Part of `1f54ec0`.)

### 2.5 Audit missed `self.rotary_fn` / `self.kernel_fn_dual` initially

Same audit tool initially only looked for `self.kernel_fn`. Missed
`_FusedQKVAttentionReplacement`'s `self.rotary_fn(...)` call AND
`_FusedResidualRMSNormBlockReplacement`'s `self.kernel_fn_dual(...)`.
Both are raw pybind; both are UNSAFE.

**Fix**: broadened UNSAFE pattern set to include `self.rotary_fn`,
`self.kernel_fn_dual`. Post-broadening, correctly flagged both as
UNSAFE (until Phase B fixed them).

### 2.6 Windows CP1252 couldn't print ✓ / ✗

`print(f"[{flag}] autokernel::{name}")` where `flag = "✓"` crashed the
audit script with `UnicodeEncodeError` on Windows (code page 1252 only).

**Fix**: replaced all unicode checkmarks with `[Y]`/`[N]` ASCII. Cheap.

### 2.7 A few OdinFlat30M variants fail the A.3 batch probe

`odin_flat_30m` V3 and `odin_flat_30m_mini` V0/V1/V3 failed with
`exitcode 1` during the Phase A.3 batch run. Logs show they crashed
during training startup.

**Status**: not yet root-caused (likely model constructor args or
`num_heads` incompatibility with `block=512` fallback). Batch runner
continues on failure; the analyzer will mark these rows as "(missing
data)" in the final matrix. Can be revisited in Phase A.4 if the gap
matters for final audit conclusions.

### 2.8 Pursued but deferred

- **Phase E.1 post-replacement grad_fn sanity in `autokernel.optimize()`**:
  would duplicate E.2 (CI test) and E.3 (train-launch preflight). Decided
  redundant noise for the common case; can add later if warranted.
- **Phase D.C z-loss Triton rewrite**: gated on Phase B leaving residual
  z-loss cost. Phase B.5 may have eliminated most of the 16.7% already.
  Will measure at Phase C completion.
- **Phase D.D NorMuon Newton-Schulz Triton kernel**: budgeted 2 weeks,
  not started this session. Isolated as a pure follow-up.

---

## 3. Findings

### 3.1 Static audit pre/post

7 Replacement classes in `autokernel/_patterns.py`:

| Class | Pre-fix verdict | Post-fix verdict | Root cause of pre-fix |
|---|---|---|---|
| `_RMSNormReplacement` | CONDITIONAL-SAFE | CONDITIONAL-SAFE | Phase III fix wired autograd op; raw call kept as fallback |
| `_LayerNormReplacement` | UNSAFE | SAFE | `self.kernel_fn(x, w, b)` raw pybind — **Phase B.3 fix** |
| `_SiluGateMulReplacement` | UNSAFE | SAFE | `self.kernel_fn(gate, up)` — **Phase B.2 fix** |
| `_FusedQKVAttentionReplacement` | UNSAFE | CONDITIONAL-SAFE | `self.rotary_fn(q, cos, sin)` raw pybind — **Phase B.4 fix** |
| `_FusedResidualRMSNormBlockReplacement` | UNSAFE | SAFE | `self.kernel_fn_dual(...)` raw pybind — **Phase B.4b fix** |
| `_FusedGriffinBlockReplacement` | (initially UNKNOWN) | SAFE | audit classifier improved; class already used plain PyTorch |
| `_FusedSwiGLUReplacement` | UNSAFE | SAFE | `self.kernel_fn(gate, up)` raw pybind — **Phase B.1 fix** (the KNOWN bug) |

### 3.2 B4 probe (OdinHalo with pre-B-fix --optimize-kernels)

**Primary result**: OdinHalo IS affected by the silent-freeze bug.
- 14 of 61 named params (23%) frozen across 2000 steps
- Same blast-radius pattern as OdinFlat V1:
  - 6 × `shared_layers.*.ffn.w_gate_up.weight` (always_none)
  - 6 × `shared_layers.*.ffn_norm.weight` (always_zero)
  - 2 × layer-3 attention params (`v_res_scale`, `head_gate`)
  - Total frozen weight count ≈ 38 M

**Secondary result**: Despite freeze, OdinHalo trained to loss 2.51 at
step 2000 (well below 3.8 ship-gate). Healthier than OdinFlat V1 likely
because:
- block=256 vs OdinFlat's 512 (shorter-context task)
- 3× `iter_norm` resets per forward dampens accumulated drift
- 57.6M params vs 121M (proportionally similar freeze ratio)
- Lower LR (2e-3 vs OdinFlat's 5e-3)

Means: **all previous OdinHalo `--optimize-kernels` checkpoints** (Phase
0.3 probe, Stage 1 variants S1.1–S1.4) **were trained with 23% of params
frozen at init**. Loss trajectories looked healthy but the models are
suboptimal.

### 3.3 Track 1.3 step profile (from 2026-05-10, key numbers for Phase D targeting)

| Section | % of step wall |
|---|---:|
| aten::mm (all matmuls fwd+bwd) | 37.85% |
| NorMuon optimizer NS iterations | 12.5% |
| aten::logsumexp (z-loss forward) | 11.1% |
| LogsumexpBackward | 5.6% |
| CausalConv1d fwd+bwd (HyPE) | ~5% |

Attention forward ≈ 7.5% of step (derived from FLOP ratio × forward
share of step). QKV fusion (Track 2.a) upside ≤ 1.5% — failed ship gate.

Z-loss total (fwd + bwd) = 16.7% of wall — target of Phase B.5 fused
z-loss fix.

### 3.4 Phase E.3 preflight is live and working

Confirmed on Phase C launch (2026-05-11 ~11:50 AM local):

```
  [autokernel] preflight OK: all parameters received gradients after dummy forward+backward
```

This proves:
1. `_autokernel_autograd_preflight()` successfully dispatches a dummy
   forward+backward during real training setup.
2. Phase B fixes produce a clean grad flow — every `requires_grad` param
   received a finite grad.
3. Future regressions will be caught at launch, not discovered 2000
   steps later via a diag probe.

---

## 4. Open / pending / in-flight

### 4.1 Active jobs

| Machine | Job | Started | ETA | Expected verdict |
|---|---|---|---|---|
| A | **Phase C OdinFlat** 2000-step post-fix, `--optimize-kernels --use-fused-zloss` | ~11:50 AM | ~14:30 PM (~2.5h) | loss ≤ 3.20, tok/s ≥ 28K |
| B | **Phase A.3 batch** 14-model × 3-config diag probes | ~11:47 AM | ~15:30 PM (~3-4h) | per-model blast-radius table |

### 4.2 Decision tree after Phase C completes

```
  Phase C loss @ step 2000:
    ≤ 3.20 → Sprint 3A ships with --optimize-kernels + --use-fused-zloss
    3.20 < x ≤ 3.80 → investigate gap; partial ship (maybe drop --use-fused-zloss)
    > 3.80 → Phase B fixes regressed something; root-cause before any ship

  Tok/s @ steady state:
    ≥ 28K → ship
    25-28K → ship but flag throughput in Phase D targeting
    < 25K → Phase B added overhead beyond expectations; profile
```

### 4.3 Decision tree after Phase A.3 completes

```
  For each (model, config) probe:
    always_none + always_zero count > 0 → flag as affected
    N shared model families affected → enumerate pattern × model matrix
    All models clean post-Phase-B? If we re-run V1 with FIXED code,
      the expectation is zero-freeze across all 14.
```

Current Phase A.3 data is using **PRE-Phase-B** code (I launched it
after the Phase B commit but Machine B did a fresh sync; actually the
sync happened AFTER the A.3 launch — check fresh-sync timing). Wait —
I synced Machine B at multiple points. Let me be explicit:

- Commit `5ebe594` Phase B.1-B.4 at ~11:00 AM
- Sync to B at ~11:00 AM (same commit included)
- Phase A.3 launched ~11:47 AM — **runs Phase B post-fix code**

So Phase A.3's V1 config should show **zero freeze** across all 14
models if Phase B is correct. This is effectively a broader verification
than Phase C covers.

### 4.4 Remaining roadmap

| Phase | Status | Unblocks |
|---|---|---|
| C complete | pending (~2.5h) | Sprint 3A / 3B launch |
| A.3 complete | pending (~3-4h) | Phase A.4 synthesis doc |
| A.4 synthesis | pending on A.3 | — |
| Sprint 3A run | pending on C | ~61h wall |
| Sprint 3B run | pending on C | ~48-77h wall depending on `--optimize-kernels` recipe |
| D.B e2e ship gate | pending on C baseline | Phase D.C / D.D targeting |
| D.C z-loss Triton | conditional on post-Phase-B z-loss % | — |
| D.D NorMuon NS Triton | scheduled multi-week | — |

---

## 5. Artifacts shipped (complete inventory)

### 5.1 Code

| Path | What |
|---|---|
| `autokernel/_patterns.py` | 5 Replacement classes rewired to autograd-safe paths |
| `autokernel/triton_base.py` | Triton autograd.Function base class |
| `autokernel/triton_autotune.py` | Shape+git-SHA keyed autotune cache |
| `kernel.py` | `_CrossEntropyHIP` extended with z_loss_weight parameter |
| `kernels/triton/__init__.py` | new package |
| `kernels/triton/fused_swiglu.py` | First Triton kernel: fused silu_gate_mul |
| `scripts/train_ddp.py` | `--use-fused-zloss`, `--diag-frozen-params`, `--profile-steps`, preflight |

### 5.2 Tests

| Path | Scope |
|---|---|
| `scripts/test_profile_flag.py` | argparse (9 tests) |
| `scripts/test_qkv_fusion.py` | QKV fusion (6 tests) |
| `scripts/test_phase_b_autograd_safety.py` | replacement autograd flow (7 tests, 1 CPU pass + 6 CUDA) |
| `scripts/test_autokernel_autograd_safety.py` | CI smoke (2 tests, CUDA) |
| `scripts/test_triton_swiglu.py` | Triton SwiGLU parity+bench (5 tests, CUDA) |

### 5.3 Tooling

| Path | What |
|---|---|
| `scripts/audit_autokernel_replacements.py` | Static audit (AST) — Phase A.1 |
| `scripts/autokernel_coverage_matrix.py` | 14-model pattern coverage — Phase A.2 |
| `scripts/audit_phase_a3_batch.sh` | V0/V1/V3 batch runner — Phase A.3 |
| `scripts/analyze_audit_phase_a3.py` | Per-config blast-radius aggregator — Phase A.3 analyzer |
| `scripts/analyze_diag_frozen_params.py` | Per-file diag summarizer (existed earlier) |
| `scripts/kernel_parity_harness.py` | fwd+bwd parity runner — Phase D.A.3 |
| `scripts/kernel_bench_harness.py` | Throughput bench — Phase D.A.4 |
| `scripts/probe_odinhalo_b4.sh` | OdinHalo 2000-step verification launcher |
| `scripts/probe_phase_c_odinflat.sh` | Phase C verification launcher |

### 5.4 Documentation

| Path | What |
|---|---|
| `docs/perf/autokernel-deep-analysis.md` | From 2026-05-10 — root-cause + mechanism |
| `docs/perf/autokernel-frozen-blast-radius.md` | OdinFlat V0/V1/V3 blast radius |
| `docs/perf/autokernel-static-audit.md` + `.json` | Current static audit (all SAFE) |
| `docs/perf/odinflat-qkv-fusion.md` | Track 2.a ship decision |
| `docs/perf/odinflat-step-profile.md` | Track 1.3 step profile |
| `docs/perf/odinhalo-b4-findings.md` | B4 verification result |
| `docs/perf/odinhalo-b4-{rank0.log,train-log.jsonl,diag.jsonl}` | B4 artifacts |
| `knowledge/training/autograd_safety_hip_kernels.md` | Principle + author workflow |
| `knowledge/kernels/triton_author_guide.md` | Triton kernel authoring manual |
| `CONSTRAINTS.md` | Autograd-safety rule + Triton authoring rules |
| `AGENTS.md` | Training gotchas for autokernel + `--use-fused-zloss` |
| `STATUS.md` | Autokernel remediation summary section |
| `docs/perf/session-2026-05-11-autokernel-remediation.md` | **THIS DOCUMENT** |

---

## 6. Key observations not yet in the synthesis docs

### 6.1 B4's 2.51 vs OdinFlat V1's 3.80: why the differential?

Both models had 23% of params frozen. OdinFlat loss regressed visibly
(+0.65), OdinHalo did not. The four contributing factors (block size,
iter_norm, model size, LR) are in `docs/perf/odinhalo-b4-findings.md`.

What this suggests for Sprint 3B:
- Even though OdinHalo converged despite the bug, the post-Phase-B
  Sprint 3B run should produce a **strictly better** checkpoint.
- **Re-running prior OdinHalo experiments** (Stage 1 S1.3b, Phase 0.3,
  Sprint 3 smoke) under Phase B fixes is a reasonable follow-up.
  Deferred pending post-Sprint-3B triage.

### 6.2 `_FusedQKVAttentionReplacement` was probably affected in practice, but not tested pre-fix

OdinFlat/OdinHalo use `NoPECodaAttention` with `_skip_autokernel = True`
(Phase 0 escape hatch). So `FusedQKVPattern` never matched on their
`NoPECodaAttention` instances. Other models using plain `Attention` or
`CodaAttention` (jormungandr_halo, baldr_halo, etc.) would have hit the
bug when trained with `--optimize-kernels`. None of those models are
on the Sprint 3 path.

Phase A.3 will produce empirical data for all of them.

### 6.3 `_FusedResidualRMSNormBlockReplacement` may have been overlooked entirely

This class has `kernel_fn_dual` with no related test coverage prior to
this session. The static audit initially missed it until I broadened
UNSAFE patterns. The fact that no failure has been reported suggests it
never fired in training (likely because no model on the production path
has the right module attribute naming convention). Phase A.3 will
reveal empirically.

### 6.4 Preflight's "allowed zero" list

`_autokernel_autograd_preflight` allows `v_res_scale` as a legitimate
always_none param (documented Track 3.A finding: first-layer
`NoPECodaAttention` has no `v_prev` input, so the residual scalar never
sees a gradient on step 1 of a minibatch).

If we add more "legitimate frozen" params later, update
`ALLOWED_ZERO_PATTERNS` in `scripts/train_ddp.py` (also in the CI test
`scripts/test_autokernel_autograd_safety.py`). Keep the two in sync.

### 6.5 `--use-fused-zloss` is OFF by default; Phase C ENABLES it

Phase C tests the path. If Phase C passes at loss parity, Phase F+G
follow-up should include flipping the default to ON for Sprint 3A/3B.

### 6.6 QKV fusion (Track 2.a) was kept despite 0.07% gain

Per Track 2.a ship decision: cleaner architecture + migration hook =
zero downside, nonzero future optionality. Don't revisit unless a
larger `d_model` model is introduced.

### 6.7 Sprint 3A/3B wall-time projections

Under Phase B post-fix code with `--optimize-kernels` on OdinHalo:
- B4 ran at 31.3K tok/s aggregate. 8.9B tokens / 31.3K tok/s ≈ 79 h.
- Expected Sprint 3B (~48h projection from Phase 0.3) may be too
  optimistic; plan for 60-80 h.

Under Phase B post-fix code WITHOUT `--optimize-kernels` on OdinFlat:
- pre-fix baseline 31.3K tok/s steady-state.
- Expected Sprint 3A: ~61 h for 6.9B tokens.

Phase C provides actual numbers.

---

## 7. Git log for reference

```
fca7cf3 Phase C: OdinFlat 2000-step post-fix verification probe
093bd3d B4 findings: silent-freeze confirmed in OdinHalo too
82e0655 Phase F.1: STATUS.md autokernel remediation summary
2a4dcb4 Phase F: knowledge/autograd_safety doc + AGENTS.md update
404b140 Phase E: runtime guardrails + CI test + CONSTRAINTS.md
2501dd9 Phase D.B: Triton fused SwiGLU
eaccbd4 Phase D.A: Triton kernel harness
f24d8dd Phase B.5+6: fused z-loss + autograd-safety tests
5ebe594 Phase B.1-B.4: wire autograd-safe paths for all UNSAFE
3dcd43e Phase A.2+A.3 tooling
1f54ec0 Phase A.1: static audit tool
1c53c03 Track 3.F synthesis doc (previous session)
119cb78 B4: source venv fix
b7d896c B4: port 29511
19e92c4 B4: OdinHalo verification probe launcher
24844a0 Track 2.a ship + 2.b z-loss fp16 opt
52bcf92 Track 3.A: frozen-params blast radius
```

---

## 8. What's next in the queue

1. **Monitor Phase C** — poll `/checkpoints/phase-c-odinflat-postfix/rank0.log`
   every 5-10 min. Expected completion ~14:30 PM local.
2. **Monitor Phase A.3** — poll `/checkpoints/audit-phase-a3/_failures.log`
   and iterate through the analyzer as probes complete. Expected
   completion ~15:30 PM local.
3. **Analyze Phase C**:
   - extract loss@250/500/1000/1500/2000
   - compare to pre-session OdinFlat V0 baseline (3.15 @ 2000) and V1
     broken (3.80 @ 2000)
   - check `diag-frozen-params` output: expect zero always_none/always_zero
   - verify steady tok/s
4. **Analyze Phase A.3**:
   - run `scripts/analyze_audit_phase_a3.py`
   - confirm all models V1 now show clean grad flow
   - produce `docs/perf/autokernel-audit-2026-05-11.md` (Phase A.4)
5. **Sprint 3A launch** (if Phase C green):
   - update launcher `scripts/run_sprint3a.sh` with `--optimize-kernels`
     and `--use-fused-zloss` if validated
   - launch DDP, ~61h wall
6. **Sprint 3B launch** (after 3A starts; both run in parallel via DDP
   on A + B simultaneously… actually both use both machines so sequential):
   - with `--optimize-kernels` confirmed via Phase C
7. **Phase D.B e2e ship-gate measurement**: isolated bench (already have
   harness) + 200-step training probe vs autograd-correct HIP baseline.
8. **Phase A.4 synthesis** once A.3 completes.

---

**End of session record.** Document written at 2026-05-11 ~11:55 AM
local while Phase C and Phase A.3 both run. Will append monitoring
results and Phase C/A.4 analyses as they land.
