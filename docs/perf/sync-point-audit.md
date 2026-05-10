# T-0.3 Sync-point audit (2026-05-10)

**Scope:** catalog every `.item()` / `.cpu()` / `.tolist()` / `float(tensor)` call in training hot-path code. Hot-path = code that executes per optimizer step or more frequently.

**Method:** `rg -n --with-filename "\.item\(\)|\.cpu\(\)\.|\.tolist\(\)|float\(.*\.detach\(\)" halo_training/ scripts/train_ddp.py kernels/`

**Source profile evidence:** `docs/perf/odinflat-profile-2026-05-10/profile-summary.txt:54-56` shows **870 `aten::item` calls across 10 opt steps = 87 calls per optimizer step**. CPU spent 79% of total wall in `hipMemcpyWithStream` (32.547s / 41.174s). CPU wall is not the binding constraint on CUDA wall, but each `.item()` forces a stream sync that serializes forward/backward overlap with host-side work.

---

## Hot-path calls (per optimizer step)

### 1. `halo_training/spectra.py:88` — BIGGEST REAL COST

```python
sigma1_val = sigma1.item()
# Fast path: no clipping needed ...
if sigma1_val * (1.0 / safety_margin) <= clip_norm:
    return M
scale = (clip_norm * safety_margin) / max(sigma1_val, 1e-12)
return M * scale
```

**Frequency:** 1× per 2D param per opt step. NorMuon scope covers ~50 params (FFN `w_gate_up`, `w_down`, attention `wq/wk/wv/wo`, each of 14 layers). **~50 syncs/step.**

**Purpose:** (a) gate the "no clip needed" fast path; (b) compute scale factor.

**Proposed fix (branchless, no sync):**
```python
scale = torch.clamp(clip_norm * safety_margin / torch.clamp(sigma1, min=1e-12), max=1.0)
return M * scale
```
Preserves exact behavior when no clip is needed (scale=1.0 → multiplication is a no-op kernel). Marginal extra kernel call cost but eliminates per-param sync.

**Estimated saving:** ~1-2% step wall.

---

### 2. `halo_training/trainer.py:443` — per microstep

```python
running_loss += loss.item() * accum_steps
```

**Frequency:** 1× per microstep = 8 calls/opt step at accum=8. **~8 syncs/step.**

**Purpose:** running_loss for log aggregation.

**Proposed fix:** keep running_loss as a 0-d tensor on GPU; sync only at log interval (every N=50 steps, not every step).

```python
if rank == 0:
    running_loss_t = running_loss_t + loss.detach() * accum_steps
# at log time:
running_loss = running_loss_t.item(); running_loss_t.zero_()
```

**Estimated saving:** ~0.5-1% step wall.

---

### 3. `kernels/hip/chunked_linear_cross_entropy.py:73`

```python
n_valid_global = int(valid_global.sum().item())
```

**Frequency:** 1× per forward call. **~8 syncs/step at accum=8.**

**Purpose:** normalize CE loss by valid token count.

**Proposed fix:** return loss with `n_valid` attached as tensor; defer normalization to scalar emission point (loss.item() call site). Or compute loss as sum and divide by valid_global tensor on-device, then emit one `.item()` at log point.

**Estimated saving:** ~0.3-0.5% step wall.

---

### 4. `scripts/train_ddp.py:1468,1502,1550` — per opt step

```python
loss_val = loss.item()        # line 1468
# ...
(global_step, float(last_grad_norm.item()))  # line 1502
# ...
"grad_norm": last_grad_norm.item() ...  # line 1550
```

**Frequency:** 3× per opt step. **~3 syncs/step.**

**Purpose:** JSONL logging and console print.

**Proposed fix:** batch into single sync per log_interval. Replace per-step `.item()` with tensor accumulator, sync only at log_interval.

**Estimated saving:** ~0.3% step wall (diminishing returns; logging cost is largely masked).

---

## Medium-frequency calls (observability-only)

| File:Line | Pattern | Frequency | Suggested disposition |
|---|---|---|---|
| `halo_training/streaming.py:179,180` | `loss.item()`, `grad_norm.item()` | per step | consolidate into log_interval |
| `halo_training/trainer.py:521,579` | `grad_norm.item()`, `p.grad.norm().item()` | per log_interval | keep as-is (not per-step) |
| `halo_training/smoke.py:152,160` | `grad_norm.item()`, `loss.item()` | per microstep (smoke) | smoke-path only; not production |
| `halo_training/callbacks.py:127,182` | `h.float().norm().item()`, `param.grad.data.float().norm().item()` | callback-specific | not hot-path |

---

## Diagnostic-only calls (not hot-path)

Not currently hot but worth documenting for when `--diag-frozen-params` or `--activation-monitor` are enabled:

| File:Line | Pattern | Trigger | Cost |
|---|---|---|---|
| `scripts/train_ddp.py:217` | `p.data.detach().abs().max().item()` | NaN forensics dump only | negligible |
| `scripts/train_ddp.py:614,1280` | grad norm diagnostic | only if `--diag-frozen-params` | adds ~10 syncs/step when on |
| `halo_training/activation_monitor.py:197,224` | `maxabs.item()` | on commit (every N steps) | already deferred |

---

## Eval / alignment / DPO paths (not training hot-path)

Excluded from this audit. `halo_training/alignment.py`, `halo_training/dpo.py`, `halo_training/evaluate.py`, `halo_training/eval/*.py` all contain `.item()` calls but only execute in eval or post-training phases.

---

## Total estimated savings (T-1.2 scope)

| Fix | Sync count reduction | % step wall |
|---|---:|---:|
| SPECTRA branchless | ~50/step | 1-2% |
| trainer.py running_loss on-device | ~8/step | 0.5-1% |
| chunked_ce n_valid deferred | ~8/step | 0.3-0.5% |
| train_ddp.py per-step loss/grad_norm | ~3/step | 0.3% |
| **Total** | **~69/step** | **2-4%** |

Remaining `aten::item` baseline after fixes: ~18/step (down from 87), mostly from medium-frequency log paths that only sync at log_interval.

---

## Cross-reference to profile data

`profile-summary.txt:54-55` shows:
```
hipMemcpyWithStream    79.05%    32.547s    79.05%    32.547s   29.322ms    0 ms    0    ops    1110
aten::item             0.00%      1.689ms  79.02%    32.537s   37.398ms    ...                870
```

79% of CPU wall is spent on sync. If we reduce `aten::item` calls from 870 to ~180 (the non-hot-path residual), we'd expect corresponding reduction in `hipMemcpyWithStream` CPU wall. CUDA wall may not drop proportionally (much of sync is masked by concurrent GPU work) but step p90 latency will improve, which is good for stability telemetry quality.

---

## Items that should NOT be removed

1. `halo_training/spectra.py:88` — if we keep the `.item()`, the gating saves a multiplication kernel dispatch per param. The fix-option above eliminates sync but adds always-runs multiplication. Net is sync-saving-favored (on APU).

2. `aten::item` calls inside GradScaler / overflow detection — these are required for step skip decisions and cannot be deferred.

3. Forensics and NaN-dump paths — acceptable because they only fire on failure.

---

## Gate for T-1.2

T-1.2 (removal pass) should proceed **after v2 feedback from research engineer**. Specifically:
- SPECTRA branchless change affects clip-gating semantics in edge cases; want engineer's mathematical review before shipping.
- `running_loss` deferred aggregation changes log output cadence; want to confirm no test depends on per-step loss values.

If v2 is silent on these, proceed with SPECTRA fix first (biggest ROI + most surgical), then the others.
