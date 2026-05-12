"""Training stability detection and recovery.

Provides StabilityGuard (4-mechanism instability detector with auto-rollback)
and save_nan_forensics (full diagnostic dump + replay bundle for post-mortem).

Extracted from scripts/train_ddp.py to give both training entry points access
and to provide unit-testable interfaces.
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn


class StabilityGuard:
    """Detects training instability and auto-recovers from last checkpoint.

    Four detection mechanisms:
    1. NaN loss -- immediate rollback
    2. Loss spike -- loss > spike_factor * EMA triggers rollback
    3. Parameter NaN -- periodic weight scan catches silent corruption
    4. GradScaler scale collapse (2026-05-07+) -- scale < scale_floor triggers
       rollback. Closes the gap where sustained microstep-NaN causes scale
       to halve repeatedly without ever tripping a loss-level NaN (step_loss
       is NaN-filtered in train_ddp.py; runaway-backoff silently drives
       scale -> 0 which deads training).

    On trigger: reload last checkpoint, reduce LR by decay_factor, continue.

    Interface:
        guard = StabilityGuard(checkpoint_dir, ...)
        # Each step:
        if not guard.check_loss(loss_val, step): trigger rollback
        if not guard.check_params(model, step): trigger rollback
        if not guard.check_scaler(scaler, step): trigger rollback
        # On trigger:
        step, ok = guard.rollback(model, optimizer, device, scaler)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_rollbacks: int = 5,
        spike_factor: float = 2.0,
        lr_decay_on_rollback: float = 0.5,
        ema_alpha: float = 0.99,
        param_check_interval: int = 500,
        scale_floor: float = 1.0,
        rank: int = 0,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_rollbacks = max_rollbacks
        self.spike_factor = spike_factor
        self.lr_decay_on_rollback = lr_decay_on_rollback
        self.ema_alpha = ema_alpha
        self.param_check_interval = param_check_interval
        # fp16-stability (smoke finding 2026-05-07): scale below this floor
        # signals runaway backoff; default 1.0 catches scale collapse before
        # it reaches zero and kills training.
        self.scale_floor = scale_floor
        self.rank = rank

        self.loss_ema = None
        self.rollback_count = 0
        self.last_good_step = 0
        self._steps_seen = 0
        self._spike_warmup = 2000

    def _find_latest_checkpoint(self, before_step: int = None):
        """Find the most recent valid checkpoint."""
        ckpt_dir = self.checkpoint_dir
        if not os.path.exists(ckpt_dir):
            return None
        checkpoints = []
        for f in os.listdir(ckpt_dir):
            if f.startswith("step_") and f.endswith(".pt"):
                try:
                    step = int(f.replace("step_", "").replace(".pt", ""))
                except ValueError:
                    step = 0
                if before_step is None or step < before_step:
                    checkpoints.append((step, os.path.join(ckpt_dir, f)))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0]  # (step, path)

    def check_loss(self, loss_val: float, step: int) -> bool:
        """Returns True if loss is healthy, False if rollback needed."""
        self._steps_seen += 1

        if math.isnan(loss_val) or math.isinf(loss_val):
            if self.rank == 0:
                print(f"  [StabilityGuard] NaN/Inf loss at step {step}")
            return False

        if self.loss_ema is None:
            self.loss_ema = loss_val
            return True

        self.loss_ema = self.ema_alpha * self.loss_ema + (1 - self.ema_alpha) * loss_val

        # Skip spike detection during warmup (loss is volatile early)
        if self._steps_seen < self._spike_warmup:
            self.last_good_step = step
            return True

        if loss_val > self.spike_factor * self.loss_ema:
            if self.rank == 0:
                print(f"  [StabilityGuard] Loss spike at step {step}: "
                      f"{loss_val:.4f} > {self.spike_factor}x EMA {self.loss_ema:.4f}")
            return False

        self.last_good_step = step
        return True

    def check_params(self, model: nn.Module, step: int) -> bool:
        """Periodic parameter NaN check. Returns True if healthy."""
        if step % self.param_check_interval != 0:
            return True
        for name, p in model.named_parameters():
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                if self.rank == 0:
                    print(f"  [StabilityGuard] NaN/Inf in param '{name}' at step {step}")
                return False
        return True

    def check_scaler(self, scaler, step: int) -> bool:
        """fp16-stability: detect GradScaler scale collapse.

        Returns True if healthy. The default scaler init_scale is 1024; any
        value below scale_floor (default 1.0) means at least 10 consecutive
        overflow backoffs have happened -- a clear sign that fp16 is not
        sufficient for current training state. Triggers rollback.
        """
        try:
            current_scale = float(scaler.get_scale())
        except Exception:
            return True  # Can't read scale; don't trigger a false positive.
        if current_scale < self.scale_floor:
            if self.rank == 0:
                print(f"  [StabilityGuard] GradScaler scale collapse at step {step}: "
                      f"scale={current_scale:.2e} < floor={self.scale_floor:.2e}")
            return False
        return True

    def rollback(self, model, optimizer, device, scaler=None):
        """Reload last good checkpoint and reduce LR. Returns (step, success).

        fp16-stability R3 (2026-05-07): when ``scaler`` is passed, also
        halves its ``growth_interval`` so scale recovery is more
        conservative after a NaN trip. This reduces the chance of a
        second overflow at the same scale level.
        """
        if self.rollback_count >= self.max_rollbacks:
            if self.rank == 0:
                print(f"  [StabilityGuard] Max rollbacks ({self.max_rollbacks}) reached, aborting")
            return -1, False

        latest = self._find_latest_checkpoint(before_step=None)
        if latest is None:
            if self.rank == 0:
                print("  [StabilityGuard] No checkpoint found for rollback")
            return -1, False

        ckpt_step, ckpt_path = latest
        if self.rank == 0:
            print(f"  [StabilityGuard] Rolling back to step {ckpt_step} from {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        raw = model.module if hasattr(model, "module") else model
        raw = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
        raw.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                if self.rank == 0:
                    print("  [StabilityGuard] Could not restore optimizer, using fresh state")

        # Reduce LR
        decay = self.lr_decay_on_rollback
        for pg in optimizer.param_groups:
            old_lr = pg["lr"]
            pg["lr"] = old_lr * decay
            if self.rank == 0 and pg is optimizer.param_groups[0]:
                print(f"  [StabilityGuard] LR reduced: {old_lr:.2e} -> {pg['lr']:.2e}")

        # fp16-stability R3: also halve the scaler's growth_interval (private
        # attribute; best-effort if torch version changes the name). Floors at
        # 100 so growth doesn't stall completely.
        if scaler is not None:
            try:
                old_gi = scaler._growth_interval
                scaler._growth_interval = max(100, old_gi // 2)
                if self.rank == 0:
                    print(f"  [StabilityGuard] scaler.growth_interval: "
                          f"{old_gi} -> {scaler._growth_interval}")
            except AttributeError:
                pass

        self.rollback_count += 1
        self.loss_ema = None  # reset EMA after rollback

        if self.rank == 0:
            print(f"  [StabilityGuard] Rollback #{self.rollback_count} complete, "
                  f"resuming from step {ckpt_step}")

        del ckpt
        return ckpt_step, True


def save_nan_forensics(
    dump_dir: str,
    step: int,
    trigger: str,
    loss_val: float,
    batch_idx: int,
    input_ids: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    doc_ids: Optional[torch.Tensor],
    scaler,
    model: nn.Module,
    recent_grad_norms,
    monitor=None,
    global_step: int = 0,
    optimizer=None,
    args=None,
    consecutive_grad_skips: int = 0,
) -> Optional[str]:
    """fp16-stability R1 + v3 T-5 C.0: dump diagnostic state + replay bundle.

    Writes ``{dump_dir}/nan_dump_step_{step}.pt`` AND (when optimizer/args
    are provided) ``{dump_dir}/replay-bundle-step-{step}/`` containing
    everything a future replay executor needs to re-run the failing batch
    offline under alternate configs.

    Writes ``{dump_dir}/nan_dump_step_{step}.pt`` capturing everything a
    post-mortem needs:

      - The offending batch (so it can be re-run offline without
        reproducing a full 2-epoch run).
      - Per-parameter weight max-abs (flags which params are out of range).
      - GradScaler scale + growth_tracker (tells us if scale runaway was
        the proximate cause).
      - Recent grad-norm history (last 50 steps) -- a slow drift upward is
        a canonical pre-NaN signature.
      - ActivationMonitor's per-layer max-abs if D1 is active.
      - trigger: "nan_loss" | "loss_spike" | "param_nan" | "grad_skips"
        | "scale_collapse"

    Rank 0 only. Fail-quiet: any exception is caught and logged; we never
    let the dump failure prevent the subsequent rollback.
    """
    import traceback
    try:
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"nan_dump_step_{step}.pt")

        raw = model.module if hasattr(model, "module") else model
        raw = raw._orig_mod if hasattr(raw, "_orig_mod") else raw

        # Per-parameter max-abs (cheap scan; avoids full state-dict dump)
        weight_maxabs = {}
        for name, p in raw.named_parameters():
            with torch.no_grad():
                try:
                    weight_maxabs[name] = float(p.data.detach().abs().max().item())
                except Exception:
                    weight_maxabs[name] = None

        # GradScaler state
        scaler_state = {}
        try:
            scaler_state["scale"] = float(scaler.get_scale())
        except Exception:
            scaler_state["scale"] = None
        try:
            # private attributes; best-effort
            scaler_state["growth_tracker"] = int(scaler._growth_tracker)
            scaler_state["growth_interval"] = int(scaler._growth_interval)
            scaler_state["init_scale"] = float(scaler._init_scale)
        except Exception:
            pass

        dump = {
            "step": int(step),
            "global_step": int(global_step),
            "trigger": str(trigger),
            "loss_val": float(loss_val) if loss_val is not None else None,
            "microbatch_idx": int(batch_idx) if batch_idx is not None else None,
            "input_ids_cpu": input_ids.detach().cpu() if input_ids is not None else None,
            "targets_cpu": targets.detach().cpu() if targets is not None else None,
            "doc_ids_cpu": doc_ids.detach().cpu() if doc_ids is not None else None,
            "scaler_state": scaler_state,
            "weight_maxabs": weight_maxabs,
            "grad_norm_history": list(recent_grad_norms),
            "activation_stats": monitor.current_stats() if monitor is not None else None,
            "consecutive_grad_skips": consecutive_grad_skips,
        }
        torch.save(dump, path)
        print(f"  [fp16-forensics] NaN dump -> {path} "
              f"(trigger={trigger}, scale={scaler_state.get('scale')})")

        # v3 T-5 C.0: replay bundle (optimizer + args present => full bundle).
        # A future session can add scripts/replay_step.py to re-run the batch
        # under alternate configs for diagnosis.
        if optimizer is not None and args is not None:
            try:
                bundle_dir = os.path.join(dump_dir, f"replay-bundle-step-{step}")
                os.makedirs(bundle_dir, exist_ok=True)

                # batch tensors
                torch.save(
                    {
                        "input_ids": input_ids.detach().cpu() if input_ids is not None else None,
                        "targets": targets.detach().cpu() if targets is not None else None,
                        "doc_ids": doc_ids.detach().cpu() if doc_ids is not None else None,
                        "batch_idx": int(batch_idx) if batch_idx is not None else None,
                    },
                    os.path.join(bundle_dir, "batch.pt"),
                )

                # full model state BEFORE the failing step (model is still
                # pre-optimizer-step state here since we dump before rollback)
                torch.save(
                    {"model": raw.state_dict()},
                    os.path.join(bundle_dir, "model_state.pt"),
                )

                # optimizer + scaler state
                optim_bundle = {
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                }
                torch.save(optim_bundle, os.path.join(bundle_dir, "optim_state.pt"))

                # RNG state (torch + numpy) for deterministic replay
                try:
                    import numpy as _np
                    rng = {
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "numpy": _np.random.get_state(),
                    }
                    torch.save(rng, os.path.join(bundle_dir, "rng.pt"))
                except Exception:
                    pass

                # Config snapshot: all args + git SHA + env flags.
                import json as _json
                import subprocess as _sp
                try:
                    git_sha = _sp.check_output(
                        ["git", "rev-parse", "HEAD"], stderr=_sp.DEVNULL
                    ).decode().strip()
                except Exception:
                    git_sha = "unknown"
                ak_envs = {
                    k: os.environ.get(k, "")
                    for k in [
                        "AUTOKERNEL_FIX_ROPE_GATE",
                        "AUTOKERNEL_CAUSAL_CONV_SHIM",
                        "AUTOKERNEL_SPECTRA_BRANCHLESS",
                        "HSA_OVERRIDE_GFX_VERSION",
                        "TORCH_COMPILE_MODE",
                    ]
                }
                # args namespace -> plain dict (drop non-serializable keys)
                args_dict = {}
                for k, v in vars(args).items():
                    try:
                        _json.dumps(v)
                        args_dict[k] = v
                    except Exception:
                        args_dict[k] = repr(v)
                cfg = {
                    "git_sha": git_sha,
                    "env": ak_envs,
                    "args": args_dict,
                    "trigger": trigger,
                    "step": int(step),
                    "global_step": int(global_step),
                }
                with open(os.path.join(bundle_dir, "config.json"), "w", encoding="utf-8") as f:
                    _json.dump(cfg, f, indent=2, default=str)

                # Activation snapshot (last N samples, if monitor has any)
                if monitor is not None:
                    try:
                        stats = monitor.current_stats()
                        with open(os.path.join(bundle_dir, "activation_stats_window.json"), "w", encoding="utf-8") as f:
                            _json.dump(stats, f, indent=2, default=str)
                    except Exception:
                        pass

                print(f"  [fp16-forensics] replay bundle -> {bundle_dir}")
            except Exception as exc:
                print(f"  [fp16-forensics] WARNING: replay bundle dump failed: "
                      f"{type(exc).__name__}: {exc}")
        return path
    except Exception as exc:
        print(f"  [fp16-forensics] WARNING: NaN dump failed: "
              f"{type(exc).__name__}: {exc}")
        try:
            traceback.print_exc()
        except Exception:
            pass
        return None
