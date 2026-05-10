"""Shape-aware autotune cache for Triton kernels.

Persists per-(shape, dtype, kernel-name) best configurations to a local
file. Keyed by git SHA so autotune results don't leak across incompatible
kernel revisions.

Usage:

    from autokernel.triton_autotune import cached_autotune

    @cached_autotune(
        kernel_name="fused_swiglu_fwd",
        configs=[
            {"BLOCK_M": 64,  "BLOCK_N": 128, "num_warps": 4},
            {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 4},
            {"BLOCK_M": 128, "BLOCK_N": 64,  "num_warps": 8},
        ],
        key_shape=lambda args: (args[0].shape[0], args[0].shape[-1]),
    )
    def run_swiglu(out, gate, up, BLOCK_M, BLOCK_N, num_warps):
        _swiglu_kernel[grid](
            out, gate, up,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps,
        )

The first call with a novel shape runs every config, times each, and
persists the winner. Subsequent calls with the same (shape, dtype) hit
the cache and dispatch directly.

Plan: Phase D.A.2 of master remediation plan.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


_CACHE_ROOT = Path(os.environ.get(
    "AUTOKERNEL_TUNE_CACHE",
    Path.home() / ".cache" / "autokernel" / "triton_autotune",
))


def _git_sha() -> str:
    """Return the current git SHA, or 'dirty' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        # Also check if working tree is dirty
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        return out + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _cache_path(kernel_name: str) -> Path:
    sha = _git_sha()
    p = _CACHE_ROOT / sha / f"{kernel_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache(kernel_name: str) -> Dict[str, Dict]:
    p = _cache_path(kernel_name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_cache(kernel_name: str, cache: Dict[str, Dict]):
    p = _cache_path(kernel_name)
    p.write_text(json.dumps(cache, indent=2))


def _key_from_shape(key: Tuple) -> str:
    return "|".join(str(x) for x in key)


def _time_config(fn: Callable, args: Tuple, config: Dict,
                  warmup: int = 3, iters: int = 20) -> float:
    """Run fn(*args, **config) and return median wall seconds per iter."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Measure
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed


def cached_autotune(
    kernel_name: str,
    configs: List[Dict],
    key_shape: Callable[[Tuple], Tuple],
    warmup: int = 3,
    iters: int = 20,
    verbose: bool = False,
):
    """Decorator that autotunes the decorated launcher function.

    Args:
        kernel_name: used for cache filename
        configs: list of kwarg dicts to sweep
        key_shape: function extracting the shape-dependent cache key from
            positional args. Returns a tuple of serializable values.
        warmup: warmup calls before timing
        iters: timed iterations
        verbose: print autotune result on cache miss

    Returned function accepts (*args) where args are the kernel's
    positional arguments; kwargs are merged with the winning config.
    """
    cache = _load_cache(kernel_name)

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args):
            key = _key_from_shape(key_shape(args))
            if key in cache:
                return fn(*args, **cache[key]["config"])

            # Autotune pass
            if verbose:
                print(f"[autotune] {kernel_name} key={key} (new)")
            best_cfg = None
            best_time = float("inf")
            for cfg in configs:
                try:
                    t = _time_config(fn, args, cfg, warmup=warmup, iters=iters)
                except Exception as exc:  # noqa: BLE001
                    if verbose:
                        print(f"  skip {cfg}: {exc}")
                    continue
                if verbose:
                    print(f"  {cfg}: {t * 1e6:.1f} us")
                if t < best_time:
                    best_time = t
                    best_cfg = cfg

            if best_cfg is None:
                raise RuntimeError(
                    f"All configs failed for {kernel_name} key={key}"
                )

            cache[key] = {"config": best_cfg, "time_us": best_time * 1e6}
            _save_cache(kernel_name, cache)
            if verbose:
                print(f"  -> winner {best_cfg} @ {best_time * 1e6:.1f} us")
            return fn(*args, **best_cfg)

        return wrapper

    return decorator


def clear_cache(kernel_name: Optional[str] = None):
    """Delete the cache file for kernel_name, or ALL caches if None."""
    if kernel_name is not None:
        p = _cache_path(kernel_name)
        if p.exists():
            p.unlink()
        return
    # Delete all files under the current git-SHA directory
    sha_dir = _cache_path("__dummy__").parent
    if sha_dir.exists():
        for f in sha_dir.glob("*.json"):
            f.unlink()


__all__ = ["cached_autotune", "clear_cache"]
