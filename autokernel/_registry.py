"""
AutoKernel — Optimization engine.

Walks a model's module tree, matches optimization patterns, and applies
replacements.  Tracks state for restore() and report().
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

from autokernel._patterns import ALL_PATTERNS, Pattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a named submodule in the model."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _preserve_complex_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Save complex buffers before dtype cast (model.to(float16) destroys them)."""
    saved = {}
    for name, buf in model.named_buffers():
        if buf.is_complex():
            saved[name] = buf.clone()
    return saved


def _restore_complex_buffers(model: nn.Module, saved: Dict[str, torch.Tensor]) -> None:
    """Restore complex buffers after dtype cast."""
    for name, original_buf in saved.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        parent.register_buffer(parts[-1], original_buf.to(model.parameters().__next__().device),
                               persistent=False)


# ---------------------------------------------------------------------------
# State key attached to optimized models
# ---------------------------------------------------------------------------

_STATE_ATTR = "_autokernel_state"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize(
    model: nn.Module,
    dtype: torch.dtype = torch.float16,
    compile: bool = False,
    compile_mode: str = "default",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> nn.Module:
    """
    Auto-detect model architecture and apply all applicable HIP kernel optimizations.

    Args:
        model: PyTorch model (will be modified in-place).
        dtype: Target dtype (default: float16).
        compile: Also apply torch.compile with custom ops for graph-level fusion.
        compile_mode: torch.compile mode ("default", "reduce-overhead", "max-autotune").
        include: Only apply these pattern names (None = all).
        exclude: Skip these pattern names (None = skip none).

    Returns:
        The optimized model (same object, modified in-place).
    """
    # 1. Preserve complex buffers, cast dtype, restore
    complex_bufs = _preserve_complex_buffers(model)
    model = model.to(dtype=dtype)
    if complex_bufs:
        _restore_complex_buffers(model, complex_bufs)

    # 2. Move to GPU
    if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
        model = model.cuda()

    model.eval()

    # 3. Select applicable patterns
    patterns = _select_patterns(include, exclude)

    # 4. Walk modules, match and replace
    originals: Dict[str, nn.Module] = {}
    applied: Dict[str, int] = {}
    replaced_names: Set[str] = set()  # exact names replaced (not prefixes)

    # Sort by priority (highest first)
    patterns.sort(key=lambda p: -p.priority)

    for pattern in patterns:
        count = 0
        for name, module in list(model.named_modules()):
            # Skip if this exact module was already replaced by a higher-priority pattern
            if name in replaced_names:
                continue
            if pattern.matches(name, module, model):
                try:
                    replacement = pattern.apply(name, module, model)
                    originals[name] = module
                    _replace_module(model, name, replacement)
                    replaced_names.add(name)
                    count += 1
                except Exception as e:
                    print(f"  autokernel: {pattern.name} failed on {name}: {e}")
        if count > 0:
            applied[pattern.name] = count

    # 5. Optionally apply torch.compile
    if compile:
        model = _apply_compile(model, compile_mode)

    # 6. Store state for report/restore
    setattr(model, _STATE_ATTR, {
        "originals": originals,
        "applied": applied,
        "patterns": {p.name: p for p in patterns},
        "compiled": compile,
    })

    return model


def report(model: nn.Module) -> Dict[str, Any]:
    """Return a report of applied optimizations."""
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        return {"status": "not optimized", "patterns": {}}

    result = {"status": "optimized", "compiled": state["compiled"], "patterns": {}}
    for name, count in state["applied"].items():
        pattern = state["patterns"].get(name)
        result["patterns"][name] = {
            "modules_replaced": count,
            "op_speedup": f"{pattern.op_speedup:.1f}x" if pattern else "unknown",
        }
    return result


def restore(model: nn.Module) -> nn.Module:
    """Restore model to its original (unoptimized) state."""
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        return model

    for name, original in state["originals"].items():
        _replace_module(model, name, original)

    delattr(model, _STATE_ATTR)
    return model


def list_patterns() -> List[str]:
    """Return names of all available optimization patterns."""
    return [p.name for p in ALL_PATTERNS]


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _select_patterns(
    include: Optional[List[str]], exclude: Optional[List[str]]
) -> List[Pattern]:
    """Filter patterns by include/exclude lists."""
    patterns = list(ALL_PATTERNS)
    if include is not None:
        include_set = set(include)
        patterns = [p for p in patterns if p.name in include_set]
    if exclude is not None:
        exclude_set = set(exclude)
        patterns = [p for p in patterns if p.name not in exclude_set]
    return patterns


def _apply_compile(model: nn.Module, mode: str) -> nn.Module:
    """Apply torch.compile with profile.py sys.path workaround."""
    cwd = os.getcwd()
    cwd_in_path = cwd in sys.path

    try:
        # Import our custom ops first (needs CWD in path)
        import kernels.hip._torch_ops  # noqa: F401

        # Fix profile.py conflict: project's profile.py shadows stdlib's.
        # cProfile (already imported by torch) has cached the wrong 'profile'
        # module. We must replace it with the real stdlib profile.
        # Remove BOTH absolute CWD and '' (Python's CWD shorthand) from sys.path
        _removed_paths = []
        for p in [cwd, "", "."]:
            while p in sys.path:
                sys.path.remove(p)
                if p not in _removed_paths:
                    _removed_paths.append(p)

        # Force-reload stdlib profile and cProfile with clean sys.path
        import importlib
        if "profile" in sys.modules:
            pm = sys.modules["profile"]
            if hasattr(pm, "__file__") and pm.__file__ and cwd in str(pm.__file__):
                del sys.modules["profile"]
        if "cProfile" in sys.modules:
            del sys.modules["cProfile"]
        # Re-import clean versions
        stdlib_profile = importlib.import_module("profile")
        stdlib_cprofile = importlib.import_module("cProfile")

        # Compile
        if mode == "default":
            model = torch.compile(model, backend="inductor")
        else:
            model = torch.compile(model, backend="inductor", mode=mode)
    except Exception as e:
        print(f"  autokernel: torch.compile failed: {e}")
        print("  Continuing without compilation...")
    finally:
        # Restore removed paths
        for p in reversed(_removed_paths):
            if p not in sys.path:
                sys.path.insert(0, p)

    return model
