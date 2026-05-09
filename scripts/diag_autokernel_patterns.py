"""Phase I diagnostic: enumerate exactly which autokernel replacements
apply to OdinFlat after the Phase 0 escape hatches.

Produces a ground-truth inventory:
  - Which patterns fire (via autokernel.report)
  - Which specific modules each pattern replaces (name, class before, class after)
  - Total parameter count covered by each pattern

Run:
    python scripts/diag_autokernel_patterns.py

Also runs on OdinHalo for comparison (shows WHY Phase 0.3 on OdinHalo worked).
"""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402


def _load(model_path: str, class_name: str):
    p = _REPO_ROOT / model_path
    spec = spec_from_file_location(p.stem, p)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _inventory(model_name: str, model_path: str, class_name: str,
               use_mup: bool = False) -> dict:
    """Instantiate the model, run autokernel, return {pattern, module_name, before, after}."""
    import autokernel

    print(f"\n=== {model_name}: {model_path}::{class_name} ===")
    cls = _load(model_path, class_name)
    kwargs = {}
    if use_mup and "OdinFlat" in class_name:
        kwargs["use_mup"] = True
        kwargs["mup_base_width"] = 256

    model = cls(**kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params / 1e6:.1f}M; use_mup={use_mup}")

    if torch.cuda.is_available():
        model = model.cuda()

    # Snapshot BEFORE
    before_map = {name: type(mod).__name__ for name, mod in model.named_modules()}

    # Apply autokernel
    model = autokernel.optimize(model, training=True)
    report = autokernel.report(model)

    # Snapshot AFTER
    changed = []
    for name, mod in model.named_modules():
        after_cls = type(mod).__name__
        if name in before_map and before_map[name] != after_cls:
            # Count own params (not including children) where possible
            own_param_count = sum(p.numel() for p in mod.parameters(recurse=False))
            changed.append({
                "name": name,
                "before": before_map[name],
                "after": after_cls,
                "own_params": own_param_count,
            })

    # Print summary
    print(f"\n  autokernel.report() patterns applied:")
    for pname, info in report["patterns"].items():
        print(f"    - {pname:30s} replaced {info['modules_replaced']:>3d} module(s)  "
              f"({info['op_speedup']} expected)")

    # Group changed modules by (before -> after)
    print(f"\n  Module class transitions ({len(changed)} modules):")
    groups = {}
    for c in changed:
        key = (c["before"], c["after"])
        groups.setdefault(key, []).append(c["name"])
    for (before, after), names in sorted(groups.items()):
        sample = names[:3]
        more = f"  (+{len(names) - 3} more)" if len(names) > 3 else ""
        print(f"    {before:28s} -> {after:40s} ({len(names):>3d}x)")
        for n in sample:
            print(f"        {n}")
        if more:
            print(f"       {more}")

    return {
        "model": f"{model_path}::{class_name}",
        "n_params_M": round(n_params / 1e6, 2),
        "use_mup": use_mup,
        "patterns_applied": report["patterns"],
        "changed_modules": changed,
        "n_changed": len(changed),
    }


def main() -> int:
    results = {}
    # OdinFlat production config (with μP, matches Sprint 3A)
    results["odin_flat_mup"] = _inventory(
        "OdinFlat (Sprint 3A config, μP on)",
        "models/odin_flat.py", "OdinFlat",
        use_mup=True,
    )
    # OdinFlat without μP (Phase 0.4 repro)
    results["odin_flat_base"] = _inventory(
        "OdinFlat (base, no μP) — Phase 0.4 probe config",
        "models/odin_flat.py", "OdinFlat",
        use_mup=False,
    )
    # OdinHalo for comparison (shows why Phase 0.3 worked)
    results["odin_halo_base"] = _inventory(
        "OdinHalo (Sprint 3B config)",
        "models/odin_halo.py", "OdinHalo",
        use_mup=False,
    )

    # Cross-compare
    print("\n=== Cross-comparison ===")
    print(f"  {'Pattern':30s}  {'OdinFlat(μP)':>14s}  {'OdinFlat(base)':>14s}  {'OdinHalo':>10s}")
    all_patterns = set()
    for r in results.values():
        all_patterns.update(r["patterns_applied"].keys())
    for p in sorted(all_patterns):
        flat_mup = results["odin_flat_mup"]["patterns_applied"].get(p, {}).get("modules_replaced", 0)
        flat_base = results["odin_flat_base"]["patterns_applied"].get(p, {}).get("modules_replaced", 0)
        halo = results["odin_halo_base"]["patterns_applied"].get(p, {}).get("modules_replaced", 0)
        print(f"  {p:30s}  {flat_mup:>14d}  {flat_base:>14d}  {halo:>10d}")

    # Write JSON + MD
    import json
    out_json = _REPO_ROOT / "docs/perf/odinflat-autokernel-inventory.json"
    out_md = _REPO_ROOT / "docs/perf/odinflat-autokernel-inventory.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_json}")

    # Write a concise MD
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# OdinFlat autokernel pattern inventory (Phase I diagnostic)\n\n")
        f.write("Runtime ground truth of which `autokernel` replacements apply after "
                "Phase 0's `_skip_autokernel` escape hatches.\n\n")
        f.write("Generated by `scripts/diag_autokernel_patterns.py`.\n\n")

        f.write("## Cross-comparison (pattern → modules replaced)\n\n")
        f.write(f"| Pattern | OdinFlat (μP) | OdinFlat (base) | OdinHalo |\n")
        f.write(f"|---|---:|---:|---:|\n")
        for p in sorted(all_patterns):
            flat_mup = results["odin_flat_mup"]["patterns_applied"].get(p, {}).get("modules_replaced", 0)
            flat_base = results["odin_flat_base"]["patterns_applied"].get(p, {}).get("modules_replaced", 0)
            halo = results["odin_halo_base"]["patterns_applied"].get(p, {}).get("modules_replaced", 0)
            f.write(f"| `{p}` | {flat_mup} | {flat_base} | {halo} |\n")
        f.write("\n")

        for key, r in results.items():
            f.write(f"## {key}\n\n")
            f.write(f"Model: `{r['model']}` — {r['n_params_M']}M params; use_mup={r['use_mup']}\n\n")
            f.write(f"Patterns applied ({len(r['patterns_applied'])}):\n\n")
            for pn, info in r["patterns_applied"].items():
                f.write(f"- `{pn}` — {info['modules_replaced']} modules ({info['op_speedup']} expected)\n")
            f.write(f"\nChanged modules ({r['n_changed']} total):\n\n")
            # Group
            groups = {}
            for c in r["changed_modules"]:
                key2 = (c["before"], c["after"])
                groups.setdefault(key2, []).append(c["name"])
            for (b, a), names in sorted(groups.items()):
                f.write(f"- `{b}` → `{a}` ({len(names)}x): "
                        f"examples {names[:3]}{' ...' if len(names) > 3 else ''}\n")
            f.write("\n")

    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
