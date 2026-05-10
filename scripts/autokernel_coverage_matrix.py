"""Phase A.2 pattern-firing matrix: run `autokernel.optimize` across all 14
production-candidate models and record which patterns fire on each.

Uses the same mechanism as scripts/diag_autokernel_patterns.py but iterates
a curated list of model classes.

Outputs:
  - docs/perf/autokernel-pattern-coverage-matrix.md
  - docs/perf/autokernel-pattern-coverage-matrix.json

Run on a GPU-equipped host (needs CUDA for autokernel.optimize).

Usage:
    python scripts/autokernel_coverage_matrix.py

Plan: Phase A.2 of master remediation plan.
"""

from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


# Curated model list. Each entry: (model_path, class_name, construction_kwargs, notes).
# Mini variants included for quick audit on resource-constrained probes.
MODELS = [
    ("models/odin_flat.py",   "OdinFlat",          {},                    "Sprint 3A primary"),
    ("models/odin_flat.py",   "OdinFlatAblation",  {},                    "OdinFlat ablation variant"),
    ("models/odin_flat.py",   "OdinFlatMini",      {},                    "tiny OdinFlat for CI"),
    ("models/odin_flat_30m.py","OdinFlat30M",      {},                    "Sprint 1.5 probe model"),
    ("models/odin_flat_30m.py","OdinFlat30MMini",  {},                    "tiny OdinFlat30M"),
    ("models/odin_halo.py",   "OdinHalo",          {},                    "Sprint 3B primary (looped)"),
    ("models/odin_halo.py",   "OdinHaloAblation",  {},                    "OdinHalo ablation variant"),
    ("models/odin_halo.py",   "OdinHaloMini",      {},                    "tiny OdinHalo for CI"),
    ("models/vidar_halo.py",  "VidarHalo",         {},                    "VidarHalo (research)"),
    ("models/baldr_halo.py",  "BaldrHalo",         {},                    "BaldrHalo (research)"),
    ("models/chimera_halo.py","ChimeraHalo",       {},                    "ChimeraHalo (research)"),
    ("models/fenrir_halo.py", "FenrirHalo",        {},                    "FenrirHalo (research)"),
    ("models/tyr_halo.py",    "TyrHalo",           {},                    "TyrHalo (research)"),
    ("models/jormungandr_halo.py", "JormungandrHalo", {},                 "JormungandrHalo (research)"),
]


def _load_class(model_path: str, class_name: str):
    p = REPO_ROOT / model_path
    spec = spec_from_file_location(p.stem, p)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _audit_one(entry):
    model_path, class_name, kwargs, notes = entry
    print(f"\n=== {class_name} ({model_path}) ===")
    print(f"    notes: {notes}")
    try:
        cls = _load_class(model_path, class_name)
    except Exception as exc:
        print(f"  LOAD FAILED: {exc}")
        return {"class": class_name, "path": model_path, "error": str(exc)}

    try:
        model = cls(**kwargs)
    except Exception as exc:
        print(f"  INSTANTIATE FAILED: {exc}")
        return {"class": class_name, "path": model_path, "error": f"ctor: {exc}"}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    params: {n_params / 1e6:.1f}M")

    if torch.cuda.is_available():
        try:
            model = model.cuda()
        except Exception as exc:
            print(f"  CUDA move FAILED: {exc}")
            return {"class": class_name, "path": model_path, "error": f"cuda: {exc}"}

    # snapshot BEFORE
    before_map = {name: type(mod).__name__ for name, mod in model.named_modules()}

    # apply autokernel
    try:
        import autokernel
        optimized = autokernel.optimize(model, training=True)
        report = autokernel.report(optimized)
    except Exception as exc:
        print(f"  autokernel.optimize FAILED: {exc}")
        return {"class": class_name, "path": model_path, "error": f"optimize: {exc}"}

    # snapshot AFTER
    transitions = {}
    for name, mod in optimized.named_modules():
        after_cls = type(mod).__name__
        if name in before_map and before_map[name] != after_cls:
            key = (before_map[name], after_cls)
            transitions.setdefault(key, []).append(name)

    # Print report
    patterns_applied = report.get("patterns", {})
    pattern_counts = {}
    for pname, info in patterns_applied.items():
        count = info.get("modules_replaced", 0)
        if count > 0:
            pattern_counts[pname] = count
            print(f"    pattern {pname!r}: {count} module(s)")

    return {
        "class": class_name,
        "path": model_path,
        "notes": notes,
        "n_params_M": round(n_params / 1e6, 2),
        "pattern_counts": pattern_counts,
        "transitions": {f"{a}->{b}": names[:3] + (["..."] if len(names) > 3 else [])
                        for (a, b), names in transitions.items()},
        "n_transitions": sum(len(v) for v in transitions.values()),
    }


def main():
    results = [_audit_one(e) for e in MODELS]

    # Pattern → set of models where it fired
    all_patterns = set()
    for r in results:
        if "pattern_counts" in r:
            all_patterns.update(r["pattern_counts"].keys())
    all_patterns = sorted(all_patterns)

    # Write matrix
    out_md = REPO_ROOT / "docs" / "perf" / "autokernel-pattern-coverage-matrix.md"
    out_json = REPO_ROOT / "docs" / "perf" / "autokernel-pattern-coverage-matrix.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Autokernel Pattern Coverage Matrix (Phase A.2, 2026-05-10)\n\n")
        f.write("Which `autokernel` patterns fire on which production-candidate model. "
                "Ground truth from `autokernel.report()` after `optimize(model, training=True)`.\n\n")
        f.write("Generator: `scripts/autokernel_coverage_matrix.py`.\n\n")

        # Matrix
        f.write("## Pattern × Model matrix (count of modules replaced)\n\n")
        header = "| Model | " + " | ".join(f"`{p}`" for p in all_patterns) + " |"
        f.write(header + "\n")
        f.write("|---|" + "|".join(["---:"] * len(all_patterns)) + "|\n")
        for r in results:
            if "error" in r:
                row = f"| **{r['class']}** (FAILED: {r['error'][:40]}) | " + " | ".join(["—"] * len(all_patterns)) + " |"
            else:
                cells = []
                for p in all_patterns:
                    c = r["pattern_counts"].get(p, 0)
                    cells.append(str(c) if c > 0 else ".")
                row = f"| `{r['class']}` ({r['n_params_M']}M) | " + " | ".join(cells) + " |"
            f.write(row + "\n")
        f.write("\n")

        # Per-model detail
        f.write("## Per-model transition detail\n\n")
        for r in results:
            if "error" in r:
                f.write(f"### `{r['class']}` — FAILED\n\n`{r['error']}`\n\n")
                continue
            f.write(f"### `{r['class']}` — {r['n_params_M']}M params\n\n")
            f.write(f"_{r['notes']}_\n\n")
            if not r["transitions"]:
                f.write("_(no modules replaced)_\n\n")
                continue
            for transition, examples in r["transitions"].items():
                f.write(f"- `{transition}` ({len(examples)} shown): examples {examples}\n")
            f.write("\n")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "all_patterns": all_patterns,
        }, f, indent=2, default=str)

    print(f"\nWrote {out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
