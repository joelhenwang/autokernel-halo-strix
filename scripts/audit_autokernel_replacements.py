"""Phase A.1 static audit: classify every autokernel Replacement by whether
it calls a raw pybind kernel (training-unsafe) vs routes through a registered
torch.ops.autokernel.* custom op (training-safe).

Parses autokernel/_patterns.py with ast, inspects each Replacement class's
forward method body, tags each call-site.

Output:
  - stdout: per-class audit table
  - docs/perf/autokernel-static-audit.md

Usage:
    python scripts/audit_autokernel_replacements.py [--repo-root <path>]

Plan: Phase A.1 of master remediation plan.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path


# Patterns of call-site risk classification.
# SAFE: goes through dispatcher + autograd
SAFE_PATTERNS = {
    "torch.ops.autokernel",       # registered custom ops
    "_autograd_op",                # class attribute aliasing custom op
    "_autograd_rotary",            # Phase B.4 convention: _autograd_<shortname>
    "_autograd_dual",              # Phase B.4b convention
    "_autograd_",                  # generic prefix match for any _autograd_* attr
    "F.silu", "F.linear", "F.rms_norm", "F.layer_norm", "F.softmax",  # ATen ops
    "torch.nn.functional", "torch.rms_norm", "torch.layer_norm",
    "self.reference", "self._reference",  # eager fallback
}

# UNSAFE: raw pybind call that severs autograd
UNSAFE_PATTERNS = {
    "self.kernel_fn",              # generic raw-kernel field name
    "self._kernel_fn",
    "self.hip_kernel",
    "self.kernel_fn_dual",         # FusedResidualRMSNormBlock
    "self.rotary_fn",              # FusedQKVAttention RoPE call
    "_hip(",                       # direct _hip-suffixed pybind call
    "mod.",                        # dynamic module import pattern (e.g., mod.rmsnorm_hip)
    "_get_fwd_module",             # pattern in kernel.py / CE; inspect but not per se unsafe
}


def classify_call(call_source: str) -> str:
    """Return 'SAFE', 'UNSAFE', or 'UNKNOWN'."""
    src = call_source.strip()
    for pat in SAFE_PATTERNS:
        if pat in src:
            return "SAFE"
    for pat in UNSAFE_PATTERNS:
        if pat in src:
            return "UNSAFE"
    return "UNKNOWN"


def analyze_patterns(patterns_path: Path):
    """Walk autokernel/_patterns.py, find each Replacement class, classify its forward()."""
    src = patterns_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(patterns_path))

    class_infos = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        name = node.name
        if not name.startswith("_") or "Replacement" not in name:
            continue  # only nn.Module replacements (by convention, prefixed with _)

        # find forward method
        forward_node = None
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                forward_node = item
                break
        if forward_node is None:
            continue

        # extract all Call expressions in forward() and their source line
        calls_info = []
        forward_src_lines = ast.get_source_segment(src, forward_node).splitlines()
        for inner in ast.walk(forward_node):
            if isinstance(inner, ast.Call):
                try:
                    call_src = ast.get_source_segment(src, inner)
                except Exception:
                    call_src = ast.dump(inner.func)
                if call_src is None:
                    continue
                verdict = classify_call(call_src)
                if verdict != "UNKNOWN":  # skip generic python calls
                    calls_info.append({
                        "src": call_src.strip().split("\n")[0][:120],
                        "lineno": inner.lineno,
                        "verdict": verdict,
                    })

        # Also find torch.ops.autokernel.<name> references via actual AST attribute
        # chains (not comments). This avoids false positives where a comment
        # mentions the op name but the code doesn't actually call it.
        autograd_ops_seen = set()
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Attribute):
                continue
            # Match: torch.ops.autokernel.<name> — three-level Attribute chain.
            v = inner.value
            if not isinstance(v, ast.Attribute) or v.attr != "autokernel":
                continue
            v2 = v.value
            if not isinstance(v2, ast.Attribute) or v2.attr != "ops":
                continue
            v3 = v2.value
            if isinstance(v3, ast.Name) and v3.id == "torch":
                autograd_ops_seen.add(inner.attr)

        # Determine overall class verdict:
        # SAFE if no UNSAFE call-sites present. Absence of explicit
        #   autograd marker doesn't matter — plain PyTorch ops are all
        #   autograd-safe by default.
        # CONDITIONAL-SAFE if both SAFE-tagged and UNSAFE-tagged calls
        #   present AND an autograd op is referenced (runtime picks safe).
        # UNSAFE if any UNSAFE call and no autograd fallback path.
        verdicts = [c["verdict"] for c in calls_info]
        has_safe = any(v == "SAFE" for v in verdicts)
        has_unsafe = any(v == "UNSAFE" for v in verdicts)
        has_autograd_ref = bool(autograd_ops_seen)

        if not has_unsafe:
            # No raw-kernel calls at all — autograd flows naturally.
            overall = "SAFE"
        elif has_safe and has_unsafe and has_autograd_ref:
            overall = "CONDITIONAL-SAFE"
        elif has_unsafe and not has_autograd_ref:
            overall = "UNSAFE"
        else:
            overall = "UNSAFE"  # unsafe call without fallback proof

        class_infos.append({
            "class": name,
            "lineno": node.lineno,
            "overall_verdict": overall,
            "autograd_ops_referenced": sorted(autograd_ops_seen),
            "forward_calls": calls_info,
        })

    return class_infos


def analyze_torch_ops(path: Path):
    """Find every torch.library.custom_op and whether it has register_autograd."""
    if not path.exists():
        return {}
    src = path.read_text(encoding="utf-8")
    # Find blocks like:
    #   @torch.library.custom_op("autokernel::foo", ...)
    #   def foo_op(...): ...
    #   foo_op.register_autograd(...)
    ops = {}
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "torch.library.custom_op(" in line:
            # extract op name
            s = line.split('"', 1)[1].split('"', 1)[0]
            op = s.split("::")[1] if "::" in s else s
            ops[op] = {"defined_line": i + 1, "has_autograd": False}

    for op in list(ops.keys()):
        marker = f"{op}_op.register_autograd("
        marker2 = f"{op}.register_autograd("
        for j, line in enumerate(lines):
            if marker in line or marker2 in line:
                ops[op]["has_autograd"] = True
                ops[op]["autograd_line"] = j + 1
                break

    return ops


def render_markdown(class_infos, torch_ops, out_path: Path):
    lines = []
    lines.append("# Autokernel Static Audit (Phase A.1, 2026-05-10)")
    lines.append("")
    lines.append("AST-based scan of every `_*Replacement` class in "
                 "`autokernel/_patterns.py`. Tags each `forward()` call-site as "
                 "SAFE (goes through autograd) or UNSAFE (raw pybind → severs "
                 "gradient flow).\n")
    lines.append("Source: `autokernel/_patterns.py`; generator: `scripts/audit_autokernel_replacements.py`.\n")

    # Summary table
    lines.append("## Summary: replacement class verdicts\n")
    lines.append("| Class | Line | Overall | Custom ops referenced | Call count |")
    lines.append("|---|---:|:---:|---|---:|")
    for ci in sorted(class_infos, key=lambda c: c["lineno"]):
        ops = ", ".join(f"`{o}`" for o in ci["autograd_ops_referenced"]) or "_(none)_"
        lines.append(
            f"| `{ci['class']}` | {ci['lineno']} | "
            f"**{ci['overall_verdict']}** | {ops} | "
            f"{len(ci['forward_calls'])} |"
        )
    lines.append("")

    # Custom-op registration table
    lines.append("## Registered custom ops (`kernels/hip/_torch_ops.py`)\n")
    lines.append("| Op name | Defined | Autograd registered |")
    lines.append("|---|---:|:---:|")
    for name, info in sorted(torch_ops.items()):
        has = "yes" if info["has_autograd"] else "**NO**"
        lines.append(f"| `autokernel::{name}` | L{info['defined_line']} | {has} |")
    lines.append("")

    # Per-class details
    lines.append("## Per-class call-site breakdown\n")
    for ci in sorted(class_infos, key=lambda c: c["lineno"]):
        lines.append(f"### `{ci['class']}` (line {ci['lineno']}): **{ci['overall_verdict']}**\n")
        if not ci["forward_calls"]:
            lines.append("_(no classifiable calls in forward)_\n")
            continue
        lines.append("| Line | Verdict | Source |")
        lines.append("|---:|:---:|---|")
        for c in ci["forward_calls"]:
            lines.append(f"| {c['lineno']} | **{c['verdict']}** | `{c['src']}` |")
        lines.append("")

    # Conclusions
    unsafe = [c for c in class_infos if c["overall_verdict"] == "UNSAFE"]
    unknown = [c for c in class_infos if c["overall_verdict"] == "UNKNOWN"]
    lines.append("## Conclusions\n")
    lines.append(f"- **UNSAFE replacements (require Phase B fix):** {len(unsafe)}")
    for c in unsafe:
        lines.append(f"  - `{c['class']}`")
    lines.append(f"- **UNKNOWN replacements (require manual review):** {len(unknown)}")
    for c in unknown:
        lines.append(f"  - `{c['class']}`")
    lines.append(f"- **Total registered custom ops with autograd:** "
                 f"{sum(1 for o in torch_ops.values() if o['has_autograd'])} / {len(torch_ops)}")
    lines.append("")
    lines.append("_A UNSAFE replacement calls raw pybind (or equivalent) inside "
                 "forward(), which returns a tensor with `grad_fn=None`. Under "
                 "training mode, this severs gradient flow to upstream parameters. "
                 "See `docs/perf/autokernel-deep-analysis.md` for mechanism._\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--output",
                    default="docs/perf/autokernel-static-audit.md")
    args = ap.parse_args()

    root = Path(args.repo_root)
    patterns_path = root / "autokernel" / "_patterns.py"
    torch_ops_path = root / "kernels" / "hip" / "_torch_ops.py"

    class_infos = analyze_patterns(patterns_path)
    torch_ops = analyze_torch_ops(torch_ops_path)

    # Console summary
    print(f"Audited {len(class_infos)} Replacement classes in {patterns_path}")
    for ci in class_infos:
        print(f"  [{ci['overall_verdict']:>7s}]  {ci['class']:<45s}  "
              f"(line {ci['lineno']})")
    print()
    print(f"Registered custom ops: {len(torch_ops)} "
          f"(autograd: {sum(1 for o in torch_ops.values() if o['has_autograd'])})")
    for name, info in torch_ops.items():
        flag = "[Y]" if info["has_autograd"] else "[N]"
        print(f"  {flag} autokernel::{name}")

    out = render_markdown(class_infos, torch_ops, root / args.output)
    print(f"\nWrote {out}")

    # Also write JSON for machine-readable follow-up
    json_path = (root / args.output).with_suffix(".json")
    json_path.write_text(json.dumps({
        "class_infos": class_infos,
        "torch_ops": torch_ops,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
