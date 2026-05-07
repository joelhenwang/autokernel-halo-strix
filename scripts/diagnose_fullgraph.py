"""Diagnose torch.compile graph breaks on OdinHalo forward.

Phase A tool for the OdinHalo Fullgraph Compile Sprint.
Runs OdinHalo forward under torch.compile(fullgraph=True, dynamic=False)
with verbose Dynamo logging. Parses the break report and categorizes
each break by (file:line, reason_code, user_stack).

Usage:
    python scripts/diagnose_fullgraph.py \\
        --model models/odin_halo.py --class-name OdinHalo \\
        --output docs/perf/fullgraph-diagnostic-2026-05-07.md

    # Or with a custom config:
    python scripts/diagnose_fullgraph.py \\
        --model models/odin_halo.py --class-name OdinHaloMini \\
        --batch-size 2 --block-size 64 \\
        --output /tmp/diag.md

The script produces a markdown report with:
1. Total break count and recommended path (S1/S2/S3).
2. Per-break: file:line, Dynamo reason code, summary.
3. Categorization: HIP kernel / list mutation / detach / other.
4. Detailed Dynamo log as an appendix.

Exit codes:
    0 — diagnostic ran successfully (breaks OK, report written)
    1 — torch.compile succeeded with zero breaks (unexpected; nothing to fix)
    2 — fatal error during model load or forward
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List, Tuple


# HIP kernel names we expect to see in the break report. Used for
# categorization of break sources.
HIP_KERNEL_NAMES = {
    "fused_rope_gate_mul",
    "causal_conv1d_fn",
    "kernel_fn",  # generic HIP kernel entry point
}

# Break categorization keywords — first match wins.
BREAK_CATEGORIES: List[Tuple[str, List[str]]] = [
    ("hip_kernel",         ["fused_rope_gate_mul", "causal_conv1d", "@torch.compiler.disable"]),
    ("list_mutation",      ["list.append", "append", "depth_kv_buffer", "List[", "dict"]),
    ("detach",             ["detach", "requires_grad"]),
    ("data_dependent",     ["data-dependent", "ID_OF", "SymBool"]),
    ("user_defined_class", ["user defined class", "UserDefinedObjectVariable"]),
    ("other",              []),
]


def load_model(model_path: str, class_name: str):
    """Import a model file and instantiate its class with default args."""
    path = Path(model_path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"{model_path} has no class {class_name}")
    return cls()


def categorize_break(text: str) -> str:
    """Return the first matching category for a break description."""
    text_lower = text.lower()
    for category, keywords in BREAK_CATEGORIES:
        if not keywords:
            continue
        for kw in keywords:
            if kw.lower() in text_lower:
                return category
    return "other"


def parse_dynamo_log(log_text: str) -> List[Dict[str, str]]:
    """Extract individual break records from verbose Dynamo output.

    Dynamo emits lines like:
      torch._dynamo.convert_frame: [WARNING] Graph break: <reason>
        File "foo.py", line 42, in <fn>
          actual_code()

    This parser is deliberately heuristic — Dynamo's log format has
    varied across PyTorch releases and we don't want to fail on a
    small format change. Returns a list of dicts with keys:
      'reason', 'file', 'line', 'category', 'raw'
    """
    breaks: List[Dict[str, str]] = []
    lines = log_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "graph break" in line.lower() or "could not be traced" in line.lower():
            reason = line.strip()
            file_ref = ""
            lineno = ""
            # Look ahead up to 5 lines for a 'File "...", line N' reference.
            for j in range(i + 1, min(i + 6, len(lines))):
                if 'File "' in lines[j] and ", line " in lines[j]:
                    # Extract file and line
                    try:
                        f_part = lines[j].split('File "', 1)[1]
                        file_ref = f_part.split('"', 1)[0]
                        lineno = f_part.split(", line ", 1)[1].split(",", 1)[0].strip()
                    except (IndexError, ValueError):
                        pass
                    break
            raw = "\n".join(lines[i:min(i + 6, len(lines))])
            breaks.append({
                "reason": reason,
                "file": file_ref,
                "line": lineno,
                "category": categorize_break(raw),
                "raw": raw,
            })
            i += 6
        else:
            i += 1
    return breaks


def recommend_path(breaks: List[Dict[str, str]]) -> str:
    """S1 / S2 / S3 recommendation based on break count + distribution."""
    n = len(breaks)
    categories = Counter(b["category"] for b in breaks)

    if n == 0:
        return "NONE (model already fullgraph-compatible — unexpected)"
    if n <= 3 and all(c in {"hip_kernel", "list_mutation"} for c in categories):
        return "S1 (minimal: allow_in_graph + static buffer)"
    if n <= 8 and categories["hip_kernel"] + categories["list_mutation"] >= n * 0.7:
        return "S2 (custom_op + manual unroll) — DEFAULT"
    if n >= 10:
        return "S3 (scan-based rewrite)"
    return "S2 (default — mixed distribution, custom_op is still the right call)"


def write_markdown_report(
    output_path: str,
    model_path: str,
    class_name: str,
    breaks: List[Dict[str, str]],
    recommendation: str,
    raw_log: str,
    error: str = None,
) -> None:
    categories = Counter(b["category"] for b in breaks)
    report = []
    report.append(f"# Fullgraph Diagnostic Report — {class_name}")
    report.append("")
    report.append(f"- **Model**: `{model_path}::{class_name}`")
    report.append(f"- **Total breaks**: {len(breaks)}")
    report.append(f"- **Recommended path**: {recommendation}")
    report.append(f"- **Category distribution**:")
    for cat, count in categories.most_common():
        report.append(f"  - `{cat}`: {count}")
    report.append("")

    if error:
        report.append("## Errors during diagnostic")
        report.append("")
        report.append("```")
        report.append(error)
        report.append("```")
        report.append("")

    report.append("## Break table")
    report.append("")
    if not breaks:
        report.append("_No breaks detected._")
    else:
        report.append("| # | Category | File | Line | Reason |")
        report.append("|--:|----------|------|-----:|--------|")
        for i, b in enumerate(breaks, 1):
            reason_short = b["reason"][:120].replace("|", "\\|")
            report.append(
                f"| {i} | {b['category']} | "
                f"`{b['file']}` | {b['line']} | {reason_short} |"
            )
    report.append("")

    report.append("## Categorization")
    report.append("")
    report.append("Categories follow `scripts/diagnose_fullgraph.py::BREAK_CATEGORIES`:")
    report.append("")
    for cat, kws in BREAK_CATEGORIES:
        if not kws:
            continue
        report.append(f"- **`{cat}`**: matched by keywords `{', '.join(kws)}`")
    report.append("")

    report.append("## Recommended next step")
    report.append("")
    if "S1" in recommendation:
        report.append(
            "Proceed with **S1 minimal**: `@torch.compiler.allow_in_graph` "
            "wrappers on HIP kernels + static `depth_kv_buffer`. Lower "
            "engineering cost, but Inductor loses some scheduling "
            "flexibility around the HIP calls."
        )
    elif "S2" in recommendation:
        report.append(
            "Proceed with **S2 (default)**: `torch.library.custom_op` + "
            "`register_fake` wrappers on HIP kernels + static "
            "`depth_kv_buffer` + manual loop unroll. Spec: "
            "`docs/superpowers/specs/2026-05-07-odinhalo-fullgraph-compile-design.md`."
        )
    elif "S3" in recommendation:
        report.append(
            "Many breaks scattered throughout — consider **S3 scan-based** "
            "rewrite via `torch.ops.higher_order.scan`. Larger blast "
            "radius but sidesteps the break-by-break patching."
        )
    elif "NONE" in recommendation:
        report.append(
            "_No breaks detected._ Re-run with `fullgraph=True` and check "
            "the compiled forward passes execution. If so, whole-model "
            "compile should already work on this model."
        )
    report.append("")

    report.append("## Appendix — raw Dynamo log")
    report.append("")
    report.append("```text")
    # Truncate very long logs to first 500 lines; full log in diagnostic dir
    truncated = raw_log.splitlines()[:500]
    report.append("\n".join(truncated))
    if len(raw_log.splitlines()) > 500:
        report.append(f"\n... (truncated; {len(raw_log.splitlines())} total lines)")
    report.append("```")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(report), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to model file (e.g., models/odin_halo.py)")
    ap.add_argument("--class-name", required=True,
                    help="Model class to instantiate (e.g., OdinHalo)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--output", required=True,
                    help="Markdown report output path")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import torch._dynamo
    import torch._dynamo.config

    # Verbose mode captures all graph-break reasons with stacks.
    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = False
    # Don't cache — we want a clean trace per run
    torch._dynamo.reset()

    print(f"[diagnose] Loading {args.model}::{args.class_name}")
    try:
        model = load_model(args.model, args.class_name)
        model = model.to(args.device).eval()
    except Exception as exc:
        traceback.print_exc()
        write_markdown_report(
            args.output, args.model, args.class_name,
            breaks=[], recommendation="ERROR — model load failed",
            raw_log="", error=traceback.format_exc(),
        )
        return 2

    print(f"[diagnose] Compiling with fullgraph=True, dynamic=False")
    try:
        compiled = torch.compile(model, fullgraph=True, dynamic=False)
    except Exception:
        traceback.print_exc()
        return 2

    # Capture stderr during the forward — Dynamo writes break reports there.
    stderr_buf = io.StringIO()
    stdout_buf = io.StringIO()

    input_ids = torch.randint(
        0, args.vocab_size, (args.batch_size, args.block_size),
        device=args.device, dtype=torch.long,
    )

    error_text = None
    print(f"[diagnose] Running forward (batch={args.batch_size}, block={args.block_size})")
    try:
        with redirect_stderr(stderr_buf), redirect_stdout(stdout_buf):
            with torch.no_grad():
                _ = compiled(input_ids)
        print("[diagnose] Forward completed with fullgraph=True — zero breaks!")
    except torch._dynamo.exc.Unsupported as exc:
        # Graph break under fullgraph raises Unsupported. Expected.
        error_text = str(exc)
        print(f"[diagnose] Graph break detected (expected): {error_text[:200]}")
    except Exception as exc:
        # Other failures — likely a model config issue. Still report.
        error_text = traceback.format_exc()
        print(f"[diagnose] Forward failed (non-Unsupported): {exc}")

    log_text = stderr_buf.getvalue() + "\n" + stdout_buf.getvalue()
    if error_text:
        log_text = f"{error_text}\n\n{log_text}"

    breaks = parse_dynamo_log(log_text)
    recommendation = recommend_path(breaks)

    print(f"[diagnose] Found {len(breaks)} breaks; recommendation: {recommendation}")

    write_markdown_report(
        args.output, args.model, args.class_name,
        breaks=breaks, recommendation=recommendation,
        raw_log=log_text, error=error_text,
    )
    print(f"[diagnose] Report written to {args.output}")

    if len(breaks) == 0 and not error_text:
        return 1  # unexpected clean pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
