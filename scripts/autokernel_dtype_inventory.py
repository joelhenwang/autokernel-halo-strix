#!/usr/bin/env python
"""v3 T-0.7 dtype/autocast inventory across all training-path custom ops.

Per docs/research/autokernel-40k-v3-execution-plan.md section 5.2, emit
JSONL records describing:
  - whether each custom_op has register_autograd / register_fake / register_autocast
  - forward input/output dtypes under fp16 autocast
  - backward input/output dtypes
  - whether the op is a graph-break source (from T-0.4 inventory)
  - which --ak-* flag(s) enable the op

This is the Tier 0 cheap inventory (no parity testing, no backward exec).
Tier 1 (fix graph-break ops) + Tier 2 (deep parity) are separate scripts.

Usage:
    python scripts/autokernel_dtype_inventory.py \
        --output docs/perf/dtype-autocast-inventory.md \
        --jsonl docs/perf/dtype-autocast-inventory.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Dict, Any

import torch

# Ensure repo root on sys.path regardless of invocation cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --------------------------------------------------------------------------
# Op catalog: training-path custom ops on OdinFlat/OdinHalo.
# From T-0.4 findings (docs/perf/graph-breaks-inventory.md):
#   - 2 graph-break-causing ops: fused_rope_gate_mul + causal_conv1d
#   - 5 Phase-B clean ops: silu_gate_mul, rmsnorm, fused_res_rmsnorm,
#     rotary_emb_fp32, fused_ple_gate
# --------------------------------------------------------------------------

_TRAINING_PATH_OPS = [
    {
        "op_name": "autokernel::silu_gate_mul",
        "graph_break_source": False,
        "enabled_by_flags": ["--ak-swiglu-fwd", "--ak-swiglu-bwd"],
        "module": "kernels.hip._torch_ops",
        "attr": "silu_gate_mul_op",
        "phase_b_fixed": True,
    },
    {
        "op_name": "autokernel::rmsnorm",
        "graph_break_source": False,
        "enabled_by_flags": ["--ak-rmsnorm"],
        "module": "kernels.hip._torch_ops",
        "attr": "rmsnorm_op",
        "phase_b_fixed": True,
    },
    {
        "op_name": "autokernel::fused_res_rmsnorm",
        "graph_break_source": False,
        "enabled_by_flags": ["--ak-res-rmsnorm"],
        "module": "kernels.hip._torch_ops",
        "attr": "fused_res_rmsnorm_op",
        "phase_b_fixed": True,
    },
    {
        "op_name": "autokernel::rotary_emb_fp32",
        "graph_break_source": False,
        "enabled_by_flags": ["--ak-rope"],
        "module": "kernels.hip._torch_ops",
        "attr": "rotary_emb_fp32_op",
        "phase_b_fixed": True,
    },
    {
        "op_name": "autokernel::fused_ple_gate",
        "graph_break_source": False,
        "enabled_by_flags": ["--ak-ple-gate"],
        "module": "kernels.hip._torch_ops",
        "attr": "fused_ple_gate_op",
        "phase_b_fixed": True,
    },
    {
        "op_name": "kernels.hip.fused_rope_gate_mul.kernel_fn",
        "graph_break_source": True,
        "enabled_by_flags": ["--ak-rope-gate", "--ak-fix-rope-gate-op"],
        "module": "kernels.hip.fused_rope_gate_mul",
        "attr": "kernel_fn",
        "phase_b_fixed": False,
        "graph_break_cause": "@torch.compiler.disable on wrapper",
    },
    {
        "op_name": "DaoAILab::causal_conv1d_fn",
        "graph_break_source": True,
        "enabled_by_flags": ["--ak-causal-conv", "--ak-causal-conv-shim"],
        "module": "kernels.hip.causal_conv1d",  # check module presence only
        "attr": None,
        "phase_b_fixed": False,
        "graph_break_cause": "older custom_op semantics (external extension)",
    },
]


def _introspect_op(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return an inventory record for one op spec."""
    record: Dict[str, Any] = {
        "op_name": spec["op_name"],
        "graph_break_source": spec["graph_break_source"],
        "enabled_by_flags": spec["enabled_by_flags"],
        "phase_b_fixed": spec["phase_b_fixed"],
        "graph_break_cause": spec.get("graph_break_cause"),
    }

    # Try to import the module and introspect the op
    try:
        import importlib
        mod = importlib.import_module(spec["module"])
    except Exception as e:
        record["module_importable"] = False
        record["import_error"] = str(e)
        # Still emit the record with whatever static info we have
        record["has_register_fake"] = None
        record["has_register_autograd"] = None
        record["has_register_autocast"] = None
        return record

    record["module_importable"] = True

    if spec["attr"] is None:
        record["has_register_fake"] = None
        record["has_register_autograd"] = None
        record["has_register_autocast"] = None
        return record

    op = getattr(mod, spec["attr"], None)
    if op is None:
        record["attr_present"] = False
        return record
    record["attr_present"] = True

    # For torch.library.custom_op objects, introspect registrations.
    # Heuristic: scan source for register_autocast/register_autograd/register_fake
    # CALLS (not just method presence) in the module text. This catches whether
    # a rule was actually registered vs just the API being available.
    import inspect
    try:
        src = inspect.getsource(mod)
    except Exception:
        src = ""

    # The op-specific decorators look like:
    #   @rmsnorm_op.register_fake
    #   rmsnorm_op.register_autograd(...)
    #   rmsnorm_op.register_autocast(...)
    attr = spec["attr"]
    record["has_register_fake"] = (
        f"@{attr}.register_fake" in src
    )
    record["has_register_autograd"] = (
        f"{attr}.register_autograd" in src
    )
    record["has_register_autocast"] = (
        f"{attr}.register_autocast" in src
    )

    # Note: actual registration semantics is hard to detect without running
    # the op. For now, we rely on source-text heuristics above. Tier 2 deep
    # parity will exercise the ops with actual tensors.

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True,
                        help="Markdown output path")
    parser.add_argument("--jsonl", required=True,
                        help="JSONL output path (per v3 section 9.2 schema)")
    args = parser.parse_args()

    # Import the ops module so custom_ops are registered
    try:
        import kernels.hip._torch_ops  # noqa: F401
    except Exception as e:
        print(f"[inventory] WARNING: failed to import kernels.hip._torch_ops: {e}")

    records: List[Dict[str, Any]] = []
    for spec in _TRAINING_PATH_OPS:
        rec = _introspect_op(spec)
        # v3 §9.2 schema fields we can't fill without running ops set to unknown.
        rec.setdefault("forward_input_dtypes", None)
        rec.setdefault("forward_output_dtype", None)
        rec.setdefault("backward_input_dtype", None)
        rec.setdefault("backward_output_dtypes", None)
        rec.setdefault("internal_accumulation", "unknown")
        records.append(rec)

    # Emit JSONL
    os.makedirs(os.path.dirname(args.jsonl) or ".", exist_ok=True)
    with open(args.jsonl, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    # Emit markdown summary
    lines = ["# v3 T-0.7 Dtype/Autocast Inventory\n"]
    lines.append("**Source:** `scripts/autokernel_dtype_inventory.py` (Tier 0 cheap inventory).\n")
    lines.append(
        "**Schema:** v3 §9.2 (docs/research/autokernel-40k-v3-execution-plan.md).\n"
    )
    lines.append(
        "\nThis is a static inventory of registration metadata. Tier 1 fixes graph-break ops; "
        "Tier 2 runs deep parity with actual tensors.\n"
    )
    lines.append("\n## Headline\n")
    n_phase_b = sum(1 for r in records if r.get("phase_b_fixed"))
    n_graph_break = sum(1 for r in records if r.get("graph_break_source"))
    n_autocast = sum(1 for r in records if r.get("has_register_autocast"))
    lines.append(f"- {n_phase_b}/{len(records)} ops are Phase-B-fixed (autograd-safe).")
    lines.append(f"- {n_graph_break}/{len(records)} ops cause graph breaks (T-3.2 targets).")
    lines.append(f"- {n_autocast}/{len(records)} ops have `register_autocast` rules registered.\n")

    lines.append("\n## Per-op table\n")
    lines.append("| op | phase_b_fixed | graph_break | has_autograd | has_fake | has_autocast | flags |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in records:
        lines.append(
            f"| `{r['op_name']}` | {r.get('phase_b_fixed')} | "
            f"{r.get('graph_break_source')} | "
            f"{r.get('has_register_autograd')} | "
            f"{r.get('has_register_fake')} | "
            f"{r.get('has_register_autocast')} | "
            f"{','.join(r.get('enabled_by_flags', []))} |"
        )

    lines.append("\n## Graph-break details\n")
    for r in records:
        if r.get("graph_break_source"):
            lines.append(
                f"- **`{r['op_name']}`**: {r.get('graph_break_cause', 'unknown')}. "
                f"Fix target: T-3.2."
            )
    lines.append("\n## Tier 2 targets\n")
    lines.append(
        "Deep parity (forward/backward rel_err, grad cosine, post-NorMuon update cosine) "
        "required for ops in the training path. From v3 §4.2 Tier 2 list:\n"
    )
    tier2 = [
        "autokernel::silu_gate_mul",
        "autokernel::rmsnorm",
        "autokernel::fused_res_rmsnorm",
        "kernels.hip.fused_rope_gate_mul.kernel_fn",
        "DaoAILab::causal_conv1d_fn",
    ]
    for t in tier2:
        lines.append(f"- `{t}`")
    lines.append("")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write("\n".join(lines))

    print(f"[inventory] wrote {args.jsonl} ({len(records)} records)")
    print(f"[inventory] wrote {args.output}")
    print(f"[inventory] summary: {n_phase_b} Phase-B-fixed, "
          f"{n_graph_break} graph-break, {n_autocast} with autocast")


if __name__ == "__main__":
    main()
