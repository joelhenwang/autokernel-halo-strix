"""WI6: Parse Inductor output log + cache to build fusion catalog.

Reads the concatenated inductor cache (`<out>.cache.py`) produced by
scripts/dump_inductor_output.py and extracts one row per triton kernel:
  - kernel name
  - number of fused ops (derived from the name: triton_poi_fused_<op1>_<op2>_... )
  - fused op list
  - heuristic category (pointwise/reduction/persistent)
  - number of buffer inputs/outputs (approx)
  - source file (FX graph ID)

Also extracts FX graph summaries: for each compiled graph, list op names
found in call_function / call_module markers to tie kernels back to model code.

Output: markdown table + top-level summary.

Usage:
  python scripts/parse_inductor_cache.py \
      --cache docs/perf/inductor-triton-dump.cache.py \
      --out docs/perf/inductor-fusion-catalog.md
"""
import argparse, pathlib, re, collections, json


KERNEL_RE = re.compile(r"^(?:@\w+\s+)*def\s+(triton_\w+)\s*\(", re.M)
KERNEL_DECL_RE = re.compile(
    r"(triton_(?P<kind>poi|per|red|tem)_fused_(?P<ops>[\w]+?)_(?P<n>\d+))\b"
)
# top of a kernel source block: capture args & grid from the wrapper just above
# but easier: we grep the whole file for `async_compile.triton('<name>', '''` blocks.
ASYNC_TRITON_RE = re.compile(
    r"async_compile\.triton\(\s*['\"](?P<name>triton_\w+)['\"]\s*,\s*'''(?P<body>.*?)'''",
    re.S,
)
FILE_HEADER_RE = re.compile(r"################ FILE: (?P<rel>.*?) ################")


def parse_kernel_name(name: str):
    """Extract (kind, op_list, seq) from a triton kernel name.

    triton_poi_fused__to_copy_mul_transpose_view_8 ->
      kind='poi', ops=['_to_copy', 'mul', 'transpose', 'view'], seq=8
    """
    m = re.match(r"triton_(?P<kind>poi|per|red|tem)_fused_(?P<ops>.+)_(?P<n>\d+)$", name)
    if not m:
        return None, [], None
    kind = m.group("kind")
    ops_raw = m.group("ops")
    seq = int(m.group("n"))
    # op names are _-separated. Some ops have leading underscores (_to_copy, _unsafe_view).
    # We split on _ but reattach leading underscores to the NEXT token if it follows a split pt.
    # Heuristic: split on '_' then merge any empty chunk with the following token.
    raw = ops_raw.split("_")
    ops = []
    pending_prefix = ""
    for tok in raw:
        if tok == "":
            pending_prefix += "_"
        else:
            ops.append(pending_prefix + tok)
            pending_prefix = ""
    return kind, ops, seq


KIND_LABEL = {
    "poi": "pointwise",
    "per": "persistent-reduction",
    "red": "reduction",
    "tem": "template (matmul-fused)",
}


def scan_kernel_body(body: str):
    """Extract a few stats from a kernel body:
      - num loads (tl.load)
      - num stores (tl.store)
      - num tl.dot (if any => matmul-like)
      - grid/block hints
    """
    num_load = len(re.findall(r"\btl\.load\b", body))
    num_store = len(re.findall(r"\btl\.store\b", body))
    num_dot = len(re.findall(r"\btl\.dot\b", body))
    num_exp = len(re.findall(r"\btl\.exp|\blibdevice\.exp", body))
    num_sqrt = len(re.findall(r"\btl\.sqrt|\blibdevice\.rsqrt", body))
    return {
        "loads": num_load,
        "stores": num_store,
        "dots": num_dot,
        "exps": num_exp,
        "sqrts": num_sqrt,
        "body_bytes": len(body),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str,
                        default="docs/perf/inductor-triton-dump.cache.py")
    parser.add_argument("--out", type=str,
                        default="docs/perf/inductor-fusion-catalog.md")
    args = parser.parse_args()

    cache_path = pathlib.Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: cache file not found: {cache_path}")
        return 2

    text = cache_path.read_text(encoding="utf-8", errors="replace")
    # Segment by file header
    segments = []
    cur_name = "<preamble>"
    cur_start = 0
    for m in FILE_HEADER_RE.finditer(text):
        segments.append((cur_name, text[cur_start:m.start()]))
        cur_name = m.group("rel")
        cur_start = m.end()
    segments.append((cur_name, text[cur_start:]))

    # For each segment, find async_compile.triton blocks
    kernels = []  # list of dicts
    for seg_name, seg in segments:
        for km in ASYNC_TRITON_RE.finditer(seg):
            name = km.group("name")
            body = km.group("body")
            kind, ops, seq = parse_kernel_name(name)
            stats = scan_kernel_body(body)
            kernels.append({
                "name": name,
                "file": seg_name,
                "kind": kind or "unknown",
                "kind_label": KIND_LABEL.get(kind, "unknown"),
                "ops": ops,
                "n_ops": len(ops),
                "seq": seq,
                **stats,
            })

    # Deduplicate by name (Inductor can emit same kernel name across files if cache-hit)
    by_name = {}
    for k in kernels:
        if k["name"] not in by_name:
            by_name[k["name"]] = k
            by_name[k["name"]]["instances"] = 1
        else:
            by_name[k["name"]]["instances"] += 1
    unique = sorted(by_name.values(), key=lambda k: (-k["n_ops"], k["name"]))

    # Build markdown
    lines = []
    lines.append("# Inductor Fusion Catalog (WI6 — Phase 2)\n")
    lines.append(f"Source: `{cache_path}` ({cache_path.stat().st_size:,} bytes)\n")
    lines.append(f"**Total unique triton kernels:** {len(unique)}")
    lines.append(f"**Total kernel declarations (including dupes):** {len(kernels)}\n")

    # Summary by kind
    by_kind = collections.Counter(k["kind_label"] for k in unique)
    lines.append("## Kernels by kind\n")
    lines.append("| Kind | Count |")
    lines.append("|------|------:|")
    for kind, cnt in by_kind.most_common():
        lines.append(f"| {kind} | {cnt} |")

    # Histogram of fusion size (n_ops)
    sizes = collections.Counter(k["n_ops"] for k in unique)
    lines.append("\n## Fusion size distribution (# of ops fused per kernel)\n")
    lines.append("| # ops fused | # kernels |")
    lines.append("|-----------:|----------:|")
    for n in sorted(sizes.keys()):
        lines.append(f"| {n} | {sizes[n]} |")

    # Top-fused kernels (biggest)
    lines.append("\n## Top 30 most-fused kernels (by op count)\n")
    lines.append("| # | Kernel | Kind | Fused ops | Loads | Stores | Exps | Sqrts | Body bytes | File |")
    lines.append("|--:|--------|------|----------|------:|-------:|-----:|------:|-----------:|------|")
    for i, k in enumerate(unique[:30], 1):
        ops_str = ", ".join(k["ops"])
        file_short = pathlib.Path(k["file"]).name[:40]
        lines.append(
            f"| {i} | `{k['name']}` | {k['kind_label']} | `{ops_str}` | "
            f"{k['loads']} | {k['stores']} | {k['exps']} | {k['sqrts']} | "
            f"{k['body_bytes']:,} | `{file_short}` |"
        )

    # Catalog of ALL kernels sorted by name
    lines.append("\n## Full kernel catalog (sorted by name)\n")
    lines.append("| Kernel | Kind | # ops | Fused ops | Loads | Stores |")
    lines.append("|--------|------|------:|----------|------:|-------:|")
    for k in sorted(unique, key=lambda x: x["name"]):
        ops_str = ", ".join(k["ops"])
        lines.append(
            f"| `{k['name']}` | {k['kind_label']} | {k['n_ops']} | `{ops_str}` | "
            f"{k['loads']} | {k['stores']} |"
        )

    # Per-op frequency: which ops appear in fusions most often?
    op_freq = collections.Counter()
    for k in unique:
        for op in k["ops"]:
            op_freq[op] += 1
    lines.append("\n## Most frequently fused ops (per-op frequency across all kernels)\n")
    lines.append("| Op | # kernels fusing it |")
    lines.append("|----|-------------------:|")
    for op, cnt in op_freq.most_common(40):
        lines.append(f"| `{op}` | {cnt} |")

    # Analysis notes section
    lines.append("\n## Analysis notes\n")
    lines.append("**How to read this catalog:**")
    lines.append("- Each `triton_<kind>_fused_<op1>_<op2>_..._<seq>` kernel represents one")
    lines.append("  Inductor-generated fused GPU kernel. The op list is the set of ATen/prims")
    lines.append("  ops Inductor determined can run as a single kernel — usually a chain of")
    lines.append("  pointwise ops, or a pointwise chain terminating in a reduction.")
    lines.append("- `kind=poi` = pointwise only. `per` = persistent reduction (small inner dim).")
    lines.append("  `red` = reduction with loop. `tem` = matmul template with epilogue fused.\n")
    lines.append("**Phase 2 implication:** every op listed here is ALREADY FUSED by Inductor.")
    lines.append("Re-implementing it as a custom HIP kernel buys nothing unless the HIP kernel")
    lines.append("beats Inductor's triton output on this specific shape/layout. Do not re-fuse")
    lines.append("patterns that appear in this catalog without isolated benchmark proof.\n")
    lines.append("**Kernels to investigate (non-trivial fusions worth understanding):**")
    for i, k in enumerate(unique[:10], 1):
        lines.append(f"{i}. `{k['name']}` — fuses {k['n_ops']} ops: `{', '.join(k['ops'])}`")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"Unique triton kernels: {len(unique)}")
    print(f"Most-fused top 5:")
    for k in unique[:5]:
        print(f"  {k['name']} ({k['n_ops']} ops: {', '.join(k['ops'])})")

    # Also dump JSON for programmatic access
    json_out = out_path.with_suffix(".json")
    json_out.write_text(json.dumps(unique, indent=2), encoding="utf-8")
    print(f"JSON: {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
