"""One-off: print per-domain BPB from a directory of Sprint 1.1 Phase B scorecards."""
import glob, json, os, sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "docs/perf/eval-scorecards/sprint1.1-B*.json"

rows = []
for f in sorted(glob.glob(pattern)):
    name = os.path.basename(f).replace(".json", "")
    d = json.load(open(f))
    pd = d.get("per_domain_bpb", {})
    # per_domain_bpb values are directly at pd["wikitext_val"] (floats), not nested.
    # Compute average across the 4 public domain keys (skip leading-underscore meta).
    public = {k: v for k, v in pd.items() if not k.startswith("_") and isinstance(v, (int, float))}
    avg = sum(public.values()) / len(public) if public else None
    row = {
        "name": name,
        "wiki_val": pd.get("wikitext_val"),
        "gpt_small": pd.get("gpt_small_val"),
        "stem_crawl": pd.get("stem_crawl_val"),
        "dolma": pd.get("dolma_val"),
        "avg": avg,
    }
    rows.append(row)

if not rows:
    print(f"No scorecards found matching {pattern}")
    sys.exit(1)

hdr = f"{'name':<36s} {'wiki':>7s} {'gpt-sm':>7s} {'stem':>7s} {'dolma':>7s} {'avg':>7s}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    def fmt(x):
        return f"{x:7.4f}" if x is not None else "    -  "
    print(f"{r['name']:<36s} {fmt(r['wiki_val'])} {fmt(r['gpt_small'])} "
          f"{fmt(r['stem_crawl'])} {fmt(r['dolma'])} {fmt(r['avg'])}")
