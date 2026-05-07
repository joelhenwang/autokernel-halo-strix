"""Analyze sprint3-smoke-dolma activation stats progression."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sprint3-smoke-dolma/activation_stats.jsonl"
stats = [json.loads(l) for l in open(path) if l.strip()]

by_step = {}
for s in stats:
    by_step.setdefault(s["step"], {})[s["layer"]] = s["maxabs"]

print(f"{'step':>6} {'max_layer':<30} {'maxabs':>10} {'headroom':>9} {'nan_layers'}")
for step in sorted(by_step.keys()):
    layers = by_step[step]
    # separate NaN from valid
    values = [(l, v) for l, v in layers.items() if isinstance(v, (int, float)) and not (v != v)]
    nans = [l for l, v in layers.items() if isinstance(v, float) and v != v]
    if values:
        max_layer, max_v = max(values, key=lambda x: x[1])
        headroom = 65504.0 / max_v if max_v > 0 else float("inf")
        print(f"{step:>6} {max_layer:<30} {max_v:10.3f} {headroom:9.1f} {len(nans)}")
    else:
        print(f"{step:>6} (all NaN)")
