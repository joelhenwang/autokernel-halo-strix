#!/usr/bin/env bash
# profile_step_rocprof.sh — run profile_step_deep.py under rocprof to capture HIP kernel timing.
# Emits docs/perf/odinhalo-profile-<date>/rocprof_stats.csv and rocprof.md.

set -e

if ! command -v rocprof &> /dev/null; then
    echo "rocprof not installed; skipping kernel-level profile"
    echo "(You can still use the torch.profiler output from profile_step_deep.py)"
    exit 0
fi

DATE=$(date +%Y-%m-%d)
OUTDIR="docs/perf/odinhalo-profile-${DATE}"
mkdir -p "$OUTDIR"

echo "Running rocprof on profile_step_deep.py --rocprof-subrun..."
cd "$OUTDIR"
rocprof --stats --hip-trace -o rocprof_trace python ../../../scripts/profile_step_deep.py --rocprof-subrun
cd - > /dev/null

# rocprof outputs rocprof_trace.stats.csv and rocprof_trace.csv
if [ -f "$OUTDIR/rocprof_trace.stats.csv" ]; then
    echo "Generating rocprof.md summary..."
    python scripts/summarize_rocprof.py --input "$OUTDIR/rocprof_trace.stats.csv" --output "$OUTDIR/rocprof.md"
    echo "Done. See $OUTDIR/rocprof.md"
else
    echo "WARNING: rocprof_trace.stats.csv not found in $OUTDIR; rocprof may have changed output format"
    ls -la "$OUTDIR"
fi
