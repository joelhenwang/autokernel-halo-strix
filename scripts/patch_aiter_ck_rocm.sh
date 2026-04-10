#!/usr/bin/env bash
set -euo pipefail

# Patch aiter's CK headers and csrc for ROCm 7.12 gfx1151.
#
# ROCm 7.12's HIP compiler rejects bare C math functions (exp2f, expf, powf,
# expm1f, __logf, sincosf, log1pf, log2f) in device code. This script replaces
# them with __builtin_ equivalents across all aiter source files.
#
# Usage:
#   bash scripts/patch_aiter_ck_rocm.sh [aiter_dir]
#   Default aiter_dir: ./aiter

AITER_DIR="${1:-aiter}"

if [ ! -d "${AITER_DIR}/3rdparty/composable_kernel" ]; then
  echo "ERROR: ${AITER_DIR}/3rdparty/composable_kernel not found" >&2
  exit 1
fi

echo "Patching aiter CK headers + csrc in ${AITER_DIR}..."

python3 - "${AITER_DIR}" <<'PATCH_SCRIPT'
import re
import sys
from pathlib import Path

aiter_dir = Path(sys.argv[1])

scan_dirs = [
    aiter_dir / "3rdparty" / "composable_kernel" / "include",
    aiter_dir / "csrc",
]

extensions = {".hpp", ".h", ".cuh", ".cu", ".cpp"}

# Each entry: (regex pattern, replacement)
# Negative lookbehind ensures we don't patch already-patched calls
# (?<!builtin_) and (?<!amdgcn_) prevent matching __builtin_exp2f or __builtin_amdgcn_exp2f
patterns = [
    # std:: prefixed — replace std::func with __builtin_func
    (re.compile(r'\bstd::exp2f\('),   '__builtin_exp2f('),
    (re.compile(r'\bstd::expf\('),    '__builtin_expf('),
    (re.compile(r'\bstd::powf\('),    '__builtin_powf('),
    (re.compile(r'\bstd::expm1f\('),  '__builtin_expm1f('),
    (re.compile(r'\bstd::log2f\('),   '__builtin_log2f('),
    (re.compile(r'\bstd::log1pf\('),  '__builtin_log1pf('),
    # Bare function names — only match if NOT preceded by builtin_ or amdgcn_
    (re.compile(r'(?<!builtin_)(?<!amdgcn_)\bexp2f\('),   '__builtin_exp2f('),
    (re.compile(r'(?<!builtin_)(?<!amdgcn_)\bexpf\('),    '__builtin_expf('),
    (re.compile(r'(?<!builtin_)(?<!amdgcn_)\b__logf\('),  '__builtin_logf('),
    (re.compile(r'(?<!builtin_)(?<!amdgcn_)\bpowf\('),    '__builtin_powf('),
    (re.compile(r'(?<!builtin_)(?<!amdgcn_)\bexpm1f\('),  '__builtin_expm1f('),
    (re.compile(r'(?<!builtin_)(?<!amdgcn_)\bsincosf\('), '__builtin_sincosf('),
    (re.compile(r'(?<!builtin_)(?<!amdgcn_)\blog1pf\('),  '__builtin_log1pf('),
    (re.compile(r'(?<!builtin_)(?<!amdgcn_)\blog2f\('),   '__builtin_log2f('),
]

patched_files = 0

for scan_dir in scan_dirs:
    if not scan_dir.exists():
        print(f"  SKIP: {scan_dir} not found")
        continue

    for path in sorted(scan_dir.rglob("*")):
        if path.suffix not in extensions:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        original = text
        for pat, repl in patterns:
            text = pat.sub(repl, text)

        if text != original:
            path.write_text(text, encoding="utf-8")
            rel = path.relative_to(aiter_dir)
            print(f"  Patched: {rel}")
            patched_files += 1

print(f"\nDone: {patched_files} files patched.")
PATCH_SCRIPT

# Clear cached failed JIT build
if [ -d "${AITER_DIR}/aiter/jit/build/module_aiter_core" ]; then
  rm -rf "${AITER_DIR}/aiter/jit/build/module_aiter_core"
  echo "Cleared JIT build cache."
fi

echo "aiter patched for ROCm 7.12 gfx1151."
