#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
SRC_DIR="${ROOT_DIR}/external/mamba"
REPO_URL="https://github.com/state-spaces/mamba.git"
REPO_TAG="v2.3.0"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing virtualenv at ${VENV_DIR}" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/external"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
  git clone --branch "${REPO_TAG}" --depth 1 "${REPO_URL}" "${SRC_DIR}"
else
  git -C "${SRC_DIR}" fetch --tags --depth 1 origin "${REPO_TAG}"
  git -C "${SRC_DIR}" checkout "${REPO_TAG}"
fi

COMMON_FILE="${SRC_DIR}/csrc/selective_scan/selective_scan_common.h"
FWD_FILE="${SRC_DIR}/csrc/selective_scan/selective_scan_fwd_kernel.cuh"
BWD_FILE="${SRC_DIR}/csrc/selective_scan/selective_scan_bwd_kernel.cuh"
REV_SCAN_FILE="${SRC_DIR}/csrc/selective_scan/reverse_scan.cuh"

sed -i '/^#include <cmath>$/d' "${COMMON_FILE}"
sed -i '/^#include <cuda_fp16.h>$/a #include <cmath>' "${COMMON_FILE}"

"${VENV_DIR}/bin/python" - <<'PY' "${COMMON_FILE}" "${FWD_FILE}" "${BWD_FILE}"
from pathlib import Path
import sys

replacements = {
    "exp2f(": "__builtin_exp2f(",
    "sincosf(": "__builtin_sincosf(",
    "expf(": "__builtin_expf(",
}

for name in sys.argv[1:]:
    path = Path(name)
    text = path.read_text()
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.replace("__builtin_log1pf(__builtin_expf(", "__builtin_logf(1.0f + __builtin_expf(")
    text = text.replace("log1pf(__builtin_expf(", "__builtin_logf(1.0f + __builtin_expf(")
    path.write_text(text)
PY

sed -i 's/#define WARP_THREADS HIPCUB_WARP_THREADS/#define WARP_THREADS 32/' "${REV_SCAN_FILE}"

(
  cd "${SRC_DIR}"
  MAMBA_FORCE_BUILD=TRUE "${VENV_DIR}/bin/python" -m pip install --no-build-isolation --no-cache-dir -v .
)
