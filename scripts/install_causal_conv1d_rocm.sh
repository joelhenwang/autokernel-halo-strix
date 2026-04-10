#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
SRC_DIR="${ROOT_DIR}/external/causal-conv1d"
REPO_URL="https://github.com/Dao-AILab/causal-conv1d.git"
REPO_TAG="v1.6.1"

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

for file in \
  "${SRC_DIR}/csrc/causal_conv1d_fwd.cu" \
  "${SRC_DIR}/csrc/causal_conv1d_bwd.cu" \
  "${SRC_DIR}/csrc/causal_conv1d_update.cu"
do
  sed -i '/^#include <hip\\/hip_runtime.h>$/d' "${file}"
  sed -i '/^#include <cmath>$/d' "${file}"
  sed -i '1i #include <cmath>' "${file}"
  sed -i '1i #include <hip/hip_runtime.h>' "${file}"
  sed -i 's/__expf(/__builtin_expf(/g' "${file}"
  sed -i 's/expf(/__builtin_expf(/g' "${file}"
  sed -i 's/exp(/__builtin_expf(/g' "${file}"
done

"${VENV_DIR}/bin/python" -m pip install --no-build-isolation --no-cache-dir -v "${SRC_DIR}"
