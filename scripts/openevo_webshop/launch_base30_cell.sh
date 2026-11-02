#!/usr/bin/env bash
# Stable entrypoint for one BASE-30 WebShop cell.
set -euo pipefail

JAVA_HOME="${JAVA_HOME:-/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1}"
OPENEVO_PYTHON="${OPENEVO_PYTHON:-/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python}"
export JAVA_HOME
export PATH="${JAVA_HOME}/bin:${PATH}"

test -x "${JAVA_HOME}/bin/javac" || {
  echo "JDK contract failed: ${JAVA_HOME}/bin/javac is missing" >&2
  exit 2
}
test -x "${OPENEVO_PYTHON}" || {
  echo "Python contract failed: ${OPENEVO_PYTHON} is missing" >&2
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${OPENEVO_PYTHON}" "${SCRIPT_DIR}/run_base30_cell.py" "$@"
