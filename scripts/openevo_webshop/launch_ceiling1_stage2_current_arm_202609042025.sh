#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
PATH_CONTRACT="${CEILING1_STAGE2_PATH_CONTRACT:?Stage2 path contract required}"
exec "$ROOT/scripts/oe" host \
  --path-contract "$PATH_CONTRACT" \
  --cwd "$ROOT" \
  -- "$ROOT/scripts/openevo_webshop/ceiling1_stage2_current_arm_driver_202609042025.sh"
