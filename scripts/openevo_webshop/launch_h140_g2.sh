#!/usr/bin/env bash
set -euo pipefail

# This is a future launch boundary, not an execution shortcut. Preparation is
# CPU-only; formal stages remain unavailable until the versioned router is
# explicitly transitioned and the parent live-resource gate passes.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-prepare}"

die() { echo "H1.40-G2 blocked: $*" >&2; exit 2; }

case "${MODE}" in
  prepare)
    python3 "${ROOT}/scripts/openevo_webshop/validate_h140_g2_preregistration.py" --root "${ROOT}"
    python3 "${ROOT}/scripts/openevo_webshop/plan_h140_g2.py" --root "${ROOT}"
    ;;
  execute)
    campaign="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaign_id"])' "${ROOT}/configs/experiment/current-campaign.json")"
    status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${ROOT}/configs/experiment/current-campaign.json")"
    formal="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["formal_task_consumption_allowed"])' "${ROOT}/configs/experiment/current-campaign.json")"
    [[ "${campaign}" == "20260820-0140-h140-g2-multigeneration" ]] || die "campaign router is not H1.40-G2"
    [[ "${status}" == "launch-authorized" && "${formal}" == "True" ]] || die "router is still closed; no formal task or GPU work may start"
    die "formal execution wiring requires a separately reviewed launcher and current parent live-idle gate"
    ;;
  *)
    die "usage: $0 [prepare|execute]"
    ;;
esac
