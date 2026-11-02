#!/usr/bin/env bash
set -euo pipefail
WT="${CEILING1_AUTOPILOT_REPO:?execution checkout required}"
if [[ -n "${OPENEVO_E3_CURRENT_CAMPAIGN:-}" ]]; then
  CONTRACT="$WT/configs/experiment/path-contracts/post-advisor-e3-rank-capacity-path-contract-202609300734.json"
elif [[ -n "${OPENEVO_POST_ADVISOR_AUTHORITY_REPO:-}" ]]; then
  CONTRACT="$WT/configs/experiment/path-contracts/post-advisor-e1-beta-direction-path-contract-202609282338.json"
else
  CONTRACT="$WT/configs/experiment/path-contracts/stage2-qwen3-path-contract-202609300734.json"
fi
exec "$WT/scripts/oe" host --path-contract "$CONTRACT" --cwd "$WT" -- "$WT/scripts/openevo_webshop/bounded_carrier_container_driver_20260916.sh"
