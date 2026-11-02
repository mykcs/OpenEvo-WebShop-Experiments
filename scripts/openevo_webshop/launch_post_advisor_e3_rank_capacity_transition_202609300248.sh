#!/usr/bin/env bash
set -euo pipefail
WT="${BOUNDED_EXECUTION_REPO:?execution checkout required}"
CONTRACT="$WT/configs/experiment/path-contracts/post-advisor-e3-rank-capacity-path-contract-202609300734.json"
exec "$WT/scripts/oe" host --path-contract "$CONTRACT" --cwd "$WT" -- "$WT/scripts/openevo_webshop/post_advisor_e3_rank_capacity_transition_driver_202609300248.sh"
