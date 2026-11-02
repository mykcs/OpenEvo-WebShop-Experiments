#!/usr/bin/env bash
set -euo pipefail
WT="${BOUNDED_EXECUTION_REPO:?execution checkout required}"
CONTRACT="$WT/configs/experiment/path-contracts/post-advisor-e1-beta-direction-path-contract-202609282338.json"
exec "$WT/scripts/oe" host --path-contract "$CONTRACT" --cwd "$WT" -- "$WT/scripts/openevo_webshop/post_advisor_magnitude_transition_driver_20260930.sh"
