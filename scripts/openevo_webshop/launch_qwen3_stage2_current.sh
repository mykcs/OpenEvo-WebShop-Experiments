#!/usr/bin/env bash
set -euo pipefail

ARM=qwen3-1p7b
REPO="${Q17_STAGE2_EXEC_REPO:?exact detached checkout}"
ROOT="${Q17_STAGE2_ROOT:?current shared Stage1/MiniMax root}"
ACT="${Q17_STAGE2_ACTIVATION:?immutable activation receipt}"
OUT="${Q17_STAGE2_OUTPUT:?formal Stage2 output root}"
SEED_ROOT="${Q17_STAGE2_SEED_ROOT:-/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1}"
CARRIER_ROOT="${Q17_STAGE2_CARRIER_ROOT:?Harness201 PASS carrier seed}"
EXEC_SHA="${Q17_STAGE2_EXECUTION_SHA:?exact execution Git SHA}"
CAPACITY="$REPO/configs/experiment/designs/openevo-ceiling1-stage2-vnext-qwen3-1p7b-capacity-v1.json"
STEP_SEED="$REPO/configs/experiment/designs/openevo-ceiling-1.0-harness201-step-seed-amendment-v1.json"

[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXEC_SHA" ]] || { echo "NO-GO execution SHA drift" >&2; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "NO-GO dirty execution checkout" >&2; exit 2; }
[[ -s "$ACT" ]] || { echo "NO-GO activation missing" >&2; exit 3; }
[[ -s "$ROOT/PRE_STAGE2_READY_qwen3-1p7b.json" ]] || { echo "NO-GO PRE_STAGE2_READY missing" >&2; exit 3; }
[[ ! -e "$OUT/execution-state.json" ]] || { echo "NO-GO formal Stage2 output already has execution-state" >&2; exit 4; }
mkdir -p "$OUT"
chown 1001:1001 "$OUT"
chmod 700 "$OUT"

export CEILING1_STAGE2_RESOURCE_POLICY=auto
exec python3 "$REPO/scripts/openevo_webshop/ceiling1_stage2_arm_controller.py" \
  --arm "$ARM" \
  --repo "$REPO" \
  --analysis-root "$ROOT" \
  --seed-root "$SEED_ROOT" \
  --output "$OUT" \
  --stage2-activation "$ACT" \
  --capacity-policy "$CAPACITY" \
  --harness-profile 201.1 \
  --step-seed-policy injective-v1 \
  --step-seed-amendment "$STEP_SEED" \
  --harness201-carrier-root "$CARRIER_ROOT" \
  --execution-git-sha "$EXEC_SHA"
