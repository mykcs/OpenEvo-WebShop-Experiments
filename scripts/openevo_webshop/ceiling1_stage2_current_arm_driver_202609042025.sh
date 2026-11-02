#!/usr/bin/env bash
set -euo pipefail
ARM="${CEILING1_STAGE2_ARM:?3b or qwen3-1p7b}"
REPO="${CEILING1_STAGE2_EXEC_REPO:?exact clean checkout}"
ROOT="${CEILING1_STAGE2_ROOT:?analysis root}"
ACT="${CEILING1_STAGE2_ACTIVATION:?immutable activation receipt}"
OUT="${CEILING1_STAGE2_OUTPUT:?formal Stage2 output root}"
CARRIER_ROOT="${CEILING1_STAGE2_CARRIER_ROOT:?Harness201 PASS carrier seed}"
EXEC_SHA="${CEILING1_STAGE2_EXECUTION_SHA:?exact execution Git SHA}"
RAY_MIGRATION="${CEILING1_STAGE2_RAY_MIGRATION:?authorized Ray migration/continuation receipt}"
SCI_RECOVERY="${CEILING1_STAGE2_SCIENCE_RECOVERY:-}"
SEED_ROOT="${CEILING1_STAGE2_SEED_ROOT:?seed root required by path contract}"
STEP_SEED="$REPO/configs/experiment/campaigns/202609111730/authority-snapshots/020b6b2fccaf3e924f3e22299c45467cf0544f3f5e8d9cbd5df6f7ab839003d7.json"
AUTH="$REPO/configs/experiment/openevo-ceiling1-stage2-runtime-authority-202609042025.json"
case "$ARM" in
  3b) DESIGN="$REPO/configs/experiment/designs/openevo-ceiling1-stage2-vnext-3b-v1.json"; CAPACITY="$REPO/configs/experiment/designs/openevo-ceiling-1.0-harness201-3b-successor-capacity-v1.json" ;;  # legacy-revision-compat
  qwen3-1p7b) DESIGN="$REPO/configs/experiment/designs/openevo-ceiling1-stage2-vnext-qwen3-1p7b-v1.json"; CAPACITY="$REPO/configs/experiment/designs/openevo-ceiling1-stage2-vnext-qwen3-1p7b-capacity-v1.json" ;;  # legacy-revision-compat
  *) echo "NO-GO unsupported arm=$ARM" >&2; exit 2 ;;
esac
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXEC_SHA" ]] || { echo "NO-GO execution SHA drift" >&2; exit 3; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "NO-GO dirty execution checkout" >&2; exit 3; }
for p in "$ACT" "$RAY_MIGRATION" "$AUTH"; do [[ -s "$p" ]] || { echo "NO-GO required authority missing: $p" >&2; exit 4; }; done
[[ -s "$OUT/execution-state.json" ]] || { echo "NO-GO current Ray launcher resumes an initialized Stage2 state; prepare activation/migration first" >&2; exit 5; }
EXTRA=(); [[ -n "$SCI_RECOVERY" ]] && EXTRA+=(--science-recovery-amendment "$SCI_RECOVERY")
exec python3 "$REPO/scripts/openevo_webshop/run_ceiling1_stage2_ray_supervised_202609042025.py" \
  --arm "$ARM" --repo "$REPO" --analysis-root "$ROOT" --seed-root "$SEED_ROOT" --output "$OUT" \
  --stage2-activation "$ACT" --recovery-amendment "$RAY_MIGRATION" "${EXTRA[@]}" \
  --design "$DESIGN" --capacity-policy "$CAPACITY" --step-seed-amendment "$STEP_SEED" \
  --harness201-carrier-root "$CARRIER_ROOT" --execution-git-sha "$EXEC_SHA"
