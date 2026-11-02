#!/usr/bin/env bash
set -euo pipefail
echo "NO-GO E1 is sealed; historical transition body is evidence only" >&2; exit 2
C="${BOUNDED_TRANSITION_CONTAINER:?}"; WT="${BOUNDED_EXECUTION_REPO:?}"; ROOT="${BOUNDED_ANALYSIS_ROOT:?}"
GPU="${BOUNDED_GPU_UUID:?}"; ROUND="${BOUNDED_ROUND_INDEX:?}"; MODE="${BOUNDED_TREATMENT_MODE:?}"
DS="${BOUNDED_DATASET_ROOT:?}"; OUT="${BOUNDED_TRANSITION_OUTPUT:?}"; PRIOR="${BOUNDED_PRIOR_ADAPTER:-}"
REPLAY="${BOUNDED_PRIOR_REPLAY:-}"
STATE_RANK="${BOUNDED_STATE_RANK:?}"; AUTH_ROOT="${OPENEVO_POST_ADVISOR_AUTHORITY_REPO:?}"
FORMAL_START="${OPENEVO_FORMAL_START_RECEIPT:?}"; ASSIGNED="${OPENEVO_ASSIGNED_GPU_UUIDS:?}"
[[ "$MODE" == MAGNITUDE_SCALAR_GATE_V1 ]] || { echo "NO-GO E1 transition mode=$MODE" >&2; exit 2; }  # legacy-revision-compat
[[ "$ROUND" -ge 152 && "$ROUND" -le 159 ]] || { echo "NO-GO E1 round=$ROUND" >&2; exit 2; }
CORE=/workspace/control/openevo-core
RUNS=/workspace/runs/post-advisor-unanswered-20260928
QUAL=/workspace/openevo-workspace/qualifications
MODEL="$ROOT/stage2-base/qwen3-1p7b"
EMPTY="$WT/docs/evidence/bounded-recurrence-gdr-20260915/empty-bounded-history-manifest-20260915T104300.json"
CFG="$WT/docs/evidence/bounded-recurrence-gdr-20260915/first-state-rank128-adapter-config-template-20260915T104300.json"
DRIVER="$WT/scripts/openevo_webshop/post_advisor_bounded_rank_sweep_transition_driver_202609282338.py"
ARM_AUTH="${OPENEVO_POST_ADVISOR_ARM_AUTHORITY:?}"
[[ -f "$ARM_AUTH" && -f "$FORMAL_START" && -f "$DRIVER" ]] || { echo "NO-GO missing E1 authority, Formal Start, or transition driver" >&2; exit 3; }
case "$FORMAL_START" in "$RUNS"/*) ;; *) echo "NO-GO Formal Start outside campaign run root" >&2; exit 3;; esac
ARGS=(--dataset-manifest "$DS/manifest.json" --dataset-artifact-id "bounded-formal-r${ROUND}"
  --output-dir "$OUT" --model-path "$MODEL" --round-index "$ROUND" --state-rank "$STATE_RANK"
  --transition-id "bounded-formal:${MODE}:r${ROUND}" --treatment-mode "$MODE"
  --empty-history-manifest "$EMPTY" --adapter-config-template "$CFG")
PRIOR_MOUNTS=()
if [[ "$ROUND" -gt 0 ]]; then
  [[ -n "$PRIOR" && -n "$REPLAY" && -f "$REPLAY" ]] || { echo "NO-GO short fork missing prior adapter/replay" >&2; exit 3; }
  [[ "$(stat -c %h "$REPLAY")" == 1 && "$(stat -c %u "$REPLAY")" == 1001 ]] || { echo "NO-GO replay requires single-link UID1001 file" >&2; exit 3; }
  ARGS+=(--prior-adapter "$PRIOR" --prior-replay "$REPLAY")
  case "$PRIOR" in "$RUNS"/*) ;; *) PRIOR_MOUNTS+=(-v "$PRIOR:$PRIOR:ro");; esac
  case "$REPLAY" in "$RUNS"/*) ;; *) PRIOR_MOUNTS+=(-v "$REPLAY:$REPLAY:ro");; esac
fi
if [[ "$ROUND" -ge 152 && "$ROUND" -le 159 ]]; then
  HISTORY=/workspace/runs/bounded-effective-state-gdr-recovery-1e7c4952-20260916/off/round-151/evidence/history-after.json
  R151_ADAPTER=/workspace/runs/bounded-effective-state-gdr-recovery-1e7c4952-20260916/off/round-151/sd-lora/bounded-state
  R151_REPLAY="$R151_ADAPTER/openevo_sd_lora_replay.jsonl"
  BEFORE="$RUNS/e1-beta-direction-formal-a6-6b4ff450/round-152/state-before.json"
  EVIDENCE="$RUNS/e1-beta-direction-formal-a6-6b4ff450/round-152/evidence/evidence.json"
  [[ -f "$HISTORY" && -f "$R151_REPLAY" && -f "$BEFORE" && -f "$EVIDENCE" ]] || { echo "NO-GO missing frozen R151 evidence/replay" >&2; exit 3; }
  python3 - "$BEFORE" "$EVIDENCE" "$HISTORY" "$R151_REPLAY" "$ARM_AUTH" <<"PYHISTORY" || { echo "NO-GO frozen R151 evidence/replay drift" >&2; exit 3; }
import hashlib, json, sys
from pathlib import Path
before, evidence, history, replay, arm_path = map(Path, sys.argv[1:])
state = json.loads(before.read_text())
receipt = json.loads(evidence.read_text())
arm = json.loads(arm_path.read_text())
recovery_ref = arm.get("engineering_recovery") or {}
recovery_path = Path(str(recovery_ref.get("path") or ""))
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
if not recovery_path.is_file() or sha(recovery_path) != recovery_ref.get("sha256"):
    raise SystemExit(1)
recovery = json.loads(recovery_path.read_text())
if (state.get("evidence_history") != str(history)
        or receipt.get("prior_history_sha256") != sha(history)
        or Path(str(state.get("current_sd_adapter") or "")) / replay.name != replay
        or recovery.get("predecessor_replay_sha256") != sha(replay)):
    raise SystemExit(1)
PYHISTORY
  PRIOR_MOUNTS+=(-v "$HISTORY:$HISTORY:ro")
  if [[ "$PRIOR" != "$R151_ADAPTER" ]]; then PRIOR_MOUNTS+=(-v "$R151_REPLAY:$R151_REPLAY:ro"); fi
fi
OUTPUT_PARENT="$(dirname "$OUT")"
EXPECTED_PARENT="$RUNS/e1-beta-direction-formal-a6-6b4ff450/round-$ROUND/a8-transition-output"
[[ "$OUTPUT_PARENT" == "$EXPECTED_PARENT" && "$(basename "$OUT")" == bounded-state ]] || { echo "NO-GO A8 transition output route drift" >&2; exit 3; }
[[ -d "$OUTPUT_PARENT" && "$(stat -c %u "$OUTPUT_PARENT")" == 1001 ]] || { echo "NO-GO A8 output parent must be UID1001" >&2; exit 3; }
if [[ -e "$OUT" && ! -f "$OUT/bounded_recurrence_receipt.json" ]]; then echo "NO-GO partial A8 transition output" >&2; exit 3; fi
docker run -d --name "$C" --restart no --gpus "device=$GPU" --network none --user 1001:1001 \
  --tmpfs /tmp:rw,exec,nosuid,size=4g,mode=1777 \
  -e CUDA_VISIBLE_DEVICES="$GPU" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e HOME=/tmp/openevo-home -e XDG_CACHE_HOME=/tmp/openevo-cache -e HF_HOME=/tmp/huggingface -e TRITON_CACHE_DIR=/tmp/triton -e TMPDIR=/tmp -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_openevo -e PYTHONDONTWRITEBYTECODE=1 \
  -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e PYTHONPATH="$CORE/src:$WT" \
  -e OPENEVO_POST_ADVISOR_AUTHORITY_REPO="$AUTH_ROOT" \
  -e OPENEVO_POST_ADVISOR_ARM_AUTHORITY="$ARM_AUTH" \
  -e OPENEVO_FORMAL_START_RECEIPT="$FORMAL_START" \
  -e OPENEVO_ASSIGNED_GPU_UUIDS="$ASSIGNED" \
  -e GIT_CONFIG_COUNT=2 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0="$WT" \
  -e GIT_CONFIG_KEY_1=safe.directory -e GIT_CONFIG_VALUE_1="$AUTH_ROOT" \
  -v "$CORE:$CORE:ro" -v "$WT:$WT:ro" -v "$ROOT:$ROOT:ro" \
  -v "$AUTH_ROOT:$AUTH_ROOT:ro" -v "$RUNS:$RUNS:ro" -v "$QUAL:$QUAL:ro" \
  "${PRIOR_MOUNTS[@]}" -v "$OUTPUT_PARENT:$OUTPUT_PARENT" \
  --entrypoint /usr/local/bin/python3 experiment-host/openevo-scientific:h144-runtime-v1 "$DRIVER" "${ARGS[@]}"  # legacy-revision-compat
