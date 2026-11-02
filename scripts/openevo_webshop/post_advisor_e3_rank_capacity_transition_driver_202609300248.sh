#!/usr/bin/env bash
set -euo pipefail
IMAGE="experiment-host/openevo-scientific:h144-runtime-v1"  # legacy-revision-compat

C="${BOUNDED_TRANSITION_CONTAINER:?}"
WT="${BOUNDED_EXECUTION_REPO:?}"
ROOT="${BOUNDED_ANALYSIS_ROOT:?}"
GPU="${BOUNDED_GPU_UUID:?}"
ROUND="${BOUNDED_ROUND_INDEX:?}"
MODE="${BOUNDED_TREATMENT_MODE:?}"
RANK="${BOUNDED_STATE_RANK:?}"
DS="${BOUNDED_DATASET_ROOT:?}"
OUT="${BOUNDED_TRANSITION_OUTPUT:?}"
PRIOR="${BOUNDED_PRIOR_ADAPTER:?}"
AUTH_ROOT="${OPENEVO_POST_ADVISOR_AUTHORITY_REPO:?}"
AGGREGATE="${OPENEVO_E3_AGGREGATE_AUTHORITY:?}"
ARM="${OPENEVO_E3_ARM_AUTHORITY:?}"
E3_ROUTER="${OPENEVO_E3_CURRENT_CAMPAIGN:?}"
FORMAL_START="${OPENEVO_FORMAL_START_RECEIPT:?}"
ASSIGNED="${OPENEVO_ASSIGNED_GPU_UUIDS:?}"
RUNS=/workspace/runs/post-advisor-unanswered-20260928
CORE=/workspace/control/openevo-core
QUAL=/workspace/openevo-workspace/qualifications
MODEL="$ROOT/stage2-base/qwen3-1p7b"
DRIVER="$WT/scripts/openevo_webshop/post_advisor_e3_rank_capacity_transition_driver_202609300248.py"
EMPTY="$WT/docs/evidence/bounded-recurrence-gdr-20260915/empty-bounded-history-manifest-20260915T104300.json"
CFG="$WT/docs/evidence/bounded-recurrence-gdr-20260915/first-state-rank128-adapter-config-template-20260915T104300.json"

[[ "$MODE" == BOUNDED_OFF ]] || { echo "NO-GO E3 treatment mode" >&2; exit 2; }
case "$RANK" in 8|16|64) ;; *) echo "NO-GO E3 rank" >&2; exit 2;; esac
[[ "$ROUND" -ge 152 && "$ROUND" -le 159 ]] || { echo "NO-GO E3 round" >&2; exit 2; }
[[ -f "$DRIVER" && -f "$AGGREGATE" && -f "$ARM" && -f "$E3_ROUTER" && -f "$FORMAL_START" ]] || { echo "NO-GO E3 authority/driver/Formal Start" >&2; exit 2; }
case "$FORMAL_START" in "$RUNS"/*) ;; *) echo "NO-GO E3 Formal Start path" >&2; exit 2;; esac
PYTHONPATH="$WT" python3 -m scripts.openevo_webshop.post_advisor_e3_guard_202609300248 --repo "$WT" --authority "$AGGREGATE" --formal >/dev/null

BINDING="$(python3 - "$AGGREGATE" "$ARM" "$RANK" "$ROUND" "$OUT" "$PRIOR" "${BOUNDED_PRIOR_REPLAY:-}" <<'PY'
import json, sys
from pathlib import Path
aggregate, arm_path, rank_text, round_text, out_text, prior_text, replay_text = sys.argv[1:]
rank, round_index = int(rank_text), int(round_text)
a = json.loads(Path(aggregate).read_text())
arm = json.loads(Path(arm_path).read_text())
cell = a['arms'][str(rank)]
if (arm.get('persistent_state_rank') != rank or arm.get('output_root') != cell['output_root']
        or arm.get('initial_adapter_path') != cell['initial_adapter_path']
        or Path(out_text) != Path(cell['output_root']) / f'round-{round_index:03d}' / 'sd-lora' / 'bounded-state'):
    raise SystemExit('E3 arm/output authority drift')
if round_index == 152 and Path(prior_text) != Path(cell['initial_adapter_path']):
    raise SystemExit('E3 R152 prior adapter drift')
if round_index == 152 and Path(replay_text) != Path(arm['initial_replay_path']):
    raise SystemExit('E3 R152 replay route drift')
if round_index > 152 and not str(Path(prior_text)).startswith(str(Path(cell['output_root'])) + '/'):
    raise SystemExit('E3 continuing prior adapter outside arm output')
print(cell['output_root'])
PY
)"
E3_RUN_ROOT="${BINDING:?}"
REPLAY="${BOUNDED_PRIOR_REPLAY:-$PRIOR/openevo_sd_lora_replay.jsonl}"
[[ -f "$REPLAY" && -f "$PRIOR/adapter_model.safetensors" ]] || { echo "NO-GO E3 prior files" >&2; exit 2; }
[[ "$(stat -c %u "$REPLAY")" == 1001 && "$(stat -c %h "$REPLAY")" == 1 ]] || { echo "NO-GO E3 replay must be UID1001/nlink1" >&2; exit 2; }
OUTPUT_PARENT="$(dirname "$OUT")"
[[ -d "$OUTPUT_PARENT" && "$(stat -c %u "$OUTPUT_PARENT")" == 1001 ]] || { echo "NO-GO E3 output parent owner" >&2; exit 2; }
[[ ! -e "$OUT" || -f "$OUT/bounded_recurrence_receipt.json" ]] || { echo "NO-GO E3 partial transition output" >&2; exit 2; }

docker run -d --name "$C" --restart no --gpus "device=$GPU" --network none --user 1001:1001 \
  --tmpfs /tmp:rw,exec,nosuid,size=4g,mode=1777 \
  -e CUDA_VISIBLE_DEVICES="$GPU" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e HOME=/tmp/openevo-home -e XDG_CACHE_HOME=/tmp/openevo-cache -e HF_HOME=/tmp/huggingface \
  -e TRITON_CACHE_DIR=/tmp/triton -e TMPDIR=/tmp -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_openevo \
  -e PYTHONDONTWRITEBYTECODE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH="$CORE/src:$WT" \
  -e OPENEVO_POST_ADVISOR_AUTHORITY_REPO="$AUTH_ROOT" \
  -e OPENEVO_E3_AGGREGATE_AUTHORITY="$AGGREGATE" -e OPENEVO_E3_ARM_AUTHORITY="$ARM" \
  -e OPENEVO_E3_CURRENT_CAMPAIGN="$E3_ROUTER" \
  -e OPENEVO_FORMAL_START_RECEIPT="$FORMAL_START" -e OPENEVO_ASSIGNED_GPU_UUIDS="$ASSIGNED" \
  -e GIT_CONFIG_COUNT=2 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0="$WT" \
  -e GIT_CONFIG_KEY_1=safe.directory -e GIT_CONFIG_VALUE_1="$AUTH_ROOT" \
  -v "$CORE:$CORE:ro" -v "$WT:$WT:ro" -v "$ROOT:$ROOT:ro" \
  -v "$AUTH_ROOT:$AUTH_ROOT:ro" -v "$RUNS:$RUNS:ro" -v "$QUAL:$QUAL:ro" \
  -v "$OUTPUT_PARENT:$OUTPUT_PARENT" \
  --entrypoint /usr/local/bin/python3 "$IMAGE" "$DRIVER" \
  --dataset-manifest "$DS/manifest.json" --dataset-artifact-id "bounded-formal-r${ROUND}" \
  --output-dir "$OUT" --model-path "$MODEL" --round-index "$ROUND" --state-rank "$RANK" \
  --transition-id "e3-rank${RANK}:r${ROUND}" --treatment-mode BOUNDED_OFF \
  --empty-history-manifest "$EMPTY" --adapter-config-template "$CFG" \
  --prior-adapter "$PRIOR" --prior-replay "$REPLAY"
