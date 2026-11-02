#!/usr/bin/env bash
set -euo pipefail

# H1.37-R: the preregistered 48 original-period H1.37 BASE cells only.  This
# is a same-period bridge, not a new task sample, an adapter evaluation, or a
# training run.  It is intentionally idempotent for *complete* cells and fails
# closed for partial cells, so a launcher interruption cannot silently consume
# duplicate formal episodes.

CAMPAIGN_ID="20260818-2200-h137-r-period-bridge"
RUN_NAME="phase-h1.37-r-period-bridge-v1-20260818"
EXECUTION_SHA="${OPENEVO_H137R_EXECUTION_SHA:?set to the committed H1.37-R preregistration SHA}"
HOST_REPO="${OPENEVO_H137R_HOST_REPO:?set to the exact standalone server clone}"
HOST_RUNS="${OPENEVO_H137R_HOST_RUNS:-/workspace/runs}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
ORIGINAL_ROOT="${HOST_RUNS}/phase-h1.37-frozen-adapter-fresh-task-eval-v1-20260818"
ROOT="/workspace/runs/${RUN_NAME}"
IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
PYTHON_FREEZE_SHA="e3cb53bced07b519157ceca2698935bf0be45a58d08465c87c618ef62c31ee18"
CORE_HOST="/workspace/migration/runtime-src-snapshot/openevo-core"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
MODEL="/workspace/models/qwen2.5-7b-instruct"
MODEL_HOST_DIR="/workspace/models/qwen2.5-7b-instruct"
MODEL_REVISION="0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
ORIGINAL_MANIFEST="${ORIGINAL_ROOT}/manifests/h1.37-task-manifest.json"
ORIGINAL_MANIFEST_SHA="6e55fd5d06a0f2461af2c28643c0aa5d6a15e3048fe7cd182ec9b3b440a0ed07"
BRIDGE_MANIFEST="${ROOT}/manifests/h1.37-r-period-bridge-manifest.json"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"
JDK="/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1"

GPU_UUIDS=(
  GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000
)

die() { printf 'H1.37-R gate failed: %s\n' "$*" >&2; exit 2; }
[[ "$(hostname)" == experiment-host && "$(id -u)" == 0 ]] || die "expected root@experiment-host control plane"
[[ "${HOST_REPO}" == /workspace/* ]] || die "HOST_REPO must be under /workspace"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "server clone SHA mismatch"
jq -e --arg id "${CAMPAIGN_ID}" '.campaign_id == $id and .phase == "H1.37-R" and .status == "launch-authorized" and .formal_task_consumption_allowed == true' "${HOST_REPO}/configs/experiment/current-campaign.json" >/dev/null || die "current campaign mismatch"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo SHA mismatch"
[[ "$("${PYTHON}" -m pip freeze --all | LC_ALL=C sort | sha256sum | awk '{print $1}')" == "${PYTHON_FREEZE_SHA}" ]] || die "Python freeze mismatch"
[[ -f "${MODEL_HOST_DIR}/config.json" && -f "${MODEL_HOST_DIR}/model.safetensors.index.json" && -f "${MODEL_HOST_DIR}/tokenizer_config.json" && -f "${MODEL_HOST_DIR}/tokenizer.json" ]] || die "model tokenizer incomplete"
_cfg="$(sha256sum "${MODEL_HOST_DIR}/config.json" | awk '{print $1}')"; _idx="$(sha256sum "${MODEL_HOST_DIR}/model.safetensors.index.json" | awk '{print $1}')"; _tok="$(sha256sum "${MODEL_HOST_DIR}/tokenizer_config.json" | awk '{print $1}')"
[[ "$(printf '%s\n%s\n%s\n' "${_cfg}" "${_idx}" "${_tok}" | sha256sum | awk '{print $1}')" == "${MODEL_REVISION}" ]] || die "model content pin mismatch"
[[ -f "${ORIGINAL_MANIFEST}" ]] || die "original H1.37 manifest missing"
[[ "$(sha256sum "${ORIGINAL_MANIFEST}" | awk '{print $1}')" == "${ORIGINAL_MANIFEST_SHA}" ]] || die "original manifest pin mismatch"

mkdir -p "${HOST_ROOT}"/{manifests,evaluation,evidence}; chown -R 1001:1001 "${HOST_ROOT}"
TASKS_JSON="$(jq -c '.planned_matrix.tasks' "${HOST_REPO}/configs/experiment/h1.37-r-period-bridge.json")"
[[ "$(jq 'length' <<<"${TASKS_JSON}")" == 48 ]] || die "bridge task list must contain 48 tasks"
if [[ ! -f "${HOST_ROOT}/manifests/h1.37-r-period-bridge-manifest.json" ]]; then
  jq --argjson tasks "${TASKS_JSON}" --arg campaign "${CAMPAIGN_ID}" --arg sha "${ORIGINAL_MANIFEST_SHA}" --arg generated "$(date -u +%FT%TZ)" \
    '{schema:"openevo.webshop-h1.37-r-period-bridge-manifest.v1",campaign_id:$campaign,generated_at_utc:$generated,source_h1_37_manifest_sha256:$sha,evaluation_tasks:$tasks,tasks:[.tasks[] | select(.goal_idx as $id | $tasks | index($id))],evaluation:{arms:["base-r"],rollout_seeds:[2026090200,2026090201,2026090202,2026090203],cells:48,attempts:192},contamination:{no_new_heldout_tasks:true,no_training:true,adapter:"none"}}' "${ORIGINAL_MANIFEST}" > "${HOST_ROOT}/manifests/h1.37-r-period-bridge-manifest.json"
fi
jq -e --argjson tasks "${TASKS_JSON}" --arg sha "${ORIGINAL_MANIFEST_SHA}" '.source_h1_37_manifest_sha256 == $sha and .evaluation_tasks == $tasks and (.tasks|length) == 48 and .evaluation.attempts == 192' "${HOST_ROOT}/manifests/h1.37-r-period-bridge-manifest.json" >/dev/null || die "bridge manifest identity mismatch"
MANIFEST_SHA="$(sha256sum "${HOST_ROOT}/manifests/h1.37-r-period-bridge-manifest.json" | awk '{print $1}')"
LAUNCHER_SHA="$(sha256sum "${HOST_REPO}/scripts/openevo_webshop/launch_h137_r_period_bridge.sh" | awk '{print $1}')"
jq -n --arg campaign "${CAMPAIGN_ID}" --arg execution_sha "${EXECUTION_SHA}" --arg original_manifest_sha "${ORIGINAL_MANIFEST_SHA}" --arg bridge_manifest_sha "${MANIFEST_SHA}" --arg launcher_sha "${LAUNCHER_SHA}" --arg generated "$(date -u +%FT%TZ)" --argjson tasks "${TASKS_JSON}" '{schema:"openevo.webshop-h1.37-r-seal.v1",campaign_id:$campaign,execution_sha:$execution_sha,original_manifest_sha256:$original_manifest_sha,bridge_manifest_sha256:$bridge_manifest_sha,launcher_relative_path:"scripts/openevo_webshop/launch_h137_r_period_bridge.sh",launcher_file_sha256:$launcher_sha,tasks:$tasks,sealed_at_utc:$generated}' > "${HOST_ROOT}/evidence/h1.37-r-seal.json"
chown -R 1001:1001 "${HOST_ROOT}"

utility() { docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=2g --shm-size=2g --user 1001:1001 -v "${HOST_REPO}":/workspace/repo:ro -v "${HOST_RUNS}":/workspace/runs -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -w /workspace/repo -e JAVA_HOME="${JDK}" -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src -e PYTHONDONTWRITEBYTECODE=1 --entrypoint "${PYTHON}" "${IMAGE}" "$@"; }
utility -m unittest discover -s tests -p test_experiment_governance.py || die "CPU governance gate failed"
if [[ "${TRACKING_MODE}" == online ]]; then
  set +e
  probe="$(timeout 45s docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=512m -e WANDB_MODE=online -e WANDB_INIT_TIMEOUT=10 -e WANDB_SILENT=true -e WANDB_API_KEY "${IMAGE}" python3 -c 'import json, wandb
try:
 run=wandb.init(project="openevo-webshop-experiments",entity="example-entity",name="h137r-online-probe",job_type="preflight",mode="online",reinit="create_new"); print(json.dumps({"status":"ok","run_id":getattr(run,"id",None)})); run.finish()
except BaseException as exc:
 print(json.dumps({"status":"failed","error_type":type(exc).__name__,"error":str(exc)})); raise' 2>&1)"
  probe_rc=$?
  set -e
  jq -n --arg output "${probe}" --argjson exit_code "${probe_rc}" '{schema:"openevo.wandb-online-probe.v1",requested_mode:"online",exit_code:$exit_code,output:$output}' > "${HOST_ROOT}/evidence/wandb-online-probe.json"
  if [[ "${probe_rc}" != 0 || "${probe}" != *'"status": "ok"'* ]]; then TRACKING_MODE=offline; fi
elif [[ "${TRACKING_MODE}" != offline ]]; then die "tracking mode must be online or offline"; fi

idle_gpus() {
  local uuid line mem util apps
  for uuid in "${GPU_UUIDS[@]}"; do
    line="$(nvidia-smi -i "${uuid}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" || continue
    IFS=',' read -r mem util <<<"${line}"; mem="${mem//[[:space:]]/}"; util="${util//[[:space:]]/}"
    apps="$(nvidia-smi -i "${uuid}" --query-compute-apps=pid --format=csv,noheader,nounits)"
    [[ "${mem}" =~ ^[0-9]+$ && "${mem}" -le 10 && "${util}" == 0 && -z "${apps}" ]] && printf '%s\n' "${uuid}"
  done
}
gpu_snapshot() { nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader; }
gpu_snapshot > "${HOST_ROOT}/evidence/gpu-preflight-$(date -u +%Y%m%dT%H%M%SZ).csv"

cell_state() {
  local task="$1" path="${HOST_ROOT}/evaluation/base-r-task${1}/attempts.jsonl"
  [[ ! -e "${path}" ]] && { echo missing; return; }
  if jq -s --argjson task "${task}" 'length == 4 and ([.[] | select(.arm == "base-r" and .goal_idx == $task) | .rollout_seed] | sort) == [2026090200,2026090201,2026090202,2026090203]' "${path}" >/dev/null 2>&1; then echo complete; return; fi
  die "partial or malformed formal cell task=${task}; do not overwrite ${path}"
}
first_cell_gate() {
  local task="$1" path="${HOST_ROOT}/evaluation/base-r-task${1}/attempts.jsonl"
  jq -es --argjson task "${task}" 'length == 4 and ([.[] | select(.arm == "base-r" and .goal_idx == $task) | .rollout_seed] | sort) == [2026090200,2026090201,2026090202,2026090203] and all(.[]; .identity_ok and .trace_complete and (.parser_or_fallback_failure | not) and (.error == null))' "${path}" >/dev/null || die "bounded first-cell GPU gate did not produce four valid frozen attempts; do not scale"
}
run_cell() {
  local task="$1" uuid="$2" out="${ROOT}/evaluation/base-r-task${1}" host_out="${HOST_ROOT}/evaluation/base-r-task${1}"
  [[ "$(cell_state "${task}")" == complete ]] && return 0
  mkdir -p "${host_out}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts,hf}; chown -R 1001:1001 "${host_out}"
  docker run --rm --name "h137r-base-${task}" --read-only --tmpfs /tmp:rw,nosuid,size=4g --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 --gpus "device=${uuid}" --user 1001:1001 -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -v "${HOST_REPO}":/workspace/repo:ro -v "${CORE_HOST}":/workspace/openevo-core:ro -v /workspace/models:/workspace/models:ro -v /workspace/models:/workspace/workspace/models:ro -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro -v "${HOST_RUNS}":/workspace/runs -w /workspace/repo -e WANDB_API_KEY -e WANDB_ENTITY=example-entity -e WANDB_PROJECT=openevo-webshop-experiments -e WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_CAMPAIGN_ID="${CAMPAIGN_ID}" -e CUDA_VISIBLE_DEVICES="${uuid}" -e TOKENIZERS_PARALLELISM=false -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e JAVA_HOME="${JDK}" -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" -e PYTHONPATH=/workspace/openevo-core/src:/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src -e PYTHONDONTWRITEBYTECODE=1 -e HOME="${out}/home" -e TMPDIR="${out}/tmp" -e XDG_CACHE_HOME="${out}/xdg-cache" -e HF_HOME=/workspace/hf-cache -e WANDB_DIR="${out}/wandb" -e WANDB_CACHE_DIR="${out}/wandb-cache" -e WANDB_CONFIG_DIR="${out}/wandb-config" -e WANDB_ARTIFACT_DIR="${out}/wandb-artifacts" --entrypoint "${PYTHON}" "${IMAGE}" -m scripts.openevo_webshop.h1_closed_loop_tracked --task-manifest "${BRIDGE_MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}" --adapter none --arm base-r --tasks "${task}" --rollout-seeds 2026090200 2026090201 2026090202 2026090203 --action-contract seed3090.webshop-action-parser.v3-action-command --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature 0.0 --max-steps 30 --max-new 512 --prompt-contract seed3090.webshop-compact-action-prompt.v2 --wandb-mode "${TRACKING_MODE}" --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb"
}

mapfile -t TASKS < <(jq -r '.[]' <<<"${TASKS_JSON}")
mapfile -t FIRST_GPU < <(idle_gpus); [[ "${#FIRST_GPU[@]}" -gt 0 ]] || die "no authorized live-idle GPU for bounded first cell"
run_cell "${TASKS[0]}" "${FIRST_GPU[0]}"
first_cell_gate "${TASKS[0]}"
for ((start=1; start < ${#TASKS[@]}; )); do
  mapfile -t AVAILABLE < <(idle_gpus); [[ "${#AVAILABLE[@]}" -gt 0 ]] || die "no authorized live-idle GPU for next bridge wave"
  pids=(); limit="${#AVAILABLE[@]}"
  for ((i=0; i<limit && start+i<${#TASKS[@]}; i++)); do run_cell "${TASKS[$((start+i))]}" "${AVAILABLE[$i]}" & pids+=("$!"); done
  rc=0; for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done; [[ "${rc}" == 0 ]] || die "a bridge cell failed; completed cells are preserved, partial cells require reconciliation"
  start=$((start+limit))
done
utility -m scripts.openevo_webshop.h1_37_r_analysis --config configs/experiment/h1.37-r-period-bridge.json --original-root /workspace/runs/phase-h1.37-frozen-adapter-fresh-task-eval-v1-20260818 --bridge-root "${ROOT}" --output "${ROOT}/evidence/h1.37-r-analysis.json"
if [[ "${TRACKING_MODE}" == offline ]]; then
  spool_hash="$(find "${HOST_ROOT}" -type f -path '*/wandb/*' -print0 | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | awk '{print $1}')"
  jq -n --arg spool_hash "${spool_hash}" '{schema:"openevo.wandb-fallback-receipt.v1",mode:"offline",reason:"online initialization failed; see wandb-online-probe.json",spool_hash:$spool_hash}' > "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
fi
printf 'H1.37-R complete under %s\n' "${HOST_ROOT}"
