#!/usr/bin/env bash
set -euo pipefail

# H1.39-MR frozen-adapter independent replication.
# This driver consumes only the already-sealed panel and frozen H1.38B adapters.
# It does not train, select, replace, or retry a scientific attempt.

CAMPAIGN_ID="20260820-0139-h139-mr-independent-replication"
EXECUTION_SHA="${OPENEVO_H139_EXECUTION_SHA:?set the exact launch commit SHA}"
HOST_REPO="${OPENEVO_H139_HOST_REPO:?set the standalone server worktree path}"
HOST_RUNS="${OPENEVO_H139_HOST_RUNS:-/workspace/runs}"
RUN_NAME="phase-h139-mr-independent-replication-20260820"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
ROOT="/workspace/runs/${RUN_NAME}"
IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
JDK="/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1"
CORE_HOST="/workspace/migration/runtime-src-snapshot/openevo-core"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
MODEL="/workspace/models/qwen2.5-7b-instruct"
MODEL_REVISION="0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
MANIFEST="${ROOT}/manifests/h1.39-mr-independent-panel-v1.json"
MANIFEST_HOST="${HOST_ROOT}/manifests/h1.39-mr-independent-panel-v1.json"
EVAL_DIR="${ROOT}/evaluation"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"
GPU_UUIDS=(
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
)
ARMS=(base ordinary-lora sd-lora)
SEEDS=(2026103900 2026103901 2026103902 2026103903)
ORDINARY_ADAPTER="/workspace/runs/phase-h1.38b-method-control-v1-20260820-correction-02/artifacts/ordinary-lora/stage07"
SD_ADAPTER="/workspace/runs/phase-h1.38b-method-control-v1-20260820-correction-02/artifacts/sd-lora/workers/sd-lora-b2e7e1aae2fd0f37a79073e9148da390/adapter"
ORDINARY_ADAPTER_HOST="${HOST_RUNS}/phase-h1.38b-method-control-v1-20260820-correction-02/artifacts/ordinary-lora/stage07"
SD_ADAPTER_HOST="${HOST_RUNS}/phase-h1.38b-method-control-v1-20260820-correction-02/artifacts/sd-lora/workers/sd-lora-b2e7e1aae2fd0f37a79073e9148da390/adapter"
ORDINARY_FILE_SHA="22a5adb1a0bdda5b56372a2926bc2724e45b416493403ece86755330add8cd7d"
ORDINARY_TENSOR_SHA="e3be874b71fde1c989e26531255cad063607b73b4727bda59ba824a0f4a7c262"
SD_FILE_SHA="47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c"
SD_TENSOR_SHA="eedc36232f04a8a7d6e8fc6a38b935758db65206f4945a6cfa35c4ab666a48f9"

die() { printf 'H1.39-MR launch gate failed: %s\n' "$*" >&2; exit 2; }
[[ "$(hostname)" == "experiment-host" ]] || die "wrong control-plane hostname"
[[ "$(id -u)" == "0" ]] || die "control plane is not root"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "server worktree SHA mismatch"
[[ -x "${JDK}/bin/javac" ]] || die "pinned JDK is missing javac"
jq -e --arg campaign "${CAMPAIGN_ID}" '.campaign_id == $campaign and .status == "launch-authorized" and .formal_task_consumption_allowed == true' "${HOST_REPO}/configs/experiment/current-campaign.json" >/dev/null || die "campaign is not formally launch-authorized"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "container image digest mismatch"
[[ "$(sha256sum "${HOST_REPO}/configs/experiment/manifests/h1.39-mr-independent-panel-v1.json" | awk '{print $1}')" == "cf2dc0726c34ed94640c36509d5a5018d1261d888b9a75a97fdaddbf54662e00" ]] || die "panel file hash mismatch"
[[ "$(sha256sum "${ORDINARY_ADAPTER_HOST}/adapter_model.safetensors" | awk '{print $1}')" == "22a5adb1a0bdda5b56372a2926bc2724e45b416493403ece86755330add8cd7d" ]] || die "ordinary adapter file hash mismatch"
[[ "$(sha256sum "${SD_ADAPTER_HOST}/adapter_model.safetensors" | awk '{print $1}')" == "47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c" ]] || die "SD adapter file hash mismatch"

mkdir -p "${HOST_ROOT}"/{manifests,evaluation,evidence}
chown -R 1001:1001 "${HOST_ROOT}"
cp "${HOST_REPO}/configs/experiment/manifests/h1.39-mr-independent-panel-v1.json" "${HOST_ROOT}/manifests/"
chown -R 1001:1001 "${HOST_ROOT}/manifests"
[[ "$(sha256sum "${MANIFEST_HOST}" | awk '{print $1}')" == "cf2dc0726c34ed94640c36509d5a5018d1261d888b9a75a97fdaddbf54662e00" ]] || die "run manifest copy changed"
mapfile -t TASKS < <(jq -r '.evaluation_tasks[]' "${MANIFEST_HOST}")
[[ "${#TASKS[@]}" == 64 ]] || die "manifest task count is not 64"

for receipt in base ordinary-lora sd-lora; do
  [[ "$(jq -r '.status' "${HOST_ROOT}/qualification/${receipt}.json" 2>/dev/null || true)" == "pass" ]] || die "missing/pending no-task qualification: ${receipt}"
done

gpu_gate() {
  local uuid row memory utilization apps
  for uuid in "${GPU_UUIDS[@]}"; do
    row="$(nvidia-smi -i "${uuid}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
    IFS=',' read -r memory utilization <<<"${row}"
    memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
    [[ "${memory}" =~ ^[0-9]+$ && "${memory}" -le 10 ]] || die "${uuid} occupied/owner-unresolved (${memory} MiB)"
    [[ "${utilization}" == 0 ]] || die "${uuid} utilization is ${utilization}%"
    apps="$(nvidia-smi -i "${uuid}" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits)"
    [[ -z "${apps}" ]] || die "${uuid} has a visible compute process"
  done
}

gpu_hours() {
  local files
  files="$(find "${HOST_ROOT}/evaluation" -name summary.json -type f -print -quit 2>/dev/null || true)"
  [[ -n "${files}" ]] || { printf '0\n'; return; }
  find "${HOST_ROOT}/evaluation" -name summary.json -type f -print0 \
    | xargs -0 jq -s '[.[].wall_time_seconds // 0] | add / 3600'
}

within_cap() {
  local used
  used="$(gpu_hours)"
  awk -v used="${used}" 'BEGIN { exit !(used < 5.0) }' || die "measured GPU-hour cap reached (${used}h)"
}

run_gpu_container() {
  local uuid="$1" arm="$2" task="$3" out="$4" adapter="$5" file_sha="$6" tensor_sha="$7"
  local host_out="${HOST_RUNS}/${out#/workspace/runs/}"
  mkdir -p "${host_out}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts}
  chown -R 1001:1001 "${host_out}"
  local args=(
    -m scripts.openevo_webshop.h1_closed_loop_tracked
    --task-manifest "${MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}"
    --adapter "${adapter}" --arm "${arm}" --tasks "${task}" --rollout-seeds "${SEEDS[@]}"
    --evaluation-stage fresh --action-contract seed3090.webshop-action-parser.v3-action-command
    --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}"
    --runtime-fingerprint "e3cb53bced07b519157ceca2698935bf0be45a58d08465c87c618ef62c31ee18" --image-digest "${IMAGE_DIGEST}"
    --temperature 0.0 --max-steps 30 --max-new 512 --prompt-contract seed3090.webshop-compact-action-prompt.v2
    --wandb-mode "${TRACKING_MODE}" --wandb-project openevo-webshop-experiments --wandb-entity example-entity
    --wandb-group "${CAMPAIGN_ID}" --campaign-id "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb"
  )
  if [[ "${arm}" != base ]]; then
    args+=(--expected-adapter-file-sha256 "${file_sha}" --expected-adapter-tensor-digest "${tensor_sha}")
  fi
  docker run --rm --name "h139-${arm}-${task}" --read-only --tmpfs /tmp:rw,nosuid,size=4g --shm-size=64g --pids-limit 4096 --gpus "device=${uuid}" --user 1001:1001 \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -v "${HOST_REPO}":/workspace/repo:ro \
    -v "${CORE_HOST}":/workspace/openevo-core:ro -v /workspace/models:/workspace/models:ro -v /workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4:/workspace/hf-cache:ro -v "${HOST_RUNS}":/workspace/runs \
    -w /workspace/repo -e CUDA_VISIBLE_DEVICES="${uuid}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
    -e JAVA_HOME="${JDK}" -e JDK_HOME="${JDK}" -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -e PYTHONPATH=/workspace/openevo-core/src:/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e WANDB_MODE="${TRACKING_MODE}" -e WANDB_ENTITY=example-entity -e WANDB_PROJECT=openevo-webshop-experiments \
    -e HOME="${out}/home" -e TMPDIR="${out}/tmp" -e XDG_CACHE_HOME="${out}/xdg-cache" -e HF_HOME=/workspace/hf-cache \
    -e WANDB_DIR="${out}/wandb" -e WANDB_CACHE_DIR="${out}/wandb-cache" -e WANDB_CONFIG_DIR="${out}/wandb-config" -e WANDB_ARTIFACT_DIR="${out}/wandb-artifacts" \
    --entrypoint "${PYTHON}" "${IMAGE}" "${args[@]}"
}

archive_javac_invalid_cell() {
  local arm="$1" task="$2" cell_root="$3" archive_root
  [[ -f "${cell_root}/attempts.jsonl" ]] || return 1
  jq -s -e 'length == 4 and all(.[]; ((.error // "") | contains("Unable to find javac")))' "${cell_root}/attempts.jsonl" >/dev/null || return 1
  archive_root="${HOST_ROOT}/invalid-engineering/javac/${arm}-task${task}"
  [[ ! -e "${archive_root}" ]] || die "duplicate javac-invalid archive: ${arm}:${task}"
  mkdir -p "$(dirname "${archive_root}")"
  mv "${cell_root}" "${archive_root}"
  return 0
}

gpu_gate
jq -n --arg observed_at_utc "$(date -u +%FT%TZ)" --argjson uuids "$(printf '%s\n' "${GPU_UUIDS[@]}" | jq -R . | jq -s .)" '{schema:"openevo.webshop-h1.39-mr-live-gate.v1",status:"pass",observed_at_utc:$observed_at_utc,authorized_gpu_uuids:$uuids,excluded_gpu_indices:[4,7],note:"all selected UUIDs passed immediate <=10 MiB/0%/no-visible-compute-app gate"}' > "${HOST_ROOT}/evidence/live-gpu-gate.json"
chown 1001:1001 "${HOST_ROOT}/evidence/live-gpu-gate.json"

for arm in "${ARMS[@]}"; do
  case "${arm}" in
    base) adapter=none; file_sha=none; tensor_sha=none ;;
    ordinary-lora) adapter="${ORDINARY_ADAPTER}"; file_sha="${ORDINARY_FILE_SHA}"; tensor_sha="${ORDINARY_TENSOR_SHA}" ;;
    sd-lora) adapter="${SD_ADAPTER}"; file_sha="${SD_FILE_SHA}"; tensor_sha="${SD_TENSOR_SHA}" ;;
  esac
  for ((wave=0; wave<${#TASKS[@]}; wave+=${#GPU_UUIDS[@]})); do
    # One task per UUID per wave. Every wave rechecks live idleness before launch.
    gpu_gate
    within_cap
    pids=()
    for ((offset=0; offset<${#GPU_UUIDS[@]} && wave+offset<${#TASKS[@]}; offset++)); do
      uuid="${GPU_UUIDS[$offset]}"; task="${TASKS[$((wave + offset))]}"; out="${EVAL_DIR}/${arm}-task${task}"
      cell_root="${HOST_RUNS}/${out#/workspace/runs/}"
      if [[ -f "${cell_root}/attempts.jsonl" ]]; then
        if ! archive_javac_invalid_cell "${arm}" "${task}" "${cell_root}"; then
          lines="$(wc -l < "${cell_root}/attempts.jsonl")"
          [[ "${lines}" == 4 ]] || die "partial cell exists and cannot be resumed: ${arm}:${task}"
          continue
        fi
      fi
      run_gpu_container "${uuid}" "${arm}" "${task}" "${out}" "${adapter}" "${file_sha}" "${tensor_sha}" &
      pids+=("$!")
    done
    rc=0; for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
    [[ "${rc}" == 0 ]] || die "evaluation wave failed for ${arm}"
    within_cap
  done
done

cell_args=()
for arm in "${ARMS[@]}"; do for task in "${TASKS[@]}"; do cell_args+=(--cell "${arm}:${task}:${EVAL_DIR}/${arm}-task${task}"); done; done
run_utility_container() {
  docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=2g --shm-size=2g --pids-limit 1024 --user 1001:1001 \
    -v "${HOST_REPO}":/workspace/repo:ro -v "${HOST_RUNS}":/workspace/runs:rw -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
    -w /workspace/repo -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}
run_utility_container -m scripts.openevo_webshop.reconcile_h139_mr --manifest "${MANIFEST}" "${cell_args[@]}" --output "${ROOT}/evidence/h1.39-mr-reconciliation.json"
run_utility_container -m scripts.openevo_webshop.analyze_h139_mr --reconciliation "${ROOT}/evidence/h1.39-mr-reconciliation.json" --output "${ROOT}/evidence/h1.39-mr-analysis.json"
printf 'H1.39-MR completed: %s\n' "${HOST_ROOT}"
