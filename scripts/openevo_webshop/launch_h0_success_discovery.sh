#!/usr/bin/env bash
set -euo pipefail

# H0.1 is a train-only Qwen natural-success discovery diagnostic.  It never
# consumes the H1.30 evaluation panel and never enables formal comparison.

RUN_NAME="${OPENEVO_H01_RUN_NAME:-phase-h0.1-qwen-success-discovery-v2-20260817}"
HOST_RUNS="${OPENEVO_H01_HOST_RUNS:-/workspace/runs}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
ROOT="/workspace/runs/${RUN_NAME}"
HOST_REPO="${OPENEVO_H01_HOST_REPO:?set OPENEVO_H01_HOST_REPO to an exact server checkout}"
EXECUTION_SHA="${OPENEVO_H01_EXECUTION_SHA:?set OPENEVO_H01_EXECUTION_SHA to the code identity}"
HOST_MANIFEST="${OPENEVO_H01_HOST_MANIFEST:?set OPENEVO_H01_HOST_MANIFEST to the captured manifest}"
HOST_SEED_ROOT="${OPENEVO_H01_HOST_SEED_ROOT:?set OPENEVO_H01_HOST_SEED_ROOT to the pinned SEED checkout}"
HOST_DATA_ROOT="${OPENEVO_H01_HOST_DATA_ROOT:?set OPENEVO_H01_HOST_DATA_ROOT to the pinned WebShop catalog}"
HOST_INDEX_ROOT="${OPENEVO_H01_HOST_INDEX_ROOT:?set OPENEVO_H01_HOST_INDEX_ROOT to indexes_1k}"
MANIFEST="${ROOT}/manifests/h0.1-success-discovery-v2.json"
IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
PYTHON_FREEZE_SHA="d7220829d4fc81e232523ea3aa22adfb4b55dba95d8236605a3e6f26b3b379d4"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
SEED_SOURCE_SHA="9996427d3ae7c71b3cc8ac82940efe764a8a2485"
MODEL_REVISION="aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL="/workspace/models/qwen2.5-3b-instruct"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"
CAMPAIGN_ID="20260817-0000-h0.1-qwen-success-discovery-v2"

GPU_UUIDS=(
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
)

die() { printf 'H0.1 success discovery failed: %s\n' "$*" >&2; exit 2; }

[[ "$(hostname)" == "experiment-host" ]] || die "unexpected host"
[[ "$(id -u)" == "0" ]] || die "control plane is not root"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/h0_natural_success_tracked.py" ]] || die "host repo lacks H0.1 runner"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/reconcile_h0_success_discovery.py" ]] || die "host repo lacks reconciliation"
[[ -f "${HOST_MANIFEST}" ]] || die "manifest is absent"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "image digest mismatch"
[[ "$(git -c safe.directory="${HOST_SEED_ROOT}" -C "${HOST_SEED_ROOT}" rev-parse HEAD 2>/dev/null)" == "${SEED_SOURCE_SHA}" ]] || die "SEED source SHA mismatch"

jq -e '.schema == "openevo.webshop-h0.1-success-discovery.v2" and .status == "preregistered-not-authorized" and .formal_task_consumption_allowed == false and .task_count == 16 and (.controls.rollout_seeds | length == 8)' "${HOST_MANIFEST}" >/dev/null || die "H0.1 manifest gate failed"

mkdir -p "${HOST_ROOT}"/{harvest,evidence,manifests}
manifest_copy="${HOST_ROOT}/manifests/h0.1-success-discovery-v2.json"
if [[ "${HOST_MANIFEST}" != "${manifest_copy}" ]]; then
  cp --reflink=auto "${HOST_MANIFEST}" "${manifest_copy}"
fi
chown -R 1001:1001 "${HOST_ROOT}"

gpu_gate() {
  local uuid row memory utilization apps
  for uuid in "${GPU_UUIDS[@]}"; do
    row="$(nvidia-smi -i "${uuid}" --query-gpu=memory.used,utilization.gpu,pstate --format=csv,noheader,nounits)"
    IFS=',' read -r memory utilization _ <<<"${row}"
    memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
    [[ "${memory}" =~ ^[0-9]+$ && "${memory}" -le 10 ]] || die "${uuid} memory is ${memory} MiB"
    [[ "${utilization}" == "0" ]] || die "${uuid} utilization is ${utilization}%"
    apps="$(nvidia-smi -i "${uuid}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
    [[ -z "${apps}" ]] || die "${uuid} has a compute process"
  done
}

common_mounts=(
  -v "${HOST_REPO}":/workspace/repo:ro
  -v "${HOST_RUNS}":/workspace/runs
  -v "${HOST_SEED_ROOT}":/workspace/seed-source:ro
  -v "${HOST_DATA_ROOT}":/workspace/seed-source/agent_system/environments/env_package/webshop/webshop/data:ro
  -v "${HOST_INDEX_ROOT}":/workspace/seed-source/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k:ro
  -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro
  -v /workspace/models:/workspace/models:ro
  -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro
)

run_utility_container() {
  docker run --rm --read-only --tmpfs /tmp:rw,exec,nosuid,size=2g --shm-size=2g --pids-limit 1024 --user 1001:1001 \
    "${common_mounts[@]}" -w /workspace/repo \
    -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e PYTHONDONTWRITEBYTECODE=1 --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}

run_gpu_container() {
  local uuid="$1" name="$2" out="$3"; shift 3
  mkdir -p "${HOST_RUNS}/${out#/workspace/runs/}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts}
  chown -R 1001:1001 "${HOST_RUNS}/${out#/workspace/runs/}"
  docker run --rm --name "${name}" --read-only --tmpfs /tmp:rw,exec,nosuid,size=4g \
    --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 \
    --gpus "device=${uuid}" --user 1001:1001 "${common_mounts[@]}" -w /workspace/repo \
    -e WANDB_API_KEY -e WANDB_ENTITY=example-entity -e WANDB_PROJECT=openevo-webshop-experiments \
    -e WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_CAMPAIGN_ID="${CAMPAIGN_ID}" \
    -e CUDA_VISIBLE_DEVICES="${uuid}" -e TOKENIZERS_PARALLELISM=false -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
    -e JAVA_HOME=/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1 \
    -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e PYTHONDONTWRITEBYTECODE=1 -e HOME="${out}/home" -e TMPDIR="${out}/tmp" -e XDG_CACHE_HOME="${out}/xdg-cache" \
    -e HF_HOME=/workspace/hf-cache -e HF_HUB_CACHE=/workspace/hf-cache -e WANDB_DIR="${out}/wandb" \
    -e WANDB_CACHE_DIR="${out}/wandb-cache" -e WANDB_CONFIG_DIR="${out}/wandb-config" -e WANDB_ARTIFACT_DIR="${out}/wandb-artifacts" \
    --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}

gpu_gate
mapfile -t TASKS < <(jq -r '.tasks[].goal_idx' "${HOST_MANIFEST}")
mapfile -t ROLLOUT_SEEDS < <(jq -r '.controls.rollout_seeds[]' "${HOST_MANIFEST}")
[[ "${#TASKS[@]}" == "16" && "${#ROLLOUT_SEEDS[@]}" == "8" ]] || die "unexpected manifest counts"

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for wave in 0 1; do
  pids=()
  for slot in 0 1 2 3 4 5 6 7; do
    index=$((wave * 8 + slot)); task="${TASKS[$index]}"; uuid="${GPU_UUIDS[$slot]}"
    out="${ROOT}/harvest/wave${wave}/task${task}"
    run_gpu_container "${uuid}" "h01-discovery-${wave}-${task}" "${out}" \
      -m scripts.openevo_webshop.h0_natural_success_tracked \
      --manifest "${MANIFEST}" --seed-root /workspace/seed-source \
      --model "${MODEL}" --model-revision "${MODEL_REVISION}" --tokenizer-revision "${MODEL_REVISION}" \
      --upstream-openevo-sha "${CORE_SHA}" --experiment-git-sha "${EXECUTION_SHA}" \
      --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --output "${out}" \
      --gpu-uuid "${uuid}" --tasks "${task}" --rollout-seeds "${ROLLOUT_SEEDS[@]}" \
      --max-steps 30 --max-new 512 --temperature 0.4 \
      --prompt-contract seed3090.webshop-compact-action-prompt.v2 \
      --wandb-mode "${TRACKING_MODE}" --wandb-project openevo-webshop-experiments \
      --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" \
      --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb" &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
  [[ "${rc}" == "0" ]] || die "wave ${wave} failed"
  gpu_gate
done

run_utility_container -m scripts.openevo_webshop.reconcile_h0_success_discovery \
  --manifest "${MANIFEST}" --run-root "${ROOT}" --output "${ROOT}/evidence/h0.1-reconciliation.json"
gpu_gate
ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
jq -n --arg started "${started_at}" --arg ended "${ended_at}" --arg manifest "${HOST_ROOT}/manifests/h0.1-success-discovery-v2.json" \
  --arg execution_sha "${EXECUTION_SHA}" --arg image_digest "${IMAGE_DIGEST}" --arg mode "${TRACKING_MODE}" \
  --arg reconciliation "${HOST_ROOT}/evidence/h0.1-reconciliation.json" \
  '{schema:"openevo.webshop-h0.1-success-discovery-launch.v1",status:"pass",scientific_result:true,formal_task_consumption_allowed:false,gpu_count:8,started_at_utc:$started,ended_at_utc:$ended,manifest:$manifest,execution_git_sha:$execution_sha,image_digest:$image_digest,wandb_mode:$mode,reconciliation:$reconciliation}' \
  > "${HOST_ROOT}/evidence/h0.1-launch-receipt.json"
chown 1001:1001 "${HOST_ROOT}/evidence/h0.1-launch-receipt.json"
printf 'H0.1 success discovery completed: %s\n' "${HOST_ROOT}"
