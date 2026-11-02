#!/usr/bin/env bash
set -euo pipefail

# H1.30 train-only capability gate.  This consumes no evaluation task and does
# not authorize the OpenEvo-versus-SEED formal comparison.  Docker remains the
# control plane; every model process runs as UID/GID 1001 in one UUID-bound
# child container per GPU.

RUN_NAME="${OPENEVO_H130_RUN_NAME:-phase-h1.30-open-evo-capability-v2-20260817}"
HOST_RUNS="${OPENEVO_H130_HOST_RUNS:-/workspace/runs}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
ROOT="/workspace/runs/${RUN_NAME}"
HOST_REPO="${OPENEVO_H130_HOST_REPO:?set OPENEVO_H130_HOST_REPO to an exact server checkout}"
PATCHED_RUNNER="${OPENEVO_H130_PATCHED_RUNNER:?set OPENEVO_H130_PATCHED_RUNNER to the reviewed runner file}"
EXECUTION_SHA="${OPENEVO_H130_EXECUTION_SHA:?set OPENEVO_H130_EXECUTION_SHA to the experiment commit}"
HOST_TASK_MANIFEST="${OPENEVO_H130_HOST_TASK_MANIFEST:?set host training manifest path}"
HOST_EVALUATION_MANIFEST="${OPENEVO_H130_HOST_EVALUATION_MANIFEST:?set host evaluation manifest path}"
TASK_MANIFEST="${OPENEVO_H130_TASK_MANIFEST:?set container-visible training manifest path}"
EVALUATION_MANIFEST="${OPENEVO_H130_EVALUATION_MANIFEST:?set container-visible evaluation manifest path}"
CAMPAIGN_ID="${OPENEVO_H130_CAMPAIGN_ID:-20260817-0000-h130-open-evo-capability-v2}"

IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
PYTHON_FREEZE_SHA="d7220829d4fc81e232523ea3aa22adfb4b55dba95d8236605a3e6f26b3b379d4"
CORE_HOST="/workspace/migration/runtime-src-snapshot/openevo-core"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
CORE_PATCH_SHA="1370686f1d773c66c85860bd2b57faf6f6a974aedce4a88573b2c12bf3394494"
MODEL_REVISION="aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL="/workspace/models/qwen2.5-3b-instruct"
TRAIN_MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"
TEMPERATURE="${OPENEVO_H130_TEMPERATURE:-0.4}"
MAX_STEPS="${OPENEVO_H130_MAX_STEPS:-30}"
MAX_NEW="${OPENEVO_H130_MAX_NEW:-512}"
ACTION_CONTRACT="${OPENEVO_H130_ACTION_CONTRACT:-seed3090.webshop-action-parser.v1}"
PROMPT_CONTRACT="${OPENEVO_H130_PROMPT_CONTRACT:-seed3090.webshop-compact-action-prompt.v2}"

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

die() { printf 'H1.30 capability gate failed: %s\n' "$*" >&2; exit 2; }

[[ "$(hostname)" == "experiment-host" ]] || die "unexpected host"
[[ "$(id -u)" == "0" ]] || die "control plane is not root"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/h1_closed_loop_tracked.py" ]] || die "host repo lacks closed-loop runner"
[[ -f "${PATCHED_RUNNER}" ]] || die "patched runner is absent"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo core SHA mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" diff -- src/openevo/evolution/parametric/sd_lora_trainer.py src/openevo/evolution/parametric/trainer_service.py | sha256sum | awk '{print $1}')" == "${CORE_PATCH_SHA}" ]] || die "OpenEvo patch digest mismatch"
[[ "$(sha256sum "${PATCHED_RUNNER}" | awk '{print $1}')" != "" ]] || die "runner digest unavailable"
[[ "${ACTION_CONTRACT}" == "seed3090.webshop-action-parser.v1" || "${ACTION_CONTRACT}" == "seed3090.webshop-action-parser.v2-missing-close" ]] || die "unsupported action contract"
[[ "${PROMPT_CONTRACT}" == "seed3090.webshop-compact-action-prompt.v2" || "${PROMPT_CONTRACT}" == "seed3090.webshop-state-aware-action-prompt.v3" ]] || die "unsupported prompt contract"

jq -e '.schema == "openevo.webshop-h1.30-training-manifest.v1" and .status == "draft-not-authorized" and .formal_task_consumption_allowed == false and (.train_tasks | length == 16) and (.validation_tasks | length == 8)' "${HOST_TASK_MANIFEST}" >/dev/null || die "training manifest gate failed"
jq -e '.schema == "openevo.webshop-h1.30-matched-manifest.v1" and .task_count == 8' "${HOST_EVALUATION_MANIFEST}" >/dev/null || die "evaluation manifest gate failed"

mkdir -p "${HOST_ROOT}"/{harvest,datasets,evidence,manifests}
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

run_utility_container() {
  docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=2g --shm-size=2g --pids-limit 1024 --user 1001:1001 \
    -v "${HOST_REPO}":/workspace/repo:ro -v "${HOST_RUNS}":/workspace/runs \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -w /workspace/repo \
    -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e PYTHONDONTWRITEBYTECODE=1 --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}

run_gpu_container() {
  local uuid="$1" name="$2" out="$3"; shift 3
  local host_out="${HOST_RUNS}/${out#/workspace/runs/}"
  mkdir -p "${host_out}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts,hf}
  chown -R 1001:1001 "${host_out}"
  docker run --rm --name "${name}" --read-only --tmpfs /tmp:rw,nosuid,size=4g \
    --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 --gpus "device=${uuid}" --user 1001:1001 \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -v "${HOST_REPO}":/workspace/repo:ro \
    -v "${PATCHED_RUNNER}":/workspace/repo/scripts/openevo_webshop/h1_closed_loop_tracked.py:ro \
    -v "${CORE_HOST}":/workspace/openevo-core:ro -v /workspace/models:/workspace/models:ro \
    -v /workspace/models:/workspace/workspace/models:ro -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro \
    -v "${HOST_RUNS}":/workspace/runs -w /workspace/repo \
    -e WANDB_API_KEY -e WANDB_ENTITY=example-entity -e WANDB_PROJECT=openevo-webshop-experiments \
    -e WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_CAMPAIGN_ID="${CAMPAIGN_ID}" \
    -e CUDA_VISIBLE_DEVICES="${uuid}" -e TOKENIZERS_PARALLELISM=false -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
    -e JAVA_HOME=/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1 \
    -e PYTHONPATH=/workspace/openevo-core/src:/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e PYTHONDONTWRITEBYTECODE=1 -e HOME="${out}/home" -e TMPDIR="${out}/tmp" -e XDG_CACHE_HOME="${out}/xdg-cache" \
    -e HF_HOME=/workspace/hf-cache -e WANDB_DIR="${out}/wandb" -e WANDB_CACHE_DIR="${out}/wandb-cache" \
    -e WANDB_CONFIG_DIR="${out}/wandb-config" -e WANDB_ARTIFACT_DIR="${out}/wandb-artifacts" \
    --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}

gpu_gate
mapfile -t TRAIN_TASKS < <(jq -r '.train_tasks[].goal_idx' "${HOST_TASK_MANIFEST}")
mapfile -t EVAL_TASKS < <(jq -r '.tasks[].goal_idx' "${HOST_EVALUATION_MANIFEST}")
[[ "${#TRAIN_TASKS[@]}" == "16" && "${#EVAL_TASKS[@]}" == "8" ]] || die "unexpected train/evaluation counts"

for wave in 0 1; do
  pids=()
  for slot in 0 1 2 3 4 5 6 7; do
    index=$((wave * 8 + slot)); task="${TRAIN_TASKS[$index]}"; uuid="${GPU_UUIDS[$slot]}"
    out="${ROOT}/harvest/wave${wave}/task${task}"
    run_gpu_container "${uuid}" "h130-capability-${wave}-${task}" "${out}" -m scripts.openevo_webshop.h1_closed_loop_tracked \
      --task-manifest "${TASK_MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}" \
      --adapter none --arm self-generated-harvest --tasks "${task}" --rollout-seeds 2026091200 2026091201 2026091202 2026091203 \
      --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" \
      --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature "${TEMPERATURE}" --max-steps "${MAX_STEPS}" --max-new "${MAX_NEW}" \
      --prompt-contract "${PROMPT_CONTRACT}" --action-contract "${ACTION_CONTRACT}" --wandb-mode "${TRACKING_MODE}" \
      --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" \
      --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb" &
    pids+=("$!")
  done
  rc=0; for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
  [[ "${rc}" == "0" ]] || die "harvest wave ${wave} failed"
  gpu_gate
done

train_args=()
for task in "${TRAIN_TASKS[@]:0:8}"; do train_args+=(--train-task-id "$(printf 'webshop_%06d' "${task}")" --cell-root "${ROOT}/harvest/wave0/task${task}"); done
for task in "${TRAIN_TASKS[@]:8}"; do train_args+=(--train-task-id "$(printf 'webshop_%06d' "${task}")" --cell-root "${ROOT}/harvest/wave1/task${task}"); done
for task in "${EVAL_TASKS[@]}"; do train_args+=(--evaluation-task-id "$(printf 'webshop_%06d' "${task}")"); done

set +e
gate_output="$(run_utility_container -m scripts.openevo_webshop.build_self_evolution_dataset "${train_args[@]}" \
  --minimum-success-task-count 2 --minimum-success-episode-count 8 --output-dir "${ROOT}/datasets/self-generated-success-v2" 2>&1)"
gate_rc=$?
set -e
printf '%s\n' "${gate_output}" > "${HOST_ROOT}/evidence/capability-gate-output.log"
chown 1001:1001 "${HOST_ROOT}/evidence/capability-gate-output.log"
status="fail"; [[ "${gate_rc}" == "0" ]] && status="pass"
jq -n --arg status "${status}" --arg output "${ROOT}/evidence/capability-gate-output.log" \
  --arg training_manifest "${TASK_MANIFEST}" --arg eval_manifest "${EVALUATION_MANIFEST}" \
  --arg execution_sha "${EXECUTION_SHA}" --argjson exit_code "${gate_rc}" \
  --arg campaign_id "${CAMPAIGN_ID}" --arg temperature "${TEMPERATURE}" \
  --arg max_steps "${MAX_STEPS}" --arg max_new "${MAX_NEW}" --arg action_contract "${ACTION_CONTRACT}" --arg prompt_contract "${PROMPT_CONTRACT}" \
  '{schema:"openevo.webshop-h1.30-open-evo-capability-gate.v1",status:$status,formal_task_consumption_allowed:false,gpu_workload_launched:true,exit_code:$exit_code,campaign_id:$campaign_id,training_manifest:$training_manifest,evaluation_manifest:$eval_manifest,execution_git_sha:$execution_sha,action_contract:$action_contract,prompt_contract:$prompt_contract,rollout_contract:{temperature:($temperature|tonumber),max_steps:($max_steps|tonumber),max_new:($max_new|tonumber),rollout_seeds:[2026091200,2026091201,2026091202,2026091203]},harvest_attempts:64,qualification:{minimum_success_task_count:2,minimum_success_episode_count:8},output_log:$output}' \
  > "${HOST_ROOT}/evidence/h1.30-open-evo-capability-gate.json"
chown 1001:1001 "${HOST_ROOT}/evidence/h1.30-open-evo-capability-gate.json"
gpu_gate
printf 'H1.30 capability gate completed: %s\n' "${status}"
