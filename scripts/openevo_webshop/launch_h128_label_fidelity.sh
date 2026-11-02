#!/usr/bin/env bash
set -euo pipefail

# Root is only the Docker control plane. Every scientific process below runs
# as the immutable experiment UID/GID 1001:1001 in a one-GPU sibling container.

CAMPAIGN_ID="20260816-2130-h128-label-fidelity"
RUN_NAME="phase-h1.28-action-label-fidelity-v1-20260816"
EXECUTION_SHA="${OPENEVO_H128_EXECUTION_SHA:?set OPENEVO_H128_EXECUTION_SHA to the launch-gate commit}"
HOST_REPO="${OPENEVO_H128_HOST_REPO:?set OPENEVO_H128_HOST_REPO to the exact standalone server clone}"
HOST_RUNS="${OPENEVO_H128_HOST_RUNS:-/workspace/runs}"
ROOT="/workspace/runs/${RUN_NAME}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
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
TASK_MANIFEST="/workspace/repo/docs/evidence/remote-runs/2026-08-16/h1.28/manifests/h1.28-task-manifest.json"
DATASET_MANIFEST="${ROOT}/datasets/oracle-action-v1/manifest.json"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
TEACHER_ADAPTER="/workspace/runs/phase-h1.18-reachability-filtered-recovery-diagnostic-v1-20260815/artifacts/h1.18-success-state-fit-online-gpu0-v2/workers/sd-lora-51af9d6482c175e4834cbda7fa49bd79/adapter"

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
TASKS=(680 688 1101 2532 4819 4924 5469 6228)
SEEDS=(2026085200 2026085201 2026085202 2026085203)

die() { printf 'H1.28 launch gate failed: %s\n' "$*" >&2; exit 2; }

[[ "$(hostname)" == "experiment-host" ]] || die "control plane hostname is not experiment-host"
[[ "$(id -u)" == "0" ]] || die "control plane is not root"
[[ -n "${WANDB_API_KEY:-}" ]] || die "WANDB_API_KEY is not present in the ephemeral launcher environment"
[[ "${HOST_REPO}" == /workspace/* ]] || die "HOST_REPO must use the Docker-daemon-visible /workspace path"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/run_parametric_smoke_tracked.py" ]] || die "HOST_REPO does not expose the experiment scripts to the Docker daemon"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "server clone SHA mismatch"
jq -e '.phase == "H1.28" and .formal_task_consumption_allowed == true and .status == "launch-authorized"' "${HOST_REPO}/configs/experiment/current-campaign.json" >/dev/null || die "current campaign is not launch-authorized"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "container image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo core SHA mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" diff -- src/openevo/evolution/parametric/sd_lora_trainer.py src/openevo/evolution/parametric/trainer_service.py | sha256sum | awk '{print $1}')" == "${CORE_PATCH_SHA}" ]] || die "OpenEvo runtime patch digest mismatch"
[[ "$(/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python -m pip freeze --all | LC_ALL=C sort | sha256sum | awk '{print $1}')" == "${PYTHON_FREEZE_SHA}" ]] || die "Python freeze digest mismatch"
[[ "$(sha256sum "${HOST_ROOT}/datasets/oracle-action-v1/records.jsonl" | awk '{print $1}')" == "448c86a09306e8659fe9bdd77007cae27be3d24f3d0b7ff0917e708b82d63e47" ]] || die "oracle-action records hash mismatch"
[[ "$(sha256sum "${HOST_ROOT}/datasets/oracle-action-v1/manifest.json" | awk '{print $1}')" == "701789e9c1309f4d343f522e50a6c147d0183830d057ea39f65cfd8ee8b83609" ]] || die "oracle-action manifest hash mismatch"
[[ "$(stat -c '%a' "${HOST_ROOT}/datasets/oracle-action-v1/manifest.json")" == "600" ]] || die "oracle-action manifest mode mismatch"
[[ "$(stat -c '%a' "${HOST_ROOT}/datasets/oracle-action-v1/records.jsonl")" == "600" ]] || die "oracle-action records mode mismatch"
[[ "$(sha256sum "${HOST_REPO}/docs/evidence/remote-runs/2026-08-16/h1.28/manifests/h1.28-task-manifest.json" | awk '{print $1}')" == "bb380de563289614f5969f5e45c97dc5dc2561490e1575db755b82b63d8fe6a0" ]] || die "task manifest file hash mismatch"
[[ "$(sha256sum "${HOST_RUNS}/phase-h1.18-reachability-filtered-recovery-diagnostic-v1-20260815/artifacts/h1.18-success-state-fit-online-gpu0-v2/workers/sd-lora-51af9d6482c175e4834cbda7fa49bd79/adapter/adapter_model.safetensors" | awk '{print $1}')" == "8672ad93d860fff3dacc1a987811deb5b65426d65b854e3b9b41b3a1ce69afa2" ]] || die "teacher adapter model hash mismatch"
[[ "$(sha256sum "${HOST_RUNS}/phase-h1.18-reachability-filtered-recovery-diagnostic-v1-20260815/artifacts/h1.18-success-state-fit-online-gpu0-v2/workers/sd-lora-51af9d6482c175e4834cbda7fa49bd79/adapter/openevo_sd_lora_state.safetensors" | awk '{print $1}')" == "c7ede1f5143d953632a026e2dec3b0676e0a79b42d46282080a4f7a88c35de35" ]] || die "teacher adapter state hash mismatch"
[[ -d "${HF_CACHE_HOST}/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/${MODEL_REVISION}" ]] || die "pinned training model snapshot is absent from the read-only HF cache"

gpu_gate() {
  local uuid row memory utilization pstate apps
  for uuid in "${GPU_UUIDS[@]}"; do
    row="$(nvidia-smi -i "${uuid}" --query-gpu=memory.used,utilization.gpu,pstate --format=csv,noheader,nounits)"
    IFS=',' read -r memory utilization pstate <<<"${row}"
    memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"; pstate="${pstate//[[:space:]]/}"
    [[ "${memory}" =~ ^[0-9]+$ && "${memory}" -le 10 ]] || die "${uuid} memory is ${memory} MiB"
    [[ "${utilization}" == "0" ]] || die "${uuid} utilization is ${utilization}%"
    apps="$(nvidia-smi -i "${uuid}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
    [[ -z "${apps}" ]] || die "${uuid} has a compute process"
  done
}

run_gpu_container() {
  local uuid="$1" name="$2" out="$3"; shift 3
  local host_out="${HOST_RUNS}/${out#/workspace/runs/}"
  mkdir -p "${host_out}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts,hf}
  chown -R 1001:1001 "${host_out}"
  docker run --rm --name "${name}" --read-only --tmpfs /tmp:rw,nosuid,size=4g --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 \
    --gpus "device=${uuid}" --user 1001:1001 \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
    -v "${HOST_REPO}":/workspace/repo:ro \
    -v "${CORE_HOST}":/workspace/openevo-core:ro \
    -v /workspace/models:/workspace/models:ro \
    -v /workspace/models:/workspace/workspace/models:ro \
    -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro \
    -v "${HOST_RUNS}":/workspace/runs \
    -w /workspace/repo \
    -e WANDB_API_KEY -e WANDB_ENTITY=example-entity -e WANDB_PROJECT=openevo-webshop-experiments \
    -e WANDB_MODE=online -e OPENEVO_WANDB_MODE=online -e OPENEVO_CAMPAIGN_ID="${CAMPAIGN_ID}" \
    -e CUDA_VISIBLE_DEVICES="${uuid}" -e TOKENIZERS_PARALLELISM=false -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
    -e JAVA_HOME=/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1 \
    -e PYTHONPATH=/workspace/openevo-core/src:/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e HOME="${out}/home" -e TMPDIR="${out}/tmp" -e XDG_CACHE_HOME="${out}/xdg-cache" -e HF_HOME=/workspace/hf-cache \
    -e WANDB_DIR="${out}/wandb" -e WANDB_CACHE_DIR="${out}/wandb-cache" -e WANDB_CONFIG_DIR="${out}/wandb-config" -e WANDB_ARTIFACT_DIR="${out}/wandb-artifacts" \
    --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}

gpu_gate
TRAIN_OUT="${ROOT}/artifacts/h1.28-oracle-action-fit-online-gpu0-v9"
run_gpu_container "${GPU_UUIDS[0]}" h128-oracle-train-v9 "${TRAIN_OUT}" \
  -m scripts.openevo_webshop.run_parametric_smoke_tracked \
  --dataset-manifest "${DATASET_MANIFEST}" \
  --artifact-root "${TRAIN_OUT}" --evidence "${ROOT}/evidence/h1.28-oracle-action-training.json" \
  --run-id h1.28-oracle-action-fit-online-gpu0-v9 --benchmark webshop --python "${PYTHON}" \
  --base-model "${TRAIN_MODEL_ID}" --model-revision "${MODEL_REVISION}" --max-steps 4 --max-records 96 --minimum-reward 0 \
  --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" \
  --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --gpu-uuid "${GPU_UUIDS[0]}" \
  --wandb-mode online --wandb-project openevo-webshop-experiments --wandb-entity example-entity \
  --campaign-id "${CAMPAIGN_ID}" --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${TRAIN_OUT}/wandb"

ORACLE_ADAPTER="$(jq -r '.adapter_uri' "${HOST_ROOT}/evidence/h1.28-oracle-action-training.json")"
ORACLE_ADAPTER="${ORACLE_ADAPTER#file://}"
[[ "${ORACLE_ADAPTER}" == /workspace/runs/* ]] || die "oracle adapter URI escaped the run root"

launch_wave() {
  local arm="$1" adapter="$2" pids=() index task uuid out rc=0
  gpu_gate
  for index in "${!TASKS[@]}"; do
    task="${TASKS[$index]}"; uuid="${GPU_UUIDS[$index]}"; out="${ROOT}/evaluation/${arm}-task${task}"
    run_gpu_container "${uuid}" "h128-${arm}-${task}" "${out}" \
      -m scripts.openevo_webshop.h1_closed_loop_tracked \
      --task-manifest "${TASK_MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}" \
      --adapter "${adapter}" --arm "${arm}" --tasks "${task}" --rollout-seeds "${SEEDS[@]}" \
      --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" \
      --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature 0.0 --max-steps 30 --max-new 512 \
      --prompt-contract seed3090.webshop-compact-action-prompt.v2 --wandb-mode online \
      --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" \
      --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb" &
    pids+=("$!")
  done
  for index in "${!pids[@]}"; do wait "${pids[$index]}" || rc=1; done
  [[ "${rc}" == "0" ]] || die "${arm} evaluation wave failed"
}

launch_wave base none
launch_wave teacher-action "${TEACHER_ADAPTER}"
launch_wave oracle-action "${ORACLE_ADAPTER}"

cell_args=()
for arm in base teacher-action oracle-action; do
  for task in "${TASKS[@]}"; do cell_args+=(--cell "${arm}:${task}:${ROOT}/evaluation/${arm}-task${task}"); done
done
docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=1g --user 1001:1001 \
  -v "${HOST_REPO}":/workspace/repo:ro -v "${HOST_RUNS}":/workspace/runs -w /workspace/repo \
  --entrypoint "${PYTHON}" -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
  "${IMAGE}" -m scripts.openevo_webshop.reconcile_h1_28 --manifest "${TASK_MANIFEST}" \
  "${cell_args[@]}" --output "${ROOT}/evidence/h1.28-reconciliation-v1.json"

printf 'H1.28 completed under %s\n' "${HOST_ROOT}"
