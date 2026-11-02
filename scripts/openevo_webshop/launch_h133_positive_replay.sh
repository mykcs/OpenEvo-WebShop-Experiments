#!/usr/bin/env bash
set -euo pipefail

# H1.33 is a non-formal pure self-evolution diagnostic. It replays only
# qualified native-Qwen successes from H0.1, then compares BASE and one
# bounded OpenEvo SD-LoRA adapter on fresh tasks.

CAMPAIGN_ID="20260817-0000-h1.33-positive-replay-transfer-v1"
RUN_NAME="phase-h1.33-positive-replay-transfer-v1-20260817"
EXECUTION_SHA="${OPENEVO_H133_EXECUTION_SHA:?set OPENEVO_H133_EXECUTION_SHA}"
HOST_REPO="${OPENEVO_H133_HOST_REPO:?set OPENEVO_H133_HOST_REPO}"
HOST_RUNS="${OPENEVO_H133_HOST_RUNS:-/workspace/runs}"
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
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
H0_ROOT="/workspace/runs/phase-h0.1-qwen-success-discovery-v2-20260817"
H0_RECON="${H0_ROOT}/evidence/h0.1-reconciliation.json"
H0_HOST_ROOT="${HOST_RUNS}/phase-h0.1-qwen-success-discovery-v2-20260817"
H0_RECON_HOST="${H0_HOST_ROOT}/evidence/h0.1-reconciliation.json"
H0_RECON_SHA="e3b83f03bece66868293f1f2bda5a4a9afe366f0e497e6eeff193338814e52f4"
TASK_MANIFEST="${ROOT}/manifests/h1.33-positive-replay-manifest.json"
DATASET_MANIFEST="${ROOT}/datasets/h0.1-positive-replay-v1/manifest.json"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"
RESUME_AFTER_BASE="${OPENEVO_H133_RESUME_AFTER_BASE:-0}"

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

die() { printf 'H1.33 launch gate failed: %s\n' "$*" >&2; exit 2; }

[[ "$(hostname)" == "experiment-host" ]] || die "hostname is not experiment-host"
[[ "$(id -u)" == "0" ]] || die "control plane is not root"
[[ "${HOST_REPO}" == /workspace/* ]] || die "HOST_REPO is not daemon-visible"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/h1_closed_loop_tracked.py" ]] || die "incomplete checkout"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "server SHA mismatch"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo SHA mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" diff -- src/openevo/evolution/parametric/sd_lora_trainer.py src/openevo/evolution/parametric/trainer_service.py | sha256sum | awk '{print $1}')" == "${CORE_PATCH_SHA}" ]] || die "OpenEvo patch mismatch"
[[ "$("${PYTHON}" -m pip freeze --all | LC_ALL=C sort | sha256sum | awk '{print $1}')" == "${PYTHON_FREEZE_SHA}" ]] || die "runtime freeze mismatch"
[[ -f "${H0_RECON_HOST}" ]] || die "H0.1 reconciliation absent"
[[ "$(sha256sum "${H0_RECON_HOST}" | awk '{print $1}')" == "${H0_RECON_SHA}" ]] || die "H0.1 reconciliation hash mismatch"
jq -e '.status == "pass" and .expected_attempts == 128 and .observed_attempts == 128 and .formal_task_consumption_allowed == false and .qualified_positive_count >= 16' "${H0_RECON_HOST}" >/dev/null || die "H0.1 source gate failed"

mkdir -p "${HOST_ROOT}"/{manifests,datasets,evaluation,artifacts,evidence}
chown -R 1001:1001 "${HOST_ROOT}"

gpu_gate() {
  local uuid row memory utilization apps attempt all_idle
  for attempt in $(seq 1 30); do
    all_idle=1
    for uuid in "${GPU_UUIDS[@]}"; do
      row="$(nvidia-smi -i "${uuid}" --query-gpu=memory.used,utilization.gpu,pstate --format=csv,noheader,nounits)"
      IFS=',' read -r memory utilization _ <<<"${row}"
      memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
      [[ "${memory}" =~ ^[0-9]+$ && "${memory}" -le 10 ]] || die "${uuid} memory=${memory} MiB"
      apps="$(nvidia-smi -i "${uuid}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
      [[ -z "${apps}" ]] || die "${uuid} has a compute process"
      [[ "${utilization}" == "0" ]] || all_idle=0
    done
    [[ "${all_idle}" == "1" ]] && return 0
    sleep 1
  done
  die "GPU utilization did not settle to zero within 30 seconds"
}

run_utility_container() {
  docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=2g --shm-size=2g --pids-limit 1024 --user 1001:1001 \
    -v "${HOST_REPO}":/workspace/repo:ro -v "${HOST_RUNS}":/workspace/runs \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -w /workspace/repo \
    -e JAVA_HOME=/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1 \
    -e PATH=/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e PYTHONDONTWRITEBYTECODE=1 --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}

run_gpu_container() {
  local uuid="$1" name="$2" out="$3"; shift 3
  local host_out="${HOST_RUNS}/${out#/workspace/runs/}"
  mkdir -p "${host_out}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts,hf}
  chown -R 1001:1001 "${host_out}"
  docker run --rm --name "${name}" --read-only --tmpfs /tmp:rw,nosuid,size=4g \
    --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 \
    --gpus "device=${uuid}" --user 1001:1001 \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
    -v "${HOST_REPO}":/workspace/repo:ro -v "${CORE_HOST}":/workspace/openevo-core:ro \
    -v /workspace/models:/workspace/models:ro -v /workspace/models:/workspace/workspace/models:ro \
    -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro -v "${HOST_RUNS}":/workspace/runs -w /workspace/repo \
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
mapfile -t TRAIN_IDS < <(jq -r '.qualified_positive_task_ids[] | split("_")[1] | tonumber' "${H0_RECON_HOST}")
[[ "${#TRAIN_IDS[@]}" -ge 8 ]] || die "fewer than eight H0.1 positive task identities"
MANIFEST_ARGS=()
for task in "${TRAIN_IDS[@]}"; do MANIFEST_ARGS+=(--train-task-id "${task}"); done
run_utility_container -m scripts.openevo_webshop.build_h1_33_manifest --seed-root "${SEED_ROOT}" \
  --prior-json-root /workspace/repo/configs/experiment --prior-json-root /workspace/repo/docs \
  --experiment-sha "${EXECUTION_SHA}" --h0-reconciliation-sha256 "${H0_RECON_SHA}" \
  "${MANIFEST_ARGS[@]}" --output "${TASK_MANIFEST}"
mapfile -t EVAL_IDS < <(jq -r '.evaluation_tasks[]' "${HOST_ROOT}/manifests/h1.33-positive-replay-manifest.json")
[[ "${#EVAL_IDS[@]}" == 8 ]] || die "H1.33 evaluation task count is not eight"

DATA_ARGS=()
harvest_cell_root() {
  local task="$1" host_attempts host_cell relative
  mapfile -t host_attempts < <(find "${H0_HOST_ROOT}/harvest" -type f -path "*/task${task}/attempts.jsonl" -print)
  [[ "${#host_attempts[@]}" == 1 ]] || die "expected one H0.1 harvest cell for task ${task}, found ${#host_attempts[@]}"
  host_cell="${host_attempts[0]%/attempts.jsonl}"
  relative="${host_cell#"${H0_HOST_ROOT}"}"
  [[ "${relative}" != "${host_cell}" ]] || die "H0.1 harvest cell escaped source root for task ${task}"
  printf '%s%s\n' "${H0_ROOT}" "${relative}"
}

for task in "${TRAIN_IDS[@]}"; do
  DATA_ARGS+=(--train-task-id "webshop_$(printf '%06d' "${task}")" --cell-root "$(harvest_cell_root "${task}")")
done
for task in "${EVAL_IDS[@]}"; do DATA_ARGS+=(--evaluation-task-id "webshop_$(printf '%06d' "${task}")"); done
run_utility_container -m scripts.openevo_webshop.build_self_evolution_dataset "${DATA_ARGS[@]}" \
  --minimum-success-task-count 8 --minimum-success-episode-count 16 --output-dir "${ROOT}/datasets/h0.1-positive-replay-v1"

TRAIN_OUT="${ROOT}/artifacts/h1.33-positive-replay-fit-gpu0-v1"
if [[ "${RESUME_AFTER_BASE}" == "1" ]]; then
  [[ -f "${HOST_ROOT}/evidence/h1.33-training.json" ]] || die "resume requested but training evidence is absent"
else
  gpu_gate
  run_gpu_container "${GPU_UUIDS[0]}" h133-positive-replay-train-gpu0 "${TRAIN_OUT}" \
  -m scripts.openevo_webshop.run_parametric_smoke_tracked --dataset-manifest "${DATASET_MANIFEST}" \
  --artifact-root "${TRAIN_OUT}" --evidence "${ROOT}/evidence/h1.33-training.json" \
  --run-id h1.33-positive-replay-fit-gpu0-v1 --benchmark webshop --python /workspace/repo/scripts/openevo_webshop/server_trainer_python.sh --base-model "${TRAIN_MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" --max-steps 8 --max-records 96 --minimum-reward 0 \
  --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" --runtime-fingerprint "${PYTHON_FREEZE_SHA}" \
  --image-digest "${IMAGE_DIGEST}" --gpu-uuid "${GPU_UUIDS[0]}" --wandb-mode "${TRACKING_MODE}" \
  --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" \
  --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${TRAIN_OUT}/wandb"
fi

SELF_ADAPTER="$(jq -r '.adapter_uri' "${HOST_ROOT}/evidence/h1.33-training.json")"
SELF_ADAPTER="${SELF_ADAPTER#file://}"
[[ "${SELF_ADAPTER}" == /workspace/runs/* && -d "${HOST_RUNS}/${SELF_ADAPTER#/workspace/runs/}" ]] || die "adapter URI escaped H1.33 root"

launch_wave() {
  local arm="$1" adapter="$2"; local pids=(); local index task uuid out rc=0
  gpu_gate
  for index in "${!EVAL_IDS[@]}"; do
    task="${EVAL_IDS[$index]}"; uuid="${GPU_UUIDS[$index]}"; out="${ROOT}/evaluation/${arm}-task${task}"
    run_gpu_container "${uuid}" "h133-${arm}-${task}" "${out}" -m scripts.openevo_webshop.h1_closed_loop_tracked \
      --task-manifest "${TASK_MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}" \
      --adapter "${adapter}" --arm "${arm}" --tasks "${task}" --rollout-seeds 2026081800 2026081801 2026081802 2026081803 \
      --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" \
      --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature 0.0 --max-steps 30 --max-new 512 \
      --prompt-contract seed3090.webshop-compact-action-prompt.v2 --wandb-mode "${TRACKING_MODE}" \
      --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" \
      --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
  [[ "${rc}" == 0 ]] || die "${arm} evaluation wave failed"
}

if [[ "${RESUME_AFTER_BASE}" != "1" ]]; then
  launch_wave base none
else
  for task in "${EVAL_IDS[@]}"; do
    base_cell="${HOST_RUNS}/${RUN_NAME}/evaluation/base-task${task}/attempts.jsonl"
    [[ -f "${base_cell}" && "$(wc -l < "${base_cell}")" == "4" ]] || die "resume base cell incomplete for task ${task}"
  done
fi
launch_wave self-evolution "${SELF_ADAPTER}"

CELL_ARGS=()
for arm in base self-evolution; do
  for task in "${EVAL_IDS[@]}"; do
    CELL_ARGS+=(--cell "${arm}:${task}:${ROOT}/evaluation/${arm}-task${task}")
  done
done
run_utility_container -m scripts.openevo_webshop.reconcile_h133_positive_replay \
  --manifest "${TASK_MANIFEST}" "${CELL_ARGS[@]}" --output "${ROOT}/evidence/h1.33-reconciliation.json"

printf 'H1.33 completed under %s\n' "${HOST_ROOT}"
