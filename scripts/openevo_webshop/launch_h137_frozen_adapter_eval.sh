#!/usr/bin/env bash
set -euo pipefail

# H1.37 frozen-adapter fresh-task paired evaluation (direction A, no retraining).
#
# Takes the ALREADY-TRAINED H1.36 self-evolution SD-LoRA adapter and its frozen
# Qwen2.5-7B base, and evaluates them paired on 64 previously untouched WebShop
# tasks. NOTHING is retrained and no model/adapter/parser/inference variable
# changes: the only independent variable is the fresh evaluation sample.
#
# Sealing discipline: the 64-task manifest is generated and hashed BEFORE any
# evaluation container launches, and its sha256 is baked into every attempt's
# protocol_fingerprint (via h1_closed_loop_tracked) so a post-hoc task swap is
# detectable. The frozen adapter content sha256 is pinned fail-closed at launch.
#
# Docker root is only the control plane. Scientific Python and model processes
# run as UID/GID 1001:1001 in one UUID-bound child container per GPU.

CAMPAIGN_ID="20260818-0000-h137-frozen-adapter-fresh-task-eval"
RUN_NAME="phase-h1.37-frozen-adapter-fresh-task-eval-v1-20260818"
EXECUTION_SHA="${OPENEVO_H137_EXECUTION_SHA:?set OPENEVO_H137_EXECUTION_SHA to the launch-gate commit}"
HOST_REPO="${OPENEVO_H137_HOST_REPO:?set OPENEVO_H137_HOST_REPO to the exact standalone server clone}"
HOST_RUNS="${OPENEVO_H137_HOST_RUNS:-/workspace/runs}"
ROOT="/workspace/runs/${RUN_NAME}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
PYTHON_FREEZE_SHA="e3cb53bced07b519157ceca2698935bf0be45a58d08465c87c618ef62c31ee18"
CORE_HOST="/workspace/migration/runtime-src-snapshot/openevo-core"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
CORE_PATCH_SHA="38fd5305a23f9834abc8f022f4636d1b820e6117ee3e554c0defe03a7b5e726f"
MODEL_REVISION="0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
MODEL="/workspace/models/qwen2.5-7b-instruct"
MODEL_HOST_DIR="/workspace/models/qwen2.5-7b-instruct"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
TASK_MANIFEST="${ROOT}/manifests/h1.37-task-manifest.json"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"

# Frozen inference contract (identical to the validated H1.36 v3 re-run). The
# action parser is fixed to the v3 action-command contract; it is NOT a variable.
EVAL_ACTION_CONTRACT="seed3090.webshop-action-parser.v3-action-command"
EVAL_DIR="${ROOT}/evaluation"
RECONCILE_OUT="${ROOT}/evidence/h1.37-reconciliation.json"
RECONCILE_SCHEMA="openevo.webshop-h1.37-reconciliation.v1"
ANALYSIS_OUT="${ROOT}/evidence/h1.37-analysis.json"

# Frozen self-evolution adapter trained by H1.36 (reused, never retrained).
H136_RUN_NAME="phase-h1.36-strong-base-self-evolution-v1-20260818"
ADAPTER_WORKER="sd-lora-b7e36404ca6320620e3f86e5225699e3"
ADAPTER_CONTAINER="/workspace/runs/${H136_RUN_NAME}/artifacts/h1.36-self-evolution-fit-gpu0-v1/workers/${ADAPTER_WORKER}/adapter"
ADAPTER_HOST="${HOST_RUNS}/${H136_RUN_NAME}/artifacts/h1.36-self-evolution-fit-gpu0-v1/workers/${ADAPTER_WORKER}/adapter"

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

die() { printf 'H1.37 launch gate failed: %s\n' "$*" >&2; exit 2; }

[[ "$(hostname)" == "experiment-host" ]] || die "control plane hostname is not experiment-host"
[[ "$(id -u)" == "0" ]] || die "control plane is not root"
[[ "${HOST_REPO}" == /workspace/* ]] || die "HOST_REPO must use the Docker-daemon-visible /workspace path"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/h1_closed_loop_tracked.py" ]] || die "HOST_REPO does not expose the eval entrypoint"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/build_h1_37_manifest.py" ]] || die "HOST_REPO does not expose the H1.37 manifest builder"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/reconcile_h1_37.py" ]] || die "HOST_REPO does not expose the H1.37 reconcile script"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/h1_37_analysis.py" ]] || die "HOST_REPO does not expose the H1.37 analysis script"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "server clone SHA mismatch"
jq -e '.phase == "H1.37" and .formal_task_consumption_allowed == true and .status == "launch-authorized"' "${HOST_REPO}/configs/experiment/current-campaign.json" >/dev/null || die "current campaign is not launch-authorized H1.37"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "container image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo core SHA mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" diff -- src/openevo/evolution/parametric/sd_lora_trainer.py src/openevo/evolution/parametric/trainer_service.py | sha256sum | awk '{print $1}')" == "${CORE_PATCH_SHA}" ]] || die "OpenEvo runtime patch digest mismatch"
[[ "$("${PYTHON}" -m pip freeze --all | LC_ALL=C sort | sha256sum | awk '{print $1}')" == "${PYTHON_FREEZE_SHA}" ]] || die "Python freeze digest mismatch"

# 7B local-model content pin gate (same frozen checkpoint as H1.36).
[[ -d "${MODEL_HOST_DIR}" ]] || die "7B local model dir is absent"
for shard in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  [[ -f "${MODEL_HOST_DIR}/${shard}" ]] || die "7B weight shard ${shard} is absent"
done
[[ -f "${MODEL_HOST_DIR}/config.json" && -f "${MODEL_HOST_DIR}/model.safetensors.index.json" && -f "${MODEL_HOST_DIR}/tokenizer_config.json" && -f "${MODEL_HOST_DIR}/tokenizer.json" ]] || die "7B config/tokenizer files are absent"
_cfg_sha="$(sha256sum "${MODEL_HOST_DIR}/config.json" | awk '{print $1}')"
_idx_sha="$(sha256sum "${MODEL_HOST_DIR}/model.safetensors.index.json" | awk '{print $1}')"
_tok_sha="$(sha256sum "${MODEL_HOST_DIR}/tokenizer_config.json" | awk '{print $1}')"
[[ "$(printf '%s\n%s\n%s\n' "${_cfg_sha}" "${_idx_sha}" "${_tok_sha}" | sha256sum | awk '{print $1}')" == "${MODEL_REVISION}" ]] || die "7B content pin mismatch"

# Frozen H1.36 adapter pin: the directory must exist and be non-empty, and its
# content sha256 is recorded fail-closed before any evaluation.
[[ -d "${ADAPTER_HOST}" ]] || die "frozen H1.36 adapter dir is absent: ${ADAPTER_HOST}"
mapfile -t ADAPTER_FILES < <(find "${ADAPTER_HOST}" -type f | LC_ALL=C sort)
[[ "${#ADAPTER_FILES[@]}" -gt 0 ]] || die "frozen H1.36 adapter dir is empty"
ADAPTER_SHA256="$(printf '%s\n' "${ADAPTER_FILES[@]}" | xargs sha256sum | sha256sum | awk '{print $1}')"
[[ "${ADAPTER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "frozen adapter content sha256 could not be computed"

mkdir -p "${HOST_ROOT}"/{manifests,evaluation,evidence}
chown -R 1001:1001 "${HOST_ROOT}"

probe_wandb() {
  local probe_output probe_rc=0
  set +e
  probe_output="$(timeout 45s docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=512m --shm-size=1g \
    -e WANDB_MODE=online -e WANDB_INIT_TIMEOUT=10 -e WANDB_SILENT=true -e WANDB_API_KEY \
    "${IMAGE}" python3 -c 'import json, wandb
try:
    run = wandb.init(project="openevo-webshop-experiments", entity="example-entity", name="h137-online-probe", job_type="preflight", mode="online", reinit="create_new")
    print(json.dumps({"status": "ok", "run_id": getattr(run, "id", None), "run_url": getattr(run, "url", None)}))
    run.finish()
except BaseException as exc:
    print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
    raise' 2>&1)" || probe_rc=$?
  jq -n --arg status "${probe_rc}" --arg output "${probe_output}" --arg requested_mode "${TRACKING_MODE}" \
    '{schema:"openevo.wandb-online-probe.v1", requested_mode:$requested_mode, exit_code:($status|tonumber), output:$output}' \
    > "${HOST_ROOT}/evidence/wandb-online-probe.json"
  chown 1001:1001 "${HOST_ROOT}/evidence/wandb-online-probe.json"
  if [[ "${probe_rc}" == "0" && "${probe_output}" == *'"status": "ok"'* ]]; then
    TRACKING_MODE="online"
  else
    TRACKING_MODE="offline"
    printf 'W&B online initialization failed; using offline only for this run. See %s\n' "${HOST_ROOT}/evidence/wandb-online-probe.json" >&2
  fi
}

if [[ "${TRACKING_MODE}" == "online" ]]; then
  probe_wandb
elif [[ "${TRACKING_MODE}" != "offline" ]]; then
  die "unsupported tracking mode ${TRACKING_MODE}; use online or offline"
fi

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
    -e JAVA_HOME=/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1 \
    -e PATH=/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}

run_gpu_container() {
  local uuid="$1" name="$2" out="$3"; shift 3
  local host_out="${HOST_RUNS}/${out#/workspace/runs/}"
  mkdir -p "${host_out}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts,hf}
  chown -R 1001:1001 "${host_out}"
  docker run --rm --name "${name}" --read-only --tmpfs /tmp:rw,nosuid,size=4g \
    --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 --gpus "device=${uuid}" --user 1001:1001 \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -v "${HOST_REPO}":/workspace/repo:ro \
    -v "${CORE_HOST}":/workspace/openevo-core:ro -v /workspace/models:/workspace/models:ro \
    -v /workspace/models:/workspace/workspace/models:ro -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro \
    -v "${HOST_RUNS}":/workspace/runs -w /workspace/repo \
    -e WANDB_API_KEY -e WANDB_ENTITY=example-entity -e WANDB_PROJECT=openevo-webshop-experiments \
    -e WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_CAMPAIGN_ID="${CAMPAIGN_ID}" \
    -e CUDA_VISIBLE_DEVICES="${uuid}" -e TOKENIZERS_PARALLELISM=false -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
    -e JAVA_HOME=/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1 \
    -e PYTHONPATH=/workspace/openevo-core/src:/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src -e PYTHONDONTWRITEBYTECODE=1 \
    -e HOME="${out}/home" -e TMPDIR="${out}/tmp" -e XDG_CACHE_HOME="${out}/xdg-cache" -e HF_HOME=/workspace/hf-cache \
    -e WANDB_DIR="${out}/wandb" -e WANDB_CACHE_DIR="${out}/wandb-cache" -e WANDB_CONFIG_DIR="${out}/wandb-config" -e WANDB_ARTIFACT_DIR="${out}/wandb-artifacts" \
    --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}

# --- Seal the 64-task fresh manifest BEFORE any evaluation. ---
# Idempotent: if a manifest already exists it is reused and its sha recorded, but
# the preregistered selection (seed 20260818, 229-task exclusion) makes any
# post-hoc change detectable via the per-attempt protocol_fingerprint.
if [[ -f "${HOST_ROOT}/manifests/h1.37-task-manifest.json" ]]; then
  printf 'H1.37: reusing existing sealed manifest at %s\n' "${TASK_MANIFEST}" >&2
else
  GENERATED_AT_UTC="$(date -u +%FT%TZ)"
  run_utility_container -m scripts.openevo_webshop.build_h1_37_manifest --seed-root "${SEED_ROOT}" \
    --experiment-sha "${EXECUTION_SHA}" --generated-at-utc "${GENERATED_AT_UTC}" --output "${TASK_MANIFEST}"
fi
chown -R 1001:1001 "${HOST_ROOT}/manifests" || true

mapfile -t EVAL_TASKS < <(jq -r '.evaluation_tasks[]' "${HOST_ROOT}/manifests/h1.37-task-manifest.json")
[[ "${#EVAL_TASKS[@]}" == "64" ]] || die "manifest evaluation task count is not 64"
MANIFEST_FILE_SHA="$(sha256sum "${HOST_ROOT}/manifests/h1.37-task-manifest.json" | awk '{print $1}')"
jq -e '.schema == "openevo.webshop-h1.37-frozen-adapter-fresh-task-eval-manifest.v1" and .source.experiment_git_sha == $sha and (.evaluation_tasks | length == 64) and ((.evaluation_tasks | length) == (.evaluation_tasks | unique | length)) and (.contamination.no_retraining == true) and (.contamination.frozen_adapter_reused == true) and (.contamination.evaluation_tasks_disjoint_from_all_prior == true)' \
  --arg sha "${EXECUTION_SHA}" "${HOST_ROOT}/manifests/h1.37-task-manifest.json" >/dev/null || die "manifest identity / contamination gate failed"

# Seal receipt: recorded before any evaluation container launches.
jq -n --arg manifest "${TASK_MANIFEST}" --arg manifest_file_sha256 "${MANIFEST_FILE_SHA}" \
  --arg experiment_sha "${EXECUTION_SHA}" --arg sealed_at_utc "$(date -u +%FT%TZ)" \
  --arg adapter_container "${ADAPTER_CONTAINER}" --arg adapter_sha256 "${ADAPTER_SHA256}" \
  --argjson task_count "${#EVAL_TASKS[@]}" \
  '{schema:"openevo.webshop-h1.37-manifest-seal.v1", manifest:$manifest, manifest_file_sha256:$manifest_file_sha256, experiment_git_sha:$experiment_sha, sealed_at_utc:$sealed_at_utc, evaluation_task_count:$task_count, frozen_adapter:{container_path:$adapter_container, content_sha256:$adapter_sha256}}' \
  > "${HOST_ROOT}/evidence/h1.37-manifest-seal.json"
chown 1001:1001 "${HOST_ROOT}/evidence/h1.37-manifest-seal.json"
printf 'H1.37 manifest sealed: %s tasks, file sha %s, adapter sha %s\n' "${#EVAL_TASKS[@]}" "${MANIFEST_FILE_SHA}" "${ADAPTER_SHA256}" >&2

# --- Paired evaluation: 64 tasks x 4 seeds x 2 arms, waved 8 GPUs at a time. ---
launch_wave() {
  local arm="$1" adapter="$2"
  local pids=() index task uuid out rc=0 wave
  local n="${#EVAL_TASKS[@]}"
  local ngpu="${#GPU_UUIDS[@]}"
  for (( wave=0; wave*ngpu < n; wave++ )); do
    gpu_gate
    pids=()
    for (( index=wave*ngpu; index < (wave+1)*ngpu && index < n; index++ )); do
      task="${EVAL_TASKS[$index]}"; uuid="${GPU_UUIDS[$((index % ngpu))]}"; out="${EVAL_DIR}/${arm}-task${task}"
      run_gpu_container "${uuid}" "h137-${arm}-${task}" "${out}" -m scripts.openevo_webshop.h1_closed_loop_tracked \
        --task-manifest "${TASK_MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}" \
        --adapter "${adapter}" --arm "${arm}" --tasks "${task}" --rollout-seeds 2026090200 2026090201 2026090202 2026090203 \
        --action-contract "${EVAL_ACTION_CONTRACT}" \
        --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" \
        --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature 0.0 --max-steps 30 --max-new 512 \
        --prompt-contract seed3090.webshop-compact-action-prompt.v2 --wandb-mode "${TRACKING_MODE}" \
        --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" \
        --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
    [[ "${rc}" == "0" ]] || die "${arm} evaluation wave ${wave} failed"
  done
}

launch_wave base none
launch_wave self-evolution "${ADAPTER_CONTAINER}"

cell_args=()
for arm in base self-evolution; do
  for task in "${EVAL_TASKS[@]}"; do cell_args+=(--cell "${arm}:${task}:${EVAL_DIR}/${arm}-task${task}"); done
done
run_utility_container -m scripts.openevo_webshop.reconcile_h1_37 --manifest "${TASK_MANIFEST}" "${cell_args[@]}" \
  --experiment "H1.37-frozen-adapter-fresh-task-eval-v1" --schema "${RECONCILE_SCHEMA}" \
  --output "${RECONCILE_OUT}"
run_utility_container -m scripts.openevo_webshop.h1_37_analysis --reconcile "${RECONCILE_OUT}" --output "${ANALYSIS_OUT}"

if [[ "${TRACKING_MODE}" == "offline" ]]; then
  spool_hash="$(find "${HOST_ROOT}" -type f -path '*/wandb/*' -print0 | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | awk '{print $1}')"
  jq -n --arg probe "${HOST_ROOT}/evidence/wandb-online-probe.json" --arg spool_hash "${spool_hash}" \
    '{schema:"openevo.wandb-fallback-receipt.v1", mode:"offline", reason:"online initialization failed; see probe", online_probe:$probe, spool_hash:$spool_hash}' \
    > "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
  chown 1001:1001 "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
fi

printf 'H1.37 completed under %s\n' "${HOST_ROOT}"
