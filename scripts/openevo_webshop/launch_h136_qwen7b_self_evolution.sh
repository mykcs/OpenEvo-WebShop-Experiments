#!/usr/bin/env bash
set -euo pipefail

# H1.36 strong-base pure self-evolution (direction A): repeat the exact H1.29
# self-evolution route with the base model scaled Qwen2.5-3B -> Qwen2.5-7B and
# nothing else changed, on the SAME frozen H1.29 task manifest. No teacher data.
#
# Docker root is only the control plane. Scientific Python and model processes
# run as UID/GID 1001:1001 in one UUID-bound child container per GPU.
#
# Staged launch (OPENEVO_H136_STAGE):
#   harvest (default): run the harvest wave + the fail-closed capability gate,
#                      record the verdict, and stop before any training/eval.
#   full:              reuse an existing harvest dataset when present, require
#                      the capability gate to be open, then train + evaluate.

CAMPAIGN_ID="20260818-0000-h136-strong-base-self-evolution"
RUN_NAME="phase-h1.36-strong-base-self-evolution-v1-20260818"
EXECUTION_SHA="${OPENEVO_H136_EXECUTION_SHA:?set OPENEVO_H136_EXECUTION_SHA to the launch-gate commit}"
HOST_REPO="${OPENEVO_H136_HOST_REPO:?set OPENEVO_H136_HOST_REPO to the exact standalone server clone}"
HOST_RUNS="${OPENEVO_H136_HOST_RUNS:-/workspace/runs}"
ROOT="/workspace/runs/${RUN_NAME}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
# Runtime fingerprint pin. H1.29 pinned d7220829; the shared env was later changed
# by the deliberate doctor-openevo-327d2f5-20260817 dependency repair (added
# auxiliary serving/async/lightgbm packages; torch/transformers/peft/accelerate/
# numpy/safetensors verified byte-identical, so no scientific confound). H1.36
# therefore pins the current doctor-passed state e3cb53bc instead of rebuilding.
PYTHON_FREEZE_SHA="e3cb53bced07b519157ceca2698935bf0be45a58d08465c87c618ef62c31ee18"
CORE_HOST="/workspace/migration/runtime-src-snapshot/openevo-core"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
CORE_PATCH_SHA="38fd5305a23f9834abc8f022f4636d1b820e6117ee3e554c0defe03a7b5e726f"
# Qwen2.5-7B is a pinned read-only local directory, not an HF hub snapshot, so a
# content pin replaces the HF commit hash. MODEL_REVISION is recorded for
# provenance; transformers ignores `revision` for a local directory path.
MODEL_REVISION="0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
MODEL="/workspace/models/qwen2.5-7b-instruct"
TRAIN_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
MODEL_HOST_DIR="/workspace/models/qwen2.5-7b-instruct"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
FROZEN_H129_MANIFEST="/workspace/repo/docs/evidence/remote-runs/2026-08-16/h1.29/manifests/h1.29-task-manifest-v1.json"
TASK_MANIFEST="${ROOT}/manifests/h1.36-task-manifest.json"
DATASET_MANIFEST="${ROOT}/datasets/self-generated-success-v1/manifest.json"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"
A1_STAGE="${OPENEVO_H136_STAGE:-harvest}"

# Evaluation action-parser contract. harvest/full keep the strict v1 default so the
# original H1.36 route stays byte-identical. The eval stage pins the tolerant v3
# action-command contract (a measurement-tool fix for the 7B wrapper brittleness,
# NOT a model/hyperparameter/prompt sweep) and writes to a separate evaluation-v3
# tree + reconciliation so the original strict-v1 results are preserved verbatim for
# the before/after comparison. Same tasks, seeds, temperature, adapter: the parser
# is the ONLY changed variable.
EVAL_ACTION_CONTRACT="${OPENEVO_H136_EVAL_ACTION_CONTRACT:-seed3090.webshop-action-parser.v1}"
EVAL_DIR="${ROOT}/evaluation"
RECONCILE_OUT="${ROOT}/evidence/h1.36-reconciliation-v1.json"
RECONCILE_SCHEMA="openevo.webshop-h1.36-reconciliation.v1"
if [[ "${A1_STAGE}" == "eval" ]]; then
  EVAL_ACTION_CONTRACT="${OPENEVO_H136_EVAL_ACTION_CONTRACT:-seed3090.webshop-action-parser.v3-action-command}"
  EVAL_DIR="${ROOT}/evaluation-v3"
  RECONCILE_OUT="${ROOT}/evidence/h1.36-reconciliation-v3.json"
  RECONCILE_SCHEMA="openevo.webshop-h1.36-reconciliation.v3-action-command"
fi

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

die() { printf 'H1.36 launch gate failed: %s\n' "$*" >&2; exit 2; }

[[ "${A1_STAGE}" == "harvest" || "${A1_STAGE}" == "full" || "${A1_STAGE}" == "eval" ]] || die "OPENEVO_H136_STAGE must be harvest, full, or eval"
[[ "$(hostname)" == "experiment-host" ]] || die "control plane hostname is not experiment-host"
[[ "$(id -u)" == "0" ]] || die "control plane is not root"
[[ "${HOST_REPO}" == /workspace/* ]] || die "HOST_REPO must use the Docker-daemon-visible /workspace path"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/h1_closed_loop_tracked.py" ]] || die "HOST_REPO does not expose H1.36 scripts"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/build_h1_36_manifest.py" ]] || die "HOST_REPO does not expose the H1.36 manifest builder"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "server clone SHA mismatch"
jq -e '.phase == "H1.36" and .formal_task_consumption_allowed == true and .status == "launch-authorized"' "${HOST_REPO}/configs/experiment/current-campaign.json" >/dev/null || die "current campaign is not launch-authorized"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "container image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo core SHA mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" diff -- src/openevo/evolution/parametric/sd_lora_trainer.py src/openevo/evolution/parametric/trainer_service.py | sha256sum | awk '{print $1}')" == "${CORE_PATCH_SHA}" ]] || die "OpenEvo runtime patch digest mismatch"
[[ "$("${PYTHON}" -m pip freeze --all | LC_ALL=C sort | sha256sum | awk '{print $1}')" == "${PYTHON_FREEZE_SHA}" ]] || die "Python freeze digest mismatch"

# 7B local-model pin gate (replaces the 3B HF-hub snapshot gate).
[[ -d "${MODEL_HOST_DIR}" ]] || die "7B local model dir is absent"
for shard in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  [[ -f "${MODEL_HOST_DIR}/${shard}" ]] || die "7B weight shard ${shard} is absent"
done
[[ -f "${MODEL_HOST_DIR}/config.json" && -f "${MODEL_HOST_DIR}/model.safetensors.index.json" && -f "${MODEL_HOST_DIR}/tokenizer_config.json" && -f "${MODEL_HOST_DIR}/tokenizer.json" ]] || die "7B config/tokenizer files are absent"
_cfg_sha="$(sha256sum "${MODEL_HOST_DIR}/config.json" | awk '{print $1}')"
_idx_sha="$(sha256sum "${MODEL_HOST_DIR}/model.safetensors.index.json" | awk '{print $1}')"
_tok_sha="$(sha256sum "${MODEL_HOST_DIR}/tokenizer_config.json" | awk '{print $1}')"
[[ "$(printf '%s\n%s\n%s\n' "${_cfg_sha}" "${_idx_sha}" "${_tok_sha}" | sha256sum | awk '{print $1}')" == "${MODEL_REVISION}" ]] || die "7B content pin mismatch"

mkdir -p "${HOST_ROOT}"/{manifests,harvest,datasets,evaluation,artifacts,evidence}
chown -R 1001:1001 "${HOST_ROOT}"

probe_wandb() {
  local probe_output probe_rc=0
  set +e
  probe_output="$(timeout 45s docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=512m --shm-size=1g \
    -e WANDB_MODE=online -e WANDB_INIT_TIMEOUT=10 -e WANDB_SILENT=true -e WANDB_API_KEY \
    "${IMAGE}" python3 -c 'import json, wandb
try:
    run = wandb.init(project="openevo-webshop-experiments", entity="example-entity", name="h136-online-probe", job_type="preflight", mode="online", reinit="create_new")
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

gpu_gate
run_utility_container -m scripts.openevo_webshop.build_h1_36_manifest --seed-root "${SEED_ROOT}" \
  --frozen-manifest "${FROZEN_H129_MANIFEST}" --experiment-sha "${EXECUTION_SHA}" --output "${TASK_MANIFEST}"

mapfile -t TRAIN_TASKS < <(jq -r '.train_tasks[]' "${HOST_ROOT}/manifests/h1.36-task-manifest.json")
mapfile -t EVAL_TASKS < <(jq -r '.evaluation_tasks[]' "${HOST_ROOT}/manifests/h1.36-task-manifest.json")
[[ "${#TRAIN_TASKS[@]}" == "8" && "${#EVAL_TASKS[@]}" == "8" ]] || die "manifest task counts are not 8/8"
jq -e '.schema == "openevo.webshop-h1.36-strong-base-self-evolution-manifest.v1" and .source.experiment_git_sha == $sha and (.train_tasks | length == 8) and (.evaluation_tasks | length == 8) and (([.train_tasks[], .evaluation_tasks[]] | length) == ([.train_tasks[], .evaluation_tasks[]] | unique | length))' --arg sha "${EXECUTION_SHA}" "${HOST_ROOT}/manifests/h1.36-task-manifest.json" >/dev/null || die "manifest identity or train/evaluation disjointness gate failed"

# Harvest wave (skip in full stage when a harvest dataset already exists; always
# skip in the eval stage, which reuses the existing dataset + trained adapter).
if [[ "${A1_STAGE}" == "eval" ]]; then
  printf 'H1.36 eval stage: reusing existing harvest dataset and trained adapter; skipping harvest/gate/train.\n' >&2
elif [[ "${A1_STAGE}" == "full" && -f "${HOST_RUNS}/${DATASET_MANIFEST#/workspace/runs/}" ]]; then
  printf 'H1.36 full stage: reusing existing harvest dataset at %s\n' "${DATASET_MANIFEST}" >&2
else
  harvest_pids=()
  for index in "${!TRAIN_TASKS[@]}"; do
    task="${TRAIN_TASKS[$index]}"; uuid="${GPU_UUIDS[$index]}"; out="${ROOT}/harvest/task${task}"
    run_gpu_container "${uuid}" "h136-harvest-${task}" "${out}" -m scripts.openevo_webshop.h1_closed_loop_tracked \
      --task-manifest "${TASK_MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}" \
      --adapter none --arm self-generated-harvest --tasks "${task}" --rollout-seeds 2026090100 2026090101 2026090102 2026090103 \
      --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" \
      --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature 0.4 --max-steps 30 --max-new 512 \
      --prompt-contract seed3090.webshop-compact-action-prompt.v2 --wandb-mode "${TRACKING_MODE}" \
      --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" \
      --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb" &
    harvest_pids+=("$!")
  done
  harvest_rc=0
  for pid in "${harvest_pids[@]}"; do wait "${pid}" || harvest_rc=1; done
  [[ "${harvest_rc}" == "0" ]] || die "self-generated harvest wave failed"
fi

# Fail-closed capability gate + training. The eval stage skips both and reuses the
# existing trained adapter (no retrain) so the parser is the only changed variable.
if [[ "${A1_STAGE}" != "eval" ]]; then
# Fail-closed capability gate: build the self-evolution dataset and record the verdict.
train_args=()
for task in "${TRAIN_TASKS[@]}"; do train_args+=(--train-task-id "$(printf 'webshop_%06d' "${task}")" --cell-root "${ROOT}/harvest/task${task}"); done
for task in "${EVAL_TASKS[@]}"; do train_args+=(--evaluation-task-id "$(printf 'webshop_%06d' "${task}")"); done
set +e
run_utility_container -m scripts.openevo_webshop.build_self_evolution_dataset "${train_args[@]}" \
  --minimum-success-task-count 2 --minimum-success-episode-count 8 --output-dir "${ROOT}/datasets/self-generated-success-v1" \
  >"${HOST_ROOT}/evidence/h1.36-harvest-dataset-build.log" 2>&1
dataset_rc=$?
set -e
chown 1001:1001 "${HOST_ROOT}/evidence/h1.36-harvest-dataset-build.log" || true

gate_open="false"
qualified_tasks=0
qualified_episodes=0
if [[ "${dataset_rc}" == "0" && -f "${HOST_RUNS}/${DATASET_MANIFEST#/workspace/runs/}" ]]; then
  gate_open="true"
  qualified_tasks="$(jq '.qualified_success_task_ids | length' "${HOST_RUNS}/${DATASET_MANIFEST#/workspace/runs/}")"
  qualified_episodes="$(jq '.record_count' "${HOST_RUNS}/${DATASET_MANIFEST#/workspace/runs/}")"
fi
jq -n --arg stage "${A1_STAGE}" --argjson dataset_rc "${dataset_rc}" --argjson gate_open "${gate_open}" \
  --argjson qualified_task_count "${qualified_tasks}" --argjson qualified_episode_count "${qualified_episodes}" \
  --arg model "${MODEL}" --arg model_revision "${MODEL_REVISION}" --arg build_log "${HOST_ROOT}/evidence/h1.36-harvest-dataset-build.log" \
  '{schema:"openevo.h1.36-harvest-capability-gate.v1", stage:$stage, model:$model, model_revision:$model_revision, dataset_build_exit_code:$dataset_rc, capability_gate_open:$gate_open, qualified_success_task_count:$qualified_task_count, qualified_episode_count:$qualified_episode_count, minimum_success_task_count:2, minimum_success_episode_count:8, build_log:$build_log}' \
  > "${HOST_ROOT}/evidence/h1.36-harvest-gate.json"
chown 1001:1001 "${HOST_ROOT}/evidence/h1.36-harvest-gate.json"

if [[ "${A1_STAGE}" == "harvest" ]]; then
  if [[ "${gate_open}" == "true" ]]; then
    printf 'H1.36 harvest capability gate OPEN at 7B (%s task identities, %s qualified episodes). Evidence: %s\n' \
      "${qualified_tasks}" "${qualified_episodes}" "${HOST_ROOT}/evidence/h1.36-harvest-gate.json"
  else
    printf 'H1.36 harvest capability gate CLOSED at 7B (dataset build exit %s). Capability is still the bottleneck at this scale. Evidence: %s\n' \
      "${dataset_rc}" "${HOST_ROOT}/evidence/h1.36-harvest-gate.json"
  fi
  printf 'H1.36 harvest stage complete under %s\n' "${HOST_ROOT}"
  exit 0
fi

[[ "${gate_open}" == "true" ]] || die "H1.36 capability gate closed at 7B; not proceeding to train/eval"

TRAIN_OUT="${ROOT}/artifacts/h1.36-self-evolution-fit-gpu0-v1"
gpu_gate
run_gpu_container "${GPU_UUIDS[0]}" h136-self-evolution-train-gpu0 "${TRAIN_OUT}" -m scripts.openevo_webshop.run_parametric_smoke_tracked \
  --dataset-manifest "${DATASET_MANIFEST}" --artifact-root "${TRAIN_OUT}" --evidence "${ROOT}/evidence/h1.36-self-evolution-training.json" \
  --run-id h1.36-self-evolution-fit-gpu0-v1 --benchmark webshop --python "${PYTHON}" --base-model "${TRAIN_MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" --max-steps 8 --max-records 96 --minimum-reward 0 --experiment-git-sha "${EXECUTION_SHA}" \
  --upstream-openevo-sha "${CORE_SHA}" --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" \
  --gpu-uuid "${GPU_UUIDS[0]}" --wandb-mode "${TRACKING_MODE}" --wandb-project openevo-webshop-experiments \
  --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${TRAIN_OUT}/wandb"
fi

SELF_ADAPTER="$(jq -r '.adapter_uri' "${HOST_ROOT}/evidence/h1.36-self-evolution-training.json")"
SELF_ADAPTER="${SELF_ADAPTER#file://}"
[[ "${SELF_ADAPTER}" == /workspace/runs/* && -d "${HOST_RUNS}/${SELF_ADAPTER#/workspace/runs/}" ]] || die "self-evolution adapter URI escaped the run root"

launch_wave() {
  local arm="$1" adapter="$2" pids=() index task uuid out rc=0
  gpu_gate
  for index in "${!EVAL_TASKS[@]}"; do
    task="${EVAL_TASKS[$index]}"; uuid="${GPU_UUIDS[$index]}"; out="${EVAL_DIR}/${arm}-task${task}"
    run_gpu_container "${uuid}" "h136-${arm}-${task}" "${out}" -m scripts.openevo_webshop.h1_closed_loop_tracked \
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
  [[ "${rc}" == "0" ]] || die "${arm} evaluation wave failed"
}

launch_wave base none
launch_wave self-evolution "${SELF_ADAPTER}"

cell_args=()
for arm in base self-evolution; do
  for task in "${EVAL_TASKS[@]}"; do cell_args+=(--cell "${arm}:${task}:${EVAL_DIR}/${arm}-task${task}"); done
done
run_utility_container -m scripts.openevo_webshop.reconcile_h1_29 --manifest "${TASK_MANIFEST}" "${cell_args[@]}" \
  --experiment "H1.36-strong-base-self-evolution-v1" --schema "${RECONCILE_SCHEMA}" \
  --output "${RECONCILE_OUT}"

if [[ "${TRACKING_MODE}" == "offline" ]]; then
  spool_hash="$(find "${HOST_ROOT}" -type f -path '*/wandb/*' -print0 | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | awk '{print $1}')"
  jq -n --arg probe "${HOST_ROOT}/evidence/wandb-online-probe.json" --arg spool_hash "${spool_hash}" \
    '{schema:"openevo.wandb-fallback-receipt.v1", mode:"offline", reason:"online initialization failed; see probe", online_probe:$probe, spool_hash:$spool_hash}' \
    > "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
  chown 1001:1001 "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
fi

printf 'H1.36 completed under %s\n' "${HOST_ROOT}"
