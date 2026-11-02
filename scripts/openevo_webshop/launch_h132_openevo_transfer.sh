#!/usr/bin/env bash
set -euo pipefail

# H1.32 is a fresh-task transfer diagnostic for the frozen H1.29 adapter.
# Docker is the control plane; model/evaluator processes run as UID/GID 1001
# and bind to one explicit GPU UUID. This does not authorize H1.30 formal work.

CAMPAIGN_ID="20260817-1200-h132-openevo-transfer"
# v2 is a runtime-contract repair after v1 produced valid rollout payloads but
# invalid Run Manifest PASS receipts because the resource field was omitted.
RUN_NAME="phase-h1.32-openevo-transfer-v2-20260817"
EXECUTION_SHA="${OPENEVO_H132_EXECUTION_SHA:?set OPENEVO_H132_EXECUTION_SHA to the exact server checkout SHA}"
HOST_REPO="${OPENEVO_H132_HOST_REPO:?set OPENEVO_H132_HOST_REPO to the exact standalone server checkout}"
HOST_RUNS="${OPENEVO_H132_HOST_RUNS:-/workspace/runs}"
ROOT="/workspace/runs/${RUN_NAME}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
CONFIG="${HOST_REPO}/configs/experiment/h1.32-openevo-transfer.json"
SOURCE_MANIFEST="${HOST_REPO}/docs/evidence/remote-runs/2026-08-17/h1.30/manifests/h1.30-task-manifest-v2.json"
ADAPTER_HOST="${OPENEVO_H132_ADAPTER_HOST:-/workspace/runs/phase-h1.29-self-evolution-v1-20260816/artifacts/h1.29-self-evolution-fit-gpu0-v1/workers/sd-lora-f2c0c9ef3a94b9cc0c5dd9ba43fba020/adapter}"
IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
PYTHON_FREEZE_SHA="d7220829d4fc81e232523ea3aa22adfb4b55dba95d8236605a3e6f26b3b379d4"
CORE_HOST="/workspace/migration/runtime-src-snapshot/openevo-core"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
MODEL="/workspace/models/qwen2.5-3b-instruct"
MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION="aa8e72537993ba99e69dfaafa59ed015b17504d1"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"

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
TASKS=(6603 1354 2279 2562 3166 5661 4047 4980)
ROLLOUT_SEEDS=(2026091100 2026091101 2026091102 2026091103)

die() { printf 'H1.32 launch gate failed: %s\n' "$*" >&2; exit 2; }

[[ "$(hostname)" == "experiment-host" ]] || die "control-plane hostname is not experiment-host"
[[ "$(id -u)" == "0" ]] || die "control-plane is not root"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/h1_closed_loop_tracked.py" ]] || die "host repo is missing the tracked runner"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "server checkout SHA mismatch"
[[ -f "${CONFIG}" && -f "${SOURCE_MANIFEST}" ]] || die "H1.32 config or source manifest is absent"
[[ "$(jq -r '.status' "${CONFIG}")" == "launch-authorized" ]] || die "H1.32 config is not launch-authorized"
[[ "$(jq -r '.formal_task_consumption_allowed' "${CONFIG}")" == "false" ]] || die "H1.32 must not authorize H1.30 formal consumption"
[[ "$(jq -r '.panel.source_manifest_sha256' "${CONFIG}")" == "$(jq -r '.manifest_sha256' "${SOURCE_MANIFEST}")" ]] || die "panel manifest digest mismatch"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "container image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo core SHA mismatch"
[[ -d "${ADAPTER_HOST}" ]] || die "H1.29 adapter directory is absent"
[[ "$(sha256sum "${ADAPTER_HOST}/adapter_model.safetensors" | awk '{print $1}')" == "ea5b3eddaae9ce60ed1210edf2d46993102b2642de382a0e954a4ddc3c95ddf7" ]] || die "adapter_model.safetensors digest mismatch"
[[ "$(sha256sum "${ADAPTER_HOST}/openevo_sd_lora_state.safetensors" | awk '{print $1}')" == "ebffc412219fc6711188511891296624f768e1bd711fab469f807f281d6cd3fe" ]] || die "OpenEvo state digest mismatch"
jq -e --arg revision "${MODEL_REVISION}" '.base_model == "Qwen/Qwen2.5-3B-Instruct" and .model_revision == $revision and .component_count == 1' "${ADAPTER_HOST}/openevo_sd_lora_state.json" >/dev/null || die "adapter state identity mismatch"

mkdir -p "${HOST_ROOT}"/{manifests,evaluation,evidence}
chown -R 1001:1001 "${HOST_ROOT}"
cp "${CONFIG}" "${HOST_ROOT}/manifests/h1.32-openevo-transfer.json"
cp "${SOURCE_MANIFEST}" "${HOST_ROOT}/manifests/h1.30-task-manifest-v2.json"
chown 1001:1001 "${HOST_ROOT}/manifests"/*.json
MANIFEST_SHA="$(jq -r '.manifest_sha256' "${SOURCE_MANIFEST}")"
CONFIG_SHA="$(sha256sum "${CONFIG}" | awk '{print $1}')"

probe_wandb() {
  local output rc=0
  set +e
  output="$(timeout 45s docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=512m --shm-size=1g \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
    -e WANDB_MODE=online -e WANDB_INIT_TIMEOUT=10 -e WANDB_SILENT=true -e WANDB_API_KEY \
    --entrypoint "${PYTHON}" "${IMAGE}" -c 'import json, wandb
try:
    run = wandb.init(project="openevo-webshop-experiments", entity="example-entity", name="h132-online-probe", job_type="preflight", mode="online", reinit="create_new")
    print(json.dumps({"status": "ok", "run_id": getattr(run, "id", None), "run_url": getattr(run, "url", None)}))
    run.finish()
except BaseException as exc:
    print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
    raise' 2>&1)" || rc=$?
  jq -n --arg status "${rc}" --arg output "${output}" \
    '{schema:"openevo.wandb-online-probe.v1",requested_mode:"online",exit_code:($status|tonumber),output:$output}' \
    > "${HOST_ROOT}/evidence/wandb-online-probe.json"
  chown 1001:1001 "${HOST_ROOT}/evidence/wandb-online-probe.json"
  if [[ "${rc}" == "0" && "${output}" == *'"status": "ok"'* ]]; then
    TRACKING_MODE="online"
  else
    TRACKING_MODE="offline"
    printf 'W&B online initialization failed; explicit offline fallback is recorded.\n' >&2
  fi
}

if [[ "${TRACKING_MODE}" == "online" ]]; then
  probe_wandb
elif [[ "${TRACKING_MODE}" != "offline" ]]; then
  die "unsupported tracking mode ${TRACKING_MODE}"
fi

gpu_gate() {
  local uuid memory utilization apps
  for uuid in "${GPU_UUIDS[@]}"; do
    IFS=',' read -r memory utilization _ < <(nvidia-smi -i "${uuid}" --query-gpu=memory.used,utilization.gpu,pstate --format=csv,noheader,nounits)
    memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
    [[ "${memory}" =~ ^[0-9]+$ && "${memory}" -le 10 ]] || die "${uuid} memory is ${memory} MiB"
    [[ "${utilization}" == "0" ]] || die "${uuid} utilization is ${utilization}%"
    apps="$(nvidia-smi -i "${uuid}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
    [[ -z "${apps}" ]] || die "${uuid} has a compute process"
  done
}

run_gpu_container() {
  local uuid="$1" name="$2" out="$3"; shift 3
  local host_out="${HOST_RUNS}/${out#/workspace/runs/}"
  mkdir -p "${host_out}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts,manifest-output}
  chown -R 1001:1001 "${host_out}"
  docker run --rm --name "${name}" --read-only --tmpfs /tmp:rw,nosuid,size=4g \
    --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 \
    --gpus "device=${uuid}" --user 1001:1001 \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
    -v "${HOST_REPO}":/workspace/repo:ro -v "${CORE_HOST}":/workspace/openevo-core:ro \
    -v "${ADAPTER_HOST}":/workspace/h132-adapter:ro -v /workspace/models:/workspace/models:ro \
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

create_manifest() {
  local arm="$1" task="$2" uuid="$3" gpu_index="$4" out="$5"
  local path="${HOST_ROOT}/manifests/${arm}-task${task}.json"
  cp "${HOST_REPO}/templates/run-manifest.example.json" "${path}"
  jq --arg run_id "h132-${arm}-task${task}" --arg image "${IMAGE}" --arg image_digest "${IMAGE_DIGEST}" \
    --arg uuid "${uuid}" --arg source_sha "${EXECUTION_SHA}" --arg core_sha "${CORE_SHA}" \
    --arg model_id "${MODEL_ID}" --arg revision "${MODEL_REVISION}" --arg manifest "sha256:${MANIFEST_SHA}" \
    --arg config "sha256:${CONFIG_SHA}" --arg adapter "${arm}" --arg mode "${TRACKING_MODE}" \
    --argjson gpu_index "${gpu_index}" \
    '.run={run_id:$run_id,timestamp:.run.timestamp,phase:"phase_g",status:"planned"} |
     .server={server_identity:"experiment-host",control_plane:"experiment-host",experiment_container:("h132-"+$adapter),image_tag:$image,image_digest:$image_digest} |
     .hardware={gpu_indices:[$gpu_index],gpu_uuids:[$uuid],gpu_model:"NVIDIA GeForce RTX 5090"} |
     .code={openevo_experiment_sha:$source_sha,upstream_openevo_repo:"https://github.com/mykcs/OpenEvo.git",upstream_openevo_sha:$core_sha} |
     .model={model_id:$model_id,model_revision:$revision,tokenizer_revision:$revision,dtype:"bfloat16",adapter_identity:$adapter} |
     .data={benchmark:"webshop",split:"h1.32-transfer",dataset_revision:"webshop-1000",dataset_hash:null,index_hash:null,task_manifest_identity:$manifest} |
     .experiment={method:"h132-openevo-transfer",seed:2026,generation:null,config_hash:$config,gate_evidence:{panel_manifest_sha256:$manifest}} |
     .command={argv:[],environment:{WANDB_MODE:$mode,CUDA_VISIBLE_DEVICES:$uuid}} |
     .tracking={mode:$mode,wandb:{enabled:true,entity:"example-entity",project:"openevo-webshop-experiments",id:null},weave:{enabled:false,entity:null,project:null,id:null}}' \
    "${path}" > "${path}.tmp"
  mv "${path}.tmp" "${path}"
  chown 1001:1001 "${path}"
}

launch_wave() {
  local arm="$1" adapter="$2"
  local pids=() rc=0 index task uuid out manifest
  gpu_gate
  for index in "${!TASKS[@]}"; do
    task="${TASKS[$index]}"; uuid="${GPU_UUIDS[$index]}"; out="${ROOT}/evaluation/${arm}-task${task}"
    manifest="${ROOT}/manifests/${arm}-task${task}.json"
    create_manifest "${arm}" "${task}" "${uuid}" "${index}" "${out}"
    run_gpu_container "${uuid}" "h132-${arm}-${task}" "${out}" \
      scripts/run_with_manifest.py --manifest "${manifest}" --output-root "${out}/manifest-output" --cwd /workspace/repo --artifact "${out}/summary.json" --peak-gpu-memory-mb 8192 -- \
      "${PYTHON}" -m scripts.openevo_webshop.h1_closed_loop_tracked \
      --task-manifest "${ROOT}/manifests/h1.30-task-manifest-v2.json" --seed-root "${SEED_ROOT}" \
      --model "${MODEL}" --model-revision "${MODEL_REVISION}" --adapter "${adapter}" --arm "${arm}" \
      --tasks "${task}" --rollout-seeds "${ROLLOUT_SEEDS[@]}" --output "${out}" --gpu-uuid "${uuid}" \
      --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" \
      --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature 0.0 --max-steps 30 --max-new 512 \
      --prompt-contract seed3090.webshop-compact-action-prompt.v2 --wandb-mode "${TRACKING_MODE}" \
      --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" \
      --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
  [[ "${rc}" == "0" ]] || die "${arm} wave failed"
  gpu_gate
}

gpu_gate
cp "${SOURCE_MANIFEST}" "${HOST_ROOT}/manifests/h1.30-task-manifest-v2.json"
chown 1001:1001 "${HOST_ROOT}/manifests/h1.30-task-manifest-v2.json"
launch_wave base none
launch_wave openevo-transfer /workspace/h132-adapter

cell_args=()
for arm in base openevo-transfer; do
  for task in "${TASKS[@]}"; do
    cell_args+=(--cell "${arm}:${task}:${ROOT}/evaluation/${arm}-task${task}")
  done
done
set +e
run_utility_container() {
  docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=2g --shm-size=2g --pids-limit 1024 --user 1001:1001 \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
    -v "${HOST_REPO}":/workspace/repo:ro -v "${HOST_RUNS}":/workspace/runs -w /workspace/repo \
    -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e PYTHONDONTWRITEBYTECODE=1 --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}
run_utility_container -m scripts.openevo_webshop.reconcile_h132_transfer \
  --config "${ROOT}/manifests/h1.32-openevo-transfer.json" \
  --manifest "${ROOT}/manifests/h1.30-task-manifest-v2.json" "${cell_args[@]}" \
  --output "${ROOT}/evidence/h1.32-transfer-reconciliation.json"
reconcile_rc=$?
set -e
gpu_gate

if [[ "${TRACKING_MODE}" == "offline" ]]; then
  spool_hash="$(find "${HOST_ROOT}" -type f -path '*/wandb/*' -print0 | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | awk '{print $1}')"
  jq -n --arg probe "${HOST_ROOT}/evidence/wandb-online-probe.json" --arg spool_hash "${spool_hash}" \
    '{schema:"openevo.wandb-fallback-receipt.v1",mode:"offline",reason:"online initialization failed; see probe",online_probe:$probe,spool_hash:$spool_hash}' \
    > "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
  chown 1001:1001 "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
fi

status="blocked"
[[ "${reconcile_rc}" == "0" ]] && status="pass"
jq -n --arg status "${status}" --arg config_sha "sha256:${CONFIG_SHA}" --arg manifest_sha "${MANIFEST_SHA}" \
  --arg image "${IMAGE_DIGEST}" --arg tracking "${TRACKING_MODE}" --arg run_root "${HOST_ROOT}" \
  '{schema:"openevo.webshop-h1.32-transfer-launch-receipt.v1",status:$status,formal_task_consumption_allowed:false,gpu_workload_launched:true,config_sha256:$config_sha,panel_manifest_sha256:$manifest_sha,image_digest:$image,tracking_mode:$tracking,run_root:$run_root,attempts:64,arms:["base","openevo-transfer"],release_gate:"all UUIDs idle after both waves"}' \
  > "${HOST_ROOT}/evidence/h1.32-transfer-launch-receipt.json"
chown 1001:1001 "${HOST_ROOT}/evidence/h1.32-transfer-launch-receipt.json"
printf 'H1.32 transfer completed with reconciliation status %s at %s\n' "${status}" "${HOST_ROOT}"
exit "${reconcile_rc}"
