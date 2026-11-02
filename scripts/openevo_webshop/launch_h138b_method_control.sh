#!/usr/bin/env bash
set -euo pipefail

# H1.38B only: matched ordinary sequential LoRA versus the pinned WebShop
# SD-LoRA implementation. Partial formal cells are never retried or overwritten.
CAMPAIGN_ID="20260820-0129-h138b-method-control"
RUN_NAME="${OPENEVO_H138B_RUN_NAME:-phase-h1.38b-method-control-v1-20260820}"
EXECUTION_SHA="${OPENEVO_H138B_EXECUTION_SHA:?set to the pushed preregistration descendant SHA}"
HOST_REPO="${OPENEVO_H138B_HOST_REPO:?set to the exact clean server checkout}"
HOST_RUNS="${OPENEVO_H138B_HOST_RUNS:-/workspace/runs}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
ROOT="/workspace/runs/${RUN_NAME}"
AMENDMENT="configs/experiment/h1.38b-continuation-amendment-2026-08-20.json"
CONFIG="configs/experiment/h1.38b-method-control-v1.json"
COMMITTED_MANIFEST="configs/experiment/manifests/h1.38b-method-control-eval-v1.json"
COMMITTED_PLANS="configs/experiment/manifests/h1.38b-frozen-sample-plans-v1.json"
IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
PYTHON_FREEZE_SHA="e3cb53bced07b519157ceca2698935bf0be45a58d08465c87c618ef62c31ee18"
CORE_HOST="/workspace/migration/runtime-src-snapshot/openevo-core"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
CORE_PATCH_SHA="38fd5305a23f9834abc8f022f4636d1b820e6117ee3e554c0defe03a7b5e726f"
MODEL="/workspace/models/qwen2.5-7b-instruct"
MODEL_HOST="/workspace/models/qwen2.5-7b-instruct"
MODEL_REVISION="0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
D8_ROOT="/workspace/runs/phase-h1.38a-pooled-diversity-v1-20260819/datasets/diversity-8"
D8_HOST="${HOST_RUNS}/phase-h1.38a-pooled-diversity-v1-20260819/datasets/diversity-8"
HF_CACHE_HOST="${HOST_RUNS}/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
JDK="/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1"
ABORTED_LEDGER="/workspace/repo/configs/experiment/h1.38-reservoir-v1-aborted-preprotocol-ledger.json"
MANIFEST="${ROOT}/manifests/h1.38b-method-control-eval-v1.json"
PLANS="${ROOT}/manifests/h1.38b-frozen-sample-plans-v1.json"
DEADLINE_EPOCH=""
CAP=6.0
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"
IFS=',' read -r -a GPU_UUIDS <<< "${OPENEVO_H138B_GPU_UUIDS:?set the explicitly authorized comma-separated GPU UUID set after preregistration publication}"

die() { printf 'H1.38B gate failed: %s\n' "$*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
safe_time() { (( "$(date +%s)" < DEADLINE_EPOCH - 600 )) || die "10-hour window reached; stop at safe cell boundary"; }

[[ "$(hostname)" == experiment-host && "$(id -u)" == 0 ]] || die "expected root@experiment-host through experiment-host"
(( ${#GPU_UUIDS[@]} >= 1 && ${#GPU_UUIDS[@]} <= 8 )) || die "authorized GPU UUID count must be 1..8"
declare -A seen_gpu_uuids=()
for uuid in "${GPU_UUIDS[@]}"; do
  [[ "${uuid}" =~ ^GPU-[0-9A-Fa-f-]+$ ]] || die "malformed authorized GPU UUID"
  [[ -z "${seen_gpu_uuids[${uuid}]:-}" ]] || die "duplicate authorized GPU UUID"
  seen_gpu_uuids["${uuid}"]=1
done
[[ "${HOST_REPO}" == /workspace/* ]] || die "checkout escaped approved workspace"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "checkout SHA mismatch"
[[ -z "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" status --porcelain)" ]] || die "server checkout is dirty"
[[ -f "${HOST_REPO}/${CONFIG}" && -f "${HOST_REPO}/${COMMITTED_MANIFEST}" && -f "${HOST_REPO}/${COMMITTED_PLANS}" && -f "${HOST_REPO}/${AMENDMENT}" ]] || die "preregistration/amendment assets missing"
DEADLINE_EPOCH="$(jq -r '.window.no_new_cell_after_epoch // empty' "${HOST_REPO}/${AMENDMENT}")"
[[ "${DEADLINE_EPOCH}" =~ ^[0-9]+$ ]] || die "amendment no-new-cell epoch missing"
jq -e --arg id "${CAMPAIGN_ID}" --arg amendment "${AMENDMENT}" '.campaign_id == $id and .phase == "H1.38B" and .status == "launch-authorized" and .formal_task_consumption_allowed == true and .continuation_amendment == $amendment' "${HOST_REPO}/configs/experiment/current-campaign.json" >/dev/null || die "continuation campaign is not launch-authorized"
jq -e --arg id "${CAMPAIGN_ID}" --arg parent "23aa9ada98920c7f0577f06d97f6804f548a330f" '.status == "published" and .campaign_id == $id and .parent_scientific_head == $parent and .scope == "time-window-and-execution-orchestration-only" and .scientific_contract_changed == false and .window.duration_hours == 24 and .window.safe_boundary_minutes == 20 and .budget.campaign_total_gpu_hours_cap == 6 and (.preserved_contract.planned_cells == 184) and (.preserved_contract.planned_attempts == 704)' "${HOST_REPO}/${AMENDMENT}" >/dev/null || die "continuation amendment contract mismatch"
[[ "$(sha256_file "${HOST_REPO}/scripts/openevo_webshop/launch_h138b_method_control.sh")" == "$(jq -r '.execution.launcher_sha256' "${HOST_REPO}/${AMENDMENT}")" ]] || die "continuation launcher hash mismatch"
jq -e --argjson cap "${CAP}" '.status == "launch-authorized" and .formal_task_consumption_allowed == true and .gpu_authorization_status == "campaign-contract-authorized" and .gpu_authorization.session_reconfirmation_required == false and .execution_sha_binding == "launcher-environment" and .budget.gpu_hours_cap == $cap' "${HOST_REPO}/${CONFIG}" >/dev/null || die "preregistration execution/budget/authorization contract mismatch"
selected_gpu_json="$(printf '%s\n' "${GPU_UUIDS[@]}" | jq -Rsc 'split("\n")[:-1]')"
prereg_gpu_json="$(jq -c '.authorized_gpu_uuids' "${HOST_REPO}/${CONFIG}")"
current_gpu_json="$(jq -c '.authorized_gpu_uuids' "${HOST_REPO}/configs/experiment/current-campaign.json")"
[[ "${selected_gpu_json}" == "${prereg_gpu_json}" && "${selected_gpu_json}" == "${current_gpu_json}" ]] || die "selected GPU UUID sequence differs from the exact published authorization set"
prereg_validation="$("${PYTHON}" "${HOST_REPO}/scripts/openevo_webshop/validate_h138b_preregistration.py" --repo-root "${HOST_REPO}")" || die "published preregistration validator failed"
jq -e '.status == "pass" and ((.launch_authorized == true and .formal_task_consumption_allowed == true and (.blockers | length) == 0) or (.launch_authorized == false and .formal_task_consumption_allowed == false and .blockers == ["preregistered-time-window-closed"]))' <<<"${prereg_validation}" >/dev/null || die "published preregistration validation did not preserve the exact closed-window contract"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo stable pin mismatch"
CORE_RUNTIME_PATHS=(src/openevo/evolution/parametric/sd_lora_trainer.py src/openevo/evolution/parametric/trainer_service.py)
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" diff -- "${CORE_RUNTIME_PATHS[@]}" | sha256sum | awk '{print $1}')" == "${CORE_PATCH_SHA}" ]] || die "approved trainer-only runtime patch receipt mismatch"
mapfile -t core_dirty_paths < <(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" status --porcelain=v1 | sed 's/^...//' | LC_ALL=C sort)
expected_core_dirty_paths=(desktop/.env.openevo-desktop src/openevo/evolution/parametric/sd_lora_trainer.py src/openevo/evolution/parametric/trainer_service.py)
[[ "${core_dirty_paths[*]}" == "${expected_core_dirty_paths[*]}" ]] || die "OpenEvo checkout has an unexpected dirty path"
[[ "$("${PYTHON}" -m pip freeze --all | LC_ALL=C sort | sha256sum | awk '{print $1}')" == "${PYTHON_FREEZE_SHA}" ]] || die "Python freeze mismatch"
[[ "$(sha256_file "${D8_HOST}/records.jsonl")" == d206c20e4d08a9c1785a6af9c2edb510db4a478d5d459395537706a2d7d8191c ]] || die "sealed D8 records mismatch"
jq -e '.selection_sha256 == "b8c35b51aed3bf942cb2e45e7c9c0e68239d6efc27a3d7be8bc6a5ed54ff70c7" and .record_count == 16 and .identity_count == 8' "${D8_HOST}/manifest.json" >/dev/null || die "sealed D8 manifest mismatch"
for f in config.json model.safetensors.index.json tokenizer_config.json tokenizer.json; do [[ -f "${MODEL_HOST}/${f}" ]] || die "missing model component ${f}"; done
cfg="$(sha256_file "${MODEL_HOST}/config.json")"; idx="$(sha256_file "${MODEL_HOST}/model.safetensors.index.json")"; tok="$(sha256_file "${MODEL_HOST}/tokenizer_config.json")"
[[ "$(printf '%s\n%s\n%s\n' "${cfg}" "${idx}" "${tok}" | sha256sum | awk '{print $1}')" == "${MODEL_REVISION}" ]] || die "model content pin mismatch"

GIT_COMMON_HOST="$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse --git-common-dir)"
[[ "${GIT_COMMON_HOST}" == /* ]] || GIT_COMMON_HOST="${HOST_REPO}/${GIT_COMMON_HOST}"
GIT_COMMON_HOST="$(realpath -e "${GIT_COMMON_HOST}")"
[[ "${GIT_COMMON_HOST}" == /workspace/* ]] || die "Git common directory escaped workspace"
mkdir -p "${HOST_ROOT}"/{manifests,datasets,artifacts,evaluation,evidence,logs} "${HOST_ROOT}/evidence/gpu-job-receipts"
chown -R 1001:1001 "${HOST_ROOT}"

utility() {
  docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=2g --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=4g --user 1001:1001 \
    -v "${HOST_REPO}":/workspace/repo:ro -v "${GIT_COMMON_HOST}":"${GIT_COMMON_HOST}":ro \
    -v "${CORE_HOST}":/workspace/openevo-core:ro -v /workspace/models:/workspace/models:ro \
    -v /workspace/models:/workspace/workspace/models:ro -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro \
    -v "${HOST_RUNS}":/workspace/runs -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
    -w /workspace/repo -e PYTHONPATH=/workspace/openevo-core/src:/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/repo \
    -e PYTHONDONTWRITEBYTECODE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e WANDB_API_KEY -e WANDB_INIT_TIMEOUT=10 \
    -e HF_HOME=/workspace/hf-cache -e JAVA_HOME="${JDK}" -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}

on_exit() {
  local rc=$? receipt partial_reconciliation receipt_args=()
  trap - EXIT
  if (( rc != 0 )) && [[ -f "${HOST_ROOT}/manifests/h1.38b-method-control-eval-v1.json" ]]; then
    for receipt in "${HOST_ROOT}"/evidence/training-receipts/*.json; do
      [[ -f "${receipt}" ]] && receipt_args+=(--training-evidence "${ROOT}/${receipt#"${HOST_ROOT}"/}")
    done
    partial_reconciliation="${HOST_ROOT}/evidence/h1.38b-partial-reconciliation.json"
    utility -m scripts.openevo_webshop.reconcile_h138b_method_control --manifest "${MANIFEST}" --run-root "${ROOT}" \
      "${receipt_args[@]}" --output "${ROOT}/evidence/h1.38b-partial-reconciliation.json" >/dev/null 2>&1 || true
    [[ -f "${partial_reconciliation}" ]] && attach_gpu_job_accounting "${partial_reconciliation}" || true
  fi
  exit "${rc}"
}
trap on_exit EXIT

idle_gpus() {
  local u line mem util apps
  for u in "${GPU_UUIDS[@]}"; do
    line="$(nvidia-smi -i "${u}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" || continue
    IFS=',' read -r mem util <<<"${line}"; mem="${mem//[[:space:]]/}"; util="${util//[[:space:]]/}"
    apps="$(nvidia-smi -i "${u}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"
    [[ "${mem}" =~ ^[0-9]+$ && "${mem}" -le 10 && "${util}" == 0 && -z "${apps}" ]] && printf '%s\n' "${u}"
  done
  return 0
}

run_gpu() {
  local uuid="$1" name="$2" out="$3"; shift 3
  local runtime_out="${ROOT}/runtime/${name}"
  local host_runtime="${HOST_ROOT}/runtime/${name}"
  local host_model_cache="${host_runtime}/hf-hub/models--Qwen--Qwen2.5-7B-Instruct"
  local receipt="${HOST_ROOT}/evidence/gpu-job-receipts/${name}.json"
  local receipt_tmp="${receipt}.tmp.$$.$RANDOM"
  local line mem util apps started_at ended_at started_epoch ended_epoch wall_seconds rc status
  [[ "${name}" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe GPU job receipt name"
  [[ ! -e "${receipt}" ]] || die "GPU job receipt already exists: ${name}"
  safe_time
  within_cap
  line="$(nvidia-smi -i "${uuid}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" || die "GPU ${uuid} unavailable"
  IFS=',' read -r mem util <<<"${line}"; mem="${mem//[[:space:]]/}"; util="${util//[[:space:]]/}"
  apps="$(nvidia-smi -i "${uuid}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ "${mem}" -le 10 && "${util}" == 0 && -z "${apps}" ]] || die "selected GPU ${uuid} is not immediately live-idle"
  [[ ! -e "${host_runtime}" ]] || die "GPU job runtime directory already exists: ${name}"
  mkdir -p "${host_runtime}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts} \
    "${host_model_cache}"/{refs,snapshots}
  ln -s "${MODEL}" "${host_model_cache}/snapshots/${MODEL_REVISION}"
  printf '%s' "${MODEL_REVISION}" > "${host_model_cache}/refs/${MODEL_REVISION}"
  chown -R 1001:1001 "${host_runtime}"
  started_at="$(date --iso-8601=seconds --utc)"
  started_epoch="$(date +%s)"
  if docker run --rm --name "${name}" --read-only --tmpfs /tmp:rw,nosuid,size=4g --shm-size=64g --pids-limit 4096 \
    --gpus "device=${uuid}" --user 1001:1001 -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
    -v "${HOST_REPO}":/workspace/repo:ro -v "${GIT_COMMON_HOST}":"${GIT_COMMON_HOST}":ro -v "${CORE_HOST}":/workspace/openevo-core:ro \
    -v /workspace/models:/workspace/models:ro -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro -v "${HOST_RUNS}":/workspace/runs \
    -w /workspace/repo -e CUDA_VISIBLE_DEVICES="${uuid}" -e WANDB_API_KEY -e WANDB_ENTITY=example-entity \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/repo \
    -e WANDB_PROJECT=openevo-webshop-experiments -e WANDB_MODE="${TRACKING_MODE}" -e TOKENIZERS_PARALLELISM=false \
    -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e JAVA_HOME="${JDK}" \
    -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -e PYTHONPATH=/workspace/openevo-core/src:/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e PYTHONDONTWRITEBYTECODE=1 -e HOME="${runtime_out}/home" -e TMPDIR="${runtime_out}/tmp" -e XDG_CACHE_HOME="${runtime_out}/xdg-cache" \
    -e HF_HOME=/workspace/hf-cache -e HF_HUB_CACHE="${runtime_out}/hf-hub" -e TRANSFORMERS_CACHE="${runtime_out}/hf-hub" \
    -e WANDB_DIR="${runtime_out}/wandb" -e WANDB_CACHE_DIR="${runtime_out}/wandb-cache" \
    -e WANDB_CONFIG_DIR="${runtime_out}/wandb-config" -e WANDB_ARTIFACT_DIR="${runtime_out}/wandb-artifacts" \
    --entrypoint "${PYTHON}" "${IMAGE}" "$@"
  then
    rc=0
  else
    rc=$?
  fi
  ended_epoch="$(date +%s)"
  ended_at="$(date --iso-8601=seconds --utc)"
  wall_seconds=$((ended_epoch - started_epoch))
  (( rc == 0 )) && status=pass || status=failed
  jq -n \
    --arg schema "openevo.webshop-h1.38b-gpu-job-wall-time.v1" \
    --arg status "${status}" --arg job "${name}" --arg uuid "${uuid}" --arg output "${out}" \
    --arg started "${started_at}" --arg ended "${ended_at}" \
    --argjson started_epoch "${started_epoch}" --argjson ended_epoch "${ended_epoch}" \
    --argjson wall_seconds "${wall_seconds}" --argjson exit_code "${rc}" \
    '{schema:$schema,status:$status,job_name:$job,gpu_uuid:$uuid,output_path:$output,started_at_utc:$started,ended_at_utc:$ended,started_epoch:$started_epoch,ended_epoch:$ended_epoch,wall_time_seconds:$wall_seconds,exit_code:$exit_code}' \
    > "${receipt_tmp}" || { rm -f "${receipt_tmp}"; die "failed to write GPU job receipt: ${name}"; }
  chmod 600 "${receipt_tmp}"
  mv "${receipt_tmp}" "${receipt}"
  (( rc == 0 )) || return "${rc}"
}

gpu_job_accounting() {
  local receipts=()
  while IFS= read -r -d '' receipt; do receipts+=("${receipt}"); done < <(
    find "${HOST_ROOT}/evidence/gpu-job-receipts" -maxdepth 1 -name '*.json' -type f -print0
  )
  if ((${#receipts[@]} == 0)); then
    jq -n '{source:"outer-docker-wall-time-receipts-v1",receipt_count:0,failed_job_count:0,training:0,reload:0,evaluation:0,total_seconds:0,total:0}'
    return
  fi
  jq -s '
    def seconds: map(.wall_time_seconds) | add // 0;
    {
      source:"outer-docker-wall-time-receipts-v1",
      receipt_count:length,
      failed_job_count:(map(select(.exit_code != 0)) | length),
      training:((map(select(.job_name | startswith("h138b-train-"))) | seconds) / 3600),
      reload:((map(select(.job_name | startswith("h138b-reload-"))) | seconds) / 3600),
      evaluation:((map(select(.job_name | test("^h138b-(acquisition|retention|fresh)-"))) | seconds) / 3600),
      total_seconds:seconds,
      total:(seconds / 3600)
    }
  ' "${receipts[@]}"
}
gpu_hours() { gpu_job_accounting | jq -r '.total'; }
attach_gpu_job_accounting() {
  local reconciliation="$1" accounting temporary
  [[ -f "${reconciliation}" ]] || return 1
  accounting="$(gpu_job_accounting)" || return 1
  temporary="${reconciliation}.tmp.$$.$RANDOM"
  jq --argjson accounting "${accounting}" '.gpu_hours = $accounting' "${reconciliation}" > "${temporary}" || { rm -f "${temporary}"; return 1; }
  chmod 600 "${temporary}"
  mv "${temporary}" "${reconciliation}"
}
within_cap() { jq -e --argjson used "$(gpu_hours)" --argjson cap "${CAP}" '$used < $cap' >/dev/null || die "preregistered GPU-hour cap reached"; }
one_idle_gpu() { local values=(); while true; do safe_time; within_cap; mapfile -t values < <(idle_gpus); ((${#values[@]})) && { printf '%s\n' "${values[0]}"; return; }; sleep 60; done; }

# CPU/contract gate and deterministic input re-derivation consume no task rollouts.
utility -m unittest discover -s tests -p 'test_*.py' || die "CPU tests failed"
if [[ ! -f "${HOST_ROOT}/evidence/input-build.json" ]]; then
  utility -m scripts.openevo_webshop.build_h138b_method_control_inputs \
    --d8-manifest "${D8_ROOT}/manifest.json" --increment-root "${ROOT}/datasets/increments" --seed-root "${SEED_ROOT}" \
    --data-dir "${SEED_ROOT}/agent_system/environments/env_package/webshop/webshop/data" \
    --index-dir "${SEED_ROOT}/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k" \
    --prior-root /workspace/repo/docs/evidence/remote-runs --aborted-ledger "${ABORTED_LEDGER}" \
    --evaluation-manifest "${MANIFEST}" --experiment-git-sha "$(jq -r '.source.experiment_git_sha' "${HOST_REPO}/${COMMITTED_MANIFEST}")" \
    > "${HOST_ROOT}/evidence/input-build.json"
  chown 1001:1001 "${HOST_ROOT}/evidence/input-build.json"
fi
[[ "$(sha256_file "${HOST_ROOT}/manifests/h1.38b-method-control-eval-v1.json")" == "$(sha256_file "${HOST_REPO}/${COMMITTED_MANIFEST}")" ]] || die "fresh panel re-derivation mismatch"
if [[ ! -f "${HOST_ROOT}/manifests/h1.38b-frozen-sample-plans-v1.json" ]]; then
  utility -m scripts.openevo_webshop.freeze_h138b_sample_plans --increment-root "${ROOT}/datasets/increments" \
    --tokenizer-root "${MODEL}" --output "${PLANS}"
fi
[[ "$(sha256_file "${HOST_ROOT}/manifests/h1.38b-frozen-sample-plans-v1.json")" == "$(sha256_file "${HOST_REPO}/${COMMITTED_PLANS}")" ]] || die "sample-plan re-derivation mismatch"

if [[ "${TRACKING_MODE}" == online ]]; then
  set +e
  probe="$(utility -c 'import json, wandb; r=wandb.init(project="openevo-webshop-experiments",entity="example-entity",name="h138b-online-probe",mode="online",reinit="create_new",settings=wandb.Settings(init_timeout=10)); print(json.dumps({"status":"ok","run_id":r.id})); r.finish()' 2>&1)"; probe_rc=$?
  set -e
  jq -n --arg output "${probe}" --argjson exit_code "${probe_rc}" '{schema:"openevo.wandb-online-probe.v1",requested_mode:"online",exit_code:$exit_code,output:$output}' > "${HOST_ROOT}/evidence/wandb-online-probe.json"
  [[ "${probe_rc}" == 0 && "${probe}" == *'"status": "ok"'* ]] || TRACKING_MODE=offline
elif [[ "${TRACKING_MODE}" != offline ]]; then die "tracking mode must be online or offline"; fi

run_structural_gpu_gate() {
  local gate_training_host="${HOST_ROOT}/evidence/gpu-gate/sd-lora-training.json"
  local gate_reload_host="${HOST_ROOT}/evidence/gpu-gate/sd-lora-reload.json"
  local gate_summary_host="${HOST_ROOT}/evidence/h1.38b-pinned-7b-gpu-gate.json"
  local gate_dataset gate_uuid gate_adapter_uri gate_adapter
  [[ -f "${gate_summary_host}" ]] && { jq -e '.status == "pass" and .formal_task_consumption_allowed == false and .fresh_panel_consumed == false' "${gate_summary_host}" >/dev/null || die "existing pinned-7B GPU gate receipt is invalid"; return; }
  gate_dataset="$(find "${HOST_ROOT}/datasets/increments" -maxdepth 2 -path '*/increment-00-*/manifest.json' -print -quit)"
  [[ -n "${gate_dataset}" ]] || die "bounded GPU gate dataset is missing"
  gate_dataset="${ROOT}/${gate_dataset#"${HOST_ROOT}"/}"
  gate_uuid="$(one_idle_gpu)"
  run_gpu "${gate_uuid}" "h138b-gpu-gate-sd" "${ROOT}/evidence/gpu-gate/sd-lora" -m scripts.openevo_webshop.run_parametric_smoke_tracked \
    --dataset-manifest "${gate_dataset}" --artifact-root "${ROOT}/artifacts/gpu-gate/sd-lora" \
    --evidence "${ROOT}/evidence/gpu-gate/sd-lora-training.json" --run-id "h138b-pinned-7b-gpu-gate" \
    --benchmark webshop --python "${PYTHON}" --base-model Qwen/Qwen2.5-7B-Instruct --model-revision "${MODEL_REVISION}" \
    --max-steps 16 --max-records 16 --minimum-reward 0 --experiment-git-sha "${EXECUTION_SHA}" \
    --upstream-openevo-sha "${CORE_SHA}" --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" \
    --gpu-uuid "${gate_uuid}" --wandb-mode "${TRACKING_MODE}" --wandb-project openevo-webshop-experiments \
    --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" --wandb-group "${CAMPAIGN_ID}" \
    --wandb-dir "${ROOT}/evidence/gpu-gate/wandb"
  jq -e --arg revision "${MODEL_REVISION}" \
    '.schema == "seed3090.parametric_smoke.v1" and .dataset_records_sha256 != null and (.adapter_uri | startswith("file://")) and .artifact_manifest.adapter_format == "lora" and .artifact_manifest.model_revision == $revision and .artifact_manifest.effective_rank == 4 and .training_record_count == 16 and .optimizer_steps > 0 and .artifact_manifest.state_manifest_path != null' \
    "${gate_training_host}" >/dev/null || die "bounded pinned-7B SD-LoRA training gate did not produce a qualified adapter"
  gate_adapter_uri="$(jq -er '.adapter_uri' "${gate_training_host}")"
  gate_adapter="${gate_adapter_uri#file://}"
  run_gpu "${gate_uuid}" "h138b-gpu-gate-reload" "${ROOT}/evidence/gpu-gate/sd-lora-reload" \
    -m scripts.openevo_webshop.verify_adapter_reload --model "${MODEL}" --adapter "${gate_adapter}" \
    --model-revision "${MODEL_REVISION}" --increment-index 0 --output "${ROOT}/evidence/gpu-gate/sd-lora-reload.json" --require-cuda
  jq -e '.schema == "openevo.webshop-adapter-reload.v1" and .status == "pass" and .task_consumption == "none" and .loaded_lora_tensor_count > 0 and .base_trainable_parameter_count == 0 and .first_load_digest == .second_load_digest' "${gate_reload_host}" >/dev/null || die "bounded pinned-7B adapter reload gate failed"
  jq -n --arg execution_sha "${EXECUTION_SHA}" --arg uuid "${gate_uuid}" --arg training "${ROOT}/evidence/gpu-gate/sd-lora-training.json" --arg reload "${ROOT}/evidence/gpu-gate/sd-lora-reload.json" \
    --arg training_sha "$(sha256_file "${gate_training_host}")" --arg reload_sha "$(sha256_file "${gate_reload_host}")" \
    '{schema:"openevo.webshop-h1.38b-pinned-7b-gpu-gate.v1",status:"pass",formal_task_consumption_allowed:false,fresh_panel_consumed:false,execution_sha:$execution_sha,gpu_uuid:$uuid,training_evidence:$training,reload_evidence:$reload,training_evidence_sha256:$training_sha,reload_evidence_sha256:$reload_sha,no_base_fallback:true,active_lora_tensor_verified:true,inference_load_verified:true}' \
    > "${gate_summary_host}"
  chown -R 1001:1001 "${HOST_ROOT}/evidence/gpu-gate"
}

run_structural_gpu_gate

eval_cell() {
  local stage="$1" arm="$2" task="$3" adapter="$4" file_sha="$5" tensor_sha="$6" uuid="$7" increment="$8"; shift 8
  local out="${ROOT}/evaluation/${stage}/${arm}/task${task}" host_out="${HOST_ROOT}/evaluation/${stage}/${arm}/task${task}"
  if [[ -f "${host_out}/attempts.jsonl" && -f "${host_out}/summary.json" ]]; then
    expected_seeds="$(printf '%s\n' "$@" | jq -R 'tonumber' | jq -s .)"
    jq -es --arg stage "${stage}" --arg arm "${arm}" --argjson task "${task}" --argjson seeds "${expected_seeds}" \
      'length == ($seeds | length) and all(.[]; .evaluation_stage == $stage and .arm == $arm and .goal_idx == $task) and ([.[].rollout_seed] | sort) == ($seeds | sort)' \
      "${host_out}/attempts.jsonl" >/dev/null || die "completed cell identity mismatch ${stage}/${arm}/${task}"
    return 0
  fi
  [[ ! -e "${host_out}" ]] || die "partial formal cell ${stage}/${arm}/${task}"
  local digest_args=()
  [[ "${arm}" == base ]] || digest_args=(--expected-adapter-file-sha256 "${file_sha}" --expected-adapter-tensor-digest "${tensor_sha}")
  run_gpu "${uuid}" "h138b-${stage}-${arm}-${task}" "${out}" -m scripts.openevo_webshop.h1_closed_loop_tracked \
    --task-manifest "${MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}" \
    --adapter "${adapter}" --arm "${arm}" --tasks "${task}" --rollout-seeds "$@" --evaluation-stage "${stage}" \
    --increment-index "${increment}" --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" \
    --upstream-openevo-sha "${CORE_SHA}" --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" \
    --temperature 0 --max-steps 30 --max-new 512 --prompt-contract seed3090.webshop-compact-action-prompt.v2 \
    --action-contract seed3090.webshop-action-parser.v3-action-command --wandb-mode "${TRACKING_MODE}" \
    --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" \
    --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb" "${digest_args[@]}"
}

for increment in $(seq 0 7); do
  safe_time; within_cap; uuid="$(one_idle_gpu)"
  dataset="${ROOT}/datasets/increments/increment-$(printf '%02d' "${increment}")-$(jq -r ".training_tasks[${increment}].task_id" "${HOST_REPO}/${COMMITTED_MANIFEST}")/manifest.json"
  # Directory names use the task_id already present in the D8 records.
  dataset="$(find "${HOST_ROOT}/datasets/increments" -maxdepth 2 -path "*/increment-$(printf '%02d' "${increment}")-*/manifest.json" -print -quit)"
  [[ -n "${dataset}" ]] || die "missing increment ${increment}"
  dataset="${ROOT}/${dataset#"${HOST_ROOT}"/}"
  plan_sha="$(jq -r ".increments[${increment}].training_plan.plan_sha256" "${HOST_REPO}/${COMMITTED_PLANS}")"
  examples_sha="$(jq -r ".increments[${increment}].current_examples_sha256" "${HOST_REPO}/${COMMITTED_PLANS}")"
  ordinary_out="${ROOT}/artifacts/ordinary-lora/stage$(printf '%02d' "${increment}")"
  ordinary_ev="${ROOT}/evidence/training/ordinary-lora-stage$(printf '%02d' "${increment}").json"
  if [[ ! -f "${HOST_ROOT}/evidence/training/ordinary-lora-stage$(printf '%02d' "${increment}").json" ]]; then
    prior_args=(); (( increment == 0 )) || prior_args=(--prior-adapter "${ROOT}/artifacts/ordinary-lora/stage$(printf '%02d' "$((increment-1))")")
    run_gpu "${uuid}" "h138b-train-ordinary-${increment}" "${ordinary_out}" -m scripts.openevo_webshop.ordinary_lora_trainer \
      --dataset-manifest "${dataset}" --dataset-artifact-id "h138b-increment-$(printf '%02d' "${increment}")" --increment-index "${increment}" \
      --model-root "${MODEL}" --output-adapter "${ordinary_out}" --evidence "${ordinary_ev}" --upstream-openevo-sha "${CORE_SHA}" \
      --expected-current-examples-sha256 "${examples_sha}" --expected-plan-sha256 "${plan_sha}" "${prior_args[@]}"
  fi
  sd_root="${ROOT}/artifacts/sd-lora/stage$(printf '%02d' "${increment}")"; sd_ev="${ROOT}/evidence/training/sd-lora-stage$(printf '%02d' "${increment}").json"
  if [[ ! -f "${HOST_ROOT}/evidence/training/sd-lora-stage$(printf '%02d' "${increment}").json" ]]; then
    sd_job_name="h138b-train-sd-${increment}"
    failed_sd_receipt="${HOST_ROOT}/evidence/gpu-job-receipts/${sd_job_name}.json"
    if [[ -f "${failed_sd_receipt}" ]]; then
      jq -e '.exit_code != 0' "${failed_sd_receipt}" >/dev/null || die "existing SD-LoRA training receipt is not a failed engineering attempt"
      transition="${HOST_ROOT}/evidence/source-transitions/h1.38b-sd-stage$(printf '%02d' "${increment}")-artifact-root.json"
      mkdir -p "${HOST_ROOT}/evidence/source-transitions"
      if [[ ! -f "${transition}" ]]; then
        jq -n --arg campaign "${CAMPAIGN_ID}" --arg parent "${OPENEVO_H138B_PARENT_EXECUTION_SHA:-e35f118b9ef594cba34522e682ded0105aa18d65}" --arg execution "${EXECUTION_SHA}" --arg failed "${ROOT}/evidence/gpu-job-receipts/h138b-train-sd-${increment}.json" --arg stage "${increment}" \
          '{schema:"openevo.webshop-h1.38b-source-transition.v1",status:"pass",campaign_id:$campaign,parent_execution_sha:$parent,execution_sha:$execution,affected_method:"sd-lora",affected_increment:($stage|tonumber),trigger:"prior adapter path was outside the per-stage artifact payload root",correction:"bind all SD-LoRA stage artifacts to the shared method artifact root so prior adapters remain inside the allowed payload root",scientific_contract_changed:false,failed_engineering_receipt:$failed,preserved_completed_cells:true,fresh_panel_consumed:false}' \
          > "${transition}"
        chmod 600 "${transition}"
      fi
      sd_job_name="${sd_job_name}-correction-01"
    fi
    prior_args=()
    if (( increment > 0 )); then
      prior_uri="$(jq -r '.adapter_uri' "${HOST_ROOT}/evidence/training/sd-lora-stage$(printf '%02d' "$((increment-1))").json")"
      prior_id="$(jq -r '.artifact.artifact_id // .artifact.name' "${HOST_ROOT}/evidence/training/sd-lora-stage$(printf '%02d' "$((increment-1))").json")"
      prior_args=(--prior-adapter "${prior_uri#file://}" --prior-artifact-id "${prior_id}")
    fi
    run_gpu "${uuid}" "${sd_job_name}" "${sd_root}" -m scripts.openevo_webshop.run_parametric_smoke_tracked \
      --dataset-manifest "${dataset}" --dataset-artifact-id "h138b-increment-$(printf '%02d' "${increment}")" \
      --artifact-root "${ROOT}/artifacts/sd-lora" --evidence "${sd_ev}" --run-id "h138b-sd-stage${increment}" \
      --benchmark webshop --python "${PYTHON}" --base-model Qwen/Qwen2.5-7B-Instruct --model-revision "${MODEL_REVISION}" \
      --max-steps 16 --max-records 16 --minimum-reward 0 --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" \
      --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --gpu-uuid "${uuid}" --wandb-mode "${TRACKING_MODE}" \
      --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" --wandb-group "${CAMPAIGN_ID}" \
      --wandb-dir "${sd_root}/wandb" "${prior_args[@]}"
  fi
  sd_adapter="$(jq -r '.adapter_uri | sub("^file://";"")' "${HOST_ROOT}/evidence/training/sd-lora-stage$(printf '%02d' "${increment}").json")"
  for method_adapter in "ordinary-lora|${ordinary_out}" "sd-lora|${sd_adapter}"; do
    IFS='|' read -r method adapter <<<"${method_adapter}"
    receipt="${ROOT}/evidence/reload/${method}-stage$(printf '%02d' "${increment}").json"
    if [[ ! -f "${HOST_ROOT}/evidence/reload/${method}-stage$(printf '%02d' "${increment}").json" ]]; then
      run_gpu "${uuid}" "h138b-reload-${method}-${increment}" "${ROOT}/evidence/reload/${method}-stage$(printf '%02d' "${increment}")" \
        -m scripts.openevo_webshop.verify_adapter_reload --model "${MODEL}" --adapter "${adapter}" --model-revision "${MODEL_REVISION}" \
        --increment-index "${increment}" --output "${receipt}" --require-cuda
    fi
  done
  task="$(jq -r ".training_tasks[${increment}].goal_idx" "${HOST_REPO}/${COMMITTED_MANIFEST}")"
  for method_adapter in "ordinary-lora|${ordinary_out}" "sd-lora|${sd_adapter}"; do
    IFS='|' read -r method adapter <<<"${method_adapter}"
    reload="${HOST_ROOT}/evidence/reload/${method}-stage$(printf '%02d' "${increment}").json"
    file_sha="$(jq -r '.peft_weights_sha256' "${reload}")"; tensor_sha="$(jq -r '.first_load_digest' "${reload}")"
    safe_time; within_cap; uuid="$(one_idle_gpu)"
    eval_cell acquisition "${method}" "${task}" "${adapter}" "${file_sha}" "${tensor_sha}" "${uuid}" "${increment}" 2026090600 2026090601
  done
done

qualification_args=()
for increment in $(seq 0 7); do
  qualification_args+=(--ordinary-evidence "${ROOT}/evidence/training/ordinary-lora-stage$(printf '%02d' "${increment}").json")
  qualification_args+=(--sd-evidence "${ROOT}/evidence/training/sd-lora-stage$(printf '%02d' "${increment}").json")
  qualification_args+=(--ordinary-reload "${ROOT}/evidence/reload/ordinary-lora-stage$(printf '%02d' "${increment}").json")
  qualification_args+=(--sd-reload "${ROOT}/evidence/reload/sd-lora-stage$(printf '%02d' "${increment}").json")
done
utility -m scripts.openevo_webshop.verify_h138b_qualification --plans "${PLANS}" "${qualification_args[@]}" --output "${ROOT}/evidence/h1.38b-qualification.json" || die "qualification failed; fresh panel remains untouched"
mkdir -p "${HOST_ROOT}/evidence/training-receipts"
"${PYTHON}" - "${HOST_ROOT}/evidence/h1.38b-qualification.json" "${HOST_ROOT}/evidence/training-receipts" <<'PY'
import json
import sys
from pathlib import Path

qualification_path = Path(sys.argv[1])
receipt_root = Path(sys.argv[2])
payload = json.loads(qualification_path.read_text(encoding="utf-8"))
for receipt in payload["training_receipts"]:
    arm = str(receipt["arm"])
    increment = int(receipt["increment_index"])
    output = receipt_root / f"{arm}-stage{increment:02d}.json"
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
PY
chown -R 1001:1001 "${HOST_ROOT}/evidence/training-receipts"

matrix_stage() {
  local stage="$1"; shift; local seeds=("$1" "$2" "$3" "$4"); shift 4
  local tasks=("$@") pending=() task arm adapter file_sha tensor_sha
  for arm in base ordinary-lora sd-lora; do for task in "${tasks[@]}"; do pending+=("${arm}|${task}"); done; done
  while ((${#pending[@]})); do
    safe_time; within_cap; mapfile -t available < <(idle_gpus)
    ((${#available[@]})) || { sleep 60; continue; }
    wave=("${pending[@]:0:${#available[@]}}")
    pending=("${pending[@]:${#wave[@]}}")
    pids=()
    for index in "${!wave[@]}"; do
      IFS='|' read -r arm task <<<"${wave[$index]}"; uuid="${available[$index]}"
      if [[ "${arm}" == base ]]; then adapter=none; file_sha=""; tensor_sha=""; else
        adapter="$(jq -r ".final_adapters[\"${arm}\"].path" "${HOST_ROOT}/evidence/h1.38b-qualification.json")"
        file_sha="$(jq -r ".final_adapters[\"${arm}\"].adapter_file_sha256" "${HOST_ROOT}/evidence/h1.38b-qualification.json")"
        tensor_sha="$(jq -r ".final_adapters[\"${arm}\"].evaluation_tensor_digest" "${HOST_ROOT}/evidence/h1.38b-qualification.json")"
      fi
      eval_cell "${stage}" "${arm}" "${task}" "${adapter}" "${file_sha}" "${tensor_sha}" "${uuid}" -1 "${seeds[@]}" > "${HOST_ROOT}/logs/${stage}-${arm}-${task}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}" || die "formal ${stage} wave failed; no cell is retried"; done
  done
}

mapfile -t retention_tasks < <(jq -r '.training_tasks[].goal_idx' "${HOST_REPO}/${COMMITTED_MANIFEST}")
matrix_stage retention 2026090610 2026090611 2026090612 2026090613 "${retention_tasks[@]}"
mapfile -t fresh_tasks < <(jq -r '.fresh_tasks[].goal_idx' "${HOST_REPO}/${COMMITTED_MANIFEST}")
matrix_stage fresh 2026090620 2026090621 2026090622 2026090623 "${fresh_tasks[@]}"

receipt_args=()
for receipt in "${HOST_ROOT}"/evidence/training-receipts/*.json; do receipt_args+=(--training-evidence "${ROOT}/${receipt#"${HOST_ROOT}"/}"); done
utility -m scripts.openevo_webshop.reconcile_h138b_method_control --manifest "${MANIFEST}" --run-root "${ROOT}" "${receipt_args[@]}" --output "${ROOT}/evidence/h1.38b-reconciliation.json"
attach_gpu_job_accounting "${HOST_ROOT}/evidence/h1.38b-reconciliation.json" || die "failed to attach GPU job accounting to reconciliation"
printf 'H1.38B formal launcher reached reconciliation at %s; GPU-hours=%s\n' "$(date --iso-8601=seconds)" "$(gpu_hours)"
