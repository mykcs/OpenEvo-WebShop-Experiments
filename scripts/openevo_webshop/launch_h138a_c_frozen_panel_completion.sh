#!/usr/bin/env bash
set -euo pipefail

# H1.38A-C: post-closeout completion of only the original frozen D8/D16 cells.
# This launcher never trains, never rewrites the original run, and fails closed
# on any provenance drift or partial cell.
CAMPAIGN_ID="20260819-0215-h138a-c-frozen-panel-completion"
RUN_NAME="phase-h1.38a-c-frozen-panel-completion-v1-20260819"
EXECUTION_SHA="${OPENEVO_H138A_C_EXECUTION_SHA:?set to pushed continuation SHA}"
HOST_REPO="${OPENEVO_H138A_C_HOST_REPO:?set to exact server checkout}"
HOST_RUNS="${OPENEVO_H138A_C_HOST_RUNS:-/workspace/runs}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
ROOT="/workspace/runs/${RUN_NAME}"
ORIGINAL_HOST_ROOT="${HOST_RUNS}/phase-h1.38a-pooled-diversity-v1-20260819"
ORIGINAL_ROOT="/workspace/runs/phase-h1.38a-pooled-diversity-v1-20260819"
IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
PYTHON_FREEZE_SHA="e3cb53bced07b519157ceca2698935bf0be45a58d08465c87c618ef62c31ee18"
CORE_HOST="/workspace/migration/runtime-src-snapshot/openevo-core"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
MODEL="/workspace/models/qwen2.5-7b-instruct"
MODEL_HOST="/workspace/models/qwen2.5-7b-instruct"
MODEL_REVISION="0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
CONFIG="configs/experiment/h1.38a-c-frozen-panel-completion-v1.json"
CONFIG_HOST="${HOST_REPO}/${CONFIG}"
ORCHESTRATION_SHA="$(jq -r '.continuation_orchestration_sha' "${CONFIG_HOST}")"
MANIFEST_HOST="${ORIGINAL_HOST_ROOT}/manifests/h1.38a-evaluation-manifest.json"
MANIFEST="${ORIGINAL_ROOT}/manifests/h1.38a-evaluation-manifest.json"
EVAL_SCRIPT_HOST="${HOST_REPO}/scripts/openevo_webshop/h1_closed_loop_tracked.py"
EVAL_SCRIPT="/workspace/repo/scripts/openevo_webshop/h1_closed_loop_tracked.py"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-$(jq -r '.tracking.default_mode' "${CONFIG_HOST}")}"
REQUESTED_TRACKING_MODE="${TRACKING_MODE}"
PROBE_FILENAME="wandb-online-probe-${EXECUTION_SHA}.json"
PROBE_PATH="${HOST_ROOT}/evidence/${PROBE_FILENAME}"
PROBE_RELATIVE="evidence/${PROBE_FILENAME}"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
JDK="/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1"
GPU_UUIDS=(GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000)
SEEDS=(2026090500 2026090501 2026090502 2026090503)
CAP=3.0

die() { printf 'H1.38A-C gate failed: %s\n' "$*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
adapter_digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'); }

[[ "$(hostname)" == experiment-host && "$(id -u)" == 0 ]] || die "expected root@experiment-host"
[[ "${HOST_REPO}" == /workspace/* ]] || die "host checkout outside approved workspace"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "continuation checkout SHA mismatch"
[[ "${ORCHESTRATION_SHA}" =~ ^[0-9a-f]{40}$ ]] || die "continuation orchestration SHA missing"
git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" show "${ORCHESTRATION_SHA}:scripts/openevo_webshop/launch_h138a_c_frozen_panel_completion.sh" >/dev/null || die "continuation orchestration SHA unavailable"
git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" merge-base --is-ancestor "${ORCHESTRATION_SHA}" "${EXECUTION_SHA}" || die "continuation checkout does not descend from frozen orchestration"
[[ -f "${CONFIG_HOST}" && -f "${MANIFEST_HOST}" && -f "${EVAL_SCRIPT_HOST}" ]] || die "required source file missing"
ORIGINAL_RECONCILIATION_REL="$(jq -r '.original_campaign.reconciliation_evidence_path' "${CONFIG_HOST}")"
ORIGINAL_CLOSEOUT_REL="$(jq -r '.original_campaign.closeout_summary_evidence_path' "${CONFIG_HOST}")"
ORIGINAL_WANDB_PROBE_REL="$(jq -r '.original_campaign.wandb_online_probe_evidence_path' "${CONFIG_HOST}")"
SEALED_RECEIPTS_REL="$(jq -r '.tracking.sealed_fallback_receipt_manifest.path' "${CONFIG_HOST}")"
[[ "${ORIGINAL_RECONCILIATION_REL}" == docs/evidence/* && "${ORIGINAL_RECONCILIATION_REL}" != *..* ]] || die "original reconciliation evidence path unsafe"
[[ "${ORIGINAL_CLOSEOUT_REL}" == docs/evidence/* && "${ORIGINAL_CLOSEOUT_REL}" != *..* ]] || die "original closeout evidence path unsafe"
[[ "${ORIGINAL_WANDB_PROBE_REL}" == docs/evidence/* && "${ORIGINAL_WANDB_PROBE_REL}" != *..* ]] || die "original W&B probe evidence path unsafe"
[[ "${SEALED_RECEIPTS_REL}" == docs/evidence/* && "${SEALED_RECEIPTS_REL}" != *..* ]] || die "sealed fallback receipt manifest path unsafe"
[[ -f "${HOST_REPO}/${ORIGINAL_RECONCILIATION_REL}" && ! -L "${HOST_REPO}/${ORIGINAL_RECONCILIATION_REL}" ]] || die "original reconciliation evidence missing or unsafe"
[[ -f "${HOST_REPO}/${ORIGINAL_CLOSEOUT_REL}" && ! -L "${HOST_REPO}/${ORIGINAL_CLOSEOUT_REL}" ]] || die "original closeout evidence missing or unsafe"
[[ -f "${HOST_REPO}/${ORIGINAL_WANDB_PROBE_REL}" && ! -L "${HOST_REPO}/${ORIGINAL_WANDB_PROBE_REL}" ]] || die "original W&B probe evidence missing or unsafe"
[[ -f "${HOST_REPO}/${SEALED_RECEIPTS_REL}" && ! -L "${HOST_REPO}/${SEALED_RECEIPTS_REL}" ]] || die "sealed fallback receipt manifest missing or unsafe"
git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" cat-file -e "${EXECUTION_SHA}:${SEALED_RECEIPTS_REL}" || die "sealed fallback receipt manifest absent from execution commit"
git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" diff --quiet "${EXECUTION_SHA}" -- "${CONFIG}" "configs/experiment/current-campaign.json" "scripts/openevo_webshop/analyze_h138a_c_combined.py" "scripts/openevo_webshop/launch_h138a_c_frozen_panel_completion.sh" "${SEALED_RECEIPTS_REL}" || die "continuation control files differ from execution commit"
EVIDENCE_ROOT_REAL="$(realpath -e "${HOST_REPO}/docs/evidence")"
HOST_REPO_REAL="$(realpath -e "${HOST_REPO}")"
ORIGINAL_RECONCILIATION_REAL="$(realpath -e "${HOST_REPO}/${ORIGINAL_RECONCILIATION_REL}")"
ORIGINAL_CLOSEOUT_REAL="$(realpath -e "${HOST_REPO}/${ORIGINAL_CLOSEOUT_REL}")"
ORIGINAL_WANDB_PROBE_REAL="$(realpath -e "${HOST_REPO}/${ORIGINAL_WANDB_PROBE_REL}")"
SEALED_RECEIPTS_REAL="$(realpath -e "${HOST_REPO}/${SEALED_RECEIPTS_REL}")"
[[ "${EVIDENCE_ROOT_REAL}" == "${HOST_REPO_REAL}/docs/evidence" ]] || die "repository evidence root escaped checkout"
[[ "${ORIGINAL_RECONCILIATION_REAL}" == "${EVIDENCE_ROOT_REAL}/"* ]] || die "original reconciliation evidence escaped evidence root"
[[ "${ORIGINAL_CLOSEOUT_REAL}" == "${EVIDENCE_ROOT_REAL}/"* ]] || die "original closeout evidence escaped evidence root"
[[ "${ORIGINAL_WANDB_PROBE_REAL}" == "${EVIDENCE_ROOT_REAL}/"* ]] || die "original W&B probe evidence escaped evidence root"
[[ "${SEALED_RECEIPTS_REAL}" == "${EVIDENCE_ROOT_REAL}/"* ]] || die "sealed fallback receipt manifest escaped evidence root"
[[ "$(sha256_file "${MANIFEST_HOST}")" == "$(jq -r '.original_campaign.evaluation_manifest_sha256' "${CONFIG_HOST}")" ]] || die "original manifest SHA mismatch"
[[ "$(sha256_file "${ORIGINAL_RECONCILIATION_REAL}")" == "$(jq -r '.original_campaign.reconciliation_sha256' "${CONFIG_HOST}")" ]] || die "original reconciliation SHA mismatch"
[[ "$(sha256_file "${ORIGINAL_CLOSEOUT_REAL}")" == "$(jq -r '.original_campaign.closeout_summary_sha256' "${CONFIG_HOST}")" ]] || die "original closeout summary SHA mismatch"
[[ "$(sha256_file "${ORIGINAL_WANDB_PROBE_REAL}")" == "$(jq -r '.original_campaign.wandb_online_probe_sha256' "${CONFIG_HOST}")" ]] || die "original W&B probe SHA mismatch"
[[ "$(sha256_file "${SEALED_RECEIPTS_REAL}")" == "$(jq -r '.tracking.sealed_fallback_receipt_manifest.sha256' "${CONFIG_HOST}")" ]] || die "sealed fallback receipt manifest SHA mismatch"
[[ "$(wc -l < "${SEALED_RECEIPTS_REAL}" | tr -d ' ')" == "$(jq -r '.tracking.sealed_fallback_receipt_manifest.count' "${CONFIG_HOST}")" ]] || die "sealed fallback receipt manifest count mismatch"
jq -e '.frozen_source_split.original == {"base":48,"diversity-4":48,"diversity-8":10,"diversity-16":0} and .frozen_source_split.continuation == {"base":0,"diversity-4":0,"diversity-8":38,"diversity-16":48} and ((.original_campaign.completed_diversity_8_task_ids | length) == 10) and ((.original_campaign.completed_diversity_8_task_ids | unique | length) == 10)' "${CONFIG_HOST}" >/dev/null || die "frozen source split mismatch"
jq -e --arg model "${MODEL}" --arg revision "${MODEL_REVISION}" --argjson seeds "$(printf '%s\n' "${SEEDS[@]}" | jq -R 'tonumber' | jq -s .)" --argjson uuids "$(printf '%s\n' "${GPU_UUIDS[@]}" | jq -R . | jq -s .)" '.frozen_provenance.evaluation_model_path == $model and .frozen_provenance.model_content_pin == $revision and .frozen_provenance.evaluation_environment_seed == 2026 and .frozen_provenance.rollout_seeds == $seeds and .frozen_provenance.temperature == 0 and .frozen_provenance.max_steps == 30 and .frozen_provenance.max_new_tokens == 512 and .frozen_provenance.prompt_contract == "seed3090.webshop-compact-action-prompt.v2" and .frozen_provenance.action_contract == "seed3090.webshop-action-parser.v3-action-command" and .authorized_gpu_uuids == $uuids and .tracking.default_mode == "online" and .tracking.online_probe_required == true' "${CONFIG_HOST}" >/dev/null || die "frozen evaluation control contract mismatch"
[[ "$(sha256_file "${EVAL_SCRIPT_HOST}")" == "$(jq -r '.frozen_provenance.evaluation_script_sha256' "${CONFIG_HOST}")" ]] || die "evaluation script drift"
[[ "$(sha256_file "${HOST_REPO}/configs/experiment/h1.38a-pooled-diversity-v1.json")" == "$(jq -r '.frozen_provenance.original_config_sha256' "${CONFIG_HOST}")" ]] || die "original config drift"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" show -s --format=%H "$(jq -r '.original_campaign.scientific_execution_sha' "${CONFIG_HOST}")")" == "$(jq -r '.original_campaign.scientific_execution_sha' "${CONFIG_HOST}")" ]] || die "original scientific SHA unavailable"
[[ "$(jq -r '.campaign_id' "${HOST_REPO}/configs/experiment/current-campaign.json")" == "${CAMPAIGN_ID}" ]] || die "continuation is not current campaign"
[[ "$(jq -r '.status' "${HOST_REPO}/configs/experiment/current-campaign.json")" == "launch-authorized" && "$(jq -r '.formal_task_consumption_allowed' "${HOST_REPO}/configs/experiment/current-campaign.json")" == true ]] || die "continuation campaign is not launch-authorized"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo SHA mismatch"
[[ "$("${PYTHON}" -m pip freeze --all | LC_ALL=C sort | sha256sum | awk '{print $1}')" == "${PYTHON_FREEZE_SHA}" ]] || die "Python freeze mismatch"
for f in config.json model.safetensors.index.json tokenizer_config.json tokenizer.json; do [[ -f "${MODEL_HOST}/${f}" ]] || die "missing model component ${f}"; done
cfg="$(sha256_file "${MODEL_HOST}/config.json")"; idx="$(sha256_file "${MODEL_HOST}/model.safetensors.index.json")"; tok="$(sha256_file "${MODEL_HOST}/tokenizer_config.json")"
[[ "$(printf '%s\n%s\n%s\n' "${cfg}" "${idx}" "${tok}" | sha256sum | awk '{print $1}')" == "${MODEL_REVISION}" ]] || die "model content pin mismatch"

for arm in diversity-4 diversity-8 diversity-16; do
  evidence="${HOST_REPO}/docs/evidence/remote-runs/2026-08-19/h1.38a/evidence/${arm}-training.json"
  expected="$(jq -r ".frozen_provenance.adapters[\"${arm}\"].training_evidence_sha256" "${CONFIG_HOST}")"
  [[ "$(sha256_file "${evidence}")" == "${expected}" ]] || die "${arm} training evidence drift"
  uri="$(jq -r '.adapter_uri' "${evidence}")"
  [[ "${uri}" == file:///workspace/runs/* ]] || die "${arm} adapter URI escaped run root"
  [[ "${uri#file://}" == "$(jq -r ".frozen_provenance.adapters[\"${arm}\"].adapter_uri" "${CONFIG_HOST}")" ]] || die "${arm} adapter URI drift"
  rel="${uri#file:///workspace/runs/}"
  adapter_host="${HOST_RUNS}/${rel}"
  [[ -d "${adapter_host}" ]] || die "${arm} adapter missing"
  [[ "$(adapter_digest "${adapter_host}")" == "$(jq -r ".frozen_provenance.adapters[\"${arm}\"].adapter_digest" "${CONFIG_HOST}")" ]] || die "${arm} adapter digest drift"
done

mkdir -p "${HOST_ROOT}"/{manifests,evaluation,evidence}
chown -R 1001:1001 "${HOST_ROOT}"
if [[ -f "${HOST_ROOT}/evidence/provenance.json" ]]; then
  jq -e --arg campaign "${CAMPAIGN_ID}" --arg orchestration_sha "${ORCHESTRATION_SHA}" --arg original_sha "$(jq -r '.original_campaign.scientific_execution_sha' "${CONFIG_HOST}")" --arg manifest "$(sha256_file "${MANIFEST_HOST}")" --arg root "${HOST_ROOT}" --argjson uuids "$(printf '%s\n' "${GPU_UUIDS[@]}" | jq -R . | jq -s .)" '.schema == "openevo.webshop-h1.38a-c-provenance.v1" and .campaign_id == $campaign and .orchestration_sha == $orchestration_sha and .original_scientific_sha == $original_sha and .original_manifest_sha256 == $manifest and .continuation_root == $root and .authorized_gpu_uuids == $uuids and .selection_policy == "fixed missing D8/D16 cells only" and (.continuation_execution_sha | type == "string")' "${HOST_ROOT}/evidence/provenance.json" >/dev/null || die "existing continuation provenance mismatch"
  RECORDED_EXECUTION_SHA="$(jq -r '.continuation_execution_sha' "${HOST_ROOT}/evidence/provenance.json")"
  [[ "${RECORDED_EXECUTION_SHA}" =~ ^[0-9a-f]{40}$ ]] || die "existing continuation execution SHA invalid"
  git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" show -s --format=%H "${RECORDED_EXECUTION_SHA}" >/dev/null || die "existing continuation execution SHA unavailable"
  git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" merge-base --is-ancestor "${ORCHESTRATION_SHA}" "${RECORDED_EXECUTION_SHA}" || die "existing continuation execution predates frozen orchestration"
  git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" merge-base --is-ancestor "${RECORDED_EXECUTION_SHA}" "${EXECUTION_SHA}" || die "current continuation checkout does not descend from recorded execution"
fi
EXECUTION_PROVENANCE_PATH="${HOST_ROOT}/evidence/provenance-${EXECUTION_SHA}.json"
if [[ -f "${EXECUTION_PROVENANCE_PATH}" ]]; then
  jq -e --arg campaign "${CAMPAIGN_ID}" --arg orchestration_sha "${ORCHESTRATION_SHA}" --arg execution_sha "${EXECUTION_SHA}" --arg original_sha "$(jq -r '.original_campaign.scientific_execution_sha' "${CONFIG_HOST}")" --arg manifest "$(sha256_file "${MANIFEST_HOST}")" --arg root "${HOST_ROOT}" --argjson uuids "$(printf '%s\n' "${GPU_UUIDS[@]}" | jq -R . | jq -s .)" '.schema == "openevo.webshop-h1.38a-c-execution-provenance.v1" and .campaign_id == $campaign and .orchestration_sha == $orchestration_sha and .execution_sha == $execution_sha and .original_scientific_sha == $original_sha and .original_manifest_sha256 == $manifest and .continuation_root == $root and .authorized_gpu_uuids == $uuids and .selection_policy == "fixed missing D8/D16 cells only"' "${EXECUTION_PROVENANCE_PATH}" >/dev/null || die "existing per-execution provenance mismatch"
else
  provenance_tmp="${EXECUTION_PROVENANCE_PATH}.tmp.$$"
  jq -n --arg campaign "${CAMPAIGN_ID}" --arg orchestration_sha "${ORCHESTRATION_SHA}" --arg execution_sha "${EXECUTION_SHA}" --arg original_sha "$(jq -r '.original_campaign.scientific_execution_sha' "${CONFIG_HOST}")" --arg manifest "$(sha256_file "${MANIFEST_HOST}")" --arg root "${HOST_ROOT}" --argjson uuids "$(printf '%s\n' "${GPU_UUIDS[@]}" | jq -R . | jq -s .)" '{schema:"openevo.webshop-h1.38a-c-execution-provenance.v1",campaign_id:$campaign,orchestration_sha:$orchestration_sha,execution_sha:$execution_sha,original_scientific_sha:$original_sha,original_manifest_sha256:$manifest,continuation_root:$root,authorized_gpu_uuids:$uuids,selection_policy:"fixed missing D8/D16 cells only"}' > "${provenance_tmp}"
  chown 1001:1001 "${provenance_tmp}"
  mv -n "${provenance_tmp}" "${EXECUTION_PROVENANCE_PATH}"
  [[ ! -e "${provenance_tmp}" ]] || die "per-execution provenance already exists"
fi
for execution_provenance in "${HOST_ROOT}"/evidence/provenance-*.json; do
  [[ -e "${execution_provenance}" ]] || continue
  [[ -f "${execution_provenance}" && ! -L "${execution_provenance}" ]] || die "unsafe per-execution provenance"
  provenance_execution_sha="$(jq -r '.execution_sha' "${execution_provenance}")"
  [[ "$(basename "${execution_provenance}")" == "provenance-${provenance_execution_sha}.json" ]] || die "per-execution provenance filename mismatch"
  jq -e --arg campaign "${CAMPAIGN_ID}" --arg orchestration_sha "${ORCHESTRATION_SHA}" --arg execution_sha "${provenance_execution_sha}" --arg original_sha "$(jq -r '.original_campaign.scientific_execution_sha' "${CONFIG_HOST}")" --arg manifest "$(sha256_file "${MANIFEST_HOST}")" --arg root "${HOST_ROOT}" --argjson uuids "$(printf '%s\n' "${GPU_UUIDS[@]}" | jq -R . | jq -s .)" '.schema == "openevo.webshop-h1.38a-c-execution-provenance.v1" and .campaign_id == $campaign and .orchestration_sha == $orchestration_sha and .execution_sha == $execution_sha and .original_scientific_sha == $original_sha and .original_manifest_sha256 == $manifest and .continuation_root == $root and .authorized_gpu_uuids == $uuids and .selection_policy == "fixed missing D8/D16 cells only"' "${execution_provenance}" >/dev/null || die "per-execution provenance mismatch"
  git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" cat-file -e "${provenance_execution_sha}^{commit}" || die "per-execution provenance commit unavailable"
  git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" merge-base --is-ancestor "${ORCHESTRATION_SHA}" "${provenance_execution_sha}" || die "per-execution provenance predates frozen orchestration"
  git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" merge-base --is-ancestor "${provenance_execution_sha}" "${EXECUTION_SHA}" || die "per-execution provenance is outside current execution chain"
done
cp "${MANIFEST_HOST}" "${HOST_ROOT}/manifests/h1.38a-evaluation-manifest.json"
chown 1001:1001 "${HOST_ROOT}/manifests/h1.38a-evaluation-manifest.json"

utility() { docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=2g --shm-size=2g --user 1001:1001 -v "${HOST_REPO}":/workspace/repo:ro -v "${GIT_COMMON_HOST}:${GIT_COMMON_HOST}:ro" -v "${HOST_RUNS}":/workspace/runs -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -w /workspace/repo -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/repo -e JAVA_HOME="${JDK}" -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src -e PYTHONDONTWRITEBYTECODE=1 --entrypoint "${PYTHON}" "${IMAGE}" "$@"; }
GIT_COMMON_HOST="$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse --git-common-dir)"
[[ "${GIT_COMMON_HOST}" == /workspace/* && -d "${GIT_COMMON_HOST}" ]] || die "unexpected Git common directory"

idle_gpus() { local u line mem util apps; for u in "${GPU_UUIDS[@]}"; do line="$(nvidia-smi -i "${u}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" || continue; IFS=',' read -r mem util <<<"${line}"; mem="${mem//[[:space:]]/}"; util="${util//[[:space:]]/}"; apps="$(nvidia-smi -i "${u}" --query-compute-apps=pid --format=csv,noheader,nounits)"; [[ "${mem}" =~ ^[0-9]+$ && "${mem}" -le 10 && "${util}" == 0 && -z "${apps}" ]] && printf '%s\n' "${u}"; done; return 0; }
gpu_hours() { local files; files="$(find "${HOST_ROOT}/evaluation" -name summary.json -type f -print -quit)"; [[ -n "${files}" ]] || { printf '0\n'; return; }; find "${HOST_ROOT}/evaluation" -name summary.json -type f -print0 | xargs -0 jq -s '[.[].wall_time_seconds // 0] | add / 3600'; }
within_cap() { jq -e --argjson used "$(gpu_hours)" '.continuation_budget.gpu_hours_cap > $used' "${CONFIG_HOST}" >/dev/null || die "continuation GPU-hour cap reached"; }
complete_cell() { local arm="$1" task="$2" path="${3:-${HOST_ROOT}/evaluation/${1}/task${2}}"; [[ ! -e "${path}" ]] && return 1; [[ -d "${path}" ]] || die "invalid continuation cell path ${arm}/${task}"; PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" "${HOST_REPO}/scripts/openevo_webshop/analyze_h138a_c_combined.py" --manifest "${MANIFEST_HOST}" --config "${CONFIG_HOST}" --validate-cell "${path}" --arm "${arm}" --task "${task}" >/dev/null || die "invalid completed cell ${arm}/${task}"; }

PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" "${HOST_REPO}/scripts/openevo_webshop/analyze_h138a_c_combined.py" --manifest "${MANIFEST_HOST}" --config "${CONFIG_HOST}" --validate-source original --source-root "${ORIGINAL_HOST_ROOT}" >/dev/null || die "original source validation failed"
PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" "${HOST_REPO}/scripts/openevo_webshop/analyze_h138a_c_combined.py" --manifest "${MANIFEST_HOST}" --config "${CONFIG_HOST}" --validate-source continuation --source-root "${HOST_ROOT}" >/dev/null || die "continuation source validation failed"

if [[ -f "${PROBE_PATH}" ]]; then
  jq -e --arg execution_sha "${EXECUTION_SHA}" '.schema == "openevo.wandb-online-probe.v1" and .execution_sha == $execution_sha and ((.status == "ok" and .effective_mode == "online") or (.status == "failed" and .effective_mode == "offline") or (.status == "skipped" and .effective_mode == "offline"))' "${PROBE_PATH}" >/dev/null || die "existing W&B probe evidence mismatch"
  if [[ "${REQUESTED_TRACKING_MODE}" == offline && "$(jq -r '.status' "${PROBE_PATH}")" != failed && "$(jq -r '.tracking.explicit_offline_allowed' "${CONFIG_HOST}")" != true ]]; then die "explicit offline requires an existing failed online probe or config authorization"; fi
  TRACKING_MODE="$(jq -r '.effective_mode' "${PROBE_PATH}")"
elif [[ "${TRACKING_MODE}" == online ]]; then
  set +e
  probe="$(timeout 45s docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=512m -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -e WANDB_MODE=online -e WANDB_INIT_TIMEOUT=10 -e WANDB_SILENT=true -e WANDB_API_KEY --entrypoint "${PYTHON}" "${IMAGE}" -c 'import json, wandb
try:
 r=wandb.init(project="openevo-webshop-experiments",entity="example-entity",name="h138a-c-online-probe-'"${EXECUTION_SHA:0:12}"'",job_type="preflight",mode="online",reinit="create_new"); print(json.dumps({"status":"ok","run_id":getattr(r,"id",None)})); r.finish()
except BaseException as e: print(json.dumps({"status":"failed","error_type":type(e).__name__,"error":str(e)})); raise' 2>&1)"
  probe_rc=$?
  set -e
  probe_tmp="${PROBE_PATH}.tmp.$$"
  if [[ "${probe_rc}" == 0 && "${probe}" == *'"status": "ok"'* ]]; then
    jq -n --arg execution_sha "${EXECUTION_SHA}" --arg output "${probe}" --argjson exit_code "${probe_rc}" '{schema:"openevo.wandb-online-probe.v1",execution_sha:$execution_sha,requested_mode:"online",effective_mode:"online",status:"ok",exit_code:$exit_code,output:$output}' > "${probe_tmp}"
  else
    jq -n --arg execution_sha "${EXECUTION_SHA}" --arg output "${probe}" --argjson exit_code "${probe_rc}" '{schema:"openevo.wandb-online-probe.v1",execution_sha:$execution_sha,requested_mode:"online",effective_mode:"offline",status:"failed",exit_code:$exit_code,output:$output}' > "${probe_tmp}"
    TRACKING_MODE=offline
  fi
  chown 1001:1001 "${probe_tmp}"
  mv -n "${probe_tmp}" "${PROBE_PATH}"
elif [[ "${TRACKING_MODE}" == offline ]]; then
  [[ "$(jq -r '.tracking.explicit_offline_allowed' "${CONFIG_HOST}")" == true ]] || die "explicit offline requires an existing failed online probe or config authorization"
  OFFLINE_REASON="${OPENEVO_WANDB_OFFLINE_REASON:-}"
  [[ -n "${OFFLINE_REASON}" ]] || die "explicit offline requires OPENEVO_WANDB_OFFLINE_REASON"
  probe_tmp="${PROBE_PATH}.tmp.$$"
  jq -n --arg execution_sha "${EXECUTION_SHA}" --arg reason "${OFFLINE_REASON}" '{schema:"openevo.wandb-online-probe.v1",execution_sha:$execution_sha,requested_mode:"offline",effective_mode:"offline",status:"skipped",exit_code:null,output:$reason}' > "${probe_tmp}"
  chown 1001:1001 "${probe_tmp}"
  mv -n "${probe_tmp}" "${PROBE_PATH}"
else
  die "tracking mode must be online or offline"
fi
[[ -f "${PROBE_PATH}" && ! -L "${PROBE_PATH}" ]] || die "W&B probe evidence missing or unsafe"
jq -e --arg execution_sha "${EXECUTION_SHA}" '.schema == "openevo.wandb-online-probe.v1" and .execution_sha == $execution_sha and (.effective_mode == "online" or .effective_mode == "offline")' "${PROBE_PATH}" >/dev/null || die "sealed W&B probe evidence mismatch"

available=( $(idle_gpus) ); [[ "${#available[@]}" -gt 0 ]] || die "no authorized live-idle GPU"
within_cap
if [[ ! -f "${HOST_ROOT}/evidence/provenance.json" ]]; then
  nvidia_smi_lines=()
  for u in "${GPU_UUIDS[@]}"; do nvidia_smi_lines+=("$(nvidia-smi -i "${u}" --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits)"); done
  printf '%s\n' "${nvidia_smi_lines[@]}" > "${HOST_ROOT}/evidence/gpu-preflight.txt"
  jq -n --arg schema "openevo.webshop-h1.38a-c-provenance.v1" --arg campaign "${CAMPAIGN_ID}" --arg orchestration_sha "${ORCHESTRATION_SHA}" --arg execution_sha "${EXECUTION_SHA}" --arg original_sha "$(jq -r '.original_campaign.scientific_execution_sha' "${CONFIG_HOST}")" --arg manifest_sha "$(sha256_file "${MANIFEST_HOST}")" --arg continuation_root "${HOST_ROOT}" --argjson uuids "$(printf '%s\n' "${GPU_UUIDS[@]}" | jq -R . | jq -s .)" '{schema:$schema,campaign_id:$campaign,orchestration_sha:$orchestration_sha,continuation_execution_sha:$execution_sha,original_scientific_sha:$original_sha,original_manifest_sha256:$manifest_sha,continuation_root:$continuation_root,authorized_gpu_uuids:$uuids,selection_policy:"fixed missing D8/D16 cells only",source_runs:{original:"original H1.38A run root",continuation:"this run root"}}' > "${HOST_ROOT}/evidence/provenance.json"
  chown 1001:1001 "${HOST_ROOT}/evidence/gpu-preflight.txt" "${HOST_ROOT}/evidence/provenance.json"
fi

run_gpu() { local uuid="$1" name="$2" out="$3"; shift 3; mkdir -p "${HOST_ROOT}/${out#/workspace/runs/${RUN_NAME}/}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts}; chown -R 1001:1001 "${HOST_ROOT}/${out#/workspace/runs/${RUN_NAME}/}"; docker run --rm --name "${name}" --read-only --tmpfs /tmp:rw,nosuid,size=4g --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 --gpus "device=${uuid}" --user 1001:1001 -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -v "${HOST_REPO}":/workspace/repo:ro -v "${GIT_COMMON_HOST}:${GIT_COMMON_HOST}:ro" -v "${CORE_HOST}":/workspace/openevo-core:ro -v /workspace/models:/workspace/models:ro -v /workspace/models:/workspace/workspace/models:ro -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro -v "${HOST_RUNS}":/workspace/runs -w /workspace/repo -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/repo -e WANDB_API_KEY -e WANDB_ENTITY=example-entity -e WANDB_PROJECT=openevo-webshop-experiments -e WANDB_MODE="${TRACKING_MODE}" -e CUDA_VISIBLE_DEVICES="${uuid}" -e TOKENIZERS_PARALLELISM=false -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e JAVA_HOME="${JDK}" -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" -e PYTHONPATH=/workspace/openevo-core/src:/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src -e PYTHONDONTWRITEBYTECODE=1 -e HOME="${out}/home" -e TMPDIR="${out}/tmp" -e XDG_CACHE_HOME="${out}/xdg-cache" -e HF_HOME=/workspace/hf-cache -e WANDB_DIR="${out}/wandb" -e WANDB_CACHE_DIR="${out}/wandb-cache" -e WANDB_CONFIG_DIR="${out}/wandb-config" -e WANDB_ARTIFACT_DIR="${out}/wandb-artifacts" --entrypoint "${PYTHON}" "${IMAGE}" "$@"; }
eval_cell() { local arm="$1" task="$2" adapter="$3" uuid="$4" out="${ROOT}/evaluation/${arm}/task${task}"; complete_cell "${arm}" "${task}" && return 0; [[ ! -e "${HOST_ROOT}/evaluation/${arm}/task${task}" ]] || die "partial continuation cell ${arm}/${task}"; within_cap; run_gpu "${uuid}" "h138a-c-${arm}-${task}" "${out}" -m scripts.openevo_webshop.h1_closed_loop_tracked --task-manifest "${MANIFEST}" --seed-root /workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1 --model "${MODEL}" --model-revision "${MODEL_REVISION}" --adapter "${adapter}" --arm "${arm}" --tasks "${task}" --rollout-seeds "${SEEDS[@]}" --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature 0 --max-steps 30 --max-new 512 --prompt-contract seed3090.webshop-compact-action-prompt.v2 --action-contract seed3090.webshop-action-parser.v3-action-command --wandb-mode "${TRACKING_MODE}" --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb"; complete_cell "${arm}" "${task}"; }

utility -m unittest discover -s tests -p 'test_*.py'
mapfile -t tasks < <(jq -r '.evaluation_tasks[]' "${MANIFEST_HOST}")
mapfile -t original_d8_tasks < <(jq -r '.original_campaign.completed_diversity_8_task_ids[]' "${CONFIG_HOST}")
is_original_d8() { local candidate="$1" item; for item in "${original_d8_tasks[@]}"; do [[ "${candidate}" == "${item}" ]] && return 0; done; return 1; }
d8_uri="$(jq -r '.adapter_uri' "${HOST_REPO}/docs/evidence/remote-runs/2026-08-19/h1.38a/evidence/diversity-8-training.json")"; d8_host="${HOST_RUNS}/${d8_uri#file:///workspace/runs/}"; d8="/workspace/runs/${d8_uri#file:///workspace/runs/}"
d16_uri="$(jq -r '.adapter_uri' "${HOST_REPO}/docs/evidence/remote-runs/2026-08-19/h1.38a/evidence/diversity-16-training.json")"; d16_host="${HOST_RUNS}/${d16_uri#file:///workspace/runs/}"; d16="/workspace/runs/${d16_uri#file:///workspace/runs/}"
for arm_adapter in "diversity-8|${d8}" "diversity-16|${d16}"; do IFS='|' read -r arm adapter <<<"${arm_adapter}"; for task in "${tasks[@]}"; do if [[ "${arm}" == diversity-8 ]] && is_original_d8 "${task}"; then continue; fi; [[ ! -e "${ORIGINAL_HOST_ROOT}/evaluation/${arm}/task${task}" ]] || die "forbidden original source cell ${arm}/${task}"; mapfile -t available < <(idle_gpus); [[ "${#available[@]}" -gt 0 ]] || die "no authorized live-idle GPU for ${arm}/${task}"; eval_cell "${arm}" "${task}" "${adapter}" "${available[0]}"; done; done

utility -m scripts.openevo_webshop.analyze_h138a_c_combined --manifest "${MANIFEST}" --config "/workspace/repo/${CONFIG}" --original-root "${ORIGINAL_ROOT}" --continuation-root "${ROOT}" --output "${ROOT}/evidence/h1.38a-c-combined-reconciliation.json" >/dev/null || die "combined reconciliation invalid"
RECONCILIATION_PATH="${HOST_ROOT}/evidence/h1.38a-c-combined-reconciliation.json"
tracking_payload="$(jq -c '.continuation_tracking' "${RECONCILIATION_PATH}")"
jq -e '.summary_count == 86 and (.mode_counts.online + .mode_counts.offline == 86)' <<<"${tracking_payload}" >/dev/null || die "continuation tracking summary coverage invalid"
effective_tracking_mode="$(jq -r '.effective_mode' <<<"${tracking_payload}")"
spool_manifest_relative=""
spool_manifest_sha256=""
spool_count=0
if [[ "${effective_tracking_mode}" == offline || "${effective_tracking_mode}" == mixed ]]; then
  spool_manifest="${HOST_ROOT}/evidence/wandb-offline-spool.sha256"
  jq -r '.offline_spools[] | "\(.sha256 | sub("^sha256:"; ""))  \(.path)"' <<<"${tracking_payload}" > "${spool_manifest}"
  spool_count="$(wc -l < "${spool_manifest}" | tr -d ' ')"
  [[ "${spool_count}" -gt 0 ]] || die "offline W&B spool missing"
  spool_manifest_relative="evidence/wandb-offline-spool.sha256"
  spool_manifest_sha256="$(sha256_file "${spool_manifest}")"
  chown 1001:1001 "${spool_manifest}"
fi
probe_sha256="$(sha256_file "${PROBE_PATH}")"
jq -n --arg requested_mode "${REQUESTED_TRACKING_MODE}" --arg effective_mode "${effective_tracking_mode}" --arg probe_path "${PROBE_RELATIVE}" --arg probe_sha256 "${probe_sha256}" --arg spool_manifest "${spool_manifest_relative}" --arg spool_manifest_sha256 "${spool_manifest_sha256}" --argjson spool_file_count "${spool_count}" --argjson tracking "${tracking_payload}" '{schema:"openevo.wandb-fallback-receipt.v1",requested_mode:$requested_mode,effective_mode:$effective_mode,status:"completed",online_probe:{path:$probe_path,sha256:$probe_sha256},tracking:$tracking,run_ids:$tracking.run_ids,fallback_receipts:$tracking.fallback_receipts,spool_manifest:(if $spool_manifest=="" then null else $spool_manifest end),spool_file_count:$spool_file_count,spool_manifest_sha256:(if $spool_manifest_sha256=="" then null else $spool_manifest_sha256 end)}' > "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
chown 1001:1001 "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
printf 'H1.38A-C launcher completed or reached cap under %s\n' "${HOST_ROOT}"
