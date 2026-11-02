#!/usr/bin/env bash
set -euo pipefail

# H1.38A: one selector-chosen, pooled 4/8/16 identity comparison.  This script
# is deliberately non-resumptive for partial cells: a partial formal cell is a
# reconciliation failure, never a reason to retry or substitute data.
CAMPAIGN_ID="20260819-0015-h138a-pooled-diversity"
RUN_NAME="phase-h1.38a-pooled-diversity-v1-20260819"
EXECUTION_SHA="${OPENEVO_H138A_EXECUTION_SHA:?set to the pushed H1.38A commit SHA}"
HOST_REPO="${OPENEVO_H138A_HOST_REPO:?set to exact server checkout}"
HOST_RUNS="${OPENEVO_H138A_HOST_RUNS:-/workspace/runs}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"; ROOT="/workspace/runs/${RUN_NAME}"
IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
PYTHON_FREEZE_SHA="e3cb53bced07b519157ceca2698935bf0be45a58d08465c87c618ef62c31ee18"
CORE_HOST="/workspace/migration/runtime-src-snapshot/openevo-core"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
MODEL="/workspace/models/qwen2.5-7b-instruct"; MODEL_HOST="/workspace/models/qwen2.5-7b-instruct"
MODEL_REVISION="0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
SEED_HOST="${HOST_RUNS}/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
RESERVOIR_HOST="${HOST_RUNS}/phase-h1.38-endogenous-reservoir-v2-20260818"
RESERVOIR_ROOT="/workspace/runs/phase-h1.38-endogenous-reservoir-v2-20260818"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
JDK="/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1"
CONFIG="configs/experiment/h1.38a-pooled-diversity-v1.json"
ABORTED_LEDGER="/workspace/repo/configs/experiment/h1.38-reservoir-v1-aborted-preprotocol-ledger.json"
ANALYSIS="/workspace/repo/docs/evidence/remote-runs/2026-08-18/h1.38-reservoir/evidence/h1.38-endogenous-reservoir-analysis.json"
MANIFEST="${ROOT}/manifests/h1.38a-evaluation-manifest.json"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"
GPU_UUIDS=(GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000)

die() { printf 'H1.38A gate failed: %s\n' "$*" >&2; exit 2; }
[[ "$(hostname)" == experiment-host && "$(id -u)" == 0 ]] || die "expected root@experiment-host"
[[ "${HOST_REPO}" == /workspace/* ]] || die "host checkout outside approved workspace"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "checkout SHA mismatch"
GIT_COMMON_HOST="$(git -C "${HOST_REPO}" rev-parse --git-common-dir)"
[[ "${GIT_COMMON_HOST}" == /workspace/* && -d "${GIT_COMMON_HOST}" ]] || die "unexpected Git common directory"
jq -e --arg id "${CAMPAIGN_ID}" '.campaign_id == $id and .phase == "H1.38A" and .status == "launch-authorized" and .formal_task_consumption_allowed == true' "${HOST_REPO}/configs/experiment/current-campaign.json" >/dev/null || die "campaign not authorized"
[[ "$(jq -r '.frozen_contract.trainer_base_model_argument' "${HOST_REPO}/${CONFIG}")" == "Qwen/Qwen2.5-7B-Instruct" ]] || die "trainer model-ID argument drifted"
[[ "$(sha256sum "${HOST_REPO}/scripts/openevo_webshop/build_h138a_diversity_inputs.py" | awk '{print $1}')" == "$(jq -r '.dataset_selection.builder_sha256' "${HOST_REPO}/${CONFIG}")" ]] || die "dataset-selection builder hash mismatch"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo SHA mismatch"
[[ "$("${PYTHON}" -m pip freeze --all | LC_ALL=C sort | sha256sum | awk '{print $1}')" == "${PYTHON_FREEZE_SHA}" ]] || die "Python freeze mismatch"
for f in config.json model.safetensors.index.json tokenizer_config.json tokenizer.json; do [[ -f "${MODEL_HOST}/${f}" ]] || die "missing model pin component ${f}"; done
cfg="$(sha256sum "${MODEL_HOST}/config.json" | awk '{print $1}')"; idx="$(sha256sum "${MODEL_HOST}/model.safetensors.index.json" | awk '{print $1}')"; tok="$(sha256sum "${MODEL_HOST}/tokenizer_config.json" | awk '{print $1}')"
[[ "$(printf '%s\n%s\n%s\n' "${cfg}" "${idx}" "${tok}" | sha256sum | awk '{print $1}')" == "${MODEL_REVISION}" ]] || die "model content pin mismatch"
[[ -d "${SEED_HOST}" && -d "${RESERVOIR_HOST}" ]] || die "seed root or reservoir raw root missing"
mkdir -p "${HOST_ROOT}"/{manifests,datasets,artifacts,evaluation,evidence}; chown -R 1001:1001 "${HOST_ROOT}"

utility() { docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=2g --shm-size=2g --user 1001:1001 -v "${HOST_REPO}":/workspace/repo:ro -v "${GIT_COMMON_HOST}:${GIT_COMMON_HOST}:ro" -v "${HOST_RUNS}":/workspace/runs -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -w /workspace/repo -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/repo -e JAVA_HOME="${JDK}" -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src -e PYTHONDONTWRITEBYTECODE=1 --entrypoint "${PYTHON}" "${IMAGE}" "$@"; }
gpu_hours() { local files; files="$(find "${HOST_ROOT}" -name summary.json -type f -print -quit)"; [[ -n "${files}" ]] || { printf '0\n'; return; }; find "${HOST_ROOT}" -name summary.json -type f -print0 | xargs -0 jq -s '[.[].wall_time_seconds // 0] | add / 3600'; }
within_cap() { jq -e --argjson used "$(gpu_hours)" '.budget.gpu_hours_cap > $used' "${HOST_REPO}/${CONFIG}" >/dev/null || die "measured GPU-hour cap reached"; }
idle_gpus() { local u line mem util apps; for u in "${GPU_UUIDS[@]}"; do line="$(nvidia-smi -i "${u}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" || continue; IFS=',' read -r mem util <<<"${line}"; mem="${mem//[[:space:]]/}"; util="${util//[[:space:]]/}"; apps="$(nvidia-smi -i "${u}" --query-compute-apps=pid --format=csv,noheader,nounits)"; [[ "${mem}" =~ ^[0-9]+$ && "${mem}" -le 10 && "${util}" == 0 && -z "${apps}" ]] && printf '%s\n' "${u}"; done; }
run_gpu() { local uuid="$1" name="$2" out="$3"; shift 3; mkdir -p "${HOST_ROOT}/${out#/workspace/runs/${RUN_NAME}/}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts}; chown -R 1001:1001 "${HOST_ROOT}/${out#/workspace/runs/${RUN_NAME}/}"; docker run --rm --name "${name}" --read-only --tmpfs /tmp:rw,nosuid,size=4g --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 --gpus "device=${uuid}" --user 1001:1001 -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -v "${HOST_REPO}":/workspace/repo:ro -v "${GIT_COMMON_HOST}:${GIT_COMMON_HOST}:ro" -v "${CORE_HOST}":/workspace/openevo-core:ro -v /workspace/models:/workspace/models:ro -v /workspace/models:/workspace/workspace/models:ro -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro -v "${HOST_RUNS}":/workspace/runs -w /workspace/repo -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/repo -e WANDB_API_KEY -e WANDB_ENTITY=example-entity -e WANDB_PROJECT=openevo-webshop-experiments -e WANDB_MODE="${TRACKING_MODE}" -e CUDA_VISIBLE_DEVICES="${uuid}" -e TOKENIZERS_PARALLELISM=false -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e JAVA_HOME="${JDK}" -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" -e PYTHONPATH=/workspace/openevo-core/src:/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src -e PYTHONDONTWRITEBYTECODE=1 -e HOME="${out}/home" -e TMPDIR="${out}/tmp" -e XDG_CACHE_HOME="${out}/xdg-cache" -e HF_HOME=/workspace/hf-cache -e WANDB_DIR="${out}/wandb" -e WANDB_CACHE_DIR="${out}/wandb-cache" -e WANDB_CONFIG_DIR="${out}/wandb-config" -e WANDB_ARTIFACT_DIR="${out}/wandb-artifacts" --entrypoint "${PYTHON}" "${IMAGE}" "$@"; }
complete_cell() { local arm task path; arm="$1"; task="$2"; path="${HOST_ROOT}/evaluation/${arm}/task${task}/attempts.jsonl"; [[ -f "${path}" ]] || return 1; jq -es --arg arm "${arm}" --argjson task "${task}" 'length == 4 and all(.[]; .arm == $arm and .goal_idx == $task and (.rollout_seed | IN(2026090500,2026090501,2026090502,2026090503)))' "${path}" >/dev/null; }
eval_cell() { local arm task adapter uuid out; arm="$1"; task="$2"; adapter="$3"; uuid="$4"; out="${ROOT}/evaluation/${arm}/task${task}"; complete_cell "${arm}" "${task}" && return 0; [[ ! -e "${HOST_ROOT}/evaluation/${arm}/task${task}" ]] || die "partial formal evaluation cell ${arm}/${task}"; run_gpu "${uuid}" "h138a-${arm}-${task}" "${out}" -m scripts.openevo_webshop.h1_closed_loop_tracked --task-manifest "${MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}" --adapter "${adapter}" --arm "${arm}" --tasks "${task}" --rollout-seeds 2026090500 2026090501 2026090502 2026090503 --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature 0 --max-steps 30 --max-new 512 --prompt-contract seed3090.webshop-compact-action-prompt.v2 --action-contract seed3090.webshop-action-parser.v3-action-command --wandb-mode "${TRACKING_MODE}" --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb"; }

utility -m unittest discover -s tests -p 'test_*.py' || die "CPU test gate failed"
if [[ "${TRACKING_MODE}" == online ]]; then
  set +e; probe="$(timeout 45s docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=512m -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -e WANDB_MODE=online -e WANDB_INIT_TIMEOUT=10 -e WANDB_SILENT=true -e WANDB_API_KEY --entrypoint "${PYTHON}" "${IMAGE}" -c 'import json, wandb
try:
 r=wandb.init(project="openevo-webshop-experiments",entity="example-entity",name="h138a-online-probe",job_type="preflight",mode="online",reinit="create_new"); print(json.dumps({"status":"ok","run_id":getattr(r,"id",None)})); r.finish()
except BaseException as e: print(json.dumps({"status":"failed","error_type":type(e).__name__,"error":str(e)})); raise' 2>&1)"; probe_rc=$?; set -e
  jq -n --arg output "${probe}" --argjson exit_code "${probe_rc}" '{schema:"openevo.wandb-online-probe.v1",requested_mode:"online",exit_code:$exit_code,output:$output}' > "${HOST_ROOT}/evidence/wandb-online-probe.json"
  [[ "${probe_rc}" == 0 && "${probe}" == *'"status": "ok"'* ]] || TRACKING_MODE=offline
elif [[ "${TRACKING_MODE}" != offline ]]; then die "tracking mode must be online or offline"; fi
utility -m scripts.openevo_webshop.build_h138a_diversity_inputs --analysis "${ANALYSIS}" --harvest-root "${RESERVOIR_ROOT}" --dataset-root "${ROOT}/datasets" --seed-root "${SEED_ROOT}" --data-dir "${SEED_ROOT}/agent_system/environments/env_package/webshop/webshop/data" --index-dir "${SEED_ROOT}/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k" --prior-root /workspace/repo/docs/evidence/remote-runs --aborted-ledger "${ABORTED_LEDGER}" --evaluation-manifest "${MANIFEST}" --experiment-git-sha "${EXECUTION_SHA}" > "${HOST_ROOT}/evidence/input-build.json"
jq -e '([.evaluation.arms[]] | sort) == ["base","diversity-16","diversity-4","diversity-8"] and (.evaluation_tasks | length) == 48' "${HOST_ROOT}/manifests/h1.38a-evaluation-manifest.json" >/dev/null || die "evaluation manifest gate failed"
mapfile -t available < <(idle_gpus); [[ "${#available[@]}" -gt 0 ]] || die "no authorized live-idle GPU"
within_cap
train_arm() { local arm uuid out evidence; arm="$1"; uuid="$2"; out="${ROOT}/artifacts/${arm}"; evidence="${ROOT}/evidence/${arm}-training.json"; [[ ! -e "${HOST_ROOT}/evidence/${arm}-training.json" ]] || return 0; run_gpu "${uuid}" "h138a-train-${arm}" "${out}" -m scripts.openevo_webshop.run_parametric_smoke_tracked --dataset-manifest "${ROOT}/datasets/${arm}/manifest.json" --artifact-root "${out}" --evidence "${evidence}" --run-id "h138a-${arm}-fit" --benchmark webshop --python "${PYTHON}" --base-model Qwen/Qwen2.5-7B-Instruct --model-revision "${MODEL_REVISION}" --max-steps 16 --max-records 16 --minimum-reward 0 --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --gpu-uuid "${uuid}" --wandb-mode "${TRACKING_MODE}" --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb"; }
train_arm diversity-4 "${available[0]}"
adapter_path() { local uri="$1"; [[ "${uri}" == file:///workspace/runs/* ]] || die "adapter URI escaped run root"; printf '%s\n' "${uri#file://}"; }
d4="$(adapter_path "$(jq -r '.adapter_uri' "${HOST_ROOT}/evidence/diversity-4-training.json")")"
task0="$(jq -r '.evaluation_tasks[0]' "${HOST_ROOT}/manifests/h1.38a-evaluation-manifest.json")"; eval_cell base "${task0}" none "${available[0]}"; eval_cell diversity-4 "${task0}" "${d4}" "${available[0]}"
for gate_arm in base diversity-4; do
  jq -es 'length == 4 and all(.[]; .identity_ok and .trace_complete and (.parser_or_fallback_failure | not) and (.error == null))' "${HOST_ROOT}/evaluation/${gate_arm}/task${task0}/attempts.jsonl" >/dev/null || die "bounded first gate invalid for ${gate_arm}"
done
mapfile -t available < <(idle_gpus); [[ "${#available[@]}" -gt 0 ]] || die "no live-idle GPU after gate"; within_cap
train_arm diversity-8 "${available[0]}"; train_arm diversity-16 "${available[0]}"
d8="$(adapter_path "$(jq -r '.adapter_uri' "${HOST_ROOT}/evidence/diversity-8-training.json")")"; d16="$(adapter_path "$(jq -r '.adapter_uri' "${HOST_ROOT}/evidence/diversity-16-training.json")")"
mapfile -t tasks < <(jq -r '.evaluation_tasks[]' "${HOST_ROOT}/manifests/h1.38a-evaluation-manifest.json")
for arm_adapter in "base|none" "diversity-4|${d4}" "diversity-8|${d8}" "diversity-16|${d16}"; do IFS='|' read -r arm adapter <<<"${arm_adapter}"; for task in "${tasks[@]}"; do complete_cell "${arm}" "${task}" && continue; within_cap; mapfile -t available < <(idle_gpus); [[ "${#available[@]}" -gt 0 ]] || die "no live-idle GPU for ${arm}/${task}"; eval_cell "${arm}" "${task}" "${adapter}" "${available[0]}"; done; done
cells=(); for arm in base diversity-4 diversity-8 diversity-16; do for task in "${tasks[@]}"; do cells+=(--cell "${arm}:${task}:${ROOT}/evaluation/${arm}/task${task}"); done; done
utility -m scripts.openevo_webshop.reconcile_h138a_diversity --manifest "${MANIFEST}" "${cells[@]}" --output "${ROOT}/evidence/h1.38a-reconciliation.json"
printf 'H1.38A completed under %s\n' "${HOST_ROOT}"
