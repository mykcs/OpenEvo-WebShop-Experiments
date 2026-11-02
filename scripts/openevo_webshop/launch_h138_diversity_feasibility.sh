#!/usr/bin/env bash
set -euo pipefail

# H1.38 feasibility is a fixed, train-only success reservoir.  It never trains
# an adapter or accesses a held-out evaluation panel.
CAMPAIGN_ID="20260818-2345-h138-endogenous-reservoir-v2"
RUN_NAME="phase-h1.38-endogenous-reservoir-v2-20260818"
EXECUTION_SHA="${OPENEVO_H138F_EXECUTION_SHA:?set to the committed feasibility preregistration SHA}"
HOST_REPO="${OPENEVO_H138F_HOST_REPO:?set to the exact server checkout}"
HOST_RUNS="${OPENEVO_H138F_HOST_RUNS:-/workspace/runs}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"; ROOT="/workspace/runs/${RUN_NAME}"
IMAGE="experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST="sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PYTHON="/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
PYTHON_FREEZE_SHA="e3cb53bced07b519157ceca2698935bf0be45a58d08465c87c618ef62c31ee18"
CORE_HOST="/workspace/migration/runtime-src-snapshot/openevo-core"
CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"
MODEL="/workspace/models/qwen2.5-7b-instruct"; MODEL_HOST_DIR="/workspace/models/qwen2.5-7b-instruct"
MODEL_REVISION="0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
SEED_HOST="${HOST_RUNS}/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
DATA_HOST="${SEED_HOST}/agent_system/environments/env_package/webshop/webshop/data"
INDEX_HOST="${SEED_HOST}/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
MANIFEST="${ROOT}/manifests/h1.38-endogenous-reservoir-manifest.json"
CONFIG="configs/experiment/h1.38-reservoir-v2.json"
ABORTED_LEDGER="/workspace/repo/configs/experiment/h1.38-reservoir-v1-aborted-preprotocol-ledger.json"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"
JDK="/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1"
GPU_UUIDS=(
  GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000 GPU-00000000-0000-0000-0000-000000000000
)
die() { printf 'H1.38 feasibility gate failed: %s\n' "$*" >&2; exit 2; }
[[ "$(hostname)" == experiment-host && "$(id -u)" == 0 ]] || die "expected root@experiment-host"
[[ "${HOST_REPO}" == /workspace/* ]] || die "host checkout must be under /workspace"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "server checkout SHA mismatch"
jq -e --arg id "${CAMPAIGN_ID}" '.campaign_id == $id and .phase == "H1.38-reservoir" and .status == "launch-authorized" and .formal_task_consumption_allowed == true' "${HOST_REPO}/configs/experiment/current-campaign.json" >/dev/null || die "campaign not launch-authorized"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "image digest mismatch"
[[ "$(git -c safe.directory="${CORE_HOST}" -C "${CORE_HOST}" rev-parse HEAD)" == "${CORE_SHA}" ]] || die "OpenEvo SHA mismatch"
[[ "$("${PYTHON}" -m pip freeze --all | LC_ALL=C sort | sha256sum | awk '{print $1}')" == "${PYTHON_FREEZE_SHA}" ]] || die "Python freeze mismatch"
for f in config.json model.safetensors.index.json tokenizer_config.json tokenizer.json; do [[ -f "${MODEL_HOST_DIR}/${f}" ]] || die "model file missing: ${f}"; done
cfg="$(sha256sum "${MODEL_HOST_DIR}/config.json" | awk '{print $1}')"; idx="$(sha256sum "${MODEL_HOST_DIR}/model.safetensors.index.json" | awk '{print $1}')"; tok="$(sha256sum "${MODEL_HOST_DIR}/tokenizer_config.json" | awk '{print $1}')"
[[ "$(printf '%s\n%s\n%s\n' "${cfg}" "${idx}" "${tok}" | sha256sum | awk '{print $1}')" == "${MODEL_REVISION}" ]] || die "model content pin mismatch"
[[ -d "${DATA_HOST}" && -d "${INDEX_HOST}" ]] || die "pinned data/index absent"
mkdir -p "${HOST_ROOT}"/{manifests,harvest,evidence}; chown -R 1001:1001 "${HOST_ROOT}"

utility() { docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=2g --shm-size=2g --user 1001:1001 -v "${HOST_REPO}":/workspace/repo:ro -v "${HOST_RUNS}":/workspace/runs -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -w /workspace/repo -e JAVA_HOME="${JDK}" -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src -e PYTHONDONTWRITEBYTECODE=1 --entrypoint "${PYTHON}" "${IMAGE}" "$@"; }
utility -m unittest discover -s tests -p test_experiment_governance.py || die "CPU governance gate failed"
if [[ ! -f "${HOST_ROOT}/manifests/h1.38-endogenous-reservoir-manifest.json" ]]; then
  utility -m scripts.openevo_webshop.build_h138_diversity_feasibility_manifest --seed-root "${SEED_ROOT}" --data-dir "${SEED_ROOT}/agent_system/environments/env_package/webshop/webshop/data" --index-dir "${SEED_ROOT}/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k" --prior-root /workspace/repo/docs/evidence/remote-runs --aborted-exclusion-ledger "${ABORTED_LEDGER}" --out "${MANIFEST}" --experiment-git-sha "${EXECUTION_SHA}" --seed-source-commit 9996427d3ae7c71b3cc8ac82940efe764a8a2485
fi
utility -m scripts.openevo_webshop.build_h138_diversity_feasibility_manifest --seed-root "${SEED_ROOT}" --data-dir "${SEED_ROOT}/agent_system/environments/env_package/webshop/webshop/data" --index-dir "${SEED_ROOT}/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k" --prior-root /workspace/repo/docs/evidence/remote-runs --aborted-exclusion-ledger "${ABORTED_LEDGER}" --out "${MANIFEST}" --experiment-git-sha "${EXECUTION_SHA}" --seed-source-commit 9996427d3ae7c71b3cc8ac82940efe764a8a2485 --verify

if [[ "${TRACKING_MODE}" == online ]]; then
  set +e; probe="$(timeout 45s docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=512m -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -e WANDB_MODE=online -e WANDB_INIT_TIMEOUT=10 -e WANDB_SILENT=true -e WANDB_API_KEY --entrypoint "${PYTHON}" "${IMAGE}" -c 'import json, wandb
try:
 r=wandb.init(project="openevo-webshop-experiments",entity="example-entity",name="h138f-online-probe",job_type="preflight",mode="online",reinit="create_new"); print(json.dumps({"status":"ok","run_id":getattr(r,"id",None)})); r.finish()
except BaseException as e: print(json.dumps({"status":"failed","error_type":type(e).__name__,"error":str(e)})); raise' 2>&1)"; rc=$?; set -e
  jq -n --arg output "${probe}" --argjson exit_code "${rc}" '{schema:"openevo.wandb-online-probe.v1",requested_mode:"online",exit_code:$exit_code,output:$output}' > "${HOST_ROOT}/evidence/wandb-online-probe.json"
  [[ "${rc}" == 0 && "${probe}" == *'"status": "ok"'* ]] || TRACKING_MODE=offline
elif [[ "${TRACKING_MODE}" != offline ]]; then die "tracking mode must be online or offline"; fi

idle_gpus() { local u line mem util apps; for u in "${GPU_UUIDS[@]}"; do line="$(nvidia-smi -i "${u}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" || continue; IFS=',' read -r mem util <<<"${line}"; mem="${mem//[[:space:]]/}"; util="${util//[[:space:]]/}"; apps="$(nvidia-smi -i "${u}" --query-compute-apps=pid --format=csv,noheader,nounits)"; [[ "${mem}" =~ ^[0-9]+$ && "${mem}" -le 10 && "${util}" == 0 && -z "${apps}" ]] && printf '%s\n' "${u}"; done; }
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader > "${HOST_ROOT}/evidence/gpu-preflight-$(date -u +%Y%m%dT%H%M%SZ).csv"
cell_state() { local task="$1" path="${HOST_ROOT}/harvest/task${1}/attempts.jsonl"; [[ ! -e "${path}" ]] && { echo missing; return; }; jq -s --argjson task "${task}" 'length == 4 and ([.[] | select(.arm == "self-generated-harvest" and .goal_idx == $task) | .rollout_seed] | sort) == [2026090400,2026090401,2026090402,2026090403]' "${path}" >/dev/null 2>&1 && { echo complete; return; }; die "partial/malformed formal cell ${task}"; }
first_gate() { jq -es 'length == 4 and all(.[]; .identity_ok and .trace_complete and (.parser_or_fallback_failure | not) and (.error == null))' "${HOST_ROOT}/harvest/task${1}/attempts.jsonl" >/dev/null || die "bounded first cell invalid"; }
run_cell() { local task="$1" uuid="$2" out="${ROOT}/harvest/task${1}" host_out="${HOST_ROOT}/harvest/task${1}"; [[ "$(cell_state "${task}")" == complete ]] && return 0; mkdir -p "${host_out}"/{home,tmp,xdg-cache,wandb,wandb-cache,wandb-config,wandb-artifacts,hf}; chown -R 1001:1001 "${host_out}"; docker run --rm --name "h138r-harvest-${task}" --read-only --tmpfs /tmp:rw,nosuid,size=4g --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 --gpus "device=${uuid}" --user 1001:1001 -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro -v "${HOST_REPO}":/workspace/repo:ro -v "${CORE_HOST}":/workspace/openevo-core:ro -v /workspace/models:/workspace/models:ro -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro -v "${HOST_RUNS}":/workspace/runs -w /workspace/repo -e WANDB_API_KEY -e WANDB_ENTITY=example-entity -e WANDB_PROJECT=openevo-webshop-experiments -e WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_CAMPAIGN_ID="${CAMPAIGN_ID}" -e CUDA_VISIBLE_DEVICES="${uuid}" -e TOKENIZERS_PARALLELISM=false -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e JAVA_HOME="${JDK}" -e PATH="${JDK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" -e PYTHONPATH=/workspace/openevo-core/src:/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src -e PYTHONDONTWRITEBYTECODE=1 -e HOME="${out}/home" -e TMPDIR="${out}/tmp" -e XDG_CACHE_HOME="${out}/xdg-cache" -e HF_HOME=/workspace/hf-cache -e WANDB_DIR="${out}/wandb" -e WANDB_CACHE_DIR="${out}/wandb-cache" -e WANDB_CONFIG_DIR="${out}/wandb-config" -e WANDB_ARTIFACT_DIR="${out}/wandb-artifacts" --entrypoint "${PYTHON}" "${IMAGE}" -m scripts.openevo_webshop.h1_closed_loop_tracked --task-manifest "${MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}" --adapter none --arm self-generated-harvest --tasks "${task}" --rollout-seeds 2026090400 2026090401 2026090402 2026090403 --action-contract seed3090.webshop-action-parser.v3-action-command --output "${out}" --gpu-uuid "${uuid}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature 0.4 --max-steps 30 --max-new 512 --prompt-contract seed3090.webshop-compact-action-prompt.v2 --wandb-mode "${TRACKING_MODE}" --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb"; }
mapfile -t TASKS < <(jq -r '.tasks[].goal_idx' "${HOST_ROOT}/manifests/h1.38-endogenous-reservoir-manifest.json"); [[ "${#TASKS[@]}" == 64 ]] || die "manifest task count"
mapfile -t available < <(idle_gpus); [[ "${#available[@]}" -gt 0 ]] || die "no authorized live-idle GPU"
run_cell "${TASKS[0]}" "${available[0]}"; first_gate "${TASKS[0]}"
checkpoint() { utility -m scripts.openevo_webshop.h1_38_diversity_feasibility_analysis --config "${CONFIG}" --run-root "${ROOT}" --output "${ROOT}/evidence/h1.38-endogenous-reservoir-progress.json" --allow-incomplete; }
stop_if_sufficient_or_capped() { checkpoint; jq -e '.sufficiency_reached == true' "${HOST_ROOT}/evidence/h1.38-endogenous-reservoir-progress.json" >/dev/null && return 0; cap="$(jq -r '.budget.gpu_hours_cap' "${HOST_REPO}/${CONFIG}")"; jq -e --argjson cap "${cap}" '.measured_gpu_hours >= $cap' "${HOST_ROOT}/evidence/h1.38-endogenous-reservoir-progress.json" >/dev/null && return 0; return 1; }
for ((start=1; start<${#TASKS[@]};)); do if stop_if_sufficient_or_capped; then break; fi; mapfile -t available < <(idle_gpus); [[ "${#available[@]}" -gt 0 ]] || die "no authorized live-idle GPU for next wave"; pids=(); for ((i=0; i<${#available[@]} && start+i<${#TASKS[@]}; i++)); do run_cell "${TASKS[$((start+i))]}" "${available[$i]}" & pids+=("$!"); done; rc=0; for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done; [[ "${rc}" == 0 ]] || die "a formal cell failed"; start=$((start+${#available[@]})); done
utility -m scripts.openevo_webshop.h1_38_diversity_feasibility_analysis --config "${CONFIG}" --run-root "${ROOT}" --output "${ROOT}/evidence/h1.38-endogenous-reservoir-analysis.json" --allow-incomplete
if [[ "${TRACKING_MODE}" == offline ]]; then spool="$(find "${HOST_ROOT}" -type f -path '*/wandb/*' -print0 | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | awk '{print $1}')"; jq -n --arg spool_hash "${spool}" '{schema:"openevo.wandb-fallback-receipt.v1",mode:"offline",spool_hash:$spool_hash}' > "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"; fi
printf 'H1.38 diversity feasibility complete under %s\n' "${HOST_ROOT}"
