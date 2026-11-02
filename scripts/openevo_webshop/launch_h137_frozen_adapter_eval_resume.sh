#!/usr/bin/env bash
set -euo pipefail

# H1.37 RESUME launcher (ops driver only; NOT a change to the frozen protocol).
#
# The primary launch (launch_h137_frozen_adapter_eval.sh) stopped at the wave-7 GPU
# gate: GPUs d41486cf (index 4) and fb620174 (index 7) retained ~21 GB while no
# owner was visible from the experiment-host PID namespace and utilization sampled at
# 0%. Ownership was unresolved, so the gate failed closed. The resume finished
# the remaining cells on six authorized live-idle GPUs (indices 0,1,2,3,5,6);
# the two occupied/unattributed GPUs were never scheduled.
#
# Scientific contract is UNCHANGED: same sealed 64-task manifest (sha re-checked
# against the primary seal), same v3 action parser, same frozen Qwen2.5-7B base and
# frozen H1.36 adapter, same 4 rollout seeds, temperature 0.0. Every attempt still
# carries experiment-git-sha=<EXECUTION_SHA> (79a5963...) in its
# protocol_fingerprint, identical to the 48 already-completed base cells.
#
# This driver only (a) skips already-complete cells (attempts.jsonl with >=4 rows)
# and (b) uses a settle-tolerant GPU pool that drops any GPU that stays busy, so a
# recurring memory leak cannot abort the whole run. It runs OUTSIDE the repo clone;
# the clone stays pinned at EXECUTION_SHA and supplies the eval/reconcile/analysis
# code read-only.

CAMPAIGN_ID="20260818-0000-h137-frozen-adapter-fresh-task-eval"
RUN_NAME="phase-h1.37-frozen-adapter-fresh-task-eval-v1-20260818"
EXECUTION_SHA="${OPENEVO_H137_EXECUTION_SHA:?set OPENEVO_H137_EXECUTION_SHA to the launch-gate commit (79a5963...)}"
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
MODEL_REVISION="0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
MODEL="/workspace/models/qwen2.5-7b-instruct"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"
SEED_ROOT="/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
TASK_MANIFEST="${ROOT}/manifests/h1.37-task-manifest.json"
# Resume stays offline to match the primary run's offline W&B fallback.
TRACKING_MODE="offline"
EVAL_ACTION_CONTRACT="seed3090.webshop-action-parser.v3-action-command"
EVAL_DIR="${ROOT}/evaluation"
RECONCILE_OUT="${ROOT}/evidence/h1.37-reconciliation.json"
RECONCILE_SCHEMA="openevo.webshop-h1.37-reconciliation.v1"
ANALYSIS_OUT="${ROOT}/evidence/h1.37-analysis.json"

H136_RUN_NAME="phase-h1.36-strong-base-self-evolution-v1-20260818"
ADAPTER_WORKER="sd-lora-b7e36404ca6320620e3f86e5225699e3"
ADAPTER_CONTAINER="/workspace/runs/${H136_RUN_NAME}/artifacts/h1.36-self-evolution-fit-gpu0-v1/workers/${ADAPTER_WORKER}/adapter"
ADAPTER_HOST="${HOST_RUNS}/${H136_RUN_NAME}/artifacts/h1.36-self-evolution-fit-gpu0-v1/workers/${ADAPTER_WORKER}/adapter"

# Six authorized live-idle GPUs; idx4 and idx7 were occupied/unattributed at the gate.
GPU_UUIDS=(
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
)
GPU_SETTLE_TIMEOUT_S=600

die() { printf 'H1.37 resume gate failed: %s\n' "$*" >&2; exit 2; }

[[ "$(hostname)" == "experiment-host" ]] || die "control plane hostname is not experiment-host"
[[ "$(id -u)" == "0" ]] || die "control plane is not root"
[[ "${HOST_REPO}" == /workspace/* ]] || die "HOST_REPO must use the Docker-daemon-visible /workspace path"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/h1_closed_loop_tracked.py" ]] || die "no eval entrypoint"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/reconcile_h1_37.py" ]] || die "no reconcile script"
[[ -f "${HOST_REPO}/scripts/openevo_webshop/h1_37_analysis.py" ]] || die "no analysis script"
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "server clone SHA mismatch"
jq -e '.phase == "H1.37" and .formal_task_consumption_allowed == true' "${HOST_REPO}/configs/experiment/current-campaign.json" >/dev/null || die "campaign not H1.37 / consumption not allowed"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "image digest mismatch"
[[ -f "${HOST_ROOT}/manifests/h1.37-task-manifest.json" ]] || die "sealed manifest absent (must reuse the primary run's seal)"
[[ -d "${ADAPTER_HOST}" ]] || die "frozen adapter dir absent: ${ADAPTER_HOST}"

mapfile -t EVAL_TASKS < <(jq -r '.evaluation_tasks[]' "${HOST_ROOT}/manifests/h1.37-task-manifest.json")
[[ "${#EVAL_TASKS[@]}" == "64" ]] || die "manifest task count is not 64"
# Re-pin the manifest against the primary seal so a swapped manifest is caught.
MANIFEST_FILE_SHA="$(sha256sum "${HOST_ROOT}/manifests/h1.37-task-manifest.json" | awk '{print $1}')"
SEALED_SHA="$(jq -r '.manifest_file_sha256' "${HOST_ROOT}/evidence/h1.37-manifest-seal.json")"
[[ "${MANIFEST_FILE_SHA}" == "${SEALED_SHA}" ]] || die "manifest sha ${MANIFEST_FILE_SHA} differs from primary seal ${SEALED_SHA}"
printf 'H1.37 resume: manifest re-pinned to primary seal %s\n' "${MANIFEST_FILE_SHA}" >&2

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

cell_done() { # $1=arm $2=task -> success if attempts.jsonl already has >=4 seed rows
  local f="${HOST_RUNS}/${RUN_NAME}/evaluation/$1-task$2/attempts.jsonl"
  [[ -f "${f}" ]] && [[ "$(wc -l < "${f}")" -ge 4 ]]
}

gpu_free() { # $1=uuid -> success if idle (<=10 MiB, 0% util, no compute app)
  local row memory utilization apps
  row="$(nvidia-smi -i "$1" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" || return 1
  IFS=',' read -r memory utilization <<<"${row}"
  memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
  [[ "${memory}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || return 1
  apps="$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
  [[ "${memory}" =~ ^[0-9]+$ && "${memory}" -le 10 && "${utilization}" == "0" && -z "${apps}" ]]
}

# Remaining cells = all 128 minus the already-complete ones.
TODO=()
for task in "${EVAL_TASKS[@]}"; do cell_done base "${task}" || TODO+=("base:${task}"); done
for task in "${EVAL_TASKS[@]}"; do cell_done self-evolution "${task}" || TODO+=("self-evolution:${task}"); done
printf 'H1.37 resume: %d of 128 cells remaining\n' "${#TODO[@]}" >&2

pool=("${GPU_UUIDS[@]}")
i=0
while [[ "${i}" -lt "${#TODO[@]}" ]]; do
  # Settle: wait until >=1 pool GPU is free. A GPU that stays busy simply never
  # appears in free[], so a recurring leak drops it from scheduling without dying.
  deadline=$(( $(date +%s) + GPU_SETTLE_TIMEOUT_S ))
  while :; do
    free=()
    for u in "${pool[@]}"; do gpu_free "${u}" && free+=("${u}"); done
    [[ "${#free[@]}" -gt 0 ]] && break
    [[ "$(date +%s)" -lt "${deadline}" ]] || die "no GPU freed within ${GPU_SETTLE_TIMEOUT_S}s (pool: ${pool[*]})"
    sleep 20
  done
  pids=()
  for u in "${free[@]}"; do
    [[ "${i}" -lt "${#TODO[@]}" ]] || break
    cell="${TODO[$i]}"; arm="${cell%%:*}"; task="${cell##*:}"
    if [[ "${arm}" == "base" ]]; then adapter="none"; else adapter="${ADAPTER_CONTAINER}"; fi
    out="${EVAL_DIR}/${arm}-task${task}"
    printf 'H1.37 resume: launch %s on %s\n' "${cell}" "${u}" >&2
    run_gpu_container "${u}" "h137-${arm}-${task}" "${out}" -m scripts.openevo_webshop.h1_closed_loop_tracked \
      --task-manifest "${TASK_MANIFEST}" --seed-root "${SEED_ROOT}" --model "${MODEL}" --model-revision "${MODEL_REVISION}" \
      --adapter "${adapter}" --arm "${arm}" --tasks "${task}" --rollout-seeds 2026090200 2026090201 2026090202 2026090203 \
      --action-contract "${EVAL_ACTION_CONTRACT}" \
      --output "${out}" --gpu-uuid "${u}" --experiment-git-sha "${EXECUTION_SHA}" --upstream-openevo-sha "${CORE_SHA}" \
      --runtime-fingerprint "${PYTHON_FREEZE_SHA}" --image-digest "${IMAGE_DIGEST}" --temperature 0.0 --max-steps 30 --max-new 512 \
      --prompt-contract seed3090.webshop-compact-action-prompt.v2 --wandb-mode "${TRACKING_MODE}" \
      --wandb-project openevo-webshop-experiments --wandb-entity example-entity --campaign-id "${CAMPAIGN_ID}" \
      --wandb-group "${CAMPAIGN_ID}" --wandb-dir "${out}/wandb" &
    pids+=("$!"); i=$((i+1))
  done
  rc=0; for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
  [[ "${rc}" == "0" ]] || die "a cell container failed in this batch"
done

# Reconcile + analysis over the full 128-cell panel (same commands as primary run).
cell_args=()
for arm in base self-evolution; do
  for task in "${EVAL_TASKS[@]}"; do cell_args+=(--cell "${arm}:${task}:${EVAL_DIR}/${arm}-task${task}"); done
done
run_utility_container -m scripts.openevo_webshop.reconcile_h1_37 --manifest "${TASK_MANIFEST}" "${cell_args[@]}" \
  --experiment "H1.37-frozen-adapter-fresh-task-eval-v1" --schema "${RECONCILE_SCHEMA}" \
  --output "${RECONCILE_OUT}"
run_utility_container -m scripts.openevo_webshop.h1_37_analysis --reconcile "${RECONCILE_OUT}" --output "${ANALYSIS_OUT}"

spool_hash="$(find "${HOST_ROOT}" -type f -path '*/wandb/*' -print0 | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | awk '{print $1}')"
jq -n --arg probe "${HOST_ROOT}/evidence/wandb-online-probe.json" --arg spool_hash "${spool_hash}" \
  '{schema:"openevo.wandb-fallback-receipt.v1", mode:"offline", reason:"resume driver runs offline; primary probe failed online init", online_probe:$probe, spool_hash:$spool_hash}' \
  > "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
chown 1001:1001 "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"

printf 'H1.37 resume completed under %s\n' "${HOST_ROOT}"
