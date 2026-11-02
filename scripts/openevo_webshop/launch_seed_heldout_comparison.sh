#!/usr/bin/env bash
set -euo pipefail

# SEED official held-out WebShop comparison (report-oriented, inference-only).
#
# Evaluates a FROZEN reproduced OpenEvo Qwen2.5-7B SD-LoRA policy against its
# frozen BASE on a frozen, SEED-compatible official-held-out 128-task WebShop
# panel (env.seed=0, goal_idx 0-499). TWO arms (BASE, OPEN_EVO_SD) x TWO
# contracts (PRIMARY SEED-strict, DIAGNOSTIC OpenEvo-native). NOTHING is
# trained or retuned. SEED is a paper-reported reference only; it is never
# reproduced here.
#
# Discipline:
#  - The campaign config, frozen panel manifest, and PREREGISTRATION are
#    committed to the repo BEFORE any formal GPU task consumption.
#  - A fail-closed qualification gate (16 checks, ZERO held-out consumption;
#    train-pool goal_idx 5500/6500 only) must PASS and write a hash-bound
#    receipt before any formal cell runs.
#  - Default ONE GPU, sequential. Amendment 2026-08-25 (user directive "必须用6+7两块"):
#    the two arms may run in parallel on two authorized live-idle GPUs (6 + 7) via
#    OPENEVO_SEEDCMP_PIN_UUID + OPENEVO_SEEDCMP_CELLS. GPUs are selected from the
#    standing authorized pool intersected with the immediate live-idle gate. We never
#    kill/reset/preempt any workload and never treat 0% util or an invisible
#    PID as proof of free.
#  - Every GPU-bearing attempt records UUID/start/end/wall-time/GPU-hours/rc.
#
# Docker root is only the control plane. Scientific Python and model processes
# run as UID/GID 1001:1001 in one UUID-bound child container per cell.

CAMPAIGN_ID="20260825-0624-webshop-seed-official-heldout-comparison"
RUN_NAME="phase-seed-official-heldout-comparison-v1-20260825"
EXECUTION_SHA="${OPENEVO_SEEDCMP_EXECUTION_SHA:?set OPENEVO_SEEDCMP_EXECUTION_SHA to the launch-gate commit}"
HOST_REPO="${OPENEVO_SEEDCMP_HOST_REPO:?set OPENEVO_SEEDCMP_HOST_REPO to the exact standalone server clone}"
HOST_RUNS="${OPENEVO_SEEDCMP_HOST_RUNS:-/workspace/runs}"
ROOT="/workspace/runs/${RUN_NAME}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"

IMAGE="${OPENEVO_SEEDCMP_IMAGE:-experiment-host/openevo-scientific:qwen25-3b-webshop-20260814}"
IMAGE_DIGEST="${OPENEVO_SEEDCMP_IMAGE_DIGEST:-sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1}"
PYTHON="${OPENEVO_SEEDCMP_PYTHON:-/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python}"
MODEL="/workspace/models/qwen2.5-7b-instruct"
MODEL_HOST_DIR="/workspace/models/qwen2.5-7b-instruct"
MODEL_REVISION="0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
HF_CACHE_HOST="/workspace/runs/phase-h1.15-recovery-corpus-carrier-diagnostic-v1/hf-cache-v4"

# Server-discovered bindings (verified fail-closed below; never guessed).
SEED_ROOT_HOST="${OPENEVO_SEEDCMP_SEED_ROOT_HOST:?set to the host path of the small WebShop seed root (items_shuffle_1000/items_ins_v2_1000)}"
ADAPTER_HOST="${OPENEVO_SEEDCMP_ADAPTER_HOST:?set to the host path of the frozen H1.38B SD-LoRA adapter dir}"
ADAPTER_CONTAINER="/workspace/adapter"
EXPECTED_ADAPTER_FILE_SHA256="47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c"
EXPECTED_ADAPTER_TENSOR_DIGEST="eedc36232f04a8a7d6e8fc6a38b935758db65206f4945a6cfa35c4ab666a48f9"

CAMPAIGN_CONFIG="/workspace/repo/configs/experiment/webshop-seed-official-heldout-comparison-v1.json"
TASK_MANIFEST="/workspace/repo/configs/experiment/manifests/webshop-seed-official-heldout-comparison-v1-heldout-panel-v1.json"
EXPECTED_MANIFEST_FILE_SHA256="fe902f9610f4f43d7d183c6c57a96084bcb5bf07efacfe5df77942da97eb7635"
EXPECTED_DATA_SHUFFLE_SHA256="30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269"
EXPECTED_DATA_INS_SHA256="f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f"

EVAL_DIR="${ROOT}/evaluation"
EVIDENCE_DIR="${ROOT}/evidence"
TRACKING_MODE="${OPENEVO_WANDB_MODE:-online}"

# Standing authorized pool (subset recorded in the frozen campaign/parent
# contract). This is authorization, not reservation; live-idle is rechecked.
# Amendment 2026-08-25: GPU-fb620174 (index 7) added by explicit user directive
# ("必须用6+7两块") to run the two arms in parallel; see PREREGISTRATION amendment.
GPU_UUIDS=(
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
)

# Orchestration knobs (env-overridable) for splitting the four cells across two
# authorized live-idle GPUs. Defaults preserve the original single-GPU sequential run.
# The scientific contract is GPU-count-agnostic (independent, deterministic episodes,
# per-episode generation_seed, UUID recorded per attempt); only wall-clock changes.
OPENEVO_SEEDCMP_PIN_UUID="${OPENEVO_SEEDCMP_PIN_UUID:-}"        # pin to one authorized UUID (else auto-select)
# Default to all four cells ONLY when the variable is unset; an explicitly empty
# value means "run no cells" (qual-only or finalize-only stage). ${VAR:-default}
# would wrongly treat an explicit empty string as unset, so test for set-ness.
if [[ -z "${OPENEVO_SEEDCMP_CELLS+x}" ]]; then
  OPENEVO_SEEDCMP_CELLS="primary/base primary/sd-lora diagnostic/base diagnostic/sd-lora"
fi
OPENEVO_SEEDCMP_RUN_QUAL="${OPENEVO_SEEDCMP_RUN_QUAL:-1}"       # 0 = reuse existing PASS receipt (still verified fail-closed)
OPENEVO_SEEDCMP_RUN_FINALIZE="${OPENEVO_SEEDCMP_RUN_FINALIZE:-1}"  # 0 = skip reconcile/analyze/gpu-hours/wandb tail
want_cell() { case " ${OPENEVO_SEEDCMP_CELLS} " in *" $1 "*) return 0;; *) return 1;; esac; }

die() { printf 'SEED held-out launch gate failed: %s\n' "$*" >&2; exit 2; }

[[ "$(hostname)" == "experiment-host" ]] || die "control plane hostname is not experiment-host"
[[ "$(id -u)" == "0" ]] || die "control plane is not root"
[[ "${HOST_REPO}" == /workspace/* ]] || die "HOST_REPO must use the Docker-daemon-visible /workspace path"
for f in run_seed_heldout_eval.py qualify_seed_heldout.py reconcile_seed_heldout.py analyze_seed_heldout.py; do
  [[ -f "${HOST_REPO}/scripts/openevo_webshop/${f}" ]] || die "HOST_REPO missing ${f}"
done
[[ "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)" == "${EXECUTION_SHA}" ]] || die "server clone SHA mismatch"
[[ -z "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" status --porcelain)" ]] || die "server clone worktree is dirty"
# Control-plane git evidence for the qualification receipt. The scientific container
# has no reliable git; the authoritative HEAD/clean proof is this control plane (just
# enforced fail-closed above) on the same repo that is bind-mounted read-only.
CONTROL_GIT_HEAD="$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" rev-parse HEAD)"
CONTROL_GIT_CLEAN="false"
[[ -z "$(git -c safe.directory="${HOST_REPO}" -C "${HOST_REPO}" status --porcelain)" ]] && CONTROL_GIT_CLEAN="true"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "container image digest mismatch"

# Frozen panel manifest pin (must match the committed, preregistered manifest).
[[ -f "${HOST_REPO}/configs/experiment/manifests/webshop-seed-official-heldout-comparison-v1-heldout-panel-v1.json" ]] || die "panel manifest absent from server clone"
[[ "$(sha256sum "${HOST_REPO}/configs/experiment/manifests/webshop-seed-official-heldout-comparison-v1-heldout-panel-v1.json" | awk '{print $1}')" == "${EXPECTED_MANIFEST_FILE_SHA256}" ]] || die "panel manifest file sha mismatch (post-freeze edit?)"

# 7B local-model content pin gate.
[[ -d "${MODEL_HOST_DIR}" ]] || die "7B local model dir is absent"
for shard in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  [[ -f "${MODEL_HOST_DIR}/${shard}" ]] || die "7B weight shard ${shard} is absent"
done
_cfg_sha="$(sha256sum "${MODEL_HOST_DIR}/config.json" | awk '{print $1}')"
_idx_sha="$(sha256sum "${MODEL_HOST_DIR}/model.safetensors.index.json" | awk '{print $1}')"
_tok_sha="$(sha256sum "${MODEL_HOST_DIR}/tokenizer_config.json" | awk '{print $1}')"
[[ "$(printf '%s\n%s\n%s\n' "${_cfg_sha}" "${_idx_sha}" "${_tok_sha}" | sha256sum | awk '{print $1}')" == "${MODEL_REVISION}" ]] || die "7B content pin mismatch"

# Frozen H1.38B adapter pin: dir non-empty and the adapter file sha matches the
# preregistered frozen artifact, fail-closed.
[[ -d "${ADAPTER_HOST}" ]] || die "frozen H1.38B adapter dir is absent: ${ADAPTER_HOST}"
ADAPTER_FILE="$(find "${ADAPTER_HOST}" -maxdepth 1 \( -name 'adapter_model.safetensors' -o -name 'adapter_model.bin' \) | head -1)"
[[ -n "${ADAPTER_FILE}" && -f "${ADAPTER_FILE}" ]] || die "frozen adapter weight file not found under ${ADAPTER_HOST}"
[[ "$(sha256sum "${ADAPTER_FILE}" | awk '{print $1}')" == "${EXPECTED_ADAPTER_FILE_SHA256}" ]] || die "frozen adapter file sha mismatch (would substitute a different adapter; forbidden)"

# Seed root data-file pins (the small WebShop the panel was built from).
[[ -d "${SEED_ROOT_HOST}" ]] || die "seed root absent: ${SEED_ROOT_HOST}"
DATA_DIR="${SEED_ROOT_HOST}/agent_system/environments/env_package/webshop/webshop/data"
[[ "$(sha256sum "${DATA_DIR}/items_shuffle_1000.json" | awk '{print $1}')" == "${EXPECTED_DATA_SHUFFLE_SHA256}" ]] || die "items_shuffle_1000.json sha mismatch"
[[ "$(sha256sum "${DATA_DIR}/items_ins_v2_1000.json" | awk '{print $1}')" == "${EXPECTED_DATA_INS_SHA256}" ]] || die "items_ins_v2_1000.json sha mismatch"

mkdir -p "${HOST_ROOT}"/{evaluation,evidence}
chown -R 1001:1001 "${HOST_ROOT}"

# --- W&B online probe (at most once per execution epoch; reused across all cells). ---
# The campaign contract allows offline only after a saved online probe failure plus a
# completed spool hash. Probe online init ONCE here; if it fails, run every cell in
# offline mode and emit a run-level fallback receipt at the end (H1.37 precedent).
probe_wandb() {
  local probe_output probe_rc=0
  set +e
  probe_output="$(timeout 45s docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=512m --shm-size=1g \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
    -e WANDB_MODE=online -e WANDB_INIT_TIMEOUT=10 -e WANDB_SILENT=true -e WANDB_API_KEY -e HOME=/tmp \
    --entrypoint "${PYTHON}" "${IMAGE}" -c 'import json, wandb
try:
    run = wandb.init(project="openevo-webshop-experiments", entity="example-entity", name="seed-heldout-online-probe", job_type="preflight", mode="online", reinit="create_new")
    print(json.dumps({"status": "ok", "run_id": getattr(run, "id", None), "run_url": getattr(run, "url", None)}))
    run.finish()
except BaseException as exc:
    print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
    raise' 2>&1)" || probe_rc=$?
  set -e
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

# --- Live-idle GPU selection (ONE GPU). Fail closed if none is idle. ---
gpu_idle() {
  local uuid="$1" row memory utilization apps
  row="$(nvidia-smi -i "${uuid}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -n "${row}" ]] || return 1
  IFS=',' read -r memory utilization <<<"${row}"
  memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
  [[ "${memory}" =~ ^[0-9]+$ && "${memory}" -le 10 ]] || return 1
  [[ "${utilization}" == "0" ]] || return 1
  apps="$(nvidia-smi -i "${uuid}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "${apps}" ]] || return 1
  return 0
}
SELECTED_UUID=""
if [[ -n "${OPENEVO_SEEDCMP_PIN_UUID}" ]]; then
  in_pool=0
  for uuid in "${GPU_UUIDS[@]}"; do [[ "${uuid}" == "${OPENEVO_SEEDCMP_PIN_UUID}" ]] && in_pool=1; done
  [[ "${in_pool}" == "1" ]] || die "pinned GPU ${OPENEVO_SEEDCMP_PIN_UUID} is not in the authorized pool"
  gpu_idle "${OPENEVO_SEEDCMP_PIN_UUID}" || die "pinned GPU ${OPENEVO_SEEDCMP_PIN_UUID} is not live-idle; refusing to launch (no kill/reset/preempt)"
  SELECTED_UUID="${OPENEVO_SEEDCMP_PIN_UUID}"
else
  for uuid in "${GPU_UUIDS[@]}"; do
    if gpu_idle "${uuid}"; then SELECTED_UUID="${uuid}"; break; fi
  done
fi
[[ -n "${SELECTED_UUID}" ]] || die "no authorized live-idle GPU; campaign stays preregistered. Rerun later; do not kill/reset/preempt."
printf 'SEED held-out: selected live-idle GPU %s\n' "${SELECTED_UUID}" >&2

run_gpu_container() {
  local out="$1"; shift
  local host_out="${HOST_RUNS}/${out#/workspace/runs/}"
  mkdir -p "${host_out}"/{home,tmp,xdg-cache,wandb,hf}
  chown -R 1001:1001 "${host_out}"
  docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=4g \
    --mount type=tmpfs,destination=/root,tmpfs-mode=0755 --shm-size=64g --pids-limit 4096 \
    --gpus "device=${SELECTED_UUID}" --user 1001:1001 \
    -v /workspace/openevo-workspace:/workspace/openevo-workspace:ro \
    -v "${HOST_REPO}":/workspace/repo:ro \
    -v /workspace/models:/workspace/models:ro \
    -v "${ADAPTER_HOST}":"${ADAPTER_CONTAINER}":ro \
    -v "${SEED_ROOT_HOST}":/workspace/seed-root:ro \
    -v "${HF_CACHE_HOST}":/workspace/hf-cache:ro \
    -v "${HOST_RUNS}":/workspace/runs -w /workspace/repo \
    -e WANDB_API_KEY -e WANDB_ENTITY=example-entity -e WANDB_PROJECT=openevo-webshop-experiments \
    -e WANDB_MODE="${TRACKING_MODE}" -e OPENEVO_WANDB_MODE="${TRACKING_MODE}" \
    -e CUDA_VISIBLE_DEVICES="${SELECTED_UUID}" -e TOKENIZERS_PARALLELISM=false \
    -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
    -e JAVA_HOME=/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1 \
    -e PYTHONPATH=/workspace/repo:/workspace/repo/scripts/openevo_webshop:/workspace/repo/vendor/openevo-webshop/src \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e HOME="${out}/home" -e TMPDIR="${out}/tmp" -e XDG_CACHE_HOME="${out}/xdg-cache" -e HF_HOME=/workspace/hf-cache \
    -e WANDB_DIR="${out}/wandb" \
    --entrypoint "${PYTHON}" "${IMAGE}" "$@"
}

SEED_ROOT="/workspace/seed-root"
attempt_index=0
record_attempt() {
  local cell="$1" uuid="$2" start="$3" end="$4" rc="$5"
  local wall gpu_hours
  wall="$(( end - start ))"
  gpu_hours="$("${PYTHON}" -c "print(${wall}/3600.0)")"
  jq -c -n --arg cell "${cell}" --arg uuid "${uuid}" --argjson start "${start}" --argjson endv "${end}" \
    --argjson wall "${wall}" --arg gpu_hours "${gpu_hours}" --argjson rc "${rc}" \
    '{cell:$cell, gpu_uuid:$uuid, start_epoch:$start, end_epoch:$endv, wall_time_seconds:$wall, gpu_count:1, gpu_hours:($gpu_hours|tonumber), exit_code:$rc}' \
    >> "${HOST_ROOT}/evidence/gpu-attempts.jsonl"
  chown 1001:1001 "${HOST_ROOT}/evidence/gpu-attempts.jsonl" || true
}

run_cell() {
  local cell="$1"; shift
  local start end rc=0
  start="$(date +%s)"
  gpu_idle "${SELECTED_UUID}" || die "GPU ${SELECTED_UUID} became busy before cell ${cell}; stop and rerun later"
  set +e
  run_gpu_container "${EVAL_DIR}/${cell}" "$@"
  rc=$?
  set -e
  end="$(date +%s)"
  record_attempt "${cell}" "${SELECTED_UUID}" "${start}" "${end}" "${rc}"
  [[ "${rc}" == "0" ]] || die "cell ${cell} failed with rc=${rc}"
  attempt_index=$(( attempt_index + 1 ))
}

# --- Qualification gate (fail-closed, ZERO held-out consumption). ---
if [[ "${OPENEVO_SEEDCMP_RUN_QUAL}" == "1" ]]; then
  run_cell "qualification" -m scripts.openevo_webshop.qualify_seed_heldout \
    --campaign-config "${CAMPAIGN_CONFIG}" --task-manifest "${TASK_MANIFEST}" \
    --repo-path /workspace/repo --seed-root "${SEED_ROOT}" \
    --model "${MODEL}" --model-revision "${MODEL_REVISION}" --sd-adapter "${ADAPTER_CONTAINER}" \
    --experiment-git-sha "${EXECUTION_SHA}" --control-git-head "${CONTROL_GIT_HEAD}" \
    --control-git-clean "${CONTROL_GIT_CLEAN}" \
    --output "${EVAL_DIR}/qualification" --gpu-uuid "${SELECTED_UUID}"
fi
# Always verify (fail-closed) that a PASS receipt covering the CURRENT scientific
# contract allows formal task consumption. The preregistered binding (PREREGISTRATION
# §10) hashes the campaign config + panel manifest + model + adapter + evaluator
# contract — NOT the orchestration driver SHA — so reusing a receipt across an
# orchestration-only driver change is sound and does not require re-qualification.
if [[ "${OPENEVO_SEEDCMP_RUN_QUAL}" != "1" ]]; then
  CONFIG_SHA="$(sha256sum "${HOST_REPO}/configs/experiment/webshop-seed-official-heldout-comparison-v1.json" | awk '{print $1}')"
  jq -e \
    --arg adapter "${EXPECTED_ADAPTER_FILE_SHA256}" \
    --arg manifest "${EXPECTED_MANIFEST_FILE_SHA256}" \
    --arg config "${CONFIG_SHA}" \
    '.status == "pass"
     and .formal_task_consumption_allowed == true
     and .binding.adapter_file_sha256 == $adapter
     and .binding.task_manifest_file_sha256 == $manifest
     and .binding.campaign_config_sha256 == $config' \
    "${HOST_ROOT}/evaluation/qualification/qualification-receipt.json" >/dev/null \
    || die "no valid PASS qualification receipt matching the pinned adapter/manifest/config; run qualification first"
else
  jq -e '.status == "pass" and .formal_task_consumption_allowed == true' \
    "${HOST_ROOT}/evaluation/qualification/qualification-receipt.json" >/dev/null \
    || die "qualification did not PASS; formal consumption blocked"
fi

# --- Formal cells: PRIMARY (seed-eval) then DIAGNOSTIC (openevo-native). ---
if want_cell "primary/base"; then
run_cell "primary/base" -m scripts.openevo_webshop.run_seed_heldout_eval \
  --contract seed-eval --arm base --task-manifest "${TASK_MANIFEST}" --seed-root "${SEED_ROOT}" \
  --model "${MODEL}" --model-revision "${MODEL_REVISION}" --adapter none \
  --output "${EVAL_DIR}/primary/base" --gpu-uuid "${SELECTED_UUID}" \
  --campaign-id "${CAMPAIGN_ID}" --experiment-git-sha "${EXECUTION_SHA}" --image-digest "${IMAGE_DIGEST}" \
  --wandb-mode "${TRACKING_MODE}" --wandb-group "${CAMPAIGN_ID}"
fi

if want_cell "primary/sd-lora"; then
run_cell "primary/sd-lora" -m scripts.openevo_webshop.run_seed_heldout_eval \
  --contract seed-eval --arm sd-lora --task-manifest "${TASK_MANIFEST}" --seed-root "${SEED_ROOT}" \
  --model "${MODEL}" --model-revision "${MODEL_REVISION}" --adapter "${ADAPTER_CONTAINER}" \
  --expected-adapter-file-sha256 "${EXPECTED_ADAPTER_FILE_SHA256}" \
  --expected-adapter-tensor-digest "${EXPECTED_ADAPTER_TENSOR_DIGEST}" \
  --output "${EVAL_DIR}/primary/sd-lora" --gpu-uuid "${SELECTED_UUID}" \
  --campaign-id "${CAMPAIGN_ID}" --experiment-git-sha "${EXECUTION_SHA}" --image-digest "${IMAGE_DIGEST}" \
  --wandb-mode "${TRACKING_MODE}" --wandb-group "${CAMPAIGN_ID}"
fi

if want_cell "diagnostic/base"; then
run_cell "diagnostic/base" -m scripts.openevo_webshop.run_seed_heldout_eval \
  --contract openevo-native --arm base --task-manifest "${TASK_MANIFEST}" --seed-root "${SEED_ROOT}" \
  --model "${MODEL}" --model-revision "${MODEL_REVISION}" --adapter none \
  --output "${EVAL_DIR}/diagnostic/base" --gpu-uuid "${SELECTED_UUID}" \
  --campaign-id "${CAMPAIGN_ID}" --experiment-git-sha "${EXECUTION_SHA}" --image-digest "${IMAGE_DIGEST}" \
  --wandb-mode "${TRACKING_MODE}" --wandb-group "${CAMPAIGN_ID}"
fi

if want_cell "diagnostic/sd-lora"; then
run_cell "diagnostic/sd-lora" -m scripts.openevo_webshop.run_seed_heldout_eval \
  --contract openevo-native --arm sd-lora --task-manifest "${TASK_MANIFEST}" --seed-root "${SEED_ROOT}" \
  --model "${MODEL}" --model-revision "${MODEL_REVISION}" --adapter "${ADAPTER_CONTAINER}" \
  --expected-adapter-file-sha256 "${EXPECTED_ADAPTER_FILE_SHA256}" \
  --expected-adapter-tensor-digest "${EXPECTED_ADAPTER_TENSOR_DIGEST}" \
  --output "${EVAL_DIR}/diagnostic/sd-lora" --gpu-uuid "${SELECTED_UUID}" \
  --campaign-id "${CAMPAIGN_ID}" --experiment-git-sha "${EXECUTION_SHA}" --image-digest "${IMAGE_DIGEST}" \
  --wandb-mode "${TRACKING_MODE}" --wandb-group "${CAMPAIGN_ID}"
fi

# --- Reconcile + analyze + accounting (run once, after all cells, via RUN_FINALIZE=1). ---
if [[ "${OPENEVO_SEEDCMP_RUN_FINALIZE}" == "1" ]]; then
run_gpu_container "${EVAL_DIR}/reconcile" -m scripts.openevo_webshop.reconcile_seed_heldout \
  --task-manifest "${TASK_MANIFEST}" --run-root "${EVAL_DIR}" \
  --output "${EVIDENCE_DIR}/reconciliation.json" --campaign-id "${CAMPAIGN_ID}"
run_gpu_container "${EVAL_DIR}/analyze" -m scripts.openevo_webshop.analyze_seed_heldout \
  --run-root "${EVAL_DIR}" --task-manifest "${TASK_MANIFEST}" --campaign-id "${CAMPAIGN_ID}" \
  --output "${EVIDENCE_DIR}/analysis.json" --conclusion "${EVIDENCE_DIR}/CONCLUSION.md"

# GPU-hours accounting total.
"${PYTHON}" - "${HOST_ROOT}/evidence/gpu-attempts.jsonl" "${RUN_NAME}" <<'PY' > "${HOST_ROOT}/evidence/gpu-hours-total.json" 2>/dev/null || true
import json, sys
attempts_path, run_name = sys.argv[1], sys.argv[2]
total = 0.0
try:
    with open(attempts_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                total += float(json.loads(line)["gpu_hours"])
            except Exception:
                continue
except FileNotFoundError:
    pass
print(json.dumps({"schema": "openevo.gpu-hours-total.v1", "run": run_name, "gpu_hours_total": total}))
PY
chown -R 1001:1001 "${HOST_ROOT}/evidence" || true

# Run-level W&B offline fallback receipt (only when the probe forced offline).
if [[ "${TRACKING_MODE}" == "offline" ]]; then
  spool_hash="$(find "${HOST_ROOT}" -type f -path '*/wandb/*' -print0 | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | awk '{print $1}')"
  jq -n --arg probe "${HOST_ROOT}/evidence/wandb-online-probe.json" --arg spool_hash "${spool_hash}" \
    '{schema:"openevo.wandb-fallback-receipt.v1", mode:"offline", reason:"online initialization failed; see probe", online_probe:$probe, spool_hash:$spool_hash}' \
    > "${HOST_ROOT}/evidence/wandb-fallback-receipt.json"
  chown 1001:1001 "${HOST_ROOT}/evidence/wandb-fallback-receipt.json" || true
fi
fi

printf 'SEED held-out comparison stage completed under %s (cells: %s, qual: %s, finalize: %s)\n' \
  "${HOST_ROOT}" "${OPENEVO_SEEDCMP_CELLS}" "${OPENEVO_SEEDCMP_RUN_QUAL}" "${OPENEVO_SEEDCMP_RUN_FINALIZE}"
