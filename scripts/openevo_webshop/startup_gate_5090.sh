#!/usr/bin/env bash
set -euo pipefail

# Fail-closed startup gate for root-first OpenEvo GPU experiments on the ZJU
# server. The gate never deletes, kills, resets, preempts, restarts, or changes
# host services.
#
# Required resource contract:
#   OPENEVO_AUTHORIZED_GPU_UUIDS=GPU-a[,GPU-b,...]
#   OPENEVO_SELECTED_GPU_UUIDS=GPU-a[,GPU-b,...]
#
# If exactly one UUID is authorized, OPENEVO_SELECTED_GPU_UUIDS may be omitted
# and the unique authorized UUID is selected. With multiple authorized UUIDs the
# selected subset must be explicit.
#
# Usage:
#   OPENEVO_AUTHORIZED_GPU_UUIDS=GPU-... \
#   OPENEVO_SELECTED_GPU_UUIDS=GPU-... \
#   OPENEVO_SCRATCH_ROOT=/dev/shm/<run-id> \
#   JAVA_HOME=/path/to/jdk \
#     bash scripts/openevo_webshop/startup_gate_5090.sh -- python ...

scratch_root=${OPENEVO_SCRATCH_ROOT:-/dev/shm/openevo-experiment}
min_free_gib=${OPENEVO_MIN_FREE_GIB:-8}
# This gate runs inside experiment-host, whose empty process list can omit sibling
# containers.  Keep the threshold fixed to the strict resource-gate baseline;
# an environment override would turn namespace-local negative evidence into an
# unsafe authorization decision.
max_idle_memory_mib=16
max_idle_util_percent=0

fail() {
  echo "startup gate: $*" >&2
  exit 2
}

normalize_uuid_list() {
  printf '%s' "$1" | tr ',;\n\t' '    ' | xargs
}

authorized_text="$(normalize_uuid_list "${OPENEVO_AUTHORIZED_GPU_UUIDS:-}")"
[[ -n "$authorized_text" ]] || fail "OPENEVO_AUTHORIZED_GPU_UUIDS is required; GPU visibility is not authorization"
read -r -a authorized_gpu_uuids <<< "$authorized_text"

selected_text="$(normalize_uuid_list "${OPENEVO_SELECTED_GPU_UUIDS:-}")"
if [[ -z "$selected_text" ]]; then
  if [[ "${#authorized_gpu_uuids[@]}" -eq 1 ]]; then
    selected_gpu_uuids=("${authorized_gpu_uuids[0]}")
  else
    fail "OPENEVO_SELECTED_GPU_UUIDS is required when more than one GPU UUID is authorized"
  fi
else
  read -r -a selected_gpu_uuids <<< "$selected_text"
fi

[[ "${#selected_gpu_uuids[@]}" -gt 0 ]] || fail "selected GPU UUID set is empty"
[[ "${#selected_gpu_uuids[@]}" -le 8 ]] || fail "selected GPU UUID set exceeds the eight-GPU server inventory"

declare -A authorized_set=()
for uuid in "${authorized_gpu_uuids[@]}"; do
  [[ "$uuid" == GPU-* ]] || fail "invalid authorized GPU UUID: $uuid"
  [[ -z "${authorized_set[$uuid]:-}" ]] || fail "duplicate authorized GPU UUID: $uuid"
  authorized_set["$uuid"]=1
done

declare -A selected_set=()
for uuid in "${selected_gpu_uuids[@]}"; do
  [[ "$uuid" == GPU-* ]] || fail "invalid selected GPU UUID: $uuid"
  [[ -n "${authorized_set[$uuid]:-}" ]] || fail "selected GPU UUID is not authorized for this run: $uuid"
  [[ -z "${selected_set[$uuid]:-}" ]] || fail "duplicate selected GPU UUID: $uuid"
  selected_set["$uuid"]=1
done

mkdir -p \
  "${scratch_root}/tmp" \
  "${scratch_root}/wandb" \
  "${scratch_root}/wandb-cache" \
  "${scratch_root}/wandb-config" \
  "${scratch_root}/wandb-artifacts" \
  "${scratch_root}/xdg-cache"
test -w "${scratch_root}" || fail "scratch root is not writable: ${scratch_root}"

free_kib="$(df -Pk "${scratch_root}" | awk 'NR==2 {print $4}')"
min_free_kib="$((min_free_gib * 1024 * 1024))"
if [[ -z "${free_kib}" || "${free_kib}" -lt "${min_free_kib}" ]]; then
  fail "scratch free space below ${min_free_gib} GiB: ${free_kib:-unknown} KiB"
fi

root_free_kib="$(df -Pk / | awk 'NR==2 {print $4}')"
if [[ -n "${root_free_kib}" && "${root_free_kib}" -lt 262144 ]]; then
  echo "startup gate: warning: root filesystem has <256 MiB free; all temp/W&B paths are isolated under ${scratch_root}" >&2
fi

command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable"
all_gpu_rows="$(nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
visible_gpu_count="$(printf '%s\n' "${all_gpu_rows}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"

resource_failures=()
selected_gpu_rows=()
for requested_uuid in "${selected_gpu_uuids[@]}"; do
  set +e
  gpu_row="$(nvidia-smi -i "${requested_uuid}" --query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1)"
  gpu_row_rc=$?
  set -e
  if [[ "$gpu_row_rc" -ne 0 ]]; then
    resource_failures+=("GPU ${requested_uuid} is not currently visible/queryable: ${gpu_row}")
    continue
  fi

  IFS=',' read -r gpu_index gpu_uuid gpu_name gpu_driver gpu_used gpu_total gpu_util <<< "$gpu_row"
  gpu_index="$(printf '%s' "$gpu_index" | xargs)"
  gpu_uuid="$(printf '%s' "$gpu_uuid" | xargs)"
  gpu_name="$(printf '%s' "$gpu_name" | xargs)"
  gpu_driver="$(printf '%s' "$gpu_driver" | xargs)"
  gpu_used="$(printf '%s' "$gpu_used" | xargs)"
  gpu_total="$(printf '%s' "$gpu_total" | xargs)"
  gpu_util="$(printf '%s' "$gpu_util" | xargs)"

  if [[ "$gpu_uuid" != "$requested_uuid" ]]; then
    resource_failures+=("GPU identity mismatch: requested ${requested_uuid}, observed ${gpu_uuid}")
    continue
  fi

  selected_gpu_rows+=("${gpu_index}, ${gpu_uuid}, ${gpu_name}, ${gpu_driver}, ${gpu_used}, ${gpu_total}, ${gpu_util}")

  if [[ ! "${gpu_used}" =~ ^[0-9]+$ ]]; then
    resource_failures+=("GPU ${gpu_uuid} returned non-numeric memory.used: ${gpu_used:-empty}")
    continue
  fi
  if [[ ! "${gpu_util}" =~ ^[0-9]+$ ]]; then
    resource_failures+=("GPU ${gpu_uuid} returned non-numeric utilization.gpu: ${gpu_util:-empty}")
    continue
  fi

  set +e
  compute_apps="$(nvidia-smi -i "${gpu_uuid}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1)"
  compute_apps_rc=$?
  set -e
  if [[ "${compute_apps_rc}" -ne 0 ]]; then
    resource_failures+=("GPU ${gpu_uuid} compute-app query failed (exit ${compute_apps_rc}): ${compute_apps}")
    continue
  fi
  if [[ -n "$(printf '%s\n' "${compute_apps}" | sed '/^[[:space:]]*$/d')" ]]; then
    resource_failures+=("GPU ${gpu_uuid} has compute apps: ${compute_apps}")
  fi
  if [[ "${gpu_used}" -gt "${max_idle_memory_mib}" ]]; then
    resource_failures+=("GPU ${gpu_uuid} memory.used=${gpu_used} MiB exceeds idle limit ${max_idle_memory_mib} MiB")
  fi
  if [[ "${gpu_util}" -gt "${max_idle_util_percent}" ]]; then
    resource_failures+=("GPU ${gpu_uuid} utilization=${gpu_util}% exceeds idle limit ${max_idle_util_percent}%")
  fi
done

if [[ "${#resource_failures[@]}" -gt 0 ]]; then
  echo "startup gate: selected authorized GPU set is not genuinely idle/usable; no workload was started" >&2
  printf 'startup gate: %s\n' "${resource_failures[@]}" >&2
  exit 2
fi

if [[ -z "${JAVA_HOME:-}" || ! -x "${JAVA_HOME}/bin/javac" ]]; then
  fail "JAVA_HOME/bin/javac is required"
fi

export TMPDIR="${scratch_root}/tmp"
export XDG_CACHE_HOME="${scratch_root}/xdg-cache"
export WANDB_DIR="${scratch_root}/wandb"
export WANDB_CACHE_DIR="${scratch_root}/wandb-cache"
export WANDB_CONFIG_DIR="${scratch_root}/wandb-config"
export WANDB_ARTIFACT_DIR="${scratch_root}/wandb-artifacts"
export WANDB_MODE=online
export OPENEVO_WANDB_MODE=online

IFS=, read -r -a _unused <<< ""
authorized_csv="$(IFS=,; printf '%s' "${authorized_gpu_uuids[*]}")"
selected_csv="$(IFS=,; printf '%s' "${selected_gpu_uuids[*]}")"
selected_snapshot="$(printf '%s;' "${selected_gpu_rows[@]}")"

gate_receipt="${scratch_root}/startup-gate.env"
{
  echo "schema=openevo.startup-gate.v2"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "user=$(id -un)"
  echo "hostname=$(hostname)"
  echo "scratch_root=${scratch_root}"
  echo "scratch_free_kib=${free_kib}"
  echo "root_free_kib=${root_free_kib:-unknown}"
  echo "visible_gpu_count=${visible_gpu_count}"
  echo "authorized_gpu_uuids=${authorized_csv}"
  echo "selected_gpu_uuids=${selected_csv}"
  echo "max_idle_memory_mib=${max_idle_memory_mib}"
  echo "max_idle_util_percent=${max_idle_util_percent}"
  echo "gpu_process_observation_scope=current PID namespace; not a physical-host ownership audit"
  echo "gpu_ownership_attribution=not attempted; this receipt decides launch eligibility only"
  echo "wandb_mode=online"
  echo "selected_gpu_snapshot=${selected_snapshot}"
} > "${gate_receipt}"
chmod 600 "${gate_receipt}"

if [[ "${1:-}" == "--" ]]; then shift; fi
if [[ "$#" -eq 0 ]]; then
  echo "startup gate passed; no command supplied" >&2
  printf '%s\n' "${selected_gpu_rows[@]}"
  exit 0
fi

echo "startup gate passed: selected=${selected_csv}, visible=${visible_gpu_count}, scratch=${scratch_root}, W&B=online"
exec "$@"
