#!/usr/bin/env bash
set -euo pipefail

# Durable root-first entry point for future experiments.
# It waits for the fail-closed 5090 startup gate, preserves every failed check,
# then execs exactly one command with W&B forced online by the gate.
# It never resets, kills, preempts, restarts, or changes host services.
#
# The underlying gate requires OPENEVO_AUTHORIZED_GPU_UUIDS and, when more than
# one UUID is authorized, an explicit OPENEVO_SELECTED_GPU_UUIDS subset.

usage() {
  cat >&2 <<'USAGE'
usage: wait_for_idle_5090.sh [options] -- command [args...]

required environment:
  OPENEVO_AUTHORIZED_GPU_UUIDS   comma/space separated UUIDs authorized for run
  OPENEVO_SELECTED_GPU_UUIDS     UUID subset for this launch; optional only when exactly one UUID is authorized

options:
  --scratch-root PATH       isolated wait/log root (default: /dev/shm/openevo-experiment)
  --interval-seconds N      seconds between checks (default: 60)
  --timeout-seconds N       0 means wait indefinitely (default: 0)
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gate_script="${OPENEVO_GATE_SCRIPT:-${script_dir}/startup_gate_5090.sh}"
scratch_root="${OPENEVO_SCRATCH_ROOT:-/dev/shm/openevo-experiment}"
interval_seconds="${OPENEVO_WAIT_INTERVAL_SECONDS:-60}"
timeout_seconds="${OPENEVO_WAIT_TIMEOUT_SECONDS:-0}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --scratch-root)
      [[ "$#" -ge 2 ]] || { usage; exit 2; }
      scratch_root="$2"
      shift 2
      ;;
    --interval-seconds)
      [[ "$#" -ge 2 ]] || { usage; exit 2; }
      interval_seconds="$2"
      shift 2
      ;;
    --timeout-seconds)
      [[ "$#" -ge 2 ]] || { usage; exit 2; }
      timeout_seconds="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ "$#" -gt 0 ]] || { usage; exit 2; }
[[ -n "${OPENEVO_AUTHORIZED_GPU_UUIDS:-}" ]] || {
  echo "wait gate: OPENEVO_AUTHORIZED_GPU_UUIDS is required" >&2
  usage
  exit 2
}
[[ "$interval_seconds" =~ ^[0-9]+$ && "$interval_seconds" -gt 0 ]] || {
  echo "wait gate: interval must be a positive integer" >&2
  exit 2
}
[[ "$timeout_seconds" =~ ^[0-9]+$ ]] || {
  echo "wait gate: timeout must be a non-negative integer" >&2
  exit 2
}
[[ -f "$gate_script" ]] || {
  echo "wait gate: gate script is missing: $gate_script" >&2
  exit 2
}
mkdir -p "$scratch_root/wait"
test -w "$scratch_root" || {
  echo "wait gate: scratch root is not writable: $scratch_root" >&2
  exit 2
}

lock_dir="$scratch_root/wait/launch.lock.d"
lock_owner="$$-$(date +%s)-${RANDOM:-0}"
stale_lock_dir="${lock_dir}.stale.${lock_owner}"
lock_acquired=false
for attempt in 1 2 3; do
  if mkdir "$lock_dir" 2>/dev/null; then
    printf '%s\n' "$$" > "$lock_dir/pid"
    printf '%s\n' "$lock_owner" > "$lock_dir/owner"
    lock_acquired=true
    break
  fi
  owner_pid="$(cat "$lock_dir/pid" 2>/dev/null || true)"
  if [[ "$owner_pid" =~ ^[0-9]+$ ]] && kill -0 "$owner_pid" 2>/dev/null; then
    echo "wait gate: another launcher already owns $lock_dir (pid=$owner_pid)" >&2
    exit 3
  fi
  if [[ ! "$owner_pid" =~ ^[0-9]+$ && "$attempt" -lt 3 ]]; then
    sleep 0.05
    continue
  fi
  if mv "$lock_dir" "$stale_lock_dir" 2>/dev/null; then
    rm -f -- "$stale_lock_dir/pid" "$stale_lock_dir/owner"
    rmdir "$stale_lock_dir" 2>/dev/null || {
      echo "wait gate: stale launcher lock contains unexpected files: $stale_lock_dir" >&2
      exit 3
    }
  fi
done
[[ "$lock_acquired" == true ]] || {
  echo "wait gate: launcher lock could not be acquired safely: $lock_dir" >&2
  exit 3
}
cleanup_lock() {
  observed_owner="$(cat "$lock_dir/owner" 2>/dev/null || true)"
  if [[ "$observed_owner" == "$lock_owner" ]]; then
    rm -f -- "$lock_dir/pid" "$lock_dir/owner"
    rmdir "$lock_dir" 2>/dev/null || true
  fi
}
trap cleanup_lock EXIT

log_file="$scratch_root/wait/gate-wait.log"
status_file="$scratch_root/wait/gate-wait.env"
started_epoch="$(date +%s)"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

write_status() {
  local state="$1" last_check="$2" detail="$3"
  {
    echo "schema=openevo.wait-for-idle-5090.v2"
    echo "state=$state"
    echo "started_utc=$started_utc"
    echo "last_check_utc=$last_check"
    echo "hostname=$(hostname)"
    echo "user=$(id -un)"
    echo "scratch_root=$scratch_root"
    echo "interval_seconds=$interval_seconds"
    echo "timeout_seconds=$timeout_seconds"
    echo "authorized_gpu_uuids=${OPENEVO_AUTHORIZED_GPU_UUIDS}"
    echo "selected_gpu_uuids=${OPENEVO_SELECTED_GPU_UUIDS:-<unique-authorized-default>}"
    echo "detail=$detail"
  } > "$status_file"
  chmod 600 "$status_file"
}

trap 'write_status interrupted "$(date -u +%Y-%m-%dT%H:%M:%SZ)" signal; exit 130' INT TERM

echo "wait gate: started=$started_utc scratch=$scratch_root interval=${interval_seconds}s timeout=${timeout_seconds}s" | tee -a "$log_file"
write_status waiting "$started_utc" "no successful selected-uuid idle gate yet"

attempt=0
while :; do
  attempt=$((attempt + 1))
  check_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  check_log="$scratch_root/wait/gate-check-${attempt}.log"
  set +e
  OPENEVO_SCRATCH_ROOT="$scratch_root/gate" bash "$gate_script" -- true >"$check_log" 2>&1
  rc=$?
  set -e
  cat "$check_log" >> "$log_file"
  if [[ "$rc" -eq 0 ]]; then
    write_status launching "$check_utc" "selected authorized UUID gate passed on attempt $attempt"
    echo "wait gate: selected UUID gate passed at $check_utc; launching exactly one command" | tee -a "$log_file"
    set +e
    env OPENEVO_SCRATCH_ROOT="$scratch_root/gate" bash "$gate_script" -- "$@"
    command_rc=$?
    set -e
    finish_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ "$command_rc" -eq 0 ]]; then
      write_status completed "$finish_utc" "command completed successfully"
    else
      write_status failed "$finish_utc" "command exited rc=$command_rc"
    fi
    exit "$command_rc"
  fi

  write_status waiting "$check_utc" "gate rc=$rc on attempt $attempt; see $check_log"
  echo "wait gate: attempt=$attempt check=$check_utc rc=$rc; retrying in ${interval_seconds}s" | tee -a "$log_file"
  if [[ "$timeout_seconds" -gt 0 && "$(( $(date +%s) - started_epoch ))" -ge "$timeout_seconds" ]]; then
    write_status timed_out "$check_utc" "timeout after $attempt attempts"
    echo "wait gate: timeout reached; command was not started" | tee -a "$log_file"
    exit 4
  fi
  sleep "$interval_seconds"
done
