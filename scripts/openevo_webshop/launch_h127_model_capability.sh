#!/usr/bin/env bash
set -euo pipefail

# H1.27 is preregistered but intentionally not auto-started. Run this script
# only after the ordinary-user resource gate reports the selected UUIDs idle.
# It launches two waves of eight independent cells and never kills/preempts.

ROOT="${OPENEVO_H127_ROOT:-/dev/shm/private-source-h1.27-model-capability-20260815}"
REPO_ROOT="${OPENEVO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SEED_ROOT="${OPENEVO_SEED_ROOT:-/workspace/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1}"
BENCHMARK_ROOT="${OPENEVO_BENCHMARK_ROOT:-$REPO_ROOT}"
JAVA_HOME="${JAVA_HOME:-/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1}"
PYTHON="${OPENEVO_PYTHON:-/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python}"
MANIFEST="$REPO_ROOT/docs/evidence/remote-runs/2026-08-15/h1.27/H1.27_TASK_MANIFEST.json"
RUNNER="$REPO_ROOT/scripts/openevo_webshop/run_base30_cell.py"
GATE="$REPO_ROOT/scripts/openevo_webshop/startup_gate_5090.sh"
GROUP="20260815-1400-h127-model-capability"
SEEDS=(2026085000 2026085001 2026085002 2026085003)

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
TASKS=(505 506 507 508)
SEED_ARGS=("${SEEDS[@]}")

mkdir -p "$ROOT/formal/cells"
unset CUDA_VISIBLE_DEVICES

OPENEVO_SCRATCH_ROOT="$ROOT/gate" \
  JAVA_HOME="$JAVA_HOME" \
  bash "$GATE" -- true

run_cell() {
  local cell="$1" model="$2" revision="$3" task="$4" gpu="$5"
  local output="$ROOT/formal/cells/$cell/output"
  mkdir -p "$output"
  JAVA_HOME="$JAVA_HOME" \
    "$PYTHON" "$RUNNER" \
      --seed-root "$SEED_ROOT" \
      --benchmark-root "$BENCHMARK_ROOT" \
      --model "$model" \
      --model-revision "$revision" \
      --task-manifest "$MANIFEST" \
      --task "$task" \
      --gpu-uuid "$gpu" \
      --output-root "$output" \
      --group "$GROUP" \
      --phase h1.27-model-capability \
      --rollout-seeds "${SEED_ARGS[@]}" \
      --prompt-contract seed3090.webshop-compact-action-prompt.v2
}

launch_wave() {
  local model1="$1" revision1="$2" arm1="$3"
  local model2="$4" revision2="$5" arm2="$6"
  local pids=()
  for i in "${!TASKS[@]}"; do
    run_cell "${arm1}-${TASKS[$i]}" "$model1" "$revision1" "${TASKS[$i]}" "${GPU_UUIDS[$i]}" &
    pids+=("$!")
    run_cell "${arm2}-${TASKS[$i]}" "$model2" "$revision2" "${TASKS[$i]}" "${GPU_UUIDS[$((i + 4))]}" &
    pids+=("$!")
  done
  local rc=0 pid
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  return "$rc"
}

launch_wave \
  /workspace/workspace/models/qwen2.5-3b-instruct aa8e72537993ba99e69dfaafa59ed015b17504d1 QWEN25-3B \
  /workspace/workspace/models/qwen2.5-7b-instruct main QWEN25-7B

launch_wave \
  /workspace/workspace/models/granite-3.3-2b-instruct main GRANITE33-2B \
  /workspace/workspace/models/phi-4-mini-instruct main PHI4-MINI

printf '%s\n' "H1.27 formal cells completed under $ROOT/formal/cells"
