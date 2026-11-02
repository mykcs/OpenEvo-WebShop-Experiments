#!/usr/bin/env sh
set -eu

# The rc15 inference image ships Python 3.10, while the pinned server
# scientific runtime is Python 3.11.  OpenEvo's isolated trainer subprocess
# invokes this executable with ``-I``; remove that flag and expose the pinned
# source checkout explicitly so the trainer can import the same Core SHA.
if [ "${1:-}" = "-I" ]; then
  shift
fi

TRAINER_PYTHON="${OPENEVO_TRAINER_PYTHON:-/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python}"
TRAINER_CORE="${OPENEVO_TRAINER_CORE:-/workspace/openevo-core}"
export PYTHONPATH="${TRAINER_CORE}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [ -z "${OPENEVO_SD_LORA_PARENT_PID:-}" ]; then
  export OPENEVO_SD_LORA_PARENT_PID="${PPID}"
fi
# The pinned Core trainer has a narrow environment allowlist that omits
# WANDB_MODE.  Preserve the explicit parent fallback so a failed online
# initialization cannot make the isolated trainer retry online.
export WANDB_MODE="${WANDB_MODE:-offline}"
exec "${TRAINER_PYTHON}" "$@"
