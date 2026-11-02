#!/usr/bin/env bash
# H1.37 filesystem quiescence gate (closeout requirement #3).
#
# Do NOT reconcile the moment tmux exits. This gate returns QUIESCED only when ALL hold:
#   1. the `h137resume` tmux session has exited;
#   2. no `h137-*` docker container is running;
#   3. no `h137-*` process is alive;
#   4. all 128 cells present and 512 attempts complete (each attempts.jsonl has 4 rows);
#   5. file count + key-file hashes are UNCHANGED across two consecutive invocations
#      (this invocation's snapshot must equal the stored previous snapshot).
#
# Exit codes: 0 = QUIESCED (safe to reconcile); 3 = not ready (run still active / panel
# incomplete); 4 = stable-check pending or snapshot changed (wait for the next invocation).
# Read-only against the run root; state is kept in a small state file outside the panel.

set -uo pipefail

RUN_ROOT="${1:?usage: h1_37_closeout_quiescence.sh RUN_ROOT [STATE_FILE]}"
STATE_FILE="${2:-/tmp/h137_quiescence_snapshot.json}"
EVAL_DIR="${RUN_ROOT}/evaluation"
EXPECT_CELLS=128
EXPECT_ATTEMPTS=512

note() { printf 'h137-quiescence: %s\n' "$*" >&2; }

# --- 1. tmux session gone ------------------------------------------------------
if tmux has-session -t h137resume 2>/dev/null; then
  note "NOT READY: tmux session h137resume still active"; exit 3
fi

# --- 2. no h137-* container ---------------------------------------------------
if command -v docker >/dev/null 2>&1; then
  running="$(docker ps --filter 'name=h137-' --format '{{.Names}}' 2>/dev/null || true)"
  if [[ -n "${running}" ]]; then
    note "NOT READY: h137-* container(s) running: ${running}"; exit 3
  fi
fi

# --- 3. no h137-* process ------------------------------------------------------
procs="$(pgrep -af 'h137' 2>/dev/null | grep -v 'h1_37_closeout_quiescence' || true)"
if [[ -n "${procs}" ]]; then
  note "NOT READY: h137 process(es) alive: ${procs}"; exit 3
fi

# --- 4. 128 cells / 512 attempts complete -------------------------------------
python3 - "$EVAL_DIR" "$EXPECT_CELLS" "$EXPECT_ATTEMPTS" <<'PYEOF' || { note "NOT READY: panel incomplete"; exit 3; }
import glob, os, sys
ev, ec, ea = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cells = [d for d in glob.glob(os.path.join(ev, "*-task*")) if os.path.isdir(d)]
attempts = 0
for d in cells:
    ap = os.path.join(d, "attempts.jsonl")
    if not os.path.isfile(ap):
        sys.exit(1)
    with open(ap, encoding="utf-8") as fh:
        attempts += sum(1 for line in fh if line.strip())
sys.exit(0 if (len(cells) == ec and attempts == ea) else 1)
PYEOF

# --- 5. snapshot: file count + key-file hashes --------------------------------
snapshot="$(python3 - "$RUN_ROOT" <<'PYEOF'
import hashlib, json, os, sys
root = sys.argv[1]
file_count = 0
key_hashes = {}
for dirpath, _dirs, files in os.walk(root):
    for name in files:
        file_count += 1
        full = os.path.join(dirpath, name)
        if name in ("attempts.jsonl", "summary.json") or dirpath.endswith("/evidence"):
            try:
                with open(full, "rb") as fh:
                    key_hashes[os.path.relpath(full, root)] = hashlib.sha256(fh.read()).hexdigest()
            except OSError:
                pass
print(json.dumps({"file_count": file_count, "key_hashes": key_hashes}, sort_keys=True))
PYEOF
)"

if [[ ! -f "${STATE_FILE}" ]]; then
  printf '%s' "${snapshot}" > "${STATE_FILE}"
  note "PENDING: baseline snapshot stored; re-check next invocation"; exit 4
fi

previous="$(cat "${STATE_FILE}")"
if [[ "${previous}" != "${snapshot}" ]]; then
  printf '%s' "${snapshot}" > "${STATE_FILE}"
  note "CHANGED: files still settling; snapshot updated, re-check next invocation"; exit 4
fi

note "QUIESCED: h137resume exited, no h137-* container/process, 128 cells / 512 attempts, files stable across two checks"
exit 0
