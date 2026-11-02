#!/usr/bin/env bash
set -euo pipefail
ARM="${1:?qwen25-3b or qwen3-1p7b}"
case "$ARM" in qwen25-3b|qwen3-1p7b) ;; *) echo unsupported-arm >&2; exit 2;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANEXEC="${STAGE1_MINIMAX_ANALYZER_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ANSHA="${STAGE1_MINIMAX_EXECUTION_SHA:-$(git -c safe.directory=$ANEXEC -C "$ANEXEC" rev-parse HEAD)}"
ANHASH="${STAGE1_MINIMAX_ANALYZER_SHA256:-46f9ecda2b912baf17261541df277ca87289f7aaaeb27ef946d089d36938b940}"
OUT="${STAGE1_ANALYSIS_ROOT:?set STAGE1_ANALYSIS_ROOT}"
AUTH="${STAGE1_MINIMAX_AUTHORITY:?set STAGE1_MINIMAX_AUTHORITY}"
SEED_SRC="${STAGE1_SEED_SOURCE:-/workspace/worktrees/seed-source-2cf2fadca}"
SECRET="${STAGE1_MINIMAX_SECRET:-/workspace/.secrets/openevo/minimax-teacher.env}"
IMAGE="${STAGE1_ANALYSIS_IMAGE:-experiment-host/openevo-scientific:h144-runtime-v1}"
WORKERS="${STAGE1_MINIMAX_WORKERS:-4}"
ATTEMPT="${STAGE1_MINIMAX_ATTEMPT:-v1}"
NAME="stage1-202609030400-minimax-${ARM}-${ATTEMPT}"
READY="$OUT/READY_FOR_MINIMAX_${ARM}.json"
[[ "$(git -c safe.directory=$ANEXEC -C "$ANEXEC" rev-parse HEAD)" == "$ANSHA" ]]
[[ -z "$(git -c safe.directory=$ANEXEC -C "$ANEXEC" status --porcelain)" ]]
[[ "$(sha256sum "$ANEXEC/scripts/openevo_webshop/h146_analyze_trajectories.py" | awk '{print $1}')" == "$ANHASH" ]]
[[ "$(git -c safe.directory=$SEED_SRC -C "$SEED_SRC" rev-parse HEAD)" == 2cf2fadca3c5aba28da68e8e1405182ba8d90e6c ]]
[[ -s "$READY" && -s "$AUTH" && -s "$SECRET" ]]
python3 - "$READY" "$AUTH" "$ARM" "$ANHASH" <<'PY'
import hashlib,json,sys
r=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); arm=sys.argv[3]; analyzer_sha=sys.argv[4]
assert r['status']=='PASS' and r['arm']==arm and r['trajectory_count']==1440
assert r['handoff_authority_sha256']==hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
assert r['resume_success_definition']=='parse_ok=true only' and r['new_webshop_calls']==0
assert a['minimax']['thinking_type']=='adaptive' and a['minimax']['reasoning_split'] is True
assert a['analyzer_execution']['successor_analyzer_sha256']==analyzer_sha
PY
if docker inspect "$NAME" >/dev/null 2>&1; then
  state="$(docker inspect "$NAME" --format '{{.State.Status}}')"
  [[ "$state" == running ]] && { echo "ALREADY_RUNNING $NAME"; exit 0; }
  echo "NO-GO immutable attempt exists: $NAME state=$state" >&2; exit 4
fi
set -a; source "$SECRET"; set +a
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "NO-GO MiniMax credential unavailable" >&2; exit 3; }
chown -R 1001:1001 "$OUT"
docker run -d --name "$NAME" --restart no --user 1001:1001 \
  -v "$ANEXEC:/workspace/repo:ro" -v "$SEED_SRC:/workspace/seed-source:ro" -v "$OUT:/workspace/analysis" \
  -e PYTHONPATH=/workspace/repo:/workspace/repo/vendor/openevo-webshop/src -e OPENAI_API_KEY \
  --entrypoint /usr/local/bin/python3 "$IMAGE" /workspace/repo/scripts/openevo_webshop/h146_analyze_trajectories.py \
  --baseline-rollouts "/workspace/analysis/${ARM}-student-trajectories.jsonl" --seed-source /workspace/seed-source \
  --output "/workspace/analysis/${ARM}-minimax.jsonl" --model MiniMax-M3 --base-url https://api.minimaxi.com/v1 \
  --max-completion-tokens 1024 --transport-attempts 2 --parse-attempts 2 --timeout 120 --max-workers "$WORKERS" \
  --thinking-type adaptive --reasoning-split --resume >/dev/null
echo "STARTED $NAME workers=$WORKERS analyzer_execution_sha=$ANSHA analyzer_file_sha=$ANHASH"
file="$OUT/${ARM}-minimax.jsonl"; deadline=$((SECONDS+180))
while (( SECONDS < deadline )); do
  if [[ -s "$file" ]] && python3 - "$file" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f: rows=[json.loads(x) for x in f if x.strip()]
raise SystemExit(0 if any(r.get('parse_ok') is True for r in rows) else 1)
PY
  then echo "FIRST_RESULT_PASS $NAME"; exit 0; fi
  [[ "$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null || true)" == running ]] || { echo "NO-GO $NAME exited before first parse_ok" >&2; exit 5; }
  sleep 1
done
echo "NO-GO no parse_ok within SLA" >&2; exit 6
