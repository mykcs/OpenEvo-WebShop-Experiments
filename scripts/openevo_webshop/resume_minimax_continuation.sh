#!/usr/bin/env bash
set -euo pipefail
ARM="${1:?arm}"
BINDING="${MINIMAX_CONTINUATION_BINDING:?continuation receipt}"
WT="${STAGE1_SUCCESSOR_REPO:?successor checkout}"
OUT="${STAGE1_ANALYSIS_ROOT:?analysis root}"
EXEC_SHA="${STAGE1_SUCCESSOR_EXECUTION_SHA:?execution SHA}"
AUTH="${STAGE1_SUCCESSOR_AUTHORITY:?authority path}"
SEED_SRC="${STAGE1_SEED_SOURCE:-/workspace/worktrees/seed-source-2cf2fadca}"
SECRET="${STAGE1_MINIMAX_SECRET:-/workspace/.secrets/openevo/minimax-teacher.env}"
IMAGE="${STAGE1_ANALYSIS_IMAGE:-experiment-host/openevo-scientific:h144-runtime-v1}"
WORKERS="${STAGE1_MINIMAX_WORKERS:-4}"
ATTEMPT="${STAGE1_MINIMAX_CONTINUATION_ATTEMPT:-continuation-v2}"
NAME="stage1-202609021800-minimax-${ARM}-${ATTEMPT}"
[[ "$(git -C "$WT" rev-parse HEAD)" == "$EXEC_SHA" && -z "$(git -C "$WT" status --porcelain)" ]] || { echo 'NO-GO continuation source drift' >&2; exit 3; }
python3 - "$BINDING" "$ARM" "$EXEC_SHA" "$OUT/${ARM}-minimax.jsonl" <<'PY'
import hashlib,json,sys
p,arm,sha,out=sys.argv[1:]; d=json.load(open(p)); h=lambda x:hashlib.sha256(open(x,'rb').read()).hexdigest()
assert d['schema']=='openevo.autonomous-minimax-continuation-binding.v1' and d['status']=='PASS'
assert d['arm']==arm and d['execution_sha']==sha and d['exact_resume_authorized'] is True and d['scientific_semantics_changed'] is False
assert d['partial_analysis_sha256']==h(out), 'analysis changed after continuation binding; reconcile again'
PY
if docker inspect "$NAME" >/dev/null 2>&1; then
  state="$(docker inspect "$NAME" --format '{{.State.Status}}')"; [[ "$state" == running ]] && { echo "ALREADY_RUNNING $NAME"; exit 0; }
  echo "NO-GO immutable continuation attempt exists: $NAME state=$state" >&2; exit 4
fi
set -a; source "$SECRET"; set +a
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo 'NO-GO MiniMax credential unavailable' >&2; exit 3; }
chown -R 1001:1001 "$OUT"
docker run -d --name "$NAME" --restart no --user 1001:1001 \
  -v "$WT:/workspace/repo:ro" -v "$SEED_SRC:/workspace/seed-source:ro" -v "$OUT:/workspace/analysis" \
  -e PYTHONPATH=/workspace/repo:/workspace/repo/vendor/openevo-webshop/src -e OPENAI_API_KEY \
  --entrypoint /usr/local/bin/python3 "$IMAGE" \
  /workspace/repo/scripts/openevo_webshop/h146_analyze_trajectories.py \
  --baseline-rollouts "/workspace/analysis/${ARM}-student-trajectories.jsonl" \
  --seed-source /workspace/seed-source --output "/workspace/analysis/${ARM}-minimax.jsonl" \
  --model MiniMax-M3 --base-url https://api.minimaxi.com/v1 --max-completion-tokens 1024 \
  --transport-attempts 2 --parse-attempts 2 --timeout 120 --max-workers "$WORKERS" \
  --thinking-type adaptive --reasoning-split --resume >/dev/null
sleep 2
[[ "$(docker inspect "$NAME" --format '{{.State.Status}}')" == running ]] || { echo 'NO-GO continuation worker failed startup' >&2; exit 5; }
echo "STARTED $NAME exact_resume=true execution_sha=$EXEC_SHA"
