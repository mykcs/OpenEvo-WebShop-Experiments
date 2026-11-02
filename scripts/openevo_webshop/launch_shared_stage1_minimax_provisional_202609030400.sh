#!/usr/bin/env bash
set -euo pipefail
SNAP=${1:?snapshot root}
ARM=${2:?arm}
case "$ARM" in qwen25-3b|qwen3-1p7b) ;; *) exit 2;; esac
MINI=/workspace/control/openevo-experiment-stage1-minimax-exec-c60419ca
MINI_SHA=c60419cad2e4ff0aad0b7e22177ac6e16475fe97
SEED_SRC=/workspace/worktrees/seed-source-2cf2fadca
SECRET=/workspace/.secrets/openevo/minimax-teacher.env
IMAGE=experiment-host/openevo-scientific:h144-runtime-v1
INPUT=$SNAP/${ARM}-student-trajectories.jsonl
OUTPUT=$SNAP/${ARM}-minimax.jsonl
SID=$(basename "$SNAP")
NAME="stage1-030400-prov-${SID}-${ARM}-v1"
[[ "$(git -c safe.directory=$MINI -C "$MINI" rev-parse HEAD)" == "$MINI_SHA" ]]
[[ -z "$(git -c safe.directory=$MINI -C "$MINI" status --porcelain)" ]]
[[ -s "$INPUT" && -s "$SECRET" ]]
if docker inspect "$NAME" >/dev/null 2>&1; then
  [[ "$(docker inspect -f '{{.State.Status}}' "$NAME")" == running ]] && { echo "ALREADY_RUNNING $NAME"; exit 0; }
  echo "immutable provisional container exists: $NAME" >&2; exit 3
fi
set -a; source "$SECRET"; set +a
[[ -n "${OPENAI_API_KEY:-}" ]]
docker run -d --name "$NAME" --restart no --user 1001:1001 \
  -v "$MINI:/workspace/repo:ro" -v "$SEED_SRC:/workspace/seed-source:ro" -v "$SNAP:/workspace/snapshot" \
  -e PYTHONPATH=/workspace/repo:/workspace/repo/vendor/openevo-webshop/src -e OPENAI_API_KEY \
  --entrypoint /usr/local/bin/python3 "$IMAGE" \
  /workspace/repo/scripts/openevo_webshop/h146_analyze_trajectories.py \
  --baseline-rollouts "/workspace/snapshot/${ARM}-student-trajectories.jsonl" \
  --seed-source /workspace/seed-source --output "/workspace/snapshot/${ARM}-minimax.jsonl" \
  --model MiniMax-M3 --base-url https://api.minimaxi.com/v1 --max-completion-tokens 1024 \
  --transport-attempts 2 --parse-attempts 2 --timeout 120 --max-workers 4 \
  --thinking-type adaptive --reasoning-split --resume >/dev/null
printf '%s\n' "$NAME" > "$SNAP/CONTAINER_${ARM}.txt"
chown 1001:1001 "$SNAP/CONTAINER_${ARM}.txt"
echo "STARTED $NAME input=$(wc -l < "$INPUT") analyzer_sha=$MINI_SHA local_gpu=false"
