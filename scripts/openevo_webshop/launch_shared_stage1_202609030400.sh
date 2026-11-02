#!/usr/bin/env bash
set -euo pipefail
FREEZE=202609030400
EXEC_SHA=23f51ab4b7038d23f16076f047d16e767529a319
GOV=/workspace/control/openevo-experiment-shared-min-deliberation-20260903
EXEC=/workspace/control/openevo-experiment-shared-min-deliberation-exec-23f51ab4
DESIGN="$GOV/configs/experiment/stage1/$FREEZE/design.json"
MANIFEST="$GOV/configs/experiment/stage1/$FREEZE/freeze.json"
WRAPPER="$GOV/scripts/openevo_webshop/shared_stage1_202609030400_worker.py"
SEALER="$GOV/scripts/openevo_webshop/seal_shared_stage1_202609030400.py"
SEED=/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1
MODEL3=/workspace/runs/q25-3b-stage1-sampling-pilot-20260902/model-snapshot
MODEL17=/workspace/models/qwen3-1.7b
ROOT3=/workspace/runs/qwen25-3b-stage1-202609030400
ROOT17=/workspace/runs/qwen3-1p7b-stage1-202609030400
IMAGE=experiment-host/openevo-scientific:h144-runtime-v1
[[ "$(git -C "$EXEC" rev-parse HEAD)" == "$EXEC_SHA" ]] || { echo execution-sha-drift >&2; exit 2; }
[[ -z "$(git -C "$EXEC" status --porcelain)" ]] || { echo execution-worktree-dirty >&2; exit 2; }
[[ ! -e "$ROOT3" && ! -e "$ROOT17" ]] || { echo formal-output-must-be-fresh >&2; exit 2; }
[[ "$(sha256sum /workspace/runs/20260903-shared-min-deliberation/confirmation64/FINAL_CONFIRMATION.json | awk '{print $1}')" == e54be6a4d01353a5a04b9d617c2e1e3a2d6378d386769d234e7b5268fe03aa5b ]] || { echo confirmation-drift >&2; exit 2; }
[[ "$(sha256sum "$MODEL3/model-00001-of-00002.safetensors" | awk '{print $1}')" == 67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2 ]] || exit 2
[[ "$(sha256sum "$MODEL3/model-00002-of-00002.safetensors" | awk '{print $1}')" == a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d ]] || exit 2
[[ "$(sha256sum "$MODEL17/model-00001-of-00002.safetensors" | awk '{print $1}')" == 169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5 ]] || exit 2
[[ "$(sha256sum "$MODEL17/model-00002-of-00002.safetensors" | awk '{print $1}')" == 912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9 ]] || exit 2
python3 - <<PY
import json
for p in ['$DESIGN','$MANIFEST']:
 d=json.load(open(p)); assert d['freeze_id']=='$FREEZE' and d['status']=='FROZEN'
d=json.load(open('$DESIGN')); assert d['execution_git_sha']=='$EXEC_SHA'; assert d['confirmation']['sha256']=='e54be6a4d01353a5a04b9d617c2e1e3a2d6378d386769d234e7b5268fe03aa5b'
m=json.load(open('$MANIFEST'))
import hashlib, pathlib
for rel,expected in m.get('generation_files',{}).items():
    p=pathlib.Path('$EXEC')/rel
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected, ('generation drift',rel)
for rel,expected in m.get('governance_files',{}).items():
    p=pathlib.Path('$GOV')/rel
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected, ('governance drift',rel)
PY
mkdir -p "$ROOT3" "$ROOT17"
launch_one(){
  arm="$1"; model="$2"; gpu="$3"; start="$4"; end="$5"; root="$6"
  out="$root/shard-$(printf '%04d' "$start")-$(printf '%04d' "$end")"
  name="shared-s1-${FREEZE}-${arm//[^a-zA-Z0-9]/-}-${start}-${end}"
  mkdir -p "$out"; chown -R 1001:1001 "$out"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --restart no --gpus "device=$gpu" --network none --ipc=host --user 1001:1001 \
    -e CUDA_VISIBLE_DEVICES="$gpu" -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/repo:/workspace/repo/vendor/openevo-webshop/src \
    -e SHARED_STAGE1_DESIGN_PATH=/workspace/shared_design.json -e SHARED_STAGE1_FREEZE_PATH=/workspace/shared_freeze.json \
    -v "$EXEC:/workspace/repo:ro" -v "$SEED:/workspace/seed:ro" -v "$model:/workspace/model:ro" -v "$out:/workspace/out:rw" \
    -v "$WRAPPER:/workspace/shared_worker.py:ro" -v "$DESIGN:/workspace/shared_design.json:ro" -v "$MANIFEST:/workspace/shared_freeze.json:ro" \
    -w /workspace/repo --entrypoint /usr/local/bin/python3 "$IMAGE" /workspace/shared_worker.py \
    --arm "$arm" --start-position "$start" --end-position "$end" --seed-root /workspace/seed \
    --model /workspace/model --output /workspace/out --gpu-uuid "$gpu" --experiment-git-sha "$EXEC_SHA" >/dev/null
  echo "$name|$arm|$out" >> "/workspace/runs/shared-stage1-${FREEZE}-containers.tsv"
}
rm -f "/workspace/runs/shared-stage1-${FREEZE}-containers.tsv"
launch_one qwen25-3b "$MODEL3" GPU-00000000-0000-0000-0000-000000000000 1 720 "$ROOT3"
launch_one qwen25-3b "$MODEL3" GPU-00000000-0000-0000-0000-000000000000 721 1440 "$ROOT3"
launch_one qwen3-1p7b "$MODEL17" GPU-00000000-0000-0000-0000-000000000000 1 720 "$ROOT17"
launch_one qwen3-1p7b "$MODEL17" GPU-00000000-0000-0000-0000-000000000000 721 1440 "$ROOT17"
declare -A EXITS
while IFS='|' read -r name arm out; do
  code=$(docker wait "$name")
  EXITS["$name"]="$code"
  docker logs "$name" >"$out/container.log" 2>&1 || true
done < "/workspace/runs/shared-stage1-${FREEZE}-containers.tsv"
for arm in qwen25-3b qwen3-1p7b; do
  if [ "$arm" = qwen25-3b ]; then root="$ROOT3"; else root="$ROOT17"; fi
  args=(); exits=()
  while IFS='|' read -r name row_arm out; do
    [ "$row_arm" = "$arm" ] || continue
    args+=(--attempts "$out/attempts.jsonl")
    exits+=(--process-exit "${EXITS[$name]}")
  done < "/workspace/runs/shared-stage1-${FREEZE}-containers.tsv"
  PYTHONPATH="$GOV:$GOV/vendor/openevo-webshop/src" python3 "$SEALER" --arm "$arm" "${args[@]}" "${exits[@]}" --execution-sha "$EXEC_SHA" --design "$DESIGN" --freeze-manifest "$MANIFEST" --output-root "$root"
done
echo SHARED_STAGE1_202609030400_FORMAL_CLOSEOUT_COMPLETE
