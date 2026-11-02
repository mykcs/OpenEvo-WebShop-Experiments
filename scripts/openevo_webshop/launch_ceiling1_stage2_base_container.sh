#!/usr/bin/env bash
set -euo pipefail
MODE="${CEILING1_BASE_MODE:?merge|verify}"; ARM="${CEILING1_STAGE2_ARM:?}"; C="${CEILING1_STAGE2_CONTAINER:?}"; UUID="${CEILING1_STAGE2_GPU_UUID:?}"; WT="${CEILING1_AUTOPILOT_REPO:?}"; ROOT="${CEILING1_ANALYSIS_ROOT:?}"; CFGSHA="${CEILING1_STAGE2_CONFIG_SHA:?}"
case "$ARM" in
  3b) BASE=/workspace/models/qwen2.5-3b-instruct; REV=aa8e72537993ba99e69dfaafa59ed015b17504d1; AD="$ROOT/opsd/$ARM/formal-v1/final-adapter" ;;
  7b) BASE=/workspace/models/qwen2.5-7b-instruct; REV=0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e; AD="$ROOT/opsd/$ARM/formal-v1/final-adapter" ;;
  qwen3-1p7b) BASE=/workspace/models/qwen3-1.7b; REV=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e; READY="$ROOT/PRE_STAGE2_READY_qwen3-1p7b.json"; [[ -s "$READY" ]] || { echo "NO-GO missing Qwen3 PRE_STAGE2_READY" >&2; exit 3; }; AD="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["initial_stage2_adapter"]["path"])' "$READY")" ;;
  *) echo "NO-GO unsupported arm=$ARM" >&2; exit 2 ;;
esac
OUT="$ROOT/stage2-base/$ARM"; SHA="$(git -C "$WT" rev-parse HEAD)"; common=(--base /workspace/base --adapter /workspace/adapter --revision "$REV" --arm "$ARM" --gpu-uuid "$UUID")
if [[ "$MODE" == merge ]]; then
 docker run -d --name "$C" --restart no --gpus "device=$UUID" --network none --user 1001:1001 -e CUDA_VISIBLE_DEVICES="$UUID" -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workspace/repo -v "$WT:/workspace/repo:ro" -v "$BASE:/workspace/base:ro" -v "$AD:/workspace/adapter:ro" -v "$OUT:/workspace/out" --entrypoint /usr/local/bin/python3 experiment-host/openevo-scientific:h144-runtime-v1 /workspace/repo/scripts/openevo_webshop/ceiling1_merge_opsd_base.py "${common[@]}" --output /workspace/out --source-sha "$SHA" >/dev/null
else
 docker run -d --name "$C" --restart no --gpus "device=$UUID" --network none --user 1001:1001 -e CUDA_VISIBLE_DEVICES="$UUID" -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workspace/repo -v "$WT:/workspace/repo:ro" -v "$BASE:/workspace/base:ro" -v "$AD:/workspace/adapter:ro" -v "$OUT:/workspace/merged:ro" -v "$OUT:/workspace/out" --entrypoint /usr/local/bin/python3 experiment-host/openevo-scientific:h144-runtime-v1 /workspace/repo/scripts/openevo_webshop/ceiling1_verify_merged_base.py "${common[@]}" --experiment-git-sha "$SHA" --merged /workspace/merged --output /workspace/out/verification-receipt.json --stage2-config-sha256 "$CFGSHA" >/dev/null
fi
echo "STAGE2_BASE_${MODE^^}_STARTED arm=$ARM container=$C"
