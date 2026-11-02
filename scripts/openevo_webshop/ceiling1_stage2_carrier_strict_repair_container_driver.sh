#!/usr/bin/env bash
set -euo pipefail
C="${REC_CONTAINER:?}"; UUID="${REC_GPU_UUID:?}"; REPO="${REC_REPO:?}"; ROOT="${REC_ANALYSIS_ROOT:?}"; OUTROOT="${REC_OUTPUT_ROOT:?}"; ARM="${REC_ARM:?}"
KIND="${REC_KIND:?}"; EVI="${REC_EVIDENCE:?}"; PRIOR="${REC_PRIOR:?}"; STATE="${REC_STATE_ID:?}"; ROUND="${REC_ROUND:?}"; PRIMARY="${REC_EXISTING_PRIMARY:?}"; PRECREC="${REC_EXISTING_RECEIPT:?}"; CONTRACT="${REC_CONTRACT:?}"; MODEL="${REC_MODEL:?}"; MEM="${REC_MEMORY:-}"; SKILL="${REC_SKILL:-}"; SD="${REC_SD_ADAPTER:-}"
install -d -m0755 -o1001 -g1001 "$OUTROOT"
extra=(); [[ -n "$MEM" ]] && extra+=(--memory "$MEM"); [[ -n "$SKILL" ]] && extra+=(--skill "$SKILL"); [[ -n "$SD" ]] && extra+=(--sd-adapter "$SD")
mounts=(); for P in "$PRIOR" "$MEM" "$SKILL" "$SD" "$PRIMARY" "$PRECREC"; do [[ -n "$P" ]] || continue; case "$P" in "$ROOT"/*|"$REPO"/*) ;; *) mounts+=(-v "$P:$P:ro");; esac; done
exec docker run -d --name "$C" --restart no --network none --ipc=host --tmpfs /tmp:rw,size=2g,mode=1777 --user 1001:1001 --gpus "device=$UUID" \
 -e "CUDA_VISIBLE_DEVICES=$UUID" -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e "PYTHONPATH=$REPO" \
 -v "$REPO:$REPO:ro" -v "$ROOT:$ROOT:ro" "${mounts[@]}" -v "$OUTROOT:$OUTROOT" \
 --entrypoint /usr/local/bin/python3 experiment-host/openevo-scientific:h144-runtime-v1 \
 "$REPO/scripts/openevo_webshop/recover_ceiling1_stage2_carrier_strict_repair_v1.py" --arm "$ARM" --kind "$KIND" --evidence "$EVI" --prior "$PRIOR" --model "$MODEL" --output-root "$OUTROOT" --gpu-uuid "$UUID" --state-id "$STATE" --round-index "$ROUND" --existing-primary "$PRIMARY" --existing-receipt "$PRECREC" --artifact-contract "$CONTRACT" "${extra[@]}"
