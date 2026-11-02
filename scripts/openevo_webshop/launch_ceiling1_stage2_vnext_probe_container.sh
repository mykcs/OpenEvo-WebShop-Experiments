#!/usr/bin/env bash
set -euo pipefail
C="${G4_CONTAINER:?}"; UUID="${G4_GPU_UUID:?}"; REPO="${VNEXT_REPO:?}"; ARM="${VNEXT_ARM:-3b}"; ROOT="${VNEXT_ANALYSIS_ROOT:?}"; SEED="${VNEXT_SEED_ROOT:?}"; MEM="${VNEXT_MEMORY:?}"; SKILL="${VNEXT_SKILL:?}"; AGENT="${VNEXT_AGENT:?}"; OUT="${VNEXT_PROBE_OUTPUT:?}"; SD="${VNEXT_SD_ADAPTER:-}"
CORE=/workspace/control/openevo-core; MODE=pre; args=(); [[ -n "$SD" ]] && { MODE=post; args+=(--sd-adapter "$SD"); }; case "$ARM" in 3b) MODEL="$ROOT/stage2-base/3b"; REV=aa8e72537993ba99e69dfaafa59ed015b17504d1 ;; qwen3-1p7b) MODEL="$ROOT/stage2-base/qwen3-1p7b"; REV=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e ;; *) echo "NO-GO unsupported vNext probe arm=$ARM" >&2; exit 2 ;; esac
OUTDIR="$(dirname "$OUT")"; install -d -m0755 -o1001 -g1001 "$OUTDIR"
mounts=(); for P in "$MEM" "$SKILL" "$AGENT" "$SD"; do [[ -n "$P" ]] || continue; case "$P" in "$ROOT"/*) ;; *) mounts+=(-v "$P:$P:ro");; esac; done
exec docker run -d --name "$C" --restart no --network none --ipc=host --user 1001:1001 --gpus "device=$UUID" \
 -e "CUDA_VISIBLE_DEVICES=$UUID" -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e "PYTHONPATH=$CORE/src:$REPO:$REPO/vendor/openevo-webshop/src" -e "VNEXT_PROBE_OUTPUT=$OUT" \
 -v "$CORE:$CORE:ro" -v "$REPO:$REPO:ro" -v "$ROOT:$ROOT:ro" -v "$SEED:$SEED:ro" "${mounts[@]}" -v "$OUTDIR:$OUTDIR" \
 --entrypoint /bin/bash experiment-host/openevo-scientific:h144-runtime-v1 "$REPO/scripts/openevo_webshop/ceiling1_stage2_vnext_probe_pid1.sh" \
 "$REPO/scripts/openevo_webshop/ceiling1_stage2_vnext_probe_worker.py" --arm "$ARM" --mode "$MODE" --manifest "$REPO/configs/experiment/manifests/openevo-ceiling-1.0-harness201-g4-probe-v1.json" --seed-root "$SEED" --model "$MODEL" --revision "$REV" --memory "$MEM" --skill "$SKILL" --agent "$AGENT" --gpu-uuid "$UUID" --output "$OUT" "${args[@]}"
