#!/usr/bin/env bash
set -euo pipefail
GOV=${POST_STAGE1_GOVERNANCE_ROOT:-/workspace/control/openevo-experiment-shared-min-deliberation-20260903}
FREEZE=202609030400
R3=/workspace/runs/qwen25-3b-stage1-202609030400
R17=/workspace/runs/qwen3-1p7b-stage1-202609030400
OUT=/workspace/runs/20260903-shared-stage1-minimax-m3-202609030400
AUTH=$GOV/configs/experiment/stage1/202609030400/minimax_handoff.json
LOCK=$GOV/configs/experiment/stage1/202609030400/formal_provenance_lock.json
AUDIT=$GOV/scripts/openevo_webshop/audit_shared_stage1_202609030400.py
PREP=$GOV/scripts/openevo_webshop/prepare_shared_stage1_minimax_202609030400.py
LAUNCH=$GOV/scripts/openevo_webshop/launch_shared_stage1_minimax_202609030400.sh
VERIFY=$GOV/scripts/openevo_webshop/verify_shared_stage1_minimax_202609030400.py
PROMOTE=$GOV/scripts/openevo_webshop/promote_provisional_minimax_202609030400.py
PROV_AUTH=$OUT/PROVISIONAL_ACTIVE_SNAPSHOT.json
STATE=$OUT/PIPELINE_STATE.json
mkdir -p "$OUT"; chown -R 1001:1001 "$OUT"
state(){ python3 - "$STATE" "$1" "$2" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); old=json.load(open(p)) if p.exists() else {}
old.update({'schema':'openevo.shared-stage1.postcollection-pipeline.v1','status':sys.argv[2],'detail':sys.argv[3],'updated_at_epoch':time.time()})
p.write_text(json.dumps(old,indent=2,sort_keys=True)+'\n')
PY
}
state WAITING "waiting for both Stage1 final seals"
while true; do
  if [[ -s "$R3/FINALIZER_FAILURE.txt" || -s "$R17/FINALIZER_FAILURE.txt" ]]; then state BLOCKED "Stage1 finalizer reported worker/seal failure"; exit 20; fi
  if [[ -s "$R3/STAGE1_COMPLETE_${FREEZE}.json" && -s "$R17/STAGE1_COMPLETE_${FREEZE}.json" && -s "$R3/FINALIZER_COMPLETE.txt" && -s "$R17/FINALIZER_COMPLETE.txt" ]]; then break; fi
  sleep 10
done
state AUDITING "both Stage1 seals present; running engineering/provenance/behavior audit"
A3=$R3/POSTCOLLECTION_AUDIT_${FREEZE}.json
A17=$R17/POSTCOLLECTION_AUDIT_${FREEZE}.json
rc=0
PYTHONPATH="$GOV:$GOV/vendor/openevo-webshop/src" python3 "$AUDIT" --arm qwen25-3b --stage1-root "$R3" --handoff "$AUTH" --provenance-lock "$LOCK" --output "$A3" >"$R3/POSTCOLLECTION_AUDIT.log" 2>&1 || rc=1
PYTHONPATH="$GOV:$GOV/vendor/openevo-webshop/src" python3 "$AUDIT" --arm qwen3-1p7b --stage1-root "$R17" --handoff "$AUTH" --provenance-lock "$LOCK" --output "$A17" >"$R17/POSTCOLLECTION_AUDIT.log" 2>&1 || rc=1
if [[ $rc -ne 0 ]]; then state BLOCKED "postcollection audit failed; MiniMax not launched"; exit 21; fi
python3 - "$A3" "$A17" <<'PY'
import json,sys
for p in sys.argv[1:]:
    x=json.load(open(p)); assert x['status']=='PASS' and x['mechanical_pass'] and x['behavior_sanity_pass'],p
PY
CHATGPT_GATE=$OUT/CHATGPT_REVIEW_PASS.json
state WAITING_CHATGPT_REVIEW "automated audits PASS; waiting for final ChatGPT engineering/behavior review receipt"
while [[ ! -s "$CHATGPT_GATE" ]]; do sleep 10; done
python3 - "$CHATGPT_GATE" "$A3" "$A17" <<'PY'
import hashlib,json,sys
g=json.load(open(sys.argv[1])); a3=sys.argv[2]; a17=sys.argv[3]
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
assert g['schema']=='openevo.shared-stage1.chatgpt-review.v1'
assert g['status']=='PASS' and g['decision']=='READY_FOR_MINIMAX'
assert g['freeze_id']=='202609030400' and g['formal_generation_code_sha']=='362271d3cdc411581dd7f09d98d5ce69cfb1cb57'
assert g['audit_sha256']['qwen25-3b']==sha(a3)
assert g['audit_sha256']['qwen3-1p7b']==sha(a17)
assert g['review_scope']=='final_1440_both_arms'
PY
state CHATGPT_REVIEW_PASS "final ChatGPT review PASS; canonical MiniMax handoff may proceed"
state PREPARING_MINIMAX "audits PASS; converting canonical 1440 trajectories without new WebShop calls"
if [[ ! -s "$OUT/READY_FOR_MINIMAX_qwen25-3b.json" ]]; then
  PYTHONPATH="$GOV:$GOV/vendor/openevo-webshop/src" python3 "$PREP" --arm qwen25-3b --stage1-root "$R3" --audit "$A3" --authority "$AUTH" --output-root "$OUT" >"$OUT/PREPARE_qwen25-3b.log" 2>&1
fi
if [[ ! -s "$OUT/READY_FOR_MINIMAX_qwen3-1p7b.json" ]]; then
  PYTHONPATH="$GOV:$GOV/vendor/openevo-webshop/src" python3 "$PREP" --arm qwen3-1p7b --stage1-root "$R17" --audit "$A17" --authority "$AUTH" --output-root "$OUT" >"$OUT/PREPARE_qwen3-1p7b.log" 2>&1
fi
if [[ -s "$PROV_AUTH" ]]; then
  state PROMOTING_PROVISIONAL "canonical audit PASS; waiting for provisional analyzers then promoting exact SHA-matched parse_ok rows"
  SNAP=$(python3 - "$PROV_AUTH" <<PY2
import json,sys
x=json.load(open(sys.argv[1])); assert x['status']=='ACTIVE' and x['freeze_id']=='202609030400'
assert x['formal_generation_code_sha']=='362271d3cdc411581dd7f09d98d5ce69cfb1cb57'
assert x['minimax_analyzer_execution_sha']=='c60419cad2e4ff0aad0b7e22177ac6e16475fe97'
assert x['no_feedback_to_stage1'] is True and x['provisional_results_require_sha_promotion'] is True
print(x['snapshot_root'])
PY2
  )
  for arm in qwen25-3b qwen3-1p7b; do
    C=$(cat "$SNAP/CONTAINER_${arm}.txt" 2>/dev/null || true)
    if [[ -n "$C" ]] && docker inspect "$C" >/dev/null 2>&1 && [[ "$(docker inspect -f '{{.State.Running}}' "$C")" == true ]]; then docker wait "$C" >/dev/null || true; fi
    if [[ -s "$SNAP/${arm}-minimax.jsonl" ]]; then
      python3 "$PROMOTE" --arm "$arm" --canonical-source "$OUT/${arm}-student-trajectories.jsonl" --provisional-analysis "$SNAP/${arm}-minimax.jsonl" --output "$OUT/${arm}-minimax.jsonl" >"$OUT/PROMOTE_${arm}.log" 2>&1
    fi
  done
fi
state MINIMAX_RUNNING "both READY_FOR_MINIMAX receipts PASS; promoted provisional parse_ok rows when available; MiniMax-M3 resume started"
run_arm(){
  local arm="$1" max=3 i name rc
  for i in $(seq 1 "$max"); do
    name="stage1-202609030400-minimax-${arm}-v${i}"
    STAGE1_ANALYSIS_ROOT="$OUT" STAGE1_MINIMAX_AUTHORITY="$AUTH" STAGE1_MINIMAX_WORKERS=4 STAGE1_MINIMAX_ATTEMPT="v${i}" \
      "$LAUNCH" "$arm" >"$OUT/LAUNCH_${arm}_v${i}.log" 2>&1 || true
    if docker inspect "$name" >/dev/null 2>&1; then
      rc=$(docker wait "$name" || true)
    else
      rc=99
    fi
    echo "$rc" >"$OUT/PROCESS_EXIT_${arm}_v${i}.txt"
    if [[ "$rc" == 0 ]] && PYTHONPATH="$GOV:$GOV/vendor/openevo-webshop/src" python3 "$VERIFY" --arm "$arm" --analysis-root "$OUT" --authority "$AUTH" --output "$OUT/MINIMAX_COMPLETE_${arm}.json" >"$OUT/VERIFY_${arm}_v${i}.log" 2>&1; then
      return 0
    fi
  done
  return 1
}
rc=0
run_arm qwen25-3b & p3=$!
run_arm qwen3-1p7b & p17=$!
wait "$p3" || rc=1
wait "$p17" || rc=1
if [[ $rc -ne 0 ]]; then state BLOCKED "MiniMax did not close 1440/1440 parse_ok within three resume attempts"; exit 22; fi
state COMPLETE "both MiniMax arms sealed 1440/1440 parse_ok; downstream intentionally stopped"
python3 - "$OUT" "$AUTH" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]); auth=Path(sys.argv[2])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
x={'schema':'openevo.shared-stage1.postcollection-pipeline-complete.v1','status':'PASS','freeze_id':'202609030400',
   'minimax_complete':{},'handoff_authority_sha256':sha(auth),'new_webshop_calls_after_stage1':0,
   'downstream_auto_launch_forbidden':True,'completed_at_epoch':time.time()}
for arm in ('qwen25-3b','qwen3-1p7b'):
    p=root/f'MINIMAX_COMPLETE_{arm}.json'; r=json.load(open(p)); assert r['status']=='PASS' and r['parse_ok']==1440
    x['minimax_complete'][arm]={'path':str(p),'sha256':sha(p),'analysis_corpus_sha256':r['analysis_corpus_sha256']}
(root/'PIPELINE_COMPLETE.json').write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
PY
chown -R 1001:1001 "$OUT"
echo "PIPELINE_COMPLETE $(date -Is)"
