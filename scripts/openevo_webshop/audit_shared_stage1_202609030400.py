#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, re, statistics, time
from pathlib import Path

FREEZE_ID="202609030400"
EXEC_SHA="362271d3cdc411581dd7f09d98d5ce69cfb1cb57"
PROFILE="harness201.1-stage1-shared-seed-native-min-deliberation-v1"
PROMPT_PROFILE="harness201.1-stage1-shared-seed-native-soft-deliberation-v1"
PROMPT_CONTRACT="76b780231fe0298b44ddb25c397537ab81cfdba645258f5643a49514a50b7e55"
ARM_SPEC={
 "qwen25-3b":{"revision":"aa8e72537993ba99e69dfaafa59ed015b17504d1","dtype":"bfloat16","thinking":None},
 "qwen3-1p7b":{"revision":"70d244cc86ccca08cf5af4e1e306ecf908b1ad5e","dtype":"float16","thinking":False},
}

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p:Path): return json.loads(p.read_text(encoding="utf-8"))
def rows_from(root:Path):
    rows=[]
    for p in sorted(root.glob("shard-*/attempts.jsonl")):
        with p.open(encoding="utf-8") as f: rows.extend(json.loads(x) for x in f if x.strip())
    return sorted(rows,key=lambda r:int(r["schedule_position"]))

def metrics(rows):
    scores=[float(r.get("final_task_score",0) or 0) for r in rows]
    steps=[int((r.get("episode") or {}).get("steps",0) or 0) for r in rows]
    adj=comp=0; pre=[]
    for r in rows:
        acts=[]
        for st in (r.get("episode") or {}).get("trace") or []:
            a=str(st.get("action") or "")
            if a: acts.append(a)
            raw=str(st.get("raw_completion") or "")
            p=re.split(r"<action>",raw,maxsplit=1,flags=re.I)[0].strip()
            pre.append(len(p.split()) if p else 0)
        adj += sum(a==b for a,b in zip(acts,acts[1:])); comp += max(0,len(acts)-1)
    return {
      "n":len(rows),"positive":sum(x>0 for x in scores),
      "exact":sum(bool(r.get("won")) and float(r.get("final_task_score",0) or 0)==1 for r in rows),
      "mean_raw":sum(scores)/max(1,len(scores)),"max_steps":sum(x>=15 for x in steps),
      "mean_steps":sum(steps)/max(1,len(steps)),"adjacent_repeat_rate":adj/max(1,comp),
      "pre_action_nonempty_rate":sum(x>0 for x in pre)/max(1,len(pre)),
      "median_pre_action_words":statistics.median(pre) if pre else 0,
    }

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--arm",choices=tuple(ARM_SPEC),required=True)
    ap.add_argument("--stage1-root",type=Path,required=True); ap.add_argument("--handoff",type=Path,required=True)
    ap.add_argument("--provenance-lock",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); root=a.stage1_root.resolve(); hand=load(a.handoff); lock=load(a.provenance_lock)
    receipt_path=root/f"STAGE1_COMPLETE_{FREEZE_ID}.json"; receipt=load(receipt_path); rows=rows_from(root); spec=ARM_SPEC[a.arm]
    mech=collections.Counter(); tasks=collections.Counter()
    for r in rows:
        tasks[(r.get("worker_seed"),r.get("source_session_index"),r.get("instruction_sha256"))]+=1
        ep=r.get("episode") or {}; mech["engineering"]+=int(bool(r.get("engineering_invalid")))
        mech["invalid"]+=int(bool(r.get("terminated_on_invalid_action"))); mech["repair"]+=int(bool(r.get("parser_repair_used")))
        mech["incomplete"]+=int(ep.get("trace_complete") is not True); mech["constraint"]+=int(bool(ep.get("constraint_escape")))
        mech["env_schema"]+=int(bool(ep.get("environment_action_schema_violation")))
        for st in ep.get("trace") or []:
            rec=st.get("runtime_context_receipt") or {}; ch=st.get("action_channel") or rec.get("action_channel") or {}
            samp=rec.get("sampling") or {}; contract=rec.get("successor_contract") or {}
            mech["missing_channel"]+=int(not ch); mech["inadmissible"]+=int(bool(ch) and ch.get("admissible") is not True)
            mech["sampling_drift"]+=int(not (samp.get("temperature")==1.0 and samp.get("top_p")==1.0 and samp.get("top_k")==0))
            mech["delib_drift"]+=int(not (contract.get("minimum_pre_action_model_tokens")==4 and contract.get("minimum_deliberation_enforced_by_interface") is True and contract.get("deliberation_content_guidance") is False and contract.get("observed_pre_action_nonempty") is True))
            mech["interface_drift"]+=int(not (contract.get("interface_owned_action_opener") is True and contract.get("search_query_token_budget")==256 and contract.get("observation_representation")=="SEED WebshopEnvironmentManager.format_obs" and contract.get("history_representation")=="SEED SimpleMemory full formatted pre-action observation + action" and contract.get("strategy_guidance_injected") is False))
    checks={
      "seal_status_pass":receipt.get("status")=="PASS",
      "seal_checks_all_pass":all((receipt.get("checks") or {}).values()),
      "receipt_exec_sha":receipt.get("execution_git_sha")==EXEC_SHA,
      "receipt_freeze":receipt.get("freeze_id")==FREEZE_ID,
      "rows_1440":len(rows)==1440,"positions_exact":[int(r["schedule_position"]) for r in rows]==list(range(1,1441)),
      "tasks_180x8":len(tasks)==180 and set(tasks.values())=={8},
      "row_exec_sha":all(r.get("experiment_git_sha")==EXEC_SHA for r in rows),
      "row_freeze":all(r.get("freeze_id")==FREEZE_ID and r.get("shared_harness_freeze_id")==FREEZE_ID for r in rows),
      "model_revision":all(r.get("model_revision")==spec["revision"] for r in rows),
      "dtype":all(r.get("model_dtype")==spec["dtype"] for r in rows),
      "thinking":all(r.get("chat_template_enable_thinking") is spec["thinking"] for r in rows),
      "profiles":all(r.get("harness_profile")==PROFILE and r.get("prompt_profile")==PROMPT_PROFILE and r.get("prompt_contract_sha256")==PROMPT_CONTRACT for r in rows),
      "qualification_false":all(r.get("qualification_only") is False for r in rows),
      "zero_engineering":mech["engineering"]==0,"zero_invalid":mech["invalid"]==0,"zero_repair":mech["repair"]==0,
      "trace_complete":mech["incomplete"]==0,"zero_constraint":mech["constraint"]==0,"zero_env_schema":mech["env_schema"]==0,
      "action_channel_complete":mech["missing_channel"]==0,"all_actions_admissible":mech["inadmissible"]==0,
      "sampling_readback_exact":mech["sampling_drift"]==0,"deliberation_contract_exact":mech["delib_drift"]==0,
      "interface_contract_exact":mech["interface_drift"]==0,
      "provenance_lock":lock.get("status")=="LOCKED" and lock["formal_execution"]["formal_generation_code_sha"]==EXEC_SHA,
      "confirmation_hash":lock["scientific_identity"]["confirmation_sha256"]==hand["confirmation_sha256"],
      "qualified_generation_sha":lock["scientific_identity"]["qualified_generation_git_sha"]==hand["qualified_generation_git_sha"],
    }
    replay=metrics([r for r in rows if 1<=int(r["schedule_position"])<=64]); gate=hand["behavior_replay_panel"]["gates"][a.arm]
    behavior={
      "positive":replay["positive"]>=gate["positive_min"],"mean_raw":replay["mean_raw"]>=gate["mean_raw_min"],
      "max_steps":replay["max_steps"]<=gate["max_steps_max"],"adjacent_repeat":replay["adjacent_repeat_rate"]<=gate["adjacent_repeat_rate_max"],
    }
    full=metrics(rows); mechanical_pass=all(checks.values()); behavior_pass=all(behavior.values())
    result={
      "schema":"openevo.shared-stage1.audit-before-minimax.v1","status":"PASS" if mechanical_pass and behavior_pass else "FAIL",
      "arm":a.arm,"freeze_id":FREEZE_ID,"formal_generation_code_sha":EXEC_SHA,
      "mechanical_pass":mechanical_pass,"behavior_sanity_pass":behavior_pass,"checks":checks,"mechanics":dict(mech),
      "first64_replay":replay,"first64_gate":gate,"first64_checks":behavior,"full1440_observational":full,
      "reward_or_success_is_stage1_pass_gate":False,"behavior_sanity_is_minimax_handoff_gate":True,
      "stage1_complete_sha256":sha(receipt_path),"handoff_authority_sha256":sha(a.handoff),"provenance_lock_sha256":sha(a.provenance_lock),
      "audited_at_epoch":time.time(),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True)); return 0 if result["status"]=="PASS" else 1

if __name__=="__main__": raise SystemExit(main())
