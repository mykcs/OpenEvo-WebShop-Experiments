#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path

FREEZE_ID="202609030400"; EXEC_SHA="362271d3cdc411581dd7f09d98d5ce69cfb1cb57"
ARMS=("qwen25-3b","qwen3-1p7b")
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p:Path): return json.loads(p.read_text(encoding="utf-8"))
def read_jsonl(p:Path):
    with p.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def atomic_json(p:Path,v):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name("."+p.name+".tmp")
    t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n",encoding="utf-8"); t.replace(p)
def task_from_prompt(prompt:str)->str:
    m=re.search(r"(?:^|\n)Task:\s*(.+?)\nPrior to this step",prompt,re.S)
    if not m: m=re.search(r"(?:^|\n)Task:\s*(.+?)\nRecent history",prompt,re.S)
    return " ".join((m.group(1) if m else "").split())
def convert(row,arm):
    trace=(row.get("episode") or {}).get("trace") or []
    if not trace: raise ValueError(f"position {row.get('schedule_position')} has no trace")
    pos=int(row["schedule_position"]); rollout=(pos-1)%8; task_num=(pos-1)//8
    task_id=f"shared030400-{arm}-task{task_num:03d}"; desc=task_from_prompt(str(trace[0].get("prompt") or ""))
    if not desc: raise ValueError(f"position {pos} missing task description")
    steps=[]
    for i,st in enumerate(trace):
        rec=st.get("runtime_context_receipt") or {}; ch=st.get("action_channel") or rec.get("action_channel") or {}
        valid=bool(st.get("action_valid",ch.get("admissible",False)))
        obs=str(st.get("formatted_observation") or "")
        if not obs:
            p=str(st.get("prompt") or ""); m=re.search(r"\nCurrent observation:\s*(.+?)\nAllowed actions:\n",p,re.S); obs=(m.group(1) if m else p).strip()
        steps.append({"step_idx":int(st.get("step",i)),"observation":obs,"observation_prompt":str(st.get("prompt") or ""),
                      "model_response":str(st.get("raw_completion") or ""),"action_valid":valid,
                      "info":{"is_action_valid":valid,"task_score":float(st.get("task_score",0.0) or 0)}})
    return {"task_id":task_id,"task_type":"webshop","goal_idx":int(row["source_session_index"]),"rollout_id":rollout,
            "baseline_key":f"{task_id}:{rollout}","task_description":desc,"steps":steps,
            "success":bool(row.get("won",False)),"completed":bool(row.get("trace_complete",False)),"num_steps":len(steps),
            "final_reward":float(row.get("final_task_score",0) or 0),"final_task_score":float(row.get("final_task_score",0) or 0),
            "source_schedule_position":pos,"source_instruction_sha256":row.get("instruction_sha256"),
            "source_rollout_seed":int(row.get("rollout_seed",0)),"source_execution_git_sha":str(row.get("experiment_git_sha")),
            "source_freeze_id":str(row.get("freeze_id")),"source_harness_profile":str(row.get("harness_profile")),
            "source_prompt_contract_sha256":str(row.get("prompt_contract_sha256"))}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--arm",choices=ARMS,required=True); ap.add_argument("--stage1-root",type=Path,required=True)
    ap.add_argument("--audit",type=Path,required=True); ap.add_argument("--authority",type=Path,required=True); ap.add_argument("--output-root",type=Path,required=True)
    a=ap.parse_args(); audit=load(a.audit); auth=load(a.authority); root=a.stage1_root.resolve()
    if audit.get("status")!="PASS" or audit.get("arm")!=a.arm: raise SystemExit("READY_FOR_MINIMAX audit is not PASS")
    if auth.get("status")!="FROZEN" or auth.get("freeze_id")!=FREEZE_ID or auth.get("formal_generation_code_sha")!=EXEC_SHA: raise SystemExit("handoff authority drift")
    corpus=root/f"{a.arm}-stage1-{FREEZE_ID}.jsonl"; seal=root/f"STAGE1_COMPLETE_{FREEZE_ID}.json"
    if not corpus.is_file() or not seal.is_file(): raise SystemExit("sealed Stage1 corpus missing")
    receipt=load(seal)
    if receipt.get("status")!="PASS" or receipt.get("execution_git_sha")!=EXEC_SHA or receipt.get("corpus_sha256")!=sha(corpus): raise SystemExit("Stage1 seal/corpus drift")
    rows=read_jsonl(corpus)
    if len(rows)!=1440 or [int(r["schedule_position"]) for r in rows]!=list(range(1,1441)): raise SystemExit("Stage1 corpus coverage drift")
    converted=[convert(r,a.arm) for r in rows]
    a.output_root.mkdir(parents=True,exist_ok=True)
    out=a.output_root/f"{a.arm}-student-trajectories.jsonl"
    out.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in converted),encoding="utf-8")
    conversion={"schema":"openevo.shared-stage1.minimax-conversion.v1","status":"PASS","arm":a.arm,"freeze_id":FREEZE_ID,
                "trajectory_count":1440,"total_actions":sum(x["num_steps"] for x in converted),"corpus_sha256":sha(out),
                "source_stage1_corpus":str(corpus),"source_stage1_corpus_sha256":sha(corpus),"source_stage1_complete_sha256":sha(seal),
                "source_audit_sha256":sha(a.audit),"formal_generation_code_sha":EXEC_SHA,"handoff_authority_sha256":sha(a.authority),
                "new_webshop_calls":0,"external_teacher_calls_before_analysis":0,"prepared_at_epoch":time.time()}
    atomic_json(a.output_root/f"{a.arm}-conversion-receipt.json",conversion)
    ready={"schema":"openevo.shared-stage1.ready-for-minimax.v1","status":"PASS","arm":a.arm,"freeze_id":FREEZE_ID,
           "trajectory_count":1440,"corpus_sha256":conversion["corpus_sha256"],"source_stage1_complete_sha256":conversion["source_stage1_complete_sha256"],
           "source_audit_sha256":conversion["source_audit_sha256"],"handoff_authority_sha256":conversion["handoff_authority_sha256"],
           "analyzer_model":auth["minimax"]["model"],"thinking_type":auth["minimax"]["thinking_type"],"reasoning_split":auth["minimax"]["reasoning_split"],
           "resume_success_definition":"parse_ok=true only","new_webshop_calls":0,"prepared_at_epoch":time.time()}
    atomic_json(a.output_root/f"READY_FOR_MINIMAX_{a.arm}.json",ready); print(json.dumps(ready,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
