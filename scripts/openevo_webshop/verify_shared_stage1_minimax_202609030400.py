#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_sha(v)->str: return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def read_jsonl(p:Path):
    with p.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--arm",choices=("qwen25-3b","qwen3-1p7b"),required=True)
    ap.add_argument("--analysis-root",type=Path,required=True); ap.add_argument("--authority",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--analyzer-execution-sha",required=True)
    a=ap.parse_args(); root=a.analysis_root.resolve(); auth=json.loads(a.authority.read_text())
    srcp=root/f"{a.arm}-student-trajectories.jsonl"; outp=root/f"{a.arm}-minimax.jsonl"; sump=root/f"{a.arm}-minimax.summary.json"
    for p in (srcp,outp,sump,a.authority):
        if not p.is_file(): raise SystemExit(f"missing required file: {p}")
    src=read_jsonl(srcp); out=read_jsonl(outp); summary=json.loads(sump.read_text())
    if len(src)!=1440 or len(out)!=1440: raise SystemExit(f"row count mismatch src={len(src)} analysis={len(out)}")
    if int(summary.get("records",-1))!=1440 or int(summary.get("requested_trajectories",-1))!=1440: raise SystemExit("MiniMax summary count mismatch")
    if int(summary.get("parse_ok",-1))!=1440 or int(summary.get("parse_failures",-1))!=0: raise SystemExit("MiniMax parse close failed")
    sb={str(r["baseline_key"]):r for r in src}; ab={str(r["trajectory_id"]):r for r in out}
    if len(sb)!=1440 or len(ab)!=1440 or set(sb)!=set(ab): raise SystemExit("source/analyzer identity mismatch")
    bad=[]
    for tid,row in ab.items():
        if row.get("parse_ok") is not True: bad.append((tid,"parse_ok")); continue
        if row.get("trajectory_sha256")!=canonical_sha(sb[tid]): bad.append((tid,"sha"))
        if row.get("analyzer_provider")!="MiniMax" or row.get("analyzer_model")!=auth["minimax"]["model"]: bad.append((tid,"analyzer"))
        if row.get("thinking_type")!=auth["minimax"]["thinking_type"] or row.get("reasoning_split") is not True: bad.append((tid,"reasoning"))
    if bad: raise SystemExit(f"MiniMax identity/profile drift: {bad[:5]}")
    receipt={"schema":"openevo.shared-stage1.minimax-complete.v1","status":"PASS","freeze_id":"202609030400","arm":a.arm,
             "trajectory_count":1440,"analysis_count":1440,"parse_ok":1440,"parse_failures":0,
             "trajectory_corpus_sha256":sha(srcp),"analysis_corpus_sha256":sha(outp),"summary_sha256":sha(sump),
             "handoff_authority_sha256":sha(a.authority),"analyzer_execution_git_sha":a.analyzer_execution_sha,
             "analyzer_sha256":auth["analyzer_execution"]["successor_analyzer_sha256"],
             "analyzer_base_git_sha":auth["analyzer_execution"]["base_git_sha"],
             "analyzer_regression_receipt_sha256":auth["analyzer_execution"]["regression_receipt_sha256"],
             "thinking_type":auth["minimax"]["thinking_type"],
             "reasoning_split":True,"external_teacher_budget_closed":True,"new_webshop_calls":0,
             "downstream_auto_launch_forbidden":True,"closed_at_epoch":time.time()}
    a.output.parent.mkdir(parents=True,exist_ok=True); t=a.output.with_name("."+a.output.name+".tmp")
    t.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8"); t.replace(a.output)
    print(json.dumps(receipt,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
