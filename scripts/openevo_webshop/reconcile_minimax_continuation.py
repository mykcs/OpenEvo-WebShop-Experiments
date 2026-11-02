#!/usr/bin/env python3
"""Prove that an interrupted MiniMax corpus can exact-resume after control-receipt drift."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, time
from pathlib import Path
from typing import Any

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v:Any)->str: return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def load(p:Path)->dict: return json.loads(p.read_text())
def rows(p:Path)->list[dict]: return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def git(repo:Path,*args:str)->str: return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()
def docker(name:str)->dict:
    return json.loads(subprocess.check_output(["docker","inspect",name],text=True))[0]

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--arm",required=True); p.add_argument("--analysis-root",type=Path,required=True)
    p.add_argument("--execution-source",type=Path,required=True); p.add_argument("--execution-sha",required=True)
    p.add_argument("--authority",type=Path,required=True); p.add_argument("--authority-sha",required=True); p.add_argument("--container",required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); root=a.analysis_root.resolve(); source=a.execution_source.resolve()
    if git(source,"rev-parse","HEAD")!=a.execution_sha or git(source,"status","--porcelain"): raise SystemExit("execution source is not exact/clean")
    if sha(a.authority)!=a.authority_sha: raise SystemExit("authority SHA drift")
    authority=load(a.authority); spec=authority["minimax"]
    ready=root/f"READY_FOR_MINIMAX_{a.arm}.json"; conv=root/f"{a.arm}-conversion-receipt.json"; students=root/f"{a.arm}-student-trajectories.jsonl"; analyzed=root/f"{a.arm}-minimax.jsonl"
    for path in (ready,conv,students,analyzed):
        if not path.is_file(): raise SystemExit(f"missing continuation input {path}")
    r=load(ready); c=load(conv); src=rows(students); out=rows(analyzed)
    if r.get("status")!="PASS" or r.get("arm")!=a.arm or int(r.get("trajectory_count",-1))!=1440: raise SystemExit("READY contract drift")
    if c.get("status")!="PASS" or c.get("arm")!=a.arm or int(c.get("trajectory_count",-1))!=1440: raise SystemExit("conversion contract drift")
    if len(src)!=1440 or not (0 < len(out) < 1440): raise SystemExit("continuation denominator is not partial 1..1439")
    if sha(students)!=c.get("corpus_sha256") or r.get("corpus_sha256")!=c.get("corpus_sha256"): raise SystemExit("student corpus identity drift")
    source_by={str(x["baseline_key"]):x for x in src}; seen=set(); bad=[]
    for row in out:
        tid=str(row.get("trajectory_id"));
        if tid in seen: bad.append((tid,"duplicate")); continue
        seen.add(tid)
        if tid not in source_by: bad.append((tid,"unknown")); continue
        if row.get("parse_ok") is not True: bad.append((tid,"not-parse-ok")); continue
        if row.get("trajectory_sha256")!=canonical(source_by[tid]): bad.append((tid,"trajectory-sha")); continue
        if row.get("analyzer_model")!=spec["model"] or row.get("thinking_type")!=spec["thinking_type"] or row.get("reasoning_split") is not True: bad.append((tid,"profile"))
    if bad: raise SystemExit(f"partial analyzer corpus drift: {bad[:5]}")
    info=docker(a.container); mounts={m["Destination"]:m["Source"] for m in info.get("Mounts",[])}; state=info["State"]
    if mounts.get("/workspace/repo")!=str(source) or mounts.get("/workspace/analysis")!=str(root): raise SystemExit("historical container mount provenance drift")
    if state.get("Running") is True: raise SystemExit("continuation reconciliation only applies to a stopped worker")
    if state.get("OOMKilled") is True: raise SystemExit("true OOM requires resource review, not automatic continuation binding")
    receipt={
      "schema":"openevo.autonomous-minimax-continuation-binding.v1","status":"PASS",
      "arm":a.arm,"scientific_semantics_changed":False,"exact_resume_authorized":True,
      "execution_source":str(source),"execution_sha":a.execution_sha,"authority_path":str(a.authority.resolve()),"authority_sha256":a.authority_sha,
      "current_ready_sha256":sha(ready),"current_ready_successor_authority_sha256":r.get("successor_authority_sha256"),
      "conversion_sha256":sha(conv),"student_corpus_sha256":sha(students),"partial_analysis_sha256":sha(analyzed),"partial_rows":len(out),"partial_parse_ok":len(out),
      "historical_container":{"name":a.container,"exit_code":state.get("ExitCode"),"oom_killed":state.get("OOMKilled"),"started_at":state.get("StartedAt"),"finished_at":state.get("FinishedAt"),"repo_mount":mounts.get("/workspace/repo")},
      "budget_delta":{"new_webshop_calls":0,"new_teacher_calls":0,"final_panel_access":0},
      "resume_contract":{"count_complete_only_if":"parse_ok=true","reuse_existing_parse_ok_rows":True,"retry_only_missing_or_failed_rows":True,"analyzer_model":spec["model"],"thinking_type":spec["thinking_type"],"reasoning_split":True},
      "created_at_epoch":time.time()
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); tmp=a.output.with_name("."+a.output.name+".tmp")
    tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n"); tmp.replace(a.output); print(json.dumps(receipt,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
