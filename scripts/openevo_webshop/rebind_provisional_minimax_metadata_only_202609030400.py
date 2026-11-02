#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path

ALLOWED_CANONICAL_EXTENSIONS={"source_harness_profile","source_prompt_contract_sha256"}

def read_jsonl(p:Path):
    with p.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def canonical_sha(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def projection(t):
    return {k:t.get(k) for k in ("task_id","rollout_id","task_description","success","num_steps","final_task_score","steps")}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",required=True); ap.add_argument("--provisional-source",type=Path,required=True)
    ap.add_argument("--canonical-source",type=Path,required=True); ap.add_argument("--provisional-analysis",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--receipt",type=Path,required=True)
    a=ap.parse_args()
    prov={r["baseline_key"]:r for r in read_jsonl(a.provisional_source)}
    canon={r["baseline_key"]:r for r in read_jsonl(a.canonical_source)}
    analyses={r["trajectory_id"]:r for r in read_jsonl(a.provisional_analysis)}
    promoted=[]; failures=[]
    for key,p in prov.items():
        c=canon.get(key); ar=analyses.get(key)
        if c is None or ar is None or ar.get("parse_ok") is not True: failures.append([key,"missing_or_unparsed"]); continue
        extras=set(c)-set(p)
        if extras != ALLOWED_CANONICAL_EXTENSIONS: failures.append([key,"unexpected_key_delta",sorted(extras)]); continue
        if {k:v for k,v in c.items() if k not in ALLOWED_CANONICAL_EXTENSIONS} != p: failures.append([key,"nonmetadata_content_diff"]); continue
        if projection(p)!=projection(c): failures.append([key,"analyzer_projection_diff"]); continue
        oldsha=canonical_sha(p); newsha=canonical_sha(c)
        if ar.get("trajectory_sha256")!=oldsha: failures.append([key,"provisional_analysis_sha_mismatch"]); continue
        if ar.get("task_description")!=c.get("task_description") or bool(ar.get("source_success"))!=bool(c.get("success")):
            failures.append([key,"analysis_source_metadata_diff"]); continue
        x=dict(ar); x["provisional_trajectory_sha256"]=oldsha; x["trajectory_sha256"]=newsha
        x["canonical_metadata_only_rebind"]={"status":"PASS","added_fields":sorted(ALLOWED_CANONICAL_EXTENSIONS)}
        promoted.append(x)
    if failures: raise SystemExit(f"metadata-only rebind proof failed: {failures[:5]} total={len(failures)}")
    order={k:i for i,k in enumerate(canon)}; promoted.sort(key=lambda r:order[r["trajectory_id"]])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in promoted),encoding="utf-8")
    receipt={"schema":"openevo.shared-stage1.minimax-metadata-only-rebind.v1","status":"PASS","arm":a.arm,
      "promoted_parse_ok":len(promoted),"canonical_count":len(canon),"allowed_added_fields":sorted(ALLOWED_CANONICAL_EXTENSIONS),
      "analyzer_consumed_projection_exact":True,"provisional_source_sha256":sha(a.provisional_source),
      "canonical_source_sha256":sha(a.canonical_source),"provisional_analysis_sha256":sha(a.provisional_analysis),
      "promoted_analysis_sha256":sha(a.output),"created_at_epoch":time.time()}
    a.receipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(receipt,sort_keys=True))
if __name__=="__main__": main()
