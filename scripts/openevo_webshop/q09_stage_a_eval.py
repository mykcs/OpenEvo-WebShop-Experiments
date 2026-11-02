#!/usr/bin/env python3
"""One frozen Q09 Stage-A endpoint over the shared 64-task +40 panel.

Run only inside the validated Q09 runtime image after each endpoint config binds
its finished checkpoint/carriers and the current panel SHA. This creates 64
paired observations per endpoint, with no automatic retries or score selection.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any
import scripts.openevo_webshop.ceiling1_stage2_rollout_worker_qwen3 as q09


def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def canonical(x: Any) -> bytes: return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def atomic(path:Path,value:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name("."+path.name+".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    dfd=os.open(str(path.parent),os.O_RDONLY)
    try: os.fsync(dfd)
    finally: os.close(dfd)
def check_hash(path:Path,expected:str,label:str):
    if not expected or sha(path)!=expected: raise SystemExit(f"{label} SHA-256 mismatch: {path}")
def validate_stage_identity(cfg, panel_status):
    stage=cfg.get("evaluation_stage")
    if stage not in {"common-start","common-start-confirmation","development-20","development-40","confirmation-40"}:
        raise SystemExit("unknown Q09 A evaluation stage")
    if (stage in {"confirmation-40","common-start-confirmation"}) != (panel_status=="FROZEN_PREOUTCOME"):
        raise SystemExit("panel role does not match the requested evaluation stage")
    if stage in {"common-start","common-start-confirmation"}:
        if cfg.get("treatment")!="SHARED_START" or cfg.get("stream_id")!="shared-common-start":
            raise SystemExit("common-start must be the shared pre-treatment endpoint")
    elif cfg.get("treatment") not in {"FROZEN","ACTIVE"} or cfg.get("stream_id") not in {"q09-stream-1","q09-stream-2"}:
        raise SystemExit("endpoint identity outside frozen A matrix")
    return stage

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--endpoint",type=Path,required=True); ap.add_argument("--seed-root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--gpu-uuid",required=True); ap.add_argument("--resume",action="store_true"); a=ap.parse_args()
    manifest=json.loads(a.manifest.read_text()); manifest_for_sha=dict(manifest); panel_sha=manifest_for_sha.pop("panel_sha256",None)
    if hashlib.sha256(canonical(manifest_for_sha)).hexdigest()!=panel_sha: raise SystemExit("panel canonical SHA mismatch")
    if manifest.get("status") not in {"FROZEN_PREOUTCOME", "FROZEN_EXISTING_DEVELOPMENT_PANEL"} or len(manifest.get("tasks",[]))!=64: raise SystemExit("Q09 evaluation manifest is not a complete 64-task panel")
    cfg=json.loads(a.endpoint.read_text()); cfg_sha=sha(a.endpoint)
    stage=validate_stage_identity(cfg,manifest.get("status"))
    required=("endpoint_id","backbone","treatment","stream_id","state_id","model","model_revision","memory","memory_sha256","skill","skill_sha256","agent_system","agent_system_sha256","adapter_sha256","experiment_git_sha","config_sha256","image_digest")
    missing=[k for k in required if not cfg.get(k)]
    if missing: raise SystemExit("endpoint config missing: "+", ".join(missing))
    if cfg.get("panel_sha256")!=panel_sha: raise SystemExit("endpoint config panel SHA mismatch")
    if cfg["backbone"] not in {"ORDINARY_DIRECT_APPLY","SIMPLE_BOUNDED"}: raise SystemExit("endpoint backbone outside frozen A matrix")
    if os.environ.get("CUDA_VISIBLE_DEVICES","").strip()!=a.gpu_uuid: raise SystemExit("CUDA_VISIBLE_DEVICES mismatch")
    memory=Path(cfg["memory"]); skill=Path(cfg["skill"]); agent=Path(cfg["agent_system"]); adapter=Path(cfg.get("sd_adapter") or "")
    check_hash(memory,cfg["memory_sha256"],"memory"); check_hash(skill,cfg["skill_sha256"],"skill"); check_hash(agent,cfg["agent_system_sha256"],"agent_system")
    adapter_file=(adapter/"adapter_model.safetensors") if cfg.get("sd_adapter") else None
    if adapter_file is not None: check_hash(adapter_file,cfg["adapter_sha256"],"adapter")
    elif cfg["adapter_sha256"] not in {"none", ""}: raise SystemExit("adapter SHA is set but sd_adapter path is absent")
    if not a.seed_root.is_dir(): raise SystemExit("seed root does not exist")
    tasks=manifest["tasks"]
    if [int(x["probe_ordinal"]) for x in tasks]!=list(range(64)): raise SystemExit("panel order/coverage mismatch")
    for t in tasks:
        if t.get("worker_seed")!=0 or t.get("source_session_index")!=t.get("goal_idx") or hashlib.sha256(t["instruction"].encode()).hexdigest()!=t.get("instruction_sha256"): raise SystemExit("task identity mismatch")
    if a.output.exists() and not a.resume: raise SystemExit("output already exists; pass --resume to continue the same endpoint")
    if not a.output.exists() and a.resume: raise SystemExit("cannot resume: endpoint output path does not exist")
    a.output.mkdir(parents=True,exist_ok=True)
    shared=q09.shared
    identity={"schema":"openevo.advisor.q09.stage-a-endpoint-run-identity.v1","panel_sha256":panel_sha,"endpoint_config_sha256":cfg_sha,"endpoint_id":cfg["endpoint_id"],"evaluation_stage":stage,"gpu_uuid":a.gpu_uuid,"seed_root":str(a.seed_root.resolve()),"task_count":64}
    idpath=a.output/"run-identity.json"
    if idpath.exists():
        if json.loads(idpath.read_text())!=identity: raise SystemExit("resume identity mismatch; refusing to mix endpoint, panel, seed root, or GPU")
    else:
        if a.resume: raise SystemExit("cannot resume without run-identity.json")
        atomic(idpath,identity)
    rows=[]
    for t in tasks:
        ordinal=int(t["probe_ordinal"]); episode_path=a.output/f"episode-{ordinal:03d}.json"; inflight_path=a.output/f"inflight-{ordinal:03d}.json"
        if episode_path.exists():
            row=json.loads(episode_path.read_text())
            if row.get("probe_ordinal")!=ordinal or row.get("task_id")!=t["task_id"] or row.get("panel_sha256")!=panel_sha or row.get("endpoint_config_sha256")!=cfg_sha or row.get("endpoint_id")!=cfg["endpoint_id"]:
                raise SystemExit(f"saved episode {ordinal} identity mismatch; refusing resume")
            if row.get("engineering_invalid") or not row.get("identity_ok") or not row.get("trace_complete"):
                raise SystemExit(f"saved episode {ordinal} is terminal/invalid; preserve consumed work and stop for review")
            rows.append(row)
            continue
        if inflight_path.exists(): raise SystemExit(f"episode {ordinal} has an unresolved STARTED marker; preserve state and stop instead of consuming it again")
        break
    next_ordinal=len(rows)
    for j in range(next_ordinal+1,64):
        if (a.output/f"episode-{j:03d}.json").exists() or (a.output/f"inflight-{j:03d}.json").exists():
            raise SystemExit(f"resume state has a gap before episode {j}; refusing out-of-order execution")
    if len(rows)==64:
        print(json.dumps({"status":"ALREADY_COMPLETE","endpoint_id":cfg["endpoint_id"],"attempt_count":64},sort_keys=True)); return 0
    gpu_before=shared.gpu_snapshot(a.gpu_uuid); atomic(a.output/"gpu-before.json",gpu_before)
    runtime=shared.Harness2011Runtime(cfg["model"],cfg["model_revision"],sd_adapter=adapter if adapter_file else None,memory=memory.read_text(),skill_bundle=skill.read_text(),agent_system=agent.read_text())
    start=time.time(); failure=None
    try:
        for t in tasks[next_ordinal:]:
            ordinal=int(t["probe_ordinal"]); task={"task_id":t["task_id"],"source_session_index":int(t["source_session_index"]),"worker_seed":int(t["worker_seed"]),"instruction":t["instruction"]}
            inflight_path=a.output/f"inflight-{ordinal:03d}.json"
            atomic(inflight_path,{"schema":"openevo.advisor.q09.stage-a-inflight.v1","status":"STARTED","probe_ordinal":ordinal,"task_id":task["task_id"],"panel_sha256":panel_sha,"endpoint_config_sha256":cfg_sha,"endpoint_id":cfg["endpoint_id"],"paired_generation_seed":t["paired_generation_seed"],"gpu_uuid":a.gpu_uuid,"started_at_unix":time.time()})
            raw=None; error=None; ep=None
            try:
                raw=shared.WebshopEnv(shared.WebshopEnvConfig(seed_root=str(a.seed_root),num_products=1000,seed=task["worker_seed"]))
                ep=shared.run_episode(shared.FixedEnvironmentSeed(raw,task["worker_seed"]),runtime,task,int(t["paired_generation_seed"]),harness_profile="201.1",step_seed_policy="injective-v1")
            except BaseException as exc: error=f"{type(exc).__name__}: {exc}"
            finally:
                if raw is not None: raw.close()
            ep=ep or {}; trace=ep.get("trace",[]); identity_ok=bool(trace and task["instruction"] in str(trace[0].get("prompt",""))) if error is None else False
            row={"schema":"openevo.advisor.q09.stage-a-confirmation-attempt.v1","panel_sha256":panel_sha,"probe_ordinal":ordinal,"task_id":task["task_id"],"goal_idx":t["goal_idx"],"task_identity":t["task_identity"],"instruction_sha256":t["instruction_sha256"],"worker_seed":task["worker_seed"],"paired_generation_seed":t["paired_generation_seed"],"endpoint_id":cfg["endpoint_id"],"evaluation_stage":stage,"backbone":cfg["backbone"],"treatment":cfg["treatment"],"stream_id":cfg["stream_id"],"state_id":cfg["state_id"],"endpoint_config_sha256":cfg_sha,"adapter_sha256":cfg["adapter_sha256"],"memory_sha256":cfg["memory_sha256"],"skill_sha256":cfg["skill_sha256"],"agent_system_sha256":cfg["agent_system_sha256"],"experiment_git_sha":cfg["experiment_git_sha"],"config_sha256":cfg["config_sha256"],"image_digest":cfg["image_digest"],"gpu_uuid":a.gpu_uuid,"task_score":float(ep["task_score"]) if error is None and "task_score" in ep else None,"won":bool(ep.get("won")) if error is None else None,"steps":int(ep.get("steps",0)) if error is None else None,"trace_complete":bool(ep.get("trace_complete")) if error is None else False,"identity_ok":identity_ok,"engineering_invalid":error is not None,"error":error,"parser_repair_used":bool(ep.get("parser_repair_used")) if error is None else None,"terminated_on_invalid_action":bool(ep.get("terminated_on_invalid_action")) if error is None else None,"constraint_escape":bool(ep.get("constraint_escape")) if error is None else None,"environment_action_schema_violation":bool(ep.get("environment_action_schema_violation")) if error is None else None,"webshop_env_step_calls":int(ep.get("webshop_env_step_calls",0)) if error is None else None,"episode":ep if error is None else None,"final_panel_access":0}
            atomic(a.output/f"episode-{ordinal:03d}.json",row)
            atomic(inflight_path,{"schema":"openevo.advisor.q09.stage-a-inflight.v1","status":"TERMINAL_ERROR" if error else "COMPLETED","probe_ordinal":ordinal,"task_id":task["task_id"],"panel_sha256":panel_sha,"endpoint_config_sha256":cfg_sha,"endpoint_id":cfg["endpoint_id"],"paired_generation_seed":t["paired_generation_seed"],"gpu_uuid":a.gpu_uuid,"finished_at_unix":time.time(),"error":error})
            rows.append(row)
            if error is not None:
                failure=f"episode {ordinal} failed: {error}"; break
    finally: runtime.close()
    gpu_after=shared.gpu_snapshot(a.gpu_uuid); atomic(a.output/"gpu-after.json",gpu_after)
    invalid=sum(int(bool(x["engineering_invalid"]) or not x["identity_ok"] or not x["trace_complete"]) for x in rows)
    complete=(len(rows)==64 and invalid==0 and failure is None)
    summary={"schema":"openevo.advisor.q09.stage-a-confirmation-endpoint-summary.v1","status":"PASS" if complete else "PARTIAL","endpoint_id":cfg["endpoint_id"],"evaluation_stage":stage,"panel_sha256":panel_sha,"endpoint_config_sha256":cfg_sha,"backbone":cfg["backbone"],"treatment":cfg["treatment"],"stream_id":cfg["stream_id"],"state_id":cfg["state_id"],"attempt_count":sum(1 for p in a.output.glob("episode-*.json")),"completed_valid_count":sum(1 for x in rows if not x["engineering_invalid"] and x["identity_ok"] and x["trace_complete"]),"engineering_invalid_count":invalid,"failure":failure,"mean_task_score":sum(x["task_score"] for x in rows)/len(rows) if rows and invalid==0 else None,"exact_success_count":sum(int(x["task_score"]==1.0 and x["won"]) for x in rows if x["task_score"] is not None),"final_panel_access":0,"automatic_retries":0,"episode_budget":64,"wall_time_seconds":time.time()-start,"rows":[{k:x.get(k) for k in ("probe_ordinal","task_id","task_identity","task_score","won","engineering_invalid","error")} for x in rows]}
    atomic(a.output/"summary.json",summary)
    print(json.dumps({k:summary[k] for k in ("status","endpoint_id","panel_sha256","attempt_count","engineering_invalid_count","failure")},sort_keys=True))
    return 0 if complete else 2

if __name__=="__main__": raise SystemExit(main())
