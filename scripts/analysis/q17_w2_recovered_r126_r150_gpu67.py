#!/usr/bin/env python3
"""Frozen W2: exact native-restored Q17 R126/R150 on GPU6 and GPU7 only.

CPU preflight -> model-load qualifier (0 WebShop) -> explicit 64-task per-arm
DEVELOPMENT probe -> full SHA-bound paired receipt. No training or final panel.
The original run and recovered adapter directories are ALWAYS read-only mounts.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
from typing import Any

PLAN_REL=Path("configs/experiment/metric-campaigns/q17-w2-r126-r150-gpu67-20261013.json")
EXPECTED_SCHEMA="openevo.q17.w2.rehydrated_adapters_gpu67_frozen_dev64.v1"
EXPECTED_UUID={6:"GPU-00000000-0000-0000-0000-000000000000",
               7:"GPU-00000000-0000-0000-0000-000000000000"}
EXPECTED_ADAPTER={"r126":"74cdc9d37d4adc9bd01db22cdafa3a7b5f35d3cde5c2247d8a2467ed529ae973",
                  "r150":"decf98f62076f66aff1a54c71a91e7df6bbfcdf20700271cfa76808d2e405a54"}
EXPECTED_CONFIG={"r126":"dff7f86e4d143414a58d050477b709ed770bb5722b71b34a2ef91329c9df71e3",
                 "r150":"7e297952d5d4055e7eaf9855003a4875eff401814b1fa2709bf30406f9380ec4"}
ARMS=("r126","r150")
RUN_ID=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,64}$")

class Hold(RuntimeError):pass

def read(path:Path)->dict:
    data=json.loads(path.read_bytes())
    if not isinstance(data,dict):raise Hold("JSON root not an object")
    return data

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4<<20),b""):
            h.update(block)
    return h.hexdigest()

def check_sha(path:Path,expected:str,label:str)->None:
    if not path.is_file() or path.is_symlink() or sha(path)!=expected:
        raise Hold(label+" exact file SHA mismatch/missing")

def canon(obj:Any)->bytes:
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode("utf-8")

def write_once(path:Path,data:bytes)->None:
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())

def command(argv:list[str],timeout:int=15)->str:
    cp=subprocess.run(argv,text=True,capture_output=True,timeout=timeout,check=False)
    if cp.returncode:raise Hold("read-only command failed: "+argv[0]+" rc="+str(cp.returncode)+" stderr="+cp.stderr[-140:])
    return cp.stdout

def verify_plan(plan:dict)->None:
    if plan.get("schema")!=EXPECTED_SCHEMA:raise Hold("unrecognized science contract")
    safety=plan["safety"]
    if (safety.get("exact_gpu_indices")!=[6,7]
        or safety.get("exact_gpu_uuid")!=[EXPECTED_UUID[6],EXPECTED_UUID[7]]
        or safety.get("forbidden_gpu_indices")!=list(range(6))):
        raise Hold("GPU0-5 forbidden and GPU6/7 exact UUID mandatory")
    for key,required in [("maximum_new_development_episodes",128),
                         ("task_count_each_arm",64),("horizon",15),
                         ("formal_task_consumption",0),("final_panel_access",0),
                         ("new_training_updates",0),("original_checkpoint_modifications",0)]:
        if safety.get(key)!=required:raise Hold("science budget mismatch "+key)
    frozen=plan["frozen_hashes"]
    if (frozen["panel_id"]!="4033523c01697576b7f9ce8cddc3a63cf6c1514ef5554e01a8c82b59ee62d478"
        or frozen["panel_manifest"]!="2d5d1620065a15840ae20fed324038312aec74b0895e13c9d283177928bf1a3b"
        or frozen["probe_script"]!="2906ef13392f182d1cf4a644b69ff19cd8f177c74b35ffd18896459e77130070"
        or frozen["qualifier_script"]!="84b967524c90bb9dea8397ac5fb1706cddbb94b80bf8fbc120a7079928a4f1b3"):
        raise Hold("source scientific panel/probe drift")
    if set(plan["arms"])!=set(ARMS):raise Hold("must have exactly recovered R126/R150")
    for arm,round_index,index in (("r126",126,6),("r150",150,7)):
        a=plan["arms"][arm]
        if (a["round"]!=round_index or a["index"]!=index
            or a["update_count"]!=round_index+1
            or a["adapter_sha256"]!=EXPECTED_ADAPTER[arm]
            or a["config_sha256"]!=EXPECTED_CONFIG[arm]
            or a["adapter_dir_rel"]!="round-"+str(round_index)):
            raise Hold(arm+" source/recovered identity changed")
    if (plan["docker_image_id"]!="sha256:d0f8796010899ae41a7af8ae3f44f47b96dea73b297b5af124049b3ae88e2850"
        or plan["base_model_revision"]!="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        or plan["runtime_execution_sha"]!="ac130148ee08b6462d9728e482a858fe5f514047"
        or plan["core_execution_sha"]!="59478929d884ca794cce2cb49cc6cc16103d6abe"):
        raise Hold("runtime/model/image changes not authorized")
    if plan["evaluation"]["sampling"]!={"max_steps":15,"max_new_tokens":512,"temperature":1,"top_p":1,"top_k":0,"repetition_penalty":1,"step_seed_policy":"paired_generation_seed + step_index * 100000"}:
        raise Hold("sampling contract drift")
    if plan["evaluation"]["instrumentation"]!="on" or plan["evaluation"]["enable_q06_pre_action_diagnostics"] is not True:
        raise Hold("instrumentation required")


def paths(plan:dict)->dict:
    v={k:Path(x) for k,x in plan["paths"].items() if not k.endswith("_rel")}
    home=Path("/workspace")
    if not all(home in v[k].parents for k in ("source_repo_root","original_q17_root","independent_reconstructed_adapters","results_root","runtime_root","core_root","base_model_root","seed_root","lock_root")):
        raise Hold("source/outputs must be under owner workspace")
    if v["results_root"] in v["original_q17_root"].parents or v["original_q17_root"] in v["results_root"].parents:
        raise Hold("new science output overlaps original Q17")
    if v["results_root"] in v["independent_reconstructed_adapters"].parents or v["independent_reconstructed_adapters"] in v["results_root"].parents:
        raise Hold("science output overlaps rehydrated adapter source")
    return v


def preflight(plan:dict,full:bool)->dict:
    verify_plan(plan);v=paths(plan);f=plan["frozen_hashes"]
    science=v["source_repo_root"]
    panel=science/plan["paths"]["panel_manifest_rel"]
    check_sha(panel,f["panel_manifest"],"development panel")
    src=read(panel)
    if src["panel_sha256"]!=f["panel_id"] or src["panel_size"]!=64 or src["final_panel_access_allowed"] is not False or src["formal_task_consumption_allowed"] is not False:
        raise Hold("frozen task list not intact")
    expected={"max_new_tokens":512,"max_steps":15,"repetition_penalty":1.0,
              "step_seed_policy":"paired_generation_seed + step_index * 100000",
              "temperature":1.0,"top_k":0,"top_p":1.0}
    if src["sampling"]!=expected:raise Hold("frozen evaluator sampling changes")
    for k,file,expected_sha in [
      ("probe",plan["paths"]["probefile_rel"],f["probe_script"]),
      ("qualifier",plan["paths"]["qualifier_rel"],f["qualifier_script"])
    ]:
        check_sha(science/file,expected_sha,k)
    if (v["runtime_root"]/"EXECUTION_SHA").read_text().strip()!=plan["runtime_execution_sha"] or (v["core_root"]/"EXECUTION_SHA").read_text().strip()!=plan["core_execution_sha"]:
        raise Hold("core/runtime execution source drift")
    for name,h in src["runtime_files"].items():
        check_sha(v["runtime_root"]/name,h,"frozen runtime "+name)
    if src["model"]["local_model_path"]!=str(v["base_model_root"]) or src["model"]["revision"]!=plan["base_model_revision"]:
        raise Hold("base model identity differs from frozen development source")
    q17=v["original_q17_root"]
    for name,rel,hashname in (
      ("memory",plan["carriers"]["memory"],f["memory"]),
      ("skill",plan["carriers"]["skill"],f["skill"]),
      ("agent",plan["carriers"]["agent"],f["agent"])
    ):
        check_sha(q17/rel,hashname,"carrier "+name)
    adapters={}
    for arm in ARMS:
        a=plan["arms"][arm];src_dir=v["independent_reconstructed_adapters"]/a["adapter_dir_rel"]
        if src_dir.is_symlink() or not src_dir.is_dir():raise Hold("no recovered adapter stage "+arm)
        weights=src_dir/"adapter_model.safetensors"
        cfg=src_dir/"adapter_config.json"
        if not weights.is_file() or weights.stat().st_size<1_000_000:
            raise Hold("adapter missing/partial "+arm)
        check_sha(cfg,a["config_sha256"],"adapter config "+arm)
        if full:check_sha(weights,a["adapter_sha256"],"recovered exact adapter "+arm)
        if not full and weights.is_symlink():raise Hold("symlink source adapter refused")
        prior=read(q17/f"round-{a['round']:03d}"/"sd-lora/update.json")
        if prior["round_index"]!=a["round"] or prior["adapter_digest_after"]!=a["adapter_sha256"]:
            raise Hold("historical producer identity differs "+arm)
        adapters[arm]={"source_sha256":a["adapter_sha256"],"bytes":weights.stat().st_size,"recovery_verified":bool(full)}
    filesystem=os.statvfs(v["results_root"].parent)
    free=filesystem.f_bavail*filesystem.f_frsize
    if free<plan["safety"]["minimum_free_disk_bytes"]:raise Hold("not enough free disk")
    if not v["lock_root"].is_dir():raise Hold("exact system shared GPU lock path absent")
    return {"status":"PASS_CPU_SCIENCE_CONTRACT","adapter_sources":adapters,
            "frozen_panel_sha256":f["panel_id"],"read_only_source_checks":True,
            "disk_available_bytes":free,"no_gpu_calls":True,"new_webshop_calls":0}


def live_gpus(plan:dict)->list:
    result=command(["nvidia-smi","--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],15)
    apps=command(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],15)
    holders={l.split(",")[0].strip() for l in apps.splitlines() if l.strip()}
    table={}
    for line in result.splitlines():
        z=[c.strip() for c in line.split(",")]
        if len(z)!=6:raise Hold("malformed nvidia-smi inventory")
        table[int(z[0])]=z
    out=[]
    for i in (6,7):
        if i not in table:raise Hold("GPU6/7 absent")
        z=table[i];uuid=z[1];used=int(z[3]);capacity=int(z[4])
        if (uuid!=EXPECTED_UUID[i] or "RTX 5090" not in z[2]
            or uuid in holders or used>plan["safety"]["maximum_live_prelaunch_vram_used_mib"]
            or capacity-used<plan["safety"]["minimum_free_vram_mib"]):
            raise Hold("GPU"+str(i)+" is busy, reassigned, or not qualified; no takeover")
        out.append({"index":i,"uuid":uuid,"memory_used_mib":used,"gpu_util_pct":int(z[5])})
    return out


@contextlib.contextmanager
def gpu_flocks(plan:dict):
    root=paths(plan)["lock_root"];streams=[]
    try:
        for i in (6,7):
            h=(root/(EXPECTED_UUID[i]+".lock")).open("a+")
            try:fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as e:
                h.close();raise Hold("GPU"+str(i)+" live kernel lock already held") from e
            streams.append(h)
        yield
    finally:
        for h in reversed(streams):
            fcntl.flock(h.fileno(),fcntl.LOCK_UN);h.close()


def derive_manifest(plan:dict,arm:str)->dict:
    v=paths(plan)
    frozen=read(v["source_repo_root"]/plan["paths"]["panel_manifest_rel"])
    if frozen["checkpoints"]["after_160_updates"]["adapter_sha256"]!="18032c5cabd135b2464d5cbb2a3949aeb6130046dcc9f33acb33aee38c1baf46":
        raise Hold("canonical source panel checkpoint contract changed")
    a=plan["arms"][arm]
    result=copy.deepcopy(frozen)
    result["checkpoints"]={a["checkpoint_id"]:{
       "produced_by_update_round":a["round"],
       "update_count":a["update_count"],
       "state_id":a["state_id"],
       "adapter_sha256":a["adapter_sha256"]
    }}
    result["completed_stage2_rounds"]=a["update_count"]
    result["recovery_evidence"]="R126_R150_EXACT_ADAPTER_RECOVERY: original producer SHA exact"
    result["W2_arm"]=arm
    result["purpose"]="Q17 recovery-only new instrumented DEVELOPMENT evaluation, never final panel"
    return result


def prepare(plan:dict,run_id:str)->Path:
    if not RUN_ID.fullmatch(run_id):raise Hold("invalid run-id")
    root=paths(plan)["results_root"]
    root.mkdir(mode=0o770,exist_ok=True,parents=True)
    if os.geteuid()==0:os.chown(root,1001,1001)
    p=root/run_id
    p.mkdir(mode=0o770,exist_ok=True)
    if os.geteuid()==0:os.chown(p,1001,1001)
    for arm in ARMS:
        dest=p/arm
        dest.mkdir(mode=0o770,exist_ok=True)
        if os.geteuid()==0:os.chown(dest,1001,1001)
        doc=canon(derive_manifest(plan,arm))
        target=dest/"science_manifest.json"
        if target.exists():
            if target.read_bytes()!=doc:raise Hold("existing frozen science manifest drift "+arm)
        else:
            write_once(target,doc)
            if os.geteuid()==0:os.chown(target,1001,1001)
    return p


def run_args(plan:dict,root:Path,arm:str,kind:str)->list[str]:
    if arm not in ARMS or kind not in ("qualify","dev64"):raise Hold("unrecognized scientific arm/phase")
    v=paths(plan);a=plan["arms"][arm];uuid=EXPECTED_UUID[a["index"]]
    out=root/arm;adapter=v["independent_reconstructed_adapters"]/a["adapter_dir_rel"]
    pp=":".join([str(v["core_root"]/"src"),str(v["runtime_root"]),
           str(v["runtime_root"]/"vendor/openevo-webshop/src"),
           "/work/experiment",str(v["runtime_root"]/"vendor/openevo-webshop")])
    cmd=["docker","run","--rm","--restart","no","--network","none",
         "--gpus","device="+uuid,"--user","1001:1001","--cap-drop=ALL",
         "--security-opt","no-new-privileges","--pids-limit","1024",
         "--tmpfs","/tmp:rw,exec,nosuid,size=8g,mode=1777",
         "--label","openevo.owner=experiment-host","--label","openevo.experiment=q17-w2-r126-r150-20261013",
         "--label","openevo.gpu_uuid="+uuid,
         "-e","CUDA_VISIBLE_DEVICES="+uuid,"-e","NVIDIA_VISIBLE_DEVICES="+uuid,
         "-e","TRANSFORMERS_OFFLINE=1","-e","HF_HUB_OFFLINE=1",
         "-e","PYTHONDONTWRITEBYTECODE=1","-e","PYTHONPATH="+pp]
    for name in ("runtime_root","core_root","seed_root","base_model_root","original_q17_root","independent_reconstructed_adapters"):
        path=v[name]
        cmd.extend(["--mount",f"type=bind,source={path},target={path},readonly"])
    cmd.extend(["--mount",f"type=bind,source={v['source_repo_root']},target=/work/experiment,readonly"])
    cmd.extend(["--mount",f"type=bind,source={root},target={root}"])
    cmd.extend(["--entrypoint","/usr/local/bin/python3",plan["docker_image_id"]])
    args_common=["--runtime-source-root",str(v["runtime_root"]),
       "--model",str(v["base_model_root"]),"--revision",plan["base_model_revision"],
       "--memory",str(v["original_q17_root"]/plan["carriers"]["memory"]),
       "--skill",str(v["original_q17_root"]/plan["carriers"]["skill"]),
       "--agent",str(v["original_q17_root"]/plan["carriers"]["agent"])]
    if kind=="qualify":
        cmd.extend(["/work/experiment/"+plan["paths"]["qualifier_rel"],
                    *args_common,"--adapter",str(adapter),"--output",str(out/"qualification.json")])
    else:
        cmd.extend(["/work/experiment/"+plan["paths"]["probefile_rel"],
                    "--manifest",str(out/"science_manifest.json"),
                    "--panel-manifest","/work/experiment/"+plan["paths"]["panel_manifest_rel"],
                    "--checkpoint-id",a["checkpoint_id"],
                    *args_common,"--seed-root",str(v["seed_root"]),
                    "--sd-adapter",str(adapter),"--gpu-uuid",uuid,
                    "--output-dir",str(out/"dev64"),
                    "--instrumentation","on","--q06-pre-action-diagnostics",
                    "--q06-lineage-id","q17_"+arm+"_recovered_native_W2"])
    if [cmd[i+1] for i,z in enumerate(cmd) if z=="--gpus"]!=["device="+uuid]:
        raise Hold("GPU visibility mismatch")
    return cmd


def native_qualify(plan:dict,work:Path,arm:str)->dict:
    p=work/arm/"qualification.json"
    j=read(p)
    if (j.get("status")!="PASS" or j.get("generation_calls")!=0
        or j.get("formal_rollouts_consumed")!=0 or j.get("final_panel_access")!=0
        or j.get("inference_started") is not False
        or j.get("adapter_sha256")!=plan["arms"][arm]["adapter_sha256"]):
        raise Hold("model load qualification receipt incomplete/wrong "+arm)
    return j


def invoke(plan:dict,work:Path,arm:str,kind:str,timeout:int)->dict:
    cmd=run_args(plan,work,arm,kind)
    if kind=="dev64":
        p=work/arm/"dev64"
        p.mkdir(mode=0o770,exist_ok=False)
        if os.geteuid()==0:os.chown(p,1001,1001)
    stdout=work/arm/(kind+".stdout.log")
    stderr=work/arm/(kind+".stderr.log")
    started=time.monotonic()
    with stdout.open("xb") as out,stderr.open("xb") as err:
        try:cp=subprocess.run(cmd,stdout=out,stderr=err,timeout=timeout,check=False)
        except subprocess.TimeoutExpired as exc:
            raise Hold(arm+" timed out; preserve logs and avoid replay") from exc
    info={"status":"PASS" if cp.returncode==0 else "FAIL","arm":arm,"kind":kind,
          "exit_code":cp.returncode,"duration_s":time.monotonic()-started,
          "gpu_uuid":EXPECTED_UUID[plan["arms"][arm]["index"]],
          "stdout_sha256":sha(stdout),"stderr_sha256":sha(stderr)}
    if cp.returncode!=0:raise Hold(arm+"/"+kind+" Docker failed rc="+str(cp.returncode)+" logs="+str(stderr))
    return info


def verify_summary(plan:dict,work:Path,arm:str)->dict:
    j=read(work/arm/"dev64/summary.json")
    if (j.get("status")!="PASS" or j.get("task_count")!=64
        or j.get("panel_sha256")!=plan["frozen_hashes"]["panel_id"]
        or j.get("task_panel_manifest_sha256")!=plan["frozen_hashes"]["panel_manifest"]
        or j.get("instrumentation")!="on" or j.get("final_panel_access")!=0
        or j.get("formal_task_consumption")!=0
        or j.get("checkpoint",{}).get("adapter_sha256")!=plan["arms"][arm]["adapter_sha256"]
        or j.get("probe_script_sha256")!=plan["frozen_hashes"]["probe_script"]):
        raise Hold(arm+" failed exact science/panel/adapter identity")
    rows=j.get("rows",[])
    if len(rows)!=64 or any(x.get("engineering_invalid") for x in rows):
        raise Hold(arm+" incomplete or invalid 64 tasks")
    if any(any(s.get("generation_diagnostics") is None or s.get("q06_pre_action") is None
               for s in r["episode"]["trace"]) for r in rows):
        raise Hold(arm+" missing Q06 pre-action/entropy evidence")
    return j


def paired_report(plan:dict,work:Path)->dict:
    j={arm:verify_summary(plan,work,arm) for arm in ARMS}
    keys=("task_identity","paired_generation_seed","probe_ordinal","source_session_index","worker_seed")
    mapped={arm:{tuple(r[k] for k in keys):r for r in s["rows"]} for arm,s in j.items()}
    if len(mapped["r126"])!=64 or len(mapped["r150"])!=64 or set(mapped["r126"])!=set(mapped["r150"]):
        raise Hold("R126/R150 task and seed pairing mismatch")
    diffs=[]
    matched_initial=0
    matched_step_prefix=0
    for key in sorted(mapped["r126"]):
        left,right=(mapped[arm][key] for arm in ARMS)
        diffs.append(float(left["episode"]["score"])-float(right["episode"]["score"]))
        lt=left["episode"]["trace"];rt=right["episode"]["trace"]
        if not lt or not rt:raise Hold("empty matched trajectory")
        if lt[0]["q06_pre_action"]["prompt_sha256"]!=rt[0]["q06_pre_action"]["prompt_sha256"]:
            raise Hold("first pre-action prompt changed across R126/R150")
        matched_initial+=1
        matched_step_prefix+=sum(x["q06_pre_action"]["prompt_sha256"]==y["q06_pre_action"]["prompt_sha256"]
                                 for x,y in zip(lt,rt))
    m={}
    for arm in ARMS:
        s=j[arm]["metrics"]
        for name in ("reasoning_generation","action_generation","click_choice"):
            if s.get(name) is None:raise Hold(arm+" missing "+name+" science diagnostics")
        m[arm]={
          "adapter_sha256":plan["arms"][arm]["adapter_sha256"],
          "mean_task_score_0_to_100":100*float(s["mean_score"]),
          "exact_success_rate":s["success_rate"],
          "positive_reward_rate":s["positive_reward_rate"],
          "reasoning_token_entropy_nats":s["reasoning_generation"]["entropy_mean_nats"],
          "action_token_entropy_nats":s["action_generation"]["entropy_mean_nats"],
          "selected_action_nll_nats":s["selected_action_nll_mean_nats"],
          "click_destination_entropy_nats":s["click_choice"]["mean_entropy_nats"],
          "instrumented_steps":s["click_choice"]["observed_step_count"],
          "wall_time_seconds":j[arm]["wall_time_seconds"],
          "source_summary_sha256":sha(work/arm/"dev64/summary.json"),
        }
    rng=random.Random(12615020261013)
    sorted_boot=sorted(statistics.mean(diffs[rng.randrange(64)] for _ in range(64)) for _ in range(20_000))
    return {
      "schema":"openevo.q17.w2.recoveredR126R150.instrumented_dev64_paired.v1",
      "status":"PASS",
      "run_id":work.name,"panel_sha256":plan["frozen_hashes"]["panel_id"],
      "task_count_per_arm":64,
      "arms":m,"paired_R126_minus_R150_task_score":statistics.mean(diffs),
      "paired_bootstrap_95CI":[sorted_boot[500],sorted_boot[19500]],
      "same_initial_pre_action_prompts":matched_initial,
      "same_step_pre_action_prompt_pairs":matched_step_prefix,
      "science_limitations":["Only 64 frozen development tasks, not a final heldout test.",
         "Matched task/seed does not imply same visited WebShop state after divergent actions.",
         "Recovered adapter exact byte SHA establishes identity, not causal direction or rank effect."],
      "execution":{"gpu_indices":[6,7],"other_gpu_indices_used":[],"new_development_rollouts":128,
                   "final_panel_access":0,"formal_task_consumption":0,"training_updates":0,
                   "original_checkpoint_files_modified":0}}


def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase",choices=("preflight","qualify","dev64","report"),default="preflight")
    ap.add_argument("--plan",type=Path)
    ap.add_argument("--run-id",default="")
    ap.add_argument("--approve-development-rollouts",action="store_true")
    ap.add_argument("--full-hash",action="store_true")
    args=ap.parse_args(argv)
    try:
        plan=read(args.plan or (Path(__file__).resolve().parents[2]/PLAN_REL))
        verify_plan(plan)
        if args.phase=="report":
            if not RUN_ID.fullmatch(args.run_id):raise Hold("report requires exact run id")
            print(json.dumps(paired_report(plan,paths(plan)["results_root"]/args.run_id),sort_keys=True))
            return 0
        state=preflight(plan,full=args.full_hash or args.phase in ("qualify","dev64"))
        gpu=live_gpus(plan)
        if args.phase=="preflight":
            print(json.dumps({"status":"READY_FOR_LOAD_QUALIFICATION" if args.full_hash else "CPU_PREFLIGHT_PASS_SHA_PENDING",
                "science":state,"gpu6_7":gpu,"gpu_compute_calls":0},sort_keys=True))
            return 0
        if not RUN_ID.fullmatch(args.run_id):raise Hold("GPU phase requires fixed run id")
        if args.phase=="dev64" and not args.approve_development_rollouts:
            raise Hold("--approve-development-rollouts required for task-bearing phase")
        with gpu_flocks(plan):
            live_gpus(plan)
            work=prepare(plan,args.run_id)
            if args.phase=="qualify":
                for arm in ARMS:
                    if (work/arm/"qualification.json").exists() or (work/arm/"qualify.stdout.log").exists():
                        raise Hold("qualifier already exists; no blind rerun")
                    live_gpus(plan)
                    info=invoke(plan,work,arm,"qualify",900)
                    native_qualify(plan,work,arm)
                    print(json.dumps(info,sort_keys=True),flush=True)
                print(json.dumps({"status":"QUALIFIED_BOTH_WITH_ZERO_TASKS","gpu_indices":[6,7],
                    "model_calls":0,"final_panel_access":0},sort_keys=True))
                return 0
            for arm in ARMS:
                native_qualify(plan,work,arm)
                if (work/arm/"dev64").exists() or (work/arm/"dev64.stdout.log").exists():
                    raise Hold("existing task results/logs; no unsupervised replay "+arm)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futs=[pool.submit(invoke,plan,work,arm,"dev64",14400) for arm in ARMS]
                for fut in concurrent.futures.as_completed(futs):
                    print(json.dumps(fut.result(),sort_keys=True),flush=True)
            report=paired_report(plan,work)
            write_once(work/"paired-metrics.json",canon(report))
            print(json.dumps({"status":"PASS_PAIRED_R126_R150_64_DEV",
                "report_sha256":sha(work/"paired-metrics.json"),"gpu_indices":[6,7],
                "formal_task_consumption":0,"final_panel_access":0},sort_keys=True))
            return 0
    except (Hold,FileExistsError,FileNotFoundError,KeyError,ValueError,TypeError,subprocess.TimeoutExpired) as error:
        print(json.dumps({"status":"BLOCKED","reason":str(error),"gpu_scope":[6,7],
            "no_other_gpu_launch":True,"source_model_modified":False},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
