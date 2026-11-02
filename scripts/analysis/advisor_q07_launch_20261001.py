#!/usr/bin/env python3
import concurrent.futures, fcntl, hashlib, json, os, shutil, subprocess
from pathlib import Path

ROOT=Path("/workspace/maintenance/ADVISOR-Q07-TASK-VECTOR-CAUSAL-20261001")
WT=Path("/workspace/tmp/q07-sync-main-20261001")
OLD=Path("/workspace/maintenance/Q17-VALIDATION-CURVE-202609231600")
RUNTIME=OLD/"runtime-ac130148"; CORE=OLD/"core-594789"
MANIFEST=WT/"configs/experiment/manifests/advisor-q07-task-vector-causal-20261001.json"
PROBE=WT/"scripts/analysis/advisor_q07_fixed_checkpoint_probe_20261001.py"
QUALIFIER=WT/"scripts/analysis/advisor_q07_load_qualifier_20261001.py"
RUNS=Path("/workspace/runs")
Q17_RUN=RUNS/"q17-direct-apply-only-w2-202609082022/q17-direct-apply-only-w2-202609082022-direct"
SEED_ROOT=RUNS/"phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
MODEL=RUNS/"20260903-shared-stage1-minimax-m3-202609030400/stage2-base/qwen3-1p7b"
Q01_SUMMARY=Path("/workspace/maintenance/ADVISOR-PLATEAU-COMPARE-20260930/results/off/after_160_rounds/summary.json")
REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"; IMAGE="experiment-host/openevo-scientific:h144-runtime-v1"
MEMORY=Q17_RUN/"round-095/text-memory/memory.md"; SKILL=Q17_RUN/"round-000/skill-bundle/SKILL.md"; AGENT=Q17_RUN/"round-115/agent-system/agent_system.md"
OUT=ROOT/"results"; STAGING=ROOT/"adapter-staging"; LOCKROOT=Path("/workspace/.gpu-locks")
PROBE_SHA="e79383fcbe8b5b8c26b86ccc878bf8363c684ea708127dae0614520348ebd571"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def static_check():
    m=json.loads(MANIFEST.read_text())
    assert sha(PROBE)==PROBE_SHA
    assert (RUNTIME/"EXECUTION_SHA").read_text().strip()=="ac130148ee08b6462d9728e482a858fe5f514047"
    assert (CORE/"EXECUTION_SHA").read_text().strip()=="59478929d884ca794cce2cb49cc6cc16103d6abe"
    assert sha(MEMORY)==m["common_carriers"]["memory_sha256"]
    assert sha(SKILL)==m["common_carriers"]["skill_sha256"]
    assert sha(AGENT)==m["common_carriers"]["agent_system_sha256"]
    for cid,c in m["checkpoints"].items():
        assert sha(Path(c["adapter_path"])/"adapter_model.safetensors")==c["adapter_sha256"], cid
    stage=json.loads((ROOT/"ADAPTER_STAGING_RECEIPT.json").read_text())
    assert stage["status"]=="PASS" and stage["state_count"]==17 and stage["historical_source_mutated"] is False
    for cid,c in m["checkpoints"].items():
        if cid=="intact_w159": continue
        rec=stage["records"][cid]; sp=Path(rec["staging_path"])
        assert sha(sp/"adapter_model.safetensors")==c["adapter_sha256"]
        assert sha(sp/"adapter_config.json")==rec["config_sha256"]
    audit=json.loads((WT/"docs/evidence/advisor-q07-task-vector-causal-20261001/Q07_INTACT_REUSE_AUDIT.json").read_text())
    assert audit["status"]=="PASS" and sha(Q01_SUMMARY)==audit["source_summary_sha256"]
    return m,audit

def gpu_rows():
    cp=subprocess.run(["docker","run","--rm","--gpus","all","--entrypoint","bash",IMAGE,"-lc",
      "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits; echo COMPUTE; nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true"],
      text=True,capture_output=True,check=True)
    lines=cp.stdout.splitlines(); cut=lines.index("COMPUTE")
    all_rows=[]; rows=[]
    for line in lines[:cut]:
        idx,uuid,used,util=[x.strip() for x in line.split(",")]
        row=(int(idx),uuid,int(used),int(util)); all_rows.append(row)
        if int(used)<=256: rows.append(row)
    compute=[x.strip() for x in lines[cut+1:] if x.strip()]
    if not rows: raise RuntimeError("no live-idle GPU passes <=256 MiB gate")
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/"resource-gate.tsv").open("w") as f:
        f.write("index\tuuid\tmemory_used_mib\tutilization_pct\tadmission\n")
        for x in all_rows:
            f.write("\t".join(map(str,x))+("\tADMIT\n" if x in rows else "\tSKIP_OCCUPIED\n"))
    (ROOT/"resource-gate-compute-apps.txt").write_text("\n".join(compute)+("\n" if compute else ""))
    return rows

def lock_all(rows):
    LOCKROOT.mkdir(parents=True,exist_ok=True); hs=[]
    for _,uuid,_,_ in rows:
        h=open(LOCKROOT/f"{uuid}.lock","a+")
        try: fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError(f"GPU lock busy: {uuid}")
        hs.append(h)
    return hs

def common_docker(uuid):
    py=f"{CORE}/src:{RUNTIME}:{RUNTIME}/vendor/openevo-webshop/src:{OLD}/repo:{OLD}/repo/vendor/openevo-webshop/src"
    return ["docker","run","--rm","--restart","no","--gpus",f"device={uuid}","--network","none","--user","1001:1001",
      "--tmpfs","/tmp:rw,exec,nosuid,size=8g,mode=1777","-e",f"CUDA_VISIBLE_DEVICES={uuid}","-e","TRANSFORMERS_OFFLINE=1","-e","HF_HUB_OFFLINE=1","-e",f"PYTHONPATH={py}",
      "-v",f"{OLD}:{OLD}:ro","-v",f"{WT}:{WT}:ro","-v",f"{RUNS}:{RUNS}:ro","-v",f"{ROOT}:{ROOT}:ro","-v",f"{OUT}:{OUT}"]

def qualify(m,uuid):
    q=OUT/"load-qualification.json"
    adapter=str(STAGING/"loo_r152")
    args=common_docker(uuid)+["--entrypoint","/usr/local/bin/python3",IMAGE,str(QUALIFIER),
      "--runtime-source-root",str(RUNTIME),"--model",str(MODEL),"--revision",REVISION,"--adapter",adapter,
      "--memory",str(MEMORY),"--skill",str(SKILL),"--agent",str(AGENT),"--output",str(q)]
    cp=subprocess.run(args)
    if cp.returncode: raise RuntimeError("rank384 load qualification failed")
    z=json.loads(q.read_text())
    if z["status"]!="PASS" or z["inference_started"] or z["formal_rollouts_consumed"]!=0: raise RuntimeError("bad load qualification")

def reuse_intact(m,audit):
    dest=OUT/"intact_w159"; dest.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Q01_SUMMARY,dest/"summary.json")
    z=json.loads((dest/"summary.json").read_text())
    if z["panel_sha256"]!=m["panel_sha256"] or z["checkpoint"]["adapter_sha256"]!=m["checkpoints"]["intact_w159"]["adapter_sha256"]:
        raise RuntimeError("intact reuse identity drift")
    receipt={"schema":"openevo.advisor.q07.reuse-receipt.v1","status":"PASS","source":str(Q01_SUMMARY),
      "source_sha256":audit["source_summary_sha256"],"copied_sha256":sha(dest/"summary.json"),"new_episodes":0,"final_panel_access":0}
    (ROOT/"INTACT_REUSE_RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")

def run_one(cid,uuid,m):
    c=m["checkpoints"][cid]; adapter=STAGING/cid; dest=OUT/cid
    dest.mkdir(parents=True,exist_ok=True); os.chown(dest,1001,1001); os.chmod(dest,0o770)
    summary=dest/"summary.json"
    if summary.is_file():
        z=json.loads(summary.read_text())
        if (z.get("status")=="PASS" and z.get("task_count")==64 and
            z.get("panel_sha256")==m["panel_sha256"] and
            z.get("checkpoint",{}).get("adapter_sha256")==c["adapter_sha256"] and
            not any(bool(r.get("engineering_invalid")) for r in z.get("rows",[]))):
            return cid,z["metrics"]["mean_score"],z["metrics"]["success_rate"]
    args=common_docker(uuid)+["--entrypoint","/usr/local/bin/python3",IMAGE,str(PROBE),
      "--manifest",str(MANIFEST),"--checkpoint-id",cid,"--runtime-source-root",str(RUNTIME),"--seed-root",str(SEED_ROOT),
      "--model",str(MODEL),"--revision",REVISION,"--sd-adapter",str(adapter),"--memory",str(MEMORY),"--skill",str(SKILL),
      "--agent",str(AGENT),"--gpu-uuid",uuid,"--output-dir",str(dest),"--instrumentation","off"]
    with (dest/"stdout.log").open("w") as so,(dest/"stderr.log").open("w") as se:
        cp=subprocess.run(args,stdout=so,stderr=se)
    if cp.returncode: raise RuntimeError(f"probe failed {cid} rc={cp.returncode}")
    s=json.loads((dest/"summary.json").read_text())
    if s["status"]!="PASS" or s["task_count"]!=64 or s["panel_sha256"]!=m["panel_sha256"]: raise RuntimeError(f"bad summary {cid}")
    return cid,s["metrics"]["mean_score"],s["metrics"]["success_rate"]

def lane(uuid,cids,m):
    out=[]
    for cid in cids:
        r=run_one(cid,uuid,m); print("RESULT",*r,flush=True); out.append(r)
    return out

def main():
    m,audit=static_check(); rows=gpu_rows(); locks=lock_all(rows)
    OUT.mkdir(parents=True,exist_ok=True); os.chown(OUT,1001,1001); os.chmod(OUT,0o770)
    print("GATE_PASS",rows,flush=True)
    qualify(m,rows[0][1]); print("LOAD_QUALIFICATION_PASS",flush=True)
    reuse_intact(m,audit); print("INTACT_REUSE_PASS",flush=True)
    cids=["r151_reference"]+[f"loo_r{x}" for x in range(152,160)]+[f"reverse_r{x}" for x in range(152,160)]
    lanes=[[] for _ in rows]
    for i,cid in enumerate(cids): lanes[i%len(rows)].append(cid)
    print("LANES",json.dumps([{"gpu":rows[i][1],"states":lanes[i]} for i in range(len(rows))]),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as ex:
        fs=[ex.submit(lane,rows[i][1],lanes[i],m) for i in range(len(rows))]
        for f in concurrent.futures.as_completed(fs): f.result()
    print("Q07_EVALUATION_COMPLETE",flush=True)

if __name__=="__main__": main()
