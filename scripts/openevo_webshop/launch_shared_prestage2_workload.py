#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p: Path): return json.loads(p.read_text(encoding="utf-8"))
def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--root",type=Path,required=True); p.add_argument("--arm",required=True); p.add_argument("--kind",choices=("text-memory","skill-bundle","agent-system","runtime-readback"),required=True); p.add_argument("--gpu-uuid",required=True); p.add_argument("--container",required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--reuse-from",type=Path); a=p.parse_args()
    repo=a.repo.resolve(); root=a.root.resolve(); contract_path=repo/"configs/experiment/stage1/202609030400/prestage2_shared.json"; contract=load(contract_path); spec=contract["arms"][a.arm]
    head=subprocess.check_output(["git","-c",f"safe.directory={repo}","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-c",f"safe.directory={repo}","-C",str(repo),"status","--porcelain"],text=True).strip()
    if dirty: raise SystemExit("execution repo dirty")
    if subprocess.run(["docker","inspect",a.container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0: raise SystemExit("immutable container exists")
    mm=load(root/f"MINIMAX_COMPLETE_{a.arm}.json"); op=load(root/f"OPSD_COMPLETE_{a.arm}.json")
    if mm.get("status")!="PASS" or int(mm.get("parse_ok",-1))!=1440 or op.get("status")!="PASS": raise SystemExit("upstream not closed")
    analysis=root/f"{a.arm}-minimax.jsonl"; adapter=Path(op["final_adapter_path"]); model=Path(spec["model_path"])
    if subprocess.run(["install","-d","-o","1001","-g","1001","-m","700",str(a.output.resolve())]).returncode:
        raise SystemExit("failed to create uid1001-owned output directory")
    common=["--analysis","/workspace/analysis.jsonl","--analysis-sha256",mm["analysis_corpus_sha256"],"--model","/workspace/model","--model-id",spec["model_id"],"--model-revision",spec["revision"],"--adapter","/workspace/adapter","--output","/workspace/out","--gpu-uuid",a.gpu_uuid,"--experiment-git-sha",head,"--config-sha256",sha(contract_path),"--expected-context-window",str(spec["context_window"]),"--chat-template-enable-thinking",("false" if spec["chat_template_enable_thinking"] is False else "auto")]
    mounts=["-v",f"{repo}:/workspace/repo:ro","-v",f"{analysis}:/workspace/analysis.jsonl:ro","-v",f"{model}:/workspace/model:ro","-v",f"{adapter}:/workspace/adapter:ro","-v",f"{a.output.resolve()}:/workspace/out"]
    if a.kind=="runtime-readback":
        tm=load(root/f"TEXT_MEMORY_COMPLETE_{a.arm}.json"); sk=load(root/f"SKILL_BUNDLE_COMPLETE_{a.arm}.json"); ag=load(root/f"AGENT_SYSTEM_COMPLETE_{a.arm}.json")
        core=Path("/workspace/control/openevo-core")
        cmd=["docker","run","-d","--name",a.container,"--restart","no","--network","none","--user","1001:1001","--gpus",f"device={a.gpu_uuid}","-e",f"CUDA_VISIBLE_DEVICES={a.gpu_uuid}","-v",f"{core}:/workspace/core:ro","-v",f"{repo}:/workspace/repo:ro","-v",f"{model}:/workspace/model:ro","-v",f"{adapter}:/workspace/adapter:ro","-v",f"{tm['host_memory_path']}:/workspace/memory.md:ro","-v",f"{sk['host_skill_path']}:/workspace/SKILL.md:ro","-v",f"{ag['host_agent_system_path']}:/workspace/agent_system.md:ro","-v",f"{a.output.resolve()}:/workspace/out","-e","PYTHONPATH=/workspace/core/src:/workspace/repo:/workspace/repo/vendor/openevo-webshop/src","--entrypoint","/usr/local/bin/python3","experiment-host/openevo-scientific:h144-runtime-v1","/workspace/repo/scripts/openevo_webshop/verify_ceiling1_runtime_carriers.py","--model","/workspace/model","--revision",spec["revision"],"--bootstrap-adapter","/workspace/adapter","--memory","/workspace/memory.md","--skill","/workspace/SKILL.md","--agent-system","/workspace/agent_system.md","--gpu-uuid",a.gpu_uuid,"--output","/workspace/out/runtime-readback.json"]
        run(cmd); print(f"STARTED {a.container} kind={a.kind} arm={a.arm}"); return 0
    if a.kind=="text-memory":
        script="ceiling1_build_text_memory_a2.py"
        if a.reuse_from:
            mounts += ["-v",f"{a.reuse_from.resolve()}:/workspace/reuse:ro"]
            extra=["--reuse-from","/workspace/reuse"]
        else:
            extra=[]
    elif a.kind=="skill-bundle": script="ceiling1_build_skill_bundle_fixed20.py"; extra=[]
    else:
        script="ceiling1_build_agent_system_v2.py"
        tm=load(root/f"TEXT_MEMORY_COMPLETE_{a.arm}.json"); sk=load(root/f"SKILL_BUNDLE_COMPLETE_{a.arm}.json")
        mounts += ["-v",f"{tm['host_memory_path']}:/workspace/memory.md:ro","-v",f"{sk['host_skill_path']}:/workspace/SKILL.md:ro"]
        extra=["--memory","/workspace/memory.md","--skill","/workspace/SKILL.md"]
    cmd=["docker","run","-d","--name",a.container,"--restart","no","--network","none","--user","1001:1001","--gpus",f"device={a.gpu_uuid}","-e",f"CUDA_VISIBLE_DEVICES={a.gpu_uuid}"]+mounts+["-e","PYTHONPATH=/workspace/repo:/workspace/repo/vendor/openevo-webshop/src","--entrypoint","/usr/local/bin/python3","experiment-host/openevo-scientific:h144-runtime-v1",f"/workspace/repo/scripts/openevo_webshop/{script}"]+common+extra
    run(cmd); print(f"STARTED {a.container} kind={a.kind} arm={a.arm}"); return 0

if __name__=="__main__": raise SystemExit(main())
