#!/usr/bin/env python3
import argparse, hashlib, json, sys, time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

ap=argparse.ArgumentParser()
for name in ("runtime_source_root","model","revision","adapter","memory","skill","agent","output"):
    ap.add_argument("--"+name.replace("_","-"), required=True)
a=ap.parse_args()
started=time.time()
sys.path.insert(0,a.runtime_source_root)
from scripts.openevo_webshop.ceiling1_stage2_format_hard_runtime import Qwen3Stage2FormatHardRuntime
rt=None
try:
    rt=Qwen3Stage2FormatHardRuntime(a.model,a.revision,sd_adapter=Path(a.adapter),
        memory=Path(a.memory).read_text(),skill_bundle=Path(a.skill).read_text(),agent_system=Path(a.agent).read_text())
    out={"schema":"openevo.advisor.q07.load-qualification.v1","status":"PASS",
         "adapter_sha256":sha(Path(a.adapter)/"adapter_model.safetensors"),
         "adapter_config_sha256":sha(Path(a.adapter)/"adapter_config.json"),
         "model_revision":a.revision,"inference_started":False,"generation_calls":0,
         "task_panel_accessed":False,"formal_rollouts_consumed":0,"final_panel_access":0,
         "wall_time_seconds":time.time()-started}
finally:
    if rt is not None: rt.close()
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,sort_keys=True))
