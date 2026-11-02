#!/usr/bin/env python3
"""Read the exact Q06 science mounts as UID1001, without GPU/model calls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from scripts.openevo_webshop.q06_context_probe_inputs import validate_inputs
from scripts.openevo_webshop.q06_context_host_orchestrator import build_probe_command, MANIFEST_SHA

PROGRAM = '''import json,os,pathlib,sys
inputs=json.loads(sys.argv[1]); output=pathlib.Path(sys.argv[2])
assert os.getuid()==1001 and os.getgid()==1001
for name in inputs:
 p=pathlib.Path(name)
 assert p.is_file() and not p.is_symlink(), name
 with p.open('rb') as f: f.read(1)
with (output/'uid1001-write-marker.json').open('x') as f:
 json.dump({'uid':os.getuid(),'gid':os.getgid(),'readable_files':len(inputs)},f)
 f.flush();os.fsync(f.fileno())
print(json.dumps({'status':'PASS_Q06_UID1001_NAMESPACE','readable_files':len(inputs),'model_calls':0,'environment_steps':0}))
'''


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--authority-repo',type=Path,required=True)
    p.add_argument('--execution-source',type=Path,required=True)
    p.add_argument('--qualification-root',type=Path,required=True)
    a=p.parse_args()
    repo,source=a.authority_repo.resolve(),a.execution_source.resolve()
    m=validate_inputs(repo,source,MANIFEST_SHA)
    root=a.qualification_root.absolute()
    if root.exists() or root.is_symlink() or not root.parent.is_dir() or root.parent.is_symlink():
        raise ValueError('Q06 namespace qualification requires a fresh prepared root')
    files=[source/rel for rel in m['runtime_files']]
    files += [repo/'docs/evidence/advisor-q06-e2-paired-analysis-20261001/Q06_GDR_CONTEXT_PROBE_MANIFEST.json',
              source/m['task_panel_manifest_path']]
    seed=Path(m['seed_root']['path'])
    files += [seed/rel for rel in m['seed_root']['package_files_sha256']]
    files += [Path(m['seed_root']['data_dir'])/rel for rel in m['seed_root']['webshop_data_files_sha256']]
    model=Path(m['model']['local_model_path'])
    files += [model/rel for rel in ('config.json','model.safetensors','verification-receipt.json')]
    adapter=Path(m['endpoint']['adapter_path'])
    files += [adapter/rel for rel in ('adapter_model.safetensors','adapter_config.json')]
    files += [Path(m['common_carriers'][key+'_path']) for key in ('memory','skill','agent_system')]
    command=build_probe_command(m,repo,source,root,'GPU-00000000-0000-0000-0000-000000000000')
    at=command.index('--gpus'); del command[at:at+2]
    command=[('CUDA_VISIBLE_DEVICES=' if x.startswith('CUDA_VISIBLE_DEVICES=') else x) for x in command]
    image_at=command.index('--entrypoint')+2
    command=command[:image_at+1]+['-c',PROGRAM,json.dumps([str(x) for x in files]),str(root)]
    root.mkdir(mode=0o700); os.chown(root,1001,1001)
    completed=subprocess.run(command,check=False,capture_output=True,text=True)
    receipt={'schema':'openevo.q06.namespace-qualification.202610011303',
             'status':'PASS' if completed.returncode==0 else 'FAIL',
             'uid':1001,'gid':1001,'gpu_used':False,'model_calls':0,'environment_steps':0,
             'launch_authority_granted':False,'manifest_sha256':MANIFEST_SHA,
             'input_files':[str(x) for x in files],'command':command,
             'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr}
    with (root/'namespace-qualification.json').open('x') as f:
        json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':receipt['status'],'receipt':str(root/'namespace-qualification.json'),
                      'model_calls':0,'environment_steps':0,'gpu_used':False}))
    return completed.returncode

if __name__=='__main__':
    raise SystemExit(main())
