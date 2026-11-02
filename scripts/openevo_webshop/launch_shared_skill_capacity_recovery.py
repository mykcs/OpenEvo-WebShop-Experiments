#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p:Path): return json.loads(p.read_text(encoding='utf-8'))
def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=('qwen25-3b','qwen3-1p7b'),required=True)
    p.add_argument('--gpu-uuid',required=True); p.add_argument('--container',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-from',type=Path)
    a=p.parse_args(); repo=a.repo.resolve(); root=a.root.resolve(); output=a.output.resolve()
    contract=load(repo/'configs/experiment/stage1/202609030400/prestage2_shared.json'); spec=contract['arms'][a.arm]
    authority=repo/'configs/experiment/amendments/openevo-ceiling-1.0-stage1-skill-capacity-recovery-20260903-v1.json'
    auth=load(authority)
    if auth.get('status')!='FROZEN' or a.arm not in auth.get('applies_to_arms',[]): raise SystemExit('Skill capacity authority mismatch')
    head=subprocess.check_output(['git','-c',f'safe.directory={repo}','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-c',f'safe.directory={repo}','-C',str(repo),'status','--porcelain'],text=True).strip()
    if dirty: raise SystemExit('execution repo dirty')
    if subprocess.run(['docker','inspect',a.container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0: raise SystemExit('immutable container exists')
    mm=load(root/f'MINIMAX_COMPLETE_{a.arm}.json'); op=load(root/f'OPSD_COMPLETE_{a.arm}.json')
    if mm.get('status')!='PASS' or int(mm.get('parse_ok',-1))!=1440 or op.get('status')!='PASS': raise SystemExit('upstream not closed')
    subprocess.run(['install','-d','-o','1001','-g','1001','-m','700',str(output)],check=True)
    analysis=root/f'{a.arm}-minimax.jsonl'; adapter=Path(op['final_adapter_path']); model=Path(spec['model_path'])
    mounts=['-v',f'{repo}:/workspace/repo:ro','-v',f'{analysis}:/workspace/analysis.jsonl:ro','-v',f'{model}:/workspace/model:ro','-v',f'{adapter}:/workspace/adapter:ro','-v',f'{output}:/workspace/out']
    extra=[]
    if a.reuse_from:
        mounts += ['-v',f'{a.reuse_from.resolve()}:/workspace/reuse:ro']; extra=['--reuse-from','/workspace/reuse']
    common=['--analysis','/workspace/analysis.jsonl','--analysis-sha256',mm['analysis_corpus_sha256'],'--model','/workspace/model','--model-id',spec['model_id'],'--model-revision',spec['revision'],'--adapter','/workspace/adapter','--output','/workspace/out','--gpu-uuid',a.gpu_uuid,'--experiment-git-sha',head,'--config-sha256',sha(repo/'configs/experiment/stage1/202609030400/prestage2_shared.json'),'--authority-sha256',sha(authority),'--expected-context-window',str(spec['context_window']),'--chat-template-enable-thinking',('false' if spec['chat_template_enable_thinking'] is False else 'auto')]
    cmd=['docker','run','-d','--name',a.container,'--restart','no','--network','none','--user','1001:1001','--gpus',f'device={a.gpu_uuid}','-e',f'CUDA_VISIBLE_DEVICES={a.gpu_uuid}']+mounts+['-e','PYTHONPATH=/workspace/repo:/workspace/repo/vendor/openevo-webshop/src','--entrypoint','/usr/local/bin/python3','experiment-host/openevo-scientific:h144-runtime-v1','/workspace/repo/scripts/openevo_webshop/ceiling1_build_skill_bundle_capacity_v1.py']+common+extra
    print(' '.join(cmd),flush=True); subprocess.run(cmd,check=True)
    print(f'STARTED {a.container} arm={a.arm} authority_sha={sha(authority)}'); return 0
if __name__=='__main__': raise SystemExit(main())
