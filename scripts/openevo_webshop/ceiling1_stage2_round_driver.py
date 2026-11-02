#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from collections import Counter
from pathlib import Path
from typing import Any
FROZEN_STAGE2_SAMPLING={'temperature':1.0,'do_sample':True,'top_p':1.0,'top_k':0,'repetition_penalty':1.0,'max_steps':15,'max_new_tokens':512,'generation_seed_base':148000000}

def atomic_json(p:Path,x:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'); t.replace(p)
def write_once_json(p:Path,x:Any)->None:
    rendered=json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'; p.parent.mkdir(parents=True,exist_ok=True)
    try:
        with p.open('x') as stream: stream.write(rendered)
    except FileExistsError:
        if p.read_text()!=rendered: raise RuntimeError('round shard layout drift; refusing partial resume')
def read_jsonl(p:Path):
    with p.open(encoding='utf-8') as stream:
        return [json.loads(x) for x in stream if x.strip()]
def task_identity(r:dict)->str: return f"{int(r['worker_seed'])}:{int(r['source_session_index'])}:{r['instruction_sha256']}"
def split_range(lo:int,hi:int,n:int):
    total=hi-lo+1; q,r=divmod(total,n); out=[]; cur=lo
    for i in range(n):
        k=q+(1 if i<r else 0); out.append((cur,cur+k-1)); cur+=k
    assert cur==hi+1; return out
def validate_shard_summaries(summaries:list[dict],gpus:list[str],round_index:int,state_id:str,ranges:list[tuple[int,int]],harness_profile:str,step_seed_policy:str)->None:
    if len(summaries)!=len(gpus) or len(summaries)!=len(ranges): raise RuntimeError('rollout shard summary count mismatch')
    for summary,gpu,(lo,hi) in zip(summaries,gpus,ranges,strict=True):
        if summary.get('status')!='pass' or int(summary.get('round_index',-1))!=round_index: raise RuntimeError('rollout shard status/round mismatch')
        if summary.get('state_id')!=state_id or summary.get('gpu_uuid')!=gpu: raise RuntimeError('rollout shard state/GPU mismatch')
        if int(summary.get('start_position',-1))!=lo or int(summary.get('end_position',-1))!=hi or int(summary.get('attempt_count',-1))!=hi-lo+1: raise RuntimeError('rollout shard position denominator mismatch')
        if int(summary.get('engineering_invalid_count',-1))!=0: raise RuntimeError('rollout shard contains engineering-invalid episodes')
        if summary.get('sampling')!=FROZEN_STAGE2_SAMPLING: raise RuntimeError('rollout shard sampling semantics drift')
        if summary.get('harness_profile')!=harness_profile or summary.get('step_seed_policy')!=step_seed_policy: raise RuntimeError('rollout shard Harness/step-seed readback drift')

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--schedule',type=Path,required=True); p.add_argument('--start-position',type=int,required=True); p.add_argument('--end-position',type=int,required=True); p.add_argument('--seed-root',type=Path,required=True)
    p.add_argument('--model',required=True); p.add_argument('--model-revision',required=True); p.add_argument('--sd-adapter',type=Path); p.add_argument('--memory',type=Path,required=True); p.add_argument('--skill',type=Path,required=True); p.add_argument('--agent-system',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--gpu-uuid',dest='gpus',action='append',required=True); p.add_argument('--round-index',type=int,required=True); p.add_argument('--state-id',required=True); p.add_argument('--experiment-git-sha',required=True); p.add_argument('--config-sha256',required=True); p.add_argument('--harness-profile',choices=('legacy','2.0.1','201.1'),default='legacy'); p.add_argument('--step-seed-policy',choices=('legacy','injective-v1'),default='legacy'); p.add_argument('--qualification-only',action='store_true'); p.add_argument('--resume',action='store_true'); p.add_argument('--worker-script',type=Path); a=p.parse_args()
    if (a.harness_profile in {'2.0.1','201.1'}) != (a.step_seed_policy=='injective-v1'): raise SystemExit('Harness successor profiles require injective-v1 as one bundle')
    if a.end_position-a.start_position+1!=128: raise SystemExit('round must be exactly 128 positions')
    if not a.gpus or len(set(a.gpus))!=len(a.gpus): raise SystemExit('GPU UUID list invalid')
    a.output.mkdir(parents=True,exist_ok=True); started=time.time(); ranges=split_range(a.start_position,a.end_position,len(a.gpus)); procs=[]
    existing_shards=sorted(path for path in a.output.glob('shard-*') if path.is_dir())
    if existing_shards and len(existing_shards)!=len(ranges): raise RuntimeError('existing shard count differs from retry layout; refusing duplicate rollout consumption')
    layout={'schema':'openevo.ceiling1-stage2-round-shard-layout.v1','round_index':a.round_index,'state_id':a.state_id,'experiment_git_sha':a.experiment_git_sha,'config_sha256':a.config_sha256,'harness_profile':a.harness_profile,'step_seed_policy':a.step_seed_policy,'qualification_only':bool(a.qualification_only),'worker_script':str(a.worker_script.resolve()) if a.worker_script else 'shared-default','ranges':[{'shard_index':i,'start_position':lo,'end_position':hi} for i,(lo,hi) in enumerate(ranges)]}
    write_once_json(a.output/'shard-layout.json',layout)
    for i,(gpu,(lo,hi)) in enumerate(zip(a.gpus,ranges,strict=True)):
        shard=a.output/f'shard-{i:02d}'; worker=a.worker_script.resolve() if a.worker_script else Path(__file__).resolve().with_name('ceiling1_stage2_rollout_worker.py')
        if not worker.is_file(): raise RuntimeError(f'rollout worker script missing: {worker}')
        cmd=[sys.executable,str(worker),'--schedule',str(a.schedule),'--start-position',str(lo),'--end-position',str(hi),'--seed-root',str(a.seed_root),'--model',a.model,'--model-revision',a.model_revision,'--memory',str(a.memory),'--skill',str(a.skill),'--agent-system',str(a.agent_system),'--output',str(shard),'--gpu-uuid',gpu,'--round-index',str(a.round_index),'--state-id',a.state_id,'--experiment-git-sha',a.experiment_git_sha,'--config-sha256',a.config_sha256,'--harness-profile',a.harness_profile,'--step-seed-policy',a.step_seed_policy]
        if a.sd_adapter: cmd+=['--sd-adapter',str(a.sd_adapter)]
        if a.qualification_only: cmd+=['--qualification-only']
        if a.resume: cmd+=['--resume']
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu}; procs.append((subprocess.Popen(cmd,env=env),cmd,shard))
    failures=[]
    for proc,cmd,shard in procs:
        rc=proc.wait()
        if rc: failures.append({'rc':rc,'shard':str(shard)})
    if failures: atomic_json(a.output/'driver-failure.json',{'schema':'openevo.ceiling1-stage2-round-driver-failure.v1','round_index':a.round_index,'failures':failures,'state_id':a.state_id}); raise RuntimeError(f'rollout worker failures: {failures}')
    rows=[]; shard_summaries=[]
    for _,_,shard in procs:
        summary=json.loads((shard/'summary.json').read_text()); shard_summaries.append(summary); rows.extend(read_jsonl(shard/'attempts.jsonl'))
    validate_shard_summaries(shard_summaries,a.gpus,a.round_index,a.state_id,ranges,a.harness_profile,a.step_seed_policy)
    rows.sort(key=lambda r:int(r['schedule_position'])); expected=list(range(a.start_position,a.end_position+1))
    if [int(r['schedule_position']) for r in rows]!=expected: raise RuntimeError('merged round positions are not exact')
    if any(r.get('harness_profile')!=a.harness_profile or r.get('step_seed_policy')!=a.step_seed_policy for r in rows): raise RuntimeError('merged rollout Harness/step-seed identity drift')
    invalid=sum(int(r['engineering_invalid']) for r in rows)
    if invalid: raise RuntimeError(f'round has {invalid} engineering-invalid episodes; rerun with --resume before advancement')
    (a.output/'attempts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
    clean=lambda r: r['final_task_score']==1.0 and r['won'] and r['identity_ok'] and r['trace_complete'] and not r['parser_or_fallback_failure'] and not r['parser_repair_used'] and not r['engineering_invalid'] and not r['error']
    identities=Counter(task_identity(r) for r in rows)
    if len(identities)!=16 or set(identities.values())!={8}: raise RuntimeError('round summary requires 16 task identities x 8 rollouts')
    summary={'schema':'openevo.ceiling1-stage2-round-rollout.v1','status':'PASS','round_index':a.round_index,'start_position':a.start_position,'end_position':a.end_position,'attempt_count':len(rows),'task_identity_count':len(identities),'clean_exact_success_count':sum(int(clean(r)) for r in rows),'clean_exact_success_task_count':len({task_identity(r) for r in rows if clean(r)}),'mean_score':sum(float(r['final_task_score']) for r in rows)/len(rows),'engineering_invalid_count':0,'state_id':a.state_id,'gpu_uuids':a.gpus,'harness_profile':a.harness_profile,'step_seed_policy':a.step_seed_policy,'qualification_only':bool(a.qualification_only),'sampling':FROZEN_STAGE2_SAMPLING,'shards':shard_summaries,'wall_time_seconds':time.time()-started}
    atomic_json(a.output/'summary.json',summary); print(json.dumps(summary,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
