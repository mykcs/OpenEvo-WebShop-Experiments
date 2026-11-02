#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.openevo_webshop.ceiling1_stage2_arm_controller import load, sha, validate_reflector_result

def write_once_json(p:Path,x:dict):
    text=json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n'; p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_text()!=text: raise RuntimeError(f'write-once conflict: {p}')
        return
    with p.open('x') as f: f.write(text); f.flush(); os.fsync(f.fileno())

def write_once_bytes(p:Path,data:bytes):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_bytes()!=data: raise RuntimeError(f'write-once payload conflict: {p}')
        return
    with p.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())

def wait_container(name:str,timeout:int=900)->int:
    end=time.time()+timeout
    while time.time()<end:
        q=subprocess.run(['docker','inspect','-f','{{.State.Running}} {{.State.ExitCode}} {{.State.OOMKilled}}',name],text=True,capture_output=True)
        if q.returncode==0:
            parts=q.stdout.strip().split()
            if parts and parts[0]=='false':
                if len(parts)>2 and parts[2].lower()=='true': raise RuntimeError(f'{name} OOMKilled')
                return int(parts[1])
        time.sleep(1)
    raise RuntimeError(f'{name} timeout')

def gpu_row(index:int)->dict:
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in out.splitlines():
        a=[x.strip() for x in line.split(',')]
        if int(a[0])==index: return {'index':index,'uuid':a[1],'used_mib':int(a[2]),'free_mib':int(a[3]),'util_pct':int(a[4])}
    raise RuntimeError(f'GPU{index} missing')

def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--analysis-root',type=Path,required=True);p.add_argument('--amendment',type=Path,required=True);p.add_argument('--arm',choices=('3b','qwen3-1p7b'),required=True);p.add_argument('--gpu-index',type=int,required=True);a=p.parse_args()
    repo=a.repo.resolve();root=a.analysis_root.resolve();amend=load(a.amendment.resolve()); ar=(amend.get('arms') or {}).get(a.arm) or {}
    if amend.get('schema')!='openevo.ceiling1-stage2-reflector-contract-binding-recovery.v1' or amend.get('status')!='AUTHORIZED': raise RuntimeError('recovery amendment invalid')
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    if head!=amend.get('successor_execution_git_sha') or dirty: raise RuntimeError('recovery repo is not exact clean successor')
    r=int(ar['partial_round']);kind=ar['kind'];carrier={'text_memory':'text-memory','skill_bundle':'skill-bundle'}[kind];name={'text_memory':'memory.md','skill_bundle':'SKILL.md'}[kind]
    out=root/'stage2'/a.arm/'formal-vnext-v1';rr=out/f'round-{r:03d}';d=rr/carrier; statep=out/'execution-state.json';state=load(statep)
    if state.get('status')!='BLOCKED' or int(state.get('next_round',-1))!=r or sha(statep)!=ar.get('blocked_execution_state_sha256'): raise RuntimeError('blocked state drift')
    if 'reflector failed' not in str(state.get('blocker','')): raise RuntimeError('unexpected blocker')
    frozen={'state_before':rr/'state-before.json','attempts':rr/'attempts.jsonl','summary':rr/'summary.json','evidence':rr/'evidence/evidence.json','model_call_attempt':d/'model-call-attempt.json','rejected_candidates':d/'rejected-candidates.json','rejected_primary':d/'rejected-primary.md','rejected_repair':d/'rejected-repair.md','wrong_contract_runtime':d/'WRONG_ARTIFACT_CONTRACT_RUNTIME.json'}
    for k,path in frozen.items():
        if not path.is_file() or sha(path)!=(ar.get('frozen_sha256') or {}).get(k): raise RuntimeError(f'frozen recovery input drift: {k}')
    wrong=load(frozen['wrong_contract_runtime'])
    if wrong.get('observed_artifact_contract')!='legacy-authoring-v1' or wrong.get('intended_artifact_contract')!=ar['intended_artifact_contract'] or wrong.get('contract_mismatch') is not True: raise RuntimeError('wrong-contract evidence drift')
    prior=Path(ar['prior_path']);
    if not prior.is_file() or sha(prior)!=ar['prior_sha256']: raise RuntimeError('prior artifact drift')
    sd=Path(state['current_sd_adapter']) if state.get('current_sd_adapter') else None
    if (sha(sd/'adapter_model.safetensors') if sd else None)!=ar.get('sd_adapter_sha256'): raise RuntimeError('prior SD state drift')
    canonical=d/name; canonical_receipt=canonical.with_suffix(canonical.suffix+'.receipt.json');terminal=d/'terminal-receipt.json'
    if canonical.exists() or canonical_receipt.exists() or terminal.exists(): raise RuntimeError('canonical carrier recovery output already exists')
    if a.gpu_index not in {4,5,6,7}: raise RuntimeError('recovery GPU outside managed pool')
    g=gpu_row(a.gpu_index)
    if g['used_mib']>16 or g['util_pct']!=0 or g['free_mib']<20000: raise RuntimeError(f'recovery GPU not strict idle/headroom safe: {g}')
    stage=d/'contract-recovery-v1'; stage.mkdir(parents=True,exist_ok=True); staged=stage/name
    if staged.exists() or staged.with_suffix(staged.suffix+'.receipt.json').exists(): raise RuntimeError('staged recovery output already exists')
    container=f'openevo-contract-recovery-{a.arm}-r{r:03d}-{kind.replace("_","-")}'
    if subprocess.run(['docker','inspect',container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0: raise RuntimeError('recovery container name already exists')
    uuid=g['uuid']; env={**os.environ,'CEILING1_STAGE2_CONTAINER':container,'CEILING1_ANALYSIS_ROOT':str(root),'CEILING1_AUTOPILOT_REPO':str(repo),'CEILING1_STAGE2_ARM':a.arm,'CEILING1_STAGE2_GPU_UUID':uuid,'CEILING1_STAGE2_KIND':kind,'CEILING1_STAGE2_EVIDENCE':str(rr/'evidence/evidence.json'),'CEILING1_STAGE2_PRIOR':str(prior),'CEILING1_STAGE2_OUTPUT':str(staged),'CEILING1_STAGE2_STATE_ID':state['state_id'],'CEILING1_STAGE2_ROUND':str(r),'CEILING1_STAGE2_SD_ADAPTER':str(sd) if sd else '','CEILING1_STAGE2_MEMORY':'','CEILING1_STAGE2_SKILL':'','CEILING1_STAGE2_ARTIFACT_CONTRACT':ar['intended_artifact_contract']}
    launcher=repo/'scripts/openevo_webshop'/('launch_ceiling1_stage2_reflector_qwen3_container.sh' if a.arm=='qwen3-1p7b' else 'launch_ceiling1_stage2_reflector_container.sh')
    subprocess.run([str(launcher)],env=env,check=True); rc=wait_container(container)
    if rc!=0:
        fail={'schema':'openevo.ceiling1-stage2-reflector-contract-recovery-result.v1','status':'FAILED','arm':a.arm,'round_index':r,'kind':kind,'container':container,'exit_code':rc,'intended_artifact_contract':ar['intended_artifact_contract'],'recovery_execution_git_sha':head,'formal_rollout_replay':0,'external_teacher_calls':0,'failed_at_epoch':time.time()}
        write_once_json(d/'REFLECTOR_CONTRACT_RECOVERY_FAILED.json',fail); raise RuntimeError(f'correct-contract recovery reflector exit={rc}')
    stage_receipt=staged.with_suffix(staged.suffix+'.receipt.json')
    if not staged.is_file() or not stage_receipt.is_file(): raise RuntimeError('recovery container exited without accepted artifact')
    rec=load(stage_receipt)
    if rec.get('artifact_contract')!=ar['intended_artifact_contract']: raise RuntimeError('accepted recovery contract drift')
    validate_reflector_result(rec,kind,r,state['state_id'],rr/'evidence/evidence.json',prior,None,None,sd,staged)
    write_once_bytes(canonical,staged.read_bytes());write_once_bytes(canonical_receipt,stage_receipt.read_bytes())
    changed=sha(canonical)!=sha(prior);term={'schema':'openevo.ceiling1-stage2-carrier-terminal.v1','status':'UPDATE' if changed else 'NOOP','kind':kind,'round_index':r,'reason':None if changed else 'NOOP_NO_CONTENT_CHANGE','output_path':str(canonical) if changed else str(prior),'output_sha256':sha(canonical) if changed else sha(prior),'reflector_receipt':rec,'recovered_from_wrong_artifact_contract':True,'wrong_contract_runtime_sha256':sha(frozen['wrong_contract_runtime']),'recovery_amendment_sha256':sha(a.amendment.resolve()),'recovery_execution_git_sha':head}
    write_once_json(terminal,term)
    ready={'schema':'openevo.ceiling1-stage2-reflector-contract-recovery-ready.v1','status':'PASS','arm':a.arm,'round_index':r,'kind':kind,'intended_artifact_contract':ar['intended_artifact_contract'],'wrong_artifact_contract':'legacy-authoring-v1','wrong_contract_model_calls_observed':2,'wrong_contract_calls_accepted_into_science_state':0,'recovery_primary_calls':int(rec.get('primary_calls',0)),'recovery_repair_calls':int(rec.get('repair_calls',0)),'recovery_model_calls':int(rec.get('primary_calls',0))+int(rec.get('repair_calls',0)),'canonical_terminal_sha256':sha(terminal),'canonical_output_sha256':sha(canonical),'canonical_reflector_receipt_sha256':sha(canonical_receipt),'blocked_execution_state_sha256':sha(statep),'execution_state_mutated':False,'formal_rollout_replay':0,'external_teacher_calls':0,'final_panel_access_delta':0,'scientific_contract_changed':False,'recovery_execution_git_sha':head,'recovery_amendment_sha256':sha(a.amendment.resolve()),'recovered_at_epoch':time.time()}
    write_once_json(d/'REFLECTOR_CONTRACT_RECOVERY_READY.json',ready); print(json.dumps(ready,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
