#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))

from scripts.openevo_webshop.ceiling1_stage2_reflector import bounded_projection, candidate_validation
from scripts.openevo_webshop.carrier_scope_package_v1 import load_package as load_carrier_scope_package, manifest_path_for_round
from scripts.openevo_webshop.build_carrier_health_v2 import build_round_ledger, write_json as write_carrier_health_json
from scripts.openevo_webshop.carrier_health_failure_policy_v1 import formal_round_decision, load_policy as load_carrier_health_policy, policy_identity as carrier_health_policy_identity, validate_resolution_receipt
from scripts.openevo_webshop.carrier_contract_v2_identity import load_identity as load_carrier_contract_identity
from scripts.openevo_webshop.carrier_contract_v2_adoption import validate_receipt as validate_carrier_contract_adoption
from scripts.openevo_webshop.ceiling1_harness201_g8 import ALERT_5, blocker_is_nonrecoverable, capacity_decision, consecutive_noop_diagnostic, load_capacity_policy, load_step_seed_amendment, load_harness201_carrier_seed, validate_staged_interlock_round, write_once_release, sha256 as g8_sha256
LEASE_ROOT_V2=Path('/workspace/control/openevo-ceiling1-gpu-lease-state-v2')
LEASE_ROOT_V3=Path('/workspace/control/openevo-ceiling1-gpu-lease-state-v3-eight-gpu')
LEASE_ROOT_V4=Path('/workspace/control/openevo-ceiling1-gpu-lease-state-v4-four-gpu')
LEASE_ROOT_V5=Path('/workspace/control/openevo-ceiling1-gpu-lease-state-v5-four-gpu')
LEASE_ROOT_Q17=Path('/workspace/control/openevo-ceiling1-gpu-lease-state-q17-stage2-v1')
UUID={0:'GPU-00000000-0000-0000-0000-000000000000',1:'GPU-00000000-0000-0000-0000-000000000000',2:'GPU-00000000-0000-0000-0000-000000000000',3:'GPU-00000000-0000-0000-0000-000000000000',4:'GPU-00000000-0000-0000-0000-000000000000',5:'GPU-00000000-0000-0000-0000-000000000000',6:'GPU-00000000-0000-0000-0000-000000000000',7:'GPU-00000000-0000-0000-0000-000000000000'}

def sha(p:Path|None):
    if p is None:return None
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def csha(x:Any): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'); t.replace(p)
def stable_json(x:Any): return json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
def write_once_json(p:Path,x:Any):
    rendered=stable_json(x); p.parent.mkdir(parents=True,exist_ok=True)
    try:
      with p.open('x') as f: f.write(rendered)
    except FileExistsError:
      if p.read_text()!=rendered: raise RuntimeError(f'append-only reflector attempt identity mismatch: {p}')
ROUND_STATE_CONTROL_ONLY_KEYS={'status','blocker','blocked_at_epoch','controller_recoveries','carrier_health_pause_path','carrier_health_pause_sha256','carrier_health_paused_at_epoch'}
def round_science_state(state:dict)->dict:
    return {k:v for k,v in state.items() if k not in ROUND_STATE_CONTROL_ONLY_KEYS}
def ensure_round_state_before(path:Path,state:dict)->None:
    rendered=stable_json(state); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
      prior=load(path)
      if round_science_state(prior)!=round_science_state(state): raise RuntimeError('partial round scientific state-before identity mismatch; refusing recovery rewrite')
      return
    try:
      with path.open('x') as f: f.write(rendered)
    except FileExistsError:
      prior=load(path)
      if round_science_state(prior)!=round_science_state(state): raise RuntimeError('partial round scientific state-before identity mismatch after concurrent create')
def load(p:Path): return json.loads(p.read_text())
def bind_carrier_scope_package_identity(state:dict,package_identity:str|None)->dict:
    observed=state.get('carrier_scope_package_identity_sha256')
    if package_identity is None:
      if observed is not None: raise RuntimeError('Carrier scope package omitted on resume')
      return state
    if observed is None:
      formal_consumed=int(state.get('rollouts_consumed',0))>0 or int(state.get('next_round',0))>0 or bool(state.get('rounds_complete'))
      if formal_consumed: raise RuntimeError('Carrier scope package cannot be adopted after formal consumption')
      state['carrier_scope_package_identity_sha256']=package_identity
    elif observed!=package_identity:
      raise RuntimeError('Carrier scope package identity drift on resume')
    return state
def bind_preformal_identity(state:dict,key:str,identity:str|None,label:str)->dict:
    observed=state.get(key)
    if identity is None:
      if observed is not None: raise RuntimeError(f'{label} omitted on resume')
      return state
    if observed is None:
      formal_consumed=int(state.get('rollouts_consumed',0))>0 or int(state.get('next_round',0))>0 or bool(state.get('rounds_complete'))
      if formal_consumed: raise RuntimeError(f'{label} cannot be adopted after formal consumption')
      state[key]=identity
    elif observed!=identity:
      raise RuntimeError(f'{label} identity drift on resume')
    return state
def cmd(xs,*,env=None,check=True,capture=False): return subprocess.run([str(x) for x in xs],env=env,text=True,capture_output=capture,check=check)
def assert_execution_source(repo:Path,expected_sha:str)->None:
    head=cmd(['git','-C',repo,'rev-parse','HEAD'],capture=True).stdout.strip()
    if head!=expected_sha: raise RuntimeError(f'execution Git SHA drift: expected {expected_sha}, observed {head}')
    dirty=cmd(['git','-C',repo,'status','--porcelain'],capture=True).stdout.strip()
    if dirty: raise RuntimeError('execution worktree is dirty; refusing mutable runtime source')
def dstatus(name:str):
    r=cmd(['docker','inspect',name,'--format','{{.State.Status}}'],check=False,capture=True); return r.stdout.strip() if r.returncode==0 else 'absent'
def wait_container(name:str)->int:
    r=cmd(['docker','wait',name],check=False,capture=True)
    try:return int(r.stdout.strip())
    except:return 255

def sd_noop_diagnostic(output_root:Path,through_round:int)->dict:
    start=max(0,through_round-9); statuses=[]
    for index in range(start,through_round+1):
      path=Path(output_root)/f'round-{index:03d}'/'sd-lora'/'terminal-receipt.json'
      if path.is_file(): statuses.append({'round_index':index,'status':load(path).get('status')})
    noops=sum(item['status']=='NOOP' for item in statuses)
    return {'schema':'openevo.ceiling1-sd-lora-noop-window-diagnostic.v1','status':'ALERT' if noops>=5 else 'OBSERVE',
            'through_round':through_round,'window_start_round':start,'window_size':len(statuses),'noop_count':noops,
            'diagnostics':['clean_success_tasks','raw_score_trend','parser_failures','invalid_action_terminations','carrier_updates','selector_positive_trace_accounting'],
            'modify_training':False,'modify_thresholds':False,'pause_or_retry':False,'automatic_intervention':False}

def validate_v3_switch_boundary(output_root:Path)->None:
    existing=Path(output_root)/'execution-state.json'
    if not existing.is_file(): return
    next_round=int(load(existing).get('next_round',-1)); partial=Path(output_root)/f'round-{next_round:03d}'
    if partial.exists() and not (partial/'ROUND_COMPLETE.json').is_file(): raise RuntimeError('7B v3 resource switch requires a natural new round boundary; partial round is untouched')

def state_id(base_receipt:Path,sd:Path|None,memory:Path,skill:Path,agent:Path,after_round:int)->str:
    return csha({'base_receipt':sha(base_receipt),'sd_adapter':sha(sd/'adapter_model.safetensors') if sd else None,'memory':sha(memory),'skill':sha(skill),'agent_system':sha(agent),'after_round':after_round})
def validate_rollout_summary(summary:dict,round_index:int,expected_state_id:str)->None:
    if summary.get('schema')!='openevo.ceiling1-stage2-round-rollout.v1' or summary.get('status')!='PASS': raise RuntimeError('round rollout summary is not PASS')
    if int(summary.get('round_index',-1))!=round_index: raise RuntimeError('round rollout summary round mismatch')
    if summary.get('state_id')!=expected_state_id: raise RuntimeError('round rollout summary state_id mismatch')
    if int(summary.get('attempt_count',-1))!=128 or int(summary.get('engineering_invalid_count',-1))!=0: raise RuntimeError('round rollout seal failed')
def validate_evidence_result(evidence:dict,rr:Path,round_index:int,prior_history:Path|None)->None:
    if evidence.get('schema')!='openevo.ceiling1-stage2-round-evidence.v1' or evidence.get('status')!='PASS': raise RuntimeError('round evidence is not PASS')
    if int(evidence.get('round_index',-1))!=round_index: raise RuntimeError('round evidence index mismatch')
    if int(evidence.get('attempt_count',-1))!=128 or int(evidence.get('task_identity_count',-1))!=16: raise RuntimeError('round evidence denominator mismatch')
    attempts=rr/'attempts.jsonl'
    if evidence.get('attempts_sha256')!=sha(attempts): raise RuntimeError('round evidence attempts SHA mismatch')
    expected_prior_sha=sha(prior_history) if prior_history else None
    if evidence.get('prior_history_sha256')!=expected_prior_sha: raise RuntimeError('round evidence prior history SHA mismatch')
    history_path=rr/'evidence'/'history-after.json'
    if not history_path.is_file() or evidence.get('history_after_sha256')!=sha(history_path): raise RuntimeError('round evidence successor history SHA mismatch')
    history=load(history_path)
    if history.get('schema')!='openevo.ceiling1-stage2-evidence-history.v1' or int(history.get('through_round',-1))!=round_index: raise RuntimeError('round evidence successor history identity mismatch')
    if int(evidence.get('history_before_round',-2))!=round_index-1 or int(evidence.get('history_after_round',-1))!=round_index: raise RuntimeError('round evidence history accounting mismatch')
    gates=evidence.get('gates')
    expected_gates={'text_memory','parametric_sd_lora','skill_bundle','agent_system'}
    if not isinstance(gates,dict) or set(gates)!=expected_gates or any(not isinstance(gates[k].get('eligible'),bool) for k in expected_gates): raise RuntimeError('round evidence gate contract mismatch')
def harness_runtime_step_readback_ok(step:dict,expected_runtime:str)->bool:
    if step.get('runtime_harness')==expected_runtime:
        return True
    if expected_runtime!='201.1':
        return False
    sentinel_keys={
        'step','base_prompt','generation_seed','step_seed_policy',
        'constraint_escape','parser_repair_used','error',
    }
    error=step.get('error')
    constraint_error=isinstance(error,str) and error in {
        'RuntimeError: Harness201.1 action prefix escaped all current action schemas',
        'RuntimeError: current environment exposes no compilable action schema',
    }
    return bool(
        set(step)==sentinel_keys
        and isinstance(step.get('step'),int) and not isinstance(step.get('step'),bool) and step['step']>=0
        and isinstance(step.get('base_prompt'),str)
        and isinstance(step.get('generation_seed'),int) and not isinstance(step.get('generation_seed'),bool)
        and constraint_error
        and step.get('constraint_escape') is True
        and step.get('parser_repair_used') is False
        and step.get('step_seed_policy')=='injective-v1'  # legacy-revision-compat
    )

def validate_attempt_rows(rr:Path,round_index:int,expected_state_id:str,allowed_execution_shas:set[str]|None=None)->set[str]:
    attempts=rr/'attempts.jsonl'
    try: rows=[json.loads(line) for line in attempts.read_text().split('\n') if line.strip()]
    except (OSError,json.JSONDecodeError) as exc: raise RuntimeError(f'round {round_index} attempt rows unreadable') from exc
    if len(rows)!=128: raise RuntimeError(f'round {round_index} attempt rows denominator mismatch')
    expected_positions=list(range(1441+round_index*128,1441+(round_index+1)*128))
    if [int(row.get('schedule_position',-1)) for row in rows]!=expected_positions: raise RuntimeError(f'round {round_index} attempt rows position mismatch')
    if any(row.get('state_id')!=expected_state_id for row in rows): raise RuntimeError(f'round {round_index} attempt rows state mismatch')
    if any(bool(row.get('engineering_invalid')) for row in rows): raise RuntimeError(f'round {round_index} attempt rows contain engineering-invalid rollout')
    execution_shas={str(row.get('experiment_git_sha','')) for row in rows}
    if '' in execution_shas or len(execution_shas)!=1: raise RuntimeError(f'round {round_index} attempt rows have mixed or missing execution SHA')
    if allowed_execution_shas is not None and not execution_shas<=allowed_execution_shas: raise RuntimeError(f'round {round_index} attempt rows execution SHA is not locked')
    return execution_shas
def validate_round_transaction(rr:Path,round_index:int,expected_state_id:str,allowed_execution_shas:set[str]|None=None)->tuple[str,set[str]]:
    before=load(rr/'state-before.json'); after=load(rr/'state-after.json'); complete=load(rr/'ROUND_COMPLETE.json')
    if before.get('state_id')!=expected_state_id: raise RuntimeError(f'round {round_index} state-before chain mismatch')
    if complete.get('status')!='PASS' or int(complete.get('round_index',-1))!=round_index: raise RuntimeError(f'round {round_index} completion receipt mismatch')
    if complete.get('state_id_before')!=expected_state_id or complete.get('state_id_after')!=after.get('state_id'): raise RuntimeError(f'round {round_index} completion state chain mismatch')
    if int(after.get('next_round',-1))!=round_index+1 or int(after.get('rollouts_consumed',-1))!=(round_index+1)*128: raise RuntimeError(f'round {round_index} successor accounting mismatch')
    summary=complete.get('rollout_summary',{})
    if int(summary.get('attempt_count',-1))!=128 or int(summary.get('engineering_invalid_count',-1))!=0: raise RuntimeError(f'round {round_index} summary denominator mismatch')
    execution_shas=validate_attempt_rows(rr,round_index,expected_state_id,allowed_execution_shas)
    validate_rollout_summary(summary,round_index,expected_state_id)
    try: standalone=load(rr/'summary.json')
    except (OSError,json.JSONDecodeError) as exc: raise RuntimeError(f'round {round_index} standalone summary missing or unreadable') from exc
    if standalone!=summary: raise RuntimeError(f'round {round_index} standalone summary differs from completion receipt')
    return str(after['state_id']),execution_shas
def validate_round_state_payloads(rr:Path,prior_sd_sha:str|None)->str|None:
    after=load(rr/'state-after.json'); complete=load(rr/'ROUND_COMPLETE.json')
    carrier_fields=(('text_memory','memory'),('skill_bundle','skill'),('agent_system','agent_system'))
    for terminal_key,state_key in carrier_fields:
      terminal=complete[terminal_key]; path=Path(after[state_key]).resolve()
      if Path(terminal.get('output_path','')).resolve()!=path or not path.is_file() or sha(path)!=terminal.get('output_sha256'): raise RuntimeError(f'{rr.name} {terminal_key} payload identity mismatch')
    sd_terminal=complete['parametric_sd_lora']; sd_path=Path(after['current_sd_adapter']).resolve() if after.get('current_sd_adapter') else None
    if sd_terminal.get('adapter_path'):
      if sd_path is None or Path(sd_terminal['adapter_path']).resolve()!=sd_path: raise RuntimeError(f'{rr.name} SD-LoRA state path mismatch')
    sd_sha=prior_sd_sha
    if sd_terminal.get('status')=='UPDATE': sd_sha=sd_terminal.get('update_receipt',{}).get('adapter_digest_after')
    if sd_path is not None:
      model=sd_path/'adapter_model.safetensors'
      if model.is_file(): observed_sd_sha=sha(model)
      else:
        prune=sd_path/'payload-pruned.json'
        if not prune.is_file(): raise RuntimeError(f'{rr.name} SD-LoRA payload missing without prune receipt')
        matches=[x.get('sha256') for x in load(prune).get('files_removed',[]) if x.get('name')=='adapter_model.safetensors']
        if len(matches)!=1: raise RuntimeError(f'{rr.name} SD-LoRA prune receipt lacks adapter identity')
        observed_sd_sha=matches[0]
      if observed_sd_sha!=sd_sha: raise RuntimeError(f'{rr.name} SD-LoRA payload identity mismatch')
    elif sd_sha is not None: raise RuntimeError(f'{rr.name} lost prior SD-LoRA state')
    base=Path(after['merged_base_receipt']); recomputed=csha({'base_receipt':sha(base),'sd_adapter':sd_sha,'memory':sha(Path(after['memory'])),'skill':sha(Path(after['skill'])),'agent_system':sha(Path(after['agent_system'])),'after_round':int(complete['round_index'])})
    if recomputed!=after.get('state_id') or recomputed!=complete.get('state_id_after'): raise RuntimeError(f'{rr.name} recomputed successor state identity mismatch')
    return sd_sha
def reflector_attempt_identity(kind:str,round_index:int,state_id_value:str,evidence:Path,prior:Path,memory:Path|None,skill:Path|None,sd_adapter:Path|None,artifact_contract:str|None=None,scope_manifest:Path|None=None)->dict:
    out={'schema':'openevo.ceiling1-stage2-reflector-container-attempt.v1','kind':kind,'round_index':round_index,'state_id':state_id_value,'evidence_sha256':sha(evidence),'prior_sha256':sha(prior),'memory_sha256':sha(memory) if memory else None,'skill_sha256':sha(skill) if skill else None,'sd_adapter_sha256':sha(sd_adapter/'adapter_model.safetensors') if sd_adapter else None,'model_container_max_attempts':1}
    if artifact_contract is not None: out['artifact_contract']=artifact_contract
    if scope_manifest is not None: out['scope_manifest_sha256']=sha(scope_manifest)
    return out
def validate_reflector_result(receipt:dict,kind:str,round_index:int,state_id_value:str,evidence:Path,prior:Path,memory:Path|None,skill:Path|None,sd_adapter:Path|None,output:Path,scope_manifest:Path|None=None)->None:
    if receipt.get('schema')!='openevo.ceiling1-stage2-reflector.v2' or receipt.get('status')!='UPDATE': raise RuntimeError('reflector result receipt identity mismatch')
    if receipt.get('kind')!=kind or int(receipt.get('round_index',-1))!=round_index or receipt.get('state_id')!=state_id_value: raise RuntimeError('reflector result state identity mismatch')
    expected={'evidence_sha256':sha(evidence),'prior_sha256':sha(prior),'memory_sha256':sha(memory) if memory else None,'skill_sha256':sha(skill) if skill else None,'sd_adapter_sha256':sha(sd_adapter/'adapter_model.safetensors') if sd_adapter else None,'output_sha256':sha(output)}
    if scope_manifest is not None: expected['scope_manifest_sha256']=sha(scope_manifest)
    elif receipt.get('scope_manifest_sha256') not in (None,): raise RuntimeError('unexpected reflector scope manifest identity')
    if any(receipt.get(key)!=value for key,value in expected.items()): raise RuntimeError('reflector result content identity mismatch')
    scope_payload=None
    if scope_manifest is not None:
      from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import load_scope_manifest
      scope_payload=load_scope_manifest(scope_manifest,expected_round_index=round_index)
    validation=candidate_validation(kind,output.read_text(),receipt.get('artifact_contract') or 'legacy-authoring-v1',scope_manifest=scope_payload)
    if not validation['artifact_contract_valid']: raise RuntimeError('reflector output fails shared producer/consumer carrier contract')
    if receipt.get('primary_scope_status') not in (None,validation['scope_status']) and int(receipt.get('repair_calls',0))==0: raise RuntimeError('reflector scope status readback mismatch')
    if kind=='agent_system' and skill is not None and sha(output)==sha(skill): raise RuntimeError('agent_system output aliases skill_bundle')
    projection=bounded_projection(kind,load(evidence)); projection_bytes=json.dumps(projection,ensure_ascii=False,sort_keys=True).encode()
    if receipt.get('projection_schema')!=projection.get('schema') or receipt.get('projection_sha256')!=hashlib.sha256(projection_bytes).hexdigest() or int(receipt.get('projection_bytes',-1))!=len(projection_bytes): raise RuntimeError('reflector bounded projection identity mismatch')
    if int(receipt.get('primary_calls',-1))!=1 or int(receipt.get('repair_calls',-1)) not in (0,1) or int(receipt.get('max_primary_calls',-1))!=1 or int(receipt.get('max_repair_calls',-1))!=1: raise RuntimeError('reflector call budget mismatch')
    if int(receipt.get('accepted_output_tokens',1024))>=1024 or int(receipt.get('max_new_tokens_per_call',-1))!=1024: raise RuntimeError('reflector output cap contract mismatch')
    expected_context=40960 if receipt.get('chat_template_enable_thinking') is False else 32768
    if receipt.get('context_window')!=expected_context or receipt.get('hard_context_preflight') is not True or receipt.get('silent_truncation_allowed') is not False or receipt.get('do_sample') is not False or int(receipt.get('external_teacher_calls',-1))!=0: raise RuntimeError('reflector generation contract mismatch')

REFLECTOR_CONTRACT_FAIL_NOOP_REASON='NOOP_KEEP_PRIOR_TYPED_CONTRACT_REJECTED_AFTER_REPAIR'

def build_reflector_contract_fail_noop(kind:str,round_index:int,state_id_value:str,evidence:Path,prior:Path,memory:Path|None,skill:Path|None,sd_adapter:Path|None,directory:Path,attempt_path:Path,artifact_contract:str|None=None,scope_manifest:Path|None=None)->dict|None:
    rejected_path=directory/'rejected-candidates.json'
    if not rejected_path.is_file(): return None
    rejected=load(rejected_path)
    if rejected.get('schema')!='openevo.ceiling1-stage2-reflector-rejected-candidates.v1' or rejected.get('kind')!=kind or int(rejected.get('round_index',-1))!=round_index or rejected.get('state_id')!=state_id_value: raise RuntimeError('reflector rejected-candidate identity mismatch')
    attempts=list(rejected.get('attempts') or [])
    if [x.get('attempt') for x in attempts]!=['primary','repair']: return None
    if any(x.get('artifact_contract_valid') is not False for x in attempts): return None
    if not attempt_path.is_file(): raise RuntimeError('reflector contract-fail NOOP missing model-call attempt identity')
    expected_attempt=reflector_attempt_identity(kind,round_index,state_id_value,evidence,prior,memory,skill,sd_adapter,artifact_contract,scope_manifest)
    if load(attempt_path)!=expected_attempt: raise RuntimeError('reflector contract-fail NOOP model-call attempt identity drift')
    rejected_files=[]
    for item in attempts:
      label=item['attempt']; path=directory/f'rejected-{label}.md'
      if not path.is_file() or item.get('candidate_sha256')!=sha(path): raise RuntimeError('reflector rejected candidate payload identity mismatch')
      rejected_files.append({'attempt':label,'path':str(path),'sha256':sha(path),'output_tokens':int(item.get('output_tokens',-1)),'saturated':bool(item.get('saturated')),'artifact_contract_valid':False,'scope_status':item.get('scope_status')})
    primary_sha=rejected_files[0]['sha256']; repair_sha=rejected_files[1]['sha256']; repair_changed=primary_sha!=repair_sha
    terminal={'schema':'openevo.ceiling1-stage2-carrier-terminal.v1','status':'NOOP','kind':kind,'round_index':round_index,'reason':REFLECTOR_CONTRACT_FAIL_NOOP_REASON,'scope_manifest_sha256':sha(scope_manifest) if scope_manifest else None,'prior_sha256':sha(prior),'output_path':str(prior),'output_sha256':sha(prior),'primary_calls':1,'repair_calls':1,'external_teacher_calls':0,'model_calls_consumed':2,'artifact_contract':artifact_contract,'contract_fail_evidence_dir':str(directory.resolve()),'rejected_candidates_path':str(rejected_path),'rejected_candidates_sha256':sha(rejected_path),'model_call_attempt_path':str(attempt_path),'model_call_attempt_sha256':sha(attempt_path),'rejected_candidates':rejected_files,'scientific_semantics':'unsafe carrier candidate is not promoted; unchanged carrier remains State_k; other carrier channels continue'}
    if rejected.get('carrier_contract_health_version')=='v2':
      if rejected.get('primary_payload_sha256')!=primary_sha or rejected.get('repair_payload_sha256')!=repair_sha: raise RuntimeError('reflector v2 repair payload summary drift')
      if rejected.get('primary_contract_valid') is not False or rejected.get('repair_contract_valid') is not False: raise RuntimeError('reflector v2 repair validity summary drift')
      if rejected.get('repair_requested') is not True or rejected.get('repair_changed_bytes') is not repair_changed: raise RuntimeError('reflector v2 repair-change summary drift')
      if attempts[1].get('repair_changed_bytes') is not repair_changed: raise RuntimeError('reflector v2 repair-change attempt drift')
      terminal.update({'carrier_contract_health_version':'v2','generation_attempted':True,'primary_payload_sha256':primary_sha,'primary_contract_valid':False,'primary_scope_status':attempts[0].get('scope_status'),'repair_requested':True,'repair_payload_sha256':repair_sha,'repair_changed_bytes':repair_changed,'repair_contract_valid':False,'repair_scope_status':attempts[1].get('scope_status'),'accepted_payload_sha256':sha(prior)})
    return terminal

def validate_task_vector_update(geometry:dict,expected_component_count:int)->None:
    if geometry.get('status')!='PASS': raise RuntimeError('updated SD-LoRA task vector is not PASS')
    if int(geometry.get('component_count',-1))!=expected_component_count: raise RuntimeError('updated SD-LoRA component count mismatch')
    delta=float(geometry.get('delta_norm',float('nan')))
    if not math.isfinite(delta) or delta<=0.0: raise RuntimeError('updated SD-LoRA must have a finite non-zero parameter delta')

def validate_task_vector_identity(geometry:dict,expected_component_count:int,current_adapter:Path,round_index:int,generation_index:int,arm:str,prior_geometry:Path|None)->None:
    validate_task_vector_update(geometry,expected_component_count)
    if geometry.get('schema')!='openevo.ceiling1-task-vector-geometry.v1' or geometry.get('arm')!=arm or int(geometry.get('round_index',-1))!=round_index or int(geometry.get('generation_index',-1))!=generation_index: raise RuntimeError('task-vector geometry identity mismatch')
    if int(geometry.get('component_rank',-1))!=8 or int(geometry.get('effective_rank',-1))!=expected_component_count*8: raise RuntimeError('task-vector geometry rank mismatch')
    factors=geometry.get('component_factor_sha256',[]); coefficients=geometry.get('current_coefficients',[])
    if len(factors)!=expected_component_count or len(coefficients)!=expected_component_count: raise RuntimeError('task-vector geometry component closure mismatch')
    manifest=current_adapter/'openevo_sd_lora_state.json'; state=current_adapter/'openevo_sd_lora_state.safetensors'
    if not manifest.is_file() or not state.is_file() or geometry.get('current_adapter_manifest_sha256')!=sha(manifest) or geometry.get('current_adapter_state_sha256')!=sha(state): raise RuntimeError('task-vector adapter identity mismatch')
    expected_prior_sha=sha(prior_geometry) if prior_geometry else None
    if geometry.get('prior_geometry_sha256')!=expected_prior_sha: raise RuntimeError('task-vector prior geometry identity mismatch')

def validate_sd_training_update(receipt:dict,expected_component_count_before:int)->None:
    if receipt.get('status')!='UPDATE': raise RuntimeError('SD-LoRA training receipt is not UPDATE')
    steps=int(receipt.get('optimizer_steps') or 0)
    if steps<=0: raise RuntimeError('SD-LoRA UPDATE must contain positive optimizer steps')
    examples=int(receipt.get('optimizer_example_count') or 0)
    if examples<=0: raise RuntimeError('SD-LoRA UPDATE must consume optimizer examples')
    loss=float(receipt.get('training_loss',float('nan')))
    if not math.isfinite(loss): raise RuntimeError('SD-LoRA UPDATE must contain finite training loss')
    before=int(receipt.get('component_count_before',-1)); after=int(receipt.get('component_count_after',-1))
    if before!=expected_component_count_before or after!=before+1: raise RuntimeError('SD-LoRA UPDATE component count mismatch')
    if int(receipt.get('effective_rank_after',-1))!=after*8: raise RuntimeError('SD-LoRA UPDATE effective rank mismatch')
    digest=str(receipt.get('adapter_digest_after') or '')
    if len(digest)!=64 or any(char not in '0123456789abcdef' for char in digest): raise RuntimeError('SD-LoRA UPDATE adapter digest is invalid')

def validate_parametric_selection(selection:dict,dataset_root:Path,attempts:Path,round_index:int)->None:
    if selection.get('schema')!='openevo.ceiling1-stage2-parametric-selection.v2' or int(selection.get('round_index',-1))!=round_index or int(selection.get('max_records',-1))!=256: raise RuntimeError('parametric selection identity mismatch')
    if selection.get('attempts_sha256')!=sha(attempts): raise RuntimeError('parametric selection attempts SHA mismatch')
    status=selection.get('status')
    if status=='NOOP_NO_SUCCESS':
      if int(selection.get('selected_task_count',-1))!=0 or int(selection.get('trace_count',-1))!=0: raise RuntimeError('parametric selection NOOP accounting mismatch')
      return
    if status!='DATASET_READY': raise RuntimeError('parametric selection status mismatch')
    selected=int(selection.get('selected_task_count',-1)); traces=int(selection.get('trace_count',-1)); positions=selection.get('selected_schedule_positions',[])
    if not 1<=selected<=16 or not 2<=traces<=240 or len(positions)!=selected or len(set(map(int,positions)))!=selected: raise RuntimeError('parametric selection ready accounting mismatch')
    manifest=dataset_root/'manifest.json'; records=dataset_root/'records.jsonl'
    if Path(str(selection.get('manifest',''))).resolve()!=manifest.resolve() or Path(str(selection.get('records',''))).resolve()!=records.resolve(): raise RuntimeError('parametric selection dataset path mismatch')
    if not manifest.is_file() or not records.is_file() or selection.get('manifest_sha256')!=sha(manifest) or selection.get('records_sha256')!=sha(records): raise RuntimeError('parametric selection dataset payload mismatch')
    payload=load(manifest)
    if payload.get('schema')!='openevo.dataset.v1' or int(payload.get('record_count',-1))!=selected or int(payload.get('trace_count',-1))!=traces or int(payload.get('max_records',-1))!=256 or payload.get('records_sha256')!=sha(records): raise RuntimeError('parametric dataset manifest mismatch')

def validate_sd_terminal(terminal:dict,round_index:int,prior_sd:Path|None,prior_artifact_id:str|None,component_count_before:int,arm:str,config_sha:str,execution_git_sha:str|set[str]|None)->tuple[Path|None,str|None,int,bool]:
    if terminal.get('schema')!='openevo.ceiling1-stage2-carrier-terminal.v1' or terminal.get('kind')!='parametric_sd_lora' or int(terminal.get('round_index',-1))!=round_index: raise RuntimeError('SD-LoRA terminal receipt identity mismatch')
    status=terminal.get('status')
    if status=='UPDATE':
      update=terminal.get('update_receipt',{}); validate_sd_training_update(update,component_count_before)
      if update.get('schema')!='openevo.ceiling1-stage2-sd-lora.v1' or update.get('arm')!=arm or int(update.get('round_index',-1))!=round_index: raise RuntimeError('SD-LoRA update receipt identity mismatch')
      allowed_execution_shas=({execution_git_sha} if isinstance(execution_git_sha,str) else set(execution_git_sha or []))
      if update.get('config_sha256')!=config_sha or (allowed_execution_shas and update.get('experiment_git_sha') not in allowed_execution_shas): raise RuntimeError('SD-LoRA update source identity mismatch')
      expected_user={'rank':8,'learning_rate':0.0002,'epochs':1,'max_records':256,'replay_capacity':64,'max_length':2048,'dtype':'bfloat16','seed':1993}
      user=update.get('native_user_config',{})
      if any(user.get(key)!=value for key,value in expected_user.items()) or list(user.get('target_modules',[]))!=['q_proj','k_proj','v_proj','o_proj']: raise RuntimeError('SD-LoRA update training config mismatch')
      if int(update.get('privileged_external_teacher_calls',-1))!=0: raise RuntimeError('SD-LoRA update external-teacher contract mismatch')
      if update.get('prior_artifact_id')!=prior_artifact_id: raise RuntimeError('SD-LoRA update prior artifact mismatch')
      adapter=Path(str(update.get('adapter_path','')))
      expected_after=int(update['component_count_after'])
      if not update.get('artifact_id') or Path(str(terminal.get('adapter_path',''))).resolve()!=adapter.resolve() or terminal.get('artifact_id')!=update.get('artifact_id') or int(terminal.get('component_count_after',-1))!=expected_after: raise RuntimeError('SD-LoRA terminal update payload mismatch')
      model=adapter/'adapter_model.safetensors'
      if not model.is_file() or sha(model)!=update.get('adapter_digest_after'): raise RuntimeError('SD-LoRA updated adapter digest mismatch')
      return adapter,str(update['artifact_id']),expected_after,True
    if status!='NOOP': raise RuntimeError('SD-LoRA terminal status mismatch')
    expected_path=prior_sd.resolve() if prior_sd else None
    observed_path=Path(str(terminal['adapter_path'])).resolve() if terminal.get('adapter_path') else None
    if observed_path!=expected_path or terminal.get('artifact_id')!=prior_artifact_id or int(terminal.get('component_count_after',-1))!=component_count_before: raise RuntimeError('SD-LoRA terminal NOOP state mismatch')
    model=prior_sd/'adapter_model.safetensors' if prior_sd else None; expected_sha=sha(model) if model else None
    if terminal.get('prior_sha256')!=expected_sha or terminal.get('output_sha256')!=expected_sha: raise RuntimeError('SD-LoRA terminal NOOP payload mismatch')
    if not terminal.get('reason') or any(int(terminal.get(key,-1))!=0 for key in ('primary_calls','repair_calls','external_teacher_calls')): raise RuntimeError('SD-LoRA terminal NOOP contract mismatch')
    return prior_sd,prior_artifact_id,component_count_before,False

def validate_stage2_entry_authority(root:Path,arm:str,activation_path:Path,protocol_path:Path)->dict:
    for path in (activation_path,protocol_path):
      if not path.is_file(): raise RuntimeError(f'Stage2 authority input missing: {path}')
    activation=load(activation_path)
    if activation.get('schema')!='openevo.ceiling1-stage2-activation.v1' or activation.get('status')!='authorized': raise RuntimeError('Stage2 activation is not authorized')
    if activation.get('stage2_config_sha256')!=sha(protocol_path): raise RuntimeError('Stage2 activation config SHA drift')
    if activation.get('formal_stage2_rollouts_consumed_at_activation')!=0 or activation.get('final_panel_access_count_at_activation')!=0: raise RuntimeError('Stage2 activation zero-state drift')
    per_arm=(activation.get('per_arm') or {}).get(arm) or {}
    if arm not in (activation.get('authorized_arms') or []) or per_arm.get('authorized') is not True: raise RuntimeError(f'Stage2 activation does not authorize arm {arm}')
    defaults={'stage1_complete_path':root/f'STAGE1_COMPLETE_{arm}.json','text_memory_complete_path':root/f'TEXT_MEMORY_COMPLETE_{arm}.json','skill_bundle_complete_path':root/f'SKILL_BUNDLE_COMPLETE_{arm}.json','agent_system_complete_path':root/f'AGENT_SYSTEM_COMPLETE_{arm}.json'}
    bound={key:Path(per_arm.get(key,value)).resolve() for key,value in defaults.items()}
    for path in bound.values():
      if not path.is_relative_to(root.resolve()) or not path.is_file(): raise RuntimeError(f'Stage2 authority input missing or outside analysis root: {path}')
    stage1_path=bound['stage1_complete_path']; text_path=bound['text_memory_complete_path']; skill_path=bound['skill_bundle_complete_path']
    stage1=load(stage1_path); text=load(text_path); skill=load(skill_path)
    if stage1.get('status')!='PASS' or per_arm.get('stage1_complete_sha256')!=sha(stage1_path): raise RuntimeError('Stage1 seal is not hash-bound by activation')
    if per_arm.get('text_memory_complete_sha256')!=sha(text_path) or per_arm.get('skill_bundle_complete_sha256')!=sha(skill_path): raise RuntimeError('Stage1 carrier markers are not hash-bound by activation')
    expected_text_schema=activation.get('authorized_text_memory_schema')
    if text.get('schema')!=expected_text_schema or text.get('status')!='pass': raise RuntimeError('Text Memory schema/status is not authorized')
    if expected_text_schema=='openevo.ceiling1-text-memory-stage1.a1':
      if text.get('split_events')!=0 or text.get('split_policy')!='disabled' or text.get('max_output_tokens')!=2048 or text.get('single_node_repair_max_tokens')!=2048: raise RuntimeError('A1 Text Memory no-split contract drift')
      if skill.get('status')!='pass' or skill.get('consumed_rows_total')!=1440 or skill.get('unique_source_ids')!=1440 or skill.get('split_events')!=0 or skill.get('max_output_tokens')!=2048: raise RuntimeError('A1 Skill Bundle no-split contract drift')
    else:
      agent_path=bound['agent_system_complete_path']
      agent=load(agent_path)
      exact=(
        ('text_memory',text,'authorized_text_memory_schema','authorized_text_memory_split_events','text_memory_complete_sha256',text_path),
        ('skill_bundle',skill,'authorized_skill_bundle_schema','authorized_skill_bundle_split_events','skill_bundle_complete_sha256',skill_path),
        ('agent_system',agent,'authorized_agent_system_schema','authorized_agent_system_split_events','agent_system_complete_sha256',agent_path),
      )
      for name,marker,schema_key,split_key,hash_key,path in exact:
        split=activation.get(split_key)
        if marker.get('schema')!=activation.get(schema_key) or type(split) is not int or split<0 or marker.get('split_events')!=split or per_arm.get(hash_key)!=sha(path): raise RuntimeError(f'A2 {name} split/schema/hash is not explicitly authorized')
    return {'activation_sha256':sha(activation_path),'stage1_complete_sha256':sha(stage1_path),**{key:str(path) for key,path in bound.items()}}

class Controller:
  def __init__(self,a):
    self.carrier_contract_fail_noop_authorized=False
    self.a=a; self.arm=a.arm; self.wt=a.repo.resolve(); self.root=a.analysis_root.resolve(); self.out=a.output.resolve(); self.out.mkdir(parents=True,exist_ok=True)
    self.execution_git_sha=a.execution_git_sha
    self.recovery=load(a.recovery_amendment) if a.recovery_amendment else None
    self.capacity_policy_path=a.capacity_policy.resolve() if a.capacity_policy else None
    self.manual_hold_release_path=a.manual_hold_release.resolve() if a.manual_hold_release else None
    self.capacity_policy=load_capacity_policy(self.capacity_policy_path,self.arm)
    self.capacity_policy_sha256=g8_sha256(self.capacity_policy_path) if self.capacity_policy_path else None
    self.sd_effective_rank_limit=int(self.capacity_policy["effective_rank_limit"]) if self.capacity_policy else 64*8
    if self.capacity_policy:
      core=Path(self.capacity_policy["openevo_core"]["checkout_path"]); expected=self.capacity_policy["openevo_core"]["execution_sha"]
      observed=cmd(["git","-C",core,"rev-parse","HEAD"],capture=True).stdout.strip(); dirty=cmd(["git","-C",core,"status","--porcelain"],capture=True).stdout.strip()
      if observed!=expected or dirty: raise RuntimeError("Harness201 successor capacity Core source identity drift")
    if self.execution_git_sha: assert_execution_source(self.wt,self.execution_git_sha)
    self.harness_profile=a.harness_profile
    self.step_seed_policy=a.step_seed_policy
    self.step_seed_amendment_path=a.step_seed_amendment.resolve() if a.step_seed_amendment else None
    self.step_seed_amendment=load_step_seed_amendment(self.step_seed_amendment_path)
    self.step_seed_amendment_sha256=g8_sha256(self.step_seed_amendment_path) if self.step_seed_amendment_path else None
    self.harness201_carrier_root=a.harness201_carrier_root.resolve() if a.harness201_carrier_root else None
    self.harness201_carrier_seed=load_harness201_carrier_seed(self.harness201_carrier_root)
    successor_fields=(self.harness_profile in {'2.0.1','201.1'},self.step_seed_policy=='injective-v1',self.step_seed_amendment is not None,self.harness201_carrier_seed is not None,self.capacity_policy is not None)
    if any(successor_fields) and not all(successor_fields): raise RuntimeError('Harness201 successor bundle is incomplete; require runtime+injective seed amendment+carrier seed+capacity policy')
    self.harness201_successor=all(successor_fields)
    if self.harness201_successor and self.arm not in {'3b','qwen3-1p7b'}: raise RuntimeError('Harness201 successor bundle unsupported for this arm')
    requested_resource=os.environ.get('CEILING1_STAGE2_RESOURCE_POLICY','auto')
    if requested_resource not in {'auto','v2','v3-eight-gpu','v4-four-gpu','v5-four-gpu'}: raise RuntimeError('unknown Ceiling Stage2 resource policy switch')
    full_v3_activation=LEASE_ROOT_V3/'activations/FULL_EIGHT_GPU_MIGRATED.json'
    full_v4_activation=LEASE_ROOT_V4/'activations/FOUR_GPU_CORESIDENT_V4.json'
    full_v5_activation=LEASE_ROOT_V5/'activations/FOUR_GPU_PRIORITY_ELASTIC_3B_V5.json'
    self.use_v5_resource=(requested_resource=='v5-four-gpu' or (requested_resource=='auto' and full_v5_activation.is_file()))
    self.use_v4_resource=(not self.use_v5_resource) and (requested_resource=='v4-four-gpu' or (requested_resource=='auto' and full_v4_activation.is_file()))
    if self.use_v5_resource or self.use_v4_resource:
      self.use_v3_resource=True
    elif self.arm=='3b':
      self.use_v3_resource=True
    else:
      self.use_v3_resource=full_v3_activation.is_file() if requested_resource=='auto' else requested_resource=='v3-eight-gpu'
    if full_v5_activation.is_file() and requested_resource in {'v2','v3-eight-gpu','v4-four-gpu'}: raise RuntimeError('explicit pre-v5 resource policy conflicts with committed v5 migration')
    if full_v4_activation.is_file() and requested_resource in {'v2','v3-eight-gpu'}: raise RuntimeError('explicit pre-v4 resource policy conflicts with committed v4 migration')
    if requested_resource=='v2' and full_v3_activation.is_file(): raise RuntimeError('explicit v2 resource policy conflicts with committed full v3 migration')
    if self.arm=='7b' and self.use_v3_resource and not self.use_v4_resource and not self.use_v5_resource: validate_v3_switch_boundary(self.out)
    if self.use_v5_resource:
      self.cfg=self.wt/'configs/experiment/openevo-ceiling-1.0-gpu-lease-v5-four-gpu-priority-elastic-3b.json'; self.lease_root=LEASE_ROOT_V5
    elif self.use_v4_resource:
      self.cfg=self.wt/'configs/experiment/openevo-ceiling-1.0-gpu-lease-v4-four-gpu-coresident.json'; self.lease_root=LEASE_ROOT_V4
    elif self.use_v3_resource:
      self.cfg=self.wt/'configs/experiment/openevo-ceiling-1.0-gpu-lease-v3-eight-gpu.json'; self.lease_root=LEASE_ROOT_V3
    else:
      self.cfg=self.wt/'configs/experiment/openevo-ceiling-1.0-gpu-lease-v2.json'; self.lease_root=LEASE_ROOT_V2
    if self.arm=='qwen3-1p7b':
      self.cfg=self.wt/'configs/experiment/openevo-ceiling-1.0-gpu-lease-qwen3-stage2-v1.json'; self.lease_root=LEASE_ROOT_Q17; self.use_v3_resource=True
    self.lease=self.wt/'scripts/openevo_webshop/ceiling1_gpu_lease_orchestrator.py'; self.protocol=self.wt/'configs/experiment/openevo-ceiling-1.0-stage2-final-v1.json'; self.config_sha=sha(self.protocol); self.activation=a.stage2_activation.resolve()
    protocol=load(self.protocol); schedule=self.wt/protocol['schedule']['path']
    schedule_sha=sha(schedule)
    if schedule_sha!=protocol['schedule']['sha256']: raise RuntimeError('Stage2 schedule SHA drift')
    if protocol['rounds']!=160 or protocol['round_rollouts']!=128 or protocol['total_rollouts']!=20480: raise RuntimeError('Stage2 budget config drift')
    scope_arg=getattr(a,'carrier_scope_package',None)
    self.carrier_scope_package_path=Path(scope_arg).resolve() if scope_arg else None
    self.carrier_scope_package=(load_carrier_scope_package(self.carrier_scope_package_path,expected_schedule_sha256=schedule_sha,expected_round_count=int(protocol['rounds'])) if self.carrier_scope_package_path else None)
    self.carrier_scope_package_identity=(self.carrier_scope_package.get('package_identity_sha256') if self.carrier_scope_package else None)
    health_policy_arg=getattr(a,'carrier_health_policy',None); identities_arg=getattr(a,'carrier_contract_identities',None); adoption_arg=getattr(a,'carrier_adoption_receipt',None)
    formal_carrier_mode=any(bool(x) for x in (health_policy_arg,identities_arg,adoption_arg))
    if formal_carrier_mode and not all(bool(x) for x in (health_policy_arg,identities_arg,adoption_arg,scope_arg,self.execution_git_sha)):
      raise RuntimeError('Carrier Contract formal mode requires scope package + health policy + implementation identity + adoption receipt + execution Git SHA')
    self.carrier_health_policy_path=Path(health_policy_arg).resolve() if health_policy_arg else None
    self.carrier_health_policy=load_carrier_health_policy(self.carrier_health_policy_path) if self.carrier_health_policy_path else None
    self.carrier_health_policy_identity=(carrier_health_policy_identity(self.carrier_health_policy) if self.carrier_health_policy else None)
    self.carrier_contract_identities_path=Path(identities_arg).resolve() if identities_arg else None
    self.carrier_contract_identities=(load_carrier_contract_identity(self.carrier_contract_identities_path,self.wt) if self.carrier_contract_identities_path else None)
    self.carrier_contract_identities_sha256=sha(self.carrier_contract_identities_path) if self.carrier_contract_identities_path else None
    self.carrier_contract_identity=(self.carrier_contract_identities.get('carrier_contract_identity_sha256') if self.carrier_contract_identities else None)
    self.carrier_health_component_identities=(self.carrier_contract_identities.get('health_component_identities') if self.carrier_contract_identities else None)
    self.carrier_adoption_receipt_path=Path(adoption_arg).resolve() if adoption_arg else None
    self.carrier_adoption_receipt=(load(self.carrier_adoption_receipt_path) if self.carrier_adoption_receipt_path else None)
    if self.carrier_adoption_receipt is not None:
      validate_carrier_contract_adoption(self.carrier_adoption_receipt,identity_path=self.carrier_contract_identities_path,repo_root=self.wt,scope_package_path=self.carrier_scope_package_path,expected_execution_git_sha=self.execution_git_sha)
    self.carrier_adoption_identity=(self.carrier_adoption_receipt.get('adoption_identity_sha256') if self.carrier_adoption_receipt else None)
    self.carrier_health_resolution_path=Path(a.carrier_health_resolution).resolve() if getattr(a,'carrier_health_resolution',None) else None
    self.exec_path=self.out/'execution-state.json'; self.qual=Path('/workspace/runs/20260830-openevo-ceiling-gpu7-qualification')
    self.entry_authority=validate_stage2_entry_authority(self.root,self.arm,self.activation,self.protocol)
    if not (self.root/'stage2-base'/self.arm/'merge-receipt.json').is_file(): raise RuntimeError('post-OPSD merged base not ready')
    self.ensure_writable(self.out)
    self.state=self.load_or_init()
  def allowed_rollout_shas(self,r:int)->set[str]|None:
    if not self.execution_git_sha:return None
    allowed={self.execution_git_sha}
    if self.recovery:
      allowed.update(self.recovery.get('historical_rollout_reuse',{}).get(str(r),[]))
      if r==0: allowed.add('51ce92498a4fce306d502fa3a42c67b2ae6bf5ef')
    return allowed
  def allowed_artifact_execution_shas(self,r:int)->set[str]|None:
    return {self.execution_git_sha} if self.execution_git_sha else None
  def ensure_writable(self,p:Path): cmd([self.wt/'scripts/openevo_webshop/ensure_ceiling1_science_output_writable.sh',p])
  def load_or_init(self):
    if self.exec_path.is_file():
      s=load(self.exec_path)
      if s['config_sha256']!=self.config_sha: raise RuntimeError('Stage2 config SHA drift')
      if s.get('stage2_activation_sha256') not in (None,self.entry_authority['activation_sha256']): raise RuntimeError('Stage2 activation SHA drift')
      s.setdefault('stage2_activation_sha256',self.entry_authority['activation_sha256'])
      bind_carrier_scope_package_identity(s,self.carrier_scope_package_identity)
      bind_preformal_identity(s,'carrier_health_policy_identity_sha256',self.carrier_health_policy_identity,'Carrier Health policy')
      bind_preformal_identity(s,'carrier_contract_identities_sha256',self.carrier_contract_identities_sha256,'Carrier Contract identity manifest')
      bind_preformal_identity(s,'carrier_contract_identity_sha256',self.carrier_contract_identity,'Carrier Contract content identity')
      bind_preformal_identity(s,'carrier_adoption_identity_sha256',self.carrier_adoption_identity,'Carrier Contract adoption receipt')
      if self.capacity_policy:
        expected={'capacity_policy_sha256':self.capacity_policy_sha256,'capacity_effective_rank_limit':self.sd_effective_rank_limit,'capacity_independent_component_stop':None,'capacity_core_sha':self.capacity_policy['openevo_core']['execution_sha']}
        for k,v in expected.items():
          if k in s and s[k]!=v: raise RuntimeError(f'Harness201 successor capacity state drift: {k}')
          s.setdefault(k,v)
      if self.harness201_successor:
        expected_successor={'harness_profile':self.harness_profile,'step_seed_policy':'injective-v1','step_seed_amendment_sha256':self.step_seed_amendment_sha256,'harness201_carrier_seed_receipt_sha256':self.harness201_carrier_seed['receipt_sha256']}
        for k,v in expected_successor.items():
          if k in s and s[k]!=v: raise RuntimeError(f'Harness201 successor state drift: {k}')
          s.setdefault(k,v)
      if s.get('status')=='STAGE2_SEALED': return s
      if s.get('status')=='PAUSE_FOR_DIAGNOSTIC':
        if self.carrier_health_policy is None: raise RuntimeError('Carrier Health pause requires the frozen health policy')
        pause_path=Path(str(s.get('carrier_health_pause_path') or ''))
        if not pause_path.is_file() or sha(pause_path)!=s.get('carrier_health_pause_sha256'): raise RuntimeError('Carrier Health pause receipt missing or drifted')
        pause=load(pause_path)
        if self.carrier_health_resolution_path is None: raise RuntimeError('Carrier Health pause requires explicit diagnostic resolution receipt')
        if not self.carrier_health_resolution_path.is_file(): raise RuntimeError('Carrier Health resolution receipt missing')
        resolution=load(self.carrier_health_resolution_path)
        validate_resolution_receipt(self.carrier_health_policy,resolution,expected_arm=self.arm,expected_round=int(pause.get('through_round',-1)),expected_pause_receipt_sha256=sha(pause_path))
        s.setdefault('controller_recoveries',[]).append({'recovery_kind':'CARRIER_HEALTH_DIAGNOSTIC_RESOLVED','through_round':int(pause['through_round']),'pause_receipt':str(pause_path),'pause_receipt_sha256':sha(pause_path),'resolution_receipt':str(self.carrier_health_resolution_path),'resolution_receipt_sha256':sha(self.carrier_health_resolution_path),'recovered_at_epoch':time.time()})
        for key in ('blocker','carrier_health_pause_path','carrier_health_pause_sha256','carrier_health_paused_at_epoch'): s.pop(key,None)
      if s.get('status')=='BLOCKED':
        if blocker_is_nonrecoverable(s.get('blocker')):
          if not validate_manual_hold_release(self.manual_hold_release_path,s): raise RuntimeError('Stage2 state is under a non-recoverable readiness/circuit-breaker hold')
          s.setdefault('controller_recoveries',[]).append({'recovered_at_epoch':time.time(),'previous_blocker':s.get('blocker'),'manual_hold_release_sha256':g8_sha256(self.manual_hold_release_path)})
          s.pop('blocker',None); s.pop('blocked_at_epoch',None)
        elif 'reflector failed' in str(s.get('blocker','')):
          recovery_path=getattr(self.a,'science_recovery_amendment',None)
          if not recovery_path: raise RuntimeError('blocked reflector requires explicit audited recovery')
          recovery_path=Path(recovery_path).resolve(); recovery_doc=load(recovery_path); schema=recovery_doc.get('schema')
          if schema=='openevo.ceiling1-stage2-reflector-bounded-projection-recovery.v2':
            if recovery_doc.get('status')!='AUTHORIZED': raise RuntimeError('blocked reflector recovery amendment invalid')
            return s
          if schema!='openevo.ceiling1-stage2-reflector-contract-binding-recovery.v1' or recovery_doc.get('status')!='AUTHORIZED': raise RuntimeError('blocked reflector recovery amendment invalid')
          ar=(recovery_doc.get('arms') or {}).get(self.arm) or {}; rr=int(ar.get('partial_round',-1)); ready=Path(str(ar.get('ready_receipt_path','')))
          if rr!=int(s.get('next_round',-2)) or not ready.is_file(): raise RuntimeError('reflector contract recovery ready receipt missing/round drift')
          rx=load(ready)
          if rx.get('schema')!='openevo.ceiling1-stage2-reflector-contract-recovery-ready.v1' or rx.get('status')!='PASS' or rx.get('arm')!=self.arm or int(rx.get('round_index',-1))!=rr: raise RuntimeError('reflector contract recovery ready receipt invalid')
          if rx.get('recovery_amendment_sha256')!=g8_sha256(recovery_path) or rx.get('scientific_contract_changed') is not False or int(rx.get('formal_rollout_replay',-1))!=0: raise RuntimeError('reflector contract recovery accounting drift')
          term=Path(str(ar.get('terminal_receipt_path','')))
          if not term.is_file() or g8_sha256(term)!=rx.get('canonical_terminal_sha256'): raise RuntimeError('reflector contract recovery terminal drift')
          s.setdefault('controller_recoveries',[]).append({'recovery_kind':'REFLECTOR_ARTIFACT_CONTRACT_BINDING','round_index':rr,'carrier':ar.get('kind'),'ready_receipt':str(ready),'ready_receipt_sha256':g8_sha256(ready),'previous_blocker':s.get('blocker'),'recovered_at_epoch':time.time()})
          s.pop('blocker',None);s.pop('blocked_at_epoch',None)
        else:
          s.setdefault('controller_recoveries',[]).append({'recovered_at_epoch':time.time(),'previous_blocker':s.get('blocker')}); s.pop('blocker',None); s.pop('blocked_at_epoch',None)
      s['status']='RUNNING'; atomic_json(self.exec_path,s); return s
    stage1=load(Path(self.entry_authority['stage1_complete_path'])); base=self.root/'stage2-base'/self.arm/'verification-receipt.json'
    if self.harness201_successor:
      mem=Path(self.harness201_carrier_seed['memory']); skill=Path(self.harness201_carrier_seed['skill']); agent=Path(self.harness201_carrier_seed['agent'])
      expected={'text_memory':self.harness201_carrier_seed['memory_sha256'],'skill_bundle':self.harness201_carrier_seed['skill_sha256'],'agent_system':self.harness201_carrier_seed['agent_sha256']}
    else:
      mem=Path(stage1['text_memory']['path']); skill=Path(stage1['skill_bundle']['path']); agent=Path(stage1['agent_system']['path']); expected={k:stage1[k]['sha256'] for k in ('text_memory','skill_bundle','agent_system')}
    for path,key in ((mem,'text_memory'),(skill,'skill_bundle'),(agent,'agent_system')):
      if not path.is_file() or sha(path)!=expected[key]: raise RuntimeError(f'initial {key} artifact identity drift')
    if self.harness201_successor:
      from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import require_evolution_compatible_bundle
      require_evolution_compatible_bundle(mem.read_text(),skill.read_text(),agent.read_text(),context='new Stage2 state')
    b=load(base)
    if b.get('status')!='PASS' or b.get('opsd_adapter_sha256')!=stage1['opsd']['adapter_sha256'] or b.get('stage2_config_sha256')!=self.config_sha: raise RuntimeError('post-OPSD Stage2 base identity drift')
    sid=state_id(base,None,mem,skill,agent,-1); s={'schema':'openevo.ceiling1-stage2-execution-state.v1','status':'RUNNING','arm':self.arm,'config_sha256':self.config_sha,'stage2_activation_sha256':self.entry_authority['activation_sha256'],'next_round':0,'rollouts_consumed':0,'state_id':sid,'merged_base_receipt':str(base),'current_sd_adapter':None,'current_sd_artifact_id':None,'sd_component_count':0,'sd_generation_count':0,'memory':str(mem),'skill':str(skill),'agent_system':str(agent),'evidence_history':None,'rounds_complete':[],'external_teacher_calls':0,'final_panel_access_count':0};
    if self.capacity_policy: s.update({'capacity_policy_sha256':self.capacity_policy_sha256,'capacity_effective_rank_limit':self.sd_effective_rank_limit,'capacity_independent_component_stop':None,'capacity_core_sha':self.capacity_policy['openevo_core']['execution_sha']})
    if self.harness201_successor: s.update({'harness_profile':self.harness_profile,'step_seed_policy':'injective-v1','step_seed_amendment_sha256':self.step_seed_amendment_sha256,'harness201_carrier_seed_receipt_sha256':self.harness201_carrier_seed['receipt_sha256']})
    if self.carrier_scope_package_identity is not None: s['carrier_scope_package_identity_sha256']=self.carrier_scope_package_identity
    if self.carrier_health_policy_identity is not None: s['carrier_health_policy_identity_sha256']=self.carrier_health_policy_identity
    if self.carrier_contract_identities_sha256 is not None: s['carrier_contract_identities_sha256']=self.carrier_contract_identities_sha256
    if self.carrier_contract_identity is not None: s['carrier_contract_identity_sha256']=self.carrier_contract_identity
    if self.carrier_adoption_identity is not None: s['carrier_adoption_identity_sha256']=self.carrier_adoption_identity
    atomic_json(self.exec_path,s); return s
  def save(self): atomic_json(self.exec_path,self.state)
  def scope_manifest_for_round(self,r:int)->Path|None:
    if self.carrier_scope_package is None: return None
    path=manifest_path_for_round(self.carrier_scope_package_path,self.carrier_scope_package,r)
    from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import load_scope_manifest
    load_scope_manifest(path,expected_round_index=r)
    return path
  def carrier_scope_round_binding_payload(self,r:int)->dict|None:
    scope_path=self.scope_manifest_for_round(r)
    if scope_path is None: return None
    from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import load_scope_manifest
    scope=load_scope_manifest(scope_path,expected_round_index=r)
    return {'schema':'openevo.carrier-scope-round-binding.v1','status':'PASS','round_index':r,'carrier_scope_package_identity_sha256':self.carrier_scope_package_identity,'scope_manifest_sha256':sha(scope_path),'scope_manifest_identity_sha256':scope['manifest_identity_sha256'],'source_task_digest_sha256':scope['source_task_digest_sha256'],'source_task_count':scope['source_task_count'],'scientific_semantics':'outcome-blind current-round scope authority; validates newly generated carrier candidates only; unchanged prior carrier bytes are not reclassified as current-task leakage'}
  def write_carrier_scope_round_binding(self,r:int,rr:Path)->dict|None:
    receipt=self.carrier_scope_round_binding_payload(r)
    if receipt is None: return None
    path=rr/'carrier-scope-round-binding.json'
    if not path.is_file() and any((rr/name).exists() for name in ('attempts.jsonl','summary.json','ROUND_COMPLETE.json','state-after.json')):
      raise RuntimeError('Carrier scope binding missing after formal round materialization')
    write_once_json(path,receipt)
    return receipt
  def validate_carrier_scope_round_binding(self,r:int,rr:Path)->dict|None:
    expected=self.carrier_scope_round_binding_payload(r)
    if expected is None: return None
    path=rr/'carrier-scope-round-binding.json'
    if not path.is_file(): raise RuntimeError('Carrier scope round binding missing')
    observed=load(path)
    if observed!=expected: raise RuntimeError('Carrier scope round binding identity drift')
    return observed
  def apply_carrier_health_boundary(self,r:int,rr:Path)->dict|None:
    if self.carrier_health_policy is None: return None
    complete=rr/'ROUND_COMPLETE.json'; after=rr/'state-after.json'
    if not complete.is_file() or not after.is_file(): raise RuntimeError('Carrier Health boundary requires a sealed round transaction')
    ledger=build_round_ledger(rr,self.arm,self.carrier_health_component_identities)
    ledger_path=rr/'carrier-health-ledger.json'; write_carrier_health_json(ledger_path,ledger)
    decision=formal_round_decision(self.carrier_health_policy,ledger)
    decision_path=rr/'carrier-health-boundary.json'; write_once_json(decision_path,decision)
    if decision['action']=='PAUSE_FOR_DIAGNOSTIC':
      pause_path=self.out/'carrier-health-pauses'/f'round-{r:03d}.json'
      pause={'schema':'openevo.carrier-health-pause.v1','status':'PAUSE_FOR_DIAGNOSTIC','arm':self.arm,'through_round':r,'next_round':r+1,'policy_identity_sha256':self.carrier_health_policy_identity,'carrier_contract_identities_sha256':self.carrier_contract_identities_sha256,'carrier_contract_identity_sha256':self.carrier_contract_identity,'carrier_adoption_identity_sha256':self.carrier_adoption_identity,'health_ledger_sha256':sha(ledger_path),'boundary_decision_sha256':sha(decision_path),'round_complete_sha256':sha(complete),'state_after_sha256':sha(after),'automatic_scientific_change':False,'formal_transaction_preserved':True}
      write_once_json(pause_path,pause)
      self.state=load(after); self.state['status']='PAUSE_FOR_DIAGNOSTIC'; self.state['blocker']=f'Carrier Health FAIL through round {r}; diagnostic resolution required before round {r+1}'; self.state['carrier_health_pause_path']=str(pause_path); self.state['carrier_health_pause_sha256']=sha(pause_path); self.state['carrier_health_paused_at_epoch']=time.time(); self.save()
      raise RuntimeError(self.state['blocker'])
    return decision
  def ensure_previous_carrier_health_boundary(self,r:int)->None:
    if self.carrier_health_policy is None or r<=0: return
    previous=self.out/f'round-{r-1:03d}'
    if not (previous/'ROUND_COMPLETE.json').is_file(): raise RuntimeError('previous round missing before Carrier Health boundary check')
    self.apply_carrier_health_boundary(r-1,previous)
  def write_round_interlock(self,rr:Path,r:int)->None:
    if self.capacity_policy and r < 4:
      release=validate_staged_interlock_round(rr,self.capacity_policy_sha256); write_once_release(rr/'STAGED_INTERLOCK_RELEASE.json',release)
  def lease_reconcile(self): cmd(['python3',self.lease,'--config',self.cfg,'--state-root',self.lease_root,'reconcile'],check=False,capture=True)
  def live_leases(self):
    self.lease_reconcile(); p=self.lease_root/'state.json'; return load(p) if p.is_file() else {'gpus':{str(i):{'leases':[]} for i in UUID}}
  def choose(self,kind:str,round_idx:int|None=None,workload:str|None=None):
    while True:
      dynamic_v3=self.use_v3_resource
      if dynamic_v3:
        if round_idx is None: raise RuntimeError('v3 resource selection requires round identity')
        workload=workload or f'stage2_rollout_{self.arm}'
        rr=self.out/f'round-{round_idx:03d}'
        selection=rr/('resource-selection.json' if kind=='rollout' else f'{workload}-resource-selection.json')
        if kind=='rollout': expected_count=None if (self.arm=='3b' and self.use_v5_resource) else (8 if self.arm=='3b' else 4)
        else: expected_count=1
        if selection.is_file():
          value=load(selection)
          if value.get('workload')!=workload or value.get('config_sha256')!=sha(self.cfg): raise RuntimeError('round resource selection identity mismatch')
          selected=list(map(int,value.get('selected_gpus',[])))
          if kind=='rollout':
            if self.arm=='3b' and self.use_v5_resource: allowed_counts=set(range(1,5))
            else: allowed_counts={8} if self.arm=='3b' else {2,4}
            if len(selected) not in allowed_counts: raise RuntimeError('round rollout resource selection count mismatch')
          elif len(selected)!=1: raise RuntimeError('single-GPU resource selection count mismatch')
          return selected,selection
        args=['python3',self.lease,'--config',self.cfg,'--state-root',self.lease_root,'select','--workload',workload,'--output',selection]
        if expected_count is not None: args += ['--count',str(expected_count)]
        owner=os.environ.get('CEILING1_GPU_OWNER_RECEIPT'); qualification=os.environ.get('CEILING1_3B_7B_CORESIDENCY_QUALIFICATION')
        if owner: args+=['--owner-receipt',owner]
        if qualification: args+=['--coresidency-qualification',qualification]
        result=cmd(args,check=False,capture=True)
        if result.returncode==0: return list(map(int,load(selection)['selected_gpus'])),selection
        time.sleep(2); continue
      s=self.live_leases();
      if kind=='rollout': candidates=[4,5]; allow={'stage2_rollout_3b'}; chosen=[]
      elif kind=='reflector': candidates=[6,7,0,1,2,3] if self.arm=='3b' else [4,5]; allow=set(); chosen=[]
      else: candidates=[6,7] if self.arm=='3b' else [5,4]; allow=set(); chosen=[]
      for i in candidates:
        ws={q['workload'] for q in s['gpus'][str(i)]['leases']}
        if not ws or ws<=allow: chosen.append(i)
      if chosen: return (chosen if kind=='rollout' else [chosen[0]]),None
      time.sleep(2)
  def lease_run(self,*,lease_id,workload,container,launcher,gpus,env,vram=None,selection=None):
    args=['python3',self.lease,'--config',self.cfg,'--state-root',self.lease_root,'run','--lease-id',lease_id,'--workload',workload,'--container',container,'--launch-script',launcher,'--gpus',*map(str,gpus)]
    if vram: args+=['--vram-qualification',vram]
    if selection: args+=['--selection-receipt',selection]
    r=cmd(args,env={**os.environ,**env},check=False,capture=True)
    if r.returncode: return False,r.stderr.strip() or r.stdout.strip()
    return True,r.stdout.strip()
  def launch_retry(self,kind,round_idx,workload,launcher,env,vram=None,max_attempts=4):
    last=''; attempt=0; launched=0
    while launched<max_attempts:
      stem=f'openevo-ceiling1-stage2-{self.arm}-r{round_idx:03d}-{env.get("CEILING1_STAGE2_KIND",kind)}'
      while dstatus(f'{stem}-a{attempt}')!='absent': attempt+=1
      gpus,selection=self.choose(kind,round_idx,workload); c=f'{stem}-a{attempt}'; lid=f'ceiling1-stage2-{self.arm}-r{round_idx:03d}-{env.get("CEILING1_STAGE2_KIND",kind)}-a{attempt}'
      launched+=1; attempt+=1
      env2={**env,'CEILING1_STAGE2_CONTAINER':c}
      if self.execution_git_sha: env2['CEILING1_STAGE2_EXECUTION_GIT_SHA']=self.execution_git_sha
      if kind=='rollout': env2['CEILING1_STAGE2_GPU_UUIDS']=','.join(UUID[i] for i in gpus)
      else: env2['CEILING1_STAGE2_GPU_UUID']=UUID[gpus[0]]
      ok,last=self.lease_run(lease_id=lid,workload=workload,container=c,launcher=launcher,gpus=gpus,env=env2,vram=vram,selection=selection)
      if not ok: time.sleep(2); continue
      rc=wait_container(c); self.lease_reconcile()
      if rc==0: return c,gpus
      last=f'container {c} exit={rc}'; time.sleep(2)
    raise RuntimeError(f'{kind} failed after {max_attempts} engineering attempts: {last}')
  def noop_payload(self,kind:str,reason:str,prior:Path|None,round_idx:int):
    return {'schema':'openevo.ceiling1-stage2-carrier-terminal.v1','status':'NOOP','kind':kind,'round_index':round_idx,'reason':reason,'prior_sha256':sha(prior) if prior else None,'output_sha256':sha(prior) if prior else None,'primary_calls':0,'repair_calls':0,'external_teacher_calls':0}
  def noop(self,path:Path,kind:str,reason:str,prior:Path|None,round_idx:int):
    x=self.noop_payload(kind,reason,prior,round_idx); atomic_json(path,x); return x
  def validate_rollout_harness_readback(self,summary:dict,rr:Path)->None:
    if summary.get('harness_profile')!=self.harness_profile or summary.get('step_seed_policy')!=self.step_seed_policy: raise RuntimeError('round Harness/step-seed summary readback mismatch')
    if self.harness201_successor:
      rows=[json.loads(line) for line in (rr/'attempts.jsonl').read_text().split('\n') if line.strip()]
      if len(rows)!=128: raise RuntimeError('Harness201 round readback requires exact 128 rows')
      if any(row.get('harness_profile')!=self.harness_profile or row.get('step_seed_policy')!='injective-v1' for row in rows): raise RuntimeError('Harness201 successor row-level profile/seed readback mismatch')
      if self.harness_profile=='201.1':
        expected_prompt='8efcba565fc11e485192f6ca85cb62119f7322662cff1a5e6597de0608d441b5'; expected_runtime='201.1'
      else:
        expected_prompt='5f458165f277384036df532059d4e771c3ab96a496923ad1b5ae568d4eb9499a'; expected_runtime='2.0.1-thin'
      if any(row.get('prompt_contract_sha256')!=expected_prompt for row in rows): raise RuntimeError('Harness201 successor prompt contract readback mismatch')
      if any(any(not harness_runtime_step_readback_ok(step,expected_runtime) for step in row.get('episode',{}).get('trace',[])) for row in rows): raise RuntimeError('Harness201 successor runtime receipt readback mismatch')

  def run_rollout(self,r:int,rr:Path):
    allowed=self.allowed_rollout_shas(r)
    if (rr/'summary.json').is_file():
      summary=load(rr/'summary.json'); validate_rollout_summary(summary,r,self.state['state_id']); validate_attempt_rows(rr,r,self.state['state_id'],allowed); self.validate_rollout_harness_readback(summary,rr); return
    sd_adapter=Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None
    env={'CEILING1_STAGE2_ARM':self.arm,'CEILING1_ANALYSIS_ROOT':str(self.root),'CEILING1_AUTOPILOT_REPO':str(self.wt),'CEILING1_SEED_ROOT':str(self.a.seed_root.resolve()),'CEILING1_STAGE2_ROUND':str(r),'CEILING1_STAGE2_STATE_ID':self.state['state_id'],'CEILING1_STAGE2_ROUND_ROOT':str(rr),'CEILING1_STAGE2_MEMORY':self.state['memory'],'CEILING1_STAGE2_SKILL':self.state['skill'],'CEILING1_STAGE2_AGENT':self.state['agent_system'],'CEILING1_STAGE2_SD_ADAPTER':str(sd_adapter) if sd_adapter else '','CEILING1_STAGE2_CONFIG_SHA':self.config_sha,'CEILING1_STAGE2_HARNESS_PROFILE':self.harness_profile,'CEILING1_STAGE2_STEP_SEED_POLICY':self.step_seed_policy}
    self.ensure_writable(rr); rollout_launcher=self.wt/'scripts/openevo_webshop'/('launch_ceiling1_stage2_round_qwen3_container.sh' if self.arm=='qwen3-1p7b' else 'launch_ceiling1_stage2_round_container.sh')
    self.launch_retry('rollout',r,f'stage2_rollout_{self.arm}',rollout_launcher,env,max_attempts=4)
    summary=load(rr/'summary.json'); validate_rollout_summary(summary,r,self.state['state_id']); validate_attempt_rows(rr,r,self.state['state_id'],allowed); self.validate_rollout_harness_readback(summary,rr)
  def evidence(self,r:int,rr:Path):
    er=rr/'evidence'; e=er/'evidence.json'
    prior=Path(self.state['evidence_history']) if self.state.get('evidence_history') else None
    if e.is_file(): x=load(e); validate_evidence_result(x,rr,r,prior); return x
    args=['python3',self.wt/'scripts/openevo_webshop/ceiling1_stage2_evidence.py','--attempts',rr/'attempts.jsonl','--output',er,'--round-index',r]
    if prior: args+=['--prior-history',prior]
    cmd(args); x=load(e); validate_evidence_result(x,rr,r,prior); return x
  def reflect(self,r:int,rr:Path,kind:str,prior:Path,memory:Path|None=None,skill:Path|None=None,sd_adapter:Path|None=None):
    ev=load(rr/'evidence/evidence.json'); gate=ev['gates'][kind]; d=rr/kind.replace('_','-'); d.mkdir(parents=True,exist_ok=True); self.ensure_writable(d)
    name={'text_memory':'memory.md','skill_bundle':'SKILL.md','agent_system':'agent_system.md'}[kind]; out=d/name; receipt_path=out.with_suffix(out.suffix+'.receipt.json'); term=d/'terminal-receipt.json'; attempt_path=d/'model-call-attempt.json'; evidence_path=rr/'evidence/evidence.json'
    contract=({'text_memory':'harness201-runtime-memory-v1','skill_bundle':'harness201-runtime-skill-v1','agent_system':'harness201-runtime-agent-v1'}[kind] if self.harness201_successor else None)
    scope_manifest=self.scope_manifest_for_round(r)
    if term.is_file():
      cached=load(term)
      if cached.get('schema')!='openevo.ceiling1-stage2-carrier-terminal.v1' or cached.get('kind')!=kind or int(cached.get('round_index',-1))!=r: raise RuntimeError('carrier terminal receipt identity mismatch')
      if cached.get('reflector_receipt') is not None:
        if not out.is_file(): raise RuntimeError('cached reflector output is missing')
        validate_reflector_result(cached['reflector_receipt'],kind,r,self.state['state_id'],evidence_path,prior,memory,skill,sd_adapter,out,scope_manifest)
        changed=sha(out)!=sha(prior); expected_status='UPDATE' if changed else 'NOOP'; expected_path=out if changed else prior
      else:
        if gate.get('eligible'):
          if cached.get('reason')!=REFLECTOR_CONTRACT_FAIL_NOOP_REASON: raise RuntimeError('eligible carrier cannot resume from a gate-only NOOP terminal')
          if not self.carrier_contract_fail_noop_authorized: raise RuntimeError('carrier contract-fail NOOP requires explicit authorized science amendment')
          source_dir=Path(cached.get('contract_fail_evidence_dir') or d)
          source_attempt=Path(cached.get('model_call_attempt_path') or attempt_path)
          expected_noop=build_reflector_contract_fail_noop(kind,r,self.state['state_id'],evidence_path,prior,memory,skill,sd_adapter,source_dir,source_attempt,contract,scope_manifest)
          if expected_noop is None or cached!=expected_noop: raise RuntimeError('cached reflector contract-fail NOOP evidence drift')
        expected_status='NOOP'; expected_path=prior
      if cached.get('status')!=expected_status or Path(cached.get('output_path','')).resolve()!=expected_path.resolve() or cached.get('output_sha256')!=sha(expected_path): raise RuntimeError('carrier terminal receipt payload mismatch')
      return expected_path
    if not gate['eligible']:
      x=self.noop(term,kind,gate['noop_reason'],prior,r); x['output_path']=str(prior); atomic_json(term,x); return prior
    attempt=reflector_attempt_identity(kind,r,self.state['state_id'],evidence_path,prior,memory,skill,sd_adapter,contract,scope_manifest)
    if out.exists() or receipt_path.exists():
      if not out.is_file() or not receipt_path.is_file(): raise RuntimeError('incomplete reflector result; refusing duplicate model call')
      rec=load(receipt_path); validate_reflector_result(rec,kind,r,self.state['state_id'],evidence_path,prior,memory,skill,sd_adapter,out,scope_manifest)
      changed=sha(out)!=sha(prior); x={'schema':'openevo.ceiling1-stage2-carrier-terminal.v1','status':'UPDATE' if changed else 'NOOP','kind':kind,'round_index':r,'reason':None if changed else 'NOOP_NO_CONTENT_CHANGE','output_path':str(out) if changed else str(prior),'output_sha256':sha(out) if changed else sha(prior),'reflector_receipt':rec,'recovered_without_model_recall':True}; atomic_json(term,x); return out if changed else prior
    if attempt_path.exists():
      noop=(build_reflector_contract_fail_noop(kind,r,self.state['state_id'],evidence_path,prior,memory,skill,sd_adapter,d,attempt_path,contract,scope_manifest) if self.carrier_contract_fail_noop_authorized else None)
      if noop is not None: atomic_json(term,noop); return prior
      raise RuntimeError('reflector attempt already exists without a complete result; refusing duplicate model call')
    write_once_json(attempt_path,attempt)
    env={'CEILING1_STAGE2_ARM':self.arm,'CEILING1_ANALYSIS_ROOT':str(self.root),'CEILING1_AUTOPILOT_REPO':str(self.wt),'CEILING1_STAGE2_KIND':kind,'CEILING1_STAGE2_EVIDENCE':str(rr/'evidence/evidence.json'),'CEILING1_STAGE2_PRIOR':str(prior),'CEILING1_STAGE2_OUTPUT':str(out),'CEILING1_STAGE2_STATE_ID':self.state['state_id'],'CEILING1_STAGE2_ROUND':str(r),'CEILING1_STAGE2_SD_ADAPTER':str(sd_adapter) if sd_adapter else '','CEILING1_STAGE2_MEMORY':str(memory) if memory else '','CEILING1_STAGE2_SKILL':str(skill) if skill else ''}
    if contract is not None: env['CEILING1_STAGE2_ARTIFACT_CONTRACT']=contract
    if scope_manifest is not None: env['CEILING1_STAGE2_SCOPE_MANIFEST']=str(scope_manifest)
    workload={'text_memory':f'stage2_text_memory_{self.arm}','skill_bundle':f'stage2_skill_bundle_{self.arm}','agent_system':f'stage2_agent_system_{self.arm}'}[kind]
    reflector_launcher=self.wt/'scripts/openevo_webshop'/('launch_ceiling1_stage2_reflector_qwen3_container.sh' if self.arm=='qwen3-1p7b' else 'launch_ceiling1_stage2_reflector_container.sh')
    try:
      self.launch_retry('reflector',r,workload,reflector_launcher,env,max_attempts=1)
    except RuntimeError:
      noop=(build_reflector_contract_fail_noop(kind,r,self.state['state_id'],evidence_path,prior,memory,skill,sd_adapter,d,attempt_path,contract,scope_manifest) if self.carrier_contract_fail_noop_authorized else None)
      if noop is not None: atomic_json(term,noop); return prior
      raise
    if not out.is_file() or not receipt_path.is_file():
      noop=(build_reflector_contract_fail_noop(kind,r,self.state['state_id'],evidence_path,prior,memory,skill,sd_adapter,d,attempt_path,contract,scope_manifest) if self.carrier_contract_fail_noop_authorized else None)
      if noop is not None: atomic_json(term,noop); return prior
      raise RuntimeError('reflector container exited without a complete result')
    rec=load(receipt_path); validate_reflector_result(rec,kind,r,self.state['state_id'],evidence_path,prior,memory,skill,sd_adapter,out,scope_manifest); changed=sha(out)!=sha(prior); x={'schema':'openevo.ceiling1-stage2-carrier-terminal.v1','status':'UPDATE' if changed else 'NOOP','kind':kind,'round_index':r,'reason':None if changed else 'NOOP_NO_CONTENT_CHANGE','output_path':str(out) if changed else str(prior),'output_sha256':sha(out) if changed else sha(prior),'reflector_receipt':rec}; atomic_json(term,x); return out if changed else prior
  def run_sd(self,r:int,rr:Path):
    d=rr/'sd-lora'; d.mkdir(parents=True,exist_ok=True); self.ensure_writable(d); term=d/'terminal-receipt.json'
    if term.is_file():
      x=load(term)
      prior=Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None
      return validate_sd_terminal(x,r,prior,self.state.get('current_sd_artifact_id'),int(self.state.get('sd_component_count',0)),self.arm,self.config_sha,self.execution_git_sha)
    prior=Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None
    rec=d/'update.json'; ds=d/'dataset'; sel=ds/'selection.json'
    if rec.is_file():
      if not sel.is_file(): raise RuntimeError('SD-LoRA update receipt exists without dataset selection; refusing duplicate training')
      sx=load(sel); validate_parametric_selection(sx,ds,rr/'attempts.jsonl',r)
      if sx.get('status')!='DATASET_READY': raise RuntimeError('SD-LoRA update receipt exists for a non-ready dataset; refusing duplicate training')
      u=load(rec); manifest=load(ds/'manifest.json')
      if int(u.get('dataset_record_count',-1))!=int(manifest.get('record_count',-2)) or int(u.get('dataset_trace_count',-1))!=int(manifest.get('trace_count',-2)): raise RuntimeError('SD-LoRA recovered update dataset accounting mismatch')
      recovered={'schema':'openevo.ceiling1-stage2-carrier-terminal.v1','status':'UPDATE','kind':'parametric_sd_lora','round_index':r,'adapter_path':u.get('adapter_path'),'artifact_id':u.get('artifact_id'),'component_count_after':int(u.get('component_count_after',-1)),'update_receipt':u,'recovered_without_retraining':True}
      result=validate_sd_terminal(recovered,r,prior,self.state.get('current_sd_artifact_id'),int(self.state.get('sd_component_count',0)),self.arm,self.config_sha,self.execution_git_sha)
      if self.capacity_policy:
        if u.get('capacity_policy_sha256')!=self.capacity_policy_sha256 or int(u.get('effective_rank_limit',-1))!=self.sd_effective_rank_limit or u.get('openevo_core_execution_sha')!=self.capacity_policy['openevo_core']['execution_sha']: raise RuntimeError('Harness201 successor recovered SD-LoRA capacity/source receipt mismatch')
      atomic_json(term,recovered); return result
    ev=load(rr/'evidence/evidence.json')['gates']['parametric_sd_lora']
    component_count=int(self.state.get('sd_component_count',0)); cap=capacity_decision(component_count,self.capacity_policy)
    if not cap['allow_update']:
      x=self.noop(term,'parametric_sd_lora',cap['reason'],prior/'adapter_model.safetensors' if prior else None,r); x.update({'adapter_path':str(prior) if prior else None,'artifact_id':self.state.get('current_sd_artifact_id'),'component_count_after':component_count,'capacity_decision':cap}); atomic_json(term,x); return prior,self.state.get('current_sd_artifact_id'),component_count,False
    if not sel.is_file(): cmd(['python3',self.wt/'scripts/openevo_webshop/ceiling1_stage2_parametric_dataset_v2.py','--attempts',rr/'attempts.jsonl','--output',ds,'--round-index',r,'--max-records','256'])
    for fp in (ds/'manifest.json',ds/'records.jsonl'):
      if fp.is_file(): os.chown(fp,1001,1001); os.chmod(fp,0o600)
    sx=load(sel)
    validate_parametric_selection(sx,ds,rr/'attempts.jsonl',r)
    if sx['status']!='DATASET_READY':
      x=self.noop(term,'parametric_sd_lora',sx['status'],prior/'adapter_model.safetensors' if prior else None,r); x.update({'adapter_path':str(prior) if prior else None,'artifact_id':self.state.get('current_sd_artifact_id'),'component_count_after':int(self.state.get('sd_component_count',0)),'selection':sx}); atomic_json(term,x); return prior,self.state.get('current_sd_artifact_id'),int(self.state.get('sd_component_count',0)),False
    gen=int(self.state.get('sd_generation_count',0))
    vram=(self.root/'stage2-readiness/qwen3-1p7b/sd-lora-vram.json') if self.arm=='qwen3-1p7b' else Path('/workspace/runs/20260830-openevo-ceiling-gpu7-qualification')/('stage2-lease-3b.json' if self.arm=='3b' else 'stage2-lease-7b.json')
    env={'CEILING1_STAGE2_ARM':self.arm,'CEILING1_ANALYSIS_ROOT':str(self.root),'CEILING1_AUTOPILOT_REPO':str(self.wt),'CEILING1_STAGE2_GPU_UUID':'','CEILING1_STAGE2_ROUND':str(r),'CEILING1_STAGE2_GENERATION':str(gen),'CEILING1_STAGE2_DATASET':str(ds),'CEILING1_STAGE2_SD_ROOT':str(d),'CEILING1_STAGE2_SD_RECEIPT':str(rec),'CEILING1_STAGE2_PRIOR_SD':str(prior) if prior else '','CEILING1_STAGE2_PRIOR_SD_ID':self.state.get('current_sd_artifact_id') or '','CEILING1_STAGE2_CONFIG_SHA':self.config_sha}
    if self.capacity_policy:
      env.update({'CEILING1_OPENEVO_CORE_REPO':self.capacity_policy['openevo_core']['checkout_path'],'CEILING1_OPENEVO_CORE_GIT_SHA':self.capacity_policy['openevo_core']['execution_sha'],'CEILING1_STAGE2_CAPACITY_POLICY':str(self.capacity_policy_path),'CEILING1_STAGE2_CAPACITY_POLICY_SHA256':self.capacity_policy_sha256})
    self.launch_retry('training',r,f'stage2_sd_lora_{self.arm}',self.wt/'scripts/openevo_webshop/launch_ceiling1_stage2_sd_lora_container.sh',env,vram=vram,max_attempts=3)
    u=load(rec); validate_sd_training_update(u,int(self.state.get('sd_component_count',0)))
    if self.capacity_policy:
      if u.get('capacity_policy_sha256')!=self.capacity_policy_sha256 or int(u.get('effective_rank_limit',-1))!=self.sd_effective_rank_limit or u.get('openevo_core_execution_sha')!=self.capacity_policy['openevo_core']['execution_sha']: raise RuntimeError('Harness201 successor SD-LoRA capacity/source receipt mismatch')
    new=Path(u['adapter_path']); x={'schema':'openevo.ceiling1-stage2-carrier-terminal.v1','status':'UPDATE','kind':'parametric_sd_lora','round_index':r,'adapter_path':str(new),'artifact_id':u['artifact_id'],'component_count_after':int(u['component_count_after']),'update_receipt':u}; atomic_json(term,x); return new,u['artifact_id'],int(u['component_count_after']),True
  def task_vector_receipt(self,r:int,rr:Path,sd_term:dict,prior_sd:Path|None,new_sd:Path|None):
    p=rr/'task-vector.json'
    if p.is_file():
      cached=load(p)
      if sd_term['status']=='UPDATE':
        if new_sd is None: raise RuntimeError('updated SD-LoRA has no adapter path')
        prior_geometry=Path(self.state['last_task_vector_geometry']) if self.state.get('last_task_vector_geometry') else None
        validate_task_vector_identity(cached,int(sd_term['update_receipt']['component_count_after']),new_sd,r,int(sd_term['update_receipt']['generation_index']),self.arm,prior_geometry)
      elif cached.get('status')!='STRUCTURAL_NULL' or int(cached.get('round_index',-1))!=r or float(cached.get('delta_norm',float('nan')))!=0.0: raise RuntimeError('cached structural-null task vector identity mismatch')
      return cached
    if sd_term['status']!='UPDATE':
      prior_geom=Path(self.state['last_task_vector_geometry']) if self.state.get('last_task_vector_geometry') else None
      prior_path=float(load(prior_geom).get('path_length',0.0)) if prior_geom and prior_geom.is_file() else 0.0
      x={'schema':'openevo.ceiling1-task-vector-diagnostic.v1','round_index':r,'status':'STRUCTURAL_NULL','reason':sd_term.get('reason'),'delta_norm':0.0,'path_length':prior_path,'adjacent_update_cosine':None,'cumulative_alignment':None,'effective_rank_participation':int(self.state.get('sd_component_count',0))*8/self.sd_effective_rank_limit,'online_use':'diagnostic-only','modify_training':False,'modify_next_round':False}
      atomic_json(p,x); return x
    if new_sd is None: raise RuntimeError('updated SD-LoRA has no adapter path')
    g=rr/'task-vector-geometry.json'; args=['docker','run','--rm','--network','none','--user','1001:1001','-v',f'{self.wt}:/workspace/repo:ro','-v',f'{self.root}:{self.root}:ro','-v',f'{rr}:{rr}','--entrypoint','/usr/local/bin/python3','experiment-host/openevo-scientific:h144-runtime-v1','/workspace/repo/scripts/openevo_webshop/ceiling1_task_vector_geometry.py','--current-adapter',str(new_sd),'--output',str(g),'--round-index',str(r),'--generation-index',str(sd_term['update_receipt']['generation_index']),'--arm',self.arm]
    if prior_sd is not None:
      pg=self.state.get('last_task_vector_geometry')
      if not pg: raise RuntimeError('prior native SD-LoRA exists but prior task-vector geometry is absent')
      args += ['--prior-adapter',str(prior_sd),'--prior-geometry',str(pg)]
    expected_component_count=int(sd_term['update_receipt']['component_count_after'])
    self.ensure_writable(rr); cmd(args); x=load(g); validate_task_vector_identity(x,expected_component_count,new_sd,r,int(sd_term['update_receipt']['generation_index']),self.arm,Path(self.state['last_task_vector_geometry']) if self.state.get('last_task_vector_geometry') else None)
    x={'schema':'openevo.ceiling1-task-vector-diagnostic.v1','round_index':r,'status':'PASS',**x,'online_use':'diagnostic-only','modify_training':False,'modify_next_round':False}
    atomic_json(p,x); return x

  def retention(self,r:int,rr:Path,current_geometry:dict):
    entries=self.state.setdefault('sd_checkpoints',[])
    current_gen=int(self.state.get('sd_generation_count',0))-1
    keep_milestones={1,2,4,8,16,32,64}; keep_gens={current_gen,current_gen-1}
    pruned=[]
    for e in entries:
      if not e.get('retained_payload',True): continue
      gen=int(e['generation_index']); count=int(e['component_count'])
      if count in keep_milestones or gen in keep_gens: continue
      path=Path(e['path'])
      if not path.is_dir(): continue
      eg=Path(e['geometry_path']); gx=load(eg) if eg.is_file() else None
      if gx is None or current_geometry.get('component_factor_sha256',[])[:count] != gx.get('component_factor_sha256',[]): raise RuntimeError('retention refuses payload pruning without transitive component identity proof')
      prune_path=path/'payload-pruned.json'
      if prune_path.is_file():
        receipt=load(prune_path)
        if receipt.get('schema')!='openevo.ceiling1-stage2-checkpoint-prune.v1' or int(receipt.get('generation_index',-1))!=gen or int(receipt.get('component_count',-1))!=count:
          raise RuntimeError('retention prune receipt identity mismatch during recovery')
        removed={str(item.get('name')) for item in receipt.get('files_removed',[]) if item.get('name')}
        if not removed or any((path/name).exists() for name in removed): raise RuntimeError('retention prune receipt/payload state mismatch during recovery')
        e['retained_payload']=False; e['prune_receipt']=str(prune_path); pruned.append(receipt); continue
      files=[]
      for name in ('adapter_model.safetensors','openevo_sd_lora_state.safetensors'):
        fp=path/name
        if fp.is_file():
          files.append({'name':name,'bytes':fp.stat().st_size,'sha256':sha(fp)}); fp.unlink()
      if files:
        receipt={'schema':'openevo.ceiling1-stage2-checkpoint-prune.v1','round_index':r,'generation_index':gen,'component_count':count,'reason':'non-milestone older than latest-two recovery window; exact geometry/lineage retained and component directions proven present in successor cumulative state','files_removed':files,'recovery_checkpoint':False,'postcloseout_reconstruction':'directions preserved in successor/final cumulative state; coefficient histories preserved in task-vector geometry receipts'}
        write_once_json(prune_path,receipt); e['retained_payload']=False; e['prune_receipt']=str(prune_path); pruned.append(receipt)
    atomic_json(rr/'checkpoint-retention.json',{'schema':'openevo.ceiling1-stage2-retention.v1','round_index':r,'pruned':pruned,'milestone_component_counts':sorted(keep_milestones),'latest_two_generations':sorted(x for x in keep_gens if x>=0)})

  def resume_after_compose_before_round_seal(self,r:int,rr:Path,complete:Path)->bool:
    after_path=rr/'state-after.json'
    if not after_path.is_file(): return False
    if complete.is_file(): return False
    before=load(rr/'state-before.json'); after=load(after_path)
    if before.get('state_id')!=self.state.get('state_id'): raise RuntimeError('pending round state-before chain mismatch')
    if int(after.get('next_round',-1))!=r+1 or int(after.get('rollouts_consumed',-1))!=(r+1)*128: raise RuntimeError('pending round successor accounting mismatch')
    if after.get('rounds_complete')!=list(range(r+1)): raise RuntimeError('pending round completion index mismatch')
    if int(after.get('external_teacher_calls',-1))!=0 or int(after.get('final_panel_access_count',-1))!=0: raise RuntimeError('pending round authority boundary mismatch')
    summary=load(rr/'summary.json'); allowed=self.allowed_rollout_shas(r)
    validate_rollout_summary(summary,r,self.state['state_id']); validate_attempt_rows(rr,r,self.state['state_id'],allowed); self.validate_rollout_harness_readback(summary,rr)
    text_term=load(rr/'text-memory/terminal-receipt.json'); skill_term=load(rr/'skill-bundle/terminal-receipt.json'); agent_term=load(rr/'agent-system/terminal-receipt.json'); sdterm=load(rr/'sd-lora/terminal-receipt.json')
    for term,key in ((text_term,'memory'),(skill_term,'skill'),(agent_term,'agent_system')):
      path=Path(str(after.get(key,'')))
      if not path.is_file() or Path(str(term.get('output_path',''))).resolve()!=path.resolve() or term.get('output_sha256')!=sha(path): raise RuntimeError(f'pending round {key} payload mismatch')
    prior_sd=Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None
    new_sd,new_sd_id,new_count,sd_updated=validate_sd_terminal(sdterm,r,prior_sd,self.state.get('current_sd_artifact_id'),int(self.state.get('sd_component_count',0)),self.arm,self.config_sha,self.execution_git_sha)
    observed_sd=Path(after['current_sd_adapter']) if after.get('current_sd_adapter') else None
    if (observed_sd.resolve() if observed_sd else None)!=(new_sd.resolve() if new_sd else None) or after.get('current_sd_artifact_id')!=new_sd_id or int(after.get('sd_component_count',-1))!=new_count: raise RuntimeError('pending round SD successor mismatch')
    expected_gen=int(self.state.get('sd_generation_count',0))+(1 if sd_updated else 0)
    if int(after.get('sd_generation_count',-1))!=expected_gen: raise RuntimeError('pending round SD generation accounting mismatch')
    expected_sid=state_id(Path(after['merged_base_receipt']),new_sd,Path(after['memory']),Path(after['skill']),Path(after['agent_system']),r)
    if after.get('state_id')!=expected_sid: raise RuntimeError('pending round recomputed State digest mismatch')
    if Path(str(after.get('evidence_history',''))).resolve()!=(rr/'evidence/history-after.json').resolve(): raise RuntimeError('pending round evidence history mismatch')
    retention=load(rr/'checkpoint-retention.json')
    if retention.get('schema')!='openevo.ceiling1-stage2-retention.v1' or int(retention.get('round_index',-1))!=r: raise RuntimeError('pending round retention receipt mismatch')
    noop_diag=load(rr/'sd-lora/noop-window-diagnostic.json'); tv=load(rr/'task-vector.json')
    round_rec={'schema':'openevo.ceiling1-stage2-round-complete.v1','status':'PASS','arm':self.arm,'round_index':r,'rollout_summary':summary,'text_memory':text_term,'parametric_sd_lora':sdterm,'sd_lora_noop_monitor':noop_diag,'skill_bundle':skill_term,'agent_system':agent_term,'task_vector':tv,'state_id_before':before['state_id'],'state_id_after':expected_sid,'external_teacher_calls':0}
    write_once_json(complete,round_rec)
    self.state=after; self.save()
    self.write_round_interlock(rr,r)
    self.apply_carrier_health_boundary(r,rr)
    if noop_diag.get('warning'): print(noop_diag['warning'],flush=True)
    if noop_diag.get('pause_before_next_round'):
      self.state['status']='BLOCKED'; self.state['blocker']=f"READINESS_CIRCUIT_BREAKER_HOLD: {noop_diag['consecutive_noop_no_success']} consecutive NOOP_NO_SUCCESS through round {r}"; self.state['blocked_at_epoch']=time.time(); self.save(); raise RuntimeError(self.state['blocker'])
    return True

  def run_round(self,r:int):
    if self.execution_git_sha: assert_execution_source(self.wt,self.execution_git_sha)
    self.ensure_previous_carrier_health_boundary(r)
    rr=self.out/f'round-{r:03d}'; rr.mkdir(parents=True,exist_ok=True); complete=rr/'ROUND_COMPLETE.json'
    self.write_carrier_scope_round_binding(r,rr)
    if complete.is_file():
      if not (rr/'state-after.json').is_file(): raise RuntimeError('ROUND_COMPLETE exists without state-after; refusing partial round transaction')
      allowed=self.allowed_rollout_shas(r)
      validate_round_transaction(rr,r,self.state['state_id'],allowed)
      prior_sd=Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None
      prior_sd_sha=sha(prior_sd/'adapter_model.safetensors') if prior_sd else None
      validate_round_state_payloads(rr,prior_sd_sha)
      retention=load(rr/'checkpoint-retention.json')
      if retention.get('schema')!='openevo.ceiling1-stage2-retention.v1' or int(retention.get('round_index',-1))!=r: raise RuntimeError('completed round retention receipt mismatch')
      self.state=load(rr/'state-after.json'); self.save(); self.apply_carrier_health_boundary(r,rr); return
    if self.resume_after_compose_before_round_seal(r,rr,complete): return
    ensure_round_state_before(rr/'state-before.json',self.state); self.run_rollout(r,rr); ev=self.evidence(r,rr)
    prior_sd=Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None
    prior_mem=Path(self.state['memory']); prior_skill=Path(self.state['skill']); prior_agent=Path(self.state['agent_system'])
    new_mem=self.reflect(r,rr,'text_memory',prior_mem,sd_adapter=prior_sd)
    new_sd,new_sd_id,new_count,sd_updated=self.run_sd(r,rr)
    new_skill=self.reflect(r,rr,'skill_bundle',prior_skill,sd_adapter=prior_sd)
    new_agent=self.reflect(r,rr,'agent_system',prior_agent,memory=prior_mem,skill=prior_skill,sd_adapter=prior_sd)
    if sd_updated:
      self.state['sd_generation_count']=int(self.state.get('sd_generation_count',0))+1
    sdterm=load(rr/'sd-lora/terminal-receipt.json'); noop_diag=consecutive_noop_diagnostic(self.out,r); atomic_json(rr/'sd-lora/noop-window-diagnostic.json',noop_diag); tv=self.task_vector_receipt(r,rr,sdterm,prior_sd,new_sd)
    if sdterm['status']=='UPDATE':
      gpath=rr/'task-vector-geometry.json'; self.state['last_task_vector_geometry']=str(gpath); self.state.setdefault('sd_checkpoints',[]).append({'generation_index':int(sdterm['update_receipt']['generation_index']),'round_index':r,'component_count':new_count,'path':str(new_sd),'adapter_sha256':sdterm['update_receipt']['adapter_digest_after'],'geometry_path':str(gpath),'retained_payload':True})
    new_sid=state_id(Path(self.state['merged_base_receipt']),new_sd,new_mem,new_skill,new_agent,r)
    self.state.update({'next_round':r+1,'rollouts_consumed':(r+1)*128,'state_id':new_sid,'current_sd_adapter':str(new_sd) if new_sd else None,'current_sd_artifact_id':new_sd_id,'sd_component_count':new_count,'memory':str(new_mem),'skill':str(new_skill),'agent_system':str(new_agent),'evidence_history':str(rr/'evidence/history-after.json')})
    self.state['rounds_complete']=list(range(r+1)); self.state['external_teacher_calls']=0
    round_rec={'schema':'openevo.ceiling1-stage2-round-complete.v1','status':'PASS','arm':self.arm,'round_index':r,'rollout_summary':load(rr/'summary.json'),'text_memory':load(rr/'text-memory/terminal-receipt.json'),'parametric_sd_lora':sdterm,'sd_lora_noop_monitor':noop_diag,'skill_bundle':load(rr/'skill-bundle/terminal-receipt.json'),'agent_system':load(rr/'agent-system/terminal-receipt.json'),'task_vector':tv,'state_id_before':load(rr/'state-before.json')['state_id'],'state_id_after':new_sid,'external_teacher_calls':0}
    current_geom=load(Path(self.state['last_task_vector_geometry'])) if self.state.get('last_task_vector_geometry') else {'component_factor_sha256':[]}
    self.retention(r,rr,current_geom)
    write_once_json(rr/'state-after.json',self.state); write_once_json(complete,round_rec); self.save()
    self.write_round_interlock(rr,r)
    self.apply_carrier_health_boundary(r,rr)
    if noop_diag.get('warning'):
      print(noop_diag['warning'],flush=True)
    if noop_diag.get('pause_before_next_round'):
      self.state['status']='BLOCKED'; self.state['blocker']=f"READINESS_CIRCUIT_BREAKER_HOLD: {noop_diag['consecutive_noop_no_success']} consecutive NOOP_NO_SUCCESS through round {r}"; self.state['blocked_at_epoch']=time.time(); self.save()
      raise RuntimeError(self.state['blocker'])
  def seal(self):
    if self.execution_git_sha: assert_execution_source(self.wt,self.execution_git_sha)
    if int(self.state['next_round'])!=160 or int(self.state['rollouts_consumed'])!=20480: raise RuntimeError('cannot seal before 160x128')
    missing=[i for i in range(160) if not (self.out/f'round-{i:03d}'/'ROUND_COMPLETE.json').is_file()]
    if missing: raise RuntimeError(f'missing round receipts {missing[:5]}')
    locked_execution_shas=set()
    for lock_path in (self.out/'execution-source-locks').glob('*.json'):
      lock=load(lock_path)
      if lock.get('schema') not in {'openevo.ceiling1-stage2-execution-source-lock.v1','openevo.ceiling1-stage2-execution-source-lock.202609042025'} or lock.get('clean_worktree') is not True or not lock.get('execution_git_sha'): raise RuntimeError(f'invalid execution source lock: {lock_path}')
      locked_execution_shas.add(lock['execution_git_sha'])
    if self.a.recovery_amendment: locked_execution_shas.add('51ce92498a4fce306d502fa3a42c67b2ae6bf5ef')
    if not locked_execution_shas: raise RuntimeError('no execution source locks available for seal')
    total=0; invalid=0; execution_shas=set(); expected_state_id=load(self.out/'round-000/state-before.json')['state_id']; prior_sd_sha=None
    updates={k:0 for k in ('text_memory','parametric_sd_lora','skill_bundle','agent_system')}; noops=dict(updates)
    for i in range(160):
      round_root=self.out/f'round-{i:03d}'; expected_state_id,round_shas=validate_round_transaction(round_root,i,expected_state_id,locked_execution_shas); execution_shas.update(round_shas); prior_sd_sha=validate_round_state_payloads(round_root,prior_sd_sha)
      if self.carrier_scope_package is not None: self.validate_carrier_scope_round_binding(i,round_root)
      x=load(self.out/f'round-{i:03d}'/'ROUND_COMPLETE.json'); total+=int(x['rollout_summary']['attempt_count']); invalid+=int(x['rollout_summary']['engineering_invalid_count'])
      for k in updates:
        if x[k]['status']=='UPDATE': updates[k]+=1
        else:noops[k]+=1
    if total!=20480 or invalid!=0: raise RuntimeError('Stage2 exact denominator seal failed')
    if expected_state_id!=self.state['state_id']: raise RuntimeError('final execution state does not match round chain')
    final={'schema':'openevo.ceiling1-stage2-sealed.v1','status':'PASS','arm':self.arm,'rounds':160,'rollouts':total,'engineering_invalid':invalid,'external_teacher_calls':0,'updates':updates,'noops':noops,'final_state_id':self.state['state_id'],'final_sd_adapter':self.state.get('current_sd_adapter'),'final_memory':self.state['memory'],'final_skill':self.state['skill'],'final_agent_system':self.state['agent_system'],'config_sha256':self.config_sha,'execution_git_shas':sorted(execution_shas),'output_root':str(self.out),'final_panel_access_count':0}
    if self.recovery: final['recovery_accounting']=self.recovery['accounting']
    atomic_json(self.root/f'STAGE2_SEALED_{self.arm}.json',final); self.state['status']='STAGE2_SEALED'; self.save(); return final
  def run(self):
    try:
      if self.a.recovery_amendment and int(self.state['next_round'])>0: self.validate_recovery()
      while int(self.state['next_round'])<160:
        r=int(self.state['next_round']); self.run_round(r)
        if self.a.recovery_amendment and r==0: self.validate_recovery()
      final=self.seal(); print(json.dumps(final,sort_keys=True)); return 0
    except BaseException as exc:
      self.state['status']='BLOCKED'; self.state['blocker']=f'{type(exc).__name__}: {exc}'; self.state['blocked_at_epoch']=time.time(); self.save(); raise
  def validate_recovery(self):
    schema=self.recovery.get('schema') if self.recovery else None
    if schema=='openevo.ceiling1-stage2-agent-contract-recovery-amendment.v1': script='validate_ceiling1_stage2_agent_recovery.py'
    elif schema=='openevo.ceiling1-stage2-carrier-distinctness-recovery-amendment.v1': script='validate_ceiling1_stage2_carrier_recovery.py'
    else: script='validate_ceiling1_stage2_recovery.py'
    cmd(['python3',self.wt/'scripts/openevo_webshop'/script,'--output',self.out,'--require-corrected-state'])

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--arm',choices=('3b','7b','qwen3-1p7b'),required=True); p.add_argument('--repo',type=Path,required=True); p.add_argument('--analysis-root',type=Path,required=True); p.add_argument('--seed-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--stage2-activation',type=Path,required=True); p.add_argument('--recovery-amendment',type=Path); p.add_argument('--capacity-policy',type=Path); p.add_argument('--harness-profile',choices=('legacy','2.0.1','201.1'),default='legacy'); p.add_argument('--step-seed-policy',choices=('legacy','injective-v1'),default='legacy'); p.add_argument('--step-seed-amendment',type=Path); p.add_argument('--harness201-carrier-root',type=Path); p.add_argument('--carrier-scope-package',type=Path); p.add_argument('--carrier-health-policy',type=Path); p.add_argument('--carrier-contract-identities',type=Path); p.add_argument('--carrier-adoption-receipt',type=Path); p.add_argument('--carrier-health-resolution',type=Path); p.add_argument('--execution-git-sha'); a=p.parse_args(); return Controller(a).run()
if __name__=='__main__': raise SystemExit(main())
