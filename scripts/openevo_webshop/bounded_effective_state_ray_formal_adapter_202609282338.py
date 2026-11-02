#!/usr/bin/env python3
"""Formal Bounded OFF/ON adapter for the generic Ray capacity backend.

Science controllers decide WHAT is ready. This module decides only WHERE a ready
GPU workload fits, using frozen workload profiles, Ray logical reservations, and
live physical VRAM/UUID gates. It never changes treatment/round/task semantics.
"""
from __future__ import annotations
import json, os, subprocess
from pathlib import Path
from scripts.openevo_webshop import ceiling1_ray_placement_backend_202609042025 as ray_backend

SCHEMA='openevo.bounded-gdr.ray-capacity-formal-successor.20260916'

class RayCapacityUnavailable(RuntimeError):
    pass

def sha(path:Path)->str: return ray_backend.sha(path)
def load(path:Path)->dict: return json.loads(path.read_text())

def load_policy(repo:Path, package:dict)->dict|None:
    ref=package.get('ray_capacity_resource_successor')
    if not ref: return None
    path=repo/str(ref.get('path') or '')
    if not path.is_file() or ref.get('sha256')!=sha(path): raise RuntimeError('Ray capacity formal successor binding drift')
    p=load(path)
    if p.get('schema')!=SCHEMA or p.get('status')!='AUTHORIZED_FORMAL_RESOURCE_SUCCESSOR': raise RuntimeError('Ray capacity formal successor not authorized')
    if p.get('scientific_semantics_changed') is not False or p.get('final_panel_access_allowed') is not False: raise RuntimeError('Ray capacity formal successor science/protected boundary drift')
    for key in ('base_config','profile_registry','logical_qualification','cuda_qualification','capacity_contract'):
        r=p.get(key) or {}; fp=Path(r.get('path') or '')
        if not fp.is_absolute(): fp=repo/fp
        if not fp.is_file() or r.get('sha256')!=sha(fp): raise RuntimeError(f'Ray capacity formal successor {key} drift')
    managed=list(map(int,p.get('managed_gpus') or []))
    if len(managed)!=4 or len(set(managed))!=4 or int(p.get('project_budget_mib',-1))!=28000: raise RuntimeError('Ray capacity formal successor GPU/budget drift')
    cfg_path=Path(p['base_config']['path']); cfg_path=cfg_path if cfg_path.is_absolute() else repo/cfg_path
    contract_path=Path(p['capacity_contract']['path']); contract_path=contract_path if contract_path.is_absolute() else repo/contract_path
    if list(map(int,load(cfg_path).get('managed_gpus') or []))!=managed: raise RuntimeError('Ray capacity formal successor/base config GPU drift')
    if list(map(int,load(contract_path).get('managed_gpus') or []))!=managed: raise RuntimeError('Ray capacity formal successor/contract GPU drift')
    return {**p,'_path':str(path),'_sha256':sha(path)}

def arm_key(experiment_arm_id:str)->str:
    if experiment_arm_id=='BOUNDED_OFF': return 'off'
    if experiment_arm_id=='BOUNDED_EFFECTIVE_STATE_ON': return 'on'
    raise RuntimeError('unknown Bounded arm for Ray capacity')

def channel_for(kind:str, active_reflector_kind:str|None=None)->str:
    if kind=='rollout': return 'rollout'
    if kind in {'transition','training'}: return 'sd_lora'
    if kind=='reflector' and active_reflector_kind in {'text_memory','skill_bundle','agent_system'}: return str(active_reflector_kind)
    raise RuntimeError(f'unsupported Bounded Ray channel kind={kind} reflector={active_reflector_kind}')

def _paths(repo:Path, policy:dict)->tuple[Path,Path,Path]:
    cfg=Path(policy['base_config']['path']); reg=Path(policy['profile_registry']['path'])
    if not cfg.is_absolute(): cfg=repo/cfg
    if not reg.is_absolute(): reg=repo/reg
    return cfg,reg,Path(policy['state_root'])

def _selection_gate(repo:Path, policy:dict, *, experiment_arm_id:str, kind:str, round_index:int, output:Path, active_reflector_kind:str|None=None, count:int|None=None)->dict:
    cfg,reg,_=_paths(repo,policy); akey=arm_key(experiment_arm_id); channel=channel_for(kind,active_reflector_kind)
    contract=load(repo/policy['capacity_contract']['path']); workload=contract['workload_map'][akey][channel]; doc=load(output)
    if doc.get('schema')!='openevo.ceiling1-ray-selection.202609042025' or doc.get('workload')!=workload: raise RuntimeError('Ray capacity selection identity drift')
    if doc.get('config_sha256')!=sha(cfg) or doc.get('profile_registry_sha256')!=sha(reg): raise RuntimeError('Ray capacity selection config/profile drift')
    selected=list(map(int,doc.get('selected_gpus') or []))
    if count is not None and len(selected)!=int(count): raise RuntimeError('Ray capacity selection GPU-count drift')
    if not selected or any(i not in policy['managed_gpus'] for i in selected): raise RuntimeError('Ray capacity selection GPU set drift')
    return {'schema':'openevo.bounded-gdr.ray-capacity-selection-gate.20260916','status':'PASS','policy_sha256':policy['_sha256'],'workload':workload,'channel':channel,'selection_path':str(output),'selection_sha256':sha(output),'selected_gpus':selected,'selected_gpu_uuids':doc['selected_gpu_uuids'],'vram_request_mib_per_gpu':doc['vram_request_mib_per_gpu'],'live_gate':doc['live_gate'],'reservation_id':doc['reservation_id'],'placement_group_name':doc['placement_group_name'],'round_index':round_index,'final_panel_access':0}

def select(repo:Path, policy:dict, *, experiment_arm_id:str, kind:str, round_index:int, output:Path, active_reflector_kind:str|None=None, count:int|None=None)->dict:
    if output.is_file(): return _selection_gate(repo,policy,experiment_arm_id=experiment_arm_id,kind=kind,round_index=round_index,output=output,active_reflector_kind=active_reflector_kind,count=count)
    cfg,reg,root=_paths(repo,policy); akey=arm_key(experiment_arm_id); channel=channel_for(kind,active_reflector_kind)
    contract=load(repo/policy['capacity_contract']['path']); workload=contract['workload_map'][akey][channel]
    cmd=['python3',str(repo/'scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py'),'--config',str(cfg),'--state-root',str(root),'--profile-registry',str(reg),'select','--workload',workload]
    if count is not None: cmd += ['--count',str(count)]
    cmd += ['--output',str(output)]
    q=subprocess.run(cmd,text=True,capture_output=True)
    if q.returncode!=0:
        message=q.stderr.strip() or q.stdout.strip() or 'Ray capacity select failed'
        if 'RAY_NO_FEASIBLE_PLACEMENT' in message: raise RayCapacityUnavailable(message)
        raise RuntimeError(message)
    return _selection_gate(repo,policy,experiment_arm_id=experiment_arm_id,kind=kind,round_index=round_index,output=output,active_reflector_kind=active_reflector_kind,count=count)

def launch(repo:Path, policy:dict, gate:dict, *, lease_id:str, container:str, launcher:Path, env:dict)->subprocess.CompletedProcess[str]:
    cfg,reg,root=_paths(repo,policy)
    cmd=['python3',str(repo/'scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py'),'--config',str(cfg),'--state-root',str(root),'--profile-registry',str(reg),'run','--lease-id',lease_id,'--workload',gate['workload'],'--container',container,'--launch-script',str(launcher),'--gpus',*map(str,gate['selected_gpus']),'--selection-receipt',gate['selection_path']]
    return subprocess.run(cmd,env={**os.environ,**env},text=True,capture_output=True)

def reconcile(repo:Path,policy:dict)->None:
    cfg,reg,root=_paths(repo,policy)
    subprocess.run(['python3',str(repo/'scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py'),'--config',str(cfg),'--state-root',str(root),'--profile-registry',str(reg),'reconcile'],text=True,capture_output=True)
