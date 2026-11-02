#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
from typing import Any
ARM='3b'; READY_ARM='qwen25-3b'; MODEL='Qwen/Qwen2.5-3B-Instruct'; REV='aa8e72537993ba99e69dfaafa59ed015b17504d1'; CONTEXT=32768
PROMPT_SHA='8efcba565fc11e485192f6ca85cb62119f7322662cff1a5e6597de0608d441b5'
SAMPLING={'do_sample':True,'temperature':1.0,'top_p':1.0,'top_k':0,'repetition_penalty':1.0,'max_new_tokens':512,'max_steps':15}
RUNTIME_FILES={
 'scripts/openevo_webshop/ceiling1_stage2_format_hard_runtime.py','scripts/openevo_webshop/ceiling1_stage2_rollout_worker.py',
 'scripts/openevo_webshop/ceiling1_stage2_rollout_worker_format_hard.py','scripts/openevo_webshop/ceiling1_stage2_round_driver.py',
 'scripts/openevo_webshop/ceiling1_harness2011_runtime.py','scripts/openevo_webshop/ceiling1_harness2011_prompt.py',
 'scripts/openevo_webshop/launch_ceiling1_stage2_round_container.sh','configs/experiment/manifests/h1.44-on-3b-training-schedule-v1.json',
 'configs/experiment/openevo-ceiling-1.0-stage2-final-v1.json'}
EXECUTABLES={'scripts/openevo_webshop/launch_ceiling1_stage2_round_container.sh','scripts/openevo_webshop/launch_ceiling1_stage2_reflector_container.sh','scripts/openevo_webshop/launch_ceiling1_stage2_sd_lora_container.sh','scripts/openevo_webshop/launch_ceiling1_stage2_vnext_probe_container.sh','scripts/openevo_webshop/launch_ceiling1_stage2_vnext_current_arm.sh'}
def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p:Path)->dict[str,Any]:return json.loads(p.read_text())
def git(repo:Path,*args:str)->str:return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
def blob(repo:Path,ref:str,path:str)->bytes:return subprocess.check_output(['git','-C',str(repo),'show',f'{ref}:{path}'])
def mode(repo:Path,ref:str,path:str)->str:
 x=subprocess.check_output(['git','-C',str(repo),'ls-tree',ref,path],text=True).strip()
 if not x:raise RuntimeError(f'missing Git path {path}')
 return x.split()[0]
def clean(repo:Path,ref:str):
 if git(repo,'rev-parse','HEAD')!=ref or git(repo,'status','--porcelain'):raise RuntimeError('execution checkout drift')
def rows(p:Path):
 with p.open() as f:return [json.loads(x) for x in f if x.strip()]
def create(p:Path,x:dict):
 p.parent.mkdir(parents=True,exist_ok=True);r=json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
 if p.exists():
  if p.read_text()!=r:raise RuntimeError('immutable activation identity conflict')
 else:p.write_text(r)
def main()->int:
 q=argparse.ArgumentParser();q.add_argument('--repo',type=Path,required=True);q.add_argument('--governance-repo',type=Path,required=True);q.add_argument('--governance-sha',required=True);q.add_argument('--execution-sha',required=True);q.add_argument('--qualification-sha',required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--carrier-root',type=Path,required=True);q.add_argument('--format-smoke',type=Path,required=True);q.add_argument('--initial-probe',type=Path,required=True);q.add_argument('--sd-vram',type=Path,required=True);q.add_argument('--output-basename',required=True);q.add_argument('--output',type=Path,required=True);a=q.parse_args()
 repo=a.repo.resolve();gov=a.governance_repo.resolve();root=a.root.resolve();carrier=a.carrier_root.resolve();smoke=a.format_smoke.resolve();clean(repo,a.execution_sha);clean(gov,a.governance_sha)
 diff={x for x in git(gov,'diff','--name-only',f'{a.execution_sha}..{a.governance_sha}').splitlines() if x}
 if diff!={'scripts/openevo_webshop/prepare_3b_stage2_vnext_activation.py'}:raise RuntimeError(f'3B activation governance scope drift: {sorted(diff)}')
 design_p=repo/'configs/experiment/designs/openevo-ceiling1-stage2-vnext-3b-v1.json';cap_p=repo/'configs/experiment/designs/openevo-ceiling-1.0-harness201-3b-successor-capacity-v1.json';seed_p=repo/'configs/experiment/designs/openevo-ceiling-1.0-harness201-step-seed-amendment-v1.json';protocol_p=repo/'configs/experiment/openevo-ceiling-1.0-stage2-final-v1.json';lease_p=repo/'configs/experiment/openevo-ceiling-1.0-gpu-lease-qwen3-stage2-v1.json'
 design=load(design_p);f=design['frozen_protocol']
 if design.get('arm')!='3b' or design.get('status')!='FROZEN_DESIGN_NO_ACTIVATION_AUTHORITY' or design.get('activation_authority') is not False:raise RuntimeError('3B design drift')
 if int(f['formal_rollouts'])!=20480 or int(f['rounds'])!=160 or int(f['rollouts_per_round'])!=128 or f['step_seed_policy']!='injective-v1':raise RuntimeError('3B Stage2 budget/seed drift')
 ready_p=root/'PRE_STAGE2_READY_qwen25-3b.json';ready=load(ready_p);m=ready['model']
 if ready.get('status')!='PASS' or ready.get('arm')!=READY_ARM or ready.get('stage2_authorized') is not False:raise RuntimeError('3B PRE_STAGE2_READY drift')
 if m.get('model_id')!=MODEL or m.get('revision')!=REV or int(m.get('context_window',-1))!=CONTEXT:raise RuntimeError('3B model identity drift')
 compat_p=root/'stage2-readiness/3b/STAGE1_PRESTAGE2_COMPAT.json';compat=load(compat_p);adapter_sha=ready['initial_stage2_adapter']['sha256']
 if compat.get('status')!='PASS' or compat.get('arm')!='3b' or compat.get('source_ready_sha256')!=sha(ready_p) or compat['opsd']['adapter_sha256']!=adapter_sha:raise RuntimeError('3B entry compatibility drift')
 base_p=root/'stage2-base/3b/verification-receipt.json';base=load(base_p)
 if base.get('status')!='PASS' or base.get('arm')!='3b' or base.get('opsd_adapter_sha256')!=adapter_sha:raise RuntimeError('3B merged base drift')
 cr_p=carrier/'finalize-receipt.json';cr=load(cr_p)
 if cr.get('status')!='PASS' or cr.get('schema')!='openevo.ceiling1-harness2.0.1-carrier-finalize-receipt.v2' or any(int(cr.get(k,-1))!=0 for k in ('model_calls','external_teacher_calls','formal_task_consumption','final_panel_access')):raise RuntimeError('3B safe carrier drift')
 sr=rows(smoke/'attempts.jsonl');
 if len(sr)!=16 or any(r.get('experiment_git_sha')!=a.qualification_sha or r.get('harness_profile')!='201.1' or r.get('step_seed_policy')!='injective-v1' or r.get('engineering_invalid') or r.get('parser_or_fallback_failure') or r.get('terminated_on_invalid_action') for r in sr):raise RuntimeError('3B format smoke drift')
 steps=[s for r in sr for s in r['episode']['trace']]
 if not steps or any(not (s.get('action_channel') or {}).get('interface_owned_action_opener') or not (s.get('action_channel') or {}).get('admissible') or int((s.get('action_channel') or {}).get('observed_pre_action_model_tokens',-1))<4 for s in steps):raise RuntimeError('3B format envelope drift')
 parity={}
 for path in sorted(RUNTIME_FILES):
  a0=blob(repo,a.qualification_sha,path);b0=blob(repo,a.execution_sha,path);same=a0==b0;parity[path]={'qualification_sha256':hashlib.sha256(a0).hexdigest(),'formal_sha256':hashlib.sha256(b0).hexdigest(),'identical':same}
  if not same:raise RuntimeError(f'3B runtime changed after qualification: {path}')
 modes={p:mode(repo,a.execution_sha,p) for p in sorted(EXECUTABLES)}
 if any(v!='100755' for v in modes.values()):raise RuntimeError(f'3B launcher mode drift {modes}')
 probe_p=a.initial_probe.resolve();probe=load(probe_p)
 if probe.get('status')!='PASS' or probe.get('arm')!='3b' or int(probe.get('attempt_count',-1))!=16 or probe.get('prompt_contract_sha256')!=PROMPT_SHA or int(probe.get('invalid_terminated',-1))!=0 or int(probe.get('constraint_escape_count',-1))!=0 or int(probe.get('environment_action_schema_violation_count',-1))!=0:raise RuntimeError('3B initial probe drift')
 ps=[s for r in probe.get('rows',[]) for s in r['episode']['trace']]
 if not ps or any(not (s.get('action_channel') or {}).get('interface_owned_action_opener') or not (s.get('action_channel') or {}).get('admissible') for s in ps):raise RuntimeError('3B probe action envelope drift')
 vram_p=a.sd_vram.resolve();v=load(vram_p)
 if v.get('status')!='pass' or v.get('workload')!='stage2_sd_lora_3b' or v.get('science_execution_git_sha')!=a.execution_sha or int(v.get('safety_headroom_mib',0))<4096:raise RuntimeError('3B SD VRAM drift')
 cap=load(cap_p);core=Path(cap['openevo_core']['checkout_path']);clean(core,cap['openevo_core']['execution_sha'])
 if int(cap['effective_rank_limit'])!=4096 or int(cap['rank'])!=8 or int(cap['replay_capacity'])!=64:raise RuntimeError('3B capacity drift')
 lease_sha=sha(lease_p);state_p=Path('/workspace/control/openevo-ceiling1-gpu-lease-state-q17-stage2-v1/state.json');state=load(state_p)
 if state.get('config_sha256')!=lease_sha:raise RuntimeError('shared small-arm lease state drift')
 tm_p=root/'TEXT_MEMORY_COMPLETE_qwen25-3b.json';sk_p=root/'SKILL_BUNDLE_COMPLETE_qwen25-3b.json';ag_p=root/'AGENT_SYSTEM_COMPLETE_qwen25-3b.json';tm,sk,ag=load(tm_p),load(sk_p),load(ag_p)
 for n,x in [('text',tm),('skill',sk),('agent',ag)]:
  if x.get('status')!='pass' or x.get('scientific_result') is not True or type(x.get('split_events')) is not int:raise RuntimeError(f'3B carrier marker drift {n}')
 act={'schema':'openevo.ceiling1-stage2-activation.v1','status':'authorized','stage2_config_sha256':sha(protocol_p),'authorized_text_memory_schema':tm['schema'],'authorized_text_memory_split_events':tm['split_events'],'authorized_skill_bundle_schema':sk['schema'],'authorized_skill_bundle_split_events':sk['split_events'],'authorized_agent_system_schema':ag['schema'],'authorized_agent_system_split_events':ag['split_events'],'authorized_arms':['3b'],'authorized_output_basename':a.output_basename,'vnext_design_sha256':sha(design_p),'vnext_execution_git_sha':a.execution_sha,'vnext_initial_probe_path':str(probe_p),'vnext_initial_probe_sha256':sha(probe_p),'small_arm_resource_config_sha256':lease_sha,'per_arm':{'3b':{'authorized':True,'stage1_complete_path':str(compat_p),'stage1_complete_sha256':sha(compat_p),'text_memory_complete_path':str(tm_p),'text_memory_complete_sha256':sha(tm_p),'skill_bundle_complete_path':str(sk_p),'skill_bundle_complete_sha256':sha(sk_p),'agent_system_complete_path':str(ag_p),'agent_system_complete_sha256':sha(ag_p)}},'formal_stage2_rollouts_consumed_at_activation':0,'final_panel_access_count_at_activation':0,'stage2_per_arm_independent':True,'final_panel_locked_until_stage2_sealed':True,'qualification_execution_git_sha':a.qualification_sha,'activation_governance_source_sha':a.governance_sha,'activation_governance_script_sha256':sha(Path(__file__).resolve()),'qualification_to_formal_runtime_parity':parity,'formal_launcher_modes':modes,'evidence':{'prestage2_ready':{'path':str(ready_p),'sha256':sha(ready_p)},'entry_compat':{'path':str(compat_p),'sha256':sha(compat_p)},'base':{'path':str(base_p),'sha256':sha(base_p)},'safe_carrier':{'path':str(cr_p),'sha256':sha(cr_p)},'format_smoke':{'path':str(smoke/'summary.json'),'sha256':sha(smoke/'summary.json')},'initial_probe':{'path':str(probe_p),'sha256':sha(probe_p)},'sd_vram':{'path':str(vram_p),'sha256':sha(vram_p)},'capacity':{'path':str(cap_p),'sha256':sha(cap_p)},'step_seed':{'path':str(seed_p),'sha256':sha(seed_p)},'lease':{'path':str(lease_p),'sha256':lease_sha}}}
 create(a.output.resolve(),act);print(json.dumps(act,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
