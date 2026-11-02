#!/usr/bin/env python3
"""Build immutable Qwen3-1.7B Stage2 activation after all readiness gates.

The scientific execution source may be older than this governance helper. We
permit that only when the intervening Git diff contains no rollout/runtime/
training algorithm changes. This tool itself consumes zero scientific budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ARM="qwen3-1p7b"
MODEL_ID="Qwen/Qwen3-1.7B"
REV="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CONTEXT=40960
SAMPLING={"do_sample":True,"temperature":1.0,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"max_new_tokens":512,"max_steps":15}
REQUIRED_WORKLOADS={
 "stage2_rollout_qwen3-1p7b","stage2_text_memory_qwen3-1p7b","stage2_skill_bundle_qwen3-1p7b",
 "stage2_agent_system_qwen3-1p7b","stage2_sd_lora_qwen3-1p7b",
}
ALLOWED_POST_SCIENCE_DIFF={
 "configs/experiment/designs/openevo-ceiling1-stage2-vnext-qwen3-1p7b-capacity-v1.json",
 "scripts/openevo_webshop/ceiling1_harness201_generic_safety_finalize.py",
 "tests/test_harness201_generic_safety_finalize.py",
 "scripts/openevo_webshop/prepare_qwen3_stage2_activation.py",
 "scripts/openevo_webshop/prepare_qwen3_stage2_activation_v2.py",
 "scripts/openevo_webshop/launch_qwen3_stage2_current.sh",
}

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load(path:Path)->dict[str,Any]: return json.loads(path.read_text())

def git(repo:Path,*args:str)->str:
 return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()

def assert_clean_exact(repo:Path,expected:str)->None:
 head=git(repo,'rev-parse','HEAD'); dirty=git(repo,'status','--porcelain')
 if head!=expected or dirty: raise RuntimeError(f'Git source drift expected={expected} observed={head} dirty={bool(dirty)}')

def read_jsonl(path:Path)->list[dict[str,Any]]:
 with path.open() as f: return [json.loads(line) for line in f if line.strip()]

def atomic_create(path:Path,value:dict[str,Any])->None:
 path.parent.mkdir(parents=True,exist_ok=True); rendered=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
 try:
  with path.open('x') as f: f.write(rendered)
 except FileExistsError:
  if path.read_text()!=rendered: raise RuntimeError(f'immutable activation identity conflict: {path}')

def main()->int:
 p=argparse.ArgumentParser()
 p.add_argument('--governance-repo',type=Path,required=True)
 p.add_argument('--governance-git-sha',required=True)
 p.add_argument('--science-repo',type=Path,required=True)
 p.add_argument('--science-execution-sha',required=True)
 p.add_argument('--root',type=Path,required=True)
 p.add_argument('--carrier-root',type=Path,required=True)
 p.add_argument('--round0-smoke',type=Path,required=True)
 p.add_argument('--sd-vram',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 gov=a.governance_repo.resolve(); sci=a.science_repo.resolve(); root=a.root.resolve(); carrier=a.carrier_root.resolve(); smoke=a.round0_smoke.resolve()
 assert_clean_exact(gov,a.governance_git_sha); assert_clean_exact(sci,a.science_execution_sha)
 diff={x for x in git(gov,'diff','--name-only',f'{a.science_execution_sha}..{a.governance_git_sha}').splitlines() if x}
 unexpected=sorted(diff-ALLOWED_POST_SCIENCE_DIFF)
 if unexpected: raise RuntimeError(f'post-qualification governance source changed science-relevant files: {unexpected}')

 design_path=gov/'configs/experiment/designs/openevo-ceiling1-stage2-vnext-qwen3-1p7b-v1.json'
 protocol_path=gov/'configs/experiment/openevo-ceiling-1.0-stage2-final-v1.json'
 capacity_path=gov/'configs/experiment/designs/openevo-ceiling1-stage2-vnext-qwen3-1p7b-capacity-v1.json'
 seed_path=gov/'configs/experiment/designs/openevo-ceiling-1.0-harness201-step-seed-amendment-v1.json'
 lease_path=sci/'configs/experiment/openevo-ceiling-1.0-gpu-lease-qwen3-stage2-v1.json'
 for x in (design_path,protocol_path,capacity_path,seed_path,lease_path):
  if not x.is_file(): raise RuntimeError(f'missing authority input {x}')

 design=load(design_path); model=design.get('model') or {}; frozen=design.get('frozen_protocol') or {}
 if design.get('status')!='FROZEN_PREREG_NEEDS_LAUNCH_READINESS' or design.get('activation_authority') is not False or design.get('arm')!=ARM: raise RuntimeError('design identity drift')
 expected_model={'id':MODEL_ID,'revision':REV,'path':'/workspace/models/qwen3-1.7b','context_window':CONTEXT,'chat_template_enable_thinking':False}
 if model!=expected_model: raise RuntimeError('model identity drift')
 if int(frozen.get('formal_rollouts',-1))!=20480 or int(frozen.get('rounds',-1))!=160 or int(frozen.get('rollouts_per_round',-1))!=128: raise RuntimeError('budget drift')
 if frozen.get('sampling')!=SAMPLING or frozen.get('step_seed_policy')!='injective-v1': raise RuntimeError('sampling/step-seed drift')
 if int(frozen.get('external_teacher_calls',-1))!=0 or int(frozen.get('final_panel_access_before_own_stage2_seal',-1))!=0: raise RuntimeError('forbidden budget drift')

 ready_path=root/'PRE_STAGE2_READY_qwen3-1p7b.json'; ready=load(ready_path)
 if ready.get('status')!='PASS' or ready.get('arm')!=ARM or ready.get('stage2_authorized') is not False: raise RuntimeError('PRE_STAGE2_READY drift')
 if (ready.get('model') or {}).get('chat_template_enable_thinking') is not False or int((ready.get('model') or {}).get('context_window',-1))!=CONTEXT: raise RuntimeError('prestage2 model readback drift')
 compat_path=root/'stage2-readiness/qwen3-1p7b/STAGE1_PRESTAGE2_COMPAT.json'; compat=load(compat_path)
 if compat.get('status')!='PASS' or compat.get('arm')!=ARM or compat.get('source_ready_sha256')!=sha(ready_path): raise RuntimeError('entry compatibility drift')
 adapter_sha=(ready.get('initial_stage2_adapter') or {}).get('sha256')
 if (compat.get('opsd') or {}).get('adapter_sha256')!=adapter_sha: raise RuntimeError('entry OPSD identity drift')

 base_path=root/'stage2-base/qwen3-1p7b/verification-receipt.json'; base=load(base_path)
 if base.get('status')!='PASS' or base.get('arm')!=ARM or base.get('opsd_adapter_sha256')!=adapter_sha: raise RuntimeError('merged base drift')
 if int(base.get('formal_task_consumption',-1))!=0 or int(base.get('final_panel_consumption',-1))!=0: raise RuntimeError('merged base budget drift')

 preflight_path=root/'stage2-preflight/qwen3-1p7b/runtime-preflight.json'; pre=load(preflight_path)
 if pre.get('status')!='PASS' or pre.get('harness_profile')!='201.1' or pre.get('chat_template_enable_thinking') is not False or int(pre.get('context_window',-1))!=CONTEXT: raise RuntimeError('initial runtime preflight drift')
 if int(pre.get('formal_webshop_calls',-1))!=0 or int(pre.get('final_panel_consumption',-1))!=0: raise RuntimeError('preflight budget drift')

 carrier_receipt=carrier/'finalize-receipt.json'; cr=load(carrier_receipt)
 if cr.get('schema')!='openevo.ceiling1-harness2.0.1-carrier-finalize-receipt.v2' or cr.get('status')!='PASS': raise RuntimeError('Harness201 carrier seed drift')
 if any(int(cr.get(k,-1))!=0 for k in ('model_calls','external_teacher_calls','formal_task_consumption','final_panel_access')): raise RuntimeError('carrier seed budget drift')

 summary_path=smoke/'summary.json'; summary=load(summary_path)
 if summary.get('status')!='PASS' or summary.get('qualification_only') is not True: raise RuntimeError('round0 qualification not PASS')
 if int(summary.get('attempt_count',-1))!=128 or int(summary.get('task_identity_count',-1))!=16 or int(summary.get('engineering_invalid_count',-1))!=0: raise RuntimeError('round0 denominator/engineering drift')
 if summary.get('harness_profile')!='201.1' or summary.get('step_seed_policy')!='injective-v1' or summary.get('sampling')!=SAMPLING: raise RuntimeError('round0 protocol drift')
 rows=read_jsonl(smoke/'attempts.jsonl')
 if len(rows)!=128: raise RuntimeError('round0 row count drift')
 if any(r.get('experiment_git_sha')!=a.science_execution_sha for r in rows): raise RuntimeError('round0 execution SHA drift')
 if any(r.get('harness_profile')!='201.1' or r.get('step_seed_policy')!='injective-v1' for r in rows): raise RuntimeError('round0 row protocol drift')
 if any(bool(r.get('engineering_invalid')) or bool(r.get('parser_or_fallback_failure')) or bool(r.get('terminated_on_invalid_action')) for r in rows): raise RuntimeError('round0 mechanical validity drift')
 steps=[st for r in rows for st in (r.get('episode') or {}).get('trace',[])]
 if not steps: raise RuntimeError('round0 contains no runtime steps')
 if any(st.get('runtime_harness')!='201.1' for st in steps): raise RuntimeError('round0 runtime harness drift')
 if any(((st.get('runtime_context_receipt') or {}).get('chat_template_enable_thinking') is not False) for st in steps): raise RuntimeError('round0 Qwen3 thinking readback drift')
 if any(not bool((st.get('action_channel') or {}).get('interface_owned_action_opener')) for st in steps): raise RuntimeError('round0 interface-owned action opener drift')
 if any(not bool((st.get('action_channel') or {}).get('admissible')) for st in steps): raise RuntimeError('round0 action admissibility drift')
 if any(bool((st.get('action_channel') or {}).get('unconstrained_fallback')) or bool((st.get('action_channel') or {}).get('fuzzy_repair_used')) for st in steps): raise RuntimeError('round0 action fallback/repair drift')
 if any(int((st.get('action_channel') or {}).get('observed_pre_action_model_tokens',-1))<4 for st in steps): raise RuntimeError('round0 minimum deliberation readback drift')

 vram_path=a.sd_vram.resolve(); vram=load(vram_path)
 if vram.get('status')!='pass' or vram.get('workload')!='stage2_sd_lora_qwen3-1p7b': raise RuntimeError('SD-LoRA VRAM qualification drift')
 if int(vram.get('safety_headroom_mib',0))<4096 or int(vram.get('peak_memory_mib',0))<=0: raise RuntimeError('SD-LoRA VRAM qualification insufficient')

 capacity=load(capacity_path); core=Path((capacity.get('openevo_core') or {}).get('checkout_path','')); core_sha=(capacity.get('openevo_core') or {}).get('execution_sha')
 if not core.is_dir() or not core_sha: raise RuntimeError('capacity Core binding missing')
 assert_clean_exact(core,core_sha)
 if int(capacity.get('effective_rank_limit',-1))!=4096 or int(capacity.get('rank',-1))!=8 or int(capacity.get('replay_capacity',-1))!=64: raise RuntimeError('capacity algorithm drift')

 lease=load(lease_path); workloads=set((lease.get('workloads') or {}).keys())
 if not REQUIRED_WORKLOADS<=workloads: raise RuntimeError('Qwen3 lease workloads incomplete')
 rollout=lease['workloads']['stage2_rollout_qwen3-1p7b']
 if rollout.get('candidate_gpus')!=[4,5,6,7] or int(rollout.get('target_gpus',-1))!=4 or int(rollout.get('min_gpus',-1))!=4: raise RuntimeError('formal rollout topology drift')

 tm_path=root/'TEXT_MEMORY_COMPLETE_qwen3-1p7b.json'; sk_path=root/'SKILL_BUNDLE_COMPLETE_qwen3-1p7b.json'; ag_path=root/'AGENT_SYSTEM_COMPLETE_qwen3-1p7b.json'
 tm,sk,ag=load(tm_path),load(sk_path),load(ag_path)
 for name,m in (('text',tm),('skill',sk),('agent',ag)):
  if m.get('status')!='pass' or m.get('scientific_result') is not True: raise RuntimeError(f'raw carrier marker not PASS: {name}')

 activation={
  'schema':'openevo.ceiling1-stage2-activation.v1','status':'authorized','stage2_config_sha256':sha(protocol_path),
  'authorized_text_memory_schema':tm.get('schema'),'authorized_text_memory_split_events':int(tm.get('split_events',0)),
  'authorized_skill_bundle_schema':sk.get('schema'),'authorized_skill_bundle_split_events':int(sk.get('split_events',0)),
  'authorized_agent_system_schema':ag.get('schema'),'authorized_agent_system_split_events':int(ag.get('split_events',0)),
  'authorized_arms':[ARM],
  'per_arm':{ARM:{'authorized':True,'stage1_complete_path':str(compat_path),'stage1_complete_sha256':sha(compat_path),'text_memory_complete_path':str(tm_path),'text_memory_complete_sha256':sha(tm_path),'skill_bundle_complete_path':str(sk_path),'skill_bundle_complete_sha256':sha(sk_path),'agent_system_complete_path':str(ag_path),'agent_system_complete_sha256':sha(ag_path)}},
  'formal_stage2_rollouts_consumed_at_activation':0,'final_panel_access_count_at_activation':0,'stage2_per_arm_independent':True,'final_panel_locked_until_stage2_sealed':True,
  'authorization_basis':'User authorized Qwen3-1.7B Stage2 launch after frozen first-class launch-readiness PASS; all bound gates passed with zero formal Stage2/final-panel/external-teacher consumption.',
  'activation_created_from_source_sha':a.science_execution_sha,'activation_governance_source_sha':a.governance_git_sha,
  'science_to_governance_diff':sorted(diff),
  'evidence':{
   'design':{'path':str(design_path),'sha256':sha(design_path)},'prestage2_ready':{'path':str(ready_path),'sha256':sha(ready_path)},'stage2_entry_compat':{'path':str(compat_path),'sha256':sha(compat_path)},
   'merged_base_verification':{'path':str(base_path),'sha256':sha(base_path)},'initial_runtime_preflight':{'path':str(preflight_path),'sha256':sha(preflight_path)},'harness201_carrier_seed':{'path':str(carrier_receipt),'sha256':sha(carrier_receipt)},
   'round0_qualification':{'path':str(summary_path),'sha256':sha(summary_path)},'sd_lora_vram_qualification':{'path':str(vram_path),'sha256':sha(vram_path)},'capacity_policy':{'path':str(capacity_path),'sha256':sha(capacity_path)},
   'step_seed_amendment':{'path':str(seed_path),'sha256':sha(seed_path)},'lease_policy':{'path':str(lease_path),'sha256':sha(lease_path)},
  }
 }
 atomic_create(a.output.resolve(),activation); print(json.dumps(activation,sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
