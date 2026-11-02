#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.openevo_webshop.ceiling1_stage2_arm_controller import (
 Controller,atomic_json,capacity_decision,ensure_round_state_before,load,sha,state_id,validate_attempt_rows,
 validate_round_state_payloads,validate_round_transaction,validate_rollout_summary,validate_sd_terminal,
 validate_sd_training_update,write_once_json,
)
from scripts.openevo_webshop.ceiling1_stage2_vnext_state import artifact_digest,evaluate_acceptance
from scripts.openevo_webshop.run_ceiling1_harness201_g4 import select_run

H2011_PROMPT_SHA='8efcba565fc11e485192f6ca85cb62119f7322662cff1a5e6597de0608d441b5'
V5_CONFIG_SHA='f7b1301fe3448d6f530fa691f4fc53e1aeac7e2169137f2cec63612f52a1d96f'

def clean_git(repo:Path)->str:
 head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(); dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
 if dirty: raise RuntimeError(f'dirty execution source: {repo}')
 return head
def digest_ids(ids:list[str])->str|None:
 return hashlib.sha256(''.join(ids).encode()).hexdigest() if ids else None
def read_rows(p:Path)->list[dict]: return [json.loads(x) for x in p.read_text().split('\n') if x.strip()]
def sd_sha(path:Path|None)->str|None: return sha(path/'adapter_model.safetensors') if path else None

def validate_vnext_evidence(x:dict,rr:Path,r:int)->None:
 if x.get('schema')!='openevo.ceiling1-stage2-vnext-round-evidence.v1' or x.get('status')!='PASS' or int(x.get('round_index',-1))!=r: raise RuntimeError('vNext evidence identity/status drift')
 if int(x.get('attempt_count',-1))!=128 or int(x.get('task_identity_count',-1))!=16 or x.get('attempts_sha256')!=sha(rr/'attempts.jsonl'): raise RuntimeError('vNext evidence denominator/source drift')
 if x.get('summary_scope') is not None:
  if x.get('summary_scope')!='current-round-unconsumed-step-evidence-v2': raise RuntimeError('vNext evidence summary scope drift')
  if int(x.get('summary_count',-1))!=len(x.get('summaries') or []) or int(x['summary_count'])>int(x['task_identity_count']): raise RuntimeError('vNext bounded summary count drift')
 v=x.get('verifier') or {}
 if int(v.get('identity_failures',-1))!=0 or int(v.get('terminal_reward_parity_failures',-1))!=0 or int(v.get('missing_recorded_state_steps',-1))!=0: raise RuntimeError('vNext deterministic verifier parity/readback failed')
 if v.get('source')!='recorded-rollout-state': raise RuntimeError('vNext formal evidence unexpectedly replayed environment')

def validate_vnext_selection(s:dict,ds:Path,r:int)->None:
 if s.get('schema')!='openevo.ceiling1-stage2-vnext-parametric-selection.v1' or int(s.get('round_index',-1))!=r: raise RuntimeError('vNext SD selection identity drift')
 if s.get('status')=='NOOP_NO_UNCONSUMED_CLEAN_EXACT':
  if s.get('bound_evidence_ids') or s.get('bound_task_identities'): raise RuntimeError('vNext SD NOOP has bound evidence')
  return
 if s.get('status')!='DATASET_READY': raise RuntimeError('vNext SD selection status drift')
 man=load(ds/'manifest.json'); rec=ds/'records.jsonl'
 if s.get('manifest_sha256')!=sha(ds/'manifest.json') or s.get('records_sha256')!=sha(rec) or man.get('records_sha256')!=sha(rec): raise RuntimeError('vNext SD dataset payload drift')
 if list(s.get('bound_evidence_ids') or [])!=list(man.get('bound_evidence_ids') or []) or list(s.get('bound_task_identities') or [])!=list(man.get('bound_task_identities') or []): raise RuntimeError('vNext SD evidence/task binding drift')
 if len(set(s['bound_task_identities']))!=len(s['bound_task_identities']): raise RuntimeError('vNext SD selected duplicate task identity')
 if int(man.get('trace_count',-1))>256 or int(man.get('max_training_examples',-1))!=256: raise RuntimeError('vNext SD resource capacity drift')

class VNextController(Controller):
 def __init__(self,a):
  if a.arm not in {'3b','qwen3-1p7b'}: raise RuntimeError(f'unsupported vNext arm {a.arm}')
  a.harness_profile='201.1'; a.step_seed_policy='injective-v1'; a.recovery_amendment=None; a.manual_hold_release=None
  os.environ['CEILING1_STAGE2_RESOURCE_POLICY']='auto'
  super().__init__(a)
  self.vnext_design_path=a.vnext_design.resolve(); self.vnext_design=load(self.vnext_design_path); self.vnext_design_sha=sha(self.vnext_design_path)
  activation=load(self.activation)
  expected_design_status='FROZEN_DESIGN_NO_ACTIVATION_AUTHORITY' if self.arm=='3b' else 'FROZEN_PREREG_NEEDS_LAUNCH_READINESS'
  if self.vnext_design.get('status')!=expected_design_status or self.vnext_design.get('activation_authority') is not False: raise RuntimeError('vNext design is not frozen/no-authority')
  if activation.get('vnext_design_sha256')!=self.vnext_design_sha or activation.get('vnext_execution_git_sha')!=self.execution_git_sha: raise RuntimeError('vNext activation design/execution source drift')
  if activation.get('authorized_output_basename')!=self.out.name: raise RuntimeError('vNext activation output scope drift')
  shared_cfg=self.wt/'configs/experiment/openevo-ceiling-1.0-gpu-lease-qwen3-stage2-v1.json'
  shared_root=Path('/workspace/control/openevo-ceiling1-gpu-lease-state-q17-stage2-v1')
  self.cfg=shared_cfg; self.lease=self.wt/'scripts/openevo_webshop/ceiling1_gpu_lease_orchestrator.py'; self.lease_root=shared_root; self.use_v3_resource=True
  resource_key='small_arm_resource_config_sha256' if self.arm=='3b' else 'qwen3_resource_config_sha256'
  if activation.get(resource_key)!=sha(self.cfg): raise RuntimeError(f'{self.arm} shared resource config drift')
  live=load(self.lease_root/'state.json')
  if live.get('config_sha256')!=sha(self.cfg): raise RuntimeError(f'{self.arm} shared live resource state config drift')
  if self.arm=='3b' and str((live.get('arm_states') or {}).get('3b','')).upper()!='STAGE2_ACTIVE': raise RuntimeError('formal 3B requires shared arm state STAGE2_ACTIVE')
  initial_probe=Path(activation['vnext_initial_probe_path'])
  if not initial_probe.is_file() or sha(initial_probe)!=activation.get('vnext_initial_probe_sha256'): raise RuntimeError('vNext initial causal probe identity drift')
  px=load(initial_probe)
  if px.get('status')!='PASS' or int(px.get('attempt_count',-1))!=16 or px.get('prompt_contract_sha256')!=H2011_PROMPT_SHA: raise RuntimeError('vNext initial causal probe contract drift')
  current_digest=artifact_digest(Path(self.state['merged_base_receipt']),Path(self.state['memory']),Path(self.state['skill']),Path(self.state['agent_system']),Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None)
  if 'vnext_design_sha256' not in self.state:
   if int(self.state.get('next_round',-1))!=0: raise RuntimeError('cannot initialize vNext metadata after round0')
   self.state.update({'vnext_design_sha256':self.vnext_design_sha,'vnext_active_artifact_digest':current_digest,'vnext_consumed_evidence_ids':[],'vnext_sd_consumed_task_identities':[],'vnext_last_candidate_attempt':None,'vnext_last_sd_candidate_attempt':None,'vnext_sd_candidate_generation_count':0,'vnext_candidate_accept_count':0,'vnext_candidate_reject_count':0,'vnext_active_probe_path':str(initial_probe),'vnext_active_probe_sha256':sha(initial_probe),'sd_checkpoints':[]})
  if self.state.get('vnext_design_sha256')!=self.vnext_design_sha or self.state.get('vnext_active_artifact_digest')!=current_digest: raise RuntimeError('vNext active artifact/design state drift')
  active_probe=Path(self.state['vnext_active_probe_path'])
  if not active_probe.is_file() or sha(active_probe)!=self.state.get('vnext_active_probe_sha256'): raise RuntimeError('vNext active probe cache drift')
  locks=self.out/'execution-source-locks'; locks.mkdir(parents=True,exist_ok=True)
  write_once_json(locks/f'{self.execution_git_sha}.json',{'schema':'openevo.ceiling1-stage2-execution-source-lock.v1','clean_worktree':True,'execution_git_sha':self.execution_git_sha,'vnext_design_sha256':self.vnext_design_sha})
  self.save()

 def launch_retry(self,kind,round_idx,workload,launcher,env,vram=None,max_attempts=4):
  env=dict(env)
  contracts={f'stage2_text_memory_{self.arm}':'harness201-runtime-memory-v1',f'stage2_skill_bundle_{self.arm}':'harness201-runtime-skill-v1',f'stage2_agent_system_{self.arm}':'harness201-runtime-agent-v1'}
  if workload in contracts: env['CEILING1_STAGE2_ARTIFACT_CONTRACT']=contracts[workload]
  return super().launch_retry(kind,round_idx,workload,launcher,env,vram=vram,max_attempts=max_attempts)

 def run_rollout(self,r:int,rr:Path):
  super().run_rollout(r,rr)
  rows=read_rows(rr/'attempts.jsonl')
  escapes=sum(bool((x.get('episode') or {}).get('constraint_escape')) for x in rows); schema=sum(bool((x.get('episode') or {}).get('environment_action_schema_violation')) for x in rows)
  fallback=sum(sum(bool((s.get('action_channel') or {}).get('unconstrained_fallback')) for s in (x.get('episode') or {}).get('trace',[])) for x in rows)
  missing=sum(sum(bool(s.get('action')) and not isinstance(s.get('verifier_state'),dict) for s in (x.get('episode') or {}).get('trace',[])) for x in rows)
  if escapes or schema or fallback or missing: raise RuntimeError(f'vNext rollout mechanical hold escape={escapes} schema={schema} fallback={fallback} missing_verifier_state={missing}')

 def evidence(self,r:int,rr:Path):
  er=rr/'evidence'; e=er/'evidence.json'
  if e.is_file(): x=load(e); validate_vnext_evidence(x,rr,r); return x
  env={**os.environ,'PYTHONPATH':f'{self.wt}:{self.wt}/vendor/openevo-webshop/src'}
  subprocess.run(['python3',str(self.wt/'scripts/openevo_webshop/ceiling1_stage2_vnext_evidence.py'),'--require-recorded-state','--attempts',str(rr/'attempts.jsonl'),'--seed-root',str(self.a.seed_root.resolve()),'--stage2-root',str(self.out),'--state-before',str(rr/'state-before.json'),'--output',str(er),'--round-index',str(r)],env=env,check=True)
  x=load(e); validate_vnext_evidence(x,rr,r); return x

 def run_sd(self,r:int,rr:Path):
  d=rr/'sd-lora'; d.mkdir(parents=True,exist_ok=True); self.ensure_writable(d); term=d/'terminal-receipt.json'; prior=Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None; before=int(self.state.get('sd_component_count',0))
  if term.is_file(): return validate_sd_terminal(load(term),r,prior,self.state.get('current_sd_artifact_id'),before,self.arm,self.config_sha,self.allowed_artifact_execution_shas(r))
  ev=load(rr/'evidence/evidence.json'); gate=ev['gates']['parametric_sd_lora']
  if not gate.get('eligible'):
   x=self.noop(term,'parametric_sd_lora',gate.get('noop_reason') or 'VNEXT_PARAMETRIC_NOOP',prior/'adapter_model.safetensors' if prior else None,r);x.update({'adapter_path':str(prior) if prior else None,'artifact_id':self.state.get('current_sd_artifact_id'),'component_count_after':before});atomic_json(term,x);return prior,self.state.get('current_sd_artifact_id'),before,False
  ds=d/'dataset'; sel=ds/'selection.json'
  if not sel.is_file():
   subprocess.run(['python3',str(self.wt/'scripts/openevo_webshop/ceiling1_stage2_vnext_parametric_dataset.py'),'--candidate-evidence',str(rr/'evidence/candidate-evidence-unconsumed.jsonl'),'--state-before',str(rr/'state-before.json'),'--output',str(ds),'--round-index',str(r),'--max-training-examples','256'],check=True)
  sx=load(sel); validate_vnext_selection(sx,ds,r)
  if sx['status']!='DATASET_READY':
   x=self.noop(term,'parametric_sd_lora',sx['status'],prior/'adapter_model.safetensors' if prior else None,r);x.update({'adapter_path':str(prior) if prior else None,'artifact_id':self.state.get('current_sd_artifact_id'),'component_count_after':before,'selection':sx});atomic_json(term,x);return prior,self.state.get('current_sd_artifact_id'),before,False
  ids=list(sx['bound_evidence_ids']); selection_digest=digest_ids(ids); last=self.state.get('vnext_last_sd_candidate_attempt') or {}; priorsha=sd_sha(prior)
  if last.get('prior_sd_sha256')==priorsha and last.get('selection_evidence_digest')==selection_digest:
   x=self.noop(term,'parametric_sd_lora','VNEXT_SD_NOOP_IDENTICAL_DATASET_ALREADY_ATTEMPTED',prior/'adapter_model.safetensors' if prior else None,r);x.update({'adapter_path':str(prior) if prior else None,'artifact_id':self.state.get('current_sd_artifact_id'),'component_count_after':before,'selection':sx});atomic_json(term,x);return prior,self.state.get('current_sd_artifact_id'),before,False
  cap=capacity_decision(before,self.capacity_policy)
  if not cap['allow_update']:
   x=self.noop(term,'parametric_sd_lora',cap['reason'],prior/'adapter_model.safetensors' if prior else None,r);x.update({'adapter_path':str(prior) if prior else None,'artifact_id':self.state.get('current_sd_artifact_id'),'component_count_after':before,'capacity_decision':cap,'selection':sx});atomic_json(term,x);return prior,self.state.get('current_sd_artifact_id'),before,False
  for fp in (ds/'manifest.json',ds/'records.jsonl'):
   os.chown(fp,1001,1001);os.chmod(fp,0o600)
  rec=d/'update.json'
  if not rec.is_file():
   gen=int(self.state.get('vnext_sd_candidate_generation_count',0)); vram=self.root/'stage2-readiness'/('qwen3-1p7b' if self.arm=='qwen3-1p7b' else '3b')/'sd-lora-vram.json'
   env={'CEILING1_STAGE2_ARM':self.arm,'CEILING1_ANALYSIS_ROOT':str(self.root),'CEILING1_AUTOPILOT_REPO':str(self.wt),'CEILING1_STAGE2_GPU_UUID':'','CEILING1_STAGE2_ROUND':str(r),'CEILING1_STAGE2_GENERATION':str(gen),'CEILING1_STAGE2_DATASET':str(ds),'CEILING1_STAGE2_SD_ROOT':str(d),'CEILING1_STAGE2_SD_RECEIPT':str(rec),'CEILING1_STAGE2_PRIOR_SD':str(prior) if prior else '','CEILING1_STAGE2_PRIOR_SD_ID':self.state.get('current_sd_artifact_id') or '','CEILING1_STAGE2_CONFIG_SHA':self.config_sha,'CEILING1_OPENEVO_CORE_REPO':self.capacity_policy['openevo_core']['checkout_path'],'CEILING1_OPENEVO_CORE_GIT_SHA':self.capacity_policy['openevo_core']['execution_sha'],'CEILING1_STAGE2_CAPACITY_POLICY':str(self.capacity_policy_path),'CEILING1_STAGE2_CAPACITY_POLICY_SHA256':self.capacity_policy_sha256}
   self.launch_retry('training',r,f'stage2_sd_lora_{self.arm}',self.wt/'scripts/openevo_webshop/launch_ceiling1_stage2_sd_lora_container.sh',env,vram=vram,max_attempts=3)
  u=load(rec);validate_sd_training_update(u,before)
  if u.get('capacity_policy_sha256')!=self.capacity_policy_sha256 or int(u.get('effective_rank_limit',-1))!=self.sd_effective_rank_limit or u.get('openevo_core_execution_sha')!=self.capacity_policy['openevo_core']['execution_sha']: raise RuntimeError('vNext SD capacity/source receipt mismatch')
  new=Path(u['adapter_path']);x={'schema':'openevo.ceiling1-stage2-carrier-terminal.v1','status':'UPDATE','kind':'parametric_sd_lora','round_index':r,'adapter_path':str(new),'artifact_id':u['artifact_id'],'component_count_after':int(u['component_count_after']),'update_receipt':u,'vnext_selection':sx};atomic_json(term,x);return new,u['artifact_id'],int(u['component_count_after']),True

 def candidate_probe(self,r:int,rr:Path,memory:Path,skill:Path,agent:Path,sd:Path|None,candidate_digest:str)->dict:
  active_digest=self.state['vnext_active_artifact_digest']
  if candidate_digest==active_digest:
   p=Path(self.state['vnext_active_probe_path']); return load(p)
  d=rr/'candidate-acceptance'; d.mkdir(parents=True,exist_ok=True); result=d/f'probe-{candidate_digest}.json'
  if result.is_file(): return load(result)
  attempt=0; sel=d/f'resource-selection-{candidate_digest[:12]}.json'
  while sel.exists(): attempt+=1;sel=d/f'resource-selection-{candidate_digest[:12]}-retry-{attempt}.json'
  env={'G4_OUTPUT_ROOT':str(d),'VNEXT_ARM':self.arm,'VNEXT_REPO':str(self.wt),'VNEXT_ANALYSIS_ROOT':str(self.root),'VNEXT_SEED_ROOT':str(self.a.seed_root.resolve()),'VNEXT_MEMORY':str(memory),'VNEXT_SKILL':str(skill),'VNEXT_AGENT':str(agent),'VNEXT_SD_ADAPTER':str(sd) if sd else '','VNEXT_PROBE_OUTPUT':str(result)}
  probe_workload=f'stage2_text_memory_{self.arm}';lease_id=f'ceiling1-vnext-r{r:03d}-probe-{candidate_digest[:8]}-a{attempt}';container=f'openevo-vnext-r{r:03d}-probe-{candidate_digest[:8]}-a{attempt}'
  select_run(self.lease,self.cfg,self.lease_root,probe_workload,sel,lease_id,container,self.wt/'scripts/openevo_webshop/ceiling1_stage2_vnext_probe_container_driver.sh',env,timeout=900)
  if not result.is_file():
   tmp=result.with_name('.'+result.name+'.docker-cp-tmp')
   cp=subprocess.run(['docker','cp',f'{container}:{result}',str(tmp)],text=True,capture_output=True)
   if cp.returncode!=0 or not tmp.is_file(): raise RuntimeError(f'vNext candidate probe container exited without recoverable result: {cp.stderr.strip()}')
   os.chmod(tmp,0o644); tmp.replace(result)
  return load(result)

 def _active_nonparam_terminal(self,r:int,kind:str,candidate:dict,prior:Path,accepted:bool)->dict:
  if accepted or candidate.get('status')!='UPDATE': return candidate
  x=self.noop_payload(kind,'VNEXT_CANDIDATE_REJECTED_KEEP_PRIOR',prior,r);x.update({'output_path':str(prior),'output_sha256':sha(prior),'candidate_status':'UPDATE','candidate_output_path':candidate.get('output_path'),'candidate_output_sha256':candidate.get('output_sha256')});return x
 def _active_sd_terminal(self,r:int,candidate:dict,prior:Path|None,prior_id:str|None,prior_count:int,accepted:bool)->dict:
  if accepted or candidate.get('status')!='UPDATE': return candidate
  x=self.noop_payload('parametric_sd_lora','VNEXT_CANDIDATE_REJECTED_KEEP_PRIOR',prior/'adapter_model.safetensors' if prior else None,r);x.update({'adapter_path':str(prior) if prior else None,'artifact_id':prior_id,'component_count_after':prior_count,'candidate_status':'UPDATE','candidate_adapter_path':candidate.get('adapter_path')});return x

 def _resume_staged(self,r:int,rr:Path)->bool:
  staged=rr/'VNEXT_TRANSACTION_STAGED.json'; complete=rr/'ROUND_COMPLETE.json'
  if not staged.is_file() or complete.is_file(): return False
  x=load(staged); after=x['state_after']; rec=x['round_complete']
  if rec.get('state_id_before')!=self.state.get('state_id') or int(rec.get('round_index',-1))!=r: raise RuntimeError('vNext staged transaction state/round drift')
  write_once_json(rr/'state-after.json',after);write_once_json(complete,rec)
  validate_round_transaction(rr,r,self.state['state_id'],self.allowed_rollout_shas(r));prior=Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None;validate_round_state_payloads(rr,sd_sha(prior));self.state=after;self.save();return True

 def run_round(self,r:int):
  if self.execution_git_sha:
   from scripts.openevo_webshop.ceiling1_stage2_arm_controller import assert_execution_source;assert_execution_source(self.wt,self.execution_git_sha)
  rr=self.out/f'round-{r:03d}';rr.mkdir(parents=True,exist_ok=True);complete=rr/'ROUND_COMPLETE.json'
  if complete.is_file():
   expected=self.state['state_id'];validate_round_transaction(rr,r,expected,self.allowed_rollout_shas(r));prior=Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None;validate_round_state_payloads(rr,sd_sha(prior));self.state=load(rr/'state-after.json');self.save();return
  if self._resume_staged(r,rr): return
  ensure_round_state_before(rr/'state-before.json',self.state);self.run_rollout(r,rr);ev=self.evidence(r,rr)
  prior_sd=Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None;prior_mem=Path(self.state['memory']);prior_skill=Path(self.state['skill']);prior_agent=Path(self.state['agent_system']);prior_id=self.state.get('current_sd_artifact_id');prior_count=int(self.state.get('sd_component_count',0));prior_active_digest=self.state['vnext_active_artifact_digest']
  # Science semantics already freeze these four candidate channels as independent reads of the same State_k and sealed round evidence.
  # Ray therefore fans them out concurrently; composition/probe remains a barrier after all four terminal outcomes materialize.
  from concurrent.futures import ThreadPoolExecutor
  with ThreadPoolExecutor(max_workers=4,thread_name_prefix=f'ce1-r{r:03d}-carrier') as pool:
   f_mem=pool.submit(self.reflect,r,rr,'text_memory',prior_mem,sd_adapter=prior_sd)
   f_sd=pool.submit(self.run_sd,r,rr)
   f_skill=pool.submit(self.reflect,r,rr,'skill_bundle',prior_skill,sd_adapter=prior_sd)
   f_agent=pool.submit(self.reflect,r,rr,'agent_system',prior_agent,memory=prior_mem,skill=prior_skill,sd_adapter=prior_sd)
   new_mem=f_mem.result();new_sd,new_sd_id,new_count,sd_updated=f_sd.result();new_skill=f_skill.result();new_agent=f_agent.result()
  memterm=load(rr/'text-memory/terminal-receipt.json');skillterm=load(rr/'skill-bundle/terminal-receipt.json');agentterm=load(rr/'agent-system/terminal-receipt.json');sdterm=load(rr/'sd-lora/terminal-receipt.json')
  tv=self.task_vector_receipt(r,rr,sdterm,prior_sd,new_sd)
  candidate_digest=artifact_digest(Path(self.state['merged_base_receipt']),new_mem,new_skill,new_agent,new_sd);changed=candidate_digest!=prior_active_digest
  prior_probe=load(Path(self.state['vnext_active_probe_path']));candidate_probe=self.candidate_probe(r,rr,new_mem,new_skill,new_agent,new_sd,candidate_digest) if changed else prior_probe
  acceptance=evaluate_acceptance(prior_probe,candidate_probe) if changed else {'accepted':False,'outcome':'NO_CANDIDATE_ARTIFACT_CHANGE','prior':{},'candidate':{},'delta':{},'mechanical_checks':{},'nonregression_checks':{},'strict_improvement_checks':{},'significance_claim':False}
  accepted=bool(acceptance['accepted'])
  partial_bound=list(ev.get('partial_candidate_evidence_ids') or []) if any(t.get('status')=='UPDATE' for t in (memterm,skillterm,agentterm)) else []
  sx=load(rr/'sd-lora/dataset/selection.json') if (rr/'sd-lora/dataset/selection.json').is_file() else {};exact_bound=list(sx.get('bound_evidence_ids') or []) if sdterm.get('status')=='UPDATE' else [];task_bound=list(sx.get('bound_task_identities') or []) if sdterm.get('status')=='UPDATE' else [];bound=list(dict.fromkeys([*partial_bound,*exact_bound]))
  active_mem,new_active_skill,new_active_agent,active_sd,active_id,active_count=(new_mem,new_skill,new_agent,new_sd,new_sd_id,new_count) if accepted else (prior_mem,prior_skill,prior_agent,prior_sd,prior_id,prior_count)
  active_digest=artifact_digest(Path(self.state['merged_base_receipt']),active_mem,new_active_skill,new_active_agent,active_sd)
  after=dict(self.state);after.update({'next_round':r+1,'rollouts_consumed':(r+1)*128,'memory':str(active_mem),'skill':str(new_active_skill),'agent_system':str(new_active_agent),'current_sd_adapter':str(active_sd) if active_sd else None,'current_sd_artifact_id':active_id,'sd_component_count':active_count,'evidence_history':str(rr/'evidence/history-after.json'),'vnext_active_artifact_digest':active_digest,'vnext_active_probe_path':str((rr/'candidate-acceptance'/f'probe-{candidate_digest}.json') if accepted and changed else Path(self.state['vnext_active_probe_path'])),'vnext_active_probe_sha256':sha((rr/'candidate-acceptance'/f'probe-{candidate_digest}.json') if accepted and changed else Path(self.state['vnext_active_probe_path'])),'rounds_complete':list(range(r+1)),'external_teacher_calls':0,'final_panel_access_count':0})
  after['state_id']=state_id(Path(after['merged_base_receipt']),active_sd,active_mem,new_active_skill,new_active_agent,r)
  consumed=set(self.state.get('vnext_consumed_evidence_ids') or []);sd_tasks=set(self.state.get('vnext_sd_consumed_task_identities') or [])
  if accepted: consumed.update(bound); sd_tasks.update(task_bound if sdterm.get('status')=='UPDATE' else [])
  after['vnext_consumed_evidence_ids']=sorted(consumed);after['vnext_sd_consumed_task_identities']=sorted(sd_tasks)
  if ev.get('should_attempt_candidate'):
   after['vnext_last_candidate_attempt']={'round_index':r,'active_artifact_digest':prior_active_digest,'evidence_digest':ev.get('candidate_evidence_digest'),'outcome':acceptance['outcome'],'accepted':accepted,'bound_evidence_ids':bound}
  if sdterm.get('status')=='UPDATE':
   after['vnext_sd_candidate_generation_count']=int(self.state.get('vnext_sd_candidate_generation_count',0))+1;after['vnext_last_sd_candidate_attempt']={'round_index':r,'prior_sd_sha256':sd_sha(prior_sd),'selection_evidence_digest':digest_ids(list(sx.get('bound_evidence_ids') or [])),'outcome':acceptance['outcome'],'accepted':accepted}
   if accepted:
    after['sd_generation_count']=int(self.state.get('sd_generation_count',0))+1;after['last_task_vector_geometry']=str(rr/'task-vector-geometry.json');after.setdefault('sd_checkpoints',[]).append({'generation_index':int(sdterm['update_receipt']['generation_index']),'round_index':r,'component_count':new_count,'path':str(new_sd),'adapter_sha256':sdterm['update_receipt']['adapter_digest_after'],'geometry_path':str(rr/'task-vector-geometry.json'),'retained_payload':True})
  after['vnext_candidate_accept_count']=int(self.state.get('vnext_candidate_accept_count',0))+(1 if accepted else 0);after['vnext_candidate_reject_count']=int(self.state.get('vnext_candidate_reject_count',0))+(1 if changed and not accepted else 0)
  active_terms={'text_memory':self._active_nonparam_terminal(r,'text_memory',memterm,prior_mem,accepted),'skill_bundle':self._active_nonparam_terminal(r,'skill_bundle',skillterm,prior_skill,accepted),'agent_system':self._active_nonparam_terminal(r,'agent_system',agentterm,prior_agent,accepted),'parametric_sd_lora':self._active_sd_terminal(r,sdterm,prior_sd,prior_id,prior_count,accepted)}
  decision={'schema':'openevo.ceiling1-stage2-vnext-candidate-decision.v1','round_index':r,'prior_active_artifact_digest':prior_active_digest,'candidate_artifact_digest':candidate_digest,'active_artifact_digest_after':active_digest,'candidate_changed':changed,'acceptance':acceptance,'candidate_probe_sha256':sha(rr/'candidate-acceptance'/f'probe-{candidate_digest}.json') if changed else self.state['vnext_active_probe_sha256'],'prior_probe_sha256':self.state['vnext_active_probe_sha256'],'candidate_evidence_digest':ev.get('candidate_evidence_digest'),'bound_evidence_ids':bound,'bound_sd_task_identities':task_bound,'candidate_terminals':{'text_memory':memterm,'skill_bundle':skillterm,'agent_system':agentterm,'parametric_sd_lora':sdterm},'active_terminals':active_terms,'formal_task_consumption_delta':0,'final_panel_access_delta':0,'external_teacher_calls_delta':0}
  write_once_json(rr/'VNEXT_CANDIDATE_DECISION.json',decision)
  diag={'schema':'openevo.ceiling1-stage2-vnext-noop-diagnostic.v1','status':'OBSERVE','round_index':r,'candidate_changed':changed,'candidate_accepted':accepted,'sd_candidate_status':sdterm.get('status'),'modify_training':False,'pause_before_next_round':False};atomic_json(rr/'sd-lora/noop-window-diagnostic.json',diag)
  atomic_json(rr/'checkpoint-retention.json',{'schema':'openevo.ceiling1-stage2-retention.v1','round_index':r,'pruned':[],'policy':'vnext-preserve-candidate-provenance-v1'})
  round_rec={'schema':'openevo.ceiling1-stage2-round-complete.v1','status':'PASS','arm':self.arm,'round_index':r,'rollout_summary':load(rr/'summary.json'),'text_memory':active_terms['text_memory'],'parametric_sd_lora':active_terms['parametric_sd_lora'],'sd_lora_noop_monitor':diag,'skill_bundle':active_terms['skill_bundle'],'agent_system':active_terms['agent_system'],'task_vector':tv,'state_id_before':load(rr/'state-before.json')['state_id'],'state_id_after':after['state_id'],'external_teacher_calls':0,'vnext_candidate_decision_sha256':sha(rr/'VNEXT_CANDIDATE_DECISION.json')}
  staged={'schema':'openevo.ceiling1-stage2-vnext-staged-transaction.v1','round_index':r,'state_after':after,'round_complete':round_rec};write_once_json(rr/'VNEXT_TRANSACTION_STAGED.json',staged);write_once_json(rr/'state-after.json',after);write_once_json(complete,round_rec)
  validate_round_transaction(rr,r,self.state['state_id'],self.allowed_rollout_shas(r));validate_round_state_payloads(rr,sd_sha(prior_sd));self.state=after;self.save()

def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=('3b','qwen3-1p7b'),default='3b');p.add_argument('--repo',type=Path,required=True);p.add_argument('--v5-repo',type=Path);p.add_argument('--state-root',type=Path);p.add_argument('--analysis-root',type=Path,required=True);p.add_argument('--seed-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage2-activation',type=Path,required=True);p.add_argument('--vnext-design',type=Path,required=True);p.add_argument('--capacity-policy',type=Path,required=True);p.add_argument('--step-seed-amendment',type=Path,required=True);p.add_argument('--harness201-carrier-root',type=Path,required=True);p.add_argument('--execution-git-sha',required=True);a=p.parse_args();return VNextController(a).run()
if __name__=='__main__':raise SystemExit(main())
