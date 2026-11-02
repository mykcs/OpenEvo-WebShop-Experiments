#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def artifact_digest(base_receipt:Path,memory:Path,skill:Path,agent:Path,sd_adapter:Path|None)->str:
 payload={'base_receipt_sha256':sha(base_receipt),'memory_sha256':sha(memory),'skill_sha256':sha(skill),'agent_sha256':sha(agent),'sd_adapter_sha256':sha(sd_adapter/'adapter_model.safetensors') if sd_adapter else None}
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def probe_metrics(x:dict[str,Any])->dict[str,Any]:
 return {'exact_success':int(x['exact_success']),'positive_score':int(x['positive_score']),'mean_score':float(x['mean_score']),'invalid_terminated':int(x['invalid_terminated'])}
def evaluate_acceptance(prior:dict[str,Any],candidate:dict[str,Any])->dict[str,Any]:
 pm,cm=probe_metrics(prior),probe_metrics(candidate)
 mechanics={'prior_pass':prior.get('status')=='PASS','candidate_pass':candidate.get('status')=='PASS','candidate_constraint_escape_zero':int(candidate.get('constraint_escape_count',0))==0,'candidate_environment_schema_violation_zero':int(candidate.get('environment_action_schema_violation_count',0))==0}
 nonreg={'exact':cm['exact_success']>=pm['exact_success'],'positive':cm['positive_score']>=pm['positive_score'],'mean':cm['mean_score']+1e-15>=pm['mean_score'],'invalid':cm['invalid_terminated']<=pm['invalid_terminated']}
 strict={'exact':cm['exact_success']>pm['exact_success'],'positive':cm['positive_score']>pm['positive_score'],'mean':cm['mean_score']>pm['mean_score']+1e-15}
 accepted=all(mechanics.values()) and all(nonreg.values()) and any(strict.values())
 if accepted: outcome='ACCEPT_STRICT_TASK_IMPROVEMENT'
 elif not all(mechanics.values()): outcome='REJECT_MECHANICAL'
 elif not all(nonreg.values()): outcome='REJECT_TASK_REGRESSION'
 else: outcome='REJECT_NEUTRAL_NO_TASK_IMPROVEMENT'
 return {'accepted':accepted,'outcome':outcome,'prior':pm,'candidate':cm,'delta':{'exact_success':cm['exact_success']-pm['exact_success'],'positive_score':cm['positive_score']-pm['positive_score'],'mean_score':cm['mean_score']-pm['mean_score'],'invalid_terminated':cm['invalid_terminated']-pm['invalid_terminated']},'mechanical_checks':mechanics,'nonregression_checks':nonreg,'strict_improvement_checks':strict,'significance_claim':False}
