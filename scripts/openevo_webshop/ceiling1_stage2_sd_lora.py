#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path
from typing import Any
from openevo.evolution.framework.contracts import canonical_digest
from openevo.evolution.framework.execution import MethodExecutionContext, MethodExecutionServices, ResolvedMethodInputBinding, build_execution_envelope, worker_input_artifact_digest
from openevo.evolution.models import WorkerClaimInputArtifact, WorkerClaimedJob
from openevo.evolution.parametric.contracts import SdLoraMethodConfig
from openevo.evolution.parametric.sd_lora import parametric_memory_sd_lora
from openevo.evolution.parametric.trainer_service import SubprocessSdLoraTrainerService
from scripts.openevo_webshop import run_parametric_smoke
from scripts.openevo_webshop.ceiling1_harness201_g8 import load_capacity_policy, sha256 as g8_sha256
from scripts.openevo_webshop.h145_execute import materialize_local_hf_snapshot
MAX_COMPONENTS=64

def sha(p:Path)->str:
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def atomic_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); t.replace(p)
class NoInferenceHarness:
    def infer(self,request): raise AssertionError('SD-LoRA attempted unexpected LLM inference')
def prior_artifact(path:Path|None,aid:str|None):
    if (path is None)!=(aid is None): raise ValueError('prior path/id mismatch')
    if path is None: return None
    root=path.resolve(); req=['adapter_config.json','adapter_model.safetensors','openevo_sd_lora_replay.jsonl','openevo_sd_lora_state.json','openevo_sd_lora_state.safetensors']; miss=[x for x in req if not (root/x).is_file()]
    if miss: raise ValueError(f'prior cumulative adapter incomplete: {miss}')
    return WorkerClaimInputArtifact(artifact_id=str(aid),type='parametric_memory',uri=root.as_uri(),name=root.name)
def prior_count(path:Path|None):
    if path is None:return 0
    x=json.loads((path/'openevo_sd_lora_state.json').read_text()); return len(x.get('components',[])) if isinstance(x.get('components'),list) else int(x['component_count'])
def stage_prior_adapter(path:Path|None, artifact_root:Path)->Path|None:
    if path is None: return None
    src=path.resolve(); dst=artifact_root/'prior-input'; dst.mkdir(mode=0o700,parents=True,exist_ok=True)
    required=['adapter_config.json','adapter_model.safetensors','openevo_sd_lora_replay.jsonl','openevo_sd_lora_state.json','openevo_sd_lora_state.safetensors']
    for name in required:
        sp=src/name; dp=dst/name
        if not sp.is_file(): raise ValueError(f'prior cumulative adapter incomplete: {name}')
        if dp.exists():
            if sha(sp)!=sha(dp): raise ValueError(f'prior input staging drift: {name}')
        else: shutil.copyfile(sp,dp)
        dp.chmod(0o600)
    return dst
def cleanup_staged_prior_adapter(path:Path|None,replacement:Path)->dict|None:
    if path is None:return None
    path=path.resolve(); replacement=replacement.resolve(); required=['adapter_config.json','adapter_model.safetensors','openevo_sd_lora_replay.jsonl','openevo_sd_lora_state.json','openevo_sd_lora_state.safetensors']
    if path.name!='prior-input': raise RuntimeError(f'refusing to retire unexpected staged prior path: {path}')
    observed={item.name for item in path.iterdir()}
    if observed!=set(required): raise RuntimeError(f'staged prior inventory drift: {sorted(observed)}')
    for name in required:
        if not (replacement/name).is_file(): raise RuntimeError(f'replacement cumulative adapter incomplete: {name}')
    files=[{'name':name,'bytes':(path/name).stat().st_size,'sha256':sha(path/name)} for name in required]
    replacement_identity={name:sha(replacement/name) for name in required}
    for name in required:(path/name).unlink()
    path.rmdir()
    receipt={'schema':'openevo.ceiling1-stage2-prior-input-retirement.v1','status':'PASS','reason':'transient trainer input copy retired only after replacement cumulative adapter validation','prior_input_path':str(path),'files_removed':files,'replacement_adapter_path':str(replacement),'replacement_identity':replacement_identity}
    atomic_json(path.parent/'prior-input-retired.json',receipt); return receipt
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--dataset-manifest',type=Path,required=True); p.add_argument('--dataset-artifact-id',required=True); p.add_argument('--artifact-root',type=Path,required=True); p.add_argument('--evidence',type=Path,required=True); p.add_argument('--arm',choices=('3b','7b','qwen3-1p7b'),required=True); p.add_argument('--round-index',type=int,required=True); p.add_argument('--generation-index',type=int,required=True); p.add_argument('--python',required=True); p.add_argument('--base-model',required=True); p.add_argument('--local-model-path',type=Path,required=True); p.add_argument('--model-revision',required=True); p.add_argument('--prior-adapter',type=Path); p.add_argument('--prior-artifact-id'); p.add_argument('--gpu-uuid',required=True); p.add_argument('--experiment-git-sha',required=True); p.add_argument('--config-sha256',required=True); p.add_argument('--capacity-policy',type=Path); p.add_argument('--openevo-core-sha'); a=p.parse_args()
    run_parametric_smoke._install_runtime_compatibility()
    if os.environ.get('CUDA_VISIBLE_DEVICES','').strip()!=a.gpu_uuid: raise ValueError('CUDA_VISIBLE_DEVICES mismatch')
    if '/' not in a.base_model or a.base_model.startswith(('/', '.')): raise ValueError('base_model must be canonical model ID')
    if not (a.local_model_path/'config.json').is_file(): raise ValueError('local merged base is incomplete')
    a.artifact_root.mkdir(parents=True,exist_ok=True)
    hf_home=a.artifact_root/'hf-home'; hf_home.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(hf_home)
    materialize_local_hf_snapshot(a.base_model,str(a.local_model_path),a.model_revision)
    man=json.loads(a.dataset_manifest.read_text()); records=a.dataset_manifest.with_name('records.jsonl')
    if sha(records)!=man['records_sha256']: raise ValueError('dataset hash mismatch')
    before=prior_count(a.prior_adapter); staged_prior=stage_prior_adapter(a.prior_adapter,a.artifact_root)
    prior=prior_artifact(staged_prior,a.prior_artifact_id)
    capacity_policy=load_capacity_policy(a.capacity_policy,a.arm)
    if capacity_policy is None:
        if before>=MAX_COMPONENTS: raise RuntimeError('NATIVE_SCALABILITY_LIMIT')
        capacity_policy_sha256=None; effective_rank_limit=MAX_COMPONENTS*8
    else:
        candidate_effective_rank=(before+1)*int(capacity_policy['rank']); effective_rank_limit=int(capacity_policy['effective_rank_limit'])
        if candidate_effective_rank>effective_rank_limit: raise RuntimeError('NATIVE_SD_LORA_EFFECTIVE_RANK_LIMIT')
        capacity_policy_sha256=g8_sha256(a.capacity_policy)
        if a.openevo_core_sha!=capacity_policy['openevo_core']['execution_sha']: raise RuntimeError('Harness201 successor OpenEvo Core SHA mismatch')
    dataset=WorkerClaimInputArtifact(artifact_id=a.dataset_artifact_id,type='dataset',uri=a.dataset_manifest.resolve().as_uri(),name=str(man.get('name') or a.dataset_artifact_id),manifest_sha256=canonical_digest(man),records_byte_size=int(man['records_byte_size']),records_sha256=str(man['records_sha256']))
    priors=(prior,) if prior else (); bindings=(ResolvedMethodInputBinding(binding_id='current_dataset',artifact_ids=(dataset.artifact_id,),artifact_digests=(worker_input_artifact_digest(dataset),)),ResolvedMethodInputBinding(binding_id='prior_target_artifacts',artifact_ids=tuple(x.artifact_id for x in priors),artifact_digests=tuple(worker_input_artifact_digest(x) for x in priors)))
    cfg=SdLoraMethodConfig(base_model=a.base_model,model_revision=a.model_revision,max_records=256); user=cfg.model_dump(mode='json')
    expected={'rank':8,'learning_rate':0.0002,'epochs':1,'max_records':256,'replay_capacity':64,'max_length':2048,'dtype':'bfloat16','seed':1993}
    for k,v in expected.items():
        actual=user[k]; actual=getattr(actual,'value',actual)
        if actual!=v: raise RuntimeError(f'native config drift {k}={actual!r} expected={v!r}')
    if list(user['target_modules'])!=['q_proj','k_proj','v_proj','o_proj']: raise RuntimeError('target module drift')
    core={'name':f'ceiling1-{a.arm}-stage2-generation-{a.generation_index:03d}','experiment_id':'OpenEVO-Ceiling-1.0','lineage':{'experiment_git_sha':a.experiment_git_sha,'config_sha256':a.config_sha256,'round_index':a.round_index,'dataset_manifest_sha256':canonical_digest(man),'dataset_records_sha256':sha(records),'prior_parametric_memory_artifact_id':prior.artifact_id if prior else None,'upstream_openevo_sha':'796e6248a61112a61c880885844d3b0420c0b6ae'},'tags':['ceiling1','webshop',a.arm,'stage2','sd-lora'],'scores':{},'compatibility':{'agent_harnesses':['local_transformers']},'promoted':False}
    env=build_execution_envelope(plan_id=f'ceiling1-plan-{a.arm}-{a.round_index:03d}',plan_digest='a'*64,registry_snapshot_digest='b'*64,target_id='parametric_memory',method_id='parametric_memory_sd_lora',method_identity_digest='c'*64,user_config=user,core_config=core,input_bindings=bindings,output_artifact_types=('parametric_memory',))
    job=WorkerClaimedJob(job_id=f'ceiling1-job-{a.arm}-{a.round_index:03d}',lease_id=f'ceiling1-method-{a.arm}-{a.round_index:03d}',job_type='parametric_memory',method='parametric_memory_sd_lora',input_artifacts=[dataset,*priors],config={**user,**core}); a.artifact_root.mkdir(parents=True,exist_ok=True)
    with SubprocessSdLoraTrainerService(a.artifact_root,python_executable=a.python) as trainer:
        ctx=MethodExecutionContext(job=job,artifact_root=a.artifact_root,envelope=env,services=MethodExecutionServices(harness=NoInferenceHarness(),parametric_trainer=trainer)); arts=parametric_memory_sd_lora(ctx)
    if len(arts)!=1: raise RuntimeError('expected exactly one SD-LoRA artifact')
    art=arts[0]; outdir=Path(str(art.uri).removeprefix('file://')); after=int(art.manifest.get('component_count') or 0)
    if after!=before+1: raise RuntimeError(f'component count drift {before}->{after}')
    retirement=cleanup_staged_prior_adapter(staged_prior,outdir)
    payload={'schema':'openevo.ceiling1-stage2-sd-lora.v1','status':'UPDATE','arm':a.arm,'round_index':a.round_index,'generation_index':a.generation_index,'gpu_uuid':a.gpu_uuid,'base_model':a.base_model,'local_model_path':str(a.local_model_path.resolve()),'model_revision':a.model_revision,'dataset_record_count':int(man['record_count']),'dataset_trace_count':int(man.get('trace_count',0)),'current_training_example_count':art.manifest.get('training_record_count'),'replay_example_count':art.manifest.get('replay_training_record_count'),'optimizer_example_count':art.manifest.get('optimizer_training_record_count'),'optimizer_steps':art.manifest.get('steps_completed'),'training_loss':art.manifest.get('training_loss'),'component_count_before':before,'component_count_after':after,'effective_rank_after':art.manifest.get('effective_rank'),'prior_artifact_id':prior.artifact_id if prior else None,'prior_input_retirement':retirement,'artifact_id':getattr(art,'artifact_id',None) or art.name,'adapter_path':str(outdir),'adapter_digest_after':sha(outdir/'adapter_model.safetensors'),'gpu_peak_memory_bytes':art.manifest.get('gpu_peak_memory_bytes'),'privileged_external_teacher_calls':0,'native_user_config':user,'experiment_git_sha':a.experiment_git_sha,'config_sha256':a.config_sha256,'capacity_policy_sha256':capacity_policy_sha256,'effective_rank_limit':effective_rank_limit,'openevo_core_execution_sha':a.openevo_core_sha}
    atomic_json(a.evidence,payload); print(json.dumps(payload,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
