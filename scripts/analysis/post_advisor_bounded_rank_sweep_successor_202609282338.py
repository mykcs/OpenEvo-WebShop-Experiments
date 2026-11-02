#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, random, time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.openevo_webshop.bounded_effective_state_launch_guard_202609282338 import validate_post_advisor_activation
from typing import Any

ENGINE_NAME = 'post_advisor_bounded_rank_sweep_train_202609282338.py'
ENGINE_SHA256 = 'e3b670fd8a1e3ffb24f0ae8ab22b7095554cabafa552d40eaac4f64743932498'
EMPTY_MANIFEST_SHA256 = '643e9f56342a664eca5ee8637487732d15f67c57f921467bfe68f5a53244dd19'
EMPTY_IDENTITY_SHA256 = '0b60f9751e81b096fb33aa08c1041a18de435ee561078d366d62a18ce6adc4a0'
ADAPTER_CONFIG_SHA256 = 'cbd30da3262c20cd0df9542d7577d14568db14e7d9cbb69291c0182da96d4ee8'
TARGET_NAMES_SHA256 = '1295417b9c7bdd221d58dc6647829c1033023f91f426b4745c934ea208e591cc'
ROUND0_REPLAY_SHA256 = '3de6234a02be06e38c44533866663c5e8bfd462cb1e002f7012c984e6a42952a'
OFF_MODE = 'BOUNDED_OFF'
ON_MODE = 'EFFECTIVE_STATE_GDR_LORA_V1'  # legacy-revision-compat: exact historical frozen identifier
MAG_MODE = 'MAGNITUDE_SCALAR_GATE_V1'  # legacy-revision-compat: exact historical frozen identifier

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def canonical_sha(value: Any) -> str:
    blob=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()
def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def load_engine():
    path=Path(__file__).with_name(ENGINE_NAME)
    if sha256_file(path)!=ENGINE_SHA256:
        raise RuntimeError('frozen deterministic steady-state engine SHA mismatch')
    spec=importlib.util.spec_from_file_location('bounded_gdr_frozen_engine',path)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load frozen deterministic steady-state engine')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def load_empty_manifest(path: Path) -> dict[str,Any]:
    if sha256_file(path)!=EMPTY_MANIFEST_SHA256:
        raise RuntimeError('EMPTY_BOUNDED_HISTORY manifest SHA mismatch')
    value=json.loads(path.read_text())
    if value.get('status')!='FROZEN_EMPTY_BOUNDED_HISTORY':
        raise RuntimeError('EMPTY_BOUNDED_HISTORY manifest not frozen')
    if value.get('treatment_start_identity_sha256')!=EMPTY_IDENTITY_SHA256:
        raise RuntimeError('EMPTY_BOUNDED_HISTORY identity mismatch')
    if value['execution_boundary']['formal_rows_consumed']!=0 or value['execution_boundary']['final_panel_access']!=0:
        raise RuntimeError('EMPTY_BOUNDED_HISTORY protected counters are not zero')
    return value

def load_adapter_config_template(path: Path) -> dict[str,Any]:
    if sha256_file(path)!=ADAPTER_CONFIG_SHA256:
        raise RuntimeError('rank128 adapter config template SHA mismatch')
    return json.loads(path.read_text())
def first_state_balance(torch, current_a, current_b, coefficient: float, target_rank: int):
    if target_rank!=128:
        raise RuntimeError('first Bounded state must be rank128')
    a=current_a.float().cuda(); b=(current_b.float()*float(coefficient)).cuda()
    if a.shape[0]>target_rank or b.shape[1]!=a.shape[0]:
        raise RuntimeError('current factor shape cannot be embedded into rank128')
    qa,ra=torch.linalg.qr(a.T,mode='reduced'); qb,rb=torch.linalg.qr(b,mode='reduced')
    u,s,vh=torch.linalg.svd(rb @ ra.T,full_matrices=False)
    rr=int(s.numel())
    if rr!=int(a.shape[0]):
        raise RuntimeError('current-only SVD rank surface drift')
    sq=s.sqrt()
    b_small=(qb @ (u[:,:rr]*sq.unsqueeze(0))).cpu().contiguous()
    a_small=((sq.unsqueeze(1)*vh[:rr,:]) @ qa.T).cpu().contiguous()
    a_out=torch.zeros((target_rank,a_small.shape[1]),dtype=a_small.dtype)
    b_out=torch.zeros((b_small.shape[0],target_rank),dtype=b_small.dtype)
    a_out[:rr]=a_small; b_out[:,:rr]=b_small
    dense_in=(current_b.detach().float().cpu()*float(coefficient)) @ current_a.detach().float().cpu()
    dense_out=b_out @ a_out
    denom=float(torch.linalg.vector_norm(dense_in)); numer=float(torch.linalg.vector_norm(dense_out-dense_in))
    rel=(numer/denom) if denom else (0.0 if numer==0.0 else float('inf'))
    if not math.isfinite(rel) or rel>1e-5:
        raise RuntimeError(f'first-state dense reconstruction error {rel} exceeds 1e-5')
    return a_out.contiguous(),b_out.contiguous(),rel,int((s>0).sum().item())

def existing_disposition(output_dir: Path, transition_id: str) -> str:
    if not output_dir.exists(): return 'START'
    receipt=output_dir/'bounded_recurrence_receipt.json'
    if not receipt.is_file(): return 'BLOCKED_PARTIAL_OUTPUT'
    row=json.loads(receipt.read_text())
    if row.get('status')=='PASS' and row.get('transition_id')==transition_id: return 'REUSE_COMPLETE'
    return 'BLOCKED_IDENTITY_MISMATCH'
def train_empty(args, engine) -> dict[str,Any]:
    validate_post_advisor_activation(REPO_ROOT)
    engine.configure_deterministic_cuda_environment()
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from openevo.evolution.parametric import sd_lora_trainer as trainer
    from openevo.evolution.parametric.sd_lora import _training_examples
    manifest=load_empty_manifest(args.empty_history_manifest)
    config_template=load_adapter_config_template(args.adapter_config_template)
    started=time.monotonic()
    ds=json.loads(args.dataset_manifest.read_text()); records=args.dataset_manifest.with_name('records.jsonl')
    if engine.sha256_file(records)!=args.expected_records_sha or ds['records_sha256']!=args.expected_records_sha:
        raise RuntimeError('dataset records SHA mismatch')
    dataset=engine.build_dataset_artifact(args.dataset_manifest,args.dataset_artifact_id)
    current=_training_examples((dataset,),maximum=256,minimum_reward=0.5)
    replay=trainer._select_replay_examples([],count=len(current),seed=args.seed)
    replay_buffer=trainer._bounded_replay_buffer([],current,capacity=args.replay_capacity)
    if len(current)!=args.expected_current_count or replay:
        raise RuntimeError('Round0 current/replay count mismatch')
    if len(current)!=args.expected_steps or len(replay_buffer)!=args.expected_output_replay_count:
        raise RuntimeError('Round0 optimizer/replay-buffer count mismatch')
    from openevo.evolution.framework.contracts import canonical_json
    replay_blob=''.join(canonical_json(x)+'\n' for x in replay_buffer).encode()
    replay_sha=hashlib.sha256(replay_blob).hexdigest()
    if replay_sha!=args.expected_output_replay_sha:
        raise RuntimeError('Round0 bounded replay-buffer SHA mismatch')
    engine.configure_deterministic_torch(torch); trainer._initialize_cuda_runtime(torch)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats(); load_started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(str(args.model_path),trust_remote_code=False,torch_dtype=torch.bfloat16,local_files_only=True)
    model.to(torch.device('cuda',0))
    tokenizer=AutoTokenizer.from_pretrained(str(args.model_path),trust_remote_code=False,local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model_load_seconds=time.monotonic()-load_started
    suffixes=('q_proj','k_proj','v_proj','o_proj')
    target_names=trainer._target_module_names(model,suffixes)
    if canonical_sha(target_names)!=TARGET_NAMES_SHA256 or list(target_names)!=manifest['identity']['target_modules']['names']:
        raise RuntimeError('target-module identity mismatch')
    original_side=tokenizer.truncation_side; tokenizer.truncation_side='left'
    try: encoded=[trainer._encode_example(tokenizer,x,args.max_length) for x in current]
    finally: tokenizer.truncation_side=original_side
    for p in model.parameters(): p.requires_grad=False
    coefficients=torch.nn.Parameter(torch.tensor([args.current_coefficient_init],dtype=torch.float32,device='cuda'))
    layer_class=trainer._sd_lora_layer_class(torch); wrappers={}
    for name in target_names:
        base=model.get_submodule(name)
        wrapper=layer_class(base,[],coefficients,rank=args.current_rank,dtype=torch.bfloat16)
        trainer._set_submodule(model,name,wrapper); wrappers[name]=wrapper
    model.gradient_checkpointing_enable()
    if hasattr(model,'enable_input_require_grads'): model.enable_input_require_grads()
    if hasattr(model.config,'use_cache'): model.config.use_cache=False
    direction=[]
    for w in wrappers.values(): direction.extend((w.current_a,w.current_b))
    trainable=[coefficients,*direction]
    optimizer=torch.optim.AdamW((
        {'params':[coefficients],'lr':args.coefficient_learning_rate,'weight_decay':0.0},
        {'params':direction,'lr':args.learning_rate,'weight_decay':0.0},
    ))
    scaler=torch.amp.GradScaler('cuda',enabled=False); optimizer.zero_grad(set_to_none=True); model.train()
    gdr=None; gdr_diag=[]
    if args.treatment_mode==ON_MODE:
        cls=engine.load_effective_state_gdr(args.gdr_runtime_path)
        if engine.sha256_file(args.beta_policy_path)!=engine.GDR_BETA_SHA256:
            raise RuntimeError('Effective-State beta checkpoint SHA mismatch')
        gdr=cls(torch=torch,wrappers=wrappers,coefficients=coefficients,optimizer=optimizer,
                checkpoint_path=args.beta_policy_path,expected_sha256=engine.GDR_BETA_SHA256,
                planned_steps=args.expected_steps)
    losses=[]; coefficient_trajectory=[]; steps=0; train_started=time.monotonic()
    for batch in trainer._batches(encoded,batch_size=1,seed=args.seed,epochs=1):
        tensors={k:v.to('cuda',non_blocking=True) for k,v in trainer._collate(torch,batch,int(tokenizer.pad_token_id)).items()}
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True): loss=model(**tensors).loss
        if not bool(torch.isfinite(loss)): raise RuntimeError('non-finite Round0 loss')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable,args.max_grad_norm)
        if gdr is not None: gdr.capture_before_step()
        scaler.step(optimizer)
        if gdr is not None: gdr_diag.append(gdr.apply_after_shadow_step())
        scaler.update(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().float().cpu())); steps+=1
        coefficient_trajectory.append([steps,float(coefficients.detach().float().cpu()[0])])
    train_seconds=time.monotonic()-train_started
    if steps!=args.expected_steps: raise RuntimeError(f'Round0 optimizer step mismatch: {steps}')
    current_coefficient=float(coefficients.detach().float().cpu()[0])
    current_pairs={name:(w.current_a.detach().float().cpu().contiguous(),w.current_b.detach().float().cpu().contiguous()) for name,w in wrappers.items()}
    for name,w in wrappers.items(): trainer._set_submodule(model,name,w.base_layer)
    del optimizer; torch.cuda.empty_cache(); training_peak=int(torch.cuda.max_memory_allocated())
    del model,tokenizer; torch.cuda.empty_cache()
    out_tensors={}; module_errors={}; nonzero_directions={}; compression_started=time.monotonic()
    for name in target_names:
        ca,cb=current_pairs[name]
        an,bn,err,nz=first_state_balance(torch,ca,cb,current_coefficient,args.state_rank)
        stem=f'base_model.model.{name}'
        out_tensors[f'{stem}.lora_A.weight']=an
        out_tensors[f'{stem}.lora_B.weight']=bn
        module_errors[name]=err; nonzero_directions[name]=nz
    compression_seconds=time.monotonic()-compression_started
    if max(module_errors.values())>1e-5: raise RuntimeError('first-state module reconstruction gate failed')
    if args.output_dir.exists(): raise RuntimeError('output directory already exists')
    args.output_dir.mkdir(parents=True)
    save_file(out_tensors,str(args.output_dir/'adapter_model.safetensors'),metadata={'format':'pt'})
    cfg=dict(config_template); cfg['r']=128; cfg['lora_alpha']=128; cfg['rank_pattern']={}; cfg['alpha_pattern']={}; cfg['inference_mode']=True
    write_json(args.output_dir/'adapter_config.json',cfg)
    replay_written_sha=engine.write_replay(args.output_dir/'openevo_sd_lora_replay.jsonl',replay_buffer)
    if replay_written_sha!=args.expected_output_replay_sha: raise RuntimeError('written Round0 replay SHA mismatch')
    adapter_sha=sha256_file(args.output_dir/'adapter_model.safetensors')
    config_sha=sha256_file(args.output_dir/'adapter_config.json')
    total_seconds=time.monotonic()-started
    receipt={
        'schema':'openevo.bounded-gdr.effective-state-successor-trace.202609152030','status':'PASS',
        'transition_kind':'EMPTY_HISTORY_FIRST_STATE','transition_id':args.transition_id,
        'round_index':0,'logical_history_count_before':0,'logical_history_count_after':1,
        'empty_history_identity_sha256':EMPTY_IDENTITY_SHA256,'prior_adapter_sha256':None,
        'physical_state_rank':128,'current_update_rank':8,'output_adapter_sha256':adapter_sha,
        'adapter_config_sha256':config_sha,'adapter_config_template_sha256':ADAPTER_CONFIG_SHA256,
        'dataset_records_sha256':args.expected_records_sha,'current_training_example_count':len(current),
        'replay_training_example_count':0,'optimizer_steps':steps,'replay_buffer_record_count':len(replay_buffer),
        'replay_output_sha256':replay_written_sha,'training_loss_mean':sum(losses)/len(losses),
        'loss_trajectory':losses,'learned_coefficients':{'current':current_coefficient},
        'coefficient_trajectory':coefficient_trajectory,
        'first_state_materializer':{
            'method':'current_coefficient_absorbed_into_B_then_sealed_algebra_QR_SVD_then_zero_pad_rank128',
            'slot_order':'balanced source slots 0..7; deterministic zeros 8..127',
            'compute_dtype':'float32','persisted_factor_dtype':'float32',
            'module_dense_relative_frobenius_errors':module_errors,
            'max_dense_relative_frobenius_error':max(module_errors.values()),
            'nonzero_singular_direction_count_by_module':nonzero_directions,
            'threshold':1e-5},
        'target_module_count':len(target_names),'target_module_names_sha256':canonical_sha(target_names),
        'deterministic_runtime':{
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'torch_deterministic_algorithms':bool(torch.are_deterministic_algorithms_enabled()),
            'cudnn_benchmark':bool(torch.backends.cudnn.benchmark),'cudnn_deterministic':bool(torch.backends.cudnn.deterministic),
            'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32),'cuda_matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32)},
        'gdr_core_sha':engine.GDR_CORE_SHA if args.treatment_mode==ON_MODE else None,
        'gdr_runtime_sha256':engine.GDR_RUNTIME_SHA256 if args.treatment_mode==ON_MODE else None,
        'gdr_beta_sha256':engine.GDR_BETA_SHA256 if args.treatment_mode==ON_MODE else None,
        'gdr_mapping_version':'EFFECTIVE_STATE_GDR_LORA_V1' if args.treatment_mode==ON_MODE else None,  # legacy-revision-compat: exact historical frozen identifier
        'gdr_mapping_prereg_sha256':engine.GDR_MAPPING_PREREG_SHA256 if args.treatment_mode==ON_MODE else None,
        'gdr_optimizer_event_count':len(gdr_diag),
        'gdr_applied_write_count':int(sum(x['applied_write_count'] for x in gdr_diag)),
        'gdr_zero_direction_noop_count':int(sum(x['zero_direction_noop_count'] for x in gdr_diag)),
        'gdr_g_values':sorted({float(x['g']) for x in gdr_diag}),
        'gdr_retention_values':sorted({float(x['retention']) for x in gdr_diag}),
        'gdr_effective_beta_min':min((float(x['effective_beta_min']) for x in gdr_diag),default=None),
        'gdr_effective_beta_max':max((float(x['effective_beta_max']) for x in gdr_diag),default=None),
        'gdr_factor_displacement_min':min((float(x['factor_displacement_min']) for x in gdr_diag),default=None),
        'gdr_factor_displacement_max':max((float(x['factor_displacement_max']) for x in gdr_diag),default=None),
        'gdr_max_intended_effective_write_abs_error':max((float(x['max_intended_effective_write_abs_error']) for x in gdr_diag),default=None),
        'gdr_optimizer_step_trace':[
            {'optimizer_step':index+1,**{key:float(value) for key,value in row.items()}}
            for index,row in enumerate(gdr_diag)
        ],
        'ordering':'current_optimizer_step:gdr_capture->adam_shadow_step->gdr_apply; round_end:first_state_qr_svd_balance->zero_pad_rank128',
        'timing':{'model_load_seconds':model_load_seconds,'train_loop_seconds':train_seconds,'compression_seconds':compression_seconds,'total_seconds':total_seconds},
        'gpu_peak_memory_bytes_training':training_peak,'gpu_peak_memory_bytes_total':int(torch.cuda.max_memory_allocated()),
        'protected_final_panel_access':0,'formal_rows_consumed':0,'treatment_mode':args.treatment_mode,
        'frozen_steady_state_engine_sha256':ENGINE_SHA256,
    }
    write_json(args.output_dir/'bounded_recurrence_receipt.json',receipt)
    print(json.dumps(receipt,sort_keys=True)); return receipt

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--transition-kind',choices=('empty-first-state','nonempty-steady-state'),required=True)
    p.add_argument('--indexing-mode',choices=('new-zero-based','historical-parity'),required=True)
    p.add_argument('--dataset-manifest',type=Path,required=True); p.add_argument('--dataset-artifact-id',required=True)
    p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--round-index',type=int,required=True); p.add_argument('--logical-history-count',type=int,required=True)
    p.add_argument('--state-rank',type=int,default=128); p.add_argument('--current-rank',type=int,default=8)
    p.add_argument('--seed',type=int,default=1993); p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--coefficient-learning-rate',type=float,default=1e-2); p.add_argument('--current-coefficient-init',type=float,default=0.8)
    p.add_argument('--max-length',type=int,default=2048); p.add_argument('--max-grad-norm',type=float,default=1.0)
    p.add_argument('--replay-capacity',type=int,default=64); p.add_argument('--expected-current-count',type=int,required=True)
    p.add_argument('--expected-prior-replay-count',type=int,default=0); p.add_argument('--expected-replay-count',type=int,default=0)
    p.add_argument('--expected-output-replay-count',type=int,default=63); p.add_argument('--expected-steps',type=int,required=True)
    p.add_argument('--expected-prior-sha'); p.add_argument('--expected-records-sha',required=True); p.add_argument('--expected-output-replay-sha',required=True)
    p.add_argument('--prior-adapter',type=Path); p.add_argument('--prior-replay',type=Path)
    p.add_argument('--empty-history-manifest',type=Path); p.add_argument('--adapter-config-template',type=Path)
    p.add_argument('--transition-id',required=True); p.add_argument('--treatment-mode',choices=(OFF_MODE,ON_MODE,MAG_MODE),required=True)
    p.add_argument('--gdr-runtime-path',type=Path); p.add_argument('--beta-policy-path',type=Path)
    return p.parse_args()

def main() -> int:
    args=parse_args()
    try:
        validate_post_advisor_activation(REPO_ROOT)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    engine=load_engine()
    if args.current_rank!=8:
        raise SystemExit('post-advisor short-fork successor requires current_rank=8')
    disposition=existing_disposition(args.output_dir,args.transition_id)
    if disposition=='REUSE_COMPLETE':
        print(json.dumps({'status':'REUSE_COMPLETE','transition_id':args.transition_id},sort_keys=True)); return 0
    if disposition!='START': raise SystemExit(f'successor cannot restart: {disposition}')
    if args.treatment_mode==ON_MODE and (args.gdr_runtime_path is None or args.beta_policy_path is None):
        raise SystemExit('GDR ON requires runtime and beta policy')
    if args.treatment_mode!=ON_MODE and (args.gdr_runtime_path is not None or args.beta_policy_path is not None):
        raise SystemExit('non-GDR treatment forbids GDR runtime/beta inputs')
    if args.transition_kind=='empty-first-state':
        if args.state_rank!=128:
            raise SystemExit('empty-history path remains historical rank128 only')
        if args.indexing_mode!='new-zero-based' or args.round_index!=0 or args.logical_history_count!=0:
            raise SystemExit('EMPTY_BOUNDED_HISTORY first transition is exactly Round0/history0')
        if args.prior_adapter is not None or args.prior_replay is not None or args.expected_prior_sha is not None:
            raise SystemExit('Round0 EMPTY_BOUNDED_HISTORY forbids prior adapter/replay')
        if args.empty_history_manifest is None or args.adapter_config_template is None:
            raise SystemExit('Round0 requires frozen empty-history manifest and rank128 config template')
        if args.expected_prior_replay_count!=0 or args.expected_replay_count!=0:
            raise SystemExit('Round0 requires zero prior/replay training examples')
        train_empty(args,engine); return 0
    if args.state_rank not in (8,16,64,128):
        raise SystemExit('post-advisor non-empty short fork requires persistent state_rank in {8,16,64,128}')
    if args.prior_adapter is None or args.prior_replay is None or args.expected_prior_sha is None:
        raise SystemExit('non-empty steady state requires matching-rank predecessor and prior replay')
    if args.empty_history_manifest is not None:
        raise SystemExit('EMPTY_BOUNDED_HISTORY manifest is first-transition-only')
    if args.indexing_mode=='new-zero-based':
        if args.round_index<1 or args.logical_history_count!=args.round_index:
            raise SystemExit('new Round1+ indexing requires history_count == round_index >= 1')
    else:
        if not (150<=args.round_index<=159) or args.logical_history_count!=args.round_index-1:
            raise SystemExit('historical parity indexing requires sealed 150..159/count=round-1')
    engine.train(args); return 0

if __name__=='__main__':
    raise SystemExit(main())
