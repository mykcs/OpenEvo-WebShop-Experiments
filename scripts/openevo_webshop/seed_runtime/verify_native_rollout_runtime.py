#!/usr/bin/env python3
"""Bounded zero-formal target native worker/RNG/teacher qualification."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--prior-interface-log',type=Path,required=True)
    ap.add_argument('--temporary-checkpoints',type=Path,required=True)
    ap.add_argument('--prior-native-checkpoint',type=Path)
    args=ap.parse_args()
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('DP8/SP1/TP1 required')
    if not str(args.temporary_checkpoints.resolve()).startswith('/dev/shm/'):raise ValueError('task-owned ephemeral checkpoint root required')
    sys.path[:0]=[str(args.source_root.resolve()),str(Path(__file__).resolve().parent)]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    import verl.workers.fsdp_workers as fw
    from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager as BM
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from seed.analysis import SEEDEpisodeAnalyzer
    rank=int(os.environ['RANK']); torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(1)
    torch.distributed.init_process_group('cpu:gloo,cuda:nccl')
    phases={}
    def phase(name):
        torch.cuda.synchronize();phases[name]=time.monotonic()
        print(json.dumps({'phase':name,'rank':rank,'peak_bytes':torch.cuda.max_memory_allocated()}),flush=True)
    def rng_hash(state):
        normalized={k:(v.cpu().tolist() if torch.is_tensor(v) else v) for k,v in state.items()}
        return hashlib.sha256(pickle.dumps(normalized)).hexdigest()
    def tensor_hash(t):
        return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()
    torch.manual_seed(73);np.random.seed(73)
    import random
    random.seed(73)
    config=OmegaConf.load(args.source_root/'verl/trainer/config/ppo_trainer.yaml')
    config.data.max_prompt_length=4096;config.data.max_response_length=512;config.data.truncation='left'
    ar=config.actor_rollout_ref
    ar.model.path=str(args.model.resolve());ar.model.attn_implementation='sdpa';ar.model.use_remove_padding=False;ar.model.use_fused_kernels=False
    ar.actor.ppo_mini_batch_size=64;ar.actor.ppo_micro_batch_size_per_gpu=8;ar.actor.use_torch_compile=False
    ar.actor.use_kl_loss=True;ar.actor.kl_loss_coef=.01;ar.actor.opd_loss_coef=.01;ar.actor.optim.total_training_steps=160
    ar.rollout.tensor_model_parallel_size=1;ar.rollout.mode='sync';ar.rollout.engine_seed=20261001
    ar.rollout.name='vllm';ar.rollout.temperature=1.;ar.rollout.top_p=1.;ar.rollout.top_k=-1;ar.rollout.n=1
    ar.rollout.prompt_length=4096;ar.rollout.response_length=512;ar.rollout.max_model_len=20480
    ar.rollout.gpu_memory_utilization=.35;ar.rollout.enforce_eager=True;ar.rollout.max_num_seqs=16
    ar.rollout.max_num_batched_tokens=20480
    ar.rollout.load_format='safetensors';ar.rollout.log_prob_micro_batch_size_per_gpu=16
    ar.rollout.log_prob_use_dynamic_bsz=False;ar.ref.log_prob_micro_batch_size_per_gpu=16;ar.ref.log_prob_use_dynamic_bsz=False
    config.algorithm.seed.analysis_backend='policy_vllm';config.algorithm.seed.analysis_max_completion_tokens=512
    config.algorithm.seed.analysis_context_length=16384
    construction={}; original=fw._construct_with_engine_rng_isolation
    def observe_constructor(factory,**kw):
        construction['learner_before']=rng_hash(BM.get_rng_state())
        result=original(factory,**kw)
        construction['learner_after']=rng_hash(BM.get_rng_state())
        assert construction['learner_before']==construction['learner_after'],'engine initialization changed learner RNG'
        return result
    fw._construct_with_engine_rng_isolation=observe_constructor
    phase('before_public_worker_init')
    worker=fw.ActorRolloutRefWorker(ar,role='actor_rollout_ref');worker.init_model()
    phase('public_worker_initialized')
    assert worker.rollout.sampling_params.seed is None,'qualification must not seed individual requests'
    prior=None
    for line in args.prior_interface_log.read_text().split('\n'):
        try:item=json.loads(line)
        except json.JSONDecodeError:continue
        if item.get('status')=='NATIVE_ONE_STEP_INTERFACE_COMPLETED':prior=item
    assert prior and prior['goal_index']==500 and not prior['action']['used_fallback']
    # Reuse the already consumed fixed train-pool prompt. No WebShop reset/step.
    prompt_ids=worker.tokenizer.encode(prior['generation']['prompt'],add_special_tokens=False)
    ids=torch.tensor([prompt_ids],dtype=torch.long)
    raw=np.empty(1,dtype=object);raw[0]=prompt_ids
    def policy_prompt():
        return DataProto.from_dict(tensors={'input_ids':ids.clone(),'attention_mask':torch.ones_like(ids),'position_ids':torch.arange(ids.shape[1]).unsqueeze(0)},non_tensors={'raw_prompt_ids':raw.copy()},meta_info={'do_sample':True,'validate':False})
    class NativeGroupTransport:
        # Fixture transport only: real public worker methods on all eight ranks,
        # all-gather CPU outputs. This is not a Ray RPC qualification claim.
        world_size=8
        def generate_sequences(self,batch):
            output=worker.generate_sequences(batch.chunk(8)[rank])
            self.calls=getattr(self,'calls',0)+1
            physical={'response_sha256':tensor_hash(output.batch['responses']),'raw_outputs':[worker.tokenizer.decode(row,skip_special_tokens=True) for row in output.batch['responses']],'response_shape':list(output.batch['responses'].shape),'scope':'actual per-rank physical teacher rows including padding'}
            with (args.output/f'physical-teacher-{self.calls}-rank-{rank}.json').open('x') as f:json.dump(physical,f,indent=2,sort_keys=True)
            gathered=[None]*8;torch.distributed.all_gather_object(gathered,output)
            return DataProto.concat(gathered)
    teacher=RayPPOTrainer.__new__(RayPPOTrainer)
    teacher.config=config;teacher.tokenizer=worker.tokenizer;teacher.traj_collector=TrajectoryCollector(config,worker.tokenizer)
    teacher.actor_rollout_wg=NativeGroupTransport()
    analyzer=SEEDEpisodeAnalyzer(backend='policy_vllm',max_completion_tokens=512,analysis_prompt_version='seed',skill_mode='episode_step')
    tasks={'fixed-goal500-engineering-context':{'steps':[{'observation':prior['generation']['prompt'],'action':prior['action']['selected'],'reward':0.}], 'candidate_step_indices':[0],'analysis_mode':'teacher_bootstrap','episode_success':False,'task_description':'Fixed train-pool goal500 context reused for engineering RNG verification'}}
    def schedule(label):
        calls=[]
        for kind in ['policy','teacher','policy']:
            before=rng_hash(BM.get_rng_state()); gen_before=tensor_hash(worker.rollout_sharding_manager.gen_random_states)
            if kind=='teacher':
                result,batch_calls=teacher._analyze_seed_episodes_with_policy_vllm(analyzer,tasks)
                sample=result['fixed-goal500-engineering-context']['_skill_gen_sample']
                response=sample['responses'];raw_text=result['fixed-goal500-engineering-context']['llm_raw_output']
            else:
                result=worker.generate_sequences(policy_prompt());response=result.batch['responses'];raw_text=worker.tokenizer.decode(response[0],skip_special_tokens=True)
            after=rng_hash(BM.get_rng_state())
            gen_after=tensor_hash(worker.rollout_sharding_manager.gen_random_states)
            calls.append({'kind':kind,'learner_rng_before':before,'learner_rng_after':after,'rollout_rng_before':gen_before,'rollout_rng_after':gen_after,'response_sha256':tensor_hash(response),'raw_output':raw_text,'response_shape':list(response.shape)})
            # Persist actual raw output before any assertion can terminate the worker.
            raw_path=args.output/f'call-{label}-{len(calls)}-rank-{rank}.json'
            with raw_path.open('x') as f:json.dump(calls[-1],f,sort_keys=True,indent=2)
            if kind=='teacher':assert batch_calls==1
            assert before==after,'generation session changed learner RNG'
            if kind=='policy':assert gen_before!=gen_after,'random policy generation did not advance rollout RNG'
        phase(label);return calls
    # Coupled native checkpoint provider qualification; stock saver always writes
    # model/optimizer/extra even if contents names suggest a smaller save.
    checkpoint=args.prior_native_checkpoint or args.temporary_checkpoints/'native-rng-checkpoint'
    if args.prior_native_checkpoint:
        if not str(checkpoint.resolve()).startswith('/dev/shm/q03-native-worker-rng-005/'):raise ValueError('reuse only this task-owned original checkpoint')
        worker.checkpoint_manager.load_checkpoint(str(checkpoint),del_local_after_load=False)
        phase('prior_native_rng_checkpoint_loaded')
    else:
        worker.checkpoint_manager.save_checkpoint(str(checkpoint),global_step=0,max_ckpt_to_keep=None)
        phase('native_rng_checkpoint_saved')
    first=schedule('policy_teacher_policy_first')
    worker.checkpoint_manager.load_checkpoint(str(checkpoint),del_local_after_load=False)
    phase('native_rng_checkpoint_restored')
    resumed=schedule('policy_teacher_policy_resumed')
    assert [c['response_sha256'] for c in first]==[c['response_sha256'] for c in resumed],'same-schedule restored output differs'
    assert [c['rollout_rng_after'] for c in first]==[c['rollout_rng_after'] for c in resumed],'same-schedule restored rollout RNG differs'
    effective=OmegaConf.to_container(config,resolve=True)
    (args.output/f'effective-config-rank-{rank}.json').write_text(json.dumps(effective,indent=2,sort_keys=True))
    receipt={'status':'PASS_TARGET_PUBLIC_WORKER_ENGINE_TEACHER_RNG_RESUME','scope':'Actual native worker loader/FSDP/ref/vLLM weight-sync and public teacher consumer through DP8 fixture transport, fixed goal500 prompt reuse; real native RNG checkpoint and same-schedule resume; not Ray RPC/full trainer or canonical Q17 qualification','rank':rank,'pid':os.getpid(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'].split(',')[rank],'engine_construction':construction,'request_sampling_seed':None,'first_schedule':first,'resumed_schedule':resumed,'phase_seconds':phases,'cuda_peak_bytes':torch.cuda.max_memory_allocated(),'policy_generations':4,'teacher_generations':2,'webshop_resets':0,'webshop_actions':0,'formal_task_consumption':0,'formal_activation_allowed':False,'full_native_trainer_qualified':False,'canonical_q17_qualified':False,'persistent_storage_qualified':False,'effective_config_scope':'post-worker-normalization; actor local minibatch8 derives from global minibatch64/DP8','physical_samples_all_ranks':48,'reference_scorer_qualified':False,'effective_config_sha256':hashlib.sha256((args.output/f'effective-config-rank-{rank}.json').read_bytes()).hexdigest()}
    with (args.output/f'rank-{rank}.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True)
    torch.distributed.barrier();torch.distributed.destroy_process_group()
    print(json.dumps({'rank':rank,'status':receipt['status']}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
