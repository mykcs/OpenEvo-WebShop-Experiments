#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,subprocess,time
from pathlib import Path
SCHEMAS={'openevo.ceiling1-stage2-resource-migration.ray-v1','openevo.ceiling1-stage2-resource-migration.ray-continuation-v1'}
SCIENCE_FILES=['configs/experiment/openevo-ceiling-1.0-stage2-final-v1.json','configs/experiment/manifests/h1.44-on-3b-training-schedule-v1.json','scripts/openevo_webshop/ceiling1_harness2011_prompt.py','scripts/openevo_webshop/ceiling1_stage2_format_hard_runtime.py','scripts/openevo_webshop/ceiling1_stage2_rollout_worker.py','scripts/openevo_webshop/ceiling1_stage2_rollout_worker_qwen3.py','scripts/openevo_webshop/ceiling1_stage2_round_driver.py','scripts/openevo_webshop/ceiling1_stage2_reflector_qwen3.py','scripts/openevo_webshop/ceiling1_stage2_sd_lora.py','scripts/openevo_webshop/ceiling1_stage2_vnext_evidence.py','scripts/openevo_webshop/ceiling1_stage2_vnext_parametric_dataset.py','scripts/openevo_webshop/ceiling1_stage2_vnext_state.py']
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gitb(repo,ref,path): return subprocess.check_output(['git','-C',str(repo),'show',f'{ref}:{path}'])
def rows_shas(path):
    vals=set()
    for line in Path(path).read_text().split('\n'):
        if line.strip():
            v=json.loads(line).get('experiment_git_sha')
            if v: vals.add(v)
    return sorted(vals)
def write_once(p,x):
    p=Path(p);text=json.dumps(x,indent=2,sort_keys=True)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_text()!=text: raise RuntimeError(f'immutable arm-continuation conflict: {p}')
        return
    p.write_text(text)
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--predecessor-sha',required=True);ap.add_argument('--successor-sha',required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--ray-state-root',type=Path,required=True);ap.add_argument('--ray-config',type=Path,required=True);ap.add_argument('--ray-ready',type=Path,required=True);ap.add_argument('--prior-ray-migration',type=Path,required=True);ap.add_argument('--activation',type=Path,required=True);ap.add_argument('--science-amendment',type=Path,required=True);ap.add_argument('--carrier-recovery-receipt',type=Path,required=True);ap.add_argument('--arm',choices=('3b','qwen3-1p7b'),required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    repo=a.repo.resolve();root=a.root.resolve();rayroot=a.ray_state_root.resolve();cfg=load(a.ray_config);ready=load(a.ray_ready);prior=load(a.prior_ray_migration);am=load(a.science_amendment);rec=load(a.carrier_recovery_receipt)
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip();require(head==a.successor_sha and not dirty,'arm-science continuation repo is not exact clean successor')
    require(prior.get('schema') in SCHEMAS and prior.get('status') in {'AUTHORIZED','AUTHORIZED_PENDING_SCIENCE_RECOVERY'} and prior.get('successor_execution_git_sha')==a.predecessor_sha,'prior Ray migration lineage invalid')
    require(cfg.get('schema')=='openevo.ceiling1-ray-placement-backend.v1' and ready.get('status')=='READY' and ready.get('config_sha256')==sha(a.ray_config),'Ray pool/config identity invalid')
    require(am.get('schema')=='openevo.ceiling1-stage2-carrier-contract-fail-noop-amendment.v1' and am.get('status')=='AUTHORIZED' and a.arm in set(am.get('applies_to_arms') or []),'science amendment invalid/scope drift')
    require(am.get('transition',{}).get('reason')=='NOOP_KEEP_PRIOR_TYPED_CONTRACT_REJECTED_AFTER_REPAIR','science amendment transition drift')
    require(rec.get('status')=='PASS' and rec.get('arm')==a.arm and rec.get('science_amendment_sha256')==sha(a.science_amendment),'carrier recovery receipt invalid')
    if rec.get('schema')=='openevo.ceiling1-stage2-r6-carrier-noop-recovery.v1':
        require(a.arm=='3b' and int(rec.get('round_index',-1))==6 and rec.get('scientific_transition')=='NOOP_KEEP_PRIOR_TYPED_CONTRACT_REJECTED_AFTER_REPAIR','aggregate R6 carrier recovery scope drift')
        carriers=rec.get('carriers') or {};require(set(carriers)=={'text_memory','skill_bundle','agent_system'},'aggregate R6 carrier recovery is incomplete')
        for name,item in carriers.items():
            tp=Path(item['terminal_path']);require(tp.is_file() and sha(tp)==item.get('terminal_sha256'),f'aggregate R6 terminal drift: {name}')
    else:
        require(rec.get('schema')=='openevo.ceiling1-stage2-carrier-contract-fail-noop-recovery.v1','unsupported carrier recovery receipt schema')
    state=load(rayroot/'state.json');profiles=cfg['profiles']
    opposite=[]
    def work_arm(w):
        require(w in profiles,f'unknown Ray workload in live state: {w}')
        return profiles[w]['arm']
    for gpu in state.get('gpus',{}).values():
        for lease in gpu.get('leases',[]):
            arm=work_arm(lease['workload'])
            if arm==a.arm: raise RuntimeError(f'target arm has live Ray lease: {lease["lease_id"]}')
            opposite.append({'lease_id':lease['lease_id'],'workload':lease['workload'],'arm':arm})
    for reservation in (state.get('reservations') or {}).values():
        arm=work_arm(reservation['workload'])
        if arm==a.arm: raise RuntimeError(f'target arm has live Ray reservation: {reservation.get("reservation_id")}')
        opposite.append({'reservation_id':reservation.get('reservation_id'),'workload':reservation['workload'],'arm':arm})
    out=root/'stage2'/a.arm/'formal-vnext-v1';st=load(out/'execution-state.json');nr=int(st['next_round']);cons=int(st['rollouts_consumed']);require(cons==nr*128 and st.get('status') in {'BLOCKED','RUNNING'},'target execution state/accounting invalid')
    rr=out/f'round-{nr:03d}';kind='EXACT_ROUND_BOUNDARY';partial=None;attempt_sha=None;summary_sha=None;rollout_shas=[];artifact_shas=[]
    if (rr/'summary.json').is_file():
        sm=load(rr/'summary.json');require(sm.get('status')=='PASS' and int(sm.get('attempt_count',-1))==128 and int(sm.get('engineering_invalid_count',-1))==0,'partial round rollout invalid')
        kind='PARTIAL_ROUND_AFTER_ROLLOUT';partial=nr;attempt_sha=sha(rr/'attempts.jsonl');summary_sha=sha(rr/'summary.json');rollout_shas=rows_shas(rr/'attempts.jsonl')
        u=rr/'sd-lora/update.json'
        if u.is_file():
            ux=load(u);v=ux.get('experiment_git_sha')
            if v: artifact_shas=[v]
    require(int(rec.get('round_index',-1))==nr and rec.get('state_id')==st.get('state_id'),'carrier recovery receipt round/state drift')
    boundary={'kind':kind,'next_round':nr,'rollouts_consumed':cons,'status':st.get('status'),'blocker':st.get('blocker'),'state_id':st['state_id'],'execution_state_sha256':sha(out/'execution-state.json'),'partial_round':partial,'partial_attempts_sha256':attempt_sha,'partial_summary_sha256':summary_sha,'historical_rollout_execution_shas':rollout_shas,'historical_artifact_execution_shas':artifact_shas,'formal_rollout_replay':0,'carrier_noop_recovery_receipt_path':str(a.carrier_recovery_receipt.resolve()),'carrier_noop_recovery_receipt_sha256':sha(a.carrier_recovery_receipt)}
    parity={}
    for f in SCIENCE_FILES:
        x=gitb(repo,a.predecessor_sha,f);y=gitb(repo,a.successor_sha,f);require(x==y,f'frozen science runtime changed: {f}');parity[f]={'identical':True,'sha256':hashlib.sha256(y).hexdigest()}
    reflector='scripts/openevo_webshop/ceiling1_stage2_reflector.py';old_ref=gitb(repo,a.predecessor_sha,reflector).decode();new_ref=gitb(repo,a.successor_sha,reflector).decode()
    def without_valid(text):
        start=text.index('def valid(');end=text.index('def mechanically_valid(',start);return text[:start]+text[end:]
    require(without_valid(old_ref)==without_valid(new_ref),'reflector changed outside typed-contract validator')
    reflector_change={'path':reflector,'only_valid_function_changed':True,'predecessor_sha256':hashlib.sha256(old_ref.encode()).hexdigest(),'successor_sha256':hashlib.sha256(new_ref.encode()).hexdigest(),'authority':'explicit typed artifact-contract + carrier-NOOP amendment'}
    mig={'schema':'openevo.ceiling1-stage2-resource-migration.ray-continuation-v1','status':'AUTHORIZED','scope':'single-arm-ray-science-continuation','arm':a.arm,'opposite_arm_live_work_allowed':True,'opposite_arm_live_snapshot':opposite,'predecessor_execution_git_sha':a.predecessor_sha,'successor_execution_git_sha':a.successor_sha,'prior_ray_migration_path':str(a.prior_ray_migration.resolve()),'prior_ray_migration_sha256':sha(a.prior_ray_migration),'predecessor_activation_sha256':{a.arm:sha(a.activation.resolve())},'ray_backend_config_path':str(a.ray_config.resolve()),'ray_backend_config_sha256':sha(a.ray_config),'ray_pool_ready_path':str(a.ray_ready.resolve()),'ray_pool_ready_sha256':sha(a.ray_ready),'science_amendment_path':str(a.science_amendment.resolve()),'science_amendment_sha256':sha(a.science_amendment),'boundaries':{a.arm:boundary},'science_runtime_parity':parity,'authorized_reflector_validator_change':reflector_change,'authorized_runtime_changes':['scripts/openevo_webshop/ceiling1_stage2_arm_controller.py','scripts/openevo_webshop/ceiling1_stage2_reflector.py','scripts/openevo_webshop/recover_ceiling1_stage2_carrier_strict_repair_v1.py','scripts/openevo_webshop/ceiling1_stage2_carrier_strict_repair_container_driver.sh'],'accounting':{'formal_rollout_replay':0,'completed_round_rewrite':False,'external_teacher_calls_delta':0,'final_panel_access_delta':0,'scientific_semantics_changed':True,'scientific_semantics_change_authority':'explicit carrier-contract-fail NOOP amendment'},'authorized_at_epoch':time.time()}
    write_once(a.output,mig);print(json.dumps({'status':'PASS','migration_sha256':sha(a.output),'arm':a.arm,'boundary':boundary,'opposite_arm_live_items':len(opposite)},indent=2,sort_keys=True));return 0
if __name__=='__main__': raise SystemExit(main())
