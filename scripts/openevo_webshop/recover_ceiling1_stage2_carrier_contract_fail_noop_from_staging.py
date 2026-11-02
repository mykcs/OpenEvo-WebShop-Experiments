#!/usr/bin/env python3
from __future__ import annotations
import argparse,ast,hashlib,json,os,shutil,subprocess,time,sys
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.openevo_webshop.ceiling1_stage2_arm_controller import (
    REFLECTOR_CONTRACT_FAIL_NOOP_REASON, build_reflector_contract_fail_noop,
    reflector_attempt_identity, load, sha,
)
CONTRACTS={'text_memory':'harness201-runtime-memory-v1','skill_bundle':'harness201-runtime-skill-v1','agent_system':'harness201-runtime-agent-v1'}
DIRS={'text_memory':'text-memory','skill_bundle':'skill-bundle','agent_system':'agent-system'}
NAMES={'text_memory':'memory.md','skill_bundle':'SKILL.md','agent_system':'agent_system.md'}
def write_once(p:Path,x:dict):
    text=json.dumps(x,indent=2,sort_keys=True)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_text()!=text: raise RuntimeError(f'write-once recovery receipt drift: {p}')
        return
    with p.open('x') as f: f.write(text);f.flush();os.fsync(f.fileno())
def atomic(p:Path,x:dict):
    t=p.with_name(f'.{p.name}.tmp-{os.getpid()}');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def docker_json(name:str,fmt:str):
    q=subprocess.run(['docker','inspect',name,'--format',fmt],text=True,capture_output=True)
    require(q.returncode==0,f'recovery source container missing: {name}')
    return json.loads(q.stdout)
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--science-amendment',type=Path,required=True);p.add_argument('--execution-git-sha',required=True);p.add_argument('--arm',choices=('3b','qwen3-1p7b'),required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--kind',choices=tuple(CONTRACTS),required=True);p.add_argument('--staging',type=Path,required=True);p.add_argument('--container',required=True);p.add_argument('--staging-source-git-sha');a=p.parse_args()
    repo=a.repo.resolve();root=a.root.resolve();staging=a.staging.resolve();amend=a.science_amendment.resolve();contract=CONTRACTS[a.kind]
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    require(head==a.execution_git_sha and not dirty,'recovery repo is not exact clean successor')
    ax=load(amend);require(ax.get('schema')=='openevo.ceiling1-stage2-carrier-contract-fail-noop-amendment.v1' and ax.get('status')=='AUTHORIZED','carrier-NOOP amendment invalid')
    require(a.arm in set(ax.get('applies_to_arms') or []) and a.kind in set(ax.get('applies_to_carriers') or []),'carrier-NOOP amendment scope drift')
    require(ax.get('transition',{}).get('reason')==REFLECTOR_CONTRACT_FAIL_NOOP_REASON,'carrier-NOOP amendment reason drift')
    require(int(ax.get('unchanged',{}).get('max_primary_calls',-1))==1 and int(ax.get('unchanged',{}).get('max_repair_calls',-1))==1,'carrier-NOOP call budget drift')
    out=root/'stage2'/a.arm/'formal-vnext-v1';rr=out/f'round-{a.round:03d}';state_path=out/'execution-state.json';state_sha_before=sha(state_path);state=load(state_path);before=load(rr/'state-before.json');summary=load(rr/'summary.json');evidence=rr/'evidence/evidence.json';ev=load(evidence)
    require(int(state.get('next_round',-1))==a.round and state.get('state_id')==before.get('state_id'),'recovery state/round identity drift')
    require(summary.get('status')=='PASS' and int(summary.get('attempt_count',-1))==128 and int(summary.get('engineering_invalid_count',-1))==0 and summary.get('state_id')==before.get('state_id'),'recovery rollout summary invalid')
    require(bool((ev.get('gates') or {}).get(a.kind,{}).get('eligible')),'carrier gate is not eligible')
    prior=Path(before[{'text_memory':'memory','skill_bundle':'skill','agent_system':'agent_system'}[a.kind]])
    sd=Path(before['current_sd_adapter']) if before.get('current_sd_adapter') else None
    memory=Path(before['memory']) if a.kind=='agent_system' else None;skill=Path(before['skill']) if a.kind=='agent_system' else None
    require(prior.is_file(),'prior carrier missing')
    rejected_path=staging/'rejected-candidates.json';require(rejected_path.is_file(),'staging rejected-candidates missing');rej=load(rejected_path);attempts=list(rej.get('attempts') or [])
    require(rej.get('schema')=='openevo.ceiling1-stage2-reflector-rejected-candidates.v1' and rej.get('kind')==a.kind and int(rej.get('round_index',-1))==a.round and rej.get('state_id')==before.get('state_id'),'staging rejected identity drift')
    require([x.get('attempt') for x in attempts]==['primary','repair'] and all(x.get('artifact_contract_valid') is False for x in attempts),'staging does not contain exact primary+repair contract rejection')
    for x in attempts:
        fp=staging/f"rejected-{x['attempt']}.md";require(fp.is_file() and sha(fp)==x.get('candidate_sha256'),'staging rejected payload hash drift')
    st=docker_json(a.container,'{{json .State}}');require(st.get('Status')=='exited' and int(st.get('ExitCode',0))==1 and st.get('OOMKilled') is False,'staging container is not an exited non-OOM contract failure')
    cmd=docker_json(a.container,'{{json .Config.Cmd}}');joined='\x00'.join(map(str,cmd));expected_output=staging/NAMES[a.kind]
    source_scripts=[Path(str(x)) for x in cmd if str(x).endswith('/scripts/openevo_webshop/ceiling1_stage2_reflector.py')]
    require(len(source_scripts)==1 and source_scripts[0].is_file(),'staging reflector source path missing/ambiguous')
    source_script=source_scripts[0];successor_script=repo/'scripts/openevo_webshop/ceiling1_stage2_reflector.py'
    if sha(source_script)!=sha(successor_script):
        require(a.kind=='text_memory' and bool(a.staging_source_git_sha),'non-identical staging reflector is allowed only for Text Memory with explicit source SHA')
        source_repo=source_script.parents[2];observed_source_sha=subprocess.check_output(['git','-C',str(source_repo),'rev-parse','HEAD'],text=True).strip()
        require(observed_source_sha==a.staging_source_git_sha,'staging reflector source git SHA drift')
        git_bytes=subprocess.check_output(['git','-C',str(source_repo),'show',f'{a.staging_source_git_sha}:scripts/openevo_webshop/ceiling1_stage2_reflector.py'])
        require(hashlib.sha256(git_bytes).hexdigest()==sha(source_script),'staging reflector working bytes drift from source SHA')
        old_text=source_script.read_text();new_text=successor_script.read_text();old_tree=ast.parse(old_text);new_tree=ast.parse(new_text)
        def fn(tree,name): return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
        for name in ('system_prompt','render','mechanically_valid','main'):
            require(ast.dump(fn(old_tree,name),include_attributes=False)==ast.dump(fn(new_tree,name),include_attributes=False),f'Text Memory staging path changed in {name}')
        def memory_if(tree):
            f=fn(tree,'valid')
            return next(n for n in f.body if isinstance(n,ast.If) and 'text_memory' in ast.dump(n.test,include_attributes=False))
        require(ast.dump(memory_if(old_tree),include_attributes=False)==ast.dump(memory_if(new_tree),include_attributes=False),'Text Memory strict-validation branch changed')
        from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import validate_carrier
        for item in attempts:
            require(validate_carrier('text_memory',(staging/f"rejected-{item['attempt']}.md").read_text()).get('status')=='FAIL','staging candidate is no longer strict-invalid under successor validator')
    checks=['--kind\x00'+a.kind, '--evidence\x00'+str(evidence), '--prior\x00'+str(prior), '--output\x00'+str(expected_output), '--state-id\x00'+before['state_id'], '--round-index\x00'+str(a.round), '--artifact-contract\x00'+contract]
    require(all(x in joined for x in checks),'staging container command does not bind exact correct-contract recovery inputs')
    selection=load(staging/'ray-selection.json');uuids=selection.get('selected_gpu_uuids') or [];require(len(uuids)==1 and ('--gpu-uuid\x00'+uuids[0]) in joined,'staging GPU selection/container identity drift')
    attempt=reflector_attempt_identity(a.kind,a.round,before['state_id'],evidence,prior,memory,skill,sd,contract);attempt_path=staging/'model-call-attempt.reconstructed.json'
    write_once(attempt_path,attempt)
    terminal=build_reflector_contract_fail_noop(a.kind,a.round,before['state_id'],evidence,prior,memory,skill,sd,staging,attempt_path,contract);require(terminal is not None,'failed to materialize authorized carrier NOOP')
    canonical=rr/DIRS[a.kind];oldterm=canonical/'terminal-receipt.json';require(oldterm.is_file(),'canonical predecessor terminal missing');old=load(oldterm);old_sha=sha(oldterm)
    old_art=canonical/NAMES[a.kind];old_rec=old_art.with_suffix(old_art.suffix+'.receipt.json');require(old_art.is_file() and old_rec.is_file(),'superseded wrong-contract artifact/receipt missing')
    require((old.get('reflector_receipt') or {}).get('artifact_contract')=='legacy-authoring-v1','canonical predecessor is not the expected wrong-contract terminal')
    archive=canonical/f'terminal-receipt.superseded-wrong-contract-{old_sha[:12]}.json'
    if archive.exists(): require(sha(archive)==old_sha,'superseded terminal archive drift')
    else: shutil.copy2(oldterm,archive)
    recovery={'schema':'openevo.ceiling1-stage2-carrier-contract-fail-noop-recovery.v1','status':'PASS','arm':a.arm,'round_index':a.round,'kind':a.kind,'state_id':before['state_id'],'science_amendment_path':str(amend),'science_amendment_sha256':sha(amend),'recovery_execution_git_sha':a.execution_git_sha,'artifact_contract':contract,'staging_path':str(staging),'staging_rejected_candidates_sha256':sha(rejected_path),'staging_primary_sha256':sha(staging/'rejected-primary.md'),'staging_repair_sha256':sha(staging/'rejected-repair.md'),'staging_container':a.container,'staging_container_exit_code':1,'staging_oom_killed':False,'staging_ray_selection_sha256':sha(staging/'ray-selection.json'),'reconstructed_attempt_path':str(attempt_path),'reconstructed_attempt_sha256':sha(attempt_path),'reconstructed_from_exact_runtime_container_evidence':True,'superseded_terminal_path':str(archive),'superseded_terminal_sha256':old_sha,'superseded_artifact_path':str(old_art),'superseded_artifact_sha256':sha(old_art),'superseded_reflector_receipt_path':str(old_rec),'superseded_reflector_receipt_sha256':sha(old_rec),'new_terminal_status':'NOOP','new_terminal_reason':REFLECTOR_CONTRACT_FAIL_NOOP_REASON,'new_active_carrier_path':str(prior),'new_active_carrier_sha256':sha(prior),'model_calls_already_consumed':2,'new_model_calls':0,'formal_rollout_replay':0,'external_teacher_calls_delta':0,'final_panel_access_delta':0,'execution_state_changed':False,'recovered_at_epoch':time.time()}
    # Install terminal, then bind recovery receipt to installed terminal.
    atomic(oldterm,terminal);recovery['new_terminal_sha256']=sha(oldterm);recpath=canonical/'CONTRACT_FAIL_NOOP_RECOVERY.json';write_once(recpath,recovery)
    require(sha(state_path)==state_sha_before,'execution-state changed during carrier-NOOP recovery')
    print(json.dumps({'status':'PASS','terminal_sha256':sha(oldterm),'recovery_receipt':str(recpath),'recovery_receipt_sha256':sha(recpath),'active_carrier_sha256':sha(prior)},sort_keys=True));return 0
if __name__=='__main__': raise SystemExit(main())
