from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path
from scripts.openevo_webshop import autonomous_campaign as ac
import pytest

def write(p:Path,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,sort_keys=True)+"\n")
def git_source(root:Path)->tuple[Path,str]:
    src=root/'src'; src.mkdir(); subprocess.run(['git','init','-q',str(src)],check=True); subprocess.run(['git','-C',str(src),'config','user.email','test@example.com'],check=True); subprocess.run(['git','-C',str(src),'config','user.name','test'],check=True)
    (src/'x').write_text('x'); subprocess.run(['git','-C',str(src),'add','x'],check=True); subprocess.run(['git','-C',str(src),'commit','-qm','x'],check=True)
    return src,subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
def fixture(root:Path,*,phase='stage1b_minimax',arm='a',rows=0,duplicates=False):
    src,head=git_source(root); auth=root/'authority.json'; write(auth,{'status':'FROZEN','minimax':{'model':'MiniMax-M3','thinking_type':'adaptive','reasoning_split':True}})
    progress=root/'progress.jsonl'; data=[]
    for i in range(rows): data.append({'trajectory_id':'x' if duplicates else str(i),'parse_ok':True})
    progress.write_text(''.join(json.dumps(x)+'\n' for x in data))
    release=root/'release.json'; write(release,{'schema':'openevo.autonomous-phase-release.v1','arm':arm,'phase':phase,'activation_authority':True,'execution_source':{'checkout':str(src),'execution_sha':head},'scientific_authority':{'path':str(auth),'sha256':ac.sha(auth),'required_equal':{'status':'FROZEN'}},'budget_contract':{'formal_task_rollouts_max':0,'external_teacher_requests_max':0,'optimizer_steps_max':0,'local_reflector_requests_max':0,'final_panel_tasks_max':0},'completion':{'path':str(root/'complete.json'),'required_equal':{'status':'PASS'}},'progress':{'kind':'jsonl','path':str(progress),'expected_rows':10,'identity_key':'trajectory_id','required_true_key':'parse_ok'},'runtime':{'container_name':'fake'},'retry':{'exact_resume':True},'run_action':{'kind':'command','argv':['true']}})
    manifest={'schema':'openevo.autonomous-campaign.v1','campaign_id':'t','activation_authority':False,'arms':{arm:{'model_family':'qwen2.5','phases':[{'id':phase,'depends_on':[],'release':str(release)}]}}}
    return manifest,release,src,auth,progress

def test_missing_release_is_blocked_science():
    m={'schema':'openevo.autonomous-campaign.v1','campaign_id':'t','arms':{'a':{'phases':[{'id':'x','depends_on':[],'release':None}]}}}
    with tempfile.TemporaryDirectory() as td: assert ac.evaluate(Path(td),m)['arms']['a'][0]['state']=='BLOCKED_SCIENCE'
def test_partial_running_vs_resume_required():
    with tempfile.TemporaryDirectory() as td:
        m,_,_,_,_=fixture(Path(td),rows=3); old=ac.docker_state
        try:
            ac.docker_state=lambda n:{'Running':True,'StartedAt':'now'}
            assert ac.evaluate(Path(td),m)['arms']['a'][0]['state']=='RUNNING'
            ac.docker_state=lambda n:{'Running':False,'ExitCode':137,'OOMKilled':False,'FinishedAt':'now'}
            row=ac.evaluate(Path(td),m)['arms']['a'][0]
            assert row['state']=='RESUME_REQUIRED' and row['progress']['runtime'][0]['oom_killed'] is False
        finally: ac.docker_state=old

def test_duplicate_jsonl_never_auto_repairs():
    with tempfile.TemporaryDirectory() as td:
        m,_,_,_,_=fixture(Path(td),rows=3,duplicates=True); old=ac.docker_state; ac.docker_state=lambda n:None
        try: assert ac.evaluate(Path(td),m)['arms']['a'][0]['state']=='MANUAL_REVIEW_REQUIRED'
        finally: ac.docker_state=old

def test_dirty_source_blocks_before_runtime():
    with tempfile.TemporaryDirectory() as td:
        m,_,src,_,_=fixture(Path(td),rows=0); (src/'dirty').write_text('x')
        assert ac.evaluate(Path(td),m)['arms']['a'][0]['state']=='BLOCKED_SOURCE'

def test_low_disk_gate_blocks_without_launch():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); m,release,_,_,_=fixture(root,rows=0); d=json.loads(release.read_text()); d['resource_gate']={'path':str(root),'min_free_bytes':10**30}; write(release,d); old=ac.docker_state; ac.docker_state=lambda n:None
        try: assert ac.evaluate(root,m)['arms']['a'][0]['state']=='BLOCKED_RESOURCE'
        finally: ac.docker_state=old

def test_input_drift_requires_reconciliation_not_guessing():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); m,release,_,_,_=fixture(root,rows=3); inp=root/'input.json'; write(inp,{'status':'PASS','authority':'new'}); d=json.loads(release.read_text()); d['inputs']=[{'path':str(inp),'required_equal':{'authority':'old'}}]; d['input_reconciliation']={'receipt':{'path':str(root/'binding.json'),'required_equal':{'status':'PASS'}},'action':{'kind':'command','argv':['reconcile']}}; write(release,d)
        row=ac.evaluate(root,m)['arms']['a'][0]; assert row['state']=='ENGINEERING_RETRYABLE' and row['next_action']['argv']==['reconcile']
        write(root/'binding.json',{'status':'PASS'}); old=ac.docker_state; ac.docker_state=lambda n:None
        try: assert ac.evaluate(root,m)['arms']['a'][0]['state']=='RESUME_REQUIRED'
        finally: ac.docker_state=old
def test_qwen3_opsd_requires_explicit_thinking_false():
    manifest={'arms':{'q3':{'model_family':'qwen3'}}}
    ok,why=ac.release_invariant(manifest,'q3','stage1c_opsd',{'budget_contract':{'formal_task_rollouts_max':0,'external_teacher_requests_max':0,'optimizer_steps_max':1,'local_reflector_requests_max':0,'final_panel_tasks_max':0},'model_compat':{}})
    assert not ok and 'enable_thinking=false' in why
    ok,_=ac.release_invariant(manifest,'q3','stage1c_opsd',{'budget_contract':{'formal_task_rollouts_max':0,'external_teacher_requests_max':0,'optimizer_steps_max':1,'local_reflector_requests_max':0,'final_panel_tasks_max':0},'model_compat':{'enable_thinking':False}})
    assert ok

def test_final_eval_requires_two_irreversible_gates():
    manifest={'arms':{'a':{'model_family':'qwen2.5'}}}
    ok,why=ac.release_invariant(manifest,'a','final_evaluation',{})
    assert not ok and 'stage2_seal' in why
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); seal=root/'seal.json'; auth=root/'auth.json'; write(seal,{'status':'PASS'}); write(auth,{'status':'PASS'})
        release={'final_panel_gate':{'stage2_seal':{'path':str(seal),'required_equal':{'status':'PASS'}},'final_panel_authority':{'path':str(auth),'required_equal':{'status':'PASS'}}}}
        assert ac.release_invariant(manifest,'a','final_evaluation',release)[0]

def test_dry_run_never_needs_master_activation_but_real_run_does():
    manifest={'activation_authority':False}; row={'phase':'x','state':'READY','next_action':{'kind':'command','argv':['true']}}
    assert ac.execute_action(manifest,'a',row,True)==0
    try: ac.execute_action(manifest,'a',row,False)
    except ac.CampaignError: pass
    else: raise AssertionError('real action bypassed master activation authority')

def test_dependencies_are_fail_closed():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); m,release,_,_,_=fixture(root,rows=0); first=m['arms']['a']['phases'][0]; m['arms']['a']['phases']=[{'id':'pre','depends_on':[],'release':None},{'id':first['id'],'depends_on':['pre'],'release':str(release)}]
        rows=ac.evaluate(root,m)['arms']['a']; assert rows[0]['state']=='BLOCKED_SCIENCE' and rows[1]['state']=='WAITING_DEPENDENCY'

def test_repo_manifest_keeps_unfrozen_science_slots_empty():
    repo=Path(__file__).parents[1]; p=repo/'configs/experiment/campaigns/202609021800-ceiling1-autonomous-v1.json'; d=json.loads(p.read_text())
    assert d['activation_authority'] is False
    for arm,cfg in d['arms'].items():
        by={x['id']:x for x in cfg['phases']}
        assert by['stage1a_collection']['release'] and by['stage1b_minimax']['release']
        assert by['stage1c_opsd']['release'] is None
        assert by['stage2_vnext']['release'] is None
        if arm=='qwen3-1p7b': assert cfg['required_downstream_chat_template_policy']=='enable_thinking=false for both student and privileged teacher OPSD views'

def test_router_hash_binds_campaign_instance():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); manifest=root/'instance.json'; write(manifest,{'schema':'openevo.autonomous-campaign.v1','campaign_id':'x','arms':{'a':{'phases':[{'id':'p','depends_on':[],'release':None}]}}})
        router=root/'router.json'; write(router,{'schema':'openevo.autonomous-campaign-router.v1','active_manifest':{'path':'instance.json','sha256':ac.sha(manifest),'campaign_id':'x','binding_role':'test'}})
        resolved,meta=ac.resolve_manifest(root,Path('router.json')); assert resolved['campaign_id']=='x' and meta['kind']=='router' and meta['binding_role']=='test'
        manifest.write_text(manifest.read_text()+'\n')
        try: ac.resolve_manifest(root,Path('router.json'))
        except ac.CampaignError as exc: assert 'SHA drift' in str(exc)
        else: raise AssertionError('router accepted target drift')

def test_default_repo_router_points_to_current_timestamped_stage2_authority():
    repo=Path(__file__).parents[1]; router=repo/'configs/experiment/campaigns/current-ceiling1-autonomous.json'; manifest,meta=ac.resolve_manifest(repo,router)
    assert meta['binding_role']=='current_202609030400_stage2_external_execution_adoption'
    assert manifest['campaign_id']=='202609030400-ceiling1-autonomous-v1' and manifest['replaceable_lineage_bindings'] is True
    assert manifest['instance_role']=='current_202609030400_stage2_external_execution_adoption_pending_single_executor_takeover'
    current=meta['repository_current_stage2_runtime_authority']; assert current['schema']=='openevo.ceiling1-stage2-runtime-authority.202609042025' and current['status']=='READY_FOR_AUDITED_NATURAL_MIGRATION'
    old=json.loads((repo/'configs/experiment/campaigns/202609021800-ceiling1-autonomous-v1.json').read_text())
    assert old['instance_role']=='migration_fixture_not_current_scientific_authority'

def test_new_lineage_requires_no_controller_change():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); r=root/'router.json'
        ids=[]
        for i in (1,2):
            m=root/f'instance-{i}.json'; write(m,{'schema':'openevo.autonomous-campaign.v1','campaign_id':f'c{i}','arms':{'a':{'phases':[{'id':'p','depends_on':[],'release':None}]}}})
            write(r,{'schema':'openevo.autonomous-campaign-router.v1','active_manifest':{'path':m.name,'sha256':ac.sha(m),'campaign_id':f'c{i}'}})
            resolved,_=ac.resolve_manifest(root,Path('router.json')); ids.append(resolved['campaign_id'])
        assert ids==['c1','c2']

def test_campaign_lock_blocks_concurrent_controller():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); manifest={'campaign_id':'lock-test','control_state_root':str(root/'state')}
        repo=Path(__file__).parents[1]
        code="from scripts.openevo_webshop import autonomous_campaign as ac; m={'campaign_id':'lock-test','control_state_root':r'%s'}; import sys;\ntry:\n  with ac.campaign_lock(m): pass\nexcept ac.CampaignError as e:\n  print(str(e),file=sys.stderr); sys.exit(7)" % str(root/'state')
        with ac.campaign_lock(manifest):
            env=os.environ.copy(); env['PYTHONPATH']=str(repo)
            r=subprocess.run([sys.executable,'-c',code],cwd=repo,env=env,text=True,capture_output=True)
            assert r.returncode==7 and 'lock is already held' in r.stderr

def test_transition_journal_is_append_only():
    with tempfile.TemporaryDirectory() as td:
        manifest={'campaign_id':'j','control_state_root':str(Path(td)/'state')}
        p=ac.append_transition_journal(manifest,{'event':'ACTION_STARTED','action_id':'a'})
        ac.append_transition_journal(manifest,{'event':'ACTION_FINISHED','action_id':'a','exit_code':0})
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        assert [x['event'] for x in rows]==['ACTION_STARTED','ACTION_FINISHED']
        assert all(x['campaign_id']=='j' for x in rows)

def test_router_and_instance_are_two_execution_keys():
    manifest={'activation_authority':True}
    try: ac.assert_campaign_execution_authority(manifest,{'kind':'router','router_execution_activation_authority':False})
    except ac.CampaignError as exc: assert 'router has no execution activation authority' in str(exc)
    else: raise AssertionError('router execution gate bypassed')
    ac.assert_campaign_execution_authority(manifest,{'kind':'router','router_execution_activation_authority':True})

def test_migration_fixture_has_durable_control_state_root():
    repo=Path(__file__).parents[1]; d=json.loads((repo/'configs/experiment/campaigns/202609021800-ceiling1-autonomous-v1.json').read_text())
    assert d['control_state_root'].endswith('/202609021800-ceiling1-autonomous-v1')
    assert 'Exactly one real campaign controller' in d['controller_concurrency_contract']

def test_phase_release_core_contract_fails_closed():
    base={'schema':'openevo.autonomous-phase-release.v1','arm':'a','phase':'p','activation_authority':False,'execution_source':{'checkout':'/x','execution_sha':'0'*40},'scientific_authority':{'path':'/a','sha256':'1'*64,'required_equal':{'status':'PASS'}},'budget_contract':{'formal_task_rollouts_max':0,'external_teacher_requests_max':0,'optimizer_steps_max':0,'local_reflector_requests_max':0,'final_panel_tasks_max':0},'completion':{'path':'/c','required_equal':{'status':'PASS'}},'retry':{'exact_resume':True}}
    assert ac.validate_release_structure(base)[0]
    bad=dict(base); bad.pop('budget_contract'); ok,why=ac.validate_release_structure(bad); assert not ok and 'budget_contract' in why
    bad=json.loads(json.dumps(base)); bad['execution_source']['execution_sha']='abc'; ok,why=ac.validate_release_structure(bad); assert not ok and '40-hex' in why
    bad=json.loads(json.dumps(base)); bad['scientific_authority']['sha256']='x'*64; ok,why=ac.validate_release_structure(bad); assert not ok and '64-hex' in why

def test_activated_release_requires_explicit_launcher():
    r={'schema':'openevo.autonomous-phase-release.v1','arm':'a','phase':'p','activation_authority':True,'execution_source':{'checkout':'/x','execution_sha':'0'*40},'scientific_authority':{'path':'/a','sha256':'1'*64,'required_equal':{'status':'PASS'}},'budget_contract':{'formal_task_rollouts_max':0,'external_teacher_requests_max':0,'optimizer_steps_max':0,'local_reflector_requests_max':0,'final_panel_tasks_max':0},'completion':{'path':'/c','required_equal':{'status':'PASS'}},'retry':{'exact_resume':True}}
    ok,why=ac.validate_release_structure(r); assert not ok and 'run_action' in why
    r['run_action']={'kind':'command','argv':['true']}; assert ac.validate_release_structure(r)[0]

def test_repo_releases_satisfy_core_contract():
    repo=Path(__file__).parents[1]
    for root in (repo/'configs/experiment/campaigns/202609021800/releases', repo/'configs/experiment/campaigns/202609030400/releases'):
        for p in root.glob('*.json'):
            ok,why=ac.validate_release_structure(json.loads(p.read_text())); assert ok,(p.name,why)

def test_ceiling_program_contract_freezes_arms_and_full_dag():
    repo=Path(__file__).parents[1]; src=json.loads((repo/'configs/experiment/campaigns/202609021800-ceiling1-autonomous-v1.json').read_text())
    ac.validate_manifest(repo,src)
    bad=json.loads(json.dumps(src)); bad['arms']['qwen25-3b']['phases'].pop(4)
    try: ac.validate_manifest(repo,bad)
    except ac.CampaignError as exc: assert 'DAG drift' in str(exc)
    else: raise AssertionError('Ceiling DAG phase deletion was accepted')
    bad=json.loads(json.dumps(src)); bad['arms']['qwen3-1p7b']['model_revision']='wrong'
    try: ac.validate_manifest(repo,bad)
    except ac.CampaignError as exc: assert 'model_revision drift' in str(exc)
    else: raise AssertionError('Ceiling model revision drift was accepted')

def test_router_program_contract_is_hash_bound():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); m=root/'m.json'; write(m,{'schema':'openevo.autonomous-campaign.v1','campaign_id':'x','program_contract':'A','arms':{'a':{'phases':[{'id':'p','depends_on':[],'release':None}]}}})
        r=root/'r.json'; write(r,{'schema':'openevo.autonomous-campaign-router.v1','active_manifest':{'path':'m.json','sha256':ac.sha(m),'campaign_id':'x','program_contract':'B'}})
        try: ac.resolve_manifest(root,Path('r.json'))
        except ac.CampaignError as exc: assert 'program_contract binding drift' in str(exc)
        else: raise AssertionError('router accepted wrong program contract')

def test_ceiling_phase_budget_boundaries_are_fixed():
    repo=Path(__file__).parents[1]; manifest=json.loads((repo/'configs/experiment/campaigns/202609021800-ceiling1-autonomous-v1.json').read_text())
    r=json.loads((repo/'configs/experiment/campaigns/202609021800/releases/qwen25-3b-stage1a.json').read_text()); assert ac.release_invariant(manifest,'qwen25-3b','stage1a_collection',r)[0]
    r['budget_contract']['formal_task_rollouts_max']=1439; ok,why=ac.release_invariant(manifest,'qwen25-3b','stage1a_collection',r); assert not ok and '1440' in why
    stage2={'budget_contract':{'formal_task_rollouts_max':20479,'external_teacher_requests_max':0,'optimizer_steps_max':1,'local_reflector_requests_max':1,'final_panel_tasks_max':0}}
    ok,why=ac.release_invariant(manifest,'qwen25-3b','stage2_vnext',stage2); assert not ok and '20,480' in why
    final={'budget_contract':{'formal_task_rollouts_max':0,'external_teacher_requests_max':0,'optimizer_steps_max':0,'local_reflector_requests_max':0,'final_panel_tasks_max':127},'final_panel_gate':{}}
    ok,why=ac.release_invariant(manifest,'qwen25-3b','final_evaluation',final); assert not ok and '128' in why

def _ceiling_budget_for_phase(phase):
    z={'formal_task_rollouts_max':0,'external_teacher_requests_max':0,'optimizer_steps_max':0,'local_reflector_requests_max':0,'final_panel_tasks_max':0}
    if phase=='stage1a_collection': z['formal_task_rollouts_max']=1440
    elif phase=='stage1b_minimax': z['external_teacher_requests_max']=1440
    elif phase=='stage1c_opsd': z['optimizer_steps_max']=1
    elif phase in {'stage1d_text_memory','stage1e_skill_bundle','stage1f_agent_system'}: z['local_reflector_requests_max']=1
    elif phase=='stage2_vnext': z.update(formal_task_rollouts_max=20480,optimizer_steps_max=1,local_reflector_requests_max=1)
    elif phase=='final_evaluation': z['final_panel_tasks_max']=128
    return z

def test_full_two_arm_engineering_dag_runs_to_completion_without_science():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); src,head=git_source(root); authority=root/'authority.json'; write(authority,{'status':'FROZEN'})
        final_auth=root/'final-authority.json'; write(final_auth,{'status':'PASS'})
        arms={}
        for arm,contract in ac.CEILING1_ARM_CONTRACT.items():
            phases=[]; previous=None; stage2_completion=None
            for phase,deps in ac.CEILING1_PHASE_DAG:
                release=root/f'{arm}-{phase}-release.json'; completion=root/f'{arm}-{phase}-complete.json'
                action_code=f"import json,pathlib; pathlib.Path({str(completion)!r}).write_text(json.dumps({{'status':'PASS'}})+'\\n')"
                r={'schema':'openevo.autonomous-phase-release.v1','arm':arm,'phase':phase,'activation_authority':True,'execution_source':{'checkout':str(src),'execution_sha':head},'scientific_authority':{'path':str(authority),'sha256':ac.sha(authority),'required_equal':{'status':'FROZEN'}},'budget_contract':_ceiling_budget_for_phase(phase),'completion':{'path':str(completion),'required_equal':{'status':'PASS'}},'retry':{'exact_resume':True},'run_action':{'kind':'command','argv':[sys.executable,'-c',action_code]},'asset_contract':{'registry_policy':'observation-first-canonical-if-proven','consumes':[],'produces':[{'canonical_id':f'asset-{arm}-{phase}','role':'synthetic-output'}],'recovery':[],'telemetry':[],'publication':[]}}
                if phase=='stage1c_opsd' and contract['model_family']=='qwen3': r['model_compat']={'enable_thinking':False}
                if phase=='stage2_vnext':
                    r['engineering_successor']={'interface_owned_action_opener':True,'applies_to_both_arms':True,'shared_action_envelope_sha256':'0'*64}
                    if contract['model_family']=='qwen3': r['model_compat']={'enable_thinking':False}
                if phase=='final_evaluation':
                    assert stage2_completion is not None
                    r['final_panel_gate']={'stage2_seal':{'path':str(stage2_completion),'required_equal':{'status':'PASS'}},'final_panel_authority':{'path':str(final_auth),'required_equal':{'status':'PASS'}}}
                write(release,r); phases.append({'id':phase,'depends_on':list(deps),'release':str(release)})
                if phase=='stage2_seal': stage2_completion=completion
                previous=phase
            arms[arm]={**contract,'phases':phases}
        runtime_authority=root/'stage2-runtime-authority-202609042230.json'; write(runtime_authority,{'schema':'test.stage2-runtime-authority.202609042230','status':'PASS'})
        runtime_sha=ac.sha(runtime_authority)
        manifest={'schema':'openevo.autonomous-campaign.v1','program_contract':ac.CEILING1_PROGRAM_CONTRACT,'campaign_id':'engineering-full-dag','activation_authority':True,'control_state_root':str(root/'control-state'),'arms':arms,'repository_current_stage2_runtime_authority':{'path':str(runtime_authority),'sha256':runtime_sha},'bindings':{'shared_stage2_runtime_authority_path':str(runtime_authority),'shared_stage2_runtime_authority_sha256':runtime_sha}}  # legacy-revision-compat
        mp=root/'campaign.json'; write(mp,manifest)
        router=root/'router.json'; write(router,{'schema':'openevo.autonomous-campaign-router.v1','execution_activation_authority':True,'active_manifest':{'path':str(mp),'sha256':ac.sha(mp),'campaign_id':'engineering-full-dag','program_contract':ac.CEILING1_PROGRAM_CONTRACT},'repository_current_stage2_runtime_authority':{'path':str(runtime_authority),'sha256':runtime_sha,'schema':'test.stage2-runtime-authority.202609042230','status':'PASS'}})  # legacy-revision-compat
        rc=ac.main(['run','--manifest',str(router),'--repo-root',str(root),'--until-blocked'])
        assert rc==0
        status=ac.evaluate(root,manifest); assert status['campaign_complete'] is True
        assert all(all(row['state']=='COMPLETE' for row in rows) for rows in status['arms'].values())
        journal=root/'control-state'/'TRANSITIONS.jsonl'; records=[json.loads(x) for x in journal.read_text().splitlines() if x.strip()]
        expected_actions=len(ac.CEILING1_PHASE_DAG)*len(ac.CEILING1_ARM_CONTRACT)
        assert len(records)==expected_actions*2
        assert sum(x['event']=='ACTION_STARTED' for x in records)==expected_actions and sum(x['event']=='ACTION_FINISHED' for x in records)==expected_actions

def test_orphan_transition_ready_state_blocks_relaunch():
    with tempfile.TemporaryDirectory() as td:
        manifest={'campaign_id':'orphan','control_state_root':str(Path(td)/'state')}
        ac.append_transition_journal(manifest,{'event':'ACTION_STARTED','action_id':'x','arm':'a','phase':'p'})
        health=ac.transition_journal_health(manifest,{'arms':{'a':[{'phase':'p','state':'READY'}]}})
        assert health['status']=='BLOCKED' and health['blocked_actions'][0]['state_now']=='READY'

def test_orphan_transition_with_resume_evidence_auto_reconciles():
    with tempfile.TemporaryDirectory() as td:
        manifest={'campaign_id':'orphan','control_state_root':str(Path(td)/'state')}
        ac.append_transition_journal(manifest,{'event':'ACTION_STARTED','action_id':'x','arm':'a','phase':'p'})
        status={'arms':{'a':[{'phase':'p','state':'RESUME_REQUIRED'}]}}
        health=ac.transition_journal_health(manifest,status); assert health['status']=='RECOVERABLE'
        ac.reconcile_recoverable_journal(manifest,health)
        assert ac.transition_journal_health(manifest,status)['status']=='PASS'
        rows=[json.loads(x) for x in (Path(td)/'state'/'TRANSITIONS.jsonl').read_text().splitlines() if x.strip()]
        assert rows[-1]['event']=='ACTION_RECONCILED' and rows[-1]['scientific_semantics_changed'] is False

def test_corrupt_transition_journal_blocks_execution():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)/'state'; root.mkdir(); (root/'TRANSITIONS.jsonl').write_text('{bad json\n')
        health=ac.transition_journal_health({'campaign_id':'c','control_state_root':str(root)},{'arms':{}})
        assert health['status']=='BLOCKED' and 'unreadable' in health['reason']


def test_glob_progress_and_multi_container_runtime_group():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); src,head=git_source(root); auth=root/'authority.json'; write(auth,{'status':'FROZEN'})
        ep=root/'episodes'; ep.mkdir(); [write(ep/f'{i:06d}-x.json',{'i':i}) for i in (1,2,3)]
        release=root/'release.json'; write(release,{'schema':'openevo.autonomous-phase-release.v1','arm':'a','phase':'p','activation_authority':False,'execution_source':{'checkout':str(src),'execution_sha':head},'scientific_authority':{'path':str(auth),'sha256':ac.sha(auth),'required_equal':{'status':'FROZEN'}},'budget_contract':{'formal_task_rollouts_max':0,'external_teacher_requests_max':0,'optimizer_steps_max':0,'local_reflector_requests_max':0,'final_panel_tasks_max':0},'completion':{'path':str(root/'done.json'),'required_equal':{'status':'PASS'}},'progress':{'kind':'glob_files','path_glob':str(ep/'*.json'),'expected_files':4,'identity_regex':r'^(\d{6})-'},'runtime':{'container_names':['w1','w2']},'retry':{'exact_resume':False}})
        manifest={'schema':'openevo.autonomous-campaign.v1','campaign_id':'x','arms':{'a':{'model_family':'qwen2.5','phases':[{'id':'p','depends_on':[],'release':str(release)}]}}}
        old=ac.docker_state
        try:
            ac.docker_state=lambda n: {'Running':n=='w2','ExitCode':0,'OOMKilled':False,'StartedAt':'x','FinishedAt':'x'}
            row=ac.evaluate(root,manifest)['arms']['a'][0]
            assert row['state']=='RUNNING' and row['progress']['files']==3 and row['progress']['runtime_running_count']==1 and row['progress']['runtime_expected_count']==2
        finally: ac.docker_state=old

def test_current_202609030400_instance_is_v2_prestage2_handoff():
    repo=Path(__file__).parents[1]
    manifest=json.loads((repo/'configs/experiment/campaigns/202609030400-ceiling1-autonomous-v1.json').read_text())  # legacy-revision-compat
    assert manifest['program_contract']==ac.CEILING1_PROGRAM_CONTRACT_V2
    assert manifest['activation_authority'] is False
    ac.validate_manifest(repo,manifest)
    for arm,cfg in manifest['arms'].items():
        by={p['id']:p for p in cfg['phases']}
        for phase in ('stage1a_collection','stage1a_postcollection_audit','stage1b_minimax','stage1b_teacher_evidence','stage1c_opsd','stage1d_text_memory','stage1e_skill_bundle','stage1f_agent_system','stage1g_closeout','stage2_vnext'):
            assert by[phase]['release']
        rel=json.loads((repo/by['stage2_vnext']['release']).read_text())
        assert rel['activation_authority'] is False and rel['external_executor']['read_only_adoption'] is True
        assert rel['engineering_successor']['shared_action_envelope_sha256']=='d15356e7a1e1843219c1a2753bb6bef2714011dc61b21fb9e41ef7c4a01b8335'
    q=json.loads((repo/{p['id']:p for p in manifest['arms']['qwen3-1p7b']['phases']}['stage2_vnext']['release']).read_text())
    assert q['model_compat']['enable_thinking'] is False and q['execution_source']['execution_sha']=='ac130148ee08b6462d9728e482a858fe5f514047'
    assert q['engineering_successor']['current_runtime_authority']['schema']=='openevo.ceiling1-stage2-runtime-authority.202609042025'
    assert any(i.get('sha256')=='bcb34f2db001a320ac3228790a080a691658fe2ae04f59bf662f0cb0ab6a1a7d' for i in q['inputs'])
    b=json.loads((repo/{p['id']:p for p in manifest['arms']['qwen25-3b']['phases']}['stage2_vnext']['release']).read_text())
    assert b['execution_source']['execution_sha']=='4915dbdd75678da9a5752f991e70685c67dd497b'
    assert b['engineering_successor']['current_runtime_authority']['schema']=='openevo.ceiling1-stage2-runtime-authority.202609042025'
    assert any(i.get('sha256')=='e1744035504318712a847fc464a960b0a2fa39c755e972e74fc18ab5143bb104' for i in b['inputs'])

def test_completed_current_adoption_releases_do_not_require_retired_execution_checkouts():
    repo=Path(__file__).parents[1]
    manifest=json.loads((repo/'configs/experiment/campaigns/202609030400-ceiling1-autonomous-v1.json').read_text())  # legacy-revision-compat
    for arm,cfg in manifest['arms'].items():
        for phase in cfg['phases']:
            if phase['id']=='stage2_vnext' or not phase.get('release'):  # legacy-revision-compat
                continue
            rel=json.loads((repo/phase['release']).read_text())
            assert rel['release_mode']=='adoption'
            assert 'execution_source' not in rel
            ok,detail=ac.validate_authority(repo,rel); assert ok,detail

def test_v1_migration_fixture_contract_remains_unchanged():
    repo=Path(__file__).parents[1]
    manifest=json.loads((repo/'configs/experiment/campaigns/202609021800-ceiling1-autonomous-v1.json').read_text())
    assert manifest['program_contract']==ac.CEILING1_PROGRAM_CONTRACT_V1
    ac.validate_manifest(repo,manifest)
    for arm,cfg in manifest['arms'].items():
        assert [(x['id'],x.get('depends_on') or []) for x in cfg['phases']]==ac.CEILING1_PHASE_DAG_V1

def test_v2_empty_release_must_have_explicit_gate():
    manifest={'schema':'openevo.autonomous-campaign.v1','program_contract':ac.CEILING1_PROGRAM_CONTRACT_V2,'campaign_id':'x','activation_authority':False,'arms':{}}
    for arm,contract in ac.CEILING1_ARM_CONTRACT.items():
        phases=[]
        for phase,deps in ac.CEILING1_PHASE_DAG_V2:
            row={'id':phase,'depends_on':deps,'release':None,'gate':{'state':'BLOCKED_UPSTREAM','reason':'test'}}
            phases.append(row)
        manifest['arms'][arm]={**contract,'phases':phases}
    ac.validate_manifest(Path('.'),manifest)
    del manifest['arms']['qwen25-3b']['phases'][0]['gate']
    with pytest.raises(ac.CampaignError,match='explicit gate'):
        ac.validate_manifest(Path('.'),manifest)

def test_external_executor_adoption_is_read_only_running_and_missing_pause():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); src,head=git_source(root); auth=root/'authority.json'; write(auth,{'status':'FROZEN'})
        state=root/'external-state.json'; write(state,{'schema':'external.v1','status':'RUNNING','arm':'a','rollouts_consumed':128,'next_round':1})
        release=root/'release.json'; write(release,{'schema':'openevo.autonomous-phase-release.v1','arm':'a','phase':'p','activation_authority':False,'execution_source':{'checkout':str(src),'execution_sha':head},'scientific_authority':{'path':str(auth),'sha256':ac.sha(auth),'required_equal':{'status':'FROZEN'}},'budget_contract':{'formal_task_rollouts_max':20480,'external_teacher_requests_max':0,'optimizer_steps_max':0,'local_reflector_requests_max':0,'final_panel_tasks_max':0},'completion':{'path':str(root/'done.json'),'required_equal':{'status':'PASS'}},'retry':{'exact_resume':True},'external_executor':{'read_only_adoption':True,'state_path':str(state),'required_equal':{'schema':'external.v1','arm':'a'},'status_map':{'RUNNING':'RUNNING','BLOCKED':'PAUSED'},'missing_state':'PAUSED','missing_reason':'external executor has not started'}})
        manifest={'schema':'openevo.autonomous-campaign.v1','campaign_id':'x','arms':{'a':{'model_family':'qwen2.5','phases':[{'id':'p','depends_on':[],'release':str(release)}]}}}
        row=ac.evaluate(root,manifest)['arms']['a'][0]; assert row['state']=='RUNNING' and row['progress']['rollouts_consumed']==128
        state.unlink(); row=ac.evaluate(root,manifest)['arms']['a'][0]; assert row['state']=='PAUSED'
        r=json.loads(release.read_text()); r['activation_authority']=True; write(release,r); ok,why=ac.validate_release_structure(r); assert not ok and 'external executor adoption' in why
