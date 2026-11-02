import json, re
from pathlib import Path
from scripts.governance.build_server_run_passports import extract_root, infer_lifecycle

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'schemas/server-run-passport.schema.json'
CLOSEOUT=ROOT/'docs/evidence/server-audits/2026-09-03/cross-platform-asset-governance-dry-run/RUN_PASSPORT_BACKFILL_CLOSEOUT.json'
STANDARD=ROOT/'docs/operations/governance/CROSS_PLATFORM_EXPERIMENT_ASSET_STANDARD.md'
MASTER=ROOT/'docs/experiment-tracking/OPEN_EVO_CEILING_1.0_END_TO_END_MASTER_PLAN.md'
PROMPT=ROOT/'docs/agents/handoffs/CEILING1_AUTONOMOUS_CAMPAIGN_FRESH_AGENT_PROMPT.md'

def test_run_passport_schema_freezes_identity_and_resolution_fields():
    s=json.loads(SCHEMA.read_text())
    req=set(s['required'])
    for key in {'passport_id','provider_observation_id','canonical_root_id','canonical_run_id','root_role','canonical_resolution','field_resolution','retention','evidence','resolution_action'}:
        assert key in req
    assert s['properties']['passport_status']['const']=='ISSUED'
    assert s['properties']['retention']['properties']['safe_to_delete']['const'] is False

def test_public_closeout_proves_current_server_coverage():
    c=json.loads(CLOSEOUT.read_text())
    assert c['status']=='PASS'
    assert c['all_server_run_roots_have_passport'] is True
    assert c['server_run_root_count']==244==c['passport_count']
    assert c['unique_passport_ids']==244
    assert c['unique_canonical_root_ids']==244
    assert c['unique_observation_ids']==244

def test_closeout_preserves_unresolved_without_mutation_or_delete_authority():
    c=json.loads(CLOSEOUT.read_text())
    assert c['resolution_counts']['LEGACY_UNCERTAIN'] > 0
    assert c['resolution_counts']['PARTIAL'] > 0
    assert c['safety']=={
        'automatic_delete_authorizations':0,'provider_deletes':0,'provider_renames':0,
        'scientific_budget_consumed':0,'scientific_reruns':0,'secret_contents_read':False,
    }
    for digest in c['private_evidence_hashes'].values():
        assert re.fullmatch(r'[0-9a-f]{64}',digest)

def test_run_passport_is_a_normative_long_term_rule():
    text=STANDARD.read_text()
    assert '### 10.3 Server Run Passport' in text
    assert '`canonical_root_id`' in text
    assert '`LEGACY_UNCERTAIN`' in text
    assert 'passport coverage 不是 100%' in text
    assert 'Run Passport' in MASTER.read_text()
    assert 'Historical run passport rule' in PROMPT.read_text()

def test_running_state_beats_completed_subartifact_filename_and_validity(tmp_path):
    root=tmp_path/'run'
    root.mkdir()
    (root/'execution-state.json').write_text(
        json.dumps({'status':'RUNNING','scientific_validity':'PASS'}),encoding='utf-8'
    )
    (root/'NATIVE_PARTITION_COMPLETE.json').write_text('{}',encoding='utf-8')
    worker=root/'workers'
    worker.mkdir()
    (worker/'rank-0.json').write_text(json.dumps({'status':'PASS'}),encoding='utf-8')
    _,evidence,_,status_tokens=extract_root(root)
    assert {'value':'RUNNING','file':'execution-state.json'} in status_tokens
    assert {'value':'PASS','file':'workers/rank-0.json'} in status_tokens
    assert not any(row=={'value':'PASS','file':'execution-state.json'} for row in status_tokens)
    lifecycle=infer_lifecycle(status_tokens)
    assert lifecycle=='running'

def test_root_execution_state_beats_stale_nested_worker_status(tmp_path):
    records=[
        {'value':'COMPLETE','file':'execution-state.json'},
        {'value':'RUNNING','file':'workers/rank-0.json'},
    ]
    assert infer_lifecycle(records)=='closed'

def test_conflicting_statuses_without_root_authority_are_not_assessed():
    records=[
        {'value':'RUNNING','file':'run-status.json'},
        {'value':'PASS','file':'summary.json'},
    ]
    assert infer_lifecycle(records)=='not-assessed'

def test_nested_worker_status_does_not_define_root_lifecycle():
    records=[{'value':'RUNNING','file':'workers/rank-0.json'}]
    assert infer_lifecycle(records)=='not-assessed'

def test_suffix_pass_and_complete_filename_do_not_close_unsealed_run(tmp_path):
    root=tmp_path/'run'
    root.mkdir()
    (root/'NATIVE_SUFFIX_COMPLETE.json').write_text(
        json.dumps({'status':'PASS','canonical_40_round_seal':False,'remaining_round_transitions':32}),
        encoding='utf-8',
    )
    _,evidence,_,status_records=extract_root(root)
    assert {'value':'PASS','file':'NATIVE_SUFFIX_COMPLETE.json'} in status_records
    lifecycle=infer_lifecycle(status_records)
    assert lifecycle=='not-assessed'

def test_filename_completion_without_authoritative_state_is_not_assessed():
    assert infer_lifecycle([])=='not-assessed'

def test_conflicting_root_execution_statuses_are_not_assessed():
    records=[
        {'value':'RUNNING','file':'execution-state.json'},
        {'value':'COMPLETE','file':'execution-state.json'},
    ]
    assert infer_lifecycle(records)=='not-assessed'
