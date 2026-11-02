import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DESIGN=ROOT/'configs/experiment/designs/openevo-mechanism1-seed-stage2-minimax-sft-successor-202609070053.json'
ADAPTER=ROOT/'scripts/openevo_webshop/prepare_mechanism1_seed_stage2_minimax_sft_202609070053.py'
ROUTER=ROOT/'configs/experiment/campaigns/current-mechanism1-autonomous.json'
CURRENT=ROOT/'docs/science/webshop/program/OPEN_EVO_MECHANISM_1.0_CURRENT.md'
SFT_PREP=ROOT/'configs/experiment/receipts/openevo-mechanism1-seed-stage2-sft-preparation-202609071201.json'


def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_is_new_inactive_identity_and_parent_m1d_stays_closed():
    d=load(DESIGN)
    assert d['activation_authority'] is False
    assert d['formal_task_consumption_allowed'] is False
    assert d['parent_m1d_remains_immutable'] is True
    assert d['parent_m1d_hard_stop_is_preserved'] is True
    assert 'not a paper-exact SEED reproduction' in d['claim_boundary']
    assert d['parent_bindings']['m1d_baseline_corpus_sha256']=='843ac0b441c84884df83203adb66d4203330dcc48536a64b58619f99e37ee308'
    assert d['parent_bindings']['m1d_minimax_required_parse_ok']==1440


def test_teacher_and_public_sft_contract_are_frozen_prospectively():
    d=load(DESIGN)
    t=d['teacher_contract']; s=d['sft_contract']
    assert (t['provider'],t['model'])==('MiniMax','MiniMax-M3')
    assert 'GLM-5.2' in t['paper_default_teacher_difference']
    assert t['temperature']==0.0 and t['max_completion_tokens']==1024
    assert s['expected_records_if_minimax_seals']==1440
    assert (s['expected_train_rows_if_minimax_seals'],s['expected_validation_rows_if_minimax_seals'])==(1296,144)
    assert s['epochs']==3 and s['learning_rate']==0.000005 and s['max_length']==12288
    assert s['global_train_batch_size']==8 and s['micro_batch_size_per_gpu']==1


def test_stage2_algorithm_and_resource_variant_are_explicit():
    d=load(DESIGN); a=d['stage2_algorithm_contract']; r=d['resource_contract']
    assert (a['train_data_size'],a['validation_data_size'],a['group_size'])==(16,128,8)
    assert a['outer_updates']==160 and a['source_trajectories']==20480
    assert a['history_length']==2 and a['max_steps']==15
    assert a['seed_mode']=='mean_norm' and a['seed_skill_mode']=='episode_only'
    assert a['analysis_backend']=='policy_vllm' and a['validation_temperature']==0.4
    assert r['user_authorized_gpus']==list(range(8))
    assert r['current_four_gpu_runtime_variant']==[0,1,2,3]
    assert r['no_mid_run_reshard'] is True
    assert r['eight_gpu_public_default_requires_new_resource_lineage'] is True


def test_adapter_is_mechanical_and_uses_exact_public_exporter():
    text=ADAPTER.read_text()
    assert 'EXPECTED_BASELINE_SHA = "843ac0b441c84884df83203adb66d4203330dcc48536a64b58619f99e37ee308"' in text
    assert 'build_sft_exports_from_candidates' in text
    assert 'new_teacher_calls' in text and 'new_webshop_calls' in text
    assert 'sft_min_source_score=0.0' in text and 'sft_include_success=True' in text
    assert 'if (train_rows, val_rows) != (1296, 144)' in text


def test_router_and_current_doc_project_inactive_successor_without_touching_m1a():
    r=load(ROUTER); s=r['m1d_stage2_successor']; m1a=r['m1a_lineage']
    assert r['execution_activation_authority'] is False
    assert s['design']==str(DESIGN.relative_to(ROOT)) and s['design_sha256']==sha(DESIGN)
    assert s['data_adapter']==str(ADAPTER.relative_to(ROOT)) and s['data_adapter_sha256']==sha(ADAPTER)
    assert m1a['activation_authority'] is True
    assert m1a['status']=='SUCCESSOR_ACTIVATION_RELEASE_BOUND_CONDITIONAL_CURRENT_MAIN'
    text=CURRENT.read_text()
    assert 'M1-D Stage2 successor — MiniMax-teacher variant' in text
    if s['status']=='PREREGISTERED_WAITING_M1D_MINIMAX_SEAL':
        assert s['activation_authority'] is False and s['formal_task_consumption_allowed'] is False
        assert 'PREREGISTERED_WAITING_M1D_MINIMAX_SEAL' in text
    else:
        prep=load(SFT_PREP)
        assert prep['offline_sft_preparation_authority'] is True
        assert prep['stage2_activation_authority'] is False
        assert s['sft_preparation_receipt_sha256']==sha(SFT_PREP)
        assert s['activation_authority'] is False and s['formal_task_consumption_allowed'] is False
        assert 'SFT preparation' in text or 'SFT_PREPARATION' in text
