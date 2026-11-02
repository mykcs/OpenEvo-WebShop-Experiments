import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(): return json.loads((ROOT/'configs/experiment/designs/openevo-ceiling1-stage2-vnext-3b-v1.json').read_text())
def test_no_activation_or_lineage_inheritance():
 d=load(); assert d['activation_authority'] is False; o=d['origin']; assert not o['old_formal_3200_prefix_inherited']; assert not o['g3_shadow_state_inherited']; assert not o['step_sd_C_adapter_inherited']
def test_round_is_accounting_not_reset():
 d=load(); assert d['round_semantics']['noop_resets_evidence'] is False; assert d['typed_evidence_ledger']['persistent_across_rounds'] is True
def test_partial_steps_do_not_enter_sd_v1():
 d=load(); sd=d['carrier_channels']['sd_lora']; assert sd['verified_partial_step_evidence_eligible'] is False; assert sd['eligible_online_evidence']=='clean exact-success whole trajectories only in v1'
def test_candidate_never_auto_promotes():
 d=load(); assert all(not d['carrier_channels'][k]['auto_promotion'] for k in ('text_memory','skill_bundle','agent_system')); assert d['candidate_state_transaction']['probe_required_if_candidate_differs'] is True
def test_promotion_requires_strict_task_improvement_without_numeric_magic():
 r=load()['candidate_state_transaction']['promotion_rule']; assert r['candidate_exact_success_gte_prior']; assert r['candidate_positive_score_gte_prior']; assert r['candidate_mean_score_gte_prior']; assert 'strictly greater' in r['strict_task_improvement_required']; assert r['neutral_behavior_change']=='REJECT_KEEP_PRIOR_STATE'
def test_resource_preference_and_7b_priority():
 r=load()['resource_policy']; assert r['managed_gpu_indices']==[4,5,6,7]; assert r['three_b_selection_preference']==[7,6,5,4]; assert r['seven_b_priority'] is True
