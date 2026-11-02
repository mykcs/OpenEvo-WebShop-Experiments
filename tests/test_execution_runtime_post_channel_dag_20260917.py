from __future__ import annotations

import copy

import pytest

from scripts.openevo_runtime.post_channel_dag import (
    PostChannelDagError,
    audit_controller_source,
    build_post_channel_dag,
    validate_post_channel_dag,
)


CONTROLLER_SOURCE = '''
class BoundedEffectiveStateController:
    def run_round(self, r):
        rr = self.out
        self.ensure_round_before(r, rr)
        self.run_rollout(r, rr)
        self.evidence(r, rr)
        prior_sd = self.state["current_sd_adapter"]
        prior_mem = self.state["memory"]
        prior_skill = self.state["skill"]
        prior_agent = self.state["agent_system"]
        new_mem = self.reflect(r, rr, "text_memory", prior_mem, sd_adapter=prior_sd)
        new_sd, new_sd_id, ignored, updated = self.run_sd(r, rr)
        new_skill = self.reflect(r, rr, "skill_bundle", prior_skill, sd_adapter=prior_sd)
        new_agent = self.reflect(r, rr, "agent_system", prior_agent, memory=prior_mem, skill=prior_skill, sd_adapter=prior_sd)
        sd_terminal = load_json(rr / "sd-lora" / "terminal-receipt.json")
        tv = self.task_vector_receipt(r, rr, sd_terminal, prior_sd, new_sd)
        self.retention(r, rr, tv)
        self.state.update({"current_sd_adapter": new_sd, "memory": new_mem, "skill": new_skill, "agent_system": new_agent})
        write_once_json(rr / "state-after.json", self.state)
        write_once_json(rr / "ROUND_COMPLETE.json", self.state)
        self.apply_carrier_health_boundary(r, rr)
'''

BASE_SOURCE = '''
class Controller:
    def reflect(self, r, rr, kind, prior, memory=None, skill=None, sd_adapter=None):
        evidence_path = rr / "evidence/evidence.json"
        attempt = reflector_attempt_identity(kind,r,self.state_id,evidence_path,prior,memory,skill,sd_adapter)
        return attempt
'''


def _dag(controller_source: str = CONTROLLER_SOURCE, base_source: str = BASE_SOURCE):
    return build_post_channel_dag(
        source_execution_sha="a" * 40,
        controller_source=controller_source,
        base_controller_source=base_source,
        controller_path="controller.py",
        base_controller_path="base.py",
    )


def test_mechanical_audit_proves_four_channels_read_prior_state_not_sibling_outputs() -> None:
    audit = audit_controller_source(CONTROLLER_SOURCE)
    assert audit["all_post_channels_use_prior_state_not_sibling_outputs"] is True
    bindings = audit["post_channel_bindings"]
    assert bindings["text_memory"]["keyword_bindings"] == {"sd_adapter": "prior_sd"}
    assert bindings["skill_bundle"]["keyword_bindings"] == {"sd_adapter": "prior_sd"}
    assert bindings["agent_system"]["keyword_bindings"] == {
        "memory": "prior_mem", "skill": "prior_skill", "sd_adapter": "prior_sd"
    }
    assert all(not row["current_round_sibling_inputs"] for row in bindings.values())


def test_frozen_dag_allows_prospective_fanout_but_does_not_activate_parallel_execution() -> None:
    dag = _dag()
    assert dag["prospective_parallel_group"] == ["text_memory", "sd_lora", "skill_bundle", "agent_system"]
    assert dag["join_contract"]["required_terminal_nodes"] == dag["prospective_parallel_group"]
    assert dag["join_contract"]["channel_scoped_activity_state_required"] is True
    assert dag["join_contract"]["atomic_state_commit"] is True
    assert dag["activation"]["parallel_execution_activated"] is False
    assert dag["authority_boundary"]["resource_topology_authority_granted"] is False
    validate_post_channel_dag(dag)


def test_sd_lora_is_released_after_evidence_without_claiming_evidence_is_a_data_input() -> None:
    dag = _dag()
    assert dag["nodes"]["sd_lora"]["reads"] == ["attempts.jsonl", "prior_sd"]
    assert ["evidence", "sd_lora", "CONSERVATIVE_RELEASE_BARRIER"] in dag["edges"]
    assert dag["audit"]["sd_lora_does_not_read_evidence_but_is_conservatively_released_after_evidence"] is True


def test_current_round_sibling_dependency_fails_closed() -> None:
    drifted = CONTROLLER_SOURCE.replace("memory=prior_mem", "memory=new_mem")
    with pytest.raises(PostChannelDagError, match="agent_system dependency drift|sibling output"):
        audit_controller_source(drifted)


def test_reflector_must_keep_evidence_and_prior_input_identity_binding() -> None:
    drifted = BASE_SOURCE.replace('evidence/evidence.json', 'different/input.json')
    with pytest.raises(PostChannelDagError, match="sealed round evidence"):
        _dag(base_source=drifted)


def test_dag_tampering_cannot_activate_parallelism_or_change_identity_silently() -> None:
    dag = _dag()
    tampered = copy.deepcopy(dag)
    tampered["activation"]["parallel_execution_activated"] = True
    with pytest.raises(PostChannelDagError, match="constant mismatch|may not activate"):
        validate_post_channel_dag(tampered)


def test_source_change_changes_dag_identity_even_when_dependency_shape_still_matches() -> None:
    first = _dag()
    second = _dag(controller_source=CONTROLLER_SOURCE + "\n# exact-source identity change\n")
    assert first["dag_id"] != second["dag_id"]
    assert first["source_files"] != second["source_files"]
