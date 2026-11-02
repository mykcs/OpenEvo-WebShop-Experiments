import hashlib
import json
import base64
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "configs/experiment/advisor-questions/q06-s2-bounded-r120-prospective-contract.v1.json"
)


def read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_bounded_r120_contract_is_single_cell_and_fails_closed():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["status"] == "BLOCKED"
    assert contract["append_only"] is True
    assert contract["supersedes"] is None
    assert contract["stage"] == "S2"
    assert contract["cell"]["cell_id"] == "bounded-r120"
    assert contract["cell"]["mechanism"] == "Bounded State"
    assert contract["cell"]["checkpoint_id"] == "after_120_rounds"
    assert contract["cell"]["evaluation_tasks"] == 64
    assert contract["cell"]["pass_count"] == 1
    assert contract["execution_constraints"] == {
        "training_rollouts": 0,
        "optimizer_updates": 0,
        "teacher_calls": 0,
        "adaptive_reruns": 0,
        "retry_on_scientific_outcome": False,
        "episodes": 64,
        "passes": 1,
        "max_generation_calls": 960,
        "max_generated_tokens": 491520,
        "max_scorer_forward_calls": 100000,
        "max_scorer_input_tokens": 1000000,
        "max_gpu_hours": 2,
        "max_persistent_output_bytes": 1073741824,
        "final_panel_access_allowed": False,
        "formal_task_consumption_allowed": False,
        "gpu_allocation_allowed": False,
    }

    gates = contract["gates"]
    adapter = contract["known_shared_identity"]["adapter"]
    assert adapter["status"] == "VERIFIED_READ_ONLY"
    assert adapter["relative_file"] == "round-119/sd-lora/bounded-state/adapter_model.safetensors"
    assert adapter["size_bytes"] == 205551528
    assert adapter["adapter_sha256"] == (
        "c8ef26c3c96515495703d97148c1f4845c09814b686987f7150022099c67fd11"
    )
    assert gates["checkpoint_identity"]["status"] == "PASS_READ_ONLY_IDENTITY"
    assert gates["checkpoint_identity"]["source_summary_sha256"] == (
        "18b7212023b3c589294b66555d2d4a0d6444eb3e33a0706c14f5b4574a9964d9"
    )
    assert gates["scoring_cost_ceiling"]["status"] == "FROZEN"
    assert gates["scoring_cost_ceiling"]["max_generation_calls"] == 960
    assert gates["scoring_cost_ceiling"]["max_generated_tokens"] == 491520
    assert gates["scoring_cost_ceiling"]["max_scorer_forward_calls"] == 100000
    assert gates["scoring_cost_ceiling"]["max_scorer_input_tokens"] == 1000000
    cost_source = gates["scorer_cost_source"]
    assert cost_source["classification"] == "DERIVED_CPU_ONLY_NOT_OBSERVED_FORWARDS"
    cost_receipt = read_json(cost_source["path"])
    assert hashlib.sha256((ROOT / cost_source["path"]).read_bytes()).hexdigest() == cost_source["sha256"]
    assert sum(x["estimated_forward_calls"] for x in cost_receipt["arms"]["beta"]["groups"].values()) == 42073
    assert sum(x["estimated_input_tokens"] for x in cost_receipt["arms"]["beta"]["groups"].values()) == 386471
    assert gates["scoring_cost_ceiling"]["max_gpu_hours"] == 2
    assert gates["scoring_cost_ceiling"]["max_persistent_output_bytes"] == 1073741824
    assert gates["resource_qualification"]["status"] == "BLOCKED"
    assert gates["resource_qualification"]["authorized_gpu_uuids"] == []
    assert gates["resource_qualification"]["selected_gpu_uuid"] is None
    assert gates["resource_qualification"]["owner_admission"] is None
    assert gates["resource_qualification"]["lease"] is None
    assert gates["resource_qualification"]["preemption_allowed"] is False
    assert gates["canonical_executor_router"]["status"] == "NOT_IMPLEMENTED"
    assert gates["canonical_executor_router"]["path"] is None
    assert gates["launch_qualification"]["status"] == "NOT_RUN"
    assert gates["launch_qualification"]["receipt"] is None
    assert gates["task_bearing_activation"]["status"] == "NOT_CREATED"
    assert gates["task_bearing_activation"]["path"] is None


def test_contract_binds_only_identities_verified_in_existing_records():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    q01 = read_json(
        "docs/evidence/advisor-q01-horizon-plateau-20261001/Q01_THREE_WAY_RESULT.json"
    )
    bounded_analysis = read_json(
        "docs/evidence/advisor-q01-horizon-plateau-20261001/bounded-off-analysis.json"
    )
    prereg = read_json(
        "configs/experiment/preregistrations/q06-gdr-context-preregistration-202610011303.json"
    )

    assert contract["cell"]["evaluation_panel_sha256"] == q01["panel_sha256"]
    assert contract["cell"]["evaluation_panel_sha256"] == bounded_analysis["panel_sha256"]
    assert contract["cell"]["evaluation_tasks"] == q01["panel_size"]
    assert contract["cell"]["checkpoint_id"] == "after_120_rounds"
    assert contract["cell"]["actual_parameter_update_events"] == 118
    assert contract["cell"]["source_summary_sha256"] == (
        "18b7212023b3c589294b66555d2d4a0d6444eb3e33a0706c14f5b4574a9964d9"
    )
    assert contract["cell"]["source_summary_path"].endswith(
        "/after_120_rounds/summary.json"
    )
    assert contract["known_shared_identity"]["adapter"]["adapter_sha256"] == (
        "c8ef26c3c96515495703d97148c1f4845c09814b686987f7150022099c67fd11"
    )
    assert contract["known_shared_identity"]["base_model"] == prereg[
        "frozen_scientific_contract"
    ]["model"]["base_model"]
    assert contract["known_shared_identity"]["base_model_revision"] == prereg[
        "frozen_scientific_contract"
    ]["model"]["revision"]
    assert contract["known_shared_identity"]["sampling"] == prereg[
        "frozen_scientific_contract"
    ]["sampling"]
    assert bounded_analysis["points"][6]["checkpoint_id"] == "after_120_rounds"
    assert bounded_analysis["points"][6]["actual_parameter_update_events"] == 118


def test_contract_does_not_modify_or_rebind_existing_s1_activation():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    s1 = read_json(
        "configs/experiment/activations/q06-gdr-context-202610011303-current-campaign.json"
    )

    assert contract["gates"]["task_bearing_activation"]["status"] == "NOT_CREATED"
    assert s1["endpoint"] == "historical_gdr_r159_rank128"
    assert s1["formal_rollout_limit"] == 64
    assert s1["formal_task_consumption_allowed"] is True
    assert s1["authorized_gpu_uuids"] == ["GPU-00000000-0000-0000-0000-000000000000"]


def test_historical_q06_candidate_is_zero_formal_and_preserves_e1_seal():
    # Preserve the prior candidate as evidence; the main router remains owned by E1.
    index = read_json("docs/evidence/q06-frozen-input-compatibility-202610031219/INDEX.json")
    archived = index["historical_preflight_campaign"]
    raw = (ROOT / archived["snapshot"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == archived["sha256"] == "a963474337b151ec4e0230435ac40fb5a7628882fba238992472335007456eb3"
    assert len(raw) == archived["bytes"]
    assert archived["evidence_only"] is True and archived["current_authority"] is False
    current = json.loads(raw)
    paired_evidence = read_json(
        "docs/evidence/advisor-q06-e2-paired-analysis-20261001/Q06_E2_PAIRED_ANALYSIS.json"
    )
    source_revalidation = read_json(
        "docs/evidence/advisor-q06-e2-paired-analysis-20261001/source-revalidated-20261002/Q06_GDR_CONTEXT_SOURCE_REVALIDATION.json"
    )
    design_path = current["design"]
    design = read_json(design_path)
    activation_path = current["activation"]
    activation = read_json(activation_path)
    expected_revision = "".join(
        char for char in current["campaign_id"].split("-", 2)[:2] if char.isdigit()
    )

    assert current["status"] == "owner-authorized-plan-frozen-formal-locked"
    assert current["owner_authorized"] is True
    assert current["formal_task_consumption_allowed"] is False
    assert current["task_bearing_train_dev_allowed"] is False
    assert current["gpu_allocation_allowed"] is True
    assert current["final_test_status"] == "locked"
    assert current["final_panel_access_allowed"] is False
    assert current["final_pool_consumed"] is False
    assert current["final_unlock_allowed"] is False
    assert design["final_panel_access"] == 0
    assert design["frozen_scientific_contract"]["final_panel_unlock"] is False
    assert design["preflight_contract"]["formal_task_consumption_allowed"] is False
    assert design["preflight_contract"]["task_bearing_train_dev_allowed"] is False
    assert design["preflight_contract"]["gpu_allocation_allowed"] is True
    assert design["frozen_scientific_contract"]["max_scorer_forward_calls"] == 100000
    assert design["frozen_scientific_contract"]["max_scorer_input_tokens"] == 1000000

    assert activation["status"] == "PENDING_ZERO_FORMAL_PREFLIGHT"
    assert activation["revision"] == expected_revision == "202610020001"
    assert activation["design_path"] == design_path
    assert activation["design_sha256"] == hashlib.sha256(
        (ROOT / design_path).read_bytes()
    ).hexdigest()
    assert activation["owner_authorized"] is True
    assert activation["formal_task_consumption_allowed"] is False
    assert activation["task_bearing_train_dev_allowed"] is False
    assert activation["final_panel_access_allowed"] is False
    assert activation["gpu_uuids"] == current["authorized_gpu_uuids"]
    assert len(activation["gpu_uuids"]) == 8
    assert activation["selected_gpu_uuid"] is None
    assert activation["preemption_allowed"] is False

    matches = [
        read_json(f"configs/experiment/activations/{path.name}")
        for path in (ROOT / "configs/experiment/activations").glob("*.json")
        if read_json(f"configs/experiment/activations/{path.name}").get("design_path")
        == design_path
        and read_json(f"configs/experiment/activations/{path.name}").get("revision")
        == expected_revision
    ]
    assert len(matches) == 1

    prior = current["completed_prior_phase"]
    assert prior["campaign_id"] == "20260928-2338-post-advisor-unanswered"
    assert prior["phase"] == "POST-ADVISOR-E1-EXECUTION-SEALED"
    assert prior["status"] == "execution-sealed"
    assert prior["already_consumed_e1_rollouts"] == 1024
    assert prior["remaining_e1_rollout_budget"] == 0
    assert prior["latest_event"]["seal_sha256"] == (
        "b03d0cbcb9e5fc89a8d8974952222aa5d35ffb083bc6a47239cfa30b2b893725"
    )
    assert current["archive"] == prior["archive"]
    assert prior["archive"]["release"]["path"].endswith(
        "post-advisor-e1-a11-terminal-seal-release-202609302100.json"
    )

    live = current["prior_live_control_reconciliation"]
    assert live["status"] == "RECONCILED_READ_ONLY"
    evidence_path = ROOT / live["evidence_receipt"]["path"]
    evidence_bytes = evidence_path.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == live["evidence_receipt"]["sha256"]
    evidence = json.loads(evidence_bytes)
    assert evidence["schema"] == "openevo.q06.e2-live-control-reconciliation.v1"
    assert evidence["read_only"] is True
    for record in evidence["raw_files"].values():
        raw_bytes = base64.b64decode(record["base64"], validate=True)
        assert len(raw_bytes) == record["bytes"]
        assert hashlib.sha256(raw_bytes).hexdigest() == record["sha256"]
    def raw_json(name):
        return json.loads(base64.b64decode(evidence["raw_files"][name]["base64"], validate=True))
    assert raw_json("e2_claim")["status"] == "CONSUMED"
    assert raw_json("e2_formal_start")["status"] == "FORMAL_START_COMMITTED"
    assert raw_json("host_current_campaign")["campaign_id"] == "20260928-2338-post-advisor-unanswered"
    assert raw_json("e2_probe_receipt")["task_count"] == 64
    observed = evidence["observed"]
    assert live["canonical_host_control_head"] == observed["canonical_control_head"]
    assert live["host_current_campaign_sha256"] == observed["host_current_campaign_sha256"]
    assert evidence["raw_files"]["host_current_campaign"]["sha256"] == observed["host_current_campaign_sha256"]
    assert observed["claim_status"] == "CONSUMED"
    assert observed["formal_start_status"] == "FORMAL_START_COMMITTED"
    assert observed["output_summary_status"] == "PASS"
    assert observed["output_task_count"] == 64
    assert observed["matched_processes"] == []
    assert observed["matched_docker_containers"] == []
    assert len(observed["gpus"]) == 8
    assert all(gpu["memory_used_mib"] == 2 and gpu["utilization_percent"] == 0
               for gpu in observed["gpus"])
    assert evidence["source_revalidation"]["bytes_verified"] is True
    source_path = ROOT / evidence["source_revalidation"]["path"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == evidence["source_revalidation"]["sha256"]
    assert evidence["source_revalidation"]["bounded_attempts_sha256"] == paired_evidence["input_files"]["bounded"]["sha256"]
    assert evidence["source_revalidation"]["bounded_summary_sha256"] == paired_evidence["input_files"]["bounded_summary"]["sha256"]
    route = live["prior_live_route"]
    assert route["route_label"] == "E2 bounded retry"
    assert route["status_observed_before_reconciliation"] == "launch-authorized"
    assert route["pending_at_observation"] is True
    assert route["disposition"] == "SUPERSEDED_BY_CONSUMED_OUTPUT_RECONCILIATION"
    assert route["runnable"] is False

    consumed = live["consumption_reconciliation"]
    assert consumed["claim_sha256"] == evidence["raw_files"]["e2_claim"]["sha256"]
    assert consumed["claim_status"] == "CONSUMED"
    assert consumed["output_status"] == "PASS"
    assert consumed["valid_tasks"] == consumed["expected_tasks"] == 64
    assert consumed["attempts_sha256"] == paired_evidence["input_files"]["bounded"]["sha256"]
    assert consumed["summary_sha256"] == paired_evidence["input_files"]["bounded_summary"]["sha256"]
    bounded_source_hashes = {
        (entry["kind"], entry["sha256"])
        for entry in source_revalidation["source_provenance"]["files"]
        if entry["arm"] == "bounded"
    }
    assert ("attempts", consumed["attempts_sha256"]) in bounded_source_hashes
    assert ("summary", consumed["summary_sha256"]) in bounded_source_hashes
    assert consumed["receipt_sha256"] == evidence["raw_files"]["e2_probe_receipt"]["sha256"]

    evidence_references = {entry["path"]: entry["sha256"] for entry in live["evidence_references"]}
    assert evidence_references == {
        "docs/evidence/advisor-q06-e2-paired-analysis-20261001/Q06_E2_PAIRED_ANALYSIS.json": hashlib.sha256(
            (ROOT / "docs/evidence/advisor-q06-e2-paired-analysis-20261001/Q06_E2_PAIRED_ANALYSIS.json").read_bytes()
        ).hexdigest(),
        "docs/evidence/advisor-q06-e2-paired-analysis-20261001/source-revalidated-20261002/Q06_GDR_CONTEXT_SOURCE_REVALIDATION.json": hashlib.sha256(
            (ROOT / "docs/evidence/advisor-q06-e2-paired-analysis-20261001/source-revalidated-20261002/Q06_GDR_CONTEXT_SOURCE_REVALIDATION.json").read_bytes()
        ).hexdigest(),
    }

    live_execution = live["live_execution_observation"]
    assert live_execution["process_count"] == 0
    assert live_execution["active_gpu_use"] is False
    assert live_execution["active_gpu_uuids"] == []
    assert "Read-only" in live_execution["verification"]


def test_current_main_campaign_remains_sealed_without_q06_activation():
    current = read_json("configs/experiment/current-campaign.json")
    assert current["campaign_id"] == "20260928-2338-post-advisor-unanswered"
    assert current["status"] == "execution-sealed"
    assert current["remaining_e1_rollout_budget"] == 0
    assert current["formal_task_consumption_allowed"] is False
    assert current["task_bearing_train_dev_allowed"] is False
    assert current["gpu_allocation_allowed"] is False
    assert current["final_panel_access_allowed"] is False
    assert current["final_unlock_allowed"] is False
    assert current["final_pool_consumed"] is False
