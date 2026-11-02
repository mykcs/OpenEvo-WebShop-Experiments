from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.openevo_runtime.prospective_adoption import (
    ProspectiveAdoptionError,
    build_prospective_adoption_release,
    point_of_use_adoption_decision,
    validate_prospective_adoption_release,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "4f5db462ab3cc728b70f6bd1f982c8a781635548"
H64 = lambda text: hashlib.sha256(text.encode()).hexdigest()


def _release(*, ready: bool = False, consumer: str | None = None):
    return build_prospective_adoption_release(
        target_campaign_id="202609171752-execution-runtime-next-campaign-canary",
        campaign_class="ZERO_FORMAL_ENGINEERING_CANARY",
        status="READY_FOR_ADOPTION_VALIDATION" if ready else "BLOCKED_ENGINEERING",
        blocker_reason=None if ready else "ZERO_FORMAL_PRODUCTION_RUNTIME_RESOURCE_CANARY_PENDING",
        implementation_contract_sha256=H64("prospective-adoption-contract"),
        production_runtime_package_identity=H64("runtime-package") if ready else None,
        execution_generation_id=H64("generation") if ready else None,
        resource_canary_receipt_identity=H64("resource-canary") if ready else None,
        exact_consumer_validation_receipt_identity=consumer,
        reference_controller_git_sha=REFERENCE,
        predecessor_controller_retained=True,
        predecessor_retention_until_closure=True,
        exact_consumer_validation_required_before_first_formal_task=True,
        target_campaign_is_new=True,
        target_campaign_formal_rows_consumed=0,
        existing_formal_lineage_migration_allowed=False,
        hot_migration_allowed=False,
        live_adoption=False,
        formal_launch_authority=False,
        scientific_authority=False,
        resource_topology_authority=False,
        repair_semantics_authority=False,
        reuse_authority=False,
        final_panel_authority=False,
        closure_gates=[
            "ZERO_FORMAL_RESOURCE_CANARY_PASS",
            "EXACT_CONSUMER_VALIDATION_PASS",
            "NEW_CAMPAIGN_BOUNDARY_CONFIRMED",
            "PREDECESSOR_REFERENCE_RETAINED",
        ],
    )


def test_blocked_release_is_new_campaign_only_and_non_authoritative() -> None:
    release = _release()
    validate_prospective_adoption_release(release)
    assert release["target_campaign_is_new"] is True
    assert release["target_campaign_formal_rows_consumed"] == 0
    assert release["existing_formal_lineage_migration_allowed"] is False
    assert release["formal_launch_authority"] is False


def test_formal_rows_already_consumed_fail_closed() -> None:
    release = _release(ready=True, consumer=H64("consumer"))
    decision = point_of_use_adoption_decision(
        release, observed_campaign_id=release["target_campaign_id"], formal_rows_consumed=1,
        production_runtime_package_identity=release["production_runtime_package_identity"],
        execution_generation_id=release["execution_generation_id"],
        resource_canary_receipt_identity=release["resource_canary_receipt_identity"],
        consumer_validation_receipt_identity=release["exact_consumer_validation_receipt_identity"],
        predecessor_controller_available=True,
    )
    assert decision["decision"] == "BLOCKED"
    assert "FORMAL_TASK_ALREADY_CONSUMED" in decision["blockers"]


def test_exact_consumer_validation_is_required_before_first_formal_task() -> None:
    release = _release(ready=True, consumer=H64("consumer"))
    decision = point_of_use_adoption_decision(
        release, observed_campaign_id=release["target_campaign_id"], formal_rows_consumed=0,
        production_runtime_package_identity=release["production_runtime_package_identity"],
        execution_generation_id=release["execution_generation_id"],
        resource_canary_receipt_identity=release["resource_canary_receipt_identity"],
        consumer_validation_receipt_identity=None,
        predecessor_controller_available=True,
    )
    assert decision["decision"] == "BLOCKED"
    assert "EXACT_CONSUMER_VALIDATION_MISSING_OR_DRIFT" in decision["blockers"]


def test_predecessor_reference_must_remain_available_until_closure() -> None:
    release = _release(ready=True, consumer=H64("consumer"))
    decision = point_of_use_adoption_decision(
        release, observed_campaign_id=release["target_campaign_id"], formal_rows_consumed=0,
        production_runtime_package_identity=release["production_runtime_package_identity"],
        execution_generation_id=release["execution_generation_id"],
        resource_canary_receipt_identity=release["resource_canary_receipt_identity"],
        consumer_validation_receipt_identity=release["exact_consumer_validation_receipt_identity"],
        predecessor_controller_available=False,
    )
    assert "PREDECESSOR_REFERENCE_UNAVAILABLE" in decision["blockers"]


def test_all_engineering_preconditions_still_do_not_grant_formal_launch() -> None:
    release = _release(ready=True, consumer=H64("consumer"))
    decision = point_of_use_adoption_decision(
        release, observed_campaign_id=release["target_campaign_id"], formal_rows_consumed=0,
        production_runtime_package_identity=release["production_runtime_package_identity"],
        execution_generation_id=release["execution_generation_id"],
        resource_canary_receipt_identity=release["resource_canary_receipt_identity"],
        consumer_validation_receipt_identity=release["exact_consumer_validation_receipt_identity"],
        predecessor_controller_available=True,
    )
    assert decision["decision"] == "ADOPTION_PRECONDITIONS_MET"
    assert decision["formal_launch_authority"] is False
    assert decision["scientific_authority"] is False
    assert decision["external_campaign_release_still_required"] is True
