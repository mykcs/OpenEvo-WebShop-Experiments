#!/usr/bin/env python3
"""Validate the published H1.38B preregistration and exact GPU authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.openevo_webshop.ordinary_lora_contract import (
    PINNED_DATASET_HELPER_SHA256,
    PINNED_TRAINER_HELPERS_SHA256,
    digest_json,
)
from scripts.openevo_webshop.reconcile_h138b_method_control import (
    D8_RECORDS_SHA256,
    D8_SELECTION_SHA256,
    _validate_manifest,
    _validate_training_receipts,
)
from scripts.openevo_webshop.campaign_router import is_external_successor_router


PREREG = Path("configs/experiment/h1.38b-method-control-v1.json")
CURRENT = Path("configs/experiment/current-campaign.json")
EVAL_MANIFEST = Path("configs/experiment/manifests/h1.38b-method-control-eval-v1.json")
SAMPLE_PLANS = Path("configs/experiment/manifests/h1.38b-frozen-sample-plans-v1.json")
QUALIFICATION = Path("docs/evidence/remote-runs/2026-08-20/h1.38b-qualification/qualification.json")
QUALIFICATION_RECEIPTS = QUALIFICATION.parent / "receipts"
PREAUTH_VALIDATION = QUALIFICATION.parent / "preregistration-validation.json"
FORMAL_ATTEMPT_1 = QUALIFICATION.parent / "formal-attempt-3aac79c.json"
FORMAL_ATTEMPT_2 = QUALIFICATION.parent / "formal-attempt-ba2b450.json"
FORMAL_ATTEMPT_3 = QUALIFICATION.parent / "formal-attempt-f751a76.json"
FORMAL_ATTEMPT_4 = QUALIFICATION.parent / "formal-attempt-010c622.json"
CORRECTION_VALIDATION = QUALIFICATION.parent / "implementation-correction-validation.json"
OUTPUT_CORRECTION_VALIDATION = QUALIFICATION.parent / "output-isolation-correction-validation.json"
OFFLINE_MODEL_CORRECTION_VALIDATION = QUALIFICATION.parent / "offline-model-cache-correction-validation.json"
OFFLINE_MODEL_REF_CORRECTION_VALIDATION = QUALIFICATION.parent / "offline-model-ref-correction-validation.json"
H138A_MANIFEST = Path("docs/evidence/remote-runs/2026-08-19/h1.38a/manifests/h1.38a-evaluation-manifest.json")
AUTHORIZED_GPU_UUIDS = [
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
]
H138B_CAMPAIGN_ID = "20260820-0129-h138b-method-control"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_h138b_current_router(current: dict[str, Any]) -> list[str]:
    """Validate the historical H1.38B router without coupling to successors."""

    _require(current.get("campaign_id") == H138B_CAMPAIGN_ID, "H1.38B current campaign identity mismatch")
    _require(current.get("preregistration") == PREREG.as_posix(), "H1.38B current router preregistration mismatch")
    _require(current.get("phase") == "H1.38B", "H1.38B current campaign phase mismatch")
    _require(current.get("authorized_gpu_uuids") == AUTHORIZED_GPU_UUIDS, "current campaign GPU UUID authorization set mismatch")

    if current.get("status") == "launch-authorized":
        _require(current.get("formal_task_consumption_allowed") is True, "H1.38B open campaign did not open formal consumption")
        _require(
            current.get("protocol_status")
            in {
                "continuation-amendment-published-pre-gpu-gate",
                "preregistered-cpu-qualified-launch-authorized",
            },
            "H1.38B open campaign protocol status mismatch",
        )
        return []
    if current.get("status") == "sealed-result":
        _require(current.get("protocol_status") == "continuation-reconciled-selector-b1", "sealed campaign result status mismatch")
        _require(current.get("formal_task_consumption_allowed") is False, "sealed campaign reopened formal consumption")
        latest = current.get("latest_event") if isinstance(current.get("latest_event"), dict) else {}
        _require(latest.get("reconciliation_status") == "complete", "sealed campaign reconciliation is not complete")
        _require(latest.get("selector_code") == "B1", "sealed campaign selector is missing")
        return ["preregistered-campaign-closed"]
    _require(current.get("status") == "partial-window-closed", "H1.38B current campaign status is neither open nor safely closed")
    _require(current.get("protocol_status") == "partial-unqualified-safe-boundary-closed", "H1.38B closed campaign protocol status mismatch")
    _require(current.get("formal_task_consumption_allowed") is False, "closed campaign reopened formal consumption")
    return ["preregistered-time-window-closed"]


def _validate_successor_router(current: dict[str, Any]) -> list[str]:
    """Prove H1.38B is no longer current without enumerating future campaigns."""

    _require(
        is_external_successor_router(
            current,
            {H138B_CAMPAIGN_ID},
            predecessor_routes={PREREG.as_posix()},
        ),
        "successor campaign router is incomplete or unexpectedly reopens H1.38B",
    )
    if current.get("status") in {
        "partial-window-closed",
        "sealed-result",
        "preregistered-pending-live-gate",
        "preregistered-pre-execution",
        "blocked-pre-execution",
    }:
        _require(current.get("formal_task_consumption_allowed") is False, "closed campaign reopened formal consumption")
    return ["preregistered-campaign-closed"]


def _validate_launch_boundary(prereg: dict[str, Any], current: dict[str, Any]) -> list[str]:
    uuids = prereg.get("authorized_gpu_uuids")
    _require(uuids == AUTHORIZED_GPU_UUIDS, "published GPU UUID authorization set mismatch")
    _require(len(set(uuids)) == len(uuids), "published GPU UUID authorization set contains duplicates")
    _require(
        prereg.get("gpu_authorization_status") == "campaign-contract-authorized",
        "GPU authorization status mismatch",
    )
    authorization = prereg.get("gpu_authorization") or {}
    _require(authorization.get("scope") == "H1.38B formal run only", "GPU authorization scope mismatch")
    _require(authorization.get("session_reconfirmation_required") is False, "campaign authorization incorrectly requires session reconfirmation")
    _require("standing delegation" in str(authorization.get("machine_gate_basis") or "").lower(), "standing delegation basis mismatch")
    _require(authorization.get("authorization_is_reservation") is False, "GPU authorization was misrepresented as a reservation")
    _require(prereg.get("formal_task_consumption_allowed") is True, "preregistration did not open formal consumption")
    _require(prereg.get("status") == "launch-authorized", "published preregistration status mismatch")
    if current.get("campaign_id") == H138B_CAMPAIGN_ID:
        return _validate_h138b_current_router(current)
    return _validate_successor_router(current)


def validate(root: Path) -> dict[str, Any]:
    prereg_path, current_path = root / PREREG, root / CURRENT
    eval_path, plans_path = root / EVAL_MANIFEST, root / SAMPLE_PLANS
    qualification_path = root / QUALIFICATION
    prereg, current = _read(prereg_path), _read(current_path)
    evaluation, plans, qualification = _read(eval_path), _read(plans_path), _read(qualification_path)

    campaign_id = H138B_CAMPAIGN_ID
    _require(prereg.get("campaign_id") == campaign_id, "H1.38B preregistration campaign identity mismatch")
    blockers = _validate_launch_boundary(prereg, current)
    amendment = prereg.get("authorization_amendment") or {}
    _require(amendment.get("pre_authorization_publication_sha") == "3ca5d05a37480b6a16f2d4aaf6a32bd5c22325d0", "pre-authorization publication SHA mismatch")
    _require(amendment.get("immediate_parent_sha") == "ee6c9695a4b9c34d3fce3d09653bce5817e511c7", "authorization amendment parent SHA mismatch")
    _require(amendment.get("pre_authorization_validation_path") == PREAUTH_VALIDATION.as_posix(), "pre-authorization validation path mismatch")
    _require(_sha256(root / PREAUTH_VALIDATION) == amendment.get("pre_authorization_validation_sha256") == "120632ce8a740e3918170a96fe1e64809336f1dcc7c1403b63be7d155e8091bd", "pre-authorization validation SHA mismatch")
    correction = prereg.get("implementation_correction") or {}
    _require(correction.get("parent_execution_sha") == "3aac79c4dd3cee533156e58ff1a4e2fb0cfd3128", "implementation correction parent SHA mismatch")
    _require(correction.get("scientific_contract_changed") is False and correction.get("data_tasks_arms_seeds_budget_changed") is False, "implementation correction changed the scientific contract")
    _require(correction.get("first_attempt_receipt_path") == FORMAL_ATTEMPT_1.as_posix(), "first attempt receipt path mismatch")
    _require(_sha256(root / FORMAL_ATTEMPT_1) == correction.get("first_attempt_receipt_sha256") == "874243cbde28cf4851ab36fec19db10742d5aa3e439a1e2622fb71752ce30f0d", "first attempt receipt SHA mismatch")
    _require(correction.get("validation_path") == CORRECTION_VALIDATION.as_posix(), "implementation correction validation path mismatch")
    _require(_sha256(root / CORRECTION_VALIDATION) == correction.get("validation_sha256") == "c2db038d526bee7309edebab7c6fe52d23de46edff1394681c8ee4cbcaff70b1", "implementation correction validation SHA mismatch")
    first_attempt = _read(root / FORMAL_ATTEMPT_1)
    _require(first_attempt.get("status") == "invalid-engineering-zero-consumption", "first attempt status mismatch")
    _require(first_attempt.get("formal_attempts") == 0 and first_attempt.get("gpu_hours") == 0.0 and first_attempt.get("fresh_panel_consumed") is False, "first attempt consumed scientific work")
    output_correction = prereg.get("output_isolation_correction") or {}
    _require(output_correction.get("parent_execution_sha") == "ba2b4508bcc76c6970a95fa31d069ae83cb6588c", "output isolation correction parent SHA mismatch")
    _require(output_correction.get("scientific_contract_changed") is False and output_correction.get("data_tasks_arms_seeds_budget_changed") is False, "output isolation correction changed the scientific contract")
    _require(output_correction.get("second_attempt_receipt_path") == FORMAL_ATTEMPT_2.as_posix(), "second attempt receipt path mismatch")
    _require(_sha256(root / FORMAL_ATTEMPT_2) == output_correction.get("second_attempt_receipt_sha256") == "9902ad12d294707df228421a362e9c8d1d56a08b896676d7a7fc4a32a2f33d81", "second attempt receipt SHA mismatch")
    _require(output_correction.get("validation_path") == OUTPUT_CORRECTION_VALIDATION.as_posix(), "output correction validation path mismatch")
    _require(_sha256(root / OUTPUT_CORRECTION_VALIDATION) == output_correction.get("validation_sha256") == "ae9981516f9836dcf9dbd72ca9c4f0fbf877b28e6073a8926384a145e366b60a", "output correction validation SHA mismatch")
    second_attempt = _read(root / FORMAL_ATTEMPT_2)
    _require(second_attempt.get("status") == "invalid-engineering-zero-scientific-consumption", "second attempt status mismatch")
    _require(second_attempt.get("formal_attempts") == 0 and second_attempt.get("gpu_hours") == 0.0033333333333333335 and second_attempt.get("fresh_panel_consumed") is False, "second attempt scientific consumption mismatch")
    model_correction = prereg.get("offline_model_cache_correction") or {}
    _require(model_correction.get("parent_execution_sha") == "f751a76a3ba133bb389064354ad430256eb2a67f", "offline model correction parent SHA mismatch")
    _require(model_correction.get("scientific_contract_changed") is False and model_correction.get("data_tasks_arms_seeds_budget_changed") is False, "offline model correction changed the scientific contract")
    _require(model_correction.get("third_attempt_receipt_path") == FORMAL_ATTEMPT_3.as_posix(), "third attempt receipt path mismatch")
    _require(_sha256(root / FORMAL_ATTEMPT_3) == model_correction.get("third_attempt_receipt_sha256") == "d5b874c68eee76cd90ead5e256a23bbd34885460b3320f4c6ee9b8b0c0d326f2", "third attempt receipt SHA mismatch")
    _require(model_correction.get("validation_path") == OFFLINE_MODEL_CORRECTION_VALIDATION.as_posix(), "offline model correction validation path mismatch")
    _require(_sha256(root / OFFLINE_MODEL_CORRECTION_VALIDATION) == model_correction.get("validation_sha256") == "25e99f2eae459bf2dbdb38fd2f772129c6d7b159d0f75d784f38eb98fabf22f9", "offline model correction validation SHA mismatch")
    third_attempt = _read(root / FORMAL_ATTEMPT_3)
    _require(third_attempt.get("status") == "invalid-engineering-partial-training-zero-evaluation", "third attempt status mismatch")
    _require(third_attempt.get("formal_attempts") == 0 and third_attempt.get("attempt_gpu_hours") == 0.0075 and third_attempt.get("fresh_panel_consumed") is False, "third attempt denominator/GPU-hours mismatch")
    ref_correction = prereg.get("offline_model_ref_correction") or {}
    _require(ref_correction.get("parent_execution_sha") == "010c6226114186dfbeca61d71dcc8025298fdc56", "offline model ref correction parent SHA mismatch")
    _require(ref_correction.get("scientific_contract_changed") is False and ref_correction.get("data_tasks_arms_seeds_budget_changed") is False, "offline model ref correction changed the scientific contract")
    _require(ref_correction.get("fourth_attempt_receipt_path") == FORMAL_ATTEMPT_4.as_posix(), "fourth attempt receipt path mismatch")
    _require(_sha256(root / FORMAL_ATTEMPT_4) == ref_correction.get("fourth_attempt_receipt_sha256") == "1474ad6db1d15499fff0a621e745fb6bfdd9cdec153bf23b6c0eb65645e8ecf0", "fourth attempt receipt SHA mismatch")
    _require(_sha256(root / Path(ref_correction.get("partial_reconciliation_path", ""))) == ref_correction.get("partial_reconciliation_sha256") == "cf10e8643d5fccc408ea4799959e74c126e16b9deaa3b89f2f807d112d778bb5", "partial reconciliation SHA mismatch")
    _require(ref_correction.get("validation_path") == OFFLINE_MODEL_REF_CORRECTION_VALIDATION.as_posix(), "offline model ref correction validation path mismatch")
    _require(_sha256(root / OFFLINE_MODEL_REF_CORRECTION_VALIDATION) == ref_correction.get("validation_sha256") == "66fb8f6ae5341d27fb00d1150c30757b286b4c26db844cc2e0ea4bbd4b6afae2", "offline model ref correction validation SHA mismatch")
    fourth_attempt = _read(root / FORMAL_ATTEMPT_4)
    _require(fourth_attempt.get("status") == "invalid-engineering-partial-training-zero-evaluation", "fourth attempt status mismatch")
    _require(fourth_attempt.get("formal_attempts") == 0 and fourth_attempt.get("campaign_cumulative_gpu_hours") == 0.014166666666666666 and fourth_attempt.get("fresh_panel_consumed") is False, "fourth attempt denominator/GPU-hours mismatch")

    manifest_errors = _validate_manifest(evaluation)
    _require(not manifest_errors, "; ".join(manifest_errors))
    _require(_sha256(eval_path) == prereg["evaluation"]["manifest_file_sha256"], "evaluation file hash mismatch")
    _require(evaluation.get("manifest_sha256") == prereg["evaluation"]["manifest_content_sha256"], "evaluation content hash mismatch")
    selection = evaluation.get("task_selection") or {}
    _require(selection.get("prior_identity_count") == 493, "prior identity count mismatch")
    _require(selection.get("prior_identity_digest") == "24efaed40ff35f5624fc82b4183ce3d58cb72690d686b8627aee2772d7719eeb", "prior identity digest mismatch")
    _require(len(selection.get("prior_identity_indices") or []) == 493, "prior identity ledger was not frozen")
    _require(len(evaluation.get("fresh_tasks") or []) == 48 and evaluation.get("planned_attempts") == 704, "evaluation matrix mismatch")
    h138a = _read(root / H138A_MANIFEST)
    prior_fresh = {
        int(item["goal_idx"]) if isinstance(item, dict) else int(item)
        for item in h138a.get("evaluation_tasks", h138a.get("tasks", []))
    }
    fresh = {int(item["goal_idx"]) for item in evaluation["fresh_tasks"]}
    _require(not (prior_fresh & fresh), "H1.38B fresh panel reuses an H1.38A task")

    unsigned_plans = dict(plans)
    observed_bundle = unsigned_plans.pop("bundle_sha256", None)
    _require(observed_bundle == digest_json(unsigned_plans), "sample-plan self-hash mismatch")
    _require(observed_bundle == prereg["matched_training_contract"]["sample_plan_bundle_sha256"], "sample-plan bundle identity mismatch")
    _require(_sha256(plans_path) == prereg["matched_training_contract"]["sample_plan_file_sha256"], "sample-plan file hash mismatch")
    _require(len(plans.get("increments") or []) == 8, "sample-plan increment count mismatch")

    source = qualification.get("sealed_source") or {}
    _require(qualification.get("schema") == "openevo.webshop-h1.38b-cpu-sandbox-qualification.v1", "qualification schema mismatch")
    _require(qualification.get("status") == "pass", "CPU qualification did not pass")
    _require(qualification.get("gpu_requested") is False, "CPU qualification requested a GPU")
    _require(qualification.get("formal_training_or_evaluation") is False, "qualification consumed formal work")
    _require(source.get("records_sha256") == D8_RECORDS_SHA256 and source.get("selection_sha256") == D8_SELECTION_SHA256, "qualification D8 identity mismatch")
    unsigned_qualification = dict(qualification)
    observed_qualification = unsigned_qualification.pop("qualification_sha256", None)
    _require(observed_qualification == digest_json(unsigned_qualification), "qualification self-hash mismatch")
    qualification_contract = prereg["implementation_qualification"]
    _require(_sha256(qualification_path) == qualification_contract["file_sha256"], "qualification file hash mismatch")
    _require(observed_qualification == qualification_contract["content_sha256"], "qualification content hash mismatch")
    receipt_paths = sorted((root / QUALIFICATION_RECEIPTS).glob("*.json"))
    _require(len(receipt_paths) == qualification_contract["receipt_count"] == 16, "qualification receipt count mismatch")
    declared_receipts = sorted(Path(str(path)).name for path in qualification.get("training_receipts") or [])
    _require(declared_receipts == [path.name for path in receipt_paths], "published qualification receipt set mismatch")
    _, receipt_errors, _ = _validate_training_receipts(evaluation, receipt_paths)
    _require(not receipt_errors, "; ".join(receipt_errors))

    runtime = prereg.get("runtime") or {}
    _require(runtime.get("trainer_helpers_sha256") == PINNED_TRAINER_HELPERS_SHA256, "trainer helper pin mismatch")
    _require(runtime.get("dataset_helper_sha256") == PINNED_DATASET_HELPER_SHA256, "dataset helper pin mismatch")
    _require(prereg.get("budget", {}).get("gpu_hours_cap") == 6.0, "GPU-hour cap mismatch")
    _require(prereg.get("budget", {}).get("maximum_evaluation_attempts") == 704, "attempt cap mismatch")

    implementation_files = {
        "ordinary_trainer_sha256": "scripts/openevo_webshop/ordinary_lora_trainer.py",
        "ordinary_contract_sha256": "scripts/openevo_webshop/ordinary_lora_contract.py",
        "launcher_sha256": "scripts/openevo_webshop/launch_h138b_method_control.sh",
        "sample_plan_freezer_sha256": "scripts/openevo_webshop/freeze_h138b_sample_plans.py",
        "qualification_verifier_sha256": "scripts/openevo_webshop/verify_h138b_qualification.py",
        "reload_verifier_sha256": "scripts/openevo_webshop/verify_adapter_reload.py",
        "reconciliation_sha256": "scripts/openevo_webshop/reconcile_h138b_method_control.py",
        "evaluation_runner_sha256": "scripts/openevo_webshop/h1_closed_loop_tracked.py",
        "inference_load_receipt_sha256": "docs/evidence/remote-runs/2026-08-20/h1.38b-qualification/implementation/inference.py",
    }
    for field, relative in implementation_files.items():
        _require(_sha256(root / relative) == prereg["implementation"][field], f"implementation hash mismatch: {field}")

    return {
        "schema": "openevo.webshop-h1.38b-preregistration-validation.v1",
        "status": "pass",
        "campaign_id": campaign_id,
        "launch_authorized": not blockers,
        "formal_task_consumption_allowed": not blockers,
        "authorized_gpu_uuids": AUTHORIZED_GPU_UUIDS,
        "blockers": blockers,
        "sealed_records_sha256": D8_RECORDS_SHA256,
        "evaluation_manifest_sha256": evaluation["manifest_sha256"],
        "sample_plan_bundle_sha256": observed_bundle,
        "qualification_sha256": observed_qualification,
        "validated_receipts": len(receipt_paths),
        "zero_consumption_engineering_attempts": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(validate(args.repo_root.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
