#!/usr/bin/env python3
"""Fail-closed coverage and claim-boundary checks for the WebShop program ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.openevo_webshop.campaign_router import is_external_successor_router

LEDGER_PATH = Path("docs/science/webshop/program/WEBSHOP_PROGRAM_LEDGER.v1.json")
REPORT_PATH = Path("docs/science/webshop/reports/OPEN_EVO_WEBSHOP_PROGRAM_REPORT_2026-08-21.md")
CURRENT_CAMPAIGN_PATH = Path("configs/experiment/current-campaign.json")
H141_CAMPAIGN_ID = "20260821-0141-h141-magnitude-screen"
H140_CAMPAIGN_ID = "20260820-0140-h140-g2-multigeneration"
H140_R2_CAMPAIGN_ID = "20260820-0140r2-h140-g2-multigeneration"
H141_CONFIG_PATH = Path("configs/experiment/h1.41-magnitude-screen-v2.json")
H141_PATH_CONTRACT = Path("configs/experiment/h1.41-path-contract-v1.json")
H141_RELEASE = Path("configs/experiment/activations/h1.41-magnitude-screen-execution-release-v2.json")
H141_REFERENCE = Path("docs/evidence/remote-runs/2026-08-21/h1.41/h1.41-g1-deterministic-reference-v1.json")
H138_RECONCILIATION_PATH = Path("docs/evidence/remote-runs/2026-08-19/h1.38a-c/evidence/h1.38a-c-combined-reconciliation.json")
H138_CLOSEOUT_PATH = Path("docs/evidence/remote-runs/2026-08-19/h1.38a-c/evidence/h1.38a-c-closeout-summary.json")
WB1_PROTOCOL_PATH = Path("configs/experiment/wb1/protocol.json")
CURRENT_ROUTER_FIELDS = (
    "campaign_id",
    "phase",
    "status",
    "protocol_status",
    "formal_task_consumption_allowed",
    "gpu_allocation_allowed",
    "task_bearing_train_dev_allowed",
)
ALLOWED_RECORD_STATUSES = {
    "sealed-result",
    "diagnostic-only",
    "protocol-only",
    "invalid",
    "superseded-protocol",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssertionError(f"{label}: expected integer, got {value!r}")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AssertionError(f"{label}: expected finite number, got {value!r}")
    return float(value)


def assert_close(actual: Any, expected: Any, label: str) -> None:
    if not math.isclose(require_number(actual, label), require_number(expected, f"{label} expected"), rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def assert_references_are_tracked(root: Path, paths: list[str]) -> None:
    """Reject locally present evidence that would disappear from a clean checkout."""
    inside = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
        text=True,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    missing = sorted(set(paths) - set(tracked))
    if missing:
        raise AssertionError(f"ledger references untracked paths: {missing}")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    ledger = load_json(root / LEDGER_PATH)
    current = load_json(root / CURRENT_CAMPAIGN_PATH)
    report = (root / REPORT_PATH).read_text(encoding="utf-8")
    h138_reconciliation_path = root / H138_RECONCILIATION_PATH
    h138_closeout_path = root / H138_CLOSEOUT_PATH
    h138_reconciliation = load_json(h138_reconciliation_path)
    h138_closeout = load_json(h138_closeout_path)

    expected_configs = {
        path.relative_to(root).as_posix()
        for path in (root / "configs/experiment").glob("h*.json")
    }
    expected_configs.add(WB1_PROTOCOL_PATH.as_posix())
    entries = ledger.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise AssertionError("ledger experiments must be a non-empty list")

    ids: list[str] = []
    covered_configs: list[str] = []
    evidence_paths: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise AssertionError("every experiment entry must be an object")
        experiment_id = entry.get("id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise AssertionError("every experiment entry needs a non-empty id")
        ids.append(experiment_id)
        by_id[experiment_id] = entry
        status = entry.get("record_status")
        if status not in ALLOWED_RECORD_STATUSES:
            raise AssertionError(f"{experiment_id}: unsupported record_status {status!r}")
        config = entry.get("config")
        if config is not None:
            if not isinstance(config, str):
                raise AssertionError(f"{experiment_id}: config must be a path or null")
            covered_configs.append(config)
        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise AssertionError(f"{experiment_id}: evidence must be non-empty")
        for path in evidence:
            if not isinstance(path, str):
                raise AssertionError(f"{experiment_id}: non-string evidence path")
            evidence_paths.append(path)

    if len(ids) != len(set(ids)):
        raise AssertionError("experiment ids must be unique")
    if len(covered_configs) != len(set(covered_configs)):
        raise AssertionError("each ledger config must be covered exactly once")
    covered_set = set(covered_configs)
    if covered_set != expected_configs:
        missing = sorted(expected_configs - covered_set)
        extra = sorted(covered_set - expected_configs)
        raise AssertionError(f"config coverage mismatch: missing={missing}, extra={extra}")

    missing_paths = sorted(
        path for path in covered_configs + evidence_paths if not (root / path).exists()
    )
    if missing_paths:
        raise AssertionError(f"ledger references missing paths: {missing_paths}")
    assert_references_are_tracked(root, covered_configs + evidence_paths)

    ledger_current = ledger.get("current_state", {})
    for key in CURRENT_ROUTER_FIELDS:
        if key not in current:
            raise AssertionError(f"current-campaign {key} is missing")
        if ledger_current.get(key) != current.get(key):
            raise AssertionError(f"ledger current_state {key}: expected {current.get(key)!r}, got {ledger_current.get(key)!r}")
    h141_current = current.get("campaign_id") == H141_CAMPAIGN_ID
    r2_current = current.get("campaign_id") == H140_R2_CAMPAIGN_ID
    external_successor_current = is_external_successor_router(
        current,
        {H141_CAMPAIGN_ID, H140_R2_CAMPAIGN_ID, H140_CAMPAIGN_ID},
        predecessor_routes={str(H141_CONFIG_PATH)},
    )
    latest_classification = (current.get("latest_event") or {}).get("classification")
    r2_e2_passed = r2_current and latest_classification == "r2-e2-harvest-selection-passed"
    r2_g2_passed = r2_current and latest_classification == "r2-g2-training-passed"
    r2_upstream_failed = r2_current and latest_classification == "r2-upstream-gates-failed"
    r2_qualified = r2_current and latest_classification in {"r2-zero-task-qualification-passed", "r2-formal-execution-release-published", "r2-e2-harvest-selection-passed", "r2-g2-training-passed"}
    r2_released = r2_current and latest_classification == "r2-formal-execution-release-published"
    expected_event = None
    if h141_current:
        h141_closed_measurement_invalid = current.get("status") == "closed-measurement-invalid"
        expected_current = {
            "campaign_id": H141_CAMPAIGN_ID,
            "phase": "H1.41-MD",
            "status": current.get("status"),
            "protocol_status": current.get("protocol_status"),
            "formal_task_consumption_allowed": current.get("formal_task_consumption_allowed"),
        }
        expected_event = (
            {
                "classification": "h1.41-v2-reconciled-measurement-invalid",
                "formal_task_consumption_allowed": False,
                "formal_task_consumption": 384,
                "formal_gpu_hours_consumed": 1.5673375382128392,
                "g2_adapter_created": True,
                "t2_panel_evaluated": False,
            }
            if h141_closed_measurement_invalid
            else {
                "classification": "h1.41-v2-launch-authorized",
                "formal_task_consumption_allowed": True,
                "formal_task_consumption": 0,
                "formal_gpu_hours_consumed": 0.0,
                "g2_adapter_created": False,
                "t2_panel_evaluated": False,
            }
        )
    else:
        expected_current = None
    if not h141_current and r2_upstream_failed:
        expected_protocol_status = "r2-upstream-gates-failed-t2-closed"
    elif r2_g2_passed:
        expected_protocol_status = "r2-g2-training-passed-acquisition-retention-preservation-pending"
    elif r2_e2_passed:
        expected_protocol_status = "r2-e2-harvest-selection-passed-g2-training-pending"
    elif r2_released:
        expected_protocol_status = "r2-formal-execution-release-published-formal-execution-pending"
    elif r2_qualified:
        expected_protocol_status = "r2-zero-task-qualification-passed-formal-execution-pending"
    elif r2_current and current.get("status") == "launch-authorized":
        expected_protocol_status = "r2-launch-authorized-zero-task-qualification-pending"
    else:
        expected_protocol_status = "r2-preregistered" if r2_current else "runtime-gate-blocked"
    if not h141_current and not external_successor_current:
        expected_current = {
            "campaign_id": "20260820-0140r2-h140-g2-multigeneration" if r2_current else "20260820-0140-h140-g2-multigeneration",
            "phase": "H1.40-G2-R2" if r2_current else "H1.40-G2",
            "status": "closed-upstream-gates-failed" if r2_upstream_failed else ("launch-authorized" if r2_current and current.get("status") == "launch-authorized" else ("preregistered-pre-execution" if r2_current else "blocked-pre-execution")),
            "protocol_status": expected_protocol_status,
            "formal_task_consumption_allowed": True if r2_current and current.get("status") == "launch-authorized" else False,
        }
    if expected_current is not None:
        for key, expected in expected_current.items():
            if current.get(key) != expected:
                raise AssertionError(f"current-campaign {key}: expected {expected!r}, got {current.get(key)!r}")

    latest_event = current.get("latest_event")
    if not external_successor_current and not isinstance(latest_event, dict):
        raise AssertionError("current-campaign latest_event is missing")
    if not h141_current and not external_successor_current:
        expected_event = {
            "classification": latest_classification if r2_current else "runtime-gate-blocked",
            "formal_task_consumption_allowed": True if r2_current and current.get("status") == "launch-authorized" else False,
            "formal_task_consumption": 640 if r2_upstream_failed else (256 if r2_e2_passed or r2_g2_passed else 0),
            "formal_gpu_hours_consumed": 3.7551676636612837 if r2_upstream_failed else (2.219149268141514 if r2_g2_passed else (2.185317164430954 if r2_e2_passed else 0.0)),
            "g2_adapter_created": True if r2_g2_passed or r2_upstream_failed else False,
            "t2_panel_evaluated": False,
        }
    if expected_event is not None:
        for key, expected in expected_event.items():
            if latest_event.get(key) != expected:
                raise AssertionError(f"latest_event {key}: expected {expected!r}, got {latest_event.get(key)!r}")
    if h141_current:
        h141_config = load_json(root / H141_CONFIG_PATH)
        h141_path_contract = load_json(root / H141_PATH_CONTRACT)
        h141_release = load_json(root / H141_RELEASE)
        if h141_config.get("campaign_id") != H141_CAMPAIGN_ID or h141_config.get("protocol_revision") != "v2-deterministic-g1-reference" or h141_config.get("formal_task_consumption_allowed") is not False:
            raise AssertionError("H1.41 preregistration is not sealed closed")
        if h141_release.get("status") != "launch-authorized" or h141_release.get("formal_task_consumption_allowed") is not True or h141_release.get("scientific_contract_changed") is not True:
            raise AssertionError("H1.41 v2 release is not launch-authorized with its amendment")
        if h141_release.get("preregistration", {}).get("sha256") != hashlib.sha256((root / H141_CONFIG_PATH).read_bytes()).hexdigest():
            raise AssertionError("H1.41 preregistration hash mismatch")
        if h141_release.get("path_contract", {}).get("sha256") != hashlib.sha256((root / H141_PATH_CONTRACT).read_bytes()).hexdigest() or h141_path_contract.get("campaign_id") != H141_CAMPAIGN_ID:
            raise AssertionError("H1.41 path contract hash or campaign mismatch")
        if h141_release.get("reference_audit", {}).get("path") != str(H141_REFERENCE) or h141_release.get("reference_audit", {}).get("sha256") != hashlib.sha256((root / H141_REFERENCE).read_bytes()).hexdigest():
            raise AssertionError("H1.41 deterministic reference hash mismatch")
        if h141_release.get("zero_state_at_release") != {"formal_task_consumption": 0, "formal_gpu_hours_consumed": 0.0, "g2_adapter_created": False, "t2_panel_evaluated": False, "qualification_status": "not-started"}:
            raise AssertionError("H1.41 release zero state mismatch")
        h141_record_status = by_id.get("h1.41-magnitude-screen", {}).get("record_status")
        expected_h141_record_status = "sealed-result" if h141_closed_measurement_invalid else "protocol-only"
        if h141_record_status != expected_h141_record_status:
            raise AssertionError(f"H1.41 ledger status mismatch: expected {expected_h141_record_status!r}, got {h141_record_status!r}")
    h138b = by_id.get("h1.38b-method-control", {})
    h138b_evidence = h138b.get("evidence", [])
    reconciliation_path = next((path for path in h138b_evidence if path.endswith("reconciliation-summary.json")), None)
    selector_path = next((path for path in h138b_evidence if path.endswith("selector.json")), None)
    if not isinstance(reconciliation_path, str) or not isinstance(selector_path, str):
        raise AssertionError("sealed H1.38B evidence pointers are missing")
    if reconciliation_path not in evidence_paths or selector_path not in evidence_paths:
        raise AssertionError("latest_event closeout evidence is not indexed by the ledger")
    reconciliation = load_json(root / reconciliation_path)
    if reconciliation.get("status") != "complete" or reconciliation.get("attempts") != {"expected": 704, "observed": 704, "missing": 0, "valid": 668, "parser_or_fallback_invalid": 36}:
        raise AssertionError("latest reconciliation denominator mismatch")
    selector = load_json(root / selector_path)
    if selector.get("selector") != "B1" or selector.get("decision") != "H1.39-MR independent frozen replication":
        raise AssertionError("latest selector mismatch")
    stage0_receipt_path = "docs/evidence/remote-runs/2026-08-20/h1.38b-qualification/formal-attempt-f751a76.json"
    if stage0_receipt_path not in evidence_paths:
        raise AssertionError("ordinary-LoRA stage-0 preservation receipt is not indexed")
    stage0_receipt = load_json(root / stage0_receipt_path)
    if stage0_receipt.get("completed_training_increments") != {"ordinary-lora": [0], "sd-lora": []}:
        raise AssertionError("ordinary-LoRA stage-0 preservation receipt mismatch")
    if stage0_receipt.get("ordinary_stage00_training_evidence_sha256") != "5e7dade4f3adf1e13456a6b400b00fa13d740aa622abed6c1bac8d36882cbce2":
        raise AssertionError("ordinary-LoRA stage-0 evidence SHA mismatch")

    if h138b.get("record_status") != "sealed-result":
        raise AssertionError("H1.38B must be recorded as a sealed result after reconciliation")
    if "184 cells" not in str(h138b.get("denominator", "")) or "704 attempts" not in str(h138b.get("denominator", "")):
        raise AssertionError("H1.38B must record the complete formal denominator")
    next_gate = str(ledger_current.get("next_gate", "")).lower()
    if h141_current:
        required_tokens = ("measurement", "t2") if h141_closed_measurement_invalid else ("startup", "deterministic", "live-idle")
        for token in required_tokens:
            if token not in next_gate:
                raise AssertionError(f"H1.41 current_state must route to {token} gate")
    elif r2_current:
        if r2_g2_passed:
            if "acquisition" not in next_gate or "retention" not in next_gate or "preservation" not in next_gate:
                raise AssertionError("R2 G2-passed current_state must route to acquisition, retention, and preservation gates")
        elif r2_e2_passed:
            if "g2" not in next_gate or "increment" not in next_gate:
                raise AssertionError("R2 E2-passed current_state must route to sequential G2 increments")
        elif r2_released:
            if "live-idle" not in next_gate or "harvest" not in next_gate:
                raise AssertionError("R2 released current_state must route to the immediate live-idle gate and E2 harvest")
        elif r2_qualified:
            if "activation" not in next_gate or "formal" not in next_gate:
                raise AssertionError("R2 qualified current_state must route to reviewed formal activation")
        elif r2_upstream_failed:
            if "new" not in next_gate or "preregistration" not in next_gate:
                raise AssertionError("R2 upstream-failed current_state must route to a new preregistration")
        elif current.get("status") == "launch-authorized":
            if "startup" not in next_gate or "qualification" not in next_gate:
                raise AssertionError("R2 active current_state must route to zero-task qualification")
        elif "wrapper" not in next_gate or "activation" not in next_gate:
            raise AssertionError("R2 current_state must route to wrapper/path validation and activation")
    elif external_successor_current:
        if not next_gate or ledger_current.get("campaign_id") != current.get("campaign_id"):
            raise AssertionError("current_state must preserve the external campaign router state")
    elif "unknown" not in next_gate or "receipt" not in next_gate:
        raise AssertionError("current_state must keep H1.40-G2 blocked on the unknown receipt")

    h139 = by_id.get("h1.39-mr-independent-replication", {})
    if h139.get("record_status") != "sealed-result" or "R4 BOTH_REPLICATE" not in str(h139.get("conclusion", "")):
        raise AssertionError("sealed H1.39-MR closeout is missing from the ledger")
    h140 = by_id.get("h1.40-g2-multigeneration", {})
    if h140.get("record_status") != "protocol-only" or h140.get("config") != "configs/experiment/h1.40-g2-multigeneration-v1.json":
        raise AssertionError("H1.40-G2 protocol-only entry is missing")
    h140_r2 = by_id.get("h1.40-g2-multigeneration-r2", {})
    if h140_r2.get("record_status") != "sealed-result" or h140_r2.get("config") != "configs/experiment/h1.40-r2-g2-multigeneration-v1.json":
        raise AssertionError("H1.40-G2-R2 diagnostic entry is missing")

    original = by_id.get("h1.38a-original", {})
    combined = by_id.get("h1.38a-c", {})
    if original.get("record_status") != "invalid":
        raise AssertionError("original H1.38A must remain invalid/incomplete")
    if combined.get("record_status") != "sealed-result":
        raise AssertionError("H1.38A-C must be the sealed final panel")
    denominator = str(combined.get("denominator", ""))
    for token in ("192 cells", "768 attempts", "D4 188", "D8 192", "D16 192"):
        if token not in denominator:
            raise AssertionError(f"H1.38A-C denominator missing {token!r}")
    conclusion = str(combined.get("conclusion", ""))
    for token in ("+0.011660", "-0.005792", "+0.055646", "non-monotonic"):
        if token not in conclusion:
            raise AssertionError(f"H1.38A-C conclusion missing {token!r}")

    # Recompute the headline denominator and inferential fields from the immutable
    # reconciliation, then cross-check the independently sealed closeout. This is
    # intentionally typed: mutually wrong prose/ledger literals must not pass.
    cells = require_int(h138_reconciliation.get("observed_cell_count"), "H1.38 observed cells")
    attempts = require_int(h138_reconciliation.get("observed_attempt_count"), "H1.38 observed attempts")
    if cells != 192 or attempts != 768:
        raise AssertionError(f"H1.38 source denominator changed: cells={cells}, attempts={attempts}")
    arm_names = ("base", "diversity-4", "diversity-8", "diversity-16")
    arm_metrics = h138_reconciliation.get("arm_metrics")
    comparisons = h138_reconciliation.get("comparisons_vs_base")
    closeout_combined = h138_closeout.get("combined")
    if not isinstance(arm_metrics, dict) or not isinstance(comparisons, dict) or not isinstance(closeout_combined, dict):
        raise AssertionError("H1.38 typed metric objects are missing")
    source_valid: dict[str, int] = {}
    source_invalid: dict[str, int] = {}
    source_runtime: dict[str, int] = {}
    for arm in arm_names:
        metric = arm_metrics.get(arm)
        if not isinstance(metric, dict):
            raise AssertionError(f"H1.38 arm metric missing: {arm}")
        if require_int(metric.get("attempt_count"), f"{arm} attempts") != 192:
            raise AssertionError(f"{arm}: expected 192 attempts")
        source_valid[arm] = require_int(metric.get("scientific_valid_count"), f"{arm} valid")
        source_invalid[arm] = require_int(metric.get("parser_or_fallback_failures"), f"{arm} parser invalid")
        source_runtime[arm] = require_int(metric.get("runtime_errors"), f"{arm} runtime invalid")
    if sum(source_valid.values()) != 764 or sum(source_invalid.values()) != 4 or sum(source_runtime.values()) != 0:
        raise AssertionError("H1.38 source valid/invalid denominator changed")
    if require_int(closeout_combined.get("cells"), "closeout cells") != cells or require_int(closeout_combined.get("attempts"), "closeout attempts") != attempts:
        raise AssertionError("H1.38 reconciliation/closeout cell or attempt mismatch")
    for field, source in (("valid_attempts", source_valid), ("parser_or_fallback_invalid", source_invalid), ("runtime_invalid", source_runtime)):
        if closeout_combined.get(field) != source:
            raise AssertionError(f"H1.38 reconciliation/closeout {field} mismatch")

    expected_tokens = [f"{cells} cells", f"{attempts} attempts"]
    for arm in arm_names[1:]:
        comparison = comparisons.get(arm)
        if not isinstance(comparison, dict):
            raise AssertionError(f"H1.38 comparison missing: {arm}")
        delta = require_number(comparison.get("task_weighted_delta"), f"{arm} delta")
        ci = comparison.get("task_cluster_bootstrap_ci95")
        if not isinstance(ci, list) or len(ci) != 2:
            raise AssertionError(f"{arm}: invalid CI")
        ci_numbers = [require_number(value, f"{arm} CI") for value in ci]
        p_value = require_number(comparison.get("task_level_sign_flip_p"), f"{arm} p")
        closeout_deltas = closeout_combined.get("task_weighted_delta_vs_base")
        closeout_cis = closeout_combined.get("task_cluster_bootstrap_ci95")
        closeout_ps = closeout_combined.get("task_level_sign_flip_p")
        if not isinstance(closeout_deltas, dict) or not isinstance(closeout_cis, dict) or not isinstance(closeout_ps, dict):
            raise AssertionError("H1.38 closeout comparison fields are missing")
        assert_close(closeout_deltas.get(arm), delta, f"{arm} closeout delta")
        closeout_ci = closeout_cis.get(arm)
        if not isinstance(closeout_ci, list) or len(closeout_ci) != 2:
            raise AssertionError(f"{arm}: closeout CI missing")
        for index in range(2):
            assert_close(closeout_ci[index], ci_numbers[index], f"{arm} closeout CI[{index}]")
        assert_close(closeout_ps.get(arm), p_value, f"{arm} closeout p")
        expected_tokens.extend((f"{delta:+.6f}", f"{ci_numbers[0]:+.6f}", f"{ci_numbers[1]:+.6f}", f"{p_value:.4f}"))

    sealed_hash = h138_closeout.get("combined_reconciliation_sha256")
    actual_hash = hashlib.sha256(h138_reconciliation_path.read_bytes()).hexdigest()
    if not isinstance(sealed_hash, str) or sealed_hash != actual_hash:
        raise AssertionError(f"H1.38 reconciliation SHA-256 mismatch: {actual_hash} != {sealed_hash!r}")
    derived_claim_text = json.dumps(combined, ensure_ascii=False) + "\n" + report
    missing_derived_tokens = [token for token in expected_tokens if token not in derived_claim_text]
    if missing_derived_tokens:
        raise AssertionError(f"ledger/report missing source-derived H1.38 values: {missing_derived_tokens}")

    recommendations = ledger.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        raise AssertionError("ledger recommendations must be non-empty")
    for recommendation in recommendations:
        if recommendation.get("label") != "inference/recommendation":
            raise AssertionError("every next-step proposal must be labelled inference/recommendation")

    required_report_tokens = (
        "最初问题与判定标准",
        "实验谱系",
        "已证实、未证实与仍未知",
        "为什么失败结果有价值",
        "H1.38A-C",
        "推断/建议",
        "H1.38B continuation 已按冻结合同完成",
        "184/184 cells / 704/704 attempts",
        "selector 为 B1",
        "SD-LoRA 相对 BASE",
        "H1.39-MR",
    )
    missing_report_tokens = [token for token in required_report_tokens if token not in report]
    if missing_report_tokens:
        raise AssertionError(f"program report missing sections/guards: {missing_report_tokens}")

    return {
        "status": "pass",
        "config_count": len(expected_configs),
        "experiment_entry_count": len(entries),
        "evidence_reference_count": len(evidence_paths),
        "current_campaign": current["campaign_id"],
        "h1_38b_launch_authorized": False,
        "h1_38b_standing_uuid_authorization": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
