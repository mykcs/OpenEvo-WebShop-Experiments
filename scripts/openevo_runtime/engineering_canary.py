"""Non-scientific end-to-end canary for Execution Runtime modernization.

The canary writes only to an explicit scratch root outside the repository. It
exercises existing ExecutionGeneration, Passport operation identity, typed
terminal, and event-reconciliation primitives without GPU, model, formal rows,
or final-panel access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.openevo_runtime.event_reconciliation import reconcile_runtime_events
from scripts.openevo_runtime.execution_generation import build_execution_generation
from scripts.openevo_runtime.operation_records import (
    build_operation_intent,
    build_operation_terminal,
    validate_operation_terminal,
)
from scripts.openevo_runtime.result_codes import OperationResult, OperationStatus, ReasonCode
from scripts.openevo_runtime.runtime_events import build_runtime_event
from scripts.openevo_webshop.experiment_passport import canonical_bytes

ROOT = REPO_ROOT
REPORT_SCHEMA = "openevo.execution-runtime-engineering-canary.202609171651"
RECEIPT_SCHEMA = "openevo.engineering-canary-receipt.202609171651"
RUNTIME_FILES = (
    "scripts/openevo_runtime/execution_generation.py",
    "scripts/openevo_runtime/operation_records.py",
    "scripts/openevo_runtime/result_codes.py",
    "scripts/openevo_runtime/runtime_events.py",
    "scripts/openevo_runtime/event_reconciliation.py",
    "scripts/openevo_runtime/engineering_canary.py",
)
PAYLOAD = b"OpenEVO Execution Runtime non-scientific engineering canary\n"


class EngineeringCanaryError(RuntimeError):
    """The canary cannot prove an exact, non-scientific lifecycle."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_payload(value: Mapping[str, Any]) -> str:
    return _sha_bytes(canonical_bytes(value))


def _ref(sha256: str, *, reason: str) -> dict[str, Any]:
    return {
        "kind": "unresolved",
        "repository": None,
        "revision": None,
        "path": None,
        "sha256": sha256,
        "evidence_path": None,
        "reason": reason,
    }


def runtime_package_identity() -> str:
    rows = []
    for relative in RUNTIME_FILES:
        path = ROOT / relative
        rows.append({"path": relative, "sha256": _sha_bytes(path.read_bytes())})
    return _sha_payload({"files": rows})


def _generation(source_git_sha: str) -> dict[str, Any]:
    return build_execution_generation(
        {
            "schema": "openevo.execution-generation.202609170115",
            "execution_git_sha": source_git_sha,
            "science_contract_identity": _sha_bytes(b"NON_SCIENTIFIC_ENGINEERING_CANARY"),
            "runtime_package_identity": runtime_package_identity(),
            "ready_identity": None,
            "owner_release_identity": None,
            "recovery_policy_identity": None,
            "carrier_policy_identity": None,
            "resource_policy_identity": None,
            "artifact_reuse_manifest_identity": None,
        }
    )


def _operation_spec(generation_id: str, package_identity: str) -> dict[str, Any]:
    payload_sha = _sha_bytes(PAYLOAD)
    return {
        "operation_type": "runtime.engineering-canary",
        "operation_schema_version": "1",
        "side_effect_class": "PURE_ENGINEERING",
        "identity_payload": {
            "execution_generation_id": generation_id,
            "code_identity": _ref(package_identity, reason="exact canary runtime package"),
            "config_identity": _ref(
                _sha_bytes(b"NON_SCIENTIFIC_CANARY_CONFIG_202609171651"),
                reason="fixed canary config",
            ),
            "runtime_identity_policy": "NOT_IDENTITY_BEARING",
            "runtime_identity": None,
            "dataset_schedule_identity": None,
            "contract_identities": [
                {
                    "name": "non_scientific_canary",
                    "identity": _ref(
                        _sha_bytes(b"NON_SCIENTIFIC_ENGINEERING_CANARY"),
                        reason="explicit non-scientific contract",
                    ),
                }
            ],
            "input_artifacts": [
                {
                    "artifact_type": "synthetic_canary_payload",
                    "digest_algorithm": "sha256",
                    "digest": payload_sha,
                    "size_bytes": len(PAYLOAD),
                    "logical_role": "canary_input",
                }
            ],
            "stochastic_identity": {},
        },
    }


def _write_new(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_new(path, canonical_bytes(payload) + b"\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineeringCanaryError(f"unreadable canary record: {path.name}") from exc
    if not isinstance(value, dict):
        raise EngineeringCanaryError(f"canary record must be an object: {path.name}")
    return value


def _receipt_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_sha256"}


def _event_reconciliation(intent: Mapping[str, Any], terminal: Mapping[str, Any], receipt_sha: str) -> dict[str, Any]:
    ready = build_runtime_event(
        event_type="OPERATION_READY",
        subject_id=str(intent["intent_id"]),
        source_receipt_identity=str(intent["intent_id"]),
        payload_identity=str(intent["operation_digest"]),
    )
    materialized = build_runtime_event(
        event_type="SIDE_EFFECT_MATERIALIZED",
        subject_id=str(terminal["terminal_id"]),
        source_receipt_identity=receipt_sha,
        payload_identity=str(terminal["terminal_id"]),
    )
    projections = [
        {key: ready[key] for key in ("event_type", "subject_id", "source_receipt_identity", "payload_identity")},
        {key: materialized[key] for key in ("event_type", "subject_id", "source_receipt_identity", "payload_identity")},
    ]
    reconciled = reconcile_runtime_events(projections, [ready, ready])
    return {
        "canonical_event_ids": [event["event_id"] for event in reconciled.events],
        "reconstructed_missing": reconciled.reconstructed_missing,
        "ignored_duplicates": reconciled.ignored_duplicates,
        "authority_source": reconciled.authority_source,
        "framework_history_authority": reconciled.framework_history_authority,
    }


def _readback(operation_dir: Path, intent: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_json(operation_dir / "receipt.json")
    expected_receipt_sha = _sha_payload(_receipt_projection(receipt))
    if receipt.get("receipt_sha256") != expected_receipt_sha:
        raise EngineeringCanaryError("receipt identity mismatch")
    artifact = operation_dir / "artifact.bin"
    artifact_bytes = artifact.read_bytes()
    if _sha_bytes(artifact_bytes) != receipt.get("artifact_sha256"):
        raise EngineeringCanaryError("artifact readback SHA mismatch")
    terminal = _load_json(operation_dir / "terminal.json")
    validate_operation_terminal(terminal)
    if terminal.get("terminal_receipt_identity") != expected_receipt_sha:
        raise EngineeringCanaryError("terminal does not bind exact receipt")
    if terminal.get("operation_digest") != intent.get("operation_digest"):
        raise EngineeringCanaryError("terminal operation identity drift")
    return {
        "mode": "READBACK_EXISTING_TERMINAL",
        "receipt_sha256": expected_receipt_sha,
        "artifact_sha256": receipt["artifact_sha256"],
        "artifact_size_bytes": receipt["artifact_size_bytes"],
        "terminal": terminal,
        "events": _event_reconciliation(intent, terminal, expected_receipt_sha),
    }


def _execute_once(scratch_root: Path, source_git_sha: str) -> dict[str, Any]:
    package_identity = runtime_package_identity()
    generation = _generation(source_git_sha)
    spec = _operation_spec(generation["generation_id"], package_identity)
    admission_binding = _sha_payload(
        {"scope": "NON_SCIENTIFIC_ENGINEERING_CANARY", "source_git_sha": source_git_sha}
    )
    intent = build_operation_intent(spec, admission_binding_identity=admission_binding)
    operation_dir = scratch_root / str(intent["operation_digest"])
    terminal_path = operation_dir / "terminal.json"
    if terminal_path.exists():
        readback = _readback(operation_dir, intent)
        return {**readback, "generation": generation, "intent": intent}
    if operation_dir.exists():
        raise EngineeringCanaryError("partial canary state without terminal; fail closed")
    operation_dir.mkdir(parents=True)
    _write_new_json(operation_dir / "intent.json", intent)
    _write_new(operation_dir / "artifact.bin", PAYLOAD)
    artifact_sha = _sha_bytes(PAYLOAD)
    receipt_projection = {
        "schema": RECEIPT_SCHEMA,
        "operation_digest": intent["operation_digest"],
        "execution_generation_id": generation["generation_id"],
        "source_git_sha": source_git_sha,
        "artifact_sha256": artifact_sha,
        "artifact_size_bytes": len(PAYLOAD),
        "formal_rows_consumed": 0,
        "model_calls": 0,
        "gpu_requests": 0,
        "final_panel_access_count": 0,
        "scientific_state_mutated": False,
        "scientific_authority_granted": False,
    }
    receipt_sha = _sha_payload(receipt_projection)
    receipt = {**receipt_projection, "receipt_sha256": receipt_sha}
    _write_new_json(operation_dir / "receipt.json", receipt)
    result = OperationResult(
        status=OperationStatus.SUCCEEDED,
        reason_code=ReasonCode.NONE,
        operation_id=str(intent["idempotency_key"]),
        scientific_state_mutated=False,
        side_effect_materialized=True,
        message="non-scientific engineering canary materialized once",
    )
    terminal = build_operation_terminal(intent, result, terminal_receipt_identity=receipt_sha)
    _write_new_json(terminal_path, terminal)
    return {
        "mode": "MATERIALIZED_NEW",
        "receipt_sha256": receipt_sha,
        "artifact_sha256": artifact_sha,
        "artifact_size_bytes": len(PAYLOAD),
        "terminal": terminal,
        "events": _event_reconciliation(intent, terminal, receipt_sha),
        "generation": generation,
        "intent": intent,
    }


def run_canary_cycle(scratch_root: Path, source_git_sha: str) -> dict[str, Any]:
    scratch_root = scratch_root.resolve()
    if scratch_root == ROOT or ROOT in scratch_root.parents:
        raise EngineeringCanaryError("scratch_root must be outside the repository")
    first = _execute_once(scratch_root, source_git_sha)
    if first["mode"] != "MATERIALIZED_NEW":
        raise EngineeringCanaryError("fresh canary root did not materialize a new operation")
    before = {
        "artifact": (scratch_root / first["intent"]["operation_digest"] / "artifact.bin").read_bytes(),
        "terminal": (scratch_root / first["intent"]["operation_digest"] / "terminal.json").read_bytes(),
    }
    second = _execute_once(scratch_root, source_git_sha)
    after = {
        "artifact": (scratch_root / first["intent"]["operation_digest"] / "artifact.bin").read_bytes(),
        "terminal": (scratch_root / first["intent"]["operation_digest"] / "terminal.json").read_bytes(),
    }
    if second["mode"] != "READBACK_EXISTING_TERMINAL" or before != after:
        raise EngineeringCanaryError("second execution rewrote or duplicated the sealed operation")
    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS_NON_SCIENTIFIC_ENGINEERING_CANARY",
        "source_git_sha": source_git_sha,
        "execution_generation_id": first["generation"]["generation_id"],
        "runtime_package_identity": first["generation"]["runtime_package_identity"],
        "operation_digest": first["intent"]["operation_digest"],
        "terminal_id": first["terminal"]["terminal_id"],
        "receipt_sha256": first["receipt_sha256"],
        "artifact_sha256": first["artifact_sha256"],
        "artifact_size_bytes": first["artifact_size_bytes"],
        "first_pass": first["mode"],
        "second_pass": second["mode"],
        "materialization_count": 1,
        "readback_count": 1,
        "event_reconciliation": first["events"],
        "formal_rows_consumed": 0,
        "model_calls": 0,
        "gpu_requests": 0,
        "final_panel_access_count": 0,
        "scientific_state_mutated": False,
        "scientific_authority_granted": False,
        "live_formal_execution_mutated": False,
        "scratch_is_disposable": True,
    }


def _current_git_sha_and_clean() -> str:
    sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
    if status.strip():
        raise EngineeringCanaryError("repository must be clean before exact-source canary execution")
    return sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    actual_sha = _current_git_sha_and_clean()
    if actual_sha != args.expected_git_sha:
        raise EngineeringCanaryError("expected_git_sha does not match clean checkout HEAD")
    if args.scratch_root.exists():
        raise EngineeringCanaryError("scratch_root must be new for the canary cycle")
    report = run_canary_cycle(args.scratch_root, actual_sha)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(canonical_bytes(report) + b"\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
