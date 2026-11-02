#!/usr/bin/env python3
"""Read-only recovery planning from Experiment Passport / Registry evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop.experiment_passport import PassportError, load_json
from scripts.openevo_webshop.experiment_registry import (
    RegistryError,
    load_bound_passport,
    load_current_registry,
    resolve_experiment,
    validate_registry,
)
from scripts.openevo_webshop.experiment_verify import VerifyError, verify_experiment

ROOT = Path(__file__).resolve().parents[2]
STATE_MODEL_POLICY = "docs/operations/publication/RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md"
VERDICTS = {
    "SAFE_TO_PREPARE_RESUME", "COMPLETE_NO_RESUME_NEEDED", "NO_RESUME_HISTORICAL",
    "MISSING_ASSET", "IDENTITY_MISMATCH", "MANUAL_DECISION_REQUIRED",
    "LIVE_STATE_NEWER_THAN_ARCHIVE",
}


class RevivePlanError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_live_state(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    source = entry.get("state_source") or {}
    if source.get("type") != "server_json":
        return None
    path = Path(str(source.get("path") or ""))
    if not path.is_absolute() or not path.is_file():
        return None
    try:
        return load_json(path, max_bytes=8 * 1024 * 1024)
    except PassportError as exc:
        raise RevivePlanError(f"live state unreadable: {exc}") from exc


def _verified_boundary(run_root: Path, archive_receipt: Path) -> dict[str, Any] | None:
    try:
        receipt = load_json(archive_receipt, max_bytes=2 * 1024 * 1024)
    except PassportError:
        return None
    if receipt.get("status") != "PASS" or type(receipt.get("round_index")) is not int:
        return None
    round_index = int(receipt["round_index"])
    local_round = receipt.get("local_round_path")
    if isinstance(local_round, str) and Path(local_round).is_absolute():
        round_dir = Path(local_round)
        if not _within(run_root, round_dir):
            return None
    else:
        round_dir = run_root / f"round-{round_index:03d}"
    round_complete = round_dir / "ROUND_COMPLETE.json"
    state_after = round_dir / "state-after.json"
    if not round_complete.is_file() or not state_after.is_file():
        return None
    wanted = receipt.get("round_complete_sha256")
    if not isinstance(wanted, str) or _sha256(round_complete) != wanted:
        return None
    try:
        complete = load_json(round_complete, max_bytes=8 * 1024 * 1024)
        state = load_json(state_after, max_bytes=8 * 1024 * 1024)
    except PassportError:
        return None
    if complete.get("status") != "PASS" or complete.get("round_index") != round_index:
        return None
    if complete.get("state_id_after") != state.get("state_id"):
        return None
    if state.get("next_round") != round_index + 1:
        return None
    return {
        "round_index": round_index,
        "next_round": round_index + 1,
        "state_id": state.get("state_id"),
        "rollouts_consumed": state.get("rollouts_consumed"),
        "round_complete_path": str(round_complete),
        "round_complete_sha256": wanted,
        "state_after_path": str(state_after),
        "state_after_sha256": _sha256(state_after),
        "archive_receipt_path": str(archive_receipt),
        "archive_receipt_sha256": _sha256(archive_receipt),
        "archive_policy": receipt.get("archive_policy"),
        "remote_target": receipt.get("remote_target"),
        "remote_archive_deferred": bool(receipt.get("remote_archive_deferred")),
        "selection_evidence": "verified archive receipt content + bound ROUND_COMPLETE hash + state-after identity",
    }


def collect_receipt_chain(run_root: Path, *, limit: int = 1024) -> list[dict[str, Any]]:
    archive_root = run_root / "archive-receipts"
    if not archive_root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for index, path in enumerate(sorted(archive_root.glob("*.json"))):
        if index >= limit:
            raise RevivePlanError("archive receipt scan exceeded bounded limit")
        boundary = _verified_boundary(run_root, path)
        if boundary is not None:
            out.append(boundary)
    by_round: dict[int, dict[str, Any]] = {}
    for item in out:
        idx = int(item["round_index"])
        prior = by_round.get(idx)
        if prior is None:
            by_round[idx] = item
            continue
        identity = (item.get("round_complete_sha256"), item.get("state_id"), item.get("next_round"))
        prior_identity = (prior.get("round_complete_sha256"), prior.get("state_id"), prior.get("next_round"))
        if identity != prior_identity:
            raise RevivePlanError(f"conflicting verified archive receipts for round {idx}")
    return [by_round[idx] for idx in sorted(by_round)]

def find_last_safe_boundary(run_root: Path, live_state: Mapping[str, Any] | None) -> dict[str, Any]:
    boundaries = collect_receipt_chain(run_root)
    if not boundaries:
        return {"status": "NOT_OBSERVED", "reason": "no verified archive-bound round receipt is locally observable"}
    latest = boundaries[-1]
    result = {
        "status": "PASS",
        "reason": "latest boundary selected from verified receipt contents, not directory numbering",
        "boundary": latest,
        "verified_boundary_count": len(boundaries),
    }
    if live_state is None:
        result["reconciliation"] = "LIVE_NOT_OBSERVED"
        return result
    live_next = live_state.get("next_round")
    live_state_id = live_state.get("state_id")
    archive_next = latest["next_round"]
    if type(live_next) is int and live_next > archive_next:
        result["reconciliation"] = "LIVE_STATE_NEWER_THAN_ARCHIVE"
    elif type(live_next) is int and live_next < archive_next:
        result["status"] = "FAIL"
        result["reconciliation"] = "ARCHIVE_AHEAD_OF_LIVE"
    elif live_state_id != latest.get("state_id"):
        result["status"] = "FAIL"
        result["reconciliation"] = "STATE_ID_MISMATCH"
    else:
        result["reconciliation"] = "SYNCED"
    return result

def _file_requirement(role: str, path: str | None, expected_sha: str | None) -> dict[str, Any]:
    item: dict[str, Any] = {"role": role, "path": path, "expected_sha256": expected_sha}
    if not path:
        item.update(status="MISSING", reason="receipt does not identify required path")
        return item
    target = Path(path)
    if not target.is_file():
        item.update(status="MISSING", reason="required artifact is not locally observable")
        return item
    if expected_sha and _sha256(target) != expected_sha:
        item.update(status="IDENTITY_MISMATCH", reason="artifact hash mismatch")
        return item
    item.update(status="PASS", reason="artifact exists and hash matches" if expected_sha else "artifact exists; receipt has no payload hash")
    return item


def collect_carrier_requirements(boundary: Mapping[str, Any]) -> list[dict[str, Any]]:
    round_complete = Path(str(boundary["round_complete_path"]))
    round_dir = round_complete.parent
    result: list[dict[str, Any]] = []
    roles = {
        "sd_lora": round_dir / "sd-lora" / "terminal-receipt.json",
        "text_memory": round_dir / "text-memory" / "terminal-receipt.json",
        "skill_bundle": round_dir / "skill-bundle" / "terminal-receipt.json",
        "agent_system": round_dir / "agent-system" / "terminal-receipt.json",
    }
    for role, receipt_path in roles.items():
        if not receipt_path.is_file():
            result.append({"role": role, "status": "MISSING", "receipt_path": str(receipt_path), "reason": "terminal receipt missing"})
            continue
        doc = load_json(receipt_path, max_bytes=4 * 1024 * 1024)
        base = {
            "role": role,
            "status": "PASS",
            "receipt_path": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "terminal_status": doc.get("status"),
        }
        if role == "sd_lora":
            update = doc.get("update_receipt") or {}
            adapter_dir = Path(str(doc.get("adapter_path") or update.get("adapter_path") or ""))
            adapter_file = adapter_dir / "adapter_model.safetensors"
            expected = update.get("adapter_digest_after")
            req = _file_requirement(role, str(adapter_file) if str(adapter_dir) else None, expected if isinstance(expected, str) else None)
            base.update(req)
            base["adapter_dir"] = str(adapter_dir) if str(adapter_dir) else None
            base["artifact_id"] = doc.get("artifact_id") or update.get("artifact_id")
        else:
            expected = doc.get("output_sha256")
            req = _file_requirement(role, doc.get("output_path"), expected if isinstance(expected, str) else None)
            base.update(req)
        result.append(base)
    return result


def _static_requirements(passport: Mapping[str, Any]) -> dict[str, Any]:
    science = passport["scientific_identity"]
    execution = passport["initial_execution_identity"]
    return {
        "code": {
            "experiment_repository": execution["experiment_repository"],
            "initial_experiment_sha": execution["initial_experiment_sha"],
            "frozen_science_sha": execution["frozen_science_sha"],
            "upstream_repository": execution["upstream_repository"],
            "upstream_openevo_sha": execution["upstream_openevo_sha"],
            "initial_ray_sha": execution["initial_ray_sha"],
        },
        "runtime": {
            "image_digest": execution["runtime_image_digest"],
            "digest_kind": execution["runtime_digest_kind"],
            "runtime_fingerprint": execution["runtime_fingerprint"],
            "asset_lock": execution["runtime_asset_lock"],
        },
        "model": science["model"],
        "data": {
            "task_manifest": science["task_manifest"],
            "training_data_identity": science["training_data_identity"],
            "harness": science["harness"],
            "seed_policy": science["seed_policy"],
        },
    }


def _resource_requirements(live_state: Mapping[str, Any] | None) -> dict[str, Any]:
    if live_state is None:
        return {
            "allocation_performed": False,
            "status": "NOT_OBSERVED",
            "instruction": "Re-read current GPU ownership/lease before any future execution; revive-plan never allocates resources.",
        }
    return {
        "allocation_performed": False,
        "status": "OBSERVED_HISTORICAL_OR_CURRENT",
        "rollout_gpu_uuids": list(live_state.get("twin_rollout_gpu_uuids") or []),
        "sd_gpu_uuid": live_state.get("twin_sd_designated_gpu_uuid"),
        "instruction": "Observed resource identity is descriptive only; current campaign/release/live GPU gate must be re-run before execution.",
    }


def _latest_lineage_receipt(verification: Mapping[str, Any]) -> dict[str, Any] | None:
    layer = (verification.get("layers") or {}).get("EXEC_LINEAGE") or {}
    chain = layer.get("chain") or []
    if not chain:
        return None
    edge = chain[-1]
    return {key: edge.get(key) for key in ("previous", "successor", "evidence", "sha256", "proof") if edge.get(key) is not None}


RECOVERY_OBSERVATION_PREFIX = "openevo.experiment-recovery-observation."


def validate_durable_recovery_observation(
    observation: Mapping[str, Any], entry: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    schema = observation.get("schema")
    if not isinstance(schema, str) or not schema.startswith(RECOVERY_OBSERVATION_PREFIX):
        errors.append("unsupported recovery observation schema")
    if observation.get("experiment_id") != entry.get("experiment_id"):
        errors.append("experiment_id mismatch")
    if observation.get("passport_sha256") != entry.get("passport_sha256"):
        errors.append("passport_sha256 mismatch")
    state = observation.get("last_observed_state")
    boundary = observation.get("last_safe_boundary")
    if not isinstance(state, Mapping):
        errors.append("last_observed_state missing")
    if not isinstance(boundary, Mapping):
        errors.append("last_safe_boundary missing")
    if isinstance(state, Mapping) and isinstance(boundary, Mapping):
        round_index = boundary.get("round_index")
        next_round = boundary.get("next_round")
        if type(round_index) is not int or round_index < 0:
            errors.append("last_safe_boundary.round_index must be a non-negative integer")
        if type(next_round) is not int or (type(round_index) is int and next_round != round_index + 1):
            errors.append("last_safe_boundary.next_round must equal round_index + 1")
        if boundary.get("state_id") != state.get("state_id"):
            errors.append("last_safe_boundary state_id disagrees with last_observed_state")
        if boundary.get("next_round") != state.get("next_round"):
            errors.append("last_safe_boundary next_round disagrees with last_observed_state")
        if type(boundary.get("verified_boundary_count")) is not int or boundary.get("verified_boundary_count", 0) < 1:
            errors.append("verified_boundary_count must be a positive integer")
    remote = observation.get("remote_artifact_pointers") or []
    if not isinstance(remote, list):
        errors.append("remote_artifact_pointers must be a list")
    else:
        for index, item in enumerate(remote):
            if not isinstance(item, Mapping):
                errors.append(f"remote_artifact_pointers[{index}] must be an object")
                continue
            recovery_state = item.get("recovery_state")
            if recovery_state is not None and recovery_state not in {
                "SERVER_ONLY", "REMOTE_BACKED_UP", "RECOVERY_VERIFIED"
            }:
                errors.append(
                    f"remote_artifact_pointers[{index}].recovery_state is unsupported"
                )
            disclosure_state = item.get("disclosure_state")
            if disclosure_state is not None and disclosure_state not in {
                "PRIVATE", "EMBARGOED", "PUBLIC", "NEVER_PUBLIC"
            }:
                errors.append(
                    f"remote_artifact_pointers[{index}].disclosure_state is unsupported"
                )
            provider_visibility = item.get("provider_visibility")
            if provider_visibility is not None and provider_visibility not in {
                "private", "public", "unknown", "not_applicable"
            }:
                errors.append(
                    f"remote_artifact_pointers[{index}].provider_visibility is unsupported"
                )
            if item.get("status") == "REMOTE_VERIFIED_EXACT":
                if recovery_state is not None and recovery_state != "RECOVERY_VERIFIED":
                    errors.append(
                        f"remote_artifact_pointers[{index}].recovery_state must be RECOVERY_VERIFIED for REMOTE_VERIFIED_EXACT"
                    )
                for key in ("provider", "repository", "revision", "path_in_repo"):
                    if not isinstance(item.get(key), str) or not item.get(key):
                        errors.append(f"remote_artifact_pointers[{index}].{key} is required")
                if item.get("revision") == "main":
                    errors.append(f"remote_artifact_pointers[{index}].revision must be immutable, not main")
    return errors


def load_durable_recovery_observation(
    repo_root: Path, entry: Mapping[str, Any]
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for rel in entry.get("latest_authority_paths") or []:
        path = (repo_root / str(rel)).resolve()
        if not _within(repo_root, path) or not path.is_file() or path.suffix != ".json":
            continue
        try:
            doc = load_json(path, max_bytes=4 * 1024 * 1024)
        except PassportError:
            continue
        schema = doc.get("schema") if isinstance(doc, Mapping) else None
        if not isinstance(schema, str) or not schema.startswith(RECOVERY_OBSERVATION_PREFIX):
            continue
        errors = validate_durable_recovery_observation(doc, entry)
        if errors:
            raise RevivePlanError("durable recovery observation invalid: " + "; ".join(errors))
        item = dict(doc)
        item["_receipt_path"] = str(path.relative_to(repo_root))
        item["_receipt_sha256"] = _sha256(path)
        candidates.append(item)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: str(item.get("observed_at") or ""))[-1]


def _boundary_from_recovery_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    boundary = dict(observation["last_safe_boundary"])
    return {
        "status": "PASS",
        "reason": "last safe boundary recovered from a Passport-bound durable repository observation; live server state is not required for this historical recovery projection",
        "boundary": boundary,
        "verified_boundary_count": boundary.get("verified_boundary_count"),
        "reconciliation": "DURABLE_OBSERVATION_ONLY",
        "observation_scope": observation.get("observation_scope"),
        "observation_receipt": {
            "path": observation.get("_receipt_path"),
            "sha256": observation.get("_receipt_sha256"),
        },
    }


def _carriers_from_recovery_observation(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    carriers: list[dict[str, Any]] = []
    for raw in observation.get("carriers") or []:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        item["status"] = item.get("status") or item.get("last_local_observation") or "NOT_OBSERVED"
        carriers.append(item)
    return carriers

def _verdict_registered(
    verification: Mapping[str, Any],
    live_state: Mapping[str, Any] | None,
    boundary: Mapping[str, Any],
    missing_items: list[Mapping[str, Any]],
) -> str:
    if verification.get("identity") != "PASS" or verification.get("overall") == "FAIL":
        return "IDENTITY_MISMATCH"
    lifecycle = str((live_state or {}).get("status") or "")
    if boundary.get("reconciliation") == "LIVE_STATE_NEWER_THAN_ARCHIVE":
        return "LIVE_STATE_NEWER_THAN_ARCHIVE"
    if boundary.get("status") == "FAIL":
        return "IDENTITY_MISMATCH"
    if missing_items:
        return "MISSING_ASSET"
    if lifecycle in {"STAGE2_SEALED", "COMPLETE"}:
        return "COMPLETE_NO_RESUME_NEEDED"
    if live_state is None:
        return "MANUAL_DECISION_REQUIRED"
    if lifecycle in {"BLOCKED", "PAUSED", "STOPPED", "INTERRUPTED"}:
        return "SAFE_TO_PREPARE_RESUME"
    if lifecycle == "RUNNING":
        return "MANUAL_DECISION_REQUIRED"
    return "MANUAL_DECISION_REQUIRED"


def _recommended_next(experiment_id: str, verdict: str, lifecycle: str | None) -> str | None:
    if verdict in {"COMPLETE_NO_RESUME_NEEDED", "NO_RESUME_HISTORICAL"}:
        return None
    if verdict == "LIVE_STATE_NEWER_THAN_ARCHIVE":
        return None
    if verdict == "MISSING_ASSET":
        return f"./scripts/oe experiments verify {experiment_id}  # after restoring only the exact missing artifact; do not retrain"
    if lifecycle == "RUNNING":
        return None
    return f"./scripts/oe experiments verify {experiment_id}  # re-check identity before any separate execution-authority workflow"

def _registered_plan(root: Path, experiment_or_alias: str) -> dict[str, Any]:
    registry, router = load_current_registry(root)
    errors = validate_registry(registry, root)
    if errors:
        raise RevivePlanError("Registry validation failed: " + "; ".join(errors))
    entry = resolve_experiment(registry, experiment_or_alias)
    passport = load_bound_passport(root, entry)
    verification = verify_experiment(root, experiment_or_alias)
    live_state = _load_live_state(entry)
    recovery_observation = load_durable_recovery_observation(root, entry)
    run_root = Path(passport["initial_execution_identity"]["initial_run_root"])
    observation_used = not run_root.is_dir() and recovery_observation is not None
    if run_root.is_dir():
        boundary = find_last_safe_boundary(run_root, live_state)
    elif observation_used:
        boundary = _boundary_from_recovery_observation(recovery_observation)
    else:
        boundary = {
            "status": "NOT_OBSERVED",
            "reason": "initial run root is not visible on this host and no bound durable recovery observation is registered",
            "reconciliation": "LIVE_NOT_OBSERVED",
        }
    carriers: list[dict[str, Any]] = []
    if run_root.is_dir() and boundary.get("status") == "PASS" and isinstance(boundary.get("boundary"), Mapping):
        carriers = collect_carrier_requirements(boundary["boundary"])
    elif observation_used:
        carriers = _carriers_from_recovery_observation(recovery_observation)
    missing_items = [item for item in carriers if item.get("status") in {"MISSING", "MISSING_ASSET", "IDENTITY_MISMATCH"}]
    if observation_used:
        missing_items.extend(
            dict(item) for item in recovery_observation.get("artifact_requirements") or []
            if isinstance(item, Mapping) and item.get("status") in {"MISSING", "MISSING_ASSET", "IDENTITY_MISMATCH"}
        )
    effective_state = live_state or ((recovery_observation or {}).get("last_observed_state") if observation_used else None)
    verdict = _verdict_registered(verification, effective_state, boundary, missing_items)
    lifecycle = str((effective_state or {}).get("status") or "") or None
    recovery_receipt = _latest_lineage_receipt(verification)
    if recovery_receipt is None and observation_used:
        recovery_receipt = {
            "path": recovery_observation.get("_receipt_path"),
            "sha256": recovery_observation.get("_receipt_sha256"),
            "scope": recovery_observation.get("observation_scope"),
        }
    return {
        "schema": "openevo.experiment-revive-plan.202609120110",
        "mutation": False,
        "experiment_id": entry["experiment_id"],
        "requested_identity": experiment_or_alias,
        "passport_sha256": entry["passport_sha256"],
        "registry": router,
        "verification": verification,
        "current_or_last_observed_state": {
            "status": lifecycle or "NOT_OBSERVED",
            "state_id": (effective_state or {}).get("state_id"),
            "next_round": (effective_state or {}).get("next_round"),
            "rollouts_consumed": (effective_state or {}).get("rollouts_consumed"),
            "execution_sha": (effective_state or {}).get("twin_control_git_sha") or (effective_state or {}).get("execution_sha"),
            "observation_kind": "LIVE" if live_state is not None else ("DURABLE_RECOVERY_OBSERVATION" if observation_used else "NOT_OBSERVED"),
        },
        "last_safe_boundary": boundary,
        "requirements": _static_requirements(passport),
        "carriers": carriers,
        "artifacts": {
            "state_model_policy": passport["artifact_policy"].get("state_model_policy") or STATE_MODEL_POLICY,
            "passport_policy": passport["artifact_policy"],
            "archive": (boundary.get("boundary") or {}) if isinstance(boundary.get("boundary"), Mapping) else None,
            "remote_artifacts": list((recovery_observation or {}).get("remote_artifact_pointers") or []) if observation_used else [],
        },
        "missing_items": missing_items,
        "latest_valid_recovery_receipt": recovery_receipt,
        "resource_requirements": _resource_requirements(live_state),
        "recommended_next_command": _recommended_next(entry["experiment_id"], verdict, lifecycle),
        "verdict": verdict,
        "execution_action": "NONE",
    }


def _legacy_plan(root: Path, experiment_id: str) -> dict[str, Any]:
    from scripts.openevo_webshop import experiment_console as console
    catalog = console._read_json(root / "configs/experiment/experiment-console-202609110009.json")
    errors = console.validate_catalog(root, catalog)
    if errors:
        raise RevivePlanError("legacy console catalog invalid: " + "; ".join(errors))
    row = next((item for item in catalog["experiments"] if item.get("id") == experiment_id), None)
    if row is None:
        raise RevivePlanError(f"unknown experiment identity or alias: {experiment_id}")
    observation = console.observe(root, row)
    policy = str((row.get("resume") or {}).get("policy") or "")
    if policy in {"NO_RESUME", "NO_RESUME_HISTORICAL"}:
        verdict = "NO_RESUME_HISTORICAL"
    elif policy == "COMPLETE_NO_RESUME_NEEDED":
        verdict = "COMPLETE_NO_RESUME_NEEDED"
    else:
        verdict = "MANUAL_DECISION_REQUIRED"
    return {
        "schema": "openevo.experiment-revive-plan.202609120110",
        "mutation": False,
        "experiment_id": experiment_id,
        "legacy_compatibility": True,
        "passport_sha256": None,
        "current_or_last_observed_state": {
            "status": observation.get("effective_status"),
            "observation_scope": observation.get("observation_scope"),
            "metrics": observation.get("metrics") or {},
        },
        "last_safe_boundary": {"status": "NOT_APPLICABLE", "reason": "legacy row has no Passport receipt-chain contract"},
        "requirements": {
            "model": row.get("model"),
            "runtime": row.get("runtime"),
            "authority_paths": list(row.get("authority_paths") or []),
        },
        "carriers": [],
        "artifacts": {
            "state_model_policy": STATE_MODEL_POLICY,
            "authority_paths": list(row.get("authority_paths") or []),
        },
        "missing_items": [],
        "latest_valid_recovery_receipt": None,
        "resource_requirements": {
            "allocation_performed": False,
            "status": "NOT_OBSERVED",
            "instruction": "Legacy console is read-only; no resources are allocated by revive-plan.",
        },
        "recommended_next_command": None if verdict != "MANUAL_DECISION_REQUIRED" else f"./scripts/oe experiments verify {experiment_id}",
        "verdict": verdict,
        "execution_action": "NONE",
    }


def build_revive_plan(repo_root: Path, experiment_or_alias: str) -> dict[str, Any]:
    try:
        return _registered_plan(repo_root, experiment_or_alias)
    except (RegistryError, VerifyError, RevivePlanError) as exc:
        if "unknown experiment identity or alias" not in str(exc):
            raise RevivePlanError(str(exc)) from exc
        return _legacy_plan(repo_root, experiment_or_alias)

def render_revive_plan(plan: Mapping[str, Any]) -> str:
    state = plan["current_or_last_observed_state"]
    boundary = plan["last_safe_boundary"]
    lines = [
        f"Experiment: {plan['experiment_id']}",
        f"Verdict: {plan['verdict']}",
        f"Current/last observed state: {state.get('status')}",
        f"Last sealed safe boundary: {((boundary.get('boundary') or {}).get('round_index') if isinstance(boundary.get('boundary'), Mapping) else boundary.get('status'))}",
        f"Passport: {plan.get('passport_sha256') or 'legacy / not backfilled'}",
        "",
        "Required code/runtime/model/data:",
        json.dumps(plan.get("requirements") or {}, ensure_ascii=False, sort_keys=True),
        "",
        "Required adapters/carriers:",
    ]
    carriers = plan.get("carriers") or []
    lines.extend(
        f"- {item.get('role')}: {item.get('status')} path={item.get('path') or item.get('adapter_dir') or '-'} sha={item.get('expected_sha256') or '-'}"
        for item in carriers
    )
    if not carriers:
        lines.append("- none observed / not applicable")
    lines.extend([
        "",
        f"Latest recovery/continuation receipt: {plan.get('latest_valid_recovery_receipt') or 'none observed'}",
        f"Missing items: {plan.get('missing_items') or 'none'}",
        f"GPU/resource requirement: {plan.get('resource_requirements')}",
        f"Recommended next command: {plan.get('recommended_next_command') or 'NONE'}",
        "Execution action: NONE — revive-plan never starts Docker, Ray, a controller, or GPU work.",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_revive_plan(args.repo_root.expanduser().resolve(), args.experiment_id)
    except (RevivePlanError, RegistryError, PassportError, VerifyError) as exc:
        print(f"experiment-revive-plan: {exc}")
        return 2
    print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) if args.json else render_revive_plan(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
