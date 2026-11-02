#!/usr/bin/env python3
"""Read-only Experiment Registry identity routing and Passport projection."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    load_json,
    load_passport,
    repository_path,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schemas/experiment-registry-202609112120.schema.json"
ROUTER_PATH = ROOT / "configs/experiment/current-experiment-registry.json"
SCHEMA_ID = "openevo.experiment-registry.202609112120"
ROLE = "identity-routing-only; never live state or execution authority"
LIFECYCLES = {"ACTIVE_FORMAL", "COMPLETE_SEALED", "HISTORICAL_NO_RESUME", "PAUSED_MANUAL_REVIEW", "DIAGNOSTIC_ONLY"}


class RegistryError(ValueError):
    pass
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_repo_path(root: Path, value: str) -> Path:
    try:
        return repository_path(root, value)
    except PassportError as exc:
        raise RegistryError(str(exc)) from exc


def load_current_registry(repo_root: Path = ROOT) -> tuple[dict[str, Any], dict[str, str]]:
    router_path = repo_root / "configs/experiment/current-experiment-registry.json"
    router = load_json(router_path)
    if set(router) != {"path", "sha256"}:
        raise RegistryError("stable Registry router must contain only path + sha256")
    rel, wanted = router.get("path"), router.get("sha256")
    if not isinstance(rel, str) or not isinstance(wanted, str) or not re.fullmatch(r"[0-9a-f]{64}", wanted):
        raise RegistryError("stable Registry router identity is malformed")
    path = _safe_repo_path(repo_root, rel)
    if not path.is_file():
        raise RegistryError(f"timestamped Registry missing: {rel}")
    observed = _sha256(path)
    if observed != wanted:
        raise RegistryError("timestamped Registry digest does not match stable router")
    return load_json(path), {"path": rel, "sha256": wanted}


def load_registry(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != SCHEMA_ID:
        raise RegistryError(f"Registry schema must be {SCHEMA_ID}")
    return value


def load_bound_passport(repo_root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    rel = entry.get("passport_path")
    if not isinstance(rel, str):
        raise RegistryError("Registry passport_path must be a string")
    path = _safe_repo_path(repo_root, rel)
    if not path.is_file():
        raise RegistryError(f"Passport missing: {rel}")
    try:
        passport = load_passport(path, root=repo_root)
    except PassportError as exc:
        raise RegistryError(f"Passport invalid for {entry.get('experiment_id')}: {exc}") from exc
    expected = entry.get("passport_sha256")
    observed = passport.get("passport_payload_sha256")
    if expected != observed:
        raise RegistryError(
            f"Passport digest mismatch for {entry.get('experiment_id')}: expected {expected}, observed {observed}"
        )
    return passport


def _validate_state_source(repo_root: Path, source: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    kind = source.get("type")
    path = source.get("path")
    release_path = source.get("release_path")
    if kind == "server_json":
        if not isinstance(path, str) or not Path(path).is_absolute():
            errors.append("server_json state source requires an absolute path")
        if release_path is not None:
            errors.append("server_json state source cannot also name a release_path")
    elif kind == "repo_json":
        if not isinstance(path, str):
            errors.append("repo_json state source requires a repository path")
        else:
            try:
                resolved = _safe_repo_path(repo_root, path)
                if not resolved.is_file():
                    errors.append(f"repo_json state source missing: {path}")
            except RegistryError as exc:
                errors.append(str(exc))
        if release_path is not None:
            errors.append("repo_json state source cannot also name a release_path")
    elif kind == "release_external_executor":
        if path is not None:
            errors.append("release_external_executor uses release_path, not path")
        if not isinstance(release_path, str):
            errors.append("release_external_executor requires release_path")
        else:
            try:
                resolved = _safe_repo_path(repo_root, release_path)
                if not resolved.is_file():
                    errors.append(f"release_external_executor release missing: {release_path}")
            except RegistryError as exc:
                errors.append(str(exc))
    elif kind == "static":
        if path is not None or release_path is not None:
            errors.append("static state source cannot name path/release_path")
    else:
        errors.append(f"unsupported Registry state source: {kind}")
    required_equal = source.get("required_equal")
    if not isinstance(required_equal, Mapping):
        errors.append("state_source.required_equal must be an object")
    return errors


def validate_registry(registry: Mapping[str, Any], repo_root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    try:
        validate_schema(registry, load_json(repo_root / "schemas/experiment-registry-202609112120.schema.json"))
    except PassportError as exc:
        return [f"Registry schema validation failed: {exc}"]
    entries = registry.get("entries")
    if not isinstance(entries, list):
        return ["Registry entries must be a list"]
    names: dict[str, str] = {}
    canonical_ids = {str(e.get("experiment_id")) for e in entries if isinstance(e, Mapping)}
    for entry in entries:
        if not isinstance(entry, Mapping):
            errors.append("Registry entry must be an object")
            continue
        experiment_id = str(entry.get("experiment_id") or "")
        if experiment_id in names:
            errors.append(f"duplicate experiment identity: {experiment_id}")
        names[experiment_id] = experiment_id
        for alias in entry.get("aliases") or []:
            owner = names.get(str(alias))
            if owner is not None and owner != experiment_id:
                errors.append(f"alias collision: {alias} belongs to both {owner} and {experiment_id}")
            if alias in canonical_ids and alias != experiment_id:
                errors.append(f"alias shadows canonical experiment_id: {alias}")
            names[str(alias)] = experiment_id
        try:
            passport = load_bound_passport(repo_root, entry)
        except RegistryError as exc:
            errors.append(str(exc))
            continue
        if passport.get("experiment_id") != experiment_id:
            errors.append(f"{experiment_id}: Passport experiment_id mismatch")
        if passport.get("parent_experiment_id") != entry.get("lineage_parent"):
            errors.append(f"{experiment_id}: lineage_parent disagrees with Passport")
        policy = str((passport.get("resume_policy") or {}).get("policy") or "")
        if entry.get("lifecycle_class") == "ACTIVE_FORMAL" and policy in {"NO_RESUME", "NO_RESUME_HISTORICAL"}:
            errors.append(f"{experiment_id}: NO_RESUME Passport cannot be ACTIVE_FORMAL")
        for rel in entry.get("latest_authority_paths") or []:
            try:
                path = _safe_repo_path(repo_root, str(rel))
                if not path.exists():
                    errors.append(f"{experiment_id}: authority path missing: {rel}")
            except RegistryError as exc:
                errors.append(f"{experiment_id}: {exc}")
        source = entry.get("state_source")
        if isinstance(source, Mapping):
            errors.extend(f"{experiment_id}: {error}" for error in _validate_state_source(repo_root, source))
        else:
            errors.append(f"{experiment_id}: state_source must be an object")
    return errors


def resolve_experiment(registry: Mapping[str, Any], experiment_or_alias: str) -> dict[str, Any]:
    matches: list[Mapping[str, Any]] = []
    for entry in registry.get("entries") or []:
        if not isinstance(entry, Mapping):
            continue
        if experiment_or_alias == entry.get("experiment_id") or experiment_or_alias in (entry.get("aliases") or []):
            matches.append(entry)
    if len(matches) != 1:
        if not matches:
            raise RegistryError(f"unknown experiment identity or alias: {experiment_or_alias}")
        raise RegistryError(f"ambiguous experiment alias: {experiment_or_alias}")
    return dict(matches[0])


def _path_get(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def observe_registered_state(entry: Mapping[str, Any]) -> dict[str, Any]:
    source = entry["state_source"]
    kind = source["type"]
    if kind != "server_json":
        return {
            "lifecycle_status": "LIVE_NOT_OBSERVED",
            "activity_status": "NOT_OBSERVED",
            "identity_trust": "NOT_OBSERVED",
            "semantic_progress": {},
        }
    path = Path(str(source["path"]))
    if not path.exists():
        return {
            "lifecycle_status": "LIVE_NOT_OBSERVED",
            "activity_status": "NOT_OBSERVED",
            "identity_trust": "NOT_OBSERVED",
            "semantic_progress": {},
        }
    try:
        payload = load_json(path, max_bytes=8 * 1024 * 1024)
    except PassportError:
        return {
            "lifecycle_status": "IDENTITY_MISMATCH",
            "activity_status": "UNKNOWN",
            "identity_trust": "FAIL",
            "semantic_progress": {},
        }
    mismatches = []
    for key, expected in (source.get("required_equal") or {}).items():
        observed = _path_get(payload, str(key))
        if observed != expected:
            mismatches.append({"field": key, "expected": expected, "observed": observed})
    if mismatches:
        return {
            "lifecycle_status": "IDENTITY_MISMATCH",
            "activity_status": "UNKNOWN",
            "identity_trust": "FAIL",
            "semantic_progress": {},
            "identity_mismatches": mismatches,
        }
    progress_keys = (
        "next_round", "rollouts_consumed", "sd_generation_count", "sd_component_count",
        "external_teacher_calls", "final_panel_access_count", "blocker",
    )
    progress = {key: payload[key] for key in progress_keys if key in payload}
    explicit_activity = payload.get("activity_status")
    activity = explicit_activity if explicit_activity in {
        "ACTIVE_GPU", "ACTIVE_CPU_OR_IO", "IDLE_AT_BOUNDARY", "NOT_OBSERVED", "UNKNOWN"
    } else "NOT_OBSERVED"
    return {
        "lifecycle_status": str(payload.get("status") or "UNKNOWN"),
        "activity_status": activity,
        "identity_trust": "PASS",
        "semantic_progress": progress,
    }


def build_passport_projection(repo_root: Path, experiment_or_alias: str) -> dict[str, Any]:
    registry, router = load_current_registry(repo_root)
    errors = validate_registry(registry, repo_root)
    if errors:
        raise RegistryError("Registry validation failed: " + "; ".join(errors))
    entry = resolve_experiment(registry, experiment_or_alias)
    passport = load_bound_passport(repo_root, entry)
    science = passport["scientific_identity"]
    model = science["model"]
    current = observe_registered_state(entry)
    execution = passport["initial_execution_identity"]
    return {
        "schema": "openevo.experiment-passport-view.202609112120",
        "mutation": False,
        "experiment_id": entry["experiment_id"],
        "aliases": list(entry["aliases"]),
        "display_name": entry["display_name"],
        "parent_experiment_id": passport["parent_experiment_id"],
        "passport_path": entry["passport_path"],
        "passport_sha256": entry["passport_sha256"],
        "registry": {"path": router["path"], "sha256": router["sha256"]},
        "research_question": science["research_question"],
        "hypothesis": science["hypothesis"],
        "treatment": science["treatment"],
        "what_changed": list(science["what_changed"]),
        "what_must_stay_fixed": list(science["what_must_stay_fixed"]),
        "model": {
            "id": model["model_id"],
            "revision": model["model_revision"],
            "tokenizer_revision": model["tokenizer_revision"],
            "dtype": model["dtype"],
        },
        "benchmark": {
            "name": science["benchmark"],
            "split": science["split"],
            "task_manifest": science["task_manifest"],
        },
        "seed_policy": science["seed_policy"],
        "budget_contract": passport["budget_contract"],
        "protocol_fingerprint": science["protocol_fingerprint"],
        "initial_execution_identity": {
            "initial_experiment_sha": execution["initial_experiment_sha"],
            "frozen_science_sha": execution["frozen_science_sha"],
            "upstream_openevo_sha": execution["upstream_openevo_sha"],
            "initial_ray_sha": execution["initial_ray_sha"],
            "runtime_image_digest": execution["runtime_image_digest"],
            "runtime_fingerprint": execution["runtime_fingerprint"],
            "initial_run_root": execution["initial_run_root"],
        },
        "current": current,
        "tracking": passport["tracking_policy"],
        "artifacts": passport["artifact_policy"],
        "evidence_paths": list(entry["latest_authority_paths"]),
        "resume_policy": passport["resume_policy"],
    }


def _short_hash(value: Any) -> str:
    text = str(value or "unknown")
    return text if len(text) <= 16 else text[:12] + "…"


def render_passport_projection(view: Mapping[str, Any]) -> str:
    model = view["model"]
    budget = view["budget_contract"]
    current = view["current"]
    execution = view["initial_execution_identity"]
    tracking = view["tracking"]
    lines = [
        f"Experiment: {view['display_name']}",
        f"Canonical ID: {view['experiment_id']}",
        f"Passport: {_short_hash(view['passport_sha256'])} (FROZEN identity)",
        f"Parent: {view['parent_experiment_id'] or 'not backfilled / none recorded'}",
        "",
        f"Purpose: {view['research_question']}",
        "What changed:",
    ]
    lines.extend(f"- {item}" for item in view["what_changed"])
    lines.append("What must stay fixed:")
    lines.extend(f"- {item}" for item in view["what_must_stay_fixed"])
    lines.extend([
        "",
        f"Model: {model['id']} @ {model['revision'] or 'unknown revision'}; dtype={model['dtype'] or 'unknown'}",
        f"Benchmark: {view['benchmark']['name']} / {view['benchmark']['split']}",
        f"Seed: {view['seed_policy']['schedule_id']} (base={view['seed_policy']['seed_base']})",
        f"Budget: {budget['round_limit']} rounds × {budget['rollouts_per_round']} = {budget['rollout_limit']} rollouts; horizon={budget['horizon']}",
        f"Protocol fingerprint: {_short_hash(view['protocol_fingerprint'])}",
        f"Initial code/runtime: experiment={_short_hash(execution['initial_experiment_sha'])}, science={_short_hash(execution['frozen_science_sha'])}, image={_short_hash(execution['runtime_image_digest'])}",
        "",
        f"Lifecycle now: {current['lifecycle_status']}",
        f"Active compute now: {current['activity_status']}",
        f"Semantic progress: {current['semantic_progress'] or 'NOT_OBSERVED'}",
        f"Live identity trust: {current['identity_trust']}",
        "",
        f"W&B: {tracking.get('entity')}/{tracking.get('project')} ({tracking.get('role')}; not scientific authority)",
        f"Large-artifact provider: {view['artifacts'].get('large_asset_provider')}; W&B sole backup={view['artifacts'].get('wandb_is_sole_backup')}",
        f"Resume policy: {view['resume_policy'].get('policy')} — {view['resume_policy'].get('reason')}",
        "Evidence entrypoints:",
    ])
    lines.extend(f"- {path}" for path in view["evidence_paths"])
    lines.append("Read-only: this command never authorizes or starts continuation.")
    return "\n".join(lines)


def console_static_overlay(repo_root: Path, legacy_id: str) -> dict[str, Any] | None:
    """Return Registry identity when installed; old console fixtures remain readable."""
    router = repo_root / "configs/experiment/current-experiment-registry.json"
    if not router.exists():
        return None
    registry, _ = load_current_registry(repo_root)
    errors = validate_registry(registry, repo_root)
    if errors:
        raise RegistryError("Registry validation failed: " + "; ".join(errors))
    try:
        entry = resolve_experiment(registry, legacy_id)
    except RegistryError as exc:
        if str(exc).startswith("unknown experiment identity or alias"):
            return None
        raise
    passport = load_bound_passport(repo_root, entry)
    science = passport["scientific_identity"]
    return {
        "canonical_experiment_id": entry["experiment_id"],
        "passport_sha256": entry["passport_sha256"],
        "display_name": entry["display_name"],
        "model": science["model"]["model_id"],
        "treatment": science["treatment"],
        "what_changed": list(science["what_changed"]),
        "what_stays_fixed": list(science["what_must_stay_fixed"]),
        "state_source": dict(entry["state_source"]),
        "resume": {
            "policy": passport["resume_policy"]["policy"],
            "next_action": passport["resume_policy"]["reason"],
        },
        "authority_paths": list(entry["latest_authority_paths"]),
    }
