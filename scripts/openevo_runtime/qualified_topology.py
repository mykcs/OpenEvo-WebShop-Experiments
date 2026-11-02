"""Fail-closed qualified-topology identity registry for the Execution Runtime broker.

This layer preserves already-qualified resource topology identities.  It does not
qualify a new topology, grant launch/scientific authority, or infer fallback from
available GPUs.  Requests and backend selections must stay inside one exact entry.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop.experiment_passport import assert_no_credentials, canonical_bytes

REV = "202609170845"
REGISTRY_SCHEMA = f"openevo.qualified-topology-registry.{REV}"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class QualifiedTopologyError(ValueError):
    """A request/selection escaped the frozen qualified topology registry."""


def _digest(value: Mapping[str, Any]) -> str:
    assert_no_credentials(value)
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise QualifiedTopologyError(f"{label} must be a SHA-256 identity")
    return value


def topology_projection(entry: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "name", "workloads", "placement_mode", "gpu_count", "candidate_gpu_indices",
        "allowed_selected_gpu_sets", "fallback_allowed", "authority_refs",
        "scientific_authority_granted", "formal_activation_granted",
    )
    missing = [key for key in required if key not in entry]
    if missing:
        raise QualifiedTopologyError(f"missing topology field: {missing[0]}")
    name = entry["name"]
    if not isinstance(name, str) or not name:
        raise QualifiedTopologyError("topology name must be non-empty")
    workloads = entry["workloads"]
    if not isinstance(workloads, list) or not workloads or any(not isinstance(v, str) or not v for v in workloads):
        raise QualifiedTopologyError("workloads must be a non-empty string list")
    if len(workloads) != len(set(workloads)):
        raise QualifiedTopologyError("workloads must be unique")
    if entry["placement_mode"] not in {"GANG_ALL_OR_NOTHING", "QUALIFIED_SUBSET"}:
        raise QualifiedTopologyError("unsupported placement_mode")
    count = entry["gpu_count"]
    if type(count) is not int or count < 1:
        raise QualifiedTopologyError("gpu_count must be >=1")
    candidates = entry["candidate_gpu_indices"]
    if not isinstance(candidates, list) or any(type(v) is not int or v < 0 for v in candidates):
        raise QualifiedTopologyError("candidate_gpu_indices must be non-negative integers")
    if len(candidates) != len(set(candidates)) or len(candidates) < count:
        raise QualifiedTopologyError("candidate_gpu_indices cannot satisfy exact topology")
    allowed = entry["allowed_selected_gpu_sets"]
    if not isinstance(allowed, list) or not allowed:
        raise QualifiedTopologyError("allowed_selected_gpu_sets must be non-empty")
    normalized_allowed: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    candidate_set = set(candidates)
    for row in allowed:
        if not isinstance(row, list) or any(type(v) is not int or v < 0 for v in row):
            raise QualifiedTopologyError("allowed selected GPU set must contain non-negative integers")
        if len(row) != count or len(row) != len(set(row)) or not set(row).issubset(candidate_set):
            raise QualifiedTopologyError("allowed selected GPU set violates exact topology")
        normalized = tuple(sorted(row))
        if normalized in seen:
            raise QualifiedTopologyError("duplicate allowed selected GPU set")
        seen.add(normalized)
        normalized_allowed.append(list(normalized))
    if entry["fallback_allowed"] is not False:
        raise QualifiedTopologyError("this registry revision does not authorize topology fallback")
    refs = entry["authority_refs"]
    if not isinstance(refs, list) or not refs:
        raise QualifiedTopologyError("qualified topology needs authority_refs")
    normalized_refs: list[dict[str, str]] = []
    for ref in refs:
        if not isinstance(ref, Mapping):
            raise QualifiedTopologyError("authority ref must be an object")
        git_sha = ref.get("git_sha")
        path = ref.get("path")
        sha256 = ref.get("sha256")
        status = ref.get("status")
        if not isinstance(git_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", git_sha):
            raise QualifiedTopologyError("authority ref git_sha must be exact 40-hex commit")
        if not isinstance(path, str) or not path or path.startswith("/"):
            raise QualifiedTopologyError("authority ref path must be repository-relative")
        _sha(sha256, "authority ref sha256")
        if not isinstance(status, str) or not status:
            raise QualifiedTopologyError("authority ref status must be non-empty")
        normalized_refs.append({"git_sha": git_sha, "path": path, "sha256": sha256, "status": status})
    if entry["scientific_authority_granted"] is not False or entry["formal_activation_granted"] is not False:
        raise QualifiedTopologyError("topology registry cannot grant science/formal activation authority")
    projection = {
        "name": name,
        "workloads": list(workloads),
        "placement_mode": entry["placement_mode"],
        "gpu_count": count,
        "candidate_gpu_indices": list(candidates),
        "allowed_selected_gpu_sets": normalized_allowed,
        "fallback_allowed": False,
        "authority_refs": normalized_refs,
        "scientific_authority_granted": False,
        "formal_activation_granted": False,
    }
    canonical_bytes(projection)
    return projection


def topology_id(entry: Mapping[str, Any]) -> str:
    return _digest(topology_projection(entry))


def load_qualified_topology_registry(path: Path, *, base_config_sha256: str, profile_registry_sha256: str | None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualifiedTopologyError(f"unreadable qualified topology registry: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != REGISTRY_SCHEMA:
        raise QualifiedTopologyError("qualified topology registry schema drift")
    if value.get("status") != "PROSPECTIVE_IDENTITY_PRESERVATION_ONLY":
        raise QualifiedTopologyError("qualified topology registry status drift")
    if value.get("scientific_authority_granted") is not False or value.get("formal_activation_granted") is not False:
        raise QualifiedTopologyError("qualified topology registry cannot grant science/formal authority")
    if value.get("base_config_sha256") != _sha(base_config_sha256, "base_config_sha256"):
        raise QualifiedTopologyError("qualified topology registry base-config drift")
    expected_profile = profile_registry_sha256
    if value.get("profile_registry_sha256") != expected_profile:
        raise QualifiedTopologyError("qualified topology registry profile-registry drift")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise QualifiedTopologyError("qualified topology registry entries missing")
    ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise QualifiedTopologyError("qualified topology entry must be object")
        actual = topology_id(entry)
        if entry.get("topology_id") != actual:
            raise QualifiedTopologyError("qualified topology_id does not match canonical projection")
        if actual in ids:
            raise QualifiedTopologyError("duplicate qualified topology identity")
        ids.add(actual)
    projection = {
        "schema": REGISTRY_SCHEMA,
        "status": value["status"],
        "base_config_sha256": value["base_config_sha256"],
        "profile_registry_sha256": value["profile_registry_sha256"],
        "entries": [dict(entry) for entry in entries],
        "scientific_authority_granted": False,
        "formal_activation_granted": False,
    }
    if value.get("registry_id") != _digest(projection):
        raise QualifiedTopologyError("qualified topology registry_id drift")
    return value


def resolve_qualified_topology(registry: Mapping[str, Any], *, topology_identity: str) -> dict[str, Any]:
    _sha(topology_identity, "qualified_topology_identity")
    for entry in registry.get("entries") or []:
        if entry.get("topology_id") == topology_identity:
            return dict(entry)
    raise QualifiedTopologyError("request references unregistered qualified topology identity")


def validate_request_topology(registry: Mapping[str, Any], request: Mapping[str, Any], *, workload: str) -> dict[str, Any]:
    entry = resolve_qualified_topology(registry, topology_identity=request.get("qualified_topology_identity"))
    if workload not in entry["workloads"]:
        raise QualifiedTopologyError("qualified topology is not authorized for workload")
    expected_count = int(request["envelope"]["gpu_scope"]["gpu_count"])
    if expected_count != int(entry["gpu_count"]):
        raise QualifiedTopologyError("request GPU count changes qualified topology")
    if request.get("placement_mode") != entry["placement_mode"]:
        raise QualifiedTopologyError("request placement_mode changes qualified topology")
    if set(request.get("preferred_gpu_indices") or []) != set(entry["candidate_gpu_indices"]):
        raise QualifiedTopologyError("request preferred GPU set changes qualified topology")
    if request.get("fallback_topology_identity") is not None:
        raise QualifiedTopologyError("topology fallback is not authorized by this qualified registry")
    return entry


def validate_selection_topology(entry: Mapping[str, Any], selected_gpu_indices: list[int]) -> None:
    selected = tuple(sorted(selected_gpu_indices))
    allowed = {tuple(row) for row in entry["allowed_selected_gpu_sets"]}
    if selected not in allowed:
        raise QualifiedTopologyError("Ray selection changed the prospectively qualified topology")
