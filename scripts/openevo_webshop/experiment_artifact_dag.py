#!/usr/bin/env python3
"""Content-addressed operation/artifact DAG for Experiment Passport.

This is a conservative Passport/Registry extension, not a second authority.
A matching PASS can yield REUSE_CANDIDATE; it never grants launch, replay,
model-update, or final-evaluation authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    assert_no_credentials,
    canonical_bytes,
    load_json,
    repository_path,
    validate_schema,
)
from scripts.openevo_webshop.experiment_registry import (
    RegistryError,
    load_bound_passport,
    load_current_registry,
    resolve_experiment,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[2]
NODE_SCHEMA_PATH = ROOT / "schemas/experiment-passport-operation-node-202609202238.schema.json"
RUNTIME_NODE_SCHEMA_PATH = ROOT / "schemas/experiment-passport-operation-node-202609210003.schema.json"
LEGACY_RUNTIME_NODE_SCHEMA_PATH = ROOT / "schemas/experiment-passport-operation-node-202609170251.schema.json"
LEGACY_NODE_SCHEMA_PATH = ROOT / "schemas/experiment-passport-operation-node-202609142055.schema.json"
EDGE_SCHEMA_PATH = ROOT / "schemas/experiment-passport-artifact-edge-202609202238.schema.json"
INDEX_SCHEMA_PATH = ROOT / "schemas/experiment-passport-artifact-index-202609202238.schema.json"
NODE_SCHEMA_ID = "openevo.experiment-passport-operation-node.202609202238"
RUNTIME_NODE_SCHEMA_ID = "openevo.experiment-passport-operation-node.202609210003"
LEGACY_RUNTIME_NODE_SCHEMA_ID = "openevo.experiment-passport-operation-node.202609170251"
LEGACY_NODE_SCHEMA_ID = "openevo.experiment-passport-operation-node.202609142055"
NODE_SCHEMA_PATHS = {
    NODE_SCHEMA_ID: NODE_SCHEMA_PATH,
    RUNTIME_NODE_SCHEMA_ID: RUNTIME_NODE_SCHEMA_PATH,
    LEGACY_RUNTIME_NODE_SCHEMA_ID: LEGACY_RUNTIME_NODE_SCHEMA_PATH,
    LEGACY_NODE_SCHEMA_ID: LEGACY_NODE_SCHEMA_PATH,
}
GENERATION_BOUND_NODE_SCHEMAS = {
    RUNTIME_NODE_SCHEMA_ID,
    LEGACY_RUNTIME_NODE_SCHEMA_ID,
}
EDGE_SCHEMA_ID = "openevo.experiment-passport-artifact-edge.202609202238"
INDEX_SCHEMA_ID = "openevo.experiment-passport-artifact-index.202609202238"
INDEX_ROLE = "lookup-acceleration-only; Passport/receipts remain execution and provenance authority"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

SIDE_EFFECT_CLASSES = {
    "PURE_ENGINEERING", "DETERMINISTIC_COMPUTATION",
    "STOCHASTIC_SCIENTIFIC_RESULT", "FORMAL_ROLLOUT",
    "MODEL_UPDATE", "FINAL_EVALUATION",
}
SCIENTIFIC_NO_AUTO_REUSE = {
    "STOCHASTIC_SCIENTIFIC_RESULT", "FORMAL_ROLLOUT",
    "MODEL_UPDATE", "FINAL_EVALUATION",
}
DEFAULT_REUSE_POLICIES = {
    "PURE_ENGINEERING": "ALLOWLISTED_ENGINEERING_ONLY",
    "DETERMINISTIC_COMPUTATION": "REVIEW_REQUIRED_DETERMINISM",
    "STOCHASTIC_SCIENTIFIC_RESULT": "EXPERIMENT_POLICY_REQUIRED",
    "FORMAL_ROLLOUT": "FORMAL_REPLAY_POLICY_REQUIRED",
    "MODEL_UPDATE": "MODEL_UPDATE_REUSE_FORBIDDEN_BY_DEFAULT",
    "FINAL_EVALUATION": "FINAL_EVAL_POLICY_REQUIRED",
}


class ArtifactDagError(ValueError):
    """Invalid content identity, DAG record, provenance, or reuse query."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _identity_ref(ref: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {key: ref.get(key) for key in ("kind", "repository", "revision", "path", "sha256")}


def artifact_identity(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Content/logical identity only; locations and lineage are provenance metadata."""
    return {key: artifact.get(key) for key in (
        "artifact_type", "digest_algorithm", "digest", "size_bytes", "logical_role"
    )}


def operation_identity_payload(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact fields included in the canonical operation digest."""
    if not isinstance(spec, Mapping):
        raise ArtifactDagError("operation spec must be an object")
    identity = spec.get("identity_payload") if "identity_payload" in spec else spec
    if not isinstance(identity, Mapping):
        raise ArtifactDagError("identity_payload must be an object")
    schema_id = spec.get("schema")
    execution_generation_id = identity.get("execution_generation_id")
    if schema_id in GENERATION_BOUND_NODE_SCHEMAS:
        if not isinstance(execution_generation_id, str) or not HEX64.fullmatch(execution_generation_id):
            raise ArtifactDagError("execution_generation_id must bind one content-addressed ExecutionGeneration")
    elif schema_id in {NODE_SCHEMA_ID, LEGACY_NODE_SCHEMA_ID}:
        if execution_generation_id is not None:
            raise ArtifactDagError(
                f"{schema_id} cannot claim execution_generation_id; use {RUNTIME_NODE_SCHEMA_ID}"
            )
    elif execution_generation_id is not None:
        if not isinstance(execution_generation_id, str) or not HEX64.fullmatch(execution_generation_id):
            raise ArtifactDagError("execution_generation_id must be a SHA-256 identity")
    operation_type = spec.get("operation_type")
    version = spec.get("operation_schema_version")
    side_effect = spec.get("side_effect_class")
    if not isinstance(operation_type, str) or not operation_type:
        raise ArtifactDagError("operation_type is required")
    if not isinstance(version, str) or not version:
        raise ArtifactDagError("operation_schema_version is required")
    if side_effect not in SIDE_EFFECT_CLASSES:
        raise ArtifactDagError("side_effect_class is required and must use the controlled vocabulary")
    runtime_policy = identity.get("runtime_identity_policy")
    runtime = identity.get("runtime_identity")
    if runtime_policy not in {"BOUND", "NOT_IDENTITY_BEARING"}:
        raise ArtifactDagError("runtime_identity_policy must be BOUND or NOT_IDENTITY_BEARING")
    if runtime_policy == "BOUND" and not isinstance(runtime, Mapping):
        raise ArtifactDagError("BOUND runtime identity requires runtime_identity")
    if runtime_policy == "NOT_IDENTITY_BEARING" and runtime is not None:
        raise ArtifactDagError("NOT_IDENTITY_BEARING runtime must set runtime_identity=null")
    contracts = identity.get("contract_identities")
    inputs = identity.get("input_artifacts")
    if not isinstance(contracts, list) or not isinstance(inputs, list):
        raise ArtifactDagError("contract_identities and input_artifacts must be arrays")
    names = set()
    for index, contract in enumerate(contracts):
        if not isinstance(contract, Mapping):
            raise ArtifactDagError(f"contract_identities[{index}] must be an object")
        name = contract.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ArtifactDagError("contract identity names must be unique non-empty strings")
        names.add(name)
        _validate_ref(contract.get("identity"), label=f"contract_identities[{index}]", required=True)
    roles = set()
    for index, artifact in enumerate(inputs):
        if not isinstance(artifact, Mapping):
            raise ArtifactDagError(f"input_artifacts[{index}] must be an object")
        _validate_artifact(artifact, label=f"input_artifacts[{index}]")
        role = artifact.get("logical_role")
        if not isinstance(role, str) or not role or role in roles:
            raise ArtifactDagError("input artifact logical_role values must be unique")
        roles.add(role)
    contract_view = [
        {"name": item.get("name"), "identity": _identity_ref(item.get("identity"))}
        for item in contracts
    ]
    contract_view.sort(key=lambda item: str(item.get("name")))
    input_view = [artifact_identity(item) for item in inputs]
    input_view.sort(key=lambda item: (
        str(item.get("logical_role")), str(item.get("artifact_type")), str(item.get("digest"))
    ))
    _validate_ref(identity.get("code_identity"), label="code_identity", required=True)
    _validate_ref(identity.get("config_identity"), label="config_identity", required=True)
    _validate_ref(runtime, label="runtime_identity", required=runtime_policy == "BOUND")
    _validate_ref(identity.get("dataset_schedule_identity"), label="dataset_schedule_identity", required=False)
    stochastic = copy.deepcopy(identity.get("stochastic_identity") or {})
    if side_effect in SCIENTIFIC_NO_AUTO_REUSE and not contracts:
        raise ArtifactDagError(f"{side_effect} requires at least one bound science/runtime contract identity")
    if side_effect in {"STOCHASTIC_SCIENTIFIC_RESULT", "FORMAL_ROLLOUT", "FINAL_EVALUATION"}:
        if identity.get("dataset_schedule_identity") is None:
            raise ArtifactDagError(f"{side_effect} requires dataset_schedule_identity")
        if not stochastic:
            raise ArtifactDagError(f"{side_effect} requires explicit stochastic/determinism identity")
    payload = {
        "operation_type": operation_type,
        "operation_schema_version": version,
        "side_effect_class": side_effect,
        "code_identity": _identity_ref(identity.get("code_identity")),
        "config_identity": _identity_ref(identity.get("config_identity")),
        "runtime_identity_policy": runtime_policy,
        "runtime_identity": _identity_ref(runtime),
        "dataset_schedule_identity": _identity_ref(identity.get("dataset_schedule_identity")),
        "contract_identities": contract_view,
        "input_artifacts": input_view,
        "stochastic_identity": stochastic,
    }
    if execution_generation_id is not None:
        payload["execution_generation_id"] = execution_generation_id
    assert_no_credentials(payload)
    canonical_bytes(payload)
    return payload


def operation_digest(spec: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_bytes(operation_identity_payload(spec)))


def _validate_ref(ref: Mapping[str, Any] | None, *, label: str, required: bool) -> None:
    if ref is None:
        if required:
            raise ArtifactDagError(f"{label} is required")
        return
    if not isinstance(ref, Mapping):
        raise ArtifactDagError(f"{label} must be an object")
    sha = ref.get("sha256")
    if not isinstance(sha, str) or not HEX64.fullmatch(sha):
        raise ArtifactDagError(f"{label}.sha256 must bind exact content")
    if ref.get("kind") == "git_file":
        revision = ref.get("revision")
        if not isinstance(revision, str) or not HEX40.fullmatch(revision):
            raise ArtifactDagError(f"{label}.revision must be an immutable git SHA")


def _validate_artifact(artifact: Mapping[str, Any], *, label: str) -> None:
    if artifact.get("digest_algorithm") != "sha256":
        raise ArtifactDagError(f"{label}.digest_algorithm must be sha256")
    if not isinstance(artifact.get("digest"), str) or not HEX64.fullmatch(str(artifact.get("digest"))):
        raise ArtifactDagError(f"{label}.digest must be SHA-256")
    if type(artifact.get("size_bytes")) is not int or artifact["size_bytes"] < 0:
        raise ArtifactDagError(f"{label}.size_bytes must be a non-negative integer")


def _validate_reuse_policy(node: Mapping[str, Any]) -> None:
    side_effect = node.get("side_effect_class")
    if side_effect not in SIDE_EFFECT_CLASSES:
        raise ArtifactDagError("unknown side_effect_class")
    reuse = node.get("reuse")
    if not isinstance(reuse, Mapping):
        raise ArtifactDagError("reuse must be an object")
    wanted = DEFAULT_REUSE_POLICIES[str(side_effect)]
    if reuse.get("default_policy") != wanted:
        raise ArtifactDagError(f"reuse.default_policy must be {wanted} for {side_effect}")
    auto = reuse.get("automatic_reuse_permitted")
    if type(auto) is not bool:
        raise ArtifactDagError("reuse.automatic_reuse_permitted must be boolean")
    if side_effect in SCIENTIFIC_NO_AUTO_REUSE and auto:
        raise ArtifactDagError(f"{side_effect} can never be automatically reused")
    if side_effect == "DETERMINISTIC_COMPUTATION" and auto:
        raise ArtifactDagError("deterministic computation has no blanket automatic reuse")
    if side_effect == "PURE_ENGINEERING" and auto and reuse.get("operation_allowlisted") is not True:
        raise ArtifactDagError("PURE_ENGINEERING automatic reuse requires operation_allowlisted=true")


def validate_operation_node(node: Mapping[str, Any], *, repo_root: Path = ROOT) -> None:
    canonical_bytes(node)
    assert_no_credentials(node)
    schema_id = str(node.get("schema"))
    schema_path = NODE_SCHEMA_PATHS.get(schema_id)
    if schema_path is None:
        raise ArtifactDagError(f"unsupported operation-node schema: {schema_id}")
    try:
        validate_schema(node, load_json(schema_path))
    except PassportError as exc:
        raise ArtifactDagError(str(exc)) from exc
    if node.get("operation_digest") != operation_digest(node):
        raise ArtifactDagError("operation_digest mismatch")
    identity = node["identity_payload"]
    _validate_ref(identity.get("code_identity"), label="code_identity", required=True)
    _validate_ref(identity.get("config_identity"), label="config_identity", required=True)
    _validate_ref(identity.get("runtime_identity"), label="runtime_identity",
                  required=identity.get("runtime_identity_policy") == "BOUND")
    _validate_ref(identity.get("dataset_schedule_identity"), label="dataset_schedule_identity", required=False)
    names: set[str] = set()
    for index, contract in enumerate(identity.get("contract_identities") or []):
        name = contract.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ArtifactDagError("contract identity names must be unique non-empty strings")
        names.add(name)
        _validate_ref(contract.get("identity"), label=f"contract_identities[{index}]", required=True)
    roles: set[str] = set()
    for index, artifact in enumerate(identity.get("input_artifacts") or []):
        _validate_artifact(artifact, label=f"input_artifacts[{index}]")
        role = artifact.get("logical_role")
        if not isinstance(role, str) or not role or role in roles:
            raise ArtifactDagError("input artifact logical_role values must be unique")
        roles.add(role)
    output_roles: set[str] = set()
    for index, artifact in enumerate(node.get("outputs") or []):
        _validate_artifact(artifact, label=f"outputs[{index}]")
        role = artifact.get("logical_role")
        if not isinstance(role, str) or not role or role in output_roles:
            raise ArtifactDagError("output artifact logical_role values must be unique")
        output_roles.add(role)
    _validate_reuse_policy(node)
    provenance = node["provenance"]
    if provenance["producer_scope"] == "EXPERIMENT":
        if not provenance.get("experiment_id"):
            raise ArtifactDagError("EXPERIMENT producer scope requires experiment_id")
        if not isinstance(provenance.get("passport_sha256"), str) or not HEX64.fullmatch(provenance["passport_sha256"]):
            raise ArtifactDagError("EXPERIMENT producer scope requires passport_sha256")
    elif provenance.get("experiment_id") is not None or provenance.get("passport_sha256") is not None:
        raise ArtifactDagError("ENGINEERING_GLOBAL cannot claim experiment/passport binding")
    execution_sha = node["producer_attempt"].get("execution_sha")
    if execution_sha is not None and (not isinstance(execution_sha, str) or not HEX40.fullmatch(execution_sha)):
        raise ArtifactDagError("producer_attempt.execution_sha must be a git SHA or null")
    side_effect = node["side_effect_class"]
    if side_effect in SCIENTIFIC_NO_AUTO_REUSE:
        if provenance["producer_scope"] != "EXPERIMENT":
            raise ArtifactDagError(f"{side_effect} requires EXPERIMENT producer provenance")
        if execution_sha is None:
            raise ArtifactDagError(f"{side_effect} requires producer_attempt.execution_sha")
        if node["producer_attempt"].get("source") == "PILOT":
            raise ArtifactDagError(f"{side_effect} cannot use PILOT producer provenance")
    if node.get("status") == "PASS" and not node.get("outputs"):
        raise ArtifactDagError("PASS node requires output artifacts")


def _verify_ref(repo_root: Path, ref: Mapping[str, Any]) -> tuple[str, str]:
    kind, rel, wanted = ref.get("kind"), ref.get("path"), ref.get("sha256")
    if kind == "repository_file" and isinstance(rel, str):
        try:
            path = repository_path(repo_root, rel)
        except PassportError as exc:
            return "FAIL", str(exc)
        if not path.is_file():
            return "MISSING", f"repository file missing: {rel}"
        return (("PASS", f"repository file verified: {rel}")
                if _sha256_path(path) == wanted
                else ("FAIL", f"repository file hash mismatch: {rel}"))
    if kind == "git_file" and isinstance(rel, str):
        revision = ref.get("revision")
        if not isinstance(revision, str) or not HEX40.fullmatch(revision):
            return "FAIL", f"git_file lacks immutable revision: {rel}"
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{revision}:{rel}"],
            capture_output=True, check=False,
        )
        if result.returncode != 0:
            return "MISSING", f"git object unavailable: {revision}:{rel}"
        return (("PASS", f"git file verified: {revision}:{rel}")
                if _sha256_bytes(result.stdout) == wanted
                else ("FAIL", f"git file hash mismatch: {revision}:{rel}"))
    return "UNVERIFIED", f"{kind} reference requires an external verifier"


def _verify_artifact(repo_root: Path, artifact: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Require one exact locally verifiable location for MVP reuse admission."""
    expected = artifact["digest"]
    details: list[str] = []
    for ref in artifact.get("locations") or []:
        if ref.get("kind") not in {"repository_file", "git_file"}:
            details.append(f"UNVERIFIED: {ref.get('kind')} location is a view/pointer only")
            continue
        if ref.get("sha256") != expected:
            details.append("FAIL: location sha256 disagrees with artifact content identity")
            continue
        status, reason = _verify_ref(repo_root, ref)
        details.append(f"{status}: {reason}")
        if status == "PASS":
            return "PASS", details
    return "FAIL", details


def _identity_refs_from_payload(identity: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    refs = [("code_identity", identity["code_identity"]),
            ("config_identity", identity["config_identity"])]
    if identity.get("runtime_identity") is not None:
        refs.append(("runtime_identity", identity["runtime_identity"]))
    if identity.get("dataset_schedule_identity") is not None:
        refs.append(("dataset_schedule_identity", identity["dataset_schedule_identity"]))
    refs.extend((f"contract:{item['name']}", item["identity"])
                for item in identity.get("contract_identities") or [])
    return refs


def _identity_refs(node: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    return _identity_refs_from_payload(node["identity_payload"])


def verify_operation_spec_provenance(spec: Mapping[str, Any], *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Verify the *current request* really matches its declared exact identities."""
    try:
        operation_identity_payload(spec)
    except (ArtifactDagError, PassportError) as exc:
        return {"status": "FAIL", "details": [str(exc)]}
    identity = spec.get("identity_payload") if "identity_payload" in spec else spec
    if not isinstance(identity, Mapping):
        return {"status": "FAIL", "details": ["identity_payload must be an object"]}
    ok = True
    details: list[str] = []
    for label, ref in _identity_refs_from_payload(identity):
        status, reason = _verify_ref(repo_root, ref)
        details.append(f"{label}: {status}: {reason}")
        if status != "PASS":
            ok = False
    for artifact in identity.get("input_artifacts") or []:
        status, artifact_details = _verify_artifact(repo_root, artifact)
        details.extend(f"input:{artifact['logical_role']}: {item}" for item in artifact_details)
        if status != "PASS":
            ok = False
    return {"status": "PASS" if ok else "FAIL", "details": details}


def verify_node_provenance(node: Mapping[str, Any], *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Verify identity inputs separately from output liveness/content.

    Identity failures classify as PROVENANCE_MISMATCH; output failures classify as
    STALE_OR_INVALID_OUTPUT. External-only views fail closed in this MVP.
    """
    try:
        validate_operation_node(node, repo_root=repo_root)
    except ArtifactDagError as exc:
        return {"identity_status": "FAIL", "output_status": "UNKNOWN", "identity_details": [str(exc)], "output_details": []}
    identity_ok = True
    identity_details: list[str] = []
    for label, ref in _identity_refs(node):
        status, reason = _verify_ref(repo_root, ref)
        identity_details.append(f"{label}: {status}: {reason}")
        if status != "PASS":
            identity_ok = False
    for artifact in node["identity_payload"].get("input_artifacts") or []:
        status, details = _verify_artifact(repo_root, artifact)
        identity_details.extend(f"input:{artifact['logical_role']}: {detail}" for detail in details)
        if status != "PASS":
            identity_ok = False
    for rel in node["provenance"].get("evidence_paths") or []:
        try:
            exists = repository_path(repo_root, rel).is_file()
        except PassportError:
            exists = False
        identity_details.append(f"evidence_path:{rel}: {'PASS' if exists else 'FAIL'}")
        if not exists:
            identity_ok = False
    execution_sha = node["producer_attempt"].get("execution_sha")
    if execution_sha is not None:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{execution_sha}^{{commit}}"],
            capture_output=True, check=False,
        )
        identity_details.append(f"producer_attempt.execution_sha: {'PASS' if result.returncode == 0 else 'FAIL'}")
        if result.returncode != 0:
            identity_ok = False
    provenance = node["provenance"]
    if provenance.get("producer_scope") == "EXPERIMENT":
        try:
            registry, _ = load_current_registry(repo_root)
            registry_errors = validate_registry(registry, repo_root)
            if registry_errors:
                raise RegistryError("; ".join(registry_errors))
            entry = resolve_experiment(registry, str(provenance.get("experiment_id")))
            if entry.get("passport_sha256") != provenance.get("passport_sha256"):
                raise RegistryError("producer Passport digest disagrees with current Registry binding")
            load_bound_passport(repo_root, entry)
            identity_details.append("producer_passport_registry_binding: PASS")
        except (RegistryError, PassportError, OSError) as exc:
            identity_details.append(f"producer_passport_registry_binding: FAIL: {exc}")
            identity_ok = False
    output_ok = True
    output_details: list[str] = []
    for artifact in node.get("outputs") or []:
        status, details = _verify_artifact(repo_root, artifact)
        output_details.extend(f"output:{artifact['logical_role']}: {detail}" for detail in details)
        if status != "PASS":
            output_ok = False
    return {
        "identity_status": "PASS" if identity_ok else "FAIL",
        "output_status": "PASS" if output_ok else "FAIL",
        "identity_details": identity_details,
        "output_details": output_details,
    }


def node_relative_path(digest: str, node_sha256: str) -> str:
    """Content-address an attempt record beneath its stable operation identity."""
    if not HEX64.fullmatch(digest):
        raise ArtifactDagError("operation digest must be SHA-256")
    if not HEX64.fullmatch(node_sha256):
        raise ArtifactDagError("operation-node record digest must be SHA-256")
    return f"configs/experiment/passport-artifacts/operation-nodes/{digest}/{node_sha256}.json"


def _pretty_bytes(value: Any) -> bytes:
    canonical_bytes(value)
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_operation_node(repo_root: Path, node: Mapping[str, Any]) -> dict[str, Any]:
    """Write one immutable producer-attempt record under a stable operation digest."""
    validate_operation_node(node, repo_root=repo_root)
    raw = _pretty_bytes(node)
    node_sha256 = _sha256_bytes(raw)
    rel = node_relative_path(str(node["operation_digest"]), node_sha256)
    path = repository_path(repo_root, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ArtifactDagError("content-addressed operation-node path contains different bytes")
    if not path.exists():
        path.write_bytes(raw)
    return {
        "operation_digest": node["operation_digest"], "path": rel,
        "node_sha256": node_sha256, "operation_type": node["operation_type"],
        "side_effect_class": node["side_effect_class"], "status": node["status"],
    }


def build_index(entries: list[Mapping[str, Any]], *, generated_at: str | None = None) -> dict[str, Any]:
    values = [dict(item) for item in entries]
    values.sort(key=lambda item: (str(item.get("operation_digest")), str(item.get("node_sha256"))))
    return {"schema": INDEX_SCHEMA_ID, "role": INDEX_ROLE,
            "generated_at": generated_at or _utc_now(), "entries": values}


def validate_index(index: Mapping[str, Any], *, repo_root: Path = ROOT, verify_nodes: bool = True) -> list[str]:
    try:
        canonical_bytes(index)
        assert_no_credentials(index)
        validate_schema(index, load_json(INDEX_SCHEMA_PATH))
    except PassportError as exc:
        return [str(exc)]
    errors: list[str] = []
    seen_records: set[tuple[str, str]] = set()
    for entry in index.get("entries") or []:
        digest = str(entry.get("operation_digest"))
        record_key = (str(entry.get("path")), str(entry.get("node_sha256")))
        if record_key in seen_records:
            errors.append(f"duplicate operation-node record in index: {record_key[0]}")
            continue
        seen_records.add(record_key)
        if not verify_nodes:
            continue
        expected_rel = node_relative_path(digest, str(entry.get("node_sha256")))
        if entry.get("path") != expected_rel:
            errors.append(f"non-canonical operation-node path for {digest}")
            continue
        try:
            path = repository_path(repo_root, str(entry.get("path")))
        except PassportError as exc:
            errors.append(str(exc)); continue
        if not path.is_file():
            errors.append(f"operation node missing: {entry.get('path')}"); continue
        if _sha256_path(path) != entry.get("node_sha256"):
            errors.append(f"operation node file hash mismatch: {entry.get('path')}"); continue
        try:
            node = load_json(path)
            validate_operation_node(node, repo_root=repo_root)
        except (PassportError, ArtifactDagError) as exc:
            errors.append(f"operation node invalid: {entry.get('path')}: {exc}"); continue
        for key in ("operation_digest", "operation_type", "side_effect_class", "status"):
            if node.get(key) != entry.get(key):
                errors.append(f"index/node mismatch for {digest}: {key}")
    return errors


def write_index_snapshot(repo_root: Path, index: Mapping[str, Any], *, snapshot_rel: str) -> dict[str, str]:
    errors = validate_index(index, repo_root=repo_root, verify_nodes=True)
    if errors:
        raise ArtifactDagError("index validation failed: " + "; ".join(errors))
    snapshot = repository_path(repo_root, snapshot_rel)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    raw = _pretty_bytes(index)
    if snapshot.exists() and snapshot.read_bytes() != raw:
        raise ArtifactDagError("index snapshot path already contains different bytes")
    if not snapshot.exists():
        snapshot.write_bytes(raw)
    router = {"path": snapshot_rel, "sha256": _sha256_path(snapshot)}
    router_path = repository_path(repo_root, "configs/experiment/current-passport-artifact-index.json")
    router_path.parent.mkdir(parents=True, exist_ok=True)
    router_path.write_bytes(_pretty_bytes(router))
    return router


def load_current_index(repo_root: Path = ROOT) -> tuple[dict[str, Any], dict[str, str]]:
    router = load_json(repo_root / "configs/experiment/current-passport-artifact-index.json")
    if set(router) != {"path", "sha256"}:
        raise ArtifactDagError("stable artifact-index router must contain only path + sha256")
    rel, wanted = router.get("path"), router.get("sha256")
    if not isinstance(rel, str) or not isinstance(wanted, str) or not HEX64.fullmatch(wanted):
        raise ArtifactDagError("artifact-index router identity is malformed")
    try:
        path = repository_path(repo_root, rel)
    except PassportError as exc:
        raise ArtifactDagError(str(exc)) from exc
    if not path.is_file() or _sha256_path(path) != wanted:
        raise ArtifactDagError("artifact-index snapshot missing or router digest mismatched")
    index = load_json(path)
    errors = validate_index(index, repo_root=repo_root, verify_nodes=True)
    if errors:
        raise ArtifactDagError("artifact-index invalid: " + "; ".join(errors))
    return index, {"path": rel, "sha256": wanted}


def _load_indexed_node(repo_root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    try:
        digest = str(entry["operation_digest"])
        node_sha256 = str(entry["node_sha256"])
        expected_rel = node_relative_path(digest, node_sha256)
        if entry["path"] != expected_rel:
            raise ArtifactDagError("indexed node path is not the canonical content-addressed path")
        path = repository_path(repo_root, str(entry["path"]))
    except (KeyError, PassportError) as exc:
        raise ArtifactDagError("indexed node path is invalid") from exc
    node = load_json(path)
    validate_operation_node(node, repo_root=repo_root)
    if _sha256_path(path) != entry.get("node_sha256"):
        raise ArtifactDagError("indexed node file digest mismatch")
    return node


def _reuse_projection(node: Mapping[str, Any]) -> dict[str, Any]:
    reuse = node["reuse"]
    side_effect = node["side_effect_class"]
    return {
        "default_policy": reuse["default_policy"],
        "automatic_reuse_permitted": bool(reuse["automatic_reuse_permitted"]),
        "operation_allowlisted": bool(reuse["operation_allowlisted"]),
        "requires_experiment_policy_review": (
            side_effect in SCIENTIFIC_NO_AUTO_REUSE or side_effect == "DETERMINISTIC_COMPUTATION"
        ),
    }


def query_exact(spec: Mapping[str, Any], *, repo_root: Path = ROOT,
                index: Mapping[str, Any] | None = None) -> dict[str, Any]:
    digest = operation_digest(spec)
    query_verification = verify_operation_spec_provenance(spec, repo_root=repo_root)
    if query_verification["status"] != "PASS":
        return {
            "status": "PROVENANCE_MISMATCH",
            "operation_digest": digest,
            "reason": "current query identities do not verify against current bytes",
            "query_verification": query_verification,
        }
    if index is None:
        index, _ = load_current_index(repo_root)
    matches = [entry for entry in index.get("entries") or []
               if entry.get("operation_digest") == digest]
    if not matches:
        return {"status": "MISS", "operation_digest": digest,
                "query_verification": query_verification}

    valid: list[dict[str, Any]] = []
    provenance_failures: list[dict[str, Any]] = []
    stale_failures: list[dict[str, Any]] = []
    for entry in matches:
        try:
            node = _load_indexed_node(repo_root, entry)
        except (PassportError, ArtifactDagError) as exc:
            provenance_failures.append({"node_path": entry.get("path"), "reason": str(exc)})
            continue
        if node.get("status") != "PASS" or node["reuse"].get("mechanical_eligibility") != "ELIGIBLE":
            stale_failures.append({
                "node_path": entry.get("path"),
                "reason": "matching node is not an eligible PASS record",
            })
            continue
        verification = verify_node_provenance(node, repo_root=repo_root)
        if verification["output_status"] != "PASS":
            stale_failures.append({"node_path": entry.get("path"), "verification": verification})
            continue
        if verification["identity_status"] != "PASS":
            provenance_failures.append({"node_path": entry.get("path"), "verification": verification})
            continue
        valid.append({
            "node_path": entry.get("path"),
            "node_sha256": entry.get("node_sha256"),
            "producer_attempt": copy.deepcopy(node["producer_attempt"]),
            "side_effect_class": node["side_effect_class"],
            "execution_generation_id": node["identity_payload"].get("execution_generation_id"),
            "outputs": [artifact_identity(item) for item in node["outputs"]],
            "reuse": _reuse_projection(node),
            "verification": verification,
        })

    if valid:
        if len(valid) == 1:
            candidate = valid[0]
            return {
                "status": "REUSE_CANDIDATE", "classification": "EXACT_CACHE_HIT",
                "operation_digest": digest, "candidate_count": 1,
                "node_path": candidate["node_path"],
                "node_sha256": candidate["node_sha256"],
                "side_effect_class": candidate["side_effect_class"],
                "execution_generation_id": candidate["execution_generation_id"],
                "outputs": candidate["outputs"], "reuse": candidate["reuse"],
                "verification": candidate["verification"],
                "query_verification": query_verification,
                "producer_attempt": candidate["producer_attempt"],
            }
        return {
            "status": "REUSE_CANDIDATE", "classification": "EXACT_CACHE_HIT",
            "operation_digest": digest, "candidate_count": len(valid),
            "candidates": valid,
            "reuse": {
                "automatic_reuse_permitted": False,
                "requires_experiment_policy_review": True,
                "reason": "multiple producer attempts exist for the same operation identity; choose explicitly",
            },
            "query_verification": query_verification,
        }

    if provenance_failures:
        return {
            "status": "PROVENANCE_MISMATCH", "operation_digest": digest,
            "reason": "no exact candidate has fully verified provenance",
            "failures": provenance_failures, "stale_candidates": stale_failures,
            "query_verification": query_verification,
        }
    return {
        "status": "STALE_OR_INVALID_OUTPUT", "operation_digest": digest,
        "reason": "all exact candidates are stale, invalid, or ineligible",
        "failures": stale_failures, "query_verification": query_verification,
    }


CompatibilityPredicate = Callable[[Mapping[str, Any], Mapping[str, Any]], bool]


def find_compatible_historical_candidates(
    spec: Mapping[str, Any], *, predicate: CompatibilityPredicate,
    repo_root: Path = ROOT, index: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Caller supplies compatibility policy; similarity is never inferred here."""
    exact = operation_digest(spec)
    query_verification = verify_operation_spec_provenance(spec, repo_root=repo_root)
    if query_verification["status"] != "PASS":
        raise ArtifactDagError("current query provenance does not verify; compatibility search denied")
    query_identity = operation_identity_payload(spec)
    if index is None:
        index, _ = load_current_index(repo_root)
    candidates: list[dict[str, Any]] = []
    for entry in index.get("entries") or []:
        if entry.get("operation_digest") == exact:
            continue
        try:
            node = _load_indexed_node(repo_root, entry)
        except (PassportError, ArtifactDagError):
            continue
        if node.get("status") != "PASS":
            continue
        verification = verify_node_provenance(node, repo_root=repo_root)
        if verification["identity_status"] != "PASS" or verification["output_status"] != "PASS":
            continue
        candidate_identity = operation_identity_payload(node)
        if predicate(query_identity, candidate_identity):
            candidates.append({
                "operation_digest": node["operation_digest"],
                "node_path": entry["path"],
                "side_effect_class": node["side_effect_class"],
                "reuse": _reuse_projection(node),
            })
    return candidates


def query_operation(
    spec: Mapping[str, Any], *, repo_root: Path = ROOT,
    index: Mapping[str, Any] | None = None,
    compatibility_predicate: CompatibilityPredicate | None = None,
) -> dict[str, Any]:
    exact = query_exact(spec, repo_root=repo_root, index=index)
    if exact["status"] != "MISS" or compatibility_predicate is None:
        return exact
    candidates = find_compatible_historical_candidates(
        spec, predicate=compatibility_predicate, repo_root=repo_root, index=index
    )
    if candidates:
        return {
            "status": "COMPATIBLE_HISTORICAL_CANDIDATE",
            "operation_digest": exact["operation_digest"],
            "candidates": candidates,
            "automatic_reuse_permitted": False,
        }
    return exact


def _artifact_id(digest: str) -> str:
    return f"artifact:sha256:{digest}"


def _operation_id(digest: str) -> str:
    return f"operation:sha256:{digest}"


def build_edges(node: Mapping[str, Any], *, repo_root: Path = ROOT) -> list[dict[str, Any]]:
    """Project required edge types; supersedes remains outside content identity."""
    validate_operation_node(node, repo_root=repo_root)
    op_id = _operation_id(str(node["operation_digest"]))
    edges: list[dict[str, Any]] = []
    for artifact in node["identity_payload"].get("input_artifacts") or []:
        edges.append({"schema": EDGE_SCHEMA_ID, "edge_type": "consumed_by",
                      "source_id": _artifact_id(artifact["digest"]), "target_id": op_id,
                      "operation_digest": node["operation_digest"]})
    for artifact in node.get("outputs") or []:
        out_id = _artifact_id(artifact["digest"])
        edges.append({"schema": EDGE_SCHEMA_ID, "edge_type": "produced_by",
                      "source_id": op_id, "target_id": out_id,
                      "operation_digest": node["operation_digest"]})
        for parent_digest in artifact.get("derived_from_digests") or []:
            edges.append({"schema": EDGE_SCHEMA_ID, "edge_type": "derived_from",
                          "source_id": out_id, "target_id": _artifact_id(parent_digest),
                          "operation_digest": node["operation_digest"]})
    for previous in node.get("supersedes") or []:
        edges.append({"schema": EDGE_SCHEMA_ID, "edge_type": "supersedes",
                      "source_id": op_id, "target_id": _operation_id(previous),
                      "operation_digest": node["operation_digest"]})
    schema = load_json(EDGE_SCHEMA_PATH)
    for edge in edges:
        try:
            validate_schema(edge, schema)
        except PassportError as exc:
            raise ArtifactDagError(str(exc)) from exc
    return edges
