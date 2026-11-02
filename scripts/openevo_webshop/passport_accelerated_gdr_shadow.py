"""Shadow/advisory Passport content-addressed DAG adoption for accelerated-GDR prelaunch work.

This module is deliberately non-executing: it builds/query identities, normalizes
reuse advice, writes append-only advisory records, and counts work actually avoided.
It never launches workers, consumes WebShop rows, mutates models/carriers/GDR state,
or authorizes final evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop import experiment_artifact_dag as dag
from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    canonical_bytes,
    load_json,
    repository_path,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_SCHEMA_PATH = ROOT / "schemas/passport-accelerated-gdr-resource-topology-202609142055.schema.json"
TOPOLOGY_SCHEMA_ID = "openevo.accelerated-gdr-resource-topology.202609142055"
SHADOW_RECORD_SCHEMA = "openevo.passport-accelerated-gdr-shadow-record.202609142055"
SHADOW_METRICS_SCHEMA = "openevo.passport-accelerated-gdr-shadow-metrics.202609142055"

PREFERRED_TOPOLOGY = "FOUR_GPU_2_WORKERS_PER_GPU"
FALLBACK_TOPOLOGY = "TWO_GPU_PER_ARM_1_WORKER_PER_GPU"

ENGINEERING_OPERATIONS: dict[str, str] = {
    "exact_checkout_resolution": "PURE_ENGINEERING",
    "acceleration_successor_qualification": "DETERMINISTIC_COMPUTATION",
    "common_start_materialization": "PURE_ENGINEERING",
    "model_config_image_hash_verification": "DETERMINISTIC_COMPUTATION",
    "carrier_contract_preflight": "DETERMINISTIC_COMPUTATION",
    "gdr_mechanism_preflight": "DETERMINISTIC_COMPUTATION",
    "worker_resource_topology_qualification": "DETERMINISTIC_COMPUTATION",
    "deterministic_benchmark_analysis": "DETERMINISTIC_COMPUTATION",
    "artifact_archive": "PURE_ENGINEERING",
    "remote_readback": "PURE_ENGINEERING",
}

PROTECTED_SCIENTIFIC_OPERATIONS: dict[str, str] = {
    "formal_rollout": "FORMAL_ROLLOUT",
    "model_update": "MODEL_UPDATE",
    "gdr_write": "STOCHASTIC_SCIENTIFIC_RESULT",
    "stochastic_scientific_result": "STOCHASTIC_SCIENTIFIC_RESULT",
    "carrier_scientific_transition": "STOCHASTIC_SCIENTIFIC_RESULT",
    "final_evaluation": "FINAL_EVALUATION",
}

ALL_OPERATIONS = {**ENGINEERING_OPERATIONS, **PROTECTED_SCIENTIFIC_OPERATIONS}


class ShadowPassportError(ValueError):
    """Invalid shadow request/record. No execution authority is implied."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gpu_uuid_set_digest(gpu_uuids: list[str]) -> str:
    """Order-independent exact identity for the bound physical GPU UUID set."""
    if not gpu_uuids or any(not isinstance(item, str) or not item for item in gpu_uuids):
        raise ShadowPassportError("gpu UUID set must contain non-empty strings")
    if len(set(gpu_uuids)) != len(gpu_uuids):
        raise ShadowPassportError("gpu UUID set must not contain duplicates")
    return _sha256(canonical_bytes(sorted(gpu_uuids)))


def topology_digest(topology: Mapping[str, Any]) -> str:
    """Stable digest over semantic topology fields, independent of file formatting."""
    validate_topology(topology)
    return _sha256(canonical_bytes(topology))


def validate_topology(topology: Mapping[str, Any]) -> None:
    """Validate schema plus exact semantics of the two authorized topology shapes."""
    try:
        validate_schema(topology, load_json(TOPOLOGY_SCHEMA_PATH))
    except PassportError as exc:
        raise ShadowPassportError(str(exc)) from exc
    if topology.get("schema") != TOPOLOGY_SCHEMA_ID:
        raise ShadowPassportError("resource topology schema mismatch")
    topology_id = topology.get("topology_id")
    if topology_id == PREFERRED_TOPOLOGY:
        expected = {
            "gpu_count": 4,
            "gpus_per_arm": None,
            "workers_per_gpu": 2,
            "total_workers": 8,
            "co_resident": True,
            "baseline_workers": 4,
            "gdr_workers": 4,
        }
    elif topology_id == FALLBACK_TOPOLOGY:
        expected = {
            "gpu_count": 4,
            "gpus_per_arm": 2,
            "workers_per_gpu": 1,
            "total_workers": 4,
            "co_resident": False,
            "baseline_workers": 2,
            "gdr_workers": 2,
        }
    else:
        raise ShadowPassportError("unsupported accelerated-GDR resource topology")
    for key, wanted in expected.items():
        if topology.get(key) != wanted:
            raise ShadowPassportError(
                f"{topology_id}.{key}: expected {wanted!r}, observed {topology.get(key)!r}"
            )


def load_topology(repo_root: Path, relative_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a repository topology descriptor and return payload + exact DAG ref."""
    try:
        path = repository_path(repo_root, relative_path)
    except PassportError as exc:
        raise ShadowPassportError(str(exc)) from exc
    payload = load_json(path)
    validate_topology(payload)
    return payload, {
        "kind": "repository_file",
        "repository": None,
        "revision": None,
        "path": relative_path,
        "sha256": _sha256(path.read_bytes()),
        "evidence_path": None,
        "reason": None,
    }


def _copy_ref(ref: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return copy.deepcopy(dict(ref)) if ref is not None else None


def build_operation_spec(
    operation_name: str,
    *,
    code_identity: Mapping[str, Any],
    config_identity: Mapping[str, Any],
    input_artifacts: list[Mapping[str, Any]],
    contract_identities: list[Mapping[str, Any]] | None = None,
    runtime_identity_policy: str = "NOT_IDENTITY_BEARING",
    runtime_identity: Mapping[str, Any] | None = None,
    dataset_schedule_identity: Mapping[str, Any] | None = None,
    stochastic_identity: Mapping[str, Any] | None = None,
    topology: Mapping[str, Any] | None = None,
    topology_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a #479 DAG operation identity for one shadow prelaunch operation."""
    if operation_name not in ALL_OPERATIONS:
        raise ShadowPassportError(f"unknown shadow operation: {operation_name}")
    contracts = [copy.deepcopy(dict(item)) for item in (contract_identities or [])]
    if operation_name == "worker_resource_topology_qualification":
        if topology is None or topology_ref is None:
            raise ShadowPassportError("resource qualification requires topology + topology_ref")
        if runtime_identity_policy != "BOUND" or runtime_identity is None:
            raise ShadowPassportError("resource qualification requires BOUND runtime_identity")
        validate_topology(topology)
        contracts.append({
            "name": f"resource_topology:{topology_digest(topology)}",
            "identity": _copy_ref(topology_ref),
        })
    elif topology is not None or topology_ref is not None:
        raise ShadowPassportError("topology identity belongs only to resource qualification")

    side_effect = ALL_OPERATIONS[operation_name]
    science_protected = operation_name in PROTECTED_SCIENTIFIC_OPERATIONS
    stochastic = copy.deepcopy(dict(stochastic_identity or {}))
    if science_protected and not stochastic:
        stochastic = {"mode": "authority-required", "shadow_only": True}

    spec = {
        "operation_type": f"accelerated-gdr.shadow.{operation_name}",
        "operation_schema_version": "202609142055",
        "side_effect_class": side_effect,
        "identity_payload": {
            "code_identity": _copy_ref(code_identity),
            "config_identity": _copy_ref(config_identity),
            "runtime_identity_policy": runtime_identity_policy,
            "runtime_identity": _copy_ref(runtime_identity),
            "dataset_schedule_identity": _copy_ref(dataset_schedule_identity),
            "contract_identities": contracts,
            "input_artifacts": [copy.deepcopy(dict(item)) for item in input_artifacts],
            "stochastic_identity": stochastic,
        },
    }
    try:
        dag.operation_identity_payload(spec)
    except (dag.ArtifactDagError, PassportError) as exc:
        raise ShadowPassportError(str(exc)) from exc
    return spec


def normalize_lookup_outcome(operation_name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert raw DAG lookup into accelerated-GDR shadow/advisory semantics."""
    if operation_name not in ALL_OPERATIONS:
        raise ShadowPassportError(f"unknown shadow operation: {operation_name}")
    normalized = copy.deepcopy(dict(result))
    protected = operation_name in PROTECTED_SCIENTIFIC_OPERATIONS
    if protected and normalized.get("status") == "REUSE_CANDIDATE":
        normalized["raw_status"] = "REUSE_CANDIDATE"
        normalized["status"] = "REUSE_CANDIDATE_REQUIRES_SCIENTIFIC_AUTHORITY"
        normalized["scientific_authority_required"] = True
        reuse = copy.deepcopy(dict(normalized.get("reuse") or {}))
        reuse["automatic_reuse_permitted"] = False
        reuse["requires_experiment_policy_review"] = True
        normalized["reuse"] = reuse
    else:
        normalized["scientific_authority_required"] = False
    normalized["shadow_mode"] = True
    normalized["advisory_only"] = True
    return normalized


def query_shadow(
    operation_name: str,
    spec: Mapping[str, Any],
    *,
    repo_root: Path = ROOT,
    index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read-only exact lookup. No execution is started on MISS."""
    if operation_name not in ALL_OPERATIONS:
        raise ShadowPassportError(f"unknown shadow operation: {operation_name}")
    expected = f"accelerated-gdr.shadow.{operation_name}"
    if spec.get("operation_type") != expected:
        raise ShadowPassportError("operation_name does not match spec.operation_type")
    raw = dag.query_operation(spec, repo_root=repo_root, index=index)
    return normalize_lookup_outcome(operation_name, raw)


def build_shadow_record(
    operation_name: str,
    spec: Mapping[str, Any],
    lookup: Mapping[str, Any],
    *,
    topology: Mapping[str, Any] | None = None,
    consumer_reuse_accepted: bool = False,
    note: str | None = None,
) -> dict[str, Any]:
    """Build a zero-science advisory observation; caller may persist append-only."""
    if operation_name not in ALL_OPERATIONS:
        raise ShadowPassportError(f"unknown shadow operation: {operation_name}")
    expected = f"accelerated-gdr.shadow.{operation_name}"
    if spec.get("operation_type") != expected:
        raise ShadowPassportError("operation_name does not match spec.operation_type")
    guarded = operation_name in PROTECTED_SCIENTIFIC_OPERATIONS
    if guarded and consumer_reuse_accepted:
        raise ShadowPassportError("scientific reuse cannot be accepted by the shadow layer")
    identity = dag.operation_identity_payload(spec)
    record = {
        "schema": SHADOW_RECORD_SCHEMA,
        "shadow_mode": True,
        "advisory_only": True,
        "operation_name": operation_name,
        "operation_type": spec["operation_type"],
        "operation_digest": dag.operation_digest(spec),
        "side_effect_class": spec["side_effect_class"],
        "input_identities": copy.deepcopy(identity["input_artifacts"]),
        "output_identities": copy.deepcopy(list(lookup.get("outputs") or [])),
        "lookup_outcome": lookup.get("status"),
        "lookup_classification": lookup.get("classification"),
        "source_passport_node": lookup.get("node_path"),
        "producer_attempt": copy.deepcopy(lookup.get("producer_attempt")),
        "provenance_verification": copy.deepcopy(
            lookup.get("query_verification") or lookup.get("verification")
        ),
        "scientific_authority_required": bool(
            lookup.get("scientific_authority_required", guarded)
        ),
        "consumer_reuse_accepted": bool(consumer_reuse_accepted),
        "resource_topology": copy.deepcopy(dict(topology)) if topology is not None else None,
        "resource_topology_digest": topology_digest(topology) if topology is not None else None,
        "formal_webshop_rows_delta": 0,
        "model_update_delta": 0,
        "gdr_write_delta": 0,
        "carrier_scientific_transition_delta": 0,
        "final_evaluation_delta": 0,
        "extra_gpu_work_for_cache_hit": 0,
        "note": note,
    }
    canonical_bytes(record)
    return record


def write_shadow_record(
    repo_root: Path,
    record: Mapping[str, Any],
    *,
    evidence_root: str = "docs/evidence/passport-accelerated-gdr-shadow-20260914/observations",
) -> dict[str, str]:
    """Write one immutable, content-addressed advisory observation."""
    raw = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = _sha256(raw)
    relative = f"{evidence_root}/{digest}.json"
    try:
        path = repository_path(repo_root, relative)
    except PassportError as exc:
        raise ShadowPassportError(str(exc)) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ShadowPassportError("content-addressed shadow record path contains different bytes")
    if not path.exists():
        path.write_bytes(raw)
    return {"path": relative, "sha256": digest}


def new_metrics(*, c_workline_head: str | None = None) -> dict[str, Any]:
    return {
        "schema": SHADOW_METRICS_SCHEMA,
        "shadow_mode": True,
        "c_workline_head": c_workline_head,
        "lookup_exact_hits": 0,
        "lookup_misses": 0,
        "lookup_provenance_mismatches": 0,
        "lookup_stale_outputs": 0,
        "scientific_authority_required_hits": 0,
        "gpu_qualification_candidates_identified": 0,
        "avoided_repeat_checkout_resolutions": 0,
        "avoided_repeat_hash_verifications": 0,
        "avoided_repeat_qualifications": 0,
        "avoided_repeat_materializations": 0,
        "avoided_repeat_deterministic_analyses": 0,
        "avoided_repeat_archive_operations": 0,
        "avoided_repeat_remote_readbacks": 0,
        "avoided_gpu_qualification_operations": 0,
        "formal_webshop_rows_consumed": 0,
        "extra_gpu_work_for_cache_hits": 0,
    }


def observe_lookup(metrics: Mapping[str, Any], operation_name: str, lookup: Mapping[str, Any]) -> dict[str, Any]:
    """Count observed lookup classes only; finding a candidate is not saved work."""
    result = copy.deepcopy(dict(metrics))
    status = lookup.get("status")
    if status == "REUSE_CANDIDATE":
        result["lookup_exact_hits"] += 1
        if operation_name in {
            "acceleration_successor_qualification",
            "worker_resource_topology_qualification",
        }:
            result["gpu_qualification_candidates_identified"] += 1
    elif status == "REUSE_CANDIDATE_REQUIRES_SCIENTIFIC_AUTHORITY":
        result["lookup_exact_hits"] += 1
        result["scientific_authority_required_hits"] += 1
    elif status == "MISS":
        result["lookup_misses"] += 1
    elif status == "PROVENANCE_MISMATCH":
        result["lookup_provenance_mismatches"] += 1
    elif status == "STALE_OR_INVALID_OUTPUT":
        result["lookup_stale_outputs"] += 1
    return result


_AVOIDED_COUNTER = {
    "exact_checkout_resolution": "avoided_repeat_checkout_resolutions",
    "acceleration_successor_qualification": "avoided_repeat_qualifications",
    "common_start_materialization": "avoided_repeat_materializations",
    "model_config_image_hash_verification": "avoided_repeat_hash_verifications",
    "carrier_contract_preflight": "avoided_repeat_qualifications",
    "gdr_mechanism_preflight": "avoided_repeat_qualifications",
    "worker_resource_topology_qualification": "avoided_repeat_qualifications",
    "deterministic_benchmark_analysis": "avoided_repeat_deterministic_analyses",
    "artifact_archive": "avoided_repeat_archive_operations",
    "remote_readback": "avoided_repeat_remote_readbacks",
}


def accept_engineering_reuse(
    metrics: Mapping[str, Any],
    operation_name: str,
    lookup: Mapping[str, Any],
) -> dict[str, Any]:
    """Count real avoided work only after the external engineering consumer accepts it."""
    if operation_name not in ENGINEERING_OPERATIONS:
        raise ShadowPassportError("scientific operation cannot be accepted by shadow reuse")
    if lookup.get("status") != "REUSE_CANDIDATE":
        raise ShadowPassportError("only an exact verified REUSE_CANDIDATE can count as avoided work")
    result = copy.deepcopy(dict(metrics))
    counter = _AVOIDED_COUNTER[operation_name]
    result[counter] += 1
    if operation_name in {
        "acceleration_successor_qualification",
        "worker_resource_topology_qualification",
    }:
        result["avoided_gpu_qualification_operations"] += 1
    return result


def assert_zero_science(metrics: Mapping[str, Any]) -> None:
    if metrics.get("formal_webshop_rows_consumed") != 0:
        raise ShadowPassportError("shadow metrics cannot consume formal WebShop rows")
    if metrics.get("extra_gpu_work_for_cache_hits") != 0:
        raise ShadowPassportError("shadow adoption cannot run extra GPU work to create cache hits")
