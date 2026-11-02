"""Read-only readiness gate for the Execution Runtime zero-formal resource canary.

This helper deliberately cannot acquire a GPU or launch a workload. It verifies the
frozen production package, ExecutionGeneration, qualified-topology registry, and
prospective next-campaign release, then reports the remaining external authority
blocker literally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_runtime.execution_generation import ExecutionGenerationError, validate_execution_generation
from scripts.openevo_runtime.production_package import ProductionPackageError, build_production_package, validate_production_package
from scripts.openevo_runtime.prospective_adoption import ProspectiveAdoptionError, validate_prospective_adoption_release
from scripts.openevo_runtime.qualified_topology import QualifiedTopologyError, load_qualified_topology_registry
from scripts.openevo_webshop.experiment_passport import canonical_bytes

SCHEMA = "openevo.execution-runtime-resource-canary-readiness.202609210125"
DEFAULT_RELEASE = ROOT / "configs/experiment/runtime-adoption/execution-runtime-next-campaign-adoption-202609171752.json"
DEFAULT_PRECANARY = ROOT / "docs/evidence/execution-runtime-202609162118/phase-g/PRODUCTION_RUNTIME_PACKAGE_REFRESH_202609210031.json"
DEFAULT_TOPOLOGY = ROOT / "configs/experiment/bounded-effective-state-qualified-topologies-202609170845.json"
DEFAULT_CANARY = ROOT / "docs/evidence/execution-runtime-202609162118/phase-g/ZERO_FORMAL_RESOURCE_CANARY_GPU7_202609210119.json"
DEFAULT_CONSUMER = ROOT / "docs/evidence/execution-runtime-202609162118/phase-g/EXACT_CONSUMER_VALIDATION_202609210122.json"


class ResourceCanaryReadinessError(RuntimeError):
    """The frozen pre-canary identities cannot be reconciled exactly."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceCanaryReadinessError(f"unreadable JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ResourceCanaryReadinessError(f"JSON root must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ResourceCanaryReadinessError(f"unreadable file: {path}") from exc


def _clean_head() -> str:
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ResourceCanaryReadinessError("repository must be clean for exact readiness readback")
    return head


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ResourceCanaryReadinessError(f"{label} drift")


def build_readiness_report(
    *,
    release_path: Path = DEFAULT_RELEASE,
    precanary_path: Path = DEFAULT_PRECANARY,
    topology_path: Path = DEFAULT_TOPOLOGY,
    canary_path: Path = DEFAULT_CANARY,
    consumer_path: Path = DEFAULT_CONSUMER,
    require_clean_head: bool = True,
) -> dict[str, Any]:
    source_head = _clean_head() if require_clean_head else None
    release = _read_object(release_path)
    try:
        validate_prospective_adoption_release(release)
    except ProspectiveAdoptionError as exc:
        raise ResourceCanaryReadinessError("prospective adoption release validation failed") from exc
    precanary = _read_object(precanary_path)
    generation = precanary.get("execution_generation")
    if not isinstance(generation, Mapping):
        raise ResourceCanaryReadinessError("pre-canary execution_generation missing")
    try:
        validate_execution_generation(generation)
    except ExecutionGenerationError as exc:
        raise ResourceCanaryReadinessError("execution generation validation failed") from exc

    try:
        package = build_production_package()
        validate_production_package(package)
    except ProductionPackageError as exc:
        raise ResourceCanaryReadinessError("production package validation failed") from exc
    _require_equal(package["package_identity"], precanary.get("production_runtime_package", {}).get("package_identity"), "pre-canary package identity")
    _require_equal(package["package_identity"], release.get("production_runtime_package_identity"), "release package identity")
    _require_equal(generation["runtime_package_identity"], package["package_identity"], "generation runtime package identity")
    _require_equal(generation["generation_id"], release.get("execution_generation_id"), "release generation identity")
    _require_equal(generation.get("resource_policy_identity"), _file_sha256(topology_path), "generation resource-policy file identity")

    topology_raw = _read_object(topology_path)
    try:
        registry = load_qualified_topology_registry(
            topology_path,
            base_config_sha256=topology_raw.get("base_config_sha256"),
            profile_registry_sha256=topology_raw.get("profile_registry_sha256"),
        )
    except QualifiedTopologyError as exc:
        raise ResourceCanaryReadinessError("qualified topology registry validation failed") from exc
    canary_identity = release.get("resource_canary_receipt_identity")
    consumer_identity = release.get("exact_consumer_validation_receipt_identity")
    canary_executed = canary_identity is not None
    consumer_validated = consumer_identity is not None
    blockers: list[str] = []
    if canary_executed:
        canary = _read_object(canary_path)
        _require_equal(_file_sha256(canary_path), canary_identity, "resource canary receipt identity")
        _require_equal(canary.get("status"), "PASS_ZERO_FORMAL_RESOURCE_CANARY", "resource canary status")
        _require_equal(canary.get("production_runtime_package_identity"), package["package_identity"], "canary package identity")
        _require_equal(canary.get("execution_generation_id"), generation["generation_id"], "canary generation identity")
        if canary.get("formal_rows_consumed") != 0 or canary.get("final_panel_access_count") != 0:
            raise ResourceCanaryReadinessError("resource canary crossed zero-formal/final-panel boundary")
        if canary.get("resource_topology_authority") is not False:
            raise ResourceCanaryReadinessError("resource canary cannot grant topology authority")
    else:
        blockers.extend([
            "DEFAULT_ZERO_GPU_BOUNDARY",
            "SEPARATE_RESOURCE_EXECUTION_AUTHORITY_NOT_BOUND",
            "ZERO_FORMAL_RESOURCE_CANARY_NOT_EXECUTED",
        ])

    if consumer_validated:
        consumer = _read_object(consumer_path)
        _require_equal(_file_sha256(consumer_path), consumer_identity, "consumer validation receipt identity")
        _require_equal(consumer.get("status"), "PASS_EXACT_CONSUMER_VALIDATION", "consumer validation status")
        _require_equal(consumer.get("production_runtime_package_identity"), package["package_identity"], "consumer package identity")
        _require_equal(consumer.get("execution_generation_id"), generation["generation_id"], "consumer generation identity")
        _require_equal(consumer.get("resource_canary_receipt_identity"), canary_identity, "consumer canary identity")
        if consumer.get("external_campaign_release_still_required") is not True:
            raise ResourceCanaryReadinessError("consumer validation cannot grant campaign release")
    elif canary_executed:
        blockers.append("EXACT_CONSUMER_VALIDATION_PENDING")

    status = (
        "PASS_CANARY_AND_CONSUMER_VALIDATION"
        if canary_executed and consumer_validated
        else "RESOURCE_CANARY_PASS_CONSUMER_VALIDATION_PENDING"
        if canary_executed
        else "BLOCKED_RESOURCE_EXECUTION_AUTHORITY"
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "source_git_sha": source_head,
        "production_runtime_package_identity": package["package_identity"],
        "execution_generation_id": generation["generation_id"],
        "qualified_topology_registry_id": registry["registry_id"],
        "prospective_adoption_release_id": release["release_id"],
        "prospective_adoption_status": release["status"],
        "blockers": blockers,
        "gpu_requests": 0,
        "formal_rows_consumed": 0,
        "model_calls": 0,
        "final_panel_access_count": 0,
        "resource_canary_executed": canary_executed,
        "exact_consumer_validation_passed": consumer_validated,
        "scientific_authority": False,
        "formal_launch_authority": False,
        "resource_topology_authority": False,
        "canary_execution_authority": False,
    }
    report["readiness_identity"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_readiness_report(require_clean_head=True)
    _require_equal(report["source_git_sha"], args.expected_git_sha, "expected Git SHA")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(report) + b"\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
