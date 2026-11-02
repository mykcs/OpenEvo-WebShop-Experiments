#!/usr/bin/env python3
"""Shared-arm Carrier Contract v2 adoption receipt and parity validator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.carrier_contract_v2_identity import load_identity
from scripts.openevo_webshop.carrier_scope_package_v1 import load_package
from scripts.openevo_webshop.carrier_contract_v2_preflight import validate_preflight

SCHEMA = "openevo.carrier-contract-v2-adoption.v1"
TEMPLATE_SCHEMA = "openevo.carrier-contract-v2-adoption-template.v1"
PARITY_FIELDS = (
    "carrier_contract_identity_sha256",
    "implementation_manifest_sha256",
    "scope_policy_id",
    "scope_package_identity_sha256",
    "health_policy_id",
    "health_policy_identity_sha256",
    "health_schema_sha256",
    "execution_git_sha",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def common_binding(identity_path: Path, identity: dict[str, Any], scope_package: dict[str, Any], execution_git_sha: str) -> dict[str, Any]:
    projection = identity["identity_projection"]
    return {
        "carrier_contract_identity_sha256": identity["carrier_contract_identity_sha256"],
        "implementation_manifest_sha256": file_sha(identity_path),
        "scope_policy_id": projection["scope_policy_id"],
        "scope_package_identity_sha256": scope_package["package_identity_sha256"],
        "health_policy_id": projection["health_policy_id"],
        "health_policy_identity_sha256": projection["health_policy_identity_sha256"],
        "health_schema_sha256": projection["health_schema_sha256"],
        "execution_git_sha": execution_git_sha,
    }


def build_receipt(*, identity_path: Path, repo_root: Path, scope_package_path: Path,
                  execution_git_sha: str, arms: list[dict[str, Any]]) -> dict[str, Any]:
    identity = load_identity(identity_path, repo_root)
    scope_package = load_package(scope_package_path)
    if len(arms) != 2:
        raise ValueError("Carrier Contract adoption requires exactly two arms")
    arm_ids = [str(arm.get("arm_id") or "") for arm in arms]
    if len(set(arm_ids)) != 2 or any(not value for value in arm_ids):
        raise ValueError("Carrier Contract arm IDs must be two unique non-empty values")
    common = common_binding(identity_path, identity, scope_package, execution_git_sha)
    bindings = []
    for arm in arms:
        preflight = validate_preflight(dict(arm.get("preflight_receipt") or {}), expected_arm=arm["arm_id"])
        for field, expected in common.items():
            if preflight.get(field) != expected:
                raise ValueError(f"Carrier Contract preflight shared binding drift: {field}")
        bindings.append({
            "arm_id": arm["arm_id"],
            "treatment_role": arm["treatment_role"],
            "pre_round0_health_status": "PASS",
            "preflight_identity_sha256": preflight["preflight_identity_sha256"],
            "preflight_receipt": preflight,
            **common,
        })
    payload = {
        "schema": SCHEMA,
        "status": "RESOLVED",
        "formal_task_consumption_allowed": True,
        "shared_mechanism_parity": "PASS",
        "arm_bindings": sorted(bindings, key=lambda x: x["arm_id"]),
        "realized_carrier_payload_parity_required": False,
    }
    payload["adoption_identity_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def validate_receipt(receipt: dict[str, Any], *, identity_path: Path, repo_root: Path,
                     scope_package_path: Path, expected_execution_git_sha: str | None = None) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "RESOLVED":
        raise ValueError("Carrier Contract adoption receipt status/schema mismatch")
    if receipt.get("shared_mechanism_parity") != "PASS" or receipt.get("formal_task_consumption_allowed") is not True:
        raise ValueError("Carrier Contract adoption is not launch-admissible")
    if receipt.get("realized_carrier_payload_parity_required") is not False:
        raise ValueError("Carrier Contract adoption incorrectly requires realized payload parity")
    bindings = receipt.get("arm_bindings")
    if not isinstance(bindings, list) or len(bindings) != 2:
        raise ValueError("Carrier Contract adoption requires exactly two arm bindings")
    if len({str(x.get("arm_id") or "") for x in bindings}) != 2:
        raise ValueError("Carrier Contract adoption arm identity drift")
    for field in PARITY_FIELDS:
        values = {str(binding.get(field) or "") for binding in bindings}
        if len(values) != 1 or "" in values:
            raise ValueError(f"Carrier Contract shared-arm parity mismatch: {field}")
    if any(binding.get("pre_round0_health_status") != "PASS" for binding in bindings):
        raise ValueError("Carrier Contract pre-Round0 health parity mismatch")
    for binding in bindings:
        preflight = validate_preflight(dict(binding.get("preflight_receipt") or {}), expected_arm=binding["arm_id"])
        if binding.get("preflight_identity_sha256") != preflight.get("preflight_identity_sha256"):
            raise ValueError("Carrier Contract preflight identity binding drift")
    identity = load_identity(identity_path, repo_root)
    scope_package = load_package(scope_package_path)
    execution_sha = str(bindings[0]["execution_git_sha"])
    if expected_execution_git_sha is not None and execution_sha != expected_execution_git_sha:
        raise ValueError("Carrier Contract adoption execution Git SHA mismatch")
    expected_common = common_binding(identity_path, identity, scope_package, execution_sha)
    for binding in bindings:
        preflight = validate_preflight(dict(binding.get("preflight_receipt") or {}), expected_arm=binding["arm_id"])
        for field, expected in expected_common.items():
            if binding.get(field) != expected:
                raise ValueError(f"Carrier Contract binding drift: {field}")
            if preflight.get(field) != expected:
                raise ValueError(f"Carrier Contract preflight binding drift: {field}")
    expected_identity = str(receipt.get("adoption_identity_sha256") or "")
    projection = {k: v for k, v in receipt.items() if k != "adoption_identity_sha256"}
    if expected_identity != sha256_bytes(canonical_json(projection)):
        raise ValueError("Carrier Contract adoption receipt identity drift")
    return receipt


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--scope-package", type=Path, required=True)
    ap.add_argument("--execution-git-sha", required=True)
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--preflight-a", type=Path, required=True)
    ap.add_argument("--preflight-b", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    repo_root = args.repo.resolve()
    if args.validate:
        receipt = json.loads(args.output.read_text())
        validate_receipt(receipt, identity_path=args.identity.resolve(), repo_root=repo_root,
                         scope_package_path=args.scope_package.resolve(),
                         expected_execution_git_sha=args.execution_git_sha)
    else:
        arms = [
            {"arm_id": args.arm_a, "treatment_role": "ACCELERATION_ONLY", "preflight_receipt": json.loads(args.preflight_a.read_text())},
            {"arm_id": args.arm_b, "treatment_role": "ACCELERATION_PLUS_GDR", "preflight_receipt": json.loads(args.preflight_b.read_text())},
        ]
        receipt = build_receipt(identity_path=args.identity.resolve(), repo_root=repo_root,
                                scope_package_path=args.scope_package.resolve(),
                                execution_git_sha=args.execution_git_sha, arms=arms)
        write_receipt(args.output, receipt)
    print(json.dumps({"status": "PASS", "adoption_identity_sha256": receipt["adoption_identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
