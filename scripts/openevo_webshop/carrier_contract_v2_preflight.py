#!/usr/bin/env python3
"""Pre-Round0 Carrier Contract v2 mechanism preflight receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.carrier_contract_v2_identity import load_identity
from scripts.openevo_webshop.carrier_scope_package_v1 import load_package
from scripts.openevo_webshop.carrier_health_failure_policy_v1 import load_policy, policy_identity
from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import require_evolution_compatible_bundle

SCHEMA = "openevo.carrier-contract-v2-preflight.v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def build_preflight(*, repo_root: Path, identity_path: Path, scope_package_path: Path,
                    health_policy_path: Path, arm_id: str, execution_git_sha: str,
                    memory_path: Path, skill_path: Path, agent_path: Path) -> dict[str, Any]:
    identity = load_identity(identity_path, repo_root)
    scope_package = load_package(scope_package_path)
    policy = load_policy(health_policy_path)
    projection = identity["identity_projection"]
    if projection["health_policy_id"] != policy["policy_id"]:
        raise ValueError("Carrier Contract preflight health policy ID drift")
    if projection["health_policy_identity_sha256"] != policy_identity(policy):
        raise ValueError("Carrier Contract preflight health policy identity drift")
    if projection["scope_policy_id"] != scope_package["construction_policy_id"]:
        raise ValueError("Carrier Contract preflight scope policy drift")
    if projection["health_schema_sha256"] != file_sha(repo_root / "schemas/carrier-health-v2.schema.json"):
        raise ValueError("Carrier Contract preflight health schema drift")
    memory = memory_path.read_text()
    skill = skill_path.read_text()
    agent = agent_path.read_text()
    checks = require_evolution_compatible_bundle(memory, skill, agent, context="Carrier Contract v2 pre-Round0")
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "formal_task_consumption_allowed": False,
        "arm_id": arm_id,
        "execution_git_sha": execution_git_sha,
        "carrier_contract_identity_sha256": identity["carrier_contract_identity_sha256"],
        "implementation_manifest_sha256": file_sha(identity_path),
        "scope_policy_id": projection["scope_policy_id"],
        "scope_package_identity_sha256": scope_package["package_identity_sha256"],
        "health_policy_id": policy["policy_id"],
        "health_policy_identity_sha256": policy_identity(policy),
        "health_schema_sha256": projection["health_schema_sha256"],
        "initial_carrier_contract_status": checks["status"],
        "initial_carrier_sha256": {
            "text_memory": file_sha(memory_path),
            "skill_bundle": file_sha(skill_path),
            "agent_system": file_sha(agent_path),
        },
        "final_panel_access_count": 0,
        "reward_access_count": 0,
    }
    payload["preflight_identity_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def validate_preflight(receipt: dict[str, Any], *, expected_arm: str | None = None) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "PASS":
        raise ValueError("Carrier Contract preflight status/schema mismatch")
    if receipt.get("formal_task_consumption_allowed") is not False:
        raise ValueError("Carrier Contract preflight cannot consume formal tasks")
    if receipt.get("initial_carrier_contract_status") != "PASS":
        raise ValueError("Carrier Contract initial carrier contract did not PASS")
    if expected_arm is not None and receipt.get("arm_id") != expected_arm:
        raise ValueError("Carrier Contract preflight arm mismatch")
    if int(receipt.get("final_panel_access_count", -1)) != 0 or int(receipt.get("reward_access_count", -1)) != 0:
        raise ValueError("Carrier Contract preflight authority boundary violated")
    expected = str(receipt.get("preflight_identity_sha256") or "")
    projection = {k: v for k, v in receipt.items() if k != "preflight_identity_sha256"}
    if expected != sha256_bytes(canonical_json(projection)):
        raise ValueError("Carrier Contract preflight receipt identity drift")
    return receipt


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--scope-package", type=Path, required=True)
    ap.add_argument("--health-policy", type=Path, required=True)
    ap.add_argument("--arm-id", required=True)
    ap.add_argument("--execution-git-sha", required=True)
    ap.add_argument("--memory", type=Path, required=True)
    ap.add_argument("--skill", type=Path, required=True)
    ap.add_argument("--agent", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    payload = build_preflight(
        repo_root=args.repo.resolve(), identity_path=args.identity.resolve(),
        scope_package_path=args.scope_package.resolve(), health_policy_path=args.health_policy.resolve(),
        arm_id=args.arm_id, execution_git_sha=args.execution_git_sha,
        memory_path=args.memory.resolve(), skill_path=args.skill.resolve(), agent_path=args.agent.resolve(),
    )
    write_receipt(args.output.resolve(), payload)
    print(json.dumps({"status": "PASS", "preflight_identity_sha256": payload["preflight_identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
