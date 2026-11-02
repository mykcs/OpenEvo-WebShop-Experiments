#!/usr/bin/env python3
"""Content-addressed Carrier Contract v2 implementation identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "openevo.carrier-contract-v2-implementation-identity.v1"
SUCCESSOR_SCHEMA = "openevo.carrier-implementation-identity.202609300734"
SOURCE_MAP_SCHEMA = "openevo.carrier-implementation-source-map.202609300734"
CONTRACT_NAME = "Carrier Contract v2"

ROLE_PATHS: dict[str, tuple[str, ...]] = {
    "generator": (
        "scripts/openevo_webshop/ceiling1_stage2_reflector.py",
        "scripts/openevo_webshop/ceiling1_stage2_reflector_qwen3.py",
        "scripts/openevo_webshop/launch_ceiling1_stage2_reflector_container.sh",
        "scripts/openevo_webshop/launch_ceiling1_stage2_reflector_qwen3_container.sh",
    ),
    "validator": (
        "scripts/openevo_webshop/ceiling1_harness201_carrier_contract.py",
    ),
    "repair": (
        "scripts/openevo_webshop/ceiling1_stage2_reflector.py",
        "scripts/openevo_webshop/ceiling1_stage2_reflector_qwen3.py",
    ),
    "scope_policy": (
        "scripts/openevo_webshop/carrier_scope_manifest_v1.py",
        "scripts/openevo_webshop/carrier_scope_package_v1.py",
        "schemas/carrier-scope-manifest-v1.schema.json",
    ),
    "state_entry_resume": (
        "scripts/openevo_webshop/ceiling1_stage2_arm_controller.py",
        "scripts/openevo_webshop/verify_ceiling1_stage2_initial_state.py",
    ),
    "router": (
        "scripts/openevo_webshop/ceiling1_harness2.py",
    ),
    "governor": (
        "scripts/openevo_webshop/ceiling1_harness2.py",
    ),
    "runtime_readback": (
        "scripts/openevo_webshop/ceiling1_harness2_runtime.py",
        "scripts/openevo_webshop/ceiling1_harness2011_runtime_qwen3.py",
    ),
    "instrumentation": (
        "scripts/openevo_webshop/ceiling1_harness2_runtime.py",
        "scripts/openevo_webshop/ceiling1_harness2011_runtime_qwen3.py",
        "scripts/openevo_webshop/carrier_health_v2.py",
        "scripts/openevo_webshop/build_carrier_health_v2.py",
        "schemas/carrier-health-v2.schema.json",
    ),
    "health": (
        "scripts/openevo_webshop/carrier_health_v2.py",
        "scripts/openevo_webshop/build_carrier_health_v2.py",
        "schemas/carrier-health-v2.schema.json",
    ),
    "health_failure": (
        "scripts/openevo_webshop/carrier_health_failure_policy_v1.py",
        "configs/experiment/carriers/carrier-health-failure-policy-v1.json",
        "schemas/carrier-health-failure-policy-v1.schema.json",
    ),
    "identity_admission": (
        "scripts/openevo_webshop/carrier_contract_v2_identity.py",
        "scripts/openevo_webshop/carrier_contract_v2_adoption.py",
        "scripts/openevo_webshop/carrier_contract_v2_preflight.py",
        "schemas/carrier-contract-v2-implementation-identity-v1.schema.json",
        "schemas/carrier-contract-v2-adoption-v1.schema.json",
        "schemas/carrier-contract-v2-preflight-v1.schema.json",
    ),
}

SUCCESSOR_EXTRA_ROLE_PATHS: dict[str, tuple[str, ...]] = {
    "generator": (
        "scripts/openevo_webshop/ceiling1_stage2_reflector_container_driver.sh",
        "scripts/openevo_webshop/ceiling1_stage2_reflector_qwen3_container_driver.sh",
        "scripts/openevo_webshop/bounded_carrier_container_driver_20260916.sh",
    ),
    "state_entry_resume": (
        "scripts/openevo_webshop/ceiling1_stage2_round_qwen3_container_driver.sh",
    ),
    "identity_admission": (
        "scripts/openevo_webshop/carrier_contract_successor_202609300734.py",
    ),
}


HEALTH_ROLE_KEYS = {
    "generator": "generator",
    "validator": "validator",
    "repair": "repair",
    "router": "router",
    "governor": "governor",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _components(repo_root: Path) -> dict[str, str]:
    paths = sorted({path for values in ROLE_PATHS.values() for path in values})
    out: dict[str, str] = {}
    for rel in paths:
        path = repo_root / rel
        if not path.is_file():
            raise ValueError(f"Carrier Contract component missing: {rel}")
        out[rel] = file_sha(path)
    return out


def _role_identities(
    components: dict[str, str], role_paths: dict[str, tuple[str, ...]] | None = None
) -> dict[str, str]:
    result: dict[str, str] = {}
    for role, paths in sorted((role_paths or ROLE_PATHS).items()):
        projection = [{"path": path, "sha256": components[path]} for path in paths]
        result[role] = sha256_bytes(canonical_json({"role": role, "components": projection}))
    return result


def build_identity(repo_root: Path) -> dict[str, Any]:
    from scripts.openevo_webshop.carrier_scope_manifest_v1 import POLICY_ID as scope_policy_id
    from scripts.openevo_webshop.carrier_health_failure_policy_v1 import load_policy, policy_identity

    components = _components(repo_root)
    roles = _role_identities(components)
    policy = load_policy(repo_root / "configs/experiment/carriers/carrier-health-failure-policy-v1.json")
    projection = {
        "contract_name": CONTRACT_NAME,
        "contract_version": 2,
        "components": components,
        "role_identities": roles,
        "carrier_generator_identities": {
            "text_memory": roles["generator"],
            "skill_bundle": roles["generator"],
            "agent_system": roles["generator"],
        },
        "scope_policy_id": scope_policy_id,
        "health_policy_id": policy["policy_id"],
        "health_policy_identity_sha256": policy_identity(policy),
        "health_schema_sha256": file_sha(repo_root / "schemas/carrier-health-v2.schema.json"),
        "generation_call_budget": {"max_primary_calls": 1, "max_repair_calls": 1},
        "minimum_update_quota": None,
        "paired_arm_mechanism_parity_required": True,
    }
    contract_identity = sha256_bytes(canonical_json(projection))
    return {
        "schema": SCHEMA,
        "carrier_contract_identity_sha256": contract_identity,
        "identity_projection": projection,
        "health_component_identities": {
            key: roles[role] for key, role in HEALTH_ROLE_KEYS.items()
        },
        "component_file_count": len(components),
    }


def load_identity(path: Path, repo_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text())
    expected = build_identity(repo_root)
    if observed != expected:
        raise ValueError("Carrier Contract implementation identity drift")
    return observed


def write_identity(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = build_identity(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _successor_role_paths() -> dict[str, tuple[str, ...]]:
    return {
        role: (*paths, *SUCCESSOR_EXTRA_ROLE_PATHS.get(role, ()))
        for role, paths in ROLE_PATHS.items()
    }


def _successor_components(repo_root: Path) -> dict[str, str]:
    paths = sorted({path for values in _successor_role_paths().values() for path in values})
    return {path: file_sha(repo_root / path) for path in paths}


def successor_source_map(repo_root: Path) -> dict[str, Any]:
    components = _successor_components(repo_root)
    return {
        "schema": SOURCE_MAP_SCHEMA,
        "evidence_only": True,
        "execution_authority": False,
        "components": [
            {"slot": f"component_{index:03d}", "original_path": path, "sha256": digest}
            for index, (path, digest) in enumerate(sorted(components.items()))
        ],
    }


def build_successor_identity(repo_root: Path) -> dict[str, Any]:
    legacy_projection = build_identity(repo_root)["identity_projection"]
    components = _successor_components(repo_root)
    roles = _role_identities(components, _successor_role_paths())
    source_map = successor_source_map(repo_root)
    projection = {
        "contract_name": "Carrier Contract",
        "contract_version": 2,
        "component_slots": {
            row["slot"]: row["sha256"] for row in source_map["components"]
        },
        "source_map_identity_sha256": sha256_bytes(canonical_json(source_map)),
        "role_identities": roles,
        "carrier_generator_identities": {
            name: roles["generator"] for name in ("text_memory", "skill_bundle", "agent_system")
        },
        "scope_policy_identity_sha256": sha256_bytes(legacy_projection["scope_policy_id"].encode()),
        "health_policy_identity_sha256": legacy_projection["health_policy_identity_sha256"],
        "health_schema_sha256": legacy_projection["health_schema_sha256"],
        "generation_call_budget": {"max_primary_calls": 1, "max_repair_calls": 1},
        "minimum_update_quota": None,
        "paired_arm_mechanism_parity_required": True,
    }
    return {
        "schema": SUCCESSOR_SCHEMA,
        "evidence_only": True,
        "execution_authority": False,
        "carrier_contract_identity_sha256": sha256_bytes(canonical_json(projection)),
        "identity_projection": projection,
        "health_component_identities": {key: roles[role] for key, role in HEALTH_ROLE_KEYS.items()},
        "component_file_count": len(components),
    }


def load_successor_identity(identity_path: Path, source_map_path: Path, repo_root: Path) -> dict[str, Any]:
    identity = json.loads(identity_path.read_text())
    source_map = json.loads(source_map_path.read_text())
    if source_map != successor_source_map(repo_root):
        raise ValueError("Carrier source map drift")
    if identity != build_successor_identity(repo_root):
        raise ValueError("Carrier successor implementation identity drift")
    return identity


def write_successor_identity(identity_path: Path, source_map_path: Path, repo_root: Path) -> dict[str, Any]:
    source_map = successor_source_map(repo_root)
    identity = build_successor_identity(repo_root)
    for path, payload in ((source_map_path, source_map), (identity_path, identity)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return identity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--successor", action="store_true")
    ap.add_argument("--source-map", type=Path)
    args = ap.parse_args()
    repo_root = args.repo.resolve()
    if args.successor:
        if args.source_map is None:
            raise ValueError("successor identity requires --source-map")
        source_map_path = args.source_map.resolve()
        if args.validate:
            payload = load_successor_identity(args.output.resolve(), source_map_path, repo_root)
        else:
            payload = write_successor_identity(args.output.resolve(), source_map_path, repo_root)
    elif args.validate:
        payload = load_identity(args.output.resolve(), repo_root)
    else:
        payload = write_identity(args.output.resolve(), repo_root)
    print(json.dumps({
        "status": "PASS",
        "carrier_contract_identity_sha256": payload["carrier_contract_identity_sha256"],
        "component_file_count": payload["component_file_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
