#!/usr/bin/env python3
"""Prospective Bounded OFF/ON adapter into the single Execution Runtime broker.

This is zero-formal integration code.  It maps science-ready rollout and post-rollout
channels onto the exact #525 workload profiles, then delegates placement to
scripts.openevo_runtime.resource_broker.  It does not activate or migrate the live run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.openevo_runtime.resource_broker import RayBrokerBinding, RayResourceBroker
from scripts.openevo_runtime.resource_records import build_resource_envelope, build_resource_request
from scripts.openevo_webshop import ceiling1_ray_placement_backend_202609042025 as ray_backend

CONTRACT_SCHEMA = "openevo.bounded-gdr.ray-capacity-resource-contract.20260916"
CONTRACT_REL = Path("configs/experiment/bounded-effective-state-ray-capacity-contract-20260916.json")
ARM_KEYS = {
    "BOUNDED_OFF": "off",
    "BOUNDED_EFFECTIVE_STATE_ON": "on",
}


@dataclass(frozen=True)
class BoundedBrokerRoute:
    workload: str
    request: Mapping[str, Any]
    broker: RayResourceBroker


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def channel_for(*, kind: str, active_reflector_kind: str | None = None) -> str:
    if kind == "rollout":
        return "rollout"
    if kind in {"transition", "training"}:
        return "sd_lora"
    if kind == "reflector" and active_reflector_kind in {"text_memory", "skill_bundle", "agent_system"}:
        return str(active_reflector_kind)
    raise RuntimeError(f"unsupported Bounded GPU channel kind={kind} reflector={active_reflector_kind}")


def _resource_inputs(repo: Path) -> tuple[dict[str, Any], Path, Path, Path, dict[str, Any]]:
    contract_path = (repo / CONTRACT_REL).resolve()
    contract = _load(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise RuntimeError("Bounded #525 capacity contract schema drift")
    if contract.get("status") != "ZERO_FORMAL_QUALIFICATION_ONLY":
        raise RuntimeError("Bounded #525 capacity contract status drift")
    if contract.get("formal_activation_allowed") is not False:
        raise RuntimeError("prospective broker adapter cannot activate formal science")
    config_path = (repo / contract["base_ray_config_path"]).resolve()
    registry_path = (repo / contract["profile_registry_path"]).resolve()
    topology_registry_path = (repo / contract["qualified_topology_registry_path"]).resolve()
    config = _load(config_path)
    registry = ray_backend.load_profile_registry(registry_path, _sha(config_path))
    return contract, config_path, registry_path, topology_registry_path, registry


def workload_for(repo: Path, *, experiment_arm_id: str, channel: str) -> tuple[str, dict[str, Any], Path, Path]:
    arm_key = ARM_KEYS.get(experiment_arm_id)
    if arm_key is None:
        raise RuntimeError(f"unknown Bounded arm: {experiment_arm_id}")
    contract, config_path, registry_path, _topology_registry_path, registry = _resource_inputs(repo)
    mapping = contract.get("workload_map") or {}
    workload = ((mapping.get(arm_key) or {}).get(channel))
    if not workload:
        raise RuntimeError(f"#525 capacity contract has no workload for {arm_key}/{channel}")
    config = _load(config_path)
    profile = ray_backend.profile(config, workload, registry)
    return str(workload), profile, config_path, registry_path



def qualified_topology_registry_path(repo: Path) -> Path:
    contract, _config, _profiles, topology_registry, _registry = _resource_inputs(repo.resolve())
    del contract
    return topology_registry


def qualified_topology_identity_for(repo: Path, *, workload: str) -> str:
    topology_path = qualified_topology_registry_path(repo)
    document = _load(topology_path)
    matches = [entry for entry in document.get("entries", []) if workload in entry.get("workloads", [])]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one frozen qualified topology for workload {workload}")
    identity = matches[0].get("topology_id")
    if not isinstance(identity, str):
        raise RuntimeError("qualified topology identity missing")
    return identity

def build_broker_route(
    repo: Path,
    *,
    state_root: Path,
    experiment_arm_id: str,
    kind: str,
    operation_id: str,
    execution_generation_id: str,
    resource_policy_identity: str,
    qualified_topology_identity: str,
    gpu_count: int,
    preferred_gpu_indices: Sequence[int],
    active_reflector_kind: str | None = None,
    host_ram_mib: int = 0,
    cpu_slots: int = 0,
    disk_io_class: str = "ADVISORY_ONLY",
    fallback_topology_identity: str | None = None,
) -> BoundedBrokerRoute:
    repo = repo.resolve()
    channel = channel_for(kind=kind, active_reflector_kind=active_reflector_kind)
    workload, profile, config_path, registry_path = workload_for(
        repo,
        experiment_arm_id=experiment_arm_id,
        channel=channel,
    )
    minimum = int(profile["min_gpus"])
    maximum = int(profile["max_gpus"])
    if not minimum <= gpu_count <= maximum:
        raise RuntimeError(f"gpu_count={gpu_count} outside #525 profile range {minimum}..{maximum}")
    envelope = build_resource_envelope(
        gpu_count=gpu_count,
        vram_mib_per_gpu=int(profile["vram_request_mib"]),
        process_slots_per_gpu=1,
        compute_share_advisory=None,
        host_ram_mib=host_ram_mib,
        cpu_slots=cpu_slots,
        disk_io_class=disk_io_class,
    )
    request = build_resource_request(
        operation_id=operation_id,
        execution_generation_id=execution_generation_id,
        resource_policy_identity=resource_policy_identity,
        qualified_topology_identity=qualified_topology_identity,
        envelope=envelope,
        placement_mode="GANG_ALL_OR_NOTHING" if gpu_count > 1 else "QUALIFIED_SUBSET",
        preferred_gpu_indices=preferred_gpu_indices,
        fallback_topology_identity=fallback_topology_identity,
    )
    broker = RayResourceBroker(
        RayBrokerBinding(
            config_path=config_path,
            state_root=state_root,
            profile_registry_path=registry_path,
            topology_registry_path=qualified_topology_registry_path(repo),
            workload=workload,
        )
    )
    return BoundedBrokerRoute(workload=workload, request=request, broker=broker)
