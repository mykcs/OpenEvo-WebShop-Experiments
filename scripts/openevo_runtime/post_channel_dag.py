"""Mechanical audit and frozen post-rollout DAG for Execution Runtime modernization.

This module is deliberately read-only.  It extracts the actual data bindings from an
exact formal-controller source snapshot, then freezes a prospective fan-out/join DAG.
The record does not activate parallel execution, qualify a resource topology, or grant
scientific authority.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    assert_no_credentials,
    canonical_bytes,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[2]
REV = "202609170650"
SCHEMA_ID = f"openevo.post-channel-dag.{REV}"
SCHEMA_PATH = ROOT / f"schemas/post-channel-dag-{REV}.schema.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
POST_CHANNELS = ("text_memory", "sd_lora", "skill_bundle", "agent_system")


class PostChannelDagError(ValueError):
    """The audited source or frozen DAG violates the fail-closed contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _digest(payload: Mapping[str, Any]) -> str:
    assert_no_credentials(payload)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    if isinstance(item, ast.AsyncFunctionDef):
                        raise PostChannelDagError(f"{class_name}.{method_name} unexpectedly became async")
                    return item
    raise PostChannelDagError(f"missing {class_name}.{method_name}")


def _self_call(call: ast.Call, method_name: str) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == method_name
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
    )


def _name(node: ast.AST | None) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _literal_str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _assignment_calls(method: ast.FunctionDef) -> list[tuple[list[str], ast.Call]]:
    rows: list[tuple[list[str], ast.Call]] = []
    for stmt in method.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            targets: list[str] = []
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    targets.append(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    targets.extend(elt.id for elt in target.elts if isinstance(elt, ast.Name))
            rows.append((targets, stmt.value))
    return rows


def _call_order(method: ast.FunctionDef) -> list[str]:
    order: list[str] = []
    for stmt in method.body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                order.append(node.func.attr)
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "write_once_json":
                order.append("write_once_json")
    return order


def audit_controller_source(source: str) -> dict[str, Any]:
    """Extract the actual same-state post-channel bindings from the formal controller."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PostChannelDagError("formal controller source does not parse") from exc
    method = _method(tree, "BoundedEffectiveStateController", "run_round")
    assignments = _assignment_calls(method)

    bindings: dict[str, dict[str, Any]] = {}
    for targets, call in assignments:
        if _self_call(call, "run_sd"):
            if not {"new_sd", "new_sd_id", "updated"}.issubset(set(targets)):
                raise PostChannelDagError("run_sd result binding drift")
            bindings["sd_lora"] = {
                "call": "run_sd",
                "prior_state_inputs": ["current_sd_adapter"],
                "round_inputs": ["attempts.jsonl"],
                "current_round_sibling_inputs": [],
                "output_binding": "new_sd",
            }
        if not _self_call(call, "reflect"):
            continue
        kind = _literal_str(call.args[2] if len(call.args) > 2 else None)
        if kind not in {"text_memory", "skill_bundle", "agent_system"}:
            continue
        positional_prior = _name(call.args[3] if len(call.args) > 3 else None)
        keywords = {kw.arg: _name(kw.value) for kw in call.keywords if kw.arg}
        output = next((name for name in targets if name.startswith("new_")), None)
        bindings[kind] = {
            "call": "reflect",
            "positional_prior_binding": positional_prior,
            "keyword_bindings": keywords,
            "current_round_sibling_inputs": sorted(
                name for name in ([positional_prior] + list(keywords.values())) if name and name.startswith("new_")
            ),
            "output_binding": output,
        }

    if set(bindings) != set(POST_CHANNELS):
        raise PostChannelDagError(f"post-channel set drift: {sorted(bindings)}")
    if bindings["text_memory"].get("positional_prior_binding") != "prior_mem":
        raise PostChannelDagError("text_memory no longer reads prior memory")
    if bindings["text_memory"].get("keyword_bindings") != {"sd_adapter": "prior_sd"}:
        raise PostChannelDagError("text_memory dependency drift")
    if bindings["skill_bundle"].get("positional_prior_binding") != "prior_skill":
        raise PostChannelDagError("skill_bundle no longer reads prior skill")
    if bindings["skill_bundle"].get("keyword_bindings") != {"sd_adapter": "prior_sd"}:
        raise PostChannelDagError("skill_bundle dependency drift")
    if bindings["agent_system"].get("positional_prior_binding") != "prior_agent":
        raise PostChannelDagError("agent_system no longer reads prior agent")
    if bindings["agent_system"].get("keyword_bindings") != {
        "memory": "prior_mem", "skill": "prior_skill", "sd_adapter": "prior_sd"
    }:
        raise PostChannelDagError("agent_system dependency drift")
    if any(row["current_round_sibling_inputs"] for row in bindings.values()):
        raise PostChannelDagError("current-round sibling output became a post-channel input")

    order = _call_order(method)
    required_order = ["run_rollout", "evidence", "reflect", "run_sd", "reflect", "reflect"]
    cursor = 0
    for item in order:
        if cursor < len(required_order) and item == required_order[cursor]:
            cursor += 1
    if cursor != len(required_order):
        raise PostChannelDagError("expected formal post-rollout call sequence is no longer present")

    source_segment = ast.get_source_segment(source, method) or ""
    if "state-after.json" not in source_segment or "ROUND_COMPLETE.json" not in source_segment:
        raise PostChannelDagError("atomic round compose/seal paths are no longer visible in run_round")
    if "self.apply_carrier_health_boundary" not in source_segment:
        raise PostChannelDagError("Carrier Health post-seal boundary is no longer visible in run_round")

    return {
        "post_channel_bindings": bindings,
        "observed_call_order": order,
        "all_post_channels_use_prior_state_not_sibling_outputs": True,
        "state_after_and_round_complete_written_after_post_channels": True,
        "carrier_health_invoked_after_round_seal": True,
    }


def audit_base_reflector_source(source: str) -> dict[str, Any]:
    """Verify that reflector identities bind evidence and all declared prior-state inputs."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PostChannelDagError("base controller source does not parse") from exc
    method = _method(tree, "Controller", "reflect")
    args = [arg.arg for arg in method.args.args]
    for required in ("r", "rr", "kind", "prior", "memory", "skill", "sd_adapter"):
        if required not in args:
            raise PostChannelDagError(f"reflect signature lost {required}")
    segment = ast.get_source_segment(source, method) or ""
    required_fragments = (
        "evidence/evidence.json",
        "reflector_attempt_identity",
        "prior,memory,skill,sd_adapter",
    )
    compact = segment.replace(" ", "").replace("\n", "")
    if "evidence/evidence.json" not in segment:
        raise PostChannelDagError("reflect no longer reads sealed round evidence")
    if "reflector_attempt_identity" not in segment:
        raise PostChannelDagError("reflect no longer binds an attempt identity")
    if "prior,memory,skill,sd_adapter" not in compact:
        raise PostChannelDagError("reflect attempt identity no longer binds declared prior inputs")
    return {
        "reflect_reads_round_evidence": True,
        "reflect_attempt_identity_binds_prior_memory_skill_sd_inputs": True,
        "required_fragments": list(required_fragments),
    }


def dag_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema", "source_execution_sha", "source_files", "audit", "nodes", "edges",
        "prospective_parallel_group", "join_contract", "activation", "authority_boundary",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise PostChannelDagError(f"missing DAG field: {missing[0]}")
    if payload["schema"] != SCHEMA_ID:
        raise PostChannelDagError(f"schema must equal {SCHEMA_ID}")
    if not isinstance(payload["source_execution_sha"], str) or not HEX40.fullmatch(payload["source_execution_sha"]):
        raise PostChannelDagError("source_execution_sha must be a 40-char Git SHA")
    for name, digest in payload["source_files"].items():
        if not isinstance(name, str) or not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise PostChannelDagError("source_files must map paths to SHA-256 values")
    edges = payload["edges"]
    if not isinstance(edges, list) or not edges:
        raise PostChannelDagError("edges must be a non-empty list")
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 3 or any(not isinstance(v, str) or not v for v in edge):
            raise PostChannelDagError("every edge must be [source, target, kind]")
    if payload["prospective_parallel_group"] != list(POST_CHANNELS):
        raise PostChannelDagError("prospective parallel group must be the four audited post channels")
    activation = payload["activation"]
    if activation != {
        "parallel_execution_activated": False,
        "live_execution_mutated": False,
        "scientific_semantics_changed": False,
    }:
        raise PostChannelDagError("DAG audit may not activate or mutate execution/science")
    boundary = payload["authority_boundary"]
    if boundary != {
        "scientific_authority_granted": False,
        "resource_topology_authority_granted": False,
        "scientific_retry_authorized": False,
    }:
        raise PostChannelDagError("DAG record may not grant science/resource/retry authority")
    join = payload["join_contract"]
    if join.get("required_terminal_nodes") != list(POST_CHANNELS):
        raise PostChannelDagError("join must require all four post-channel terminals")
    if join.get("atomic_state_commit") is not True or join.get("carrier_health_after_round_seal") is not True:
        raise PostChannelDagError("join/state/seal boundary drift")
    projection = {key: payload[key] for key in required}
    canonical_bytes(projection)
    return projection


def build_post_channel_dag(
    *, source_execution_sha: str, controller_source: str, base_controller_source: str,
    controller_path: str, base_controller_path: str,
) -> dict[str, Any]:
    audit = {
        "formal_controller": audit_controller_source(controller_source),
        "base_reflector": audit_base_reflector_source(base_controller_source),
        "historical_serial_order_is_not_a_data_dependency": True,
        "sd_lora_does_not_read_evidence_but_is_conservatively_released_after_evidence": True,
        "in_process_thread_parallelism_is_not_authorized": True,
    }
    source_files = {
        controller_path: sha256_text(controller_source),
        base_controller_path: sha256_text(base_controller_source),
    }
    nodes = {
        "rollout": {"reads": ["state_k", "task_schedule"], "writes": ["attempts.jsonl", "summary.json"]},
        "evidence": {"reads": ["attempts.jsonl", "prior_evidence_history"], "writes": ["evidence/evidence.json"]},
        "text_memory": {"reads": ["evidence/evidence.json", "prior_memory", "prior_sd"], "writes": ["text-memory/terminal-receipt.json"]},
        "sd_lora": {"reads": ["attempts.jsonl", "prior_sd"], "writes": ["sd-lora/terminal-receipt.json"]},
        "skill_bundle": {"reads": ["evidence/evidence.json", "prior_skill", "prior_sd"], "writes": ["skill-bundle/terminal-receipt.json"]},
        "agent_system": {"reads": ["evidence/evidence.json", "prior_agent", "prior_memory", "prior_skill", "prior_sd"], "writes": ["agent-system/terminal-receipt.json"]},
        "task_vector": {"reads": ["sd-lora/terminal-receipt.json", "prior_sd", "new_sd"], "writes": ["task-vector.json"], "scientific_effect": False},
        "retention": {"reads": ["task-vector.json"], "writes": ["checkpoint-retention.json"], "control_sequence_preserved": True},
        "compose_state": {"reads": ["text_memory", "sd_lora", "skill_bundle", "agent_system", "retention"], "writes": ["state-after.json"]},
        "round_seal": {"reads": ["state-after.json", "all_channel_terminals", "task-vector.json"], "writes": ["ROUND_COMPLETE.json"]},
        "carrier_health": {"reads": ["ROUND_COMPLETE.json", "state-after.json", "carrier_terminals"], "writes": ["carrier-health-ledger.json", "carrier-health-boundary.json"]},
    }
    edges = [
        ["rollout", "evidence", "DATA"],
        ["evidence", "text_memory", "DATA"],
        ["evidence", "skill_bundle", "DATA"],
        ["evidence", "agent_system", "DATA"],
        ["evidence", "sd_lora", "CONSERVATIVE_RELEASE_BARRIER"],
        ["text_memory", "compose_state", "TERMINAL_JOIN"],
        ["sd_lora", "task_vector", "DIAGNOSTIC"],
        ["task_vector", "retention", "CONTROL_SEQUENCE"],
        ["retention", "compose_state", "CONTROL_SEQUENCE"],
        ["sd_lora", "compose_state", "TERMINAL_JOIN"],
        ["skill_bundle", "compose_state", "TERMINAL_JOIN"],
        ["agent_system", "compose_state", "TERMINAL_JOIN"],
        ["compose_state", "round_seal", "ATOMIC_STATE_COMMIT"],
        ["round_seal", "carrier_health", "POST_SEAL_HEALTH"],
    ]
    payload = {
        "schema": SCHEMA_ID,
        "source_execution_sha": source_execution_sha,
        "source_files": source_files,
        "audit": audit,
        "nodes": nodes,
        "edges": edges,
        "prospective_parallel_group": list(POST_CHANNELS),
        "join_contract": {
            "required_terminal_nodes": list(POST_CHANNELS),
            "isolated_round_output_paths_required": True,
            "channel_scoped_activity_state_required": True,
            "atomic_state_commit": True,
            "retention_control_sequence_preserved": True,
            "carrier_health_after_round_seal": True,
        },
        "activation": {
            "parallel_execution_activated": False,
            "live_execution_mutated": False,
            "scientific_semantics_changed": False,
        },
        "authority_boundary": {
            "scientific_authority_granted": False,
            "resource_topology_authority_granted": False,
            "scientific_retry_authorized": False,
        },
    }
    projection = dag_projection(payload)
    result = {**payload, "dag_id": _digest(projection)}
    validate_post_channel_dag(result)
    return result


def validate_post_channel_dag(payload: Mapping[str, Any]) -> None:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        validate_schema(dict(payload), schema)
    except (OSError, json.JSONDecodeError, PassportError) as exc:
        raise PostChannelDagError(str(exc)) from exc
    projection = dag_projection(payload)
    if payload.get("dag_id") != _digest(projection):
        raise PostChannelDagError("dag_id does not match canonical projection")
