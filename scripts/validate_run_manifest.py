#!/usr/bin/env python3
"""Validate the dependency-free JSON Run Manifest contract.

The JSON Schema is the interoperable definition. This checker is the CI-safe
minimum validator: it needs only Python's standard library and adds the
phase-specific PASS safeguards that are intentionally hard to encode in a
generic schema alone.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PHASES = {"phase_c", "phase_d", "phase_e", "phase_f", "phase_g", "phase_h"}
STATUSES = {"planned", "running", "pass", "fail", "blocked", "cancelled"}
SECRET_KEY = re.compile(r"(^|_)(api_key|credential|password|secret|token)(_|$)", re.I)
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
TRAINABLE_STATE_DIGEST_KIND = "trainable_tensor_values_v1"


def fail(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def object_at(data: Any, key: str, errors: list[str], parent: str = "$") -> dict[str, Any]:
    path = f"{parent}.{key}"
    value = data.get(key) if isinstance(data, dict) else None
    if not isinstance(value, dict):
        fail(errors, path, "must be an object")
        return {}
    return value


def require_keys(value: dict[str, Any], keys: tuple[str, ...], errors: list[str], path: str) -> None:
    for key in keys:
        if key not in value:
            fail(errors, path, f"missing required field '{key}'")


def reject_unknown_keys(value: dict[str, Any], keys: tuple[str, ...], errors: list[str], path: str) -> None:
    for key in value:
        if key not in keys:
            fail(errors, path, f"unknown field '{key}' is not allowed by the schema")


def scan_for_secrets(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if SECRET_KEY.search(key):
                fail(errors, child_path, "credential-like fields are forbidden in manifests")
            scan_for_secrets(child, child_path, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_for_secrets(child, f"{path}[{index}]", errors)


def is_json_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_json_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool)) or is_json_number(value)


def validate_nullable_string(
    value: Any, path: str, errors: list[str], *, non_empty: bool = False
) -> None:
    if value is None:
        return
    if not isinstance(value, str) or (non_empty and not value):
        suffix = " non-empty" if non_empty else ""
        fail(errors, path, f"must be null or a{suffix} string")


def validate_nullable_digest(value: Any, path: str, errors: list[str]) -> None:
    if value is not None and (
        not isinstance(value, str) or not SHA256.fullmatch(value)
    ):
        fail(errors, path, "must be null or a sha256 digest")


def validate_non_negative_number_or_null(
    value: Any, path: str, errors: list[str]
) -> None:
    if value is not None and (not is_json_number(value) or value < 0):
        fail(errors, path, "must be null or a non-negative number")


def validate_rfc3339(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str):
        fail(errors, path, "must be an RFC 3339 timestamp")
        return
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        fail(errors, path, "must be an RFC 3339 timestamp with a timezone")


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"$: cannot read JSON manifest: {exc}"]
    if not isinstance(data, dict):
        return ["$: manifest root must be an object"]

    required_sections = (
        "run",
        "server",
        "hardware",
        "code",
        "model",
        "data",
        "experiment",
        "command",
        "outputs",
        "resources",
        "tracking",
    )
    require_keys(data, required_sections, errors, "$")
    reject_unknown_keys(data, required_sections, errors, "$")
    scan_for_secrets(data, "$", errors)

    run = object_at(data, "run", errors)
    require_keys(run, ("run_id", "timestamp", "phase", "status"), errors, "$.run")
    reject_unknown_keys(run, ("run_id", "timestamp", "phase", "status"), errors, "$.run")
    if not isinstance(run.get("run_id"), str) or not run.get("run_id"):
        fail(errors, "$.run.run_id", "must be a non-empty string")
    validate_rfc3339(run.get("timestamp"), "$.run.timestamp", errors)
    if run.get("phase") not in PHASES:
        fail(errors, "$.run.phase", f"must be one of {sorted(PHASES)}")
    if run.get("status") not in STATUSES:
        fail(errors, "$.run.status", f"must be one of {sorted(STATUSES)}")

    server = object_at(data, "server", errors)
    require_keys(
        server,
        ("server_identity", "control_plane", "experiment_container", "image_tag", "image_digest"),
        errors,
        "$.server",
    )
    reject_unknown_keys(
        server,
        ("server_identity", "control_plane", "experiment_container", "image_tag", "image_digest"),
        errors,
        "$.server",
    )
    image_digest = server.get("image_digest")
    for key in ("server_identity", "control_plane", "experiment_container", "image_tag"):
        validate_nullable_string(server.get(key), f"$.server.{key}", errors)
    validate_nullable_digest(image_digest, "$.server.image_digest", errors)

    hardware = object_at(data, "hardware", errors)
    require_keys(hardware, ("gpu_indices", "gpu_uuids", "gpu_model"), errors, "$.hardware")
    reject_unknown_keys(hardware, ("gpu_indices", "gpu_uuids", "gpu_model"), errors, "$.hardware")
    gpu_indices = hardware.get("gpu_indices")
    if not isinstance(gpu_indices, list) or not all(
        is_json_integer(index) and index >= 0 for index in gpu_indices or []
    ):
        fail(errors, "$.hardware.gpu_indices", "must be a list of non-negative integers")
    elif len(gpu_indices) != len(set(gpu_indices)):
        fail(errors, "$.hardware.gpu_indices", "must not contain duplicate indices")
    gpu_uuids = hardware.get("gpu_uuids")
    if not isinstance(gpu_uuids, list) or not all(
        isinstance(uuid, str) and uuid for uuid in gpu_uuids or []
    ):
        fail(errors, "$.hardware.gpu_uuids", "must be a list of non-empty strings")
    elif len(gpu_uuids) != len(set(gpu_uuids)):
        fail(errors, "$.hardware.gpu_uuids", "must not contain duplicate UUIDs")
    validate_nullable_string(hardware.get("gpu_model"), "$.hardware.gpu_model", errors)

    code = object_at(data, "code", errors)
    require_keys(
        code,
        ("openevo_experiment_sha", "upstream_openevo_repo", "upstream_openevo_sha"),
        errors,
        "$.code",
    )
    reject_unknown_keys(
        code,
        ("openevo_experiment_sha", "upstream_openevo_repo", "upstream_openevo_sha"),
        errors,
        "$.code",
    )
    if not isinstance(code.get("openevo_experiment_sha"), str) or not GIT_SHA.fullmatch(
        code["openevo_experiment_sha"]
    ):
        fail(errors, "$.code.openevo_experiment_sha", "must be a 40-character Git SHA")
    elif code["openevo_experiment_sha"] == "0" * 40:
        fail(errors, "$.code.openevo_experiment_sha", "must not be the all-zero placeholder SHA")
    upstream_sha = code.get("upstream_openevo_sha")
    if upstream_sha is not None and (
        not isinstance(upstream_sha, str) or not GIT_SHA.fullmatch(upstream_sha)
    ):
        fail(errors, "$.code.upstream_openevo_sha", "must be null or a 40-character Git SHA")
    validate_nullable_string(
        code.get("upstream_openevo_repo"), "$.code.upstream_openevo_repo", errors
    )

    model = object_at(data, "model", errors)
    require_keys(
        model,
        ("model_id", "model_revision", "tokenizer_revision", "dtype", "adapter_identity"),
        errors,
        "$.model",
    )
    reject_unknown_keys(
        model,
        ("model_id", "model_revision", "tokenizer_revision", "dtype", "adapter_identity"),
        errors,
        "$.model",
    )
    for key in ("model_id", "model_revision", "tokenizer_revision", "dtype", "adapter_identity"):
        validate_nullable_string(model.get(key), f"$.model.{key}", errors)

    data_section = object_at(data, "data", errors)
    require_keys(
        data_section,
        ("benchmark", "split", "dataset_revision", "dataset_hash", "index_hash", "task_manifest_identity"),
        errors,
        "$.data",
    )
    reject_unknown_keys(
        data_section,
        ("benchmark", "split", "dataset_revision", "dataset_hash", "index_hash", "task_manifest_identity"),
        errors,
        "$.data",
    )
    for key in ("benchmark", "split", "dataset_revision", "task_manifest_identity"):
        validate_nullable_string(data_section.get(key), f"$.data.{key}", errors)
    for key in ("dataset_hash", "index_hash"):
        validate_nullable_digest(data_section.get(key), f"$.data.{key}", errors)

    experiment = object_at(data, "experiment", errors)
    experiment_keys = ("experiment_id", "passport_sha256", "method", "seed", "generation", "config_hash", "gate_evidence")
    require_keys(experiment, experiment_keys, errors, "$.experiment")
    reject_unknown_keys(experiment, experiment_keys, errors, "$.experiment")
    experiment_id = experiment.get("experiment_id")
    passport_sha = experiment.get("passport_sha256")
    validate_nullable_string(experiment_id, "$.experiment.experiment_id", errors, non_empty=True)
    if passport_sha is not None and (not isinstance(passport_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", passport_sha)):
        fail(errors, "$.experiment.passport_sha256", "must be null or a bare SHA-256")
    if (experiment_id is None) != (passport_sha is None):
        fail(errors, "$.experiment", "experiment_id and passport_sha256 must both be null or both be bound")
    if not isinstance(experiment.get("method"), str) or not experiment.get("method"):
        fail(errors, "$.experiment.method", "must be a non-empty string")
    if not is_json_integer(experiment.get("seed")) or experiment.get("seed", -1) < 0:
        fail(errors, "$.experiment.seed", "must be a non-negative integer")
    generation = experiment.get("generation")
    if generation is not None and (
        not is_json_integer(generation) or generation < 0
    ):
        fail(errors, "$.experiment.generation", "must be null or a non-negative integer")
    validate_nullable_digest(
        experiment.get("config_hash"), "$.experiment.config_hash", errors
    )
    evidence = experiment.get("gate_evidence")
    if not isinstance(evidence, dict):
        fail(errors, "$.experiment.gate_evidence", "must be an object")
        evidence = {}

    command = object_at(data, "command", errors)
    require_keys(command, ("argv", "environment"), errors, "$.command")
    reject_unknown_keys(command, ("argv", "environment"), errors, "$.command")
    if not isinstance(command.get("argv"), list) or not command.get("argv") or not all(
        isinstance(arg, str) and arg for arg in command.get("argv", [])
    ):
        fail(errors, "$.command.argv", "must be a non-empty argument vector")
    if not isinstance(command.get("environment"), dict):
        fail(errors, "$.command.environment", "must be an object")
    elif not all(is_json_scalar(value) for value in command["environment"].values()):
        fail(errors, "$.command.environment", "values must be scalar or null")

    outputs = object_at(data, "outputs", errors)
    require_keys(outputs, ("artifacts", "metrics"), errors, "$.outputs")
    reject_unknown_keys(outputs, ("artifacts", "metrics"), errors, "$.outputs")
    artifacts = outputs.get("artifacts")
    if not isinstance(artifacts, list):
        fail(errors, "$.outputs.artifacts", "must be a list")
        artifacts = []
    for index, artifact in enumerate(artifacts):
        item_path = f"$.outputs.artifacts[{index}]"
        if not isinstance(artifact, dict):
            fail(errors, item_path, "must be an object")
            continue
        require_keys(artifact, ("logical_name", "sha256", "locations"), errors, item_path)
        reject_unknown_keys(artifact, ("logical_name", "sha256", "locations"), errors, item_path)
        if not isinstance(artifact.get("logical_name"), str) or not artifact.get("logical_name"):
            fail(errors, f"{item_path}.logical_name", "must be a non-empty string")
        if not isinstance(artifact.get("sha256"), str) or not SHA256.fullmatch(artifact["sha256"]):
            fail(errors, f"{item_path}.sha256", "must be a sha256 digest")
        if (
            not isinstance(artifact.get("locations"), list)
            or not artifact.get("locations")
            or not all(isinstance(location, str) and location for location in artifact["locations"])
        ):
            fail(errors, f"{item_path}.locations", "must be a non-empty list; a path alone is not an identity")
    if not isinstance(outputs.get("metrics"), dict):
        fail(errors, "$.outputs.metrics", "must be an object")
    elif not all(is_json_scalar(value) for value in outputs["metrics"].values()):
        fail(errors, "$.outputs.metrics", "values must be scalar or null")

    resources = object_at(data, "resources", errors)
    require_keys(resources, ("peak_gpu_memory_mb", "wall_time_s"), errors, "$.resources")
    reject_unknown_keys(resources, ("peak_gpu_memory_mb", "wall_time_s"), errors, "$.resources")
    for key in ("peak_gpu_memory_mb", "wall_time_s"):
        validate_non_negative_number_or_null(
            resources.get(key), f"$.resources.{key}", errors
        )

    tracking = object_at(data, "tracking", errors)
    require_keys(tracking, ("mode", "wandb", "weave"), errors, "$.tracking")
    reject_unknown_keys(tracking, ("mode", "wandb", "weave"), errors, "$.tracking")
    if tracking.get("mode") not in {"disabled", "offline", "online"}:
        fail(errors, "$.tracking.mode", "must be disabled, offline, or online")
    for service in ("wandb", "weave"):
        reference = object_at(tracking, service, errors, "$.tracking")
        require_keys(reference, ("enabled", "entity", "project", "id"), errors, f"$.tracking.{service}")
        reject_unknown_keys(reference, ("enabled", "entity", "project", "id"), errors, f"$.tracking.{service}")
        if not isinstance(reference.get("enabled"), bool):
            fail(errors, f"$.tracking.{service}.enabled", "must be boolean")
        for key in ("entity", "project", "id"):
            validate_nullable_string(
                reference.get(key), f"$.tracking.{service}.{key}", errors
            )
        if tracking.get("mode") == "disabled" and reference.get("enabled"):
            fail(errors, f"$.tracking.{service}.enabled", "cannot be true when tracking.mode is disabled")

    if run.get("status") == "pass":
        phase = run.get("phase")

        # Phase C is a policy/fact receipt. It may be CPU-only and may record
        # unresolved image/GPU details as null/empty. Scientific execution
        # gates D-F require the immutable runtime and exactly one physical GPU.
        if phase in {"phase_d", "phase_e", "phase_f"}:
            if not image_digest:
                fail(errors, "$.server.image_digest", "is required for a Phase D-F PASS receipt")
            if len(hardware.get("gpu_indices", [])) != 1:
                fail(errors, "$.hardware.gpu_indices", "must record exactly one GPU for Phase D-F PASS receipts")
            if len(hardware.get("gpu_uuids", [])) != 1:
                fail(errors, "$.hardware.gpu_uuids", "must record exactly one physical GPU UUID for Phase D-F PASS receipts")

        if not artifacts:
            fail(errors, "$.outputs.artifacts", "must include at least one digest-identified output for a PASS receipt")

        if resources.get("wall_time_s") is None:
            fail(errors, "$.resources.wall_time_s", "is required for a PASS receipt")
        if phase != "phase_c" and resources.get("peak_gpu_memory_mb") is None:
            fail(errors, "$.resources.peak_gpu_memory_mb", "is required for non-Phase-C PASS receipts")

        if phase in {"phase_d", "phase_e", "phase_f"}:
            for key in ("model_id", "model_revision", "tokenizer_revision", "dtype"):
                if not model.get(key):
                    fail(errors, f"$.model.{key}", "is required for a Phase D-F PASS receipt")
            if not evidence.get("real_model_execution"):
                fail(errors, "$.experiment.gate_evidence.real_model_execution", "must be true for Phase D-F PASS")

        if phase == "phase_d" and not evidence.get("real_generation"):
            fail(errors, "$.experiment.gate_evidence.real_generation", "must be true for Phase D PASS")

        if phase == "phase_e":
            for key in ("upstream_openevo_repo", "upstream_openevo_sha"):
                if not code.get(key):
                    fail(errors, f"$.code.{key}", "is required for a Phase E PASS receipt")
            if not model.get("adapter_identity"):
                fail(errors, "$.model.adapter_identity", "is required for a Phase E PASS receipt")
            for key in ("adapter_loaded", "optimizer_initialized", "serialized", "reload_verified"):
                if not evidence.get(key):
                    fail(errors, f"$.experiment.gate_evidence.{key}", "must be true for Phase E PASS")

        if phase == "phase_f":
            for key in ("upstream_openevo_repo", "upstream_openevo_sha"):
                if not code.get(key):
                    fail(errors, f"$.code.{key}", "is required for a Phase F PASS receipt")
            if not model.get("adapter_identity"):
                fail(errors, "$.model.adapter_identity", "is required for a Phase F PASS receipt")
            for key in (
                "finite_loss",
                "finite_nonzero_gradient",
                "optimizer_step",
                "parameter_changed",
                "serialized",
                "reload_verified",
            ):
                if not evidence.get(key):
                    fail(errors, f"$.experiment.gate_evidence.{key}", "must be true for Phase F PASS")

            if evidence.get("state_digest_kind") != TRAINABLE_STATE_DIGEST_KIND:
                fail(
                    errors,
                    "$.experiment.gate_evidence.state_digest_kind",
                    f"must be '{TRAINABLE_STATE_DIGEST_KIND}' for Phase F PASS",
                )

            before = evidence.get("state_before_sha256")
            after = evidence.get("state_after_sha256")
            reloaded = evidence.get("state_reloaded_sha256")

            before_after_valid = all(
                isinstance(value, str) and SHA256.fullmatch(value) for value in (before, after)
            )
            if not before_after_valid:
                fail(
                    errors,
                    "$.experiment.gate_evidence.state_before_sha256/state_after_sha256",
                    "must both be sha256 digests for Phase F PASS",
                )
            elif before == after:
                fail(
                    errors,
                    "$.experiment.gate_evidence.state_before_sha256/state_after_sha256",
                    "must be distinct trainable-state digests for Phase F PASS",
                )

            if not (isinstance(reloaded, str) and SHA256.fullmatch(reloaded)):
                fail(
                    errors,
                    "$.experiment.gate_evidence.state_reloaded_sha256",
                    "must be a sha256 digest for Phase F PASS",
                )
            elif isinstance(after, str) and SHA256.fullmatch(after) and reloaded != after:
                fail(
                    errors,
                    "$.experiment.gate_evidence.state_reloaded_sha256",
                    "must equal state_after_sha256 after reload",
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON Run Manifest to validate")
    args = parser.parse_args()
    errors = validate(args.manifest)
    if errors:
        print("Run Manifest validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"Run Manifest valid: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
