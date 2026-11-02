#!/usr/bin/env python3
"""Resolve a formal experiment config onto Golden Runtime capability + assets.

This module is CPU/read-only infrastructure. It does not authorize execution,
allocate a GPU, mutate a runtime, consume a WebShop task, or inspect a formal
held-out panel. Scientific configuration remains authoritative; this resolver
only proves that the declared identities are supported by the execution
platform and emits a deterministic resolution receipt on stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_webshop import golden_runtime_registry as registry


SCHEMA = "openevo.experiment-resolution.v1"


class ExperimentResolutionError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentResolutionError(f"experiment JSON unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentResolutionError("experiment config must be a JSON object")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _required_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ExperimentResolutionError(f"experiment.{key} must be an object")
    return value


def _required_string(parent: dict[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ExperimentResolutionError(f"{context}.{key} must be a non-empty string")
    return value


def _validate_experiment_header(experiment: dict[str, Any]) -> tuple[str, str, str]:
    schema = _required_string(experiment, "schema", "experiment")
    if not schema.startswith("openevo.experiment-") or not schema.endswith(".v1"):
        raise ExperimentResolutionError(f"unsupported experiment schema: {schema}")
    experiment_id = _required_string(experiment, "experiment", "experiment")
    campaign_id = _required_string(experiment, "campaign_id", "experiment")
    return schema, experiment_id, campaign_id


def _validate_model_declaration(experiment: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    provenance = _required_object(experiment, "provenance")
    content_pin = _required_string(provenance, "base_model_content_pin", "experiment.provenance")
    if not _is_hex64(content_pin):
        raise ExperimentResolutionError("experiment.provenance.base_model_content_pin must be a 64-character lowercase hex digest")
    if content_pin != resolved["model_content_pin"]:
        raise ExperimentResolutionError(
            f"model content pin mismatch: experiment={content_pin} asset={resolved['model_content_pin']}"
        )
    model = registry.load_model(resolved["model_asset"])
    declared_files = provenance.get("model_files_sha256")
    checked_files: dict[str, str] = {}
    if declared_files is not None:
        if not isinstance(declared_files, dict) or not declared_files:
            raise ExperimentResolutionError("experiment.provenance.model_files_sha256 must be a non-empty object when present")
        asset_files = model["required_files_sha256"]
        for relative, declared_hash in declared_files.items():
            if relative not in asset_files:
                raise ExperimentResolutionError(f"experiment declares unknown model file: {relative}")
            if declared_hash != asset_files[relative]:
                raise ExperimentResolutionError(
                    f"model file hash mismatch {relative}: experiment={declared_hash} asset={asset_files[relative]}"
                )
            checked_files[relative] = declared_hash
    return {
        "asset_id": model["asset_id"],
        "hf_id": model["hf_id"],
        "revision": model.get("revision"),
        "content_pin": model["content_pin"],
        "declared_file_hashes_checked": checked_files,
    }


def _validate_dataset_declaration(experiment: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    benchmark = _required_object(experiment, "benchmark")
    if benchmark.get("environment") != "WebAgentTextEnv-v0":
        raise ExperimentResolutionError(
            f"unsupported benchmark environment: {benchmark.get('environment')!r}; expected WebAgentTextEnv-v0"
        )
    if benchmark.get("num_products") != 1000:
        raise ExperimentResolutionError(
            f"unsupported WebShop product count: {benchmark.get('num_products')!r}; expected 1000"
        )
    declared_files = benchmark.get("data_files")
    if not isinstance(declared_files, dict) or not declared_files:
        raise ExperimentResolutionError("experiment.benchmark.data_files must be a non-empty object")
    dataset = registry.load_dataset(resolved["dataset_asset"])
    asset_files = dataset["required_files_sha256"]
    mandatory = {"items_shuffle_1000.json", "items_ins_v2_1000.json"}
    missing = sorted(mandatory - declared_files.keys())
    if missing:
        raise ExperimentResolutionError(f"experiment benchmark is missing required WebShop data files: {missing}")
    checked: dict[str, str] = {}
    for relative, declared_hash in declared_files.items():
        if relative not in asset_files:
            raise ExperimentResolutionError(f"experiment declares unknown dataset file: {relative}")
        if declared_hash != asset_files[relative]:
            raise ExperimentResolutionError(
                f"dataset file hash mismatch {relative}: experiment={declared_hash} asset={asset_files[relative]}"
            )
        checked[relative] = declared_hash
    return {
        "asset_id": dataset["asset_id"],
        "benchmark_environment": benchmark["environment"],
        "num_products": benchmark["num_products"],
        "declared_file_hashes_checked": checked,
    }


def _validate_runtime_source(experiment: dict[str, Any], runtime: dict[str, Any], dataset_asset: str) -> dict[str, Any]:
    provenance = _required_object(experiment, "provenance")
    declared = _required_string(provenance, "upstream_openevo_sha", "experiment.provenance")
    evidence_rel = runtime["runtime_evidence"]["path"]
    evidence = registry._read(ROOT / evidence_rel)
    expected = evidence.get("sources", {}).get("openevo_revision")
    if not isinstance(expected, str) or not expected:
        raise ExperimentResolutionError("Golden Runtime evidence has no OpenEvo source revision")
    if declared != expected:
        raise ExperimentResolutionError(f"OpenEvo revision mismatch: experiment={declared} runtime={expected}")

    seed_repository = _required_string(provenance, "seed_repository", "experiment.provenance")
    seed_commit = _required_string(provenance, "seed_commit", "experiment.provenance")
    dataset = registry.load_dataset(dataset_asset)
    dataset_source = dataset.get("source")
    if not isinstance(dataset_source, dict):
        raise ExperimentResolutionError("dataset asset has no SEED source identity")
    expected_seed_repository = dataset_source.get("repository")
    expected_seed_commit = dataset_source.get("revision")
    if seed_repository != expected_seed_repository:
        raise ExperimentResolutionError(
            f"SEED repository mismatch: experiment={seed_repository} asset={expected_seed_repository}"
        )
    if seed_commit != expected_seed_commit:
        raise ExperimentResolutionError(
            f"SEED commit mismatch: experiment={seed_commit} asset={expected_seed_commit}"
        )
    return {
        "openevo_revision": expected,
        "seed_repository": expected_seed_repository,
        "seed_commit": expected_seed_commit,
    }


def _science_projection(experiment: dict[str, Any]) -> dict[str, Any]:
    interface = _required_object(experiment, "interface_contract")
    execution = _required_object(experiment, "execution_contract")
    schedule = _required_object(experiment, "schedule")
    parser_contract = _required_string(interface, "action_parser_contract", "experiment.interface_contract")
    prompt_contract = _required_string(interface, "prompt_contract", "experiment.interface_contract")
    rollout_runner = execution.get("rollout_runner")
    trainer = execution.get("native_trainer_entrypoint")
    if not isinstance(rollout_runner, dict) or not isinstance(trainer, dict):
        raise ExperimentResolutionError("execution_contract requires rollout_runner and native_trainer_entrypoint objects")
    for label, item in (("rollout_runner", rollout_runner), ("native_trainer_entrypoint", trainer)):
        _required_string(item, "path", f"experiment.execution_contract.{label}")
        digest = _required_string(item, "sha256", f"experiment.execution_contract.{label}")
        if not _is_hex64(digest):
            raise ExperimentResolutionError(f"experiment.execution_contract.{label}.sha256 must be 64 lowercase hex characters")
    schedule_path = _required_string(schedule, "path", "experiment.schedule")
    schedule_content_sha = _required_string(schedule, "content_sha256", "experiment.schedule")
    schedule_file_sha = _required_string(schedule, "file_sha256", "experiment.schedule")
    if not _is_hex64(schedule_content_sha) or not _is_hex64(schedule_file_sha):
        raise ExperimentResolutionError("experiment schedule hashes must be 64 lowercase hex characters")
    return {
        "prompt_contract": prompt_contract,
        "action_parser_contract": parser_contract,
        "rollout_runner": dict(rollout_runner),
        "native_trainer_entrypoint": dict(trainer),
        "schedule": {
            "path": schedule_path,
            "content_sha256": schedule_content_sha,
            "file_sha256": schedule_file_sha,
        },
    }


def resolve(experiment_path: Path, *, verify_local_assets: bool = False) -> dict[str, Any]:
    path = experiment_path.expanduser().resolve()
    experiment = _read(path)
    schema, experiment_id, campaign_id = _validate_experiment_header(experiment)
    try:
        base = registry.resolve_experiment(experiment)
        runtime = registry.load_runtime()
        if verify_local_assets:
            local = registry.validate_all(verify_local=True)
            if local["status"] != "PASS":
                raise ExperimentResolutionError(f"local Golden Runtime asset validation failed: {local['errors']}")
        model = _validate_model_declaration(experiment, base)
        dataset = _validate_dataset_declaration(experiment, base)
        source = _validate_runtime_source(experiment, runtime, base["dataset_asset"])
        science = _science_projection(experiment)
    except registry.RegistryError as exc:
        raise ExperimentResolutionError(str(exc)) from exc

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RESOLVED",
        "experiment": {
            "id": experiment_id,
            "campaign_id": campaign_id,
            "schema": schema,
            "config_path": str(path),
            "config_sha256": _sha256_file(path),
        },
        "runtime": {
            "runtime_family": runtime["runtime_family"],
            "runtime_evidence": runtime["runtime_evidence"],
        },
        "model": model,
        "dataset": dataset,
        "source": source,
        "science_contract": science,
        "authority_boundary": {
            "execution_authorized": False,
            "formal_task_consumption_allowed": False,
            "final_panel_accessed": False,
            "note": "resolution proves identity compatibility only; launch authority is checked separately",
        },
    }
    digest_payload = json.loads(json.dumps(payload))
    digest_payload["experiment"].pop("config_path", None)
    payload["resolution_sha256"] = _sha256_bytes(_canonical(digest_payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--verify-local-assets", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = resolve(args.experiment, verify_local_assets=args.verify_local_assets)
    except ExperimentResolutionError as exc:
        print(json.dumps({"schema": SCHEMA, "status": "FAIL", "errors": [str(exc)]}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
