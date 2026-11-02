#!/usr/bin/env python3
"""Fail-closed registry validation for reusable OpenEvo/WebShop runtime + assets.

This module is deliberately infrastructure-only. It never mutates environments,
installs packages, builds images, downloads assets, allocates GPUs, or starts a
scientific rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATH = ROOT / "configs/runtime/runtime-families/openevo-webshop-v1.json"
BINDINGS_PATH = ROOT / "configs/execution/golden-runtime-bindings-v1.json"
MODEL_ROOT = ROOT / "configs/assets/models"
DATASET_ROOT = ROOT / "configs/assets/datasets"


class RegistryError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def load_runtime() -> dict[str, Any]:
    runtime = _read(RUNTIME_PATH)
    if runtime.get("schema") != "openevo.runtime-family.v1":
        raise RegistryError("invalid runtime-family schema")
    if runtime.get("runtime_family") != "openevo-webshop-v1":
        raise RegistryError("unexpected runtime-family identity")
    forbidden = {"model", "dataset", "assets"} & set(runtime)
    if forbidden:
        raise RegistryError(f"runtime family embeds experiment assets: {sorted(forbidden)}")
    contract = runtime.get("asset_contract")
    if not isinstance(contract, dict) or contract.get("model") != "external-model-asset-required" or contract.get("dataset") != "external-dataset-asset-required":
        raise RegistryError("runtime family does not require external model/dataset assets")
    guarantees = runtime.get("guarantees", {})
    if guarantees.get("runtime_pip_install_allowed") is not False:
        raise RegistryError("runtime family permits pip install")
    if guarantees.get("host_python_dependency_injection_allowed") is not False:
        raise RegistryError("runtime family permits host Python dependency injection")
    evidence = runtime.get("runtime_evidence", {})
    evidence_path = ROOT / str(evidence.get("path", ""))
    fingerprint_path = ROOT / str(evidence.get("fingerprint_path", ""))
    if not evidence_path.is_file() or not fingerprint_path.is_file():
        raise RegistryError("runtime evidence/fingerprint path is missing")
    observed = _read(evidence_path)
    if observed.get("runtime_id") != evidence.get("runtime_id") or observed.get("status") != "KNOWN_GOOD":
        raise RegistryError("runtime-family pointer does not match KNOWN_GOOD evidence")
    return runtime


def load_model(asset_id: str) -> dict[str, Any]:
    path = MODEL_ROOT / f"{asset_id}.json"
    value = _read(path)
    if value.get("schema") != "openevo.model-asset.v1" or value.get("asset_id") != asset_id or value.get("kind") != "model":
        raise RegistryError(f"invalid model asset: {asset_id}")
    if not isinstance(value.get("hf_id"), str) or not value["hf_id"]:
        raise RegistryError(f"model asset lacks HF identity: {asset_id}")
    revision = value.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision or revision.startswith("unknown")):
        raise RegistryError(f"model asset has non-authoritative revision placeholder: {asset_id}")
    if not _is_sha256(value.get("content_pin")):
        raise RegistryError(f"model asset lacks authoritative content pin: {asset_id}")
    required = value.get("required_files_sha256")
    if not isinstance(required, dict) or not required or any(not _is_sha256(v) for v in required.values()):
        raise RegistryError(f"model asset has incomplete file hashes: {asset_id}")
    return value


def load_dataset(asset_id: str) -> dict[str, Any]:
    path = DATASET_ROOT / f"{asset_id}.json"
    value = _read(path)
    if value.get("schema") != "openevo.dataset-asset.v1" or value.get("asset_id") != asset_id or value.get("kind") != "dataset":
        raise RegistryError(f"invalid dataset asset: {asset_id}")
    required = value.get("required_files_sha256")
    if not isinstance(required, dict) or not required or any(not _is_sha256(v) for v in required.values()):
        raise RegistryError(f"dataset asset has incomplete file hashes: {asset_id}")
    return value


def _declared_model(experiment: dict[str, Any]) -> tuple[str, str | None, str | None]:
    provenance = experiment.get("provenance")
    if not isinstance(provenance, dict):
        raise RegistryError("experiment has no provenance object")
    hf_id = provenance.get("base_model")
    if not isinstance(hf_id, str) or not hf_id:
        raise RegistryError("experiment has no declared base model")
    revision = provenance.get("base_model_revision")
    pin = provenance.get("base_model_content_pin")
    return hf_id, revision if isinstance(revision, str) else None, pin if isinstance(pin, str) else None


def resolve_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    runtime = load_runtime()
    bindings = _read(BINDINGS_PATH)
    if bindings.get("schema") != "openevo.golden-runtime-bindings.v1" or bindings.get("fallback_allowed") is not False:
        raise RegistryError("invalid or fallback-permitting binding registry")
    hf_id, revision, pin = _declared_model(experiment)
    matches = [rule for rule in bindings.get("rules", []) if isinstance(rule, dict) and rule.get("match", {}).get("base_model_hf_id") == hf_id]
    if len(matches) != 1:
        raise RegistryError(f"experiment model must match exactly one binding rule: {hf_id}")
    rule = matches[0]
    model = load_model(str(rule.get("model_asset")))
    dataset = load_dataset(str(rule.get("dataset_asset")))
    if model.get("hf_id") != hf_id:
        raise RegistryError("binding points to a different model HF identity")
    if revision and model.get("revision") != revision:
        raise RegistryError(f"model revision mismatch: experiment={revision} asset={model.get('revision')}")
    if pin and model.get("content_pin") != pin:
        raise RegistryError(f"model content pin mismatch: experiment={pin} asset={model.get('content_pin')}")
    return {
        "schema": "openevo.golden-runtime-resolution.v1",
        "runtime_family": runtime["runtime_family"],
        "model_asset": model["asset_id"],
        "dataset_asset": dataset["asset_id"],
        "model_hf_id": model["hf_id"],
        "model_revision": model.get("revision"),
        "model_content_pin": model["content_pin"],
    }


def verify_local_asset(asset: dict[str, Any]) -> list[str]:
    root = Path(str(asset.get("canonical_path", "")))
    errors: list[str] = []
    if not root.is_dir():
        return [f"asset path missing: {root}"]
    for relative, expected in asset.get("required_files_sha256", {}).items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing asset file: {path}")
        elif _sha(path) != expected:
            errors.append(f"asset hash mismatch: {path}")
    return errors


def validate_all(*, verify_local: bool = False) -> dict[str, Any]:
    runtime = load_runtime()
    bindings = _read(BINDINGS_PATH)
    errors: list[str] = []
    seen_models: set[str] = set()
    for rule in bindings.get("rules", []):
        try:
            model = load_model(str(rule.get("model_asset")))
            dataset = load_dataset(str(rule.get("dataset_asset")))
            if rule.get("match", {}).get("base_model_hf_id") != model.get("hf_id"):
                errors.append(f"binding/model HF identity mismatch for {model.get('asset_id')}")
            if model["asset_id"] in seen_models:
                errors.append(f"duplicate model binding: {model['asset_id']}")
            seen_models.add(model["asset_id"])
            if verify_local:
                errors.extend(verify_local_asset(model))
                errors.extend(verify_local_asset(dataset))
        except RegistryError as exc:
            errors.append(str(exc))
    if seen_models != {"qwen2.5-3b-instruct", "qwen2.5-7b-instruct"}:
        errors.append(f"expected exact 3B/7B model assets, got {sorted(seen_models)}")
    return {
        "schema": "openevo.golden-runtime-registry-validation.v1",
        "status": "PASS" if not errors else "FAIL",
        "runtime_family": runtime["runtime_family"],
        "model_assets": sorted(seen_models),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-local-assets", action="store_true")
    parser.add_argument("--experiment", type=Path)
    args = parser.parse_args()
    try:
        if args.experiment:
            result = resolve_experiment(_read(args.experiment))
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        result = validate_all(verify_local=args.verify_local_assets)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    except RegistryError as exc:
        print(json.dumps({"status": "FAIL", "errors": [str(exc)]}, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
