#!/usr/bin/env python3
"""Fail-closed audit for the two H1.30 method routes.

This is an identity gate, not a launcher.  It proves that an OpenEvo adapter
and a SEED checkpoint can be evaluated under the same frozen H1.30 contract;
it never starts a model or consumes a GPU task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(value: Any, expected: Any, label: str, failures: list[str]) -> None:
    if value != expected:
        failures.append(f"{label}: expected {expected!r}, observed {value!r}")


def audit(config: dict[str, Any], manifest: dict[str, Any], routes: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    if len(routes) != 2:
        failures.append(f"expected exactly two method routes, observed {len(routes)}")

    contract = config.get("frozen_comparison_contract", {})
    manifest_identity = manifest.get("manifest_sha256")
    if not manifest_identity:
        failures.append("H1.30 manifest has no manifest_sha256")

    expected_common = {
        "model_revision": contract.get("model_revision"),
        "prompt_contract": contract.get("prompt_contract"),
        "action_contract": contract.get("action_contract"),
        "evaluator": contract.get("evaluator"),
        "task_manifest_sha256": manifest_identity,
    }
    observed_arms: set[str] = set()
    route_summaries: list[dict[str, Any]] = []

    for route in routes:
        arm = str(route.get("arm", ""))
        observed_arms.add(arm)
        if route.get("status") != "audited":
            failures.append(f"{arm or '<missing arm>'} route status is not audited")
        evaluation = route.get("evaluation", {})
        for key, expected in expected_common.items():
            _require(evaluation.get(key), expected, f"{arm}.evaluation.{key}", failures)

        if arm == "OPEN_EVO":
            adapter = route.get("adapter", {})
            reload_receipt = adapter.get("fresh_reload", {})
            if reload_receipt.get("status") != "pass":
                failures.append("OPEN_EVO adapter fresh reload receipt is not PASS")
            for key in ("adapter_identity_digest", "state_weights_sha256", "peft_weights_sha256"):
                if not adapter.get(key):
                    failures.append(f"OPEN_EVO adapter.{key} is missing")
            if not adapter.get("source_training_manifest_sha256"):
                failures.append("OPEN_EVO source training manifest identity is missing")
            route_summaries.append({
                "arm": arm,
                "method": route.get("method"),
                "adapter_identity_digest": adapter.get("adapter_identity_digest"),
                "reload_status": reload_receipt.get("status"),
            })
        elif arm == "SEED":
            checkpoint = route.get("checkpoint", {})
            if checkpoint.get("reload_status") != "pass":
                failures.append("SEED checkpoint reload status is not PASS")
            for key in ("checkpoint_sha256", "reload_receipt_sha256", "source_revision"):
                if not checkpoint.get(key):
                    failures.append(f"SEED checkpoint.{key} is missing")
            route_summaries.append({
                "arm": arm,
                "method": route.get("method"),
                "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
                "reload_status": checkpoint.get("reload_status"),
            })
        else:
            failures.append(f"unsupported method arm: {arm!r}")

    for arm in ("OPEN_EVO", "SEED"):
        if arm not in observed_arms:
            failures.append(f"missing required method arm: {arm}")

    return {
        "schema": "openevo.webshop-h1.30-route-audit.v1",
        "status": "pass" if not failures else "blocked",
        "formal_task_consumption_allowed": False,
        "release_gate": "pass" if not failures else "fail",
        "manifest_sha256": manifest_identity,
        "required_arms": ["OPEN_EVO", "SEED"],
        "routes": route_summaries,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--route", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = audit(
            load_json(args.config),
            load_json(args.manifest),
            [load_json(path) for path in args.route],
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema": "openevo.webshop-h1.30-route-audit.v1",
            "status": "blocked",
            "formal_task_consumption_allowed": False,
            "release_gate": "fail",
            "failures": [f"audit input error: {type(exc).__name__}: {exc}"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
