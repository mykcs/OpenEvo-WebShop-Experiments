#!/usr/bin/env python3
"""Build/validate a pre-Round0 package of all deterministic carrier scope manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.carrier_scope_manifest_v1 import (
    POLICY_ID,
    build_manifest,
    canonical_sha,
)
from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import load_scope_manifest

SCHEMA = "openevo.carrier-scope-package.v1"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != rendered:
            raise ValueError(f"carrier scope package immutable file drift: {path}")
        return
    path.write_text(rendered)


def build_package(schedule_path: Path, output_dir: Path) -> dict[str, Any]:
    schedule_bytes = schedule_path.read_bytes()
    schedule_sha = hashlib.sha256(schedule_bytes).hexdigest()
    schedule = json.loads(schedule_bytes)
    rounds = (schedule.get("stage2") or {}).get("rounds") or []
    round_indices = [int(row.get("round_index", -1)) for row in rounds]
    if round_indices != list(range(len(rounds))):
        raise ValueError("scope package requires contiguous ordered Stage2 rounds")
    entries = []
    for round_index in round_indices:
        manifest = build_manifest(
            schedule,
            schedule_file_sha256=schedule_sha,
            round_index=round_index,
        )
        filename = f"round-{round_index:03d}.json"
        manifest_path = output_dir / filename
        write_json(manifest_path, manifest)
        entries.append({
            "round_index": round_index,
            "manifest_file": filename,
            "manifest_file_sha256": file_sha(manifest_path),
            "manifest_identity_sha256": manifest["manifest_identity_sha256"],
            "source_task_digest_sha256": manifest["source_task_digest_sha256"],
            "source_task_count": manifest["source_task_count"],
        })
    payload = {
        "schema": SCHEMA,
        "construction_policy_id": POLICY_ID,
        "schedule_file_sha256": schedule_sha,
        "round_count": len(entries),
        "rounds": entries,
    }
    payload["package_identity_sha256"] = canonical_sha(payload)
    return payload


def load_package(path: Path, *, expected_schedule_sha256: str | None = None,
                 expected_round_count: int | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA or payload.get("construction_policy_id") != POLICY_ID:
        raise ValueError("carrier scope package identity mismatch")
    if expected_schedule_sha256 is not None and payload.get("schedule_file_sha256") != expected_schedule_sha256:
        raise ValueError("carrier scope package schedule SHA mismatch")
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or int(payload.get("round_count", -1)) != len(rounds):
        raise ValueError("carrier scope package round count mismatch")
    if expected_round_count is not None and len(rounds) != int(expected_round_count):
        raise ValueError("carrier scope package expected round count mismatch")
    if [int(row.get("round_index", -1)) for row in rounds] != list(range(len(rounds))):
        raise ValueError("carrier scope package round ordering mismatch")
    expected_identity = str(payload.get("package_identity_sha256") or "")
    identity_payload = {k: v for k, v in payload.items() if k != "package_identity_sha256"}
    if expected_identity != canonical_sha(identity_payload):
        raise ValueError("carrier scope package content identity mismatch")
    for entry in rounds:
        manifest_path = path.parent / str(entry.get("manifest_file") or "")
        if not manifest_path.is_file() or file_sha(manifest_path) != entry.get("manifest_file_sha256"):
            raise ValueError("carrier scope package manifest file drift")
        manifest = load_scope_manifest(manifest_path, expected_round_index=int(entry["round_index"]))
        if manifest.get("schedule_file_sha256") != payload.get("schedule_file_sha256"):
            raise ValueError("carrier scope package per-round schedule SHA drift")
        for key in ("manifest_identity_sha256", "source_task_digest_sha256", "source_task_count"):
            if manifest.get(key) != entry.get(key):
                raise ValueError(f"carrier scope package manifest {key} drift")
    return payload


def manifest_path_for_round(package_path: Path, package: dict[str, Any], round_index: int) -> Path:
    entry = package["rounds"][int(round_index)]
    if int(entry["round_index"]) != int(round_index):
        raise ValueError("carrier scope package lookup drift")
    return package_path.parent / entry["manifest_file"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = build_package(args.schedule, args.output_dir)
    index = args.output_dir / "index.json"
    write_json(index, package)
    load_package(index, expected_schedule_sha256=package["schedule_file_sha256"], expected_round_count=package["round_count"])
    print(json.dumps({"status": "PASS", "package_identity_sha256": package["package_identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
