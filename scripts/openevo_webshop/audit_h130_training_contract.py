#!/usr/bin/env python3
"""Fail-closed audit of the H1.30 SEED training-input contract.

This audit never launches a trainer or consumes a task.  It detects the two
known invalid route shapes: the upstream WebShop launcher generating a
Geometry3K placeholder parquet, and the environment sampling random goals
instead of consuming a manifest-bound goal list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def source_identity(source_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(source_root.resolve())}
    try:
        result["head"] = subprocess.check_output(
            ["git", "-c", f"safe.directory={source_root}", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        result["status"] = subprocess.check_output(
            ["git", "-c", f"safe.directory={source_root}", "-C", str(source_root), "status", "--short"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip() or "detached-clean-or-clean"
    except (OSError, subprocess.CalledProcessError) as exc:
        result["git_error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_source(source_root: Path) -> dict[str, Any]:
    prepare = source_root / "examples/data_preprocess/prepare.py"
    envs = source_root / "agent_system/environments/env_package/webshop/envs.py"
    launcher = source_root / "examples/seed_trainer/_common/webshop.sh"
    files = {"prepare": prepare, "envs": envs, "launcher": launcher}
    contents: dict[str, str] = {}
    file_identity: dict[str, Any] = {}
    missing: list[str] = []
    for name, path in files.items():
        if not path.is_file():
            missing.append(f"missing source file: {path}")
            continue
        contents[name] = path.read_text(encoding="utf-8")
        file_identity[name] = {"path": str(path), "sha256": sha256_file(path)}

    findings = list(missing)
    placeholder_generator = False
    if "geometry3k" in contents.get("prepare", "") or "hiyouga/geometry3k" in contents.get("prepare", ""):
        placeholder_generator = True
        findings.append("upstream prepare.py generates Geometry3K-shaped placeholder parquet, not WebShop task rows")
    if "prepare.py" in contents.get("launcher", "") and "--mode text" in contents.get("launcher", ""):
        findings.append("WebShop launcher invokes the generic placeholder data generator")

    random_goal_sampling = "range(500, len(goals))" in contents.get("envs", "")
    manifest_binding = any(
        marker in contents.get("envs", "")
        for marker in ("task_manifest", "goal_indices", "train_goal_indices")
    )
    if random_goal_sampling and not manifest_binding:
        findings.append("WebShop environment samples random train goals and has no manifest-bound goal-index input")

    return {
        "files": file_identity,
        "placeholder_generator": placeholder_generator,
        "random_goal_sampling": random_goal_sampling,
        "manifest_binding_marker": manifest_binding,
        "findings": findings,
    }


def audit_manifest_bound_overlay(
    manifest: dict[str, Any],
    source_root: Path,
    overlay_root: Path,
    data_root: Path,
) -> dict[str, Any]:
    """Audit the disposable manifest-bound route without authorizing science.

    The pinned upstream source is intentionally still audited separately and
    may remain invalid. This function proves that the exact overlay and
    parquet inputs used by the bounded trainer route are bound to the same
    training manifest and contain the goal-index injection markers.
    """

    failures: list[str] = []
    manifest_sha = manifest.get("manifest_sha256")
    train_indices = [row["goal_idx"] for row in manifest.get("train_tasks", [])]
    validation_indices = [row["goal_idx"] for row in manifest.get("validation_tasks", [])]
    overlay_root = overlay_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    receipt_path = overlay_root / "h130-overlay-receipt.json"
    data_receipt_path = data_root / "h1.30-seed-parquet-receipt.json"

    try:
        overlay_receipt = load_json(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        overlay_receipt = {}
        failures.append(f"overlay receipt is unreadable: {type(exc).__name__}: {exc}")
    try:
        data_receipt = load_json(data_receipt_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        data_receipt = {}
        failures.append(f"parquet receipt is unreadable: {type(exc).__name__}: {exc}")

    if overlay_receipt.get("schema") != "openevo.webshop-h1.30-seed-source-overlay.v1":
        failures.append("overlay receipt schema mismatch")
    if overlay_receipt.get("status") != "prepared-not-authorized":
        failures.append("overlay is not prepared-not-authorized")
    if overlay_receipt.get("formal_task_consumption_allowed") is not False:
        failures.append("overlay is not fail-closed")
    if overlay_receipt.get("source_commit") != source_identity(source_root).get("head"):
        failures.append("overlay source commit does not match pinned source")
    if overlay_receipt.get("training_manifest_sha256") != manifest_sha:
        failures.append("overlay training manifest digest mismatch")
    if overlay_receipt.get("train_goal_indices") != train_indices:
        failures.append("overlay train goal indices do not match manifest")
    if overlay_receipt.get("validation_goal_indices") != validation_indices:
        failures.append("overlay validation goal indices do not match manifest")

    overlay_files = {
        "envs": overlay_root / "agent_system/environments/env_package/webshop/envs.py",
        "env_manager": overlay_root / "agent_system/environments/env_manager.py",
        "config": overlay_root / "verl/trainer/config/ppo_trainer.yaml",
    }
    overlay_markers = {
        "envs": ("goal_indices", "self._goal_indices"),
        "env_manager": ("train_goal_indices", "val_goal_indices"),
        "config": ("train_goal_indices: null", "val_goal_indices: null"),
    }
    overlay_file_identity: dict[str, Any] = {}
    for name, path in overlay_files.items():
        if not path.is_file():
            failures.append(f"overlay file is missing: {path}")
            continue
        content = path.read_text(encoding="utf-8")
        overlay_file_identity[name] = {"path": str(path), "sha256": sha256_file(path)}
        for marker in overlay_markers[name]:
            if marker not in content:
                failures.append(f"overlay {name} is missing marker: {marker}")

    if data_receipt.get("schema") != "openevo.webshop-h1.30-seed-parquet.v1":
        failures.append("parquet receipt schema mismatch")
    if data_receipt.get("status") != "prepared-not-authorized":
        failures.append("parquet inputs are not prepared-not-authorized")
    if data_receipt.get("formal_task_consumption_allowed") is not False:
        failures.append("parquet inputs are not fail-closed")
    if data_receipt.get("manifest_sha256") != manifest_sha:
        failures.append("parquet manifest digest mismatch")
    if data_receipt.get("train_goal_indices") != train_indices:
        failures.append("parquet train goal indices do not match manifest")
    if data_receipt.get("validation_goal_indices") != validation_indices:
        failures.append("parquet validation goal indices do not match manifest")

    parquet_files: dict[str, Any] = {}
    for key, expected_rows, filename in (
        ("train", len(train_indices), "train.parquet"),
        ("validation", len(validation_indices), "val.parquet"),
    ):
        path = data_root / filename
        observed = data_receipt.get("files", {}).get(key, {})
        if not path.is_file():
            failures.append(f"parquet file is missing: {path}")
            continue
        observed_sha = sha256_file(path)
        if observed.get("sha256") != observed_sha:
            failures.append(f"{key} parquet sha256 mismatch")
        if observed.get("rows") != expected_rows:
            failures.append(f"{key} parquet row count mismatch")
        parquet_files[key] = {
            "path": str(path),
            "rows": observed.get("rows"),
            "sha256": observed_sha,
        }

    return {
        "schema": "openevo.webshop-h1.30-seed-overlay-training-contract.v1",
        "status": "pass" if not failures else "blocked",
        "release_gate": "non-formal-input-contract-pass" if not failures else "fail",
        "formal_task_consumption_allowed": False,
        "manifest_sha256": manifest_sha,
        "source": source_identity(source_root),
        "overlay": {
            "root": str(overlay_root),
            "receipt": str(receipt_path),
            "files": overlay_file_identity,
        },
        "parquet": {
            "root": str(data_root),
            "receipt": str(data_receipt_path),
            "files": parquet_files,
        },
        "failures": failures,
    }


def audit(manifest: dict[str, Any], source_root: Path, *, training_task_ids: list[str] | None = None) -> dict[str, Any]:
    failures: list[str] = []
    tasks = manifest.get("tasks", [])
    evaluation_ids = [str(task.get("task_id")) for task in tasks if isinstance(task, dict)]
    if manifest.get("schema") != "openevo.webshop-h1.30-matched-manifest.v1":
        failures.append("H1.30 manifest schema mismatch")
    if not manifest.get("manifest_sha256"):
        failures.append("H1.30 manifest has no manifest_sha256")
    if len(evaluation_ids) != len(set(evaluation_ids)):
        failures.append("H1.30 evaluation task identities are duplicated")

    if training_task_ids is None:
        failures.append("no explicit H1.30 training task identity contract was supplied")
    else:
        overlap = sorted(set(training_task_ids) & set(evaluation_ids))
        if overlap:
            failures.append(f"training/evaluation task overlap: {overlap}")
        if len(set(training_task_ids)) != len(training_task_ids):
            failures.append("H1.30 training task identities are duplicated")

    source = inspect_source(source_root)
    failures.extend(source["findings"])
    return {
        "schema": "openevo.webshop-h1.30-training-contract-audit.v1",
        "status": "pass" if not failures else "blocked",
        "release_gate": "pass" if not failures else "fail",
        "formal_task_consumption_allowed": False,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "evaluation_task_ids": evaluation_ids,
        "training_task_ids": training_task_ids,
        "seed_source": source_identity(source_root),
        "source_audit": source,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-source", type=Path, required=True)
    parser.add_argument("--training-task-id", action="append")
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        manifest = load_json(args.manifest)
        if (args.overlay_root is None) != (args.data_root is None):
            raise ValueError("--overlay-root and --data-root must be supplied together")
        if args.overlay_root is not None and args.data_root is not None:
            result = audit_manifest_bound_overlay(
                manifest,
                args.seed_source.resolve(),
                args.overlay_root,
                args.data_root,
            )
        else:
            result = audit(manifest, args.seed_source.resolve(), training_task_ids=args.training_task_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema": "openevo.webshop-h1.30-training-contract-audit.v1",
            "status": "blocked",
            "release_gate": "fail",
            "formal_task_consumption_allowed": False,
            "failures": [f"audit input error: {type(exc).__name__}: {exc}"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
