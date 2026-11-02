#!/usr/bin/env python3
"""Prepare a disposable manifest-bound SEED source overlay.

The pinned checkout is never modified.  A clean archive is extracted into a
new output directory, the exact H1.30 route patch is applied, and a receipt is
written beside the overlay.  This is a runtime preparation step only; it
does not launch Ray, vLLM, a trainer, or a formal task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(argv: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "openevo.webshop-h1.30-training-manifest.v1":
        raise ValueError("training manifest schema mismatch")
    if document.get("status") != "draft-not-authorized":
        raise ValueError("training manifest must remain draft-not-authorized")
    if document.get("formal_task_consumption_allowed") is not False:
        raise ValueError("training manifest is not fail-closed")
    if document.get("manifest_sha256") != expected_sha256:
        raise ValueError("training manifest digest mismatch")
    train = document.get("train_tasks")
    validation = document.get("validation_tasks")
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError("training manifest has no train/validation task arrays")
    task_indices = [row.get("goal_idx") for row in train + validation if isinstance(row, dict)]
    if len(task_indices) != len(set(task_indices)):
        raise ValueError("training manifest train/validation identities overlap")
    return document


def archive_checkout(source_root: Path, output_root: Path) -> None:
    archive = subprocess.Popen(
        ["git", "-C", str(source_root), "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert archive.stdout is not None
    extract = subprocess.run(
        ["tar", "-xf", "-", "-C", str(output_root)],
        stdin=archive.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    archive.stdout.close()
    archive_rc = archive.wait()
    if archive_rc != 0:
        raise RuntimeError("git archive failed: " + archive.stderr.read().decode(errors="replace"))
    if extract.returncode != 0:
        raise RuntimeError("tar extraction failed: " + extract.stderr)


def prepare(source_root: Path, manifest_path: Path, patch_path: Path, output_root: Path, expected_source_commit: str, expected_manifest_sha256: str) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    patch_path = patch_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise ValueError(f"refusing to overwrite existing overlay: {output_root}")
    observed_commit = run_text(["git", "-C", str(source_root), "rev-parse", "HEAD"])
    if observed_commit != expected_source_commit:
        raise ValueError(f"source commit mismatch: expected {expected_source_commit}, observed {observed_commit}")
    dirty = run_text(["git", "-C", str(source_root), "status", "--porcelain"])
    if dirty:
        raise ValueError("source checkout is dirty")
    manifest = load_manifest(manifest_path, expected_manifest_sha256)
    if not patch_path.is_file():
        raise FileNotFoundError(patch_path)

    output_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_root.mkdir(mode=0o700)
    try:
        archive_checkout(source_root, output_root)
        subprocess.run(
            ["patch", "-p1", "--batch", "--forward", "--input", str(patch_path)],
            cwd=output_root,
            check=True,
            capture_output=True,
            text=True,
        )
        changed = [
            output_root / "agent_system/environments/env_package/webshop/envs.py",
            output_root / "agent_system/environments/env_manager.py",
        ]
        subprocess.run(["python3", "-m", "py_compile", *(str(path) for path in changed)], check=True)
    except Exception:
        shutil.rmtree(output_root)
        raise

    receipt = {
        "schema": "openevo.webshop-h1.30-seed-source-overlay.v1",
        "status": "prepared-not-authorized",
        "formal_task_consumption_allowed": False,
        "source_root": str(source_root),
        "source_commit": observed_commit,
        "source_status": "detached-clean",
        "overlay_root": str(output_root),
        "patch": str(patch_path),
        "patch_sha256": sha256_file(patch_path),
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": expected_manifest_sha256,
        "train_goal_indices": [row["goal_idx"] for row in manifest["train_tasks"]],
        "validation_goal_indices": [row["goal_idx"] for row in manifest["validation_tasks"]],
        "gpu_workload_launched": False,
        "formal_next_gate": "validate one non-formal manifest-bound reset and trainer dry-run, then explicitly freeze the scientific compute contract",
    }
    receipt_path = output_root / "h130-overlay-receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path.chmod(0o600)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    receipt = prepare(
        args.source_root,
        args.manifest,
        args.patch,
        args.output_root,
        args.expected_source_commit,
        args.expected_manifest_sha256,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
