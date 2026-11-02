#!/usr/bin/env python3
"""Execute one bounded command while maintaining its canonical Run Manifest.

This wrapper is transport-neutral. A caller supplies a valid planned manifest;
the wrapper records the actual command, immutable Git identity, timestamps,
logs, output hashes, and exit status before validating the final receipt.
It intentionally does not infer scientific success from a zero exit code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_run_manifest import validate


SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OPENEVO_SELECTED_GPU_UUIDS",
    "OPENEVO_SCHEDULER_CELL_ID",
    "OPENEVO_WANDB_MODE",
    "WANDB_MODE",
    "WANDB_RUN_ID",
    "WANDB_RUN_GROUP",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def git_value(*argv: str) -> str | None:
    result = subprocess.run(
        ["git", *argv], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def gpu_identity() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    indices: list[int] = []
    uuids: list[str] = []
    names: list[str] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            indices.append(int(parts[0]))
        except ValueError:
            continue
        uuids.append(parts[1])
        names.append(parts[2])
    if not indices:
        return None
    return {
        "gpu_indices": indices,
        "gpu_uuids": uuids,
        "gpu_model": names[0] if len(set(names)) == 1 else ", ".join(dict.fromkeys(names)),
    }


def update_identity(manifest: dict[str, Any], command: list[str], env: dict[str, str]) -> None:
    manifest["run"]["timestamp"] = utc_now()
    manifest["run"]["status"] = "running"
    manifest["code"]["openevo_experiment_sha"] = git_value("rev-parse", "HEAD") or manifest["code"].get("openevo_experiment_sha")
    manifest["command"] = {"argv": command, "environment": {key: env[key] for key in SAFE_ENV_KEYS if key in env}}
    gpu = gpu_identity()
    if gpu is not None:
        manifest["hardware"].update(gpu)


def add_artifacts(manifest: dict[str, Any], paths: list[Path]) -> None:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        artifacts.append(
            {
                "logical_name": resolved.name,
                "sha256": sha256_file(resolved),
                "locations": [str(resolved)],
            }
        )
    manifest["outputs"]["artifacts"] = artifacts


def run(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Run Manifest must be a JSON object")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = os.environ.copy()
    update_identity(document, command, environment)
    atomic_write(manifest_path, document)

    started = time.monotonic()
    stdout_path = output_root / "stdout.log"
    stderr_path = output_root / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=args.cwd or REPO_ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    document["run"]["status"] = "pass" if process.returncode == 0 else "fail"
    document["resources"]["wall_time_s"] = time.monotonic() - started
    if args.peak_gpu_memory_mb is not None:
        document["resources"]["peak_gpu_memory_mb"] = args.peak_gpu_memory_mb
    document["outputs"]["metrics"]["exit_status"] = process.returncode
    document["outputs"]["metrics"]["wall_time_s"] = document["resources"]["wall_time_s"]
    add_artifacts(document, [stdout_path, stderr_path, *args.artifact])
    atomic_write(manifest_path, document)
    errors = validate(manifest_path)
    if errors:
        document["run"]["status"] = "blocked"
        document["experiment"]["gate_evidence"]["manifest_validation_errors"] = errors
        atomic_write(manifest_path, document)
        print("Run Manifest validation failed after execution:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    return process.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--peak-gpu-memory-mb", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"manifest runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
