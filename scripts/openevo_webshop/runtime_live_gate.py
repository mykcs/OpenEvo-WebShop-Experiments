#!/usr/bin/env python3
"""Recheck volatile launch facts after a reusable runtime qualification.

Stable software/model/source qualification may be reused by exact fingerprint.
This gate intentionally rechecks facts that can drift between launches: CUDA/GPU
visibility, cache readability, artifact/output writability and collision, storage
headroom, and the current experiment/run manifests. GPU ownership/idleness and
W&B online/offline selection remain delegated to ``scripts/oe`` immediately at
launch, where they are already double-sampled / epoch-cached respectively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.openevo_webshop import environment_doctor as envdoc  # noqa: E402


SCHEMA = "openevo.runtime-live-gate.v1"
QUALIFICATION_SCHEMA = "openevo.runtime-qualification.v1"


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_qualification(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"qualification receipt is unreadable: {path}: {exc}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != QUALIFICATION_SCHEMA:
        raise ValueError("qualification receipt has the wrong schema")
    if receipt.get("status") != "PASS":
        raise ValueError("qualification receipt is not PASS")
    return receipt


def check_output_collision(path: Path | None, *, resume: bool) -> dict[str, Any]:
    if path is None:
        return {"status": "skip", "errors": [], "reason": "no output root requested"}
    target = path.expanduser().resolve()
    if target.exists() and not resume:
        return {
            "status": "fail",
            "path": str(target),
            "exists": True,
            "errors": ["output root already exists; use a new run root or explicit resume"],
        }
    return {
        "status": "pass",
        "path": str(target),
        "exists": target.exists(),
        "resume": resume,
        "errors": [],
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    qualification_path = args.qualification_receipt.expanduser().resolve()
    qualification = load_qualification(qualification_path)
    runtime_python = str(args.runtime_python.expanduser()) if args.runtime_python else None
    checks: dict[str, Any] = {
        "qualification": {
            "status": "pass",
            "path": str(qualification_path),
            "sha256": sha256_file(qualification_path),
            "cache_key": qualification.get("cache_key"),
            "runtime_fingerprint_sha256": qualification.get("request", {}).get("runtime_fingerprint_sha256"),
            "reused": bool(qualification.get("reused", False)),
            "errors": [],
        },
        "cuda": envdoc.check_cuda(args.require_cuda, python_executable=runtime_python),
        "nvidia_smi": envdoc.check_nvidia_smi(args.require_cuda),
        "cache": envdoc.check_cache(args.cache_root),
        "artifacts": envdoc.check_write_target(args.artifact_root),
        "output_writable": envdoc.check_write_target(args.output_root),
        "storage": envdoc.check_storage(
            args.storage_root or args.artifact_root or args.output_root,
            args.minimum_free_bytes,
        ),
        "output_collision": check_output_collision(args.output_root, resume=args.resume),
        "experiment_config": envdoc.check_experiment_config(args.experiment_config),
        "run_manifest": envdoc.check_run_manifest(args.run_manifest),
        "gpu_idle": {
            "status": "delegated",
            "delegate": "scripts/oe double-sampled UUID gate",
            "errors": [],
        },
        "wandb": {
            "status": "delegated",
            "delegate": "scripts/oe execution-epoch circuit breaker",
            "errors": [],
        },
    }
    failures = [
        f"{name}: {error}"
        for name, value in checks.items()
        for error in value.get("errors", [])
    ]
    return {
        "schema": SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "observed_at_epoch": time.time(),
        "checks": checks,
        "failures": failures,
        "formal_task_consumption_allowed": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--minimum-free-bytes", type=int, default=envdoc.DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = run_gate(args)
    except (OSError, ValueError) as exc:
        print(f"runtime live gate: {exc}", file=sys.stderr)
        return 2
    if args.output:
        atomic_json(args.output.expanduser().resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"RUNTIME_LIVE_GATE: {receipt['status']}")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
