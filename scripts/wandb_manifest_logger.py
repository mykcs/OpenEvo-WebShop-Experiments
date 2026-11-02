#!/usr/bin/env python3
"""Log an existing experiment manifest and evidence bundle to W&B.

Online is the default. Scientific execution and local evidence remain
canonical; if online initialization fails, the caller must explicitly retry
with ``--mode offline`` and retain the original probe error plus spool hash.
Tracking must never become a prerequisite for a GPU run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import wandb


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]", "-", value).strip("-")
    if not normalized:
        raise ValueError("run ID is empty after normalization")
    return normalized[:64]


def _metrics(manifest: dict[str, Any]) -> dict[str, int | float | bool]:
    result: dict[str, int | float | bool] = {}
    for group_name in ("outputs", "experiment"):
        group = manifest.get(group_name, {})
        for metric_group_name in ("metrics", "gate_evidence"):
            values = group.get(metric_group_name, {}) if isinstance(group, dict) else {}
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if isinstance(value, bool) or isinstance(value, (int, float)):
                    result[f"{metric_group_name}/{key}"] = value
    return result


def _config(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": manifest.get("run"),
        "server": manifest.get("server"),
        "hardware": manifest.get("hardware"),
        "code": manifest.get("code"),
        "model": manifest.get("model"),
        "data": manifest.get("data"),
        "experiment": manifest.get("experiment"),
        "tracking": manifest.get("tracking"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--parameter-evidence", type=Path, action="append", default=[])
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "openevo-experiments"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=("offline", "online", "disabled"), default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()

    manifest_path = args.run_manifest.resolve()
    manifest = _json(manifest_path)
    run_id = _run_id(args.run_id or manifest["run"]["run_id"])
    evidence_paths = [path.resolve() for path in [*args.evidence, *args.parameter_evidence]]
    artifact_paths = [path.resolve() for path in [manifest_path, *evidence_paths, *args.artifact]]
    missing = [str(path) for path in artifact_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing artifact files: " + ", ".join(missing))

    init_kwargs: dict[str, Any] = {
        "project": args.project,
        "id": run_id,
        "mode": args.mode,
        "config": _config(manifest),
    }
    if args.mode != "offline":
        init_kwargs["resume"] = "allow"
    if args.entity:
        init_kwargs["entity"] = args.entity
    if args.wandb_dir is not None:
        init_kwargs["dir"] = str(args.wandb_dir.resolve())

    with wandb.init(**init_kwargs) as run:
        metrics = _metrics(manifest)
        if metrics:
            run.log(metrics, step=0)
        run.summary.update(
            {
                "source_manifest": str(manifest_path),
                "source_status": manifest.get("run", {}).get("status"),
                "tracking_mode": args.mode,
                "artifact_file_count": len(artifact_paths),
            }
        )
        evidence_artifact = wandb.Artifact(
            name=f"{run_id}-evidence",
            type="experiment-evidence",
            metadata={"run_id": run_id, "source_manifest": str(manifest_path)},
        )
        for path in artifact_paths:
            evidence_artifact.add_file(str(path), name=path.name)
        run.log_artifact(evidence_artifact)
        receipt_dir = (args.wandb_dir or Path.cwd()).resolve() / "receipts"
        receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        entries = {
            name: {
                "digest": getattr(entry, "digest", None),
                "size": getattr(entry, "size", None),
            }
            for name, entry in evidence_artifact.manifest.entries.items()
        }
        (receipt_dir / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "schema": "openevo.wandb.offline_receipt.v1",
                    "status": "pass",
                    "mode": args.mode,
                    "project": args.project,
                    "run_id": run.id,
                    "artifact_name": evidence_artifact.name,
                    "artifact_type": evidence_artifact.type,
                    "entries": entries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mode": args.mode,
                    "project": args.project,
                    "run_id": run.id,
                    "wandb_dir": str((args.wandb_dir or Path.cwd()).resolve()),
                    "metric_count": len(metrics),
                    "artifact_file_count": len(artifact_paths),
                    "artifact_manifest_entry_count": len(entries),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
