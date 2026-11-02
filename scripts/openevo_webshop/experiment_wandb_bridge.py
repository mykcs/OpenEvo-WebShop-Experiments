#!/usr/bin/env python3
"""Mirror Experiment Passport identity and bounded metrics to W&B.

Git/Passport/Run Manifest evidence stays authoritative. This module never
mutates a frozen Passport and never makes W&B a scientific dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    assert_no_credentials,
    load_passport,
)
from scripts.openevo_webshop.experiment_registry import (
    ROOT,
    RegistryError,
    load_bound_passport,
    load_current_registry,
    resolve_experiment,
    validate_registry,
)
from scripts.openevo_webshop.wandb_tracking import start_wandb_run

SCHEMA = "openevo.experiment-wandb-tracking-receipt.202609120230"
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
SECRET_FILENAME = re.compile(
    r"(?:^|[-_.])(api[-_]?key|password|secret|credentials?|token|\.netrc|\.env)(?:$|[-_.])",
    re.I,
)

STABLE_METRIC_AXES: dict[str, str] = {
    "eval/round_score": "system/round",
    "experience/success_rate": "system/rollouts_consumed",
    "system/sd_component_count": "system/round",
    "system/accepted_updates": "system/round",
    "train/loss": "system/optimizer_steps",
    "train/grad_norm": "system/optimizer_steps",
    "eval/final_score": "eval/final_eval_step",
}
STEP_METRICS = {
    "system/round",
    "system/rollouts_consumed",
    "system/optimizer_steps",
    "eval/final_eval_step",
}


class WandbBridgeError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_registered_passport(
    repo_root: Path, experiment_or_alias: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry, _ = load_current_registry(repo_root)
    errors = validate_registry(registry, repo_root)
    if errors:
        raise WandbBridgeError("Registry validation failed: " + "; ".join(errors))
    entry = resolve_experiment(registry, experiment_or_alias)
    passport = load_bound_passport(repo_root, entry)
    return entry, passport


def passport_wandb_config(
    entry: Mapping[str, Any], passport: Mapping[str, Any], *, fixture_kind: str = "observational_mirror"
) -> dict[str, Any]:
    science = passport["scientific_identity"]
    execution = passport["initial_execution_identity"]
    model = science["model"]
    seed = science["seed_policy"]
    config = {
        "experiment_id": entry["experiment_id"],
        "passport_sha256": entry["passport_sha256"],
        "campaign_id": passport["campaign_id"],
        "parent_experiment_id": passport["parent_experiment_id"],
        "protocol_fingerprint": science["protocol_fingerprint"],
        "runtime_fingerprint": execution["runtime_fingerprint"],
        "initial_experiment_sha": execution["initial_experiment_sha"],
        "frozen_science_sha": execution["frozen_science_sha"],
        "upstream_openevo_sha": execution["upstream_openevo_sha"],
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "harness_profile": science["harness"]["profile"],
        "seed_schedule_id": seed["schedule_id"],
        "seed_base": seed["seed_base"],
        "treatment": science["treatment"],
        "benchmark": science["benchmark"],
        "split": science["split"],
        "round_limit": passport["budget_contract"]["round_limit"],
        "rollout_limit": passport["budget_contract"]["rollout_limit"],
        "tracking_role": passport["tracking_policy"]["role"],
        "tracking_fixture_kind": fixture_kind,
        "tracking_scientific_authority": passport["tracking_policy"]["scientific_authority"],
    }
    assert_no_credentials(config)
    return config


def validate_metric_row(row: Mapping[str, Any]) -> None:
    unknown = set(row) - set(STABLE_METRIC_AXES) - STEP_METRICS
    if unknown:
        raise WandbBridgeError(f"unapproved metric keys: {sorted(unknown)}")
    for metric, step_metric in STABLE_METRIC_AXES.items():
        if metric in row and step_metric not in row:
            raise WandbBridgeError(f"metric {metric} requires explicit x-axis {step_metric}")
    for key, value in row.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise WandbBridgeError(f"metric {key} must be numeric")

def approved_artifact(
    passport: Mapping[str, Any], role: str, path: Path
) -> dict[str, Any]:
    approved = set(passport["tracking_policy"].get("approved_artifact_roles") or [])
    if role not in approved:
        raise WandbBridgeError(f"artifact role is not approved by Passport: {role}")
    path = path.resolve()
    if not path.is_file():
        raise WandbBridgeError(f"artifact is not a file: {path}")
    if SECRET_FILENAME.search(path.name):
        raise WandbBridgeError("credential-like artifact filename forbidden")
    size = path.stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        raise WandbBridgeError(
            f"artifact too large for default W&B upload: {size} > {MAX_ARTIFACT_BYTES}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise WandbBridgeError("binary artifact denied by bounded-default policy") from exc
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        value = text
    try:
        assert_no_credentials(value)
    except PassportError as exc:
        raise WandbBridgeError(str(exc)) from exc
    return {"role": role, "path": str(path), "sha256": _sha256(path), "bytes": size}

def mirror_passport(
    *, repo_root: Path, experiment_or_alias: str, mode: str,
    output_dir: Path, metric_rows: Sequence[Mapping[str, Any]] = (),
    artifacts: Sequence[tuple[str, Path]] = (), scientific_output: Path | None = None,
    run_name: str = "passport-mirror", run_id: str | None = None,
    fixture_kind: str = "observational_mirror",
) -> dict[str, Any]:
    if fixture_kind not in {"observational_mirror", "synthetic_smoke"}:
        raise WandbBridgeError(f"unsupported tracking fixture kind: {fixture_kind}")
    entry, passport = load_registered_passport(repo_root, experiment_or_alias)
    config = passport_wandb_config(entry, passport, fixture_kind=fixture_kind)
    tracking = passport["tracking_policy"]
    output_dir = output_dir.resolve()
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    scientific_before = _sha256(scientific_output) if scientific_output else None
    approved = [approved_artifact(passport, role, path) for role, path in artifacts]
    for row in metric_rows:
        validate_metric_row(row)
    tracker = start_wandb_run(
        mode=mode, project=str(tracking["project"]), entity=tracking.get("entity"),
        name=run_name, job_type="passport-mirror", config=config,
        tags=(f"experiment:{entry['experiment_id']}", "passport-mirror"),
        directory=output_dir, group=str(passport["campaign_id"]),
        notes="Experiment Passport observability mirror; never scientific authority.",
        metric_axes=STABLE_METRIC_AXES, run_id=run_id,
    )
    for row in metric_rows:
        tracker.log(dict(row))
    if tracker.run is not None:
        for item in approved:
            tracker.log_artifact_file(
                Path(item["path"]),
                name=f"{entry['experiment_id']}-{item['role']}-{item['sha256'][:12]}",
                artifact_type=f"experiment-{item['role']}",
                metadata={
                    "experiment_id": entry["experiment_id"],
                    "passport_sha256": entry["passport_sha256"],
                    "role": item["role"],
                    "sha256": item["sha256"],
                },
            )
        tracker.set_summary({
            "experiment_id": entry["experiment_id"],
            "passport_sha256": entry["passport_sha256"],
            "tracking_role": "MIRROR_ONLY",
        })
    metadata = tracker.metadata()
    tracker.finish()
    scientific_after = _sha256(scientific_output) if scientific_output else None
    if scientific_before != scientific_after:
        raise WandbBridgeError("tracking mutated the scientific output fixture")
    fallback_receipt = metadata.get("fallback_receipt")
    fallback_payload = None
    if isinstance(fallback_receipt, str) and Path(fallback_receipt).is_file():
        fallback_payload = json.loads(Path(fallback_receipt).read_text(encoding="utf-8"))
    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": entry["experiment_id"],
        "passport_sha256": entry["passport_sha256"],
        "requested_mode": mode,
        "effective_mode": metadata.get("mode", "disabled"),
        "project": str(tracking["project"]),
        "entity": tracking.get("entity"),
        "run_id": metadata.get("run_id"),
        "run_url": metadata.get("run_url"),
        "run_path": metadata.get("run_path"),
        "sync_dir": metadata.get("sync_dir"),
        "fallback_reason": metadata.get("fallback_reason"),
        "fallback_spool_sha256": fallback_payload.get("spool_sha256") if fallback_payload else None,
        "metric_axes": STABLE_METRIC_AXES,
        "metric_row_count": len(metric_rows),
        "approved_artifacts": approved,
        "scientific_output_sha256": scientific_after,
        "scientific_authority": False,
        "tracking_fixture_kind": fixture_kind,
        "passport_mutated": False,
    }
    assert_no_credentials(receipt)
    receipt_dir = output_dir / "receipts"
    receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{entry['experiment_id']}-{run_name}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": _sha256(receipt_path)}

def _parse_artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be ROLE=PATH")
    role, raw = value.split("=", 1)
    if not role or not raw:
        raise argparse.ArgumentTypeError("artifact must be ROLE=PATH")
    return role, Path(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--scientific-output", type=Path)
    parser.add_argument("--artifact", action="append", type=_parse_artifact, default=[])
    parser.add_argument("--run-name", default="passport-mirror")
    parser.add_argument("--run-id")
    parser.add_argument("--fixture-kind", choices=("observational_mirror", "synthetic_smoke"), default="observational_mirror")
    args = parser.parse_args(argv)
    rows: list[Mapping[str, Any]] = []
    if args.metrics_json:
        value = json.loads(args.metrics_json.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
            raise WandbBridgeError("metrics JSON must be an array of objects")
        rows = value
    try:
        receipt = mirror_passport(
            repo_root=args.repo_root.expanduser().resolve(),
            experiment_or_alias=args.experiment_id,
            mode=args.mode,
            output_dir=args.output_dir,
            metric_rows=rows,
            artifacts=args.artifact,
            scientific_output=args.scientific_output,
            run_name=args.run_name,
            run_id=args.run_id,
            fixture_kind=args.fixture_kind,
        )
    except (WandbBridgeError, RegistryError, PassportError, OSError, ValueError) as exc:
        print(f"experiment-wandb-bridge: {exc}")
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
