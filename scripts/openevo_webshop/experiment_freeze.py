#!/usr/bin/env python3
"""Freeze a complete DRAFT Experiment Passport without execution side effects.

Freeze is a repository identity operation only: require a clean checkout,
reject incomplete scientific/execution identity, compute canonical fingerprints
and the payload hash, validate the FROZEN result, then atomically replace the
DRAFT file. It never registers, launches, allocates GPU, or touches live state.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop.experiment_create import PLAN_PATH, STATE_MODEL_POLICY
from scripts.openevo_webshop.experiment_passport import (
    HASH_FIELD,
    PassportError,
    iter_references,
    load_json,
    repository_path,
    validate_passport,
    with_fingerprints,
)

ROOT = Path(__file__).resolve().parents[2]


class ExperimentFreezeError(ValueError):
    pass

_REQUIRED_DOTTED_FIELDS = (
    "scientific_identity.research_question",
    "scientific_identity.hypothesis",
    "scientific_identity.treatment",
    "scientific_identity.human_intervention_policy",
    "scientific_identity.benchmark",
    "scientific_identity.split",
    "scientific_identity.claim_boundary",
    "scientific_identity.model.model_id",
    "scientific_identity.model.model_revision",
    "scientific_identity.model.dtype",
    "scientific_identity.harness.profile",
    "scientific_identity.harness.thinking",
    "scientific_identity.seed_policy.schedule_id",
    "scientific_identity.seed_policy.seed_base",
    "scientific_identity.seed_policy.rng_contract",
    "scientific_identity.sampling.temperature",
    "scientific_identity.sampling.top_p",
    "scientific_identity.sampling.top_k",
    "scientific_identity.sampling.do_sample",
    "scientific_identity.acceptance_policy.name",
    "scientific_identity.acceptance_policy.candidate_rule",
    "scientific_identity.acceptance_policy.credit_assignment",
    "scientific_identity.external_teacher.role",
    "scientific_identity.external_teacher.future_call_budget",
    "scientific_identity.final_panel_policy.policy",
    "scientific_identity.final_panel_policy.future_access_limit",
    "scientific_identity.stop_rule.max_rollouts",
    "scientific_identity.stop_rule.max_rounds",
)
_REQUIRED_DOTTED_FIELDS += (
    "budget_contract.rollout_limit",
    "budget_contract.round_limit",
    "budget_contract.rollouts_per_round",
    "budget_contract.horizon",
    "budget_contract.external_teacher_call_limit",
    "budget_contract.final_panel_access_limit",
    "budget_contract.accounting_rule",
    "initial_execution_identity.initial_experiment_sha",
    "initial_execution_identity.frozen_science_sha",
    "initial_execution_identity.upstream_openevo_sha",
    "initial_execution_identity.initial_ray_sha",
    "initial_execution_identity.experiment_repository",
    "initial_execution_identity.upstream_repository",
    "initial_execution_identity.python_version",
    "initial_execution_identity.initial_run_root",
    "initial_execution_identity.runtime_image_digest",
    "initial_execution_identity.runtime_digest_kind",
    "artifact_policy.state_model_policy",
    "artifact_policy.default_recovery_state",
    "artifact_policy.default_disclosure_state",
)

_DRAFT_MARKERS = (
    "TODO before freeze",
    "Draft placeholder:",
    "DRAFT identity is not execution authority",
)


def _get_dotted(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _walk_draft_markers(value: Any, path: str = "$") -> list[str]:
    blockers: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            blockers.extend(_walk_draft_markers(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            blockers.extend(_walk_draft_markers(item, f"{path}[{index}]"))
    elif isinstance(value, str) and any(marker in value for marker in _DRAFT_MARKERS):
        blockers.append(f"{path}: draft placeholder remains")
    return blockers


def freeze_blockers(payload: Mapping[str, Any], root: Path) -> list[str]:
    blockers = _walk_draft_markers(payload)
    for dotted in _REQUIRED_DOTTED_FIELDS:
        if _get_dotted(payload, dotted) is None:
            blockers.append(f"$.{dotted}: required identity is missing")
    for location, ref in iter_references(payload):
        if ref["kind"] == "unresolved":
            blockers.append(f"{location}: unresolved source remains")
    if payload["authority_paths"] == [PLAN_PATH]:
        blockers.append("$.authority_paths: replace implementation-plan placeholder")
    if (payload.get("artifact_policy") or {}).get("state_model_policy") != STATE_MODEL_POLICY:
        blockers.append("$.artifact_policy.state_model_policy: canonical recovery/disclosure policy is required")
    if not payload["provenance"]["evidence_paths"]:
        blockers.append("$.provenance.evidence_paths: durable evidence is required")
    if payload["provenance"]["unknown_fields"]:
        blockers.append("$.provenance.unknown_fields: new experiment freeze requires no unknown fields")
    for value in [*payload["authority_paths"], *payload["provenance"]["evidence_paths"]]:
        if not repository_path(root, value).is_file():
            blockers.append(f"repository evidence path does not exist: {value}")
    return sorted(set(blockers))


def clean_git_head(root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root, text=True, capture_output=True, check=False,
    )
    if status.returncode != 0:
        raise ExperimentFreezeError("git status failed; freeze requires a healthy Git checkout")
    if status.stdout.strip():
        raise ExperimentFreezeError("Git checkout is dirty; commit or stash DRAFT edits before freeze")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=False,
    )
    if head.returncode != 0 or not head.stdout.strip():
        raise ExperimentFreezeError("cannot resolve Git HEAD for freeze")
    return head.stdout.strip()


def _resolve_draft_path(root: Path, identity_or_path: str) -> Path:
    candidate = Path(identity_or_path)
    if "/" in identity_or_path or identity_or_path.endswith(".json"):
        path = (root / candidate).resolve()
    else:
        path = (root / "configs" / "experiment" / "passports" / identity_or_path / "passport.json").resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ExperimentFreezeError("Passport path must stay inside repository root") from exc
    return path


def freeze_draft(*, root: Path, identity_or_path: str) -> dict[str, Any]:
    root = root.resolve()
    pre_freeze_head = clean_git_head(root)
    path = _resolve_draft_path(root, identity_or_path)
    if not path.is_file():
        raise ExperimentFreezeError(f"DRAFT Passport not found: {path}")
    payload = load_json(path)
    if payload.get("status") == "FROZEN":
        raise ExperimentFreezeError("Passport is already FROZEN")
    if payload.get("status") != "DRAFT":
        raise ExperimentFreezeError("freeze accepts only DRAFT Passports")
    try:
        validate_passport(payload, root=root, require_frozen=False)
    except PassportError as exc:
        raise ExperimentFreezeError(f"DRAFT validation failed: {exc}") from exc
    blockers = freeze_blockers(payload, root)
    if blockers:
        raise ExperimentFreezeError("freeze blocked: " + "; ".join(blockers[:10]))
    frozen = dict(payload)
    frozen["status"] = "FROZEN"
    frozen = with_fingerprints(frozen)
    try:
        validate_passport(frozen, root=root, require_frozen=True)
    except PassportError as exc:
        raise ExperimentFreezeError(f"FROZEN validation failed: {exc}") from exc
    tmp = path.with_suffix(path.suffix + ".freeze-tmp")
    tmp.write_text(json.dumps(frozen, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return {
        "status": "FROZEN",
        "experiment_id": frozen["experiment_id"],
        "passport_path": str(path),
        "passport_sha256": frozen[HASH_FIELD],
        "protocol_fingerprint": frozen["scientific_identity"]["protocol_fingerprint"],
        "runtime_fingerprint": frozen["initial_execution_identity"]["runtime_fingerprint"],
        "pre_freeze_git_head": pre_freeze_head,
        "mutation": True,
        "gpu_allocation_performed": False,
        "execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identity_or_path")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        receipt = freeze_draft(root=args.repo_root, identity_or_path=args.identity_or_path)
    except (ExperimentFreezeError, PassportError, OSError, KeyError) as exc:
        print(f"oe experiments freeze: {exc}")
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
