#!/usr/bin/env python3
"""Create a schema-complete DRAFT Experiment Passport without execution side effects.

This command only writes a new repository-local draft file. It never freezes an
identity, registers it, allocates a GPU, launches a controller, or authorizes a
formal experiment. All scientific and execution facts remain explicit TODOs until
an operator fills them and a later freeze gate validates them.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.experiment_passport import PassportError, SCHEMA_ID, validate_passport

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = "docs/experiment-tracking/EXPERIMENT_PASSPORT_REGISTRY_IMPLEMENTATION_PLAN_202609111800.md"
STATE_MODEL_POLICY = "docs/operations/publication/RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md"
DEFAULT_ARTIFACT_ROLES = [
    "passport", "run-manifest", "startup-receipt", "closeout-receipt",
    "evaluation-summary", "bounded-table",
]


class ExperimentCreateError(ValueError):
    pass


def _unresolved(reason: str) -> dict[str, Any]:
    return {
        "kind": "unresolved", "repository": None, "revision": None,
        "path": None, "sha256": None, "evidence_path": None, "reason": reason,
    }


def build_draft_passport(*, experiment_id: str, campaign_id: str,
                         created_at: str, parent_experiment_id: str | None = None) -> dict[str, Any]:
    todo = "Draft placeholder: replace with hash-bound authority before freeze."
    ref = lambda: _unresolved(todo)
    science: dict[str, Any] = {
        "research_question": None,
        "hypothesis": None,
        "treatment": None,
        "human_intervention_policy": None,
        "benchmark": None,
        "split": None,
        "claim_boundary": None,
        "what_changed": ["TODO before freeze: describe the scientific intervention in plain language."],
        "what_must_stay_fixed": ["TODO before freeze: enumerate the scientific controls that must remain fixed."],
        "task_manifest": ref(),
        "prompt_contract": ref(),
        "action_contract": ref(),
        "parser_contract": ref(),
        "evaluator_contract": ref(),
        "training_data_identity": ref(),
        "model": {"model_id": None, "model_revision": None, "tokenizer_revision": None,
                  "dtype": None, "asset": ref()},
        "harness": {"profile": None, "thinking": None, "source": ref()},
        "seed_policy": {"schedule_id": None, "seed_base": None, "schedule": ref(), "rng_contract": None},
        "sampling": {"temperature": None, "top_p": None, "top_k": None, "do_sample": None,
                     "max_new_tokens": None, "extra_settings": {}},
        "optimizer": {"name": None, "learning_rate": None, "batch_size": None,
                      "gradient_accumulation_steps": None, "epochs": None, "extra_settings": {}},
        "sd_lora": {"rank": None, "alpha": None, "target_modules": None,
                    "effective_rank_cap": None, "source": ref(), "extra_settings": {}},
        "acceptance_policy": {"name": None, "candidate_rule": None,
                              "credit_assignment": None, "source": ref()},
        "protocol_fingerprint": None,
        "external_teacher": {"role": None, "future_call_budget": None, "prior_analyses": ref()},
        "final_panel_policy": {"policy": None, "future_access_limit": None, "source": ref()},
        "stop_rule": {"max_rollouts": None, "max_rounds": None, "source": ref()},
    }
    execution: dict[str, Any] = {
        "initial_experiment_sha": None, "frozen_science_sha": None,
        "upstream_openevo_sha": None, "initial_ray_sha": None,
        "experiment_repository": "mykcs/openevo-experiment", "upstream_repository": None,
        "python_version": None, "initial_run_root": None,
        "runtime_asset_lock": ref(), "dependency_lock": ref(), "launcher": ref(),
        "release": ref(), "activation": ref(), "runtime_image_digest": None,
        "runtime_digest_kind": "UNKNOWN", "runtime_fingerprint": None,
    }
    return {
        "schema": SCHEMA_ID,
        "experiment_id": experiment_id,
        "status": "DRAFT",
        "created_at": created_at,
        "parent_experiment_id": parent_experiment_id,
        "campaign_id": campaign_id,
        "scientific_identity": science,
        "initial_execution_identity": execution,
        "budget_contract": {
            "rollout_limit": None, "round_limit": None, "rollouts_per_round": None,
            "horizon": None, "optimizer_step_limit": None,
            "external_teacher_call_limit": None, "final_panel_access_limit": None,
            "accounting_rule": None, "source": ref(),
        },
        "tracking_policy": {
            "role": "MIRROR_ONLY", "entity": None, "project": None,
            "mode_order": ["online", "offline", "disabled"],
            "approved_artifact_roles": list(DEFAULT_ARTIFACT_ROLES),
            "scientific_authority": False,
        },
        "artifact_policy": {
            "large_asset_provider": "TODO before freeze: choose durable artifact storage.",
            "assets": [], "wandb_is_sole_backup": False,
            "state_model_policy": STATE_MODEL_POLICY,
            "default_recovery_state": "SERVER_ONLY",
            "default_disclosure_state": "PRIVATE",
        },
        "resume_policy": {
            "policy": "MANUAL_REVIEW_ONLY", "automatic_gpu_start": False,
            "requires_receipt_chain": True,
            "reason": "DRAFT identity is not execution authority; freeze and current launch gates are required.",
        },
        "authority_paths": [PLAN_PATH],
        "provenance": {
            "status": "NEW_DECLARATION", "evidence_paths": [],
            "unknown_fields": ["scientific identity", "execution identity", "budget and artifact bindings"],
            "parent_lineage_note": "New declaration; parent relationship must be verified before freeze." if parent_experiment_id else None,
            "revision_naming_note": None,
        },
        "passport_payload_sha256": None,
    }


def create_draft(*, root: Path, experiment_id: str, campaign_id: str,
                 created_at: str, parent_experiment_id: str | None = None,
                 output: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = root.resolve()
    draft = build_draft_passport(
        experiment_id=experiment_id, campaign_id=campaign_id,
        created_at=created_at, parent_experiment_id=parent_experiment_id,
    )
    validate_passport(draft, root=root, require_frozen=False)
    destination = (output or root / "configs" / "experiment" / "passports" / experiment_id / "passport.json").resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ExperimentCreateError("draft output must stay inside the repository root") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(draft, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise ExperimentCreateError(f"refusing to overwrite existing Passport draft: {destination}") from exc
    return destination, draft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", help="Canonical YYYYMMDDHHMM-scientific-slug identity")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--successor-of", dest="parent_experiment_id")
    parser.add_argument("--created-at", default=None, help="Timezone-aware ISO-8601 timestamp; defaults to current UTC")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    try:
        path, draft = create_draft(
            root=args.repo_root, experiment_id=args.experiment_id,
            campaign_id=args.campaign_id, created_at=created_at,
            parent_experiment_id=args.parent_experiment_id, output=args.output,
        )
    except (ExperimentCreateError, PassportError, OSError) as exc:
        print(f"oe experiments create: {exc}")
        return 2
    print(json.dumps({
        "status": "DRAFT", "experiment_id": draft["experiment_id"],
        "passport_path": str(path), "mutation": True,
        "gpu_allocation_performed": False, "execution_authorized": False,
        "next": "fill the DRAFT fields; freezing is a separate fail-closed step",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
