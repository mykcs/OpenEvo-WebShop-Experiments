#!/usr/bin/env python3
"""Build a fail-closed H1.30 task-identity heterogeneity diagnostic manifest.

The diagnostic replays two predeclared identity groups under one fresh rollout
contract. It is train-only and never authorizes H1.30 formal evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.build_h1_30_training_manifest import validate_manifest
    from scripts.openevo_webshop.webshop_task_manifest import canonical_bytes
except ModuleNotFoundError:
    from build_h1_30_training_manifest import validate_manifest  # type: ignore
    from webshop_task_manifest import canonical_bytes  # type: ignore


SCHEMA = "openevo.webshop-h1.30-training-manifest.v1"
DIAGNOSTIC_SCHEMA = "openevo.webshop-h1.30-task-identity-diagnostic.v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def index_tasks(document: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    rows = document.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"manifest field is not a list: {key}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise ValueError(f"invalid task row in {key}")
        task_id = row["task_id"]
        if task_id in result:
            raise ValueError(f"duplicate task identity: {task_id}")
        result[task_id] = row
    return result


def diagnostic_task(row: dict[str, Any], sample_index: int) -> dict[str, Any]:
    task_id = str(row["task_id"])
    goal_idx = row.get("goal_idx")
    description = str(row.get("task_description") or "").strip()
    if not isinstance(goal_idx, int) or not description:
        raise ValueError(f"task row lacks identity/description: {task_id}")
    return {
        "sample_index": sample_index,
        "task_id": task_id,
        "goal_idx": goal_idx,
        "split": "h1.30-training",
        "task_description": description,
        "task_description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest(),
    }


def build(config: dict[str, Any], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if config.get("schema") != DIAGNOSTIC_SCHEMA:
        raise ValueError("diagnostic config schema mismatch")
    if config.get("status") != "preregistered-not-authorized":
        raise ValueError("diagnostic config must remain preregistered-not-authorized")
    if config.get("formal_task_consumption_allowed") is not False:
        raise ValueError("diagnostic config must fail closed")

    groups = config.get("task_groups")
    if not isinstance(groups, dict):
        raise ValueError("diagnostic config has no task_groups")
    positive_ids = groups.get("prior_positive")
    nonpositive_ids = groups.get("v3_nonpositive")
    if not isinstance(positive_ids, list) or not isinstance(nonpositive_ids, list):
        raise ValueError("diagnostic task groups must be lists")
    if len(positive_ids) != 8 or len(nonpositive_ids) != 8:
        raise ValueError("diagnostic requires exactly 8 identities per group")
    if len(set(positive_ids + nonpositive_ids)) != 16:
        raise ValueError("diagnostic groups must be disjoint")

    h0_tasks = index_tasks(sources["h0_manifest"], "tasks")
    v3_train = index_tasks(sources["v3_training_manifest"], "train_tasks")
    v3_validation = index_tasks(sources["v3_training_manifest"], "validation_tasks")
    v3_evaluation = index_tasks(sources["v3_evaluation_manifest"], "tasks")
    taxonomy_ids = set(sources["v3_taxonomy"].get("qualified_positive_task_ids", []))
    evaluation_ids = set(v3_evaluation)

    missing_positive = sorted(set(positive_ids) - set(h0_tasks))
    missing_nonpositive = sorted(set(nonpositive_ids) - set(v3_train))
    if missing_positive:
        raise ValueError(f"prior-positive identities missing from H0.1: {missing_positive}")
    if missing_nonpositive:
        raise ValueError(f"v3 identities missing from training manifest: {missing_nonpositive}")
    if set(nonpositive_ids) & taxonomy_ids:
        raise ValueError("v3_nonpositive group contains a qualified v3 positive")
    selected_ids = set(positive_ids + nonpositive_ids)
    if selected_ids & evaluation_ids:
        raise ValueError("diagnostic identity overlaps H1.30 evaluation")
    if selected_ids & set(v3_validation):
        raise ValueError("diagnostic identity overlaps H1.30 validation")

    train_rows = [
        diagnostic_task(h0_tasks[task_id], index)
        for index, task_id in enumerate(positive_ids)
    ] + [
        diagnostic_task(v3_train[task_id], index + 8)
        for index, task_id in enumerate(nonpositive_ids)
    ]
    validation_rows = [
        diagnostic_task(row, index) | {"split": "h1.30-validation"}
        for index, row in enumerate(v3_validation.values())
    ]
    controls = config["controls"]
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "draft-not-authorized",
        "formal_task_consumption_allowed": False,
        "experiment": "H1.30-task-identity-diagnostic",
        "benchmark": "webshop",
        "arms": ["PRIOR_POSITIVE", "V3_NONPOSITIVE"],
        "source": {
            "diagnostic_config_sha256": hashlib.sha256(
                canonical_bytes(config)
            ).hexdigest(),
            "h0_manifest_sha256": sources["h0_manifest"].get("manifest_sha256"),
            "h0_manifest_file_sha256": sources["h0_manifest_file_sha256"],
            "v3_training_manifest_sha256": sources["v3_training_manifest"].get("manifest_sha256"),
            "v3_training_manifest_file_sha256": sources["v3_training_manifest_file_sha256"],
            "v3_evaluation_manifest_sha256": sources["v3_evaluation_manifest"].get("manifest_sha256"),
            "v3_evaluation_manifest_file_sha256": sources["v3_evaluation_manifest_file_sha256"],
            "v3_taxonomy_file_sha256": sources["v3_taxonomy_file_sha256"],
        },
        "controls": {
            "environment_seed": 2026,
            "model_id": controls["model_id"],
            "model_revision": controls["model_revision"],
            "temperature": controls["temperature"],
            "max_steps": controls["max_steps"],
            "max_new_tokens": controls["max_new_tokens"],
            "rollout_seeds": controls["rollout_seeds"],
            "prompt_contract": controls["prompt_contract"],
            "action_contract": controls["action_contract"],
            "evaluator": controls["evaluator"],
        },
        "diagnostic_groups": {
            "prior_positive": list(positive_ids),
            "v3_nonpositive": list(nonpositive_ids),
        },
        "training_contract": {
            "status": "draft-requires-explicit-scientific-freeze",
            "train_task_count": 16,
            "validation_task_count": 8,
            "method": "train-only native capability harvest; no adapter training",
            "evaluation_task_consumption": "forbidden",
            "teacher_data": "forbidden",
        },
        "train_tasks": train_rows,
        "validation_tasks": validation_rows,
        "evaluation_manifest_sha256": sources["v3_evaluation_manifest"].get("manifest_sha256"),
        "selection_policy": {
            "fixed_before_execution": True,
            "post_hoc_resampling": False,
            "evaluation_disjoint": True,
            "validation_disjoint": True,
        },
    }
    document["manifest_sha256"] = manifest_sha256(document)
    validate_manifest(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--h0-manifest", type=Path, required=True)
    parser.add_argument("--v3-training-manifest", type=Path, required=True)
    parser.add_argument("--v3-evaluation-manifest", type=Path, required=True)
    parser.add_argument("--v3-taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "h0_manifest": args.h0_manifest,
        "v3_training_manifest": args.v3_training_manifest,
        "v3_evaluation_manifest": args.v3_evaluation_manifest,
        "v3_taxonomy": args.v3_taxonomy,
    }
    sources = {key: load_json(path) for key, path in paths.items()}
    for key, path in paths.items():
        sources[f"{key}_file_sha256"] = file_sha256(path)
    document = build(load_json(args.config), sources)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"H1.30_TASK_IDENTITY_DIAGNOSTIC_MANIFEST={document['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
