#!/usr/bin/env python3
"""Reconcile complete or safely partial H1.38B method-control evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import math
from pathlib import Path
from typing import Any, Iterable


ARMS = ("base", "ordinary-lora", "sd-lora")
TREATMENTS = ARMS[1:]
RESAMPLES = 10_000
RESAMPLE_SEED = 202608200138
MANIFEST_SCHEMA = "openevo.webshop-h1.38b-method-control-eval-manifest.v1"
TRAINING_RECEIPT_SCHEMA = "openevo.webshop-h1.38b-training-receipt.v1"
QUALIFICATION_SCHEMA = "openevo.webshop-h1.38b-qualification.v1"
D8_RECORDS_SHA256 = "d206c20e4d08a9c1785a6af9c2edb510db4a478d5d459395537706a2d7d8191c"
D8_SELECTION_SHA256 = "b8c35b51aed3bf942cb2e45e7c9c0e68239d6efc27a3d7be8bc6a5ed54ff70c7"
PRODUCTION_CAMPAIGN_ID = "20260820-0129-h138b-method-control"
PRODUCTION_ACQUISITION_SEEDS = [2026090600, 2026090601]
PRODUCTION_RETENTION_SEEDS = [2026090610, 2026090611, 2026090612, 2026090613]
PRODUCTION_FRESH_SEEDS = [2026090620, 2026090621, 2026090622, 2026090623]
PRODUCTION_ATTEMPTS = 704


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    return (minimum is None or float(value) >= minimum) and (maximum is None or float(value) <= maximum)


def _validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("manifest schema mismatch")
    unsigned = dict(manifest)
    observed_hash = unsigned.pop("manifest_sha256", None)
    if observed_hash != _canonical_hash(unsigned):
        errors.append("manifest self-hash mismatch")
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    if source.get("d8_records_sha256") != D8_RECORDS_SHA256:
        errors.append("manifest D8 records identity mismatch")
    if source.get("d8_selection_sha256") != D8_SELECTION_SHA256:
        errors.append("manifest D8 selection identity mismatch")
    task_order = manifest.get("task_order")
    training_tasks = manifest.get("training_tasks")
    if not isinstance(task_order, list) or not isinstance(training_tasks, list):
        errors.append("manifest training task contract missing")
    elif [int(item.get("goal_idx", -1)) for item in training_tasks] != [int(value) for value in task_order]:
        errors.append("manifest training task order mismatch")
    return errors


def _validate_preregistered_contract(
    manifest: dict[str, Any], manifest_path: Path, prereg: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    evaluation = prereg.get("evaluation") if isinstance(prereg.get("evaluation"), dict) else {}
    if prereg.get("campaign_id") != PRODUCTION_CAMPAIGN_ID:
        errors.append("prereg campaign identity mismatch")
    if manifest.get("campaign_id") != prereg.get("campaign_id"):
        errors.append("manifest/prereg campaign mismatch")
    if _file_sha256(manifest_path) != evaluation.get("manifest_file_sha256"):
        errors.append("prereg manifest file SHA mismatch")
    if manifest.get("manifest_sha256") != evaluation.get("manifest_content_sha256"):
        errors.append("prereg manifest content SHA mismatch")
    training = manifest.get("training_tasks") if isinstance(manifest.get("training_tasks"), list) else []
    fresh = manifest.get("fresh_tasks") if isinstance(manifest.get("fresh_tasks"), list) else []
    train_ids = [item.get("goal_idx") for item in training if isinstance(item, dict)]
    fresh_ids = [item.get("goal_idx") for item in fresh if isinstance(item, dict)]
    if len(training) != 8 or len(train_ids) != 8 or len(set(train_ids)) != 8:
        errors.append("production training task cardinality mismatch")
    if len(fresh) != 48 or len(fresh_ids) != 48 or len(set(fresh_ids)) != 48:
        errors.append("production fresh task cardinality mismatch")
    if set(train_ids) & set(fresh_ids):
        errors.append("production train/fresh task overlap")
    expected_stages = {
        "acquisition": (list(TREATMENTS), PRODUCTION_ACQUISITION_SEEDS, 16, 32),
        "retention": (list(ARMS), PRODUCTION_RETENTION_SEEDS, 24, 96),
        "fresh": (list(ARMS), PRODUCTION_FRESH_SEEDS, 144, 576),
    }
    for stage, (arms, seeds, cells, attempts) in expected_stages.items():
        observed = manifest.get(stage) if isinstance(manifest.get(stage), dict) else {}
        registered = evaluation.get(stage) if isinstance(evaluation.get(stage), dict) else {}
        expected = {"arms": arms, "rollout_seeds": seeds, "cells": cells, "attempts": attempts}
        if any(observed.get(key) != value for key, value in expected.items()):
            errors.append(f"production {stage} contract mismatch")
        if any(registered.get(key) != value for key, value in expected.items()):
            errors.append(f"preregistered {stage} contract mismatch")
    if manifest.get("planned_attempts") != PRODUCTION_ATTEMPTS or evaluation.get("planned_attempts") != PRODUCTION_ATTEMPTS:
        errors.append("production planned attempt count mismatch")
    return errors


def _validate_training_receipts(
    manifest: dict[str, Any], paths: list[Path], *, include_missing: bool = False
) -> tuple[list[dict[str, Any]], list[str], float] | tuple[
    list[dict[str, Any]], list[str], list[tuple[str, int]], float
]:
    errors: list[str] = []
    receipts: dict[tuple[str, int], dict[str, Any]] = {}
    tasks = manifest.get("training_tasks") if isinstance(manifest.get("training_tasks"), list) else []
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    elapsed = 0.0
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"invalid training evidence {path}: {type(exc).__name__}")
            continue
        arm = str(payload.get("arm") or "")
        increment = payload.get("increment_index")
        key = (arm, int(increment)) if isinstance(increment, int) and not isinstance(increment, bool) else (arm, -1)
        if key in receipts:
            errors.append(f"duplicate training evidence {key}")
            continue
        receipts[key] = payload
        if payload.get("schema") != TRAINING_RECEIPT_SCHEMA or payload.get("status") != "pass":
            errors.append(f"training evidence schema/status mismatch {key}")
            continue
        if arm not in TREATMENTS or not 0 <= key[1] < len(tasks):
            errors.append(f"unexpected training evidence {key}")
            continue
        task = tasks[key[1]]
        if payload.get("task_id") != task.get("task_id") or payload.get("goal_idx") != task.get("goal_idx"):
            errors.append(f"training identity mismatch {key}")
        if (
            payload.get("parent_records_sha256") != source.get("d8_records_sha256")
            or payload.get("parent_selection_sha256") != source.get("d8_selection_sha256")
        ):
            errors.append(f"training D8 source mismatch {key}")
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        proof = payload.get("state_proof") if isinstance(payload.get("state_proof"), dict) else {}
        fallback = payload.get("no_base_fallback") if isinstance(payload.get("no_base_fallback"), dict) else {}
        numeric = [metrics.get("training_loss")]
        if arm == "ordinary-lora":
            numeric.extend((metrics.get("maximum_gradient_norm"), proof.get("update_l2_norm")))
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in numeric
        ):
            errors.append(f"non-finite training metrics {key}")
        if int(metrics.get("optimizer_steps") or 0) < 1 or (
            arm == "ordinary-lora" and float(proof.get("update_l2_norm") or 0.0) <= 0.0
        ):
            errors.append(f"empty optimizer/update proof {key}")
        if arm == "ordinary-lora":
            before = proof.get("before_trainable_tensor_sha256")
            after = proof.get("after_trainable_tensor_sha256")
            reloaded = proof.get("reloaded_trainable_tensor_sha256")
            if not all(isinstance(value, str) and len(value) == 64 for value in (before, after, reloaded)):
                errors.append(f"malformed tensor proof {key}")
            elif before == after or after != reloaded:
                errors.append(f"state change/reload mismatch {key}")
        else:
            reloaded = proof.get("reloaded_trainable_tensor_sha256")
            explicit_change = proof.get("changed_from_prior") is True
            legacy_change = (
                proof.get("before_trainable_tensor_sha256")
                != proof.get("after_trainable_tensor_sha256")
                == reloaded
            )
            if not (explicit_change or legacy_change) or not isinstance(reloaded, str) or len(reloaded) != 64:
                errors.append(f"SD-LoRA state change/reload mismatch {key}")
        if not (
            fallback.get("adapter_requested") is True
            and fallback.get("adapter_loaded") is True
            and int(fallback.get("loaded_lora_tensor_count") or 0) > 0
            and fallback.get("base_trainable_parameter_count") == 0
            and fallback.get("loaded_adapter_tensor_sha256") == reloaded
        ):
            errors.append(f"no-BASE-fallback proof failed {key}")
        for field in ("sample_plan_sha256", "adapter_file_sha256", "evaluation_tensor_digest"):
            if not isinstance(payload.get(field), str) or len(str(payload.get(field))) != 64:
                errors.append(f"training {field} malformed {key}")
        seconds = metrics.get("training_time_seconds")
        if _finite_number(seconds, minimum=0.0):
            elapsed += float(seconds)
        else:
            errors.append(f"training time malformed {key}")
    expected = {(arm, increment) for arm in TREATMENTS for increment in range(len(tasks))}
    missing = sorted(expected - set(receipts))
    for arm in TREATMENTS:
        for increment in range(len(tasks)):
            receipt = receipts.get((arm, increment))
            if receipt is None:
                continue
            if increment == 0 and receipt.get("prior_adapter_file_sha256") is not None:
                errors.append(f"training adapter chain mismatch {(arm, increment)}")
            elif increment > 0 and (prior := receipts.get((arm, increment - 1))) is not None and (
                receipt.get("prior_adapter_file_sha256") != prior.get("adapter_file_sha256")
            ):
                errors.append(f"training adapter chain mismatch {(arm, increment)}")
    ordered = [receipts[key] for key in sorted(receipts) if key in expected]
    if include_missing:
        return ordered, errors, missing, elapsed
    return ordered, [*errors, *(f"missing training evidence {key}" for key in missing)], elapsed


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str], Counter[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    partial = Counter()
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeError) as exc:
        return rows, [f"invalid attempts file {path}: {type(exc).__name__}"], partial
    nonblank = [index for index, line in enumerate(lines) if line.strip()]
    last_nonblank = nonblank[-1] if nonblank else -1
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            is_truncated_tail = index - 1 == last_nonblank and not line.endswith(("\n", "\r"))
            if is_truncated_tail:
                partial["truncated-jsonl"] += 1
            else:
                errors.append(f"malformed attempts JSON {path}:{index}")
            continue
        if not isinstance(row, dict):
            errors.append(f"non-object attempt row {path}:{index}")
            continue
        rows.append(row)
    return rows, errors, partial


def classify(row: dict[str, Any]) -> str:
    override = row.get("_reconciliation_classification")
    if isinstance(override, str):
        return override
    if row.get("error") is not None:
        return "runtime"
    if not bool(row.get("identity_ok")):
        return "identity"
    if not bool(row.get("trace_complete")):
        return "incomplete-trace"
    if bool(row.get("parser_or_fallback_failure")):
        return "parser-or-fallback"
    receipt = row.get("adapter_load_receipt")
    if not isinstance(receipt, dict) or not bool(receipt.get("verified")):
        return "adapter-load-unverified"
    expected_loaded = str(row.get("arm")) != "base"
    if bool(receipt.get("adapter_loaded")) != expected_loaded:
        return "base-fallback"
    if not _finite_number(row.get("final_task_score"), minimum=0.0, maximum=1.0):
        return "score-invalid"
    if not _finite_number(row.get("attempt_wall_time_seconds"), minimum=0.0):
        return "time-invalid"
    return "valid"


def _ci(values: list[float], rng: random.Random) -> list[float] | None:
    if not values:
        return None
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(RESAMPLES)
    )
    return [samples[int(0.025 * (RESAMPLES - 1))], samples[int(0.975 * (RESAMPLES - 1))]]


def _sign_flip(values: list[float], rng: random.Random) -> float | None:
    if not values:
        return None
    observed = abs(sum(values) / len(values))
    extreme = 0
    for _ in range(RESAMPLES):
        estimate = abs(sum(value if rng.randrange(2) else -value for value in values) / len(values))
        extreme += estimate >= observed
    return (extreme + 1) / (RESAMPLES + 1)


def _comparison(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    stage: str,
    left: str,
    right: str,
    tasks: list[int],
    seeds: list[int],
    rng: random.Random,
) -> dict[str, Any]:
    complete: list[dict[str, Any]] = []
    itt: list[float] = []
    for task in tasks:
        valid_pairs: list[float] = []
        for seed in seeds:
            lrow = indexed.get((stage, left, task, seed))
            rrow = indexed.get((stage, right, task, seed))
            if lrow is None or rrow is None:
                continue
            lscore = float(lrow["final_task_score"]) if classify(lrow) == "valid" else 0.0
            rscore = float(rrow["final_task_score"]) if classify(rrow) == "valid" else 0.0
            itt.append(lscore - rscore)
            if classify(lrow) == classify(rrow) == "valid":
                valid_pairs.append(
                    float(lrow["final_task_score"])
                    - float(rrow["final_task_score"])
                )
        if len(valid_pairs) == len(seeds):
            complete.append(
                {"goal_idx": task, "paired_seed_count": len(valid_pairs), "mean_delta": sum(valid_pairs) / len(valid_pairs)}
            )
    deltas = [item["mean_delta"] for item in complete]
    return {
        "contrast": f"{left} - {right}",
        "complete_valid_task_count": len(complete),
        "complete_valid_task_deltas": complete,
        "task_weighted_delta": sum(deltas) / len(deltas) if deltas else None,
        "task_cluster_bootstrap_ci95": _ci(deltas, rng),
        "task_level_randomization_p": _sign_flip(deltas, rng),
        "attempt_weighted_itt_delta": sum(itt) / len(itt) if itt else None,
        "positive_tie_negative_tasks": {
            "positive": sum(value > 0 for value in deltas),
            "tie": sum(value == 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
        },
    }


def _expected(manifest: dict[str, Any]) -> dict[str, tuple[list[str], list[int], list[int]]]:
    train = [int(value) for value in manifest.get("task_order", [])]
    fresh = [int(item["goal_idx"]) for item in manifest.get("fresh_tasks", [])]
    return {
        "acquisition": (
            [str(value) for value in manifest.get("acquisition", {}).get("arms", [])],
            train,
            [int(value) for value in manifest.get("acquisition", {}).get("rollout_seeds", [])],
        ),
        "retention": (
            [str(value) for value in manifest.get("retention", {}).get("arms", [])],
            train,
            [int(value) for value in manifest.get("retention", {}).get("rollout_seeds", [])],
        ),
        "fresh": (
            [str(value) for value in manifest.get("fresh", {}).get("arms", [])],
            fresh,
            [int(value) for value in manifest.get("fresh", {}).get("rollout_seeds", [])],
        ),
    }


def _validate_qualification(
    path: Path,
    prereg: dict[str, Any] | None,
    training_rows: list[dict[str, Any]],
) -> tuple[bool, list[str], str | None]:
    if not path.is_file():
        return False, [], None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, [f"invalid qualification evidence: {type(exc).__name__}"], None
    errors: list[str] = []
    if payload.get("schema") != QUALIFICATION_SCHEMA or payload.get("status") != "pass":
        errors.append("qualification schema/status mismatch")
    if payload.get("increment_count") != 8 or payload.get("formal_fresh_task_consumption_allowed") is not True:
        errors.append("qualification increment/fresh gate mismatch")
    capacity = payload.get("final_capacity") if isinstance(payload.get("final_capacity"), dict) else {}
    ordinary_inventory = capacity.get("ordinary_inventory") if isinstance(capacity.get("ordinary_inventory"), dict) else {}
    sd_inventory = capacity.get("sd_inventory") if isinstance(capacity.get("sd_inventory"), dict) else {}
    ordinary_count = ordinary_inventory.get("parameter_count")
    sd_count = sd_inventory.get("parameter_count")
    if (
        capacity.get("ordinary_effective_rank") != 32
        or capacity.get("sd_effective_rank") != 32
        or not isinstance(ordinary_count, int)
        or isinstance(ordinary_count, bool)
        or ordinary_count < 1
        or ordinary_count != sd_count
        or capacity.get("serialized_parameter_count") != ordinary_count
    ):
        errors.append("qualification final capacity mismatch")
    if prereg is not None:
        training = prereg.get("matched_training_contract") if isinstance(prereg.get("matched_training_contract"), dict) else {}
        if payload.get("sample_plan_bundle_sha256") != training.get("sample_plan_bundle_sha256"):
            errors.append("qualification sample-plan bundle mismatch")
    qualified_rows = payload.get("training_receipts") if isinstance(payload.get("training_receipts"), list) else []
    qualified_by_key = {
        (str(row.get("arm")), row.get("increment_index")): row
        for row in qualified_rows
        if isinstance(row, dict)
    }
    expected_keys = {
        (arm, increment) for arm in TREATMENTS for increment in range(8)
    }
    formal_training_keys = {
        (str(row.get("arm")), row.get("increment_index"))
        for row in training_rows
        if isinstance(row, dict)
    }
    if not (
        len(qualified_rows) == 16
        and len(qualified_by_key) == 16
        and set(qualified_by_key) == formal_training_keys == expected_keys
    ):
        errors.append("qualification training receipt key set mismatch")
    for row in training_rows:
        key = (str(row.get("arm")), row.get("increment_index"))
        qualified = qualified_by_key.get(key)
        if qualified is not None and _canonical_hash(row) != _canonical_hash(qualified):
            errors.append(f"qualification/training receipt mismatch {key}")
    return not errors, errors, _file_sha256(path)


def _forgetting_comparison(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    tasks: list[int],
    acquisition_seeds: list[int],
    retention_seeds: list[int],
    rng: random.Random,
) -> dict[str, Any]:
    complete: list[dict[str, Any]] = []
    for task in tasks:
        forgetting: dict[str, float] = {}
        for arm in TREATMENTS:
            acquisition_rows = [indexed.get(("acquisition", arm, task, seed)) for seed in acquisition_seeds]
            retention_rows = [indexed.get(("retention", arm, task, seed)) for seed in retention_seeds]
            if (
                all(row is not None and classify(row) == "valid" for row in acquisition_rows)
                and all(row is not None and classify(row) == "valid" for row in retention_rows)
            ):
                acquisition_mean = sum(float(row["final_task_score"]) for row in acquisition_rows if row is not None) / len(acquisition_seeds)
                retention_mean = sum(float(row["final_task_score"]) for row in retention_rows if row is not None) / len(retention_seeds)
                forgetting[arm] = acquisition_mean - retention_mean
        if set(forgetting) == set(TREATMENTS):
            complete.append(
                {
                    "goal_idx": task,
                    "ordinary_forgetting": forgetting["ordinary-lora"],
                    "sd_forgetting": forgetting["sd-lora"],
                    "mean_delta": forgetting["sd-lora"] - forgetting["ordinary-lora"],
                }
            )
    deltas = [item["mean_delta"] for item in complete]
    return {
        "contrast": "SD acquisition-minus-retention forgetting - ordinary acquisition-minus-retention forgetting",
        "complete_valid_task_count": len(complete),
        "complete_valid_task_deltas": complete,
        "task_weighted_delta": sum(deltas) / len(deltas) if deltas else None,
        "task_cluster_bootstrap_ci95": _ci(deltas, rng),
        "task_level_randomization_p": _sign_flip(deltas, rng),
        "positive_tie_negative_tasks": {
            "positive": sum(value > 0 for value in deltas),
            "tie": sum(value == 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
        },
    }


def _mean_valid(rows: Iterable[dict[str, Any]]) -> float | None:
    scores = [float(row["final_task_score"]) for row in rows if classify(row) == "valid"]
    return sum(scores) / len(scores) if scores else None


def reconcile(
    manifest_path: Path,
    run_root: Path,
    training_evidence: list[Path],
    output: Path,
    *,
    prereg_path: Path | None = None,
    qualification_path: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = _expected(manifest)
    structural_errors: list[str] = _validate_manifest(manifest)
    prereg: dict[str, Any] | None = None
    prereg_file_sha256: str | None = None
    if prereg_path is not None:
        try:
            prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
            prereg_file_sha256 = _file_sha256(prereg_path)
        except (OSError, ValueError) as exc:
            structural_errors.append(f"invalid preregistration: {type(exc).__name__}")
        else:
            structural_errors.extend(_validate_preregistered_contract(manifest, manifest_path, prereg))
    training_rows, training_errors, missing_training, training_seconds = _validate_training_receipts(
        manifest, training_evidence, include_missing=True
    )
    structural_errors.extend(training_errors)
    training_by_key = {(str(row["arm"]), int(row["increment_index"])): row for row in training_rows}
    final_increment = len(contract["acquisition"][1]) - 1
    qualification_required = prereg_path is not None or qualification_path is not None
    qualification_passed = not qualification_required
    qualification_file_sha256: str | None = None
    if qualification_path is not None:
        qualification_passed, qualification_errors, qualification_file_sha256 = _validate_qualification(
            qualification_path, prereg, training_rows
        )
        structural_errors.extend(qualification_errors)
    if contract["acquisition"][0] != list(TREATMENTS):
        structural_errors.append("acquisition arm contract mismatch")
    if contract["retention"][0] != list(ARMS) or contract["fresh"][0] != list(ARMS):
        structural_errors.append("final arm contract mismatch")
    indexed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    expected_keys: set[tuple[str, str, int, int]] = set()
    expected_cells: set[Path] = set()
    for stage, (arms, tasks, seeds) in contract.items():
        for arm in arms:
            for task in tasks:
                expected_cells.add(Path("evaluation") / stage / arm / f"task{task}")
                expected_keys.update((stage, arm, task, seed) for seed in seeds)
    actual_artifacts = {
        path.relative_to(run_root)
        for name in ("attempts.jsonl", "summary.json")
        for path in run_root.glob(f"evaluation/**/{name}")
        if path.is_file()
    }
    for artifact in sorted(actual_artifacts, key=str):
        if artifact.parent not in expected_cells:
            structural_errors.append(f"unexpected evaluation artifact {artifact}")
    task_rows = [
        item
        for field in ("training_tasks", "fresh_tasks")
        for item in (manifest.get(field) if isinstance(manifest.get(field), list) else [])
        if isinstance(item, dict)
    ]
    task_id_by_goal = {int(item["goal_idx"]): item.get("task_id") for item in task_rows if isinstance(item.get("goal_idx"), int)}
    acquisition_increment = {task: increment for increment, task in enumerate(contract["acquisition"][1])}
    partial_taxonomy: Counter[str] = Counter()
    if missing_training:
        partial_taxonomy["missing-training-receipt"] += len(missing_training)
    if qualification_required and not qualification_passed and not structural_errors:
        partial_taxonomy["qualification-unverified"] += 1
    evaluation_gpu_seconds = 0.0
    for stage, (arms, tasks, seeds) in contract.items():
        for arm in arms:
            for task in tasks:
                cell = run_root / "evaluation" / stage / arm / f"task{task}"
                attempts = cell / "attempts.jsonl"
                summary = cell / "summary.json"
                if not attempts.is_file():
                    if summary.is_file():
                        partial_taxonomy["summary-without-attempts"] += 1
                    continue
                if summary.is_file():
                    try:
                        summary_payload = json.loads(summary.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        structural_errors.append(f"invalid evaluation summary {summary}: {type(exc).__name__}")
                    else:
                        seconds = summary_payload.get("wall_time_seconds") if isinstance(summary_payload, dict) else None
                        if _finite_number(seconds, minimum=0.0):
                            evaluation_gpu_seconds += float(seconds)
                        else:
                            structural_errors.append(f"evaluation summary time malformed {summary}")
                else:
                    partial_taxonomy["missing-summary"] += 1
                rows, row_errors, row_partial = _load_rows(attempts)
                structural_errors.extend(row_errors)
                partial_taxonomy.update(row_partial)
                for row in rows:
                    row_arm = str(row.get("arm") or "")
                    goal = row.get("goal_idx")
                    seed = row.get("rollout_seed")
                    if (
                        isinstance(goal, bool)
                        or not isinstance(goal, int)
                        or isinstance(seed, bool)
                        or not isinstance(seed, int)
                    ):
                        structural_errors.append(f"malformed evaluation key {stage}/{arm}/task{task}")
                        continue
                    key = (stage, row_arm, goal, seed)
                    if row.get("evaluation_stage") != stage:
                        structural_errors.append(f"evaluation stage mismatch {key}")
                    expected_increment = acquisition_increment.get(task, -1) if stage == "acquisition" else -1
                    if row.get("increment_index") != expected_increment:
                        structural_errors.append(f"evaluation increment index mismatch {key}")
                    if row.get("task_id") is not None and row.get("task_id") != task_id_by_goal.get(task):
                        structural_errors.append(f"evaluation task identity mismatch {key}")
                    if row_arm in TREATMENTS:
                        receipt_increment = expected_increment if stage == "acquisition" else final_increment
                        training = training_by_key.get((row_arm, receipt_increment))
                        receipt = row.get("adapter_load_receipt") if isinstance(row.get("adapter_load_receipt"), dict) else {}
                        if training is None:
                            row["_reconciliation_classification"] = "qualification-unverified"
                        elif (
                            receipt.get("adapter_file_sha256") != training.get("adapter_file_sha256")
                            or receipt.get("loaded_tensor_digest") != training.get("evaluation_tensor_digest")
                        ):
                            structural_errors.append(f"evaluation adapter identity mismatch {key}")
                    else:
                        receipt = row.get("adapter_load_receipt") if isinstance(row.get("adapter_load_receipt"), dict) else {}
                        if receipt.get("adapter_loaded") is not False or any(
                            receipt.get(field) is not None for field in ("adapter_file_sha256", "loaded_tensor_digest")
                        ):
                            structural_errors.append(f"BASE adapter receipt mismatch {key}")
                    if qualification_required and not qualification_passed:
                        row["_reconciliation_classification"] = "qualification-unverified"
                    if key in indexed:
                        structural_errors.append(f"duplicate attempt {key}")
                    elif key not in expected_keys:
                        structural_errors.append(f"unexpected attempt {key}")
                    indexed[key] = row
    missing = sorted(expected_keys - set(indexed))
    taxonomy = Counter(classify(row) for row in indexed.values())
    stage_metrics: dict[str, Any] = {}
    for stage, (arms, tasks, seeds) in contract.items():
        stage_metrics[stage] = {}
        for arm in arms:
            rows = [row for (s, a, _, _), row in indexed.items() if s == stage and a == arm]
            stage_metrics[stage][arm] = {
                "planned_attempts": len(tasks) * len(seeds),
                "observed_attempts": len(rows),
                "scientific_valid_attempts": sum(classify(row) == "valid" for row in rows),
                "mean_score_valid": _mean_valid(rows),
                "invalid_taxonomy": dict(sorted(Counter(classify(row) for row in rows if classify(row) != "valid").items())),
            }
    rng = random.Random(RESAMPLE_SEED)
    acquisition_tasks, acquisition_seeds = contract["acquisition"][1], contract["acquisition"][2]
    fresh_tasks, fresh_seeds = contract["fresh"][1], contract["fresh"][2]
    retention_tasks, retention_seeds = contract["retention"][1], contract["retention"][2]
    acquisition_comparisons = {
        "sd_minus_ordinary": _comparison(indexed, stage="acquisition", left="sd-lora", right="ordinary-lora", tasks=acquisition_tasks, seeds=acquisition_seeds, rng=rng),
    }
    fresh_comparisons = {
        "primary_sd_minus_ordinary": _comparison(indexed, stage="fresh", left="sd-lora", right="ordinary-lora", tasks=fresh_tasks, seeds=fresh_seeds, rng=rng),
        "ordinary_minus_base": _comparison(indexed, stage="fresh", left="ordinary-lora", right="base", tasks=fresh_tasks, seeds=fresh_seeds, rng=rng),
        "sd_minus_base": _comparison(indexed, stage="fresh", left="sd-lora", right="base", tasks=fresh_tasks, seeds=fresh_seeds, rng=rng),
    }
    retention_comparisons = {
        "ordinary_minus_base": _comparison(indexed, stage="retention", left="ordinary-lora", right="base", tasks=retention_tasks, seeds=retention_seeds, rng=rng),
        "sd_minus_base": _comparison(indexed, stage="retention", left="sd-lora", right="base", tasks=retention_tasks, seeds=retention_seeds, rng=rng),
        "sd_minus_ordinary": _comparison(indexed, stage="retention", left="sd-lora", right="ordinary-lora", tasks=retention_tasks, seeds=retention_seeds, rng=rng),
    }
    forgetting_comparisons = {
        "sd_minus_ordinary": _forgetting_comparison(
            indexed,
            tasks=acquisition_tasks,
            acquisition_seeds=acquisition_seeds,
            retention_seeds=retention_seeds,
            rng=rng,
        )
    }
    acquisition: list[dict[str, Any]] = []
    for increment, task in enumerate(contract["acquisition"][1]):
        item: dict[str, Any] = {"increment": increment, "goal_idx": task}
        for arm in TREATMENTS:
            rows = [
                indexed[key]
                for key in indexed
                if key[0] == "acquisition" and key[1] == arm and key[2] == task
            ]
            acquisition_mean = _mean_valid(rows)
            retention_rows = [
                indexed[key]
                for key in indexed
                if key[0] == "retention" and key[1] == arm and key[2] == task
            ]
            retention_mean = _mean_valid(retention_rows)
            item[arm] = {
                "mean_score_valid": acquisition_mean,
                "valid_attempts": sum(classify(row) == "valid" for row in rows),
                "final_retention_mean_score_valid": retention_mean,
                "forgetting_acquisition_minus_retention": (
                    acquisition_mean - retention_mean
                    if acquisition_mean is not None and retention_mean is not None
                    else None
                ),
            }
        acquisition.append(item)
    unqualified = bool(missing_training) or (qualification_required and not qualification_passed)
    status = "invalid" if structural_errors else (
        "partial-unqualified" if unqualified else ("complete" if not missing and not partial_taxonomy else "partial")
    )
    result = {
        "schema": "openevo.webshop-h1.38b-method-control-reconciliation.v1",
        "status": status,
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "expected_attempts": len(expected_keys),
        "observed_attempts": len(indexed),
        "missing_attempts": len(missing),
        "missing_keys": [list(key) for key in missing],
        "partial_taxonomy": dict(sorted(partial_taxonomy.items())),
        "valid_invalid_taxonomy": dict(sorted(taxonomy.items())),
        "stage_metrics": stage_metrics,
        "acquisition": acquisition,
        "acquisition_comparisons": acquisition_comparisons,
        "retention_comparisons": retention_comparisons,
        "forgetting_comparisons": forgetting_comparisons,
        "fresh_comparisons": fresh_comparisons,
        "gpu_hours": {
            "training": training_seconds / 3600.0,
            "evaluation": evaluation_gpu_seconds / 3600.0,
            "total": (training_seconds + evaluation_gpu_seconds) / 3600.0,
        },
        "training_evidence_count": len(training_rows),
        "missing_training_receipts": [list(key) for key in missing_training],
        "preregistration_file_sha256": prereg_file_sha256,
        "qualification_file_sha256": qualification_file_sha256,
        "qualification_passed": qualification_passed,
        "structural_errors": structural_errors,
        "resampling": {"rounds": RESAMPLES, "seed": RESAMPLE_SEED},
        "claim_boundary": "H1.38B matched D8 WebShop method control only; no SD-LoRA-RR, teacher, SEED, memory, or retrieval claim.",
    }
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-evidence", type=Path, action="append", default=[])
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path("configs/experiment/h1.38b-method-control-v1.json"),
    )
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    qualification = (
        args.qualification.resolve()
        if args.qualification is not None
        else run_root / "evidence" / "h1.38b-qualification.json"
    )
    result = reconcile(
        args.manifest.resolve(),
        run_root,
        args.training_evidence,
        args.output.resolve(),
        prereg_path=args.prereg.resolve(),
        qualification_path=qualification,
    )
    print(json.dumps({"status": result["status"], "observed": result["observed_attempts"], "expected": result["expected_attempts"]}, sort_keys=True))
    raise SystemExit(0 if result["status"] == "complete" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
