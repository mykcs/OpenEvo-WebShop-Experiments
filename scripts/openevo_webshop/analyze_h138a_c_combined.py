#!/usr/bin/env python3
"""Fail-closed combined reconciliation for the closed H1.38A run plus H1.38A-C."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ARMS = ("base", "diversity-4", "diversity-8", "diversity-16")
SEEDS = (2026090500, 2026090501, 2026090502, 2026090503)
SOURCE_SPLIT = {
    "original": {"base": 48, "diversity-4": 48, "diversity-8": 10, "diversity-16": 0},
    "continuation": {"base": 0, "diversity-4": 0, "diversity-8": 38, "diversity-16": 48},
}
RESAMPLES = 10_000
RESAMPLE_SEED = 20260818138
H138A_CAMPAIGN_ID = "20260819-0015-h138a-pooled-diversity"
H138AC_CAMPAIGN_ID = "20260819-0215-h138a-c-frozen-panel-completion"
H138A_LEGACY_CELLS = [
    {"arm": "base", "task": 536, "execution_sha": "ea8b3d4c5b18345ade9f7d9307498b29d00b4939", "manifest_sha256": "e60c100701cc35a237fb298ea8c2372569e26eeade581aa5030f68e7ac87807f", "protocol_fingerprint": "1268bb47b58872dd32865ca7b9ce1fbacf253236fcbba4bd4ab42567fab17f7e"},
    {"arm": "diversity-4", "task": 536, "execution_sha": "ea8b3d4c5b18345ade9f7d9307498b29d00b4939", "manifest_sha256": "e60c100701cc35a237fb298ea8c2372569e26eeade581aa5030f68e7ac87807f", "protocol_fingerprint": "97e579611c5da2dcb2368388a111136a39a95713804cce898492f2a60129ddbd"},
]


def valid(row: dict[str, Any]) -> bool:
    return row.get("identity_ok") is True and row.get("trace_complete") is True and row.get("parser_or_fallback_failure") is False and row.get("error") is None


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ci(values: list[float], rng: random.Random) -> list[float] | None:
    if not values:
        return None
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(RESAMPLES))
    return [samples[int(0.025 * (RESAMPLES - 1))], samples[int(0.975 * (RESAMPLES - 1))]]


def sign_flip(values: list[float], rng: random.Random) -> float | None:
    if not values:
        return None
    observed = abs(sum(values) / len(values))
    extreme = 0
    for _ in range(RESAMPLES):
        estimate = abs(sum(value if rng.randrange(2) else -value for value in values) / len(values))
        extreme += estimate >= observed
    return (extreme + 1) / (RESAMPLES + 1)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_fingerprint(*, arm: str, gpu_uuid: str, manifest_sha256: str, frozen: dict[str, Any]) -> str:
    protocol = {
        "phase": "H1",
        "arm": arm,
        "model": frozen["evaluation_model_path"],
        "model_revision": frozen["model_content_pin"],
        "task_manifest_sha256": manifest_sha256,
        "environment_seed": int(frozen["evaluation_environment_seed"]),
        "generation_seeds": [int(seed) for seed in frozen["rollout_seeds"]],
        "temperature": float(frozen["temperature"]),
        "max_steps": int(frozen["max_steps"]),
        "max_new_tokens": int(frozen["max_new_tokens"]),
        "prompt_contract": frozen["prompt_contract"],
        "action_contract": frozen["action_contract"],
        "evaluator": frozen["evaluator"],
        "parser_retry_seed_offset": int(frozen["parser_retry_seed_offset"]),
        "cuda_visible_devices": gpu_uuid,
        "gpu_binding_contract": frozen["gpu_binding_contract"],
        "latency_contract": frozen["latency_contract"],
    }
    return hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label}: {path}: {type(exc).__name__}")
        return None
    if not isinstance(value, dict):
        errors.append(f"invalid {label} object: {path}")
        return None
    return value


def _repo_evidence_path(config_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    repo = config_path.resolve().parents[2]
    evidence_root = (repo / "docs/evidence").resolve()
    if not evidence_root.is_dir() or not evidence_root.is_relative_to(repo):
        return None
    declared = repo / value
    if declared.is_symlink():
        return None
    candidate = declared.resolve()
    if not candidate.is_relative_to(evidence_root):
        return None
    return candidate


def _legacy_original_cell(config: dict[str, Any], arm: str, task: int) -> dict[str, Any] | None:
    original = config.get("original_campaign")
    if not isinstance(original, dict) or original.get("campaign_id") != H138A_CAMPAIGN_ID:
        return None
    if original.get("legacy_completed_cells") != H138A_LEGACY_CELLS:
        return None
    for item in H138A_LEGACY_CELLS:
        if item["arm"] == arm and item["task"] == task:
            return item
    return None


def _sealed_continuation_receipts(config_path: Path, config: dict[str, Any], errors: list[str]) -> dict[str, str] | None:
    if "tracking" not in config:
        if config.get("campaign_id") == H138AC_CAMPAIGN_ID:
            errors.append("sealed fallback receipt manifest missing")
        return None
    tracking = config["tracking"]
    if not isinstance(tracking, dict):
        errors.append("tracking contract invalid")
        return None
    declared = tracking.get("sealed_fallback_receipt_manifest")
    if declared is None:
        if config.get("campaign_id") == H138AC_CAMPAIGN_ID:
            errors.append("sealed fallback receipt manifest missing")
        return None
    if not isinstance(declared, dict):
        errors.append("sealed fallback receipt manifest contract invalid")
        return None
    path = _repo_evidence_path(config_path, declared.get("path"))
    expected_sha = declared.get("sha256")
    expected_count = declared.get("count")
    if path is None or not path.is_file() or not isinstance(expected_sha, str) or _file_sha256(path) != expected_sha:
        errors.append("sealed fallback receipt manifest SHA mismatch")
        return None
    records: dict[str, str] = {}
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    except OSError:
        errors.append("sealed fallback receipt manifest unreadable")
        return None
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (evaluation/(diversity-8|diversity-16)/task([0-9]+)/wandb/wandb-fallback-receipt\.json)", line)
        if match is None or match.group(2) in records:
            errors.append("sealed fallback receipt manifest entry invalid")
            return None
        records[match.group(2)] = "sha256:" + match.group(1)
    if not isinstance(expected_count, int) or expected_count != len(records):
        errors.append("sealed fallback receipt manifest count mismatch")
        return None
    return records


def _valid_receipt_timestamps(started: Any, finished: Any) -> bool:
    if not isinstance(started, str) or not isinstance(finished, str) or not started or not finished:
        return False
    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except ValueError:
        return False
    return start.tzinfo is not None and finish.tzinfo is not None and finish >= start


def _original_failed_online_probe(config_path: Path, config: dict[str, Any]) -> dict[str, str] | None:
    original = config.get("original_campaign", {})
    probe = _repo_evidence_path(config_path, original.get("wandb_online_probe_evidence_path"))
    if probe is None or not probe.is_file() or _file_sha256(probe) != original.get("wandb_online_probe_sha256"):
        return None
    try:
        payload = json.loads(probe.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "openevo.wandb-online-probe.v1"
        or payload.get("requested_mode") != "online"
        or not isinstance(payload.get("exit_code"), int)
        or payload.get("exit_code") == 0
        or not isinstance(payload.get("output"), str)
        or not payload["output"]
    ):
        return None
    return {"path": original["wandb_online_probe_evidence_path"], "sha256": "sha256:" + _file_sha256(probe)}


def _read_wandb_run_config(cell: Path) -> tuple[Path, dict[str, Any]]:
    """Read the config embedded in the cell's one immutable W&B transaction log."""
    spools = sorted((cell / "wandb").rglob("run-*.wandb"))
    if len(spools) != 1:
        raise ValueError(f"expected one W&B run spool, found {len(spools)}")
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError as exc:
        raise ValueError("wandb internal reader unavailable") from exc
    observed: dict[str, Any] | None = None
    exits: list[int] = []
    later_config_keys: list[str] = []
    with tempfile.NamedTemporaryFile(suffix=".wandb") as copy:
        shutil.copyfile(spools[0], copy.name)
        store = DataStore()
        store.open_for_scan(copy.name)
        while True:
            payload = store.scan_data()
            if payload is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            record_type = record.WhichOneof("record_type")
            if record_type == "exit":
                exits.append(record.exit.exit_code)
                continue
            if record_type == "config":
                for item in [*record.config.update, *record.config.remove]:
                    if bool(item.key) == bool(item.nested_key):
                        raise ValueError("invalid later W&B config key encoding")
                    later_config_keys.append(item.key or ".".join(item.nested_key))
                continue
            if record_type != "run":
                continue
            if observed is not None:
                raise ValueError("multiple W&B run records")
            values: dict[str, Any] = {}
            for item in record.run.config.update:
                if bool(item.key) == bool(item.nested_key):
                    raise ValueError("invalid W&B config key encoding")
                key = item.key or ".".join(item.nested_key)
                if key in values:
                    raise ValueError(f"duplicate W&B config key: {key}")
                try:
                    values[key] = json.loads(item.value_json)
                except (AttributeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid W&B config value for {key!r}") from exc
            values["__wandb_run_id"] = record.run.run_id
            observed = values
        store.close()
    if observed is None or exits != [0]:
        raise ValueError("W&B run record or clean exit missing")
    observed["__wandb_later_config_keys"] = later_config_keys
    return spools[0], observed


def _git_execution_is_allowed(config_path: Path, orchestration_sha: str, execution_sha: str) -> bool:
    try:
        repo = str(config_path.resolve().parents[2])
        git = ["git", "-c", f"safe.directory={repo}", "-C", repo]
        if subprocess.run([*git, "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True).stdout.strip() != repo:
            return False
        subprocess.run([*git, "cat-file", "-e", f"{execution_sha}^{{commit}}"], check=True, capture_output=True)
        subprocess.run([*git, "merge-base", "--is-ancestor", orchestration_sha, execution_sha], check=True, capture_output=True)
        subprocess.run([*git, "merge-base", "--is-ancestor", execution_sha, "HEAD"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _continuation_execution_shas(
    root: Path, config_path: Path, config: dict[str, Any], manifest_sha256: str, errors: list[str]
) -> set[str]:
    evidence = root / "evidence"
    candidates = sorted(evidence.glob("provenance-*.json"))
    if (evidence / "provenance.json").is_file():
        candidates.insert(0, evidence / "provenance.json")
    valid_shas: set[str] = set()
    for path in candidates:
        if path.is_symlink():
            errors.append(f"unsafe continuation execution provenance: {path}")
            continue
        payload = _load_json(path, "continuation execution provenance", errors)
        if payload is None:
            continue
        legacy = path.name == "provenance.json"
        execution_sha = payload.get("continuation_execution_sha" if legacy else "execution_sha")
        expected_schema = "openevo.webshop-h1.38a-c-provenance.v1" if legacy else "openevo.webshop-h1.38a-c-execution-provenance.v1"
        if (
            payload.get("schema") != expected_schema
            or payload.get("campaign_id") != config.get("campaign_id")
            or payload.get("orchestration_sha") != config.get("continuation_orchestration_sha")
            or payload.get("original_scientific_sha") != config.get("original_campaign", {}).get("scientific_execution_sha")
            or payload.get("original_manifest_sha256") != manifest_sha256
            or payload.get("continuation_root") != config.get("continuation_run_root")
            or payload.get("authorized_gpu_uuids") != config.get("authorized_gpu_uuids")
            or payload.get("selection_policy") != "fixed missing D8/D16 cells only"
            or not isinstance(execution_sha, str)
            or len(execution_sha) != 40
            or any(character not in "0123456789abcdef" for character in execution_sha)
            or (not legacy and path.name != f"provenance-{execution_sha}.json")
            or not _git_execution_is_allowed(config_path, config.get("continuation_orchestration_sha", ""), execution_sha)
        ):
            errors.append(f"continuation execution provenance mismatch: {path}")
            continue
        valid_shas.add(execution_sha)
    if not valid_shas:
        errors.append("no valid continuation execution provenance")
    return valid_shas


def _expected_wandb_config(
    *, source: str, arm: str, task: int, gpu_uuid: str, manifest_sha256: str, config: dict[str, Any]
) -> dict[str, Any]:
    frozen = config["frozen_provenance"]
    adapter = "none" if arm == "base" else frozen["adapters"][arm]["adapter_uri"]
    legacy = _legacy_original_cell(config, arm, task) if source == "original" else None
    return {
        "phase": "H1",
        "arm": arm,
        "adapter": adapter.removeprefix("file://"),
        "campaign_id": config["original_campaign"]["campaign_id"] if source == "original" else config["campaign_id"],
        "upstream_openevo_sha": frozen["upstream_open_evo_sha"],
        "runtime_fingerprint": frozen["python_runtime_fingerprint"],
        "image_digest": frozen["image_digest"],
        "gpu_uuid": gpu_uuid,
        "model": frozen["evaluation_model_path"],
        "model_revision": frozen["model_content_pin"],
        "task_manifest_sha256": legacy["manifest_sha256"] if legacy else manifest_sha256,
        "environment_seed": int(frozen["evaluation_environment_seed"]),
        "generation_seeds": [int(seed) for seed in frozen["rollout_seeds"]],
        "prompt_contract": frozen["prompt_contract"],
        "action_contract": frozen["action_contract"],
        "evaluator": frozen["evaluator"],
        "parser_retry_seed_offset": int(frozen["parser_retry_seed_offset"]),
        "cuda_visible_devices": gpu_uuid,
        "gpu_binding_contract": frozen["gpu_binding_contract"],
        "latency_contract": frozen["latency_contract"],
        "temperature": float(frozen["temperature"]),
        "max_steps": int(frozen["max_steps"]),
        "max_new_tokens": int(frozen["max_new_tokens"]),
    }


def _cell_manifest_sha256(*, source: str, arm: str, task: int, manifest_sha256: str, config: dict[str, Any]) -> str:
    """Return the manifest identity sealed by this cell's execution contract.

    The first H1.38A gate (BASE/D4 task 536) predates the final launcher and
    recorded the earlier manifest file digest. That sealed evidence is retained
    as-is; all later cells use the final immutable file digest.
    """
    legacy = _legacy_original_cell(config, arm, task) if source == "original" else None
    return legacy["manifest_sha256"] if legacy else manifest_sha256


def _legacy_wandb_spool_sha256(receipt: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(receipt.parent.rglob("*")):
        if not path.is_file() or path == receipt:
            continue
        digest.update(str(path.relative_to(receipt.parent)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _failed_online_probe(root: Path, execution_sha: str) -> tuple[Path, str] | None:
    probe = root / "evidence" / f"wandb-online-probe-{execution_sha}.json"
    if not probe.is_file() or probe.is_symlink():
        return None
    try:
        payload = json.loads(probe.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "openevo.wandb-online-probe.v1"
        or payload.get("execution_sha") != execution_sha
        or payload.get("requested_mode") != "online"
        or payload.get("effective_mode") != "offline"
        or payload.get("status") != "failed"
        or not isinstance(payload.get("exit_code"), int)
        or payload.get("exit_code") == 0
        or not isinstance(payload.get("output"), str)
        or not payload["output"]
    ):
        return None
    return probe, "sha256:" + _file_sha256(probe)


def _tracking_evidence(
    cell: Path,
    *,
    source: str,
    arm: str,
    task: int,
    summary: dict[str, Any],
    gpu_uuid: str,
    manifest_sha256: str,
    config: dict[str, Any],
    continuation_execution_shas: set[str],
    original_fallback_probe: dict[str, str] | None,
    sealed_receipt_hashes: dict[str, str] | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    label = f"{source}/{arm}/{task}"
    try:
        spool, run_config = _read_wandb_run_config(cell)
    except Exception as exc:
        errors.append(f"invalid W&B run spool: {label}: {exc}")
        return None
    expected_config = _expected_wandb_config(source=source, arm=arm, task=task, gpu_uuid=gpu_uuid, manifest_sha256=manifest_sha256, config=config)
    for key, expected in expected_config.items():
        if run_config.get(key) != expected:
            errors.append(f"W&B config mismatch: {label}: {key}")
    if set(run_config.get("__wandb_later_config_keys", [])) & set(expected_config):
        errors.append(f"W&B frozen config changed after initialization: {label}")
    execution_sha = run_config.get("experiment_git_sha")
    legacy = _legacy_original_cell(config, arm, task) if source == "original" else None
    allowed = ({legacy["execution_sha"]} if legacy else {config["original_campaign"]["scientific_execution_sha"]}) if source == "original" else continuation_execution_shas
    if execution_sha not in allowed:
        errors.append(f"W&B execution provenance mismatch: {label}")
    tracking = summary.get("tracking")
    if not isinstance(tracking, dict):
        errors.append(f"summary tracking missing: {label}")
        return None
    mode = tracking.get("mode")
    if tracking.get("run_id") != run_config.get("__wandb_run_id"):
        errors.append(f"W&B run id mismatch: {label}")
    receipt = cell / "wandb/wandb-fallback-receipt.json"
    spool_sha256 = "sha256:" + _file_sha256(spool)
    receipt_hash_mode = None
    receipt_sha256 = None
    fallback_probe = None
    if mode == "online":
        if not isinstance(tracking.get("run_id"), str) or not tracking["run_id"] or receipt.exists():
            errors.append(f"invalid online W&B tracking evidence: {label}")
    elif mode == "offline":
        if receipt.exists():
            if receipt.is_symlink():
                errors.append(f"offline W&B fallback receipt unsafe: {label}")
            fallback = _load_json(receipt, f"offline W&B fallback receipt {label}", errors)
            receipt_sha256 = "sha256:" + _file_sha256(receipt)
            receipt_path = str(receipt.relative_to(cell.parents[2]))
            receipt_sha = fallback.get("spool_sha256") if fallback else None
            receipt_hash_mode = "direct-spool" if receipt_sha == spool_sha256 else ("legacy-directory" if receipt_sha == _legacy_wandb_spool_sha256(receipt) else None)
            receipt_fields_valid = (
                fallback is None
                or fallback.get("schema") != "openevo.wandb-fallback-receipt.v1"
                or fallback.get("requested_mode") != "online"
                or fallback.get("effective_mode") != "offline"
                or fallback.get("project") != "openevo-webshop-experiments"
                or not isinstance(fallback.get("online_error"), str)
                or not fallback["online_error"]
                or not _valid_receipt_timestamps(fallback.get("started_at_utc"), fallback.get("finished_at_utc"))
                or not isinstance(receipt_sha, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", receipt_sha) is None
            ) is False
            if not receipt_fields_valid:
                errors.append(f"offline W&B spool receipt mismatch: {label}")
            elif sealed_receipt_hashes is not None and sealed_receipt_hashes.get(receipt_path) != receipt_sha256:
                errors.append(f"offline W&B sealed receipt mismatch: {label}")
            elif receipt_hash_mode is None:
                # wandb can append/flush the spool after the per-cell receipt
                # is written.  Accept only the observed explicit API-key
                # fallback shape and retain the mismatch as audit evidence.
                online_error = str(fallback.get("online_error", "")) if fallback else ""
                if (
                    source == "continuation"
                    and sealed_receipt_hashes is not None
                    and sealed_receipt_hashes.get(receipt_path) == receipt_sha256
                    and online_error.startswith("wandb.init failed:")
                ):
                    receipt_hash_mode = "post-finalization-spool-mismatch"
                    warnings.append(f"offline W&B spool receipt hash drift: {label}")
                else:
                    errors.append(f"offline W&B spool receipt mismatch: {label}")
        elif source == "original" and original_fallback_probe is not None:
            fallback_probe = original_fallback_probe
        elif source == "continuation" and isinstance(execution_sha, str):
            failed_probe = _failed_online_probe(cell.parents[2], execution_sha)
            if failed_probe is None:
                errors.append(f"offline W&B fallback evidence missing: {label}")
            else:
                probe_path, probe_sha256 = failed_probe
                fallback_probe = {"path": str(probe_path.relative_to(cell.parents[2])), "sha256": probe_sha256}
        else:
            errors.append(f"offline W&B fallback evidence missing: {label}")
    else:
        errors.append(f"invalid W&B tracking mode: {label}")
    return {
        "source": source,
        "arm": arm,
        "task": task,
        "mode": mode,
        "run_id": tracking.get("run_id"),
        "fallback_receipt": (
            {"path": str(receipt.relative_to(cell.parents[2])), "sha256": receipt_sha256}
            if mode == "offline" and receipt.exists() and receipt_sha256 is not None
            else None
        ),
        "fallback_probe": fallback_probe,
        "spool": str(spool.relative_to(cell.parents[2])),
        "spool_sha256": spool_sha256,
        "receipt_hash_mode": receipt_hash_mode,
        "experiment_git_sha": execution_sha,
    }


def _source_contract(config: dict[str, Any], tasks: list[int], errors: list[str]) -> dict[str, dict[str, set[int]]]:
    declared = config.get("frozen_source_split", {})
    for source, counts in SOURCE_SPLIT.items():
        if declared.get(source) != counts:
            errors.append(f"source split count contract mismatch for {source}")
    original = config.get("original_campaign")
    if not isinstance(original, dict) or original.get("campaign_id") != H138A_CAMPAIGN_ID:
        errors.append("original campaign identity mismatch")
        original = {}
    if original.get("completed_cells") != SOURCE_SPLIT["original"]:
        errors.append("original closeout source split mismatch")
    legacy = original.get("legacy_completed_cells")
    if not isinstance(legacy, list) or legacy != H138A_LEGACY_CELLS:
        errors.append("original legacy completed-cell contract mismatch")
    original_d8 = original.get("completed_diversity_8_task_ids", [])
    if not isinstance(original_d8, list) or len(original_d8) != 10 or len(set(original_d8)) != 10 or not set(original_d8).issubset(tasks):
        errors.append("original diversity-8 task split mismatch")
        original_d8 = []
    original_d8_set = {int(task) for task in original_d8}
    all_tasks = set(tasks)
    return {
        "original": {
            "base": all_tasks,
            "diversity-4": all_tasks,
            "diversity-8": original_d8_set,
            "diversity-16": set(),
        },
        "continuation": {
            "base": set(),
            "diversity-4": set(),
            "diversity-8": all_tasks - original_d8_set,
            "diversity-16": all_tasks,
        },
    }


def _scan_evaluation_tree(root: Path, source: str, allowed: dict[str, set[int]], errors: list[str]) -> None:
    evaluation = root / "evaluation"
    if not evaluation.is_dir():
        errors.append(f"missing evaluation tree: {source}")
        return
    for arm_path in sorted(path for path in evaluation.iterdir() if path.is_dir()):
        if arm_path.is_symlink():
            errors.append(f"unexpected evaluation directory: {source}/{arm_path.name} is symlink")
            continue
        if arm_path.name not in ARMS:
            errors.append(f"unexpected evaluation directory: {source}/{arm_path.name}")
            continue
        for task_path in sorted(path for path in arm_path.iterdir() if path.is_dir()):
            if task_path.is_symlink():
                errors.append(f"unexpected evaluation directory: {source}/{arm_path.name}/{task_path.name} is symlink")
                continue
            name = task_path.name
            try:
                task = int(name.removeprefix("task")) if name.startswith("task") else -1
            except ValueError:
                task = -1
            if name != f"task{task}" or task not in allowed[arm_path.name]:
                errors.append(f"source split mismatch: unexpected evaluation directory {source}/{arm_path.name}/{name}")


def _cell_rows(
    cell: Path,
    *,
    source: str,
    arm: str,
    task: int,
    manifest_sha256: str,
    config: dict[str, Any],
    authorized_uuids: set[str],
    continuation_execution_shas: set[str],
    original_fallback_probe: dict[str, str] | None,
    sealed_receipt_hashes: dict[str, str] | None,
    errors: list[str],
    warnings: list[str],
    tracking_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]] | None:
    label = f"{source}/{arm}/{task}"
    summary = _load_json(cell / "summary.json", f"summary {label}", errors)
    gpu_before = _load_json(cell / "gpu-before.json", f"gpu-before {label}", errors)
    attempts_path = cell / "attempts.jsonl"
    if not attempts_path.is_file():
        errors.append(f"missing attempts {label}")
        return None
    try:
        rows = load_rows(attempts_path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid attempts {label}: {type(exc).__name__}")
        return None
    if summary is None or gpu_before is None:
        return None
    gpu_uuid = gpu_before.get("expected_uuid")
    if not isinstance(gpu_uuid, str) or gpu_uuid not in authorized_uuids:
        errors.append(f"unauthorized gpu-before expected_uuid: {label}: {gpu_uuid!r}")
        return rows
    if not any(gpu_uuid in str(row) for row in gpu_before.get("gpu_rows", [])):
        errors.append(f"gpu-before expected_uuid absent from gpu_rows: {label}")
    cell_manifest_sha256 = _cell_manifest_sha256(source=source, arm=arm, task=task, manifest_sha256=manifest_sha256, config=config)
    tracking = _tracking_evidence(
        cell,
        source=source,
        arm=arm,
        task=task,
        summary=summary,
        gpu_uuid=gpu_uuid,
        manifest_sha256=cell_manifest_sha256,
        config=config,
        continuation_execution_shas=continuation_execution_shas,
        original_fallback_probe=original_fallback_probe,
        sealed_receipt_hashes=sealed_receipt_hashes,
        errors=errors,
        warnings=warnings,
    )
    if tracking is not None and tracking_records is not None:
        tracking_records.append(tracking)
    frozen = config["frozen_provenance"]
    try:
        computed_fingerprint = _protocol_fingerprint(arm=arm, gpu_uuid=gpu_uuid, manifest_sha256=cell_manifest_sha256, frozen=frozen)
        legacy = _legacy_original_cell(config, arm, task) if source == "original" else None
        if legacy is not None and legacy.get("protocol_fingerprint") != computed_fingerprint:
            raise ValueError("legacy fingerprint does not match frozen protocol")
        expected_fingerprint = legacy["protocol_fingerprint"] if legacy else computed_fingerprint
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"frozen protocol contract invalid: {type(exc).__name__}: {exc}")
        return rows
    if summary.get("schema") != "seed3090.webshop-h1-closed-loop.v1" or summary.get("arm") != arm or summary.get("attempt_count") != 4:
        errors.append(f"summary identity mismatch: {label}")
    if summary.get("protocol_fingerprint") != expected_fingerprint:
        errors.append(f"summary protocol fingerprint mismatch: {label}")
    observed_seeds: list[int] = []
    if len(rows) != 4:
        errors.append(f"attempt count mismatch: {label}: {len(rows)}")
    for row in rows:
        if not isinstance(row, dict):
            errors.append(f"invalid attempt object: {label}")
            continue
        try:
            row_task = int(row.get("goal_idx", -1))
            seed = int(row.get("rollout_seed", -1))
        except (TypeError, ValueError):
            row_task, seed = -1, -1
        observed_seeds.append(seed)
        if row.get("arm") != arm or row_task != task or seed not in SEEDS:
            errors.append(f"attempt identity mismatch: {label}/{seed}")
        if row.get("protocol_fingerprint") != expected_fingerprint or row.get("protocol_fingerprint") != summary.get("protocol_fingerprint"):
            errors.append(f"attempt protocol fingerprint mismatch: {label}/{seed}")
    if sorted(observed_seeds) != sorted(SEEDS):
        errors.append(f"attempt seed coverage mismatch: {label}")
    valid_count = sum(valid(row) for row in rows if isinstance(row, dict))
    qualified_count = sum(row.get("qualified_positive") is True for row in rows if isinstance(row, dict))
    scores: list[float] = []
    for row in rows:
        try:
            score = float(row.get("final_task_score") or 0.0)
        except (AttributeError, TypeError, ValueError):
            score = math.nan
        if not math.isfinite(score):
            errors.append(f"non-finite attempt score: {label}")
        scores.append(score)
    mean_score = sum(scores) / len(scores) if scores and all(math.isfinite(score) for score in scores) else math.nan
    if summary.get("valid_count") != valid_count or summary.get("qualified_positive_count") != qualified_count:
        errors.append(f"summary denominator mismatch: {label}")
    try:
        mean_matches = math.isfinite(float(summary.get("mean_task_score"))) and math.isclose(float(summary["mean_task_score"]), mean_score, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        mean_matches = False
    if not mean_matches:
        errors.append(f"summary mean score mismatch: {label}")
    return rows


def combined(manifest_path: Path, config_path: Path, original_root: Path, continuation_root: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = [int(value) for value in manifest.get("evaluation_tasks", [])]
    errors: list[str] = []
    warnings: list[str] = []
    if original_root.resolve() == continuation_root.resolve():
        errors.append("original and continuation roots must differ")
    manifest_sha256 = _file_sha256(manifest_path)
    if len(tasks) != 48 or len(set(tasks)) != 48:
        errors.append("manifest task contract mismatch")
    if tuple(manifest.get("evaluation", {}).get("arms", [])) != ARMS:
        errors.append("manifest arm contract mismatch")
    if tuple(manifest.get("evaluation", {}).get("rollout_seeds", [])) != SEEDS:
        errors.append("manifest rollout seed contract mismatch")
    if manifest_sha256 != config["original_campaign"]["evaluation_manifest_sha256"]:
        errors.append("original manifest SHA mismatch")
    original_reconciliation = _repo_evidence_path(
        config_path, config["original_campaign"].get("reconciliation_evidence_path")
    )
    original_closeout = _repo_evidence_path(
        config_path, config["original_campaign"].get("closeout_summary_evidence_path")
    )
    if original_reconciliation is None or not original_reconciliation.is_file() or _file_sha256(original_reconciliation) != config["original_campaign"].get("reconciliation_sha256"):
        errors.append("original reconciliation SHA mismatch")
    if original_closeout is None or not original_closeout.is_file() or _file_sha256(original_closeout) != config["original_campaign"].get("closeout_summary_sha256"):
        errors.append("original closeout summary SHA mismatch")
    source_contract = _source_contract(config, tasks, errors)
    _scan_evaluation_tree(original_root, "original", source_contract["original"], errors)
    _scan_evaluation_tree(continuation_root, "continuation", source_contract["continuation"], errors)
    authorized_uuids = set(config.get("authorized_gpu_uuids", []))
    continuation_execution_shas = _continuation_execution_shas(continuation_root, config_path, config, manifest_sha256, errors)
    original_fallback_probe = _original_failed_online_probe(config_path, config)
    sealed_receipt_hashes = _sealed_continuation_receipts(config_path, config, errors)
    if original_fallback_probe is None:
        errors.append("original W&B online probe evidence invalid")
    tracking_records: list[dict[str, Any]] = []
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    source_counts: dict[str, dict[str, int]] = {arm: {"original": 0, "continuation": 0} for arm in ARMS}
    observed_cells = 0
    for arm in ARMS:
        for task in tasks:
            source = "original" if task in source_contract["original"][arm] else "continuation"
            root = (original_root if source == "original" else continuation_root) / "evaluation" / arm / f"task{task}"
            if not root.is_dir():
                errors.append(f"source split mismatch: missing {source} cell {arm}/{task}")
                continue
            rows = _cell_rows(root, source=source, arm=arm, task=task, manifest_sha256=manifest_sha256, config=config, authorized_uuids=authorized_uuids, continuation_execution_shas=continuation_execution_shas, original_fallback_probe=original_fallback_probe, sealed_receipt_hashes=sealed_receipt_hashes, errors=errors, warnings=warnings, tracking_records=tracking_records)
            if rows is None:
                continue
            observed_cells += 1
            source_counts[arm][source] += 1
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    key = (str(row.get("arm")), int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1)))
                except (TypeError, ValueError):
                    errors.append(f"attempt identity mismatch: {source}/{arm}/{task}")
                    continue
                if key[0] != arm or key[1] != task or key[2] not in SEEDS or key in indexed:
                    errors.append(f"attempt identity mismatch: {key}")
                row = dict(row)
                row["provenance_source"] = source
                indexed[key] = row
    expected = {(arm, task, seed) for arm in ARMS for task in tasks for seed in SEEDS}
    if set(indexed) != expected:
        errors.append(f"combined attempt coverage mismatch missing={len(expected - set(indexed))} extra={len(set(indexed) - expected)}")
    for arm in ARMS:
        for source in ("original", "continuation"):
            if source_counts[arm][source] != SOURCE_SPLIT[source][arm]:
                errors.append(f"source split mismatch: {source}/{arm} expected={SOURCE_SPLIT[source][arm]} observed={source_counts[arm][source]}")
    if observed_cells != 192:
        errors.append(f"combined cell coverage mismatch expected=192 observed={observed_cells}")
    arm_metrics: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in indexed.values() if row.get("arm") == arm]
        good = [row for row in rows if valid(row)]
        arm_metrics[arm] = {
            "attempt_count": len(rows),
            "scientific_valid_count": len(good),
            "mean_score_valid": sum(float(row.get("final_task_score") or 0.0) for row in good) / len(good) if good else None,
            "mean_score_all_attempts": sum(float(row.get("final_task_score") or 0.0) for row in rows) / len(rows) if rows else None,
            "parser_or_fallback_failures": sum(bool(row.get("parser_or_fallback_failure")) for row in rows),
            "runtime_errors": sum(row.get("error") is not None for row in rows),
            "provenance_cells": source_counts[arm],
        }
        if not good:
            errors.append(f"zero scientific-valid attempts for arm: {arm}")
    comparisons: dict[str, Any] = {}
    rng = random.Random(RESAMPLE_SEED)
    for arm in ARMS[1:]:
        complete: list[dict[str, Any]] = []
        all_attempt_deltas: list[float] = []
        for task in tasks:
            pairs: list[float] = []
            for seed in SEEDS:
                base, treatment = indexed.get(("base", task, seed)), indexed.get((arm, task, seed))
                if base is not None and treatment is not None:
                    all_attempt_deltas.append(float(treatment.get("final_task_score") or 0.0) - float(base.get("final_task_score") or 0.0))
                    if valid(base) and valid(treatment):
                        pairs.append(float(treatment.get("final_task_score") or 0.0) - float(base.get("final_task_score") or 0.0))
            if len(pairs) == len(SEEDS):
                complete.append({"goal_idx": task, "paired_seed_count": len(pairs), "mean_delta": sum(pairs) / len(pairs)})
        deltas = [row["mean_delta"] for row in complete]
        comparisons[arm] = {
            "complete_valid_task_count": len(complete),
            "task_weighted_delta": sum(deltas) / len(deltas) if deltas else None,
            "task_cluster_bootstrap_ci95": ci(deltas, rng),
            "task_level_sign_flip_p": sign_flip(deltas, rng),
            "attempt_weighted_itt_delta": sum(all_attempt_deltas) / len(all_attempt_deltas) if all_attempt_deltas else None,
            "positive_tie_negative_tasks": {"positive": sum(value > 0 for value in deltas), "tie": sum(value == 0 for value in deltas), "negative": sum(value < 0 for value in deltas)},
            "complete_valid_task_deltas": complete,
        }
    deltas = [comparisons[arm]["task_weighted_delta"] for arm in ARMS[1:]]
    monotonic = not errors and all(value is not None for value in deltas) and all(deltas[index] <= deltas[index + 1] for index in range(len(deltas) - 1))
    d4_task600 = [row for row in indexed.values() if row.get("arm") == "diversity-4" and int(row.get("goal_idx", -1)) == 600]
    continuation_tracking = [item for item in tracking_records if item["source"] == "continuation"]
    if sealed_receipt_hashes is not None:
        observed_receipts = {
            item["fallback_receipt"]["path"]: item["fallback_receipt"]["sha256"]
            for item in continuation_tracking
            if item["fallback_receipt"] is not None
        }
        if observed_receipts != sealed_receipt_hashes:
            errors.append("sealed fallback receipt manifest coverage mismatch")
    mode_counts = {mode: sum(item["mode"] == mode for item in continuation_tracking) for mode in ("online", "offline")}
    effective_mode = "mixed" if all(mode_counts.values()) else ("offline" if mode_counts["offline"] else "online")
    result = {
        "schema": "openevo.webshop-h1.38a-c-combined-reconciliation.v1",
        "status": "complete" if not errors else "invalid",
        "campaign_id": config["campaign_id"],
        "original_campaign_id": config["original_campaign"]["campaign_id"],
        "manifest_file_sha256": manifest_sha256,
        "expected_cell_count": 192,
        "observed_cell_count": observed_cells,
        "expected_attempt_count": len(expected),
        "observed_attempt_count": len(indexed),
        "source_cells": source_counts,
        "continuation_tracking": {
            "summary_count": len(continuation_tracking),
            "mode_counts": mode_counts,
            "effective_mode": effective_mode,
            "run_ids": sorted({item["run_id"] for item in continuation_tracking if isinstance(item["run_id"], str) and item["run_id"]}),
            "fallback_receipts": sorted(
                (item["fallback_receipt"] for item in continuation_tracking if item["fallback_receipt"]),
                key=lambda item: item["path"],
            ),
            "fallback_probes": sorted(
                {
                    item["fallback_probe"]["path"]: item["fallback_probe"]
                    for item in continuation_tracking
                    if item["fallback_probe"]
                }.values(),
                key=lambda item: item["path"],
            ),
            "offline_spools": sorted(
                ({"path": item["spool"], "sha256": item["spool_sha256"]} for item in continuation_tracking if item["mode"] == "offline"),
                key=lambda item: item["path"],
            ),
        },
        "arm_metrics": arm_metrics,
        "comparisons_vs_base": comparisons,
        "monotonicity": {"ordered_task_weighted_deltas": deltas, "non_decreasing": monotonic if not errors else None, "classification": "monotonic-nondecreasing" if monotonic else ("non-monotonic" if not errors else "not-evaluable-incomplete")},
        "original_d4_task600_failure_preserved": len(d4_task600) == 4 and all(bool(row.get("parser_or_fallback_failure")) and row.get("provenance_source") == "original" for row in d4_task600),
        "resampling": {"task_cluster_bootstrap_rounds": RESAMPLES, "task_level_sign_flip_rounds": RESAMPLES, "seed": RESAMPLE_SEED},
        "structural_errors": errors,
        "tracking_warnings": warnings,
        "claim_boundary": "Combined frozen-panel analysis only; not an upstream-v16 continual-learning result and not an adaptive continuation.\n",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def validate_cell(manifest_path: Path, config_path: Path, cell: Path, arm: str, task: int) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    manifest_sha256 = _file_sha256(manifest_path)
    continuation_root = cell.parents[2]
    execution_shas = _continuation_execution_shas(continuation_root, config_path, config, manifest_sha256, errors)
    sealed_receipt_hashes = _sealed_continuation_receipts(config_path, config, errors)
    rows = _cell_rows(
        cell,
        source="continuation",
        arm=arm,
        task=task,
        manifest_sha256=manifest_sha256,
        config=config,
        authorized_uuids=set(config.get("authorized_gpu_uuids", [])),
        continuation_execution_shas=execution_shas,
        original_fallback_probe=None,
        sealed_receipt_hashes=sealed_receipt_hashes,
        errors=errors,
        warnings=warnings,
    )
    return {"status": "complete" if rows is not None and not errors else "invalid", "attempt_count": len(rows or []), "structural_errors": errors, "tracking_warnings": warnings}


def validate_source(manifest_path: Path, config_path: Path, root: Path, source: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = [int(task) for task in manifest.get("evaluation_tasks", [])]
    errors: list[str] = []
    warnings: list[str] = []
    contract = _source_contract(config, tasks, errors)[source]
    _scan_evaluation_tree(root, source, contract, errors)
    authorized = set(config.get("authorized_gpu_uuids", []))
    manifest_sha256 = _file_sha256(manifest_path)
    execution_shas = _continuation_execution_shas(root, config_path, config, manifest_sha256, errors) if source == "continuation" else set()
    sealed_receipt_hashes = _sealed_continuation_receipts(config_path, config, errors)
    original_fallback_probe = _original_failed_online_probe(config_path, config) if source == "original" else None
    if source == "original" and original_fallback_probe is None:
        errors.append("original W&B online probe evidence invalid")
    validated = 0
    for arm in ARMS:
        for task in sorted(contract[arm]):
            cell = root / "evaluation" / arm / f"task{task}"
            if not cell.is_dir():
                if source == "original":
                    errors.append(f"source split mismatch: missing original cell {arm}/{task}")
                continue
            rows = _cell_rows(cell, source=source, arm=arm, task=task, manifest_sha256=manifest_sha256, config=config, authorized_uuids=authorized, continuation_execution_shas=execution_shas, original_fallback_probe=original_fallback_probe, sealed_receipt_hashes=sealed_receipt_hashes, errors=errors, warnings=warnings)
            validated += rows is not None
    return {"status": "complete" if not errors else "invalid", "validated_cells": validated, "structural_errors": errors, "tracking_warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--continuation-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-cell", type=Path)
    parser.add_argument("--validate-source", choices=("original", "continuation"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.validate_cell is not None:
        if args.arm is None or args.task is None:
            parser.error("--validate-cell requires --arm and --task")
        result = validate_cell(args.manifest.resolve(), args.config.resolve(), args.validate_cell.resolve(), args.arm, args.task)
    elif args.validate_source is not None:
        if args.source_root is None:
            parser.error("--validate-source requires --source-root")
        result = validate_source(args.manifest.resolve(), args.config.resolve(), args.source_root.resolve(), args.validate_source)
    else:
        if args.original_root is None or args.continuation_root is None or args.output is None:
            parser.error("combined analysis requires --original-root, --continuation-root, and --output")
        result = combined(args.manifest.resolve(), args.config.resolve(), args.original_root.resolve(), args.continuation_root.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "errors": result["structural_errors"], "observed_attempts": result.get("observed_attempt_count")}, sort_keys=True))
    raise SystemExit(0 if result["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
