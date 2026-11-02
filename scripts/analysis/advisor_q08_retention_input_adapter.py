#!/usr/bin/env python3
"""Build one Q08 lineage's normalized input from frozen owner receipts.

This adapter is read-only. It never infers learning from rollout/generation
counts: task exposure is joined through exact task identities, evidence IDs,
the SD-LoRA dataset manifest, the update receipt, and changed adapter-content
digests. Missing links are carried as audit gaps and block scientific analysis.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
from typing import Any
from pathlib import Path

try:
    from .advisor_q08_retention_analysis import PANEL_PATH, PANEL_SHA256
except ImportError:  # direct script execution
    from advisor_q08_retention_analysis import PANEL_PATH, PANEL_SHA256

MATRIX_PATH = Path(__file__).resolve().parents[2] / "docs/evidence/advisor-q08-retention-202610010523/checkpoint-matrix.json"
INPUT_SCHEMA = "openevo.q08.training-ledger-input.v1"
ABSENT_ADAPTER_SHA256 = hashlib.sha256(b"openevo:q08:no-active-adapter:v1").hexdigest()
SEEN_CANDIDATE_SHA256 = "56fb5ad9100b67bb16accd6cb55ebfb4ea0293582c1aab572b5ae9715618cd66"
SEEN_CANDIDATE_PATH = Path(__file__).resolve().parents[2] / "docs/evidence/advisor-q08-retention-202610010523/seen-cohort-candidate.json"
ORDINARY_EXECUTION_SHA = "ac130148ee08b6462d9728e482a858fe5f514047"
ORDINARY_CORE_POLICIES = {
    "af59656a27c24c8d5d4b40fc4cfdf52a80207aaf": "336fe29331aca08360cd8aa26a1d4013d546acdc6f2c31645aed974832c4c198",
    "59478929d884ca794cce2cb49cc6cc16103d6abe": "5a8cea229d463336df301dfe55c959a8c708fc5d3268017352674c3f81e414aa",
}


def _ordinary_source_contract(experiment_repo, core_repo, receipts):
    """Reproduce the audited historical predicate from exact Git object bytes.

    The pinned controller/launcher verified clean execution/core revisions before
    launching, and checked the returned core/policy identities. No current HEAD
    or mutable checkout is substituted for those historical objects.
    """
    if experiment_repo is None or core_repo is None:
        return None
    def blob(repo, revision, path, expected):
        data = subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"])
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"historical execution source hash mismatch: {path}")
        receipts.append({"role": "historical_execution_source", "path": path,
                         "revision": revision, "sha256": expected})
        return data.decode()
    for path, digest in {
        "scripts/openevo_webshop/ceiling1_stage2_sd_lora.py": "dc027da9459d0780fdc0d39bc4e39ad02c46acbae3ca6c12406fcd86b5b2b7a4",
        "scripts/openevo_webshop/run_ceiling1_stage2_vnext_arm.py": "8eb7d9ac3fbafab54ac23ae806a3422fbc67f80ebb0c5780eab746233802e963",
        "scripts/openevo_webshop/ceiling1_stage2_arm_controller.py": "1daeccc0a2a24b4684c5926b0293b88b79b465213ed5d2c3aad7cba65bdada06",
        "scripts/openevo_webshop/launch_ceiling1_stage2_sd_lora_container.sh": "13997bb2a7e599c905bbe037b707b4e7b243b9ccf2ff48901c6eecd849c453fd",
    }.items():
        blob(experiment_repo, ORDINARY_EXECUTION_SHA, path, digest)
    functions = {}
    for revision in ORDINARY_CORE_POLICIES:
        sd = blob(core_repo, revision, "src/openevo/evolution/parametric/sd_lora.py",
                  "52ec6cca94e597f39a875358c72a434e836af132847657128ab1b76ace55fe31")
        data = blob(core_repo, revision, "src/openevo/evolution/parametric/training_data.py",
                    "6b088284dd29deb9ae08afe805ca9581c85d0528c7fa5c1a3829f067b2ca3f4e")
        # Only the plain-text, tool-free shapes present in the old dataset are
        # admitted below. Execute the exact pinned helpers for those shapes;
        # unsupported shapes remain gaps rather than approximated semantics.
        nodes = [n for source in (sd, data) for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef)
                 and n.name in ("_successful_record", "_text", "normalize_chat_messages")]
        namespace = {"Any": Any, "math": math, "MAX_TRAINING_MESSAGES": 1024,
                     "_MAX_TEXT_BYTES": 4 * 1024 * 1024,
                     "_SUCCESS_STATUSES": frozenset(("completed", "pass", "passed", "success", "succeeded"))}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-q08-training-predicate", "exec"), namespace)
        functions[revision] = namespace
    return functions


def _ordinary_bound_execution(update, before, after, contract):
    core = update.get("openevo_core_execution_sha")
    policy = ORDINARY_CORE_POLICIES.get(core)
    return (contract is not None and core in contract and policy is not None
            and update.get("experiment_git_sha") == ORDINARY_EXECUTION_SHA
            and update.get("capacity_policy_sha256") == policy
            and all(state.get("capacity_core_sha") == core
                    and state.get("capacity_policy_sha256") == policy for state in (before, after)))


def _ordinary_retired_payloads(root, contract, receipts):
    """Index measured retired inputs and replacement hashes by actual path/ID."""
    index = {}
    required = {"adapter_config.json", "adapter_model.safetensors", "openevo_sd_lora_replay.jsonl",
                "openevo_sd_lora_state.json", "openevo_sd_lora_state.safetensors"}
    for r in range(160):
        d = root / f"round-{r:03d}"
        paths = [d / name for name in ("sd-lora/update.json", "state-before.json", "state-after.json", "ROUND_COMPLETE.json")]
        if not all(p.is_file() for p in paths):
            continue
        update, before, after, complete = [read_json(p) for p in paths]
        retirement_path = d / "sd-lora/artifacts/prior-input-retired.json"
        retired = update.get("prior_input_retirement")
        if not isinstance(retired, dict) or not retirement_path.is_file():
            continue
        removed = retired.get("files_removed", [])
        replacement = retired.get("replacement_identity", {})
        if (not _ordinary_bound_execution(update, before, after, contract)
                or (complete.get("parametric_sd_lora") or {}).get("update_receipt") != update
                or retired != read_json(retirement_path)
                or retired.get("schema") != "openevo.ceiling1-stage2-prior-input-retirement.v1"
                or retired.get("status") != "PASS"
                or retired.get("prior_input_path") != str(d / "sd-lora/artifacts/prior-input")
                or retired.get("replacement_adapter_path") != update.get("adapter_path")
                or len(removed) != 5 or {f.get("name") for f in removed} != required
                or set(replacement) != required
                or any(type(f.get("bytes")) is not int or f["bytes"] <= 0
                       or not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", ""))) for f in removed)
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in replacement.values())
                or before.get("current_sd_artifact_id") != update.get("prior_artifact_id")
                or after.get("current_sd_artifact_id") != update.get("artifact_id")
                or after.get("current_sd_adapter") != update.get("adapter_path")
                or replacement["adapter_model.safetensors"] != update.get("adapter_digest_after")):
            continue
        measured = next(f["sha256"] for f in removed if f["name"] == "adapter_model.safetensors")
        for state, digest in ((before, measured), (after, replacement["adapter_model.safetensors"])):
            path, aid = state.get("current_sd_adapter"), state.get("current_sd_artifact_id")
            matches = [c for c in state.get("sd_checkpoints", []) if c.get("path") == path]
            if not path or not aid or len(matches) != 1 or matches[0].get("adapter_sha256") != digest:
                continue
            key = (str(Path(path).resolve()), aid)
            if key in index and index[key] != digest:
                raise ValueError("conflicting measured retirement payload identities")
            index[key] = digest
        receipts.append({"role": "measured_prior_input_retirement", "round": r,
                         "path": str(retirement_path), "sha256": sha256(retirement_path)})
    return index


def _ordinary_training_keys(r, d, sd, dataset, update, evidence_path, evidence_by_id, contract, receipts):
    selection = sd.get("vnext_selection") or {}
    records_path = d / "sd-lora/dataset/records.jsonl"
    manifest_path = d / "sd-lora/dataset/manifest.json"
    if contract is None or update.get("openevo_core_execution_sha") not in contract or not records_path.is_file():
        return None
    records = list(read_jsonl(records_path))
    digest = sha256(records_path)
    if (selection.get("schema") != "openevo.ceiling1-stage2-vnext-parametric-selection.v1"
            or selection.get("status") != "DATASET_READY" or selection.get("round_index") != r
            or selection.get("manifest") != str(manifest_path) or selection.get("records") != str(records_path)
            or selection.get("manifest_sha256") != sha256(manifest_path)
            or selection.get("records_sha256") != digest or dataset.get("records_sha256") != digest
            or dataset.get("records_byte_size") != records_path.stat().st_size
            or selection.get("candidate_evidence_sha256") != sha256(evidence_path)
            or selection.get("bound_evidence_ids") != dataset.get("bound_evidence_ids")
            or selection.get("bound_task_identities") != dataset.get("bound_task_identities")
            or len(set(dataset["bound_task_identities"])) != len(dataset["bound_task_identities"])
            or len(records) != dataset.get("record_count") or len(records) != update.get("dataset_record_count")
            or len(records) != selection.get("selected_task_count")):
        return None
    cfg = update.get("native_user_config") or {}
    if (cfg.get("epochs") != 1 or cfg.get("per_device_train_batch_size") != 1
            or cfg.get("gradient_accumulation_steps") != 1 or cfg.get("max_steps") is not None
            or cfg.get("max_records") != 256 or cfg.get("minimum_reward") != 0.5):
        return None
    predicate = contract[update["openevo_core_execution_sha"]]
    keys, count, seen_evidence = set(), 0, set()
    for record in records:
        key = record.get("task_id")
        traces = record.get("traces")
        if key in keys or key not in dataset["bound_task_identities"] or not isinstance(traces, list) or not traces:
            return None
        if not predicate["_successful_record"](record, cfg["minimum_reward"]):
            return None
        for trace in traces:
            if not isinstance(trace, dict) or trace.get("tools") is not None:
                return None
            metadata = trace.get("metadata") or {}
            eid = metadata.get("source_evidence_id")
            evidence = evidence_by_id.get(eid)
            if (eid not in dataset["bound_evidence_ids"] or evidence is None
                    or evidence.get("task_identity") != key or metadata.get("source_task_identity") != key
                    or metadata.get("source_schedule_position") != evidence.get("schedule_position")):
                return None
            for field in ("prompt_messages", "response_messages"):
                messages = trace.get(field)
                if not isinstance(messages, list) or any(not isinstance(m, dict) or set(m) != {"role", "content"}
                                                        or not isinstance(m["content"], str) for m in messages):
                    return None
            try:
                prompt = predicate["normalize_chat_messages"](trace["prompt_messages"])
                response = predicate["normalize_chat_messages"](trace["response_messages"])
            except ValueError:
                return None
            if not prompt or not any(m["role"] == "assistant" and m["content"] for m in response):
                return None
            count += 1
            seen_evidence.add(eid)
        keys.add(key)
    replay, optimizer, steps = (update.get(k) for k in ("replay_example_count", "optimizer_example_count", "optimizer_steps"))
    if (keys != set(dataset["bound_task_identities"]) or seen_evidence != set(dataset["bound_evidence_ids"])
            or not 0 < count <= 256 or dataset.get("max_training_examples") != 256
            or count != dataset.get("trace_count") or count != selection.get("trace_count")
            or count != update.get("dataset_trace_count") or count != update.get("current_training_example_count")
            or type(replay) is not int or replay < 0 or type(optimizer) is not int
            or count + replay != optimizer or steps != optimizer):
        return None
    receipts.append({"role": "accepted_training_records", "round": r,
                     "path": str(records_path), "sha256": digest})
    return keys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object at {path}:{line_no}")
                yield value


def task_key(worker_seed, session_index, instruction_sha):
    if type(worker_seed) is not int or type(session_index) is not int:
        raise ValueError("integer worker_seed and source_session_index required")
    if not isinstance(instruction_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", instruction_sha):
        raise ValueError("64-character instruction SHA-256 required")
    return f"{worker_seed}:{session_index}:{instruction_sha}"


def _panel():
    if sha256(PANEL_PATH) != PANEL_SHA256:
        raise ValueError("frozen Q08 seen-panel bytes changed")
    panel = read_json(PANEL_PATH)
    tasks = panel.get("tasks", [])
    if len(tasks) != 96:
        raise ValueError("exact frozen 96-task panel required")
    by_key = {}
    normalized = []
    for task in tasks:
        key = task_key(task["worker_seed"], task["source_session_index"], task["instruction_sha256"])
        if key in by_key:
            raise ValueError("duplicate panel training identity")
        by_key[key] = task
        normalized.append({"task_id": task["task_identity"], "cohort": task["cohort"]})
    return panel, by_key, normalized


def _ledger_files(manifest_path: Path, lineage: str, training_root: Path, expected_manifest_sha256: str | None,
                  private_manifest_root: Path | None = None):
    manifest = read_json(manifest_path)
    is_pinned_candidate = (lineage == "ordinary" and manifest_path.resolve() == SEEN_CANDIDATE_PATH.resolve()
                           and sha256(manifest_path) == SEEN_CANDIDATE_SHA256)
    if is_pinned_candidate:
        rows = manifest.get("ordinary_ledger_files")
    elif (manifest.get("schema") == INPUT_SCHEMA and manifest.get("lineage") == lineage
          and re.fullmatch(r"[0-9a-f]{64}", str(expected_manifest_sha256 or ""))
          and sha256(manifest_path) == expected_manifest_sha256):
        rows = manifest.get("ledger_files")
    else:
        raise ValueError("ledger manifest schema/lineage mismatch or unpinned owner inventory")
    root = training_root.resolve(strict=True)
    if not isinstance(rows, list) or len(rows) != 160:
        raise ValueError("exactly 160 frozen attempts ledgers required")
    # The original Ordinary inventory is pinned by the complete manifest bytes;
    # its historical format predates the separate inventory digest field.
    if not is_pinned_candidate and (not isinstance(manifest.get("inventory_sha256"), str)
            or hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            != manifest["inventory_sha256"]):
        raise ValueError("frozen ledger inventory digest mismatch")
    by_round = {}
    for row in rows:
        r, path, expected = row.get("round"), Path(str(row.get("path") or "")), row.get("sha256")
        if type(r) is not int or not 0 <= r < 160 or r in by_round:
            raise ValueError("ledger rounds must uniquely cover 0..159")
        resolved = path.resolve(strict=True)
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or sha256(resolved) != expected:
            raise ValueError(f"frozen attempts ledger hash mismatch at round {r}")
        if resolved.parent.parent.resolve() != root:
            raise ValueError(f"attempts ledger is outside the declared lineage root: {resolved}")
        by_round[r] = resolved
    if set(by_round) != set(range(160)):
        raise ValueError("frozen attempts ledgers must cover every round 0..159")
    source_rows = (manifest.get("source_provenance") or {}).get("round_sources", [])
    round_sources = {}
    if lineage in ("bounded", "gdr"):
        provenance = manifest.get("source_provenance") or {}
        arm = "off" if lineage == "bounded" else "on"
        canonical_dataset = "openevo-recovery/openevo-bounded-effective-state-gdr-1p7b-raw-evidence"
        dataset_revision = "9be9e97700ac125d891e53798832c77836b51758"
        if (private_manifest_root is None or provenance.get("canonical_dataset") != canonical_dataset
                or provenance.get("dataset_revision") != dataset_revision
                or provenance.get("arm") != arm
                or manifest.get("source_root") != str(root)):
            raise ValueError("bounded/GDR input requires the pinned source identity and original PRIVATE_MANIFEST files")
        if not isinstance(source_rows, list) or len(source_rows) != 160:
            raise ValueError("bounded/GDR inputs require 160 private-manifest-bound round source rows")
        by_source_round = {}
        for row in source_rows:
            r = row.get("round")
            if type(r) is not int or r not in range(160) or r in by_source_round:
                raise ValueError("bounded/GDR round source rows must uniquely cover 0..159")
            if row.get("path") != str(by_round[r]) or row.get("sha256") != next(
                    item["sha256"] for item in rows if item["round"] == r):
                raise ValueError(f"round {r}: source row differs from its frozen attempts ledger")
            if type(row.get("size_bytes")) is not int or row["size_bytes"] < 0:
                raise ValueError(f"round {r}: source row lacks an attempts byte size")
            if by_round[r].stat().st_size != row["size_bytes"]:
                raise ValueError(f"round {r}: attempts byte size differs from the frozen source row")
            if not re.fullmatch(r"[0-9a-f]{40}", str(row.get("execution_git_sha") or "")):
                raise ValueError(f"round {r}: source execution commit is not pinned")
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("private_manifest_sha256") or "")):
                raise ValueError(f"round {r}: private manifest SHA-256 is not pinned")
            members = row.get("private_manifest_members")
            if not isinstance(members, dict):
                raise ValueError(f"round {r}: private manifest member hashes are missing")
            uri = (f"hf://datasets/{canonical_dataset}@{dataset_revision}/{arm}/"
                   f"round-{r:03d}/PRIVATE_MANIFEST.json")
            if row.get("private_manifest_path") != uri:
                raise ValueError(f"round {r}: private-manifest URI is not the pinned canonical source")
            private_manifest = private_manifest_root / arm / f"round-{r:03d}" / "PRIVATE_MANIFEST.json"
            if (not private_manifest.is_file() or sha256(private_manifest) != row["private_manifest_sha256"]):
                raise ValueError(f"round {r}: original private-manifest bytes are missing or changed")
            private_data = read_json(private_manifest)
            declared_members = {item.get("path"): item for item in private_data.get("private_members", [])
                                if isinstance(item, dict)}
            if len(declared_members) != len(private_data.get("private_members", [])):
                raise ValueError(f"round {r}: private manifest contains duplicate or malformed members")
            expected_round_path = str(root / f"round-{r:03d}")
            if (private_data.get("arm") != arm or private_data.get("round") != r
                    or private_data.get("source_round") != expected_round_path
                    or private_data.get("schema") != "openevo.bounded-private-recovery-round.v1"):
                raise ValueError(f"round {r}: private-manifest identity mismatch")
            for name, item in members.items():
                declared = declared_members.get(name)
                if (not isinstance(item, dict) or not isinstance(declared, dict)
                        or item.get("sha256") != declared.get("sha256")
                        or item.get("size_bytes") != declared.get("size")):
                    raise ValueError(f"round {r}: inventory member is not bound by original private manifest: {name}")
                if (not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256") or ""))
                        or type(item.get("size_bytes")) is not int or item["size_bytes"] < 0):
                    raise ValueError(f"round {r}: invalid private-manifest member digest or size: {name}")
            attempts_member = declared_members.get("attempts.jsonl")
            if (not isinstance(attempts_member, dict) or attempts_member.get("sha256") != row["sha256"]
                    or attempts_member.get("size") != row["size_bytes"]):
                raise ValueError(f"round {r}: attempts ledger is not bound by original private manifest")
            by_source_round[r] = row
        if set(by_source_round) != set(range(160)):
            raise ValueError("bounded/GDR source rows must cover every round 0..159")
        round_sources = by_source_round
    elif isinstance(source_rows, list):
        round_sources = {row["round"]: row for row in source_rows if type(row.get("round")) is int}
    return manifest, by_round, round_sources


def _private_member(round_sources: dict, round_index: int, round_dir: Path, relative_path: str):
    """Return one local file only when its frozen private-manifest hash/size match."""
    source = round_sources[round_index]
    member = source["private_manifest_members"].get(relative_path)
    if not isinstance(member, dict):
        raise ValueError(f"round {round_index}: private manifest does not bind {relative_path}")
    path = round_dir / relative_path
    if not path.is_file():
        return None
    if (path.stat().st_size != member.get("size_bytes")
            or sha256(path) != member.get("sha256")):
        raise ValueError(f"round {round_index}: private-manifest member mismatch: {relative_path}")
    return path


def _bounded_gdr_round(lineage: str, round_index: int, round_dir: Path, source_rows: dict,
                       panel_by_key: dict, attempts_by_position: dict, expected_previous_state: str | None,
                       events: list,
                       audit_gaps: list, source_receipts: list, adapter_source_receipts: list):
    """Validate the historical Bounded/GDR receipt chain without guessing trainer semantics."""
    required = ("ROUND_COMPLETE.json", "state-before.json", "state-after.json",
                "sd-lora/terminal-receipt.json", "sd-lora/dataset/selection.json")
    paths = {}
    for name in required:
        paths[name] = _private_member(source_rows, round_index, round_dir, name)
    if any(path is None for path in paths.values()):
        audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_REQUIRED_PRIVATE_MEMBER_MISSING"})
        return
    complete = read_json(paths["ROUND_COMPLETE.json"])
    terminal = read_json(paths["sd-lora/terminal-receipt.json"])
    state_before = read_json(paths["state-before.json"])
    state_after = read_json(paths["state-after.json"])
    state = complete.get("parametric_bounded_state") or {}
    status = state.get("status")
    arm_id = "BOUNDED_OFF" if lineage == "bounded" else "BOUNDED_EFFECTIVE_STATE_ON"
    if (complete.get("status") != "PASS" or complete.get("round_index") != round_index
            or complete.get("experiment_arm_id") != arm_id
            or complete.get("execution_git_sha") != source_rows[round_index]["execution_git_sha"]
            or state_before.get("state_id") != complete.get("state_id_before")
            or state_after.get("state_id") != complete.get("state_id_after")
            or (round_index > 0 and state_before.get("state_id") != expected_previous_state)
            or status not in ("UPDATE", "NOOP") or terminal.get("status") != status
            or terminal.get("round_index") != round_index
            or terminal.get("treatment_mode") != complete.get("treatment_mode")):
        audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_TERMINAL_STATUS_MISMATCH"})
        return
    for name, path in paths.items():
        source_receipts.append({"role": "private_manifest_bound_source", "round": round_index,
                                "path": str(path), "sha256": sha256(path)})
    if status == "NOOP":
        # NOOP rounds intentionally have no dataset or recurrence receipt.
        selection = read_json(paths["sd-lora/dataset/selection.json"])
        before_adapter, after_adapter = state_before.get("current_sd_adapter"), state_after.get("current_sd_adapter")
        before_artifact, after_artifact = state_before.get("current_sd_artifact_id"), state_after.get("current_sd_artifact_id")
        before_digest = sha256(Path(before_adapter).resolve(strict=True) / "adapter_model.safetensors") if before_adapter else None
        after_digest = sha256(Path(after_adapter).resolve(strict=True) / "adapter_model.safetensors") if after_adapter else None
        if after_digest is not None:
            adapter_source_receipts.append({"role": "active_sd_adapter_payload", "round": round_index,
                                            "path": str(Path(after_adapter).resolve(strict=True) / "adapter_model.safetensors"),
                                            "sha256": after_digest})
        member_names = set(source_rows[round_index]["private_manifest_members"])
        if (terminal.get("reason") != "NOOP_NO_SUCCESS" or terminal.get("replay_preserved") is not True
                or selection.get("status") != "NOOP_NO_SUCCESS" or selection.get("round_index") != round_index
                or selection.get("attempts_sha256") != source_rows[round_index]["sha256"]
                or terminal.get("reason") != selection.get("status")
                or any(name in member_names for name in ("sd-lora/dataset/manifest.json",
                                                         "sd-lora/dataset/records.jsonl",
                                                         "sd-lora/bounded-state/bounded_recurrence_receipt.json"))
                or before_adapter != after_adapter or before_artifact != after_artifact
                or terminal.get("adapter_path") != after_adapter
                or terminal.get("artifact_id") != after_artifact
                or terminal.get("adapter_sha256") != before_digest
                or before_digest != after_digest):
            audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_NOOP_RECEIPT_UNPROVEN"})
        return state_after.get("state_id")

    extra_names = ("sd-lora/dataset/manifest.json", "sd-lora/dataset/records.jsonl",
                   "sd-lora/bounded-state/bounded_recurrence_receipt.json")
    for name in extra_names:
        paths[name] = _private_member(source_rows, round_index, round_dir, name)
    if any(path is None for name, path in paths.items() if name in extra_names):
        audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_UPDATE_DATASET_OR_RECURRENCE_MISSING"})
        return
    dataset = read_json(paths["sd-lora/dataset/manifest.json"])
    selection = read_json(paths["sd-lora/dataset/selection.json"])
    recurrence = read_json(paths["sd-lora/bounded-state/bounded_recurrence_receipt.json"])
    records = list(read_jsonl(paths["sd-lora/dataset/records.jsonl"]))
    records_digest = sha256(paths["sd-lora/dataset/records.jsonl"])
    if (dataset.get("records_sha256") != records_digest
            or dataset.get("records_byte_size") != paths["sd-lora/dataset/records.jsonl"].stat().st_size
            or selection.get("records_sha256") != records_digest
            or selection.get("manifest_sha256") != sha256(paths["sd-lora/dataset/manifest.json"])
            or terminal.get("dataset_selection_sha256") != sha256(paths["sd-lora/dataset/selection.json"])
            or recurrence.get("dataset_records_sha256") != records_digest):
        audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_DATASET_HASH_CHAIN_MISMATCH"})
        return
    selected = selection.get("selected_schedule_positions")
    if (selection.get("status") != "DATASET_READY" or selection.get("round_index") != round_index
            or selection.get("attempts_sha256") != source_rows[round_index]["sha256"]
            or not isinstance(selected, list) or any(type(pos) is not int for pos in selected)
            or len(set(selected)) != len(selected)):
        audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_SELECTION_POSITIONS_INVALID"})
        return
    if recurrence.get("status") != "PASS" or type(recurrence.get("current_training_example_count")) is not int:
        audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_RECURRENCE_RECEIPT_INVALID"})
        return
    training_keys = set()
    observed_examples = 0
    positions = attempts_by_position
    observed_positions = set()
    for record in records:
        record_key = record.get("task_id")
        traces = record.get("traces")
        if not isinstance(record_key, str) or not isinstance(traces, list):
            audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_RECORD_SCHEMA_INVALID"})
            return
        if record_key in panel_by_key:
            training_keys.add(record_key)
        for trace in traces:
            metadata = trace.get("metadata") or {}
            position = metadata.get("source_schedule_position")
            row = positions.get(position)
            if (type(position) is not int or position not in selected or row is None
                    or metadata.get("source_task_identity") != record_key
                    or record_key != task_key(row.get("worker_seed"), row.get("source_session_index"),
                                              row.get("instruction_sha256"))
                    or row.get("qualification_only") is not False
                    or row.get("task_bearing_budget_consumed") is not True
                    or row.get("engineering_invalid") is not False
                    or row.get("identity_ok") is not True or row.get("trace_complete") is not True
                    or row.get("parser_or_fallback_failure") is not False
                    or row.get("parser_repair_used") is not False
                    or row.get("error") is not None or row.get("terminated_on_invalid_action") is not False
                    or row.get("won") is not True or row.get("final_task_score") != 1.0):
                audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_TRACE_NOT_JOINED_TO_CLEAN_CONSUMED_ATTEMPT"})
                return
            observed_examples += 1
            observed_positions.add(position)
    if observed_positions != set(selected):
        audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_SELECTED_POSITIONS_NOT_EXACTLY_REPRESENTED"})
        return
    if (dataset.get("record_count") != len(records)
            or dataset.get("trace_count") != observed_examples
            or recurrence.get("current_training_example_count") != observed_examples
            or recurrence.get("round_index") != round_index
            or recurrence.get("treatment_mode") != state.get("treatment_mode")
            or terminal.get("round_index") != round_index
            or terminal.get("treatment_mode") != state.get("treatment_mode")
            or terminal.get("bounded_receipt_sha256") != sha256(paths["sd-lora/bounded-state/bounded_recurrence_receipt.json"])):
        audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_TRAINING_EXAMPLE_COUNT_MISMATCH"})
        return
    before_adapter = state_before.get("current_sd_adapter")
    after_adapter = state_after.get("current_sd_adapter")
    before_digest = sha256(Path(before_adapter).resolve(strict=True) / "adapter_model.safetensors") if before_adapter else None
    after_digest = sha256(Path(after_adapter).resolve(strict=True) / "adapter_model.safetensors")
    adapter_source_receipts.append({"role": "active_sd_adapter_payload", "round": round_index,
                                    "path": str(Path(after_adapter).resolve(strict=True) / "adapter_model.safetensors"),
                                    "sha256": after_digest})
    if (terminal.get("adapter_path") != after_adapter
            or terminal.get("artifact_id") != state_after.get("current_sd_artifact_id")
            or terminal.get("adapter_sha256") != after_digest
            or recurrence.get("output_adapter_sha256") != after_digest
            or recurrence.get("prior_adapter_sha256") != before_digest
            or before_digest == after_digest):
        audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_ACTIVE_ADAPTER_TRANSITION_UNPROVEN"})
        return state_after.get("state_id")
    # The frozen receipt proves records were produced, but not the historical trainer's
    # exact inclusion predicate. Keep the overall input PARTIAL until that source contract
    # is independently bound; still expose the observed identities without claiming effect.
    audit_gaps.append({"round": round_index, "reason": "BOUNDED_GDR_TRAINER_INCLUSION_PREDICATE_UNBOUND"})
    for key in sorted(training_keys):
        events.append({"task_id": panel_by_key[key]["task_identity"], "round": round_index,
                       "update_id": f"r{round_index:03d}:bounded-state", "rollout_consumed": True,
                       "training_included": True, "accepted_update": True,
                       "state_before": state_before.get("state_id"), "state_after": state_after.get("state_id"),
                       "training_inclusion_verified": False})
    for name in extra_names:
        path = paths[name]
        source_receipts.append({"role": "private_manifest_bound_source", "round": round_index,
                                "path": str(path), "sha256": sha256(path)})
    return state_after.get("state_id")


def normalize(lineage: str, training_root: Path, ledger_manifest: Path, evaluation_root: Path,
              expected_manifest_sha256: str | None = None,
              private_manifest_root: Path | None = None,
              experiment_source_repo: Path | None = None, core_source_repo: Path | None = None) -> dict:
    panel, panel_by_key, tasks = _panel()
    matrix = read_json(MATRIX_PATH)
    if sha256(MATRIX_PATH) != "93ad7718a07633e38e15197f43ea21649fac84abc650aa29fad7f3107415f89d":
        raise ValueError("frozen checkpoint matrix bytes changed")
    ledger, attempts_by_round, round_sources = _ledger_files(
        ledger_manifest, lineage, training_root, expected_manifest_sha256, private_manifest_root)

    planned = [40, 80, 120, 160]
    cells = [c for c in matrix["cells"] if c.get("lineage") == lineage and c.get("completed_rounds") in planned]
    if len(cells) != 4:
        raise ValueError("checkpoint matrix must explicitly list all four planned points")
    statuses = {}
    for cell in cells:
        status = cell.get("status")
        statuses[str(cell["completed_rounds"])] = "AVAILABLE" if status == "PASS" else "MISSING_PAYLOAD" if status.startswith("MISSING_PAYLOAD") else None
    if any(value is None for value in statuses.values()):
        raise ValueError("checkpoint matrix contains an unrecognized cell status")

    # First record actual consumed rollouts by exact frozen task identity.
    consumed_rounds = {}
    raw_rows = {}
    consumed_source_rows = {}
    attempts_by_position = {}
    for round_index, path in attempts_by_round.items():
        position_index = {}
        for row in read_jsonl(path):
            position = row.get("schedule_position")
            if type(position) is int:
                if position in position_index:
                    raise ValueError(f"round {round_index}: duplicate source schedule position {position}")
                position_index[position] = row
            if row.get("qualification_only") is True or row.get("task_bearing_budget_consumed") is not True:
                continue
            key = task_key(row.get("worker_seed"), row.get("source_session_index"), row.get("instruction_sha256"))
            consumed_source_rows.setdefault((key, round_index), []).append(row)
            if key not in panel_by_key:
                continue
            consumed_rounds.setdefault(key, set()).add(round_index)
            raw_rows.setdefault((key, round_index), []).append(row)
        attempts_by_position[round_index] = position_index

    events = []
    audit_gaps = []
    adapter_sha_cache = {}
    adapter_receipt_paths = set()
    adapter_source_receipts = []
    source_receipts = []
    ordinary_contract = (_ordinary_source_contract(experiment_source_repo, core_source_repo, source_receipts)
                         if lineage == "ordinary" else None)
    retired_payloads = (_ordinary_retired_payloads(training_root, ordinary_contract, adapter_source_receipts)
                       if lineage == "ordinary" else {})
    source_receipts.extend({"role": "rollout_attempts", "round": r, "path": str(p), "sha256": sha256(p)}
                           for r, p in sorted(attempts_by_round.items()))
    if lineage in ("bounded", "gdr"):
        arm = "off" if lineage == "bounded" else "on"
        for r, source in sorted(round_sources.items()):
            private_manifest = private_manifest_root / arm / f"round-{r:03d}" / "PRIVATE_MANIFEST.json"
            source_receipts.append({"role": "private_manifest", "round": r, "path": str(private_manifest),
                                    "sha256": sha256(private_manifest),
                                    "size_bytes": private_manifest.stat().st_size})
    event_ids = set()
    previous_state_id = None
    for key, round_index in sorted(raw_rows, key=lambda item: (item[1], item[0])):
        event_id = f"rollout-r{round_index:03d}"
        pair = (key, event_id, round_index)
        if pair in event_ids:
            raise ValueError("duplicate rollout identity")
        event_ids.add(pair)
        events.append({"task_id": panel_by_key[key]["task_identity"], "round": round_index,
                       "update_id": event_id, "rollout_consumed": True,
                       "training_included": False, "accepted_update": False})

    # Join each parameter update to its exact dataset/evidence and state receipts.
    for round_index in range(160):
        round_dir = training_root / f"round-{round_index:03d}"
        if lineage in ("bounded", "gdr"):
            previous_state_id = _bounded_gdr_round(lineage, round_index, round_dir, round_sources,
                               panel_by_key, attempts_by_position[round_index], previous_state_id,
                               events, audit_gaps, source_receipts, adapter_source_receipts)
            if previous_state_id is None:
                after_path = round_dir / "state-after.json"
                if after_path.is_file():
                    previous_state_id = read_json(after_path).get("state_id")
            continue
        complete_path = round_dir / "ROUND_COMPLETE.json"
        evidence_path = round_dir / "evidence/candidate-evidence-unconsumed.jsonl"
        dataset_path = round_dir / "sd-lora/dataset/manifest.json"
        update_path = round_dir / "sd-lora/update.json"
        before_path, after_path = round_dir / "state-before.json", round_dir / "state-after.json"
        if not complete_path.is_file():
            audit_gaps.append({"round": round_index, "reason": "MISSING_ROUND_COMPLETE"})
            continue
        complete = read_json(complete_path)
        sd = complete.get("parametric_sd_lora") or {}
        if sd.get("status") not in ("UPDATE", "NOOP"):
            audit_gaps.append({"round": round_index, "reason": "ORDINARY_TERMINAL_STATUS_UNPROVEN"})
            continue
        update_expected = sd.get("status") == "UPDATE"
        if before_path.is_file() and after_path.is_file():
            before_state, after_state = read_json(before_path), read_json(after_path)
            if (round_index > 0 and before_state.get("state_id") != previous_state_id
                    or before_state.get("state_id") is None or after_state.get("state_id") is None):
                audit_gaps.append({"round": round_index, "reason": "ORDINARY_STATE_CONTINUITY_UNPROVEN"})
            previous_state_id = after_state.get("state_id")
        else:
            audit_gaps.append({"round": round_index, "reason": "ORDINARY_STATE_RECEIPT_MISSING"})
        if sd.get("status") == "NOOP":
            selection = sd.get("selection") or {}
            if (selection.get("status") != "NOOP_NO_UNCONSUMED_CLEAN_EXACT"
                    or not re.fullmatch(r"[0-9a-f]{64}", str(sd.get("output_sha256") or ""))
                    or selection.get("bound_evidence_ids") or selection.get("bound_task_identities")
                    or not before_path.is_file() or not after_path.is_file()
                    or before_state.get("current_sd_adapter") != after_state.get("current_sd_adapter")
                    or before_state.get("current_sd_artifact_id") != after_state.get("current_sd_artifact_id")
                    or sd.get("adapter_path") != after_state.get("current_sd_adapter")
                    or sd.get("artifact_id") != after_state.get("current_sd_artifact_id")
                    or sd.get("prior_sha256") != sd.get("output_sha256")
                    or retired_payloads.get((str(Path(sd.get("adapter_path") or "").resolve()), sd.get("artifact_id")))
                    != sd.get("output_sha256")):
                audit_gaps.append({"round": round_index, "reason": "ORDINARY_NOOP_IDENTITY_UNPROVEN"})
            continue
        if not dataset_path.is_file() or not update_path.is_file():
            if update_expected:
                audit_gaps.append({"round": round_index, "reason": "UPDATE_MISSING_DATASET_OR_RECEIPT"})
            continue
        dataset, update = read_json(dataset_path), read_json(update_path)
        receipt_paths = [complete_path, dataset_path, update_path, before_path, after_path]
        if evidence_path.is_file():
            receipt_paths.append(evidence_path)
        source_receipts.extend({"role": path.name, "round": round_index, "path": str(path), "sha256": sha256(path)}
                               for path in receipt_paths if path.is_file())
        evidence_by_id = {}
        if evidence_path.is_file():
            for row in read_jsonl(evidence_path):
                evidence_by_id[row.get("evidence_id")] = row
        evidence_ids = set(dataset.get("bound_evidence_ids", []))
        bound_keys = set(dataset.get("bound_task_identities", []))
        if not evidence_ids and not bound_keys:
            if update_expected:
                audit_gaps.append({"round": round_index, "reason": "UPDATE_DATASET_HAS_NO_BINDING_FIELDS"})
            continue
        if not evidence_path.is_file() and update_expected:
            audit_gaps.append({"round": round_index, "reason": "UPDATE_MISSING_CANDIDATE_EVIDENCE"})
            continue
        selected_evidence = [row for eid, row in evidence_by_id.items() if eid in evidence_ids]
        derived_keys = {row.get("task_identity") for row in selected_evidence}
        if derived_keys != bound_keys:
            raise ValueError(f"round {round_index}: dataset evidence IDs do not exactly bind its task identities")
        for evidence in selected_evidence:
            source_round = evidence.get("source_round")
            source_key = evidence.get("task_identity")
            source_path = evidence.get("source_attempts_path")
            if (type(source_round) is not int or source_round > round_index
                    or attempts_by_round.get(source_round) is None
                    or Path(str(source_path or "")).resolve() != attempts_by_round[source_round]
                    or (source_key, source_round) not in consumed_source_rows):
                raise ValueError(f"round {round_index}: selected evidence is not bound to a consumed source attempt")
            task_id = evidence.get("task_id")
            source_rows = [row for row in consumed_source_rows[(source_key, source_round)] if row.get("task_id") == task_id]
            if (evidence.get("evidence_type") != "clean_exact_trajectory" or evidence.get("source_final_score") != 1.0
                    or not source_rows or any(row.get("engineering_invalid") is not False or row.get("trace_complete") is not True
                                      or row.get("identity_ok") is not True
                                      or row.get("schedule_position") != evidence.get("schedule_position")
                                      or row.get("won") is not True or row.get("final_task_score") != 1.0
                                      or row.get("parser_or_fallback_failure") is not False
                                      or row.get("parser_repair_used") is not False
                                      or row.get("terminated_on_invalid_action") is not False
                                      or row.get("error") is not None for row in source_rows)):
                raise ValueError(f"round {round_index}: selected evidence source attempt is invalid or incomplete")
        panel_bound_keys = bound_keys.intersection(panel_by_key)
        if any(key not in consumed_rounds or not any(r <= round_index for r in consumed_rounds[key]) for key in panel_bound_keys):
            raise ValueError(f"round {round_index}: dataset includes a task without a consumed source rollout")
        if sd.get("status") != update.get("status") or sd.get("update_receipt") != update:
            raise ValueError(f"round {round_index}: update receipt differs from ROUND_COMPLETE")
        if before_path.is_file() and after_path.is_file():
            before, after = read_json(before_path), read_json(after_path)
            state_before, state_after = before.get("state_id"), after.get("state_id")
        else:
            before = after = {}
            state_before = state_after = None
        bound_execution = _ordinary_bound_execution(update, before, after, ordinary_contract)
        training_keys = _ordinary_training_keys(round_index, round_dir, sd, dataset, update,
                                               evidence_path, evidence_by_id, ordinary_contract, source_receipts)
        if update_expected and (not bound_execution or training_keys is None):
            audit_gaps.append({"round": round_index, "reason": "ORDINARY_TRAINING_INCLUSION_CHAIN_UNPROVEN"})
        def active_adapter_digest(state):
            checkpoints = state.get("sd_checkpoints")
            adapter_path = state.get("current_sd_adapter")
            if adapter_path is None:
                if ("current_sd_adapter" in state and state.get("current_sd_artifact_id") is None
                        and state.get("sd_component_count") == 0
                        and checkpoints == []):
                    return ABSENT_ADAPTER_SHA256
                return None
            if not isinstance(adapter_path, str) or not adapter_path:
                return None
            matches = [item for item in checkpoints or []
                       if item.get("path") and Path(item["path"]).resolve() == Path(adapter_path).resolve()]
            if len(matches) != 1:
                return None
            declared = matches[0].get("adapter_sha256")
            model_file = Path(adapter_path).resolve() / "adapter_model.safetensors"
            if not model_file.is_file():
                measured = retired_payloads.get((str(Path(adapter_path).resolve()), state.get("current_sd_artifact_id")))
                return measured if measured is not None and measured == declared else None
            cache_key = str(model_file)
            if cache_key not in adapter_sha_cache:
                adapter_sha_cache[cache_key] = sha256(model_file)
            actual = adapter_sha_cache[cache_key]
            if cache_key not in adapter_receipt_paths:
                adapter_source_receipts.append({"role": "active_sd_adapter_payload", "round": round_index,
                                                "path": cache_key, "sha256": actual})
                adapter_receipt_paths.add(cache_key)
            return actual if re.fullmatch(r"[0-9a-f]{64}", str(declared or "")) and actual == declared else None
        before_digest, after_digest = active_adapter_digest(before), active_adapter_digest(after)
        update_digest = update.get("adapter_digest_after")
        digest_valid = (re.fullmatch(r"[0-9a-f]{64}", str(update_digest or ""))
                        and after_digest == update_digest
                        and after.get("current_sd_artifact_id") == update.get("artifact_id")
                        and after.get("current_sd_adapter")
                        and Path(after["current_sd_adapter"]).resolve() == Path(update.get("adapter_path", "")).resolve())
        if update_expected and (not digest_valid or before_digest is None):
            audit_gaps.append({"round": round_index, "reason": "UPDATE_ACTIVE_ADAPTER_CONTENT_HASH_UNPROVEN"})
        accepted = (update.get("status") == "UPDATE" and bound_execution and training_keys is not None and digest_valid
                    and state_before is not None and state_after is not None and state_before != state_after
                    and before_digest is not None and before_digest != after_digest and after_digest is not None)
        receipt_sha = sha256(update_path)
        update_id = f"r{round_index:03d}:{update.get('artifact_id') or 'sd-lora'}"
        for key in sorted(panel_bound_keys):
            task_id = panel_by_key[key]["task_identity"]
            events.append({"task_id": task_id, "round": round_index, "update_id": update_id,
                           "rollout_consumed": True, "training_included": bool(training_keys is not None and key in training_keys),
                           "accepted_update": bool(accepted), "state_before": state_before,
                           "state_after": state_after, "adapter_digest_before": before_digest,
                           "adapter_digest_after": after_digest, "update_receipt_sha256": receipt_sha})

    scores = []
    for completed_rounds in planned:
        if statuses[str(completed_rounds)] != "AVAILABLE":
            continue
        path = evaluation_root / "cells" / f"{lineage}-r{completed_rounds:03d}" / "seen-eval/attempts.jsonl"
        if not path.is_file():
            continue
        source_receipts.append({"role": "q08_seen_evaluation", "completed_rounds": completed_rounds,
                                "path": str(path), "sha256": sha256(path)})
        for row in read_jsonl(path):
            task_id = row.get("task_identity")
            if task_id not in {task["task_id"] for task in tasks}:
                raise ValueError(f"unknown task in Q08 evaluation output: {task_id}")
            episode = row.get("episode") or {}
            valid = row.get("engineering_invalid") is False
            # The frozen q08_seen_checkpoint_probe emits Task Score as episode.score.
            score = episode.get("score")
            if valid and type(score) not in (int, float):
                raise ValueError("valid evaluation row lacks numeric episode Task Score")
            scores.append({"task_id": task_id, "completed_rounds": completed_rounds,
                           "score": score if valid else None, "engineering_valid": valid,
                           "evaluation_seed": row.get("paired_generation_seed")})

    return {"panel_role": "development", "lineage_id": lineage,
            "planned_checkpoints": planned, "late_checkpoint": 160,
            "checkpoint_status": statuses, "tasks": tasks, "events": events, "scores": scores,
            "audit_status": "PARTIAL" if audit_gaps else "COMPLETE",
            "audit_gaps": audit_gaps,
            "source_provenance": {"panel_sha256": PANEL_SHA256,
                                  "checkpoint_matrix_sha256": sha256(MATRIX_PATH),
                                  "ledger_manifest_sha256": sha256(ledger_manifest),
                                  "ledger_inventory_sha256": ledger.get("inventory_sha256"),
                                  "lineage_root": str(training_root.resolve()),
                                  "evaluation_root": str(evaluation_root.resolve()),
                                  "source_receipts": source_receipts + adapter_source_receipts}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", choices=("ordinary", "bounded", "gdr"), required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--ledger-manifest", type=Path, required=True,
                        help="frozen openevo.q08.training-ledger-input.v1 JSON with 160 path/hash rows")
    parser.add_argument("--ledger-manifest-sha256",
                        help="required hash for lineage manifests other than the pinned Ordinary owner inventory")
    parser.add_argument("--private-manifest-root", type=Path,
                        help="local HF-exported root containing {off,on}/round-NNN/PRIVATE_MANIFEST.json")
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--experiment-source-repo", type=Path,
                        help="Git object store containing the audited historical Ordinary launcher/driver revision")
    parser.add_argument("--core-source-repo", type=Path,
                        help="Git object store containing the two audited historical Ordinary Core revisions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = normalize(args.lineage, args.training_root, args.ledger_manifest, args.evaluation_root,
                        args.ledger_manifest_sha256, args.private_manifest_root,
                        args.experiment_source_repo, args.core_source_repo)
    if str(args.output) == "-":
        print(json.dumps(payload, ensure_ascii=False))
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": "NORMALIZED_INPUT_WRITTEN", "lineage": args.lineage,
                          "events": len(payload["events"]), "scores": len(payload["scores"]),
                          "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
