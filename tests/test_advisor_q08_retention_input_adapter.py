import hashlib
import json

import pytest

from scripts.analysis.advisor_q08_retention_analysis import frozen_tasks
from scripts.analysis.advisor_q08_retention_input_adapter import INPUT_SCHEMA, PANEL_PATH, normalize
from scripts.analysis import advisor_q08_retention_input_adapter as adapter


@pytest.fixture(autouse=True)
def isolated_source_contract(monkeypatch):
    # Source-object qualification is exercised with real historical inputs.
    # These unit fixtures isolate the receipt joins and use plain chat records.
    monkeypatch.setattr(adapter, "_ordinary_source_contract", lambda *args: {
        core: {"_successful_record": lambda record, minimum: record.get("reward", 0) >= minimum,
               "normalize_chat_messages": lambda messages: messages}
        for core in adapter.ORDINARY_CORE_POLICIES})


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def fixture_inputs(tmp_path, *, changed_state=True):
    root = tmp_path / "ordinary"
    task = frozen_tasks()[0]
    panel = json.loads(PANEL_PATH.read_text())
    source = next(t for t in panel["tasks"] if t["task_identity"] == task["task_id"])
    attempt = {"worker_seed": source["worker_seed"], "source_session_index": source["source_session_index"],
               "instruction_sha256": source["instruction_sha256"], "qualification_only": False,
               "task_bearing_budget_consumed": True, "engineering_invalid": False,
               "trace_complete": True, "identity_ok": True, "task_id": "source-task-r005",
               "schedule_position": 25, "won": True, "final_task_score": 1.0,
               "parser_or_fallback_failure": False, "parser_repair_used": False,
               "terminated_on_invalid_action": False, "error": None}
    ledger_rows = []
    for round_index in range(160):
        path = root / f"round-{round_index:03d}" / "attempts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(attempt) + "\n" if round_index == 5 else "", encoding="utf-8")
        ledger_rows.append({"round": round_index, "path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    key = f"{source['worker_seed']}:{source['source_session_index']}:{source['instruction_sha256']}"
    evidence_id = "a" * 64
    round_dir = root / "round-022"
    adapter_path_before = str(root / "round-021/adapter")
    adapter_path_after = str(root / "round-022/adapter")
    before_model = root / "round-021/adapter/adapter_model.safetensors"
    after_model = root / "round-022/adapter/adapter_model.safetensors"
    before_model.parent.mkdir(parents=True, exist_ok=True)
    after_model.parent.mkdir(parents=True, exist_ok=True)
    before_model.write_bytes(b"old adapter")
    after_model.write_bytes(b"new adapter" if changed_state else b"old adapter")
    before_digest = hashlib.sha256(before_model.read_bytes()).hexdigest()
    after_digest = hashlib.sha256(after_model.read_bytes()).hexdigest()
    core = next(iter(adapter.ORDINARY_CORE_POLICIES))
    update = {"status": "UPDATE", "artifact_id": "update-22", "adapter_path": adapter_path_after,
              "adapter_digest_after": after_digest, "prior_artifact_id": "update-21", "experiment_git_sha": adapter.ORDINARY_EXECUTION_SHA,
              "openevo_core_execution_sha": core, "capacity_policy_sha256": adapter.ORDINARY_CORE_POLICIES[core],
              "dataset_record_count": 1, "dataset_trace_count": 1, "current_training_example_count": 1,
              "replay_example_count": 0, "optimizer_example_count": 1, "optimizer_steps": 1,
              "native_user_config": {"epochs": 1, "per_device_train_batch_size": 1,
                                     "gradient_accumulation_steps": 1, "max_steps": None,
                                     "max_records": 256, "minimum_reward": 0.5}}
    write_json(round_dir / "ROUND_COMPLETE.json", {"parametric_sd_lora": {"status": "UPDATE",
                                                                             "update_receipt": update}})
    (round_dir / "evidence").mkdir(parents=True, exist_ok=True)
    (round_dir / "evidence/candidate-evidence-unconsumed.jsonl").write_text(
        json.dumps({"evidence_id": evidence_id, "task_identity": key, "source_round": 5,
                    "task_id": "source-task-r005", "schedule_position": 25,
                    "evidence_type": "clean_exact_trajectory", "source_final_score": 1.0,
                    "source_attempts_path": str(root / "round-005/attempts.jsonl")}) + "\n", encoding="utf-8")
    write_json(round_dir / "sd-lora/dataset/manifest.json", {
        "bound_evidence_ids": [evidence_id], "bound_task_identities": [key]})
    records_path = round_dir / "sd-lora/dataset/records.jsonl"
    records_path.write_text(json.dumps({"task_id": key, "reward": 1.0, "status": "completed", "traces": [{
        "metadata": {"source_evidence_id": evidence_id, "source_task_identity": key, "source_schedule_position": 25},
        "prompt_messages": [{"role": "user", "content": "task"}],
        "response_messages": [{"role": "assistant", "content": "action"}]}]}) + "\n")
    write_json(round_dir / "sd-lora/update.json", update)
    write_json(round_dir / "state-before.json", {"state_id": "before", "current_sd_adapter": adapter_path_before,
                                                    "current_sd_artifact_id": "update-21",
                                                    "sd_checkpoints": [{"path": adapter_path_before,
                                                                        "adapter_sha256": before_digest}]})
    write_json(round_dir / "state-after.json", {"state_id": "after" if changed_state else "before",
                                                   "current_sd_adapter": adapter_path_after,
                                                   "current_sd_artifact_id": "update-22",
                                                   "sd_checkpoints": [{"path": adapter_path_after,
                                                                       "adapter_sha256": after_digest}]})
    for name in ("state-before.json", "state-after.json"):
        path = round_dir / name
        state = json.loads(path.read_text())
        state.update(capacity_core_sha=core, capacity_policy_sha256=adapter.ORDINARY_CORE_POLICIES[core])
        write_json(path, state)
    refresh_training_bindings(round_dir)
    manifest = tmp_path / "ledger.json"
    inventory_sha256 = hashlib.sha256(json.dumps(ledger_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(manifest, {"schema": INPUT_SCHEMA, "lineage": "ordinary", "ledger_files": ledger_rows,
                          "inventory_sha256": inventory_sha256})
    return root, manifest, tmp_path / "q08-run", task["task_id"]


def refresh_training_bindings(round_dir):
    mp = round_dir / "sd-lora/dataset/manifest.json"
    rp = mp.with_name("records.jsonl")
    ep = round_dir / "evidence/candidate-evidence-unconsumed.jsonl"
    records = list(adapter.read_jsonl(rp))
    count = sum(len(r["traces"]) for r in records)
    manifest = json.loads(mp.read_text())
    manifest.update(records_sha256=adapter.sha256(rp), records_byte_size=rp.stat().st_size,
                    record_count=len(records), trace_count=count, max_training_examples=256)
    write_json(mp, manifest)
    up = round_dir / "sd-lora/update.json"
    update = json.loads(up.read_text())
    update.update(dataset_record_count=len(records), dataset_trace_count=count,
                  current_training_example_count=count, optimizer_example_count=count, optimizer_steps=count)
    write_json(up, update)
    cp = round_dir / "ROUND_COMPLETE.json"
    complete = json.loads(cp.read_text())
    complete["parametric_sd_lora"].update(update_receipt=update, vnext_selection={
        "schema": "openevo.ceiling1-stage2-vnext-parametric-selection.v1", "status": "DATASET_READY",
        "round_index": 22, "manifest": str(mp), "records": str(rp),
        "manifest_sha256": adapter.sha256(mp), "records_sha256": adapter.sha256(rp),
        "candidate_evidence_sha256": adapter.sha256(ep), "bound_evidence_ids": manifest["bound_evidence_ids"],
        "bound_task_identities": manifest["bound_task_identities"], "selected_task_count": len(records), "trace_count": count})
    write_json(cp, complete)


def test_adapter_requires_evidence_dataset_and_changed_state_chain(tmp_path):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path)
    payload = normalize("ordinary", root, manifest, evaluation_root,
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    events = [event for event in payload["events"] if event["task_id"] == task_id]
    assert [(e["round"], e["rollout_consumed"], e["training_included"], e["accepted_update"])
            for e in events] == [(5, True, False, False), (22, True, True, True)]
    assert events[1]["state_before"] == "before"
    assert len(events[1]["update_receipt_sha256"]) == 64
    assert any(receipt["role"] == "active_sd_adapter_payload"
               and receipt["sha256"] == events[1]["adapter_digest_after"]
               for receipt in payload["source_provenance"]["source_receipts"])


def test_adapter_keeps_noop_or_unchanged_state_out_of_effective_exposure(tmp_path):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path, changed_state=False)
    payload = normalize("ordinary", root, manifest, evaluation_root,
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    event = next(e for e in payload["events"] if e["task_id"] == task_id and e["round"] == 22)
    assert event["training_included"] is True
    assert event["accepted_update"] is False


def test_adapter_rejects_drift_from_frozen_ledger_hashes(tmp_path):
    root, manifest, evaluation_root, _ = fixture_inputs(tmp_path)
    with (root / "round-005/attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        normalize("ordinary", root, manifest, evaluation_root,
                  hashlib.sha256(manifest.read_bytes()).hexdigest())


def test_original_ordinary_inventory_uses_exact_manifest_pin(tmp_path, monkeypatch):
    from scripts.analysis import advisor_q08_retention_input_adapter as adapter
    root, manifest, evaluation_root, _ = fixture_inputs(tmp_path)
    rows = json.loads(manifest.read_text())["ledger_files"]
    write_json(manifest, {"ordinary_ledger_files": rows})
    monkeypatch.setattr(adapter, "SEEN_CANDIDATE_PATH", manifest)
    monkeypatch.setattr(adapter, "SEEN_CANDIDATE_SHA256", adapter.sha256(manifest))
    payload = normalize("ordinary", root, manifest, evaluation_root)
    assert payload["events"]
    with (root / "round-005/attempts.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        normalize("ordinary", root, manifest, evaluation_root)


def test_original_ordinary_inventory_rejects_changed_manifest(tmp_path, monkeypatch):
    from scripts.analysis import advisor_q08_retention_input_adapter as adapter
    root, manifest, evaluation_root, _ = fixture_inputs(tmp_path)
    rows = json.loads(manifest.read_text())["ledger_files"]
    write_json(manifest, {"ordinary_ledger_files": rows})
    monkeypatch.setattr(adapter, "SEEN_CANDIDATE_PATH", manifest)
    monkeypatch.setattr(adapter, "SEEN_CANDIDATE_SHA256", adapter.sha256(manifest))
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="unpinned owner inventory"):
        normalize("ordinary", root, manifest, evaluation_root)


def test_dataset_sources_outside_panel_are_validated_without_adding_panel_tasks(tmp_path):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path)
    attempts = root / "round-005/attempts.jsonl"
    row = json.loads(attempts.read_text())
    row.update(worker_seed=99, source_session_index=999999, instruction_sha256="f" * 64,
               task_id="non-panel-task", schedule_position=26)
    with attempts.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    ledger = json.loads(manifest.read_text())
    ledger["ledger_files"][5]["sha256"] = hashlib.sha256(attempts.read_bytes()).hexdigest()
    ledger["inventory_sha256"] = hashlib.sha256(json.dumps(
        ledger["ledger_files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(manifest, ledger)
    key = "99:999999:" + "f" * 64
    evidence = root / "round-022/evidence/candidate-evidence-unconsumed.jsonl"
    with evidence.open("a") as stream:
        stream.write(json.dumps({"evidence_id": "b" * 64, "task_identity": key,
                                 "source_round": 5, "task_id": "non-panel-task", "schedule_position": 26,
                                 "evidence_type": "clean_exact_trajectory", "source_final_score": 1.0,
                                 "source_attempts_path": str(attempts)}) + "\n")
    dataset_path = root / "round-022/sd-lora/dataset/manifest.json"
    dataset = json.loads(dataset_path.read_text())
    dataset["bound_evidence_ids"].append("b" * 64)
    dataset["bound_task_identities"].append(key)
    write_json(dataset_path, dataset)
    records_path = dataset_path.with_name("records.jsonl")
    record = next(adapter.read_jsonl(records_path))
    record["task_id"] = key
    record["traces"][0]["metadata"].update(source_task_identity=key, source_evidence_id="b" * 64,
                                           source_schedule_position=26)
    with records_path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    refresh_training_bindings(root / "round-022")
    payload = normalize("ordinary", root, manifest, evaluation_root,
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert {event["task_id"] for event in payload["events"]} == {task_id}
    assert any(event["accepted_update"] for event in payload["events"])


def test_evaluation_reads_frozen_probe_score_field(tmp_path):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path)
    path = evaluation_root / "cells/ordinary-r080/seen-eval/attempts.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"task_identity": task_id, "engineering_invalid": False,
                                "episode": {"score": 0.6, "won": False},
                                "paired_generation_seed": 202610010523001}) + "\n")
    payload = normalize("ordinary", root, manifest, evaluation_root,
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert payload["scores"] == [{"task_id": task_id, "completed_rounds": 80,
                                  "score": 0.6, "engineering_valid": True,
                                  "evaluation_seed": 202610010523001}]


def test_missing_dataset_binding_is_typed_partial_not_no_exposure(tmp_path):
    root, manifest, evaluation_root, _ = fixture_inputs(tmp_path)
    path = root / "round-022/sd-lora/dataset/manifest.json"
    path.write_text(json.dumps({"schema": "openevo.dataset.v1"}) + "\n", encoding="utf-8")
    payload = normalize("ordinary", root, manifest, evaluation_root,
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert payload["audit_status"] == "PARTIAL"
    assert any(gap["reason"] == "UPDATE_DATASET_HAS_NO_BINDING_FIELDS" for gap in payload["audit_gaps"])


def test_changed_state_id_without_changed_adapter_digest_is_not_update(tmp_path):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path)
    after_path = root / "round-022/state-after.json"
    after = json.loads(after_path.read_text())
    after["state_id"] = "another-state-id"
    after["sd_checkpoints"][0]["adapter_sha256"] = "d" * 64
    after_path.write_text(json.dumps(after) + "\n", encoding="utf-8")
    payload = normalize("ordinary", root, manifest, evaluation_root,
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    event = next(e for e in payload["events"] if e["task_id"] == task_id and e["round"] == 22)
    assert event["accepted_update"] is False
    assert payload["audit_status"] == "PARTIAL"


def test_missing_active_adapter_path_is_not_treated_as_known_empty_baseline(tmp_path):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path)
    before_path = root / "round-022/state-before.json"
    before_path.write_text(json.dumps({"state_id": "before", "current_sd_adapter": None,
                                       "current_sd_artifact_id": "claimed-active", "sd_component_count": 1,
                                       "sd_checkpoints": []}) + "\n", encoding="utf-8")
    payload = normalize("ordinary", root, manifest, evaluation_root,
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    event = next(e for e in payload["events"] if e["task_id"] == task_id and e["round"] == 22)
    assert event["accepted_update"] is False
    assert payload["audit_status"] == "PARTIAL"


def test_falsey_nonnull_adapter_path_is_not_treated_as_empty_baseline(tmp_path):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path)
    before_path = root / "round-022/state-before.json"
    before_path.write_text(json.dumps({"state_id": "before", "current_sd_adapter": "",
                                       "current_sd_artifact_id": None, "sd_component_count": 0,
                                       "sd_checkpoints": []}) + "\n", encoding="utf-8")
    payload = normalize("ordinary", root, manifest, evaluation_root,
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    event = next(e for e in payload["events"] if e["task_id"] == task_id and e["round"] == 22)
    assert event["accepted_update"] is False
    assert payload["audit_status"] == "PARTIAL"


def test_unbound_historical_core_cannot_claim_training_inclusion(tmp_path, monkeypatch):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path)
    monkeypatch.setattr(adapter, "_ordinary_source_contract", lambda *args: None)
    payload = normalize("ordinary", root, manifest, evaluation_root, adapter.sha256(manifest))
    event = next(e for e in payload["events"] if e["task_id"] == task_id and e["round"] == 22)
    assert event["training_included"] is False and event["accepted_update"] is False
    assert any(g["reason"] == "ORDINARY_TRAINING_INCLUSION_CHAIN_UNPROVEN" for g in payload["audit_gaps"])


def test_unknown_terminal_status_is_audit_gap(tmp_path):
    root, manifest, evaluation_root, _ = fixture_inputs(tmp_path)
    write_json(root / "round-023/ROUND_COMPLETE.json", {"parametric_sd_lora": {"status": "UNKNOWN"}})
    payload = normalize("ordinary", root, manifest, evaluation_root, adapter.sha256(manifest))
    assert {"round": 23, "reason": "ORDINARY_TERMINAL_STATUS_UNPROVEN"} in payload["audit_gaps"]


def test_records_hash_drift_cannot_claim_training_inclusion(tmp_path):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path)
    path = root / "round-022/sd-lora/dataset/records.jsonl"
    path.write_text(path.read_text() + "\n")
    payload = normalize("ordinary", root, manifest, evaluation_root, adapter.sha256(manifest))
    event = next(e for e in payload["events"] if e["task_id"] == task_id and e["round"] == 22)
    assert event["training_included"] is False and event["accepted_update"] is False


def test_optimizer_count_mismatch_cannot_claim_training_inclusion(tmp_path):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path)
    d = root / "round-022"
    update = json.loads((d / "sd-lora/update.json").read_text())
    update["optimizer_steps"] = 0
    write_json(d / "sd-lora/update.json", update)
    complete = json.loads((d / "ROUND_COMPLETE.json").read_text())
    complete["parametric_sd_lora"]["update_receipt"] = update
    write_json(d / "ROUND_COMPLETE.json", complete)
    payload = normalize("ordinary", root, manifest, evaluation_root, adapter.sha256(manifest))
    event = next(e for e in payload["events"] if e["task_id"] == task_id and e["round"] == 22)
    assert event["training_included"] is False and event["accepted_update"] is False


def test_measured_retirement_supports_missing_historical_tensor(tmp_path):
    root, manifest, evaluation_root, task_id = fixture_inputs(tmp_path)
    d = root / "round-022"
    before, after = (json.loads((d / name).read_text()) for name in ["state-before.json", "state-after.json"])
    names = ["adapter_config.json", "adapter_model.safetensors", "openevo_sd_lora_replay.jsonl",
             "openevo_sd_lora_state.json", "openevo_sd_lora_state.safetensors"]
    removed = [{"name": name, "bytes": 11, "sha256": "a" * 64} for name in names]
    next(f for f in removed if f["name"] == "adapter_model.safetensors")["sha256"] = before["sd_checkpoints"][0]["adapter_sha256"]
    replacement = dict.fromkeys(names, "b" * 64)
    replacement["adapter_model.safetensors"] = after["sd_checkpoints"][0]["adapter_sha256"]
    retirement = {"schema": "openevo.ceiling1-stage2-prior-input-retirement.v1", "status": "PASS",
                  "prior_input_path": str(d / "sd-lora/artifacts/prior-input"),
                  "replacement_adapter_path": after["current_sd_adapter"],
                  "files_removed": removed, "replacement_identity": replacement}
    write_json(d / "sd-lora/artifacts/prior-input-retired.json", retirement)
    update = json.loads((d / "sd-lora/update.json").read_text())
    update["prior_input_retirement"] = retirement
    write_json(d / "sd-lora/update.json", update)
    complete = json.loads((d / "ROUND_COMPLETE.json").read_text())
    complete["parametric_sd_lora"]["update_receipt"] = update
    write_json(d / "ROUND_COMPLETE.json", complete)
    for state in [before, after]:
        path = adapter.Path(state["current_sd_adapter"]) / "adapter_model.safetensors"
        path.rename(path.with_name("preserved-model.bin"))
    payload = normalize("ordinary", root, manifest, evaluation_root, adapter.sha256(manifest))
    event = next(e for e in payload["events"] if e["task_id"] == task_id and e["round"] == 22)
    assert event["accepted_update"] is True
    assert any(r["role"] == "measured_prior_input_retirement" for r in payload["source_provenance"]["source_receipts"])
