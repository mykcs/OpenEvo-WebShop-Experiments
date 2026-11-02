from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.openevo_webshop import experiment_revive_plan as rp
from scripts.openevo_webshop import experiment_verify as ev

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boundary_fixture(tmp_path: Path, round_index: int = 3, state_id: str = "s3") -> tuple[Path, dict]:
    run = tmp_path / "run"
    round_dir = run / f"round-{round_index:03d}"
    complete = {
        "status": "PASS", "round_index": round_index,
        "state_id_before": "before", "state_id_after": state_id,
    }
    complete_sha = _write(round_dir / "ROUND_COMPLETE.json", complete)
    state = {
        "status": "RUNNING", "state_id": state_id, "next_round": round_index + 1,
        "rollouts_consumed": (round_index + 1) * 128,
    }
    _write(round_dir / "state-after.json", state)
    receipt = {
        "status": "PASS", "round_index": round_index,
        "round_complete_sha256": complete_sha,
        "local_round_path": str(round_dir),
        "archive_policy": "local_durable_deferred_remote",
        "remote_target": f"arms/direct_apply/round-{round_index:03d}.tar.gz",
        "remote_archive_deferred": True,
    }
    _write(run / "archive-receipts" / f"anything-{round_index}.json", receipt)
    return run, state


def test_last_safe_boundary_comes_from_verified_receipt_not_largest_directory(tmp_path: Path) -> None:
    run, state = _boundary_fixture(tmp_path, 3, "state-3")
    (run / "round-999").mkdir()
    result = rp.find_last_safe_boundary(run, state)
    assert result["status"] == "PASS"
    assert result["boundary"]["round_index"] == 3
    assert result["reconciliation"] == "SYNCED"
    assert "receipt contents" in result["reason"]


def test_live_newer_than_archive_is_explicit(tmp_path: Path) -> None:
    run, _ = _boundary_fixture(tmp_path, 3, "state-3")
    live = {"next_round": 5, "state_id": "state-4", "rollouts_consumed": 640}
    result = rp.find_last_safe_boundary(run, live)
    assert result["reconciliation"] == "LIVE_STATE_NEWER_THAN_ARCHIVE"

def test_missing_carrier_is_reported_without_retraining(tmp_path: Path) -> None:
    run, state = _boundary_fixture(tmp_path, 1, "state-1")
    boundary = rp.find_last_safe_boundary(run, state)["boundary"]
    carriers = rp.collect_carrier_requirements(boundary)
    assert {item["role"] for item in carriers} == {"sd_lora", "text_memory", "skill_bundle", "agent_system"}
    assert all(item["status"] == "MISSING" for item in carriers)


def test_active_running_plan_never_emits_launch_command() -> None:
    verification = {"identity": "PASS", "overall": "PASS"}
    verdict = rp._verdict_registered(
        verification,
        {"status": "RUNNING"},
        {"status": "PASS", "reconciliation": "SYNCED"},
        [],
    )
    assert verdict == "MANUAL_DECISION_REQUIRED"
    assert rp._recommended_next("202609120000-demo", verdict, "RUNNING") is None


def test_sealed_plan_is_complete_no_resume() -> None:
    verification = {"identity": "PASS", "overall": "PASS"}
    verdict = rp._verdict_registered(
        verification,
        {"status": "STAGE2_SEALED"},
        {"status": "PASS", "reconciliation": "SYNCED"},
        [],
    )
    assert verdict == "COMPLETE_NO_RESUME_NEEDED"
    assert rp._recommended_next("202609120000-demo", verdict, "STAGE2_SEALED") is None

def test_live_newer_verdict_outranks_terminal_assumption() -> None:
    verdict = rp._verdict_registered(
        {"identity": "PASS", "overall": "PASS"},
        {"status": "STAGE2_SEALED"},
        {"status": "PASS", "reconciliation": "LIVE_STATE_NEWER_THAN_ARCHIVE"},
        [],
    )
    assert verdict == "LIVE_STATE_NEWER_THAN_ARCHIVE"


def test_missing_asset_verdict_is_explicit() -> None:
    verdict = rp._verdict_registered(
        {"identity": "PASS", "overall": "PASS"},
        {"status": "BLOCKED"},
        {"status": "PASS", "reconciliation": "SYNCED"},
        [{"role": "sd_lora", "status": "MISSING"}],
    )
    assert verdict == "MISSING_ASSET"


def test_legacy_no_resume_and_complete_are_terminal() -> None:
    legacy = rp.build_revive_plan(ROOT, "qwen25-3b-legacy-stage2")
    complete = rp.build_revive_plan(ROOT, "qwen3-1p7b-gdr")
    assert legacy["verdict"] == "NO_RESUME_HISTORICAL"
    assert complete["verdict"] == "COMPLETE_NO_RESUME_NEEDED"
    assert legacy["recommended_next_command"] is None
    assert legacy["artifacts"]["state_model_policy"].endswith("RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md")
    assert complete["recommended_next_command"] is None

def test_offserver_q17_uses_bound_durable_recovery_observation() -> None:
    plan = rp.build_revive_plan(ROOT, "qwen3-1p7b-directapply-no-gdr")
    assert plan["verdict"] == "MISSING_ASSET"
    assert plan["current_or_last_observed_state"]["status"] == "STAGE2_SEALED"
    assert plan["current_or_last_observed_state"]["observation_kind"] == "DURABLE_RECOVERY_OBSERVATION"
    assert plan["last_safe_boundary"]["status"] == "PASS"
    assert plan["last_safe_boundary"]["boundary"]["round_index"] == 159
    assert plan["last_safe_boundary"]["boundary"]["verified_boundary_count"] == 160
    assert plan["artifacts"]["remote_artifacts"][0]["revision"] == "210690c11bf95256532fdaa681c69cd338793fc0"
    assert plan["artifacts"]["state_model_policy"].endswith("RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md")
    assert {item["role"] for item in plan["missing_items"]} == {"round_159_recovery_archive"}
    assert plan["execution_action"] == "NONE"
    assert plan["mutation"] is False


def test_durable_recovery_observation_identity_mismatch_fails_closed() -> None:
    entry = {"experiment_id": "202609120000-demo", "passport_sha256": "a" * 64}
    observation = {
        "schema": "openevo.experiment-recovery-observation.test",
        "experiment_id": "202609120000-other",
        "passport_sha256": "b" * 64,
        "last_observed_state": {"status": "STAGE2_SEALED", "state_id": "s", "next_round": 2},
        "last_safe_boundary": {"round_index": 1, "next_round": 2, "state_id": "s", "verified_boundary_count": 2},
        "remote_artifact_pointers": [],
    }
    errors = rp.validate_durable_recovery_observation(observation, entry)
    assert "experiment_id mismatch" in errors
    assert "passport_sha256 mismatch" in errors


def test_durable_remote_pointer_rejects_mutable_main_revision() -> None:
    entry = {"experiment_id": "202609120000-demo", "passport_sha256": "a" * 64}
    observation = {
        "schema": "openevo.experiment-recovery-observation.test",
        "experiment_id": entry["experiment_id"],
        "passport_sha256": entry["passport_sha256"],
        "last_observed_state": {"status": "STAGE2_SEALED", "state_id": "s", "next_round": 2},
        "last_safe_boundary": {"round_index": 1, "next_round": 2, "state_id": "s", "verified_boundary_count": 2},
        "remote_artifact_pointers": [{
            "provider": "huggingface", "repository": "owner/repo", "revision": "main",
            "path_in_repo": "archive/x", "status": "REMOTE_VERIFIED_EXACT",
        }],
    }
    errors = rp.validate_durable_recovery_observation(observation, entry)
    assert any("immutable, not main" in error for error in errors)


def test_durable_remote_pointer_validates_two_axis_states() -> None:
    entry = {"experiment_id": "202609120000-demo", "passport_sha256": "a" * 64}
    observation = {
        "schema": "openevo.experiment-recovery-observation.test",
        "experiment_id": entry["experiment_id"],
        "passport_sha256": entry["passport_sha256"],
        "last_observed_state": {"status": "STAGE2_SEALED", "state_id": "s", "next_round": 2},
        "last_safe_boundary": {"round_index": 1, "next_round": 2, "state_id": "s", "verified_boundary_count": 2},
        "remote_artifact_pointers": [{
            "provider": "huggingface", "repository": "owner/repo", "revision": "1" * 40,
            "path_in_repo": "archive/x", "status": "REMOTE_VERIFIED_EXACT",
            "recovery_state": "REMOTE_BACKED_UP",
            "disclosure_state": "PRIVATE",
            "provider_visibility": "private",
        }],
    }
    errors = rp.validate_durable_recovery_observation(observation, entry)
    assert any("must be RECOVERY_VERIFIED" in error for error in errors)

    observation["remote_artifact_pointers"][0]["recovery_state"] = "RECOVERY_VERIFIED"
    observation["remote_artifact_pointers"][0]["disclosure_state"] = "NOT_A_STATE"
    errors = rp.validate_durable_recovery_observation(observation, entry)
    assert any("disclosure_state is unsupported" in error for error in errors)


def test_cli_json_and_human_are_read_only() -> None:
    machine = subprocess.run(
        [sys.executable, str(ROOT / "scripts/oe"), "experiments", "revive-plan", "qwen25-3b-legacy-stage2", "--json"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert machine.returncode == 0, machine.stderr
    payload = json.loads(machine.stdout)
    assert payload["verdict"] == "NO_RESUME_HISTORICAL"
    assert payload["execution_action"] == "NONE"
    human = subprocess.run(
        [sys.executable, str(ROOT / "scripts/oe"), "experiments", "revive-plan", "qwen25-3b-legacy-stage2"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert human.returncode == 0, human.stderr
    assert "Execution action: NONE" in human.stdout
    assert "NO_RESUME_HISTORICAL" in human.stdout


def test_revive_plan_does_not_spawn_gpu_or_network_tools(monkeypatch) -> None:
    calls: list[list[str]] = []
    original = ev.subprocess.run
    def wrapped(cmd, *args, **kwargs):
        calls.append([str(x) for x in cmd])
        return original(cmd, *args, **kwargs)
    monkeypatch.setattr(ev.subprocess, "run", wrapped)
    plan = rp.build_revive_plan(ROOT, "qwen3-1p7b-directapply-no-gdr")
    assert plan["mutation"] is False
    assert calls and {call[0] for call in calls} == {"git"}

def test_duplicate_archive_receipt_same_identity_is_deduplicated(tmp_path: Path) -> None:
    run, state = _boundary_fixture(tmp_path, 2, "same-state")
    original = next((run / "archive-receipts").glob("*.json"))
    duplicate = run / "archive-receipts" / "remote-sync.json"
    duplicate.write_bytes(original.read_bytes())
    result = rp.find_last_safe_boundary(run, state)
    assert result["verified_boundary_count"] == 1
    assert result["boundary"]["round_index"] == 2


def test_conflicting_duplicate_archive_receipt_fails_closed(tmp_path: Path) -> None:
    run, _ = _boundary_fixture(tmp_path, 2, "state-a")
    other_round = run / "other-round"
    complete_sha = _write(other_round / "ROUND_COMPLETE.json", {
        "status": "PASS", "round_index": 2, "state_id_before": "x", "state_id_after": "state-b",
    })
    _write(other_round / "state-after.json", {
        "status": "RUNNING", "state_id": "state-b", "next_round": 3, "rollouts_consumed": 384,
    })
    _write(run / "archive-receipts" / "conflict.json", {
        "status": "PASS", "round_index": 2, "round_complete_sha256": complete_sha,
        "local_round_path": str(other_round),
    })
    import pytest
    with pytest.raises(rp.RevivePlanError, match="conflicting verified archive receipts"):
        rp.collect_receipt_chain(run)
