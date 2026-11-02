from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.openevo_runtime import unattended_supervisor as us
from scripts.oe_impl import main as oe_main


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _base_config(tmp_path: Path, *, arm_count: int = 3) -> tuple[Path, dict]:
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True)
    (checkout / "scripts").mkdir()
    (checkout / "scripts" / "oe").write_text("#!/bin/sh\n", encoding="utf-8")
    lanes = [
        {
            "id": "lane-a",
            "gpu_uuids": [
                "GPU-00000000-0000-0000-0000-000000000000",
                "GPU-00000000-0000-0000-0000-000000000000",
            ],
        },
        {
            "id": "lane-b",
            "gpu_uuids": [
                "GPU-00000000-0000-0000-0000-000000000000",
                "GPU-00000000-0000-0000-0000-000000000000",
            ],
        },
    ]
    arms = []
    for idx in range(arm_count):
        root = tmp_path / f"run-{idx}"
        root.mkdir()
        arms.append(
            {
                "id": f"rank-{idx}",
                "priority": idx,
                "run_output_root": str(root),
                "authority_path": str(checkout / "authority.json"),
                "rounds": [152, 153],
                "eligible_lanes": ["lane-a", "lane-b"],
                "launch": {
                    "cwd": str(checkout),
                    "expected_git_head": "a" * 40,
                    "argv": [
                        "./scripts/oe",
                        "run-campaign",
                        "--release",
                        "release.json",
                        "--output-root",
                        "${CONTROL_ROOT}",
                        "--run-output-root",
                        "${RUN_ROOT}",
                        "--gpus",
                        "${GPU_UUIDS}",
                        "--wait",
                        "--resume",
                    ],
                    "env": {"OPENEVO_AUTHORIZED_GPU_UUIDS": "${GPU_UUIDS}"},
                },
            }
        )
    doc = {
        "schema": us.SCHEMA,
        "scientific_authority": False,
        "state_root": str(tmp_path / "state"),
        "poll_seconds": 0.01,
        "max_parallel_arms": 2,
        "lanes": lanes,
        "arms": arms,
        "service": {
            "name": "openevo-test-supervisor",
            "description": "test supervisor",
            "user": "root",
            "group": "root",
        },
    }
    path = tmp_path / "supervisor.json"
    _json(path, doc)
    return path, doc


def test_config_rejects_overlapping_gpu_lanes(tmp_path: Path) -> None:
    path, doc = _base_config(tmp_path)
    doc["lanes"][1]["gpu_uuids"][0] = doc["lanes"][0]["gpu_uuids"][0].upper()
    _json(path, doc)
    with pytest.raises(us.SupervisorError, match="disjoint"):
        us.config(path)


def test_config_rejects_malformed_gpu_uuid(tmp_path: Path) -> None:
    path, doc = _base_config(tmp_path)
    doc["lanes"][0]["gpu_uuids"][0] = "GPU-not-a-uuid"
    _json(path, doc)
    with pytest.raises(us.SupervisorError, match="lane identity"):
        us.config(path)


def test_config_rejects_duplicate_lane_uuids_and_excess_parallelism(tmp_path: Path) -> None:
    path, doc = _base_config(tmp_path)
    doc["lanes"][0]["gpu_uuids"][1] = doc["lanes"][0]["gpu_uuids"][0]
    _json(path, doc)
    with pytest.raises(us.SupervisorError, match="lane identity"):
        us.config(path)

    path, doc = _base_config(tmp_path / "parallel")
    doc["max_parallel_arms"] = 3
    _json(path, doc)
    with pytest.raises(us.SupervisorError, match="lanes/max_parallel_arms"):
        us.config(path)


def test_config_rejects_out_of_order_frozen_rounds(tmp_path: Path) -> None:
    path, doc = _base_config(tmp_path, arm_count=1)
    doc["arms"][0]["rounds"] = [153, 152]
    _json(path, doc)
    with pytest.raises(us.SupervisorError, match="rounds invalid"):
        us.config(path)


def test_plan_runs_two_independent_arms_and_queues_third(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path)
    cfg = us.config(path)
    monkeypatch.setattr(
        us,
        "discover",
        lambda arm: {
            "action": "READY_TO_DISPATCH",
            "next_round": 152,
            "reason": "ready",
        },
    )
    result = us.plan(cfg)
    assert [(x["arm_id"], x["lane_id"]) for x in result["assignments"]] == [
        ("rank-0", "lane-a"),
        ("rank-1", "lane-b"),
    ]
    assert result["arms"]["rank-2"]["action"] == "WAIT_RESOURCE"
    assert result["scientific_authority_granted"] is False


def test_completed_arm_releases_lane_for_queued_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path)
    cfg = us.config(path)

    def fake(arm: dict) -> dict:
        if arm["id"] == "rank-0":
            return {"action": "COMPLETE", "reason": "sealed"}
        return {"action": "READY_TO_DISPATCH", "next_round": 152, "reason": "ready"}

    monkeypatch.setattr(us, "discover", fake)
    result = us.plan(cfg)
    assert [(x["arm_id"], x["lane_id"]) for x in result["assignments"]] == [
        ("rank-1", "lane-a"),
        ("rank-2", "lane-b"),
    ]


def test_active_child_has_running_state_and_holds_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=2)
    cfg = us.config(path)
    monkeypatch.setattr(
        us,
        "discover",
        lambda arm: {"action": "READY_TO_DISPATCH", "next_round": 152, "reason": "ready"},
    )
    result = us.plan(
        cfg,
        {
            "rank-0": {"lane_id": "lane-a", "dispatch_id": "running-1"},
        },
    )
    assert result["arms"]["rank-0"]["action"] == "RUNNING"
    assert result["lanes"]["lane-a"]["assigned_arm"] == "rank-0"
    assert result["assignments"] == [{"arm_id": "rank-1", "lane_id": "lane-b", "action": "READY_TO_DISPATCH"}]


def test_controller_lock_blocks_second_supervisor(tmp_path: Path) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    with us.controller_lock(cfg):
        with pytest.raises(us.SupervisorError, match="another supervisor"):
            with us.controller_lock(cfg):
                pytest.fail("second controller unexpectedly acquired the lock")


def test_planning_fault_does_not_terminate_running_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=2)
    cfg = us.config(path)

    class RunningProc:
        pid = 12345
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    proc = RunningProc()
    calls = 0

    def plan_then_fail(d, running=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "schema": us.STATUS_SCHEMA,
                "arms": {"rank-0": {"action": "READY_TO_DISPATCH"}},
                "assignments": [{"arm_id": "rank-0", "lane_id": "lane-a"}],
            }
        raise RuntimeError("planning failed for another arm")

    monkeypatch.setattr(us, "plan", plan_then_fail)
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "git_head", lambda cwd: "a" * 40)
    monkeypatch.setattr(us.subprocess, "Popen", lambda *args, **kwargs: proc)
    with pytest.raises(RuntimeError, match="planning failed"):
        us.run(cfg, execute=True, sleep_fn=lambda _: None)

    assert proc.terminated is False
    dispatch_root = Path(cfg["state_root"]) / "dispatches" / "rank-0"
    intents = list(dispatch_root.glob("*/intent.json"))
    assert len(intents) == 1
    assert not list(dispatch_root.glob("*/terminal.json"))
    events = [json.loads(line) for line in (Path(cfg["state_root"]) / "events.jsonl").read_text().splitlines()]
    stopped = next(row for row in events if row["event"] == "SUPERVISOR_STOPPED")
    assert stopped["remaining_children"] == ["rank-0"]


def test_external_controller_blocks_new_lane_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path)
    cfg = us.config(path)

    def fake(arm: dict) -> dict:
        if arm["id"] == "rank-0":
            return {
                "action": "RUNNING_EXTERNAL",
                "next_round": 152,
                "reason": "lock held",
            }
        return {
            "action": "READY_TO_DISPATCH",
            "next_round": 152,
            "reason": "ready",
        }

    monkeypatch.setattr(us, "discover", fake)
    result = us.plan(cfg)
    assert result["assignments"] == []
    assert result["blocked_by_external_controller"] is True
    assert result["arms"]["rank-1"]["action"] == "WAIT_RESOURCE"


def test_final_panel_access_blocks_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    _json(
        Path(arm["run_output_root"]) / "execution-state.json",
        {
            "status": "RUNNING",
            "next_round": 152,
            "final_panel_access_count": 1,
        },
    )
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "lock_busy", lambda root: False)
    result = us.discover(arm)
    assert result["action"] == "BLOCKED_NEEDS_HUMAN"
    assert "final-panel" in result["reason"]


@pytest.mark.parametrize(
    ("identity_result", "expected_reason"),
    [
        ((False, "launch checkout is dirty"), "dirty"),
        ((False, "launch checkout HEAD drift: expected A got B"), "HEAD drift"),
    ],
)
def test_launch_checkout_identity_blocks_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_result: tuple[bool, str],
    expected_reason: str,
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: identity_result)
    result = us.discover(cfg["arms"][0])
    assert result["action"] == "BLOCKED_NEEDS_HUMAN"
    assert expected_reason in result["reason"]


def test_rollout_receipt_is_never_replayed_by_recovery_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    root = Path(arm["run_output_root"])
    _json(
        root / "execution-state.json",
        {
            "status": "RUNNING",
            "next_round": 152,
            "final_panel_access_count": 0,
        },
    )
    pre_recovery = us.recovery_identity(arm)
    pre_recovery_sha = pre_recovery["sha256"]
    rr = root / "round-152"
    _json(rr / "state-before.json", {"round_index": 152, "state_id": "before"})
    _json(
        rr / "ROLLOUT_PHASE_COMPLETE.json",
        {
            "status": "PASS",
            "round_index": 152,
            "state_id_before": "before",
        },
    )
    us.atomic(
        us.terminal_marker_path(arm),
        {
            "schema": "openevo.unattended-dispatch-terminal.20261004",
            "dispatch_id": "materialized-rollout",
            "arm_id": arm["id"],
            "lane_id": "lane-a",
            "kind": "launch",
            "returncode": 1,
            "pre_state_sha256": us.sha(root / "execution-state.json"),
            "pre_recovery_sha256": pre_recovery_sha,
            "pre_recovery_projection": pre_recovery["projection"],
            "pre_git_head": "same-head",
            "progress_cwd": arm["launch"]["cwd"],
        },
    )
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "git_head", lambda cwd: "same-head")
    monkeypatch.setattr(us, "lock_busy", lambda root: False)
    result = us.discover(arm)
    assert result["action"] == "READY_TO_DISPATCH"
    assert result["recovery_plan"]["verdict"] == "SAFE_TO_PREPARE_NEXT_PHASE"
    assert "rollout" in result["recovery_plan"]["must_not_replay"]


def test_materialized_round_side_effect_reconciles_after_child_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    root = Path(arm["run_output_root"])
    state_path = root / "execution-state.json"
    _json(
        state_path,
        {"status": "RUNNING", "next_round": 152, "final_panel_access_count": 0},
    )
    pre_recovery = us.recovery_identity(arm)
    pre_recovery_sha = pre_recovery["sha256"]
    round_root = root / "round-152"
    from tests.test_execution_runtime_round_transaction_20260916 import write_complete_prefix
    write_complete_prefix(round_root, "rollout", round_index=152)
    us.atomic(
        us.terminal_marker_path(arm),
        {
            "dispatch_id": "dispatch-1",
            "arm_id": arm["id"],
            "kind": "launch",
            "returncode": 1,
            "pre_state_sha256": us.sha(state_path),
            "pre_recovery_sha256": pre_recovery_sha,
            "pre_recovery_projection": pre_recovery["projection"],
            "pre_git_head": "same-head",
            "progress_cwd": arm["launch"]["cwd"],
        },
    )
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "git_head", lambda cwd: "same-head")
    monkeypatch.setattr(us, "lock_busy", lambda root: False)
    result = us.discover(arm)
    assert result["action"] == "READY_TO_DISPATCH"
    assert "exact resume" in result["reason"]
    assert result["recovery_plan"]["must_not_replay"] == ["state_before", "rollout"]


def _write_known_health_pause(root: Path, *, unknown: bool = False) -> str:
    rr = root / "round-152"
    ledger_path = rr / "carrier-health-ledger.json"
    boundary_path = rr / "carrier-health-boundary.json"
    complete_path = rr / "ROUND_COMPLETE.json"
    after_path = rr / "state-after.json"
    reason = "SOME_NEW_FAILURE" if unknown else us.KNOWN_REASON
    _json(
        ledger_path,
        {
            "status": "FAIL",
            "round_index": 152,
            "rows": [
                {
                    "carrier": "skill_bundle",
                    "health_status": "FAIL",
                    "health_reasons": [reason],
                }
            ],
        },
    )
    _json(
        boundary_path,
        {
            "action": "PAUSE_FOR_DIAGNOSTIC",
            "round_index": 152,
            "preserve_completed_transaction": True,
            "block_next_round": True,
            "automatic_mutations": {key: False for key in us.MUTATIONS},
        },
    )
    _json(complete_path, {"status": "PASS", "round_index": 152})
    _json(after_path, {"next_round": 153, "state_id": "after"})
    _json(
        rr / "skill-bundle" / "terminal-receipt.json",
        {
            "status": "NOOP",
            "reason": us.KNOWN_TERMINAL,
            "primary_calls": 1,
            "repair_calls": 1,
            "repair_requested": True,
            "primary_contract_valid": False,
            "repair_contract_valid": False,
            "repair_changed_bytes": False,
            "primary_payload_sha256": "same",
            "repair_payload_sha256": "same",
            "accepted_payload_sha256": "prior",
            "prior_sha256": "prior",
            "external_teacher_calls": 0,
        },
    )
    pause_path = root / "carrier-health-pauses" / "round-152.json"
    _json(
        pause_path,
        {
            "status": "PAUSE_FOR_DIAGNOSTIC",
            "through_round": 152,
            "next_round": 153,
            "formal_transaction_preserved": True,
            "automatic_scientific_change": False,
            "health_ledger_sha256": us.sha(ledger_path),
            "boundary_decision_sha256": us.sha(boundary_path),
            "round_complete_sha256": us.sha(complete_path),
            "state_after_sha256": us.sha(after_path),
        },
    )
    return us.sha(pause_path)


def test_known_identical_invalid_keep_prior_without_bound_resolution_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    root = Path(arm["run_output_root"])
    pause_sha = _write_known_health_pause(root)
    _json(
        root / "execution-state.json",
        {
            "status": "PAUSE_FOR_DIAGNOSTIC",
            "next_round": 153,
            "final_panel_access_count": 0,
            "carrier_health_pause_sha256": pause_sha,
        },
    )
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "lock_busy", lambda root: False)
    monkeypatch.setattr(us, "committed_json", lambda repo, candidate: {})
    result = us.discover(arm)
    assert result["action"] == "BLOCKED_NEEDS_HUMAN"
    assert "carrier_health_resolution" in result["reason"]


def test_known_pause_with_exact_bound_resolution_routes_to_canonical_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, doc = _base_config(tmp_path, arm_count=1)
    doc["arms"][0]["authority_path"] = str(tmp_path / "checkout" / "authority.json")
    _json(path, doc)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    root = Path(arm["run_output_root"])
    pause_sha = _write_known_health_pause(root)
    _json(
        root / "execution-state.json",
        {
            "status": "PAUSE_FOR_DIAGNOSTIC",
            "next_round": 153,
            "final_panel_access_count": 0,
            "carrier_health_pause_sha256": pause_sha,
        },
    )
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "lock_busy", lambda root: False)
    monkeypatch.setattr(
        us,
        "bound_health_resolution",
        lambda arm, detail: (
            True,
            {
                "resolution_path": "/committed/resolution.json",
                "resolution_sha256": "b" * 64,
                "through_round": detail["through_round"],
            },
        ),
    )
    result = us.discover(arm)
    assert result["action"] == "READY_TO_DISPATCH"
    assert result["bound_resolution"]["through_round"] == 152
    assert "authority-bound" in result["reason"]


def test_unknown_carrier_health_reason_stays_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    root = Path(arm["run_output_root"])
    pause_sha = _write_known_health_pause(root, unknown=True)
    _json(
        root / "execution-state.json",
        {
            "status": "PAUSE_FOR_DIAGNOSTIC",
            "next_round": 153,
            "final_panel_access_count": 0,
            "carrier_health_pause_sha256": pause_sha,
        },
    )
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "lock_busy", lambda root: False)
    result = us.discover(arm)
    assert result["action"] == "BLOCKED_NEEDS_HUMAN"
    assert "unknown health failure" in result["reason"]


def test_systemd_render_reuses_fail_closed_manual_start_bridge(tmp_path: Path) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    oe = tmp_path / "checkout" / "scripts" / "oe"
    rendered = us.render_service(cfg, oe)
    unit = rendered["unit"]
    receipt = rendered["receipt"]
    assert "Restart=no" in unit
    assert "KillMode=process" in unit
    assert "[Install]" not in unit
    assert receipt["restart_policy"] == "NONE_FAIL_CLOSED"
    assert receipt["manual_start_required"] is True
    assert receipt["scientific_retry_authorized"] is False
    assert receipt["kill_mode"] == "process"
    assert rendered["scientific_authority_granted"] is False


def test_config_requires_wait_and_run_root_placeholder(tmp_path: Path) -> None:
    path, doc = _base_config(tmp_path, arm_count=1)
    argv = doc["arms"][0]["launch"]["argv"]
    argv.remove("--wait")
    _json(path, doc)
    with pytest.raises(us.SupervisorError, match="--wait"):
        us.config(path)


def test_config_only_accepts_canonical_run_campaign_entrypoint(tmp_path: Path) -> None:
    path, doc = _base_config(tmp_path, arm_count=1)
    doc["arms"][0]["launch"]["argv"][0] = "/tmp/other/scripts/oe"
    _json(path, doc)
    with pytest.raises(us.SupervisorError, match="scripts/oe run-campaign"):
        us.config(path)


def test_default_cli_is_read_only_and_render_systemd_uses_existing_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    monkeypatch.setattr(
        us,
        "discover",
        lambda arm: {"action": "READY_TO_DISPATCH", "next_round": 152, "reason": "ready"},
    )
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(
        us.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("read-only run dispatched a child"),
    )
    assert oe_main(["supervise-campaign", "--config", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["assignments"]) == 1
    assert (Path(cfg["state_root"]) / "status.json").is_file()
    assert not (Path(cfg["state_root"]) / "dispatches").exists()

    assert oe_main(
        ["supervise-campaign", "--config", str(path), "--render-systemd"]
    ) == 0
    unit = json.loads(capsys.readouterr().out)["unit"]
    assert "Restart=no" in unit
    assert "[Install]" not in unit


def test_prior_launch_exit_without_state_or_authority_progress_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    root = Path(arm["run_output_root"])
    state_path = root / "execution-state.json"
    _json(
        state_path,
        {
            "status": "RUNNING",
            "next_round": 152,
            "final_panel_access_count": 0,
        },
    )
    marker = {
        "schema": "openevo.unattended-dispatch-terminal.20261004",
        "dispatch_id": "d1",
        "arm_id": arm["id"],
        "lane_id": "lane-a",
        "kind": "launch",
        "returncode": 2,
        "pre_state_sha256": us.sha(state_path),
        "pre_recovery_sha256": us.recovery_identity(arm)["sha256"],
        "pre_git_head": "same-head",
        "progress_cwd": arm["launch"]["cwd"],
    }
    us.atomic(us.terminal_marker_path(arm), marker)
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "git_head", lambda cwd: "same-head")
    monkeypatch.setattr(us, "lock_busy", lambda root: False)
    result = us.discover(arm)
    assert result["action"] == "BLOCKED_NEEDS_HUMAN"
    assert "without observable state/authority progress" in result["reason"]


def test_dry_run_formal_environment_parity_projects_selected_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    lane = cfg["lanes"][0]
    captured: dict[str, object] = {}

    class FakeProc:
        pid = 12345

        def poll(self):
            return None

        def terminate(self):
            return None

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return FakeProc()

    monkeypatch.setenv("OPENEVO_AUTHORIZED_GPU_UUIDS", "GPU-00000000-0000-0000-0000-000000000000")
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "git_head", lambda cwd: "a" * 40)
    monkeypatch.setattr(us.subprocess, "Popen", fake_popen)
    child = us.start(cfg, arm, lane)
    try:
        assert captured["env"]["OPENEVO_AUTHORIZED_GPU_UUIDS"] == ",".join(
            lane["gpu_uuids"]
        )
        assert captured["argv"][captured["argv"].index("--gpus") + 1] == ",".join(
            lane["gpu_uuids"]
        )
        assert captured["env"]["OPENEVO_AUTHORIZED_GPU_UUIDS"] != "GPU-00000000-0000-0000-0000-000000000000"
    finally:
        child["stream"].close()


def test_dispatch_terminal_receipt_carries_launch_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    lane = cfg["lanes"][0]

    class ExitedProc:
        pid = 24680

        def poll(self):
            return 17

    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "git_head", lambda cwd: "a" * 40)
    monkeypatch.setattr(us.subprocess, "Popen", lambda *args, **kwargs: ExitedProc())
    child = us.start(cfg, arm, lane)
    us.reap(cfg, {arm["id"]: child})
    terminal_path = (
        Path(cfg["state_root"])
        / "dispatches"
        / arm["id"]
        / child["dispatch_id"]
        / "terminal.json"
    )
    terminal = json.loads(terminal_path.read_text())
    assert terminal["config_sha256"] == cfg["_sha256"]
    assert terminal["arm_id"] == arm["id"]
    assert terminal["lane_id"] == lane["id"]
    assert terminal["gpu_uuids"] == lane["gpu_uuids"]
    assert terminal["argv"][1] == "run-campaign"
    assert terminal["pre_git_head"] == "a" * 40
    assert terminal["pre_recovery_sha256"] == child["pre_recovery_sha256"]
    assert terminal["pre_state_sha256"] is None
    assert terminal["returncode"] == 17
    assert terminal["scientific_authority_granted_by_supervisor"] is False


def test_dispatch_refuses_checkout_head_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    lane = cfg["lanes"][0]
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "git_head", lambda cwd: "b" * 40)
    with pytest.raises(us.SupervisorError, match="HEAD drift"):
        us.start(cfg, arm, lane)


def test_state_progress_supersedes_old_terminal_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    root = Path(arm["run_output_root"])
    old_state = root / "old-state.json"
    _json(old_state, {"status": "RUNNING", "next_round": 152})
    _json(root / "execution-state.json", {"status": "RUNNING", "next_round": 152})
    pre_recovery = us.recovery_identity(arm)
    pre_recovery_sha = pre_recovery["sha256"]
    us.atomic(
        us.terminal_marker_path(arm),
        {
            "schema": "openevo.unattended-dispatch-terminal.20261004",
            "dispatch_id": "old",
            "arm_id": arm["id"],
            "lane_id": "lane-a",
            "kind": "launch",
            "returncode": 1,
            "pre_state_sha256": us.sha(old_state),
            "pre_recovery_sha256": pre_recovery_sha,
            "pre_recovery_projection": pre_recovery["projection"],
            "pre_git_head": "head",
            "progress_cwd": arm["launch"]["cwd"],
        },
    )
    from tests.test_execution_runtime_round_transaction_20260916 import write_complete_prefix
    write_complete_prefix(root / "round-152", "round_complete", round_index=152)
    _json(
        root / "execution-state.json",
        {
            "status": "RUNNING",
            "next_round": 153,
            "final_panel_access_count": 0,
        },
    )
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "git_head", lambda cwd: "head")
    monkeypatch.setattr(us, "lock_busy", lambda root: False)
    result = us.discover(arm)
    assert result["action"] == "READY_TO_DISPATCH"
    assert result["next_round"] == 153


def test_health_pause_hash_drift_blocks_automatic_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    root = Path(arm["run_output_root"])
    pause_sha = _write_known_health_pause(root)
    pause_path = root / "carrier-health-pauses" / "round-152.json"
    pause = json.loads(pause_path.read_text())
    pause["health_ledger_sha256"] = "0" * 64
    _json(pause_path, pause)
    _json(
        root / "execution-state.json",
        {
            "status": "PAUSE_FOR_DIAGNOSTIC",
            "next_round": 153,
            "final_panel_access_count": 0,
            "carrier_health_pause_sha256": pause_sha,
        },
    )
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "lock_busy", lambda root: False)
    result = us.discover(arm)
    assert result["action"] == "BLOCKED_NEEDS_HUMAN"
    assert "exact sealed transaction hashes" in result["reason"]


def test_bound_health_resolution_requires_exact_committed_same_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, doc = _base_config(tmp_path, arm_count=1)
    authority_path = tmp_path / "checkout" / "authority.json"
    doc["arms"][0]["authority_path"] = str(authority_path)
    _json(path, doc)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    detail = {
        "through_round": 152,
        "pause_sha256": "1" * 64,
        "health_ledger_sha256": "2" * 64,
        "boundary_decision_sha256": "3" * 64,
        "round_complete_sha256": "4" * 64,
        "state_after_sha256": "5" * 64,
    }
    resolution = tmp_path / "checkout" / "resolution.json"
    authority = {
        "carrier_health_resolution": {
            "path": "resolution.json",
            "sha256": "6" * 64,
        }
    }
    receipt = {
        "schema": "openevo.carrier-health-resolution.v1",
        "status": "RESOLVED",
        "contract_transition": "SAME_CONTRACT",
        "through_round": 152,
        "output_root": arm["run_output_root"],
        "external_teacher_calls": 0,
        "final_panel_access": 0,
        "automatic_scientific_change": False,
        "no_change": {
            "formal_budget": True,
            "repair_call_limit": True,
            "sampling": True,
            "scientific_algorithm": True,
            "token_cap": True,
        },
        **{
            key: detail[key]
            for key in (
                "pause_sha256",
                "health_ledger_sha256",
                "boundary_decision_sha256",
                "round_complete_sha256",
                "state_after_sha256",
            )
        },
    }

    def fake_committed_json(repo: Path, candidate: Path) -> dict:
        if candidate == authority_path:
            return authority
        if candidate == resolution:
            return receipt
        raise AssertionError(candidate)

    monkeypatch.setattr(us, "committed_json", fake_committed_json)
    monkeypatch.setattr(us, "sha", lambda candidate: "6" * 64)
    ok, info = us.bound_health_resolution(arm, detail)
    assert ok is True
    assert info["through_round"] == 152

    receipt["final_panel_access"] = 1
    ok, info = us.bound_health_resolution(arm, detail)
    assert ok is False
    assert "exact preserved pause" in info["reason"]


def test_config_requires_exact_head_and_whole_value_env_tokens(tmp_path: Path) -> None:
    path, doc = _base_config(tmp_path, arm_count=1)
    del doc["arms"][0]["launch"]["expected_git_head"]
    _json(path, doc)
    with pytest.raises(us.SupervisorError, match="expected_git_head"):
        us.config(path)

    path, doc = _base_config(tmp_path / "second", arm_count=1)
    doc["arms"][0]["launch"]["env"]["OPENEVO_AUTHORIZED_GPU_UUIDS"] = (
        "prefix-${GPU_UUIDS}"
    )
    _json(path, doc)
    with pytest.raises(us.SupervisorError, match="token must occupy whole value"):
        us.config(path)

    path, doc = _base_config(tmp_path / "third", arm_count=1)
    del doc["arms"][0]["launch"]["env"]["OPENEVO_AUTHORIZED_GPU_UUIDS"]
    _json(path, doc)
    with pytest.raises(us.SupervisorError, match="OPENEVO_AUTHORIZED_GPU_UUIDS"):
        us.config(path)


def test_arm_specific_sealed_status_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _base_config(tmp_path, arm_count=1)
    cfg = us.config(path)
    arm = cfg["arms"][0]
    root = Path(arm["run_output_root"])
    _json(
        root / "execution-state.json",
        {
            "status": "RANK16_SHORT_FORK_SEALED",
            "next_round": 154,
            "final_panel_access_count": 0,
        },
    )
    monkeypatch.setattr(us, "git_clean", lambda cwd, expected_head=None: (True, "clean"))
    monkeypatch.setattr(us, "lock_busy", lambda root: False)
    result = us.discover(arm)
    assert result["action"] == "COMPLETE"
    assert result["reason"] == "arm sealed"
