import json
from pathlib import Path

import pytest

from scripts.openevo_webshop.formal_priority_witness import (
    FormalPriorityWitnessError,
    build_formal_priority_witness,
)
from scripts.openevo_webshop.opportunistic_canary_guard import WITNESS_SCHEMA

EXECUTION = "4" * 40
TOPOLOGY = "5" * 64


def fixture(tmp_path: Path):
    repo = tmp_path / "formal"; repo.mkdir(parents=True)
    ready = tmp_path / "READY.json"; release = tmp_path / "release.json"
    off = tmp_path / "off"; on = tmp_path / "on"; off.mkdir(); on.mkdir()
    for root, next_round in ((off, 9), (on, 12)):
        (root / "execution-state.json").write_text(json.dumps({
            "status": "RUNNING", "next_round": next_round,
            "rollouts_consumed": next_round * 128, "final_panel_access_count": 0,
        }))
    ready.write_text(json.dumps({
        "status": "FORMAL_READY_AWAITING_OWNER_LAUNCH", "execution_git_sha": EXECUTION,
        "arms": {"off": {"output_root": str(off)}, "on": {"output_root": str(on)}},
    }))
    release.write_text(json.dumps({"status": "AUTHORIZED_FORMAL_LAUNCH", "execution_git_sha": EXECUTION}))
    ray = tmp_path / "ray"; ray.mkdir(); (ray / "state.lock").write_text("")
    return repo, ready, release, ray


def state(ray: Path, *, reservations=None, leases=None):
    per_gpu = {str(i): {"leases": []} for i in range(4, 8)}
    for lease in leases or []:
        for index in lease["gpus"]:
            per_gpu[str(index)]["leases"].append(lease)
    (ray / "state.json").write_text(json.dumps({
        "schema": "openevo.ceiling1-ray-placement-state.202609042025",
        "reservations": reservations or {}, "gpus": per_gpu,
    }))


def build(tmp_path: Path, *, probes=([], []), reservations=None, leases=None):
    repo, ready, release, ray = fixture(tmp_path); state(ray, reservations=reservations, leases=leases)
    rows = iter(probes)
    return build_formal_priority_witness(
        formal_repo=repo, ready_path=ready, release_path=release, ray_state_root=ray,
        expected_execution_git_sha=EXECUTION, topology_id=TOPOLOGY, candidate_gpu_indices=[4],
        head_reader=lambda _: EXECUTION, pending_probe=lambda *_: next(rows), now_epoch=1000.0,
    )


def test_idle_witness_is_read_only_and_grants_zero_authority(tmp_path: Path) -> None:
    doc = build(tmp_path)
    assert doc["schema"] == WITNESS_SCHEMA
    assert doc["scheduler_status"] == "IDLE"
    assert doc["pending_formal_gpu_requests"] == 0
    assert doc["active_formal_reservations"] == 0
    assert doc["active_formal_gpu_leases"] == 0
    assert doc["final_panel_access_count"] == 0
    assert doc["gpu_requests"] == 0
    assert doc["canary_execution_authority"] is False


def test_active_lease_is_deduplicated_across_gpu_rows(tmp_path: Path) -> None:
    lease = {"lease_id": "formal-rollout", "gpus": [4, 5, 6, 7]}
    doc = build(tmp_path, leases=[lease])
    assert doc["scheduler_status"] == "ACTIVE"
    assert doc["active_formal_gpu_leases"] == 1
    assert doc["active_formal_gpu_lease_ids"] == ["formal-rollout"]


def test_pending_or_transitioning_probe_never_reports_idle(tmp_path: Path) -> None:
    row = {"arm_key": "off", "round_index": 8}
    assert build(tmp_path / "a", probes=([row], [row]))["scheduler_status"] == "PENDING"
    assert build(tmp_path / "b", probes=([], [row]))["scheduler_status"] == "TRANSITIONING"


def test_final_panel_nonzero_is_preserved_in_witness(tmp_path: Path) -> None:
    repo, ready, release, ray = fixture(tmp_path); state(ray)
    off = Path(json.loads(ready.read_text())["arms"]["off"]["output_root"])
    doc = json.loads((off / "execution-state.json").read_text()); doc["final_panel_access_count"] = 1
    (off / "execution-state.json").write_text(json.dumps(doc))
    rows = iter(([], []))
    witness = build_formal_priority_witness(
        formal_repo=repo, ready_path=ready, release_path=release, ray_state_root=ray,
        expected_execution_git_sha=EXECUTION, topology_id=TOPOLOGY, candidate_gpu_indices=[4],
        head_reader=lambda _: EXECUTION, pending_probe=lambda *_: next(rows), now_epoch=1000.0,
    )
    assert witness["final_panel_access_count"] == 1


def test_execution_head_drift_fails_closed(tmp_path: Path) -> None:
    repo, ready, release, ray = fixture(tmp_path); state(ray)
    with pytest.raises(FormalPriorityWitnessError, match="HEAD drift"):
        build_formal_priority_witness(
            formal_repo=repo, ready_path=ready, release_path=release, ray_state_root=ray,
            expected_execution_git_sha=EXECUTION, topology_id=TOPOLOGY, candidate_gpu_indices=[4],
            head_reader=lambda _: "6" * 40, pending_probe=lambda *_: [], now_epoch=1000.0,
        )
