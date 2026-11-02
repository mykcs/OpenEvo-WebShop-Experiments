from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.openevo_webshop import wb1_gen28_state_repair as repair


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "WB1-ON-S-seed-20260821-generation-28"
    attempt = root / "attempt-01"
    attempt.mkdir(parents=True)

    rows = []
    for task_offset in range(16):
        goal_idx = 500 + task_offset
        for rollout in range(8):
            rows.append(
                {
                    "goal_idx": goal_idx,
                    "valid_completed": True,
                    "won": task_offset == 0 and rollout < 2,
                }
            )
    (attempt / "episodes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    for name in ("summary.json", "manifest.json", "config.json"):
        (attempt / name).write_text(
            json.dumps({"execution_sha": repair.HISTORICAL_EXECUTION_SHA}) + "\n",
            encoding="utf-8",
        )

    state = {
        "state_format": "native-openevo-static-markdown-v1",
        "state_version": 27,
        "state_sha256": "fixture-state-checksum",
        "memory_markdown": "fixture memory",
        "adapter_write": False,
        "optimizer_write": False,
        "parameter_update": False,
    }
    state_path = root / "state-v27.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    memory_path = root / "memory-v27.md"
    memory_path.write_text("fixture memory", encoding="utf-8")
    task_path = root / "task-ids-gen28.json"
    task_path.write_text(json.dumps(list(range(500, 516))) + "\n", encoding="utf-8")

    monkeypatch.setattr(repair, "STATE_V27_CHECKSUM", "fixture-state-checksum")
    monkeypatch.setattr(repair, "STATE_V27_FILE_SHA256", _sha(state_path))
    monkeypatch.setattr(repair, "MEMORY_V27_SHA256", _sha(memory_path))
    monkeypatch.setattr(repair, "TASK_STREAM_SHA256", _sha(task_path))
    return root


def test_gen28_input_seal_passes_on_exact_frozen_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    seal = repair.build_seal(root)
    assert seal["status"] == "PASS"
    assert seal["episode_contract"]["valid_completed"] == 128
    assert seal["episode_contract"]["exact_successes"] == 2
    assert seal["episode_contract"]["unique_tasks"] == 16
    assert seal["episode_contract"]["rollouts_per_task"] == 8
    assert seal["episode_contract"]["final_pool_tasks"] == 0
    assert seal["state_v28_preexisting"] is False
    assert seal["seal_sha256"] == repair._document_sha(seal, "seal_sha256")


def test_gen28_input_seal_rejects_final_pool_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    episodes = root / "attempt-01" / "episodes.jsonl"
    rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()]
    rows[0]["goal_idx"] = 42
    episodes.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(repair.RepairError, match="final-pool leakage"):
        repair.build_seal(root)


def test_gen28_input_seal_rejects_existing_state_v28(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    (root / "state-v28.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(repair.RepairError, match="pre-existing state-v28"):
        repair.build_seal(root)


def test_apply_gate_rejects_input_change_after_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    seal = repair.build_seal(root)
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    episodes = root / "attempt-01" / "episodes.jsonl"
    episodes.write_bytes(episodes.read_bytes() + b"\n")
    with pytest.raises(repair.RepairError, match="inputs changed after sealing"):
        repair._assert_seal_matches(seal_path, root)


def test_git_blob_sha_matches_git_object_formula() -> None:
    content = b"hello\n"
    expected = hashlib.sha1(b"blob 6\0hello\n").hexdigest()
    assert repair._git_blob_sha(content) == expected
