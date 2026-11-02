from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import validate_frozen_passport_immutability as guard


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _write_passport(root: Path, experiment_id: str, status: str) -> Path:
    path = root / "configs" / "experiment" / "passports" / experiment_id / "passport.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"experiment_id": experiment_id, "status": status}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _repo(tmp_path: Path, status: str = "FROZEN") -> tuple[Path, str, Path]:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    experiment_id = "202609120900-parent"
    path = _write_passport(root, experiment_id, status)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base")
    _git(root, "switch", "-c", "feature")
    return root, experiment_id, path


def test_frozen_passport_modified_in_place_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, experiment_id, path = _repo(tmp_path)
    payload = json.loads(path.read_text())
    payload["scientific_identity"] = {"treatment": "changed"}
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "bad mutation")
    monkeypatch.chdir(root)
    failures = guard.check("main")
    assert len(failures) == 1
    assert "FROZEN Passport" in failures[0]
    assert f"--successor-of {experiment_id}" in failures[0]


def test_frozen_passport_delete_and_rename_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _, path = _repo(tmp_path)
    path.unlink()
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "delete")
    monkeypatch.chdir(root)
    assert "deleted" in guard.check("main")[0]

    root2, _, path2 = _repo(tmp_path / "rename")
    new_path = path2.parent.parent / "202609120901-renamed" / "passport.json"
    new_path.parent.mkdir(parents=True)
    path2.replace(new_path)
    _git(root2, "add", "-A")
    _git(root2, "commit", "-m", "rename")
    monkeypatch.chdir(root2)
    assert "renamed" in guard.check("main")[0]


def test_draft_passport_can_change_and_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _, path = _repo(tmp_path, status="DRAFT")
    payload = json.loads(path.read_text())
    payload["status"] = "FROZEN"
    payload["scientific_identity"] = {"treatment": "filled before freeze"}
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "freeze draft")
    monkeypatch.chdir(root)
    assert guard.check("main") == []


def test_new_successor_keeps_frozen_parent_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, parent, _ = _repo(tmp_path)
    successor = "202609120902-successor"
    path = _write_passport(root, successor, "DRAFT")
    payload = json.loads(path.read_text())
    payload["parent_experiment_id"] = parent
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "add successor")
    monkeypatch.chdir(root)
    assert guard.check("main") == []
