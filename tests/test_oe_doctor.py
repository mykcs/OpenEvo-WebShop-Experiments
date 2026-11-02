"""Regression tests for the canonical fast-start entrypoint (scripts/oe doctor)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OE = REPO_ROOT / "scripts" / "oe"


def _run_oe(*args: str, env: dict[str, str] | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, str(OE), "doctor", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )


def test_doctor_help_no_longer_requires_profile():
    """Fresh-Agent mode must not require a positional profile."""
    proc = _run_oe("--help")
    assert proc.returncode == 0, proc.stderr
    assert "[profile]" in proc.stdout


def test_doctor_from_non_canonical_checkout_fails_fast(tmp_path: Path):
    """Running from a non-canonical checkout must fail before expensive runtime work."""
    registry = tmp_path / "server-registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "openevo.server-workspace-registry.v1",
                "host": "test-host",
                "canonical_control_checkout": "/does/not/exist/control",
                "canonical_runtime_checkout": "/does/not/exist/runtime",
                "checkouts": {},
            }
        ),
        encoding="utf-8",
    )
    proc = _run_oe("--registry", str(registry), "--no-run-campaign-dry-run")
    assert proc.returncode == 3, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["next_action"] == "NON_CANONICAL_EXECUTION_CHECKOUT"
    assert receipt["canonical_checkout"]["is_canonical_control"] is False
    assert receipt["runtime_qualification"] is None


def test_doctor_blocked_campaign_closed(tmp_path: Path):
    """A closed campaign must BLOCK before any runtime/GPU work."""
    env = {"OPENEVO_CANONICAL_CONTROL_CHECKOUT": str(REPO_ROOT)}
    closed_campaign = tmp_path / "closed-campaign.json"
    closed_campaign.write_text(
        json.dumps(
            {
                "campaign_id": "test-closed-campaign",
                "status": "closed-test",
                "formal_task_consumption_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    proc = _run_oe(
        "--no-run-campaign-dry-run",
        "--current-campaign",
        str(closed_campaign),
        env=env,
    )
    assert proc.returncode == 4, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["next_action"] == "BLOCKED_CAMPAIGN_CLOSED"
    assert receipt["runtime_qualification"] is None


def test_legacy_run_profile_is_disabled():
    """Legacy 'oe run <profile>' must remain fail-closed."""
    proc = subprocess.run(
        [sys.executable, str(OE), "run", "some-profile.json", "--gpus", "auto:1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "disabled" in proc.stderr.lower() or "bypasses" in proc.stderr.lower()
