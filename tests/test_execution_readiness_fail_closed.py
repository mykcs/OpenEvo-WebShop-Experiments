"""Fail-closed regression tests for the scientific execution boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.openevo_webshop import execution_readiness


REPO_ROOT = Path(__file__).parents[1]


def test_missing_execution_profile_fails_closed_before_runtime_work(tmp_path: Path) -> None:
    with pytest.raises(execution_readiness.ExecutionReadinessError) as excinfo:
        execution_readiness.resolve_execution_profile(
            {"execution_readiness": {}}, repo_root=tmp_path
        )
    assert excinfo.value.verdict == execution_readiness.VERDICT_NOT_PREPARED
    assert "hash-bound execution_readiness.profile" in excinfo.value.reason


def test_scientific_identity_is_mandatory(tmp_path: Path) -> None:
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps({"frozen_scientific_contract": {}}), encoding="utf-8")
    with pytest.raises(execution_readiness.ExecutionReadinessError) as excinfo:
        execution_readiness.require_scientific_identity_document(prereg)
    assert excinfo.value.verdict == execution_readiness.VERDICT_NOT_PREPARED
    assert "scientific identity must be explicit" in excinfo.value.reason


def test_router_release_hash_is_mandatory_and_well_formed() -> None:
    with pytest.raises(execution_readiness.ExecutionReadinessError) as missing:
        execution_readiness._require_router_release_sha256({"latest_event": {}})
    assert missing.value.verdict == execution_readiness.VERDICT_BLOCKED

    with pytest.raises(execution_readiness.ExecutionReadinessError) as malformed:
        execution_readiness._require_router_release_sha256(
            {"latest_event": {"release_receipt_sha256": "not-a-sha"}}
        )
    assert malformed.value.verdict == execution_readiness.VERDICT_BLOCKED

    digest = "a" * 64
    assert execution_readiness._require_router_release_sha256(
        {"latest_event": {"release_receipt_sha256": digest}}
    ) == f"sha256:{digest}"


def test_router_release_pointer_may_not_escape_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside-release.json"
    outside.write_text("{}", encoding="utf-8")
    router = repo / "current-campaign.json"
    router.write_text(
        json.dumps(
            {
                "campaign_id": "c1",
                "latest_event": {"activation": "../outside-release.json"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(execution_readiness.ExecutionReadinessError) as excinfo:
        execution_readiness.resolve_release_path(
            "c1",
            None,
            current_campaign_path=router,
            repo_root=repo,
        )
    assert excinfo.value.verdict == execution_readiness.VERDICT_BLOCKED
    assert "escapes" in excinfo.value.reason


def test_router_absolute_release_pointer_may_not_escape_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside-release.json"
    outside.write_text("{}", encoding="utf-8")
    router = repo / "current-campaign.json"
    router.write_text(
        json.dumps(
            {
                "campaign_id": "c1",
                "latest_event": {"activation": str(outside)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(execution_readiness.ExecutionReadinessError) as excinfo:
        execution_readiness.resolve_release_path(
            "c1",
            None,
            current_campaign_path=router,
            repo_root=repo,
        )
    assert excinfo.value.verdict == execution_readiness.VERDICT_BLOCKED
    assert "escapes" in excinfo.value.reason


def test_execution_readiness_script_help_is_directly_runnable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "openevo_webshop" / "execution_readiness.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "fingerprint-file" in result.stdout


def test_legacy_oe_run_is_disabled_before_profile_or_gpu_checks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "oe"),
            "run",
            "/definitely/missing/profile.json",
            "--gpus",
            "auto:1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "direct 'oe run <profile>' GPU execution is disabled" in result.stderr
    assert "profile unreadable" not in result.stderr
    assert "nvidia-smi" not in result.stderr
