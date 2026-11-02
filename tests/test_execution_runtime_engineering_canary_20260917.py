from __future__ import annotations

from pathlib import Path

import pytest

from scripts.openevo_runtime.engineering_canary import (
    EngineeringCanaryError,
    _execute_once,
    run_canary_cycle,
)


def test_non_scientific_engineering_canary_materializes_once_then_reads_back(tmp_path: Path) -> None:
    report = run_canary_cycle(tmp_path / "canary", "1" * 40)
    assert report["status"] == "PASS_NON_SCIENTIFIC_ENGINEERING_CANARY"
    assert report["first_pass"] == "MATERIALIZED_NEW"
    assert report["second_pass"] == "READBACK_EXISTING_TERMINAL"
    assert report["materialization_count"] == 1
    assert report["readback_count"] == 1
    assert report["event_reconciliation"]["reconstructed_missing"] == 1
    assert report["event_reconciliation"]["ignored_duplicates"] == 1
    assert report["event_reconciliation"]["framework_history_authority"] is False
    assert report["formal_rows_consumed"] == 0
    assert report["model_calls"] == 0
    assert report["gpu_requests"] == 0
    assert report["final_panel_access_count"] == 0
    assert report["scientific_state_mutated"] is False
    assert report["scientific_authority_granted"] is False
    assert report["live_formal_execution_mutated"] is False


def test_canary_readback_fails_closed_if_materialized_artifact_changes(tmp_path: Path) -> None:
    scratch = tmp_path / "canary"
    report = run_canary_cycle(scratch, "2" * 40)
    artifact = scratch / report["operation_digest"] / "artifact.bin"
    artifact.write_bytes(b"tampered\n")
    with pytest.raises(EngineeringCanaryError, match="artifact readback SHA mismatch"):
        _execute_once(scratch, "2" * 40)


def test_engineering_canary_direct_cli_import_path_is_self_contained(tmp_path: Path) -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "scripts/openevo_runtime/engineering_canary.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--scratch-root" in completed.stdout
