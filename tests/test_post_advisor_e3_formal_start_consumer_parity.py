from pathlib import Path

import pytest

from scripts.openevo_webshop.post_advisor_e3_guard_202609300248 import validate_e3_formal_start_receipt_location


def test_accepts_transition_visible_subtree(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "campaign"
    receipt = runs / "arm" / "preflight" / "control" / "formal-start-attempt-0001.json"
    validate_e3_formal_start_receipt_location(receipt, runs_root=runs)


def test_rejects_outer_control_root_before_science(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "campaign"
    outer = tmp_path / "control" / "formal-start-attempt-0001.json"
    with pytest.raises(RuntimeError, match="outside the transition-visible campaign runs root"):
        validate_e3_formal_start_receipt_location(outer, runs_root=runs)


def test_rejects_relative_path(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="must be absolute"):
        validate_e3_formal_start_receipt_location(Path("formal-start.json"), runs_root=tmp_path / "runs")
