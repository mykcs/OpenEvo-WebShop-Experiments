from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "openevo_webshop"))

from build_h0_manifest import (  # noqa: E402
    EXCLUDED_GOAL_INDICES,
    H0_TASK_COUNT,
    select_h0_indices,
)


def test_h0_selection_is_frozen_and_disjoint_from_phase_g() -> None:
    selected = select_h0_indices(6910)
    assert selected == [2822, 5135, 5294, 6769]
    assert len(selected) == H0_TASK_COUNT
    assert len(set(selected)) == H0_TASK_COUNT
    assert not set(selected) & EXCLUDED_GOAL_INDICES


def test_h0_selection_changes_only_when_selection_seed_changes() -> None:
    first = select_h0_indices(6910, selection_seed=20260814)
    second = select_h0_indices(6910, selection_seed=20260815)
    assert first != second
    assert all(index >= 500 for index in second)
    assert not set(second) & EXCLUDED_GOAL_INDICES
