from __future__ import annotations

import json
from pathlib import Path

from scripts.openevo_webshop.audit_seedcmp_wrapper_drift import DEFAULT_ROOT, build_report


RECEIPT = Path("docs/evidence/seed-webshop-public-code-reproduction-v1/WRAPPER_AUDIT.json")


def test_seedcmp_wrapper_audit_scientific_invariants() -> None:
    report = build_report(DEFAULT_ROOT)
    assert report["totals"]["episodes"] == 256
    assert report["arms"]["base"]["episode_count"] == 128
    assert report["arms"]["sd-lora"]["episode_count"] == 128
    assert report["interpretation"]["wrapper_drift_observed"] is True
    assert report["interpretation"]["unrecoverable_steps_fail_closed"] is True
    assert report["interpretation"]["measurement_pollution_observed"] is False
    assert report["totals"]["unrecoverable_fail_closed_steps"] == 119
    assert report["totals"]["unrecoverable_fail_closed_steps"] == report["totals"]["recorded_invalid_action_steps"]
    assert report["totals"]["unrecoverable_fail_closed_steps"] == report["totals"]["environment_invalid_steps"]
    assert report["arms"]["base"]["unknown_wrapper_steps"] == 0
    assert report["arms"]["sd-lora"]["unknown_wrapper_steps"] == 0
    assert report["totals"]["formal_invalid_episode_count_from_raw"] == 0
    assert report["totals"]["formal_summary_invalid_count"] == 0
    assert report["totals"]["closeout_parser_invalid_total"] == 0


def test_seedcmp_wrapper_audit_receipt_recomputes_exactly() -> None:
    expected = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert build_report(DEFAULT_ROOT) == expected
