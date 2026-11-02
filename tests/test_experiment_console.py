from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.openevo_webshop import experiment_console as console


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "experiment" / "experiment-console-202609110009.json"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_catalog_is_read_only_and_covers_major_lineages() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert console.validate_catalog(ROOT, catalog) == []
    assert catalog["role"] == "operator-observability-only; never scientific or execution authority"
    assert [row["id"] for row in catalog["experiments"]] == [
        "qwen25-7b-long-run",
        "qwen3-1p7b-gdr",
        "qwen25-3b-legacy-stage2",
        "qwen25-3b-h2011-corrected",
        "qwen3-1p7b-directapply-no-gdr",
    ]

def test_completed_gdr_line_reads_immutable_closeout() -> None:
    catalog = console._read_json(CATALOG)
    snapshot = console.build_snapshot(ROOT, catalog)
    row = next(item for item in snapshot["experiments"] if item["id"] == "qwen3-1p7b-gdr")
    assert row["observation"]["effective_status"] == "COMPLETE"
    assert row["observation"]["identity_trust"] == "PASS"
    assert row["observation"]["metrics"] == {
        "rounds": 160,
        "rollouts": 20480,
        "external_teacher_calls": 0,
        "final_panel_access_count": 0,
        "accepted_parametric_updates": 7,
    }
    assert row["runtime"]["image_digest"] == (
        "sha256:d0f8796010899ae41a7af8ae3f44f47b96dea73b297b5af124049b3ae88e2850"
    )


def test_corrected_3b_lifecycle_is_recovery_bound_not_legacy_resume() -> None:
    catalog = console._read_json(CATALOG)
    snapshot = console.build_snapshot(ROOT, catalog)
    source_row = next(item for item in catalog["experiments"] if item["id"] == "qwen25-3b-h2011-corrected")
    row = next(item for item in snapshot["experiments"] if item["id"] == source_row["id"])
    assert row["treatment"] == "corrected Harness201.1 Stage2 successor with gated candidate acceptance"
    assert row["resume"]["policy"] == "RECONCILE_THEN_EXACT_RESUME_ONLY"
    assert "zero-replay" in row["resume"]["next_action"]
    assert row["observation"]["release_path"] == source_row["state_source"]["release_path"]
    assert row["observation"]["effective_status"] in {
        "LIVE_UNAVAILABLE_OFF_SERVER", "PAUSED", "RUNNING", "COMPLETE"
    }
    if row["observation"]["observation_scope"] == "off-server":
        assert row["observation"]["identity_trust"] == "NOT_OBSERVED"
    else:
        assert row["observation"]["identity_trust"] == "PASS"


def test_server_json_identity_check_and_progress(tmp_path: Path) -> None:
    state = tmp_path / "execution-state.json"
    write_json(state, {
        "schema": "openevo.ceiling1-stage2-execution-state.v1",  # legacy-revision-compat
        "arm": "qwen3-1p7b",
        "status": "RUNNING",
        "next_round": 12,
        "rounds_complete": list(range(12)),
        "rollouts_consumed": 1536,
        "vnext_candidate_accept_count": 11,  # legacy-revision-compat
    })
    result = console._observe_server_json({
        "path": str(state),
        "required_equal": {
            "schema": "openevo.ceiling1-stage2-execution-state.v1",  # legacy-revision-compat
            "arm": "qwen3-1p7b",
        },
    })
    assert result["effective_status"] == "RUNNING"
    assert result["identity_trust"] == "PASS"
    assert result["metrics"]["rounds_complete"] == 12
    assert result["metrics"]["rollouts_consumed"] == 1536
    assert result["metrics"]["vnext_candidate_accept_count"] == 11  # legacy-revision-compat


def test_release_live_state_overrides_stale_missing_note(tmp_path: Path) -> None:
    state = tmp_path / "live" / "execution-state.json"
    write_json(state, {
        "schema": "openevo.ceiling1-stage2-execution-state.v1",  # legacy-revision-compat
        "arm": "3b",
        "status": "BLOCKED",
        "next_round": 132,
        "rounds_complete": list(range(132)),
        "rollouts_consumed": 16896,
        "blocker": "RuntimeError: receipt readback mismatch",
    })
    release = tmp_path / "release.json"
    write_json(release, {
        "status": "ADOPTED",
        "external_executor": {
            "state_path": str(state),
            "missing_state": "MANUAL_REVIEW_REQUIRED",
            "missing_reason": "old snapshot said state was absent",
            "required_equal": {
                "schema": "openevo.ceiling1-stage2-execution-state.v1",  # legacy-revision-compat
                "arm": "3b",
            },
            "status_map": {"BLOCKED": "PAUSED", "RUNNING": "RUNNING"},
        },
    })
    result = console._observe_release_external_executor(
        tmp_path,
        {"release_path": "release.json"},
    )
    assert result["effective_status"] == "PAUSED"
    assert result["raw_live_status"] == "BLOCKED"
    assert result["identity_trust"] == "PASS"
    assert result["metrics"]["rounds_complete"] == 132
    assert result["metrics"]["blocker"] == "RuntimeError: receipt readback mismatch"


def test_oe_experiments_check_routes_through_canonical_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "oe"), "experiments", "check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {"experiment_count": 5, "mutation": False, "status": "PASS"}


def test_operator_docs_keep_console_below_execution_authority() -> None:
    master = (ROOT / "docs/experiment-tracking/OPEN_EVO_CEILING_1.0_END_TO_END_MASTER_PLAN.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/operations/execution/EXPERIMENT_CONSOLE.md").read_text(encoding="utf-8")
    for text in (master, agents, readme, guide):
        assert "./scripts/oe experiments status" in text
    normalized_guide = " ".join(guide.lower().split())
    assert "it is an observability layer, never scientific or execution authority." in normalized_guide
    assert "or grant resume authority" in normalized_guide
    assert "read-only" in normalized_guide
    assert "does not grant execution authority" in guide
    assert "must not duplicate mutable\nlive progress into Git" in guide


def test_registry_overlay_is_incremental_and_legacy_console_remains_compatible() -> None:
    catalog = console._read_json(CATALOG)
    snapshot = console.build_snapshot(ROOT, catalog)
    rows = {row["id"]: row for row in snapshot["experiments"]}
    assert set(rows) == {
        "qwen25-7b-long-run", "qwen3-1p7b-gdr", "qwen25-3b-legacy-stage2",
        "qwen25-3b-h2011-corrected", "qwen3-1p7b-directapply-no-gdr",
    }
    q17 = rows["qwen3-1p7b-directapply-no-gdr"]
    assert q17["static_identity_source"] == "experiment-registry/passport"
    assert q17["canonical_experiment_id"] == "202609082022-q17-directapply-no-gdr"
    assert q17["passport_sha256"]
    for legacy_id in set(rows) - {"qwen3-1p7b-directapply-no-gdr"}:
        assert rows[legacy_id]["static_identity_source"] == "legacy-console-compat"
        assert rows[legacy_id]["canonical_experiment_id"] is None
        assert rows[legacy_id]["passport_sha256"] is None


def test_passport_definition_of_done_covers_four_lifecycle_classes() -> None:
    catalog = console._read_json(CATALOG)
    rows = {row["id"]: row for row in console.build_snapshot(ROOT, catalog)["experiments"]}

    q17 = rows["qwen3-1p7b-directapply-no-gdr"]
    assert q17["static_identity_source"] == "experiment-registry/passport"
    assert q17["canonical_experiment_id"] == "202609082022-q17-directapply-no-gdr"
    assert q17["passport_sha256"]
    assert q17["resume"]["policy"] == "EXACT_RECEIPT_LINEAGE_ONLY"

    corrected = rows["qwen25-3b-h2011-corrected"]
    assert corrected["resume"]["policy"] == "RECONCILE_THEN_EXACT_RESUME_ONLY"
    assert corrected["observation"]["effective_status"] in {
        "LIVE_UNAVAILABLE_OFF_SERVER", "RUNNING", "PAUSED", "BLOCKED", "COMPLETE"
    }

    completed = rows["qwen3-1p7b-gdr"]
    assert completed["observation"]["effective_status"] == "COMPLETE"
    assert completed["resume"]["policy"] == "COMPLETE_NO_RESUME_NEEDED"

    legacy = rows["qwen25-3b-legacy-stage2"]
    assert legacy["observation"]["effective_status"] == "NO_RESUME"
    assert legacy["resume"]["policy"] == "NO_RESUME"
