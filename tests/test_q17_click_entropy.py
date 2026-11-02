from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import pytest

from scripts.openevo_webshop.q17_fixed_checkpoint_probe import (
    build_token_trie,
    summarize_click_log_probabilities,
    summarize_rows,
    top_two_logit_margin,
    validate_full_panel_manifest,
    FROZEN_PANEL_MANIFEST_REL,
    FROZEN_PANEL_MANIFEST_SHA256,
    build_probe_receipt,
)


def test_token_trie_reuses_shared_action_prefix() -> None:
    trie = build_token_trie({"click[red]": [1, 2, 8], "click[blue]": [1, 2, 9]})

    assert set(trie["children"]) == {1}
    shared = trie["children"][1]["children"][2]
    assert set(shared["children"]) == {8, 9}


def test_click_distribution_emits_entropy_hhi_top3_and_selected_probability() -> None:
    summary = summarize_click_log_probabilities(
        {
            "click[a]": math.log(0.24),
            "click[b]": math.log(0.12),
            "click[c]": math.log(0.04),
            "click[search]": math.log(0.2),
        },
        selected_action="click[b]",
    )

    assert summary["candidate_count"] == 3
    assert summary["status"] == "PASS"
    assert math.isclose(summary["entropy_nats"], -(0.6 * math.log(0.6) + 0.3 * math.log(0.3) + 0.1 * math.log(0.1)))
    assert math.isclose(summary["hhi"], 0.46)
    assert math.isclose(summary["top_k_mass"], 1.0)
    assert math.isclose(summary["selected_action_conditional_probability"], 0.3)
    assert math.isclose(summary["selected_action_log_probability"], math.log(0.12))
    assert [row["action"] for row in summary["candidates"]] == ["click[a]", "click[b]", "click[c]"]
    assert summary["click_search_sentinel"]["status"] == "ADMISSIBLE"
    assert math.isclose(
        summary["click_search_sentinel"]["conditional_probability_given_click_family"], 1 / 3
    )


def test_click_search_sentinel_only_is_reported_without_a_destination_candidate() -> None:
    summary = summarize_click_log_probabilities({"click[search]": math.log(0.2)})

    assert summary["status"] == "NO_CLICK_DESTINATIONS"
    assert summary["candidate_count"] == 0
    assert summary["click_search_sentinel"]["status"] == "ADMISSIBLE"
    assert math.isclose(
        summary["click_search_sentinel"]["conditional_probability_given_click_family"], 1.0
    )


def test_click_distribution_fails_closed_for_nonfinite_candidate_scores() -> None:
    with pytest.raises(ValueError, match="non-finite candidate log probabilities"):
        summarize_click_log_probabilities({"click[a]": -math.inf, "click[b]": math.nan})


def test_top_two_margin_handles_constraint_forced_token() -> None:
    assert top_two_logit_margin([1.0, 4.0, 2.0, -math.inf]) == 2.0
    assert top_two_logit_margin([-math.inf, 7.0]) is None


def test_probe_is_bound_to_the_existing_complete_64_task_manifest() -> None:
    repo = Path(__file__).resolve().parents[1]
    path = repo / FROZEN_PANEL_MANIFEST_REL
    manifest = json.loads(path.read_text())

    identity = validate_full_panel_manifest(path, manifest)

    assert identity["file_sha256"] == FROZEN_PANEL_MANIFEST_SHA256
    assert identity["panel_size"] == 64
    assert len(identity["task_identities"]) == 64


def test_probe_receipt_binds_script_and_panel_hashes(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text("static probe script\n")
    script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    checkpoint_manifest = tmp_path / "checkpoint-manifest.json"
    checkpoint_manifest.write_text('{"model":{"revision":"rev-test"}}\n')
    checkpoint_manifest_sha = hashlib.sha256(checkpoint_manifest.read_bytes()).hexdigest()
    receipt = build_probe_receipt(
        script,
        {
            "file_sha256": "manifest-sha",
            "task_identity_sha256": "task-identity-sha",
            "panel_sha256": "panel-sha",
        },
        checkpoint_id="checkpoint",
        checkpoint_manifest_sha256=checkpoint_manifest_sha,
        base_model="model/test",
        base_model_revision="rev-test",
        instrumentation="on",
        task_count=64,
    )

    assert receipt["probe_script_sha256"] == script_sha
    assert receipt["checkpoint_manifest_sha256"] == checkpoint_manifest_sha
    assert receipt["base_model"] == "model/test"
    assert receipt["base_model_revision"] == "rev-test"
    assert receipt["task_panel_manifest_sha256"] == "manifest-sha"
    assert receipt["panel_sha256"] == "panel-sha"
    assert receipt["task_count"] == 64


def test_probe_rejects_task_subset_and_incomplete_manifest(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / FROZEN_PANEL_MANIFEST_REL
    manifest = json.loads(source.read_text())
    with pytest.raises(RuntimeError, match="task-limit is forbidden"):
        validate_full_panel_manifest(source, manifest, task_limit=32)

    manifest["tasks"] = manifest["tasks"][:-1]
    path = tmp_path / "incomplete-panel.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="must contain all 64 task rows"):
        validate_full_panel_manifest(
            path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_summary_reports_candidate_count_distribution_and_search_sentinel() -> None:
    phase = {"generated_token_count": 1, "entropy_sum_nats": 0.2, "chosen_nll_sum_nats": 0.1}
    rows = [{
        "episode": {
            "score": 0.0,
            "steps": 2,
            "won": False,
            "positive_reward_rate": False,
            "invalid_terminated": False,
            "constraint_escape": False,
            "environment_action_schema_violation": False,
            "trace": [
                {"raw_completion_token_count": 1, "generation_diagnostics": {
                    "reasoning": phase,
                    "action": phase,
                    "selected_action_nll_nats": 0.1,
                    "click_choice": {
                        "candidate_count": 2,
                        "entropy_nats": 0.4,
                        "hhi": 0.6,
                        "top_k_mass": 1.0,
                        "selected_action_conditional_probability": 0.7,
                        "click_search_sentinel": {"status": "ADMISSIBLE", "selected": True,
                                                   "conditional_probability_given_click_family": 0.2},
                    },
                }},
                {"raw_completion_token_count": 1, "generation_diagnostics": {
                    "reasoning": phase,
                    "action": phase,
                    "selected_action_nll_nats": 0.1,
                    "click_choice": {
                        "candidate_count": 0,
                        "entropy_nats": None,
                        "hhi": None,
                        "top_k_mass": None,
                        "selected_action_conditional_probability": None,
                        "click_search_sentinel": {"status": "NOT_ADMISSIBLE", "selected": False,
                                                   "conditional_probability_given_click_family": None},
                    },
                }},
            ],
        }
    }]

    summary = summarize_rows(rows, instrumented=True)["click_choice"]

    assert summary["destination_candidate_count_distribution"] == {"0": 1, "2": 1}
    assert summary["destination_candidate_count_distribution_by_search_sentinel"] == {
        "sentinel_admissible": {"2": 1},
        "sentinel_not_admissible": {"0": 1},
    }
    assert summary["click_search_sentinel_admissible_step_count"] == 1
    assert summary["click_search_sentinel_selected_step_count"] == 1
