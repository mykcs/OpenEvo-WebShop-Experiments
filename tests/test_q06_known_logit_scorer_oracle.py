from __future__ import annotations

import copy
import math

import pytest

from scripts.analysis.q06_known_logit_scorer_oracle import (
    canonical_sha256,
    constrained_candidate_log_probabilities,
    score_known_logits,
    summarize_candidate_log_probabilities,
)


def test_complete_shared_prefix_paths_use_all_legal_tokens_in_each_denominator():
    # The fourth token in each legal mask has no candidate continuation. It
    # must still absorb probability mass at that constrained decoding step.
    targets = {
        "click[red]": [1, 2, 5],
        "click[blue]": [1, 2, 6],
        "click[search]": [1, 3],
    }
    logits = {
        (): [0.0] * 8,
        (1,): [0.0] * 9,
        (1, 2): [0.0] * 10,
    }
    legal = {
        (): [1, 7],
        (1,): [2, 3, 8],
        (1, 2): [5, 6, 9],
    }

    scored = score_known_logits(
        targets, logits, legal, selected_action="click[blue]"
    )

    probabilities = {
        action: math.exp(logp)
        for action, logp in scored["action_log_probabilities"].items()
    }
    assert probabilities == pytest.approx({
        "click[red]": 1 / 18,
        "click[blue]": 1 / 18,
        "click[search]": 1 / 6,
    })
    assert scored["action_log_probabilities"]["click[red]"] == pytest.approx(math.log(1 / 18))
    summary = scored["summary"]
    assert summary["candidate_count"] == 2
    assert summary["entropy_nats"] == pytest.approx(math.log(2))
    assert summary["hhi"] == pytest.approx(0.5)
    assert summary["selected_action_conditional_probability"] == pytest.approx(0.5)
    assert summary["selected_action_log_probability"] == pytest.approx(math.log(1 / 18))
    assert summary["click_search_sentinel"]["status"] == "ADMISSIBLE"
    # Sentinel family mass includes destinations; destination entropy excludes it.
    assert summary["click_search_sentinel"]["conditional_probability_given_click_family"] == pytest.approx(0.6)
    assert summary["click_search_sentinel"]["selected"] is False
    assert scored["model_calls"] == scored["environment_calls"] == 0


def test_one_destination_has_zero_entropy_even_when_decoder_mass_is_below_one():
    targets = {"click[only]": [2, 4]}
    logits = {(): [0.0] * 5, (2,): [0.0] * 5}
    legal = {(): [2, 3], (2,): [4, 1]}

    result = score_known_logits(targets, logits, legal, selected_action="click[only]")

    summary = result["summary"]
    assert math.exp(result["action_log_probabilities"]["click[only]"]) == pytest.approx(0.25)
    assert summary["candidate_count"] == 1
    assert summary["entropy_nats"] == pytest.approx(0.0)
    assert summary["hhi"] == pytest.approx(1.0)
    assert summary["top_k_mass"] == pytest.approx(1.0)
    assert summary["selected_action_conditional_probability"] == pytest.approx(1.0)
    assert summary["click_search_sentinel"]["status"] == "NOT_ADMISSIBLE"


def test_zero_destination_candidates_and_sentinel_only_are_distinguished():
    empty = summarize_candidate_log_probabilities({})
    assert empty["status"] == "NO_CLICK_DESTINATIONS"
    assert empty["candidate_count"] == 0
    assert empty["entropy_nats"] is None
    assert empty["click_search_sentinel"]["status"] == "NOT_ADMISSIBLE"

    sentinel_only_scores = constrained_candidate_log_probabilities(
        {"click[search]": [4, 5]},
        {(): [0.0] * 7, (4,): [0.0] * 7},
        {(): [4, 6], (4,): [5, 1]},
    )
    sentinel_only = summarize_candidate_log_probabilities(
        sentinel_only_scores, selected_action="click[search]"
    )
    assert sentinel_only["status"] == "NO_CLICK_DESTINATIONS"
    assert sentinel_only["candidate_count"] == 0
    assert sentinel_only["click_search_sentinel"]["status"] == "ADMISSIBLE"
    assert sentinel_only["click_search_sentinel"]["conditional_probability_given_click_family"] == pytest.approx(1.0)
    assert sentinel_only["click_search_sentinel"]["selected"] is True


@pytest.mark.parametrize(
    "targets,logits,legal,error",
    [
        ({"click[a]": [1]}, {(): [0.0, math.nan]}, {(): [1]}, "finite"),
        ({"click[a]": [1]}, {(): [0.0, math.inf]}, {(): [1]}, "finite"),
        ({"click[a]": [1]}, {(): [0.0, 0.0]}, {(): [0]}, "illegal"),
        ({"click[a]": [1]}, {(): [0.0, 0.0]}, {(): []}, "nonempty"),
        ({"click[a]": [1]}, {(): [0.0, 0.0]}, {(): [0, 0]}, "unique"),
    ],
)
def test_invalid_known_logits_or_masks_fail_closed(targets, logits, legal, error):
    with pytest.raises(ValueError, match=error):
        constrained_candidate_log_probabilities(targets, logits, legal)


def test_prefix_related_action_targets_are_rejected_as_non_disjoint_events():
    with pytest.raises(ValueError, match="prefix-free"):
        constrained_candidate_log_probabilities(
            {"click[short]": [2], "click[long]": [2, 3]},
            {(): [0.0] * 5, (2,): [0.0] * 5},
            {(): [2, 1], (2,): [3, 4]},
        )


def test_duplicate_action_token_paths_are_rejected_as_double_counted_events():
    with pytest.raises(ValueError, match="must be unique"):
        constrained_candidate_log_probabilities(
            {"click[a]": [2, 3], "click[b]": [2, 3]},
            {(): [0.0] * 5, (2,): [0.0] * 5},
            {(): [2, 1], (2,): [3, 4]},
        )


def test_scoring_preserves_prompt_action_and_environment_evidence_digest():
    evidence = {
        "prompt": {"messages": [{"role": "user", "content": "find red"}]},
        "action": {"finite_actions": ["click[red]", "click[search]"]},
        "environment": {"state_sha256": "a" * 64, "step": 7},
    }
    original = copy.deepcopy(evidence)
    expected_digest = canonical_sha256(original)
    result = score_known_logits(
        {"click[red]": [2, 3], "click[search]": [2, 4]},
        {(): [0.0] * 5, (2,): [0.0] * 5},
        {(): [2, 1], (2,): [3, 4]},
        evidence=evidence,
    )

    assert evidence == original
    assert result["evidence_sha256_before"] == expected_digest
    assert result["evidence_sha256_after"] == expected_digest
    assert result["evidence_unchanged"] is True
