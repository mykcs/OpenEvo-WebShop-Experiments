from scripts.openevo_webshop.label_student_recovery_states import candidate_states


def _trace(actions):
    return {"trace": [{"action": action, "action_valid": True, "used_fallback": False, "parser_failure": False, "prompt": f"state {i}"} for i, action in enumerate(actions)]}


def test_candidate_states_selects_repeated_navigation_and_backtrack():
    episode = _trace(["search[x]", "click[next >]", "click[next >]", "click[back to search]"])
    assert candidate_states(episode, max_per_episode=4) == [
        {"source_step": 2, "reason": "repeated-next+repeated-action"},
        {"source_step": 3, "reason": "backtrack"},
    ]


def test_candidate_states_is_bounded():
    episode = _trace(["click[next >]"] * 8)
    assert len(candidate_states(episode, max_per_episode=2)) == 2
