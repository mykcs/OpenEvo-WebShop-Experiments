from scripts.openevo_webshop.build_h1_18_manifest import EXCLUDED_GOAL_INDICES, select_indices


def test_h1_18_selection_is_frozen_and_fresh():
    selected = select_indices(6910)
    assert selected == [744, 1263, 1623, 3020, 3848, 4415, 5593, 6091]
    assert not EXCLUDED_GOAL_INDICES.intersection(selected)
