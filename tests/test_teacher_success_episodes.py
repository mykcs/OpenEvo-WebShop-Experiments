from scripts.openevo_webshop.generate_teacher_success_episodes import qualified_episode
from scripts.openevo_webshop.generate_teacher_success_episodes import _selected_tasks


def test_teacher_manifest_train_tasks_are_selectable(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"train_tasks": [{"goal_idx": 2198, "task_id": "webshop_002198"}]}')
    assert _selected_tasks(manifest, [2198])[0]["task_id"] == "webshop_002198"


def test_teacher_episode_requires_positive_fallback_free_trace():
    good = {"task_score": 1.0, "error": False, "trace": [{"action_valid": True, "used_fallback": False, "parser_failure": False}]}
    assert qualified_episode(good)
    assert not qualified_episode({**good, "task_score": 0.0})
    assert not qualified_episode({**good, "task_score": 0.5})
    assert not qualified_episode({"task_score": 1.0, "error": False, "trace": [{"action_valid": True, "used_fallback": True, "parser_failure": False}]})
