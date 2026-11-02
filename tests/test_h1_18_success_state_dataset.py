import json

from scripts.openevo_webshop.build_success_state_dataset import build_dataset


def test_success_state_dataset_is_provenance_bound(tmp_path):
    states = tmp_path / "states.jsonl"
    labels = tmp_path / "labels.jsonl"
    state = {
        "task_id": "webshop_003420",
        "rollout_seed": 2026082600,
        "source_step": 0,
        "source_state": "deterministic-oracle-success-path",
        "prompt": "Allowed actions:\n['search[x]']",
        "oracle_action": "search[x]",
    }
    label = {
        "task_id": "webshop_003420",
        "rollout_seed": 2026082600,
        "source_step": 0,
        "source_state": "deterministic-oracle-success-path",
        "teacher_identity": "Qwen/Qwen2.5-14B-Instruct-local",
        "teacher_action": "search[x]",
    }
    states.write_text(json.dumps(state) + "\n", encoding="utf-8")
    labels.write_text(json.dumps(label) + "\n", encoding="utf-8")

    result = build_dataset(
        states,
        labels,
        tmp_path / "dataset",
        train_task_ids={"webshop_003420"},
        evaluation_task_ids={"webshop_007440"},
        expected_count=1,
    )

    assert result["record_count"] == 1
    record = json.loads((tmp_path / "dataset" / "records.jsonl").read_text(encoding="utf-8"))
    assert record["metadata"]["state_source"] == "deterministic-oracle-success-path"
    assert record["traces"][0]["response_messages"][0]["content"] == "search[x]"
