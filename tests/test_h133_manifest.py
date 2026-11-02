import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_h133_is_fail_closed_and_teacher_free():
    document = json.loads((ROOT / "configs/experiment/h1.33-positive-replay-transfer.json").read_text())
    assert document["schema"] == "openevo.experiment-h1.33-positive-replay-transfer.v1"
    assert document["formal_task_consumption_allowed"] is False
    assert document["scientific_result"] is False
    assert document["source_condition"]["teacher"] == "disabled"
    assert document["evaluation"]["total_attempts"] == 64
    assert "H1.30 formal SEED tasks" in " ".join(document["contamination_rules"])


def test_h133_launcher_references_h0_source_and_reconciler():
    launcher = (ROOT / "scripts/openevo_webshop/launch_h133_positive_replay.sh").read_text()
    assert "h0.1-reconciliation.json" in launcher
    assert "build_h1_33_manifest" in launcher
    assert "reconcile_h133_positive_replay" in launcher
    assert "formal_task_consumption_allowed == false" in launcher


def test_h133_launcher_resolves_h0_cells_across_waves():
    launcher = (ROOT / "scripts/openevo_webshop/launch_h133_positive_replay.sh").read_text()
    assert 'find "${H0_HOST_ROOT}/harvest"' in launcher
    assert 'path "*/task${task}/attempts.jsonl"' in launcher
    assert 'H0_ROOT}" "${relative}' in launcher
    assert 'harvest/wave0/task${task}' not in launcher
    assert '--python /workspace/repo/scripts/openevo_webshop/server_trainer_python.sh' in launcher
    assert 'OPENEVO_H133_RESUME_AFTER_BASE' in launcher
    assert 'GPU utilization did not settle to zero within 30 seconds' in launcher


def test_trainer_wrapper_preserves_parent_death_signal_boundary():
    wrapper = (ROOT / "scripts/openevo_webshop/server_trainer_python.sh").read_text()
    assert 'OPENEVO_SD_LORA_PARENT_PID="${PPID}"' in wrapper
    assert 'OPENEVO_SD_LORA_PARENT_PID:-' in wrapper
