from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "vendor/openevo-webshop/src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/openevo_webshop"))

from scripts.openevo_webshop.carrier_diagnostic_tracked import RecoveryMemory, _adapter_path


def test_adapter_path_normalizes_cli_sentinel_and_string_path(tmp_path):
    assert _adapter_path("none") is None
    assert _adapter_path(str(tmp_path / "adapter")) == tmp_path / "adapter"


def _write_records(path: Path) -> None:
    rows = [
        {
            "event_id": "webshop-recovery:a",
            "traces": [{
                "prompt_messages": [{"role": "user", "content": "Task: blue chair click[next >]"}],
                "response_messages": [{"role": "assistant", "content": "click[next >]"}],
            }],
        },
        {
            "event_id": "webshop-recovery:b",
            "traces": [{
                "prompt_messages": [{"role": "user", "content": "Task: red lamp backtrack next"}],
                "response_messages": [{"role": "assistant", "content": "click[back to search]"}],
            }],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_recovery_memory_retrieval_is_deterministic(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    _write_records(records)
    memory = RecoveryMemory(records, max_prompt_chars=100)

    text, receipt = memory.injection("Task: blue chair current state")

    assert "click[next >]" in text
    assert receipt["event_id"] == "webshop-recovery:a"
    assert receipt["memory_records_sha256"]


def test_recovery_memory_tie_breaks_by_event_id(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    _write_records(records)
    memory = RecoveryMemory(records, max_prompt_chars=100)

    _, receipt = memory.injection("Task")

    assert receipt["event_id"] == "webshop-recovery:a"


def test_recovery_memory_admissible_only_hides_inadmissible_action(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    _write_records(records)
    memory = RecoveryMemory(records, max_prompt_chars=100)

    text, receipt = memory.injection(
        "Task: blue chair\nAllowed actions:\n[\n'click[search]',\n]\n",
        action_policy="admissible-only",
    )

    assert "Recorded recovery action: click[next >]" not in text
    assert "(not admissible in the current state; use the state only)" in text
    assert receipt["action_admissible"] is False
    assert receipt["action_injected"] is False
    assert receipt["action_policy"] == "admissible-only"


def test_recovery_memory_admissible_only_keeps_admissible_action(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    _write_records(records)
    memory = RecoveryMemory(records, max_prompt_chars=100)

    text, receipt = memory.injection(
        "Task: blue chair\nAllowed actions:\n[\n'click[next >]',\n]\n",
        action_policy="admissible-only",
    )

    assert "click[next >]" in text
    assert receipt["action_admissible"] is True
    assert receipt["action_injected"] is True
