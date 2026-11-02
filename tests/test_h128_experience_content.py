import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.openevo_webshop.audit_h128_experience_content import audit
from scripts.openevo_webshop.build_h128_oracle_action_dataset import build_dataset


def _record(event_id, task_id, action, *, oracle_action=None, reward=0.0, prompt=None):
    trace_metadata = {}
    if oracle_action is not None:
        trace_metadata["oracle_action"] = oracle_action
    return {
        "event_id": event_id,
        "task_id": task_id,
        "session_id": event_id,
        "status": "completed",
        "reward": reward,
        "traces": [{
            "prompt_messages": [{
                "role": "user",
                "content": prompt or "Allowed actions:\n['search[<your query>]']",
            }],
            "response_messages": [{"role": "assistant", "content": action}],
            "metadata": trace_metadata,
        }],
        "metadata": {},
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class H128ExperienceContentTest(unittest.TestCase):
    def test_audit_freezes_label_fidelity_factor(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            paths = {}
            for name in (
                "h1.11-positive",
                "h1.11-negative",
                "h1.13-positive-action-only",
                "h1.18-student-recovery",
            ):
                path = tmp_path / f"{name}.jsonl"
                _write(path, [_record(name, "webshop_000001", "<think>x</think><action>search[x]</action>", reward=0.5)])
                paths[name] = path
            teacher = tmp_path / "teacher.jsonl"
            _write(teacher, [
                _record("a", "webshop_000001", "search[x]", oracle_action="search[x]"),
                _record("b", "webshop_000001", "click[next >]", oracle_action="click[p1]", prompt="Allowed actions:\n['click[next >]',\n'click[p1]']"),
            ])
            paths["h1.18-teacher-action"] = teacher

            result = audit(paths)

            self.assertEqual(result["teacher_oracle_action_agreement"]["exact_rate"], 0.5)
            self.assertEqual(result["teacher_oracle_action_agreement"]["mismatch_action_kind_counts"], {"next->other_click": 1})
            self.assertTrue(result["frozen_candidate_factor"].startswith("next-action label fidelity"))
            self.assertEqual(result["datasets"]["h1.11-positive"]["think_trace_count"], 1)

    def test_oracle_action_dataset_changes_only_targets_and_provenance(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "records.jsonl"
            prompt = "Allowed actions:\n['search[<your query>]',\n'click[p1]']"
            rows = [
                _record("a", "webshop_000001", "search[too broad]", oracle_action="search[exact query]", prompt=prompt),
                _record("b", "webshop_000001", "click[p1]", oracle_action="click[p1]", prompt=prompt),
            ]
            _write(source, rows)

            result = build_dataset(source, tmp_path / "oracle", expected_count=2)
            output = [json.loads(line) for line in (tmp_path / "oracle" / "records.jsonl").read_text().splitlines()]

            self.assertEqual(result["changed_target_count"], 1)
            self.assertEqual(result["teacher_oracle_exact_count"], 1)
            self.assertEqual([row["event_id"] for row in output], ["a", "b"])
            self.assertEqual([row["traces"][0]["prompt_messages"] for row in output], [row["traces"][0]["prompt_messages"] for row in rows])
            self.assertEqual(output[0]["traces"][0]["response_messages"][0]["content"], "search[exact query]")
            self.assertEqual(output[1]["traces"][0]["response_messages"][0]["content"], "click[p1]")

    def test_oracle_action_dataset_rejects_action_missing_from_prompt(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "records.jsonl"
            _write(source, [_record("a", "webshop_000001", "click[p1]", oracle_action="click[p2]", prompt="Allowed actions:\n['click[p1]']")])

            with self.assertRaisesRegex(ValueError, "absent from the prompt"):
                build_dataset(source, tmp_path / "oracle", expected_count=1)


if __name__ == "__main__":
    unittest.main()
