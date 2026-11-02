import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.openevo_webshop import build_h1_36_manifest as h136


TRAIN_TASKS = [2822, 5135, 5294, 6769, 2901, 3449, 4313, 4600]
EVAL_TASKS = [594, 1082, 1568, 1614, 1719, 2087, 4709, 5790]


def _fake_goals(count: int = 7000) -> list[dict]:
    return [{"instruction_text": f"goal {index}"} for index in range(count)]


def _description_sha(goal_idx: int) -> str:
    return hashlib.sha256(f"goal {goal_idx}".encode()).hexdigest()


def _frozen_record() -> dict:
    rows = []
    for goal_idx in TRAIN_TASKS:
        rows.append({
            "goal_idx": goal_idx,
            "task_id": f"webshop_{goal_idx:06d}",
            "split": "train-only-self-evolution",
            "task_description_sha256": _description_sha(goal_idx),
        })
    for goal_idx in EVAL_TASKS:
        rows.append({
            "goal_idx": goal_idx,
            "task_id": f"webshop_{goal_idx:06d}",
            "split": "fresh-unused-self-evolution-evaluation",
            "task_description_sha256": _description_sha(goal_idx),
        })
    return {
        "schema": h136.H129_FROZEN_SCHEMA,
        "canonical_sha256": "c" * 64,
        "server_file_sha256": "s" * 64,
        "execution_git_sha": "290385835f1232fa6853971fe6a5f00ceca43f6c",
        "train_tasks": TRAIN_TASKS,
        "evaluation_tasks": EVAL_TASKS,
        "task_rows": rows,
    }


class H136StrongBaseManifestTests(unittest.TestCase):
    def _build(self, record: dict, goals: list[dict], errata: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            frozen_path = Path(directory) / "frozen.json"
            frozen_path.write_text(json.dumps(record), encoding="utf-8")
            file_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
            with patch.object(h136, "H129_FROZEN_FILE_SHA256", file_sha), patch.object(
                h136, "H129_CANONICAL_SHA256", record["canonical_sha256"]
            ), patch.object(h136, "H129_SERVER_FILE_SHA256", record["server_file_sha256"]), patch.object(
                h136, "H129_EXECUTION_GIT_SHA", record["execution_git_sha"]
            ), patch.object(
                h136, "H129_RECORD_ERRATA", {} if errata is None else errata
            ), patch.object(
                h136, "all_goals", return_value=goals
            ):
                return h136.build_manifest(Path("/seed"), frozen_path, experiment_sha="b" * 40)

    def test_reuses_frozen_h1_29_tasks_verbatim(self):
        manifest = self._build(_frozen_record(), _fake_goals())
        self.assertEqual(manifest["train_tasks"], TRAIN_TASKS)
        self.assertEqual(manifest["evaluation_tasks"], EVAL_TASKS)
        self.assertEqual(len(manifest["tasks"]), 16)
        self.assertEqual(
            {row["goal_idx"] for row in manifest["tasks"]},
            set(TRAIN_TASKS) | set(EVAL_TASKS),
        )
        for row in manifest["tasks"]:
            self.assertEqual(row["task_description"], f"goal {row['goal_idx']}")
            self.assertEqual(row["task_description_sha256"], _description_sha(row["goal_idx"]))

    def test_emits_h1_36_schema_and_7b_provenance(self):
        manifest = self._build(_frozen_record(), _fake_goals())
        self.assertEqual(manifest["schema"], "openevo.webshop-h1.36-strong-base-self-evolution-manifest.v1")
        self.assertEqual(manifest["source"]["model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(manifest["source"]["model_revision"], h136.MODEL_CONTENT_PIN)
        self.assertEqual(manifest["source"]["model_revision_kind"], "local-content-pin")
        self.assertEqual(
            manifest["source"]["base_model_scale_change"],
            "Qwen/Qwen2.5-3B-Instruct -> Qwen/Qwen2.5-7B-Instruct",
        )
        self.assertTrue(manifest["manifest_sha256"])

    def test_train_evaluation_disjoint(self):
        manifest = self._build(_frozen_record(), _fake_goals())
        self.assertFalse(set(manifest["train_tasks"]) & set(manifest["evaluation_tasks"]))
        self.assertFalse(manifest["contamination"]["teacher_oracle_gpt_used"])

    def test_description_identity_mismatch_rejected(self):
        record = _frozen_record()
        record["task_rows"][0]["task_description_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self._build(record, _fake_goals())

    def test_frozen_file_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen_path = Path(directory) / "frozen.json"
            frozen_path.write_text(json.dumps(_frozen_record()), encoding="utf-8")
            with patch.object(h136, "H129_FROZEN_FILE_SHA256", "9" * 64), patch.object(
                h136, "all_goals", return_value=_fake_goals()
            ):
                with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                    h136.build_manifest(Path("/seed"), frozen_path, experiment_sha="b" * 40)

    def test_server_file_anchor_mismatch_rejected(self):
        # record carries server_file_sha256 = "s"*64, but the pinned anchor expects "a"*64,
        # so the record no longer provably points at the real H1.29 server manifest.
        record = _frozen_record()
        with tempfile.TemporaryDirectory() as directory:
            frozen_path = Path(directory) / "frozen.json"
            frozen_path.write_text(json.dumps(record), encoding="utf-8")
            file_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
            with patch.object(h136, "H129_FROZEN_FILE_SHA256", file_sha), patch.object(
                h136, "H129_CANONICAL_SHA256", record["canonical_sha256"]
            ), patch.object(h136, "H129_EXECUTION_GIT_SHA", record["execution_git_sha"]), patch.object(
                h136, "H129_SERVER_FILE_SHA256", "a" * 64
            ), patch.object(h136, "all_goals", return_value=_fake_goals()):
                with self.assertRaisesRegex(ValueError, "does not anchor"):
                    h136.build_manifest(Path("/seed"), frozen_path, experiment_sha="b" * 40)

    def test_record_errata_corrects_defective_hash(self):
        # The frozen record carries a corrupt hash for goal 1082; the errata overrides the
        # expected value with the authoritative one, which the live seed root reproduces.
        record = _frozen_record()
        for row in record["task_rows"]:
            if row["goal_idx"] == 1082:
                row["task_description_sha256"] = "9" * 63  # corrupt, as in the real record
        manifest = self._build(record, _fake_goals(), errata={1082: _description_sha(1082)})
        row1082 = next(t for t in manifest["tasks"] if t["goal_idx"] == 1082)
        self.assertEqual(row1082["task_description_sha256"], _description_sha(1082))
        errata_field = manifest["source"]["reused_from_h1_29"]["record_errata"]
        self.assertEqual(errata_field["corrected_goal_indices"], [1082])

    def test_record_errata_stays_fail_closed(self):
        # An errata entry that does NOT match the live seed-root derivation must still fail.
        record = _frozen_record()
        for row in record["task_rows"]:
            if row["goal_idx"] == 1082:
                row["task_description_sha256"] = "9" * 63
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self._build(record, _fake_goals(), errata={1082: "0" * 64})


if __name__ == "__main__":
    unittest.main()
