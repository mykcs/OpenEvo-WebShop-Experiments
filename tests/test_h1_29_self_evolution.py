import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.openevo_webshop.build_h1_29_manifest import build_manifest
from scripts.openevo_webshop.build_self_evolution_dataset import build_dataset
from scripts.openevo_webshop.reconcile_h1_29 import reconcile


def _attempt(task: int, seed: int, *, score: float = 0.2) -> dict:
    return {
        "arm": "base",
        "task_id": f"webshop_{task:06d}",
        "goal_idx": task,
        "rollout_seed": seed,
        "final_task_score": score,
        "identity_ok": True,
        "trace_complete": True,
        "parser_or_fallback_failure": False,
        "error": None,
        "qualified_positive": score > 0,
        "protocol_fingerprint": "a" * 64,
        "episode": {
            "trace": [{
                "step": 0,
                "prompt": "Allowed actions: search[item]",
                "raw_completion": "<think>search</think><action>search[item]</action>",
                "action_valid": True,
                "used_fallback": False,
                "parser_failure": False,
                "generation_seed": seed,
                "environment_seed": 2026,
            }],
        },
    }


class H129SelfEvolutionTests(unittest.TestCase):
    def test_manifest_contains_runner_task_rows(self):
        fake_goals = [{"instruction_text": f"goal {index}"} for index in range(7000)]
        with patch("scripts.openevo_webshop.build_h1_29_manifest.all_goals", return_value=fake_goals):
            manifest = build_manifest(Path("/seed"), [], experiment_sha="a" * 40, selection_seed=9)
        self.assertEqual(len(manifest["tasks"]), 16)
        self.assertEqual(
            {row["goal_idx"] for row in manifest["tasks"]},
            set(manifest["train_tasks"]) | set(manifest["evaluation_tasks"]),
        )

    def test_dataset_requires_two_success_task_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "cell"
            cell.mkdir()
            (cell / "attempts.jsonl").write_text(json.dumps(_attempt(1, 10)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "task identities"):
                build_dataset([cell], root / "dataset", train_task_ids={"webshop_000001"}, evaluation_task_ids={"webshop_000099"}, minimum_success_task_count=2, minimum_success_episode_count=1)

    def test_dataset_preserves_only_qualified_self_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = []
            for task in (1, 2):
                cell = root / f"cell-{task}"
                cell.mkdir()
                rows = [_attempt(task, 10), _attempt(task, 11, score=0.0)]
                rows[1]["qualified_positive"] = False
                (cell / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                cells.append(cell)
            result = build_dataset(cells, root / "dataset", train_task_ids={"webshop_000001", "webshop_000002"}, evaluation_task_ids={"webshop_000099"}, minimum_success_task_count=2, minimum_success_episode_count=2)
            self.assertEqual(result["record_count"], 2)
            self.assertEqual(result["qualified_success_task_ids"], ["webshop_000001", "webshop_000002"])
            manifest = json.loads((root / "dataset" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_route"], "self-evolution; no GPT or external teacher data")

    def test_reconcile_requires_positive_delta_on_two_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "evaluation_tasks": list(range(1, 9)),
                "evaluation": {"rollout_seeds": [10, 11, 12, 13], "arms": ["base", "self-evolution"]},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            specs = []
            for arm in ("base", "self-evolution"):
                for task in range(1, 9):
                    cell = root / f"{arm}-{task}"
                    cell.mkdir()
                    score = 0.2 if arm == "self-evolution" else 0.0
                    (cell / "attempts.jsonl").write_text("".join(json.dumps({**_attempt(task, seed, score=score), "arm": arm}) + "\n" for seed in (10, 11, 12, 13)), encoding="utf-8")
                    specs.append(f"{arm}:{task}:{cell}")
            result = reconcile(manifest_path, specs, root / "reconcile.json")
            self.assertEqual(result["classification"], "self-evolution-positive-candidate")
            self.assertAlmostEqual(result["paired_delta_sums"]["self_evolution_minus_base"], 6.4)


if __name__ == "__main__":
    unittest.main()
