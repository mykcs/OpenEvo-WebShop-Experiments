from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.openevo_webshop.reconcile_h127_model_capability import reconcile


def _row(task: int, seed: int, score: float, *, parser_invalid: bool = False) -> dict:
    return {
        "arm": "base-30",
        "task_id": f"webshop_{task:06d}",
        "goal_idx": task,
        "rollout_seed": seed,
        "identity_ok": True,
        "trace_complete": True,
        "parser_or_fallback_failure": parser_invalid,
        "error": None,
        "qualified_positive": score > 0 and not parser_invalid,
        "final_task_score": score,
        "won": score == 1.0,
        "episode": {"trace": [{"action": "click[Buy Now]"}] if score else []},
    }


def _fixture(tmp_path: Path, *, invalidate_one: bool = False) -> tuple[Path, Path]:
    arms = [
        {"model_arm": "QWEN25-3B", "model_path": "/models/q3"},
        {"model_arm": "QWEN25-7B", "model_path": "/models/q7"},
    ]
    manifest = {
        "manifest_sha256": "internal",
        "model_arms": arms,
        "tasks": [{"goal_idx": task, "task_id": f"webshop_{task:06d}"} for task in (505, 506)],
        "controls": {"rollout_seeds": [11, 12]},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    for arm in arms:
        for task in (505, 506):
            output = tmp_path / "formal" / "cells" / f"{arm['model_arm']}-{task}" / "output"
            output.mkdir(parents=True)
            score = 1.0 if arm["model_arm"] == "QWEN25-7B" else 0.0
            rows = [
                _row(task, seed, score, parser_invalid=invalidate_one and arm["model_arm"] == "QWEN25-7B" and task == 505 and seed == 11)
                for seed in (11, 12)
            ]
            (output / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (output / "summary.json").write_text(json.dumps({"goal_idx": task, "model": arm["model_path"], "wandb_run_id": "run"}))
    return tmp_path, manifest_path


class H127ModelCapabilityReconciliationTest(unittest.TestCase):
    def test_reconciliation_detects_cross_task_capability_signal(self) -> None:
        with TemporaryDirectory() as directory:
            run_root, manifest = _fixture(Path(directory))
            result = reconcile(run_root, manifest)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["observed_attempt_count"], 8)
            self.assertEqual(result["scientific_valid_attempt_count"], 8)
            self.assertEqual(result["gpu_duration_coverage_cells"], 0)
            self.assertEqual(result["classification"], "cross-task-capability-signal-needs-independent-confirmation")
            self.assertEqual(result["paired_against_qwen25_3b"]["QWEN25-7B"]["positive_task_ids"], [505, 506])

    def test_reconciliation_excludes_parser_invalid_pair(self) -> None:
        with TemporaryDirectory() as directory:
            run_root, manifest = _fixture(Path(directory), invalidate_one=True)
            result = reconcile(run_root, manifest)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["scientific_valid_attempt_count"], 7)
            self.assertEqual(result["arms"]["QWEN25-7B"]["invalid_count"], 1)
            self.assertEqual(result["classification"], "scale-only-not-supported-or-task-local")

    def test_reconciliation_sums_per_cell_gpu_wall_time(self) -> None:
        with TemporaryDirectory() as directory:
            run_root, manifest = _fixture(Path(directory))
            output = run_root / "formal" / "cells" / "QWEN25-3B-505" / "output"
            (output / "gpu-before.json").write_text(json.dumps({"timestamp_utc": "2026-08-16T13:00:00Z"}))
            (output / "gpu-after.json").write_text(json.dumps({"timestamp_utc": "2026-08-16T13:01:00Z"}))

            result = reconcile(run_root, manifest)

            self.assertEqual(result["gpu_duration_coverage_cells"], 1)
            self.assertEqual(result["measured_gpu_hours"], 1 / 60)


if __name__ == "__main__":
    unittest.main()
