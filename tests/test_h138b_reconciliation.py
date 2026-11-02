import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.openevo_webshop import reconcile_h138b_method_control as h138b


class H138BReconciliationTests(unittest.TestCase):
    def test_cli_only_treats_complete_reconciliation_as_success(self):
        source = Path(h138b.__file__).read_text(encoding="utf-8")
        self.assertIn('raise SystemExit(0 if result["status"] == "complete" else 2)', source)

    def _manifest(self, root: Path, training_tasks=(501,), fresh_tasks=(701,)) -> Path:
        payload = {
            "schema": h138b.MANIFEST_SCHEMA,
            "campaign_id": "test-h138b",
            "source": {
                "d8_records_sha256": h138b.D8_RECORDS_SHA256,
                "d8_selection_sha256": h138b.D8_SELECTION_SHA256,
            },
            "task_order": list(training_tasks),
            "training_tasks": [
                {"task_id": f"webshop_{task:06d}", "goal_idx": task}
                for task in training_tasks
            ],
            "fresh_tasks": [{"task_id": f"webshop_{task:06d}", "goal_idx": task} for task in fresh_tasks],
            "acquisition": {"arms": ["ordinary-lora", "sd-lora"], "rollout_seeds": [1]},
            "retention": {"arms": ["base", "ordinary-lora", "sd-lora"], "rollout_seeds": [2]},
            "fresh": {"arms": ["base", "ordinary-lora", "sd-lora"], "rollout_seeds": [3]},
        }
        payload["manifest_sha256"] = h138b._canonical_hash(payload)
        path = root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _row(
        self,
        arm: str,
        task: int,
        seed: int,
        score: float,
        *,
        loaded: bool | None = None,
        parser: bool = False,
        increment: int = -1,
        adapter_increment: int = 0,
    ):
        if loaded is None:
            loaded = arm != "base"
        return {
            "arm": arm,
            "evaluation_stage": "fresh",
            "goal_idx": task,
            "rollout_seed": seed,
            "increment_index": increment,
            "final_task_score": score,
            "identity_ok": True,
            "trace_complete": True,
            "parser_or_fallback_failure": parser,
            "error": None,
            "attempt_wall_time_seconds": 1.0,
            "adapter_load_receipt": {
                "verified": True,
                "adapter_loaded": loaded,
                "adapter_file_sha256": (self._adapter_digest(arm, adapter_increment) if loaded else None),
                "loaded_tensor_digest": (self._tensor_digest(arm, adapter_increment) if loaded else None),
            },
        }

    @staticmethod
    def _adapter_digest(arm: str, increment: int = 0) -> str:
        return hashlib.sha256(f"adapter:{arm}:{increment}".encode()).hexdigest()

    @staticmethod
    def _tensor_digest(arm: str, increment: int = 0) -> str:
        return hashlib.sha256(f"tensor:{arm}:{increment}".encode()).hexdigest()

    def _training(self, root: Path, tasks=(501,)) -> list[Path]:
        paths: list[Path] = []
        for arm in h138b.TREATMENTS:
          prior = None
          for increment, task in enumerate(tasks):
            reloaded = hashlib.sha256(f"reloaded:{arm}:{increment}".encode()).hexdigest()
            receipt = {
                "schema": h138b.TRAINING_RECEIPT_SCHEMA,
                "status": "pass",
                "arm": arm,
                "increment_index": increment,
                "task_id": f"webshop_{task:06d}",
                "goal_idx": task,
                "parent_records_sha256": h138b.D8_RECORDS_SHA256,
                "parent_selection_sha256": h138b.D8_SELECTION_SHA256,
                "sample_plan_sha256": "e" * 64,
                "prior_adapter_file_sha256": prior,
                "adapter_file_sha256": self._adapter_digest(arm, increment),
                "evaluation_tensor_digest": self._tensor_digest(arm, increment),
                "metrics": {
                    "optimizer_steps": 2,
                    "training_loss": 1.0,
                    "maximum_gradient_norm": 0.5,
                    "training_time_seconds": 180,
                },
                "state_proof": {
                    "before_trainable_tensor_sha256": hashlib.sha256(f"before:{arm}:{increment}".encode()).hexdigest(),
                    "after_trainable_tensor_sha256": reloaded,
                    "reloaded_trainable_tensor_sha256": reloaded,
                    "update_l2_norm": 0.25,
                    "changed_from_prior": True,
                },
                "no_base_fallback": {
                    "adapter_requested": True,
                    "adapter_loaded": True,
                    "loaded_lora_tensor_count": 2,
                    "base_trainable_parameter_count": 0,
                    "loaded_adapter_tensor_sha256": reloaded,
                },
            }
            path = root / f"{arm}-{increment}.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            paths.append(path)
            prior = receipt["adapter_file_sha256"]
        return paths

    def _cell(self, root: Path, stage: str, arm: str, task: int, row: dict):
        row["evaluation_stage"] = stage
        cell = root / "evaluation" / stage / arm / f"task{task}"
        cell.mkdir(parents=True)
        (cell / "attempts.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (cell / "summary.json").write_text(json.dumps({"wall_time_seconds": 36}), encoding="utf-8")

    def test_complete_primary_sd_minus_ordinary_and_gpu_hours(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            for arm, score in (("ordinary-lora", 0.2), ("sd-lora", 0.4)):
                self._cell(root, "acquisition", arm, 501, self._row(arm, 501, 1, score, increment=0))
            for arm, score in (("base", 0.1), ("ordinary-lora", 0.3), ("sd-lora", 0.5)):
                self._cell(root, "retention", arm, 501, self._row(arm, 501, 2, score))
                self._cell(root, "fresh", arm, 701, self._row(arm, 701, 3, score))
            result = h138b.reconcile(manifest, root, self._training(root), root / "reconciliation.json")
            self.assertEqual(result["status"], "complete")
            self.assertAlmostEqual(result["fresh_comparisons"]["primary_sd_minus_ordinary"]["task_weighted_delta"], 0.2)
            self.assertAlmostEqual(result["acquisition_comparisons"]["sd_minus_ordinary"]["task_weighted_delta"], 0.2)
            self.assertAlmostEqual(result["forgetting_comparisons"]["sd_minus_ordinary"]["task_weighted_delta"], 0.0)
            self.assertAlmostEqual(result["gpu_hours"]["training"], 0.1)
            self.assertEqual(result["valid_invalid_taxonomy"], {"valid": 8})
            self.assertAlmostEqual(
                result["acquisition"][0]["ordinary-lora"]["forgetting_acquisition_minus_retention"],
                -0.1,
            )

    def test_partial_is_preserved_without_becoming_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            self._cell(root, "fresh", "base", 701, self._row("base", 701, 3, 0.1))
            result = h138b.reconcile(manifest, root, self._training(root), root / "reconciliation.json")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["observed_attempts"], 1)
            self.assertEqual(result["missing_attempts"], 7)

    def test_missing_training_receipts_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            result = h138b.reconcile(manifest, root, [], root / "reconciliation.json")
            self.assertEqual(result["status"], "partial-unqualified")
            self.assertFalse(result["structural_errors"])
            self.assertEqual(result["partial_taxonomy"], {"missing-training-receipt": 2})

    def test_manifest_self_hash_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["fresh"]["rollout_seeds"] = [999]
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            result = h138b.reconcile(manifest, root, self._training(root), root / "reconciliation.json")
            self.assertEqual(result["status"], "invalid")
            self.assertIn("manifest self-hash mismatch", result["structural_errors"])

    def test_parser_and_silent_base_fallback_are_invalid_taxonomy(self):
        parser_row = self._row("sd-lora", 701, 3, 0.0, parser=True)
        fallback_row = self._row("ordinary-lora", 701, 3, 0.0, loaded=False)
        self.assertEqual(h138b.classify(parser_row), "parser-or-fallback")
        self.assertEqual(h138b.classify(fallback_row), "base-fallback")

    def test_acquisition_uses_corresponding_increment_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, training_tasks=(501, 502))
            for arm in h138b.TREATMENTS:
                self._cell(
                    root,
                    "acquisition",
                    arm,
                    501,
                    self._row(arm, 501, 1, 0.2, increment=0, adapter_increment=0),
                )
                self._cell(
                    root,
                    "acquisition",
                    arm,
                    502,
                    self._row(arm, 502, 1, 0.3, increment=1, adapter_increment=1),
                )
            result = h138b.reconcile(manifest, root, self._training(root, (501, 502)), root / "out.json")
            self.assertEqual(result["status"], "partial")
            self.assertFalse(any("adapter identity mismatch" in item for item in result["structural_errors"]))

    def test_wrong_increment_index_is_structurally_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            self._cell(
                root,
                "acquisition",
                "ordinary-lora",
                501,
                self._row("ordinary-lora", 501, 1, 0.2, increment=7),
            )
            result = h138b.reconcile(manifest, root, self._training(root), root / "out.json")
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("increment index mismatch" in item for item in result["structural_errors"]))

    def test_truncated_last_jsonl_line_is_partial_not_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            cell = root / "evaluation" / "fresh" / "base" / "task701"
            cell.mkdir(parents=True)
            (cell / "attempts.jsonl").write_text(
                json.dumps(self._row("base", 701, 3, 0.1)) + "\n{\"rollout_seed\":",
                encoding="utf-8",
            )
            (cell / "summary.json").write_text(json.dumps({"wall_time_seconds": 1}), encoding="utf-8")
            result = h138b.reconcile(manifest, root, self._training(root), root / "out.json")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["partial_taxonomy"], {"truncated-jsonl": 1})

    def test_extra_cell_inventory_is_structurally_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            self._cell(root, "fresh", "base", 999, self._row("base", 999, 3, 0.1))
            result = h138b.reconcile(manifest, root, self._training(root), root / "out.json")
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("unexpected evaluation artifact" in item for item in result["structural_errors"]))

    def test_non_finite_score_and_time_are_invalid_taxonomy(self):
        score = self._row("base", 701, 3, float("nan"))
        timing = self._row("base", 701, 3, 0.1)
        timing["attempt_wall_time_seconds"] = -1
        self.assertEqual(h138b.classify(score), "score-invalid")
        self.assertEqual(h138b.classify(timing), "time-invalid")

    def test_exact_prereg_manifest_without_qualification_is_partial_unqualified(self):
        repo = Path(__file__).resolve().parents[1]
        manifest = repo / "configs/experiment/manifests/h1.38b-method-control-eval-v1.json"
        prereg = repo / "configs/experiment/h1.38b-method-control-v1.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = h138b.reconcile(
                manifest,
                root,
                [],
                root / "out.json",
                prereg_path=prereg,
                qualification_path=root / "missing-qualification.json",
            )
            self.assertEqual(result["status"], "partial-unqualified")
            self.assertEqual(result["missing_attempts"], 704)
            self.assertEqual(result["training_evidence_count"], 0)

    def test_qualification_rejects_missing_real_key_replaced_by_forged_extra_key(self):
        training_rows = [
            {"arm": arm, "increment_index": increment, "receipt": f"{arm}-{increment}"}
            for arm in h138b.TREATMENTS
            for increment in range(8)
        ]
        qualified_rows = [dict(row) for row in training_rows]
        qualified_rows.pop()
        qualified_rows.append(
            {"arm": "forged-method", "increment_index": 99, "receipt": "forged-extra"}
        )
        qualification = {
            "schema": h138b.QUALIFICATION_SCHEMA,
            "status": "pass",
            "increment_count": 8,
            "formal_fresh_task_consumption_allowed": True,
            "final_capacity": {
                "ordinary_effective_rank": 32,
                "sd_effective_rank": 32,
                "serialized_parameter_count": 1024,
                "ordinary_inventory": {"parameter_count": 1024},
                "sd_inventory": {"parameter_count": 1024},
            },
            "training_receipts": qualified_rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            path.write_text(json.dumps(qualification), encoding="utf-8")
            passed, errors, _ = h138b._validate_qualification(path, None, training_rows)

        self.assertFalse(passed)
        self.assertIn("qualification training receipt key set mismatch", errors)

    def test_self_rehashed_manifest_still_fails_prereg_file_binding(self):
        repo = Path(__file__).resolve().parents[1]
        prereg = repo / "configs/experiment/h1.38b-method-control-v1.json"
        source = repo / "configs/experiment/manifests/h1.38b-method-control-eval-v1.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["fresh"]["rollout_seeds"] = [999]
            unsigned = dict(payload)
            unsigned.pop("manifest_sha256")
            payload["manifest_sha256"] = h138b._canonical_hash(unsigned)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            result = h138b.reconcile(
                manifest,
                root,
                [],
                root / "out.json",
                prereg_path=prereg,
                qualification_path=root / "missing-qualification.json",
            )
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("prereg manifest file SHA" in item for item in result["structural_errors"]))


if __name__ == "__main__":
    unittest.main()
