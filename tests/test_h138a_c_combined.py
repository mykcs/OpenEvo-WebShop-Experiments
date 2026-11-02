import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from scripts.openevo_webshop import analyze_h138a_c_combined as analysis


ARMS = ("base", "diversity-4", "diversity-8", "diversity-16")
SEEDS = (2026090500, 2026090501, 2026090502, 2026090503)
GPU_UUID = "GPU-00000000-0000-0000-0000-000000000000"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class CombinedFixture:
    def __init__(self, root: Path):
        self.root = root
        self.original = root / "original"
        self.continuation = root / "continuation"
        self.output = root / "combined.json"
        self.manifest_path = root / "manifest.json"
        self.config_path = root / "configs/experiment/config.json"
        self.config_path.parent.mkdir(parents=True)
        self.tasks = list(range(1000, 1048))
        self.original_d8 = self.tasks[:10]
        self.original_execution_sha = "a" * 40
        self.continuation_execution_sha = "b" * 40
        self.manifest = {
            "evaluation_tasks": self.tasks,
            "evaluation": {"arms": list(ARMS), "rollout_seeds": list(SEEDS)},
        }
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        evidence = root / "docs/evidence/h1.38a"
        evidence.mkdir(parents=True)
        reconciliation = evidence / "h1.38a-reconciliation.json"
        closeout = evidence / "h1.38a-closeout-summary.json"
        reconciliation.write_text('{"status":"invalid","observed_attempt_count":424}\n', encoding="utf-8")
        closeout.write_text('{"launcher":{"observed_cells":106,"observed_attempts":424}}\n', encoding="utf-8")
        probe = evidence / "wandb-online-probe.json"
        probe.write_text(json.dumps({"schema": "openevo.wandb-online-probe.v1", "requested_mode": "online", "exit_code": 127, "output": "python path missing"}) + "\n")
        self.config = {
            "campaign_id": "h138a-c-test",
            "original_campaign": {
                "campaign_id": analysis.H138A_CAMPAIGN_ID,
                "evaluation_manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                "reconciliation_evidence_path": str(reconciliation.relative_to(root)),
                "reconciliation_sha256": hashlib.sha256(reconciliation.read_bytes()).hexdigest(),
                "closeout_summary_evidence_path": str(closeout.relative_to(root)),
                "closeout_summary_sha256": hashlib.sha256(closeout.read_bytes()).hexdigest(),
                "wandb_online_probe_evidence_path": str(probe.relative_to(root)),
                "wandb_online_probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
                "completed_cells": {"base": 48, "diversity-4": 48, "diversity-8": 10, "diversity-16": 0},
                "completed_diversity_8_task_ids": self.original_d8,
                "scientific_execution_sha": self.original_execution_sha,
                "legacy_completed_cells": [dict(item) for item in analysis.H138A_LEGACY_CELLS],
            },
            "continuation_orchestration_sha": "c" * 40,
            "continuation_run_root": str(self.continuation),
            "frozen_source_split": {
                "original": {"base": 48, "diversity-4": 48, "diversity-8": 10, "diversity-16": 0},
                "continuation": {"base": 0, "diversity-4": 0, "diversity-8": 38, "diversity-16": 48},
            },
            "frozen_provenance": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "evaluation_model_path": "/workspace/models/qwen2.5-7b-instruct",
                "model_content_pin": "model-revision",
                "evaluation_environment_seed": 2026,
                "rollout_seeds": list(SEEDS),
                "temperature": 0,
                "max_steps": 30,
                "max_new_tokens": 512,
                "prompt_contract": "prompt-v2",
                "action_contract": "action-v3",
                "evaluator": "WebShop task_score/won",
                "parser_retry_seed_offset": 900000,
                "gpu_binding_contract": "single UUID via CUDA_VISIBLE_DEVICES; logical cuda:0",
                "latency_contract": "attempt_wall_time_seconds measured around one closed-loop episode; invalid attempts remain outside speed denominators",
                "upstream_open_evo_sha": "upstream-sha",
                "python_runtime_fingerprint": "runtime-sha",
                "image_digest": "sha256:image",
                "adapters": {
                    "diversity-4": {"adapter_uri": "/workspace/runs/d4/adapter"},
                    "diversity-8": {"adapter_uri": "/workspace/runs/d8/adapter"},
                    "diversity-16": {"adapter_uri": "/workspace/runs/d16/adapter"},
                },
            },
            "authorized_gpu_uuids": [GPU_UUID],
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        continuation_evidence = self.continuation / "evidence"
        continuation_evidence.mkdir(parents=True)
        (continuation_evidence / "provenance.json").write_text(json.dumps({
            "schema": "openevo.webshop-h1.38a-c-provenance.v1",
            "campaign_id": self.config["campaign_id"],
            "orchestration_sha": self.config["continuation_orchestration_sha"],
            "continuation_execution_sha": self.continuation_execution_sha,
            "original_scientific_sha": self.original_execution_sha,
            "original_manifest_sha256": self.config["original_campaign"]["evaluation_manifest_sha256"],
            "continuation_root": str(self.continuation),
            "authorized_gpu_uuids": self.config["authorized_gpu_uuids"],
            "selection_policy": "fixed missing D8/D16 cells only",
        }), encoding="utf-8")
        for arm in ARMS:
            for task in self.tasks:
                source = self.original if arm in {"base", "diversity-4"} or (arm == "diversity-8" and task in self.original_d8) else self.continuation
                self.write_cell(source, arm, task)

    def fingerprint(self, arm: str) -> str:
        frozen = self.config["frozen_provenance"]
        return digest(
            {
                "phase": "H1",
                "arm": arm,
                "model": frozen["evaluation_model_path"],
                "model_revision": frozen["model_content_pin"],
                "task_manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                "environment_seed": frozen["evaluation_environment_seed"],
                "generation_seeds": list(SEEDS),
                "temperature": float(frozen["temperature"]),
                "max_steps": frozen["max_steps"],
                "max_new_tokens": frozen["max_new_tokens"],
                "prompt_contract": frozen["prompt_contract"],
                "action_contract": frozen["action_contract"],
                "evaluator": frozen["evaluator"],
                "parser_retry_seed_offset": frozen["parser_retry_seed_offset"],
                "cuda_visible_devices": GPU_UUID,
                "gpu_binding_contract": frozen["gpu_binding_contract"],
                "latency_contract": frozen["latency_contract"],
            }
        )

    def write_cell(self, source: Path, arm: str, task: int, *, valid: bool = True):
        cell = source / "evaluation" / arm / f"task{task}"
        cell.mkdir(parents=True, exist_ok=True)
        fingerprint = self.fingerprint(arm)
        (cell / "gpu-before.json").write_text(json.dumps({"expected_uuid": GPU_UUID, "gpu_rows": [f"0, {GPU_UUID}, NVIDIA RTX, 2, 0"], "compute_apps": []}), encoding="utf-8")
        tracking = {"mode": "online", "run_id": f"{arm}-{task}", "fallback_receipt": None}
        (cell / "summary.json").write_text(
            json.dumps({"schema": "seed3090.webshop-h1-closed-loop.v1", "arm": arm, "attempt_count": 4, "valid_count": 4 if valid else 0, "qualified_positive_count": 4 if valid else 0, "mean_task_score": 1.0 if valid else 0.0, "protocol_fingerprint": fingerprint, "tracking": tracking}),
            encoding="utf-8",
        )
        rows = [
            {
                "arm": arm,
                "goal_idx": task,
                "rollout_seed": seed,
                "identity_ok": valid,
                "trace_complete": valid,
                "parser_or_fallback_failure": False,
                "error": None,
                "final_task_score": 1.0 if valid else 0.0,
                "qualified_positive": valid,
                "protocol_fingerprint": fingerprint,
            }
            for seed in SEEDS
        ]
        (cell / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        spool = cell / "wandb/run-test.wandb"
        spool.parent.mkdir(exist_ok=True)
        spool.write_bytes(f"{arm}:{task}".encode())

    def wandb_config(self, cell: Path):
        arm = cell.parent.name
        source = "original" if self.original in cell.parents else "continuation"
        frozen = self.config["frozen_provenance"]
        adapter = "none" if arm == "base" else frozen["adapters"][arm]["adapter_uri"]
        return {
            "__wandb_run_id": f"{arm}-{int(cell.name.removeprefix('task'))}",
            "phase": "H1",
            "arm": arm,
            "adapter": adapter,
            "campaign_id": self.config["original_campaign"]["campaign_id"] if source == "original" else self.config["campaign_id"],
            "experiment_git_sha": self.original_execution_sha if source == "original" else self.continuation_execution_sha,
            "upstream_openevo_sha": frozen["upstream_open_evo_sha"],
            "runtime_fingerprint": frozen["python_runtime_fingerprint"],
            "image_digest": frozen["image_digest"],
            "gpu_uuid": GPU_UUID,
            "model": frozen["evaluation_model_path"],
            "model_revision": frozen["model_content_pin"],
            "task_manifest_sha256": self.config["original_campaign"]["evaluation_manifest_sha256"],
            "environment_seed": frozen["evaluation_environment_seed"],
            "generation_seeds": list(SEEDS),
            "prompt_contract": frozen["prompt_contract"],
            "action_contract": frozen["action_contract"],
            "evaluator": frozen["evaluator"],
            "parser_retry_seed_offset": frozen["parser_retry_seed_offset"],
            "cuda_visible_devices": GPU_UUID,
            "gpu_binding_contract": frozen["gpu_binding_contract"],
            "latency_contract": frozen["latency_contract"],
            "temperature": float(frozen["temperature"]),
            "max_steps": frozen["max_steps"],
            "max_new_tokens": frozen["max_new_tokens"],
        }

    def run(self, mutate=None):
        def read(cell):
            config = self.wandb_config(cell)
            if mutate is not None:
                mutate(cell, config)
            return cell / "wandb/run-test.wandb", config
        with (
            mock.patch.object(analysis, "_read_wandb_run_config", side_effect=read),
            mock.patch.object(analysis, "_git_execution_is_allowed", return_value=True),
        ):
            return analysis.combined(self.manifest_path, self.config_path, self.original, self.continuation, self.output)

    def seal_receipts(self, *receipts: Path):
        manifest = self.root / "docs/evidence/h1.38a-c/continuation-fallback-receipts.sha256"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for receipt in sorted(receipts):
            relative = receipt.relative_to(self.continuation)
            lines.append(f"{hashlib.sha256(receipt.read_bytes()).hexdigest()}  {relative}\n")
        manifest.write_text("".join(lines), encoding="utf-8")
        self.config["tracking"] = {
            "sealed_fallback_receipt_manifest": {
                "path": str(manifest.relative_to(self.root)),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "count": len(receipts),
            }
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")


class H138ACCombinedTest(unittest.TestCase):
    def test_repo_evidence_path_rejects_non_evidence_repo_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "configs/experiment/config.json"
            config.parent.mkdir(parents=True)
            script = root / "scripts/example.py"
            script.parent.mkdir()
            script.write_text("pass\n")
            (root / "docs/evidence").mkdir(parents=True)
            self.assertIsNone(analysis._repo_evidence_path(config, "scripts/example.py"))

    @classmethod
    def setUpClass(cls):
        cls.original_resamples = analysis.RESAMPLES
        analysis.RESAMPLES = 40

    @classmethod
    def tearDownClass(cls):
        analysis.RESAMPLES = cls.original_resamples

    def test_accepts_exact_frozen_192_cell_panel(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            result = fixture.run()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["observed_cell_count"], 192)
            self.assertEqual(result["observed_attempt_count"], 768)
            self.assertEqual(result["source_cells"], {
                "base": {"original": 48, "continuation": 0},
                "diversity-4": {"original": 48, "continuation": 0},
                "diversity-8": {"original": 10, "continuation": 38},
                "diversity-16": {"original": 0, "continuation": 48},
            })

    def test_rejects_protocol_fingerprint_drift(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            attempts = fixture.original / "evaluation/base/task1000/attempts.jsonl"
            rows = [json.loads(line) for line in attempts.read_text().splitlines()]
            rows[0]["protocol_fingerprint"] = "0" * 64
            attempts.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("protocol fingerprint mismatch" in error for error in result["structural_errors"]))

    def test_rejects_wrong_wandb_adapter(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            def mutate(cell, config):
                if cell.parent.name == "diversity-8" and cell.name == "task1010":
                    config["adapter"] = "/workspace/runs/wrong/adapter"
            result = fixture.run(mutate)
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("W&B config mismatch" in error and "adapter" in error for error in result["structural_errors"]))

    def test_rejects_wrong_original_wandb_adapter(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            def mutate(cell, config):
                if fixture.original in cell.parents and cell.parent.name == "diversity-4" and cell.name == "task1000":
                    config["adapter"] = "/workspace/runs/wrong/adapter"
            result = fixture.run(mutate)
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("W&B config mismatch" in error and "adapter" in error for error in result["structural_errors"]))

    def test_rejects_production_legacy_allowlist_drift(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            fixture.config["original_campaign"]["campaign_id"] = analysis.H138A_CAMPAIGN_ID
            fixture.config["original_campaign"]["legacy_completed_cells"] = [{"arm": "base", "task": 1000}]
            fixture.config_path.write_text(json.dumps(fixture.config))
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertIn("original legacy completed-cell contract mismatch", result["structural_errors"])

    def test_rejects_legacy_exception_for_nonproduction_campaign(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            fixture.config["original_campaign"]["campaign_id"] = "h138a-test"
            fixture.config["original_campaign"]["legacy_completed_cells"] = [{
                "arm": "base", "task": 1000, "execution_sha": "d" * 40,
                "manifest_sha256": fixture.config["original_campaign"]["evaluation_manifest_sha256"],
                "protocol_fingerprint": fixture.fingerprint("base"),
            }]
            fixture.config_path.write_text(json.dumps(fixture.config))
            result = fixture.run(lambda cell, config: config.update(
                {"experiment_git_sha": "d" * 40}
            ) if fixture.original in cell.parents and cell.parent.name == "base" and cell.name == "task1000" else None)
            self.assertEqual(result["status"], "invalid")
            self.assertIn("original campaign identity mismatch", result["structural_errors"])

    def test_legacy_contract_null_fails_closed(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            fixture.config["original_campaign"]["legacy_completed_cells"] = None
            fixture.config_path.write_text(json.dumps(fixture.config))
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertIn("original legacy completed-cell contract mismatch", result["structural_errors"])

    def test_production_campaign_requires_sealed_receipt_manifest(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            fixture.config["campaign_id"] = analysis.H138AC_CAMPAIGN_ID
            fixture.config_path.write_text(json.dumps(fixture.config))
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertIn("sealed fallback receipt manifest missing", result["structural_errors"])

    def test_tracking_null_fails_closed(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            fixture.config["tracking"] = None
            fixture.config_path.write_text(json.dumps(fixture.config))
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertIn("tracking contract invalid", result["structural_errors"])

    def test_rejects_unrecorded_continuation_execution(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            def mutate(cell, config):
                if fixture.continuation in cell.parents:
                    config["experiment_git_sha"] = "d" * 40
            result = fixture.run(mutate)
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("W&B execution provenance mismatch" in error for error in result["structural_errors"]))

    def test_rejects_provenance_with_wrong_continuation_root(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            provenance = fixture.continuation / "evidence/provenance.json"
            payload = json.loads(provenance.read_text())
            payload["continuation_root"] = "/wrong/root"
            provenance.write_text(json.dumps(payload))
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("continuation execution provenance mismatch" in error for error in result["structural_errors"]))

    def test_rejects_provenance_outside_git_execution_chain(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            def read(cell):
                return cell / "wandb/run-test.wandb", fixture.wandb_config(cell)
            with (
                mock.patch.object(analysis, "_read_wandb_run_config", side_effect=read),
                mock.patch.object(analysis, "_git_execution_is_allowed", return_value=False),
            ):
                result = analysis.combined(
                    fixture.manifest_path, fixture.config_path, fixture.original, fixture.continuation, fixture.output
                )
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("continuation execution provenance mismatch" in error for error in result["structural_errors"]))

    def test_rejects_offline_cell_without_receipt(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            cell = fixture.continuation / "evaluation/diversity-8/task1010"
            summary = json.loads((cell / "summary.json").read_text())
            summary["tracking"]["mode"] = "offline"
            (cell / "summary.json").write_text(json.dumps(summary))
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("offline W&B" in error for error in result["structural_errors"]))

    def test_accepts_offline_cell_with_matching_failed_global_probe(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            cell = fixture.continuation / "evaluation/diversity-8/task1010"
            summary = json.loads((cell / "summary.json").read_text())
            summary["tracking"]["mode"] = "offline"
            (cell / "summary.json").write_text(json.dumps(summary))
            probe = fixture.continuation / "evidence" / f"wandb-online-probe-{fixture.continuation_execution_sha}.json"
            probe.write_text(json.dumps({
                "schema": "openevo.wandb-online-probe.v1",
                "execution_sha": fixture.continuation_execution_sha,
                "requested_mode": "online",
                "effective_mode": "offline",
                "status": "failed",
                "exit_code": 1,
                "output": "WANDB_API_KEY missing",
            }))
            result = fixture.run()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["continuation_tracking"]["mode_counts"]["offline"], 1)
            self.assertEqual(len(result["continuation_tracking"]["fallback_probes"]), 1)

    def test_accepts_original_offline_cell_with_pinned_probe(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            cell = fixture.original / "evaluation/base/task1000"
            summary = json.loads((cell / "summary.json").read_text())
            summary["tracking"]["mode"] = "offline"
            (cell / "summary.json").write_text(json.dumps(summary))
            result = fixture.run()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["tracking_warnings"], [])

    def test_accepts_post_finalization_receipt_hash_drift_as_warning(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            cell = fixture.continuation / "evaluation/diversity-8/task1010"
            summary = json.loads((cell / "summary.json").read_text())
            summary["tracking"]["mode"] = "offline"
            (cell / "summary.json").write_text(json.dumps(summary))
            (cell / "wandb/wandb-fallback-receipt.json").write_text(json.dumps({
                "schema": "openevo.wandb-fallback-receipt.v1",
                "requested_mode": "online",
                "effective_mode": "offline",
                "project": "openevo-webshop-experiments",
                "online_error": "wandb.init failed: API key missing",
                "started_at_utc": "2026-08-19T00:00:00Z",
                "finished_at_utc": "2026-08-19T00:00:01Z",
                "spool_sha256": "sha256:" + "0" * 64,
            }))
            fixture.seal_receipts(cell / "wandb/wandb-fallback-receipt.json")
            result = fixture.run()
            self.assertEqual(result["status"], "complete")
            self.assertTrue(any("spool receipt hash drift" in warning for warning in result["tracking_warnings"]))

    def test_rejects_unsealed_post_finalization_receipt_hash_drift(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            cell = fixture.continuation / "evaluation/diversity-8/task1010"
            summary = json.loads((cell / "summary.json").read_text())
            summary["tracking"]["mode"] = "offline"
            (cell / "summary.json").write_text(json.dumps(summary))
            (cell / "wandb/wandb-fallback-receipt.json").write_text(json.dumps({
                "schema": "openevo.wandb-fallback-receipt.v1",
                "requested_mode": "online",
                "effective_mode": "offline",
                "project": "openevo-webshop-experiments",
                "online_error": "wandb.init failed: fabricated",
                "started_at_utc": "2026-08-19T00:00:00Z",
                "finished_at_utc": "2026-08-19T00:00:01Z",
                "spool_sha256": "sha256:" + "0" * 64,
            }))
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("spool receipt mismatch" in error for error in result["structural_errors"]))

    def test_rejects_offline_spool_receipt_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            cell = fixture.continuation / "evaluation/diversity-8/task1010"
            summary = json.loads((cell / "summary.json").read_text())
            summary["tracking"]["mode"] = "offline"
            (cell / "summary.json").write_text(json.dumps(summary))
            (cell / "wandb/wandb-fallback-receipt.json").write_text(json.dumps({
                "schema": "openevo.wandb-fallback-receipt.v1",
                "requested_mode": "online",
                "effective_mode": "offline",
                "project": "openevo-webshop-experiments",
                "online_error": "network unavailable",
                "started_at_utc": "2026-08-19T00:00:00Z",
                "finished_at_utc": "2026-08-19T00:00:00Z",
                "spool_sha256": "sha256:" + "0" * 64,
            }))
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("spool receipt mismatch" in error for error in result["structural_errors"]))

    def test_accepts_sealed_legacy_directory_spool_receipt(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            cell = fixture.continuation / "evaluation/diversity-8/task1010"
            summary = json.loads((cell / "summary.json").read_text())
            summary["tracking"]["mode"] = "offline"
            (cell / "summary.json").write_text(json.dumps(summary))
            receipt = cell / "wandb/wandb-fallback-receipt.json"
            receipt.write_text(json.dumps({
                "schema": "openevo.wandb-fallback-receipt.v1",
                "requested_mode": "online",
                "effective_mode": "offline",
                "project": "openevo-webshop-experiments",
                "online_error": "network unavailable",
                "started_at_utc": "2026-08-19T00:00:00Z",
                "finished_at_utc": "2026-08-19T00:00:00Z",
                "spool_sha256": analysis._legacy_wandb_spool_sha256(receipt),
            }))
            result = fixture.run()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["continuation_tracking"]["offline_spools"][0]["path"], "evaluation/diversity-8/task1010/wandb/run-test.wandb")

    def test_rejects_extra_arm_and_task_directories(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            (fixture.original / "evaluation/bonus/task1000").mkdir(parents=True)
            (fixture.continuation / "evaluation/diversity-8/task9999").mkdir(parents=True)
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("unexpected evaluation directory" in error for error in result["structural_errors"]))

    def test_rejects_wrong_source_split(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            task = fixture.tasks[10]
            fixture.write_cell(fixture.original, "diversity-8", task)
            continuation = fixture.continuation / "evaluation/diversity-8" / f"task{task}"
            shutil.rmtree(continuation)
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("source split mismatch" in error for error in result["structural_errors"]))

    def test_zero_valid_arm_is_structural_error_not_type_error(self):
        with TemporaryDirectory() as directory:
            fixture = CombinedFixture(Path(directory))
            for task in fixture.tasks:
                fixture.write_cell(fixture.continuation, "diversity-16", task, valid=False)
            result = fixture.run()
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("zero scientific-valid attempts" in error for error in result["structural_errors"]))


if __name__ == "__main__":
    unittest.main()
