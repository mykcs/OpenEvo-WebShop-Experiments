import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.openevo_webshop import environment_doctor


class EnvironmentDoctorTests(unittest.TestCase):
    def test_missing_asset_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing"
            result = environment_doctor.check_paths({"model": path})
        self.assertEqual(result["status"], "fail")
        self.assertIn("model is missing", result["errors"][0])

    def test_runtime_asset_check_requires_data_files_and_model_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            data = root / "data"
            model = root / "model"
            (source / "seed").mkdir(parents=True)
            data.mkdir()
            model.mkdir()
            (source / "seed" / "analysis.py").write_text("# test\n", encoding="utf-8")
            for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"):
                (data / name).write_text("{}\n", encoding="utf-8")
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"test")
            result = environment_doctor.check_runtime_assets(source, data, model)
        self.assertEqual(result["status"], "pass")

    def test_runtime_asset_check_includes_index_receipt_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            data = root / "data"
            model = root / "model"
            index = root / "index"
            (source / "seed").mkdir(parents=True)
            data.mkdir()
            model.mkdir()
            (index / "indexes_1k").mkdir(parents=True)
            (index / "resources_1k").mkdir()
            (source / "seed" / "analysis.py").write_text("# test\n", encoding="utf-8")
            for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"):
                (data / name).write_text("{}\n", encoding="utf-8")
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"test")
            (index / "resources_1k" / "documents.jsonl").write_text("{}\n", encoding="utf-8")
            (index / "build_manifest.json").write_text("{}\n", encoding="utf-8")
            result = environment_doctor.check_runtime_assets(source, data, model, index)
        self.assertEqual(result["status"], "pass")

    def test_openevo_profile_does_not_require_seed_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            data = root / "data"
            model = root / "model"
            source.mkdir()
            data.mkdir()
            model.mkdir()
            for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"):
                (data / name).write_text("{}\n", encoding="utf-8")
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"test")
            result = environment_doctor.check_runtime_assets(source, data, model, require_seed_entry=False)
        self.assertEqual(result["status"], "pass")

    def test_model_completeness_requires_tokenizer_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"test")
            result = environment_doctor.check_model_completeness(model)
        self.assertEqual(result["status"], "fail")
        self.assertIn("complete tokenizer payload", result["errors"][0])

    def test_external_runtime_probe_is_explicit_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = environment_doctor.check_external_runtime_imports(Path(tmp), enabled=True)
        self.assertEqual(result["status"], "fail")
        self.assertIn("OpenEvo root", result["errors"][0])

    def test_openevo_src_layout_runtime_probe_imports_and_records_src_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiment"
            openevo_root = root / "openevo-core"
            package = openevo_root / "src/openevo/evolution/framework"
            source.mkdir()
            package.mkdir(parents=True)
            for path in (
                openevo_root / "src/openevo/__init__.py",
                openevo_root / "src/openevo/evolution/__init__.py",
                package / "__init__.py",
                package / "contracts.py",
            ):
                path.write_text("# test\n", encoding="utf-8")
            result = environment_doctor.check_external_runtime_imports(
                source,
                openevo_root=openevo_root,
                enabled=True,
                profile="openevo",
                python_executable=sys.executable,
            )
        self.assertEqual(result["status"], "pass")
        self.assertIn(str((openevo_root / "src").resolve()), result["paths"])
        self.assertLess(
            result["paths"].index(str((openevo_root / "src").resolve())),
            result["paths"].index(str(openevo_root.resolve())),
        )

    def test_openevo_flat_layout_runtime_probe_retains_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiment"
            openevo_root = root / "openevo-core"
            package = openevo_root / "openevo/evolution/framework"
            source.mkdir()
            package.mkdir(parents=True)
            for path in (
                openevo_root / "openevo/__init__.py",
                openevo_root / "openevo/evolution/__init__.py",
                package / "__init__.py",
                package / "contracts.py",
            ):
                path.write_text("# test\n", encoding="utf-8")
            result = environment_doctor.check_external_runtime_imports(
                source,
                openevo_root=openevo_root,
                enabled=True,
                profile="openevo",
                python_executable=sys.executable,
            )
        self.assertEqual(result["status"], "pass")
        self.assertIn(str(openevo_root.resolve()), result["paths"])

    def test_seed_runtime_probe_does_not_require_openevo_root(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            environment_doctor,
            "run",
            return_value={"ok": True, "stdout": "seed-pass", "stderr": ""},
        ):
            result = environment_doctor.check_external_runtime_imports(
                Path(tmp), enabled=True, profile="seed"
            )
        self.assertEqual(result["status"], "pass")
        self.assertNotIn("openevo", result["modules"])

    def test_seed_source_identity_does_not_require_openevo_root(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            environment_doctor,
            "check_git",
            return_value={"status": "pass", "errors": []},
        ):
            result = environment_doctor.check_external_source_identity(
                Path(tmp), enabled=True, profile="seed"
            )
        self.assertEqual(result["status"], "pass")
        self.assertIn("seed_source", result["checks"])
        self.assertNotIn("openevo_core", result["checks"])

    def test_runtime_python_identity_is_recorded(self):
        result = environment_doctor.check_python_executable(Path(environment_doctor.sys.executable))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["requested"], environment_doctor.sys.executable)
        self.assertTrue(result["executable"])

    def test_runtime_python_import_probe_uses_selected_interpreter(self):
        result = environment_doctor.check_imports(("json",), python_executable=environment_doctor.sys.executable)
        self.assertEqual(result["status"], "pass")
        self.assertIn("json", result["loaded"])

    def test_external_source_identity_requires_upstream_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = environment_doctor.check_external_source_identity(Path(tmp), enabled=True)
        self.assertEqual(result["status"], "fail")
        self.assertIn("upstream SHA", result["errors"][0])

    def test_openevo_source_identity_accepts_exact_approved_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            target = root / "src/openevo/evolution/parametric/sd_lora_trainer.py"
            target.parent.mkdir(parents=True)
            target.write_text("base\n", encoding="utf-8")
            (target.parent / "trainer_service.py").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
            target.write_text("patched\n", encoding="utf-8")
            diff = subprocess.check_output(
                ["git", "-C", str(root), "diff", "--", "src/openevo/evolution/parametric/sd_lora_trainer.py", "src/openevo/evolution/parametric/trainer_service.py"],
                text=True,
            )
            expected = hashlib.sha256(diff.encode("utf-8")).hexdigest()
            result = environment_doctor.check_external_source_identity(
                root,
                openevo_root=root,
                enabled=True,
                profile="openevo",
                expected_openevo_patch_sha=expected,
            )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["checks"]["openevo_core"]["patch_verified"])

    def test_experiment_config_contract_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"schema": "x", "status": "draft", "formal_task_consumption_allowed": False}), encoding="utf-8")
            result = environment_doctor.check_experiment_config(path)
        self.assertEqual(result["status"], "pass")

    def test_write_target_checks_parent_without_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "new-run"
            result = environment_doctor.check_write_target(target)
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["exists"])

    def test_storage_gate_fails_when_shared_volume_is_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            usage = environment_doctor.shutil.disk_usage(tmp)
            full = usage._replace(used=usage.total, free=0)
            with patch.object(environment_doctor.shutil, "disk_usage", return_value=full):
                result = environment_doctor.check_storage(Path(tmp) / "run", 1)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["free_bytes"], 0)
        self.assertIn("below required", result["errors"][0])

    def test_java_gate_fails_when_jdk_is_required_but_missing(self):
        with patch.object(environment_doctor.shutil, "which", return_value=None):
            result = environment_doctor.check_java(True)
        self.assertEqual(result["status"], "fail")
        self.assertIn("javac", result["errors"][1])

    def test_wandb_defaults_to_explicit_offline_fallback_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = environment_doctor.check_wandb("online", fallback=True)
        self.assertEqual(result["status"], "warn")
        self.assertEqual(result["effective"], "offline")
        self.assertIn("WANDB_API_KEY", result["fallback_reason"])

    def test_runtime_container_defers_image_identity_to_host_control_plane(self):
        result = environment_doctor.check_docker(
            "example/runtime:latest",
            "sha256:expected",
            runtime_container=True,
        )
        self.assertEqual(result["status"], "skip")
        self.assertIn("host control plane", result["reason"])

    def test_host_docker_check_still_fails_closed_without_cli(self):
        with patch.object(environment_doctor.shutil, "which", return_value=None):
            result = environment_doctor.check_docker("example/runtime:latest", "sha256:expected")
        self.assertEqual(result["status"], "fail")
        self.assertIn("docker CLI", result["errors"][0])

    def test_doctor_result_is_machine_readable_and_formal_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source", "data", "model"):
                (root / name).mkdir()
            (root / "source" / "seed").mkdir()
            (root / "source" / "seed" / "analysis.py").write_text("# test\n", encoding="utf-8")
            for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"):
                (root / "data" / name).write_text("{}\n", encoding="utf-8")
            (root / "model" / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "model" / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (root / "model" / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            (root / "model" / "model.safetensors").write_bytes(b"test")
            args = Namespace(
                repo_root=root,
                source_root=root / "source",
                webshop_data=root / "data",
                model=root / "model",
                image=None,
                expected_image_digest=None,
                runtime_python=None,
                profile="full",
                require_cuda=False,
                allow_dirty=True,
                wandb_mode="disabled",
                wandb_fallback=True,
                package=[],
                import_name=["json"],
                storage_root=None,
                artifact_root=None,
                minimum_free_bytes=1,
            )
            with patch.object(environment_doctor, "check_git", return_value={"status": "pass", "errors": []}), patch.object(environment_doctor, "check_packages", return_value={"status": "pass", "errors": []}), patch.object(environment_doctor, "check_pip", return_value={"status": "pass", "errors": []}), patch.object(environment_doctor, "check_cuda", return_value={"status": "pass", "errors": []}):
                result = environment_doctor.doctor(args)
        json.dumps(result)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["formal_task_consumption_allowed"])


if __name__ == "__main__":
    unittest.main()
