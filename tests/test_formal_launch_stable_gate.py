from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "openevo_webshop" / "formal_launch_stable_gate.py"
SPEC = importlib.util.spec_from_file_location("formal_launch_stable_gate", MODULE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def init_repo(path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    return path


def commit_all(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "test"], check=True)


def write_runtime(repo: Path, *, include_dependency: bool = True, include_env_bridge: bool = True) -> None:
    root = repo / "scripts" / "openevo_webshop"
    root.mkdir(parents=True, exist_ok=True)
    worker = "from scripts.openevo_webshop.helper import VALUE\n" if include_dependency else "from scripts.openevo_webshop.missing import VALUE\n"
    (root / "worker.py").write_text(worker, encoding="utf-8")
    if include_dependency:
        (root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    tokens = " ".join(gate.REQUIRED_GPU_ENV_TOKENS) if include_env_bridge else "CEILING1_STAGE2_GPU_UUIDS"
    (root / "safety.py").write_text(f"TOKENS = {tokens!r}\n", encoding="utf-8")


class StableLaunchGateTests(unittest.TestCase):
    def test_default_cli_safety_exists_in_current_execution_tree(self) -> None:
        gate.launcher_env_bridge(MODULE_PATH.parents[2], gate.DEFAULT_SAFETY)

    def test_webshop_boundary_rejects_unversioned_core(self) -> None:
        core = self.repo / "unversioned"
        core.mkdir()
        with self.assertRaisesRegex(RuntimeError, "core source identity unavailable"):
            gate.webshop_boundary_identity(self.repo, core, self.repo / "seed")

    def _webshop_fixture(self, root: Path) -> tuple[Path, Path]:
        core = init_repo(root / "core")
        (core / "source.py").write_text("VALUE = 1\n")
        commit_all(core)
        seed = root / "seed"
        webshop = seed / "agent_system/environments/env_package/webshop/webshop"
        (webshop / "data").mkdir(parents=True)
        for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"):
            (webshop / "data" / name).write_text("[]\n")
        index = webshop / "search_engine/indexes_1k"
        index.mkdir(parents=True)
        (index / "segment").write_text("index-v1")
        vendor = self.repo / "vendor/openevo-webshop/src/openevo_webshop"
        vendor.mkdir(parents=True)
        (vendor / "env_compat.py").write_text("# frozen wrapper\n")
        write_runtime(self.repo)
        commit_all(self.repo)
        return core, seed

    def test_webshop_boundary_change_during_construction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core, seed = self._webshop_fixture(Path(tmp))
            def mutate(*args, **kwargs):
                (core / "source.py").write_text("VALUE = 2\n")
                return {"status": "PASS"}
            with patch.object(gate, "image_id", return_value="sha256:frozen"), patch.object(gate, "production_import_smoke", side_effect=mutate):
                with self.assertRaisesRegex(RuntimeError, "boundary changed during"):
                    gate.build_receipt(self.repo, core, "science:image", "scripts.openevo_webshop.worker",
                                       "scripts/openevo_webshop/safety.py", seed_root=seed)

    def test_receipt_reuse_rejects_core_data_and_index_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core, seed = self._webshop_fixture(Path(tmp))
            with patch.object(gate, "image_id", return_value="sha256:frozen"), patch.object(gate, "production_import_smoke", return_value={"status": "PASS"}):
                doc = gate.build_receipt(self.repo, core, "science:image", "scripts.openevo_webshop.worker",
                                          "scripts/openevo_webshop/safety.py", seed_root=seed)
                receipt = Path(tmp) / "receipt.json"
                receipt.write_text(json.dumps(doc))
                gate.validate_receipt(self.repo, receipt)
                webshop = seed / "agent_system/environments/env_package/webshop/webshop"
                for path in (core / "source.py", webshop / "data/items_shuffle_1000.json", webshop / "search_engine/indexes_1k/segment"):
                    original = path.read_bytes()
                    path.write_bytes(original + b"drift")
                    with self.assertRaisesRegex(RuntimeError, "WebShop boundary drift"):
                        gate.validate_receipt(self.repo, receipt)
                    path.write_bytes(original)

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = init_repo(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_real_webshop_construction_uses_frozen_image_without_gpu_or_install(self) -> None:
        result = subprocess.CompletedProcess([], 0, "FORMAL_IMPORT_SMOKE_PASS\nWEBSHOP_CONSTRUCTION_PASS\n", "")
        with patch.object(gate.subprocess, "run", return_value=result) as run:
            receipt = gate.production_import_smoke(
                self.repo, self.repo, "sha256:frozen", "scripts.openevo_webshop.worker",
                seed_root=self.repo / "seed",
            )
        command = run.call_args.args[0]
        self.assertIn("--read-only", command)
        self.assertNotIn("--gpus", command)
        self.assertEqual(receipt["real_boundary"], "WebshopEnv._build")
        self.assertFalse(receipt["formal_task_consumption_allowed"])

    def test_dependency_closure_fails_on_missing_local_runtime(self) -> None:
        write_runtime(self.repo, include_dependency=False)
        with self.assertRaisesRegex(RuntimeError, "dependency closure missing"):
            gate.dependency_closure(self.repo, "scripts.openevo_webshop.worker")

    def test_launcher_env_bridge_requires_all_gpu_env_contracts(self) -> None:
        write_runtime(self.repo, include_env_bridge=False)
        with self.assertRaisesRegex(RuntimeError, "GPU env bridge missing"):
            gate.launcher_env_bridge(self.repo, "scripts/openevo_webshop/safety.py")

    def test_build_receipt_freezes_exact_execution_and_30s_contract(self) -> None:
        write_runtime(self.repo)
        commit_all(self.repo)
        with patch.object(gate, "image_id", return_value="sha256:image"), patch.object(
            gate, "production_import_smoke", return_value={"status": "PASS"}
        ):
            receipt = gate.build_receipt(
                self.repo,
                self.repo,
                "science:image",
                "scripts.openevo_webshop.worker",
                "scripts/openevo_webshop/safety.py",
            )
        self.assertEqual(receipt["status"], gate.PASS_STATUS)
        self.assertEqual(receipt["execution_git_sha"], gate.git_head(self.repo))
        self.assertEqual(receipt["owner_hot_path_contract"]["workload_start_target_seconds"], 30)
        self.assertTrue(receipt["owner_hot_path_contract"]["runner_pid_alone_is_not_success"])

    def test_validate_receipt_rejects_execution_drift(self) -> None:
        write_runtime(self.repo)
        commit_all(self.repo)
        with patch.object(gate, "image_id", return_value="sha256:image"), patch.object(
            gate, "production_import_smoke", return_value={"status": "PASS"}
        ):
            receipt = gate.build_receipt(
                self.repo,
                self.repo,
                "science:image",
                "scripts.openevo_webshop.worker",
                "scripts/openevo_webshop/safety.py",
            )
        receipt_path = self.repo / "receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        commit_all(self.repo)
        with patch.object(gate, "image_id", return_value="sha256:image"):
            with self.assertRaisesRegex(RuntimeError, "execution SHA drift"):
                gate.validate_receipt(self.repo, receipt_path)


if __name__ == "__main__":
    unittest.main()
