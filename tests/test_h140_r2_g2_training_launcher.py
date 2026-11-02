import unittest
from pathlib import Path

from scripts.openevo_webshop.launch_h140_r2_g2_training import CONTAINER_RUN_ROOT, bind_gpu_uuid, container_path, docker_command


class H140R2G2TrainingLauncherTests(unittest.TestCase):
    def test_container_path_preserves_training_run_root(self):
        base = Path("/workspace/runs/phase-h140-g2-r2-20260820")
        path = base / "formal-g2-training" / "inputs" / "datasets" / "increment-00-webshop_005386" / "manifest.json"
        self.assertEqual(container_path(path, base), f"{CONTAINER_RUN_ROOT}/formal-g2-training/inputs/datasets/increment-00-webshop_005386/manifest.json")

    def test_training_command_binds_frozen_parameters(self):
        base = "/workspace/runs/phase-h140-g2-r2-20260820"
        root = Path(__file__).parents[1]
        command = docker_command(
            {"run_root": base, "experiment": "/repo", "openevo": "/core", "model": "/model", "cache": "/cache", "adapter": "/adapter"},
            Path(base), Path(base) / "formal-g2-training/inputs/datasets/increment-00-webshop_005386/manifest.json",
            Path(base) / "formal-g2-training/artifacts/sd-lora", Path(base) / "formal-g2-training/evidence/g2-increment-00.json", "g2-00", 0,
            "/workspace/runs/g1-adapter", "g1", "online", root, "sha",
        )
        self.assertIn("--max-steps", command)
        self.assertEqual(command[command.index("--max-steps") + 1], "16")
        self.assertEqual(command[command.index("--max-records") + 1], "2")
        self.assertIn("--init", command)
        self.assertIn("--prior-adapter", command)
        self.assertTrue(any("/workspace/runs/phase-h140-g2-r2-20260820/formal-g2-training/artifacts/sd-lora/prior-g1" in item for item in command))
        self.assertIn("/core:/workspace/openevo-core:ro", command)
        self.assertTrue(any("/workspace/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/" in item for item in command))
        self.assertIn("HF_HUB_CACHE=/workspace/hf-cache/hub", command)
        self.assertIn("--upstream-openevo-sha", command)
        self.assertEqual(command[command.index("--base-model") + 1], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(command[command.index("--python") + 1], "/workspace/src/openevo-experiment/scripts/openevo_webshop/server_trainer_python.sh")
        self.assertIn("OPENEVO_TRAINER_CORE=/workspace/src/openevo-core", command)
        self.assertIn("/workspace/runs/phase-h140-g2-r2-20260820/formal-g2-training/inputs/datasets/increment-00-webshop_005386/manifest.json", command)
        self.assertIn("/workspace/runs/phase-h140-g2-r2-20260820/formal-g2-training/artifacts/sd-lora/wandb", command)

    def test_gpu_uuid_binding_replaces_embedded_docker_values(self):
        bound = bind_gpu_uuid(["--gpus", "device=GPU_PLACEHOLDER", "-e", "CUDA_VISIBLE_DEVICES=GPU_PLACEHOLDER"], "GPU-test")
        self.assertEqual(bound, ["--gpus", "device=GPU-test", "-e", "CUDA_VISIBLE_DEVICES=GPU-test"])


if __name__ == "__main__":
    unittest.main()
