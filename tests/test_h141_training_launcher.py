from pathlib import Path

from scripts.openevo_webshop.run_h141_magnitude_training import container_path, docker_command, training_artifact_id


def test_h141_container_path_is_run_root_bound():
    root = Path("/workspace/runs/phase-h141-magnitude-screen-20260821")
    path = root / "inputs" / "datasets" / "increment-00-webshop_005386" / "manifest.json"
    assert container_path(path, root) == "/workspace/runs/phase-h141-magnitude-screen-20260821/inputs/datasets/increment-00-webshop_005386/manifest.json"


def test_h141_reset_arm_uses_only_explicit_reset_env():
    paths = {"experiment": "/repo", "openevo_clean": "/clean", "openevo_reset_patch": "/patched", "model": "/model", "cache": "/cache", "run_root": "/workspace/runs/phase-h141-magnitude-screen-20260821", "adapter": "/adapter"}
    run_root = Path(paths["run_root"])
    manifest = run_root / "inputs" / "manifest.json"
    artifact = run_root / "training" / "C1_reset_magnitude" / "artifacts"
    evidence = run_root / "training" / "C1_reset_magnitude" / "evidence.json"
    command = docker_command(paths, run_root, manifest, artifact, evidence, "run", 0, "/workspace/prior-g1", "g1", "C1_reset_magnitude", "GPU-x", "sha")
    assert "/patched:/workspace/src/openevo-core:ro" in command
    assert "/workspace/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/" in " ".join(command)
    assert "/adapter:/workspace/runs/phase-h141-magnitude-screen-20260821/training/C1_reset_magnitude/artifacts/prior-g1:ro" in command
    assert "OPENEVO_SD_LORA_RESET_ALL_MAGNITUDES=1" in command


def test_h141_control_does_not_enable_reset_env():
    paths = {"experiment": "/repo", "openevo_clean": "/clean", "openevo_reset_patch": "/patched", "model": "/model", "cache": "/cache", "run_root": "/workspace/runs/phase-h141-magnitude-screen-20260821", "adapter": "/adapter"}
    run_root = Path(paths["run_root"])
    command = docker_command(paths, run_root, run_root / "manifest.json", run_root / "artifact", run_root / "evidence.json", "run", 0, "/workspace/prior-g1", "g1", "C0_exact_h140", "GPU-x", "sha")
    assert "/clean:/workspace/src/openevo-core:ro" in command
    assert "OPENEVO_SD_LORA_RESET_ALL_MAGNITUDES=1" not in command


def test_h141_training_artifact_identity_falls_back_to_manifest_adapter_id():
    assert training_artifact_id({"artifact": {}, "artifact_manifest": {"adapter_id": "sd-lora-example"}}) == "sd-lora-example"
