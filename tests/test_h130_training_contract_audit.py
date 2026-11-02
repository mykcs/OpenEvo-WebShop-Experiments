import hashlib
import json
from pathlib import Path

from scripts.openevo_webshop.audit_h130_training_contract import audit, audit_manifest_bound_overlay


MANIFEST = json.loads(
    Path("docs/evidence/remote-runs/2026-08-17/h1.30/manifests/h1.30-task-manifest-v1.json").read_text()
)


def _source(tmp_path: Path, *, placeholder: bool, random_goals: bool) -> Path:
    root = tmp_path / "seed"
    (root / "examples/data_preprocess").mkdir(parents=True)
    (root / "agent_system/environments/env_package/webshop").mkdir(parents=True)
    (root / "examples/seed_trainer/_common").mkdir(parents=True)
    (root / "examples/data_preprocess/prepare.py").write_text(
        "hiyouga/geometry3k\n" if placeholder else "webshop manifest rows\n"
    )
    (root / "agent_system/environments/env_package/webshop/envs.py").write_text(
        "range(500, len(goals))\n" if random_goals else "goal_indices = manifest['tasks']\n"
    )
    (root / "examples/seed_trainer/_common/webshop.sh").write_text(
        "prepare.py --mode text\n" if placeholder else "python train_with_manifest.py\n"
    )
    return root


def test_audit_blocks_placeholder_and_random_goal_route(tmp_path):
    result = audit(
        MANIFEST,
        _source(tmp_path, placeholder=True, random_goals=True),
        training_task_ids=["webshop_000901"],
    )
    assert result["status"] == "blocked"
    assert any("placeholder parquet" in item for item in result["failures"])
    assert any("manifest-bound goal-index" in item for item in result["failures"])
    assert result["formal_task_consumption_allowed"] is False


def test_audit_blocks_missing_training_contract(tmp_path):
    result = audit(MANIFEST, _source(tmp_path, placeholder=False, random_goals=False))
    assert result["status"] == "blocked"
    assert any("training task identity contract" in item for item in result["failures"])


def test_audit_blocks_train_eval_overlap(tmp_path):
    task_id = MANIFEST["tasks"][0]["task_id"]
    result = audit(
        MANIFEST,
        _source(tmp_path, placeholder=False, random_goals=False),
        training_task_ids=[task_id],
    )
    assert result["status"] == "blocked"
    assert any("training/evaluation task overlap" in item for item in result["failures"])


def test_manifest_bound_overlay_contract_passes_without_authorizing_formal_work(tmp_path):
    source = _source(tmp_path, placeholder=True, random_goals=True)
    source_commit = "source-commit"
    overlay = tmp_path / "overlay"
    (overlay / "agent_system/environments/env_package/webshop").mkdir(parents=True)
    (overlay / "verl/trainer/config").mkdir(parents=True)
    (overlay / "agent_system/environments/env_package/webshop/envs.py").write_text(
        "goal_indices\nself._goal_indices\n"
    )
    (overlay / "agent_system/environments/env_manager.py").write_text(
        "train_goal_indices\nval_goal_indices\n"
    )
    (overlay / "verl/trainer/config/ppo_trainer.yaml").write_text(
        "train_goal_indices: null\nval_goal_indices: null\n"
    )
    (overlay / "h130-overlay-receipt.json").write_text(
        json.dumps(
            {
                "schema": "openevo.webshop-h1.30-seed-source-overlay.v1",
                "status": "prepared-not-authorized",
                "formal_task_consumption_allowed": False,
                "source_commit": source_commit,
                "training_manifest_sha256": "manifest",
                "train_goal_indices": [],
                "validation_goal_indices": [],
            }
        )
    )
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.parquet").write_bytes(b"train")
    (data / "val.parquet").write_bytes(b"val")
    (data / "h1.30-seed-parquet-receipt.json").write_text(
        json.dumps(
            {
                "schema": "openevo.webshop-h1.30-seed-parquet.v1",
                "status": "prepared-not-authorized",
                "formal_task_consumption_allowed": False,
                "manifest_sha256": "manifest",
                "train_goal_indices": [],
                "validation_goal_indices": [],
                "files": {
                    "train": {"rows": 0, "sha256": hashlib.sha256(b"train").hexdigest()},
                    "validation": {"rows": 0, "sha256": hashlib.sha256(b"val").hexdigest()},
                },
            }
        )
    )

    import scripts.openevo_webshop.audit_h130_training_contract as module

    original = module.source_identity
    module.source_identity = lambda path: {"root": str(path), "head": source_commit}
    try:
        result = audit_manifest_bound_overlay(
            {"manifest_sha256": "manifest", "train_tasks": [], "validation_tasks": []},
            source,
            overlay,
            data,
        )
    finally:
        module.source_identity = original
    assert result["status"] == "pass"
    assert result["formal_task_consumption_allowed"] is False
