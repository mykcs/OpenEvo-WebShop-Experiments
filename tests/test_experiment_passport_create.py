from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.openevo_webshop.experiment_passport import PassportError, validate_passport

ROOT = Path(__file__).resolve().parents[1]
OE = ROOT / "scripts" / "oe"
CREATE = ROOT / "scripts" / "openevo_webshop" / "experiment_create.py"
PLAN = "docs/experiment-tracking/EXPERIMENT_PASSPORT_REGISTRY_IMPLEMENTATION_PLAN_202609111800.md"
EXPERIMENT_ID = "202609120700-passport-create-smoke"


def _repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    plan = root / PLAN
    plan.parent.mkdir(parents=True)
    plan.write_text("test authority pointer\n", encoding="utf-8")
    return root


def _run_create(root: Path, experiment_id: str = EXPERIMENT_ID) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, str(OE), "experiments", "create", experiment_id,
            "--campaign-id", "test-passport-campaign",
            "--created-at", "2026-09-12T07:00:00+08:00",
            "--repo-root", str(root),
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def test_create_writes_schema_complete_draft_without_freezing(tmp_path: Path) -> None:
    root = _repo_root(tmp_path)
    result = _run_create(root)
    assert result.returncode == 0, result.stderr + result.stdout
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "DRAFT"
    assert receipt["experiment_id"] == EXPERIMENT_ID
    assert receipt["gpu_allocation_performed"] is False
    assert receipt["execution_authorized"] is False

    path = root / "configs" / "experiment" / "passports" / EXPERIMENT_ID / "passport.json"
    draft = json.loads(path.read_text(encoding="utf-8"))
    assert draft["status"] == "DRAFT"
    assert draft["passport_payload_sha256"] is None
    assert draft["scientific_identity"]["protocol_fingerprint"] is None
    assert draft["initial_execution_identity"]["runtime_fingerprint"] is None
    assert draft["artifact_policy"]["default_recovery_state"] == "SERVER_ONLY"
    assert draft["artifact_policy"]["default_disclosure_state"] == "PRIVATE"
    assert draft["artifact_policy"]["state_model_policy"] == "docs/operations/publication/RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md"
    validate_passport(draft, root=root, require_frozen=False)
    with pytest.raises(PassportError, match="DRAFT is not a frozen scientific identity"):
        validate_passport(draft, root=root)


def test_create_refuses_overwrite_and_invalid_identity(tmp_path: Path) -> None:
    root = _repo_root(tmp_path)
    first = _run_create(root)
    assert first.returncode == 0
    second = _run_create(root)
    assert second.returncode == 2
    assert "refusing to overwrite existing Passport draft" in second.stdout

    invalid = _run_create(root, "not-a-canonical-id")
    assert invalid.returncode == 2
    assert "string identity format mismatch" in invalid.stdout


def test_successor_metadata_is_draft_only(tmp_path: Path) -> None:
    root = _repo_root(tmp_path)
    parent = "202609082022-q17-directapply-no-gdr"
    result = subprocess.run(
        [
            sys.executable, str(OE), "experiments", "create", EXPERIMENT_ID,
            "--campaign-id", "test-passport-campaign", "--successor-of", parent,
            "--created-at", "2026-09-12T07:00:00+08:00", "--repo-root", str(root),
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    path = root / "configs" / "experiment" / "passports" / EXPERIMENT_ID / "passport.json"
    draft = json.loads(path.read_text(encoding="utf-8"))
    assert draft["parent_experiment_id"] == parent
    assert draft["resume_policy"]["policy"] == "MANUAL_REVIEW_ONLY"
    assert draft["resume_policy"]["automatic_gpu_start"] is False


def test_create_module_has_no_execution_or_tracking_imports() -> None:
    tree = ast.parse(CREATE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"subprocess", "torch", "wandb", "ray"})
