from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.openevo_webshop.experiment_create import build_draft_passport
from scripts.openevo_webshop.experiment_passport import PassportError, validate_passport

ROOT = Path(__file__).resolve().parents[1]
OE = ROOT / "scripts" / "oe"
FREEZE = ROOT / "scripts" / "openevo_webshop" / "experiment_freeze.py"
Q17 = ROOT / "configs/experiment/passports/202609082022-q17-directapply-no-gdr/passport.json"
EXPERIMENT_ID = "202609120825-passport-freeze-smoke"


def _git(root: Path, *args: str) -> None:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Passport Test")
    _git(root, "config", "user.email", "passport-test@example.invalid")


def _complete_draft(root: Path) -> Path:
    payload = json.loads(Q17.read_text(encoding="utf-8"))
    payload["experiment_id"] = EXPERIMENT_ID
    payload["status"] = "DRAFT"
    payload["created_at"] = "2026-09-12T08:25:00+08:00"
    payload["parent_experiment_id"] = None
    payload["campaign_id"] = "phase-g-passport-freeze-smoke"
    payload["scientific_identity"]["protocol_fingerprint"] = None
    payload["initial_execution_identity"]["runtime_fingerprint"] = None
    payload["passport_payload_sha256"] = None
    payload["provenance"]["status"] = "NEW_DECLARATION"
    payload["provenance"]["unknown_fields"] = []
    payload["provenance"]["parent_lineage_note"] = None
    payload["authority_paths"] = ["docs/test-authority.md"]
    payload["provenance"]["evidence_paths"] = ["docs/test-authority.md"]
    payload["resume_policy"]["reason"] = "Frozen identity still requires current launch gates and receipt lineage."
    payload["artifact_policy"]["state_model_policy"] = "docs/operations/publication/RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md"
    payload["artifact_policy"]["default_recovery_state"] = "SERVER_ONLY"
    payload["artifact_policy"]["default_disclosure_state"] = "PRIVATE"
    authority = root / "docs/test-authority.md"
    authority.parent.mkdir(parents=True, exist_ok=True)
    authority.write_text("synthetic CPU-only freeze authority\n", encoding="utf-8")
    authority_sha = hashlib.sha256(authority.read_bytes()).hexdigest()
    def resolve_refs(value):
        if isinstance(value, dict):
            if value.get("kind") == "unresolved":
                return {
                    "kind": "repository_file", "repository": None, "revision": None,
                    "path": "docs/test-authority.md", "sha256": authority_sha,
                    "evidence_path": "docs/test-authority.md", "reason": None,
                }
            return {key: resolve_refs(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve_refs(item) for item in value]
        return value
    payload = resolve_refs(payload)
    payload["initial_execution_identity"]["python_version"] = "3.11"
    path = root / "configs/experiment/passports" / EXPERIMENT_ID / "passport.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_freeze(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(OE), "experiments", "freeze", EXPERIMENT_ID, "--repo-root", str(root)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def test_freeze_requires_clean_git_and_writes_valid_frozen_identity(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repo(root)
    path = _complete_draft(root)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "complete passport draft")
    pre_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

    result = _run_freeze(root)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "FROZEN"
    assert receipt["pre_freeze_git_head"] == pre_head
    assert receipt["gpu_allocation_performed"] is False
    assert receipt["execution_authorized"] is False
    assert len(receipt["protocol_fingerprint"]) == 64
    assert len(receipt["runtime_fingerprint"]) == 64
    assert len(receipt["passport_sha256"]) == 64

    frozen = json.loads(path.read_text(encoding="utf-8"))
    validate_passport(frozen, root=root, require_frozen=True)
    assert frozen["passport_payload_sha256"] == receipt["passport_sha256"]


def test_freeze_rejects_dirty_checkout_before_mutation(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repo(root)
    path = _complete_draft(root)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "complete passport draft")
    before = path.read_bytes()
    (root / "docs/test-authority.md").write_text("dirty edit\n", encoding="utf-8")
    result = _run_freeze(root)
    assert result.returncode == 2
    assert "Git checkout is dirty" in result.stdout
    assert path.read_bytes() == before


def test_freeze_rejects_noncanonical_asset_state_policy(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repo(root)
    path = _complete_draft(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact_policy"]["state_model_policy"] = "docs/operations/publication/not-canonical.md"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "tamper asset state policy")

    result = _run_freeze(root)
    assert result.returncode == 2
    assert "artifact_policy.state_model_policy" in result.stdout
    assert "constant mismatch" in result.stdout
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "DRAFT"


def test_freeze_rejects_incomplete_g1_draft(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repo(root)
    plan = root / "docs/experiment-tracking/EXPERIMENT_PASSPORT_REGISTRY_IMPLEMENTATION_PLAN_202609111800.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("synthetic plan pointer\n", encoding="utf-8")
    payload = build_draft_passport(
        experiment_id=EXPERIMENT_ID,
        campaign_id="phase-g-passport-freeze-smoke",
        created_at="2026-09-12T08:25:00+08:00",
    )
    path = root / "configs/experiment/passports" / EXPERIMENT_ID / "passport.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "incomplete passport draft")

    result = _run_freeze(root)
    assert result.returncode == 2
    assert "freeze blocked" in result.stdout
    assert "required identity is missing" in result.stdout or "unresolved source remains" in result.stdout
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "DRAFT"


def test_freeze_module_is_git_only_and_has_no_execution_stack() -> None:
    tree = ast.parse(FREEZE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"torch", "wandb", "ray", "docker"})
    text = FREEZE.read_text(encoding="utf-8")
    for forbidden in ("docker run", "ray start", "wandb.init", "nvidia-smi"):
        assert forbidden not in text
    assert '["git", "status"' in text
    assert '["git", "rev-parse"' in text
