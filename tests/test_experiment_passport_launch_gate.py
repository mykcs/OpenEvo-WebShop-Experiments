from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from scripts.openevo_webshop import execution_readiness, experiment_passport_gate

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "execution_readiness_legacy_cases.py"
_loader = SourceFileLoader("passport_gate_legacy_cases", str(CASES))
legacy = ModuleType(_loader.name)
legacy.__file__ = str(CASES)
_loader.exec_module(legacy)


def test_campaign_cutoff_preserves_legacy_and_requires_new_formal() -> None:
    assert experiment_passport_gate.gate_requirement("20260831-2359-old", True) == "GRANDFATHERED"
    assert experiment_passport_gate.gate_requirement("20260912-0000-new", True) == "MANDATORY"
    assert experiment_passport_gate.gate_requirement("20260912-0000-new", False) == "NOT_APPLICABLE"


def test_real_q17_registered_frozen_binding_passes() -> None:
    result = experiment_passport_gate.validate_binding(ROOT, {
        "experiment_id": "202609082022-q17-directapply-no-gdr",
        "passport_sha256": "155e8885d2af2d4ee8b281f75cfec5c8b3af18b2c42b63f440da2af3e00805b6",
    })
    assert result["status"] == "PASS"
    assert result["passport_status"] == "FROZEN"
    assert "run-manifest" in result["approved_artifact_roles"]

def _run_manifest(tmp_path: Path, experiment_id: str | None, passport_sha256: str | None) -> Path:
    document = legacy.json.loads((ROOT / "templates/run-manifest.example.json").read_text(encoding="utf-8"))
    document["experiment"]["experiment_id"] = experiment_id
    document["experiment"]["passport_sha256"] = passport_sha256
    path = tmp_path / "run-manifest.json"
    path.write_text(legacy.json.dumps(document), encoding="utf-8")
    return path


def test_run_manifest_crosslink_matches_registered_passport(tmp_path: Path) -> None:
    passport_sha = "155e8885d2af2d4ee8b281f75cfec5c8b3af18b2c42b63f440da2af3e00805b6"
    gate = experiment_passport_gate.validate_binding(ROOT, {
        "experiment_id": "202609082022-q17-directapply-no-gdr",
        "passport_sha256": passport_sha,
    })
    manifest = _run_manifest(tmp_path, gate["experiment_id"], passport_sha)
    result = execution_readiness.validate_run_manifest_passport_link(manifest, gate, repo_root=ROOT)
    assert result["status"] == "PASS"
    assert result["experiment_id"] == gate["experiment_id"]


def test_run_manifest_crosslink_mismatch_fails_closed(tmp_path: Path) -> None:
    passport_sha = "155e8885d2af2d4ee8b281f75cfec5c8b3af18b2c42b63f440da2af3e00805b6"
    gate = experiment_passport_gate.validate_binding(ROOT, {
        "experiment_id": "202609082022-q17-directapply-no-gdr",
        "passport_sha256": passport_sha,
    })
    manifest = _run_manifest(tmp_path, gate["experiment_id"], "0" * 64)
    with pytest.raises(execution_readiness.ExecutionReadinessError, match="passport_sha256"):
        execution_readiness.validate_run_manifest_passport_link(manifest, gate, repo_root=ROOT)



def _campaign(tmp_path: Path, campaign_id: str):
    profile_path = legacy.write_profile(tmp_path)
    return legacy.make_campaign(
        tmp_path,
        campaign_id=campaign_id,
        profile=legacy.profile_spec(profile_path),
    )


def test_new_formal_campaign_missing_passport_fails_before_runtime(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, "20260912-1200-new-formal")
    args = legacy.args_for(tmp_path, campaign)
    qual_patch, gate_patch = legacy.mocked_gates(tmp_path)
    with qual_patch as qual_mock, gate_patch as live_mock:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED_NOT_PREPARED"
    assert result["stages"][-1]["stage"] == "experiment-passport"
    assert qual_mock.call_count == 0
    assert live_mock.call_count == 0

def test_readiness_blocks_manifest_identity_mismatch_before_runtime(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, "20260831-1200-existing-formal")
    args = legacy.args_for(tmp_path, campaign)
    args.run_manifest = _run_manifest(
        tmp_path, "202609082022-q17-directapply-no-gdr", "0" * 64
    )
    gate = {
        "status": "PASS", "requirement": "MANDATORY",
        "experiment_id": "202609082022-q17-directapply-no-gdr",
        "passport_sha256": "155e8885d2af2d4ee8b281f75cfec5c8b3af18b2c42b63f440da2af3e00805b6",
        "passport_status": "FROZEN", "approved_artifact_roles": ["run-manifest"],
    }
    qual_patch, gate_patch = legacy.mocked_gates(tmp_path)
    with patch.object(experiment_passport_gate, "evaluate_release_gate", return_value=gate), qual_patch as qual_mock, gate_patch as live_mock:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED_NOT_PREPARED"
    assert result["stages"][-1]["stage"] == "run-manifest-passport-link"
    assert qual_mock.call_count == 0
    assert live_mock.call_count == 0



def test_pre_cutoff_formal_campaign_is_grandfathered(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, "20260831-1200-existing-formal")
    args = legacy.args_for(tmp_path, campaign)
    qual_patch, gate_patch = legacy.mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "READY"
    assert result["experiment_passport"]["status"] == "GRANDFATHERED"
    receipt = Path(result["ready_receipt_path"])
    persisted = legacy.json.loads(receipt.read_text(encoding="utf-8"))
    assert persisted["experiment_passport"]["status"] == "GRANDFATHERED"


def test_startup_projection_is_informational_and_surfaces_binding() -> None:
    from scripts.experiment_startup_report import passport_projection
    missing = passport_projection({
        "campaign_id": "20260912-1200-new-formal",
        "formal_task_consumption_allowed": True,
    })
    assert missing["status"] == "MISSING"
    assert missing["informational_only"] is True
    bound = passport_projection({
        "campaign_id": "20260912-1200-new-formal",
        "formal_task_consumption_allowed": True,
        "experiment_identity": {
            "experiment_id": "202609082022-q17-directapply-no-gdr",
            "passport_sha256": "155e8885d2af2d4ee8b281f75cfec5c8b3af18b2c42b63f440da2af3e00805b6",
        },
    })
    assert bound["status"] == "PASS"
    assert bound["passport_status"] == "FROZEN"
    assert bound["informational_only"] is True


def test_current_campaign_startup_projection_does_not_retroactively_block() -> None:
    from scripts.experiment_startup_report import load_campaign, passport_projection
    current = load_campaign(ROOT / "configs/experiment/current-campaign.json")
    projection = passport_projection(current)
    assert projection["status"] in {"NOT_BOUND", "GRANDFATHERED", "PASS"}
    assert projection["informational_only"] is True


def test_new_formal_release_with_registered_frozen_binding_passes_gate() -> None:
    result = experiment_passport_gate.evaluate_release_gate({
        "campaign_id": "20260912-1200-new-formal",
        "formal_task_consumption_allowed": True,
        "experiment_identity": {
            "experiment_id": "202609082022-q17-directapply-no-gdr",
            "passport_sha256": "155e8885d2af2d4ee8b281f75cfec5c8b3af18b2c42b63f440da2af3e00805b6",
        },
    }, ROOT)
    assert result["status"] == "PASS"
    assert result["requirement"] == "MANDATORY"
    assert result["passport_status"] == "FROZEN"
