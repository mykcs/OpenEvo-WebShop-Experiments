"""Compatibility collector for the historical execution-readiness regression suite.

The production contract now requires a hash-bound execution profile. Historical
unit fixtures predated that invariant and treated a missing profile as the
implicit default. Rather than weakening production readiness again, this module
keeps the historical cases intact and gives their *default fixture* an explicit
frozen profile. Tests that intentionally exercise missing-profile behavior still
pass ``profile=None`` explicitly.

The original cases live in ``execution_readiness_legacy_cases.py`` only to avoid
rewriting a large, already-reviewed regression corpus while the platform
contract migrates. Pytest collects the re-exported tests here.
"""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


_CASES_PATH = Path(__file__).with_name("execution_readiness_legacy_cases.py")
_loader = SourceFileLoader("execution_readiness_legacy_cases", str(_CASES_PATH))
_legacy = ModuleType(_loader.name)
_legacy.__file__ = str(_CASES_PATH)
_loader.exec_module(_legacy)


def test_readiness_preserves_runtime_reuse_decision(tmp_path):
    campaign = make_campaign(tmp_path)
    args = _legacy.args_for(tmp_path, campaign)
    qual_patch, gate_patch = _legacy.mocked_gates(tmp_path)
    decision = {"reason": "exact_request_match", "changed_request_fields": []}
    with qual_patch as qualify, gate_patch:
        qualify.return_value["qualification_decision"] = decision
        result = _legacy.execution_readiness.evaluate_readiness(args)
    assert result["runtime_qualification_decision"] == decision

_AUTO_PROFILE = object()
_original_make_campaign = _legacy.make_campaign


def make_campaign(tmp_path: Path, *args, profile=_AUTO_PROFILE, **kwargs):
    """Make new test campaigns scientifically executable by default.

    ``profile=None`` remains an explicit request for a release with no execution
    profile and is used by the negative missing-profile test below.
    """
    if profile is _AUTO_PROFILE:
        profile_path = _legacy.write_profile(tmp_path)
        profile = _legacy.profile_spec(profile_path)
    return _original_make_campaign(tmp_path, *args, profile=profile, **kwargs)


# Historical test functions resolve ``make_campaign`` from their defining
# module at call time, so this updates every default fixture without editing the
# old corpus itself.
_legacy.make_campaign = make_campaign

_OVERRIDDEN = {
    "test_prepared_matching_runtime_is_ready",
    "test_oe_run_campaign_requires_hash_bound_profile",
}

for _name, _value in vars(_legacy).items():
    if _name.startswith("test_") and callable(_value) and _name not in _OVERRIDDEN:
        globals()[_name] = _value


def test_prepared_matching_runtime_is_ready(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    args = _legacy.args_for(tmp_path, campaign)
    qual_patch, gate_patch = _legacy.mocked_gates(tmp_path)
    with qual_patch as qual_mock, gate_patch as gate_mock:
        result = _legacy.execution_readiness.evaluate_readiness(args)

    assert result["verdict"] == "READY"
    assert result["qualification_reused"] is True
    assert qual_mock.call_count == 1
    assert gate_mock.call_count == 1
    receipt_path = Path(result["ready_receipt_path"])
    assert receipt_path.is_file()
    assert receipt_path.parent.name == "control"
    persisted = _legacy.json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["stages"] == result["stages"]
    assert persisted["stages"][-1]["stage"] == "ready-receipt"
    assert [stage["stage"] for stage in persisted["stages"]] == [
        "resolve-campaign",
        "current-authority",
        "frozen-scientific-artifacts",
        "scientific-identity",
        "execution-profile",
        "runtime-qualification",
        "live-gate",
        "ready-receipt",
    ]
    assert persisted["authority"]["router_status"] == "launch-authorized"
    assert persisted["scientific_identity_sha256"].startswith("sha256:")
    assert persisted["profile"]["sha256"].startswith("sha256:")


def test_oe_run_campaign_requires_hash_bound_profile(tmp_path: Path) -> None:
    oe = _legacy.load_oe_module()
    campaign = make_campaign(tmp_path, profile=None)
    ns = _legacy.oe_campaign_namespace(tmp_path, campaign, dry_run=True)
    qual_patch, gate_patch = _legacy.mocked_gates(tmp_path)
    with qual_patch as qual_mock, gate_patch as gate_mock:
        with pytest.raises(oe.LaunchError, match="hash-bound"):
            oe.command_run_campaign(ns)
    # Missing scientific execution plan fails before runtime/live-resource work.
    assert qual_mock.call_count == 0
    assert gate_mock.call_count == 0
