from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import oe_impl
from scripts.openevo_webshop import runtime_live_gate


def qualification(tmp_path: Path, *, status: str = "PASS", reused: bool = True) -> Path:
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(
            {
                "schema": "openevo.runtime-qualification.v1",
                "status": status,
                "cache_key": "abc",
                "reused": reused,
                "request": {"runtime_fingerprint_sha256": "sha256:runtime"},
            }
        ),
        encoding="utf-8",
    )
    return path


def args_for(tmp_path: Path) -> Namespace:
    return Namespace(
        qualification_receipt=qualification(tmp_path),
        runtime_python=None,
        cache_root=None,
        artifact_root=None,
        storage_root=None,
        output_root=tmp_path / "new-output",
        minimum_free_bytes=1,
        require_cuda=False,
        resume=False,
        experiment_config=None,
        run_manifest=None,
        output=None,
    )


def pass_check() -> dict:
    return {"status": "pass", "errors": []}


def test_startup_doctor_checks_a_separate_run_output_root(tmp_path: Path) -> None:
    campaign = tmp_path / "current-campaign.json"
    campaign.write_text(
        json.dumps({"campaign_id": "q03-test", "status": "launch-authorized", "formal_task_consumption_allowed": True}),
        encoding="utf-8",
    )
    release = tmp_path / "release.json"
    release.write_text("{\"upstream_patch\": {}}\n", encoding="utf-8")
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "startup"
    qargs = Namespace(
        runtime_python=Path("/venv/bin/python"),
        source_root=None,
        webshop_data=None,
        model=None,
        openevo_root=None,
        index_root=None,
        image=None,
        expected_image_digest=None,
        profile="platform",
        runtime_container=False,
        require_java=False,
        allow_dirty=False,
    )
    qualification_result = {
        "status": "PASS",
        "reused": True,
        "cache_key": "cache-key",
        "request": {"runtime_fingerprint_sha256": "sha256:fingerprint"},
        "qualification_decision": {},
        "failures": [],
    }
    def evaluate_dry_run(readiness_args: Namespace) -> dict[str, object]:
        dry_run_root = Path(readiness_args.output_root)
        control_root = dry_run_root / "control"
        assert dry_run_root.is_dir()
        assert runtime_live_gate.envdoc.check_write_target(control_root)["status"] == "pass"
        assert runtime_live_gate.envdoc.check_storage(control_root, 0)["status"] == "pass"
        return {"verdict": "READY", "ready_receipt_path": str(tmp_path / "ready.json")}

    args = Namespace(
        output_root=output_root,
        registry=tmp_path / "registry.json",
        current_campaign=campaign,
        release=release,
        fingerprint_file=fingerprint,
        cache_root=tmp_path / "cache",
        allow_dirty=False,
        profile_route="platform",
        runtime_python=qargs.runtime_python,
        source_root=None,
        webshop_data=None,
        model=None,
        openevo_root=None,
        index_root=None,
        require_java=None,
        run_campaign_dry_run=True,
    )
    with (
        patch.object(
            oe_impl.workspace_discovery,
            "classify_checkout",
            return_value={"is_canonical_control": True, "status": "OK"},
        ),
        patch.object(oe_impl.execution_readiness, "build_qualification_args", return_value=qargs),
        patch.object(oe_impl.runtime_qualification, "qualify", return_value=qualification_result),
        patch.object(oe_impl.runtime_live_gate, "run_gate", return_value={"status": "PASS", "failures": []}) as gate,
        patch.object(
            oe_impl.execution_readiness,
            "evaluate_readiness",
            side_effect=evaluate_dry_run,
        ),
    ):
        assert oe_impl._command_doctor_startup(args) == 0

    live_gate_args = gate.call_args.args[0]
    assert live_gate_args.output_root == output_root / "run"
    assert output_root.is_dir()
    assert (output_root / "dry-run").is_dir()
    assert not live_gate_args.output_root.exists()
    assert runtime_live_gate.check_output_collision(live_gate_args.output_root, resume=False)["status"] == "pass"


def test_non_pass_qualification_is_rejected(tmp_path: Path) -> None:
    path = qualification(tmp_path, status="FAIL")
    with pytest.raises(ValueError, match="not PASS"):
        runtime_live_gate.load_qualification(path)


def test_existing_output_requires_explicit_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    blocked = runtime_live_gate.check_output_collision(output, resume=False)
    allowed = runtime_live_gate.check_output_collision(output, resume=True)
    assert blocked["status"] == "fail"
    assert allowed["status"] == "pass"


def test_reused_qualification_still_rechecks_live_facts(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    with (
        patch.object(runtime_live_gate.envdoc, "check_cuda", return_value=pass_check()) as cuda,
        patch.object(runtime_live_gate.envdoc, "check_nvidia_smi", return_value=pass_check()) as smi,
        patch.object(runtime_live_gate.envdoc, "check_cache", return_value=pass_check()) as cache,
        patch.object(runtime_live_gate.envdoc, "check_write_target", return_value=pass_check()) as writable,
        patch.object(runtime_live_gate.envdoc, "check_storage", return_value=pass_check()) as storage,
        patch.object(runtime_live_gate.envdoc, "check_experiment_config", return_value=pass_check()) as config,
        patch.object(runtime_live_gate.envdoc, "check_run_manifest", return_value=pass_check()) as manifest,
    ):
        result = runtime_live_gate.run_gate(args)
    assert result["status"] == "PASS"
    assert result["checks"]["qualification"]["reused"] is True
    assert result["checks"]["gpu_idle"]["status"] == "delegated"
    assert result["checks"]["wandb"]["status"] == "delegated"
    for mocked in (cuda, smi, cache, storage, config, manifest):
        assert mocked.call_count == 1
    assert writable.call_count == 2


def test_live_failure_blocks_even_with_reused_qualification(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    failed_storage = {"status": "fail", "errors": ["disk full"]}
    with (
        patch.object(runtime_live_gate.envdoc, "check_cuda", return_value=pass_check()),
        patch.object(runtime_live_gate.envdoc, "check_nvidia_smi", return_value=pass_check()),
        patch.object(runtime_live_gate.envdoc, "check_cache", return_value=pass_check()),
        patch.object(runtime_live_gate.envdoc, "check_write_target", return_value=pass_check()),
        patch.object(runtime_live_gate.envdoc, "check_storage", return_value=failed_storage),
        patch.object(runtime_live_gate.envdoc, "check_experiment_config", return_value=pass_check()),
        patch.object(runtime_live_gate.envdoc, "check_run_manifest", return_value=pass_check()),
    ):
        result = runtime_live_gate.run_gate(args)
    assert result["status"] == "FAIL"
    assert "storage: disk full" in result["failures"]
