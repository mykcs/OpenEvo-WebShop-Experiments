from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from scripts.openevo_webshop import gpu_resource_wrapper
from scripts.openevo_webshop.path_contract import (
    PathContractError,
    run_cpu_no_gpu_probe,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
OE_PATH = ROOT / "scripts" / "oe"
SPEC = importlib.util.spec_from_loader(
    "openevo_path_contract_launcher",
    SourceFileLoader("openevo_path_contract_launcher", str(OE_PATH)),
)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _wrapper_args(receipt: Path, command: list[str], *, path_contract: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        receipt=str(receipt),
        attempt_id="attempt-test-001",
        campaign_id="campaign-test",
        stage="test",
        command_identity="sha256:test",
        gpu_uuids="GPU-a,GPU-b",
        path_contract=path_contract,
        container_root=None,
        command=command,
    )


def test_host_only_container_mapping_is_rejected() -> None:
    with pytest.raises(PathContractError, match="host-only"):
        validate_contract(
            {
                "paths": {
                    "output": {
                        "host_path": "/tmp",
                        "container_path": "/tmp",
                        "expected_type": "dir",
                        "access": "rw",
                        "writable": True,
                    }
                }
            }
        )


def test_relative_container_mapping_is_rejected() -> None:
    with pytest.raises(PathContractError, match="absolute"):
        validate_contract(
            {
                "paths": {
                    "output": {
                        "host_path": "/tmp",
                        "container_path": "workspace/runs",
                        "expected_type": "dir",
                        "access": "rw",
                        "writable": True,
                    }
                }
            }
        )


def test_cpu_probe_forces_no_gpu_visibility() -> None:
    observed: dict[str, str] = {}

    def runner(command, *, check, env, capture_output, text):
        observed["cuda"] = env["CUDA_VISIBLE_DEVICES"]
        return type("Completed", (), {"returncode": 0})()

    run_cpu_no_gpu_probe(["true"], runner=runner)
    assert observed["cuda"] == ""


def test_resource_wrapper_accounts_nonzero_payload_exactly(tmp_path: Path) -> None:
    receipt = tmp_path / "resource.json"
    return_code = gpu_resource_wrapper.run(
        _wrapper_args(receipt, [sys.executable, "-c", "raise SystemExit(7)"])
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    accounting = gpu_resource_wrapper.account_receipt(payload)
    assert return_code == 7
    assert payload["status"] == "failed"
    assert payload["wrapper_version"] == "openevo.gpu-resource-wrapper.v1"
    assert accounting["status"] == "exact"
    assert accounting["return_code"] == 7
    assert accounting["pre_gpu_failure"] is False
    assert payload["gpu_hours"] == pytest.approx(payload["wall_seconds"] * 2 / 3600.0)


def test_resource_wrapper_preflight_failure_is_exact_zero_gpu_hours(tmp_path: Path) -> None:
    receipt = tmp_path / "resource.json"
    contract = tmp_path / "bad-contract.json"
    contract.write_text(
        json.dumps(
            {
                "paths": {
                    "output": {
                        "host_path": str(tmp_path),
                        "container_path": str(tmp_path),
                        "expected_type": "dir",
                        "access": "rw",
                        "writable": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return_code = gpu_resource_wrapper.run(
        _wrapper_args(
            receipt,
            [sys.executable, "-c", "raise SystemExit(99)"],
            path_contract=str(contract),
        )
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    accounting = gpu_resource_wrapper.classify_receipt(receipt)
    assert return_code == 125
    assert payload["status"] == "preflight-failed"
    assert payload["gpu_hours"] == 0.0
    assert payload["return_code"] == 125
    assert accounting == {
        "status": "exact",
        "gpu_hours": 0.0,
        "return_code": 125,
        "pre_gpu_failure": True,
    }


def test_post_payload_accounting_corruption_is_non_retryable(tmp_path: Path) -> None:
    receipt = tmp_path / "resource.json"
    return_code = gpu_resource_wrapper.run(
        _wrapper_args(receipt, [sys.executable, "-c", "raise SystemExit(3)"])
    )
    assert return_code == 3
    receipt.write_text("corrupt-after-payload", encoding="utf-8")
    assert issubclass(gpu_resource_wrapper.ResourceAccountingError, subprocess.CalledProcessError)
    assert not issubclass(gpu_resource_wrapper.ResourceAccountingError, launcher.LaunchError)
    with pytest.raises(gpu_resource_wrapper.ResourceAccountingError, match="may have executed"):
        gpu_resource_wrapper.classify_receipt(receipt)


def _profile(tmp_path: Path) -> dict:
    for relative in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / relative).mkdir()
    return {
        "run_id": "run-test",
        "workspace": str(tmp_path),
        "identity": {"uid": 1001, "gid": 1001},
        "source": {
            "experiment": {"path": "experiment", "revision": "exp-sha"},
            "openevo": {"path": "openevo", "revision": "oe-sha"},
        },
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "command": ["true"],
        "tracking": {"mode": "offline", "group": "campaign-test"},
    }


def test_oe_builds_canonical_mount_contract_before_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(launcher, "git_identity", lambda path, revision: None)

    def fake_validate(contract, container_root=None, probe=None):
        captured.update(contract)
        if probe is not None:
            probe()
        return {"status": "pass"}

    monkeypatch.setattr(launcher, "validate_contract", fake_validate)
    monkeypatch.setattr(launcher, "run_cpu_no_gpu_probe", lambda: None)
    launcher.validate_sources_and_paths(profile)

    assert captured["paths"]["experiment"]["container_path"] == "/workspace/src/openevo-experiment"
    assert captured["paths"]["openevo"]["container_path"] == "/workspace/src/openevo-core"
    assert captured["paths"]["models"]["container_path"] == "/workspace/models"
    assert captured["paths"]["data"]["container_path"] == "/workspace/data"
    assert captured["paths"]["runs"]["container_path"] == "/workspace/runs"
    assert captured["paths"]["runs"]["access"] == "rw"
    assert captured["paths"]["cache"]["container_path"] == "/workspace/cache"


def test_oe_converts_path_contract_failure_to_launch_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile(tmp_path)
    monkeypatch.setattr(launcher, "git_identity", lambda path, revision: None)

    def fail_contract(*args, **kwargs):
        raise PathContractError("wrong run-root mapping")

    monkeypatch.setattr(launcher, "validate_contract", fail_contract)
    with pytest.raises(launcher.LaunchError, match="before GPU allocation"):
        launcher.validate_sources_and_paths(profile)


def test_oe_gpu_command_uses_atomic_exact_accounting_without_persisting_argv_secret(tmp_path: Path) -> None:
    profile = {"run_id": "run-test", "tracking": {"mode": "offline", "group": "campaign-test"}}
    secret = "secret-must-not-be-persisted"
    return_code, resource = launcher.execute_gpu_command(
        profile,
        ["GPU-a"],
        [sys.executable, "-c", f"raise SystemExit(3) # {secret}"],
        tmp_path,
        "sha256:image",
    )
    receipt = Path(resource["path"])
    text = receipt.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert return_code == 3
    assert resource["accounting"]["status"] == "exact"
    assert payload["gpu_count"] == 1
    assert payload["gpu_hours"] == pytest.approx(payload["wall_seconds"] / 3600.0)
    assert secret not in text


def test_h140_launcher_sunset_has_no_temporary_exception() -> None:
    baseline_path = ROOT / "configs" / "platform" / "campaign-launcher-legacy-baseline.v1.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["temporary_path_exceptions"] == []
    expected = {
        "scripts/openevo_webshop/launch_h140_r2_e2_harvest.py",
        "scripts/openevo_webshop/launch_h140_r2_g2_training.py",
        "scripts/openevo_webshop/launch_h140_r2_upstream_evaluation.py",
    }
    launchers = {row["path"]: row["blob_sha"] for row in baseline["launchers"]}
    assert expected <= launchers.keys()
    for path in expected:
        actual = subprocess.check_output(["git", "-C", str(ROOT), "hash-object", path], text=True).strip()
        assert launchers[path] == actual
