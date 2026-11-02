import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.openevo_webshop.execution_runtime_resource_canary_readiness import (
    DEFAULT_PRECANARY,
    DEFAULT_RELEASE,
    ResourceCanaryReadinessError,
    build_readiness_report,
)


@pytest.fixture
def frozen_package_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate report logic at the sealed checkout; never qualify current bytes."""
    from scripts.openevo_webshop import execution_runtime_resource_canary_readiness as readiness
    from scripts.openevo_webshop.experiment_passport import canonical_bytes

    package = json.loads(DEFAULT_PRECANARY.read_text())["production_runtime_package"]
    projection = {key: value for key, value in package.items() if key != "package_identity"}
    assert hashlib.sha256(canonical_bytes(projection)).hexdigest() == package["package_identity"]
    assert package["package_identity"] == "aa7b28c23a44540ae07d4322bdde540fe4d46cc5773d1d0c716ca33f20fd5da0"

    def validate_frozen(observed: dict) -> None:
        assert observed == package

    monkeypatch.setattr(readiness, "build_production_package", lambda: package)
    monkeypatch.setattr(readiness, "validate_production_package", validate_frozen)


def test_post_canary_readiness_validates_canary_and_consumer_without_authority(
    frozen_package_checkout: None,
) -> None:
    report = build_readiness_report(require_clean_head=False)
    assert report["status"] == "PASS_CANARY_AND_CONSUMER_VALIDATION"
    assert report["gpu_requests"] == 0
    assert report["formal_rows_consumed"] == 0
    assert report["model_calls"] == 0
    assert report["final_panel_access_count"] == 0
    assert report["resource_canary_executed"] is True
    assert report["exact_consumer_validation_passed"] is True
    assert report["canary_execution_authority"] is False
    assert report["resource_topology_authority"] is False
    assert report["formal_launch_authority"] is False
    assert report["blockers"] == []


def test_release_package_drift_fails_closed(tmp_path: Path, frozen_package_checkout: None) -> None:
    release = json.loads(DEFAULT_RELEASE.read_text(encoding="utf-8"))
    release["production_runtime_package_identity"] = "0" * 64
    drifted = tmp_path / "release.json"
    drifted.write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ResourceCanaryReadinessError):
        build_readiness_report(release_path=drifted, require_clean_head=False)


def test_current_checkout_cannot_reuse_a_different_sealed_package() -> None:
    from scripts.openevo_runtime.production_package import build_production_package

    frozen = json.loads(DEFAULT_PRECANARY.read_text())["production_runtime_package"]
    current = build_production_package()
    if current["package_identity"] != frozen["package_identity"]:
        with pytest.raises(ResourceCanaryReadinessError, match="pre-canary package identity drift"):
            build_readiness_report(require_clean_head=False)
    else:
        assert build_readiness_report(require_clean_head=False)["production_runtime_package_identity"] == frozen["package_identity"]


def test_direct_cli_help_is_import_safe() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/openevo_webshop/execution_runtime_resource_canary_readiness.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--expected-git-sha" in completed.stdout
