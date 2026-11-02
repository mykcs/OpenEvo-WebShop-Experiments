from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from scripts.ci import preflight

ROOT = Path(__file__).resolve().parents[1]


def copy_successor(root: Path) -> None:
    for relative in (
        "scripts/ci/workers_builds_trusted_entry_20260928.py",
        "scripts/ci/workers_builds_immutable_executor_20260928.py",
    ):
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_current_successor_contract_passes() -> None:
    assert preflight.check_workers_builds_successor_contract(ROOT) == []


def test_provider_command_binds_stable_trigger_to_live_pr_head() -> None:
    contract = json.loads(
        (ROOT / "scripts/ci/workers_builds_provider_contract_20260928.json").read_text(
            encoding="utf-8"
        )
    )
    command = contract["build_command"]
    assert contract["production_branch"] == "ci-final"
    assert contract["automatic_production_builds"] is True
    assert contract["automatic_preview_builds"] is False
    assert contract["preview_branch_includes"] == []
    assert "git fetch origin main:refs/remotes/origin/main" in command
    assert "git show \"$m:scripts/ci/workers_builds_trusted_entry_20260928.py\"" in command
    assert "--mode full" in command
    assert subprocess.run(["bash", "-n", "-c", command], check=False).returncode == 0


def test_preflight_rejects_provider_environment_inheritance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        copy_successor(root)
        path = root / "scripts/ci/workers_builds_immutable_executor_20260928.py"
        text = path.read_text(encoding="utf-8").replace(
            'host_env = {',
            'host_env = os.environ.copy()\n    host_env.update({',
            1,
        ).replace(
            '        "WHEELHOUSE": str(wheelhouse),\n    }',
            '        "WHEELHOUSE": str(wheelhouse),\n    })',
            1,
        )
        path.write_text(text, encoding="utf-8")

        errors = preflight.check_workers_builds_successor_contract(root)

        assert any("must not inherit the provider environment" in error for error in errors)


def test_preflight_rejects_missing_exact_head_identity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        copy_successor(root)
        path = root / "scripts/ci/workers_builds_trusted_entry_20260928.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "WORKERS_CI_COMMIT_SHA",
                "REMOVED_PROVIDER_SHA",
            ),
            encoding="utf-8",
        )

        errors = preflight.check_workers_builds_successor_contract(root)

        assert any("exact provider SHA" in error for error in errors)


def test_preflight_rejects_missing_empty_environment_boundary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        copy_successor(root)
        path = root / "scripts/ci/workers_builds_immutable_executor_20260928.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace("/usr/bin/env -i", "/usr/bin/env"),
            encoding="utf-8",
        )

        errors = preflight.check_workers_builds_successor_contract(root)

        assert any("empty candidate environment" in error for error in errors)


def test_preflight_rejects_missing_credential_surface_probe() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        copy_successor(root)
        path = root / "scripts/ci/workers_builds_immutable_executor_20260928.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "python3 /wheelhouse/credential_probe.py", "true"
            ),
            encoding="utf-8",
        )

        errors = preflight.check_workers_builds_successor_contract(root)

        assert any("credential surface probe" in error for error in errors)
