from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "ci" / "check_campaign_launcher_contract.py"
SPEC = importlib.util.spec_from_file_location("campaign_launcher_contract", MODULE_PATH)
assert SPEC and SPEC.loader
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def git_blob(root: Path, path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "hash-object", str(path)], text=True).strip()


def make_root(tmp_path: Path, legacy_text: str = "#!/bin/sh\ndocker run old-image\n") -> tuple[Path, Path]:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    launcher_dir = tmp_path / "scripts" / "openevo_webshop"
    launcher_dir.mkdir(parents=True)
    baseline_dir = tmp_path / "configs" / "platform"
    baseline_dir.mkdir(parents=True)
    legacy = launcher_dir / "launch_legacy.sh"
    legacy.write_text(legacy_text, encoding="utf-8")
    baseline = {
        "schema": "openevo.platform-campaign-launcher-legacy-baseline.v1",
        "cutoff_git_sha": "test",
        "policy": "test",
        "launchers": [{"path": "scripts/openevo_webshop/launch_legacy.sh", "blob_sha": git_blob(tmp_path, legacy)}],
    }
    (baseline_dir / "campaign-launcher-legacy-baseline.v1.json").write_text(json.dumps(baseline), encoding="utf-8")
    return tmp_path, legacy


def test_exact_legacy_blob_is_grandfathered(tmp_path: Path) -> None:
    root, _ = make_root(tmp_path)
    errors, notes = contract.check(root)
    assert errors == []
    assert notes == []


def test_modifying_legacy_launcher_loses_grandfathering(tmp_path: Path) -> None:
    root, legacy = make_root(tmp_path)
    legacy.write_text("#!/bin/sh\ndocker run changed-image\n", encoding="utf-8")
    errors, _ = contract.check(root)
    assert any("must delegate to scripts/oe" in error for error in errors)
    assert any("direct docker orchestration" in error for error in errors)


def test_new_launcher_cannot_redeclare_runtime_facts(tmp_path: Path) -> None:
    root, _ = make_root(tmp_path)
    launcher = root / "scripts" / "openevo_webshop" / "launch_h141_bad.py"
    launcher.write_text(
        'PYTHON = "/workspace/runtime/bin/python"\n'
        'JAVA_HOME = "/workspace/jdk"\n'
        'cmd = ["docker", "run", "image"]\n',
        encoding="utf-8",
    )
    errors, _ = contract.check(root)
    assert any("must delegate to scripts/oe" in error for error in errors)
    assert any("absolute host runtime path" in error for error in errors)
    assert any("direct docker orchestration" in error for error in errors)
    assert any("launcher-owned JAVA_HOME" in error for error in errors)
    assert any("launcher-owned Python runtime" in error for error in errors)


def test_new_thin_launcher_may_not_use_legacy_oe_run(tmp_path: Path) -> None:
    """A new launcher hiding formal GPU work behind `oe run <profile>` bypasses run-campaign."""
    root, _ = make_root(tmp_path)
    launcher = root / "scripts" / "openevo_webshop" / "launch_h141_legacy_profile.sh"
    launcher.write_text(
        '#!/bin/sh\nexec "$(git rev-parse --show-toplevel)/scripts/oe" run profile.json --gpus "${2:-auto:1}"\n',
        encoding="utf-8",
    )
    errors, _ = contract.check(root)
    assert any("legacy 'oe run <profile>'" in error for error in errors)


def test_new_launcher_referencing_formal_campaign_config_must_use_run_campaign(tmp_path: Path) -> None:
    """Machine-verifiable classification: a referenced formal campaign config forces run-campaign."""
    root, _ = make_root(tmp_path)
    (root / "configs" / "experiment").mkdir(parents=True)
    config = root / "configs" / "experiment" / "h-new-campaign-v1.json"
    config.write_text(
        json.dumps(
            {
                "schema": "openevo.experiment-h-new-campaign-preregistration.v1",
                "campaign_id": "h-new",
                "formal_task_consumption_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    launcher = root / "scripts" / "openevo_webshop" / "launch_h_new.sh"
    launcher.write_text(
        '#!/bin/sh\nexec "$(git rev-parse --show-toplevel)/scripts/oe" host '
        '--path-contract configs/experiment/h-new-campaign-v1.json -- true\n',
        encoding="utf-8",
    )
    errors, _ = contract.check(root)
    assert any("must delegate through scripts/oe run-campaign" in error for error in errors)


def test_new_launcher_referencing_non_formal_config_is_allowed(tmp_path: Path) -> None:
    """A path-contract reference is campaign-scoped but carries no execution authority."""
    root, _ = make_root(tmp_path)
    (root / "configs" / "experiment").mkdir(parents=True)
    contract_path = root / "configs" / "experiment" / "h-new-path-contract-v1.json"
    contract_path.write_text(
        json.dumps({"schema": "openevo.experiment-h-new-path-contract.v1", "paths": {}}),
        encoding="utf-8",
    )
    launcher = root / "scripts" / "openevo_webshop" / "launch_h_new_host.sh"
    launcher.write_text(
        '#!/bin/sh\nexec "$(git rev-parse --show-toplevel)/scripts/oe" host '
        '--path-contract configs/experiment/h-new-path-contract-v1.json -- true\n',
        encoding="utf-8",
    )
    errors, _ = contract.check(root)
    assert errors == []


def test_temporary_active_campaign_exception_is_explicit(tmp_path: Path) -> None:
    root, _ = make_root(tmp_path)
    baseline_path = root / "configs" / "platform" / "campaign-launcher-legacy-baseline.v1.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["temporary_path_exceptions"] = [{"path": "scripts/openevo_webshop/launch_h140_r2_active.py"}]
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    launcher = root / "scripts" / "openevo_webshop" / "launch_h140_r2_active.py"
    launcher.write_text('cmd = ["docker", "run", "/workspace"]\n', encoding="utf-8")
    errors, notes = contract.check(root)
    assert errors == []
    assert notes == ["PLATFORM_LAUNCHER temporary active-campaign exception: scripts/openevo_webshop/launch_h140_r2_active.py"]


def test_new_launcher_cannot_call_qualification_directly(tmp_path: Path) -> None:
    root, _ = make_root(tmp_path)
    launcher = root / "scripts" / "openevo_webshop" / "launch_h141_direct_qual.py"
    launcher.write_text(
        "from scripts.openevo_webshop import runtime_qualification\n"
        "from scripts.openevo_webshop import runtime_live_gate\n"
        "from scripts.openevo_webshop import execution_readiness\n"
        "runtime_qualification.qualify('--all')\n"
        "runtime_live_gate.run_gate()\n"
        "execution_readiness.check()\n",
        encoding="utf-8",
    )
    errors, _ = contract.check(root)
    assert any("must delegate to scripts/oe" in error for error in errors)
    assert any("direct runtime qualification" in error for error in errors)
    assert any("direct runtime live gate" in error for error in errors)
    assert any("direct execution readiness bypass" in error for error in errors)


def test_new_campaign_launcher_must_use_run_campaign(tmp_path: Path) -> None:
    root, _ = make_root(tmp_path)
    launcher = root / "scripts" / "openevo_webshop" / "launch_h141_legacy_run.sh"
    launcher.write_text(
        '#!/bin/sh\n'
        'exec "$(git rev-parse --show-toplevel)/scripts/oe" run --release configs/activations/h1.json --gpus auto:1\n',
        encoding="utf-8",
    )
    errors, _ = contract.check(root)
    assert any("campaign launcher must delegate through scripts/oe run-campaign" in error for error in errors)


def test_thin_run_campaign_delegate_is_allowed(tmp_path: Path) -> None:
    root, _ = make_root(tmp_path)
    launcher = root / "scripts" / "openevo_webshop" / "launch_h141_run_campaign.sh"
    launcher.write_text(
        '#!/bin/sh\n'
        'exec "$(git rev-parse --show-toplevel)/scripts/oe" run-campaign --release configs/activations/h1.json --gpus auto:1\n',
        encoding="utf-8",
    )
    errors, _ = contract.check(root)
    assert errors == []


def test_exact_imported_execution_snapshot_is_grandfathered_but_not_executable(tmp_path: Path) -> None:
    root, _ = make_root(tmp_path)
    launcher = root / "scripts" / "openevo_webshop" / "launch_imported.sh"
    launcher.write_text('#!/bin/sh\ndocker run imported-image\n', encoding='utf-8')
    baseline_path = root / "configs" / "platform" / "campaign-launcher-legacy-baseline.v1.json"
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    baseline['imported_execution_snapshots'] = [{
        'path':'scripts/openevo_webshop/launch_imported.sh',
        'blob_sha':git_blob(root, launcher),
        'source_commit':'a'*40,
        'execution_authority':False,
        'role':'immutable-imported-execution-snapshot',
    }]
    baseline_path.write_text(json.dumps(baseline), encoding='utf-8')
    errors, notes = contract.check(root)
    assert errors == []
    assert notes == ['PLATFORM_LAUNCHER immutable imported execution snapshot: scripts/openevo_webshop/launch_imported.sh']
    launcher.write_text('#!/bin/sh\ndocker run changed-image\n', encoding='utf-8')
    errors, _ = contract.check(root)
    assert any('must delegate to scripts/oe' in error for error in errors)


def test_imported_execution_snapshot_cannot_claim_execution_authority(tmp_path: Path) -> None:
    root, _ = make_root(tmp_path)
    launcher = root / "scripts" / "openevo_webshop" / "launch_imported.sh"
    launcher.write_text('#!/bin/sh\ndocker run imported-image\n', encoding='utf-8')
    baseline_path = root / "configs" / "platform" / "campaign-launcher-legacy-baseline.v1.json"
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    baseline['imported_execution_snapshots'] = [{
        'path':'scripts/openevo_webshop/launch_imported.sh','blob_sha':git_blob(root, launcher),
        'source_commit':'a'*40,'execution_authority':True,'role':'immutable-imported-execution-snapshot',
    }]
    baseline_path.write_text(json.dumps(baseline), encoding='utf-8')
    errors, _ = contract.check(root)
    assert 'PLATFORM_LAUNCHER malformed imported execution snapshot row' in errors
