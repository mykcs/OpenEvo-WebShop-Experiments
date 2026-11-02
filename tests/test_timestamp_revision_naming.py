from __future__ import annotations

import hashlib
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_timestamp_revision_naming.py"
NAMING = runpy.run_path(str(SCRIPT))
FROZEN_INPUTS = NAMING["FROZEN_INPUT_COMPATIBILITY"]
SNAPSHOT_ROOT = NAMING["FROZEN_INPUT_SNAPSHOT_ROOT"]


def sh(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=root, text=True, capture_output=True, check=False)


def repo(tmp_path: Path) -> Path:
    sh(tmp_path, "git", "init", "-q")
    sh(tmp_path, "git", "config", "user.email", "test@example.invalid")
    sh(tmp_path, "git", "config", "user.name", "test")
    (tmp_path / "seed.txt").write_text("seed\n")
    sh(tmp_path, "git", "add", ".")
    sh(tmp_path, "git", "commit", "-qm", "base")
    return tmp_path


def run(root: Path) -> subprocess.CompletedProcess[str]:
    return sh(root, sys.executable, str(SCRIPT), "--base-ref", "HEAD^")


def run_against(root: Path, base_ref: str) -> subprocess.CompletedProcess[str]:
    return sh(root, sys.executable, str(SCRIPT), "--base-ref", base_ref)


def commit_base(root: Path, path: str, text: str) -> None:
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    sh(root, "git", "add", ".")
    sh(root, "git", "commit", "-qm", "add legacy")


def test_modified_legacy_filename_is_not_reintroduced(tmp_path: Path) -> None:
    root = repo(tmp_path)
    commit_base(root, "configs/example-v1.json", '{"schema":"legacy-v1","note":"old"}\n')  # legacy-revision-compat
    (root / "configs/example-v1.json").write_text('{"schema":"legacy-v1","note":"updated"}\n')  # legacy-revision-compat
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "modify")
    assert run(root).returncode == 0


def test_new_ordinal_filename_still_fails(tmp_path: Path) -> None:
    root = repo(tmp_path)
    (root / "new-v1.json").write_text("{}\n")  # legacy-revision-compat
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad")
    result = run(root)
    assert result.returncode != 0 and "filename: new-v1.json" in result.stderr  # legacy-revision-compat


def test_net_new_ordinal_content_still_fails(tmp_path: Path) -> None:
    root = repo(tmp_path)
    commit_base(root, "note.txt", "clean\n")
    (root / "note.txt").write_text("clean\nnew protocol v2\n")  # legacy-revision-compat
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad content")
    result = run(root)
    assert result.returncode != 0 and "introduced v2 x1" in result.stderr  # legacy-revision-compat


def test_reformatted_existing_ordinal_content_does_not_false_positive(tmp_path: Path) -> None:
    root = repo(tmp_path)
    commit_base(root, "legacy.json", '{"a":"v1","b":"v1"}\n')  # legacy-revision-compat
    (root / "legacy.json").write_text('{\n  "b": "v1",\n  "a": "v1"\n}\n')  # legacy-revision-compat
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "reformat")
    assert run(root).returncode == 0


def test_hash_named_authority_snapshot_preserves_literal_historical_bytes(tmp_path: Path) -> None:
    root = repo(tmp_path)
    body = b'{"schema":"historical-v1"}\n'  # legacy-revision-compat
    digest = hashlib.sha256(body).hexdigest()
    p = root / f"configs/experiment/campaigns/202609030400/authority-snapshots/{digest}.json"
    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(body)
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "snapshot")
    assert run(root).returncode == 0


def test_fake_hash_named_snapshot_does_not_bypass_guard(tmp_path: Path) -> None:
    root = repo(tmp_path)
    p = root / ("configs/experiment/campaigns/202609030400/authority-snapshots/" + "0" * 64 + ".json")
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{"canonical_experiment_id":"study-v3"}\n')  # legacy-revision-compat
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "fake snapshot")
    result = run(root)
    assert result.returncode != 0 and "introduced v3 x1" in result.stderr  # legacy-revision-compat


def test_diverged_base_ref_compares_legacy_tokens_to_merge_base(tmp_path: Path) -> None:
    root = repo(tmp_path)
    sh(root, "git", "branch", "-M", "main")
    commit_base(root, "legacy.txt", "protocol v1\n")  # legacy-revision-compat

    sh(root, "git", "checkout", "-qb", "feature")
    (root / "legacy.txt").write_text("protocol v1\nfeature note\n")  # legacy-revision-compat
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "feature edit")

    sh(root, "git", "checkout", "-q", "main")
    (root / "legacy.txt").write_text("timestamped protocol\n")
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "main removes legacy token")

    sh(root, "git", "checkout", "-q", "feature")
    result = run_against(root, "main")
    assert result.returncode == 0, result.stderr


def test_diverged_base_ref_still_rejects_token_added_by_feature(tmp_path: Path) -> None:
    root = repo(tmp_path)
    sh(root, "git", "branch", "-M", "main")
    commit_base(root, "legacy.txt", "protocol v1\n")  # legacy-revision-compat

    sh(root, "git", "checkout", "-qb", "feature")
    (root / "legacy.txt").write_text("protocol v1\nnew protocol v2\n")  # legacy-revision-compat
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "feature adds ordinal token")

    sh(root, "git", "checkout", "-q", "main")
    (root / "legacy.txt").write_text("timestamped protocol\n")
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "main removes legacy token")

    sh(root, "git", "checkout", "-q", "feature")
    result = run_against(root, "main")
    assert result.returncode != 0 and "introduced v2 x1" in result.stderr  # legacy-revision-compat


def test_default_base_ref_uses_local_main_merge_base_for_feature(tmp_path: Path) -> None:
    root = repo(tmp_path)
    sh(root, "git", "branch", "-M", "main")
    sh(root, "git", "checkout", "-qb", "feature")
    (root / "new-v4.json").write_text("{}\n")  # legacy-revision-compat
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "feature")
    result = sh(root, sys.executable, str(SCRIPT))
    assert result.returncode != 0 and "filename: new-v4.json" in result.stderr  # legacy-revision-compat


def test_default_base_ref_checks_parent_when_head_is_main(tmp_path: Path) -> None:
    root = repo(tmp_path)
    sh(root, "git", "branch", "-M", "main")
    (root / "new-v5.json").write_text("{}\n")  # legacy-revision-compat
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "main change")
    result = sh(root, sys.executable, str(SCRIPT))
    assert result.returncode != 0 and "filename: new-v5.json" in result.stderr  # legacy-revision-compat


def test_schema_format_version_is_not_experiment_revision(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "configs/experiment/activations/202609301200-authority.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema":"openevo.authority.v1", "canonical_experiment_id":"202609301200-study"}\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "schema format")
    assert run(root).returncode == 0


def test_experiment_identity_field_still_rejects_ordinal_revision(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "configs/experiment/activations/202609301200-authority.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema":"openevo.authority.202609301200", "canonical_experiment_id":"study-v2"}\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad identity")
    result = run(root)
    assert result.returncode != 0 and "introduced v2 x1" in result.stderr


def test_treatment_identity_field_remains_enforced(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "configs/experiment/designs/202609301200-design.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"treatment_mode":"scalar-gate-v2"}\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad treatment identity")
    result = run(root)
    assert result.returncode != 0 and "introduced v2 x1" in result.stderr


def test_exact_sealed_e1_treatment_identity_is_preserved(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "configs/experiment/activations/202609301200-authority.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"treatment_mode":"MAGNITUDE_SCALAR_GATE_V1"}\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "sealed treatment identity")
    assert run(root).returncode == 0


def test_source_schema_api_versions_do_not_mask_experiment_identity(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "scripts/new_api_v2.py"
    path.parent.mkdir(parents=True)
    path.write_text('SCHEMA = "openevo.payload.v1"\nEXPERIMENT_ID = "study-v3"\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad source identity")
    result = run(root)
    assert result.returncode != 0 and "introduced v3 x1" in result.stderr


def test_timestamped_experiment_path_rejects_ordinal_revision(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "configs/experiment/activations/authority-v1.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"canonical_experiment_id":"202609301200-study"}\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad authority path")
    result = run(root)
    assert result.returncode != 0 and "filename: configs/experiment/activations/authority-v1.json" in result.stderr


def test_versioned_implementation_module_name_is_not_experiment_identity(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "scripts/openevo_webshop/carrier_contract_v2_identity.py"
    path.parent.mkdir(parents=True)
    path.write_text('SCHEMA = "openevo.carrier.v2"\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "versioned module")
    assert run(root).returncode == 0


def test_manifest_and_output_identity_fields_still_reject_versions(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "configs/experiment/manifests/202609301200-manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"manifest_id":"manifest-v2", "output_root":"runs/output-v1"}\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad manifest identity")
    result = run(root)
    assert result.returncode != 0 and "introduced v1" in result.stderr and "introduced v2" in result.stderr


def test_python_identity_keyword_is_checked(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "scripts/identity.py"
    path.parent.mkdir(parents=True)
    path.write_text('activate(canonical_id="study-v2")\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad keyword identity")
    result = run(root)
    assert result.returncode != 0 and "introduced v2 x1" in result.stderr


def test_experiment_named_python_module_is_not_exempt(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "scripts/experiment-v1.py"
    path.parent.mkdir(parents=True)
    path.write_text('pass\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad experiment path")
    result = run(root)
    assert result.returncode != 0 and "filename: scripts/experiment-v1.py" in result.stderr


def test_patch_rng_format_namespace_is_not_experiment_revision(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "patches/seed-consumer-202610031454.patch"
    path.parent.mkdir(parents=True)
    path.write_text("+" + NAMING["PATCH_RNG_FORMAT_ASSIGNMENT"] + "\n")
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "RNG format namespace")
    result = run(root)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("suffix", [
    '\n+canonical_experiment_id = "study-v2"',
    '; canonical_experiment_id = "study-v2"',
])
def test_patch_rng_format_exception_does_not_hide_identity(tmp_path: Path, suffix: str) -> None:
    root = repo(tmp_path)
    path = root / "patches/seed-consumer-202610031454.patch"
    path.parent.mkdir(parents=True)
    path.write_text("+" + NAMING["PATCH_RNG_FORMAT_ASSIGNMENT"] + suffix + "\n")
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad identity in patch")
    result = run(root)
    assert result.returncode != 0 and "introduced v2" in result.stderr


def test_patch_rng_format_exception_does_not_exempt_ordinal_filename(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "patches/seed-consumer-v1.patch"  # legacy-revision-compat: rejection fixture
    path.parent.mkdir(parents=True)
    path.write_text("+" + NAMING["PATCH_RNG_FORMAT_ASSIGNMENT"] + "\n")
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "bad patch filename")
    result = run(root)
    assert result.returncode != 0 and "filename: patches/seed-consumer-v1.patch" in result.stderr


def test_arbitrary_patch_format_label_is_not_exempt(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = root / "patches/seed-consumer-202610031454.patch"
    path.parent.mkdir(parents=True)
    path.write_text('+material = "unapproved-experiment-v2"\n')
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "unregistered patch label")
    result = run(root)
    assert result.returncode != 0 and "introduced v2" in result.stderr


def frozen_input(root: Path, path: str) -> tuple[Path, Path]:
    body = (SCRIPT.parents[1] / path).read_bytes()
    digest = FROZEN_INPUTS[path]
    assert hashlib.sha256(body).hexdigest() == digest
    original = root / path
    snapshot = root / SNAPSHOT_ROOT / f"{digest}.json"
    for target in (original, snapshot):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    return original, snapshot


@pytest.mark.parametrize("path", sorted(FROZEN_INPUTS))
def test_exact_frozen_q06_input_requires_identical_snapshot(tmp_path: Path, path: str) -> None:
    root = repo(tmp_path)
    frozen_input(root, path)
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "frozen history")
    result = run(root)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", sorted(FROZEN_INPUTS))
@pytest.mark.parametrize("mutated", ("original", "snapshot"))
def test_frozen_q06_byte_mutation_fails_on_existing_path(tmp_path: Path, path: str, mutated: str) -> None:
    root = repo(tmp_path)
    original, snapshot = frozen_input(root, path)
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "frozen history")
    target = original if mutated == "original" else snapshot
    target.write_bytes(target.read_bytes() + b" ")
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "mutation")
    result = run(root)
    assert result.returncode != 0 and f"frozen input: {path}:" in result.stderr


def test_known_digest_does_not_allow_new_ordinal_sibling(tmp_path: Path) -> None:
    root = repo(tmp_path)
    known = sorted(FROZEN_INPUTS)[0]
    original, _ = frozen_input(root, known)
    sibling = root / "configs/experiment/advisor-questions/q06-s2-bounded-r120-prospective-contract.v3.json"
    sibling.write_bytes(original.read_bytes())
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "new ordinal sibling")
    result = run(root)
    assert result.returncode != 0 and f"filename: {sibling.relative_to(root)}" in result.stderr


def test_frozen_q06_alias_with_unverified_snapshot_fails(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = sorted(FROZEN_INPUTS)[0]
    _, snapshot = frozen_input(root, path)
    snapshot.write_bytes(b"{}\n")
    sh(root, "git", "add", "."); sh(root, "git", "commit", "-qm", "unverified snapshot")
    result = run(root)
    assert result.returncode != 0 and f"frozen input: {path}:" in result.stderr
