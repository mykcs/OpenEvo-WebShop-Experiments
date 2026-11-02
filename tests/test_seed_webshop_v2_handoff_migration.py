from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.openevo_webshop import migrate_seed_webshop_v2_handoff as migration
from scripts.openevo_webshop import successor_handoff as sh


SOURCE_PATHS = (
    "agent_system/environments/env_package/webshop/envs.py",
    "agent_system/environments/env_package/webshop/projection.py",
    "agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/"
    "web_agent_text_env.py",
    "agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/"
    "engine.py",
    "agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/"
    "goal.py",
)
DATA_PATHS = (
    "agent_system/environments/env_package/webshop/webshop/data/"
    "items_shuffle_1000.json",
    "agent_system/environments/env_package/webshop/webshop/data/"
    "items_ins_v2_1000.json",
)


def build_fake_seed_root(
    tmp_path: Path,
) -> tuple[Path, dict[str, str], dict[str, str]]:
    phase_root = tmp_path / "phase-v2"
    seed_root = phase_root / "seed-roots" / "base-P1"
    source_blobs: dict[str, str] = {}
    data_sha256: dict[str, str] = {}

    for index, relative in enumerate(SOURCE_PATHS):
        path = seed_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"PINNED_{index} = True\n", encoding="utf-8")
        source_blobs[relative] = migration.git_blob_sha1(path)

    for index, relative in enumerate(DATA_PATHS):
        path = seed_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"fixture": index}, sort_keys=True).encode("utf-8")
        path.write_bytes(payload)
        data_sha256[relative] = hashlib.sha256(payload).hexdigest()

    return seed_root, source_blobs, data_sha256


def _migrate_fixture(
    tmp_path: Path,
    *,
    output_name: str = "output",
) -> tuple[Path, dict[str, str], dict[str, str], dict]:
    seed_root, source_blobs, data_sha256 = build_fake_seed_root(tmp_path)
    result = migration.migrate(
        seed_root,
        output_dir=tmp_path / output_name,
        auto_discover_runtime_qualification=False,
        source_blobs=source_blobs,
        data_sha256=data_sha256,
    )
    return seed_root, source_blobs, data_sha256, result


def test_one_command_migration_captures_and_bootstraps_without_execution_authority(
    tmp_path: Path,
) -> None:
    _, _, _, result = _migrate_fixture(tmp_path)

    assert result["status"] == "MIGRATED"
    assert result["runtime_qualification"] is None
    assert result["formal_task_consumption_allowed"] is False
    assert result["gpu_allocation_allowed"] is False

    handoff = json.loads(Path(result["handoff"]).read_text(encoding="utf-8"))
    bootstrap = json.loads(Path(result["bootstrap"]).read_text(encoding="utf-8"))
    assert handoff["schema"] == sh.HANDOFF_SCHEMA
    assert handoff["formal_task_consumption_allowed"] is False
    assert handoff["gpu_allocation_allowed"] is False
    assert bootstrap["status"] == "READY_FOR_SUCCESSOR_PREP"
    assert bootstrap["formal_task_consumption_allowed"] is False
    assert bootstrap["gpu_allocation_allowed"] is False
    assert "seed-source-faithfulness" in bootstrap["reused_roles"]


def test_historical_migration_is_deterministic_for_same_evidence(tmp_path: Path) -> None:
    seed_root, source_blobs, data_sha256 = build_fake_seed_root(tmp_path)

    first = migration.migrate(
        seed_root,
        output_dir=tmp_path / "first",
        auto_discover_runtime_qualification=False,
        source_blobs=source_blobs,
        data_sha256=data_sha256,
    )
    second = migration.migrate(
        seed_root,
        output_dir=tmp_path / "second",
        auto_discover_runtime_qualification=False,
        source_blobs=source_blobs,
        data_sha256=data_sha256,
    )

    assert Path(first["handoff"]).read_bytes() == Path(second["handoff"]).read_bytes()
    assert first["handoff_sha256"] == second["handoff_sha256"]


def test_materialized_seed_root_does_not_need_git_metadata(tmp_path: Path) -> None:
    seed_root, source_blobs, data_sha256 = build_fake_seed_root(tmp_path)
    assert not (seed_root / ".git").exists()

    verification = migration.verify_pinned_seed_root(
        seed_root,
        source_blobs=source_blobs,
        data_sha256=data_sha256,
    )

    assert verification["seed_root"] == str(seed_root.resolve())
    assert len(verification["source_checks"]) == 5
    assert len(verification["data_checks"]) == 2


def test_source_drift_fails_closed_before_capture(tmp_path: Path) -> None:
    seed_root, source_blobs, data_sha256 = build_fake_seed_root(tmp_path)
    changed = seed_root / SOURCE_PATHS[0]
    changed.write_text("PINNED = False\n", encoding="utf-8")

    with pytest.raises(migration.MigrationError, match="Git blob mismatch"):
        migration.migrate(
            seed_root,
            output_dir=tmp_path / "output",
            auto_discover_runtime_qualification=False,
            source_blobs=source_blobs,
            data_sha256=data_sha256,
        )


def test_bound_source_drift_becomes_stale_after_capture(tmp_path: Path) -> None:
    seed_root, _, _, result = _migrate_fixture(tmp_path)
    (seed_root / SOURCE_PATHS[1]).write_text("CHANGED = True\n", encoding="utf-8")

    bootstrap = sh.bootstrap(Path(result["handoff"]))

    assert bootstrap["status"] == "STALE"
    assert any(
        item["role"].endswith(SOURCE_PATHS[1])
        and "content changed" in item["reason"]
        for item in bootstrap["failures"]
    )


def test_data_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    seed_root, source_blobs, data_sha256 = build_fake_seed_root(tmp_path)
    first_data = seed_root / DATA_PATHS[0]
    first_data.write_text('{"fixture":"changed"}\n', encoding="utf-8")

    with pytest.raises(migration.MigrationError, match="SHA-256 mismatch"):
        migration.verify_pinned_seed_root(
            seed_root,
            source_blobs=source_blobs,
            data_sha256=data_sha256,
        )


def test_unrelated_9996427_checkout_cannot_invalidate_handoff(tmp_path: Path) -> None:
    _, _, _, result = _migrate_fixture(tmp_path)

    unrelated = tmp_path / "SEED-9996427" / "REVISION"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("9996427\n", encoding="utf-8")
    unrelated.write_text("changed-unrelated-checkout\n", encoding="utf-8")

    bootstrap = sh.bootstrap(Path(result["handoff"]))
    assert bootstrap["status"] == "READY_FOR_SUCCESSOR_PREP"


def test_runtime_qualification_is_only_a_cache_candidate(tmp_path: Path) -> None:
    seed_root, source_blobs, data_sha256 = build_fake_seed_root(tmp_path)
    phase_root = seed_root.parent.parent
    qualification = (
        phase_root
        / "runtime-seed-git-9996427-v2"
        / "qualification-receipt.json"
    )
    qualification.parent.mkdir(parents=True)
    qualification.write_text(
        json.dumps(
            {
                "schema": migration.RUNTIME_QUAL_SCHEMA,
                "status": "PASS",
                "request": {
                    "runtime_fingerprint_sha256": "sha256:" + "1" * 64,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = migration.migrate(
        seed_root,
        output_dir=tmp_path / "output",
        source_blobs=source_blobs,
        data_sha256=data_sha256,
    )

    assert result["runtime_qualification"] == str(qualification.resolve())
    assert result["runtime_qualification_cache_candidates"] == [
        "runtime-qualification"
    ]
    assert "runtime-qualification" not in result["reused_roles"]
    assert result["formal_task_consumption_allowed"] is False
    assert result["gpu_allocation_allowed"] is False


def test_archived_adapter_task_and_protocol_identities_are_captured(tmp_path: Path) -> None:
    _, _, _, result = _migrate_fixture(tmp_path)
    handoff = json.loads(Path(result["handoff"]).read_text(encoding="utf-8"))
    facts = {fact["role"]: fact for fact in handoff["facts"]}

    assert facts["frozen-adapter-sha256"]["value"] == (
        f"sha256:{migration.ADAPTER_SHA256}"
    )
    assert facts["predecessor-task-manifest-sha256"]["value"] == (
        f"sha256:{migration.TASK_MANIFEST_FILE_SHA256}"
    )
    assert facts["predecessor-selected-panel-digest"]["value"] == (
        f"sha256:{migration.SELECTED_PANEL_DIGEST}"
    )
    assert facts["predecessor-goals-list-digest"]["value"] == (
        f"sha256:{migration.GOALS_LIST_DIGEST}"
    )
    assert facts["primary-v2-protocol-fingerprints"]["value"] == (
        migration.PRIMARY_V2_PROTOCOL_FINGERPRINTS
    )


def test_all_source_and_data_roles_are_required(tmp_path: Path) -> None:
    _, source_blobs, data_sha256, result = _migrate_fixture(tmp_path)
    handoff_path = Path(result["handoff"])
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    missing_role = f"seed-task-identity:{SOURCE_PATHS[0]}"
    handoff["facts"] = [
        fact for fact in handoff["facts"] if fact.get("role") != missing_role
    ]
    tampered = tmp_path / "tampered-handoff.json"
    sh.atomic_json(tampered, handoff)

    bootstrap = sh.bootstrap(
        tampered,
        required_roles=migration.handoff_required_roles(
            source_blobs=source_blobs,
            data_sha256=data_sha256,
        ),
    )

    assert bootstrap["status"] == "STALE"
    assert {"role": missing_role, "reason": "required role absent from handoff"} in (
        bootstrap["failures"]
    )


def test_ambiguous_runtime_receipts_are_not_chosen(tmp_path: Path) -> None:
    seed_root, source_blobs, data_sha256 = build_fake_seed_root(tmp_path)
    phase_root = seed_root.parent.parent
    for name in ("runtime-a-v2", "runtime-b-v2"):
        qualification = phase_root / name / "qualification-receipt.json"
        qualification.parent.mkdir(parents=True)
        qualification.write_text(
            json.dumps(
                {"schema": migration.RUNTIME_QUAL_SCHEMA, "status": "PASS"},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    result = migration.migrate(
        seed_root,
        output_dir=tmp_path / "output",
        source_blobs=source_blobs,
        data_sha256=data_sha256,
    )

    assert result["runtime_qualification"] is None
    assert result["runtime_qualification_cache_candidates"] == []
