from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/huggingface-asset-classification-20260922/PRIVATE_WORKSPACE_SOURCE_BINDING_202609221024.json"
REGISTRY = ROOT / "docs/operations/publication/HUGGINGFACE_ASSET_REGISTRY.json"


def _evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def _by_name() -> dict[str, dict]:
    return {Path(row["path"]).name: row for row in _evidence()["items"]}


def test_workspace_source_binding_covers_exact_prior_group() -> None:
    data = _evidence()
    assert data["status"] == "PASS_13_OF_13_ARCHIVES_BOUND_NO_PROVIDER_MUTATION"
    assert data["prior_group_bytes"] == 741_917_593
    assert len(data["items"]) == 13
    assert sum(row["bytes"] for row in data["items"]) == 741_917_593
    assert data["result_by_retention_intent"] == {
        "KEEP_EXACT": {"count": 8, "bytes": 440_004_398},
        "KEEP_RECONSTRUCTIBLE": {"count": 1, "bytes": 64_743_801},
        "RETIREMENT_CANDIDATE": {"count": 4, "bytes": 237_169_394},
    }
    assert data["provider_mutation_performed"] is False
    assert data["deletion_authorized"] is False


def test_stage1_source_named_archives_remain_exact_scientific_evidence() -> None:
    rows = _by_name()
    for name in (
        "qwen25-3b-stage1-202609030400-source-canonical-20260906.tar.gz",
        "qwen3-1p7b-stage1-202609030400-source-canonical-20260906.tar.gz",
        "openevo-ceiling1-predecessor-analysis-20260830-source-cold-20260906.tar.gz",
    ):
        assert rows[name]["retention_intent"] == "KEEP_EXACT"
        assert "scientific" in rows[name]["reason"].lower() or "stage2-readiness" in rows[name]["reason"].lower()


def test_runtime_seed_source_retirement_requires_exact_git_archive_equivalence() -> None:
    row = _by_name()["runtime-seed-source-9996427.tar.gz"]
    binding = row["replacement_binding"]
    assert row["retention_intent"] == "RETIREMENT_CANDIDATE"
    assert binding["commit"] == "9996427d3ae7c71b3cc8ac82940efe764a8a2485"
    assert binding["git_archive_entries_exact"] == 1566
    assert binding["diff_entries"] == 0
    assert binding["only_source_entries"] == 0
    assert binding["only_commit_entries"] == 0


def test_runtime_view_retirement_is_git_bound_and_only_has_rebuildable_extras() -> None:
    row = _by_name()["runtime-views.tar.gz"]
    binding = row["replacement_binding"]
    assert row["retention_intent"] == "RETIREMENT_CANDIDATE"
    assert binding["commit"] == "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c"
    assert binding["tracked_entries_exact"] == 1491
    assert binding["tracked_diff_entries"] == 0
    assert binding["ignored_cache_entries"] == 33
    assert binding["untracked_nonbyte_item"].endswith("-> /webshop-data")
    assert binding["git_pack_sha256_same_as_replacement"] == "3e742d2b54ce8ddc19420c3ef6e903f0f66bcd325ad8d74f8c6163bd9675be7e"


def test_jre_binding_uses_ubuntu_package_not_unrelated_temurin_runtime() -> None:
    row = _by_name()["runtime-jre.tar.gz"]
    binding = row["reconstruction_binding"]
    assert row["retention_intent"] == "KEEP_RECONSTRUCTIBLE"
    assert binding["distribution"] == "Ubuntu 22.04 (jammy)"
    assert binding["version"] == "11.0.31+11-1ubuntu1~22.04.2"
    assert "openjdk-11-jre" in binding["binary_packages"]


def test_registry_routes_current_effective_totals_to_successor_review() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    state = registry["private_heavy_payload_decomposition_202609220158"]
    assert state["effective_current_tree_logical_bytes"] == 120_667_237_575
    assert state["effective_current_by_retention_intent"] == {
        "KEEP_EXACT": {"bytes": 62_272_564_866},
        "KEEP_RECONSTRUCTIBLE": {"bytes": 46_994_469_425},
        "RETIREMENT_CANDIDATE": {"bytes": 11_400_203_284},
    }
    workspace = next(x for x in state["successor_reviews"] if x["review_id"] == "workspace-source-binding-202609221024")
    assert workspace["provider_mutation_performed"] is False
    assert workspace["deletion_authorized"] is False
    seed_remote = next(x for x in state["successor_reviews"] if x["review_id"] == "runtime-seed-remote-binding-202609221440")
    assert seed_remote["transition_bytes"] == 169_815_441
    assert seed_remote["retention_transition"] == "KEEP_EXACT -> RETIREMENT_CANDIDATE"
    assert seed_remote["provider_mutation_performed"] is False
    assert seed_remote["deletion_authorized"] is False
