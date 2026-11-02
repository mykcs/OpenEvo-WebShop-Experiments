from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/huggingface-asset-classification-20260922/PRIVATE_RUNTIME_SEED_REMOTE_BINDING_202609221440.json"
REGISTRY = ROOT / "docs/operations/publication/HUGGINGFACE_ASSET_REGISTRY.json"


def _data() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_runtime_seed_remote_binding_replaces_only_prior_durable_authority_blocker() -> None:
    data = _data()
    assert data["status"] == "PASS_INDEPENDENT_REMOTE_GIT_AUTHORITY_REPLACES_LOCAL_BUNDLE_HOLD"
    assert data["source_archive"]["bytes"] == 169_815_441
    assert data["source_archive"]["sha256"] == "022ca715b6f9990ef8d6d5c2379c5f7b94f06a1785f1e26352e12b43e4bef7ba"
    assert data["prior_review"]["prior_retention_intent"] == "KEEP_EXACT"
    assert data["prior_review"]["proven_local_git_state"]["head_worktree_status"] == "CLEAN"
    decision = data["decision"]
    assert decision["retention_intent_before"] == "KEEP_EXACT"
    assert decision["retention_intent_after"] == "RETIREMENT_CANDIDATE"
    assert decision["provider_mutation_performed"] is False
    assert decision["deletion_authorized"] is False


def test_runtime_seed_remote_binding_has_all_three_content_addressed_git_identities() -> None:
    remotes = {(row["repository"], row["branch"]): row for row in _data()["independent_remote_git_authority"]}
    assert remotes[("mykcs/seed3090", "codex/webshop-seed-contract")]["branch_head"] == "9996427d3ae7c71b3cc8ac82940efe764a8a2485"
    assert remotes[("mykcs/seed3090", "codex/webshop-seed-contract")]["tree"] == "69129c19658ab1b48fea597f4244c81d52bcbc00"
    assert remotes[("mykcs/seed3090", "main")]["branch_head"] == "6b3285e7ac021f1b69d12223f229d8c9b9a1ed3a"
    assert remotes[("mykcs/seed3090", "main")]["tree"] == "1dc0231b9de5326e802dc0119a5d53cac533aff3"
    assert remotes[("jinyangwu/SEED", "main")]["branch_head"] == "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c"
    assert remotes[("jinyangwu/SEED", "main")]["tree"] == "3f91f467473d7134d17692b050003f872686da04"
    assert all(row["live_branch_ref_check"] == "PASS" for row in remotes.values())
    assert all(row["git_commit_object_check"] == "PASS" for row in remotes.values())


def test_dependent_archives_no_longer_require_private_hf_git_tar_as_replacement() -> None:
    rows = {Path(row["path"]).name: row for row in _data()["dependent_archive_rebinding"]}
    assert rows["runtime-seed-source-9996427.tar.gz"]["current_replacement"].startswith("GitHub mykcs/seed3090@9996427d")
    assert rows["runtime-views.tar.gz"]["current_replacement"].startswith("GitHub jinyangwu/SEED@2cf2fad")


def test_registry_current_effective_heavy_totals_include_runtime_seed_successor() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    state = registry["private_heavy_payload_decomposition_202609220158"]
    totals = state["effective_current_by_retention_intent"]
    assert totals == {
        "KEEP_EXACT": {"bytes": 62_272_564_866},
        "KEEP_RECONSTRUCTIBLE": {"bytes": 46_994_469_425},
        "RETIREMENT_CANDIDATE": {"bytes": 11_400_203_284},
    }
    assert sum(row["bytes"] for row in totals.values()) == state["effective_current_tree_logical_bytes"] == 120_667_237_575
    current = registry["private_runtime_seed_remote_binding_202609221440"]
    assert current["retention_intent_after"] == "RETIREMENT_CANDIDATE"
    assert current["provider_mutation_performed"] is False
    assert current["deletion_authorized"] is False
