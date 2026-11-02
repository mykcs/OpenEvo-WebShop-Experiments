"""CPU-only Passport Artifact DAG/cache regressions; never runs formal science."""
from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path

import pytest

from scripts.openevo_webshop import experiment_artifact_dag as dag


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(root: Path, rel: str) -> dict:
    path = root / rel
    return {
        "kind": "repository_file", "repository": None, "revision": None,
        "path": rel, "sha256": _sha(path), "evidence_path": None, "reason": None,
    }


def _artifact(root: Path, rel: str, role: str, *, derived: list[str] | None = None) -> dict:
    path = root / rel
    return {
        "artifact_type": "test-file", "digest_algorithm": "sha256",
        "digest": _sha(path), "size_bytes": path.stat().st_size, "logical_role": role,
        "locations": [_ref(root, rel)], "derived_from_digests": list(derived or []),
    }


def _fixture_files(root: Path) -> None:
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures/code.py").write_text("print('stable')\n")
    (root / "fixtures/config.json").write_text('{"mode":"deterministic"}\n')
    (root / "fixtures/input.txt").write_text("alpha\n")
    (root / "fixtures/output.txt").write_text("ALPHA\n")
    (root / "fixtures/evidence.txt").write_text("pilot evidence\n")


def _spec(root: Path, *, side_effect: str = "PURE_ENGINEERING") -> dict:
    scientific = side_effect in dag.SCIENTIFIC_NO_AUTO_REUSE
    return {
        "operation_type": "pilot.uppercase-analysis",
        "operation_schema_version": "1",
        "side_effect_class": side_effect,
        "identity_payload": {
            "code_identity": _ref(root, "fixtures/code.py"),
            "config_identity": _ref(root, "fixtures/config.json"),
            "runtime_identity_policy": "NOT_IDENTITY_BEARING",
            "runtime_identity": None,
            "dataset_schedule_identity": _ref(root, "fixtures/config.json") if scientific else None,
            "contract_identities": ([{
                "name": "science-contract",
                "identity": _ref(root, "fixtures/evidence.txt"),
            }] if scientific else []),
            "input_artifacts": [_artifact(root, "fixtures/input.txt", "source-text")],
            "stochastic_identity": {"mode": "deterministic", "seed": 7} if scientific else {"mode": "deterministic"},
        },
    }


def _node(root: Path, *, side_effect: str = "PURE_ENGINEERING",
          auto: bool = True, allowlisted: bool = True) -> dict:
    spec = _spec(root, side_effect=side_effect)
    return {
        "schema": dag.NODE_SCHEMA_ID, "node_kind": "operation",
        "operation_type": spec["operation_type"],
        "operation_schema_version": spec["operation_schema_version"],
        "operation_digest": dag.operation_digest(spec),
        "identity_payload": copy.deepcopy(spec["identity_payload"]),
        "side_effect_class": side_effect, "status": "PASS",
        "outputs": [_artifact(root, "fixtures/output.txt", "analysis-output",
                              derived=[spec["identity_payload"]["input_artifacts"][0]["digest"]])],
        "provenance": {
            "producer_scope": "ENGINEERING_GLOBAL", "experiment_id": None,
            "passport_sha256": None, "evidence_paths": ["fixtures/evidence.txt"],
            "external_views": [],
        },
        "producer_attempt": {
            "attempt_id": "pilot-attempt-1", "source": "PILOT",
            "execution_sha": None, "created_at": "2026-09-14T11:00:00Z",
        },
        "reuse": {
            "mechanical_eligibility": "ELIGIBLE",
            "default_policy": dag.DEFAULT_REUSE_POLICIES[side_effect],
            "operation_allowlisted": allowlisted,
            "automatic_reuse_permitted": auto,
            "reason": "CPU-only deterministic pilot; no scientific side effect.",
        },
        "created_at": "2026-09-14T11:00:00Z",
        "completed_at": "2026-09-14T11:00:01Z",
        "supersedes": ["f" * 64],
    }


def _write_index(root: Path, node: dict) -> dict:
    entry = dag.write_operation_node(root, node)
    index = dag.build_index([entry], generated_at="2026-09-14T11:01:00Z")
    dag.write_index_snapshot(
        root, index,
        snapshot_rel="configs/experiment/passport-artifacts/index/experiment-passport-artifact-index-test.json",
    )
    return index


def test_digest_is_canonical_and_input_order_independent(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    spec = _spec(tmp_path)
    spec["identity_payload"]["input_artifacts"].append(
        _artifact(tmp_path, "fixtures/output.txt", "second-input")
    )
    before = dag.operation_digest(spec)
    spec["identity_payload"]["input_artifacts"].reverse()
    assert dag.operation_digest(spec) == before


def test_nonidentity_observations_do_not_change_digest(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    node = _node(tmp_path)
    before = dag.operation_digest(node)
    node["status"] = "STALE"
    node["producer_attempt"]["attempt_id"] = "another-observation"
    node["reuse"]["reason"] = "policy note changed"
    node["supersedes"] = ["e" * 64]
    assert dag.operation_digest(node) == before


def test_side_effect_class_is_identity_bearing(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    engineering = _spec(tmp_path, side_effect="PURE_ENGINEERING")
    deterministic = _spec(tmp_path, side_effect="DETERMINISTIC_COMPUTATION")
    assert dag.operation_digest(engineering) != dag.operation_digest(deterministic)


def test_formal_rollout_identity_requires_dataset_and_science_contract(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    spec = _spec(tmp_path, side_effect="FORMAL_ROLLOUT")
    spec["identity_payload"]["contract_identities"] = []
    with pytest.raises(dag.ArtifactDagError, match="science/runtime contract"):
        dag.operation_digest(spec)
    spec = _spec(tmp_path, side_effect="FORMAL_ROLLOUT")
    spec["identity_payload"]["dataset_schedule_identity"] = None
    with pytest.raises(dag.ArtifactDagError, match="dataset_schedule_identity"):
        dag.operation_digest(spec)


@pytest.mark.parametrize("side_effect", sorted(dag.SCIENTIFIC_NO_AUTO_REUSE))
def test_scientific_classes_can_never_auto_reuse(tmp_path: Path, side_effect: str) -> None:
    _fixture_files(tmp_path)
    node = _node(tmp_path, side_effect=side_effect, auto=True)
    with pytest.raises(dag.ArtifactDagError, match="never be automatically reused"):
        dag.validate_operation_node(node, repo_root=tmp_path)


def test_deterministic_computation_has_no_blanket_auto_reuse(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    node = _node(tmp_path, side_effect="DETERMINISTIC_COMPUTATION", auto=True)
    with pytest.raises(dag.ArtifactDagError, match="no blanket automatic reuse"):
        dag.validate_operation_node(node, repo_root=tmp_path)


def test_pure_engineering_auto_requires_allowlist(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    node = _node(tmp_path, auto=True, allowlisted=False)
    with pytest.raises(dag.ArtifactDagError, match="operation_allowlisted"):
        dag.validate_operation_node(node, repo_root=tmp_path)


def test_exact_hit_changed_input_and_fail_closed_provenance(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    node = _node(tmp_path)
    index = _write_index(tmp_path, node)

    hit = dag.query_operation(_spec(tmp_path), repo_root=tmp_path, index=index)
    assert hit["status"] == "REUSE_CANDIDATE"
    assert hit["classification"] == "EXACT_CACHE_HIT"
    assert hit["reuse"]["automatic_reuse_permitted"] is True

    input_path = tmp_path / "fixtures/input.txt"
    original_input = input_path.read_text()
    input_path.write_text("beta\n")
    changed = _spec(tmp_path)
    assert dag.query_operation(changed, repo_root=tmp_path, index=index)["status"] == "MISS"
    input_path.write_text(original_input)

    code_path = tmp_path / "fixtures/code.py"
    original_code = code_path.read_text()
    code_path.write_text("print('tampered')\n")
    mismatch = dag.query_operation(_spec_from_node(node), repo_root=tmp_path, index=index)
    assert mismatch["status"] == "PROVENANCE_MISMATCH"
    assert mismatch["query_verification"]["status"] == "FAIL"
    code_path.write_text(original_code)

    output_path = tmp_path / "fixtures/output.txt"
    output_path.write_text("BROKEN\n")
    stale = dag.query_operation(_spec_from_node(node), repo_root=tmp_path, index=index)
    assert stale["status"] == "STALE_OR_INVALID_OUTPUT"


def _spec_from_node(node: dict) -> dict:
    return {
        "operation_type": node["operation_type"],
        "operation_schema_version": node["operation_schema_version"],
        "side_effect_class": node["side_effect_class"],
        "identity_payload": copy.deepcopy(node["identity_payload"]),
    }


def test_compatible_historical_candidate_requires_explicit_predicate(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    node = _node(tmp_path)
    index = _write_index(tmp_path, node)
    (tmp_path / "fixtures/config2.json").write_text('{"mode":"deterministic-mutated"}\n')
    query = _spec(tmp_path)
    query["identity_payload"]["config_identity"] = _ref(tmp_path, "fixtures/config2.json")

    assert dag.query_operation(query, repo_root=tmp_path, index=index)["status"] == "MISS"
    result = dag.query_operation(
        query, repo_root=tmp_path, index=index,
        compatibility_predicate=lambda wanted, old: wanted["operation_type"] == old["operation_type"],
    )
    assert result["status"] == "COMPATIBLE_HISTORICAL_CANDIDATE"
    assert result["automatic_reuse_permitted"] is False
    assert len(result["candidates"]) == 1


def test_edges_cover_required_relations_and_supersedes_is_not_identity(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    node = _node(tmp_path)
    before = node["operation_digest"]
    edges = dag.build_edges(node, repo_root=tmp_path)
    assert {item["edge_type"] for item in edges} == {
        "consumed_by", "produced_by", "derived_from", "supersedes"
    }
    node["supersedes"] = ["e" * 64]
    assert dag.operation_digest(node) == before


def test_same_operation_identity_can_record_multiple_producer_attempts(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    first = _node(tmp_path)
    second = copy.deepcopy(first)
    second["producer_attempt"]["attempt_id"] = "pilot-attempt-2"
    second["producer_attempt"]["created_at"] = "2026-09-14T11:00:02Z"
    second["completed_at"] = "2026-09-14T11:00:03Z"
    first_entry = dag.write_operation_node(tmp_path, first)
    second_entry = dag.write_operation_node(tmp_path, second)
    assert first_entry["operation_digest"] == second_entry["operation_digest"]
    assert first_entry["path"] != second_entry["path"]
    index = dag.build_index([first_entry, second_entry], generated_at="2026-09-14T11:01:00Z")
    assert dag.validate_index(index, repo_root=tmp_path) == []
    result = dag.query_operation(_spec(tmp_path), repo_root=tmp_path, index=index)
    assert result["status"] == "REUSE_CANDIDATE"
    assert result["candidate_count"] == 2
    assert result["reuse"]["automatic_reuse_permitted"] is False


def test_formal_scientific_node_requires_exact_experiment_producer_provenance(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    node = _node(tmp_path, side_effect="FORMAL_ROLLOUT", auto=False, allowlisted=False)
    with pytest.raises(dag.ArtifactDagError, match="requires EXPERIMENT producer provenance"):
        dag.validate_operation_node(node, repo_root=tmp_path)
    node["provenance"] = {
        "producer_scope": "EXPERIMENT", "experiment_id": "202609141200-pilot-formal",
        "passport_sha256": "a" * 64, "evidence_paths": ["fixtures/evidence.txt"],
        "external_views": [],
    }
    node["producer_attempt"]["source"] = "LIVE"
    node["producer_attempt"]["execution_sha"] = "b" * 40
    dag.validate_operation_node(node, repo_root=tmp_path)


def test_stable_router_is_path_plus_sha_and_roundtrips(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    index = _write_index(tmp_path, _node(tmp_path))
    loaded, router = dag.load_current_index(tmp_path)
    assert loaded == index
    assert set(router) == {"path", "sha256"}
    assert dag.validate_index(loaded, repo_root=tmp_path) == []


def test_index_rejects_noncanonical_operation_node_path(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    entry = dag.write_operation_node(tmp_path, _node(tmp_path))
    bad = copy.deepcopy(entry)
    bad["path"] = "configs/experiment/passport-artifacts/operation-nodes/not-canonical.json"
    index = dag.build_index([bad], generated_at="2026-09-14T11:01:00Z")
    errors = dag.validate_index(index, repo_root=tmp_path)
    assert any("non-canonical operation-node path" in item for item in errors)


def test_experiment_scoped_provenance_resolves_existing_passport_registry() -> None:
    index, _ = dag.load_current_index(dag.ROOT)
    entry = index["entries"][0]
    node = dag._load_indexed_node(dag.ROOT, entry)
    registry, _ = dag.load_current_registry(dag.ROOT)
    registered = dag.resolve_experiment(registry, "202609082022-q17-directapply-no-gdr")
    current_sha = subprocess.check_output(
        ["git", "-C", str(dag.ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    scoped = copy.deepcopy(node)
    scoped["provenance"] = {
        "producer_scope": "EXPERIMENT",
        "experiment_id": registered["experiment_id"],
        "passport_sha256": registered["passport_sha256"],
        "evidence_paths": [],
        "external_views": [],
    }
    scoped["producer_attempt"]["execution_sha"] = current_sha
    verification = dag.verify_node_provenance(scoped, repo_root=dag.ROOT)
    assert verification["identity_status"] == "PASS"
    assert "producer_passport_registry_binding: PASS" in verification["identity_details"]


def test_committed_pilot_index_and_router_validate() -> None:
    index, router = dag.load_current_index(dag.ROOT)
    assert set(router) == {"path", "sha256"}
    assert dag.validate_index(index, repo_root=dag.ROOT) == []
    assert len(index["entries"]) >= 1


def test_malformed_query_missing_code_identity_fails_closed(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    spec = _spec(tmp_path)
    spec["identity_payload"]["code_identity"] = None
    with pytest.raises(dag.ArtifactDagError, match="code_identity is required"):
        dag.operation_digest(spec)


def test_digest_rejects_nonfinite_stochastic_identity(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    spec = _spec(tmp_path)
    spec["identity_payload"]["stochastic_identity"] = {"bad": float("nan")}
    with pytest.raises(dag.PassportError, match="Non-finite"):
        dag.operation_digest(spec)


def _runtime_spec(root: Path) -> dict:
    spec = _spec(root)
    spec["schema"] = dag.RUNTIME_NODE_SCHEMA_ID
    spec["identity_payload"]["execution_generation_id"] = "a" * 64
    return spec


def _runtime_node(root: Path) -> dict:
    spec = _runtime_spec(root)
    node = _node(root)
    node["schema"] = dag.RUNTIME_NODE_SCHEMA_ID
    node["identity_payload"] = copy.deepcopy(spec["identity_payload"])
    node["operation_digest"] = dag.operation_digest(spec)
    return node


def test_runtime_execution_generation_is_identity_bearing(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    spec = _runtime_spec(tmp_path)
    before = dag.operation_digest(spec)
    spec["identity_payload"]["execution_generation_id"] = "b" * 64
    assert dag.operation_digest(spec) != before
    spec["identity_payload"].pop("execution_generation_id")
    with pytest.raises(dag.ArtifactDagError, match="execution_generation_id"):
        dag.operation_digest(spec)


def test_main_passport_schema_remains_unbound_to_execution_generation(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    spec = _spec(tmp_path)
    baseline = dag.operation_digest(spec)
    spec["schema"] = dag.NODE_SCHEMA_ID
    assert dag.operation_digest(spec) == baseline
    spec["identity_payload"]["execution_generation_id"] = "a" * 64
    with pytest.raises(dag.ArtifactDagError, match="cannot claim execution_generation_id"):
        dag.operation_digest(spec)


def test_runtime_node_validates_and_query_returns_generation_and_node_hash(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    node = _runtime_node(tmp_path)
    dag.validate_operation_node(node, repo_root=tmp_path)
    index = _write_index(tmp_path, node)
    result = dag.query_operation(_runtime_spec(tmp_path), repo_root=tmp_path, index=index)
    assert result["status"] == "REUSE_CANDIDATE"
    assert result["execution_generation_id"] == "a" * 64
    assert result["node_sha256"] == index["entries"][0]["node_sha256"]


def test_legacy_runtime_schema_remains_readable(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    node = _runtime_node(tmp_path)
    node["schema"] = dag.LEGACY_RUNTIME_NODE_SCHEMA_ID
    node["operation_digest"] = dag.operation_digest(node)
    dag.validate_operation_node(node, repo_root=tmp_path)


def test_pre_runtime_legacy_schema_remains_readable_without_generation(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    node = _node(tmp_path)
    node["schema"] = dag.LEGACY_NODE_SCHEMA_ID
    node["operation_digest"] = dag.operation_digest(node)
    dag.validate_operation_node(node, repo_root=tmp_path)
