"""CPU-only shadow-adoption regressions; no formal rows or GPU work."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.openevo_webshop import experiment_artifact_dag as dag
from scripts.openevo_webshop import passport_accelerated_gdr_shadow as shadow


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(root: Path, rel: str) -> dict:
    path = root / rel
    return {
        "kind": "repository_file",
        "repository": None,
        "revision": None,
        "path": rel,
        "sha256": _sha(path),
        "evidence_path": None,
        "reason": None,
    }


def _artifact(root: Path, rel: str, role: str) -> dict:
    path = root / rel
    return {
        "artifact_type": "test-file",
        "digest_algorithm": "sha256",
        "digest": _sha(path),
        "size_bytes": path.stat().st_size,
        "logical_role": role,
        "locations": [_ref(root, rel)],
        "derived_from_digests": [],
    }


def _topology(topology_id: str, *, gpu_uuid_sha: str = "4" * 64) -> dict:
    common = {
        "schema": shadow.TOPOLOGY_SCHEMA_ID,
        "topology_id": topology_id,
        "gpu_count": 4,
        "arm_partition_policy": "matched-two-arm-static-sharding",
        "gpu_uuid_set_sha256": gpu_uuid_sha,
        "runtime_image_digest": "sha256:" + "5" * 64,
        "model_identity_sha256": "6" * 64,
        "resource_contract_sha256": "7" * 64,
    }
    if topology_id == shadow.PREFERRED_TOPOLOGY:
        common.update({
            "gpus_per_arm": None,
            "workers_per_gpu": 2,
            "total_workers": 8,
            "co_resident": True,
            "baseline_workers": 4,
            "gdr_workers": 4,
        })
    else:
        common.update({
            "gpus_per_arm": 2,
            "workers_per_gpu": 1,
            "total_workers": 4,
            "co_resident": False,
            "baseline_workers": 2,
            "gdr_workers": 2,
        })
    return common


def _write_json(root: Path, rel: str, payload: dict) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _fixture(root: Path) -> None:
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures/code.py").write_text("print('shadow')\n")
    (root / "fixtures/config.json").write_text('{"mode":"shadow"}\n')
    (root / "fixtures/input.txt").write_text("input\n")
    (root / "fixtures/output.txt").write_text("output\n")
    (root / "fixtures/contract.json").write_text('{"contract":"science"}\n')
    (root / "fixtures/schedule.json").write_text('{"schedule":"frozen"}\n')
    (root / "fixtures/runtime.json").write_text('{"runtime":"bound"}\n')
    _write_json(root, "fixtures/topology-a.json", _topology(shadow.PREFERRED_TOPOLOGY))
    _write_json(root, "fixtures/topology-b.json", _topology(shadow.FALLBACK_TOPOLOGY))
    _write_json(root, "fixtures/topology-a-uuid2.json", _topology(shadow.PREFERRED_TOPOLOGY, gpu_uuid_sha="8" * 64))


def _engineering_spec(root: Path, operation_name: str) -> dict:
    return shadow.build_operation_spec(
        operation_name,
        code_identity=_ref(root, "fixtures/code.py"),
        config_identity=_ref(root, "fixtures/config.json"),
        input_artifacts=[_artifact(root, "fixtures/input.txt", "input")],
    )


def _topology_spec(root: Path, rel: str) -> tuple[dict, dict]:
    topology, topology_ref = shadow.load_topology(root, rel)
    spec = shadow.build_operation_spec(
        "worker_resource_topology_qualification",
        code_identity=_ref(root, "fixtures/code.py"),
        config_identity=_ref(root, "fixtures/config.json"),
        input_artifacts=[_artifact(root, "fixtures/input.txt", "qualification-fixture")],
        runtime_identity_policy="BOUND",
        runtime_identity=_ref(root, "fixtures/runtime.json"),
        topology=topology,
        topology_ref=topology_ref,
    )
    return topology, spec


def _node(root: Path, spec: dict) -> dict:
    side_effect = spec["side_effect_class"]
    return {
        "schema": dag.NODE_SCHEMA_ID,
        "node_kind": "operation",
        "operation_type": spec["operation_type"],
        "operation_schema_version": spec["operation_schema_version"],
        "operation_digest": dag.operation_digest(spec),
        "identity_payload": copy.deepcopy(spec["identity_payload"]),
        "side_effect_class": side_effect,
        "status": "PASS",
        "outputs": [_artifact(root, "fixtures/output.txt", "qualification-output")],
        "provenance": {
            "producer_scope": "ENGINEERING_GLOBAL",
            "experiment_id": None,
            "passport_sha256": None,
            "evidence_paths": [],
            "external_views": [],
        },
        "producer_attempt": {
            "attempt_id": "shadow-test-attempt",
            "source": "PILOT",
            "execution_sha": None,
            "created_at": "2026-09-14T13:00:00Z",
        },
        "reuse": {
            "mechanical_eligibility": "ELIGIBLE",
            "default_policy": dag.DEFAULT_REUSE_POLICIES[side_effect],
            "operation_allowlisted": False,
            "automatic_reuse_permitted": False,
            "reason": "CPU-only shadow qualification fixture.",
        },
        "created_at": "2026-09-14T13:00:00Z",
        "completed_at": "2026-09-14T13:00:01Z",
        "supersedes": [],
    }


def _index(root: Path, node: dict) -> dict:
    entry = dag.write_operation_node(root, node)
    return dag.build_index([entry], generated_at="2026-09-14T13:01:00Z")


def test_priority_registry_is_exactly_ten_engineering_operations() -> None:
    assert set(shadow.ENGINEERING_OPERATIONS) == {
        "exact_checkout_resolution",
        "acceleration_successor_qualification",
        "common_start_materialization",
        "model_config_image_hash_verification",
        "carrier_contract_preflight",
        "gdr_mechanism_preflight",
        "worker_resource_topology_qualification",
        "deterministic_benchmark_analysis",
        "artifact_archive",
        "remote_readback",
    }


def test_both_resource_topologies_validate_and_have_distinct_identity(tmp_path: Path) -> None:
    _fixture(tmp_path)
    top_a, spec_a = _topology_spec(tmp_path, "fixtures/topology-a.json")
    top_b, spec_b = _topology_spec(tmp_path, "fixtures/topology-b.json")
    assert shadow.topology_digest(top_a) != shadow.topology_digest(top_b)
    assert dag.operation_digest(spec_a) != dag.operation_digest(spec_b)


def test_topology_pass_does_not_cross_reuse_fallback(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _, spec_a = _topology_spec(tmp_path, "fixtures/topology-a.json")
    _, spec_b = _topology_spec(tmp_path, "fixtures/topology-b.json")
    index = _index(tmp_path, _node(tmp_path, spec_a))
    assert shadow.query_shadow(
        "worker_resource_topology_qualification", spec_a, repo_root=tmp_path, index=index
    )["status"] == "REUSE_CANDIDATE"
    assert shadow.query_shadow(
        "worker_resource_topology_qualification", spec_b, repo_root=tmp_path, index=index
    )["status"] == "MISS"


def test_gpu_uuid_set_change_invalidates_exact_resource_identity(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _, spec_a = _topology_spec(tmp_path, "fixtures/topology-a.json")
    _, spec_changed = _topology_spec(tmp_path, "fixtures/topology-a-uuid2.json")
    index = _index(tmp_path, _node(tmp_path, spec_a))
    assert dag.operation_digest(spec_a) != dag.operation_digest(spec_changed)
    assert shadow.query_shadow(
        "worker_resource_topology_qualification", spec_changed, repo_root=tmp_path, index=index
    )["status"] == "MISS"


@pytest.mark.parametrize("operation_name", sorted(shadow.PROTECTED_SCIENTIFIC_OPERATIONS))
def test_scientific_exact_hit_is_advisory_only(operation_name: str) -> None:
    raw = {
        "status": "REUSE_CANDIDATE",
        "classification": "EXACT_CACHE_HIT",
        "reuse": {"automatic_reuse_permitted": True},
    }
    result = shadow.normalize_lookup_outcome(operation_name, raw)
    assert result["status"] == "REUSE_CANDIDATE_REQUIRES_SCIENTIFIC_AUTHORITY"
    assert result["scientific_authority_required"] is True
    assert result["reuse"]["automatic_reuse_permitted"] is False
    assert result["shadow_mode"] is True
    assert result["advisory_only"] is True


def test_shadow_record_forbids_accepting_scientific_reuse(tmp_path: Path) -> None:
    _fixture(tmp_path)
    spec = shadow.build_operation_spec(
        "formal_rollout",
        code_identity=_ref(tmp_path, "fixtures/code.py"),
        config_identity=_ref(tmp_path, "fixtures/config.json"),
        input_artifacts=[_artifact(tmp_path, "fixtures/input.txt", "input")],
        contract_identities=[{"name": "science-contract", "identity": _ref(tmp_path, "fixtures/contract.json")}],
        dataset_schedule_identity=_ref(tmp_path, "fixtures/schedule.json"),
        stochastic_identity={"mode": "frozen", "seed_schedule": "test"},
    )
    fake = shadow.normalize_lookup_outcome("formal_rollout", {"status": "REUSE_CANDIDATE"})
    with pytest.raises(shadow.ShadowPassportError, match="scientific reuse"):
        shadow.build_shadow_record(
            "formal_rollout", spec, fake, consumer_reuse_accepted=True
        )


def test_metrics_count_candidate_separately_from_work_actually_avoided(tmp_path: Path) -> None:
    _fixture(tmp_path)
    topology, spec = _topology_spec(tmp_path, "fixtures/topology-a.json")
    lookup = shadow.query_shadow(
        "worker_resource_topology_qualification",
        spec,
        repo_root=tmp_path,
        index=_index(tmp_path, _node(tmp_path, spec)),
    )
    metrics = shadow.new_metrics(c_workline_head="271a496c" + "0" * 32)
    metrics = shadow.observe_lookup(metrics, "worker_resource_topology_qualification", lookup)
    assert metrics["gpu_qualification_candidates_identified"] == 1
    assert metrics["avoided_gpu_qualification_operations"] == 0
    metrics = shadow.accept_engineering_reuse(
        metrics, "worker_resource_topology_qualification", lookup
    )
    assert metrics["avoided_repeat_qualifications"] == 1
    assert metrics["avoided_gpu_qualification_operations"] == 1
    shadow.assert_zero_science(metrics)
    record = shadow.build_shadow_record(
        "worker_resource_topology_qualification",
        spec,
        lookup,
        topology=topology,
        consumer_reuse_accepted=True,
    )
    assert record["formal_webshop_rows_delta"] == 0
    assert record["extra_gpu_work_for_cache_hit"] == 0


def test_engineering_candidate_can_be_written_append_only(tmp_path: Path) -> None:
    _fixture(tmp_path)
    spec = _engineering_spec(tmp_path, "model_config_image_hash_verification")
    lookup = {"status": "MISS"}
    record = shadow.build_shadow_record(
        "model_config_image_hash_verification", spec, lookup,
        consumer_reuse_accepted=False,
    )
    first = shadow.write_shadow_record(tmp_path, record)
    second = shadow.write_shadow_record(tmp_path, record)
    assert first == second
    assert (tmp_path / first["path"]).is_file()


def test_topology_semantics_are_not_label_only() -> None:
    bad = _topology(shadow.PREFERRED_TOPOLOGY)
    bad["workers_per_gpu"] = 1
    with pytest.raises(shadow.ShadowPassportError, match="workers_per_gpu"):
        shadow.validate_topology(bad)


def test_non_resource_operation_cannot_smuggle_topology(tmp_path: Path) -> None:
    _fixture(tmp_path)
    topology, ref = shadow.load_topology(tmp_path, "fixtures/topology-a.json")
    with pytest.raises(shadow.ShadowPassportError, match="topology identity belongs only"):
        shadow.build_operation_spec(
            "exact_checkout_resolution",
            code_identity=_ref(tmp_path, "fixtures/code.py"),
            config_identity=_ref(tmp_path, "fixtures/config.json"),
            input_artifacts=[_artifact(tmp_path, "fixtures/input.txt", "input")],
            topology=topology,
            topology_ref=ref,
        )


@pytest.mark.parametrize("operation_name", sorted(shadow.ENGINEERING_OPERATIONS))
def test_every_priority_engineering_operation_builds_identity(tmp_path: Path, operation_name: str) -> None:
    _fixture(tmp_path)
    kwargs = {}
    if operation_name == "worker_resource_topology_qualification":
        topology, topology_ref = shadow.load_topology(tmp_path, "fixtures/topology-a.json")
        kwargs = {
            "topology": topology,
            "topology_ref": topology_ref,
            "runtime_identity_policy": "BOUND",
            "runtime_identity": _ref(tmp_path, "fixtures/runtime.json"),
        }
    spec = shadow.build_operation_spec(
        operation_name,
        code_identity=_ref(tmp_path, "fixtures/code.py"),
        config_identity=_ref(tmp_path, "fixtures/config.json"),
        input_artifacts=[_artifact(tmp_path, "fixtures/input.txt", "input")],
        **kwargs,
    )
    assert spec["operation_type"] == f"accelerated-gdr.shadow.{operation_name}"
    assert len(dag.operation_digest(spec)) == 64


def test_shadow_record_rejects_mismatched_operation_name(tmp_path: Path) -> None:
    _fixture(tmp_path)
    spec = _engineering_spec(tmp_path, "deterministic_benchmark_analysis")
    with pytest.raises(shadow.ShadowPassportError, match="operation_name does not match"):
        shadow.build_shadow_record("artifact_archive", spec, {"status": "MISS"})


def test_resource_qualification_requires_bound_runtime(tmp_path: Path) -> None:
    _fixture(tmp_path)
    topology, topology_ref = shadow.load_topology(tmp_path, "fixtures/topology-a.json")
    with pytest.raises(shadow.ShadowPassportError, match="BOUND runtime_identity"):
        shadow.build_operation_spec(
            "worker_resource_topology_qualification",
            code_identity=_ref(tmp_path, "fixtures/code.py"),
            config_identity=_ref(tmp_path, "fixtures/config.json"),
            input_artifacts=[_artifact(tmp_path, "fixtures/input.txt", "input")],
            topology=topology,
            topology_ref=topology_ref,
        )


def test_gpu_uuid_set_digest_is_order_independent_and_exact() -> None:
    a = shadow.gpu_uuid_set_digest(["GPU-b", "GPU-a", "GPU-d", "GPU-c"])
    b = shadow.gpu_uuid_set_digest(["GPU-d", "GPU-c", "GPU-b", "GPU-a"])
    c = shadow.gpu_uuid_set_digest(["GPU-d", "GPU-c", "GPU-b", "GPU-x"])
    assert a == b
    assert a != c


def test_fallback_topology_exact_hit_is_candidate_for_fallback_only(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _, spec_a = _topology_spec(tmp_path, "fixtures/topology-a.json")
    _, spec_b = _topology_spec(tmp_path, "fixtures/topology-b.json")
    index = _index(tmp_path, _node(tmp_path, spec_b))
    assert shadow.query_shadow(
        "worker_resource_topology_qualification", spec_b, repo_root=tmp_path, index=index
    )["status"] == "REUSE_CANDIDATE"
    assert shadow.query_shadow(
        "worker_resource_topology_qualification", spec_a, repo_root=tmp_path, index=index
    )["status"] == "MISS"


@pytest.mark.parametrize("operation_name", sorted(shadow.ENGINEERING_OPERATIONS))
def test_every_priority_engineering_operation_is_queryable_without_execution(
    tmp_path: Path, operation_name: str
) -> None:
    _fixture(tmp_path)
    kwargs = {}
    if operation_name == "worker_resource_topology_qualification":
        topology, topology_ref = shadow.load_topology(tmp_path, "fixtures/topology-a.json")
        kwargs = {
            "topology": topology,
            "topology_ref": topology_ref,
            "runtime_identity_policy": "BOUND",
            "runtime_identity": _ref(tmp_path, "fixtures/runtime.json"),
        }
    spec = shadow.build_operation_spec(
        operation_name,
        code_identity=_ref(tmp_path, "fixtures/code.py"),
        config_identity=_ref(tmp_path, "fixtures/config.json"),
        input_artifacts=[_artifact(tmp_path, "fixtures/input.txt", "input")],
        **kwargs,
    )
    empty_index = dag.build_index([], generated_at="2026-09-14T13:02:00Z")
    result = shadow.query_shadow(operation_name, spec, repo_root=tmp_path, index=empty_index)
    assert result["status"] == "MISS"
    assert result["shadow_mode"] is True
    assert result["advisory_only"] is True


def test_shadow_module_has_no_process_gpu_or_model_execution_surface() -> None:
    source = Path(shadow.__file__).read_text(encoding="utf-8")
    forbidden = ("import subprocess", "subprocess.", "import torch", "torch.", "docker ", "nvidia-smi")
    assert all(token not in source for token in forbidden)
    public_callables = {
        name for name, value in vars(shadow).items()
        if callable(value) and not name.startswith("_")
    }
    assert not any(name.startswith(("launch", "run_worker", "update_model", "write_gdr")) for name in public_callables)


def test_acceleration_qualification_candidate_counts_gpu_candidate_but_not_avoided_until_accepted() -> None:
    lookup = {"status": "REUSE_CANDIDATE"}
    metrics = shadow.new_metrics(c_workline_head="f" * 40)
    metrics = shadow.observe_lookup(metrics, "acceleration_successor_qualification", lookup)
    assert metrics["gpu_qualification_candidates_identified"] == 1
    assert metrics["avoided_gpu_qualification_operations"] == 0
    metrics = shadow.accept_engineering_reuse(metrics, "acceleration_successor_qualification", lookup)
    assert metrics["avoided_repeat_qualifications"] == 1
    assert metrics["avoided_gpu_qualification_operations"] == 1


def test_historical_shadow_backfills_are_not_promoted_into_current_index() -> None:
    index, _ = dag.load_current_index(dag.ROOT)
    acceleration = {
        entry["operation_digest"]
        for entry in index["entries"]
        if entry["operation_type"] == "accelerated-gdr.shadow.acceleration_successor_qualification"
    }
    evidence_root = dag.ROOT / "docs/evidence/passport-accelerated-gdr-shadow-20260914"
    a = json.loads((evidence_root / "CANDIDATE_A_BACKFILL_LOOKUP_202609142122.json").read_text())
    packed = json.loads((evidence_root / "PACKED_TRANSFER_BACKFILL_LOOKUP_202609142144.json").read_text())
    historical = {a["operation_digest"], packed["operation_digest"]}
    assert historical.isdisjoint(acceleration)
    assert a["counts_as_avoided_real_work"] is False
    assert packed["counts_as_avoided_real_work"] is False
    metrics = json.loads((evidence_root / "SHADOW_METRICS_CURRENT_202609142144.json").read_text())
    assert metrics["gpu_qualification_candidates_identified"] == 2
    assert metrics["avoided_gpu_qualification_operations"] == 0
