from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.openevo_webshop import successor_handoff as sh


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def make_spec(tmp_path: Path) -> Path:
    stable = tmp_path / "seed-envs.py"
    stable.write_text("PINNED = True\n", encoding="utf-8")
    qual = tmp_path / "runtime-qualification.json"
    write_json(
        qual,
        {
            "schema": "openevo.runtime-qualification.v1",
            "status": "PASS",
            "request": {"runtime_fingerprint_sha256": "sha256:" + "1" * 64},
        },
    )
    spec = {
        "schema": sh.SPEC_SCHEMA,
        "predecessor": {
            "phase_id": "seed-webshop-v2",
            "experiment_revision": "4db98b0",
        },
        "successor": {"phase_id": "seed-webshop-pubcode-repro-v1"},
        "facts": [
            {
                "role": "seed-upstream-revision",
                "kind": "identity",
                "lifetime": "immutable",
                "value": "2cf2fad",
            },
            {
                "role": "seed-task-identity:envs.py",
                "kind": "file",
                "lifetime": "immutable",
                "path": stable.name,
            },
            {
                "role": "runtime-qualification",
                "kind": "receipt",
                "lifetime": "runtime-qualified",
                "path": qual.name,
                "required_schema": "openevo.runtime-qualification.v1",
                "required_status": "PASS",
            },
        ],
    }
    path = tmp_path / "handoff-spec.json"
    write_json(path, spec)
    return path


def test_capture_is_deterministic_and_never_authorizes_execution(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    first = sh.capture(spec, root=tmp_path)
    second = sh.capture(spec, root=tmp_path)
    assert sh.canonical_json_bytes(first) == sh.canonical_json_bytes(second)
    assert first["status"] == "PASS"
    assert first["formal_task_consumption_allowed"] is False
    assert first["gpu_allocation_allowed"] is False


def test_bootstrap_reuses_immutable_and_delegates_runtime_qualification(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    receipt = sh.capture(spec, root=tmp_path)
    handoff = tmp_path / "handoff.json"
    sh.atomic_json(handoff, receipt)

    result = sh.bootstrap(
        handoff,
        root=tmp_path,
        required_roles=("seed-upstream-revision", "runtime-qualification"),
    )

    assert result["status"] == "READY_FOR_SUCCESSOR_PREP"
    assert "seed-upstream-revision" in result["reused_roles"]
    assert "seed-task-identity:envs.py" in result["reused_roles"]
    assert result["runtime_qualification_cache_candidates"] == [
        "runtime-qualification"
    ]
    by_role = {item["role"]: item["status"] for item in result["checks"]}
    assert by_role["runtime-qualification"] == "CACHE_CANDIDATE"
    assert result["formal_task_consumption_allowed"] is False


def test_bootstrap_fails_closed_when_immutable_file_changes(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    handoff = tmp_path / "handoff.json"
    sh.atomic_json(handoff, sh.capture(spec, root=tmp_path))
    (tmp_path / "seed-envs.py").write_text("PINNED = False\n", encoding="utf-8")

    result = sh.bootstrap(handoff, root=tmp_path)

    assert result["status"] == "STALE"
    assert any(
        item["role"] == "seed-task-identity:envs.py"
        and "content changed" in item["reason"]
        for item in result["failures"]
    )


def test_capture_rejects_expected_digest_mismatch(tmp_path: Path) -> None:
    spec_path = make_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["facts"][1]["sha256"] = "sha256:" + "0" * 64
    write_json(spec_path, spec)

    with pytest.raises(sh.HandoffError, match="content mismatch"):
        sh.capture(spec_path, root=tmp_path)


def test_capture_rejects_volatile_fact(tmp_path: Path) -> None:
    spec_path = make_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["facts"].append(
        {
            "role": "gpu-occupancy",
            "kind": "identity",
            "lifetime": "volatile",
            "value": "idle",
        }
    )
    write_json(spec_path, spec)

    with pytest.raises(sh.HandoffError, match="volatile facts must be rechecked"):
        sh.capture(spec_path, root=tmp_path)


def test_bootstrap_requires_declared_roles(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    handoff = tmp_path / "handoff.json"
    sh.atomic_json(handoff, sh.capture(spec, root=tmp_path))

    result = sh.bootstrap(
        handoff,
        root=tmp_path,
        required_roles=("missing-protocol-fingerprint",),
    )

    assert result["status"] == "STALE"
    assert result["failures"][-1] == {
        "role": "missing-protocol-fingerprint",
        "reason": "required role absent from handoff",
    }


def test_runtime_receipt_status_drift_fails_closed(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    handoff = tmp_path / "handoff.json"
    sh.atomic_json(handoff, sh.capture(spec, root=tmp_path))
    qual = tmp_path / "runtime-qualification.json"
    write_json(
        qual,
        {"schema": "openevo.runtime-qualification.v1", "status": "FAIL"},
    )

    result = sh.bootstrap(handoff, root=tmp_path)

    assert result["status"] == "STALE"
    assert any(
        item["role"] == "runtime-qualification"
        and "content changed" in item["reason"]
        for item in result["failures"]
    )


def test_bootstrap_rejects_forged_kind_lifetime_pairing(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    handoff = tmp_path / "handoff.json"
    receipt = sh.capture(spec, root=tmp_path)
    receipt["facts"][0]["lifetime"] = "runtime-qualified"
    sh.atomic_json(handoff, receipt)

    result = sh.bootstrap(handoff, root=tmp_path)

    assert result["status"] == "STALE"
    assert any(
        item["role"] == "seed-upstream-revision"
        and "invalid kind/lifetime pairing" in item["reason"]
        for item in result["failures"]
    )


def test_bootstrap_can_bind_handoff_digest(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    handoff = tmp_path / "handoff.json"
    sh.atomic_json(handoff, sh.capture(spec, root=tmp_path))
    expected = sh.sha256_file(handoff)

    result = sh.bootstrap(
        handoff,
        root=tmp_path,
        expected_handoff_sha256=expected,
    )
    assert result["status"] == "READY_FOR_SUCCESSOR_PREP"
    assert result["handoff_sha256"] == expected

    with pytest.raises(sh.HandoffError, match="successor handoff: content mismatch"):
        sh.bootstrap(
            handoff,
            root=tmp_path,
            expected_handoff_sha256="sha256:" + "0" * 64,
        )
