import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.analysis import build_advisor_fresh_canonical_20261020 as builder
from scripts.openevo_webshop import advisor_fresh_checkpoint_host_20261020 as host
from scripts.openevo_webshop.experiment_passport_gate import evaluate_release_gate
from scripts.openevo_webshop import execution_readiness as readiness

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.mark.parametrize("has_scientific_contract", [True, False])
def test_frozen_metadata_passes_real_native_readiness_contract(tmp_path, monkeypatch, has_scientific_contract):
    for rel in ["schemas/experiment-passport-202609111800.schema.json",
                "schemas/experiment-registry-202609112120.schema.json", builder.HOST, builder.WORKER,
                builder.RUNTIME_PYTHON, "configs/runtime/advisor-fresh-runtime-fingerprint-base-20261020.json",
                "configs/assets/runtimes/h144-scientific-runtime-v1.json"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    image = "sha256:" + "d" * 64
    manifest = {
        "image_id": image, "final_panel_access_allowed": False, "optimizer_updates": 0,
        "tasks": [{"id": i} for i in range(64)], "panel_sha256": "e" * 64,
        "model": {"local_model_path": "/workspace/models/frozen", "revision": "a" * 40},
        "seed_root": "/workspace/assets/seed",
        "runtime_source_root": "/workspace/control/runtime",
        "core_root": "/workspace/control/openevo-core",
        "common_carriers": {kind: {"path": f"/workspace/assets/{kind}", "sha256": "b" * 64}
                            for kind in ("memory", "skill", "agent")},
        "checkpoints": {"fixed-a": {"adapter_path": "/workspace/runs/fixed-a",
                                   "files": {"adapter_model.safetensors": "c" * 64}}},
    }
    manifest_rel = Path("configs/experiment/advisor-questions/fresh-test.json")
    manifest["panel_sha256"] = hashlib.sha256(json.dumps(manifest["tasks"], ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prereg_rel = Path("configs/experiment/preregistrations/fresh-test.json")
    dump(tmp_path / manifest_rel, manifest)
    prereg = {"evaluation_episodes": 64}
    if has_scientific_contract:
        prereg["frozen_scientific_contract"] = {
            "manifest": {"path": str(manifest_rel), "sha256": builder.sha(tmp_path / manifest_rel)},
            "optimizer_updates": 0, "final_panel_access_allowed": False}
    dump(tmp_path / prereg_rel, prereg)
    lanes = tmp_path / "lanes.json"
    dump(lanes, [{"lane_id": "gpu1", "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
                  "checkpoint_ids": ["fixed-a"]}])
    base_rel = Path("configs/runtime/advisor-fresh-runtime-fingerprint-base-20261020.json")
    base = json.loads((tmp_path / base_rel).read_text())
    base["execution_image"]["id"] = image
    base["openevo_core"] = {"path": manifest["core_root"], "revision": "2" * 40}
    base["runtime_code"]["path"] = manifest["runtime_source_root"]
    dump(tmp_path / base_rel, base)
    empty = {"schema": "openevo.experiment-registry.202609112120",
             "role": "identity-routing-only; never live state or execution authority", "entries": []}
    monkeypatch.setattr(builder, "load_current_registry", lambda root: (empty, {}))
    real_validate = builder.validate_registry
    monkeypatch.setattr(builder, "validate_registry", lambda registry, root:
                        [] if not registry["entries"] else real_validate(registry, root))
    args = argparse.Namespace(repo_root=tmp_path, stamp="202610201200", execution_sha="1" * 40,
        image=image, manifest=manifest_rel, manifest_sha256=builder.sha(tmp_path / manifest_rel),
        preregistration=prereg_rel, runtime_lock=Path("configs/assets/runtimes/h144-scientific-runtime-v1.json"),
        lane_map=lanes, execution_root=Path("/workspace/control/fresh-execution"),
        workspace=Path("/workspace"), core_root=Path("/workspace/control/openevo-core"),
        core_sha="2" * 40, output_base=Path("/workspace/runs"), python_version="3.11.16",
        runtime_fingerprint_base=base_rel)
    if not has_scientific_contract:
        with pytest.raises(readiness.ExecutionReadinessError, match="frozen_scientific_contract"):
            builder.build(args)
        assert not (tmp_path / "configs/experiment/releases").exists()
        return
    result = builder.build(args)
    release = json.loads((tmp_path / result["lanes"][0]["release"]).read_text())
    gate = evaluate_release_gate(release, tmp_path)
    assert gate["status"] == "PASS"
    lane = result["lanes"][0]
    check_args = argparse.Namespace(repo_root=tmp_path, release=tmp_path / lane["release"],
        current_campaign=tmp_path / lane["current_campaign"], campaign=None, run_manifest=None)
    assert readiness._fail_fast_contract(check_args) is None
    profile = readiness.resolve_execution_profile(release, repo_root=tmp_path)
    assert Path(profile["path"]).is_absolute()
    assert builder.sha(Path(profile["path"])) == profile["recorded_sha256"].removeprefix("sha256:")
    artifacts = readiness._legacy.hash_bound_artifacts(release, repo_root=tmp_path)
    assert not artifacts["missing"]
    assert not artifacts["mismatched"]
    qualification = readiness._legacy.build_qualification_args(release,
        fingerprint_file=tmp_path / result["fingerprint"], cache_root=tmp_path / "cache", repo_root=tmp_path)
    assert qualification.runtime_python == args.execution_root / builder.RUNTIME_PYTHON
    assert qualification.openevo_root == args.core_root
    assert qualification.source_root == Path(manifest["seed_root"])
    assert qualification.model == Path(manifest["model"]["local_model_path"])
    assert qualification.expected_image_digest == image
    assert qualification.runtime_container is True
    assert release["evaluation_episode_limit"] == 64
    assert release["training_rollout_limit"] == 0
    assert release["experiment_identity"]["experiment_id"] == "202610201200-advisor-fresh-gpu1"


def test_manifest_hash_drift_rejected_before_runtime(tmp_path):
    manifest = tmp_path / "manifest.json"
    dump(manifest, {"checkpoints": {}})
    with pytest.raises(ValueError, match="SHA drift"):
        host.checked_manifest(manifest, "0" * 64, ["fixed-a"])


def test_partial_lane_start_never_replayed(tmp_path, monkeypatch):
    output = tmp_path / "evaluation"
    output.mkdir()
    dump(output / "START.json", {"partial_consumption": True})
    monkeypatch.setattr(host, "validate_start", lambda args: ({}, {}))
    monkeypatch.setattr(host.subprocess, "run", lambda *a, **kw: pytest.fail("existing lane must not launch Docker"))
    with pytest.raises(FileExistsError):
        host.host(argparse.Namespace(output=output))
    assert json.loads((output / "START.json").read_text()) == {"partial_consumption": True}


def test_gpu_assignment_drift_rejected_before_docker_or_output(tmp_path, monkeypatch):
    tasks = [{"id": i} for i in range(64)]
    panel_sha = hashlib.sha256(json.dumps(tasks, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = tmp_path / "manifest.json"
    image = "sha256:" + "d" * 64
    dump(manifest, {"image_id": image, "tasks": tasks, "panel_sha256": panel_sha,
                    "checkpoints": {"fixed-a": {}}, "optimizer_updates": 0,
                    "final_panel_access_allowed": False})
    formal = tmp_path / "formal.json"
    dump(formal, {"formal_start_committed": True, "formal_task_consumption_may_begin": True,
                  "run_id": "fresh", "selected_gpu_uuids": ["GPU-other"], "image_id": image})
    monkeypatch.setenv("OPENEVO_FORMAL_START_RECEIPT", str(formal))
    monkeypatch.setenv("OPENEVO_ASSIGNED_GPU_UUIDS", "GPU-expected")
    monkeypatch.setattr(host.subprocess, "check_output", lambda *a, **kw: pytest.fail("no Docker/Git after failed assignment"))
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="Canonical Formal Start"):
        host.validate_start(argparse.Namespace(manifest=manifest, manifest_sha256=host.sha(manifest),
            checkpoint_id=["fixed-a"], gpu_uuid="GPU-expected", run_id="fresh", image=image, output=output))
    assert not output.exists()


def test_q06_ordinary160_record_identity_is_reusable_but_not_science():
    from scripts.analysis import audit_advisor_fresh_ordinary160_q06_20261021 as audit

    report = audit.audit(ROOT)
    assert report["status"] == "PASS_GITHUB_RECORD_IDENTITY_ONLY"
    assert report["same_base_revision_and_adapter_config"] is True
    assert report["q06_synthetic_calls_not_scientific"] is True
    assert report["recorded_completed_snapshot"] == 320
    assert report["recorded_remaining_snapshot"] == 128
    assert report["live_formal_start_and_all_tasks_verified"] is False


def test_q06_ordinary160_sha_substitution_fails_closed(tmp_path):
    from scripts.analysis import audit_advisor_fresh_ordinary160_q06_20261021 as audit

    for relative in (audit.Q06, audit.QUALIFICATION, audit.FRESH,
                     audit.PREREG, audit.REMAINING, audit.PROFILE):
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    manifest_path = tmp_path / audit.FRESH
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoints"]["ordinary160"]["files"]["adapter_model.safetensors"] = "0" * 64
    dump(manifest_path, manifest)
    with pytest.raises(ValueError, match="source weight SHA"):
        audit.audit(tmp_path)


def test_q06_synthetic_qualification_cannot_become_scientific_receipt(tmp_path):
    from scripts.analysis import audit_advisor_fresh_ordinary160_q06_20261021 as audit

    for relative in (audit.Q06, audit.QUALIFICATION, audit.FRESH,
                     audit.PREREG, audit.REMAINING, audit.PROFILE):
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    qualification_path = tmp_path / audit.QUALIFICATION
    qualification = json.loads(qualification_path.read_text())
    qualification["environment_calls"] = 1
    dump(qualification_path, qualification)
    with pytest.raises(ValueError, match="non-scientific scope"):
        audit.audit(tmp_path)


def test_448_readonly_reconciler_rejects_protocol_drift():
    from scripts.analysis import reconcile_advisor_fresh_448_20261021 as reconcile

    assert reconcile.percentile([0, 2, 4, 6, 8], 0.25) == 2
    assert reconcile.percentile([0, 2, 4, 6, 8], 0.125) == 1
    zero_deltas = [[0.0] * 64 for _ in range(8)]
    with pytest.raises(ValueError, match="prospective bootstrap settings drift"):
        reconcile.paired_bootstrap(zero_deltas, draws=49999, seed=20261020, tail=0.003125)
    with pytest.raises(ValueError, match="prospective bootstrap settings drift"):
        reconcile.paired_bootstrap(zero_deltas, draws=50000, seed=20261021, tail=0.003125)
    with pytest.raises(ValueError, match="4 contrasts times 2 paired endpoints"):
        reconcile.paired_bootstrap(zero_deltas[:-1], draws=50000, seed=20261020, tail=0.003125)
