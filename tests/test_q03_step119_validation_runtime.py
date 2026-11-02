from __future__ import annotations

import base64
import builtins
import hashlib
import json
import os
import pwd
import types
from pathlib import Path

import pytest

from scripts.openevo_webshop import q03_step119_validation_runtime as runtime
from scripts.openevo_webshop import q03_step119_validation_results as results


def _write(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    path.chmod(0o444)
    return hashlib.sha256(body).hexdigest()


def _make_tree(tmp_path: Path, monkeypatch):
    model = tmp_path / "models" / "qwen"
    model.mkdir(parents=True)
    model_files = {
        "model.safetensors": ("MODEL_SHA256", b"weights"),
        "config.json": ("CONFIG_SHA256", b"config"),
        "generation_config.json": ("GENERATION_CONFIG_SHA256", b"generation"),
        "tokenizer.json": ("TOKENIZER_SHA256", b"tokenizer"),
        "tokenizer_config.json": ("TOKENIZER_CONFIG_SHA256", b"tokenizer-config"),
        "chat_template.jinja": ("CHAT_TEMPLATE_SHA256", b"template"),
    }
    model_paths = {}
    for name, (constant, body) in model_files.items():
        model_paths[name] = model / name
        monkeypatch.setattr(runtime, constant, _write(model_paths[name], body))
    model.chmod(0o555)
    model.parent.chmod(0o555)

    source_root = tmp_path / "runtime" / "host-effective-sources" / "fixed-sha"
    validation_dir = source_root / runtime.VALIDATION_ROOT_RELATIVE
    parquet = validation_dir / "test.parquet"
    manifest = validation_dir / "task_manifest.json"
    monkeypatch.setattr(runtime, "PARQUET_SHA256", _write(parquet, b"carrier"))
    monkeypatch.setattr(runtime, "MANIFEST_SHA256", _write(manifest, b"manifest"))
    adapter = source_root / runtime.CUSTOM_DATASET_RELATIVE
    helper = source_root / runtime.MANIFEST_HELPER_RELATIVE
    monkeypatch.setattr(runtime, "CUSTOM_DATASET_SHA256", _write(adapter, b"adapter"))
    monkeypatch.setattr(runtime, "MANIFEST_HELPER_SHA256", _write(helper, b"helper"))
    for directory in sorted((p for p in source_root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        directory.chmod(0o555)
    source_root.chmod(0o555)
    effective_parent = source_root.parent
    effective_parent.chmod(0o555)
    staging_root = effective_parent.parent
    staging_root.chmod(0o555)

    run_dir = tmp_path / "runs" / "fresh"
    run_dir.mkdir(parents=True)
    run_dir.parent.chmod(0o700)
    run_dir.chmod(0o700)
    monkeypatch.setattr(runtime, "SEALED_TMPFS_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "EXPECTED_RUN_ROOT", run_dir)
    uid = os.getuid()
    gid = pwd.getpwuid(uid).pw_gid
    return model, parquet, manifest, source_root, run_dir, uid, gid


def test_model_load_preflight_rejects_direct_runtime_invocation_without_launcher_context(monkeypatch) -> None:
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 1001)
    monkeypatch.setattr(runtime.os, "getegid", lambda: 1001)
    monkeypatch.setattr(runtime, "_validated_run_identity", lambda _uid, _gid: object())

    with pytest.raises(runtime.RuntimeContractError, match="requires launcher authorization context"):
        runtime.main([
            "--run-as-uid", "1001", "--run-as-gid", "1001",
            "--model-path", "/unused/model", "--parquet-path", "/unused/test.parquet",
            "--manifest-path", "/unused/manifest.json", "--source-root", "/unused/source",
            "--run-dir", "/workspace/runs/.zero-formal-direct-call",
            "--model-load-preflight-only",
        ])


def test_model_load_preflight_accepts_container_run_identity_without_root_drop(monkeypatch) -> None:
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 1001)
    monkeypatch.setattr(runtime.os, "getegid", lambda: 1001)
    monkeypatch.setattr(runtime, "_validated_run_identity", lambda _uid, _gid: object())
    monkeypatch.setattr(runtime, "validate_model_load_preflight_authorization", lambda *_args, **_kwargs: {})

    def reached_input_validation(**_kwargs):
        raise runtime.RuntimeContractError("input validation reached")

    monkeypatch.setattr(runtime, "validate_inputs", reached_input_validation)
    with pytest.raises(runtime.RuntimeContractError, match="input validation reached"):
        runtime.main([
            "--run-as-uid", "1001", "--run-as-gid", "1001",
            "--model-path", "/unused/model", "--parquet-path", "/unused/test.parquet",
            "--manifest-path", "/unused/manifest.json", "--source-root", "/unused/source",
            "--run-dir", "/workspace/runs/.zero-formal-direct-call",
            "--model-load-preflight-only", "--model-load-preflight-context", "placeholder",
        ])


def test_model_load_preflight_root_control_plane_drops_before_runtime_imports(tmp_path: Path, monkeypatch) -> None:
    identity = {"uid": 0, "gid": 0}
    monkeypatch.setattr(runtime.os, "geteuid", lambda: identity["uid"])
    monkeypatch.setattr(runtime.os, "getegid", lambda: identity["gid"])
    monkeypatch.setattr(runtime.os, "initgroups", lambda *_args: None)
    monkeypatch.setattr(runtime.os, "setgid", lambda gid: identity.update(gid=gid))
    monkeypatch.setattr(runtime.os, "setuid", lambda uid: identity.update(uid=uid))
    monkeypatch.setattr(runtime, "_validated_run_identity", lambda _uid, _gid: types.SimpleNamespace(pw_name="runner"))
    monkeypatch.setattr(runtime, "validate_model_load_preflight_authorization", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runtime, "validate_inputs", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_prepare_run_root", lambda *_args, **_kwargs: None)
    model = tmp_path / "model"
    model.mkdir()
    parquet = tmp_path / "test.parquet"
    parquet.write_bytes(b"data")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "runs" / ".zero-formal-root-control"
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(runtime, "build_hydra_overrides", lambda **_kwargs: [])
    original_import = builtins.__import__

    def assert_dropped_import(name, *args, **kwargs):
        if name == "hydra":
            assert identity == {"uid": 1001, "gid": 1001}
            raise runtime.RuntimeContractError("dropped before runtime setup")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", assert_dropped_import)
    with pytest.raises(runtime.RuntimeContractError, match="dropped before runtime setup"):
        runtime.main([
            "--run-as-uid", "1001", "--run-as-gid", "1001",
            "--model-path", str(model), "--parquet-path", str(parquet),
            "--manifest-path", str(manifest), "--source-root", str(source),
            "--run-dir", str(run_dir),
            "--model-load-preflight-only", "--model-load-preflight-context", "placeholder",
        ])


def test_model_load_preflight_authorization_binds_candidate_profile_gpu_and_source() -> None:
    context = {
        "schema": "openevo.q03-step119-launcher-preflight-context.v1",
        "status": "LAUNCHER_AUTHORIZED_ZERO_TASK_PREFLIGHT",
        "campaign_id": "q03-step119",
        "execution_sha": "1" * 40,
        "candidate_hashes": {
            "current_campaign": "2" * 64, "design": "3" * 64, "activation": "4" * 64,
            "runtime_profile": "5" * 64, "registry_pointer": "6" * 64,
        },
        "runtime_profile_sha256": "5" * 64,
        "selected_gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000000"],
        "effective_source_sha256": "7" * 64,
        "image_id": "sha256:" + "8" * 64,
        "run_id": ".zero-formal-q03-step119",
        "formal_task_count": 0, "training_rollouts": 0, "generation_calls": 0,
        "optimizer_updates": 0, "final_panel_access": 0, "scientific_state_mutated": False,
    }
    context["context_sha256"] = runtime._canonical_sha256(context)
    encoded = base64.urlsafe_b64encode(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    env = {
        "OPENEVO_STEP119_PREFLIGHT_CONTEXT_SHA256": context["context_sha256"],
        "OPENEVO_STEP119_CAMPAIGN_ID": context["campaign_id"],
        "OPENEVO_STEP119_EXECUTION_SHA": context["execution_sha"],
        "OPENEVO_STEP119_PROFILE_SHA256": context["runtime_profile_sha256"],
        "OPENEVO_EFFECTIVE_SOURCE_SHA256": context["effective_source_sha256"],
        "OPENEVO_STEP119_IMAGE_ID": context["image_id"],
        "OPENEVO_AUTHORIZED_GPU_UUIDS": context["selected_gpu_uuids"][0],
        "OPENEVO_ASSIGNED_GPU_UUIDS": context["selected_gpu_uuids"][0],
    }

    validated = runtime.validate_model_load_preflight_authorization(
        encoded, run_dir=Path("/workspace/runs/.zero-formal-q03-step119"), environ=env
    )
    assert validated["context_sha256"] == context["context_sha256"]

    env["OPENEVO_EFFECTIVE_SOURCE_SHA256"] = "9" * 64
    with pytest.raises(runtime.RuntimeContractError, match="does not match effective_source_sha256"):
        runtime.validate_model_load_preflight_authorization(
            encoded, run_dir=Path("/workspace/runs/.zero-formal-q03-step119"), environ=env
        )

    tampered = dict(context)
    tampered["campaign_id"] = "other-campaign"
    tampered_encoded = base64.urlsafe_b64encode(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    env["OPENEVO_EFFECTIVE_SOURCE_SHA256"] = context["effective_source_sha256"]
    with pytest.raises(runtime.RuntimeContractError, match="context hash mismatch"):
        runtime.validate_model_load_preflight_authorization(
            tampered_encoded, run_dir=Path("/workspace/runs/.zero-formal-q03-step119"), environ=env
        )

    inconsistent = dict(context)
    inconsistent["candidate_hashes"] = dict(context["candidate_hashes"])
    inconsistent["candidate_hashes"]["runtime_profile"] = "9" * 64
    inconsistent["context_sha256"] = runtime._canonical_sha256(
        {key: value for key, value in inconsistent.items() if key != "context_sha256"}
    )
    inconsistent_encoded = base64.urlsafe_b64encode(
        json.dumps(inconsistent, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    env["OPENEVO_STEP119_PREFLIGHT_CONTEXT_SHA256"] = inconsistent["context_sha256"]
    with pytest.raises(runtime.RuntimeContractError, match="candidate profile hash is inconsistent"):
        runtime.validate_model_load_preflight_authorization(
            inconsistent_encoded,
            run_dir=Path("/workspace/runs/.zero-formal-q03-step119"),
            environ=env,
        )


def test_hydra_config_is_frozen_to_public_validation_and_disables_resume(tmp_path: Path) -> None:
    source = tmp_path / "effective-source"
    model = tmp_path / "models" / "Qwen3-1.7B"
    run_dir = tmp_path / "runs" / "q03-step119"

    overrides = runtime.build_hydra_overrides(
        model_path=model,
        parquet_path=source / runtime.PARQUET_RELATIVE,
        manifest_path=source / runtime.MANIFEST_RELATIVE,
        source_root=source,
        run_dir=run_dir,
    )
    values = dict(item.split("=", 1) for item in overrides)
    assert values["data.train_files"] == "q03://step119/non-readable-training-sentinel"
    assert values["data.val_files"] == str(source / runtime.PARQUET_RELATIVE)
    assert values["data.custom_cls.path"] == str(source / runtime.CUSTOM_DATASET_RELATIVE)
    assert values["data.custom_cls.name"] == "Q03Step119ValidationDataset"
    assert values["+data.q03_step119_validation_parquet_sha256"] == runtime.PARQUET_SHA256
    assert values["+data.q03_step119_task_manifest_sha256"] == runtime.MANIFEST_SHA256
    assert values["trainer.val_only"] == "True"
    assert values["trainer.val_before_train"] == "True"
    assert values["trainer.resume_mode"] == "disable"
    assert values["trainer.test_freq"] == "-1"
    assert values["trainer.save_freq"] == "-1"
    assert values["data.val_batch_size"] == "128"
    assert values["env.seed"] == "0"
    assert not any("train.parquet" in item or "final" in item.lower() for item in overrides)


def test_model_load_preflight_runs_native_init_workers_without_dataset_rows_or_fit() -> None:
    class FakeOmegaConf:
        @staticmethod
        def merge(left, right):
            return {**left, **right}

        @staticmethod
        def create(value):
            return value

        @staticmethod
        def to_container(value, resolve=False):
            assert resolve is True
            return value

    class FakeRay:
        initialized = False
        init_calls = 0
        shutdown_calls = 0

        def is_initialized(self):
            return self.initialized

        def init(self, **_kwargs):
            self.initialized = True
            self.init_calls += 1

        def shutdown(self):
            self.initialized = False
            self.shutdown_calls += 1

    class FakeTrainer:
        def __init__(self):
            self.envs = None
            self.val_envs = None
            self.actor_rollout_wg = None
            self.train_dataset = None
            self.val_dataset = None

        def init_workers(self):
            self.actor_rollout_wg = object()

        def fit(self):
            raise AssertionError("native fit must be replaced by the preflight boundary")

    native = types.SimpleNamespace(
        create_rl_dataset=lambda *_args, **_kwargs: None,
        create_rl_sampler=lambda *_args, **_kwargs: None,
        get_ppo_ray_runtime_env=lambda: {"env_vars": {"TOKENIZERS_PARALLELISM": "false"}},
        OmegaConf=FakeOmegaConf,
    )
    environments = types.SimpleNamespace(make_envs=lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("environment creation must be intercepted")
    ))
    ray = FakeRay()
    seen = {}

    def native_run(_self, config):
        envs, val_envs = environments.make_envs(config)
        train = native.create_rl_dataset("train", config)
        validation = native.create_rl_dataset("validation", config)
        trainer = FakeTrainer()
        trainer.envs, trainer.val_envs = envs, val_envs
        trainer.train_dataset, trainer.val_dataset = train, validation
        trainer.init_workers()
        trainer.fit()
        seen["trainer"] = trainer

    receipt = runtime.run_zero_task_model_load_preflight(
        native_run=native_run,
        native_module=native,
        trainer_class=FakeTrainer,
        environments_module=environments,
        config={"ray_init": {"num_cpus": 1}},
        ray_module=ray,
    )

    assert receipt["status"] == "MODEL_LOAD_PREFLIGHT_PASS"
    assert receipt["actor_rollout_model_initialized"] is True
    assert receipt["vllm_runtime_initialized"] is True
    assert receipt["formal_task_count"] == 0
    assert receipt["dataset_factory_calls"] == 2
    assert receipt["dataset_item_reads"] == 0
    assert receipt["generation_calls"] == 0
    assert receipt["optimizer_updates"] == 0
    assert seen["trainer"].actor_rollout_wg is not None
    assert ray.init_calls == ray.shutdown_calls == 1


def test_preflight_output_must_be_private_fresh_and_separate_from_formal_root(tmp_path: Path, monkeypatch) -> None:
    formal_root = tmp_path / "runs" / "q03-step119"
    formal_root.parent.mkdir(parents=True)
    formal_root.mkdir()
    monkeypatch.setattr(runtime, "EXPECTED_RUN_ROOT", formal_root)
    preflight_root = formal_root.parent / ".zero-formal-q03-step119-unique"
    preflight_root.mkdir(mode=0o700)
    uid = os.getuid()
    gid = pwd.getpwuid(uid).pw_gid

    runtime._check_preflight_run_dir(preflight_root, uid=uid, gid=gid)
    (preflight_root / "unexpected.json").write_text("{}")
    with pytest.raises(runtime.RuntimeContractError, match="fresh .zero-formal"):
        runtime._check_preflight_run_dir(preflight_root, uid=uid, gid=gid)


def test_validation_accepts_exact_read_only_snapshot_and_rejects_mutable_manifest(tmp_path: Path, monkeypatch) -> None:
    model, parquet, manifest, source_root, run_dir, uid, gid = _make_tree(tmp_path, monkeypatch)
    runtime.validate_inputs(
        model_path=model, parquet_path=parquet, manifest_path=manifest,
        source_root=source_root, run_as_uid=uid, run_as_gid=gid, run_dir=run_dir,
    )

    outside_manifest = tmp_path / "mutable-worktree-manifest.json"
    outside_manifest.write_bytes(manifest.read_bytes())
    with pytest.raises(runtime.RuntimeContractError, match="immutable source root"):
        runtime.validate_inputs(
            model_path=model, parquet_path=parquet, manifest_path=outside_manifest,
            source_root=source_root, run_as_uid=uid, run_as_gid=gid, run_dir=run_dir,
        )


def test_validation_rejects_prior_checkpoint_and_drifted_tokenizer(tmp_path: Path, monkeypatch) -> None:
    model, parquet, manifest, source_root, run_dir, uid, gid = _make_tree(tmp_path, monkeypatch)
    (run_dir / "global_step_1").mkdir()
    with pytest.raises(runtime.RuntimeContractError, match="not fresh"):
        runtime.validate_inputs(
            model_path=model, parquet_path=parquet, manifest_path=manifest,
            source_root=source_root, run_as_uid=uid, run_as_gid=gid, run_dir=run_dir,
        )
    (run_dir / "global_step_1").rmdir()
    (model / "tokenizer.json").chmod(0o644)
    (model / "tokenizer.json").write_bytes(b"drift")
    with pytest.raises(runtime.RuntimeContractError, match="tokenizer SHA-256 mismatch"):
        runtime.validate_inputs(
            model_path=model, parquet_path=parquet, manifest_path=manifest,
            source_root=source_root, run_as_uid=uid, run_as_gid=gid, run_dir=run_dir,
        )


def test_validation_rejects_any_unrecognized_prior_run_output(tmp_path: Path, monkeypatch) -> None:
    model, parquet, manifest, source_root, run_dir, uid, gid = _make_tree(tmp_path, monkeypatch)
    (run_dir / "metrics.json").write_text("{}\n")
    with pytest.raises(runtime.RuntimeContractError, match="unexpected prior output: metrics.json"):
        runtime.validate_inputs(
            model_path=model, parquet_path=parquet, manifest_path=manifest,
            source_root=source_root, run_as_uid=uid, run_as_gid=gid, run_dir=run_dir,
        )


def test_validation_allows_only_current_launcher_receipts(tmp_path: Path, monkeypatch) -> None:
    model, parquet, manifest, source_root, run_dir, uid, gid = _make_tree(tmp_path, monkeypatch)
    body = json.dumps({"run_id": run_dir.name}) + "\n"
    (run_dir / "launch-receipt.json").write_text(body)
    (run_dir / "host-runtime-import-identity.json").write_text(body)
    runtime.validate_inputs(
        model_path=model, parquet_path=parquet, manifest_path=manifest,
        source_root=source_root, run_as_uid=uid, run_as_gid=gid, run_dir=run_dir,
    )


def test_formal_runtime_creates_only_the_exact_fresh_private_run_root(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    runs.chmod(0o700)
    run_dir = runs / "q03-step119"
    monkeypatch.setattr(runtime, "EXPECTED_RUN_ROOT", run_dir)
    uid = os.getuid()
    gid = pwd.getpwuid(uid).pw_gid

    runtime._create_exact_run_root(run_dir, uid=uid, gid=gid)

    info = run_dir.stat()
    assert run_dir.is_dir() and not run_dir.is_symlink()
    assert info.st_uid == uid and info.st_gid == gid
    assert info.st_mode & 0o777 == 0o700
    with pytest.raises(runtime.RuntimeContractError, match="already exists"):
        runtime._create_exact_run_root(run_dir, uid=uid, gid=gid)


def test_formal_runtime_rejects_run_root_path_substitution(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    runs.chmod(0o700)
    expected = runs / "q03-step119"
    other = runs / "other"
    monkeypatch.setattr(runtime, "EXPECTED_RUN_ROOT", expected)
    uid = os.getuid()
    gid = pwd.getpwuid(uid).pw_gid

    with pytest.raises(runtime.RuntimeContractError, match="exact fresh run root"):
        runtime._create_exact_run_root(other, uid=uid, gid=gid)
    assert not expected.exists() and not other.exists()


def test_preflight_preparation_keeps_temporary_root_separate_from_formal_root(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    runs.chmod(0o700)
    formal_root = runs / "q03-step119"
    preflight_root = runs / ".zero-formal-q03-step119-unique"
    preflight_root.mkdir(mode=0o700)
    monkeypatch.setattr(runtime, "EXPECTED_RUN_ROOT", formal_root)
    uid = os.getuid()
    gid = pwd.getpwuid(uid).pw_gid

    runtime._prepare_run_root(
        preflight_root, uid=uid, gid=gid, model_load_preflight_only=True
    )
    assert preflight_root.is_dir() and not formal_root.exists()

    runtime._prepare_run_root(
        formal_root, uid=uid, gid=gid, model_load_preflight_only=False
    )
    assert formal_root.is_dir() and not formal_root.is_symlink()


def test_formal_input_validation_checks_sources_before_creating_run_root(tmp_path: Path, monkeypatch) -> None:
    model, parquet, manifest, source_root, run_dir, uid, gid = _make_tree(tmp_path, monkeypatch)
    run_dir.rmdir()
    runtime.validate_inputs(
        model_path=model,
        parquet_path=parquet,
        manifest_path=manifest,
        source_root=source_root,
        run_as_uid=uid,
        run_as_gid=gid,
        run_dir=run_dir,
        allow_missing_formal_run_root=True,
    )
    assert not run_dir.exists()

    (model / "tokenizer.json").chmod(0o644)
    (model / "tokenizer.json").write_bytes(b"drift")
    with pytest.raises(runtime.RuntimeContractError, match="tokenizer SHA-256 mismatch"):
        runtime.validate_inputs(
            model_path=model,
            parquet_path=parquet,
            manifest_path=manifest,
            source_root=source_root,
            run_as_uid=uid,
            run_as_gid=gid,
            run_dir=run_dir,
            allow_missing_formal_run_root=True,
        )
    assert not run_dir.exists()


def test_run_identity_rejects_wrong_primary_gid_before_root_preparation(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    runs.chmod(0o700)
    run_dir = runs / "q03-step119"
    monkeypatch.setattr(runtime, "EXPECTED_RUN_ROOT", run_dir)
    uid = os.getuid()
    gid = pwd.getpwuid(uid).pw_gid

    with pytest.raises(runtime.RuntimeContractError, match="GID differs"):
        identity = runtime._validated_run_identity(uid, gid + 1)
        runtime._prepare_run_root(
            run_dir, uid=identity.pw_uid, gid=identity.pw_gid, model_load_preflight_only=False
        )
    assert not run_dir.exists()


def test_validation_rejects_replaceable_input_parent_and_unbound_output_path(tmp_path: Path, monkeypatch) -> None:
    model, parquet, manifest, source_root, run_dir, uid, gid = _make_tree(tmp_path, monkeypatch)
    effective_parent = source_root.parent
    effective_parent.chmod(0o755)
    with pytest.raises(runtime.RuntimeContractError, match="not writable by run-as UID"):
        runtime.validate_inputs(
            model_path=model, parquet_path=parquet, manifest_path=manifest,
            source_root=source_root, run_as_uid=uid, run_as_gid=gid, run_dir=run_dir,
        )
    effective_parent.chmod(0o555)

    wrong_run_dir = tmp_path / "runs" / "wrong"
    wrong_run_dir.mkdir()
    wrong_run_dir.chmod(0o700)
    with pytest.raises(runtime.RuntimeContractError, match="exact fresh run root"):
        runtime.validate_inputs(
            model_path=model, parquet_path=parquet, manifest_path=manifest,
            source_root=source_root, run_as_uid=uid, run_as_gid=gid, run_dir=wrong_run_dir,
        )


def test_step119_result_writer_preserves_exact_rows_and_aggregates(tmp_path: Path) -> None:
    columns = {name: [] for name in results.ROW_FIELDS}
    for slot in range(128):
        values = {
            "q03_panel_slot": slot,
            "q03_task_id": f"webshop_seed_source_faithful_slot{slot:03d}_session{slot:05d}",
            "q03_sampled_session_index": slot,
            "q03_worker_seed": 1000 + slot,
            "q03_generation_seed": 2026082500 + slot,
            "q03_instruction_sha256": f"{slot:064x}",
            "q03_validation_task_score": 1.0 if slot == 127 else slot / 128,
            "q03_validation_exact_success": slot == 127,
        }
        for name in results.ROW_FIELDS:
            columns[name].append(values[name])
    # The rollout collector emits one row per active action. Repeated rows for
    # the same task must agree and collapse to one measured task.
    for name in results.ROW_FIELDS:
        columns[name].extend(columns[name])
    rows = results.collect_batch_rows(columns)
    assert len(rows) == 128
    doc = results.build_document(
        rows,
        task_manifest_sha256="a" * 64,
        parquet_sha256="b" * 64,
        experiment_id="202610251359-q03-step119-val128",
    )
    assert doc["denominator"] == doc["observed_rows"] == 128
    assert doc["aggregates"]["exact_success_count"] == 1
    path = results.write_document(tmp_path, doc)
    assert json.loads(path.read_text()) == doc
    with pytest.raises(FileExistsError):
        results.write_document(tmp_path, doc)


def test_step119_result_writer_rejects_missing_or_conflicting_rows() -> None:
    with pytest.raises(ValueError, match="each frozen slot"):
        results.build_document([], task_manifest_sha256="a" * 64, parquet_sha256="b" * 64,
                               experiment_id="202610251359-q03-step119-val128")

    columns = {name: [0, 0] for name in results.ROW_FIELDS}
    columns["q03_panel_slot"] = [0, 0]
    columns["q03_task_id"] = [
        "webshop_seed_source_faithful_slot000_session00000",
        "webshop_seed_source_faithful_slot000_session00000",
    ]
    columns["q03_sampled_session_index"] = [0, 0]
    columns["q03_worker_seed"] = [1000, 1000]
    columns["q03_generation_seed"] = [2026082500, 2026082500]
    columns["q03_instruction_sha256"] = ["a" * 64, "a" * 64]
    columns["q03_validation_task_score"] = [0.0, 1.0]
    columns["q03_validation_exact_success"] = [False, True]
    with pytest.raises(ValueError, match="disagree"):
        results.collect_batch_rows(columns)


def test_step119_result_writer_rejects_score_success_disagreement() -> None:
    columns = {name: [0] for name in results.ROW_FIELDS}
    columns.update({
        "q03_panel_slot": [0],
        "q03_task_id": ["webshop_seed_source_faithful_slot000_session00000"],
        "q03_sampled_session_index": [0],
        "q03_worker_seed": [1000],
        "q03_generation_seed": [2026082500],
        "q03_instruction_sha256": ["a" * 64],
        "q03_validation_task_score": [1.0],
        "q03_validation_exact_success": [False],
    })
    with pytest.raises(ValueError, match="disagrees with the terminal task score"):
        results.collect_batch_rows(columns)
