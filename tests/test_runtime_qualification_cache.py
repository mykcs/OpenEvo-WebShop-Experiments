from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import subprocess
import venv
import pytest
from unittest.mock import patch

from scripts.openevo_webshop import runtime_qualification


def args_for(tmp_path: Path) -> Namespace:
    fingerprint = tmp_path / "runtime-fingerprint.json"
    fingerprint.write_text('{"runtime":"stable-a"}\n', encoding="utf-8")
    for name in ("source", "data", "model"):
        (tmp_path / name).mkdir()
    return Namespace(
        fingerprint_file=fingerprint,
        cache_root=tmp_path / "qualification-cache",
        repo_root=tmp_path,
        source_root=tmp_path / "source",
        webshop_data=tmp_path / "data",
        model=tmp_path / "model",
        profile="openevo",
        runtime_python=None,
        openevo_root=tmp_path / "source",
        openevo_patch_sha=None,
        index_root=None,
        image=None,
        expected_image_digest=None,
        runtime_container=False,
        require_java=False,
        allow_dirty=False,
        package=[],
        import_name=[],
        refresh=False,
        output=None,
    )


def pass_result() -> dict:
    return {"checks": {"runtime": {"status": "pass", "errors": []}}, "failures": []}


def test_docker_tag_drift_cannot_reuse_previous_pass(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    args.image = "science:known-good"
    stable = {"status": "pass", "image_id": "sha256:old", "errors": []}
    changed = {"status": "fail", "image_id": "sha256:new", "errors": ["image identity mismatch"]}
    with (
        patch.object(runtime_qualification.envdoc, "check_docker", side_effect=[stable, stable, changed]),
        patch.object(runtime_qualification, "perform_checks", return_value=pass_result()) as checks,
    ):
        first = runtime_qualification.qualify(args)
        second = runtime_qualification.qualify(args)
        third = runtime_qualification.qualify(args)
    assert second["reused"] is True
    assert third["reused"] is False
    assert third["status"] == "FAIL"
    assert third["cache_key"] != first["cache_key"]
    assert checks.call_count == 1
    assert "live_identity.docker_image.image_id" in third["qualification_decision"]["changed_request_fields"]


def test_qualification_records_reuse_and_explicit_refresh_reason(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    with patch.object(runtime_qualification, "perform_checks", return_value=pass_result()):
        first = runtime_qualification.qualify(args)
        second = runtime_qualification.qualify(args)
        args.refresh = True
        third = runtime_qualification.qualify(args)
    assert first["qualification_decision"]["reason"] == "no_matching_pass_receipt"
    assert second["qualification_decision"]["reason"] == "exact_request_match"
    assert third["qualification_decision"]["reason"] == "explicit_refresh_requested"


def init_git_repo(path: Path) -> None:
    (path / "tracked.txt").write_text("stable\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Runtime Test",
            "-c",
            "user.email=runtime-test@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        check=True,
    )


def test_exact_runtime_fingerprint_reuses_pass(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    with patch.object(runtime_qualification, "perform_checks", return_value=pass_result()) as checks:
        first = runtime_qualification.qualify(args)
        second = runtime_qualification.qualify(args)
    assert first["status"] == "PASS"
    assert first["reused"] is False
    assert second["status"] == "PASS"
    assert second["reused"] is True
    assert checks.call_count == 1


def test_fingerprint_change_invalidates_cache(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    with patch.object(runtime_qualification, "perform_checks", return_value=pass_result()) as checks:
        runtime_qualification.qualify(args)
        args.fingerprint_file.write_text('{"runtime":"stable-b"}\n', encoding="utf-8")
        second = runtime_qualification.qualify(args)
    assert second["reused"] is False
    assert checks.call_count == 2


def test_live_runtime_identity_change_invalidates_cache(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    stable = {"runtime_python": {"version": "3.11"}, "distributions": "sha256:stable"}
    drifted = {"runtime_python": {"version": "3.11"}, "distributions": "sha256:drifted"}
    with (
        patch.object(runtime_qualification, "collect_live_identity", side_effect=[stable, stable, drifted]),
        patch.object(runtime_qualification, "perform_checks", return_value=pass_result()) as checks,
    ):
        first = runtime_qualification.qualify(args)
        second = runtime_qualification.qualify(args)
        third = runtime_qualification.qualify(args)
    assert first["reused"] is False
    assert second["reused"] is True
    assert third["reused"] is False
    assert third["cache_key"] != first["cache_key"]
    assert checks.call_count == 2


def test_live_identity_binds_runtime_git_and_declared_assets(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    init_git_repo(args.source_root)
    (args.model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (args.model / "model.safetensors.index.json").write_text('{"weight_map":{}}\n', encoding="utf-8")
    (args.model / "model-00001-of-00001.safetensors").write_bytes(b"not-a-real-model")
    (args.webshop_data / "MANIFEST.json").write_text('{"dataset":"fixture"}\n', encoding="utf-8")
    (args.webshop_data / "items_shuffle_1000.json").write_text("[]\n", encoding="utf-8")

    request = runtime_qualification.request_document(args)
    live = request["live_identity"]

    assert live["runtime_python"]["status"] == "observed"
    assert live["runtime_python"]["distributions_count"] > 0
    assert live["runtime_python"]["distributions_sha256"].startswith("sha256:")
    assert live["git"]["source"]["head"]
    assert live["git"]["source"]["dirty"] is False
    assert live["assets"]["model"]["descriptors"]["config.json"]["sha256"].startswith("sha256:")
    assert "sha256" not in live["assets"]["model"]["payloads"][0]
    assert live["assets"]["model"]["payloads"][0]["ctime_ns"] > 0
    assert live["assets"]["webshop_data"]["descriptors"]["MANIFEST.json"]["sha256"].startswith("sha256:")


def test_runtime_qualification_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime-venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
    python = runtime / "bin" / "python"
    if not python.is_symlink():
        pytest.skip("this platform creates copied rather than symlinked venv interpreters")

    args = args_for(tmp_path)
    args.runtime_python = python
    request = runtime_qualification.request_document(args)

    assert request["runtime_python"] == str(python.absolute())
    assert request["live_identity"]["runtime_python"]["prefix"] == str(runtime)

    args.require_cuda = False
    args.wandb_mode = "offline"
    args.wandb_fallback = False
    ok = {"status": "pass", "errors": []}
    with (
        patch.object(runtime_qualification.envdoc, "check_python_executable", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_git", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_runtime_assets", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_model_completeness", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_packages", return_value=ok) as packages,
        patch.object(runtime_qualification.envdoc, "check_imports", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_external_runtime_imports", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_external_source_identity", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_tokenizer", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_pip", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_java", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_docker", return_value=ok),
        patch.object(runtime_qualification.envdoc, "check_wandb", return_value=ok),
    ):
        runtime_qualification.perform_checks(args)

    assert packages.call_args.kwargs["python_executable"] == str(python.absolute())


def test_malformed_runtime_probe_is_not_accepted_as_identity(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    for output in (b"not-json\n", b"[]\n", b"{}\n"):
        malformed = subprocess.CompletedProcess(["python"], 0, stdout=output, stderr=b"")
        with patch.object(runtime_qualification, "_command", return_value=malformed):
            identity = runtime_qualification._runtime_identity(args)
        assert identity["status"] == "unavailable"


def test_source_or_data_drift_changes_live_cache_key(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    init_git_repo(args.source_root)
    data_file = args.webshop_data / "items_shuffle_1000.json"
    data_file.write_text("[]\n", encoding="utf-8")
    first = runtime_qualification.request_document(args)

    (args.source_root / "tracked.txt").write_text("drifted\n", encoding="utf-8")
    source_drift = runtime_qualification.request_document(args)
    assert source_drift["live_identity"]["git"]["source"]["dirty"] is True
    assert runtime_qualification.cache_key(source_drift) != runtime_qualification.cache_key(first)

    data_file.write_text('[{"drifted":true}]\n', encoding="utf-8")
    data_drift = runtime_qualification.request_document(args)
    assert runtime_qualification.cache_key(data_drift) != runtime_qualification.cache_key(source_drift)


def test_untracked_source_content_drift_changes_live_cache_key(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    init_git_repo(args.source_root)
    untracked = args.source_root / "new-runtime.py"
    untracked.write_text("VALUE = 1\n", encoding="utf-8")
    first = runtime_qualification.request_document(args)

    untracked.write_text("VALUE = 2\n", encoding="utf-8")
    second = runtime_qualification.request_document(args)

    assert runtime_qualification.cache_key(second) != runtime_qualification.cache_key(first)


def test_tokenizer_content_drift_changes_live_cache_key(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    tokenizer = args.model / "tokenizer.json"
    tokenizer.write_text('{"version":1}\n', encoding="utf-8")
    first = runtime_qualification.request_document(args)

    tokenizer.write_text('{"version":2}\n', encoding="utf-8")
    second = runtime_qualification.request_document(args)

    assert runtime_qualification.cache_key(second) != runtime_qualification.cache_key(first)


def test_qualification_implementation_change_invalidates_cache(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    with patch.object(runtime_qualification, "perform_checks", return_value=pass_result()) as checks:
        runtime_qualification.qualify(args)
        with patch.object(
            runtime_qualification,
            "implementation_identity",
            return_value={"qualification_sha256": "sha256:new", "environment_doctor_sha256": "sha256:new"},
        ):
            second = runtime_qualification.qualify(args)
    assert second["reused"] is False
    assert checks.call_count == 2


def test_failed_qualification_is_never_reused(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    failed = {"checks": {"runtime": {"status": "fail", "errors": ["broken"]}}, "failures": ["runtime: broken"]}
    with patch.object(runtime_qualification, "perform_checks", return_value=failed) as checks:
        first = runtime_qualification.qualify(args)
        second = runtime_qualification.qualify(args)
    assert first["status"] == "FAIL"
    assert second["status"] == "FAIL"
    assert first["reused"] is False
    assert second["reused"] is False
    assert checks.call_count == 2


def test_refresh_forces_requalification(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    with patch.object(runtime_qualification, "perform_checks", return_value=pass_result()) as checks:
        runtime_qualification.qualify(args)
        args.refresh = True
        refreshed = runtime_qualification.qualify(args)
    assert refreshed["reused"] is False
    assert checks.call_count == 2


def test_platform_profile_has_no_scientific_asset_or_wandb_requirements(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    args.profile = "platform"
    request = runtime_qualification.request_document(args)
    route = request["route"]
    assert route["require_webshop_assets"] is False
    assert route["require_model_assets"] is False
    assert route["require_external_runtime"] is False
    assert route["require_tokenizer"] is False
    assert route["packages"] == ["torch"]
    assert route["imports"] == ["torch"]
    assert "wandb" not in route["packages"]


def test_profile_change_invalidates_cache(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    with patch.object(runtime_qualification, "perform_checks", return_value=pass_result()) as checks:
        runtime_qualification.qualify(args)
        args.profile = "platform"
        second = runtime_qualification.qualify(args)
    assert second["reused"] is False
    assert checks.call_count == 2


def test_openevo_profile_does_not_skip_required_scientific_assets(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    args.model = None
    with patch.object(runtime_qualification.envdoc, "check_python_executable", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_git", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_packages", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_imports", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_external_runtime_imports", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_external_source_identity", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_pip", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_java", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_docker", return_value={"status": "pass", "errors": []}):
        result = runtime_qualification.perform_checks(args)
    assert result["checks"]["model_completeness"]["status"] == "fail"
    assert any("model" in failure for failure in result["failures"])


def test_platform_profile_does_not_call_scientific_model_or_openevo_checks(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    args.profile = "platform"
    args.webshop_data = None
    args.model = None
    args.openevo_root = None
    with patch.object(runtime_qualification.envdoc, "check_python_executable", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_git", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_packages", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_imports", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_pip", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_java", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_docker", return_value={"status": "pass", "errors": []}), patch.object(runtime_qualification.envdoc, "check_model_completeness", side_effect=AssertionError("model check must not run")), patch.object(runtime_qualification.envdoc, "check_tokenizer", side_effect=AssertionError("tokenizer check must not run")), patch.object(runtime_qualification.envdoc, "check_external_runtime_imports", side_effect=AssertionError("OpenEvo import must not run")), patch.object(runtime_qualification.envdoc, "check_external_source_identity", side_effect=AssertionError("OpenEvo identity must not run")):
        result = runtime_qualification.perform_checks(args)
    assert result["failures"] == []
    assert result["checks"]["tokenizer"]["status"] == "skip"
    assert result["checks"]["external_runtime_imports"]["status"] == "skip"


def _stub_full_validation(stack):
    """Stub expensive probes while exercising the real qualification/cache orchestration."""
    ok = {"status": "pass", "errors": []}
    names = (
        "check_python_executable",
        "check_git",
        "check_runtime_assets",
        "check_model_completeness",
        "check_packages",
        "check_imports",
        "check_external_runtime_imports",
        "check_external_source_identity",
        "check_tokenizer",
        "check_pip",
        "check_java",
        "check_docker",
    )
    return {
        name: stack.enter_context(patch.object(runtime_qualification.envdoc, name, return_value=ok))
        for name in names
    }


def _commit_source_change(source: Path) -> None:
    (source / "tracked.txt").write_text("new source behavior\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=Runtime Test",
         "-c", "user.email=runtime-test@example.invalid", "commit", "-qm", "source change"],
        check=True,
    )


def test_source_only_change_reuses_environment_but_checks_new_code(tmp_path: Path) -> None:
    from contextlib import ExitStack

    args = args_for(tmp_path)
    init_git_repo(args.source_root)
    with ExitStack() as stack:
        probes = _stub_full_validation(stack)
        first = runtime_qualification.qualify(args)
        _commit_source_change(args.source_root)
        second = runtime_qualification.qualify(args)

    assert first["status"] == second["status"] == "PASS"
    assert second["reused"] is False  # Never reuse the old source PASS.
    assert first["cache_key"] != second["cache_key"]
    assert second["environment_cache"]["status"] == "REUSED"
    assert probes["check_python_executable"].call_count == 1
    assert probes["check_packages"].call_count == 1
    assert probes["check_pip"].call_count == 1
    for name in ("check_git", "check_imports", "check_external_runtime_imports",
                 "check_external_source_identity", "check_tokenizer"):
        assert probes[name].call_count == 2, name


def test_changed_environment_fingerprint_invalidates_environment_checks(tmp_path: Path) -> None:
    from contextlib import ExitStack
    import json

    args = args_for(tmp_path)
    args.fingerprint_file.write_text(json.dumps({
        "schema": "openevo.runtime-fingerprint.v1",
        "torch": {"version": "A"},
        "openevo_core": {"revision": "src-a"},
        "model": {"content_pin": "weights-a"},
    }), encoding="utf-8")
    with ExitStack() as stack:
        probes = _stub_full_validation(stack)
        first = runtime_qualification.qualify(args)
        args.fingerprint_file.write_text(json.dumps({
            "schema": "openevo.runtime-fingerprint.v1",
            "torch": {"version": "A"},
            "openevo_core": {"revision": "src-b"},
            "model": {"content_pin": "weights-b"},
        }), encoding="utf-8")
        source_only = runtime_qualification.qualify(args)
        args.fingerprint_file.write_text(json.dumps({
            "schema": "openevo.runtime-fingerprint.v1",
            "torch": {"version": "B"},
            "openevo_core": {"revision": "src-b"},
            "model": {"content_pin": "weights-b"},
        }), encoding="utf-8")
        changed_torch = runtime_qualification.qualify(args)

    assert first["cache_key"] != source_only["cache_key"]
    assert source_only["environment_cache"]["status"] == "REUSED"
    assert changed_torch["environment_cache"]["status"] == "BUILT"
    assert probes["check_packages"].call_count == 2
    assert probes["check_pip"].call_count == 2
    assert probes["check_external_runtime_imports"].call_count == 3


def test_dependency_identity_change_invalidates_environment_proof(tmp_path: Path) -> None:
    from contextlib import ExitStack

    args = args_for(tmp_path)
    original = runtime_qualification._runtime_identity(args)
    assert original["status"] == "observed"
    changed = {**original, "distributions_sha256": "sha256:updated-dependency-metadata"}
    with ExitStack() as stack:
        probes = _stub_full_validation(stack)
        with patch.object(runtime_qualification, "_runtime_identity", side_effect=[original, changed]):
            first = runtime_qualification.qualify(args)
            second = runtime_qualification.qualify(args)

    assert first["cache_key"] != second["cache_key"]
    assert second["environment_cache"]["status"] == "BUILT"
    assert probes["check_packages"].call_count == 2
    assert probes["check_pip"].call_count == 2


def test_cached_environment_does_not_hide_new_source_import_failure(tmp_path: Path) -> None:
    from contextlib import ExitStack

    args = args_for(tmp_path)
    init_git_repo(args.source_root)
    with ExitStack() as stack:
        probes = _stub_full_validation(stack)
        probes["check_external_runtime_imports"].side_effect = [
            {"status": "pass", "errors": []},
            {"status": "fail", "errors": ["wrong OpenEvo import path"]},
            {"status": "fail", "errors": ["wrong OpenEvo import path"]},
        ]
        first = runtime_qualification.qualify(args)
        _commit_source_change(args.source_root)
        failed = runtime_qualification.qualify(args)
        repeated = runtime_qualification.qualify(args)

    assert first["status"] == "PASS"
    assert failed["environment_cache"]["status"] == "REUSED"
    assert failed["status"] == repeated["status"] == "FAIL"
    assert repeated["reused"] is False
    assert probes["check_packages"].call_count == 1
    assert probes["check_external_runtime_imports"].call_count == 3


def test_corrupt_environment_receipt_cannot_be_reused(tmp_path: Path) -> None:
    from contextlib import ExitStack
    import json

    args = args_for(tmp_path)
    init_git_repo(args.source_root)
    with ExitStack() as stack:
        probes = _stub_full_validation(stack)
        first = runtime_qualification.qualify(args)
        cache_file = Path(first["environment_cache"]["cache_path"])
        document = json.loads(cache_file.read_text(encoding="utf-8"))
        document["checks"]["packages"]["errors"] = ["tampered"]
        cache_file.write_text(json.dumps(document), encoding="utf-8")
        _commit_source_change(args.source_root)
        second = runtime_qualification.qualify(args)

    assert second["environment_cache"]["status"] == "BUILT"
    assert probes["check_packages"].call_count == 2


def test_explicit_refresh_rechecks_environment_even_with_cache(tmp_path: Path) -> None:
    from contextlib import ExitStack

    args = args_for(tmp_path)
    with ExitStack() as stack:
        probes = _stub_full_validation(stack)
        first = runtime_qualification.qualify(args)
        args.refresh = True
        refreshed = runtime_qualification.qualify(args)

    assert first["environment_cache"]["status"] == "BUILT"
    assert refreshed["environment_cache"]["status"] == "BUILT"
    assert probes["check_packages"].call_count == 2
    assert probes["check_pip"].call_count == 2


def test_dependency_manifest_change_rechecks_base_even_if_python_is_unchanged(tmp_path: Path) -> None:
    from contextlib import ExitStack

    args = args_for(tmp_path)
    init_git_repo(args.source_root)
    manifest = args.source_root / "pyproject.toml"
    manifest.write_text('[project]\ndependencies = ["foo"]\n', encoding="utf-8")
    with ExitStack() as stack:
        probes = _stub_full_validation(stack)
        first = runtime_qualification.qualify(args)
        manifest.write_text('[project]\ndependencies = ["bar"]\n', encoding="utf-8")
        second = runtime_qualification.qualify(args)

    assert first["status"] == second["status"] == "PASS"
    assert second["environment_cache"]["status"] == "BUILT"
    assert probes["check_packages"].call_count == 2
    assert probes["check_pip"].call_count == 2


def test_real_source_import_is_rechecked_on_source_edit(tmp_path: Path) -> None:
    from contextlib import ExitStack

    args = args_for(tmp_path)
    root = args.source_root
    contract = root / "src/openevo/evolution/framework/contracts.py"
    contract.parent.mkdir(parents=True)
    for package in (root / "src/openevo", root / "src/openevo/evolution",
                    root / "src/openevo/evolution/framework"):
        (package / "__init__.py").write_text("", encoding="utf-8")
    contract.write_text("VALUE = 1\n", encoding="utf-8")
    init_git_repo(root)
    subprocess.run(["git", "-C", str(root), "add", "src"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Runtime Test",
         "-c", "user.email=runtime-test@example.invalid", "commit", "-qm", "fake core"],
        check=True,
    )
    real_import = runtime_qualification.envdoc.check_external_runtime_imports
    real_identity = runtime_qualification.envdoc.check_external_source_identity
    with ExitStack() as stack:
        probes = _stub_full_validation(stack)
        probes["check_external_runtime_imports"].side_effect = real_import
        probes["check_external_source_identity"].side_effect = real_identity
        first = runtime_qualification.qualify(args)
        contract.write_text("raise RuntimeError('new Core is broken')\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "src"], check=True)
        subprocess.run(
            ["git", "-C", str(root), "-c", "user.name=Runtime Test",
             "-c", "user.email=runtime-test@example.invalid", "commit", "-qm", "broken core"],
            check=True,
        )
        second = runtime_qualification.qualify(args)

    assert first["status"] == "PASS", first["failures"]
    assert second["status"] == "FAIL"
    assert second["environment_cache"]["status"] == "REUSED"
    assert any("new Core is broken" in failure for failure in second["failures"])
    assert probes["check_packages"].call_count == 1
