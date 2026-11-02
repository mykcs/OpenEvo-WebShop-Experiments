from __future__ import annotations

import hashlib
import json
import venv
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.openevo_webshop import execution_readiness, manifest_identity, runtime_qualification


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PANEL = REPO_ROOT / "configs" / "experiment" / "manifests" / "h1.40-g2-t2-panel-v1.json"
# Historical validator pins (validate_h140_r2_preregistration.py): file and
# content hashes of the same manifest are intentionally different digests.
REAL_PANEL_FILE_SHA = "b7499d094327fa294d4e561473e6853738e5ff00e5e6dd0fe183adacf6f63b9f"
REAL_PANEL_CONTENT_SHA = "28f9bc9830f9e26fffe518372c3e6f79df76a2c020748fe8263c7cc36aae194e"


def load_oe_module() -> ModuleType:
    """Load scripts/oe as an importable module for unit-test assertions."""
    from importlib.machinery import SourceFileLoader

    oe_path = Path(__file__).resolve().parents[1] / "scripts" / "oe"
    loader = SourceFileLoader("scripts_oe", str(oe_path))
    module = ModuleType(loader.name)
    module.__file__ = str(oe_path)
    loader.exec_module(module)
    return module


def _bare_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path: Path, descriptions: list[str] = ("alpha", "beta")) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "openevo.test-manifest.v1",
        "tasks": [
            {
                "goal_idx": index,
                "task_id": f"webshop_{index:06d}",
                "sample_index": index,
                "task_description": text,
                "task_description_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            for index, text in enumerate(descriptions)
        ],
    }
    payload["manifest_sha256"] = manifest_identity.canonical_content_sha256(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def make_campaign(
    tmp_path: Path,
    *,
    campaign_id: str = "h-test",
    router_status: str = "launch-authorized",
    router_formal: bool = True,
    router_campaign_id: str | None = None,
    router_replacement_of: str | None = None,
    router_binds_release: bool = True,
    release_status: str = "launch-authorized",
    release_formal: bool = True,
    panels: dict[str, dict[str, object]] | None = None,
    extra_artifacts: list[dict[str, str]] | None = None,
    profile: dict[str, str] | str | None = None,
    upstream_patch: dict[str, str] | None = None,
) -> SimpleNamespace:
    """Build a tmp repo with preregistration, release receipt and router."""
    root = tmp_path / "repo"
    configs = root / "configs" / "experiment"
    activations = configs / "activations"
    activations.mkdir(parents=True)

    pc = configs / "path-contract.json"
    pc.write_bytes(b'{"schema":"openevo.test-path-contract.v1","paths":{}}')

    prereg = configs / "h-test.json"
    prereg_content: dict[str, object] = {
        "schema": "openevo.test.v1",
        "campaign_id": campaign_id,
        "frozen_scientific_contract": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "model_content_pin": "0" * 64,
            "evaluator": "WebShop task_score/won",
            "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
            "parser_contract": "seed3090.webshop-action-parser.v3-action-command",
            "temperature": 0,
            "max_steps": 12,
            "evaluation_panels": panels or {},
        },
    }
    prereg.write_text(json.dumps(prereg_content, indent=2), encoding="utf-8")

    release: dict[str, object] = {
        "schema": "openevo.experiment-h-test-execution-release.v1",
        "campaign_id": campaign_id,
        "status": release_status,
        "formal_task_consumption_allowed": release_formal,
        "preregistration": {
            "path": "configs/experiment/h-test.json",
            "sha256": _bare_sha(prereg),
        },
        "path_contract": {
            "path": "configs/experiment/path-contract.json",
            "sha256": _bare_sha(pc),
        },
    }
    if upstream_patch:
        release["upstream_patch"] = upstream_patch
    readiness_block: dict[str, object] = {}
    if extra_artifacts:
        readiness_block["frozen_scientific_artifacts"] = extra_artifacts
    if profile is not None:
        readiness_block["profile"] = profile
    if readiness_block:
        release["execution_readiness"] = readiness_block

    release_path = activations / "h-test-execution-release-v1.json"
    release_path.write_text(json.dumps(release, indent=2), encoding="utf-8")

    router: dict[str, object] = {
        "campaign_id": router_campaign_id or campaign_id,
        "status": router_status,
        "formal_task_consumption_allowed": router_formal,
        "preregistration": "configs/experiment/h-test.json",
    }
    if router_replacement_of:
        router["replacement_of"] = router_replacement_of
    if router_binds_release:
        router["latest_event"] = {
            "activation": "configs/experiment/activations/h-test-execution-release-v1.json",
            "release_receipt_sha256": _bare_sha(release_path),
            "preregistration": "configs/experiment/h-test.json",
            "preregistration_sha256": _bare_sha(prereg),
        }
    router_path = configs / "current-campaign.json"
    router_path.write_text(json.dumps(router, indent=2), encoding="utf-8")

    return SimpleNamespace(
        root=root,
        configs=configs,
        release_path=release_path,
        release=release,
        router_path=router_path,
        router=router,
        prereg_path=prereg,
        prereg=prereg_content,
        path_contract_path=pc,
    )


def args_for(
    tmp_path: Path,
    campaign: SimpleNamespace,
    *,
    release_path: Path | None = None,
    resume: bool = False,
    refresh: bool = False,
    dry_run: bool = False,
    output_root: Path | None = None,
    run_output_root: Path | None = None,
    profile: str = "platform",
) -> Namespace:
    fingerprint = tmp_path / "runtime-fingerprint.json"
    if not fingerprint.exists():
        fingerprint.write_text('{"runtime":"stable-a"}\n', encoding="utf-8")
    return Namespace(
        release=release_path or campaign.release_path,
        campaign=None,
        current_campaign=campaign.router_path,
        repo_root=campaign.root,
        fingerprint_file=fingerprint,
        cache_root=tmp_path / "qualification-cache",
        output_root=output_root or (tmp_path / "output"),
        run_output_root=run_output_root,
        runtime_python=None,
        source_root=None,
        webshop_data=None,
        model=None,
        openevo_root=None,
        index_root=None,
        image=None,
        expected_image_digest=None,
        artifact_root=None,
        storage_root=None,
        experiment_config=None,
        run_manifest=None,
        profile=profile,
        runtime_container=False,
        require_java=False,
        require_cuda=False,
        allow_dirty=False,
        resume=resume,
        refresh_qualification=refresh,
        dry_run=dry_run,
    )


def qualification_receipt(
    tmp_path: Path,
    *,
    status: str = "PASS",
    reused: bool = True,
    cache_key: str = "abc123",
) -> dict[str, object]:
    fingerprint = tmp_path / "runtime-fingerprint.json"
    cache_path = tmp_path / "qualification-cache" / f"{cache_key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"schema": "openevo.runtime-qualification.v1", "status": status, "cache_key": cache_key}),
        encoding="utf-8",
    )
    return {
        "schema": "openevo.runtime-qualification.v1",
        "status": status,
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "reused": reused,
        "request": {"runtime_fingerprint_sha256": execution_readiness.sha256_file(fingerprint)},
    }


def live_gate_receipt(*, status: str = "PASS") -> dict[str, object]:
    return {
        "schema": "openevo.runtime-live-gate.v1",
        "status": status,
        "checks": {},
        "failures": [],
    }


def mocked_gates(tmp_path: Path, *, reused: bool = True, live_status: str = "PASS"):
    qual = qualification_receipt(tmp_path, reused=reused)
    live = live_gate_receipt(status=live_status)
    return (
        patch.object(execution_readiness.runtime_qualification, "qualify", return_value=qual),
        patch.object(execution_readiness.runtime_live_gate, "run_gate", return_value=live),
    )


def write_profile(tmp_path: Path) -> Path:
    profile = {
        "schema": "openevo.webshop.run-profile.v1",
        "run_id": "test-run-1",
        "image": "openevo/test:latest",
        "image_id": "sha256:" + "0" * 64,
        "workspace": str(tmp_path),
        "identity": {"uid": 1001, "gid": 1001},
        "source": {
            "experiment": {"path": ".", "revision": "0" * 40},
            "openevo": {"path": ".", "revision": "0" * 40},
        },
        "paths": {"models": ".", "data": ".", "runs": ".", "cache": "."},
        "command": ["python3", "-c", "print('ran')"],
        "tracking": {"mode": "disabled"},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return path


def profile_spec(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _bare_sha(path)}


# ---------------------------------------------------------------------------
# READY path
# ---------------------------------------------------------------------------


def test_prepared_matching_runtime_is_ready(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch as qual_mock, gate_patch as gate_mock:
        result = execution_readiness.evaluate_readiness(args)

    assert result["verdict"] == "READY"
    assert result["qualification_reused"] is True
    assert qual_mock.call_count == 1
    assert gate_mock.call_count == 1
    receipt_path = Path(result["ready_receipt_path"])
    assert receipt_path.is_file()
    # Control receipts live in the control layer, not the output root.
    assert receipt_path.parent.name == "control"
    # Persisted receipt and in-memory result carry identical stages.
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["stages"] == result["stages"]
    assert persisted["stages"][-1]["stage"] == "ready-receipt"
    assert [s["stage"] for s in persisted["stages"]] == [
        "resolve-campaign",
        "current-authority",
        "frozen-scientific-artifacts",
        "scientific-identity",
        "runtime-qualification",
        "live-gate",
        "ready-receipt",
    ]
    assert persisted["authority"]["router_status"] == "launch-authorized"
    assert persisted["scientific_identity_sha256"].startswith("sha256:")


def test_repeated_readiness_does_not_collide_with_itself(tmp_path: Path) -> None:
    """Control receipts are idempotent; a second readiness run still passes."""
    campaign = make_campaign(tmp_path)
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        first = execution_readiness.evaluate_readiness(args)
        second = execution_readiness.evaluate_readiness(args)
    assert first["verdict"] == "READY"
    assert second["verdict"] == "READY"


def test_recorded_sha_without_prefix_is_normalized(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    release = json.loads(campaign.release_path.read_text(encoding="utf-8"))
    release["preregistration"]["sha256"] = release["preregistration"]["sha256"].removeprefix("sha256:")
    campaign.release_path.write_text(json.dumps(release), encoding="utf-8")
    # Rebind the router to the mutated release bytes.
    router = json.loads(campaign.router_path.read_text(encoding="utf-8"))
    router["latest_event"]["release_receipt_sha256"] = _bare_sha(campaign.release_path)
    campaign.router_path.write_text(json.dumps(router), encoding="utf-8")

    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "READY"


# ---------------------------------------------------------------------------
# Blocker 1 / 6A: current execution authority
# ---------------------------------------------------------------------------


def test_closed_campaign_blocks_historical_launch_authorized_release(tmp_path: Path) -> None:
    """Historical release launch-authorized=true; router closed -> BLOCKED before qualification."""
    campaign = make_campaign(
        tmp_path,
        router_status="closed-measurement-invalid",
        router_formal=False,
    )
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch as qual_mock, gate_patch as gate_mock:
        result = execution_readiness.evaluate_readiness(args)

    assert result["verdict"] == "BLOCKED"
    assert "closed" in result["error"]
    # Never reaches runtime qualification, the live gate, or GPU selection.
    assert qual_mock.call_count == 0
    assert gate_mock.call_count == 0
    assert not (tmp_path / "output" / "control" / "execution-readiness.json").exists()


def test_explicit_release_cannot_bypass_closed_router(tmp_path: Path) -> None:
    """Passing --release explicitly makes no difference: authority is current-state."""
    campaign = make_campaign(tmp_path, router_status="closed-result", router_formal=False)
    args = args_for(tmp_path, campaign, release_path=campaign.release_path)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED"


def test_router_formal_consumption_disallowed_blocks(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path, router_formal=False)
    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED"
    assert "formal task consumption" in result["error"]


def test_successor_router_supersedes_old_release(tmp_path: Path) -> None:
    campaign = make_campaign(
        tmp_path,
        router_campaign_id="h-successor",
        router_replacement_of="h-test",
    )
    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED"
    assert "superseded" in result["error"]


def test_router_binding_a_different_release_blocks(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    router = json.loads(campaign.router_path.read_text(encoding="utf-8"))
    router["latest_event"]["release_receipt_sha256"] = "f" * 64
    campaign.router_path.write_text(json.dumps(router), encoding="utf-8")
    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED"
    assert "superseded" in result["error"] or "different release" in result["error"]


def test_release_outside_repo_boundary_fails_closed(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    outside = tmp_path / "outside-release.json"
    outside.write_text(campaign.release_path.read_text(encoding="utf-8"), encoding="utf-8")
    args = args_for(tmp_path, campaign, release_path=outside)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED"
    assert "boundary" in result["error"]


# ---------------------------------------------------------------------------
# Blocker 2 / 6B: hash-bound execution profile
# ---------------------------------------------------------------------------


def test_hash_bound_profile_passes(tmp_path: Path) -> None:
    profile_path = write_profile(tmp_path)
    campaign = make_campaign(tmp_path, profile=profile_spec(profile_path))
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "READY"
    assert result["profile"]["sha256"] == execution_readiness.sha256_file(profile_path)
    persisted = json.loads(Path(result["ready_receipt_path"]).read_text(encoding="utf-8"))
    assert persisted["profile"]["sha256"] == result["profile"]["sha256"]


def test_reverify_keeps_bound_ready_and_live_gate_receipts_immutable(tmp_path: Path) -> None:
    profile_path = write_profile(tmp_path)
    campaign = make_campaign(tmp_path, profile=profile_spec(profile_path))
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        first = execution_readiness.evaluate_readiness(args)
        first_ready = Path(first["ready_receipt_path"])
        first_ready_bytes = first_ready.read_bytes()
        first_live = Path(json.loads(first_ready_bytes)["live_gate_receipt_path"])
        first_live_bytes = first_live.read_bytes()
        second = execution_readiness.evaluate_readiness(args)

    assert first["verdict"] == second["verdict"] == "READY"
    assert first_ready != Path(second["ready_receipt_path"])
    assert first_ready.read_bytes() == first_ready_bytes
    assert first_live != Path(second["live_gate_receipt_path"])
    assert first_live.read_bytes() == first_live_bytes


def test_profile_one_byte_mutation_is_stale(tmp_path: Path) -> None:
    profile_path = write_profile(tmp_path)
    campaign = make_campaign(tmp_path, profile=profile_spec(profile_path))
    profile_path.write_text(profile_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch as qual_mock, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "STALE"
    assert "execution profile" in result["error"]
    assert qual_mock.call_count == 0


def test_profile_command_mutation_is_stale(tmp_path: Path) -> None:
    profile_path = write_profile(tmp_path)
    campaign = make_campaign(tmp_path, profile=profile_spec(profile_path))
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["command"] = ["python3", "-c", "print('tampered')"]
    profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "STALE"


def test_profile_source_revision_mutation_is_stale(tmp_path: Path) -> None:
    profile_path = write_profile(tmp_path)
    campaign = make_campaign(tmp_path, profile=profile_spec(profile_path))
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["source"]["experiment"]["revision"] = "1" * 40
    profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "STALE"


def test_legacy_string_profile_is_not_hash_bound(tmp_path: Path) -> None:
    profile_path = write_profile(tmp_path)
    campaign = make_campaign(tmp_path, profile=str(profile_path))
    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED_NOT_PREPARED"
    assert "sha256" in result["error"]


def test_missing_profile_blocks_not_prepared(tmp_path: Path) -> None:
    campaign = make_campaign(
        tmp_path, profile={"path": "configs/experiment/absent-profile.json", "sha256": "0" * 64}
    )
    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED_NOT_PREPARED"


# ---------------------------------------------------------------------------
# Blocker 3 / 6D / 6E: canonical manifest hash semantics (real repo fixtures)
# ---------------------------------------------------------------------------


def test_manifest_identity_helper_matches_historical_validator_constants() -> None:
    """Helper output must equal the pinned EXPECTED_FILE/CONTENT_HASHES."""
    assert manifest_identity.manifest_file_sha256(REAL_PANEL) == REAL_PANEL_FILE_SHA
    assert manifest_identity.manifest_content_sha256(REAL_PANEL) == REAL_PANEL_CONTENT_SHA
    assert manifest_identity.manifest_self_consistent(REAL_PANEL)


def _campaign_with_real_panel(tmp_path: Path) -> SimpleNamespace:
    manifests = tmp_path / "repo" / "configs" / "experiment" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    panel_copy = manifests / "held-out-panel.json"
    panel_copy.write_bytes(REAL_PANEL.read_bytes())
    panels = {
        "held_out": {
            "manifest": "configs/experiment/manifests/held-out-panel.json",
            "manifest_content_sha256": REAL_PANEL_CONTENT_SHA,
        }
    }
    return make_campaign(tmp_path, panels=panels)


def test_real_manifest_content_hash_passes_with_distinct_file_hash(tmp_path: Path) -> None:
    """The panel binds canonical content; file SHA and content SHA differ on disk."""
    campaign = _campaign_with_real_panel(tmp_path)
    panel_copy = campaign.root / "configs" / "experiment" / "manifests" / "held-out-panel.json"
    assert manifest_identity.manifest_file_sha256(panel_copy) != REAL_PANEL_CONTENT_SHA
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "READY"


def test_real_manifest_reformatting_keeps_content_identity(tmp_path: Path) -> None:
    """File bytes change; canonical content unchanged -> still READY."""
    campaign = _campaign_with_real_panel(tmp_path)
    panel_copy = campaign.root / "configs" / "experiment" / "manifests" / "held-out-panel.json"
    payload = json.loads(panel_copy.read_text(encoding="utf-8"))
    panel_copy.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")
    assert manifest_identity.manifest_file_sha256(panel_copy) != REAL_PANEL_FILE_SHA
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "READY"


def test_real_manifest_content_change_is_stale(tmp_path: Path) -> None:
    campaign = _campaign_with_real_panel(tmp_path)
    panel_copy = campaign.root / "configs" / "experiment" / "manifests" / "held-out-panel.json"
    payload = json.loads(panel_copy.read_text(encoding="utf-8"))
    payload["tasks"][0]["task_description"] = "tampered description"
    panel_copy.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch as qual_mock, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "STALE"
    assert "manifest_content" in result["stale_artifacts"][0]["role"]
    assert qual_mock.call_count == 0


def test_synthetic_manifest_content_semantics(tmp_path: Path) -> None:
    manifests = tmp_path / "repo" / "configs" / "experiment" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    panel = manifests / "panel.json"
    payload = write_manifest(panel)
    panels = {
        "held_out": {
            "manifest": "configs/experiment/manifests/panel.json",
            "manifest_content_sha256": payload["manifest_sha256"],
        }
    }
    campaign = make_campaign(tmp_path, panels=panels)
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "READY"


# ---------------------------------------------------------------------------
# Blocker 4 / 6C / 6G: scientific identity binding
# ---------------------------------------------------------------------------


def _implementation_artifact(tmp_path: Path, campaign: SimpleNamespace, role: str) -> dict[str, str]:
    target = campaign.root / "scripts" / "openevo_webshop" / f"{role}_impl.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"# {role} implementation v1\n", encoding="utf-8")
    return {
        "role": role,
        "path": f"scripts/openevo_webshop/{role}_impl.py",
        "sha256": _bare_sha(target),
    }


@pytest.mark.parametrize("role", ["evaluator", "prompt", "parser"])
def test_implementation_mutation_requires_repreparation(tmp_path: Path, role: str) -> None:
    campaign = make_campaign(tmp_path, extra_artifacts=[])
    artifact = _implementation_artifact(tmp_path, campaign, role)
    campaign.release["execution_readiness"] = {"frozen_scientific_artifacts": [artifact]}
    campaign.release_path.write_text(json.dumps(campaign.release, indent=2), encoding="utf-8")
    router = json.loads(campaign.router_path.read_text(encoding="utf-8"))
    router["latest_event"]["release_receipt_sha256"] = _bare_sha(campaign.release_path)
    campaign.router_path.write_text(json.dumps(router), encoding="utf-8")

    target = campaign.root / "scripts" / "openevo_webshop" / f"{role}_impl.py"
    target.write_text(f"# {role} implementation v2 TAMPERED\n", encoding="utf-8")

    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch as qual_mock, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "REQUIRES_REPREPARATION"
    assert "scientific identity changed" in result["error"]
    assert qual_mock.call_count == 0


@pytest.mark.parametrize("role", ["model", "adapter", "checkpoint", "tokenizer", "webshop_data"])
def test_model_adapter_identity_mutation_requires_repreparation(tmp_path: Path, role: str) -> None:
    campaign = make_campaign(tmp_path, extra_artifacts=[])
    artifact = _implementation_artifact(tmp_path, campaign, role)
    campaign.release["execution_readiness"] = {"frozen_scientific_artifacts": [artifact]}
    campaign.release_path.write_text(json.dumps(campaign.release, indent=2), encoding="utf-8")
    router = json.loads(campaign.router_path.read_text(encoding="utf-8"))
    router["latest_event"]["release_receipt_sha256"] = _bare_sha(campaign.release_path)
    campaign.router_path.write_text(json.dumps(router), encoding="utf-8")

    target = campaign.root / "scripts" / "openevo_webshop" / f"{role}_impl.py"
    target.write_text("tampered-bytes\n", encoding="utf-8")

    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "REQUIRES_REPREPARATION"


def test_scientific_identity_digest_is_stable_and_recorded(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        first = execution_readiness.evaluate_readiness(args)
        second = execution_readiness.evaluate_readiness(args)
    assert first["verdict"] == second["verdict"] == "READY"
    assert first["scientific_identity_sha256"] == second["scientific_identity_sha256"]


def test_contract_patch_pin_discovery_binds_patch(tmp_path: Path) -> None:
    """A `<path>` + `<path>_sha256` pin inside the frozen contract is enforced."""
    campaign = make_campaign(tmp_path)
    patch_file = campaign.root / "scripts" / "openevo_webshop" / "patches" / "mechanism.patch"
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    patch_file.write_text("diff --git a/x b/x\n", encoding="utf-8")
    campaign.prereg["frozen_scientific_contract"]["arms"] = {
        "C1": {
            "upstream_patch": "scripts/openevo_webshop/patches/mechanism.patch",
            "upstream_patch_sha256": _bare_sha(patch_file),
        }
    }
    campaign.prereg_path.write_text(json.dumps(campaign.prereg, indent=2), encoding="utf-8")
    # Rebind release + router to the new preregistration bytes.
    campaign.release["preregistration"]["sha256"] = _bare_sha(campaign.prereg_path)
    campaign.release_path.write_text(json.dumps(campaign.release, indent=2), encoding="utf-8")
    router = json.loads(campaign.router_path.read_text(encoding="utf-8"))
    router["latest_event"]["release_receipt_sha256"] = _bare_sha(campaign.release_path)
    router["latest_event"]["preregistration_sha256"] = _bare_sha(campaign.prereg_path)
    campaign.router_path.write_text(json.dumps(router), encoding="utf-8")

    patch_file.write_text("diff --git a/x b/x TAMPERED\n", encoding="utf-8")
    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "STALE"
    assert "upstream_patch" in result["stale_artifacts"][0]["role"]


# ---------------------------------------------------------------------------
# Blocker 6I: missing artifacts; stale preregistration
# ---------------------------------------------------------------------------


def test_missing_declared_artifact_blocks_not_prepared(tmp_path: Path) -> None:
    campaign = make_campaign(
        tmp_path,
        extra_artifacts=[{"role": "evaluator", "path": "scripts/absent.py", "sha256": "0" * 64}],
    )
    args = args_for(tmp_path, campaign)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch as qual_mock, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED_NOT_PREPARED"
    assert "missing declared scientific artifact" in result["error"]
    assert result["missing_artifacts"]
    assert qual_mock.call_count == 0


def test_stale_preregistration_blocks(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    campaign.prereg_path.write_text(json.dumps({"changed": True}), encoding="utf-8")
    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "STALE"
    assert "stale artifact" in result["error"]
    assert result["stale_artifacts"]


def test_release_without_formal_permission_blocks(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path, release_formal=False)
    args = args_for(tmp_path, campaign)
    result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED_NOT_PREPARED"


# ---------------------------------------------------------------------------
# Live gate / output layering
# ---------------------------------------------------------------------------


def test_live_gate_failure_blocks(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    args = args_for(tmp_path, campaign)
    qual = qualification_receipt(tmp_path, reused=True)
    live = live_gate_receipt(status="FAIL")
    live["failures"] = ["storage: disk full"]
    with patch.object(execution_readiness.runtime_qualification, "qualify", return_value=qual):
        with patch.object(execution_readiness.runtime_live_gate, "run_gate", return_value=live):
            result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED_NOT_PREPARED"
    assert "live gate did not pass" in result["error"]


def test_live_gate_receives_control_layer_output_root(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    args = args_for(tmp_path, campaign)
    qual = qualification_receipt(tmp_path, reused=True)
    captured: list[Namespace] = []

    def fake_gate(ns: Namespace) -> dict[str, object]:
        captured.append(ns)
        return live_gate_receipt()

    with patch.object(execution_readiness.runtime_qualification, "qualify", return_value=qual):
        with patch.object(execution_readiness.runtime_live_gate, "run_gate", side_effect=fake_gate):
            result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "READY"
    assert captured[0].output_root == (tmp_path / "output" / "control").resolve()
    assert captured[0].resume is True


def test_formal_run_output_collision_fails_closed(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    run_output = tmp_path / "formal-run"
    run_output.mkdir()
    (run_output / "episode-0.json").write_text("{}", encoding="utf-8")
    args = args_for(tmp_path, campaign, run_output_root=run_output)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "BLOCKED_NOT_PREPARED"
    assert "collision" in result["error"]


def test_formal_run_output_collision_allowed_with_resume(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    run_output = tmp_path / "formal-run"
    run_output.mkdir()
    args = args_for(tmp_path, campaign, run_output_root=run_output, resume=True)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        result = execution_readiness.evaluate_readiness(args)
    assert result["verdict"] == "READY"


# ---------------------------------------------------------------------------
# Blocker 5: real runtime-qualification cache integration (no qualify mock)
# ---------------------------------------------------------------------------


def _real_qualification_args(tmp_path: Path, fingerprint: Path, *, refresh: bool = False) -> Namespace:
    return Namespace(
        fingerprint_file=fingerprint,
        cache_root=tmp_path / "qualification-cache",
        repo_root=REPO_ROOT,
        profile="platform",
        source_root=None,
        webshop_data=None,
        model=None,
        openevo_root=None,
        openevo_patch_sha=None,
        runtime_python=None,
        index_root=None,
        image=None,
        expected_image_digest=None,
        runtime_container=False,
        require_java=False,
        allow_dirty=True,
        package=["pip"],
        import_name=["json"],
        refresh=refresh,
        output=None,
    )


def test_runtime_qualification_cache_integration(tmp_path: Path) -> None:
    fingerprint = tmp_path / "runtime-fingerprint.json"
    fingerprint.write_text('{"runtime":"stable-a"}\n', encoding="utf-8")

    # Run 1: no cache -> real qualification runs and materializes the cache.
    first = runtime_qualification.qualify(_real_qualification_args(tmp_path, fingerprint))
    assert first["status"] == "PASS"
    assert first["reused"] is False
    assert Path(first["cache_path"]).is_file()

    # Run 2: same fingerprint and inputs -> real cache hit.
    second = runtime_qualification.qualify(_real_qualification_args(tmp_path, fingerprint))
    assert second["status"] == "PASS"
    assert second["reused"] is True
    assert second["cache_key"] == first["cache_key"]

    # Run 3: fingerprint drift -> cache miss, real requalification.
    fingerprint.write_text('{"runtime":"stable-b"}\n', encoding="utf-8")
    third = runtime_qualification.qualify(_real_qualification_args(tmp_path, fingerprint))
    assert third["status"] == "PASS"
    assert third["reused"] is False
    assert third["cache_key"] != first["cache_key"]


# ---------------------------------------------------------------------------
# scripts/oe run-campaign integration
# ---------------------------------------------------------------------------


def oe_campaign_namespace(tmp_path: Path, campaign: SimpleNamespace, *, dry_run: bool = True, wait: bool = False) -> Namespace:
    fingerprint = tmp_path / "runtime-fingerprint.json"
    fingerprint.write_text('{"runtime":"stable-a"}\n', encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    return Namespace(
        release=campaign.release_path,
        campaign=None,
        current_campaign=campaign.router_path,
        repo_root=campaign.root,
        fingerprint_file=fingerprint,
        cache_root=tmp_path / "qualification-cache",
        output_root=output,
        run_output_root=None,
        runtime_python=None,
        source_root=None,
        webshop_data=None,
        model=None,
        openevo_root=None,
        index_root=None,
        image=None,
        expected_image_digest=None,
        artifact_root=None,
        storage_root=None,
        experiment_config=None,
        run_manifest=None,
        profile="platform",
        runtime_container=False,
        require_java=False,
        require_cuda=False,
        allow_dirty=False,
        gpus="auto:1",
        wait=wait,
        timeout_seconds=0,
        run_id=None,
        resume=False,
        refresh_qualification=False,
        refresh_tracking_probe=False,
        dry_run=dry_run,
    )


def test_oe_run_campaign_dry_run_ready(tmp_path: Path) -> None:
    oe = load_oe_module()
    profile_path = write_profile(tmp_path)
    campaign = make_campaign(tmp_path, profile=profile_spec(profile_path))
    ns = oe_campaign_namespace(tmp_path, campaign, dry_run=True)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        rc = oe.command_run_campaign(ns)
    assert rc == 0


def test_oe_run_campaign_requires_hash_bound_profile(tmp_path: Path) -> None:
    oe = load_oe_module()
    campaign = make_campaign(tmp_path)  # no profile declared
    ns = oe_campaign_namespace(tmp_path, campaign, dry_run=True)
    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        with pytest.raises(oe.LaunchError, match="hash-bound"):
            oe.command_run_campaign(ns)


def test_oe_run_campaign_blocks_closed_campaign(tmp_path: Path) -> None:
    oe = load_oe_module()
    campaign = make_campaign(tmp_path, router_status="closed-measurement-invalid", router_formal=False)
    ns = oe_campaign_namespace(tmp_path, campaign)
    with pytest.raises(oe.LaunchError, match="BLOCKED"):
        oe.command_run_campaign(ns)


def test_oe_run_campaign_reverifies_readiness_before_gpu_bind(tmp_path: Path) -> None:
    """A READY receipt is not a reservation: revocation during wait aborts launch."""
    oe = load_oe_module()
    profile_path = write_profile(tmp_path)
    campaign = make_campaign(tmp_path, profile=profile_spec(profile_path))
    ns = oe_campaign_namespace(tmp_path, campaign, dry_run=False, wait=True)

    qual_patch, gate_patch = mocked_gates(tmp_path)
    calls = {"evaluate": 0}
    real_evaluate = execution_readiness.evaluate_readiness

    def evaluate_then_revoke(args):
        calls["evaluate"] += 1
        if calls["evaluate"] == 1:
            return real_evaluate(args)
        return {"verdict": "BLOCKED", "error": "current campaign closed while waiting", "stages": []}

    def fake_launch(launch_args, pre_launch_hook=None):
        assert pre_launch_hook is not None
        pre_launch_hook()
        return 0

    with qual_patch, gate_patch:
        with patch.object(execution_readiness, "evaluate_readiness", side_effect=evaluate_then_revoke):
            with patch.object(oe, "command_launch", side_effect=fake_launch):
                with pytest.raises(oe.LaunchError, match="revoked"):
                    oe.command_run_campaign(ns)
    assert calls["evaluate"] == 2


def test_oe_run_campaign_pending_no_gpu_preserves_config(tmp_path: Path) -> None:
    oe = load_oe_module()
    profile_path = write_profile(tmp_path)
    campaign = make_campaign(tmp_path, profile=profile_spec(profile_path))
    ns = oe_campaign_namespace(tmp_path, campaign, dry_run=False)

    config_bytes = campaign.release_path.read_bytes()
    router_bytes = campaign.router_path.read_bytes()
    profile_bytes = profile_path.read_bytes()

    qual_patch, gate_patch = mocked_gates(tmp_path)
    with qual_patch, gate_patch:
        with patch.object(
            oe, "command_launch", side_effect=oe.LaunchError("no live-idle GPU matching auto:1")
        ):
            rc = oe.command_run_campaign(ns)

    assert rc == 2
    assert campaign.release_path.read_bytes() == config_bytes
    assert campaign.router_path.read_bytes() == router_bytes
    assert profile_path.read_bytes() == profile_bytes


# ---------------------------------------------------------------------------
# Canonical wiring: qualification inputs default to the runtime fingerprint
# ---------------------------------------------------------------------------


def _known_good_fingerprint(tmp_path: Path) -> Path:
    fingerprint = tmp_path / "known-good-fingerprint.json"
    fingerprint.write_text(
        json.dumps(
            {
                "schema": "openevo.runtime-fingerprint.v1",
                "execution_mode": "host-venv",
                "container_image_required": False,
                "host_runtime_python": {"executable": "/runtime/env/runtime-python"},
                "seed_root": {"path": "/runtime/seed-root"},
                "webshop_data": {"path": "/data/webshop-1000"},
                "model": {"path": "/models/qwen"},
                "openevo_core": {"path": "/openevo/core"},
                "route": {"profile": "openevo", "requires_java": True},
            }
        ),
        encoding="utf-8",
    )
    return fingerprint


def test_qualification_inputs_default_to_fingerprint(tmp_path: Path) -> None:
    """A plain run (no per-flag overrides) reproduces the cached request."""
    fingerprint = _known_good_fingerprint(tmp_path)
    qargs = execution_readiness.build_qualification_args(
        {}, fingerprint_file=fingerprint, cache_root=tmp_path / "qc"
    )
    assert qargs.profile == "openevo"
    assert qargs.require_java is True
    assert qargs.runtime_container is False
    assert str(qargs.runtime_python) == "/runtime/env/runtime-python"
    assert str(qargs.source_root) == "/runtime/seed-root"
    assert str(qargs.webshop_data) == "/data/webshop-1000"
    assert str(qargs.model) == "/models/qwen"
    assert str(qargs.openevo_root) == "/openevo/core"
    assert qargs.image is None
    assert qargs.expected_image_digest is None


def test_explicit_qualification_overrides_win(tmp_path: Path) -> None:
    """An explicit CLI argument always beats the fingerprint default."""
    fingerprint = _known_good_fingerprint(tmp_path)
    qargs = execution_readiness.build_qualification_args(
        {},
        fingerprint_file=fingerprint,
        cache_root=tmp_path / "qc",
        profile="seed",
        require_java=False,
        runtime_python="/other/python",
        model="/other/model",
    )
    assert qargs.profile == "seed"
    assert qargs.require_java is False
    assert str(qargs.runtime_python) == "/other/python"
    assert str(qargs.model) == "/other/model"
    # Unset fields still fall through to the fingerprint.
    assert str(qargs.webshop_data) == "/data/webshop-1000"


def test_qualification_keeps_virtualenv_python_symlink(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime-venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
    python = runtime / "bin" / "python"
    if not python.is_symlink():
        pytest.skip("this platform creates copied rather than symlinked venv interpreters")
    fingerprint = tmp_path / "runtime.json"
    fingerprint.write_text("{}\n", encoding="utf-8")

    qargs = execution_readiness.build_qualification_args(
        {},
        fingerprint_file=fingerprint,
        cache_root=tmp_path / "cache",
        runtime_python=python,
    )

    assert qargs.runtime_python == python.absolute()


def test_live_gate_keeps_virtualenv_python_symlink(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime-venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
    python = runtime / "bin" / "python"
    if not python.is_symlink():
        pytest.skip("this platform creates copied rather than symlinked venv interpreters")

    args = execution_readiness.build_live_gate_args(
        tmp_path / "qualification.json",
        output_root=tmp_path / "out",
        runtime_python=python,
        cache_root=tmp_path / "cache",
        artifact_root=tmp_path / "artifacts",
        storage_root=tmp_path / "storage",
    )

    assert args.runtime_python == python.absolute()


def test_route_less_fingerprint_falls_back_to_legacy_defaults(tmp_path: Path) -> None:
    """A fingerprint without route/asset info preserves the old defaults."""
    fingerprint = tmp_path / "bare.json"
    fingerprint.write_text('{"runtime": "stable-a"}\n', encoding="utf-8")
    qargs = execution_readiness.build_qualification_args(
        {}, fingerprint_file=fingerprint, cache_root=tmp_path / "qc"
    )
    assert qargs.profile == "full"
    assert qargs.require_java is False
    assert qargs.runtime_container is False
    assert qargs.runtime_python is None
    assert qargs.model is None
