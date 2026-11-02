from __future__ import annotations

import importlib.util
import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
import venv
from concurrent.futures import ThreadPoolExecutor
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "oe"
SPEC = importlib.util.spec_from_loader("openevo_launcher", SourceFileLoader("openevo_launcher", str(MODULE_PATH)))
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_gpu_selector_requires_authorized_uuids() -> None:
    cards = [
        {"index": 0, "uuid": "GPU-aaa", "memory_used_mib": 0, "memory_total_mib": 24576, "utilization_gpu_percent": 0},
        {"index": 1, "uuid": "GPU-bbb", "memory_used_mib": 0, "memory_total_mib": 24576, "utilization_gpu_percent": 0},
    ]
    assert launcher.select_gpu_uuids("auto:1", ["GPU-bbb"], cards) == ["GPU-bbb"]
    with pytest.raises(launcher.LaunchError, match="ordinal"):
        launcher.select_gpu_uuids("0", ["GPU-aaa"], cards)
    with pytest.raises(launcher.LaunchError, match="not authorized"):
        launcher.select_gpu_uuids("GPU-aaa", ["GPU-bbb"], cards)


def test_auto_selector_skips_busy_authorized_gpu() -> None:
    cards = [
        {"index": 0, "uuid": "GPU-aaa", "memory_used_mib": 128, "memory_total_mib": 24576, "utilization_gpu_percent": 0, "compute_apps": [], "device_handles": []},
        {"index": 1, "uuid": "GPU-bbb", "memory_used_mib": 0, "memory_total_mib": 24576, "utilization_gpu_percent": 0, "compute_apps": [], "device_handles": []},
        {"index": 2, "uuid": "GPU-ccc", "memory_used_mib": 0, "memory_total_mib": 24576, "utilization_gpu_percent": 0, "compute_apps": [], "device_handles": []},
    ]
    assert launcher.select_gpu_uuids(
        "auto:2",
        ["GPU-aaa", "GPU-bbb", "GPU-ccc"],
        cards,
        max_memory_mib=16,
        max_utilization=0,
    ) == ["GPU-bbb", "GPU-ccc"]


def test_auto_selector_fails_only_when_live_idle_capacity_is_insufficient() -> None:
    cards = [
        {"index": 0, "uuid": "GPU-aaa", "memory_used_mib": 128, "memory_total_mib": 24576, "utilization_gpu_percent": 0, "compute_apps": [], "device_handles": []},
        {"index": 1, "uuid": "GPU-bbb", "memory_used_mib": 0, "memory_total_mib": 24576, "utilization_gpu_percent": 0, "compute_apps": [], "device_handles": []},
    ]
    with pytest.raises(launcher.LaunchError, match="authorized eligible GPU count under exclusive-live-idle is 1; requested 2"):
        launcher.select_gpu_uuids(
            "auto:2",
            ["GPU-aaa", "GPU-bbb"],
            cards,
            max_memory_mib=16,
            max_utilization=0,
        )


def test_auto_selector_allows_capacity_colocation_with_resolved_owner() -> None:
    card = {"index": 0, "uuid": "GPU-aaa", "memory_used_mib": 10000, "memory_total_mib": 24576,
            "utilization_gpu_percent": 50, "compute_apps": [], "device_handles": [],
            "gpu_owner_containers": [{"container_id": "abc", "container_name": "peer", "image": "img"}],
            "gpu_owner_query_error": None}
    assert launcher.select_gpu_uuids("auto:1", ["GPU-aaa"], [card], resource_mode="co-resident-capacity",
                                     minimum_free_memory_mib=12000, max_utilization=100) == ["GPU-aaa"]


def test_auto_selector_capacity_colocation_fails_closed_on_unknown_owner() -> None:
    card = {"index": 0, "uuid": "GPU-aaa", "memory_used_mib": 10000, "memory_total_mib": 24576,
            "utilization_gpu_percent": 50, "compute_apps": [], "device_handles": [],
            "gpu_owner_containers": [], "gpu_owner_query_error": "docker unavailable"}
    with pytest.raises(launcher.LaunchError, match="authorized eligible GPU count under co-resident-capacity is 0; requested 1"):
        launcher.select_gpu_uuids("auto:1", ["GPU-aaa"], [card], resource_mode="co-resident-capacity",
                                  minimum_free_memory_mib=12000, max_utilization=100)


def test_explicit_uuid_selector_preserves_exact_mapping_even_if_busy() -> None:
    cards = [
        {"index": 0, "uuid": "GPU-aaa", "memory_used_mib": 128, "memory_total_mib": 24576, "utilization_gpu_percent": 0, "compute_apps": [], "device_handles": []},
        {"index": 1, "uuid": "GPU-bbb", "memory_used_mib": 0, "memory_total_mib": 24576, "utilization_gpu_percent": 0, "compute_apps": [], "device_handles": []},
    ]
    assert launcher.select_gpu_uuids(
        "GPU-aaa",
        ["GPU-aaa", "GPU-bbb"],
        cards,
        max_memory_mib=16,
        max_utilization=0,
    ) == ["GPU-aaa"]
    eligible, failures = launcher.idle_cards(cards, ["GPU-aaa"], max_memory_mib=16, max_utilization=0)
    assert not eligible
    assert failures == ["GPU-aaa:memory=128MiB"]


def test_idle_gate_rejects_hidden_or_busy_gpu() -> None:
    cards = [{
        "uuid": "GPU-aaa",
        "memory_used_mib": 128,
        "utilization_gpu_percent": 0,
        "compute_apps": [],
        "device_handles": [],
    }]
    eligible, failures = launcher.idle_cards(cards, ["GPU-aaa"], max_memory_mib=16, max_utilization=0)
    assert not eligible
    assert failures == ["GPU-aaa:memory=128MiB"]


def test_idle_gate_rejects_unresolved_compute_app_query() -> None:
    cards = [{
        "uuid": "GPU-aaa",
        "memory_used_mib": 0,
        "utilization_gpu_percent": 0,
        "compute_apps": [],
        "compute_apps_query_error": "permission denied",
        "device_handles": [],
    }]
    eligible, failures = launcher.idle_cards(cards, ["GPU-aaa"], max_memory_mib=16, max_utilization=0)
    assert not eligible
    assert failures == ["GPU-aaa:compute-app-query-failed"]


def test_idle_gate_rejects_owner_container_and_auto_selector_skips_it() -> None:
    owned_card = _idle_card(gpu_owner_containers=[{"container_name": "owner-task"}], gpu_owner_query_error=None)
    eligible, failures = launcher.idle_cards(
        [owned_card], ["GPU-aaa"], max_memory_mib=16, max_utilization=0
    )
    assert not eligible
    assert failures == ["GPU-aaa:gpu-owner-container"]
    assert launcher.select_gpu_uuids(
        "auto:1", ["GPU-aaa", "GPU-bbb"], [owned_card, _idle_card(uuid="GPU-bbb")],
        max_memory_mib=16, max_utilization=0,
    ) == ["GPU-bbb"]


def test_idle_gate_rejects_unresolved_gpu_owner_query() -> None:
    eligible, failures = launcher.idle_cards(
        [_idle_card(gpu_owner_containers=[], gpu_owner_query_error="docker unavailable")],
        ["GPU-aaa"], max_memory_mib=16, max_utilization=0,
    )
    assert not eligible
    assert failures == ["GPU-aaa:gpu-owner-query-failed"]


def _monitoring_handle() -> dict[str, str]:
    return {
        "pid": "123",
        "fd": "9",
        "device": "/dev/nvidia0",
        "user": "root",
        "comm": "nvitop",
        "exe": "/usr/local/bin/python3",
        "cmdline_programs": "nvitop",
        "parent_pid": "1",
        "cgroup": "0::/",
    }


def _idle_card(**updates: object) -> dict[str, object]:
    card: dict[str, object] = {
        "uuid": "GPU-aaa",
        "memory_used_mib": 2,
        "utilization_gpu_percent": 0,
        "compute_apps": [],
        "device_handles": [],
    }
    card.update(updates)
    return card


def test_idle_gate_accepts_verified_monitor_only_nvitop_handle() -> None:
    eligible, failures = launcher.idle_cards(
        [_idle_card(device_handles=[_monitoring_handle()])],
        ["GPU-aaa"],
        max_memory_mib=16,
        max_utilization=0,
    )
    assert eligible
    assert failures == []


def test_idle_gate_rejects_unknown_device_handle() -> None:
    eligible, failures = launcher.idle_cards(
        [_idle_card(device_handles=[{"pid": "123", "fd": "9", "device": "/dev/nvidia0"}])],
        ["GPU-aaa"],
        max_memory_mib=16,
        max_utilization=0,
    )
    assert not eligible
    assert failures == ["GPU-aaa:device-handle"]


def test_idle_gate_rejects_compute_like_device_handle() -> None:
    handle = _monitoring_handle() | {"comm": "python3", "cmdline_programs": "worker.py"}
    eligible, failures = launcher.idle_cards(
        [_idle_card(device_handles=[handle])],
        ["GPU-aaa"],
        max_memory_mib=16,
        max_utilization=0,
    )
    assert not eligible
    assert failures == ["GPU-aaa:device-handle"]


def test_monitor_handle_never_overrides_memory_gate() -> None:
    eligible, failures = launcher.idle_cards(
        [_idle_card(memory_used_mib=17, device_handles=[_monitoring_handle()])],
        ["GPU-aaa"],
        max_memory_mib=16,
        max_utilization=0,
    )
    assert not eligible
    assert failures == ["GPU-aaa:memory=17MiB"]


def test_monitor_handle_never_overrides_utilization_gate() -> None:
    eligible, failures = launcher.idle_cards(
        [_idle_card(utilization_gpu_percent=1, device_handles=[_monitoring_handle()])],
        ["GPU-aaa"],
        max_memory_mib=16,
        max_utilization=0,
    )
    assert not eligible
    assert failures == ["GPU-aaa:utilization=1%"]


def test_monitor_handle_never_overrides_compute_app_gate() -> None:
    eligible, failures = launcher.idle_cards(
        [_idle_card(compute_apps=["456, worker, 100 MiB"], device_handles=[_monitoring_handle()])],
        ["GPU-aaa"],
        max_memory_mib=16,
        max_utilization=0,
    )
    assert not eligible
    assert failures == ["GPU-aaa:compute-apps"]


def test_device_handle_classification_failure_fails_closed() -> None:
    handle = _monitoring_handle() | {"classification_error": "proc disappeared"}
    eligible, failures = launcher.idle_cards(
        [_idle_card(device_handles=[handle])],
        ["GPU-aaa"],
        max_memory_mib=16,
        max_utilization=0,
    )
    assert not eligible
    assert failures == ["GPU-aaa:device-handle-classification-failed"]


def test_snapshot_preserves_monitoring_and_blocking_handle_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = _monitoring_handle()
    unknown = {"pid": "456", "fd": "10", "device": "/dev/nvidia0"}
    responses = iter(
        [
            SimpleNamespace(stdout="0, GPU-aaa, 2, 24576, 0\n"),
            SimpleNamespace(stdout="", stderr="", returncode=0),
        ]
    )
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(launcher, "_run", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(launcher, "visible_device_handles", lambda _: [monitor, unknown])
    monkeypatch.setattr(launcher, "running_gpu_container_owners", lambda: ({}, None))

    card = launcher.snapshot_gpus()[0]

    assert card["device_handles"] == [monitor, unknown]
    assert card["monitoring_handles"] == [monitor]
    assert card["blocking_device_handles"] == [unknown]


def _tracking_profile(tmp_path: Path, run_id: str = "cell-001") -> dict:
    (tmp_path / "cache").mkdir(exist_ok=True)
    return {
        "run_id": run_id,
        "workspace": str(tmp_path),
        "paths": {"cache": "cache"},
        "tracking": {
            "mode": "online",
            "on_error": "fallback_offline",
            "probe_epoch": "campaign-execution-001",
            "entity": "entity",
            "project": "project",
        },
        "container": {"network": "bridge"},
    }


def test_wandb_probe_failure_is_reused_across_same_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _tracking_profile(tmp_path)
    monkeypatch.setenv("WANDB_API_KEY", "do-not-persist-this-secret")
    calls: list[list[str]] = []

    def fake_run(command, *, check=True, capture=True):
        calls.append(command)
        return SimpleNamespace(returncode=7, stdout="", stderr="network failed")

    monkeypatch.setattr(launcher, "_run", fake_run)
    first = launcher.choose_effective_tracking(profile, "sha256:image")
    profile_second = {**profile, "run_id": "cell-002"}
    second = launcher.choose_effective_tracking(profile_second, "sha256:image")

    assert first[0] == "offline"
    assert first[3] is False
    assert second[0] == "offline"
    assert second[3] is True
    assert len(calls) == 1
    image_index = calls[0].index("sha256:image")
    assert calls[0][image_index - 2:image_index] == ["--entrypoint", "python3"]
    assert calls[0][image_index + 1:] == [
        "-c",
        "import wandb; api=wandb.Api(timeout=10); api.viewer; print('wandb-preflight-pass')",
    ]
    receipt = Path(first[2])
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert "do-not-persist-this-secret" not in receipt.read_text(encoding="utf-8")


def test_wandb_probe_refresh_is_deliberate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _tracking_profile(tmp_path)
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    calls: list[list[str]] = []

    def fake_run(command, *, check=True, capture=True):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="wandb-preflight-pass", stderr="")

    monkeypatch.setattr(launcher, "_run", fake_run)
    first = launcher.choose_effective_tracking(profile, "sha256:image")
    reused = launcher.choose_effective_tracking(profile, "sha256:image")
    refreshed = launcher.choose_effective_tracking(profile, "sha256:image", refresh=True)

    assert first[0] == "online" and first[3] is False
    assert reused[0] == "online" and reused[3] is True
    assert refreshed[0] == "online" and refreshed[3] is False
    assert len(calls) == 2


def test_replace_tokens_preserves_argv_shape() -> None:
    rendered = launcher.replace_tokens(
        ["python", "--output", "${RUN_DIR}/result.json", "--data=${DATA_DIR}"],
        {"${RUN_DIR}": "/runs/x", "${DATA_DIR}": "/data", "${MODELS_DIR}": "/models", "${EXPERIMENT_DIR}": "/exp", "${OPENEVO_DIR}": "/evo"},
    )
    assert rendered == ["python", "--output", "/runs/x/result.json", "--data=/data"]


def test_resolve_image_identity_rejects_a_drifted_mutable_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = {
        "image": "experiment-host/openevo-scientific:openevo-py311-205cfd8",
        "image_id": "sha256:" + "a" * 64,
    }
    monkeypatch.setattr(launcher, "image_id", lambda _: "sha256:" + "b" * 64)

    with pytest.raises(launcher.LaunchError, match="image identity mismatch"):
        launcher.resolve_image_identity(profile)


def test_resolve_image_identity_returns_the_frozen_local_id_on_an_exact_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "sha256:" + "a" * 64
    profile = {"image": "image:provenance", "image_id": expected}
    monkeypatch.setattr(launcher, "image_id", lambda _: expected)

    assert launcher.resolve_image_identity(profile) == expected


def test_resolve_image_identity_falls_back_to_the_frozen_id_when_reference_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "sha256:" + "a" * 64
    profile = {"image": "image:provenance", "image_id": expected}
    inspected: list[str] = []

    def inspect(image: str) -> str:
        inspected.append(image)
        if image == profile["image"]:
            raise launcher.subprocess.CalledProcessError(1, ["docker", "image", "inspect", image])
        return expected

    monkeypatch.setattr(launcher, "image_id", inspect)

    assert launcher.resolve_image_identity(profile) == expected
    assert inspected == [profile["image"], expected]


def test_resolve_image_identity_fails_if_reference_and_frozen_id_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "sha256:" + "a" * 64
    profile = {"image": "image:provenance", "image_id": expected}

    def unavailable(image: str) -> str:
        raise launcher.subprocess.CalledProcessError(1, ["docker", "image", "inspect", image])

    monkeypatch.setattr(launcher, "image_id", unavailable)

    with pytest.raises(launcher.LaunchError, match="frozen image ID unavailable"):
        launcher.resolve_image_identity(profile)


def test_profile_requires_a_well_formed_immutable_image_id(tmp_path: Path) -> None:
    profile = {
        "schema": "openevo.webshop.run-profile.v1",
        "run_id": "image-contract-test",
        "image": "experiment-host/openevo-scientific:openevo-py311-205cfd8",
        "image_id": "not-a-local-image-id",
        "workspace": str(tmp_path),
        "identity": {"uid": 1001, "gid": 1001},
        "source": {
            "experiment": {"path": "experiment", "revision": "a" * 40},
            "openevo": {"path": "openevo", "revision": "b" * 40},
        },
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "command": ["python3", "-c", "print('ok')"],
        "tracking": {"mode": "offline"},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(launcher.LaunchError, match="image_id"):
        launcher.load_profile(path)


def test_profile_rejects_launcher_private_runtime_snapshot_injection(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "schema": "openevo.webshop.run-profile.v1",
        "_runtime_asset_snapshots": [{"source": "/etc/passwd", "target": "/workspace/x"}],
    }), encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="launcher-private"):
        launcher.load_profile(path)


def test_tag_drift_stops_before_gpu_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = {
        "run_id": "image-drift-test",
        "image": "experiment-host/openevo-scientific:openevo-py311-205cfd8",
        "image_id": "sha256:" + "a" * 64,
        "workspace": str(tmp_path),
        "identity": {"uid": 1001, "gid": 1001},
        "source": {},
        "paths": {},
        "command": ["python3", "-c", "print('ok')"],
        "tracking": {"mode": "offline"},
    }
    args = SimpleNamespace(
        profile=tmp_path / "profile.json",
        timeout_seconds=0,
        run_id=None,
        wait=False,
        resume=False,
        refresh_tracking_probe=False,
        gpus="auto:1",
    )
    monkeypatch.setattr(launcher, "load_profile", lambda _: profile)
    monkeypatch.setattr(launcher, "validate_sources_and_paths", lambda _: None)
    monkeypatch.setattr(launcher, "image_id", lambda _: "sha256:" + "b" * 64)
    monkeypatch.setattr(launcher, "snapshot_gpus", lambda: pytest.fail("GPU selection must not run"))

    with pytest.raises(launcher.LaunchError, match="image identity mismatch"):
        launcher.command_launch(args)


def test_docker_command_binds_selected_uuids_only(tmp_path: Path) -> None:
    for relative in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / relative).mkdir()
    profile = {
        "run_id": "smoke-001",
        "image": "experiment-host/openevo-scientific:openevo-py311-205cfd8",
        "workspace": str(tmp_path),
        "identity": {"uid": 1001, "gid": 1001},
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "command": ["python3", "-c", "print('ok')"],
        "tracking": {"mode": "offline"},
    }
    command = launcher.docker_command(profile, ["GPU-aaa"], "sha256:image", "offline")
    joined = " ".join(command)
    assert "--gpus device=GPU-aaa" in joined
    assert "--gpus all" not in joined
    multi = launcher.docker_command(profile, ["GPU-aaa", "GPU-bbb"], "sha256:image", "offline")
    gpu_index = multi.index("--gpus")
    assert multi[gpu_index + 1] == '"device=GPU-aaa,GPU-bbb"'
    assert "docker.sock" not in joined
    assert "--user 1001:1001" in joined
    image_index = command.index("sha256:image")
    assert command[image_index - 2:image_index] == ["--entrypoint", "python3"]
    assert command[image_index + 1:] == ["-c", "print('ok')"]
    assert "experiment-host/openevo-scientific:openevo-py311-205cfd8" not in command


def test_docker_command_makes_an_arbitrary_profile_executable_the_entrypoint(tmp_path: Path) -> None:
    for relative in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / relative).mkdir()
    profile = {
        "run_id": "entrypoint-001",
        "workspace": str(tmp_path),
        "identity": {"uid": 1001, "gid": 1001},
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "command": ["bash", "-lc", "echo ok"],
        "tracking": {"mode": "offline"},
    }

    command = launcher.docker_command(profile, ["GPU-aaa"], "sha256:image", "offline")

    image_index = command.index("sha256:image")
    assert command[image_index - 2:image_index] == ["--entrypoint", "bash"]
    assert command[image_index + 1:] == ["-lc", "echo ok"]


def test_docker_command_serializes_mount_modes_and_scoped_git_trust(tmp_path: Path) -> None:
    for relative in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / relative).mkdir()
    profile = {
        "run_id": "mount-grammar-001",
        "workspace": str(tmp_path),
        "identity": {"uid": 1001, "gid": 1001},
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "command": ["true"],
        "tracking": {"mode": "offline"},
    }
    command = launcher.docker_command(profile, ["GPU-aaa"], "sha256:image", "offline")
    mounts = [command[index + 1] for index, token in enumerate(command) if token == "--mount"]
    assert mounts == [
        f"type=bind,src={tmp_path / 'experiment'},dst=/workspace/src/openevo-experiment,readonly",
        f"type=bind,src={tmp_path / 'openevo'},dst=/workspace/src/openevo-core,readonly",
        f"type=bind,src={tmp_path / 'models'},dst=/workspace/models,readonly",
        f"type=bind,src={tmp_path / 'data'},dst=/workspace/data,readonly",
        f"type=bind,src={tmp_path / 'runs'},dst=/workspace/runs",
        f"type=bind,src={tmp_path / 'cache'},dst=/workspace/cache",
    ]
    env = [command[index + 1] for index, token in enumerate(command) if token == "--env"]
    assert "GIT_CONFIG_COUNT=2" in env
    assert "GIT_CONFIG_KEY_0=safe.directory" in env
    assert "GIT_CONFIG_VALUE_0=/workspace/src/openevo-experiment" in env
    assert "GIT_CONFIG_KEY_1=safe.directory" in env
    assert "GIT_CONFIG_VALUE_1=/workspace/src/openevo-core" in env
    assert not any("safe.directory=*" in value for value in env)


def test_runtime_assets_are_content_verified_staged_and_mounted_read_only(tmp_path: Path) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    (source / "worker.py").write_text("print('pinned')\n")
    expected = launcher.runtime_asset_digest(source, "dir")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "run_id": "runtime-snapshot-001",
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{
            "source": "runtime-source",
            "target": "/workspace/public-seed",
            "expected_type": "dir",
            "sha256": expected,
        }],
    }
    snapshots = launcher.stage_runtime_assets(profile)

    assert snapshots[0]["target"] == "/workspace/public-seed"
    staged = Path(snapshots[0]["source"])
    assert staged.is_dir()
    assert launcher.runtime_asset_digest(staged, "dir") == expected
    (staged / "worker.py").chmod(0o644)
    launcher.stage_runtime_assets(profile)
    assert stat.S_IMODE((staged / "worker.py").stat().st_mode) == 0o444
    assert stat.S_IMODE(staged.stat().st_mode) == 0o555
    (source / "worker.py").write_text("changed after staging\n")
    assert (staged / "worker.py").read_text() == "print('pinned')\n"
    profile.update({
        "identity": {"uid": 1001, "gid": 1001},
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "command": ["true"],
        "tracking": {"mode": "offline"},
        "_runtime_asset_snapshots": snapshots,
    })
    command = launcher.docker_command(profile, ["GPU-aaa"], "sha256:image", "offline")
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    assert f"type=bind,src={staged},dst=/workspace/public-seed,readonly" in mounts


def test_runtime_asset_docker_visibility_probe_is_cpu_only_and_hashes_daemon_view(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging_root = tmp_path / "assets"
    staging_root.mkdir()
    source = staging_root / "sealed-tree"
    source.mkdir()
    (source / "worker.py").write_text("print('pinned')\n")
    expected = launcher.runtime_asset_digest(source, "dir")
    profile = {"workspace": str(tmp_path), "runtime_asset_staging_root": "assets",
               "_runtime_asset_snapshots": [{
        "source": str(source), "target": "/workspace/public-seed",
        "expected_type": "dir", "sha256": expected,
    }]}
    commands: list[list[str]] = []
    host_sources: list[str] = []

    def fake_run(command, *, check=True, capture=True):
        commands.append(command)
        script_index = command.index("-c")
        mount = command[command.index("--mount") + 1]
        host_source = mount.split("src=", 1)[1].split(",dst=", 1)[0]
        host_sources.append(host_source)
        return subprocess.run(
            [sys.executable, "-c", command[script_index + 1],
             str(source), command[-1]],
            check=False, text=True, capture_output=True,
        )

    monkeypatch.setattr(launcher, "_run", fake_run)
    launcher.verify_runtime_assets_visible_to_docker(profile, "sha256:" + "a" * 64)

    assert host_sources == [str(source)]
    assert len(commands) == 1
    command = commands[0]
    assert command[1:5] == ["run", "--pull=never", "--rm", "--network"]
    assert "none" in command
    assert "--read-only" in command
    assert "--gpus" not in command
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "readonly" in command[command.index("--mount") + 1]


def test_runtime_asset_docker_visibility_failure_uses_no_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging_root = tmp_path / "assets"
    staging_root.mkdir()
    source = staging_root / "sealed-file"
    source.write_text("pinned\n")
    expected = launcher.runtime_asset_digest(source, "file")
    profile = {"workspace": str(tmp_path), "runtime_asset_staging_root": "assets",
               "_runtime_asset_snapshots": [{
        "source": str(source), "target": "/workspace/pinned-file",
        "expected_type": "file", "sha256": expected,
    }]}
    calls: list[list[str]] = []

    def failed_run(command, *, check=True, capture=True):
        calls.append(command)
        return SimpleNamespace(returncode=125, stdout="", stderr="bind source missing")

    monkeypatch.setattr(launcher, "_run", failed_run)
    with pytest.raises(launcher.LaunchError, match="not exactly visible to Docker daemon before GPU selection"):
        launcher.verify_runtime_assets_visible_to_docker(profile, "sha256:" + "b" * 64)
    assert len(calls) == 1
    assert "--gpus" not in calls[0]
    assert "--pull=never" in calls[0]


@pytest.mark.parametrize("stdout", ["", "probe log\\n" + "c" * 64])
def test_runtime_asset_docker_visibility_rejects_missing_or_noisy_hash_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout: str
) -> None:
    staging_root = tmp_path / "assets"
    staging_root.mkdir()
    source = staging_root / "sealed-file"
    source.write_text("pinned\\n")
    expected = launcher.runtime_asset_digest(source, "file")
    profile = {"workspace": str(tmp_path), "runtime_asset_staging_root": "assets",
               "_runtime_asset_snapshots": [{
        "source": str(source), "target": "/workspace/pinned-file",
        "expected_type": "file", "sha256": expected,
    }]}
    monkeypatch.setattr(
        launcher, "_run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )
    with pytest.raises(launcher.LaunchError, match="not exactly visible to Docker daemon"):
        launcher.verify_runtime_assets_visible_to_docker(profile, "sha256:" + "c" * 64)


def test_runtime_asset_docker_visibility_wraps_missing_docker_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging_root = tmp_path / "assets"
    staging_root.mkdir()
    source = staging_root / "sealed-file"
    source.write_text("pinned\\n")
    expected = launcher.runtime_asset_digest(source, "file")
    profile = {"workspace": str(tmp_path), "runtime_asset_staging_root": "assets",
               "_runtime_asset_snapshots": [{
        "source": str(source), "target": "/workspace/pinned-file",
        "expected_type": "file", "sha256": expected,
    }]}

    def missing_docker(*_args, **_kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(launcher, "_run", missing_docker)
    with pytest.raises(launcher.LaunchError, match="visibility probe could not run"):
        launcher.verify_runtime_assets_visible_to_docker(profile, "sha256:" + "c" * 64)


def test_runtime_asset_docker_visibility_rejects_symlink_before_resolving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging_root = tmp_path / "assets"
    staging_root.mkdir()
    target = staging_root / "sealed-file"
    target.write_text("pinned\\n")
    link = staging_root / "runtime-link"
    link.symlink_to(target)
    profile = {"workspace": str(tmp_path), "runtime_asset_staging_root": "assets",
               "_runtime_asset_snapshots": [{
        "source": str(link), "target": "/workspace/pinned-file",
        "expected_type": "file", "sha256": launcher.runtime_asset_digest(target, "file"),
    }]}
    monkeypatch.setattr(launcher, "_run", lambda *_args, **_kwargs: pytest.fail("Docker must not run"))
    with pytest.raises(launcher.LaunchError, match="not link-free before Docker probe"):
        launcher.verify_runtime_assets_visible_to_docker(profile, "sha256:" + "c" * 64)


def test_runtime_asset_snapshot_preserves_executable_mode_without_write_bits(tmp_path: Path) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    script = source / "worker"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o500)
    source.chmod(0o700)
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{"source": "runtime-source", "target": "/workspace/public-seed",
                            "expected_type": "dir", "sha256": launcher.runtime_asset_digest(source, "dir")}],
    }
    snapshot = Path(launcher.stage_runtime_assets(profile)[0]["source"])
    assert stat.S_IMODE((snapshot / "worker").stat().st_mode) == 0o555
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o555
    assert (snapshot / "worker").stat().st_mode & stat.S_IROTH
    assert (snapshot / "worker").stat().st_mode & stat.S_IXOTH
    assert launcher.runtime_asset_digest(snapshot, "dir") == profile["runtime_assets"][0]["sha256"]


def test_host_orchestrator_can_stage_sealed_runtime_assets_outside_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (workspace / name).mkdir()
    source = workspace / "seed-source"
    source.mkdir()
    (source / "worker.py").write_text("print('pinned')\n")
    tmpfs = tmp_path / "tmpfs"
    tmpfs.mkdir()
    monkeypatch.setattr(launcher, "HOST_TMPFS_ROOT", tmpfs)
    monkeypatch.setattr(launcher, "_validate_sealed_tmpfs_ancestry", lambda _path: None)
    profile = {
        "workspace": str(workspace),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {name: name for name in ("models", "data", "runs", "cache")},
        "runtime_asset_staging_root": str(tmpfs / "q03-step119"),
        "runtime_assets": [{
            "source": "seed-source", "target": "/workspace/public-seed",
            "expected_type": "dir", "sha256": launcher.runtime_asset_digest(source, "dir"),
        }],
        "execution": {"mode": "host-orchestrator", "runtime_source_target": "/workspace/public-seed"},
        "identity": {"uid": os.getuid(), "gid": os.getgid()},
    }

    snapshot = Path(launcher.stage_runtime_assets(profile)[0]["source"])
    assert snapshot.is_relative_to(tmpfs)
    assert snapshot.is_dir()
    assert not (workspace / "runtime-staging").exists()
    profile["_runtime_asset_snapshots"] = launcher.stage_runtime_assets(profile)
    effective = launcher.stage_host_runtime_source(profile)
    assert Path(effective["source_root"]).is_relative_to(tmpfs)


def test_runtime_assets_reject_the_dev_shm_root_as_staging_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (workspace / name).mkdir()
    source = workspace / "seed-source"
    source.mkdir()
    (source / "worker.py").write_text("print('pinned')\n")
    tmpfs = tmp_path / "tmpfs"
    tmpfs.mkdir()
    monkeypatch.setattr(launcher, "HOST_TMPFS_ROOT", tmpfs)
    profile = {
        "workspace": str(workspace),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {name: name for name in ("models", "data", "runs", "cache")},
        "runtime_asset_staging_root": str(tmpfs),
        "runtime_assets": [{
            "source": "seed-source", "target": "/workspace/public-seed",
            "expected_type": "dir", "sha256": launcher.runtime_asset_digest(source, "dir"),
        }],
        "execution": {"mode": "host-orchestrator"},
    }
    with pytest.raises(launcher.LaunchError, match="under /dev/shm"):
        launcher.stage_runtime_assets(profile)
    assert list(tmpfs.iterdir()) == []


def test_validate_only_checks_existing_tmpfs_ancestor_without_creating_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (workspace / name).mkdir()
    source = workspace / "seed-source"
    source.mkdir()
    (source / "worker.py").write_text("print('pinned')\n")
    tmpfs = tmp_path / "tmpfs"
    tmpfs.mkdir()
    monkeypatch.setattr(launcher, "HOST_TMPFS_ROOT", tmpfs)
    observed = []
    monkeypatch.setattr(launcher, "_validate_sealed_tmpfs_creation_parent", lambda path: observed.append(path))
    staging = tmpfs / "new-parent" / "new-root"
    profile = {
        "workspace": str(workspace),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {name: name for name in ("models", "data", "runs", "cache")},
        "runtime_asset_staging_root": str(staging),
        "runtime_assets": [{
            "source": "seed-source", "target": "/workspace/public-seed",
            "expected_type": "dir", "sha256": launcher.runtime_asset_digest(source, "dir"),
        }],
        "execution": {"mode": "host-orchestrator"},
    }

    launcher.stage_runtime_assets(profile, validate_only=True)
    assert observed == [staging]
    assert not (tmpfs / "new-parent").exists()


def test_validate_only_requires_root_for_uncreated_tmpfs_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(launcher, "HOST_TMPFS_ROOT", tmp_path)
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 1001)
    with pytest.raises(launcher.LaunchError, match="root launcher"):
        launcher._validate_sealed_tmpfs_creation_parent(tmp_path / "new-root")


def test_absolute_runtime_staging_cannot_overlap_a_fixed_workspace_mount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tmpfs = tmp_path / "tmpfs"
    workspace = tmpfs / "workspace"
    workspace.mkdir(parents=True)
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (workspace / name).mkdir()
    source = workspace / "seed-source"
    source.mkdir()
    (source / "worker.py").write_text("print('pinned')\n")
    monkeypatch.setattr(launcher, "HOST_TMPFS_ROOT", tmpfs)
    monkeypatch.setattr(launcher, "_validate_sealed_tmpfs_ancestry", lambda _path: None)
    profile = {
        "workspace": str(workspace),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {name: name for name in ("models", "data", "runs", "cache")},
        "runtime_asset_staging_root": str(workspace / "cache" / "runtime-staging"),
        "runtime_assets": [{
            "source": "seed-source", "target": "/workspace/public-seed",
            "expected_type": "dir", "sha256": launcher.runtime_asset_digest(source, "dir"),
        }],
        "execution": {"mode": "host-orchestrator"},
    }
    with pytest.raises(launcher.LaunchError, match="overlap any fixed Docker bind source"):
        launcher.stage_runtime_assets(profile, validate_only=True)


def test_doctor_profile_validates_runtime_assets_without_staging_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = tmp_path / "seed-source"
    runtime.mkdir()
    (runtime / "worker.py").write_text("print('pinned')\n")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "workspace": str(tmp_path),
        "source": {
            "experiment": {"path": "experiment", "revision": "a" * 40},
            "openevo": {"path": "openevo", "revision": "b" * 40},
        },
        "paths": {name: name for name in ("models", "data", "runs", "cache")},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{
            "source": "seed-source", "target": "/workspace/public-seed",
            "expected_type": "dir", "sha256": launcher.runtime_asset_digest(runtime, "dir"),
        }],
        "image": "example/runtime:tag", "image_id": "sha256:" + "c" * 64,
    }
    monkeypatch.setattr(launcher, "load_profile", lambda _path: profile)
    monkeypatch.setattr(launcher, "git_identity", lambda *_args: None)
    monkeypatch.setattr(launcher, "resolve_image_identity", lambda _profile: profile["image_id"])

    assert launcher._command_doctor_profile(SimpleNamespace(profile=tmp_path / "profile.json")) == 0
    assert not (tmp_path / "runtime-staging").exists()
    assert profile["_runtime_asset_snapshots"][0]["source"] == str(runtime)


def test_execution_rejects_writable_inner_runtime_snapshot_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    script = source / "worker.py"
    script.write_text("pass\n")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    digest = launcher.runtime_asset_digest(source, "dir")
    profile = {
        "run_id": "inner-writable-check",
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{"source": "runtime-source", "target": "/workspace/public-seed",
                            "expected_type": "dir", "sha256": digest}],
        "tracking": {"group": "q03"},
    }
    staged = launcher.stage_runtime_assets(profile)
    profile["_runtime_asset_snapshots"] = staged
    (Path(staged[0]["source"]) / "worker.py").chmod(0o644)
    monkeypatch.setattr(launcher.gpu_resource_wrapper, "run", lambda _args: pytest.fail("payload must not start"))
    with pytest.raises(launcher.LaunchError, match="writable or linked entry"):
        launcher.execute_gpu_command(
            profile, ["GPU-aaa"], ["docker", "run"], tmp_path / "runs" / "run", "sha256:" + "a" * 64
        )


def test_runtime_file_overlay_is_hash_bound_read_only_and_ordered_after_base(tmp_path: Path) -> None:
    base = tmp_path / "seed-source"
    (base / "pkg").mkdir(parents=True)
    (base / "pkg" / "compat.py").write_text("public\n")
    overlay = tmp_path / "compat-overlay.py"
    overlay.write_text("pinned compatibility\n")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "run_id": "runtime-overlay-test",
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{
            "source": "seed-source", "target": "/workspace/public-seed", "expected_type": "dir",
            "sha256": launcher.runtime_asset_digest(base, "dir"),
        }],
        "runtime_asset_overlays": [{
            "source": "compat-overlay.py", "target": "/workspace/public-seed/pkg/compat.py",
            "base_target": "/workspace/public-seed", "expected_type": "file",
            "sha256": launcher.runtime_asset_digest(overlay, "file"),
        }],
    }
    snapshots = launcher.stage_runtime_assets(profile)
    profile.update({
        "identity": {"uid": 1001, "gid": 1001}, "command": ["true"],
        "tracking": {"mode": "offline"}, "_runtime_asset_snapshots": snapshots,
    })
    command = launcher.docker_command(profile, ["GPU-aaa"], "sha256:image", "offline")
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    base_mount = "type=bind,src=" + snapshots[0]["source"] + ",dst=/workspace/public-seed,readonly"
    overlay_mount = "type=bind,src=" + snapshots[1]["source"] + ",dst=/workspace/public-seed/pkg/compat.py,readonly"
    assert mounts.index(base_mount) < mounts.index(overlay_mount)
    assert Path(snapshots[1]["source"]).read_text() == "pinned compatibility\n"
    assert launcher._runtime_asset_receipt_identity(profile)[1]["base_target"] == "/workspace/public-seed"


def test_step119_model_load_preflight_is_candidate_bound_and_zero_task(tmp_path: Path, monkeypatch) -> None:
    profile = {
        "run_id": "q03-final-run",
        "workspace": str(tmp_path),
        "identity": {"uid": os.getuid(), "gid": os.getgid()},
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "command": [
            "/runtime/python", "-m", "q03_step119_validation_runtime",
                "--run-dir", str(tmp_path / "runs" / "q03-final-run"),
        ],
        "execution": {
            "mode": "host-orchestrator",
            "runtime_entrypoint": {"kind": "module", "module": "q03_step119_validation_runtime"},
        },
        "_host_runtime_source": {"source_root": str(tmp_path / "effective"), "sha256": "a" * 64},
        "_runtime_asset_snapshots": [],
    }
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    (tmp_path / "effective").mkdir()
    control = tmp_path / "control"
    control.mkdir()
    preflight_dir = tmp_path / "runs" / ".zero-formal-q03-attempt"
    preflight_dir.mkdir()
    observed = {}

    monkeypatch.setattr(launcher, "validate_host_runtime_source", lambda _profile: None)
    monkeypatch.setattr(launcher, "_host_runtime_entrypoint_identity", lambda *_args: {})
    monkeypatch.setattr(launcher, "_runtime_asset_receipt_identity", lambda _profile: [])
    monkeypatch.setattr(launcher, "_command_identity", lambda _argv, _image: "sha256:command")
    def fake_model_load_run(argv, check=False):
        observed["argv"] = argv
        command = argv
        encoded = command[command.index("--model-load-preflight-context") + 1]
        context = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "status": "MODEL_LOAD_PREFLIGHT_PASS",
                "native_init_workers_returned": True,
                "actor_rollout_model_initialized": True,
                "vllm_runtime_initialized": True,
                "formal_task_count": 0,
                "environment_creation_calls": 0,
                "dataset_item_reads": 0,
                "generation_calls": 0,
                "optimizer_updates": 0,
                "preflight_authorization": {"context_sha256": context["context_sha256"]},
            }) + "\n",
            stderr="",
        )

    monkeypatch.setattr(launcher, "_run", fake_model_load_run)

    result = launcher.run_step119_model_load_preflight(
        profile, ["GPU-00000000-0000-0000-0000-000000000000"], preflight_dir,
        "sha256:image", control,
        {
            "campaign_id": "q03-step119",
            "execution_sha": "1" * 40,
            "current_campaign_sha256": "2" * 64,
            "design_sha256": "3" * 64,
            "activation_sha256": "4" * 64,
            "host_router_path": "configs/experiment/current-campaign.json",
            "host_control_head": "5" * 40,
            "registry_pointer_path": "configs/experiment/registry.json",
            "registry_pointer_sha256": "6" * 64,
        },
        "sha256:profile",
    )

    command = observed["argv"]
    assert command[command.index("--run-dir") + 1] == f"/workspace/runs/{preflight_dir.name}"
    assert command[-1] == "--model-load-preflight-only"
    assert "OPENEVO_FORMAL_START_RECEIPT" not in " ".join(command)
    assert "OPENEVO_STEP119_PREFLIGHT_CONTEXT_SHA256=" in " ".join(command)
    encoded = command[command.index("--model-load-preflight-context") + 1]
    context = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    assert f"CUDA_VISIBLE_DEVICES={context['selected_gpu_uuids'][0]}" in command
    assert context["campaign_id"] == "q03-step119"
    assert context["execution_sha"] == "1" * 40
    assert context["runtime_profile_sha256"] == "sha256:profile"
    assert context["selected_gpu_uuids"] == ["GPU-00000000-0000-0000-0000-000000000000"]
    assert context["effective_source_sha256"] == "a" * 64
    assert context["execution_surface"] == "host-orchestrator"
    assert context["formal_task_count"] == context["optimizer_updates"] == 0
    assert result["receipt"]["status"] == "PASS"
    assert result["receipt"]["formal_task_count"] == 0
    assert result["receipt"]["optimizer_updates"] == 0


def test_model_load_preflight_flag_cannot_enter_formal_campaign_route(tmp_path: Path, capsys) -> None:
    result = launcher.main([
        "run-campaign",
        "--model-load-preflight",
        "--output-root", str(tmp_path / "runs"),
        "--gpus", "GPU-00000000-0000-0000-0000-000000000000",
    ])

    assert result == 2
    assert "--model-load-preflight requires --preflight-only" in capsys.readouterr().err


def test_host_orchestrator_imports_effective_source_tree_with_overlay(tmp_path: Path) -> None:
    base = tmp_path / "seed-source"
    package = base / "q03probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (base / "probe.py").write_text(
        "from q03probe.compat import VALUE\n"
        "import q03probe.compat as compat\n"
        "print(VALUE)\n"
        "print(compat.__file__)\n"
    )
    overlay = tmp_path / "compat-overlay.py"
    overlay.write_text("VALUE = 'overlay'\n")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "run_id": "host-runtime-source-test",
        "workspace": str(tmp_path),
        "identity": {"uid": os.getuid(), "gid": os.getgid()},
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{
            "source": "seed-source", "target": "/workspace/public-seed", "expected_type": "dir",
            "sha256": launcher.runtime_asset_digest(base, "dir"),
        }],
        "runtime_asset_overlays": [{
            "source": "compat-overlay.py", "target": "/workspace/public-seed/q03probe/compat.py",
            "base_target": "/workspace/public-seed", "expected_type": "file",
            "sha256": launcher.runtime_asset_digest(overlay, "file"),
        }],
        "execution": {
            "mode": "host-orchestrator",
            "runtime_source_target": "/workspace/public-seed",
            "runtime_entrypoint": {"kind": "script", "path": "probe.py"},
            "runtime_python": sys.executable,
            "runtime_import_modules": ["q03probe.compat"],
        },
        "command": [sys.executable, "${SOURCE_ROOT}/probe.py", "--run-dir", "${RUN_DIR}"],
        "tracking": {"mode": "offline"},
        "_formal_start_context": {"ready_at_epoch": 1.0},
    }
    profile["_runtime_asset_snapshots"] = launcher.stage_runtime_assets(profile)
    with ThreadPoolExecutor(max_workers=3) as pool:
        staged_sets = list(pool.map(lambda _index: launcher.stage_runtime_assets(profile), range(3)))
        effective_sets = list(
            pool.map(
                lambda staged: launcher.stage_host_runtime_source({**profile, "_runtime_asset_snapshots": staged}),
                staged_sets,
            )
        )
    assert staged_sets[0] == staged_sets[1] == staged_sets[2]
    assert effective_sets[0] == effective_sets[1] == effective_sets[2]
    profile["_host_runtime_source"] = effective_sets[0]

    docker_argv = launcher.docker_command(profile, ["GPU-aaa"], "sha256:image", "offline")
    docker_mounts = [
        docker_argv[index + 1]
        for index, value in enumerate(docker_argv)
        if value == "--mount"
    ]
    effective_mount = (
        f"type=bind,src={effective_sets[0]['source_root']},"
        "dst=/workspace/public-seed,readonly"
    )
    assert docker_mounts.count(effective_mount) == 1
    assert not any("dst=/workspace/public-seed/q03probe/compat.py" in mount for mount in docker_mounts)
    assert not any(f"src={asset['source']}" in mount for asset in profile["_runtime_asset_snapshots"] for mount in docker_mounts)

    import_identity = launcher.probe_host_runtime_imports(profile, cwd=tmp_path)
    assert import_identity is not None
    assert import_identity["status"] == "PASS"
    effective_root = Path(profile["_host_runtime_source"]["source_root"])
    assert launcher.runtime_asset_digest(effective_root, "dir") == profile["_host_runtime_source"]["sha256"]
    staging_root = tmp_path / "runtime-staging"
    effective_parent = staging_root / "host-effective-sources"
    for directory in (staging_root, effective_parent):
        info = directory.stat()
        assert info.st_gid == os.getgid()
        assert info.st_mode & 0o010
        assert not info.st_mode & 0o007
    for path in (effective_root, *effective_root.rglob("*")):
        info = path.stat()
        assert info.st_gid == os.getgid()
        assert not info.st_mode & 0o222
        assert info.st_mode & 0o040
        if path.is_dir():
            assert info.st_mode & 0o010
    assert import_identity["modules"]["q03probe.compat"]["path"] == (
        f"{profile['_host_runtime_source']['source_root']}/q03probe/compat.py"
    )
    assert import_identity["modules"]["q03probe.compat"]["sha256"] == hashlib.sha256(
        (Path(profile["_host_runtime_source"]["source_root"]) / "q03probe/compat.py").read_bytes()
    ).hexdigest()
    assert import_identity["entrypoint"]["relative_path"] == "probe.py"
    assert import_identity["entrypoint"]["sha256"] == hashlib.sha256(
        (Path(profile["_host_runtime_source"]["source_root"]) / "probe.py").read_bytes()
    ).hexdigest()
    assert import_identity["python_no_user_site"] == 1
    profile["_host_runtime_import_identity"] = import_identity

    command = launcher.host_orchestrator_identity_command(
        profile, ["GPU-aaa"], tmp_path / "runs" / profile["run_id"], "offline", tmp_path / "formal-start.json"
    )
    effective_root = profile["_host_runtime_source"]["source_root"]
    assert not (base / "q03probe/compat.py").exists()
    assert (Path(effective_root) / "q03probe/compat.py").read_text() == "VALUE = 'overlay'\n"
    assert launcher.runtime_asset_digest(Path(effective_root), "dir") == profile["_host_runtime_source"]["sha256"]
    assert f"PYTHONPATH={effective_root}" in command
    assert "-u" in command and command[command.index("-u") + 1] == "PYTHONHOME"
    assert "PYTHONNOUSERSITE=1" in command
    assert "PYTHONDONTWRITEBYTECODE=1" in command
    assert f"OPENEVO_EFFECTIVE_SOURCE_SHA256={profile['_host_runtime_source']['sha256']}" in command
    assert f"{effective_root}/probe.py" in command
    assert command[command.index("--run-dir") + 1] == f"/workspace/runs/{profile['run_id']}"
    assert launcher._runtime_asset_receipt_identity(profile)[-2]["runtime_role"] == "host-effective-source"
    import_receipt = launcher.write_host_runtime_import_receipt(profile, tmp_path / "runs" / profile["run_id"])
    assert import_receipt is not None
    assert launcher._runtime_asset_receipt_identity(profile)[-1]["runtime_role"] == "host-runtime-import-identity"

    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    assert completed.stdout.splitlines()[0] == "overlay"
    assert completed.stdout.splitlines()[1].startswith(effective_root + "/q03probe/compat.py")
    launcher.validate_host_runtime_source(profile)

    literal_run_dir = tmp_path / "runs" / "q03-step119-val128-20261025"
    literal_profile = {
        **profile,
        "command": [sys.executable, "${SOURCE_ROOT}/probe.py", "--run-dir", str(literal_run_dir)],
    }
    literal_command = launcher.host_orchestrator_identity_command(
        literal_profile, ["GPU-aaa"], tmp_path / "runs" / profile["run_id"], "offline", tmp_path / "formal-start.json"
    )
    assert literal_command[literal_command.index("--run-dir") + 1] == (
        "/workspace/runs/q03-step119-val128-20261025"
    )
    escaping_profile = {
        **profile,
        "command": [sys.executable, "${SOURCE_ROOT}/probe.py", "--run-dir", str(tmp_path / "outside")],
    }
    with pytest.raises(launcher.LaunchError, match="--run-dir must be under the mounted runs root"):
        launcher.host_orchestrator_identity_command(
            escaping_profile, ["GPU-aaa"], tmp_path / "runs" / profile["run_id"], "offline", tmp_path / "formal-start.json"
        )
    relative_profile = {
        **profile,
        "command": [sys.executable, "${SOURCE_ROOT}/probe.py", "--run-dir", "relative-output"],
    }
    with pytest.raises(launcher.LaunchError, match=r"must be \$\{RUN_DIR\} or an absolute host path"):
        launcher.host_orchestrator_identity_command(
            relative_profile, ["GPU-aaa"], tmp_path / "runs" / profile["run_id"], "offline", tmp_path / "formal-start.json"
        )

    shadow = tmp_path / "shadow"
    (shadow / "q03probe").mkdir(parents=True)
    (shadow / "q03probe/__init__.py").write_text("")
    (shadow / "q03probe/compat.py").write_text("VALUE = 'wrong'")
    with pytest.raises(launcher.LaunchError, match="escapes effective source tree"):
        launcher.probe_host_runtime_imports(profile, cwd=shadow)
    outside = dict(profile)
    outside["command"] = [sys.executable, "/tmp/untrusted_main.py", "${SOURCE_ROOT}"]
    with pytest.raises(launcher.LaunchError, match="declared script"):
        launcher.host_orchestrator_identity_command(
            outside, ["GPU-aaa"], tmp_path / "runs" / "outside", "offline", tmp_path / "formal-start.json"
        )


def test_host_runtime_import_probe_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime-venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
    python = runtime / "bin" / "python"
    if not python.is_symlink():
        pytest.skip("this platform creates copied rather than symlinked venv interpreters")

    site_packages = runtime / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    (site_packages / "q03_runtime_dependency.py").write_text("VALUE = 'venv-site-packages'\n")

    source_root = tmp_path / "effective-source"
    package = source_root / "q03probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    module = package / "compat.py"
    module.write_text("import q03_runtime_dependency\nVALUE = q03_runtime_dependency.VALUE\n")
    entrypoint = source_root / "probe.py"
    entrypoint.write_text("print('zero-formal-import-probe')\n")
    profile = {
        "runtime_assets": [{}],
        "execution": {
            "mode": "host-orchestrator",
            "runtime_python": str(python),
            "runtime_import_modules": ["q03probe.compat"],
            "runtime_entrypoint": {"kind": "script", "path": "probe.py"},
        },
        "command": [str(python), "${SOURCE_ROOT}/probe.py"],
        "_host_runtime_source": {
            "source_root": str(source_root),
            "sha256": launcher.runtime_asset_digest(source_root, "dir"),
        },
    }

    identity = launcher.probe_host_runtime_imports(profile, cwd=tmp_path)

    assert identity is not None
    assert identity["status"] == "PASS"
    assert identity["python_realpath"] == str(python.resolve())
    assert any(str(site_packages) in item for item in identity["python_sys_path"])
    assert identity["modules"]["q03probe.compat"]["path"] == str(module.resolve())


def test_runtime_snapshot_lock_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n")
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep mode\n")
    original_mode = victim.stat().st_mode
    (bucket / ".snapshot.lock").symlink_to(victim)

    with pytest.raises(launcher.LaunchError, match="cannot safely open launcher lock"):
        launcher._stage_runtime_asset_snapshot(source, bucket, "/workspace/source", "dir", launcher.runtime_asset_digest(source, "dir"))
    assert victim.read_text() == "keep mode\n"
    assert victim.stat().st_mode == original_mode


def test_host_effective_source_lock_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    temp_tree = tmp_path / "temporary-tree"
    temp_tree.mkdir()
    (temp_tree / "main.py").write_text("pass\n")
    effective_parent = tmp_path / "effective"
    effective_parent.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep mode\n")
    original_mode = victim.stat().st_mode
    effective_sha = launcher.runtime_asset_digest(temp_tree, "dir")
    (effective_parent / f".{effective_sha}.lock").symlink_to(victim)

    with pytest.raises(launcher.LaunchError, match="cannot safely open launcher lock"):
        launcher._publish_host_effective_source(
            temp_tree, effective_parent, effective_sha, "/workspace/source", "base-sha", []
        )
    assert victim.read_text() == "keep mode\n"
    assert victim.stat().st_mode == original_mode


def test_host_orchestrator_fails_closed_when_runtime_assets_are_not_bound_to_source_root(tmp_path: Path) -> None:
    base = tmp_path / "seed-source"
    base.mkdir()
    (base / "main.py").write_text("pass\n")
    (base / "overlay.py").write_text("BASE = 0\n")
    overlay = tmp_path / "overlay.py"
    overlay.write_text("VALUE = 1\n")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{
            "source": "seed-source", "target": "/workspace/public-seed", "expected_type": "dir",
            "sha256": launcher.runtime_asset_digest(base, "dir"),
        }],
        "runtime_asset_overlays": [{
            "source": "overlay.py", "target": "/workspace/public-seed/overlay.py",
            "base_target": "/workspace/public-seed", "expected_type": "file",
            "sha256": launcher.runtime_asset_digest(overlay, "file"),
        }],
        "execution": {"mode": "host-orchestrator"},
    }
    profile["_runtime_asset_snapshots"] = launcher.stage_runtime_assets(profile)
    with pytest.raises(launcher.LaunchError, match="runtime_source_target"):
        launcher.stage_host_runtime_source(profile)


def test_runtime_file_overlay_must_be_a_unique_file_below_declared_base(tmp_path: Path) -> None:
    base = tmp_path / "seed-source"
    base.mkdir()
    overlay = tmp_path / "overlay.py"
    overlay.write_text("overlay\n")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{"source": "seed-source", "target": "/workspace/public-seed", "expected_type": "dir",
                            "sha256": launcher.runtime_asset_digest(base, "dir")}],
        "runtime_asset_overlays": [{"source": "overlay.py", "target": "/workspace/elsewhere/override.py",
                                    "base_target": "/workspace/public-seed", "expected_type": "file",
                                    "sha256": launcher.runtime_asset_digest(overlay, "file")}],
    }
    with pytest.raises(launcher.LaunchError, match="below its base_target"):
        launcher.stage_runtime_assets(profile)


@pytest.mark.parametrize("target", [
    "/workspace",
    "/workspace/models/overlay",
    "/tmp/untrusted-runtime",
])
def test_runtime_asset_targets_reject_reserved_or_overlapping_mounts(tmp_path: Path, target: str) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    (source / "worker.py").write_text("x")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{
            "source": "runtime-source",
            "target": target,
            "expected_type": "dir",
            "sha256": launcher.runtime_asset_digest(source, "dir"),
        }],
    }
    with pytest.raises(launcher.LaunchError):
        launcher.stage_runtime_assets(profile)


def test_runtime_asset_targets_reject_nested_or_duplicate_runtime_mounts(tmp_path: Path) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    (source / "worker.py").write_text("x")
    digest = launcher.runtime_asset_digest(source, "dir")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [
            {"source": "runtime-source", "target": "/workspace/public-seed", "expected_type": "dir", "sha256": digest},
            {"source": "runtime-source", "target": "/workspace/public-seed/nested", "expected_type": "dir", "sha256": digest},
        ],
    }
    with pytest.raises(launcher.LaunchError, match="overlaps"):
        launcher.stage_runtime_assets(profile)


def test_runtime_asset_source_hash_drift_and_symlinks_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    (source / "worker.py").write_text("original")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{"source": "runtime-source", "target": "/workspace/public-seed",
                            "expected_type": "dir", "sha256": "0" * 64}],
    }
    with pytest.raises(launcher.LaunchError, match="SHA-256"):
        launcher.stage_runtime_assets(profile)
    (source / "escape").symlink_to(tmp_path / "cache", target_is_directory=True)
    profile["runtime_assets"][0]["sha256"] = "1" * 64
    with pytest.raises(launcher.LaunchError, match="symlink"):
        launcher.stage_runtime_assets(profile)


def test_runtime_asset_staging_rejects_hash_bucket_symlink_escape(tmp_path: Path) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    (source / "worker.py").write_text("pinned")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    digest = launcher.runtime_asset_digest(source, "dir")
    outside = tmp_path / "outside"
    outside.mkdir()
    staging = tmp_path / "runtime-staging"
    staging.mkdir(mode=0o700)
    (staging / digest).symlink_to(outside, target_is_directory=True)
    profile = {
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{"source": "runtime-source", "target": "/workspace/public-seed",
                            "expected_type": "dir", "sha256": digest}],
    }
    with pytest.raises(launcher.LaunchError, match="real directory"):
        launcher.stage_runtime_assets(profile)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("staging_root,source", [
    ("cache/runtime-staging", "runtime-source"),
    ("runtime-staging", "../runtime-source"),
    ("runtime-staging", "linked-source"),
])
def test_runtime_asset_staging_rejects_writable_mount_overlap_and_unsafe_source(
    tmp_path: Path, staging_root: str, source: str
) -> None:
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    real_source = tmp_path / "runtime-source"
    real_source.mkdir()
    (real_source / "worker.py").write_text("x")
    (tmp_path / "linked-source").symlink_to(real_source, target_is_directory=True)
    source_path = tmp_path / source
    digest = launcher.runtime_asset_digest(real_source, "dir")
    profile = {
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": staging_root,
        "runtime_assets": [{"source": source, "target": "/workspace/public-seed",
                            "expected_type": "dir", "sha256": digest}],
    }
    if source not in {"../runtime-source", "linked-source"} and not source_path.exists():
        source_path.mkdir()
        (source_path / "worker.py").write_text("x")
    with pytest.raises(launcher.LaunchError):
        launcher.stage_runtime_assets(profile)


def test_execution_revalidates_effective_source_after_payload_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    (source / "worker.py").write_text("original\n")
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    digest = launcher.runtime_asset_digest(source, "dir")
    profile = {
        "run_id": "postrun-source-check",
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "runtime_asset_staging_root": "runtime-staging",
        "runtime_assets": [{
            "source": "runtime-source",
            "target": "/workspace/public-seed",
            "expected_type": "dir",
            "sha256": digest,
        }],
        "tracking": {"group": "q03"},
    }
    snapshots = launcher.stage_runtime_assets(profile)
    profile["_runtime_asset_snapshots"] = snapshots
    staged = Path(snapshots[0]["source"])

    monkeypatch.setattr(launcher, "validate_host_runtime_source", lambda _profile: None)
    monkeypatch.setattr(launcher, "verify_host_runtime_imports", lambda _profile: None)

    def mutate_after_start(args):
        (staged / "worker.py").chmod(0o644)
        (staged / "worker.py").write_text("mutated by payload\n")
        receipt = Path(args.receipt)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("{}\n")
        return 0

    monkeypatch.setattr(launcher.gpu_resource_wrapper, "run", mutate_after_start)
    monkeypatch.setattr(launcher.gpu_resource_wrapper, "classify_receipt", lambda _path: {"status": "exact"})
    with pytest.raises(launcher.NonRetryableLaunchError, match="runtime asset snapshot changed during execution"):
        launcher.execute_gpu_command(
            profile,
            ["GPU-aaa"],
            ["python", "worker.py"],
            tmp_path / "runs" / "run",
            "sha256:" + "a" * 64,
        )


def test_non_exact_gpu_resource_receipt_is_nonretryable_after_payload_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "run_id": "non-exact-resource-receipt",
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "tracking": {"group": "q03"},
    }

    def completed_payload(args):
        receipt = Path(args.receipt)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("{}\n")
        return 0

    monkeypatch.setattr(launcher.gpu_resource_wrapper, "run", completed_payload)
    monkeypatch.setattr(launcher.gpu_resource_wrapper, "classify_receipt", lambda _path: {"status": "ambiguous"})
    with pytest.raises(launcher.NonRetryableLaunchError, match="resource receipt"):
        launcher.execute_gpu_command(
            profile,
            ["GPU-aaa"],
            ["python", "payload.py"],
            tmp_path / "runs" / "run",
            "sha256:" + "a" * 64,
        )


def test_gpu_resource_receipt_classification_exception_is_nonretryable_after_payload_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (tmp_path / name).mkdir()
    profile = {
        "run_id": "receipt-classification-error",
        "workspace": str(tmp_path),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
        "tracking": {"group": "q03"},
    }

    def completed_payload(args):
        receipt = Path(args.receipt)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("{}\n")
        return 0

    def classification_error(_path):
        raise launcher.gpu_resource_wrapper.ResourceAccountingError(1, ["classify_receipt"])

    monkeypatch.setattr(launcher.gpu_resource_wrapper, "run", completed_payload)
    monkeypatch.setattr(launcher.gpu_resource_wrapper, "classify_receipt", classification_error)
    with pytest.raises(launcher.NonRetryableLaunchError, match="task consumption may have occurred"):
        launcher.execute_gpu_command(
            profile,
            ["GPU-aaa"],
            ["python", "payload.py"],
            tmp_path / "runs" / "run",
            "sha256:" + "a" * 64,
        )


def test_root_disk_staging_checks_entire_ancestry(monkeypatch):
    safe = Path("/workspace/.openevo-runtime-sealed/q03")
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 0)
    def info(path):
        return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=0, st_gid=0)
    monkeypatch.setattr(Path, "lstat", info)
    launcher._validate_sealed_disk_ancestry(safe)
    for bad_uid, bad_mode in [(1001, 0o755), (0, 0o775), (0, stat.S_IFLNK | 0o777)]:
        def unsafe(path, uid=bad_uid, mode=bad_mode):
            if path == Path("/workspace"):
                return SimpleNamespace(st_mode=(stat.S_IFDIR | mode) if not stat.S_ISLNK(mode) else mode, st_uid=uid, st_gid=0)
            return info(path)
        monkeypatch.setattr(Path, "lstat", unsafe)
        with pytest.raises(launcher.LaunchError, match="sealed disk staging ancestor"):
            launcher._validate_sealed_disk_ancestry(safe)
    monkeypatch.setattr(Path, "lstat", info)


def test_absolute_disk_staging_is_limited_to_sealed_account_cache(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (workspace / name).mkdir()
    source = workspace / "seed-source"
    source.mkdir()
    (source / "worker.py").write_text("pass\n")
    root = tmp_path / ".openevo-runtime-sealed" / "q03"
    profile = {
        "workspace": str(workspace),
        "source": {"experiment": {"path": "experiment"}, "openevo": {"path": "openevo"}},
        "paths": {n:n for n in ("models", "data", "runs", "cache")},
        "runtime_asset_staging_root": str(root),
        "runtime_assets": [{"source": "seed-source", "target": "/workspace/public-seed", "expected_type": "dir", "sha256": launcher.runtime_asset_digest(source,"dir")}],
        "execution": {"mode":"host-orchestrator"},
    }
    checks=[]
    monkeypatch.setattr(launcher, "_validate_sealed_disk_ancestry", lambda path, **kw: checks.append((path,kw)))
    rows=launcher.stage_runtime_assets(profile)
    assert Path(rows[0]["source"]).is_relative_to(root)
    assert len(checks)>=2
    assert checks[0][1].get("allow_missing") is True
    profile["runtime_asset_staging_root"]=str(tmp_path / "unapproved" / "q03")
    with pytest.raises(launcher.LaunchError,match="absolute runtime_asset_staging_root"):
        launcher.stage_runtime_assets(profile)


def test_preflight_command_build_failure_is_nonretryable(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "_prelaunch_runtime_python", lambda p:"python3")
    def fail(*a,**kw):
        raise launcher.LaunchError("host effective source must stay sealed")
    monkeypatch.setattr(launcher,"docker_command",fail)
    with pytest.raises(launcher.PrelaunchEngineeringError,match="host effective source must stay sealed"):
        launcher.run_selected_gpu_preflight({"run_id":"test"},["GPU-aaa"],tmp_path,"sha256:image",tmp_path)


def test_unsafe_disk_ancestor_fails_before_creating_staging(tmp_path, monkeypatch):
    # An immutable leaf cannot protect a namespace owned by the run user.
    workspace=tmp_path / "workspace";workspace.mkdir()
    root=tmp_path / ".openevo-runtime-sealed" / "q03"
    source=workspace / "source";source.mkdir();(source / "x.py").write_text("pass\n")
    profile={"workspace":str(workspace),"runtime_asset_staging_root":str(root),
      "source":{"experiment":{"path":"experiment"},"openevo":{"path":"openevo"}},
      "paths":{n:n for n in ("models","data","runs","cache")},
      "runtime_assets":[{"source":"source","target":"/workspace/seed","expected_type":"dir","sha256":launcher.runtime_asset_digest(source,"dir")}],
      "execution":{"mode":"host-orchestrator"}}
    def unsafe(*args,**kwargs):raise launcher.LaunchError("sealed disk staging ancestor is unsafe")
    monkeypatch.setattr(launcher,"_validate_sealed_disk_ancestry",unsafe)
    with pytest.raises(launcher.LaunchError,match="ancestor is unsafe"):
        launcher.stage_runtime_assets(profile)
    assert not root.parent.exists()


def test_disk_sealed_source_reaches_docker_consumer_and_rejects_hidden_parent(tmp_path, monkeypatch):
    tmp_path.chmod(0o755)
    workspace=tmp_path / "workspace";workspace.mkdir()
    for name in ("experiment","openevo","models","data","runs","cache"):
        (workspace/name).mkdir()
    namespace=tmp_path / ".openevo-runtime-sealed";namespace.mkdir(mode=0o755)
    staging=namespace / "q03";staging.mkdir(mode=0o710)
    parent=staging / "host-effective-sources";parent.mkdir(mode=0o710)
    root=parent / "content";root.mkdir()
    module=root / "worker.py";module.write_text("pass\n");module.chmod(0o444);root.chmod(0o555)
    digest=launcher.runtime_asset_digest(root,"dir")
    identity={"source_root":str(root),"target":"/workspace/seed","sha256":digest,"base_sha256":digest}
    receipt=parent / "receipt.json";receipt.write_text(json.dumps(identity));receipt.chmod(0o440)
    identity.update(receipt_path=str(receipt),receipt_sha256=launcher.prelaunch_zero_formal.sha256_file(receipt))
    run_gid=os.getgid() or 1001
    profile={"workspace":str(workspace),"run_id":"sealed-consumer","identity":{"uid":1001,"gid":run_gid},
      "source":{"experiment":{"path":"experiment"},"openevo":{"path":"openevo"}},
      "paths":{n:n for n in ("models","data","runs","cache")},
      "runtime_asset_staging_root":str(staging),
      "runtime_assets":[{"target":"/workspace/seed","expected_type":"dir"}],
      "_runtime_asset_snapshots":[{"source":str(root),"target":"/workspace/seed","expected_type":"dir","sha256":digest}],
      "_host_runtime_source":identity,"execution":{"mode":"host-orchestrator","runtime_source_target":"/workspace/seed"},
      "command":["python3","-c","pass"]}
    original=Path.lstat
    # Model the production root controller and its protected account ancestry on a non-root CI host.
    with monkeypatch.context() as m:
        m.setattr(launcher.os,"geteuid",lambda:0)
        # Avoid rglob inside lstat: membership is lexical, not another filesystem walk.
        def mapped(path):
            info=original(path);mode=info.st_mode if path==tmp_path or tmp_path in path.parents else stat.S_IFDIR | 0o755
            return os.stat_result((mode,info.st_ino,info.st_dev,info.st_nlink,0,run_gid,info.st_size,info.st_atime,info.st_mtime,info.st_ctime))
        m.setattr(Path,"lstat",mapped)
        launcher.validate_host_runtime_source(profile)
        command=launcher.docker_command(profile,["GPU-aaa"],"sha256:image","disabled")
        assert any(str(root) in arg and "dst=/workspace/seed" in arg for arg in command)
        namespace.chmod(0o700)
        with pytest.raises(launcher.LaunchError,match="not traversable"):
            launcher.validate_host_runtime_source(profile)
