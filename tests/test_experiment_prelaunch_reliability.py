from __future__ import annotations

import json
import multiprocessing
import os
import stat
import subprocess
import tempfile
import time
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts import oe_impl as oe
from scripts.openevo_webshop import experiment_reliability_runtime as reliability
from scripts.openevo_webshop import prelaunch_zero_formal as pre
from scripts.openevo_webshop import rank_consistency as rank


def _record_lifecycle_event_in_process(args: tuple[str, int]) -> int:
    control_dir, index = args
    reliability.record_lifecycle_event(
        Path(control_dir),
        run_id="multiprocess-r1",
        campaign_id="multiprocess-c1",
        key="T5",
        epoch=float(index),
        meaning="synthetic multiprocess event",
        source={"writer": index},
    )
    return index


def pass_payload(count: int = 1) -> dict:
    return {
        "schema": pre.PAYLOAD_SCHEMA,
        "status": "PASS",
        "stage": "COMPLETE",
        "expected_gpu_count": count,
        "observed_gpu_count": count,
        "formal_task_consumed": False,
        "model_calls": 0,
        "training_calls": 0,
        "final_panel_access": 0,
        "scientific_state_mutated": False,
        "torch_version": "test",
        "torch_cuda_version": "test",
        "devices": [],
    }
def workspace_profile(root: Path) -> dict:
    for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
        (root / name).mkdir(exist_ok=True)
    return {
        "schema": "openevo.webshop.run-profile." + "v" + "1",
        "run_id": "prelaunch-test",
        "image": "openevo/test:latest",
        "image_id": "sha256:" + "a" * 64,
        "workspace": str(root),
        "identity": {"uid": os.getuid(), "gid": os.getgid()},
        "source": {
            "experiment": {"path": "experiment", "revision": "a" * 40},
            "openevo": {"path": "openevo", "revision": "b" * 40},
        },
        "paths": {
            "models": "models",
            "data": "data",
            "runs": "runs",
            "cache": "cache",
        },
        "command": ["python3", "-c", "print('formal')"],
        "tracking": {"mode": "disabled"},
    }


class PrelaunchReceiptTests(unittest.TestCase):
    def test_payload_parser_enforces_zero_formal_fields(self) -> None:
        payload = pass_payload()
        parsed = pre.parse_payload_output("noise\n" + json.dumps(payload) + "\n")
        self.assertEqual(parsed["status"], "PASS")
        payload["model_calls"] = 1
        with self.assertRaises(pre.PrelaunchError):
            pre.parse_payload_output(json.dumps(payload))
    def test_prelaunch_receipt_is_identity_bound_and_zero_formal(self) -> None:
        payload = pass_payload()
        receipt = pre.build_prelaunch_receipt(
            run_id="r1",
            attempt=1,
            selected_gpu_uuids=["GPU-aaa"],
            image_id="sha256:" + "a" * 64,
            command_identity="sha256:" + "b" * 64,
            return_code=0,
            wall_seconds=0.5,
            payload=payload,
            payload_error=None,
            stderr_tail="",
        )
        pre.validate_prelaunch_receipt(receipt, expected_gpu_uuids=["GPU-aaa"])
        self.assertFalse(receipt["formal_task_consumed"])
        self.assertEqual(receipt["scientific_budget_consumed"], 0)
        tampered = dict(receipt)
        tampered["selected_gpu_uuids"] = ["GPU-bbb"]
        with self.assertRaises(pre.PrelaunchError):
            pre.validate_prelaunch_receipt(tampered)

    def test_failed_payload_never_becomes_pass(self) -> None:
        payload = pass_payload()
        payload["status"] = "FAIL"
        payload["stage"] = "CUDA_KERNEL"
        receipt = pre.build_prelaunch_receipt(
            run_id="r1",
            attempt=1,
            selected_gpu_uuids=["GPU-aaa"],
            image_id="sha256:" + "a" * 64,
            command_identity="sha256:" + "b" * 64,
            return_code=43,
            wall_seconds=0.1,
            payload=payload,
            payload_error=None,
            stderr_tail="kernel unsupported",
        )
        self.assertEqual(receipt["status"], "FAIL")
        self.assertEqual(receipt["failure_class"], "PRELAUNCH_ENGINEERING")
        self.assertEqual(receipt["failure_stage"], "CUDA_KERNEL")
        with self.assertRaises(pre.PrelaunchError):
            pre.validate_prelaunch_receipt(receipt)


class GpuCapacityGateTests(unittest.TestCase):
    def _busy_card(self, *, used: int = 12000, total: int = 32600) -> dict[str, Any]:
        return {
            "index": 0,
            "uuid": "GPU-aaa",
            "memory_used_mib": used,
            "memory_total_mib": total,
            "utilization_gpu_percent": 73,
            "compute_apps": ["123, python, 12000"],
            "compute_apps_query_error": None,
            "device_handles": [{"pid": "123", "fd": "9", "device": "/dev/nvidia0", "user": "root", "comm": "python", "exe": "/usr/bin/python", "cmdline_programs": "python", "parent_pid": "1", "cgroup": "test"}],
        }

    def test_capacity_gate_allows_authorized_busy_card_when_peak_fits(self) -> None:
        card = self._busy_card()
        ok, failures = oe.capacity_cards(
            [card],
            ["GPU-aaa"],
            qualified_peak_memory_mib=8000,
            min_transient_headroom_mib=4096,
            co_residency_authorized=True,
        )
        self.assertTrue(ok)
        self.assertEqual(failures, [])
        selected = oe.select_gpu_uuids(
            "auto:1",
            ["GPU-aaa"],
            [card],
            capacity_gate={
                "qualified_peak_memory_mib": 8000,
                "min_transient_headroom_mib": 4096,
                "co_residency_authorized": True,
            },
        )
        self.assertEqual(selected, ["GPU-aaa"])

    def test_capacity_gate_rejects_insufficient_free_memory(self) -> None:
        ok, failures = oe.capacity_cards(
            [self._busy_card(used=23000, total=32600)],
            ["GPU-aaa"],
            qualified_peak_memory_mib=8000,
            min_transient_headroom_mib=4096,
            co_residency_authorized=True,
        )
        self.assertFalse(ok)
        self.assertTrue(any("free-memory=" in failure for failure in failures))

    def test_capacity_gate_requires_explicit_co_residency_authority_for_busy_card(self) -> None:
        ok, failures = oe.capacity_cards(
            [self._busy_card()],
            ["GPU-aaa"],
            qualified_peak_memory_mib=8000,
            min_transient_headroom_mib=4096,
            co_residency_authorized=False,
        )
        self.assertFalse(ok)
        self.assertIn("GPU-aaa:occupied-co-residency-not-authorized", failures)

    def test_profile_capacity_mode_requires_frozen_peak_headroom_and_authority(self) -> None:
        with self.assertRaises(oe.PrelaunchEngineeringError):
            oe._capacity_gate_settings({"gpu_gate": {"mode": "capacity"}})
        settings = oe._capacity_gate_settings({
            "gpu_gate": {
                "mode": "capacity",
                "qualified_peak_memory_mib": 7000,
                "min_transient_headroom_mib": 4096,
                "co_residency_authorized": True,
            }
        })
        self.assertEqual(settings["qualified_peak_memory_mib"], 7000)
        self.assertEqual(settings["min_transient_headroom_mib"], 4096)
        self.assertTrue(settings["co_residency_authorized"])


class LauncherIntegrationTests(unittest.TestCase):
    def test_selected_gpu_smoke_uses_exact_docker_surface(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = workspace_profile(root)
            run_dir = root / "runs" / profile["run_id"]
            run_dir.mkdir()
            control = root / "control"
            stdout = json.dumps(pass_payload()) + "\n"
            completed = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
            seen: list[list[str]] = []

            def fake_run(command, *, check=True, capture=True):
                seen.append(command)
                return completed

            with patch.object(oe, "_run", side_effect=fake_run):
                result = oe.run_selected_gpu_preflight(
                    profile, ["GPU-aaa"], run_dir, profile["image_id"], control
                )
            self.assertEqual(len(seen), 1)
            command = seen[0]
            self.assertIn("--gpus", command)
            self.assertIn("device=GPU-aaa", command)
            self.assertIn("--user", command)
            self.assertIn(f"{os.getuid()}:{os.getgid()}", command)
            self.assertEqual(command[command.index("--entrypoint") + 1], "python3")
            payload_program_index = command.index("-c", command.index("--entrypoint")) + 1
            self.assertEqual(command[payload_program_index], pre.PAYLOAD_PROGRAM)
            self.assertIn('"formal_task_consumed": False', pre.PAYLOAD_PROGRAM)
            self.assertNotIn("print('formal')", command)
            receipt = json.loads(Path(result["path"]).read_text())
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["selected_gpu_uuids"], ["GPU-aaa"])

    def test_non_python_profile_requires_explicit_smoke_interpreter(self) -> None:
        profile = {"command": ["bash", "-lc", "python train.py"]}
        with self.assertRaises(oe.PrelaunchEngineeringError):
            oe._prelaunch_runtime_python(profile)
        profile["prelaunch"] = {"runtime_python": "/opt/venv/bin/python"}
        self.assertEqual(oe._prelaunch_runtime_python(profile), "/opt/venv/bin/python")

    def test_explicit_prelaunch_interpreter_must_be_python_executable(self) -> None:
        profile = {
            "command": ["python3", "train.py"],
            "prelaunch": {"runtime_python": "/bin/bash"},
        }
        with self.assertRaisesRegex(oe.PrelaunchEngineeringError, "must name a Python interpreter"):
            oe._prelaunch_runtime_python(profile)

    def test_formal_start_binds_exact_prelaunch_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = pass_payload()
            receipt = pre.build_prelaunch_receipt(
                run_id="r1",
                attempt=1,
                selected_gpu_uuids=["GPU-aaa"],
                image_id="sha256:" + "a" * 64,
                command_identity="sha256:" + "b" * 64,
                return_code=0,
                wall_seconds=0.2,
                payload=payload,
                payload_error=None,
                stderr_tail="",
            )
            pre_path = root / "prelaunch-zero-formal-attempt-0001.json"
            pre_path.write_text(json.dumps(receipt), encoding="utf-8")
            formal = pre.build_formal_start_receipt(
                run_id="r1",
                campaign_id="c1",
                selected_gpu_uuids=["GPU-aaa"],
                image_id="sha256:" + "a" * 64,
                command_identity="sha256:" + "c" * 64,
                prelaunch_receipt_path=pre_path,
                readiness={"ready_receipt": {"sha256": "sha256:" + "d" * 64}},
                output_root="/runs/r1",
            )
            self.assertEqual(formal["status"], "FORMAL_START_COMMITTED")
            self.assertFalse(formal["formal_task_consumption_before"])
            self.assertTrue(formal["formal_task_consumption_may_begin"])
            self.assertFalse(formal["scientific_authority_granted_by_receipt"])
            self.assertEqual(
                formal["prelaunch_receipt"]["receipt_sha256"],
                receipt["receipt_sha256"],
            )

    def test_formal_start_requires_runtime_asset_identity_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            assets = [{"target": "/workspace/public-seed", "expected_type": "dir", "sha256": "e" * 64}]
            receipt = pre.build_prelaunch_receipt(
                run_id="r1", attempt=1, selected_gpu_uuids=["GPU-aaa"],
                image_id="sha256:" + "a" * 64, command_identity="sha256:" + "b" * 64,
                return_code=0, wall_seconds=0.2, payload=pass_payload(), payload_error=None,
                stderr_tail="", runtime_assets=assets,
            )
            pre_path = root / "prelaunch.json"
            pre_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(pre.PrelaunchError, "runtime asset identity"):
                pre.build_formal_start_receipt(
                    run_id="r1", campaign_id="c1", selected_gpu_uuids=["GPU-aaa"],
                    image_id="sha256:" + "a" * 64, command_identity="sha256:" + "c" * 64,
                    prelaunch_receipt_path=pre_path, readiness={}, output_root="/runs/r1",
                    runtime_assets=[],
                )

    def test_prelaunch_engineering_error_is_not_wait_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = workspace_profile(root)
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            args = Namespace(
                profile=profile_path,
                timeout_seconds=0,
                run_id=None,
                wait=True,
                resume=False,
                refresh_tracking_probe=False,
                gpus="auto:1",
                formal_start_context=None,
            )
            idle = {
                "index": 0,
                "uuid": "GPU-aaa",
                "memory_used_mib": 0,
                "memory_total_mib": 24576,
                "utilization_gpu_percent": 0,
                "compute_apps": [],
                "device_handles": [],
            }
            calls = {"preflight": 0}

            def fail_once(*_args, **_kwargs):
                calls["preflight"] += 1
                if calls["preflight"] > 1:
                    raise AssertionError("prelaunch engineering failure was retried")
                raise oe.PrelaunchEngineeringError("kernel capability mismatch")

            run_root = root / "runs" / profile["run_id"]
            run_root.mkdir()
            with patch.dict(os.environ, {oe.DEFAULT_AUTH_ENV: "GPU-aaa"}):
                with patch.object(oe, "validate_sources_and_paths", return_value=None), \
                     patch.object(oe, "resolve_image_identity", return_value=profile["image_id"]), \
                     patch.object(oe, "ensure_run_directory", return_value=run_root), \
                     patch.object(oe, "choose_effective_tracking", return_value=("disabled", None, None, False)), \
                     patch.object(oe, "snapshot_gpus", return_value=[idle]), \
                     patch.object(oe, "wait_for_idle", return_value=[idle]), \
                     patch.object(oe, "run_selected_gpu_preflight", side_effect=fail_once):
                    with self.assertRaisesRegex(
                        oe.PrelaunchEngineeringError, "kernel capability mismatch"
                    ):
                        oe.command_launch(args)
            self.assertEqual(calls["preflight"], 1)

    def test_post_formal_telemetry_failure_never_releases_expired_lease(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = workspace_profile(root)
            profile["prelaunch"] = {"lease_ttl_seconds": 1}
            run_dir = root / "runs" / profile["run_id"]
            run_dir.mkdir()
            control = root / "control"
            idle = {
                "index": 0,
                "uuid": "GPU-aaa",
                "memory_used_mib": 0,
                "memory_total_mib": 24576,
                "utilization_gpu_percent": 0,
                "compute_apps": [],
                "device_handles": [],
            }
            prelaunch_result = {
                "path": str(control / "prelaunch.json"),
                "sha256": "sha256:" + "1" * 64,
                "receipt_sha256": "sha256:" + "2" * 64,
                "receipt": {"observed_at_epoch": 1.0},
            }
            formal = control / "formal-start.synthetic.json"

            def commit_formal_start(*_args, on_committed=None, **_kwargs):
                # Model a durable Formal Start commit followed by delayed
                # bookkeeping that crosses lease expiry. No real run starts.
                committed_at = time.time()
                if on_committed is not None:
                    on_committed()
                time.sleep(1.05)
                return {
                    "path": str(formal),
                    "sha256": "sha256:" + "3" * 64,
                    "receipt_sha256": "sha256:" + "4" * 64,
                    "receipt": {"committed_at_epoch": committed_at},
                }

            record = reliability.record_lifecycle_event

            def fail_t4(*args, **kwargs):
                if kwargs.get("key") == "T4":
                    raise OSError("synthetic telemetry storage failure")
                return record(*args, **kwargs)

            args = Namespace(
                profile=root / "profile.json",
                timeout_seconds=0,
                run_id=None,
                wait=False,
                resume=False,
                refresh_tracking_probe=False,
                gpus="auto:1",
                formal_start_context={
                    "control_root": str(control),
                    "ready_at_epoch": time.time(),
                    "ready_receipt": {"sha256": "sha256:" + "5" * 64},
                },
            )
            with patch.dict(os.environ, {oe.DEFAULT_AUTH_ENV: "GPU-aaa"}):
                with patch.object(oe, "load_profile", return_value=profile), \
                     patch.object(oe, "validate_sources_and_paths", return_value=None), \
                     patch.object(oe, "resolve_image_identity", return_value=profile["image_id"]), \
                     patch.object(oe, "ensure_run_directory", return_value=run_dir), \
                     patch.object(oe, "choose_effective_tracking", return_value=("disabled", None, None, False)), \
                     patch.object(oe, "snapshot_gpus", return_value=[idle]), \
                     patch.object(oe, "wait_for_idle", return_value=[idle]), \
                     patch.object(oe, "run_selected_gpu_preflight", return_value=prelaunch_result), \
                     patch.object(oe, "write_formal_start_receipt", side_effect=commit_formal_start), \
                     patch.object(reliability, "record_lifecycle_event", side_effect=fail_t4), \
                     patch.object(oe, "execute_gpu_command", side_effect=AssertionError("payload must not run")):
                    with self.assertRaisesRegex(OSError, "synthetic telemetry storage failure"):
                        oe.command_launch(args)

            self.assertTrue(any(control.glob("prelaunch-lease-handoff-*.json")))
            self.assertFalse(any(control.glob("prelaunch-lease-release-*.json")))

    def test_formal_start_readback_failure_after_commit_never_releases_lease(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = workspace_profile(root)
            profile["prelaunch"] = {"lease_ttl_seconds": 1}
            run_dir = root / "runs" / profile["run_id"]
            run_dir.mkdir()
            control = root / "control"
            idle = {
                "index": 0,
                "uuid": "GPU-aaa",
                "memory_used_mib": 0,
                "memory_total_mib": 24576,
                "utilization_gpu_percent": 0,
                "compute_apps": [],
                "device_handles": [],
            }
            prelaunch_result = {
                "path": str(control / "prelaunch.json"),
                "sha256": "sha256:" + "1" * 64,
                "receipt_sha256": "sha256:" + "2" * 64,
                "receipt": {"observed_at_epoch": 1.0},
            }
            args = Namespace(
                profile=root / "profile.json",
                timeout_seconds=0,
                run_id=None,
                wait=False,
                resume=False,
                refresh_tracking_probe=False,
                gpus="auto:1",
                formal_start_context={
                    "control_root": str(control),
                    "ready_at_epoch": time.time(),
                    "ready_receipt": {"sha256": "sha256:" + "5" * 64},
                },
            )
            receipt = {
                "schema": pre.FORMAL_START_SCHEMA,
                "status": "FORMAL_START_COMMITTED",
                "run_id": profile["run_id"],
                "campaign_id": profile["run_id"],
                "selected_gpu_uuids": ["GPU-aaa"],
                "committed_at_epoch": time.time(),
                "receipt_sha256": "sha256:" + "6" * 64,
            }
            real_sha256_file = pre.sha256_file

            def fail_formal_readback(path: Path) -> str:
                if Path(path).name.startswith("formal-start-"):
                    self.assertTrue(Path(path).is_file(), "Formal Start must be committed first")
                    time.sleep(1.05)
                    raise OSError("synthetic Formal Start readback failure")
                return real_sha256_file(path)

            with patch.dict(os.environ, {oe.DEFAULT_AUTH_ENV: "GPU-aaa"}):
                with patch.object(oe, "load_profile", return_value=profile), \
                     patch.object(oe, "validate_sources_and_paths", return_value=None), \
                     patch.object(oe, "resolve_image_identity", return_value=profile["image_id"]), \
                     patch.object(oe, "ensure_run_directory", return_value=run_dir), \
                     patch.object(oe, "choose_effective_tracking", return_value=("disabled", None, None, False)), \
                     patch.object(oe, "snapshot_gpus", return_value=[idle]), \
                     patch.object(oe, "wait_for_idle", return_value=[idle]), \
                     patch.object(oe, "run_selected_gpu_preflight", return_value=prelaunch_result), \
                     patch.object(pre, "build_formal_start_receipt", return_value=receipt), \
                     patch.object(pre, "sha256_file", side_effect=fail_formal_readback), \
                     patch.object(oe, "execute_gpu_command", side_effect=AssertionError("payload must not run")):
                    with self.assertRaisesRegex(OSError, "synthetic Formal Start readback failure"):
                        oe.command_launch(args)

            self.assertTrue(any(control.glob("formal-start-*.json")))
            self.assertFalse(any(control.glob("prelaunch-lease-release-*.json")))


class CanonicalRunCampaignTests(unittest.TestCase):
    def test_run_campaign_passes_exact_readiness_context_to_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile_path = root / "profile.json"
            profile_path.write_text("{}", encoding="utf-8")
            control = root / "control"
            control.mkdir()
            ready_path = control / "execution-readiness.json"
            ready_path.write_text('{"verdict":"READY"}\n', encoding="utf-8")
            result = {
                "verdict": "READY",
                "profile_path": str(profile_path),
                "profile": {"sha256": "sha256:" + "1" * 64},
                "ready_receipt_path": str(ready_path),
                "scientific_identity_sha256": "sha256:" + "2" * 64,
                "authority": {"router_status": "launch-authorized"},
            }
            args = Namespace(
                fingerprint_file=root / "fingerprint.json",
                cache_root=root / "cache",
                output_root=root / "output",
                resume=False,
                refresh_qualification=False,
                gpus="auto:1",
                wait=False,
                timeout_seconds=0,
                run_id=None,
                refresh_tracking_probe=False,
                dry_run=False,
            )
            captured: list[Namespace] = []

            def fake_launch(launch_args, pre_launch_hook=None):
                captured.append(launch_args)
                return 0

            with patch.object(oe.execution_readiness, "evaluate_readiness", return_value=result), \
                 patch.object(oe, "command_launch", side_effect=fake_launch):
                self.assertEqual(oe.command_run_campaign(args), 0)

            context = captured[0].formal_start_context
            self.assertEqual(context["control_root"], str(control.resolve()))
            self.assertEqual(
                context["ready_receipt"]["sha256"],
                pre.sha256_file(ready_path),
            )
            self.assertEqual(
                context["scientific_identity_sha256"],
                "sha256:" + "2" * 64,
            )

    def test_formal_start_exists_before_task_bearing_payload_executes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = workspace_profile(root)
            run_dir = root / "runs" / profile["run_id"]
            run_dir.mkdir()
            control = root / "control"
            control.mkdir()
            pre_receipt = pre.build_prelaunch_receipt(
                run_id=profile["run_id"],
                attempt=1,
                selected_gpu_uuids=["GPU-aaa"],
                image_id=profile["image_id"],
                command_identity="sha256:" + "3" * 64,
                return_code=0,
                wall_seconds=0.1,
                payload=pass_payload(),
                payload_error=None,
                stderr_tail="",
            )
            pre_path = control / "prelaunch-zero-formal-attempt-0001.json"
            pre_path.write_text(json.dumps(pre_receipt), encoding="utf-8")
            prelaunch_result = {
                "path": str(pre_path),
                "sha256": pre.sha256_file(pre_path),
                "receipt_sha256": pre_receipt["receipt_sha256"],
                "receipt": pre_receipt,
            }
            idle = {
                "index": 0,
                "uuid": "GPU-aaa",
                "memory_used_mib": 0,
                "memory_total_mib": 24576,
                "utilization_gpu_percent": 0,
                "compute_apps": [],
                "device_handles": [],
            }
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            args = Namespace(
                profile=profile_path,
                timeout_seconds=0,
                run_id=None,
                wait=False,
                resume=False,
                refresh_tracking_probe=False,
                gpus="auto:1",
                formal_start_context={
                    "control_root": str(control),
                    "ready_at_epoch": 10.0,
                    "ready_receipt": {"sha256": "sha256:" + "4" * 64},
                    "scientific_identity_sha256": "sha256:" + "5" * 64,
                },
            )
            def fake_execute(*_args, **_kwargs):
                formal = control / "formal-start-attempt-0001.json"
                self.assertTrue(formal.is_file())
                payload = json.loads(formal.read_text(encoding="utf-8"))
                self.assertTrue(payload["formal_start_committed"])
                self.assertTrue(payload["formal_task_consumption_may_begin"])
                return 0, {
                    "path": str(run_dir / "gpu-resource-receipt.json"),
                    "sha256": "sha256:" + "6" * 64,
                    "accounting": {"status": "exact"},
                }

            with patch.dict(os.environ, {oe.DEFAULT_AUTH_ENV: "GPU-aaa"}):
                with patch.object(oe, "load_profile", return_value=profile), \
                     patch.object(oe, "validate_sources_and_paths", return_value=None), \
                     patch.object(oe, "resolve_image_identity", return_value=profile["image_id"]), \
                     patch.object(oe, "ensure_run_directory", return_value=run_dir), \
                     patch.object(oe, "choose_effective_tracking", return_value=("disabled", None, None, False)), \
                     patch.object(oe, "snapshot_gpus", return_value=[idle]), \
                     patch.object(oe, "wait_for_idle", return_value=[idle]), \
                     patch.object(oe, "run_selected_gpu_preflight", return_value=prelaunch_result), \
                     patch.object(oe, "execute_gpu_command", side_effect=fake_execute):
                    self.assertEqual(oe.command_launch(args), 0)

            launch = json.loads((run_dir / "launch-receipt.json").read_text())
            self.assertIn("formal_start", launch)
            self.assertEqual(
                launch["prelaunch_zero_formal"]["receipt_sha256"],
                pre_receipt["receipt_sha256"],
            )
            self.assertIsNotNone(launch["prelaunch_coordination_lease"])
            self.assertIn("prelaunch_coordination_handoff", launch)
            handoff = json.loads(
                Path(launch["prelaunch_coordination_handoff"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(handoff["status"], "FORMAL_HANDOFF")
            self.assertFalse(handoff["reservation_claimed"])
            timeline = reliability.lifecycle_summary(control)
            for key in ("T0", "T1", "T2", "T3", "T4"):
                self.assertIsNotNone(timeline["summary"][key])
            self.assertIsNone(timeline["summary"]["T5"])
            self.assertIn("time_to_formal_start_seconds", timeline["metrics"])


class ReliabilityRuntimeTests(unittest.TestCase):
    def test_prelaunch_coordination_lease_is_bounded_and_non_authoritative(self) -> None:
        lease = reliability.build_prelaunch_lease(
            run_id="r1",
            campaign_id="c1",
            selected_gpu_uuids=["GPU-aaa"],
            runtime_candidate_uuids=["GPU-aaa", "GPU-bbb"],
            readiness_receipt_sha256="sha256:" + "1" * 64,
            ttl_seconds=10,
            acquired_at_epoch=100.0,
        )
        reliability.require_live_lease(
            lease, now_epoch=105.0, expected_gpu_uuids=["GPU-aaa"]
        )
        with self.assertRaises(reliability.ReliabilityRecordError):
            reliability.require_live_lease(lease, now_epoch=110.0)
        handoff = reliability.build_lease_handoff(
            lease,
            formal_start_receipt_sha256="sha256:" + "2" * 64,
            handed_off_at_epoch=111.0,
        )
        self.assertEqual(handoff["status"], "FORMAL_HANDOFF")
        self.assertFalse(handoff["reservation_claimed"])
        release = reliability.build_lease_release(
            lease, reason="TEST_RELEASE", released_at_epoch=109.0
        )
        self.assertEqual(release["status"], "RELEASED")
        self.assertFalse(release["resource_authority_granted"])

    def test_lifecycle_events_are_not_lost_under_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            control = Path(td)

            def record(index: int) -> None:
                reliability.record_lifecycle_event(
                    control,
                    run_id="r1",
                    campaign_id="c1",
                    key="T5",
                    epoch=float(index + 1),
                    meaning="concurrent synthetic event",
                    source={"index": index},
                )

            with ThreadPoolExecutor(max_workers=12) as pool:
                list(pool.map(record, range(32)))
            summary = reliability.lifecycle_summary(control)
            self.assertEqual(len(summary["events"]), 32)
            self.assertEqual(summary["summary"]["T5"]["epoch"], 1.0)

    def test_lifecycle_events_are_not_lost_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            control = Path(td)
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=8, mp_context=context) as pool:
                written = list(pool.map(
                    _record_lifecycle_event_in_process,
                    [(str(control), index) for index in range(64)],
                ))
            summary = reliability.lifecycle_summary(control)
            self.assertEqual(len(written), 64)
            self.assertEqual(len(summary["events"]), 64)
            self.assertEqual(len({event["event_sha256"] for event in summary["events"]}), 64)

    def test_second_incident_requires_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            kwargs = dict(
                category="PRELAUNCH_ENGINEERING",
                layer="selected_gpu_preflight",
                code="CUDA_KERNEL_UNSUPPORTED",
                validator_id="prelaunch_zero_formal",
                repair_class="SEMANTICS_PRESERVING_RUNTIME_REPAIR",
                detail="sm_120 kernel rejected",
                run_id="r1",
                selected_gpu_uuids=["GPU-aaa"],
            )
            first = reliability.record_incident(root, observed_at_epoch=1.0, **kwargs)
            second = reliability.record_incident(root, observed_at_epoch=2.0, **kwargs)
            self.assertEqual(first["occurrence_index"], 1)
            self.assertFalse(first["promotion_required"])
            self.assertEqual(second["occurrence_index"], 2)
            self.assertTrue(second["promotion_required"])
            summary = reliability.incident_summary(root)
            self.assertEqual(summary["repeated_fingerprint_count"], 1)

    def test_typed_runtime_outcomes_map_to_dev_failure_taxonomy(self) -> None:
        from scripts.openevo_runtime.result_codes import (
            OperationResult,
            OperationStatus,
            ReasonCode,
        )

        wait = OperationResult(
            status=OperationStatus.WAITING,
            reason_code=ReasonCode.GPU_RESOURCE_UNAVAILABLE,
            operation_id="op-wait",
            scientific_state_mutated=False,
            side_effect_materialized=False,
        ).to_dict()
        exact_resume = OperationResult(
            status=OperationStatus.BLOCKED,
            reason_code=ReasonCode.ENGINEERING_FAILURE_NO_SIDE_EFFECT,
            operation_id="op-resume",
            scientific_state_mutated=False,
            side_effect_materialized=False,
        ).to_dict()
        science = OperationResult(
            status=OperationStatus.BLOCKED,
            reason_code=ReasonCode.SCIENCE_SEMANTICS_CHANGE_REQUIRED,
            operation_id="op-science",
            scientific_state_mutated=False,
            side_effect_materialized=False,
        ).to_dict()

        self.assertEqual(
            reliability.classify_operation_result(wait, lifecycle_phase="FORMAL"),
            "RESOURCE_ORCHESTRATION",
        )
        self.assertEqual(
            reliability.classify_operation_result(
                exact_resume, lifecycle_phase="PRELAUNCH"
            ),
            "PRELAUNCH_ENGINEERING",
        )
        self.assertEqual(
            reliability.classify_operation_result(
                exact_resume, lifecycle_phase="FORMAL"
            ),
            "RUNTIME_INFRA",
        )
        self.assertEqual(
            reliability.classify_operation_result(science, lifecycle_phase="FORMAL"),
            "SCIENTIFIC",
        )

    def test_t0_to_t5_metrics_are_derived_without_becoming_authority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            control = Path(td)
            for key, epoch in (
                ("T0", 10.0),
                ("T1", 12.0),
                ("T2", 13.0),
                ("T3", 20.0),
                ("T4", 25.0),
                ("T5", 40.0),
            ):
                reliability.record_lifecycle_event(
                    control,
                    run_id="r1",
                    campaign_id="c1",
                    key=key,
                    epoch=epoch,
                    meaning=key,
                    source={"test": True},
                )
            summary = reliability.lifecycle_summary(control)
            self.assertEqual(summary["metrics"]["time_to_formal_start_seconds"], 15.0)
            self.assertEqual(
                summary["metrics"]["time_to_first_valid_step_seconds"], 30.0
            )
            self.assertTrue(summary["observability_only"])
            self.assertFalse(summary["scientific_authority_granted"])


class ReliabilityCliTests(unittest.TestCase):
    def test_mark_first_valid_step_records_t5_from_source_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reliability.record_lifecycle_event(
                root,
                run_id="r1",
                campaign_id="c1",
                key="T0",
                epoch=10.0,
                meaning="ready",
                source={"test": True},
            )
            source = root / "step-receipt.json"
            source.write_text(
                '{"status":"SEALED","run_id":"r1"}\n', encoding="utf-8"
            )
            args = Namespace(
                control_root=root,
                source_receipt=source,
                epoch=20.0,
            )
            self.assertEqual(oe.command_mark_first_valid_step(args), 0)
            summary = reliability.lifecycle_summary(root)
            self.assertEqual(summary["summary"]["T5"]["epoch"], 20.0)
            self.assertEqual(
                summary["summary"]["T5"]["source"]["sha256"],
                reliability.sha256_file(source),
            )

    def test_mark_first_valid_step_rejects_invalid_or_wrong_run_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reliability.record_lifecycle_event(
                root,
                run_id="r1",
                campaign_id="c1",
                key="T0",
                epoch=10.0,
                meaning="ready",
                source={"test": True},
            )
            source = root / "step-receipt.json"
            args = Namespace(control_root=root, source_receipt=source, epoch=20.0)
            for payload in (
                "",
                "not-json",
                '{"status":"OPEN","run_id":"r1"}',
                '{"status":"SEALED","run_id":"other"}',
                '{"status":"SEALED"}',
            ):
                source.write_text(payload, encoding="utf-8")
                with self.subTest(payload=payload):
                    with self.assertRaises(oe.LaunchError):
                        oe.command_mark_first_valid_step(args)
            self.assertIsNone(reliability.lifecycle_summary(root)["summary"]["T5"])

    def test_reliability_report_accepts_lifecycle_and_incident_roots(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            control = root / "control"
            incidents = root / "incidents"
            reliability.record_lifecycle_event(
                control,
                run_id="r1",
                campaign_id="c1",
                key="T0",
                epoch=10.0,
                meaning="ready",
                source={"test": True},
            )
            reliability.record_incident(
                incidents,
                category="PRELAUNCH_ENGINEERING",
                layer="rank_consistency",
                code="RANK_CONSISTENCY_FAILED",
                validator_id="rank_consistency",
                repair_class="SEMANTICS_PRESERVING_CONFIG_CLOSURE",
                detail="rank drift",
                run_id="r1",
                observed_at_epoch=11.0,
            )
            args = Namespace(control_root=control, incident_root=incidents)
            self.assertEqual(oe.command_reliability_report(args), 0)

    def test_recover_plan_delegates_without_launching(self) -> None:
        from scripts.openevo_runtime import round_transaction

        plan = {
            "round_root": "/tmp/round",
            "round_index": 3,
            "verdict": "EXACT_RESUME",
            "recovery_class": "MISSING_SUFFIX",
            "target_phase": "seal",
            "must_not_replay": ["rollout"],
            "phases": [],
            "integrity_errors": [],
        }
        args = Namespace(
            round_root=Path("/tmp/round"),
            round_index=3,
            rollouts_per_round=128,
            json=True,
        )
        with patch.object(
            round_transaction,
            "build_round_recovery_plan",
            return_value=plan,
        ) as build:
            self.assertEqual(oe.command_recover_plan(args), 0)
        build.assert_called_once_with(
            Path("/tmp/round"), 3, rollouts_per_round=128
        )


class RankConsistencyTests(unittest.TestCase):
    def _contract(self, root: Path, *, stale_controller: bool = False) -> dict:
        (root / "config.json").write_text(
            json.dumps({"state_rank": 32}), encoding="utf-8"
        )
        (root / "state.json").write_text(
            json.dumps({"bounded_state_rank": 32}), encoding="utf-8"
        )
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"r": 32, "lora_alpha": 32}), encoding="utf-8"
        )
        compare_rank = 128 if stale_controller else 32
        (root / "controller.py").write_text(
            "state_rank = 32\n"
            f"if state_rank != {compare_rank}:\n"
            "    raise RuntimeError('rank drift')\n",
            encoding="utf-8",
        )
        return {
            "expected_rank": 32,
            "required_roles": ["config", "controller", "state", "checkpoint"],
            "json_bindings": [
                {"role": "config", "path": "config.json", "pointers": ["/state_rank"]},
                {"role": "state", "path": "state.json", "pointers": ["/bounded_state_rank"]},
                {
                    "role": "checkpoint",
                    "path": "checkpoint/adapter_config.json",
                    "pointers": ["/r", "/lora_alpha"],
                },
            ],
            "python_bindings": [
                {
                    "role": "controller",
                    "path": "controller.py",
                    "symbols": ["state_rank", "bounded_state_rank"],
                }
            ],
        }

    def test_rank32_full_chain_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            receipt = rank.evaluate_contract(
                self._contract(root),
                resolve_path=lambda raw: root / raw,
            )
            rank.require_pass(receipt)
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(
                receipt["observed_roles"],
                ["checkpoint", "config", "controller", "state"],
            )
            self.assertFalse(receipt["formal_task_consumed"])
            self.assertEqual(receipt["gpu_requests"], 0)

    def test_stale_state_rank_128_gate_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            receipt = rank.evaluate_contract(
                self._contract(root, stale_controller=True),
                resolve_path=lambda raw: root / raw,
            )
            self.assertEqual(receipt["status"], "FAIL")
            self.assertTrue(
                any("state_rank=128" in failure for failure in receipt["failures"])
            )
            with self.assertRaises(rank.RankConsistencyError):
                rank.require_pass(receipt)

    def test_rank_contract_runs_before_gpu_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = workspace_profile(root)
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            run_root = root / "runs" / profile["run_id"]
            run_root.mkdir()
            args = Namespace(
                profile=profile_path,
                timeout_seconds=0,
                run_id=None,
                wait=False,
                resume=False,
                refresh_tracking_probe=False,
                gpus="auto:1",
                formal_start_context=None,
            )
            with patch.object(oe, "validate_sources_and_paths", return_value=None), \
                 patch.object(oe, "resolve_image_identity", return_value=profile["image_id"]), \
                 patch.object(oe, "ensure_run_directory", return_value=run_root), \
                 patch.object(
                     oe,
                     "run_rank_consistency_preflight",
                     side_effect=oe.PrelaunchEngineeringError("rank consistency failed"),
                 ), \
                 patch.object(
                     oe,
                     "snapshot_gpus",
                     side_effect=AssertionError("GPU discovery ran before rank validation"),
                 ):
                with self.assertRaisesRegex(
                    oe.PrelaunchEngineeringError, "rank consistency failed"
                ):
                    oe.command_launch(args)


class SchemaContractTests(unittest.TestCase):
    def test_new_receipt_schemas_are_well_formed_and_match_runtime_constants(self) -> None:
        schemas = {
            ROOT / "schemas/prelaunch-zero-formal-202609281010.schema.json": pre.PRELAUNCH_SCHEMA,
            ROOT / "schemas/formal-start-202609281010.schema.json": pre.FORMAL_START_SCHEMA,
            ROOT / "schemas/rank-consistency-202609281939.schema.json": rank.SCHEMA,
            ROOT / "schemas/prelaunch-coordination-lease-202609281950.schema.json": reliability.LEASE_SCHEMA,
            ROOT / "schemas/prelaunch-coordination-release-202609281950.schema.json": reliability.LEASE_RELEASE_SCHEMA,
            ROOT / "schemas/prelaunch-coordination-handoff-202609281950.schema.json": reliability.LEASE_HANDOFF_SCHEMA,
            ROOT / "schemas/engineering-incident-202609281950.schema.json": reliability.INCIDENT_SCHEMA,
            ROOT / "schemas/experiment-lifecycle-telemetry-202609281950.schema.json": reliability.TIMELINE_SCHEMA,
        }
        for path, expected in schemas.items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertEqual(payload["properties"]["schema"]["const"], expected)
            self.assertIn("receipt_sha256", payload["required"])


class ZeroFormalCampaignPreflightTests(unittest.TestCase):
    def external_pointer(self, root: Path, host_root: Path) -> Path:
        pointer = root / "server-registry.json"
        pointer.write_text(json.dumps({"canonical_control_checkout": str(host_root)}), encoding="utf-8")
        return pointer

    def candidate_checkout(
        self, root: Path, host_root: Path, campaign_id: str = "20260928-2338-candidate"
    ) -> Path:
        design_rel = "configs/experiment/designs/candidate.json"
        campaign_path = root / "configs/experiment/campaigns/candidate.json"
        current_campaign_path = root / "configs/experiment/current-campaign.json"
        design_path = root / design_rel
        activation_path = root / "configs/experiment/activations/candidate.json"
        for path in (design_path, activation_path, campaign_path, current_campaign_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        local_registry = root / "configs/platform/server-registry.json"
        local_registry.parent.mkdir(parents=True, exist_ok=True)
        local_registry.write_text(json.dumps({"canonical_control_checkout": str(root / "candidate-controlled-host")}), encoding="utf-8")
        campaign = {
            "campaign_id": campaign_id,
            "status": "owner-authorized-plan-frozen-formal-locked",
            "formal_task_consumption_allowed": False,
            "task_bearing_train_dev_allowed": False,
            "gpu_allocation_allowed": True,
            "final_test_status": "locked",
            "final_panel_access_allowed": False,
            "authorized_gpu_uuids": ["GPU-aaa"],
            "design": design_rel,
        }
        design = {"final_panel_access": 0, "frozen_scientific_contract": {"final_panel_unlock": False}}
        activation = {
            "revision": "202609282338", "design_path": design_rel,
            "status": "PENDING_ZERO_FORMAL_PREFLIGHT", "execution_sha": None,
            "owner_authorized": True, "formal_task_consumption_allowed": False,
            "final_panel_access_allowed": False, "gpu_uuids": ["GPU-aaa"],
        }
        campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
        current_campaign_path.write_text(json.dumps({"campaign_id": "sealed-predecessor", "status": "execution-sealed"}), encoding="utf-8")
        design_path.write_text(json.dumps(design), encoding="utf-8")
        activation_path.write_text(json.dumps(activation), encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
        subprocess.run(["git", "-C", str(root), "add", "configs"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "candidate"], check=True)
        subprocess.run(["git", "clone", "-q", str(root), str(host_root)], check=True)
        subprocess.run(["git", "-C", str(host_root), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(host_root), "config", "user.name", "Test"], check=True)
        (host_root / "host-descendant.txt").write_text("host control descendant\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(host_root), "add", "host-descendant.txt"], check=True)
        subprocess.run(["git", "-C", str(host_root), "commit", "-qm", "host descendant"], check=True)
        return campaign_path

    def profile_file(self, root: Path, execution_sha: str) -> Path:
        workspace = root / "workspace"
        for name in ("experiment", "openevo", "models", "data", "runs", "cache"):
            (workspace / name).mkdir(parents=True, exist_ok=True)
        profile = {
            "schema": "openevo.webshop.run-profile.v1",  # legacy-revision-compat: fixture for the existing profile reader
            "run_id": "never-run-science",
            "image": "openevo/test:latest", "image_id": "sha256:" + "a" * 64,
            "workspace": str(workspace), "identity": {"uid": os.getuid(), "gid": os.getgid()},
            "source": {"experiment": {"path": "experiment", "revision": execution_sha}, "openevo": {"path": "openevo", "revision": "b" * 40}},
            "paths": {"models": "models", "data": "data", "runs": "runs", "cache": "cache"},
            "command": ["python3", "-c", "raise SystemExit('science must not run')"],
            "tracking": {"mode": "disabled"},
        }
        path = root / "preflight-profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return path

    def test_formal_false_candidate_enters_zero_formal_qualification_when_router_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, host = base / "candidate", base / "host"
            root.mkdir()
            current_path = self.candidate_checkout(root, host)
            pointer = self.external_pointer(base, host)
            head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
            profile_path = self.profile_file(base, head)
            args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=profile_path, gpus="GPU-aaa", output_root=base / "evidence", preflight_only=True)
            payload = pass_payload()
            prelaunch = pre.build_prelaunch_receipt(
                run_id="synthetic", attempt=1, selected_gpu_uuids=["GPU-aaa"], image_id="sha256:" + "a" * 64,
                command_identity="sha256:" + "b" * 64, return_code=0, wall_seconds=0.01,
                payload=payload, payload_error=None, stderr_tail="",
            )
            prelaunch_path = base / "prelaunch.json"
            prelaunch_path.write_text(json.dumps(prelaunch), encoding="utf-8")
            prelaunch_result = {"path": str(prelaunch_path), "sha256": pre.sha256_file(prelaunch_path), "receipt": prelaunch}
            with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer), \
                 patch.object(oe, "validate_sources_and_paths"), \
                 patch.object(oe, "resolve_image_identity", return_value="sha256:" + "a" * 64), \
                 patch.object(oe, "snapshot_gpus", return_value=[{"uuid": "GPU-aaa"}]), \
                 patch.object(oe, "select_gpu_uuids", return_value=["GPU-aaa"]), \
                 patch.object(oe, "wait_for_idle", return_value=[]), \
                 patch.object(oe, "run_selected_gpu_preflight", return_value=prelaunch_result) as smoke, \
                 patch.object(oe, "write_formal_start_receipt", side_effect=AssertionError("Formal Start forbidden")), \
                 patch.object(oe, "execute_gpu_command", side_effect=AssertionError("science forbidden")), \
                 patch.object(oe.execution_readiness, "evaluate_readiness", side_effect=AssertionError("formal readiness forbidden")):
                self.assertEqual(oe.command_run_campaign(args), 0)
            smoke.assert_called_once()
            receipts = list((base / "evidence").rglob("zero-formal-qualification-attempt-*.json"))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(receipt["status"], "ZERO_FORMAL_QUALIFIED")
            self.assertFalse(receipt["formal_task_consumed"])
            self.assertEqual(receipt["formal_task_budget"], 0)
            self.assertEqual(receipt["final_panel_access"], 0)
            self.assertEqual(receipt["host_control_head"], subprocess.run(["git", "-C", str(host), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip())
            self.assertEqual(receipt["registry_pointer"]["path"], str(pointer.resolve()))
            self.assertEqual(receipt["registry_pointer"]["sha256"], pre.sha256_file(pointer))
            smoke_profile = smoke.call_args.args[0]
            self.assertTrue(smoke_profile["run_id"].startswith(".zero-formal-20260928-2338-candidate-"))
            self.assertNotEqual(smoke_profile["run_id"], "never-run-science")
            self.assertFalse(smoke.call_args.args[2].exists())
            self.assertFalse(list((base / "workspace/runs").glob("*/launch-receipt.json")))

    def test_host_router_mismatch_fails_before_gpu_or_science_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, host = base / "candidate", base / "host"
            root.mkdir()
            current_path = self.candidate_checkout(root, host)
            pointer = self.external_pointer(base, host)
            candidate_router = host / "configs/experiment/campaigns/candidate.json"
            candidate_router.write_text("{}", encoding="utf-8")
            subprocess.run(["git", "-C", str(host), "add", "configs/experiment/campaigns/candidate.json"], check=True)
            subprocess.run(["git", "-C", str(host), "commit", "-qm", "router mismatch"], check=True)
            args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=base / "missing.json", gpus="GPU-aaa", output_root=base / "evidence", preflight_only=True)
            with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer), \
                 patch.object(oe, "snapshot_gpus") as snapshot, \
                 patch.object(oe, "run_selected_gpu_preflight") as smoke, \
                 patch.object(oe, "execute_gpu_command") as science:
                with self.assertRaisesRegex(oe.LaunchError, "host candidate router does not exactly match"):
                    oe.command_run_campaign(args)
            snapshot.assert_not_called()
            smoke.assert_not_called()
            science.assert_not_called()
            self.assertFalse((base / "evidence").exists())

    def test_missing_external_pointer_fails_before_authoritative_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, host = base / "candidate", base / "host"
            root.mkdir()
            current_path = self.candidate_checkout(root, host)
            missing_pointer = base / "missing-server-registry.json"
            args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=None, gpus="GPU-aaa", output_root=base / "evidence", preflight_only=True)
            with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", missing_pointer), \
                 patch.object(oe, "snapshot_gpus") as snapshot:
                with self.assertRaisesRegex(oe.LaunchError, "repo-external canonical workspace pointer"):
                    oe.command_run_campaign(args)
            snapshot.assert_not_called()
            self.assertFalse((base / "evidence").exists())

    def test_campaign_id_path_traversal_and_output_control_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, host = base / "candidate", base / "host"
            root.mkdir()
            current_path = self.candidate_checkout(root, host, campaign_id="../escape")
            pointer = self.external_pointer(base, host)
            args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=None, gpus="GPU-aaa", output_root=base / "evidence", preflight_only=True)
            with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer), \
                 patch.object(oe, "snapshot_gpus") as snapshot:
                with self.assertRaisesRegex(oe.LaunchError, "safe path slug"):
                    oe.command_run_campaign(args)
            snapshot.assert_not_called()
            self.assertFalse((base / "evidence").exists())

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, host = base / "candidate", base / "host"
            root.mkdir()
            current_path = self.candidate_checkout(root, host)
            pointer = self.external_pointer(base, host)
            output_root = base / "evidence"
            output_root.mkdir()
            (output_root / "control").symlink_to(base / "external-control", target_is_directory=True)
            args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=None, gpus="GPU-aaa", output_root=output_root, preflight_only=True)
            with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer), \
                 patch.object(oe, "snapshot_gpus") as snapshot:
                with self.assertRaisesRegex(oe.LaunchError, "escapes output_root"):
                    oe.command_run_campaign(args)
            snapshot.assert_not_called()

    def test_existing_unique_run_directory_is_rejected_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, host = base / "candidate", base / "host"
            root.mkdir()
            current_path = self.candidate_checkout(root, host)
            pointer = self.external_pointer(base, host)
            head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
            profile_path = self.profile_file(base, head)
            profile = json.loads(profile_path.read_text())
            existing_run_dir = Path(profile["workspace"]) / profile["paths"]["runs"] / ".zero-formal-20260928-2338-candidate-fixed"
            existing_run_dir.mkdir()
            marker = existing_run_dir / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=profile_path, gpus="GPU-aaa", output_root=base / "evidence", preflight_only=True)
            fixed_uuid = SimpleNamespace(hex="fixed")
            with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer), \
                 patch.object(oe.uuid, "uuid4", return_value=fixed_uuid), \
                 patch.object(oe, "validate_sources_and_paths"), \
                 patch.object(oe, "resolve_image_identity", return_value="sha256:" + "a" * 64), \
                 patch.object(oe, "snapshot_gpus", return_value=[{"uuid": "GPU-aaa"}]), \
                 patch.object(oe, "select_gpu_uuids", return_value=["GPU-aaa"]), \
                 patch.object(oe, "wait_for_idle", return_value=[]), \
                 patch.object(oe, "run_selected_gpu_preflight") as smoke:
                with self.assertRaises(FileExistsError):
                    oe.command_run_campaign(args)
            smoke.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_authority_is_revalidated_after_idle_gate_before_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, host = base / "candidate", base / "host"
            root.mkdir()
            current_path = self.candidate_checkout(root, host)
            pointer = self.external_pointer(base, host)
            head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
            profile_path = self.profile_file(base, head)
            args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=profile_path, gpus="GPU-aaa", output_root=base / "evidence", preflight_only=True)

            def change_host_router(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
                router = host / "configs/experiment/campaigns/candidate.json"
                router.write_text("{}", encoding="utf-8")
                subprocess.run(["git", "-C", str(host), "add", "configs/experiment/campaigns/candidate.json"], check=True)
                subprocess.run(["git", "-C", str(host), "commit", "-qm", "router changed during idle gate"], check=True)
                return []

            with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer), \
                 patch.object(oe, "validate_sources_and_paths"), \
                 patch.object(oe, "resolve_image_identity", return_value="sha256:" + "a" * 64), \
                 patch.object(oe, "snapshot_gpus", return_value=[{"uuid": "GPU-aaa"}]), \
                 patch.object(oe, "select_gpu_uuids", return_value=["GPU-aaa"]), \
                 patch.object(oe, "wait_for_idle", side_effect=change_host_router), \
                 patch.object(oe, "run_selected_gpu_preflight") as smoke:
                with self.assertRaisesRegex(oe.LaunchError, "host candidate router does not exactly match"):
                    oe.command_run_campaign(args)
            smoke.assert_not_called()

    def test_campaign_preflight_lock_blocks_second_selector_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, host = base / "candidate", base / "host"
            root.mkdir()
            current_path = self.candidate_checkout(root, host)
            pointer = self.external_pointer(base, host)
            args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=None, gpus="GPU-other", output_root=base / "evidence-one", preflight_only=True)
            with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer):
                identity = oe._zero_formal_candidate_identity(current_path, root)
                lock = oe.zero_formal_campaign_lock(pointer, identity["campaign_id"])
                lock.__enter__()
                try:
                    with patch.object(oe, "snapshot_gpus") as snapshot:
                        args.output_root = base / "evidence-two"
                        with self.assertRaisesRegex(oe.LaunchError, "already running"):
                            oe.command_run_campaign(args)
                    snapshot.assert_not_called()
                    self.assertFalse((base / "evidence-two").exists())
                finally:
                    lock.__exit__(None, None, None)
            with oe.zero_formal_campaign_lock(pointer, identity["campaign_id"]):
                pass
            self.assertEqual(stat.S_IMODE((base / ".locks").stat().st_mode), 0o700)

    def test_dirty_or_unrelated_host_checkout_fails_closed(self) -> None:
        for condition in ("dirty", "unrelated"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as td:
                base = Path(td)
                root, host = base / "candidate", base / "host"
                root.mkdir()
                current_path = self.candidate_checkout(root, host)
                pointer = self.external_pointer(base, host)
                if condition == "dirty":
                    (host / "uncommitted.txt").write_text("dirty\n", encoding="utf-8")
                    expected = "canonical host control checkout must be clean"
                else:
                    subprocess.run(["git", "-C", str(host), "checkout", "--orphan", "unrelated"], check=True, capture_output=True)
                    subprocess.run(["git", "-C", str(host), "add", "-A"], check=True)
                    subprocess.run(["git", "-C", str(host), "commit", "-qm", "unrelated history"], check=True)
                    expected = "candidate execution SHA is not contained"
                with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer), \
                     patch.object(oe, "snapshot_gpus") as snapshot:
                    with self.assertRaisesRegex(oe.LaunchError, expected):
                        oe._zero_formal_candidate_identity(current_path, root)
                snapshot.assert_not_called()

    def test_missing_profile_writes_blocked_receipt_before_gpu_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, host = base / "candidate", base / "host"
            root.mkdir()
            current_path = self.candidate_checkout(root, host)
            pointer = self.external_pointer(base, host)
            args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=None, gpus="GPU-aaa", output_root=base / "evidence", preflight_only=True)
            with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer), \
                 patch.object(oe, "snapshot_gpus") as snapshot:
                with self.assertRaisesRegex(oe.LaunchError, "requires --preflight-profile"):
                    oe.command_run_campaign(args)
            snapshot.assert_not_called()
            receipts = list((base / "evidence").rglob("zero-formal-qualification-attempt-*.json"))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(receipt["status"], "ZERO_FORMAL_BLOCKED")
            self.assertIsNone(receipt["candidate_hashes"]["runtime_profile"])
            self.assertIn("requires --preflight-profile", receipt["failure_reason"])
            self.assertFalse(receipt["formal_task_consumed"])
            self.assertEqual(receipt["formal_task_budget"], 0)

    def test_unreadable_profile_writes_blocked_receipt_without_masking_profile_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, host = base / "candidate", base / "host"
            root.mkdir()
            current_path = self.candidate_checkout(root, host)
            pointer = self.external_pointer(base, host)
            bad_profile = base / "broken-profile.json"
            bad_profile.write_text("{ invalid json", encoding="utf-8")
            args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=bad_profile, gpus="GPU-aaa", output_root=base / "evidence", preflight_only=True)
            with patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer), \
                 patch.object(oe, "snapshot_gpus") as snapshot:
                with self.assertRaisesRegex(oe.LaunchError, "profile unreadable") as caught:
                    oe.command_run_campaign(args)
            snapshot.assert_not_called()
            receipts = list((base / "evidence").rglob("zero-formal-qualification-attempt-*.json"))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(receipt["status"], "ZERO_FORMAL_BLOCKED")
            self.assertEqual(receipt["failure_reason"], str(caught.exception))
            self.assertEqual(receipt["candidate_hashes"]["runtime_profile"], pre.sha256_file(bad_profile))

    def test_post_identity_validation_image_gpu_and_smoke_errors_are_recorded(self) -> None:
        scenarios = (
            ("validation", OSError("validation filesystem unavailable")),
            ("image", subprocess.CalledProcessError(17, ["docker", "image", "inspect"])),
            ("gpu", oe.LaunchError("live GPU gate failed")),
            ("smoke", oe.PrelaunchEngineeringError("CUDA smoke failed")),
        )
        for stage, original_error in scenarios:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as td:
                base = Path(td)
                root, host = base / "candidate", base / "host"
                root.mkdir()
                current_path = self.candidate_checkout(root, host)
                pointer = self.external_pointer(base, host)
                head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
                profile_path = self.profile_file(base, head)
                args = Namespace(repo_root=root, current_campaign=current_path, preflight_profile=profile_path, gpus="GPU-aaa", output_root=base / "evidence", preflight_only=True)
                patches: list[Any] = [
                    patch.object(oe, "CANONICAL_WORKSPACE_REGISTRY_POINTER", pointer),
                ]
                if stage == "validation":
                    patches.append(patch.object(oe, "validate_sources_and_paths", side_effect=original_error))
                else:
                    patches.append(patch.object(oe, "validate_sources_and_paths"))
                if stage == "image":
                    patches.append(patch.object(oe, "resolve_image_identity", side_effect=original_error))
                else:
                    patches.append(patch.object(oe, "resolve_image_identity", return_value="sha256:" + "a" * 64))
                if stage == "gpu":
                    patches.append(patch.object(oe, "snapshot_gpus", side_effect=original_error))
                else:
                    patches.append(patch.object(oe, "snapshot_gpus", return_value=[{"uuid": "GPU-aaa"}]))
                patches.append(patch.object(oe, "select_gpu_uuids", return_value=["GPU-aaa"]))
                patches.append(patch.object(oe, "wait_for_idle", return_value=[]))
                if stage == "smoke":
                    def write_failed_prelaunch(profile: dict, uuids: list[str], run_dir: Path, image_id: str, control_dir: Path) -> None:
                        failed = pre.build_prelaunch_receipt(
                            run_id="synthetic", attempt=1, selected_gpu_uuids=uuids,
                            image_id=image_id, command_identity="sha256:" + "b" * 64,
                            return_code=43, wall_seconds=0.02, payload={**pass_payload(), "status": "FAIL", "stage": "CUDA_KERNEL"},
                            payload_error=None, stderr_tail="synthetic CUDA failure",
                        )
                        oe.atomic_json(control_dir / "prelaunch-zero-formal-attempt-0001.json", failed)
                        raise original_error
                    patches.append(patch.object(oe, "run_selected_gpu_preflight", side_effect=write_failed_prelaunch))
                else:
                    patches.append(patch.object(oe, "run_selected_gpu_preflight", side_effect=AssertionError("unexpected smoke")))
                with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                    with self.assertRaises(type(original_error)) as caught:
                        oe.command_run_campaign(args)
                self.assertIs(caught.exception, original_error)
                receipts = list((base / "evidence").rglob("zero-formal-qualification-attempt-*.json"))
                self.assertEqual(len(receipts), 1)
                receipt = json.loads(receipts[0].read_text())
                self.assertEqual(receipt["status"], "ZERO_FORMAL_BLOCKED")
                self.assertEqual(receipt["failure_reason"], str(original_error))
                if stage == "smoke":
                    details = next((base / "evidence").rglob("prelaunch-zero-formal-attempt-0001.json"))
                    self.assertEqual(receipt["prelaunch_receipt_file_sha256"], pre.sha256_file(details))
                    detailed_receipt = json.loads(details.read_text())
                    self.assertEqual(receipt["prelaunch_receipt_sha256"], detailed_receipt["receipt_sha256"])


if __name__ == "__main__":
    unittest.main()


def test_host_selected_preflight_uses_declared_host_python_and_run_identity():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        profile = workspace_profile(root)
        profile.update({"identity": {"uid": 1001, "gid": 1001},
                        "execution": {"mode": "host-orchestrator"},
                        "_host_runtime_source": {"source_root": str(root / "sealed"), "sha256": "a" * 64,
                                                 "target": "/workspace/seed", "receipt_sha256": "b" * 64}})
        run = root / "runs" / profile["run_id"]
        run.mkdir()
        observed = []
        def capture(argv, check=False):
            observed.append(argv)
            return SimpleNamespace(returncode=0, stdout=json.dumps(pass_payload()), stderr="")
        def wrong_route(*args, **kwargs):
            raise AssertionError("host interpreter must not be executed inside an unrelated Docker filesystem")
        with patch.object(oe.os, "geteuid", return_value=0), \
             patch.object(oe, "validate_host_runtime_source", return_value=None), \
             patch.object(oe, "docker_command", side_effect=wrong_route), \
             patch.object(oe, "_run", side_effect=capture):
            result = oe.run_selected_gpu_preflight(profile, ["GPU-aaa"], run, profile["image_id"], root)
        assert result["receipt"]["status"] == "PASS"
        argv = observed[0]
        assert argv[0] == "/usr/bin/env"
        assert "CUDA_VISIBLE_DEVICES=GPU-aaa" in argv
        assert "--gpus" not in argv
        assert argv[argv.index("-I") - 1] == "/usr/bin/python3"
        assert json.loads(argv[-1])[0] == profile["command"][0]
        assert "os.setuid" in " ".join(argv)
        assert str(run) in " ".join(argv)
        assert "/workspace/runs/" not in " ".join(argv)


def test_host_preflight_rejects_privileged_identity_and_wrong_interpreter():
    profile = {"identity": {"uid": 0, "gid": 1001}, "command": ["python3"],
               "execution": {"mode": "host-orchestrator", "runtime_python": "python3"}}
    case = unittest.TestCase()
    with patch.object(oe.os, "geteuid", return_value=0):
        with case.assertRaisesRegex(oe.PrelaunchEngineeringError, "positive run UID/GID"):
            oe.host_selected_gpu_preflight_command(profile, ["GPU-aaa"], ["python3", "-c", "pass"])
        profile["identity"]["uid"] = 1001
        with case.assertRaisesRegex(oe.PrelaunchEngineeringError, "declared scientific interpreter"):
            oe.host_selected_gpu_preflight_command(profile, ["GPU-aaa"], ["/other/python", "-c", "pass"])
    with patch.object(oe.os, "geteuid", return_value=1001):
        with case.assertRaisesRegex(oe.PrelaunchEngineeringError, "root control plane"):
            oe.host_selected_gpu_preflight_command(profile, ["GPU-aaa"], ["python3", "-c", "pass"])


def test_current_step119_holder_hold_blocks_historical_launch_release():
    from scripts.openevo_webshop import execution_readiness as readiness
    router_path = ROOT / "configs/experiment/activations/q03-step119-val128-current-campaign-20261025.json"
    router = json.loads(router_path.read_text())
    release_path = ROOT / router["release_receipt"]
    release = json.loads(release_path.read_text())
    with unittest.TestCase().assertRaisesRegex(readiness.ExecutionReadinessError, "not executable"):
        readiness.verify_current_authority(release, release_path, current_campaign_path=router_path, repo_root=ROOT)


def test_host_preflight_bootstrap_drops_identity_and_untrusted_cwd_before_exec():
    import sys
    state={"uid":0,"gid":0,"groups":[0],"cwd":"/untrusted-shadow-modules"}
    def store(k):
        return lambda value: state.__setitem__(k,value)
    observed=[]
    def child_exec(exe,argv,env):
        assert (state["uid"],state["gid"],state["groups"],state["cwd"]) == (1001,1001,[],"/")
        observed.append((exe,argv,env))
    fake_os=SimpleNamespace(setgroups=store("groups"),setgid=store("gid"),setuid=store("uid"),
        geteuid=lambda:state["uid"],getegid=lambda:state["gid"],getgroups=lambda:state["groups"],
        chdir=store("cwd"),execvpe=child_exec,environ={"CUDA_VISIBLE_DEVICES":"GPU-aaa"})
    profile={"identity":{"uid":1001,"gid":1001},"command":["/declared/python"],"execution":{"mode":"host-orchestrator"}}
    payload=["/declared/python","-c","import torch"]
    with patch.object(oe.os,"geteuid",return_value=0), patch.object(oe,"validate_host_runtime_source",return_value=None):
        argv=oe.host_selected_gpu_preflight_command(profile,["GPU-aaa"],payload)
    offset=argv.index("-c")
    fake_sys=SimpleNamespace(argv=["-c",*argv[offset+2:]])
    with patch.dict(sys.modules,{"os":fake_os,"sys":fake_sys}):
        exec(argv[offset+1],{})
    assert observed == [(payload[0],payload,fake_os.environ)]
