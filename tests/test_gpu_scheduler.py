import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts.openevo_webshop import schedule_gpu_cells as scheduler


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "openevo_webshop" / "schedule_gpu_cells.py"


def _env_with_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    return env


class GpuSchedulerTests(unittest.TestCase):
    def test_output_root_claim_is_atomic_across_launchers(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "shared-output"
            barrier = threading.Barrier(2)
            outcomes: list[str] = []

            def claim() -> None:
                barrier.wait()
                try:
                    scheduler.claim_output_root(output)
                except SystemExit:
                    outcomes.append("blocked")
                else:
                    outcomes.append("claimed")

            threads = [threading.Thread(target=claim) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            self.assertEqual(sorted(outcomes), ["blocked", "claimed"])

    def test_atomic_receipt_preserves_previous_file_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scheduler-receipt.json"
            previous = {"schema": "previous", "cells": [{"cell_id": "sealed"}]}
            path.write_text(json.dumps(previous), encoding="utf-8")

            with mock.patch.object(scheduler.os, "replace", side_effect=OSError("simulated crash")):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    scheduler.atomic_json(path, {"schema": "next", "cells": []})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), previous)
            self.assertEqual(list(path.parent.glob(".scheduler-receipt.json.*.tmp")), [])

    def test_scheduler_refuses_nonempty_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "resource_assignment_mode": "dynamic-live-idle",
                        "formal_task_consumption_allowed": False,
                        "cells": [{"cell_id": "a", "argv": ["true"]}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "out"
            output.mkdir()
            marker = output / "existing-receipt.json"
            marker.write_text('{"status":"preserve"}\n', encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--plan",
                    str(plan),
                    "--output-root",
                    str(output),
                    "--runtime-gpu-uuids",
                    "GPU-0",
                    "--poll-seconds",
                    "0.001",
                    "--timeout-seconds",
                    "0.01",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=_env_with_pythonpath(),
                timeout=5,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output root is not empty", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), '{"status":"preserve"}\n')

    def test_v2_plan_requires_explicit_resource_assignment_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "contract_version": scheduler.PLAN_CONTRACT_VERSION,
                        "formal_task_consumption_allowed": False,
                        "cells": [{"cell_id": "a", "argv": ["true"]}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "explicitly declare resource_assignment_mode"):
                scheduler.load_plan(path)

    def test_v2_template_defaults_to_dynamic_live_idle(self):
        template = json.loads((ROOT / "templates" / "gpu-plan.v2.example.json").read_text(encoding="utf-8"))
        self.assertEqual(template["contract_version"], scheduler.PLAN_CONTRACT_VERSION)
        self.assertEqual(template["resource_assignment_mode"], scheduler.DYNAMIC_MODE)
        self.assertTrue(all("gpu_uuid" not in cell for cell in template["cells"]))

    def test_legacy_fixed_plan_preserves_exact_gpu_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "formal_task_consumption_allowed": False,
                        "cells": [
                            {"cell_id": "a", "gpu_uuid": "GPU-0", "argv": ["true"]},
                            {"cell_id": "b", "gpu_uuid": "GPU-0", "argv": ["true"]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            document, cells = scheduler.load_plan(path)
        self.assertEqual(scheduler.assignment_mode(document), scheduler.FIXED_MODE)
        self.assertEqual([cell["gpu_uuid"] for cell in cells], ["GPU-0", "GPU-0"])

    def test_dynamic_plan_requires_resource_neutral_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.json"
            valid.write_text(
                json.dumps(
                    {
                        "resource_assignment_mode": "dynamic-live-idle",
                        "formal_task_consumption_allowed": False,
                        "cells": [
                            {"cell_id": "a", "argv": ["true"]},
                            {"cell_id": "b", "argv": ["true"]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            document, cells = scheduler.load_plan(valid)
            self.assertEqual(scheduler.assignment_mode(document), scheduler.DYNAMIC_MODE)
            self.assertTrue(all("gpu_uuid" not in cell for cell in cells))

            invalid = root / "invalid.json"
            invalid.write_text(
                json.dumps(
                    {
                        "resource_assignment_mode": "dynamic-live-idle",
                        "formal_task_consumption_allowed": False,
                        "cells": [{"cell_id": "a", "gpu_uuid": "GPU-0", "argv": ["true"]}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must omit gpu_uuid"):
                scheduler.load_plan(invalid)

    def test_dry_run_never_queries_gpu_and_reports_resource_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "resource_assignment_mode": "dynamic-live-idle",
                        "formal_task_consumption_allowed": False,
                        "cells": [{"cell_id": "a", "argv": ["true"]}],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--plan",
                    str(plan),
                    "--output-root",
                    str(root / "out"),
                    "--runtime-gpu-uuids",
                    "GPU-0,GPU-1",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=_env_with_pythonpath(),
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"status": "dry-run"', result.stdout)
        self.assertIn('"resource_assignment_mode": "dynamic-live-idle"', result.stdout)
        self.assertIn('"GPU-0"', result.stdout)

    def _write_formal_plan_and_receipt(
        self,
        root: Path,
        *,
        mode: str = "dynamic-live-idle",
        plan_reassignment: bool = True,
        receipt_reassignment: bool = True,
        include_manifest: bool = True,
    ) -> Path:
        contract_sha = "sha256:test-contract"
        receipt = root / "gate.json"
        receipt.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "formal_task_consumption_allowed": True,
                    "contract_sha256": contract_sha,
                    "gpu_reassignment_allowed": receipt_reassignment,
                }
            ),
            encoding="utf-8",
        )
        cell = {"cell_id": "a", "argv": ["true"]}
        if mode == "fixed":
            cell["gpu_uuid"] = "GPU-0"
        if include_manifest:
            cell["run_manifest"] = "run.json"
        plan_payload = {
            "contract_version": scheduler.PLAN_CONTRACT_VERSION,
            "resource_assignment_mode": mode,
            "formal_task_consumption_allowed": True,
            "contract_sha256": contract_sha,
            "launch_gate_receipt": "gate.json",
            "cells": [cell],
        }
        if mode == "dynamic-live-idle":
            plan_payload["gpu_reassignment_allowed"] = plan_reassignment
        plan = root / "plan.json"
        plan.write_text(json.dumps(plan_payload), encoding="utf-8")
        return plan

    def test_formal_permission_comes_from_hash_bound_receipt_not_allow_formal_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._write_formal_plan_and_receipt(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--plan",
                    str(plan),
                    "--output-root",
                    str(root / "out"),
                    "--runtime-gpu-uuids",
                    "GPU-0,GPU-1",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=_env_with_pythonpath(),
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"basis": "hash-bound-launch-gate-receipt"', result.stdout)
        self.assertNotIn("--allow-formal", result.stderr)

    def test_formal_plan_still_requires_manifest_and_hash_bound_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._write_formal_plan_and_receipt(root, include_manifest=False)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--plan",
                    str(plan),
                    "--output-root",
                    str(root / "out"),
                    "--runtime-gpu-uuids",
                    "GPU-0",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=_env_with_pythonpath(),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("run_manifest", result.stderr)

            missing_receipt = root / "missing-receipt.json"
            missing_receipt.write_text(
                json.dumps(
                    {
                        "formal_task_consumption_allowed": True,
                        "cells": [
                            {
                                "cell_id": "a",
                                "gpu_uuid": "GPU-0",
                                "argv": ["true"],
                                "run_manifest": "run.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--plan",
                    str(missing_receipt),
                    "--output-root",
                    str(root / "out2"),
                    "--runtime-gpu-uuids",
                    "GPU-0",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=_env_with_pythonpath(),
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("launch_gate_receipt", result.stderr)

    def test_formal_v2_dynamic_does_not_require_duplicate_reassignment_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._write_formal_plan_and_receipt(root, plan_reassignment=False, receipt_reassignment=False)
            document, _ = scheduler.load_plan(plan)
            result = scheduler.validate_formal_launch_gate(document, plan)
            self.assertEqual(result["resource_assignment_mode"], scheduler.DYNAMIC_MODE)
            self.assertNotIn("gpu_reassignment_allowed", result)

    def test_historical_dynamic_contract_with_old_reassignment_fields_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._write_formal_plan_and_receipt(root)
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload.pop("contract_version")
            plan.write_text(json.dumps(payload), encoding="utf-8")
            document, _ = scheduler.load_plan(plan)
            result = scheduler.validate_formal_launch_gate(document, plan)
            self.assertEqual(result["resource_assignment_mode"], scheduler.DYNAMIC_MODE)

    def test_v2_fixed_plan_requires_nonempty_binding_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            payload = {
                "contract_version": scheduler.PLAN_CONTRACT_VERSION,
                "resource_assignment_mode": "fixed",
                "formal_task_consumption_allowed": False,
                "cells": [{"cell_id": "a", "gpu_uuid": "GPU-0", "argv": ["true"]}],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed_gpu_binding_reason"):
                scheduler.load_plan(path)
            payload["fixed_gpu_binding_reason"] = "heterogeneous GPU comparison"
            path.write_text(json.dumps(payload), encoding="utf-8")
            document, _ = scheduler.load_plan(path)
            self.assertEqual(document["fixed_gpu_binding_reason"], "heterogeneous GPU comparison")

    def test_v2_fixed_formal_receipt_preserves_reason_and_exact_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_sha = "sha256:fixed-contract"
            receipt = root / "gate.json"
            receipt.write_text(
                json.dumps({
                    "status": "pass",
                    "formal_task_consumption_allowed": True,
                    "contract_sha256": contract_sha,
                    "fixed_gpu_binding_reason": "device-specific reproducibility investigation",
                }),
                encoding="utf-8",
            )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps({
                    "contract_version": scheduler.PLAN_CONTRACT_VERSION,
                    "resource_assignment_mode": "fixed",
                    "fixed_gpu_binding_reason": "device-specific reproducibility investigation",
                    "formal_task_consumption_allowed": True,
                    "contract_sha256": contract_sha,
                    "launch_gate_receipt": "gate.json",
                    "cells": [{"cell_id": "a", "gpu_uuid": "GPU-exact", "argv": ["true"], "run_manifest": "run.json"}],
                }),
                encoding="utf-8",
            )
            document, cells = scheduler.load_plan(plan)
            gate = scheduler.validate_formal_launch_gate(document, plan)
            self.assertEqual(cells[0]["gpu_uuid"], "GPU-exact")
            self.assertEqual(gate["fixed_gpu_binding_reason"], document["fixed_gpu_binding_reason"])

    def test_runtime_candidates_ignore_historical_plan_uuid_subset(self):
        document = {
            "contract_version": scheduler.PLAN_CONTRACT_VERSION,
            "resource_assignment_mode": scheduler.DYNAMIC_MODE,
            "authorized_gpu_uuids": ["GPU-historical-0", "GPU-historical-1"],
        }
        self.assertEqual(
            scheduler.resolve_runtime_candidates(document, {"GPU-current-0", "GPU-current-1"}),
            {"GPU-current-0", "GPU-current-1"},
        )

    def test_idle_defaults_are_strict(self):
        self.assertEqual(scheduler.DEFAULT_MAX_IDLE_MEMORY_MIB, 16)
        self.assertEqual(scheduler.DEFAULT_MAX_IDLE_UTIL_PERCENT, 0)
        self.assertFalse(
            scheduler.idle(
                {"memory_used_mib": 17, "utilization_percent": 0, "compute_apps": []},
                max_memory_mib=16,
                max_util_percent=0,
            )
        )
        self.assertFalse(
            scheduler.idle(
                {"memory_used_mib": 0, "utilization_percent": 1, "compute_apps": []},
                max_memory_mib=16,
                max_util_percent=0,
            )
        )
        self.assertFalse(
            scheduler.idle(
                {"memory_used_mib": 0, "utilization_percent": 0, "compute_apps": [], "compute_apps_query_error": "unknown"},
                max_memory_mib=16,
                max_util_percent=0,
            )
        )
        self.assertFalse(
            scheduler.idle(
                {"memory_used_mib": 0, "utilization_percent": 0, "compute_apps": [], "device_handles": [{"pid": "1"}]},
                max_memory_mib=16,
                max_util_percent=0,
            )
        )
        self.assertTrue(
            scheduler.idle(
                {
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                    "compute_apps": [],
                    "device_handles": [{"user": "root", "comm": "nvitop", "exe": "/usr/bin/nvitop", "cmdline_programs": "nvitop"}],
                },
                max_memory_mib=16,
                max_util_percent=0,
            )
        )

    def _dynamic_args(self, output_root: Path, *, poll_seconds: float = 0.001, max_concurrent_cells: int | None = None):
        return argparse.Namespace(
            output_root=output_root,
            max_idle_memory_mib=16,
            max_idle_util_percent=0,
            # Functional scheduler tests must not depend on sub-20ms thread scheduling.
            timeout_seconds=1.0,
            poll_seconds=poll_seconds,
            cwd=None,
            cell_timeout_seconds=0.0,
            max_concurrent_cells=max_concurrent_cells,
        )

    def test_dynamic_scheduler_backfills_two_idle_cards_from_eight_pool(self):
        cells = [{"cell_id": f"cell-{index}", "argv": ["true"]} for index in range(2)]
        assigned: list[tuple[str, str]] = []
        concurrent_launches = threading.Barrier(2)
        original_wait = scheduler.wait_for_idle
        original_snapshot = scheduler.gpu_snapshot
        original_run_cell = scheduler.run_cell

        def fake_wait(uuid, args, *, work_remaining=None):
            if uuid in {"GPU-0", "GPU-1", "GPU-2", "GPU-3", "GPU-4", "GPU-6"}:
                raise TimeoutError("busy")
            return {"uuid": uuid, "memory_used_mib": 0, "utilization_percent": 0, "compute_apps": []}

        def fake_snapshot(uuid):
            return {"uuid": uuid, "memory_used_mib": 0, "utilization_percent": 0, "compute_apps": []}

        def fake_run_cell(cell, gpu_uuid, args, receipt, lock, *, pre_gpu=None):
            assigned.append((cell["cell_id"], gpu_uuid))
            # Hold the first synthetic cell open until a second GPU worker starts.
            # Without this barrier the zero-duration fake lets one thread consume
            # both queue entries before another idle worker gets CPU time, which
            # tests thread scheduling rather than concurrent GPU backfill.
            concurrent_launches.wait(timeout=1.0)
            scheduler.append_receipt(receipt, lock, args.output_root, {"cell_id": cell["cell_id"], "assigned_gpu_uuid": gpu_uuid, "status": "pass"})

        try:
            scheduler.wait_for_idle, scheduler.gpu_snapshot, scheduler.run_cell = fake_wait, fake_snapshot, fake_run_cell
            with tempfile.TemporaryDirectory() as tmp:
                receipt = {"cells": [], "resource_events": []}
                scheduler.run_dynamic(cells, {f"GPU-{index}" for index in range(8)}, self._dynamic_args(Path(tmp), max_concurrent_cells=2), receipt, threading.Lock())
            self.assertEqual({uuid for _, uuid in assigned}, {"GPU-5", "GPU-7"})
            self.assertEqual(len(assigned), 2)
        finally:
            scheduler.wait_for_idle, scheduler.gpu_snapshot, scheduler.run_cell = original_wait, original_snapshot, original_run_cell

    def test_busy_gpu_that_becomes_idle_later_backfills_without_restart(self):
        cells = [{"cell_id": "late", "argv": ["true"]}]
        calls = {"GPU-late": 0}
        assigned: list[str] = []
        original_wait = scheduler.wait_for_idle
        original_snapshot = scheduler.gpu_snapshot
        original_run_cell = scheduler.run_cell

        def fake_wait(uuid, args, *, work_remaining=None):
            calls[uuid] += 1
            if calls[uuid] < 2:
                time.sleep(0.002)
                raise TimeoutError("busy sample")
            return {"uuid": uuid, "memory_used_mib": 0, "utilization_percent": 0, "compute_apps": []}

        def fake_snapshot(uuid):
            return {"uuid": uuid, "memory_used_mib": 0, "utilization_percent": 0, "compute_apps": []}

        def fake_run_cell(cell, gpu_uuid, args, receipt, lock, *, pre_gpu=None):
            assigned.append(gpu_uuid)
            scheduler.append_receipt(receipt, lock, args.output_root, {"cell_id": cell["cell_id"], "assigned_gpu_uuid": gpu_uuid, "status": "pass"})

        try:
            scheduler.wait_for_idle, scheduler.gpu_snapshot, scheduler.run_cell = fake_wait, fake_snapshot, fake_run_cell
            with tempfile.TemporaryDirectory() as tmp:
                receipt = {"cells": [], "resource_events": []}
                scheduler.run_dynamic(cells, {"GPU-late"}, self._dynamic_args(Path(tmp), poll_seconds=0.001), receipt, threading.Lock())
            self.assertEqual(assigned, ["GPU-late"])
            self.assertGreaterEqual(calls["GPU-late"], 2)
        finally:
            scheduler.wait_for_idle, scheduler.gpu_snapshot, scheduler.run_cell = original_wait, original_snapshot, original_run_cell

    def test_completed_lane_refills_next_pending_cell(self):
        cells = [{"cell_id": "a", "argv": ["true"]}, {"cell_id": "b", "argv": ["true"]}]
        assigned: list[str] = []
        original_wait = scheduler.wait_for_idle
        original_snapshot = scheduler.gpu_snapshot
        original_run_cell = scheduler.run_cell

        def fake_wait(uuid, args, *, work_remaining=None):
            return {"uuid": uuid, "memory_used_mib": 0, "utilization_percent": 0, "compute_apps": []}

        def fake_snapshot(uuid):
            return {"uuid": uuid, "memory_used_mib": 0, "utilization_percent": 0, "compute_apps": []}

        def fake_run_cell(cell, gpu_uuid, args, receipt, lock, *, pre_gpu=None):
            assigned.append(cell["cell_id"])
            scheduler.append_receipt(receipt, lock, args.output_root, {"cell_id": cell["cell_id"], "assigned_gpu_uuid": gpu_uuid, "status": "pass"})

        try:
            scheduler.wait_for_idle, scheduler.gpu_snapshot, scheduler.run_cell = fake_wait, fake_snapshot, fake_run_cell
            with tempfile.TemporaryDirectory() as tmp:
                receipt = {"cells": [], "resource_events": []}
                scheduler.run_dynamic(cells, {"GPU-only-lane"}, self._dynamic_args(Path(tmp)), receipt, threading.Lock())
            self.assertEqual(assigned, ["a", "b"])
        finally:
            scheduler.wait_for_idle, scheduler.gpu_snapshot, scheduler.run_cell = original_wait, original_snapshot, original_run_cell

    def test_no_pending_cells_do_not_use_idle_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = {"cells": [], "resource_events": []}
            scheduler.run_dynamic([], {"GPU-idle"}, self._dynamic_args(Path(tmp)), receipt, threading.Lock())
        self.assertEqual(receipt["cells"], [])

    def test_capacity_starvation_emits_after_threshold_without_cap(self):
        event = scheduler.capacity_starvation_event(
            candidate_uuid="GPU-idle",
            pending_cell_count=2,
            eligible_since_monotonic=10.0,
            observed_monotonic=131.0,
            poll_seconds=30.0,
            assigned=False,
            max_concurrent_cells=None,
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["event"], "capacity_starvation")
        self.assertEqual(event["backfill_latency_seconds"], 121.0)

    def test_concurrency_cap_suppresses_capacity_starvation(self):
        self.assertIsNone(
            scheduler.capacity_starvation_event(
                candidate_uuid="GPU-idle",
                pending_cell_count=2,
                eligible_since_monotonic=10.0,
                observed_monotonic=131.0,
                poll_seconds=30.0,
                assigned=False,
                max_concurrent_cells=1,
            )
        )

    def test_dynamic_scheduler_busy_lane_does_not_block_idle_lane_backfill(self):
        cells = [
            {"cell_id": "a", "argv": ["true"]},
            {"cell_id": "b", "argv": ["true"]},
            {"cell_id": "c", "argv": ["true"]},
        ]
        assigned: list[tuple[str, str]] = []
        assigned_lock = threading.Lock()
        original_wait = scheduler.wait_for_idle
        original_snapshot = scheduler.gpu_snapshot
        original_run_cell = scheduler.run_cell

        def fake_wait(uuid, args, *, work_remaining=None):
            if uuid == "GPU-busy":
                raise TimeoutError("busy")
            return {
                "uuid": uuid,
                "memory_used_mib": 0,
                "utilization_percent": 0,
                "compute_apps": [],
                "observed_at_utc": scheduler.utc_now(),
            }

        def fake_snapshot(uuid):
            return {
                "uuid": uuid,
                "memory_used_mib": 0,
                "utilization_percent": 0,
                "compute_apps": [],
                "observed_at_utc": scheduler.utc_now(),
            }

        def fake_run_cell(cell, gpu_uuid, args, receipt, lock, *, pre_gpu=None):
            with assigned_lock:
                assigned.append((cell["cell_id"], gpu_uuid))
            scheduler.append_receipt(
                receipt,
                lock,
                args.output_root,
                {
                    "cell_id": cell["cell_id"],
                    "requested_gpu_uuid": None,
                    "assigned_gpu_uuid": gpu_uuid,
                    "argv": cell["argv"],
                    "start_utc": scheduler.utc_now(),
                    "end_utc": scheduler.utc_now(),
                    "status": "pass",
                },
            )

        try:
            scheduler.wait_for_idle = fake_wait
            scheduler.gpu_snapshot = fake_snapshot
            scheduler.run_cell = fake_run_cell
            with tempfile.TemporaryDirectory() as tmp:
                args = argparse.Namespace(
                    output_root=Path(tmp),
                    max_idle_memory_mib=16,
                    max_idle_util_percent=0,
                    # This test exercises lane backfill, not a 10ms scheduler SLA.
                    timeout_seconds=1.0,
                    poll_seconds=0.001,
                    cwd=None,
                    cell_timeout_seconds=0.0,
                )
                receipt = {"cells": [], "resource_events": []}
                scheduler.run_dynamic(
                    cells,
                    {"GPU-busy", "GPU-idle"},
                    args,
                    receipt,
                    threading.Lock(),
                )
                self.assertEqual(sorted(cell_id for cell_id, _ in assigned), ["a", "b", "c"])
                self.assertEqual({uuid for _, uuid in assigned}, {"GPU-idle"})
                self.assertTrue(
                    any(
                        event.get("uuid") == "GPU-busy" and event.get("status") in {"poll_retry", "unavailable"}
                        for event in receipt["resource_events"]
                    )
                )
                self.assertFalse(any(row["status"] != "pass" for row in receipt["cells"]))
        finally:
            scheduler.wait_for_idle = original_wait
            scheduler.gpu_snapshot = original_snapshot
            scheduler.run_cell = original_run_cell


if __name__ == "__main__":
    unittest.main()
