"""CPU-only safety checks for the reviewed replacement of the Q04 sole writer.

The real server script/lock/process is never imported, stopped or modified.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.openevo_runtime import native_advisor_task_pool as pool


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def owner_snapshot(*, target_owner="idle", attribution="IDLE",
                   used=2, source="verified-observer", age=0):
    now = datetime.now(timezone.utc) - timedelta(seconds=age)
    cards = []
    for i in range(8):
        row = {"index": i, "uuid": pool.GPU_UUIDS[i] if i < 4 else f"GPU-other-{i}",
               "memory_used_mib": 2, "memory_total_mib": 32607,
               "utilization_gpu_percent": 0, "owner": "idle",
               "owners": [], "attribution": "IDLE"}
        if i == 0:
            row.update(owner=target_owner,
                       owners=[] if target_owner == "idle" else [target_owner],
                       attribution=attribution, memory_used_mib=used)
        cards.append(row)
    return {"schema": "zju.gpu-ownership-snapshot.v1", "mutation": False,
            "execution_authority": False, "gpu_attribution_source": source,
            "gpu_attribution_error": None, "unresolved_global_gpu_uuids": [],
            "observed_at_utc": now.isoformat(), "gpus": cards}


class ProcessInventoryTests(unittest.TestCase):
    def test_disappearing_pid_during_cmdline_read_is_skipped(self):
        # PID 123 vanishes after the top-level /proc readdir snapshot.
        # The remaining live PID still contributes to worker liveness.
        with patch.object(pool.os, "scandir") as scan, patch.object(
                pool.Path, "read_bytes",
                side_effect=[ProcessLookupError("PID 123 exited"), b"python\0worker\0"]):
            scan.return_value.__enter__.return_value = iter([
                SimpleNamespace(name="123"), SimpleNamespace(name="not-a-pid"),
                SimpleNamespace(name="456")])
            self.assertEqual(pool.processes(), [(456, ["python", "worker"])])

    def test_proc_root_enumeration_error_fails_closed(self):
        # Do not treat an unreadable/unstable entire /proc snapshot as idle.
        with patch.object(pool.os, "scandir", side_effect=OSError("/proc unavailable")):
            with self.assertRaisesRegex(RuntimeError, "PROCESS_INVENTORY_UNAVAILABLE"):
                pool.processes()


class OwnerSnapshotTests(unittest.TestCase):
    def test_verified_idle_has_exact_uuid_and_observation_identity(self):
        result = pool.verified_owner_admission(owner_snapshot(), pool.GPU_UUIDS[0])
        self.assertEqual(result["gpu_uuid"], pool.GPU_UUIDS[0])
        self.assertEqual(result["free_mib"], 32605)
        self.assertEqual(len(result["snapshot_sha256"]), 64)

    def test_peer_card_cannot_be_inferred_free(self):
        with self.assertRaisesRegex(ValueError, "GPU_OWNER_NOT_EXCLUSIVELY_IDLE"):
            pool.verified_owner_admission(
                owner_snapshot(target_owner="luht", attribution="EXPLICIT_BINDING", used=12000),
                pool.GPU_UUIDS[0])

    def test_unresolved_identity_and_false_idle_telemetry_refused(self):
        value = owner_snapshot()
        value["unresolved_global_gpu_uuids"] = [pool.GPU_UUIDS[0]]
        with self.assertRaises(ValueError):
            pool.verified_owner_admission(value, pool.GPU_UUIDS[0])
        with self.assertRaises(ValueError):
            pool.verified_owner_admission(owner_snapshot(used=6000), pool.GPU_UUIDS[0])

    def test_source_timestamp_authority_and_inventory_fail_closed(self):
        for bad in (
            owner_snapshot(source="disposable-fallback"),
            owner_snapshot(age=30),
            {**owner_snapshot(), "execution_authority": True},
            {**owner_snapshot(), "gpus": owner_snapshot()["gpus"][:-1]},
        ):
            with self.subTest(bad=bad.get("gpu_attribution_source")):
                with self.assertRaises(ValueError):
                    pool.verified_owner_admission(bad, pool.GPU_UUIDS[0])


class NativeReleaseTests(unittest.TestCase):
    def test_exact_q09_ready_inputs_and_scientific_pause(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            science = base / "science"
            science.mkdir()
            state_path = science / "execution-state.json"
            ready_path = base / "ready.json"
            native = {"schema": "openevo.ceiling1-stage2-execution-state.v1",
                      "status": "RUNNING", "next_round": 134,
                      "rounds_complete": list(range(134)), "rollouts_consumed": 17152, "final_panel_access_count": 0,
                      "external_teacher_calls": 0}
            ready = {"schema": "openevo.execution-readiness.v1",
                     "formal_task_consumption_allowed": True,
                     "authority": {"router_status": "launch-authorized"},
                     "stages": [{"stage": "current-authority", "verdict": "PASS"}]}
            state_path.write_text(json.dumps(native))
            ready_path.write_text(json.dumps(ready))
            task = {"id": "q09-s2-r134-exact-test", "native_campaign": {
                "readiness_receipt": str(ready_path), "science_root": str(science),
                "next_round": 134, "rollouts_consumed": 17152},
                "source_files": {str(state_path): sha(state_path.read_bytes()),
                                 str(ready_path): sha(ready_path.read_bytes())}}
            pool.native_science_admission(task)
            native["status"] = "PAUSE_FOR_DIAGNOSTIC"
            state_path.write_text(json.dumps(native))
            with self.assertRaisesRegex(ValueError, "DRIFTED"):
                pool.native_science_admission(task)
            task["source_files"][str(state_path)] = sha(state_path.read_bytes())
            with self.assertRaisesRegex(ValueError, "NOT_UNSPENT_READY"):
                pool.native_science_admission(task)


    def test_q09_exact_ready_chain_launches_once_and_paused_chain_does_not(self):
        for science_status in ("RUNNING", "PAUSE_FOR_DIAGNOSTIC"):
            with self.subTest(science_status=science_status), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                source = root / "launcher.py"
                source.write_text("# no paid CUDA in test\n")
                science = root / "science"
                science.mkdir()
                state_file = science / "execution-state.json"
                receipt_file = root / "readiness.json"
                state_file.write_text(json.dumps({
                    "schema": "openevo.ceiling1-stage2-execution-state.v1",
                    "status": science_status, "next_round": 134,
                    "rounds_complete": list(range(134)), "rollouts_consumed": 17152,
                    "final_panel_access_count": 0, "external_teacher_calls": 0}))
                receipt_file.write_text(json.dumps({
                    "schema": "openevo.execution-readiness.v1",
                    "formal_task_consumption_allowed": True,
                    "authority": {"router_status": "launch-authorized"},
                    "stages": [{"verdict": "PASS"}]}))
                task_id = "q09-s2-r134-cpu-native-only"
                task = {"id": task_id, "mode": "queued", "gpu_uuid": pool.GPU_UUIDS[3],
                        "argv": ["python3", str(source)],
                        "worker_tokens": [str(source), task_id],
                        "source_files": {str(p): sha(p.read_bytes())
                                         for p in (source, state_file, receipt_file)},
                        "peak_mib": 256, "headroom_mib": 4096,
                        "output_root": str(root / "unused-output"),
                        "completion": {"path": str(root / "completion.json"),
                                       "equals": {"status": "PASS"}},
                        "native_campaign": {
                            "readiness_receipt": str(receipt_file),
                            "science_root": str(science),
                            "next_round": 134, "rollouts_consumed": 17152}}
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({
                    "schema": "openevo.advisor-task-pool.v1",
                    "authorized_gpu_uuids": list(pool.GPU_UUIDS),
                    "tasks": [task], "min_free_bytes": 0}))
                (root / "state.json").write_text(json.dumps({
                    "manifest_sha256": pool.digest(manifest),
                    "tasks": {task_id: {"status": "QUEUED"}}}))
                with patch.object(pool.subprocess, "Popen",
                                  return_value=type("Child", (), {"pid": 9123})()) as spawn:
                    result = pool.tick(
                        manifest, root,
                        gpu_query=lambda: {k: {"total": 32607, "free": 32605}
                                           for k in pool.GPU_UUIDS},
                        live_query=lambda: [],
                        owner_query=lambda uuid: pool.verified_owner_admission(
                            owner_snapshot(), uuid))
                current = result["tasks"][task_id]
                if science_status == "RUNNING":
                    self.assertEqual(current["status"], "RUNNING")
                    self.assertTrue((root / (task_id + "-dispatch.json")).exists())
                    self.assertEqual(spawn.call_count, 1)
                else:
                    self.assertEqual(current["status"], "QUEUED")
                    self.assertIn("SCIENCE_NOT_READY", current["wait_reason"])
                    self.assertFalse((root / (task_id + "-dispatch.json")).exists())
                    spawn.assert_not_called()


class NativeDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "frozen_launcher.py"
        self.source.write_text("# frozen CPU mock\n")
        self.output = self.root / "scientific_output"
        self.task_id = "q04-cpu-zero-budget-fixture"
        self.task = {
            "id": self.task_id, "mode": "queued", "gpu_uuid": pool.GPU_UUIDS[3],
            "argv": ["python3", str(self.source)],
            "worker_tokens": [str(self.source), self.task_id],
            "source_files": {str(self.source): sha(self.source.read_bytes())},
            "peak_mib": 256, "headroom_mib": 4096,
            "output_root": str(self.output),
            "completion": {"path": str(self.root / "seal.json"),
                           "equals": {"status": "PASS"}},
        }
        self.manifest = self.root / "manifest.json"
        self.tasks = [self.task]
        self.write_manifest()

    def write_manifest(self, min_free_bytes=0):
        self.manifest.write_text(json.dumps({
            "schema": pool.POOL_SCHEMA if hasattr(pool, "POOL_SCHEMA") else "openevo.advisor-task-pool.v1",
            "authorized_gpu_uuids": list(pool.GPU_UUIDS),
            "min_free_bytes": min_free_bytes,
            "tasks": self.tasks}))
        (self.root / "state.json").write_text(json.dumps({
            "manifest_sha256": pool.digest(self.manifest),
            "tasks": {item['id']: {"status": "QUEUED"} for item in self.tasks}}))

    def gpu(self):
        return {uuid: {"total": 32607, "free": 32605} for uuid in pool.GPU_UUIDS}

    def owner(self, uuid):
        return {"gpu_uuid": uuid, "free_mib": 32605,
                "snapshot_sha256": "a" * 64, "observed_at_utc": "2026-10-26T00:00:00+00:00"}

    def test_one_claim_one_launch_and_restart_never_replays(self):
        class Child:
            pid = 12345
        with patch.object(pool.subprocess, "Popen", return_value=Child()) as spawn:
            first = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                              live_query=lambda: [], owner_query=self.owner)
            second = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=self.owner)
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(first["tasks"][self.task_id]["status"], "RUNNING")
        self.assertEqual(first["tick_decision"]["code"], "DISPATCH_STARTED")
        self.assertEqual(first["tick_decision"]["spawned_task_ids"], [self.task_id])
        self.assertEqual(second["tasks"][self.task_id]["status"], "DIAGNOSTIC_REQUIRED")
        self.assertEqual(second["tick_decision"]["code"], "NO_REGISTERED_QUEUED_TASKS")
        self.assertTrue((self.root / (self.task_id + "-dispatch.json")).exists())
        self.assertTrue((self.root / (self.task_id + "-owner-admission.json")).exists())

    def test_complete_observation_clears_stale_queue_fields_but_keeps_receipt(self):
        # Real pool had six COMPLETE records with leftover admission reasons.
        self.task["mode"] = "hold"
        self.task["reason"] = "historic observer"
        self.write_manifest()
        marker = Path(self.task["completion"]["path"])
        marker.write_text(json.dumps({"status": "PASS"}))
        state_path = self.root / "state.json"
        previous = json.loads(state_path.read_text())
        previous["tasks"][self.task_id] = {
            "status": "QUEUED", "attempted": True,
            "wait_reason": "GPU memory admission",
            "reason": "past diagnostic", "pids": [987654]}
        state_path.write_text(json.dumps(previous))
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(
                self.manifest, self.root, gpu_query=self.gpu,
                live_query=lambda: [],
                owner_query=lambda _: self.fail("no owner query for complete"),
                candidate_query=lambda _: self.fail("no queue projection"))
        record = result["tasks"][self.task_id]
        self.assertEqual(record["status"], "COMPLETE")
        self.assertTrue(record["attempted"])
        self.assertEqual(record["completion_sha256"], sha(marker.read_bytes()))
        for stale in ("wait_reason", "reason", "pids"):
            self.assertNotIn(stale, record)
        spawn.assert_not_called()

    def test_running_observation_drops_old_diagnostic_not_durable_claim(self):
        state_path = self.root / "state.json"
        previous = json.loads(state_path.read_text())
        previous["tasks"][self.task_id] = {
            "status": "DIAGNOSTIC_REQUIRED", "attempted": True,
            "reason": "historical native gpu lock failure",
            "wait_reason": "GPU memory admission"}
        state_path.write_text(json.dumps(previous))
        live = [(24680, ["python3", str(self.source), self.task_id])]
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(
                self.manifest, self.root, gpu_query=self.gpu,
                live_query=lambda: live,
                owner_query=lambda _: self.fail("no owner query for running"),
                candidate_query=lambda _: self.fail("no queue projection"))
        record = result["tasks"][self.task_id]
        self.assertEqual(record["status"], "RUNNING")
        self.assertEqual(record["pids"], [24680])
        self.assertTrue(record["attempted"])
        self.assertNotIn("wait_reason", record)
        self.assertNotIn("reason", record)
        spawn.assert_not_called()

    def test_attempted_diagnostic_never_requeues_when_not_live(self):
        state_path = self.root / "state.json"
        previous = json.loads(state_path.read_text())
        previous["tasks"][self.task_id] = {
            "status": "DIAGNOSTIC_REQUIRED", "attempted": True,
            "reason": "original native gpu lock diagnostic",
            "wait_reason": "GPU memory admission"}
        state_path.write_text(json.dumps(previous))
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(
                self.manifest, self.root, gpu_query=self.gpu,
                live_query=lambda: [],
                owner_query=lambda _: self.fail("no owner query for diagnostic"),
                candidate_query=lambda _: self.fail("no queue projection"))
        record = result["tasks"][self.task_id]
        self.assertEqual(record["status"], "DIAGNOSTIC_REQUIRED")
        self.assertTrue(record["attempted"])
        self.assertEqual(result["tick_decision"]["code"], "NO_REGISTERED_QUEUED_TASKS")
        spawn.assert_not_called()

    def test_changed_science_projection_blocks_before_owner_and_claim(self):
        with patch.object(pool, "PROJECTION_SHA256", "0" * 64):
            with patch.object(pool.subprocess, "Popen") as spawn:
                state = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                                  live_query=lambda: [],
                                  owner_query=lambda _: self.fail("owner should not be sampled"))
        self.assertEqual(state["tasks"][self.task_id]["status"], "QUEUED")
        self.assertIn("SCIENCE_PROJECTION_UNTRUSTED", state["tasks"][self.task_id]["wait_reason"])
        self.assertEqual(state["tick_decision"]["code"], "SCIENCE_PROJECTION_UNTRUSTED")
        self.assertFalse((self.root / (self.task_id + "-dispatch.json")).exists())
        spawn.assert_not_called()

    def test_no_registered_queue_is_not_global_no_science_authority(self):
        self.task["mode"] = "hold"
        self.task["reason"] = "native recovery review required"
        self.write_manifest()
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [],
                               owner_query=lambda _: self.fail("owner sampled for no queue"),
                               candidate_query=lambda _: self.fail("projection queried for no queue"))
        verdict = result["tick_decision"]
        self.assertEqual(verdict["code"], "NO_REGISTERED_QUEUED_TASKS")
        self.assertEqual(verdict["task_counts"]["CONTINUATION_REQUIRED"], 1)
        self.assertEqual(verdict["global_ready_supply"], "UNVERIFIED_CROSS_CAMPAIGN")
        spawn.assert_not_called()

    def test_queued_but_not_science_projected_is_explicit(self):
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=self.owner,
                               candidate_query=lambda _: set())
        self.assertEqual(result["tick_decision"]["code"], "NO_PROJECTED_LAUNCH_CANDIDATES")
        self.assertEqual(result["tick_decision"]["task_counts"]["QUEUED"], 1)
        spawn.assert_not_called()

    def test_revoked_gpu_never_claims_even_with_ready_science(self):
        with patch.object(pool.subprocess, 'Popen') as spawn:
            state = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                              live_query=lambda: [],
                              owner_query=lambda _: self.fail('owner should not be sampled'),
                              excluded_gpu_uuids=[self.task['gpu_uuid']])
        self.assertEqual(state['tasks'][self.task_id]['status'], 'QUEUED')
        self.assertEqual(state['tasks'][self.task_id]['wait_reason'],
                         'GPU_EXCLUDED_BY_CURRENT_OWNER_INSTRUCTION')
        self.assertFalse((self.root / (self.task_id + '-dispatch.json')).exists())
        spawn.assert_not_called()

    def test_missing_flag_does_not_override_live_gpu1_withdrawal(self):
        # Regression for 2026-10-28 17:09 UTC accidental restart.
        self.task['gpu_uuid'] = pool.GPU_UUIDS[0]
        self.write_manifest()
        with patch.object(pool.subprocess, 'Popen') as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [],
                               owner_query=lambda _: self.fail('GPU1 owner query must not run'))
        self.assertEqual(result['tasks'][self.task_id]['status'], 'QUEUED')
        self.assertEqual(result['tasks'][self.task_id]['wait_reason'],
                         'GPU_EXCLUDED_BY_CURRENT_OWNER_INSTRUCTION')
        self.assertFalse((self.root / (self.task_id + '-dispatch.json')).exists())
        spawn.assert_not_called()

    def test_reviewed_gpu1_route_keeps_exclusion_flag_and_native_gates(self):
        self.task['gpu_uuid'] = pool.GPU_UUIDS[0]
        self.write_manifest()
        with patch.object(pool, 'scoped_gpu1_reauthorization', return_value=True), patch.object(pool, 'native_science_admission') as science, patch.object(pool.subprocess, 'Popen', return_value=SimpleNamespace(pid=12345)) as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=self.owner,
                               candidate_query=lambda _: {self.task_id},
                               excluded_gpu_uuids=[pool.GPU_UUIDS[0]])
        science.assert_called_once_with(self.task)
        spawn.assert_called_once()
        self.assertEqual(result['tasks'][self.task_id]['status'], 'RUNNING')

    def test_gpu1_expiry_after_claim_never_spawns(self):
        self.task['gpu_uuid'] = pool.GPU_UUIDS[0]
        self.write_manifest()
        with patch.object(pool, 'scoped_gpu1_reauthorization', side_effect=[True, False]), patch.object(pool, 'native_science_admission'), patch.object(pool.subprocess, 'Popen') as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=self.owner,
                               candidate_query=lambda _: {self.task_id},
                               excluded_gpu_uuids=[pool.GPU_UUIDS[0]])
        spawn.assert_not_called()
        self.assertEqual(result['tasks'][self.task_id]['status'], 'DIAGNOSTIC_REQUIRED')
        self.assertTrue((self.root / (self.task_id + '-dispatch.json')).exists())

    def test_missing_mandatory_gpu1_flag_rejects_startup_before_lock(self):
        with patch('sys.argv', ['pool', '--manifest', str(self.manifest),
                                '--root', str(self.root), '--once']):
            with self.assertRaises(SystemExit) as caught:
                pool.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse((self.root / 'dispatcher.lock').exists())

    def test_invalid_gpu_exclusion_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'GPU_EXCLUSION_INVALID'):
            pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                      live_query=lambda: [], excluded_gpu_uuids=['GPU-invalid'])

    def test_peer_owner_denies_before_any_claim_or_spawn(self):
        def denied(uuid):
            raise ValueError("GPU_OWNER_NOT_EXCLUSIVELY_IDLE")
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=denied)
        self.assertEqual(result["tasks"][self.task_id]["status"], "QUEUED")
        self.assertIn("RESOURCE_NOT_ADMITTED", result["tasks"][self.task_id]["wait_reason"])
        self.assertEqual(result["tick_decision"]["code"], "CANDIDATE_ADMISSION_BLOCKED")
        self.assertFalse((self.root / (self.task_id + "-dispatch.json")).exists())
        spawn.assert_not_called()

    def test_owner_change_after_claim_quarantines_never_paid_task(self):
        calls = 0
        def change(uuid):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("owner now peer")
            return self.owner(uuid)
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=change)
        self.assertEqual(result["tasks"][self.task_id]["status"], "DIAGNOSTIC_REQUIRED")
        self.assertTrue((self.root / (self.task_id + "-dispatch.json")).exists())
        self.assertFalse((self.root / (self.task_id + "-owner-admission.json")).exists())
        spawn.assert_not_called()

    def test_disk_gate_blocks_without_sampling_owner(self):
        self.write_manifest(min_free_bytes=10**30)
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=lambda _: self.fail("owner queried"))
        self.assertEqual(result["tasks"][self.task_id]["wait_reason"], "shared disk admission")
        spawn.assert_not_called()

    def test_already_attempted_or_output_present_never_spawns(self):
        self.output.mkdir()
        (self.output / "partial.json").write_text("{}")
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=self.owner,
                               candidate_query=lambda _: {self.task_id})
        self.assertEqual(result["tasks"][self.task_id]["status"], "DIAGNOSTIC_REQUIRED")
        self.assertIn("nonempty scientific output", result["tasks"][self.task_id]["reason"])
        spawn.assert_not_called()
        self.assertFalse((self.root / (self.task_id + "-dispatch.json")).exists())

    def test_claimed_task_cannot_replay_even_if_prior_state_is_queued(self):
        (self.root / (self.task_id + "-dispatch.json")).write_text('{"previous": true}')
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=self.owner,
                               candidate_query=lambda _: {self.task_id})
        self.assertEqual(result["tasks"][self.task_id]["status"], "DIAGNOSTIC_REQUIRED")
        self.assertTrue(result["tasks"][self.task_id]["attempted"])
        spawn.assert_not_called()

    def test_unadapted_non_q04_task_is_never_admitted(self):
        self.task["id"] = self.task_id = "engineering-not-an-authorized-science-cell"
        self.write_manifest()
        with patch.object(pool.subprocess, "Popen") as spawn:
            result = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=self.owner,
                               candidate_query=lambda _: {self.task_id})
        self.assertEqual(result["tasks"][self.task_id]["status"], "QUEUED")
        self.assertIn("NATIVE_RELEASE_ADAPTER_MISSING", result["tasks"][self.task_id]["wait_reason"])
        spawn.assert_not_called()


    def add_task(self, task_id, gpu_uuid, *, dependencies=(), mode="queued"):
        task = {**self.task, "id": task_id, "gpu_uuid": gpu_uuid, "mode": mode,
                "worker_tokens": [str(self.source), task_id],
                "output_root": str(self.root / (task_id + "-output")),
                "completion": {"path": str(self.root / (task_id + "-seal.json")),
                               "equals": {"status": "PASS"}},
                "depends_on": list(dependencies)}
        self.tasks.append(task)
        return task

    def test_two_idle_cards_and_late_owner_release_backfill(self):
        second_id = "q04-cpu-second-cell"
        self.add_task(second_id, pool.GPU_UUIDS[1])
        self.write_manifest()
        spawned = []
        def fake_spawn(*args, **kwargs):
            spawned.append(kwargs["env"]["OPENEVO_AUTHORIZED_GPU_UUIDS"])
            return type("Child", (), {"pid": 1000 + len(spawned)})()
        def first_owner(uuid):
            if uuid == pool.GPU_UUIDS[1]:
                raise ValueError("peer held this card")
            return self.owner(uuid)
        with patch.object(pool.subprocess, "Popen", side_effect=fake_spawn):
            first = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                              live_query=lambda: [], owner_query=first_owner)
            second = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=self.owner)
        self.assertEqual(spawned, [pool.GPU_UUIDS[3], pool.GPU_UUIDS[1]])
        self.assertEqual(first["tasks"][second_id]["status"], "QUEUED")
        self.assertEqual(second["tasks"][second_id]["status"], "RUNNING")
        self.assertEqual(second["tasks"][self.task_id]["status"], "DIAGNOSTIC_REQUIRED")

    def test_sealed_predecessor_unblocks_successor_without_replay(self):
        predecessor = self.task
        predecessor["mode"] = "adopt"
        predecessor["argv"] = []
        predecessor["source_files"] = {}
        next_id = "q04-cpu-third-after-seal"
        successor = self.add_task(next_id, pool.GPU_UUIDS[3], dependencies=(self.task_id,))
        successor['argv'] = ['python3', str(self.source)]
        successor['source_files'] = {str(self.source): sha(self.source.read_bytes())}
        self.write_manifest()
        Path(predecessor["completion"]["path"]).write_text(json.dumps({"status": "PASS"}))
        with patch.object(pool.subprocess, "Popen",
                          return_value=type("Child", (), {"pid": 1444})()) as spawn:
            first = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                              live_query=lambda: [], owner_query=self.owner)
            second = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                               live_query=lambda: [], owner_query=self.owner)
        self.assertEqual(first["tasks"][self.task_id]["status"], "COMPLETE")
        self.assertEqual(first["tasks"][next_id]["status"], "QUEUED")
        self.assertEqual(second["tasks"][next_id]["status"], "RUNNING")
        self.assertEqual(spawn.call_count, 1)


    def test_frozen_source_changed_after_claim_does_not_spawn(self):
        seen = 0
        def owner_and_mutate(uuid):
            nonlocal seen
            seen += 1
            if seen == 2:
                self.source.write_text("# changed after approval\n")
            return self.owner(uuid)
        with patch.object(pool.subprocess, "Popen") as spawn:
            state = pool.tick(self.manifest, self.root, gpu_query=self.gpu,
                              live_query=lambda: [], owner_query=owner_and_mutate)
        self.assertEqual(state["tasks"][self.task_id]["status"], "DIAGNOSTIC_REQUIRED")
        self.assertIn("SOURCE_CHANGED_AFTER", state["tasks"][self.task_id]["reason"])
        self.assertTrue((self.root / (self.task_id + "-dispatch.json")).exists())
        spawn.assert_not_called()

    def test_wrong_manifest_path_cannot_borrow_projection_candidate(self):
        foreign = self.root / "foreign.json"
        foreign.write_bytes(self.manifest.read_bytes())
        with self.assertRaisesRegex(ValueError, "MANIFEST_PATH_MISMATCH"):
            pool.tick(foreign, self.root, gpu_query=self.gpu,
                      live_query=lambda: [], owner_query=self.owner)


class ScopedGpu1ReauthorizationTests(unittest.TestCase):
    def setUp(self):
        base = '/workspace'
        self.runner = base + '/maintenance/prepared_q09_runner_202610281815.py'
        self.proof = base + '/maintenance/Q09-S2-R143-SAME-SOURCE-RESUME-20261028/reviewed-partial-resume.json'
        self.task = {
            'id': pool.GPU1_REAUTHORIZED_CLAIM, 'gpu_uuid': pool.GPU_UUIDS[0],
            'argv': ['python3', self.runner, '--root', base + '/maintenance/Q04-POOL-20261022', '--task-id', pool.GPU1_REAUTHORIZED_CLAIM],
            'output_root': base + '/maintenance/Q09-S2-R143-SAME-SOURCE-POOL-RUN-20261028',
            'source_files': {self.runner: pool.GPU1_REAUTHORIZED_RUNNER_SHA256, self.proof: 'b'*64},
            'native_campaign': {'science_root': base + '/runs/q09-active-s2-r120-gpu2-formal-20261020',
                'dispatch_dir': base + '/control/q09-s2-r143-reviewed-dispatch-20261028',
                'source_sha': 'f952c09bbdefc0aa6cef570003702bd09bec87cc',
                'authority_sha': '33cbc3ac082c0764094c1ca68012e87c6bb29161',
                'next_round': 143, 'rollouts_consumed': 18304,
                'reservation': {'enabled': False, 'expires_at': None}, 'reviewed_partial_resume': self.proof}}
        self.now = pool.GPU1_REAUTHORIZATION_START + 1
        self.hashes = dict(self.task['source_files'])

    def allowed(self):
        with patch.object(pool.time, 'time', return_value=self.now), patch.object(pool, 'digest', side_effect=lambda p: self.hashes[str(p)]), patch.object(Path, 'is_symlink', return_value=False):
            return pool.scoped_gpu1_reauthorization(self.task)

    def test_exact_claim_withdrawal_remains_default(self):
        self.assertTrue(self.allowed())
        self.assertIn(pool.GPU_UUIDS[0], pool.CURRENT_OWNER_WITHDRAWN_UUIDS)

    def test_time_window_is_bounded(self):
        for now in [pool.GPU1_REAUTHORIZATION_START-1, pool.GPU1_REAUTHORIZATION_DEADLINE, pool.GPU1_REAUTHORIZATION_DEADLINE+1]:
            self.now = now
            self.assertFalse(self.allowed())

    def test_other_task_gpu_or_argv_refused(self):
        for key, wrong in [('id', 'q09-other-task'), ('gpu_uuid', pool.GPU_UUIDS[3]), ('argv', ['python3', self.runner])]:
            with self.subTest(key=key):
                original = self.task[key];self.task[key] = wrong
                self.assertFalse(self.allowed());self.task[key] = original

    def test_holder_and_scientific_drift_refused(self):
        cfg=self.task['native_campaign']
        for key, wrong in [('reservation', {'enabled': True, 'expires_at': None}), ('source_sha', '0'*40), ('authority_sha', '0'*40), ('next_round', 144), ('rollouts_consumed', 18432), ('science_root', '/another/run'), ('dispatch_dir', '/another/control')]:
            with self.subTest(key=key):
                original=cfg[key];cfg[key]=wrong
                self.assertFalse(self.allowed());cfg[key]=original

    def test_runner_or_proof_hash_drift_refused(self):
        for path in [self.runner, self.proof]:
            original=self.hashes[path];self.hashes[path]='c'*64
            self.assertFalse(self.allowed());self.hashes[path]=original
        self.task['source_files'].pop(self.proof)
        self.assertFalse(self.allowed())

    def use_preserved101(self):
        base='/workspace'
        self.runner=base+'/maintenance/prepared_q09_runner_202610281958.py'
        self.proof=base+'/maintenance/Q09-S2-R143-PRESERVED101-202610281958/reviewed-partial-resume.json'
        self.task.update(id=pool.GPU1_PRESERVED101_CLAIM,argv=['python3',self.runner,'--root',base+'/maintenance/Q04-POOL-20261022','--task-id',pool.GPU1_PRESERVED101_CLAIM],output_root=base+'/maintenance/Q09-S2-R143-PRESERVED101-POOL-RUN-202610281958')
        self.task['native_campaign']['reviewed_partial_resume']=self.proof
        self.task['source_files']={self.runner:pool.GPU1_PRESERVED101_RUNNER_SHA256,self.proof:'b'*64};self.hashes=dict(self.task['source_files'])

    def test_preserved101_exact_scope_and_unfrozen_runner_refused(self):
        self.use_preserved101()
        with patch.object(pool,'GPU1_PRESERVED101_RUNNER_SHA256',None):
            self.assertFalse(self.allowed())
        self.assertTrue(self.allowed())
        with patch.object(pool,'GPU1_PRESERVED101_RUNNER_SHA256',self.hashes[self.runner]):
            self.assertTrue(self.allowed())
            self.now=pool.GPU1_REAUTHORIZATION_DEADLINE
            self.assertFalse(self.allowed())

    def test_preserved101_cannot_bind_old_claim_runner_proof_or_other_science(self):
        self.use_preserved101()
        with patch.object(pool,'GPU1_PRESERVED101_RUNNER_SHA256',self.hashes[self.runner]):
            for key,value in [('id',pool.GPU1_REAUTHORIZED_CLAIM),('output_root','/other/output'),('argv',['python3','/other/runner'])]:
                old=self.task[key];self.task[key]=value;self.assertFalse(self.allowed());self.task[key]=old
            cfg=self.task['native_campaign']
            for key,value in [('reviewed_partial_resume','/old/proof'),('source_sha','0'*40),('reservation',{'enabled':True,'expires_at':None})]:
                old=cfg[key];cfg[key]=value;self.assertFalse(self.allowed());cfg[key]=old
            self.hashes[self.proof]='c'*64;self.assertFalse(self.allowed())

    def test_missing_scope_refused(self):
        self.task.pop('native_campaign');self.assertFalse(self.allowed())

    def test_symbolic_link_refused(self):
        with patch.object(pool.time, 'time', return_value=self.now), patch.object(Path, 'is_symlink', return_value=True):
            self.assertFalse(pool.scoped_gpu1_reauthorization(self.task))


class FirstPartyGpu1ClaimTests(unittest.TestCase):
    """Actual-human-event binding cannot be replaced by a PR or an agent claim."""
    allowed = ScopedGpu1ReauthorizationTests.allowed

    def setUp(self):
        ScopedGpu1ReauthorizationTests.setUp(self)
        base='/workspace'
        self.runner=base+'/maintenance/prepared_q09_runner_202610282132.py'
        self.proof=base+'/maintenance/Q09-S2-R143-FIRSTPARTY-GRANT-202610282118/reviewed-partial-resume.json'
        self.event_path=base+'/maintenance/q09-firstparty-event-verification-202610282118.json'
        self.task.update(id=pool.GPU1_FIRSTPARTY_CLAIM,argv=['python3',self.runner,'--root',base+'/maintenance/Q04-POOL-20261022','--task-id',pool.GPU1_FIRSTPARTY_CLAIM],output_root=base+'/maintenance/Q09-S2-R143-FIRSTPARTY-GRANT-POOL-RUN-202610282118')
        self.task['native_campaign']['reviewed_partial_resume']=self.proof
        self.task['source_files']={self.runner:pool.GPU1_FIRSTPARTY_RUNNER_SHA256,self.proof:pool.GPU1_FIRSTPARTY_PROOF_SHA256,self.event_path:pool.GPU1_FIRSTPARTY_EVENT_SHA256}
        self.hashes=dict(self.task['source_files'])
        self.event=dict(source='original_native_codex_session_event_direct_mac_read',event_type='response_item',payload_type='message',role='user',timestamp='2026-10-28T16:39:58.750Z',line=16,raw_line_sha256='8e4604051b8829e2fae4fb1950a38c6275fcc5ace2fa108a579066d1a9701a7a',verbatim_grant='你可以积极利用 GPU1 和 4',exact_grant_present=True,raw_user_text_exported=False)
        mocker=patch.object(pool,'read',side_effect=lambda p:self.event if str(p)==self.event_path else {})
        mocker.start();self.addCleanup(mocker.stop)

    def test_exact_event_and_partial_proof_are_admitted(self):
        self.assertTrue(self.allowed())
        self.assertIn(pool.GPU_UUIDS[0],pool.CURRENT_OWNER_WITHDRAWN_UUIDS)

    def test_agent_or_tool_message_cannot_be_human_authorization(self):
        for key,value in [('role','assistant'),('event_type','event_msg'),('payload_type','tool_result')]:
            old=self.event[key];self.event[key]=value
            with self.subTest(key=key):self.assertFalse(self.allowed())
            self.event[key]=old

    def test_wrong_chronology_or_native_line_refused(self):
        for key,value in [('timestamp','2026-10-28T15:00:00Z'),('line',15),('raw_line_sha256','0'*64)]:
            old=self.event[key];self.event[key]=value
            with self.subTest(key=key):self.assertFalse(self.allowed())
            self.event[key]=old

    def test_missing_grant_or_private_text_export_refused(self):
        for key,value in [('verbatim_grant',''),('exact_grant_present',False),('raw_user_text_exported',True)]:
            old=self.event[key];self.event[key]=value
            with self.subTest(key=key):self.assertFalse(self.allowed())
            self.event[key]=old

    def test_repinning_a_different_event_or_partial_proof_is_not_authority(self):
        for path in [self.event_path,self.proof]:
            old=self.hashes[path];self.hashes[path]='d'*64;self.task['source_files'][path]='d'*64
            with self.subTest(path=path):self.assertFalse(self.allowed())
            self.hashes[path]=old;self.task['source_files'][path]=old

    def test_other_claim_and_science_cannot_borrow_the_grant(self):
        self.task['id']+='-copy';self.assertFalse(self.allowed())
        self.task['id']=pool.GPU1_FIRSTPARTY_CLAIM
        self.task['native_campaign']['next_round']=144;self.assertFalse(self.allowed())

    def test_lease_expiry_still_denies_this_real_grant(self):
        self.now=pool.GPU1_REAUTHORIZATION_DEADLINE;self.assertFalse(self.allowed())


if __name__ == "__main__":
    unittest.main()
