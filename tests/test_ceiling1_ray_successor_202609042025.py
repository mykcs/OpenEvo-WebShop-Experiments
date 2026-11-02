import ast
import hashlib
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TS = "202609042025"
CFG = ROOT / f"configs/experiment/openevo-ceiling1-ray-placement-backend-{TS}.json"
AUTH = ROOT / f"configs/experiment/openevo-ceiling1-stage2-runtime-authority-{TS}.json"
BACKEND = ROOT / f"scripts/openevo_webshop/ceiling1_ray_placement_backend_{TS}.py"
RUNNER = ROOT / f"scripts/openevo_webshop/run_ceiling1_stage2_ray_{TS}.py"
ARM = ROOT / f"scripts/openevo_webshop/run_ceiling1_stage2_arm_{TS}.py"
POOL = ROOT / f"scripts/openevo_webshop/ceiling1_ray_gpu_pool_{TS}.py"
MIG = ROOT / f"scripts/openevo_webshop/prepare_ceiling1_ray_continuation_{TS}.py"

spec = importlib.util.spec_from_file_location("ray_backend_timestamp", BACKEND)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class FakeGroup:
    def ready(self):
        return "ready-ref"


class FakeRay:
    def shutdown(self):
        pass
    def wait(self, refs, timeout=None):
        return (refs, [])


class RayTimestampSuccessorTest(unittest.TestCase):
    def test_timestamp_identity_and_authority(self):
        cfg = json.loads(CFG.read_text())
        auth = json.loads(AUTH.read_text())
        self.assertEqual(cfg["schema"], f"openevo.ceiling1-ray-placement-backend.{TS}")
        self.assertEqual(cfg["identity_timestamp"], TS)
        self.assertEqual(auth["schema"], f"openevo.ceiling1-stage2-runtime-authority.{TS}")
        self.assertEqual(auth["status"], "READY_FOR_AUDITED_NATURAL_MIGRATION")
        self.assertTrue(auth["migration"]["zero_live_both_arms_required"])
        self.assertEqual(auth["platform_boundary"]["path_contract_status"], "DEFERRED_UNTIL_AUDITED_PUBLIC_LAUNCHER_MIGRATION")
        self.assertFalse(auth["platform_boundary"]["public_launcher_activation_authority"])
        self.assertFalse(auth["platform_boundary"]["current_formal_arms_use_public_launcher"])

    def test_every_cross_arm_profile_pair_fits(self):
        cfg = json.loads(CFG.read_text())
        budget = int(cfg["gpu_project_budget_mib"])
        left = {k: v for k, v in cfg["profiles"].items() if v["arm"] == "3b"}
        right = {k: v for k, v in cfg["profiles"].items() if v["arm"] == "qwen3-1p7b"}
        bad = []
        for lk, lv in left.items():
            for rk, rv in right.items():
                total = int(lv["vram_request_mib"]) + int(rv["vram_request_mib"])
                if total > budget:
                    bad.append((lk, rk, total, budget))
        self.assertEqual(bad, [])
        self.assertGreaterEqual(int(cfg["process_slots_per_gpu"]), 2)
        self.assertEqual(int(cfg["same_arm_slots_per_gpu"]), 1)

    def test_gpu_identity_is_index_and_uuid_pinned(self):
        cfg = json.loads(CFG.read_text())
        self.assertEqual(
            cfg["managed_gpu_uuids"],
            {
                "4": "GPU-00000000-0000-0000-0000-000000000000",
                "5": "GPU-00000000-0000-0000-0000-000000000000",
                "6": "GPU-00000000-0000-0000-0000-000000000000",
                "7": "GPU-00000000-0000-0000-0000-000000000000",
            },
        )
        old = backend.gpu_rows
        try:
            backend.gpu_rows = lambda: {4: {"uuid": "bad", "used_mib": 0, "free_mib": 100, "util_pct": 0}}
            with self.assertRaisesRegex(RuntimeError, "UUID identity drift"):
                backend.live_gate({"managed_gpus": [4], "managed_gpu_uuids": {"4": "good"}, "global_unallocated_headroom_mib": 0}, {"vram_request_mib": 1}, [4])
        finally:
            backend.gpu_rows = old

    def test_physical_prefilter_skips_busy_card_before_ray_placement(self):
        cfg={"managed_gpus":[4,5,6],"managed_gpu_uuids":{"4":"u4","5":"u5","6":"u6"},"global_unallocated_headroom_mib":4096}
        profile={"vram_request_mib":9000}
        rows={
            4:{"uuid":"u4","free_mib":8000,"used_mib":24000,"util_pct":99},
            5:{"uuid":"u5","free_mib":25000,"used_mib":7000,"util_pct":30},
            6:{"uuid":"u6","free_mib":20000,"used_mib":12000,"util_pct":50},
        }
        eligible,observed,need=backend.physically_eligible_gpus(cfg,profile,rows)
        self.assertEqual(need,13096)
        self.assertEqual(eligible,[5,6])
        self.assertNotIn(4,eligible)
        auth=json.loads(AUTH.read_text())
        self.assertTrue(auth["control_plane_guarantees"]["physical_vram_prefilter_prevents_false_no_placement_on_partially_busy_pool"])
        self.assertIn("itertools.combinations",BACKEND.read_text())

    def test_single_writer_lock(self):
        with tempfile.TemporaryDirectory() as td:
            first = backend.acquire_controller_lock(Path(td), "3b", "a" * 40)
            try:
                with self.assertRaisesRegex(RuntimeError, "CONTROLLER_SINGLE_WRITER_HELD"):
                    backend.acquire_controller_lock(Path(td), "3b", "b" * 40)
            finally:
                first.close()
            second = backend.acquire_controller_lock(Path(td), "3b", "b" * 40)
            second.close()

    def test_failed_launch_cleanup_is_truthful(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            cp = t / "cfg.json"
            cfg = {"managed_gpus": [4], "profiles": {"w": {"arm": "3b", "vram_request_mib": 1}}, "global_unallocated_headroom_mib": 0}
            cp.write_text(json.dumps(cfg))
            cfgsha = backend.sha(cp)
            backend.dump(t / "state.json", {"schema": f"openevo.ceiling1-ray-placement-state.{TS}", "config_sha256": cfgsha, "gpus": {"4": {"leases": []}}, "reservations": {"rid": {"placement_group_name": "pg", "workload": "w", "gpus": [4]}}, "admission": {"paused": False}})
            sel = t / "sel.json"
            sel.write_text(json.dumps({"schema": f"openevo.ceiling1-ray-selection.{TS}", "workload": "w", "config_sha256": cfgsha, "selected_gpus": [4], "reservation_id": "rid", "placement_group_name": "pg"}))
            launch = t / "fail.sh"
            launch.write_text("#!/bin/sh\nexit 7\n")
            launch.chmod(0o755)
            removed = []
            old = (backend.ray_api, backend.live_gate, backend.running)
            backend.ray_api = lambda *_: (FakeRay(), {}, None, lambda name: name, lambda pg: removed.append(pg), None)
            backend.live_gate = lambda *_: {}
            backend.running = lambda _: False
            try:
                args = SimpleNamespace(config=cp, state_root=t, workload="w", selection_receipt=sel, gpus=[4], lease_id="L", container="C", launch_script=launch)
                with self.assertRaisesRegex(RuntimeError, "science launch script failed"):
                    backend.cmd_run(args, cfg)
            finally:
                backend.ray_api, backend.live_gate, backend.running = old
            self.assertEqual(backend.load(t / "state.json")["reservations"], {})
            self.assertEqual(removed, ["pg"])

    def test_immutable_selection_reconstructs_exact_gpu_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            cp = t / "cfg.json"
            cfg = {"managed_gpus": [4, 5], "reservation_ttl_seconds": 180, "rollout_strategy": "STRICT_SPREAD", "single_gpu_strategy": "PACK", "placement_ready_timeout_seconds": 2.0}
            cp.write_text(json.dumps(cfg))
            cfgsha = backend.sha(cp)
            backend.dump(t / "state.json", {"schema": f"openevo.ceiling1-ray-placement-state.{TS}", "config_sha256": cfgsha, "gpus": {"4": {"leases": []}, "5": {"leases": []}}, "reservations": {}, "admission": {"paused": False}})
            sel = t / "selection.json"
            sel.write_text(json.dumps({"schema": f"openevo.ceiling1-ray-selection.{TS}", "workload": "roll", "selected_gpus": [4, 5], "reservation_id": "rid", "placement_group_name": "pg"}))
            captured = {}
            def placement_group(bundles, strategy, name, lifetime):
                captured.update({"bundles": bundles, "strategy": strategy, "name": name, "lifetime": lifetime})
                return FakeGroup()
            old = backend.live_gate
            backend.live_gate = lambda *_: {}
            try:
                args = SimpleNamespace(config=cp, state_root=t, workload="roll", selection_receipt=sel)
                rec = backend.ensure_selection_reservation(args, cfg, {"arm": "3b", "vram_request_mib": 1}, json.loads(sel.read_text()), FakeRay(), placement_group, lambda _: (_ for _ in ()).throw(RuntimeError("absent")), lambda _: None)
            finally:
                backend.live_gate = old
            self.assertTrue(rec["reconstructed_from_immutable_selection"])
            self.assertEqual(rec["gpus"], [4, 5])
            self.assertEqual(captured["bundles"][0]["gpu_index_4"], 1.0)
            self.assertEqual(captured["bundles"][1]["gpu_index_5"], 1.0)
            self.assertEqual(backend.load(t / "state.json")["reservations"]["rid"]["gpus"], [4, 5])

    def test_admission_pause_keeps_existing_lease(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            cp = t / "cfg.json"
            cfg = {"managed_gpus": [4]}
            cp.write_text(json.dumps(cfg))
            cfgsha = backend.sha(cp)
            lease = {"lease_id": "live", "container": "live-c", "workload": "w"}
            backend.dump(t / "state.json", {"schema": f"openevo.ceiling1-ray-placement-state.{TS}", "config_sha256": cfgsha, "gpus": {"4": {"leases": [lease]}}, "reservations": {"rid": {"placement_group_name": "pg", "workload": "w", "gpus": [4], "launching": False}}, "admission": {"paused": False}})
            removed = []
            old = backend.ray_api
            backend.ray_api = lambda *_: (FakeRay(), {}, None, lambda name: name, lambda pg: removed.append(pg), None)
            try:
                backend.cmd_pause(SimpleNamespace(config=cp, state_root=t, reason="switch experiment"), cfg)
            finally:
                backend.ray_api = old
            state = backend.load(t / "state.json")
            self.assertTrue(state["admission"]["paused"])
            self.assertEqual(state["gpus"]["4"]["leases"], [lease])
            self.assertEqual(state["reservations"], {})

    def test_candidate_probe_has_dedicated_arm_identity(self):
        text = ARM.read_text()
        self.assertIn("stage2_candidate_probe_{self.arm}", text)
        self.assertIn("ceiling1-vnext-{self.arm}-r{r:03d}-probe", text)  # legacy-revision-compat
        self.assertIn("openevo-vnext-{self.arm}-r{r:03d}-probe", text)  # legacy-revision-compat

    def test_fresh_recursive_local_import_closure(self):
        pending = [RUNNER]
        seen = set()
        missing = []
        prefix = "scripts.openevo_webshop."
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            tree = ast.parse(path.read_text())
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix):
                    modules.append(node.module)
                elif isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names if alias.name.startswith(prefix))
            for mod in modules:
                target = ROOT / Path(*mod.split(".")).with_suffix(".py")
                if not target.is_file():
                    missing.append((path.relative_to(ROOT).as_posix(), mod))
                elif target not in seen:
                    pending.append(target)
        self.assertEqual(missing, [])

    def test_candidate_state_logic_hash_is_live_qualified_copy(self):
        state = ROOT / f"scripts/openevo_webshop/ceiling1_stage2_state_{TS}.py"
        self.assertEqual(hashlib.sha256(state.read_bytes()).hexdigest(), "9081a2608007abad4724a39fd072fe469c3cba1993cc92996d76b04f42dc367a")

    def test_whole_pool_migration_is_zero_live(self):
        text = MIG.read_text()
        self.assertIn("whole-pool-zero-live-timestamp-successor", text)
        self.assertIn("requires_zero_live_both_arms", text)
        self.assertIn("candidate_state_logic_renamed", text)
        self.assertIn(f"openevo.ceiling1-stage2-resource-migration.ray-continuation.{TS}", text)

    def test_pool_restart_and_placement_timeout_are_fail_closed(self):
        cfg = json.loads(CFG.read_text())
        self.assertEqual(float(cfg["placement_ready_timeout_seconds"]), 2.0)
        backend_text = BACKEND.read_text()
        pool_text = POOL.read_text()
        self.assertIn("placement_ready_timeout_seconds", backend_text)
        self.assertIn("Ray pool restart forbidden while recorded live containers exist", pool_text)
        auth = json.loads(AUTH.read_text())
        self.assertTrue(auth["control_plane_guarantees"]["immutable_rollout_selection_can_reconstruct_exact_gpu_reservation"])
        self.assertTrue(auth["control_plane_guarantees"]["pool_restart_forbidden_while_recorded_live_containers_exist"])

    def test_public_launcher_delegates_platform(self):
        launch = (ROOT / f"scripts/openevo_webshop/launch_ceiling1_stage2_current_arm_{TS}.sh").read_text()
        self.assertIn("scripts/oe", launch)
        self.assertIn(" host ", launch)
        self.assertNotIn("docker run", launch)


if __name__ == "__main__":
    unittest.main()
