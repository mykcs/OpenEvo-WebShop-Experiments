import contextlib
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.experiment_startup_report import (
    CAMPAIGN_ID_RE,
    GPU_ATTRIBUTION_WARNING,
    load_campaign,
    main as startup_report_main,
    render_report,
    validate_campaign,
)
from scripts.openevo_webshop import gpu_resource_gate


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = REPO_ROOT / "configs" / "experiment" / "current-campaign.json"


class ExperimentGovernanceTest(unittest.TestCase):
    def test_current_campaign_is_valid(self):
        campaign = load_campaign(CAMPAIGN_PATH)
        self.assertEqual(validate_campaign(campaign), [])

    def test_campaign_group_and_name_are_stable(self):
        campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
        self.assertRegex(campaign["campaign_id"], CAMPAIGN_ID_RE)
        self.assertEqual(campaign["tracking"]["group"], campaign["campaign_id"])
        self.assertEqual(campaign["standard_core"], "v1")
        self.assertEqual(campaign["standard_profile"], "vNext")

    def test_preflight_cannot_consume_formal_tasks(self):
        campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
        if campaign["status"] == "preflight":
            self.assertFalse(campaign["formal_task_consumption_allowed"])

    def test_tracking_destination_is_explicit(self):
        campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
        tracking = campaign["tracking"]
        self.assertEqual(tracking["entity"], "example-entity")
        self.assertEqual(tracking["project"], "openevo-webshop-experiments")
        self.assertIn(tracking["default_mode"], {"online", "offline", "disabled"})

    def test_startup_report_surfaces_gpu_namespace_boundary(self):
        campaign = load_campaign(CAMPAIGN_PATH)
        runtime = {
            "repo": {"git_sha": "test", "branch": "test", "dirty": False},
            "identity": {"user": "root", "uid": 0, "gid": 0, "hostname": "experiment-host"},
            "container": {
                "containerized_hint": True,
                "docker_cli": "available",
                "docker_server": "available",
                "image_identity": "sha256:test",
            },
            "python": {"version": "test", "executable": "python", "torch": "test", "transformers": "test", "peft": "test", "wandb": "test"},
            "gpu": {"snapshot": "GPU-test", "cuda_visible_devices": "GPU-test"},
            "tracking": {"entity": "test", "project": "test", "mode": "offline"},
        }
        self.assertIn(GPU_ATTRIBUTION_WARNING, render_report(campaign, runtime))

    def test_startup_report_explicit_scope_uses_its_router_release_profile_and_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c",
                            "user.email=fixture@example.invalid", "commit", "-q", "--allow-empty",
                            "-m", "fixture"], check=True)
            expected_sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            campaign_id = "20261002-1200-independent-science"
            binding = {"experiment_id": "independent-science", "passport_sha256": "1" * 64}
            profile = {"workspace": str(root), "source": {"experiment": {"path": ".", "revision": expected_sha}},
                       "execution": {"mode": "host-orchestrator"}, "image": "science:fixture",
                       "image_id": "sha256:" + "2" * 64}
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            release = {"schema": "openevo.experiment-fixture.v1", "campaign_id": campaign_id,
                       "status": "launch-authorized", "formal_task_consumption_allowed": True,
                       "experiment_identity": binding,
                       "preregistration": {"path": "prereg.json", "sha256": "3" * 64},
                       "path_contract": {"path": "paths.json", "sha256": "4" * 64},
                       "execution_readiness": {"profile": {"path": "profile.json",
                           "sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest()}}}
            release_path = root / "release.json"
            release_path.write_text(json.dumps(release), encoding="utf-8")
            router = {"standard_core": "v1", "standard_profile": "vNext",
                      "campaign_id": campaign_id, "status": "launch-authorized",
                      "title": "Explicit fixture campaign", "objective": "Exercise explicit startup projection.",
                      "scientific_question": "Can the selected frozen profile be projected without using platform defaults?",
                      "benchmark": "fixture", "phase": "FIXTURE_VALIDATION",
                      "protocol_status": "launch-authorized",
                      "formal_task_consumption_allowed": True,
                      "next_gate": "fixture complete",
                      "tracking": {"entity": "fixture", "group": campaign_id,
                                   "project": "fixture", "default_mode": "disabled"},
                      "release_receipt": "release.json",
                      "release_receipt_sha256": hashlib.sha256(release_path.read_bytes()).hexdigest()}
            (root / "router.json").write_text(json.dumps(router), encoding="utf-8")
            checkout = {"current_path": str(root), "is_canonical_control": False,
                        "canonical_control_checkout": "/global-control",
                        "next_action": "NON_CANONICAL_EXECUTION_CHECKOUT", "next_command": "cd /global-control && oe doctor"}
            with patch("scripts.experiment_startup_report.shutil.which", return_value=None), \
                 patch("scripts.experiment_startup_report._package_version", return_value="not-installed"), \
                 patch("scripts.experiment_startup_report.workspace_discovery.classify_checkout", return_value=checkout) as classify, \
                 patch("scripts.experiment_startup_report.experiment_passport_gate.validate_binding",
                       return_value={"status": "PASS", "passport_status": "FROZEN", **binding}) as passport:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(startup_report_main(["--repo-root", str(root), "--campaign", "router.json", "--json"]), 0)
                report = json.loads(output.getvalue())
                self.assertEqual(report["campaign"]["campaign_id"], campaign_id)
                self.assertTrue(report["campaign"]["formal_task_consumption_allowed"])
                self.assertEqual(report["scope"]["authority"]["status"], "MATCHED", report["scope"]["authority"])
                self.assertEqual(report["runtime"]["repo"]["git_sha"], expected_sha)
                self.assertEqual(report["passport"]["experiment_id"], binding["experiment_id"])
                passport.assert_called_once_with(root, binding)
                classify.assert_called_once_with(root)
                frozen_profile = report["scope"]["authority"]["profile"]
                self.assertEqual(frozen_profile["source"]["resolved_path"], str(root))
                self.assertEqual(frozen_profile["source"]["revision"], expected_sha)
                self.assertEqual(frozen_profile["mode"], "host-orchestrator")
                self.assertEqual(frozen_profile["image_id"], profile["image_id"])
                self.assertNotIn("cd /global-control", output.getvalue())
                text = render_report(report["campaign"], report["runtime"], repo_root=root,
                                     scope=report["scope"], passport=report["passport"])
                self.assertNotIn("cd /global-control", text)
                self.assertIn("observed platform classification: NON_CANONICAL_EXECUTION_CHECKOUT", text)
                self.assertIn("report process", text)
                self.assertIn("scientific image: science:fixture", text)
                # A closed scoped target remains observable and never falls
                # back to the unrelated default platform or runs qualification.
                router["formal_task_consumption_allowed"] = False
                (root / "router.json").write_text(json.dumps(router), encoding="utf-8")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(startup_report_main(["--repo-root", str(root), "--campaign", "router.json", "--json"]), 0)
                closed = json.loads(output.getvalue())
                self.assertEqual(closed["campaign"]["campaign_id"], campaign_id)
                self.assertEqual(closed["scope"]["authority"]["status"], "NOT_CURRENT_EXECUTION_AUTHORITY")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(startup_report_main(["--json"]), 0)
                default = json.loads(output.getvalue())
                self.assertEqual(default["scope"]["kind"], "default_platform")
                self.assertEqual(default["campaign"]["campaign_id"], load_campaign()["campaign_id"])
                self.assertNotEqual(default["campaign"]["campaign_id"], campaign_id)
                self.assertIn("Default platform summary only", render_report(default["campaign"], default["runtime"]))

    def test_startup_report_repo_root_requires_contained_explicit_target(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("scripts.experiment_startup_report.collect_runtime") as runtime, \
             contextlib.redirect_stderr(io.StringIO()):
            root = Path(directory)
            for arguments in (["--repo-root", str(root)],
                              ["--repo-root", str(root), "--campaign", str(CAMPAIGN_PATH)]):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as result:
                    startup_report_main(arguments)
                self.assertEqual(result.exception.code, 2)
            runtime.assert_not_called()

    def test_claude_memory_imports_agent_rules_and_gpu_case(self):
        claude_memory = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        normalized = " ".join(claude_memory.split())
        self.assertIn("@AGENTS.md", claude_memory)
        self.assertIn("owner unresolved from this namespace", normalized)
        self.assertIn("2026-08-15-gpu-1-6-7-occupancy-and-nvitop-misdiagnosis.md", claude_memory)

    def test_case_index_covers_every_case(self):
        case_dir = REPO_ROOT / "docs" / "cases"
        index = (case_dir / "README.md").read_text(encoding="utf-8")
        unindexed = [path.name for path in case_dir.glob("*.md") if path.name != "README.md" and f"]({path.name})" not in index]
        self.assertEqual(unindexed, [])

    def test_current_state_entrypoints_route_to_current_campaign(self):
        entrypoints = (
            REPO_ROOT / "README.md",
            REPO_ROOT / "configs" / "experiment" / "README.md",
            REPO_ROOT / "docs" / "evidence" / "README.md",
            REPO_ROOT / "docs" / "experiment-runs" / "README.md",
            REPO_ROOT / "docs" / "webshop" / "CURRENT_FINDINGS.md",
        )
        for path in entrypoints:
            text = path.read_text(encoding="utf-8")
            self.assertIn("current-campaign.json", text, path)
        findings = (REPO_ROOT / "docs" / "webshop" / "CURRENT_FINDINGS.md").read_text(encoding="utf-8")
        self.assertIn("SUPERSEDED historical snapshot", findings)

    def test_q03_router_is_scoped_and_hash_bound(self):
        global_campaign = load_campaign(CAMPAIGN_PATH)
        self.assertEqual(global_campaign["campaign_id"], "20260928-2338-post-advisor-unanswered")
        self.assertFalse(global_campaign["formal_task_consumption_allowed"])

        scoped_path = REPO_ROOT / "configs" / "experiment" / "activations" / "q03-step119-val128-current-campaign-20261025.json"
        scoped = load_campaign(scoped_path)
        self.assertEqual(scoped["campaign_id"], "20261025-0825-q03-step119-val128")
        self.assertFalse(scoped["formal_task_consumption_allowed"])
        self.assertFalse(scoped["gpu_allocation_allowed"])
        self.assertEqual(scoped["status"], "blocked-pre-execution")
        release_path = REPO_ROOT / scoped["release_receipt"]
        self.assertTrue(json.loads(release_path.read_text())["formal_task_consumption_allowed"])
        self.assertEqual(hashlib.sha256(release_path.read_bytes()).hexdigest(), scoped["release_receipt_sha256"])

    def test_historical_nvitop_receipts_have_a_correction_map(self):
        case = (REPO_ROOT / "docs" / "cases" / "2026-08-15-gpu-1-6-7-occupancy-and-nvitop-misdiagnosis.md").read_text(encoding="utf-8")
        self.assertIn("Historical correction map", case)
        self.assertIn("h1.9-state-aware-rescue-fresh-seed-confirmation-v1.json", case)
        self.assertIn("h1.10-new-task-state-aware-transfer-v1.json", case)

    def test_active_handoff_does_not_restore_the_orphaned_vram_diagnosis(self):
        handoff = (REPO_ROOT / "docs" / "agent-handoffs" / "OPEN_EVO_WEBSHOP_AUTONOMOUS_WINDOW_2026-08-18_2134_UTC8.md").read_text(encoding="utf-8")
        self.assertIn("owner was unresolved from the `experiment-host` namespace", handoff)
        self.assertNotIn("orphaned VRAM", handoff)

    def test_h137_gpu_deviation_is_namespace_scoped_and_unresolved(self):
        text_targets = (
            REPO_ROOT / "scripts" / "openevo_webshop" / "launch_h137_frozen_adapter_eval_resume.sh",
            REPO_ROOT / "scripts" / "openevo_webshop" / "verify_h1_37_integrity.py",
            REPO_ROOT / "docs" / "evidence" / "remote-runs" / "2026-08-18" / "h1.37" / "REPORT.md",
        )
        forbidden = ("orphaned cuda vram", "orphaned memory", "no owning process/container", "no owning process, no container")
        for path in text_targets:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            self.assertIn("experiment-host", text, path)
            self.assertIn("unresolved", text, path)
            self.assertTrue("fail closed" in text or "failed closed" in text, path)
            self.assertFalse(any(phrase in text for phrase in forbidden), path)

        integrity = json.loads(
            (REPO_ROOT / "docs" / "evidence" / "remote-runs" / "2026-08-18" / "h1.37" / "evidence" / "h1.37-integrity.json").read_text(encoding="utf-8")
        )["preregistration_deviation"]
        self.assertEqual(integrity["gpu_observation_scope"], "experiment-host container PID namespace")
        self.assertIn("unresolved", integrity["gpu_ownership_conclusion"])
        summary = integrity["summary"].lower()
        self.assertIn("experiment-host", summary)
        self.assertIn("unresolved", summary)
        self.assertIn("failed closed", summary)
        self.assertFalse(any(phrase in summary for phrase in forbidden))

    def test_gpu_resource_gate_labels_its_namespace_and_non_attribution_scope(self):
        row = {"index": 0, "uuid": "GPU-test", "memory_used_mib": 2, "memory_total_mib": 32607, "utilization_gpu_percent": 0}
        with (
            patch.object(gpu_resource_gate, "_gpu_row", return_value=row),
            patch.object(gpu_resource_gate, "_compute_apps", return_value=[]),
            patch.object(gpu_resource_gate, "_device_handles", return_value=[]),
            patch.object(gpu_resource_gate, "_run", side_effect=lambda args: {"hostname": "experiment-host", "whoami": "root"}[args[0]]),
        ):
            snapshot = gpu_resource_gate.snapshot(["GPU-test"])
        self.assertIn("PID namespace", snapshot["observation_scope"])
        self.assertIn("not attempted", snapshot["ownership_attribution"])
        self.assertTrue(snapshot["all_eligible"])

    def test_startup_gate_does_not_coerce_compute_query_errors_to_empty_results(self):
        gate = (REPO_ROOT / "scripts" / "openevo_webshop" / "startup_gate_5090.sh").read_text(encoding="utf-8")
        self.assertIn("compute_apps_rc", gate)
        self.assertIn("compute-app query failed", gate)
        self.assertIn("returned non-numeric memory.used", gate)
        self.assertIn("returned non-numeric utilization.gpu", gate)
        self.assertIn("max_idle_memory_mib=16", gate)
        self.assertIn("max_idle_util_percent=0", gate)
        self.assertNotIn("OPENEVO_MAX_IDLE_MEMORY_MIB", gate)
        self.assertNotIn("compute_apps=\"$(nvidia-smi -i \"${gpu_uuid}\" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)\"", gate)

    def test_wait_for_idle_releases_its_launcher_lock_after_success(self):
        waiter = REPO_ROOT / "scripts" / "openevo_webshop" / "wait_for_idle_5090.sh"
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "scratch"
            gate = Path(directory) / "always-pass-gate.sh"
            gate.write_text("#!/usr/bin/env bash\nshift\nexec \"$@\"\n", encoding="utf-8")
            gate.chmod(0o700)
            environment = {
                **os.environ,
                "OPENEVO_AUTHORIZED_GPU_UUIDS": "GPU-test",
                "OPENEVO_GATE_SCRIPT": str(gate),
            }
            result = subprocess.run(
                ["bash", str(waiter), "--scratch-root", str(scratch), "--", "true"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((scratch / "wait" / "launch.lock.d").exists())

    def test_wait_for_idle_cleanup_does_not_remove_a_replacement_lock(self):
        waiter = REPO_ROOT / "scripts" / "openevo_webshop" / "wait_for_idle_5090.sh"
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "scratch"
            gate = Path(directory) / "always-pass-gate.sh"
            gate.write_text("#!/usr/bin/env bash\nshift\nexec \"$@\"\n", encoding="utf-8")
            gate.chmod(0o700)
            replace = Path(directory) / "replace-lock.sh"
            lock_dir = scratch / "wait" / "launch.lock.d"
            replace.write_text(
                "#!/usr/bin/env bash\n"
                f'rm -f -- "{lock_dir}/pid" "{lock_dir}/owner"\n'
                f'rmdir "{lock_dir}"\n'
                f'mkdir "{lock_dir}"\n'
                f'printf "%s\\n" replacement > "{lock_dir}/pid"\n',
                encoding="utf-8",
            )
            replace.chmod(0o700)
            environment = {
                **os.environ,
                "OPENEVO_AUTHORIZED_GPU_UUIDS": "GPU-test",
                "OPENEVO_GATE_SCRIPT": str(gate),
            }
            result = subprocess.run(
                ["bash", str(waiter), "--scratch-root", str(scratch), "--", str(replace)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((lock_dir / "pid").read_text(encoding="utf-8").strip(), "replacement")

    def test_h137_resume_waiter_fails_closed_on_invalid_gpu_queries(self):
        launcher = (REPO_ROOT / "scripts" / "openevo_webshop" / "launch_h137_frozen_adapter_eval_resume.sh").read_text(encoding="utf-8")
        self.assertIn('query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" || return 1', launcher)
        self.assertIn('query-compute-apps=pid --format=csv,noheader,nounits)" || return 1', launcher)


if __name__ == "__main__":
    unittest.main()
