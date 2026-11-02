"""No-GPU negative controls for source-bound candidate publication."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts.openevo_runtime import science_ready_projection as projection
from scripts.openevo_runtime.science_ready_projection import ProjectionError, project_pool


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root / "exact_launcher.py"
        self.src.write_text("print('frozen')\n", encoding="utf-8")
        self.done = self.root / "sealed.json"
        self.done.write_text('{"status":"PASS"}', encoding="utf-8")
        self.manifest = {
            "schema": "openevo.advisor-task-pool.v1",
            "tasks": [
                {"id": "sealed", "mode": "adopt", "gpu_uuid": "GPU-111", "completion": {
                    "path": str(self.done), "equals": {"status": "PASS"}}},
                {"id": "next", "mode": "queued", "gpu_uuid": "GPU-222", "depends_on": ["sealed"],
                 "argv": ["python3", str(self.src), "--launch"],
                 "source_files": {str(self.src): digest(self.src.read_bytes())},
                 "output_root": str(self.root / "run-next")},
            ],
        }
        self.state = {"tasks": {
            "sealed": {"status": "COMPLETE", "attempted": True, "completion_sha256": digest(self.done.read_bytes())},
            "next": {"status": "QUEUED"},
        }}
        self.write()

    def write(self) -> None:
        raw = json.dumps(self.manifest, sort_keys=True).encode()
        (self.root / "manifest.json").write_bytes(raw)
        self.state["manifest_sha256"] = digest(raw)
        (self.root / "state.json").write_text(json.dumps(self.state, sort_keys=True), encoding="utf-8")

    def view(self) -> dict:
        return project_pool(self.root, required_campaigns=("Q02", "Q04", "Q10"))

    def test_only_unused_exact_candidate_is_reported(self) -> None:
        out = self.view()
        self.assertEqual(out["candidate_count"], 1)
        self.assertEqual(out["coverage"]["unobserved_campaigns"], ["Q02", "Q10"])
        self.assertEqual(out["coverage"]["global_verdict"], "READY_SUPPLY_UNVERIFIED_CROSS_CAMPAIGN")
        self.assertFalse(out["execution_authority"])
        self.assertEqual(out["tasks"][1]["verdict"], "CANDIDATE_FOR_EXISTING_LAUNCHER")

    def test_manifest_queued_mode_is_not_replay_authority(self) -> None:
        self.state["tasks"]["next"] = {"status": "CONTINUATION_REQUIRED", "attempted": True}
        self.write()
        self.assertEqual(self.view()["candidate_count"], 0)

    def test_queued_after_attempt_must_not_retry(self) -> None:
        self.state["tasks"]["next"]["attempted"] = True
        self.write()
        self.assertEqual(self.view()["tasks"][1]["reason"], "MODE_OR_ATTEMPTED")

    def test_missing_or_changed_seal_blocks_dependency(self) -> None:
        self.done.write_text('{"status":"FAIL"}', encoding="utf-8")
        self.assertEqual(self.view()["tasks"][1]["reason"], "DEPENDENCY_NOT_SEALED")

    def test_source_hash_drift_blocks(self) -> None:
        self.src.write_text("print('drifted')\n", encoding="utf-8")
        self.assertEqual(self.view()["tasks"][1]["reason"], "SOURCE_IDENTITY_DRIFT")

    def test_changed_launcher_without_repin_blocks(self) -> None:
        self.manifest["tasks"][1]["argv"][1] = str(self.root / "unregistered.py")
        self.write()
        self.assertEqual(self.view()["tasks"][1]["reason"], "LAUNCHER_OR_ACTIVATION_UNBOUND")

    def test_activation_must_bind_exact_bytes(self) -> None:
        activation = self.root / "activation.json"
        activation.write_text('{"status":"AUTHORIZED"}', encoding="utf-8")
        self.manifest["tasks"][1]["argv"] += ["--activation", str(activation),
                                                  "--activation-sha256", "f" * 64]
        self.write()
        self.assertEqual(self.view()["tasks"][1]["reason"], "LAUNCHER_OR_ACTIVATION_UNBOUND")

    def test_prior_dispatch_claim_blocks(self) -> None:
        (self.root / "next-dispatch.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.view()["tasks"][1]["reason"], "ALREADY_CLAIMED")

    def test_existing_output_blocks(self) -> None:
        out = self.root / "run-next"
        out.mkdir()
        (out / "do-not-overwrite").write_text("science", encoding="utf-8")
        self.assertEqual(self.view()["tasks"][1]["reason"], "OUTPUT_ALREADY_PRESENT")

    def test_identity_mismatch_is_hard_failure(self) -> None:
        self.state["manifest_sha256"] = "0" * 64
        (self.root / "state.json").write_text(json.dumps(self.state), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "manifest bytes"):
            self.view()

    def test_unknown_state_fails_closed(self) -> None:
        self.state["tasks"]["next"]["status"] = "READY"  # not a native producer status
        self.write()
        with self.assertRaisesRegex(ProjectionError, "unknown status"):
            self.view()

    def test_symlink_source_fails(self) -> None:
        (self.root / "manifest.json").unlink()
        (self.root / "manifest.json").symlink_to(self.src)
        with self.assertRaisesRegex(ProjectionError, "symlink"):
            self.view()

    def test_missing_task_is_hard_failure(self) -> None:
        del self.state["tasks"]["next"]
        self.write()
        with self.assertRaisesRegex(ProjectionError, "identity set"):
            self.view()


class Q09NativeStateTests(ProjectionTests):
    """Q09 native execution is observable but never a cross-campaign launch grant."""

    def setUp(self) -> None:
        super().setUp()
        self.q09_path = self.root / "q09-execution-state.json"
        self.q09_state = {
            "schema": projection.Q09_STATE_SCHEMA,
            "q09_stream_id": "q09-stream-1",
            "execution_git_sha": "a" * 40,
            "owner_launch_release_sha256": "b" * 64,
            "formal_ready_package_sha256": "c" * 64,
            "q09_schedule_sha256": "d" * 64,
            "state_id": "e" * 64,
            "status": "RUNNING",
            "next_round": 140,
            "rounds_complete": list(range(140)),
            "rollouts_consumed": 128 * 140,
            "final_panel_access_count": 0,
            "external_teacher_calls": 0,
            # GPU assignment is resource history, not a new scientific budget.
            "q09_selected_gpu_uuids": ["GPU-original"],
        }
        self.write_q09()

    def write_q09(self) -> None:
        self.q09_path.write_text(json.dumps(self.q09_state, sort_keys=True), encoding="utf-8")

    def joint_view(self, *, sha256: str | None = None) -> dict:
        return project_pool(self.root, required_campaigns=("Q04", "Q09", "Q10"),
                            q09_execution_state=self.q09_path, q09_expected_sha256=sha256)

    def test_running_round_is_observed_but_never_new_ready_work(self) -> None:
        out = self.joint_view()
        self.assertEqual(out["coverage"]["observed_campaigns"], ["Q04", "Q09"])
        self.assertEqual(out["coverage"]["unobserved_campaigns"], ["Q10"])
        self.assertEqual(out["candidate_count"], 1)  # only original Q04 candidate
        self.assertEqual(out["campaigns"]["Q09"]["verdict"], "STATE_REPORTS_RUNNING")
        self.assertFalse(out["campaigns"]["Q09"]["execution_authority"])
        self.assertEqual(out["campaigns"]["Q09"]["next_round"], 140)
        self.assertEqual(out["campaigns"]["Q09"]["remaining_rounds_nominal"], 20)
        self.assertIn("q09_execution_state_sha256", out["sources"])

    def test_health_pause_is_a_hold_not_a_retry(self) -> None:
        self.q09_state["status"] = "PAUSE_FOR_DIAGNOSTIC"
        self.write_q09()
        out = self.joint_view()
        self.assertEqual(out["campaigns"]["Q09"]["verdict"], "BLOCKED")
        self.assertFalse(out["campaigns"]["Q09"]["execution_authority"])

    def test_nominal_terminal_round_needs_native_seal(self) -> None:
        self.q09_state.update(status="COMPLETE", next_round=160,
                              rounds_complete=list(range(160)), rollouts_consumed=128 * 160)
        self.write_q09()
        self.assertEqual(self.joint_view()["campaigns"]["Q09"]["verdict"], "AWAITING_NATIVE_SEAL")

    def test_round_accounting_drift_fails_closed(self) -> None:
        self.q09_state["rounds_complete"].pop()
        self.write_q09()
        with self.assertRaisesRegex(ProjectionError, "completed-round/rollout accounting"):
            self.joint_view()

    def test_paid_rollout_count_drift_fails_closed(self) -> None:
        self.q09_state["rollouts_consumed"] += 128
        self.write_q09()
        with self.assertRaisesRegex(ProjectionError, "completed-round/rollout accounting"):
            self.joint_view()

    def test_protected_budget_drift_fails_closed(self) -> None:
        self.q09_state["external_teacher_calls"] = 1
        self.write_q09()
        with self.assertRaisesRegex(ProjectionError, "protected budget"):
            self.joint_view()

    def test_unrecognized_release_or_state_fails_closed(self) -> None:
        self.q09_state["owner_launch_release_sha256"] = "unbound"
        self.write_q09()
        with self.assertRaisesRegex(ProjectionError, "science identity"):
            self.joint_view()
        self.q09_state["owner_launch_release_sha256"] = "b" * 64
        self.q09_state["status"] = "READY_FOR_RETRY"
        self.write_q09()
        with self.assertRaisesRegex(ProjectionError, "status/round combination"):
            self.joint_view()

    def test_exact_caller_pin_detects_state_changed_during_handoff(self) -> None:
        original_sha = digest(self.q09_path.read_bytes())
        self.q09_state["next_round"] = 141
        self.q09_state["rounds_complete"].append(140)
        self.q09_state["rollouts_consumed"] += 128
        self.write_q09()
        with self.assertRaisesRegex(ProjectionError, "caller-pinned SHA-256"):
            self.joint_view(sha256=original_sha)

    def test_pin_without_native_state_path_is_invalid(self) -> None:
        with self.assertRaisesRegex(ProjectionError, "Q09 expected SHA requires"):
            project_pool(self.root, q09_expected_sha256="a" * 64)


class Q10NativeBlockedTests(ProjectionTests):
    """Q10 preflight is a verified blocker, never spendable science work."""

    def setUp(self) -> None:
        super().setUp()
        self.candidate_path = self.root / "q10-candidate.json"
        self.candidate = {
            "schema": "openevo.q10.native-activation.v1",
            "status": "PREPARED_NOT_ACTIVATED",
            "owner_pr": 637,
            "execution_git_sha": "a" * 40,
            "formal_task_consumption_allowed": False,
            "gpu_allocation_allowed": False,
            "final_panel_access": 0,
        }
        self.readiness_path = self.root / "q10-cpu-readiness.json"
        self.readiness = {
            "status": "CPU_INPUTS_VERIFIED_NOT_FORMAL_READY",
            "identity": {"status": "CPU_IDENTITY_ELIGIBILITY_PASS"},
            "panel": {"status": "CPU_PANEL_ELIGIBILITY_PASS", "final_panel_access": 0, "tasks": 128},
            "formal_task_consumption": 0,
            "gpu_requests": 0,
            "execution_git_sha": "a" * 40,
            "remaining": ["exact resource matrix", "canonical committed release", "native zero-formal qualification"],
        }
        self.write_q10()

    def write_q10(self) -> None:
        self.candidate_path.write_text(json.dumps(self.candidate), encoding="utf-8")
        self.readiness["candidate"] = {
            "path": str(self.candidate_path),
            "sha256": digest(self.candidate_path.read_bytes()),
        }
        self.readiness_path.write_text(json.dumps(self.readiness), encoding="utf-8")

    def joined(self) -> dict:
        return project_pool(self.root, required_campaigns=("Q02", "Q04", "Q10"),
                            q10_cpu_readiness=self.readiness_path)

    def test_cpu_readiness_is_blocker_not_ready_task(self) -> None:
        out = self.joined()
        self.assertEqual(out["candidate_count"], 1)  # still only Q04 candidate
        self.assertEqual(out["coverage"]["observed_campaigns"], ["Q04", "Q10"])
        self.assertEqual(out["coverage"]["unobserved_campaigns"], ["Q02"])
        self.assertEqual(out["campaigns"]["Q10"]["verdict"], "BLOCKED")
        self.assertFalse(out["campaigns"]["Q10"]["execution_authority"])
        self.assertEqual(out["coverage"]["global_verdict"], "READY_SUPPLY_UNVERIFIED_CROSS_CAMPAIGN")

    def test_missing_candidate_blocks_whole_view(self) -> None:
        self.candidate_path.unlink()
        with self.assertRaisesRegex(ProjectionError, "unreadable source"):
            self.joined()

    def test_candidate_hash_drift_blocks(self) -> None:
        self.candidate_path.write_text('{}', encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "candidate bytes differ"):
            self.joined()

    def test_cpu_report_claiming_gpu_use_is_inconsistent(self) -> None:
        self.readiness["gpu_requests"] = 1
        self.write_q10()
        with self.assertRaisesRegex(ProjectionError, "science consumption"):
            self.joined()

    def test_candidate_granting_formal_task_consumption_is_not_cpu_only(self) -> None:
        self.candidate["formal_task_consumption_allowed"] = True
        self.write_q10()
        with self.assertRaisesRegex(ProjectionError, "forbidden-to-start"):
            self.joined()

    def test_candidate_execution_sha_divergence_is_rejected(self) -> None:
        self.candidate["execution_git_sha"] = "b" * 40
        self.write_q10()
        with self.assertRaisesRegex(ProjectionError, "forbidden-to-start"):
            self.joined()

    def test_new_formal_status_cannot_inherit_cpu_eligibility(self) -> None:
        self.readiness["status"] = "READY_FOR_FORMAL"
        self.write_q10()
        with self.assertRaisesRegex(ProjectionError, "release-specific adapter"):
            self.joined()

    def test_caller_pinned_q10_readiness_sha_must_match(self) -> None:
        original_sha = digest(self.readiness_path.read_bytes())
        self.readiness["remaining"].append("new unresolved gate")
        self.write_q10()
        with self.assertRaisesRegex(ProjectionError, "caller-pinned SHA-256"):
            project_pool(self.root, q10_cpu_readiness=self.readiness_path,
                         q10_cpu_expected_sha256=original_sha)

    def test_caller_pinned_q10_readiness_sha_accepts_exact_source(self) -> None:
        expected = digest(self.readiness_path.read_bytes())
        result = project_pool(self.root, q10_cpu_readiness=self.readiness_path,
                              q10_cpu_expected_sha256=expected)
        self.assertEqual(result["sources"]["q10_cpu_readiness_sha256"], expected)
        self.assertEqual(result["campaigns"]["Q10"]["verdict"], "BLOCKED")

    def test_expected_sha_requires_actual_q10_source(self) -> None:
        with self.assertRaisesRegex(ProjectionError, "requires a Q10 readiness path"):
            project_pool(self.root, q10_cpu_expected_sha256="a" * 64)

    def test_changed_candidate_status_not_cpu_only(self) -> None:
        self.candidate["status"] = "READY_FOR_FORMAL"
        self.write_q10()
        with self.assertRaisesRegex(ProjectionError, "forbidden-to-start"):
            self.joined()

    def test_wrong_candidate_owner_is_not_the_cpu_contract(self) -> None:
        self.candidate["owner_pr"] = 999
        self.write_q10()
        with self.assertRaisesRegex(ProjectionError, "forbidden-to-start"):
            self.joined()

    def test_boolean_zero_side_effect_cannot_masquerade_as_integer(self) -> None:
        self.readiness["gpu_requests"] = False
        self.write_q10()
        with self.assertRaisesRegex(ProjectionError, "science consumption"):
            self.joined()

    def test_boolean_final_panel_access_is_not_zero_counter(self) -> None:
        self.readiness["panel"]["final_panel_access"] = False
        self.write_q10()
        with self.assertRaisesRegex(ProjectionError, "panel identity"):
            self.joined()

    def test_q04_rotating_during_q10_join_rejects_mixed_snapshot(self) -> None:
        original = projection._q10_cpu_blocked

        def rotate(path: Path, **kwargs: object):
            q10 = original(path, **kwargs)
            self.state["tasks"]["next"]["status"] = "RUNNING"
            self.write()
            return q10

        with mock.patch.object(projection, "_q10_cpu_blocked", side_effect=rotate):
            with self.assertRaisesRegex(ProjectionError, "Q04 producer changed"):
                self.joined()

    def test_symlink_preflight_rejected(self) -> None:
        self.readiness_path.unlink()
        self.readiness_path.symlink_to(self.candidate_path)
        with self.assertRaisesRegex(ProjectionError, "symlink"):
            self.joined()


if __name__ == "__main__":
    unittest.main()
