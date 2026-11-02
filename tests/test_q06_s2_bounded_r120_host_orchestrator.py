from __future__ import annotations

import hashlib
import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.openevo_webshop import q06_s2_bounded_r120_host_orchestrator as host
from scripts.openevo_webshop import q06_s2_bounded_r120_probe_guard as probe_guard


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return digest(raw)


def make_contract(adapter_path: Path, activation_path: str, cost_source_ref: dict | None = None) -> dict:
    adapter_bytes = adapter_path.read_bytes()
    return {
        "schema": host.CONTRACT_SCHEMA,
        "contract_id": "fixture-q06-s2-bounded-r120",
        "question_id": "Q06_TOKEN_CLICK_ENTROPY",
        "stage": "S2",
        "status": "LAUNCH_AUTHORIZED",
        "append_only": True,
        "cell": {
            "cell_id": host.CELL_ID,
            "mechanism": "Bounded State",
            "checkpoint_id": host.CHECKPOINT_ID,
            "completed_stage2_rounds": 120,
            "actual_parameter_update_events": 118,
            "source_summary_sha256": "c" * 64,
            "evaluation_panel_sha256": "4" * 64,
        },
        "known_shared_identity": {
            "base_model": "Qwen/Qwen3-1.7B",
            "base_model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "common_agent_system_sha256": "a" * 64,
            "common_skill_sha256": "b" * 64,
            "common_memory_sha256": "d" * 64,
            "sampling": {"temperature": 1.0},
            "adapter": {
                "candidate_checkpoint_id": host.CHECKPOINT_ID,
                "candidate_source_run": str(adapter_path.parent),
                "relative_file": adapter_path.name,
                "size_bytes": len(adapter_bytes),
                "adapter_sha256": digest(adapter_bytes),
                "status": "VERIFIED_READ_ONLY",
            },
        },
        "execution_constraints": {
            "training_rollouts": 0, "optimizer_updates": 0, "teacher_calls": 0,
            "adaptive_reruns": 0, "retry_on_scientific_outcome": False,
            "episodes": 64, "passes": 1, "max_generation_calls": 960,
            "max_generated_tokens": 491520, "max_scorer_forward_calls": 100000,
            "max_scorer_input_tokens": 1000000, "max_gpu_hours": 2,
            "max_persistent_output_bytes": 1073741824,
            "final_panel_access_allowed": False,
            "formal_task_consumption_allowed": True,
            "gpu_allocation_allowed": True,
        },
        "gates": {
            "checkpoint_identity": {"status": "PASS_READ_ONLY_IDENTITY"},
            "scoring_cost_ceiling": {"status": "FROZEN"},
            "scorer_cost_source": {
                **(cost_source_ref or {"path": "cost.json", "sha256": "0" * 64}),
                "classification": "derived_cpu_only_not_observed_forward_calls",
                "largest_single_arm_forward_calls": 42073,
                "largest_single_arm_input_tokens": 386471,
            },
            "resource_qualification": {"status": "PASS"},
            "task_bearing_activation": {"status": "PASS", "path": activation_path},
        },
    }


class Q06S2BoundedR120HostTests(unittest.TestCase):
    def assertGateError(self, pattern, callback, *args, **kwargs):
        with self.assertRaises(host.GateError) as caught:
            callback(*args, **kwargs)
        self.assertRegex(str(caught.exception), pattern)

    def test_formal_start_digest_accepts_bare_and_prefixed_same_sha256(self):
        value = "a" * 64
        self.assertTrue(host.same_sha256_digest(value, "sha256:" + value))
        self.assertTrue(host.same_sha256_digest("sha256:" + value, value))
        self.assertFalse(host.same_sha256_digest(value, "b" * 64))
        self.assertFalse(host.same_sha256_digest("sha256:bad", value))

    def test_current_prospective_contract_fails_closed_without_activation(self):
        repo = Path(__file__).resolve().parents[1]
        contract_path = repo / "configs/experiment/advisor-questions/q06-s2-bounded-r120-prospective-contract.v1.json"
        ref = {"path": contract_path.relative_to(repo).as_posix(), "sha256": host.sha256_file(contract_path)}
        self.assertGateError("not launch-authorized", host.load_contract, repo, ref)

    def test_contract_reference_hash_and_cell_identity_are_required(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            adapter = root / "payload" / "adapter.safetensors"
            adapter.parent.mkdir()
            adapter.write_bytes(b"adapter-fixture")
            contract_path = repo / (host.CONTRACT_PREFIX + "v2.json")
            contract_path.parent.mkdir(parents=True)
            contract = make_contract(adapter, "configs/experiment/activations/s2.json")
            contract_path.write_text(json.dumps(contract))
            bad_ref = {"path": contract_path.relative_to(repo).as_posix(), "sha256": "0" * 64}
            self.assertGateError("byte hash mismatch", host.load_contract, repo, bad_ref)

            contract["cell"]["checkpoint_id"] = "historical_gdr_r159_rank128"
            contract_path.write_text(json.dumps(contract))
            ref = {"path": contract_path.relative_to(repo).as_posix(), "sha256": host.sha256_file(contract_path)}
            self.assertGateError("not the exact single Bounded-R120 cell", host.load_contract, repo, ref)

    def test_release_router_and_profile_bind_exact_contract_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            adapter = root / "payload" / "adapter.safetensors"
            adapter.parent.mkdir()
            adapter.write_bytes(b"bounded-adapter")
            contract_rel = host.CONTRACT_PREFIX + "v2.json"
            router_rel = "configs/experiment/activations/q06-s2.json"
            release_rel = "configs/experiment/activations/q06-s2-release.json"
            profile_rel = "configs/experiment/profiles/q06-s2.json"
            orchestrator_rel = "scripts/openevo_webshop/q06_s2_bounded_r120_host_orchestrator.py"
            guard_rel = "scripts/openevo_webshop/q06_s2_bounded_r120_probe_guard.py"
            manifest_rel = "docs/evidence/q06-s2/manifest.json"
            cost_rel = "docs/evidence/q06-s2/cost.json"
            for relative in (contract_rel, router_rel, release_rel, profile_rel,
                             orchestrator_rel, guard_rel, manifest_rel, cost_rel):
                (repo / relative).parent.mkdir(parents=True, exist_ok=True)
            (repo / orchestrator_rel).write_text("adapter source fixture")
            (repo / guard_rel).write_text("guard source fixture")
            orchestrator_ref = {"path": orchestrator_rel, "sha256": host.sha256_file(repo / orchestrator_rel)}
            guard_ref = {"path": guard_rel, "sha256": host.sha256_file(repo / guard_rel)}
            release = {
                "schema": "openevo.experiment-q06-s2-bounded-r120-release.v1",
                "status": "launch-authorized", "formal_task_consumption_allowed": True,
                "formal_rollout_limit": 64, "final_panel_access_limit": 0,
                "campaign_id": "s2-fixture", "experiment_id": "q06-s2-fixture",
                "endpoint": host.CHECKPOINT_ID,
                "execution_source": {
                    "path": str(repo.resolve()), "repository": "https://example.invalid/repo.git",
                    "revision": "a" * 40,
                },
                "canonical_activation": {"path": router_rel},
                "prospective_contract": {"path": contract_rel, "sha256": "0" * 64},
                "source_manifest": {"path": manifest_rel, "sha256": "1" * 64},
                "execution_readiness": {"profile": {"path": profile_rel, "sha256": "f" * 64}},
                "host_orchestrator": orchestrator_ref,
                "probe_guard": guard_ref,
            }
            cost_sha = write_json(repo / cost_rel, {
                "classification": "derived_cpu_only_not_observed_forward_calls",
                "arms": {"beta": {"groups": {
                    "all": {"estimated_forward_calls": 42073,
                            "estimated_input_tokens": 386471}
                }}},
            })
            contract = make_contract(adapter, router_rel, {"path": cost_rel, "sha256": cost_sha})
            contract_sha = write_json(repo / contract_rel, contract)
            release["prospective_contract"]["sha256"] = contract_sha
            router = {
                "schema": "openevo.q06.s2-bounded-r120-current-campaign.v1",
                "status": "launch-authorized", "formal_task_consumption_allowed": True,
                "formal_rollout_limit": 64, "final_panel_access_allowed": False,
                "campaign_id": "s2-fixture", "experiment_id": "q06-s2-fixture",
                "endpoint": host.CHECKPOINT_ID,
                "prospective_contract": release["prospective_contract"],
                "authorized_gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000000"],
            }
            release_sha = write_json(repo / release_rel, release)
            router["release_receipt_sha256"] = release_sha
            write_json(repo / router_rel, router)
            profile = {
                "execution": {"mode": "host-orchestrator"},
                "s2_prospective_contract": release["prospective_contract"],
                "host_orchestrator": orchestrator_ref,
                "probe_guard": guard_ref,
                "execution_source": release["execution_source"],
                "command": ["python3", str((repo / orchestrator_rel).resolve()),
                            "--authority-repo", str(repo.resolve()),
                            "--release", str((repo / release_rel).resolve()),
                            "--router", str((repo / router_rel).resolve()),
                            "--execution-source", str(repo.resolve())],
            }
            write_json(repo / profile_rel, profile)
            release["execution_readiness"]["profile"]["sha256"] = host.sha256_file(repo / profile_rel)
            release_sha = write_json(repo / release_rel, release)
            router["release_receipt_sha256"] = release_sha
            write_json(repo / router_rel, router)

            loaded_release, loaded_router = host.validate_release_router(repo, repo / release_rel, repo / router_rel)
            loaded_contract_path, loaded_contract, loaded_contract_sha = host.load_contract(
                repo, loaded_release["prospective_contract"])
            profile_path, _ = host.validate_profile_and_contract_binding(
                repo, repo / release_rel, repo / router_rel, loaded_release, loaded_router,
                loaded_contract_path, loaded_contract, loaded_contract_sha)
            self.assertEqual(profile_path, (repo / profile_rel).resolve())

            loaded_router["release_receipt_sha256"] = "0" * 64
            write_json(repo / router_rel, loaded_router)
            self.assertGateError("does not bind exact release bytes", host.validate_release_router,
                                 repo, repo / release_rel, repo / router_rel)

    def test_payload_hash_and_q17_endpoint_are_exact(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "round-119" / "adapter_model.safetensors"
            adapter.parent.mkdir()
            adapter.write_bytes(b"bounded-r120-exact-payload")
            contract = make_contract(adapter, "configs/experiment/activations/s2.json")
            payload = contract["known_shared_identity"]["adapter"]
            manifest = {
                "endpoint": {
                    "adapter_path": str(adapter.parent),
                    "checkpoint_id": host.CHECKPOINT_ID,
                    "source_files": {"adapter": {
                        "path": str(adapter), "bytes": adapter.stat().st_size,
                        "sha256": host.sha256_file(adapter),
                    }},
                },
                "checkpoints": {host.CHECKPOINT_ID: {"adapter_sha256": payload["adapter_sha256"]}},
            }
            self.assertEqual(host.validate_adapter_payload(contract, manifest), adapter)
            for name in ("source", "seed", "model", "memory", "skill", "agent"):
                (root / name).mkdir(exist_ok=True)
            manifest.update({
                "manifest_path": str(root / "manifest.json"),
                "authority_repo": str(root),
                "probe_guard_path": str(root / "guard.py"),
                "max_output_bytes": 1073741824,
                "budget": {"max_scorer_forward_calls": 100000,
                           "max_scorer_input_tokens": 1000000},
                "seed_root": {"path": str(root / "seed")},
                "model": {"local_model_path": str(root / "model"), "revision": "rev"},
                "common_carriers": {
                    "memory_path": str(root / "memory"),
                    "skill_path": str(root / "skill"),
                    "agent_system_path": str(root / "agent"),
                },
                "task_panel_manifest_path": "panel.json",
            })
            argv = host.build_probe_command(
                manifest, root / "source", root / "out",
                "GPU-00000000-0000-0000-0000-000000000000")
            self.assertIn(str(root / "source" / host.PROBE_PATH), argv)
            self.assertEqual(argv[argv.index("--checkpoint-id") + 1], host.CHECKPOINT_ID)
            self.assertEqual(argv[argv.index("--sd-adapter") + 1], str(adapter.parent))
            self.assertEqual(argv[argv.index("--max-scorer-forward-calls") + 1], "100000")
            self.assertEqual(argv[argv.index("--max-scorer-input-tokens") + 1], "1000000")
            manifest["endpoint"]["source_files"]["adapter"]["sha256"] = "0" * 64
            self.assertGateError("identical adapter payload bytes", host.validate_adapter_payload,
                                 contract, manifest)

    def test_adapter_path_is_derived_from_hash_bound_source_summary(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter_dir = root / "runs" / "actual-run" / "round-119" / "sd-lora" / "bounded-state"
            adapter_dir.mkdir(parents=True)
            adapter_file = adapter_dir / "adapter_model.safetensors"
            adapter_file.write_bytes(b"the exact frozen adapter")
            contract = make_contract(adapter_file, "configs/experiment/activations/s2.json")
            summary_path = root / "source-summary.json"
            summary_sha = write_json(summary_path, {
                "checkpoint_id": host.CHECKPOINT_ID,
                "checkpoint": {
                    "adapter_path": str(adapter_dir),
                    "adapter_sha256": digest(adapter_file.read_bytes()),
                    "actual_parameter_update_events": 118,
                    "state_id": "state-r120",
                },
            })
            contract["cell"].update({
                "source_summary_path": str(summary_path),
                "source_summary_sha256": summary_sha,
                "actual_parameter_update_events": 118,
            })
            contract["known_shared_identity"]["adapter"]["relative_file"] = (
                "a-stale-run-root/round-119/sd-lora/bounded-state/adapter_model.safetensors"
            )
            contract["known_shared_identity"]["adapter"]["size_bytes"] = adapter_file.stat().st_size
            source_summary, resolved_dir, resolved_file = host.resolve_source_summary_adapter(contract)
            self.assertEqual(source_summary["checkpoint"]["adapter_sha256"], digest(adapter_file.read_bytes()))
            self.assertEqual(resolved_dir, adapter_dir)
            self.assertEqual(resolved_file, adapter_file)
            manifest = {"endpoint": {
                "adapter_path": str(adapter_dir),
                "source_files": {"adapter": {
                    "path": str(adapter_file), "bytes": adapter_file.stat().st_size,
                    "sha256": digest(adapter_file.read_bytes()),
                }},
            }}
            self.assertEqual(host.validate_adapter_payload(contract, manifest), adapter_file)

            summary_path.write_text('{"checkpoint_id":"after_120_rounds"}\n')
            self.assertGateError("source summary is missing or differs", host.resolve_source_summary_adapter, contract)

    def test_execution_source_must_be_exact_clean_repo_and_ref(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary).resolve()
            expected = "a" * 40
            run = Mock(side_effect=[
                SimpleNamespace(stdout=str(source) + "\n"),
                SimpleNamespace(stdout="https://example.invalid/source.git\n"),
                SimpleNamespace(stdout=expected + "\n"),
                SimpleNamespace(stdout=""),
            ])
            self.assertEqual(host.verify_execution_source(source, {
                "path": str(source), "repository": "https://example.invalid/source.git",
                "revision": expected,
            }, run=run), expected)
            ssh_alias = Mock(side_effect=[
                SimpleNamespace(stdout=str(source) + "\n"),
                SimpleNamespace(stdout="git@github-openevo:mykcs/openevo-experiment.git\n"),
                SimpleNamespace(stdout=(
                    "url.git@github-openevo:mykcs/openevo-experiment.insteadof "
                    "https://github.com/mykcs/openevo-experiment\n")),
                SimpleNamespace(stdout=expected + "\n"),
                SimpleNamespace(stdout=""),
            ])
            self.assertEqual(host.verify_execution_source(source, {
                "path": str(source),
                "repository": "https://github.com/mykcs/openevo-experiment.git",
                "revision": expected,
            }, run=ssh_alias), expected)
            dirty = Mock(side_effect=[
                SimpleNamespace(stdout=str(source) + "\n"),
                SimpleNamespace(stdout="https://example.invalid/source.git\n"),
                SimpleNamespace(stdout=expected + "\n"),
                SimpleNamespace(stdout=" M file.py\n"),
            ])
            self.assertGateError("exact clean pinned", host.verify_execution_source, source, {
                "path": str(source), "repository": "https://example.invalid/source.git",
                "revision": expected,
            }, dirty)
            changed_head = Mock(side_effect=[
                SimpleNamespace(stdout=str(source) + "\n"),
                SimpleNamespace(stdout="https://example.invalid/source.git\n"),
                SimpleNamespace(stdout="b" * 40 + "\n"),
                SimpleNamespace(stdout=""),
            ])
            self.assertGateError("exact clean pinned", host.verify_execution_source, source, {
                "path": str(source), "repository": "https://example.invalid/source.git",
                "revision": expected,
            }, changed_head)

    def test_uid_gate_and_q17_wrapper_run_between_model_load_and_first_generation(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            host.uid_write_rename_gate(output, geteuid=lambda: 1001)
            self.assertEqual(list(output.iterdir()), [])
            with self.assertRaisesRegex(host.GateError, "UID 1001"):
                host.uid_write_rename_gate(output, geteuid=lambda: 501)

            events = []

            class Runtime:
                def __init__(self):
                    events.append("model-loaded")

            fake_probe = SimpleNamespace(load_science_modules=lambda _root: {"Runtime": Runtime})
            probe_guard.install_post_load_gate(
                fake_probe, output, gate=lambda _path: events.append("write-rename-gate"))
            wrapped_modules = fake_probe.load_science_modules(Path("unused"))
            wrapped_modules["Runtime"]()
            events.append("first-generation")
            self.assertEqual(events, ["model-loaded", "write-rename-gate", "first-generation"])

    def test_output_writer_guard_enforces_cumulative_utf8_bytes_before_write(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = output / "first.txt"
            second = output / "second.txt"
            over = output / "over.txt"
            _, restore = probe_guard.install_output_byte_guard(output, 6)
            try:
                first.write_text("é")  # default text output is forced to UTF-8: 2 bytes
                second.write_text("1234")  # cumulative 6 bytes
                self.assertEqual(host.output_tree_bytes(output), 6)
                with self.assertRaisesRegex(host.GateError, "before write"):
                    over.write_text("x", encoding="utf-8")
                self.assertEqual(over.stat().st_size, 0)
                self.assertEqual(host.output_tree_bytes(output), 6)
                non_utf8 = output / "non-utf8.txt"
                with self.assertRaisesRegex(host.GateError, "must use UTF-8"):
                    non_utf8.write_text("x", encoding="latin-1")
                self.assertFalse(non_utf8.exists())
            finally:
                restore()

    def test_scorer_cost_guard_counts_before_forward_and_seals_receipt(self):
        class FakeIds:
            def __init__(self, count):
                self.count = count

            def numel(self):
                return self.count

        model_calls = []

        class FakeModel:
            device = "cpu"

            def __call__(self, **kwargs):
                model_calls.append(kwargs["input_ids"].numel())
                return "model-output"

        def score(self):
            self.model(input_ids=FakeIds(2))
            self.model(input_ids=FakeIds(1))
            return "scored"

        saved = {}
        fake_probe = SimpleNamespace(
            GenerateObserver=SimpleNamespace(score_click_destinations=score),
            run_episode=lambda: "episode-started",
            atomic_json=lambda path, payload: saved.setdefault(Path(path).name, payload),
        )
        state, restore = probe_guard.install_scorer_cost_guard(
            fake_probe, max_forward_calls=2, max_input_tokens=3)
        observer = SimpleNamespace(model=FakeModel())
        try:
            self.assertEqual(fake_probe.GenerateObserver.score_click_destinations(observer), "scored")
            self.assertEqual(state["forward_calls"], 2)
            self.assertEqual(state["input_tokens"], 3)
            self.assertEqual(len(model_calls), 2)
            with self.assertRaisesRegex(host.GateError, "ceiling reached before model forward"):
                fake_probe.GenerateObserver.score_click_destinations(observer)
            self.assertEqual(len(model_calls), 2)
            self.assertTrue(state["budget_exceeded"])
            with self.assertRaisesRegex(host.GateError, "remaining episodes were not started"):
                fake_probe.run_episode()
            payload = {"probe_receipt": {"schema": "test"}}
            fake_probe.atomic_json(Path("summary.json"), payload)
            self.assertFalse(saved["summary.json"]["scorer_cost_receipt"]["within_budget"])
            receipt_payload = {"schema": "test"}
            fake_probe.atomic_json(Path("probe-receipt.json"), receipt_payload)
            self.assertEqual(receipt_payload["scorer_cost"]["forward_calls"], 2)
        finally:
            restore()

    def test_wall_and_output_caps_kill_only_owned_process_and_container(self):
        class FakeProcess:
            pid = 4242
            returncode = None
            create_output = None

            def poll(self):
                if self.create_output is not None:
                    self.create_output.write_bytes(b"x" * 11)
                    self.create_output = None
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = -15
                return self.returncode

        for reason, max_seconds, oversized in (
            ("wall cap", host.MAX_WALL_SECONDS, False),
            ("output-byte ceiling", 10, True),
        ):
            with self.subTest(reason=reason), TemporaryDirectory() as temporary:
                output = Path(temporary)
                child = FakeProcess()
                if oversized:
                    child.create_output = output / "too-large.bin"
                launched = {}

                def popen(command, **kwargs):
                    launched["command"] = command
                    launched["kwargs"] = kwargs
                    return child

                def run(command, **kwargs):
                    launched.setdefault("cleanup", []).append(command)
                    if command[1] == "inspect":
                        label = next(part for part in launched["command"] if part.startswith(
                            "com.openevo.q06.s2.owner="))
                        return SimpleNamespace(returncode=0, stdout=label.split("=", 1)[1])
                    return SimpleNamespace(returncode=0, stdout="")

                clock = Mock(side_effect=[0, 0 if oversized else max_seconds])
                with patch.object(host.os, "killpg") as killpg:
                    self.assertGateError(reason, host.run_with_limits,
                                         ["docker", "run", "--rm", "--network", "none", "--user", "1001:1001"],
                                         output, max_wall_seconds=max_seconds,
                                         max_output_bytes=10, popen=popen, run=run,
                                         monotonic=clock, sleep=lambda _seconds: None)
                command = launched["command"]
                self.assertTrue(launched["kwargs"]["start_new_session"])
                self.assertEqual(command[0:2], ["docker", "run"])
                name = command[command.index("--name") + 1]
                self.assertTrue(name.startswith("q06-s2-bounded-r120-"))
                self.assertEqual(launched["cleanup"][-1], ["docker", "rm", "--force", name])
                killpg.assert_called_once_with(child.pid, host.signal.SIGTERM)


class Q06S2BoundedR120SuccessorArtifactTests(unittest.TestCase):
    def test_successor_release_router_and_profile_are_hash_bound(self):
        repo = Path(__file__).resolve().parents[1]
        release_path = repo / "configs/experiment/activations/q06-s2-bounded-r120-task-bearing-release-20261002.json"
        router_path = repo / "configs/experiment/activations/q06-s2-bounded-r120-task-bearing-current-campaign-20261002.json"
        release, router = host.validate_release_router(repo, release_path, router_path)
        contract_path, contract, contract_sha = host.load_contract(repo, release["prospective_contract"])

        self.assertEqual(router["release_receipt_sha256"], host.sha256_file(release_path))
        self.assertEqual(router["prospective_contract"], release["prospective_contract"])
        self.assertEqual(contract_path.name, "q06-s2-bounded-r120-prospective-contract.v2.json")
        self.assertEqual(contract["execution_constraints"]["episodes"], 64)
        self.assertEqual(contract["execution_constraints"]["passes"], 1)
        self.assertEqual(contract["execution_constraints"]["training_rollouts"], 0)
        self.assertEqual(contract["execution_constraints"]["optimizer_updates"], 0)
        self.assertIs(contract["execution_constraints"]["final_panel_access_allowed"], False)

        manifest_ref = release["source_manifest"]
        manifest_path, _ = host.repo_file(repo, manifest_ref, "S2 probe manifest")
        manifest = host.read_object(manifest_path)
        self.assertEqual(manifest["contract"], {"path": contract_path.relative_to(repo).as_posix(), "sha256": contract_sha})
        self.assertEqual(manifest["checkpoints"][host.CHECKPOINT_ID]["state_id"],
                         "e5a3480ec014f230de46095b8092813730e4b7e74bc3b5e83499530c33d94770")
        self.assertEqual(manifest["endpoint"]["adapter_path"],
                         "/workspace/runs/bounded-effective-state-gdr-recovery-1e7c4952-20260916/off/round-119/sd-lora/bounded-state")
        self.assertEqual(contract["gates"]["launch_qualification"]["receipt"]["receipt_sha256"],
                         "08e3899afe5b3f61727457781971432c188d2f1079ca9ae122520e85e76974cc")

        profile_ref = release["execution_readiness"]["profile"]
        profile_path, _ = host.repo_file(repo, profile_ref, "S2 runtime profile")
        profile = host.read_object(profile_path)
        self.assertEqual(profile["s2_prospective_contract"], release["prospective_contract"])
        self.assertEqual(profile["execution_source"], release["execution_source"])
        self.assertEqual(profile["execution"]["mode"], "host-orchestrator")
        authority = "/workspace/maintenance/ADVISOR-UNANSWERED-20260928/zero-formal/q06-s2-bounded-r120-268fc8e3/authority-source-f96bfcb2"
        self.assertEqual(profile["command"][1], f"PYTHONPATH={authority}")
        self.assertEqual(profile["command"][3], f"{authority}/scripts/openevo_webshop/q06_s2_bounded_r120_host_orchestrator.py")
        self.assertEqual(profile["command"][profile["command"].index("--authority-repo") + 1], authority)
        for flag in ("--release", "--router"):
            self.assertTrue(profile["command"][profile["command"].index(flag) + 1].startswith(authority + "/"))


if __name__ == "__main__":
    unittest.main()
