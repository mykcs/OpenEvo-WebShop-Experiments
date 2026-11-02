import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys

from scripts.analysis.q03_identity_bridge_audit import ROOT, audit_plan, audit_source, compare_evaluators, verify_artifact, audit_native_rng_configs

sys.path.insert(0, str(ROOT / "vendor/openevo-webshop/src"))
from openevo_webshop.comparison_contract import seed_project_action
from openevo_webshop.seed_eval import run_seed_eval_episode
from scripts.openevo_webshop.seed_runtime.verify_llm_webshop import validate_train_pool_index


def _added_patch_block(target, marker):
    patch = (ROOT / "patches/q03/seed-consumer-wiring-202610041457.patch").read_text()
    header = f"diff --git a/{target} b/{target}\n"
    sections = patch.split("diff --git ")
    section = next((part for part in sections if part.startswith(f"a/{target} b/{target}\n")), None)
    if section is None:
        raise AssertionError(f"patch does not modify {target}")
    lines = section.splitlines()
    marker_index = next((i for i, line in enumerate(lines) if line == f"+{marker}"), None)
    if marker_index is None:
        raise AssertionError(f"patch is missing {marker}")
    block = []
    for line in lines[marker_index:]:
        if not line.startswith("+") or line.startswith("+++"):
            break
        block.append(line[1:])
    return "\n".join(block) + "\n"


def _exec_added_patch_block(target, marker):
    namespace = {"hashlib": hashlib}
    exec(_added_patch_block(target, marker), namespace)
    return namespace


class FixtureEnvironment:
    """An interface fixture; it never constructs WebShop or consumes a task."""

    def __init__(self):
        self.actions = []
        self.resets = []
        self.observation = "WebShop [SEP] Instruction: [SEP] buy red shoes [SEP] Search"

    def reset(self, idx, seed=None):
        self.resets.append((idx, seed))
        return self.observation, {}

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": []}

    def step(self, action):
        self.actions.append(action)
        return self.observation, 0.0, False, {"task_score": 0.0}


class FixtureBackend:
    """Prewritten completions, with no model, provider, or GPU calls."""

    def __init__(self, completion):
        self.completion = completion
        self.requests = []

    def generate(self, prompt, **kwargs):
        self.requests.append((prompt, kwargs))
        return self.completion


class Q03SeedConsumerTests(unittest.TestCase):
    def test_actual_native_configs_enforce_the_checkpoint_and_validation_consumers(self):
        package = json.loads((ROOT / "configs/experiment/advisor-questions/q03-work-package-202610041457.json").read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            def receipt(checkpoint=True, test_freq=-1, env_name="Webshop"):
                refs = {}
                for stream in package["stage2"]["streams"]:
                    path = Path(directory) / f'{stream["id"]}-{checkpoint}-{test_freq}-{env_name}.json'
                    config = {"trainer": {"checkpoint_environment_rng": checkpoint, "test_freq": test_freq,
                                          "val_before_train": False},
                              "env": {"env_name": env_name, "seed": stream["environment_seed"]}, "data": {"seed": stream["data_seed"]},
                              "actor_rollout_ref": {"rollout": {"engine_seed": stream["policy_seed"]}}}
                    path.write_text(json.dumps(config))
                    refs[stream["id"]] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                return {"effective_native_configs": refs}
            audit_native_rng_configs(receipt(), package)
            with self.assertRaisesRegex(ValueError, "checkpoint environment RNG"):
                audit_native_rng_configs(receipt(checkpoint=False), package)
            with self.assertRaisesRegex(ValueError, "auxiliary validation"):
                audit_native_rng_configs(receipt(test_freq=5), package)
            with self.assertRaisesRegex(ValueError, "owner must be Webshop"):
                audit_native_rng_configs(receipt(env_name="Alfworld"), package)
            with self.assertRaisesRegex(ValueError, "both actual"):
                audit_native_rng_configs({}, package)

    def test_environment_rng_resume_is_required_by_the_machine_contract(self):
        package = json.loads((ROOT / "configs/experiment/advisor-questions/q03-work-package-202610041457.json").read_bytes())
        self.assertTrue(package["stage2"]["environment_rng_checkpoint_required"])
        for value in (False, None):
            changed = copy.deepcopy(package)
            changed["stage2"]["environment_rng_checkpoint_required"] = value
            with self.assertRaisesRegex(Exception, "task-selection RNG"):
                audit_plan(changed)

    def test_environment_rng_patch_uses_native_owner_and_checkpoint_preflight(self):
        patch = (ROOT / "patches/q03/seed-consumer-wiring-202610041457.patch").read_text()
        for owner in ("agent_system/environments/env_package/webshop/envs.py",
                      "agent_system/environments/base.py", "verl/trainer/ppo/ray_trainer.py"):
            self.assertIn(f"diff --git a/{owner} b/{owner}", patch)
        self.assertIn("+  checkpoint_environment_rng: False", patch)
        self.assertIn("+            raise ValueError('WebShop task-selection checkpoint identity mismatch')", patch)
        self.assertIn("+                raise RuntimeError(\"checkpoint is missing required environment task-selection RNG state\")", patch)
        self.assertIn("+                raise RuntimeError(\"checkpoint is missing required dataloader state for exact resume\")", patch)
        self.assertIn("+            self._restore_environment_rng(environment_rng)", patch)

    def test_qualification_requires_an_actual_native_training_session(self):
        for index in (-1, 0, 499, 1000):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "native train pool"):
                    validate_train_pool_index(index, goal_count=1000)
        self.assertEqual(validate_train_pool_index(500, goal_count=1000), 500)

    def test_smoke_completion_is_invalid_and_passes_through_without_substitution(self):
        raw = "results: [SEP] 1. [SEP"
        action, valid = seed_project_action(raw)
        self.assertEqual(action, raw.lower()[-20:])
        self.assertFalse(valid)
        env, backend = FixtureEnvironment(), FixtureBackend(raw)
        result = run_seed_eval_episode(env, backend, 500, max_steps=1)
        self.assertEqual(env.actions, [action])
        self.assertNotEqual(env.actions, ["search"])
        self.assertEqual(result["invalid_action_steps"], 1)
        self.assertEqual(result["task_score"], 0.0)
        self.assertFalse(result["won"])

    def test_valid_action_does_not_require_reward_or_success(self):
        env = FixtureEnvironment()
        backend = FixtureBackend("<think>look</think><action>SEARCH[RED SHOES]</action>")
        result = run_seed_eval_episode(env, backend, 500, max_steps=1)
        self.assertEqual(env.actions, ["search[red shoes]"])
        self.assertEqual(result["invalid_action_steps"], 0)
        self.assertEqual(result["task_score"], 0.0)
        self.assertFalse(result["won"])

    def test_seed_prompt_history_and_generation_seed_are_consumed(self):
        env = FixtureEnvironment()
        backend = FixtureBackend("<think>look</think><action>search[red shoes]</action>")
        run_seed_eval_episode(env, backend, 500, environment_seed=7,
                              generation_seed=101, max_steps=2)
        self.assertEqual(env.resets, [(500, 7)])
        self.assertEqual([r[1]["generation_seed"] for r in backend.requests], [101, 102])
        self.assertTrue(all("<think>" in r[0] and "<action>" in r[0]
                            for r in backend.requests))
        self.assertIn("search[red shoes]", backend.requests[1][0])
        self.assertEqual(len(env.actions), 2)

    def test_task_identity_mismatch_stops_before_generation_or_action(self):
        env, backend = FixtureEnvironment(), FixtureBackend("unused")
        with self.assertRaisesRegex(ValueError, "task description hash mismatch"):
            run_seed_eval_episode(env, backend, 500,
                                  expected_task_description_sha256="0" * 64)
        self.assertEqual(backend.requests, [])
        self.assertEqual(env.actions, [])

    def test_missing_think_and_cjk_remain_invalid_without_action_repair(self):
        for raw in ("<action>click[buy now]</action>",
                    "<think>购买</think><action>click[buy now]</action>"):
            with self.subTest(raw=raw):
                action, valid = seed_project_action(raw)
                self.assertEqual(action, "click[buy now]")
                self.assertFalse(valid)


class Q03AuditTests(unittest.TestCase):
    def setUp(self):
        self.package = json.loads((ROOT / "configs/experiment/advisor-questions/q03-work-package-202610041457.json").read_bytes())

    def test_two_streams_and_zero_native_budget(self):
        self.assertEqual(audit_plan(self.package)["canonical_episodes"], 640)
        changed = copy.deepcopy(self.package)
        changed["stage2"]["streams"][1]["environment_seed"] = 0
        with self.assertRaisesRegex(ValueError, "environment_seed"):
            audit_plan(changed)
        changed = copy.deepcopy(self.package)
        changed["stage2"]["native_validation"]["trainer.test_freq"] = 5
        with self.assertRaisesRegex(ValueError, "native validation"):
            audit_plan(changed)

    def test_rollout_engine_seed_derivation_is_source_bound_and_keeps_default(self):
        patch = (ROOT / "patches/q03/seed-consumer-wiring-202610041457.patch").read_text()
        self.assertIn(
            "index d5249ce0fbcf0a85ecbb60893e2b858f0f02cdee..",
            patch,
        )
        self.assertIn(
            "index 990f7c1fabfd7083960412635909042b1961071e..",
            patch,
        )
        self.assertIn(
            "index f5980129e91e5a4baf3ad544516aceeb829cc258..",
            patch,
        )
        namespace = _exec_added_patch_block(
            "verl/workers/sharding_manager/fsdp_vllm.py",
            "def _rollout_seed_for_dp_rank(engine_seed, dp_rank):",
        )
        derive = namespace["_rollout_seed_for_dp_rank"]
        self.assertEqual(derive(None, 0), 1000)
        self.assertEqual(derive(None, 7), 1007)
        stream_a = [derive(20261001, rank) for rank in range(8)]
        stream_b = [derive(20261002, rank) for rank in range(8)]
        self.assertEqual(len(set(stream_a)), 8)
        self.assertEqual(len(set(stream_b)), 8)
        self.assertTrue(set(stream_a).isdisjoint(stream_b))

    def test_rollout_rng_payload_binds_dp_rank_and_cpu_tensor_shape(self):
        namespace = _exec_added_patch_block(
            "verl/workers/sharding_manager/fsdp_vllm.py",
            "def _validate_rollout_rng_state_payload(",
        )
        validate = namespace["_validate_rollout_rng_state_payload"]

        class CpuTensor:
            device = type("Device", (), {"type": "cpu"})()
            dtype = "uint8"
            shape = (4,)

            def numel(self):
                return 4

        tensor = CpuTensor()
        base = {"version": 1, "engine_seed": 20261001, "dp_rank": 3,
                "derived_seed": 1234, "gen_random_states": tensor}
        self.assertIs(validate(base, engine_seed=20261001, dp_rank=3, derived_seed=1234,
                               expected_shape=(4,), expected_dtype="uint8", tensor_type=CpuTensor), tensor)
        with self.assertRaisesRegex(ValueError, "dp_rank mismatch"):
            validate(base, engine_seed=20261001, dp_rank=4, derived_seed=1234,
                     expected_shape=(4,), expected_dtype="uint8", tensor_type=CpuTensor)
        with self.assertRaisesRegex(ValueError, "dtype or shape mismatch"):
            validate(base, engine_seed=20261001, dp_rank=3, derived_seed=1234,
                     expected_shape=(8,), expected_dtype="uint8", tensor_type=CpuTensor)

    def test_native_rollout_rng_checkpoint_roundtrip_and_missing_state_is_fatal(self):
        namespace = _exec_added_patch_block(
            "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
            "ROLLOUT_RNG_STATE_KEY = \"q03_rollout_rng\"",
        )
        save_state = namespace["_save_rollout_rng_state"]
        load_state = namespace["_load_rollout_rng_state"]
        validate_checkpoint = namespace["_validate_rollout_rng_checkpoint"]

        class StateProvider:
            def __init__(self):
                self.state = {"version": 1, "engine_seed": 20261001, "rng": b"before"}

            def get_rollout_rng_state(self):
                return copy.deepcopy(self.state)

            def validate_rollout_rng_state(self, state):
                if state.get("engine_seed") != 20261001:
                    raise ValueError("engine seed mismatch")
                return state

            def set_rollout_rng_state(self, state):
                self.state = copy.deepcopy(state)

        provider = StateProvider()
        extra = {"rng": "learner-rng"}
        save_state(extra, provider)
        self.assertEqual(extra["q03_rollout_rng"], provider.state)
        provider.state = {"rng": b"advanced"}
        load_state(extra, provider)
        self.assertEqual(provider.state["rng"], b"before")

        missing = {"rng": "learner-rng"}
        before = copy.deepcopy(provider.state)
        with self.assertRaisesRegex(RuntimeError, "missing required rollout RNG"):
            load_state(missing, provider)
        self.assertEqual(provider.state, before)

        unchanged = {"rng": "legacy"}
        save_state(unchanged, None)
        load_state(unchanged, None)
        validate_checkpoint(unchanged, None)
        self.assertEqual(unchanged, {"rng": "legacy"})

        unexpected = {"rng": "legacy", "q03_rollout_rng": {"version": 1}}
        with self.assertRaisesRegex(RuntimeError, "no rollout RNG owner"):
            validate_checkpoint(unexpected, None)
        with self.assertRaisesRegex(RuntimeError, "no rollout RNG owner"):
            load_state(unexpected, None)

    def test_conditional_forward_budget_rejects_missing_teacher_padding(self):
        primary = next(stage for stage in self.package["stages"]
                       if stage["id"] == "B_TWO_SEED_STAGE2")
        self.assertEqual(primary["budget"]["policy_teacher_logprob_forward_calls_upper_bound"], 40960)
        self.assertTrue(primary["budget"]["learner_reference_teacher_forward_bound_basis"].startswith(
            "CONDITIONAL_PROFILE_NOT_EXECUTION_FROZEN:"))
        changed = copy.deepcopy(self.package)
        next(stage for stage in changed["stages"] if stage["id"] == "B_TWO_SEED_STAGE2")["budget"]["policy_teacher_logprob_forward_calls_upper_bound"] = 38400
        with self.assertRaisesRegex(ValueError, "conditional native compute ceiling"):
            audit_plan(changed)

    def test_engineering_revalidation_is_cumulative_and_preserves_science(self):
        budget = self.package["budget_policy"]
        self.assertIsNone(budget["qualification_generation_calls"])
        self.assertIsNone(budget["qualification_gpu_hours_upper_bound"])
        self.assertEqual(budget["qualification_gpu_hours_planning_estimate"], 2)
        self.assertIn("may continue after model generation or action", budget["qualification_no_science_retry_rule"])
        changed = copy.deepcopy(self.package)
        changed["budget_policy"]["qualification_no_science_retry_rule"] = "after any generation or action, stop without retry"
        with self.assertRaisesRegex(ValueError, "engineering repair after generation"):
            audit_plan(changed)
        changed = copy.deepcopy(self.package)
        changed["budget_policy"]["qualification_gpu_hours_planning_estimate"] = 3
        audit_plan(changed)
        changed["budget_policy"]["qualification_gpu_hours_upper_bound"] = 3
        with self.assertRaisesRegex(ValueError, "qualification budget provenance"):
            audit_plan(changed)

    def test_optimizer_ceiling_counts_active_steps_instead_of_trajectories(self):
        primary = next(stage for stage in self.package["stages"]
                       if stage["id"] == "B_TWO_SEED_STAGE2")
        self.assertEqual(primary["budget"]["actor_optimizer_steps_upper_bound"], 9600)
        changed = copy.deepcopy(self.package)
        next(stage for stage in changed["stages"] if stage["id"] == "B_TWO_SEED_STAGE2")["budget"]["actor_optimizer_steps_upper_bound"] = 640
        with self.assertRaisesRegex(ValueError, "optimizer-step"):
            audit_plan(changed)

    def test_canonical_generation_budget_counts_both_physical_phases(self):
        self.assertEqual(self.package["budget_policy"]["evaluation_generated_token_upper_bound_per_64_episode_milestone"], 552960)
        changed = copy.deepcopy(self.package)
        next(stage for stage in changed["stages"] if stage["id"] == "B_TWO_SEED_STAGE2")["budget"]["evaluation_generated_tokens_upper_bound"] = 4915200
        with self.assertRaisesRegex(ValueError, "evaluation-token"):
            audit_plan(changed)

    def test_unqualified_optimizer_ceiling_cannot_claim_execution_freeze(self):
        changed = copy.deepcopy(self.package)
        primary = next(stage for stage in changed["stages"] if stage["id"] == "B_TWO_SEED_STAGE2")
        primary["budget"]["actor_optimizer_step_bound_basis"] = "EXECUTION_FROZEN"
        with self.assertRaisesRegex(ValueError, "unfrozen execution"):
            audit_plan(changed)

    def test_context_accounts_for_both_generation_phases(self):
        changed = copy.deepcopy(self.package)
        context = next(stage for stage in changed["stages"] if stage["id"] == "C_OPENEVO_CONTEXT")
        context["budget"]["evaluation_generation_calls_upper_bound"] = 9600
        with self.assertRaisesRegex(ValueError, "evaluation-call"):
            audit_plan(changed)

    def test_same_panel_cannot_hide_sampling_or_carrier_change(self):
        evaluator = {"panel_sha256": self.package["evaluation"]["panel_sha256"],
                     "panel_manifest_sha256": self.package["evaluation"]["panel_manifest_sha256"],
                     "environment_seed_policy": self.package["evaluation"]["environment_seed_policy"],
                     "harness_sha256": "1" * 64, "runtime_execution_sha": "2" * 40,
                     "runtime_fingerprint_sha256": "3" * 64,
                     "prompt_contract_sha256": "4" * 64, "action_contract_sha256": "5" * 64,
                     "parser_sha256": "6" * 64, "carriers": {"memory": "7" * 64, "skill": "8" * 64, "agent": "9" * 64},
                     "sampling": self.package["evaluation"]["sampling"]}
        self.assertEqual(len(compare_evaluators(evaluator, copy.deepcopy(evaluator), self.package)), 64)
        for field in ("temperature", "max_steps", "step_seed_policy"):
            changed = copy.deepcopy(evaluator)
            changed["sampling"][field] = .4 if field == "temperature" else 30 if field == "max_steps" else "shared seed"
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                compare_evaluators(evaluator, changed, self.package)
        changed = copy.deepcopy(evaluator)
        changed["carriers"]["skill"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            compare_evaluators(evaluator, changed, self.package)

    def test_receipt_bytes_are_verified_not_just_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            path.write_text('{"status":"PASS"}')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(verify_artifact({"path": str(path), "sha256": digest}, digest), digest)
            path.write_text('{"status":"FAIL"}')
            with self.assertRaisesRegex(ValueError, "bytes SHA"):
                verify_artifact({"path": str(path), "sha256": digest}, digest)

    def test_source_bytes_not_repository_label_decide_fidelity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = copy.deepcopy(self.package)
            for relative in package["source"]["files"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)
                package["source"]["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(len(audit_source(root, package)["verified"]), 5)
            first = next(iter(package["source"]["files"]))
            (root / first).write_text("a semantic compatibility mutation")
            with self.assertRaisesRegex(ValueError, "source bytes mismatch"):
                audit_source(root, package)

    def test_source_inventory_cannot_be_empty_incomplete_or_unhashed(self):
        for mutation in ("empty", "missing", "extra", "bad_sha"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(self.package)
                files = changed["source"]["files"]
                first = next(iter(files))
                if mutation == "empty":
                    files.clear()
                elif mutation == "missing":
                    del files[first]
                elif mutation == "extra":
                    files["unexpected.py"] = "0" * 64
                else:
                    files[first] = "not-a-sha"
                with self.assertRaisesRegex(ValueError, "source (inventory|SHA)"):
                    audit_plan(changed)
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaisesRegex(ValueError, "source (inventory|SHA)"):
                        audit_source(Path(tmp), changed)

    def test_runtime_requires_a_complete_immutable_fingerprint(self):
        evaluator = {"panel_sha256": self.package["evaluation"]["panel_sha256"],
                     "panel_manifest_sha256": self.package["evaluation"]["panel_manifest_sha256"],
                     "environment_seed_policy": self.package["evaluation"]["environment_seed_policy"],
                     "harness_sha256": "1" * 64, "runtime_execution_sha": "2" * 40,
                     "runtime_fingerprint_sha256": "3" * 64,
                     "prompt_contract_sha256": "4" * 64, "action_contract_sha256": "5" * 64,
                     "parser_sha256": "6" * 64, "carriers": {"memory": "7" * 64, "skill": "8" * 64, "agent": "9" * 64},
                     "sampling": self.package["evaluation"]["sampling"]}
        for value in ("", "g" * 64, "3" * 63, "3" * 64 + "junk"):
            with self.subTest(value=value):
                changed = copy.deepcopy(evaluator)
                changed["runtime_fingerprint_sha256"] = value
                with self.assertRaisesRegex(ValueError, "complete immutable SHA256 fingerprint"):
                    compare_evaluators(changed, copy.deepcopy(changed), self.package)


if __name__ == "__main__":
    unittest.main()
