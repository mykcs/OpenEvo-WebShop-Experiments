"""The Step119 candidate binds one exact checkpoint and 128 val-only tasks."""
import hashlib
import json
import re
import copy
import importlib.util
import sys
import types
from pathlib import Path

from scripts.openevo_webshop.q03_source_faithful_manifest import (
    bind_validation_carrier_rows,
    canonical_sha256,
    environment_identity,
    to_evaluation_tasks,
    validate_and_select,
    validate_file_sha256,
    step_generation_seed,
)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/experiment/designs/q03-step119-validation-only-20261025.json"
PUBLIC_TASKS = ROOT / "configs/experiment/manifests/q03-step119-public-validation-id-manifest-20261025.json"
PASSPORT = ROOT / "configs/experiment/passports/202610251359-q03-step119-val128/passport.json"


def test_candidate_is_exactly_step119_and_not_a_neighbor_checkpoint():
    design = json.loads(DESIGN.read_text())
    assert design["status"] == "FORMAL_RELEASE_REQUIRED"
    assert design["checkpoint"]["native_step"] == 119
    assert design["checkpoint"]["native_actor_path"].endswith("global_step_119/actor")
    assert len(design["checkpoint"]["native_model_members"]) == 8
    assert all("global_step_119" in member["path"] for member in design["checkpoint"]["native_model_members"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", member["sha256"]) for member in design["checkpoint"]["native_model_members"])


def test_task_manifest_binds_source_faithful_public_ids_and_dataset_bytes():
    design = json.loads(DESIGN.read_text())
    manifest_path = ROOT / design["evaluation"]["task_manifest_path"]
    assert manifest_path == PUBLIC_TASKS
    tasks = json.loads(manifest_path.read_text())
    selected = validate_and_select(tasks)
    assert tasks["schema"] == "openevo.q03-step119-public-validation-id-manifest.v1"
    assert tasks["source_manifest_file_sha256"] == "60df28877c1168c46aa05f505830bb2923f8ee6100fa10b3b9597099a6efa76e"
    assert tasks["source_manifest_content_sha256"] == "dbc0e39f4ee4d3d3df3e9308aac57729881f4066996e7df3d0f4f7071fd4e355"
    assert tasks["task_count"] == design["evaluation"]["validation_rows"] == 128
    assert design["evaluation"]["task_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert design["evaluation"]["task_manifest_payload_sha256"] == tasks["payload_sha256"]
    assert design["evaluation"]["selected_panel_digest"] == tasks["selected_panel_digest"]
    assert design["evaluation"]["task_id_field"] == "tasks[].task_id"
    assert design["evaluation"]["task_session_field"] == "tasks[].sampled_session_index"
    assert design["evaluation"]["task_worker_seed_field"] == "tasks[].worker_seed"
    assert design["evaluation"]["task_split"] == "held_out"
    assert len({row["task_id"] for row in selected}) == 128
    assert all("instruction" not in row and "prompt" not in row for row in selected)

    passport = json.loads(PASSPORT.read_text())
    passport_ref = passport["scientific_identity"]["task_manifest"]
    assert passport_ref["path"] == "configs/experiment/manifests/q03-step119-public-validation-id-manifest-20261025.json"
    assert passport_ref["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    carrier = design["evaluation"]["parquet_carrier_binding"]
    assert carrier["row_count"] == 128
    assert carrier["producer_split"] == "test"
    assert carrier["producer_index_field"] == "extra_info.index"
    assert carrier["producer_index_values"].startswith("0..127;")
    assert carrier["shuffle"] is False
    assert carrier["session_draw"] == (
        "np.random.RandomState(env.seed + 1000).choice(np.arange(500), size=128, replace=False)"
    )
    assert carrier["worker_seed_formula"] == "env.seed + 1000 + panel_slot"


def test_validation_budget_cannot_train_or_open_final_panel():
    evaluation = json.loads(DESIGN.read_text())["evaluation"]
    runtime = evaluation["runtime"]
    assert evaluation["validation_rows"] == 128
    assert evaluation["val_only"] is True and evaluation["val_before_train"] is True
    assert evaluation["trainer_test_freq"] == -1
    assert evaluation["optimizer_updates"] == 0
    assert evaluation["training_rollouts"] == 0
    assert evaluation["final_panel_access"] == 0
    assert runtime["execution_mode"] == "host-orchestrator"
    assert runtime["runtime_entrypoint"] == {"kind": "module", "module": evaluation["entrypoint"]}
    assert runtime["runtime_import_modules"] == [
        "verl.trainer.main_ppo",
        "agent_system.multi_turn_rollout.rollout_loop",
        "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
        "agent_system.environments.env_manager",
        "agent_system.environments.env_package.webshop.envs",
        "verl.workers.actor.dp_actor",
        "verl.trainer.ppo.ray_trainer",
        "verl.trainer.ppo.q03_step119_validation_results",
    ]
    assert evaluation["entrypoint"] in runtime["runtime_import_modules"]
    assert evaluation["memory"] is False
    assert evaluation["skill"] is False
    assert evaluation["agent_system"] is False


def test_validation_sampling_matches_the_pinned_public_qwen3_recipe():
    evaluation = json.loads(DESIGN.read_text())["evaluation"]
    authority = evaluation["sampling_authority"]
    assert authority["repository"] == "jinyangwu/SEED"
    assert authority["revision"] == "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c"
    assert authority["launcher_path"] == "examples/grpo_trainer/run_webshop_qwen3.sh"
    assert evaluation["rollout_engine"] == "vllm"
    assert evaluation["max_new_tokens"] == 512
    assert evaluation["top_p"] == 1.0
    assert evaluation["top_k"] == -1
    assert evaluation["environment_seed"] == 0


def test_gpu_scope_is_user_authorized_pool_only():
    resources = json.loads(DESIGN.read_text())["resource_contract"]
    assert resources["authorized_gpu_indices"] == [1, 2, 3, 4]
    assert len(resources["authorized_gpu_uuids"]) == 4
    assert resources["preemption_allowed"] is False
    assert resources["kill_or_reset_allowed"] is False


def test_exporter_probe_identity_is_pinned_and_not_a_persistent_download():
    exporter = json.loads(DESIGN.read_text())["exporter"]
    assert exporter["sha256"] == "ad50e027112ad9c5d067764e05a5eda6ef644041a099b68dee60670a7a6f2cff"
    assert exporter["output_model_bytes"] == 4063515640
    assert exporter["output_model_sha256"] == "4b9207ce1d7fdbe9c3acb3e06684ef8d0a1bcdadf34d3fe6c23e8c47ec7f7b1b"
    assert exporter["persistent_export"] is False
    assert exporter["preserve_exporter_probe_identity_and_do_not_download_hf_checkpoint"] is True


def test_source_faithful_step119_manifest_is_id_only_and_maps_worker_session_pairs():
    manifest = json.loads(PUBLIC_TASKS.read_text())
    tasks = validate_and_select(manifest)
    evaluated = to_evaluation_tasks(tasks)
    assert len(evaluated) == 128
    assert len({task["task_id"] for task in evaluated}) == 128
    assert all("instruction" not in task for task in tasks)
    assert [task["panel_ordinal"] for task in evaluated] == list(range(128))
    assert all(task["goal_idx"] == task["sampled_session_index"] for task in evaluated)
    assert all(task["worker_seed"] == 1000 + task["panel_ordinal"] for task in evaluated)
    assert all(task["generation_seed"] == 2026082500 + task["panel_ordinal"] for task in evaluated)
    assert step_generation_seed(evaluated[0], 0) == 2026082500
    assert step_generation_seed(evaluated[0], 14) == 2026082514
    assert environment_identity(evaluated[0], default_seed=0) == (1000, 319)
    assert validate_file_sha256(PUBLIC_TASKS, hashlib.sha256(PUBLIC_TASKS.read_bytes()).hexdigest())


def test_source_faithful_step119_manifest_rejects_row_ordinals_and_identity_drift():
    manifest = json.loads(PUBLIC_TASKS.read_text())
    bad_id = copy.deepcopy(manifest)
    bad_id["tasks"][0]["task_id"] = "0"
    bad_id["payload_sha256"] = canonical_sha256({k: v for k, v in bad_id.items() if k != "payload_sha256"})
    try:
        validate_and_select(bad_id)
    except ValueError as exc:
        assert "task ID" in str(exc)
    else:
        raise AssertionError("row ordinal accepted as canonical public task ID")

    bad_session = copy.deepcopy(manifest)
    bad_session["tasks"][0]["sampled_session_index"] = 500
    bad_session["payload_sha256"] = canonical_sha256({k: v for k, v in bad_session.items() if k != "payload_sha256"})
    try:
        validate_and_select(bad_session)
    except ValueError as exc:
        assert "seed/session" in str(exc)
    else:
        raise AssertionError("out-of-pool session index accepted")

    for field in ("slot", "worker_seed", "sampled_session_index"):
        malformed = copy.deepcopy(manifest)
        malformed["tasks"][0][field] = float(malformed["tasks"][0][field])
        malformed["payload_sha256"] = canonical_sha256(
            {key: value for key, value in malformed.items() if key != "payload_sha256"}
        )
        try:
            validate_and_select(malformed)
        except ValueError as exc:
            assert "must be integers" in str(exc)
        else:
            raise AssertionError(f"non-integer {field} was silently coerced")

    try:
        validate_and_select(manifest, selected_slots=[0.0])
    except ValueError as exc:
        assert "must be integers" in str(exc)
    else:
        raise AssertionError("floating-point selected slot was silently coerced")

    try:
        validate_and_select(manifest, selected_slots=[0, 0])
    except ValueError as exc:
        assert "must be unique" in str(exc)
    else:
        raise AssertionError("duplicate selected slots were silently collapsed")


def test_legacy_goal_index_tasks_keep_their_original_environment_seed():
    assert environment_identity({"goal_idx": 17}, default_seed=0) == (0, 17)


def test_source_faithful_step119_manifest_rejects_wrong_release_file_hash():
    try:
        validate_file_sha256(PUBLIC_TASKS, "0" * 64)
    except ValueError as exc:
        assert "file SHA-256" in str(exc)
    else:
        raise AssertionError("manifest not bound to the release was accepted")


def test_source_faithful_step119_manifest_rejects_wrong_generation_seed_schedule():
    manifest = json.loads(PUBLIC_TASKS.read_text())
    manifest["tasks"][7]["generation_seed"] += 1
    manifest["payload_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "payload_sha256"}
    )
    try:
        validate_and_select(manifest)
    except ValueError as exc:
        assert "frozen per-slot schedule" in str(exc)
    else:
        raise AssertionError("manifest with a non-frozen per-slot generation seed was accepted")


def test_parquet_carrier_slots_bind_to_environment_sessions_without_index_confusion():
    manifest = json.loads(PUBLIC_TASKS.read_text())
    tasks = validate_and_select(manifest)
    carriers = [
        {"extra_info": {"split": "test", "index": slot}, "prompt": ["not emitted"]}
        for slot in range(128)
    ]
    bound = bind_validation_carrier_rows(carriers, tasks)

    assert [row["q03_panel_slot"] for row in bound] == list(range(128))
    assert bound[0]["extra_info"]["index"] == 0
    assert bound[0]["q03_sampled_session_index"] == 319
    assert bound[0]["q03_worker_seed"] == 1000
    assert bound[0]["q03_generation_seed"] == 2026082500
    assert bound[0]["q03_task_id"] == tasks[0]["task_id"]


def test_parquet_carrier_binding_rejects_non_test_or_reordered_rows():
    manifest = json.loads(PUBLIC_TASKS.read_text())
    tasks = validate_and_select(manifest)
    carriers = [
        {"extra_info": {"split": "test", "index": slot}}
        for slot in range(128)
    ]
    carriers[0]["extra_info"]["index"] = 319
    try:
        bind_validation_carrier_rows(carriers, tasks)
    except ValueError as exc:
        assert "carrier slot 0" in str(exc)
    else:
        raise AssertionError("session index was accepted as a parquet carrier row index")


def test_formal_custom_dataset_never_opens_train_and_enriches_only_ordered_test_carriers(tmp_path, monkeypatch):
    manifest = json.loads(PUBLIC_TASKS.read_text())
    tasks = validate_and_select(manifest)
    expected_sessions = [task["sampled_session_index"] for task in tasks]

    class FakeRandomState:
        def __init__(self, seed):
            assert seed == 1000

        def choice(self, values, size, replace):
            assert values == list(range(500)) and size == 128 and replace is False
            return expected_sessions

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.arange = lambda size: list(range(size))
    fake_numpy.random = types.SimpleNamespace(RandomState=FakeRandomState)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    class FakeRLHFDataset:
        def __init__(self, data_files, tokenizer, processor, config):
            assert data_files == [str(config["q03_step119_validation_parquet"])]
            self.dataframe = [
                {"extra_info": {"split": "test", "index": slot}, "prompt": [{"content": "unused"}]}
                for slot in range(128)
            ]

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, item):
            row = dict(self.dataframe[item])
            row["index"] = row["extra_info"]["index"]
            return row

    for name in ("verl", "verl.utils", "verl.utils.dataset"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    rl_dataset = types.ModuleType("verl.utils.dataset.rl_dataset")
    rl_dataset.RLHFDataset = FakeRLHFDataset
    monkeypatch.setitem(sys.modules, "verl.utils.dataset.rl_dataset", rl_dataset)

    module_path = ROOT / "scripts/openevo_webshop/q03_step119_validation_dataset.py"
    spec = importlib.util.spec_from_file_location("q03_step119_dataset_test", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    parquet = tmp_path / "test.parquet"
    parquet.write_bytes(b"test-only fixture")
    config = {
        "q03_step119_validation_parquet": str(parquet),
        "q03_step119_validation_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "q03_step119_task_manifest": str(PUBLIC_TASKS),
        "q03_step119_task_manifest_sha256": hashlib.sha256(PUBLIC_TASKS.read_bytes()).hexdigest(),
        "q03_step119_validation_base_seed": 1000,
        "train_batch_size": 16,
    }

    train_sentinel = module.Q03Step119ValidationDataset(
        data_files=[module.NO_TRAIN_DATASET], tokenizer=None, processor=None, config=config
    )
    # Frozen SEED's RayPPOTrainer asserts len(train_dataloader) >= 1 during
    # construction even though val_only returns before training starts.
    assert len(train_sentinel) == 16
    for index in range(len(train_sentinel)):
        try:
            train_sentinel[index]
        except IndexError as exc:
            assert "never exposes a readable row" in str(exc)
        else:
            raise AssertionError("Q03 training sentinel exposed a row")

    validation = module.Q03Step119ValidationDataset(
        data_files=[str(parquet)], tokenizer=None, processor=None, config=config
    )
    assert len(validation) == 128
    first = validation[0]
    assert first["index"] == 0
    assert first["q03_panel_slot"] == 0
    assert first["q03_sampled_session_index"] == 319
    assert first["q03_worker_seed"] == 1000
    assert first["q03_generation_seed"] == 2026082500


def test_source_overlay_passes_one_frozen_seed_per_vllm_prompt_and_step():
    import logging
    from copy import deepcopy
    from contextlib import contextmanager
    from types import SimpleNamespace

    from scripts.openevo_webshop.build_q03_step119_source_overlay import (
        patch_rollout_loop,
        patch_vllm_rollout,
    )

    rollout_source = (
        "def step(self, batch_input, gen_batch, _step):\n"
        "        batch = batch_input\n"
        "        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))\n"
        "        if True:\n"
        "            batch_input.meta_info = gen_batch.meta_info\n"
        "            return batch_input.meta_info\n"
    )
    patched_rollout = patch_rollout_loop(rollout_source)
    assert 'batch_input.non_tensor_batch[key] = values' in patched_rollout
    assert 'batch_input.meta_info["q03_step_index"] = int(_step)' in patched_rollout
    rollout_namespace = {}
    exec(compile(patched_rollout, "rollout_loop.py", "exec"), rollout_namespace)
    q03_metadata = {
        "q03_panel_slot": list(range(128)),
        "q03_task_id": [f"webshop_seed_source_faithful_slot{slot:03d}_session{slot + 100:05d}" for slot in range(128)],
        "q03_sampled_session_index": [slot + 100 for slot in range(128)],
        "q03_worker_seed": [1000 + slot for slot in range(128)],
        "q03_generation_seed": [2026082500 + slot for slot in range(128)],
        "q03_instruction_sha256": ["a" * 64 for _slot in range(128)],
    }
    env_calls = []
    envs = SimpleNamespace(reset=lambda **kwargs: (env_calls.append(kwargs) or ([], [])))
    rollout_namespace["envs"] = envs
    batch = SimpleNamespace(batch=[None] * 128, non_tensor_batch={}, meta_info={})
    gen = SimpleNamespace(non_tensor_batch=q03_metadata, meta_info={"validate": True})
    rollout_namespace["step"](SimpleNamespace(config=SimpleNamespace(trainer=SimpleNamespace(experiment_name="202610251359-q03-step119-val128"))), batch, gen, 14)
    assert batch.meta_info["q03_step_index"] == 14
    assert batch.non_tensor_batch == q03_metadata
    assert env_calls[0]["kwargs"]["q03_sampled_session_indices"] == q03_metadata["q03_sampled_session_index"]

    incomplete = SimpleNamespace(non_tensor_batch={"q03_generation_seed": [1] * 128}, meta_info={})
    try:
        rollout_namespace["step"](SimpleNamespace(config=SimpleNamespace(trainer=SimpleNamespace(experiment_name="202610251359-q03-step119-val128"))), batch, incomplete, 14)
    except ValueError as exc:
        assert "metadata" in str(exc)
    else:
        raise AssertionError("incomplete Q03 metadata reached the actor worker")

    wrong_size = SimpleNamespace(
        non_tensor_batch={**q03_metadata, "q03_generation_seed": [1]},
        meta_info={},
    )
    try:
        rollout_namespace["step"](SimpleNamespace(config=SimpleNamespace(trainer=SimpleNamespace(experiment_name="202610251359-q03-step119-val128"))), batch, wrong_size, 14)
    except ValueError as exc:
        assert "count differs" in str(exc)
    else:
        raise AssertionError("Q03 metadata with a wrong actor-batch size was accepted")

    vllm_source = '''import logging
import os
from copy import deepcopy

class Worker:
    def __init__(self):
        self.sampling_params = type("Params", (), {"seed": None, "n": 1})()
        self.inference_engine = Engine()

    @contextmanager
    def update_sampling_params(self, **kwargs):
        yield

    def generate_sequences(self, prompts, vllm_inputs, kwargs):
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=None,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
        return outputs
'''

    class Engine:
        def generate(self, prompts, sampling_params, lora_request, use_tqdm):
            self.prompts = prompts
            self.params = sampling_params
            return ["ok"] * len(prompts)

    class NumpyBool:
        pass

    namespace = {
        "contextmanager": contextmanager,
        "Engine": Engine,
        "logger": logging.getLogger("q03-source-overlay-test"),
        "np": SimpleNamespace(bool_=NumpyBool, integer=int),
        "deepcopy": deepcopy,
    }
    patched_vllm = patch_vllm_rollout(vllm_source)
    exec(compile(patched_vllm, "vllm_rollout_spmd.py", "exec"), namespace)
    worker = namespace["Worker"]()
    prompts = SimpleNamespace(
        non_tensor_batch={
            "q03_panel_slot": list(range(128)),
            "q03_task_id": [f"webshop_seed_source_faithful_slot{slot:03d}_session{slot + 100:05d}" for slot in range(128)],
            "q03_sampled_session_index": [slot + 100 for slot in range(128)],
            "q03_worker_seed": [1000 + slot for slot in range(128)],
            "q03_generation_seed": [2026082500 + slot for slot in range(128)],
            "q03_instruction_sha256": ["a" * 64 for _slot in range(128)],
        },
        meta_info={"q03_step_index": 14},
    )
    assert worker.generate_sequences(prompts, list(range(128)), {}) == ["ok"] * 128
    assert [param.seed for param in worker.inference_engine.params] == [2026082514 + slot for slot in range(128)]
    assert worker.sampling_params.seed is None

    # Sparse and reordered request batches are legal; metadata order binds each
    # sampling seed to its own slot, rather than assuming array index == slot.
    sparse_slots = [7, 2, 90]
    prompts.non_tensor_batch = {
        "q03_panel_slot": sparse_slots,
        "q03_task_id": [f"webshop_seed_source_faithful_slot{s:03d}_session{s + 100:05d}" for s in sparse_slots],
        "q03_sampled_session_index": [s + 100 for s in sparse_slots],
        "q03_worker_seed": [1000 + s for s in sparse_slots],
        "q03_generation_seed": [2026082500 + s for s in sparse_slots],
        "q03_instruction_sha256": ["a" * 64 for _s in sparse_slots],
    }
    assert worker.generate_sequences(prompts, [0, 1, 2], {}) == ["ok"] * 3
    assert [param.seed for param in worker.inference_engine.params] == [2026082514 + s for s in sparse_slots]

    prompts.non_tensor_batch["q03_panel_slot"] = [7, 7, 90]
    prompts.non_tensor_batch["q03_task_id"][1] = "webshop_seed_source_faithful_slot007_session00107"
    prompts.non_tensor_batch["q03_sampled_session_index"][1] = 107
    prompts.non_tensor_batch["q03_worker_seed"][1] = 1007
    prompts.non_tensor_batch["q03_generation_seed"][1] = 2026082507
    try:
        worker.generate_sequences(prompts, [0, 1, 2], {})
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate Q03 request slots were accepted")

    prompts.non_tensor_batch["q03_panel_slot"] = sparse_slots
    prompts.non_tensor_batch["q03_task_id"][0] = "wrong-id"
    try:
        worker.generate_sequences(prompts, [0, 1, 2], {})
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched Q03 request task ID was accepted")

    prompts.non_tensor_batch["q03_task_id"][0] = "webshop_seed_source_faithful_slot007_session00107"
    prompts.non_tensor_batch["q03_worker_seed"][0] = 1000
    try:
        worker.generate_sequences(prompts, [0, 1, 2], {})
    except ValueError as exc:
        assert "worker seed" in str(exc)
    else:
        raise AssertionError("mismatched Q03 request worker seed was accepted")

    prompts.non_tensor_batch["q03_worker_seed"][0] = 1007
    prompts.non_tensor_batch["q03_generation_seed"][0] = 2026082500
    try:
        worker.generate_sequences(prompts, [0, 1, 2], {})
    except ValueError as exc:
        assert "episode seed" in str(exc)
    else:
        raise AssertionError("mismatched Q03 request episode seed was accepted")

    prompts.non_tensor_batch["q03_generation_seed"][0] = 2026082507
    prompts.non_tensor_batch["q03_sampled_session_index"][0] = 500
    try:
        worker.generate_sequences(prompts, [0, 1, 2], {})
    except ValueError as exc:
        assert "sampled session" in str(exc)
    else:
        raise AssertionError("out-of-range Q03 session was accepted")

    prompts.non_tensor_batch["q03_panel_slot"] = list(range(128))
    prompts.non_tensor_batch["q03_task_id"] = [f"webshop_seed_source_faithful_slot{s:03d}_session{s + 100:05d}" for s in range(128)]
    prompts.non_tensor_batch["q03_sampled_session_index"] = [s + 100 for s in range(128)]
    prompts.non_tensor_batch["q03_worker_seed"] = [1000 + s for s in range(128)]
    prompts.non_tensor_batch["q03_generation_seed"] = [2026082500 + s for s in range(128)]
    prompts.non_tensor_batch["q03_instruction_sha256"] = ["a" * 64 for _s in range(128)]

    prompts.non_tensor_batch["q03_panel_slot"][0] = 7.0
    try:
        worker.generate_sequences(prompts, list(range(128)), {})
    except ValueError as exc:
        assert "integer values" in str(exc)
    else:
        raise AssertionError("floating-point panel slot was truncated and accepted")

    prompts.non_tensor_batch["q03_panel_slot"][0] = 7
    prompts.non_tensor_batch["q03_generation_seed"][0] = 2026082507.0
    try:
        worker.generate_sequences(prompts, list(range(128)), {})
    except ValueError as exc:
        assert "integer values" in str(exc)
    else:
        raise AssertionError("floating-point generation seed was truncated and accepted")

    prompts.non_tensor_batch["q03_generation_seed"][0] = 2026082507
    prompts.non_tensor_batch["q03_panel_slot"] = list(range(128))
    prompts.non_tensor_batch["q03_task_id"] = [f"webshop_seed_source_faithful_slot{s:03d}_session{s + 100:05d}" for s in range(128)]
    prompts.non_tensor_batch["q03_sampled_session_index"] = [s + 100 for s in range(128)]
    prompts.non_tensor_batch["q03_worker_seed"] = [1000 + s for s in range(128)]
    prompts.non_tensor_batch["q03_generation_seed"] = [2026082500 + s for s in range(128)]
    original_generate = worker.inference_engine.generate
    worker.inference_engine.generate = lambda **kwargs: ["short"]
    try:
        worker.generate_sequences(prompts, list(range(128)), {})
    except ValueError as exc:
        assert "response count" in str(exc)
    else:
        raise AssertionError("short vLLM response batch was accepted")
    finally:
        worker.inference_engine.generate = original_generate


def test_per_step_generation_seed_rejects_invalid_step_or_boolean_seed():
    task = {"generation_seed": 2026082500}
    for step in (-1, True):
        try:
            step_generation_seed(task, step)
        except ValueError as exc:
            assert "step index" in str(exc)
        else:
            raise AssertionError("invalid generation step index was accepted")
    try:
        step_generation_seed({"generation_seed": True}, 0)
    except ValueError as exc:
        assert "per-episode generation seed" in str(exc)
    else:
        raise AssertionError("boolean generation seed was accepted")


def test_webshop_overlay_forwards_the_manifest_sessions_to_actual_reset():
    import pytest

    from scripts.openevo_webshop.build_q03_step119_source_overlay import (
        patch_rollout_loop,
        patch_webshop_manager,
        patch_webshop_vector_env,
    )

    rollout = (
        "def run(gen_batch, envs):\n"
        "    if True:\n"
        "        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))\n"
        "        if True:\n"
        "            batch_input.meta_info = gen_batch.meta_info\n"
        "            return obs, infos\n"
    )
    patched_rollout = patch_rollout_loop(rollout)
    assert 'q03_env_kwargs["q03_sampled_session_indices"]' in patched_rollout
    assert 'q03_env_kwargs["q03_expected_instruction_sha256s"]' in patched_rollout
    assert 'q03_contract or any(q03_presence)' in patched_rollout
    assert 'if q03_contract and q03_sessions is None' in patched_rollout
    compile(patched_rollout, "rollout_loop.py", "exec")

    manager = (
        "from typing import Any, Dict\n"
        "class Manager:\n"
        "    def reset(self, kwargs) -> Dict[str, Any]:\n"
        "        obs, infos = self.envs.reset()\n"
        "        self.tasks = self.extract_task(obs)\n"
        "        return obs, infos\n"
    )
    patched_manager = patch_webshop_manager(manager)
    assert "expected_instruction_sha256s=q03_expected_hashes" in patched_manager
    assert "Q03_STEP119_INSTRUCTION_IDENTITY_RECEIPT" in patched_manager
    compile(patched_manager, "env_manager.py", "exec")

    vector_env = (
        "import ray\nimport gym\nimport numpy as np\n"
        "class WebshopWorker:\n"
        "    def __init__(self, seed, env_kwargs):\n"
        "        env_kwargs['seed'] = seed\n"
        "        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs)\n"
        "    def reset(self, idx):\n"
        "        \"\"\"Reset the environment with given session index\"\"\"\n"
        "        obs, info = self.env.reset(session=idx)\n"
        "        info = dict(info or {})\n"
        "        info['available_actions'] = self.env.get_available_actions()\n"
        "        info['won'] = False\n"
        "        return obs, info\n"
        "class VectorEnv:\n"
        "    def reset(self):\n"
        "        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)\n"
        "        idx = np.repeat(idx, self.group_n).tolist()\n"
        "        futures = []\n"
        "        for worker, session in zip(self._workers, idx):\n"
        "            futures.append(worker.reset.remote(session))\n"
        "        results = ray.get(futures)\n"
        "        obs_list, info_list = [], []\n"
        "        for obs, info in results:\n"
        "            obs_list.append(obs)\n"
        "            info_list.append(info)\n"
        "        return obs_list, info_list\n"
    )
    patched_vector = patch_webshop_vector_env(vector_env)
    assert "Q03_STEP119_WEB_SHOP_RESET_RECEIPT" in patched_vector
    assert "Q03_STEP119_INSTRUCTION_IDENTITY_RECEIPT" not in patched_vector
    assert "self.env.server.goals" in patched_vector
    assert "expected_instruction_sha256s" in patched_vector
    assert "self._rng.choice(self.goal_idxs" in patched_vector
    assert "idx.append(session)" in patched_vector
    compile(patched_vector, "webshop_envs.py", "exec")

    # Exercise both real reset branches. The pinned implementation performs
    # the group repeat after the anchor, so the default training path must
    # still repeat each sampled goal exactly group_n times.
    import sys
    import types

    class Array(list):
        def tolist(self):
            return list(self)

    fake_numpy = types.SimpleNamespace(
        integer=int,
        repeat=lambda values, count: Array(value for value in values for _ in range(count)),
    )

    fake_ray = types.SimpleNamespace(get=lambda futures: futures)
    fake_gym = types.SimpleNamespace(Env=object)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "gym", fake_gym)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    try:
        namespace = {}
        exec(compile(patched_vector, "webshop_envs.py", "exec"), namespace)
        VectorEnv = namespace["VectorEnv"]

        class RemoteReset:
            def remote(self, value):
                return value, {
                    "q03_goal_index": value,
                    "q03_worker_seed": 1000 + value,
                    "q03_instruction_sha256": "a" * 64,
                }

        class Worker:
            reset = RemoteReset()

        train_env = VectorEnv.__new__(VectorEnv)
        train_env._rng = types.SimpleNamespace(choice=lambda goals, size, replace: Array([12, 19]))
        train_env.goal_idxs = list(range(10, 30))
        train_env.env_num = 2
        train_env.group_n = 3
        train_env.is_train = True
        train_env._workers = [Worker() for _ in range(6)]
        train_sessions, train_infos = train_env.reset()
        assert len(train_sessions) == 6
        assert all(train_sessions.count(session) == 3 for session in set(train_sessions))
        assert all("q03_goal_index" in info for info in train_infos)

        validation_env = VectorEnv.__new__(VectorEnv)
        validation_env._rng = types.SimpleNamespace(choice=lambda goals, size, replace: Array([1, 2, 3]))
        validation_env.goal_idxs = list(range(500))
        validation_env.env_num = 3
        validation_env.group_n = 1
        validation_env.is_train = False
        validation_env._workers = [Worker() for _ in range(3)]
        validation_sessions, validation_infos = validation_env.reset([4, 17, 29], ["a" * 64] * 3)
        assert validation_sessions == [4, 17, 29]
        assert [info["q03_goal_index"] for info in validation_infos] == [4, 17, 29]
        with pytest.raises(ValueError, match="expected instruction hash"):
            validation_env.reset([4, 17, 29], ["bad"] * 3)
    finally:
        monkeypatch.undo()


def test_q03_webshop_worker_hashes_the_actual_loaded_goal_after_reset():
    import hashlib
    import sys
    import types
    import pytest

    from scripts.openevo_webshop.build_q03_step119_source_overlay import (
        patch_webshop_vector_env,
    )

    source = (
        "import ray\nimport gym\nimport numpy as np\n"
        "class Worker:\n"
        "    def __init__(self, seed, env_kwargs):\n"
        "        env_kwargs['seed'] = seed\n"
        "        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs)\n"
        "    def reset(self, idx):\n"
        "        \"\"\"Reset the environment with given session index\"\"\"\n"
        "        obs, info = self.env.reset(session=idx)\n"
        "        info = dict(info or {})\n"
        "        info['available_actions'] = self.env.get_available_actions()\n"
        "        info['won'] = False\n"
        "        return obs, info\n"
        "    def get_goals(self):\n"
        "        return self.env.server.goals\n"
        "class Env:\n"
        "    def reset(self):\n"
        "        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)\n"
    )
    patched = patch_webshop_vector_env(source)
    fake_ray = types.SimpleNamespace()
    fake_gym = types.SimpleNamespace(Env=object)
    fake_numpy = types.SimpleNamespace(integer=int)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "gym", fake_gym)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    try:
        namespace = {}
        exec(compile(patched, "webshop_envs.py", "exec"), namespace)
        worker = namespace["Worker"].__new__(namespace["Worker"])
        worker._q03_worker_seed = 1000
        text = "Buy the red tea cup — size 12"

        class FakeEnv:
            server = types.SimpleNamespace(goals=[{"instruction_text": text}])

            @staticmethod
            def reset(session):
                return "observation", {"source": "worker"}

            @staticmethod
            def get_available_actions():
                return ["search[tea]"]

        worker.env = FakeEnv()
        _obs, info = worker.reset(0)
        assert info["q03_goal_index"] == 0
        assert info["q03_instruction_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert info["won"] is False

        worker.env.server.goals[0]["instruction_text"] = ""
        with pytest.raises(ValueError, match="no instruction text"):
            worker.reset(0)
    finally:
        monkeypatch.undo()


def test_q03_webshop_manager_rejects_wrong_actual_instruction_hash_before_rollout():
    import pytest
    from types import SimpleNamespace

    from scripts.openevo_webshop.build_q03_step119_source_overlay import patch_webshop_manager

    source = (
        "from typing import Any, Dict\n"
        "class Manager:\n"
        "    def reset(self, kwargs) -> Dict[str, Any]:\n"
        "        obs, infos = self.envs.reset()\n"
        "        self.tasks = self.extract_task(obs)\n"
        "        return obs, infos\n"
    )
    namespace = {}
    exec(compile(patch_webshop_manager(source), "env_manager.py", "exec"), namespace)

    class Envs:
        env_num = 128
        group_n = 1
        is_train = False

        def reset(self, sampled_session_indices, expected_instruction_sha256s):
            return ["obs"], [{
                "q03_goal_index": sampled_session_indices[0],
                "q03_instruction_sha256": "b" * 64,
            }]

    manager = namespace["Manager"]()
    manager.envs = Envs()
    manager.config = SimpleNamespace(
        env=SimpleNamespace(seed=0, webshop=SimpleNamespace(use_small=True, human_goals=False)),
        data=SimpleNamespace(val_batch_size=128),
        trainer=SimpleNamespace(experiment_name="202610251359-q03-step119-val128", val_only=True, val_before_train=True, test_freq=-1),
    )
    manager.extract_task = lambda obs: ["task"]
    kwargs = {
        "q03_panel_slots": [0],
        "q03_task_ids": ["webshop_seed_source_faithful_slot000_session00004"],
        "q03_worker_seeds": [1000],
        "q03_sampled_session_indices": [4],
        "q03_expected_instruction_sha256s": ["a" * 64],
    }
    with pytest.raises(ValueError, match="actual WebShop goal/instruction"):
        manager.reset(kwargs)

    with pytest.raises(ValueError, match="validation metadata is missing"):
        manager.reset({})


def test_overlay_builder_serializes_same_hash_publication_and_rejects_symlink(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import os
    import stat
    import pytest

    import scripts.openevo_webshop.build_q03_step119_source_overlay as builder

    seed_root = tmp_path / "seed"
    output_root = tmp_path / "out"
    sources = {
        builder.ACTOR_PATH: (
            "if is_cuda_available:\n"
            "    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n"
            "elif is_npu_available:\n"
            "    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input\n"
            "\n"
            "class DataParallelPPOActor:\n"
            "    def _forward_micro_batch(self, micro_batch):\n"
            "        if self.use_remove_padding:\n"
            "            return unpad_input(micro_batch)\n"
            "        return micro_batch\n"
        ),
        "agent_system/multi_turn_rollout/rollout_loop.py": (
            "def run(gen_batch, envs, batch_input):\n"
            "    if True:\n"
            "        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))\n"
            "        if True:\n"
            "            batch_input.meta_info = gen_batch.meta_info\n"
    ),
        "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py": (
            "import logging\nimport os\nfrom copy import deepcopy\n"
            "def run(self, prompts, vllm_inputs, kwargs):\n"
            "        with self.update_sampling_params(**kwargs):\n"
            "            outputs = self.inference_engine.generate(\n"
            "                prompts=vllm_inputs,\n"
            "                sampling_params=self.sampling_params,\n"
            "                lora_request=None,\n"
            "                use_tqdm=False,\n"
            "            )\n\n            # TODO(sgm):\n"
    ),
        "verl/utils/vllm_utils.py": "from vllm.lora.models import LoRAModel\n",
        "scripts/model_merger.py": (
            "from transformers import (\n    AutoModelForCausalLM,\n    AutoModelForVision2Seq,\n)\n"
            "class Merger:\n"
            "    def get_transformers_auto_model_class(self):\n"
            "        if 'ForCausalLM' in self.model_config.architectures[0]:\n"
            "            return AutoModelForCausalLM\n"
            "        elif \"ForConditionalGeneration\" in self.model_config.architectures[0]:\n"
            "            return AutoModelForVision2Seq\n"
    ),
        "agent_system/environments/env_manager.py": (
            "from typing import Any, Dict\n"
            "class Manager:\n"
            "    def reset(self, kwargs) -> Dict[str, Any]:\n"
            "        obs, infos = self.envs.reset()\n"
            "        self.tasks = self.extract_task(obs)\n"
    ),
        "agent_system/environments/env_package/webshop/envs.py": (
            "import ray\nimport gym\nimport numpy as np\n"
            "class WebshopWorker:\n"
            "    def __init__(self, seed, env_kwargs):\n"
            "        env_kwargs['seed'] = seed\n"
            "        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs)\n"
            "    def reset(self, idx):\n"
            "        \"\"\"Reset the environment with given session index\"\"\"\n"
            "        obs, info = self.env.reset(session=idx)\n"
            "        info = dict(info or {})\n"
            "        info['available_actions'] = self.env.get_available_actions()\n"
            "        info['won'] = False\n"
            "        return obs, info\n"
            "class VectorEnv:\n"
            "    def reset(self):\n"
            "        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)\n"
    ),
        "verl/trainer/ppo/ray_trainer.py": (
            "class Trainer:\n"
            "    def _validate(self):\n"
            "        # Lists to collect samples for the table\n"
            "        sample_inputs = []\n"
            "        for test_data in self.val_dataloader:\n"
            "            result = self.val_reward_fn(test_batch, return_dict=True)\n"
            "        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}\n"
            "        return success_rate\n"
    ),
    }
    expected_commit = "a" * 40
    for relative, body in sources.items():
        path = seed_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    monkeypatch.setattr(builder, "EXPECTED_SEED_COMMIT", expected_commit)
    monkeypatch.setattr(builder, "SOURCE_FILES", {
        relative: hashlib.sha256(body.encode()).hexdigest() for relative, body in sources.items()
    })
    monkeypatch.setattr(builder, "_git", lambda _root, *args: expected_commit if args == ("rev-parse", "HEAD") else "")
    # The test host may be macOS; production Linux syscall behavior is checked
    # separately, while this verifies the builder's serialized transaction.
    monkeypatch.setattr(builder, "_rename_noreplace", lambda source, target: os.rename(source, target))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: builder.build_overlay(seed_root, output_root), range(4)))
    assert len({result["path"] for result in results}) == 1
    bundle = Path(results[0]["path"])
    assert stat.S_IMODE(bundle.stat().st_mode) & 0o222 == 0
    actor_overlay = (bundle / builder.ACTOR_PATH).read_text()
    assert actor_overlay.index("if self.use_remove_padding") < actor_overlay.index("from flash_attn.bert_padding")
    vllm_overlay = (bundle / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py").read_text()
    lora_util_overlay = (bundle / "verl/utils/vllm_utils.py").read_text()
    assert "from vllm.lora.lora_model import LoRAModel" in lora_util_overlay
    assert "from vllm.lora.models import LoRAModel" not in lora_util_overlay
    merger_overlay = (bundle / "scripts/model_merger.py").read_text()
    assert "AutoModelForVision2Seq," not in merger_overlay
    assert "from transformers import AutoModelForVision2Seq" in merger_overlay
    ray_overlay = (bundle / "verl/trainer/ppo/ray_trainer.py").read_text()
    assert "collect_batch_rows(test_output_gen_batch.non_tensor_batch)" in ray_overlay
    assert "write_document(Path(str(self.config.trainer.default_local_dir)), result_document)" in ray_overlay
    assert (bundle / "verl/trainer/ppo/q03_step119_validation_results.py").is_file()
    assert (bundle / "sitecustomize.py").is_file()
    rollout_overlay = (bundle / "agent_system/multi_turn_rollout/rollout_loop.py").read_text()
    assert "batch.non_tensor_batch[key] = values" in rollout_overlay

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_root = tmp_path / "symlink-out"
    symlink_root.mkdir()
    (symlink_root / bundle.name).symlink_to(outside, target_is_directory=True)
    with pytest.raises(builder.OverlayError, match="real directory"):
        builder.build_overlay(seed_root, symlink_root)

    raced_root = tmp_path / "raced-out"

    def publish_after_racing_creator(source, target):
        target.mkdir()
        (target / "race-marker").write_text("preserve")
        raise FileExistsError("content-addressed destination appeared before publication")

    monkeypatch.setattr(builder, "_rename_noreplace", publish_after_racing_creator)
    with pytest.raises(FileExistsError):
        builder.build_overlay(seed_root, raced_root)
    raced_target = raced_root / bundle.name
    assert (raced_target / "race-marker").read_text() == "preserve"
    assert not list(raced_root.glob(".q03-step119-overlay-*"))


def test_rename_noreplace_linux_syscall_refuses_existing_target(tmp_path):
    """Exercise the real Linux syscall used to publish an overlay bundle."""
    import pytest

    if not sys.platform.startswith("linux"):
        pytest.skip("renameat2 is a Linux publication primitive")

    from scripts.openevo_webshop import build_q03_step119_source_overlay as builder

    source = tmp_path / "staging"
    target = tmp_path / "published"
    source.mkdir()
    (source / "payload").write_text("source")
    target.mkdir()
    (target / "race-marker").write_text("preserve")

    with pytest.raises(FileExistsError):
        builder._rename_noreplace(source, target)

    assert (source / "payload").read_text() == "source"
    assert (target / "race-marker").read_text() == "preserve"

    absent_target = tmp_path / "published-absent"
    builder._rename_noreplace(source, absent_target)
    assert not source.exists()
    assert (absent_target / "payload").read_text() == "source"


def test_overlay_bootstrap_shadows_module_inside_upstream_regular_package(tmp_path):
    """A regular upstream package must not make Python bypass patched modules."""
    import os
    import subprocess
    import sys

    import scripts.openevo_webshop.build_q03_step119_source_overlay as builder

    upstream = tmp_path / "upstream"
    overlay = tmp_path / "overlay"
    package = "verl.workers.actor"
    relative = "verl/workers/actor/dp_actor.py"
    (upstream / "verl/workers/actor").mkdir(parents=True)
    (overlay / "verl/workers/actor").mkdir(parents=True)
    # These __init__.py files model SEED's regular packages, whose __path__ does
    # not include the sparse overlay directory.
    for directory in (upstream / "verl", upstream / "verl/workers", upstream / package.replace(".", "/")):
        (directory / "__init__.py").write_text("PACKAGE = True\n")
    (upstream / relative).write_text("ORIGIN = 'upstream'\n")
    (overlay / relative).write_text("ORIGIN = 'sealed-overlay'\n")
    (overlay / "sitecustomize.py").write_bytes(
        builder._overlay_import_bootstrap({relative: b"sealed-overlay"})
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(overlay), str(upstream)))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", "import verl.workers.actor.dp_actor as m; print(m.ORIGIN); print(m.__file__)"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert lines[0] == "sealed-overlay"
    assert Path(lines[1]).resolve() == (overlay / relative).resolve()


def test_pinned_seed_trainer_real_fit_returns_after_initial_validation_without_training():
    """Execute the exact val-before-train prefix from the pinned real trainer."""
    import ast
    import hashlib
    import subprocess
    from unittest.mock import patch

    # Extracted verbatim from jinyangwu/SEED@2cf2fadca3c5aba28da68e8e1405182ba8d90e6c,
    # verl/trainer/ppo/ray_trainer.py, RayPPOTrainer.fit, through the first training action.
    # The complete pinned source SHA is also recorded by the Step119 runtime contract.
    source = '''def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
'''
    expected = "62ee05e175f535081ca43f9f079cddc2dc7fbc652f357ef1c0b1928cd3931c11"
    assert hashlib.sha256(source.encode()).hexdigest() == expected
    tree = ast.parse(source)
    function = tree.body[0]
    namespace = {"pprint": lambda *_: None, "tqdm": lambda **_: (_ for _ in ()).throw(AssertionError("training started"))}
    exec(compile(tree, "pinned_ray_trainer.py", "exec"), namespace)

    events = []
    class AttrConfig(dict):
        __getattr__ = dict.__getitem__
    class Logger:
        def __init__(self, **kwargs):
            events.append(("logger", kwargs))
        def log(self, *, data, step):
            events.append(("log", data, step))
    class FakeSelf:
        def __init__(self):
            self.config = AttrConfig(trainer=AttrConfig(
                project_name="q03-step119-test", experiment_name="validation-only",
                logger=["console"], val_before_train=True, val_only=True))
            self.val_reward_fn = object()
            self.total_training_steps = 1
        def _load_checkpoint(self):
            events.append(("load_checkpoint",))
        def _validate(self):
            events.append(("validate",))
            return {"validation/cases": 128}

    omega = types.ModuleType("omegaconf")
    omega.OmegaConf = types.SimpleNamespace(to_container=lambda config, resolve: {"resolved": resolve})
    verl = types.ModuleType("verl")
    utils = types.ModuleType("verl.utils")
    tracking = types.ModuleType("verl.utils.tracking")
    tracking.Tracking = Logger
    names = ("omegaconf", "verl", "verl.utils", "verl.utils.tracking")
    old = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules.update({"omegaconf": omega, "verl": verl, "verl.utils": utils, "verl.utils.tracking": tracking})
        namespace["fit"](FakeSelf())
    finally:
        for name, module in old.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    assert [event[0] for event in events] == ["logger", "load_checkpoint", "validate", "log"]
    assert events[-1][1:] == ({"validation/cases": 128}, 0)
