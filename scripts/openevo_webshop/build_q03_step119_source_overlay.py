"""Build immutable, hash-bound SEED runtime overlays for Step119 validation."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from .q03_step119_actor_import_compat import ACTOR_PATH, ACTOR_SHA256, patch_padding_imports
except ImportError:  # Direct script execution from scripts/openevo_webshop.
    from q03_step119_actor_import_compat import ACTOR_PATH, ACTOR_SHA256, patch_padding_imports


EXPECTED_SEED_COMMIT = "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c"
Q03_STEP119_EXPERIMENT_NAME = "202610251359-q03-step119-val128"
SOURCE_FILES = {
    ACTOR_PATH: ACTOR_SHA256,
    "agent_system/multi_turn_rollout/rollout_loop.py": "a917f733e7a265dfd04fc82526e458b9ceb55858bf7e34c3713c9159941336ae",
    "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py": "6fc07d5a6ddcc087d26d4fe24b9d2c219a3c93978049dc93f2efe6d683137dfc",
    "verl/utils/vllm_utils.py": "764617a3699bb4eff4e5bb31afbb3fc35afdb9247e13e6a5773b44715dd87482",
    "scripts/model_merger.py": "ad50e027112ad9c5d067764e05a5eda6ef644041a099b68dee60670a7a6f2cff",
    "agent_system/environments/env_manager.py": "73d8a3580456a96403bfe5c2897dffd5828d961e7a5e5a69f780d14ac00c4f3e",
    "agent_system/environments/env_package/webshop/envs.py": "1a8a73b5e02854a822aa03dcb66a9fbf8d67c028f963f36845bf34fb1b60ea7c",
    "verl/trainer/ppo/ray_trainer.py": "cf48fc1f96b3f80cbe0de2d80672fc8afa047bed1d01b44d564cc5ada9067b75",
}
VALIDATION_SOURCE_FILES = {
    "q03_step119_validation_dataset.py": (
        "verl/utils/dataset/q03_step119_validation_dataset.py",
        "3ac6ae9336cb477168be79951b579ecad589869db68dd5d7dd56fa385a9ad954",
    ),
    "q03_source_faithful_manifest.py": (
        "verl/utils/dataset/q03_source_faithful_manifest.py",
        "b508f4ccd9e384bfbab537b4c5a447628a519885397bcdf37195d60d643975bc",
    ),
}
ROLLOUT_ANCHOR = "            batch_input.meta_info = gen_batch.meta_info\n"
GATHER_ANCHOR = "        success_rate = {}\n        for key, value in success.items():\n            success_rate[key] = np.mean(value)\n"
GATHER_ROW_ANCHOR = "                    data['tool_callings'] = tool_callings[bs]\n"
VALIDATION_ANCHOR = "        # Lists to collect samples for the table\n"
VALIDATION_REWARD_ANCHOR = "            result = self.val_reward_fn(test_batch, return_dict=True)\n"
VALIDATION_METRICS_ANCHOR = "        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}\n"
VLLM_ANCHOR = "        with self.update_sampling_params(**kwargs):\n            outputs = self.inference_engine.generate(\n"
ENV_RESET_ANCHOR = "        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))\n"
WEBSHOP_MANAGER_RESET_ANCHOR = (
    "    def reset(self, kwargs) -> Dict[str, Any]:\n"
    "        obs, infos = self.envs.reset()\n"
    "        self.tasks = self.extract_task(obs)\n"
)
WEBSHOP_VECTOR_RESET_ANCHOR = "    def reset(self):\n        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)\n"
WEBSHOP_WORKER_RESET_ANCHOR = (
    "    def reset(self, idx):\n"
    "        \"\"\"Reset the environment with given session index\"\"\"\n"
    "        obs, info = self.env.reset(session=idx)\n"
    "        info = dict(info or {})\n"
    "        info['available_actions'] = self.env.get_available_actions()\n"
    "        info['won'] = False\n"
    "        return obs, info\n"
)
WEBSHOP_WORKER_SEED_ANCHOR = (
    "        env_kwargs['seed'] = seed\n"
    "        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs)\n"
)


class OverlayError(RuntimeError):
    pass


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish only when the content-addressed destination is unused."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OverlayError("atomic no-replace directory publication is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), str(target))
        raise OSError(error_number, os.strerror(error_number), str(target))


def _remove_staging_tree(path: Path) -> None:
    """Remove this builder's read-only staging tree after a failed publish."""
    import shutil

    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(path_stat.st_mode):
        raise OverlayError("private staging path changed type; refusing cleanup")
    for directory, subdirectories, files in os.walk(path, topdown=True, followlinks=False):
        for name in files:
            child = Path(directory) / name
            if not child.is_symlink():
                child.chmod(0o600)
        for name in subdirectories:
            child = Path(directory) / name
            if not child.is_symlink():
                child.chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def patch_rollout_loop(source: str) -> str:
    if source.count(ROLLOUT_ANCHOR) != 1:
        raise OverlayError("pinned rollout-loop call anchor is missing or ambiguous")
    injected = ROLLOUT_ANCHOR + (
        "            q03_contract = getattr(getattr(self.config, \"trainer\", None), \"experiment_name\", None) == \"202610251359-q03-step119-val128\"\n"
        "            q03_keys = (\"q03_panel_slot\", \"q03_task_id\", \"q03_sampled_session_index\", \"q03_worker_seed\", \"q03_generation_seed\", \"q03_instruction_sha256\")\n"
        "            q03_presence = [key in gen_batch.non_tensor_batch for key in q03_keys]\n"
        "            if q03_contract or any(q03_presence):\n"
        "                if not all(q03_presence):\n"
        "                    raise ValueError(\"Q03 Step119 rollout metadata is incomplete\")\n"
        "                for key in q03_keys:\n"
        "                    values = gen_batch.non_tensor_batch[key]\n"
        "                    if len(values) != len(batch_input.batch):\n"
        "                        raise ValueError(\"Q03 Step119 rollout metadata count differs from actor batch\")\n"
        "                    batch_input.non_tensor_batch[key] = values\n"
        "                    batch.non_tensor_batch[key] = values\n"
        "                batch_input.meta_info[\"q03_step_index\"] = int(_step)\n"
    )
    result = source
    gather_counts = (source.count(GATHER_ANCHOR), source.count(GATHER_ROW_ANCHOR))
    if gather_counts not in ((0, 0), (1, 1)):
        raise OverlayError("pinned rollout gather anchors are missing or ambiguous")
    if gather_counts == (1, 1):
        gather_guard = GATHER_ANCHOR + (
            "        q03_success = success.get(\"success_rate\")\n"
            "        q03_task_scores = success.get(\"webshop_task_score (not success_rate)\")\n"
            "        q03_has_identity = any(\"q03_task_id\" in row for trajectory in total_batch_list for row in trajectory)\n"
            "        if q03_has_identity and (q03_success is None or q03_task_scores is None or len(q03_success) != batch_size or len(q03_task_scores) != batch_size):\n"
            "            raise ValueError(\"Q03 Step119 per-task result count differs from rollout batch\")\n"
        )
        result = source.replace(GATHER_ANCHOR, gather_guard, 1)
        row_result = GATHER_ROW_ANCHOR + (
            "                    if q03_has_identity:\n"
            "                        q03_keys = (\"q03_panel_slot\", \"q03_task_id\", \"q03_sampled_session_index\", \"q03_worker_seed\", \"q03_generation_seed\", \"q03_instruction_sha256\")\n"
            "                        if not all(key in data for key in q03_keys):\n"
            "                            raise ValueError(\"Q03 Step119 rollout output lost task identity\")\n"
            "                        data[\"q03_validation_task_score\"] = float(q03_task_scores[bs])\n"
            "                        data[\"q03_validation_exact_success\"] = bool(float(q03_success[bs]) == 1.0)\n"
        )
        result = result.replace(GATHER_ROW_ANCHOR, row_result, 1)
    result = result.replace(ROLLOUT_ANCHOR, injected, 1)
    if result.count(ENV_RESET_ANCHOR) != 1:
        raise OverlayError("pinned WebShop reset call anchor is missing or ambiguous")
    replacement = (
        "        q03_contract = getattr(getattr(self.config, \"trainer\", None), \"experiment_name\", None) == \"202610251359-q03-step119-val128\"\n"
        "        q03_sessions = gen_batch.non_tensor_batch.get(\"q03_sampled_session_index\")\n"
        "        if q03_contract and q03_sessions is None:\n"
        "            raise ValueError(\"Q03 Step119 validation metadata is missing\")\n"
        "        q03_env_kwargs = gen_batch.non_tensor_batch.pop('env_kwargs', None)\n"
        "        if q03_sessions is not None:\n"
        "            q03_env_kwargs = dict(q03_env_kwargs or {})\n"
        "            q03_env_kwargs[\"q03_sampled_session_indices\"] = [int(value) for value in q03_sessions]\n"
        "            q03_env_kwargs[\"q03_panel_slots\"] = [int(value) for value in gen_batch.non_tensor_batch[\"q03_panel_slot\"]]\n"
        "            q03_env_kwargs[\"q03_task_ids\"] = [str(value) for value in gen_batch.non_tensor_batch[\"q03_task_id\"]]\n"
        "            q03_env_kwargs[\"q03_worker_seeds\"] = [int(value) for value in gen_batch.non_tensor_batch[\"q03_worker_seed\"]]\n"
        "            q03_env_kwargs[\"q03_expected_instruction_sha256s\"] = [str(value) for value in gen_batch.non_tensor_batch[\"q03_instruction_sha256\"]]\n"
        "        obs, infos = envs.reset(kwargs=q03_env_kwargs)\n"
    )
    return result.replace(ENV_RESET_ANCHOR, replacement, 1)


def patch_ray_trainer_validation_results(source: str) -> str:
    if (source.count(VALIDATION_ANCHOR) != 1
            or source.count(VALIDATION_REWARD_ANCHOR) != 1
            or source.count(VALIDATION_METRICS_ANCHOR) != 1):
        raise OverlayError("pinned validation result anchors are missing or ambiguous")
    result = source.replace(
        VALIDATION_ANCHOR,
        VALIDATION_ANCHOR
        + "        q03_step119_result_rows = []\n"
        + "        q03_step119_enabled = self.config.trainer.get(\"experiment_name\") == \"202610251359-q03-step119-val128\"\n",
        1,
    )
    result = result.replace(
        VALIDATION_REWARD_ANCHOR,
        VALIDATION_REWARD_ANCHOR
        + "            if q03_step119_enabled:\n"
        + "                from verl.trainer.ppo.q03_step119_validation_results import collect_batch_rows\n"
        + "                q03_step119_result_rows.extend(collect_batch_rows(test_output_gen_batch.non_tensor_batch))\n",
        1,
    )
    result = result.replace(
        VALIDATION_METRICS_ANCHOR,
        VALIDATION_METRICS_ANCHOR
        + "        if q03_step119_enabled:\n"
        + "            from pathlib import Path\n"
        + "            from verl.trainer.ppo.q03_step119_validation_results import build_document, write_document\n"
        + "            result_document = build_document(\n"
        + "                q03_step119_result_rows,\n"
        + "                task_manifest_sha256=str(self.config.data.q03_step119_task_manifest_sha256),\n"
        + "                parquet_sha256=str(self.config.data.q03_step119_validation_parquet_sha256),\n"
        + "                experiment_id=str(self.config.trainer.experiment_name),\n"
        + "            )\n"
        + "            result_path = write_document(Path(str(self.config.trainer.default_local_dir)), result_document)\n"
        + "            print(\"Q03_STEP119_VALIDATION_RESULTS_RECEIPT \" + str(result_path))\n",
        1,
    )
    return result


def patch_webshop_manager(source: str) -> str:
    if source.count(WEBSHOP_MANAGER_RESET_ANCHOR) != 1:
        raise OverlayError("pinned WebShop manager reset anchor is missing or ambiguous")
    replacement = (
        "    def reset(self, kwargs) -> Dict[str, Any]:\n"
        "        q03_contract = getattr(getattr(self.config, \"trainer\", None), \"experiment_name\", None) == \"202610251359-q03-step119-val128\"\n"
        "        q03_sessions = (kwargs or {}).get(\"q03_sampled_session_indices\")\n"
        "        if q03_contract and q03_sessions is None:\n"
        "            raise ValueError(\"Q03 Step119 validation metadata is missing\")\n"
        "        if q03_sessions is None:\n"
        "            obs, infos = self.envs.reset()\n"
        "        else:\n"
        "            config = self.config\n"
        "            try:\n"
        "                env_seed = int(config.env.seed)\n"
        "                val_batch_size = int(config.data.val_batch_size)\n"
        "                use_small = config.env.webshop.use_small\n"
        "                human_goals = config.env.webshop.human_goals\n"
        "                val_only = config.trainer.val_only\n"
        "                val_before_train = config.trainer.val_before_train\n"
        "                test_freq = int(config.trainer.test_freq)\n"
        "            except (AttributeError, TypeError, ValueError) as exc:\n"
        "                raise ValueError(\"Q03 Step119 effective runtime configuration is incomplete\") from exc\n"
        "            if (env_seed != 0 or val_batch_size != 128 or use_small is not True\n"
        "                    or human_goals not in (False, 0) or val_only is not True\n"
        "                    or val_before_train is not True or test_freq != -1\n"
        "                    or self.envs.env_num != 128 or self.envs.group_n != 1 or self.envs.is_train is not False):\n"
        "                raise ValueError(\"Q03 Step119 effective validation configuration differs from frozen contract\")\n"
        "            q03_slots = (kwargs or {}).get(\"q03_panel_slots\")\n"
        "            q03_task_ids = (kwargs or {}).get(\"q03_task_ids\")\n"
        "            q03_worker_seeds = (kwargs or {}).get(\"q03_worker_seeds\")\n"
        "            q03_expected_hashes = (kwargs or {}).get(\"q03_expected_instruction_sha256s\")\n"
        "            if not (len(q03_sessions) == len(q03_slots or []) == len(q03_task_ids or []) == len(q03_worker_seeds or []) == len(q03_expected_hashes or [])):\n"
        "                raise ValueError(\"Q03 Step119 reset identity metadata counts differ\")\n"
        "            obs, infos = self.envs.reset(sampled_session_indices=q03_sessions, expected_instruction_sha256s=q03_expected_hashes)\n"
        "            if len(infos) != len(q03_sessions):\n"
        "                raise ValueError(\"Q03 Step119 worker reset receipt count differs from panel\")\n"
        "            q03_receipts = []\n"
        "            for slot, task_id, worker_seed, session, expected_hash, info in zip(q03_slots, q03_task_ids, q03_worker_seeds, q03_sessions, q03_expected_hashes, infos, strict=True):\n"
        "                expected_worker_seed = 1000 + int(slot)\n"
        "                if worker_seed != expected_worker_seed or not isinstance(info, dict) or info.get(\"q03_goal_index\") != session or info.get(\"q03_worker_seed\") != worker_seed or info.get(\"q03_instruction_sha256\") != expected_hash:\n"
        "                    raise ValueError(\"Q03 Step119 actual WebShop goal/instruction differs from frozen manifest\")\n"
        "                expected_task_id = f\"webshop_seed_source_faithful_slot{int(slot):03d}_session{int(session):05d}\"\n"
        "                if task_id != expected_task_id:\n"
        "                    raise ValueError(\"Q03 Step119 reset slot/task/session binding differs from manifest\")\n"
        "                q03_receipts.append({\"panel_slot\": int(slot), \"task_id\": str(task_id), \"worker_index\": int(slot), \"worker_seed\": int(info[\"q03_worker_seed\"]), \"sampled_session_index\": int(session), \"instruction_sha256\": str(expected_hash)})\n"
        "            if len({row[\"panel_slot\"] for row in q03_receipts}) != len(q03_receipts) or len({row[\"sampled_session_index\"] for row in q03_receipts}) != len(q03_receipts):\n"
        "                raise ValueError(\"Q03 Step119 reset slots and sessions must be unique\")\n"
        "            import json\n"
        "            print(\"Q03_STEP119_EFFECTIVE_CONFIG_RECEIPT \" + json.dumps({\"env_seed\": env_seed, \"val_batch_size\": val_batch_size, \"group_n\": int(self.envs.group_n), \"use_small\": bool(use_small), \"human_goals\": bool(human_goals), \"val_only\": bool(val_only), \"val_before_train\": bool(val_before_train), \"test_freq\": test_freq, \"worker_seed_formula\": \"env.seed + 1000 + slot\"}, sort_keys=True))\n"
        "            print(\"Q03_STEP119_INSTRUCTION_IDENTITY_RECEIPT \" + json.dumps(q03_receipts, sort_keys=True))\n"
        "        self.tasks = self.extract_task(obs)\n"
    )
    return source.replace(WEBSHOP_MANAGER_RESET_ANCHOR, replacement, 1)


def patch_webshop_vector_env(source: str) -> str:
    if (source.count("import ray\nimport gym\nimport numpy as np\n") != 1
            or source.count(WEBSHOP_VECTOR_RESET_ANCHOR) != 1
            or source.count(WEBSHOP_WORKER_RESET_ANCHOR) != 1
            or source.count(WEBSHOP_WORKER_SEED_ANCHOR) != 1):
        raise OverlayError("pinned WebShop vector reset anchors are missing or ambiguous")
    result = source.replace(
        "import ray\nimport gym\nimport numpy as np\n",
        "import hashlib\nimport json\nimport ray\nimport gym\nimport numpy as np\n",
        1,
    )
    result = result.replace(
        WEBSHOP_WORKER_SEED_ANCHOR,
        "        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):\n"
        "            raise ValueError(\"Q03 worker constructor seed must be an integer\")\n"
        "        self._q03_worker_seed = int(seed)\n"
        + WEBSHOP_WORKER_SEED_ANCHOR,
        1,
    )
    worker_replacement = (
        "    def reset(self, idx):\n"
        "        \"\"\"Reset the environment with given session index.\"\"\"\n"
        "        obs, info = self.env.reset(session=idx)\n"
        "        info = dict(info or {})\n"
        "        info['available_actions'] = self.env.get_available_actions()\n"
        "        info['won'] = False\n"
        "        info[\"q03_worker_seed\"] = int(self._q03_worker_seed)\n"
        "        goals = self.env.server.goals\n"
        "        if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)) or not 0 <= int(idx) < len(goals):\n"
        "            raise ValueError(\"Q03 Step119 worker reset index is invalid\")\n"
        "        instruction = goals[int(idx)].get(\"instruction_text\")\n"
        "        if not isinstance(instruction, str) or not instruction:\n"
        "            raise ValueError(\"Q03 Step119 actual WebShop goal has no instruction text\")\n"
        "        info[\"q03_goal_index\"] = int(idx)\n"
        "        info[\"q03_instruction_sha256\"] = hashlib.sha256(instruction.encode(\"utf-8\")).hexdigest()\n"
        "        return obs, info\n"
    )
    result = result.replace(WEBSHOP_WORKER_RESET_ANCHOR, worker_replacement, 1)
    replacement = (
        "    def reset(self, sampled_session_indices=None, expected_instruction_sha256s=None):\n"
        "        if sampled_session_indices is None:\n"
        "            if expected_instruction_sha256s is not None:\n"
        "                raise ValueError(\"Q03 instruction expectations require explicit validation sessions\")\n"
        "            idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)\n"
        "        else:\n"
        "            if self.is_train or self.group_n != 1:\n"
        "                raise ValueError(\"Q03 explicit sessions are allowed only for ungrouped validation environments\")\n"
        "            if len(sampled_session_indices) != self.env_num:\n"
        "                raise ValueError(\"Q03 explicit validation session count differs from environment count\")\n"
        "            idx = []\n"
        "            for session in sampled_session_indices:\n"
        "                if isinstance(session, bool) or not isinstance(session, (int, np.integer)):\n"
        "                    raise ValueError(\"Q03 explicit validation sessions must be integer values\")\n"
        "                session = int(session)\n"
        "                if session not in self.goal_idxs:\n"
        "                    raise ValueError(\"Q03 explicit validation session is outside the public held-out split\")\n"
        "                idx.append(session)\n"
        "            if len(set(idx)) != len(idx):\n"
        "                raise ValueError(\"Q03 explicit validation sessions must be unique\")\n"
        "            if not isinstance(expected_instruction_sha256s, (list, tuple)) or len(expected_instruction_sha256s) != self.env_num:\n"
        "                raise ValueError(\"Q03 expected instruction hash count differs from validation environment count\")\n"
        "            for expected_hash in expected_instruction_sha256s:\n"
        "                if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(char not in \"0123456789abcdef\" for char in expected_hash):\n"
        "                    raise ValueError(\"Q03 expected instruction hash is malformed\")\n"
        "            print(\"Q03_STEP119_WEB_SHOP_RESET_RECEIPT \" + json.dumps({\"sessions\": idx, \"expected_instruction_sha256s\": list(expected_instruction_sha256s)}, sort_keys=True))\n"
    )
    return result.replace(WEBSHOP_VECTOR_RESET_ANCHOR, replacement, 1)


def patch_vllm_rollout(source: str) -> str:
    if (
        source.count("import logging\nimport os\n") != 1
        or source.count(VLLM_ANCHOR) != 1
        or source.count("sampling_params=self.sampling_params,") != 1
        or source.count("                use_tqdm=False,\n            )\n\n            # TODO(sgm):") != 1
    ):
        raise OverlayError("pinned vLLM import/call anchors are missing or ambiguous")
    result = source.replace("import logging\nimport os\n", "import json\nimport logging\nimport os\n", 1)
    replacement = '''        with self.update_sampling_params(**kwargs):
            request_sampling_params = self.sampling_params
            q03_request_receipt = None
            q03_keys = ("q03_panel_slot", "q03_task_id", "q03_sampled_session_index", "q03_worker_seed", "q03_generation_seed", "q03_instruction_sha256")
            present_q03_keys = [key in prompts.non_tensor_batch for key in q03_keys]
            if any(present_q03_keys) and not all(present_q03_keys):
                raise ValueError("Q03 Step119 request metadata is incomplete")
            if all(present_q03_keys):
                step_index = prompts.meta_info.get("q03_step_index")
                if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
                    raise ValueError("Q03 Step119 request lacks a valid rollout step index")
                slots = prompts.non_tensor_batch["q03_panel_slot"]
                task_ids = prompts.non_tensor_batch["q03_task_id"]
                sessions = prompts.non_tensor_batch["q03_sampled_session_index"]
                worker_seeds = prompts.non_tensor_batch["q03_worker_seed"]
                episode_seeds = prompts.non_tensor_batch["q03_generation_seed"]
                instruction_hashes = prompts.non_tensor_batch["q03_instruction_sha256"]
                if not (len(slots) == len(task_ids) == len(sessions) == len(worker_seeds) == len(episode_seeds) == len(instruction_hashes) == len(vllm_inputs)):
                    raise ValueError("Q03 Step119 metadata count differs from vLLM prompt count")
                numeric_slots = []
                request_sampling_params = []
                request_receipt = []
                for slot, task_id, session, worker_seed, episode_seed, instruction_hash in zip(
                    slots, task_ids, sessions, worker_seeds, episode_seeds, instruction_hashes, strict=True
                ):
                    if (
                        isinstance(slot, (bool, np.bool_))
                        or not isinstance(slot, (int, np.integer))
                        or isinstance(session, (bool, np.bool_))
                        or not isinstance(session, (int, np.integer))
                        or isinstance(worker_seed, (bool, np.bool_))
                        or not isinstance(worker_seed, (int, np.integer))
                        or isinstance(episode_seed, (bool, np.bool_))
                        or not isinstance(episode_seed, (int, np.integer))
                    ):
                        raise ValueError("Q03 Step119 slot, session, and seeds must be integer values")
                    slot = int(slot)
                    session = int(session)
                    worker_seed = int(worker_seed)
                    episode_seed = int(episode_seed)
                    if not 0 <= slot < 128:
                        raise ValueError("Q03 Step119 task slot is outside 0..127")
                    if not 0 <= session < 500:
                        raise ValueError("Q03 Step119 sampled session is outside 0..499")
                    if task_id != f"webshop_seed_source_faithful_slot{slot:03d}_session{session:05d}":
                        raise ValueError("Q03 Step119 task ID does not match its slot and sampled session")
                    if worker_seed != 1000 + slot:
                        raise ValueError("Q03 Step119 worker seed does not match its frozen slot")
                    if episode_seed != 2026082500 + slot:
                        raise ValueError("Q03 Step119 episode seed does not match its frozen slot")
                    if (not isinstance(instruction_hash, str) or len(instruction_hash) != 64
                            or any(char not in "0123456789abcdef" for char in instruction_hash)):
                        raise ValueError("Q03 Step119 instruction identity hash is malformed")
                    numeric_slots.append(slot)
                    request_seed = episode_seed + step_index
                    sampling = deepcopy(self.sampling_params)
                    sampling.seed = request_seed
                    request_sampling_params.append(sampling)
                    request_receipt.append({
                        "panel_slot": slot,
                        "task_id": str(task_id),
                        "sampled_session_index": session,
                        "worker_seed": worker_seed,
                        "step_index": step_index,
                        "episode_seed": episode_seed,
                        "instruction_sha256": instruction_hash,
                        "request_seed": request_seed,
                    })
                q03_request_receipt = request_receipt
                if len(set(numeric_slots)) != len(numeric_slots):
                    raise ValueError("Q03 Step119 vLLM request slots must be unique")
                logger.info("Q03_STEP119_VLLM_REQUEST_RECEIPT %s", json.dumps(request_receipt, sort_keys=True))
            outputs = self.inference_engine.generate(
'''
    result = result.replace(VLLM_ANCHOR, replacement, 1)
    result = result.replace("sampling_params=self.sampling_params,", "sampling_params=request_sampling_params,", 1)
    response_anchor = "                use_tqdm=False,\n            )\n\n            # TODO(sgm):"
    response_receipt = (
        "                use_tqdm=False,\n            )\n"
        "            if q03_request_receipt is not None:\n"
        "                if len(outputs) != len(q03_request_receipt):\n"
        "                    raise ValueError(\"Q03 Step119 vLLM response count differs from request count\")\n"
        "                logger.info(\n"
        "                    \"Q03_STEP119_VLLM_RESPONSE_RECEIPT %s\",\n"
        "                    json.dumps({\"response_count\": len(outputs), \"request_count\": len(q03_request_receipt)}, sort_keys=True),\n"
        "                )\n\n"
        "            # TODO(sgm):"
    )
    if result.count(response_anchor) != 1:
        raise OverlayError("pinned vLLM response receipt anchor is missing or ambiguous")
    return result.replace(response_anchor, response_receipt, 1)


def patch_vllm_utils(source: str) -> str:
    old = "from vllm.lora.models import LoRAModel\n"
    if source.count(old) != 1:
        raise OverlayError("pinned vLLM 0.27 LoRAModel utility import anchor is missing or ambiguous")
    return source.replace(old, "from vllm.lora.lora_model import LoRAModel\n", 1)


def patch_model_merger(source: str) -> str:
    old_import = "    AutoModelForVision2Seq,\n"
    old_branch = (
        '        elif "ForConditionalGeneration" in self.model_config.architectures[0]:\n'
        "            return AutoModelForVision2Seq\n"
    )
    if source.count(old_import) != 1 or source.count(old_branch) != 1:
        raise OverlayError("pinned SEED model-merger vision-class anchors are missing or ambiguous")
    source = source.replace(old_import, "", 1)
    new_branch = (
        '        elif "ForConditionalGeneration" in self.model_config.architectures[0]:\n'
        "            try:\n"
        "                from transformers import AutoModelForVision2Seq\n"
        "            except ImportError as exc:\n"
        '                raise NotImplementedError("installed Transformers lacks AutoModelForVision2Seq") from exc\n'
        "            return AutoModelForVision2Seq\n"
    )
    return source.replace(old_branch, new_branch, 1)


def _git(seed_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(seed_root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _read_regular_member(root: Path, relative: str) -> bytes:
    """Read a bundle file while rejecting symlinks in every path component."""
    current = root
    parts = Path(relative).parts
    for part in parts[:-1]:
        current = current / part
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise OverlayError(f"existing overlay path component is unsafe: {relative}")
    path = current / parts[-1]
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OverlayError(f"existing overlay member is not a regular file: {relative}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)



def _overlay_import_bootstrap(members: Mapping[str, bytes]) -> bytes:
    """Return a stdlib-only sitecustomize hook that makes sealed modules win.

    The overlay is intentionally sparse. A plain PYTHONPATH entry does not
    shadow files inside an upstream regular package: its ``__init__.py`` fixes
    that package's ``__path__`` at the upstream checkout. Intercept only the
    exact hash-bound module names in this bundle so patched files cannot be
    silently bypassed by that package-resolution rule.
    """
    module_map = {
        Path(relative).with_suffix("").as_posix().replace("/", "."): relative
        for relative in members
        if relative.endswith(".py") and relative != "sitecustomize.py"
    }
    mapping_json = json.dumps(module_map, sort_keys=True, separators=(",", ":"))
    source = f'''"""Load only the content-addressed Q03 Step119 modules from this overlay."""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_MEMBERS = json.loads({mapping_json!r})

class _Q03Step119OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        relative = _MEMBERS.get(fullname)
        if relative is None:
            return None
        candidate = _ROOT / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(_ROOT)
        except (OSError, ValueError) as exc:
            raise ImportError("sealed Q03 Step119 overlay member is missing or escaped") from exc
        if candidate.is_symlink() or not resolved.is_file():
            raise ImportError("sealed Q03 Step119 overlay member is not a regular file")
        return importlib.util.spec_from_file_location(fullname, resolved)

if not any(type(finder).__name__ == "_Q03Step119OverlayFinder" for finder in sys.meta_path):
    sys.meta_path.insert(0, _Q03Step119OverlayFinder())
'''
    return source.encode("utf-8")

def _add_validation_source_modules(
    patched: dict[str, bytes], source_records: dict[str, dict[str, str]]
) -> None:
    """Package project-owned validation modules beside the upstream SEED overlays."""
    source_root = Path(__file__).resolve().parent
    for source_name, (target_relative, expected_sha) in VALIDATION_SOURCE_FILES.items():
        source_path = source_root / source_name
        if source_path.is_symlink() or not source_path.is_file():
            raise OverlayError(f"Step119 validation source must be a regular file: {source_name}")
        source_bytes = source_path.read_bytes()
        observed_sha = sha256_bytes(source_bytes)
        if observed_sha != expected_sha:
            raise OverlayError(f"pinned Step119 validation source identity mismatch: {source_name}")
        compile(source_bytes.decode("utf-8"), str(source_path), "exec")
        patched[target_relative] = source_bytes
        source_records[target_relative] = {
            "source_sha256": observed_sha,
            "overlay_sha256": observed_sha,
        }


def build_overlay(seed_root: Path, output_root: Path) -> dict[str, Any]:
    seed_root = seed_root.resolve(strict=True)
    if _git(seed_root, "rev-parse", "HEAD") != EXPECTED_SEED_COMMIT:
        raise OverlayError("SEED source revision differs from the frozen Step119 source")
    if _git(seed_root, "status", "--porcelain"):
        raise OverlayError("SEED source checkout must be clean before overlay generation")

    patched: dict[str, bytes] = {}
    source_records: dict[str, dict[str, str]] = {}
    for relative, expected_sha in SOURCE_FILES.items():
        path = seed_root / relative
        source = path.read_bytes()
        observed_sha = sha256_bytes(source)
        if observed_sha != expected_sha:
            raise OverlayError(f"pinned source file identity mismatch: {relative}")
        text = source.decode("utf-8")
        if relative == "agent_system/multi_turn_rollout/rollout_loop.py":
            output = patch_rollout_loop(text)
        elif relative == ACTOR_PATH:
            output = patch_padding_imports(text)
        elif relative == "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py":
            output = patch_vllm_rollout(text)
        elif relative == "verl/utils/vllm_utils.py":
            output = patch_vllm_utils(text)
        elif relative == "scripts/model_merger.py":
            output = patch_model_merger(text)
        elif relative == "agent_system/environments/env_manager.py":
            output = patch_webshop_manager(text)
        elif relative == "agent_system/environments/env_package/webshop/envs.py":
            output = patch_webshop_vector_env(text)
        elif relative == "verl/trainer/ppo/ray_trainer.py":
            output = patch_ray_trainer_validation_results(text)
        else:
            raise OverlayError(f"unexpected Step119 overlay source member: {relative}")
        output_bytes = output.encode("utf-8")
        compile(output, str(path), "exec")
        patched[relative] = output_bytes
        source_records[relative] = {
            "source_sha256": observed_sha,
            "overlay_sha256": sha256_bytes(output_bytes),
        }

    helper_relative = "verl/trainer/ppo/q03_step119_validation_results.py"
    helper_path = Path(__file__).with_name("q03_step119_validation_results.py")
    helper_bytes = helper_path.read_bytes()
    compile(helper_bytes.decode("utf-8"), str(helper_path), "exec")
    patched[helper_relative] = helper_bytes
    source_records[helper_relative] = {
        "source_sha256": sha256_bytes(helper_bytes),
        "overlay_sha256": sha256_bytes(helper_bytes),
    }
    _add_validation_source_modules(patched, source_records)

    bootstrap = _overlay_import_bootstrap(patched)
    patched["sitecustomize.py"] = bootstrap
    source_records["sitecustomize.py"] = {
        "source_sha256": sha256_bytes(bootstrap),
        "overlay_sha256": sha256_bytes(bootstrap),
    }

    manifest: dict[str, Any] = {
        "schema": "openevo.q03-step119-runtime-overlays.v1",
        "seed_commit": EXPECTED_SEED_COMMIT,
        "sources": source_records,
        "seed_transport": "one vLLM SamplingParams.seed per prompt request; episode_seed + rollout_step_index",
        "receipt_marker": "Q03_STEP119_VLLM_REQUEST_RECEIPT",
    }
    manifest_sha = sha256_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    manifest["overlay_bundle_sha256"] = manifest_sha
    target = output_root / manifest_sha
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise OverlayError("overlay output root must be a real directory, not a symlink")

    lock_path = output_root / ".q03-step119-overlay.lock"
    lock_fd = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise OverlayError("overlay publication lock is not a regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            target_stat = target.lstat()
        except FileNotFoundError:
            target_stat = None
        if target_stat is not None:
            if not stat.S_ISDIR(target_stat.st_mode):
                raise OverlayError("content-addressed overlay target is not a real directory")
            try:
                existing_manifest = json.loads(_read_regular_member(target, "manifest.json"))
            except (OSError, json.JSONDecodeError) as exc:
                raise OverlayError("existing overlay manifest is missing or unsafe") from exc
            if existing_manifest != manifest:
                raise OverlayError("existing content-addressed overlay bundle has different metadata")
            for relative, body in patched.items():
                try:
                    existing_body = _read_regular_member(target, relative)
                except OSError as exc:
                    raise OverlayError("existing content-addressed overlay member is unsafe or missing") from exc
                if existing_body != body:
                    raise OverlayError("existing content-addressed overlay file is unsafe or drifted")
            return {"path": str(target), **manifest}

        temp = Path(tempfile.mkdtemp(prefix=".q03-step119-overlay-", dir=output_root))
        try:
            for relative, body in patched.items():
                path = temp / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                path.chmod(0o444)
            (temp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            (temp / "manifest.json").chmod(0o444)
            for directory, subdirectories, _files in os.walk(temp, topdown=False):
                Path(directory).chmod(0o555)
            # The lock coordinates normal builders; RENAME_NOREPLACE also
            # protects against a non-cooperating creator racing publication.
            _rename_noreplace(temp, target)
        except Exception:
            _remove_staging_tree(temp)
            raise
    finally:
        os.close(lock_fd)
    return {"path": str(target), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_overlay(args.seed_root, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
