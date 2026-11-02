#!/usr/bin/env python3
"""Non-scientific one-step SEED WebShop plus local-vLLM integration probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from functools import partial
from pathlib import Path


def validate_train_pool_index(index: int, goal_count: int | None = None) -> int:
    if index < 500 or (goal_count is not None and index >= goal_count):
        raise ValueError("qualification task must belong to native train pool [500, goal_count)")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--webshop-data", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--goal-index", type=int, required=True)
    parser.add_argument("--worker-seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    validate_train_pool_index(args.goal_index)

    source_root = args.source_root.resolve()
    data_root = args.webshop_data.resolve()
    index_root = args.index_root.resolve()
    model_path = args.model.resolve()
    if not (source_root / "seed" / "analysis.py").is_file():
        raise SystemExit(f"SEED source is incomplete: {source_root}")
    if not model_path.is_dir():
        raise SystemExit(f"model directory is missing: {model_path}")
    if not index_root.is_dir():
        raise SystemExit(f"WebShop Lucene index is missing: {index_root}")

    sys.path.insert(0, str(source_root))
    webshop_root = source_root / "agent_system/environments/env_package/webshop/webshop"
    sys.path.insert(0, str(webshop_root))

    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf

    import web_agent_site.utils as webshop_utils
    from web_agent_site.engine import engine as webshop_engine
    from agent_system.environments.env_package.webshop.envs import WebshopWorker
    from agent_system.environments.env_package.webshop.projection import webshop_projection
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from pyserini.search.lucene import LuceneSearcher

    webshop_utils.DEFAULT_FILE_PATH = str(data_root / "items_shuffle_1000.json")
    webshop_utils.DEFAULT_ATTR_PATH = str(data_root / "items_ins_v2_1000.json")
    webshop_utils.HUMAN_ATTR_PATH = str(data_root / "items_human_ins.json")
    webshop_engine.DEFAULT_FILE_PATH = webshop_utils.DEFAULT_FILE_PATH
    webshop_engine.DEFAULT_ATTR_PATH = webshop_utils.DEFAULT_ATTR_PATH
    webshop_engine.HUMAN_ATTR_PATH = webshop_utils.HUMAN_ATTR_PATH

    def init_search_engine(num_products=None):
        if num_products != 1000:
            raise ValueError(f"runtime probe only supports the frozen 1000-product index, got {num_products}")
        return LuceneSearcher(str(index_root))

    webshop_engine.init_search_engine = init_search_engine

    cuda_available = bool(torch.cuda.is_available())
    if args.require_cuda and not cuda_available:
        raise SystemExit("CUDA is required but unavailable")

    worker = WebshopWorker(
        args.worker_seed,
        {
            "observation_mode": "text",
            "num_products": 1000,
            "file_path": str(data_root / "items_shuffle_1000.json"),
            "attr_path": str(data_root / "items_ins_v2_1000.json"),
        },
    )
    try:
        validate_train_pool_index(args.goal_index, len(worker.get_goals()))

        class SingleWorkerVector:
            def reset(self):
                obs, info = worker.reset(args.goal_index)
                return [obs], [info]

            def step(self, actions):
                if len(actions) != 1:
                    raise ValueError("qualification permits exactly one worker action")
                obs, reward, done, info = worker.step(actions[0])
                return [obs], [reward], [done], [info]

        config = OmegaConf.load(source_root / "verl/trainer/config/ppo_trainer.yaml")
        config.data.max_prompt_length = 4096
        config.data.max_response_length = 512
        config.data.truncation = "left"
        config.env.env_name = "Webshop"
        config.env.history_length = 2
        manager = WebshopEnvironmentManager(
            SingleWorkerVector(), partial(webshop_projection, require_think=True), config
        )
        observations, infos = manager.reset({})
        observation = observations["text"][0]
        reset_info = infos[0]
        available_actions = reset_info["available_actions"]
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        row = TrajectoryCollector(config, tokenizer).build_prompt_sample(observation)
        prompt = tokenizer.decode(row["raw_prompt_ids"], skip_special_tokens=False)
        llm = LLM(
            model=str(model_path),
            tensor_parallel_size=1,
            gpu_memory_utilization=0.50,
            max_model_len=20480,
            enforce_eager=True,
            seed=args.policy_seed,
        )
        generated = llm.generate(
            [{"prompt_token_ids": row["raw_prompt_ids"]}],
            SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=512),
        )[0].outputs[0].text
        selected_action, valids = webshop_projection([generated], require_think=True)
        next_observations, rewards, dones, step_infos = manager.step([generated])
        next_observation, reward, done, step_info = (
            next_observations["text"][0], float(rewards[0]), bool(dones[0]), step_infos[0]
        )
    finally:
        worker.close()

    result = {
        "status": "NATIVE_ONE_STEP_INTERFACE_COMPLETED",
        "formal_activation_allowed": False,
        "native_ppo_update_qualified": False,
        "canonical_q17_qualified": False,
        "formal_task_consumption": 0,
        "goal_index": args.goal_index,
        "worker_seed": args.worker_seed,
        "policy_engine_seed": args.policy_seed,
        "request_sampling_seed": None,
        "policy_seed_scope": "vLLM engine seed only; per-request sampling seed and native PPO RNG are not qualified",
        "cuda": cuda_available,
        "versions": {
            "torch": importlib.metadata.version("torch"),
            "vllm": importlib.metadata.version("vllm"),
            "flashinfer-python": importlib.metadata.version("flashinfer-python"),
            "flashinfer-cubin": importlib.metadata.version("flashinfer-cubin"),
            "flashinfer-jit-cache": importlib.metadata.version("flashinfer-jit-cache"),
        },
        "webshop_reset": {
            "observation_type": type(observation).__name__,
            "available_actions": len(available_actions),
            "info_keys": sorted(reset_info),
        },
        "generation": {"text": generated, "prompt": prompt},
        "action": {"selected": selected_action[0], "valid": bool(valids[0]),
                   "used_fallback": False, "invalid_action_is_scientific_outcome": True},
        "webshop_step": {
            "observation_type": type(next_observation).__name__,
            "reward": reward,
            "done": done,
            "info_keys": sorted(step_info),
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
