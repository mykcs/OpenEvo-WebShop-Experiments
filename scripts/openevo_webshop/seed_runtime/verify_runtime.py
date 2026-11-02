#!/usr/bin/env python3
"""Fail-closed runtime probe for the external SEED WebShop source."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--webshop-data", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    data_root = args.webshop_data.resolve()
    index_root = args.index_root.resolve()
    if not (source_root / "seed" / "analysis.py").is_file():
        raise SystemExit(f"SEED source is incomplete: {source_root}")
    for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json"):
        if not (data_root / name).is_file():
            raise SystemExit(f"WebShop data is incomplete: {data_root / name}")
    if not index_root.is_dir():
        raise SystemExit(f"WebShop Lucene index is missing: {index_root}")

    sys.path.insert(0, str(source_root))
    webshop_root = source_root / "agent_system/environments/env_package/webshop/webshop"
    sys.path.insert(0, str(webshop_root))
    import torch
    import transformers
    import vllm

    import seed  # noqa: F401
    import verl  # noqa: F401
    import web_agent_site.utils as webshop_utils
    from web_agent_site.engine import engine as webshop_engine
    from agent_system.environments.env_package.webshop.envs import WebshopWorker
    from pyserini.search.lucene import LuceneSearcher

    # The source checkout and external data/index assets are separate mounts.
    # Rebind the source module constants instead of mutating the checkout.
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

    cuda = {
        "available": bool(torch.cuda.is_available()),
        "count": int(torch.cuda.device_count()),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    if args.require_cuda and not cuda["available"]:
        raise SystemExit("CUDA is required but unavailable")

    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "vllm", "ray", "hydra-core", "datasets", "pyarrow", "wandb")
    }
    worker = WebshopWorker(
        2026,
        {
            "observation_mode": "text",
            "num_products": 1000,
            "file_path": str(data_root / "items_shuffle_1000.json"),
            "attr_path": str(data_root / "items_ins_v2_1000.json"),
        },
    )
    try:
        observation, reset_info = worker.reset(0)
        action_state = worker.get_available_actions()
        actions = action_state.get("clickables", [])
        if not actions:
            raise SystemExit("WebShop reset returned no admissible actions")
        next_observation, reward, done, step_info = worker.step(actions[0])
    finally:
        worker.close()

    print(json.dumps({
        "cuda": cuda,
        "versions": versions,
        "imports": {"transformers": transformers.__name__, "vllm": vllm.__name__},
        "webshop_worker": WebshopWorker.__name__,
        "webshop_reset": {"observation_type": type(observation).__name__, "info_keys": sorted(reset_info)},
        "webshop_step": {"observation_type": type(next_observation).__name__, "reward": reward, "done": done, "info_keys": sorted(step_info)},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
