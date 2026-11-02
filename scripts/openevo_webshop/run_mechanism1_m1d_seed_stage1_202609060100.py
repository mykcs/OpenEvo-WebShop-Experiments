#!/usr/bin/env python3
"""M1-D matched-world driver for pinned public SEED WebShop Stage1.

This adapter preserves the public SEED policy/action rollout path while binding
WebShop world identity to the already-frozen OpenEvo Stage1 tasks. It never
runs SEED skill generation or downstream training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c"
PIPELINE_REL = Path("scripts/sft/webshop/pipeline.py")
SOURCE_HASHES = {
    "scripts/sft/webshop/pipeline.py": "39d1523181194d1c9d385354c74cbe6b937542c297ad1e75247b0b9c3222effd",
    "agent_system/environments/env_package/webshop/envs.py": "1a8a73b5e02854a822aa03dcb66a9fbf8d67c028f963f36845bf34fb1b60ea7c",
    "agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py": "9703c8583244e2041182eb14856186b998145ab16ce2ba9aef8cb680877c8ed5",
    "agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py": "9b86ffc82951124f5c4a7eb8103de14100dbee11ba8869951ed6fb8feaeef804",
}
MANIFEST_SCHEMA = "openevo.mechanism1-m1d-seed-stage1-matched-manifest.202609060000"
WORLD_SEED = 2026
NUM_PRODUCTS = 1000
ROLLOUTS_PER_TASK = 8
TASK_COUNT = 180
TRAJECTORY_COUNT = 1440


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("M1-D matched manifest schema drift")
    if doc.get("activation_authority") is not False or doc.get("formal_task_consumption_allowed") is not False:
        raise ValueError("M1-D matched manifest must remain execution locked")
    if int(doc.get("task_count", len(doc.get("tasks", [])))) != TASK_COUNT:
        raise ValueError("M1-D task count drift")
    if len(doc.get("tasks", [])) != TASK_COUNT:
        raise ValueError("M1-D task table must contain 180 tasks")
    matched = doc.get("matched_axes") or {}
    if int(matched.get("webshop_world_construction_seed", -1)) != WORLD_SEED:
        raise ValueError("M1-D WebShop world seed drift")
    if int(matched.get("webshop_num_products", -1)) != NUM_PRODUCTS:
        raise ValueError("M1-D WebShop product-count drift")
    if matched.get("lucene_index") != "indexes_1k":
        raise ValueError("M1-D Lucene index identity drift")
    validate_task_table(doc)
    return doc


def task_identity_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "task_ordinal": int(row["task_ordinal"]),
            "source_session_index": int(row["source_session_index"]),
            "instruction_sha256": str(row["instruction_sha256"]),
        }
        for row in doc["tasks"]
    ]


def expanded_identity_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in doc["tasks"]:
        ordinal = int(task["task_ordinal"])
        for rollout_id in range(ROLLOUTS_PER_TASK):
            position = ordinal * ROLLOUTS_PER_TASK + rollout_id + 1
            rows.append(
                {
                    "position": position,
                    "task_ordinal": ordinal,
                    "rollout_id": rollout_id,
                    "source_session_index": int(task["source_session_index"]),
                    "instruction_sha256": str(task["instruction_sha256"]),
                    "openevo_rollout_seed": 148_000_000 + position,
                    "seed_recorded_metadata": WORLD_SEED + ordinal * 1000 + rollout_id,
                }
            )
    return rows


def validate_task_table(doc: dict[str, Any]) -> None:
    tasks = doc["tasks"]
    ordinals = [int(row["task_ordinal"]) for row in tasks]
    if ordinals != list(range(TASK_COUNT)):
        raise ValueError("M1-D task ordinals must be exactly 0..179")
    sessions = [int(row["source_session_index"]) for row in tasks]
    if len(set(sessions)) != TASK_COUNT:
        raise ValueError("M1-D task sessions are not unique")
    for row in tasks:
        instruction = str(row["instruction"])
        observed = hashlib.sha256(instruction.encode()).hexdigest()
        if observed != row["instruction_sha256"]:
            raise ValueError(f"M1-D instruction hash drift at task {row['task_ordinal']}")
    matched = doc["matched_axes"]
    if canonical_sha(task_identity_rows(doc)) != matched["ordered_task_identity_sha256"]:
        raise ValueError("M1-D ordered task identity digest drift")
    expansion = doc["task_expansion_contract"]
    rows = expanded_identity_rows(doc)
    if len(rows) != TRAJECTORY_COUNT:
        raise ValueError("M1-D expanded trajectory count drift")
    if canonical_sha(rows) != expansion["expanded_schedule_identity_sha256"]:
        raise ValueError("M1-D expanded schedule digest drift")


def verify_source(seed_root: Path, source_git_root: Path | None = None) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for rel, expected in SOURCE_HASHES.items():
        path = seed_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"pinned public SEED source hash drift: {rel}")
        observed[rel] = digest
    commit = None
    if source_git_root is not None:
        commit = subprocess.check_output(
            ["git", "-C", str(source_git_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != SOURCE_COMMIT:
            raise ValueError("pinned public SEED commit drift")
        if subprocess.check_output(
            ["git", "-C", str(source_git_root), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError("pinned public SEED source checkout is dirty")
    return {"source_commit": commit or SOURCE_COMMIT, "source_hashes": observed}


def tasks_for_upstream(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": str(row["seed_task_id"]),
            "task_type": "webshop",
            "goal_idx": int(row["source_session_index"]),
            "sample_index": int(row["task_ordinal"]),
            "split": "train",
            "task_description": str(row["instruction"]),
        }
        for row in doc["tasks"]
    ]


def matched_world_worker_seeds(worker_count: int) -> list[int]:
    if worker_count < 0:
        raise ValueError("worker_count must be non-negative")
    return [WORLD_SEED] * worker_count


def bind_world_env_kwargs(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(env_kwargs)
    out["num_products"] = NUM_PRODUCTS
    return out


def validate_world_identity(seed_root: Path, doc: dict[str, Any]) -> dict[str, Any]:
    webshop_dir = seed_root / "agent_system/environments/env_package/webshop/webshop"
    data_dir = webshop_dir / "data"
    if str(seed_root) not in sys.path:
        sys.path.insert(0, str(seed_root))
    if str(webshop_dir) not in sys.path:
        sys.path.insert(0, str(webshop_dir))
    previous_ray = sys.modules.get("ray")
    if previous_ray is None:
        sys.modules["ray"] = types.ModuleType("ray")
    try:
        envs_path = seed_root / "agent_system/environments/env_package/webshop/envs.py"
        envs_spec = importlib.util.spec_from_file_location("m1d_seed_public_webshop_envs", envs_path)
        if envs_spec is None or envs_spec.loader is None:
            raise RuntimeError("cannot load pinned public SEED WebShop envs file")
        envs_module = importlib.util.module_from_spec(envs_spec)
        sys.modules[envs_spec.name] = envs_module
        envs_spec.loader.exec_module(envs_module)
        WebshopWorker = envs_module.WebshopWorker
        env_kwargs = {
            "observation_mode": "text",
            "num_products": NUM_PRODUCTS,
            "human_goals": 0,
            "file_path": str(data_dir / "items_shuffle_1000.json"),
            "attr_path": str(data_dir / "items_ins_v2_1000.json"),  # legacy-revision-compat
        }
        worker = WebshopWorker(WORLD_SEED, env_kwargs)
        try:
            goals = list(worker.get_goals())
        finally:
            worker.close()
    finally:
        if previous_ray is None:
            sys.modules.pop("ray", None)
        else:
            sys.modules["ray"] = previous_ray
    mismatches = []
    for task in doc["tasks"]:
        idx = int(task["source_session_index"])
        text = str(goals[idx]["instruction_text"])
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != task["instruction_sha256"]:
            mismatches.append({"task_ordinal": int(task["task_ordinal"]), "session": idx, "observed": digest})
    if mismatches:
        raise ValueError(f"M1-D matched-world identity mismatch count={len(mismatches)}")
    return {"status": "PASS", "goal_count": len(goals), "matched_tasks": TASK_COUNT, "world_seed": WORLD_SEED, "num_products": NUM_PRODUCTS}


def import_upstream_pipeline(seed_root: Path):
    pipeline_path = seed_root / PIPELINE_REL
    spec = importlib.util.spec_from_file_location("m1d_seed_public_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned public SEED WebShop pipeline")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_matched_build_manager(upstream):
    def build_manager(*, session_indices, seed, args):
        import ray
        from agent_system.environments.env_manager import WebshopEnvironmentManager
        from agent_system.environments.env_package.webshop import webshop_projection
        from agent_system.environments.env_package.webshop.envs import WebshopWorker

        indices = [int(value) for value in session_indices]
        if not indices:
            raise ValueError("M1-D matched manager requires at least one session")
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, num_gpus=0)
        env_kwargs = bind_world_env_kwargs(upstream.resolve_webshop_data_paths(args))
        worker_cls = ray.remote(num_cpus=max(float(args.num_cpus_per_env_worker), 0.01))(WebshopWorker)
        workers = [worker_cls.remote(WORLD_SEED, dict(env_kwargs)) for _ in indices]

        class MatchedEnvBatch:
            def __init__(self):
                self._closed = False
                self._workers = workers
                self._ray = ray
                self.session_indices = indices

            def reset(self):
                results = self._ray.get([
                    worker.reset.remote(idx) for worker, idx in zip(self._workers, self.session_indices)
                ])
                return [row[0] for row in results], [row[1] for row in results]

            def step(self, actions):
                results = self._ray.get([
                    worker.step.remote(action) for worker, action in zip(self._workers, actions)
                ])
                return tuple([row[i] for row in results] for i in range(4))

            def close(self):
                if self._closed:
                    return
                for worker in self._workers:
                    try:
                        self._ray.get(worker.close.remote())
                    except Exception:
                        pass
                    self._ray.kill(worker)
                self._closed = True

        config = upstream.AttrDict(
            {"env": upstream.AttrDict({"env_name": "webshop", "history_length": 2, "use_skills_only_memory": False})}
        )
        time.sleep(max(float(args.startup_wait_seconds), len(indices) * 0.02))
        return WebshopEnvironmentManager(MatchedEnvBatch(), webshop_projection, config)

    return build_manager


def formal_args_are_exact(args) -> None:
    required = {
        "num_tasks": TASK_COUNT,
        "rollouts_per_task": ROLLOUTS_PER_TASK,
        "max_steps": 15,
        "history_length": 2,
        "seed": WORLD_SEED,
        "policy_temperature": 0.4,
        "policy_max_completion_tokens": 512,
        "task_batch_size": 16,
        "request_workers": 128,
    }
    for key, expected in required.items():
        if getattr(args, key) != expected:
            raise ValueError(f"M1-D formal argument drift: {key}")
    if upstream_history_length(args) != 2:
        raise ValueError("M1-D baseline history length drift")


def upstream_history_length(args) -> int:
    value = args.baseline_history_length
    return int(args.history_length if value is None else value)


def validate_driver_runtime(seed_root: Path, doc: dict[str, Any]) -> dict[str, Any]:
    import importlib.metadata as metadata

    upstream = import_upstream_pipeline(seed_root)
    old_argv = sys.argv
    try:
        sys.argv = [str(seed_root / PIPELINE_REL)]
        args = upstream.parse_args()
    finally:
        sys.argv = old_argv
    formal_args_are_exact(args)
    args.webshop_use_small = True
    args.baseline_history_length = 2
    env_kwargs = bind_world_env_kwargs(upstream.resolve_webshop_data_paths(args))
    if int(env_kwargs.get("num_products", -1)) != NUM_PRODUCTS:
        raise ValueError("M1-D driver world binding drift")
    webshop_dir = seed_root / "agent_system/environments/env_package/webshop/webshop"
    index_dir = webshop_dir / "search_engine/indexes_1k"
    if not index_dir.is_dir():
        raise FileNotFoundError(index_dir)
    index_files = [path for path in index_dir.iterdir() if path.is_file()]
    if len(index_files) != 19:
        raise ValueError("M1-D Lucene index file-count drift")
    import ray
    import openai
    import omegaconf
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import webshop_projection

    if WebshopEnvironmentManager is None or webshop_projection is None:
        raise RuntimeError("M1-D SEED environment manager import failed")
    build_manager = make_matched_build_manager(upstream)
    if not callable(build_manager):
        raise RuntimeError("M1-D matched build manager is not callable")
    return {
        "status": "PASS",
        "source_commit": SOURCE_COMMIT,
        "ray": metadata.version("ray"),
        "openai": metadata.version("openai"),
        "omegaconf": metadata.version("omegaconf"),
        "hydra_core": metadata.version("hydra-core"),
        "world_seed": WORLD_SEED,
        "num_products": NUM_PRODUCTS,
        "lucene_index_files": len(index_files),
        "formal_task_consumption": 0,
        "model_calls": 0,
        "ray_actors_created": 0,
        "webshop_resets": 0,
    }


def run_formal(seed_root: Path, doc: dict[str, Any], remainder: list[str]) -> dict[str, Any]:
    verify_source(seed_root)
    upstream = import_upstream_pipeline(seed_root)
    old_argv = sys.argv
    try:
        sys.argv = [str(seed_root / PIPELINE_REL), *remainder]
        args = upstream.parse_args()
    finally:
        sys.argv = old_argv
    formal_args_are_exact(args)
    args.webshop_use_small = True
    args.baseline_history_length = 2
    args.stop_after_baseline_rollouts = True
    output_dir = Path(args.output_dir)
    upstream.load_env_file(args.env_file)
    upstream.prepare_output_dir(output_dir, overwrite=args.overwrite, resume=args.resume, regenerate_candidates=False)
    upstream.setup_logging(output_dir, args.log_level)
    policy_endpoint = upstream.resolve_endpoint(
        prefix="policy",
        args=args,
        default_base_url_env="POLICY_OPENAI_BASE_URL",
        default_model_env="POLICY_OPENAI_MODEL",
        default_model="Qwen2.5-3B-Instruct",
        temperature=args.policy_temperature,
        max_completion_tokens=args.policy_max_completion_tokens,
        timeout=args.policy_timeout,
        retries=args.policy_retries,
        retry_delay=args.policy_retry_delay,
        extra_body_json=args.policy_extra_body_json,
    )
    upstream.build_manager = make_matched_build_manager(upstream)
    tasks = tasks_for_upstream(doc)
    records = upstream.collect_baseline_rollouts(
        tasks=tasks,
        args=args,
        output_dir=output_dir,
        policy_endpoint=policy_endpoint,
    )
    if len(records) != TRAJECTORY_COUNT:
        raise RuntimeError("M1-D baseline collection did not produce exactly 1,440 trajectories")
    expected = {(row["task_id"], rid) for row in tasks for rid in range(ROLLOUTS_PER_TASK)}
    observed = {(str(row.get("task_id")), int(row.get("rollout_id", -1))) for row in records}
    if observed != expected:
        raise RuntimeError("M1-D baseline collection identity coverage drift")
    corpus = output_dir / "baseline_rollouts.jsonl"
    receipt = {
        "schema": "openevo.mechanism1-m1d-seed-stage1-baseline-complete.202609060100",
        "status": "PASS",
        "trajectory_count": TRAJECTORY_COUNT,
        "task_count": TASK_COUNT,
        "world_seed": WORLD_SEED,
        "num_products": NUM_PRODUCTS,
        "policy_temperature": 0.4,
        "source_commit": SOURCE_COMMIT,
        "baseline_rollouts_sha256": sha256_file(corpus),
        "skill_generation_started": False,
        "seed_training_started": False,
        "seed_stage2_started": False,
    }
    (output_dir / "M1D_BASELINE_COMPLETE_202609060100.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--source-git-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("source", "world", "driver", "formal"), required=True)
    known, remainder = parser.parse_known_args()
    doc = load_manifest(known.manifest)
    source = verify_source(known.seed_root, known.source_git_root)
    if known.mode == "source":
        print(json.dumps({"status": "PASS", **source}, sort_keys=True))
        return 0
    if known.mode == "world":
        result = validate_world_identity(known.seed_root, doc)
        print(json.dumps(result, sort_keys=True))
        return 0
    if known.mode == "driver":
        result = validate_driver_runtime(known.seed_root, doc)
        print(json.dumps(result, sort_keys=True))
        return 0
    result = run_formal(known.seed_root, doc, remainder)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
