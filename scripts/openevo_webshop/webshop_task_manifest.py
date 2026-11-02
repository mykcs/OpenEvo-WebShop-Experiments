#!/usr/bin/env python3
"""Create and verify the shared WebShop task contract for method comparisons.

The manifest binds a WebShop session index to its exact instruction text under
one catalog and one environment seed. Model rollout randomness is deliberately
kept separate from the environment seed so parallel workers cannot silently
reinterpret the same session index as different tasks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "seed3090.webshop-task-manifest.v1"
ACCESS_LEDGER_SCHEMA = "seed3090.webshop-task-access-ledger.v1"
DATA_SOURCE = "github-api:WooooDyy/AgentGym@3ef9235d23e68e7c2920c5422ad957dc8ced5c6c"
DATA_FILES = {
    "items_shuffle_1000.json": (4_467_013, "30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269"),
    "items_ins_v2_1000.json": (147_099, "f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f"),
    "items_human_ins.json": (5_137_548, "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3"),
}
ENVIRONMENT_SEED = 2026
SELECTION_SEED = 2026
TRAIN_START = 500
GOAL_COUNT = 6_910
QUALIFICATION_TASKS = 8
MEMORY_TASKS = 2
PROMOTION_TASKS = 2
REFLECTION_TASKS = MEMORY_TASKS + PROMOTION_TASKS
QUALIFICATION_ROLLOUTS = 2
NUM_PRODUCTS = 1000
MAX_STEPS = 15
HISTORY_LENGTH = 2
TEMPERATURE = 0.4
MAX_NEW_TOKENS = 256
PROMPT_CONTRACT = "seed3090.webshop-shared-prompt.v1"
ACTION_CONTRACT = "seed3090.webshop-action-parser.v1"
ROLLOUT_SEEDS = [202600, 202601]
MEMORY_ROLLOUT_SEEDS = [202650, 202651]
PROMOTION_ROLLOUT_SEEDS = [202652, 202653]
REFLECTOR_SEED = 202699
INDEX_RULES = {
    "name": "indexes_1k",
    "document_count": 1000,
    "semantic_queries": [
        {"query": "toothpaste", "requested": 3, "minimum_hits": 1},
        {"query": "wireless charging iphone", "requested": 3, "minimum_hits": 1},
        {"query": "men dress shirt", "requested": 3, "minimum_hits": 1},
    ],
}
HISTORICAL_INDEX_IDENTITY = {
    "file_count": 19,
    "logical_bytes": 3_031_499,
    "tree_sha256": "0c02272483e1b2c88f99b3dea264f5636d1b6e70427b037958a45b5a9ff6b200",
}
EVALUATOR = {
    "score": "info.task_score (continuous 0..1 WebShop reward)",
    "success": "info.won == true and info.task_score == 1.0",
}
FROZEN_TASKS = (
    (612, "train-memory", "Find me men's shorts with comfortable fit, elastic waistband for gym workout, daily wear with color: grey, and size: large, and price lower than 30.00 dollars"),
    (566, "train-memory", "Find me machine wash men's dress shirts with polyester heathers, heathers cotton, cotton heather, needle sleeve, classic fit with color: dark heather, and fit type: youth, and size: 3x-large, and price lower than 40.00 dollars"),
    (6680, "promotion-dev", "Find me slim fit women's jumpsuits, rompers & overalls with long sleeve for daily wear with color: black, and size: small, and price lower than 50.00 dollars"),
    (2590, "promotion-dev", "Find me quick release, hands free iphone accessories for wireless charging with color: plus, and style: dash, and price lower than 60.00 dollars"),
    (3420, "engineering-qualification", "Find me wash cold, machine wash men's shirts with polyester heathers, heathers cotton, cotton heather, needle sleeve, classic fit with color: olive, and fit type: youth, and size: xx-large, and price lower than 50.00 dollars"),
    (6624, "engineering-qualification", "Find me slim fit, loose fit men's tuxedo shirts with long sleeve, short sleeve, contrast color, classic fit for teen girls with color: a-bk2, and size: 5x-large, and price lower than 40.00 dollars"),
    (5216, "engineering-qualification", "Find me high performance, high speed flashes with color: black, and size: 256mb, and price lower than 130.00 dollars"),
    (2655, "engineering-qualification", "Find me loose fit, day comfort, hand wash women's tops, tees & blouses with short sleeve, polyester spandex for teen girls with color: a6-blue, and size: small, and price lower than 20.00 dollars"),
    (5457, "engineering-qualification", "Find me hand wash, machine wash women's sweaters with long sleeve, unique design with color: coffee brown, and size: large, and price lower than 50.00 dollars"),
    (3768, "engineering-qualification", "Find me machine wash men's dress shirts with cotton spandex, classic fit, short sleeve with color: bright white, and size: x-large, and price lower than 70.00 dollars"),
    (4513, "engineering-qualification", "Find me home office furniture sets for dining room, living room with color: brown | black, and item shape: rectangular, and size: 5 ft 3 in x 7 ft 5 in, and price lower than 70.00 dollars"),
    (1657, "engineering-qualification", "Find me home office furniture sets for dining room, living room with color: navy-multi, and item shape: rectangular, and size: 1 ft 9 in x 7 ft 2 in, and price lower than 70.00 dollars"),
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def document_digest(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def goal_description(goal: Any) -> str:
    value = goal.get("instruction_text") if isinstance(goal, dict) else goal
    text = str(value or "").strip()
    if not text:
        raise ValueError("WebShop goal has an empty instruction_text")
    return text


def build_qualification_manifest(goals: Sequence[Any], *, environment_seed: int = ENVIRONMENT_SEED, selection_seed: int = SELECTION_SEED, index_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    if len(goals) != GOAL_COUNT:
        raise ValueError(f"qualification requires exactly {GOAL_COUNT} pinned goals; found {len(goals)}")
    pool = list(range(TRAIN_START, len(goals)))
    random.Random(selection_seed).shuffle(pool)
    selected = pool[: REFLECTION_TASKS + QUALIFICATION_TASKS]
    expected_indices = [goal_idx for goal_idx, _, _ in FROZEN_TASKS]
    if selected != expected_indices:
        raise ValueError("qualification selection algorithm drifted from the frozen task set")

    expected_offset = 0
    def task_rows(indices: Sequence[int], split: str) -> list[dict[str, Any]]:
        nonlocal expected_offset
        rows = []
        for sample_index, goal_idx in enumerate(indices):
            description = goal_description(goals[goal_idx])
            expected_description = FROZEN_TASKS[expected_offset + sample_index][2]
            if description != expected_description:
                raise ValueError(f"frozen task text mismatch at goal {goal_idx}")
            rows.append({
                "sample_index": sample_index,
                "task_id": f"webshop_{goal_idx:06d}",
                "goal_idx": goal_idx,
                "split": split,
                "task_description": description,
                "task_description_sha256": sha256_text(description),
            })
        expected_offset += len(indices)
        return rows

    memory_tasks = task_rows(selected[:MEMORY_TASKS], "train-memory")
    promotion_tasks = task_rows(selected[MEMORY_TASKS:REFLECTION_TASKS], "promotion-dev")
    tasks = task_rows(selected[REFLECTION_TASKS:], "engineering-qualification")
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "phase": "qualification",
        "benchmark": "webshop",
        "environment_seed": int(environment_seed),
        "selection_seed": int(selection_seed),
        "goal_count": GOAL_COUNT,
        "num_products": NUM_PRODUCTS,
        "catalog": {"source": DATA_SOURCE, "files": [{"name": name, "bytes": size, "sha256": digest} for name, (size, digest) in sorted(DATA_FILES.items())]},
        "controls": {
            "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "history_length": HISTORY_LENGTH,
            "temperature": TEMPERATURE,
            "prompt_contract": PROMPT_CONTRACT,
            "action_contract": ACTION_CONTRACT,
            "rollout_seeds": ROLLOUT_SEEDS,
            "memory_rollout_seeds": MEMORY_ROLLOUT_SEEDS,
            "promotion_rollout_seeds": PROMOTION_ROLLOUT_SEEDS,
            "reflector_seed": REFLECTOR_SEED,
        },
        "index": {**INDEX_RULES, "materialized_tree": index_identity or HISTORICAL_INDEX_IDENTITY},
        "rollouts_per_task": QUALIFICATION_ROLLOUTS,
        "evaluator": EVALUATOR,
        "memory_tasks": memory_tasks,
        "promotion_tasks": promotion_tasks,
        "tasks": tasks,
    }
    document["manifest_sha256"] = document_digest(document)
    validate_manifest(document)
    return document


def validate_manifest(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("task manifest schema mismatch")
    if document.get("benchmark") != "webshop" or document.get("phase") != "qualification":
        raise ValueError("task manifest benchmark/phase mismatch")
    if document.get("num_products") != NUM_PRODUCTS or document.get("environment_seed") != ENVIRONMENT_SEED or document.get("selection_seed") != SELECTION_SEED or document.get("rollouts_per_task") != QUALIFICATION_ROLLOUTS or document.get("goal_count") != GOAL_COUNT:
        raise ValueError("task manifest fixed controls mismatch")
    expected_controls = {
        "max_steps": MAX_STEPS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "history_length": HISTORY_LENGTH,
        "temperature": TEMPERATURE,
        "prompt_contract": PROMPT_CONTRACT,
        "action_contract": ACTION_CONTRACT,
        "rollout_seeds": ROLLOUT_SEEDS,
        "memory_rollout_seeds": MEMORY_ROLLOUT_SEEDS,
        "promotion_rollout_seeds": PROMOTION_ROLLOUT_SEEDS,
        "reflector_seed": REFLECTOR_SEED,
    }
    if document.get("controls") != expected_controls:
        raise ValueError("task manifest control settings mismatch")
    index = document.get("index")
    if not isinstance(index, dict) or {key: index.get(key) for key in INDEX_RULES} != INDEX_RULES:
        raise ValueError("task manifest search index mismatch")
    tree = index.get("materialized_tree")
    if not isinstance(tree, dict) or not isinstance(tree.get("file_count"), int) or tree["file_count"] <= 0 or not isinstance(tree.get("logical_bytes"), int) or tree["logical_bytes"] <= 0 or len(str(tree.get("tree_sha256", ""))) != 64:
        raise ValueError("task manifest materialized index identity mismatch")
    if document.get("evaluator") != EVALUATOR:
        raise ValueError("task manifest evaluator mismatch")
    expected_catalog = {"source": DATA_SOURCE, "files": [{"name": name, "bytes": size, "sha256": digest} for name, (size, digest) in sorted(DATA_FILES.items())]}
    if document.get("catalog") != expected_catalog:
        raise ValueError("task manifest catalog identity mismatch")
    groups = [("train-memory", document.get("memory_tasks"), MEMORY_TASKS), ("promotion-dev", document.get("promotion_tasks"), PROMOTION_TASKS), ("engineering-qualification", document.get("tasks"), QUALIFICATION_TASKS)]
    seen_ids: set[str] = set(); seen_indices: set[int] = set(); frozen_offset = 0
    for expected_split, rows, expected_count in groups:
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError("task manifest task count mismatch")
        for sample_index, task in enumerate(rows):
            if not isinstance(task, dict):
                raise ValueError("task manifest task must be an object")
            goal_idx = task.get("goal_idx"); task_id = task.get("task_id")
            if not isinstance(goal_idx, int) or goal_idx < TRAIN_START or task_id != f"webshop_{goal_idx:06d}" or task.get("sample_index") != sample_index or task.get("split") != expected_split:
                raise ValueError("task manifest task identity mismatch")
            if task_id in seen_ids or goal_idx in seen_indices:
                raise ValueError("task manifest contains a duplicate task")
            seen_ids.add(task_id); seen_indices.add(goal_idx)
            description = goal_description(task.get("task_description"))
            if not hmac.compare_digest(str(task.get("task_description_sha256", "")), sha256_text(description)):
                raise ValueError(f"description hash mismatch for {task_id}")
            frozen_goal_idx, frozen_split, frozen_description = FROZEN_TASKS[frozen_offset + sample_index]
            if goal_idx != frozen_goal_idx or expected_split != frozen_split or description != frozen_description:
                raise ValueError(f"frozen task identity mismatch for {task_id}")
        frozen_offset += len(rows)
    if not hmac.compare_digest(str(document.get("manifest_sha256", "")), document_digest(document)):
        raise ValueError("task manifest document hash mismatch")
    return document


def expected_episode_keys(document: dict[str, Any], stage: str) -> list[tuple[str, str, int, str]]:
    validate_manifest(document)
    if stage == "promotion-dev":
        tasks = document["promotion_tasks"]; seeds = document["controls"]["promotion_rollout_seeds"]
    elif stage == "engineering-qualification":
        tasks = document["tasks"]; seeds = document["controls"]["rollout_seeds"]
    else:
        raise ValueError(f"unsupported qualification stage: {stage}")
    return [(stage, str(task["task_id"]), int(seed), arm) for task in tasks for seed in seeds for arm in ("v0", "o1")]


def build_access_ledger(document: dict[str, Any], *, protocol_version: str) -> dict[str, Any]:
    validate_manifest(document)
    rows = []
    for task in list(document["memory_tasks"]) + list(document["promotion_tasks"]) + list(document["tasks"]):
        engineering = task["split"] == "engineering-qualification"
        rows.append({
            "task_id": task["task_id"], "task_description_sha256": task["task_description_sha256"], "split": task["split"], "protocol_version": protocol_version,
            "selected_at": "unknown", "first_identity_accessed_at": "unknown", "first_content_accessed_at": "unknown" if engineering else "not_verified",
            "first_outcome_accessed_at": "not_verified", "actor": "unknown", "purpose": "historical reset/source review" if engineering else "manifest identity freeze",
            "influenced_tuning": False, "run_uuid": None, "state": "CONTENT_ACCESSED" if engineering else "IDENTITY_EXPOSED",
        })
    ledger = {"schema": ACCESS_LEDGER_SCHEMA, "task_manifest_sha256": document["manifest_sha256"], "rows": rows}
    ledger["ledger_sha256"] = document_digest(ledger)
    return ledger


def advance_access_ledger(ledger: dict[str, Any], task_ids: set[str], state: str, *, run_uuid: str) -> dict[str, Any]:
    terminal_states = {"TRAIN_MEMORY_CONSUMED", "PROMOTION_CONSUMED", "ENGINEERING_CONSUMED", "PILOT_CONSUMED", "FINAL_CONSUMED"}
    if state not in {"IDENTITY_EXPOSED", "CONTENT_ACCESSED", *terminal_states}:
        raise ValueError(f"unsupported access ledger state: {state}")
    updated = json.loads(json.dumps(ledger)); updated.pop("ledger_sha256", None); found = set(); rank = {"IDENTITY_EXPOSED": 0, "CONTENT_ACCESSED": 1}
    for row in updated.get("rows", []):
        if row.get("task_id") not in task_ids: continue
        found.add(row["task_id"]); current = row.get("state")
        if current in terminal_states and current != state: raise ValueError("access ledger cannot move backwards or between consumed states")
        if current in rank and state in rank and rank[state] < rank[current]: raise ValueError("access ledger cannot move backwards")
        row["state"] = state; row["run_uuid"] = run_uuid
        if state in terminal_states: row.setdefault("consumed_at", datetime.now(timezone.utc).isoformat())
    if found != task_ids: raise ValueError("access ledger task identity mismatch")
    updated["ledger_sha256"] = document_digest(updated)
    return updated


def validate_access_ledger(ledger: Any, *, task_manifest_sha256: str) -> dict[str, Any]:
    if not isinstance(ledger, dict) or ledger.get("schema") != ACCESS_LEDGER_SCHEMA or ledger.get("task_manifest_sha256") != task_manifest_sha256:
        raise ValueError("access ledger schema/manifest mismatch")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("access ledger must contain exactly twelve rows")
    ids = [row.get("task_id") for row in rows if isinstance(row, dict)]
    if len(ids) != 12 or len(set(ids)) != 12:
        raise ValueError("access ledger task identities are incomplete or duplicated")
    expected = dict(ledger); actual_digest = expected.pop("ledger_sha256", None)
    if actual_digest != document_digest(expected): raise ValueError("access ledger digest mismatch")
    return ledger


def load_access_ledger(path: Path, *, task_manifest_sha256: str) -> dict[str, Any]:
    return validate_access_ledger(json.loads(path.read_text(encoding="utf-8")), task_manifest_sha256=task_manifest_sha256)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        return validate_manifest(json.load(handle))


def write_manifest(path: Path, document: dict[str, Any]) -> None:
    validate_manifest(document); destination = path.expanduser().resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name); json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(destination)


def write_access_ledger(path: Path, ledger: dict[str, Any]) -> None:
    validate_access_ledger(ledger, task_manifest_sha256=str(ledger.get("task_manifest_sha256"))); destination = path.expanduser().resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name); json.dump(ledger, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, destination)


def verify_catalog(data_dir: Path) -> None:
    for name, (expected_size, expected_digest) in DATA_FILES.items():
        path = data_dir / name
        if not path.is_file() or path.stat().st_size != expected_size: raise ValueError(f"catalog size mismatch: {name}")
        if not hmac.compare_digest(hashlib.sha256(path.read_bytes()).hexdigest(), expected_digest): raise ValueError(f"catalog hash mismatch: {name}")


def index_tree_identity(index_dir: Path) -> dict[str, Any]:
    rows = [{"name": path.relative_to(index_dir).as_posix(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in index_dir.rglob("*") if path.is_file()]
    rows.sort(key=lambda row: row["name"])
    if not rows: raise ValueError("WebShop indexes_1k is empty")
    return {"file_count": len(rows), "logical_bytes": sum(row["bytes"] for row in rows), "tree_sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest()}


def verify_index(index_dir: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    actual = index_tree_identity(index_dir)
    if expected is not None and actual != expected: raise ValueError(f"WebShop indexes_1k identity mismatch: {actual}")
    return actual


def live_goals(
    seed_root: Path,
    environment_seed: int,
    *,
    data_dir: Path | None = None,
    index_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize deterministic goals while allowing mounted assets.

    Runtime source checkouts intentionally do not own the dataset or Lucene
    index.  The optional paths make that boundary explicit and prevent an
    empty source-side ``data`` directory from being mistaken for the active
    benchmark assets.
    """
    webshop_root = (seed_root / "agent_system/environments/env_package/webshop/webshop").resolve()
    data_dir = (data_dir or (webshop_root / "data")).expanduser().resolve()
    verify_catalog(data_dir)
    index_dir = (index_dir or (webshop_root / "search_engine/indexes_1k")).expanduser().resolve()
    if not index_dir.is_dir(): raise FileNotFoundError(f"WebShop indexes_1k is not ready: {index_dir}")
    index_identity = verify_index(index_dir)
    import sys
    if str(webshop_root) not in sys.path: sys.path.insert(0, str(webshop_root))
    previous_cwd = Path.cwd(); env = None
    try:
        os.chdir(webshop_root)
        import gym
        import web_agent_site.engine.engine as engine_module
        import web_agent_site.envs.web_agent_text_env as text_env_module
        from pyserini.search.lucene import LuceneSearcher
        from web_agent_site.envs import WebAgentTextEnv  # noqa: F401
        previous_paths = {
            "file": engine_module.DEFAULT_FILE_PATH,
            "attr": engine_module.DEFAULT_ATTR_PATH,
            "human_attr": engine_module.HUMAN_ATTR_PATH,
            "search": text_env_module.init_search_engine,
        }
        engine_module.DEFAULT_FILE_PATH = str(data_dir / "items_shuffle_1000.json")
        engine_module.DEFAULT_ATTR_PATH = str(data_dir / "items_ins_v2_1000.json")
        engine_module.HUMAN_ATTR_PATH = str(data_dir / "items_human_ins.json")
        text_env_module.init_search_engine = lambda num_products=None: LuceneSearcher(str(index_dir))
        env = gym.make("WebAgentTextEnv-v0", observation_mode="text", file_path=str(data_dir / "items_shuffle_1000.json"), num_products=NUM_PRODUCTS, human_goals=0, seed=int(environment_seed), disable_env_checker=True)
        return [dict(goal) for goal in env.unwrapped.server.goals], index_identity
    finally:
        if "previous_paths" in locals():
            engine_module.DEFAULT_FILE_PATH = previous_paths["file"]
            engine_module.DEFAULT_ATTR_PATH = previous_paths["attr"]
            engine_module.HUMAN_ATTR_PATH = previous_paths["human_attr"]
            text_env_module.init_search_engine = previous_paths["search"]
        if env is not None: env.close()
        os.chdir(previous_cwd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="Capture 8 qualification tasks."); capture.add_argument("--seed-root", type=Path, required=True); capture.add_argument("--out", type=Path, required=True); capture.add_argument("--ledger-out", type=Path)
    verify = subparsers.add_parser("verify", help="Validate a task manifest."); verify.add_argument("manifest", type=Path); args = parser.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "capture":
        goals, index_identity = live_goals(args.seed_root.expanduser().resolve(), ENVIRONMENT_SEED); document = build_qualification_manifest(goals, index_identity=index_identity); write_manifest(args.out, document)
        if args.ledger_out is not None:
            ledger = build_access_ledger(document, protocol_version="qualification-v1"); write_access_ledger(args.ledger_out, ledger); print(f"WEBSHOP_TASK_ACCESS_LEDGER={args.ledger_out.expanduser().resolve()}"); print(f"WEBSHOP_TASK_ACCESS_LEDGER_SHA256={ledger['ledger_sha256']}")
        print(f"WEBSHOP_TASK_MANIFEST={args.out.expanduser().resolve()}"); print(f"WEBSHOP_TASK_MANIFEST_SHA256={document['manifest_sha256']}"); print(f"WEBSHOP_TASK_COUNT={len(document['tasks'])}"); print("GPU_WORKLOAD_LAUNCHED=0"); return 0
    document = load_manifest(args.manifest); print(f"WEBSHOP_TASK_MANIFEST_VERIFY_PASS={document['manifest_sha256']}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
