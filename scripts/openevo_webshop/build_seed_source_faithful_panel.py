#!/usr/bin/env python3
"""Build the source-faithful SEED WebShop first-validation 128-slot manifest.

This is a CPU-only catalog builder. It imports the pinned public SEED
environment source (``jinyangwu/SEED@2cf2fadca3c5aba28da68e8e1405182ba8d90e6c``)
with heavy runtime dependencies stubbed, reproduces the per-slot worker-seed
semantics, and materializes the exact ordered semantic panel required by
``docs/science/webshop/benchmark/SEED_WEBSHOP_PUBLIC_CODE_REPRODUCTION_V1.md``.

Semantics:

- ``validation_base_seed = 1000`` (``env.seed=0 + 1000``);
- session draw = ``np.random.RandomState(1000).choice(np.arange(500), size=128, replace=False)``;
- slot ``i`` uses ``worker_seed = 1000 + i``;
- the worker shuffles its own goal list with Python ``random`` before
  ``reset(session=sampled_session_index)`` selects the goal;
- the manifest records the exact instruction text for that worker/session pair.

The manifest is canonicalized with stable key ordering and no self-hash field,
then SHA-256 hashed. Two independent clean-process runs must produce the same
canonical digest; otherwise the builder fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any

EXPECTED_GOALS_TOTAL = 6910
HELD_OUT_SESSION_MIN = 0
HELD_OUT_SESSION_MAX = 499
PANEL_SIZE = 128
VALIDATION_BASE_SEED = 1000
WORKER_SEED_OFFSET = 1000
SCHEMA = "openevo.webshop-seed-source-faithful-panel.v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_digest(value: Any) -> str:
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _install_dependency_stubs() -> None:
    """Inject lightweight stubs for SEED imports that are not needed for goal construction."""
    # pyserini
    pyserini = types.ModuleType("pyserini")
    pyserini.search = types.ModuleType("pyserini.search")
    pyserini.search.lucene = types.ModuleType("pyserini.search.lucene")

    class _LuceneSearcher:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def search(self, *args: Any, **kwargs: Any) -> list:
            return []

        def doc(self, *args: Any, **kwargs: Any) -> Any:
            return None

    pyserini.search.lucene.LuceneSearcher = _LuceneSearcher
    sys.modules["pyserini"] = pyserini
    sys.modules["pyserini.search"] = pyserini.search
    sys.modules["pyserini.search.lucene"] = pyserini.search.lucene

    # flask
    flask = types.ModuleType("flask")

    def _render_template_string(template: str, **kwargs: Any) -> str:
        return template

    flask.render_template_string = _render_template_string

    class _Flask:
        pass

    flask.Flask = _Flask
    sys.modules["flask"] = flask

    # cleantext, tqdm, rank_bm25
    sys.modules["cleantext"] = types.ModuleType("cleantext")
    tqdm_mod = types.ModuleType("tqdm")

    def _tqdm(iterable: Any, **kwargs: Any) -> Any:
        return iterable

    tqdm_mod.tqdm = _tqdm
    sys.modules["tqdm"] = tqdm_mod
    rank_bm25 = types.ModuleType("rank_bm25")

    class _BM25Okapi:
        pass

    rank_bm25.BM25Okapi = _BM25Okapi
    sys.modules["rank_bm25"] = rank_bm25

    # bs4
    bs4 = types.ModuleType("bs4")

    class _BeautifulSoup:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _Comment:
        pass

    bs4.BeautifulSoup = _BeautifulSoup
    bs4.Comment = _Comment
    bs4.element = types.ModuleType("bs4.element")
    bs4.element.Comment = _Comment
    sys.modules["bs4"] = bs4
    sys.modules["bs4.element"] = bs4.element

    # selenium
    for name in ("selenium", "selenium.webdriver"):
        sys.modules[name] = types.ModuleType(name)

    # thefuzz
    thefuzz = types.ModuleType("thefuzz")
    thefuzz.fuzz = types.ModuleType("thefuzz.fuzz")
    thefuzz.fuzz.token_set_ratio = lambda a, b: 0
    sys.modules["thefuzz"] = thefuzz
    sys.modules["thefuzz.fuzz"] = thefuzz.fuzz

    # rich
    rich_mod = types.ModuleType("rich")

    def _silent_print(*args: Any, **kwargs: Any) -> None:
        return None

    rich_mod.print = _silent_print
    sys.modules["rich"] = rich_mod

    # spacy
    spacy = types.ModuleType("spacy")

    def _load_spacy(name: str) -> Any:
        class _NLP:
            def __call__(self, text: str) -> Any:
                class _Doc:
                    pass

                return _Doc()

        return _NLP()

    spacy.load = _load_spacy
    sys.modules["spacy"] = spacy


def _setup_seed_imports(seed_root: Path, data_dir: Path) -> tuple[Any, Any]:
    """Prepare sys.modules so the real SEED engine/goal modules import cleanly."""
    webshop_dir = seed_root / "agent_system/environments/env_package/webshop/webshop"
    engine_dir = webshop_dir / "web_agent_site/engine"

    # Pre-create the package hierarchy so file-based imports resolve.
    web_agent_site = types.ModuleType("web_agent_site")
    web_agent_site.engine = types.ModuleType("web_agent_site.engine")
    web_agent_site.engine.engine = types.ModuleType("web_agent_site.engine.engine")
    web_agent_site.engine.goal = types.ModuleType("web_agent_site.engine.goal")
    web_agent_site.engine.normalize = types.ModuleType("web_agent_site.engine.normalize")
    web_agent_site.utils = types.ModuleType("web_agent_site.utils")
    sys.modules["web_agent_site"] = web_agent_site
    sys.modules["web_agent_site.engine"] = web_agent_site.engine
    sys.modules["web_agent_site.engine.engine"] = web_agent_site.engine.engine
    sys.modules["web_agent_site.engine.goal"] = web_agent_site.engine.goal
    sys.modules["web_agent_site.engine.normalize"] = web_agent_site.engine.normalize
    sys.modules["web_agent_site.utils"] = web_agent_site.utils

    data_dir_str = str(data_dir)
    web_agent_site.utils.BASE_DIR = data_dir_str
    web_agent_site.utils.DEFAULT_FILE_PATH = os.path.join(data_dir_str, "items_shuffle_1000.json")
    web_agent_site.utils.DEFAULT_ATTR_PATH = os.path.join(data_dir_str, "items_ins_v2_1000.json")
    web_agent_site.utils.DEFAULT_REVIEW_PATH = os.path.join(data_dir_str, "reviews.json")
    web_agent_site.utils.HUMAN_ATTR_PATH = os.path.join(data_dir_str, "items_human_ins.json")
    web_agent_site.utils.FEAT_CONV = os.path.join(data_dir_str, "feat_conv.pt")
    web_agent_site.utils.FEAT_IDS = os.path.join(data_dir_str, "feat_ids.pt")

    def _random_idx(cum_weights: list[float]) -> int:
        import bisect
        pos = random.uniform(0, cum_weights[-1])
        idx = bisect.bisect(cum_weights, pos)
        return min(idx, len(cum_weights) - 2)

    web_agent_site.utils.random_idx = _random_idx

    # normalize_color is not used during goal construction, but the import must succeed.
    normalize_mod = types.ModuleType("web_agent_site.engine.normalize")
    normalize_mod.normalize_color = lambda c: c
    sys.modules["web_agent_site.engine.normalize"] = normalize_mod

    sys.path.insert(0, str(webshop_dir))
    sys.path.insert(0, str(engine_dir))

    goal_path = engine_dir / "goal.py"
    spec = importlib.util.spec_from_file_location("web_agent_site.engine.goal", str(goal_path))
    goal_mod = importlib.util.module_from_spec(spec)
    sys.modules["web_agent_site.engine.goal"] = goal_mod
    spec.loader.exec_module(goal_mod)  # type: ignore[union-attr]

    engine_path = engine_dir / "engine.py"
    spec = importlib.util.spec_from_file_location("web_agent_site.engine.engine", str(engine_path))
    engine_mod = importlib.util.module_from_spec(spec)
    sys.modules["web_agent_site.engine.engine"] = engine_mod
    spec.loader.exec_module(engine_mod)  # type: ignore[union-attr]

    return engine_mod, goal_mod


def _compute_session_draw() -> list[int]:
    import numpy as np

    selection = np.random.RandomState(VALIDATION_BASE_SEED).choice(
        np.arange(HELD_OUT_SESSION_MAX + 1), size=PANEL_SIZE, replace=False
    )
    return [int(x) for x in selection]


def _build_worker_goals(
    worker_seed: int,
    engine_mod: Any,
    goal_mod: Any,
    data_dir: Path,
    num_products: int,
) -> list[dict[str, Any]]:
    """Reproduce one worker's shuffled goal list exactly as SEED constructs it."""
    random.seed(worker_seed)
    all_products, _, product_prices, _ = engine_mod.load_products(
        str(data_dir / "items_shuffle_1000.json"),
        num_products=num_products,
        human_goals=False,
    )
    goals = goal_mod.get_synthetic_goals(all_products, product_prices)
    # Pinned SEED SimServer.__init__ re-seeds with worker_seed after goal
    # construction and before random.shuffle(self.goals).
    random.seed(worker_seed)
    random.shuffle(goals)
    return goals


def _hash_data_files(data_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"):
        path = data_dir / name
        if not path.is_file():
            raise SystemExit(f"missing WebShop data file: {path}")
        hashes[name] = _sha256_bytes(path.read_bytes())
    return hashes


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    _install_dependency_stubs()
    engine_mod, goal_mod = _setup_seed_imports(args.seed_root, args.data_dir)
    data_hashes = _hash_data_files(args.data_dir)

    sampled_session_indices = _compute_session_draw()
    if len(set(sampled_session_indices)) != PANEL_SIZE:
        raise SystemExit("session draw produced duplicates")

    tasks: list[dict[str, Any]] = []
    for slot, session_index in enumerate(sampled_session_indices):
        worker_seed = WORKER_SEED_OFFSET + slot
        goals = _build_worker_goals(
            worker_seed=worker_seed,
            engine_mod=engine_mod,
            goal_mod=goal_mod,
            data_dir=args.data_dir,
            num_products=args.num_products,
        )
        if len(goals) != EXPECTED_GOALS_TOTAL:
            raise SystemExit(
                f"slot {slot}: expected {EXPECTED_GOALS_TOTAL} synthetic goals, got {len(goals)}"
            )
        if not (HELD_OUT_SESSION_MIN <= session_index <= HELD_OUT_SESSION_MAX):
            raise SystemExit(f"slot {slot}: session_index {session_index} outside held-out range")
        goal = goals[session_index]
        instruction = str(goal.get("instruction_text", ""))
        if not instruction:
            raise SystemExit(f"slot {slot}: empty instruction_text")
        tasks.append(
            {
                "slot": slot,
                "task_id": f"webshop_seed_source_faithful_slot{slot:03d}_session{session_index:05d}",
                "seed_source_commit": args.seed_commit,
                "validation_base_seed": VALIDATION_BASE_SEED,
                "worker_seed": worker_seed,
                "sampled_session_index": session_index,
                "instruction": instruction,
                "instruction_sha256": _sha256_text(instruction),
                "asin": goal.get("asin"),
                "category": goal.get("category"),
                "webshop_data_identity": data_hashes,
            }
        )

    panel_fingerprint = [
        {
            "slot": t["slot"],
            "worker_seed": t["worker_seed"],
            "sampled_session_index": t["sampled_session_index"],
            "instruction_sha256": t["instruction_sha256"],
        }
        for t in tasks
    ]
    selected_panel_digest = _canonical_digest(panel_fingerprint)

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "design_id": args.design_id,
        "campaign_id": args.campaign_id,
        "title": "Source-faithful SEED WebShop first-validation 128-slot semantic panel",
        "created_utc": args.created_utc,
        "source_pins": {
            "seed_repository": "github.com/jinyangwu/SEED",
            "seed_commit": args.seed_commit,
            "public_launcher_env_seed": 0,
            "validation_base_seed": VALIDATION_BASE_SEED,
            "validation_batch_size": PANEL_SIZE,
            "heldout_session_index_min": HELD_OUT_SESSION_MIN,
            "heldout_session_index_max": HELD_OUT_SESSION_MAX,
            "worker_seed_formula": "1000 + slot",
        },
        "benchmark": {
            "name": "WebShop small (source-faithful first-validation panel)",
            "environment": "WebAgentTextEnv-v0",
            "num_products": int(args.num_products),
            "human_goals": False,
            "goals_total": EXPECTED_GOALS_TOTAL,
            "data_files": data_hashes,
            "session_draw": "np.random.RandomState(1000).choice(np.arange(500), size=128, replace=False)",
            "selected_panel_digest": selected_panel_digest,
        },
        "tasks": tasks,
    }
    manifest["manifest_content_sha256"] = _canonical_digest(
        {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--seed-commit", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--design-id", default="seed-webshop-public-code-reproduction-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-products", type=int, default=1000)
    parser.add_argument("--created-utc", required=True)
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    file_sha = _sha256_bytes(args.output.read_bytes())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "panel_size": len(manifest["tasks"]),
                "selected_panel_digest": manifest["benchmark"]["selected_panel_digest"],
                "manifest_content_sha256": manifest["manifest_content_sha256"],
                "manifest_file_sha256": file_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
