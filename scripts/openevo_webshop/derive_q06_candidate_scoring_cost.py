#!/usr/bin/env python3
"""Derive Q17 click-trie scorer work from sealed traces and pinned tokenizer.

This CPU-only accounting estimates forwards/input tokens from the frozen
scorer's trie algorithm. It does not load a model or execute a model call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TOKENIZER_SHA = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "3c0884a30471f4f542dc89630f62a380bb70a341fafda826136a7be921fec7ea",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
}
SCORER_SOURCE_SHA = "fdbe3b69296c39ff3d1a855398c4ea69ddcde97144f8a5efa9ce50c6e3808c05"
ACTION_RUNTIME_SOURCE_SHA = "b5df331a51117eef216da8e45270dcd06920570db2035e1ea32a1bc438413fd0"
ACTION_CLOSE = "</action>"
ACTION_OPEN = "<action>"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def trie_nonroot_internal_nodes(targets: list[list[int]]) -> int:
    root: dict = {"children": {}}
    for target in targets:
        node = root
        for token_id in target:
            node = node["children"].setdefault(int(token_id), {"children": {}})
    def count(node: dict, is_root: bool = False) -> int:
        n = int(bool(node["children"]) and not is_root)
        return n + sum(count(child) for child in node["children"].values())
    return count(root, True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bounded", type=Path, required=True)
    p.add_argument("--beta", type=Path, required=True)
    p.add_argument("--gdr", type=Path, help="optional sealed Q06 historical context attempts")
    p.add_argument("--tokenizer-root", type=Path, required=True)
    p.add_argument("--q17-scorer-source", type=Path, required=True)
    p.add_argument("--action-runtime-source", type=Path, required=True)
    a = p.parse_args()
    actual_tokenizer_sha = {name: sha256(a.tokenizer_root / name) for name in TOKENIZER_SHA}
    if actual_tokenizer_sha != TOKENIZER_SHA:
        raise SystemExit("tokenizer asset digest mismatch")
    if sha256(a.q17_scorer_source) != SCORER_SOURCE_SHA:
        raise SystemExit("Q17 scorer source digest mismatch")
    if sha256(a.action_runtime_source) != ACTION_RUNTIME_SOURCE_SHA:
        raise SystemExit("action-runtime source digest mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(a.tokenizer_root), local_files_only=True, trust_remote_code=False
    )
    action_open_ids = tokenizer(ACTION_OPEN, add_special_tokens=False)["input_ids"]
    eos_id = int(tokenizer.eos_token_id)
    arms = {}
    inputs = [("bounded", a.bounded), ("beta", a.beta)]
    if a.gdr is not None:
        inputs.append(("gdr", a.gdr))
    for arm, path in inputs:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 64:
            raise SystemExit(f"{arm}: expected 64 sealed rows")
        state_receipts = []
        for row in rows:
            for trace in row["episode"]["trace"]:
                schema = trace["action_channel"]["compiled_action_schema"]
                schema_payload = {
                    "finite_actions": schema["finite_actions"],
                    "search_allowed": bool(schema["search_allowed"]),
                }
                if canonical_sha(schema_payload) != schema["sha256"]:
                    raise SystemExit(f"{arm}/{row['task_identity']}: action schema hash mismatch")
                click_actions = sorted(
                    str(x) for x in schema["finite_actions"]
                    if str(x).lower().startswith("click[")
                )
                if not click_actions:
                    continue
                targets = [
                    [int(x) for x in tokenizer(action + ACTION_CLOSE, add_special_tokens=False)["input_ids"]]
                    + [eos_id]
                    for action in click_actions
                ]
                internal = trie_nonroot_internal_nodes(targets)
                click = trace["generation_diagnostics"]["click_choice"]
                destinations = [x for x in click_actions if x.lower() != "click[search]"]
                if len(destinations) != int(click["candidate_count"]):
                    raise SystemExit(f"{arm}/{row['task_identity']}: click candidate count mismatch")
                sentinel_admissible = click.get("click_search_sentinel", {}).get("status") == "ADMISSIBLE"
                if ("click[search]" in click_actions) != sentinel_admissible:
                    raise SystemExit(f"{arm}/{row['task_identity']}: sentinel admissibility mismatch")
                context_tokens = (
                    int(trace["runtime_input_tokens"])
                    + int(trace["generation_diagnostics"]["observed_pre_action_model_tokens"])
                    + len(action_open_ids)
                )
                state_receipts.append({
                    "task_identity": row["task_identity"],
                    "probe_ordinal": row["probe_ordinal"],
                    "step": trace["step"],
                    "action_schema_sha256": schema["sha256"],
                    "click_action_count": len(click_actions),
                    "click_action_set_sha256": canonical_sha(click_actions),
                    "destination_count": len(destinations),
                    "sentinel_admissible": sentinel_admissible,
                    "classification": "real_destination" if destinations else "sentinel_only",
                    "runtime_input_tokens": int(trace["runtime_input_tokens"]),
                    "observed_pre_action_model_tokens": int(trace["generation_diagnostics"]["observed_pre_action_model_tokens"]),
                    "action_opener_tokens": len(action_open_ids),
                    "root_context_tokens": context_tokens,
                    "nonroot_internal_trie_nodes": internal,
                    "estimated_forward_calls": 1 + internal,
                    "estimated_input_tokens": context_tokens + internal,
                })
        groups = {}
        for kind in ("real_destination", "sentinel_only"):
            selected = [x for x in state_receipts if x["classification"] == kind]
            groups[kind] = {
                "states": len(selected),
                "estimated_forward_calls": sum(x["estimated_forward_calls"] for x in selected),
                "estimated_input_tokens": sum(x["estimated_input_tokens"] for x in selected),
            }
        arms[arm] = {
            "input_attempts_basename": path.name,
            "input_attempts_sha256": sha256(path),
            "states_scored": len(state_receipts),
            "groups": groups,
            "state_receipts": state_receipts,
        }
    print(json.dumps({
        "schema": "openevo.q06.candidate-scoring-cost-derivation.v1",
        "classification": "derived_cpu_only_not_observed_forward_calls",
        "model_loaded": False,
        "model_calls": 0,
        "environment_calls": 0,
        "source_identity": {
            "q17_scorer_path": str(a.q17_scorer_source),
            "q17_scorer_sha256": sha256(a.q17_scorer_source),
            "action_runtime_path": str(a.action_runtime_source),
            "action_runtime_sha256": sha256(a.action_runtime_source),
            "action_close": ACTION_CLOSE,
            "action_open": ACTION_OPEN,
        },
        "tokenizer": {
            "root": str(a.tokenizer_root),
            "class": type(tokenizer).__name__,
            "eos_token_id": eos_id,
            "transformers_version": __import__("transformers").__version__,
            "assets_sha256": actual_tokenizer_sha,
            "action_open_token_count": len(action_open_ids),
        },
        "scorer_counting_rule": "one root forward per nonempty click target trie plus one forward per nonroot internal trie node; root input tokens are runtime_input_tokens + observed_pre_action_model_tokens + encoded <action> length; each additional forward consumes its one branch token while prior context is in cache",
        "arms": arms,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
