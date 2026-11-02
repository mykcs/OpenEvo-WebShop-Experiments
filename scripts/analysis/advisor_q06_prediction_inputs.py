#!/usr/bin/env python3
"""CPU-only eligibility checks for frozen Q06 features/splits and scoring cost.

Input is normalized evidence, not a model runner. The allowlist and event phase
are deliberately strict: an upstream adapter must prove feature provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

Q06_CONTEXT_SCHEMA = "openevo.q06.pre-action-state.v1"
Q06_STATE_TYPE_DEFINITION = (
    "q06-available-actions-v1: has_search_bar => search; else clickables contain "
    "buy now or start with description/features => product; else results"
)
S3_REFERENCE_CELL = "bounded-r40"
S3_REFERENCE_LINEAGE = "/workspace/runs/bounded-effective-state-gdr-recovery-1e7c4952-20260916/off"
S3_REFERENCE_CHECKPOINT = "after_040_rounds"
S3_QUOTAS = ((5, 8), (9, 16), (17, 32), (33, None))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_s3_context(context: Any, *, step: int, lineage_id: str) -> dict:
    """Validate the retained pre-action payload and recompute each content digest."""
    if not isinstance(context, dict) or context.get("schema") != Q06_CONTEXT_SCHEMA:
        raise ValueError("S3 source lacks the frozen Q06 pre-action context schema")
    if context.get("phase") != "pre_action" or context.get("panel_role") != "development":
        raise ValueError("S3 bank accepts development pre-action snapshots only")
    if context.get("lineage_id") != lineage_id or context.get("step_index") != step:
        raise ValueError("pre-action identity does not bind the raw record")
    if context.get("state_type_definition") != Q06_STATE_TYPE_DEFINITION:
        raise ValueError("state_type definition differs from the frozen Q06 definition")
    state_type = context.get("state_type")
    if state_type not in {"search", "product", "results"}:
        raise ValueError("state_type is not a frozen Q06 category")
    for key in ("observation", "prompt", "rendered_history"):
        if not isinstance(context.get(key), str):
            raise ValueError(f"exact {key} bytes are required")
    if not isinstance(context.get("history"), list):
        raise ValueError("exact history list is required")
    for key in ("available_actions", "action_schema"):
        if not isinstance(context.get(key), dict):
            raise ValueError(f"exact {key} object is required")
    actions = context["available_actions"]
    clickables = [str(value).strip().lower() for value in (actions.get("clickables") or [])]
    expected_state = (
        "search" if actions.get("has_search_bar") else
        "product" if any(value == "buy now" or value.startswith(("description", "features"))
                         for value in clickables) else "results"
    )
    if state_type != expected_state:
        raise ValueError("state_type does not match the frozen pre-action available-actions mapping")
    token_ids = context.get("action_context_token_ids")
    if not isinstance(token_ids, list) or not token_ids or any(type(v) is not int for v in token_ids):
        raise ValueError("exact constrained-action token IDs are required")
    content = {
        "observation_sha256": context["observation"].encode("utf-8"),
        "prompt_sha256": context["prompt"].encode("utf-8"),
        "history_sha256": _canonical_bytes(context["history"]),
        "rendered_history_sha256": context["rendered_history"].encode("utf-8"),
        "available_actions_sha256": _canonical_bytes(context["available_actions"]),
        "action_schema_sha256": _canonical_bytes(context["action_schema"]),
        "action_context_token_ids_sha256": _canonical_bytes(token_ids),
    }
    for digest_key, raw in content.items():
        if context.get(digest_key) != _sha256(raw):
            raise ValueError(f"{digest_key} does not bind retained exact payload")
    if context.get("action_context_token_count") != len(token_ids):
        raise ValueError("action-context token count mismatch")
    if context.get("action_context_phase") != "pre_action_action_channel_prefix_before_action_tokens":
        raise ValueError("action-context tokens are not the frozen pre-action prefix")
    if context.get("token_count_definition") != (
        "prompt_tokens and history_tokens are independently tokenized exact UTF-8 source strings "
        "using the pinned runtime tokenizer with add_special_tokens=False"
    ):
        raise ValueError("token count provenance is missing or changed")
    if context.get("event_order_definition") != (
        "0 exact state snapshot; 1 generated uncertainty and click scorer diagnostics; "
        "2 action commit to environment; 3 environment transition call"
    ):
        raise ValueError("event-order definition is missing or changed")
    for name in ("prompt_tokens", "history_tokens", "completed_rounds"):
        if type(context.get(name)) is not int or context[name] < 0:
            raise ValueError(f"explicit nonnegative {name} required")
    if context["history_tokens"] > context["prompt_tokens"]:
        raise ValueError("history token count exceeds prompt token count")
    events = [context.get(k) for k in ("feature_event_index", "decision_event_index", "action_event_index")]
    if any(type(v) is not int or v < 0 for v in events) or not events[0] < events[1] <= events[2]:
        raise ValueError("pre-action event order is invalid")
    return context


def _stratum(count: int) -> int | None:
    for index, (lower, upper) in enumerate(S3_QUOTAS):
        if count >= lower and (upper is None or count <= upper):
            return index
    return None


def select_s3_bank(source_spec: dict, source_bytes: bytes) -> dict:
    """Select from one retained bounded-r40 source, without reading outcome fields."""
    if source_spec.get("cell_id") != S3_REFERENCE_CELL:
        raise ValueError("S3 reference actor is frozen to bounded-r40")
    lineage = source_spec.get("lineage_id")
    checkpoint = source_spec.get("checkpoint_id")
    if lineage != S3_REFERENCE_LINEAGE or checkpoint != S3_REFERENCE_CHECKPOINT:
        raise ValueError("S3 source identity differs from the frozen bounded-r40 actor")
    expected_source_sha = _require_sha(source_spec.get("attempts_sha256"), "attempts_sha256")
    actual_source_sha = _sha256(source_bytes)
    if actual_source_sha != expected_source_sha:
        raise ValueError("attempts source bytes do not match the supplied exact digest")

    candidates: list[list[dict]] = [[] for _ in S3_QUOTAS]
    rejected = 0
    seen_decisions = set()
    for record_number, line_bytes in enumerate(source_bytes.splitlines(), 1):
        if not line_bytes.strip():
            continue
        attempt = json.loads(line_bytes)
        if not isinstance(attempt, dict):
            raise ValueError("attempt JSONL records must be objects")
        # Only identity, validity, panel, and pre-action trace fields are consulted here.
        if attempt.get("engineering_invalid") is not False or attempt.get("panel_role") != "development":
            rejected += 1
            continue
        if attempt.get("lineage_id") != lineage or attempt.get("checkpoint_id") != checkpoint:
            raise ValueError("attempt lineage/checkpoint does not match the source contract")
        task_identity = attempt.get("task_identity")
        episode = attempt.get("episode")
        trace = episode.get("trace") if isinstance(episode, dict) else None
        if not isinstance(task_identity, str) or not task_identity or not isinstance(trace, list):
            raise ValueError("attempt identity or trace is missing")
        for trace_row in trace:
            if not isinstance(trace_row, dict):
                raise ValueError("trace rows must be objects")
            step = trace_row.get("step")
            if type(step) is not int or step < 0:
                raise ValueError("S3 decision step must be a nonnegative integer")
            context = _validate_s3_context(trace_row.get("q06_pre_action"), step=step, lineage_id=lineage)
            diag = trace_row.get("generation_diagnostics")
            click = diag.get("click_choice") if isinstance(diag, dict) else None
            if not isinstance(click, dict) or click.get("phase") != "pre_action":
                raise ValueError("pre-action click-choice diagnostics are missing")
            n = click.get("candidate_count")
            decision_event = context["decision_event_index"]
            click_event = click.get("source_event_index")
            if type(n) is not int or n < 0 or type(click_event) is not int or not 0 <= click_event < decision_event:
                raise ValueError("click-choice count/event provenance is invalid")
            decision_identity = (task_identity, step)
            if decision_identity in seen_decisions:
                raise ValueError("duplicate task/step decision identity in S3 source")
            seen_decisions.add(decision_identity)
            stratum = _stratum(n)
            if stratum is None:
                continue
            order_key = _sha256(f"q06-bank-v1|{S3_REFERENCE_CELL}|{task_identity}|{step}".encode("utf-8"))
            candidates[stratum].append({
                "selection_sha256": order_key,
                "task_identity": task_identity,
                "step": step,
                "choice_count": n,
                "source_record_number": record_number,
                "source_record_sha256": _sha256(line_bytes),
                "context": context,
            })
    selected: list[dict] = []
    coverage = []
    for index, (quota, (lower, upper)) in enumerate(zip((64,) * len(S3_QUOTAS), S3_QUOTAS)):
        ordered = sorted(candidates[index], key=lambda item: (item["selection_sha256"], item["task_identity"], item["step"]))
        chosen = ordered[:quota]
        selected.extend(chosen)
        coverage.append({"stratum": [lower, upper], "eligible": len(ordered), "selected": len(chosen), "quota": quota})
    full = all(item["selected"] == item["quota"] for item in coverage)
    return {
        "schema": "openevo.q06.s3-preaction-bank-selection.v1",
        "status": "S3_BANK_SELECTED" if full else "PARTIAL",
        "reference_actor": {"cell_id": S3_REFERENCE_CELL, "lineage_id": lineage, "checkpoint_id": checkpoint},
        "selection": "SHA256(q06-bank-v1|cell_id|task_identity|step), ascending within strata; no refill",
        "source_attempts_sha256": actual_source_sha,
        "coverage": coverage,
        "selected_count": len(selected),
        "required_count": 256,
        "rejected_non_development_or_invalid_records": rejected,
        "selected": selected,
        "model_calls": 0, "environment_calls": 0, "new_episodes": 0,
        "claim": "outcome-blind input bank selection only; no cross-checkpoint scoring or predictive result",
    }

FEATURES = frozenset({
    "click_entropy_normalized", "reasoning_entropy_mean", "action_entropy_mean",
    "choice_count", "step_index", "prompt_tokens", "history_tokens", "round",
})


def validate_rows(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("nonempty development rows required")
    identities = set()
    for row in rows:
        if row.get("panel_role") != "development" or row.get("feature_phase") != "pre_action":
            raise ValueError("development/pre_action provenance required")
        if row.get("target_phase") != "post_action_or_later":
            raise ValueError("target must follow the feature decision")
        for key in ("task_id", "lineage_id", "checkpoint_id", "feature_provenance_sha256"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"missing identity/provenance: {key}")
        sha = row["feature_provenance_sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("feature provenance must bind exact source bytes")
        steps = [row.get(k) for k in ("decision_step", "feature_observed_step", "target_observed_step", "completed_rounds")]
        if any(type(v) is not int or v < 0 for v in steps):
            raise ValueError("nonnegative integer steps/round required")
        decision, observed, target_step, _ = steps
        if observed > decision or target_step < decision:
            raise ValueError("future feature or backwards target leaks outcome")
        identity = tuple(row[k] for k in ("lineage_id", "checkpoint_id", "task_id", "decision_step"))
        if identity in identities:
            raise ValueError("duplicate decision identity")
        identities.add(identity)
        features = row.get("features")
        if not isinstance(features, dict) or not features or set(features) - FEATURES:
            raise ValueError("unregistered feature (reward, success and episode totals are forbidden)")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in features.values()):
            raise ValueError("features must be finite numeric scalars")
        if "step_index" in features and features["step_index"] != decision:
            raise ValueError("step feature does not bind this decision")
        if "round" in features and features["round"] != row["completed_rounds"]:
            raise ValueError("round feature does not bind this checkpoint")
        value = row.get("target")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("held-out task score target must be finite in [0,1]")


def split_rows(rows: list[dict], spec: dict) -> dict:
    """Produce indices from a prospectively frozen split, never select by target."""
    validate_rows(rows)
    mode = spec.get("mode")
    if mode in ("task", "lineage"):
        field = "task_id" if mode == "task" else "lineage_id"
        heldout = spec.get("heldout_ids")
        if not isinstance(heldout, list) or not heldout or len(set(heldout)) != len(heldout):
            raise ValueError("explicit unique heldout IDs required")
        heldout = set(heldout)
        if not heldout <= {r[field] for r in rows}:
            raise ValueError("unknown heldout ID")
        train = [i for i, r in enumerate(rows) if r[field] not in heldout]
        test = [i for i, r in enumerate(rows) if r[field] in heldout]
        if {rows[i][field] for i in train} & {rows[i][field] for i in test}:
            raise ValueError("group leakage")
    elif mode == "time":
        cutoff = spec.get("train_max_round")
        if type(cutoff) is not int or cutoff < 0:
            raise ValueError("frozen train_max_round required")
        train = [i for i, r in enumerate(rows) if r["completed_rounds"] <= cutoff]
        test = [i for i, r in enumerate(rows) if r["completed_rounds"] > cutoff]
    else:
        raise ValueError("split mode must be task, time or lineage")
    if not train or not test:
        raise ValueError("both train and heldout partitions must be nonempty")
    return {
        "status": "CPU_INPUT_ELIGIBILITY_PASS", "split": spec,
        "train_indices": train, "heldout_indices": test,
        "model_calls": 0, "environment_calls": 0, "final_panel_access": 0,
        "claim": "partition eligibility only; no prediction performance or causal result",
    }


def audit_candidate_cost(receipt: dict) -> dict:
    if receipt.get("classification") != "derived_cpu_only_not_observed_forward_calls":
        raise ValueError("this reader requires explicitly derived scoring cost")
    if receipt.get("model_calls") != 0 or receipt.get("environment_calls") != 0:
        raise ValueError("cost derivation must be CPU-only")
    arms = receipt.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ValueError("cost receipt has no arms")
    output = {}
    for arm, value in arms.items():
        rows = value.get("state_receipts", [])
        groups = value.get("groups", {})
        if not rows or len(rows) != value.get("states_scored"):
            raise ValueError("cost state inventory mismatch")
        sums = {"estimated_forward_calls": 0, "estimated_input_tokens": 0}
        for row in rows:
            for key in sums:
                n = row.get(key)
                if type(n) is not int or n < 0:
                    raise ValueError("cost must be a nonnegative integer")
                sums[key] += n
        if any(sum(g.get(k, -1) for g in groups.values()) != n for k, n in sums.items()):
            raise ValueError("group and per-state cost totals differ")
        if sum(g.get("states", -1) for g in groups.values()) != len(rows):
            raise ValueError("group state inventory mismatch")
        output[arm] = {**sums, "states": len(rows)}
    return {
        "classification": "derived_not_observed", "arms": output,
        "total_estimated_forward_calls": sum(v["estimated_forward_calls"] for v in output.values()),
        "total_estimated_input_tokens": sum(v["estimated_input_tokens"] for v in output.values()),
        "model_calls_by_this_audit": 0, "environment_calls_by_this_audit": 0,
        "claim": "diagnostic scoring has model cost even when additional episodes equal zero",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("split", "cost", "s3-bank"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload_bytes = args.input.read_bytes()
    payload = json.loads(payload_bytes)
    if args.mode == "split":
        if args.split is None:
            parser.error("split mode requires --split")
        result = split_rows(payload["rows"], json.loads(args.split.read_text(encoding="utf-8")))
    elif args.mode == "cost":
        result = audit_candidate_cost(payload)
    else:
        sources = payload.get("sources")
        if not isinstance(sources, list) or len(sources) != 1:
            parser.error("s3-bank requires exactly one source spec for frozen bounded-r40")
        source = sources[0]
        if args.output is None:
            parser.error("s3-bank requires --output for durable selection evidence")
        source_path = Path(source["path"])
        source_bytes = source_path.read_bytes()
        result = select_s3_bank(source, source_bytes)
        result["source_path"] = str(source_path)
        result["selector_input_manifest_sha256"] = _sha256(payload_bytes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        if result["status"] == "PARTIAL":
            parser.exit(2, f"PARTIAL bank written to {args.output}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
