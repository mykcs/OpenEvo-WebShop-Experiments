#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

MAX_STEPS = 15
STEP_SEED_STRIDE = 100_000
FROZEN_PANEL_MANIFEST_REL = Path(
    "docs/evidence/q17-validation-curve-20260923/"
    "coarse-manifest-q17-directapply-validation-curve-202609231600.json"
)
FROZEN_PANEL_MANIFEST_SHA256 = "2d5d1620065a15840ae20fed324038312aec74b0895e13c9d283177928bf1a3b"
FROZEN_PANEL_SHA256 = "4033523c01697576b7f9ce8cddc3a63cf6c1514ef5554e01a8c82b59ee62d478"
FROZEN_PANEL_SIZE = 64
FROZEN_PANEL_SCHEMA = "openevo.q17.validation-curve.v1"  # legacy-revision-compat: frozen historical manifest schema


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_full_panel_manifest(
    path: Path, manifest: dict[str, Any], *, task_limit: int | None = None,
    expected_manifest_sha256: str = FROZEN_PANEL_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Require the byte-pinned complete Q17 64-task panel before model loading."""
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            f"frozen 64-task manifest SHA mismatch: {actual_sha256} != {expected_manifest_sha256}"
        )
    if task_limit is not None:
        raise RuntimeError("mechanism probe requires the complete frozen 64-task panel; --task-limit is forbidden")
    if manifest.get("schema") != FROZEN_PANEL_SCHEMA:
        raise RuntimeError("frozen mechanism panel manifest schema drift")
    if int(manifest.get("panel_size", -1)) != FROZEN_PANEL_SIZE:
        raise RuntimeError("frozen mechanism panel must declare exactly 64 tasks")
    if manifest.get("panel_sha256") != FROZEN_PANEL_SHA256:
        raise RuntimeError("frozen mechanism panel identity drift")
    if manifest.get("final_panel_access_allowed") is not False:
        raise RuntimeError("frozen mechanism panel must forbid final-panel access")
    if manifest.get("formal_task_consumption_allowed") is not False:
        raise RuntimeError("frozen mechanism panel must forbid formal-task consumption")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != FROZEN_PANEL_SIZE:
        raise RuntimeError("frozen mechanism manifest must contain all 64 task rows")
    ordinals = [int(task.get("probe_ordinal", -1)) for task in tasks]
    if ordinals != list(range(FROZEN_PANEL_SIZE)):
        raise RuntimeError("frozen mechanism panel ordinals must be exactly 0..63 in order")
    identities = [str(task.get("task_identity", "")) for task in tasks]
    if any(not identity for identity in identities) or len(set(identities)) != FROZEN_PANEL_SIZE:
        raise RuntimeError("frozen mechanism panel task identities must be present and unique")
    return {
        "file_sha256": actual_sha256,
        "panel_sha256": FROZEN_PANEL_SHA256,
        "panel_size": FROZEN_PANEL_SIZE,
        "task_identity_sha256": hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode()
        ).hexdigest(),
        "task_identities": identities,
    }


def build_probe_receipt(
    script_path: Path, task_panel: dict[str, Any], *, checkpoint_id: str,
    checkpoint_manifest_sha256: str, base_model: str, base_model_revision: str,
    instrumentation: str, task_count: int,
) -> dict[str, Any]:
    return {
        "probe_script_sha256": sha256_file(script_path),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "task_panel_manifest_sha256": task_panel["file_sha256"],
        "task_panel_task_identity_sha256": task_panel["task_identity_sha256"],
        "panel_sha256": task_panel["panel_sha256"],
        "task_count": int(task_count),
        "checkpoint_id": checkpoint_id,
        "instrumentation": instrumentation,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def top_two_logit_margin(logits: list[float]) -> float | None:
    """Return the top1-top2 margin, or None when the constraint forces one token."""
    finite = sorted((float(x) for x in logits if math.isfinite(float(x))), reverse=True)
    return finite[0] - finite[1] if len(finite) >= 2 else None


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _q06_state_type(available_actions: dict[str, Any]) -> str:
    """Frozen pre-action state mapping; depends only on current legal actions."""
    clickables = [str(value).strip().lower() for value in (available_actions.get("clickables") or [])]
    if available_actions.get("has_search_bar"):
        return "search"
    if any(value == "buy now" or value.startswith("description") or value.startswith("features")
           for value in clickables):
        return "product"
    return "results"


def _q06_pre_action_context(
    *, step_index: int, completed_rounds: int, lineage_id: str,
    panel_role: str, observation: Any, prompt: str, history: list[str],
    rendered_history: str, available_actions: dict[str, Any], action_schema: Any,
    action_context_token_ids: list[int], tokenizer: Any,
) -> dict[str, Any]:
    """Capture byte-bound Q06 inputs without feeding them back into generation."""
    if type(step_index) is not int or step_index < 0:
        raise ValueError("Q06 step_index must be a nonnegative integer")
    if type(completed_rounds) is not int or completed_rounds < 0:
        raise ValueError("Q06 completed_rounds must be an explicit nonnegative integer")
    if not isinstance(lineage_id, str) or not lineage_id.strip():
        raise ValueError("Q06 lineage_id must be explicitly supplied")
    if panel_role != "development":
        raise ValueError("Q06 pre-action diagnostics are restricted to the development panel")
    if not isinstance(observation, str):
        raise TypeError("Q06 observation must be the exact pre-action text")
    if not isinstance(prompt, str) or not isinstance(rendered_history, str):
        raise TypeError("Q06 prompt and rendered history must be exact strings")
    if not isinstance(available_actions, dict) or not isinstance(action_schema, dict):
        raise TypeError("Q06 available actions and compiled action schema must be objects")
    if not action_context_token_ids or any(type(value) is not int for value in action_context_token_ids):
        raise ValueError("Q06 action context token IDs must be captured before action generation")

    observation_bytes = observation.encode("utf-8")
    prompt_bytes = prompt.encode("utf-8")
    history_bytes = _canonical_json_bytes(history)
    rendered_history_bytes = rendered_history.encode("utf-8")
    available_actions_bytes = _canonical_json_bytes(available_actions)
    action_schema_bytes = _canonical_json_bytes(action_schema)
    action_context_bytes = _canonical_json_bytes(action_context_token_ids)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    history_tokens = len(tokenizer(rendered_history, add_special_tokens=False)["input_ids"])
    if history_tokens > prompt_tokens:
        raise ValueError("Q06 separately tokenized history exceeds prompt token count")

    return {
        "schema": "openevo.q06.pre-action-state.v1",
        "phase": "pre_action",
        "step_index": step_index,
        "completed_rounds": completed_rounds,
        "lineage_id": lineage_id,
        "panel_role": panel_role,
        "state_type": _q06_state_type(available_actions),
        "state_type_definition": "q06-available-actions-v1: has_search_bar => search; else clickables contain buy now or start with description/features => product; else results",
        "feature_event_index": 0,
        "decision_event_index": 2,
        "action_event_index": 3,
        "event_order_definition": "0 exact state snapshot; 1 generated uncertainty and click scorer diagnostics; 2 action commit to environment; 3 environment transition call",
        "observation": observation,
        "observation_sha256": hashlib.sha256(observation_bytes).hexdigest(),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "history": history,
        "history_sha256": hashlib.sha256(history_bytes).hexdigest(),
        "rendered_history": rendered_history,
        "rendered_history_sha256": hashlib.sha256(rendered_history_bytes).hexdigest(),
        "history_tokens": history_tokens,
        "available_actions": available_actions,
        "available_actions_sha256": hashlib.sha256(available_actions_bytes).hexdigest(),
        "action_schema": action_schema,
        "action_schema_sha256": hashlib.sha256(action_schema_bytes).hexdigest(),
        "action_context_token_ids": action_context_token_ids,
        "action_context_token_ids_sha256": hashlib.sha256(action_context_bytes).hexdigest(),
        "action_context_token_count": len(action_context_token_ids),
        "action_context_phase": "pre_action_action_channel_prefix_before_action_tokens",
        "token_count_definition": "prompt_tokens and history_tokens are independently tokenized exact UTF-8 source strings using the pinned runtime tokenizer with add_special_tokens=False",
    }


def build_token_trie(targets: dict[str, list[int]]) -> dict[str, Any]:
    """Build one shared-prefix trie for complete constrained action token paths."""
    root: dict[str, Any] = {"children": {}, "actions": []}
    for action, tokens in sorted(targets.items()):
        node = root
        for token_id in tokens:
            node = node["children"].setdefault(
                int(token_id), {"children": {}, "actions": []}
            )
        node["actions"].append(action)
    return root


def summarize_click_log_probabilities(
    action_log_probabilities: dict[str, float], *, selected_action: str | None = None,
    top_k: int = 3, sentinel_action: str = "click[search]",
) -> dict[str, Any]:
    """Summarize real destinations separately from the click[search] sentinel."""
    nonfinite_actions = [
        action for action, logp in action_log_probabilities.items()
        if not math.isfinite(float(logp))
    ]
    if nonfinite_actions:
        raise ValueError(
            "click-destination entropy is undefined for non-finite candidate log probabilities: "
            + ", ".join(sorted(nonfinite_actions))
        )
    sentinel_logp = action_log_probabilities.get(sentinel_action)
    destination_log_probabilities = {
        action: float(logp) for action, logp in action_log_probabilities.items()
        if action != sentinel_action
    }
    sentinel_summary = {
        "status": "ADMISSIBLE" if sentinel_logp is not None else "NOT_ADMISSIBLE",
        "action": sentinel_action,
        "log_probability_under_constrained_decoder": (
            float(sentinel_logp) if sentinel_logp is not None else None
        ),
        "conditional_probability_given_click_family": None,
        "selected": selected_action == sentinel_action,
    }
    if sentinel_logp is not None:
        max_click_logp = max(float(x) for x in action_log_probabilities.values())
        click_log_normalizer = max_click_logp + math.log(
            sum(math.exp(float(x) - max_click_logp) for x in action_log_probabilities.values())
        )
        sentinel_summary["conditional_probability_given_click_family"] = math.exp(
            float(sentinel_logp) - click_log_normalizer
        )
    if not destination_log_probabilities:
        return {
            "status": "NO_CLICK_DESTINATIONS",
            "candidate_count": 0,
            "entropy_nats": None,
            "hhi": None,
            "top_k": int(top_k),
            "top_k_mass": None,
            "selected_action": selected_action,
            "selected_action_log_probability": None,
            "selected_action_conditional_probability": None,
            "candidates": [],
            "click_search_sentinel": sentinel_summary,
        }
    max_logp = max(destination_log_probabilities.values())
    log_normalizer = max_logp + math.log(
        sum(math.exp(x - max_logp) for x in destination_log_probabilities.values())
    )
    candidates = [
        {
            "action": action,
            "log_probability_under_constrained_decoder": float(logp),
            "conditional_probability_given_click": math.exp(float(logp) - log_normalizer),
        }
        for action, logp in sorted(destination_log_probabilities.items())
    ]
    probabilities = [float(x["conditional_probability_given_click"]) for x in candidates]
    ordered_probabilities = sorted(probabilities, reverse=True)
    selected_logp = destination_log_probabilities.get(selected_action) if selected_action else None
    selected_probability = (
        math.exp(float(selected_logp) - log_normalizer) if selected_logp is not None else None
    )
    return {
        "status": "PASS",
        "candidate_count": len(candidates),
        "entropy_nats": -sum(p * math.log(p) for p in probabilities if p > 0),
        "hhi": sum(p * p for p in probabilities),
        "top_k": int(top_k),
        "top_k_mass": sum(ordered_probabilities[: max(0, int(top_k))]),
        "selected_action": selected_action,
        "selected_action_log_probability": (
            float(selected_logp) if selected_logp is not None else None
        ),
        "selected_action_conditional_probability": selected_probability,
        "candidates": candidates,
        "click_search_sentinel": sentinel_summary,
    }


class GenerateObserver:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.original = model.generate
        self.calls: list[dict[str, Any]] = []
        self.action_context_input_ids: list[int] | None = None
        self.action_constraint: Any | None = None

    def install(self) -> None:
        self.model.generate = self._wrapped

    def close(self) -> None:
        self.model.generate = self.original

    def mark(self) -> int:
        return len(self.calls)

    def since(self, mark: int) -> list[dict[str, Any]]:
        return self.calls[mark:]

    def _wrapped(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        call_kwargs = dict(kwargs)
        input_ids = call_kwargs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("diagnostic observer requires explicit input_ids")
        processors = call_kwargs.get("logits_processor") or []
        action_constraint = None
        for processor in processors:
            constraint = getattr(processor, "constraint", None)
            if constraint is not None and hasattr(constraint, "finite_targets"):
                action_constraint = constraint
                break
        if action_constraint is None:
            self.action_context_input_ids = None
            self.action_constraint = None
        else:
            # Bind the actual action-channel prefix before the model samples any
            # action token. The click scorer reuses this same frozen prefix.
            self.action_context_input_ids = [int(x) for x in input_ids[0].tolist()]
            self.action_constraint = action_constraint
        call_kwargs["return_dict_in_generate"] = True
        call_kwargs["output_scores"] = True
        output = self.original(*args, **call_kwargs)
        sequences = output.sequences
        scores = list(output.scores or [])
        n_input = int(input_ids.shape[1])
        generated_ids = sequences[0, n_input:n_input + len(scores)]
        entropies: list[float] = []
        nlls: list[float] = []
        top_two_margins: list[float | None] = []
        for score, token_id in zip(scores, generated_ids.tolist()):
            logits = score[0].float()
            entropy = torch.distributions.Categorical(logits=logits).entropy()
            log_probs = torch.log_softmax(logits, dim=-1)
            entropies.append(float(entropy.item()))
            nlls.append(float((-log_probs[int(token_id)]).item()))
            legal_logits = logits[torch.isfinite(logits)]
            top_two_margins.append(top_two_logit_margin(legal_logits.tolist()))
        self.calls.append({
            "generated_token_count": len(scores),
            "entropy_sum_nats": float(sum(entropies)),
            "entropy_mean_nats": statistics.mean(entropies) if entropies else None,
            "entropy_p90_nats": pctl(entropies, 0.90),
            "chosen_nll_sum_nats": float(sum(nlls)),
            "chosen_nll_mean_nats": statistics.mean(nlls) if nlls else None,
            "top1_top2_logit_margins": top_two_margins,
            "top1_top2_logit_margin_mean": statistics.mean(
                [x for x in top_two_margins if x is not None]
            ) if any(x is not None for x in top_two_margins) else None,
            "top1_top2_logit_margin_p90": pctl(
                [x for x in top_two_margins if x is not None], 0.90
            ),
            "generated_ids_sha256": hashlib.sha256(
                json.dumps([int(x) for x in generated_ids.tolist()], separators=(",", ":")).encode()
            ).hexdigest(),
        })
        return sequences

    def score_click_destinations(self, *, selected_action: str | None) -> dict[str, Any]:
        """Teacher-force all legal click completions, reusing their shared-prefix cache.

        The distribution at each trie node uses the exact hard-action mask from
        the runtime's compiled constraint. Search and non-click actions remain in
        that node's normalization denominator; final probabilities are then
        conditioned on the click-destination subset.
        """
        import torch

        context_ids = self.action_context_input_ids
        constraint = self.action_constraint
        if context_ids is None or constraint is None:
            raise RuntimeError("click-destination scoring requires the constrained action generation")
        targets = {
            str(action): [int(token_id) for token_id in tokens]
            for action, tokens in zip(
                constraint.schema["finite_actions"], constraint.finite_targets
            )
            if str(action).lower().startswith("click[")
        }
        if not targets:
            return summarize_click_log_probabilities({}, selected_action=selected_action)

        trie = build_token_trie(targets)
        model = self.model
        device = model.device
        root_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

        def fork_cache(cache: Any) -> Any:
            # Qwen3 under the pinned Transformers 4.51.1 rejects a legacy tuple
            # as past_key_values. Clone the DynamicCache through its legacy
            # representation so sibling trie branches get independent Cache
            # objects while retaining immutable tensor storage until extension.
            if cache is None:
                return None
            to_legacy = getattr(cache, "to_legacy_cache", None)
            from_legacy = getattr(type(cache), "from_legacy_cache", None)
            if not callable(to_legacy) or not callable(from_legacy):
                raise RuntimeError(
                    "pinned Qwen3 cache cannot be safely forked for click-destination scoring"
                )
            return from_legacy(to_legacy())

        def allowed_next(prefix: list[int]) -> set[int]:
            allowed = set(constraint._next_from_targets(prefix, constraint.finite_targets))
            allowed.update(constraint._search_next(prefix))
            if not allowed:
                raise RuntimeError("click candidate escaped the compiled action constraint")
            return allowed

        log_probabilities: dict[str, float] = {}

        def visit(
            node: dict[str, Any], prefix: list[int], past: Any, next_logits: Any,
            accumulated_log_probability: float,
        ) -> None:
            children = node["children"]
            if not children:
                for action in node["actions"]:
                    log_probabilities[action] = accumulated_log_probability
                return
            allowed = allowed_next(prefix)
            allowed_ids = sorted(allowed)
            masked_logits = torch.full_like(next_logits, float("-inf"))
            allowed_tensor = torch.tensor(allowed_ids, dtype=torch.long, device=next_logits.device)
            masked_logits[allowed_tensor] = next_logits[allowed_tensor]
            next_log_probs = torch.log_softmax(masked_logits, dim=-1)
            for token_id, child in sorted(children.items()):
                if int(token_id) not in allowed:
                    raise RuntimeError("candidate token is not legal under the compiled action constraint")
                child_logp = accumulated_log_probability + float(next_log_probs[int(token_id)].item())
                if child["actions"]:
                    for action in child["actions"]:
                        log_probabilities[action] = child_logp
                if child["children"]:
                    token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
                    output = model(
                        input_ids=token,
                        past_key_values=fork_cache(past),
                        use_cache=True,
                        return_dict=True,
                    )
                    visit(
                        child,
                        [*prefix, int(token_id)],
                        output.past_key_values,
                        output.logits[0, -1].float(),
                        child_logp,
                    )

        with torch.no_grad():
            root_output = model(input_ids=root_ids, use_cache=True, return_dict=True)
            visit(
                trie,
                [],
                root_output.past_key_values,
                root_output.logits[0, -1].float(),
                0.0,
            )
        return summarize_click_log_probabilities(
            log_probabilities, selected_action=selected_action
        )


def phase_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = sum(int(x["generated_token_count"]) for x in calls)
    entropy_sum = sum(float(x["entropy_sum_nats"]) for x in calls)
    nll_sum = sum(float(x["chosen_nll_sum_nats"]) for x in calls)
    margins = [
        float(value)
        for call in calls
        for value in call.get("top1_top2_logit_margins", [])
        if value is not None
    ]
    return {
        "generated_token_count": tokens,
        "entropy_mean_nats": entropy_sum / tokens if tokens else None,
        "chosen_nll_mean_nats": nll_sum / tokens if tokens else None,
        "top1_top2_logit_margin_mean": statistics.mean(margins) if margins else None,
        "top1_top2_logit_margin_p90": pctl(margins, 0.90),
    }


def load_science_modules(runtime_source_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(runtime_source_root))
    from openevo_webshop.comparison_contract import extract_goal
    from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
    from scripts.openevo_webshop.ceiling1_harness2011_prompt import (
        PROMPT_CONTRACT_SHA256,
        render_harness2011_compact_webshop_prompt,
    )
    from scripts.openevo_webshop.ceiling1_stage2_format_hard_runtime import (
        Qwen3Stage2FormatHardRuntime,
    )
    from scripts.openevo_webshop.parametric_rnd import FixedEnvironmentSeed
    return {
        "extract_goal": extract_goal,
        "WebshopEnv": WebshopEnv,
        "WebshopEnvConfig": WebshopEnvConfig,
        "PROMPT_CONTRACT_SHA256": PROMPT_CONTRACT_SHA256,
        "render_prompt": render_harness2011_compact_webshop_prompt,
        "Runtime": Qwen3Stage2FormatHardRuntime,
        "FixedEnvironmentSeed": FixedEnvironmentSeed,
    }


def validate_runtime_files(manifest: dict[str, Any], runtime_source_root: Path) -> None:
    for rel, expected in manifest["runtime_files"].items():
        actual = sha256_file(runtime_source_root / rel)
        if actual != expected:
            raise RuntimeError(f"runtime file hash mismatch: {rel}: {actual} != {expected}")


def validate_payload(path: Path, expected_sha: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {expected_sha}")


def run_episode(
    modules: dict[str, Any], env: Any, runtime: Any, observer: GenerateObserver | None,
    task: dict[str, Any], base_seed: int, *, q06_pre_action_diagnostics: bool = False,
    q06_completed_rounds: int | None = None, q06_lineage_id: str | None = None,
    q06_panel_role: str = "development",
) -> dict[str, Any]:
    obs, _ = env.reset(int(task["source_session_index"]), seed=int(task["worker_seed"]))
    goal = modules["extract_goal"](obs)
    observed_instruction_sha = hashlib.sha256(goal.encode()).hexdigest()
    if observed_instruction_sha != task["instruction_sha256"]:
        raise RuntimeError(
            f"instruction identity mismatch for {task['task_identity']}: "
            f"{observed_instruction_sha} != {task['instruction_sha256']}"
        )
    history: list[str] = []
    trace: list[dict[str, Any]] = []
    env_steps = 0
    for step_idx in range(MAX_STEPS):
        available = env.get_available_actions()
        rendered_history = " | ".join(history[-2:]) or "(none)"
        prompt = modules["render_prompt"](
            task_description=goal, current_observation=obs, available_actions=available,
            step_count=step_idx, history_length=2,
            action_history=rendered_history, max_prompt_chars=20_000,
        )
        # Preserve the exact predictor inputs at their real observation time;
        # generation receives the original objects and arguments unchanged.
        q06_snapshot = None
        if q06_pre_action_diagnostics:
            q06_snapshot = {
                "observation": copy.deepcopy(obs),
                "prompt": copy.deepcopy(prompt),
                "history": copy.deepcopy(history),
                "rendered_history": copy.deepcopy(rendered_history),
                "available_actions": copy.deepcopy(available),
            }
        generation_seed = int(base_seed) + step_idx * STEP_SEED_STRIDE
        mark = observer.mark() if observer else 0
        try:
            raw, receipt = runtime.generate_h2(
                prompt, available_actions=available, generation_seed=generation_seed,
                max_new_tokens=512, do_sample=True, temperature=1.0, top_p=1.0,
                top_k=0, repetition_penalty=1.0,
            )
        except RuntimeError as exc:
            if "action prefix escaped" in str(exc) or "no compilable action schema" in str(exc):
                return {"trace": trace, "steps": step_idx + 1, "score": 0.0, "won": False,
                        "invalid_terminated": True, "constraint_escape": True,
                        "environment_action_schema_violation": False, "webshop_env_step_calls": env_steps}
            raise
        calls = observer.since(mark) if observer else []
        if observer and len(calls) != 2:
            raise RuntimeError(f"expected exactly two generate calls per step, got {len(calls)}")
        action_channel = receipt.get("action_channel") or {}
        action = action_channel.get("action")
        valid = bool(action_channel.get("admissible"))
        tok = runtime.base._tok
        raw_token_count = len(tok(raw, add_special_tokens=False)["input_ids"])
        diag = None
        if observer:
            click_choice = observer.score_click_destinations(selected_action=action if valid else None)
            if q06_pre_action_diagnostics:
                # These are provenance annotations only. They are attached after
                # the unchanged generation/scoring calls and never enter a prompt.
                for channel in (calls[0], calls[1], click_choice):
                    channel["phase"] = "pre_action"
                    channel["source_event_index"] = 1
            diag = {
                "reasoning": calls[0],
                "action": calls[1],
                "combined": phase_summary(calls),
                "selected_action_nll_nats": calls[1]["chosen_nll_sum_nats"],
                "click_choice": click_choice,
                "raw_completion_token_count": raw_token_count,
                "observed_pre_action_model_tokens": action_channel.get("observed_pre_action_model_tokens"),
            }
        step = {
            "step": step_idx,
            "generation_seed": generation_seed,
            "raw_completion": raw,
            "raw_completion_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "raw_completion_token_count": raw_token_count,
            "action": action if valid else None,
            "action_valid": valid,
            "runtime_input_tokens": receipt.get("input_tokens"),
            "action_channel": action_channel,
            "generation_diagnostics": diag,
        }
        if q06_pre_action_diagnostics:
            if observer is None or not diag:
                raise RuntimeError("Q06 context capture requires existing Q17 instrumentation")
            if q06_completed_rounds is None or q06_lineage_id is None:
                raise RuntimeError("Q06 context capture requires bound round and lineage identity")
            action_context_ids = observer.action_context_input_ids
            if action_context_ids is None:
                raise RuntimeError("Q06 context capture lacks the real constrained action input IDs")
            action_schema = action_channel.get("compiled_action_schema")
            if action_schema is None:
                raise RuntimeError("Q06 context capture lacks the compiled action schema")
            step["q06_pre_action"] = _q06_pre_action_context(
                step_index=step_idx,
                completed_rounds=q06_completed_rounds,
                lineage_id=q06_lineage_id,
                panel_role=q06_panel_role,
                observation=q06_snapshot["observation"],
                prompt=q06_snapshot["prompt"],
                history=q06_snapshot["history"],
                rendered_history=q06_snapshot["rendered_history"],
                available_actions=q06_snapshot["available_actions"],
                action_schema=action_schema,
                action_context_token_ids=list(action_context_ids),
                tokenizer=tok,
            )
        trace.append(step)
        if not valid:
            return {"trace": trace, "steps": step_idx + 1, "score": 0.0, "won": False,
                    "invalid_terminated": True, "constraint_escape": False,
                    "environment_action_schema_violation": False, "webshop_env_step_calls": env_steps}
        try:
            obs, reward, done, info = env.step(action)
            env_steps += 1
        except BaseException as exc:
            step["environment_error"] = f"{type(exc).__name__}: {exc}"
            return {"trace": trace, "steps": step_idx + 1, "score": 0.0, "won": False,
                    "invalid_terminated": True, "constraint_escape": False,
                    "environment_action_schema_violation": True, "webshop_env_step_calls": env_steps}
        info = info or {}
        score = float(info.get("task_score", reward) or 0.0)
        step["score"] = score
        history.append(f"act:{action} -> obs:{str(obs)[:80]}")
        if done:
            return {"trace": trace, "steps": step_idx + 1, "score": score,
                    "won": bool(info.get("won", reward == 1.0)), "invalid_terminated": False,
                    "constraint_escape": False, "environment_action_schema_violation": False,
                    "webshop_env_step_calls": env_steps}
    return {"trace": trace, "steps": MAX_STEPS,
            "score": max([float(x.get("score", 0.0)) for x in trace] or [0.0]),
            "won": False, "invalid_terminated": False, "constraint_escape": False,
            "environment_action_schema_violation": False, "webshop_env_step_calls": env_steps}


def summarize_rows(rows: list[dict[str, Any]], instrumented: bool) -> dict[str, Any]:
    scores = [float(r["episode"]["score"]) for r in rows]
    steps = [int(r["episode"]["steps"]) for r in rows]
    raw_tokens = [
        int(step["raw_completion_token_count"])
        for row in rows for step in row["episode"]["trace"]
    ]
    reasoning_calls = [
        step["generation_diagnostics"]["reasoning"]
        for row in rows for step in row["episode"]["trace"]
        if step.get("generation_diagnostics")
    ]
    action_calls = [
        step["generation_diagnostics"]["action"]
        for row in rows for step in row["episode"]["trace"]
        if step.get("generation_diagnostics")
    ]
    instrumented_steps = [
        step["generation_diagnostics"]
        for row in rows for step in row["episode"]["trace"]
        if step.get("generation_diagnostics")
    ]
    click_choices_all = [step["click_choice"] for step in instrumented_steps]
    click_choices = [
        item for item in click_choices_all if int(item.get("candidate_count", 0)) > 0
    ]
    candidate_count_distribution: dict[str, int] = {}
    candidate_count_distribution_by_sentinel = {
        "sentinel_admissible": {},
        "sentinel_not_admissible": {},
    }
    for item in click_choices_all:
        count = int(item.get("candidate_count", 0))
        key = str(count)
        candidate_count_distribution[key] = candidate_count_distribution.get(key, 0) + 1
        sentinel = item.get("click_search_sentinel", {})
        group = (
            "sentinel_admissible"
            if sentinel.get("status") == "ADMISSIBLE"
            else "sentinel_not_admissible"
        )
        group_distribution = candidate_count_distribution_by_sentinel[group]
        group_distribution[key] = group_distribution.get(key, 0) + 1
    sentinel_summaries = [
        item.get("click_search_sentinel", {}) for item in click_choices_all
    ]
    sentinel_probabilities = [
        float(item["conditional_probability_given_click_family"])
        for item in sentinel_summaries
        if item.get("conditional_probability_given_click_family") is not None
    ]
    selected_action_nlls = [
        float(step["selected_action_nll_nats"])
        for step in instrumented_steps
        if step.get("selected_action_nll_nats") is not None
    ]
    return {
        "attempt_count": len(rows),
        "mean_score": statistics.mean(scores) if scores else None,
        "success_rate": sum(bool(r["episode"]["won"]) for r in rows) / len(rows) if rows else None,
        "positive_reward_rate": sum(float(r["episode"]["score"]) > 0 for r in rows) / len(rows) if rows else None,
        "mean_episode_steps": statistics.mean(steps) if steps else None,
        "invalid_terminated_count": sum(bool(r["episode"]["invalid_terminated"]) for r in rows),
        "constraint_escape_count": sum(bool(r["episode"]["constraint_escape"]) for r in rows),
        "environment_action_schema_violation_count": sum(bool(r["episode"]["environment_action_schema_violation"]) for r in rows),
        "mean_raw_completion_tokens_per_step": statistics.mean(raw_tokens) if raw_tokens else None,
        "p90_raw_completion_tokens_per_step": pctl([float(x) for x in raw_tokens], 0.90),
        "reasoning_generation": phase_summary(reasoning_calls) if instrumented else None,
        "action_generation": phase_summary(action_calls) if instrumented else None,
        "selected_action_nll_mean_nats": (
            statistics.mean(selected_action_nlls) if selected_action_nlls else None
        ) if instrumented else None,
        "click_choice": {
            "step_count": len(click_choices),
            "observed_step_count": len(click_choices_all),
            "destination_candidate_count_distribution": dict(sorted(
                candidate_count_distribution.items(), key=lambda item: int(item[0])
            )),
            "destination_candidate_count_distribution_by_search_sentinel": {
                group: dict(sorted(values.items(), key=lambda item: int(item[0])))
                for group, values in candidate_count_distribution_by_sentinel.items()
            },
            "click_search_sentinel_admissible_step_count": sum(
                item.get("status") == "ADMISSIBLE" for item in sentinel_summaries
            ),
            "click_search_sentinel_selected_step_count": sum(
                bool(item.get("selected")) for item in sentinel_summaries
            ),
            "mean_click_search_sentinel_probability_given_click_family": (
                statistics.mean(sentinel_probabilities) if sentinel_probabilities else None
            ),
            "mean_entropy_nats": statistics.mean(
                [float(x["entropy_nats"]) for x in click_choices]
            ) if click_choices else None,
            "mean_hhi": statistics.mean([float(x["hhi"]) for x in click_choices]) if click_choices else None,
            "top_k": 3,
            "mean_top_k_mass": statistics.mean(
                [float(x["top_k_mass"]) for x in click_choices]
            ) if click_choices else None,
            "mean_selected_action_conditional_probability": statistics.mean(
                [float(x["selected_action_conditional_probability"]) for x in click_choices
                 if x["selected_action_conditional_probability"] is not None]
            ) if any(x["selected_action_conditional_probability"] is not None for x in click_choices) else None,
        } if instrumented else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--runtime-source-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sd-adapter", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instrumentation", choices=("off", "on"), default="on")
    parser.add_argument("--q06-pre-action-diagnostics", action="store_true",
                        help="opt in to byte-bound Q06 development-panel pre-action context capture")
    parser.add_argument("--q06-lineage-id",
                        help="explicit Q06 training lineage identity; required with context capture")
    parser.add_argument("--task-limit", type=int)
    args = parser.parse_args()

    if args.q06_pre_action_diagnostics and args.instrumentation != "on":
        raise SystemExit("Q06 pre-action diagnostics require --instrumentation on")
    if args.q06_pre_action_diagnostics and not args.q06_lineage_id:
        raise SystemExit("Q06 pre-action diagnostics require an explicit --q06-lineage-id")

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != args.gpu_uuid:
        raise SystemExit("CUDA_VISIBLE_DEVICES mismatch")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    checkpoint_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if args.q06_pre_action_diagnostics:
        if type(manifest.get("completed_stage2_rounds")) is not int or manifest["completed_stage2_rounds"] < 0:
            raise SystemExit("Q06 manifest must bind an explicit nonnegative completed_stage2_rounds")
        if args.checkpoint_id not in manifest.get("checkpoints", {}):
            raise SystemExit("Q06 manifest does not bind the requested checkpoint identity")
    panel_manifest_path = args.panel_manifest
    if panel_manifest_path is None:
        panel_manifest_path = Path(__file__).resolve().parents[2] / FROZEN_PANEL_MANIFEST_REL
    panel_manifest_path = panel_manifest_path.resolve()
    panel_manifest = json.loads(panel_manifest_path.read_text())
    task_panel = validate_full_panel_manifest(
        panel_manifest_path, panel_manifest, task_limit=args.task_limit
    )
    if manifest.get("final_panel_access_allowed") is not False:
        raise RuntimeError("diagnostic manifest must explicitly forbid final-panel access")
    checkpoint = manifest["checkpoints"][args.checkpoint_id]
    validate_runtime_files(manifest, args.runtime_source_root)
    validate_payload(args.sd_adapter / "adapter_model.safetensors", checkpoint["adapter_sha256"], "adapter")
    common = manifest["common_carriers"]
    validate_payload(args.memory, common["memory_sha256"], "memory")
    validate_payload(args.skill, common["skill_sha256"], "skill")
    validate_payload(args.agent, common["agent_system_sha256"], "agent_system")
    if args.model != manifest["model"]["local_model_path"] or args.revision != manifest["model"]["revision"]:
        raise RuntimeError("model identity mismatch")
    modules = load_science_modules(args.runtime_source_root)
    runtime = modules["Runtime"](
        args.model, args.revision, sd_adapter=args.sd_adapter,
        memory=args.memory.read_text(), skill_bundle=args.skill.read_text(),
        agent_system=args.agent.read_text(),
    )
    observer: GenerateObserver | None = None
    if args.instrumentation == "on":
        observer = GenerateObserver(runtime.base._model)
        observer.install()
    rows: list[dict[str, Any]] = []
    tasks = list(panel_manifest["tasks"])
    started = time.time()
    try:
        for task in tasks:
            raw_env = modules["WebshopEnv"](
                modules["WebshopEnvConfig"](
                    seed_root=str(args.seed_root), num_products=1000,
                    seed=int(task["worker_seed"]),
                )
            )
            error = None
            try:
                episode = run_episode(
                    modules,
                    modules["FixedEnvironmentSeed"](raw_env, int(task["worker_seed"])),
                    runtime,
                    observer,
                    task,
                    int(task["paired_generation_seed"]),
                    q06_pre_action_diagnostics=args.q06_pre_action_diagnostics,
                    q06_completed_rounds=(manifest["completed_stage2_rounds"]
                                          if args.q06_pre_action_diagnostics else None),
                    q06_lineage_id=args.q06_lineage_id,
                    q06_panel_role="development",
                )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                episode = {"trace": [], "steps": 0, "score": 0.0, "won": False,
                           "invalid_terminated": False, "constraint_escape": False,
                           "environment_action_schema_violation": False,
                           "webshop_env_step_calls": 0}
            finally:
                raw_env.close()
            row = {
                "task_identity": task["task_identity"],
                "probe_ordinal": int(task["probe_ordinal"]),
                "source_session_index": int(task["source_session_index"]),
                "worker_seed": int(task["worker_seed"]),
                "paired_generation_seed": int(task["paired_generation_seed"]),
                "checkpoint_id": args.checkpoint_id,
                "state_id": checkpoint["state_id"],
                "adapter_sha256": checkpoint["adapter_sha256"],
                "instrumentation": args.instrumentation,
                "engineering_invalid": bool(error),
                "error": error,
                "episode": episode,
            }
            if args.q06_pre_action_diagnostics:
                row.update({
                    "lineage_id": args.q06_lineage_id,
                    "panel_role": "development",
                })
            rows.append(row)
    finally:
        if observer:
            observer.close()
        runtime.close()
    summary = {
        "schema": "openevo.q17.advisor-fixed-checkpoint-probe.202609282338",
        "status": "PASS" if len(rows) == FROZEN_PANEL_SIZE and not any(r["engineering_invalid"] for r in rows) else "FAIL",
        "checkpoint_id": args.checkpoint_id,
        "checkpoint": checkpoint,
        "instrumentation": args.instrumentation,
        "task_count": len(rows),
        "task_limit": args.task_limit,
        "probe_script_sha256": sha256_file(Path(__file__).resolve()),
        "task_panel_manifest_path": str(panel_manifest_path),
        "task_panel_manifest_sha256": task_panel["file_sha256"],
        "task_panel_task_identity_sha256": task_panel["task_identity_sha256"],
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "base_model": manifest["model"]["base_model"],
        "base_model_revision": manifest["model"]["revision"],
        "formal_task_consumption": 0,
        "final_panel_access": 0,
        "external_teacher_calls": 0,
        "runtime_execution_sha": manifest["runtime_execution_sha"],
        "prompt_contract_sha256": modules["PROMPT_CONTRACT_SHA256"],
        "panel_sha256": task_panel["panel_sha256"],
        "checkpoint_manifest_panel_sha256": manifest.get("panel_sha256"),
        "metrics": summarize_rows(rows, args.instrumentation == "on"),
        "wall_time_seconds": time.time() - started,
        "rows": rows,
    }
    summary["probe_receipt"] = build_probe_receipt(
        Path(__file__).resolve(), task_panel,
        checkpoint_id=args.checkpoint_id,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        base_model=manifest["model"]["base_model"],
        base_model_revision=manifest["model"]["revision"],
        instrumentation=args.instrumentation,
        task_count=len(rows),
    )
    if summary["probe_script_sha256"] != summary["probe_receipt"]["probe_script_sha256"]:
        raise RuntimeError("probe script SHA changed while writing output receipt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "probe-receipt.json", summary["probe_receipt"])
    with (args.output_dir / "attempts.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "checkpoint_id": args.checkpoint_id,
        "instrumentation": args.instrumentation,
        "probe_script_sha256": summary["probe_script_sha256"],
        "checkpoint_manifest_sha256": summary["checkpoint_manifest_sha256"],
        "base_model": summary["base_model"],
        "base_model_revision": summary["base_model_revision"],
        "task_panel_manifest_sha256": summary["task_panel_manifest_sha256"],
        "panel_sha256": summary["panel_sha256"],
        **summary["metrics"],
    }, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
