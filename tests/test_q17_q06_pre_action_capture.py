from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

from scripts.openevo_webshop import q17_fixed_checkpoint_probe as probe
from scripts.analysis.advisor_q06_raw_adapter import adapt_attempts_bytes


class Tokenizer:
    def __call__(self, value, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in value]}


class FakeEnvironment:
    def __init__(self):
        self.step_calls = []

    def reset(self, source_session_index, *, seed):
        return "initial observation", {}

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": ["item 1"], "actions": ["search"]}

    def step(self, action):
        self.step_calls.append(action)
        return "post-action observation", 1.0, True, {"task_score": 1.0, "won": True}


class FakeObserver:
    action_context_input_ids = [11, 12, 13]

    def __init__(self):
        self.calls = [
            {"generated_token_count": 1, "entropy_sum_nats": 0.2,
             "chosen_nll_sum_nats": 0.3, "top1_top2_logit_margins": [0.4]},
            {"generated_token_count": 1, "entropy_sum_nats": 0.1,
             "chosen_nll_sum_nats": 0.2, "top1_top2_logit_margins": [0.5]},
        ]

    def mark(self):
        return 0

    def since(self, mark):
        assert mark == 0
        return self.calls

    def score_click_destinations(self, *, selected_action):
        return {"candidate_count": 2, "entropy_nats": 0.5, "selected_action": selected_action}


class FakeRuntime:
    def __init__(self, *, mutate_available=False):
        self.base = SimpleNamespace(_tok=Tokenizer())
        self.arguments = []
        self.seen_available_actions = []
        self.mutate_available = mutate_available

    def generate_h2(self, prompt, **kwargs):
        self.seen_available_actions.append(json.loads(json.dumps(kwargs["available_actions"])))
        self.arguments.append((prompt, kwargs))
        if self.mutate_available:
            kwargs["available_actions"]["clickables"].append("runtime mutation")
        return "<think>go</think><action>search</action>", {
            "input_tokens": 123,
            "action_channel": {
                "action": "search",
                "admissible": True,
                "compiled_action_schema": {"finite_actions": ["search"]},
            },
        }


def run(*, capture, mutate_available=False):
    env = FakeEnvironment()
    runtime = FakeRuntime(mutate_available=mutate_available)
    observer = FakeObserver()
    modules = {
        "extract_goal": lambda observation: "find item",
        "render_prompt": lambda **kwargs: (
            "goal=" + kwargs["task_description"] + "; observation="
            + kwargs["current_observation"] + "; history=" + kwargs["action_history"]
        ),
    }
    episode = probe.run_episode(
        modules, env, runtime, observer,
        {"source_session_index": 4, "worker_seed": 7,
         "instruction_sha256": hashlib.sha256(b"find item").hexdigest()},
        99,
        q06_pre_action_diagnostics=capture,
        q06_completed_rounds=120 if capture else None,
        q06_lineage_id="training-lineage-abc" if capture else None,
    )
    return episode, env, runtime


def test_q06_capture_is_opt_in_and_does_not_change_generation_arguments():
    ordinary, ordinary_env, ordinary_runtime = run(capture=False)
    captured, captured_env, captured_runtime = run(capture=True)

    assert "q06_pre_action" not in ordinary["trace"][0]
    context = captured["trace"][0]["q06_pre_action"]
    assert context["schema"] == "openevo.q06.pre-action-state.v1"
    assert context["phase"] == "pre_action"
    assert context["completed_rounds"] == 120
    assert context["lineage_id"] == "training-lineage-abc"
    assert context["panel_role"] == "development"
    assert context["state_type"] == "search"
    assert context["feature_event_index"] < context["decision_event_index"] <= context["action_event_index"]
    assert context["prompt"] == "goal=find item; observation=initial observation; history=(none)"
    assert context["observation"] == "initial observation"
    assert context["action_context_token_ids"] == [11, 12, 13]
    assert context["action_context_token_ids_sha256"] == hashlib.sha256(b"[11,12,13]").hexdigest()
    assert context["prompt_tokens"] == len(context["prompt"])
    assert context["history_tokens"] == len("(none)")
    assert captured["trace"][0]["generation_diagnostics"]["click_choice"]["source_event_index"] == 1
    assert captured["trace"][0]["generation_diagnostics"]["action"]["phase"] == "pre_action"
    assert ordinary_runtime.arguments == captured_runtime.arguments
    assert ordinary_env.step_calls == captured_env.step_calls == ["search"]


def test_context_hashes_bind_canonical_raw_values_and_action_schema():
    episode, _, _ = run(capture=True)
    context = episode["trace"][0]["q06_pre_action"]
    assert context["observation_sha256"] == hashlib.sha256(b"initial observation").hexdigest()
    assert context["prompt_sha256"] == hashlib.sha256(context["prompt"].encode()).hexdigest()
    assert context["history_sha256"] == hashlib.sha256(b"[]").hexdigest()
    assert context["available_actions_sha256"] == hashlib.sha256(
        json.dumps(context["available_actions"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert context["action_schema_sha256"] == hashlib.sha256(b'{"finite_actions":["search"]}').hexdigest()


def test_opt_in_trace_is_accepted_by_existing_fail_closed_raw_adapter():
    episode, _, _ = run(capture=True)
    attempt = {
        "task_identity": "fixture-task",
        "lineage_id": "training-lineage-abc",
        "checkpoint_id": "after_120_rounds",
        "panel_role": "development",
        "engineering_invalid": False,
        "episode": {**episode, "score": 1.0, "won": True},
    }
    raw = (json.dumps(attempt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    adapted = adapt_attempts_bytes(
        raw, lineage_id="training-lineage-abc", checkpoint_id="after_120_rounds",
    )
    assert adapted["status"] == "CPU_ADAPTER_PASS"
    assert adapted["normalized_row_count"] == 1
    assert adapted["rows"][0]["features"]["round"] == 120
    assert adapted["rows"][0]["state_type"] == "search"
    assert adapted["rows"][0]["feature_provenance_sha256"] == hashlib.sha256(raw).hexdigest()


def test_context_uses_pre_generation_snapshot_if_runtime_mutates_actions():
    episode, _, runtime = run(capture=True, mutate_available=True)
    context = episode["trace"][0]["q06_pre_action"]
    assert runtime.seen_available_actions[0]["clickables"] == ["item 1"]
    assert runtime.arguments[0][1]["available_actions"]["clickables"] == ["item 1", "runtime mutation"]
    assert context["available_actions"]["clickables"] == ["item 1"]
    assert context["available_actions_sha256"] == hashlib.sha256(
        json.dumps(context["available_actions"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def test_generate_observer_binds_action_prefix_before_original_generation():
    class Tensor:
        def __init__(self, values):
            self.values = values
            self.shape = (1, len(values))

        def __getitem__(self, index):
            if isinstance(index, tuple):
                row, selection = index
                assert row == 0
                return Tensor(self.values[selection])
            if index == 0:
                return Tensor(self.values)
            raise IndexError(index)

        def tolist(self):
            return list(self.values)

    class Logits:
        def __getitem__(self, index):
            if isinstance(index, list):
                return Tensor([0.8, 0.2])
            return Scalar()

        def float(self):
            return self

        def tolist(self):
            return [0.8, 0.2]

    class BatchedLogits:
        def __getitem__(self, index):
            assert index == 0
            return Logits()

    class Scalar:
        def item(self):
            return 0.4

        def __neg__(self):
            return self

    fake_torch = SimpleNamespace(
        distributions=SimpleNamespace(Categorical=lambda **kwargs: SimpleNamespace(entropy=lambda: Scalar())),
        log_softmax=lambda logits, dim: logits,
        isfinite=lambda logits: [True, True],
    )
    constraint = SimpleNamespace(finite_targets=[[7]])
    observed_at_original = []

    def original(*args, **kwargs):
        observed_at_original.append(observer.action_context_input_ids)
        return SimpleNamespace(sequences=Tensor([1, 2, 3, 4]), scores=[BatchedLogits()])

    model = SimpleNamespace(generate=original)
    observer = probe.GenerateObserver(model)
    observer.install()
    try:
        with patch.dict(sys.modules, {"torch": fake_torch}):
            observer._wrapped(
                input_ids=Tensor([1, 2, 3]),
                logits_processor=[SimpleNamespace(constraint=constraint)],
            )
    finally:
        observer.close()
    assert observed_at_original == [[1, 2, 3]]


def test_q06_context_rejects_non_development_panel_and_missing_action_prefix():
    base = dict(
        step_index=0, completed_rounds=120, lineage_id="lineage", panel_role="final",
        observation="obs", prompt="prompt", history=[], rendered_history="(none)",
        available_actions={}, action_schema={}, action_context_token_ids=[1], tokenizer=Tokenizer(),
    )
    try:
        probe._q06_pre_action_context(**base)
    except ValueError as exc:
        assert "development panel" in str(exc)
    else:
        raise AssertionError("final-panel capture must be rejected")

    base["panel_role"] = "development"
    base["action_context_token_ids"] = []
    try:
        probe._q06_pre_action_context(**base)
    except ValueError as exc:
        assert "token IDs" in str(exc)
    else:
        raise AssertionError("missing action prefix IDs must be rejected")
