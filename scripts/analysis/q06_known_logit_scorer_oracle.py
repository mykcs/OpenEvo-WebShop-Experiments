#!/usr/bin/env python3
"""Pure-CPU known-logit conformance oracle for constrained Q06 candidate scoring.

This independent reference does not import a model, tokenizer, environment, or
the production Q17 probe.  Callers provide complete candidate token paths,
synthetic per-prefix logits, and the full legal-token mask at each scored
prefix.  Legal tokens without a candidate continuation remain in the
normalization denominator, matching constrained decoding semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def qualify_production_scorer(*, device: str = "cpu") -> dict[str, Any]:
    """Exercise the production trie/cache path against this independent oracle.

    This is zero-task engineering qualification. The injected model has known
    logits, performs tensor operations on the requested device, and never
    generates a policy completion or advances an environment.
    """
    import copy
    from types import SimpleNamespace

    import torch
    from scripts.openevo_webshop.q17_fixed_checkpoint_probe import GenerateObserver

    class Cache:
        def __init__(self, prefix=()):
            self.prefix = tuple(prefix)

        def to_legacy_cache(self):
            return self.prefix

        @classmethod
        def from_legacy_cache(cls, prefix):
            return cls(prefix)

    class Constraint:
        def __init__(self, targets, legal):
            self.schema = {"finite_actions": list(targets)}
            self.finite_targets = list(targets.values())
            self.legal = legal

        def _next_from_targets(self, prefix, targets):
            return set(self.legal[tuple(prefix)][:-1])

        def _search_next(self, prefix):
            # A legal non-candidate continuation exists only in the search
            # branch. Production must retain it in the mask denominator.
            return {self.legal[tuple(prefix)][-1]}

    class Model:
        def __init__(self, logits):
            self.device = torch.device(device)
            self.logits = logits
            self.forward_calls = 0

        def generate(self, *args, **kwargs):
            raise AssertionError("qualification must not generate")

        def __call__(self, *, input_ids, past_key_values=None, **kwargs):
            prefix = () if past_key_values is None else (
                *past_key_values.prefix, *input_ids[0].tolist()
            )
            # Mutating this cache exposes accidental sibling cache reuse.
            if past_key_values is not None:
                past_key_values.prefix = prefix
            self.forward_calls += 1
            logits = torch.tensor(self.logits[prefix], dtype=torch.float32,
                                  device=self.device).reshape(1, 1, -1)
            return SimpleNamespace(logits=logits, past_key_values=Cache(prefix))

    targets = {"click[red]": [1, 2, 5], "click[blue]": [1, 2, 6],
               "click[search]": [1, 3], "click[green]": [4, 2, 6]}
    legal = {(): [1, 4, 7], (1,): [2, 3, 8], (1, 2): [5, 6, 9],
             (4,): [2, 8], (4, 2): [5, 6, 9]}
    cases = []
    for kind in ("uniform", "unequal", "extreme", "no_sentinel", "one", "zero"):
        case_targets = copy.deepcopy(targets)
        if kind == "no_sentinel":
            case_targets.pop("click[search]")
        elif kind == "one":
            case_targets = {"click[red]": [1, 2, 5]}
        elif kind == "zero":
            case_targets = {}
        logits = {prefix: ([0.0] * 10 if kind == "uniform" else
                          [(i - 5) * (20.0 if kind == "extreme" else 0.7)
                           * (1 + 0.13 * sum(prefix)) + ((i + sum(prefix)) % 3) * 0.2
                           for i in range(10)]) for prefix in legal}
        model = Model(logits)
        observer = GenerateObserver(model)
        observer.action_context_input_ids = [10, 11, 12]
        observer.action_constraint = Constraint(case_targets, legal)
        def input_digest():
            constraint = observer.action_constraint
            return canonical_sha256({
                "prompt_input_ids": observer.action_context_input_ids,
                "action_schema": constraint.schema,
                "finite_targets": constraint.finite_targets,
                "legal_masks": {str(k): v for k, v in constraint.legal.items()},
            })

        before = input_digest()
        rng_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(model.device).clone() if model.device.type == "cuda" else None
        expected = score_known_logits(case_targets, logits, legal,
                                      selected_action="click[blue]")["summary"]
        actual = observer.score_click_destinations(selected_action="click[blue]")

        def compare(left, right):
            if isinstance(left, dict):
                if left.keys() != right.keys():
                    raise AssertionError("production summary keys differ from oracle")
                for key in left:
                    compare(left[key], right[key])
            elif isinstance(left, list):
                if len(left) != len(right):
                    raise AssertionError("production candidate count differs")
                for a, b in zip(left, right):
                    compare(a, b)
            elif type(left) is float:
                if not math.isclose(left, right, abs_tol=2e-5, rel_tol=2e-5):
                    raise AssertionError(f"production score differs: {left} != {right}")
            elif left != right:
                raise AssertionError(f"production value differs: {left!r} != {right!r}")

        compare(expected, actual)
        if model.forward_calls != (0 if kind == "zero" else 3 if kind == "one" else 5):
            raise AssertionError("production shared-prefix forward count differs")
        if before != input_digest():
            raise AssertionError("production scorer mutated its inputs")
        if not torch.equal(rng_before, torch.random.get_rng_state()):
            raise AssertionError("production scorer consumed CPU sampling RNG")
        if cuda_before is not None and not torch.equal(cuda_before, torch.cuda.get_rng_state(model.device)):
            raise AssertionError("production scorer consumed CUDA sampling RNG")
        cases.append({"case": kind, "status": "PASS", "model_forward_calls": model.forward_calls,
                      "input_sha256_before": before, "input_sha256_after": input_digest(),
                      "sampling_rng_unchanged": True})
    return {"schema": "openevo.q06.production-scorer-qualification.v1", "status": "PASS",
            "scope": "known-logit production trie/cache correctness and input/RNG invariance",
            "device": str(torch.device(device)), "torch_version": torch.__version__, "cases": cases,
            "task_bearing_calls": 0, "environment_steps": 0, "generation_calls": 0,
            "scientific_completion": False}


def canonical_sha256(value: Any) -> str:
    """Hash JSON evidence without changing the supplied object."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token_ids(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of token IDs")
    result = tuple(values)
    if any(type(token_id) is not int or token_id < 0 for token_id in result):
        raise ValueError(f"{label} must contain nonnegative integer token IDs")
    return result


def _log_softmax_for_legal_tokens(
    logits: Sequence[float], legal_tokens: Sequence[int], *, prefix: tuple[int, ...]
) -> dict[int, float]:
    legal = _token_ids(legal_tokens, label=f"legal token mask at {prefix}")
    if not legal or len(set(legal)) != len(legal):
        raise ValueError(f"legal token mask at {prefix} must be nonempty and unique")
    if max(legal) >= len(logits):
        raise ValueError(f"logits at {prefix} do not cover every legal token")
    values = [float(logits[token_id]) for token_id in legal]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"legal logits at {prefix} must all be finite")
    maximum = max(values)
    log_normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return {token_id: float(logits[token_id]) - log_normalizer for token_id in legal}


def constrained_candidate_log_probabilities(
    targets: Mapping[str, Sequence[int]],
    logits_by_prefix: Mapping[tuple[int, ...], Sequence[float]],
    legal_tokens_by_prefix: Mapping[tuple[int, ...], Sequence[int]],
) -> dict[str, float]:
    """Score each full candidate path under per-prefix constrained softmaxes.

    Prefix keys are tuples of previously consumed token IDs.  The denominator
    at each prefix includes *all* legal tokens, including legal continuations
    that are not among the requested candidate targets.
    """
    if not isinstance(targets, Mapping):
        raise ValueError("targets must map action names to complete token paths")
    normalized_targets: dict[str, tuple[int, ...]] = {}
    for action, path in targets.items():
        if not isinstance(action, str) or not action:
            raise ValueError("candidate action names must be nonempty strings")
        token_path = _token_ids(path, label=f"target path for {action}")
        if not token_path:
            raise ValueError(f"target path for {action} must not be empty")
        normalized_targets[action] = token_path
    if not normalized_targets:
        return {}
    paths = list(normalized_targets.items())
    for i, (left_action, left_path) in enumerate(paths):
        for right_action, right_path in paths[i + 1:]:
            if left_path == right_path:
                raise ValueError(
                    f"candidate token paths must be unique ({left_action!r}, {right_action!r})"
                )
            shorter, longer = sorted((left_path, right_path), key=len)
            if len(shorter) < len(longer) and longer[:len(shorter)] == shorter:
                raise ValueError(
                    f"candidate token paths must be prefix-free ({left_action!r}, {right_action!r})"
                )

    trie: dict[str, Any] = {"children": {}, "actions": []}
    for action, path in sorted(normalized_targets.items()):
        node = trie
        for token_id in path:
            node = node["children"].setdefault(token_id, {"children": {}, "actions": []})
        node["actions"].append(action)

    output: dict[str, float] = {}

    def visit(node: dict[str, Any], prefix: tuple[int, ...], accumulated: float) -> None:
        children = node["children"]
        if not children:
            for action in node["actions"]:
                output[action] = accumulated
            return
        if prefix not in logits_by_prefix or prefix not in legal_tokens_by_prefix:
            raise ValueError(f"missing logits or legal-token mask at prefix {prefix}")
        log_probs = _log_softmax_for_legal_tokens(
            logits_by_prefix[prefix], legal_tokens_by_prefix[prefix], prefix=prefix
        )
        for token_id, child in sorted(children.items()):
            if token_id not in log_probs:
                raise ValueError(f"candidate token {token_id} is illegal at prefix {prefix}")
            child_log_probability = accumulated + log_probs[token_id]
            for action in child["actions"]:
                output[action] = child_log_probability
            if child["children"]:
                visit(child, (*prefix, token_id), child_log_probability)

    visit(trie, (), 0.0)
    return output


def summarize_candidate_log_probabilities(
    action_log_probabilities: Mapping[str, float], *,
    selected_action: str | None = None,
    top_k: int = 3,
    sentinel_action: str = "click[search]",
) -> dict[str, Any]:
    """Summarize destination uncertainty while reporting sentinel separately."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    for action, log_probability in action_log_probabilities.items():
        if not math.isfinite(float(log_probability)):
            raise ValueError(f"non-finite candidate log probability for {action}")

    sentinel_logp = action_log_probabilities.get(sentinel_action)
    destinations = {
        action: float(logp)
        for action, logp in action_log_probabilities.items()
        if action != sentinel_action
    }
    family_probabilities = _normalized_probabilities(action_log_probabilities)
    sentinel = {
        "status": "ADMISSIBLE" if sentinel_logp is not None else "NOT_ADMISSIBLE",
        "action": sentinel_action,
        "log_probability_under_constrained_decoder": (
            float(sentinel_logp) if sentinel_logp is not None else None
        ),
        "conditional_probability_given_click_family": (
            family_probabilities[sentinel_action] if sentinel_logp is not None else None
        ),
        "selected": selected_action == sentinel_action,
    }
    if not destinations:
        return {
            "status": "NO_CLICK_DESTINATIONS",
            "candidate_count": 0,
            "entropy_nats": None,
            "hhi": None,
            "top_k": top_k,
            "top_k_mass": None,
            "selected_action": selected_action,
            "selected_action_log_probability": None,
            "selected_action_conditional_probability": None,
            "candidates": [],
            "click_search_sentinel": sentinel,
        }

    probabilities = _normalized_probabilities(destinations)
    candidates = [
        {
            "action": action,
            "log_probability_under_constrained_decoder": logp,
            "conditional_probability_given_click": probabilities[action],
        }
        for action, logp in sorted(destinations.items())
    ]
    ordered = sorted(probabilities.values(), reverse=True)
    selected_logp = action_log_probabilities.get(selected_action) if selected_action else None
    return {
        "status": "PASS",
        "candidate_count": len(destinations),
        "entropy_nats": -sum(p * math.log(p) for p in probabilities.values()),
        "hhi": sum(p * p for p in probabilities.values()),
        "top_k": top_k,
        "top_k_mass": sum(ordered[:top_k]),
        "selected_action": selected_action,
        "selected_action_log_probability": (
            float(selected_logp) if selected_logp is not None else None
        ),
        "selected_action_conditional_probability": (
            probabilities.get(selected_action) if selected_action is not None else None
        ),
        "candidates": candidates,
        "click_search_sentinel": sentinel,
    }


def _normalized_probabilities(log_probabilities: Mapping[str, float]) -> dict[str, float]:
    if not log_probabilities:
        return {}
    maximum = max(float(value) for value in log_probabilities.values())
    denominator = sum(math.exp(float(value) - maximum) for value in log_probabilities.values())
    log_normalizer = maximum + math.log(denominator)
    return {
        action: math.exp(float(value) - log_normalizer)
        for action, value in log_probabilities.items()
    }


def score_known_logits(
    targets: Mapping[str, Sequence[int]],
    logits_by_prefix: Mapping[tuple[int, ...], Sequence[float]],
    legal_tokens_by_prefix: Mapping[tuple[int, ...], Sequence[int]],
    *,
    evidence: Mapping[str, Any] | None = None,
    selected_action: str | None = None,
    top_k: int = 3,
    sentinel_action: str = "click[search]",
) -> dict[str, Any]:
    """Run scoring and bind a before/after digest to supplied evidence."""
    before = canonical_sha256(evidence) if evidence is not None else None
    action_log_probabilities = constrained_candidate_log_probabilities(
        targets, logits_by_prefix, legal_tokens_by_prefix
    )
    after = canonical_sha256(evidence) if evidence is not None else None
    if before != after:
        raise RuntimeError("known-logit scoring mutated supplied evidence")
    return {
        "action_log_probabilities": action_log_probabilities,
        "summary": summarize_candidate_log_probabilities(
            action_log_probabilities,
            selected_action=selected_action,
            top_k=top_k,
            sentinel_action=sentinel_action,
        ),
        "evidence_sha256_before": before,
        "evidence_sha256_after": after,
        "evidence_unchanged": before == after,
        "model_calls": 0,
        "environment_calls": 0,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Zero-task production Q17 scorer qualification")
    parser.add_argument("--qualify-production", action="store_true", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify_production_scorer(device=args.device)
    from scripts.openevo_webshop import q17_fixed_checkpoint_probe as production
    receipt["qualification_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt["production_script_sha256"] = hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest()
    receipt["gpu_uuid"] = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        __import__("os").fsync(stream.fileno())
    print(json.dumps({"status": receipt["status"], "device": receipt["device"],
                      "cases": len(receipt["cases"]), "receipt": str(args.receipt)}))
