#!/usr/bin/env python3
"""Harness201.1 runtime: free reasoning, constrained environment action."""
from __future__ import annotations
import hashlib, json, re
from typing import Any
from openevo_webshop.comparison_contract import format_webshop_actions
from scripts.openevo_webshop.ceiling1_harness2_runtime import Harness201Runtime

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.I | re.S)
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
SEARCH_RE = re.compile(r"^search\[([^\]\r\n]+)\]$")
ACTION_OPEN = "<action>"
ACTION_CLOSE = "</action>"
PRE_ACTION_TOKEN_BUDGET = 64
MIN_PRE_ACTION_MODEL_TOKENS = 4
SEARCH_QUERY_TOKEN_BUDGET = 256



def extract_unique_action(text: str) -> str | None:
    matches = ACTION_RE.findall(str(text))
    if len(matches) != 1:
        return None
    action = matches[0].strip()
    return action or None


def compile_action_schema(available_actions: dict[str, Any]) -> dict[str, Any]:
    rendered = [str(x) for x in format_webshop_actions(available_actions)]
    finite = sorted({x for x in rendered if not x.startswith("search[")})
    search_allowed = bool(available_actions.get("has_search_bar"))
    payload = {"finite_actions": finite, "search_allowed": search_allowed}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def action_is_admissible(action: str | None, available_actions: dict[str, Any]) -> bool:
    if not action:
        return False
    schema = compile_action_schema(available_actions)
    if action in schema["finite_actions"]:
        return True
    m = SEARCH_RE.fullmatch(action)
    return bool(m and schema["search_allowed"] and m.group(1).strip())


class ActionPrefixConstraint:
    """Prefix gate that is unconstrained before <action>, hard-constrained after it."""
    def __init__(self, tokenizer: Any, available_actions: dict[str, Any], n_input: int) -> None:
        self.tok = tokenizer
        self.n_input = int(n_input)
        self.schema = compile_action_schema(available_actions)
        self.eos = int(tokenizer.eos_token_id)
        self.all_tokens = list(range(len(tokenizer)))
        self.special_ids = {int(x) for x in getattr(tokenizer, "all_special_ids", [])}
        self.finite_targets = [self._encode(x + ACTION_CLOSE) + [self.eos] for x in self.schema["finite_actions"]]
        self.search_prefix = self._encode("search[")
        self.search_close = self._encode("]" + ACTION_CLOSE) + [self.eos]
        self.safe_query_tokens = self._safe_query_token_ids() if self.schema["search_allowed"] else []
        self.safe_query_set = set(self.safe_query_tokens)
        self.open_end_token_index: int | None = None
        if self.schema["search_allowed"] and (not self.search_prefix or not self.search_close):
            raise RuntimeError("search schema tokenizer compilation failed")
        if not self.finite_targets and not self.schema["search_allowed"]:
            raise RuntimeError("current environment exposes no compilable action schema")

    def _encode(self, text: str) -> list[int]:
        return [int(x) for x in self.tok(text, add_special_tokens=False)["input_ids"]]

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _safe_query_token_ids(self) -> list[int]:
        out = []
        for tid in self.all_tokens:
            if tid in self.special_ids:
                continue
            piece = self._decode([tid])
            if not piece or any(ch in piece for ch in ("]", "<", ">", "\n", "\r")):
                continue
            out.append(tid)
        if not out:
            raise RuntimeError("no tokenizer tokens available for search query content")
        return out

    @staticmethod
    def _next_from_targets(prefix: list[int], targets: list[list[int]]) -> set[int]:
        nxt: set[int] = set()
        for target in targets:
            if len(prefix) < len(target) and target[: len(prefix)] == prefix:
                nxt.add(int(target[len(prefix)]))
        return nxt

    def _search_next(self, prefix: list[int]) -> set[int]:
        if not self.schema["search_allowed"]:
            return set()
        if len(prefix) < len(self.search_prefix) and self.search_prefix[: len(prefix)] == prefix:
            return {int(self.search_prefix[len(prefix)])}
        if prefix[: len(self.search_prefix)] != self.search_prefix:
            return set()
        tail = prefix[len(self.search_prefix):]
        first_non_query = None
        for i, tid in enumerate(tail):
            if tid not in self.safe_query_set:
                first_non_query = i
                break
        if first_non_query is None:
            nxt = set(self.safe_query_tokens)
            if tail:
                nxt.add(int(self.search_close[0]))
            return nxt
        query = tail[:first_non_query]
        closing = tail[first_non_query:]
        if not query or closing != self.search_close[: len(closing)]:
            return set()
        if len(closing) < len(self.search_close):
            return {int(self.search_close[len(closing)])}
        return set()

    def _action_suffix(self, generated: list[int]) -> list[int] | None:
        if self.open_end_token_index is not None:
            return generated[self.open_end_token_index:]
        text = self._decode(generated)
        if ACTION_OPEN not in text.lower():
            return None
        for k in range(1, len(generated) + 1):
            if ACTION_OPEN in self._decode(generated[:k]).lower():
                self.open_end_token_index = k
                return generated[k:]
        return None

    def __call__(self, batch_id: int, input_ids: Any) -> list[int]:
        if int(batch_id) != 0:
            raise RuntimeError("Harness201.1 only supports batch size 1")
        ids = [int(x) for x in input_ids.tolist()]
        generated = ids[self.n_input:]
        suffix = self._action_suffix(generated)
        if suffix is None:
            return self.all_tokens
        nxt = self._next_from_targets(suffix, self.finite_targets)
        nxt.update(self._search_next(suffix))
        if not nxt:
            raise RuntimeError("Harness201.1 action prefix escaped all current action schemas")
        return sorted(nxt)


class Stage2MinimumPreActionLogitsProcessor:
    """Keep reasoning alive for >=4 model tokens and forbid EOS before handoff."""
    _blocked_cache: dict[int, list[int]] = {}

    def __init__(self, tokenizer: Any, n_prompt: int, stop_token_ids: list[int]) -> None:
        self.tok = tokenizer
        self.n_prompt = int(n_prompt)
        self.stop_token_ids = sorted({int(x) for x in stop_token_ids})

    @property
    def blocked_first_token_ids(self) -> list[int]:
        key = id(self.tok)
        cached = self._blocked_cache.get(key)
        if cached is not None:
            return cached
        special = {int(x) for x in getattr(self.tok, "all_special_ids", [])}
        blocked: list[int] = []
        for tid in range(len(self.tok)):
            piece = self.tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if tid in special or not piece.strip() or "<" in piece:
                blocked.append(int(tid))
        self._blocked_cache[key] = blocked
        return blocked

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        for token_id in self.stop_token_ids:
            scores[0, token_id] = float("-inf")
        generated = int(input_ids.shape[1]) - self.n_prompt
        if generated < MIN_PRE_ACTION_MODEL_TOKENS:
            scores[0, self.blocked_first_token_ids] = float("-inf")
        return scores


class Stage2ActionOpenStoppingCriteria:
    """Stop free reasoning once the model itself happens to emit literal <action>."""
    def __init__(self, tokenizer: Any, n_input: int) -> None:
        self.tok = tokenizer
        self.n_input = int(n_input)

    def __call__(self, input_ids: Any, scores: Any, **_: Any) -> bool:
        ids = [int(x) for x in input_ids[0, self.n_input:].tolist()]
        text = self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return ACTION_OPEN in text.lower()


class Stage2ActionHardLogitsProcessor:
    """After interface-owned <action>, constrain the body to current Allowed-actions."""
    def __init__(self, constraint: ActionPrefixConstraint) -> None:
        self.constraint = constraint

    @staticmethod
    def _force_only(scores: Any, allowed: list[int]) -> Any:
        import torch
        masked = torch.full_like(scores, float("-inf"))
        masked[0, allowed] = scores[0, allowed]
        return masked

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if int(input_ids.shape[0]) != 1:
            raise RuntimeError("Harness201.1 Stage2 action channel only supports batch size 1")
        ids = [int(x) for x in input_ids[0].tolist()]
        generated = ids[self.constraint.n_input:]
        suffix = self.constraint._action_suffix(generated)
        if suffix is None:
            raise RuntimeError("Stage2 interface-owned action opener missing from constrained phase")
        c = self.constraint
        if c.schema["search_allowed"] and suffix[: len(c.search_prefix)] == c.search_prefix:
            tail = suffix[len(c.search_prefix):]
            if tail and all(t in c.safe_query_set for t in tail) and len(tail) >= SEARCH_QUERY_TOKEN_BUDGET:
                return self._force_only(scores, [int(c.search_close[0])])
        nxt = c._next_from_targets(suffix, c.finite_targets)
        nxt.update(c._search_next(suffix))
        if not nxt:
            raise RuntimeError("Harness201.1 action prefix escaped all current action schemas")
        return self._force_only(scores, sorted(nxt))


class Harness2011Runtime(Harness201Runtime):
    PROFILE = "201.1"

    def generate_h2(self, prompt: str, *, available_actions: dict[str, Any], generation_seed: int,
                    max_new_tokens: int = 512, do_sample: bool = True, temperature: float = 1.0,
                    top_p: float = 1.0, top_k: int = 0, repetition_penalty: float = 1.0):
        import torch
        from transformers import LogitsProcessorList, StoppingCriteriaList

        enc, receipt = self.prepare_h2(prompt, available_actions=available_actions)
        model, tok = self.base._model, self.base._tok
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_ids = enc["input_ids"]
        n_prompt = int(prompt_ids.shape[1])

        configured_eos = getattr(model.generation_config, "eos_token_id", None)
        if configured_eos is None:
            stop_token_ids = [int(tok.eos_token_id)]
        elif isinstance(configured_eos, int):
            stop_token_ids = [int(configured_eos)]
        else:
            stop_token_ids = [int(x) for x in configured_eos]
        if int(tok.eos_token_id) not in stop_token_ids:
            stop_token_ids.append(int(tok.eos_token_id))

        common: dict[str, Any] = {
            "do_sample": bool(do_sample),
            "repetition_penalty": float(repetition_penalty),
            "pad_token_id": tok.eos_token_id,
        }
        if do_sample:
            common.update(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))

        device_index = getattr(model.device, "index", None)
        devices = [device_index] if getattr(model.device, "type", None) == "cuda" and device_index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(generation_seed))
            with torch.no_grad():
                reasoning_out = model.generate(
                    **enc,
                    max_new_tokens=PRE_ACTION_TOKEN_BUDGET,
                    logits_processor=LogitsProcessorList([
                        Stage2MinimumPreActionLogitsProcessor(tok, n_prompt, stop_token_ids)
                    ]),
                    stopping_criteria=StoppingCriteriaList([
                        Stage2ActionOpenStoppingCriteria(tok, n_prompt)
                    ]),
                    **common,
                )

            reasoning_raw = tok.decode(
                reasoning_out[0][n_prompt:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            match = re.search(re.escape(ACTION_OPEN), reasoning_raw, re.I)
            reasoning_text = reasoning_raw[: match.start()] if match else reasoning_raw
            reasoning_ids = tok(
                reasoning_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(device=model.device, dtype=prompt_ids.dtype)
            opener_ids = tok(
                ACTION_OPEN, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(device=model.device, dtype=prompt_ids.dtype)
            action_input_ids = torch.cat([prompt_ids, reasoning_ids, opener_ids], dim=1)
            n_constraint_input = n_prompt + int(reasoning_ids.shape[1])
            constraint = ActionPrefixConstraint(tok, available_actions, n_constraint_input)
            action_budget = max(32, int(max_new_tokens) - int(reasoning_ids.shape[1]))
            with torch.no_grad():
                action_out = model.generate(
                    input_ids=action_input_ids,
                    attention_mask=torch.ones_like(action_input_ids),
                    max_new_tokens=action_budget,
                    logits_processor=LogitsProcessorList([Stage2ActionHardLogitsProcessor(constraint)]),
                    **common,
                )

        action_tail = tok.decode(
            action_out[0][int(action_input_ids.shape[1]):],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        raw = reasoning_text + ACTION_OPEN + action_tail
        action = extract_unique_action(raw)
        admissible = action_is_admissible(action, available_actions)
        receipt["sampling"] = {
            "do_sample": bool(do_sample), "temperature": float(temperature) if do_sample else None,
            "top_p": float(top_p) if do_sample else None, "top_k": int(top_k) if do_sample else None,
            "repetition_penalty": float(repetition_penalty), "max_new_tokens": int(max_new_tokens),
        }
        receipt["harness"] = "201.1"
        receipt["action_channel"] = {
            "schema": "openevo.ceiling1-harness2011-action-channel.v2-interface-owned",
            "compiled_action_schema": constraint.schema,
            "action_channel_entered": True,
            "action": action,
            "admissible": bool(admissible),
            "constraint_escape": False,
            "unconstrained_fallback": False,
            "fuzzy_repair_used": False,
            "reasoning_contains_cjk": bool(CJK_RE.search(reasoning_text)),
            "constraint_engine": "shared-stage2-interface-owned-action-envelope-v1",
            "interface_owned_action_opener": True,
            "minimum_pre_action_model_tokens": MIN_PRE_ACTION_MODEL_TOKENS,
            "pre_action_token_budget": PRE_ACTION_TOKEN_BUDGET,
            "search_query_token_budget": SEARCH_QUERY_TOKEN_BUDGET,
            "model_action_choice_preserved": True,
            "strategy_guidance_injected": False,
        }
        return raw, receipt
