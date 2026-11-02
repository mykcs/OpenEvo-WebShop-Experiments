#!/usr/bin/env python3
"""Harness201.1 Stage1 interface-only runtime.

No learned Memory / Skill / Agent System and no SD adapter are accepted here.
Only the universal WebShop interface contract is inherited: reasoning-soft,
action-hard constrained decoding against the current dynamic Allowed Actions.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from openevo_webshop.inference import TransformersBackend
from scripts.openevo_webshop.ceiling1_harness2011_runtime import (
    ACTION_OPEN,
    ActionPrefixConstraint,
    action_is_admissible,
    extract_unique_action,
)

CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
PROFILE = "201.1-stage1-interface-only"
RECEIPT_SCHEMA = "openevo.ceiling1-harness2011-stage1-interface-context.v1"
PRE_ACTION_TOKEN_BUDGET = 64


_SAFE_QUERY_CACHE: dict[int, tuple[list[int], list[int]]] = {}


class Stage1ActionPrefixConstraint(ActionPrefixConstraint):
    """Same action schema as Harness201.1, with tokenizer-local query-token cache."""

    def _safe_query_token_ids(self) -> list[int]:
        key = id(self.tok)
        cached = _SAFE_QUERY_CACHE.get(key)
        if cached is not None:
            return list(cached[0])
        safe = super()._safe_query_token_ids()
        safe_set = set(safe)
        unsafe = [int(tid) for tid in self.all_tokens if int(tid) not in safe_set]
        _SAFE_QUERY_CACHE[key] = (list(safe), unsafe)
        return list(safe)

    @property
    def unsafe_query_token_ids(self) -> list[int]:
        cached = _SAFE_QUERY_CACHE.get(id(self.tok))
        if cached is None:
            _ = self._safe_query_token_ids()
            cached = _SAFE_QUERY_CACHE[id(self.tok)]
        return cached[1]


class ActionHardLogitsProcessor:
    """Free pre-action text, mandatory action entry, hard Allowed-actions mask."""

    def __init__(self, constraint: ActionPrefixConstraint, *, stop_token_ids: list[int],
                 pre_action_token_budget: int = PRE_ACTION_TOKEN_BUDGET) -> None:
        self.constraint = constraint
        self.stop_token_ids = sorted({int(x) for x in stop_token_ids})
        self.pre_action_token_budget = int(pre_action_token_budget)
        self.action_open_tokens = constraint._encode(ACTION_OPEN)
        if self.pre_action_token_budget < 0 or not self.action_open_tokens:
            raise RuntimeError("invalid Stage1 action-channel compilation")

    def _force_only(self, scores: Any, allowed: list[int]) -> Any:
        import torch

        masked = torch.full_like(scores, float("-inf"))
        masked[0, allowed] = scores[0, allowed]
        return masked

    def _search_query_phase(self, suffix: list[int], scores: Any) -> Any | None:
        c = self.constraint
        if not c.schema["search_allowed"] or suffix[: len(c.search_prefix)] != c.search_prefix:
            return None
        tail = suffix[len(c.search_prefix):]
        first_non_query = next((i for i, tid in enumerate(tail) if tid not in c.safe_query_set), None)
        if first_non_query is not None:
            return None
        # Exact semantic complement of c._search_next() during free query text:
        # all safe query tokens, plus the closing token only after >=1 query token.
        masked = scores.clone()
        masked[0, c.unsafe_query_token_ids] = float("-inf")
        close_id = int(c.search_close[0])
        if tail:
            masked[0, close_id] = scores[0, close_id]
        else:
            masked[0, close_id] = float("-inf")
        return masked

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if int(input_ids.shape[0]) != 1:
            raise RuntimeError("Harness201.1 Stage1 action channel only supports batch size 1")
        ids = [int(x) for x in input_ids[0].tolist()]
        generated = ids[self.constraint.n_input:]
        suffix = self.constraint._action_suffix(generated)
        if suffix is None:
            # Reasoning/preamble is unconstrained, but EOS may not terminate the
            # response before the mandatory environment-action channel exists.
            if len(generated) < self.pre_action_token_budget:
                for token_id in self.stop_token_ids:
                    scores[0, token_id] = float("-inf")
                return scores
            forced = generated[self.pre_action_token_budget:]
            if len(forced) >= len(self.action_open_tokens) or forced != self.action_open_tokens[: len(forced)]:
                raise RuntimeError("Harness201.1 mandatory action opener escaped compiled prefix")
            return self._force_only(scores, [int(self.action_open_tokens[len(forced)])])
        query_scores = self._search_query_phase(suffix, scores)
        if query_scores is not None:
            return query_scores
        nxt = self.constraint._next_from_targets(suffix, self.constraint.finite_targets)
        nxt.update(self.constraint._search_next(suffix))
        if not nxt:
            raise RuntimeError("Harness201.1 action prefix escaped all current action schemas")
        return self._force_only(scores, sorted(nxt))


class Harness2011Stage1InterfaceRuntime:
    """Shared Stage1 runtime with no learned carrier or adapter state."""

    def __init__(self, model: str, revision: str, *, dtype: str = "bfloat16",
                 chat_template_enable_thinking: bool | None = None) -> None:
        self.base = TransformersBackend(
            model,
            device="cuda",
            revision=revision,
            dtype=dtype,
            adapter_path=None,
            local_files_only=True,
        )
        self.dtype = dtype
        self.chat_template_enable_thinking = chat_template_enable_thinking

    def close(self) -> None:
        self.base.close()

    @property
    def active_adapters(self) -> list[str]:
        return []

    def prepare(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        self.base._load()  # same pinned local Transformers backend as the original Stage1
        tok = self.base._tok
        messages = [{"role": "user", "content": prompt}]
        template_kwargs: dict[str, Any] = {}
        if self.chat_template_enable_thinking is not None:
            template_kwargs["enable_thinking"] = bool(self.chat_template_enable_thinking)
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **template_kwargs,
        )
        decoded = tok.decode(enc["input_ids"][0], skip_special_tokens=False)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "harness": PROFILE,
            "messages": messages,
            "decoded_input": decoded,
            "input_tokens": int(enc["input_ids"].shape[1]),
            "carrier_injection": {
                "text_memory": False,
                "skill_bundle": False,
                "agent_system": False,
            },
            "active_adapters": [],
            "sd_adapter": None,
            "learned_context_tokens": 0,
            "chat_template_enable_thinking": self.chat_template_enable_thinking,
        }
        return enc, receipt

    def generate(
        self,
        prompt: str,
        *,
        available_actions: dict[str, Any],
        generation_seed: int,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.4,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ) -> tuple[str, dict[str, Any]]:
        import torch

        enc, receipt = self.prepare(prompt)
        model, tok = self.base._model, self.base._tok
        enc = {k: v.to(model.device) for k, v in enc.items()}
        n_in = int(enc["input_ids"].shape[1])
        constraint = Stage1ActionPrefixConstraint(tok, available_actions, n_in)
        device_index = getattr(model.device, "index", None)
        devices = [device_index] if getattr(model.device, "type", None) == "cuda" and device_index is not None else []
        from transformers import LogitsProcessorList
        configured_eos = getattr(model.generation_config, "eos_token_id", None)
        if configured_eos is None:
            stop_token_ids = [int(tok.eos_token_id)]
        elif isinstance(configured_eos, int):
            stop_token_ids = [int(configured_eos)]
        else:
            stop_token_ids = [int(x) for x in configured_eos]
        if int(tok.eos_token_id) not in stop_token_ids:
            stop_token_ids.append(int(tok.eos_token_id))
        processor = ActionHardLogitsProcessor(constraint, stop_token_ids=stop_token_ids)
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "repetition_penalty": float(repetition_penalty),
            "pad_token_id": tok.eos_token_id,
            "logits_processor": LogitsProcessorList([processor]),
        }
        if do_sample:
            kwargs.update(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(generation_seed))
            with torch.no_grad():
                out = model.generate(**enc, **kwargs)
        raw = tok.decode(out[0][n_in:], skip_special_tokens=True)
        action = extract_unique_action(raw)
        admissible = action_is_admissible(action, available_actions)
        receipt["sampling"] = {
            "do_sample": bool(do_sample),
            "temperature": float(temperature) if do_sample else None,
            "top_p": float(top_p) if do_sample else None,
            "top_k": int(top_k) if do_sample else None,
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens),
            "generation_seed": int(generation_seed),
        }
        receipt["action_channel"] = {
            "schema": "openevo.ceiling1-harness2011-action-channel.v1",
            "compiled_action_schema": constraint.schema,
            "action_channel_entered": constraint.open_end_token_index is not None,
            "action": action,
            "admissible": bool(admissible),
            "constraint_escape": False,
            "unconstrained_fallback": False,
            "fuzzy_repair_used": False,
            "reasoning_contains_cjk": bool(CJK_RE.search(raw.split(ACTION_OPEN, 1)[0])),
            "constraint_engine": "action-hard-logits-v2-mandatory-entry",
            "pre_action_token_budget": PRE_ACTION_TOKEN_BUDGET,
            "eos_before_action_forbidden": True,
            "forced_action_open_after_budget": True,
            "pre_action_blocked_stop_token_ids": stop_token_ids,
            "query_constraint_implementation": "cached-safe-set-mask-unsafe-v1",
        }
        return raw, receipt
