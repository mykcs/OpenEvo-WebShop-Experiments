#!/usr/bin/env python3
"""3B Stage1 sampling pilot runtime with format-only hard action envelope.

This pilot does not rank, recommend, repair, or otherwise choose WebShop
actions. It only makes the already-preregistered <action> channel truly hard:
reasoning is free, the interface owns the literal opening delimiter, and the
action body is constrained to the current environment schema.
"""
from __future__ import annotations

import re
from typing import Any

from scripts.openevo_webshop.ceiling1_stage1_harness2011_interface_runtime import (
    CJK_RE,
    PRE_ACTION_TOKEN_BUDGET,
    Stage1ActionPrefixConstraint,
    Harness2011Stage1InterfaceRuntime,
    ActionHardLogitsProcessor,
)
from scripts.openevo_webshop.ceiling1_harness2011_runtime import (
    ACTION_OPEN,
    action_is_admissible,
    extract_unique_action,
)

PROFILE = "201.1-stage1-interface-only-format-hard-pilot"
SEARCH_QUERY_TOKEN_BUDGET = 256


class NoStopBeforeActionLogitsProcessor:
    """Keep the reasoning channel alive until action handoff or its fixed budget."""

    def __init__(self, stop_token_ids: list[int]) -> None:
        self.stop_token_ids = sorted({int(x) for x in stop_token_ids})

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        for token_id in self.stop_token_ids:
            scores[0, token_id] = float("-inf")
        return scores


class ActionOpenStoppingCriteria:
    """Stop free reasoning as soon as the decoded literal action opener appears."""

    def __init__(self, tokenizer: Any, n_input: int) -> None:
        self.tok = tokenizer
        self.n_input = int(n_input)

    def __call__(self, input_ids: Any, scores: Any, **_: Any) -> bool:
        ids = [int(x) for x in input_ids[0, self.n_input:].tolist()]
        text = self.tok.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return ACTION_OPEN in text.lower()


class PilotActionHardLogitsProcessor(ActionHardLogitsProcessor):
    """Base action grammar plus a generous syntax-only search closure budget."""

    def _search_query_phase(self, suffix: list[int], scores: Any) -> Any | None:
        c = self.constraint
        if not c.schema["search_allowed"] or suffix[: len(c.search_prefix)] != c.search_prefix:
            return None
        tail = suffix[len(c.search_prefix):]
        first_non_query = next(
            (i for i, tid in enumerate(tail) if tid not in c.safe_query_set), None
        )
        if first_non_query is not None:
            return None
        if len(tail) >= SEARCH_QUERY_TOKEN_BUDGET:
            return self._force_only(scores, [int(c.search_close[0])])
        return super()._search_query_phase(suffix, scores)


class PilotHarness2011Stage1InterfaceRuntime(Harness2011Stage1InterfaceRuntime):
    """Two-phase format envelope; WebShop strategy remains entirely model-chosen."""

    def _reasoning_logits_processor(self, tok: Any, n_prompt: int, stop_token_ids: list[int]) -> Any:
        return NoStopBeforeActionLogitsProcessor(stop_token_ids)

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
        from transformers import LogitsProcessorList, StoppingCriteriaList

        enc, receipt = self.prepare(prompt)
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
            common.update(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )

        device_index = getattr(model.device, "index", None)
        devices = (
            [device_index]
            if getattr(model.device, "type", None) == "cuda" and device_index is not None
            else []
        )

        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(generation_seed))
            with torch.no_grad():
                reasoning_out = model.generate(
                    **enc,
                    max_new_tokens=PRE_ACTION_TOKEN_BUDGET,
                    logits_processor=LogitsProcessorList(
                        [self._reasoning_logits_processor(tok, n_prompt, stop_token_ids)]
                    ),
                    stopping_criteria=StoppingCriteriaList(
                        [ActionOpenStoppingCriteria(tok, n_prompt)]
                    ),
                    **common,
                )

            reasoning_raw = tok.decode(
                reasoning_out[0][n_prompt:],
                skip_special_tokens=True,
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
            action_input_ids = torch.cat(
                [prompt_ids, reasoning_ids, opener_ids], dim=1
            )
            n_constraint_input = n_prompt + int(reasoning_ids.shape[1])
            constraint = Stage1ActionPrefixConstraint(
                tok, available_actions, n_constraint_input
            )
            processor = PilotActionHardLogitsProcessor(
                constraint, stop_token_ids=stop_token_ids
            )
            action_budget = max(
                32, int(max_new_tokens) - int(reasoning_ids.shape[1])
            )
            with torch.no_grad():
                action_out = model.generate(
                    input_ids=action_input_ids,
                    attention_mask=torch.ones_like(action_input_ids),
                    max_new_tokens=action_budget,
                    logits_processor=LogitsProcessorList([processor]),
                    **common,
                )

        action_tail = tok.decode(
            action_out[0][int(action_input_ids.shape[1]):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw = reasoning_text + ACTION_OPEN + action_tail
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
            "schema": "openevo.ceiling1-harness2011-action-channel.pilot-v1",
            "compiled_action_schema": constraint.schema,
            "action_channel_entered": True,
            "action": action,
            "admissible": bool(admissible),
            "constraint_escape": False,
            "unconstrained_fallback": False,
            "fuzzy_repair_used": False,
            "reasoning_contains_cjk": bool(CJK_RE.search(reasoning_text)),
            "constraint_engine": "action-hard-two-phase-format-envelope-pilot-v1",
            "pre_action_token_budget": PRE_ACTION_TOKEN_BUDGET,
            "search_query_token_budget": SEARCH_QUERY_TOKEN_BUDGET,
            "interface_owned_action_opener": True,
            "model_action_choice_preserved": True,
            "strategy_guidance_injected": False,
        }
        return raw, receipt
