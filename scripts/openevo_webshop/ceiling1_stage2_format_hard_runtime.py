#!/usr/bin/env python3
"""Shared Stage2 Harness201.1 format envelope for 3B and Qwen3-1.7B.

Scientific semantics are unchanged: task schedule, sampling, carrier content,
reward, and model action choice stay model-controlled. This module only owns
the literal <action> opener and hard-constrains the action body to the current
Allowed-actions schema, matching the already-qualified Stage1 interface.
"""
from __future__ import annotations

import re
from typing import Any

from scripts.openevo_webshop.ceiling1_harness2011_runtime import (
    ACTION_OPEN,
    Harness2011Runtime,
    action_is_admissible,
    extract_unique_action,
)
from scripts.openevo_webshop.ceiling1_harness2011_runtime_qwen3 import (
    Qwen3Harness2011Runtime,
)
from scripts.openevo_webshop.ceiling1_stage1_harness2011_interface_runtime import (
    PRE_ACTION_TOKEN_BUDGET,
    Stage1ActionPrefixConstraint,
)
from scripts.openevo_webshop.q25_stage1_sampling_pilot_runtime import (
    ActionOpenStoppingCriteria,
    PilotActionHardLogitsProcessor,
    SEARCH_QUERY_TOKEN_BUDGET,
)
from scripts.openevo_webshop.shared_stage1_min_deliberation_runtime import (
    MIN_PRE_ACTION_MODEL_TOKENS,
    MinimumPreActionLogitsProcessor,
)


class Stage2FormatHardRuntime(Harness2011Runtime):
    """Harness201.1 with the shared two-phase interface-owned action channel."""

    def _reasoning_logits_processor(
        self, tok: Any, n_prompt: int, stop_token_ids: list[int]
    ) -> Any:
        return MinimumPreActionLogitsProcessor(
            stop_token_ids, tok, n_prompt, MIN_PRE_ACTION_MODEL_TOKENS
        )

    def generate_h2(
        self,
        prompt: str,
        *,
        available_actions: dict[str, Any],
        generation_seed: int,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ):
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
            common.update(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )

        device_index = getattr(model.device, "index", None)
        devices = (
            [device_index]
            if getattr(model.device, "type", None) == "cuda"
            and device_index is not None
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(generation_seed))
            with torch.no_grad():
                reasoning_out = model.generate(
                    **enc,
                    max_new_tokens=PRE_ACTION_TOKEN_BUDGET,
                    logits_processor=LogitsProcessorList(
                        [
                            self._reasoning_logits_processor(
                                tok, n_prompt, stop_token_ids
                            )
                        ]
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
                reasoning_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(device=model.device, dtype=prompt_ids.dtype)
            opener_ids = tok(
                ACTION_OPEN,
                add_special_tokens=False,
                return_tensors="pt",
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
        receipt["harness"] = "201.1"
        receipt["action_channel"] = {
            "schema": "openevo.ceiling1-harness2011-action-channel.stage2-interface-owned-v1",
            "compiled_action_schema": constraint.schema,
            "action_channel_entered": True,
            "action": action,
            "admissible": bool(admissible),
            "constraint_escape": False,
            "unconstrained_fallback": False,
            "fuzzy_repair_used": False,
            "reasoning_contains_cjk": bool(
                re.search(r"[\u3400-\u9fff\uf900-\ufaff]", reasoning_text)
            ),
            "constraint_engine": "shared-stage2-two-phase-format-envelope-v1",
            "interface_owned_action_opener": True,
            "minimum_pre_action_model_tokens": MIN_PRE_ACTION_MODEL_TOKENS,
            "pre_action_token_budget": PRE_ACTION_TOKEN_BUDGET,
            "observed_pre_action_model_tokens": int(reasoning_ids.shape[1]),
            "search_query_token_budget": SEARCH_QUERY_TOKEN_BUDGET,
            "model_action_choice_preserved": True,
            "strategy_guidance_injected": False,
        }
        return raw, receipt


class Qwen3Stage2FormatHardRuntime(Stage2FormatHardRuntime):
    """Same Stage2 envelope with Qwen3's frozen thinking-off prompt encoding."""

    prepare_h2 = Qwen3Harness2011Runtime.prepare_h2
    _thin_page_state = staticmethod(Qwen3Harness2011Runtime._thin_page_state)
