#!/usr/bin/env python3
"""Qwen3 binding for the shared Harness201.1 runtime.

Scientific behavior is inherited from Harness2011Runtime. The only model-specific
change is explicit chat-template enable_thinking=False at every prompt encoding.
"""
from __future__ import annotations
from typing import Any
from scripts.openevo_webshop.ceiling1_harness2011_runtime import Harness2011Runtime
from scripts.openevo_webshop.ceiling1_harness2_runtime import H201_AUTHORITY_PREAMBLE, build_carrier_readback, govern_context


class Qwen3Harness2011Runtime(Harness2011Runtime):
    """Harness201.1 with Qwen3's frozen chat-template thinking policy."""

    def prepare_h2(self, prompt: str, *, available_actions: dict[str, Any]):
        page_state = self._thin_page_state(available_actions)
        agent, skill, memory, route = self._thin_inputs(page_state)
        tok = self.base._tok

        def count(text: str) -> int:
            return len(tok(text, add_special_tokens=False)["input_ids"])

        governed = govern_context(
            environment_text=prompt,
            recent_history="",
            agent=agent,
            skill=skill,
            memory=memory,
            token_counter=count,
            caps=self.h2_caps,
        )
        selected = governed["selected"]
        non_global_selected = sum(int(k in selected) for k in ("skill_bundle", "text_memory"))
        if non_global_selected > 1:
            raise RuntimeError("Harness 2.0.1 thin contract selected more than one non-global carrier")
        system = H201_AUTHORITY_PREAMBLE
        if selected.get("agent_system"):
            system += "\nHistorical global policy:\n" + selected["agent_system"]
        user_parts = []
        if selected.get("text_memory"):
            user_parts.append("Historical reusable knowledge (lower authority than current environment):\n" + selected["text_memory"])
        if selected.get("skill_bundle"):
            user_parts.append("Currently relevant historical strategy (apply only when its preconditions hold):\n" + selected["skill_bundle"])
        user_parts.append("CURRENT ENVIRONMENT — AUTHORITATIVE:\n" + prompt)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(user_parts)}]
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
        encoded_tokens = int(enc["input_ids"].shape[1])
        if encoded_tokens > int(self.h2_caps.total_tokens):
            raise RuntimeError(f"Harness 2.0.1 encoded input {encoded_tokens} exceeds hard cap {self.h2_caps.total_tokens}")
        decoded = tok.decode(enc["input_ids"][0], skip_special_tokens=False)
        receipt = {
            "schema": "openevo.ceiling1-harness2-context-receipt.v2.0.1-qwen3",
            "page_state": page_state,
            "thin_route": route,
            "messages": messages,
            "decoded_input": decoded,
            "input_tokens": encoded_tokens,
            "encoded_budget": {"status": "PASS", "hard_cap_tokens": int(self.h2_caps.total_tokens), "encoded_tokens": encoded_tokens},
            "context_governor": governed,
            "carrier_lint": self.h2_lint,
            "carrier_readback": build_carrier_readback(
                requested={"text_memory": memory, "skill_bundle": skill, "agent_system": agent},
                selected=selected, governed=governed,
            ),
            "active_adapters": self.active_adapters,
            "adapters": list(self._adapter_receipts),
            "harness": self.PROFILE,
            "chat_template_enable_thinking": False,
        }
        return enc, receipt

    @staticmethod
    def _thin_page_state(available_actions: dict[str, Any]) -> str:
        from scripts.openevo_webshop.ceiling1_harness2_runtime import classify_page_state
        return classify_page_state(available_actions)
