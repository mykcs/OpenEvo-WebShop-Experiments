#!/usr/bin/env python3
"""Final candidate runtime for the Qwen2.5-3B Stage1 successor.

Only interface mechanics change: the action opener is interface-owned and search
queries have a liveness bound. WebShop action choice remains model-controlled.
"""
from __future__ import annotations
from typing import Any

from scripts.openevo_webshop.q25_stage1_sampling_pilot_runtime import (
    PilotHarness2011Stage1InterfaceRuntime,
    SEARCH_QUERY_TOKEN_BUDGET,
)

PROFILE = "harness201.1-stage1-successor-format-hard-history-faithful-v1"


class SuccessorHarness2011Stage1InterfaceRuntime(PilotHarness2011Stage1InterfaceRuntime):
    def prepare(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        enc, receipt = super().prepare(prompt)
        receipt["harness"] = PROFILE
        receipt["successor_contract"] = {
            "interface_owned_action_opener": True,
            "search_query_token_budget": SEARCH_QUERY_TOKEN_BUDGET,
            "strategy_guidance_injected": False,
        }
        return enc, receipt
