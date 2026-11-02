#!/usr/bin/env python3
"""Format-hard Stage1 runtime paired with SEED-native neutral observation representation."""
from __future__ import annotations
from typing import Any
from scripts.openevo_webshop.q25_stage1_successor_runtime import (
    SuccessorHarness2011Stage1InterfaceRuntime,
)

PROFILE = "harness201.1-stage1-seed-native-neutral-format-hard-v1"


class SeedNativeNeutralStage1Runtime(SuccessorHarness2011Stage1InterfaceRuntime):
    def prepare(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        enc, receipt = super().prepare(prompt)
        receipt["harness"] = PROFILE
        contract = dict(receipt.get("successor_contract") or {})
        contract.update({
            "observation_representation": "SEED WebshopEnvironmentManager.format_obs",
            "history_representation": "SEED SimpleMemory full formatted pre-action observation + action",
            "strategy_guidance_injected": False,
        })
        receipt["successor_contract"] = contract
        return enc, receipt
