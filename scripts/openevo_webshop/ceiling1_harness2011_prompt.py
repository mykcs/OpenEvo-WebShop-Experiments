#!/usr/bin/env python3
"""Harness201.1 prompt contract: reasoning soft, action hard."""
from __future__ import annotations
import hashlib
from typing import Any
from openevo_webshop.comparison_contract import render_compact_webshop_prompt
from scripts.openevo_webshop.ceiling1_harness201_prompt import BASE_INSTRUCTION

H2011_INSTRUCTION = (
    "Keep reasoning under 30 words; prefer English for consistency. Output exactly two tagged lines: "
    "<think>brief reason</think> and <action>one current admissible action</action>. "
    "Reasoning language does not determine action validity. Do not list products, repeat the instruction, "
    "or write text after </action>."
)
PROMPT_PROFILE = "harness201.1-reasoning-soft-action-hard-v1"
PROMPT_CONTRACT_SHA256 = hashlib.sha256(H2011_INSTRUCTION.encode("utf-8")).hexdigest()


def render_harness2011_compact_webshop_prompt(**kwargs: Any) -> str:
    prompt = render_compact_webshop_prompt(**kwargs)
    if prompt.count(BASE_INSTRUCTION) != 1:
        raise RuntimeError("Harness201.1 base compact prompt contract drift")
    return prompt.replace(BASE_INSTRUCTION, H2011_INSTRUCTION, 1)
