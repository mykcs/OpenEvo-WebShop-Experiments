#!/usr/bin/env python3
"""Qualification-local shared prompt contract for Harness 2.0.1.

This module deliberately does not mutate the repository-wide WebShop prompt.
Both paired cells call the same renderer; only the 2.0.1 qualification uses it.
"""
from __future__ import annotations
import hashlib
from typing import Any
from openevo_webshop.comparison_contract import render_compact_webshop_prompt

BASE_INSTRUCTION = (
    "Keep reasoning under 30 words. Output exactly two tagged lines: "
    "<think>brief reason</think> and <action>one admissible action</action>. "
    "Do not list products, repeat the instruction, or write text after </action>."
)
ENGLISH_ONLY_INSTRUCTION = (
    "Keep reasoning under 30 words, in English only. Output exactly two tagged lines: "
    "<think>brief English reason</think> and <action>one admissible English action</action>. "
    "CJK text is invalid under the frozen parser contract. Do not list products, repeat the instruction, "
    "or write text after </action>."
)
PROMPT_PROFILE = "harness2.0.1-english-only-shared-v1"
PROMPT_CONTRACT_SHA256 = hashlib.sha256(ENGLISH_ONLY_INSTRUCTION.encode("utf-8")).hexdigest()


def render_harness201_compact_webshop_prompt(**kwargs: Any) -> str:
    prompt = render_compact_webshop_prompt(**kwargs)
    if prompt.count(BASE_INSTRUCTION) != 1:
        raise RuntimeError("Harness 2.0.1 base compact prompt contract drift")
    return prompt.replace(BASE_INSTRUCTION, ENGLISH_ONLY_INSTRUCTION, 1)


def upgrade_legacy_compact_prompt_to_harness201(prompt: str) -> str:
    """Upgrade only the output-contract sentence of a recorded compact prompt."""
    text=str(prompt)
    if text.count(BASE_INSTRUCTION)!=1:
        raise RuntimeError("recorded compact prompt contract drift")
    return text.replace(BASE_INSTRUCTION, ENGLISH_ONLY_INSTRUCTION, 1)
