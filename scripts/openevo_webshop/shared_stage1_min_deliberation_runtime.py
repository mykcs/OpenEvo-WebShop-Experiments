#!/usr/bin/env python3
"""Shared runtime that guarantees a minimal model-authored pre-action deliberation."""
from __future__ import annotations

import re
from typing import Any

from scripts.openevo_webshop.ceiling1_harness2011_runtime import ACTION_OPEN
from scripts.openevo_webshop.q25_stage1_sampling_pilot_runtime import (
    NoStopBeforeActionLogitsProcessor,
)
from scripts.openevo_webshop.q25_stage1_seed_native_runtime import (
    SeedNativeNeutralStage1Runtime,
)

PROFILE = "harness201.1-stage1-shared-seed-native-min-deliberation-v1"
MIN_PRE_ACTION_MODEL_TOKENS = 4
_BLOCKED_FIRST_TOKEN_CACHE: dict[int, list[int]] = {}


class MinimumPreActionLogitsProcessor(NoStopBeforeActionLogitsProcessor):
    def __init__(self, stop_token_ids: list[int], tokenizer: Any, n_prompt: int,
                 min_tokens: int = MIN_PRE_ACTION_MODEL_TOKENS) -> None:
        super().__init__(stop_token_ids)
        self.tok = tokenizer
        self.n_prompt = int(n_prompt)
        self.min_tokens = int(min_tokens)
    @property
    def blocked_first_token_ids(self) -> list[int]:
        key=id(self.tok)
        cached=_BLOCKED_FIRST_TOKEN_CACHE.get(key)
        if cached is not None:
            return cached
        blocked=[]
        special={int(x) for x in getattr(self.tok,'all_special_ids',[])}
        for tid in range(len(self.tok)):
            piece=self.tok.decode([tid],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if tid in special or not piece.strip() or '<' in piece:
                blocked.append(int(tid))
        _BLOCKED_FIRST_TOKEN_CACHE[key]=blocked
        return blocked

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        scores=super().__call__(input_ids,scores)
        generated=int(input_ids.shape[1])-self.n_prompt
        if generated < self.min_tokens:
            scores[0,self.blocked_first_token_ids]=float('-inf')
        return scores


class SharedMinimumDeliberationRuntime(SeedNativeNeutralStage1Runtime):
    def _reasoning_logits_processor(self, tok: Any, n_prompt: int,
                                    stop_token_ids: list[int]) -> Any:
        return MinimumPreActionLogitsProcessor(
            stop_token_ids, tok, n_prompt, MIN_PRE_ACTION_MODEL_TOKENS
        )
    def prepare(self, prompt: str) -> tuple[dict[str,Any],dict[str,Any]]:
        enc,receipt=super().prepare(prompt)
        receipt['harness']=PROFILE
        contract=dict(receipt.get('successor_contract') or {})
        contract.update({
            'minimum_pre_action_model_tokens':MIN_PRE_ACTION_MODEL_TOKENS,
            'minimum_deliberation_enforced_by_interface':True,
            'deliberation_content_guidance':False,
            'think_tags_required':False,
            'deliberation_word_cap':None,
        })
        receipt['successor_contract']=contract
        return enc,receipt

    def generate(self,*args:Any,**kwargs:Any):
        raw,receipt=super().generate(*args,**kwargs)
        pre=re.split(re.escape(ACTION_OPEN),str(raw),maxsplit=1,flags=re.I)[0].strip()
        contract=dict(receipt.get('successor_contract') or {})
        contract['observed_pre_action_nonempty']=bool(pre)
        contract['observed_pre_action_word_count']=len(pre.split()) if pre else 0
        receipt['successor_contract']=contract
        return raw,receipt
