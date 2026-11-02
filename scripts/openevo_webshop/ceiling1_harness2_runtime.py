#!/usr/bin/env python3
"""Harness 2.0 candidate runtime. Never grants formal-task authority."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.ceiling1_harness2 import ContextCaps, govern_context, lint_carriers

H201_AUTHORITY_PREAMBLE = """OpenEVO Harness 2.0.1 authority:\n- Current task, current observation, and current Allowed actions are the sole authority for current-state facts and legal actions.\n"""

AUTHORITY_PREAMBLE = """OpenEVO Harness 2.0 authority:\n- The current task, current observation, and current Allowed actions are environment truth.\n- Environment truth overrides all historical memory, skills, and agent guidance.\n- Historical carriers provide reusable strategy only; they never create an action that is absent from the current Allowed actions.\n- Choose the next action yourself from the current environment.\n"""

SECTION_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
STATE_HEADINGS = {
    "search": ("Search State", "Any State"),
    "results": ("Results State", "Any State"),
    "product": ("Product State", "Any State"),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


CARRIER_KEYS = ("text_memory", "skill_bundle", "agent_system")


def build_carrier_readback(*, requested: dict[str, str], selected: dict[str, str], governed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build one shared requested/selected/readback identity record for all runtimes."""
    decisions = governed.get("decisions") or {}
    out: dict[str, dict[str, Any]] = {}
    for key in CARRIER_KEYS:
        requested_payload = requested.get(key) or ""
        selected_payload = selected.get(key) or ""
        requested_flag = bool(requested_payload)
        selected_flag = key in selected
        decision = decisions.get(key) or {}
        if decision:
            if bool(decision.get("requested")) != requested_flag or bool(decision.get("selected")) != selected_flag:
                raise ValueError(f"carrier governor/readback decision drift for {key}")
        requested_sha = sha256_text(requested_payload) if requested_flag else None
        selected_sha = sha256_text(selected_payload) if selected_flag else None
        reason = decision.get("reason")
        if reason is None:
            reason = "selected" if selected_flag else ("not_requested" if not requested_flag else "unexplained_drop")
        out[key] = {
            "requested": requested_flag,
            "selected": selected_flag,
            "sha256": requested_sha,
            "router_requested": requested_flag,
            "router_selected": selected_flag,
            "requested_payload_sha256": requested_sha,
            "selected_payload_sha256": selected_sha,
            "runtime_readback_payload_sha256": selected_sha,
            "requested_tokens": decision.get("requested_tokens"),
            "selected_tokens": decision.get("selected_tokens"),
            "governor_reason": reason,
        }
    return out


def classify_page_state(available_actions: dict[str, Any]) -> str:
    clickables = [str(x).strip().lower() for x in (available_actions.get("clickables") or [])]
    if available_actions.get("has_search_bar"):
        return "search"
    if any(x == "buy now" or x.startswith("description") or x.startswith("features") for x in clickables):
        return "product"
    return "results"


def markdown_sections(text: str) -> dict[str, str]:
    matches=list(SECTION_RE.finditer(text))
    if not matches:
        return {}
    out={}
    for i,m in enumerate(matches):
        start=m.end(); end=matches[i+1].start() if i+1<len(matches) else len(text)
        out[m.group(1).strip()] = text[start:end].strip()
    return out


def select_relevant_skill(skill: str, page_state: str) -> str:
    sections=markdown_sections(skill)
    wanted=STATE_HEADINGS.get(page_state,("Any State",))
    kept=[]
    for heading in wanted:
        body=sections.get(heading,"").strip()
        if body:
            kept.append(f"## {heading}\n{body}")
    return "\n\n".join(kept).strip()


def required_safe_structure(memory: str, skill: str, agent: str) -> list[str]:
    errors=[]
    if "# Memory" not in memory or "## Reusable Knowledge" not in memory:
        errors.append("memory missing # Memory / ## Reusable Knowledge")
    for h in ("# Skill Bundle","## Search State","## Results State","## Product State","## Any State"):
        if h not in skill: errors.append(f"skill missing {h}")
    for h in ("# Agent System","## Authority","## Coordination"):
        if h not in agent: errors.append(f"agent missing {h}")
    return errors


class Harness2Runtime:
    """Composition wrapper over CeilingRuntime with Harness-2.0 context composition."""
    def __init__(self, model: str, revision: str, *, sd_adapter: Path|None,
                 memory: str, skill_bundle: str, agent_system: str,
                 total_input_tokens: int = 4096) -> None:
        self.h2_memory=memory.strip(); self.h2_skill=skill_bundle.strip(); self.h2_agent=agent_system.strip()
        structural=required_safe_structure(self.h2_memory,self.h2_skill,self.h2_agent)
        if structural: raise ValueError("; ".join(structural))
        lint=lint_carriers(memory=self.h2_memory,skill=self.h2_skill,agent=self.h2_agent)
        if lint["status"]!="PASS": raise ValueError("carrier lint failed: "+"; ".join(lint["violations"]))
        self.h2_lint=lint
        self.h2_caps=ContextCaps(total_tokens=int(total_input_tokens))
        from scripts.openevo_webshop.ceiling1_runtime import CeilingRuntime
        self._inner=CeilingRuntime(model,revision,sd_adapter=sd_adapter,memory="",skill_bundle="",agent_system="")
        self.base=self._inner.base
        self._adapter_receipts=self._inner._adapter_receipts

    @property
    def active_adapters(self) -> list[str]:
        return self._inner.active_adapters

    def close(self) -> None:
        self._inner.close()

    def prepare_h2(self, prompt: str, *, available_actions: dict[str,Any]) -> tuple[dict[str,Any],dict[str,Any]]:
        page_state=classify_page_state(available_actions)
        relevant_skill=select_relevant_skill(self.h2_skill,page_state)
        tok=self.base._tok
        def count(text:str)->int:
            return len(tok(text,add_special_tokens=False)["input_ids"])
        governed=govern_context(
            environment_text=prompt,recent_history="",agent=self.h2_agent,
            skill=relevant_skill,memory=self.h2_memory,token_counter=count,caps=self.h2_caps,
        )
        selected=governed["selected"]
        system=AUTHORITY_PREAMBLE
        if selected.get("agent_system"):
            system += "\nHistorical global policy:\n"+selected["agent_system"]
        user_parts=[]
        if selected.get("text_memory"):
            user_parts.append("Historical reusable knowledge (lower authority than current environment):\n"+selected["text_memory"])
        if selected.get("skill_bundle"):
            user_parts.append("Currently relevant historical strategy (apply only when its preconditions hold):\n"+selected["skill_bundle"])
        user_parts.append("CURRENT ENVIRONMENT — AUTHORITATIVE:\n"+prompt)
        messages=[{"role":"system","content":system},{"role":"user","content":"\n\n".join(user_parts)}]
        enc=tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt",return_dict=True)
        decoded=tok.decode(enc["input_ids"][0],skip_special_tokens=False)
        receipt={
            "schema":"openevo.ceiling1-harness2-context-receipt.v1",
            "page_state":page_state,
            "messages":messages,
            "decoded_input":decoded,
            "input_tokens":int(enc["input_ids"].shape[1]),
            "context_governor":governed,
            "carrier_lint":self.h2_lint,
            "carrier_readback": build_carrier_readback(
                requested={"text_memory": self.h2_memory, "skill_bundle": relevant_skill, "agent_system": self.h2_agent},
                selected=selected, governed=governed,
            ),
            "active_adapters":self.active_adapters,
            "adapters":list(self._adapter_receipts),
            "harness":"2.0-candidate",
        }
        return enc,receipt

    def generate_h2(self,prompt:str,*,available_actions:dict[str,Any],generation_seed:int,
                    max_new_tokens:int=512,do_sample:bool=True,temperature:float=1.0,
                    top_p:float=1.0,top_k:int=0,repetition_penalty:float=1.0):
        import torch
        enc,receipt=self.prepare_h2(prompt,available_actions=available_actions)
        model,tok=self.base._model,self.base._tok
        enc={k:v.to(model.device) for k,v in enc.items()}
        n_in=int(enc["input_ids"].shape[1]); device_index=getattr(model.device,"index",None)
        devices=[device_index] if getattr(model.device,"type",None)=="cuda" and device_index is not None else []
        kwargs={"max_new_tokens":int(max_new_tokens),"do_sample":bool(do_sample),"repetition_penalty":float(repetition_penalty),"pad_token_id":tok.eos_token_id}
        if do_sample: kwargs.update(temperature=float(temperature),top_p=float(top_p),top_k=int(top_k))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(generation_seed))
            with torch.no_grad(): out=model.generate(**enc,**kwargs)
        receipt["sampling"]={"do_sample":bool(do_sample),"temperature":float(temperature) if do_sample else None,"top_p":float(top_p) if do_sample else None,"top_k":int(top_k) if do_sample else None,"repetition_penalty":float(repetition_penalty),"max_new_tokens":int(max_new_tokens)}
        return tok.decode(out[0][n_in:],skip_special_tokens=True),receipt


class Harness201Runtime(Harness2Runtime):
    """Harness 2.0.1: English-aligned, action-noun-safe, truly thin composition."""
    PROFILE = "2.0.1-thin"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        strict = lint_carriers(memory=self.h2_memory, skill=self.h2_skill, agent=self.h2_agent, forbid_action_like_abstractions=True)
        if strict["status"] != "PASS":
            raise ValueError("Harness 2.0.1 action-like carrier lint failed: " + "; ".join(strict["violations"]))
        self.h2_lint = strict
        self.h2_caps = ContextCaps(total_tokens=self.h2_caps.total_tokens, agent_tokens=96, skill_tokens=160, memory_tokens=192, carrier_fraction_max=0.20)

    def _thin_inputs(self, page_state: str) -> tuple[str, str, str, dict[str, str]]:
        """Always keep tiny Agent authority; admit at most one non-global carrier."""
        agent = self.h2_agent
        skill = ""
        memory = ""
        route = {"agent_system": "always-global-authority", "non_global": "none"}
        if page_state == "search":
            candidate = select_relevant_skill(self.h2_skill, "search")
            candidate = candidate.replace("## Any State\n- No additional guidance.", "").strip()
            if candidate and candidate != "## Search State\n- No additional guidance.":
                skill = candidate
                route["non_global"] = "skill_bundle"
        elif page_state in {"results", "product"}:
            memory = self.h2_memory
            route["non_global"] = "text_memory"
        return agent, skill, memory, route

    def prepare_h2(self, prompt: str, *, available_actions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        page_state = classify_page_state(available_actions)
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
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        encoded_tokens = int(enc["input_ids"].shape[1])
        if encoded_tokens > int(self.h2_caps.total_tokens):
            raise RuntimeError(f"Harness 2.0.1 encoded input {encoded_tokens} exceeds hard cap {self.h2_caps.total_tokens}")
        decoded = tok.decode(enc["input_ids"][0], skip_special_tokens=False)
        receipt = {
            "schema": "openevo.ceiling1-harness2-context-receipt.v2.0.1",
            "page_state": page_state,
            "thin_route": route,
            "messages": messages,
            "decoded_input": decoded,
            "input_tokens": encoded_tokens,
            "encoded_budget": {"status":"PASS","hard_cap_tokens":int(self.h2_caps.total_tokens),"encoded_tokens":encoded_tokens},
            "context_governor": governed,
            "carrier_lint": self.h2_lint,
            "carrier_readback": build_carrier_readback(
                requested={"text_memory": memory, "skill_bundle": skill, "agent_system": agent},
                selected=selected, governed=governed,
            ),
            "active_adapters": self.active_adapters,
            "adapters": list(self._adapter_receipts),
            "harness": self.PROFILE,
        }
        return enc, receipt
