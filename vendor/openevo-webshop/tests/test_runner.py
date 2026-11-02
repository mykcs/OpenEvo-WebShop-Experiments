"""Smoke test for runner episode loop using a fake backend + fake env.

Verifies the runner produces the full episode-record contract (goal non-empty,
steps>=1, task_score is float, won is bool) and that the D6 evolution-injection
point places injected text into the final prompt (acceptance C3), all without
a real model or WebShop install.
"""

from __future__ import annotations

import os
import sys
import types

from openevo_webshop.runner import inject_evolution, run_episode
from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig


class _FakeBackend:
    def __init__(self, action_text="<action>search[shoe]</action>"):
        self.model_name = "fake-model"
        self.prompts = []
        self._action_text = action_text

    def generate(self, prompt: str, max_new: int = 128, **kwargs) -> str:
        self.prompts.append(prompt)
        return self._action_text


class _FakeGymEnv:
    def __init__(self):
        self._n = 0

    def reset(self, session=0):
        return "Instruction:  [SEP]  buy a shoe  [SEP]", {}

    def step(self, action):
        self._n += 1
        done = self._n >= 2
        reward = 1.0 if done else 0.0
        return "OBS", reward, done, {}

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": []}

    def close(self):
        pass


def _env(monkeypatch, tmp_path):
    gym_mod = types.ModuleType("gym")
    fake = _FakeGymEnv()
    gym_mod.make = lambda *a, **k: fake
    monkeypatch.setitem(sys.modules, "gym", gym_mod)
    monkeypatch.setitem(sys.modules, "web_agent_site", types.ModuleType("web_agent_site"))
    envs = types.ModuleType("web_agent_site.envs")
    envs.WebAgentTextEnv = object
    monkeypatch.setitem(sys.modules, "web_agent_site.envs", envs)
    cfg = WebshopEnvConfig(seed_root=str(tmp_path))
    os.makedirs(cfg.webshop_dir, exist_ok=True)
    return WebshopEnv(cfg)


def test_inject_evolution_appends_and_is_greppable():
    base = "PROMPT"
    out = inject_evolution(base, "remember: prefer cheap shoes")
    assert "remember: prefer cheap shoes" in out
    assert out.startswith("PROMPT")
    assert inject_evolution(base, None) == "PROMPT"


def test_reset_accepts_legacy_webshop_env_signature(monkeypatch, tmp_path):
    env = _env(monkeypatch, tmp_path)

    obs, info = env.reset(0, seed=2026)

    assert "buy a shoe" in obs
    assert info["won"] is False
    assert "available_actions" in info


def test_run_episode_full_contract_and_evolution_in_prompt(monkeypatch, tmp_path):
    env = _env(monkeypatch, tmp_path)
    backend = _FakeBackend()
    res = run_episode(env, backend, idx=0, evolution_text="EVO-MEMORY-123")

    assert res.goal != ""
    assert res.steps >= 1
    assert isinstance(res.task_score, float)
    assert isinstance(res.won, bool)
    assert res.won is True

    assert any("EVO-MEMORY-123" in p for p in backend.prompts)
    assert any("[evolution]" in p for p in backend.prompts)
