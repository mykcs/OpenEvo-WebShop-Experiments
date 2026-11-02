"""Single-process WebShop environment wrapper (Ray-free).

Adapts SEED's Ray actor ``WebshopWorker``
(``seed-workspace/SEED/agent_system/environments/env_package/webshop/envs.py``)
into a plain in-process class. Reuses SEED's info augmentation and win
semantics verbatim:

- ``step()`` augments ``info`` with ``available_actions`` / ``task_score``
  (the raw continuous 0..1 reward) / ``won``.
- ``won`` is True iff ``done and reward == 1.0``. The returned ``reward``
  follows SEED's binarized semantics (10.0 on win else 0.0); the continuous
  score is carried separately in ``info['task_score']``.

gym/webshop are lazily imported (decision D1) so the package can be imported
and its interface-contract tests run on a bare python without gym installed.
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WebshopEnvConfig:
    """Construction config for a WebShop env (mirrors webshop_rollout.make_env)."""

    seed_root: str  # path to the SEED repo root (contains agent_system/...)
    num_products: int = 100
    seed: int = 2026
    observation_mode: str = "text"
    data_subdir: str = "data"
    disable_env_checker: bool = True
    extra_env_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def webshop_dir(self) -> str:
        return os.path.join(
            self.seed_root, "agent_system/environments/env_package/webshop/webshop"
        )

    @property
    def data_dir(self) -> str:
        return os.path.join(self.webshop_dir, self.data_subdir)


class WebshopEnv:
    """Ray-free, single-process WebShop env.

    Interface:
        reset(idx) -> (obs, info)
        step(action) -> (obs, reward, done, info)
        get_available_actions() -> dict
        close() -> None
    """

    def __init__(self, config: WebshopEnvConfig) -> None:
        self.config = config
        self._env = None  # lazy

    def _build(self):
        if self._env is not None:
            return self._env
        cfg = self.config
        # Match webshop_rollout.py:102-111 setup order.
        if cfg.seed_root not in sys.path:
            sys.path.insert(0, cfg.seed_root)
        # web_agent_site lives inside webshop_dir; chdir alone does NOT put it
        # on sys.path for a plain `python script.py` process (sys.path[0] is the
        # script's dir, not the cwd), so add it explicitly.
        if cfg.webshop_dir not in sys.path:
            sys.path.insert(0, cfg.webshop_dir)
        os.chdir(cfg.webshop_dir)

        import gym  # noqa: WPS433 (lazy by design)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: F401,WPS433

        self._env = gym.make(
            "WebAgentTextEnv-v0",
            observation_mode=cfg.observation_mode,
            file_path=os.path.join(cfg.data_dir, "items_shuffle_1000.json"),
            attr_path=os.path.join(cfg.data_dir, "items_ins_v2_1000.json"),
            num_products=cfg.num_products,
            seed=cfg.seed,
            disable_env_checker=cfg.disable_env_checker,
            **cfg.extra_env_kwargs,
        )
        return self._env

    def reset(self, idx: int, seed: int | None = None):
        """Reset with the given session index. Returns (obs, info)."""
        env = self._build()
        reset_parameters = inspect.signature(env.reset).parameters
        reset_kwargs = {"session": idx}
        if "seed" in reset_parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in reset_parameters.values()
        ):
            reset_kwargs["seed"] = self.config.seed if seed is None else int(seed)
        obs, info = env.reset(**reset_kwargs)
        info = dict(info or {})
        info["available_actions"] = env.get_available_actions()
        info["won"] = False
        return obs, info

    def step(self, action: str):
        """Step the env. Returns (obs, reward, done, info).

        Reuses SEED envs.py:40-55 semantics: info gets available_actions /
        task_score (raw continuous reward) / won; reward is binarized 10/0.
        """
        env = self._build()
        obs, reward, done, info = env.step(action)
        info = dict(info or {})
        info["available_actions"] = env.get_available_actions()
        info["task_score"] = reward
        if done and reward == 1.0:
            info["won"] = True
            reward = 10.0
        else:
            info["won"] = False
            reward = 0
        return obs, reward, done, info

    def get_available_actions(self) -> dict:
        return self._build().get_available_actions()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
