"""Linux end-to-end submission (Phase D).

STATUS: BLOCKED this round — CODE ONLY, NOT EXECUTED.

Running this requires a Linux host with Docker or Apptainer plus a reachable
OpenEvo rollout server and gateway node (see topology.template.yaml). Colab
has no Docker daemon, so the official OpenEvo container end-to-end cannot run
there. This module documents and implements the submit+poll flow so it is
ready the moment a Linux host is available.

Flow:
  1. build_task_payload()  → a TaskRequest-shaped dict (extra=forbid safe).
  2. POST {rollout_url}/rollout/task/submit
  3. poll GET {rollout_url}/rollout/task/{task_id} until terminal status.
  4. read back session trajectory reward.

rollout_url comes from TopologyConfig.load(topology_yaml).rollout.public_url.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def build_task_payload(
    task_id: str,
    image: str,
    model_name: str,
    num_samples: int = 1,
    timeout_seconds: float = 900.0,
    seed_root_target: str = "/openevo/session/seed",
) -> dict[str, Any]:
    """Construct a TaskRequest-compatible payload (extra=forbid safe).

    Fields match openevo.rollout.models.TaskRequest exactly. The runtime
    prepare uploads the SEED source under /openevo/session/... (sandbox root),
    then the in-container agent runs openevo_webshop.runner.
    """
    return {
        "task_id": task_id,
        "instruction": "Run the WebShop benchmark and report task_score.",
        "num_samples": num_samples,
        "timeout_seconds": timeout_seconds,
        "runtime": {
            "backend": "docker",
            "image": image,
            "prepare": [
                {
                    "type": "upload_dir",
                    "source": "seed/",
                    "target": seed_root_target,
                },
                {
                    "type": "exec",
                    "command": "pip install gym==0.24.0 transformers accelerate",
                },
            ],
        },
        "agent": {
            "import_path": "openevo_webshop.harness:WebshopHarness",
            "model_name": model_name,
        },
        "evaluator": {
            "strategy": "openevo_webshop.evaluator:WebshopTrajectoryEvaluator",
            "config": {"score_relpath": "webshop_score.json"},
            "env": {},
            "refresh_runtime": False,
        },
    }


def rollout_url_from_topology(topology_yaml: str | Path = "topology.yaml") -> str:
    """Resolve the rollout public_url from a TopologyConfig file."""
    from openevo.config.topology import TopologyConfig

    return TopologyConfig.load(topology_yaml).rollout.public_url.rstrip("/")


def submit_task(rollout_url: str, payload: dict[str, Any]) -> str:
    """POST the task and return its task_id."""
    resp = httpx.post(f"{rollout_url}/rollout/task/submit", json=payload, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("task_id") or payload["task_id"]


def poll_task(
    rollout_url: str,
    task_id: str,
    interval_s: float = 5.0,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Poll GET /rollout/task/{task_id} until a terminal status."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = httpx.get(f"{rollout_url}/rollout/task/{task_id}", timeout=30.0)
        resp.raise_for_status()
        status = resp.json()
        if status.get("status") in TERMINAL_STATUSES:
            return status
        time.sleep(interval_s)
    raise TimeoutError(f"task {task_id} did not reach a terminal status in {timeout_s}s")


def extract_reward(status: dict[str, Any]) -> float | None:
    """Read the trajectory outcome reward back from a terminal TaskStatus."""
    for session in status.get("results", []):
        traj = session.get("trajectory") or {}
        traces = traj.get("traces") or []
        if traces:
            reward = traces[-1].get("reward")
            if reward is not None:
                return float(reward)
    return None
