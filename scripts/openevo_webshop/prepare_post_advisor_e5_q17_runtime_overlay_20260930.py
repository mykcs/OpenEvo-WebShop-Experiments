#!/usr/bin/env python3
"""Prepare a fail-closed E5 Q17 validation overlay; never edits pinned source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA = "1a8a73b5e02854a822aa03dcb66a9fbf8d67c028f963f36845bf34fb1b60ea7c"
MANIFEST_SHA = "2d5d1620065a15840ae20fed324038312aec74b0895e13c9d283177928bf1a3b"
PANEL_SHA = "4033523c01697576b7f9ce8cddc3a63cf6c1514ef5554e01a8c82b59ee62d478"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"pinned {label} anchor changed: expected 1 occurrence, got {count}")
    return source.replace(old, new, 1)


def make_overlay(source_bytes: bytes) -> bytes:
    if sha(source_bytes) != SOURCE_SHA:
        raise ValueError("pinned public WebShop envs.py SHA mismatch")
    source = source_bytes.decode("utf-8")
    source = replace_once(
        source,
        "import ray\nimport gym\nimport numpy as np\n",
        "import ray\nimport gym\nimport numpy as np\nimport hashlib\nimport json\nimport os\n",
        "imports",
    )

    source = replace_once(
        source,
        """        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)
""",
        f"""        self.is_train = is_train
        self.q17_dev_tasks = None
        self.q17_dev_panel_sha256 = None
        if not is_train:
            schedule_path = os.environ.get("OPENEVO_E5_Q17_DEV_SCHEDULE")
            expected_sha = os.environ.get("OPENEVO_E5_Q17_SCHEDULE_SHA256")
            if not schedule_path or not expected_sha:
                raise RuntimeError("E5 validation requires a hash-bound Q17 schedule")
            with open(schedule_path, "rb") as schedule_file:
                payload = schedule_file.read()
            if hashlib.sha256(payload).hexdigest() != expected_sha:
                raise RuntimeError("E5 Q17 schedule SHA mismatch")
            schedule = json.loads(payload.decode("utf-8"))
            if schedule.get("schema") != "openevo.post-advisor-e5-q17-dev-schedule.v1":
                raise RuntimeError("unsupported E5 Q17 schedule schema")
            if schedule.get("source_manifest_sha256") != "{MANIFEST_SHA}":
                raise RuntimeError("E5 Q17 source manifest SHA mismatch")
            if schedule.get("panel_sha256") != "{PANEL_SHA}":
                raise RuntimeError("E5 Q17 panel SHA mismatch")
            tasks = schedule.get("tasks")
            if not isinstance(tasks, list) or len(tasks) != 64:
                raise RuntimeError("E5 validation must contain exactly 64 tasks")
            ordinals = [row.get("probe_ordinal") for row in tasks]
            indices = [row.get("goal_idx") for row in tasks]
            sessions = [row.get("source_session_index") for row in tasks]
            identities = [row.get("task_identity") for row in tasks]
            if ordinals != list(range(64)) or len(set(indices)) != 64 or len(set(identities)) != 64:
                raise RuntimeError("E5 Q17 tasks are not fixed and unique")
            if indices != sessions or any(row.get("worker_seed") != 0 for row in tasks):
                raise RuntimeError("E5 Q17 session or worker-seed semantics drifted")
            if any(not isinstance(i, int) or i < 0 or i >= 500 for i in indices):
                raise RuntimeError("E5 Q17 task outside frozen held-out pool")
            if env_num != 64 or group_n != 1:
                raise RuntimeError("E5 Q17 validation requires env_num=64, group_n=1")
            self.q17_dev_tasks = tasks
            self.q17_dev_panel_sha256 = schedule["panel_sha256"]

        self._rng = np.random.RandomState(seed)
""",
        "initialization",
    )

    source = replace_once(
        source,
        """        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)
""",
        """        for i in range(self.num_processes):
            worker_seed = (self.q17_dev_tasks[i]["worker_seed"]
                           if not self.is_train else seed + (i // self.group_n))
            worker = env_worker.remote(worker_seed, self._env_kwargs)
            self._workers.append(worker)
""",
        "worker construction",
    )

    source = replace_once(
        source,
        """        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)
""",
        """        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        if not self.is_train:
            for task in self.q17_dev_tasks:
                goal_idx = task["goal_idx"]
                if goal_idx >= len(goals):
                    raise RuntimeError("E5 Q17 goal index exceeds the WebShop goal list")
                instruction = str(goals[goal_idx].get("instruction_text", ""))
                instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
                if instruction_sha != task["instruction_sha256"]:
                    raise RuntimeError(f"E5 Q17 instruction identity mismatch at ordinal {task['probe_ordinal']}")
""",
        "goal identity validation",
    )

    source = replace_once(
        source,
        """        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
""",
        """        if not self.is_train:
            if self.q17_dev_tasks is None:
                raise RuntimeError("E5 cannot use stock random validation")
            self.goal_idxs = tuple(row["source_session_index"] for row in self.q17_dev_tasks)
        else:
            self.goal_idxs = range(500, len(goals))
""",
        "validation goal pool",
    )

    source = replace_once(
        source,
        """    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()
""",
        """    def reset(self):
        if not self.is_train:
            if self.q17_dev_tasks is None or len(self.goal_idxs) != 64:
                raise RuntimeError("E5 Q17 schedule unavailable")
            idx = list(self.goal_idxs)
        else:
            idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False).tolist()
        idx = np.repeat(idx, self.group_n).tolist()
""",
        "reset order",
    )

    source = replace_once(
        source,
        """        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
""",
        """        for position, (obs, info) in enumerate(results):
            if not self.is_train:
                task = self.q17_dev_tasks[position]
                info = dict(info or {})
                info["post_advisor_q17_panel_sha256"] = self.q17_dev_panel_sha256
                info["q17_probe_ordinal"] = task["probe_ordinal"]
                info["q17_task_identity"] = task["task_identity"]
                info["q17_instruction_sha256"] = task["instruction_sha256"]
                info["q17_source_session_index"] = task["source_session_index"]
                info["q17_goal_idx"] = task["goal_idx"]
                info["q17_worker_seed"] = task["worker_seed"]
                info["q17_paired_generation_seed"] = task["paired_generation_seed"]
            obs_list.append(obs)
            info_list.append(info)
""",
        "episode identity receipt",
    )
    return source.encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-envs", type=Path, required=True)
    parser.add_argument("--output-envs", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_envs.read_bytes()
    if args.source_envs.resolve() == args.output_envs.resolve():
        raise ValueError("refusing to overwrite pinned public source")
    output = make_overlay(source)
    args.output_envs.parent.mkdir(parents=True, exist_ok=True)
    args.output_envs.write_bytes(output)
    print(json.dumps({
        "status": "INERT_OVERLAY_PREPARED",
        "public_source_sha256": sha(source),
        "derived_envs_sha256": sha(output),
        "output": str(args.output_envs),
        "model_calls": 0,
        "gpu_calls": 0,
        "final_panel_access": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
