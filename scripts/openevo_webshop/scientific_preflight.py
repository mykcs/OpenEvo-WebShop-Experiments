#!/usr/bin/env python3
"""Bounded real model -> frozen parser -> WebShop engineering preflight.

The gate uses exactly one deterministic train-candidate WebShop identity outside
the formal final pool.  It proves model loading, prompt rendering, parser/recovery
semantics, and one real environment step.  Task success is deliberately not a
PASS criterion: this gate distinguishes an engineering-broken chain from a weak
model that simply fails the benchmark task.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor/openevo-webshop/src"
for value in (ROOT, VENDOR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.openevo_webshop import experiment_resolver, golden_runtime_registry
from scripts.openevo_webshop.gpu_resource_gate import idle, snapshot_gpu
from scripts.openevo_webshop.parametric_rnd import FixedEnvironmentSeed, ParserSafeRetryBackend
from scripts.openevo_webshop.webshop_comparison_contract import (
    ACTION_COMMAND_CONTRACT,
    COMPACT_PROMPT_CONTRACT,
    extract_goal,
    format_webshop_actions,
    parse_action_command_contract,
)


SCHEMA = "openevo.model-parser-webshop-preflight.v1"
POLICY_PATH = ROOT / "configs/execution/qualification-policy-v1.json"


class PreflightError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"JSON object required: {path}")
    return value


def _git_identity(root: Path) -> str:
    root = root.resolve()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise PreflightError(f"Git checkout is dirty: {root}")
    return head


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
    path.chmod(0o600)


def load_policy() -> dict[str, Any]:
    policy = _read(POLICY_PATH)
    if policy.get("schema") != "openevo.execution-qualification-policy.v1":
        raise PreflightError("unexpected execution qualification policy schema")
    gate = policy.get("model_parser_webshop_gate")
    if not isinstance(gate, dict):
        raise PreflightError("model/parser/WebShop gate policy is missing")
    if int(gate.get("goal_idx", -1)) < 500 or gate.get("formal_panel_access_allowed") is not False:
        raise PreflightError("scientific preflight is not guarded outside the final pool")
    if gate.get("task_success_required") is not False:
        raise PreflightError("engineering preflight must not require benchmark task success")
    if gate.get("prompt_contract") != COMPACT_PROMPT_CONTRACT or gate.get("action_contract") != ACTION_COMMAND_CONTRACT:
        raise PreflightError("scientific preflight policy drifted from the frozen prompt/parser contracts")
    return policy


def _configure_registry_jdk(runtime_evidence: dict[str, Any]) -> Path:
    """Expose the Golden Runtime's verified JDK to pyserini/jnius."""

    try:
        jdk_home = Path(runtime_evidence["assets"]["jdk"]["path"]).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightError("Golden Runtime JDK path is missing") from exc
    javac = jdk_home / "bin" / "javac"
    if not jdk_home.is_dir() or not javac.is_file():
        raise PreflightError(f"Golden Runtime JDK is unavailable: {jdk_home}")
    os.environ["JAVA_HOME"] = str(jdk_home)
    java_bin = str(jdk_home / "bin")
    path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    if java_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([java_bin, *path_entries])
    return jdk_home


def _verify_declared_file(source_root: Path, item: dict[str, Any], label: str) -> Path:
    relative = item.get("path")
    expected = item.get("sha256")
    if not isinstance(relative, str) or not relative or not isinstance(expected, str) or len(expected) != 64:
        raise PreflightError(f"{label} path/SHA-256 declaration is incomplete")
    path = (source_root / relative).resolve()
    source_root = source_root.resolve()
    if path != source_root and source_root not in path.parents:
        raise PreflightError(f"{label} path escapes experiment source root")
    if not path.is_file():
        raise PreflightError(f"{label} does not exist: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise PreflightError(f"{label} SHA-256 mismatch: expected={expected} observed={observed}")
    return path


def _sample_live_idle(uuid: str, policy: dict[str, Any]) -> dict[str, Any]:
    gpu = policy["gpu_gate"]
    sample = snapshot_gpu(uuid)
    if not idle(
        sample,
        max_memory_mib=int(gpu["max_idle_memory_mib"]),
        max_util_percent=int(gpu["max_idle_util_percent"]),
    ):
        raise PreflightError(f"GPU is not live-idle: {uuid}: {sample}")
    return sample


def _admissible(action: str, available: dict[str, Any]) -> bool:
    lowered = action.strip().lower()
    if lowered.startswith("search[") and lowered.endswith("]"):
        return bool(available.get("has_search_bar")) and bool(lowered[len("search[") : -1].strip())
    actions = {value.lower() for value in format_webshop_actions(available)}
    return lowered in actions


class _TrackedEnv:
    def __init__(self, env: Any) -> None:
        self._env = env
        self.reset_count = 0
        self.step_count = 0

    def reset(self, idx: int, seed: int | None = None):
        self.reset_count += 1
        return self._env.reset(idx, seed=seed)

    def step(self, action: str):
        self.step_count += 1
        return self._env.step(action)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def validate_episode(
    episode: dict[str, Any],
    *,
    expected_goal_sha256: str,
    initial_available_actions: dict[str, Any],
    tracked_step_count: int,
    parser_events: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    gate = policy["model_parser_webshop_gate"]
    errors: list[str] = []
    trace = episode.get("trace")
    if not isinstance(trace, list) or len(trace) != 1:
        errors.append(f"expected exactly one bounded trace step, got {len(trace) if isinstance(trace, list) else 'invalid'}")
        step: dict[str, Any] = {}
    else:
        step = trace[0] if isinstance(trace[0], dict) else {}
    goal = episode.get("goal")
    # EpisodeResult truncates goal to 80 chars, so identity is proven by the
    # expected-task hash check performed inside run_episode, not by this field.
    if tracked_step_count != 1:
        errors.append(f"expected exactly one real env.step call, got {tracked_step_count}")
    if step.get("action_valid") is not True:
        errors.append("final frozen parser did not accept the model action")
    if step.get("used_fallback") is not False:
        errors.append("fallback action was used")
    action = step.get("action")
    if not isinstance(action, str) or not _admissible(action, initial_available_actions):
        errors.append(f"final action was not admissible in the initial WebShop state: {action!r}")
    prompt = step.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        errors.append("model did not receive a non-empty WebShop prompt")
    if step.get("generation_seed") != gate["generation_seed"]:
        errors.append("generation seed drifted from qualification policy")
    if step.get("environment_seed") != gate["worker_seed"]:
        errors.append("environment seed drifted from qualification policy")
    if any(event.get("event_type") == "parser-retry" for event in parser_events) and not gate.get("parser_retry_allowed"):
        errors.append("parser retry occurred but policy forbids it")
    if any(event.get("event_type") == "action-command-recovery" for event in parser_events) and not gate.get("bounded_wrapper_recovery_allowed"):
        errors.append("bounded wrapper recovery occurred but policy forbids it")
    if errors:
        raise PreflightError("; ".join(errors))
    return {
        "goal_sha256": expected_goal_sha256,
        "episode_goal_preview": goal,
        "action": action,
        "action_valid": True,
        "used_fallback": False,
        "env_step_count": tracked_step_count,
        "parser_event_count": len(parser_events),
        "parser_retry_count": sum(event.get("event_type") == "parser-retry" for event in parser_events),
        "wrapper_recovery_count": sum(event.get("event_type") == "action-command-recovery" for event in parser_events),
        "task_score": episode.get("task_score"),
        "won": episode.get("won"),
        "task_success_required": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy()
    gate = policy["model_parser_webshop_gate"]
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise PreflightError(f"preflight output must be empty: {output}")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)

    experiment_path = args.experiment.resolve()
    source_root = args.experiment_source_root.resolve()
    if experiment_path != source_root and source_root not in experiment_path.parents:
        raise PreflightError("experiment config is outside --experiment-source-root")
    experiment_source_revision = _git_identity(source_root)
    platform_source_revision = _git_identity(ROOT)
    resolution = experiment_resolver.resolve(experiment_path, verify_local_assets=True)
    model = golden_runtime_registry.load_model(resolution["model"]["asset_id"])
    model_path = Path(model["canonical_path"]).resolve()
    effective_revision = str(model.get("revision") or model["content_pin"])
    experiment = _read(experiment_path)
    execution = experiment.get("execution_contract")
    interface = experiment.get("interface_contract")
    if not isinstance(execution, dict) or not isinstance(interface, dict):
        raise PreflightError("experiment execution/interface contracts are missing")
    # This gate drives the bounded run_episode path directly.  It does not
    # consume the formal schedule or execute the experiment's task-bearing
    # rollout runner, so a stale formal-runner hash must not block this
    # qualification-only chain.  Formal execution validates that hash at its
    # own launch boundary.
    if interface.get("prompt_contract") != gate["prompt_contract"]:
        raise PreflightError("experiment prompt contract does not match qualification policy")
    if interface.get("action_parser_contract") != gate["action_contract"]:
        raise PreflightError("experiment action parser contract does not match qualification policy")
    if float(interface.get("temperature")) != float(gate["temperature"]):
        raise PreflightError("experiment temperature does not match qualification policy")
    if int(interface.get("max_new_tokens")) != int(gate["max_new_tokens"]):
        raise PreflightError("experiment max_new_tokens does not match qualification policy")

    runtime = golden_runtime_registry.load_runtime()
    runtime_evidence = golden_runtime_registry._read(ROOT / runtime["runtime_evidence"]["path"])
    jdk_home = _configure_registry_jdk(runtime_evidence)
    seed_root = Path(args.seed_root or runtime_evidence["assets"]["seed_root"]["path"]).resolve()
    if not seed_root.is_dir():
        raise PreflightError(f"SEED root is unavailable: {seed_root}")

    from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig

    lock_root = Path(os.environ.get("OPENEVO_WEBSHOP_GPU_LOCK_ROOT", policy["gpu_gate"]["lock_root"]))
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_root / f"{args.gpu_uuid}.lock"
    prelaunch: list[dict[str, Any]] = []
    started = time.monotonic()
    backend = None
    raw_env = None
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PreflightError(f"GPU lock busy: {lock_path}") from exc
        for index in range(int(policy["gpu_gate"]["prelaunch_samples"])):
            prelaunch.append(_sample_live_idle(args.gpu_uuid, policy))
            if index + 1 < int(policy["gpu_gate"]["prelaunch_samples"]):
                time.sleep(0.25)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Preview-reset only the fixed train candidate to bind its task identity
        # and initial admissible actions.  This must happen after the live gate:
        # WebShop's reset path imports torch and may open CUDA device handles.
        preview_env = WebshopEnv(WebshopEnvConfig(seed_root=str(seed_root), num_products=1000, seed=int(gate["worker_seed"])))
        try:
            preview_obs, _ = preview_env.reset(int(gate["goal_idx"]), seed=int(gate["worker_seed"]))
            expected_goal = extract_goal(preview_obs)
            expected_goal_sha256 = hashlib.sha256(expected_goal.encode("utf-8")).hexdigest()
            initial_available = preview_env.get_available_actions()
        finally:
            preview_env.close()
        if int(gate["goal_idx"]) < 500:
            raise PreflightError("qualification goal entered the final pool")

        import torch
        from openevo_webshop.inference import make_backend
        import openevo_webshop.runner as runner_module
        from openevo_webshop.runner import run_episode

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise PreflightError(f"qualification must see exactly one CUDA device, got {torch.cuda.device_count()}")
        runner_module.parse_webshop_action = parse_action_command_contract
        backend = make_backend(
            "transformers",
            str(model_path),
            device="cuda",
            revision=effective_revision,
            local_files_only=True,
        )
        load_receipt = getattr(backend, "load_receipt", None)
        if not isinstance(load_receipt, dict) or load_receipt.get("verified") is not True or load_receipt.get("adapter_loaded") is not False:
            raise PreflightError(f"base model load receipt is not verified/no-adapter: {load_receipt}")
        safe_backend = ParserSafeRetryBackend(backend, action_contract=gate["action_contract"])
        raw_env = WebshopEnv(WebshopEnvConfig(seed_root=str(seed_root), num_products=1000, seed=int(gate["worker_seed"])))
        tracked = _TrackedEnv(FixedEnvironmentSeed(raw_env, int(gate["worker_seed"])))
        result = run_episode(
            tracked,
            safe_backend,
            int(gate["goal_idx"]),
            max_steps=int(gate["max_steps"]),
            max_new=int(gate["max_new_tokens"]),
            temperature=float(gate["temperature"]),
            generation_seed=int(gate["generation_seed"]),
            environment_seed=int(gate["worker_seed"]),
            prompt_contract=gate["prompt_contract"],
            expected_task_description_sha256=expected_goal_sha256,
        )
        episode = result.__dict__
        summary = validate_episode(
            episode,
            expected_goal_sha256=expected_goal_sha256,
            initial_available_actions=initial_available,
            tracked_step_count=tracked.step_count,
            parser_events=safe_backend.events,
            policy=policy,
        )
        if raw_env is not None:
            raw_env.close()
            raw_env = None
        if backend is not None:
            backend.close()
            backend = None
        torch.cuda.empty_cache()

    wall = time.monotonic() - started
    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "experiment": resolution["experiment"],
        "resolution_sha256": resolution["resolution_sha256"],
        "runtime_family": runtime["runtime_family"],
        "runtime_environment": {"java_home": str(jdk_home)},
        "source": {
            **resolution["source"],
            "experiment_source_revision": experiment_source_revision,
            "platform_source_revision": platform_source_revision,
        },
        "model": {
            "asset_id": model["asset_id"],
            "hf_id": model["hf_id"],
            "canonical_path": str(model_path),
            "revision": model.get("revision"),
            "content_pin": model["content_pin"],
            "load_receipt": load_receipt,
        },
        "task": {
            "worker_seed": gate["worker_seed"],
            "goal_idx": gate["goal_idx"],
            "goal_sha256": expected_goal_sha256,
            "final_pool_guard": gate["final_pool_guard"],
            "preview_reset_only": True,
        },
        "chain": summary,
        "gpu": {
            "uuid": args.gpu_uuid,
            "prelaunch_samples": prelaunch,
            "wall_seconds": wall,
            "gpu_hours": wall / 3600.0,
        },
        "authority_boundary": {
            "qualification_only": True,
            "execution_authorized": False,
            "formal_task_consumption_allowed": False,
            "formal_schedule_positions_consumed": 0,
            "qualification_train_candidate_steps": 1,
            "formal_panel_accessed": False,
            "task_success_required": False,
        },
    }
    _atomic_json(output / "preflight-receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--experiment-source-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (PreflightError, experiment_resolver.ExperimentResolutionError, golden_runtime_registry.RegistryError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "FAIL", "errors": [str(exc)]}, indent=2, sort_keys=True))
        return 1
    finally:
        # Best-effort cleanup for failures after model/env construction.
        if "raw_env" in locals() and raw_env is not None:
            try:
                raw_env.close()
            except Exception:
                pass
        if "backend" in locals() and backend is not None:
            try:
                backend.close()
            except Exception:
                pass
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
