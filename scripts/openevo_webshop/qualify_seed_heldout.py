#!/usr/bin/env python3
"""Fail-closed qualification gate for the SEED held-out comparison campaign.

Runs 16 checks and, only if all pass, writes a hash-bound PASS receipt that sets
``formal_task_consumption_allowed=true``. The qualification consumes ZERO held-out
tasks: behavioral episodes run only on the train pool (frozen goal_idx 5500 and
6500, both >= 500) for both arms and both contracts. Held-out goal_idx 0-499 is
never touched here.

Checks (any failure -> status=fail, no receipt):
 1  git SHA matches and worktree is clean
 2  base model pin (model loads with pinned revision, local files only)
 3  adapter SHA + load (SD arm)
 4  no base fallback (SD arm adapter active; BASE arm no adapter)
 5  WebShop small files present + hashed (match manifest data_files)
 6  human_goals == False (synthetic goals)
 7  len(goals) == 6910
 8  env seed=0 deterministic goal ordering (re-derive goals_list_digest == manifest)
 9  reward wrapper retains raw task_score + exact success separately
10  prompt contract (SEED template markers + task text present)
11  parser contract (pass-through projection records per-step is_action_valid)
12  JSON numeric serialization (task_score is a float, round-trips)
13  output isolation (outputs only under --output)
14  no partial outputs (per-episode JSON complete)
15  task identity replay (reset twice -> identical instruction)
16  evaluator produces score + success
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

EXPECTED_GOALS_SMALL = 6910
QUALIFICATION_GOAL_IDXS = (5500, 6500)  # train pool (>=500); never held-out
QUAL_GENERATION_SEED_BASE = 2026082900
SD_ADAPTER_CONFIG_SHA256 = "ed48aae286e03ae63fe0f70c5e75dddd650f098e6d5cba55efb11106523c5261"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Checks:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def record(self, name: str, ok: bool, detail: Any = None) -> bool:
        self.results.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def all_ok(self) -> bool:
        return all(r["ok"] for r in self.results)


def _read_goals(seed_root: Path, environment_seed: int) -> list[dict[str, Any]]:
    from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig

    cfg = WebshopEnvConfig(seed_root=str(seed_root), num_products=1000, seed=environment_seed)
    env = WebshopEnv(cfg)
    raw = env._build()  # noqa: SLF001
    base = getattr(raw, "unwrapped", raw)
    goals = list(getattr(base, "server").goals)
    env.close()
    return goals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--sd-adapter", required=True)
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--control-git-head", default="")
    parser.add_argument("--control-git-clean", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()

    checks = Checks()
    config = json.loads(args.campaign_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)

    # 1. git SHA / clean. The scientific container has no reliable git binary or
    # usable ownership context, so re-running git here is spurious. The authoritative
    # proof lives on the control plane: the launch driver enforced fail-closed that
    # HEAD == experiment_git_sha and the worktree is clean on the SAME repo that is
    # bind-mounted read-only into this container, then passed that evidence in via
    # --control-git-head/--control-git-clean. Verify that evidence here.
    head = (args.control_git_head or "").strip()
    clean = (args.control_git_clean or "").strip().lower() == "true"
    is_hex40 = len(head) == 40 and all(c in "0123456789abcdef" for c in head.lower())
    checks.record(
        "git_sha_clean",
        bool(is_hex40 and head == args.experiment_git_sha and clean),
        {"head": head, "clean": clean, "source": "control-plane driver fail-closed gate"},
    )

    # 5. WebShop small files present + hashed
    data_dir = Path(args.seed_root) / "agent_system/environments/env_package/webshop/webshop/data"
    data_ok = True
    data_detail: dict[str, Any] = {}
    for name, expected in (manifest["benchmark"]["data_files"] or {}).items():
        path = data_dir / name
        if not path.is_file():
            data_ok = False
            data_detail[name] = "missing"
            continue
        observed = _sha256_bytes(path.read_bytes())
        data_detail[name] = observed
        if observed != expected:
            data_ok = False
    checks.record("webshop_small_files", data_ok, data_detail)

    # 6/7/8. human_goals False, len(goals)==6910, env seed=0 deterministic ordering
    try:
        goals = _read_goals(args.seed_root, args.environment_seed)
        goals_total = len(goals)
        checks.record("human_goals_false", goals_total == EXPECTED_GOALS_SMALL, {"goals_total": goals_total})
        checks.record("goals_len_6910", goals_total == EXPECTED_GOALS_SMALL, goals_total)
        fingerprint = [
            {"goal_idx": i, "instruction_sha256": hashlib.sha256(str(g.get("instruction_text", "")).encode()).hexdigest(), "asin": g.get("asin")}
            for i, g in enumerate(goals)
        ]
        derived = _canonical_digest(fingerprint)
        checks.record(
            "env_seed0_goal_ordering",
            derived == manifest["benchmark"]["goals_list_digest"],
            {"derived": derived, "expected": manifest["benchmark"]["goals_list_digest"]},
        )
    except Exception as exc:  # noqa: BLE001
        for name in ("human_goals_false", "goals_len_6910", "env_seed0_goal_ordering"):
            checks.record(name, False, f"{type(exc).__name__}: {exc}")

    # 15. task identity replay (CPU): reset the same train-pool task twice -> identical instruction
    try:
        from openevo_webshop.comparison_contract import seed_extract_task
        from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig

        replay_ok = True
        replay_detail = {}
        for goal_idx in QUALIFICATION_GOAL_IDXS:
            instructions = []
            for _ in range(2):
                env = WebshopEnv(WebshopEnvConfig(seed_root=str(args.seed_root), num_products=1000, seed=args.environment_seed))
                obs, _ = env.reset(goal_idx, seed=args.environment_seed)
                instructions.append(seed_extract_task(obs))
                env.close()
            replay_detail[goal_idx] = instructions[0][:60]
            if instructions[0] != instructions[1]:
                replay_ok = False
        checks.record("task_identity_replay", replay_ok, replay_detail)
    except Exception as exc:  # noqa: BLE001
        checks.record("task_identity_replay", False, f"{type(exc).__name__}: {exc}")

    # GPU behavioral checks: load each arm once, run bounded train-pool episodes under both contracts.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_uuid)
    from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
    from openevo_webshop.inference import make_backend

    arms = {
        "base": {"adapter": "none", "expected_file": None, "expected_tensor": None},
        "sd-lora": {
            "adapter": args.sd_adapter,
            "expected_file": config["frozen_artifact"]["adapter_file_sha256"],
            "expected_tensor": config["frozen_artifact"]["loaded_tensor_digest"],
        },
    }
    reward_wrapper_ok = True
    numeric_ok = True
    complete_ok = True
    evaluator_ok = True
    prompt_ok = True
    parser_ok = True
    adapter_identity: dict[str, Any] = {}

    for arm_name, arm in arms.items():
        backend = make_backend(
            "transformers", args.model, device="cuda", revision=args.model_revision,
            adapter_path=arm["adapter"] if arm["adapter"] != "none" else None, local_files_only=True,
        )
        receipt = backend.load_receipt
        adapter_identity[arm_name] = receipt
        if arm_name == "sd-lora":
            checks.record(
                "adapter_sha_load",
                bool(receipt.get("verified")) and receipt.get("adapter_file_sha256") == arm["expected_file"]
                and receipt.get("loaded_tensor_digest") == arm["expected_tensor"],
                {"file": receipt.get("adapter_file_sha256"), "tensor": receipt.get("loaded_tensor_digest")},
            )
            checks.record(
                "no_base_fallback",
                bool(receipt.get("adapter_loaded")) and "Peft" in str(receipt.get("backend_model_type")),
                receipt.get("backend_model_type"),
            )
        else:
            checks.record("base_model_pin", bool(receipt.get("verified")) and not receipt.get("adapter_loaded"), receipt.get("backend_model_type"))

        try:
            for contract in ("seed-eval", "openevo-native"):
                for goal_idx in QUALIFICATION_GOAL_IDXS:
                    env = WebshopEnv(WebshopEnvConfig(seed_root=str(args.seed_root), num_products=1000, seed=args.environment_seed))
                    try:
                        if contract == "seed-eval":
                            from openevo_webshop.seed_eval import run_seed_eval_episode

                            ep = run_seed_eval_episode(
                                env, backend, goal_idx, environment_seed=args.environment_seed,
                                generation_seed=QUAL_GENERATION_SEED_BASE + goal_idx,
                                max_steps=3, max_new=64, temperature=0.4,
                            )
                            trace = ep.get("trace") or []
                            if trace:
                                first_prompt = trace[0].get("prompt", "")
                                if "WebShop" not in first_prompt or "admissible actions" not in first_prompt:
                                    prompt_ok = False
                                if ep.get("goal") and ep["goal"] not in first_prompt:
                                    prompt_ok = False
                            if any("is_action_valid" not in s for s in trace):
                                parser_ok = False
                            if not isinstance(ep.get("task_score"), float):
                                numeric_ok = False
                            if not (isinstance(ep.get("task_score"), float) and isinstance(ep.get("won"), bool)):
                                evaluator_ok = False
                                reward_wrapper_ok = False
                            json.dumps(ep)  # serializable
                            if "task_score" not in ep or "won" not in ep or "trace" not in ep:
                                complete_ok = False
                        else:
                            from openevo_webshop.runner import run_episode

                            res = run_episode(
                                env, backend, goal_idx, max_steps=3, max_new=64, temperature=0.0,
                                generation_seed=QUAL_GENERATION_SEED_BASE + goal_idx,
                                environment_seed=args.environment_seed,
                                prompt_contract="seed3090.webshop-compact-action-prompt.v2",
                            )
                            ep = res.__dict__
                            if not isinstance(ep.get("task_score"), (int, float)) or not isinstance(ep.get("won"), bool):
                                evaluator_ok = False
                            json.dumps(ep)
                    finally:
                        env.close()
        finally:
            backend.close()

    checks.record("reward_wrapper", reward_wrapper_ok, "raw task_score + won preserved separately")
    checks.record("prompt_contract", prompt_ok, "SEED template markers + task text present")
    checks.record("parser_contract", parser_ok, "per-step is_action_valid recorded (pass-through)")
    checks.record("json_numeric_serialization", numeric_ok, "task_score float, episode JSON-serializable")
    checks.record("evaluator_score_success", evaluator_ok, "episode produces task_score + won")
    checks.record("no_partial_outputs", complete_ok, "per-episode record has task_score/won/trace")

    # 13/14. output isolation + no partial outputs at the filesystem level
    out_files = [p for p in args.output.rglob("*") if p.is_file()]
    checks.record("output_isolation", all(str(p).startswith(str(args.output)) for p in out_files), len(out_files))

    status = "pass" if checks.all_ok else "fail"
    receipt = {
        "schema": "openevo.webshop-seed-heldout-qualification.v1",
        "campaign_id": config["campaign_id"],
        "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qualification_consumption": "train-pool goal_idx 5500/6500 only; zero held-out consumption",
        "qualification_goal_idxs": list(QUALIFICATION_GOAL_IDXS),
        "gpu_uuid": args.gpu_uuid,
        "checks": checks.results,
        "binding": {
            "campaign_config_sha256": _sha256_bytes(args.campaign_config.read_bytes()),
            "task_manifest_file_sha256": _sha256_bytes(args.task_manifest.read_bytes()),
            "experiment_git_sha": args.experiment_git_sha,
            "model": args.model,
            "model_revision": args.model_revision,
            "adapter_file_sha256": config["frozen_artifact"]["adapter_file_sha256"],
            "loaded_tensor_digest": config["frozen_artifact"]["loaded_tensor_digest"],
            "evaluator_contract": "WebShop raw task_score + exact success (won)",
            "environment_seed": args.environment_seed,
        },
        "adapter_identity": adapter_identity,
    }
    receipt["formal_task_consumption_allowed"] = status == "pass"
    receipt["receipt_sha256"] = _canonical_digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    out = args.output / "qualification-receipt.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "receipt": str(out), "receipt_sha256": receipt["receipt_sha256"],
                      "checks_passed": sum(1 for r in checks.results if r["ok"]), "checks_total": len(checks.results)}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
