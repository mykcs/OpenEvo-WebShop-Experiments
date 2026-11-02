#!/usr/bin/env python3
"""Audit Q03 local source/start/evaluator bridge without models, network, or GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = frozenset({
    "examples/seed_trainer/_common/webshop.sh",
    "examples/seed_trainer/run_webshop_sft_glm_self.sh",
    "examples/data_preprocess/prepare.py",
    "verl/trainer/ppo/ray_trainer.py",
    "agent_system/environments/env_package/webshop/envs.py",
})
STAGE1_SUPPORT_FILES = frozenset({
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_inventory(package: dict) -> dict:
    files = package["source"].get("files")
    require(isinstance(files, dict) and set(files) == SOURCE_FILES, "source inventory requires all five exact paths")
    for relative, digest in files.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"source SHA format: {relative}")
    return files


def verify_artifact(item: dict, expected_sha: str | None = None) -> str:
    digest = item["sha256"]
    require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "artifact SHA format")
    require(expected_sha is None or digest == expected_sha, "artifact differs from frozen identity")
    require(file_sha(Path(item["path"])) == digest, "artifact bytes SHA mismatch")
    return digest


def audit_plan(package: dict) -> dict:
    require(package.get("schema") == "openevo.advisor-work-package.v1", "work-package schema")
    require(package.get("question_id") == "Q03" and package.get("owner_pr") == 630, "Q03 owner")
    require(package.get("status") == "PLANNED_NOT_ACTIVATED", "package must remain inert")
    require(package.get("execution_authority") == {
        "formal_task_consumption_allowed": False, "gpu_allocation_allowed": False,
        "final_panel_access_allowed": False, "final_panel_access": 0,
    }, "execution authority must remain closed")
    stage = package["stage2"]
    budget_policy = package["budget_policy"]
    require(budget_policy["qualification_reset_attempts_upper_bound"] is None
            and budget_policy["qualification_generation_calls"] is None
            and budget_policy["qualification_action_steps_upper_bound"] is None,
            "engineering revalidation uses the aggregate budget, not per-operation stop locks")
    require(budget_policy["qualification_gpu_hours_upper_bound"] is None
            and isinstance(budget_policy["qualification_gpu_hours_planning_estimate"], (int, float))
            and not isinstance(budget_policy["qualification_gpu_hours_planning_estimate"], bool)
            and math.isfinite(budget_policy["qualification_gpu_hours_planning_estimate"])
            and budget_policy["qualification_gpu_hours_planning_estimate"] >= 0
            and budget_policy["qualification_budget_authority"]["classification"] == "AGENT_PLANNING_ESTIMATE_NOT_BINDING_LIMIT"
            and budget_policy["qualification_budget_authority"]["binding_limit_source"] is None
            and budget_policy["qualification_goal_index"] == 500
            and budget_policy["qualification_generation_tokens_per_call_upper_bound"] == 512,
            "fixed train-pool task, native token ceiling and verified qualification budget provenance")
    rule = budget_policy["qualification_no_science_retry_rule"]
    require("may continue after model generation or action" in rule
            and "low rewards alone never authorize a retry" in rule,
            "engineering repair after generation must preserve scientific no-outcome-retry rule")
    require("including historical failures" in budget_policy["qualification_budget_accounting"],
            "engineering repair must preserve cumulative budget accounting")
    require(stage["updates_per_seed"] == 160 and stage["tasks_per_update"] == 16 and stage["group_size"] == 8, "160 x 16 x 8 contract")
    require(stage["slots_per_update"] == 128 and stage["train_batch_size"] == 16, "slot/batch contract")
    require(stage.get("environment_rng_checkpoint_required") is True,
            "Q03 exact update-boundary resume requires environment task-selection RNG")
    streams = stage["streams"]
    require(len(streams) == 2 and len({s["id"] for s in streams}) == 2, "two unique streams")
    for key in ("environment_seed", "policy_seed", "data_seed"):
        require(len({s[key] for s in streams}) == 2, f"independent {key} must be frozen")
    require(stage["native_validation"] == {"trainer.test_freq": -1, "trainer.val_before_train": False, "evaluation_episodes": 0}, "native validation must remain zero")
    require(package["science"]["primary_checkpoint"] == {"policy_update": 160}, "terminal primary checkpoint")
    require(package["science"]["practical_effect_threshold"] == .02, "practical threshold")
    require(package["evaluation"]["milestones"] == [0, 40, 80, 120, 160] and package["evaluation"]["panel_size"] == 64, "fixed milestone panel")
    primary = next(s for s in package["stages"] if s["id"] == "B_TWO_SEED_STAGE2")
    require(primary["budget"]["training_slots"] == 2 * 160 * 128, "two-seed total slots")
    require(primary["budget"]["training_rollouts"] == 2 * 160 * 128, "two-seed rollout count")
    require(primary["budget"]["policy_updates"] == 2 * 160, "two-seed policy-update count")
    require(primary["budget"]["actor_optimizer_steps_upper_bound"] == 2 * 160 * (128 * 15 // 64),
            "conditional two-seed actor optimizer-step upper bound")
    require(primary["budget"]["actor_optimizer_step_bound_basis"].startswith("CONDITIONAL_PROFILE_NOT_EXECUTION_FROZEN:"),
            "actor optimizer-step ceiling must disclose unfrozen execution assumptions")
    require(primary["budget"]["learner_reference_teacher_forward_bound_basis"].startswith("CONDITIONAL_PROFILE_NOT_EXECUTION_FROZEN:"),
            "conditional forward ceilings require effective-profile disclosure")
    for field, expected in {
        "actor_old_logprob_forward_calls_upper_bound": 320 * 8 * 15,
        "reference_logprob_forward_calls_upper_bound": 320 * 8 * 15,
        "policy_teacher_logprob_forward_calls_upper_bound": 320 * 8 * 16,
        "actor_gradient_forward_calls_upper_bound": 320 * 8 * 30,
        "actor_backward_calls_upper_bound": 320 * 8 * 30,
        "actor_old_logprob_token_positions_upper_bound": 320 * 1920 * 4608,
        "reference_logprob_token_positions_upper_bound": 320 * 1920 * 4608,
        "policy_teacher_logprob_token_positions_upper_bound": 320 * 1928 * 4608,
        "actor_gradient_token_positions_upper_bound": 320 * 1920 * 4608,
    }.items():
        require(primary["budget"][field] == expected, f"conditional native compute ceiling: {field}")
    require(primary["budget"]["environment_steps_upper_bound"] == 2 * 160 * 128 * 15, "environment-step upper bound")
    require(primary["budget"]["policy_response_tokens_upper_bound"] == 2 * 160 * 128 * 15 * 512,
            "policy response-token upper bound")
    require(primary["budget"]["policy_vllm_analysis_requests_upper_bound"] == 2 * 160 * 128,
            "policy_vllm request upper bound")
    require(primary["budget"]["policy_vllm_prompt_tokens_upper_bound"] == 2 * 160 * 128 * 16384,
            "policy_vllm prompt-token upper bound")
    require(primary["budget"]["policy_vllm_completion_tokens_upper_bound"] == 2 * 160 * 128 * 4096,
            "policy_vllm completion-token upper bound")
    require(primary["budget"]["evaluation_episodes"] == 2 * 5 * 64, "canonical episodes")
    require(primary["budget"]["evaluation_generated_tokens_upper_bound"] == 2 * 5 * 64 * 15 * (64 + 512),
            "canonical evaluation-token upper bound")
    require(budget_policy["evaluation_generated_token_upper_bound_per_64_episode_milestone"] == 64 * 15 * (64 + 512),
            "two-phase canonical milestone-token upper bound")
    context = next(s for s in package["stages"] if s["id"] == "C_OPENEVO_CONTEXT")
    for evaluation_stage in (primary, context):
        count = evaluation_stage["budget"]["evaluation_episodes"]
        require(evaluation_stage["budget"]["evaluation_environment_steps_upper_bound"] == count * 15,
                "evaluation environment-step ceiling")
        require(evaluation_stage["budget"]["evaluation_generated_tokens_upper_bound"] == count * 15 * (64 + 512),
                "two-phase evaluation-token ceiling")
        require(evaluation_stage["budget"]["evaluation_generation_calls_upper_bound"] == count * 15 * 2,
                "two-phase evaluation-call ceiling")
    require(package["evaluation"]["old_start_result_reusable_as_canonical"] is False, "historical evaluator cannot supply canonical start")
    require(package["source"]["repository"] == "https://github.com/jinyangwu/SEED", "prospective public source owner")
    require(package["source"]["commit"] == "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c", "pinned public source commit")
    source_inventory(package)
    supporting = package["start"].get("checkpoint_supporting_files")
    require(isinstance(supporting, dict) and set(supporting) == STAGE1_SUPPORT_FILES,
            "complete Stage1 config/tokenizer support-file inventory required")
    for name, digest in supporting.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"Stage1 supporting-file SHA format: {name}")
    return {"training_slots": 40960, "canonical_episodes": 640,
            "native_auxiliary_episodes": 0, "internal_teacher_cost_included": False,
            "initialization_repeats": 1, "stage2_streams": 2,
            "actor_optimizer_bound_scope": "conditional-profile-consistency-only; effective config not qualified"}


def audit_source(root: Path, package: dict) -> dict:
    verified = {}
    for relative, expected in source_inventory(package).items():
        actual = file_sha(root / relative)
        require(actual == expected, f"pinned source bytes mismatch: {relative}")
        verified[relative] = actual
    return {"scope": "five exact public files, not every dependency", "verified": verified}


def compare_evaluators(start: dict, stage2: dict, package: dict) -> str:
    required = {"panel_sha256", "panel_manifest_sha256", "harness_sha256", "runtime_execution_sha", "runtime_fingerprint_sha256",
                "prompt_contract_sha256", "action_contract_sha256", "parser_sha256", "carriers", "sampling", "environment_seed_policy"}
    require(set(start) == required and set(stage2) == required, "complete evaluator identity fields required")
    require(start == stage2, "start and Stage2 evaluator identity mismatch")
    require(start["panel_sha256"] == package["evaluation"]["panel_sha256"], "panel identity mismatch")
    require(start["panel_manifest_sha256"] == package["evaluation"]["panel_manifest_sha256"], "panel manifest identity mismatch")
    require(start["environment_seed_policy"] == package["evaluation"]["environment_seed_policy"], "environment seed policy mismatch")
    require(start["sampling"] == package["evaluation"]["sampling"], "canonical sampling mismatch")
    require(set(start["carriers"]) == {"memory", "skill", "agent"}, "three frozen carrier identities required")
    for key in ("harness_sha256", "prompt_contract_sha256", "action_contract_sha256", "parser_sha256"):
        value = start[key]
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"invalid {key}")
    for name, value in start["carriers"].items():
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"invalid carrier hash: {name}")
    require(isinstance(start["runtime_execution_sha"], str) and len(start["runtime_execution_sha"]) == 40
            and all(c in "0123456789abcdef" for c in start["runtime_execution_sha"]), "runtime execution SHA")
    require(isinstance(start["runtime_fingerprint_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", start["runtime_fingerprint_sha256"]) is not None,
            "runtime requires a complete immutable SHA256 fingerprint")
    return hashlib.sha256(json.dumps(start, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit_native_rng_configs(rng: dict, package: dict) -> dict:
    refs = rng.get("effective_native_configs")
    streams = package["stage2"]["streams"]
    require(isinstance(refs, dict) and set(refs) == {s["id"] for s in streams},
            "RNG audit requires both actual native effective configurations")
    for stream in streams:
        artifact = refs[stream["id"]]
        verify_artifact(artifact)
        config = json.loads(Path(artifact["path"]).read_bytes())
        require(config.get("env", {}).get("env_name") == "Webshop",
                "actual native RNG owner must be Webshop")
        require(config.get("trainer", {}).get("checkpoint_environment_rng") is True,
                "actual native trainer must checkpoint environment RNG")
        require(config["trainer"].get("test_freq") == -1
                and config["trainer"].get("val_before_train") is False,
                "actual native auxiliary validation must remain disabled")
        require(config.get("env", {}).get("seed") == stream["environment_seed"]
                and config.get("data", {}).get("seed") == stream["data_seed"]
                and config.get("actor_rollout_ref", {}).get("rollout", {}).get("engine_seed") == stream["policy_seed"],
                "actual native stream RNG seed mismatch")
    return refs


def audit_bridge(bridge: dict, package: dict, source_root: Path) -> dict:
    require(bridge.get("schema") == "openevo.q03-identity-bridge.v1", "bridge schema")
    require(bridge.get("source_repository") == package["source"]["repository"] and bridge.get("source_commit") == package["source"]["commit"], "source identity bridge")
    source = audit_source(source_root, package)
    require(bridge.get("base_revision") == package["start"]["base_revision"], "base revision")
    verify_artifact(bridge["stage1_training_receipt"], package["start"]["stage1_training_receipt_sha256"])
    checkpoint = bridge["stage1_checkpoint_files"]
    require(set(checkpoint) == set(package["start"]["checkpoint_files"]), "complete Stage1 checkpoint index and shards required")
    for role, expected in package["start"]["checkpoint_files"].items():
        verify_artifact(checkpoint[role], expected)
    supporting = bridge.get("stage1_checkpoint_supporting_files")
    require(isinstance(supporting, dict)
            and set(supporting) == set(package["start"]["checkpoint_supporting_files"]),
            "complete Stage1 config/tokenizer support-file bridge required")
    for name, expected in package["start"]["checkpoint_supporting_files"].items():
        verify_artifact(supporting[name], expected)
    require(bridge["streams"] == package["stage2"]["streams"], "two-stream RNG consumer contract drift")
    require(bridge.get("rng_consumer_audit_status") == "PASS", "declared seeds require actual consumer audit")
    verify_artifact(bridge["rng_consumer_audit"])
    rng = json.loads(Path(bridge["rng_consumer_audit"]["path"]).read_bytes())
    require(rng.get("status") == "PASS" and rng.get("streams") == package["stage2"]["streams"]
            and rng.get("source_commit") == package["source"]["commit"], "RNG owning receipt mismatch")
    require(bool(rng.get("consumer_files")), "RNG audit requires actual consumer-file provenance")
    for artifact in rng["consumer_files"]:
        verify_artifact(artifact)
    audit_native_rng_configs(rng, package)
    require(bridge.get("historical_start_score_reused") is False, "old SEED-eval start score cannot be reused")
    fingerprint = compare_evaluators(bridge["canonical_start_evaluator"], bridge["canonical_stage2_evaluator"], package)
    require(bridge.get("compatibility_patches") is not None, "enumerate patches, including an explicit empty list")
    for patch in bridge["compatibility_patches"]:
        require(patch.get("algorithm_preserving") is True, "semantic patch requires a separate fidelity amendment")
        verify_artifact(patch)
        require(bool(patch.get("rationale")) and bool(patch.get("original_sha256")), "patch classification/provenance")
    bindings = bridge.get("evaluator_artifacts", {})
    needed = {"harness", "parser", "memory", "skill", "agent", "contract_receipt", "panel_manifest"}
    require(set(bindings) == needed, "evaluator identity requires local artifact-byte verification")
    evaluator = bridge["canonical_start_evaluator"]
    for name in ("harness", "parser"):
        key = {"harness": "harness_sha256", "parser": "parser_sha256"}[name]
        verify_artifact(bindings[name], evaluator[key])
    for name in ("memory", "skill", "agent"):
        verify_artifact(bindings[name], evaluator["carriers"][name])
    verify_artifact(bindings["panel_manifest"], evaluator["panel_manifest_sha256"])
    verify_artifact(bindings["contract_receipt"])
    contract = json.loads(Path(bindings["contract_receipt"]["path"]).read_bytes())
    for key in ("prompt_contract_sha256", "action_contract_sha256"):
        require(contract.get(key) == evaluator[key], f"semantic contract receipt mismatch: {key}")
    return {"source": source, "canonical_evaluator_fingerprint": fingerprint,
            "actual_start_checkpoint_bytes_verified": True,
            "capability_evaluated": False, "runtime_qualified": False,
            "warning": "CPU bytes/identity audit cannot prove native model semantics or erase sealed negative outcomes."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package", type=Path, default=ROOT / "configs/experiment/advisor-questions/q03-work-package-202610041457.json")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--bridge", type=Path)
    args = parser.parse_args()
    if args.bridge and not args.source_root:
        parser.error("--bridge requires --source-root for actual public-file verification")
    try:
        package = json.loads(args.work_package.read_bytes())
        result = audit_plan(package)
        bridge = audit_bridge(json.loads(args.bridge.read_bytes()), package, args.source_root) if args.bridge else None
        source = audit_source(args.source_root, package) if args.source_root and not bridge else None
        print(json.dumps({"status": "PASS_CPU_AUDIT", "scope": "local-source-start-bridge" if bridge else "local-source" if source else "plan-structure-only",
                          "scientific_complete": False, "activation_allowed": False, "model_calls": 0,
                          "gpu_calls": 0, "final_panel_access": 0, "plan": result, "source": source, "bridge": bridge}, indent=2))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "FAIL_CPU_AUDIT", "error": str(exc), "activation_allowed": False}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
