#!/usr/bin/env python3
"""Build native Passport/Registry/release/profile bindings for frozen fresh lanes.

This CPU-only builder consumes the final root-owned manifest and preregistration.
It neither selects science nor launches/retries a GPU workload.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.openevo_webshop.experiment_passport import with_fingerprints, validate_passport
from scripts.openevo_webshop.experiment_registry import load_current_registry, validate_registry
from scripts.openevo_webshop.execution_readiness import require_scientific_identity_document

HOST = "scripts/openevo_webshop/advisor_fresh_checkpoint_host_20261020.py"
WORKER = "scripts/analysis/advisor_fresh_checkpoint_validation_20261020.py"
RUNTIME_PYTHON = "scripts/runtime/advisor_fresh_runtime_python_20261020.sh"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return sha(path)


def ref(path: str, digest: str | None, reason: str | None = None) -> dict:
    return {"kind": "repository_file", "repository": None, "revision": None,
            "path": path, "sha256": digest, "evidence_path": None, "reason": reason}


def build(args: argparse.Namespace) -> dict:
    root = args.repo_root.resolve()
    if not re.fullmatch(r"\d{12}", args.stamp) or not re.fullmatch(r"[0-9a-f]{40}", args.execution_sha):
        raise ValueError("Explicit12-digit revision stamp and immutable execution SHA required")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
        raise ValueError("Immutable image identity required")
    for name in ("manifest", "preregistration", "runtime_lock"):
        path = getattr(args, name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Frozen repository inputs must be repository-relative")
    if sha(root / args.manifest) != args.manifest_sha256:
        raise ValueError("Supplied frozen manifest SHA differs")
    # Use the actual readiness consumer before writing any derived authority.
    require_scientific_identity_document(root / args.preregistration)
    manifest = json.loads((root / args.manifest).read_text())
    if (manifest.get("final_panel_access_allowed") is not False
            or manifest.get("optimizer_updates") != 0 or len(manifest.get("tasks", [])) != 64):
        raise ValueError("Manifest must freeze64 tasks, zero optimizer updates and no final access")
    if manifest.get("image_id") != args.image:
        raise ValueError("Manifest and canonical image identity must agree")
    if Path(manifest["core_root"]) != args.core_root:
        raise ValueError("Manifest and canonical Core root must agree")
    panel_sha = hashlib.sha256(json.dumps(manifest["tasks"], ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("panel_sha256") != panel_sha:
        raise ValueError("Frozen task panel SHA drift")
    lanes = json.loads(args.lane_map.read_text())
    ids = [item for lane in lanes for item in lane["checkpoint_ids"]]
    if (not lanes or len(ids) != len(set(ids)) or any(item not in manifest["checkpoints"] for item in ids)
            or len({lane["gpu_uuid"] for lane in lanes}) != len(lanes)
            or len({lane["lane_id"] for lane in lanes}) != len(lanes)):
        raise ValueError("Each frozen checkpoint and physical GPU must belong to one unique lane")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", item) for item in ids):
        raise ValueError("Checkpoint IDs must be safe single output path components")
    # No hidden checkpoint creation or expansion of the root's frozen pool.
    if any(not re.fullmatch(r"[a-z0-9-]+", lane["lane_id"]) for lane in lanes):
        raise ValueError("Lane IDs must be plain lowercase path components")
    model = manifest["model"]
    fingerprint_rel = f"configs/runtime/advisor-fresh-runtime-fingerprint-{args.stamp}.json"
    fingerprint = json.loads((root / args.runtime_fingerprint_base).read_text())
    if (fingerprint["execution_image"]["id"] != args.image
            or fingerprint["openevo_core"]["path"] != str(args.core_root)
            or fingerprint["openevo_core"]["revision"] != args.core_sha
            or fingerprint["runtime_code"]["path"] != manifest["runtime_source_root"]):
        raise ValueError("Fresh fingerprint base does not match the measured runtime family")
    fingerprint["host_runtime_python"] = {"executable": str(args.execution_root / RUNTIME_PYTHON),
        "executable_kind": "Immutable-image CPU wrapper; no GPU allocation", "wrapper_sha256": sha(root / RUNTIME_PYTHON)}
    fingerprint["model"] = {"path": model["local_model_path"], "hf_id": model.get("model_id", "Qwen/Qwen3-1.7B"),
        "content_pin": model["revision"], "manifest_sha256": args.manifest_sha256}
    fingerprint["seed_root"]["path"] = manifest["seed_root"]
    fingerprint_sha = save(root / fingerprint_rel, fingerprint)
    prereg_ref = ref(str(args.preregistration), sha(root / args.preregistration))
    manifest_ref = ref(str(args.manifest), args.manifest_sha256)
    host_ref = ref(HOST, sha(root / HOST))
    worker_ref = ref(WORKER, sha(root / WORKER))
    lock_ref = ref(fingerprint_rel, fingerprint_sha)
    dependency_ref = ref(str(args.runtime_lock), sha(root / args.runtime_lock))
    registry, _ = load_current_registry(root)
    registry = json.loads(json.dumps(registry))
    if validate_registry(registry, root):
        raise ValueError("Current native Registry is invalid; preserve and repair its real owner first")
    generated = []
    created = datetime.now(timezone.utc).isoformat()
    for lane in lanes:
        count = 64 * len(lane["checkpoint_ids"])
        if count == 0 or not re.fullmatch(r"GPU-[0-9a-f-]+", lane["gpu_uuid"]):
            raise ValueError("Each lane needs an exact physical GPU UUID and at least one checkpoint")
        experiment_id = args.stamp + "-advisor-fresh-" + lane["lane_id"]
        campaign_id = args.stamp[:8] + "-" + args.stamp[8:] + "-advisor-fresh-" + lane["lane_id"]
        run_id = "advisor-fresh-" + args.stamp + "-" + lane["lane_id"]
        profile_rel = f"configs/experiment/profiles/{run_id}.json"
        release_rel = f"configs/experiment/activations/{run_id}-release.json"
        router_rel = f"configs/experiment/activations/{run_id}-current-campaign.json"
        path_rel = f"configs/experiment/path-contracts/{run_id}.json"
        passport_rel = f"configs/experiment/passports/{experiment_id}/passport.json"
        output_root = str(args.output_base / run_id)
        command = ["python3", "${EXPERIMENT_DIR}/" + HOST,
                   "--manifest", str(args.execution_root / args.manifest),
                   "--manifest-sha256", args.manifest_sha256, "--run-id", run_id,
                   "--gpu-uuid", lane["gpu_uuid"], "--image", args.image,
                   "--output", "${RUN_DIR}/evaluation"]
        for checkpoint_id in lane["checkpoint_ids"]:
            command += ["--checkpoint-id", checkpoint_id]
        required = [model["local_model_path"], manifest["seed_root"], manifest["runtime_source_root"],
                    str(args.execution_root / args.manifest)]
        required += [manifest["checkpoints"][item]["adapter_path"] for item in lane["checkpoint_ids"]]
        required += [manifest["common_carriers"][item]["path"] for item in ("memory", "skill", "agent")]
        directories = {model["local_model_path"], manifest["seed_root"], manifest["runtime_source_root"],
                       *(manifest["checkpoints"][item]["adapter_path"] for item in lane["checkpoint_ids"])}
        profile = {
            "schema": "openevo.webshop.run-profile.v1", "run_id": run_id,
            "image": args.image, "image_id": args.image, "workspace": str(args.workspace),
            "identity": {"uid": 1001, "gid": 1001}, "command": command,
            "execution": {"mode": "host-orchestrator"}, "container": {"network": "none"},
            "source": {"experiment": {"path": str(args.execution_root), "revision": args.execution_sha},
                       "openevo": {"path": str(args.core_root), "revision": args.core_sha}},
            "paths": {"cache": "cache", "data": "datasets/webshop-1000", "models": "models", "runs": str(args.output_base)},
            "tracking": {"mode": "disabled", "entity": "example-entity", "project": "openevo-webshop-experiments",
                         "group": campaign_id, "on_error": "fail", "probe_epoch": args.stamp},
            "gpu_gate": {"authorized_uuids": [lane["gpu_uuid"]], "mode": "idle",
                         "interval_seconds": 1, "max_idle_memory_mib": 16,
                         "max_idle_utilization_percent": 0, "minimum_free_memory_mib": 16384,
                         "timeout_seconds": 0},
            "prelaunch": {"runtime_python": "python3"}, "lock_root": ".gpu-locks",
            "required_paths": [{"path": path, "expected_type": "dir" if path in directories
                                else "file"} for path in required],
        }
        profile_sha = save(root / profile_rel, profile)
        path_contract = {"schema": "openevo.advisor-fresh-path-contract.v1", "campaign_id": campaign_id,
                         "output_root": output_root, "read_only_inputs": required,
                         "runtime_uid_gid": "1001:1001", "network": "none", "max_output_bytes": 2 * 1024**3}
        path_sha = save(root / path_rel, path_contract)
        authority = [str(args.manifest), str(args.preregistration), HOST, WORKER, RUNTIME_PYTHON,
                     fingerprint_rel, profile_rel,
                     release_rel, router_rel, path_rel, passport_rel]
        science = {
            "research_question": "Do the prospectively selected fixed checkpoints generalize on64 unseen tasks?",
            "hypothesis": "Report paired checkpoint differences on the same fresh panel without training or selection.",
            "treatment": "Evaluate only the frozen lane checkpoint IDs: " + ", ".join(lane["checkpoint_ids"]),
            "what_changed": ["Prospective outcome-blind64-task panel; immutable existing checkpoints only."],
            "what_must_stay_fixed": ["Frozen panel, checkpoints, carrier contents, evaluator, task seeds and sampling."],
            "benchmark": "WebShop", "split": "Fresh64 disjoint from forbidden training/evaluation/final panels",
            "model": {"model_id": model.get("model_id", "Qwen/Qwen3-1.7B"), "model_revision": model["revision"],
                      "tokenizer_revision": model["revision"], "dtype": "bfloat16", "asset": manifest_ref},
            "harness": {"profile": run_id, "thinking": False, "source": worker_ref},
            "task_manifest": manifest_ref, "training_data_identity": manifest_ref,
            "prompt_contract": manifest_ref, "parser_contract": worker_ref, "evaluator_contract": worker_ref,
            "action_contract": worker_ref,
            "sampling": {"temperature": 1.0, "do_sample": True, "top_p": 1.0, "top_k": 0,
                         "max_new_tokens": 512, "extra_settings": {"max_steps": 15, "repetition_penalty": 1.0}},
            "seed_policy": {"seed_base": None, "schedule_id": manifest["panel_sha256"], "schedule": manifest_ref,
                            "rng_contract": "Exact manifest task paired_generation_seed, unchanged per-step evaluator rule."},
            "sd_lora": {"rank": None, "alpha": None, "target_modules": None, "effective_rank_cap": None,
                        "source": manifest_ref, "extra_settings": {"checkpoints": {
                            item: manifest["checkpoints"][item] for item in lane["checkpoint_ids"]}, "updates": 0}},
            "optimizer": {"name": None, "learning_rate": None, "batch_size": None, "epochs": None,
                          "gradient_accumulation_steps": None, "extra_settings": {"optimizer_updates": 0}},
            "external_teacher": {"role": "none", "future_call_budget": 0, "prior_analyses": prereg_ref},
            "final_panel_policy": {"policy": "LOCKED_ZERO_ACCESS", "future_access_limit": 0, "source": prereg_ref},
            "acceptance_policy": {"name": "FROZEN_FRESH_PAIRED_REPORT", "candidate_rule": "No outcome-based retry or selection",
                                  "credit_assignment": "Report all frozen checkpoint64-task results and paired uncertainty", "source": prereg_ref},
            "stop_rule": {"max_rollouts": count, "max_rounds": 0, "source": prereg_ref},
            "human_intervention_policy": "No repeated cell START; failed or partial cells need explicit reconciliation.",
            "claim_boundary": "Fixed-checkpoint fresh generalization only; no causal training claims or Q03 intervention.",
            "protocol_fingerprint": None,
        }
        passport = {
            "schema": "openevo.experiment-passport.202609111800", "experiment_id": experiment_id,
            "parent_experiment_id": None, "campaign_id": campaign_id, "status": "FROZEN", "created_at": created,
            "scientific_identity": science,
            "budget_contract": {"rollout_limit": count, "round_limit": 0, "horizon": 15, "rollouts_per_round": 0,
                                "optimizer_step_limit": 0, "external_teacher_call_limit": 0, "final_panel_access_limit": 0,
                                "source": prereg_ref, "accounting_rule": f"Exactly{len(lane['checkpoint_ids'])} fixed checkpoints ×64 fresh tasks; no outcome retry."},
            "initial_execution_identity": {
                "experiment_repository": "mykcs/openevo-experiment", "initial_experiment_sha": args.execution_sha,
                "frozen_science_sha": args.execution_sha, "initial_ray_sha": args.core_sha,
                "upstream_repository": "mykcs/OpenEvo", "upstream_openevo_sha": args.core_sha,
                "initial_run_root": output_root, "python_version": args.python_version,
                "runtime_image_digest": args.image, "runtime_digest_kind": "IMAGE_ID",
                "launcher": host_ref, "activation": manifest_ref, "dependency_lock": dependency_ref,
                "runtime_asset_lock": lock_ref, "runtime_fingerprint": None,
                "release": ref(release_rel, None, "Release embeds this Passport digest; exact release hash belongs to Formal Start."),
            },
            "authority_paths": authority,
            "tracking_policy": {"role": "MIRROR_ONLY", "scientific_authority": False, "entity": None, "project": None,
                                "mode_order": ["online", "offline", "disabled"], "approved_artifact_roles": ["passport", "run-manifest", "startup-receipt", "closeout-receipt", "evaluation-summary"]},
            "artifact_policy": {"assets": [], "default_disclosure_state": "PRIVATE", "default_recovery_state": "SERVER_ONLY",
                                "large_asset_provider": "Existing durable server storage; no external publication.",
                                "state_model_policy": "docs/operations/publication/RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md", "wandb_is_sole_backup": False},
            "resume_policy": {"policy": "MANUAL_REVIEW_ONLY", "reason": "No automatic replay of any cell START; preserve partial consumption.",
                              "automatic_gpu_start": False, "requires_receipt_chain": True},
            "provenance": {"status": "NEW_DECLARATION", "unknown_fields": [], "evidence_paths": authority,
                           "parent_lineage_note": "Prospective evaluation of pre-existing frozen checkpoint identities; no training continuation.",
                           "revision_naming_note": "Explicit12-digit creation stamp; every lane has a unique writer/output."},
            "passport_payload_sha256": None,
        }
        passport = with_fingerprints(passport)
        validate_passport(passport, root=root)
        save(root / passport_rel, passport)
        artifacts = [{"path": str(args.manifest), "sha256": args.manifest_sha256, "role": "evaluation_contract"},
                     {"path": str(args.preregistration), "sha256": prereg_ref["sha256"], "role": "preregistration"},
                     {"path": HOST, "sha256": host_ref["sha256"], "role": "launcher"},
                     {"path": WORKER, "sha256": worker_ref["sha256"], "role": "evaluator"},
                     {"path": RUNTIME_PYTHON, "sha256": sha(root / RUNTIME_PYTHON), "role": "runtime_python_wrapper"},
                     {"path": fingerprint_rel, "sha256": fingerprint_sha, "role": "runtime_fingerprint"}]
        release = {"schema": "openevo.experiment-advisor-fresh-evaluation-release.v1", "campaign_id": campaign_id,
                   "status": "launch-authorized", "formal_task_consumption_allowed": True, "gpu_allocation_allowed": True,
                   "authorized_gpu_uuids": [lane["gpu_uuid"]], "evaluation_episode_limit": count,
                   "training_rollout_limit": 0, "formal_rollout_limit": count, "final_panel_access_limit": 0,
                   "experiment_identity": {"experiment_id": experiment_id, "passport_sha256": passport["passport_payload_sha256"]},
                   "preregistration": {"path": str(args.preregistration), "sha256": prereg_ref["sha256"]},
                   "execution_readiness": {"profile": {"path": profile_rel, "sha256": profile_sha}, "frozen_scientific_artifacts": artifacts},
                   "path_contract": {"path": path_rel, "sha256": path_sha}}
        release_sha = save(root / release_rel, release)
        save(root / router_rel, {"schema": "openevo.current-campaign.advisor-fresh.v1", "campaign_id": campaign_id,
             "status": "launch-authorized", "formal_task_consumption_allowed": True,
             "authorized_gpu_uuids": [lane["gpu_uuid"]], "release_receipt": release_rel,
             "release_receipt_sha256": release_sha, "latest_event": {"release_receipt": release_rel, "release_receipt_sha256": release_sha}})
        registry["entries"].append({"experiment_id": experiment_id, "display_name": "Advisor fresh checkpoint evaluation " + lane["lane_id"],
            "aliases": [run_id], "lineage_parent": None, "lifecycle_class": "ACTIVE_FORMAL", "passport_path": passport_rel,
            "passport_sha256": passport["passport_payload_sha256"], "latest_authority_paths": authority,
            "state_source": {"type": "release_external_executor", "path": None, "release_path": release_rel, "required_equal": {}}})
        generated.append({"run_id": run_id, "release": release_rel, "current_campaign": router_rel,
                          "output_root": output_root, "gpu_uuid": lane["gpu_uuid"], "evaluation_episodes": count})
    errors = validate_registry(registry, root)
    if errors:
        raise ValueError("Generated Registry invalid: " + "; ".join(errors))
    registry_rel = f"configs/experiment/experiment-registry-advisor-fresh-{args.stamp}.json"
    registry_sha = save(root / registry_rel, registry)
    # This is the existing single Registry owner; no second routing mechanism.
    (root / "configs/experiment/current-experiment-registry.json").write_text(
        json.dumps({"path": registry_rel, "sha256": registry_sha}, indent=2, sort_keys=True) + "\n")
    return {"status": "CANONICAL_METADATA_BUILT_NOT_LAUNCHED", "lanes": generated, "registry": registry_rel,
            "fingerprint": fingerprint_rel, "fingerprint_sha256": fingerprint_sha,
            "runtime_python": str(args.execution_root / RUNTIME_PYTHON)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--preregistration", type=Path, required=True)
    p.add_argument("--lane-map", type=Path, required=True)
    p.add_argument("--stamp", required=True)
    p.add_argument("--execution-root", type=Path, required=True)
    p.add_argument("--execution-sha", required=True)
    p.add_argument("--workspace", type=Path, default=Path("/workspace"))
    p.add_argument("--core-root", type=Path, default=Path("/workspace/control/openevo-core"))
    p.add_argument("--core-sha", required=True)
    p.add_argument("--output-base", type=Path, default=Path("/workspace/runs"))
    p.add_argument("--image", required=True)
    p.add_argument("--runtime-lock", type=Path, default=Path("configs/assets/runtimes/h144-scientific-runtime-v1.json"))
    p.add_argument("--runtime-fingerprint-base", type=Path,
                   default=Path("configs/runtime/advisor-fresh-runtime-fingerprint-base-20261020.json"))
    p.add_argument("--python-version", default="3.11.16")
    print(json.dumps(build(p.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
