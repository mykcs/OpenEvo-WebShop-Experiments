#!/usr/bin/env python3
"""W&B-tracked wrapper for run_parametric_smoke.py."""

import argparse
import json
from pathlib import Path
from typing import Any

from run_parametric_smoke import run
from wandb_tracking import (
    add_wandb_arguments,
    campaign_notes,
    campaign_tracking_config,
    load_current_campaign,
    short_run_name,
    standard_tags,
    start_wandb_run,
    trainer_wandb_environment,
)


_TRAINING_SEED = 1993


def _numeric_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload.get("artifact_manifest")
    if not isinstance(manifest, dict):
        manifest = {}
    gpu_peak_bytes = manifest.get("gpu_peak_memory_bytes")
    metrics: dict[str, Any] = {
        "train/record_count": payload.get("training_record_count"),
        "train/replay_training_record_count": manifest.get("replay_training_record_count"),
        "train/optimizer_training_record_count": manifest.get("optimizer_training_record_count"),
        "train/replay_buffer_record_count": manifest.get("replay_buffer_record_count"),
        "train/optimizer_steps": payload.get("optimizer_steps"),
        "train/final_loss": payload.get("training_loss"),
        "train/training_time_s": manifest.get("training_time_seconds"),
        "train/task_index": manifest.get("task_index"),
        "train/component_count": manifest.get("component_count"),
        "train/effective_rank": manifest.get("effective_rank"),
        "artifact/state_weights_size_bytes": manifest.get("state_weights_size_bytes"),
        "artifact/replay_data_size_bytes": manifest.get("replay_data_size_bytes"),
        "system/gpu_peak_memory_mb": float(gpu_peak_bytes) / (1024 * 1024) if isinstance(gpu_peak_bytes, (int, float)) else None,
    }
    coefficients = manifest.get("components")
    if isinstance(coefficients, list):
        for index, component in enumerate(coefficients):
            if isinstance(component, dict):
                metrics[f"train/coefficient_{index}"] = component.get("coefficient")
                metrics[f"train/component_rank_{index}"] = component.get("rank")
    return metrics


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-manifest", type=Path, required=True)
    p.add_argument("--dataset-artifact-id", default=None)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--run-id", required=True, help="Local experiment run ID; not the W&B run ID.")
    p.add_argument("--benchmark", choices=("webshop", "alfworld"), default="webshop")
    p.add_argument("--python", default=None)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--model-revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    p.add_argument("--max-steps", type=int, default=2)
    p.add_argument("--max-records", type=int, default=30)
    p.add_argument("--minimum-reward", type=float, default=0.0)
    p.add_argument("--prior-adapter", type=Path)
    p.add_argument("--prior-artifact-id")
    p.add_argument("--experiment-git-sha")
    p.add_argument("--upstream-openevo-sha")
    p.add_argument("--runtime-fingerprint")
    p.add_argument("--image-digest")
    p.add_argument("--gpu-uuid")
    add_wandb_arguments(p, default_project="openevo-webshop-experiments")
    args = p.parse_args()

    campaign = load_current_campaign()
    arm = "cumulative" if args.prior_adapter is not None else "adapter1x"
    group = args.wandb_group or args.campaign_id or f"{args.benchmark}-sd-lora-{args.run_id}"
    campaign_id = args.campaign_id or group
    wrapper_name = short_run_name(arm=arm, unit="train", seed=_TRAINING_SEED)
    trainer_name = short_run_name(arm=arm, unit="trainer", seed=_TRAINING_SEED)
    config = {
        **campaign_tracking_config(campaign),
        "campaign_id": campaign_id,
        "arm": arm,
        "method": "sd-lora",
        "local_run_id": args.run_id,
        "benchmark": args.benchmark,
        "model_id": args.base_model,
        "model_revision": args.model_revision,
        "training_seed": _TRAINING_SEED,
        "max_steps": args.max_steps,
        "max_records": args.max_records,
        "minimum_reward": args.minimum_reward,
        "prior_adapter": str(args.prior_adapter) if args.prior_adapter else None,
        "prior_artifact_id": args.prior_artifact_id,
        "tracking_mode": args.wandb_mode,
        "experiment_git_sha": args.experiment_git_sha,
        "upstream_openevo_sha": args.upstream_openevo_sha,
        "runtime_fingerprint": args.runtime_fingerprint,
        "image_digest": args.image_digest,
        "gpu_uuid": args.gpu_uuid,
    }
    notes = campaign_notes(campaign)
    tracker = start_wandb_run(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wrapper_name,
        job_type="training",
        group=group,
        config=config,
        tags=standard_tags(campaign, method="sd-lora", arm=arm, extra=("class:smoke",)),
        notes=notes,
        directory=args.wandb_dir or (args.artifact_root / "wandb"),
    )
    try:
        with trainer_wandb_environment(
            tracker,
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=group,
            run_name=trainer_name,
            campaign_id=campaign_id,
            notes=notes,
        ):
            payload = run(args)

        metrics = _numeric_metrics(payload)
        tracker.log(metrics)
        tracker.set_summary({"run/status": "pass", **metrics})
        payload["tracking"] = {
            "wandb": tracker.metadata(),
            "campaign_id": campaign_id,
            "wandb_group": group,
            "wandb_run_name": wrapper_name,
            "trainer_run_name": trainer_name,
        }
        args.evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tracker.log_artifact_file(
            args.evidence,
            name=f"{campaign_id}-{arm}-training-evidence",
            metadata={"campaign_id": campaign_id, "arm": arm, "benchmark": args.benchmark, "local_run_id": args.run_id},
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except BaseException:
        tracker.set_summary({"run/status": "failed"})
        raise
    finally:
        tracker.finish()


if __name__ == "__main__":
    raise SystemExit(main())
