#!/usr/bin/env python3
"""W&B-tracked wrapper for the existing parametric R&D harvest."""

import argparse
import json
from pathlib import Path

from parametric_rnd import build_manifest, run_harvest
from wandb_tracking import (
    add_wandb_arguments,
    campaign_notes,
    campaign_tracking_config,
    load_current_campaign,
    short_run_name,
    standard_tags,
    start_wandb_run,
)


_EXPERIENCE_METRICS = (
    "experience/final_task_score",
    "experience/source_reward",
    "experience/qualified_positive",
    "experience/partial_positive",
    "experience/identity_ok",
    "experience/trace_complete",
    "experience/parser_failure",
    "experience/parser_retry_used",
    "experience/parser_retry_events",
    "experience/error",
    "experience/episode_steps",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--seed-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    harvest = sub.add_parser("harvest")
    harvest.add_argument("--manifest", type=Path, required=True)
    harvest.add_argument("--seed-root", type=Path, required=True)
    harvest.add_argument("--model", required=True)
    harvest.add_argument("--output", type=Path, required=True)
    harvest.add_argument("--gpu-uuid", required=True)
    harvest.add_argument("--lock", type=Path)
    harvest.add_argument("--tasks", type=int, nargs="+", required=True)
    add_wandb_arguments(harvest, default_project="openevo-webshop-experiments")
    args = parser.parse_args()

    if args.command == "manifest":
        payload = build_manifest(args.seed_root, args.output)
        print(json.dumps({"manifest": str(args.output), "manifest_sha256": payload["manifest_sha256"]}, sort_keys=True))
        return 0

    campaign = load_current_campaign()
    group = args.wandb_group or args.campaign_id or "webshop-experience-harvest"
    campaign_id = args.campaign_id or group
    run_name = short_run_name(arm="selfrollout", unit="harvest")
    config = {
        **campaign_tracking_config(campaign),
        "campaign_id": campaign_id,
        "arm": "selfrollout",
        "method": "self-rollout",
        "model_id": args.model,
        "gpu_uuid": args.gpu_uuid,
        "tasks": args.tasks,
        "manifest": str(args.manifest),
        "tracking_mode": args.wandb_mode,
    }
    tracker = start_wandb_run(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        job_type="experience-harvest",
        group=group,
        config=config,
        tags=standard_tags(campaign, method="self-rollout", arm="selfrollout", extra=("class:diagnostic",)),
        notes=campaign_notes(campaign),
        metric_axes={metric: "experience/attempt_index" for metric in _EXPERIENCE_METRICS},
        directory=args.wandb_dir or (args.output / "wandb"),
    )
    try:
        code = run_harvest(args)
        attempts_path = args.output / "attempts.jsonl"
        qualified = parser_failures = retries = partial_positive = errors = count = 0
        scores: list[float] = []
        source_rewards: list[float] = []
        episode_steps: list[int] = []
        table_rows: list[list[object]] = []
        if attempts_path.is_file():
            for count, line in enumerate(attempts_path.read_text(encoding="utf-8").splitlines(), start=1):
                row = json.loads(line)
                qualified_row = int(bool(row.get("qualified_positive")))
                parser_failure_row = int(bool(row.get("parser_or_fallback_failure")))
                retry_row = int(bool(row.get("parser_retry_used")))
                partial_row = int(bool(row.get("partial_positive")))
                error_row = int(bool(row.get("error")))
                qualified += qualified_row
                parser_failures += parser_failure_row
                retries += retry_row
                partial_positive += partial_row
                errors += error_row
                score = float(row.get("final_task_score") or 0.0)
                source_reward = float(row.get("source_reward") or 0.0)
                scores.append(score)
                source_rewards.append(source_reward)
                episode = row.get("episode") if isinstance(row.get("episode"), dict) else {}
                steps = int(episode.get("steps") or len(episode.get("trace") or []))
                episode_steps.append(steps)
                tracker.log({
                    "experience/attempt_index": count,
                    "experience/final_task_score": score,
                    "experience/source_reward": source_reward,
                    "experience/qualified_positive": qualified_row,
                    "experience/partial_positive": partial_row,
                    "experience/identity_ok": int(bool(row.get("identity_ok"))),
                    "experience/trace_complete": int(bool(row.get("trace_complete"))),
                    "experience/parser_failure": parser_failure_row,
                    "experience/parser_retry_used": retry_row,
                    "experience/parser_retry_events": int(row.get("parser_retry_events") or 0),
                    "experience/error": error_row,
                    "experience/episode_steps": steps,
                })
                table_rows.append([
                    campaign_id, row.get("attempt_id"), row.get("task_id"), row.get("goal_idx"), row.get("rollout_seed"),
                    score, source_reward, bool(qualified_row), bool(partial_row), bool(row.get("identity_ok")),
                    bool(row.get("trace_complete")), bool(parser_failure_row), bool(retry_row), bool(error_row), steps,
                ])

        tracker.log_table(
            "experience/attempt_table",
            columns=("campaign_id", "attempt_id", "task_id", "goal_idx", "generation_seed", "final_task_score", "source_reward", "qualified_positive", "partial_positive", "identity_ok", "trace_complete", "parser_failure", "parser_retry_used", "error", "episode_steps"),
            data=table_rows,
        )
        denominator = max(count, 1)
        summary_metrics = {
            "experience/attempt_count": count,
            "experience/qualified_positive_count": qualified,
            "experience/qualified_positive_rate": qualified / denominator,
            "experience/partial_positive_count": partial_positive,
            "experience/parser_failure_count": parser_failures,
            "experience/parser_failure_rate": parser_failures / denominator,
            "experience/parser_retry_count": retries,
            "experience/parser_retry_rate": retries / denominator,
            "experience/error_count": errors,
            "experience/error_rate": errors / denominator,
            "eval/avg_task_score": sum(scores) / len(scores) if scores else 0.0,
            "eval/max_task_score": max(scores) if scores else 0.0,
            "eval/avg_source_reward": sum(source_rewards) / len(source_rewards) if source_rewards else 0.0,
            "eval/avg_episode_steps": sum(episode_steps) / len(episode_steps) if episode_steps else 0.0,
        }
        summary_path = args.output / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_metrics["system/wall_time_s"] = summary.get("wall_time_seconds")
            summary["tracking"] = {"wandb": tracker.metadata(), "campaign_id": campaign_id, "wandb_group": group, "wandb_run_name": run_name}
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tracker.log_artifact_file(summary_path, name=f"{campaign_id}-harvest-summary", metadata={"campaign_id": campaign_id, "tasks": args.tasks, "attempt_count": count})
        tracker.log(summary_metrics)
        tracker.set_summary({"run/status": "pass" if code == 0 else "failed", **summary_metrics})
        return code
    except BaseException:
        tracker.set_summary({"run/status": "failed"})
        raise
    finally:
        tracker.finish()


if __name__ == "__main__":
    raise SystemExit(main())
