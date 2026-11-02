#!/usr/bin/env python3
"""W&B-tracked Phase H0 natural-success search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from parametric_rnd import run_harvest
from wandb_tracking import add_wandb_arguments, start_wandb_run


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _action_counts(episode: dict[str, Any]) -> dict[str, int]:
    counts = {name: 0 for name in ("search", "click", "select", "buy", "other")}
    for step in (
        episode.get("trace", []) if isinstance(episode.get("trace"), list) else []
    ):
        action = str(step.get("action") or "").lower()
        matched = next(
            (
                name
                for name in counts
                if name != "other" and action.startswith(f"{name}[")
            ),
            None,
        )
        counts[matched or "other"] += 1
    return counts


def _failure_mode(record: dict[str, Any]) -> str:
    if record.get("error"):
        return "runtime-failure"
    if not record.get("identity_ok") or not record.get("trace_complete"):
        return "protocol-invalid"
    if record.get("parser_or_fallback_failure"):
        return "parser-failure"
    if float(record.get("final_task_score") or 0.0) > 0.0:
        return "success-or-partial"
    return "zero-reward"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    harvest = parser.add_argument_group("frozen H0 harvest")
    harvest.add_argument("--manifest", type=Path, required=True)
    harvest.add_argument("--seed-root", type=Path, required=True)
    harvest.add_argument("--model", required=True)
    harvest.add_argument("--model-revision", required=True)
    harvest.add_argument("--tokenizer-revision", required=True)
    harvest.add_argument("--upstream-openevo-sha", required=True)
    harvest.add_argument("--experiment-git-sha", required=True)
    harvest.add_argument("--runtime-fingerprint", required=True)
    harvest.add_argument("--image-digest", required=True)
    harvest.add_argument("--output", type=Path, required=True)
    harvest.add_argument("--gpu-uuid", required=True)
    harvest.add_argument("--lock", type=Path)
    harvest.add_argument("--tasks", type=int, nargs="+", required=True)
    harvest.add_argument("--rollout-seeds", type=int, nargs="+", required=True)
    harvest.add_argument("--max-steps", type=int, default=15)
    harvest.add_argument("--max-new", type=int, default=256)
    harvest.add_argument("--temperature", type=float, default=0.4)
    harvest.add_argument(
        "--prompt-contract",
        default="seed3090.webshop-shared-prompt.v1",
        help="metadata contract for the frozen run; the backend remains unchanged",
    )
    add_wandb_arguments(parser, default_project="openevo-webshop-experiments")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol = {
        "benchmark": "webshop",
        "phase": "H0",
        "split": "train-only-h0",
        "task_manifest_sha256": manifest["manifest_sha256"],
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "environment_seed": 2026,
        "generation_seeds": args.rollout_seeds,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new,
        "prompt_contract": args.prompt_contract,
        "action_contract": "seed3090.webshop-action-parser.v1",
        "evaluator": "WebShop task_score/won",
        "parser_retry": {"seed_offset": 900000, "max_retries_per_step": 1},
    }
    protocol_fingerprint = _fingerprint(protocol)
    group = args.wandb_group or args.campaign_id or "H0-natural-success-search-v1"
    config = {
        "standard_core": "v1",
        "campaign_id": args.campaign_id or group,
        "experiment_group": group,
        "phase": "H0",
        "benchmark": "webshop",
        "split": "train-only-h0",
        "arm": "base",
        "method": "natural-success-search",
        "task_manifest_identity": manifest["manifest_sha256"],
        "task_ids": [f"webshop_{task:06d}" for task in args.tasks],
        "environment_seed": 2026,
        "generation_seeds": args.rollout_seeds,
        "openevo_experiment_sha": args.experiment_git_sha,
        "upstream_openevo_sha": args.upstream_openevo_sha,
        "model_id": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "runtime_fingerprint": args.runtime_fingerprint,
        "image_digest": args.image_digest,
        "protocol_fingerprint": protocol_fingerprint,
        "gpu_uuid": args.gpu_uuid,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new,
        "parser_retry_seed_offset": 900000,
        "training_dataset_manifest_digest": None,
    }
    tracker = start_wandb_run(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="base-h0-search",
        job_type="experience-harvest",
        group=group,
        config=config,
        tags=(
            "benchmark:webshop",
            "phase:h0",
            "split:train-only-h0",
            "method:natural-success-search",
            "arm:base",
            "class:diagnostic",
            "hardware:rtx5090",
        ),
        notes=(
            "H0 asks whether frozen Qwen2.5-3B has non-zero natural WebShop "
            "success probability; only generation seeds vary."
        ),
        metric_axes={
            "experience/qualified_positive": "experience/attempt_index",
            "experience/final_task_score": "experience/attempt_index",
            "experience/parser_failure": "experience/attempt_index",
            "experience/episode_steps": "experience/attempt_index",
        },
        directory=args.wandb_dir or (args.output / "wandb"),
    )
    try:
        code = run_harvest(args)
        attempts_path = args.output / "attempts.jsonl"
        table_rows: list[list[Any]] = []
        counts: dict[str, int] = {}
        scores: list[float] = []
        parser_failures = 0
        qualified = 0
        partial = 0
        for attempt_index, line in enumerate(
            attempts_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            row = json.loads(line)
            episode = row.get("episode") if isinstance(row.get("episode"), dict) else {}
            trace = (
                episode.get("trace") if isinstance(episode.get("trace"), list) else []
            )
            action_counts = _action_counts(episode)
            mode = _failure_mode(row)
            counts[mode] = counts.get(mode, 0) + 1
            score = float(row.get("final_task_score") or 0.0)
            scores.append(score)
            qualified += int(bool(row.get("qualified_positive")))
            partial += int(bool(row.get("partial_positive")))
            parser_failures += int(bool(row.get("parser_or_fallback_failure")))
            tracker.log(
                {
                    "experience/attempt_index": attempt_index,
                    "experience/final_task_score": score,
                    "experience/source_reward": float(row.get("source_reward") or 0.0),
                    "experience/qualified_positive": int(
                        bool(row.get("qualified_positive"))
                    ),
                    "experience/partial_positive": int(
                        bool(row.get("partial_positive"))
                    ),
                    "experience/parser_failure": int(
                        bool(row.get("parser_or_fallback_failure"))
                    ),
                    "experience/parser_retry_used": int(
                        bool(row.get("parser_retry_used"))
                    ),
                    "experience/identity_ok": int(bool(row.get("identity_ok"))),
                    "experience/trace_complete": int(bool(row.get("trace_complete"))),
                    "experience/episode_steps": len(trace),
                    "experience/generated_chars": sum(
                        len(str(step.get("raw_completion") or ""))
                        for step in trace
                        if isinstance(step, dict)
                    ),
                }
            )
            table_rows.append(
                [
                    attempt_index,
                    row.get("task_id"),
                    row.get("rollout_seed"),
                    score,
                    bool(row.get("qualified_positive")),
                    bool(row.get("partial_positive")),
                    bool(row.get("parser_or_fallback_failure")),
                    mode,
                    len(trace),
                    json.dumps(action_counts, sort_keys=True),
                ]
            )
        total = max(len(table_rows), 1)
        summary_metrics = {
            "experience/attempt_count": len(table_rows),
            "experience/qualified_positive_count": qualified,
            "experience/qualified_positive_rate": qualified / total,
            "experience/partial_positive_count": partial,
            "experience/nonzero_reward_count": sum(score > 0.0 for score in scores),
            "experience/nonzero_reward_rate": sum(score > 0.0 for score in scores)
            / total,
            "experience/parser_failure_count": parser_failures,
            "experience/parser_failure_rate": parser_failures / total,
            "eval/avg_task_score": sum(scores) / len(scores) if scores else 0.0,
            "eval/max_task_score": max(scores) if scores else 0.0,
        }
        for mode, count in sorted(counts.items()):
            summary_metrics[f"experience/failure_mode/{mode}"] = count
        tracker.log(summary_metrics)
        tracker.log_table(
            "experience/attempt_table",
            columns=(
                "attempt_index",
                "task_id",
                "generation_seed",
                "final_task_score",
                "qualified_positive",
                "partial_positive",
                "parser_failure",
                "failure_mode",
                "episode_steps",
                "action_type_counts",
            ),
            data=table_rows,
        )
        summary_path = args.output / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else {}
        )
        summary["tracking"] = {
            "wandb": tracker.metadata(),
            "group": group,
            "protocol_fingerprint": protocol_fingerprint,
        }
        summary["metric_sufficiency_audit"] = {
            "helps_explain_success": [
                "task_score",
                "won",
                "episode_steps",
                "parser_validity",
                "action_type_counts",
                "failure_mode",
            ],
            "helps_explain_regression": [
                "parser_validity",
                "runtime_error",
                "trace_completeness",
                "generation_length",
            ],
            "noise_or_secondary": ["gpu_utilization_time_series"],
            "missing_for_h0": [],
            "next_group_metric_changes": "none; preserve the frozen H0 contract",
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tracker.log_artifact_file(
            summary_path,
            name=f"{group}-h0-summary",
            metadata={"protocol_fingerprint": protocol_fingerprint},
        )
        tracker.set_summary(
            {"run/status": "pass" if code == 0 else "failed", **summary_metrics}
        )
        return code
    except BaseException:
        tracker.set_summary({"run/status": "failed"})
        raise
    finally:
        tracker.finish()


if __name__ == "__main__":
    raise SystemExit(main())
