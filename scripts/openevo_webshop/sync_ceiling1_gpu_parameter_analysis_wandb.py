#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sync_ceiling1_parameter_measurement_pack_wandb import (
    CANONICAL_OBSERVABILITY_REVISION,
    read_json,
    sha256,
    stable_run_id,
    validate_identity,
)
from wandb_tracking import (
    campaign_notes,
    campaign_tracking_config,
    load_current_campaign,
    short_run_name,
    standard_tags,
    start_wandb_run,
)

SCHEMA = "openevo.ceiling1-gpu-parameter-analysis-wandb-sync.202609050634"
GPU_FILES = (
    "GPU_ANALYSIS_SUMMARY.json",
    "GPU_ANALYSIS_RECEIPT.json",
    "gpu_hidden_state_drift.csv",
    "gpu_theme_ablation.csv",
    "gpu_theme_sensitivity_curvature.csv",
    "gpu_layer_projection_ablation.csv",
)
GPU_TABLES = tuple(name for name in GPU_FILES if name.endswith(".csv"))


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return (rows[0], rows[1:]) if rows else ([], [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--gpu-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("3b", "qwen3-1p7b", "7b"), required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()

    stage2_root = args.stage2_root.resolve()
    measurement_root = args.measurement_root.resolve()
    gpu_root = args.gpu_root.resolve()
    identity = validate_identity(stage2_root, measurement_root)
    if identity["arm"] != args.arm:
        raise RuntimeError("arm argument does not match measurement identity")
    cpu_pack_sha = sha256(measurement_root / "GENERATION_RECEIPT.json")
    gpu_receipt = read_json(gpu_root / "GPU_ANALYSIS_RECEIPT.json")
    gpu_summary = read_json(gpu_root / "GPU_ANALYSIS_SUMMARY.json")
    if gpu_receipt.get("status") != "PASS" or gpu_summary.get("status") != "PASS":
        raise RuntimeError("GPU analysis is not PASS")
    if gpu_receipt.get("measurement_pack_sha256") != cpu_pack_sha:
        raise RuntimeError("GPU analysis / CPU measurement pack identity mismatch")
    if int(gpu_receipt["parameter_update_count"]) != int(identity["parameter_update_count"]):
        raise RuntimeError("GPU analysis parameter-update identity mismatch")
    if int(gpu_receipt["source_round"]) != int(identity["source_round"]):
        raise RuntimeError("GPU analysis source-round identity mismatch")
    present = [gpu_root / name for name in GPU_FILES if (gpu_root / name).is_file()]
    if len(present) != len(GPU_FILES):
        missing = sorted(set(GPU_FILES) - {path.name for path in present})
        raise FileNotFoundError(f"GPU analysis pack is incomplete: {missing}")
    source = {path.name: sha256(path) for path in present}
    analysis_id = sha256(gpu_root / "GPU_ANALYSIS_RECEIPT.json")[:16]

    campaign = load_current_campaign()
    tracking = campaign.get("tracking", {}) if campaign else {}
    project = tracking.get("project") or "openevo-webshop-experiments"
    entity = tracking.get("entity")
    base_group = tracking.get("group") or campaign_tracking_config(campaign).get("campaign_id")
    group = f"{base_group}-stage2-canonical-{CANONICAL_OBSERVABILITY_REVISION}" if base_group else None
    run_id = stable_run_id(args.arm, stage2_root)
    tracker = start_wandb_run(
        mode=args.wandb_mode,
        project=project,
        entity=entity,
        name=short_run_name(arm=args.arm, unit="stage2-dynamics-canonical"),
        job_type="stage2-observability",
        config={
            **campaign_tracking_config(campaign),
            "arm": args.arm,
            "stage2_root": str(stage2_root),
            "gpu_parameter_analysis_schema": SCHEMA,
        },
        tags=standard_tags(
            campaign,
            method="ceiling1-stage2",
            arm=args.arm,
            extra=(
                "stage2-dynamics", "canonical", f"revision:{CANONICAL_OBSERVABILITY_REVISION}",
                "parameter-analysis", "gpu-mechanism-analysis",
            ),
        ),
        directory=stage2_root / "observability" / "wandb",
        group=group,
        notes=campaign_notes(campaign),
        run_id=run_id,
        resume="allow",
    )
    if tracker.run is None:
        raise RuntimeError(f"W&B tracker unavailable: {tracker.disabled_reason}")

    metadata = {
        **identity,
        "measurement_root": str(measurement_root),
        "gpu_root": str(gpu_root),
        "gpu_analysis_id": analysis_id,
        "measurement_pack_sha256": cpu_pack_sha,
        "source_sha256": source,
        "gpu_identity": gpu_receipt.get("gpu_identity"),
        "replay_sha256": gpu_receipt.get("replay_sha256"),
        "posthoc_only": True,
        "final_panel_access": False,
    }
    for path in present:
        tracker.log_artifact_file(
            path,
            name=f"{run_id}-gpu-parameter-{path.stem}-{analysis_id}",
            artifact_type="stage2-parameter-analysis",
            metadata=metadata,
        )
    for name in GPU_TABLES:
        path = gpu_root / name
        columns, rows = read_csv(path)
        if columns and rows:
            tracker.log_table(f"measurement_tables/{path.stem}", columns=columns, data=rows)

    summary = {
        "measurement/gpu_analysis_status": "PASS",
        "measurement/gpu_analysis_id": analysis_id,
        "measurement/gpu_analysis_parameter_update_count": identity["parameter_update_count"],
        "measurement/gpu_analysis_source_round": identity["source_round"],
        "measurement/gpu_analysis_sample_count": int(gpu_summary["sample_count"]),
    }
    theme_rows = gpu_summary.get("theme_ablation") or []
    if theme_rows:
        best = max(theme_rows, key=lambda row: abs(float(row.get("delta_nll_vs_full") or 0.0)))
        summary["measurement/gpu_largest_theme_ablation_delta_nll"] = float(best.get("delta_nll_vs_full") or 0.0)
        summary["measurement/gpu_largest_theme_ablation_config"] = str(best.get("configuration"))
    tracker.set_summary(summary)
    wandb_metadata = tracker.metadata()
    tracker.finish()

    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "arm": args.arm,
        "run_id": run_id,
        "analysis_id": analysis_id,
        "identity": identity,
        "measurement_root": str(measurement_root),
        "gpu_root": str(gpu_root),
        "source_sha256": source,
        "summary_metrics": summary,
        "wandb": wandb_metadata,
    }
    output = gpu_root / "WANDB_GPU_ANALYSIS_SYNC_RECEIPT.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
