#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from wandb_tracking import (
    campaign_notes,
    campaign_tracking_config,
    load_current_campaign,
    short_run_name,
    standard_tags,
    start_wandb_run,
)

SCHEMA = "openevo.ceiling1-measurement-pack-wandb-sync.202609050634"
CANONICAL_OBSERVABILITY_REVISION = "202609041833"
TABLES = (
    "trajectory_spectrum_summary.csv",
    "principal_parameter_themes.csv",
    "parameter_theme_stage_loadings.csv",
    "historical_decomposed_geometry.csv",
    "parameter_spectral_stability.csv",
    "parameter_W_spectral_over_time.csv",
    "parameter_W_rms_stablerank_over_time.csv",
)
ARTIFACTS = (
    "measurement-pack-summary.json",
    "trajectory-spectrum-manifest.json",
    "GENERATION_RECEIPT.json",
    "parameter_spectral_stability.csv",
    "parameter_W_spectral_over_time.csv",
    "parameter_W_rms_stablerank_over_time.csv",
    "trajectory_spectrum_summary.csv",
    "principal_parameter_themes.csv",
    "parameter_theme_stage_loadings.csv",
    "historical_decomposed_geometry.csv",
    "layer_grams.npz",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{number}")
            rows.append(value)
    return rows


def read_csv(path: Path) -> tuple[list[str], list[list[Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return (rows[0], rows[1:]) if rows else ([], [])


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable_run_id(arm: str, stage2_root: Path) -> str:
    digest = hashlib.sha256(str(stage2_root.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"ceiling1-stage2-{arm.replace('_','-')}-{CANONICAL_OBSERVABILITY_REVISION}-{digest}"[:64]


def validate_identity(stage2_root: Path, measurement_root: Path) -> dict[str, Any]:
    summary = read_json(measurement_root / "measurement-pack-summary.json")
    manifest = read_json(measurement_root / "trajectory-spectrum-manifest.json")
    update_count = int(summary["parameter_update_count"])
    source_round = int(summary["source_round"])
    if int(manifest["update_count"]) != update_count or int(manifest["through_round"]) != source_round:
        raise RuntimeError("measurement summary / spectrum manifest identity mismatch")

    state = read_json(stage2_root / "execution-state.json")
    checkpoints = sorted(state.get("sd_checkpoints", []), key=lambda row: int(row["component_count"]))
    current_update_count = max((int(row["component_count"]) for row in checkpoints), default=0)
    if update_count > current_update_count:
        raise RuntimeError("measurement pack refers to a future accepted parameter state")
    checkpoint = next((row for row in checkpoints if int(row["component_count"]) == update_count), None)
    if checkpoint is None:
        raise RuntimeError("measurement pack does not map to an accepted SD checkpoint")
    geometry_path = Path(checkpoint["geometry_path"])
    geometry = read_json(geometry_path)
    if int(geometry["round_index"]) != source_round:
        raise RuntimeError("measurement source round does not match accepted checkpoint")
    if sha256(geometry_path) != summary["source_sha256"]["task_vector_geometry"]:
        raise RuntimeError("measurement task-vector geometry hash drift")
    adapter_path = Path(checkpoint["path"])
    adapter_manifest_path = adapter_path / "openevo_sd_lora_state.json"
    adapter_manifest = read_json(adapter_manifest_path)
    if adapter_manifest["state_weights_sha256"] != summary["adapter_state_sha256"]:
        raise RuntimeError("measurement adapter tensor identity drift")
    if sha256(adapter_manifest_path) != summary["source_sha256"]["adapter_manifest"]:
        raise RuntimeError("measurement adapter manifest hash drift")
    if list(geometry.get("component_factor_sha256", [])) != list(summary["component_factor_sha256"]):
        raise RuntimeError("measurement component-factor identity drift")

    ledger = read_jsonl(stage2_root / "observability" / "stage2-metrics-ledger.jsonl")
    ledger_update_count = int(ledger[-1]["axes"]["parameter_update_count"]) if ledger else None
    ledger_latest_round = int(ledger[-1]["round_index"]) if ledger else None
    return {
        "arm": summary["arm"],
        "parameter_update_count": update_count,
        "source_round": source_round,
        "current_parameter_update_count": current_update_count,
        "checkpoint_path": str(adapter_path),
        "checkpoint_state_sha256": summary["adapter_state_sha256"],
        "geometry_sha256": summary["source_sha256"]["task_vector_geometry"],
        "ledger_update_count": ledger_update_count,
        "ledger_latest_round": ledger_latest_round,
        "ledger_is_stale_relative_to_measurement": ledger_update_count is not None and ledger_update_count < update_count,
    }


def summary_from_spectrum(path: Path) -> dict[str, Any]:
    columns, rows = read_csv(path)
    output: dict[str, Any] = {}
    for row in rows:
        record = dict(zip(columns, row))
        if record.get("group_kind") != "global" or record.get("group") != "all":
            continue
        family = record.get("family") or "unknown"
        for key in (
            "stable_rank", "participation_ratio_energy", "entropy_rank_energy",
            "entropy_rank_singular", "nuclear_effective_rank", "rank90_energy",
            "rank95_energy", "rank99_energy", "top1_energy_share", "top5_energy_share",
            "top10_energy_share", "spectral_gap_s1_s2",
        ):
            raw = record.get(key)
            if raw not in (None, ""):
                output[f"measurement/{family}/{key}"] = float(raw)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("3b", "qwen3-1p7b", "7b"), required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()

    stage2_root = args.stage2_root.resolve()
    measurement_root = args.measurement_root.resolve()
    identity = validate_identity(stage2_root, measurement_root)
    if identity["arm"] != args.arm:
        raise RuntimeError("arm argument does not match measurement pack")

    present = [measurement_root / name for name in ARTIFACTS if (measurement_root / name).is_file()]
    if len(present) != len(ARTIFACTS):
        missing = sorted(set(ARTIFACTS) - {path.name for path in present})
        raise FileNotFoundError(f"measurement pack is incomplete: {missing}")
    source = {path.name: sha256(path) for path in present}
    pack_id = sha256(measurement_root / "GENERATION_RECEIPT.json")[:16]

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
            "measurement_pack_schema": SCHEMA,
        },
        tags=standard_tags(
            campaign,
            method="ceiling1-stage2",
            arm=args.arm,
            extra=(
                "stage2-dynamics", "dual-axis", "canonical", "aligned-points",
                f"revision:{CANONICAL_OBSERVABILITY_REVISION}", "parameter-analysis",
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

    artifact_metadata = {
        **identity,
        "measurement_root": str(measurement_root),
        "measurement_pack_id": pack_id,
        "source_sha256": source,
        "identity_validation": "accepted_checkpoint+task_vector_geometry+adapter_state",
    }
    for path in present:
        tracker.log_artifact_file(
            path,
            name=f"{run_id}-measurement-{path.stem}-{pack_id}",
            artifact_type="stage2-parameter-analysis",
            metadata=artifact_metadata,
        )
    for name in TABLES:
        path = measurement_root / name
        columns, rows = read_csv(path)
        if columns and rows:
            tracker.log_table(f"measurement_tables/{path.stem}", columns=columns, data=rows)

    summary_metrics = summary_from_spectrum(measurement_root / "trajectory_spectrum_summary.csv")
    summary_metrics.update({
        "measurement/parameter_update_count": identity["parameter_update_count"],
        "measurement/source_round": identity["source_round"],
        "measurement/current_parameter_update_count": identity["current_parameter_update_count"],
        "measurement/pack_id": pack_id,
        "measurement/ledger_stale": int(bool(identity["ledger_is_stale_relative_to_measurement"])),
    })
    tracker.set_summary(summary_metrics)
    metadata = tracker.metadata()
    tracker.finish()

    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "arm": args.arm,
        "run_id": run_id,
        "pack_id": pack_id,
        "measurement_root": str(measurement_root),
        "identity": identity,
        "source_sha256": source,
        "summary_metrics": summary_metrics,
        "wandb": metadata,
    }
    atomic_json(measurement_root / "WANDB_SYNC_RECEIPT.json", receipt)
    atomic_json(stage2_root / "observability" / "wandb-measurement-pack-receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
