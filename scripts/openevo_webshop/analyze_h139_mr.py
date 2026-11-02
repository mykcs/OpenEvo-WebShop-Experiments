#!/usr/bin/env python3
"""Preregistered H1.39-MR task-cluster statistics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


CONTRASTS = {
    "sd_minus_base": ("sd-lora", "base"),
    "ordinary_minus_base": ("ordinary-lora", "base"),
    "sd_minus_ordinary": ("sd-lora", "ordinary-lora"),
}
BOOTSTRAP_ROUNDS = 10_000
SIGNFLIP_ROUNDS = 10_000


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap(values: list[float], seed: int) -> list[float]:
    if not values:
        return []
    rng = random.Random(seed)
    n = len(values)
    return [_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(BOOTSTRAP_ROUNDS)]


def _sign_flip(values: list[float], seed: int) -> float:
    if not values:
        return 1.0
    observed = abs(_mean(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(SIGNFLIP_ROUNDS):
        if abs(_mean([value if rng.getrandbits(1) else -value for value in values])) >= observed:
            extreme += 1
    return (extreme + 1) / (SIGNFLIP_ROUNDS + 1)


def _classify(reconciliation: dict[str, Any], comparisons: dict[str, Any]) -> tuple[str, list[str]]:
    if reconciliation.get("status") != "complete" or reconciliation.get("structural_errors"):
        return "R0 STRUCTURALLY_INVALID", []

    sd_ci = comparisons["sd_minus_base"]["bootstrap_ci95"]
    ordinary_ci = comparisons["ordinary_minus_base"]["bootstrap_ci95"]
    method_ci = comparisons["sd_minus_ordinary"]["bootstrap_ci95"]
    sd_confirmed = sd_ci[0] > 0
    ordinary_confirmed = ordinary_ci[0] > 0
    method_difference = method_ci[0] > 0 or method_ci[1] < 0

    if sd_confirmed and ordinary_confirmed:
        primary = "R4 BOTH_REPLICATE"
    elif sd_confirmed:
        primary = "R1 REPLICATED_POSITIVE_SD"
    elif ordinary_confirmed:
        primary = "R3 REPLICATED_ORDINARY_ONLY"
    else:
        primary = "R2 POSITIVE_NOT_REPLICATED"
    secondary = ["R5 CONFIRMED_METHOD_DIFFERENCE"] if method_difference else []
    return primary, secondary


def analyze(reconciliation_path: Path, output: Path) -> dict[str, Any]:
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    complete = reconciliation.get("complete_valid_tasks") or []
    comparisons: dict[str, Any] = {}
    for index, (name, (left, right)) in enumerate(CONTRASTS.items()):
        deltas = [float(row["means"][left]) - float(row["means"][right]) for row in complete]
        boot = sorted(_bootstrap(deltas, 2026082039 + index))
        ci = [boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]] if boot else [0.0, 0.0]
        comparisons[name] = {
            "left_arm": left,
            "right_arm": right,
            "complete_valid_tasks": len(deltas),
            "task_weighted_delta": _mean(deltas),
            "itt_delta": float(reconciliation["arm_metrics"][left]["itt_mean"])
            - float(reconciliation["arm_metrics"][right]["itt_mean"]),
            "bootstrap_ci95": ci,
            "sign_flip_p": _sign_flip(deltas, 2026082040 + index),
            "positive_tie_negative": [sum(value > 0 for value in deltas), sum(value == 0 for value in deltas), sum(value < 0 for value in deltas)],
        }
    classification, secondary_classifications = _classify(reconciliation, comparisons)
    result = {
        "schema": "openevo.webshop-h1.39-mr-analysis.v1",
        "reconciliation_sha256": __import__("hashlib").sha256(reconciliation_path.read_bytes()).hexdigest(),
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "sign_flip_rounds": SIGNFLIP_ROUNDS,
        "arm_metrics": reconciliation.get("arm_metrics", {}),
        "comparisons": comparisons,
        "complete_valid_task_count": len(complete),
        "classification": classification,
        "secondary_classifications": secondary_classifications,
        "claim_boundary": (
            "H1.39-MR independently supports positive fresh-task replication of both frozen "
            "H1.38B ordinary-LoRA and SD-LoRA versus BASE on the committed task-ID-disjoint "
            "panel; the SD-versus-ordinary difference is reported separately and does not "
            "establish universal transfer, upstream claims, or multi-generation self-evolution."
        ),
        "invalid_taxonomy": reconciliation.get("invalid_taxonomy", []),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.reconciliation.resolve(), args.output.resolve())
    print(json.dumps({"classification": result["classification"], "comparisons": result["comparisons"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
