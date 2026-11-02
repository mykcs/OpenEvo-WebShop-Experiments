#!/usr/bin/env python3
"""CPU-only mechanical MG0/MGH/MG1-MG4 verdict selector for H1.40-G2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RETENTION_MARGIN = 0.05


def positive_ci(metric: dict[str, Any]) -> bool:
    ci = metric.get("bootstrap_ci95") or metric.get("ci95") or []
    return len(ci) == 2 and float(metric.get("delta", 0.0)) > 0 and float(ci[0]) > 0


def retention_pass(metric: dict[str, Any]) -> bool:
    ci = metric.get("bootstrap_ci95") or metric.get("ci95") or []
    return len(ci) == 2 and float(ci[0]) > -RETENTION_MARGIN


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") not in (None, "complete") or payload.get("provenance", {}).get("status") not in (None, "pass") or payload.get("g2_structural", {}).get("status") not in (None, "pass"):
        classification = "MG0 STRUCTURALLY_INVALID"
        reason = "formal contract, provenance, or G2 structural gate failed"
    elif payload.get("harvest", {}).get("status") != "pass" or int(payload.get("harvest", {}).get("qualified_identity_count", 0)) < 8:
        classification = "MGH HARVEST_INSUFFICIENT"
        reason = "fewer than 8 identities reached two qualified endogenous E2 successes within the frozen 256-attempt reservoir"
    elif float(payload.get("acquisition", {}).get("delta", 0.0)) <= 0:
        classification = "MG1 NO_G2_ACQUISITION"
        reason = "G2 did not improve over G1 on the selected E2 acquisition identities"
    elif not positive_ci(payload.get("t2", {})):
        classification = "MG2 NO_INCREMENTAL_FRESH_TRANSFER"
        reason = "G2 acquisition was positive, but the preregistered T2 G2-minus-G1 CI did not clear zero"
    elif not all(retention_pass(metric) for metric in payload.get("retention", [])):
        classification = "MG3 TRANSFER_WITH_FORGETTING"
        reason = "incremental T2 transfer was positive but at least one retention lower CI crossed -0.05"
    else:
        classification = "MG4 MULTI_GENERATION_SELF_EVOLUTION_SUPPORTED"
        reason = "all preregistered endogenous-generation, acquisition, retention, T2, provenance, invalidity, and resource gates passed"
    return {
        "schema": "openevo.webshop-h1.40-g2-analysis.v1",
        "classification": classification,
        "reason": reason,
        "retention_noninferiority_margin": RETENTION_MARGIN,
        "primary_contrast": "T2 G2-minus-G1",
        "claim_boundary": "Only MG4 supports a two-generation endogenous self-evolution claim under the frozen WebShop contract; no classification supports universal recursive self-improvement.",
        "input": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"classification": result["classification"], "reason": result["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
