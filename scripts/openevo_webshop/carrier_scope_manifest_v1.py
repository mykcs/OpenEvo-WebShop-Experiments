#!/usr/bin/env python3
"""Deterministic, outcome-blind WebShop task-local scope manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

SCHEMA = "openevo.carrier-scope-manifest.v1"
POLICY_ID = "webshop-task-local-scope-v1"
PRICE_RE = re.compile(r"\bprice\s+lower\s+than\s+(\d+(?:\.\d{1,2})?)\s+dollars\b", re.I)
FIELD_RE = re.compile(
    r"(?:\bwith|,\s*and)\s+([a-z][a-z0-9 _/-]{0,40}?):\s*(.*?)"
    r"(?=,\s*and\s+[a-z][a-z0-9 _/-]{0,40}?:|,\s*and\s+price\s+lower\s+than|$)",
    re.I,
)
FIELD_START_RE = re.compile(r"\bwith\s+[a-z][a-z0-9 _/-]{0,40}?:", re.I)
PRICE_START_RE = re.compile(r",\s*and\s+price\s+lower\s+than\b", re.I)


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(raw)


def request_body(instruction: str) -> str:
    text = normalize(instruction).rstrip(" .")
    return re.sub(r"^find me\s+", "", text, count=1).strip()


def descriptor_span(body: str) -> str | None:
    starts = [m.start() for m in (FIELD_START_RE.search(body), PRICE_START_RE.search(body)) if m]
    end = min(starts) if starts else len(body)
    descriptor = body[:end].strip(" ,")
    return descriptor if len(descriptor.split()) >= 4 else None


def strong_value(value: str) -> bool:
    value = normalize(value)
    return bool(re.search(r"\d|[-_+|#/]", value) or len(value.split()) >= 2)


def price_variants(amount: str) -> list[str]:
    amount = normalize(amount)
    return [amount, f"${amount}", f"{amount} dollars", f"price lower than {amount} dollars"]


def extract_task_atoms(task: dict[str, Any]) -> dict[str, Any]:
    instruction = str(task.get("instruction") or "")
    expected_sha = str(task.get("instruction_sha256") or "")
    actual_sha = sha256_text(instruction)
    if not instruction or actual_sha != expected_sha:
        raise ValueError("scope manifest instruction identity mismatch")
    body = request_body(instruction)
    descriptor = descriptor_span(body)
    fields = []
    for match in FIELD_RE.finditer(body):
        label = normalize(match.group(1))
        value = normalize(match.group(2)).strip(" ,")
        if label and value:
            fields.append({"label": label, "value": value})
    price_match = PRICE_RE.search(body)
    price = normalize(price_match.group(1)) if price_match else None
    product_id = normalize(task.get("asin") or "") or None
    return {
        "source_session_index": int(task["source_session_index"]),
        "instruction_sha256": actual_sha,
        "product_id": product_id,
        "request_body": body,
        "descriptor": descriptor,
        "fields": fields,
        "price": price,
    }


def _dedupe_sorted(values: list[str]) -> list[str]:
    return sorted({normalize(value) for value in values if normalize(value)})


def build_manifest(schedule: dict[str, Any], *, schedule_file_sha256: str, round_index: int) -> dict[str, Any]:
    stage2 = schedule.get("stage2") or {}
    rounds = stage2.get("rounds") or []
    matches = [row for row in rounds if int(row.get("round_index", -1)) == int(round_index)]
    if len(matches) != 1:
        raise ValueError("scope manifest round authority missing or duplicated")
    tasks = [extract_task_atoms(task) for task in (matches[0].get("groups") or [])]
    if not tasks:
        raise ValueError("scope manifest round has no tasks")
    tasks.sort(key=lambda x: (x["source_session_index"], x["instruction_sha256"]))

    forbidden: list[str] = []
    typed_atoms: list[dict[str, Any]] = []
    source_rows = []
    for task in tasks:
        source_rows.append({
            "source_session_index": task["source_session_index"],
            "instruction_sha256": task["instruction_sha256"],
        })
        forbidden.append(task["request_body"])
        if task["descriptor"]:
            forbidden.append(task["descriptor"])
        if task["product_id"]:
            forbidden.append(task["product_id"])
            typed_atoms.append({"kind": "product_id", "label": "product_id", "value": task["product_id"], "source_instruction_sha256": task["instruction_sha256"]})
        for field in task["fields"]:
            forbidden.append(f'{field["label"]}: {field["value"]}')
            typed_atoms.append({"kind": "attribute", "label": field["label"], "value": field["value"], "source_instruction_sha256": task["instruction_sha256"]})
            if strong_value(field["value"]):
                forbidden.append(field["value"])
        if task["price"]:
            forbidden.extend(price_variants(task["price"]))
            typed_atoms.append({"kind": "price", "label": "price", "value": task["price"], "source_instruction_sha256": task["instruction_sha256"]})

    typed_atoms = sorted(
        {json.dumps(atom, sort_keys=True, separators=(",", ":")): atom for atom in typed_atoms}.values(),
        key=lambda x: (x["source_instruction_sha256"], x["kind"], x["label"], x["value"]),
    )
    source_digest = canonical_sha({"tasks": source_rows})
    payload = {
        "schema": SCHEMA,
        "round_index": int(round_index),
        "construction_policy_id": POLICY_ID,
        "schedule_file_sha256": schedule_file_sha256,
        "source_task_count": len(source_rows),
        "source_task_digest_sha256": source_digest,
        "source_tasks": source_rows,
        "forbidden_literals": _dedupe_sorted(forbidden),
        "typed_value_atoms": typed_atoms,
        "combination_threshold": 2,
    }
    payload["manifest_identity_sha256"] = canonical_sha(payload)
    return payload


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--round-index", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    schedule_bytes = args.schedule.read_bytes()
    schedule = json.loads(schedule_bytes)
    manifest = build_manifest(schedule, schedule_file_sha256=hashlib.sha256(schedule_bytes).hexdigest(), round_index=args.round_index)
    write_json(args.output, manifest)
    print(json.dumps({"status": "PASS", "manifest_identity_sha256": manifest["manifest_identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
