#!/usr/bin/env python3
"""Freeze the exact shared D8 replay/token/step plans before H1.38B training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.openevo_webshop.ordinary_lora_contract import (
    MatchedOrdinaryLoraConfig,
    build_sample_plan,
    canonical_bytes,
    digest_json,
    example_identity,
)
from scripts.openevo_webshop.ordinary_lora_trainer import _dataset_examples, _load_runtime


def freeze(increment_root: Path, tokenizer_root: Path, output: Path) -> dict:
    runtime = _load_runtime()
    config = MatchedOrdinaryLoraConfig().validate()
    tokenizer = runtime["AutoTokenizer"].from_pretrained(
        str(tokenizer_root),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("H1.38B tokenizer requires an EOS or padding token")
        tokenizer.pad_token = tokenizer.eos_token
    manifests = sorted(increment_root.glob("increment-*/manifest.json"))
    if len(manifests) != 8:
        raise ValueError("H1.38B requires exactly eight increment manifests")
    prior_replay: list[dict] = []
    plans = []
    for increment, manifest_path in enumerate(manifests):
        manifest, current = _dataset_examples(
            manifest_path.resolve(), f"h138b-increment-{increment:02d}", runtime
        )
        original_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        try:
            plan, next_replay, _ = build_sample_plan(
                current_examples=current,
                prior_replay_buffer=prior_replay,
                increment_index=increment,
                tokenizer=tokenizer,
                official=runtime["official"],
                config=config,
            )
        finally:
            tokenizer.truncation_side = original_side
        next_replay_bytes = b"".join(canonical_bytes(item) + b"\n" for item in next_replay)
        plans.append(
            {
                "increment_index": increment,
                "training_plan": plan,
                "current_examples_sha256": digest_json([example_identity(item) for item in current]),
                "current_dataset_records_sha256": manifest["records_sha256"],
                "next_replay_buffer_file_sha256": hashlib.sha256(next_replay_bytes).hexdigest(),
            }
        )
        prior_replay = next_replay
    document = {
        "schema": "openevo.webshop-h1.38b-frozen-sample-plans.v1",
        "official_open_evo_sha": "796e6248a61112a61c880885844d3b0420c0b6ae",
        "official_helper_contract": runtime["helper_receipt"],
        "ordinary_config": config.as_dict(),
        "increments": plans,
        "bundle_sha256": "",
    }
    document["bundle_sha256"] = digest_json(
        {key: value for key, value in document.items() if key != "bundle_sha256"}
    )
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--increment-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = freeze(
        args.increment_root.resolve(), args.tokenizer_root.resolve(), args.output.resolve()
    )
    print(json.dumps({"bundle_sha256": document["bundle_sha256"], "increments": len(document["increments"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
