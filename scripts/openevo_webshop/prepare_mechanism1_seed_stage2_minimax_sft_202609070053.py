#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_BASELINE_SHA = "843ac0b441c84884df83203adb66d4203330dcc48536a64b58619f99e37ee308"
EXPECTED_MINIMAX_ROWS = 1440
EXPECTED_ANALYZER = "MiniMax-M3"
PUBLIC_SEED_COMMIT = "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def candidate_from_pair(base: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    key = f"{base['task_id']}:{base['rollout_id']}"
    if analysis.get("trajectory_id") != key:
        raise ValueError(f"trajectory identity mismatch: {key}")
    if analysis.get("trajectory_sha256") != canonical_sha(base):
        raise ValueError(f"trajectory SHA mismatch: {key}")
    if analysis.get("parse_ok") is not True:
        raise ValueError(f"MiniMax parse not sealed: {key}")
    if analysis.get("analyzer_provider") != "MiniMax" or analysis.get("analyzer_model") != EXPECTED_ANALYZER:
        raise ValueError(f"analyzer identity drift: {key}")
    if analysis.get("analysis_prompt_version") != "seed" or analysis.get("skill_mode") != "episode_only":
        raise ValueError(f"analysis prompt/skill drift: {key}")
    if analysis.get("thinking_type") != "adaptive" or analysis.get("reasoning_split") is not True:
        raise ValueError(f"reasoning contract drift: {key}")
    raw_outputs = list(analysis.get("llm_raw_outputs") or [])
    return {
        "skill_id": key,
        "task_id": base["task_id"],
        "task_type": base.get("task_type", "webshop"),
        "goal_idx": int(base.get("goal_idx", -1)),
        "source_rollout_id": int(base["rollout_id"]),
        "source_success": bool(base.get("success", False)),
        "source_num_steps": int(base.get("num_steps", 0)),
        "source_final_task_score": float(base.get("final_task_score", 0.0)),
        "task_description": base.get("task_description", ""),
        "analysis_prompt": analysis.get("analysis_prompt", {}),
        "llm_raw_output": raw_outputs[-1] if raw_outputs else "",
        "llm_raw_outputs": raw_outputs,
        "episode_summary": str(analysis.get("episode_summary", "")),
        "episode_skill": str(analysis.get("episode_skill", "")),
        "parse_ok": True,
        "parse_attempts": int(analysis.get("generation_calls", 1)),
        "max_parse_attempts": 2,
        "analysis_error": None,
        "source_trajectory_sha256": analysis["trajectory_sha256"],
        "source_minimax_model": EXPECTED_ANALYZER,
    }


def main() -> int:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--minimax", type=Path, required=True)
    p.add_argument("--seed-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    if sha256(args.baseline) != EXPECTED_BASELINE_SHA:
        raise SystemExit("baseline corpus SHA drift")
    if not (args.seed_root / ".git").exists():
        raise SystemExit("exact SEED source Git identity unavailable")
    import subprocess
    commit = subprocess.check_output(["git", "-C", str(args.seed_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != PUBLIC_SEED_COMMIT:
        raise SystemExit(f"SEED source drift: {commit}")

    baseline = read_jsonl(args.baseline)
    analyses = read_jsonl(args.minimax)
    if len(baseline) != EXPECTED_MINIMAX_ROWS or len(analyses) != EXPECTED_MINIMAX_ROWS:
        raise SystemExit(f"row count mismatch baseline={len(baseline)} minimax={len(analyses)}")
    by_id = {str(row.get("trajectory_id", "")): row for row in analyses}
    if len(by_id) != EXPECTED_MINIMAX_ROWS:
        raise SystemExit("MiniMax trajectory IDs are not unique")

    candidates = []
    for base in baseline:
        key = f"{base['task_id']}:{base['rollout_id']}"
        if key not in by_id:
            raise SystemExit(f"MiniMax row missing: {key}")
        candidates.append(candidate_from_pair(base, by_id[key]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_skills.jsonl"
    write_jsonl(candidate_path, candidates)

    sys.path.insert(0, str(args.seed_root))
    from scripts.sft.webshop import pipeline as upstream
    sft_records = upstream.build_sft_exports_from_candidates(
        candidate_skills=candidates,
        output_dir=args.output_dir,
        sft_val_ratio=0.1,
        seed=2026,
        sft_min_source_score=0.0,
        sft_include_success=True,
        sft_max_zero_score_failures=None,
        sft_max_records=None,
        include_episode_summary=True,
    )
    train_path = args.output_dir / "sft_episode_skill_train.parquet"
    val_path = args.output_dir / "sft_episode_skill_val.parquet"
    if len(sft_records) != 1440 or not train_path.is_file() or not val_path.is_file():
        raise SystemExit("public SFT export did not seal 1,440 records")

    import pandas as pd
    train_rows = len(pd.read_parquet(train_path))
    val_rows = len(pd.read_parquet(val_path))
    if (train_rows, val_rows) != (1296, 144):
        raise SystemExit(f"SFT split drift: {(train_rows, val_rows)}")

    receipt = {
        "schema": "openevo.mechanism1-seed-stage2-minimax-sft-data.202609070053",
        "status": "PASS",
        "baseline_sha256": sha256(args.baseline),
        "minimax_sha256": sha256(args.minimax),
        "candidate_skills_sha256": sha256(candidate_path),
        "sft_all_sha256": sha256(args.output_dir / "sft_episode_skill_all.jsonl"),
        "train_parquet_sha256": sha256(train_path),
        "validation_parquet_sha256": sha256(val_path),
        "candidate_rows": 1440,
        "sft_records": 1440,
        "train_rows": 1296,
        "validation_rows": 144,
        "teacher_provider": "MiniMax",
        "teacher_model": EXPECTED_ANALYZER,
        "public_seed_source_commit": PUBLIC_SEED_COMMIT,
        "new_teacher_calls": 0,
        "new_webshop_calls": 0,
    }
    receipt_path = args.output_dir / "M1D_STAGE2_SFT_DATA_READY_202609070053.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
