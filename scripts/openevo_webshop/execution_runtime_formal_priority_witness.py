#!/usr/bin/env python3
"""CLI for the read-only formal-priority resource witness."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_webshop.formal_priority_witness import build_formal_priority_witness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-repo", type=Path, required=True)
    parser.add_argument("--ready-package", type=Path, required=True)
    parser.add_argument("--owner-launch-release", type=Path, required=True)
    parser.add_argument("--ray-state-root", type=Path, required=True)
    parser.add_argument("--expected-execution-git-sha", required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--candidate-gpu", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    witness = build_formal_priority_witness(
        formal_repo=args.formal_repo,
        ready_path=args.ready_package,
        release_path=args.owner_launch_release,
        ray_state_root=args.ray_state_root,
        expected_execution_git_sha=args.expected_execution_git_sha,
        topology_id=args.topology_id,
        candidate_gpu_indices=args.candidate_gpu,
    )
    text = json.dumps(witness, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
