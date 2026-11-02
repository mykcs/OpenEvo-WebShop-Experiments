#!/usr/bin/env python3
"""Formal execution wrapper v2 for frozen shared Stage1 202609030400.

This file changes no scientific semantics. It closes a module-alias wiring gap:
q25_stage1_seed_native_worker.corrected_run_episode reads sampling through its
own module-global `w`, so formal execution must bind that alias explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

from scripts.openevo_webshop import q25_stage1_seed_native_worker as seed_worker
from scripts.openevo_webshop import shared_stage1_202609030400_worker as frozen

FORMAL_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}


def configure_formal_execution() -> None:
    frozen.configure_shared_runtime()
    seed_worker.w.TEMPERATURE = FORMAL_SAMPLING["temperature"]
    seed_worker.w.TOP_P = FORMAL_SAMPLING["top_p"]
    seed_worker.w.TOP_K = FORMAL_SAMPLING["top_k"]
    seed_worker.w.REPETITION_PENALTY = FORMAL_SAMPLING["repetition_penalty"]
    # The frozen outer worker must point at the same corrected episode function.
    if frozen.w.run_episode is not seed_worker.corrected_run_episode:
        raise RuntimeError("formal shared Stage1 run_episode wiring drift")
    if seed_worker.w.TEMPERATURE != 1.0 or seed_worker.w.TOP_P != 1.0 or seed_worker.w.TOP_K != 0:
        raise RuntimeError("formal shared Stage1 sampling alias binding drift")


def main() -> int:
    configure_formal_execution()
    original = list(sys.argv)
    rc = frozen.w.main()
    out = Path(original[original.index("--output") + 1])
    frozen.postprocess(out)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
