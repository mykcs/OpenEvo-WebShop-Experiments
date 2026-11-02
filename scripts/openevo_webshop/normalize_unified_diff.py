#!/usr/bin/env python3
"""Repair malformed unified-diff hunk line counts without changing patch content.

The reviewed OpenEvo WebShop patch was hand-authored with stale @@ counts. Git
rejects such a file as "corrupt patch" before it even tests source context.
This tool recomputes old/new counts for every hunk while preserving start lines,
context, additions, deletions, filenames, and all patch payload text.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def normalize(src: Path, dst: Path) -> int:
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    i = 0
    changed = 0

    while i < len(lines):
        header = lines[i]
        match = HUNK.match(header.rstrip("\n"))
        if match is None:
            out.append(header)
            i += 1
            continue

        j = i + 1
        while j < len(lines):
            line = lines[j]
            if line.startswith("@@ ") or line.startswith("diff --git "):
                break
            j += 1

        body = lines[i + 1 : j]
        old_count = sum(1 for line in body if line.startswith((" ", "-")) and not line.startswith("--- "))
        new_count = sum(1 for line in body if line.startswith((" ", "+")) and not line.startswith("+++ "))
        old_start, new_start, suffix = match.group(1), match.group(3), match.group(5)
        fixed = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}\n"
        if fixed != header:
            changed += 1
        out.append(fixed)
        out.extend(body)
        i = j

    dst.write_text("".join(out), encoding="utf-8")
    return changed


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.patch OUTPUT.patch", file=sys.stderr)
        return 2
    changed = normalize(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"normalized_hunks={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
