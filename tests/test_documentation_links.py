from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "#",
    "/",
    "data:",
    "javascript:",
)


def test_canonical_markdown_relative_links_resolve() -> None:
    errors: list[str] = []

    for path in sorted((ROOT / "docs").rglob("*.md")):
        relative = path.relative_to(ROOT)
        # Compatibility paths are symlinks. Claim-bearing evidence is immutable
        # and may intentionally preserve historical path references.
        if path.is_symlink() or relative.parts[:2] == ("docs", "evidence"):
            continue
        text = path.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            raw_target = match.group(1).strip()
            if raw_target.startswith("<") and ">" in raw_target:
                raw_target = raw_target[1 : raw_target.index(">")]
            elif raw_target:
                raw_target = raw_target.split()[0]

            if not raw_target or raw_target.startswith(EXTERNAL_PREFIXES):
                continue

            target = unquote(raw_target.split("#", 1)[0].split("?", 1)[0])
            if target and not (path.parent / target).exists():
                errors.append(f"{relative}: {raw_target}")

    assert not errors, (
        "broken canonical relative Markdown links:\n" + "\n".join(errors)
    )
