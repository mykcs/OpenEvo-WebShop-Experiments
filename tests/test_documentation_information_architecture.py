from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docs_index_has_one_canonical_root_set() -> None:
    index = json.loads((ROOT / "docs/DOCS_INDEX.v1.json").read_text(encoding="utf-8"))  # legacy-revision-compat: existing index filename
    assert index["canonical_roots"] == [
        "docs/operations",
        "docs/science",
        "docs/infrastructure",
        "docs/agents",
        "docs/troubleshooting",
        "docs/evidence",
        "docs/dev",
    ]
    assert "docs/experiment-tracking" in index["transitional_roots"]
    assert "docs/agent-handoffs" in index["compatibility_roots"]
    assert index["entrypoints"]["dev"] == "docs/dev/README.md"
    assert index["entrypoints"]["experiment_reliability"] == "docs/dev/experiment-reliability/README.md"
    assert index["topics"]["dev"]["readme"] == "docs/dev/README.md"


def test_root_agent_files_stay_bootstrap_sized() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert len(agents.splitlines()) < 220
    assert len(claude.splitlines()) < 12
    assert "docs/README.md" in agents
    assert "@AGENTS.md" in claude


def test_compatibility_handoff_is_pointer_not_duplicate() -> None:
    alias = ROOT / "docs/agent-handoffs/BOUNDED_GDR_WANDB_PRELAUNCH_HANDOFF_2026-09-15.md"
    canonical = ROOT / "docs/agents/handoffs/BOUNDED_GDR_WANDB_PRELAUNCH_HANDOFF_2026-09-15.md"
    assert alias.is_symlink()
    assert canonical.is_file()
    assert alias.resolve() == canonical.resolve()


def test_compatibility_roots_do_not_accumulate_new_markdown_bodies() -> None:
    index = json.loads((ROOT / "docs/DOCS_INDEX.v1.json").read_text(encoding="utf-8"))  # legacy-revision-compat: existing index filename
    offenders: list[str] = []
    for item in index["compatibility_roots"]:
        root = ROOT / item
        for path in root.rglob("*.md"):
            if not path.is_symlink() and path.name != "README.md":
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []



def test_transitional_root_markdown_bodies_are_explicitly_allowlisted() -> None:
    index = json.loads((ROOT / "docs/DOCS_INDEX.v1.json").read_text(encoding="utf-8"))  # legacy-revision-compat: existing index filename
    root = ROOT / "docs/experiment-tracking"
    actual = sorted(
        str(path.relative_to(ROOT))
        for path in root.glob("*.md")
        if path.name != "README.md" and not path.is_symlink()
    )
    assert actual == index["transitional_markdown_allowlist"]


def test_governance_routes_new_docs_to_canonical_topics() -> None:
    governance = (ROOT / "docs/DOCUMENTATION_GOVERNANCE.md").read_text(encoding="utf-8")
    assert "| Agent handoff | `docs/agents/handoffs/`" in governance
    assert "| CI rule | `docs/infrastructure/ci/`" in governance
    assert "| migration record | `docs/infrastructure/migration/`" in governance
    assert "| development direction / experiment reliability | `docs/dev/`" in governance
    assert "`experiment-tracking/` is **transitional**" in governance


if __name__ == "__main__":
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"DOCUMENTATION_INFORMATION_ARCHITECTURE={len(tests)}/{len(tests)} PASS")
