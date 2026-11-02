from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_agents_requires_defined_experiment_to_formal_dispatch() -> None:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "Defined experiment → Formal Start anti-stall rule" in text
    assert "GPU_LAUNCH_READY" in text
    assert "缺少一张新 READY 单据本身不是启动理由或阻塞理由" in text
    assert "launch, do not inspect again" in text
    assert "stalled" in text and "not cautious" in text


def test_reliability_lifecycle_forbids_ready_parking_loop() -> None:
    text = (
        ROOT / "docs/dev/experiment-reliability/LIFECYCLE.md"
    ).read_text(encoding="utf-8")
    assert "READY_FOR_FORMAL" in text
    assert "same-identity preflight again" in text
    assert "no Formal Start and no new blocker" in text
    assert "first task-bearing side effect" in text


def test_delivery_standard_requires_launch_or_typed_blocker() -> None:
    text = (
        ROOT / "docs/dev/experiment-reliability/DELIVERY_STANDARD.md"
    ).read_text(encoding="utf-8")
    assert "GPU_LAUNCH_READY" in text
    assert "FORMAL_START" in text
    assert "new machine-backed typed blocker" in text
    assert "execution stall" in text


def test_critical_rule_precedes_host_and_deep_router_sections() -> None:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    start = text.index("### Defined experiment → Formal Start anti-stall rule")
    assert start < text.index("### Host storage fast path")
    assert start < 800
    assert "CODEX_EIGHT_PR_EXECUTION_STALLS_20261005.md" in text
