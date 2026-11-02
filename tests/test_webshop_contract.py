from scripts.openevo_webshop.webshop_comparison_contract import (
    ACTION_COMMAND_CONTRACT,
    COMPACT_PROMPT_CONTRACT,
    STATE_AWARE_PROMPT_CONTRACT,
    TOLERANT_ACTION_CONTRACT,
    format_webshop_actions,
    parse_webshop_action,
    render_compact_webshop_prompt,
    render_state_aware_webshop_prompt,
    render_webshop_prompt,
    repair_action_command,
    repair_missing_action_close,
)
from scripts.openevo_webshop.webshop_comparison_contract import seed_project_action
from scripts.openevo_webshop.parametric_rnd import ParserSafeRetryBackend


def _load_primary_v2_raw_completion() -> str:
    """Load a real malformed step from the immutable primary-v2 evidence archive.

    The episode file is treated as read-only historical evidence, not as a test
    fixture that can be edited to make the parser pass.
    """
    import json
    import pathlib

    path = (
        pathlib.Path(__file__).parent.parent
        / "docs/evidence/seed-official-heldout-comparison-v1/episodes/primary/base/episodes"
        / "webshop_seed0_goal00444-2026082512.json"
    )
    episode = json.loads(path.read_text(encoding="utf-8"))
    return str(episode["episode"]["trace"][0]["raw_completion"])


def test_runtime_validator_requires_expected_manifest_hashes():
    """The source-faithful runtime validator must have externally frozen manifest
    identity hashes; missing or wrong hashes must block validation before any
    runtime reconstruction begins.
    """
    import argparse
    import hashlib
    import json
    import pathlib

    import pytest

    from scripts.openevo_webshop import validate_seed_source_faithful_runtime as validator

    repo = pathlib.Path(__file__).parent.parent
    manifest_path = (
        repo
        / "configs/experiment/manifests/webshop-seed-source-faithful-reproduction-v1-panel-v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    real_file_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    real_content_sha = manifest["manifest_content_sha256"]
    wrong_sha = "0" * 64

    def make_args(file_sha, content_sha):
        return argparse.Namespace(
            source_root=repo,
            webshop_data=repo,
            index_root=repo,
            manifest=manifest_path,
            experiment_root=repo,
            expected_manifest_file_sha256=file_sha,
            expected_manifest_content_sha256=content_sha,
            output=pathlib.Path("/tmp/dummy_receipt.json"),
            created_utc="2026-08-26T10:20:00Z",
        )

    # Missing file hash -> BLOCK.
    with pytest.raises(SystemExit) as excinfo:
        validator.validate(make_args(None, real_content_sha))
    assert "expected-manifest-file-sha256" in str(excinfo.value)

    # Missing content hash -> BLOCK.
    with pytest.raises(SystemExit) as excinfo:
        validator.validate(make_args(real_file_sha, None))
    assert "expected-manifest-content-sha256" in str(excinfo.value)

    # Wrong file hash -> BLOCK (after source identity gate is stubbed).
    orig_verify = validator._verify_source_tree
    stub_identity = {
        "expected_commit": validator.PINNED_SEED_COMMIT,
        "observed_commit": validator.PINNED_SEED_COMMIT,
        "worktree_clean": True,
    }
    validator._verify_source_tree = lambda path: stub_identity
    try:
        # Wrong file hash.
        with pytest.raises(SystemExit) as excinfo:
            validator.validate(make_args(wrong_sha, real_content_sha))
        assert "file SHA-256 mismatch" in str(excinfo.value)

        # Wrong content hash.
        with pytest.raises(SystemExit) as excinfo:
            validator.validate(make_args(real_file_sha, wrong_sha))
        assert "content SHA-256 mismatch" in str(excinfo.value)
    finally:
        validator._verify_source_tree = orig_verify


def test_extract_instruction_text_regression():
    """Regression: the real WebAgentTextEnv.get_instruction_text() returns an
    HTML-rendered prefix ("Instruction: ...") that must be stripped before
    comparing against the manifest's raw goal['instruction_text']."""
    import pytest

    from scripts.openevo_webshop import validate_seed_source_faithful_runtime as validator

    assert validator._extract_instruction_text("Instruction: buy a red shoe") == "buy a red shoe"
    with pytest.raises(ValueError, match="does not start with expected prefix"):
        validator._extract_instruction_text("buy a red shoe")
    with pytest.raises(ValueError, match="does not start with expected prefix"):
        validator._extract_instruction_text("")


def test_action_parser_requires_think_and_action():
    action, valid = parse_webshop_action(
        "<think>search first</think><action>search[grey gym shorts]</action>"
    )
    assert valid
    assert action == "search[grey gym shorts]"


def test_action_parser_rejects_missing_think():
    _, valid = parse_webshop_action("<action>search[grey gym shorts]</action>")
    assert not valid


def test_tolerant_contract_repairs_only_missing_closing_action_tag():
    raw = "<think>click the matching item</think>\n<action>click[description]"
    repaired, action, changed = repair_missing_action_close(raw)
    assert TOLERANT_ACTION_CONTRACT == "seed3090.webshop-action-parser.v2-missing-close"
    assert changed
    assert repaired.endswith("</action>")
    assert action == "click[description]"


def test_tolerant_contract_does_not_accept_unbounded_text():
    repaired, _, changed = repair_missing_action_close(
        "<think>reason</think>\n<action>click[description]\nextra"
    )
    assert repaired.endswith("extra")
    assert not changed


def test_retry_backend_records_tolerant_repair_without_retry():
    class Backend:
        model_name = "test"

        def generate(self, *args, **kwargs):
            return "<think>reason</think>\n<action>click[description]"

    backend = ParserSafeRetryBackend(Backend(), action_contract=TOLERANT_ACTION_CONTRACT)
    result = backend.generate("prompt", generation_seed=7)
    assert result.endswith("</action>")
    assert [event["event_type"] for event in backend.events] == ["missing-closing-action-tag"]


def test_action_command_contract_recovers_wrapper_variants():
    assert ACTION_COMMAND_CONTRACT == "seed3090.webshop-action-parser.v3-action-command"
    think = "<think>find a red shoe under $40</think>"
    for raw in (
        f"{think}\n[action]search[red shoe]",
        f"{think}\n[action]search[red shoe]</action>",
        f"{think}\n=search[red shoe price:<40]",
        f"{think}\n<click[b07xyz - red shoe]>",
    ):
        repaired, action, changed = repair_action_command(raw)
        assert changed, raw
        _, valid = parse_webshop_action(repaired, require_think=True)
        assert valid, raw
        assert repaired.rstrip().endswith("</action>")


def test_action_command_contract_recovers_primary_v2_bracket_action_wrapper():
    """Historical primary-v2/repaired-PRIMARY episodes proved the frozen 7B policy
    can emit malformed wrappers like ``[action>search[...]</action>``. The strict
    SEED parser must reject them; the v3 action-command parser must recover the
    exact action command.
    """
    think = "<think>find a red shoe under $40</think>"
    cases = [
        ("search[red shoe under $40]", f"{think}\n[action>search[red shoe under $40]</action>"),
        ("click[b07xyz - red shoe]", f"{think}\n[action>click[b07xyz - red shoe] </action>"),
    ]
    for expected_action, raw in cases:
        strict_action, strict_valid = seed_project_action(raw)
        assert not strict_valid, f"strict parser should reject {raw!r}"
        assert strict_action != expected_action

        repaired, action, changed = repair_action_command(raw)
        assert changed, raw
        _, valid = parse_webshop_action(repaired, require_think=True)
        assert valid, f"repaired text should re-parse strictly: {repaired!r}"
        assert action == expected_action, f"expected {expected_action!r}, got {action!r}"


def test_action_command_contract_recovers_real_primary_v2_raw_completion():
    """A real primary-v2 BASE step (goal 444, step 0) emitted ``[action] search[...]``
    instead of the strict ``<action>...</action>`` wrapper. Strict SEED parser must
    mark it invalid; v3 action-command parser must recover the exact search command.
    """
    raw = _load_primary_v2_raw_completion()
    strict_action, strict_valid = seed_project_action(raw)
    assert not strict_valid, "strict SEED parser must reject the real primary-v2 wrapper"

    repaired, action, changed = repair_action_command(raw)
    assert changed, "v3 parser should repair the real primary-v2 wrapper"
    _, valid = parse_webshop_action(repaired, require_think=True)
    assert valid, f"repaired text should re-parse strictly: {repaired!r}"
    expected = "search[loose fit women's tops tees & blouses size x-large short sleeve long sleeve color:c5-blue price:<30.00]"
    assert action == expected, f"expected {expected!r}, got {action!r}"


def test_runtime_validator_requires_expected_manifest_hashes():
    """The source-faithful runtime validator must have externally frozen manifest
    identity hashes; missing or wrong hashes must block validation before any
    runtime reconstruction begins.
    """
    import argparse
    import hashlib
    import json
    import pathlib

    import pytest

    from scripts.openevo_webshop import validate_seed_source_faithful_runtime as validator

    repo = pathlib.Path(__file__).parent.parent
    manifest_path = (
        repo
        / "configs/experiment/manifests/webshop-seed-source-faithful-reproduction-v1-panel-v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    real_file_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    real_content_sha = manifest["manifest_content_sha256"]
    wrong_sha = "0" * 64

    def make_args(file_sha: str | None, content_sha: str | None) -> argparse.Namespace:
        return argparse.Namespace(
            source_root=repo,
            webshop_data=repo,
            index_root=repo,
            manifest=manifest_path,
            experiment_root=repo,
            expected_manifest_file_sha256=file_sha,
            expected_manifest_content_sha256=content_sha,
            output=pathlib.Path("/tmp/dummy_receipt.json"),
            created_utc="2026-08-26T10:20:00Z",
        )

    # Missing file hash -> BLOCK.
    with pytest.raises(SystemExit) as excinfo:
        validator.validate(make_args(None, real_content_sha))
    assert "expected-manifest-file-sha256" in str(excinfo.value)

    # Missing content hash -> BLOCK.
    with pytest.raises(SystemExit) as excinfo:
        validator.validate(make_args(real_file_sha, None))
    assert "expected-manifest-content-sha256" in str(excinfo.value)

    # Wrong file hash -> BLOCK (after source identity gate is stubbed and data files are satisfied).
    import tempfile

    orig_verify = validator._verify_source_tree
    stub_identity = {
        "expected_commit": validator.PINNED_SEED_COMMIT,
        "observed_commit": validator.PINNED_SEED_COMMIT,
        "worktree_clean": True,
    }

    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        source_root = td_path / "seed_src"
        source_root.mkdir()
        (source_root / ".git").mkdir()
        (source_root / "seed").mkdir()
        (source_root / "seed" / "analysis.py").write_text("# dummy")
        data_root = td_path / "data"
        data_root.mkdir()
        for name, expected_hash in manifest["benchmark"]["data_files"].items():
            (data_root / name).write_text("dummy")
        index_root = td_path / "index"
        index_root.mkdir()

        def make_args_with_paths(file_sha, content_sha):
            return argparse.Namespace(
                source_root=source_root,
                webshop_data=data_root,
                index_root=index_root,
                manifest=manifest_path,
                experiment_root=repo,
                expected_manifest_file_sha256=file_sha,
                expected_manifest_content_sha256=content_sha,
                output=td_path / "dummy_receipt.json",
                created_utc="2026-08-26T10:20:00Z",
            )

        validator._verify_source_tree = lambda path: stub_identity
        try:
            # Wrong file hash.
            with pytest.raises(SystemExit) as excinfo:
                validator.validate(make_args_with_paths(wrong_sha, real_content_sha))
            assert "file SHA-256 mismatch" in str(excinfo.value)

            # Wrong content hash.
            with pytest.raises(SystemExit) as excinfo:
                validator.validate(make_args_with_paths(real_file_sha, wrong_sha))
            assert "content SHA-256 mismatch" in str(excinfo.value)
        finally:
            validator._verify_source_tree = orig_verify


def test_action_command_contract_leaves_strict_and_unrecoverable_alone():
    class Backend:
        model_name = "test"

        def generate(self, *args, **kwargs):
            return "<think>reason</think>\n[action]click[item - B012]"

    backend = ParserSafeRetryBackend(Backend(), action_contract=ACTION_COMMAND_CONTRACT)
    result = backend.generate("prompt", generation_seed=7)
    assert result.rstrip().endswith("</action>")
    assert parse_webshop_action(result, require_think=True)[1]
    assert [event["event_type"] for event in backend.events] == ["action-command-recovery"]


def test_available_action_contract():
    actions = format_webshop_actions(
        {"has_search_bar": True, "clickables": ["item - B012", "Buy Now"]}
    )
    assert actions == [
        "search[<your query>]",
        "click[item - B012]",
        "click[Buy Now]",
    ]


def test_first_step_prompt_has_no_fake_history():
    prompt = render_webshop_prompt(
        task_description="buy grey gym shorts",
        current_observation="Search",
        available_actions={"has_search_bar": True, "clickables": []},
        step_count=0,
        history_length=0,
    )
    assert "buy grey gym shorts" in prompt
    assert "Prior to this step" not in prompt
    assert "search[<your query>]" in prompt


def test_compact_prompt_has_frozen_output_instruction():
    prompt = render_compact_webshop_prompt(
        task_description="buy grey gym shorts",
        current_observation="Search",
        available_actions={"has_search_bar": True, "clickables": []},
    )
    assert COMPACT_PROMPT_CONTRACT == "seed3090.webshop-compact-action-prompt.v2"
    assert "exactly two tagged lines" in prompt
    assert "<think>brief reason</think>" in prompt
    assert "<action>one admissible action</action>" in prompt
    assert "search[<your query>]" in prompt


def test_state_aware_prompt_is_a_new_contract_with_action_policy():
    prompt = render_state_aware_webshop_prompt(
        task_description="buy grey gym shorts",
        current_observation="Page 1",
        available_actions={"has_search_bar": False, "clickables": ["item - B012", "next >"]},
    )
    assert STATE_AWARE_PROMPT_CONTRACT == "seed3090.webshop-state-aware-action-prompt.v3"
    assert "prefer a plausible product link" in prompt
    assert "never more than twice consecutively" in prompt
    assert "exactly two tagged lines" in prompt
