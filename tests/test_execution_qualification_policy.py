from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess

from scripts.openevo_webshop import qualification_dataset


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/execution/qualification-policy-v1.json"


def test_policy_is_nonformal_and_single_gpu():
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    assert policy["schema"] == "openevo.execution-qualification-policy.v1"
    assert policy["trainer_gate"]["formal_schedule_positions_consumed"] == 0
    assert policy["model_parser_webshop_gate"]["goal_idx"] >= 500
    assert policy["model_parser_webshop_gate"]["formal_panel_access_allowed"] is False
    assert policy["model_parser_webshop_gate"]["task_success_required"] is False
    assert policy["gpu_gate"]["gpu_count"] == 1
    assert policy["gpu_gate"]["kill_reset_preempt_allowed"] is False
    assert policy["authority_boundary"]["execution_authorized_by_pass"] is False


def test_synthetic_dataset_content_identity_is_path_independent(tmp_path):
    left = qualification_dataset.build(tmp_path / "left")
    right = qualification_dataset.build(tmp_path / "right")
    assert left["dataset_content_sha256"] == right["dataset_content_sha256"]
    assert left["records_sha256"] == right["records_sha256"]
    assert left["record_count"] == right["record_count"] == 2
    assert left["formal_schedule_positions_consumed"] == 0
    assert left["formal_panel_members"] == 0
    # Runtime file:// URIs are allowed to differ; they are excluded from content identity.
    assert left["manifest_sha256"] != right["manifest_sha256"]


def test_synthetic_manifest_matches_frozen_openevo_canonical_bytes(tmp_path):
    built = qualification_dataset.build(tmp_path / "dataset")
    manifest_path = Path(built["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    assert manifest_path.read_bytes() == expected
    assert not manifest_path.read_bytes().endswith(b"\n")


def test_synthetic_records_are_webshop_shaped_but_not_benchmark_tasks():
    rows = qualification_dataset.canonical_records()
    assert len(rows) == 2
    for row in rows:
        assert row["reward"] == 1.0
        assert row["metadata"]["qualification_only"] is True
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["formal_panel_member"] is False
        assert row["metadata"]["formal_schedule_position"] is None
        assert len(row["traces"]) == 1
        trace = row["traces"][0]
        assert trace["metadata"]["action_valid"] is True
        assert "<action>search[" in trace["response_messages"][0]["content"]


def _resolution():
    return {
        "source": {"openevo_revision": "796e6248a61112a61c880885844d3b0420c0b6ae"},
        "model": {"asset_id": "qwen2.5-3b-instruct"},
    }


def test_native_trainer_preserves_hf_id_and_uses_verified_local_cache(tmp_path):
    from scripts.openevo_webshop import trainer_qualification
    import pytest

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
    model = {"hf_id": "Qwen/Qwen2.5-3B-Instruct"}
    model_id = trainer_qualification._native_trainer_model_id(model)
    revision = "a" * 40
    hf_home = trainer_qualification._prepare_local_hf_cache(
        tmp_path / "hf-home",
        model_id=model_id,
        model_path=model_path,
        effective_revision=revision,
    )
    repo = hf_home / "hub" / "models--Qwen--Qwen2.5-3B-Instruct"
    snapshot = repo / "snapshots" / revision
    assert model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert snapshot.is_symlink()
    assert snapshot.resolve() == model_path.resolve()
    assert (repo / "refs" / revision).read_text(encoding="utf-8") == revision
    with pytest.raises(trainer_qualification.QualificationError, match="model ID"):
        trainer_qualification._native_trainer_model_id({"hf_id": str(model_path)})


def test_runtime_trainer_launcher_preserves_resolved_core_through_isolated_service(tmp_path):
    from scripts.openevo_webshop import trainer_qualification

    observed = tmp_path / "observed.txt"
    frozen_wrapper = tmp_path / "frozen-wrapper.sh"
    frozen_wrapper.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f"printf '%s\\n' \"$OPENEVO_TRAINER_CORE\" > {observed}\n"
        f"printf '%s\\n' \"$OPENEVO_TRAINER_PYTHON\" >> {observed}\n"
        f"printf '%s\\n' \"$1 $2\" >> {observed}\n",
        encoding="utf-8",
    )
    frozen_wrapper.chmod(0o700)

    launcher = trainer_qualification._write_runtime_trainer_launcher(
        tmp_path,
        trainer_wrapper=frozen_wrapper,
        runtime_python=Path("/verified/runtime-python"),
        openevo_root=Path("/verified/openevo-core"),
    )
    completed = subprocess.run(
        [str(launcher), "-I", "-m", "openevo.evolution.parametric.sd_lora_trainer"],
        env={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
        check=False,
    )

    assert completed.returncode == 0
    assert stat.S_IMODE(launcher.stat().st_mode) == 0o700
    assert observed.read_text(encoding="utf-8").splitlines() == [
        "/verified/openevo-core",
        "/verified/runtime-python",
        "-m openevo.evolution.parametric.sd_lora_trainer",
    ]


def test_trainer_payload_validation_is_exact():
    from scripts.openevo_webshop import trainer_qualification

    policy = trainer_qualification.load_policy()
    dataset = {"records_sha256": "a" * 64}
    payload = {
        "official_open_evo_stable": _resolution()["source"]["openevo_revision"],
        "dataset_records_sha256": "a" * 64,
        "training_record_count": 2,
        "optimizer_steps": 2,
        "training_loss": 0.5,
        "artifact_manifest": {
            "model_revision": "r1",
            "steps_completed": 2,
            "training_record_count": 2,
            "method": "parametric_memory_sd_lora",
            "adapter_id": "adapter-test",
        },
    }
    result = trainer_qualification.validate_smoke_payload(
        payload,
        resolution=_resolution(),
        dataset=dataset,
        effective_revision="r1",
        policy=policy,
    )
    assert result["optimizer_steps"] == 2
    broken = json.loads(json.dumps(payload))
    broken["optimizer_steps"] = 1
    import pytest

    with pytest.raises(trainer_qualification.QualificationError, match="optimizer_steps"):
        trainer_qualification.validate_smoke_payload(
            broken,
            resolution=_resolution(),
            dataset=dataset,
            effective_revision="r1",
            policy=policy,
        )


def test_adapter_reload_validation_requires_repeatable_active_lora():
    from scripts.openevo_webshop import trainer_qualification

    payload = {
        "status": "pass",
        "model_revision": "r1",
        "loaded_parameter_digest_match": True,
        "loaded_lora_tensor_count": 8,
        "loaded_lora_parameter_count": 123,
        "base_trainable_parameter_count": 0,
        "first_load_digest": "a" * 64,
        "second_load_digest": "a" * 64,
    }
    assert trainer_qualification.validate_reload_payload(payload, effective_revision="r1")["loaded_lora_tensor_count"] == 8
    payload["loaded_parameter_digest_match"] = False
    import pytest

    with pytest.raises(trainer_qualification.QualificationError, match="identical"):
        trainer_qualification.validate_reload_payload(payload, effective_revision="r1")


def test_scientific_preflight_pass_does_not_require_task_success():
    from scripts.openevo_webshop import scientific_preflight

    policy = scientific_preflight.load_policy()
    gate = policy["model_parser_webshop_gate"]
    episode = {
        "goal": "synthetic preview only",
        "task_score": 0.0,
        "won": False,
        "trace": [
            {
                "action": "search[red shirt]",
                "action_valid": True,
                "used_fallback": False,
                "prompt": "non-empty",
                "generation_seed": gate["generation_seed"],
                "environment_seed": gate["worker_seed"],
            }
        ],
    }
    result = scientific_preflight.validate_episode(
        episode,
        expected_goal_sha256="b" * 64,
        initial_available_actions={"has_search_bar": True, "clickables": ["Search"]},
        tracked_step_count=1,
        parser_events=[],
        policy=policy,
    )
    assert result["task_score"] == 0.0
    assert result["won"] is False
    assert result["task_success_required"] is False


def test_scientific_preflight_uses_registry_jdk_without_installing_dependencies(tmp_path, monkeypatch):
    from scripts.openevo_webshop import scientific_preflight

    jdk_home = tmp_path / "jdk"
    (jdk_home / "bin").mkdir(parents=True)
    (jdk_home / "bin" / "javac").write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", "/usr/bin")

    resolved = scientific_preflight._configure_registry_jdk(
        {"assets": {"jdk": {"path": str(jdk_home)}}}
    )

    assert resolved == jdk_home.resolve()
    assert os.environ["JAVA_HOME"] == str(jdk_home.resolve())
    assert os.environ["PATH"].split(os.pathsep)[0] == str(jdk_home / "bin")


def test_scientific_preflight_live_gate_precedes_webshop_preview_reset():
    from scripts.openevo_webshop import scientific_preflight

    source = Path(scientific_preflight.__file__).read_text()
    assert source.index("prelaunch.append(_sample_live_idle") < source.index("preview_env.reset")


def test_scientific_preflight_rejects_fallback_or_non_admissible_action():
    from scripts.openevo_webshop import scientific_preflight
    import pytest

    policy = scientific_preflight.load_policy()
    gate = policy["model_parser_webshop_gate"]
    episode = {
        "goal": "x",
        "trace": [
            {
                "action": "click[not-visible]",
                "action_valid": False,
                "used_fallback": True,
                "prompt": "non-empty",
                "generation_seed": gate["generation_seed"],
                "environment_seed": gate["worker_seed"],
            }
        ],
    }
    with pytest.raises(scientific_preflight.PreflightError):
        scientific_preflight.validate_episode(
            episode,
            expected_goal_sha256="b" * 64,
            initial_available_actions={"has_search_bar": True, "clickables": ["Search"]},
            tracked_step_count=1,
            parser_events=[],
            policy=policy,
        )


def test_v3_parser_uplift_matches_frozen_h144_semantics():
    from scripts.openevo_webshop.webshop_comparison_contract import parse_action_command_contract

    valid_cases = [
        ("<think>back</think>\n<action>click[< prev]</action>", "click[< prev]"),
        ("<think>budget</think>\n<action>search[grey gym shorts price <30]</action>", "search[grey gym shorts price <30]"),
    ]
    for raw, expected in valid_cases:
        action, valid = parse_action_command_contract(raw, require_think=True)
        assert valid is True
        assert action == expected
    for raw in (
        "<think>nested</think>\n<action>click[click[item]]</action>",
        "<action>click[back to search]</action>",
        "<think>truncated</think>\n<action>click[back to search</action>",
    ):
        _, valid = parse_action_command_contract(raw, require_think=True)
        assert valid is False


def test_parser_safe_retry_uses_v3_semantic_contract_after_retry():
    from scripts.openevo_webshop.parametric_rnd import ParserSafeRetryBackend

    class Backend:
        model_name = "stub"
        def __init__(self):
            self.values = [
                "<think>nested</think>\n<action>click[click[item]]</action>",
                "<think>good</think>\n<action>search[red shirt]</action>",
            ]
        def generate(self, *args, **kwargs):
            return self.values.pop(0)

    safe = ParserSafeRetryBackend(Backend(), action_contract="seed3090.webshop-action-parser.v3-action-command")
    raw = safe.generate("prompt", generation_seed=11)
    assert raw.endswith("<action>search[red shirt]</action>")
    assert safe.events[-1]["event_type"] == "parser-retry"
    assert safe.events[-1]["retry_valid"] is True
