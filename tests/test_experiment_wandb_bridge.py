from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.openevo_webshop import experiment_wandb_bridge as bridge
from scripts.openevo_webshop.experiment_passport import load_passport

ROOT = Path(__file__).resolve().parents[1]
Q17 = ROOT / "configs/experiment/passports/202609082022-q17-directapply-no-gdr/passport.json"


def _registered():
    return bridge.load_registered_passport(ROOT, "qwen3-1p7b-directapply-no-gdr")


def test_passport_projection_has_stable_search_identity() -> None:
    entry, passport = _registered()
    config = bridge.passport_wandb_config(entry, passport)
    assert config["experiment_id"] == "202609082022-q17-directapply-no-gdr"
    assert config["passport_sha256"] == passport["passport_payload_sha256"]
    assert config["protocol_fingerprint"] == passport["scientific_identity"]["protocol_fingerprint"]
    assert config["runtime_fingerprint"] == passport["initial_execution_identity"]["runtime_fingerprint"]
    assert config["tracking_role"] == "MIRROR_ONLY"
    assert config["tracking_scientific_authority"] is False


def test_metrics_require_explicit_scientific_axis() -> None:
    bridge.validate_metric_row({"system/round": 3, "eval/round_score": 0.5})
    with pytest.raises(bridge.WandbBridgeError, match="requires explicit x-axis"):
        bridge.validate_metric_row({"eval/round_score": 0.5})
    with pytest.raises(bridge.WandbBridgeError, match="unapproved metric"):
        bridge.validate_metric_row({"system/round": 3, "eval/redefined_score": 0.5})


def test_artifact_policy_allows_small_approved_text(tmp_path: Path) -> None:
    _, passport = _registered()
    artifact = tmp_path / "summary.json"
    artifact.write_text(json.dumps({"status": "PASS", "rows": 2}) + "\n", encoding="utf-8")
    result = bridge.approved_artifact(passport, "evaluation-summary", artifact)
    assert result["sha256"] == bridge._sha256(artifact)
    assert result["bytes"] == artifact.stat().st_size


def test_artifact_policy_rejects_role_secret_binary_and_large(tmp_path: Path) -> None:
    _, passport = _registered()
    plain = tmp_path / "plain.txt"; plain.write_text("ok\n")
    with pytest.raises(bridge.WandbBridgeError, match="role is not approved"):
        bridge.approved_artifact(passport, "raw-private-trace", plain)
    secret = tmp_path / "api-key.txt"; secret.write_text("redacted\n")
    with pytest.raises(bridge.WandbBridgeError, match="credential-like"):
        bridge.approved_artifact(passport, "evaluation-summary", secret)
    binary = tmp_path / "blob.bin"; binary.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(bridge.WandbBridgeError, match="binary artifact"):
        bridge.approved_artifact(passport, "evaluation-summary", binary)
    large = tmp_path / "large.txt"
    large.write_bytes(b"x" * (bridge.MAX_ARTIFACT_BYTES + 1))
    with pytest.raises(bridge.WandbBridgeError, match="too large"):
        bridge.approved_artifact(passport, "evaluation-summary", large)
    payload = tmp_path / "summary-payload.json"
    payload.write_text(json.dumps({"api_key": "redacted-but-key-field"}) + "\n")
    with pytest.raises(bridge.WandbBridgeError, match="credential-like field"):
        bridge.approved_artifact(passport, "evaluation-summary", payload)


class _FakeArtifact:
    def __init__(self, name: str, type: str, metadata=None):
        self.name, self.type, self.metadata = name, type, metadata or {}
        self.files = []

    def add_file(self, path: str, name: str | None = None):
        self.files.append((path, name))


class _FakeRun:
    def __init__(self, mode: str):
        self.id = "synthetic123"
        self.entity = "example-entity"
        self.path = "example-entity/openevo-webshop-experiments/synthetic123"
        self.url = "https://wandb.ai/example-entity/openevo-webshop-experiments/runs/synthetic123"
        self.name = "passport-smoke"
        self.group = "synthetic-group"
        self.job_type = "passport-mirror"
        self.dir = None
        self.offline = mode == "offline"
        self.settings = SimpleNamespace(sync_dir=None)
        self.summary = {}
        self.defined = []
        self.logged = []
        self.artifacts = []

    def define_metric(self, *args, **kwargs): self.defined.append((args, kwargs))
    def log(self, value, step=None): self.logged.append((value, step))
    def log_artifact(self, value): self.artifacts.append(value)
    def finish(self): return None


class _FakeWandb:
    Artifact = _FakeArtifact
    modes: list[str] = []
    fail_online = False

    @classmethod
    def init(cls, **kwargs):
        cls.modes.append(kwargs["mode"])
        if cls.fail_online and kwargs["mode"] == "online":
            raise RuntimeError("synthetic online failure")
        return _FakeRun(kwargs["mode"])


def test_online_receipt_has_provider_identity_and_does_not_mutate_science(tmp_path: Path, monkeypatch) -> None:
    _FakeWandb.modes = []; _FakeWandb.fail_online = False
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)
    scientific = tmp_path / "scientific.json"
    scientific.write_text('{"answer":42}\n', encoding="utf-8")
    before = bridge._sha256(scientific)
    receipt = bridge.mirror_passport(
        repo_root=ROOT,
        experiment_or_alias="qwen3-1p7b-directapply-no-gdr",
        mode="online",
        output_dir=tmp_path / "wandb",
        metric_rows=[{"system/round": 1, "eval/round_score": 0.25}],
        scientific_output=scientific,
        run_name="passport-smoke",
    )
    assert _FakeWandb.modes == ["online"]
    assert receipt["run_id"] == "synthetic123"
    assert receipt["run_url"].endswith("/runs/synthetic123")
    assert receipt["scientific_output_sha256"] == before == bridge._sha256(scientific)
    assert receipt["scientific_authority"] is False
    assert receipt["passport_mutated"] is False


def test_forced_online_failure_records_offline_fallback_and_spool_hash(tmp_path: Path, monkeypatch) -> None:
    _FakeWandb.modes = []; _FakeWandb.fail_online = True
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)
    scientific = tmp_path / "scientific.json"
    scientific.write_text('{"answer":42}\n', encoding="utf-8")
    receipt = bridge.mirror_passport(
        repo_root=ROOT,
        experiment_or_alias="qwen3-1p7b-directapply-no-gdr",
        mode="online",
        output_dir=tmp_path / "wandb",
        scientific_output=scientific,
        run_name="fallback-smoke",
    )
    assert _FakeWandb.modes == ["online", "offline"]
    assert receipt["effective_mode"] == "offline"
    assert "synthetic online failure" in receipt["fallback_reason"]
    assert receipt["fallback_spool_sha256"].startswith("sha256:")


def test_disabled_mode_preserves_scientific_output(tmp_path: Path) -> None:
    scientific = tmp_path / "scientific.json"
    scientific.write_text('{"answer":42}\n', encoding="utf-8")
    before = bridge._sha256(scientific)
    receipt = bridge.mirror_passport(
        repo_root=ROOT,
        experiment_or_alias="qwen3-1p7b-directapply-no-gdr",
        mode="disabled",
        output_dir=tmp_path / "wandb",
        scientific_output=scientific,
        run_name="disabled-smoke",
    )
    assert receipt["effective_mode"] == "disabled"
    assert receipt["run_id"] is None
    assert receipt["scientific_output_sha256"] == before == bridge._sha256(scientific)


def test_synthetic_fixture_kind_is_explicit_in_config() -> None:
    entry, passport = _registered()
    config = bridge.passport_wandb_config(entry, passport, fixture_kind="synthetic_smoke")
    assert config["tracking_fixture_kind"] == "synthetic_smoke"


def test_invalid_fixture_kind_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(bridge.WandbBridgeError, match="unsupported tracking fixture kind"):
        bridge.mirror_passport(
            repo_root=ROOT,
            experiment_or_alias="qwen3-1p7b-directapply-no-gdr",
            mode="disabled",
            output_dir=tmp_path / "wandb",
            fixture_kind="formal-result",
        )
