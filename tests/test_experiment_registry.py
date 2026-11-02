from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.openevo_webshop import experiment_registry as er


def _current() -> dict:
    registry, _ = er.load_current_registry(er.ROOT)
    return registry


def test_current_registry_and_router_validate() -> None:
    registry, router = er.load_current_registry(er.ROOT)
    assert er.validate_registry(registry, er.ROOT) == []
    path = er.ROOT / router["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == router["sha256"]
    assert set(router) == {"path", "sha256"}


def test_alias_resolves_one_to_one() -> None:
    registry = _current()
    canonical = "202609082022-q17-directapply-no-gdr"
    assert er.resolve_experiment(registry, canonical)["experiment_id"] == canonical
    assert er.resolve_experiment(registry, "qwen3-1p7b-directapply-no-gdr")["experiment_id"] == canonical
    assert er.resolve_experiment(registry, "Q17-DIRECT-APPLY-ONLY-W2-202609082022")["experiment_id"] == canonical


def test_alias_collision_fails() -> None:
    registry = _current()
    duplicate = copy.deepcopy(registry["entries"][0])
    duplicate["experiment_id"] = "202609082023-q17-other"
    duplicate["passport_path"] = registry["entries"][0]["passport_path"]
    duplicate["aliases"] = ["qwen3-1p7b-directapply-no-gdr"]
    registry["entries"].append(duplicate)
    errors = er.validate_registry(registry, er.ROOT)
    assert any("alias collision" in item for item in errors)


def test_passport_digest_mismatch_fails() -> None:
    registry = _current()
    registry["entries"][0]["passport_sha256"] = "0" * 64
    errors = er.validate_registry(registry, er.ROOT)
    assert any("Passport digest mismatch" in item for item in errors)


def test_passport_path_escape_fails() -> None:
    registry = _current()
    registry["entries"][0]["passport_path"] = "../escape/passport.json"
    errors = er.validate_registry(registry, er.ROOT)
    assert any("string identity format mismatch" in item or "traversal" in item or "escapes" in item for item in errors)


def test_unknown_alias_fails_closed() -> None:
    with pytest.raises(er.RegistryError, match="unknown experiment"):
        er.resolve_experiment(_current(), "not-a-real-experiment")
